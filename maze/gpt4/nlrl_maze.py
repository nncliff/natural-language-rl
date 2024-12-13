import copy
import os
import json
import jax
import pickle as pkl
from env.maze_utils import setup_maze_env, maze_solver, maze_solver_partrandom
from env.environment import TextPolicy, TextHistory, Text, text_env_eval
from env.utils import convert_path
from IPython import embed
import tiktoken
import random
import re
import numpy as np
from collections import defaultdict
from flax.traverse_util import flatten_dict, unflatten_dict
import argparse

from nlrl.envs.maze.prompt import SYSTEM_PROMPT, EVAL_USER_PROMPT_S_TD_G2_new, EVAL_USER_PROMPT_S_V, \
    POLICY_IMPROVEMENT_PROMPT_TD, EVAL_USER_PROMPT_SA_V2, POLICY_IMPROVEMENT_PROMPT_SA
from nlrl.llm_call import openai_model
from nlrl.config import LLMSamplingParams

# Define global openai client placeholder
client = None
TOKENIZER = None
INPUT_TOKEN_COUNT = 0
OUTPUT_TOKEN_COUNT = 0


class GPT4MazePolicy_TD(TextPolicy):

    def __init__(self, ahead_steps=1, env=None, num_variations=3, part_random=False, prob=0.5,
                 env_name="double_t_maze"):
        self.env = env
        self.ahead_steps = ahead_steps
        self.num_variations = num_variations
        self.V_prompt = EVAL_USER_PROMPT_S_V
        self.G2_prompt = EVAL_USER_PROMPT_S_TD_G2_new
        self.G1_prompt = POLICY_IMPROVEMENT_PROMPT_TD
        self.part_random = part_random
        self.prob = prob
        self.env_name = env_name

    def act(self, text_history: TextHistory) -> TextHistory:
        global INPUT_TOKEN_COUNT, OUTPUT_TOKEN_COUNT
        game_content = ""
        for item in text_history:
            game_content += f" {item.text} \n\n"
        game_content = game_content.strip()

        pos_acts = ["move up", "move down", "move left", "move right"]
        q_responses = {}
        for i in range(len(pos_acts)):
            g2_prompt = self.G2_prompt.replace('{game_content}', game_content)
            g2_prompt = g2_prompt.replace('{num_variations}', str(self.num_variations))
            if self.part_random:
                optimal_policy = maze_solver_partrandom(1 - self.env.maze,
                                                        list(map(tuple, self.env.valid_goals.tolist())), prob=self.prob)
            else:
                optimal_policy = maze_solver(1 - self.env.maze, list(map(tuple, self.env.valid_goals.tolist())))
            temp_env = setup_maze_env(maze_name=self.env_name, describe_function="describe_observation_give_position",
                                      reward_function="standard_reward", last_k=20)
            env_options = {}
            env_options['init_position'] = [
                int(''.join(re.findall(r'\d+', text_history[-1].text.split('.')[1].split(',')[0]))),
                int(''.join(re.findall(r'\d+', text_history[-1].text.split('.')[1].split(',')[1])))]
            if self.num_variations > len(pos_acts):
                temp_pos_acts = random.choices(pos_acts, k=self.num_variations)
            else:
                temp_pos_acts = random.sample(pos_acts, self.num_variations)
            Variations = []
            Game_contents = []

            for j in range(self.num_variations):
                pre_history = temp_env.reset(options=env_options)
                temp_env.goal = copy.deepcopy(self.env.goal)
                temp_act = pre_history + (Text(pos_acts[i].strip() + "\n", True),)
                temp_init_history, reward, done = temp_env.step(temp_act)
                temp_step = 0
                if self.ahead_steps > 0 and not done:
                    temp_act = temp_init_history + (Text(temp_pos_acts[j].strip() + "\n", True),)
                    temp_history, reward, done = temp_env.step(temp_act)
                    temp_step += 1
                    while temp_step < self.ahead_steps and not done:
                        temp_act = temp_history + (
                            Text(optimal_policy[tuple(temp_env.position)].lower().strip() + "\n", True),)
                        temp_history, reward, done = temp_env.step(temp_act)
                        temp_step += 1
                    temp_game_content = ""
                    for temp_item in temp_history:
                        temp_game_content += f" {temp_item.text} \n\n"
                    temp_game_content_final = f" {temp_history[-1].text}".strip()
                    V_prompt = self.V_prompt.replace('{game_content}', temp_game_content)
                    messages=[
                                    {
                                        "role": "system",
                                        "content": SYSTEM_PROMPT,
                                    },
                                    {
                                        "role": "user",
                                        "content": V_prompt,
                                    },
                            ]
                    response_text = client.generate(messages, return_text=True)
                    try:
                        response_json = json.loads(response_text)
                    except:
                        response_json = {"thoughts": "", "final_evaluation": ""}
                    Variations.append(response_json)
                    Game_contents.append(temp_game_content)
                else:
                    temp_game_content = ""
                    for temp_item in temp_init_history:
                        temp_game_content += f" {temp_item.text} \n\n"
                    temp_game_content_final = f" {temp_init_history[-1].text}".strip()
                    V_prompt = self.V_prompt.replace('{game_content}', temp_game_content)
                    messages=[
                                    {
                                        "role": "system",
                                        "content": SYSTEM_PROMPT,
                                    },
                                    {
                                        "role": "user",
                                        "content": V_prompt,
                                    },
                            ]
                    response_text = client.generate(messages, return_text=True)
                    try:
                        response_json = json.loads(response_text)
                    except:
                        response_json = {"thoughts": "", "final_evaluation": ""}
                    Variations.append(response_json)
                    Game_contents.append(temp_game_content)

            g2_prompt = g2_prompt.replace('{chosen_action}', pos_acts[i])
            for kk in range(self.num_variations):
                g2_prompt = g2_prompt.replace(f'{{game_content{kk + 1}}}', json.dumps(Game_contents[kk]))
                g2_prompt = g2_prompt.replace(f'{{Variation{kk + 1}}}', json.dumps(Variations[kk]))
            g2_prompt = g2_prompt.replace('{ahead_steps}', str(self.ahead_steps))
            print(g2_prompt)

            INPUT_TOKEN_COUNT += len(TOKENIZER.encode(g2_prompt))
            client.sample_config.max_tokens = 1500
            messages=[
                            {
                                "role": "system",
                                "content": SYSTEM_PROMPT,
                            },
                            {
                                "role": "user",
                                "content": g2_prompt,
                            },
                    ]
            response_text = client.generate(messages, return_text=True)
            client.sample_config.max_tokens = 1024
            OUTPUT_TOKEN_COUNT += len(TOKENIZER.encode(response_text))

            print(response_text)
            try:
                response_json = json.loads(response_text)
            except:
                response_json = {"thoughts": "", "final_evaluation": ""}
            q_responses[pos_acts[i]] = response_json

        a_prompt = self.G1_prompt.replace('{evaluations_up}', json.dumps(q_responses["move up"]))
        a_prompt = a_prompt.replace('{evaluations_down}', json.dumps(q_responses["move down"]))
        a_prompt = a_prompt.replace('{evaluations_left}', json.dumps(q_responses["move left"]))
        a_prompt = a_prompt.replace('{evaluations_right}', json.dumps(q_responses["move right"]))
        print(a_prompt)
        INPUT_TOKEN_COUNT += len(TOKENIZER.encode(a_prompt))
        message = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": a_prompt,
            },
        ]
        response_text = client.generate(message, return_text=True)
        OUTPUT_TOKEN_COUNT += len(TOKENIZER.encode(response_text))

        print(response_text)
        try:
            response_json = json.loads(response_text)
        except:
            response_json = {"action": ""}
        action_text = response_json.get('action', '')
        if isinstance(action_text, list):
            action_text = ' '.join(map(str, action_text))  # Join list elements into a single string
        elif not isinstance(action_text, str):
            action_text = str(action_text)  # Convert any non-string type to a string

        print(
            f"total cost: {compute_cost(INPUT_TOKEN_COUNT, OUTPUT_TOKEN_COUNT)}; total input tokens: {INPUT_TOKEN_COUNT}; total output tokens: {OUTPUT_TOKEN_COUNT}")

        return text_history + (Text(action_text.strip() + "\n", True),)

def compute_cost(input_token_count: int, output_token_count: int) -> float:
    return ((0.03 * input_token_count) / 1000) + ((0.06 * output_token_count) / 1000)


def main():
    global SYSTEM_PROMPT, EVAL_USER_PROMPT_S_TD_G2_new, EVAL_USER_PROMPT_S_V, \
    POLICY_IMPROVEMENT_PROMPT_TD, EVAL_USER_PROMPT_SA_V2, POLICY_IMPROVEMENT_PROMPT_SA
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="gpt-4o-mini", help='Model name')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=1.0, help='Sampling top_p')
    parser.add_argument('--max_tokens', type=int, default=1024, help='Sampling max tokens')
    parser.add_argument('--n_interactions', type=int, default=30, help='Number of interactions')
    parser.add_argument('--td', action='store_true', help='Use TD policy')
    parser.add_argument('--outputs_path', type=str, default='./outputs/gpt4_maze/fully_observed', help='Outputs path')
    parser.add_argument('--num_variations', type=int, default=6, help='Number of variations')
    parser.add_argument('--ahead_steps', type=int, default=3, help='Number of ahead steps')
    parser.add_argument('--n_rollouts', type=int, default=3, help='Number of rollouts')
    parser.add_argument('--part_random', action='store_true', help='Use partially random policy')
    parser.add_argument('--prob', type=float, default=0.0, help='Probability for partially random policy')
    parser.add_argument('--maze_name', type=str, default='double_t_maze', help='Maze name')#"maze2d_medium"
    parser.add_argument('--last_k', type=int, default=7, help='Last k steps of info stored in obs')
    args = parser.parse_args()
    N_INTERACTIONS = args.n_interactions
    TD = args.td
    OUTPUTS_PATH = args.outputs_path
    num_variations = args.num_variations
    ahead_steps = args.ahead_steps
    n_rollouts = args.n_rollouts
    part_random = args.part_random
    prob = args.prob
    maze_name = args.maze_name
    last_k = args.last_k
    sample_config = LLMSamplingParams(n=1, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    global client # Global openai client
    client = openai_model(model=args.model, sample_config=sample_config)
    global TOKENIZER
    TOKENIZER = tiktoken.encoding_for_model(args.model)

    for ii in range(num_variations):
        EVAL_USER_PROMPT_S_TD_G2_new += f"""
        *Variation {ii + 1}*, what may happen {{ahead_steps}} look-ahead steps after taking action {{chosen_action}}: {{game_content{ii + 1}}}
        ,
        Variation {ii + 1} evaluation: {{Variation{ii + 1}}}
        ,"""

    EVAL_USER_PROMPT_S_TD_G2_new = EVAL_USER_PROMPT_S_TD_G2_new.rstrip(",") + "\n"
    print(EVAL_USER_PROMPT_S_TD_G2_new)


    def text_history_to_str(text_history: TextHistory) -> str:
        return '\n'.join(map(lambda x: x.text, text_history))


    env = setup_maze_env(maze_name=maze_name, describe_function="describe_observation_give_position",
                         reward_function="standard_reward", last_k=last_k)



    policy = GPT4MazePolicy_TD(env=env, ahead_steps=ahead_steps, num_variations=num_variations,
                               part_random=part_random, prob=prob, env_name=maze_name)


    possible_positions = list(zip(*np.where(env.maze == 0)))

    for goal in env.valid_goals:
        possible_positions.remove(tuple(goal.tolist()))
    if len(possible_positions) > N_INTERACTIONS:
        selected_positions = random.sample(possible_positions, N_INTERACTIONS)
    else:
        selected_positions = possible_positions

    interactions = dict()
    results = dict()
    avg_dict = defaultdict(float)
    for position in selected_positions:
        position = tuple(position)
        interactions[str(position)], results[str(position)] = text_env_eval(
            env=env,
            policy=policy,
            n_rollouts=n_rollouts,
            verbose=True,
            env_options={"init_position": position},
            bsize=1,
        )
        for k, v in flatten_dict(results[str(position)]).items():
            avg_dict[k] += v
    for k, v in avg_dict.items():
        avg_dict[k] = v / len(selected_positions)
    results["avg_reward"] = unflatten_dict(dict(avg_dict))

    print(results)

    # Instead of create_path, just use os.makedirs
    os.makedirs(OUTPUTS_PATH, exist_ok=True)
    with open(os.path.join(OUTPUTS_PATH, 'interactions_qsa.pkl'), 'wb') as f:
        pkl.dump(interactions, f)
    with open(os.path.join(OUTPUTS_PATH,
                           f'td{TD}_{num_variations}v_{ahead_steps}s_{N_INTERACTIONS}*{n_rollouts}_{maze_name}_fullrandom.json'),
              'w') as f:
        json.dump(jax.tree_util.tree_map(lambda x: float(x), results), f)

if __name__ == "__main__":
    main()
