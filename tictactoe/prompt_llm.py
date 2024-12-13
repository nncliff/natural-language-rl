from gym_tictactoe.env import TicTacToeEnv
from tqdm import tqdm
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from nlrl.envs.tictactoe.prompt import (
    MC_prompt,
    MC_prompt_sa,
    POLICY_prompt,
    POLICY_IMPROVEMENT_prompt_sa,
    EVAL_prompt_sa,
)
from nlrl.utils import read_jsonl, write_jsonl
from nlrl.envs import get_env
from nlrl.config import EnvConfig, LLMSamplingParams
from nlrl.envs.tictactoe.func_utils import state_to_board
from nlrl.offline_infer import offline_ray_vllm_infer

import torch
import argparse
import gc
import random
from collections import Counter, defaultdict
from nlrl.llm_call import vllm_model

import ray


def batch_generate(data, llm, args):
    if isinstance(data[0], dict):
        data = [d["prompt"] for d in data]
    batch_size = args.batch_size
    prompt_dataset_batches = [
        data[i : i + batch_size] for i in range(0, len(data), batch_size)
    ]
    print("Running batch generation")
    all_outputs = []
    for batch in tqdm(prompt_dataset_batches):
        outputs = llm.generate(batch)
        all_outputs.extend(outputs)
    return all_outputs


def hash_s_a(state, action):
    """state: A list of [list[int], str] indicate the board and player,
    action: a integer starts from 0
    """
    str_state = TicTacToeEnv.state2str(state[0]) + f"Player: {state[1]}"
    str_act = f"act: {action}"
    return str_state + "\n" + str_act


def get_mc_prompt(args, replay_buffer):
    mc_prompt = (
        MC_prompt()
        if args.method == "mc_value_v"
        else MC_prompt_sa(add_example=not args.zero_shot_eval)
    )
    # TODO: add a hyperparamter to determine to start with the first turn or not
    # replay_buffer = [board['state'] for board in replay_buffer]
    trunc_traj = []
    assert len(replay_buffer[0]["state"]) == len(replay_buffer[0]["action"]) + 1
    state_trajs = defaultdict(list)
    cnt = 0
    for i_buf, traj in enumerate(replay_buffer):
        # The final one is the terminal state
        # so no action, no need to take MC
        for i_step in range(len(traj["state"]) - 1):
            # we only include experience if this is main-player's turn
            if traj["turn"][i_step]:
                # trunc_data = {
                #     "state": traj["state"][i_step:],
                #     "action": traj["action"][i_step:],
                #     "reward": traj["reward"][i_step:],
                # }
                # trunc_traj.append(trunc_data)
                s_a_key = hash_s_a(traj["state"][i_step], traj["action"][i_step])
                state_trajs[s_a_key].append((i_buf, i_step))
                cnt += 1

    print("# Unique keys: {}, # State-Action pairs {}".format(len(state_trajs), cnt))
    print(args.n_mc_trajs)

    def query_iter():
        for start_s_a, record_indices in state_trajs.items():
            trunc_trajs = []
            idx = 0
            # random shuffle record_indices
            random.shuffle(record_indices)
            while idx < len(record_indices):
                i_b, i_s = record_indices[idx]
                traj = replay_buffer[i_b]
                trunc_data = {
                    "state": traj["state"][i_s:],
                    "action": traj["action"][i_s:],
                    "reward": traj["reward"][i_s:],
                }
                trunc_trajs.append(trunc_data)
                assert (
                    hash_s_a(trunc_data["state"][0], trunc_data["action"][0])
                    == start_s_a
                )

                idx += 1
                if len(trunc_trajs) == args.n_mc_trajs:
                    # break when collecting enough data as a batch
                    prompt = mc_prompt(trunc_trajs)
                    yield {
                        "state": trunc_data["state"][0],
                        "action": trunc_data["action"][0],
                        "prompt": prompt,
                    }
                    trunc_trajs = []
            if len(trunc_trajs) > 0:
                # return last batch of data less than args.n_mc_trajs
                prompt = mc_prompt(trunc_trajs)
                yield {
                    "state": trunc_data["state"][0],
                    "action": trunc_data["action"][0],
                    "prompt": prompt,
                }

    # return query_iter
    # XXX(ziyu): design for large corpus by iterator, now just make it into a list
    query = list(query_iter())
    return query

    # query = list(map(mc_prompt, trunc_traj))
    # query = [
    #     {
    #         "state": trunc_traj[idx]["state"][0],
    #         "action": trunc_traj[idx]["action"][0],
    #         "prompt": q,
    #     }
    #     for idx, q in enumerate(query)
    # ]
    # return query


def get_policy_prompt(args, replay_buffer):
    env_config = EnvConfig(env_name=args.env_name)
    replay_buffer = [eval(board) for board in replay_buffer]
    policy_prompt = POLICY_prompt(env_config)
    query = list(map(policy_prompt, replay_buffer))
    # if state is termnial, query will be None, and remove it from query
    query = filter(lambda x: x is not None, query)
    return query


def build_action_sampler(args):
    if args.num_policy_sample is None or args.num_policy_sample <= 0:
        print("use all valid state-action pairs")

        def action_sampler(boards):
            next_states_all = []
            indices = [0]
            for board in boards:
                next_states = POLICY_IMPROVEMENT_prompt_sa.get_state_action_pair(board)
                next_states_all.extend(next_states)
                indices.append(len(next_states_all))
            return next_states_all, indices

    else:
        assert (
            args.max_use_action > 1
        ), "Policy improvement assumes num action > 1, but get {}".format(
            args.max_use_action
        )
        print("use policy llm sample-based state-action pairs")

        def action_sampler(boards):
            policy_llm_config = LLMSamplingParams(
                # temperature=args.temp,
                temperature=0.7,
                top_k=args.top_k,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                n=args.num_samples,
            )
            from nlrl.policy.llm_policy import Agent

            agent = Agent(
                args.policy_model_path, policy_llm_config, model_tp_size=1, 
                remote=True,
                epsilon_greedy=args.eps_random_action
            )
            all_board_inputs = []
            for board in boards:
                all_board_inputs.extend([board for _ in range(args.num_policy_sample)])
            all_next_actions = agent.get_batch_action(all_board_inputs)

            next_states_all = []
            indices = [0]

            for i, board in enumerate(boards):
                next_actions = all_next_actions[
                    i * args.num_policy_sample : (i + 1) * args.num_policy_sample
                ]

                counted_actions = Counter(next_actions)
                top_actions = sorted(
                    counted_actions.items(), key=lambda x: x[1], reverse=True
                )
                # XXX(ziyu): currently no need to drop INVALID_ACTION here
                cnt = 0
                for act, _ in top_actions:
                    next_states_all.append({"state": board, "action": act})
                    cnt += 1
                    if cnt >= args.max_use_action:
                        break
                # add indices for next step all-pass vLLM inference and recover
                indices.append(len(next_states_all))

            return next_states_all, indices

    return action_sampler


def get_policy_improvement_prompt(
    args, replay_buffer, action_sampler, ues_all_states=False
):
    boards = []
    # TODO: add a hyperparamter to determine whether to use all trajectory states
    if ues_all_states:
        for board in replay_buffer:
            boards.extend(board["state"])
    else:
        for traj in replay_buffer:
            # The final one is the terminal state
            for i in range(len(traj["state"]) - 1):
                if traj["turn"][i]:
                    boards.append(traj["state"][i])
    # boards = boards[:5]

    next_states_all, indices = action_sampler(boards)
    value_prompt = EVAL_prompt_sa()
    value_prompts = list(map(value_prompt, next_states_all))
    print("Number of state action pair:", len(value_prompts))
    print(boards[0], next_states_all[0], value_prompts[0])

    value_llm_config = LLMSamplingParams(
        temperature=args.temp,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.num_samples,
    )
    # value_llm = vllm_model(
    #     args.value_model_path,
    #     value_llm_config,
    #     # , tensor_parallel_size=2
    # )

    outputs = offline_ray_vllm_infer(
        args.value_model_path,
        tensor_parallel_size=1,  # This is small model
        messages=[o["prompt"] for o in value_prompts],
        sample_config=value_llm_config,
    )
    # outputs = batch_generate(value_prompts, value_llm, args)
    print("Output example")
    print(outputs[0])
    assert len(outputs) == len(value_prompts)

    prompter = POLICY_IMPROVEMENT_prompt_sa()
    idx = 0
    query = []
    for i, board in enumerate(boards):
        state_value = []
        actions = []
        for idx in range(indices[i], indices[i + 1]):
            assert outputs[idx][-1]["role"] == "assistant"
            state_value.append({"value": outputs[idx][-1]["content"]})
            actions.append(next_states_all[idx]["action"] + 1)
            # here +1 because the action is start from 0
        prompt = prompter(board, actions=actions, next_states=state_value)
        query.append(prompt)
    return query


# def post_process_data(data, args):
#     if args.method == 'mc_value':
#         prompter = EVAL_prompt()
#         states = [board['state'] for board in data]
#         query = list(map(prompter, states))

#     if args.method == 'improve':
#         prompter = POLICY_prompt()
#         states = [board['state'] for board in data]
#         query = list(map(prompter, states))

#     assert len(query) == len(data)
#     for query_point, data_point in zip(query, data):
#         query_point['prompt'].append(data_point['prompt'][-1])
#         query_point['text'] = format_messages(query_point['prompt'])
#     return query


def main(args):
    # Read replay buffer
    replay_buffer = read_jsonl(args.input_path)

    # Get query by method
    if args.method.startswith("mc_value"):
        query = get_mc_prompt(args, replay_buffer)
    elif args.method == "policy":
        query = get_policy_prompt(args, replay_buffer)
    elif args.method == "improve":
        action_sampler = build_action_sampler(args)
        query = get_policy_improvement_prompt(args, replay_buffer, action_sampler)
    else:
        raise ValueError(f"Invalid method: {args.method}")

    print("query_example:", query[0])
    # Load LLM configuration
    SamplingParams = LLMSamplingParams(
        temperature=args.temp,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.num_samples,
    )
    # print("Loading LLM model")
    # llm = vllm_model(args.model_path, SamplingParams)
    # print("LLM model loaded")

    # # Run batch generation and write results
    # responses = batch_generate(query, llm, args)

    responses = offline_ray_vllm_infer(
        model=args.model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        messages=(
            query if not isinstance(query[0], dict) else [d["prompt"] for d in query]
        ),
        sample_config=SamplingParams,
    )
    for idx, q in enumerate(query):
        q["prompt"] = responses[idx]
    # print(results[0])
    # results = post_process_data(results, args)

    random.shuffle(query)
    print("Writing results to", args.output_path)
    write_jsonl(query, args.output_path)
    # random.shuffle(results)
    # train_data = results[:int(len(results)*0.8)]
    # test_data = results[int(len(results)*0.8):]
    # write_jsonl(train_data, args.output_path.replace('.jsonl', '_train.jsonl'))
    # write_jsonl(test_data, args.output_path.replace('.jsonl', '_test.jsonl'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["mc_value_v", "mc_value_q", "policy", "improve"],
        default="improve",
    )
    parser.add_argument("--input_path", type=str, default="data/replay_buffer.jsonl")
    parser.add_argument(
        "--n_mc_trajs",
        type=int,
        default=1,
        help="Number of trajectories when MC_value estimate as a batch",
    )
    parser.add_argument(
        "--zero_shot_eval",
        action="store_true",
        help="Whether use zeroshot prompt during evaluation",
    )
    parser.add_argument(
        "--output_path", type=str, default="pipeline/data/test_improve_response.jsonl"
    )
    parser.add_argument(
        "--model_path", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct"
    )
    # For policy improvement
    parser.add_argument(
        "--value_model_path", type=str
    )
    parser.add_argument(
        "--policy_model_path", type=str
    )
    parser.add_argument("--num_policy_sample", type=int, default=None)
    parser.add_argument("--max_use_action", type=int, default=2, 
                        help="top k actions sampled by the policy used in policy improvement")
    parser.add_argument("--eps_random_action", type=float, default=None,
                        help="probability when propose all available actions during policy improvement")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=1)

    parser.add_argument("--env_name", type=str, default="TicTacToeEnv")
    args = parser.parse_args()
    main(args)
