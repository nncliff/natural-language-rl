import sys
from pathlib import Path
import functools
from nlrl.envs import get_env
from nlrl.policy.tictactoe_policy import tictactoe_minmax, llm_dict
from nlrl.policy.llm_policy import Agent
from nlrl.policy.mcts import MCTS_AGENT
from nlrl.policy.llm_ppo_policy import PPOLLMAgent
from nlrl.config import EnvConfig

def get_policy(policy_config):
    # Open-Spiel based policy, return agent clasee
    if policy_config.policy_name == "MCTS":
        agent = MCTS_AGENT(policy_config)
        return agent
    # Non-openspiel policy, return step function
    elif policy_config.policy_name == "LLM":
        is_gpt4 = "gpt-4" in policy_config.model_path.lower()
        # remote = (not is_gpt4) and policy_config.remote
        return Agent(
            model_path=policy_config.model_path,
            sample_config=policy_config.llm_config,
            epsilon_greedy=policy_config.epsilon_greedy,
            # remote=remote,
        )
    elif policy_config.policy_name == "LLM_PPO":

        if "policy" in policy_config.model_path.lower():
            policy = PPOLLMAgent(
                model_path=policy_config.model_path,
                env_config=EnvConfig(env_name="TicTacToeEnv"),
                epsilon_greedy=policy_config.epsilon_greedy,
                temperature=policy_config.llm_config.temperature,
            )
            # policy = PPOLLMAgent.from_pretrained(
            #     pretrained_path=policy_config.model_path,
            #     env_config=EnvConfig(env_name="TicTacToeEnv"),
            #     epsilon_greedy=policy_config.epsilon_greedy,
            #     temperature=policy_config.llm_config.temperature,
            # )
        else:
            policy = PPOLLMAgent(
                model_path=policy_config.model_path,
                env_config=EnvConfig(env_name="TicTacToeEnv"),
                epsilon_greedy=policy_config.epsilon_greedy,
                temperature=policy_config.llm_config.temperature,
            )
            return policy
    else:
        return functools.partial(
            POLICY_DICT[policy_config.policy_name], policy_config=policy_config
        )


def random_policy(state, policy_config):
    import random

    env = get_env(policy_config.env_config)
    actions = []
    if isinstance(state, list):
        for s in state:
            env.set_state(s)
            action = random.choice(env.get_available_actions())
            actions.append(action)
    else:
        env.set_state(state)
        actions = random.choice(env.get_available_actions())
    return actions


def first_action(state, policy_config):
    import random

    env = get_env(policy_config.env_config)
    actions = []
    if isinstance(state, list):
        for s in state:
            env.set_state(s)
            action = env.get_available_actions()[0]
            actions.append(action)
    else:
        env.set_state(state)
        actions = env.get_available_actions()[0]
    return actions


POLICY_DICT = {
    "TicTacToe_minmax": tictactoe_minmax,
    "Random": random_policy,
    "first_action": first_action,
    "LLM_dict": llm_dict,
    "LLM_PPO": PPOLLMAgent,
}
