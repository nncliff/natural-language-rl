from copy import deepcopy
import json
import numpy as np
import warnings

def load_replay_buffer(file_path):
    import jsonlines

    data = {}
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.update(obj)
    data = {int(k): v for k, v in data.items()}
    return data


def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def write_jsonl(data, file_path, overwrite=True):
    import os
    if os.path.exists(file_path) and not overwrite:
        warnings.warn(f"File {file_path} exists, you are appending data.", UserWarning)
        write_mode = "a"
    elif os.path.exists(file_path) and overwrite:
        warnings.warn(f"File {file_path} exists, you are overwriting.", UserWarning)
        write_mode = "w"
    else:
        write_mode = "w"
    root_directory = os.path.dirname(file_path)
    os.makedirs(root_directory, exist_ok=True)
    with open(file_path, write_mode) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def rollout(env, policy, state=None):
    # policy rollout given environment
    if state is None:
        state = env.reset()
        traj_data = {"state": [deepcopy(state)], "action": [], "reward": []}
    else:
        state, done = env.set_state(state)
        traj_data = {"state": [deepcopy(state)], "action": [], "reward": []}
        if done:
            print("The state is terminal state, abort.")
            return traj_data
    while True:
        action = policy(state)
        available_actions = [i for i in range(9) if state[0][i] == 0]
        try:
            assert (
                action in available_actions
            ), f"Invalid action: {action}, available actions: {available_actions}"
        except:
            action = available_actions[
                int(np.random.choice(len(available_actions), 1)[0])
            ]
            print(f"Error: random action selected {action} from {available_actions}")
        next_state, reward, done, info = env.step(action)
        traj_data["state"].append(next_state)
        assert not state[1] == next_state[1]
        traj_data["action"].append(action)
        traj_data["reward"].append(reward)
        if done:
            break
        state = next_state
    return traj_data


if __name__ == "__main__":
    # from nlrl.config import PolicyConfig, EnvConfig, DataConfig
    # from policy import get_policy
    # from envs import get_env
    # policyconfig = PolicyConfig(policy_name='TicTacToe_minmax')
    # envconfig = EnvConfig(env_name='TicTacToeEnv')
    # dataconfig = DataConfig(data_path='./sampling/best_td_agent.dat', batch_size=1, shuffle=False, drop_reminder=True, policy_config=policyconfig, env_config=envconfig)
    # print(dataconfig)
    # env = get_env(envconfig)
    # policy = get_policy(policyconfig)
    # print(env)
    # print(policy)
    # state = env.reset()
    # print(state)
    # action = policy(state)
    # print(action)
    # next_state, reward, done, info = env.step(action)
    # print(next_state, reward, done, info)
    # traj_data = rollout(env, policy)
    # print(traj_data)
    # traj_data = rollout(env, policy, list(state[0]), state[1])
    # print(traj_data)
    # print('Done')
    def policy(state):
        from nlrl.utils import read_jsonl

        policy_dict = read_jsonl("data/policy_dict.jsonl")[0]
        state = str(tuple(state))
        action = policy_dict[state]
        return action

    from gym_tictactoe.env import TicTacToeEnv

    env = TicTacToeEnv()
    state = env.reset()
    print(state, policy(state[0]))
