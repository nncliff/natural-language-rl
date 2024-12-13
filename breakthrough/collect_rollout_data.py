import os
import numpy as np
from absl import app
import json
from functools import partial
import copy
from ml_collections import config_flags
from tqdm import tqdm

from nlrl.utils import rollout, read_jsonl, write_jsonl
from nlrl.envs import get_env
from nlrl.policy import get_policy

from multiprocessing import Pool

_CONFIG = config_flags.DEFINE_config_file("config")


def collect_multistep_given_boards(
    serialized_state, config, env_config, policy_config_list, rollout_config
):
    """Given boards, collect multi multi-step rollout data starting from the board."""
    env = get_env(env_config)

    assert not serialized_state == "Terminal State."
    state = env.deserialize_state(serialized_state)
    lookahead_data = {"current_state": state.observation_string(), "current_serializad_state": state.serialize(), "pv": []}
    assert not state.is_terminal()
    # Run multiple multi-step rollouts from the given board
    state_ori = state
    for idx in range(rollout_config.lookahead_num_rollouts):
        agents = [get_policy(policy) for policy in policy_config_list]
        state = state_ori.clone()
        multi_step_data = {
            "state": [state.observation_string()],
            "action": [],
            "reward": [],
            "turn": [],
            "serializad_state": [state.serialize()],
            "final_state": None # this is for storing final state value target
        }
        for _ in range(rollout_config.lookahead_step):
            current_player_id = state.current_player()

            agent = agents[current_player_id]
            action = agent(state)
            action_str = state.action_to_string(current_player_id, action)

            for i, agent in enumerate(agents):
                if i != current_player_id:
                    agent.inform_action(state, current_player_id, action)

            state.apply_action(action)

            multi_step_data["action"].append(action_str)
            multi_step_data["reward"].append(state.rewards())
            multi_step_data["turn"].append(current_player_id)

            if state.is_terminal():
                multi_step_data["state"].append("Terminal State.")
                multi_step_data["serializad_state"].append("Terminal State.")
                break
            else:
                multi_step_data["state"].append(state.observation_string())
                multi_step_data["serializad_state"].append(state.serialize())

        if state.is_terminal():
            multi_step_data["final_state"] = {
                "state": "Terminal State.",
                "eval": "This is the terminal state.",
                "turn": None,
            }
        else:
            multi_step_data["final_state"] = {
                "state": state.observation_string(),
                "eval": None,
                "turn": state.current_player(),
            }

        lookahead_data["pv"].append(multi_step_data)
    # deduplicate the pv
    lookahead_data["pv"] = list(
        {json.dumps(d, sort_keys=True): d for d in lookahead_data["pv"]}.values()
    )
    return lookahead_data


def collect_rollout_given_boards(
    serialized_state, config, env_config, policy_config_list, rollout_config
):
    """
    Collect rollout data from a pyspiel environment from given states.
    """

    assert env_config.env_name.startswith("spiel")
    assert not env_config.batch_sample
    env = get_env(env_config)
    agents = [get_policy(policy) for policy in policy_config_list]

    assert not serialized_state == "Terminal State."
    state = env.deserialize_state(serialized_state)
    assert not state.is_terminal()

    traj_data = {
        "state": [],
        "action": [],
        "reward": [],
        "turn": [],
        "serializad_state": [],
    }
    traj_data["state"].append(state.observation_string())
    traj_data["serializad_state"].append(state.serialize())

    while not state.is_terminal():
        current_player_id = state.current_player()
        # The state can be three different types: chance node,
        # simultaneous node, or decision node
        assert not state.is_chance_node() and not state.is_simultaneous_node()
        # Decision node: sample action for the single current player
        agent = agents[current_player_id]
        action = agent(state)
        action_str = state.action_to_string(current_player_id, action)

        for i, agent in enumerate(agents):
            if i != current_player_id:
                agent.inform_action(state, current_player_id, action)

        state.apply_action(action)
        if state.is_terminal():
            traj_data["state"].append("Terminal State.")
            traj_data["serializad_state"].append("Terminal State.")
        else:
            traj_data["state"].append(state.observation_string())
            traj_data["serializad_state"].append(state.serialize())
        traj_data["action"].append(action_str)
        traj_data["reward"].append(state.rewards())
        traj_data["turn"].append(current_player_id)
    return traj_data


def rollout_spiel_sequential_turn_based(
    rank, config, env_config, policy_config_list, rollout_config
):
    """
    Collect rollout data from a pyspiel environment from initial state.
    batch_sample_size = 1
    Suitable for non-batchable policy like MCTS.
    """
    # Make sure this is a pyspiel environment and batch_sample_size is 1
    assert env_config.env_name.startswith("spiel")
    assert not env_config.batch_sample

    env = get_env(env_config)
    agents = [get_policy(policy) for policy in policy_config_list]

    traj_data = {
        "state": [],
        "action": [],
        "reward": [],
        "turn": [],
        "serializad_state": [],
    }
    state = env.new_initial_state()
    traj_data["state"].append(state.observation_string())
    traj_data["serializad_state"].append(state.serialize())

    while not state.is_terminal():
        current_player_id = state.current_player()
        # The state can be three different types: chance node,
        # simultaneous node, or decision node
        assert not state.is_chance_node() and not state.is_simultaneous_node()
        # Decision node: sample action for the single current player
        agent = agents[current_player_id]
        action = agent(state)
        action_str = state.action_to_string(current_player_id, action)

        for i, agent in enumerate(agents):
            if i != current_player_id:
                agent.inform_action(state, current_player_id, action)

        state.apply_action(action)
        if state.is_terminal():
            traj_data["state"].append("Terminal State.")
            traj_data["serializad_state"].append("Terminal State.")
        else:
            traj_data["state"].append(state.observation_string())
            traj_data["serializad_state"].append(state.serialize())
        traj_data["action"].append(action_str)
        traj_data["reward"].append(state.rewards())
        traj_data["turn"].append(current_player_id)
    return traj_data


def main(argv):
    config = _CONFIG.value
    # save config file
    os.makedirs(config.replay_buffer_dir, exist_ok=True)
    with open(os.path.join(config.replay_buffer_dir, "config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=4)
    print(config)
    rollout_config = config.rollout_config
    policy_config = config.policy_config
    opponent_policy_config = config.opponent_policy_config
    env_config = config.env_config
    if rollout_config.rollout_method == "scratch":
        assert rollout_config.num_rollouts % env_config.batch_sample_size == 0

    overwrite_replay_buffer = True
    if env_config.batch_sample:
        # Batch rollout
        # Suitable for LLM policy
        raise NotImplementedError("Batch rollout is not implemented yet.")
    else:
        # Sequential rollout
        # Accelarate the data collection process by using multiple workers
        if rollout_config.rollout_method == "traj_scratch":
            rollout = partial(
                rollout_spiel_sequential_turn_based,
                config=config,
                env_config=env_config,
                policy_config_list=[policy_config, opponent_policy_config],
                rollout_config=rollout_config,
            )
            with Pool(rollout_config.worker_num) as pool:
                replay_buffer = list(
                    tqdm(
                        pool.imap_unordered(
                            rollout, range(rollout_config.num_rollouts)
                        ),
                        total=rollout_config.num_rollouts,
                    )
                )
        elif rollout_config.rollout_method == "multi_step_given_boards":
            path_list = config.state_data_path.split(",")
            serialized_states = []
            for path in path_list:
                state_data_buffer = read_jsonl(path)
                if "turn" in state_data_buffer[0].keys():
                    # Trajectory replay buffer
                    assert (
                        len(state_data_buffer[0]["turn"])
                        == len(state_data_buffer[0]["serializad_state"]) - 1
                    )
                    for traj in state_data_buffer:
                        for s_state in traj["serializad_state"]:
                            # no need to treat final state as initial state to rollout
                            if s_state == 'Terminal State.':
                                continue
                            serialized_states.append(s_state)
                elif set(state_data_buffer[0].keys()) == set(["serializad_state"]):
                    # State replay buffer
                    serialized_states.extend([d["serializad_state"] for d in state_data_buffer])
                else:
                    raise NotImplementedError("process func not implemented.")
            
            # if we have old look_ahead data, 
            # read the file, avoid repeated rollout,
            # append new state's look_ahead data to it
            all_serialized_states = copy.deepcopy(serialized_states)
            if config.old_lookahead_dir != "None":
                old_look_ahead_data = read_jsonl(config.old_lookahead_dir)
                old_data = set(d["current_serializad_state"] for d in old_look_ahead_data)
                # drop finished states
                serialized_states = [s_state for s_state in serialized_states if s_state not in old_data]
                overwrite_replay_buffer = False

            # Deduplicate the initial states
            if rollout_config.init_state_dedup:
                serialized_states = list(set(serialized_states))
            look_ahead_rollout = partial(
                collect_multistep_given_boards,
                config=config,
                env_config=env_config,
                policy_config_list=[policy_config, opponent_policy_config],
                rollout_config=rollout_config,
            )
            with Pool(rollout_config.worker_num) as pool:
                replay_buffer = list(
                    tqdm(
                        pool.imap_unordered(look_ahead_rollout, serialized_states),
                        total=len(serialized_states),
                    )
                )

            # if we are going to update initial state replay buffer
            # with the new subsequent states in look-ahead data
            if config.all_initial_state_save_path:
                for r in replay_buffer:
                    for multi_step_data in r["pv"]:
                        all_serialized_states.extend(multi_step_data["serializad_state"])
                all_serialized_states = list(filter(lambda x: x != "Terminal State.", all_serialized_states))
                # Deduplicate
                all_serialized_states = list(set(all_serialized_states))
                all_initial_state = []
                for s_state in all_serialized_states:
                    all_initial_state.append({"serializad_state": s_state})
                # Overwrite old file
                write_jsonl(
                    all_initial_state, os.path.join(config.all_initial_state_save_path, "init_state_buffer.jsonl"), overwrite=True)
                
                print(f"Initial states updated successfully, length {len(all_initial_state)}")
                print(
                    "Data saved at:", os.path.join(config.all_initial_state_save_path, "init_state_buffer.jsonl")
                )
        elif rollout_config.rollout_method == "traj_given_boards":
            path_list = config.state_data_path.split(",")
            serialized_states = []
            for path in path_list:
                state_data_buffer = read_jsonl(path)
                if "turn" in state_data_buffer[0].keys():
                    # Trajectory replay buffer
                    assert (
                        len(state_data_buffer[0]["turn"])
                        == len(state_data_buffer[0]["serializad_state"]) - 1
                    )
                    for traj in state_data_buffer:
                        for s_state in traj["serializad_state"]:
                            # no need to treat final state as initial state to rollout
                            if s_state == 'Terminal State.':
                                continue
                            serialized_states.append(s_state)
                elif set(state_data_buffer[0].keys()) == set(["serializad_state"]):
                    # State replay buffer
                    serialized_states.extend([d["serializad_state"] for d in state_data_buffer])
                else:
                    raise NotImplementedError("process func not implemented.")
            # Deduplicate the initial states
            if rollout_config.init_state_dedup:
                serialized_states = list(set(serialized_states))
            if rollout_config.sub_sample_state > 0:
                np.random.seed(42)
                serialized_states = np.random.choice(
                    serialized_states,
                    size=rollout_config.sub_sample_state,
                    replace=False,
                ).tolist()
            # Make copy of the serialized states
            # Run multiple rollouts from each given board
            serialized_states = serialized_states * rollout_config.num_rollouts
            rollout = partial(
                collect_rollout_given_boards,
                config=config,
                env_config=env_config,
                policy_config_list=[policy_config, opponent_policy_config],
                rollout_config=rollout_config,
            )
            with Pool(rollout_config.worker_num) as pool:
                replay_buffer = list(
                    tqdm(
                        pool.imap_unordered(rollout, serialized_states),
                        total=len(serialized_states),
                    )
                )
        else:
            raise NotImplementedError("Rollout method not implemented.")

    write_jsonl(
        replay_buffer, os.path.join(config.replay_buffer_dir, "replay_buffer.jsonl"), overwrite=overwrite_replay_buffer
    )
    print("Data collected successfully.")
    print(
        "Data saved at:", os.path.join(config.replay_buffer_dir, "replay_buffer.jsonl")
    )


if __name__ == "__main__":
    app.run(main)
