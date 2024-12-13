from absl import app
import os
import json
from ml_collections import config_flags
import numpy as np
from tqdm import tqdm

import torch
from nlrl.envs.breakthrough.prompt import (
    EVAL_prompt,
    TD_prompt,
)
from nlrl.utils import read_jsonl, write_jsonl
from nlrl.offline_infer import offline_ray_vllm_infer


_CONFIG = config_flags.DEFINE_config_file('config')


def get_eval_prompt(config, replay_buffer):
    eval_prompt = EVAL_prompt()
    states = []
    if config.method == 'eval':
        for traj in replay_buffer:
            assert traj["state"][-1] == "Terminal State."
            assert len(traj["state"]) == len(traj["turn"]) + 1
            states.extend(list(zip(traj["state"][:-1], traj["turn"])))
    elif config.method == 'eval_final_state':
        # for TD implementation's data structure
        for traj in replay_buffer:
            for pv in traj["pv"][:config.num_pv_use]:  # only use top num_pv_use to evaluate
                if pv["final_state"]["state"] == "Terminal State.":
                    continue
                states.append((pv["final_state"]["state"], pv["final_state"]["turn"]))
    else:
        raise NotImplementedError
    query = list(map(eval_prompt, states))
    return query


def get_td_prompt(config, replay_buffer):
    td_prompt = TD_prompt(config=config)
    query = list(map(td_prompt, replay_buffer))
    return query


def update_replay_buffer(config, query, replay_buffer):
    # Update the final state eval in original replay buffer
    idx = 0
    for traj in replay_buffer:
        for pv in traj["pv"][:config.num_pv_use]:  # only fill in top num_pv_use final_state eval
            if pv["final_state"]["state"] == "Terminal State.":
                continue
            # Check if the state matches
            assert query[idx]["prompt"][-1]["role"] == "assistant"
            pv["final_state"]["eval"] = query[idx]["prompt"][-1]["content"]
            idx += 1
    return replay_buffer


def main(argv):
    config = _CONFIG.value
    os.makedirs(config.output_path, exist_ok=True)
    with open(os.path.join(config.output_path, "config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=4)
    # Read replay buffer
    replay_buffer = read_jsonl(config.input_path)
    if config.sub_sample_num > 0:
        print(f"Using seed {config.seed}")
        np.random.seed(config.seed)
        assert config.sub_sample_num < len(replay_buffer)
        replay_buffer = np.random.choice(replay_buffer, config.sub_sample_num, replace=False)
    print(len(replay_buffer))
    # Get query by method
    if config.method.startswith('eval'):
        query = get_eval_prompt(config, replay_buffer)
    elif config.method == 'td':
        query = get_td_prompt(config, replay_buffer)
    else:
        raise ValueError(f"Invalid method: {config.method}")

    print("query_example:", query[0])
    # exit()
    # Load LLM configuration
    query = query
    SamplingParams = config.llm_sample_config
 
    responses = offline_ray_vllm_infer(
        model=config.model_path,
        tensor_parallel_size=config.tensor_parallel_size,
        messages=(
            query if not isinstance(query[0], dict) else [d["prompt"] for d in query]
        ),
        sample_config=SamplingParams,
    )
    for idx, q in enumerate(query):
        q["prompt"] = responses[idx]

    if config.method == 'eval_final_state':
        replay_buffer = update_replay_buffer(config, query, replay_buffer)
        print("Updating old replay buffer to", config.output_path)
        output_name = "replay_buffer_updated.jsonl"
        output_path = os.path.join(config.output_path, output_name)
        write_jsonl(replay_buffer, output_path, overwrite=True)
    print("Writing results to", config.output_path)
    write_jsonl(query, os.path.join(config.output_path, "prompt_result.jsonl"), overwrite=True)

if __name__ == '__main__':
  app.run(main)
