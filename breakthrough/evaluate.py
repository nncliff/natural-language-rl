from absl import app
import os
import json
import torch
from ml_collections import config_flags

from nlrl.envs.breakthrough.prompt import EVAL_prompt
from nlrl.utils import read_jsonl, write_jsonl
from nlrl.offline_infer import offline_ray_vllm_infer

_CONFIG = config_flags.DEFINE_config_file("config")


def get_eval_prompt(config, replay_buffer):
    eval_prompt = EVAL_prompt()
    states = []
    for traj in replay_buffer:
        states.append(traj["state_turn"])
    query = list(map(eval_prompt, states))
    return query


def main(argv):
    config = _CONFIG.value
    os.makedirs(config.output_path, exist_ok=True)
    with open(os.path.join(config.output_path, "config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=4)
    # Read replay buffer
    replay_buffer = read_jsonl(config.input_path)

    # Get query by method
    query = get_eval_prompt(config, replay_buffer)

    print("query_example:", query[0])

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

    # Calculate the evaluation accuracy
    replay_buffer_with_keys = {
        str(traj["state_turn"]): traj for traj in replay_buffer
    }

    full_eval_right = 0
    full_eval = 0
    for idx, q in enumerate(query):
        # remove states if the win rate is close to 0
        if abs(replay_buffer_with_keys[str(q["raw_data"])]["win_rate"]) < 0.2:
            continue
        answer = q["prompt"][-1]["content"]
        answer = answer.split("Advantage")[-1]
        if "<white>" in answer and "<black>" not in answer:
            predict = "white"
        elif "<black>" in answer and "<white>" not in answer:
            predict = "black"
        else:
            continue # If the answer cannot determine, skip it

        true_answer = replay_buffer_with_keys[str(q["raw_data"])]["win"]
        full_eval_right += 1 if predict == true_answer else 0
        full_eval += 1
    query.append(
        {
            "full_eval_right": full_eval_right,
            "full_eval": full_eval,
            "acc": full_eval_right / full_eval,
        }
    )
    print("Writing results to", config.output_path)
    write_jsonl(query, os.path.join(config.output_path, "eval_result.jsonl"), overwrite=True)


if __name__ == "__main__":
    app.run(main)
