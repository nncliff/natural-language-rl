# Generate batch data for OPENAI Batch API to evaluate 5x5 breakthrough test set

from absl import app
import os
import json
from ml_collections import config_flags
from nlrl.envs.breakthrough.prompt import EVAL_prompt

from nlrl.utils import read_jsonl, write_jsonl


_CONFIG = config_flags.DEFINE_config_file("config")


def get_eval_prompt(config, replay_buffer):
    eval_prompt = EVAL_prompt()
    states = []
    for traj in replay_buffer:
        states.append(traj["state_turn"])
    query = list(map(eval_prompt, states))
    return query


def get_batch_query():
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
    data_dict = []
    data_dict_with_raw = []
    for idx, q in enumerate(query):
        data = {
            "custom_id": f"request-{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": config.model_path,
                "messages": q["prompt"],
                "temperature": config.llm_sample_config.temperature,
                "top_p": config.llm_sample_config.top_p,
                "max_tokens": config.llm_sample_config.max_tokens,
            },
        }
        data_dict.append(data)
        data["raw_data"] = q["raw_data"]
        data_dict_with_raw.append(data)
    save_path = os.path.join(config.output_path, "gpt4o_batch_prompt.jsonl")
    write_jsonl(data_dict, save_path)
    print(f"Save batch prompt to {save_path}")
    save_path_raw = os.path.join(config.output_path, "gpt4o_batch_prompt_raw.jsonl")
    write_jsonl(data_dict, save_path_raw)
    return save_path


def get_openai_batch():
    file_path = get_batch_query()
    from openai import OpenAI

    client = OpenAI()
    batch_input_file = client.files.create(file=open(file_path, "rb"), purpose="batch")
    batch_input_file_id = batch_input_file.id
    print("The batch input file id is: ", batch_input_file_id)

    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "gpt4o-breakthrough-eval"},
    )
    print(batch)

def main(argv):
    response = get_batch_query()
    # get_openai_batch()

if __name__ == "__main__":
    app.run(main)

