"""Read data from TD pipline and process it for training."""

from nlrl.envs.breakthrough.prompt import (
    EVAL_prompt,
    get_state
)
from nlrl.utils import read_jsonl, write_jsonl
import argparse


def process_data(data, args):
    assert args.method == "mc_value_v"
    prompter = EVAL_prompt()
    states = [get_state(raw_data) for raw_data in data]
    query = list(map(prompter, states))

    assert len(query) == len(data)
    for query_point, data_point in zip(query, data):
        query_point["prompt"].append(data_point["prompt"][-1])
    return [q["prompt"] for q in query]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", choices=["mc_value_v"], default="mc_value_v"
    )
    parser.add_argument(
        "--data_path", type=str, default="pipeline/data/improve_target_0707.jsonl"
    )
    parser.add_argument(
        "--output_path", type=str, default="pipeline/data/improve_0707.jsonl"
    )

    args = parser.parse_args()
    data = read_jsonl(args.data_path)
    data = process_data(data, args)
    write_jsonl(data, args.output_path, overwrite=True)