from nlrl.envs.tictactoe.prompt import POLICY_prompt, EVAL_prompt, EVAL_prompt_sa
from nlrl.utils import read_jsonl, write_jsonl
import argparse


def process_data(data, args):
    if args.method == "mc_value_v":
        prompter = EVAL_prompt()
        states = [board["state"] for board in data]
        query = list(map(prompter, states))
    elif args.method == "mc_value_q":
        prompter = EVAL_prompt_sa()
        state_action_dicts = [
            {"state": board["state"], "action": board["action"]} for board in data
        ]
        query = list(map(prompter, state_action_dicts))
    elif args.method == "improve":
        prompter = POLICY_prompt()
        states = [board["state"] for board in data]
        print(len(states))
        query = list(map(prompter, states))

    assert len(query) == len(data)
    for query_point, data_point in zip(query, data):
        query_point["prompt"].append(data_point["prompt"][-1])
    return [q["prompt"] for q in query]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", choices=["mc_value_v", "mc_value_q", "improve"], default="improve"
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
    write_jsonl(data, args.output_path)
