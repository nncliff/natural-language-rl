from nlrl.utils import read_jsonl, write_jsonl
import argparse
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=int, default=1)
    parser.add_argument(
        "--data_path", type=str, default="pipeline/data/improve_target_0707.jsonl"
    )
    parser.add_argument(
        "--output_path", type=str, default="pipeline/data/improve_0707.jsonl"
    )

    args = parser.parse_args()
    index = int(args.data_path.replace(".jsonl", "").split("_")[-1])
    prev_index = max(1, index + 1 - args.history)
    data_all = []
    for idx in range(prev_index, index + 1):
        path = args.data_path.replace(f"{index}.jsonl", f"{idx}.jsonl")
        print("read from", path)
        data = read_jsonl(path)
        data_all.extend(data)
    random.shuffle(data_all)
    write_jsonl(data_all, args.output_path)
