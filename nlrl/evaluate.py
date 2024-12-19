from itertools import count
from nlrl.utils import read_jsonl, write_jsonl
from pathlib import Path
from argparse import ArgumentParser
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def reward(traj):
    if traj["turn"][0]:  # play as O
        return traj["reward"][-1]
    else:
        return -traj["reward"][-1]


def win_rate(traj):
    if traj["turn"][0]:  # play as O
        return traj["reward"][-1] == 1
    else:
        return traj["reward"][-1] == -1


def tie_rate(traj):
    return traj["reward"][-1] == 0


def lossrate(traj):
    if traj["turn"][0]:  # play as O
        return traj["reward"][-1] == -1
    else:
        return traj["reward"][-1] == 1


def eval_episodes(data_path, verbose=True):
    replay_buffer = read_jsonl(data_path)
    avg_return = sum(list(map(reward, replay_buffer))) / len(replay_buffer)
    avg_win = sum(list(map(win_rate, replay_buffer))) / len(replay_buffer)
    avg_tie = sum(list(map(tie_rate, replay_buffer))) / len(replay_buffer)
    avg_lose = sum(list(map(lossrate, replay_buffer))) / len(replay_buffer)
    if verbose:
        print("==" * 20)
        print("Read from {}".format(data_path))
        print("Avg. Reward: {:.4f}".format(avg_return))
        print("Avg. Win Rate: {:.2%}".format(avg_win))
        print("Avg. Tie Rate: {:.2%}".format(avg_tie))
        print("Avg. Lose Rate: {:.2%}".format(avg_lose))
        print("Num Episodes: {}".format(len(replay_buffer)))
        print("==" * 20)
    return {
        "avg_win": avg_win,
        "avg_lose": avg_lose,
        "avg_tie": avg_tie,
        "avg_return": avg_return,
        "num_episode": len(replay_buffer),
        "path": data_path,
    }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    assert (args.data_dir is not None) ^ (args.data_path is not None)
    if args.data_dir:
        data_dir = Path(args.data_dir)

        df_list = []
        for i in count(1):
            dp = data_dir / f"replay_buffer_{i}.jsonl"
            if dp.exists():
                data = eval_episodes(dp, verbose=args.verbose)
                data["iter"] = i
                data = {k: [v] for k, v in data.items()}
                df_list.append(pd.DataFrame(data))
            else:
                print("Path: {} not exists, break evaluation".format(dp))
                break

        df = pd.concat(df_list, ignore_index=True)
        col = df.pop("iter")
        df.insert(0, "iter", col)
        print(df)
        fig = plt.figure()
        for y_name in ["avg_win", "avg_return", "avg_tie", "avg_lose"]:
            plot = plt.plot(df["iter"], df[y_name], label=y_name)
        plt.legend()
        fig_save_path = f"results_{data_dir.as_posix().replace('/', '_')}.pdf"
        plt.savefig(fig_save_path)
        print("Save to {}".format(fig_save_path))
    elif args.data_path:
        data = eval_episodes(args.data_path, verbose=True)
