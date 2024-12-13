from absl import app
import os
import json
from ml_collections import config_flags
from nlrl.utils import read_jsonl, write_jsonl

_CONFIG = config_flags.DEFINE_config_file("config")

def main(argv):
    config = _CONFIG.value
    os.makedirs(config.output_path, exist_ok=True)
    with open(os.path.join(config.output_path, "config_winrate.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=4)
    # Read replay buffer
    replay_buffer = read_jsonl(config.input_path)

    score_dict = {}
    for traj in replay_buffer:
        state_turn = (traj["state"][0], traj["turn"][0])
        final_reward = traj["reward"][-1][0]
        if state_turn in score_dict:
            score_dict[state_turn].append(final_reward)
        else:
            score_dict[state_turn] = [final_reward]

    data_result = [
        {"state_turn": state_turn, "win": "black" if sum(rewards) >= 0 else "white", "win_rate": sum(rewards) / len(rewards)}
        for state_turn, rewards in score_dict.items()
    ]
    print("Writing results to", config.output_path)
    write_jsonl(data_result, os.path.join(config.output_path, "gt_result.jsonl"), overwrite=True)


if __name__ == "__main__":
    app.run(main)
