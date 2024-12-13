from nlrl.utils import read_jsonl, write_jsonl
from nlrl.envs.tictactoe.func_utils import convert_board
from tqdm import tqdm
from nlrl.policy.minmax import check_optimal_move


def extract_answer(response, available_actions):
    result = response.split("""\"best_move\": """)[-1][0]
    # pattern = r'(?<![\d.-])-?(?:0(?:\.\d+)?|1(?:\.0+)?)(?![\d.])'
    result = int(result) - 1
    # assert result in available_actions, f"Error: {result} not in {available_actions}"
    return result


incorrect_idx = []
data = read_jsonl("data/policy_response.jsonl")
state = read_jsonl("/root/autodl-tmp/sft/data/policy_eval.jsonl")
for item, state in zip(data, state):
    item["state"] = state["state"]
full_idx = 0
correct_idx = 0
for item in tqdm(data):
    state = item["state"]
    try:
        next_action = extract_answer(
            item["response"], [i for i in range(9) if item["state"][0][i] == 0]
        )
        # print(next_action)
        # print(state)
        board, next_player = convert_board(state[0], state[1])
        # print(board)
        if check_optimal_move(board, next_action, next_player):
            correct_idx += 1
        else:
            incorrect_idx.append(
                {
                    "state": state,
                    "response": item["response"],
                    "next_action": next_action,
                }
            )
        full_idx += 1
    except Exception as e:
        print(f"Error {e}")
        # print(item['response'])
        # break
print(f"Full: {full_idx}")
print(f"Correct: {correct_idx}")
print(f"Accuracy: {correct_idx / full_idx}")
