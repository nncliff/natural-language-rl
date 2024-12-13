from nlrl.policy.minmax import find_best_move
from nlrl.envs.tictactoe.func_utils import convert_input_to_blocks
from nlrl.utils import read_jsonl


def tictactoe_minmax(state, policy_config):
    board = convert_input_to_blocks(state[0])
    action = find_best_move(board, state[1])
    assert action[0][0] * 3 + action[0][1] in range(
        9
    ), f"Invalid action: {action}, board: {board}"
    return action[0][0] * 3 + action[0][1]


def llm_dict(state, policy_config):
    policy_dict = read_jsonl("data/policy_dict.jsonl")[0]
    state = str(tuple(state[0]))
    action = policy_dict[state]
    return action
