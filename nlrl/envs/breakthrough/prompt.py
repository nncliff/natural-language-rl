# Prompt library

GAME_RULE_PROMPT = """Here is the rule for the Breakthrough board game:
The game is played on an 5x5 board for 2 players (white and black), with each player starting with 10 pawns. white pawns are on the first two rows and black pawns are on the last two rows.
Black moves first. In each turn, players can move one of their pieces one square forward, diagonally forward if the target square is empty. Or it can capture an opponent's piece if that square is one step diagonally forward.
The game ends when one player successfully break through the opponent lines -- either move a piece to the opposite last row of the board or captures all of the opponent's pieces.

For board representation, we use b for black pieces, w for white pieces, and . for empty squares. (1-5) and (a-e) are used to show the rows and columns index respectively."""

EVAL_SYSTEM_PROMPT = f"""{GAME_RULE_PROMPT}

You are a highly skilled evaluator in this game. I will provide you with specific board information representing the current board. Your output should include the following concepts:
1. *Tactical Considerations*: Any immediate threats, potential tactics, or vulnerabilities in the position.
2. *Positional Evaluation*: Consideration of pawn structure, piece activity, control of key squares, and game safety.
3. *Suggested Moves*: One or two strong candidate moves for the side to move, along with a brief rationale for comparing different moves.
4. *Advantage*: Based on all previous rationale, determine if white or black takes advantage. Use <white> or <black> to represent the winning side.

Your response should be informative and concise."""

EVAL_USER_PROMPT = """*The board you need to evaluate:* 
{board}"""

TD_SYSTEM_PROMPT = f"""{GAME_RULE_PROMPT}

You are a highly skilled evaluator in this game, particularly adept at making accurate assessments through look-ahead analysis of the current board position.
I will provide you with current board representation, *along with several key variations starting from this position (and their corresponding natural language evaluations of the subsequent positions)*.

Your task is to aggregate and compare these look-ahead information, to summarize, derive non-trivial analysis about the *current board*. Your output should include the following concepts:
1. *Tactical Considerations*: Any immediate threats, potential tactics, or vulnerabilities in the position.
2. *Positional Evaluation*: Consideration of pawn structure, piece activity, control of key squares, and game safety.
3. *Suggested Moves*: One or two strong candidate moves for the side to move, along with a brief rationale for comparing different moves.
4. *Advantage*: Based on all previous rationale, determine if white or black takes advantage. Use <white> or <black> to represent the winning side.

Your response should be informative and concise."""

TD_USER_PROMPT = """*The board you need to evaluate:* 

{board}

Here are the look-ahead variations from the current board position:
*Key Variations and Subsequent Evaluation:*:

{variations}

Please provide your analysis and understanding of the current board position based on the provided look-ahead information.
Your response should be informative and concise."""

VARIATION_PROMPT = """*Variation {i}:* 
Description of variation's move sequence:
{move_desc}

Subsequent position evaluation:
{subsequent_eval}"""

SUBSEQUENT_PROMPT = """The subsequent board is: 

{sub_board}

The evaluation of this subsequent board is: 

{sub_eval}"""

def action_desc(actions, turns):
    desc = f"The action sequence is: {','.join(actions)}.\n"
    desc_list = []
    for idx, (turn, action) in enumerate(zip(turns, actions)):
        color = "Black" if turn == 0 else "White"
        opponent_color = "White" if turn == 0 else "Black"
        if action[-1] == "*":
            per_action_desc = f"Move {idx + 1}:{color} moves piece from {action[:2]} (Column {action[0]}, Row {action[1]}) to {action[-3:-1]} (Column {action[2]}, Row {action[3]}), capturing {opponent_color} piece."
        else:
            per_action_desc = f"Move {idx + 1}:{color} moves piece from {action[:2]} (Column {action[0]}, Row {action[1]}) to {action[-2:]} (Column {action[2]}, Row {action[3]})."
        desc_list.append(per_action_desc)
    desc += "\n".join(desc_list)
    return desc

def get_piece_position(board):
    assert len(board) == 42 # For 5x5 board
    for index in range(1,6):
        board = board.replace(str(index), "")
    board = board.split('\n ')[0]
    row_name = list(reversed([str(i) for i in range(1,6)]))
    col_name = ["a", "b", "c", "d", "e"]
    white_position_list = []
    black_position_list = []
    for idx, row in enumerate(board.split("\n")):
        for jdx, piece in enumerate(row):
            if piece == "w":
                white_position_list.append(f"{col_name[jdx]}{row_name[idx]}")
            elif piece == "b":
                black_position_list.append(f"{col_name[jdx]}{row_name[idx]}")
    return white_position_list, black_position_list

def board_desc(board, turn):
    if board == "Terminal State.":
        return board
    color = "Black" if turn == 0 else "White"
    desc = f"{board}\n It is {color}'s turn.\n"
    white_pieces, black_pieces = get_piece_position(board)
    desc += "White pieces are at: " + ", ".join(white_pieces) + ".\n"
    desc += "Black pieces are at: " + ", ".join(black_pieces) + ".\n"
    return desc


def get_state(raw_data):
    return (
        raw_data["raw_data"]["current_state"],
        raw_data["raw_data"]["pv"][0]["turn"][0],
    )


class EVAL_prompt:
    def format_input(self, board):
        message = [{"role": "system", "content": EVAL_SYSTEM_PROMPT}]
        user_prompt = self.get_user_prompt(board)
        message.append({"role": "user", "content": user_prompt})
        return message

    def get_user_prompt(self, board):
        user_prompt = EVAL_USER_PROMPT.format(board=board_desc(board[0], board[1]))
        return user_prompt

    def __call__(self, board):
        data = {"raw_data": board, "prompt": self.format_input(board)}
        return data


class TD_prompt:
    def __init__(self, config):
        self.config = config

    def format_input(self, traj_data):
        message = [{"role": "system", "content": TD_SYSTEM_PROMPT}]
        user_prompt = self.user_prompt(traj_data)
        message.append({"role": "user", "content": user_prompt})
        return message

    def user_prompt(self, traj_data):
        current_state = traj_data["current_state"]
        current_turn = traj_data["pv"][0]["turn"][0]
        current_board_desc = board_desc(current_state, current_turn)
        pv = traj_data["pv"]
        variations = []
        for idx, variation in enumerate(pv[:self.config.num_pv_use]):
            # TODO: add logic for terminal state
            actions = variation["action"]
            rewards = variation["reward"]
            final_state = variation["final_state"]["state"]
            turns = variation["turn"]
            if final_state == "Terminal State.":
                if rewards[-1] == [1.0, -1.0]:
                    win = "Black"
                elif rewards[-1] == [-1.0, 1.0]:
                    win = "White"
                else:
                    raise NotImplementedError
                final_state_eval = f"This is the terminal state. {win} wins by reaching the opposite side of the board."
                final_state_turn = None
            else:
                final_state_eval = variation["final_state"]["eval"]
                final_state_turn = variation["final_state"]["turn"]
            # Make sure we get valid final state evaluation
            assert final_state_eval is not None
            subsequent_prompt = SUBSEQUENT_PROMPT.format(
                sub_board=board_desc(final_state, final_state_turn),
                sub_eval=final_state_eval,
            )
            variation_prompt = VARIATION_PROMPT.format(
                i=idx + 1,
                move_desc=action_desc(actions, turns),
                subsequent_eval=subsequent_prompt,
            )
            variations.append(variation_prompt)
        variations = "\n\n".join(variations)
        user_prompt = TD_USER_PROMPT.format(
            board=current_board_desc, variations=variations
        )
        return user_prompt

    def __call__(self, traj_data):
        data = {"raw_data": traj_data, "prompt": self.format_input(traj_data)}
        return data


if __name__ == "__main__":
    td_prompt = TD_prompt()
    td_test_data = {
        "current_state": "6bbbbbb\n5bbbbbb\n4......\n3......\n2wwwwww\n1wwwwww\n abcdef\n",
        "pv": [
            {
                "action": ["d5d4", "a2b3"],
                "reward": [[0.0, 0.0], [0.0, 0.0]],
                "final_state": {
                    "state": "6bbbbbb\n5bbbbbb\n4......\n3......\n2wwwwww\n1wwwwww\n abcdef\n",
                    "eval": "test test test",
                    "turn": 0,
                },
                "turn": [0, 1],
            },
            {
                "action": ["d5d4", "a2b3"],
                "reward": [[0.0, 0.0], [1.0, -1.0]],
                "final_state": {
                    "state": "Terminal State.",
                    "eval": "This is the terminal state.",
                    "turn": None,
                },
                "turn": [0, 1],
            },
        ],
    }
    print(td_prompt(td_test_data))
