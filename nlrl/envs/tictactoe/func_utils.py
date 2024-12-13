# Define the utility function to check for wins or draws
def check_win(board, player):
    # Check rows, columns, and diagonals for a win
    for i in range(3):
        if all([cell == player for cell in board[i]]) or all(
            [board[j][i] == player for j in range(3)]
        ):
            return True
    if (
        board[0][0] == board[1][1] == board[2][2] == player
        or board[0][2] == board[1][1] == board[2][0] == player
    ):
        return True
    return False


def check_draw(board):
    return all(all(cell is not None for cell in row) for row in board)


def check_winner(board):
    for player in ["X", "O"]:
        for i in range(3):
            if all([cell == player for cell in board[i]]):
                position = [1 + 3 * i, 2 + 3 * i, 3 + 3 * i]
                return player, position
            elif all([board[j][i] == player for j in range(3)]):
                position = [1 + i, 4 + i, 7 + i]
                return player, position
        if board[0][0] == board[1][1] == board[2][2] == player:
            position = [1, 5, 9]
            return player, position
        elif board[0][2] == board[1][1] == board[2][0] == player:
            position = [3, 5, 7]
            return player, position
    if all(all(cell is not None for cell in row) for row in board):
        return "draw", None
    return None, None


def convert_input_to_blocks(input_board):
    blocks = [[[{"value": None} for _ in range(1)] for _ in range(3)] for _ in range(3)]
    symbols = {0: None, 1: "O", 2: "X"}
    for index, value in enumerate(input_board):
        row = index // 3
        col = index % 3
        blocks[row][col] = symbols[value]
    return blocks


def extract_board(prompt):
    import re

    prompt = prompt.split("the next player is ")[1]
    next_player = prompt[0]
    # pattern = r'\b(?:[a-zA-Z0-9]+\|){2}[a-zA-Z0-9]+\b'
    pattern = r"\b(?:\s*[a-zA-Z0-9]+\s*\|\s*){2}\s*[a-zA-Z0-9]+\s*\b"
    matches = re.findall(pattern, prompt)
    assert len(matches) == 3
    pattern = r"\d"
    matches = [m.replace("|", "") for m in matches]
    matches = [
        re.sub(pattern, "0", m).replace("O", "1").replace("X", "2") for m in matches
    ]
    matches = [int(char) for m in matches for char in m if char != " "]
    return matches, 1 if next_player == "O" else 2


def convert_board(board, next_player):
    from copy import deepcopy

    assert len(board) == 9
    new_board = deepcopy(board)
    for i in range(len(board)):
        if board[i] == 0:
            new_board[i] = None
        elif board[i] == 1:
            new_board[i] = "O"
        elif board[i] == 2:
            new_board[i] = "X"
    dim2_board = [[new_board[3 * j + i] for i in range(3)] for j in range(3)]
    return dim2_board, "O" if next_player == 1 else "X"


def state_to_board(state):
    board = ""
    for i in range(3):
        for j in range(3):
            if state[0][3 * i + j] == 0:
                board += str(3 * i + j + 1)
            elif state[0][3 * i + j] == 1:
                board += "O"
            else:
                board += "X"
            if j < 2:
                board += " | "
        if i < 2:
            board += "\n---------\n"
    return board


def load_replay_buffer(file_path):
    import jsonlines

    data = {}
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.update(obj)
    data = {int(k): v for k, v in data.items()}
    return data


def extracted_output(
    json_string, split="JSON format for the evaluation of the initial board:"
):
    return json_string.split(split)[1]


# transform the state into natural language description
def state_to_nl(state):
    """
    Transform the state into natural language description like:
    The first row first column is empty, the first row second column is O, the first row third column is empty ...
    """
    nl = ""
    for i in range(3):
        for j in range(3):
            if state[0][3 * i + j] == 0:
                nl += f"The {i+1} row {j+1} column is empty, "
            elif state[0][3 * i + j] == 1:
                nl += f"The {i+1} row {j+1} column is O, "
            else:
                nl += f"The {i+1} row {j+1} column is X, "
    return nl


def check_next_player(state):
    zero_sum = sum([1 for i in state if i == 0])
    if zero_sum % 2 == 1:
        return "O"
    else:
        return "X"
