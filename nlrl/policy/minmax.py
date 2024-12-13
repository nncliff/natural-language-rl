from gym_tictactoe.env import check_game_status, parse_current_mark
from copy import deepcopy


def minimax(board, depth, next_player):
    """
    Minimax algorithm to find the best move for the current player.
    board: 3x3 list of lists representing the game state
    depth: current depth in the game tree
    is_maximizing: boolean indicating if we are maximizing or minimizing
    next_player: 'X' or 'O' indicating the next player to move
    """

    # Check for a terminal state (win/draw) and return a value
    if next_player == "X":
        is_maximizing = True
    else:
        is_maximizing = False
    number_board = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == "X":
                number_board.append(2)
            elif board[i][j] == "O":
                number_board.append(1)
            elif board[i][j] is None:
                number_board.append(0)
    result = check_game_status(number_board)
    if result == 1:
        return depth - 10
    elif result == 2:
        return 10 - depth
    elif result == 0:
        return 0
    if is_maximizing:
        best_score = -float("inf")
        for i in range(3):
            for j in range(3):
                if board[i][j] is None:
                    board[i][j] = "X" if next_player == "X" else "O"
                    score = minimax(
                        board, depth + 1, "O" if next_player == "X" else "X"
                    )
                    board[i][j] = None
                    best_score = max(score, best_score)
        return best_score
    else:
        best_score = float("inf")
        for i in range(3):
            for j in range(3):
                if board[i][j] is None:
                    board[i][j] = "O" if next_player == "O" else "X"
                    score = minimax(
                        board, depth + 1, "X" if next_player == "O" else "O"
                    )
                    board[i][j] = None
                    best_score = min(score, best_score)
        return best_score


# Function to find the best move
def find_best_move(board, next_player=None):
    number_board = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == "X":
                number_board.append(2)
            elif board[i][j] == "O":
                number_board.append(1)
            elif board[i][j] is None:
                number_board.append(0)
    result = check_game_status(number_board)
    if next_player is None:
        next_player = parse_current_mark(number_board)
    # assert parse_current_mark(number_board) == next_player, f"Next player is {next_player}, but the current mark is {parse_current_mark(number_board)}"
    if result == 1:
        return (None, None), -10
    elif result == 2:
        return (None, None), 10
    elif result == 0:
        return (None, None), 0
    best_score = -float("inf") if next_player == "X" else float("inf")
    best_move = (-1, -1)
    for i in range(3):
        for j in range(3):
            if board[i][j] is None:
                board[i][j] = next_player
                score = minimax(board, 0, "O" if next_player == "X" else "X")
                board[i][j] = None
                if (
                    next_player == "X"
                    and score > best_score
                    or next_player == "O"
                    and score < best_score
                ):
                    best_score = score
                    best_move = (i, j)
    assert not (
        all(all(cell is not None for cell in row) for row in board)
        and best_move == (-1, -1)
    ), f"Board is full: {board}"
    return best_move, best_score


def check_optimal_move(board, action, next_player):
    """
    Check if the optimal move is made by the agent
    """
    _, best_score = find_best_move(board)
    # action is a scalar, convert it to 2D
    action_pos = (action // 3, action % 3)
    new_board = deepcopy(board)
    new_board[action_pos[0]][action_pos[1]] = next_player
    score = minimax(new_board, 0, "X" if next_player == "O" else "O")
    # if score == best_score:
    # print(new_board)
    return score == best_score


def test_minimax():
    board = [["X", "O", "X"], ["O", "X", "O"], [None, None, None]]
    next_player = "X"
    assert minimax(board, 0, next_player) == 9, minimax(board, 0, next_player)
    board = [["X", "O", "X"], ["O", "X", "O"], ["X", "O", "X"]]
    next_player = "O"
    assert minimax(board, 0, next_player) == 10, minimax(board, 0, next_player)
    board = [["O", "X", "O"], ["X", "O", "X"], [None, None, None]]
    next_player = "O"
    assert minimax(board, 0, next_player) == -9, minimax(board, 0, next_player)
    board = [["O", "O", "X"], ["X", "X", "O"], ["O", "X", None]]
    next_player = "O"
    assert minimax(board, 0, next_player) == 0, minimax(board, 0, next_player)
    board = [["X", "O", "X"], ["O", "O", "X"], [None, "X", None]]
    next_player = "O"
    assert find_best_move(board, next_player) == ((2, 2), 0), find_best_move(
        board, next_player
    )
    assert minimax(board, 0, next_player) == 0, minimax(board, 0, next_player)
    board = [["X", "O", "X"], ["O", "O", "X"], [None, "X", None]]
    next_player = "X"
    assert find_best_move(board, next_player) == ((2, 2), 10), find_best_move(
        board, next_player
    )
    assert minimax(board, 0, next_player) == 9, minimax(board, 0, next_player)
    next_player = "O"
    assert find_best_move(board, next_player) == ((2, 2), 0), find_best_move(
        board, next_player
    )
    assert minimax(board, 0, next_player) == 0, minimax(board, 0, next_player)
    board = [["X", "O", "X"], ["O", "O", "X"], [None, "X", None]]
    next_player = "X"
    assert find_best_move(board, next_player) == ((2, 2), 10), find_best_move(
        board, next_player
    )
    assert minimax(board, 1, next_player) == 8, minimax(board, 1, next_player)


if __name__ == "__main__":
    test_minimax()
