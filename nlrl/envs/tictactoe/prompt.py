# Prompt library
from copy import deepcopy

import numpy as np
from nlrl.utils import read_jsonl, write_jsonl
from nlrl.envs.tictactoe.func_utils import (
    convert_input_to_blocks,
    state_to_board,
    state_to_nl,
    check_win,
    check_winner,
    check_next_player,
)
import random

# For GPT3.5 and 4 sampling evaluation as baselines
TIC_TAC_TOE_STATE_EVAL_PROMPT_GPT = """\
You are an expert agent at playing the game tic-tac-toe on a 3*3 board. Tic Tac Toe is a two-player game played on a grid.
Players take turns marking a space with their respective symbol. The goal is to get multiple of one’s own symbols in a row, either horizontally, vertically, or diagonally, before the opponent does. If all nine squares are filled and no player has three in a row, the game is a draw.

Your task is to evaluate a given board position.
The board consists of "O", "X" and grid number. The grid number indicates empty grid.

You should output your answer in the json format. Your answer consists of two elements:
- "thought": let's think step by step. Generate your detailed evaluation by analyzing the game from different perspectives.
- "final_evaluation": Judge which side takes advantage. 1 means you think 'O' takes advantage, -1 means you think 'X' takes advantage, and 0 means you think the game will be a draw.

Here is the board position and the next player is {next_player}:
Board:
{state}
"""

TIC_TAC_TOE_STATE_EVAL_PROMPT_GPT_SA = """\
You are an expert agent at playing the game tic-tac-toe on a 3*3 board. Tic Tac Toe is a two-player game played on a grid.
Players take turns marking a space with their respective symbol. The goal is to get multiple of one’s own symbols in a row, either horizontally, vertically, or diagonally, before the opponent does. If all nine squares are filled and no player has three in a row, the game is a draw.

Your task is to evaluate a given board position and next action.
The board consists of "O", "X" and grid number. The grid number indicates empty grid. The action is represented by grid number.

You should output your answer in the json format. Your answer consists of two elements:
- "thought": let's think step by step. Generate your detailed evaluation by analyzing the game from different perspectives.
- "final_evaluation": Judge which side takes advantage. 1 means you think 'O' takes advantage, -1 means you think 'X' takes advantage, and 0 means you think the game will be a draw.

Here is the board position and the next player is {next_player}:
Board:
{state}
Action:
The {next_player}'s move is {action}.
"""


TIC_TAC_TOE_STATE_POLICY_PROMPT_GPT = """\
You are an expert agent at playing the game tic-tac-toe on a 3*3 board. Tic Tac Toe is a two-player game played on a grid.
Players take turns marking a space with their respective symbol. The goal is to get multiple of one’s own symbols in a row, either horizontally, vertically, or diagonally, before the opponent does. If all nine squares are filled and no player has three in a row, the game is a draw.

Your task is to choose the best move given board position.
The board consists of "O", "X" and grid number. The grid number indicates empty grid.
You should output your answer in the json format. Your answer consists of two elements:
- "thought": let's think step by step. Generate your detailed thought process and plan for the next move.
- "best_move": the best move for the next player. The move should be in the format of a number from 1 to 9, indicating the position on the board. And the move should be in the available positions.

{example_prompt}

Here is the board position and the next player is {next_player}:
{state}. The available move positions are {available_positions}.
"""
TIC_TAC_TOE_STATE_POLICY_SYSTEM_PROMPT_GPT = """\
You are an expert agent at playing the game tic-tac-toe on a 3*3 board. Tic Tac Toe is a two-player game played on a grid.
Players take turns marking a space with their respective symbol. The goal is to get multiple of one’s own symbols in a row, either horizontally, vertically, or diagonally, before the opponent does. If all nine squares are filled and no player has three in a row, the game is a draw.

Your task is to choose the best move given board position.
The board consists of "O", "X" and grid number. The grid number indicates empty grid.
You should output your answer in the json format. Your answer consists of two elements:
- "thought": let's think step by step. Generate your detailed thought process and plan for the next move.
- "best_move": the best move for the next player. The move should be in the format of a number from 1 to 9, indicating the position on the board. And the move should be in the available positions."""

TIC_TAC_TOE_STATE_POLICY_USER_PROMPT = """\
Here is the board position and the next player is {next_player}:
{state}. The available move positions are {available_positions}.
"""

POLICY_IMPROVEMENT_PROMPT = """\
You are playing the game tic-tac-toe on a 3*3 board. Tic Tac Toe is a two-player game played on a grid.
Players take turns marking a space with their respective symbol. The goal is to get multiple of one’s own symbols in a row, either horizontally, vertically, or diagonally, before the opponent does. If all nine squares are filled and no player has three in a row, the game is a draw.
The board consists of "O", "X" and grid number. The grid number indicates empty grid.
Your task is to determine the best move for the next player based on the given board position and the next player.
The evaluations of boards after possible moves are given.
DO NOT judge the board based on your knowledge, only use the evaluations to determine the best move.
The evaluation for the next board is in the format of a json format, consisting of two elements:
- "thought": evaluation of the board position.
- "final_evaluation": Judge which side takes advantage. 1 means 'O' takes advantage, -1 means 'X' takes advantage, and 0 means the game will be a draw.
{example_prompt}
Here is the board position and the next player is {next_player}:
{state}. The possible moves are {available_positions}.
The following are the boards after each possible move:
{next_states}

Now, please give your evaluation and the best move for {next_player} based on the given board position {state}.
You should output your answer in the json format. Your answer consists of two elements:
- "thought": let's think step by step. Generate your detailed reflection by analyzing the next board positions and their evaluations.
- "best_move": the best move for the next player. The move should be in the format of a number from 1 to 9, indicating the position on the board. And the move should be in the available positions.
Don't output extra information except for the json format.
"""

POLICY_IMPROVEMENT_SYSTEM_PROMPT_SA = """\
You are playing the game tic-tac-toe on a 3*3 board. Tic Tac Toe is a two-player game played on a grid.
Players take turns marking a space with their respective symbol. The goal is to get multiple of one’s own symbols in a row, either horizontally, vertically, or diagonally, before the opponent does. If all nine squares are filled and no player has three in a row, the game is a draw.
The board consists of "O", "X" and grid number. The grid number indicates empty grid.
Your task is to determine the best move for the next player based on the given board position and the next player.
The evaluations of (board, action) pairs after possible moves are given.
DO NOT judge the board based on your knowledge, only use the evaluations to determine the best move.
The evaluation for the next board is in the format of a json format, consisting of two elements:
- "thought": evaluation of the the board and action pair.
- "final_evaluation": Judge which side takes advantage. 1 means 'O' takes advantage, -1 means 'X' takes advantage, and 0 means the game will be a draw.
"""

POLICY_IMPROVEMENT_USER_PROMPT_SA = """\
Here is the board position and the next player is {next_player}:
{state}. The possible moves are {available_positions}.
The following are the boards after each possible move:
{next_states}

Now, please give your evaluation and the best move for {next_player} based on the given board position {state}.
You should output your answer in the json format. Your answer consists of two elements:
- "thought": let's think step by step. Generate your detailed reflection by analyzing the next board positions and their evaluations.
- "best_move": the best move for the next player. The move should be in the format of a number from 1 to 9, indicating the position on the board. And the move should be in the available positions {available_positions}.
Don't output extra information except for the json format.
"""

TIC_TAC_TOE_POLICY_EXAMPLE_PROMPT = """
Here is the board position and the next player is O:
O | O | X
---------
4 | X | 6
---------
7 | 8 | 9
The available move positions are 4, 6, 7, 8, 9.

{"thought": "It appears that the initial board position O | O | X
---------
4 | X | 6
---------
7 | 8 | 9 was favorable for X, as X has occupied the positions 3, 5 and X can win by occupying the position 7. O has occupied the positions 1, 2. Therefore, the best move for O is to occupy the position 7 to block X and create a potential winning opportunity by occupying the positions 1, 4, 7.", "best_move": 7}
"""

TIC_TAC_TOE_POLICY_EXAMPLE_USER_PROMPT = """\
Here is the board position and the next player is O:
O | O | X
---------
4 | X | 6
---------
7 | 8 | 9
The available move positions are 4, 6, 7, 8, 9.
"""

TIC_TAC_TOE_POLICY_EXAMPLE_ASSISTANT_PROMPT = """\
{"thought": "It appears that the initial board position
O | O | X
---------
4 | X | 6
---------
7 | 8 | 9 was favorable for X, as X has occupied the positions 3, 5 and X can win by occupying the position 7. O has occupied the positions 1, 2. Therefore, the best move for O is to occupy the position 7 to block X and create a potential winning opportunity by occupying the positions 1, 4, 7.", "best_move": 7}
"""

MC_PROMPT = """You are a reinforcement learning agent of the game of Tic Tac Toe. \nThe goal is to get multiple of one's own symbols in a row, either horizontally, vertically, or diagonally, before the opponent does. If all nine squares are filled and no player has three in a row, the game is a draw. \nThe board consists of \"O\", \"X\" and grid number. The grid number indicates empty grid. \nYou are learning how to evaluate a board in the tic tac toe by playing the game and reflect the playing history. \nThe following is a rollout history depicting a game in progress with a final result. \n{trajectory}\nYou do not know the value of the first board as you are a learner. \nYou have to evaluate the board in hindsight based on the rollout sequence.\nYou have to review the board by reflecting on the sequence and the ultimate outcome. \nDo not just evaluate the state based on your knowledge because you are a learner and your knowledge is inaccurate. \nJust review the board, do not guess the potential move. \nJust review the board and do not give any advice unrelated to the board. Explicitly state your evaluation based on the rollout sequence. Point out the threatens and opportunities of each board based on the rollout sequence. """
MC_SYSTEM_PROMPT = """You are a player of the game of Tic Tac Toe. \nThe goal is to get multiple of one's own symbols in a row, either horizontally, vertically, or diagonally, before the opponent does. If all nine squares are filled and no player has three in a row, the game is a draw. \nThe board consists of \"O\", \"X\" and grid number. The grid number indicates empty grid. \nYou are learning how to evaluate a board in the tic tac toe by playing the game and reflect the playing history. \nThe following is a rollout history depicting a game in progress with a final result. \nYou should output your answer in the json format. Your answer consists of two elements:
- "thought": let's think step by step. Generate your detailed evaluation by analyzing the game from different perspectives. Your evaluation should contain the following elements: Win probability, Threat, Potential strategies.
- "final_evaluation": After all your thought, finally judge which side takes advantage. 1 means you think 'O' takes advantage, -1 means you think 'X' takes advantage, and 0 means you think the game will be a draw."""
MC_EXAMPLE_USER_PROMPT = """The board to evaluate is O's turn:
O | O | X
---------
4 | X | 6
---------
7 | 8 | 9

Below is the rollout sequence:
After O's move 4, the board position is:
O | O | X
---------
O | X | 6
---------
7 | 8 | 9
After X's move 7, the board position is:
O | O | X
---------
O | X | 6
---------
X | 8 | 9
The game is over. X wins. X wins by occupying the positions 3, 5, 7.
"""
MC_EXAMPLE_ASSISTENT_PROMPT = """
{"thought": {"Reflection": "It appears that the initial board position
O | O | X
---------
4 | X | 6
---------
7 | 8 | 9
was not favorable for O, as X was able to block on O's moves and ultimately win the game.", "Win probability": "The win probability for X is large, while the win probability for O is low.", "Threat": "X has played center 5 and corner 3. X can win by playing corner 7. O was able to occupy 1, 4, 7 and create a potential winning opportunity.", "Potential strategies": "Potential strategies for O include playing the corner 7 to block X as opposite corner and win by occupying 1, 4, 7. X could have occupied 3, 5, 7 to win the game. X has already occupied 3, 5, and there is 1 step to complete the plan."}
"final_evaluation": -0.9}
"""
MC_SYSTEM_PROMPT_SA = """You are a player of the game of Tic Tac Toe. \nThe game goal is to get multiple of one's own symbols in a row, either horizontally, vertically, or diagonally, before the opponent does. If all nine squares are filled and no player has three in a row, the game is a draw. \nThe board consists of \"O\", \"X\" and grid number. The grid number indicates empty grid. \nYou are learning how to evaluate a (board, action) pair in the tic tac toe by playing the game given the (board, action) pair and reflect the playing history. \nThe playing history depicts a game in progress with a final result. Your answer consists of two elements:
- "thought": let's think step by step. Generate your detailed evaluation over the (board, action) pair by merely reflecting the playing history after this pair from different perspectives. You should only rely on the playing history as context and don't evaluate game with your own judgement. Your evaluation should contain the following elements: Win probability, Threat, Potential strategies.
- "final_evaluation": After all of your thoughts, judge which side takes advantage. 1 means you think 'O' takes advantage, -1 means you think 'X' takes advantage, and 0 means you think the game will be a draw.
You should output your answer in the json format."""
MC_EXAMPLE_USER_PROMPT_SA = """The (board, action) to evaluate is O's turn:
Board:
O | O | X
---------
4 | X | 6
---------
7 | 8 | 9
Action:
The O's move is 4.

Below is the rollout sequence after this (board, action):
After O's move 4, the board position is:
O | O | X
---------
O | X | 6
---------
7 | 8 | 9
After X's move 7, the board position is:
O | O | X
---------
O | X | 6
---------
X | 8 | 9
The game is over. X wins. X wins by occupying the positions 3, 5, 7.
"""
MC_EXAMPLE_ASSISTENT_PROMPT_SA = """
{"thought": {"Reflection": "It appears that the initial board position
O | O | X
---------
4 | X | 6
---------
7 | 8 | 9
and O's move 4 were not favorable for O, as X was able to block on O's move at 7 and ultimately win the game.", "Win probability": "The win probability for X is large, while the win probability for O is low.", "Threat": "X has played at 5 and 3. X can win by move 7. O can occupy 1, 4, 7, and create a potential winning opportunity. X occupies 5, which is a key position to win the game.", "Potential strategies": "Potential strategies for O include playing at 7 to block X and create a potential win by occupying 1, 4, 7. X could have occupied 3, 5, 7 to win the game. X has already occupied 3, 5, and needs only 1 move to complete the win."}
"final_evaluation": -0.8}
"""
STATE_TO_VALUE_PROMPT_SA = "The board to evaluate is {player}'s turn:\nBoard:\n{state}\nAction:\nThe {player}'s move is {action}.\n"
NEW_TRAJ_BEGIN_PROMPT = (
    "\nBelow is the rollout sequence {idx} after this (board, action):"
)

CONCAT_PROMPT = (
    "\nAfter {player} taking action {action}, the board position is \n{board}.\n"
)
STATE_TO_VALUE_PROMPT = "The board to evaluate is {player}'s turn: \n{state}.\n\nBelow is the rollout sequence:"
STATE_TO_ACTION_PROMPT = (
    "The current board is {player}'s turn: \n{state}.\n\nBelow is the rollout sequence:"
)
TD_PROMPT = """You are a reinforcement learning agent of the game of Tic Tac Toe. \nThe goal is to get multiple of one's own symbols in a row, either horizontally, vertically, or diagonally, before the opponent does. If all nine squares are filled and no player has three in a row, the game is a draw. \nYou are learning how to evaluate a board in the tic tac toe by playing the game and reflecting the playing history. \nThe following is a board and a next board. \nThe board consists of \"O\", \"X\" and grid number. The grid number indicates empty grid.\n{trajectory}\n"""
EVALUATION_PROMPT = """You are a reinforcement learning agent of the game of Tic Tac Toe. \nThe goal is to get multiple of one's own symbols in a row, either horizontally, vertically, or diagonally, before the opponent does. If all nine squares are filled and no player has three in a row, the game is a draw. \nYou are learning how to evaluate a board in the tic tac toe by playing the game and reflect the playing history. \nThe following is a board to evaluate. \nThe board consists of \"O\", \"X\" and grid number. The grid number indicates empty grid.\n"""
EVALUATION_EXAMPLE_PROMPT = """### EXAMPLE:
Board to evaluate:
O | O | X
---------
4 | X | 6
---------
7 | 8 | 9
Evaluation: {"Evaluation": "It appears that the board position O | O | X
---------
4 | X | 6
---------
7 | 8 | 9 was not favorable for O, as X was able to capitalize on O's moves and ultimately win the game."
"Win probability": X: 0.7, O: 0.1, Draw: 0.2
"Threats": {"O": {"threats": 0.9, explanation: "X was able to occupy 3, 5, 7 and ultimately win the game.}, "X": {"threats": 0.1, explanation: "O was able to occupy 1, 4, 7 and create a potential winning opportunity."}
"Potential strategies": {"O": "\"O\" could have occupied 1, 4, 7 to create a potential winning opportunity. \"O\" has occupied 1. There are 2 step to complete the plan", "X": "\"X\" could have occupied 3, 5, 7 to win the game. \"X\" has already occupied 3, 5. There is 1 step to complete the plan."}
}
"""
EVALUATION_QUERY_PROMPT = "Board to evaluate:\n{board}\nEvaluation: "
TD_EXAMPLE_PROMPT = """
### EXAMPLE:
The initial board is:
O | O | X
---------
4 | X | 6
---------
7 | 8 | 9
Information for the next board:
After X taking action, the board position is
O | O | X
---------
O | X | 6
---------
7 | 8 | 9.
Discription for the next board:
It appears that the board position
O | O | X
---------
O | X | 6
---------
7 | 8 | 9
was not favorable for O, as X was able to capitalize on O's moves and ultimately win the game. The win probability for X is large, while the win probability for O is low. From the perspective of threat, X was able to occupy 3, 5, 7 and ultimately win the game. O was able to occupy 1, 4, 7 and create a potential winning opportunity. In terms of control of the center, X occupies 5 and has a higher chance of winning in the initial board position. Potential strategies for O include occupying 1, 4, 7 to create a potential winning opportunity. X could have occupied 3, 5, 7 to win the game. X has already occupied 3, 5, and there is 1 step to complete the plan.
Final Evaluation: From the description of next board, it appears that the initial board position
O | O | X
---------
4 | X | 6
---------
7 | 8 | 9
was not favorable for O. X was able to occupy 3, 5, 7 and ultimately win the game. The win probability for X is large, while the win probability for O is low. From the perspective of threat, X was able to occupy 7 before O, blocking O's potential winning opportunity and ultimately win the game. O was able to occupy 1, 4, 7 and create a potential winning opportunity. From the perspective of control of the center, X occupies 5, which is a key position to win the game. Potential strategies for O include occupying 1, 4, 7 to create a potential winning opportunity. X could have occupied 3, 5, 7 to win the game. X has already occupied 3, 5, and there is 1 step to complete the plan.
"""
# TD_PROMPT = "You are a skilled player of the game of Tic Tac Toe. The goal is to get multiple of one\'s own symbols in a row, either horizontally, vertically, or diagonally, before the opponent does. If all nine squares are filled and no player has three in a row, the game is a draw. The following is a sequence boards of a game. Your task is to evaluate a given board position.\nThe board consists of \"O\", \"X\" and grid number. The grid number indicates empty grid.\n\nYou should output your answer in the json format. Your answer consists of two elements:\n- \"thought\": let's think step by step. Generate your detailed evaluation by analyzing the game from different perspectives.\n- \"final_evaluation\": Judge which side takes advantage. 1 means you think 'O' takes advantage, -1 means you think 'X' takes advantage, and 0 means you think the game will be a draw.\n\nHere is the board position and it is {player}\'s turn.\n{board}.\n The next board position is \n{next_board}.\n and it is {next_player}\'s turn.\n\n"


def convert_to_llama_format(row):
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
        + row.replace(
            "\n\n\nHere is the board position and the next player is",
            "<|eot_id|>\n<|start_header_id|>user<|end_header_id|>Here is the board position and the next player is",
        ).replace(
            '\n\n{"thought":',
            '<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>{"thought"',
        )
        + "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    )


class MC_prompt:
    def __init__(self, prompt=MC_PROMPT):
        self.prompt = prompt

    def format_input(self, traj_data):
        message = [
            {
                "role": "system",
                "content": MC_SYSTEM_PROMPT
                + "### EXAMPLE:\nuser:"
                + MC_EXAMPLE_USER_PROMPT
                + "assistant:"
                + MC_EXAMPLE_ASSISTENT_PROMPT,
            }
        ]
        trajectory_prompt = self.get_trajectory_prompt(traj_data)
        message.append({"role": "user", "content": trajectory_prompt})
        return message

    def get_trajectory_prompt(self, traj_data):
        trajectory_prompt = ""
        states = traj_data["state"]
        actions = traj_data["action"]
        for ind in range(len(states)):
            state = states[ind]
            board = state_to_board(state)
            player = "X" if state[1] == "O" else "O"
            if ind == 0:
                trajectory_prompt += STATE_TO_VALUE_PROMPT.format(
                    player=state[1], state=board
                )
            else:
                action = (
                    actions[ind - 1] + 1
                )  # +1 because action id + 1 = the digit shown on the board
                trajectory_prompt += CONCAT_PROMPT.format(
                    player=player, board=board, action=action
                )
        final_board = convert_input_to_blocks(traj_data["state"][-1][0])
        result = check_winner(final_board)
        if result[0] == "draw":
            trajectory_prompt += "The game is over. It's a draw.\n\n"
        else:
            trajectory_prompt += f"The game is over. {result[0]} wins. {result[0]} wins by occupying the positions {result[1]}.\n\n"
        return trajectory_prompt

    def __call__(self, traj_data):
        return self.format_input(traj_data)


class MC_prompt_sa:
    def __init__(self, prompt=MC_PROMPT, add_example=True):
        self.prompt = prompt
        self.add_example = add_example

    def format_input(self, traj_data):
        if self.add_example:
            message = [
                {
                    "role": "system",
                    "content": MC_SYSTEM_PROMPT_SA
                    + "### EXAMPLE:\nuser:"
                    + MC_EXAMPLE_USER_PROMPT_SA
                    + "assistant:"
                    + MC_EXAMPLE_ASSISTENT_PROMPT_SA,
                }
            ]
        else:
            message = [{"role": "system", "content": MC_SYSTEM_PROMPT_SA}]
        trajectory_prompt = self.get_trajectory_prompt(traj_data)
        message.append({"role": "user", "content": trajectory_prompt})
        return message

    def get_trajectory_prompt(self, traj_list):
        trajectory_prompt = ""
        for i_traj, traj in enumerate(traj_list):
            states = traj["state"]
            actions = traj["action"]
            for ind in range(len(states)):
                state = states[ind]
                board = state_to_board(state)
                player = "X" if state[1] == "O" else "O"
                if ind == 0:
                    if i_traj == 0:
                        trajectory_prompt += STATE_TO_VALUE_PROMPT_SA.format(
                            player=state[1], state=board, action=actions[0] + 1
                        )
                    trajectory_prompt += NEW_TRAJ_BEGIN_PROMPT.format(
                        idx=i_traj + 1
                    )  # to make it start from 1
                else:
                    action = (
                        actions[ind - 1] + 1
                    )  # +1 because action id + 1 = the digit shown on the board
                    trajectory_prompt += CONCAT_PROMPT.format(
                        player=player, board=board, action=action
                    )
            final_board = convert_input_to_blocks(traj["state"][-1][0])
            result = check_winner(final_board)
            if result[0] == "draw":
                trajectory_prompt += "The game is over. It's a draw.\n\n"
            else:
                trajectory_prompt += f"The game is over. {result[0]} wins. {result[0]} wins by occupying the positions {result[1]}.\n\n"
        trajectory_prompt += (
            "Now generate your evaluation for the (board, action) pair."
        )
        return trajectory_prompt

    def __call__(self, traj_data):
        return self.format_input(traj_data)


# class POLICY_IMPROVEMENT_prompt():
#     def __init__(self, prompt=TIC_TAC_TOE_STATE_POLICY_PROMPT_GPT):
#         self.prompt = prompt

#     def format_input(self, traj_data):
#         message = [{"role": "system", "content": MC_SYSTEM_PROMPT + "### EXAMPLE:\nuser:" + MC_EXAMPLE_USER_PROMPT + "assistant:" + MC_EXAMPLE_ASSISTENT_PROMPT}]
#         trajectory_prompt = self.get_trajectory_prompt(traj_data)
#         message.append({"role": "user", "content": trajectory_prompt})
#         return message

#     def get_trajectory_prompt(self, traj_data):
#         trajectory_prompt = ""
#         for ind, state in enumerate(traj_data):
#             board = state_to_board(state)
#             player = 'X' if state[1] == 'O' else 'O'
#             if ind == 0:
#                 trajectory_prompt += STATE_TO_ACTION_PROMPT.format(player=state[1], state=board)
#             else:
#                 trajectory_prompt += CONCAT_PROMPT.format(player=player, board=board)
#         final_board = convert_input_to_blocks(traj_data[-1][0])
#         result = check_winner(final_board)
#         if result[0] == 'draw':
#             trajectory_prompt += "The game is over. It's a draw.\n\n"
#         else:
#             trajectory_prompt += f"The game is over. {result[0]} wins. {result[0]} wins by occupying the positions {result[1]}.\n\n"
#         return trajectory_prompt

#     def __call__(self, traj_data):
#         return self.format_input(traj_data)


class TD_prompt:
    def __init__(self, prompt=TD_PROMPT):
        self.prompt = prompt

    def format_input(self, state, next_state, discription):
        board = state_to_board(state)
        next_board = state_to_board(next_state)
        trajectory_prompt = STATE_TO_VALUE_PROMPT.format(state=board)
        player = state[1]
        next_player = next_state[1]
        # trajectory_prompt += CONCAT_PROMPT.format(player=next_player, board=next_board)
        trajectory_prompt += (
            "Information for the next board: "
            + CONCAT_PROMPT.format(player=next_player, board=next_board)
            + "Discription for the next board:\n"
            + discription
            + TD_EXAMPLE_PROMPT
        )
        prompt = self.prompt.format(trajectory=trajectory_prompt)
        return prompt

    def __call__(self, state, next_state, discription):
        return self.format_input(state, next_state, discription)


class EVAL_prompt:
    def __init__(self, prompt=TIC_TAC_TOE_STATE_EVAL_PROMPT_GPT):
        self.prompt = prompt
        split = "Here is the board position and the next player is"
        self.system_prompt = TIC_TAC_TOE_STATE_EVAL_PROMPT_GPT.split(split)[0]
        self.user_prompt = split + TIC_TAC_TOE_STATE_EVAL_PROMPT_GPT.split(split)[1]

    def format_input(self, state, response_type="LLM"):
        board = state_to_board(state)
        if response_type == "LLM":
            prompts = [{"role": "system", "content": self.system_prompt}]
            prompts.append(
                {
                    "role": "user",
                    "content": self.user_prompt.format(
                        state=board, next_player=state[-1]
                    ),
                }
            )
            return {"state": state[0], "prompt": prompts}
        else:
            raise NotImplementedError
            return

    def __call__(self, state, response_type="LLM"):
        return self.format_input(state, response_type=response_type)


class EVAL_prompt_sa:
    def __init__(self):
        split = "Here is the board position and the next player is"
        self.system_prompt = TIC_TAC_TOE_STATE_EVAL_PROMPT_GPT_SA.split(split)[0]
        self.user_prompt = split + TIC_TAC_TOE_STATE_EVAL_PROMPT_GPT_SA.split(split)[1]

    def format_input(self, state_action_dict, response_type="LLM"):
        state = state_action_dict["state"]
        action = state_action_dict["action"]
        board = state_to_board(state)
        if response_type == "LLM":
            prompts = [{"role": "system", "content": self.system_prompt}]
            prompts.append(
                {
                    "role": "user",
                    "content": self.user_prompt.format(
                        state=board, next_player=state[-1], action=action + 1
                    ),
                }
            )
            return {"state": state[0], "prompt": prompts}
        else:
            raise NotImplementedError
            return

    def __call__(self, state_action_dict, response_type="LLM"):
        return self.format_input(state_action_dict, response_type=response_type)


class POLICY_prompt(EVAL_prompt):
    def __init__(self, prompt=TIC_TAC_TOE_STATE_POLICY_PROMPT_GPT):
        super().__init__(prompt)

    def format_input(
        self, state, example_prompt, available_actions=None, response_type="LLM"
    ):
        if len(state) != 2:
            state = (state, check_next_player(state))
        board = state_to_board(state)
        if available_actions is None:
            available_actions = [i + 1 for i in range(9) if state[0][i] == 0]
        else:
            available_actions = [i + 1 for i in available_actions]
        if response_type == "LLM":
            prompts = [
                {
                    "role": "system",
                    "content": TIC_TAC_TOE_STATE_POLICY_SYSTEM_PROMPT_GPT
                    + "### EXAMPLE:\nuser:"
                    + TIC_TAC_TOE_POLICY_EXAMPLE_USER_PROMPT
                    + "assistant:"
                    + TIC_TAC_TOE_POLICY_EXAMPLE_ASSISTANT_PROMPT,
                }
            ]
            prompts.append(
                {
                    "role": "user",
                    "content": TIC_TAC_TOE_STATE_POLICY_USER_PROMPT.format(
                        state=board,
                        next_player=state[-1],
                        available_positions=available_actions,
                    ),
                }
            )
            return {"state": state[0], "prompt": prompts}
        elif response_type == "llama":
            # example = "<|eot_id|>\n<|start_header_id|>user<|end_header_id|>" + TIC_TAC_TOE_POLICY_EXAMPLE_USER_PROMPT + "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>" + TIC_TAC_TOE_POLICY_EXAMPLE_ASSISTANT_PROMPT
            example = ""
            prompts = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                + TIC_TAC_TOE_STATE_POLICY_SYSTEM_PROMPT_GPT
                + example
                + "<|eot_id|>\n<|start_header_id|>user<|end_header_id|>"
                + TIC_TAC_TOE_STATE_POLICY_USER_PROMPT.format(
                    state=board,
                    next_player=state[-1],
                    available_positions=available_actions,
                )
                + "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
            )

            return prompts

    def __call__(
        self,
        state,
        example_prompt=TIC_TAC_TOE_POLICY_EXAMPLE_PROMPT,
        available_actions=None,
        response_type="LLM",
    ):
        return self.format_input(
            state,
            example_prompt=example_prompt,
            available_actions=available_actions,
            response_type=response_type,
        )


# class POLICY_IMPROVEMENT_prompt:
#     def __init__(self, prompt=POLICY_IMPROVEMENT_PROMPT):
#         self.prompt = prompt
#         self.system_prompt = self.prompt.split("{example_prompt}")[0]
#         self.user_prompt = self.prompt.split("{example_prompt}")[1]
#         # self.value_dict = read_jsonl("data/board_value_dict.jsonl")[0]
#     def format_input_v1(self, state, example_prompt):
#         board = state_to_board(state)
#         # state = ((2, 1, 2, 0, 2, 1, 0, 0, 1), 1)
#         available_positions = [i + 1 for i in range(9) if state[0][i] == 0]
#         next_states = []
#         current_player = "O" if state[-1] == 1 else "X"
#         for i in range(9):
#             if state[0][i] == 0:
#                 next_board = list(deepcopy(state[0]))
#                 next_board[i] = state[1]
#                 next_states.append(f"\n### Evaluation for taking action {i + 1}:\n" + self.value_dict[str(tuple(next_board))])
#         prompt = self.prompt.format(state=board, next_player=current_player, available_positions=available_positions, next_states="\n".join(next_states), example_prompt=example_prompt)
#         return prompt

#     def format_input_v2(self, state, example_prompt, next_states, response_type="LLM"):
#         board = state_to_board(state)
#         available_positions = [i + 1 for i in range(9) if state[0][i] == 0]
#         current_player = "O" if state[-1] == 1 else "X"
#         values = []
#         for i, next_board in enumerate(next_states):
#             values.append(f"\n### Evaluation for taking action {i + 1}:\n" + next_board['value'])
#         if response_type == "LLM":
#             prompt = [{"role": "system", "content": self.system_prompt}]
#             prompt.append({"role": "user", "content": self.user_prompt.format(state=board, next_player=current_player, available_positions=available_positions, next_states="\n".join(values), example_prompt=example_prompt)})
#             return {"state": state[0], "prompt": prompt}
#         elif response_type == "llama":
#             prompt = self.prompt.format(state=board, next_player=current_player, available_positions=available_positions, next_states="\n".join(values), example_prompt=example_prompt)
#             return prompt

#     def get_next_state(self, state):
#         next_states = []
#         for i in range(9):
#             if state[0][i] == 0:
#                 next_board = list(deepcopy(state[0]))
#                 next_board[i] = 1 if state[1] == 'O' else 2
#                 next_states.append((next_board, "X")) if state[1] == 'O' else next_states.append((next_board, "O"))
#         return next_states

#     def __call__(self, state, example_prompt=None, next_states=None, response_type="LLM"):
#         return self.format_input_v2(state, example_prompt, next_states, response_type=response_type)


class POLICY_IMPROVEMENT_prompt_sa:

    def __init__(self, prompt=POLICY_IMPROVEMENT_PROMPT):
        self.prompt = prompt
        self.system_prompt = POLICY_IMPROVEMENT_SYSTEM_PROMPT_SA
        self.user_prompt = POLICY_IMPROVEMENT_USER_PROMPT_SA

    def format_input_v2(
        self, state, example_prompt, actions, next_states, response_type="LLM"
    ):
        board = state_to_board(state)
        if actions is not None:
            available_positions = actions
        else:
            available_positions = [i + 1 for i in range(9) if state[0][i] == 0]
            assert len(available_positions) == len(next_states)
        current_player = state[-1]
        values = []
        for i, next_board in enumerate(next_states):
            values.append(
                f"\n### Evaluation for taking action {available_positions[i]}:\n"
                + next_board["value"]
            )
        if response_type == "LLM":
            prompt = [{"role": "system", "content": self.system_prompt}]
            prompt.append(
                {
                    "role": "user",
                    "content": self.user_prompt.format(
                        state=board,
                        next_player=current_player,
                        available_positions=available_positions,
                        next_states="\n".join(values),
                        example_prompt=example_prompt,
                    ),
                }
            )
            return {"state": state[0], "prompt": prompt}
        elif response_type == "llama":
            prompt = self.prompt.format(
                state=board,
                next_player=current_player,
                available_positions=available_positions,
                next_states="\n".join(values),
                example_prompt=example_prompt,
            )
            return prompt

    def get_state_action_pair(state):
        state_action_pairs = []
        for i in range(9):
            if state[0][i] == 0:
                state_action_pairs.append({"state": state, "action": i})
        return state_action_pairs

    def __call__(
        self,
        state,
        example_prompt=None,
        actions=None,
        next_states=None,
        response_type="LLM",
    ):
        return self.format_input_v2(
            state, example_prompt, actions, next_states, response_type=response_type
        )


if __name__ == "__main__":
    # from sampling.llm_call import openai_v0_chat, openai_v0_function_calling, groq_llama3_70b, gpt4_chat
    # from func_utils import load_replay_buffer
    # td_prompt = TD_prompt()
    # mc_prompt = MC_prompt()
    # data = load_replay_buffer("nlrl/envs/tictactoe/replay_buffer.jsonl")
    # description = """It appears that the board position \nO | O | X\n---------\nO | X | 6\n---------\n7 | 8 | 9\nwas not favorable for O, as X was able to capitalize on O's moves and ultimately win the game. The win probability for X is large, while the win probability for O is low. From the perspective of threat, X was able to occupy 3, 5, 7 and ultimately win the game. O was able to occupy 1, 4, 7 and create a potential winning opportunity. In terms of control of the center, X occupies 5 and has a higher chance of winning in the initial board position. Potential strategies for O include occupying 1, 4, 7 to create a potential winning opportunity. X could have occupied 3, 5, 7 to win the game. X has already occupied 3, 5, and there is 1 step to complete the plan."""
    # # prompt = td_prompt(data[7][0]['state'][1], data[0][0]['state'][2], discription=description)
    # prompt = mc_prompt.format_input(data[3][0]['state'][-3:])
    # for p in prompt:
    #     print(f"{p['role']}: {p['content']}")
    # print("------------------")
    # response = groq_llama3_70b(prompt)
    # print(response)
    prompter = POLICY_IMPROVEMENT_prompt_sa()
    board = [[0, 0, 0, 0, 0, 0, 0, 0, 0], 1]
    prompt = prompter(board, "")
    print(prompt)
