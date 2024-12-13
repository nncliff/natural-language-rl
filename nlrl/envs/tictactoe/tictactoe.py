from gym_tictactoe.env import TicTacToeEnv, check_game_status, parse_current_mark


class TicTacToeEnv_Wrapper(TicTacToeEnv):
    def __init__(self, env_config, **kwargs):
        super(TicTacToeEnv_Wrapper, self).__init__(**kwargs)

    def set_state(self, state=None, player=None):
        state, mark = state
        assert mark == parse_current_mark(state)

        info = check_game_status(state)
        if info != -1:
            done = True
        else:
            done = False
        self.board = list(state)
        self.mark = parse_current_mark(state)
        self.done = done
        # note that don't directly return self.board
        # this is a reference, not a copy
        # any modification to the board will influence outer variables
        return self._get_obs(), done

    def get_available_actions(self):
        return [i for i in range(9) if self.board[i] == 0]
