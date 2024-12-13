## Environment
def get_env(env_config):
    if env_config.env_name.startswith("spiel_"):
        import pyspiel
        # Check environment is the forked one
        env = pyspiel.load_game("breakthrough", {"rows": 5, "columns": 5})
        assert env.new_initial_state().serialize() == "bbbbbbbbbb.....wwwwwwwwww0"

        env_name = env_config.env_name.replace("spiel_", "")
        params = env_config.params
        if not isinstance(params, dict):
            params = params.to_dict()
        return pyspiel.load_game(env_name, params)
    else:
        from .tictactoe.tictactoe import TicTacToeEnv_Wrapper
        OTHER_ENV_DICT = {"TicTacToeEnv": TicTacToeEnv_Wrapper}
        return OTHER_ENV_DICT[env_config.env_name](env_config=env_config)
