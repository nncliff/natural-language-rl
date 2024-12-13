from typing import Any
from open_spiel.python.algorithms import mcts
from nlrl.envs import get_env
import numpy as np
import pyspiel


class MCTS_AGENT:
    def __init__(self, policy_config):
        env_config = policy_config.env_config

        name = env_config.env_name
        params = env_config.params
        if not isinstance(params, dict):
            params = params.to_dict()
        name = name.replace("spiel_", "")
        self.game = pyspiel.load_game(name, params)

        rng = None
        if policy_config.mcts_deterministic:
            rng = np.random.RandomState(42)

        evaluator = mcts.RandomRolloutEvaluator(policy_config.mcts_rollout_count, rng)
        self.agent = mcts.MCTSBot(
            self.game,
            policy_config.uct_c,
            policy_config.max_simulations,
            evaluator,
            random_state=rng,
            solve=policy_config.mcts_solve,
            verbose=policy_config.mcts_verbose,
        )

    # Inform the agent of the action taken by the current player
    # Only for pyspiel implementation
    def inform_action(self, state, current_player, action):
        self.agent.inform_action(state, current_player, action)

    def __call__(self, state) -> Any:
        if isinstance(state, list):
            return [self.agent.step(s) for s in state]
        else:
            return self.agent.step(state)
