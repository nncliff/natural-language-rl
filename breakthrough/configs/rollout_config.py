from ml_collections import config_dict


def get_config():
    cfg = config_dict.ConfigDict()
    cfg.env_config = get_env_config()
    cfg.policy_config = get_policy_config(
        cfg.env_config,
        policy_name="MCTS",
        uct_c=1.0,
        max_simulations=10,
        mcts_rollout_count=1
    )
    cfg.opponent_policy_config = get_policy_config(
        cfg.env_config,
        policy_name="MCTS",
        uct_c=1.0,
        max_simulations=10,
        mcts_rollout_count=1
    )
    cfg.rollout_config = get_rollout_config()

    # Data save path
    cfg.replay_buffer_dir = "" # Data save path
    # Data load path (for given board multi-step rollout)
    cfg.state_data_path = ""
    # Experiment note
    cfg.note = ""

    # Multi-step rollout config
    cfg.old_lookahead_dir = "" # Lookahead data checkpoint, append new data to it
    cfg.all_initial_state_save_path = "" # Extract all deduplicated states
    return cfg


def get_env_config():
    cfg = config_dict.ConfigDict()
    cfg.env_name = "spiel_breakthrough"
    cfg.batch_sample = False
    cfg.batch_sample_size = 1
    cfg.params = {"rows": 5, "columns": 5}
    return cfg


def get_policy_config(env_config, policy_name, uct_c, max_simulations, mcts_rollout_count):
    cfg = config_dict.ConfigDict()
    cfg.policy_name = policy_name
    cfg.env_config = env_config
    cfg.uct_c = uct_c
    cfg.max_simulations = max_simulations
    cfg.mcts_deterministic = False
    cfg.mcts_rollout_count = mcts_rollout_count
    cfg.mcts_solve = True
    cfg.mcts_verbose = False
    return cfg

def get_rollout_config():
    cfg = config_dict.ConfigDict()
    # "traj_scratch" or "traj_given_boards" or "multi_step_given_boards"
    cfg.rollout_method = "traj_scratch"
    cfg.num_rollouts = 32 # Number of rollouts
    cfg.worker_num = 192 # Number of workers

    # Whether to deduplicate the initial state when given initial boards
    cfg.init_state_dedup = False

    # whether to sub sample the state
    cfg.sub_sample_state = -1

    # Multi-step rollout config
    # How mant steps to look ahead
    cfg.lookahead_step = 4
    # How many priciple variations to take
    cfg.lookahead_num_rollouts = 2
    return cfg