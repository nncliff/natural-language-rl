p_max_sim=1000
p_max_rollout=100
o_max_sim=1000
o_max_rollout=100

STATE_DATA_PATH=breakthrough/rollout_collection_5x5_processed/evaluation_3000/merged_initial_state.jsonl
ROLLOUT_DATA_SAVE_PATH=breakthrough/rollout_collection_5x5_processed/evaluation_3000/win_rate_collection/

# Conduct rollout over given initial states
python3 breakthrough/collect_rollout_data.py --config ./breakthrough/configs/rollout_config.py \
                                                      --config.state_data_path $STATE_DATA_PATH \
                                                      --config.policy_config.max_simulations $p_max_sim \
                                                      --config.policy_config.mcts_rollout_count $p_max_rollout \
                                                      --config.opponent_policy_config.max_simulations $o_max_sim \
                                                      --config.opponent_policy_config.mcts_rollout_count $o_max_rollout \
                                                      --config.rollout_config.rollout_method "traj_given_boards" \
                                                      --config.rollout_config.init_state_dedup True \
                                                      --config.rollout_config.num_rollouts 30 \
                                                      --config.rollout_config.worker_num 256 \
                                                      --config.replay_buffer_dir $ROLLOUT_DATA_SAVE_PATH

# Calculate win-rate given rollout data and generate eval datset
python3 breakthrough/count_win_rate.py --config breakthrough/configs/win_rate.py \
                                                --config.input_path ${ROLLOUT_DATA_SAVE_PATH}/replay_buffer.jsonl \
                                                --config.output_path ${ROLLOUT_DATA_SAVE_PATH}