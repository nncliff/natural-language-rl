
STATE_DATA_PATH=breakthrough/rollout_collection_5x5_processed/sample_1_40/merged_initial_state.jsonl
BASE_REPLAY_BUFFER_PATH=breakthrough/rollout_collection_5x5_processed/sample_1_40/

# Run multi-step expansion starting from the collected states
# The for-loop is for iteratively adding new state data (from multi-step rollout) into the init_state_buffer
# Note that we set lookahead_num_rollouts=4, but we only use 2 when doing TD
START=1
END=1
for i in $(seq $START $END)
do
    echo $i
    if [ $i -eq 1 ]; then
        STATE_DATA_PATH=$STATE_DATA_PATH
        OLD_LOOKAHEAD_DIR="None"
    else
        STATE_DATA_PATH=${BASE_REPLAY_BUFFER_PATH}/initial_states/init_state_buffer.jsonl
        OLD_LOOKAHEAD_DIR=${BASE_REPLAY_BUFFER_PATH}/look_ahead/replay_buffer.jsonl
    fi
    python3 breakthrough/collect_rollout_data.py --config ./breakthrough/configs/rollout_config.py \
                                                        --config.state_data_path $STATE_DATA_PATH \
                                                        --config.old_lookahead_dir ${OLD_LOOKAHEAD_DIR} \
                                                        --config.all_initial_state_save_path ${BASE_REPLAY_BUFFER_PATH}/initial_states \
                                                        --config.policy_config.max_simulations 1000 \
                                                        --config.policy_config.mcts_rollout_count 100 \
                                                        --config.opponent_policy_config.max_simulations 1000 \
                                                        --config.opponent_policy_config.mcts_rollout_count 100 \
                                                        --config.rollout_config.rollout_method "multi_step_given_boards" \
                                                        --config.rollout_config.init_state_dedup True \
                                                        --config.replay_buffer_dir ${BASE_REPLAY_BUFFER_PATH}/look_ahead \
                                                        --config.rollout_config.lookahead_num_rollouts 4 \
                                                        --config.rollout_config.lookahead_step 4 \
                                                        --config.rollout_config.worker_num 256
done