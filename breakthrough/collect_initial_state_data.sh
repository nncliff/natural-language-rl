max_sims="2 10 100 1000"
max_rollouts="1 10 100"

# Sample replay buffer from scratch for different policy combinations
for p_max_sim in $max_sims; do
    for o_max_sim in $max_sims; do
        for p_max_rollout in $max_rollouts; do
            for o_max_rollout in $max_rollouts; do
                python3 breakthrough/collect_rollout_data.py --config ./breakthrough/configs/rollout_config.py \
                                                      --config.policy_config.max_simulations $p_max_sim \
                                                      --config.policy_config.mcts_rollout_count $p_max_rollout \
                                                      --config.opponent_policy_config.max_simulations $o_max_sim \
                                                      --config.opponent_policy_config.mcts_rollout_count $o_max_rollout \
                                                      --config.rollout_config.num_rollouts 300 \
                                                      --config.replay_buffer_dir breakthrough/rollout_collection_5x5/rollout_${p_max_sim}_${p_max_rollout}_${o_max_sim}_${o_max_rollout}
            done
        done
    done
done