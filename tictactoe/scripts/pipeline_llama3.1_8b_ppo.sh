#!/bin/bash
set -ex

# ppo hyperparameters
BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=2
LR_ACTOR=1e-5
LR_CRITIC=1e-5
PPO_ITERATIONS=2
EPS=0.1
VALUE_COEF=1.0
ENTROPY_COEF=0.0
CLIP_EPS=0.2

# export CUDA_VISIBLE_DEVICES=4,5,6,7
TOP_K_SAMPLE=10
NUM_POLICY_SAMPLE=10
N_MC_TRAJ=5
NUM_ROLLOUTS=512
OPPONENT_POLICY_NAME=${1:-"Random"}
NUM_TRAIN_EPOCH=1
N_HISTORY=3

# Directory setup
EXP_BASE_DIR=./results-tictactoe-llama3.1-8b-ppo/${2:-"exp"}
exp_dir="${EXP_BASE_DIR}/oppo_${OPPONENT_POLICY_NAME}_train_${NUM_TRAIN_EPOCH}_hist_${N_HISTORY}_rollout_${NUM_ROLLOUTS}_iterations_${PPO_ITERATIONS}"
BASE_POLICY_MODEL_PATH="${exp_dir}/models/policy"
BASE_VALUE_MODEL_PATH="${exp_dir}/models/value"
BASE_REPLAY_BUFFER_PATH="${exp_dir}/data/replay_buffer"
BASE_EVAL_TRAJ_PATH="${exp_dir}/data/eval/replay_buffer"

# Create directories
mkdir -p ${exp_dir}/data/
mkdir -p ${exp_dir}/models/
mkdir -p ${exp_dir}/data/eval/


# Set iteration range
START_ITERATION_NUM=1
FINAL_ITERATION_NUM=50

echo "Starting PPO training from iteration $START_ITERATION_NUM to $FINAL_ITERATION_NUM"

if [ $START_ITERATION_NUM -gt 1 ]; then
    POLICY_MODEL_PATH="${BASE_POLICY_MODEL_PATH}_$(($START_ITERATION_NUM-1))"
    POLICY_CHECKPOINT=$(find $POLICY_MODEL_PATH -type d -name 'checkpoint*' | sort | head -n 1)
fi

# echo $POLICY_CHECKPOINT
SMALL_LLM_NAME=${SMALL_LLM_PATH:-"path/to/llama-3.1-8b"}
BASE_MODEL_PATH=$SMALL_LLM_NAME
CURRENT_POLICY_MODEL_PATH="${SMALL_LLM_NAME}"

# Main training loop
for i in $(seq $START_ITERATION_NUM $FINAL_ITERATION_NUM)
do
    echo "Iteration $i"

    # Set paths for current iteration
    NEW_POLICY_MODEL_SAVE_PATH="${BASE_POLICY_MODEL_PATH}_${i}"
    REPLAY_BUFFER_PATH="${BASE_REPLAY_BUFFER_PATH}_${i}.jsonl"
    EVAL_TRAJ_PATH="${BASE_EVAL_TRAJ_PATH}_$(($i-1)).jsonl"

    echo "New policy model save path: $NEW_POLICY_MODEL_SAVE_PATH"
    echo "Replay buffer path: $REPLAY_BUFFER_PATH"
    echo "Evaluation trajectory path: $EVAL_TRAJ_PATH"

    if [ $i -eq 1 ]; then
        CURRENT_POLICY_MODEL_PATH="${SMALL_LLM_NAME}"

        # Initial evaluation
        CUDA_VISIBLE_DEVICES=0 python3 tictactoe/collect_rollout_data.py \
            --policy_name "LLM_PPO" \
            --opponent_policy_name $OPPONENT_POLICY_NAME \
            --replay_buffer_path $EVAL_TRAJ_PATH \
            --rollout_method scratch \
            --num_rollouts $NUM_ROLLOUTS \
            --model_path $CURRENT_POLICY_MODEL_PATH \
            --epsilon_greedy 0 --temp 0 \
            --prompt_logprobs 1 \
            --max_tokens 1 \
            --env_parallel_num 8 &

        # Initial rollout with epsilon-greedy exploration
        CUDA_VISIBLE_DEVICES=3 python3 tictactoe/collect_rollout_data.py \
            --policy_name "Random" \
            --opponent_policy_name $OPPONENT_POLICY_NAME \
            --replay_buffer_path $REPLAY_BUFFER_PATH \
            --rollout_method scratch \
            --num_rollouts $NUM_ROLLOUTS \
            --model_path $CURRENT_POLICY_MODEL_PATH \
            --epsilon_greedy 0 --temp 0 \
            --prompt_logprobs 1 \
            --max_tokens 1 \
            --env_parallel_num 8  &


        # wait
        # cp -r $REPLAY_BUFFER_PATH $EVAL_TRAJ_PATH


        # Train PPO on collected data
        CUDA_VISIBLE_DEVICES=3 python3 tictactoe/train_ppo.py \
            --model_path $CURRENT_POLICY_MODEL_PATH \
            --replay_buffer_path $REPLAY_BUFFER_PATH \
            --save_dir $NEW_POLICY_MODEL_SAVE_PATH \
            --num_iterations $PPO_ITERATIONS \
            --value_coef $VALUE_COEF \
            --entropy_coef $ENTROPY_COEF \
            --clip_epsilon $CLIP_EPS \
            --lr_actor $LR_ACTOR \
            --lr_critic $LR_CRITIC \
            --batch_size $BATCH_SIZE \
            --n_epochs $NUM_TRAIN_EPOCH \
            --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS

    else
        CURRENT_POLICY_MODEL_SAVE_PATH="${BASE_POLICY_MODEL_PATH}_$(($i-1))"
        CURRENT_POLICY_CHECKPOINT="${BASE_POLICY_MODEL_PATH}_$(($i-1))"

        CUDA_VISIBLE_DEVICES=0 python3 tictactoe/collect_rollout_data.py \
            --policy_name "LLM_PPO" \
            --opponent_policy_name $OPPONENT_POLICY_NAME \
            --replay_buffer_path $EVAL_TRAJ_PATH \
            --rollout_method scratch \
            --num_rollouts $NUM_ROLLOUTS \
            --model_path $CURRENT_POLICY_CHECKPOINT \
            --epsilon_greedy 0 \
            --prompt_logprobs 1 \
            --max_tokens 1 --temp 0 \
            --env_parallel_num 8 &

        # Collect data with current policy
        CUDA_VISIBLE_DEVICES=3 python3 tictactoe/collect_rollout_data.py \
            --policy_name "LLM_PPO" \
            --opponent_policy_name $OPPONENT_POLICY_NAME \
            --replay_buffer_path $REPLAY_BUFFER_PATH \
            --rollout_method scratch \
            --num_rollouts $NUM_ROLLOUTS \
            --model_path $CURRENT_POLICY_CHECKPOINT \
            --epsilon_greedy 0 --temp "0.1" \
            --prompt_logprobs 1 \
            --max_tokens 1 \
            --env_parallel_num 8 &

        wait

        # Train PPO on collected data
        CUDA_VISIBLE_DEVICES=3 python3 tictactoe/train_ppo.py \
            --model_path $SMALL_LLM_NAME \
            --load_dir $CURRENT_POLICY_MODEL_SAVE_PATH \
            --replay_buffer_path $REPLAY_BUFFER_PATH \
            --save_dir $NEW_POLICY_MODEL_SAVE_PATH \
            --num_iterations $PPO_ITERATIONS \
            --value_coef $VALUE_COEF \
            --entropy_coef $ENTROPY_COEF \
            --clip_epsilon $CLIP_EPS \
            --lr_actor $LR_ACTOR \
            --lr_critic $LR_CRITIC \
            --batch_size $BATCH_SIZE \
            --n_epochs $NUM_TRAIN_EPOCH \
            --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS
    fi

    # Get latest checkpoint
    NEW_POLICY_CHECKPOINT="${NEW_POLICY_MODEL_SAVE_PATH}/actor"
    # Generate evaluation report
    python nlrl/evaluate.py --data_dir "${exp_dir}/data/eval"
done

# Final evaluation
FINAL_EVAL_PATH="${BASE_EVAL_TRAJ_PATH}_${i}.jsonl"
python3 tictactoe/collect_rollout_data.py \
    --policy_name "LLM_PPO" \
    --opponent_policy_name $OPPONENT_POLICY_NAME \
    --replay_buffer_path $FINAL_EVAL_PATH \
    --rollout_method scratch \
    --num_rollouts $NUM_ROLLOUTS \
    --model_path $POLICY_CHECKPOINT \
    --epsilon_greedy 0 \
    --prompt_logprobs 1 \
    --max_tokens 1 --temp 0

# Generate evaluation report
python nlrl/evaluate.py --data_dir "${exp_dir}/data/eval"
