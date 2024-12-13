#!/bin/bash
set -ex
export PYTHONPATH="$PWD:$PYTHONPATH"

# export CUDA_VISIBLE_DEVICES=4,5,6,7
TOP_K_SAMPLE=10
NUM_POLICY_SAMPLE=10
N_MC_TRAJ=5
NUM_ROLLOUTS=512
OPPONENT_POLICY_NAME=${1:-"Random"}
NUM_TRAIN_EPOCH=2
N_HISTORY=3
EXP_BASE_DIR=./results-tictactoe-llama3.1-70b-prompting/${2:-"exp"}
exp_dir="${EXP_BASE_DIR}/oppo_${OPPONENT_POLICY_NAME}_train_${NUM_TRAIN_EPOCH}_hist_${N_HISTORY}_sample_${NUM_POLICY_SAMPLE}_top_${TOP_K_SAMPLE}_nmc_${N_MC_TRAJ}_rollout_${NUM_ROLLOUTS}"
BASE_EVAL_TRAJ_PATH="${exp_dir}/data/eval/replay_buffer"
MODEL_PATH=${3:-"path/to/llama-3.1-70b"}

EVAL_TRAJ_PATH="${BASE_EVAL_TRAJ_PATH}_0.jsonl"
python3 tictactoe/collect_rollout_data.py \
    --policy_name "LLM" \
    --opponent_policy_name $OPPONENT_POLICY_NAME \
    --replay_buffer_path $EVAL_TRAJ_PATH \
    --rollout_method scratch \
    --num_rollouts $NUM_ROLLOUTS \
    --model_path $MODEL_PATH \
    --epsilon_greedy 0 --temp 0
python nlrl/evaluate.py --data_dir ${exp_dir}/data/eval
