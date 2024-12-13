# TicTacToe: Natural Language Actor-Critic Learning

Implementation of Natural Language Actor-Critic Learning for the TicTacToe environment, demonstrating language-based reinforcement learning with Monte Carlo value estimation.

## Model Architecture
Our implementation uses three language models:
- LLaMA-3.1-70B-Instruct: Implements language aggregator and policy improvement operator
- Two LLaMA-3.1-8B-Instruct models:
  - One serves as language policy (π_L)
  - One implements language value function (Q_π^L)

## Pipeline Components

### 1. Data Collection
```bash
# Collect rollout trajectories
python tictactoe/collect_rollout_data.py \
    --policy_name "LLM" \
    --opponent_policy_name "Random" \
    --replay_buffer_path data/replay_buffer.jsonl \
    --rollout_method scratch \
    --num_rollouts 512 \
    --model_path "path/to/llama-8b"

# Monte Carlo Value Estimation
python tictactoe/prompt_llm.py \
    --method mc_value_q \
    --max_tokens 1024 \
    --model_path "path/to/llama-70b" \
    --batch_size 10000 \
    --input_path "data/replay_buffer.jsonl" \
    --output_path "data/q_target.jsonl" \
    --n_mc_trajs 5
```

### 2. Training Value Function
```bash
# Format data for training
python tictactoe/data_for_train.py \
    --data_path "data/q_target.jsonl" \
    --output_path "data/q_train.jsonl" \
    --method mc_value_q

# Merge training data with history
python tictactoe/merge_train_data.py \
    --data_path "data/q_train.jsonl" \
    --output_path "data/q_train_merged.jsonl" \
    --history 3

# Train value function
torchrun --nproc_per_node=4 tictactoe/train/train_sft.py \
    --model_name_or_path "path/to/llama-8b" \
    --train_dataset_path "data/q_train_merged.jsonl" \
    --max_seq_length=1024 \
    --eval_dataset_path=None \
    --report_to="tensorboard" \
    --learning_rate 1e-5 \
    --torch_dtype "bfloat16" \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "constant" \
    --per_device_train_batch_size 8 \
    --max_seq_length 1024 \
    --gradient_accumulation_steps 2 \
    --attn_implementation "sdpa" \
    --output_dir "path/to/output_value_dir" \
    --logging_steps 1 \
    --num_train_epoch 2 \
    --save_strategy "steps" \
    --save_steps 0.999 \
    --max_steps=-1 \
    --gradient_checkpointing \
    --bf16 \
    --evaluation_strategy "no" \
    --eval_steps 0.1 \
    --per_device_eval_batch_size 8 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer \
    --checkpoint_path "path/to/value_checkpoint"
```

### 3. Policy Improvement
```bash
# Generate improvement targets
python tictactoe/prompt_llm.py \
    --method improve \
    --max_tokens 1024 \
    --model_path "path/to/llama-70b" \
    --input_path "data/replay_buffer.jsonl" \
    --output_path "data/improve_target.jsonl" \
    --value_model_path "path/to/value_model" \
    --num_policy_sample 10 \
    --max_use_action 10

# Format improvement data
python tictactoe/data_for_train.py \
    --data_path "data/improve_target.jsonl" \
    --output_path "data/improve_train.jsonl" \
    --method improve

# Train improved policy
torchrun --nproc_per_node=4 tictactoe/train/train_sft.py \
    --model_name_or_path "path/to/llama-8b" \
    --train_dataset_path "data/improve_train.jsonl" \
    --max_seq_length 1024 \
    --eval_dataset_path "None" \
    --report_to "tensorboard" \
    --learning_rate 1e-5 \
    --torch_dtype "bfloat16" \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "constant" \
    --per_device_train_batch_size 8 \
    --max_seq_length 1024 \
    --gradient_accumulation_steps 2 \
    --attn_implementation sdpa \
    --output_dir="path/to/output_policy_dir" \
    --logging_steps 1 \
    --num_train_epoch 2 \
    --save_strategy "steps" \
    --save_steps 0.999 \
    --max_steps -1 \
    --gradient_checkpointing \
    --bf16 \
    --evaluation_strategy "no" \
    --eval_steps 0.1 \
    --per_device_eval_batch_size 8 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer \
    --checkpoint_path "path/to/policy_checkpoint"
```

## Key Parameters
- `num_rollouts`: Number of game trajectories collected per iteration (default: 512)
- `n_mc_trajs`: Number of Monte Carlo trajectories for value estimation (default: 5)
- `num_policy_sample`: Number of candidate actions sampled from policy (default: 10)
- `max_use_action`: Top-k actions used for policy improvement (default: 10)
- `history`: Number of past experience buffers to merge (default: 3)
- `num_train_epochs`: Number of training epochs per iteration (default: 2)

## Full Pipeline
Run the complete training pipeline with:
```bash
bash tictactoe/scripts/pipeline_nlac.sh \
    [TOP_K_SAMPLE] \
    [NUM_POLICY_SAMPLE] \
    [N_MC_TRAJ] \
    [NUM_ROLLOUTS] \
    [OPPONENT_POLICY_NAME] \
    [NUM_TRAIN_EPOCH] \
    [NUM_HISTORY] \
    [EXP] \
    [START_ITERATION_NUM] \
    [FINAL_ITERATION_NUM]
```

Example:
```bash
bash tictactoe/scripts/pipeline_nlac.sh 10 10 5 512 Random 2 3 20241203 1 31
```

## Evaluation
Monitor training progress and evaluate performance:
```bash
python nlrl/evaluate.py --data_dir "path/to/eval/data"
```

## Key Features
- Stable training through action selection masking
- Experience buffer merging to prevent catastrophic forgetting
- Support for both deterministic and stochastic opponents
- Comprehensive logging and evaluation metrics

## Reproducing Paper Experiments

To reproduce the experiments from our paper, we provide scripts for all methods evaluated:

1. **Natural Language Actor-Critic (Ours)**
```bash
# Full method with action selection mask
bash tictactoe/experiments/run_nlrl.sh

# Ablation: without action selection mask
bash tictactoe/experiments/run_nlrl_wo_action_selection_mask.sh
```

2. **Baseline Methods**
```bash
# PPO fine-tuning with LLaMA-3.1-8B
bash tictactoe/experiments/run_llama3.1_8b_ppo.sh

# Direct prompting baselines
bash tictactoe/experiments/run_gpt4o.sh                      # GPT-4 prompting
bash tictactoe/experiments/run_llama3.1_70b_prompting.sh     # LLaMA-3.1-70B prompting
bash tictactoe/experiments/run_llama3.1_8b_prompting.sh      # LLaMA-3.1-8B prompting
```

Before running experiments:
1. Ensure environment variables for model paths are set:
```bash
export SMALL_LLM_PATH=/path/to/llama-3.1-8b
export BIG_LLM_PATH=/path/to/llama-3.1-70b
```
2. For GPT-4 experiments, set your OpenAI API key:
```bash
export OPENAI_API_KEY=your_key_here
```

Results will be saved in the `results-` directory, organized by experiment name and date. Use the evaluation script to analyze results:
```bash
python nlrl/evaluate.py --data_dir results/experiment_name/data/eval
```
