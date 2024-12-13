set -ex

export CUDA_VISIBLE_DEVICES=0,1,2,3
exp_dir="game5x5_td_variation=2_look_ahead=4_45k"
BASE_VALUE_MODEL_PATH="breakthrough/exp/${exp_dir}/models"
BASE_REPLAY_BUFFER_PATH="breakthrough/exp/${exp_dir}/data/replay_buffer"
BASE_V_TARGET_PATH="breakthrough/exp/${exp_dir}/data/value_target"
BASE_EVAL_PATH="breakthrough/exp/${exp_dir}/data/eval"
NUM_TRAIN_EPOCH=2
TD_SUBSAMPLE_NUM=-1 # if -1, we don't subsample
TD_PV_NUM=2
                                

START_ITERATION_NUM=${1:-1}
FINAL_ITERATION_NUM=${2:-50}
echo start: $START_ITERATION_NUM end: $FINAL_ITERATION_NUM

mkdir -p breakthrough/exp/${exp_dir}/data/

SMALL_LLM_NAME=Meta-Llama-3.1-8B-Instruct
BIG_LLM_NAME=Meta-Llama-3.1-70B-Instruct

# Specify evaluation dataset
EVAL_DATA_PATH_TRAIN="" 
EVAL_DATA_PATH_TEST=breakthrough/rollout_collection_5x5_processed/evaluation_3000/win_rate_collection/gt_result.jsonl

# Specify the TD look-ahead data
TD_DATA_BUFFER=breakthrough/rollout_collection_5x5_processed/train_45k/look_ahead/replay_buffer.jsonl

# Load Checkpoints if we are not starting from the first iteration
if [ $START_ITERATION_NUM -gt 1 ]; then
    VALUE_MODEL_OUTPUT_PATH=${BASE_VALUE_MODEL_PATH}_$((START_ITERATION_NUM - 1))
    VALUE_CHECKPOINT=$(find $VALUE_MODEL_OUTPUT_PATH -type d -name 'checkpoint*' | sort | head -n 1)
fi
echo Loading from $VALUE_CHECKPOINT

# Run multiple iterations of TD-training
for i in $(seq $START_ITERATION_NUM $FINAL_ITERATION_NUM)
do
    echo $i
    if [ $i -eq 1 ]; then
        # For the first eval we use results from LLaMA-3.1-70B-Instruct
        VALUE_MODEL_PATH=$BIG_LLM_NAME
        VALUE_TP=4
    else
        VALUE_MODEL_PATH=$VALUE_CHECKPOINT
        VALUE_TP=1
    fi
    # Evaluate trained value function
    EVAL_SAVE_PATH=${BASE_EVAL_PATH}/eval_${i}
    if [ -n "$EVAL_DATA_PATH_TRAIN" ]; then
        python3 breakthrough/evaluate.py --config ./breakthrough/configs/evaluate_config.py \
                                                --config.input_path ${EVAL_DATA_PATH_TRAIN} \
                                                --config.model_path $VALUE_MODEL_PATH \
                                                --config.tensor_parallel_size $VALUE_TP \
                                                --config.output_path ${EVAL_SAVE_PATH}_IND
    fi
    if [ -n "$EVAL_DATA_PATH_TEST" ]; then
        python3 breakthrough/evaluate.py --config ./breakthrough/configs/evaluate_config.py \
                                                --config.input_path ${EVAL_DATA_PATH_TEST} \
                                                --config.model_path $VALUE_MODEL_PATH \
                                                --config.tensor_parallel_size $VALUE_TP \
                                                --config.output_path ${EVAL_SAVE_PATH}_OOD
    fi
    
    V_TARGET_PATH=${BASE_V_TARGET_PATH}/target_${i}
    V_TRAIN_PATH=${BASE_V_TARGET_PATH}/target_${i}/train_data.jsonl
    UPDATED_REPLAY_BUFFER_PATH=${BASE_REPLAY_BUFFER_PATH}/updated_look_ahead_${i}
    # Run prompt_llm to evaluate the final state for state expansion
    # To get the subsequent eval
    python3 breakthrough/prompt_llm.py --config ./breakthrough/configs/prompt_llm_config.py \
                                            --config.method eval_final_state \
                                            --config.input_path $TD_DATA_BUFFER \
                                            --config.model_path $VALUE_MODEL_PATH \
                                            --config.tensor_parallel_size $VALUE_TP \
                                            --config.output_path $UPDATED_REPLAY_BUFFER_PATH \
                                            --config.sub_sample_num $TD_SUBSAMPLE_NUM \
                                            --config.seed $i \
                                            --config.num_pv_use $TD_PV_NUM
                                
    # Prompt LLM for TD estimation
    # 70B model uses tp=4, 8B model uses tp=1
    python3 breakthrough/prompt_llm.py --config ./breakthrough/configs/prompt_llm_config.py \
                                            --config.method td \
                                            --config.input_path ${UPDATED_REPLAY_BUFFER_PATH}/replay_buffer_updated.jsonl \
                                            --config.model_path $BIG_LLM_NAME \
                                            --config.tensor_parallel_size 4 \
                                            --config.output_path $V_TARGET_PATH \
                                            --config.num_pv_use $TD_PV_NUM
                                
    # Run format transfer
    python3 breakthrough/data_for_train.py --data_path ${V_TARGET_PATH}/prompt_result.jsonl --output_path $V_TRAIN_PATH --method mc_value_v
    
    # Train our small LLM using samples from TD estimation
    if [ $i -eq 1 ]; then
        VALUE_TRAIN_START=None
    else
        VALUE_TRAIN_START=$VALUE_CHECKPOINT
    fi
    VALUE_MODEL_OUTPUT_PATH=${BASE_VALUE_MODEL_PATH}_${i}

    torchrun --nproc_per_node=4 --master_port=20002 nlrl/train/train_sft.py \
        --model_name_or_path=$SMALL_LLM_NAME \
        --train_dataset_path=$V_TRAIN_PATH \
        --max_seq_length=1024 \
        --eval_dataset_path=None \
        --report_to="tensorboard" \
        --learning_rate=2e-5 \
        --torch_dtype="bfloat16" \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "constant" \
        --per_device_train_batch_size=4 \
        --gradient_accumulation_steps=8 \
        --attn_implementation=sdpa \
        --output_dir=$VALUE_MODEL_OUTPUT_PATH \
        --logging_steps=1 \
        --num_train_epoch=$NUM_TRAIN_EPOCH \
        --save_strategy "steps" \
        --save_steps 0.999999 \
        --max_steps=-1 \
        --gradient_checkpointing \
        --bf16 \
        --evaluation_strategy "no" \
        --eval_steps 0.1 \
        --per_device_eval_batch_size 8 \
        --fsdp "full_shard auto_wrap" \
        --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer \
        --checkpoint_path $VALUE_TRAIN_START

    VALUE_CHECKPOINT=$(find $VALUE_MODEL_OUTPUT_PATH -type d -name 'checkpoint*' | sort | head -n 1)

    if [ ! -e "$VALUE_CHECKPOINT" ]; then
        echo "Error: VALUE_CHECKPOINT does not exist. Exiting..."
        asdf
    fi
    rm -rf ${VALUE_CHECKPOINT}/rng*
    rm -rf ${VALUE_CHECKPOINT}/train*
   
    # remove old policy and old value model to save memory
    if [ $i -gt 1 ]; then
        rm -rf ${BASE_VALUE_MODEL_PATH}_$(($i-1))/checkpoint*
    fi
done