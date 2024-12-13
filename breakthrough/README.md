# Breakthrough: Language TD Value Training

Experiments exploring leverage language TD to improve LLM capability analyzing the breakthrough board game.

## Environment
We forked the original [Openspiel](https://github.com/google-deepmind/open_spiel) and modified the environment so the 5x5 breakthrough board can have two lines of pawns.
```bash
git clone https://github.com/waterhorse1/open_spiel
cd open_spiel
pip3 install -e .
```
## Dataset
The training and test set is open-sourced at https://huggingface.co/datasets/Waterhorse/Breakthrough_dataset.

We also open-source our data collection scripts. Specifically, :
* Run rollout with different policy combinations, to collect initial states
```bash
sh collect_initial_state_data.sh
```
* Then build_train_and_test.ipynb is used to post-process the rollout data, which provides scripts to generate train and test initial state datasets. We give an example to generate a 10k deduplicated state dataset.
* We run few-step look-ahead rollouts over the 10k state dataset, to collect our TD dataset:
```bash
sh collect_look_ahead_data.sh
```
* Collect win-rate test data on the test set by running Monte-Carlo estimation:
```bash
sh collect_win_rate_data.sh
```
## Full Pipeline
Run pipeline.sh for the full training and testing iterations.
```bash
sh pipeline.sh
```
Here we explain how the pipeline script works:
* Evaluate state language value function on training and test set, the last line of the JSONL file will show the prediction accuracy.
```bash
# Training set Eval
python3 breakthrough/evaluate.py --config ./breakthrough/configs/evaluate_config.py \
                                                --config.input_path ${EVAL_DATA_PATH_TRAIN} \
                                                --config.model_path $VALUE_MODEL_PATH \
                                                --config.tensor_parallel_size $VALUE_TP \
                                                --config.output_path ${EVAL_SAVE_PATH}_IND
# Testing set Eval
python3 breakthrough/evaluate.py --config ./breakthrough/configs/evaluate_config.py \
                                                --config.input_path ${EVAL_DATA_PATH_TEST} \
                                                --config.model_path $VALUE_MODEL_PATH \
                                                --config.tensor_parallel_size $VALUE_TP \
                                                --config.output_path ${EVAL_SAVE_PATH}_OOD
```
* Run prompt_llm to evaluate the final expansion state to get the eval
```bash
python3 breakthrough/prompt_llm.py --config ./breakthrough/configs/prompt_llm_config.py \
                                            --config.method eval_final_state \
                                            --config.input_path $TD_DATA_BUFFER \
                                            --config.model_path $VALUE_MODEL_PATH \
                                            --config.tensor_parallel_size $VALUE_TP \
                                            --config.output_path $UPDATED_REPLAY_BUFFER_PATH \
                                            --config.sub_sample_num $TD_SUBSAMPLE_NUM \
                                            --config.seed $i \
                                            --config.num_pv_use $TD_PV_NUM
```
* Run prompt_llm to do language TD
```bash
python3 breakthrough/prompt_llm.py --config ./breakthrough/configs/prompt_llm_config.py \
                                            --config.method td \
                                            --config.input_path ${UPDATED_REPLAY_BUFFER_PATH}/replay_buffer_updated.jsonl \
                                            --config.model_path $BIG_LLM_NAME \
                                            --config.tensor_parallel_size 4 \
                                            --config.output_path $V_TARGET_PATH \
                                            --config.num_pv_use $TD_PV_NUM
```
* Run format transfer to convert dataset to trainable format
```bash
python3 breakthrough/data_for_train.py --data_path ${V_TARGET_PATH}/prompt_result.jsonl --output_path $V_TRAIN_PATH --method mc_value_v
```
* SFT over new dataset
```bash
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
```

## Hyperparameters
We set all our hyperparmaters in `configs` directory.
## Miscs
The file `eval_gpt4o_batch` is to get gpt4o results in openai batch mode.

