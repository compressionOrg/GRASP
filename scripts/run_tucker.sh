#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
# 创建日志目录
mkdir -p ./logs

# 源参数文件
source scripts/tucker_params.sh

# 转换数组为逗号分隔的字符串
MLP_TARGET_LAYER_TYPES_STR=$(IFS=,; echo "${MLP_TARGET_LAYER_TYPES[*]}")
ATTN_TARGET_LAYER_TYPES_STR=$(IFS=,; echo "${ATTN_TARGET_LAYER_TYPES[*]}")

# 运行命令
python grasp.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name $DATASET_NAME \
    --mlp_target_layer_types $MLP_TARGET_LAYER_TYPES_STR \
    --attn_target_layer_types $ATTN_TARGET_LAYER_TYPES_STR \
    --metric $METRIC \
    --device $DEVICE \
    --num_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    --padding $PADDING \
    --data_path $DATA_PATH \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --max_length $MAX_LENGTH \
    --val_set_size $VAL_SET_SIZE \
    --prompt_template_name $PROMPT_TEMPLATE_NAME \
    --eval_ppl $EVAL_PPL \
    --eval_tasks $EVAL_TASKS \
    --num_fewshot $NUM_FEWSHOT \
    --limit $LIMIT \
    --log_file $LOG_FILE \
    --train_device $TRAIN_DEVICE \
    --decomposition $DECOMPOSITION \
    --tucker_rank_out $TUCKER_RANK_OUT \
    --tucker_rank_in $TUCKER_RANK_IN \
    ${LAYERS_ID:+--layers_id $LAYERS_ID} \
    ${NUM_PRUNE_LAYERS:+--num_prune_layers $NUM_PRUNE_LAYERS} \
    ${COMPRESSION_RATIO:+--compression_ratio $COMPRESSION_RATIO} \
    ${THRESHOLD_RATIO:+--threshold_ratio $THRESHOLD_RATIO} \
    ${SAVE_PATH:+--save_path $SAVE_PATH} \
    ${RESUME_FROM_CHECKPOINT:+--resume_from_checkpoint $RESUME_FROM_CHECKPOINT} \
    $([ "$ANGULAR" = "true" ] && echo "--angular") \
    $([ "$ALLOCATION_AWARE" = "true" ] && echo "--allocation_aware") \
    $([ "$MERGE" = "true" ] && echo "--merge") \
    $([ "$VERBOSE" = "true" ] && echo "--verbose") \
    $([ "$RECOVERY" = "true" ] && echo "--recovery") \
    $([ "$TRAIN_ON_INPUTS" = "true" ] && echo "--train_on_inputs") \
    $([ "$ADD_EOS_TOKEN" = "true" ] && echo "--add_eos_token") \
    $([ "$EVALUATE" = "true" ] && echo "--evaluate")