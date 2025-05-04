#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0
# Set default values
MODEL_NAME_OR_PATH="meta-llama/Llama-2-7b-hf"
# MODEL_PATH="checkpoint/${MODEL_NAME_OR_PATH//\//-}.pth"
LOG_DIR="logs"
TASKS="boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,mathqa" # mathqa
EVAL_PPL="wikitext2,ptb"
BATCH_SIZE=1
DEVICE="cuda:0"
HF=true

# Create log directory if it doesn't exist
mkdir -p $LOG_DIR

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/evaluation_llama2_7b_dense${TIMESTAMP}.log"

# Run evaluation
python evaluate.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --tasks $TASKS \
    --eval_ppl $EVAL_PPL \
    --batch_size $BATCH_SIZE \
    --device $DEVICE \
    --log_file $LOG_FILE \
    $([ "$HF" = "true" ] && echo "--hf")

echo "Evaluation completed. Log file saved to: $LOG_FILE"
