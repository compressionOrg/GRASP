#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2
# Set default values
MODEL_NAME_OR_PATH="pruned_laco"
# MODEL_PATH="checkpoint/${MODEL_NAME_OR_PATH//\//-}.pth"
LOG_DIR="logs"
TASKS="boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,mathqa"
EVAL_PPL="wikitext2,ptb"
BATCH_SIZE=8
DEVICE="cuda:0"
HF=true

# Create log directory if it doesn't exist
mkdir -p $LOG_DIR

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/evaluation_${TIMESTAMP}.log"

# Run evaluation
python evaluate.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --tasks $TASKS \
    --eval_ppl $EVAL_PPL \
    --batch_size $BATCH_SIZE \
    --device $DEVICE \
    --log_file $LOG_FILE \
    $([ "$HF" = "true" ] && echo "--hf")
    # --model_path $MODEL_PATH \
echo "Evaluation completed. Log file saved to: $LOG_FILE"
