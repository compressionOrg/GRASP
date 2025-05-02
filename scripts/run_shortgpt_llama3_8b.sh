#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
# Set default values
MODEL_NAME_OR_PATH="meta-llama/Llama-3.1-8B"
# MODEL_PATH="checkpoint/${MODEL_NAME_OR_PATH//\//-}.pth"
NUM_PRUNE_LAYERS=7


TASKS="boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,mathqa" # mathqa
EVAL_PPL="wikitext2,ptb"

LOG_DIR="logs"
# Create log directory if it doesn't exist
mkdir -p $LOG_DIR

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/shortgpt_${MODEL_NAME_OR_PATH//\//-}_${TIMESTAMP}.log"

python shortgpt.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --num_prune_layers ${NUM_PRUNE_LAYERS} \
    --evaluate \
    --tasks $TASKS \
    --eval_ppl $EVAL_PPL \
    --log_file $LOG_FILE \


    # --recovery \