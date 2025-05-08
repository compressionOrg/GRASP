#!/bin/bash
# conda activate grasp

export CUDA_VISIBLE_DEVICES=2
# Set default values
MODEL_NAME_OR_PATH="mistralai/Mistral-7B-v0.3" # "meta-llama/Llama-3.1-8B"  "meta-llama/Llama-2-7b-hf"
# MODEL_PATH="checkpoint/${MODEL_NAME_OR_PATH//\//-}.pth"
NUM_PRUNE_LAYERS=8
THRESHOLD=0.65
HIGHTEST_LAY=32
LOWEST_LAY=1

TASKS="boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,mathqa" # mathqa
EVAL_PPL="wikitext2,ptb"

LOG_DIR="logs"
# Create log directory if it doesn't exist
mkdir -p $LOG_DIR

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/laco_${MODEL_NAME_OR_PATH//\//-}_${TIMESTAMP}.log"

python laco.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --merge_layers ${NUM_PRUNE_LAYERS} \
    --threshold ${THRESHOLD} \
    --highest_lay ${HIGHTEST_LAY} \
    --lowest_lay ${LOWEST_LAY} \
    --evaluate \
    --tasks $TASKS \
    --eval_ppl $EVAL_PPL \
    --log_file $LOG_FILE \

    # --recovery \