#!/bin/bash

# 模型和数据集配置
MODEL_NAME_OR_PATH="meta-llama/Llama-2-7b-hf"  # 替换为您的模型路径
DATASET_NAME="wikitext2"
DATA_PATH="yahma/alpaca-cleaned"  # 如果需要指定数据路径，请填写

# 压缩配置
DECOMPOSITION="tucker"  # 使用Tucker分解
# COMPRESSION_RATIO=0.5  # 压缩率，可以根据需要调整
THRESHOLD_RATIO=0.5  # 阈值比率，用于自适应秩选择
TUCKER_RANK_OUT=32  # Tucker分解输出维度的秩，可以根据需要调整
TUCKER_RANK_IN=32  # Tucker分解输入维度的秩，可以根据需要调整

# 层配置
# 如果指定了LAYERS_ID，则压缩这些层；否则，使用NUM_PRUNE_LAYERS自动选择层
#LAYERS_ID="0 1 2"  # 要压缩的层ID，空格分隔
NUM_PRUNE_LAYERS=7  # 如果未指定LAYERS_ID，则自动选择这么多层进行压缩

# 目标层类型
MLP_TARGET_LAYER_TYPES="down_proj up_proj gate_proj"
ATTN_TARGET_LAYER_TYPES="q_proj k_proj v_proj o_proj"
# 评估指标
METRIC="taylor"  # 使用taylor或gradient作为重要性度量

# 设备配置
DEVICE="cuda"
TRAIN_DEVICE="0"  # 训练设备，例如"0"表示使用第一个GPU

# 数据加载配置
NUM_SAMPLES=1024
BATCH_SIZE=1
SEQ_LEN=512
PADDING="max_length"

# 训练配置
TRAIN_BATCH_SIZE=4
MICRO_BATCH_SIZE=4
NUM_EPOCHS=1
LEARNING_RATE=3e-4
MAX_LENGTH=256
VAL_SET_SIZE=2000
PROMPT_TEMPLATE_NAME="alpaca"

# 评估配置
EVAL_PPL="wikitext2,ptb"
EVAL_TASKS="boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa"
NUM_FEWSHOT=0
LIMIT=-1

# 其他选项
ANGULAR=false
ALLOCATION_AWARE=false
MERGE=false
VERBOSE=true
RECOVERY=true
TRAIN_ON_INPUTS=false
ADD_EOS_TOKEN=false
EVALUATE=true

# 保存路径
SAVE_PATH="./checkpoint/tucker_compressed_model.pth"
LOG_FILE="./logs/tucker_compression.log"
RESUME_FROM_CHECKPOINT=""  # 如果需要从检查点恢复，请填写路径