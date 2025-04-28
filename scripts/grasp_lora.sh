#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1

# 模型参数
MODEL_NAME_OR_PATH="meta-llama/Llama-2-7b-hf"
DATASET_NAME="wikitext2"

# LoRA参数
NUM_PRUNE_LAYERS=7
LORA_RANK_RATIO=0.1
LORA_ALPHA=16
LORA_DROPOUT=0.05

# 恢复训练参数
RECOVERY=true
RECOVERY_EPOCHS=50
RECOVERY_LR=3e-4

# 数据参数 - 减小内存使用
BATCH_SIZE=1        # 从4减小到1
MAX_LENGTH=512      # 从512减小到256
NUM_SAMPLES=100      # 从100减小到50

# 其他参数
DEVICE="cuda"
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/grasp_lora_${TIMESTAMP}.log"

# 创建日志目录
mkdir -p $LOG_DIR

# 运行GRASP-LoRA模型压缩
python grasp_lora.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --dataset_name $DATASET_NAME \
  --num_prune_layers $NUM_PRUNE_LAYERS \
  --lora_rank_ratio $LORA_RANK_RATIO \
  --lora_alpha $LORA_ALPHA \
  --lora_dropout $LORA_DROPOUT \
  --device $DEVICE \
  --batch_size $BATCH_SIZE \
  --max_length $MAX_LENGTH \
  --num_samples $NUM_SAMPLES \
  --log_file $LOG_FILE \
  $([ "$RECOVERY" = "true" ] && echo "--recovery") \
  --recovery_epochs $RECOVERY_EPOCHS \
  --recovery_lr $RECOVERY_LR \
  --evaluate \
  --eval_ppl "wikitext2"\
  
echo "GRASP-LoRA 压缩完成。日志文件保存在: $LOG_FILE"