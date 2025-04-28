#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# 模型参数
MODEL_NAME_OR_PATH="meta-llama/Llama-2-7b-hf"

# 剪枝参数
NUM_PRUNER_LAYERS=7



# 补偿参数
COMPENSATION_DIRECTION="both"  # 补偿方向: up, down, both
SVD_COMPENSATION_RATIO=0.5     # SVD直接补偿比例
LORA_COMPENSATION_RATIO=0.3    # LoRA补偿比例
LORA_RANK_RATIO=0.1            # LoRA秩比例

# 其他参数
DEVICE="cuda"
SAVE_PATH=""  # 留空使用默认路径
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/lora_compensation_${TIMESTAMP}.log"

# 创建日志目录
mkdir -p $LOG_DIR

# 运行低秩适应补偿
python run_lora_compensation.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --num_prune_layers $NUM_PRUNER_LAYERS \
  --compensation_direction $COMPENSATION_DIRECTION \
  --svd_compensation_ratio $SVD_COMPENSATION_RATIO \
  --lora_compensation_ratio $LORA_COMPENSATION_RATIO \
  --lora_rank_ratio $LORA_RANK_RATIO \
  --device $DEVICE \
  --evaluate \
  --save_path "$SAVE_PATH" \
  --log_file $LOG_FILE
  # --eval_ppl "wikitext2" \
echo "低秩适应补偿完成。日志文件保存在: $LOG_FILE"