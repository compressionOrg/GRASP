#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,2
export OMP_NUM_THREADS=4
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_P2P_LEVEL=NVL
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# 确保日志目录存在
mkdir -p ./logs

# 使用单个GPU进行调试，一旦问题解决可以改回多GPU
torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nnodes=1 \
    --node_rank=0 \
    shortgpt.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --num_prune_layers 7 \
    --data_path yahma/alpaca-cleaned \
    --recovery \
    --num_epochs 1 \
    --learning_rate 3e-4 \
    --evaluate \
    --eval_ppl "wikitext2" \
    --batch_size 8 \
    --micro_batch_size 4 \
    --log_dir ./logs