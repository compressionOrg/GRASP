#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用门控残差连接进行模型剪枝的主脚本。
"""

import os
import torch
import logging
import argparse
import datetime
from typing import Optional, List, Dict, Any, Tuple, Literal
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

from gated_residual import GatedResidualModel, setup_logger
from evaluate_grasp import evaluate_model
from dataset.loader import get_calibration_dataloader

logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="使用门控残差连接进行模型剪枝")
    
    # 模型和数据参数
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="模型名称或路径")
    parser.add_argument("--dataset", type=str, default="wikitext2", help="校准数据集名称")
    parser.add_argument("--seq_len", type=int, default=256, help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--num_samples", type=int, default=50, help="用于校准的样本数量")
    parser.add_argument("--padding", type=str, default="max_length", help="填充策略")
    
    # 剪枝参数
    parser.add_argument("--layers_to_prune", type=str, default=None, help="要剪枝的层索引，用逗号分隔，例如'3,7,11'")
    parser.add_argument("--num_prune_layers", type=int, default=7, help="自动选择时要剪枝的层数")
    parser.add_argument("--auto_select", action="store_true", help="是否自动选择要剪枝的层")
    parser.add_argument("--angular", action="store_true", help="是否使用角度距离计算层重要性")
    
    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=500, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="学习率")
    
    # 其他参数
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="计算设备")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="输出目录")
    parser.add_argument("--log_dir", type=str, default="./logs", help="日志目录")
    parser.add_argument("--remove_layers", action="store_true", default=True, help="是否在保存模型时移除剪枝层")
    
    # 评估参数
    parser.add_argument("--evaluate", action="store_true", help="是否进行评估")
    parser.add_argument("--eval_ppl", type=str, default="wikitext2,ptb,c4", help="用于评估困惑度的数据集，用逗号分隔")
    parser.add_argument("--eval_tasks", type=str, default="boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa", 
                      help="评估任务，用逗号分隔")
    parser.add_argument("--num_fewshot", type=int, default=0, help="少样本示例数量")
    parser.add_argument("--limit", type=int, default=-1, help="限制评估的样本数量，用于调试")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录和日志目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 设置日志
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.log_dir, f"grasp_gated_{timestamp}.log")
    setup_logger(log_file)
    
    # 记录参数
    logger.info(f"参数: {args}")
    
    # 加载模型和分词器
    logger.info(f"加载模型 {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = model.to(args.device)  # 确保模型在正确的设备上
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # 确保 tokenizer 有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 准备校准数据加载器
    calibration_dataloader = get_calibration_dataloader(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        padding=args.padding 
    )
    
    # 创建GatedResidualModel实例
    gated_model = GatedResidualModel(model)
    
    # 确定要剪枝的层
    if args.auto_select:
        logger.info(f"自动选择要剪枝的层，数量: {args.num_prune_layers}")
        layer_importances, layers_to_prune = gated_model.compute_bi(
            num_prune_layers=args.num_prune_layers,
            calibration_dataloader=calibration_dataloader,
            angular=args.angular,
            device=args.device,
            log_file=log_file
        )
        logger.info(f"层重要性: {layer_importances}")
        logger.info(f"自动选择的剪枝层: {layers_to_prune}")
    else:
        if args.layers_to_prune is None:
            raise ValueError("请指定要剪枝的层索引或使用自动选择")
        layers_to_prune = [int(layer) for layer in args.layers_to_prune.split(",")]
        logger.info(f"指定的剪枝层: {layers_to_prune}")
    
    # 识别连续层组
    layer_groups = gated_model.identify_continuous_layers(layers_to_prune)
    logger.info(f"识别到的连续层组: {layer_groups}")
    
    # 应用门控残差连接
    gated_residuals = gated_model.apply_gated_residual(
        layers_to_prune=layers_to_prune,
        device=args.device,
        log_file=log_file,
        remove_layers=True 
    )
    
    # 训练门控残差参数
    gated_model.train_gated_residual(
        calibration_dataloader=calibration_dataloader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        log_file=log_file
    )
    
    # 打印训练后的门控值
    gate_values = gated_model.print_gate_values(log_file=log_file)
    logger.info(f"训练后的门控值: {gate_values}")
    
    logger.info("输出模型结构")
    logger.info(gated_model)

    # 保存模型
    save_dir = os.path.join(args.output_dir, f"gated_model")
    gated_model.save_model(
        save_dir=save_dir, 
        model_name="gated_model"
    )
    logger.info(f"模型已保存到: {save_dir}")
    
    # 评估模型
    if args.evaluate:
        logger.info("开始评估带有门控残差连接的模型")
        # 确保模型在正确的设备上
        gated_model.model = gated_model.model.to(args.device)
        eval_results = evaluate_model(
            model=gated_model.model,
            tokenizer=tokenizer,
            model_name=args.model_name,
            tasks=args.eval_tasks,
            eval_ppl=args.eval_ppl,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            batch_size=args.batch_size,
            device=args.device,
            log_file=log_file
        )
        logger.info(f"评估结果: {eval_results}")
    
    return gated_model

if __name__ == "__main__":
    main()