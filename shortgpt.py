import os
import datetime
import argparse
import torch
import numpy as np
from setproctitle import setproctitle
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Union, Literal, Optional, List
import logging
from modeling_grasp import GRASPModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from lora_shortgpt import train_lora
from evaluate_grasp import evaluate_model
from dataset.loader import get_calibration_dataloader


logger = logging.getLogger(__name__)

def setup_logger(log_file=None):
    """设置日志记录器"""
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    if log_file:
        handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def load_model_from_checkpoint(checkpoint_path: str, device: Literal["cuda", "cpu"] = "cuda", local_rank: int = -1):
    """
    从检查点加载模型
    
    Args:
        checkpoint_path: 检查点路径
        device: 运行设备
        local_rank: 分布式训练的本地rank
    
    Returns:
        加载的模型
    """
    checkpoint = torch.load(checkpoint_path)
    
    # 加载基础模型 - 不使用device_map="auto"
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint['model_config']._name_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    # 创建GRASPModel
    short_model = GRASPModel(model=model)
    
    # 加载状态字典
    short_model.load_state_dict(checkpoint['model_state_dict'])
    
    # 设置冗余层
    short_model.redundant_layers = checkpoint['redundant_layers']
    
    # 移动到适当的设备
    if local_rank == -1:
        if device == "cuda":
            short_model = short_model.cuda()
        else:
            short_model = short_model.to(device)
    else:
        short_model = short_model.to(f"cuda:{local_rank}")
    
    return short_model


def main(
    model_name_or_path: str,
    calibration_dataloader: DataLoader,
    layers_id: Optional[Union[List[int], int]] = None,
    num_prune_layers: Optional[int] = None,
    device: Literal["cuda", "cpu"] = "cuda",
    save_path: Optional[str] = None,
    angular: Optional[bool] = False,
    verbose: Optional[bool] = False,
    recovery: Optional[bool] = True,
    log_file: Optional[str] = None,
    train_device: Optional[str] = None,
    local_rank: int = -1,
    *args, **kwargs
):
    """
    直接剪掉冗余层的模型压缩主函数
    
    Args:
        model_name_or_path: 预训练模型路径或标识符
        calibration_dataloader: 校准数据加载器
        layers_id: 要剪枝的层ID列表
        num_prune_layers: 如果layers_id未指定，要剪枝的层数
        device: 运行设备
        save_path: 保存模型的路径
        angular: 是否使用角度距离计算层重要性
        verbose: 是否输出详细信息
        recovery: 是否启用恢复训练
        log_file: 日志文件路径
        train_device: 训练设备
        local_rank: 分布式训练的本地rank
    """
    setup_logger(log_file)

    # 加载模型和分词器 - 不使用device_map="auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    if local_rank == -1:
        # 单GPU训练
        if device == "cuda":
            model = model.cuda()
        else:
            model = model.to(device)
    else:
        # 分布式训练
        model = model.to(f"cuda:{local_rank}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"model:{model_name_or_path}")

    short_model = GRASPModel(model=model)

    # 如果未指定要剪枝的层，则计算层重要性并选择最不重要的层
    if layers_id is None:
        layers_importance, layers_id = short_model.compute_bi(
            num_prune_layers=num_prune_layers, 
            calibration_dataloader=calibration_dataloader, 
            angular=angular, 
            device=device
        )
        logger.info("层重要性测量 (BI):\n%s", layers_importance)

    # 确保layers_id是列表
    if isinstance(layers_id, int):
        layers_id = [layers_id]
    
    # 记录冗余层
    short_model.redundant_layers = layers_id

    # 按降序排序层ID，以避免删除层时的索引错误
    layers_id.sort(reverse=True)

    logger.info("=======> 开始直接剪枝模型")
    logger.info(f"要剪枝的层: {layers_id}")
    
    # 直接移除冗余层
    removed_layers = short_model.remove_layers(
        layers_to_remove=layers_id,
        angular=angular,
        num_prune_layers=num_prune_layers
    )
    
    logger.info(f"成功移除的层: {removed_layers}")
    
    # 保存模型
    logger.info("=======> 剪枝完成!")

    logger.info("输出模型结构")
    logger.info(short_model)

    if save_path:
        # 移除所有hooks
        for module in short_model.modules():
            module._forward_hooks.clear()
            module._forward_pre_hooks.clear()
            module._backward_hooks.clear()
        
        # 保存模型状态字典而不是整个模型
        torch.save({
            'model_state_dict': short_model.state_dict(),
            'model_config': short_model.model.config,
            'redundant_layers': short_model.redundant_layers
        }, save_path)
    else:
        if not os.path.exists("./checkpoint"):
            os.makedirs("./checkpoint", exist_ok=True)
        model_id: str = short_model.model.config._name_or_path
        save_path = os.path.join("./checkpoint", f"{model_id.replace('/', '-')}_shortgpt.pth")
        
        # 移除所有hooks
        for module in short_model.modules():
            module._forward_hooks.clear()
            module._forward_pre_hooks.clear()
            module._backward_hooks.clear()
        
        # 保存模型状态字典而不是整个模型
        torch.save({
            'model_state_dict': short_model.state_dict(),
            'model_config': short_model.model.config,
            'redundant_layers': short_model.redundant_layers
        }, save_path)

    # 如果启用恢复训练
    if recovery:
        logger.info("=======> 开始恢复训练")
        short_model = train_lora(
            short_model=short_model,
            tokenizer=tokenizer,
            output_dir=os.path.dirname(save_path),
            log_file=log_file,
            train_device=train_device,
            local_rank=local_rank,
            **kwargs
        )
        # 保存恢复训练后的模型
        recovered_save_path = save_path.replace(".pth", "_shortgpt_recovered.pth")
        
        # 移除所有hooks
        for module in short_model.modules():
            module._forward_hooks.clear()
            module._forward_pre_hooks.clear()
            module._backward_hooks.clear()
        
        # 保存模型状态字典而不是整个模型
        torch.save({
            'model_state_dict': short_model.state_dict(),
            'model_config': short_model.model.config,
            'redundant_layers': short_model.redundant_layers
        }, recovered_save_path)

    return short_model


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="直接层剪枝模型压缩")
    
    # 分布式训练参数
    parser.add_argument("--local_rank", type=int, default=-1,
                      help="Local rank for distributed training")
    
    # 必需参数
    parser.add_argument("--model_name_or_path", type=str, required=True,
                      help="预训练模型路径或标识符")
    parser.add_argument("--dataset_name", type=str, default="wikitext2",
                      help="用于校准的数据集名称")
    
    # 可选参数
    parser.add_argument("--log_dir", type=str, default="./logs", help="日志目录")
    parser.add_argument("--layers_id", type=int, nargs="+", default=None,
                      help="要剪枝的层ID列表")
    parser.add_argument("--num_prune_layers", type=int, default=None,
                      help="如果未指定layers_id，要剪枝的层数")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda",
                      help="运行模型的设备")
    parser.add_argument("--save_path", type=str, default=None,
                      help="保存压缩模型的路径")
    parser.add_argument("--angular", action="store_true",
                      help="使用角度距离计算层重要性")
    parser.add_argument("--verbose", action="store_true",
                      help="启用详细输出")
    parser.add_argument("--num_samples", type=int, default=1024,
                      help="用于校准的样本数")
    parser.add_argument("--batch_size", type=int, default=1,
                      help="校准的批量大小")
    parser.add_argument("--seq_len", type=int, default=512,
                      help="校准的序列长度")
    parser.add_argument("--padding", type=str, default="max_length",
                      help="校准的填充策略")
    parser.add_argument("--recovery", action="store_true",
                      help="启用恢复训练")
    parser.add_argument("--log_file", type=str, default=None,
                      help="保存程序输出的日志文件路径")
    
    # 恢复训练的参数
    parser.add_argument("--data_path", type=str, default='yahma/alpaca-cleaned',
                      help="训练数据路径")
    parser.add_argument("--train_batch_size", type=int, default=32,
                      help="训练批量大小")
    parser.add_argument("--micro_batch_size", type=int, default=4,
                      help="梯度累积的微批量大小")
    parser.add_argument("--num_epochs", type=int, default=1,
                      help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                      help="训练学习率")
    parser.add_argument("--max_length", type=int, default=256,
                      help="训练的最大序列长度")
    parser.add_argument("--val_set_size", type=int, default=2000,
                      help="验证集大小")
    parser.add_argument("--train_on_inputs", action="store_true",
                      help="在输入上训练")
    parser.add_argument("--add_eos_token", action="store_true",
                      help="添加EOS标记")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                      help="从检查点恢复的路径")
    parser.add_argument("--prompt_template_name", type=str, default="alpaca",
                      help="使用的提示模板名称")
    parser.add_argument("--train_device", type=str, default="",
                      help="训练设备")
    
    # 评估参数
    parser.add_argument("--evaluate", action="store_true",
                      help="启用评估")
    parser.add_argument("--eval_ppl", type=str, default="wikitext2,ptb",
                      help="要评估的数据集")
    parser.add_argument("--tasks", type=str, default="boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,mathqa",
                      help="要评估的任务")
    parser.add_argument("--num_fewshot", type=int, default=0,
                      help="少样本示例数")
    parser.add_argument("--limit", type=int, default=-1,
                      help="限制评估的示例数，用于调试")
    
    return parser.parse_args()


if __name__ == "__main__":
    setproctitle("ShortGPT")
    args = parse_args()
    
    # 设置分布式训练环境
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl')
        logger.info(f"Process rank: {args.local_rank}, device: {torch.cuda.current_device()}, distributed training: True")
    else:
        logger.info(f"Device: {torch.cuda.current_device()}, distributed training: False")
    
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 设置日志
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.log_dir, f"shortgpt_{timestamp}.log")
    setup_logger(log_file)

    # 加载分词器并创建校准数据加载器
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    calibration_dataloader = get_calibration_dataloader(
        dataset_name=args.dataset_name,
        tokenizer=tokenizer,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        padding=args.padding 
    )
    
    # 如果启用恢复训练，准备训练参数
    kwargs = {}
    if args.recovery:
        kwargs = {
            "data_path": args.data_path,
            "batch_size": args.train_batch_size,
            "mirco_batch_size": args.micro_batch_size,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "val_set_size": args.val_set_size,
            "train_on_inputs": args.train_on_inputs,
            "add_eos_token": args.add_eos_token,
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "prompt_template_name": args.prompt_template_name
        }
    
    # 运行主压缩函数
    short_model = main(
        model_name_or_path=args.model_name_or_path,
        calibration_dataloader=calibration_dataloader,
        layers_id=args.layers_id,
        num_prune_layers=args.num_prune_layers,
        device=args.device,
        save_path=args.save_path,
        angular=args.angular,
        verbose=args.verbose,
        recovery=args.recovery,
        log_file=args.log_file,
        train_device=args.train_device,
        local_rank=args.local_rank,
        **kwargs
    )

    # 如果启用评估
    if args.evaluate:
        results = evaluate_model(
            model=short_model.model,
            tokenizer=tokenizer,
            model_name=args.model_name_or_path,
            tasks=args.tasks,
            eval_ppl=args.eval_ppl,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            batch_size=args.batch_size,
            device=args.device,
            log_file=args.log_file
        )
        logger.info("评估结果: %s", results)