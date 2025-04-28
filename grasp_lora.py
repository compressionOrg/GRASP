import os
import argparse
import torch
import numpy as np
from setproctitle import setproctitle
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Union, Literal, Optional, List
import logging
from modeling_grasp_lora import GRASPLoRAModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from alpaca_grasp import train
from evaluate_grasp import evaluate_model
from dataset.loader import get_calibration_dataloader

logger = logging.getLogger(__name__)

def setup_logger(log_file=None):
    # 清除现有的处理程序
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    if log_file:
        handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def main(
    model_name_or_path: str,
    calibration_dataloader: DataLoader,
    layers_id: Optional[Union[List[int], int]] = None,
    num_prune_layers: Optional[int] = None,
    lora_rank_ratio: float = 0.1,
    lora_alpha: float = 16,
    lora_dropout: float = 0.05,
    target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
    device: Literal["cuda", "cpu"] = "cuda",
    save_path: Optional[str] = None,
    angular: Optional[bool] = False,
    verbose: Optional[bool] = False,
    recovery: Optional[bool] = True,
    recovery_epochs: int = 1,
    recovery_lr: float = 3e-4,
    log_file: Optional[str] = None,
    train_device: Optional[str] = None,
    use_peft: bool = False,
    continuous_layers_as_group: bool = True,
    *args, **kwargs
):
    # 设置日志记录器
    setup_logger(log_file)

    # 加载模型和分词器
    logger.info(f"加载模型 {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # 创建GRASP-LoRA模型
    grasp_lora_model = GRASPLoRAModel(model=model)
    grasp_lora_model.model.to(device=device)

    # 使用新的compress_model_with_lora方法一次性完成整个压缩流程
    compression_result = grasp_lora_model.compress_model_with_lora(
        calibration_dataloader=calibration_dataloader,
        layers_id=layers_id,
        num_prune_layers=num_prune_layers,
        lora_rank_ratio=lora_rank_ratio,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        device=device,
        angular=angular,
        verbose=verbose,
        recovery=recovery,
        recovery_epochs=recovery_epochs,
        recovery_lr=recovery_lr,
        continuous_layers_as_group=continuous_layers_as_group,
        log_file=log_file
    )
    
    # 保存模型
    logger.info("=======> 保存模型")
    if save_path:
        torch.save(grasp_lora_model, save_path)
    else:
        if not os.path.exists("./checkpoint"):
            os.makedirs("./checkpoint", exist_ok=True)
        model_id: str = grasp_lora_model.model.config._name_or_path
        save_path = os.path.join("./checkpoint", f"{model_id.replace('/', '-')}_lora_replaced.pth")
        torch.save(grasp_lora_model, save_path)
    
    logger.info(f"模型已保存到: {save_path}")
    
    # 如果需要进行额外的恢复训练
    if kwargs.get("additional_recovery", False):
        logger.info("=======> 开始额外的恢复训练")
        grasp_lora_model = train(
            grasp_model=grasp_lora_model,
            tokenizer=tokenizer,
            output_dir=os.path.dirname(save_path),
            log_file=log_file,
            train_device=train_device,
            **kwargs
        )
        # 保存恢复训练后的模型
        recovery_save_path = save_path.replace(".pth", "_recovered.pth")
        torch.save(grasp_lora_model, recovery_save_path)
        logger.info(f"恢复训练后的模型已保存到: {recovery_save_path}")
    
    return grasp_lora_model


def parse_args():
    parser = argparse.ArgumentParser(description="GRASP-LoRA 模型压缩")
    
    # 必需参数
    parser.add_argument("--model_name_or_path", type=str, required=True,
                      help="预训练模型路径或huggingface.co/models的模型标识符")
    parser.add_argument("--dataset_name", type=str, default="wikitext2",
                      help="用于校准的数据集名称")

    # 可选参数
    parser.add_argument("--layers_id", type=int, nargs="+", default=None,
                      help="要压缩的层ID列表")
    parser.add_argument("--num_prune_layers", type=int, default=None,
                      help="如果未指定layers_id，要剪枝的层数")
    parser.add_argument("--lora_rank_ratio", type=float, default=0.1,
                      help="LoRA秩与隐藏层大小的比例")
    parser.add_argument("--lora_alpha", type=float, default=16,
                      help="LoRA alpha参数")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                      help="LoRA dropout率")
    parser.add_argument("--target_modules", type=str, nargs="+", 
                      default=["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
                      help="应用LoRA的目标模块")
    parser.add_argument("--device", type=str, default="cuda",
                      help="运行设备 (cuda 或 cpu)")
    parser.add_argument("--save_path", type=str, default=None,
                      help="保存模型的路径")
    parser.add_argument("--angular", action="store_true",
                      help="是否使用角度相似度计算层相似性")
    parser.add_argument("--verbose", action="store_true",
                      help="是否输出详细日志")
    parser.add_argument("--recovery", action="store_true", default=True,
                      help="是否训练LoRA层以恢复性能")
    parser.add_argument("--recovery_epochs", type=int, default=50,
                      help="LoRA恢复训练的轮数")
    parser.add_argument("--recovery_lr", type=float, default=3e-4,
                      help="LoRA恢复训练的学习率")
    parser.add_argument("--log_file", type=str, default=None,
                      help="日志文件路径")
    parser.add_argument("--train_device", type=str, default=None,
                      help="训练设备，如果与主设备不同")
    parser.add_argument("--use_peft", action="store_true", default=False,
                      help="是否使用PEFT库实现LoRA")
    parser.add_argument("--additional_recovery", action="store_true",
                      help="是否进行额外的恢复训练")
    parser.add_argument("--batch_size", type=int, default=4,
                      help="校准数据的批量大小")
    parser.add_argument("--max_length", type=int, default=512,
                      help="输入序列的最大长度")
    parser.add_argument("--num_samples", type=int, default=100,
                      help="用于校准的样本数量")
    parser.add_argument("--continuous_layers_as_group", action="store_true", default=True,
                      help="是否将连续层作为一个组处理，只使用一个LoRA层")
    
    # 评估参数
    parser.add_argument("--evaluate", action="store_true",
                      help="是否在补偿后进行评估")
    parser.add_argument("--eval_ppl", type=str, default="",
                      help="用于评估困惑度的数据集，用逗号分隔，例如：wikitext2,ptb,c4")
    parser.add_argument("--eval_tasks", type=str, default="boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa",
                      help="评估任务，用逗号分隔")
    parser.add_argument("--num_fewshot", type=int, default=0,
                      help="少样本示例数量")
    parser.add_argument("--limit", type=int, default=-1,
                      help="限制评估的样本数量，用于调试")
    parser.add_argument("--is_peft_model", action="store_true",
                      help="是否为PEFT模型")

    return parser.parse_args()


if __name__ == "__main__":
    setproctitle("GRASP-LoRA")
    args = parse_args()
    
    # 设置日志记录器
    setup_logger(args.log_file)

    # Load tokenizer and create calibration dataloader
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"加载校准数据集: {args.dataset_name}")
    calibration_dataloader = get_calibration_dataloader(
        dataset_name=args.dataset_name,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_samples=args.num_samples
    )
    
    # 运行主函数
    grasp_lora_model = main(
        model_name_or_path=args.model_name_or_path,
        calibration_dataloader=calibration_dataloader,
        layers_id=args.layers_id,
        num_prune_layers=args.num_prune_layers,
        lora_rank_ratio=args.lora_rank_ratio,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        device=args.device,
        save_path=args.save_path,
        angular=args.angular,
        verbose=args.verbose,
        recovery=args.recovery,
        recovery_epochs=args.recovery_epochs,
        recovery_lr=args.recovery_lr,
        log_file=args.log_file,
        train_device=args.train_device,
        use_peft=args.use_peft,
        additional_recovery=args.additional_recovery
    )
    logger.info("输出模型结构")
    logger.info(grasp_lora_model)

    # 评估模型
    if args.evaluate:
        logger.info("开始评估模型性能")
        tokenizer.pad_token = tokenizer.eos_token
        
        results = evaluate_model(
            model=grasp_lora_model.model,
            tokenizer=tokenizer,
            model_name=args.model_name_or_path,
            tasks=args.eval_tasks,
            eval_ppl=args.eval_ppl,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            batch_size=args.batch_size,
            is_peft_model=args.is_peft_model,
            device=args.device,
            log_file=args.log_file
        )
        
        logger.info("评估结果:")
        logger.info(results)
    
    logger.info("GRASP-LoRA 压缩完成!")