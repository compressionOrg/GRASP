import os
import argparse
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from tools.lora_compensation import hybrid_compensation
from modeling_grasp import GRASPModel
from evaluate_grasp import evaluate_model

logger = logging.getLogger(__name__)

def setup_logger(log_file=None):
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    if log_file:
        handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def main():
    parser = argparse.ArgumentParser(description="低秩适应补偿 (SVD+LoRA)")
    
    # 模型参数
    parser.add_argument("--model_name_or_path", type=str, required=True,
                      help="预训练模型路径或huggingface.co/models的模型标识符")
    
    # 剪枝参数
    parser.add_argument("--layers_to_prune", type=int, nargs="+", 
                      help="要剪枝的层索引列表")
    parser.add_argument("--num_prune_layers", type=int, default=None,
                      help="如果未指定layers_to_prune，要剪枝的层数量")
    parser.add_argument("--angular", action="store_true",
                      help="使用角度距离计算层重要性")
    
    # 补偿参数
    parser.add_argument("--compensation_direction", type=str, choices=["up", "down", "both"], default="both",
                      help="补偿方向: up(上层), down(下层), both(两者)")
    parser.add_argument("--svd_compensation_ratio", type=float, default=0.5,
                      help="SVD直接补偿比例")
    parser.add_argument("--lora_compensation_ratio", type=float, default=0.3,
                      help="LoRA补偿比例")
    parser.add_argument("--lora_rank_ratio", type=float, default=0.1,
                      help="LoRA秩比例（相对于输入维度）")
    
    # 其他参数
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda",
                      help="计算设备")
    parser.add_argument("--save_path", type=str, default=None,
                      help="保存模型的路径")
    parser.add_argument("--log_file", type=str, default=None,
                      help="日志文件路径")
    
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
    parser.add_argument("--batch_size", type=int, default=1,
                      help="评估的批处理大小")
    parser.add_argument("--is_peft_model", action="store_true",
                      help="是否为PEFT模型")
    
    # 添加恢复训练参数
    parser.add_argument("--recovery", action="store_true",
                      help="是否在补偿后进行恢复训练")
    parser.add_argument("--data_path", type=str, default='yahma/alpaca-cleaned',
                      help="训练数据路径")
    parser.add_argument("--train_batch_size", type=int, default=32,
                      help="训练批处理大小")
    parser.add_argument("--micro_batch_size", type=int, default=4,
                      help="微批处理大小，用于梯度累积")
    parser.add_argument("--num_epochs", type=int, default=1,
                      help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                      help="学习率")
    parser.add_argument("--continuous_layers_as_group", action="store_true",
                      help="将连续的层作为一个整体进行处理和补偿")
    
    args = parser.parse_args()
    setup_logger(args.log_file)
    
    # 确保layers_to_prune是列表
    if isinstance(args.layers_to_prune, int):
        args.layers_to_prune = [args.layers_to_prune]
    
    # 加载模型
    logger.info(f"加载模型 {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # 创建GRASP模型
    grasp_model = GRASPModel(model=model)
    grasp_model.model.to(device=args.device)
    
    # 如果未指定要剪枝的层，使用BI计算层重要性
    if args.layers_to_prune is None and args.num_prune_layers is not None:
        from dataset.loader import get_calibration_dataloader
        logger.info(f"未指定要剪枝的层，使用BI计算层重要性")
        calibration_dataloader = get_calibration_dataloader(
            dataset_name="wikitext2",
            tokenizer=tokenizer,
            num_samples=1024,
            batch_size=1,
            seq_len=512,
            padding="max_length"
        )
        layers_importance, args.layers_to_prune = grasp_model.compute_bi(
            num_prune_layers=args.num_prune_layers, 
            calibration_dataloader=calibration_dataloader, 
            angular=args.angular, 
            device=args.device
        )
        logger.info(f"BI计算的层重要性：{layers_importance}")
        logger.info(f"自动选择的剪枝层：{args.layers_to_prune}")
    
    # 在应用混合补偿前处理连续层
    if args.continuous_layers_as_group and args.layers_to_prune and len(args.layers_to_prune) > 1:
        from tools.layer_compensation import identify_continuous_layers
        # 先按降序排序
        args.layers_to_prune.sort(reverse=True)
        layer_groups = identify_continuous_layers(args.layers_to_prune)
        if len(layer_groups) < len(args.layers_to_prune):
            logger.info(f"检测到连续层组: {layer_groups}")
    
    # 应用混合补偿
    logger.info(f"开始对层 {args.layers_to_prune} 进行混合补偿")
    grasp_model.model, removed_layers, lora_layers = hybrid_compensation(
        model=grasp_model.model,
        layers_to_prune=args.layers_to_prune,
        compensation_direction=args.compensation_direction,
        svd_compensation_ratio=args.svd_compensation_ratio,
        lora_compensation_ratio=args.lora_compensation_ratio,
        lora_rank_ratio=args.lora_rank_ratio,
        device=args.device,
        log_file=args.log_file
    )
    
    # 保存模型
    if args.save_path:
        save_path = args.save_path
    else:
        if not os.path.exists("./checkpoint"):
            os.makedirs("./checkpoint", exist_ok=True)
        model_id = grasp_model.model.config._name_or_path
        save_path = os.path.join("./checkpoint", f"{model_id.replace('/', '-')}_lora_compensated.pth")
    
    logger.info(f"保存模型到 {save_path}")
    torch.save(grasp_model, save_path)
    
    # 打印补偿信息
    logger.info(f"已移除层: {removed_layers}")
    logger.info(f"已添加LoRA的层: {lora_layers}")
    
    # 评估模型
    if args.evaluate:
        logger.info("开始评估模型性能")
        tokenizer.pad_token = tokenizer.eos_token
        
        results = evaluate_model(
            model=grasp_model.model,
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
    
    # 在评估后添加恢复训练
    if args.recovery:
        from alpaca_grasp import train
        logger.info("开始对补偿后的模型进行恢复训练")
        grasp_model = train(
            grasp_model=grasp_model,
            tokenizer=tokenizer,
            output_dir=os.path.dirname(save_path),
            batch_size=args.train_batch_size,
            mirco_batch_size=args.micro_batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            data_path=args.data_path,
            log_file=args.log_file
        )
        # 保存恢复训练后的模型
        recovery_save_path = save_path.replace(".pth", "_recovered.pth")
        logger.info(f"保存恢复训练后的模型到 {recovery_save_path}")
        torch.save(grasp_model, recovery_save_path)
    
    return grasp_model

if __name__ == "__main__":
    main()
    
