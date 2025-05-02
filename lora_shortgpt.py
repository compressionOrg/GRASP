# SET visible device
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List, Optional, Literal, Dict, Any, Union
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset
from prompter import Prompter
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from modeling_grasp import GRASPModel

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


def train_lora(
    # 模型参数
    short_model: torch.nn.Module, # GRASPModel
    tokenizer,
    data_path: Optional[str] = 'yahma/alpaca-cleaned',
    output_dir: Optional[str] = './checkpoint',
    # LoRA参数
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[List[str]] = None,
    # 训练超参数
    batch_size: int = 8,
    micro_batch_size: int = 4,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    max_length: int = 128,
    val_set_size: int = 2000,
    train_on_inputs: bool = True, # 如果为False，在损失中掩盖输入
    add_eos_token: bool = False,
    resume_from_checkpoint: Optional[str] = None,
    prompt_template_name: str = "alpaca",
    train_device: Optional[str] = None, # 默认在所有GPU上，设置CUDA_VISIBLE_DEVICES指定使用哪个GPU
    log_file: Optional[str] = None,
    local_rank: int = -1,
    **kwargs
):
    """
    使用LoRA对ShortGPT模型进行微调
    
    Args:
        short_model: ShortGPT模型实例
        tokenizer: 分词器
        data_path: 训练数据路径
        output_dir: 输出目录
        lora_r: LoRA的秩
        lora_alpha: LoRA的缩放参数
        lora_dropout: LoRA的dropout率
        lora_target_modules: 要应用LoRA的目标模块列表
        batch_size: 训练批量大小
        micro_batch_size: 微批量大小（用于梯度累积）
        num_epochs: 训练轮数
        learning_rate: 学习率
        max_length: 最大序列长度
        val_set_size: 验证集大小
        train_on_inputs: 是否在输入上训练
        add_eos_token: 是否添加EOS标记
        resume_from_checkpoint: 从检查点恢复训练
        prompt_template_name: 提示模板名称
        train_device: 训练设备
        log_file: 日志文件路径
        local_rank: 分布式训练的本地rank
    
    Returns:
        微调后的模型
    """
    setup_logger(log_file)
    logger.info(
        f"使用LoRA微调ShortGPT模型，参数如下:\n"
        f"基础模型: {short_model}\n"
        f"数据路径: {data_path}\n"
        f"输出目录: {output_dir}\n"
        f"LoRA秩(r): {lora_r}\n"
        f"LoRA缩放(alpha): {lora_alpha}\n"
        f"LoRA dropout: {lora_dropout}\n"
        f"批量大小: {batch_size}\n"
        f"训练轮数: {num_epochs}\n"
        f"学习率: {learning_rate}\n"
        f"验证集大小: {val_set_size}\n"
        f"在输入上训练: {train_on_inputs}\n"
        f"添加EOS标记: {add_eos_token}\n"
        f"从检查点恢复: {resume_from_checkpoint or False}\n"
        f"提示模板: {prompt_template_name}\n"
    )

    if train_device:
        # os.environ["CUDA_VISIBLE_DEVICES"] = train_device
        pass
    
    gradient_accumulation_steps = batch_size // micro_batch_size

    # 如果未指定LoRA目标模块，使用默认值
    if lora_target_modules is None:
        lora_target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    
    # 准备模型
    model = short_model.model
    
    # 首先确保所有参数不需要梯度
    for param in model.parameters():
        param.requires_grad = False
    
    # 准备模型参数以进行训练
    short_model.prepare_for_training(lora_target_modules=lora_target_modules)
    
    # 打印模型的所有模块名称
    logger.info("Model module names:")
    found_targets = False
    for name, module in model.named_modules():
        logger.info(f"Module name: {name}")
        # 检查是否包含目标模块
        for target in lora_target_modules:
            if target in name:
                found_targets = True
    
    if not found_targets:
        logger.warning(f"目标模块 {lora_target_modules} 在模型中未找到。尝试检查模型实际的模块名称。")
        # 尝试确定正确的目标模块
        suggested_targets = []
        logger.info("检查模型模块名称以找到合适的目标模块...")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "lm_head" not in name and any(keyword in name for keyword in ["attention", "mlp"]):
                parts = name.split(".")
                if len(parts) > 0:
                    last_part = parts[-1]
                    if last_part not in suggested_targets:
                        suggested_targets.append(last_part)
                        
        if suggested_targets:
            logger.info(f"建议的目标模块: {suggested_targets}")
            lora_target_modules = suggested_targets
            logger.info(f"自动更新目标模块为: {lora_target_modules}")
    
    # 配置LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules
    )
    
    # 确保模型已经移动到正确的设备
    if local_rank != -1:
        model = model.to(f"cuda:{local_rank}")
    
    # 应用LoRA配置
    model = get_peft_model(model, peft_config)
    
    # 确保LoRA参数需要梯度
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
    
    model.print_trainable_parameters()  # 打印可训练参数信息

    # 分词器初始化和训练输入的分词
    tokenizer.pad_token_id = 0  # 我们希望这与eos标记不同
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None
        )
        if (result["input_ids"][-1] != tokenizer.eos_token_id 
            and len(result["input_ids"]) < max_length
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        
        result["labels"] = result["input_ids"].copy()

        return result
    
    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            instruction=data_point["instruction"],
            input=data_point["input"],
            label=data_point["output"]
        )
        tokenized_full_prompt = tokenize(full_prompt)  # 包括用户提示和回答的完整提示的token id
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                instruction=data_point["instruction"],
                input=data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1
            
            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]

        return tokenized_full_prompt
    
    # 加载数据集
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    # 从检查点恢复
    if resume_from_checkpoint:
        # 检查可用权重并加载它们
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # 完整检查点
        if os.path.exists(checkpoint_name):
            logger.info(f"从 {checkpoint_name} 重新开始")
            model = torch.load(checkpoint_name)
        else:
            logger.info(f"未找到检查点 {checkpoint_name}")

    # 准备提示器和数据集
    prompter = Prompter(template_name=prompt_template_name)
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None
    
    # 配置训练器
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=TrainingArguments(
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False,
            group_by_length=False,
            # 添加多GPU训练相关配置
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            ddp_backend="nccl",
            deepspeed=None,
            fp16_backend="auto",
            gradient_checkpointing=True,
            # 分布式训练配置
            local_rank=local_rank,
            # 禁用自动设备映射
            no_cuda=False,
            use_mps_device=False,
            use_cpu=False
        ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    
    # 开始训练
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # 保存模型
    model.save_pretrained(output_dir)
    
    # 更新short_model中的模型
    short_model.model = model
    
    return short_model


def parse_args():
    """解析命令行参数"""
    import argparse
    parser = argparse.ArgumentParser(description="ShortGPT的LoRA微调")
    
    # 必需参数
    parser.add_argument("--model_name_or_path", type=str, required=True,
                      help="预训练模型路径或标识符")
    
    # 可选参数
    parser.add_argument("--log_dir", type=str, default="./logs", help="日志目录")
    parser.add_argument("--save_path", type=str, default=None,
                      help="保存微调模型的路径")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda",
                      help="运行模型的设备")
    parser.add_argument("--log_file", type=str, default=None,
                      help="保存程序输出的日志文件路径")
    
    # LoRA参数
    parser.add_argument("--lora_r", type=int, default=8,
                      help="LoRA的秩")
    parser.add_argument("--lora_alpha", type=int, default=16,
                      help="LoRA的缩放参数")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                      help="LoRA的dropout率")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", 
                      default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                      help="要应用LoRA的目标模块列表")
    
    # 训练参数
    parser.add_argument("--data_path", type=str, default='yahma/alpaca-cleaned',
                      help="训练数据路径")
    parser.add_argument("--output_dir", type=str, default='./checkpoint',
                      help="输出目录")
    parser.add_argument("--batch_size", type=int, default=32,
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
    parser.add_argument("--train_device", type=str, default="0",
                      help="训练设备")
    
    # 评估参数
    parser.add_argument("--evaluate", action="store_true",
                      help="启用评估")
    parser.add_argument("--eval_ppl", type=str, default="wikitext2",
                      help="要评估的数据集")
    parser.add_argument("--eval_tasks", type=str, default="boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa",
                      help="要评估的任务")
    parser.add_argument("--num_fewshot", type=int, default=0,
                      help="少样本示例数")
    parser.add_argument("--limit", type=int, default=-1,
                      help="限制评估的示例数，用于调试")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # 设置日志目录
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    
    # 生成日志文件路径
    if args.log_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = os.path.join(args.log_dir, f"lora_shortgpt_{timestamp}.log")
    
    # 设置日志
    setup_logger(args.log_file)
    
    # 加载模型和分词器
    logger.info(f"加载模型: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 创建GRASPModel
    short_model = GRASPModel(model=model)
    
    # 微调模型
    logger.info("开始LoRA微调")
    short_model = train_lora(
        short_model=short_model,
        tokenizer=tokenizer,
        data_path=args.data_path,
        output_dir=args.output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        val_set_size=args.val_set_size,
        train_on_inputs=args.train_on_inputs,
        add_eos_token=args.add_eos_token,
        resume_from_checkpoint=args.resume_from_checkpoint,
        prompt_template_name=args.prompt_template_name,
        train_device=args.train_device,
        log_file=args.log_file,
        local_rank=int(os.environ.get("LOCAL_RANK", -1))
    )
    
    # 保存模型
    if args.save_path:
        torch.save(short_model, args.save_path)
    else:
        if not os.path.exists("./checkpoint"):
            os.makedirs("./checkpoint", exist_ok=True)
        model_id = short_model.model.config._name_or_path
        save_path = os.path.join("./checkpoint", f"{model_id.replace('/', '-')}_lora.pth")
        torch.save(short_model, save_path)
        logger.info(f"模型已保存到: {save_path}")
    
    # 评估模型
    if args.evaluate:
        from evaluate_grasp import evaluate_model
        
        logger.info("开始评估模型")
        results = evaluate_model(
            model=short_model.model,
            tokenizer=tokenizer,
            model_name=args.model_name_or_path,
            tasks=args.eval_tasks,
            eval_ppl=args.eval_ppl,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            batch_size=1,
            is_peft_model=True,
            device=args.device,
            log_file=args.log_file
        )
        
        logger.info(f"评估结果: {results}")