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
from dataset.loader import get_calibration_dataloader
from copy import deepcopy
from evaluate_grasp import evaluate_model

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
    laco_model = GRASPModel(model=model)
    
    # 加载状态字典
    laco_model.load_state_dict(checkpoint['model_state_dict'])
    
    # 设置冗余层
    laco_model.redundant_layers = checkpoint['redundant_layers']
    
    # 移动到适当的设备
    if local_rank == -1:
        if device == "cuda":
            laco_model = laco_model.cuda()
        else:
            laco_model = laco_model.to(device)
    else:
        laco_model = laco_model.to(f"cuda:{local_rank}")
    
    return laco_model


def merge_layers_return_model(model, merge_base_lay, merge_layer_num):
    """
    合并模型中的多个层，将后续层的权重合并到基础层中
    
    Args:
        model: 要处理的模型
        merge_base_lay: 基础层索引
        merge_layer_num: 要合并的层数量
    
    Returns:
        合并层后的模型副本
    """
    merge_layer_num = min(merge_layer_num, len(model.model.layers) - merge_base_lay - 1)

    # 将模型移到CPU上进行操作
    original_device = next(model.parameters()).device
    model = model.cpu()
    torch.cuda.empty_cache()  # 清空CUDA缓存
    
    model_copy = deepcopy(model)
    
    for diff_lay in range(merge_base_lay+1, merge_base_lay+1+merge_layer_num):
        # gate_proj
        model_copy.model.layers[merge_base_lay].mlp.gate_proj.weight.data.add_(
            model.model.layers[diff_lay].mlp.gate_proj.weight.data - model_copy.model.layers[merge_base_lay].mlp.gate_proj.weight.data
        )
        # down_proj
        model_copy.model.layers[merge_base_lay].mlp.down_proj.weight.data.add_(
            model.model.layers[diff_lay].mlp.down_proj.weight.data - model_copy.model.layers[merge_base_lay].mlp.down_proj.weight.data
        )
        # up_proj
        model_copy.model.layers[merge_base_lay].mlp.up_proj.weight.data.add_(
            model.model.layers[diff_lay].mlp.up_proj.weight.data - model_copy.model.layers[merge_base_lay].mlp.up_proj.weight.data
        )

        # q_proj
        model_copy.model.layers[merge_base_lay].self_attn.q_proj.weight.data.add_(
            model.model.layers[diff_lay].self_attn.q_proj.weight.data - model_copy.model.layers[merge_base_lay].self_attn.q_proj.weight.data
        )

        # k_proj
        model_copy.model.layers[merge_base_lay].self_attn.k_proj.weight.data.add_(
            model.model.layers[diff_lay].self_attn.k_proj.weight.data - model_copy.model.layers[merge_base_lay].self_attn.k_proj.weight.data
        )

        # v_proj
        model_copy.model.layers[merge_base_lay].self_attn.v_proj.weight.data.add_(
            model.model.layers[diff_lay].self_attn.v_proj.weight.data - model_copy.model.layers[merge_base_lay].self_attn.v_proj.weight.data
        )

        # o_proj
        model_copy.model.layers[merge_base_lay].self_attn.o_proj.weight.data.add_(
            model.model.layers[diff_lay].self_attn.o_proj.weight.data - model_copy.model.layers[merge_base_lay].self_attn.o_proj.weight.data
        )

    for diff_lay in range(merge_base_lay+merge_layer_num, merge_base_lay, -1):
        del(model_copy.model.layers[diff_lay])
    
    # 将模型移回原始设备
    model = model.to(original_device)
    model_copy = model_copy.to(original_device)
    
    return model_copy


def cal_last_hidden_sim(model1, model2, tokenizer, sents):
    """
    计算两个模型在给定句子上的最后隐藏状态的相似度
    
    Args:
        model1: 第一个模型
        model2: 第二个模型
        tokenizer: 分词器
        sents: 用于计算相似度的句子列表
    
    Returns:
        平均相似度
    """
    sim_ls = []
    device = next(model1.parameters()).device  # 获取模型所在的设备
    
    for s in sents:
        encoded_inputs = tokenizer(s, return_tensors='pt')
        # 将输入移动到与模型相同的设备上
        encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}
        
        with torch.no_grad():
            # 禁用缓存进行前向传播
            outputs1 = model1(**encoded_inputs, output_hidden_states=True, use_cache=False)
            hidden_states1 = outputs1.hidden_states[-1]
        with torch.no_grad():
            outputs2 = model2(**encoded_inputs, output_hidden_states=True, use_cache=False)
            hidden_states2 = outputs2.hidden_states[-1]

        similarity = torch.cosine_similarity(
            hidden_states1.squeeze(0).flatten().unsqueeze(0),
            hidden_states2.squeeze(0).flatten().unsqueeze(0)
        )
        sim_ls.append(similarity)

    sim_ls = [i.item() for i in sim_ls]
    logger.info("相似度: %s, 平均相似度: %s", sim_ls, np.mean(sim_ls))
    return np.mean(sim_ls)


def main(
    model_name_or_path: str,
    calibration_dataloader: DataLoader,
    device: Literal["cuda", "cpu"] = "cuda",
    save_path: Optional[str] = None,
    log_file: Optional[str] = None,
    local_rank: int = -1,
    merge_layers: int = 7,
    threshold: float = 0.65,
    interval: int = 2,
    highest_lay: Optional[int] = None,
    lowest_lay: int = 1,
    *args, **kwargs
):
    """
    使用LaCo方法的模型压缩主函数
    
    Args:
        model_name_or_path: 预训练模型路径或标识符
        calibration_dataloader: 校准数据加载器
        layers_id: 要剪枝的层ID列表 (LaCo方法不使用此参数)
        num_prune_layers: 如果layers_id未指定，要剪枝的层数 (LaCo方法不使用此参数)
        device: 运行设备
        save_path: 保存模型的路径
        angular: 是否使用角度距离计算层重要性 (LaCo方法不使用此参数)
        verbose: 是否输出详细信息
        recovery: 是否启用恢复训练
        log_file: 日志文件路径
        train_device: 训练设备
        local_rank: 分布式训练的本地rank
        merge_layers: 每次合并的层数
        threshold: 相似度阈值，高于此值的合并将被接受
        interval: 每次尝试合并的层间隔
        highest_lay: 最高层索引，默认为模型的最后一层
        lowest_lay: 最低层索引，默认为1
    """
    setup_logger(log_file)

    # 加载模型和分词器
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

    logger.info(f"模型: {model_name_or_path}")
    
    # 准备校准数据
    sents = []
    # 从校准数据加载器中提取句子
    for batch in calibration_dataloader:
        if isinstance(batch, dict) and "input_ids" in batch:
            for input_id in batch["input_ids"]:
                sent = tokenizer.decode(input_id)
                if sent and len(sent.strip()) > 0:
                    sents.append(sent)
        if len(sents) >= 5:  # 只使用前5个句子进行校准
            break
    
    if not sents:
        # 如果没有从数据加载器中提取到句子，使用默认的英文句子
        logger.info("未从校准数据中提取到句子，使用默认句子")
        sents = [
            'Mouron () is a commune in the Arde',
            'The 81st Mechanised Brigade () is a mechanised brigade of the Romanian Land Force',
            'There are 18 National Natural Landmarks in the U.S. state of Washington, out of nearly',
            'Torreorgaz is a municipality in the',
            'Copa Libertadores 1973 was won by defending champions Independiente of A'
        ]
    
    logger.info("使用以下句子进行校准:")
    for i, sent in enumerate(sents):
        logger.info(f"句子 {i+1}: {sent}")
    
    # 创建模型副本用于压缩 - 全部在CPU上处理
    logger.info("正在创建模型副本...")
    original_device = next(model.parameters()).device
    model = model.cpu()
    
    # 确保所有缓存和梯度被清除
    torch.cuda.empty_cache()
    
    # 使用安全的深度复制方法
    model_to_compress = deepcopy(model)
    
    # 将模型移回设备
    model = model.to(original_device)
    model_to_compress = model_to_compress.to(original_device)
    
    # 设置层参数
    if highest_lay is None:
        highest_lay = len(model_to_compress.model.layers) - 1
    
    lay = highest_lay - merge_layers
    last_merge_flag = False
    
    import gc  # 导入垃圾回收模块
    
    logger.info("=======> 开始使用LaCo方法压缩模型")
    logger.info(f"初始层数: {len(model_to_compress.model.layers)}")
    logger.info(f"合并层数: {merge_layers}")
    logger.info(f"相似度阈值: {threshold}")
    logger.info(f"层间隔: {interval}")
    logger.info(f"最高层: {highest_lay}")
    logger.info(f"最低层: {lowest_lay}")
    
    # 开始LaCo压缩过程
    while lay >= lowest_lay:
        logger.info(f"当前处理层: {lay}")
        logger.info(f"当前模型层数: {len(model_to_compress.model.layers)}")
        
        # 尝试合并层
        tmp_merged_model = merge_layers_return_model(model_to_compress, lay, merge_layers-1)
        
        # 计算合并后的相似度
        sim_value = cal_last_hidden_sim(model, tmp_merged_model, tokenizer, sents)
        
        if sim_value > threshold:
            # 如果相似度高于阈值，接受合并
            logger.info(f"接受合并: 层 {lay} 到 {lay+merge_layers-1}, 相似度: {sim_value}")
            model_to_compress = tmp_merged_model
            lay -= interval
            
            # 确保lay不超出模型层数范围
            if lay >= len(model_to_compress.model.layers):
                lay = len(model_to_compress.model.layers) - 1 - merge_layers
        else:
            # 如果相似度低于阈值，拒绝合并
            logger.info(f"拒绝合并: 层 {lay}, 相似度: {sim_value} < 阈值 {threshold}")
            lay -= 1
        
        # 清理未使用的对象和缓存
        del tmp_merged_model
        torch.cuda.empty_cache()
        gc.collect()
    
    # 更新模型配置中的层数
    model_to_compress.config.num_hidden_layers = len(model_to_compress.model.layers)
    
    # 创建GRASPModel包装
    laco_model = GRASPModel(model=model_to_compress)
    
    # 记录被移除的层
    removed_layers = []
    for i in range(len(model.model.layers)):
        if i >= len(model_to_compress.model.layers):
            removed_layers.append(i)
    
    laco_model.redundant_layers = removed_layers
    
    logger.info("=======> 压缩完成!")
    logger.info(f"原始层数: {len(model.model.layers)}")
    logger.info(f"压缩后层数: {len(model_to_compress.model.layers)}")
    logger.info(f"移除的层: {removed_layers}")
    
    # 计算剪枝后的模型大小与原模型大小的比例
    original_params = sum(p.numel() for p in model.parameters())
    compressed_params = sum(p.numel() for p in model_to_compress.parameters())
    compression_ratio = compressed_params / original_params
    
    logger.info(f"原始模型参数量: {original_params:,}")
    logger.info(f"压缩后模型参数量: {compressed_params:,}")
    logger.info(f"压缩比例: {compression_ratio:.4f} ({compression_ratio*100:.2f}%)")
    logger.info(f"参数量减少: {(1-compression_ratio)*100:.2f}%")
    
    # 保存模型
    if save_path:
        # 保存模型
        torch.save(laco_model, save_path)
    else:
        if not os.path.exists("./checkpoint"):
            os.makedirs("./checkpoint", exist_ok=True)
        model_id: str = laco_model.model.config._name_or_path
        save_path = os.path.join("./checkpoint", f"{model_id.replace('/', '-')}_laco.pth")
        
        # 保存模型
        torch.save(laco_model, save_path)

    return laco_model


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="LaCo模型压缩")
    
    # 分布式训练参数
    parser.add_argument("--local_rank", type=int, default=-1,
                      help="Local rank for distributed training")
    
    # 必需参数
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf",
                      help="预训练模型路径或标识符")
    parser.add_argument("--dataset_name", type=str, default="wikitext2",
                      help="用于校准的数据集名称")
    
    # 可选参数
    parser.add_argument("--log_dir", type=str, default="./logs", help="日志目录")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda",
                      help="运行模型的设备")
    parser.add_argument("--save_path", type=str, default=None,
                      help="保存压缩模型的路径")
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
    
    # LaCo特定参数
    parser.add_argument("--merge_layers", type=int, default=7,
                      help="每次合并的层数")
    parser.add_argument("--threshold", type=float, default=0.65,
                      help="相似度阈值，高于此值的合并将被接受")
    parser.add_argument("--interval", type=int, default=2,
                      help="每次尝试合并的层间隔")
    parser.add_argument("--highest_lay", type=int, default=None,
                      help="最高层索引，默认为模型的最后一层")
    parser.add_argument("--lowest_lay", type=int, default=1,
                      help="最低层索引，默认为1")
    
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
    setproctitle("LaCo")
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # 设置日志目录
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    
    # 设置日志文件
    if args.log_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.log_file = os.path.join(args.log_dir, f"laco_{timestamp}.log")
    
    # 获取校准数据加载器
    calibration_dataloader = get_calibration_dataloader(
        dataset_name=args.dataset_name,
        tokenizer=tokenizer,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        padding=args.padding,
        batch_size=args.batch_size
    )
    
    # 运行主函数
    laco_model = main(
        model_name_or_path=args.model_name_or_path,
        calibration_dataloader=calibration_dataloader,
        device=args.device,
        save_path=args.save_path,
        verbose=args.verbose,
        log_file=args.log_file,
        local_rank=args.local_rank,
        merge_layers=args.merge_layers,
        threshold=args.threshold,
        interval=args.interval,
        highest_lay=args.highest_lay,
        lowest_lay=args.lowest_lay,
    )
    # 计算剪枝后的模型大小与原模型大小的比例


    # 如果启用评估
    if args.evaluate:
        results = evaluate_model(
            model=laco_model.model,
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