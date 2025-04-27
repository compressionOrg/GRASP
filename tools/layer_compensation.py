import torch
import torch.nn as nn
import logging
import numpy as np
from typing import Literal, Optional, Union, List, Tuple

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

def compensate_pruned_layer(
    model: nn.Module,
    layer_to_prune: int,
    compensation_direction: Literal["up", "down", "both"] = "both",
    compensation_ratio: float = 0.5,
    device: Literal["cuda", "cpu"] = "cuda",
    log_file: Optional[str] = None
) -> nn.Module:
    """
    对要剪枝的层进行SVD分解，并将其信息补偿到相邻层
    
    Args:
        model: 模型
        layer_to_prune: 要剪枝的层索引
        compensation_direction: 补偿方向，可以是"up"（上层）、"down"（下层）或"both"（两者都补偿）
        compensation_ratio: 补偿比例，控制有多少信息被补偿
        device: 计算设备
        log_file: 日志文件路径
        
    Returns:
        补偿后的模型
    """
    setup_logger(log_file)
    logger.info(f"开始对第{layer_to_prune}层进行SVD补偿")
    
    # 确保层索引有效
    num_layers = len(model.model.layers)
    if layer_to_prune < 0 or layer_to_prune >= num_layers:
        raise ValueError(f"层索引{layer_to_prune}无效，模型共有{num_layers}层")
    
    # 确定补偿的目标层
    target_layers = []
    if compensation_direction == "up" and layer_to_prune > 0:
        target_layers.append(layer_to_prune - 1)
    elif compensation_direction == "down" and layer_to_prune < num_layers - 1:
        target_layers.append(layer_to_prune + 1)
    elif compensation_direction == "both":
        if layer_to_prune > 0:
            target_layers.append(layer_to_prune - 1)
        if layer_to_prune < num_layers - 1:
            target_layers.append(layer_to_prune + 1)
    
    if not target_layers:
        logger.warning(f"没有找到可以补偿的目标层，跳过补偿")
        return model
    
    # 获取要剪枝的层
    layer_to_prune_module = model.model.layers[layer_to_prune]
    
    # 对要剪枝层的各个组件进行补偿
    compensate_attention_block(model, layer_to_prune, target_layers, compensation_ratio, device)
    compensate_mlp_block(model, layer_to_prune, target_layers, compensation_ratio, device)
    
    logger.info(f"完成对第{layer_to_prune}层的SVD补偿")
    return model

def compensate_attention_block(
    model: nn.Module,
    layer_to_prune: int,
    target_layers: List[int],
    compensation_ratio: float,
    device: Literal["cuda", "cpu"]
):
    """补偿注意力模块"""
    attention_components = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    for component in attention_components:
        # 获取要剪枝层的组件
        source_path = f"model.layers.{layer_to_prune}.self_attn.{component}"
        try:
            source_module = model.get_submodule(source_path)
        except AttributeError:
            logger.warning(f"找不到组件 {source_path}，跳过")
            continue
        
        if not isinstance(source_module, nn.Linear):
            logger.warning(f"组件 {source_path} 不是线性层，跳过")
            continue
        
        # 对每个目标层进行补偿
        for target_layer in target_layers:
            target_path = f"model.layers.{target_layer}.self_attn.{component}"
            try:
                target_module = model.get_submodule(target_path)
            except AttributeError:
                logger.warning(f"找不到目标组件 {target_path}，跳过")
                continue
            
            if not isinstance(target_module, nn.Linear):
                logger.warning(f"目标组件 {target_path} 不是线性层，跳过")
                continue
            
            # 执行SVD补偿
            compensate_linear_layer(source_module, target_module, compensation_ratio, device)

def compensate_mlp_block(
    model: nn.Module,
    layer_to_prune: int,
    target_layers: List[int],
    compensation_ratio: float,
    device: Literal["cuda", "cpu"]
):
    """补偿MLP模块"""
    mlp_components = ["down_proj", "up_proj", "gate_proj"]
    
    for component in mlp_components:
        # 获取要剪枝层的组件
        source_path = f"model.layers.{layer_to_prune}.mlp.{component}"
        try:
            source_module = model.get_submodule(source_path)
        except AttributeError:
            logger.warning(f"找不到组件 {source_path}，跳过")
            continue
        
        if not isinstance(source_module, nn.Linear):
            logger.warning(f"组件 {source_path} 不是线性层，跳过")
            continue
        
        # 对每个目标层进行补偿
        for target_layer in target_layers:
            target_path = f"model.layers.{target_layer}.mlp.{component}"
            try:
                target_module = model.get_submodule(target_path)
            except AttributeError:
                logger.warning(f"找不到目标组件 {target_path}，跳过")
                continue
            
            if not isinstance(target_module, nn.Linear):
                logger.warning(f"目标组件 {target_path} 不是线性层，跳过")
                continue
            
            # 执行SVD补偿
            compensate_linear_layer(source_module, target_module, compensation_ratio, device)

def compensate_linear_layer(
    source_layer: nn.Linear,
    target_layer: nn.Linear,
    compensation_ratio: float,
    device: Literal["cuda", "cpu"]
):
    """
    将源线性层的信息通过SVD补偿到目标线性层
    
    Args:
        source_layer: 源线性层（要剪枝的层）
        target_layer: 目标线性层（要补偿的层）
        compensation_ratio: 补偿比例
        device: 计算设备
    """
    # 检查维度是否匹配
    if source_layer.weight.shape != target_layer.weight.shape:
        logger.warning(f"源层维度 {source_layer.weight.shape} 与目标层维度 {target_layer.weight.shape} 不匹配，跳过补偿")
        return
    
    # 对源层权重进行SVD分解
    W_source = source_layer.weight.data.to(device)
    U, S, Vh = torch.linalg.svd(W_source, full_matrices=False)
    
    # 计算补偿权重
    # 我们使用SVD重构源层权重，并按比例添加到目标层
    W_reconstructed = torch.mm(U, torch.mm(torch.diag(S), Vh))
    W_compensation = W_reconstructed * compensation_ratio
    
    # 将补偿添加到目标层
    target_layer.weight.data = target_layer.weight.data + W_compensation.to(target_layer.weight.device)
    
    # 如果有偏置，也进行补偿
    if source_layer.bias is not None and target_layer.bias is not None:
        target_layer.bias.data = target_layer.bias.data + source_layer.bias.data * compensation_ratio

def compensate_and_remove_layers(
    model: nn.Module,
    layers_to_prune: List[int],
    compensation_direction: Literal["up", "down", "both"] = "both",
    compensation_ratio: float = 0.5,
    device: Literal["cuda", "cpu"] = "cuda",
    log_file: Optional[str] = None
) -> Tuple[nn.Module, List[int]]:
    """
    对多个层进行SVD补偿并移除
    
    Args:
        model: 模型
        layers_to_prune: 要剪枝的层索引列表
        compensation_direction: 补偿方向
        compensation_ratio: 补偿比例
        device: 计算设备
        log_file: 日志文件路径
        
    Returns:
        补偿并剪枝后的模型和实际移除的层索引列表
    """
    setup_logger(log_file)
    logger.info(f"开始对{len(layers_to_prune)}个层进行SVD补偿和剪枝")
    
    # 按降序排列，以避免移除层时的索引错误
    layers_to_prune = sorted(layers_to_prune, reverse=True)
    removed_layers = []
    
    for layer_idx in layers_to_prune:
        # 先进行补偿
        model = compensate_pruned_layer(
            model=model,
            layer_to_prune=layer_idx,
            compensation_direction=compensation_direction,
            compensation_ratio=compensation_ratio,
            device=device,
            log_file=log_file
        )
        
        # 然后移除层
        try:
            del model.model.layers[layer_idx]
            removed_layers.append(layer_idx)
            logger.info(f"成功移除第{layer_idx}层")
        except IndexError:
            logger.warning(f"移除第{layer_idx}层失败，可能该层已被移除")
    
    logger.info(f"完成SVD补偿和剪枝，共移除{len(removed_layers)}个层")
    return model, removed_layers