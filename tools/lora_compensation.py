import torch
import torch.nn as nn
import logging
import numpy as np
from typing import Literal, Optional, Union, List, Tuple, Dict
# 添加必要的导入
import math

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

class LoRALinear(nn.Module):
    """
    实现带有LoRA适应的线性层
    """
    def __init__(
        self, 
        base_layer: nn.Linear, 
        rank: int, 
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0
    ):
        super(LoRALinear, self).__init__()
        
        self.base_layer = base_layer
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.rank
        
        # 冻结基础层参数
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
        # 创建LoRA参数
        self.lora_A = nn.Parameter(torch.zeros((rank, self.base_layer.in_features)))
        self.lora_B = nn.Parameter(torch.zeros((self.base_layer.out_features, rank)))
        
        # 改进LoRA参数初始化
        # 使用正态分布初始化A矩阵，方差为1/rank
        nn.init.normal_(self.lora_A, mean=0.0, std=1.0 / math.sqrt(rank))
        # B矩阵仍然初始化为零，以确保训练开始时LoRA没有影响
        nn.init.zeros_(self.lora_B)
        
        # LoRA dropout
        if lora_dropout > 0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 基础层前向传播
        base_output = self.base_layer(x)
        
        # LoRA前向传播
        lora_x = self.lora_dropout(x)
        lora_output = (lora_x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        
        # 合并输出
        return base_output + lora_output

def svd_to_lora_params(
    W_source: torch.Tensor, 
    rank: int, 
    device: Literal["cuda", "cpu"]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将源权重矩阵通过SVD分解转换为LoRA参数
    
    Args:
        W_source: 源权重矩阵
        rank: LoRA的秩
        device: 计算设备
        
    Returns:
        lora_A, lora_B: LoRA参数
    """
    # SVD分解
    U, S, Vh = torch.linalg.svd(W_source.to(device), full_matrices=False)
    
    # 截取前rank个奇异值和对应的奇异向量
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]
    
    # 转换为LoRA参数
    # B = U * sqrt(S)
    # A = sqrt(S) * Vh
    lora_B = U_r * torch.sqrt(S_r.unsqueeze(0))
    lora_A = torch.sqrt(S_r.unsqueeze(1)) * Vh_r
    
    return lora_A, lora_B

def apply_lora_compensation(
    model: nn.Module,
    layer_to_prune: int,
    target_layers: List[int],
    compensation_ratio: float,
    rank_ratio: float = 0.1,
    device: Literal["cuda", "cpu"] = "cuda",
    component_type: Literal["attention", "mlp", "both"] = "both"
) -> Dict[str, List[str]]:
    """
    将要剪枝层的信息以LoRA形式添加到目标层
    
    Args:
        model: 模型
        layer_to_prune: 要剪枝的层索引
        target_layers: 目标层索引列表
        compensation_ratio: 补偿比例
        rank_ratio: LoRA秩比例（相对于输入维度）
        device: 计算设备
        component_type: 要补偿的组件类型
        
    Returns:
        已添加LoRA的层路径字典
    """
    lora_layers = {"attention": [], "mlp": []}
    
    # 处理注意力模块
    if component_type in ["attention", "both"]:
        attention_components = ["q_proj", "k_proj", "v_proj", "o_proj"]
        for component in attention_components:
            source_path = f"model.layers.{layer_to_prune}.self_attn.{component}"
            try:
                source_module = model.get_submodule(source_path)
                if not isinstance(source_module, nn.Linear):
                    continue
                
                # 对每个目标层进行LoRA补偿
                for target_layer in target_layers:
                    target_path = f"model.layers.{target_layer}.self_attn.{component}"
                    try:
                        target_module = model.get_submodule(target_path)
                        if not isinstance(target_module, nn.Linear):
                            continue
                            
                        # 计算LoRA秩
                        rank = max(1, int(target_module.in_features * rank_ratio))
                        
                        # 获取源层权重并应用补偿比例
                        W_source = source_module.weight.data * compensation_ratio
                        
                        # 转换为LoRA参数
                        lora_A, lora_B = svd_to_lora_params(W_source, rank, device)
                        
                        # 创建并替换为LoRA层
                        lora_layer = create_lora_layer(target_module, lora_A, lora_B)
                        _set_module(model, target_path, lora_layer)
                        
                        lora_layers["attention"].append(target_path)
                        logger.info(f"已将 {source_path} 的信息以LoRA形式添加到 {target_path}")
                    except (AttributeError, ValueError) as e:
                        logger.warning(f"处理 {target_path} 时出错: {e}")
            except (AttributeError, ValueError) as e:
                logger.warning(f"处理 {source_path} 时出错: {e}")
    
    # 处理MLP模块
    if component_type in ["mlp", "both"]:
        mlp_components = ["down_proj", "up_proj", "gate_proj"]
        for component in mlp_components:
            source_path = f"model.layers.{layer_to_prune}.mlp.{component}"
            try:
                source_module = model.get_submodule(source_path)
                if not isinstance(source_module, nn.Linear):
                    continue
                
                # 对每个目标层进行LoRA补偿
                for target_layer in target_layers:
                    target_path = f"model.layers.{target_layer}.mlp.{component}"
                    try:
                        target_module = model.get_submodule(target_path)
                        if not isinstance(target_module, nn.Linear):
                            continue
                            
                        # 计算LoRA秩
                        rank = max(1, int(target_module.in_features * rank_ratio))
                        
                        # 获取源层权重并应用补偿比例
                        W_source = source_module.weight.data * compensation_ratio
                        
                        # 转换为LoRA参数
                        lora_A, lora_B = svd_to_lora_params(W_source, rank, device)
                        
                        # 创建并替换为LoRA层
                        lora_layer = create_lora_layer(target_module, lora_A, lora_B)
                        _set_module(model, target_path, lora_layer)
                        
                        lora_layers["mlp"].append(target_path)
                        logger.info(f"已将 {source_path} 的信息以LoRA形式添加到 {target_path}")
                    except (AttributeError, ValueError) as e:
                        logger.warning(f"处理 {target_path} 时出错: {e}")
            except (AttributeError, ValueError) as e:
                logger.warning(f"处理 {source_path} 时出错: {e}")
    
    return lora_layers

def create_lora_layer(
    base_layer: nn.Linear,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor
) -> nn.Module:
    """
    创建一个包含预初始化LoRA参数的层
    
    Args:
        base_layer: 基础线性层
        lora_A: 预计算的LoRA A参数
        lora_B: 预计算的LoRA B参数
        
    Returns:
        带有LoRA的线性层
    """
    rank = lora_A.shape[0]
    lora_layer = LoRALinear(base_layer, rank)
    
    # 使用预计算的参数初始化LoRA权重
    lora_layer.lora_A.data.copy_(lora_A)
    lora_layer.lora_B.data.copy_(lora_B)
    
    return lora_layer

def _set_module(model, submodule_key, module):
    """
    在模型中设置子模块
    
    Args:
        model: 模型
        submodule_key: 子模块路径
        module: 要设置的新模块
    """
    tokens = submodule_key.split('.')
    sub_model = model
    for token in tokens[:-1]:
        sub_model = getattr(sub_model, token)
    setattr(sub_model, tokens[-1], module)

def hybrid_compensation(
    model: nn.Module,
    layers_to_prune: List[int] = None,
    num_prune_layers: Optional[int] = None,
    angular: bool = False,
    compensation_direction: Literal["up", "down", "both"] = "both",
    svd_compensation_ratio: float = 0.5,
    lora_compensation_ratio: float = 0.3,
    lora_rank_ratio: float = 0.1,
    device: Literal["cuda", "cpu"] = "cuda",
    log_file: Optional[str] = None
) -> Tuple[nn.Module, List[int], Dict]:
    """
    混合补偿机制：结合SVD直接补偿和LoRA适应补偿
    
    Args:
        model: 模型
        layers_to_prune: 要剪枝的层索引列表
        num_prune_layers: 要剪枝的层数量
        angular: 是否使用角度距离计算重要性
        compensation_direction: 补偿方向
        svd_compensation_ratio: SVD直接补偿比例
        lora_compensation_ratio: LoRA补偿比例
        lora_rank_ratio: LoRA秩比例
        device: 计算设备
        log_file: 日志文件路径
        
    Returns:
        补偿后的模型、已移除的层列表和LoRA层信息
    """
    from tools.layer_compensation import compensate_and_remove_layers, identify_continuous_layers
    
    setup_logger(log_file)
    
    # 如果没有提供要剪枝的层，则根据重要性计算
    if layers_to_prune is None:
        if hasattr(model, 'layer_importances'):
            layer_importances = model.layer_importances
        else:
            logger.warning("未提供要剪枝的层，且模型没有层重要性信息，无法确定要剪枝的层")
            return model, [], {}
        
        # 根据angular参数选择不同的层选择策略
        if angular:
            if num_prune_layers is None:
                num_prune_layers = 1
                logger.warning("未指定num_prune_layers，默认设置为1")
            
            # 使用角度距离时，选择连续的num_prune_layers个层
            start_layer = np.argsort(np.array(layer_importances[:-num_prune_layers+1]))[0]
            layers_to_prune = list(range(start_layer, start_layer + num_prune_layers))
        else:
            if num_prune_layers is None:
                num_prune_layers = 1
                logger.warning("未指定num_prune_layers，默认设置为1")
            
            # 不使用角度距离时，选择最不重要的num_prune_layers个层
            layers_to_prune = np.argsort(np.array(layer_importances))[:num_prune_layers].tolist()
    
    logger.info(f"开始对层 {layers_to_prune} 进行混合补偿")
    
    # 确保层索引有效并排序
    num_layers = len(model.model.layers)
    valid_layers = [layer for layer in layers_to_prune if 0 <= layer < num_layers]
    if len(valid_layers) != len(layers_to_prune):
        invalid_layers = set(layers_to_prune) - set(valid_layers)
        logger.warning(f"以下层索引无效，将被忽略: {invalid_layers}")
    
    if not valid_layers:
        logger.warning("没有有效的层索引，跳过补偿和移除")
        return model, [], {}
    
    # 按降序排序，以便从后向前移除（避免索引变化）
    valid_layers.sort(reverse=True)
    
    # 识别连续的层组
    layer_groups = identify_continuous_layers(valid_layers)
    logger.info(f"识别到的连续层组: {layer_groups}")
    
    # 第一步：应用SVD直接补偿
    logger.info(f"第一步：应用SVD直接补偿，比例为 {svd_compensation_ratio}")
    model, removed_layers = compensate_and_remove_layers(
        model=model,
        layers_to_prune=valid_layers,
        compensation_direction=compensation_direction,
        compensation_ratio=svd_compensation_ratio,
        device=device,
        log_file=log_file
    )
    
    # 第二步：应用LoRA适应补偿
    logger.info(f"第二步：应用LoRA适应补偿，比例为 {lora_compensation_ratio}")
    
    # 确定补偿的目标层
    all_lora_layers = {}
    
    # 获取更新后的层数
    updated_num_layers = len(model.model.layers)
    
    # 创建原始层索引到新层索引的映射 - 修复映射逻辑
    layer_index_map = {}
    removed_layers_set = set(removed_layers)
    
    # 计算每个原始层索引对应的新索引
    new_idx = 0
    for original_idx in range(num_layers):
        if original_idx not in removed_layers_set:
            layer_index_map[original_idx] = new_idx
            new_idx += 1
    
    # 对每个要剪枝的层应用LoRA补偿
    for layer_to_prune in valid_layers:
        if layer_to_prune in removed_layers:
            # 如果层已被移除，跳过
            continue
            
        # 获取当前层的新索引
        new_layer_idx = layer_index_map.get(layer_to_prune, layer_to_prune)
        
        # 确定目标层
        target_layers = []
        if compensation_direction in ["up", "both"] and new_layer_idx > 0:
            target_layers.append(new_layer_idx - 1)
        if compensation_direction in ["down", "both"] and new_layer_idx < updated_num_layers - 1:
            target_layers.append(new_layer_idx + 1)
            
        if not target_layers:
            logger.warning(f"层 {layer_to_prune}(新索引:{new_layer_idx}) 没有可用的目标层进行LoRA补偿")
            continue
            
        # 应用LoRA补偿
        lora_layers = apply_lora_compensation(
            model=model,
            layer_to_prune=new_layer_idx,  # 使用新的层索引
            target_layers=target_layers,
            compensation_ratio=lora_compensation_ratio,
            rank_ratio=lora_rank_ratio,
            device=device,
            component_type="both"
        )
        
        all_lora_layers[layer_to_prune] = lora_layers
    
    logger.info(f"混合补偿完成，已移除层: {sorted(removed_layers)}")
    return model, sorted(removed_layers), all_lora_layers

