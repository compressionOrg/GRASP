import torch
import logging
import numpy as np
import torch.nn as nn
import math
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Literal, Optional, List, Union, Dict, Tuple
from tools.utils_func import block_influence, adaptive_rank_selection
from peft import LoraConfig, get_peft_model

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

class LoRALayer(nn.Module):
    """
    LoRA层，用于替代被移除的层
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 rank: int, 
                 alpha: float = 1.0, 
                 bias: Optional[torch.Tensor] = None):
        super(LoRALayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        # 初始化LoRA权重
        self.lora_A = nn.Parameter(torch.zeros((rank, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, rank)))
        self.scaling = alpha / rank
        
        # 初始化偏置
        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.bias = None
        
        # 初始化LoRA权重
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor):
        # LoRA前向传播: x -> A -> B -> output
        b, s, d = x.shape
        lora_output = torch.mm(
            x.view(b*s, -1), 
            self.lora_A.t()
        )
        lora_output = torch.mm(lora_output, self.lora_B.t())
        lora_output = lora_output * self.scaling
        
        # 添加偏置
        if self.bias is not None:
            lora_output = lora_output + self.bias
            
        return lora_output.view(b, s, -1)

class GRASPLoRAModel(nn.Module):
    def __init__(self, model: nn.Module, *args, **kwargs) -> None:
        super(GRASPLoRAModel, self).__init__(*args, **kwargs)
        self.model = model
        for params in self.model.parameters():
            params.requires_grad = False

        self.layer_importances = None
        self.redundant_layers = None
        self.lora_layers_info = {}  # 存储LoRA层信息
    
    def print_trainable_params(self, log_file: Optional[str] = None):
        setup_logger(log_file=log_file)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        trainable_percentage = (trainable_params / total_params) * 100
        logger.info(f"trainable params: {trainable_params} || all params: {total_params} || trainable: {trainable_percentage:.2f}%")
    
    def compute_bi(
            self,
            num_prune_layers: Optional[int] = 1,
            calibration_dataloader: Optional[DataLoader] = None,
            hiddens: Optional[List[torch.Tensor]] = None,
            angular: bool = False,
            device: Literal["cpu", "cuda"] = "cuda",
            log_file: Optional[str] = None,
            *args, **kwargs
        ):
        setup_logger(log_file=log_file)
        self.layer_importances = [0 for _ in self.model.model.layers]
        """
        Computes layer-wise importances over input tokens.
        """
        def compute_bi_hiddens(hiddens: Optional[List[torch.Tensor]] = None):
            if not angular:
                num_prune_layers = 1

            for i in range(len(hiddens) - num_prune_layers):
                in_hidden = hiddens[i]
                out_hidden = hiddens[i+num_prune_layers]
                if angular:
                    # use only last token for angular distance as described in section 3.2
                    # https://arxiv.org/pdf/2403.17887.pdf
                    in_hidden = in_hidden[:,-1:]
                    out_hidden = out_hidden[:,-1:]
                
                self.layer_importances[i] += block_influence(
                    in_hidden,
                    out_hidden,
                    angular=angular
                ).mean().cpu().item()

        logger.info("=======>Compute Block Influence")
        assert hiddens is not None or calibration_dataloader is not None, "please provide hidden_states or calibration dataloader to compute block influence"
        if hiddens is not None:
            compute_bi_hiddens(hiddens=hiddens)
        else:
            for batch in tqdm(calibration_dataloader, desc="Compute BI", total=len(calibration_dataloader), leave=True):
                if len(batch) == 2:
                    attention_mask = None
                else:
                    attention_mask = batch["attention_mask"].to(device=device)
                input_ids = batch["input_ids"].to(device=device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, output_hidden_states=True, return_dict=True)
                hiddens = outputs.hidden_states

                compute_bi_hiddens(hiddens=hiddens)
        
        if angular:
            start_layer = np.argsort(np.array(self.layer_importances[:-num_prune_layers+1]))[0]
            layers_to_remove = list(range(start_layer, start_layer + num_prune_layers))
        else:
            layers_to_remove = np.argsort(np.array(self.layer_importances))[:num_prune_layers].tolist()
        
        self.redundant_layers = layers_to_remove
        
        return self.layer_importances, layers_to_remove

    @staticmethod
    def _extract_layer_index(module_name):
        """
        Extracts the layer index from the module name.
        Supposes the module name contains the layer index in a consistent format, e.g., 'model.layers.23.mlp'.
        """
        try:
            parts = module_name.split('.')
            if "layers" in parts:
                index = int(parts[parts.index("layers") + 1])  # Index after "layers"
                return index
        except (ValueError, IndexError):
            return None

    def _set_module(self, model, submodule_key, module):
        tokens = submodule_key.split('.')
        sub_model = model
        for token in tokens[:-1]:
            sub_model = getattr(sub_model, token)
        setattr(sub_model, tokens[-1], module)

    def remove_layers(self, layers_to_remove: Optional[List[int]] = [], angular: Optional[bool] = False, num_prune_layers: Optional[int] = None):
        """
        移除指定的层
        """
        if not layers_to_remove:
            if angular:
                assert self.layer_importances, "Need to compute importances with self.compute_bi()"
                assert num_prune_layers, "Need number of layers to prune"
                start_layer = np.argsort(np.array(self.layer_importances[:-num_prune_layers+1]))[0]
                layers_to_remove = list(range(start_layer, start_layer + num_prune_layers))
            else:
                layers_to_remove = np.argsort(np.array(self.layer_importances))[:num_prune_layers].tolist()

        if layers_to_remove is not None:
            # remove layers in reverse to avoid indexing errors
            for layer_idx in sorted(layers_to_remove, reverse=True):
                try:
                    del self.model.model.layers[layer_idx]
                except IndexError:
                    logger.info(f"layer {layer_idx} does not exist, function may have already been called")
                    return []
            
            return layers_to_remove
        else:
            raise NotImplementedError("lack layers_to_remove")

    def replace_with_LoRALayer(self, target_layer: str, lora_rank_ratio: float = 0.1, lora_alpha: float = 16.0, device: Literal["cuda", "cpu"] = "cuda", log_file: Optional[str] = None):
        """
        将目标层替换为LoRA层
        
        Args:
            target_layer: 目标层路径
            lora_rank_ratio: LoRA秩比例（相对于输入维度）
            lora_alpha: LoRA缩放因子
            device: 计算设备
            log_file: 日志文件路径
            
        Returns:
            是否成功替换
        """
        setup_logger(log_file=log_file)
        replace_flag = False
        
        try:
            module = self.model.get_submodule(target=target_layer)
            if isinstance(module, nn.Linear):
                # 获取原始层的参数
                in_features = module.weight.shape[1]
                out_features = module.weight.shape[0]
                bias = module.bias
                
                # 计算LoRA秩
                rank = max(1, int(in_features * lora_rank_ratio))
                
                # 创建LoRA层
                lora_layer = LoRALayer(
                    in_features=in_features,
                    out_features=out_features,
                    rank=rank,
                    alpha=lora_alpha,
                    bias=bias
                )
                
                # 替换原始层
                self._set_module(self.model, target_layer, lora_layer)
                replace_flag = True
                
                # 记录LoRA层信息
                layer_idx = self._extract_layer_index(target_layer)
                if layer_idx is not None:
                    if layer_idx not in self.lora_layers_info:
                        self.lora_layers_info[layer_idx] = {}
                    
                    component_name = target_layer.split('.')[-1]
                    self.lora_layers_info[layer_idx][component_name] = {
                        "in_features": in_features,
                        "out_features": out_features,
                        "rank": rank,
                        "alpha": lora_alpha
                    }
                
                logger.info(f"成功将 {target_layer} 替换为LoRA层，秩={rank}")
            else:
                logger.warning(f"目标层 {target_layer} 不是线性层，无法替换为LoRA层")
        except Exception as e:
            logger.error(f"替换 {target_layer} 时出错: {e}")
        
        if not replace_flag:
            logger.warning(f"未能将 {target_layer} 替换为LoRA层")
        
        return replace_flag

    def compress_block(
            self,
            layer_id: int,
            block_type: Literal["attention", "mlp"],
            target_layer_types: Union[List[str], str] = ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
            lora_rank_ratio: float = 0.1,
            lora_alpha: float = 16.0,
            device: Literal["cuda", "cpu"] = "cuda",
            allocation_aware: Optional[bool] = None,
            verbose: bool = False,
            log_file: Optional[str] = None
        ):
        """
        压缩Transformer块，使用LoRA层替换原始层
        
        Args:
            layer_id: 层索引
            block_type: 块类型，"attention"或"mlp"
            target_layer_types: 目标层类型列表
            lora_rank_ratio: LoRA秩比例
            lora_alpha: LoRA缩放因子
            device: 计算设备
            allocation_aware: 是否感知分配
            verbose: 是否输出详细信息
            log_file: 日志文件路径
            
        Returns:
            是否跳过压缩
        """
        setup_logger(log_file=log_file)
        if layer_id is None:
            raise ValueError("Layer id should be given, but got None")
        
        if target_layer_types is None:
            return True

        if block_type == "attention":
            default_layer_types = ["q_proj", "k_proj", "v_proj", "o_proj"] # by default
            if not target_layer_types:
                target_layer_types = default_layer_types
            else:
                is_valid = all(layer in default_layer_types for layer in target_layer_types)
                if not is_valid:
                    raise ValueError(f"values in target layer types is not valid, should be one of {default_layer_types}")
            target_layer_types = ["self_attn." + target_layer_type for target_layer_type in target_layer_types]
        elif block_type == "mlp":
            default_layer_types = ["down_proj", "up_proj", "gate_proj"] # by default
            if not target_layer_types:
                target_layer_types = default_layer_types
            else:
                is_valid = all(layer in default_layer_types for layer in target_layer_types)
                if not is_valid:
                    raise ValueError(f"values in target layer types is not valid, should be one of {default_layer_types}")
            target_layer_types = ["mlp." + target_layer_type for target_layer_type in target_layer_types]
        else:
            raise NotImplementedError(f"block type {block_type} not support")
        
        base_layer_name = f"model.layers.{layer_id}."
        target_layer_names = [base_layer_name + target_layer_type for target_layer_type in target_layer_types]

        if allocation_aware:
            compression_ratio_list = []
            for target_layer in target_layer_names:
                module = self.model.get_submodule(target_layer)
                if isinstance(module, nn.Linear):
                    compression_ratio = getattr(module, "compression_ratio", None)
                    if compression_ratio is not None:
                        if isinstance(compression_ratio, torch.Tensor):
                            compression_ratio = compression_ratio.cpu().item()
                    else:
                        compression_ratio = None
                    compression_ratio_list.append(compression_ratio)
                    if compression_ratio == 0:
                        continue
                    else:
                        self.replace_with_LoRALayer(
                            target_layer=target_layer, 
                            lora_rank_ratio=lora_rank_ratio, 
                            lora_alpha=lora_alpha, 
                            device=device,
                            log_file=log_file
                        )
            if np.all(np.array(compression_ratio_list) == 0):
                return True
        else:
            for target_layer in target_layer_names:
                self.replace_with_LoRALayer(
                    target_layer=target_layer, 
                    lora_rank_ratio=lora_rank_ratio, 
                    lora_alpha=lora_alpha, 
                    device=device,
                    log_file=log_file
                )
        return False

    def train_lora_layers(
            self,
            calibration_dataloader: DataLoader,
            num_epochs: int = 1,
            learning_rate: float = 3e-4,
            device: Literal["cuda", "cpu"] = "cuda",
            log_file: Optional[str] = None
        ):
        """
        训练LoRA层以恢复模型性能
        
        Args:
            calibration_dataloader: 校准数据加载器
            num_epochs: 训练轮数
            learning_rate: 学习率
            device: 计算设备
            log_file: 日志文件路径
        """
        setup_logger(log_file=log_file)
        logger.info("开始训练LoRA层以恢复模型性能")
        
        # 确保模型在正确的设备上
        self.model.to(device=device)
        
        # 设置优化器，只优化可训练参数
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate
        )
        
        # 训练循环
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            # 进度条
            progress_bar = tqdm(calibration_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
            
            for batch in progress_bar:
                # 准备输入
                if len(batch) == 2:
                    attention_mask = None
                else:
                    attention_mask = batch["attention_mask"].to(device=device)
                input_ids = batch["input_ids"].to(device=device)
                labels = batch["labels"].to(device=device)
                
                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False
                )
                loss = outputs[0]
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 更新统计信息
                total_loss += loss.item()
                num_batches += 1
                
                # 更新进度条
                progress_bar.set_postfix({"loss": loss.item()})
                
                # 清理缓存
                if "cuda" in device:
                    torch.cuda.empty_cache()
            
            # 打印每个epoch的平均损失
            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
        
        logger.info("LoRA层训练完成")
        
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        return self

    def apply_peft_lora(
            self,
            layers_to_prune: List[int],
            lora_rank: int = 8,
            lora_alpha: int = 16,
            lora_dropout: float = 0.05,
            target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
            device: Literal["cuda", "cpu"] = "cuda",
            log_file: Optional[str] = None
        ):
        """
        使用PEFT库应用LoRA
        
        Args:
            layers_to_prune: 要剪枝的层索引列表
            lora_rank: LoRA秩
            lora_alpha: LoRA缩放因子
            lora_dropout: LoRA dropout率
            target_modules: 要应用LoRA的模块类型
            device: 计算设备
            log_file: 日志文件路径
        """
        setup_logger(log_file=log_file)
        logger.info(f"使用PEFT库对层 {layers_to_prune} 应用LoRA")
        
        # 确保层索引有效并排序
        num_layers = len(self.model.model.layers)
        valid_layers = [layer for layer in layers_to_prune if 0 <= layer < num_layers]
        
        if not valid_layers:
            logger.warning("没有有效的层索引，跳过LoRA应用")
            return self
        
        # 配置LoRA
        config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # 应用LoRA
        self.model = get_peft_model(self.model, config)
        
        # 打印可训练参数
        self.print_trainable_params(log_file=log_file)
        
        return self

    def check_exists_lora_layer(self, log_file: Optional[str] = None):
        """
        检查模型中是否存在LoRA层
        
        Args:
            log_file: 日志文件路径
            
        Returns:
            LoRA层名称列表
        """
        setup_logger(log_file=log_file)
        lora_layer_names = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, LoRALayer):
                lora_layer_names.append(name)
        
        if lora_layer_names:
            logger.info(f"模型中存在 {len(lora_layer_names)} 个LoRA层")
            if logger.level <= logging.DEBUG:
                for name in lora_layer_names:
                    logger.debug(f"LoRA层: {name}")
        else:
            logger.info("模型中不存在LoRA层")
        
        return lora_layer_names

    def apply_lora_compensation(
            self,
            layers_to_prune: List[int],
            lora_rank_ratio: float = 0.1,
            lora_alpha: float = 16.0,
            target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
            device: Literal["cuda", "cpu"] = "cuda",
            log_file: Optional[str] = None,
            continuous_layers_as_group: bool = True  # 新增参数
        ):
            """
            对要剪枝的层应用LoRA补偿
            
            Args:
                layers_to_prune: 要剪枝的层索引列表
                lora_rank_ratio: LoRA秩比例
                lora_alpha: LoRA缩放因子
                target_modules: 要应用LoRA的模块类型
                device: 计算设备
                log_file: 日志文件路径
                continuous_layers_as_group: 是否将连续层作为一个组处理
                
            Returns:
                应用了LoRA的层信息
            """
            setup_logger(log_file=log_file)
            logger.info(f"对层 {layers_to_prune} 应用自定义LoRA补偿")
            
            # 确定要应用LoRA的目标模块类型
            attn_target_modules = [m for m in target_modules if m in ["q_proj", "k_proj", "v_proj", "o_proj"]]
            mlp_target_modules = [m for m in target_modules if m in ["down_proj", "up_proj", "gate_proj"]]
            
            if continuous_layers_as_group and len(layers_to_prune) > 1:
                # 识别连续层组
                layer_groups = self.identify_continuous_layers(layers_to_prune)
                logger.info(f"检测到连续层组: {layer_groups}")
                
                # 对每个连续层组，只在第一个层之前应用LoRA
                for group in layer_groups:
                    # 记录组信息
                    for layer_id in group:
                        if layer_id not in self.lora_layers_info:
                            self.lora_layers_info[layer_id] = {}
                    
                    # 只对组中第一个层应用LoRA
                    first_layer = min(group)
                    
                    # 对注意力模块应用LoRA
                    if attn_target_modules:
                        self.compress_block(
                            layer_id=first_layer,
                            block_type="attention",
                            target_layer_types=attn_target_modules,
                            lora_rank_ratio=lora_rank_ratio,
                            lora_alpha=lora_alpha,
                            device=device,
                            log_file=log_file
                        )
                    
                    # 对MLP模块应用LoRA
                    if mlp_target_modules:
                        self.compress_block(
                            layer_id=first_layer,
                            block_type="mlp",
                            target_layer_types=mlp_target_modules,
                            lora_rank_ratio=lora_rank_ratio,
                            lora_alpha=lora_alpha,
                            device=device,
                            log_file=log_file
                        )
            else:
                # 原有逻辑：对每个要剪枝的层应用LoRA
                for layer_id in layers_to_prune:
                    # 对注意力模块应用LoRA
                    if attn_target_modules:
                        self.compress_block(
                            layer_id=layer_id,
                            block_type="attention",
                            target_layer_types=attn_target_modules,
                            lora_rank_ratio=lora_rank_ratio,
                            lora_alpha=lora_alpha,
                            device=device,
                            log_file=log_file
                        )
                    
                    # 对MLP模块应用LoRA
                    if mlp_target_modules:
                        self.compress_block(
                            layer_id=layer_id,
                            block_type="mlp",
                            target_layer_types=mlp_target_modules,
                            lora_rank_ratio=lora_rank_ratio,
                            lora_alpha=lora_alpha,
                            device=device,
                            log_file=log_file
                        )
                
                return self.lora_layers_info

    def identify_continuous_layers(self, layers: List[int]) -> List[List[int]]:
        """
        识别连续的层组
        
        Args:
            layers: 层索引列表
            
        Returns:
            连续层组列表
        """
        if not layers:
            return []
        
        # 按升序排序
        sorted_layers = sorted(layers)
        groups = []
        current_group = [sorted_layers[0]]
        
        for i in range(1, len(sorted_layers)):
            if sorted_layers[i] == sorted_layers[i-1] + 1:
                # 连续层
                current_group.append(sorted_layers[i])
            else:
                # 不连续，开始新组
                groups.append(current_group)
                current_group = [sorted_layers[i]]
        
        # 添加最后一组
        groups.append(current_group)
        return groups