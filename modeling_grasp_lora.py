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

class WholeLayerLoRA(nn.Module):
    """
    整个Transformer层的LoRA替代，直接替换整个层而不是单独替换子模块
    """
    def __init__(
        self,
        hidden_size: int,
        rank: int,
        alpha: float = 16.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 初始化LoRA权重
        self.lora_A = nn.Parameter(torch.zeros((rank, hidden_size)))
        self.lora_B = nn.Parameter(torch.zeros((hidden_size, rank)))
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]: # Adjusted return type hint
        """
        整个Transformer层的前向传播，兼容原始Llama层的接口
        
        Args:
            hidden_states: 输入隐藏状态 (batch, seq_len, hidden_size)
            attention_mask: 注意力掩码 (可选)
            position_ids: 位置编码 (可选)
            past_key_value: 过去的键值对 (可选)
            output_attentions: 是否输出注意力权重 (可选)
            use_cache: 是否使用缓存 (可选)
            
        Returns:
            输出隐藏状态和可选的键值对
        """
        # 保存原始形状
        original_shape = hidden_states.shape

        # 检查输入维度 (可选，但有助于调试)
        if hidden_states.ndim != 3:
             logger.error(f"WholeLayerLoRA received hidden_states with unexpected shape: {hidden_states.shape}")
             raise ValueError(f"Expected hidden_states to have 3 dimensions (batch, seq_len, hidden_size), but got {hidden_states.ndim} dimensions with shape {hidden_states.shape}")

        # 计算LoRA输出
        x_flat = hidden_states.view(-1, self.hidden_size)
        lora_output = torch.mm(x_flat, self.lora_A.t())
        lora_output = torch.mm(lora_output, self.lora_B.t())
        lora_output = lora_output * self.scaling

        # 恢复原始形状
        output = lora_output.view(original_shape)

        # --- Modification Start ---
        # 始终返回一个元组，其结构与 LlamaDecoderLayer 兼容
        # 第一个元素必须是 hidden states 输出
        layer_outputs = (output,)

        # 根据标志附加占位符，模仿 LlamaDecoderLayer 的签名
        if output_attentions:
            layer_outputs += (None,)  # attention weights 的占位符

        if use_cache:
            layer_outputs += (None,)  # past_key_value 的占位符

        return layer_outputs
        # --- Modification End ---

class GRASPLoRAModel(nn.Module):
    def __init__(self, model: nn.Module, *args, **kwargs) -> None:
        super(GRASPLoRAModel, self).__init__(*args, **kwargs)
        self.model = model
        # 冻结所有参数
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
                
                # 确保只有LoRA层的参数是可训练的
                for param in lora_layer.parameters():
                    param.requires_grad = True
                
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

    def replace_whole_layer_with_lora(
            self,
            layer_id: int,
            lora_rank_ratio: float = 0.1,
            lora_alpha: float = 16.0,
            device: Literal["cuda", "cpu"] = "cuda",
            log_file: Optional[str] = None
        ) -> bool:
        """
        将整个Transformer层替换为LoRA层
        
        Args:
            layer_id: 层索引
            lora_rank_ratio: LoRA秩比例
            lora_alpha: LoRA缩放因子
            device: 计算设备
            log_file: 日志文件路径
            
        Returns:
            是否成功替换
        """
        setup_logger(log_file=log_file)
        
        try:
            # 获取隐藏层大小
            hidden_size = self.model.config.hidden_size
            
            # 计算LoRA秩
            rank = max(1, int(hidden_size * lora_rank_ratio))
            
            # 创建整层LoRA替代
            lora_layer = WholeLayerLoRA(
                hidden_size=hidden_size,
                rank=rank,
                alpha=lora_alpha
            )
            
            # 替换原始层
            self.model.model.layers[layer_id] = lora_layer
            
            # 确保只有LoRA层的参数是可训练的
            for param in lora_layer.parameters():
                param.requires_grad = True
            
            # 记录LoRA层信息
            if layer_id not in self.lora_layers_info:
                self.lora_layers_info[layer_id] = {}
            
            self.lora_layers_info[layer_id]["whole_layer"] = {
                "hidden_size": hidden_size,
                "rank": rank,
                "alpha": lora_alpha
            }
            
            logger.info(f"成功将第 {layer_id} 层整体替换为LoRA层，秩={rank}")
            return True
            
        except Exception as e:
            logger.error(f"替换第 {layer_id} 层时出错: {e}")
            return False
    
    def compress_model_with_lora(
            self,
            calibration_dataloader: DataLoader,
            layers_id: Optional[Union[List[int], int]] = None,
            num_prune_layers: Optional[int] = None,
            lora_rank_ratio: float = 0.1,
            lora_alpha: float = 16,
            target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
            device: Literal["cuda", "cpu"] = "cuda",
            angular: Optional[bool] = False,
            verbose: Optional[bool] = False,
            recovery: Optional[bool] = True,
            recovery_epochs: int = 1,
            recovery_lr: float = 3e-4,
            continuous_layers_as_group: bool = True,
            log_file: Optional[str] = None,
            tokenizer=None,
            *args, **kwargs
        ):
            """
            一次性完成整个模型压缩流程，包括层选择、LoRA替换和恢复训练
            
            Args:
                calibration_dataloader: 校准数据加载器
                layers_id: 要替换的层ID列表
                num_prune_layers: 如果未指定layers_id，要替换的层数
                lora_rank_ratio: LoRA秩与隐藏层大小的比例
                lora_alpha: LoRA alpha参数
                target_modules: 要替换的目标模块类型
                device: 计算设备
                angular: 是否使用角度相似度计算层相似性
                verbose: 是否输出详细日志
                recovery: 是否进行恢复训练
                recovery_epochs: 恢复训练的轮数
                recovery_lr: 恢复训练的学习率
                continuous_layers_as_group: 是否将连续层作为一个组处理
                log_file: 日志文件路径
                tokenizer: 分词器
                
            Returns:
                压缩结果信息
            """
            setup_logger(log_file=log_file)
            
            # 1. 计算层重要性并选择要替换的层
            if layers_id is None:
                logger.info("=======> 计算层重要性")
                layers_importance, layers_id = self.compute_bi(
                    num_prune_layers=num_prune_layers, 
                    calibration_dataloader=calibration_dataloader, 
                    angular=angular, 
                    device=device
                )
                logger.info(f"层重要性: {layers_importance}")
            
            if isinstance(layers_id, int):
                layers_id = [layers_id]
            
            self.redundant_layers = layers_id
            logger.info(f"要替换的层: {layers_id}")
            
            # 2. 检测连续层组
            layer_groups = []
            if continuous_layers_as_group and len(layers_id) > 1:
                # 先按降序排序
                layers_id.sort(reverse=True)
                layer_groups = self.identify_continuous_layers(layers_id)
                if len(layer_groups) < len(layers_id):
                    logger.info(f"=======> 检测到连续层组: {layer_groups}")
            
            # 3. 替换选定的层为LoRA层
            logger.info("=======> 开始替换层为LoRA层")
            replaced_layers = []
            
            # 如果有连续层组且启用了连续层组处理
            if continuous_layers_as_group and layer_groups:
                # 先移除所有要剪枝的层
                removed_layers = self.remove_layers(layers_to_remove=layers_id)
                logger.info(f"已移除层: {removed_layers}")
                
                # 对每个连续层组，在组的起始位置添加一个WholeLayerLoRA
                for group in layer_groups:
                    # 获取组的起始位置
                    start_layer = min(group)
                    logger.info(f"在位置 {start_layer} 添加WholeLayerLoRA替代连续层组 {group}")
                    
                    # 添加WholeLayerLoRA
                    success = self.replace_whole_layer_with_lora(
                        layer_id=start_layer,
                        lora_rank_ratio=lora_rank_ratio,
                        lora_alpha=lora_alpha,
                        device=device,
                        log_file=log_file
                    )
                    
                    if success:
                        replaced_layers.append(f"model.layers.{start_layer}")
            else:
                # 如果没有连续层组或未启用连续层组处理，按原来的方式处理每一层
                for layer_id in tqdm(layers_id, desc="替换为LoRA层", total=len(layers_id), leave=True):
                    # 构建目标层路径
                    layer_paths = []
                    for module_name in target_modules:
                        layer_paths.append(f"model.layers.{layer_id}.self_attn.{module_name}")
                        layer_paths.append(f"model.layers.{layer_id}.mlp.{module_name}")
                    
                    # 替换为LoRA层
                    for layer_path in layer_paths:
                        success = self.replace_with_LoRALayer(
                            target_layer=layer_path,
                            lora_rank_ratio=lora_rank_ratio,
                            lora_alpha=lora_alpha,
                            device=device,
                            log_file=log_file
                        )
                        if success:
                            replaced_layers.append(layer_path)
            
            logger.info(f"成功替换的层: {replaced_layers}")
            
            # 4. 如果需要恢复训练，训练LoRA参数
            if recovery and replaced_layers:
                logger.info("=======> 开始LoRA参数恢复训练")
                self.train_lora_compensation(
                    tokenizer=tokenizer,
                    recovery_epochs=recovery_epochs,
                    recovery_lr=recovery_lr,
                    log_file=log_file
                )
            
            return {
                "replaced_layers": replaced_layers,
                "redundant_layers": self.redundant_layers,
                "lora_layers_info": self.lora_layers_info
            }

    def train_lora_compensation(
            self,
            tokenizer=None,
            data_path: Optional[str] = 'yahma/alpaca-cleaned',
            output_dir: Optional[str] = './checkpoint',
            batch_size: int = 32,
            mirco_batch_size: int = 4,
            recovery_epochs: int = 1,
            recovery_lr: float = 3e-4,
            max_length: int = 256,
            val_set_size: int = 2000,
            train_on_inputs: bool = True,
            add_eos_token: bool = False,
            resume_from_checkpoint: Optional[str] = None,
            prompt_template_name: str = "alpaca",
            train_device: Optional[str] = None,
            log_file: Optional[str] = None,
            **kwargs
        ):
            """
            训练LoRA层参数以恢复模型性能
            
            Args:
                tokenizer: 分词器
                data_path: 训练数据路径
                output_dir: 输出目录
                batch_size: 批处理大小
                mirco_batch_size: 微批处理大小
                recovery_epochs: 训练轮数
                recovery_lr: 学习率
                max_length: 最大序列长度
                val_set_size: 验证集大小
                train_on_inputs: 是否在输入上训练
                add_eos_token: 是否添加EOS标记
                resume_from_checkpoint: 恢复训练的检查点
                prompt_template_name: 提示模板名称
                train_device: 训练设备
                log_file: 日志文件路径
            """
            setup_logger(log_file=log_file)
            
            # 导入train函数
            from alpaca_grasp import train
            
            # 调用train函数进行训练
            logger.info("=======> 调用alpaca_grasp.train进行LoRA参数恢复训练")
            trained_model = train(
                grasp_model=self,
                tokenizer=tokenizer,
                data_path=data_path,
                output_dir=output_dir,
                batch_size=batch_size,
                mirco_batch_size=mirco_batch_size,
                num_epochs=recovery_epochs,
                learning_rate=recovery_lr,
                max_length=max_length,
                val_set_size=val_set_size,
                train_on_inputs=train_on_inputs,
                add_eos_token=add_eos_token,
                resume_from_checkpoint=resume_from_checkpoint,
                prompt_template_name=prompt_template_name,
                train_device=train_device,
                log_file=log_file,
                **kwargs
            )
            
            # 更新当前模型
            self.model = trained_model.model
            
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

    def identify_continuous_layers(self, layers_id: List[int]) -> List[List[int]]:
        """
        识别连续的层组
        
        Args:
            layers_id: 层索引列表
            
        Returns:
            连续层组列表
        """
        if not layers_id:
            return []
        
        # 按升序排序
        sorted_layers = sorted(layers_id)
        
        # 初始化结果和当前组
        groups = []
        current_group = [sorted_layers[0]]
        
        # 遍历排序后的层ID
        for i in range(1, len(sorted_layers)):
            # 如果当前层与前一层连续
            if sorted_layers[i] == sorted_layers[i-1] + 1:
                current_group.append(sorted_layers[i])
            else:
                # 如果不连续，保存当前组并开始新组
                groups.append(current_group)
                current_group = [sorted_layers[i]]
        
        # 添加最后一个组
        groups.append(current_group)
        
        return groups

    def apply_lora_compensation(
            self,
            calibration_dataloader: DataLoader,
            device: Literal["cuda", "cpu"] = "cuda",
            log_file: Optional[str] = None
        ) -> None:
        """
        应用LoRA补偿，调整LoRA层的权重以匹配原始层的行为
        
        Args:
            calibration_dataloader: 校准数据加载器
            device: 计算设备
            log_file: 日志文件路径
        """
        setup_logger(log_file=log_file)
        
        # 检查是否有LoRA层
        lora_layers = self.check_exists_lora_layer(log_file=log_file)
        if not lora_layers:
            logger.warning("没有找到LoRA层，无法应用补偿")
            return
        
        logger.info(f"找到LoRA层: {lora_layers}")
        
        # 将模型移至指定设备
        self.model.to(device=device)
        
        # 收集校准数据的隐藏状态
        hidden_states_dict = {}
        
        logger.info("收集校准数据的隐藏状态...")
        with torch.no_grad():
            for batch in tqdm(calibration_dataloader, desc="收集隐藏状态", leave=True):
                # 准备输入
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"].to(device=device)
                    attention_mask = batch["attention_mask"].to(device=device) if "attention_mask" in batch else None
                else:
                    input_ids = batch[0].to(device=device)
                    attention_mask = batch[1].to(device=device) if len(batch) > 1 else None
                
                # 前向传播，收集所有层的隐藏状态
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # 存储隐藏状态
                for i, hidden_state in enumerate(outputs.hidden_states):
                    if i not in hidden_states_dict:
                        hidden_states_dict[i] = []
                    hidden_states_dict[i].append(hidden_state.detach().cpu())
        
        # 对每个LoRA层应用补偿
        logger.info("应用LoRA补偿...")
        for layer_name in lora_layers:
            try:
                # 解析层索引
                layer_idx = int(layer_name.split('.')[-2])
                
                # 获取该层的模块
                layer_module = self.model.model.layers[layer_idx]
                
                # 检查是否为WholeLayerLoRA
                if isinstance(layer_module, WholeLayerLoRA):
                    logger.info(f"为第 {layer_idx} 层的WholeLayerLoRA应用补偿")
                    
                    # 获取输入和输出隐藏状态
                    input_hiddens = torch.cat(hidden_states_dict[layer_idx], dim=0)
                    output_hiddens = torch.cat(hidden_states_dict[layer_idx + 1], dim=0)
                    
                    # 将隐藏状态移至计算设备
                    input_hiddens = input_hiddens.to(device=device)
                    output_hiddens = output_hiddens.to(device=device)
                    
                    # 计算最佳的LoRA权重
                    # 使用最小二乘法求解 B·A·X ≈ Y
                    X = input_hiddens.reshape(-1, input_hiddens.size(-1))
                    Y = output_hiddens.reshape(-1, output_hiddens.size(-1))
                    
                    # 计算A·X
                    A_X = torch.mm(X, layer_module.lora_A.t())
                    
                    # 计算伪逆以求解B
                    A_X_pinv = torch.pinverse(A_X)
                    B_optimal = torch.mm(Y, A_X_pinv)
                    
                    # 更新LoRA权重
                    with torch.no_grad():
                        layer_module.lora_B.copy_(B_optimal)
                    
                    logger.info(f"成功为第 {layer_idx} 层应用LoRA补偿")
                
            except Exception as e:
                logger.error(f"为层 {layer_name} 应用补偿时出错: {e}")
        
        logger.info("LoRA补偿应用完成")

    def check_exists_lora_layer(self, log_file: Optional[str] = None) -> List[str]:
        """
        检查模型中是否存在LoRA层
        
        Args:
            log_file: 日志文件路径
            
        Returns:
            LoRA层名称列表
        """
        setup_logger(log_file=log_file)
        
        lora_layers = []
        
        # 遍历模型的所有模块
        for name, module in self.model.named_modules():
            if isinstance(module, (LoRALayer, WholeLayerLoRA)):
                lora_layers.append(name)
        
        if lora_layers:
            logger.info(f"找到 {len(lora_layers)} 个LoRA层: {lora_layers}")
        else:
            logger.warning("未找到任何LoRA层")
        
        return lora_layers

    def ensure_only_lora_trainable(self, log_file: Optional[str] = None):
        """
        确保只有LoRA层的参数是可训练的
        
        Args:
            log_file: 日志文件路径
        """
        setup_logger(log_file=log_file)
        
        # 首先冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 然后只解冻LoRA层的参数
        lora_params_count = 0
        
        # 遍历所有模块，找到LoRA层并解冻其参数
        for name, module in self.model.named_modules():
            if isinstance(module, (LoRALayer, WholeLayerLoRA)):
                for param_name, param in module.named_parameters():
                    param.requires_grad = True
                    lora_params_count += param.numel()
        
        logger.info(f"已设置 {lora_params_count} 个LoRA参数为可训练状态")
        
        # 打印可训练参数信息
        self.print_trainable_params(log_file=log_file)
        
        return lora_params_count
