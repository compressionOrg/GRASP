import torch
import torch.nn as nn
import logging
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, List, Literal, Dict, Union, Tuple, Any

logger = logging.getLogger(__name__)

def setup_logger(log_file=None):
    # 清除现有处理程序
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    if log_file:
        handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class GatedResidual(nn.Module):
    """
    可学习门控残差连接模块。

    用于在Transformer剪枝区域的前后块之间插入，实现主分支与残差分支的可学习融合。

    Attributes:
        hidden_size (int): 输入特征维度
    """
    def __init__(self, hidden_size: int):
        """
        初始化GatedResidual模块。

        Args:
            hidden_size (int): 输入特征维度
        """
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(1, 1, hidden_size))
        # 初始化为0，sigmoid后初始门控为0.5

    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播，融合主分支和残差分支。

        Args:
            x (torch.Tensor): 主分支输出，形状为 (batch, seq_len, hidden_size)
            residual (Optional[torch.Tensor]): 残差分支输入，形状同x。若为None，则直接返回x。

        Returns:
            torch.Tensor: 融合后的输出
        """
        if residual is None:
            return x
        gate = torch.sigmoid(self.gate)
        return gate * x + (1 - gate) * residual

    def get_gate_value(self) -> float:
        """
        获取当前门控值（sigmoid后的值）。

        Returns:
            float: 当前门控值，范围[0,1]
        """
        return torch.sigmoid(self.gate.data.mean()).item()


class GatedResidualModel(nn.Module):
    """
    包装模型并管理门控残差连接的模型类。
    
    类似于GRASPModel，用于管理门控残差连接的添加、训练和保存。
    
    Attributes:
        model (nn.Module): 原始模型
        gated_residuals (Dict[int, GatedResidual]): 门控残差模块字典，键为层索引
        redundant_layers (List[int]): 冗余层索引列表
        layer_groups (List[List[int]]): 连续层组列表
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.gated_residuals = {}
        self.redundant_layers = []
        self._original_forwards = {}  # 保存原始前向传播函数
        self.layer_groups = []  # 存储连续层组
        
    def forward(self, *args, **kwargs):
        """
        前向传播，直接调用原始模型的前向传播函数。
        """
        return self.model(*args, **kwargs)
    
    def identify_continuous_layers(self, layers: List[int]) -> List[List[int]]:
        """
        识别连续的层组。
        
        Args:
            layers: 层索引列表
            
        Returns:
            连续层组列表，每个元素是一个连续层的列表
        """
        if not layers:
            return []
        
        # 确保层索引有序
        sorted_layers = sorted(layers)
        
        groups = []
        current_group = [sorted_layers[0]]
        
        for i in range(1, len(sorted_layers)):
            # 如果当前层与前一层连续
            if sorted_layers[i] == sorted_layers[i-1] + 1:
                current_group.append(sorted_layers[i])
            else:
                # 当前层与前一层不连续，保存当前组并开始新组
                groups.append(current_group)
                current_group = [sorted_layers[i]]
        
        # 添加最后一个组
        groups.append(current_group)
        
        return groups
        
    def apply_gated_residual(
        self,
        layers_to_prune: List[int],
        device: Literal["cuda", "cpu"] = "cuda",
        log_file: Optional[str] = None,
        remove_layers: bool = True  # 控制是否立即移除冗余层
    ) -> Dict[int, GatedResidual]:
        """
        先移除剪枝层，然后在移除层的前后添加门控残差连接。
        
        Args:
            layers_to_prune: 要剪枝的层索引列表
            device: 计算设备
            log_file: 日志文件路径
            remove_layers: 是否立即移除冗余层
            
        Returns:
            添加的门控残差模块字典，键为层索引
        """
        setup_logger(log_file)
        logger.info(f"开始处理剪枝层 {layers_to_prune}")
        
        # 保存冗余层信息
        self.redundant_layers = layers_to_prune
        
        # 确保层索引有效
        num_layers = len(self.model.model.layers)
        valid_layers = [layer for layer in layers_to_prune if 0 <= layer < num_layers]
        if len(valid_layers) != len(layers_to_prune):
            invalid_layers = set(layers_to_prune) - set(valid_layers)
            logger.warning(f"以下层索引无效，将被忽略: {invalid_layers}")
        
        if not valid_layers:
            logger.warning("没有有效的层索引，跳过处理")
            return {}
        
        # 识别连续层组
        self.layer_groups = self.identify_continuous_layers(valid_layers)
        logger.info(f"识别到的连续层组: {self.layer_groups}")
        
        # 获取隐藏层大小
        hidden_size = self.model.config.hidden_size
        
        # 创建层索引到新索引的映射
        layer_index_map = {}
        
        # 如果需要，先移除冗余层
        if remove_layers:
            logger.info("先移除冗余层")
            # 按降序移除层，避免索引错误
            for layer_idx in sorted(valid_layers, reverse=True):
                try:
                    logger.info(f"移除第 {layer_idx} 层")
                    del self.model.model.layers[layer_idx]
                except IndexError:
                    logger.warning(f"无法移除第 {layer_idx} 层，可能已被移除")
            
            # 计算移除层后的新索引映射
            new_idx = 0
            for original_idx in range(num_layers):
                if original_idx not in valid_layers:
                    layer_index_map[original_idx] = new_idx
                    new_idx += 1
        
        # 对每个连续层组，在其前后添加门控残差连接
        for group in self.layer_groups:
            # 确定组的边界
            first_layer = min(group)
            last_layer = max(group)
            
            # 确定前后层的索引
            prev_layer_idx = first_layer - 1
            next_layer_idx = last_layer + 1
            
            # 如果前一层存在且不在要剪枝的层中
            if prev_layer_idx >= 0 and prev_layer_idx not in valid_layers:
                # 获取前一层的新索引
                new_prev_idx = layer_index_map.get(prev_layer_idx, prev_layer_idx)
                
                # 创建门控残差模块
                gated_res = GatedResidual(hidden_size).to(device)
                
                # 保存原始层的前向传播函数
                original_layer = self.model.model.layers[new_prev_idx]
                original_forward = original_layer.forward
                self._original_forwards[new_prev_idx] = original_forward
                
                # 定义新的前向传播函数，添加门控残差
                def make_forward_with_residual(original_forward, gated_res):
                    def forward_with_residual(hidden_states, *args, **kwargs):
                        # 保存输入作为残差连接
                        residual = hidden_states
                        
                        # 调用原始前向传播
                        outputs = original_forward(hidden_states, *args, **kwargs)
                        
                        # 应用门控残差连接
                        if isinstance(outputs, tuple):
                            # 如果输出是元组，修改第一个元素（隐藏状态）
                            hidden_states_out = outputs[0]
                            gated_output = gated_res(hidden_states_out, residual)
                            outputs = (gated_output,) + outputs[1:]
                        else:
                            # 如果输出是张量，直接应用门控
                            outputs = gated_res(outputs, residual)
                        
                        return outputs
                    
                    return forward_with_residual
                
                # 替换层的前向传播函数
                original_layer.forward = make_forward_with_residual(original_forward, gated_res)
                
                # 记录添加的门控残差模块
                self.gated_residuals[new_prev_idx] = gated_res
                logger.info(f"成功在第 {prev_layer_idx}(新索引:{new_prev_idx}) 层添加门控残差连接")
            
            # 如果后一层存在且不在要剪枝的层中
            if next_layer_idx < num_layers and next_layer_idx not in valid_layers:
                # 获取后一层的新索引
                new_next_idx = layer_index_map.get(next_layer_idx, next_layer_idx - len([l for l in valid_layers if l < next_layer_idx]))
                
                # 创建门控残差模块
                gated_res = GatedResidual(hidden_size).to(device)
                
                # 保存原始层的前向传播函数
                original_layer = self.model.model.layers[new_next_idx]
                original_forward = original_layer.forward
                self._original_forwards[new_next_idx] = original_forward
                
                # 定义新的前向传播函数，添加门控残差
                def make_forward_with_residual(original_forward, gated_res):
                    def forward_with_residual(hidden_states, *args, **kwargs):
                        # 保存输入作为残差连接
                        residual = hidden_states
                        
                        # 调用原始前向传播
                        outputs = original_forward(hidden_states, *args, **kwargs)
                        
                        # 应用门控残差连接
                        if isinstance(outputs, tuple):
                            # 如果输出是元组，修改第一个元素（隐藏状态）
                            hidden_states_out = outputs[0]
                            gated_output = gated_res(hidden_states_out, residual)
                            outputs = (gated_output,) + outputs[1:]
                        else:
                            # 如果输出是张量，直接应用门控
                            outputs = gated_res(outputs, residual)
                        
                        return outputs
                    
                    return forward_with_residual
                
                # 替换层的前向传播函数
                original_layer.forward = make_forward_with_residual(original_forward, gated_res)
                
                # 记录添加的门控残差模块
                self.gated_residuals[new_next_idx] = gated_res
                logger.info(f"成功在第 {next_layer_idx}(新索引:{new_next_idx}) 层添加门控残差连接")
        
        return self.gated_residuals
    
    def train_gated_residual(
        self,
        calibration_dataloader: DataLoader,
        num_epochs: int = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        lr_scheduler_type: Literal["cosine", "linear", "constant"] = "cosine",
        gradient_clip_val: Optional[float] = 1.0,
        early_stopping_patience: Optional[int] = None,
        device: Literal["cuda", "cpu"] = "cuda",
        log_file: Optional[str] = None,
    ) -> None:
        """
        训练门控残差连接的参数。
        
        Args:
            calibration_dataloader: 校准数据加载器
            num_epochs: 训练轮数
            learning_rate: 最大学习率
            weight_decay: AdamW优化器的权重衰减系数
            warmup_ratio: 预热阶段占总训练步数的比例
            lr_scheduler_type: 学习率调度策略类型，支持"cosine"(余弦退火)、"linear"(线性衰减)和"constant"(恒定)
            gradient_clip_val: 梯度裁剪阈值，None表示不进行梯度裁剪
            early_stopping_patience: 早停耐心值，None表示不使用早停
            device: 计算设备
            log_file: 日志文件路径
        """
        setup_logger(log_file)
        logger.info(f"开始训练门控残差参数，共 {len(self.gated_residuals)} 个门控模块")
        
        if not self.gated_residuals:
            logger.warning("没有门控残差模块，跳过训练")
            return
        
        # 冻结模型其他参数，只训练门控残差参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 设置门控残差参数为可训练
        for gated_res in self.gated_residuals.values():
            gated_res.gate.requires_grad = True
        
        # 创建优化器 - 使用AdamW替代Adam
        parameters = [gated_res.gate for gated_res in self.gated_residuals.values()]
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
        
        # 计算总训练步数
        total_steps = len(calibration_dataloader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        
        # 创建学习率调度器
        if lr_scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
            )
        elif lr_scheduler_type == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, 
                start_factor=1.0, 
                end_factor=0.1, 
                total_iters=total_steps - warmup_steps
            )
        else:  # constant
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=total_steps)
        
        # 创建预热调度器
        if warmup_steps > 0:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(
                        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
                    ),
                    scheduler
                ],
                milestones=[warmup_steps]
            )
        
        # 早停机制
        best_loss = float('inf')
        patience_counter = 0
        best_gate_values = {}
        
        # 训练循环
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch in tqdm(calibration_dataloader, desc=f"训练门控残差 Epoch {epoch+1}/{num_epochs}"):
                # 准备输入
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch.get("attention_mask", None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)
                    
                    # 前向传播
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids,
                        return_dict=True
                    )
                else:
                    # 如果batch是元组或列表
                    inputs = batch[0].to(device)
                    labels = batch[1].to(device) if len(batch) > 1 else inputs
                    
                    # 前向传播
                    outputs = self.model(inputs=inputs, labels=labels, return_dict=True)
                
                # 计算损失
                loss = outputs.loss
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                if gradient_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(parameters, gradient_clip_val)
                
                optimizer.step()
                scheduler.step()  # 更新学习率
                
                # 记录当前学习率
                current_lr = scheduler.get_last_lr()[0]
                
                total_loss += loss.item()
                num_batches += 1
            
            # 计算平均损失
            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch+1}/{num_epochs}, 平均损失: {avg_loss:.4f}, 学习率: {current_lr:.6f}")
            
            # 早停检查
            if early_stopping_patience is not None:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                    # 保存最佳门控值
                    best_gate_values = {layer_idx: gated_res.get_gate_value() 
                                       for layer_idx, gated_res in self.gated_residuals.items()}
                else:
                    patience_counter += 1
                    logger.info(f"早停计数: {patience_counter}/{early_stopping_patience}")
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"触发早停机制，在 {epoch+1} 轮停止训练")
                        # 恢复最佳门控值
                        for layer_idx, gate_value in best_gate_values.items():
                            with torch.no_grad():
                                # 反向计算gate参数值
                                inverse_sigmoid = torch.log(torch.tensor(gate_value) / (1 - torch.tensor(gate_value)))
                                self.gated_residuals[layer_idx].gate.data.fill_(inverse_sigmoid)
                        break
        
        # 打印训练后的门控值
        for layer_idx, gated_res in self.gated_residuals.items():
            gate_value = gated_res.get_gate_value()
            logger.info(f"第 {layer_idx} 层门控值: {gate_value:.4f}")
        
        # 设置模型为评估模式
        self.model.eval()
    
    def print_gate_values(self, log_file: Optional[str] = None) -> Dict[int, float]:
        """
        打印所有门控残差模块的门控值。
        
        Args:
            log_file: 日志文件路径
            
        Returns:
            门控值字典，键为层索引，值为门控值
        """
        setup_logger(log_file)
        gate_values = {}
        
        for layer_idx, gated_res in self.gated_residuals.items():
            gate_value = gated_res.get_gate_value()
            gate_values[layer_idx] = gate_value
            logger.info(f"第 {layer_idx} 层门控值: {gate_value:.4f}")
        
        return gate_values
    
    def remove_pruned_layers(self) -> None:
        """
        从模型中移除要剪枝的层。
        
        注意：在保存模型前调用此方法。
        """
        # 先恢复原始前向传播函数，避免保存带有门控残差的层
        self.restore_original_forwards()
        
        # 检查是否已经移除了冗余层
        num_layers = len(self.model.model.layers)
        layers_to_remove = [layer for layer in self.redundant_layers if layer < num_layers]
        
        if not layers_to_remove:
            logger.info("冗余层已被移除，无需再次移除")
            return
            
        # 按降序移除层，避免索引错误
        for layer_idx in sorted(layers_to_remove, reverse=True):
            try:
                logger.info(f"移除第 {layer_idx} 层")
                del self.model.model.layers[layer_idx]
            except IndexError:
                logger.warning(f"无法移除第 {layer_idx} 层，可能已被移除")
    
    def save_model(self, save_dir: str, model_name: str = "gated_model") -> str:
        """
        保存带有门控残差连接的模型。
        
        Args:
            save_dir: 保存目录
            model_name: 模型名称
            remove_layers: 是否在保存前移除冗余层
            
        Returns:
            保存的模型路径
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存原始模型
        model_path = os.path.join(save_dir, f"{model_name}.pt")
        torch.save(self.model.state_dict(), model_path)
        
        # 保存门控残差参数
        gated_residual_path = os.path.join(save_dir, f"{model_name}_gated_residual.pt")
        gated_residual_state = {
            "redundant_layers": self.redundant_layers,
            "gate_params": {layer_idx: gated_res.state_dict() for layer_idx, gated_res in self.gated_residuals.items()}
        }
        torch.save(gated_residual_state, gated_residual_path)
        
        logger.info(f"模型已保存到 {model_path}")
        logger.info(f"门控残差参数已保存到 {gated_residual_path}")
        
        return save_dir
    
    @classmethod
    def load_model(cls, model: nn.Module, save_dir: str, model_name: str = "gated_model", device: str = "cuda") -> "GatedResidualModel":
        """
        加载带有门控残差连接的模型。
        
        Args:
            model: 原始模型
            save_dir: 保存目录
            model_name: 模型名称
            device: 设备
            
        Returns:
            加载的GatedResidualModel实例
        """
        # 加载原始模型
        model_path = os.path.join(save_dir, f"{model_name}.pt")
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # 创建GatedResidualModel实例
        gated_model = cls(model)
        
        # 加载门控残差参数
        gated_residual_path = os.path.join(save_dir, f"{model_name}_gated_residual.pt")
        gated_residual_state = torch.load(gated_residual_path, map_location=device)
        
        # 恢复门控残差连接
        redundant_layers = gated_residual_state["redundant_layers"]
        gate_params = gated_residual_state["gate_params"]
        
        # 恢复层组信息
        if "layer_groups" in gated_residual_state:
            gated_model.layer_groups = gated_residual_state["layer_groups"]
        
        # 先应用门控残差连接
        gated_model.apply_gated_residual(redundant_layers, device=device)
        
        # 然后加载训练好的参数
        for layer_idx, gate_state in gate_params.items():
            if layer_idx in gated_model.gated_residuals:
                gated_model.gated_residuals[layer_idx].load_state_dict(gate_state)
        
        logger.info(f"成功加载带有门控残差连接的模型，共 {len(gated_model.gated_residuals)} 个门控模块")
        
        return gated_model
    
    def compute_bi(
        self,
        num_prune_layers: Optional[int] = 1,
        calibration_dataloader: Optional[DataLoader] = None,
        angular: bool = False,
        device: Literal["cpu", "cuda"] = "cuda",
        log_file: Optional[str] = None,
        *args, **kwargs
    ):
        """
        计算层重要性，用于自动选择要剪枝的层。
        
        Args:
            num_prune_layers: 要剪枝的层数
            calibration_dataloader: 校准数据加载器
            angular: 是否使用角度距离
            device: 计算设备
            log_file: 日志文件路径
            
        Returns:
            层重要性列表和要剪枝的层索引列表
        """
        setup_logger(log_file)
        
        # 导入必要的函数
        from modeling_grasp import block_influence
        import numpy as np
        
        # 确保模型在正确的设备上
        self.model = self.model.to(device)
        
        layer_importances = [0 for _ in self.model.model.layers]
        
        def compute_bi_hiddens(hiddens: List[torch.Tensor]):
            if not angular:
                num_prune_layers = 1

            for i in range(len(hiddens) - num_prune_layers):
                in_hidden = hiddens[i]
                out_hidden = hiddens[i+num_prune_layers]
                if angular:
                    # use only last token for angular distance as described in section 3.2
                    in_hidden = in_hidden[:,-1:]
                    out_hidden = out_hidden[:,-1:]
                
                layer_importances[i] += block_influence(
                    in_hidden,
                    out_hidden,
                    angular=angular
                ).mean().cpu().item()

        logger.info("=======>计算层重要性 (Block Influence)")
        assert calibration_dataloader is not None, "请提供校准数据加载器以计算层重要性"
        
        for batch in tqdm(calibration_dataloader, desc="计算BI", total=len(calibration_dataloader), leave=True):
            if isinstance(batch, dict):
                # 确保所有输入都在同一设备上
                input_ids = batch["input_ids"].to(device=device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device=device)
            else:
                attention_mask = None
                input_ids = batch[0].to(device=device)
                
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, output_hidden_states=True, return_dict=True)
            hiddens = outputs.hidden_states

            compute_bi_hiddens(hiddens=hiddens)
        
        if angular:
            start_layer = np.argsort(np.array(layer_importances[:-num_prune_layers+1]))[0]
            layers_to_remove = list(range(start_layer, start_layer + num_prune_layers))
        else:
            layers_to_remove = np.argsort(np.array(layer_importances))[:num_prune_layers].tolist()
        
        self.redundant_layers = layers_to_remove
        
        return layer_importances, layers_to_remove
    
    def restore_original_forwards(self):
        """
        恢复所有被修改的层的原始前向传播函数。
        """
        for layer_idx, original_forward in self._original_forwards.items():
            if layer_idx < len(self.model.model.layers):
                self.model.model.layers[layer_idx].forward = original_forward
        
        # 清空记录
        self._original_forwards = {}
        self.gated_residuals = {}
