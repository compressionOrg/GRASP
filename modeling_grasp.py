import torch
import logging
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Literal, Optional, List, Union, Tuple
from tools.utils_func import block_influence, adaptive_rank_selection
import tensorly as tl
from tensorly.decomposition import tucker

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

class SVDLinear(nn.Module):
    def __init__(self, U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor, bias: Optional[torch.Tensor], sigma_fuse: Literal["UV", "U", "V"] = "UV"):
        '''
        **__Args__:**
            U: Left Singular Vectors after rank truncation, which is shape of [rank, out_features]
            S: Diagonal Matrix of singular values, which is shape of [rank, rank]
            Vh: Right Singular Vectors after rank truncation, which is shape of [in_features, rank]
            bias: bias
        '''
        super(SVDLinear, self).__init__()
        
        in_features = Vh.shape[1]
        out_features = U.shape[0]
        hidden_size = S.shape[0]

        self.InLinear = nn.Linear(in_features=in_features, out_features=hidden_size, bias=False)
        self.OutLinear = nn.Linear(in_features=hidden_size, out_features=out_features, bias=True if bias is not None else False)

        if bias is not None:
            self.OutLinear.bias.data = bias
        
        if sigma_fuse == "UV":
            self.InLinear.weight.data = Vh.mul(S.sqrt().view(-1, 1)).contiguous()
            self.OutLinear.weight.data = U.mul(S.sqrt()).contiguous()
        elif sigma_fuse == "U":
            self.InLinear.weight.data = Vh.contiguous()
            self.OutLinear.weight.data = U.mul(S).contiguous()
        elif sigma_fuse == "V":
            self.InLinear.weight.data = Vh.mul(S.view(-1, 1)).contiguous()
        else:
            raise ValueError(f"value of sigma_fuse {sigma_fuse} not support")
    
    def forward(self, x: torch.Tensor):
        output = self.OutLinear(self.InLinear(x))
        return output


class GRASPLayer(nn.Module):
    def __init__(self, U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor, bias: Optional[torch.Tensor], compression_ratio: Optional[float]):
        super(GRASPLayer, self).__init__()
        self.U = nn.Parameter(U.clone().detach().requires_grad_(False))
        self.S = nn.Parameter(S.clone().detach().requires_grad_(True))
        self.Vh = nn.Parameter(Vh.clone().detach().requires_grad_(False))

        self.in_features = self.Vh.shape[1]
        self.out_features = self.U.shape[0]

        self.bias = bias
        self.compression_ratio = compression_ratio

    def forward(self, x: torch.Tensor):
        b, s, d = x.shape
        sigma = torch.diag(self.S)
        W_reconstructed =  torch.mm(self.U, torch.mm(sigma, self.Vh))
        return torch.mm(x.view(b*s, -1), W_reconstructed.t()).view(b, s, -1)


class TuckerLinear(nn.Module):
    def __init__(self, core: torch.Tensor, factors: List[torch.Tensor], bias: Optional[torch.Tensor]):
        '''
        **__Args__:**
            core: 核心张量，形状为 [rank_out, rank_in]
            factors: 因子矩阵列表 [U, V]，其中 U 形状为 [out_features, rank_out]，V 形状为 [in_features, rank_in]
            bias: 偏置向量
        '''
        super(TuckerLinear, self).__init__()
        
        self.in_features = factors[1].shape[0]
        self.out_features = factors[0].shape[0]
        self.rank_in = factors[1].shape[1]
        self.rank_out = factors[0].shape[1]
        
        # 创建输入映射层
        self.InLinear = nn.Linear(in_features=self.in_features, out_features=self.rank_in, bias=False)
        # 创建核心映射层
        self.CoreLinear = nn.Linear(in_features=self.rank_in, out_features=self.rank_out, bias=False)
        # 创建输出映射层
        self.OutLinear = nn.Linear(in_features=self.rank_out, out_features=self.out_features, bias=True if bias is not None else False)
        
        # 初始化权重
        self.InLinear.weight.data = factors[1].t().contiguous()
        self.CoreLinear.weight.data = core.contiguous()
        self.OutLinear.weight.data = factors[0].contiguous()
        
        if bias is not None:
            self.OutLinear.bias.data = bias
    
    def forward(self, x: torch.Tensor):
        output = self.OutLinear(self.CoreLinear(self.InLinear(x)))
        return output


class GRASPLayer(nn.Module):
    def __init__(self, U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor, bias: Optional[torch.Tensor], compression_ratio: Optional[float]):
        super(GRASPLayer, self).__init__()
        self.U = nn.Parameter(U.clone().detach().requires_grad_(False))
        self.S = nn.Parameter(S.clone().detach().requires_grad_(True))
        self.Vh = nn.Parameter(Vh.clone().detach().requires_grad_(False))

        self.in_features = self.Vh.shape[1]
        self.out_features = self.U.shape[0]

        self.bias = bias
        self.compression_ratio = compression_ratio

    def forward(self, x: torch.Tensor):
        b, s, d = x.shape
        sigma = torch.diag(self.S)
        W_reconstructed =  torch.mm(self.U, torch.mm(sigma, self.Vh))
        return torch.mm(x.view(b*s, -1), W_reconstructed.t()).view(b, s, -1)


class TuckerGRASPLayer(nn.Module):
    def __init__(self, core: torch.Tensor, factors: List[torch.Tensor], bias: Optional[torch.Tensor], compression_ratio: Optional[float]):
        super(TuckerGRASPLayer, self).__init__()
        
        # 核心张量，可训练
        self.core = nn.Parameter(core.clone().detach().requires_grad_(True))
        # 因子矩阵，不可训练
        self.factor_0 = nn.Parameter(factors[0].clone().detach().requires_grad_(False))
        self.factor_1 = nn.Parameter(factors[1].clone().detach().requires_grad_(False))
        
        self.in_features = factors[1].shape[0]
        self.out_features = factors[0].shape[0]
        self.rank_in = factors[1].shape[1]
        self.rank_out = factors[0].shape[1]
        
        self.bias = bias
        self.compression_ratio = compression_ratio
    
    def forward(self, x: torch.Tensor):
        b, s, d = x.shape
        # 重构权重矩阵
        W_reconstructed = torch.mm(self.factor_0, torch.mm(self.core, self.factor_1.t()))
        return torch.mm(x.view(b*s, -1), W_reconstructed.t()).view(b, s, -1)


class GRASPModel(nn.Module):
    def __init__(self, model: nn.Module, *args, **kwargs) -> None:
        super(GRASPModel, self).__init__(*args, **kwargs)
        self.model = model
        for params in self.model.parameters():
            params.requires_grad = False

        self.grasp_values_dict = {}
    
    def calculate_layer_compression_ratio(self, redundant_layers: Optional[List] = None):
        '''
        calculate module-wise compression ratio
        for module_name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and "lm_head" not in module_name:
                module.compression_ratio = self_define_ratio
        '''
        # compression_ratios = {}  # To store layer compression ratios for verification/debugging
        
        # for module_name, module in self.model.named_modules():
        #     if isinstance(module, nn.Linear) and "lm_head" not in module_name:
        #         # Extract layer index from module_name, assuming a consistent naming pattern
        #         layer_index = self._extract_layer_index(module_name)
        #         if layer_index is not None:
        #             if layer_index in redundant_layers:
        #                 module.compression_ratio = 0.8
        #             else:
        #                 module.compression_ratio = 0.1
        #             compression_ratios[module_name] = module.compression_ratio
        
        # return compression_ratios  # Optional, for debugging
        pass

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

    def remove_layers(self, layers_to_remove: Optional[List[int]] = [], angular: Optional[bool] = False, num_prune_layers: Optional[int] = None):
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

    def _set_module(self, model, submodule_key, module):
        tokens = submodule_key.split('.')
        sub_model = model
        for token in tokens[:-1]:
            sub_model = getattr(sub_model, token)
        setattr(sub_model, tokens[-1], module)

    def replace_with_GRASPLayer(self, target_layer: str, device: Literal["cuda", "cpu"] = "cuda", log_file: Optional[str] = None, decomposition: Literal["svd", "tucker"] = "svd", tucker_ranks: Optional[Tuple[int, int]] = None):
        setup_logger(log_file=log_file)
        replace_flag = False
        module = self.model.get_submodule(target=target_layer)
        if isinstance(module, nn.Linear):
            w = module.weight.data
            bias = module.bias
            compression_ratio = getattr(module, "compression_ratio", None)
            
            if decomposition == "svd":
                # 使用SVD分解
                U, S, Vh = torch.linalg.svd(w.to(device=device), full_matrices=False)
                grasp_layer = GRASPLayer(U=U, S=S, Vh=Vh, bias=bias, compression_ratio=compression_ratio)
            elif decomposition == "tucker":
                # 使用Tucker分解
                # 将权重矩阵重塑为2D张量
                w_tensor = w.to(device=device)
                
                # 如果没有指定秩，则根据压缩率计算
                if tucker_ranks is None and compression_ratio is not None:
                    # 计算Tucker分解的秩
                    total_params = w.shape[0] * w.shape[1]
                    # 假设两个维度的压缩率相同
                    rank_ratio = np.sqrt(1 - compression_ratio)
                    rank_out = max(1, int(w.shape[0] * rank_ratio))
                    rank_in = max(1, int(w.shape[1] * rank_ratio))
                    tucker_ranks = (rank_out, rank_in)
                elif tucker_ranks is None:
                    # 默认使用较小的秩
                    rank_out = max(1, w.shape[0] // 4)
                    rank_in = max(1, w.shape[1] // 4)
                    tucker_ranks = (rank_out, rank_in)
                
                # 设置tensorly后端为pytorch
                tl.set_backend('pytorch')
                
                # 执行Tucker分解
                core, factors = tucker(w_tensor, rank=tucker_ranks)
                
                # 确保core和factors是PyTorch张量
                if not isinstance(core, torch.Tensor):
                    core = torch.tensor(core, device=device)
                factors = [torch.tensor(f, device=device) if not isinstance(f, torch.Tensor) else f for f in factors]
                
                grasp_layer = TuckerGRASPLayer(core=core, factors=factors, bias=bias, compression_ratio=compression_ratio)
            else:
                raise ValueError(f"不支持的分解方法: {decomposition}")
            
            self._set_module(self.model, target_layer, grasp_layer)
            replace_flag = True
        else:
            raise TypeError(f"目标层应该是Linear模块，但得到了 {type(module)}")
        
        if not replace_flag:
            logger.info(f"替换为GRASPLayer失败，在模型中找不到目标层: {target_layer}")
            return
    
    def compress_block(
            self,
            layer_id: int,
            block_type: Literal["attention", "mlp"],
            target_layer_types: Union[List[str], str] = ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
            device: Literal["cuda", "cpu"] = "cuda",
            allocation_aware: Optional[bool] = None,
            verbose: bool = False,
            log_file: Optional[str] = None,
            decomposition: Literal["svd", "tucker"] = "tucker",
            tucker_ranks: Optional[Tuple[int, int]] = None
        ):
        '''
        使用GRASP压缩基于Transformer的LLM中的一个Transformer块
        '''
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
                        self.replace_with_GRASPLayer(target_layer=target_layer, device=device, decomposition=decomposition, tucker_ranks=tucker_ranks)
            if np.all(np.array(compression_ratio_list) == 0):
                return True
        else:
            for target_layer in target_layer_names:
                self.replace_with_GRASPLayer(target_layer=target_layer, device=device, decomposition=decomposition, tucker_ranks=tucker_ranks)
        return False
    
    def get_tucker_layer_gradients(self, calibration_dataloader: DataLoader, device: Literal["cuda:0", "cpu"] = "cuda:0", log_file: Optional[str] = None, *args, **kwargs):
        setup_logger(log_file=log_file)
        grasp_layer_names = self.check_exists_grasp_layer()
        if grasp_layer_names is None:
            raise NotImplementedError("GRASPLayer未找到，无法计算梯度，请先使用GRASPModel.replace_with_GRASPLayer")

        iterator = tqdm(calibration_dataloader, desc="梯度收集", total=len(calibration_dataloader), leave=True)
        grasp_layer_grads = {}
        self.model.to(device=device)
        for batch_idx, batch in enumerate(iterator):
            if len(batch) == 2:
                attention_mask = None
            else:
                attention_mask = batch["attention_mask"].to(device=device)
            input_ids = batch["input_ids"].to(device=device)
            labels = batch["labels"].to(device=device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
            loss = outputs[0]

            # 清除梯度缓存
            self.model.zero_grad()

            # 反向传播
            loss.backward()

            for grasp_layer_name in grasp_layer_names:
                module = self.model.get_submodule(grasp_layer_name)
                if not module:
                    raise ValueError("找不到模块")
                
                if isinstance(module, GRASPLayer):
                    # SVD分解的梯度
                    if grasp_layer_name not in grasp_layer_grads:
                        grasp_layer_grads[grasp_layer_name] = module.S.grad
                    else:
                        grasp_layer_grads[grasp_layer_name] += module.S.grad
                elif isinstance(module, TuckerGRASPLayer):
                    # Tucker分解的梯度
                    if grasp_layer_name not in grasp_layer_grads:
                        grasp_layer_grads[grasp_layer_name] = module.core.grad
                    else:
                        grasp_layer_grads[grasp_layer_name] += module.core.grad

            if "cuda" in device:
                torch.cuda.empty_cache()

        self.grasp_layer_grads = grasp_layer_grads
        return grasp_layer_grads
    
    def dynamic_tucker_selection(
            self,
            grasp_layer_grads: dict,
            metric: Literal["gradient", "taylor"] = "taylor",
            compression_ratio: Optional[float] = None,
            threshold_ratio: Optional[float] = None,
            verbose: Optional[bool] = False,
            log_file: Optional[str] = None
        ):
        setup_logger(log_file=log_file)
        if not grasp_layer_grads:
            grasp_layer_grads = self.grasp_layer_grads
            raise ValueError("应提供grasp_layer的梯度，但得到了None")

        indices_dict = {}
        rank_dict = {}  # 添加rank_dict定义

        for grasp_layer_name, grasp_layer_grad in grasp_layer_grads.items():
            module = self.model.get_submodule(grasp_layer_name)
            
            if isinstance(module, GRASPLayer):
                # 处理SVD分解的情况
                S = module.S
                if metric == "gradient":
                    svd_importance: torch.Tensor = torch.abs(grasp_layer_grad)
                elif metric == "taylor":
                    svd_importance: torch.Tensor = torch.abs(grasp_layer_grad * S)
                else:
                    raise RuntimeError(f"不支持的度量方法 {metric}")

                if module.compression_ratio is not None:
                    compression_ratio = module.compression_ratio

                if compression_ratio is not None:            
                    k = self.compute_preserve_rank(module, compression_ratio=compression_ratio)
                    _, indices = torch.topk(svd_importance, k=k)
                else:
                    assert threshold_ratio, "请提供Taylor阈值以自适应选择秩"
                    indices = adaptive_rank_selection(svd_importance_list=svd_importance, target_ratio=threshold_ratio)
                indices_dict[grasp_layer_name] = indices
                
            elif isinstance(module, TuckerGRASPLayer):
                # 处理Tucker分解的情况
                core = module.core
                if metric == "gradient":
                    # 使用核心张量的梯度绝对值作为重要性度量
                    tucker_importance = torch.abs(grasp_layer_grad)
                elif metric == "taylor":
                    # 使用核心张量的梯度与值的乘积作为重要性度量
                    tucker_importance = torch.abs(grasp_layer_grad * core)
                else:
                    raise RuntimeError(f"不支持的度量方法 {metric}")
                
                if module.compression_ratio is not None:
                    compression_ratio = module.compression_ratio
                
                # 计算要保留的元素数量
                if compression_ratio is not None:
                    total_elements = core.numel()
                    k = int(total_elements * (1 - compression_ratio))
                    # 将核心张量展平并找到最重要的k个元素
                    flat_importance = tucker_importance.reshape(-1)
                    _, flat_indices = torch.topk(flat_importance, k=k)
                    # 转换回原始维度的索引
                    indices = []
                    for idx in flat_indices:
                        # 确保idx是一个整数
                        idx_val = idx.item() if isinstance(idx, torch.Tensor) else idx
                        # 确保不会越界
                        if core.shape[1] > 0:
                            i = idx_val // core.shape[1]
                            j = idx_val % core.shape[1]
                            indices.append((i, j))
                        else:
                            logger.warning(f"核心张量的第二维度为0，无法计算多维索引: {core.shape}")
                            continue
                else:
                    assert threshold_ratio, "请提供Taylor阈值以自适应选择核心张量元素"
                    # 将核心张量展平并应用自适应选择
                    flat_importance = tucker_importance.reshape(-1)
                    flat_indices = adaptive_rank_selection(svd_importance_list=flat_importance, target_ratio=threshold_ratio)
                    # 转换回原始维度的索引
                    indices = []
                    for idx in flat_indices:
                        # 确保idx是一个整数
                        idx_val = idx.item() if isinstance(idx, torch.Tensor) else idx
                        # 确保不会越界
                        if core.shape[1] > 0:
                            i = idx_val // core.shape[1]
                            j = idx_val % core.shape[1]
                            indices.append((i, j))
                        else:
                            logger.warning(f"核心张量的第二维度为0，无法计算多维索引: {core.shape}")
                            continue
                
                indices_dict[grasp_layer_name] = indices
                
                # 保存重要性值和核心张量值用于调试
                self.grasp_values_dict[grasp_layer_name] = {}
                self.grasp_values_dict[grasp_layer_name]["tucker_importance"] = torch.round(tucker_importance.cpu(), decimals=3).tolist()
                self.grasp_values_dict[grasp_layer_name]["core_value"] = torch.round(core.data.cpu(), decimals=3).tolist()

        if verbose:
            logger.info("+" * 100)
            for grasp_layer_name, indices in indices_dict.items():
                logger.info(f"{grasp_layer_name}")
                logger.info(str(indices)[:1000])  # 只显示部分索引以避免输出过长
            logger.info("+" * 100)

        self.indices_dict = indices_dict
        return indices_dict
    
    def compile_tucker_model(
        self,
        indices_dict: Optional[dict] = None,
        merge: Optional[bool] = False,
        device: Literal["cpu", "cuda"] = "cuda",
        log_file: Optional[str] = None
    ):
        setup_logger(log_file=log_file)
        if indices_dict is None:
            indices_dict = self.indices_dict

        rank_dict = {}

        for grasp_layer_name, indices in indices_dict.items():
            module = self.model.get_submodule(grasp_layer_name)
            
            if isinstance(module, GRASPLayer):
                # 处理SVD分解的情况
                S = module.S[indices]
                U = module.U[:, indices]
                Vh = module.Vh[indices, :]
                bias = module.bias

                rank_dict[grasp_layer_name] = S.shape[0]

                if merge:
                    in_features = Vh.shape[1]
                    out_features = U.shape[0]
                    self._set_module(self.model, grasp_layer_name, nn.Linear(in_features=in_features, out_features=out_features, bias=True if bias is not None else False))
                    linear_layer: nn.Linear = self.model.get_submodule(grasp_layer_name)

                    # 重新初始化线性层的权重和偏置
                    W_compressed = torch.mm(U, torch.mm(torch.diag(S), Vh))
                    linear_layer.weight.data = W_compressed

                    if bias is not None:
                        linear_layer.bias = bias
                    
                    linear_layer.requires_grad_(False)
                else:
                    self._set_module(self.model, grasp_layer_name, SVDLinear(U=U, S=S, Vh=Vh, bias=bias, sigma_fuse="UV"))
                    svd_linear_layer: SVDLinear = self.model.get_submodule(grasp_layer_name)
                    svd_linear_layer.requires_grad_(False)
                
            elif isinstance(module, TuckerGRASPLayer):
                # 处理Tucker分解的情况
                core = module.core
                factors = [module.factor_0, module.factor_1]  # 重新构建factors列表
                bias = module.bias
                
                # 创建稀疏核心张量
                sparse_core = torch.zeros_like(core)
                for i, j in indices:
                    sparse_core[i, j] = core[i, j]
                
                rank_dict[grasp_layer_name] = len(indices)
                
                if merge:
                    # 直接合并为一个线性层
                    in_features = factors[1].shape[0]
                    out_features = factors[0].shape[0]
                    self._set_module(self.model, grasp_layer_name, nn.Linear(in_features=in_features, out_features=out_features, bias=True if bias is not None else False))
                    linear_layer: nn.Linear = self.model.get_submodule(grasp_layer_name)
                    
                    # 重构权重矩阵
                    W_compressed = torch.mm(factors[0], torch.mm(sparse_core, factors[1].t()))
                    linear_layer.weight.data = W_compressed
                    
                    if bias is not None:
                        linear_layer.bias = bias
                    
                    linear_layer.requires_grad_(False)
                else:
                    # 使用TuckerLinear替代
                    self._set_module(self.model, grasp_layer_name, TuckerLinear(core=sparse_core, factors=factors, bias=bias))
                    tucker_linear_layer: TuckerLinear = self.model.get_submodule(grasp_layer_name)
                    tucker_linear_layer.requires_grad_(False)
            
            del module
            if "cuda" in device:
                torch.cuda.empty_cache()
        
        return


    def check_exists_grasp_layer(self):
        """
        检查模型中是否存在 GRASP 层，并返回这些层的名称列表。
        
        Returns:
            List[str]: GRASP 层的名称列表，如果没有找到则返回空列表
        """
        grasp_layer_names = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, (GRASPLayer, TuckerGRASPLayer)):
                grasp_layer_names.append(name)
        
        if not grasp_layer_names:
            logger.warning("未找到任何 GRASP 层，请先使用 replace_with_GRASPLayer 方法替换层")
            return []
        
        logger.info(f"找到 {len(grasp_layer_names)} 个 GRASP 层: {grasp_layer_names}")
        return grasp_layer_names
    
    def compute_preserve_rank(self, module, compression_ratio):
        """
        根据压缩率计算需要保留的秩
        
        Args:
            module: GRASPLayer模块
            compression_ratio: 压缩率
            
        Returns:
            保留的秩k
        """
        if isinstance(module, GRASPLayer):
            # SVD分解情况
            original_rank = module.S.shape[0]
            compressed_rank = max(1, int(original_rank * (1 - compression_ratio)))
            return compressed_rank
        elif isinstance(module, TuckerGRASPLayer):
            # Tucker分解情况
            core = module.core
            total_elements = core.numel()
            preserved_elements = max(1, int(total_elements * (1 - compression_ratio)))
            return preserved_elements
        else:
            raise TypeError(f"不支持的模块类型: {type(module)}")