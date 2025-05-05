import torch

def block_influence(
    input_hidden_state: torch.Tensor,
    output_hidden_state: torch.Tensor,
    angular=False,
):
    """
    input_hidden_state: B, S, D
    output_hidden_state: B, S, D
    """
    _, _, d = input_hidden_state.shape
    input_hidden_state = input_hidden_state.reshape(-1, d)
    output_hidden_state = output_hidden_state.reshape(-1, d)

    norm_input = input_hidden_state.norm(dim=-1, keepdim=True)
    norm_output = output_hidden_state.norm(dim=-1, keepdim=True)

    sim = (input_hidden_state @ output_hidden_state.T) / (norm_input * norm_output)
    sim = sim.diagonal().nan_to_num(nan=0.5)

    if angular:
        return (torch.arccos(sim) / torch.pi)

    return 1 - sim


def jaccard_similarity(list1, list2):
    if isinstance(list1, list):
        list1 = set(list1)
    
    if isinstance(list2, list):
        list2 = set(list2)
    intersection = list1.intersection(list2)
    union = list1.union(list2)

    if len(union) > 0:
        overlap_ratio = len(intersection) / len(union)
    else:
        overlap_ratio = 0

    return overlap_ratio


def adaptive_rank_selection(svd_importance_list, target_ratio):
    # 确保输入是张量或列表
    if isinstance(svd_importance_list, torch.Tensor):
        svd_importance_list = svd_importance_list.detach().cpu()
    
    # 转换为列表，处理可能的NaN值
    importance_values = []
    for val in svd_importance_list:
        if isinstance(val, torch.Tensor):
            val = val.item()
        if not (torch.isnan(torch.tensor(val)) if isinstance(val, float) else False):
            importance_values.append(val)
        else:
            importance_values.append(0.0)
    
    # 计算总和
    total_sum = sum(importance_values)
    if total_sum <= 0:
        # 如果总和小于等于0，返回前几个索引
        return [i for i in range(min(10, len(importance_values)))]
    
    target_sum = total_sum * target_ratio

    # 按重要性排序
    sorted_list = sorted(enumerate(importance_values), key=lambda x: -x[1])

    cumulative_sum = 0
    indices = []
    for index, value in sorted_list:
        if value > 0:  # 只考虑正值
            cumulative_sum += value
            indices.append(index)
            if cumulative_sum >= target_sum:
                break
    
    # 确保至少选择一个元素
    if not indices and len(importance_values) > 0:
        indices = [sorted_list[0][0]]
        
    return indices