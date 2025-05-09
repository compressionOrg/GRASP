import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import torch
import torch.nn as nn
import copy
import numpy as np
from copy import deepcopy

llama_path = 'meta-llama/Llama-2-13b-hf'
llama_model = AutoModelForCausalLM.from_pretrained(llama_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(llama_path, trust_remote_code=True)


INTERVAL = 2
MERGE_LAYERS = 8
HIGHEST_LAY = 40
LOWEST_LAY = 1
THRESHOLD = 0.58
lay = HIGHEST_LAY - MERGE_LAYERS
last_merge_flag = False

# 用于记录被合并的层
merged_layers = []


def merge_layers_return_model(model, merge_base_lay, merge_layer_num):

    merge_layer_num = min(merge_layer_num, len(model.model.layers) - merge_base_lay - 1)
    
    # 记录本次将要合并的层
    layers_to_merge = list(range(merge_base_lay+1, merge_base_lay+1+merge_layer_num))
    print(f"尝试将层 {merge_base_lay} 与层 {layers_to_merge} 合并")

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
    
    return model_copy


llama_copy_to_compress = copy.deepcopy(llama_model)


def cal_last_hidden_sim(model1, model2, tokenizer, sents):
    sim_ls = []
    for s in sents:
        encoded_inputs = tokenizer(s, return_tensors='pt')
        with torch.no_grad():
            # Disable cache during forward pass
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
    print("相似度:", sim_ls, "平均相似度:", np.mean(sim_ls))
    return np.mean(sim_ls)



sents = []
en_wiki_selected = ['Mouron () is a commune in the Arde',
 'The 81st Mechanised Brigade () is a mechanised brigade of the Romanian Land Force',
 'There are 18 National Natural Landmarks in the U.S. state of Washington, out of nearly',
 'Torreorgaz is a municipality in the',
 'Copa Libertadores 1973 was won by defending champions Independiente of A']

# zh_wiki_selected = ['月桃   \xa0\xa0月桃月桃属草本，单叶，互生，具',
#  '法国立贝尔洁白牙贴  目录产品成份：产品功效：用法用量：注意事项：产品禁忌：不良反应：规\xa0 \xa0 格：医疗器械注册号：产品执行标准：生产许可证号：授权监制：生产企业：',
#  'TIMKEN 641/632-B轴承  目录TIMK',
#  '天然碳化物质微结构研究  目录图书信息内容简介  图书信息作\u3000\u3000者： 冯有利 著 \n丛 书 名：\xa0\xa0出 版 社： 地质出版社 ISBN：9787116059771 出版时间',
#  'V字领衣服  目录基本信息']

sents.extend(en_wiki_selected)
# sents.extend(zh_wiki_selected)

# 创建一个字典来记录每次合并的信息
merge_history = {}
successful_merges = []

while lay >= LOWEST_LAY:
    print(f"当前处理层: {lay}")
    print(f"当前模型层数: {len(llama_copy_to_compress.model.layers)}")
    tmp_merged_model = merge_layers_return_model(llama_copy_to_compress, lay, MERGE_LAYERS-1)
    sim_value = cal_last_hidden_sim(llama_model, tmp_merged_model, tokenizer, sents)

    # 记录本次尝试合并的层和相似度
    layers_to_merge = list(range(lay+1, min(lay+MERGE_LAYERS, len(llama_copy_to_compress.model.layers))))
    merge_info = {
        "base_layer": lay,
        "merged_layers": layers_to_merge,
        "similarity": sim_value,
        "success": sim_value > THRESHOLD
    }
    merge_history[lay] = merge_info

    if sim_value > THRESHOLD:
        print(f"合并成功: 层 {lay} 与层 {layers_to_merge} 合并，相似度: {sim_value:.4f} > 阈值 {THRESHOLD}")
        llama_copy_to_compress = tmp_merged_model
        
        # 记录成功合并的层
        successful_merges.append({
            "base_layer": lay,
            "merged_layers": layers_to_merge,
            "similarity": sim_value
        })
        
        lay -= INTERVAL
        if lay >= len(llama_copy_to_compress.model.layers):
            lay = len(llama_copy_to_compress.model.layers) - 1 - MERGE_LAYERS
    else:
        print(f"合并失败: 层 {lay} 与层 {layers_to_merge} 合并，相似度: {sim_value:.4f} < 阈值 {THRESHOLD}")
        lay -= 1

llama_copy_to_compress.config.num_hidden_layers = len(llama_copy_to_compress.model.layers)

# 计算剪枝后的模型大小与原模型大小的比例
original_params = sum(p.numel() for p in llama_model.parameters())
compressed_params = sum(p.numel() for p in llama_copy_to_compress.parameters())
compression_ratio = compressed_params / original_params
print(f"原始模型参数量: {original_params:,}")
print(f"压缩后模型参数量: {compressed_params:,}")
print(f"压缩比例: {compression_ratio:.4f} ({compression_ratio*100:.2f}%)")
print(f"参数量减少: {(1-compression_ratio)*100:.2f}%")

# 输出所有成功合并的层信息
print("\n成功合并的层信息:")
for i, merge_info in enumerate(successful_merges):
    print(f"合并 {i+1}:")
    print(f"  基础层: {merge_info['base_layer']}")
    print(f"  被合并的层: {merge_info['merged_layers']}")
    print(f"  相似度: {merge_info['similarity']:.4f}")

# 计算总共合并了多少层
total_merged_layers = sum(len(info['merged_layers']) for info in successful_merges)
print(f"\n总计合并层数: {total_merged_layers}")
print(f"原始层数: {HIGHEST_LAY}")
print(f"剪枝后层数: {len(llama_copy_to_compress.model.layers)}")

# 保存模型
llama_copy_to_compress.save_pretrained("pruned_laco")
tokenizer.save_pretrained("pruned_laco")

# 将合并信息保存到文件
import json
with open("merge_info.json", "w") as f:
    json.dump({
        "successful_merges": successful_merges,
        "original_layers": HIGHEST_LAY,
        "compressed_layers": len(llama_copy_to_compress.model.layers),
        "compression_ratio": compression_ratio,
        "parameter_reduction": (1-compression_ratio)*100
    }, f, indent=2)

print("\n合并信息已保存到 merge_info.json")