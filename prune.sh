export CUDA_VISIBLE_DEVICES=0

python grasp.py --model_name_or_path meta-llama/Llama-2-7b-hf \
  --num_prune_layers 7 \
  --use_svd_compensation \
  --compensation_direction both \
  --continuous_layers_as_group \
  --compensation_ratio 0.9 \
  --skip_grasp_after_compensation \
  --evaluate \
  --eval_ppl "wikitext2" \
  
