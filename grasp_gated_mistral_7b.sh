export CUDA_VISIBLE_DEVICES=0

python grasp_gated.py --model_name mistralai/Mistral-7B-v0.3 \
  --auto_select \
  --num_prune_layers 7 \
  --evaluate \
  --eval_ppl "wikitext2,ptb" \
  --num_epochs 5000 \

# meta-llama/Llama-3.1-8B

# mistralai/Mistral-7B-v0.3