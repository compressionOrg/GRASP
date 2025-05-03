export CUDA_VISIBLE_DEVICES=0

python grasp_gated.py --model_name meta-llama/Llama-2-7b-hf \
  --auto_select \
  --num_prune_layers 7 \
  --evaluate \
  --eval_ppl "wikitext2,ptb" \
  --num_epochs 5000 \
