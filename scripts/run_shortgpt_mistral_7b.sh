export CUDA_VISIBLE_DEVICES=2
python shortgpt.py \
    --model_name_or_path mistralai/Mistral-7B-v0.3 \
    --num_prune_layers 7 \
    --evaluate \
    --eval_ppl "wikitext2,ptb" \


    # --recovery \

# meta-llama/Llama-3.1-8B

# mistralai/Mistral-7B-v0.3

