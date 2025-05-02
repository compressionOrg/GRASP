export CUDA_VISIBLE_DEVICES=2
python shortgpt.py \
    --model_name_or_path meta-llama/Llama-2-13b-hf \
    --num_prune_layers 7 \
    --evaluate \
    --eval_ppl "wikitext2,ptb" \


    # --recovery \