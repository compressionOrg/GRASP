# 🚀GRASP: Gradient-based Retention of Adaptive Singular Parameters

**The official code for the paper** 📃 "[Rethinking Layer Removal: A Hybrid Pruning Framework Combining Layer Removal and Singular Value Selection for Efficient LLM Compression](https://arxiv.org/abs/2501.00339)".

![GRASP](./assets/GRASP.png)

In this paper, we propose **GRASP (Gradient-based Retention of Adaptive Singular Parameters)**, which preserves representation-critical singular values to mitigate representation instability caused by pure layer removal. By selectively applying singular value decomposition (SVD) to redundant layers, GRASP achieves efficient compression while maintaining representation stability with minimal overhead.



## 📊 Evaluation Results

| Model           | Openb. | ARC_e | WinoG. | HellaS. | ARC_c | PIQA | MathQA | Average |
| --------------- | ------ | ----- | ------ | ------- | ----- | ---- | ------ | ------- |
| LLaMA-7B        | 0.28   | 0.67  | 0.67   | 0.56    | 0.38  | 0.78 | 0.27   | 0.52    |
| **GRASP-5.5B**  | 0.22   | 0.52  | 0.64   | 0.43    | 0.32  | 0.70 | 0.24   | 0.44    |
| **GRASP\*-5.5B** | 0.24   | 0.59  | 0.63   | 0.50    | 0.35  | 0.73 | 0.25   | 0.47    |
| LLaMA2-7B       | 0.32   | 0.69  | 0.67   | 0.57    | 0.40  | 0.78 | 0.28   | 0.53    |
| **GRASP-5.5B**  | 0.24   | 0.54  | 0.63   | 0.43    | 0.33  | 0.71 | 0.23   | 0.44    |
| **GRASP\*-5.5B** | 0.28   | 0.61  | 0.65   | 0.50    | 0.36  | 0.72 | 0.25   | 0.48    |

- **GRASP** denotes compressing dense model without post-training for recovery
- **GRASP*** denotes compressing dense model with post-training on Alpaca dataset for 1 epoch



## 🎯Quick Start

### 🔗Clone this repo

```sh
git clone https://github.com/LyoAI/GRASP.git
cd GRASP
```

### ⚙️Install dependencies

```sh
# step 1: create a virtual environment
conda create -n grasp python=3.9

# step 2: activate the virtual environment
conda activate grasp

# step 3: install dependencies from requirements.txt
pip install torch==2.3.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Laco

Laco pruning 

```
/bin/bash  scripts/run_laco.sh
```

```
python laco.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --merge_layers ${NUM_PRUNE_LAYERS} \
    --threshold ${THRESHOLD} \
    --highest_lay ${HIGHTEST_LAY} \
    --lowest_lay ${LOWEST_LAY} \
    --evaluate \
    --tasks $TASKS \
    --eval_ppl $EVAL_PPL \
    --log_file $LOG_FILE \
```

evaluate zero-shot and PPL

```
/bin/bash  scripts/eval_laco_llama2_7b.sh
```

```
python evaluate.py \
    --model_path $MODEL_PATH \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --tasks $TASKS \
    --eval_ppl $EVAL_PPL \
    --batch_size $BATCH_SIZE \
    --device $DEVICE \
    --log_file $LOG_FILE \
```
