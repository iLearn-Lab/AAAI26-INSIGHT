# Explicit Cognitive Reasoning for Anticipation

This repo provides code for **Explicit Cognitive Reasoning** in egocentric long-term action anticipation.  
It fine-tunes **Qwen2.5-VL-7B-Instruct** using **GRPO** in **ms-swift** to:
1. reason about visual context  
2. infer user intention  
3. predict upcoming actions  

Training uses reward functions for **intention alignment**, **action consistency**, and **structured output**:
```
<think>…</think><intention>…</intention><answer>…</answer>
```

## Key Files
```
run_external_reward_func_7B_qwen_intention.sh  
plugin_in_log_intention.py  
my_rewards_intention.py  
cal_ED.py  
```

### Reward Function
The reward function includes multiple components:
- **Length Reward** (`s_len`)
- **Accuracy Reward** (`s_cont`, based on Damerau–Levenshtein distance)
- **Intention Reward** (`s_int`, using MiniLM embeddings)
- **Tags-Order Reward and Language Consistency Reward** (`s_fmt`, `s_lang`)

The final reward is computed as:  
`R = s_len * (0.85*s_cont + 0.05*s_int + 0.05*s_lang + 0.05*s_fmt)`



## Data Format
Each sample follows ms-swift RLHF/GRPO chat style:
- `messages`: conversation turns (can include visuals)
- Ground truth (`gt_answer` or `solution`) with tags:
  ```
  <think>...</think>
  <intention>...</intention>
  <answer>verb noun, verb noun, …</answer>
  ```
Actions are lowercase pairs (e.g., *open fridge, take bottle*).

## Training Steps

### 1. Place Plugins
```
examples/train/grpo/plugin/
├── plugin_in_log_intention.py
└── my_rewards_intention.py
```
In `my_rewards_intention.py`:
```python
model_path = '/path/to/all-MiniLM-L6-v2'
use_half = True
```

### 2. Launch Training
Edit and run:
```bash
bash run_external_reward_func_7B_qwen_intention.sh
```

## Evaluation
Run:
```bash
python cal_ED.py
```
Computes **normalized edit distance** (Damerau–Levenshtein) for **action**, **verb**, and **noun** similarity — lower is better.
