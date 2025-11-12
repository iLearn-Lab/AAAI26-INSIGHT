# pip install math_verify # reward function
# pip install -U trl
# GPU memory: 96GiB
# register customized plugin in external_plugins file

CUDA_VISIBLE_DEVICES=2,3,4,5,0,1 NPROC_PER_NODE=6 MAX_PIXELS=50176 swift rlhf \
    --rlhf_type grpo \
    --model .../model_Qwen2.5-VL-7B-Instruct \
    --model_type qwen2_5_vl \
    --resume_only_model false   \
    --external_plugins .../plugin_new_in_log_intention.py \
    --reward_funcs external_action_intent external_action_cont_lin external_action_cont_pow external_intent_score external_intent_score_raw soft_overlong \
    --reward_weights 0.9 0 0 0 0 0.1 \
    --train_type full \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset '.../data_Ego4D/...' \
    --val_dataset '.../data_Ego4D/...' \
    --max_completion_length 450 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 5 \
    --learning_rate 3e-6 \
    --gradient_accumulation_steps 5 \
    --eval_strategy steps \
    --eval_steps   101\
    --save_steps  100 \
    --val_dataset_shuffle true  \
    --save_total_limit 3 \
    --logging_steps 10 \
    --max_length 4000 \
    --soft_cache_length     256 \
    --output_dir .../output \
    --warmup_ratio 0.1 \
    --kl_coef 0.08 \
    --dataloader_num_workers 12 \
    --dataset_num_proc 4 \
    --num_generations 5 \
    --temperature 0.9 \
    --deepspeed zero2 \
    --log_completions true  \
    --attn_impl flash_attn  \
