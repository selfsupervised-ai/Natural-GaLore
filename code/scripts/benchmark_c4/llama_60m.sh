# LLaMA-60M, GaLore-Adam, 1 A100, 1 Node
python pytorchrun_main.py \
    --model_config configs/llama_60m.json \
    --batch_size 256 \
    --total_batch_size 512 \
    --optimizer natural_galore_adamw \
    --lr 0.01 \
    --rank 128 \
    --update_proj_gap 200 \
    --galore_scale 0.25 \
    --weight_decay 0.01 \
    --warmup_steps 1000 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --num_training_steps 10000 \
