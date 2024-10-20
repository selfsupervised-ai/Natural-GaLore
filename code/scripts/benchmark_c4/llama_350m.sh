# LLaMA-350M, Natural GaLore AdamW, 4 A100, 1 Node
python pytorchrun_main.py \
    --model_config configs/llama_350m.json \
    --lr 0.01 \
    --galore_scale 0.25 \
    --rank 256 \
    --update_proj_gap 200 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 60000 \
    --warmup_steps 6000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer natural_galore_adamw