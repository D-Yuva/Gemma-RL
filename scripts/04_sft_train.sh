#!/bin/bash
# 04_sft_train.sh - Run SFT Training via MLX-LM
# Hardware: optimized for Apple Silicon (M4 Pro)

# 1. Free memory (important for M4 Pro 24GB)
echo "------------------------------------------------"
echo "Phase 1: Supervised Fine-Tuning (SFT)"
echo "------------------------------------------------"
echo "[IMPORTANT] Make sure LM Studio / Ollama are CLOSED to free up VRAM."
echo "Current memory pressure check..."
memory_pressure

# 2. Setup directories
mkdir -p logs checkpoints/sft

# 3. Run Training
# We use the config values by pulling them via a small python helper
echo "Starting training on google/gemma-4-E4B-it..."
python -m mlx_lm.lora \
    --model google/gemma-4-E4B-it \
    --train \
    --data data/splits \
    --batch-size 1 \
    --iters 5000 \
    --learning-rate 2e-5 \
    --num-layers 16 \
    --lora-rank 32 \
    --lora-scale 64.0 \
    --val-batches 25 \
    --steps-per-eval 500 \
    --save-every 500 \
    --adapter-path checkpoints/sft \
    --max-seq-length 512 \
    --grad-checkpoint \
    2>&1 | tee logs/sft.log

echo "------------------------------------------------"
echo "SFT Training Complete."
echo "Checkpoints saved in: checkpoints/sft/"
echo "Log saved in: logs/sft.log"
echo "------------------------------------------------"
