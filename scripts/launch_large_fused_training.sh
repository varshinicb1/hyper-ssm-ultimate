#!/bin/bash
# Launch script for larger GPU runs of Hyper-SSM with Geometry-Aware Fusion enabled.
# Supports Accelerate (recommended), torchrun DDP, and single-GPU.

set -e

GPUS=${1:-4}
HIDDEN=${2:-768}
STEPS=${3:-25000}

echo "=== Launching Large Hyper-SSM + Geometry-Aware Fusion Training ==="
echo "GPUs: $GPUS | Hidden: $HIDDEN | Steps: $STEPS"

if command -v accelerate &> /dev/null && [ "$GPUS" -gt 1 ]; then
    echo "Using Hugging Face Accelerate for distributed + bf16..."
    accelerate launch --multi_gpu --num_processes $GPUS training/train_hybrid_ultimate.py \
        --use_geometry_fusion \
        --fusion_mode tangent_gated \
        --use_tiled \
        --hidden_size $HIDDEN \
        --num_layers 24 \
        --max_steps $STEPS \
        --batch 4 \
        --seq_len 2048 \
        --precision bf16 \
        --log_interval 100 \
        --use_accelerate \
        --config configs/large_fused_hyper_ssm_aether.yaml
elif [ "$GPUS" -gt 1 ]; then
    echo "Using torchrun for DDP..."
    torchrun --nproc_per_node=$GPUS training/train_hybrid_ultimate.py \
        --use_geometry_fusion \
        --fusion_mode tangent_gated \
        --use_tiled \
        --hidden_size $HIDDEN \
        --num_layers 24 \
        --max_steps $STEPS \
        --batch 4 \
        --seq_len 2048 \
        --precision bf16 \
        --log_interval 100
else
    echo "Single GPU / CPU run..."
    python training/train_hybrid_ultimate.py \
        --use_geometry_fusion \
        --fusion_mode tangent_gated \
        --use_tiled \
        --hidden_size $HIDDEN \
        --max_steps $STEPS \
        --batch 4 \
        --seq_len 1024 \
        --precision bf16 \
        --log_interval 50
fi

echo "Launch complete. Monitor logs for fused vs non-fused metrics."