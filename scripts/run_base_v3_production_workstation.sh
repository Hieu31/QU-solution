#!/usr/bin/env bash
set -euo pipefail

# Bash Execution Script for ReparoS Base V3 (2E / 2D Arm E) Production Training on RTX 6000 PRO (96GB VRAM)
echo "======================================================================"
echo "   REPAROS BASE V3 (2E / 2D ARM E) TRAINER (RTX 6000 PRO / WORKSTATION) "
echo "   Architecture: 2 Enc / 2 Dec · Hidden 128 · FFN 2048 · 7.1M Params   "
echo "   Winning Recipe: 4,000,000 Pairs (Run C: 40% L1, 25% L2, 25% L3, 10% L4)"
echo "======================================================================"

export PYTHONUNBUFFERED=1

# Configure maximum throughput for RTX 6000 PRO (96GB VRAM)
BATCH_SIZE=65536
BUCKET_SIZE=131072
NUM_WORKERS=8
TRAIN_STEPS=50000

echo -e "\n[1/3] Auditing 4,000,000 pairs dataset for zero-leakage and zero-contradictions..."
uv run python scripts/audit_base_v3_dataset.py --ablation-dir data/base_v3_production

echo -e "\n[2/3] Executing Production Pre-training from Scratch ($TRAIN_STEPS steps, 2E/2D Arm E)..."
uv run python scripts/run_base_v3_production.py \
    --data-dir data/base_v3_production \
    --enc-layers 2 \
    --dec-layers 2 \
    --batch-size "$BATCH_SIZE" \
    --bucket-size "$BUCKET_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --train-steps "$TRAIN_STEPS" \
    --device cuda

echo -e "\n[3/3] Base V3 (2E/2D Arm E) Production Pre-training completed successfully!"
