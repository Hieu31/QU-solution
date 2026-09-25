# PowerShell Execution Script for ReparoS Base V3 (2E / 2D Arm E) Production Training on RTX 6000 PRO (96GB VRAM)
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "   REPAROS BASE V3 (2E / 2D ARM E) TRAINER (RTX 6000 PRO / WORKSTATION) " -ForegroundColor Cyan
Write-Host "   Architecture: 2 Enc / 2 Dec · Hidden 128 · FFN 2048 · 7.1M Params   " -ForegroundColor Cyan
Write-Host "   Winning Recipe: 4,000,000 Pairs (Run C: 40% L1, 25% L2, 25% L3, 10% L4)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Environment configuration
$env:PYTHONUNBUFFERED = "1"

# 2. Configure maximum throughput for RTX 6000 PRO (96GB VRAM)
$BATCH_SIZE = 65536
$BUCKET_SIZE = 131072
$NUM_WORKERS = 8
$TRAIN_STEPS = 50000

Write-Host "`n[1/3] Auditing 4,000,000 pairs dataset for zero-leakage and zero-contradictions..." -ForegroundColor Yellow
uv run python scripts/audit_base_v3_dataset.py --ablation-dir data/base_v3_production

Write-Host "`n[2/3] Executing Production Pre-training from Scratch ($TRAIN_STEPS steps, 2E/2D Arm E)..." -ForegroundColor Yellow
uv run python scripts/run_base_v3_production.py `
    --data-dir data/base_v3_production `
    --enc-layers 2 `
    --dec-layers 2 `
    --batch-size $BATCH_SIZE `
    --bucket-size $BUCKET_SIZE `
    --num-workers $NUM_WORKERS `
    --train-steps $TRAIN_STEPS `
    --device cuda

Write-Host "`n[3/3] Base V3 Production Pre-training completed successfully!" -ForegroundColor Green
