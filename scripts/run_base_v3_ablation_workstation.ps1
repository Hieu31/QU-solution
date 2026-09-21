# PowerShell Execution Script for ReparoS Base V3 Ablation on RTX 6000 PRO (Blackwell 96GB)
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "   REPAROS BASE V3 ABLATION RUNNER (RTX 6000 PRO / WORKSTATION)      " -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Ensure UV environment is active
$env:PYTHONUNBUFFERED = "1"

# 2. Configure high throughput parameters for 96GB VRAM
$BATCH_SIZE = 65536
$BUCKET_SIZE = 131072
$NUM_WORKERS = 8

Write-Host "`n[1/3] Generating OpenNMT Configs for RTX 6000 PRO..." -ForegroundColor Yellow
uv run python scripts/prepare_base_v3_ablation_configs.py `
    --batch-size $BATCH_SIZE `
    --bucket-size $BUCKET_SIZE `
    --num-workers $NUM_WORKERS `
    --gpu-rank 0

Write-Host "`n[2/3] Executing 3-Way Ablation Pre-training from Scratch (20,000 steps each)..." -ForegroundColor Yellow
uv run python scripts/run_base_v3_ablation.py `
    --runs dataset_run_a dataset_run_b dataset_run_c `
    --batch-size $BATCH_SIZE `
    --train-steps 20000 `
    --device cuda

Write-Host "`n[3/3] Ablation experiments completed! Results saved to data/ablation_comparison_report.json" -ForegroundColor Green
