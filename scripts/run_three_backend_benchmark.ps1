param(
    [string]$RunId = "$(Get-Date -Format 'yyyyMMdd-HHmmss')-production-v1",
    [string]$DataDir = "data\reparos\reparos-production-v1\base",
    [string]$WebSpellModel = "artifacts\osm-model-production-v4-leakfree",
    [string]$OpenNmtCheckpoint = "artifacts\opennmt-production-kaggle-t4-v1\reparos_base_step_25000.pt",
    [string]$TokenizerModel = "artifacts\tokenizer-production-v1\tokenizer.model",
    [string]$DecodingConfig = "artifacts\opennmt-production-kaggle-t4-v1\decoding-config.json",
    [string]$CTranslate2Model = "artifacts\opennmt-production-kaggle-t4-v1-ctranslate2-float32",
    [ValidateSet("cpu", "cuda")]
    [string]$CTranslate2Device = "cpu",
    [string]$CTranslate2ComputeType = "float32"
)

$ErrorActionPreference = "Stop"
$runDir = "benchmark\$RunId"

function Invoke-BenchmarkCommand {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & uv run benchmark-three @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "benchmark-three failed with exit code $LASTEXITCODE"
    }
}

Write-Host "Benchmark run: $RunId"
Write-Host "Output: $runDir"

Invoke-BenchmarkCommand prepare `
    --data $DataDir `
    --split test `
    --output "$runDir\gold.jsonl"

Invoke-BenchmarkCommand run-webspell `
    --gold "$runDir\gold.jsonl" `
    --model $WebSpellModel `
    --output "$runDir\predictions\webspell.jsonl"

Invoke-BenchmarkCommand run-opennmt `
    --gold "$runDir\gold.jsonl" `
    --checkpoint $OpenNmtCheckpoint `
    --tokenizer $TokenizerModel `
    --decoding-config $DecodingConfig `
    --output "$runDir\predictions\opennmt.jsonl"

Invoke-BenchmarkCommand run-ctranslate2 `
    --gold "$runDir\gold.jsonl" `
    --model $CTranslate2Model `
    --decoding-config $DecodingConfig `
    --device $CTranslate2Device `
    --compute-type $CTranslate2ComputeType `
    --output "$runDir\predictions\ctranslate2.jsonl"

Invoke-BenchmarkCommand parity `
    --opennmt "$runDir\predictions\opennmt.jsonl" `
    --ctranslate2 "$runDir\predictions\ctranslate2.jsonl" `
    --output "$runDir\parity.json"

Invoke-BenchmarkCommand score `
    --gold "$runDir\gold.jsonl" `
    --prediction "webspell=$runDir\predictions\webspell.jsonl" `
    --prediction "opennmt=$runDir\predictions\opennmt.jsonl" `
    --prediction "ctranslate2=$runDir\predictions\ctranslate2.jsonl" `
    --output "$runDir\quality.json"

Invoke-BenchmarkCommand resources `
    --prediction "webspell=$runDir\predictions\webspell.jsonl" `
    --prediction "opennmt=$runDir\predictions\opennmt.jsonl" `
    --prediction "ctranslate2=$runDir\predictions\ctranslate2.jsonl" `
    --artifact "webspell=$WebSpellModel" `
    --artifact "opennmt=$OpenNmtCheckpoint" `
    --artifact "ctranslate2=$CTranslate2Model" `
    --output "$runDir\resources.json"

Write-Host ""
Write-Host "Benchmark completed: $runDir" -ForegroundColor Green
Write-Host "Report: $runDir\report.md"
Get-Content "$runDir\report.md"
