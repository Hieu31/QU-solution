param(
    [string]$OutputRoot = "dist\qu-solution-offline"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$outputPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputRoot))
$distRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot "dist"))

if (-not $outputPath.StartsWith($distRoot + [System.IO.Path]::DirectorySeparatorChar)) {
    throw "OutputRoot must resolve inside $distRoot"
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker is required to build Linux wheels from Windows. Install/start Docker Desktop, then rerun."
}

if (Test-Path -LiteralPath $outputPath) {
    Remove-Item -LiteralPath $outputPath -Recurse -Force
}

$projectPath = Join-Path $outputPath "project"
$wheelhousePath = Join-Path $outputPath "wheelhouse"
New-Item -ItemType Directory -Path $projectPath, $wheelhousePath -Force | Out-Null

$excluded = @(
    ".git", ".venv", ".pytest_cache", "data", "artifacts", "logs",
    "tmp", "deliverables", "dist", "unused", "__pycache__"
)
Get-ChildItem -LiteralPath $repoRoot -Force |
    Where-Object { $excluded -notcontains $_.Name } |
    ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $projectPath -Recurse -Force
    }

Copy-Item -LiteralPath (Join-Path $repoRoot "offline\requirements-kaggle.txt") `
    -Destination (Join-Path $outputPath "requirements-offline.txt") -Force

$repoDocker = $repoRoot.Replace("\", "/")
$outputDocker = $outputPath.Replace("\", "/")

docker run --rm `
    --mount "type=bind,source=$repoDocker,target=/workspace,readonly" `
    --mount "type=bind,source=$outputDocker,target=/bundle" `
    python:3.11-slim `
    sh -lc "python -m pip wheel --wheel-dir /bundle/wheelhouse --requirement /workspace/offline/requirements-kaggle.txt && python -m pip wheel --no-deps --wheel-dir /bundle/wheelhouse /workspace"

if ($LASTEXITCODE -ne 0) {
    throw "Linux wheel build failed with exit code $LASTEXITCODE"
}

$commit = git -C $repoRoot rev-parse HEAD
$wheels = @(Get-ChildItem -LiteralPath $wheelhousePath -Filter "*.whl")
if ($wheels.Count -eq 0) {
    throw "Wheelhouse is empty"
}
if (@($wheels | Where-Object { $_.Name -like "qu_solution-*.whl" }).Count -ne 1) {
    throw "Expected exactly one qu_solution wheel"
}

$marker = [ordered]@{
    schema_version = 1
    profile = "qu-solution-offline/linux-python311"
    git_commit = $commit.Trim()
    python = "3.11"
    wheel_count = $wheels.Count
    built_with = "python:3.11-slim"
}
$marker | ConvertTo-Json -Depth 4 | Set-Content `
    -LiteralPath (Join-Path $outputPath "offline-bundle.json") -Encoding utf8

$archivePath = "$outputPath.zip"
if (Test-Path -LiteralPath $archivePath) {
    Remove-Item -LiteralPath $archivePath -Force
}
Compress-Archive -LiteralPath $outputPath -DestinationPath $archivePath -CompressionLevel Optimal

Write-Host "Offline bundle directory: $outputPath"
Write-Host "Uploadable archive: $archivePath"
Write-Host "Wheels: $($wheels.Count)"
