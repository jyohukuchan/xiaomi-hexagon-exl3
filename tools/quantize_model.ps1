param([switch]$Resume, [int]$CalibrationRows = 32, [int]$CalibrationColumns = 1024)
$ErrorActionPreference = 'Stop'
$repoPath = Split-Path -Parent $PSScriptRoot
$docker = 'C:\Program Files\Docker\Docker\resources\bin\docker.exe'
$conversionArgs = @('python', '-u', '/workspace/tools/quantize_entry.py', '-w', '/workspace/models/quant-work')
if ($Resume) {
    $conversionArgs += '-r'
} else {
    $outputPath = Join-Path $repoPath 'models\index-translate-2b-exl3-4'
    if ((Test-Path -LiteralPath $outputPath) -and (Get-ChildItem -LiteralPath $outputPath | Select-Object -First 1)) {
        throw 'The conversion output exists. Use -Resume for an interrupted conversion.'
    }
    $conversionArgs += @('-i', '/workspace/models/index-translate-2b-bf16',
        '-o', '/workspace/models/index-translate-2b-exl3-4', '-b', '4', '-hb', '6',
        '-eb', '16', '-vb', '16', '-cb', 'mul1', '-cr', "$CalibrationRows", '-cc', "$CalibrationColumns")
}
& $docker run --name xiaomi-exl3-convert --gpus all --env PYTHONUNBUFFERED=1 `
    --volume "${repoPath}:/workspace" --workdir /opt/exllamav3 xiaomi-exl3-quantize:dev @conversionArgs
if ($LASTEXITCODE -ne 0) { throw 'Model conversion failed; the container and checkpoint are retained for inspection.' }
