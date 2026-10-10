param(
    [ValidateSet('interop', 'interop-finish', 'dequant')][string]$Mode = 'interop',
    [ValidateRange(1, 1000)][int]$Repetitions = 31,
    [string]$Serial = '483fa196',
    [switch]$CudaFixtures
)
$ErrorActionPreference = 'Stop'
$repoPath = Split-Path -Parent $PSScriptRoot
$adb = 'C:\Android\platform-tools\platform-tools-latest-windows\platform-tools\adb.exe'
$devicePath = '/data/local/tmp/xiaomi-hexagon-exl3/gpu-probe'
. "$PSScriptRoot\adb_query.ps1"
$activePids = Invoke-ProjectAdbQuery -Adb $adb -Serial $Serial -Command 'pidof llama-server llama-completion trace_runtime gpu_npu_probe || true'
if ($activePids) { throw "Concurrent inference/probe process detected ($activePids); run an isolated measurement" }
$fixtureArgs = ''
if ($CudaFixtures) {
    if ($Mode -ne 'dequant') { throw 'CUDA fixtures apply only to dequant mode' }
    foreach ($fixture in @(@('real-k-proj', 'real-k-proj.bin'), @('real-head', 'real-head.bin'))) {
        $source = Join-Path $repoPath "benchmark-raw\$($fixture[0])\model.bin"
        if (-not (Test-Path -LiteralPath $source)) { throw "Missing private CUDA fixture: $source" }
        & $adb -s $Serial push $source "$devicePath/$($fixture[1])"
        if ($LASTEXITCODE -ne 0) { throw 'Fixture deployment failed' }
    }
    $fixtureArgs = ' real-k-proj.bin real-head.bin'
}
$logName = "$Mode-$([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()).log"
$command = "cd $devicePath && export ADSP_LIBRARY_PATH=$devicePath && ./gpu_npu_probe $Mode $Repetitions$fixtureArgs > $logName 2>&1"
$runError = $null
try {
    # Timeout kills only the ADB client; unknown remote state is reported, never
    # followed by an automatic library overwrite or global ADB restart.
    $null = Invoke-ProjectAdbQuery -Adb $adb -Serial $Serial -Command $command -TimeoutMs 60000
} catch { $runError = $_ }
$outputDirectory = Join-Path $repoPath 'benchmark-raw'
$null = New-Item -ItemType Directory -Path $outputDirectory -Force
$outputPath = Join-Path $outputDirectory "gpu-$logName"
& $adb -s $Serial pull "$devicePath/$logName" $outputPath
if ($LASTEXITCODE -ne 0) { throw 'Log retrieval failed' }
Write-Host "Saved $outputPath"
if ($runError) { throw $runError }
$auditArgs = @("$PSScriptRoot\check_gpu_probe.py", $outputPath)
if ($CudaFixtures) { $auditArgs += '--require-fixtures' }
& python @auditArgs
if ($LASTEXITCODE -ne 0) { throw 'GPU probe log validation failed' }
