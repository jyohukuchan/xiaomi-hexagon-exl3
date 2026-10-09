param(
    [ValidateSet('Start', 'Stop', 'Status')][string]$Action = 'Start',
    [string]$Serial = '483fa196',
    [ValidateRange(1024, 65535)][int]$LocalPort = 8088
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$adb = 'C:\Android\platform-tools\platform-tools-latest-windows\platform-tools\adb.exe'
$docker = 'C:\Program Files\Docker\Docker\resources\bin\docker.exe'
$deviceRoot = '/data/local/tmp/xiaomi-hexagon-exl3'
$localState = Join-Path $repo 'benchmark-raw'
$keyFile = Join-Path $localState 'device-api.key'

function Get-ProjectProcesses {
    $processIds = & $adb -s $Serial shell pidof llama-server 2>$null
    foreach ($deviceProcessId in (($processIds -join ' ') -split '\s+')) {
        if ($deviceProcessId -notmatch '^\d+$') { continue }
        $cwd = & $adb -s $Serial shell readlink "/proc/$deviceProcessId/cwd" 2>$null
        if (($cwd -join '').Trim() -ne $deviceRoot) { continue }
        $command = & $adb -s $Serial shell "tr '\000' ' ' < /proc/$deviceProcessId/cmdline" 2>$null
        if (($command -join ' ') -match '^runtime/bin/llama-server .*models/index-translate-2b-exl3-v2\.hxgguf') {
            [int]$deviceProcessId
        }
    }
}

$running = @(Get-ProjectProcesses)
if ($Action -eq 'Stop') {
    foreach ($deviceProcessId in $running) {
        & $adb -s $Serial shell kill -INT $deviceProcessId
        if ($LASTEXITCODE -ne 0) { throw "Could not stop project process $deviceProcessId" }
    }
    $stopDeadline = (Get-Date).AddSeconds(15)
    while (@(Get-ProjectProcesses).Count -gt 0) {
        if ((Get-Date) -gt $stopDeadline) { throw 'Project server has not exited; inspect its current state before retrying' }
        Start-Sleep -Seconds 1
    }
    Write-Output 'Project server stopped; the model and API key are retained.'
    exit 0
}
if ($Action -eq 'Status') {
    [pscustomobject]@{ Running = $running.Count -gt 0; DeviceProcessIds = $running; BaseUrl = "http://127.0.0.1:$LocalPort/v1" }
    exit 0
}
if ($running.Count -gt 1) { throw 'Multiple project servers exist; inspect them before starting another' }
if ($running.Count -eq 0) {
    New-Item -ItemType Directory -Path $localState -Force | Out-Null
    if (-not (Test-Path -LiteralPath $keyFile)) {
        & $docker run --rm --volume "${repo}:/workspace" ghcr.io/snapdragon-toolchain/arm64-android:v0.7 openssl rand -out /workspace/benchmark-raw/device-api.key -hex 32
        if ($LASTEXITCODE -ne 0) { throw 'Local API key generation failed' }
    }
    if (([IO.File]::ReadAllText($keyFile).Trim()) -notmatch '^[0-9a-f]{64}$') { throw 'Invalid local API key file; preserve and inspect it' }
    & $adb -s $Serial push $keyFile "$deviceRoot/device-api.key"
    if ($LASTEXITCODE -ne 0) { throw 'API key deployment failed' }
    & $adb -s $Serial push (Join-Path $PSScriptRoot 'device_server.sh') "$deviceRoot/device_server.sh"
    if ($LASTEXITCODE -ne 0) { throw 'Server launcher deployment failed' }
    & $adb -s $Serial shell "chmod 600 $deviceRoot/device-api.key"
    if ($LASTEXITCODE -ne 0) { throw 'Could not restrict the device key permissions' }
    $bridge = Start-Process -FilePath $adb -ArgumentList @('-s', $Serial, 'shell', "sh $deviceRoot/device_server.sh > $deviceRoot/server.log 2>&1") `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $localState 'server-bridge.stdout.log') `
        -RedirectStandardError (Join-Path $localState 'server-bridge.stderr.log')
    Write-Output "Started local ADB bridge process $($bridge.Id)."
} elseif (-not (Test-Path -LiteralPath $keyFile)) {
    throw 'The project server is running but the host API key is missing; do not replace its live credentials'
}
$forwards = & $adb forward --list
if (($forwards -join "`n") -notmatch "(?m)^$([regex]::Escape($Serial)) tcp:$LocalPort tcp:8080\s*$") {
    & $adb -s $Serial forward --no-rebind "tcp:$LocalPort" tcp:8080
    if ($LASTEXITCODE -ne 0) { throw "Local port $LocalPort is occupied by a different forwarding rule" }
}
$readyDeadline = (Get-Date).AddSeconds(60)
while ((Get-Date) -lt $readyDeadline) {
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$LocalPort/health" -TimeoutSec 3
        if ($health.status -eq 'ok') {
            [pscustomobject]@{ BaseUrl = "http://127.0.0.1:$LocalPort/v1"; ApiKeyFile = $keyFile; DeviceProcessIds = @(Get-ProjectProcesses) }
            exit 0
        }
    } catch { }
    Start-Sleep -Seconds 1
}
throw 'Server did not become healthy within 60 seconds; inspect the live process and device server.log before retrying'
