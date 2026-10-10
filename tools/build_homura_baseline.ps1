param([string]$Serial = '192.168.1.135:5555', [switch]$Push)
$ErrorActionPreference = 'Stop'
$repoPath = Split-Path -Parent $PSScriptRoot
$baselinePath = 'C:\coding-local\llama.cpp'
$docker = 'C:\Program Files\Docker\Docker\resources\bin\docker.exe'
$null = New-Item -ItemType Directory -Force -Path "$repoPath\build-homura-baseline"
$command = '/opt/android-ndk-r29/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O3 -Wall -Wextra -Wpedantic -Werror tools/homura_baseline.cpp -I/baseline/include -I/baseline/ggml/include -I/baseline/vendor -L/baseline/pkg-adb/llama.cpp/lib -lllama -lggml -lggml-base -ldl -llog -Wl,-rpath,/data/local/tmp/llama.cpp/lib -o build-homura-baseline/homura_baseline'
& $docker run --rm --volume "${repoPath}:/workspace" --volume "${baselinePath}:/baseline:ro" --workdir /workspace ghcr.io/snapdragon-toolchain/arm64-android:v0.7 bash -c $command
if ($LASTEXITCODE -ne 0) { throw 'Baseline build failed' }
if ($Push) {
    $adb = 'C:\Android\platform-tools\platform-tools-latest-windows\platform-tools\adb.exe'
    . "$PSScriptRoot\adb_query.ps1"
    $pids = Invoke-ProjectAdbQuery -Adb $adb -Serial $Serial -Command 'pidof homura_baseline llama-server llama-completion accelerator_bench || true'
    if ($pids) { throw "Concurrent project processes: $pids" }
    & $adb -s $Serial shell mkdir -p /data/local/tmp/xiaomi-hexagon-exl3/homura-baseline
    if ($LASTEXITCODE -ne 0) { throw 'Baseline directory creation failed' }
    & $adb -s $Serial push "$repoPath\build-homura-baseline\homura_baseline" /data/local/tmp/xiaomi-hexagon-exl3/homura-baseline/
    if ($LASTEXITCODE -ne 0) { throw 'Baseline binary push failed' }
    & $adb -s $Serial push "$repoPath\tools\prompts\homura-baseline.txt" /data/local/tmp/xiaomi-hexagon-exl3/homura-baseline/prompt.txt
    if ($LASTEXITCODE -ne 0) { throw 'Baseline prompt push failed' }
    & $adb -s $Serial shell chmod 755 /data/local/tmp/xiaomi-hexagon-exl3/homura-baseline/homura_baseline
    if ($LASTEXITCODE -ne 0) { throw 'Baseline chmod failed' }
}
