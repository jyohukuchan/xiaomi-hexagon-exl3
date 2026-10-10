param([string]$Serial = '192.168.1.135:5555', [switch]$Push)
$ErrorActionPreference = 'Stop'
$repoPath = Split-Path -Parent $PSScriptRoot
$baselinePath = 'C:\coding-local\llama.cpp'
$docker = 'C:\Program Files\Docker\Docker\resources\bin\docker.exe'
$null = New-Item -ItemType Directory -Path (Join-Path $repoPath 'build-accelerator') -Force
$command = '/opt/android-ndk-r29/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O3 -Wall -Wextra -Wpedantic -Werror tools/accelerator_bench.cpp -I/baseline/ggml/include -I/baseline/vendor -L/baseline/pkg-adb/llama.cpp/lib -lggml -lggml-base -ldl -llog -Wl,-rpath,/data/local/tmp/llama.cpp/lib -o build-accelerator/accelerator_bench'
& $docker run --rm --volume "${repoPath}:/workspace" --volume "${baselinePath}:/baseline:ro" --workdir /workspace ghcr.io/snapdragon-toolchain/arm64-android:v0.7 bash -c $command
if ($LASTEXITCODE -ne 0) { throw 'Accelerator benchmark build failed' }
if ($Push) {
    $adb = 'C:\Android\platform-tools\platform-tools-latest-windows\platform-tools\adb.exe'
    . "$PSScriptRoot\adb_query.ps1"
    $pids = Invoke-ProjectAdbQuery -Adb $adb -Serial $Serial -Command 'pidof accelerator_bench test-backend-ops llama-server llama-completion gpu_npu_probe || true'
    if ($pids) { throw "Concurrent project process ($pids); refusing deployment" }
    & $adb -s $Serial shell mkdir -p /data/local/tmp/xiaomi-hexagon-exl3/accelerator-bench
    if ($LASTEXITCODE -ne 0) { throw 'Benchmark directory creation failed' }
    & $adb -s $Serial push "$repoPath\build-accelerator\accelerator_bench" /data/local/tmp/xiaomi-hexagon-exl3/accelerator-bench/
    if ($LASTEXITCODE -ne 0) { throw 'Benchmark deployment failed' }
    & $adb -s $Serial shell chmod 755 /data/local/tmp/xiaomi-hexagon-exl3/accelerator-bench/accelerator_bench
    if ($LASTEXITCODE -ne 0) { throw 'Benchmark chmod failed' }
}
