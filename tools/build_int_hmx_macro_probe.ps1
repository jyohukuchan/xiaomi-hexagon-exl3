param([string]$Serial = '192.168.1.135:5555', [switch]$Push)
$ErrorActionPreference = 'Stop'
$repoPath = Split-Path -Parent $PSScriptRoot
$addonPath = Join-Path $repoPath 'build-int-hmx\hexkl-6.4-beta1\hexkl_addon'
if (-not (Test-Path -LiteralPath "$addonPath\LICENSE.txt")) { throw 'External HexKL addon is missing; see its separate license before use' }
$docker = 'C:\Program Files\Docker\Docker\resources\bin\docker.exe'
$command = '/opt/android-ndk-r29/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O3 -Wall -Wextra -Werror tools/int_hmx_macro_probe.cpp -isystem /hexkl/include -isystem /opt/hexagon/6.6.0.0/incs -isystem /opt/hexagon/6.6.0.0/incs/stddef -I/llama/vendor -L/hexkl/lib/armv8_android26 -L/opt/hexagon/6.6.0.0/ipc/fastrpc/remote/ship/android_aarch64 -lsdkl -lcdsprpc -ldl -llog -Wl,-rpath,/data/local/tmp/xiaomi-hexagon-exl3/int-hmx-probe -o build-int-hmx/int_hmx_macro_probe'
& $docker run --rm --volume "${repoPath}:/workspace" --volume "${addonPath}:/hexkl:ro" --volume 'C:\coding-local\llama.cpp:/llama:ro' --workdir /workspace ghcr.io/snapdragon-toolchain/arm64-android:v0.7 bash -c $command
if ($LASTEXITCODE -ne 0) { throw 'Integer HMX macro probe build failed' }
if ($Push) {
    $adb = 'C:\Android\platform-tools\platform-tools-latest-windows\platform-tools\adb.exe'
    . "$PSScriptRoot\adb_query.ps1"
    $pids = Invoke-ProjectAdbQuery -Adb $adb -Serial $Serial -Command 'pidof int_hmx_macro_probe int_hmx_micro_probe homura_baseline llama-server llama-completion accelerator_bench gpu_npu_probe test-backend-ops || true'
    if ($pids) { throw "Concurrent project process: $pids" }
    $devicePath = '/data/local/tmp/xiaomi-hexagon-exl3/int-hmx-probe'
    & $adb -s $Serial shell mkdir -p $devicePath
    if ($LASTEXITCODE -ne 0) { throw 'Probe directory creation failed' }
    foreach ($source in @("$repoPath\build-int-hmx\int_hmx_macro_probe", "$addonPath\lib\armv8_android26\libsdkl.so", "$addonPath\lib\hexagon_toolv19_v75\libhexkl_skel.so", "$addonPath\LICENSE.txt")) {
        & $adb -s $Serial push $source "$devicePath/"
        if ($LASTEXITCODE -ne 0) { throw "Probe deployment failed: $source" }
    }
    & $adb -s $Serial shell chmod 755 "$devicePath/int_hmx_macro_probe"
    if ($LASTEXITCODE -ne 0) { throw 'Probe chmod failed' }
}
