param([string]$Serial = '192.168.1.135:5555', [switch]$Push)
$ErrorActionPreference = 'Stop'
$repoPath = Split-Path -Parent $PSScriptRoot
$addonPath = Join-Path $repoPath 'build-int-hmx\hexkl-6.4-beta1\hexkl_addon'
if (-not (Test-Path -LiteralPath "$addonPath\LICENSE.txt")) { throw 'Separate HexKL dependency/license missing' }
$docker = 'C:\Program Files\Docker\Docker\resources\bin\docker.exe'
$dsp = 'cmake -G Ninja -S src/int-hmx -B build-int-hmx/dsp -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=/opt/hexagon/6.6.0.0/build/cmake/hexagon_toolchain.cmake -DHEXAGON_SDK_ROOT=/opt/hexagon/6.6.0.0 -DHEXAGON_TOOLS_ROOT=/opt/hexagon/6.6.0.0/tools/HEXAGON_Tools/19.0.07 -DDSP_VERSION=v75 -DPREBUILT_LIB_DIR=toolv19_v75 -DNO_WRAP_MEM_API=ON && cmake --build build-int-hmx/dsp -j 4'
& $docker run --rm --volume "${repoPath}:/workspace" --volume "${addonPath}:/hexkl:ro" --workdir /workspace ghcr.io/snapdragon-toolchain/arm64-android:v0.7 bash -c $dsp
if ($LASTEXITCODE -ne 0) { throw 'Micro DSP build failed' }
$hostBuild = 'cmake -G Ninja -S src/int-hmx -B build-int-hmx/host -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=/opt/android-ndk-r29/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 -DHEXAGON_SDK_ROOT=/opt/hexagon/6.6.0.0 -DPREBUILT_LIB_DIR=android_aarch64 && cmake --build build-int-hmx/host -j 4'
& $docker run --rm --volume "${repoPath}:/workspace" --volume 'C:\coding-local\llama.cpp:/baseline:ro' --workdir /workspace ghcr.io/snapdragon-toolchain/arm64-android:v0.7 bash -c $hostBuild
if ($LASTEXITCODE -ne 0) { throw 'Micro host build failed' }
if ($Push) {
    $adb = 'C:\Android\platform-tools\platform-tools-latest-windows\platform-tools\adb.exe'
    . "$PSScriptRoot\adb_query.ps1"
    $pids = Invoke-ProjectAdbQuery -Adb $adb -Serial $Serial -Command 'pidof int_hmx_macro_probe int_hmx_micro_probe homura_baseline llama-server llama-completion accelerator_bench gpu_npu_probe test-backend-ops || true'
    if ($pids) { throw "Concurrent project process: $pids" }
    $devicePath = '/data/local/tmp/xiaomi-hexagon-exl3/int-hmx-micro-probe'
    & $adb -s $Serial shell mkdir -p $devicePath
    if ($LASTEXITCODE -ne 0) { throw 'Probe directory creation failed' }
    foreach ($source in @("$repoPath\build-int-hmx\host\int_hmx_micro_probe", "$repoPath\build-int-hmx\dsp\libint_hmx_htp.so", "$addonPath\LICENSE.txt")) {
        & $adb -s $Serial push $source "$devicePath/"
        if ($LASTEXITCODE -ne 0) { throw "Deployment failed: $source" }
    }
    & $adb -s $Serial shell chmod 755 "$devicePath/int_hmx_micro_probe"
    if ($LASTEXITCODE -ne 0) { throw 'Probe chmod failed' }
}
