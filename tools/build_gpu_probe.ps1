param([string]$Serial = '483fa196', [switch]$Push)
$ErrorActionPreference = 'Stop'
$repoPath = Split-Path -Parent $PSScriptRoot
$docker = 'C:\Program Files\Docker\Docker\resources\bin\docker.exe'
$dspCommands = 'cmake -G Ninja -S src/interop -B build-gpu-dsp -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=/opt/hexagon/6.6.0.0/build/cmake/hexagon_toolchain.cmake -DHEXAGON_SDK_ROOT=/opt/hexagon/6.6.0.0 -DHEXAGON_TOOLS_ROOT=/opt/hexagon/6.6.0.0/tools/HEXAGON_Tools/19.0.07 -DDSP_VERSION=v75 -DPREBUILT_LIB_DIR=toolv19_v75 -DNO_WRAP_MEM_API=ON && cmake --build build-gpu-dsp -j 4'
& $docker run --rm --volume "${repoPath}:/workspace" --workdir /workspace ghcr.io/snapdragon-toolchain/arm64-android:v0.7 bash -c $dspCommands
if ($LASTEXITCODE -ne 0) { throw 'GPU interop DSP build failed' }
$commands = 'cmake -G Ninja -S . -B build-gpu-host -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=/opt/android-ndk-r29/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 -DEXL3_BUILD_HEXAGON=ON -DEXL3_BUILD_GPU_PROBE=ON -DHEXAGON_SDK_ROOT=/opt/hexagon/6.6.0.0 -DPREBUILT_LIB_DIR=android_aarch64 && cmake --build build-gpu-host --target gpu_npu_probe -j 4'
& $docker run --rm --volume "${repoPath}:/workspace" --workdir /workspace ghcr.io/snapdragon-toolchain/arm64-android:v0.7 bash -c $commands
if ($LASTEXITCODE -ne 0) { throw 'GPU probe build failed' }
if ($Push) {
    $adb = 'C:\Android\platform-tools\platform-tools-latest-windows\platform-tools\adb.exe'
    . "$PSScriptRoot\adb_query.ps1"
    $probePids = Invoke-ProjectAdbQuery -Adb $adb -Serial $Serial -Command 'pidof gpu_npu_probe || true'
    if ($probePids) { throw "Probe is still running ($probePids); refusing to overwrite mapped files" }
    & $adb -s $Serial shell mkdir -p /data/local/tmp/xiaomi-hexagon-exl3/gpu-probe
    if ($LASTEXITCODE -ne 0) { throw 'Probe directory creation failed' }
    & $adb -s $Serial push "$repoPath\build-gpu-dsp\libexl3_gpu_htp.so" /data/local/tmp/xiaomi-hexagon-exl3/gpu-probe/
    if ($LASTEXITCODE -ne 0) { throw 'Interop DSP deployment failed' }
    & $adb -s $Serial push "$repoPath\src\gpu\exl3_dequant.cl" /data/local/tmp/xiaomi-hexagon-exl3/gpu-probe/
    if ($LASTEXITCODE -ne 0) { throw 'OpenCL source deployment failed' }
    & $adb -s $Serial push "$repoPath\build-gpu-host\gpu_npu_probe" /data/local/tmp/xiaomi-hexagon-exl3/gpu-probe/
    if ($LASTEXITCODE -ne 0) { throw 'Probe deployment failed' }
    & $adb -s $Serial shell chmod 755 /data/local/tmp/xiaomi-hexagon-exl3/gpu-probe/gpu_npu_probe
    if ($LASTEXITCODE -ne 0) { throw 'Probe chmod failed' }
}
