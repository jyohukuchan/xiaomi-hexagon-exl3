param([string]$Serial = '483fa196', [switch]$Push)
$ErrorActionPreference = 'Stop'
$repoPath = Split-Path -Parent $PSScriptRoot
$docker = 'C:\Program Files\Docker\Docker\resources\bin\docker.exe'
$sdk = '/opt/hexagon/6.6.0.0'
$dspBuild = 'build-npu-dsp'
$hostBuild = 'build-npu-host'
$commands = "cmake -G Ninja -S src/hexagon -B $dspBuild -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=$sdk/build/cmake/hexagon_toolchain.cmake -DHEXAGON_SDK_ROOT=$sdk -DHEXAGON_TOOLS_ROOT=$sdk/tools/HEXAGON_Tools/19.0.07 -DDSP_VERSION=v75 -DPREBUILT_LIB_DIR=toolv19_v75 -DNO_WRAP_MEM_API=ON && cmake --build $dspBuild -j 4 && cmake -G Ninja -S . -B $hostBuild -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=/opt/android-ndk-r29/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 -DEXL3_BUILD_HEXAGON=ON -DHEXAGON_SDK_ROOT=$sdk -DPREBUILT_LIB_DIR=android_aarch64 && cmake --build $hostBuild -j 4"
& $docker run --rm --volume "${repoPath}:/workspace" --workdir /workspace ghcr.io/snapdragon-toolchain/arm64-android:v0.7 bash -c $commands
if ($LASTEXITCODE -ne 0) { throw 'NPU probe build failed' }
if ($Push) {
    $adb = 'C:\Android\platform-tools\platform-tools-latest-windows\platform-tools\adb.exe'
    & $adb -s $Serial shell mkdir -p /data/local/tmp/xiaomi-hexagon-exl3
    if ($LASTEXITCODE -ne 0) { throw 'Device directory creation failed' }
    & $adb -s $Serial push "$repoPath\$dspBuild\libexl3_htp.so" /data/local/tmp/xiaomi-hexagon-exl3/
    if ($LASTEXITCODE -ne 0) { throw 'DSP deployment failed' }
    & $adb -s $Serial push "$repoPath\$hostBuild\exl3_npu_probe" /data/local/tmp/xiaomi-hexagon-exl3/
    if ($LASTEXITCODE -ne 0) { throw 'Probe deployment failed' }
    & $adb -s $Serial shell 'cd /data/local/tmp/xiaomi-hexagon-exl3 && chmod 755 exl3_npu_probe && export ADSP_LIBRARY_PATH=/data/local/tmp/xiaomi-hexagon-exl3 && ./exl3_npu_probe'
    if ($LASTEXITCODE -ne 0) { throw 'NPU probe failed' }
}
