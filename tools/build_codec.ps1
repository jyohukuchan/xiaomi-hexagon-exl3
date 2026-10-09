param(
    [ValidateSet('native', 'android')][string]$Target = 'native',
    [string]$Serial = '483fa196',
    [switch]$Push
)
$ErrorActionPreference = 'Stop'
$repoPath = Split-Path -Parent $PSScriptRoot
$docker = 'C:\Program Files\Docker\Docker\resources\bin\docker.exe'
$image = 'ghcr.io/snapdragon-toolchain/arm64-android:v0.7'
$buildPath = "build-codec-$Target"
if ($Target -eq 'android') {
    $configure = '-DCMAKE_TOOLCHAIN_FILE=/opt/android-ndk-r29/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34'
    $testCommand = ''
} else {
    $configure = '-DCMAKE_CXX_COMPILER=clang++'
    $testCommand = "&& ctest --test-dir /workspace/$buildPath --output-on-failure"
}
& $docker run --rm --volume "${repoPath}:/workspace" --workdir /workspace $image bash -c "cmake --fresh -G Ninja -S . -B $buildPath -DCMAKE_BUILD_TYPE=Release $configure && cmake --build $buildPath -j 4 $testCommand"
if ($LASTEXITCODE -ne 0) { throw "Codec $Target build failed" }
if ($Push) {
    if ($Target -ne 'android') { throw 'Push requires the Android target' }
    $adb = 'C:\Android\platform-tools\platform-tools-latest-windows\platform-tools\adb.exe'
    & $adb -s $Serial shell mkdir -p /data/local/tmp/xiaomi-hexagon-exl3
    if ($LASTEXITCODE -ne 0) { throw 'Could not create device directory' }
    foreach ($binary in @('exl3_verify', 'test_codec', 'exl3_model_info', 'test_model_reader')) {
        & $adb -s $Serial push "$repoPath\$buildPath\$binary" /data/local/tmp/xiaomi-hexagon-exl3/
        if ($LASTEXITCODE -ne 0) { throw "Could not deploy $binary" }
    }
    & $adb -s $Serial shell 'cd /data/local/tmp/xiaomi-hexagon-exl3 && chmod 755 test_codec exl3_verify exl3_model_info test_model_reader && ./test_codec && TMPDIR=/data/local/tmp/xiaomi-hexagon-exl3 ./test_model_reader'
    if ($LASTEXITCODE -ne 0) { throw 'Device codec test failed' }
}
