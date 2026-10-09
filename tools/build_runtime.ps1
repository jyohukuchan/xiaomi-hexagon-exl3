param([ValidateSet('android','native')][string]$Target='android', [string]$Serial='483fa196', [switch]$Push)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'apply_runtime_patch.ps1')
if ($LASTEXITCODE -ne 0) { throw 'Runtime overlay setup failed' }
$docker = 'C:\Program Files\Docker\Docker\resources\bin\docker.exe'
$build = "build-runtime-$Target"
$package = "pkg-runtime-$Target"
if ($Target -eq 'android') {
    $options = '-DCMAKE_TOOLCHAIN_FILE=/opt/android-ndk-r29/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 -DGGML_HEXAGON=ON -DHEXAGON_SDK_ROOT=/opt/hexagon/6.6.0.0 -DHEXAGON_TOOLS_ROOT=/opt/hexagon/6.6.0.0/tools/HEXAGON_Tools/19.0.07 -DPREBUILT_LIB_DIR=android_aarch64'
} else {
    $options = '-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DGGML_HEXAGON=OFF'
}
$command = "cmake -G Ninja -S third_party/llama.cpp -B $build -DCMAKE_BUILD_TYPE=Release -DEXL3_ROOT=/workspace -DGGML_OPENCL=OFF -DGGML_OPENMP=OFF -DGGML_NATIVE=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=ON $options && cmake --build $build -j 8 && cmake --install $build --prefix /workspace/$package"
& $docker run --rm --volume "${repo}:/workspace" --workdir /workspace ghcr.io/snapdragon-toolchain/arm64-android:v0.7 bash -c $command
if ($LASTEXITCODE -ne 0) { throw "Runtime $Target build failed" }
if ($Push) {
    if ($Target -ne 'android') { throw 'Push requires the Android target' }
    $adb='C:\Android\platform-tools\platform-tools-latest-windows\platform-tools\adb.exe'
    & $adb -s $Serial shell mkdir -p /data/local/tmp/xiaomi-hexagon-exl3/runtime
    & $adb -s $Serial push "$repo\$package\." /data/local/tmp/xiaomi-hexagon-exl3/runtime/
    if ($LASTEXITCODE -ne 0) { throw 'Runtime deployment failed' }
    & $adb -s $Serial shell chmod -R 755 /data/local/tmp/xiaomi-hexagon-exl3/runtime/bin
    if ($LASTEXITCODE -ne 0) { throw 'Runtime permission setup failed' }
}
