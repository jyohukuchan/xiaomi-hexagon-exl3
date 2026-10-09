# Development

Initialize the pinned graph/runtime dependency after cloning:

```powershell
git submodule update --init --depth 1
```

## Portable codec and model audit

The C++17 codec implements integer-rate EXL3 tile decoding for 1 through 8 bits, all three upstream codebooks, and H128 reconstruction into original-basis FP16 weights. Half-integer EXL3 rates and row-coded embedding tables are not yet handled by the C++ runtime. The model audit can inspect the embedding table headers.

Build with ordinary CMake, or on the Windows development host:

```powershell
.\tools\build_codec.ps1 -Target native
.\tools\build_codec.ps1 -Target android -Push
python -m unittest discover -s tests -p test_model_audit.py
```

Download and inspect the pinned source model:

```powershell
python tools\download_model.py
python tools\inspect_model.py models\index-translate-2b-bf16 --scope text
```

For a converted artifact, add `--reference models\index-translate-2b-bf16 --require-budget`. The audit checks payload spans and counts scales and codebook markers as stored bytes.

## Hexagon prototype

```powershell
.\tools\build_npu.ps1 -Push
```

This builds a v75 shared library and Android executable in the Snapdragon toolchain Docker image, deploys them under `/data/local/tmp/xiaomi-hexagon-exl3`, and opens the stock phone's CDSP FastRPC service. The probe checks 24 integer-bitrate/codebook decode combinations and nine mul1 matrix cases (4/6/8 bits, batch 1/4/8).

The current operator is a scalar correctness reference on Hexagon. It decodes one 16 x 16 tile at a time, applies input/output H128 transforms, and avoids allocating a full decompressed weight matrix. Codebook tables are cached in the DSP session. It is not yet an optimized HVX/HMX inference kernel.

The build uses Qualcomm SDK headers and generated QAIC bindings locally; these are not copied into Git. The probe links against the SDK's FastRPC stub library at build time and uses the device's `libcdsprpc.so` at runtime.

## Upstream CUDA oracle

Use `docker/quantize.Dockerfile` for the full pinned ExLlamaV3 conversion environment. The lighter oracle path builds only its upstream `reconstruct.cu` kernel, avoiding the full Python inference stack:

```powershell
docker run --name xiaomi-exl3-oracle --gpus all `
  --env MAX_JOBS=1 --env TORCH_CUDA_ARCH_LIST=8.9 `
  --volume C:\coding-local\xiaomi-hexagon-exl3:/workspace `
  --volume C:\coding-local\exllamav3:/upstream:ro --workdir /workspace `
  pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel `
  python tests/generate_cuda_fixtures.py --upstream-dir /upstream `
    --output /workspace/benchmark-raw/cuda-fixtures
```

The upstream checkout must be commit `151539c77abc7ab7425d30da7a4e8e3c5c154e7b`. Run `exl3_verify` against each generated fixture. The inner matrix must match bit for bit; the two H128 transforms and per-channel scaling must satisfy relative and normalized-error limits. The CPU reference uses float32 butterflies, so its final FP16 rounding need not be identical to a GPU matrix implementation.

## Conversion recipe

```powershell
docker build -f docker/quantize.Dockerfile -t xiaomi-exl3-quantize:dev .
.\tools\quantize_model.ps1
```

The first hardware bring-up conversion uses 32 rows x 1,024 calibration tokens, decoder weights at 4 bpw, and the output head at 6 bpw. The source embedding and vision tower are retained in this intermediate upstream artifact. Deployment packaging will resolve the tied embedding/head storage and omit unused vision/MTP tensors; the final 5 bpw budget must pass after that step. This intermediate conversion is not claimed to meet the deployment size gate. Final translation quality still requires evaluation and may require more calibration data or a different sensitive-layer recipe.

Conversion containers are retained for diagnostic logs. After a failed conversion, inspect its logs and remove that exited container before using `-Resume`; the model checkpoint stays in `models/quant-work`.

After conversion, prepare the deployment artifact:

```powershell
python tools\package_model.py models\index-translate-2b-exl3-4 `
  --reference models\index-translate-2b-bf16 --output models\index-translate-2b-hexagon
```

The deployment manifest aliases input embedding lookup to the original-basis columns of the quantized output head. This preserves tied weights without storing a second embedding matrix. Because the upstream intermediate uses a separate unquantized input embedding, this changes the input embedding precision to the head's precision; end-to-end translation evaluation must use the shared-head deployment representation. The prepared artifact needs the project loader to resolve this alias and is not directly loadable by an unmodified upstream HF loader.

The native `exl3_model_info` utility opens that artifact and optionally reconstructs a token embedding:

```sh
build-codec-native/exl3_model_info models/index-translate-2b-hexagon 124160
```

The Windows Android build script deploys this utility as well. It can compare the embedding with a real-model CUDA fixture by passing the fixture path and its column index after the token ID.

The converter entry point applies `tools/refit_scales.py` at runtime. This adapter keeps upstream's Hessian-metric scale fitting while chunking large temporary operations. Its CUDA parity check is `tests/test_refit_cuda.py`; it is separate from the lightweight CPU CI suite.
