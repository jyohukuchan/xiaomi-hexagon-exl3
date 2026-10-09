# Xiaomi Hexagon EXL3

Experimental inference engine work for running EXL3-quantized language models on the Qualcomm Hexagon NPU in the Xiaomi 14 Ultra (Snapdragon 8 Gen 3 / SM8650).

The first end-to-end target is [IndexTeam/Index-Translate-2B](https://huggingface.co/IndexTeam/Index-Translate-2B), with request batching and an OpenAI-compatible HTTP API. Most weights will use approximately 4 bpw, while precision-sensitive tensors may use higher precision provided the complete serialized weight payload remains at or below 5.0 average bits per parameter.

> [!IMPORTANT]
> Native EXL3 inference and the phone-hosted streaming API work with FP16 KV and all 151 packed-weight matrices on Hexagon. Eight simultaneous requests/SSE clients and continuous admission match the short CUDA references. Exact decoder optimizations raise the short-context single-request rate from 0.25 to 1.02 tokens/s, still far below the 15 tokens/s target. Performance, sustained load, and broader/long-context validation remain under development. See [milestones](docs/MILESTONES.md), [optimization evidence](docs/validation/hvx-optimization.md), and [API setup](docs/API.md).

## Validated hardware

- Xiaomi 14 Ultra global (`24030PN60G`, `aurora`)
- Snapdragon 8 Gen 3 / SM8650
- Hexagon v75
- 4 HVX contexts, 1 HMX unit, 8 MiB VTCM
- Stock HyperOS / Android 16, without root access

The upstream [llama.cpp Snapdragon backend](https://github.com/ggml-org/llama.cpp/tree/master/docs/backend/snapdragon) successfully opens the CDSP FastRPC session and executes FP16, Q8_0, and Q4_0 matrix multiplication kernels on the device.

## Initial results

Matrix shape: `[4096, 14336] × [14336, batch]`, with F32 activations.

| Weight format | Batch 1 | Batch 8 | Batch 512 |
|---|---:|---:|---:|
| FP16 | 47.66 GFLOPS | 172.97 GFLOPS | 3.74 TFLOPS |
| Q8_0 | 84.24 GFLOPS | 521.79 GFLOPS | 6.15 TFLOPS |
| Q4_0 | 157.26 GFLOPS | 656.66 GFLOPS | 6.61 TFLOPS |

Additional observations:

- Effective sequential packed-weight bandwidth: approximately 44–48 GB/s
- Contiguous FP16 device copy: 40.36 GB/s
- Batch 1 Q4_0 selects an HVX kernel
- Batch 512 selects HMX kernels
- A short sustained load reduced Q4_0 batch-512 throughput from 6.61 to 5.70 TFLOPS as the NPU thermal sensors heated up

The single-request decode target is **15 tokens/second or faster** on the Xiaomi 14 Ultra. Initial bring-up uses an FP16 KV cache; cache quantization is deliberately deferred until end-to-end inference is reliable. See [project goals](docs/GOALS.md) for the current acceptance criteria and cache policy.

See the [full benchmark report](docs/benchmarks/2026-10-09-xiaomi-14-ultra.md) for methodology and power-measurement limitations.

## Planned architecture

Build and verification commands are documented in [development instructions](docs/DEVELOPMENT.md).

1. Define the EXL3 mixed-bit packed-weight layout and conversion pipeline.
2. Implement packed dequantization and matrix kernels for HVX and HMX.
3. Add the Qwen3.5-derived hybrid attention and linear-attention graph used by Index-Translate-2B.
4. Implement persistent model buffers, KV/state management, and request batching.
5. Expose streaming chat/completions endpoints compatible with the OpenAI API.
6. Measure accuracy, sustained throughput, memory use, and whole-device power.

## License

Project-authored code is available under the [MIT License](LICENSE). Third-party components retain their original licenses and notices.

## Related projects

- [Qualcomm Hexagon-MLIR](https://github.com/qualcomm/hexagon-mlir) — compiler infrastructure for Triton/PyTorch kernels targeting Hexagon
- [llama.cpp Snapdragon backend](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md) — experimental Hexagon backend and device tooling
- [Index-Translate-2B](https://huggingface.co/IndexTeam/Index-Translate-2B) — target translation model
