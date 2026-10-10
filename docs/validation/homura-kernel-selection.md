# Real Index-Homura GGUF kernel selection on Hexagon v75

Measured 2026-10-10 (Asia/Tokyo), Xiaomi 14 Ultra, stock non-root Android 16.
This is an independent real-model diagnostic, not a replacement of the EXL3
model, runtime, API or project acceptance criteria.

## Conclusion

Both official GGUF artifacts load and translate `Hello.` to `こんにちは。`.
All 187 active quantized weight tensors appear in executed Hexagon matrix
profiles. Matrix scheduler assignments are HTP0, not CPU or GPUOpenCL.

**Stored Q4/Q8 weights do not imply selection of HMX INT4/INT8 matrix arithmetic.**

| Artifact | Matrix input columns | Selected route | Arithmetic established from matched implementation |
|---|---:|---|---|
| Q4_K_M | 30 | HMX tiled | Q4_K/Q6_K tiles dequantized to FP16; HMX FP16 matrix multiply |
| Q4_K_M | 1 | HVX tiled | Packed low-bit weights unpacked to byte lanes; byte integer dot products, INT32 sums and floating scales/offsets |
| Q8_0 | 30 | HMX tiled | Q8_0 tiles dequantized to FP16; HMX FP16 matrix multiply |
| Q8_0 | 1 | HVX tiled | INT8 weight/activation dot products, INT32 sums and floating scales |

The HVX route dynamically quantizes activations to block-scaled byte formats.
Q4_K is repacked to the backend's offset/scale tiled representation; its
4-bit fields are unpacked before byte multiply-accumulate instructions. This
is **not a native HMX 4-bit matrix MAC**. INT8 computation does occur on HVX;
saying that the model does not use any integer arithmetic would also be wrong.

This is the backend's implemented routing, not evidence of a selection bug,
nor evidence that the silicon lacks other integer matrix capabilities. No QNN
graph, alternative HMX integer kernel, or firmware capability test was run.

## Artifacts and isolation

Source: [IndexTeam/Index-Homura-2B-GGUF](https://huggingface.co/IndexTeam/Index-Homura-2B-GGUF),
pinned revision `40a26e36b43c64c9495b71ebccb59fc230fed6b0`.
Both host downloads and device copies match published LFS SHA256 hashes.

| File | Bytes | SHA256 |
|---|---:|---|
| Index-Homura-2B.Q4_K_M.gguf | 1312164480 | `3fc56945f1db4c2b91b18ac9f4061b354d51c226a6fdd9ea95e72a927f13a675` |
| Index-Homura-2B.Q8_0.gguf | 2076674688 | `e3aa4850c1f4779328af5d721383bb5785dfdd7836f5a82976336c21178c7754` |

GGUF architecture is `qwen35`. Q4_K_M contains 168 Q4_K, 27 Q6_K and 140 F32
tensors; Q8_0 contains 195 Q8_0 and 140 F32 tensors. Both include eight quantized
`blk.24.*` MTP tensors that the normal non-speculative execution leaves unused.
The active quantized tensor-name set from the GGUF header exactly matches the
187 unique weight names in each model's executed profiles: zero missing and
zero extra names.

Local model files remain in ignored `models/index-homura-2b-gguf/`; device files
remain in `/data/local/tmp/xiaomi-hexagon-exl3/homura-kernel-probe/models/` for
reuse. The two artifacts occupy about 3.39 decimal GB on each machine. No
existing model or runtime library was overwritten.

The baseline is unmodified llama.cpp
`a11f57ba93797579a5d1855ee216a31f10242676`, using the existing package under
`/data/local/tmp/llama.cpp`. Device library hashes match the inspected host
package:

| Library | SHA256 |
|---|---|
| libggml-htp-v75.so | `e45ece5ca18240e1546fd96cdb8247919515aa3aaea893dacd3a867b84de2228` |
| libggml-hexagon.so | `133be0cb505a038199199f98c99b3fc98a76efde02727b6b723f6c34f7759f76` |
| libllama.so | `60561b21b0f503637c3fb02165d4f78ae0fa75893bd53f633db86db77bb0e38d` |
| libllama-completion-impl.so | `f120232bec335a5a9ca7b80fec2ab5825343a773d7949e3fd7bd6ca248d34f8a` |

## Runtime evidence

The final profile dataset is `homura-kernels-20261010-212155`, with clean exit
code 0 for both models. The [audited summary](homura-kernel-selection-summary.json)
contains counts, representative records and raw-log hashes.

| Artifact | Input columns | First weight type | Profile records | Route |
|---|---:|---|---:|---|
| Q4_K_M | 30 | Q4_K | 111 | HMX tiled |
| Q4_K_M | 30 | Q6_K | 23 | HMX tiled |
| Q4_K_M | 1 | Q4_K | 225 | HVX tiled |
| Q4_K_M | 1 | Q6_K | 52 | HVX tiled |
| Q8_0 | 30 | Q8_0 | 130 | HMX tiled |
| Q8_0 | 1 | Q8_0 | 269 | HVX tiled |

These count executed matrix operations, including fused `MUL_MAT_NX` and
`MUL_MAT+ADD`, not individual weight tensors. Fusions and quantization-dependent
scheduling explain differing operation counts. Input columns describe the
individual matrix operation: even prefill has single-column final-layer/head
work when only the last token's logits are needed. It is incorrect to label
every single-column profile as a separate decode step.

Each model's scheduler dump contains 1496 matrix-node assignment entries,
all HTP0. Those entries include graph reservation/planning and repeats; they
are not an executed-operation count. CPU GET_ROWS entries remain for embedding
lookup, so this is not a claim of zero CPU work. Generic llama.cpp messages
such as `offloaded 26/26 layers to GPU` refer to the selected HTP0 device here,
not Adreno. Likewise `CPU ... MATMUL_INT8=1` is a CPU feature flag, not proof of
NPU integer matrix execution.

## Source and binary arithmetic evidence

Runtime `hmx-tiled`/`hvx-tiled` labels alone do not identify operand precision.
They were linked to the matching source and compiled package:

- [htp-opnode.h](https://github.com/ggml-org/llama.cpp/blob/a11f57ba93797579a5d1855ee216a31f10242676/ggml/src/ggml-hexagon/htp-opnode.h)
  maps HMX/HVX kernel parameters to the labels in the profiles.
- [matmul-ops.c](https://github.com/ggml-org/llama.cpp/blob/a11f57ba93797579a5d1855ee216a31f10242676/ggml/src/ggml-hexagon/htp/matmul-ops.c)
  selects quantized-weight decoding workers and the HMX FP16 tile pipeline.
- [hmx-utils.h](https://github.com/ggml-org/llama.cpp/blob/a11f57ba93797579a5d1855ee216a31f10242676/ggml/src/ggml-hexagon/htp/hmx-utils.h)
  emits `activation.hf`, `weight.hf`, `mxclracc.hf` and FP16 output stores.
- [hvx-mm-kernels-tiled.h](https://github.com/ggml-org/llama.cpp/blob/a11f57ba93797579a5d1855ee216a31f10242676/ggml/src/ggml-hexagon/htp/hvx-mm-kernels-tiled.h)
  unpacks low-bit values and uses `Q6_Vw_vrmpyacc_VwVbVb`; Q8 uses the same
  signed-byte multiply-accumulate primitive without nibble unpacking.

The SDK 19.0.07 Hexagon objdump of the hash-matched V75 library shows the HMX
worker calling `core_dot_chunk_fp16` and signed-byte `vrmpy`/accumulation
instructions in the binary. This objdump prints the HMX opcode words as
`<unknown>`; it does **not** independently decode the HMX FP16 mnemonics. The
FP16 interpretation therefore relies on the matched source and call path as
well as the runtime routing, not a purported complete instruction trace.
Raw disassembly is kept in ignored `benchmark-raw/homura-htp-v75-disassembly.log`.

## Scope, failed setup attempts and reproduction

The prompt is a single short translation, 30 actual prompt tokens, at most 8
generated tokens (the model ends early). Context 512, batch 128, microbatch 64,
six CPU threads, FP16 KV, explicit HTP0 selection, all layers requested, fitting
disabled, flash attention on, no dummy warmup, and greedy sampling. Thinking is
disabled with `--reasoning off`. This verifies routing and basic generation,
not broad numerical/translation quality or sustained performance.

An initial attempt using `--chat-template-kwargs` was rejected by
`llama-completion` before model execution; that argument is accepted by other
entry points but not this completion executable. After removal, both models
generated correctly at INFO verbosity, but no kernel details were captured.
Those runs are not used as kernel-selection evidence. The final run uses
verbosity 5, `GGML_HEXAGON_PROFILE=1`, `GGML_HEXAGON_VERBOSE=1`, and
`GGML_SCHED_DEBUG=2`. Heavy profiling/logging substantially changes timings,
so neither these runs nor the two-decode-step INFO smoke constitute a speed
benchmark or a 15-tokens/s acceptance result.

```powershell
adb -s <adb-target> push tools/prompts/homura-kernel-probe.txt /data/local/tmp/xiaomi-hexagon-exl3/homura-kernel-probe/prompt.txt
python tools/run_homura_kernel_probe.py --serial <adb-target>
python tools/analyze_homura_kernel_probe.py <run-directory> --output <summary.json>
python -m unittest discover -s tests -p test_homura_kernel_probe.py
```

The monitor checks the exact remote executable before signaling its PID,
requires screen-off/discharging telemetry, retains the existing project
temperature guards, and refuses concurrent project inference. Both profiles
finished without a monitor abort. At release no project inference/probe remained;
battery was 79%, battery temperature 26.8°C, skin 28.45°C and thermal status 0.
The measurement transport was disconnected without changing the user's
wireless-debugging setting. The API remains stopped.
