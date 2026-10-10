# Hexagon integer inference engines

Existing engines can execute quantized models on Hexagon through Qualcomm QNN
or QAIRT. ExecuTorch with QNN is the clearest starting point for a per-channel
W8A8 comparison; MNN with QNN is especially relevant to the Qwen3.5 architecture.
Neither is yet a verified drop-in replacement for Index-Homura or Index-Translate
on this project's phone. Sources below were checked on 2026-10-11.

## Engines and quantization paths

| Engine | Hexagon route and quantization | Deployment conditions |
|---|---|---|
| ExecuTorch | QNN HTP delegate; 8a8w, per-channel weights, mixed per-layer precision | Android deployment and SM8650 compile configuration are documented; custom model export and operator delegation still need validation |
| MLLM v1 | QNN on Snapdragon 8 Gen 3; W8A8 or W8A16 | Documented older path uses QNN plus CPU for prefill and CPU for decode; not an all-NPU decode solution |
| MLLM v2 | QNN AOT; Qwen3-1.7B W4A16 package for SM8650 | Current full-NPU example uses LPBQ and uint8 KV, not the proposed per-token W8A8 recipe or project's FP16 KV policy |
| MNN | QNN LLM export; channel-wise 8-bit weights or 4-bit weights with quantized 16-bit activations | Explicit Qwen3.5-2B conversion example targets SoC 57 / v75; this is separate from MNN's direct Hexagon backend |
| Qualcomm GENIE and GenieX | QAIRT NPU bundles; common recipe is W4A16 | Model and SoC-specific compiled bundles required; GenieX's GGUF route uses llama.cpp and is not a new native-INT8 implementation |
| ONNX Runtime | QNN HTP Execution Provider with 8-bit / 16-bit quantized graphs | Android supported; operator coverage and fixed input shapes constrain custom LLM export and batching |
| LiteRT and LiteRT-LM | Qualcomm QNN accelerator; float or statically quantized models for the NPU export flow | SM8650 listed; NPU-specific compilation required, and a downloadable SM8750 LLM bundle is not a valid v75 test |

### ExecuTorch

The [Qualcomm backend documentation](https://github.com/pytorch/executorch/blob/main/docs/source/backends-qualcomm.md)
defines 8a8w as the default, supports per-channel and block quantization, and
provides an SM8650 compilation example. Its
[quantizer implementation](https://github.com/pytorch/executorch/blob/main/backends/qualcomm/quantizer/quantizer.py)
allows per-channel Linear weights and per-module precision overrides. This is
strong evidence that an existing W8A8-capable Hexagon inference framework exists.
It does not establish that dynamic per-token activation scales, all Qwen3.5
operators, and FP16 KV can be combined unchanged in the target model.

### MLLM

The [v1 documentation](https://github.com/UbiquitousLearning/mllm/blob/v1/README.md#run-qwen-with-hexagon-npu-accelerating-using-qnn)
explicitly describes W8A8 / W8A16 on Snapdragon 8 Gen 3, with CPU decode. The
[current model table](https://github.com/UbiquitousLearning/mllm#supported-models)
instead links a Qwen3-1.7B W4A16-SM8650 NPU package. The
[QNN AOT workflow](https://ubiquitouslearning.github.io/mllm/qnn_backend/aot_execute.html)
uses LPBQ Linear weights and uint8 KV and executes token generation through the
compiled QNN context. The advertised Jetson W8A8 CUDA path is a different backend;
its speedups must not be presented as Hexagon results.

### MNN

The [NPU LLM documentation](https://github.com/alibaba/MNN/blob/master/docs/transformers/llm.md#npu-推理-llm)
supports channel-wise weight quantization (`quant_block=0`) and describes
8-bit weights as well as 4-bit variants, with quantized 16-bit feature maps in
the documented LLM export path. Its QNN conversion example specifically targets
Qwen3.5-2B with `soc_id=57`, `dsp_arch=v75`. This makes it a promising model-export
reference, not proof that Index's hybrid model runs correctly without changes.
MNN's **direct Hexagon LLM backend** currently documents only symmetric 4-bit
weights; do not confuse that restriction with the separate QNN workflow.

### Qualcomm GENIE and GenieX

[GenieX's model documentation](https://github.com/qualcomm/GenieX/blob/main/docs/en/models/supported.mdx)
distinguishes compiled QAIRT bundles from its llama.cpp GGUF runtime. QAIRT's
common `w4a16` uses integer-quantized activations; the same notation in a GPU
weight-only framework may instead mean floating-point FP16 activations.
The [QAIRT plugin](https://github.com/qualcomm/geniex-qairt-plugin) has Android
builds, but its listed tested hardware omits SM8650. A compatible v75 runtime,
compiled model bundle and target architecture must be checked before adoption.
GenieX's bundled llama.cpp precision restrictions also differ from this project's
pinned llama.cpp, where actual Q4_K_M and Q8_0 HTP execution was already verified.

### ONNX Runtime and LiteRT

[ONNX Runtime QNN EP](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
supports Android and quantized HTP graphs. Its documented MatMul type pairs include
8-bit and mixed 8-/16-bit inputs, but dynamic input shapes are not supported by
that documented EP workflow. This is a general inference framework rather than
an automatic importer for every Hugging Face LLM.

[LiteRT's Qualcomm integration](https://developers.google.com/edge/litert/next/qualcomm)
lists Snapdragon 8 Gen 3. The
[GenAI export guide](https://developers.google.com/edge/litert/conversion/pytorch/genai)
requires float or statically quantized models for Qualcomm NPU export and excludes
weight-only quantization there. An INT8 weight-only CPU recipe therefore does not
prove an INT8 NPU path. Model-specific NPU bundles and Qwen3.5 export coverage
need to be checked independently.

## LPBQ preserves low-bit storage with a shared integer grid

Qualcomm's [Low-Power Blockwise Quantization documentation](https://qualcomm.github.io/aimet-pages/releases/2.20.0/techniques/lpbq.html)
describes adjusting low-bit block encodings onto a common higher-bit-width
per-channel grid, enabling existing per-channel kernels. Its example uses 4-bit
storage, 8-bit decompressed precision and input-channel blocks of 64.

This is a more specific alternative than simply abandoning blockwise precision:
weights can retain blockwise compression while avoiding arbitrary independent
floating-point block scales during the integer dot product. It changes the
quantization constraints, so it is not a lossless repacking of Q4_K or EXL3 and
must be evaluated against the original floating-point model. The public recipe
also does not establish which individual HMX instructions QNN selects.

## Comparison priorities

First evaluate a small real Linear graph through ExecuTorch/QNN with 8a8w and
per-channel weights, alongside the existing GGUF and raw integer controls.
Include activation quantization, zero-point correction, scaling and synchronization
in the timed path. Confirm delegation, output accuracy and whether the SDK can
provide arithmetic-route evidence before attributing speed to native integer HMX.

For full-model reuse, inspect MNN's Qwen3.5 QNN exporter and MLLM's SM8650 AOT
example before implementing another complete serving engine. Check hybrid
attention/recurrent state, tokenizer, target SoC, KV precision, batching and API
behavior separately. The original FP16 KV policy and EXL3 implementation remain
unchanged; an example's uint8 KV is not adopted implicitly.

W8A8 is a comparison baseline outside the final <=5 bpw weight budget. LPBQ or
another W4 scheme is a later candidate if quality and total stored-bpw gates pass.
QNN and QAIRT remain separately licensed dependencies; project-authored code
stays MIT and no vendor materials are added to this repository.
