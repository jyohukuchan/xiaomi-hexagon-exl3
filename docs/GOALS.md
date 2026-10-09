# Project goals

## Primary deliverable

Run `IndexTeam/Index-Translate-2B` on the Xiaomi 14 Ultra Snapdragon 8 Gen 3 Hexagon NPU as an OpenAI-compatible, continuously batched inference service.

## Acceptance targets

- Single-request decode target: at least 15 tokens/second on the Xiaomi 14 Ultra.
- Weight quantization: use approximately 4 bpw for the majority of quantized tensors.
- Sensitive tensors may be promoted to higher precision when justified by translation-quality measurements.
- Total serialized weight payload, including quantization scales and codebooks, must remain at or below 5.0 average bits per model parameter. Tokenizer files and runtime scratch buffers are excluded.
- Accept upstream EXL3 model files and allow a lossless device-specific repack during model preparation or first load.
- Support continuous request batching, with an initial maximum concurrency of eight requests.
- Provide `/v1/chat/completions` and `/v1/completions`, including SSE streaming.
- Run on stock, non-root HyperOS. GPU execution is not required.

The initial context target is 8,192 tokens, followed by 32,768 after the decode target is met. Longer contexts are a later optimization target.

## Compute placement

- Hexagon NPU: weight matrix kernels, attention, gated-delta/linear-attention kernels, and other throughput-critical tensor operations.
- CPU: tokenization, request scheduling, lightweight graph control, and sampling where NPU placement is not beneficial.
- GPU: not part of the required execution path.

## KV and recurrent state policy

- Implement cache storage behind a format-independent paged-cache interface.
- Use FP16 K/V for bring-up and the first end-to-end implementation.
- Do not allow KV-cache quantization work to block model execution, correctness, batching, or the 15 tokens/second decode target.
- Revisit KV-cache quantization only after the model and API operate reliably and have an end-to-end performance baseline.
- When that milestone is reached, compare llama.cpp-compatible Q8_0 against ExLlamaV3-style rotated arbitrary-bit storage using translation quality, sustained decode speed, and total cache memory.
- Keep the paged-cache interface format-independent so cache quantization can be added without redesigning request scheduling or attention state ownership.

Index-Translate-2B contains six full-attention layers and eighteen linear-attention layers. Only the full-attention layers grow their K/V storage with context length; the linear-attention layers use fixed-size recurrent state per active sequence.

## License

Project-authored code is released under the MIT License. Third-party source and binary dependencies retain their original licenses and notices.
