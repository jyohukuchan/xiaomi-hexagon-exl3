# Milestones before KV-cache quantization

The completion target is the end-to-end FP16-KV engine described in [GOALS.md](GOALS.md). Component tests do not prove the complete inference engine meets its target.

| Milestone | Gate | Current evidence |
|---|---|---|
| M1: EXL3 codec and device access | Decode upstream integer-bit EXL3 tiles; validate portable reconstruction; execute EXL3 matrix operations on the phone | Passed: 24 upstream CUDA fixtures match portable/Android inner weights bit for bit. H128 reconstruction has at most 2.90e-4 max relative error. Hexagon decode matches the portable codec in 24 cases. 4/6/8-bit mul1 matrix operations pass at batch 1/4/8. |
| M2: Real model artifact | Convert the pinned BF16 model; inspect every serialized tensor; retain basic 4 bpw with sensitive promotions and a total at most 5 bpw | Passed: 150 matrices at 4 bits, head at 6 bits. Source package: 1.073 GB / 4.562 bpw; runtime repack: 1.130 GB / 4.806 bpw. Native shared-head embedding lookup matches the CUDA reference. |
| M3: End-to-end translation | Load EXL3 directly, retain packed weights, run full-attention and linear-attention on Hexagon with FP16 KV, and generate valid translations | Passed initial bring-up: complete Japanese-to-English translation agrees with the upstream CUDA reference. All 151 EXL3 matrices execute on Hexagon; full attention uses HMX and GDN executes on Hexagon. All 24 residual streams pass CPU comparison. This is one regression case, not a broad translation-quality evaluation. |
| M4: Batched API | Phone-hosted completions/chat endpoints, SSE, cancellation, and eight concurrent requests with correct per-request KV/recurrent state | Passed initial bring-up: both text endpoints and SSE work; eight ordinary and eight SSE requests exactly match eight CUDA references. A replacement request enters while earlier requests remain active; cancellation releases its slot. FP16 KV, eight slots at 8k capacity, loopback/bearer auth. See batched API validation. |
| M5: Performance and stability | Verify at least 15 decode tokens/s for a single request; evaluate sustained performance and 8k context; exercise eight concurrent requests | Not met: optimization improves the short-context baseline from 0.25 to 4.10 tokens/s, still below 15. Fixed-bitrate group helpers preserve all 24 residual files bit for bit; 84 FP32/FP16 matrix cases and API regressions pass. An older vector-scaling runtime completes 32 timed steps after an actual 8,192-token-depth random input at 3.34 tokens/s, but this does not prove long-text quality, finite logits, or sustained stability. Broader long-context and sustained checks remain pending. |
| M6: Context and release checks | Exercise 32k context, translation regression cases, repeatable deployment, and accurate capability documentation | Pending. |

KV-cache quantization starts only after the FP16-KV engine has a reliable end-to-end baseline. Any unmet gate must remain visible in this table.

## Weight storage decision

The source checkpoint contains 1,881,825,088 parameters in the text backbone, plus a vision tower and MTP tensors. The initial text inference engine does not execute the vision tower or MTP; its budget uses the text backbone parameter count.

The 248,320 x 2,048 token embedding is tied to the output head in the source checkpoint. ExLlamaV3 conversion can serialize the embedding and quantized head separately. Duplicating both must be counted in the 5 bpw payload gate. The deployment loader must preserve a shared quantized representation, or choose a measured recipe that fits the same budget; it cannot add duplicate weights to the parameter-count denominator.

## Verification conventions

- Decode rate counts generated model tokens after prompt processing, and reports context length, output length, temperature, and any CPU fallbacks.
- Correctness comparisons use upstream EXL3 CUDA reconstruction, then BF16/EXL3 reference translations with the same tokenizer, template, and deterministic sampling settings.
- NPU kernel timings and RPC wall time are reported separately.
- Persistent packed model bytes, FP16 KV, recurrent state, and scratch buffers are measured separately.
- The public MIT repository contains project code and notices. Model weights, Qualcomm SDK files, and generated FastRPC bindings remain outside Git.
