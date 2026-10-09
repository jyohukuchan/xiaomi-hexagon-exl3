# Exact HVX decode optimization

The packed model, 150 four-bit matrices, six-bit head, tokenizer, FP16 KV, and lossless runtime record layout are unchanged. This work changes the DSP decoder's execution and temporary memory, not the quantization recipe.

## Implemented changes

- Decode 16x16 tiles directly into their positions in the 128x128 matrix. This removes 1,024 small row copies per group.
- Replace mul1's 65,536-entry table gather with vector integer arithmetic. For the four-byte sum `s` of `state * 0x83dcd12d` (modulo 2^32), the unrounded value is exactly `(1774*s - 905216) / 2^18`. The numerator magnitude fits 20 bits. The kernel uses an exact integer-to-FP32 conversion, exponent rebias, and explicit FP16 round-to-nearest-even.
- All possible mul1 values are normal FP16 numbers. Both the portable test and the DSP test enumerate all 65,536 states and compare their FP16 bit patterns. A floating-point vector formulation did not match at 1,582 states and was replaced by the integer formulation.
- Four-bit windows use four uniform-shift phases instead of the generic duplicated-word permutation. Other codebooks retain the table path. The six-bit specialized phase permutation was tested but not retained because the whole-model profile became slower.
- Remove the unused mul1 table initialization/copy and 128 KiB VTCM reservation. Host scratch planning, DSP layout, and row-chunk bounds are updated together.
- Add optional existing-profiler events around group decode and MAC. A representative group showed roughly 34k cycles in decode versus 10k in MAC before the final specialization; decode remains the major optimization target.

## Verification

- Exhaustive 65,536-state mul1 test: zero mismatches on CPU and Hexagon v75.
- Standalone scalar tile checks: all 24 bitrate/codebook cases pass. HVX 4/6/8-bit x three-codebook checks pass bit for bit.
- Runtime matrix plus residual ADD checks: real four-bit K projection, real six-bit head slice, synthetic eight-bit matrix; batches 1/4/8/32/37/128/512, all 21 HTP cases pass, normalized squared error below 3.5e-8.
- All 24 residual streams for token 100 match the previous native NPU run; their worst error against CPU is unchanged (9.72e-6).
- The two-sentence Japanese-to-English translation still matches the CUDA reference.
- Eight simultaneous SSE requests all produce the eight complete CUDA reference translations with valid identities, timestamps, usage, and DONE events. Their observed end-to-end latencies are 61.19-66.71 seconds, including prefill; these are not decode-rate measurements.

## End-to-end measurement

Conditions: Xiaomi 14 Ultra stock OS, single request, greedy sampling, 8,192-token capacity, 56-token initial prompt, 25 decode runs, logical batch 128, microbatch 4, eight CPU threads, no warmup, no conversation wrapping. The benchmark is not run alongside another inference process.

| Measurement | Initial exact baseline | Optimized decoder |
|---|---:|---:|
| Prompt processing | 50,536.31 ms / 56 tokens | 18,696.67 ms / 56 tokens |
| Decode | 101,687.11 ms / 25 runs | 24,480.53 ms / 25 runs |
| Decode rate | 0.24585 tokens/s | 1.02122 tokens/s |

This is approximately a 4.15x decode improvement. It is still far below 15 tokens/s. These are short-context bring-up runs, not sustained or actual 8k/32k-context validation.

Diagnostic single-token profiles separately observed 4.029 seconds in the 151 original EXL3 kernels, 1.151 seconds after direct scattering, and approximately 0.945 seconds with exact arithmetic/four-bit specialization. Profiles include instrumentation and do not establish complete generation throughput.

Serialized payload remains 1,130,495,232 bytes / 4.805952430792548 bpw. The unexpectedly large CPU scratch allocation for the tied embedding/head still needs investigation. Faster packed decode/MAC and memory scheduling remain necessary; no KV-cache quantization has been introduced.
