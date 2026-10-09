# Vector H128 transforms

The input and output H128 transforms now operate on four HVX vectors rather than scalar butterfly loops. Five fixed lane permutations cover strides 1/2/4/8/16; cross-vector butterflies cover strides 32/64. The normalization constant remains the same FP32 value (`0x3db504f3`). Each butterfly is converted back to FP32 before the next stage, and the helper has a separate, non-inlined register scope.

Weights, their exact integer mul1 decoder, lossless group records, tokenizer, quantization recipe, and FP16 KV are unchanged. This optimization changes floating-point intermediate arithmetic, not packed weight precision.

## Rounding and validation contract

This path is **not bit-identical to scalar IEEE FP32**. Qualcomm's [V73 HVX manual, section 5.6](https://docs.qualcomm.com/doc/80-N2040-54/80-N2040-54_REV_AB_Qualcomm_Hexagon_V73_HVX_Programmers_Reference_Manual.pdf) describes QFloat's different rounding and precision. The project already uses QFloat in its HVX arithmetic. H128 is evaluated numerically; the weight decoder must still match all 65,536 mul1 states bit for bit.

The standalone probe checks 128 blocks in each of raw and normalized modes: zero, impulse, constant, alternating signs, exact binary fractions, and nonbinary fractions with input magnitude below 4.11. Measured normalized NMSE is 2.85e-14 and maximum absolute error 1.91e-6; raw NMSE is 2.64e-14 and maximum absolute error 2.29e-5. Bit mismatches are explicitly reported (14,206 normalized and 14,955 raw), not hidden. The component gate also requires finite output, NMSE below 1e-10, and absolute error below 8e-6 normalized / 8e-5 raw for these fixtures. These fixture bounds are not a universal error guarantee.

An IEEE-intrinsic trial initially failed compiler instruction selection without the relevant target feature. After adding feature attributes consistently, it compiled but failed the on-device numeric check (NMSE 1). It is not retained. This experiment does not establish the cause or a general device capability limitation.

Existing graph and model error gates are unchanged:

- All 42 HMX/HVX fallback matrix-plus-residual cases pass. The worst normalized squared error is 9.70e-8, against the existing 1e-5 graph gate.
- All 24 token-100 residual streams pass. Compared with the previous HVX-copy runtime, worst NMSE is 6.65e-7 and maximum absolute error is 0.00521. They are no longer bit-identical.
- Against the CPU reference (with identical input-embedding hash), worst NMSE is 1.08e-5, below the unchanged 1e-4 gate. Against the upstream CUDA reference, worst NMSE is 2.65e-4, below the unchanged 5e-4 gate.
- The complete two-sentence translation and all eight simultaneous SSE translations exactly match their CUDA text references. Peak active slots is eight; SSE request latency including prefill is 12.07-13.49 seconds.
- Continuous admission inserts a replacement while older requests remain active. Cancellation after a real content event frees the slot and a new request succeeds. OpenAI Python SDK 3.27 passes streaming checks on both completions and chat endpoints.

These are initial regression cases; broader translation and long-context evaluation are still pending.

## Performance

Additional trace events separate input scale/H128 preparation (`HVX_A_PREP`) from HMX FP16 activation preparation (`HVX_A_QUANT`). On K=2,048, N=512, batch 1, input preparation decreases from 461,704 to 168,914 cycles. Output H128/scale/store intervals decrease from approximately 22k-27k to 678-1,991 cycles across the four workers. Input scaling remains scalar and is a further measured optimization candidate.

A full single-token profile includes all 151 EXL3 matrices and totals 279,895 microseconds, including 83,771 microseconds for the six-bit vocabulary head. The preceding HVX-copy profile totaled 338,270 microseconds. Instrumented component times are not generation throughput.

The isolated short-context generation keeps the same 8k capacity, 56 actual prompt tokens, 25 decode runs, logical batch 128, microbatch 4, eight CPU threads, greedy sampling, no warmup, and no conversation wrapping.

| Metric | Scalar H128 | HVX H128 |
|---|---:|---:|
| Prompt / 56 tokens | 7,272.87 ms | 4,361.31 ms |
| Decode / 25 runs | 9,264.39 ms | 7,913.23 ms |
| Decode rate | 2.70 tokens/s | 3.16 tokens/s |

This approximately 17% short-run decode gain does not meet 15 tokens/s or prove sustained/actual long-context performance. Serialized payload remains 4.805952430792548 bpw and KV remains FP16.
