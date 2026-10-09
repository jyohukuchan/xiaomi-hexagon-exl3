# Vector input/output scaling

The per-channel SU and SV scaling now uses the existing HVX FP16-to-FP32 conversion helpers and vector multiplication. The helper processes 64 elements at a time and uses unaligned-capable memory operations for source and destination pointers. Both FP32 and FP16 activation inputs are supported; this is not a change to the FP16 KV cache.

Like the vector H128 path, scaling uses QFloat intermediate arithmetic and is not bit-identical to scalar IEEE FP32. Packed-weight decoding, serialized weights, quantization recipe, and tokenizer are unchanged. Existing graph and model error thresholds are not relaxed.

## Verification

- The graph utility accepts optional `f32` or `f16` after its existing backend/fixture/batch arguments. FP16 inputs are explicitly rounded and the reference dot product uses those rounded values.
- HMX and HVX fallback each pass 42 matrix-plus-residual cases: three 4/6/8-bit fixtures, two input precisions, and batches 1/4/8/32/37/128/512. All 84 device cases pass; worst NMSE is 1.03e-7, below the existing 1e-5 graph threshold.
- Native CPU graph checks pass for both input precisions on the real K-projection fixture, including embedding checks.
- All 24 token-100 residual streams pass: worst NMSE against the previous vector-H128 runtime is 3.67e-7, maximum absolute difference 0.00268. CPU-reference worst NMSE is 1.07e-5 and CUDA-reference worst NMSE is 2.64e-4, within the unchanged 1e-4 and 5e-4 thresholds respectively.
- Complete two-sentence translation and eight simultaneous SSE translations still exactly match the CUDA text references. Peak active slots is eight; SSE request latency including prefill is 9.91-10.49 seconds.
- Continuous admission, cancellation/slot reuse, and OpenAI SDK 3.27 streaming checks for both endpoints pass.

These are regression cases, not broad translation-quality or long-context validation.

## Performance

On the K=2,048, N=512, batch-1 trace, input preparation decreases from 168,914 to 17,052 cycles. FP16 HMX activation preparation is 9,779 cycles. Output processing is 658-1,625 cycles across workers. The full single-token profile includes all 151 EXL3 matrices, totaling 260,211 microseconds; the six-bit vocabulary head is 80,764 microseconds. The previous vector-H128 profile totaled 279,895 microseconds.

Same isolated short-context generation conditions: 8k capacity, 56 actual prompt tokens, 25 decode runs, batch 128, microbatch 4, eight CPU threads, greedy sampling, no warmup, no conversation wrapping.

| Metric | Scalar scaling | Vector scaling |
|---|---:|---:|
| Prompt / 56 tokens | 4,361.31 ms | 3,550.37 ms |
| Decode / 25 runs | 7,913.23 ms | 7,277.26 ms |
| Decode rate | 3.16 tokens/s | 3.44 tokens/s |

This approximately 9% short-run decode improvement does not meet the 15 tokens/s gate. Serialized payload is still 4.805952430792548 bpw and KV remains FP16. Decode remains the main optimization candidate; sustained and actual long-context validation are still required.
