# Byte-sum reduction and rejected candidates

The retained change replaces mul1's four-byte shift/mask/add reduction with the HVX unsigned-byte dot-product instruction `Q6_Vuw_vrmpy_VubRub(product, 0x01010101)`. The integer numerator and explicit FP16 round-to-nearest-even remain unchanged. No weight, tokenizer, record layout, or KV format changes are made.

## Retained-path evidence

- DSP exhaustive mul1 test: all 65,536 states agree bit for bit with the portable reference.
- Scalar tile checks (24 cases), HVX 4/6/8-bit x three-codebook checks, and the standalone matrix checks pass.
- All 24 residual streams for token 100 exactly equal the previous HMX graph (zero absolute/squared difference).
- The complete two-sentence translation still matches the CUDA reference.
- Eight simultaneous SSE requests all match their eight complete CUDA references and retain correct IDs, timestamps, usage, and DONE events. Peak active slots is eight; end-to-end request latency is 31.49-34.79 seconds including prefill.

Same isolated, short-context CLI conditions as before: 56 prompt tokens, 25 decode runs, 8k capacity, logical batch 128, microbatch 4, greedy sampling, eight CPU threads, no warmup or conversation wrapping.

| Metric | Previous HMX path | Byte-sum dot product |
|---|---:|---:|
| Prompt time / 56 tokens | 10,754.44 ms | 10,353.83 ms |
| Decode time / 25 runs | 18,866.14 ms | 17,645.59 ms |
| Decode rate | 1.33 tokens/s | 1.42 tokens/s |

This is a modest approximately 7% improvement in this short run, not evidence of the 15 tokens/s target or sustained performance. A separate instrumented profile totals 680,491 microseconds in 151 EXL3 matrix kernels versus the previous 724,956 microseconds.

## Candidates not adopted

1. Full/larger K-span aggregation: bounded VTCM planning allowed the real FFN K=6,144 span to fit, and all 21 matrix tests passed. Nevertheless, the model profile was 731,414 microseconds and the isolated decode run was 1.31 tokens/s (19,112.52 ms / 25 runs), not faster than the 1.33 baseline. Its extra scratch and changed partial-rounding schedule were not retained. The previous queue/128-K-group behavior is restored.
2. Hardware half conversion after the exact integer numerator: the shorter conversion differed at 1,407 of 65,536 codebook states. It was rejected; the explicit round-to-nearest-even path is retained. No relaxation of the bit-exact decoder gate was made.

The serialized payload stays at 4.805952430792548 bpw, and KV stays FP16. Actual long-context and sustained checks remain pending; packed decode is still the dominant optimization problem.
