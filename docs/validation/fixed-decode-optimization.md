# Fixed-bitrate group decoders

The 128x128 packed-weight group decoder now has separate non-inlined functions for 4, 6, and 8 bits. Each function passes a compile-time bitrate to the existing tile decoder. The worker dispatches once per group rather than carrying all three inlined decoder variants through its register scope. It still waits for scatter completion once, after the group helper returns and before consuming decoded weights.

No integer codebook arithmetic, FP16 rounding, floating-point matrix arithmetic, weight records, scratch layout, tokenizer, or KV format is changed. In the v75 disassembly, the worker's stack-frame allocation decreases from `0x980` (2,432 bytes) to `0x380` (896 bytes). This code-generation observation alone is not a throughput claim.

## Verification

- All 84 HMX/HVX matrix-plus-residual cases pass: three 4/6/8-bit fixtures, FP32/FP16 activation inputs, and batches 1/4/8/32/37/128/512. Worst NMSE is 1.03e-7, unchanged from vector scaling and below the existing 1e-5 gate.
- All 24 token-100 residual files have identical SHA-256 hashes to the vector-scaling version. The numerical comparison also has zero difference, so its existing CPU/CUDA error results remain applicable.
- The complete two-sentence translation and all eight simultaneous SSE translations exactly match their CUDA text references. Peak active slots is eight; end-to-end SSE latency including prefill is 8.44-9.51 seconds.
- Continuous replacement admission, cancellation/slot reuse, and SDK 3.27 streaming checks for both text endpoints pass.

## Measured performance

In the four-bit K=2,048, N=512, batch-1 trace, group-decode median intervals decrease from 10,625-10,694 to 8,615-8,644 cycles across workers. The full single-token profile still includes all 151 EXL3 matrices and totals 213,446 microseconds, including 60,679 microseconds for the six-bit vocabulary head. The preceding profile was 260,211 microseconds with an 80,764-microsecond head.

Same isolated short-context generation: 8k capacity, 56 actual prompt tokens, 25 decode runs, batch 128, microbatch 4, eight CPU threads, greedy sampling, no warmup, and no conversation wrapping.

| Metric | Mixed inlined decode | Fixed group helpers |
|---|---:|---:|
| Prompt / 56 tokens | 3,550.37 ms | 3,234.74 ms |
| Decode / 25 runs | 7,277.26 ms | 6,100.26 ms |
| Decode rate | 3.44 tokens/s | 4.10 tokens/s |

This approximately 19% short-run decode improvement is not evidence of the 15 tokens/s gate or sustained/long-context behavior. The existing actual-8k record was measured on the older vector-scaling runtime, not this helper version. Broader numerical/text checks at long context remain pending. Serialized payload remains 4.805952430792548 bpw and KV remains FP16.
