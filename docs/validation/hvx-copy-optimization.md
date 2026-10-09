# Packed-weight HVX copy and stage profiling

Packed weights were copied from DDR/L2 into VTCM with ordinary `memcpy`. The runtime now uses the existing `hvx_copy_aa` helper for aligned sources and `hvx_copy_au` for unaligned sources. Packed group lengths are multiples of 128 bytes; the destination is already aligned by EXL3 scratch planning. This does not change any weight bits, matrix arithmetic, quantization recipe, or KV format.

## Measurements that motivated the change

The existing HTP trace mode now separates these worker stages:

| Event | EXL3 meaning |
|---|---|
| DMA | Packed group copy, not necessarily a DMA hardware instruction |
| HVX_W_DEQUANT | Tile decoding and scatter completion, excluding copy |
| FENCE | Waiting to acquire the HMX producer mutex |
| BUFF | Queue push through completed pop while holding the producer mutex |
| HVX_COMP | HMX result conversion/accumulation, or HVX fallback matrix computation |
| HVX_O_PROC | Output H128 transform, channel scaling, and store |
| HMX_COMP | Existing HMX-thread event around the actual queued computation |

Use `GGML_HEXAGON_PROFILE=3`, then run `python tools/analyze_htp_trace.py LOG --kernel exl3-hmx`. Kernel-window filtering excludes unrelated outer dispatch and residual-ADD events that reuse event IDs. The analyzer rejects empty, unmatched, duplicate, and ambiguous wrapped traces. Its unit tests cover counter wrap and parallel intervals. Per-thread totals and HMX/queue intervals overlap; they are not additive wall time.

On the real four-bit K-projection fixture (K=2,048, N=512, batch 1, four workers), median copy intervals were 10,982-11,811 cycles per 8 KiB group. HVX copy reduces them to 849-874.5 cycles. Decode still takes approximately 11,237-11,263 median cycles. The HMX-thread computation median is 94 cycles; producer queue round trips are roughly 1,084-1,116 cycles. These are instrumented component intervals, not end-to-end speed.

For the six-bit head slice (K=2,048, N=128, batch 1, only one worker has an output group), median copy decreases from 20,097 to 2,264.5 cycles per 12 KiB group. That fixture does not represent full vocabulary parallelism.

## Correctness and throughput

- HMX and HVX fallback each pass all 21 matrix-plus-residual cases, with 4/6/8-bit weights and batches 1/4/8/32/37/128/512. Maximum normalized squared error remains below 9.71e-8.
- All 24 token-100 residual streams are bit-identical to the group-sync version.
- Full Japanese-to-English two-sentence translation remains identical to the CUDA reference.
- Eight simultaneous SSE requests all match complete CUDA reference translations, with valid identities, timestamps, usage, and DONE events. Peak active slots is eight; end-to-end latency including prefill is 23.31-25.23 seconds.
- The complete single-token profile includes all 151 EXL3 matrices, totaling 338,270 microseconds, including 86,488 microseconds for the six-bit vocabulary head. The previous group-sync profile was 603,141 microseconds.

The isolated short-context generation uses the same 8k capacity, 56 actual prompt tokens, 25 decode runs, logical batch 128, microbatch 4, greedy sampling, eight CPU threads, no warmup, and no conversation wrapping.

| Metric | Ordinary copy | HVX copy |
|---|---:|---:|
| Prompt / 56 tokens | 9,765.68 ms | 7,272.87 ms |
| Decode / 25 runs | 16,164.62 ms | 9,264.39 ms |
| Decode rate | 1.55 tokens/s | 2.70 tokens/s |

This short-run gain is approximately 74%, not proof of 15 tokens/s, sustained speed, or actual long-context support. Serialized payload is still 4.805952430792548 bpw and KV remains FP16. Further measured work is needed on decode, input preparation, output processing, and end-to-end scheduling.
