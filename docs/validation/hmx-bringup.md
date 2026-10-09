# EXL3 HMX matrix path

This path retains the same EXL3 records, exact decoded FP16 weights, 4.806 bpw serialized payload, and FP16 KV. It uses HMX for the packed-weight matrix products as well as the previously enabled full attention.

## Execution and memory layout

- Decode directly into 32x32 tile-major weights, with paired K values interleaved. The same layout is readable by the HVX fallback.
- Prepare H128-transformed activations as FP16, padded to 32 rows. For each 128-element K group, HMX computes four output-column tiles. Convert the FP16 partial results back to FP32 for accumulation across K groups, then apply the existing output H128/SV processing.
- Four HVX workers decode independent output groups. A mutex serializes HMX queue submission/pop, because the existing queue has a single-producer contract. Decoding can overlap another worker's HMX job.
- All HMX activation tiles and result tiles are aligned to 2 KiB; bias/scales are aligned to 256 bytes. The public [Qualcomm HMX programmer reference](https://docs.qualcomm.com/doc/80-N2040-62/80-N2040-62_REV_AA_Qualcomm_Hexagon_V81_HMX_Programmers_Reference_Manual.pdf) documents these address constraints. That manual is for V81; the alignment behavior here was independently verified on the project's V75 phone.

During bring-up, unaligned result addresses left rows after the first eight unwritten. Canary-filled buffers demonstrated this directly. A small-K test also exposed an unaligned activation address. Both were corrected, and shared scratch-planning tests now cover alignment and allocation limits. Temporary diagnostic paths were removed.

HMX is selected by default when supported and its scratch allocation fits. Set `GGML_HEXAGON_EXL3_HMX=0` for the FP32-activation HVX fallback, or `1` to request HMX. No setting changes model storage or KV precision.

## Correctness checks

- HVX fallback: three matrices at batch 1/4/8/32/37/128/512, 21 cases pass with error below 3.5e-8.
- HMX path: the same 21 cases pass. Maximum normalized squared error against the CUDA-reconstructed matrix plus residual reference is 9.71e-8. Partial last row chunks are included.
- Unlike the exact HVX path, HMX introduces FP16 activation/partial-result rounding. Across all 24 residual streams for token 100, the maximum normalized squared difference from the prior HVX graph is 7.50e-7. It is not claimed to be bit-exact whole-model execution.
- The original two-sentence translation still exactly matches the CUDA output.
- Eight simultaneous SSE requests, continuous replacement admission, and a real-content stream cancellation followed by a new request all match the public CUDA references. Eight-SSE request latency is 31.10-34.67 seconds including prefill, versus 61.19-66.71 seconds in the previous exact-HVX run.

## Single-request throughput

Same short-context conditions as the prior benchmark: 8,192-token capacity, 56-token prompt, 25 decode runs, greedy sampling, logical batch 128, microbatch 4, eight CPU threads, no warmup, no conversation wrapping, no concurrent inference job.

| Metric | Exact HVX version | HMX version |
|---|---:|---:|
| Prompt time / 56 tokens | 18,696.67 ms | 10,754.44 ms |
| Decode time / 25 runs | 24,480.53 ms | 18,866.14 ms |
| Decode rate | 1.02 tokens/s | 1.33 tokens/s |

The diagnostic single-token profile totals 724,956 microseconds in 151 EXL3 HMX operations, including 221,545 microseconds for the six-bit output head. This profile is separate from the generation benchmark.

The 15 tokens/s target remains unmet. Actual 8k/32k context, sustained stability/thermal behavior, and broader translation evaluation remain unverified. The large CPU embedding/head scratch copy remains a memory/performance issue.
