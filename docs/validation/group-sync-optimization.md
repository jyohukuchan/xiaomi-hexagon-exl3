# Group-level scatter synchronization

The runtime now submits the 64 independent 16x16 tile scatters in each 128x128 group before waiting once on that worker's dedicated 128-byte padding. The wait still completes before HMX or HVX reads decoded weights. Workers have separate scratch and completion locations. The standalone single-tile decoder retains its synchronous wrapper.

The ordering rationale follows Qualcomm's [HVX memory and scatter-release documentation](https://docs.qualcomm.com/bundle/publicresource/80-N2040-44_REV_B_Qualcomm_Hexagon_V66_HVX_Programmers_Reference_Manual_.pdf): a release followed by its consuming load completes earlier scatter/gather operations for the current context. The [V73 manual](https://docs.qualcomm.com/doc/80-N2040-54/80-N2040-54_REV_AB_Qualcomm_Hexagon_V73_HVX_Programmers_Reference_Manual.pdf), section 3.9.8, also recommends deferring consumption to avoid scatter/gather stalls. Device validation below is on v75.

## Verification

- HMX and HVX fallback each pass all 21 matrix-plus-residual cases: four-, six-, and eight-bit weights; batches 1, 4, 8, 32, 37, 128, and 512. Maximum normalized squared error is 9.71e-8.
- All 24 token-100 residual streams are bit-identical to the previous byte-sum runtime.
- Full two-sentence Japanese-to-English translation remains identical to the CUDA reference.
- Eight simultaneous SSE requests all match their complete CUDA reference translations, with correct IDs, timestamps, usage, and DONE events. Peak active slots is eight. End-to-end latency including prefill is 30.09-31.40 seconds; this is not single-request decode throughput.
- A complete diagnostic profile includes all 151 EXL3 matrices, totaling 603,141 microseconds; the six-bit head is 182,360 microseconds. The previous byte-sum profile was 680,491 microseconds. Profiling is separate from generation throughput.

## Short-context throughput

Same isolated conditions: stock Xiaomi 14 Ultra, FP16 KV, 8k capacity, 56 actual prompt tokens, 25 decode runs, logical batch 128, microbatch 4, eight CPU threads, greedy sampling, no warmup or conversation wrapping. The executable is `llama-completion`; `llama-cli` in this pinned runtime does not accept the legacy conversation switch.

| Metric | Per-tile wait | Per-group wait |
|---|---:|---:|
| Prompt / 56 tokens | 10,353.83 ms | 9,765.68 ms |
| Decode / 25 runs | 17,645.59 ms | 16,164.62 ms |
| Decode rate | 1.42 tokens/s | 1.55 tokens/s |

The approximately 9% improvement is only a short-run observation. It does not meet 15 tokens/s, prove sustained throughput, or validate actual 8k/32k inputs. Serialized payload remains 4.805952430792548 bpw and KV remains FP16.

## Rejected small-table candidate

A 1,024-entry FP16 table indexed by the exact product-byte sum also matches all 65,536 states. A standalone VTCM microbenchmark repeats each vector 64 times with volatile input and output accesses to prevent elimination. The arithmetic path takes 5,088,912 cycles and the table path 6,761,567 cycles in the measured run. The table path is approximately 33% slower and is not selected by the runtime. Both modes remain in the standalone probe to reproduce the comparison; no table is allocated by the model matrix kernel.

An additional four-bit prototype permuted two adjacent 16x16 tiles into contiguous HMX-layout stores, removing scatters for those tiles. Both backends passed the same 21 matrix cases, and all 24 residual streams remained bit-identical. Nevertheless, its full profile was 621,778 microseconds, and its isolated translation decode took 16,436.45 ms / 25 runs (1.52 tokens/s). No speed advantage over group-level synchronization was demonstrated. The prototype is reverted; the runtime retains the tested group-level scatter path.
