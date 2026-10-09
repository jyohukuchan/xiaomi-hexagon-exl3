# Rejected CPU embedding parallelization

The CPU scheduler currently assigns one task to GET_ROWS. EXL3 embedding reconstruction is more expensive than a plain embedding-table read: each selected column is reconstructed from packed 128x128 groups. Optional `EXL3_CPU_EMBED_PROFILE=1` logging now reports thread count, input-ID count, reconstructed groups, and per-worker elapsed time without printing token IDs or prompt text. It is disabled by default.

The serial baseline has 16 groups per single token on one worker. Across 25 single-token calls, median worker duration is 15.14 ms and mean is 16.09 ms. This function interval excludes preceding backend copies and scheduler overhead.

A prototype flattened `(token ID, K group)` work across up to eight CPU workers, with private decode scratch and disjoint output slices. Each worker handled two groups for a single token; the median maximum worker duration across the eight workers was approximately 2.95 ms. Group reconstruction arithmetic was unchanged. Native CPU checks for 4/6/8-bit fixtures, counts 1/3/17, duplicate IDs, and requested thread settings 1/2/4/8 passed, with bit-identical output across settings.

However, isolated end-to-end generation did not improve. An unprofiled same-binary A/B/B/A run used `EXL3_CPU_EMBED_SERIAL=1` to select the serial path and `0` to select the prototype, leaving the main CPU thread setting at eight. All four full translations matched the reference:

| Order | Embedding mode | Prompt / 56 tokens | Decode / 25 steps | Decode rate |
|---|---|---:|---:|---:|
| A1 | Serial | 3,040.17 ms | 5,827.20 ms | 4.29 tokens/s |
| B1 | Parallel | 3,212.22 ms | 6,082.85 ms | 4.11 tokens/s |
| B2 | Parallel | 3,245.43 ms | 6,019.48 ms | 4.15 tokens/s |
| A2 | Serial | 3,025.96 ms | 5,779.87 ms | 4.33 tokens/s |

The cause of the component/end-to-end discrepancy is not established. CPU/DSP contention, clock policy, or memory/scheduler effects require separate evidence; none is claimed as the cause here.

The work partition, scheduler override, and temporary serial-selection environment variable were fully reverted. The production scheduler is still serial. Only optional profiling and expanded CPU embedding regression cases remain. In the restored build, requested thread settings do not imply parallel GET_ROWS execution. Eight SSE requests on the restored model service still match their complete CUDA references.

The single-request target remains 15 tokens/s and the prior accepted short-translation baseline remains 4.35 tokens/s. No new end-to-end speed gain is claimed, and KV stays FP16.
