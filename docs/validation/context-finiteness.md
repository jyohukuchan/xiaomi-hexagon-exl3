# Actual 8k depth: finite logits and NPU placement

`context_probe` strengthens the earlier random-depth speed check. It processes deterministic random input in logical batches of 512, checking all vocabulary logits at the last output of each batch. It then performs greedy extension while deliberately ignoring EOS and checks all logits after every step. It verifies the public hybrid-memory maximum position after each decode and writes the final logits plus a report only after model/context cleanup succeeds.

The report's hybrid minimum position reflects the recurrent-state component; it is not a full-attention KV eviction indicator. Requested cache types and actual allocation records are checked separately. Finite output logits do not prove that every internal activation or unused cache cell is finite or that the output is linguistically correct.

## Reproduce

Stop the owned API server before the diagnostic. Do not run other phone inference concurrently. Use a new, empty output directory:

```sh
export LD_LIBRARY_PATH=$PWD/runtime/lib ADSP_LIBRARY_PATH=$PWD/runtime/lib
export GGML_HEXAGON_EXL3_HMX=1 GGML_HEXAGON_PROFILE=1
runtime/bin/context_probe models/index-translate-2b-exl3-v2.hxgguf context-probe-8192 8192 32
```

After pulling the artifacts and profile log, independently validate them:

```sh
python tools/check_context_probe.py benchmark-raw/context-probe-8192 \
  --profile-log benchmark-raw/context-probe-8192.log \
  --reference-profile benchmark-raw/trace-packed-round.log
```

The reference profile must be the project's complete single-token Index-Translate profile. The validator compares each extension step's exact set of EXL3 matrix names, not just a total count, and also requires six full-attention and eighteen GDN operations. It handles the existing GDN+CPY fusion without mistaking it for a different operation. CPU token/embedding work outside these audited tensor operators is not excluded by this test.

## Observed result

The probe runs the production inference code from `fc51868`; the added diagnostic itself does not change kernels, quantization, or KV precision.

- 8,192 prefix tokens, 32 subsequent greedy steps, actual per-sequence capacity 8,448.
- All 16 prefix checkpoints and all 32 extension checkpoints pass: 48 rows x 248,320 logits = 11,919,360 finite-value checks. These are checkpoint outputs, not all 8,224 hidden/output rows.
- The independent reader confirms all 248,320 values in the saved final row are finite and checks metadata counts, position, timing consistency, token IDs, and artifact size.
- Final hybrid position is 8,223. Both hybrid min/max are 8,223 because the recurrent memory keeps the latest state; this does not imply that full-attention history was discarded.
- The actual full-attention cache allocation is 8,448 cells across six layers: K(F16) 49.50 MiB and V(F16) 49.50 MiB, 99.00 MiB total.
- All 32 extension steps execute the exact expected 151 EXL3 matrix operators on HTP0, plus six FLASH_ATTN_EXT and eighteen GATED_DELTA_NET operations per step.
- Observed logit range across checked checkpoints is approximately -18.8202 to 24.9584.
- The process exits successfully and the API server is restarted afterward.

Prefill including checks and profiling is 91,853.5 ms. Extension including checks and profiling is 8,218.29 ms / 32 steps (3.89376 diagnostic tokens/s). This is an instrumented diagnostic, not the 15 tokens/s acceptance benchmark or a controlled comparison with the older random-depth run. The uninstrumented short-translation baseline remains 4.35 tokens/s.

## Exact 8,192-capacity boundary

A separate run uses depth 8,160 plus 32 extension steps, rather than allocating room above an 8k history:

```sh
runtime/bin/context_probe models/index-translate-2b-exl3-v2.hxgguf context-probe-limit8192 8160 32
```

Actual capacity is exactly 8,192 and final position is 8,191. All 48 checkpoint rows (11,919,360 values) pass, the saved final row passes independent finite checks, and all 32 steps pass the same matrix/attention/GDN placement audit. Actual FP16 K/V allocation is 48.00 + 48.00 = 96.00 MiB across six full-attention layers. Observed checkpoint logit range is approximately -26.771 to 26.2952.

Instrumented prefill is 92,101.9 ms and extension is 8,135.63 ms / 32 steps (3.93332 diagnostic tokens/s). This remains a random-prefix numerical/placement check, not a semantic translation test, sustained benchmark, or proof of eight concurrent long contexts.

Both long diagnostic runs still report the pre-existing large CPU compute allocation for the tied embedding/head (approximately 380 MiB rather than the reserve estimate). This is not resolved by the new tool and remains a memory/scheduling investigation item.

## Short independent CPU check and safety checks

Native CPU and NPU probes at depth 32 plus four greedy steps choose identical extension IDs. Their final-logit NMSE is 3.81e-8, maximum absolute difference 0.009902. This is only a short consistency case, not an 8k CPU/CUDA reference comparison.

Malformed numeric arguments fail before creating an output directory. Nonempty output directories are rejected. Independent validator tests reject nonfinite/falsely sized data, incorrect capacity/counts/positions/IDs, missing or mismatched profile records, and malformed stage markers. The NDK compile command for the probe has no fast-math/finite-only flag that could remove `std::isfinite` checks.

## Still required

This does not establish long-text translation quality, all internal-state correctness, sustained stability, or the cause of the older HwBinder crash records. Actual long-context reference comparisons remain pending. The decode target is still 15 tokens/s, and the planned 32k stage follows that target. No KV quantization has been started; serialized payload remains 4.805952430792548 bpw.
