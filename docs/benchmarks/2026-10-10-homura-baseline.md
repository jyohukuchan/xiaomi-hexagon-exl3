# Homura real-model baseline, 2026-10-10

This is a **fixed-token, teacher-forced decode** benchmark, not a free-running
translation/quality test. It establishes matched work before integer-HMX experiments.
No integer-HMX route is enabled in the model runtime.

## Conditions

- Xiaomi 14 Ultra / Snapdragon 8 Gen 3 / Hexagon v75, stock non-root Android.
- Unmodified llama.cpp `a11f57ba93797579a5d1855ee216a31f10242676`;
  the same pinned official Homura Q4_K_M and Q8_0 artifacts as the
  [kernel-selection investigation](../validation/homura-kernel-selection.md).
- HTP0 explicitly selected, 99 offloaded layers, FP16 K/V, flash attention on,
  context 2048, batch capacity 512, microbatch capacity 64, six CPU threads.
- Profiling/debug output disabled. Model load and an eight-step warmup excluded.
- Three trials per model, identical 64 supplied decode token IDs in every trial
  and across both models. Each forward call uses one token. Greedy sampling is
  timed; printing, detokenization and finite-logit validation are outside timing.
- Fifteen seconds of loaded-model idle before each trial. USB unplugged, screen
  off; fresh HAL telemetry sampled approximately every 0.5 seconds. Project thermal,
  charging, screen and low-battery stop guards remained enabled; no trial aborted.

## Results

| Model | Pooled decode steps/s | Trial range | Mean device W | Estimated device J/step |
|---|---:|---:|---:|---:|
| Q4_K_M | 19.298 | 18.994–19.662 | 5.321 | 0.276 |
| Q8_0 | 14.991 | 14.646–15.243 | 5.011 | 0.335 |

Watts are whole-device signed battery-current × battery-voltage estimates, **not
NPU-only power**. Active windows are only about 3.3–4.4 seconds, with 6–9 samples
per trial. Q4 device power ranged 4.96–5.79 W, Q8 4.12–6.05 W. Idle device power
was roughly 0.95–0.96 W. Background activity, sensor cadence and phase boundaries
make these noisy estimates; they are not sustained accelerator efficiency ratings.
Energy uses each trial's own active duration and mean power, then pools by steps.

Natural generation attempts stopped at different EOS lengths (including 23 and
47 steps), so they were retained separately and rejected as matched-work baselines.
Teacher forcing deliberately removes that workload difference. It does **not**
establish translation accuracy, serving throughput, or attainment of the original
EXL3 / Index-Translate / OpenAI-compatible API 15 tokens/s target.

## Evidence and reproduction

- Accepted local raw dataset: `benchmark-raw/homura-baseline-20261010-220134`.
- Public audited [summary](2026-10-10-homura-baseline-summary.json) includes model,
  prompt, source and raw-log hashes, per-trial tokens, timings and power estimates.
- `tools/build_homura_baseline.ps1 -Push -Serial SERIAL` builds the independent client.
- `python tools/run_homura_baseline.py --serial SERIAL --teacher-forced`
  collects the guarded experiment (see `--help` for bounded trial options).
- `python tools/analyze_homura_baseline.py RAW_DIRECTORY` rejects bad lifecycle,
  timings, counts, tokens, power windows or raw hashes.
- `python -m unittest discover -s tests -p test_homura_baseline.py` exercises the
  acceptance/rejection logic without a phone.

Models and raw logs are ignored local artifacts; no model redistribution is included.
