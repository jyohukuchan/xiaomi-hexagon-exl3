# Q8_0 batch 4 / 16 / 64: GPU and NPU follow-up

Date: 2026-10-10 (Asia/Tokyo). Follow-up to the
[matched GPU/NPU benchmark](2026-10-10-gpu-npu.md).

## Protocol

The same isolated native binary and unmodified baseline libraries are used.
The matrix remains `[4096,14336] × [14336,batch]`, with Q8_0 weights, F32
activations and F32 outputs. There is no CPU fallback and no model execution.
Q8_0 is a stored weight format, not a guarantee of native INT8 matrix work or
identical internal arithmetic on GPU/NPU.

Each batch is tested in GPU/NPU/NPU/GPU order: two 30-second runs per backend,
with at least 15 seconds idle/cooldown preceding every run. All cases use the
same 25% software-duty protocol as before: graph compute plus synchronization,
then sleep for three times the measured compute duration. Reported GFLOPS
include these sleeps. The first five seconds are excluded from the mean power
sample, but not from the throughput calculation.

This matches the previous duty ratio, **not a fixed burst period, clock or
delivered throughput**. Graph repeat counts are selected from a warmup and
actual burst durations vary. The JSON records mean active/noncompute graph
durations to make this limitation visible. No attempt is made to extrapolate
these results to continuous full load.

The phone is USB-disconnected, discharging, screen-off/Dozing, and monitored
over its existing Android wireless-debugging connection. Native code and
library hashes match the previous package. Every matrix result is checked
before/after timing for finite outputs and 64 deterministic CPU FP64 reference
samples with NMSE at most `5e-4`. The numerical gate is permissive and is not
a model-quality or FP32-equivalence claim.

Project stop limits remain GPU/NSP 80°C, CPU 85°C, skin 42°C, battery 40°C,
Android thermal status moderate, charging, awake screen, or battery below 25%.
Sampling targets 0.5 seconds and polling can overshoot a temperature limit.
These project guards are not the phone's native throttling thresholds.

Power is signed HAL `ibat × vbat`, including the complete device and idle
overhead; it is not GPU/NPU rail power. Battery-current quantization, HAL
filtering/update cadence, background work, and monitoring overhead remain
uncertainties. Reversed repeats reduce ordering bias but two repeats do not
establish confidence intervals.

## Results

All 12 trials completed with clean exits, both numerical checks, matching log
hashes, and the offline arithmetic/telemetry audit. No thermal stop occurred.
The [audited JSON](2026-10-10-q8-batch-sweep-summary.json) preserves each trial
and the six aggregate groups. The source run directory is
`accelerator-20261010-204129-sustained` (private raw logs/telemetry are ignored).

Values below are the mean of two runs; brackets contain their min–max range,
not a statistical confidence interval. Efficiency is the mean of each run's
`GFLOPS / battery_mean_W`, not the ratio of the two column means.

| Batch | Backend | GFLOPS | Whole-device W | GFLOPS/W |
|---|---|---:|---:|---:|
| 4 | GPU | 18.63 [18.33–18.93] | 0.997 [0.905–1.089] | 18.87 [16.84–20.91] |
| 4 | NPU | 80.09 [79.64–80.54] | 3.041 [2.775–3.307] | 26.55 [24.08–29.02] |
| 16 | GPU | 75.84 [75.79–75.89] | 1.309 [1.041–1.577] | 60.45 [48.12–72.78] |
| 16 | NPU | 269.10 [268.98–269.22] | 2.844 [2.759–2.930] | 94.70 [91.80–97.60] |
| 64 | GPU | 204.06 [202.28–205.83] | 1.757 [1.528–1.986] | 118.28 [101.86–134.70] |
| 64 | NPU | 1677.64 [1647.53–1707.76] | 4.116 [3.548–4.685] | 416.54 [351.69–481.39] |

Compute-only GFLOPS are separately retained in JSON, but are not substituted
for these sleep-inclusive rates when calculating efficiency.

## Observations and limits

- NPU/GPU throughput ratios at batches 4/16/64 are approximately
  4.30× / 3.55× / 8.22× under this protocol.
- The NPU reaches about 1.68 TFLOPS at batch 64, close to the earlier
  batch-512 25%-duty observation of 1.71 TFLOPS. Those endpoint measurements
  were taken earlier, once per condition, so they are not contemporaneous
  counterbalanced repeats and do not establish a precise saturation point.
- Repeat power differences are substantial despite comparatively stable
  throughput. NPU batch 64 is 3.548 versus 4.685 W; GPU batch 16 is 1.041
  versus 1.577 W. The idle baseline before the latter GPU runs is 0.744 versus
  1.515 W, showing a sizable change in whole-device background/baseline state.
  It does not identify the source of that change or make baseline-subtracted
  power an accurate GPU rail measurement.
- Across trials, mean active graph bursts span 20.91–55.67 ms and mean
  noncompute graph time spans 62.92–167.68 ms. The ratio stays near 25%, but
  different burst periods, dynamic clock policy and current-sensor sampling
  remain confounders. More repeats or fixed-period experiments are needed
  before assigning the power changes to individual execution units.
- Maximum active-sample GPU temperature is 47.3°C, NSP 76.3°C, and skin
  34.56°C. The largest numerical-check NMSE is `3.8470e-4`, below the specified
  `5e-4` gate, but not evidence of identical internal arithmetic.

Native binary and four baseline-library SHA256 hashes matched the previous
report after the sweep. No project benchmark/inference process remained.
At release, battery was 81%, battery temperature 30.3°C, skin 32.26°C, and
Android thermal status 0. The measurement transport was disconnected; the
user's existing Android wireless-debugging setting/pairing was left unchanged.

## Reproduction

```powershell
python tools/run_accelerator_bench.py --serial <adb-target> --stage sustained --formats q8_0 --batches 4 16 64 --repeats 2
python tools/analyze_accelerator_bench.py <run-directory> --output <summary.json>
python -m unittest discover -s tests -p test_accelerator_bench.py
```

`--formats`, `--batches`, and `--repeats` permit selected matrix sweeps without
changing the original default plans or native binary. Alternate pair repeats
reverse backend order. The runner records its source hash and complete case
plan in private run metadata. Unrelated experimental inference changes are
not part of this benchmark follow-up.
