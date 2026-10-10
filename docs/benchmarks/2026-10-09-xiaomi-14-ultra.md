# Xiaomi 14 Ultra / Snapdragon 8 Gen 3 Hexagon benchmark

Date: 2026-10-09 (Asia/Tokyo)

## Device and software

- Device: Xiaomi 14 Ultra global (`24030PN60G`, `aurora`)
- SoC: Qualcomm SM8650 (Snapdragon 8 Gen 3)
- Android: 16 / API 36, HyperOS `OS3.0.304.0.WNAMIXM`
- Memory: about 14.8 GiB usable RAM
- Hexagon backend: upstream llama.cpp experimental Snapdragon backend
- Detected DSP: Hexagon v75, 4 HVX contexts, 1 HMX unit, 8 MiB VTCM
- Runtime: FastRPC over CDSP, stock non-root firmware
- Matrix shape: weights `[4096, 14336]`, activations `[14336, batch]`
- Activation type: F32

The quantized results below are GGML `Q8_0` and `Q4_0` weight kernels. They are useful proxies for 8-bit and 4-bit LLM inference, but they are not raw dense INT8×INT8 or INT4×INT4 hardware TOPS and are not EXL3 kernels.

## Matrix multiplication

Reported FLOPS use the benchmark's convention of two floating-point operations per multiply-accumulate, including dequantized quantized-weight kernels.

| Weight format | Batch 1 | Batch 4 | Batch 8 | Batch 512 |
|---|---:|---:|---:|---:|
| FP16 | 47.66 GFLOPS (2463.95 us) | 146.57 GFLOPS (3205.06 us) | 172.97 GFLOPS (5431.57 us) | 3.74 TFLOPS (16071.91 us) |
| Q8_0 | 84.24 GFLOPS (1394.16 us) | 320.50 GFLOPS (1465.72 us) | 521.79 GFLOPS (1800.58 us) | 6.15 TFLOPS (9781.48 us) |
| Q4_0 | 157.26 GFLOPS (746.80 us) | 277.38 GFLOPS (1693.56 us) | 656.66 GFLOPS (1430.76 us) | 6.61 TFLOPS (9102.48 us) |

Cold/short-run kernel selection:

- Batch 1 Q4_0 used the `hvx-tiled` path at about 1.94 GHz.
- Batch 512 used `hmx-tiled` for FP16, Q8_0, and Q4_0.
- After sustained load, profiling showed about 1.59 GHz and Q4_0 batch 512 measured 5.70 TFLOPS, versus 6.61 TFLOPS before the stress run (about 14% lower).

## Memory bandwidth

### Explicit device copy

| Operation | Tensor | Traffic counted | Time | Effective bandwidth |
|---|---:|---:|---:|---:|
| Contiguous FP16 copy | 32 MiB | 64 MiB read + write | 1550.18 us | 40.36 GiB/s |
| Transposed FP16 copy | 4096×4096 / 32 MiB | 64 MiB read + write | 118159.48 us | 0.53 GiB/s |

Correction (2026-10-10): the original copy harness reports binary GiB/s, not
decimal GB/s. Contiguous copy is approximately 43.29 GB/s. The packed-weight
bandwidth calculations below already use decimal GB/s and are unchanged.

The stock performance suite's approximately 384 MiB transposed-copy cases stalled for minutes in FastRPC completion. A temporary 32 MiB test case was used instead. The source change was reverted after measurement.

### Model-relevant sequential weight bandwidth

For batch 1, the matrix-vector kernel is predominantly weight-bandwidth limited. Effective packed-weight read bandwidth, computed from the packed tensor size divided by kernel time, is:

| Format | Packed weight bytes used in calculation | Time | Effective weight bandwidth |
|---|---:|---:|---:|
| FP16 | 117.44 MB | 2.46395 ms | 47.66 GB/s |
| Q8_0 (34 bytes per 32 weights) | 62.39 MB | 1.39416 ms | 44.75 GB/s |
| Q4_0 (18 bytes per 32 weights) | 33.03 MB | 0.74680 ms | 44.23 GB/s |

This 44–48 GB/s range is the most relevant first-order bandwidth figure for single-token LLM decoding on this backend.

## Power, thermal, and profiling visibility

### Data available without root

- Per-op Hexagon duration, cycle count, inferred clock, kernel path, VTCM use, and eight programmable PMU counters.
- Default PMU event IDs exposed by the backend: `0x3, 0x111, 0x100, 0x105, 0x240, 0x256, 0x7d, 0x8c`.
- NPU thermal zones named `nsp0` through `nsp5`, plus skin, battery, CPU, and GPU temperatures.
- Battery current (`ibat`) and voltage (`vbat`) through the thermal HAL. Sampling was practical at roughly 0.75–1 second intervals.

Example Q4_0 batch-1 per-op profile:

- `hvx-tiled`, 813 us, 1,579,410 cycles, about 1.943 GHz
- PMU raw counts: `[1813333,1626628,2403497,1419,1489325,172256,66,20]`

### Whole-device power estimate

The phone remained connected by USB, at 100% state of charge. Power was estimated as signed `ibat × vbat`.

| State | Samples / interval | Mean signed battery power | Observed range |
|---|---:|---:|---:|
| Idle/cooling | 15 / 0.75 s | 0.48 W | -1.23 to 5.69 W |
| Repeated Q4_0 batch-512 load | 25 / 0.75 s | 4.43 W | -1.22 to 10.44 W |
| Rough incremental load | baseline subtraction | about 3.95 W | not an NPU-only value |

During the repeated load, the hottest `nsp0`–`nsp5` reading reached 84.1 °C. Skin rose from 38.34 °C to 40.20 °C and battery temperature from 33.4 °C to 33.8 °C.

These power values include NPU, DRAM, application CPU, FastRPC overhead, power-conversion loss, background activity, and USB charging/discharging behavior. The signed current is quantized and noisy, so the roughly 3.95 W delta is only an order-of-magnitude whole-device estimate.

### Data unavailable on the stock non-root image

- `android.hardware.power.stats.IPowerStats/default` is not published.
- `dumpsys powerstats` exposes no energy consumers or rail measurements.
- Battery `current_now`, `voltage_now`, `power_now`, `charge_counter`, and related sysfs nodes are permission-denied to the shell user.
- No NPU/HTP power rail or per-domain energy counter is available through the public Android interfaces.

Therefore exact NPU-only watts cannot be measured on this stock image. A better next measurement is wireless ADB with USB disconnected, fixed screen/radio state, long steady workloads, and idle subtraction. That still measures the complete phone/SoC path. Exact NPU rail power would require vendor/root instrumentation or board-level/external measurement support.

## Practical conclusion

- The Hexagon backend is functional on the stock Xiaomi firmware.
- Single-token LLM-style work is limited by about 44–48 GB/s effective packed-weight bandwidth.
- Large-batch HMX throughput reaches about 6.1–6.6 TFLOPS in short runs for Q8_0/Q4_0 proxy kernels.
- Sustained performance needs thermal management; the short stress test already caused about a 14% throughput drop.
- EXL3 4 bpw will require a custom packed-weight decoder and Hexagon kernel. These proxy results suggest the performance target is plausible, but EXL3 layout, mixed-bit unpack cost, model graph coverage, and transfer/fusion overhead must be measured with the real model.
