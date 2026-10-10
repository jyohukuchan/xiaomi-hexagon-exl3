# GPU/Hexagon route investigation with calibrated PMU events

2026-10-10 (Asia/Tokyo), Xiaomi 14 Ultra, stock non-root Android 16,
Adreno 750 / Hexagon v75. Follow-up to [GPU interoperability](gpu-npu-interop.md).

## Result

The current shared-DMA-BUF/OpenCL/FastRPC path produces approximately the same
primary-AXI request count for CPU-generated and GPU-generated data. An immediate
Hexagon reread of a small buffer eliminates those requests, demonstrating a
working cache-hit control. Omitting the probe's explicit DSP invalidation does
not reveal a GPU-only request-bypass path. **No evidence identifying Direct Link
was obtained. This neither proves its absence nor proves that all traffic reaches
physical DRAM.**

The experiment does not implement GPU-dequant/HMX integration or change the
accepted 4.35 tokens/s baseline, model, FP16 KV, or server configuration. All
probes finish; the previously stopped API remains stopped.

## Public definitions and implementation

Raw event definitions were checked against the Qualcomm-authored
[V75 processor manual, PMU chapter](https://docs.alexrp.com/hexagon/hexagon_v75.pdf)
and [V75 HVX manual, table 5-1](https://docs.alexrp.com/hexagon/hexagon_v75_hvx.pdf)
(public mirrors, 80-N2040-57/58 Rev. AB). SDK 6.6 itrace IDs such as `0x8037` are
architecture-neutral identifiers, **not** the raw V75 register event numbers.

| Set | Raw events, in counter order | Purpose |
|---|---|---|
| 0 | `3f 40 42 46 47 48 49 118` | AXI reads/writes, slave traffic, secondary reads, HVX L2 loads |
| 1 | `118 81 8a 123 124 111 7c 7d` | HVX/L2 activity, castouts, stores, packets, scalar DU reads/misses |
| 2 | `40` on all eight counters | Duplicate primary-AXI event calibration |
| 3 | `118` on all eight counters | Duplicate HVX L2-load event and high-bit configuration calibration |

`0x118` counts L2-cacheable HVX loads; `0x40` counts primary-AXI read requests;
`0x47/48` count AXI-slave accesses; `0x49` counts secondary-AXI reads. None has a
public definition naming Direct Link as its exclusive source. In particular,
an AXI request is not necessarily an LPDDR transaction: downstream caching may
serve it. The shader/IDL transport has no route-selection operation.

The independent DSP service uses QuRT PMU APIs already exercised by the existing
Hexagon backend. It records PMU configuration readback at consume entry and
before/after each measured read. Configurations include the upper event-ID bits;
the `0x118` duplicates verify that the second configuration bank is not silently
selecting a low-byte event. Eight serial counter reads add instrumentation work;
the no-payload-work control measures its footprint.

There is no exclusive PMU ownership mechanism in this probe. It runs without
concurrent project inference/profiling. It temporarily enables PMU collection in
its independent session and disables it/restores saved configuration when the
current configuration still matches its own. If another configuration is already
present at cleanup, it does not overwrite that configuration. Previous counter
values and an unknown external profiler's enable state are not reconstructed.

## Conditions and windows

Four sets, seven sizes (128 B, 4 KiB, 32 KiB, 256 KiB, 1 MiB, 8 MiB, 32 MiB),
two producers and four read modes. Each condition has three warmups and eleven
accepted measurements. CPU/GPU ordering alternates by iteration, on the same
allocation. A fresh seed is generated for every attempt.

- CPU producer: OpenCL map for writing, CPU fills rpcmem's original host address,
  unmap and finish.
- GPU producer: OpenCL pattern kernel writes the same allocation, finish and
  explicit read-map/unmap handoff. No CPU payload read/copy on this producer path.
- `empty`: no payload read within the measured loop; baseline instrumentation.
- `hvx`: explicitly invalidate the DSP input range, read/XOR all vectors, then
  immediately reread without another invalidation.
- `scalar`: same invalidation/first/reread sequence, with volatile scalar loads;
  control for scalar-vs-HVX event discrimination.
- `hvx_noinv`: skip only the probe's explicit DSP invalidation. Existing OpenCL
  handoff and normal FastRPC/IDL cache maintenance remain. This is **not** a test
  with every cache-maintenance operation disabled.

Report phases are `arrival` (PMU-start RPC to consume entry), `cold` (first loop),
`warm` (immediate repeat), and `outer` (start RPC to first-loop completion). In
`hvx_noinv`, the `cold` field means **first**, not a guaranteed cold cache.
The broad windows include RPC/framework activity and may include unrelated DSP
work. They cover GPU production but do not attribute every transaction to it.

Each method validates every input word against its seed after both measured
loops; the host independently checks both 128-byte XOR reductions. Validation
is outside the counter windows. Empty timer windows may be below one 19.2 MHz
tick and are allowed to report zero microseconds, with nonzero cycle duration.

## Calibration and observations

The final four logs contain 2,464 accepted producer/consumer trials (616 per set),
or 9,856 phase records. All complete with zero payload/checksum mismatches.

- At 32 KiB and 1 MiB, baseline-subtracted `0x118` is exactly 256 and 8,192
  respectively, for both producers in the sets containing that event.
- At 32 KiB, primary AXI activity drops from approximately 256 payload requests
  on the first read to zero on the immediate repeat.
- The all-`0x40` and all-`0x118` counter groups agree across the eight counters
  within the audit bounds. No throughput-based guess is used to name an event.
- Scalar reads have no payload-sized `0x118` activity; their DU read/miss events
  provide a separate control. Residual activity includes instrumentation/stack.

Set 0 medians, **raw** primary-AXI requests (`0x40`, not baseline-subtracted):

| Payload / condition | CPU first | GPU first | CPU repeat | GPU repeat |
|---|---:|---:|---:|---:|
| 32 KiB, explicit invalidate | 258 | 258 | 0 | 0 |
| 32 KiB, no explicit invalidate | 260 | 260 | 0 | 0 |
| 1 MiB, explicit invalidate | 8,199 | 8,198 | 125 | 130 |
| 1 MiB, no explicit invalidate | 8,200 | 8,200 | 130 | 130 |
| 32 MiB, explicit invalidate | 263,001 | 262,747 | 262,579 | 262,561 |

For context, 1 MiB / 128 B is 8,192 lines and 32 MiB / 128 B is 262,144 lines.
The differences include instruction/stack/framework traffic; these are not
payload-exclusive or physical-memory-controller byte counters. Large-buffer
repeats do not exhibit the small-buffer cache-hit behavior.

Some CPU/GPU timings differ (e.g. the set-0 first 1 MiB read is 89.74 vs 59.38 us),
but bus/DDR clocks, shared-cache allocation, QoS, and background activity are not
held fixed. That difference is not attributed to Direct Link or a specific cause.
QuRT PMU/profile overhead also prevents treating these times as bare-link latency.

All set-0 raw `0x47`, `0x48`, and `0x49` deltas are zero, including broad windows.
**These events have no positive stimulus/control here. Zero therefore does not
establish that the ports are absent, supported, or unused by Direct Link.** The
processor manual does not provide the needed SM8650 Direct-Link-to-port mapping.

## Reconfiguration detection

Preliminary runs aborted on configuration mismatch. The observed replacement was
`PMUCFG=0x6000`, `PMUEVTCFG=0x463f55cd`, `PMUEVTCFG1=0x11007f03`. Its owner/cause
has not been isolated; no firmware/compiler defect is claimed.

The final implementation rejects any trial whose configuration differs at entry
or before/after either loop. It logs the rejection and retries with a fresh seed,
at most three attempts; payload errors are always fatal, never retried away.
Final sets 0/1/2/3 rejected 0/1/0/2 trials respectively. Two rejected trials were
warmups and one was a measured-index attempt. Rejected counters/times do not enter
the medians. This filters observed interference, but endpoint checks cannot rule
out a transient configuration change that is restored between checkpoints.

## Limits on the Direct Link conclusion

The observations are consistent with ordinary shared-buffer reads through the
primary memory interface and subsequent local cache hits. They do not identify a
GPU-only bypass in this transport. They cannot rule out a link behind that same
interface, a feature requiring another API/buffer type, or a path neutralized by
the remaining FastRPC cache maintenance.

The phone exposes `llcc-pmu`, but its type attribute is permission-denied and no
userspace event/format listing is exposed there. The `dcvs/bw_hwmon_meas` and
`bwprof_last_sample` tracepoints exist but their formats/enables are also denied.
`perf_event_paranoid=-1` does not remove these access restrictions. QProf is not
installed. No root, trace enable, system cache setting, clock override outside the
existing independent DSP power vote, or firmware/register-route change was made.

A conclusive identification still needs a documented exclusive link counter/
route selector or vendor mapping of the relevant ports. Measuring DRAM avoidance
alone would leave the ordinary system-cache alternative. A useful next isolation
test would avoid implicit input-buffer maintenance using an explicitly managed
persistent DSP mapping; it is not implemented in this investigation.

## Reproduction and raw evidence

```powershell
./tools/build_gpu_probe.ps1 -Push
0..3 | ForEach-Object {
    ./tools/run_gpu_probe.ps1 -Mode pmu -Repetitions 11 -EventSet $_
}
```

The runner rejects concurrent project inference/probes and uses a 60-second ADB
query bound per bank. Logs and hashes are local, ignored artifacts. The probe
requires stable configuration at the DSP checkpoints. The checker requires
schema v2, full case/phase/sample coverage, correct event IDs, zero payload errors,
recomputed summary statistics, and positive calibration. Sixteen parser tests include rejected
windows, old ABI rejection, zero-load calibration failure and duplicate-bank
mismatch rejection. Calibration does not certify the zero slave events.

| Set | Raw log | SHA-256 |
|---|---|---|
| 0 | `gpu-pmu-1791624710711.log` | `220c3f6c26161fb537eeca3cb2e477b44162d2c45e0d52b401dc20e07a70fa4d` |
| 1 | `gpu-pmu-1791624728166.log` | `a202407b4d68e9c41c880395ede3913809f5f3d73ddeb219f2b7685ff1a36e47` |
| 2 | `gpu-pmu-1791624747417.log` | `ea88caad300d28b9d08865066922bb04dba5feb9cc09d1514f81db65a1bbb8de` |
| 3 | `gpu-pmu-1791624767219.log` | `e6da9f2cd9ca55763d2f227d57c03b06d1550844448e99bd876cfb9d223897a6` |
