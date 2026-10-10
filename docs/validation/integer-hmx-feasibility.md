# Integer HMX feasibility and adoption decision — 2026-10-10

## Decision

**Integer HMX is accessible on the stock v75 phone through official HexKL APIs.
The current GGUF block-scaled adapter is not suitable for production inference at
any tested batch. Keep the existing HVX / FP16-HMX backend routing unchanged.**

This is a decision about these implementations, not a claim that integer HMX is
intrinsically slow, that no faster implementation is possible, or that every
Homura tensor/shape has been exhaustively optimized. HexKL remains a possible
production dependency under its own license, as explicitly authorized by the user.

The subsequent [existing-engine survey](hexagon-integer-engines.md) identifies
QNN-based W8A8 and LPBQ paths. The [matched HMX/HVX measurements](../benchmarks/2026-10-11-hmx-hvx.md)
evaluate the existing FP16-HMX route separately. Neither changes the conclusion
about this particular integer Micro prototype or proves a general failure of
blockwise quantization on Hexagon.

## Baseline and isolation

First established a [non-profiled, fixed-token real-model baseline](../benchmarks/2026-10-10-homura-baseline.md):
Q4_K_M 19.298 decode steps/s and Q8_0 14.991 steps/s, with FP16 K/V. These are
teacher-forced matched-work timings, not free-running translation acceptance or
the original EXL3 model/API goal. The original EXL3 implementation, model payload,
FP16 KV, and API remain unchanged.

All integer tests run in separate FastRPC processes and device directories. No
CPU/GPU fallback is implemented in the adapter's matrix product. HMX is explicitly
reserved; integer entry points are identified by the official API contract and
validated outputs. This is **not a proprietary-library disassembly, instruction
trace, or independent hardware-counter proof of every dispatched instruction**.

## What works

- HexKL runtime reports `1_0_0_beta1_HEXAGON_V75`.
- CPU Macro API `sdkl_npu_mm_u8i8_i32` and `sdkl_npu_mm_u8i4_i32` both run.
  Small 64×128×256 tests match all 8,192 CPU INT64-reference outputs exactly,
  before and after timing. Large tests validate 128 stratified outputs, all
  output-written sentinels, padded zero rows and a 4 KiB tail guard.
- Both CPU APIs reject an unpadded M=1 with status 14: this is a **Macro API**
  multiple-of-64 requirement, not proof that hardware cannot handle partial rows.
- NPU Micro integer tile APIs also run. The reference `hexkl_micro_hmx_lock()`
  returned `0x80000600` here; real-device HAP compute-resource allocation and HMX
  locking resolved it. One MiB VTCM is reserved for this independent probe.
- v75 FP32 HVX epilogues must use QFloat operations and conversion back to IEEE
  floats. A bring-up version using direct IEEE-HVX arithmetic produced zeros and
  was rejected. Replacing that path resolved the error without loosening the
  correctness gate. Temporary debug writes were removed before accepted runs.
- Actual Homura `blk.0.attn_gate.weight` (N=2048,K=2048) and
  `blk.0.ffn_gate.weight` (N=6144,K=2048) are tested in both official models.
  Q4_K_M supplies Q4_K tensors here; Q8_0 supplies Q8_0. Every weight's affine
  reconstruction matches the existing GGUF decoder, with reconstruction NMSE=0.
  Scaled HMX outputs pass the 1e-10 NMSE gate against a CPU double reference.

The small scaled fixtures use a full output reference; large fixtures use 128
stratified samples plus all-output finite/written checks and output bounds guards.
This is matrix correctness, not end-to-end translation quality validation.

## Raw capacity is not GGUF throughput

The Macro API sweep uses N=4096,K=14336, ten calls, weights prepacked once, and
zero-pads small logical batches to 64 rows. Its timing includes SDKL dispatch,
layout/transfer work but excludes one-time weight preprocessing.

| Logical batch | Executed rows | INT8 call, ms | INT4 call, ms | INT4 useful GOPS |
|---:|---:|---:|---:|---:|
| 1 | 64 | 31.196 | 1.494 | 78.6 |
| 4 | 64 | 31.119 | 1.495 | 314.2 |
| 16 | 64 | 27.605 | 1.502 | 1251.1 |
| 64 | 64 | 27.872 | 1.495 | 5028.8 |
| 128 | 128 | 29.245 | 2.877 | 5224.6 |
| 512 | 512 | 46.683 | 16.149 | 3723.4 |

These are **unscaled integer products**, not Q4_K/Q8_0 matmuls, model throughput,
or bare hardware peak. The strong INT4 result does not establish INT8 hardware
capacity from its much slower API result. The straightforward Micro raw loop is
also slow (roughly 97/155 useful GOPS for INT8/INT4 at batch 64), demonstrating
that this simple loop is not a hardware-capacity measurement. That earlier Micro
control also predates the explicit bus-performance vote, so it is not a controlled
API-only comparison. API choice, power policy, tiling and data movement must all
be considered before attributing a performance difference to hardware.

## Why block scales matter

For each K32 block, let `a_u8 = a_i8 + 128`. Q8_0 requires

```
y += (dot(a_u8, w_i8) - 128*sum(w_i8)) * d_activation * d_weight
```

For Q4_K, map unsigned nibble q to signed nibble `w_i4=q-8`. With
`s=d*scale` and `c=8*s-dmin*min`, each block requires

```
y += d_activation * (s*(dot(a_u8,w_i4)-128*sum(w_i4))
                     + c*(sum(a_u8)-32*128))
```

Different scales for each output column and K32 block prevent simply accumulating
the entire K dimension in INT32 and scaling once. This prototype reads INT32
accumulators, reshapes each tile, applies these corrections/scales with HVX, and
accumulates FP32 after **every K32 block**. Weight sums are precomputed; activation
sums are computed once per activation block, not repeatedly per output column.
The existing FP16-HMX route can absorb weight scales during dequantization and
accumulate across more K blocks before reading results.

The final small-batch optimization clips accumulator reshaping and HVX epilogues
to the valid rows, rather than computing scales for all 64 padded rows. Matrix
hardware still executes a 64-row tile; padded outputs remain zero and pass guards.
This reduced attention-gate batch-1 Q4_K time from 212.363 ms to 8.374 ms, and
Q8_0 from 216.784 ms to 9.755 ms, without changing the measured numerical error.
The remaining gap still rejects adoption; unused-row postprocessing was not
mistaken for an unavoidable hardware cost.

Q6_K tensors inside Q4_K_M are not implemented in this adapter. They have K16
scales and no direct INT6 API here. Mapping them to INT8 with split/padded partial
products or another encoding adds work and requires separate measurements; they
remain on the existing route. The successful Q4_K tests do not mean all Q4_K_M
tensors use INT4 HMX.

## Matched real-tensor comparison

The [audited data](integer-hmx-feasibility-summary.json) records per-batch timings
for both actual tensors. Each engine pair receives the same tensor and activation
fixture. Activations are constructed in an already A8-representable form; **F32→A8
quantization cost is not included in the integer result**. This favors the integer
candidate. It is not a captured real-model activation/quality dataset.

Integer time is the DSP computation interval with prepacked weights, including
activation preparation/copy, integer tile operations, accumulator reads, reshape,
scale/correction epilogues and output stores. Host RPC, model tensor loading and
weight preprocessing are excluded. Existing-route time includes GGML host graph
dispatch and synchronization, but excludes upload/download and initial warmup.
These different scopes favor the integer candidate as well.

Measured 2048×2048 attention-gate tensor (milliseconds per matrix call):

| Batch | Existing Q8_0 | Integer Q8_0 | Existing Q4_K | Integer Q4_K |
|---:|---:|---:|---:|---:|
| 1 | 0.330 | 9.755 | 0.378 | 8.374 |
| 4 | 0.342 | 23.003 | 0.397 | 19.360 |
| 16 | 0.674 | 62.460 | 0.733 | 58.526 |
| 64 | 0.544 | 214.913 | 0.637 | 210.987 |
| 128 | 0.532 | 430.004 | 0.568 | 421.279 |
| 512 | 1.048 | 1721.328 | 0.874 | 1688.359 |

Measured 6144×2048 FFN-gate tensor (milliseconds per matrix call):

| Batch | Existing Q8_0 | Integer Q8_0 | Existing Q4_K | Integer Q4_K |
|---:|---:|---:|---:|---:|
| 1 | 0.574 | 29.832 | 0.587 | 25.690 |
| 64 | 0.636 | 645.145 | 0.714 | 632.929 |
| 512 | 1.718 | 5165.129 | 1.762 | 5061.319 |

Actual-tensor integer NMSE is at most 7.30e-14 against the already-quantized
activation reference. Existing-route sampled NMSE is at most 2.59e-6, below its
5e-4 gate. This does not establish better translation quality for the integer
route; activation quantization error is deliberately absent from this fixture.

Integer trials use one repetition at large shapes, versus twenty existing-route repetitions;
these exploratory timings are not sustained-load or tightly tuned crossover
measurements. Their order-of-magnitude gap is sufficient to reject enabling the
current adapter, but not to set a universal future threshold.

## Batch-switching policy

| Case | Current decision | Requirement to reconsider |
|---|---|---|
| Batch 1/4/16 | Keep existing backend choice | Remove/offset 64-row padding and dispatch/scale overhead; matched end-to-end win |
| Batch 64/128/512 | Keep existing backend choice | Fuse/accelerate block-scale epilogues and tile movement; faster real-tensor and model runs |
| Q6_K tensors | Existing fallback only | Correct K16 scale handling, padding/repacking and measured benefit |
| New per-channel / HMX-friendly quantization | Research candidate, not enabled | Quality and total-bpw gates, scales/activation cost and actual model benchmarks |

Do not implement a `batch>=64 -> integer HMX` rule based on the raw INT4 number.
A future selector must consider tensor type, K/N dimensions, scale granularity,
activation quantization, padding, memory footprint and measured benefit. There is
no demonstrated integer crossover for the tested exact-GGUF adapter.

The next useful engineering experiment is an optimized **DSP-resident Macro /
fused block-scale** path, or a separately evaluated HMX-friendly quantization.
The simple Micro loop is not an optimized integer implementation. Do not reject
the hardware or re-quantize the user's model merely to make this prototype win.

## Power and thermal limits

USB stayed unplugged and the screen off; project temperature, thermal-status,
charging and battery guards remained enabled. The Micro service makes ordinary
HAP performance votes with DCVS enabled; it does not disable system thermal
policies. Vendor Macro initialization is vendor-managed and not inspected.

**No integer-vs-existing power-efficiency ranking is accepted from these runs.**
Short and unequal compute windows, Macro/Micro power policies and unaligned phase
telemetry make such a ranking invalid. Only whole-device battery estimates are
available, not an NPU-only power rail. A viable optimized route must later pass
matched-duty, repeated whole-device joules/token and thermal tests.

## Licensing and reproduction

Project-authored adapters are MIT. HexKL headers, sources, static libraries and
shared libraries retain Qualcomm's separate license and are **not checked in**.
Compiled probe libraries that statically contain HexKL are ignored too. The
license accompanies local device deployment; no vendor binary disassembly or
reverse engineering was used.

Official archive: [Hexagon KL 1.0.0 Linux](https://softwarecenter.qualcomm.com/api/download/software/tools/Hexagon_KL/Linux/1.0.0/Hexagon_KL.Core.1.0.0.Linux-Any.zip).

1. Obtain the official archive separately and read its LICENSE/README before use.
   Pinned SHA256: `365ea693b279c4f267a098b1ce8c4e717dca763e8c2f156aa4269aaba62d8ea9`.
2. Extract the SDK-6.4 beta1 addon locally with
   `python tools/unpack_hexkl.py ARCHIVE build-int-hmx/hexkl-6.4-beta1`.
   The tested addon is HexKL 1.0.0 beta1 for SDK6.4. Build scripts expect its
   `hexkl_addon` subdirectory. They use the local SDK6.6,
   tool19/v75 and NDK29 container and a read-only independent llama.cpp baseline.
3. Build/deploy `tools/build_int_hmx_macro_probe.ps1 -Push -Serial SERIAL` or
   `tools/build_int_hmx_micro_probe.ps1 -Push -Serial SERIAL` with no concurrent
   model/probe process. They refuse to overwrite mapped project binaries.
4. Guarded real-weight experiment:

   ```text
   python tools/run_int_hmx_macro_probe.py --serial SERIAL --engine micro --scaled --gguf-tensor blk.0.attn_gate.weight --n 2048 --k 2048 --batches 1 4 16 64 128 512 --repeats 1
   python tools/run_int_hmx_macro_probe.py --serial SERIAL --engine ggml --gguf-tensor blk.0.attn_gate.weight --n 2048 --k 2048 --batches 1 4 16 64 128 512 --repeats 20
   ```

5. The same runner defaults to the CPU Macro API; `--pad64` explicitly permits
   padding. It retains failed runs and never automatically retries them away.
   Use conservative repetition counts: a remote DSP call is synchronous and
   SIGINT is handled only once it returns. No arbitrary long stress runs are needed.
6. Audit with `python tools/analyze_int_hmx_probe.py RAW_DIRECTORIES --output SUMMARY`;
   portable rejection tests are in `tests/test_int_hmx_probe.py`.

Raw data and models remain local ignored artifacts. Public summaries include
log hashes and, for later datasets, binary/source provenance. Earlier Macro runs
predate automatic provenance capture; this limitation is retained, not backfilled.

Final integer datasets are `int-hmx-micro-20261010-231440` and
`int-hmx-micro-20261010-231603`; existing-route references are
`int-hmx-ggml-20261010-230526` and `int-hmx-ggml-20261010-230802`.
Earlier accepted stages are retained in the summary to show the optimization's
effect, not pooled together as if they were the same implementation.

Verification: both Android clients and the v75 service build with warnings as
errors; 68 portable Python tests pass, including seven integer-probe audit tests.
The full Python discovery additionally encounters the unrelated CUDA-refit test,
which cannot import `torch` on this host. No CUDA verification is claimed here.
