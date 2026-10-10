# GPU/Hexagon communication and EXL3 dequantization probe

Measured 2026-10-10 (Asia/Tokyo), Xiaomi 14 Ultra, stock non-root Android 16,
Adreno 750 / Hexagon v75. This is an independent probe, not an inference-backend
switch. The model, production runtime, FP16 KV, and API configuration were not
changed. The API was already stopped and remains stopped.

## Outcome and scope

- A FastRPC `rpcmem_alloc` system-heap allocation can also be imported by this
  phone's Adreno OpenCL driver. GPU-produced bytes reach Hexagon, which XORs them
  using HVX and writes a second allocation; a GPU kernel verifies every returned
  word. No application CPU payload memcpy occurs in this round trip.
- All seven sizes, 128 B through 32 MiB, pass 3 warmups plus 31 measured rounds,
  with a different GPU-generated seed each round. Both explicit map/unmap
  handoffs and an experimental `clFinish`-only handoff pass. An independent DSP
  scalar full-input check is made at the beginning/end, outside timing.
- GPU EXL3 inner-weight decoding passes all 65,536 states in each of three
  codebooks (196,608 states). All 1--8 integer bitrates and three codebooks pass
  random packed-matrix checks in row-major and HMX layouts. The specialized
  4/6/8-bit mul1 native-record path passes as well.
- Real 4-bit K-projection and 6-bit head-slice fixtures are bit-identical to
  the unmodified upstream CUDA reconstruction, including after GPU-to-DSP
  round-trip. The CUDA oracle pin is `151539c77abc7ab7425d30da7a4e8e3c5c154e7b`.
- There is no GPU matrix multiplication. **Actual HMX consumption of these GPU
  buffers, overlapping execution, complete-model speed/quality, and sustained
  thermal/power behavior are not tested here.** No model speedup is claimed.

The 72 dequantization cases cover 51 small synthetic variants, six real-fixture
variants, nine single-group variants, and six 24 MiB output streaming variants.

## Driver/transport selection

OpenCL reports `OpenCL 3.0 QUALCOMM build: 0762.36.1 Compiler E031.45.02.25`, with
`cl_qcom_dmabuf_host_ptr`, `cl_qcom_ext_host_ptr`, and IO-coherent/on-chip extensions.
External allocation padding is zero and the reported page size is 4,096 bytes.
The NDK's legacy `cl_mem_ion_host_ptr` descriptor (`0x40a8`, writeback cache policy)
is accepted for rpcmem's DMA-BUF fd despite the legacy extension name not being
advertised. This is a verified driver-specific ABI, not a portability promise.
Map calls are checked to return the original rpcmem host address.

Vulkan reports external-memory-fd and Android Hardware Buffer extensions but
**not** `VK_EXT_external_memory_dma_buf` or `VK_EXT_external_memory_host`. The
probe does not pretend an opaque Vulkan fd is interchangeable with an arbitrary
DMA-BUF. Vulkan/AHardwareBuffer bridging has not been attempted; OpenCL is used
for the first GPU computation/communication test.

The official [Qualcomm OpenCL guide](https://docs.qualcomm.com/bundle/publicresource/80-NB295-11_REV_C_Qualcomm_Snapdragon_Mobile_Platform_Opencl_General_Programming_and_Optimization.pdf)
describes external-buffer imports in section 7.4.2 and on-chip global memory in
section 9.1.6. The latter preserves data only within the specified GPU execution/
recording lifetime; it does not document a Hexagon export path. Its presence is
not evidence of a usable GPU-to-NPU SRAM link.

This test establishes **shared-allocation interoperability**, not Hexagon Direct
Link use, GPU-cache-to-NPU-cache transfer, physical DRAM bypass, or absence of
internal driver copies. No suitable public programmable direct-link API has
been identified. CPU orchestration and synchronization are still present.

## Timing definitions

Times are microseconds unless stated. There are three warmups and 31 samples per
size/case. p50/p95 are sorted order statistics at `floor(q*(N-1))`. GPU frequency
is left under stock policy; the independent DSP session votes for maximum core/
bus corners. These are short USB-connected runs with no concurrent project
inference, not controlled sustained thermal tests.

- `gpu_us`: OpenCL event START--END, only GPU kernel execution.
- `total_us`: host clock from GPU enqueue to DSP copy/XOR return. Includes GPU
  completion, event-profile queries, optional map/unmap/finish, FastRPC dispatch,
  explicit DSP input invalidation, HVX work and output flush. Excludes buffer
  allocation/import, program compilation, output validation, and return-to-GPU
  ownership/verification. **Not a GPU--NPU round-trip latency.**
- `rpc_us`: host time around the DSP copy/XOR method.
- `hvx_copy_us`: DSP ticks / 19.2, only the single-HVX-context read/XOR/write loop;
  explicit input invalidation/output flush are outside this interval.
- `noop_rpc_us`: same-sized IDL argument call with no payload access, not a
  zero-byte RPC or a pure hardware dispatch measurement.
- `read_rpc_us`: subsequent DSP read/XOR-reduction call; returns only 128 bytes.
  `hvx_read_us` is its loop-only time. These are separate consumer measurements
  after the round-trip check, **not part of the GPU-production-to-DSP sample**.
- `payload_GBps`: one-way payload bytes / `total_us` / 1000. It includes a GPU
  producer and a DSP read/write consumer, so it is **not physical link bandwidth**.
- Read bandwidth divides input bytes by the separate read RPC or HVX-loop time,
  respectively. Those two denominators must not be conflated.

Validation runs outside timing and uses both GPU checks and DSP checks. Explicit
DSP invalidation/flush remains in both handoff modes. The finish-only observation
is not a general external-memory coherence guarantee and is not selected for
production. No CPU/GPU/DSP operations access the payload concurrently.

## Communication results

Explicit map/unmap handoff, medians:

| Payload | GPU produce -> DSP copy done | p95 | Payload effective rate | Separate DSP read RPC | HVX read loop only |
|---|---:|---:|---:|---:|---:|
| 128 B | 2,097.86 us | 2,701.56 us | 0.000061 GB/s | 680.365 us | 2.917 us |
| 4 KiB | 2,159.27 us | 2,927.55 us | 0.00190 GB/s | 680.469 us | 2.865 us |
| 32 KiB | 2,223.54 us | 2,923.33 us | 0.0147 GB/s | 688.281 us | 4.688 us |
| 256 KiB | 2,286.04 us | 2,905.73 us | 0.1147 GB/s | 714.271 us | 18.073 us |
| 1 MiB | 2,477.50 us | 2,944.69 us | 0.4232 GB/s | 808.594 us | 70.729 us |
| 8 MiB | 3,889.27 us | 4,612.60 us | 2.1569 GB/s | 1,690.00 us | 707.188 us |
| 32 MiB | 7,613.59 us | 8,234.22 us | 4.4072 GB/s | 4,268.23 us | 2,833.39 us |

At 32 MiB, the separate read RPC is 7.861 GB/s including dispatch/cache work;
the HVX read loop alone is 11.843 GB/s. Neither is the Direct Link's bandwidth.

Experimental finish-only medians are 1,784.90 us at 128 B, 1,755.26 us at 32 KiB,
3,299.79 us at 8 MiB, and 7,137.55 us / 4.7011 GB/s at 32 MiB. Both modes have zero
mismatches across 217 measured rounds each. Different modes were run in sequence,
not interleaved ABBA; their small differences are not a causal optimization claim.

Small transfers are dominated by host/queue/RPC/cache overhead in **this probe**.
For example, the explicit-map 128 B sample has a 5.888 us GPU kernel and 3.177 us
HVX copy loop but 2.10 ms host-observed completion. The model's dspqueue backend
does not use this probe's per-transfer IDL call, so this is not its latency floor.

## GPU EXL3 results

`layout=0` uses row-major output; `1` writes HMX layout from upstream tiles;
`2` uses bitrate-specialized/coalesced writes from losslessly reordered native
128x128 records. The native record stride includes scale fields, which are dummy
padding in this fixture probe: scales are not applied by inner-weight decoding.
The actual inference pipeline must retain/apply real scales and H128 transforms.

Real fixture results from `gpu-dequant-v3.log`:

| Fixture | Layout | GPU only p50 | GPU only p95 | GPU -> DSP copy completion p50 |
|---|---|---:|---:|---:|
| 4-bit K=2048,N=512 | row-major | 279.040 us | 281.088 us | 2,172.55 us |
| 4-bit K=2048,N=512 | HMX, upstream tiles | 242.944 us | 243.200 us | 2,376.88 us |
| 4-bit K=2048,N=512 | HMX, native records | 179.968 us | 182.016 us | 2,455.94 us |
| 6-bit K=2048,N=128 head slice | row-major | 60.928 us | 61.952 us | 1,966.82 us |
| 6-bit K=2048,N=128 head slice | HMX, upstream tiles | 60.928 us | 61.184 us | 2,244.43 us |
| 6-bit K=2048,N=128 head slice | HMX, native records | 65.792 us | 66.048 us | 2,019.22 us |

Single 128x128 groups take approximately 10--11 us on the GPU, but approximately
1.6--2.6 ms through the serialized GPU/map/RPC-copy path. A synthetic
2048x6144 four-bit native-record chunk outputs 24 MiB in 2,454.02 us (10.255 GB/s
of decoded output), or 8,156.88 us through the DSP copy completion. Its six-bit
equivalent is 3,101.95 us (8.113 GB/s), or 9,167.29 us through DSP completion.

These are warmed repeated buffers, not model-wide weight streaming or an
optimized pipelined implementation. The native four-bit kernel is faster in GPU
event time than the generic one, but that does not establish end-to-end speedup
over HVX. No same-process four-worker HVX versus GPU/HMX inference comparison was
made. The prior 4.35 tokens/s accepted model baseline remains unchanged.

## Reproduction and evidence

Build/deploy only the independent files, with no overwrite of runtime libraries:

```powershell
./tools/build_gpu_probe.ps1 -Push
./tools/run_gpu_probe.ps1 -Mode interop
./tools/run_gpu_probe.ps1 -Mode interop-finish
./tools/run_gpu_probe.ps1 -Mode dequant -CudaFixtures
```

CUDA fixtures are private local generated data in `benchmark-raw/real-k-proj`
and `benchmark-raw/real-head`; omit `-CudaFixtures` to run the synthetic/exhaustive
checks alone. Device files are under
`/data/local/tmp/xiaomi-hexagon-exl3/gpu-probe`. The runner limits its ADB query to
60 seconds; on timeout it reports unknown device state and does not restart ADB
or overwrite libraries. Large repetition counts may exceed that bound.

The checker requires terminal success, full case/size/sample coverage, finite
positive timing, correct units, and zero mismatch counts. Eight parser regression
tests cover complete interop/dequant logs, missing success, incomplete coverage,
required CUDA fixtures, NaN/negative time, mismatch, and bandwidth unit errors.
Portable codec/model-reader tests also pass. The CPU-only Python checks used by
CI pass; unrestricted host discovery additionally attempts a CUDA test which
cannot import because this host Python has no PyTorch. That is not recorded as a
passing CUDA test.

Local raw evidence (ignored, not committed):

| File | SHA-256 |
|---|---|
| `gpu-interop-map-v2.log` | `379e5c1ba4d8457b57f8fc31628ec95e38ce602a379cbe1791ead2bb8f51ae38` |
| `gpu-interop-finish-v2.log` | `870d6410ed1e8f13fce96663ef30aeae045e7b5b277831d507fc54e15a99310a` |
| `gpu-dequant-v3.log` | `9d2d531fe518e93c14feb824accf400291f93d17fa07f87e821a2174b542330a` |

Next engineering gate, not implemented here: larger asynchronous chunks and
double buffering, genuine HMX consumption from the GPU output, and a matched
HVX-vs-GPU end-to-end comparison. Without reducing handoff overhead, moving
individual 32 KiB groups to the GPU is not an attractive implementation.
