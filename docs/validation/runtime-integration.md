# Runtime integration status

The runtime overlay adds EXL3 4/6/8-bit weight types to the pinned llama.cpp graph and routes their matrix operations to the project's Hexagon kernel. Input embedding lookup uses the shared quantized head on the CPU. FP16 KV and the existing HVX Gated Delta Net / HMX attention paths are retained.

## Verified component checks

- HVX bit decoder: 4/6/8 bits x three codebooks, nine cases, bit-exact against the portable reference.
- Graph matrix operations: real 4-bit K projection, real 6-bit head slice, and synthetic 8-bit matrix; batch 1/4/8/32 on CPU and HTP0, all 24 cases passed (normalized error below 5e-8).
- Graph embedding lookup: CPU 6-bit head slice agrees with the CUDA weight reference (normalized error 3.72e-11).
- Runtime repack payload: 1,130,495,232 bytes / 4.805952430792548 bpw, including duplicated group scales.
- Packed 4-bit K-projection bytes agree with an independently assembled 128x128 group layout.
- Floating-point auxiliary tensors agree with the FP16 reference, with at most 4.77e-7 difference in the exp-transformed A_log tensors.

The 128x128 record stores 64 original 16x16 trellis tiles, 128 FP16 input scales, and 128 FP16 output scales. It preserves the trellis and scale bit patterns; no weight requantization occurs.

## Issues fixed during bring-up

- The SDK 19 compiler's FP16 auto-vectorized codebook initializer produced incorrect table entries. Disabling vectorization for that one-time initializer restored reference values.
- A 6-bit tile occupies 192 bytes, so alternate tile pointers are only 64-byte aligned. The HVX decoder now uses unaligned vector loads for these pointers.
- SSM convolution weights must remain FP32 in this graph. The exporter now retains FP32 for those tensors.
- The host fused EXL3 matrix multiplication with a residual ADD even though the custom kernel did not implement ADD. This dropped residual connections unless tracing split the operation batch. EXL3 now rejects that fusion, and the DSP rejects accidental fused-op dispatch. Other supported fusions remain enabled.
- The model loader tests matrix support at 512 rows. A whole-batch EXL3 scratch allocation rejected the 6,144-input FFN down projections, putting 24 large matrices on the CPU. The kernel now processes at most 32 rows at a time within the available VTCM. All 151 EXL3 matrices, including those 24 down projections, appear in the on-device DSP profile.

## First end-to-end translation

The residual-fusion fix restored the expected Japanese-to-English translation on the phone. With the raw, already formatted prompt, conversation wrapping disabled, greedy sampling, context capacity 8,192, microbatch 4, and FP16 K/V, it generated:

> This is a test of the translation engine that runs on the smartphone's NPU. It can translate without a network connection.

This matches upstream CUDA, including the CUDA run with shared 6-bit input embeddings. The first correct run still used the 24 CPU FFN down-projection fallbacks: prompt 1.03 tokens/s (56 tokens), decode 0.23 tokens/s (25 decode runs).

After row chunking put all 151 EXL3 matrices on the NPU, the same raw prompt produced the same complete translation at 1.11 prompt tokens/s (56 tokens / 50,536.31 ms) and 0.24585 decode tokens/s (25 runs / 101,687.11 ms). Conditions: stock phone, FP16 K/V, context capacity 8,192, actual initial prompt 56 tokens, logical batch 128, microbatch 4, eight CPU threads, temperature zero, no warmup, no conversation wrapping. The requested output cap was 32 tokens; generation ended at EOS. This is a short correctness baseline, not a sustained 8k-context run. The 15 tokens/s gate is not met.

## Whole-graph and batch regression checks

- For input token 100, all 24 native residual streams are compared to the portable CPU graph. After fusion and row-chunk fixes, worst normalized squared error is 9.72e-6; final-layer error is 2.40e-6. Against upstream CUDA, the earlier fusion-fixed run has worst error 2.64e-4, including implementation precision differences.
- Residual ADD graph tests cover three matrices (4/6/8 bits) at batch 1/4/8/32 on CPU and HTP0: all 24 cases pass. HTP0 additionally passes batch 37/128/512 for all three matrices, including the partial last row chunk; error is below 3e-8 in these nine cases.
- The first all-EXL3-on-NPU single-token profile totals 4.029 seconds in 151 EXL3 kernels, including 1.100 seconds for the shared 6-bit vocabulary head. Tracing and profiling are diagnostic runs, not decode-throughput measurements.

The remaining gates are translation regression coverage, continuous batching/API, sustained throughput, and context-size validation. The current HVX decoder/matrix kernel is a correctness baseline; major performance work remains.

The single-token trace reports 378.91 MiB of CPU model buffers and 1,078.13 MiB of HTP model buffers. The tied head is one serialized tensor but currently has CPU and device-side runtime copies for embedding lookup/output projection. Serialized bpw does not describe this extra runtime memory. The completion run also reports an unexpectedly large CPU scratch allocation (378.94 MiB instead of the reserved estimate); this needs investigation during memory/performance optimization.
