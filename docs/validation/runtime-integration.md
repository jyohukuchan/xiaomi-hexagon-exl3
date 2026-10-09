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

## Unresolved end-to-end gate

The full EXL3 model loads and produces tokens, but the first translation run emitted mixed-language text. This is not a successful translation gate. The same quantized model gives the correct translation in upstream CUDA, both with original FP16 input embeddings and with input embeddings reconstructed from the 6-bit output head. The original FP16 phone graph also translates correctly at microbatch 4.

Observed faulty-run throughput: 0.23 tokens/s, prompt processing 1.03 tokens/s. The new EXL3 matrix kernel still requires substantial optimization to meet 15 tokens/s.

Remaining investigation: compare intermediate hidden states in the native graph with the CUDA reference, beginning with input embeddings and layer 0, and test serialized-runtime tensors directly rather than only fixture-built graph weights. Continuous batching/API and performance gates remain pending.
