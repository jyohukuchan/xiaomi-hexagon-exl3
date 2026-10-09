# Exact packed integer rounding

The mul1 decoder now combines its final right shift and 16-bit packing with `Q6_Vh_vasr_VwVwR_rnd_sat`. The exact integer numerator and FP32 exponent rebias are unchanged. For FP32 magnitude bits `m`, the input code is `m - 0x38000001 + ((m >> 13) & 1)`. The instruction supplies the rounding addition of 4,096 before shifting by 13, which gives the same nearest-even result as the previous explicit `m + 4,095 + tie` calculation. Signs are packed separately and aligned to the instruction's interleaved lane order.

All possible mul1 values are normal FP16. Portable exhaustive tests also prove that signed-word addition cannot overflow and signed-half saturation cannot activate. This is integer rounding, not the previously rejected QFloat-to-half conversion.

## Verification

- All 65,536 mul1 states match the portable decoder bit for bit on v75, for both the old arithmetic reference and the new packed path.
- The standalone probe retains modes 0 (old arithmetic), 1 (small-table candidate), and 2 (packed rounding), and reports correctness and cycles for each.
- Standalone scalar/HVX tile and matrix checks pass. Both native portable tests pass with the additional exhaustive packed-rounding proof.
- All 84 runtime HMX/HVX, FP32/FP16-input matrix-plus-residual cases pass, with maximum NMSE 1.03e-7.
- All 24 token-100 residual files have identical SHA-256 hashes to the fixed-group-helper runtime.
- The full two-sentence translation and all eight simultaneous SSE translations exactly match CUDA text references. Peak active slots is eight; end-to-end SSE latency including prefill is 8.56-9.04 seconds.
- Continuous replacement admission, cancellation/slot reuse, and SDK 3.27 checks for both endpoints pass.

## Speed

The VTCM-only probe repeats each input vector 64 times with volatile input and output: old arithmetic takes 5,085,676 cycles and packed rounding takes 4,429,763 cycles in the measured run. The model's four-bit group-decode median is 7,743-7,765 cycles, versus 8,615-8,644 previously. The complete profile includes all 151 EXL3 matrices, totaling 197,579 microseconds, including 56,750 microseconds for the six-bit head. These component measurements are separate from generation throughput.

Same isolated short-context generation: 8k capacity, 56 actual prompt tokens, 25 decode runs, batch 128, microbatch 4, eight CPU threads, greedy sampling, no warmup, and no conversation wrapping.

| Metric | Separate word rounding | Packed integer rounding |
|---|---:|---:|
| Prompt / 56 tokens | 3,234.74 ms | 3,041.33 ms |
| Decode / 25 runs | 6,100.26 ms | 5,753.10 ms |
| Decode rate | 4.10 tokens/s | 4.35 tokens/s |

This approximately 6% short-run gain does not meet 15 tokens/s. Serialized payload stays 4.805952430792548 bpw and KV stays FP16. Sustained and long-context numerical/text checks remain pending.
