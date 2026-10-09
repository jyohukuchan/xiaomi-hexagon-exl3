# EXL3 codec validation

The initial component gate was run on the Xiaomi 14 Ultra and the development PC on 2026-10-09.

## Upstream CUDA oracle

- Upstream ExLlamaV3 commit: `151539c77abc7ab7425d30da7a4e8e3c5c154e7b`
- PyTorch 2.10.0 with CUDA 13.0, RTX 4070
- Test matrix: 256 x 384, random packed data and non-unit signed channel scales
- Cases: integer bitrate 1 through 8, codebooks 3inst/mcg/mul1 (24 cases)
- Independent reference: the unmodified upstream `reconstruct.cu`, plus an explicit float32 H128 matrix reference

Results on both Linux x86_64 and Android arm64:

- Inner decoded FP16 matrix: bit-exact in all 24 cases
- Original-basis H128 reconstruction: maximum relative error at most 0.000289855
- Normalized mean-square error: at most 1.57e-10

## Hexagon service

- Stock non-root CDSP session, v75 shared library built with Hexagon SDK 6.6.0.0 / Tools 19.0.07
- All 24 integer-bitrate/codebook combinations decoded bit-exactly against the portable reference
- mul1 matrix operations with bitrates 4, 6, and 8, each at batch 1/4/8: nine cases passed
- Matrix output normalized error: between 3.75e-8 and 4.98e-8 versus the original-basis FP16 weight reference

The Hexagon matrix path includes input scales, H128, packed trellis decoding, matrix accumulation, output H128, and output scales. It allocates activation/result scratch and one decoded 16 x 16 tile. It does not hold a whole expanded weight matrix.

These checks establish a correct scalar reference for future HVX/HMX optimization. They do not establish full-model inference throughput, translation fidelity, API compatibility, or the 15 tokens/second target.
