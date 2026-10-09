# EXL3 model artifact

The first conversion and native model-reader checks completed on 2026-10-10.

## Quantization

- HF source: IndexTeam/Index-Translate-2B, revision `a516854233b170140b57d36b48d7148c0edebabb`
- Quantizer: ExLlamaV3 `151539c77abc7ab7425d30da7a4e8e3c5c154e7b`, with the project's bounded-memory scale-refit adapter
- Calibration: 32 rows x 1,024 tokens, upstream bundled corpus mix
- Codebook: mul1
- Main matrices: 150 matrices at 4 bits
- Shared output head: one matrix at 6 bits
- Sensitive small projection/norm/conv tensors retain floating-point precision

The initial conversion completed the 24 backbone layers but failed at the large output-head scale refit with `CUDA driver error: device not ready`. The adapter chunks the input-side Gram matrix, elementwise reductions, and rescaling; it avoids full-head elementwise temporary copies. Four small CUDA tests comparing it with the unmodified upstream refit passed with relative/absolute tensor tolerance 2e-5 and proxy-error tolerance 1e-7. The job resumed from its checkpoint and finished.

## Deployment weight budget

- Text backbone parameter count, counting the tied embedding/head once: 1,881,825,088
- Serialized tensor payload: 1,073,099,996 bytes
- Effective payload rate including scales and markers: 4.56195425533619 bpw
- Tensor count: 773
- Weight budget gate: passed (at most 5 bpw)

The deployment packager verifies each source text tensor is present either in floating-point form or as correctly shaped trellis/scales. Vision and MTP tensors are omitted from both the deployed payload and budget denominator. The input embedding is an alias to original-basis columns of the quantized head; a second embedding matrix is not stored.

## Real-weight reconstruction

Compared against the unmodified upstream CUDA reconstruction:

| Tensor | Shape | Bits | Inner weights | Max relative error | Normalized error |
|---|---|---:|---|---:|---:|
| Layer 3 attention K projection | 2048 x 512 | 4 | bit-exact | 1.91e-4 | 1.35e-10 |
| Head columns 124160..124287 | 2048 x 128 | 6 | bit-exact | 3.09e-4 | 1.28e-10 |

These checks passed on both Linux x86_64 and Android arm64. Slice reconstruction matched full portable reconstruction bit for bit.

## Native Android reader

The C++ reader maps the complete 1.07 GB deployment weight file, validates every stored span, resolves the embedding alias, and caches up to sixteen 128-token embedding groups. At hidden size 2,048 the cache limit is 8 MiB.

On the Xiaomi 14 Ultra:

- Native payload inventory: 773 tensors, 1,073,099,996 bytes, 4.56195 bpw
- Token 124160: 2,048 embedding coefficients
- First embedding lookup: 15.8889 ms
- Repeated lookup: bit-identical cache hit
- Embedding normalized error versus the CUDA reference column: 1.99e-12

The native reader is a model-storage component. Full EXL3 model execution, translation regression evaluation, continuous batching, and HTTP API integration remain pending. The changed shared-head input embedding must be included in that translation evaluation.
