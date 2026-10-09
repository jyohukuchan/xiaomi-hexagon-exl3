# Third-party notices

The EXL3 decoding equations, tensor-core tile layout, and H128 reconstruction in `src/exl3_codec.cpp` and `src/hexagon/exl3_service.c` are ported from ExLlamaV3, commit `151539c77abc7ab7425d30da7a4e8e3c5c154e7b`:

- `exllamav3/exllamav3_ext/quant/codebook.cuh`
- `exllamav3/exllamav3_ext/quant/exl3_dq.cuh`
- `exllamav3/exllamav3_ext/quant/reconstruct.cu`

Copyright (c) 2025 Turboderp. The original MIT license is included in `licenses/exllamav3.txt`.

The pinned ExLlamaV3 checkout in the conversion Docker image retains its original copyright and license. Model weights are downloaded separately and remain under their source license (Apache-2.0 for Index-Translate-2B).

The `third_party/llama.cpp` submodule is pinned to `a11f57ba93797579a5d1855ee216a31f10242676`. It is Copyright (c) 2023-2026 The ggml authors and distributed under its included MIT `LICENSE`. Its graph, scheduler, tokenizer, and Hexagon backend provide the model-integration foundation.
