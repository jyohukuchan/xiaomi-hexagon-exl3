# FP16 model baseline

The reference model was executed on the Xiaomi 14 Ultra on 2026-10-10 using the pinned llama.cpp backend. This run validates the graph and device path before EXL3 integration.

- Source HF revision: `a516854233b170140b57d36b48d7148c0edebabb`
- llama.cpp commit: `a11f57ba93797579a5d1855ee216a31f10242676`
- Text weights converted from BF16 to FP16 GGUF; KV storage FP16
- Device selection: `HTP0`, all model layers offloaded
- Context capacity: 8,192 tokens; request batch cap 128; microbatch cap 32
- Greedy sampling, thinking disabled in the raw prompt

The 56-token Japanese-to-English test prompt in `tests/prompts/translation_ja_en.txt` produced:

> This is a test of the translation engine that runs on the smartphone's NPU. It can translate without a network connection.

Measured without per-op profiling:

- Prompt processing: 384.34 ms / 56 tokens, 145.71 tokens/s
- Decode: 2,430.72 ms / 25 runs, 10.29 tokens/s
- Model load: 6,209.49 ms

A separate verbose profile confirmed Gated Delta Net on `hvx-recurrent-scalar` and full attention on `hmx-pipe`. At this configuration the backend reported 3,590.35 MiB of HTP model storage, 96 MiB of HTP FP16 KV, and 19.27 MiB of HTP recurrent state. It also reported a 970 MiB CPU model buffer for the embedding path.

The default larger microbatch attempted to map a 587,984,896-byte work buffer and aborted with FastRPC mmap error 0x1. Reducing the microbatch to 32 resolved that allocation for this run.

This is a short-request FP16 baseline, not proof of EXL3 inference, sustained 15 tokens/s, 8k/32k filled-context behavior, concurrent request isolation, or API completion. Those gates remain pending.
