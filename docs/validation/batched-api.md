# Batched API bring-up

Device: Xiaomi 14 Ultra, stock Android 16 / HyperOS, Hexagon v75. Runtime: pinned llama.cpp plus the project's EXL3 overlay. Model: Index-Translate-2B, 150 four-bit matrices and a six-bit tied head, 4.806 bpw repack payload, FP16 KV.

## Changes required for the service

- The host Flash Attention support predicate rejected all output tensors with more than one sequence, although the DSP HMX kernel already iterates over sequences. The overlay enables up to eight, checking matching Q/K/V/output sequence counts. Nine component cases (1/4/8 sequences, 1/3/17 queries, per-sequence FP16 masks) pass against CPU; worst normalized squared error is 2.54e-6.
- Chat defaults now explicitly disable thinking, matching the reference's already-closed thinking prefix. Without this, a two-token chat response used its budget for thinking tags and returned empty translation content.
- Legacy completion streams previously attached final usage to the final choice rather than a separate empty-choices event. The overlay fixes the ordering and adds null usage to ordinary chunks. It also carries one request creation timestamp through partial/final results rather than generating a new time for every event.
- A local bearer key is generated/deployed outside Git. The phone listens only on loopback. Startup/status/stop commands identify the project's actual process rather than trusting stale PID metadata.

## Observed tests

| Check | Evidence |
|---|---|
| Non-streaming completions and chat | Both return `The cat`, correct model/choice/usage schema, 21 prompt tokens and two generated tokens. |
| Streaming completions and chat | Both deliver incremental text, a final choice, empty-choices usage, and `[DONE]`; IDs and creation timestamps stay constant. |
| Eight simultaneous requests | Eight active slots observed; all eight complete translations exactly match the tied-head CUDA references. Prompt-cache reuse is disabled in the requests. |
| Eight simultaneous SSE clients | Eight active slots observed; all eight reference translations match, response IDs are unique, timestamps remain stable, and usage/DONE events are valid. |
| Continuous admission | After a short request finishes, a replacement enters while earlier requests still run. The seven remaining initial translations and the replacement's full translation match CUDA. Peak active slots remains eight. |
| Cancellation | Client closes after receiving real text from a 256-token, ignore-EOS stream. The slot becomes idle, and a subsequent request produces the expected dog translation. |
| Serialization tests | CPU and Android tests cover usage-enabled/disabled streams and stable timestamps without needing model inference. |
| Real Python SDK | OpenAI Python 3.27.0 streams both completions/chat endpoints, reconstructs `The cat`, validates two generated tokens and total usage, and observes stable request IDs/timestamps. |

The eight reference translations are: cat sleeping, dog running, student, sunny weather, tomorrow's rain, interesting book, two coffees, and station location. These short, public examples test state separation and transport; they are not a comprehensive translation benchmark.

The eight-SSE run takes 133.95-146.12 seconds per request including prefill. The single-request baseline remains approximately 0.25 decode tokens/s. This is a working correctness baseline, not an acceptable-performance result. Large compressed-matrix decode/multiply kernels dominate the profile. The 15 tokens/s, sustained load, actual 8k context, and 32k context gates remain unmet/unverified.

Generated raw references, response reports, logs, and the API key stay under ignored `benchmark-raw`. Public source fixtures contain only project-authored test sentences. Authentication checks also verify that a request without the bearer key receives HTTP 401 with a JSON error.
