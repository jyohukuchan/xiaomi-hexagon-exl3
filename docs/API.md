# Local FP16-KV service

The patched native server runs on the stock phone. It uses eight continuous-batching slots, 8,192-token capacity per slot, FP16 K/V, and the shared EXL3 6-bit head. The runtime payload remains 4.806 bpw. These are bring-up settings, not a claim of acceptable speed or tested 8k-length prompts.

## Start and stop

After building/deploying the runtime and preparing `models/index-translate-2b-exl3-v2.hxgguf` on the phone:

```powershell
.\tools\serve_model.ps1 -Action Start
.\tools\serve_model.ps1 -Action Status
.\tools\serve_model.ps1 -Action Stop
```

Start generates a random local API key if absent, deploys it with device mode 600, launches the server through a hidden ADB bridge, and forwards host port 8088 to phone port 8080. It reuses an existing project server. Stop identifies processes by both project working directory and model command before sending SIGINT; it does not delete weights or credentials.

The phone binds only `127.0.0.1`. The host endpoint is `http://127.0.0.1:8088/v1`. Credentials are in the ignored `benchmark-raw/device-api.key`, never in Git. Do not publish that file or expose the service on a network without reviewing authentication and access controls. Stop leaves the ADB forwarding rule for reuse.

## Python SDK example

```powershell
python -m venv build-api-venv
.\build-api-venv\Scripts\python.exe -m pip install -r tests/requirements-api.txt
```

```python
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8088/v1",
    api_key=Path("benchmark-raw/device-api.key").read_text().strip(),
    timeout=600,
)
stream = client.chat.completions.create(
    model="index-translate-2b-exl3",
    messages=[{"role": "user", "content": "日语翻译成英语：猫が眠っています。"}],
    temperature=0,
    max_tokens=12,
    stream=True,
    stream_options={"include_usage": True},
)
for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
    elif chunk.usage:
        print("\n", chunk.usage)
```

The default chat template disables thinking for translation. Raw `/v1/completions` prompts must already include the model's template; `/v1/chat/completions` accepts ordinary messages and applies it once. Streaming uses per-request stable IDs/timestamps, a separate empty-choices usage event when requested, then `[DONE]`, following the [Chat Completions streaming schema](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events) and [Completions reference](https://developers.openai.com/api/reference/resources/completions/methods/create).

The service selects the [validated HMX matrix path](validation/hmx-bringup.md) by default. `GGML_HEXAGON_EXL3_HMX=0` selects the HVX fallback when launching directly on the device. KV remains FP16 in both paths; HMX uses FP16 activations/partial results for its matrix arithmetic.

## Verification

Generate the eight public regression references in the pinned CUDA converter container:

```powershell
docker run --rm --gpus all --volume C:\coding-local\xiaomi-hexagon-exl3:/workspace --workdir /workspace `
  xiaomi-exl3-quantize:dev /opt/quant-venv/bin/python tests/reference_api_cases.py `
  models/index-translate-2b-exl3-4 --output benchmark-raw/api-reference.json
```

Run `tools/api_probe.py` with that reference, `--api-key-file benchmark-raw/device-api.key`, a fresh `--output` path, and one of `--mode smoke`, `stream`, `cancel`, `batch`, or `continuous`. Add `--stream-batch` to `--mode batch` to exercise eight simultaneous SSE clients. Probes reject external endpoints. The optional `tests/openai_sdk_probe.py` runs both streaming endpoints through the real Python SDK.

Compatibility validation is for text completions/chat, bearer authentication, usage, SSE, cancellation, and continuous request batching. It does not establish compatibility with every OpenAI endpoint or parameter. Vision, audio, tool calling, hosted Batch jobs, and the Responses API are outside this initial validation scope. Actual translation-quality, sustained throughput, and long-context coverage remain separate gates.
