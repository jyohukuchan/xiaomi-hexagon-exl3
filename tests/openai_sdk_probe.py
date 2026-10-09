"""Verify the real OpenAI Python SDK against the local phone service."""
import argparse
import json
from pathlib import Path
import sys
import time
import openai

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from api_probe import Client

parser = argparse.ArgumentParser()
parser.add_argument("reference", type=Path)
parser.add_argument("--api-key-file", type=Path, required=True)
parser.add_argument("--base-url", default="http://127.0.0.1:8088")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
Client(args.base_url)  # Reject external endpoints before constructing the SDK client.
if args.output.exists():
    parser.error("Output already exists")
case = json.loads(args.reference.read_text(encoding="utf-8"))["cases"][0]
results = []
with openai.OpenAI(base_url=args.base_url.rstrip("/") + "/v1", api_key=args.api_key_file.read_text(encoding="ascii").strip(),
                  timeout=600, max_retries=0) as client:
    for chat in (False, True):
        started = time.monotonic()
        options = {"model": "index-translate-2b-exl3", "max_tokens": 2, "temperature": 0,
                   "stream": True, "stream_options": {"include_usage": True}, "extra_body": {"cache_prompt": False}}
        stream = client.chat.completions.create(messages=[{"role": "user", "content": case["user_prompt"]}], **options) if chat else client.completions.create(prompt=case["prompt"], **options)
        text, usage, identity, created = "", None, None, None
        with stream:
            for chunk in stream:
                identity = identity or chunk.id
                created = chunk.created if created is None else created
                assert chunk.id == identity and chunk.created == created
                if not chunk.choices:
                    usage = chunk.usage
                else:
                    text += (chunk.choices[0].delta.content or "") if chat else chunk.choices[0].text
        assert text == "The cat" and case["expected"].startswith(text)
        assert usage is not None and usage.completion_tokens == 2
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
        result = {"chat": chat, "text": text, "seconds": time.monotonic() - started, "usage": usage.model_dump()}
        results.append(result)
        print("PASS SDK", json.dumps(result), flush=True)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps({"sdk_version": openai.__version__, "passed": True, "results": results}, indent=2), encoding="utf-8")
