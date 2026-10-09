"""Exercise the loopback phone API without calling an external inference service."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def check_usage(value):
    assert isinstance(value, dict), "Missing usage"
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        assert type(value.get(key)) is int and value[key] >= 0, f"Invalid {key}"
    assert value["total_tokens"] == value["prompt_tokens"] + value["completion_tokens"], "Usage does not sum"


def check_completion(value, chat):
    assert value["object"] == ("chat.completion" if chat else "text_completion")
    assert isinstance(value["id"], str) and value["id"]
    assert type(value["created"]) is int and isinstance(value["model"], str)
    assert len(value["choices"]) == 1 and value["choices"][0]["index"] == 0
    choice = value["choices"][0]
    assert choice["finish_reason"] in ("stop", "length")
    check_usage(value["usage"])
    if chat:
        assert choice["message"]["role"] == "assistant"
        assert isinstance(choice["message"]["content"], str)
        return choice["message"]["content"]
    assert isinstance(choice["text"], str)
    return choice["text"]


class Client:
    def __init__(self, base, key=""):
        parsed = urlsplit(base)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost") or parsed.username or parsed.query:
            raise ValueError("This probe only permits a local HTTP endpoint")
        self.base = base.rstrip("/")
        self.key = key

    def open(self, path, payload=None):
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Connection": "close"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        if data is not None:
            headers["Content-Type"] = "application/json"
        return urlopen(Request(self.base + path, data=data, headers=headers), timeout=600)

    def json(self, path, payload=None):
        with self.open(path, payload) as response:
            assert response.status == 200
            data = response.read(2 * 1024 * 1024 + 1)
            assert len(data) <= 2 * 1024 * 1024
            return json.loads(data)

    def payload(self, case, chat, count, stream=False):
        result = {"model": "index-translate-2b-exl3", "temperature": 0, "max_tokens": count,
                  "stream": stream, "cache_prompt": False}
        result.update({"messages": [{"role": "user", "content": case["user_prompt"]}]} if chat else {"prompt": case["prompt"]})
        if stream:
            result["stream_options"] = {"include_usage": True}
        return result

    def complete(self, case, chat, count):
        started = time.monotonic()
        value = self.json("/v1/chat/completions" if chat else "/v1/completions", self.payload(case, chat, count))
        text = check_completion(value, chat)
        assert case["expected"].startswith(text) and text, f"Reference mismatch: {case['id']}: {text!r}"
        assert value["usage"]["completion_tokens"] <= count
        if value["choices"][0]["finish_reason"] == "stop":
            assert text == case["expected"], "Early stop does not match the full reference"
        result = {"case": case["id"], "id": value["id"], "chat": chat, "text": text, "seconds": time.monotonic() - started, "usage": value["usage"]}
        print("PASS completion", json.dumps(result, ensure_ascii=False), flush=True)
        return result

    def stream(self, case, chat, count, cancel=False):
        started = time.monotonic()
        payload = self.payload(case, chat, count, True)
        if cancel:
            payload["ignore_eos"] = True
        text, events, identity, usage, finished, done = "", 0, None, None, False, False
        timestamp = None
        with self.open("/v1/chat/completions" if chat else "/v1/completions", payload) as response:
            assert "text/event-stream" in response.headers.get("Content-Type", "")
            while line := response.readline(1024 * 1024):
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                assert chunk["object"] == ("chat.completion.chunk" if chat else "text_completion")
                identity = identity or chunk["id"]
                assert chunk["id"] == identity, "Stream ID changed"
                assert chunk["model"] == "index-translate-2b-exl3"
                assert type(chunk["created"]) is int
                timestamp = chunk["created"] if timestamp is None else timestamp
                assert chunk["created"] == timestamp, "Stream creation timestamp changed"
                if not chunk["choices"]:
                    assert finished and usage is None, "Usage chunk is out of order"
                    check_usage(chunk["usage"])
                    usage = chunk["usage"]
                    continue
                assert len(chunk["choices"]) == 1
                assert chunk.get("usage") is None, "Usage attached to a non-final chunk"
                choice = chunk["choices"][0]
                assert choice["index"] == 0
                if chat and "role" in choice["delta"]:
                    assert choice["delta"]["role"] == "assistant"
                fragment = choice["delta"].get("content") if chat else choice["text"]
                if fragment:
                    assert not finished, "Content after the final choice"
                    text += fragment
                    events += 1
                    if cancel:
                        break
                if choice["finish_reason"] is not None:
                    assert choice["finish_reason"] in ("stop", "length")
                    finished = True
        if not cancel:
            assert done and finished and usage is not None, "Incomplete SSE sequence"
            assert usage["completion_tokens"] <= count
            assert text and case["expected"].startswith(text), f"SSE reference mismatch: {text!r}"
        else:
            assert text and not done, "Cancellation did not interrupt active content"
        result = {"case": case["id"], "id": identity, "created": timestamp, "chat": chat, "cancelled": cancel, "text": text, "content_events": events,
                  "seconds": time.monotonic() - started, "usage": usage, "done": done}
        print("PASS stream", json.dumps(result, ensure_ascii=False), flush=True)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8088")
    parser.add_argument("--mode", choices=("smoke", "batch", "stream", "cancel", "continuous"), default="smoke")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--stream-batch", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stream_batch and args.mode != "batch":
        parser.error("--stream-batch requires --mode batch")
    if args.output.exists():
        parser.error("Output already exists")
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    key = args.api_key_file.read_text(encoding="ascii").strip() if args.api_key_file else ""
    client = Client(args.base_url, key)
    assert client.json("/health")["status"] == "ok"
    slots = client.json("/slots")
    assert len(slots) == 8 and all(s["n_ctx"] == 8192 and not s["is_processing"] for s in slots)
    models = client.json("/v1/models")
    assert any(m["id"] == "index-translate-2b-exl3" for m in models["data"])
    if key:
        try:
            Client(args.base_url).json("/v1/models")
        except HTTPError as error:
            assert error.code == 401 and isinstance(json.loads(error.read())["error"], dict)
        else:
            raise AssertionError("An unauthenticated request was accepted")
    rendered = client.json("/apply-template", {"messages": [{"role": "user", "content": reference["cases"][0]["user_prompt"]}]})
    assert rendered["prompt"] == reference["cases"][0]["prompt"], "Chat prompt differs from the CUDA reference"
    results = []
    peak = 0
    if args.mode == "smoke":
        for chat in (False, True):
            results.append(client.complete(reference["cases"][0], chat, 2))
    elif args.mode == "stream":
        for chat in (False, True):
            results.append(client.stream(reference["cases"][0], chat, 4))
    elif args.mode == "cancel":
        results.append(client.stream(reference["cases"][0], True, 256, cancel=True))
        deadline = time.monotonic() + 60
        while any(s["is_processing"] for s in client.json("/slots")):
            assert time.monotonic() < deadline, "Cancelled task did not release its slot"
            time.sleep(1)
        results.append(client.complete(reference["cases"][1], True, 2))
    elif args.mode == "batch":
        barrier = threading.Barrier(8)
        def run(case):
            barrier.wait(timeout=10)
            return (client.stream if args.stream_batch else client.complete)(case, True, reference["max_tokens"])
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            pending = [pool.submit(run, case) for case in reference["cases"]]
            while not all(f.done() for f in pending):
                busy = sum(s["is_processing"] for s in client.json("/slots"))
                if busy > peak:
                    peak = busy
                    print("Active slots:", peak, flush=True)
                time.sleep(1)
            results = [f.result() for f in pending]
        assert peak == 8, "Eight simultaneous active slots were not observed"
        assert len({r["id"] for r in results}) == 8, "Concurrent response IDs are not unique"
        assert all(r["text"] == c["expected"] for r, c in zip(results, reference["cases"], strict=True)), "Batch did not produce full reference translations"
    elif args.mode == "continuous":
        barrier = threading.Barrier(8)
        def run(index, case):
            barrier.wait(timeout=10)
            return client.complete(case, True, 2 if index == 0 else reference["max_tokens"])
        late_during_existing = False
        with concurrent.futures.ThreadPoolExecutor(max_workers=9) as pool:
            initial = [pool.submit(run, i, case) for i, case in enumerate(reference["cases"])]
            late = None
            while not all(f.done() for f in initial) or late is None or not late.done():
                slots = client.json("/slots")
                peak = max(peak, sum(s["is_processing"] for s in slots))
                if initial[0].done() and late is None:
                    initial[0].result()
                    late_during_existing = any(not f.done() for f in initial[1:])
                    assert late_during_existing, "No running requests remain for continuous admission"
                    print("Admitting replacement while earlier requests remain active", flush=True)
                    late = pool.submit(client.complete, reference["cases"][0], True, reference["max_tokens"])
                time.sleep(1)
            results = [f.result() for f in initial] + [late.result()]
        assert peak == 8 and late_during_existing
        assert results[-1]["text"] == reference["cases"][0]["expected"]
        assert all(r["text"] == c["expected"] for r, c in zip(results[1:8], reference["cases"][1:], strict=True))
    assert not any(s["is_processing"] for s in client.json("/slots")), "Slots remain active"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"mode": args.mode, "stream_batch": args.stream_batch, "passed": True, "peak_active_slots": peak, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
