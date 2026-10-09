"""Download the pinned reference model without embedding credentials in the repo."""

import argparse
import json
import os
import urllib.request
from pathlib import Path

MODEL = "IndexTeam/Index-Translate-2B"
REVISION = "a516854233b170140b57d36b48d7148c0edebabb"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("models/index-translate-2b-bf16"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{MODEL}/revision/{REVISION}") as response:
        metadata = json.load(response)
    filenames = [item["rfilename"] for item in metadata["siblings"] if "/" not in item["rfilename"]]
    for filename in filenames:
        if not filename.endswith((".json", ".safetensors", ".txt", ".md")):
            continue
        target = args.output / filename
        if target.exists():
            print(f"Already present: {filename}", flush=True)
            continue
        partial = target.with_name(target.name + ".partial")
        print(f"Downloading {filename}", flush=True)
        request = urllib.request.Request(f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{filename}")
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
            total = int(response.headers.get("Content-Length", 0))
            received = 0
            last_report = 0
            while chunk := response.read(8 * 1024 * 1024):
                stream.write(chunk)
                received += len(chunk)
                if received - last_report >= 256 * 1024 * 1024:
                    print(f"  {received / 2**20:.0f} MiB" + (f" / {total / 2**20:.0f} MiB" if total else ""), flush=True)
                    last_report = received
            if total and received != total:
                raise RuntimeError(f"Incomplete download of {filename}: {received} != {total}")
        os.replace(partial, target)
    (args.output / "source.json").write_text(json.dumps({"repo": MODEL, "revision": REVISION}, indent=2) + "\n")
    print(f"Model ready: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
