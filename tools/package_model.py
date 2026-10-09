"""Prepare the text-only deployment artifact and preserve tied quantized weights."""

import argparse
import json
import shutil
import struct
from pathlib import Path

from inspect_model import audit, read_headers


def validate_inventory(selected, reference_tensors):
    for name, original in reference_tensors.items():
        if not name.startswith("model.language_model.") or name.startswith("model.language_model.embed_tokens."):
            continue
        if name in selected:
            if selected[name]["shape"] != original["shape"]:
                raise ValueError(f"Raw tensor shape changed: {name}")
            continue
        prefix = name.removesuffix(".weight")
        packed = selected.get(prefix + ".trellis")
        if not name.endswith(".weight") or len(original["shape"]) != 2 or not packed:
            raise ValueError(f"Missing text-model tensor: {name}")
        n, k = original["shape"]
        if packed["dtype"] != "I16" or len(packed["shape"]) != 3 or packed["shape"][:2] != [k // 16, n // 16]:
            raise ValueError(f"Quantized tensor shape changed: {name}")
        for suffix, size in (("suh", k), ("svh", n)):
            scale = selected.get(prefix + "." + suffix)
            if not scale or scale["shape"] != [size] or scale["dtype"] != "F16":
                raise ValueError(f"Missing or malformed scales: {prefix}.{suffix}")


def package(source, reference, output):
    config = json.loads((reference / "config.json").read_text(encoding="utf-8"))
    text = config["text_config"]
    if config.get("model_type") != "qwen3_5" or not text.get("tie_word_embeddings"):
        raise ValueError("This packager currently supports tied Qwen3.5 text backbones only")
    tensors = read_headers(source)
    selected = {name: info for name, info in tensors.items()
                if name.startswith(("model.language_model.", "lm_head."))
                and not name.startswith("model.language_model.embed_tokens.")}
    validate_inventory(selected, read_headers(reference))
    head = selected.get("lm_head.trellis")
    if not head or head["dtype"] != "I16" or len(head["shape"]) != 3 or head["shape"][:2] != [text["hidden_size"] // 16, text["vocab_size"] // 16]:
        raise ValueError("The quantized output head has an unexpected shape")
    for key, size in (("lm_head.suh", text["hidden_size"]), ("lm_head.svh", text["vocab_size"])):
        info = selected.get(key)
        if not info or info["dtype"] != "F16" or info["shape"] != [size]:
            raise ValueError(f"Missing or malformed head scales: {key}")
    source_parameters = audit(reference, scope="text")["reference_parameter_count"]
    payload = sum(info["bytes"] for info in selected.values())
    if payload * 8 > source_parameters * 5:
        raise ValueError(f"Deployment weights exceed 5 bpw: {payload * 8 / source_parameters:.4f}")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    header = {"__metadata__": {"format": "pt", "hexagon_exl3_package": "1"}}
    cursor = 0
    for name, info in sorted(selected.items()):
        header[name] = {"dtype": info["dtype"], "shape": info["shape"],
                        "data_offsets": [cursor, cursor + info["bytes"]]}
        cursor += info["bytes"]
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    temporary = output / "model.safetensors.partial"
    with temporary.open("wb") as destination:
        destination.write(struct.pack("<Q", len(encoded)) + encoded)
        for name, info in sorted(selected.items()):
            with (source / info["file"]).open("rb") as stream:
                stream.seek(info["offset"])
                remaining = info["bytes"]
                while remaining:
                    chunk = stream.read(min(remaining, 8 * 1024 * 1024))
                    if not chunk:
                        raise ValueError(f"Source was truncated while copying {name}")
                    destination.write(chunk)
                    remaining -= len(chunk)
    temporary.rename(output / "model.safetensors")
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
                     "merges.txt", "vocab.json", "source.json"):
        origin = source / filename
        if not origin.exists():
            origin = reference / filename
        if origin.exists():
            shutil.copyfile(origin, output / filename)
    manifest = {"format": "hexagon-exl3", "version": 1,
                "input_embedding": {"type": "tied_linear", "source": "lm_head", "basis": "original",
                                    "token_axis": 1, "hidden_size": text["hidden_size"], "vocab_size": text["vocab_size"]},
                "kv_cache": "f16", "reference_parameter_count": source_parameters,
                "weight_payload_bytes": payload, "effective_bpw": payload * 8 / source_parameters,
                "excluded_components": ["vision", "mtp"]}
    (output / "hexagon-exl3.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = audit(output, reference, "text")
    if not report["within_5_bpw"]:
        raise ValueError("Post-write budget verification failed")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.source, args.reference, args.output), indent=2))


if __name__ == "__main__":
    main()
