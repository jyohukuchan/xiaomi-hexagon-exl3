"""Audit serialized weight storage and EXL3 formats from safetensors headers."""

import argparse
import json
import math
import struct
from pathlib import Path

DTYPE_BYTES = {"BOOL": 1, "I8": 1, "U8": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
               "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}


def read_headers(directory):
    tensors = {}
    for path in sorted(directory.glob("*.safetensors")):
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f"Truncated safetensors file: {path}")
            length, = struct.unpack("<Q", prefix)
            if length > 64 * 1024 * 1024:
                raise ValueError(f"Unreasonable safetensors header: {path}")
            data = stream.read(length)
            if len(data) != length:
                raise ValueError(f"Truncated safetensors header: {path}")
        header = json.loads(data)
        payload_size = path.stat().st_size - 8 - length
        spans = []
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if name in tensors:
                raise ValueError(f"Duplicate tensor: {name}")
            shape = info["shape"]
            if not all(isinstance(dim, int) and dim >= 0 for dim in shape):
                raise ValueError(f"Invalid shape: {name}")
            start, end = info["data_offsets"]
            elements = math.prod(shape)
            dtype = info["dtype"]
            if dtype not in DTYPE_BYTES:
                raise ValueError(f"Unsupported dtype {dtype}: {name}")
            if start < 0 or end < start or end > payload_size or end - start != elements * DTYPE_BYTES[dtype]:
                raise ValueError(f"Invalid tensor span: {name}")
            spans.append((start, end))
            tensors[name] = {"shape": shape, "dtype": dtype, "elements": elements,
                             "bytes": end - start, "file": path.name, "offset": 8 + length + start}
        cursor = 0
        for start, end in sorted(spans):
            if start != cursor:
                raise ValueError(f"Non-contiguous or overlapping payload: {path}")
            cursor = end
        if cursor != payload_size:
            raise ValueError(f"Unaccounted payload: {path}")
    if not tensors:
        raise ValueError(f"No safetensors in {directory}")
    return tensors


def audit(directory, reference=None, scope="all"):
    all_tensors = read_headers(directory)
    def select(tensors):
        if scope == "all":
            return tensors
        return {name: value for name, value in tensors.items()
                if name.startswith(("model.language_model.", "lm_head."))}
    tensors = select(all_tensors)
    if not tensors:
        raise ValueError("No tensors found in the selected scope")
    payload = sum(t["bytes"] for t in tensors.values())
    ref = select(read_headers(reference)) if reference else tensors
    parameters = sum(t["elements"] for t in ref.values())
    matrices = []
    embeddings = []
    for name, tensor in tensors.items():
        if name.endswith(".trellis"):
            shape = tensor["shape"]
            prefix = name.removesuffix(".trellis")
            signs = tensors.get(prefix + ".signs")
            if signs is not None:
                if len(shape) != 2 or tensor["dtype"] != "I16" or signs["dtype"] != "F16" or len(signs["shape"]) != 2 or signs["shape"][1] != 256:
                    raise ValueError(f"Invalid EXL3 embedding: {name}")
                groups = signs["shape"][0]
                if not groups or shape[1] % groups:
                    raise ValueError(f"Invalid EXL3 embedding groups: {name}")
                words = shape[1] // groups
                if (words - 1) % 16 or not 1 <= (words - 1) // 16 <= 8:
                    raise ValueError(f"Invalid EXL3 embedding bitrate: {name}")
                embeddings.append({"name": prefix, "rows": shape[0], "width_padded": groups * 256,
                                   "bits": (words - 1) // 16})
                continue
            if len(shape) != 3 or tensor["dtype"] != "I16" or shape[2] % 16:
                raise ValueError(f"Unsupported EXL3 trellis: {name}, {shape}")
            bitrate = shape[2] // 16
            if not 1 <= bitrate <= 8:
                raise ValueError(f"Invalid EXL3 bitrate: {name}")
            matrices.append({"name": prefix, "in_features": shape[0] * 16,
                             "out_features": shape[1] * 16, "bits": bitrate,
                             "codebook": "mul1" if prefix + ".mul1" in tensors else
                                         "mcg" if prefix + ".mcg" in tensors else "3inst"})
    return {"directory": str(directory), "scope": scope, "tensor_count": len(tensors), "payload_bytes": payload,
            "all_payload_bytes": sum(t["bytes"] for t in all_tensors.values()),
            "reference_parameter_count": parameters, "effective_bpw": payload * 8 / parameters,
            "within_5_bpw": payload * 8 <= parameters * 5, "exl3_matrices": matrices,
            "exl3_embeddings": embeddings}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--require-budget", action="store_true")
    parser.add_argument("--scope", choices=("all", "text"), default="all")
    args = parser.parse_args()
    report = audit(args.directory, args.reference, args.scope)
    print(json.dumps(report, indent=2))
    if args.require_budget and (not args.reference or not report["within_5_bpw"]):
        raise SystemExit("EXL3 weight payload failed the 5 bpw budget gate")


if __name__ == "__main__":
    main()
