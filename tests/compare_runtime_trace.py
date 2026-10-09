"""Compare contiguous residual streams from native or upstream CUDA traces."""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("reference", type=Path)
parser.add_argument("actual", type=Path)
parser.add_argument("--cuda-reference", action="store_true")
parser.add_argument("--limit", type=float, default=1e-4)
parser.add_argument("--layers", type=int, default=24)
args = parser.parse_args()
if not 0 < args.layers <= 24 or not np.isfinite(args.limit) or args.limit <= 0:
    parser.error("Invalid layer count or error limit")

def load(directory, name, cuda=False):
    metadata = json.loads((directory / (name + ".json")).read_text())
    data = np.fromfile(directory / (name + ".bin"), dtype="<f4")
    if cuda:
        size = int(np.prod(metadata["shape"]))
    else:
        if metadata["type"] != 0:
            raise ValueError(f"Expected FP32 tensor: {name}")
        shape = metadata["ne"]
        stride = 4
        for extent, actual_stride in zip(shape, metadata["nb"]):
            if actual_stride != stride:
                raise ValueError(f"Non-contiguous tensor: {name}")
            stride *= extent
        size = int(np.prod(shape))
    if data.size != size or not np.isfinite(data).all():
        raise ValueError(f"Invalid trace values: {name}")
    return data.astype(np.float64)

results = []
for layer in range(args.layers):
    native_name = f"l_out-{layer}"
    reference_name = f"model.language_model.layers.{layer}.output" if args.cuda_reference else native_name
    reference = load(args.reference, reference_name, args.cuda_reference)
    actual = load(args.actual, native_name)
    if reference.shape != actual.shape:
        raise ValueError(f"Shape mismatch at layer {layer}")
    error = actual - reference
    norm = np.dot(reference, reference)
    nmse = float(np.dot(error, error) / max(norm, 1e-30))
    results.append({"layer": layer, "nmse": nmse, "max_abs_error": float(np.max(np.abs(error)))})
print(json.dumps({"limit": args.limit, "passed": all(r["nmse"] <= args.limit for r in results), "layers": results}, indent=2))
raise SystemExit(0 if all(r["nmse"] <= args.limit for r in results) else 1)
