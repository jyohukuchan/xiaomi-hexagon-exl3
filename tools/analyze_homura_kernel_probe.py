"""Audit real-model Hexagon routing; stored weight types are not instruction types."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

MODEL_HASHES = {
    "Q4_K_M": "3fc56945f1db4c2b91b18ac9f4061b354d51c226a6fdd9ea95e72a927f13a675",
    "Q8_0": "e3aa4850c1f4779328af5d721383bb5785dfdd7836f5a82976336c21178c7754",
}


def profiles(text):
    result = []
    for line in text.splitlines():
        if "profile-op MUL_MAT" not in line:
            continue
        fields = line.split("profile-op ", 1)[1].split("|")
        if len(fields) != 7:
            raise ValueError("Incomplete matrix profile")
        operation, names, dims, types, strides, params, timing = fields
        if not operation.startswith("MUL_MAT"):
            raise ValueError("Unexpected operation")
        output_shape = [int(value) for value in dims.split(" -> ")[-1].split(":")]
        if len(output_shape) != 2 or min(output_shape) <= 0:
            raise ValueError("Invalid matrix output dimensions")
        path = params.split()[0]
        if path not in ("hmx-tiled", "hvx-tiled"):
            raise ValueError("Unrecognized kernel path")
        names_in = names.split(" -> ")[0].split(" x ")
        types_in = types.split(" -> ")[0].split(" x ")
        if len(names_in) != len(types_in):
            raise ValueError("Input name/type count mismatch")
        weights = {name: kind for name, kind in zip(names_in, types_in) if name.endswith(".weight")}
        if not weights:
            raise ValueError("No weight tensor in matrix profile")
        usecs = re.search(r"\busec (\d+) cycles (\d+)\b", timing)
        if not usecs:
            raise ValueError("Missing native timing counters")
        result.append({"op": operation, "names": names, "input_columns": output_shape[1],
                       "weights": weights, "path": path, "usecs": int(usecs[1]), "cycles": int(usecs[2])})
    if not result:
        raise ValueError("Generation succeeded but no matrix profile was captured")
    return result


def scheduler_matrix_assignments(text):
    result = Counter(re.findall(r"node #\s*\d+\s+\(\s*MUL_MAT[^)]*\):.*?\[\s*(\S+)\s+", text))
    if not result or set(result) != {"HTP0"}:
        raise ValueError("Missing scheduler matrix evidence or a non-NPU assignment")
    return dict(result)


def summarize(directory):
    runs = [json.loads(line) for line in (directory / "runs.jsonl").read_text().splitlines()]
    if len(runs) != 2 or {r["format"] for r in runs} != set(MODEL_HASHES):
        raise ValueError("Expected one accepted profile per requested model")
    output = []
    for run in runs:
        fmt = run["format"]
        if run["exit_code"] or run["abort"] or run["model_sha256"] != MODEL_HASHES[fmt]:
            raise ValueError("Failed/interrupted process or unexpected model artifact")
        raw = (directory / (fmt + ".log")).read_bytes()
        if hashlib.sha256(raw).hexdigest() != run["raw_sha256"]:
            raise ValueError("Raw profile hash mismatch")
        text = raw.decode("utf-8")
        records = profiles(text)
        assignments = scheduler_matrix_assignments(text)
        if "こんにちは" not in text or "[end of text]" not in text or "offloaded 26/26 layers" not in text:
            raise ValueError("Missing expected smoke output, end marker or complete layer offload")
        all_weights = {name: kind for r in records for name, kind in r["weights"].items()}
        if len(all_weights) != 187:
            raise ValueError("Incomplete active quantized-weight tensor coverage")
        expected_types = {"q4_K", "q6_K"} if fmt == "Q4_K_M" else {"q8_0"}
        if set(all_weights.values()) != expected_types:
            raise ValueError("Unexpected stored weight types")
        if {(r["input_columns"], r["path"]) for r in records} != {(1, "hvx-tiled"), (30, "hmx-tiled")}:
            raise ValueError("Unexpected or incomplete selected matrix routes")
        counts = Counter((r["input_columns"], r["path"], next(iter(r["weights"].values()))) for r in records)
        output.append({"format": fmt, "model_sha256": run["model_sha256"], "raw_sha256": run["raw_sha256"],
                       "executed_matrix_profile_records": len(records), "active_weight_tensors": len(all_weights),
                       "active_weight_types": dict(Counter(all_weights.values())), "scheduler_matrix_nodes": assignments,
                       "groups": [{"input_columns": k[0], "path": k[1], "first_weight_type": k[2], "records": count}
                                  for k, count in sorted(counts.items())],
                       "examples": [next(r for r in records if r["input_columns"] == columns) for columns in (30, 1)]})
    return {"dataset": directory.name, "model_revision": "40a26e36b43c64c9495b71ebccb59fc230fed6b0",
            "baseline_revision": "a11f57ba93797579a5d1855ee216a31f10242676", "models": output,
            "interpretation_limit": "Runtime logs select HMX/HVX paths; arithmetic interpretation additionally requires matching source/binary inspection. This is not an instruction-by-instruction hardware trace or translation-quality evaluation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = json.dumps(summarize(args.directory), indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(value + "\n", encoding="utf-8")
    else:
        print(value)


if __name__ == "__main__":
    main()
