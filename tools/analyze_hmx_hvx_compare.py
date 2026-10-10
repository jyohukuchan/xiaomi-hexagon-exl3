"""Audit matched HMX-FP16 / HVX GGUF comparisons; profiled runs never supply timings."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from analyze_int_hmx_probe import audit
import run_accelerator_bench as monitor
from run_hmx_hvx_compare import cases, profile_paths


def load(directory, profiled):
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    args = provenance["arguments"]
    if args["phase"] != ("profiles" if profiled else "timings"): raise ValueError("Wrong experiment phase")
    expected = {(b, m, t, r) for b, m, t, r in cases(args["bits"], args["batches"], args["trials"])}
    runs = [json.loads(s) for s in (directory / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    seen, output = set(), []
    for run in runs:
        path = directory / (run["label"] + ".log")
        raw = path.read_text(encoding="utf-8")
        if hashlib.sha256(path.read_bytes()).hexdigest() != run["raw_sha256"]: raise ValueError("Raw log hash mismatch")
        records = [json.loads(s[8:]) for s in raw.splitlines() if s.startswith("INTEGER ")]
        if records != run["records"]: raise ValueError("Cached records mismatch")
        item = audit(run, allow_profile=profiled)
        trial = int(run["label"].rsplit("-trial", 1)[1])
        key = (item["bits"], item["batch"], trial, item["route"])
        if key not in expected or key in seen: raise ValueError("Unexpected/duplicate case")
        if item["profile"] != profiled or bool(run.get("profile")) != profiled: raise ValueError("Profiling mismatch")
        if run["mode"] != "ggml-" + item["route"]: raise ValueError("Route metadata mismatch")
        if item["n"] != args["n"] or item["k"] != args["k"] or item["tensor"] != args["tensor"]: raise ValueError("Tensor mismatch")
        if item["repeats"] != (1 if profiled else args["calls"]): raise ValueError("Call count mismatch")
        if profiled:
            paths = profile_paths(raw)
            if set(paths) != {item["route"] + "-tiled"}: raise ValueError("Wrong executed matrix route")
            item["profile_records"] = len(paths)
        item["trial"] = trial
        output.append(item); seen.add(key)
    if seen != expected: raise ValueError("Incomplete comparison")
    samples = [json.loads(s) for s in (directory / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()]
    if not samples or any(monitor.stop_reason(s) for s in samples): raise ValueError("Unsafe/charging/awake phone state")
    provenance["arguments"].pop("serial", None)
    return {"dataset": directory.name, "provenance": provenance, "cases": output,
            "max_npu_C": max(s["npu_max_C"] for s in samples),
            "telemetry_sha256": hashlib.sha256((directory / "telemetry.jsonl").read_bytes()).hexdigest()}


def comparison(rows, trials):
    grouped = defaultdict(list)
    for row in rows: grouped[(row["bits"], row["batch"])].append(row)
    result = []
    for (bits, batch), group in sorted(grouped.items()):
        routes = {r: sorted([c for c in group if c["route"] == r], key=lambda c: c["trial"]) for r in ("hmx", "hvx")}
        if any(len(v) != trials or [c["trial"] for c in v] != list(range(trials)) for v in routes.values()): raise ValueError("Incomplete route pair")
        if len({(c["model_sha256"], c["tensor"], c["n"], c["k"]) for c in group}) != 1: raise ValueError("Unmatched tensor artifacts")
        averages = {r: statistics.mean(c["us_per_call"] for c in values) for r, values in routes.items()}
        ratios = [routes["hvx"][t]["us_per_call"] / routes["hmx"][t]["us_per_call"] for t in range(trials)]
        result.append({"bits": bits, "format": "Q4_K" if bits == 4 else "Q8_0", "batch": batch,
                       "hmx_api_rows": routes["hmx"][0]["api_batch"], "hvx_api_rows": routes["hvx"][0]["api_batch"],
                       "hmx_mean_us": averages["hmx"], "hvx_mean_us": averages["hvx"],
                       "hmx_speedup_over_hvx": averages["hvx"] / averages["hmx"],
                       "paired_speedup_min": min(ratios), "paired_speedup_max": max(ratios),
                       "trials": trials})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profiles", type=Path, nargs="+", required=True)
    p.add_argument("--timings", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    profiles = [load(d, True) for d in args.profiles]
    measurements = [load(d, False) for d in args.timings]
    for dataset in measurements:
        params = dataset["provenance"]["arguments"]
        matching = [d for d in profiles if all(d["provenance"]["arguments"][key] == params[key] for key in ("n", "k", "tensor"))]
        if len(matching) != 1: raise ValueError("Missing/ambiguous same-shape profile evidence")
        proof = matching[0]
        for key in ("binary_sha256", "backend_libraries", "baseline_commit"):
            if dataset["provenance"][key] != proof["provenance"][key]: raise ValueError("Profiling/timing binaries differ")
        for route in ("hmx", "hvx"):
            for bits in dataset["provenance"]["arguments"]["bits"]:
                if not any(c["route"] == route and c["bits"] == bits for c in proof["cases"]): raise ValueError("Missing precision/route execution proof")
        dataset["comparison"] = comparison(dataset["cases"], dataset["provenance"]["arguments"]["trials"])
    summary = {"scope": "Existing GGUF pipeline HMX-FP16 versus forced HVX, not integer-HMX peak or full-model serving speed",
               "power": "Safety telemetry only; no power-efficiency ranking from these short timing windows",
               "profiles": profiles, "measurements": measurements}
    text = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    if args.output: args.output.write_text(text, encoding="utf-8")
    else: print(text)


if __name__ == "__main__": main()
