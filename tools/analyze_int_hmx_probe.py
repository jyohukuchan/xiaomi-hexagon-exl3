"""Audit isolated integer-HMX evidence; never infer rail power or hardware peak."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def positive(value):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("Nonpositive/nonfinite measurement")
    return value


def near(a, b):
    if not math.isfinite(a) or not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError("Measurement arithmetic mismatch")


def audit(run, allow_profile=False):
    if run["exit_code"] or run["abort"] or run["parse_errors"]:
        raise ValueError("Failed/incomplete experiment")
    rows = run["records"]
    def one(event):
        values = [r for r in rows if r["event"] == event]
        if len(values) != 1: raise ValueError("Missing/duplicate " + event)
        return values[0]
    init, result, finished = one("init"), one("result"), one("finished")
    if (run.get("profile") or init.get("profile")) and not allow_profile:
        raise ValueError("Profiled runs are not timing baselines")
    if result["stopped"] or finished["stopped"] or result["calls"] != run["repeats"]:
        raise ValueError("Interrupted/incomplete repetitions")
    for key in ("bits", "batch", "n", "k", "repeats"):
        if init[key] != run[key]: raise ValueError("Parameter mismatch")
    if init["bits"] not in (4, 8) or not 1 <= init["batch"] <= 512 or not 32 <= init["n"] <= 6144 or not 32 <= init["k"] <= 14336 or init["n"] % 32 or init["k"] % 32:
        raise ValueError("Invalid dimensions")
    engine = init.get("engine", "macro")
    legacy = engine == "macro" and "api_batch" not in init
    if legacy and init["batch"] % 64:
        raise ValueError("Legacy macro record cannot establish partial-row padding")
    api_batch = init.get("api_batch", init["batch"])
    route = init.get("route", "auto")
    expected_rows = ((init["batch"] + 31) // 32 * 32 if route == "hmx" else init["batch"]) if engine == "ggml" else (init["batch"] + 63) // 64 * 64
    if api_batch != expected_rows:
        raise ValueError("Unexpected padding")
    if not init["boot_s"] < result["boot_s"] < finished["boot_s"]:
        raise ValueError("Invalid lifecycle order")
    validations = [r for r in rows if r["event"] == "validation"]
    if not validations: raise ValueError("No correctness validation")
    for v in validations:
        if not init["boot_s"] < v["boot_s"] < result["boot_s"]:
            raise ValueError("Validation outside lifecycle")
        if v["mismatches"] or v["guard_bytes"] != 4096:
            raise ValueError("Output corruption")
        count = init["batch"] * init["n"] if v["full"] else min(128, init["batch"] * init["n"])
        if v["samples"] != count: raise ValueError("Insufficient reference samples")
        if "nmse" in v:
            tolerance = 5e-4 if init.get("engine") == "ggml" else 1e-10
            if not math.isfinite(v["nmse"]) or not 0 <= v["nmse"] < tolerance:
                raise ValueError("Reference error gate failed")
    if engine == "macro":
        if one("version")["value"] != "1_0_0_beta1_HEXAGON_V75": raise ValueError("Wrong library architecture")
        if len(validations) != 2: raise ValueError("Missing before/after macro validation")
    elif result["arch"] != 75:
        raise ValueError("Wrong architecture")
    if run.get("tensor"):
        mapping = one("gguf_mapping")
        if init["tensor"] != run["tensor"] or mapping["weight_elements"] != init["n"] * init["k"]:
            raise ValueError("Tensor reconstruction mismatch")
        if mapping["format"] != ("q8_0" if init["bits"] == 8 else "q4_K") or not 0 <= mapping["nmse"] < 1e-12:
            raise ValueError("GGUF reconstruction error")
    seconds = positive(result["seconds"])
    near(result["us_per_call"], seconds * 1e6 / result["calls"])
    near(result["effective_GOPS"], 2 * init["batch"] * init["n"] * init["k"] * result["calls"] / seconds / 1e9)
    if engine == "ggml":
        if result["physical_GOPS"] is not None: raise ValueError("GGML physical padding is not measured")
    else:
        near(result.get("physical_GOPS", result["effective_GOPS"]) if legacy else result["physical_GOPS"], result["effective_GOPS"] * api_batch / init["batch"])
    return {"engine": engine, "route": route, "profile": init.get("profile", False), "legacy_unpadded_macro_schema": legacy, "bits": init["bits"], "batch": init["batch"], "api_batch": api_batch,
            "n": init["n"], "k": init["k"], "mode": "scaled" if init.get("scaled") else "raw",
            "tensor": run.get("tensor"), "model_sha256": run.get("model_sha256"), "raw_sha256": run["raw_sha256"],
            "reference_samples": validations[-1]["samples"], "full_reference": validations[-1]["full"],
            "nmse": validations[-1].get("nmse", 0), "repeats": result["calls"],
            "us_per_call": result["us_per_call"], "effective_GOPS": result["effective_GOPS"],
            "physical_GOPS": result.get("physical_GOPS", result["effective_GOPS"]), "scope": result["scope"],
            "prepack_seconds": result.get("pack_seconds")}


def summarize(directory):
    runs = [json.loads(s) for s in (directory / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    output = []
    for run in runs:
        path = directory / (run["label"] + ".log")
        if hashlib.sha256(path.read_bytes()).hexdigest() != run["raw_sha256"]: raise ValueError("Raw log hash mismatch")
        records = [json.loads(s[8:]) for s in path.read_text(encoding="utf-8").splitlines() if s.startswith("INTEGER ")]
        if records != run["records"]: raise ValueError("Cached records differ from raw log")
        output.append(audit(run))
    telemetry_path = directory / "telemetry.jsonl"
    import run_accelerator_bench as monitor
    samples = [json.loads(s) for s in telemetry_path.read_text(encoding="utf-8").splitlines()]
    if not samples or any(monitor.stop_reason(s) for s in samples): raise ValueError("Unsafe/nonmatched phone state")
    provenance_path = directory / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else None
    if provenance: provenance["arguments"].pop("serial", None)
    return {"dataset": directory.name, "telemetry_sha256": hashlib.sha256(telemetry_path.read_bytes()).hexdigest(),
            "max_npu_C": max(s["npu_max_C"] for s in samples), "provenance": provenance,
            "cases": output}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directories", type=Path, nargs="+")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    summary = {"power_scope": "No integer-route power ranking: short/unmatched phase windows, no accelerator rail measurement",
               "datasets": [summarize(d) for d in args.directories]}
    result = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    if args.output: args.output.write_text(result, encoding="utf-8")
    else: print(result)


if __name__ == "__main__": main()
