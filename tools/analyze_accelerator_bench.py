"""Audit benchmark arithmetic/validation and summarize matched GPU/NPU cases."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def positive(value):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("Expected finite positive number")
    return value


def close(a, b):
    if not math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError(f"Arithmetic mismatch: {a} != {b}")


def audit_case(row):
    if not row["valid"] or row["exit_code"] != 0 or row.get("abort") or row.get("parse_errors"):
        raise ValueError("Unsuccessful process or rejected trial")
    records = row["records"]
    if row["operation"] not in ("mat", "copy"):
        raise ValueError("Unexpected operation")
    starts = [r for r in records if r["event"] == "start"]
    results = [r for r in records if r["event"] == "result"]
    validations = [r for r in records if r["event"] == "validation"]
    windows = [r for r in records if r["event"] == "window"]
    if len(starts) != 1 or len(results) != 1 or len(validations) != 2 or not windows:
        raise ValueError("Incomplete benchmark")
    start, result = starts[0], results[0]
    if result["stopped"] or start["backend"] != row["backend"] or start["format"] != row["format"] or start["batch"] != row["batch"]:
        raise ValueError("Backend/format mismatch or interrupted result")
    if row["backend"] not in ("GPUOpenCL", "HTP0") or not 0.1 <= start["duty"] <= 1 or start["duty"] != row["duty"]:
        raise ValueError("Unexpected backend/duty")
    if start["m"] != 4096 or start["k"] != (14336 if row["operation"] == "mat" else 4096):
        raise ValueError("Wrong comparison shape")
    count = start["m"] * start["k"]
    if row["operation"] == "mat":
        expected_bytes = {"f16": count * 2, "q8_0": count // 32 * 34, "q4_0": count // 32 * 18}[row["format"]]
        flops = 2 * count * row["batch"]
        for validation in validations:
            if validation["finite_elements"] != 4096 * row["batch"] or validation["reference_samples"] != 64:
                raise ValueError("Incomplete numerical validation")
            if not 0 <= validation["nmse"] <= 5e-4 or not math.isfinite(validation["max_abs"]):
                raise ValueError("Invalid numerical error")
        traffic = expected_bytes
    else:
        expected_bytes, traffic, flops = count * 2, count * 4, 0
        if any(v.get("bit_exact_elements") != count for v in validations):
            raise ValueError("Incomplete bit-exact copy check")
    if start["weight_bytes"] != expected_bytes or start["traffic_bytes_per_op"] != traffic or start["flops_per_op"] != flops:
        raise ValueError("Wrong traffic/FLOP accounting")
    for index, window in enumerate(windows):
        if window["index"] != index:
            raise ValueError("Noncontiguous timing windows")
        seconds, ops = positive(window["seconds"]), positive(window["ops"])
        close(window["us_per_op"], seconds * 1e6 / ops)
        close(window["gflops"], flops * ops / seconds / 1e9)
        close(window["effective_GBps"], traffic * ops / seconds / 1e9)
        close(window["active_gflops"], flops * ops / positive(window["active_seconds"]) / 1e9)
        if window["active_seconds"] > seconds:
            raise ValueError("Active time exceeds wall time")
    seconds, ops = sum(w["seconds"] for w in windows), sum(w["ops"] for w in windows)
    close(result["seconds"], seconds); close(result["ops"], ops)
    close(result["us_per_op"], seconds * 1e6 / ops)
    close(result["gflops"], flops * ops / seconds / 1e9)
    close(result["effective_GBps"], traffic * ops / seconds / 1e9)
    close(result["active_seconds"], sum(w["active_seconds"] for w in windows))
    close(result["active_gflops"], flops * ops / positive(result["active_seconds"]) / 1e9)
    # JSON emission is outside the windows; CLOCK_BOOTTIME also includes any
    # device suspend that steady_clock excludes. Keep the discrepancy below
    # 1% (+50 ms) and publish the alternative elapsed-time rate explicitly.
    elapsed = windows[-1]["boot_s"] - start["boot_s"]
    if elapsed < row["requested_seconds"] or seconds < row["requested_seconds"] - 0.05:
        raise ValueError("Short benchmark duration")
    if not -0.001 <= elapsed - seconds <= 0.05 + seconds * 0.01:
        raise ValueError("Excessive untimed gap")
    if not validations[0]["boot_s"] <= start["boot_s"] < windows[-1]["boot_s"] <= validations[1]["boot_s"] <= result["boot_s"]:
        raise ValueError("Invalid validation/timing ordering")
    return start, result, windows


def summarize(directory, index_range=None):
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
    accepted, rejected = [], []
    telemetry_path = directory / "telemetry.jsonl"
    telemetry = [json.loads(line) for line in telemetry_path.read_text().splitlines()] if telemetry_path.exists() else []
    for row in rows:
        if index_range is not None and not index_range[0] <= int(row["label"].split("-", 1)[0]) < index_range[1]:
            continue
        if not row["valid"]:
            rejected.append({"label": row["label"], "abort": row["abort"], "exit_code": row["exit_code"]})
            continue
        start, result, windows = audit_case(row)
        raw = (directory / (row["label"] + ".log")).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["raw_sha256"]:
            raise ValueError("Raw benchmark log hash mismatch")
        out = {"label": row["label"], "backend": row["backend"], "format": row["format"], "operation": row["operation"],
               "batch": row["batch"], "duty": start["duty"], "requested_seconds": row["requested_seconds"],
               "seconds": result["seconds"], "us_per_op": result["us_per_op"], "active_gflops": result["active_gflops"],
               "gflops": result["gflops"], "effective_GBps": result["effective_GBps"], "raw_sha256": row["raw_sha256"]}
        elapsed = windows[-1]["boot_s"] - start["boot_s"]
        out["boottime_elapsed_seconds"] = elapsed
        out["boottime_gflops"] = start["flops_per_op"] * result["ops"] / elapsed / 1e9
        out["measured_compute_fraction"] = result["active_seconds"] / result["seconds"]
        if "duplicates" in start:
            graphs = result["ops"] / positive(start["duplicates"])
            out["mean_active_graph_ms"] = result["active_seconds"] / graphs * 1000
            out["mean_noncompute_graph_ms"] = (result["seconds"] - result["active_seconds"]) / graphs * 1000
        if row["operation"] == "mat":
            validations = [r for r in row["records"] if r["event"] == "validation"]
            out["max_validation_nmse"] = max(v["nmse"] for v in validations)
            out["max_validation_abs"] = max(v["max_abs"] for v in validations)
        if row["stage"] == "sustained":
            active = [r for r in telemetry if r["label"] == row["label"] and start["boot_s"] + 5 <= r["boot_s"] <= windows[-1]["boot_s"]]
            idle = [r for r in telemetry if r["label"] == "idle-" + row["label"]]
            if len(active) < 10 or len(active) != row["telemetry_samples"] or not idle:
                raise ValueError("Insufficient sustained power samples")
            for sample in active + idle:
                if sample["usb_powered"] or sample["ac_powered"] or sample["wireless_powered"] or sample["battery_status"] != 3 or sample["wakefulness"] not in ("Asleep", "Dozing"):
                    raise ValueError("Power comparison not unplugged/screen-off")
                close(sample["signed_battery_W"], sample["ibat_A"] * sample["vbat_V"])
            close(row["battery_mean_W"], statistics.mean(r["signed_battery_W"] for r in active))
            close(row["idle_mean_W"], statistics.mean(r["signed_battery_W"] for r in idle))
            close(row["battery_increment_W"], row["battery_mean_W"] - row["idle_mean_W"])
            for output, sensor in (("gpu_max_C", "gpu_max_C"), ("npu_max_C", "npu_max_C"), ("skin_max_C", "skin_C")):
                close(row[output], max(r[sensor] for r in active))
            for key in ("battery_mean_W", "idle_mean_W", "battery_increment_W", "telemetry_samples", "gpu_max_C", "npu_max_C", "skin_max_C"):
                out[key] = row[key]
            first, last = windows[:5], windows[-5:]
            def rate(group):
                return sum(w["ops"] for w in group) * start["flops_per_op"] / sum(w["seconds"] for w in group) / 1e9
            out["first5_gflops"], out["last5_gflops"] = rate(first), rate(last)
            out["last_vs_first_percent"] = (rate(last) / rate(first) - 1) * 100
            out["estimated_device_joules_per_op"] = row["battery_mean_W"] * result["us_per_op"] / 1e6
            out["device_gflops_per_W"] = result["gflops"] / positive(row["battery_mean_W"])
        accepted.append(out)
    metadata_path = directory / "metadata.json"
    source_hash = json.loads(metadata_path.read_text())["benchmark_source_sha256"] if metadata_path.exists() else None
    return {"directory": directory.name, "benchmark_source_sha256": source_hash,
            "accepted_count": len(accepted), "rejected": rejected, "cases": accepted, "aggregate": aggregate(accepted)}


def aggregate(cases):
    grouped = {}
    for row in cases:
        key = row["operation"], row["backend"], row["format"], row["batch"], row["duty"]
        grouped.setdefault(key, []).append(row)
    result = []
    for key, group in grouped.items():
        metrics = ["us_per_op", "gflops", "effective_GBps"]
        metrics.extend(metric for metric in ("battery_mean_W", "idle_mean_W", "battery_increment_W", "device_gflops_per_W", "estimated_device_joules_per_op")
                       if all(metric in row for row in group))
        result.append({"operation": key[0], "backend": key[1], "format": key[2], "batch": key[3], "duty": key[4], "replicates": len(group),
                          **{metric + suffix: function(r[metric] for r in group)
                             for metric in metrics
                             for suffix, function in (("_mean", statistics.mean), ("_min", min), ("_max", max))}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--index-range", action="append", help="Optional START:END (exclusive), one per directory; ':' means all")
    args = parser.parse_args()
    ranges = [None] * len(args.directories)
    if args.index_range:
        if len(args.index_range) != len(args.directories):
            parser.error("Provide one --index-range per directory")
        ranges = [tuple(int(part) if part else default for part, default in zip(value.split(":"), (0, 1000000))) for value in args.index_range]
        if any(len(bounds) != 2 or bounds[0] >= bounds[1] for bounds in ranges):
            parser.error("Invalid index range")
    datasets = [summarize(path, bounds) for path, bounds in zip(args.directories, ranges)]
    result = {"schema_version": 1, "bandwidth_unit": "decimal GB/s", "power_scope": "whole-device battery estimate",
              "datasets": datasets, "aggregate": aggregate([row for dataset in datasets for row in dataset["cases"]])}
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
