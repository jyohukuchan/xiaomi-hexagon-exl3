"""Audit fixed-token real-model baseline timing and estimate whole-device energy."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def positive(value):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("Expected finite positive value")
    return value


def near(a, b):
    if not math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-8):
        raise ValueError("Timing arithmetic mismatch")


def audit_run(run, samples):
    if run["exit_code"] or run["abort"] or run["parse_errors"] or not run["teacher_forced"]:
        raise ValueError("Expected clean fixed-token baseline")
    rows = run["records"]
    init = [r for r in rows if r["event"] == "init"]
    loaded = [r for r in rows if r["event"] == "loaded"]
    finished = [r for r in rows if r["event"] == "finished"]
    if len(init) != 1 or len(loaded) != 1 or len(finished) != 1 or finished[0]["stopped"] or not init[0]["teacher_forced"]:
        raise ValueError("Incomplete process lifecycle")
    if loaded[0]["backend"] != "HTP0" or loaded[0]["kv"] != "f16":
        raise ValueError("Unexpected backend/cache precision")
    trials = []
    for i in range(run["repeats"]):
        current = {name: [r for r in rows if r.get("trial") == i and r["event"] == name]
                   for name in ("idle_start", "prefill_start", "prefill_end", "decode_start", "decode_end")}
        if any(len(group) != 1 for group in current.values()):
            raise ValueError("Missing/duplicate phase events")
        idle, ps, pe, ds, de = [current[name][0] for name in current]
        if not idle["boot_s"] < ps["boot_s"] <= pe["start_boot_s"] < pe["end_boot_s"] <= pe["boot_s"] <= ds["boot_s"] <= de["start_boot_s"] < de["end_boot_s"] <= de["boot_s"] <= finished[0]["boot_s"]:
            raise ValueError("Invalid phase ordering")
        if ps["boot_s"] - idle["boot_s"] < 14.9:
            raise ValueError("Idle baseline too short")
        for phase in (pe, de):
            positive(phase["seconds"])
            near(phase["seconds"], phase["end_boot_s"] - phase["start_boot_s"])
        if pe["tokens"] != loaded[0]["prompt_tokens"]:
            raise ValueError("Prompt count mismatch")
        if de["stopped"] or not de["teacher_forced"] or de["decode_calls"] != run["requested_steps"]:
            raise ValueError("Incomplete fixed-token decode")
        if len(de["supplied_ids"]) != de["decode_calls"] or len(de["generated_ids"]) != de["decode_calls"] + 1:
            raise ValueError("Token sequence length mismatch")
        near(de["tokens_per_second"], de["decode_calls"] / de["seconds"])
        source = [s for s in samples if s["label"].startswith(run["format"] + "/")]
        active = [s for s in source if de["start_boot_s"] <= s["boot_s"] <= de["end_boot_s"]]
        idle_samples = [s for s in source if idle["boot_s"] <= s["boot_s"] <= ps["boot_s"]]
        if len(active) < 4 or len(idle_samples) < 20:
            raise ValueError("Insufficient power/idle samples")
        for s in active + idle_samples:
            if s["usb_powered"] or s["ac_powered"] or s["wireless_powered"] or s["battery_status"] != 3 or s["wakefulness"] not in ("Asleep", "Dozing"):
                raise ValueError("Power window not unplugged/screen-off")
            near(s["signed_battery_W"], s["ibat_A"] * s["vbat_V"])
        watts = positive(statistics.mean(s["signed_battery_W"] for s in active))
        baseline = statistics.mean(s["signed_battery_W"] for s in idle_samples)
        trials.append({"trial": i, "prefill_tokens": pe["tokens"], "prefill_seconds": pe["seconds"],
                       "decode_steps": de["decode_calls"], "decode_seconds": de["seconds"],
                       "steps_per_second": de["tokens_per_second"], "battery_mean_W": watts,
                       "idle_mean_W": baseline, "battery_increment_W": watts - baseline,
                       "estimated_device_joules_per_step": watts * de["seconds"] / de["decode_calls"],
                       "power_samples": len(active), "npu_max_C": max(s["npu_max_C"] for s in active),
                       "cpu_max_C": max(s["cpu_max_C"] for s in active), "skin_max_C": max(s["skin_C"] for s in active),
                       "supplied_ids": de["supplied_ids"], "predicted_ids": de["generated_ids"]})
    if len({tuple(t["supplied_ids"]) for t in trials}) != 1 or len({tuple(t["predicted_ids"]) for t in trials}) != 1:
        raise ValueError("Repeated trial tokens differ")
    aggregate = {"pooled_steps_per_second": sum(t["decode_steps"] for t in trials) / sum(t["decode_seconds"] for t in trials),
                 "pooled_estimated_device_joules_per_step": sum(t["battery_mean_W"] * t["decode_seconds"] for t in trials) / sum(t["decode_steps"] for t in trials)}
    for key in ("steps_per_second", "battery_mean_W", "idle_mean_W", "estimated_device_joules_per_step"):
        for suffix, fn in (("mean", statistics.mean), ("min", min), ("max", max)):
            aggregate[key + "_" + suffix] = fn(t[key] for t in trials)
    return {"format": run["format"], "model_sha256": run["model_sha256"], "raw_sha256": run["raw_sha256"],
            "prompt_sha256": run["prompt_sha256"], "source_sha256": run["source_sha256"], "trials": trials, "aggregate": aggregate}


def summarize(directory):
    samples = [json.loads(line) for line in (directory / "telemetry.jsonl").read_text().splitlines()]
    runs = [json.loads(line) for line in (directory / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    result = []
    for run in runs:
        if hashlib.sha256((directory / (run["format"] + ".log")).read_bytes()).hexdigest() != run["raw_sha256"]:
            raise ValueError("Raw log hash mismatch")
        result.append(audit_run(run, samples))
    if {r["format"] for r in result} != {"Q4_K_M", "Q8_0"} or len(result) != 2:
        raise ValueError("Missing or duplicate model baseline")
    if len({tuple(r["trials"][0]["supplied_ids"]) for r in result}) != 1:
        raise ValueError("Models received different fixed decode inputs")
    return {"dataset": directory.name, "mode": "fixed-token teacher-forced decode, not free-running translation",
            "power_scope": "whole-device battery estimate, short active windows; not accelerator rail", "models": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(summarize(args.directory), indent=2)
    if args.output: args.output.write_text(text + "\n", encoding="utf-8")
    else: print(text)


if __name__ == "__main__":
    main()
