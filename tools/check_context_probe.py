"""Independently check context-probe artifacts and optional Index-Translate NPU placement."""
import argparse
from array import array
from collections import Counter
import json
import math
from pathlib import Path
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_report(report, logits):
    require(type(report) is dict, "Report must be an object")
    require(report.get("passed") is True, "Probe did not pass")
    fields = ("depth", "decode_steps", "context_capacity", "vocabulary", "batch", "checked_logit_rows",
              "checked_logit_values", "hybrid_position_min", "hybrid_position_max")
    require(all(type(report.get(name)) is int for name in fields), "Invalid integer metadata")
    depth, steps, vocabulary = report["depth"], report["decode_steps"], report["vocabulary"]
    require(0 <= depth <= 65536 and 1 <= steps <= 1024 and 2 <= vocabulary <= 1000000, "Invalid geometry")
    require(report["batch"] > 0 and report["context_capacity"] >= depth + steps, "Insufficient capacity")
    rows = (depth + report["batch"] - 1) // report["batch"] + steps
    require(report["checked_logit_rows"] == rows and report["checked_logit_values"] == rows * vocabulary, "Checkpoint count mismatch")
    require(report["hybrid_position_max"] == depth + steps - 1, "Unexpected final position")
    require(0 <= report["hybrid_position_min"] <= report["hybrid_position_max"], "Invalid hybrid position range")
    require(report.get("kv_requested") == "f16", "KV is not requested as FP16")
    tokens = report.get("decoded_token_ids")
    require(isinstance(tokens, list) and len(tokens) == steps and all(type(t) is int and 0 <= t < vocabulary for t in tokens), "Invalid decoded IDs")
    require(len(logits) == vocabulary and all(math.isfinite(value) for value in logits), "Invalid or non-finite final logits")
    for name in ("prefill_ms_including_checks", "decode_ms_including_checks", "diagnostic_decode_tokens_per_second", "logit_min", "logit_max"):
        require(type(report.get(name)) in (int, float) and math.isfinite(report[name]), f"Invalid {name}")
    require(report["prefill_ms_including_checks"] >= 0 and report["decode_ms_including_checks"] > 0, "Invalid timing")
    expected_rate = steps * 1000 / report["decode_ms_including_checks"]
    require(math.isclose(report["diagnostic_decode_tokens_per_second"], expected_rate, rel_tol=2e-5), "Rate mismatch")
    tolerance = 2e-5 * max(1, abs(report["logit_min"]), abs(report["logit_max"]))
    require(report["logit_min"] <= min(logits) + tolerance and report["logit_max"] >= max(logits) - tolerance, "Logit range mismatch")
    return {"passed": True, "depth": depth, "decode_steps": steps, "independently_finite_final_values": len(logits),
            "checkpoint_values_checked_by_probe": report["checked_logit_values"]}


def operators(text):
    weights, kinds = [], Counter()
    for line in text.splitlines():
        if "ggml-hex: HTP0 profile-op " not in line:
            continue
        body = line.split("profile-op ", 1)[1]
        kind = body.split("|", 1)[0]
        kinds[kind.split("+", 1)[0]] += 1
        if "|exl3-hmx " in body:
            weights.append(body.split("|")[1].split(" x ")[0])
    return weights, kinds


def audit_placement(log, reference, depth, steps):
    expected_weights, expected_kinds = operators(reference)
    require(len(expected_weights) == len(set(expected_weights)) == 151, "Invalid reference matrix profile")
    require(expected_kinds["FLASH_ATTN_EXT"] == 6 and expected_kinds["GATED_DELTA_NET"] == 18, "Invalid reference attention profile")
    require(any(line.startswith("llama_kv_cache:") and "K (f16):" in line and "V (f16):" in line for line in log.splitlines()), "Missing actual FP16 cache allocation record")
    require(depth > 0, "Placement audit needs prefill")
    marker = f"PREFILL checked={depth}/{depth} position={depth - 1}\n"
    require(log.count(marker) == 1, "Missing or duplicate final prefill marker")
    tail = log.split(marker, 1)[1]
    for step in range(1, steps + 1):
        marker = f"DECODE checked={step}/{steps} position={depth + step - 1}\n"
        require(tail.count(marker) == 1, "Missing or duplicate decode marker")
        section, tail = tail.split(marker, 1)
        weights, kinds = operators(section)
        require(len(weights) == 151 and set(weights) == set(expected_weights), f"Matrix placement mismatch at step {step}")
        require(kinds["FLASH_ATTN_EXT"] == 6 and kinds["GATED_DELTA_NET"] == 18, f"Attention placement mismatch at step {step}")
    return {"audited_decode_steps": steps, "exl3_matrices_per_step": 151, "full_attention_per_step": 6, "gdn_per_step": 18}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--profile-log", type=Path)
    parser.add_argument("--reference-profile", type=Path)
    args = parser.parse_args()
    if bool(args.profile_log) != bool(args.reference_profile):
        parser.error("Provide both profile paths")
    try:
        report = json.loads((args.directory / "report.json").read_text(encoding="utf-8"))
        require(type(report) is dict and type(report.get("vocabulary")) is int and 2 <= report["vocabulary"] <= 1000000, "Invalid vocabulary metadata")
        binary_path = args.directory / "last_logits.bin"
        require(binary_path.stat().st_size == report["vocabulary"] * 4, "Invalid logit artifact size")
        logits = array("f")
        require(logits.itemsize == 4, "Unsupported native float size")
        logits.frombytes(binary_path.read_bytes())
        if sys.byteorder != "little":
            logits.byteswap()
        result = validate_report(report, logits)
        if args.profile_log:
            result["placement"] = audit_placement(args.profile_log.read_text(encoding="utf-8"), args.reference_profile.read_text(encoding="utf-8"), report["depth"], report["decode_steps"])
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"CONTEXT_CHECK_FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
