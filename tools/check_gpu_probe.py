"""Fail-closed audit of the standalone Android GPU/Hexagon probe logs."""
import argparse
import hashlib
import json
import math
from pathlib import Path


SIZES = (128, 4096, 32768, 262144, 1048576, 8388608, 33554432)


def fields(line):
    result = {}
    for token in line.split()[1:]:
        key, value = token.split("=", 1)
        if key in result:
            raise ValueError(f"Duplicate field {key}")
        result[key] = value
    return result


def number(row, key, positive=True):
    value = float(row[key])
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"Invalid {key}")
    return value


def audit(text, require_fixtures=False):
    lines = text.splitlines()
    passes = [line for line in lines if line.startswith("PASS mode=")]
    if len(passes) != 1 or not lines or lines[-1] != passes[0] or "FAIL:" in text:
        raise ValueError("Missing terminal success or explicit failure")
    mode = passes[0].split("=", 1)[1]
    if mode not in ("interop", "interop-finish", "dequant"):
        raise ValueError("Unknown mode")
    prefixes = ("interop_sample ", "interop_summary ", "codebook_test ", "dequant_summary ")
    for line in lines:
        if line.startswith(prefixes):
            row = fields(line)
            if row.get("mismatches") != "0":
                raise ValueError("Nonzero/missing mismatch count")
    if mode.startswith("interop"):
        samples, summaries = {}, {}
        for line in lines:
            if not line.startswith(("interop_sample ", "interop_summary ")):
                continue
            row = fields(line)
            size = int(row["bytes"])
            if int(row["map"]) != int(mode == "interop") or size not in SIZES:
                raise ValueError("Unexpected mapping/size")
            if line.startswith("interop_sample "):
                key = size, int(row["index"])
                if key in samples:
                    raise ValueError("Duplicate sample")
                for name in ("total_us", "gpu_us", "rpc_us", "hvx_copy_us", "noop_rpc_us", "read_rpc_us", "hvx_read_us"):
                    number(row, name)
                number(row, "sync_us", positive=False)
                samples[key] = row
            else:
                if size in summaries:
                    raise ValueError("Duplicate summary")
                for name in ("total_p50_us", "total_p95_us", "gpu_p50_us", "rpc_p50_us", "hvx_copy_p50_us",
                             "noop_rpc_p50_us", "payload_GBps", "read_rpc_p50_us", "hvx_read_p50_us",
                             "read_effective_GBps", "hvx_read_GBps"):
                    number(row, name)
                if number(row, "total_p95_us") < number(row, "total_p50_us"):
                    raise ValueError("Invalid percentiles")
                if not math.isclose(number(row, "payload_GBps"), size / number(row, "total_p50_us") / 1000, rel_tol=2e-5):
                    raise ValueError("Bandwidth/unit mismatch")
                summaries[size] = row
        if set(summaries) != set(SIZES):
            raise ValueError("Incomplete size sweep")
        expected = {(size, i) for size in SIZES for i in range(int(summaries[size]["samples"]))}
        if not expected or any(int(row["samples"]) < 1 for row in summaries.values()) or set(samples) != expected:
            raise ValueError("Incomplete sample sweep")
        return {"mode": mode, "summaries": [summaries[size] for size in SIZES], "sample_count": len(samples)}
    codebooks, cases = {}, {}
    for line in lines:
        if line.startswith("codebook_test "):
            row = fields(line)
            cb = int(row["cb"])
            if cb in codebooks or row["states"] != "65536":
                raise ValueError("Invalid exhaustive codebook test")
            codebooks[cb] = row
        elif line.startswith("dequant_summary "):
            row = fields(line)
            key = (row["label"], int(row["k"]), int(row["n"]), int(row["bits"]), int(row["cb"]), int(row["layout"]))
            if key in cases or int(row["samples"]) < 1:
                raise ValueError("Duplicate/empty dequant case")
            for name in ("gpu_p50_us", "gpu_p95_us", "gpu_to_dsp_p50_us", "rpc_p50_us", "output_GBps"):
                number(row, name)
            if number(row, "gpu_p95_us") < number(row, "gpu_p50_us"):
                raise ValueError("Invalid dequant percentiles")
            cases[key] = row
    if set(codebooks) != {0, 1, 2}:
        raise ValueError("Incomplete codebook coverage")
    required = {("synthetic", 256, 384, bits, cb, layout)
                for bits in range(1, 9) for cb in range(3) for layout in (0, 1)}
    required |= {("synthetic", 256, 384, bits, 2, 2) for bits in (4, 6, 8)}
    required |= {("single_group", 128, 128, bits, 2, layout) for bits in (4, 6, 8) for layout in (0, 1, 2)}
    required |= {("streaming_chunk", 2048, 6144, bits, 2, layout) for bits in (4, 6) for layout in (0, 1, 2)}
    if require_fixtures:
        required |= {("cuda_fixture", 2048, n, bits, 2, layout)
                     for n, bits in ((512, 4), (128, 6)) for layout in (0, 1, 2)}
    if not required <= cases.keys():
        raise ValueError("Incomplete dequant coverage")
    return {"mode": mode, "codebook_states": 3 * 65536, "case_count": len(cases), "summaries": list(cases.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--require-fixtures", action="store_true")
    args = parser.parse_args()
    raw = args.log.read_bytes()
    result = audit(raw.decode("utf-8-sig"), args.require_fixtures)
    result["sha256"] = hashlib.sha256(raw).hexdigest()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
