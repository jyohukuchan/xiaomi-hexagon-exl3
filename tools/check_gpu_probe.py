"""Fail-closed audit of the standalone Android GPU/Hexagon probe logs."""
import argparse
import hashlib
import json
import math
from pathlib import Path


SIZES = (128, 4096, 32768, 262144, 1048576, 8388608, 33554432)
PMU_EVENTS = (
    (0x3F, 0x40, 0x42, 0x46, 0x47, 0x48, 0x49, 0x118),
    (0x118, 0x81, 0x8A, 0x123, 0x124, 0x111, 0x7C, 0x7D),
    (0x40,) * 8,
    (0x118,) * 8,
)


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


def audit_pmu(lines):
    schemas = [fields(line) for line in lines if line.startswith("pmu_schema ")]
    if len(schemas) != 1:
        raise ValueError("Missing/duplicate PMU schema")
    schema = schemas[0]
    event_set = int(schema["set"])
    if schema["version"] != "2" or schema["report_bytes"] != "544" or not 0 <= event_set < len(PMU_EVENTS):
        raise ValueError("Invalid PMU ABI/event set")
    if tuple(int(schema[f"e{i}"]) for i in range(8)) != PMU_EVENTS[event_set]:
        raise ValueError("Wrong raw V75 event IDs")
    phases, sources, modes = ("arrival", "cold", "warm", "outer"), ("cpu", "gpu"), ("empty", "hvx", "scalar", "hvx_noinv")
    samples, summaries = {}, {}
    for line in lines:
        if not line.startswith(("pmu_sample ", "pmu_summary ")):
            continue
        row = fields(line)
        key = int(row["bytes"]), row["src"], row["mode"], row["phase"]
        if int(row["set"]) != event_set or key[0] not in SIZES or key[1] not in sources or key[2] not in modes or key[3] not in phases:
            raise ValueError("Invalid PMU case")
        if row["mismatches"] != "0":
            raise ValueError("Invalid PMU payload")
        for i in range(8):
            if number(row, f"c{i}", positive=False) > 0xFFFFFFFF:
                raise ValueError("Counter exceeds uint32 delta")
        if line.startswith("pmu_sample "):
            full_key = key + (int(row["index"]),)
            if full_key in samples:
                raise ValueError("Duplicate PMU sample")
            number(row, "usec", positive=False)  # Empty windows may round to zero timer ticks.
            number(row, "cycles")
            samples[full_key] = row
        else:
            if key in summaries or int(row["samples"]) < 1:
                raise ValueError("Duplicate/empty PMU summary")
            if number(row, "p95_us", positive=False) < number(row, "p50_us", positive=False):
                raise ValueError("Invalid PMU percentiles")
            summaries[key] = row
    required = {(size, source, mode, phase) for size in SIZES for source in sources for mode in modes for phase in phases}
    if set(summaries) != required:
        raise ValueError("Incomplete CPU/GPU/empty/HVX/scalar/cold/warm sweep")
    expected_samples = {key + (i,) for key in required for i in range(int(summaries[key]["samples"]))}
    if set(samples) != expected_samples:
        raise ValueError("Incomplete PMU sample sweep")
    # Recompute every reported median/p95 from the printed raw samples.
    for key, row in summaries.items():
        cases = [samples[key + (i,)] for i in range(int(row["samples"]))]
        for name in ("usec",) + tuple(f"c{i}" for i in range(8)):
            ordered = sorted(float(case[name]) for case in cases)
            for fraction, field in ((0.5, "p50_us"), (0.95, "p95_us")) if name == "usec" else ((0.5, name),):
                expected = ordered[int(fraction * (len(ordered) - 1))]
                if not math.isclose(float(row[field]), expected, rel_tol=2e-5, abs_tol=1e-5):
                    raise ValueError("PMU summary does not match raw samples")
    calibration = []
    # Validate identity/scaling of the read events; zero slave/secondary values
    # have NO positive control and are deliberately not declared calibrated.
    for source in sources:
        if event_set in (0, 1, 3):
            slot = 7 if event_set == 0 else 0
            for size in (32768, 1048576):
                read = summaries[(size, source, "hvx", "cold")]
                empty = summaries[(size, source, "empty", "cold")]
                value = float(read[f"c{slot}"]) - float(empty[f"c{slot}"])
                expected = size / 128
                if not math.isclose(value, expected, rel_tol=0.02, abs_tol=8):
                    raise ValueError("HVX load event failed calibration")
                calibration.append({"source": source, "bytes": size, "event": 0x118, "observed_minus_empty": value, "expected": expected})
        if event_set in (0, 2):
            slot = 1 if event_set == 0 else 0
            cold = float(summaries[(32768, source, "hvx", "cold")][f"c{slot}"])
            warm = float(summaries[(32768, source, "hvx", "warm")][f"c{slot}"])
            if cold < 200 or warm > cold * 0.1 + 8:
                raise ValueError("AXI cold/warm event failed calibration")
            calibration.append({"source": source, "event": 0x40, "cold_32KiB": cold, "warm_32KiB": warm})
        if event_set in (2, 3):
            row = summaries[(1048576, source, "hvx", "cold")]
            values = [float(row[f"c{i}"]) for i in range(8)]
            if max(values) - min(values) > max(32, max(values) * 0.02):
                raise ValueError("Duplicated PMU event/counter bank mismatch")
    rejected = [fields(line) for line in lines if line.startswith("pmu_rejected ")]
    for row in rejected:
        if (int(row["set"]) != event_set or not 1 <= int(row["flags"]) <= 31 or
                int(row["attempt"]) not in (0, 1, 2) or row["src"] not in sources or
                row["mode"] not in modes or int(row["bytes"]) not in SIZES):
            raise ValueError("Invalid rejected PMU window")
    return {"mode": "pmu", "event_set": event_set, "events": list(PMU_EVENTS[event_set]), "rejected_count": len(rejected),
            "sample_count": len(samples), "calibration": calibration, "summaries": list(summaries.values())}


def audit(text, require_fixtures=False):
    lines = text.splitlines()
    passes = [line for line in lines if line.startswith("PASS mode=")]
    if len(passes) != 1 or not lines or lines[-1] != passes[0] or "FAIL:" in text:
        raise ValueError("Missing terminal success or explicit failure")
    mode = passes[0].split("=", 1)[1]
    if mode not in ("interop", "interop-finish", "dequant", "pmu"):
        raise ValueError("Unknown mode")
    if mode == "pmu":
        return audit_pmu(lines)
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
