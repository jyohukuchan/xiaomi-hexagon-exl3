"""Matched existing GGUF HMX-FP16 vs forced-HVX matrix routes, not integer-HMX kernels."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import re

import run_accelerator_bench as monitor
import run_int_hmx_macro_probe as probe
from run_homura_kernel_probe import FILES, ROOT as MODEL_ROOT
from analyze_int_hmx_probe import audit


def cases(bits, batches, trials):
    if not bits or len(set(bits)) != len(bits) or not set(bits) <= {4, 8}:
        raise ValueError("Invalid/duplicate precisions")
    if not batches or len(set(batches)) != len(batches) or any(not 1 <= b <= 512 for b in batches):
        raise ValueError("Invalid/duplicate batches")
    if not 1 <= trials <= 4: raise ValueError("Trials must be 1..4")
    output = []
    for b in bits:
        for m in batches:
            for t in range(trials):
                for route in (("hmx", "hvx") if t % 2 == 0 else ("hvx", "hmx")):
                    output.append((b, m, t, route))
    return output


def profile_paths(log):
    paths = []
    for line in log.splitlines():
        if "profile-op MUL_MAT|" not in line: continue
        fields = line.split("profile-op ", 1)[1].split("|")
        if len(fields) != 7 or fields[1] != "route_weights x route_activations -> route_output":
            raise ValueError("Unexpected matrix profile fields")
        route = fields[5].split()[0]
        if route not in ("hmx-tiled", "hvx-tiled"): raise ValueError("Unknown matrix route")
        paths.append(route)
    if not paths: raise ValueError("No matrix kernel profile found")
    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--serial", required=True)
    p.add_argument("--bits", nargs="+", type=int, default=[8, 4])
    p.add_argument("--batches", nargs="+", type=int, default=[1, 4, 16, 64, 128, 512])
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--calls", type=int, default=50)
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--k", type=int, default=2048)
    p.add_argument("--tensor", default="blk.0.attn_gate.weight")
    p.add_argument("--phase", choices=("profiles", "timings"), required=True)
    args = p.parse_args()
    try: work = cases(args.bits, args.batches, args.trials)
    except ValueError as e: p.error(str(e))
    if not 1 <= args.calls <= 100 or not re.fullmatch(r"blk\.\d+\.[A-Za-z0-9_]+\.weight", args.tensor): p.error("Invalid calls/tensor")
    active = monitor.shell(args.serial, "pidof int_hmx_macro_probe int_hmx_micro_probe homura_baseline llama-server llama-completion accelerator_bench gpu_npu_probe test-backend-ops || true").strip()
    if active: raise RuntimeError("Concurrent project process: " + active)
    probe.ROOT = "/data/local/tmp/xiaomi-hexagon-exl3/int-hmx-micro-probe"
    probe.BINARY = probe.ROOT + "/int_hmx_micro_probe"
    probe.REAL_TENSOR = args.tensor
    for b in args.bits:
        name, expected = FILES["Q8_0" if b == 8 else "Q4_K_M"]
        if monitor.shell(args.serial, f"sha256sum {MODEL_ROOT}/models/{name}", timeout=30).split()[0] != expected:
            raise RuntimeError("Model hash mismatch")
    project = Path(__file__).resolve().parents[1]
    directory = project / "benchmark-raw" / ("hmx-hvx-" + args.phase + "-" + datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d-%H%M%S"))
    directory.mkdir(exist_ok=False)
    sources = ["tools/int_hmx_micro_probe.cpp", "tools/run_hmx_hvx_compare.py", "tools/run_int_hmx_macro_probe.py", "tools/analyze_int_hmx_probe.py"]
    provenance = {"arguments": vars(args), "sources": {s: hashlib.sha256((project / s).read_bytes()).hexdigest() for s in sources},
                  "binary_sha256": monitor.shell(args.serial, f"sha256sum {probe.BINARY}").split()[0],
                  "backend_libraries": monitor.shell(args.serial, "sha256sum /data/local/tmp/llama.cpp/lib/libggml-hexagon.so /data/local/tmp/llama.cpp/lib/libggml-htp-v75.so"),
                  "baseline_commit": "a11f57ba93797579a5d1855ee216a31f10242676",
                  "scope": "GGUF graph pipeline: HMX FP16 vs HVX quantized; not native integer HMX throughput"}
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    with (directory / "telemetry.jsonl").open("w", encoding="utf-8") as sink:
        for b, m, trial, route in work:
            profiled = args.phase == "profiles"
            run = probe.run_case(args.serial, directory, sink, b, m, args.n, args.k,
                                 1 if profiled else args.calls, "ggml-" + route, f"-trial{trial}", profiled)
            audit(run, allow_profile=profiled)
            if profiled:
                paths = profile_paths((directory / (run["label"] + ".log")).read_text(encoding="utf-8"))
                if set(paths) != {route + "-tiled"}: raise RuntimeError("Unexpected matrix arithmetic route: " + str(paths))
                print("VERIFIED", route, "batch", m, "profile_records", len(paths), flush=True)
        monitor.cooldown(args.serial, sink, "final-idle", minimum=5)
    print("PASS", directory, flush=True)


if __name__ == "__main__": main()
