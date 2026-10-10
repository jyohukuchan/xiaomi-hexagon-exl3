"""Monitor isolated HexKL integer API trials; retain failures rather than retrying them away."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import shlex
import re
import subprocess
import threading
import time

import run_accelerator_bench as monitor

ROOT = "/data/local/tmp/xiaomi-hexagon-exl3/int-hmx-probe"
BINARY = ROOT + "/int_hmx_macro_probe"
REAL_TENSOR = None


def run_case(serial, directory, sink, bits, batch, n, k, repeats, pad):
    label = f"i{bits}-b{batch}-n{n}-k{k}" + (("-" + pad) if isinstance(pad, str) else "-pad64" if pad else "")
    monitor.cooldown(serial, sink, "idle-" + label, minimum=5)
    args = [BINARY, str(bits), str(batch), str(n), str(k), str(repeats)]
    if pad: args.append(pad if isinstance(pad, str) else "pad64")
    model_hash = None
    if REAL_TENSOR:
        from run_homura_kernel_probe import FILES, ROOT as MODEL_ROOT
        name, model_hash = FILES["Q8_0" if bits == 8 else "Q4_K_M"]
        args += [MODEL_ROOT + "/models/" + name, REAL_TENSOR]
    dsp_path = "/data/local/tmp/llama.cpp/lib" if pad == "ggml" else ROOT
    command = f"cd {ROOT} && export LD_LIBRARY_PATH={ROOT}:/data/local/tmp/llama.cpp/lib:/vendor/lib64 ADSP_LIBRARY_PATH={dsp_path} GGML_HEXAGON_PROFILE=0 GGML_HEXAGON_VERBOSE=0 GGML_SCHED_DEBUG=0; exec " + shlex.join(args)
    process = subprocess.Popen([monitor.ADB, "-s", serial, "shell", command], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
    path = directory / (label + ".log")
    records, errors = [], []
    state = {}
    def reader():
        with path.open("w", encoding="utf-8") as log:
            for line in process.stdout:
                log.write(line); log.flush()
                if line.startswith("INTEGER "):
                    try:
                        row = json.loads(line[8:]); records.append(row)
                        if row["event"] == "init": state["pid"] = row["pid"]
                    except Exception as error: errors.append(str(error))
    thread = threading.Thread(target=reader, daemon=True); thread.start()
    began, abort, stopped_at = time.monotonic(), None, None
    print("START", label, flush=True)
    while process.poll() is None:
        before = time.monotonic()
        try:
            row = monitor.sample(serial, sink, label)
            reason = monitor.stop_reason(row)
        except Exception as error:
            reason = "Telemetry failure: " + str(error)
        if time.monotonic() - began > 90: reason = "Integer probe wall timeout"
        if reason and not abort and process.poll() is None:
            pid = state.get("pid")
            if not pid: raise RuntimeError("Probe PID unknown; remote state unknown")
            try: exe = monitor.shell(serial, f"readlink /proc/{pid}/exe").strip()
            except RuntimeError:
                if process.poll() is not None: break
                raise
            if exe != BINARY: raise RuntimeError("Probe PID identity mismatch; refusing signal")
            abort, stopped_at = reason, time.monotonic()
            monitor.shell(serial, f"kill -INT {pid}")
        if abort and time.monotonic() - stopped_at > 15: raise RuntimeError("Probe did not stop; remote state unknown")
        time.sleep(max(0, 0.5 - (time.monotonic() - before)))
    thread.join(timeout=5)
    if thread.is_alive(): raise RuntimeError("Log reader did not finish")
    result = {"label": label, "bits": bits, "batch": batch, "n": n, "k": k, "repeats": repeats,
              "mode": pad if isinstance(pad, str) else "raw", "pad64": pad is True or pad in ("raw", "scaled"), "tensor": REAL_TENSOR, "model_sha256": model_hash,
              "exit_code": process.returncode, "abort": abort, "parse_errors": errors, "records": records,
              "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    with (directory / "runs.jsonl").open("a", encoding="utf-8") as output: output.write(json.dumps(result) + "\n")
    print("FINISH", label, "exit", process.returncode, "abort", abort, "directory", directory, flush=True)
    if process.returncode or abort or errors: raise RuntimeError("Integer probe failed; see retained raw evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--bits", nargs="+", type=int, choices=(4, 8), default=[8, 4])
    parser.add_argument("--batches", nargs="+", type=int, default=[64])
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--pad64", action="store_true")
    parser.add_argument("--engine", choices=("macro", "micro", "ggml"), default="macro")
    parser.add_argument("--scaled", action="store_true", help="Micro API: apply distinct K32 weight/activation scales and affine offsets")
    parser.add_argument("--gguf-tensor", help="Use an actual tensor from the pinned Homura Q4_K_M/Q8_0 models")
    args = parser.parse_args()
    if len(set(args.bits)) != len(args.bits) or len(set(args.batches)) != len(args.batches):
        parser.error("Duplicate cases would overwrite raw logs; choose unique bits/batches")
    global ROOT, BINARY, REAL_TENSOR
    REAL_TENSOR = args.gguf_tensor
    if args.engine in ("micro", "ggml"):
        ROOT = "/data/local/tmp/xiaomi-hexagon-exl3/int-hmx-micro-probe"
        BINARY = ROOT + "/int_hmx_micro_probe"
    elif args.scaled:
        parser.error("--scaled requires --engine micro")
    if args.engine == "ggml" and not REAL_TENSOR: parser.error("GGML comparison requires --gguf-tensor")
    if REAL_TENSOR:
        if args.engine == "macro" or not re.fullmatch(r"blk\.\d+\.[A-Za-z0-9_]+\.weight", REAL_TENSOR): parser.error("Invalid tensor/engine")
        if args.engine == "micro" and not args.scaled: parser.error("Real tensors require --scaled")
        from run_homura_kernel_probe import FILES, ROOT as MODEL_ROOT
        for bits in args.bits:
            name, expected = FILES["Q8_0" if bits == 8 else "Q4_K_M"]
            if monitor.shell(args.serial, f"sha256sum {MODEL_ROOT}/models/{name}", timeout=30).split()[0] != expected:
                raise RuntimeError("Model artifact hash mismatch")
    active = monitor.shell(args.serial, "pidof int_hmx_macro_probe int_hmx_micro_probe homura_baseline llama-server llama-completion accelerator_bench gpu_npu_probe test-backend-ops || true").strip()
    if active: raise RuntimeError("Concurrent project processes: " + active)
    directory = Path(__file__).resolve().parents[1] / "benchmark-raw" / (
        "int-hmx-" + args.engine + "-" + datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d-%H%M%S"))
    directory.mkdir(exist_ok=False)
    project = Path(__file__).resolve().parents[1]
    files = ["tools/run_int_hmx_macro_probe.py", "tools/int_hmx_macro_probe.cpp", "tools/int_hmx_micro_probe.cpp", "src/int-hmx/int_hmx_service.c", "src/int-hmx/int_hmx_iface.idl", "src/int-hmx/CMakeLists.txt"]
    provenance = {"arguments": vars(args), "sources": {p: hashlib.sha256((project / p).read_bytes()).hexdigest() for p in files},
                  "binary_sha256": monitor.shell(args.serial, f"sha256sum {BINARY}").split()[0]}
    library_name = "libint_hmx_htp.so" if args.engine == "micro" else "libggml-htp-v75.so" if args.engine == "ggml" else "libhexkl_skel.so"
    library_root = "/data/local/tmp/llama.cpp/lib" if args.engine == "ggml" else ROOT
    provenance["dsp_library_sha256"] = monitor.shell(args.serial, f"sha256sum {library_root}/{library_name}").split()[0]
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    with (directory / "telemetry.jsonl").open("w", encoding="utf-8") as sink:
        for bits in args.bits:
            for batch in args.batches: run_case(args.serial, directory, sink, bits, batch, args.n, args.k, args.repeats,
                                               "ggml" if args.engine == "ggml" else ("scaled" if args.scaled else "raw") if args.engine == "micro" else args.pad64)
        monitor.cooldown(args.serial, sink, "final-idle", minimum=5)
    print("PASS", directory, flush=True)


if __name__ == "__main__": main()
