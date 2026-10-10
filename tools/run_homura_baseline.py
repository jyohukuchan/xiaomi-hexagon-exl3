"""Collect non-profiled real-model timing windows and unplugged battery telemetry."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import threading
import time

import run_accelerator_bench as monitor
from run_homura_kernel_probe import FILES, ROOT as MODEL_ROOT

BINARY = "/data/local/tmp/xiaomi-hexagon-exl3/homura-baseline/homura_baseline"


def run_one(serial, directory, sink, fmt, steps, repeats, forced):
    filename, expected_hash = FILES[fmt]
    actual = monitor.shell(serial, f"sha256sum {MODEL_ROOT}/models/{filename}", timeout=30).split()[0]
    if actual != expected_hash:
        raise RuntimeError("Unexpected model artifact")
    monitor.cooldown(serial, sink, "preflight-" + fmt, minimum=10)
    args = [BINARY, MODEL_ROOT + "/models/" + filename,
            "/data/local/tmp/xiaomi-hexagon-exl3/homura-baseline/prompt.txt", str(steps), str(repeats)]
    if forced: args.append("teacher-forced")
    command = ("export LD_LIBRARY_PATH=/data/local/tmp/llama.cpp/lib:/vendor/lib64 "
               "ADSP_LIBRARY_PATH=/data/local/tmp/llama.cpp/lib GGML_HEXAGON_PROFILE=0 "
               "GGML_HEXAGON_VERBOSE=0 GGML_SCHED_DEBUG=0; exec " + shlex.join(args))
    process = subprocess.Popen([monitor.ADB, "-s", serial, "shell", command], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
    path = directory / (fmt + ".log")
    records, errors = [], []
    state = {"phase": "load", "trial": -1}
    def reader():
        with path.open("w", encoding="utf-8") as output:
            for line in process.stdout:
                output.write(line); output.flush()
                if line.startswith("BASELINE "):
                    try:
                        row = json.loads(line[9:]); records.append(row)
                        state["phase"] = row["event"]
                        state["trial"] = row.get("trial", -1)
                        if row["event"] == "init": state["pid"] = row["pid"]
                        if row["event"] == "decode_end":
                            print("TRIAL", fmt, row["trial"], "tokens/s", round(row["tokens_per_second"], 3),
                                  "steps", row["decode_calls"], "stopped", row["stopped"], flush=True)
                    except Exception as error:
                        errors.append(str(error))
    thread = threading.Thread(target=reader, daemon=True); thread.start()
    began, abort, stopped_at = time.monotonic(), None, None
    print("START", fmt, "steps", steps, "repeats", repeats, flush=True)
    while process.poll() is None:
        before = time.monotonic()
        try:
            sample = monitor.sample(serial, sink, f"{fmt}/{state['trial']}/{state['phase']}")
            reason = monitor.stop_reason(sample)
        except Exception as error:
            reason = "Telemetry failure: " + str(error)
        if time.monotonic() - began > 300:
            reason = "Baseline wall timeout"
        if reason and not abort and process.poll() is None:
            pid = state.get("pid")
            if not pid:
                raise RuntimeError("PID unknown; remote state unknown")
            try:
                exe = monitor.shell(serial, f"readlink /proc/{pid}/exe").strip()
            except RuntimeError:
                if process.poll() is not None: break
                raise
            if exe != BINARY: raise RuntimeError("PID identity mismatch; refusing signal")
            abort, stopped_at = reason, time.monotonic()
            monitor.shell(serial, f"kill -INT {pid}")
        if abort and time.monotonic() - stopped_at > 15:
            raise RuntimeError("Baseline did not stop; remote state unknown")
        time.sleep(max(0, 0.5 - (time.monotonic() - before)))
    thread.join(timeout=5)
    if thread.is_alive(): raise RuntimeError("Log reader did not finish")
    result = {"format": fmt, "model_sha256": actual, "requested_steps": steps, "repeats": repeats,
              "minimum_steps": min(32, steps),
              "teacher_forced": forced,
              "prompt_sha256": hashlib.sha256((Path(__file__).parent / "prompts/homura-baseline.txt").read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256((Path(__file__).parent / "homura_baseline.cpp").read_bytes()).hexdigest(),
              "exit_code": process.returncode, "abort": abort, "parse_errors": errors, "records": records,
              "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    with (directory / "runs.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps(result, ensure_ascii=False) + "\n")
    print("FINISH", fmt, "exit", process.returncode, "abort", abort, "directory", directory, flush=True)
    if process.returncode or abort or errors: raise RuntimeError("Baseline interrupted or failed")
    completed = [r for r in records if r["event"] == "decode_end"]
    minimum = steps if forced else min(32, steps)
    if len(completed) != repeats or any(r["decode_calls"] < minimum or
                                      (r["decode_calls"] < steps and not r["eog"]) for r in completed):
        raise RuntimeError("Translation ended before the minimum timing length or without EOS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--formats", nargs="+", choices=tuple(FILES), default=list(FILES))
    parser.add_argument("--teacher-forced", action="store_true", help="Use identical fixed input tokens, independent of natural EOS; this is not free-running translation")
    args = parser.parse_args()
    if not 8 <= args.steps <= 256 or not 1 <= args.repeats <= 5: parser.error("Invalid limits")
    active = monitor.shell(args.serial, "pidof homura_baseline llama-server llama-completion accelerator_bench || true").strip()
    if active: raise RuntimeError("Concurrent project processes: " + active)
    directory = Path(__file__).resolve().parents[1] / "benchmark-raw" / (
        "homura-baseline-" + datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d-%H%M%S"))
    directory.mkdir(exist_ok=False)
    with (directory / "telemetry.jsonl").open("w", encoding="utf-8") as sink:
        for fmt in args.formats: run_one(args.serial, directory, sink, fmt, args.steps, args.repeats, args.teacher_forced)
        monitor.cooldown(args.serial, sink, "final-idle", minimum=10)
    print("PASS", directory, flush=True)


if __name__ == "__main__":
    main()
