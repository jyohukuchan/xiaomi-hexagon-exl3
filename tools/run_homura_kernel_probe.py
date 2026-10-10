"""Run short real-model Hexagon profiles without replacing any runtime library."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import threading
import time

import run_accelerator_bench as monitor

ROOT = "/data/local/tmp/xiaomi-hexagon-exl3/homura-kernel-probe"
BINARY = "/data/local/tmp/llama.cpp/bin/llama-completion"
FILES = {
    "Q4_K_M": ("Index-Homura-2B.Q4_K_M.gguf", "3fc56945f1db4c2b91b18ac9f4061b354d51c226a6fdd9ea95e72a927f13a675"),
    "Q8_0": ("Index-Homura-2B.Q8_0.gguf", "e3aa4850c1f4779328af5d721383bb5785dfdd7836f5a82976336c21178c7754"),
}


def run_one(serial, directory, sink, fmt):
    filename, expected_hash = FILES[fmt]
    actual = monitor.shell(serial, f"sha256sum {ROOT}/models/{filename}", timeout=30).split()[0]
    if actual != expected_hash:
        raise RuntimeError("Device GGUF hash does not match the published artifact")
    monitor.cooldown(serial, sink, "idle-" + fmt, minimum=10)
    expected_exe = monitor.shell(serial, "readlink -f " + BINARY).strip()
    args = [BINARY, "-m", ROOT + "/models/" + filename, "--device", "HTP0", "-ngl", "99",
            "--fit", "off", "-c", "512", "-b", "128", "-ub", "64", "-t", "6", "-n", "8",
            "--cache-type-k", "f16", "--cache-type-v", "f16", "--flash-attn", "on",
            "--temp", "0", "--seed", "42", "--no-warmup", "--conversation", "--single-turn",
            "--jinja", "--reasoning", "off",
            "--no-display-prompt", "--simple-io", "--perf", "--log-verbosity", "5", "-f", ROOT + "/prompt.txt"]
    command = ("export LD_LIBRARY_PATH=/data/local/tmp/llama.cpp/lib:/vendor/lib64 "
               "ADSP_LIBRARY_PATH=/data/local/tmp/llama.cpp/lib GGML_HEXAGON_PROFILE=1 "
               "GGML_HEXAGON_VERBOSE=1 GGML_SCHED_DEBUG=2; "
               "printf 'PROBE_PID %s\\n' \"$$\"; exec " + shlex.join(args))
    path = directory / (fmt + ".log")
    state = {}
    process = subprocess.Popen([monitor.ADB, "-s", serial, "shell", command], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
    def reader():
        with path.open("w", encoding="utf-8") as output:
            for line in process.stdout:
                output.write(line)
                output.flush()
                match = re.fullmatch(r"PROBE_PID (\d+)\s*", line)
                if match:
                    state["pid"] = int(match[1])
    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    began, abort, stopped_at = time.monotonic(), None, None
    print("START", fmt, flush=True)
    while process.poll() is None:
        before = time.monotonic()
        try:
            sample = monitor.sample(serial, sink, fmt)
            reason = monitor.stop_reason(sample)
        except Exception as error:
            reason = "Telemetry failure: " + str(error)
        if time.monotonic() - began > 180:
            reason = "Model diagnostic timeout"
        if reason and not abort and process.poll() is None:
            pid = state.get("pid")
            if not pid:
                raise RuntimeError("No diagnostic PID; remote state unknown")
            try:
                executable = monitor.shell(serial, f"readlink /proc/{pid}/exe").strip()
            except RuntimeError:
                if process.poll() is not None:
                    break
                raise
            if executable != expected_exe:
                raise RuntimeError("PID identity mismatch; remote state unknown, refusing signal")
            abort, stopped_at = reason, time.monotonic()
            monitor.shell(serial, f"kill -INT {pid}")
        if abort and time.monotonic() - stopped_at > 15:
            raise RuntimeError("Diagnostic did not stop; remote state unknown")
        time.sleep(max(0, 0.5 - (time.monotonic() - before)))
    thread.join(timeout=5)
    if thread.is_alive():
        raise RuntimeError("Diagnostic log reader did not finish")
    result = {"format": fmt, "model_sha256": actual, "argv": args, "exit_code": process.returncode,
              "abort": abort, "host_seconds": time.monotonic() - began,
              "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    result["profile_lines"] = path.read_text(encoding="utf-8").count("profile-op ")
    with (directory / "runs.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps(result) + "\n")
    print("FINISH", fmt, "exit", process.returncode, "abort", abort, "log", path, flush=True)
    if process.returncode or abort:
        raise RuntimeError("Real-model diagnostic did not complete cleanly")
    if not result["profile_lines"]:
        raise RuntimeError("No kernel profile was captured; successful generation alone is not path evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--formats", nargs="+", choices=tuple(FILES), default=list(FILES))
    args = parser.parse_args()
    active = monitor.shell(args.serial, "pidof accelerator_bench test-backend-ops llama-server llama-cli llama-completion gpu_npu_probe || true").strip()
    if active:
        raise RuntimeError("Concurrent project process: " + active)
    directory = Path(__file__).resolve().parents[1] / "benchmark-raw" / (
        "homura-kernels-" + datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d-%H%M%S"))
    directory.mkdir(exist_ok=False)
    with (directory / "telemetry.jsonl").open("w", encoding="utf-8") as sink:
        initial = monitor.sample(args.serial, sink, "preflight")
        if monitor.stop_reason(initial):
            raise RuntimeError(monitor.stop_reason(initial))
        for fmt in args.formats:
            run_one(args.serial, directory, sink, fmt)
        monitor.cooldown(args.serial, sink, "final-idle", minimum=10)
    print("PASS", directory, flush=True)


if __name__ == "__main__":
    main()
