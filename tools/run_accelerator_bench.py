"""Isolated GPU/NPU benchmarks with fresh HAL telemetry and thermal stop guards."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import threading
import time

ADB = r"C:\Android\platform-tools\platform-tools-latest-windows\platform-tools\adb.exe"
DEVICE_DIR = "/data/local/tmp/xiaomi-hexagon-exl3/accelerator-bench"
BINARY = DEVICE_DIR + "/accelerator_bench"
TEMPERATURE = re.compile(r"Temperature\{mValue=([-+.\deE]+), mType=\d+, mName=([^,]+), mStatus=\d+\}")
TELEMETRY_COMMAND = "dumpsys thermalservice; dumpsys battery; cat /proc/uptime; dumpsys power | grep mWakefulness="


def shell(serial, command, timeout=10):
    result = subprocess.run([ADB, "-s", serial, "shell", command], capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"ADB failed ({result.returncode}): {result.stderr.strip()}")
    return result.stdout


def telemetry(text):
    section = text.split("Current temperatures from HAL:", 1)
    if len(section) != 2:
        raise ValueError("No fresh HAL temperature block")
    section = section[1].split("Current cooling devices from HAL:", 1)[0]
    values = {name: float(value) for value, name in TEMPERATURE.findall(section)}
    if not {"ibat", "vbat", "skin", "battery"} <= values.keys() or not all(math.isfinite(v) for v in values.values()):
        raise ValueError("Missing/nonfinite fresh HAL data")
    def field(pattern):
        match = re.search(pattern, text, re.MULTILINE)
        if not match:
            raise ValueError(f"Missing telemetry field {pattern}")
        return match.group(1)
    def hottest(prefix):
        group = [v for name, v in values.items() if name.startswith(prefix)]
        if not group:
            raise ValueError(f"Missing {prefix} sensors")
        return max(group)
    boot = re.findall(r"^([\d.]+) [\d.]+\s*$", text, re.MULTILINE)
    if not boot:
        raise ValueError("Missing device uptime")
    return {"boot_s": float(boot[-1]), "temperatures_C": {k: v for k, v in values.items() if k not in ("ibat", "vbat", "socd")},
            "ibat_A": values["ibat"], "vbat_V": values["vbat"], "signed_battery_W": values["ibat"] * values["vbat"],
            "skin_C": values["skin"], "battery_C": values["battery"], "gpu_max_C": hottest("GPU"),
            "npu_max_C": hottest("nsp"), "cpu_max_C": hottest("CPU"),
            "usb_powered": field(r"^\s+USB powered: (true|false)$") == "true",
            "ac_powered": field(r"^\s+AC powered: (true|false)$") == "true",
            "wireless_powered": field(r"^\s+Wireless powered: (true|false)$") == "true",
            "battery_status": int(field(r"^\s+status: (\d+)$")), "battery_level": int(field(r"^\s+level: (\d+)$")),
            "charge_counter_uAh": int(field(r"^\s+Charge counter: (\d+)$")),
            "thermal_status": int(field(r"^Thermal Status: (\d+)$")),
            "wakefulness": field(r"mWakefulness=(\w+)")}


def stop_reason(sample):
    if sample["usb_powered"] or sample["ac_powered"] or sample["wireless_powered"] or sample["battery_status"] != 3:
        return "External power/charging detected"
    if sample["wakefulness"] not in ("Asleep", "Dozing"):
        return "Screen is awake"
    if sample["battery_level"] < 25:
        return "Battery below 25 percent"
    if sample["battery_C"] >= 40 or sample["skin_C"] >= 42 or sample["gpu_max_C"] >= 80 or sample["npu_max_C"] >= 80 or sample["cpu_max_C"] >= 85:
        return "Project thermal stop threshold"
    if sample["thermal_status"] >= 2:
        return "Android thermal status moderate or higher"
    return None


def sample(serial, sink, label):
    start = time.monotonic()
    row = telemetry(shell(serial, TELEMETRY_COMMAND))
    row.update(label=label, host_unix_s=time.time(), query_seconds=time.monotonic() - start)
    sink.write(json.dumps(row) + "\n")
    sink.flush()
    return row


def cooldown(serial, sink, label, minimum=3, maximum=150):
    start = time.monotonic()
    rows = []
    while True:
        before = time.monotonic()
        row = sample(serial, sink, label)
        reason = stop_reason(row)
        if reason and reason != "Project thermal stop threshold":
            raise RuntimeError(reason)
        rows.append(row)
        elapsed = time.monotonic() - start
        cool = row["skin_C"] < 36.5 and row["battery_C"] < 33 and row["gpu_max_C"] < 50 and row["npu_max_C"] < 60
        if elapsed >= minimum and cool:
            return rows
        if elapsed > maximum:
            raise RuntimeError("Cooldown limit reached")
        time.sleep(max(0, 0.5 - (time.monotonic() - before)))


def run_case(serial, directory, sink, index, operation, backend, fmt, batch, seconds, stage, duty=1):
    label = f"{index:03d}-{stage}-{operation}-{backend}-{fmt}-b{batch}"
    print("CASE", label, "seconds=", seconds, "duty=", duty, flush=True)
    idle = cooldown(serial, sink, "idle-" + label, minimum=15 if stage == "sustained" else 5)
    command_args = [BINARY, operation, backend, fmt]
    if operation == "mat":
        command_args.append(str(batch))
    command_args.append(str(seconds))
    command_args.append(str(duty))
    command = (f"cd {DEVICE_DIR} && export LD_LIBRARY_PATH=/data/local/tmp/llama.cpp/lib:/vendor/lib64 "
               "&& export ADSP_LIBRARY_PATH=/data/local/tmp/llama.cpp/lib && export GGML_HEXAGON_PROFILE=0 "
               "&& exec " + shlex.join(command_args))
    process = subprocess.Popen([ADB, "-s", serial, "shell", command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", bufsize=1)
    records, parse_errors = [], []
    lock = threading.Lock()
    raw_path = directory / (label + ".log")
    def reader():
        with raw_path.open("w", encoding="utf-8") as log:
            for line in process.stdout:
                log.write(line); log.flush()
                if line.startswith("BENCH "):
                    try:
                        record = json.loads(line[6:])
                        with lock:
                            records.append(record)
                    except Exception as error:
                        parse_errors.append(str(error))
    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    start, abort, stopped_at = time.monotonic(), None, None
    samples = []
    while process.poll() is None:
        before = time.monotonic()
        try:
            row = sample(serial, sink, label)
            samples.append(row)
            reason = stop_reason(row)
        except Exception as error:
            reason = "Telemetry failure: " + str(error)
        if time.monotonic() - start > seconds + 60:
            reason = "Benchmark wall timeout"
        if reason and abort is None and process.poll() is None:
            abort, stopped_at = reason, time.monotonic()
            with lock:
                ids = [r["pid"] for r in records if r.get("event") == "init"]
            if len(ids) != 1:
                raise RuntimeError("Cannot identify remote benchmark PID; remote state unknown")
            pid = int(ids[0])
            try:
                executable = shell(serial, f"readlink /proc/{pid}/exe").strip()
            except RuntimeError:
                if process.poll() is not None:
                    abort = None
                    break
                raise
            if executable != BINARY:
                raise RuntimeError("Remote PID identity changed; refusing signal")
            shell(serial, f"kill -INT {pid}")
        if abort and time.monotonic() - stopped_at > 15:
            raise RuntimeError("Remote benchmark did not stop; remote state unknown")
        time.sleep(max(0, 0.5 - (time.monotonic() - before)))
    thread.join(timeout=5)
    if thread.is_alive():
        raise RuntimeError("Benchmark output reader did not finish")
    with lock:
        records = list(records)
    starts = [r for r in records if r.get("event") == "start"]
    results = [r for r in records if r.get("event") == "result"]
    validations = [r for r in records if r.get("event") == "validation"]
    windows = [r for r in records if r.get("event") == "window"]
    valid = (process.returncode == 0 and not abort and not parse_errors and len(starts) == len(results) == 1
             and len(validations) == 2 and bool(windows) and not results[0].get("stopped"))
    result = {"label": label, "stage": stage, "operation": operation, "backend": backend, "format": fmt,
              "batch": batch, "requested_seconds": seconds, "duty": duty, "valid": valid, "exit_code": process.returncode,
              "abort": abort, "parse_errors": parse_errors, "records": records,
              "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest()}
    if valid:
        settle = 5 if seconds >= 10 else 1
        active = [r for r in samples if starts[0]["boot_s"] + settle <= r["boot_s"] <= windows[-1]["boot_s"]]
        result["telemetry_samples"] = len(active)
        result["battery_mean_W"] = statistics.mean(r["signed_battery_W"] for r in active) if active else None
        result["idle_mean_W"] = statistics.mean(r["signed_battery_W"] for r in idle)
        result["battery_increment_W"] = result["battery_mean_W"] - result["idle_mean_W"] if active else None
        result["gpu_max_C"] = max((r["gpu_max_C"] for r in active), default=None)
        result["npu_max_C"] = max((r["npu_max_C"] for r in active), default=None)
        result["skin_max_C"] = max((r["skin_C"] for r in active), default=None)
        result["result"] = results[0]
        result["start"] = starts[0]
    with (directory / "results.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps(result) + "\n")
    if not valid:
        raise RuntimeError(f"Invalid case {label}; exit={process.returncode}, reason={abort}; see {raw_path}")
    print("RESULT", label, "us/op=", round(results[0]["us_per_op"], 2), "GFLOPS=", round(results[0]["gflops"], 2),
          "battery_W=", result["battery_mean_W"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="192.168.1.135:5555")
    parser.add_argument("--stage", choices=("smoke", "short", "sustained", "copy", "pilot"), required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    active = shell(args.serial, "pidof accelerator_bench test-backend-ops llama-server llama-completion gpu_npu_probe || true").strip()
    if active:
        raise RuntimeError("Concurrent project process: " + active)
    directory = args.output or Path(__file__).resolve().parents[1] / "benchmark-raw" / (
        "accelerator-" + datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d-%H%M%S") + "-" + args.stage)
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "telemetry.jsonl").open("w", encoding="utf-8") as sink:
        initial = sample(args.serial, sink, "preflight")
        if stop_reason(initial):
            raise RuntimeError(stop_reason(initial))
        (directory / "metadata.json").write_text(json.dumps({"stage": args.stage, "serial": args.serial,
            "initial": initial, "telemetry_command": TELEMETRY_COMMAND, "battery_measurement": "signed ibat_A * vbat_V; whole-device estimate, not accelerator rail",
        "sampling_target_seconds": 0.5,
        "benchmark_source_sha256": hashlib.sha256((Path(__file__).parent / "accelerator_bench.cpp").read_bytes()).hexdigest()}, indent=2), encoding="utf-8")
        cases = []
        if args.stage == "pilot":
            cases = [("mat", b, "q8_0", 512, 5, 0.5) for b in ("GPUOpenCL", "HTP0")]
        elif args.stage == "copy":
            cases = [("copy", b, f, 1, 3, 1) for f in ("contiguous", "transpose") for b in ("GPUOpenCL", "HTP0", "HTP0", "GPUOpenCL")]
        else:
            for fmt in ("f16", "q8_0", "q4_0"):
                batches = (1, 4, 8, 512) if args.stage == "short" else (1, 512)
                for batch in batches:
                    backends = ("GPUOpenCL", "HTP0", "HTP0", "GPUOpenCL") if args.stage == "short" else ("GPUOpenCL", "HTP0")
                    seconds = 30 if args.stage == "sustained" else (1 if batch == 512 else 2)
                    if args.stage != "sustained" and (fmt == "q4_0" or fmt == "q8_0" and batch > 1):
                        seconds = 0.25
                    duty = 0.25 if args.stage == "sustained" else 1
                    cases.extend(("mat", b, fmt, batch, seconds, duty) for b in backends)
        if not 0 <= args.start_index < len(cases):
            raise ValueError("Start index outside the case list")
        for index, case in enumerate(cases):
            if index < args.start_index:
                continue
            run_case(args.serial, directory, sink, index, *case[:5], args.stage, case[5])
        cooldown(args.serial, sink, "final-idle", minimum=10)
    print("PASS", args.stage, "cases=", len(cases) - args.start_index, "output=", directory, flush=True)


if __name__ == "__main__":
    main()
