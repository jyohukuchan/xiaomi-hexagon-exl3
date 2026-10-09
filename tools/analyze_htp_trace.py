"""Summarize paired HTP stage events without adding parallel-thread time."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics


EVENT = re.compile(r"trace-evt (\w+): thread (\d+) info (\d+) (start|stop) (\d+)")
WINDOW = re.compile(r"cycles (\d+) start (\d+)")


def analyze(lines, kernel=None):
    lines = list(lines)
    pending = {}
    samples = defaultdict(list)
    errors = []
    windows = []
    if kernel:
        for line in lines:
            if "profile-op " in line and f"|{kernel} " in line:
                match = WINDOW.search(line)
                if match:
                    duration, start = map(int, match.groups())
                    if 0 < duration < 2**32:
                        windows.append((start % 2**32, duration))
                    else:
                        errors.append("Kernel window cannot be resolved with 32-bit event counters")
        if not windows:
            errors.append(f"No kernel windows found: {kernel}")
    for line_number, line in enumerate(lines, 1):
        match = EVENT.search(line)
        if not match:
            continue
        event, thread, info, kind, cycles = match.groups()
        thread, info, cycles = int(thread), int(info), int(cycles)
        key = thread, event, info
        if not 0 <= cycles < 2**32:
            errors.append(f"Line {line_number}: invalid cycle counter")
            continue
        if kernel and not any((cycles - start) % 2**32 <= duration for start, duration in windows):
            continue
        if kind == "start":
            if key in pending:
                errors.append(f"Line {line_number}: duplicate start {key}")
            pending[key] = cycles
        elif key not in pending:
            errors.append(f"Line {line_number}: stop without start {key}")
        else:
            samples[thread, event].append((cycles - pending.pop(key)) % 2**32)
    errors.extend(f"Unfinished event {key}" for key in sorted(pending))
    stages = []
    for (thread, event), values in sorted(samples.items()):
        stages.append({"thread": thread, "event": event, "count": len(values),
                       "total_cycles": sum(values), "mean_cycles": statistics.mean(values),
                       "median_cycles": statistics.median(values), "max_cycles": max(values)})
    return {"complete": bool(stages) and not errors, "errors": errors, "stages": stages,
            "kernel": kernel, "kernel_windows": len(windows),
            "scope": "Per-thread cycle intervals; concurrent threads and HMX/queue intervals overlap. Do not sum them as wall time."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--kernel", help="Restrict events to matching profile-op kernel windows")
    args = parser.parse_args()
    with args.log.open(encoding="utf-8") as source:
        result = analyze(source, args.kernel)
    print(json.dumps(result, indent=2))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
