import importlib.util
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "tools" / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


runner = module("run_accelerator_bench")
analyzer = module("analyze_accelerator_bench")


def thermal_dump():
    names = {"battery": 30, "skin": 33, "CPU0": 40, "GPU0": 41, "nsp0": 42, "ibat": 0.5, "vbat": 4}
    return ("Thermal Status: 0\nCached temperatures:\nTemperature{mValue=999, mType=9, mName=nsp0, mStatus=0}\n"
            "Current temperatures from HAL:\n" + "\n".join(f"Temperature{{mValue={v}, mType=0, mName={k}, mStatus=0}}" for k, v in names.items()) +
            "\nCurrent cooling devices from HAL:\n  USB powered: false\n  AC powered: false\n  Wireless powered: false\n"
            "  status: 3\n  level: 80\n  Charge counter: 1234\n100.00 400.00\n  mWakefulness=Dozing\n")


def valid_case(operation="mat"):
    size = 4096 * (14336 if operation == "mat" else 4096)
    flops = 2 * size if operation == "mat" else 0
    traffic = size * (2 if operation == "mat" else 4)
    validation = {"finite_elements": 4096, "reference_samples": 64, "nmse": 1e-8, "max_abs": 0.01} if operation == "mat" else {"bit_exact_elements": size}
    window = {"event": "window", "index": 0, "boot_s": 31, "seconds": 30, "ops": 300,
              "us_per_op": 100000, "gflops": flops * 10 / 1e9, "effective_GBps": traffic * 10 / 1e9,
              "active_seconds": 7.5, "active_gflops": flops * 40 / 1e9}
    start = {"event": "start", "boot_s": 1, "backend": "GPUOpenCL", "format": "f16" if operation == "mat" else "contiguous",
             "batch": 1, "duty": 0.25, "m": 4096, "k": size // 4096, "weight_bytes": size * 2,
             "traffic_bytes_per_op": traffic, "flops_per_op": flops}
    result = {**window, "event": "result", "boot_s": 33, "stopped": False}
    return {"valid": True, "exit_code": 0, "abort": None, "parse_errors": [], "operation": operation,
            "backend": "GPUOpenCL", "format": start["format"], "batch": 1, "duty": 0.25,
            "requested_seconds": 30, "label": "000-sustained-test", "stage": "sustained",
            "telemetry_samples": 10, "battery_mean_W": 2, "idle_mean_W": 2, "battery_increment_W": 0,
            "gpu_max_C": 41, "npu_max_C": 42, "skin_max_C": 33,
            "raw_sha256": hashlib.sha256(b"raw log").hexdigest(),
            "records": [{**validation, "event": "validation", "boot_s": 0}, start, window,
                        {**validation, "event": "validation", "boot_s": 32}, result]}


class AcceleratorTests(unittest.TestCase):
    def test_original_case_plans_unchanged(self):
        self.assertEqual(len(runner.make_cases("short")), 48)
        self.assertEqual(len(runner.make_cases("sustained")), 12)
        self.assertEqual(len(runner.make_cases("copy")), 8)
        self.assertEqual(len(runner.make_cases("pilot")), 2)
        self.assertEqual(runner.make_cases("short")[20], ("mat", "GPUOpenCL", "q8_0", 4, 0.25, 1))

    def test_targeted_q8_power_plan(self):
        cases = runner.make_cases("sustained", ["q8_0"], [4, 16, 64], 2)
        self.assertEqual(len(cases), 12)
        for offset, batch in enumerate((4, 16, 64)):
            group = cases[offset * 4:offset * 4 + 4]
            self.assertEqual([c[1] for c in group], ["GPUOpenCL", "HTP0", "HTP0", "GPUOpenCL"])
            self.assertTrue(all(c[2:] == ("q8_0", batch, 30, 0.25) for c in group))

    def test_invalid_case_selection(self):
        for stage, formats, batches, repeats in (("copy", ["q8_0"], None, 1), ("sustained", None, [0], 1),
                                                ("sustained", None, [513], 1), ("sustained", None, [4, 4], 1),
                                                ("sustained", ["q8_0", "q8_0"], None, 1), ("sustained", None, None, 0)):
            with self.assertRaises(ValueError):
                runner.make_cases(stage, formats, batches, repeats)

    def test_fresh_hal_not_cached(self):
        row = runner.telemetry(thermal_dump())
        self.assertEqual(row["npu_max_C"], 42)
        self.assertEqual(row["signed_battery_W"], 2)
        self.assertIsNone(runner.stop_reason(row))

    def test_missing_fresh_hal_rejected(self):
        with self.assertRaises(ValueError):
            runner.telemetry(thermal_dump().replace("Current temperatures from HAL:", "missing:"))

    def test_charging_and_awake_stop(self):
        row = runner.telemetry(thermal_dump())
        row["usb_powered"] = True
        self.assertIsNotNone(runner.stop_reason(row))
        row["usb_powered"], row["wakefulness"] = False, "Awake"
        self.assertIsNotNone(runner.stop_reason(row))

    def test_thermal_stop(self):
        row = runner.telemetry(thermal_dump())
        row["npu_max_C"] = 80
        self.assertEqual(runner.stop_reason(row), "Project thermal stop threshold")

    def test_nonfinite_or_nonpositive_rejected(self):
        for value in (float("nan"), float("inf"), 0, -1):
            with self.assertRaises(ValueError):
                analyzer.positive(value)

    def test_units_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            analyzer.close(40.36, 43.33621233664)

    def test_valid_matrix_and_copy(self):
        for operation in ("mat", "copy"):
            analyzer.audit_case(valid_case(operation))

    def test_corrupt_case_rejected(self):
        for index, field, value in ((1, "weight_bytes", 1), (1, "backend", "CPU"), (1, "duty", 1),
                                    (0, "reference_samples", 63), (0, "nmse", float("nan")),
                                    (2, "active_seconds", 31), (2, "ops", 0),
                                    (2, "effective_GBps", 40.36), (4, "stopped", True),
                                    (4, "active_gflops", 0)):
            with self.subTest(field=field):
                row = copy.deepcopy(valid_case())
                row["records"][index][field] = value
                with self.assertRaises(ValueError):
                    analyzer.audit_case(row)

    def test_incomplete_copy_rejected(self):
        row = valid_case("copy")
        row["records"][0]["bit_exact_elements"] -= 1
        with self.assertRaises(ValueError):
            analyzer.audit_case(row)

    def test_duration_accounts_for_small_logging_gap(self):
        row = valid_case()
        row["requested_seconds"] = 30.005
        row["records"][2]["boot_s"] = 31.01
        analyzer.audit_case(row)
        row["records"][2]["boot_s"] = 31.5
        with self.assertRaises(ValueError):
            analyzer.audit_case(row)

    def test_power_recomputed_from_telemetry(self):
        row = valid_case()
        samples = [{**runner.telemetry(thermal_dump()), "label": row["label"], "boot_s": i + 6} for i in range(10)]
        samples.append({**samples[0], "label": "idle-" + row["label"], "boot_s": 0})
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / (row["label"] + ".log")).write_bytes(b"raw log")
            (directory / "telemetry.jsonl").write_text("\n".join(json.dumps(s) for s in samples))
            (directory / "results.jsonl").write_text(json.dumps(row))
            summary = analyzer.summarize(directory)
            self.assertEqual(summary["accepted_count"], 1)
            self.assertEqual(summary["cases"][0]["device_gflops_per_W"], row["records"][2]["gflops"] / 2)
            self.assertEqual(summary["aggregate"][0]["battery_mean_W_mean"], 2)
            self.assertEqual(analyzer.summarize(directory, (1, 2))["accepted_count"], 0)
            row["battery_mean_W"] = 1
            (directory / "results.jsonl").write_text(json.dumps(row))
            with self.assertRaises(ValueError):
                analyzer.summarize(directory)

    def test_power_aggregate_preserves_repeat_range(self):
        cases = [{"operation": "mat", "backend": "GPUOpenCL", "format": "q8_0", "batch": 16, "duty": 0.25,
                  "us_per_op": 100, "gflops": 100, "effective_GBps": 5,
                  "battery_mean_W": power, "device_gflops_per_W": 100 / power} for power in (2, 4)]
        result = analyzer.aggregate(cases)[0]
        self.assertEqual(result["replicates"], 2)
        self.assertEqual(result["battery_mean_W_mean"], 3)
        self.assertEqual(result["device_gflops_per_W_mean"], 37.5)
        self.assertEqual(result["device_gflops_per_W_min"], 25)
        self.assertEqual(result["device_gflops_per_W_max"], 50)


if __name__ == "__main__":
    unittest.main()
