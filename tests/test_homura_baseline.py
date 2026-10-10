import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("baseline_analysis", Path(__file__).resolve().parents[1] / "tools/analyze_homura_baseline.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def fixture():
    records = [
        {"event": "init", "teacher_forced": True, "boot_s": -2},
        {"event": "loaded", "backend": "HTP0", "kv": "f16", "prompt_tokens": 100, "boot_s": -1},
        {"event": "idle_start", "trial": 0, "boot_s": 0},
        {"event": "prefill_start", "trial": 0, "boot_s": 15},
        {"event": "prefill_end", "trial": 0, "boot_s": 15.31, "start_boot_s": 15.01, "end_boot_s": 15.3, "seconds": 0.29, "tokens": 100},
        {"event": "decode_start", "trial": 0, "boot_s": 15.32},
        {"event": "decode_end", "trial": 0, "boot_s": 19.34, "start_boot_s": 15.33, "end_boot_s": 19.33,
         "seconds": 4, "decode_calls": 64, "tokens_per_second": 16, "stopped": False, "teacher_forced": True,
         "supplied_ids": list(range(64)), "generated_ids": list(range(65))},
        {"event": "finished", "stopped": False, "boot_s": 20}]
    run = {"exit_code": 0, "abort": None, "parse_errors": [], "teacher_forced": True, "repeats": 1,
           "requested_steps": 64, "format": "Q4_K_M", "records": records, "model_sha256": "fixture",
           "raw_sha256": "fixture", "prompt_sha256": "fixture", "source_sha256": "fixture"}
    samples = [{"label": "Q4_K_M/0/phase", "boot_s": i / 2, "usb_powered": False,
                "ac_powered": False, "wireless_powered": False, "battery_status": 3, "wakefulness": "Dozing",
                "signed_battery_W": 4, "ibat_A": 1, "vbat_V": 4, "npu_max_C": 60, "cpu_max_C": 40, "skin_C": 30} for i in range(41)]
    return run, samples


class HomuraBaselineTests(unittest.TestCase):
    def test_exact_timing_and_energy(self):
        run, samples = fixture()
        result = analysis.audit_run(run, samples)
        self.assertEqual(result["aggregate"]["pooled_steps_per_second"], 16)
        self.assertEqual(result["aggregate"]["pooled_estimated_device_joules_per_step"], 0.25)

    def test_rejects_interrupted_and_nonfixed_work(self):
        for key, value in (("exit_code", 2), ("abort", "thermal"), ("teacher_forced", False)):
            run, samples = fixture(); run[key] = value
            with self.assertRaises(ValueError): analysis.audit_run(run, samples)

    def test_incomplete_steps_and_bad_rate(self):
        for key, value in (("decode_calls", 23), ("seconds", float("nan")), ("tokens_per_second", 64), ("stopped", True)):
            run, samples = fixture(); run["records"][6][key] = value
            with self.assertRaises(ValueError): analysis.audit_run(run, samples)

    def test_charging_or_insufficient_power(self):
        run, samples = fixture(); samples[32]["usb_powered"] = True
        with self.assertRaises(ValueError): analysis.audit_run(run, samples)
        run, samples = fixture()
        with self.assertRaises(ValueError): analysis.audit_run(run, samples[:32])

    def test_phase_ordering_and_tokens(self):
        for event, key, value in ((6, "start_boot_s", 14), (4, "tokens", 99), (6, "supplied_ids", [1])):
            run, samples = fixture(); run["records"][event][key] = value
            with self.assertRaises(ValueError): analysis.audit_run(run, samples)


if __name__ == "__main__":
    unittest.main()
