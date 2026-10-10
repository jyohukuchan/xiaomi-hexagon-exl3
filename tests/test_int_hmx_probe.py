import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("int_analysis", Path(__file__).resolve().parents[1] / "tools/analyze_int_hmx_probe.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def fixture():
    init = {"event": "init", "boot_s": 1, "bits": 4, "batch": 1, "api_batch": 64, "n": 32, "k": 32, "repeats": 2}
    valid = {"event": "validation", "boot_s": 2, "mismatches": 0, "guard_bytes": 4096, "samples": 32, "full": True}
    result = {"event": "result", "boot_s": 4, "calls": 2, "stopped": False, "seconds": .01, "us_per_call": 5000,
              "effective_GOPS": .0004096, "physical_GOPS": .0262144, "scope": "test"}
    return {"exit_code": 0, "abort": None, "parse_errors": [], "bits": 4, "batch": 1, "n": 32, "k": 32,
            "repeats": 2, "raw_sha256": "fixture", "records": [init, {"event": "version", "value": "1_0_0_beta1_HEXAGON_V75", "boot_s": 1.1},
            valid, dict(valid, boot_s=3), result, {"event": "finished", "boot_s": 5, "stopped": False}]}


class IntegerProbeTests(unittest.TestCase):
    def test_valid_and_logical_padding(self):
        r = analysis.audit(fixture())
        self.assertEqual(r["batch"], 1)
        self.assertEqual(r["api_batch"], 64)

    def test_legacy_must_be_full_rows(self):
        r = fixture(); del r["records"][0]["api_batch"]
        with self.assertRaises(ValueError): analysis.audit(r)

    def test_rejects_failure_and_abort(self):
        for key, value in (("exit_code", 1), ("abort", "thermal"), ("parse_errors", ["bad JSON"])):
            r = fixture(); r[key] = value
            with self.assertRaises(ValueError): analysis.audit(r)

    def test_rejects_corruption(self):
        for key, value in (("mismatches", 1), ("guard_bytes", 0), ("samples", 1), ("nmse", float("nan"))):
            r = fixture(); r["records"][2][key] = value
            with self.assertRaises(ValueError): analysis.audit(r)

    def test_rejects_wrong_counts_and_rates(self):
        for key, value in (("calls", 1), ("seconds", 0), ("us_per_call", 1), ("physical_GOPS", .0004096), ("stopped", True)):
            r = fixture(); r["records"][4][key] = value
            with self.assertRaises(ValueError): analysis.audit(r)

    def test_rejects_lifecycle_and_architecture(self):
        r = fixture(); r["records"].pop()
        with self.assertRaises(ValueError): analysis.audit(r)
        r = fixture(); r["records"][1]["value"] = "V73"
        with self.assertRaises(ValueError): analysis.audit(r)

    def test_rejects_wrong_gguf_reconstruction(self):
        r = fixture(); r["tensor"] = "blk.0.attn_gate.weight"; r["records"][0]["tensor"] = r["tensor"]
        r["records"].insert(2, {"event": "gguf_mapping", "format": "q4_K", "weight_elements": 1024, "nmse": .1})
        with self.assertRaises(ValueError): analysis.audit(r)


if __name__ == "__main__": unittest.main()
