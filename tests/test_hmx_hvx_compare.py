import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_hmx_hvx_compare as runner
import analyze_hmx_hvx_compare as analysis


def fixture():
    return [{"bits": 8, "batch": 1, "route": r, "trial": t, "model_sha256": "same", "tensor": "same", "n": 2048, "k": 2048,
             "api_batch": 32 if r == "hmx" else 1, "us_per_call": u}
            for t in range(2) for r, u in (("hmx", 200), ("hvx", 100))]


class RouteCompareTests(unittest.TestCase):
    def test_abba_order(self):
        self.assertEqual([r[3] for r in runner.cases([8], [1], 2)], ["hmx", "hvx", "hvx", "hmx"])

    def test_invalid_selection(self):
        for bits, batches, trials in (([8, 8], [1], 2), ([3], [1], 2), ([8], [1, 1], 2), ([8], [0], 2), ([8], [1], 5)):
            with self.assertRaises(ValueError): runner.cases(bits, batches, trials)

    def test_profile_path_not_vtcm_suffix(self):
        line = "ggml-hex: HTP0 profile-op MUL_MAT|route_weights x route_activations -> route_output|2048:2048 x 2048:32 -> 2048:32|q8_0 x f32 -> f32|strides|hmx-tiled vtcm 1234|usec 9 cycles 99"
        self.assertEqual(runner.profile_paths(line), ["hmx-tiled"])
        with self.assertRaises(ValueError): runner.profile_paths("no profile")
        with self.assertRaises(ValueError): runner.profile_paths(line.replace("hmx-tiled", "cpu"))

    def test_speedup_direction_and_padding(self):
        r = analysis.comparison(fixture(), 2)[0]
        self.assertEqual(r["hmx_speedup_over_hvx"], .5)
        self.assertEqual(r["hmx_api_rows"], 32)
        self.assertEqual(r["hvx_api_rows"], 1)

    def test_rejects_unmatched_and_missing_cases(self):
        rows = fixture(); rows[0]["model_sha256"] = "different"
        with self.assertRaises(ValueError): analysis.comparison(rows, 2)
        with self.assertRaises(ValueError): analysis.comparison(fixture()[:-1], 2)


if __name__ == "__main__": unittest.main()
