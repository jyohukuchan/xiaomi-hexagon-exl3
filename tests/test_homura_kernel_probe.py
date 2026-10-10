import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("homura_analysis", Path(__file__).resolve().parents[1] / "tools/analyze_homura_kernel_probe.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)

LINE = "D ggml-hex: HTP0 profile-op MUL_MAT|blk.0.attn.weight x act -> out|2048:6144 x 2048:30 -> 6144:30|q8_0 x f32 -> f32|34:2176 x 4:8192 -> 4:24576|hmx-tiled vtcm 8000000|usec 435 cycles 845450 start 1 mhz 1943.6"
NODE = "D node # 15 ( MUL_MAT): node_15 ( 24K) [ HTP0 ] use=1,c=1:"


class HomuraKernelProbeTests(unittest.TestCase):
    def test_valid_matrix_profile(self):
        record = analysis.profiles(LINE)[0]
        self.assertEqual(record["input_columns"], 30)
        self.assertEqual(record["path"], "hmx-tiled")
        self.assertEqual(record["weights"], {"blk.0.attn.weight": "q8_0"})

    def test_generation_alone_is_not_kernel_evidence(self):
        with self.assertRaises(ValueError):
            analysis.profiles("こんにちは。 [end of text]")

    def test_invalid_profile_rejected(self):
        for line in (LINE.replace("hmx-tiled", "unknown"), LINE.replace("6144:30|q8", "6144:0|q8"),
                     LINE.replace(" x f32 -> f32", " -> f32"), LINE.replace("usec 435", "usec nan")):
            with self.assertRaises(ValueError):
                analysis.profiles(line)

    def test_scheduler_requires_npu(self):
        self.assertEqual(analysis.scheduler_matrix_assignments(NODE), {"HTP0": 1})
        for text in ("", NODE.replace("HTP0", "CPU"), NODE + "\n" + NODE.replace("HTP0", "GPUOpenCL")):
            with self.assertRaises(ValueError):
                analysis.scheduler_matrix_assignments(text)


if __name__ == "__main__":
    unittest.main()
