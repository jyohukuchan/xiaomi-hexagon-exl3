import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("check_gpu_probe", Path(__file__).resolve().parents[1] / "tools/check_gpu_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def interop_log():
    lines = []
    for size in probe.SIZES:
        lines.append(f"interop_sample map=1 bytes={size} index=0 total_us=1000 gpu_us=10 sync_us=10 rpc_us=100 "
                     "hvx_copy_us=10 noop_rpc_us=50 read_rpc_us=90 hvx_read_us=5 mismatches=0")
        lines.append(f"interop_summary map=1 bytes={size} samples=1 total_p50_us=1000 total_p95_us=1000 "
                     "gpu_p50_us=10 sync_p50_us=10 rpc_p50_us=100 hvx_copy_p50_us=10 noop_rpc_p50_us=50 "
                     f"payload_GBps={size / 1e6} read_rpc_p50_us=90 hvx_read_p50_us=5 "
                     "read_effective_GBps=1 hvx_read_GBps=1 mismatches=0")
    return "\n".join(lines + ["PASS mode=interop"]) + "\n"


def dequant_log():
    lines = [f"codebook_test cb={cb} states=65536 mismatches=0" for cb in range(3)]
    cases = [("synthetic", 256, 384, bits, cb, layout)
             for bits in range(1, 9) for cb in range(3) for layout in (0, 1)]
    cases += [("synthetic", 256, 384, bits, 2, 2) for bits in (4, 6, 8)]
    cases += [("single_group", 128, 128, bits, 2, layout) for bits in (4, 6, 8) for layout in (0, 1, 2)]
    cases += [("streaming_chunk", 2048, 6144, bits, 2, layout) for bits in (4, 6) for layout in (0, 1, 2)]
    for label, k, n, bits, cb, layout in cases:
        lines.append(f"dequant_summary label={label} k={k} n={n} bits={bits} cb={cb} layout={layout} samples=1 "
                     "gpu_p50_us=10 gpu_p95_us=11 gpu_to_dsp_p50_us=100 rpc_p50_us=50 output_GBps=1 mismatches=0")
    return "\n".join(lines + ["PASS mode=dequant"]) + "\n"


class ProbeAuditTests(unittest.TestCase):
    def test_complete_interop(self):
        self.assertEqual(probe.audit(interop_log())["sample_count"], 7)

    def test_reject_missing_pass(self):
        with self.assertRaises(ValueError):
            probe.audit(interop_log().replace("PASS mode=interop", ""))

    def test_reject_missing_sample(self):
        with self.assertRaises(ValueError):
            probe.audit("\n".join(line for line in interop_log().splitlines() if not line.startswith("interop_sample map=1 bytes=128 ")))

    def test_reject_nan_and_mismatch(self):
        for replacement in ("gpu_us=nan", "gpu_us=-1"):
            with self.assertRaises(ValueError):
                probe.audit(interop_log().replace("gpu_us=10", replacement))
        with self.assertRaises(ValueError):
            probe.audit(interop_log().replace("mismatches=0", "mismatches=1", 1))

    def test_reject_wrong_bandwidth(self):
        with self.assertRaises(ValueError):
            probe.audit(interop_log().replace("payload_GBps=0.000128", "payload_GBps=128"))

    def test_reject_dequant_coverage(self):
        with self.assertRaises(ValueError):
            probe.audit("codebook_test cb=2 states=65536 mismatches=0\nPASS mode=dequant\n")

    def test_complete_dequant(self):
        result = probe.audit(dequant_log())
        self.assertEqual(result["case_count"], 66)
        self.assertEqual(result["codebook_states"], 196608)

    def test_require_cuda_fixtures(self):
        with self.assertRaises(ValueError):
            probe.audit(dequant_log(), require_fixtures=True)


if __name__ == "__main__":
    unittest.main()
