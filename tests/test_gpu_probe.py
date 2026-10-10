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


def pmu_log(event_set=0):
    events = probe.PMU_EVENTS[event_set]
    lines = ["pmu_schema version=2 report_bytes=544 set=" + str(event_set) +
             "".join(f" e{i}={event}" for i, event in enumerate(events))]
    for size in probe.SIZES:
        for source in ("cpu", "gpu"):
            for mode in ("empty", "hvx", "scalar", "hvx_noinv"):
                for phase in ("arrival", "cold", "warm", "outer"):
                    loads = size // 128 if mode == "hvx" and phase != "arrival" else 0
                    axi = size // 128 + 3 if mode == "hvx" and phase in ("cold", "outer") else 0
                    counters = "".join(f" c{i}={loads + 1 if event == 0x118 else axi if event == 0x40 else 0}"
                                       for i, event in enumerate(events))
                    key = f"set={event_set} src={source} mode={mode} bytes={size} phase={phase}"
                    lines.append(f"pmu_sample {key} index=0 usec=10 cycles=100{counters} mismatches=0")
                    lines.append(f"pmu_summary {key} samples=1 p50_us=10 p95_us=10{counters} mismatches=0")
    return "\n".join(lines + ["PASS mode=pmu"]) + "\n"


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

    def test_pmu_complete_all_banks(self):
        for bank in range(4):
            result = probe.audit(pmu_log(bank))
            self.assertEqual(result["sample_count"], 224)
            self.assertTrue(result["calibration"])

    def test_pmu_reject_arch_neutral_ids(self):
        with self.assertRaises(ValueError):
            probe.audit(pmu_log().replace("e0=63", "e0=32823"))

    def test_pmu_reject_missing_phase(self):
        with self.assertRaises(ValueError):
            probe.audit("\n".join(line for line in pmu_log().splitlines() if "phase=warm" not in line))

    def test_pmu_reject_summary_forgery(self):
        with self.assertRaises(ValueError):
            probe.audit(pmu_log().replace("p50_us=10", "p50_us=9", 1))

    def test_pmu_reject_uncalibrated_zero_loads(self):
        lines = []
        for line in pmu_log().splitlines():
            if line.startswith(("pmu_sample ", "pmu_summary ")):
                line = " ".join("c7=0" if token.startswith("c7=") else token for token in line.split())
            lines.append(line)
        with self.assertRaises(ValueError):
            probe.audit("\n".join(lines))

    def test_pmu_reject_mismatched_counter_banks(self):
        lines = []
        for line in pmu_log(3).splitlines():
            if line.startswith(("pmu_sample ", "pmu_summary ")):
                line = " ".join("c3=0" if token.startswith("c3=") else token for token in line.split())
            lines.append(line)
        with self.assertRaises(ValueError):
            probe.audit("\n".join(lines))

    def test_pmu_records_configuration_rejection(self):
        text = pmu_log().replace("PASS mode=pmu", "pmu_rejected set=0 src=gpu mode=empty bytes=128 index=0 attempt=0 flags=31\nPASS mode=pmu")
        self.assertEqual(probe.audit(text)["rejected_count"], 1)

    def test_pmu_rejects_old_unguarded_abi(self):
        with self.assertRaises(ValueError):
            probe.audit(pmu_log().replace("version=2 report_bytes=544", "version=1 report_bytes=528"))


if __name__ == "__main__":
    unittest.main()
