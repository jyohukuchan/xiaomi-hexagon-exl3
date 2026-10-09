from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from analyze_htp_trace import analyze


class HtpTraceTest(unittest.TestCase):
    def event(self, name, thread, info, kind, cycles):
        return f"ggml-hex: HTP0 trace-evt {name}: thread {thread} info {info} {kind} {cycles}"

    def test_parallel_intervals_stay_separate(self):
        lines = [self.event("DMA", 0, 1, "start", 10), self.event("DMA", 1, 1, "start", 12),
                 self.event("DMA", 0, 1, "stop", 30), self.event("DMA", 1, 1, "stop", 40)]
        result = analyze(lines)
        self.assertTrue(result["complete"])
        self.assertEqual([s["total_cycles"] for s in result["stages"]], [20, 28])

    def test_counter_wrap_and_reused_info(self):
        lines = [self.event("BUFF", 0, 2, "start", 2**32 - 3), self.event("BUFF", 0, 2, "stop", 5),
                 self.event("BUFF", 0, 2, "start", 20), self.event("BUFF", 0, 2, "stop", 30)]
        result = analyze(lines)
        self.assertTrue(result["complete"])
        self.assertEqual(result["stages"][0]["count"], 2)
        self.assertEqual(result["stages"][0]["total_cycles"], 18)

    def test_incomplete_and_duplicate_events_fail(self):
        for lines in ([self.event("FENCE", 0, 1, "start", 10)],
                      [self.event("FENCE", 0, 1, "stop", 10)],
                      [self.event("FENCE", 0, 1, "start", 10), self.event("FENCE", 0, 1, "start", 20)]):
            result = analyze(lines)
            self.assertFalse(result["complete"])
            self.assertTrue(result["errors"])

    def test_no_events_is_not_complete(self):
        self.assertFalse(analyze(["ordinary profile line"])["complete"])

    def test_kernel_window_excludes_outer_queue_events(self):
        lines = ["profile-op MUL_MAT|a x b|exl3-hmx vtcm 0|usec 1 cycles 30 start 100",
                 self.event("BUFF", 0, 0, "start", 50), self.event("BUFF", 0, 0, "stop", 90),
                 self.event("DMA", 0, 0, "start", 110), self.event("DMA", 0, 0, "stop", 120)]
        result = analyze(lines, "exl3-hmx")
        self.assertTrue(result["complete"])
        self.assertEqual(result["kernel_windows"], 1)
        self.assertEqual([s["event"] for s in result["stages"]], ["DMA"])
        self.assertFalse(analyze(lines, "unknown-kernel")["complete"])

    def test_kernel_window_wrap(self):
        lines = [f"profile-op MUL_MAT|a x b|exl3-hmx vtcm 0|usec 1 cycles 20 start {2**32 - 5}",
                 self.event("DMA", 0, 0, "start", 2**32 - 2), self.event("DMA", 0, 0, "stop", 5)]
        result = analyze(lines, "exl3-hmx")
        self.assertTrue(result["complete"])
        self.assertEqual(result["stages"][0]["total_cycles"], 7)


if __name__ == "__main__":
    unittest.main()
