from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from check_context_probe import audit_placement, validate_report


class ContextProbeTest(unittest.TestCase):
    def report(self):
        return {"passed": True, "depth": 512, "decode_steps": 2, "context_capacity": 768, "vocabulary": 4,
                "batch": 512, "checked_logit_rows": 3, "checked_logit_values": 12, "hybrid_position_min": 513,
                "hybrid_position_max": 513, "kv_requested": "f16", "decoded_token_ids": [0, 1],
                "prefill_ms_including_checks": 1, "decode_ms_including_checks": 100, "diagnostic_decode_tokens_per_second": 20,
                "logit_min": 1, "logit_max": 4}

    def test_report(self):
        self.assertTrue(validate_report(self.report(), [1, 2, 3, 4])["passed"])

    def test_nonfinite_and_wrong_size_fail(self):
        for logits in ([1, 2, 3], [1, 2, 3, float("nan")], [1, 2, 3, float("inf")]):
            with self.assertRaises(ValueError):
                validate_report(self.report(), logits)
        with self.assertRaises(ValueError):
            validate_report([], [1, 2, 3, 4])

    def test_metadata_mismatches_fail(self):
        for key, value in (("passed", False), ("depth", True), ("checked_logit_rows", 2), ("context_capacity", 512),
                           ("hybrid_position_max", 512), ("kv_requested", "q8_0"), ("decoded_token_ids", [4, 0]),
                           ("diagnostic_decode_tokens_per_second", 30)):
            report = self.report()
            report[key] = value
            with self.assertRaises(ValueError):
                validate_report(report, [1, 2, 3, 4])

    def profile(self, skip=None):
        lines = [f"ggml-hex: HTP0 profile-op MUL_MAT|w{i}.weight x a -> b|exl3-hmx vtcm 1|" for i in range(151) if i != skip]
        lines += ["ggml-hex: HTP0 profile-op FLASH_ATTN_EXT|a -> b|" for _ in range(6)]
        lines += ["ggml-hex: HTP0 profile-op GATED_DELTA_NET+CPY|a -> b|" for _ in range(18)]
        return "\n".join(lines)

    def test_all_extension_steps_are_audited(self):
        reference = self.profile()
        log = "llama_kv_cache: K (f16): 1 V (f16): 1\nPREFILL checked=512/512 position=511\n" + reference
        log += "\nDECODE checked=1/2 position=512\n" + reference + "\nDECODE checked=2/2 position=513\n"
        self.assertEqual(audit_placement(log, reference, 512, 2)["audited_decode_steps"], 2)
        self.assertEqual(audit_placement(log, reference.replace("GATED_DELTA_NET+CPY", "GATED_DELTA_NET"), 512, 2)["gdn_per_step"], 18)
        with self.assertRaises(ValueError):
            audit_placement(log.replace(reference, self.profile(skip=5), 1), reference, 512, 2)
        with self.assertRaises(ValueError):
            audit_placement(log.replace("DECODE checked=2/2 position=513", "missing"), reference, 512, 2)
        with self.assertRaises(ValueError):
            audit_placement(log.replace("DECODE checked=2/2 position=513", "DECODE checked=2/2 position=5139"), reference, 512, 2)


if __name__ == "__main__":
    unittest.main()
