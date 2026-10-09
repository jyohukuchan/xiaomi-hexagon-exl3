import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from inspect_model import audit, read_headers
from package_model import validate_inventory


class ModelAuditTest(unittest.TestCase):
    def test_deployment_rejects_missing_layer(self):
        reference = {"model.language_model.layers.0.mlp.up_proj.weight": {"shape": [128, 128]}}
        with self.assertRaisesRegex(ValueError, "Missing text-model tensor"):
            validate_inventory({}, reference)

    def test_deployment_rejects_missing_scales(self):
        reference = {"model.language_model.layers.0.mlp.up_proj.weight": {"shape": [128, 128]}}
        selected = {"model.language_model.layers.0.mlp.up_proj.trellis": {"shape": [8, 8, 64], "dtype": "I16"}}
        with self.assertRaisesRegex(ValueError, "scales"):
            validate_inventory(selected, reference)

    def write(self, directory, entries, payload):
        header = json.dumps(entries).encode()
        (directory / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + bytes(payload))

    def test_reject_overlapping_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.write(directory, {"a": {"dtype": "I16", "shape": [2], "data_offsets": [0, 4]},
                                   "b": {"dtype": "I16", "shape": [2], "data_offsets": [2, 6]}}, 6)
            with self.assertRaisesRegex(ValueError, "overlapping"):
                read_headers(directory)

    def test_reject_truncated_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.write(directory, {"a": {"dtype": "F16", "shape": [4], "data_offsets": [0, 8]}}, 4)
            with self.assertRaisesRegex(ValueError, "span"):
                read_headers(directory)

    def test_budget_uses_reference_and_excludes_vision(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, quant = Path(tmp) / "source", Path(tmp) / "quant"
            source.mkdir(); quant.mkdir()
            self.write(source, {"model.language_model.w.weight": {"dtype": "F16", "shape": [16, 16], "data_offsets": [0, 512]},
                                "model.visual.w.weight": {"dtype": "F16", "shape": [16, 16], "data_offsets": [512, 1024]}}, 1024)
            self.write(quant, {"model.language_model.w.trellis": {"dtype": "I16", "shape": [1, 1, 64], "data_offsets": [0, 128]},
                               "model.visual.w.weight": {"dtype": "F16", "shape": [16, 16], "data_offsets": [128, 640]}}, 640)
            text = audit(quant, source, "text")
            self.assertEqual(text["effective_bpw"], 4)
            self.assertTrue(text["within_5_bpw"])
            self.assertEqual(text["exl3_matrices"][0]["bits"], 4)
            self.assertEqual(text["all_payload_bytes"], 640)
            self.assertFalse(audit(quant, source)["within_5_bpw"])


if __name__ == "__main__":
    unittest.main()
