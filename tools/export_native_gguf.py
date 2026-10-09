"""Losslessly repack EXL3 trellis/scales into the project's 128x128 runtime format."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LLAMA = ROOT / "third_party/llama.cpp"
sys.path.insert(0, str(LLAMA / "gguf-py"))
import gguf
from inspect_model import read_headers


def tensor_array(directory, info):
    dtypes = {"F16": "<f2", "BF16": "<u2", "F32": "<f4", "I16": "<u2", "I32": "<i4"}
    return np.memmap(directory / info["file"], dtype=dtypes[info["dtype"]], mode="r",
                     offset=info["offset"], shape=tuple(info["shape"]))


def pack_matrix(packed, su, sv, bits):
    k, n = packed.shape[0] * 16, packed.shape[1] * 16
    if k % 128 or n % 128 or bits not in (4, 6, 8) or packed.shape[2] != 16 * bits:
        raise ValueError("Unsupported runtime matrix geometry or bitrate")
    words = 1024 * bits + 256
    result = np.empty((n // 128, k // 128, words), dtype="<u2")
    result[:, :, :1024 * bits] = packed.reshape(k // 128, 8, n // 128, 8, 16 * bits).transpose(2, 0, 1, 3, 4).reshape(n // 128, k // 128, -1)
    result[:, :, 1024 * bits:1024 * bits + 128] = np.asarray(su).view("<u2").reshape(1, k // 128, 128)
    result[:, :, 1024 * bits + 128:] = np.asarray(sv).view("<u2").reshape(n // 128, 1, 128)
    return result.view(np.uint8).reshape(n, (k // 128) * (16 * bits + 4))


def metadata_file(model, config, destination):
    with tempfile.TemporaryDirectory(prefix="exl3-metadata-") as temporary:
        directory = Path(temporary)
        clean = dict(config)
        clean.pop("quantization_config", None)
        (directory / "config.json").write_text(json.dumps(clean), encoding="utf-8")
        for filename in ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "merges.txt", "vocab.json"):
            if (model / filename).exists():
                shutil.copyfile(model / filename, directory / filename)
        os.symlink((model / "model.safetensors").resolve(), directory / "model.safetensors")
        subprocess.run([sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), str(directory),
                        "--vocab-only", "--no-mtp", "--outfile", str(destination), "--outtype", "f16"], check=True)


def export(model, output):
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((model / "hexagon-exl3.json").read_text(encoding="utf-8"))
    parameters = int(manifest["reference_parameter_count"])
    if parameters <= 0:
        raise ValueError("Invalid reference parameter count")
    text = config["text_config"]
    if text["linear_num_key_heads"] != text["linear_num_value_heads"]:
        raise ValueError("This runtime exporter does not yet support GDN V-head reordering")
    tensors = read_headers(model)
    names = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN35, text["num_hidden_layers"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="exl3-gguf-") as temporary:
        metadata = Path(temporary) / "metadata.gguf"
        metadata_file(model, config, metadata)
        reader = gguf.GGUFReader(metadata)
        writer = gguf.GGUFWriter(output, "qwen35", use_temp_file=True)
        for key, field in reader.fields.items():
            if key.startswith("GGUF.") or key in ("general.architecture", "general.file_type"):
                continue
            writer.add_key_value(key, field.contents(), field.types[0], field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None)
        writer.add_file_type(gguf.LlamaFileType.MOSTLY_EXL3_4)
        writer.add_uint32("hexagon.exl3.version", 1)
        writer.add_uint64("hexagon.exl3.reference_parameters", parameters)
        quant_bytes = 0
        raw_bytes = 0
        for key, info in sorted(tensors.items()):
            if key.endswith(".trellis"):
                prefix = key.removesuffix(".trellis")
                multiplier = tensor_array(model, tensors[prefix + ".mul1"]).view("<u4").item()
                if multiplier != 0x83DCD12D:
                    raise ValueError("The runtime format currently requires the mul1 codebook")
                bits = info["shape"][2] // 16
                data = pack_matrix(tensor_array(model, info), tensor_array(model, tensors[prefix + ".suh"]),
                                   tensor_array(model, tensors[prefix + ".svh"]), bits)
                name = "token_embd.weight" if prefix == "lm_head" else names.get_name(prefix.replace("model.language_model.", "model.") + ".weight", try_suffixes=(".weight",))
                if not name:
                    raise ValueError(f"No graph mapping for {prefix}")
                dtype = {4: gguf.GGMLQuantizationType.EXL3_4, 6: gguf.GGMLQuantizationType.EXL3_6, 8: gguf.GGMLQuantizationType.EXL3_8}[bits]
                writer.add_tensor(name, data, raw_dtype=dtype)
                quant_bytes += data.nbytes
                print(f"Packed {name} bits={bits} bytes={data.nbytes}", flush=True)
                continue
            if key.endswith((".suh", ".svh", ".mul1", ".mcg")):
                continue
            data = tensor_array(model, info)
            if info["dtype"] == "BF16":
                data = (data.astype(np.uint32) << 16).view(np.float32)
            else:
                data = np.asarray(data, dtype=np.float32)
            name_source = key.replace("model.language_model.", "model.")
            if key.endswith(".A_log"):
                data = -np.exp(data)
            elif key.endswith(".dt_bias"):
                name_source = name_source.removesuffix(".dt_bias") + ".dt_proj.bias"
            elif "conv1d" in key:
                data = data.squeeze()
            elif key.endswith("norm.weight") and not key.endswith("linear_attn.norm.weight"):
                data = data + 1.0
            name = names.get_name(name_source, try_suffixes=(".weight", ".bias"))
            if not name:
                raise ValueError(f"No raw tensor graph mapping for {key}")
            if data.ndim > 1 and "conv1d" not in key:
                data = data.astype(np.float16)
            writer.add_tensor(name, np.ascontiguousarray(data))
            raw_bytes += data.nbytes
        bpw = (quant_bytes + raw_bytes) * 8 / parameters
        if bpw > 5:
            raise ValueError(f"Runtime payload exceeds 5 bpw: {bpw}")
        writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
        print(json.dumps({"runtime_payload_bytes": quant_bytes + raw_bytes, "effective_bpw": bpw}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.model, args.output)
