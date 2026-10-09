"""Build portable decoder fixtures using the pinned upstream CUDA implementation."""

import argparse
import json
import struct
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, help="Build only the upstream reconstruction kernel")
    parser.add_argument("--model-file", type=Path)
    parser.add_argument("--tensor-prefix")
    parser.add_argument("--first-column", type=int, default=0)
    parser.add_argument("--column-count", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.upstream_dir:
        from torch.utils.cpp_extension import load
        ext = load(name="exl3_reconstruct_oracle",
                   sources=[str(Path(__file__).with_name("cuda_oracle.cpp")),
                            str(args.upstream_dir / "exllamav3/exllamav3_ext/quant/reconstruct.cu")],
                   extra_cuda_cflags=["-O3", "-use_fast_math"], verbose=True)
    else:
        from exllamav3.ext import exllamav3_ext as ext
    torch.cuda.set_device(0)
    h = torch.ones((1, 1), device="cuda", dtype=torch.float32)
    while h.shape[0] < 128:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    h /= 128**0.5
    results = []
    if args.model_file:
        if not args.tensor_prefix:
            parser.error("--model-file requires --tensor-prefix")
        from safetensors import safe_open
        with safe_open(args.model_file, framework="pt", device="cpu") as source:
            prefix = args.tensor_prefix
            full = source.get_tensor(prefix + ".trellis")
            first = args.first_column
            count = args.column_count or full.shape[1] * 16
            if first < 0 or first % 128 or count <= 0 or count % 128 or first + count > full.shape[1] * 16:
                parser.error("The selected column range must align to H128 groups")
            packed = full[:, first // 16:(first + count) // 16, :].contiguous().cuda()
            suh = source.get_tensor(prefix + ".suh").cuda()
            svh = source.get_tensor(prefix + ".svh")[first:first + count].contiguous().cuda()
            cb = 2 if prefix + ".mul1" in source.keys() else 1 if prefix + ".mcg" in source.keys() else 0
        k, n = packed.shape[0] * 16, packed.shape[1] * 16
        bits = packed.shape[2] // 16
        if packed.shape[2] != bits * 16 or not 1 <= bits <= 8 or k > 16384 or n > 16384:
            parser.error("The selected fixture is too large or has an unsupported bitrate")
        inner = torch.empty((k, n), device="cuda", dtype=torch.float16)
        ext.reconstruct(inner, packed, bits, cb == 1, cb == 2)
        original = torch.einsum("ij,bjn->bin", h, inner.float().view(k // 128, 128, n)).reshape(k, n)
        original = (original.view(k, n // 128, 128) @ h).reshape(k, n)
        original = (original * suh.float()[:, None] * svh.float()[None, :]).half()
        output = args.output / "model.bin"
        with output.open("wb") as stream:
            stream.write(b"EXL3REF1" + struct.pack("<4I", k, n, bits, cb))
            for tensor in (packed, suh, svh, inner, original):
                stream.write(tensor.cpu().contiguous().numpy().tobytes())
        results.append({"file": output.name, "k": k, "n": n, "bits": bits, "codebook": cb,
                        "source_tensor": prefix, "first_column": first})
        print(f"Generated real-model fixture: {prefix}", flush=True)
        (args.output / "manifest.json").write_text(json.dumps({"upstream_commit": "151539c77abc7ab7425d30da7a4e8e3c5c154e7b", "fixtures": results}, indent=2) + "\n")
        return
    for bits in range(1, 9):
        for cb in range(3):
            torch.manual_seed(bits * 100 + cb)
            k, n = 256, 384
            packed = torch.randint(0, 65536, (k // 16, n // 16, bits * 16), device="cuda", dtype=torch.int32).short()
            suh = ((torch.rand(k, device="cuda") - 0.5) * 2).half()
            svh = ((torch.rand(n, device="cuda") - 0.5) * 2).half()
            inner = torch.empty((k, n), device="cuda", dtype=torch.float16)
            ext.reconstruct(inner, packed, bits, cb == 1, cb == 2)
            original = torch.einsum("ij,bjn->bin", h, inner.float().view(k // 128, 128, n)).reshape(k, n)
            original = (original.view(k, n // 128, 128) @ h).reshape(k, n)
            original = (original * suh.float()[:, None] * svh.float()[None, :]).half()
            output = args.output / f"k{bits}-cb{cb}.bin"
            with output.open("wb") as stream:
                stream.write(b"EXL3REF1" + struct.pack("<4I", k, n, bits, cb))
                for tensor in (packed, suh, svh, inner, original):
                    stream.write(tensor.cpu().contiguous().numpy().tobytes())
            results.append({"file": output.name, "k": k, "n": n, "bits": bits, "codebook": cb})
            print(f"Generated {output.name}", flush=True)
    manifest = {"upstream_commit": "151539c77abc7ab7425d30da7a4e8e3c5c154e7b", "torch": torch.__version__,
                "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0), "fixtures": results}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
