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
