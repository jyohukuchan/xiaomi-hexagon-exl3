"""Check the memory-bounded adapter against the pinned upstream implementation."""

import sys
from pathlib import Path

import torch
from exllamav3.modules.quant.exl3_lib.quantize import refit_scales as upstream_refit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from refit_scales import refit_scales

torch.manual_seed(137)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
for original_on_cpu in (False, True):
    for damping in (0.0, 0.025):
        weight = torch.randn(64, 257, device="cuda")
        quant = weight + torch.randn_like(weight) * 0.1
        x = torch.randn(256, 64, device="cuda")
        H = x.T @ x / 256 + torch.eye(64, device="cuda") * 0.1
        su = torch.ones(64, device="cuda")
        sv = torch.ones(257, device="cuda")
        original = weight.cpu() if original_on_cpu else weight
        expected = upstream_refit(original, quant.clone(), H.clone(), su.clone(), sv.clone(), chunk=64, damping=damping)
        actual = refit_scales(original, quant.clone(), H.clone(), su.clone(), sv.clone(), chunk=64, damping=damping)
        for a, b in zip(actual[:3], expected[:3]):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)
        for a, b in zip(actual[3:], expected[3:]):
            assert abs(a - b) < 1e-7, (a, b)
        print(f"PASS refit original_on_cpu={original_on_cpu} damping={damping}", flush=True)
