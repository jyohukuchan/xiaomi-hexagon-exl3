"""Memory-bounded port of the upstream EXL3 Hessian-metric scale refit."""

import torch


def refit_scales(weight, weight_q, H, su, sv, rounds=2, chunk=16384, damping=0.0):
    device = weight_q.device
    H = H.to(device)
    H_error = H
    if damping > 0:
        H_error = H.clone()
        H_error.diagonal().sub_(damping * H.diagonal().mean().item())
    k, n = weight_q.shape
    Q = weight_q
    su = su.float().to(device).view(k, 1)
    sv = sv.float().to(device).view(1, n)
    HW = torch.empty((k, n), dtype=torch.float32, device=device)

    def cols(a, b):
        return weight[:, a:b].to(device=device, dtype=torch.float32)

    denominator = 0.0
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        original = cols(a, b)
        HW[:, a:b] = H @ original
        denominator += (original * (H_error @ original)).sum().item()

    def error():
        result = 0.0
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            delta = cols(a, b) - Q[:, a:b]
            result += (delta * (H_error @ delta)).sum().item()
        return result / max(denominator, 1e-30)

    before = error()
    for _ in range(rounds):
        num = torch.zeros(n, device=device)
        den = torch.zeros(n, device=device)
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            block = Q[:, a:b]
            num[a:b] = (block * HW[:, a:b]).sum(0)
            den[a:b] = (block * (H @ block)).sum(0)
        c = torch.where(den > 1e-30, num / den.clamp(min=1e-30), torch.ones_like(num))
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            Q[:, a:b].mul_(c[None, a:b])
        sv.mul_(c[None, :])
        A = torch.zeros((k, k), device=device)
        rhs = torch.zeros(k, device=device)
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            block = Q[:, a:b]
            A.addmm_(block, block.T)
            rhs.add_((block * HW[:, a:b]).sum(1))
        A.mul_(H)
        A.diagonal().add_(1e-6 * A.diagonal().mean())
        r = torch.linalg.solve(A, rhs)
        r = torch.where(torch.isfinite(r) & (r > 0), r, torch.ones_like(r))
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            Q[:, a:b].mul_(r[:, None])
        su.mul_(r[:, None])
    return Q, su, sv, before, error()
