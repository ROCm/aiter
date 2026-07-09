# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Correctness test for gdn_chunk_prepare (the fused intra-chunk GDN prefill prep).
# Reference = the four Triton FLA kernels it fuses:
#   chunk_local_cumsum + chunk_scaled_dot_kkt_fwd + solve_tril + recompute_w_u_fwd

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.gdn_chunk_prepare import gdn_chunk_prepare
from aiter.ops.triton._triton_kernels.gated_delta_rule.utils import (
    chunk_local_cumsum,
    chunk_scaled_dot_kkt_fwd,
    solve_tril,
    recompute_w_u_fwd,
)


def _ref_triton(k, v, g, beta, BT=64):
    g_cs = chunk_local_cumsum(g, chunk_size=BT, cu_seqlens=None)
    A = chunk_scaled_dot_kkt_fwd(
        k=k, g=g_cs, beta=beta, cu_seqlens=None, output_dtype=torch.float32
    )
    A = solve_tril(A=A, cu_seqlens=None, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g=g_cs, cu_seqlens=None)
    return g_cs, w, u


def _ref_fp32(k, v, g, beta, BT=64):
    """Exact chunk-prepare math in fp32 — ground truth (matches the kernel's
    convention: A = tril(KKᵀ ⊙ β·decay, -1); C = (I+A)⁻¹; u = C(vβ); w = C(kβe^g))."""
    B, T, H, K = k.shape
    V = v.shape[-1]
    kf, vf = k.float(), v.float()
    gf, bf = g.float(), beta.float()
    g_cs = torch.empty(B, T, H, dtype=torch.float32, device=k.device)
    w = torch.empty(B, T, H, K, dtype=torch.float32, device=k.device)
    u = torch.empty(B, T, H, V, dtype=torch.float32, device=k.device)
    for c0 in range(0, T, BT):
        c1 = min(c0 + BT, T)
        L = c1 - c0
        gc = gf[:, c0:c1].cumsum(dim=1)                      # [B,L,H]
        g_cs[:, c0:c1] = gc
        kb = kf[:, c0:c1].permute(0, 2, 1, 3)               # [B,H,L,K]
        vb = vf[:, c0:c1].permute(0, 2, 1, 3)               # [B,H,L,V]
        bb = bf[:, c0:c1].permute(0, 2, 1)                  # [B,H,L]
        gch = gc.permute(0, 2, 1)                           # [B,H,L]
        kk = torch.matmul(kb, kb.transpose(-1, -2))         # [B,H,L,L]
        decay = torch.exp(gch.unsqueeze(-1) - gch.unsqueeze(-2))  # [B,H,L,L]
        A = kk * decay * bb.unsqueeze(-1)                   # scale rows by beta[s]
        A = torch.tril(A, diagonal=-1)
        eye = torch.eye(L, device=k.device, dtype=torch.float32)
        C = torch.linalg.inv(eye + A)                       # [B,H,L,L]
        u_blk = torch.matmul(C, vb * bb.unsqueeze(-1))      # [B,H,L,V]
        w_blk = torch.matmul(C, kb * (bb * torch.exp(gch)).unsqueeze(-1))
        u[:, c0:c1] = u_blk.permute(0, 2, 1, 3)
        w[:, c0:c1] = w_blk.permute(0, 2, 1, 3)
    return g_cs, w, u


def _rel_l2(a, b):
    a, b = a.float(), b.float()
    return (torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-12)).item()


def _make(B, T, H, K=128, V=128, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    q = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device=dev), p=2, dim=-1)
    k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device=dev), p=2, dim=-1)
    v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device=dev) * 0.5
    g = F.logsigmoid(torch.randn(B, T, H, dtype=torch.float32, device=dev))
    beta = torch.rand(B, T, H, dtype=torch.float32, device=dev)
    return q, k, v, g, beta


def _report(name, a, b):
    a = a.float()
    b = b.float()
    diff = (a - b).abs()
    denom = b.abs().clamp_min(1e-4)
    print(
        f"  {name:9}: max_abs={diff.max().item():.4e} "
        f"mean_abs={diff.mean().item():.4e} "
        f"max_rel={(diff/denom).max().item():.4e}"
    )
    return diff.max().item()


@pytest.mark.parametrize("B,T,H", [
    (1, 1024, 4), (1, 2048, 16), (1, 4096, 8), (1, 8192, 32),
    (1, 16384, 16), (4, 2048, 8), (1, 1000, 4),  # last: non-multiple of BT
])
def test_gdn_chunk_prepare(B, T, H):
    q, k, v, g, beta = _make(B, T, H)
    gR, wR, uR = _ref_fp32(k, v, g, beta)            # fp32 ground truth
    gT, wT, uT = _ref_triton(k, v, g, beta)          # Triton (bf16) baseline
    w, u, g_cs = gdn_chunk_prepare(k, v, g, beta, BT=64)

    # rel-L2 vs fp32 ground truth
    k_g, k_w, k_u = _rel_l2(g_cs, gR), _rel_l2(w, wR), _rel_l2(u, uR)
    t_g, t_w, t_u = _rel_l2(gT, gR), _rel_l2(wT, wR), _rel_l2(uT, uR)
    print(f"\n[B{B} T{T} H{H}]  rel-L2 vs fp32 (kernel | triton)")
    print(f"  g_cumsum: {k_g:.3e} | {t_g:.3e}")
    print(f"  w_bar   : {k_w:.3e} | {t_w:.3e}")
    print(f"  u_bar   : {k_u:.3e} | {t_u:.3e}")

    # g_cumsum is fp32 -> should be ~exact
    assert k_g < 1e-5, f"g_cumsum rel-L2 {k_g}"
    # w_bar/u_bar are bf16 -> require the kernel be at least as accurate as Triton
    # (small slack) against fp32 ground truth.
    assert k_w <= max(1.5 * t_w, 2e-2), f"w_bar rel-L2 {k_w} vs triton {t_w}"
    assert k_u <= max(1.5 * t_u, 2e-2), f"u_bar rel-L2 {k_u} vs triton {t_u}"


if __name__ == "__main__":
    import sys
    cfgs = [(1, 1024, 4), (1, 2048, 16), (1, 4096, 8), (1, 8192, 32),
            (1, 16384, 16), (4, 2048, 8), (1, 1000, 4)]
    fail = 0
    for (B, T, H) in cfgs:
        try:
            test_gdn_chunk_prepare(B, T, H)
        except AssertionError as e:
            print("  FAIL:", e); fail += 1
    print("\n" + ("OK" if fail == 0 else f"{fail} FAILED"))
    sys.exit(1 if fail else 0)
