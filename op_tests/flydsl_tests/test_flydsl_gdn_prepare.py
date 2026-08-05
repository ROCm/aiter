# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the FlyDSL GDN prepare kernel (K1..K4 fused).

The kernel under test is ``gdn_prepare_fwd_flydsl``, which computes in one
launch what the Triton ``fused_chunk_local_cumsum_scaled_dot_kkt_fwd`` +
``fused_solve_tril_recompute_w_u`` pair computes in three dispatches. It is
checked against two oracles:

* ``_gdn_prepare_reference`` -- the pure-PyTorch algebraic spec in this file
  (fp32 triangular solve, bf16-rounded WY operands to mimic the MFMA GEMMs).
* the Triton K1+K2 pair itself, compared directly: the FlyDSL kernel is meant
  to be a drop-in, so any layout or exponent-domain shim here would defeat the
  purpose of the check.

Usage:
    HIP_VISIBLE_DEVICES=7 pytest -sv op_tests/flydsl_tests/test_flydsl_gdn_prepare.py
"""

from __future__ import annotations

import math

import pytest
import torch

from aiter.ops.flydsl.utils import is_flydsl_available

if not torch.cuda.is_available():
    pytest.skip("ROCm not available. Skipping GPU tests.", allow_module_level=True)
if not is_flydsl_available():
    pytest.skip(
        "flydsl is not installed. Skipping FlyDSL GDN prepare tests.",
        allow_module_level=True,
    )

try:
    from aiter.ops.flydsl.linear_attention_prefill_kernels import gdn_prepare_fwd_flydsl
except ImportError as exc:
    pytest.skip(
        f"Unable to import the FlyDSL GDN prepare kernel: {exc}",
        allow_module_level=True,
    )

try:
    from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.fused_cumsum_kkt import (
        fused_chunk_local_cumsum_scaled_dot_kkt_fwd,
    )
    from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.fused_solve_tril_recompute import (
        fused_solve_tril_recompute_w_u,
    )

    _HAS_TRITON_K12 = True
except Exception:  # noqa: BLE001
    _HAS_TRITON_K12 = False


# g_cumsum leaves the in-chunk prefix sum untouched, so it must stay tight;
# w_bar / u_bar go through bf16 MFMA GEMMs and a bf16 Neumann-squaring inverse.
ATOL_G = 1e-3
ATOL_WU = 5e-2


# -- Oracle ----------------------------------------------------------------


def _seq_bounds(cu_seqlens, B, T):
    """Yield (bos, seqlen) per sequence (dense -> one per batch row)."""
    if cu_seqlens is None:
        for b in range(B):
            yield b * T, T
    else:
        cu = cu_seqlens.tolist()
        for i_n in range(len(cu) - 1):
            yield cu[i_n], cu[i_n + 1] - cu[i_n]


@torch.no_grad()
def _gdn_prepare_reference(
    k: torch.Tensor,  # [B, T, Hg, K]
    v: torch.Tensor,  # [B, T, H, V]
    g: torch.Tensor,  # [B, T, H]
    beta: torch.Tensor,  # [B, T, H]
    cu_seqlens: torch.Tensor | None = None,
    BT: int = 64,
    Hg: int | None = None,
    use_exp2: bool = True,
):
    """Algebraic spec of the prepare stage, looping over sequence x chunk.

    With ``C = (I + A)^-1`` and ``A`` the strictly lower-triangular gated KKT
    matrix, returns ``w_bar = C @ (k * beta * exp(g_cumsum))``,
    ``u_bar = C @ (v * beta)`` and the in-chunk inclusive ``g_cumsum``, all
    head-major like the kernel.

    The inverse is always taken in fp32; the WY operands are rounded to bf16
    to mimic the kernel's MFMA bf16 GEMMs.
    """
    B, T, Hg_in, K = k.shape
    H, V = v.shape[2], v.shape[3]
    if Hg is None:
        Hg = Hg_in
    assert H % Hg == 0, f"H={H} must be divisible by Hg={Hg}"
    rep = H // Hg  # value heads per kv head

    dev = k.device
    kf, vf, gf, bf = k.float(), v.float(), g.float(), beta.float()

    w_bar = torch.zeros(B, T, H, K, dtype=torch.float32, device=dev)
    u_bar = torch.zeros(B, T, H, V, dtype=torch.float32, device=dev)
    g_cumsum = torch.zeros(B, T, H, dtype=torch.float32, device=dev)

    def _bt(bos, t_local):
        # Map a (bos + t_local) flat token position back to (b, t) coords.
        # dense: bos = b*T; varlen: B == 1 so the batch index is always 0.
        if cu_seqlens is None:
            return bos // T, t_local
        return 0, bos + t_local

    eye = torch.eye(BT, dtype=torch.float32, device=dev)

    for bos, seqlen in _seq_bounds(cu_seqlens, B, T):
        for i_t in range((seqlen + BT - 1) // BT):
            t0 = i_t * BT
            L = min(BT, seqlen - t0)
            rows = [_bt(bos, t0 + j) for j in range(L)]
            bb = torch.tensor([r[0] for r in rows], device=dev)
            tt = torch.tensor([r[1] for r in rows], device=dev)

            gc = torch.cumsum(gf[bb, tt], dim=0)  # [L, H] inclusive
            g_cumsum[bb, tt] = gc

            # Expand kv heads to value heads, then batch over heads.
            kh = kf[bb, tt].repeat_interleave(rep, dim=1).permute(1, 0, 2)  # [H,L,K]
            KKT = torch.bmm(kh, kh.transpose(1, 2))  # [H, L, L]

            gc_h = gc.permute(1, 0)  # [H, L]
            beta_h = bf[bb, tt].permute(1, 0)  # [H, L]
            decay = torch.exp(gc_h[:, :, None] - gc_h[:, None, :])
            A = KKT * beta_h[:, :, None] * decay
            strict_tril = torch.tril(torch.ones(L, L, device=dev), diagonal=-1).bool()
            A = A * strict_tril[None]

            IpA = eye[:L, :L][None] + A
            C = torch.linalg.solve(IpA, eye[:L, :L][None].expand(H, L, L))
            C = C.to(torch.bfloat16).float()

            kbeta = kh * beta_h[:, :, None] * torch.exp(gc_h)[:, :, None]
            vbeta = vf[bb, tt].permute(1, 0, 2) * beta_h[:, :, None]
            kbeta = kbeta.to(torch.bfloat16).float()
            vbeta = vbeta.to(torch.bfloat16).float()

            w_bar[bb, tt] = torch.bmm(C, kbeta).permute(1, 0, 2)
            u_bar[bb, tt] = torch.bmm(C, vbeta).permute(1, 0, 2)

    if use_exp2:
        g_cumsum = g_cumsum * math.log2(math.e)
    # Accumulated token-major above for readable scatter indexing; the kernel
    # publishes head-major, so transpose on the way out.
    return (
        w_bar.transpose(1, 2).contiguous().to(torch.bfloat16),
        u_bar.transpose(1, 2).contiguous().to(torch.bfloat16),
        g_cumsum.transpose(1, 2).contiguous(),
    )


# -- Fixtures --------------------------------------------------------------


def _make_inputs(B, T, Hg, H, K, V, *, cu_list=None, seed=0, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)
    k = (
        torch.randn(B, T, Hg, K, dtype=torch.bfloat16, device=device, generator=gen)
        * 0.2
    )
    v = (
        torch.randn(B, T, H, V, dtype=torch.bfloat16, device=device, generator=gen)
        * 0.2
    )
    beta = torch.rand(
        B, T, H, dtype=torch.float32, device=device, generator=gen
    ).sigmoid()
    # Decay gates are negative, as produced by the GDN gating stage.
    g = -(
        torch.rand(B, T, H, dtype=torch.float32, device=device, generator=gen) * 0.5
        + 0.2
    )
    cu = None
    if cu_list is not None:
        cu = torch.tensor(cu_list, dtype=torch.int32, device=device)
    return k, v, g, beta, cu


# (tag, B, T, Hg, H, K, V, cu_list, seed). Sequence lengths that are not a
# multiple of BT=64 exercise the kernel's ragged-tail guards.
CASES = [
    ("dense_mha", 1, 256, 8, 8, 128, 128, None, 1),
    ("dense_mha_b2", 2, 192, 4, 4, 128, 128, None, 2),
    ("dense_gqa", 1, 256, 4, 16, 128, 128, None, 3),
    ("dense_ragged", 1, 300, 4, 8, 128, 128, None, 4),
    ("varlen_gqa", 1, 600, 4, 16, 128, 128, [0, 128, 300, 600], 5),
]
CASE_IDS = [c[0] for c in CASES]


def _max_abs(a, b):
    return (a.float() - b.float()).abs().max().item()


@pytest.mark.parametrize("use_exp2", [True, False], ids=["exp2", "natlog"])
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_gdn_prepare_matches_reference(case, use_exp2):
    _, B, T, Hg, H, K, V, cu_list, seed = case
    k, v, g, beta, cu = _make_inputs(B, T, Hg, H, K, V, cu_list=cu_list, seed=seed)

    w_f, u_f, gc_f = gdn_prepare_fwd_flydsl(
        k, v, g, beta, cu_seqlens=cu, Hg=Hg, use_exp2=use_exp2
    )
    w_r, u_r, gc_r = _gdn_prepare_reference(
        k, v, g, beta, cu_seqlens=cu, Hg=Hg, use_exp2=use_exp2
    )

    assert w_f.shape == (B, H, T, K) and w_f.dtype == torch.bfloat16
    assert u_f.shape == (B, H, T, V) and u_f.dtype == torch.bfloat16
    assert gc_f.shape == (B, H, T) and gc_f.dtype == torch.float32
    assert w_f.is_contiguous() and u_f.is_contiguous() and gc_f.is_contiguous()

    assert _max_abs(gc_f, gc_r) < ATOL_G
    assert _max_abs(w_f, w_r) < ATOL_WU
    assert _max_abs(u_f, u_r) < ATOL_WU


@pytest.mark.skipif(not _HAS_TRITON_K12, reason="Triton K1+K2 kernels unavailable")
@pytest.mark.parametrize("use_exp2", [True, False], ids=["exp2", "natlog"])
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_gdn_prepare_matches_triton_k1_k2(case, use_exp2):
    """The fused kernel is a drop-in for the Triton K1+K2 pair.

    Compared with no reshaping and no rescaling on either side: matching the
    Triton head-major layout and its ``use_exp2`` exponent domain is the whole
    point, so a shim here would hide exactly the bug this test is for.
    """
    _, B, T, Hg, H, K, V, cu_list, seed = case
    k, v, g, beta, cu = _make_inputs(B, T, Hg, H, K, V, cu_list=cu_list, seed=seed)
    cu_long = None if cu is None else cu.long()

    w_f, u_f, gc_f = gdn_prepare_fwd_flydsl(
        k, v, g, beta, cu_seqlens=cu, Hg=Hg, use_exp2=use_exp2
    )

    gc_t, A_raw = fused_chunk_local_cumsum_scaled_dot_kkt_fwd(
        k, beta, g, cu_seqlens=cu_long, chunk_size=64, use_exp2=use_exp2
    )
    w_t, u_t = fused_solve_tril_recompute_w_u(
        A_raw, k, v, beta, gc_t, cu_seqlens=cu_long, use_exp2=use_exp2
    )

    assert w_f.shape == w_t.shape and u_f.shape == u_t.shape
    assert gc_f.shape == gc_t.shape
    assert _max_abs(gc_f, gc_t) < ATOL_G
    assert _max_abs(w_f, w_t) < ATOL_WU
    assert _max_abs(u_f, u_t) < ATOL_WU
