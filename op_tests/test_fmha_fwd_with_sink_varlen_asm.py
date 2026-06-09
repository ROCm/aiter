# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness + perf tests for fmha_fwd_with_sink_varlen_asm (BF16 ASM, gfx1250).

Ops layer:  aiter.fmha_fwd_with_sink_varlen_asm  (low-level, packed/varlen)

Layout (packed THD; batch folded into the token axis):
    q   : (total_q, nheads,   hdim_q)
    k   : (total_k, nheads_k, hdim_q)
    v   : (total_k, nheads_k, hdim_v)
    out : (total_q, nheads,   hdim_v)
    lse : (total_q, nheads, 1)  fp32
    cu_seqlens_q / cu_seqlens_k : int32 [batch+1] cumulative (cu[batch] == total)

Sink convention (same as the fixed-batch path / CK attention_ref):
    `sink` ([q_head_num] fp32) is a per-Q-head logit in the SAME scaled domain
    as Q·K^T * softmax_scale; it acts as a zero-value virtual KV column.  Passed
    to the kernel verbatim (no host-side scaling).  D64 kernels read it; D128
    kernels ignore it (pass None).

Only causal kernels are shipped (CSV registers mask=1 rows), so is_causal=True.
Causal uses bottom-right alignment per sequence (query i attends to key j iff
j <= i + (sk - sq)), matching flash_attn varlen semantics.
"""

from __future__ import annotations

import math
from typing import List, Optional

import pytest
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx


def _is_gfx1250_host() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return get_gfx() == "gfx1250"
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _is_gfx1250_host(),
    reason=(
        "fmha_fwd_with_sink_varlen_asm ASM kernels are only shipped for gfx1250 "
        "(hsa/gfx1250/fmha_fwd_bf16_varlen/*.co); no GPU or a different arch — skip"
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cmp(a: torch.Tensor, b: torch.Tensor, *, rtol=1e-2, atol=1e-2, msg: str = ""):
    """fp32-on-CPU compare that hard-fails on mismatch / NaN.

    Cast to fp32 CPU first to avoid the gfx1250 + ROCm bf16 element-wise hang
    that can occur right after a custom ASM kernel launch.
    """
    a32 = a.detach().float().cpu()
    b32 = b.detach().float().cpu()
    torch.testing.assert_close(a32, b32, rtol=rtol, atol=atol, msg=msg)


def _d64_sink(hq: int, device: str) -> torch.Tensor:
    """Per-head sink logits (scaled domain), varied across heads."""
    return torch.linspace(0.5, 2.0, hq, dtype=torch.float32, device=device)


def _attn_one(q, k, v, *, is_causal: bool, sink: Optional[torch.Tensor]):
    """Single-sequence attention reference (no batch dim).

    q: (sq, hq, d)   k: (sk, hk, d)   v: (sk, hk, dv)
    returns out (sq, hq, dv), lse (sq, hq) in fp32.
    """
    sq, hq, d = q.shape
    sk, hk, _ = k.shape
    if hq != hk:
        k = k.repeat_interleave(hq // hk, dim=1)
        v = v.repeat_interleave(hq // hk, dim=1)
    qf, kf, vf = q.float(), k.float(), v.float()
    scale = 1.0 / math.sqrt(d)
    # scores: (hq, sq, sk) in the scaled-logit domain.
    scores = torch.einsum("qhd,khd->hqk", qf, kf) * scale
    if is_causal:
        row = torch.arange(sq, device=q.device)[:, None]
        col = torch.arange(sk, device=q.device)[None, :]
        # bottom-right aligned causal mask
        masked = col > (row + (sk - sq))
        scores = scores.masked_fill(masked[None], float("-inf"))
    max_attn = scores.max(dim=-1).values  # (hq, sq)
    if sink is not None:
        sink_hs = sink.float()[:, None].expand(hq, sq)
        max_total = torch.maximum(max_attn, sink_hs)
    else:
        max_total = max_attn
    denom = torch.exp(scores - max_total.unsqueeze(-1)).sum(dim=-1)  # (hq, sq)
    if sink is not None:
        denom = denom + torch.exp(sink_hs - max_total)
    probs = torch.exp(scores - max_total.unsqueeze(-1)) / denom.unsqueeze(-1)
    out = torch.einsum("hqk,khd->qhd", probs, vf).to(q.dtype)  # (sq, hq, dv)
    lse = (torch.log(denom) + max_total).transpose(0, 1)  # (sq, hq)
    return out, lse


def _ref_varlen(q, k, v, cu_q, cu_k, *, is_causal: bool, sink: Optional[torch.Tensor]):
    """Packed-THD reference: loop over batches, slice via cu_seqlens."""
    total_q, hq, _ = q.shape
    dv = v.shape[-1]
    batch = cu_q.numel() - 1
    out = torch.empty((total_q, hq, dv), dtype=q.dtype, device=q.device)
    lse = torch.empty((total_q, hq), dtype=torch.float32, device=q.device)
    cuq = cu_q.tolist()
    cuk = cu_k.tolist()
    for b in range(batch):
        q0, q1 = cuq[b], cuq[b + 1]
        k0, k1 = cuk[b], cuk[b + 1]
        if q1 == q0:
            continue
        ob, lb = _attn_one(q[q0:q1], k[k0:k1], v[k0:k1], is_causal=is_causal, sink=sink)
        out[q0:q1] = ob
        lse[q0:q1] = lb
    return out, lse


def make_varlen_packed(
    seqlens: List[int], hq: int, hk: int, d: int, dv: int, device="cuda", seed=0
):
    """Build packed THD q/k/v + cu_seqlens for the given per-batch seqlens.

    Uses equal q/k seqlens per batch (standard varlen self-attention).
    """
    torch.manual_seed(seed)
    cu = torch.tensor(
        [0] + list(torch.tensor(seqlens).cumsum(0).tolist()), dtype=torch.int32
    )
    total = int(cu[-1].item())
    q = torch.randn(total, hq, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(total, hk, d, dtype=torch.bfloat16, device=device)
    v = torch.randn(total, hk, dv, dtype=torch.bfloat16, device=device)
    cu = cu.to(device)
    return q, k, v, cu


# ---------------------------------------------------------------------------
# Kernel entry points (mirrors test_fmha_fwd_with_sink_asm.run_kernel).
# ---------------------------------------------------------------------------


def run_kernel(
    q,
    k,
    v,
    cu_q,
    cu_k,
    max_seqlen_q,
    *,
    scale: float,
    is_causal: bool,
    sink: Optional[torch.Tensor] = None,
    via: str = "ops",
):
    """Call the varlen kernel and return (out, lse) with lse shaped
    (total_q, nheads) to match the in-file `_ref_varlen` reference.

    via = "ops"     → low-level aiter.fmha_fwd_with_sink_varlen_asm
                      (lse is packed (total_q, nheads, 1))
    via = "public"  → public aiter.flash_attn_varlen_func (dispatcher → asm
                      path); the varlen API returns lse as (nheads, total_q).
    """
    if via == "ops":
        out, lse = aiter.fmha_fwd_with_sink_varlen_asm(
            q, k, v, cu_q, cu_k, max_seqlen_q, scale, is_causal, True, sink=sink
        )
        return out, lse.squeeze(-1)  # (total_q, nheads, 1) -> (total_q, nheads)
    if via == "public":
        # q/k seqlens are equal in these tests, so max_seqlen_k == max_seqlen_q.
        r = aiter.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            max_seqlen_q,
            max_seqlen_q,
            softmax_scale=scale,
            causal=is_causal,
            return_lse=True,
            sink_ptr=sink,
        )
        # public varlen lse is (nheads, total_q) -> (total_q, nheads)
        return r[0], r[1].transpose(0, 1).contiguous()
    raise ValueError(f"unknown via={via!r}")


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("is_causal", [True])
@pytest.mark.parametrize(
    "head_dim,hq,hk,seqlens",
    [
        # aligned single batch
        (64, 8, 1, [256]),
        (128, 8, 1, [256]),
        # multi-batch, mixed (some unaligned) seqlens
        (64, 8, 1, [128, 256, 384]),
        (128, 8, 1, [128, 256, 384]),
        (64, 8, 2, [100, 200, 300]),  # unaligned + GQA
        (128, 8, 2, [100, 200, 300]),
        # GQA-heavy, larger
        (64, 64, 8, [512, 1024]),
        (128, 64, 4, [512, 1024]),
    ],
)
def test_fmha_fwd_with_sink_varlen_asm_correctness(
    head_dim, hq, hk, seqlens, is_causal
):
    device = "cuda"
    q, k, v, cu = make_varlen_packed(seqlens, hq, hk, head_dim, head_dim, device=device)
    cu_q = cu
    cu_k = cu  # equal q/k seqlens per batch
    max_seqlen_q = max(seqlens)
    scale = 1.0 / math.sqrt(head_dim)

    # D64 -> exercise sink; D128 -> kernel ignores sink (pass None).
    sink = _d64_sink(hq, device) if head_dim == 64 else None

    # Drive the public API (aiter.flash_attn_varlen_func), which dispatches
    # to the fmha_fwd_with_sink_varlen_asm branch on gfx1250.
    out_k, lse_k = run_kernel(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q,
        scale=scale,
        is_causal=is_causal,
        sink=sink,
        via="public",
    )

    msg = f"d={head_dim} hq={hq} hk={hk} seqlens={seqlens}"
    _ok = out_k.detach().float().cpu()
    assert not _ok.isnan().any().item(), f"KERNEL out NaN [{msg}]"
    assert not _ok.isinf().any().item(), f"KERNEL out Inf [{msg}]"

    out_ref, lse_ref = _ref_varlen(q, k, v, cu_q, cu_k, is_causal=is_causal, sink=sink)

    _cmp(out_k, out_ref, rtol=1e-2, atol=1e-2, msg=f"out mismatch [{msg}]")
    _cmp(lse_k, lse_ref, rtol=1e-2, atol=1e-2, msg=f"lse mismatch [{msg}]")


# ---------------------------------------------------------------------------
# Integration test: aiter.flash_attn_varlen_func -> _flash_attn_varlen_forward
# dispatcher -> fmha_fwd_with_sink_varlen_asm branch.  Verifies the public-API
# path on gfx1250 matches a direct ops-layer call bit-for-bit (same kernel,
# same args) — the lse layout differs (ops: (total_q, nheads); public:
# (nheads, total_q)) but run_kernel normalizes both to (total_q, nheads).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_causal", [True])
def test_fmha_fwd_with_sink_varlen_asm_via_flash_attn_varlen_func(head_dim, is_causal):
    device = "cuda"
    hq, hk, seqlens = 8, 1, [128, 256, 384]
    q, k, v, cu = make_varlen_packed(seqlens, hq, hk, head_dim, head_dim, device=device)
    max_seqlen_q = max(seqlens)
    scale = 1.0 / math.sqrt(head_dim)
    sink = _d64_sink(hq, device) if head_dim == 64 else None

    out_direct, lse_direct = run_kernel(
        q,
        k,
        v,
        cu,
        cu,
        max_seqlen_q,
        scale=scale,
        is_causal=is_causal,
        sink=sink,
        via="ops",
    )
    out_via, lse_via = run_kernel(
        q,
        k,
        v,
        cu,
        cu,
        max_seqlen_q,
        scale=scale,
        is_causal=is_causal,
        sink=sink,
        via="public",
    )

    # Same kernel, same args -> bit-identical (cast to fp32 to avoid bf16
    # element-wise hang in some ROCm builds).
    do = (out_via.float() - out_direct.float()).abs().max().item()
    dl = (lse_via.float() - lse_direct.float()).abs().max().item()
    assert do == 0.0, (
        f"flash_attn_varlen_func != fmha_fwd_with_sink_varlen_asm "
        f"(d={head_dim}, causal={is_causal})  max|dO|={do}"
    )
    assert dl == 0.0, (
        f"lse via flash_attn_varlen_func != direct "
        f"(d={head_dim}, causal={is_causal})  max|dLSE|={dl}"
    )


# ---------------------------------------------------------------------------
# Perf (single multi-batch shape per head_dim)
# ---------------------------------------------------------------------------


def _bench(fn, *args, num_iters=20, num_warmup=10, **kwargs) -> float:
    for _ in range(num_warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        fn(*args, **kwargs)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / num_iters  # us per iter


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_causal", [True])
def test_fmha_fwd_with_sink_varlen_asm_perf(head_dim, is_causal):
    device = "cuda"
    if head_dim == 64:
        hq, hk, seqlens = 64, 8, [4096, 4096]
    else:
        hq, hk, seqlens = 64, 4, [2048, 2048]
    q, k, v, cu = make_varlen_packed(seqlens, hq, hk, head_dim, head_dim, device=device)
    max_seqlen_q = max(seqlens)
    scale = 1.0 / math.sqrt(head_dim)
    sink = _d64_sink(hq, device) if head_dim == 64 else None

    us = _bench(
        aiter.fmha_fwd_with_sink_varlen_asm,
        q,
        k,
        v,
        cu,
        cu,
        max_seqlen_q,
        scale,
        is_causal,
        False,
        sink=sink,
    )
    # Causal FLOPs summed over batches (each ~ 2 * hq * s^2 * 2d / 2).
    flops = sum(2.0 * hq * s * s * (2 * head_dim) / 2.0 for s in seqlens)
    tflops = flops / (us * 1e-6) / 1e12
    print(
        f"[perf varlen] d={head_dim} causal={is_causal} seqlens={seqlens}: {us:.1f}us, {tflops:.2f} TFLOPS"
    )
    assert us > 0.0 and math.isfinite(tflops)
