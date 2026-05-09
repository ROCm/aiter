# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness + performance tests for fmha_fwd_f16 (BF16 ASM, gfx1250).

Public API:    aiter.flash_attn_func          (preferred)
Ops layer:     aiter.fmha_fwd_f16_asm         (low-level, ~v3 style)


Layout convention used in tests
--------------------------------
The aiter API only accepts bshd shape ([b, s, h, d]).  To exercise the
kernel's ability to follow strides for sbhd / bhsd memory layouts, the
test allocates qkv in the chosen `layout` and `permute()`s to bshd shape
WITHOUT calling `.contiguous()` — the resulting tensors are bshd-shaped
non-contiguous views whose `.stride()` reflects the underlying memory.

Sink convention
---------------
D64 (_rxy_sink) kernels compile ENABLE_SINK=1.  An explicit sink tensor
[q_head_num] fp32 (AITER post-scale) is required.

Sink mechanism (from common_fmha.h::fmha_merge_sink_rowwise):
  After computing standard softmax numerators/denominators, the sink acts as
  an additional "virtual KV token" with zero value vector.  It only adds to
  the softmax denominator:
      new_max    = max(max_attn_raw, sink_raw)
      sink_term  = exp2((sink_raw - new_max) * scale * log2e)
      denom      = denom * rescale + sink_term
      numer      = numer * rescale     # sink contributes 0 to output
  In AITER/post-scale convention: sink_raw = sink_user * sqrt(head_dim).
"""

from __future__ import annotations

import argparse
import math
import sys
import time as _t
from typing import Optional

import pytest
import torch

import aiter

from aiter.test_common import checkAllclose
from aiter.jit.utils.chip_info import get_gfx

# from aiter.test_mha_common import (
#    attention_ref,
#
# )  # noqa: F401  (kept for easy swap-back; see doc-block below)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="ROCm/HIP GPU not available",
)

# ---------------------------------------------------------------------------
# Reference implementation.  Inputs accepted as bshd (matches kernel API);
# output `out` is bshd, `lse` is [b, hq, sq] (matches kernel layout).
#
# We default to the in-file `_ref_attn` rather than
# `aiter.test_mha_common.attention_ref` because the latter casts its
# returned `lse` back to q.dtype (bf16) — see test_mha_common.py:615 —
# even when called with upcast=True.  That round-trip introduces ~1 bf16
# ULP of quantization on lse (~3e-2 for sq=8192 d=128), which exceeds
# tight comparison thresholds.  `_ref_attn` keeps lse in fp32 and
# matches the kernel to ~5e-6 (essentially fp32 noise floor).
#
# attention_ref is still imported above so it is trivial to swap back
# when (a) the upstream API stops casting lse to bf16, or (b) you only
# need rtol-based comparison (rtol=1% absorbs the bf16 quantization).
#
# Historical aside: an earlier ROCm 7.13 driver could enter a wedged
# state after many ASM kernel launches, after which ANY GPU op (incl.
# attention_ref) would hang in uninterruptible sleep until
# `rocm-smi --gpureset`.  The wedge is environmental, not a property
# of attention_ref itself.
# ---------------------------------------------------------------------------


def _ref_attn(q, k, v, *, is_causal: bool, sink: "Optional[torch.Tensor]" = None):
    """bshd-in / bshd-out attention reference, sink optional.  Pure-einsum
    fp32 implementation; lse is returned in fp32 (matches kernel's output).

    Math:  attn  = Q @ K^T,   scale = 1/sqrt(d),
           denom = sum(exp((attn - max) * scale))
                 [+ exp((sink_raw - max) * scale)],
           out   = (exp((attn - max) * scale) / denom) @ V,
           lse   = max * scale + log(denom).
    sink (optional): [hq] fp32, AITER post-scale; converted internally to
                     pre-scale raw via x sqrt(d) to match kernel ABI.
    """
    b, sq, hq, d = q.shape
    _, sk, hk, _ = k.shape
    if hq != hk:
        k = k.repeat_interleave(hq // hk, dim=2)
        v = v.repeat_interleave(hq // hk, dim=2)
    qf, kf, vf = q.float(), k.float(), v.float()
    scale = 1.0 / math.sqrt(d)
    attn = torch.einsum("bshd,bkhd->bhsk", qf, kf)
    if is_causal:
        m = torch.triu(
            torch.ones(sq, sk, dtype=torch.bool, device=q.device), sk - sq + 1
        )
        attn = attn.masked_fill(m, float("-inf"))
    max_attn, _ = attn.max(dim=-1)
    if sink is not None:
        sink_raw = sink.float() * math.sqrt(d)
        sink_raw_bhs = sink_raw[None, :, None].expand(b, hq, sq)
        max_total = torch.maximum(max_attn, sink_raw_bhs)
    else:
        max_total = max_attn
    denom_real = torch.exp((attn - max_total.unsqueeze(-1)) * scale).sum(dim=-1)
    if sink is not None:
        sink_term = torch.exp((sink_raw_bhs - max_total) * scale)
        denom_total = denom_real + sink_term
    else:
        denom_total = denom_real
    probs = torch.exp((attn - max_total.unsqueeze(-1)) * scale) / denom_total.unsqueeze(
        -1
    )
    out = torch.einsum("bhsk,bkhd->bshd", probs, vf).to(q.dtype)
    lse = torch.log(denom_total) + max_total * scale
    return out, lse


def _cmp(a: torch.Tensor, b: torch.Tensor, *, rtol=1e-2, atol=1e-2, msg: str = ""):
    """bf16-safe wrapper around checkAllclose.

    On gfx1250 + ROCm 7.13 some bf16 element-wise GPU ops (isnan / isclose /
    contiguous) deadlock when invoked right after a custom ASM kernel.  The
    deadlock is unrelated to fmha_fwd_f16 itself (it has been reproduced with
    pure-PyTorch programs).  As a workaround we cast both tensors to fp32 on
    CPU before comparing — this avoids triggering the buggy GPU bf16 path.
    """
    a32 = a.detach().float().cpu()
    b32 = b.detach().float().cpu()
    checkAllclose(a32, b32, rtol=rtol, atol=atol, msg=msg)


def _nrms(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Normalized RMS error on fp32 CPU tensors (avoids bf16 GPU element-wise hang).

    Definition matches op_tests/test_mha_mxfp8.py:
        nrms = sqrt(sum((|a-b| / max(|b|, eps))^2)) /
               (sqrt(numel) * max(|a|.max, |b|.max, eps))
    A small relative metric (~1e-3 for bf16, ~1e-6 for fp32) regardless of
    output magnitude — useful complement to the absolute max-diff check.
    """
    a32 = actual.detach().float().cpu()
    b32 = expected.detach().float().cpu()
    abs_diff = (a32 - b32).abs()
    eps = 1e-7
    max_item = max(a32.abs().max().item(), b32.abs().max().item(), eps)
    sq_diff = (abs_diff / b32.abs().clamp(min=eps)).pow(2)
    return (sq_diff.sum().sqrt() / (math.sqrt(b32.numel()) * max_item)).item()


def _bench(fn, *args, num_iters: int = 10, num_warmup: int = 2, **kwargs) -> float:
    """CUDA-Event-based per-iter timing (us).

    Bypasses run_perftest because torch.profiler / ROCTracer drops kernel
    events on gfx1250 + ROCm 7.x (warning: "ROCTracer produced duplicate
    flow start"), making run_perftest report 0 us / inf TFLOPS.
    """
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
    return start.elapsed_time(end) * 1000.0 / num_iters  # ms->us, per-iter


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def make_qkv_bshd(
    layout: int,
    sq: int,
    sk: int,
    batch: int,
    hq: int,
    hk: int,
    d: int,
    dtype=torch.bfloat16,
    device: str = "cuda",
):
    """Allocate (q, k, v) in `layout` memory, return **bshd-shaped views**.

    The API only accepts bshd shape ([b, s, h, d]).  But the kernel reads
    strides directly via `tensor.stride(...)`, so the underlying memory may
    be laid out differently.  This helper allocates contiguous tensors in
    the requested layout and returns a `permute()` view (no `.contiguous()`)
    so .shape == bshd while .stride() reflects the underlying memory.

    layout code:
        0 = bshd  → contiguous bshd, strides = (s*h*d, h*d, d, 1)
        1 = bhsd  → underlying [b,h,s,d], permute(0,2,1,3) → bshd view
        2 = sbhd  → underlying [s,b,h,d], permute(1,0,2,3) → bshd view
    """
    if layout == 0:  # bshd allocation, naturally contiguous
        q = torch.randn(batch, sq, hq, d, dtype=dtype, device=device)
        k = torch.randn(batch, sk, hk, d, dtype=dtype, device=device)
        v = torch.randn(batch, sk, hk, d, dtype=dtype, device=device)
    elif layout == 1:  # bhsd allocation, view as bshd
        q = torch.randn(batch, hq, sq, d, dtype=dtype, device=device).permute(
            0, 2, 1, 3
        )
        k = torch.randn(batch, hk, sk, d, dtype=dtype, device=device).permute(
            0, 2, 1, 3
        )
        v = torch.randn(batch, hk, sk, d, dtype=dtype, device=device).permute(
            0, 2, 1, 3
        )
    elif layout == 2:  # sbhd allocation, view as bshd
        q = torch.randn(sq, batch, hq, d, dtype=dtype, device=device).permute(
            1, 0, 2, 3
        )
        k = torch.randn(sk, batch, hk, d, dtype=dtype, device=device).permute(
            1, 0, 2, 3
        )
        v = torch.randn(sk, batch, hk, d, dtype=dtype, device=device).permute(
            1, 0, 2, 3
        )
    else:
        raise ValueError(f"unsupported layout={layout}")
    return q, k, v


def _d64_sink(hq: int, device: str) -> torch.Tensor:
    """Non-zero sink for D64: fixed per-head values in AITER post-scale domain.

    Values in [0.5, 2.0]; varies across heads to exercise broadcast.
    """
    return torch.linspace(0.5, 2.0, hq, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Kernel / reference helpers (mxfp8-style: one-line wrappers used by tests).
# ---------------------------------------------------------------------------


def run_kernel(
    q,
    k,
    v,
    *,
    scale: float,
    is_causal: bool,
    sink: Optional[torch.Tensor] = None,
    via: str = "ops",
):
    """Call the kernel and return (out, lse).

    via = "ops"        → low-level aiter.fmha_fwd_f16_asm
    via = "public"     → public aiter.flash_attn_func (dispatcher → asm path)
    """
    if via == "ops":
        return aiter.fmha_fwd_f16_asm(
            q,
            k,
            v,
            scale,
            is_causal,
            True,
            sink=sink,
        )
    if via == "public":
        r = aiter.flash_attn_func(
            q,
            k,
            v,
            softmax_scale=scale,
            causal=is_causal,
            return_lse=True,
            sink_ptr=sink,
        )
        return r[0], r[1]
    raise ValueError(f"unknown via={via!r}")


def run_ref(q, k, v, *, is_causal: bool, sink: Optional[torch.Tensor] = None):
    """Reference (out, lse) computed on the same bshd tensors via the in-file
    `_ref_attn`.  See doc-block above for why we don't use
    `aiter.test_mha_common.attention_ref` directly.
    """
    return _ref_attn(q, k, v, is_causal=is_causal, sink=sink)


# ---------------------------------------------------------------------------
# Correctness tests (sbhd input → bshd output, compare against bhsd reference)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize(
    "hq,hk,sq,sk",
    [
        # Shapes from run.sh aligned tests: kv_head_num=4, gqa=16
        # → q_head_num = 4 * 16 = 64
        (8, 1, 128, 2048),  # aligned (test_d64 / test_d128)
        (8, 1, 130, 2048),  # q unaligned: sq not mult of 128
        (8, 1, 128, 2300),  # kv unaligned: sk not mult of 256
    ],
)
def test_fmha_fwd_f16_correctness(batch, hq, hk, sq, sk, head_dim, is_causal):
    if get_gfx() not in ["gfx1250"]:
        return
    device = "cuda"
    torch.manual_seed(0)

    # Allocate in sbhd memory but return bshd-shaped views (kernel reads
    # strides directly so non-contiguous bshd views work).
    q, k, v = make_qkv_bshd(
        layout=2,
        sq=sq,
        sk=sk,
        batch=batch,
        hq=hq,
        hk=hk,
        d=head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    scale = 1.0 / math.sqrt(head_dim)

    # D64 -> non-zero sink (exercises ENABLE_SINK code path)
    # D128 -> no sink (kernel ignores it)
    sink = _d64_sink(hq, device) if head_dim == 64 else None

    out_kernel, lse_asm = run_kernel(
        q,
        k,
        v,
        scale=scale,
        is_causal=is_causal,
        sink=sink,
        via="public",
    )
    out_ref, lse_ref = run_ref(q, k, v, is_causal=is_causal, sink=sink)

    nrms_o = _nrms(out_kernel, out_ref)
    print(
        f"[corr d={head_dim} causal={is_causal} b={batch} sq={sq} sk={sk}] "
        f"nrms(out)={nrms_o:.3e}"
    )

    _cmp(
        out_kernel,
        out_ref,
        rtol=1e-2,
        atol=1e-2,
        msg=f"out mismatch (d={head_dim}, causal={is_causal}, b={batch})",
    )
    _cmp(
        lse_asm,
        lse_ref,
        rtol=1e-2,
        atol=1e-2,
        msg=f"lse mismatch (d={head_dim}, causal={is_causal}, b={batch})",
    )


def test_fmha_fwd_f16_ops_layer():
    """Direct ops-layer call: bshd qkv (sbhd memory layout), D64 + non-zero sink."""
    if get_gfx() not in ["gfx1250"]:
        return
    device = "cuda"
    torch.manual_seed(0)

    sq, batch, hq, hk, sk, d = 128, 1, 8, 2, 2048, 64
    q, k, v = make_qkv_bshd(
        layout=2,
        sq=sq,
        sk=sk,
        batch=batch,
        hq=hq,
        hk=hk,
        d=d,
        dtype=torch.bfloat16,
        device=device,
    )
    scale = 1.0 / math.sqrt(d)
    sink = _d64_sink(hq, device)

    out_kernel, lse_asm = run_kernel(
        q,
        k,
        v,
        scale=scale,
        is_causal=False,
        sink=sink,
        via="ops",
    )
    out_ref, lse_ref = run_ref(q, k, v, is_causal=False, sink=sink)

    _cmp(out_kernel, out_ref, rtol=1e-2, atol=1e-2)
    _cmp(lse_asm, lse_ref, rtol=1e-2, atol=1e-2)


def test_fmha_fwd_f16_d64_requires_sink():
    """Direct ops-layer call without sink on D64 must raise the C++ check.

    Note: when going through aiter.flash_attn_func, the dispatcher auto-fills
    a zero sink for D64, so this error path is unreachable from the public
    API — we exercise it via the lower-level ops stub.
    """
    if get_gfx() not in ["gfx1250"]:
        return
    device = "cuda"
    q, k, v = make_qkv_bshd(
        layout=0,
        sq=128,
        sk=2048,
        batch=1,
        hq=4,
        hk=4,
        d=64,
        dtype=torch.bfloat16,
        device=device,
    )
    scale = 1.0 / math.sqrt(64)
    with pytest.raises(RuntimeError, match="D64.*sink"):
        aiter.fmha_fwd_f16_asm(q, k, v, scale, False, True, sink=None)


# ---------------------------------------------------------------------------
# Memory-layout tests: API takes only bshd shape, but the kernel reads strides
# directly so non-contiguous bshd views (backed by sbhd / bhsd memory) must
# also produce correct results.  3 layouts x 2 head_dim = 6 cases.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("layout", [0, 1, 2])
def test_fmha_fwd_f16_layout(layout, head_dim):
    if get_gfx() not in ["gfx1250"]:
        return
    device = "cuda"
    torch.manual_seed(0)
    batch, hq, hk, sq, sk = 1, 8, 1, 128, 2048

    q, k, v = make_qkv_bshd(
        layout=layout,
        sq=sq,
        sk=sk,
        batch=batch,
        hq=hq,
        hk=hk,
        d=head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    scale = 1.0 / math.sqrt(head_dim)
    sink = _d64_sink(hq, device) if head_dim == 64 else None

    out_kernel, lse_asm = run_kernel(
        q,
        k,
        v,
        scale=scale,
        is_causal=False,
        sink=sink,
        via="public",
    )
    out_ref, lse_ref = run_ref(q, k, v, is_causal=False, sink=sink)

    _cmp(
        out_kernel,
        out_ref,
        rtol=1e-2,
        atol=1e-2,
        msg=f"out mismatch (layout={layout}, d={head_dim})",
    )
    _cmp(
        lse_asm,
        lse_ref,
        rtol=1e-2,
        atol=1e-2,
        msg=f"lse mismatch (layout={layout}, d={head_dim})",
    )


# ---------------------------------------------------------------------------
# Integration test: aiter.flash_attn_func -> mha._flash_attn_forward dispatcher
# -> our fmha_fwd_f16_asm branch.  Verifies the public-API path on gfx1250
# matches a direct ops-layer call bit-for-bit (same kernel, same args).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
def test_fmha_fwd_f16_via_flash_attn_func(head_dim, is_causal):
    if get_gfx() not in ["gfx1250"]:
        return
    device = "cuda"
    torch.manual_seed(0)
    batch, hq, hk, sq, sk = 1, 8, 1, 128, 2048

    # bshd input (flash_attn_func contract); contiguous.
    q, k, v = make_qkv_bshd(
        layout=0,
        sq=sq,
        sk=sk,
        batch=batch,
        hq=hq,
        hk=hk,
        d=head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    scale = 1.0 / math.sqrt(head_dim)
    sink = _d64_sink(hq, device) if head_dim == 64 else None

    out_direct, lse_direct = run_kernel(
        q,
        k,
        v,
        scale=scale,
        is_causal=is_causal,
        sink=sink,
        via="ops",
    )
    out_via, lse_via = run_kernel(
        q,
        k,
        v,
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
        f"flash_attn_func != fmha_fwd_f16_asm "
        f"(d={head_dim}, causal={is_causal})  max|dO|={do}"
    )
    assert dl == 0.0, (
        f"lse via flash_attn_func != direct "
        f"(d={head_dim}, causal={is_causal})  max|dLSE|={dl}"
    )


# ---------------------------------------------------------------------------
# Performance tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
def test_fmha_fwd_f16_perf(head_dim, is_causal):
    if get_gfx() not in ["gfx1250"]:
        return
    device = "cuda"
    torch.manual_seed(0)

    # perf_d64 / perf_d128 in run.sh: batch=2 kv_head_num=8 gqa=8 -> hq=64
    sq, batch, hq, hk, sk = 8192, 2, 64, 8, 8192
    q, k, v = make_qkv_bshd(
        layout=2,
        sq=sq,
        sk=sk,
        batch=batch,
        hq=hq,
        hk=hk,
        d=head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    scale = 1.0 / math.sqrt(head_dim)
    sink = _d64_sink(hq, device) if head_dim == 64 else None

    us = _bench(
        aiter.fmha_fwd_f16_asm,
        q,
        k,
        v,
        scale,
        is_causal,
        False,
        sink=sink,
        num_iters=10,
        num_warmup=2,
    )
    flops = 2.0 * batch * hq * sq * sk * (2 * head_dim)
    if is_causal:
        flops /= 2.0
    tflops = flops / (us * 1e-6) / 1e12
    print(f"[perf] d={head_dim} causal={is_causal}: {us:.1f}us, {tflops:.2f} TFLOPS")
    # Sanity: catch silent-PASS when timing infrastructure breaks (e.g. profiler
    # / ROCTracer drops events → us=0, TFLOPS=inf).  Without these asserts the
    # test would PASS with bogus numbers.
    assert us > 0.0, (
        f"perf timing returned us={us}; timing path broken "
        f"(run with -s to see live numbers)"
    )
    assert math.isfinite(tflops) and 0 < tflops < 5000, (
        f"TFLOPS={tflops} not finite / out of plausible range; " f"likely broken timing"
    )


# ---------------------------------------------------------------------------
# CLI single-shape runner: shared by `__main__` invocation and ad-hoc usage.
# ---------------------------------------------------------------------------


def run_cli(
    *,
    batch: int,
    hq: int,
    hk: int,
    sq: int,
    sk: int,
    head_dim: int,
    causal: bool = False,
    layout: int = 0,
    do_ref: bool = False,
    do_perf: bool = False,
) -> int:
    """Single-shape runner.

    Returns 0 on success, 1 if --ref check fails.  Prints a one-line summary
    of kernel shape / time and (if requested) ref / perf metrics.
    """
    device = "cuda"
    torch.manual_seed(0)
    assert hq % hk == 0, "q_head_num must be a multiple of kv_head_num"

    print(
        f"Shape: b={batch} hq={hq} hk={hk} sq={sq} sk={sk} d={head_dim} "
        f"causal={causal} layout={layout}",
        flush=True,
    )

    q, k, v = make_qkv_bshd(
        layout=layout,
        sq=sq,
        sk=sk,
        batch=batch,
        hq=hq,
        hk=hk,
        d=head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    scale = 1.0 / math.sqrt(head_dim)
    sink = _d64_sink(hq, device) if head_dim == 64 else None
    torch.cuda.synchronize()

    t0 = _t.time()
    out_kernel, lse_asm = run_kernel(
        q,
        k,
        v,
        scale=scale,
        is_causal=causal,
        sink=sink,
        via="ops",
    )
    torch.cuda.synchronize()
    print(f"asm time: {(_t.time()-t0)*1000:.2f} ms", flush=True)
    print(
        f"out.shape={tuple(out_kernel.shape)}  lse.shape={tuple(lse_asm.shape)}",
        flush=True,
    )

    rc = 0
    if do_ref:
        out_ref, lse_ref = run_ref(q, k, v, is_causal=causal, sink=sink)
        diff_o = (out_kernel.float() - out_ref.float()).abs().max().item()
        diff_l = (lse_asm.float() - lse_ref.float()).abs().max().item()
        nrms_o = _nrms(out_kernel, out_ref)
        # Pass criterion (bf16 attention conventional thresholds):
        #   |dO|   <= 2e-2   |dLSE| <= 2e-2
        ok_o = diff_o <= 2e-2
        ok_l = diff_l <= 2e-2
        print(
            f"ref:  max|dO|={diff_o:.4f} {'OK' if ok_o else 'FAIL'}   "
            f"max|dLSE|={diff_l:.4f} {'OK' if ok_l else 'FAIL'}   "
            f"nrms(O)={nrms_o:.3e}",
            flush=True,
        )
        if not (ok_o and ok_l):
            rc = 1

    if do_perf:
        us = _bench(
            aiter.fmha_fwd_f16_asm,
            q,
            k,
            v,
            scale,
            causal,
            False,
            sink=sink,
            num_iters=10,
            num_warmup=2,
        )
        flops = 2.0 * batch * hq * sq * sk * (2 * head_dim)
        if causal:
            flops /= 2.0
        tflops = flops / (us * 1e-6) / 1e12
        print(f"perf: {us:.1f} us  ({tflops:.2f} TFLOPS)", flush=True)
        # CLI surfaces the same breakage pytest would: us=0 / TFLOPS=inf
        # signals broken timing infra (profiler / ROCTracer event drop).
        if not (us > 0.0 and math.isfinite(tflops) and 0 < tflops < 5000):
            print(
                f"perf: WARNING — bogus timing (us={us}, tflops={tflops})", flush=True
            )
            rc = 1

    return rc


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Run aiter.fmha_fwd_f16_asm on a single shape and dump kernel args.",
)
parser.add_argument("-b", "--batch", type=int, default=1, help="batch size (default 1)")
parser.add_argument(
    "-n", "--q_head_num", type=int, default=8, help="q_head_num (default 8)"
)
parser.add_argument(
    "-kn",
    "--kv_head_num",
    type=int,
    default=1,
    help="kv_head_num (default 1, must divide q_head_num)",
)
parser.add_argument(
    "-q", "--seqlen_q", type=int, default=128, help="q seq length (default 128)"
)
parser.add_argument(
    "-k", "--seqlen_k", type=int, default=2048, help="kv seq length (default 2048)"
)
parser.add_argument(
    "-d",
    "--head_dim",
    type=int,
    choices=[64, 128],
    default=128,
    help="head dim, 64 or 128 (default 128)",
)
parser.add_argument("-c", "--causal", action="store_true", help="enable causal mask")
parser.add_argument(
    "-l",
    "--layout",
    type=int,
    choices=[0, 1, 2],
    default=0,
    help="input memory layout: 0=bshd 1=bhsd 2=sbhd (default 0)\n"
    "(API always sees bshd shape; non-zero layout returns a\n"
    "non-contiguous bshd view of the underlying memory)",
)
parser.add_argument(
    "--ref",
    action="store_true",
    help="also run PyTorch reference and print max diff + nrms",
)
parser.add_argument(
    "--perf",
    action="store_true",
    help="run perf benchmark for this shape (10 iters, 2 warmup)",
)

if __name__ == "__main__":
    args = parser.parse_args()
    rc = run_cli(
        batch=args.batch,
        hq=args.q_head_num,
        hk=args.kv_head_num,
        sq=args.seqlen_q,
        sk=args.seqlen_k,
        head_dim=args.head_dim,
        causal=args.causal,
        layout=args.layout,
        do_ref=args.ref,
        do_perf=args.perf,
    )
    sys.exit(rc)
