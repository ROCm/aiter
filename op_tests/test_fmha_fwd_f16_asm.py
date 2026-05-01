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

import math
from typing import Optional

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="ROCm/HIP GPU not available",
)

import aiter
from aiter.test_common import checkAllclose, run_perftest


# ---------------------------------------------------------------------------
# Reference implementations.  Inputs accepted as bshd (matches kernel API);
# output `out` is bshd; `lse` is [b, hq, sq] (matches kernel layout).
#
# NB: we intentionally *do not* use `aiter.test_mha_common.attention_ref`
# here — although it's mathematically equivalent (sink as virtual KV with
# zero V, see test_mha_common.py:584), running it on gfx1250 + ROCm 7.13
# triggers a downstream driver wedge that causes the *next* ASM kernel
# launch to hang.  The pure-einsum impl below is hand-derived from
# fmha_merge_sink_rowwise, runs reliably on gfx1250, and produces
# bit-identical numerics to attention_ref(... .float() ...).
# ---------------------------------------------------------------------------

def _ref_attn(q, k, v, *, is_causal: bool, sink: Optional[torch.Tensor] = None):
    """bshd-in / bshd-out attention reference, sink optional.

    Math:  attn = Q @ K^T,  scale = 1/sqrt(d),
           denom = sum(exp((attn - max) * scale)) [+ exp((sink_raw - max) * scale)],
           out   = (exp((attn - max) * scale) / denom) @ V,
           lse   = max * scale + log(denom).
    sink (optional): [hq] fp32, AITER post-scale; converted internally to
                     pre-scale raw via × sqrt(d) to match kernel ABI.
    """
    b, sq, hq, d = q.shape
    _, sk, hk, _ = k.shape
    if hq != hk:
        k = k.repeat_interleave(hq // hk, dim=2)
        v = v.repeat_interleave(hq // hk, dim=2)
    qf, kf, vf = q.float(), k.float(), v.float()
    scale = 1.0 / math.sqrt(d)
    attn = torch.einsum("bshd,bkhd->bhsk", qf, kf)              # raw, no scale
    if is_causal:
        m = torch.triu(torch.ones(sq, sk, dtype=torch.bool, device=q.device),
                       sk - sq + 1)
        attn = attn.masked_fill(m, float("-inf"))
    max_attn, _ = attn.max(dim=-1)                              # [b, hq, sq]
    if sink is not None:
        sink_raw = sink.float() * math.sqrt(d)                  # [hq]
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
    probs = torch.exp((attn - max_total.unsqueeze(-1)) * scale) \
            / denom_total.unsqueeze(-1)
    out = torch.einsum("bhsk,bkhd->bshd", probs, vf).to(q.dtype)
    lse = torch.log(denom_total) + max_total * scale            # [b, hq, sq]
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


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def make_qkv_bshd(layout: int, sq: int, sk: int, batch: int, hq: int, hk: int, d: int,
                   dtype=torch.bfloat16, device: str = "cuda"):
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
    if layout == 0:    # bshd allocation, naturally contiguous
        q = torch.randn(batch, sq, hq, d, dtype=dtype, device=device)
        k = torch.randn(batch, sk, hk, d, dtype=dtype, device=device)
        v = torch.randn(batch, sk, hk, d, dtype=dtype, device=device)
    elif layout == 1:  # bhsd allocation, view as bshd
        q = torch.randn(batch, hq, sq, d, dtype=dtype, device=device).permute(0, 2, 1, 3)
        k = torch.randn(batch, hk, sk, d, dtype=dtype, device=device).permute(0, 2, 1, 3)
        v = torch.randn(batch, hk, sk, d, dtype=dtype, device=device).permute(0, 2, 1, 3)
    elif layout == 2:  # sbhd allocation, view as bshd
        q = torch.randn(sq, batch, hq, d, dtype=dtype, device=device).permute(1, 0, 2, 3)
        k = torch.randn(sk, batch, hk, d, dtype=dtype, device=device).permute(1, 0, 2, 3)
        v = torch.randn(sk, batch, hk, d, dtype=dtype, device=device).permute(1, 0, 2, 3)
    else:
        raise ValueError(f"unsupported layout={layout}")
    return q, k, v


# ---------------------------------------------------------------------------
# Correctness tests (sbhd input → bshd output, compare against bhsd reference)
# ---------------------------------------------------------------------------

def _d64_sink(hq: int, device: str) -> torch.Tensor:
    """Non-zero sink for D64: fixed per-head values in AITER post-scale domain."""
    # Use values in [0.5, 2.0] post-scale; vary across heads for thorough test
    return torch.linspace(0.5, 2.0, hq, dtype=torch.float32, device=device)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize(
    "batch,hq,hk,sq,sk",
    [
        # Shapes from run.sh aligned tests: batch=1, kv_head_num=4, gqa=16
        # → q_head_num = 4 * 16 = 64
        (1, 8, 1,  128, 2048),   # aligned (test_d64 / test_d128)
        (1, 8, 1,  130, 2048),   # q unaligned: sq not mult of 128
        (1, 8, 1,  128, 2300),   # kv unaligned: sk not mult of 256
    ],
)
def test_fmha_fwd_f16_correctness(batch, hq, hk, sq, sk, head_dim, is_causal):
    device = "cuda"
    torch.manual_seed(0)

    # Allocate in sbhd memory but return bshd-shaped views (kernel reads
    # strides directly so non-contiguous bshd views work).
    q, k, v = make_qkv_bshd(layout=2, sq=sq, sk=sk, batch=batch,
                             hq=hq, hk=hk, d=head_dim,
                             dtype=torch.bfloat16, device=device)
    scale = 1.0 / math.sqrt(head_dim)

    # D64 -> non-zero sink (exercises ENABLE_SINK code path)
    # D128 -> no sink (kernel ignores it)
    sink = _d64_sink(hq, device) if head_dim == 64 else None

    _r = aiter.flash_attn_func(
        q, k, v,
        softmax_scale=scale, causal=is_causal,
        return_lse=True, sink_ptr=sink,
    )
    out_kernel, lse_asm = _r[0], _r[1]

    # Reference: bshd in / bshd out (matches kernel layout, no permute needed)
    out_ref, lse_ref = _ref_attn(q, k, v, is_causal=is_causal, sink=sink)

    _cmp(out_kernel, out_ref, rtol=1e-2, atol=1e-2,
         msg=f"out mismatch (d={head_dim}, causal={is_causal})")
    _cmp(lse_asm, lse_ref, rtol=1e-2, atol=1e-2,
         msg=f"lse mismatch (d={head_dim}, causal={is_causal})")


def test_fmha_fwd_f16_ops_layer():
    """Direct ops-layer call: bshd qkv (sbhd memory layout), D64 + non-zero sink."""
    device = "cuda"
    torch.manual_seed(0)

    sq, batch, hq, hk, sk, d = 128, 1, 8, 2, 2048, 64
    q, k, v = make_qkv_bshd(layout=2, sq=sq, sk=sk, batch=batch,
                             hq=hq, hk=hk, d=d,
                             dtype=torch.bfloat16, device=device)
    scale = 1.0 / math.sqrt(d)
    sink  = _d64_sink(hq, device)

    out_kernel, lse_asm = aiter.fmha_fwd_f16_asm(
        q, k, v, scale, False, True, sink=sink,
    )

    out_ref, lse_ref = _ref_attn(q, k, v, is_causal=False, sink=sink)
    _cmp(out_kernel, out_ref, rtol=1e-2, atol=1e-2)
    _cmp(lse_asm, lse_ref, rtol=1e-2, atol=1e-2)


def test_fmha_fwd_f16_d64_requires_sink():
    """Direct ops-layer call without sink on D64 must raise the C++ check.

    Note: when going through aiter.flash_attn_func, the dispatcher auto-fills
    a zero sink for D64, so this error path is unreachable from the public
    API — we exercise it via the lower-level ops stub.
    """
    device = "cuda"
    q, k, v = make_qkv_bshd(layout=0, sq=128, sk=2048, batch=1,
                             hq=4, hk=4, d=64,
                             dtype=torch.bfloat16, device=device)
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
    device = "cuda"
    torch.manual_seed(0)
    batch, hq, hk, sq, sk = 1, 8, 1, 128, 2048

    q, k, v = make_qkv_bshd(layout=layout, sq=sq, sk=sk, batch=batch,
                             hq=hq, hk=hk, d=head_dim,
                             dtype=torch.bfloat16, device=device)
    scale = 1.0 / math.sqrt(head_dim)
    sink  = _d64_sink(hq, device) if head_dim == 64 else None

    _r = aiter.flash_attn_func(
        q, k, v,
        softmax_scale=scale, causal=False, return_lse=True, sink_ptr=sink,
    )
    out_kernel, lse_asm = _r[0], _r[1]

    out_ref, lse_ref = _ref_attn(q, k, v, is_causal=False, sink=sink)

    _cmp(out_kernel, out_ref, rtol=1e-2, atol=1e-2,
         msg=f"out mismatch (layout={layout}, d={head_dim})")
    _cmp(lse_asm, lse_ref, rtol=1e-2, atol=1e-2,
         msg=f"lse mismatch (layout={layout}, d={head_dim})")


# ---------------------------------------------------------------------------
# Integration test: aiter.flash_attn_func -> mha._flash_attn_forward dispatcher
# -> our fmha_fwd_f16_asm branch.  Verifies the public-API path on gfx1250
# matches a direct ops-layer call bit-for-bit (same kernel, same args).
# ---------------------------------------------------------------------------

def _is_gfx1250() -> bool:
    try:
        from aiter.jit.utils.chip_info import get_gfx
        return get_gfx() == "gfx1250"
    except Exception:
        return False


@pytest.mark.skipif(not _is_gfx1250(),
                    reason="flash_attn_func dispatch to fmha_fwd_f16_asm only on gfx1250")
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
def test_fmha_fwd_f16_via_flash_attn_func(head_dim, is_causal):
    device = "cuda"
    torch.manual_seed(0)
    batch, hq, hk, sq, sk = 1, 8, 1, 128, 2048

    # bshd input (flash_attn_func contract); contiguous.
    q, k, v = make_qkv_bshd(layout=0, sq=sq, sk=sk, batch=batch,
                             hq=hq, hk=hk, d=head_dim,
                             dtype=torch.bfloat16, device=device)
    scale = 1.0 / math.sqrt(head_dim)
    sink_ptr = _d64_sink(hq, device) if head_dim == 64 else None

    # Direct ops-layer
    out_direct, lse_direct = aiter.fmha_fwd_f16_asm(
        q, k, v, scale, is_causal, True, sink=sink_ptr,
    )

    # Through public API
    result = aiter.flash_attn_func(
        q, k, v,
        softmax_scale=scale,
        causal=is_causal,
        return_lse=True,
        sink_ptr=sink_ptr,
    )
    out_via, lse_via = result[0], result[1]

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
    device = "cuda"
    torch.manual_seed(0)

    # perf_d64 / perf_d128 in run.sh: batch=2 kv_head_num=8 gqa=8 -> hq=64
    sq, batch, hq, hk, sk = 8192, 2, 64, 8, 8192
    q, k, v = make_qkv_bshd(layout=2, sq=sq, sk=sk, batch=batch,
                             hq=hq, hk=hk, d=head_dim,
                             dtype=torch.bfloat16, device=device)
    scale = 1.0 / math.sqrt(head_dim)
    sink  = _d64_sink(hq, device) if head_dim == 64 else None

    _, us = run_perftest(
        aiter.fmha_fwd_f16_asm,
        q, k, v,
        scale, is_causal, False,
        num_iters=10, num_warmup=2,
        sink=sink,
    )
    flops = 2.0 * batch * hq * sq * sk * (2 * head_dim)
    if is_causal:
        flops /= 2.0
    tflops = flops / (us * 1e-6) / 1e12
    print(f"[perf] d={head_dim} causal={is_causal}: {us:.1f}us, {tflops:.2f} TFLOPS")


# ---------------------------------------------------------------------------
# __main__: CLI single-shape runner
# ---------------------------------------------------------------------------
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Run aiter.fmha_fwd_f16_asm on a single shape and dump kernel args.",
)
parser.add_argument("-b",  "--batch",      type=int, default=1,
                    help="batch size (default 1)")
parser.add_argument("-n",  "--q_head_num", type=int, default=8,
                    help="q_head_num (default 8)")
parser.add_argument("-kn", "--kv_head_num", type=int, default=1,
                    help="kv_head_num (default 1, must divide q_head_num)")
parser.add_argument("-q",  "--seqlen_q",   type=int, default=128,
                    help="q seq length (default 128)")
parser.add_argument("-k",  "--seqlen_k",   type=int, default=2048,
                    help="kv seq length (default 2048)")
parser.add_argument("-d",  "--head_dim",   type=int, choices=[64, 128], default=128,
                    help="head dim, 64 or 128 (default 128)")
parser.add_argument("-c",  "--causal",     action="store_true",
                    help="enable causal mask")
parser.add_argument("-l",  "--layout",     type=int, choices=[0, 1, 2], default=0,
                    help="input memory layout: 0=bshd 1=bhsd 2=sbhd (default 0)\n"
                         "(API always sees bshd shape; non-zero layout returns a\n"
                         "non-contiguous bshd view of the underlying memory)")
parser.add_argument("--ref",  action="store_true",
                    help="also run PyTorch reference and print max diff")
parser.add_argument("--perf", action="store_true",
                    help="run perf benchmark for this shape (10 iters, 2 warmup)")

if __name__ == "__main__":
    args = parser.parse_args()

    device = "cuda"
    torch.manual_seed(0)

    b, hq, hk = args.batch, args.q_head_num, args.kv_head_num
    sq, sk, d = args.seqlen_q, args.seqlen_k, args.head_dim
    causal = args.causal
    assert hq % hk == 0, "q_head_num must be a multiple of kv_head_num"
    print(f"Shape: b={b} hq={hq} hk={hk} sq={sq} sk={sk} d={d} causal={causal} "
          f"layout={args.layout}", flush=True)

    q, k, v = make_qkv_bshd(layout=args.layout, sq=sq, sk=sk, batch=b,
                             hq=hq, hk=hk, d=d,
                             dtype=torch.bfloat16, device=device)
    scale = 1.0 / math.sqrt(d)
    sink  = _d64_sink(hq, device) if d == 64 else None
    torch.cuda.synchronize()

    import time as _t
    t0 = _t.time()
    out_kernel, lse_asm = aiter.fmha_fwd_f16_asm(
        q, k, v, scale, causal, True, sink=sink,
    )
    torch.cuda.synchronize()
    print(f"asm time: {(_t.time()-t0)*1000:.2f} ms", flush=True)
    print(f"out.shape={tuple(out_kernel.shape)}  lse.shape={tuple(lse_asm.shape)}", flush=True)

    if args.ref:
        # ref takes bshd directly (matches kernel layout).
        out_ref, lse_ref = _ref_attn(q, k, v, is_causal=causal, sink=sink)
        # cast asm output to fp32 to avoid bf16 element-wise hang in some ROCm builds
        out_kernel_f = out_kernel.float()
        out_ref_f    = out_ref.float()
        diff_o = (out_kernel_f - out_ref_f).abs().max().item()
        diff_l = (lse_asm - lse_ref).abs().max().item()
        # Pass criterion (bf16 attention conventional thresholds):
        #   |dO|   <= 2e-2   |dLSE| <= 2e-2
        ok_o = diff_o <= 2e-2
        ok_l = diff_l <= 2e-2
        print(f"ref:  max|dO|={diff_o:.4f} {'OK' if ok_o else 'FAIL'}   "
              f"max|dLSE|={diff_l:.4f} {'OK' if ok_l else 'FAIL'}",
              flush=True)
        if not (ok_o and ok_l):
            import sys
            sys.exit(1)

    if args.perf:
        _, us = run_perftest(
            aiter.fmha_fwd_f16_asm,
            q, k, v, scale, causal, False,
            num_iters=10, num_warmup=2,
            sink=sink,
        )
        flops = 2.0 * b * hq * sq * sk * (2 * d)
        if causal:
            flops /= 2.0
        tflops = flops / (us * 1e-6) / 1e12
        print(f"perf: {us:.1f} us  ({tflops:.2f} TFLOPS)", flush=True)
