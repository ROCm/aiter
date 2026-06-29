# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness + performance tests for fmha_fwd_with_sink_asm (BF16 ASM, gfx1250).

Public API:    aiter.flash_attn_func          (preferred)
Ops layer:     aiter.fmha_fwd_with_sink_asm         (low-level, ~v3 style)


Layout convention used in tests
--------------------------------
The aiter API only accepts bshd shape ([b, s, h, d]).  To exercise the
kernel's ability to follow strides for sbhd / bhsd memory layouts, the
test allocates qkv in the chosen `layout` and `permute()`s to bshd shape
WITHOUT calling `.contiguous()` — the resulting tensors are bshd-shaped
non-contiguous views whose `.stride()` reflects the underlying memory.

Sink convention
---------------
`sink` ([q_head_num] fp32) is passed to the kernel verbatim -- it is the
per-Q-head logit value the kernel consumes directly (no host-side scaling).
This matches aiter's CK convention (test_mha_common.attention_ref): the sink
is an extra "virtual KV token" with a zero value vector, whose score is the
sink logit in the SAME scaled domain as Q·K^T * softmax_scale.

Sink mechanism (zero-value virtual KV column):
  After computing standard softmax numerators/denominators, the sink only
  adds to the softmax denominator (contributes 0 to the output):
      new_max    = max(max_scores, sink)
      sink_term  = exp(sink - new_max)
      denom      = denom * rescale + sink_term
  where max_scores / scores are already in the scaled (softmax_scale) domain.

This is a standalone script (no pytest): it sweeps a set of shapes, runs the
kernel against a PyTorch reference and/or a perf benchmark, and prints a
markdown summary table — mirroring the style of op_tests/test_quant.py.
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Optional

import pandas as pd
import torch

import aiter

from aiter.test_common import benchmark, checkAllclose
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx

# from aiter.test_mha_common import (
#    attention_ref,
#
# )  # noqa: F401  (kept for easy swap-back; see doc-block below)


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
# ---------------------------------------------------------------------------


def _ref_attn(q, k, v, *, is_causal: bool, sink: "Optional[torch.Tensor]" = None):
    """bshd-in / bshd-out attention reference, sink optional.  Pure-einsum
    fp32 implementation; lse is returned in fp32 (matches kernel's output).

    Math:  scores = (Q @ K^T) * scale,   scale = 1/sqrt(d),
           denom  = sum(exp(scores - max)) [+ exp(sink - max)],
           out    = (exp(scores - max) / denom) @ V,
           lse    = max + log(denom).
    sink (optional): [hq] fp32, a per-Q-head logit in the SAME (scaled) domain
                     as `scores` -- it is passed to the kernel verbatim (no
                     host-side scaling), matching aiter's CK convention
                     (test_mha_common.attention_ref): sink is an extra
                     zero-value KV column appended to the scaled scores.
    """
    b, sq, hq, d = q.shape
    _, sk, hk, _ = k.shape
    if hq != hk:
        k = k.repeat_interleave(hq // hk, dim=2)
        v = v.repeat_interleave(hq // hk, dim=2)
    qf, kf, vf = q.float(), k.float(), v.float()
    scale = 1.0 / math.sqrt(d)
    # Work entirely in the scaled-logit domain so the sink (which the kernel
    # consumes verbatim) lines up with the scores.
    scores = torch.einsum("bshd,bkhd->bhsk", qf, kf) * scale
    if is_causal:
        m = torch.triu(
            torch.ones(sq, sk, dtype=torch.bool, device=q.device), sk - sq + 1
        )
        scores = scores.masked_fill(m, float("-inf"))
    max_attn, _ = scores.max(dim=-1)
    if sink is not None:
        sink_bhs = sink.float()[None, :, None].expand(b, hq, sq)
        max_total = torch.maximum(max_attn, sink_bhs)
    else:
        max_total = max_attn
    denom_real = torch.exp(scores - max_total.unsqueeze(-1)).sum(dim=-1)
    if sink is not None:
        sink_term = torch.exp(sink_bhs - max_total)
        denom_total = denom_real + sink_term
    else:
        denom_total = denom_real
    probs = torch.exp(scores - max_total.unsqueeze(-1)) / denom_total.unsqueeze(-1)
    out = torch.einsum("bhsk,bkhd->bshd", probs, vf).to(q.dtype)
    lse = torch.log(denom_total) + max_total
    return out, lse


def _nrms(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Normalized RMS error on fp32 CPU tensors (avoids bf16 GPU element-wise hang).

    Definition matches op_tests/test_mha_mxfp8.py:
        nrms = sqrt(sum((|a-b| / max(|b|, eps))^2)) /
               (sqrt(numel) * max(|a|.max, |b|.max, eps))

    eps must be chosen above the dtype's effective resolution; otherwise the
    `1 / max(|b|, eps)` term blows up for the (legitimately) near-zero
    elements common in softmax outputs, producing huge nrms values that have
    nothing to do with the kernel actually being wrong.  For bf16 (relative
    precision ~3.9e-3) we use eps=1e-3.
    """
    a32 = actual.detach().float().cpu()
    b32 = expected.detach().float().cpu()
    abs_diff = (a32 - b32).abs()
    eps = 1e-3
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

    via = "ops"        → low-level aiter.fmha_fwd_with_sink_asm
    via = "public"     → public aiter.flash_attn_func (dispatcher → asm path)
    """
    if via == "ops":
        return aiter.fmha_fwd_with_sink_asm(
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
# q/k/v perf-init helper
# ---------------------------------------------------------------------------

# Initialization patterns for q/k/v buffers.
#   "randn"     : standard normal (default; exercises real attention math).
#   "const0.25" : fill every element with 0.25 — matches the cpp perf-test
#                 init pattern (`init_pattern=10`) used in cpp perf runs.
_PERF_INITS = ["randn", "const0.25"]


def _make_qkv(init: str, *, layout, sq, sk, batch, hq, hk, d, dtype, device):
    """Allocate (q, k, v) in `layout` memory with bshd-shaped views, using the
    requested init pattern.  See `make_qkv_bshd` for layout semantics."""
    q, k, v = make_qkv_bshd(
        layout=layout,
        sq=sq,
        sk=sk,
        batch=batch,
        hq=hq,
        hk=hk,
        d=d,
        dtype=dtype,
        device=device,
    )
    if init == "const0.25":
        # In-place `.fill_()` is layout-agnostic so this works for
        # non-contiguous views.
        q.fill_(0.25)
        k.fill_(0.25)
        v.fill_(0.25)
    elif init != "randn":
        raise ValueError(f"unknown init pattern: {init!r}")
    return q, k, v


# ---------------------------------------------------------------------------
# Shape tables
# ---------------------------------------------------------------------------

# KV-length constraint (mask=0 only): the non-causal (mask=0) kernels only
# support sk (kv_seqlen) that is a multiple of 256.  Non-causal cases with a
# non-256-aligned sk are skipped automatically in the sweep.
# hq=64 for all cases; hk=8 for D64 and hk=4 for D128 (GQA ratios 8 / 16).
_CORRECTNESS_SHAPES = [
    # ----- Small shapes ---------------------------------------------------
    (64, 64, 8, 128, 2048, 1),  # D64  aligned
    (64, 64, 8, 128, 2048, 2),
    (64, 64, 8, 130, 2048, 1),  # D64  q-unaligned (sq not mult of 128)
    (64, 64, 8, 128, 2300, 1),  # D64  kv-unaligned (sk not mult of 256) -> causal only
    (128, 64, 4, 128, 2048, 1),  # D128 aligned
    (128, 64, 4, 128, 2048, 2),
    (128, 64, 4, 130, 2048, 1),  # D128 q-unaligned
    (128, 64, 4, 128, 2300, 1),  # D128 kv-unaligned -> causal only
    (64, 64, 8, 8192, 8192, 1),  # D64  perf-sized, aligned
    (128, 64, 4, 4096, 4096, 1),  # D128 perf-sized, aligned
]

# (head_dim, seqlen) perf shapes; sq == sk.  batch=1, hq=64, hk=8 (D64) / 4 (D128).
_PERF_SHAPES = [
    (64, 1024),
    (64, 4096),
    (64, 8192),
    (64, 16384),
    (64, 32768),
    (128, 1024),
    (128, 2048),
    (128, 4096),
    (128, 8192),
    (128, 16384),
]


# ---------------------------------------------------------------------------
# Benchmark entry: correctness (+ optional perf) for one shape.
#
# Decorated with @benchmark() (aiter.test_common), which logs the call args
# and merges the returned metric dict so each invocation yields one DataFrame
# row — mirroring op_tests/test_quant.py.
# ---------------------------------------------------------------------------


@benchmark()
def test_fmha_fwd_with_sink_asm(
    head_dim,
    hq,
    hk,
    sq,
    sk,
    batch,
    is_causal,
    layout,
    init,
    do_ref,
    do_perf,
):
    device = "cuda"
    torch.manual_seed(0)

    q, k, v = _make_qkv(
        init,
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
    # D64 -> non-zero sink (exercises ENABLE_SINK code path).
    # D128 -> no sink (kernel ignores it).
    sink = _d64_sink(hq, device) if head_dim == 64 else None

    ret = {}

    if do_ref:
        # Public-API path (dispatcher → asm) and low-level ops path.
        out_pub, lse_pub = run_kernel(
            q, k, v, scale=scale, is_causal=is_causal, sink=sink, via="public"
        )
        out_ops, lse_ops = run_kernel(
            q, k, v, scale=scale, is_causal=is_causal, sink=sink, via="ops"
        )
        out_ref, lse_ref = run_ref(q, k, v, is_causal=is_causal, sink=sink)

        # checkAllclose returns the mismatch ratio (0.0 == all-close) and logs
        # a nicely-formatted diff.  Cast to fp32 CPU first to avoid the
        # gfx1250 + ROCm bf16 element-wise deadlock noted in the module docs.
        msg = f"d={head_dim} c={is_causal} layout={layout}"
        err_o = checkAllclose(
            out_ref.float().cpu(),
            out_pub.float().cpu(),
            rtol=1e-2,
            atol=1e-2,
            msg=f"O {msg}",
        )
        err_l = checkAllclose(
            lse_ref.float().cpu(),
            lse_pub.float().cpu(),
            rtol=1e-2,
            atol=1e-2,
            msg=f"LSE {msg}",
        )
        ret["err(O)"] = err_o
        ret["err(LSE)"] = err_l
        ret["nrms(O)"] = _nrms(out_pub, out_ref)
        # Public dispatcher must match the low-level ops call bit-for-bit
        # (same kernel, same args).
        ret["dO(pub-ops)"] = (out_pub.float() - out_ops.float()).abs().max().item()
        ret["dLSE(pub-ops)"] = (lse_pub.float() - lse_ops.float()).abs().max().item()

    if do_perf:
        us = _bench(
            aiter.fmha_fwd_with_sink_asm,
            q,
            k,
            v,
            scale,
            is_causal,
            False,
            sink=sink,
            num_iters=20,
            num_warmup=10,
        )
        flops = 2.0 * batch * hq * sq * sk * (2 * head_dim)
        if is_causal:
            flops /= 2.0
        ret["us"] = us
        ret["tflops"] = flops / (us * 1e-6) / 1e12

    return ret


# ---------------------------------------------------------------------------
# Multi-GPU dispatch check (optional; needs >=2 ROCm GPUs).
#
# Regression for: `flash_attn_func` must launch on q.device(), not on the
# Python thread's current_device.
# ---------------------------------------------------------------------------


def run_multi_gpu_dispatch(head_dim):
    if torch.cuda.device_count() < 2:
        print("[multi-gpu] skip: needs >=2 ROCm GPUs")
        return
    torch.manual_seed(0)
    batch, hq, hk, sq, sk = 1, 4, 1, 128, 1024
    scale = 1.0 / math.sqrt(head_dim)
    dev_q = "cuda:1"  # tensors live here
    dev_other = 0  # caller's current_device when we invoke the API

    with torch.cuda.device(dev_q):
        q1, k1, v1 = make_qkv_bshd(
            layout=0,
            sq=sq,
            sk=sk,
            batch=batch,
            hq=hq,
            hk=hk,
            d=head_dim,
            dtype=torch.bfloat16,
            device=dev_q,
        )
        sink1 = _d64_sink(hq, dev_q) if head_dim == 64 else None
        out_baseline, lse_baseline = run_kernel(
            q1, k1, v1, scale=scale, is_causal=True, sink=sink1, via="public"
        )
    out_baseline = out_baseline.clone()
    lse_baseline = lse_baseline.clone()

    with torch.cuda.device(dev_other):
        out_xdev, lse_xdev = run_kernel(
            q1, k1, v1, scale=scale, is_causal=True, sink=sink1, via="public"
        )

    on_q = out_xdev.device == q1.device and lse_xdev.device == q1.device
    do = (out_xdev.float() - out_baseline.float()).abs().max().item()
    dl = (lse_xdev.float() - lse_baseline.float()).abs().max().item()
    print(
        f"[multi-gpu d={head_dim}] out.device={out_xdev.device} "
        f"on_q_device={on_q} max|dO|={do} max|dLSE|={dl}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Sweep aiter.fmha_fwd_with_sink_asm shapes and print a summary.",
)
parser.add_argument(
    "-d",
    "--head_dim",
    type=int,
    nargs="*",
    choices=[64, 128],
    default=[64, 128],
    help="head dim(s) to test (default: 64 128)",
)
parser.add_argument(
    "-c",
    "--causal",
    type=int,
    nargs="*",
    choices=[0, 1],
    default=[0, 1],
    help="causal mode(s): 0=non-causal 1=causal (default: 0 1)",
)
parser.add_argument(
    "-l",
    "--layout",
    type=int,
    nargs="*",
    choices=[0, 1, 2],
    default=[2],
    help="input memory layout(s): 0=bshd 1=bhsd 2=sbhd (default: 2)\n"
    "(API always sees bshd shape; non-zero layout returns a\n"
    "non-contiguous bshd view of the underlying memory)",
)
parser.add_argument(
    "--init",
    type=str,
    nargs="*",
    choices=_PERF_INITS,
    default=["randn"],
    help="q/k/v init pattern(s): 'randn' (default) or 'const0.25'",
)
parser.add_argument(
    "--no-ref",
    action="store_true",
    help="skip the correctness sweep (PyTorch reference comparison)",
)
parser.add_argument(
    "--no-perf",
    action="store_true",
    help="skip the perf sweep",
)
parser.add_argument(
    "--multi-gpu",
    action="store_true",
    help="also run the multi-GPU dispatch check (needs >=2 ROCm GPUs)",
)


if __name__ == "__main__":
    if get_gfx() not in ["gfx1250"]:
        print(
            "fmha_fwd_with_sink_asm ASM kernels are only shipped for gfx1250 "
            "(hsa/gfx1250/fmha_fwd_bf16/*.co); no GPU or a different arch — skip."
        )
        sys.exit(0)

    args = parser.parse_args()
    causal_modes = [bool(c) for c in args.causal]

    # ----- Correctness sweep (ref + public-vs-ops bit-exactness) ----------
    if not args.no_ref:
        rows = []
        for head_dim, hq, hk, sq, sk, batch in _CORRECTNESS_SHAPES:
            if head_dim not in args.head_dim:
                continue
            for is_causal in causal_modes:
                # mask=0 (non-causal) kernels require sk % 256 == 0.
                if not is_causal and sk % 256 != 0:
                    continue
                for layout in args.layout:
                    for init in args.init:
                        rows.append(
                            test_fmha_fwd_with_sink_asm(
                                head_dim,
                                hq,
                                hk,
                                sq,
                                sk,
                                batch,
                                is_causal,
                                layout,
                                init,
                                do_ref=True,
                                do_perf=False,
                            )
                        )
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "fmha_fwd_with_sink_asm correctness summary (markdown):\n%s",
            df.to_markdown(index=False),
        )

    # ----- Perf sweep -----------------------------------------------------
    if not args.no_perf:
        rows = []
        for head_dim, seqlen in _PERF_SHAPES:
            if head_dim not in args.head_dim:
                continue
            batch, hq = 1, 64
            hk = 8 if head_dim == 64 else 4
            for is_causal in causal_modes:
                # perf always covers both init patterns (randn + const0.25).
                for init in _PERF_INITS:
                    rows.append(
                        test_fmha_fwd_with_sink_asm(
                            head_dim,
                            hq,
                            hk,
                            seqlen,
                            seqlen,
                            batch,
                            is_causal,
                            2,  # perf uses sbhd layout
                            init,
                            do_ref=False,
                            do_perf=True,
                        )
                    )
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "fmha_fwd_with_sink_asm perf summary (markdown):\n%s",
            df.to_markdown(index=False),
        )

    # ----- Optional multi-GPU dispatch check ------------------------------
    if args.multi_gpu:
        for head_dim in args.head_dim:
            run_multi_gpu_dispatch(head_dim)
