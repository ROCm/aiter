# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness + performance tests for fmha_fwd_with_sink_asm (BF16 ASM, gfx1250).

Public API:  aiter.flash_attn_func              (the path the model calls)
Ops layer:   aiter.fmha_fwd_with_sink_asm       (low-level)

Built to the aiter op-test standard (see .claude/skills/aiter-op-test): mirror
test_quant.py — @benchmark + run_perftest candidate loop, a torch reference,
per-candidate us / TFLOPS / TB/s / err, a markdown summary table per test
function, and a __main__ guard so the module is importable.

Sink convention
---------------
`sink` ([q_head_num] fp32) is a per-Q-head logit in the SAME scaled domain as
Q·K^T * softmax_scale; it acts as a zero-value virtual KV column and is passed
to the kernel verbatim (no host-side scaling).  D64 kernels read it; D128
kernels ignore it (pass None).

Layout: the API only accepts bshd shape ([b, s, h, d]).  To exercise the
kernel's stride handling for sbhd / bhsd memory, qkv are allocated in the
chosen `layout` and `permute()`d to a bshd-shaped non-contiguous view.
"""

import argparse
import itertools
import math

import aiter
import pandas as pd
import torch
from aiter import dtypes
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx

torch.set_default_device("cuda")

# Every card these ASM kernels are built/validated for.  The .co files only
# ship for gfx1250 (hsa/gfx1250/fmha_fwd_bf16/*.co); on any other arch the
# kernel launch raises 'no kernel for arch=...'.
SUPPORTED_GFX = ["gfx1250"]


# ---------------------------------------------------------------------------
# Reference (fp32 math, cast back).  Not timed, not in the table.
# ---------------------------------------------------------------------------


def run_torch(q, k, v, *, is_causal, sink=None):
    """bshd-in / bshd-out attention reference, sink optional.

    scores = (Q·K^T) * scale,  scale = 1/sqrt(d);  lse returned in fp32.
    sink (optional): [hq] fp32 per-head logit in the scaled domain (a zero-value
    KV column appended to the scaled scores).
    """
    b, sq, hq, d = q.shape
    _, sk, hk, _ = k.shape
    if hq != hk:
        k = k.repeat_interleave(hq // hk, dim=2)
        v = v.repeat_interleave(hq // hk, dim=2)
    qf, kf, vf = q.float(), k.float(), v.float()
    scale = 1.0 / math.sqrt(d)
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
    denom = torch.exp(scores - max_total.unsqueeze(-1)).sum(dim=-1)
    if sink is not None:
        denom = denom + torch.exp(sink_bhs - max_total)
    probs = torch.exp(scores - max_total.unsqueeze(-1)) / denom.unsqueeze(-1)
    out = torch.einsum("bhsk,bkhd->bshd", probs, vf).to(q.dtype)
    lse = torch.log(denom) + max_total
    return out, lse


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------


def make_qkv_bshd(layout, sq, sk, batch, hq, hk, d, init="randn", dtype=dtypes.bf16):
    """Allocate (q, k, v) in `layout` memory, return bshd-shaped views.

    layout: 0 = bshd (contiguous), 1 = bhsd view, 2 = sbhd view.  The kernel
    reads strides directly so non-contiguous bshd views (layout 1/2) are valid.
    init:   "randn" (random normal) or "const0.25" (every element = 0.25).
    """
    if layout == 0:
        q = torch.randn(batch, sq, hq, d, dtype=dtype)
        k = torch.randn(batch, sk, hk, d, dtype=dtype)
        v = torch.randn(batch, sk, hk, d, dtype=dtype)
    elif layout == 1:
        q = torch.randn(batch, hq, sq, d, dtype=dtype).permute(0, 2, 1, 3)
        k = torch.randn(batch, hk, sk, d, dtype=dtype).permute(0, 2, 1, 3)
        v = torch.randn(batch, hk, sk, d, dtype=dtype).permute(0, 2, 1, 3)
    elif layout == 2:
        q = torch.randn(sq, batch, hq, d, dtype=dtype).permute(1, 0, 2, 3)
        k = torch.randn(sk, batch, hk, d, dtype=dtype).permute(1, 0, 2, 3)
        v = torch.randn(sk, batch, hk, d, dtype=dtype).permute(1, 0, 2, 3)
    else:
        raise ValueError(f"unsupported layout={layout}")
    if init == "const0.25":
        # In-place fill is layout-agnostic (works on non-contiguous views).
        q.fill_(0.25)
        k.fill_(0.25)
        v.fill_(0.25)
    elif init != "randn":
        raise ValueError(f"unknown init pattern: {init!r}")
    return q, k, v


def _d64_sink(hq):
    """Per-head sink logits in [0.5, 2.0] (scaled domain), varied across heads."""
    return torch.linspace(0.5, 2.0, hq, dtype=dtypes.fp32)


def run_kernel(q, k, v, *, scale, is_causal, sink=None, via="public"):
    """via="public" → aiter.flash_attn_func (the model path); "ops" → low-level."""
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
    if via == "ops":
        return aiter.fmha_fwd_with_sink_asm(q, k, v, scale, is_causal, True, sink=sink)
    raise ValueError(f"unknown via={via!r}")


def _flops_bytes(batch, hq, hk, sq, sk, d, is_causal, esz):
    """Attention roofline numerators: 2 GEMMs (QK^T, PV), HBM traffic q+k+v+o."""
    flops = 4.0 * batch * hq * sq * sk * d  # 2*(2*M*N*K) over the two matmuls
    if is_causal:
        flops /= 2.0
    nbytes = (
        2 * batch * sq * hq * d  # q read + o write
        + 2 * batch * sk * hk * d  # k + v read
    ) * esz
    return flops, nbytes


# ---------------------------------------------------------------------------
# Shape tables
# ---------------------------------------------------------------------------

# Correctness shapes (torch reference is feasible here).  hq=64; hk=8 for D64
# and hk=4 for D128 (GQA ratios 8 / 16).  Non-causal (mask=0) kernels require
# sk % 256 == 0 — non-aligned sk rows are causal-only (filtered in the sweep).
# (head_dim, hq, hk, sq, sk, batch)
_CORRECTNESS_SHAPES = [
    (64, 64, 8, 128, 2048, 1),  # D64  aligned
    (64, 64, 8, 128, 2048, 2),
    (64, 64, 8, 130, 2048, 1),  # D64  q-unaligned (sq not mult of 128)
    (64, 64, 8, 128, 2300, 1),  # D64  kv-unaligned (sk not mult of 256) -> causal
    (128, 64, 4, 128, 2048, 1),  # D128 aligned
    (128, 64, 4, 128, 2048, 2),
    (128, 64, 4, 130, 2048, 1),  # D128 q-unaligned
    (128, 64, 4, 128, 2300, 1),  # D128 kv-unaligned -> causal
    (64, 64, 8, 8192, 8192, 1),  # D64  perf-sized, aligned
    (128, 64, 4, 4096, 4096, 1),  # D128 perf-sized, aligned
]

# Perf-only shapes; sq == sk.  hq=64, hk=8 (D64) / 4 (D128), batch=1, sbhd.
# The torch reference is O(s^2) in memory (e.g. 32768 -> ~256 GB of scores) so
# these are timed but NOT correctness-checked here (use _CORRECTNESS_SHAPES).
# (head_dim, seqlen)
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
# Test functions (one markdown table each).  @benchmark logs the call args as
# the table's left columns and merges the returned metric dict.
# ---------------------------------------------------------------------------


@benchmark()
def test_fmha_fwd_with_sink_asm(
    head_dim, hq, hk, sq, sk, batch, is_causal, layout, init
):
    torch.manual_seed(0)
    q, k, v = make_qkv_bshd(layout, sq, sk, batch, hq, hk, head_dim, init=init)
    scale = 1.0 / math.sqrt(head_dim)
    # D64 -> non-zero sink (exercises ENABLE_SINK); D128 -> kernel ignores it.
    sink = _d64_sink(hq) if head_dim == 64 else None

    ref_out, ref_lse = run_torch(q, k, v, is_causal=is_causal, sink=sink)

    flops, nbytes = _flops_bytes(
        batch, hq, hk, sq, sk, head_dim, is_causal, q.element_size()
    )

    # The model calls the public dispatcher (flash_attn_func) → asm path.
    candidates = {
        "asm": lambda: run_kernel(
            q, k, v, scale=scale, is_causal=is_causal, sink=sink, via="public"
        )
    }

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        (out, lse), us = run_perftest(fn)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err(O)"] = checkAllclose(
            ref_out.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name} O d={head_dim} c={is_causal}",
        )
        ret[f"{name} err(LSE)"] = checkAllclose(
            ref_lse.to(dtypes.fp32),
            lse.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name} LSE d={head_dim} c={is_causal}",
        )
    return ret


@benchmark()
def test_fmha_fwd_with_sink_asm_perf(head_dim, hq, hk, sq, sk, batch, is_causal, init):
    torch.manual_seed(0)
    q, k, v = make_qkv_bshd(2, sq, sk, batch, hq, hk, head_dim, init=init)
    scale = 1.0 / math.sqrt(head_dim)
    sink = _d64_sink(hq) if head_dim == 64 else None

    flops, nbytes = _flops_bytes(
        batch, hq, hk, sq, sk, head_dim, is_causal, q.element_size()
    )

    candidates = {
        "asm": lambda: run_kernel(
            q, k, v, scale=scale, is_causal=is_causal, sink=sink, via="public"
        )
    }

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        _, us = run_perftest(fn)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "fmha_fwd_with_sink_asm unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
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
        help="memory layout(s): 0=bshd 1=bhsd 2=sbhd (default: 2)",
    )
    parser.add_argument(
        "--init",
        type=str,
        nargs="*",
        choices=["randn", "const0.25"],
        default=["randn", "const0.25"],
        help="q/k/v init pattern(s) (default: randn const0.25)",
    )
    args = parser.parse_args()
    causal_modes = [bool(c) for c in args.causal]

    # ---- correctness + perf table ----
    df = []
    for head_dim, hq, hk, sq, sk, batch in _CORRECTNESS_SHAPES:
        if head_dim not in args.head_dim:
            continue
        for is_causal, layout, init in itertools.product(
            causal_modes, args.layout, args.init
        ):
            if not is_causal and sk % 256 != 0:  # mask=0 kernel needs sk%256==0
                continue
            df.append(
                test_fmha_fwd_with_sink_asm(
                    head_dim, hq, hk, sq, sk, batch, is_causal, layout, init
                )
            )
    df = pd.DataFrame(df)
    aiter.logger.info(
        "fmha_fwd_with_sink_asm correctness summary (markdown):\n%s",
        df.to_markdown(index=False),
    )

    # ---- perf-only table (large shapes; ref infeasible) ----
    df = []
    for head_dim, seqlen in _PERF_SHAPES:
        if head_dim not in args.head_dim:
            continue
        hk = 8 if head_dim == 64 else 4
        for is_causal, init in itertools.product(causal_modes, args.init):
            df.append(
                test_fmha_fwd_with_sink_asm_perf(
                    head_dim, 64, hk, seqlen, seqlen, 1, is_causal, init
                )
            )
    df = pd.DataFrame(df)
    aiter.logger.info(
        "fmha_fwd_with_sink_asm perf summary (markdown):\n%s",
        df.to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
