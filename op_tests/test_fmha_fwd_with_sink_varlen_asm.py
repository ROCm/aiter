# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness + perf tests for fmha_fwd_with_sink_varlen_asm (BF16 ASM, gfx1250).

Public API:  aiter.flash_attn_varlen_func          (the path the model calls)
Ops layer:   aiter.fmha_fwd_with_sink_varlen_asm    (low-level, packed/varlen)

Built to the aiter op-test standard (see .claude/skills/aiter-op-test): mirror
test_quant.py — @benchmark + run_perftest candidate loop, a torch reference,
per-candidate us / TFLOPS / TB/s / err, a markdown summary table per test
function, and a __main__ guard so the module is importable.

Layout (packed THD; batch folded into the token axis):
    q   : (total_q, nheads,   hdim_q)
    k   : (total_k, nheads_k, hdim_q)
    v   : (total_k, nheads_k, hdim_v)
    out : (total_q, nheads,   hdim_v)
    cu_seqlens_q / cu_seqlens_k : int32 [batch+1] cumulative

Sink convention (same as the fixed-batch path): `sink` ([q_head_num] fp32) is a
per-Q-head logit in the scaled domain, a zero-value virtual KV column passed
verbatim.  D64 kernels read it; D128 kernels ignore it (pass None).

KV-length constraint (mask=0 only): non-causal kernels require per-sequence
kv_seqlen that is a multiple of 256.
"""

import argparse
import itertools
import math
from typing import List

import aiter
import pandas as pd
import torch
from aiter import dtypes
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx

torch.set_default_device("cuda")

# .co files only ship for gfx1250 (hsa/gfx1250/fmha_fwd_bf16_varlen/*.co).
SUPPORTED_GFX = ["gfx1250"]


# ---------------------------------------------------------------------------
# Reference (fp32 math, cast back).  Not timed, not in the table.
# ---------------------------------------------------------------------------


def _attn_one(q, k, v, *, is_causal, sink):
    """Single-sequence attention reference (no batch dim).

    q: (sq, hq, d)  k: (sk, hk, d)  v: (sk, hk, dv) -> out (sq, hq, dv), lse (sq, hq).
    """
    sq, hq, d = q.shape
    sk, hk, _ = k.shape
    if hq != hk:
        k = k.repeat_interleave(hq // hk, dim=1)
        v = v.repeat_interleave(hq // hk, dim=1)
    qf, kf, vf = q.float(), k.float(), v.float()
    scale = 1.0 / math.sqrt(d)
    scores = torch.einsum("qhd,khd->hqk", qf, kf) * scale
    if is_causal:
        row = torch.arange(sq, device=q.device)[:, None]
        col = torch.arange(sk, device=q.device)[None, :]
        masked = col > (row + (sk - sq))  # bottom-right aligned causal mask
        scores = scores.masked_fill(masked[None], float("-inf"))
    max_attn = scores.max(dim=-1).values
    if sink is not None:
        sink_hs = sink.float()[:, None].expand(hq, sq)
        max_total = torch.maximum(max_attn, sink_hs)
    else:
        max_total = max_attn
    denom = torch.exp(scores - max_total.unsqueeze(-1)).sum(dim=-1)
    if sink is not None:
        denom = denom + torch.exp(sink_hs - max_total)
    probs = torch.exp(scores - max_total.unsqueeze(-1)) / denom.unsqueeze(-1)
    out = torch.einsum("hqk,khd->qhd", probs, vf).to(q.dtype)
    lse = (torch.log(denom) + max_total).transpose(0, 1)
    return out, lse


def run_torch(q, k, v, cu_q, cu_k, *, is_causal, sink):
    """Packed-THD reference: loop over batches, slice via cu_seqlens."""
    total_q, hq, _ = q.shape
    dv = v.shape[-1]
    batch = cu_q.numel() - 1
    out = torch.empty((total_q, hq, dv), dtype=q.dtype, device=q.device)
    lse = torch.empty((total_q, hq), dtype=dtypes.fp32, device=q.device)
    cuq, cuk = cu_q.tolist(), cu_k.tolist()
    for b in range(batch):
        q0, q1 = cuq[b], cuq[b + 1]
        k0, k1 = cuk[b], cuk[b + 1]
        if q1 == q0:
            continue
        ob, lb = _attn_one(q[q0:q1], k[k0:k1], v[k0:k1], is_causal=is_causal, sink=sink)
        out[q0:q1] = ob
        lse[q0:q1] = lb
    return out, lse


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------


def make_varlen_packed(seqlens: List[int], hq, hk, d, dv, init="randn", seed=0):
    """Build packed THD q/k/v + cu_seqlens for the given per-batch seqlens.

    Equal q/k seqlens per batch (standard varlen self-attention).
    init: "randn" or "const0.25".
    """
    torch.manual_seed(seed)
    cu = torch.tensor(
        [0] + list(torch.tensor(seqlens).cumsum(0).tolist()), dtype=dtypes.i32
    )
    total = int(cu[-1].item())
    q = torch.randn(total, hq, d, dtype=dtypes.bf16)
    k = torch.randn(total, hk, d, dtype=dtypes.bf16)
    v = torch.randn(total, hk, dv, dtype=dtypes.bf16)
    if init == "const0.25":
        q.fill_(0.25)
        k.fill_(0.25)
        v.fill_(0.25)
    elif init != "randn":
        raise ValueError(f"unknown init pattern: {init!r}")
    return q, k, v, cu


def _d64_sink(hq):
    """Per-head sink logits in [0.5, 2.0] (scaled domain), varied across heads."""
    return torch.linspace(0.5, 2.0, hq, dtype=dtypes.fp32)


def run_kernel(
    q, k, v, cu_q, cu_k, max_seqlen_q, *, scale, is_causal, sink=None, via="public"
):
    """Return (out, lse) with lse shaped (total_q, nheads) to match run_torch.

    via="public" → aiter.flash_attn_varlen_func (the model path); lse comes back
                   (nheads, total_q) and is transposed here.
    via="ops"    → aiter.fmha_fwd_with_sink_varlen_asm; lse is (total_q, nheads, 1).
    """
    if via == "public":
        r = aiter.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            max_seqlen_q,
            max_seqlen_q,  # equal q/k seqlens in these tests
            softmax_scale=scale,
            causal=is_causal,
            return_lse=True,
            sink_ptr=sink,
        )
        return r[0], r[1].transpose(0, 1).contiguous()
    if via == "ops":
        out, lse = aiter.fmha_fwd_with_sink_varlen_asm(
            q, k, v, cu_q, cu_k, max_seqlen_q, scale, is_causal, True, sink=sink
        )
        return out, lse.squeeze(-1)
    raise ValueError(f"unknown via={via!r}")


def _flops_bytes(seqlens, hq, hk, d, is_causal, total, esz):
    """Attention roofline numerators summed over the packed batches."""
    flops = sum(4.0 * hq * s * s * d for s in seqlens)  # 2 GEMMs (QK^T, PV)
    if is_causal:
        flops /= 2.0
    nbytes = (2 * total * hq * d + 2 * total * hk * d) * esz  # q+o, k+v
    return flops, nbytes


# ---------------------------------------------------------------------------
# Shape tables
# ---------------------------------------------------------------------------

# Correctness shapes (torch reference feasible).  hq=64; hk=8 (D64) / 4 (D128).
# Non-causal (mask=0) kernels require every kv_seqlen % 256 == 0 (filtered).
# (head_dim, hq, hk, seqlens)
_CORRECTNESS_SHAPES = [
    (64, 64, 8, [256]),
    (128, 64, 4, [256]),
    (64, 64, 8, [128, 256, 384]),  # mixed (some unaligned) -> causal only
    (128, 64, 4, [128, 256, 384]),
    (64, 64, 8, [100, 200, 300]),  # unaligned
    (128, 64, 4, [100, 200, 300]),
    (64, 64, 8, [256, 512]),  # 256-aligned (causal AND mask=0)
    (128, 64, 4, [256, 512]),
    (64, 64, 8, [256, 512, 768]),
    (128, 64, 4, [256, 512, 768]),
    (64, 64, 8, [512, 1024]),
    (128, 64, 4, [512, 1024]),
]

# Perf-only shapes (torch ref O(s^2) infeasible at 16384/32768).
# (head_dim, hq, hk, seqlens)
_VARLEN_PERF_SHAPES = [
    (64, 64, 8, [4096, 4096]),
    (128, 64, 4, [2048, 2048]),
    (128, 64, 4, [16384]),
    (64, 64, 8, [32768]),
]


def _kv_256_aligned(seqlens):
    return all(s % 256 == 0 for s in seqlens)


# ---------------------------------------------------------------------------
# Test functions (one markdown table each).
# ---------------------------------------------------------------------------


@benchmark()
def test_fmha_fwd_with_sink_varlen_asm(head_dim, hq, hk, seqlens, is_causal, init):
    q, k, v, cu = make_varlen_packed(seqlens, hq, hk, head_dim, head_dim, init=init)
    max_seqlen_q = max(seqlens)
    scale = 1.0 / math.sqrt(head_dim)
    sink = _d64_sink(hq) if head_dim == 64 else None

    ref_out, ref_lse = run_torch(q, k, v, cu, cu, is_causal=is_causal, sink=sink)

    total = q.shape[0]
    flops, nbytes = _flops_bytes(
        seqlens, hq, hk, head_dim, is_causal, total, q.element_size()
    )

    # The model calls the public dispatcher (flash_attn_varlen_func) → asm path.
    candidates = {
        "asm": lambda: run_kernel(
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
def test_fmha_fwd_with_sink_varlen_asm_perf(head_dim, hq, hk, seqlens, is_causal, init):
    q, k, v, cu = make_varlen_packed(seqlens, hq, hk, head_dim, head_dim, init=init)
    max_seqlen_q = max(seqlens)
    scale = 1.0 / math.sqrt(head_dim)
    sink = _d64_sink(hq) if head_dim == 64 else None

    total = q.shape[0]
    flops, nbytes = _flops_bytes(
        seqlens, hq, hk, head_dim, is_causal, total, q.element_size()
    )

    candidates = {
        "asm": lambda: run_kernel(
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
            "fmha_fwd_with_sink_varlen_asm unsupported on %s; skipping", get_gfx()
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
    for head_dim, hq, hk, seqlens in _CORRECTNESS_SHAPES:
        if head_dim not in args.head_dim:
            continue
        for is_causal, init in itertools.product(causal_modes, args.init):
            if not is_causal and not _kv_256_aligned(seqlens):
                continue
            df.append(
                test_fmha_fwd_with_sink_varlen_asm(
                    head_dim, hq, hk, seqlens, is_causal, init
                )
            )
    df = pd.DataFrame(df)
    aiter.logger.info(
        "fmha_fwd_with_sink_varlen_asm correctness summary (markdown):\n%s",
        df.to_markdown(index=False),
    )

    # ---- perf-only table (large shapes; ref infeasible) ----
    df = []
    for head_dim, hq, hk, seqlens in _VARLEN_PERF_SHAPES:
        if head_dim not in args.head_dim:
            continue
        for is_causal, init in itertools.product(causal_modes, args.init):
            df.append(
                test_fmha_fwd_with_sink_varlen_asm_perf(
                    head_dim, hq, hk, seqlens, is_causal, init
                )
            )
    df = pd.DataFrame(df)
    aiter.logger.info(
        "fmha_fwd_with_sink_varlen_asm perf summary (markdown):\n%s",
        df.to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
