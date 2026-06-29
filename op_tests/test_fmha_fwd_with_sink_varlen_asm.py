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

KV-length constraint (mask=0 only): the non-causal (mask=0) kernels only
support per-sequence kv_seqlen that is a multiple of 256.

This is a standalone script (no pytest): it sweeps a set of shapes, runs the
kernel against a PyTorch reference and/or a perf benchmark, and prints a
markdown summary table — mirroring the style of op_tests/test_quant.py.
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import List, Optional

import pandas as pd
import torch

import aiter
from aiter.test_common import benchmark, checkAllclose
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nrms(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Normalized RMS error on fp32 CPU tensors (eps=1e-3 for bf16)."""
    a32 = actual.detach().float().cpu()
    b32 = expected.detach().float().cpu()
    abs_diff = (a32 - b32).abs()
    eps = 1e-3
    max_item = max(a32.abs().max().item(), b32.abs().max().item(), eps)
    sq_diff = (abs_diff / b32.abs().clamp(min=eps)).pow(2)
    return (sq_diff.sum().sqrt() / (math.sqrt(b32.numel()) * max_item)).item()


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


# Perf input init patterns (mirrors the fixed-batch perf test's _PERF_INITS):
#   "randn"     : standard normal (default; exercises real attention math).
#   "const0.25" : constant 0.25 fill (matches the cpp init_pattern=10 baseline).
_PERF_INITS = ["randn", "const0.25"]


def make_varlen_packed(
    seqlens: List[int],
    hq: int,
    hk: int,
    d: int,
    dv: int,
    device="cuda",
    seed=0,
    init: str = "randn",
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
    if init == "const0.25":
        q.fill_(0.25)
        k.fill_(0.25)
        v.fill_(0.25)
    elif init != "randn":
        raise ValueError(f"unknown init pattern: {init!r}")
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


# ---------------------------------------------------------------------------
# Shape tables
# ---------------------------------------------------------------------------


# hq=64 for all cases; hk=8 for D64 and hk=4 for D128 (GQA ratios 8 / 16).
_CORRECTNESS_SHAPES = [
    # aligned single batch
    (64, 64, 8, [256]),
    (128, 64, 4, [256]),
    # multi-batch, mixed (some unaligned) seqlens -> causal only
    (64, 64, 8, [128, 256, 384]),
    (128, 64, 4, [128, 256, 384]),
    (64, 64, 8, [100, 200, 300]),  # unaligned
    (128, 64, 4, [100, 200, 300]),
    # 256-aligned multi-batch (exercised under BOTH causal and mask=0)
    (64, 64, 8, [256, 512]),
    (128, 64, 4, [256, 512]),
    (64, 64, 8, [256, 512, 768]),  # aligned 3-batch
    (128, 64, 4, [256, 512, 768]),
    # larger (256-aligned)
    (64, 64, 8, [512, 1024]),
    (128, 64, 4, [512, 1024]),
]


_VARLEN_PERF_SHAPES = [
    (64, 64, 8, [4096, 4096]),  # D64  multi-batch
    (128, 64, 4, [2048, 2048]),  # D128 multi-batch
    (128, 64, 4, [16384]),  # D128 sq=sk=16384 (long context)
    (64, 64, 8, [32768]),  # D64  sq=sk=32768 (long context)
]


def _kv_256_aligned(seqlens) -> bool:
    return all(s % 256 == 0 for s in seqlens)


# ---------------------------------------------------------------------------
# Benchmark entry: correctness (+ optional perf) for one shape.
# ---------------------------------------------------------------------------


@benchmark()
def test_fmha_fwd_with_sink_varlen_asm(
    head_dim,
    hq,
    hk,
    seqlens,
    is_causal,
    init,
    do_ref,
    do_perf,
):
    device = "cuda"
    q, k, v, cu = make_varlen_packed(
        seqlens, hq, hk, head_dim, head_dim, device=device, init=init
    )
    cu_q = cu
    cu_k = cu  # equal q/k seqlens per batch
    max_seqlen_q = max(seqlens)
    scale = 1.0 / math.sqrt(head_dim)
    # D64 -> exercise sink; D128 -> kernel ignores sink (pass None).
    sink = _d64_sink(hq, device) if head_dim == 64 else None

    ret = {}

    if do_ref:
        out_pub, lse_pub = run_kernel(
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
        out_ops, lse_ops = run_kernel(
            q,
            k,
            v,
            cu_q,
            cu_k,
            max_seqlen_q,
            scale=scale,
            is_causal=is_causal,
            sink=sink,
            via="ops",
        )
        out_ref, lse_ref = _ref_varlen(
            q, k, v, cu_q, cu_k, is_causal=is_causal, sink=sink
        )

        msg = f"d={head_dim} c={is_causal} seqlens={seqlens}"
        ret["err(O)"] = checkAllclose(
            out_ref.float().cpu(),
            out_pub.float().cpu(),
            rtol=1e-2,
            atol=1e-2,
            msg=f"O {msg}",
        )
        ret["err(LSE)"] = checkAllclose(
            lse_ref.float().cpu(),
            lse_pub.float().cpu(),
            rtol=1e-2,
            atol=1e-2,
            msg=f"LSE {msg}",
        )
        ret["nrms(O)"] = _nrms(out_pub, out_ref)
        # Public dispatcher must match the low-level ops call bit-for-bit.
        ret["dO(pub-ops)"] = (out_pub.float() - out_ops.float()).abs().max().item()
        ret["dLSE(pub-ops)"] = (lse_pub.float() - lse_ops.float()).abs().max().item()

    if do_perf:
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
        # FLOPs summed over batches (each ~ 2 * hq * s^2 * 2d); causal halves it.
        flops = sum(2.0 * hq * s * s * (2 * head_dim) for s in seqlens)
        if is_causal:
            flops /= 2.0
        ret["us"] = us
        ret["tflops"] = flops / (us * 1e-6) / 1e12

    return ret


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Sweep aiter.fmha_fwd_with_sink_varlen_asm shapes and print a summary.",
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


if __name__ == "__main__":
    if get_gfx() not in ["gfx1250"]:
        print(
            "fmha_fwd_with_sink_varlen_asm ASM kernels are only shipped for gfx1250 "
            "(hsa/gfx1250/fmha_fwd_bf16_varlen/*.co); no GPU or a different arch — skip."
        )
        sys.exit(0)

    args = parser.parse_args()
    causal_modes = [bool(c) for c in args.causal]

    # ----- Correctness sweep ---------------------------------------------
    if not args.no_ref:
        rows = []
        for head_dim, hq, hk, seqlens in _CORRECTNESS_SHAPES:
            if head_dim not in args.head_dim:
                continue
            for is_causal in causal_modes:
                # mask=0 (non-causal) kernels require every kv_seqlen % 256 == 0.
                if not is_causal and not _kv_256_aligned(seqlens):
                    continue
                for init in args.init:
                    rows.append(
                        test_fmha_fwd_with_sink_varlen_asm(
                            head_dim,
                            hq,
                            hk,
                            seqlens,
                            is_causal,
                            init,
                            do_ref=True,
                            do_perf=False,
                        )
                    )
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "fmha_fwd_with_sink_varlen_asm correctness summary (markdown):\n%s",
            df.to_markdown(index=False),
        )

    # ----- Perf sweep -----------------------------------------------------
    if not args.no_perf:
        rows = []
        for head_dim, hq, hk, seqlens in _VARLEN_PERF_SHAPES:
            if head_dim not in args.head_dim:
                continue
            for is_causal in causal_modes:
                # perf always covers both init patterns (randn + const0.25).
                for init in _PERF_INITS:
                    rows.append(
                        test_fmha_fwd_with_sink_varlen_asm(
                            head_dim,
                            hq,
                            hk,
                            seqlens,
                            is_causal,
                            init,
                            do_ref=False,
                            do_perf=True,
                        )
                    )
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "fmha_fwd_with_sink_varlen_asm perf summary (markdown):\n%s",
            df.to_markdown(index=False),
        )
