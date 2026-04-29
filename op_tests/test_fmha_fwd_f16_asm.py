# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness + performance tests for aiter.fmha_fwd_f16 (ASM path).

Layout convention used in tests
--------------------------------
* i_perm=2 (sbhd): input q/k/v shape [s, b, h, d]  ← kernel default
* o_perm=0 (bshd): output shape [b, s, h, d]        ← kernel default

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
# Layout helpers
# ---------------------------------------------------------------------------

def make_sbhd(*shape, **kw) -> torch.Tensor:
    """Create contiguous sbhd [s, b, h, d] tensor."""
    return torch.randn(*shape, **kw)


def to_bhsd(t: torch.Tensor, perm: int) -> torch.Tensor:
    """Permute a 4-D tensor in `perm` layout to bhsd [b, h, s, d].

    perm code:
        0 = bshd  [b, s, h, d]
        1 = bhsd  [b, h, s, d]   (no-op)
        2 = sbhd  [s, b, h, d]
    """
    if perm == 0:    # bshd → bhsd
        return t.permute(0, 2, 1, 3).contiguous()
    elif perm == 1:  # bhsd → bhsd
        return t.contiguous()
    elif perm == 2:  # sbhd → bhsd
        return t.permute(1, 2, 0, 3).contiguous()
    raise ValueError(f"unsupported perm={perm}")


# ---------------------------------------------------------------------------
# Reference implementations (inputs/outputs in bhsd)
# ---------------------------------------------------------------------------

def ref_standard(q, k, v, scale, is_causal):
    """Standard attention, no sink.  All tensors bhsd."""
    b, hq, sq, d = q.shape
    _, hk, sk, _ = k.shape
    if hq != hk:
        k = k.repeat_interleave(hq // hk, dim=1)
        v = v.repeat_interleave(hq // hk, dim=1)
    qf, kf, vf = q.float(), k.float(), v.float()
    attn = qf @ kf.transpose(-1, -2) * scale   # [b, hq, sq, sk]
    if is_causal:
        # Bottom-right causal (matches kernel / poc_kl fmha_causal_mask):
        #   mask out k > q + (sk - sq)  for row q in [0, sq), col k in [0, sk).
        # When sq == sk this reduces to the standard lower-triangular causal.
        m = torch.triu(
            torch.ones(sq, sk, dtype=torch.bool, device=q.device),
            sk - sq + 1,
        )
        attn = attn.masked_fill(m, float("-inf"))
    lse  = torch.logsumexp(attn, dim=-1)
    out  = (torch.softmax(attn, dim=-1) @ vf).to(q.dtype)
    return out, lse  # bhsd, [b, hq, sq]


def ref_with_sink(q, k, v, scale, is_causal, sink_post_scale: torch.Tensor):
    """Attention with sink mechanism matching fmha_merge_sink_rowwise.

    sink_post_scale: [hq] fp32, AITER post-scale convention.
    Internally converted to pre-scale: sink_raw = sink_post_scale * sqrt(d).
    The sink adds to the softmax denominator with zero value contribution.
    """
    b, hq, sq, d = q.shape
    _, hk, sk, _ = k.shape
    if hq != hk:
        k = k.repeat_interleave(hq // hk, dim=1)
        v = v.repeat_interleave(hq // hk, dim=1)

    qf, kf, vf = q.float(), k.float(), v.float()
    attn = qf @ kf.transpose(-1, -2)   # [b, hq, sq, sk]  (pre-scale raw)

    if is_causal:
        # Bottom-right causal (matches kernel / poc_kl fmha_causal_mask).
        m = torch.triu(
            torch.ones(sq, sk, dtype=torch.bool, device=q.device),
            sk - sq + 1,
        )
        attn = attn.masked_fill(m, float("-inf"))

    # Convert sink from AITER post-scale to pre-scale raw
    sink_raw = (sink_post_scale * math.sqrt(d)).float()  # [hq]
    # Broadcast to [b, hq, sq] to match per-row max
    sink_raw_bhs = sink_raw[None, :, None].expand(b, hq, sq)  # [b, hq, sq]

    # Compute softmax max over real tokens
    max_attn, _ = attn.max(dim=-1)                # [b, hq, sq]
    # Effective max including sink (pre-scale domain)
    max_total = torch.maximum(max_attn, sink_raw_bhs)  # [b, hq, sq]

    # Rescale numerator (O) and denominator (sum):
    #   row_scale = exp2((old_max - new_max) * scale * log2e)
    #             = exp(( max_attn - max_total) * scale)
    row_scale = torch.exp((max_attn - max_total) * scale)   # [b, hq, sq]

    # Standard softmax numerators (rescaled)
    probs_unnorm = torch.exp((attn - max_total.unsqueeze(-1)) * scale)  # [b,hq,sq,sk]
    probs_sum    = probs_unnorm.sum(dim=-1) * row_scale   # wait, already accounted for

    # Re-derive carefully using max_total directly:
    #   exp((x - max_total) * scale) for each attn score x
    #   sum of these = denom_real
    denom_real = torch.exp((attn - max_total.unsqueeze(-1)) * scale).sum(dim=-1)  # [b,hq,sq]

    # Sink term: exp((sink_raw - max_total) * scale)
    sink_term = torch.exp((sink_raw_bhs - max_total) * scale)  # [b, hq, sq]

    denom_total = denom_real + sink_term   # [b, hq, sq]

    # Final probabilities for real tokens only (sink value=0, so no contribution to out)
    probs = torch.exp((attn - max_total.unsqueeze(-1)) * scale) / denom_total.unsqueeze(-1)

    out = (probs @ vf).to(q.dtype)  # [b, hq, sq, d]

    # LSE including sink: log(denom_total) + max_total * scale
    lse = torch.log(denom_total) + max_total * scale  # [b, hq, sq]

    return out, lse


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

    q_s = make_sbhd(sq, batch, hq, head_dim, dtype=torch.bfloat16, device=device)
    k_s = make_sbhd(sk, batch, hk, head_dim, dtype=torch.bfloat16, device=device)
    v_s = make_sbhd(sk, batch, hk, head_dim, dtype=torch.bfloat16, device=device)
    scale = 1.0 / math.sqrt(head_dim)

    # D64 → non-zero sink (exercises ENABLE_SINK code path)
    # D128 → no sink (kernel ignores it)
    sink = _d64_sink(hq, device) if head_dim == 64 else None

    # ASM forward: sbhd in → bshd out
    i_perm, o_perm = 2, 0
    out_kernel, lse_asm = aiter.fmha_fwd_f16(
        q_s, k_s, v_s,
        softmax_scale=scale, is_causal=is_causal,
        return_lse=True, i_perm=i_perm, o_perm=o_perm, sink=sink,
    )

    # Reference is always bhsd-in / bhsd-out.  Convert kernel I/O accordingly.
    q_b = to_bhsd(q_s, i_perm)
    k_b = to_bhsd(k_s, i_perm)
    v_b = to_bhsd(v_s, i_perm)

    if head_dim == 64:
        out_ref_bhsd, lse_ref = ref_with_sink(q_b, k_b, v_b, scale, is_causal, sink)
    else:
        out_ref_bhsd, lse_ref = ref_standard(q_b, k_b, v_b, scale, is_causal)

    out_asm_bhsd = to_bhsd(out_kernel, o_perm)

    checkAllclose(out_asm_bhsd, out_ref_bhsd, rtol=1e-2, atol=1e-2,
                  msg=f"out mismatch (d={head_dim}, causal={is_causal})")
    checkAllclose(lse_asm, lse_ref, rtol=1e-2, atol=1e-2,
                  msg=f"lse mismatch (d={head_dim}, causal={is_causal})")


def test_fmha_fwd_f16_ops_layer():
    """Direct ops-layer call (sbhd in, bshd out, D64 with non-zero sink)."""
    device = "cuda"
    torch.manual_seed(0)

    sq, batch, hq, hk, sk, d = 128, 1, 8, 2, 2048, 64
    q_s = make_sbhd(sq, batch, hq, d, dtype=torch.bfloat16, device=device)
    k_s = make_sbhd(sk, batch, hk, d, dtype=torch.bfloat16, device=device)
    v_s = make_sbhd(sk, batch, hk, d, dtype=torch.bfloat16, device=device)
    scale = 1.0 / math.sqrt(d)
    sink  = _d64_sink(hq, device)

    i_perm, o_perm = 2, 0
    out_kernel, lse_asm = aiter.fmha_fwd_f16_asm(
        q_s, k_s, v_s, scale, False, True,
        i_perm=i_perm, o_perm=o_perm, sink=sink,
    )

    out_ref, lse_ref = ref_with_sink(
        to_bhsd(q_s, i_perm), to_bhsd(k_s, i_perm), to_bhsd(v_s, i_perm),
        scale, False, sink,
    )
    checkAllclose(to_bhsd(out_kernel, o_perm), out_ref, rtol=1e-2, atol=1e-2)
    checkAllclose(lse_asm, lse_ref, rtol=1e-2, atol=1e-2)


def test_fmha_fwd_f16_d64_requires_sink():
    """Calling D64 without a sink tensor must raise an error."""
    device = "cuda"
    q = make_sbhd(128, 1, 4, 64, dtype=torch.bfloat16, device=device)
    k = make_sbhd(2048, 1, 4, 64, dtype=torch.bfloat16, device=device)
    v = make_sbhd(2048, 1, 4, 64, dtype=torch.bfloat16, device=device)
    with pytest.raises(RuntimeError, match="D64.*sink"):
        aiter.fmha_fwd_f16(q, k, v, sink=None)


# ---------------------------------------------------------------------------
# Performance tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
def test_fmha_fwd_f16_perf(head_dim, is_causal):
    device = "cuda"
    torch.manual_seed(0)

    # perf_d64 / perf_d128 in run.sh: batch=2 kv_head_num=8 gqa=8 → hq=64
    sq, batch, hq, hk, sk = 8192, 2, 64, 8, 8192
    q_s = make_sbhd(sq, batch, hq, head_dim, dtype=torch.bfloat16, device=device)
    k_s = make_sbhd(sk, batch, hk, head_dim, dtype=torch.bfloat16, device=device)
    v_s = make_sbhd(sk, batch, hk, head_dim, dtype=torch.bfloat16, device=device)
    scale = 1.0 / math.sqrt(head_dim)
    sink  = _d64_sink(hq, device) if head_dim == 64 else None

    _, us = run_perftest(
        aiter.fmha_fwd_f16,
        q_s, k_s, v_s,
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
    description="Run aiter.fmha_fwd_f16 on a single shape and dump kernel args.",
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
parser.add_argument("-i",  "--i_perm",     type=int, choices=[0, 1, 2], default=2,
                    help="input layout: 0=bshd 1=bhsd 2=sbhd (default 2)")
parser.add_argument("-o",  "--o_perm",     type=int, choices=[0, 1, 2], default=0,
                    help="output layout: 0=bshd 1=bhsd 2=sbhd (default 0)")
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
          f"i_perm={args.i_perm} o_perm={args.o_perm}", flush=True)

    q_s = make_sbhd(sq, b, hq, d, dtype=torch.bfloat16, device=device)
    k_s = make_sbhd(sk, b, hk, d, dtype=torch.bfloat16, device=device)
    v_s = make_sbhd(sk, b, hk, d, dtype=torch.bfloat16, device=device)
    scale = 1.0 / math.sqrt(d)
    sink  = _d64_sink(hq, device) if d == 64 else None
    torch.cuda.synchronize()

    import time as _t
    t0 = _t.time()
    out_kernel, lse_asm = aiter.fmha_fwd_f16(
        q_s, k_s, v_s, scale, causal, True,
        i_perm=args.i_perm, o_perm=args.o_perm, sink=sink,
    )
    torch.cuda.synchronize()
    print(f"asm time: {(_t.time()-t0)*1000:.2f} ms", flush=True)
    print(f"out.shape={tuple(out_kernel.shape)}  lse.shape={tuple(lse_asm.shape)}", flush=True)

    if args.ref:
        # Convert kernel I/O to bhsd; ref is always bhsd-in / bhsd-out.
        q_b = to_bhsd(q_s, args.i_perm)
        k_b = to_bhsd(k_s, args.i_perm)
        v_b = to_bhsd(v_s, args.i_perm)
        if d == 64:
            out_ref, lse_ref = ref_with_sink(q_b, k_b, v_b, scale, causal, sink)
        else:
            out_ref, lse_ref = ref_standard(q_b, k_b, v_b, scale, causal)
        # cast asm output to fp32 BEFORE permute to avoid bf16 contiguous hang
        out_asm_bhsd = to_bhsd(out_kernel.float(), args.o_perm)
        out_ref_f    = out_ref.float()
        diff_o = (out_asm_bhsd - out_ref_f).abs().max().item()
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
            aiter.fmha_fwd_f16,
            q_s, k_s, v_s, scale, causal, False,
            num_iters=10, num_warmup=2,
            i_perm=args.i_perm, o_perm=args.o_perm, sink=sink,
        )
        flops = 2.0 * b * hq * sq * sk * (2 * d)
        if causal:
            flops /= 2.0
        tflops = flops / (us * 1e-6) / 1e12
        print(f"perf: {us:.1f} us  ({tflops:.2f} TFLOPS)", flush=True)
