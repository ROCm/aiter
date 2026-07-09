# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Perf test for the flydsl flash-attention kernels (gfx1201 / RDNA4).

Drives the public ``flydsl_flash_attn_func`` wrapper (which dispatches the
bf16/f16 and per-tensor-fp8 gfx1201 kernels) and benchmarks it against torch
SDPA forced to the FLASH_ATTENTION backend. Correctness is checked against an
fp32 SDPA reference (not timed, not in the table).

Candidates per shape:
    sdpa_flash          : torch SDPA, FLASH_ATTENTION backend (baseline)
    flydsl_bf16         : flydsl_flash_attn_func on bf16/f16 inputs
    flydsl_fp8          : per-tensor fp8 (bf16 output) with the amax+cast
                          quantization INSIDE the timed region -- the realistic
                          path today, where q/k/v arrive bf16 and are quantized
                          on every attention call
    flydsl_fp8_prequant : the same fp8 kernel but with q/k/v pre-quantized
                          outside the timed region -- the producer-fused ceiling
                          (quant amortized upstream); shows fp8's headroom

Correctness (functional coverage) lives in the pytest suite
``op_tests/flydsl_tests/test_flydsl_fmha.py``; this file is the perf sweep.
"""

import argparse
import itertools

import aiter
import pandas as pd
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.nn.attention import sdpa_kernel, SDPBackend

from aiter import dtypes
from aiter.ops.flydsl import flydsl_flash_attn_func
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.jit.utils.chip_info import get_gfx

torch.set_default_device("cuda")

# The flydsl flash-attn kernels are gfx1201/RDNA4 only.
SUPPORTED_GFX = ["gfx1201"]

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0
_QBLOCK = 8192


@triton.jit
def _amax3(q, k, v, aq, ak, av, n, BLOCK: tl.constexpr):
    """Fused per-tensor amax over q/k/v in a single launch (atomic-max)."""
    pid = tl.program_id(0)
    nb = tl.cdiv(n, BLOCK)
    t = pid // nb
    b = pid % nb
    off = b * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    p = tl.where(t == 0, q, tl.where(t == 1, k, v))
    val = tl.max(tl.abs(tl.load(p + off, mask=m, other=0.0)).to(tl.float32))
    o = tl.where(t == 0, aq, tl.where(t == 1, ak, av))
    tl.atomic_max(o, val)


@triton.jit
def _sc3(q, k, v, yq, yk, yv, sq, sk, sv, n, BLOCK: tl.constexpr):
    """Fused scale+clamp+cast of q/k/v to fp8 in a single launch."""
    pid = tl.program_id(0)
    nb = tl.cdiv(n, BLOCK)
    t = pid // nb
    b = pid % nb
    off = b * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    p = tl.where(t == 0, q, tl.where(t == 1, k, v))
    yp = tl.where(t == 0, yq, tl.where(t == 1, yk, yv))
    sp = tl.where(t == 0, sq, tl.where(t == 1, sk, sv))
    inv = 1.0 / tl.load(sp)
    x = tl.load(p + off, mask=m, other=0.0).to(tl.float32) * inv
    # fp8 e4m3 max (448.0) inlined; triton @jit can't read module globals.
    x = tl.minimum(tl.maximum(x, -448.0), 448.0)
    tl.store(yp + off, x.to(yp.dtype.element_ty), mask=m)


def _quant_one(x):
    """Single-tensor per-tensor fp8 quant (cross-attn fallback where q and k/v
    differ in size). real = fp8 * descale."""
    amax = x.abs().max().to(dtypes.fp32)
    s = (amax / FP8_MAX).clamp(min=1e-12).reshape(1)
    xq = (x.to(dtypes.fp32) / s).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return xq, s


def _fp8_quant_fused(q, k, v):
    """Two fused launches (amax3, sc3) across all three tensors. Measured 3-7x
    faster than aiter's per_tensor_quant_hip (which is 3 separate two-pass
    per-tensor calls). Returns (q8, k8, v8, sq, sk, sv).

    The fused path needs one shared element count, so cross-attn (q vs k/v differ
    in size) falls back to per-tensor quant; self-attn keeps the fast path."""
    if not (q.numel() == k.numel() == v.numel()):
        q8, sq = _quant_one(q)
        k8, sk = _quant_one(k)
        v8, sv = _quant_one(v)
        return q8, k8, v8, sq, sk, sv

    n = q.numel()
    grid = (3 * triton.cdiv(n, _QBLOCK),)
    aq = torch.zeros(1, dtype=dtypes.fp32)
    ak = torch.zeros(1, dtype=dtypes.fp32)
    av = torch.zeros(1, dtype=dtypes.fp32)
    _amax3[grid](q, k, v, aq, ak, av, n, BLOCK=_QBLOCK)
    sq, sk, sv = aq / FP8_MAX, ak / FP8_MAX, av / FP8_MAX
    q8 = torch.empty_like(q, dtype=FP8_DTYPE)
    k8 = torch.empty_like(k, dtype=FP8_DTYPE)
    v8 = torch.empty_like(v, dtype=FP8_DTYPE)
    _sc3[grid](q, k, v, q8, k8, v8, sq, sk, sv, n, BLOCK=_QBLOCK)
    return q8, k8, v8, sq, sk, sv

# (label, batch, seq_len, num_heads, head_dim) -- production diffusion shapes.
SHAPES = [
    ("flux", 1, 1536, 24, 128),
    ("flux", 1, 4096, 24, 128),
    ("wan", 1, 8192, 12, 128),
    ("wan", 1, 32768, 12, 128),
    ("sd3.5", 1, 2048, 38, 64),
]


def _to_fp8_per_tensor(t):
    amax = t.abs().amax().clamp(min=1e-8)
    scale = (amax / FP8_MAX).to(dtypes.fp32)
    q = (t.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return q, scale


def run_torch(q, k, v, causal):
    # Reference only: fp32 SDPA, BSHD in/out. Not timed, not in the table.
    out_bhsd = F.scaled_dot_product_attention(
        q.transpose(1, 2).float(),
        k.transpose(1, 2).float(),
        v.transpose(1, 2).float(),
        is_causal=causal,
    )
    return out_bhsd.transpose(1, 2).contiguous()


@benchmark()
def test_flydsl_fmha(model, batch, seq_len, num_heads, head_dim, dtype, causal):
    torch.manual_seed(0)
    shape = (batch, seq_len, num_heads, head_dim)
    q = torch.randn(shape, dtype=dtype)
    k = torch.randn(shape, dtype=dtype)
    v = torch.randn(shape, dtype=dtype)
    ref = run_torch(q, k, v, causal)  # BSHD fp32 reference

    # SDPA baseline consumes BHSD; pre-transpose outside the timed region.
    qb = q.transpose(1, 2).contiguous()
    kb = k.transpose(1, 2).contiguous()
    vb = v.transpose(1, 2).contiguous()

    # Pre-quantized fp8 (producer-fused ceiling): q/k/v already fp8 in HBM, quant
    # amortized upstream. Quantized once, outside timing.
    qq, sq = _to_fp8_per_tensor(q)
    kk, sk = _to_fp8_per_tensor(k)
    vv, sv = _to_fp8_per_tensor(v)

    def _sdpa_flash():
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            return F.scaled_dot_product_attention(qb, kb, vb, is_causal=causal)

    def _flydsl_bf16():
        return flydsl_flash_attn_func(q, k, v, causal=causal)

    def _flydsl_fp8():
        # Realistic path today: q/k/v arrive bf16, so the fused amax+cast is paid
        # on every attention call, inside the timed region.
        q8, k8, v8, s_q, s_k, s_v = _fp8_quant_fused(q, k, v)
        return flydsl_flash_attn_func(
            q8, k8, v8, causal=causal,
            q_descale=s_q, k_descale=s_k, v_descale=s_v,
        )

    def _flydsl_fp8_prequant():
        return flydsl_flash_attn_func(
            qq, kk, vv, causal=causal,
            q_descale=sq, k_descale=sk, v_descale=sv,
        )

    # (fn, layout, in_bytes, out_bytes) -- layout is the fn's output layout.
    # in/out bytes count the attention tensor I/O only (the fp8 amax+cast traffic
    # in flydsl_fp8 shows up in latency/TFLOPS, not in gb_per_sec).
    elem = q.element_size()
    candidates = {
        "sdpa_flash": (_sdpa_flash, "bhsd", elem, elem),
        "flydsl_bf16": (_flydsl_bf16, "bshd", elem, elem),
        "flydsl_fp8": (_flydsl_fp8, "bshd", 1, 2),
        "flydsl_fp8_prequant": (_flydsl_fp8_prequant, "bshd", 1, 2),
    }

    # Non-causal attention fwd: QK^T + P@V = 4*B*H*S^2*D FLOPs; causal ~= half.
    flops = 4.0 * batch * num_heads * seq_len * seq_len * head_dim
    if causal:
        flops *= 0.5

    ret = {"gfx": get_gfx()}
    for name, (fn, layout, in_b, out_b) in candidates.items():
        out, us = run_perftest(fn)
        out_bshd = out.transpose(1, 2).contiguous() if layout == "bhsd" else out
        err = checkAllclose(
            ref.to(dtypes.fp32),
            out_bshd.to(dtypes.fp32),
            rtol=2e-2,
            atol=2e-2,
            msg=f"{name}: flydsl_fmha {model}",
        )
        nbytes = batch * seq_len * num_heads * head_dim * (3 * in_b + out_b)
        # us in microseconds: TFLOP/s = flop/us/1e6, GB/s = bytes/us/1e3
        # (matches aiter test_mha fwd_tflops / fwd_gb_per_sec conventions).
        ret[f"{name}_us"] = us
        ret[f"{name}_tflops"] = flops / us / 1e6
        ret[f"{name}_gb_per_sec"] = nbytes / us / 1e3
        ret[f"{name}_err"] = err
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "flydsl flash-attn unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-s",
        "--shapes",
        type=str,
        nargs="*",
        default=[s[0] for s in SHAPES],
        choices=sorted({s[0] for s in SHAPES}),
        help="""Model shape groups to run.
        e.g.: -s flux wan""",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.bf16],
        help="""Input dtypes for the bf16/f16 path (fp8 is always cross-checked).
        e.g.: -d bf16 fp16""",
    )
    parser.add_argument(
        "-c",
        "--causal",
        type=int,
        nargs="*",
        default=[0],
        help="""Causal masking flags to sweep (0/1).
        e.g.: -c 0 1""",
    )
    args = parser.parse_args()

    def summarize(name, rows):
        if rows:
            aiter.logger.info(
                "%s summary (markdown):\n%s",
                name,
                pd.DataFrame(rows).to_markdown(index=False),
            )

    rows = []
    for dtype, causal in itertools.product(args.dtype, args.causal):
        for label, batch, seq_len, num_heads, head_dim in SHAPES:
            if label not in args.shapes:
                continue
            rows.append(
                test_flydsl_fmha(
                    label, batch, seq_len, num_heads, head_dim, dtype, bool(causal)
                )
            )
    summarize("flydsl_fmha", rows)


if __name__ == "__main__":
    main()
