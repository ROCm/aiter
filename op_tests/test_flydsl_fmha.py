# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Perf test for the flydsl flash-attention kernels (gfx1201 / RDNA4).

Drives the public ``flydsl_flash_attn_func`` wrapper (which dispatches the
bf16/f16 and per-tensor-fp8 gfx1201 kernels) and benchmarks it against torch
SDPA forced to the FLASH_ATTENTION backend. Correctness is checked against an
fp32 SDPA reference (not timed, not in the table).

Candidates per shape:
    sdpa_flash            : torch SDPA, FLASH_ATTENTION backend (baseline)
    flydsl_bf16           : flydsl_flash_attn_func on bf16/f16 inputs
    flydsl_fp8_incl_quant : per-tensor fp8 (bf16 output) with the Hadamard rotation
                            of Q/K + amax+cast quantization INSIDE the timed region
                            -- the realistic xDiT path, where q/k/v arrive bf16 and
                            are rotated+quantized on every attention call
    flydsl_fp8_kernel_only: the same fp8 kernel but on q/k/v already quantized
                            outside the timed region (quant amortized upstream) --
                            the pure attention-kernel ceiling; shows fp8's headroom

Correctness (functional coverage) lives in the pytest suite
``op_tests/flydsl_tests/test_flydsl_fmha.py``; this file is the perf sweep.
"""

import argparse
import itertools

import aiter
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from aiter import dtypes
from aiter.ops.flydsl import flydsl_flash_attn_func, flydsl_fp8_quant
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.jit.utils.chip_info import get_gfx

torch.set_default_device("cuda")

# The flydsl flash-attn kernels are gfx1201/RDNA4 only.
SUPPORTED_GFX = ["gfx1201"]

# (label, batch, seq_len, num_heads, head_dim) -- production diffusion shapes.
SHAPES = [
    ("flux", 1, 1536, 24, 128),
    ("flux", 1, 4096, 24, 128),
    ("wan", 1, 8192, 12, 128),
    ("wan", 1, 32768, 12, 128),
    ("sd3.5", 1, 2048, 38, 64),
]


def run_torch(q, k, v, causal):
    # Reference only: fp32 SDPA, BSHD in/out. Not timed, not in the table.
    out_bhsd = F.scaled_dot_product_attention(
        q.transpose(1, 2).float(),
        k.transpose(1, 2).float(),
        v.transpose(1, 2).float(),
        is_causal=causal,
    )
    return out_bhsd.transpose(1, 2).contiguous()


def cosine_stats(out_bshd, ref_bshd, head_dim):
    # Per (token, head) row cosine vs the fp32 reference: min = worst row,
    # mean = average row. Both tensors are BSHD.
    cos = F.cosine_similarity(
        out_bshd.to(dtypes.fp32).reshape(-1, head_dim),
        ref_bshd.to(dtypes.fp32).reshape(-1, head_dim),
        dim=1,
    )
    return cos.min().item(), cos.mean().item()


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

    # fp8 kernel-only ceiling: q/k/v already quantized in HBM, quant amortized
    # upstream. Quantized once, outside timing (rotation off -- this row isolates
    # the attention kernel, not the quant/rotate producer cost).
    qq, kk, vv, sq, sk, sv = flydsl_fp8_quant(q, k, v, rotation=False)

    def _sdpa_flash():
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            return F.scaled_dot_product_attention(qb, kb, vb, is_causal=causal)

    def _flydsl_bf16():
        return flydsl_flash_attn_func(q, k, v, causal=causal)

    def _flydsl_fp8_incl_quant():
        # Realistic path: q/k/v arrive bf16, so the Hadamard rotation of Q/K
        # plus the fused amax+cast is paid on every call, inside the timed region.
        q8, k8, v8, s_q, s_k, s_v = flydsl_fp8_quant(q, k, v)
        return flydsl_flash_attn_func(
            q8, k8, v8, causal=causal,
            q_descale=s_q, k_descale=s_k, v_descale=s_v,
        )

    def _flydsl_fp8_kernel_only():
        return flydsl_flash_attn_func(
            qq, kk, vv, causal=causal,
            q_descale=sq, k_descale=sk, v_descale=sv,
        )

    # (fn, layout, in_bytes, out_bytes) -- layout is the fn's output layout.
    # in/out bytes count the attention tensor I/O only (the fp8 quant/rotate
    # traffic in flydsl_fp8_incl_quant shows up in latency/TFLOPS, not gb_per_sec).
    elem = q.element_size()
    candidates = {
        "sdpa_flash": (_sdpa_flash, "bhsd", elem, elem),
        "flydsl_bf16": (_flydsl_bf16, "bshd", elem, elem),
        "flydsl_fp8_incl_quant": (_flydsl_fp8_incl_quant, "bshd", 1, 2),
        "flydsl_fp8_kernel_only": (_flydsl_fp8_kernel_only, "bshd", 1, 2),
    }

    # Non-causal attention fwd: QK^T + P@V = 4*B*H*S^2*D FLOPs; causal ~= half.
    flops = 4.0 * batch * num_heads * seq_len * seq_len * head_dim
    if causal:
        flops *= 0.5

    ret = {"gfx": get_gfx()}
    for name, (fn, layout, in_b, out_b) in candidates.items():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base_mb = torch.cuda.memory_allocated() / 1e6
        out, us = run_perftest(fn)
        # Peak = high-water allocation during the run; transient = peak above the
        # resting baseline (ref + pre-quantized q/k/v are allocated before the loop
        # and shared by every candidate), i.e. the candidate's own working set.
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        out_bshd = out.transpose(1, 2).contiguous() if layout == "bhsd" else out
        err = checkAllclose(
            ref.to(dtypes.fp32),
            out_bshd.to(dtypes.fp32),
            rtol=2e-2,
            atol=2e-2,
            msg=f"{name}: flydsl_fmha {model}",
        )
        cos_min, cos_mean = cosine_stats(out_bshd, ref, head_dim)
        nbytes = batch * seq_len * num_heads * head_dim * (3 * in_b + out_b)
        # us in microseconds: TFLOP/s = flop/us/1e6, GB/s = bytes/us/1e3
        # (matches aiter test_mha fwd_tflops / fwd_gb_per_sec conventions).
        ret[f"{name}_us"] = us
        ret[f"{name}_tflops"] = flops / us / 1e6
        ret[f"{name}_gb_per_sec"] = nbytes / us / 1e3
        ret[f"{name}_cos_min"] = cos_min
        ret[f"{name}_cos_mean"] = cos_mean
        ret[f"{name}_peak_mb"] = peak_mb
        ret[f"{name}_transient_mb"] = peak_mb - base_mb
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
