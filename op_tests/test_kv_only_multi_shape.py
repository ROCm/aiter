#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sweep the (head_dim, rot_dim, group_size) shapes that the K-only HIP entry
supports today. For each shape, run a pure-torch reference and compare the
kernel output bit-for-bit on the e8m0 scale and within fp8-roundoff tolerance
on the dequant'd nope. Also report kernel us so non-V4 shapes have a perf data
point.

Keep this list in sync with KV_K_ONLY_DISPATCH_TABLE in
csrc/kernels/fused_qk_norm_rope_cache_quant.cu.
"""

import argparse
import torch
import pandas as pd

import aiter
from aiter import dtypes
from aiter.test_common import checkAllclose
from aiter.utility.fp4_utils import f32_to_mx_e8m0_scale
from aiter.utility.mx_types import MxDtypeInt, MxScaleRoundModeInt

# (head_dim, rot_dim, group_size) combos to sweep; mirrors the C++ dispatch table.
KV_KERNEL_SUPPORTED_SHAPES = (
    (512, 64, 64),  # DeepSeek V4-Pro (default)
    (192, 64, 64),  # DeepSeek V2 / V3 MLA, default group
    (384, 128, 64),  # head_dim=384, rope=128 (Qwen-style)
)


def _bench(func, *args, warmup=5, iters=50, **kwargs):
    """Manual cuda.Event timing -- avoids run_perftest's deepcopy/rotate logic
    that breaks on per-shape kwargs sets."""
    for _ in range(warmup):
        func(*args, **kwargs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        out = func(*args, **kwargs)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # ms -> us
    times.sort()
    return out, times[len(times) // 2]  # median


torch.set_default_device("cuda")
_FP8 = dtypes.fp8
_DEV = "cuda"


def _cos_sin(max_pos, rope_dim, dtype=torch.bfloat16):
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, rope_dim, 2, device=_DEV).float() / rope_dim)
    )
    freqs = torch.einsum(
        "i,j->ij", torch.arange(max_pos, device=_DEV).float(), inv_freq
    )
    return freqs.cos().to(dtype).contiguous(), freqs.sin().to(dtype).contiguous()


def _gptj_rope(pe, cos, sin, pos):
    """GPT-J adjacent-pair rotation on the PE tail. pe: [T, NK, rope_dim]."""
    T, n_heads, rope_dim = pe.shape
    c = cos[pos].float().view(T, 1, rope_dim // 2)
    s = sin[pos].float().view(T, 1, rope_dim // 2)
    x, y = pe[..., 0::2], pe[..., 1::2]
    return torch.stack([x * c - y * s, x * s + y * c], -1).reshape(T, n_heads, rope_dim)


def _ref(kv, kw, cos, sin, pos, eps, *, group_size):
    T, NK, head_dim = kv.shape
    rope_dim = cos.shape[-1] * 2
    nope_dim = head_dim - rope_dim
    n_groups = nope_dim // group_size

    normed = kv.float() * torch.rsqrt(kv.float().pow(2).mean(-1, keepdim=True) + eps)
    normed = normed * kw.float()
    nope, pe = normed[..., :nope_dim], normed[..., nope_dim:]
    pe_rot = _gptj_rope(pe, cos, sin, pos)

    amax = nope.reshape(T, NK, n_groups, group_size).abs().amax(-1).clamp_min(1e-12)
    scale_e8m0 = f32_to_mx_e8m0_scale(
        amax, mode=MxScaleRoundModeInt.RoundUp, dtype=MxDtypeInt.FP8_E4M3
    ).view(torch.uint8)
    inv_scale = 1.0 / (scale_e8m0.to(torch.int32) << 23).view(torch.float32)
    nope_fp8 = (
        nope
        * inv_scale.unsqueeze(-1)
        .expand(T, NK, n_groups, group_size)
        .reshape(T, NK, nope_dim)
    ).to(_FP8)
    return nope_fp8, scale_e8m0, pe_rot.bfloat16()


def run_shape(head_dim, rot_dim, group_size, T, *, NK=1, is_neox=False):
    nope_dim = head_dim - rot_dim
    n_groups = nope_dim // group_size
    cos, sin = _cos_sin(max(T, 64) + 4, rot_dim)
    pos = torch.randint(0, cos.shape[0] - 1, (T,), dtype=torch.int64, device=_DEV)
    kv = (torch.randn(T, NK, head_dim, device=_DEV) * 0.1).bfloat16()
    kw = (torch.randn(head_dim, device=_DEV).abs() + 0.5).bfloat16()
    eps = 1e-6

    ref_nope, ref_scale, ref_pe = _ref(
        kv, kw, cos, sin, pos, eps, group_size=group_size
    )

    nope_scale_buff = torch.zeros(T, NK, head_dim, dtype=_FP8, device=_DEV)
    rope_buff = torch.empty(T, NK, rot_dim, dtype=torch.bfloat16, device=_DEV)
    (nope_scale_buff, rope_buff), us = _bench(
        aiter.fused_kv_norm_rope_group_quant,
        kv,
        kw,
        pos,
        cos,
        sin,
        eps,
        is_neox=is_neox,
        quant_group_size=group_size,
        scale_dtype="e8m0",
        nope_scale_buff=nope_scale_buff,
        rope_buff=rope_buff,
    )

    k_nope_got = nope_scale_buff[..., :nope_dim]
    k_scale_pairs = (
        nope_scale_buff.view(torch.uint8)[..., nope_dim : nope_dim + 2 * n_groups]
        .contiguous()
        .reshape(T, NK, n_groups, 2)
    )
    got_scale = (k_scale_pairs[..., 0].to(torch.int32) << 23).view(torch.float32)
    ref_scale_f32 = (ref_scale.to(torch.int32) << 23).view(torch.float32)
    deq = k_nope_got.float() * got_scale.unsqueeze(-1).expand(
        T, NK, n_groups, group_size
    ).reshape(T, NK, nope_dim)
    ref_deq = ref_nope.float() * ref_scale_f32.unsqueeze(-1).expand(
        T, NK, n_groups, group_size
    ).reshape(T, NK, nope_dim)
    err_n = checkAllclose(
        deq,
        ref_deq,
        atol=0.05,
        rtol=0.02,
        msg=f"K-nope fp8 (D={head_dim} RD={rot_dim} G={group_size})",
    )
    checkAllclose(
        k_scale_pairs[..., 0].float(),
        ref_scale.float(),
        atol=0,
        rtol=0,
        msg=f"K e8m0 scale (D={head_dim} RD={rot_dim} G={group_size})",
    )
    checkAllclose(
        k_scale_pairs[..., 1].float(),
        ref_scale.float(),
        atol=0,
        rtol=0,
        msg=f"K e8m0 scale dup (D={head_dim} RD={rot_dim} G={group_size})",
    )
    err_pe = checkAllclose(
        rope_buff.float(),
        ref_pe.float(),
        atol=0.01,
        rtol=0.01,
        msg=f"K-pe bf16 (D={head_dim} RD={rot_dim} G={group_size})",
    )
    return {
        "head_dim": head_dim,
        "rot_dim": rot_dim,
        "group_size": group_size,
        "T": T,
        "us": round(us, 3),
        "err_nope": err_n,
        "err_pe": err_pe,
    }


parser = argparse.ArgumentParser()
parser.add_argument("-T", "--T", type=int, default=4096)
parser.add_argument("--neox", action="store_true")
args = parser.parse_args()

rows = []
for hd, rd, gs in sorted(KV_KERNEL_SUPPORTED_SHAPES):
    try:
        rows.append(run_shape(hd, rd, gs, args.T, is_neox=args.neox))
    except Exception as e:
        rows.append(
            {
                "head_dim": hd,
                "rot_dim": rd,
                "group_size": gs,
                "T": args.T,
                "us": None,
                "err_nope": f"FAIL: {e}",
                "err_pe": "n/a",
            }
        )

print()
print(f"K-only kernel multi-shape sweep (T={args.T}, is_neox={args.neox}, NK=1)")
print(pd.DataFrame(rows).to_markdown(index=False))
