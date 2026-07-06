# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from aiter.test_common import (
    checkAllclose,
    benchmark,
    run_perftest,
)
import torch
import aiter
from aiter import dtypes, get_gfx
import argparse
import pandas as pd

torch.set_default_device("cuda")

# FP4 e2m1 representable magnitudes (positive half). Symmetric around 0.
_FP4_MAGNITUDES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def fp4_act_quant_inplace(x: torch.Tensor, block_size: int = 32) -> None:
    fp4_max = 6.0
    fp4_max_inv = 1.0 / fp4_max
    eps_amax = 6.0 * (2.0**-126)

    *prefix, n = x.shape
    assert n % block_size == 0, f"last dim {n} not divisible by block_size {block_size}"

    blocks = x.reshape(*prefix, n // block_size, block_size).float()
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=eps_amax)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax * fp4_max_inv)))

    normalized = (blocks / scale).clamp(min=-fp4_max, max=fp4_max)

    fp4_vals = _FP4_MAGNITUDES.to(normalized.device)
    diff = (normalized.abs().unsqueeze(-1) - fp4_vals).abs()
    snapped_mag = fp4_vals[diff.argmin(dim=-1)]
    snapped = torch.where(normalized < 0, -snapped_mag, snapped_mag)

    dequant = snapped * scale
    x.copy_(dequant.reshape(*prefix, n).to(x.dtype))


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    assert n > 0 and (n & (n - 1)) == 0, f"last dim {n} must be a power of 2"

    orig_dtype = x.dtype
    *prefix, _ = x.shape
    flat = x.reshape(-1, n).float().contiguous()

    h = 1
    while h < n:
        view = flat.view(-1, n // (2 * h), 2, h)
        a = view[..., 0, :]
        b = view[..., 1, :]
        flat = torch.stack([a + b, a - b], dim=-2).reshape(-1, n)
        h *= 2

    flat = flat * (n**-0.5)
    return flat.reshape(*prefix, n).to(orig_dtype)


def rotate_fp4quant_inplace_torch(x: torch.Tensor, block_size: int = 32):
    x = rotate_activation(x)
    fp4_act_quant_inplace(x, block_size)
    return x


def rope_inplace_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
):
    rope = x[..., -rope_dim:]
    rope_complex = torch.view_as_complex(rope.float().unflatten(-1, (-1, 2)))
    freqs = torch.complex(cos[positions].float(), sin[positions].float())
    rope_out = torch.view_as_real(rope_complex * freqs.view(-1, 1, rope_dim // 2))
    rope.copy_(rope_out.flatten(-2).to(x.dtype))
    return x


def rope_rotate_fp4quant_inplace_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
    block_size: int = 32,
):
    rope_inplace_torch(x, cos, sin, positions, rope_dim)
    x = rotate_activation(x)
    fp4_act_quant_inplace(x, block_size)
    return x


def rope_rotate_fp8quant_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
    group_size: int = 128,
):
    """Torch reference for the fused rope+hadamard+fp8 path.

    The fp8 quant mirrors `dynamic_per_group_scaled_quant_kernel`
    (csrc/kernels/quant_kernels.cu) in fp32:
      absMax = max(1e-10, group_max|x|)            # kernel seeds absMax at 1e-10f
      scale  = absMax * (1 / fp8_max)              # stored per-group scale
      q      = round_to_nearest(x * (1 / scale))   # store multiplies by reciprocal
    fp8 saturation comes from the cast (the kernel relies on the cvt intrinsic),
    so no explicit pre-clamp — matching the dsv4 fused kernel's store path
    (`scaled_cast<fp8>(af, 1.0f / scale)`).
    Returns (dequant_bf16, scale[m, dim//group_size]).
    """
    rope_inplace_torch(x, cos, sin, positions, rope_dim)
    x = rotate_activation(x)  # bf16 rotated activation

    fp8_max = torch.finfo(dtypes.fp8).max
    inv_fp8_max = 1.0 / fp8_max
    *prefix, n = x.shape
    assert n % group_size == 0
    g = n // group_size
    xf = x.float().reshape(-1, g, group_size)
    absmax = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scale = absmax * inv_fp8_max
    q = (xf * scale.reciprocal()).to(dtypes.fp8)
    deq = (q.float() * scale).reshape(*prefix, n).to(x.dtype)
    return deq, scale.reshape(-1, g)


@benchmark()
def test_rope_rotate_fp8quant(M, head_num, N, dtype=torch.bfloat16):
    # fp8 path is supported on both gfx950 (e4m3fn) and gfx942 (e4m3fnuz) -- the
    # arch-only fp4 kernels are gfx942-gated out, so this module builds and runs
    # on gfx942 too. fp8_max (448 vs 240) is taken from torch.finfo per arch.
    rope_dim = 64
    group_size = 128
    max_pos = 2048
    x = torch.randn((M, head_num, N), dtype=dtype, device="cuda")
    positions = torch.randint(0, max_pos, (M,), dtype=torch.int64, device="cuda")
    freqs = torch.randn((max_pos, rope_dim // 2), dtype=torch.float32, device="cuda")
    cos = torch.cos(freqs).to(dtype)
    sin = torch.sin(freqs).to(dtype)

    ref_deq, ref_scale = rope_rotate_fp8quant_torch(
        x.clone(), cos, sin, positions, rope_dim, group_size=group_size
    )

    g = N // group_size
    q_fp8 = torch.empty_like(x, dtype=dtypes.fp8)
    q_scale = torch.empty((M * head_num, g), dtype=torch.float32, device="cuda")
    _, us = run_perftest(
        aiter.rope_rotate_activation,
        q_fp8,
        x,
        cos,
        sin,
        positions,
        rope_dim,
        out_scale=q_scale,
        group_size=group_size,
    )

    # Compare on the dequantized result so the check isolates the kernel's
    # rope+hadamard+quant from fp8 rounding (both paths round identically).
    got_deq = (
        q_fp8.float().reshape(M * head_num, g, group_size)
        * q_scale.reshape(M * head_num, g, 1)
    ).reshape(M, head_num, N)
    err = checkAllclose(ref_deq.float(), got_deq, atol=1e-2, rtol=1e-2)
    scale_err = checkAllclose(
        ref_scale, q_scale.reshape(M * head_num, g), atol=1e-3, rtol=1e-3
    )
    ret = {}
    ret["op"] = "rope_rotate_fp8"
    ret["head_num"] = head_num
    ret["rope_dim"] = rope_dim
    ret["err"] = err
    ret["scale_err"] = scale_err
    ret["us"] = us
    return ret


@benchmark()
def test_rotate_fp4quant_inplace(M, head_num, N, dtype=torch.bfloat16):
    if get_gfx() == "gfx942":
        aiter.logger.info("gfx942 is not supported")
        return {}
    x = torch.randn((M, head_num, N), dtype=dtype, device="cuda")
    ref = rotate_fp4quant_inplace_torch(x.clone())
    y = torch.empty_like(x)
    _, us = run_perftest(aiter.rotate_activation_fp4quant_inplace, y, x, group_size=32)
    err = checkAllclose(ref, y, atol=1e-2, rtol=1e-2)
    ret = {}
    ret["op"] = "rotate"
    ret["err"] = err
    ret["us"] = us
    return ret


@benchmark()
def test_rope_rotate_fp4quant_inplace(M, head_num, N, dtype=torch.bfloat16):
    if get_gfx() == "gfx942":
        aiter.logger.info("gfx942 is not supported")
        return {}
    rope_dim = 64
    max_pos = 2048
    x = torch.randn((M, head_num, N), dtype=dtype, device="cuda")
    positions = torch.randint(0, max_pos, (M,), dtype=torch.int64, device="cuda")
    freqs = torch.randn((max_pos, rope_dim // 2), dtype=torch.float32, device="cuda")
    cos = torch.cos(freqs).to(dtype)
    sin = torch.sin(freqs).to(dtype)
    ref = rope_rotate_fp4quant_inplace_torch(
        x.clone(), cos, sin, positions, rope_dim, block_size=32
    )
    y = torch.empty_like(x)
    _, us = run_perftest(
        aiter.rope_rotate_activation_fp4quant_inplace,
        y,
        x,
        cos,
        sin,
        positions,
        rope_dim,
        group_size=32,
    )
    err = checkAllclose(ref, y, atol=1e-2, rtol=1e-2)
    ret = {}
    ret["op"] = "rope_rotate"
    ret["head_num"] = head_num
    ret["rope_dim"] = rope_dim
    ret["err"] = err
    ret["us"] = us
    return ret


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    choices=[dtypes.d_dtypes["fp16"], dtypes.d_dtypes["bf16"]],
    nargs="*",
    metavar="{fp16, bf16}",
    default=[dtypes.d_dtypes["bf16"]],
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-m",
    type=int,
    nargs="*",
    default=[1, 32, 64, 128, 256, 512, 1024, 2048, 8192, 65536],
    help="""M.
    e.g.: -m 32""",
)

parser.add_argument(
    "-hn",
    "--head_num",
    type=int,
    nargs="*",
    default=[16],
    help="""head_num.
    e.g.: -hn 16""",
)

parser.add_argument(
    "-n",
    "--dim",
    type=int,
    nargs="*",
    choices=[128, 256, 512, 1024],
    default=[512],
    help="""dim.
    e.g.: -n 128""",
)
parser.add_argument(
    "-r",
    "--rope",
    action="store_true",
    help="""run ONLY the rope+rotate+fp4 path.
    Default (no flag): sweep both the fp4 and fused-fp8 paths.""",
)
parser.add_argument(
    "--fp8",
    action="store_true",
    help="""run ONLY the fused rope+hadamard+fp8 quant path (implies --rope).
    Default (no flag): sweep both the fp4 and fused-fp8 paths.""",
)

args = parser.parse_args()

# Which quant paths to sweep. Explicit --fp8 / --rope still select a single
# path (backward compat). With neither flag, the default sweep covers BOTH the
# fp4 path and the fused rope+hadamard+fp8 path, so CI exercises the fp8 kernel
# (aiter.rope_rotate_activation with out_scale) by default.
if args.fp8:
    test_fns = [test_rope_rotate_fp8quant]
elif args.rope:
    test_fns = [test_rope_rotate_fp4quant_inplace]
else:
    test_fns = [test_rotate_fp4quant_inplace, test_rope_rotate_fp8quant]

df = []
for test_fn in test_fns:
    for dtype in args.dtype:
        for head_num in args.head_num:
            for dim in args.dim:
                for m in args.m:
                    ret = test_fn(m, head_num, dim, dtype=dtype)
                    df.append(ret)

df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("dsv4_rotate_quant summary (markdown):\n%s", df_md)
