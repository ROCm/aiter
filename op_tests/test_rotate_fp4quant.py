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
    help="""rope. Default: False.
    --rope # True""",
)

args = parser.parse_args()

df = []
for dtype in args.dtype:
    for head_num in args.head_num:
        for dim in args.dim:
            for m in args.m:
                if args.rope:
                    ret = test_rope_rotate_fp4quant_inplace(
                        m, head_num, dim, dtype=dtype
                    )
                else:
                    ret = test_rotate_fp4quant_inplace(m, head_num, dim, dtype=dtype)
                df.append(ret)

df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("rotate_fp4quant_inplace summary (markdown):\n%s", df_md)
