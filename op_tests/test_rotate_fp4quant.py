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
from aiter.utility.fp4_utils import f32_to_mxfp4, mxfp4_to_f32
import argparse
import pandas as pd

torch.set_default_device("cuda")


def tensor_nbytes(*tensors: torch.Tensor) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def bandwidth_tbs(num_bytes: int, us: float) -> float:
    return num_bytes / us / 1e6


def fp4_act_quant(x: torch.Tensor, block_size: int = 32):
    fp4_max = 6.0
    fp4_max_inv = 1.0 / fp4_max
    eps_amax = 6.0 * (2.0**-126)

    *prefix, n = x.shape
    assert n % block_size == 0, f"last dim {n} not divisible by block_size {block_size}"

    blocks = x.reshape(*prefix, n // block_size, block_size).float()
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=eps_amax)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax * fp4_max_inv)))

    normalized = (blocks / scale).clamp(min=-fp4_max, max=fp4_max)

    return f32_to_mxfp4(normalized.reshape(*prefix, n)), scale.squeeze(-1).to(
        dtypes.fp8_e8m0
    )


def dsv4_shuffle_scale(scale: torch.Tensor):
    *prefix, head_num, groups_per_row = scale.shape
    assert head_num % 16 == 0, f"head_num {head_num} not divisible by 16"
    assert (
        groups_per_row % 4 == 0
    ), f"groups_per_row {groups_per_row} not divisible by 4"

    m_tiles = head_num // 16
    k_tiles = groups_per_row // 4
    qs_pad = ((m_tiles + 3) // 4) * 4
    prefix_dims = tuple(range(len(prefix)))
    scale_real = scale.view(torch.uint8).reshape(*prefix, m_tiles, 16, k_tiles, 4)
    scale_real = scale_real.permute(
        *prefix_dims,
        len(prefix) + 2,
        len(prefix) + 3,
        len(prefix) + 1,
        len(prefix),
    ).contiguous()
    return (
        torch.nn.functional.pad(scale_real, (0, qs_pad - m_tiles))
        .contiguous()
        .view(dtypes.fp8_e8m0)
    )


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    assert n > 0 and (n & (n - 1)) == 0, f"last dim {n} must be a power of 2"

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
    return flat.reshape(*prefix, n)


def rotate_fp4quant_torch(
    x: torch.Tensor, block_size: int = 32, shuffle_scale: bool = False
):
    x = rotate_activation(x)
    x_q, scale = fp4_act_quant(x, block_size)
    if shuffle_scale:
        scale = dsv4_shuffle_scale(scale)
    return x_q, scale


def rope_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
):
    x = x.float()
    rope = x[..., -rope_dim:]
    rope_complex = torch.view_as_complex(rope.float().unflatten(-1, (-1, 2)))
    freqs = torch.complex(cos[positions].float(), sin[positions].float())
    rope_out = torch.view_as_real(rope_complex * freqs.view(-1, 1, rope_dim // 2))
    x[..., -rope_dim:] = rope_out.flatten(-2)
    return x


def rope_rotate_fp4quant_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
    block_size: int = 32,
    shuffle_scale: bool = False,
):
    x = rope_torch(x, cos, sin, positions, rope_dim)
    x = rotate_activation(x)
    x_q, scale = fp4_act_quant(x, block_size)
    if shuffle_scale:
        scale = dsv4_shuffle_scale(scale)
    return x_q, scale


@benchmark()
def test_rotate_fp4quant(M, head_num, N, dtype=torch.bfloat16, shuffle_scale=False):
    if get_gfx() == "gfx942":
        aiter.logger.info("gfx942 is not supported")
        return {}
    x = torch.randn((M, head_num, N), dtype=dtype, device="cuda")
    y_ref, scale_ref = rotate_fp4quant_torch(
        x.clone(), block_size=32, shuffle_scale=shuffle_scale
    )
    y = torch.empty((*x.shape[:-1], N // 2), dtype=dtypes.fp4x2, device="cuda")
    scale = torch.empty_like(scale_ref)
    _, us = run_perftest(
        aiter.rotate_activation_fp4quant,
        y,
        scale,
        x,
        group_size=32,
        shuffle_scale=shuffle_scale,
    )
    err = checkAllclose(
        mxfp4_to_f32(y_ref),
        mxfp4_to_f32(y),
        atol=0,
        rtol=0,
        msg="y",
    )
    checkAllclose(
        scale_ref.view(torch.uint8).to(torch.int16),
        scale.view(torch.uint8).to(torch.int16),
        atol=0,
        rtol=0,
        msg="scale",
    )
    ret = {}
    ret["op"] = "rotate"
    ret["shuffle_scale"] = shuffle_scale
    ret["err"] = err
    ret["us"] = us
    ret["TB/s"] = bandwidth_tbs(tensor_nbytes(x, y, scale), us)
    return ret


@benchmark()
def test_rope_rotate_fp4quant(
    M, head_num, N, dtype=torch.bfloat16, shuffle_scale=False
):
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
    y_ref, scale_ref = rope_rotate_fp4quant_torch(
        x.clone(),
        cos,
        sin,
        positions,
        rope_dim,
        block_size=32,
        shuffle_scale=shuffle_scale,
    )
    y = torch.empty((*x.shape[:-1], N // 2), dtype=dtypes.fp4x2, device="cuda")
    scale = torch.empty_like(scale_ref)
    _, us = run_perftest(
        aiter.rope_rotate_activation_fp4quant,
        y,
        scale,
        x,
        cos,
        sin,
        positions,
        rope_dim,
        group_size=32,
        shuffle_scale=shuffle_scale,
    )
    err = checkAllclose(
        mxfp4_to_f32(y_ref),
        mxfp4_to_f32(y),
        atol=0,
        rtol=0,
        msg="y",
    )
    checkAllclose(
        scale_ref.view(torch.uint8).to(torch.int16),
        scale.view(torch.uint8).to(torch.int16),
        atol=0,
        rtol=0,
        msg="scale",
    )
    ret = {}
    ret["op"] = "rope_rotate"
    ret["head_num"] = head_num
    ret["rope_dim"] = rope_dim
    ret["shuffle_scale"] = shuffle_scale
    ret["err"] = err
    ret["us"] = us
    rope_bytes = M * head_num * rope_dim * cos.element_size()
    position_bytes = M * positions.element_size()
    ret["TB/s"] = bandwidth_tbs(
        tensor_nbytes(x, y, scale) + rope_bytes + position_bytes, us
    )
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
parser.add_argument(
    "-s",
    "--shuffle",
    "--shuffle_scale",
    "--shuffle-scale",
    dest="shuffle_scale",
    action="store_true",
    help="""shuffle scale. Default: False.
    --shuffle # True""",
)

args = parser.parse_args()

df = []
for dtype in args.dtype:
    for head_num in args.head_num:
        for dim in args.dim:
            for m in args.m:
                if args.rope:
                    ret = test_rope_rotate_fp4quant(
                        m,
                        head_num,
                        dim,
                        dtype=dtype,
                        shuffle_scale=args.shuffle_scale,
                    )
                else:
                    ret = test_rotate_fp4quant(
                        m,
                        head_num,
                        dim,
                        dtype=dtype,
                        shuffle_scale=args.shuffle_scale,
                    )
                df.append(ret)

df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("rotate_fp4quant summary (markdown):\n%s", df_md)
