# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from re import A
import torch
from torch import Tensor
import aiter
from aiter.test_common import checkAllclose, perftest, benchmark
from aiter.utility.dtypes import get_dtype_fp8
from aiter.utility import dtypes
import argparse


def rms_norm_forward(x: Tensor, weight: Tensor, eps: float):
    input_dtype = x.dtype
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x.to(input_dtype)
    return weight * x


def apply_interleaved_rope(x: torch.Tensor, mrope_section: list[int]) -> torch.Tensor:
    """Apply interleaved MRoPE to 3D rotary embeddings.
    Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
    interleaved [THTHWHTHW...TT], preserving frequency continuity.
    """
    x_t = x[0].clone()
    x_t[..., 1 : mrope_section[1] * 3 : 3] = x[1, ..., 1 : mrope_section[1] * 3 : 3]
    x_t[..., 2 : mrope_section[2] * 3 : 3] = x[2, ..., 2 : mrope_section[2] * 3 : 3]
    return x_t


def apply_rotary_emb_torch(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    is_neox_style: bool,
) -> Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


def apply_rotary_emb_dispatch(
    x: Tensor, cos: Tensor, sin: Tensor, is_neox_style: bool
) -> Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    return apply_rotary_emb_torch(x, cos, sin, is_neox_style)


@perftest()
def run_torch_qk_norm_rope_cache_quant_shuffle(
    qkv: Tensor,  # contiguous (num_tokens * (num_heads_q + num_heads_k + num_heads_v) * head_size)
    qw: Tensor,  #  contiguous (head_size)
    kw: Tensor,  #  contiguous (head_size)
    cos_sin: Tensor,  # contiguous (max_positions * head_size)
    positions: Tensor,  # contiguous (3 * num_tokens) or (num_tokens)
    num_tokens: int,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_size: int,
    is_neox_style: bool,
    eps: float,
    k_cache: Tensor,  # [args.num_blocks, num_heads_k, head_size // x, args.page_size, x]
    v_cache: Tensor,  # [args.num_blocks, num_heads_v, head_size, args.page_size]
    k_scale: Tensor,  # [args.num_blocks, args.page_size]
    v_scale: Tensor,  # [args.num_blocks, args.page_size]
    slot_mapping: Tensor,
    kv_cache_dtype: str,
):
    q_size = num_heads_q * head_size
    k_size = num_heads_k * head_size
    v_size = num_heads_v * head_size
    qkv = qkv.view(num_tokens, q_size + k_size + v_size)
    q, k, v = qkv.split([q_size, k_size, v_size], dim=-1)

    q_by_head = q.view(num_tokens, num_heads_q, head_size)
    q_by_head = rms_norm_forward(q_by_head, qw, eps)
    q = q_by_head.view(q.shape)

    k_by_head = k.view(num_tokens, num_heads_k, head_size)
    k_by_head = rms_norm_forward(k_by_head, kw, eps)
    k = k_by_head.view(k.shape)

    cos_sin = cos_sin.view(max_positions, head_size)
    cos_sin = cos_sin[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)

    q_shape = q.shape
    q = q.view(num_tokens, -1, head_size)
    q = apply_rotary_emb_dispatch(q, cos, sin, is_neox_style)
    q = q.reshape(q_shape)

    k_shape = k.shape
    k = k.view(num_tokens, -1, head_size)
    k = apply_rotary_emb_dispatch(k, cos, sin, is_neox_style)

    v = v.view(num_tokens, -1, head_size)

    from aiter import reshape_and_cache_with_pertoken_quant, reshape_and_cache

    if kv_cache_dtype == "auto":
        reshape_and_cache(
            k,
            v,
            k_cache,
            v_cache,
            slot_mapping,
            kv_cache_dtype,
            None,
            None,
            asm_layout=True,
        )
    else:
        reshape_and_cache_with_pertoken_quant(
            k, v, k_cache, v_cache, k_scale, v_scale, slot_mapping, asm_layout=True
        )

    k = k.reshape(k_shape)
    v = v.reshape(k_shape)
    return q, k, v, k_cache, v_cache


@perftest()
def run_aiter_qk_norm_rope_cache_quant_shuffle(
    qkv: Tensor,  # contiguous (num_tokens * (num_heads_q + num_heads_k + num_heads_v) * head_size)
    qw: Tensor,  #  contiguous (head_size)
    kw: Tensor,  #  contiguous (head_size)
    cos_sin: Tensor,  # contiguous (max_positions * head_size)
    positions: Tensor,  # contiguous (3 * num_tokens)
    num_tokens: int,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_size: int,
    is_neox_style: bool,
    eps: float,
    k_cache: Tensor,
    v_cache: Tensor,
    slot_mapping: Tensor,
    kv_cache_dtype: str,
    k_scale: Tensor,
    v_scale: Tensor,
):
    qkv = qkv.clone()  # inplace op

    aiter.fused_qk_norm_rope_cache_quant_shuffle(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_size,
        eps,
        qw,
        kw,
        cos_sin,
        is_neox_style,
        positions,
        k_cache,
        v_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )

    q_size = num_heads_q * head_size
    k_size = num_heads_k * head_size
    v_size = num_heads_v * head_size

    qkv = qkv.view(num_tokens, q_size + k_size + v_size)
    q, k, v = qkv.split([q_size, k_size, v_size], dim=-1)
    return q, k, v, k_cache, v_cache


@benchmark()
def test_qk_norm_rope_cache_quant(
    dtype,
    num_tokens,
    num_heads_q,
    num_heads_k,
    num_heads_v,
    head_size,
    is_neox_style,
    eps,
    k_cache,
    v_cache,
    slot_mapping,
    kv_cache_dtype,
    k_scale,
    v_scale,
):
    qkv = torch.randn(
        (num_tokens, (num_heads_q + num_heads_k + num_heads_v) * head_size),
        dtype=dtype,
        device="cuda",
    )
    qw = torch.randn(head_size, dtype=dtype, device="cuda")
    kw = torch.randn(head_size, dtype=dtype, device="cuda")
    cos_sin = torch.randn((max_positions, head_size), dtype=dtype, device="cuda")
    pos_shape = (num_tokens,)
    positions = torch.randint(
        0, max_positions, pos_shape, dtype=torch.int64, device="cuda"
    )
    k_scale_ref = k_scale.clone()
    v_scale_ref = v_scale.clone()

    (q_ref, k_ref, v_ref, k_cache_ref, v_cache_ref), avg_torch = (
        run_torch_qk_norm_rope_cache_quant_shuffle(
            qkv,
            qw,
            kw,
            cos_sin,
            positions,
            num_tokens,
            num_heads_q,
            num_heads_k,
            num_heads_v,
            head_size,
            is_neox_style,
            eps,
            k_cache,
            v_cache,
            k_scale_ref,
            v_scale_ref,
            slot_mapping,
            kv_cache_dtype,
        )
    )
    (q, k, v, k_cache, v_cache), avg_cu = run_aiter_qk_norm_rope_cache_quant_shuffle(
        qkv,
        qw,
        kw,
        cos_sin,
        positions,
        num_tokens,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_size,
        is_neox_style,
        eps,
        k_cache,
        v_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )

    info = f"dtype:{dtype}, num_tokens:{num_tokens}, num_heads_q:{num_heads_q}, num_heads_k:{num_heads_k}, num_heads_v:{num_heads_v}, head_size:{head_size}, is_neox_style:{is_neox_style}"
    msg = f"[perf] === {info} === torch avg: {avg_torch:<8.2f} us, cu avg: {avg_cu:<8.2f} us, uplift: {avg_torch / avg_cu - 1:<5.1%}"
    checkAllclose(q_ref, q, msg="q", rtol=1e-2, atol=0.05)
    checkAllclose(k_ref, k, msg="k", rtol=1e-2, atol=0.05)
    checkAllclose(v_ref, v, msg=msg, rtol=1e-2, atol=0.05)
    checkAllclose(
        k_cache_ref.float(), k_cache.float(), msg="k_cache", rtol=1e-2, atol=0.05
    )
    checkAllclose(
        v_cache_ref.float(), v_cache.float(), msg="v_cache", rtol=1e-2, atol=0.05
    )
    checkAllclose(k_scale_ref, k_scale, msg="k_scale", rtol=1e-2, atol=0.05)
    checkAllclose(v_scale_ref, v_scale, msg="v_scale", rtol=1e-2, atol=0.05)


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-n",
    "--is_neox_styles",
    type=dtypes.str2bool,
    nargs="*",
    default=[False, True],
    help="""Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    e.g.: -n true   # for Neox-style
          or -n false # for GPT-J-style""",
)
parser.add_argument(
    "-t",
    "--token",
    type=int,
    nargs="*",
    default=[513, 1257, 127, 778, 1024, 3, 1],
    help="""Number of tokens.
    e.g.: -t 513""",
)
parser.add_argument(
    "-hd",
    "--head",
    type=dtypes.str2tuple,
    nargs="*",
    default=[(32, 4), (64, 8), (4, 1)],
    help="""Number of heads.
    e.g.: -hd 32,4""",
)
parser.add_argument(
    "-hs",
    "--head_sizes",
    type=int,
    nargs="*",
    default=[64, 128, 256],
    help="""Head size.
    e.g.: -hs 64""",
)
parser.add_argument(
    "-m",
    "--max_positions",
    type=int,
    default=10000,
    help="""Max Positions.
    e.g.: -m 10000""",
)
parser.add_argument(
    "-b",
    "--num_blocks",
    type=int,
    default=1000,
    help="""Number of blocks.
    e.g.: -b 1000""",
)
parser.add_argument(
    "-p",
    "--page_size",
    type=int,
    default=16,
    help="""Page size.
    e.g.: -p 16""",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    default="bf16",
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-kvd",
    "--kv_cache_dtypes",
    type=str,
    nargs="*",
    choices=["fp8_e4m3", "auto"],
    default=["fp8_e4m3", "auto"],
    help="""KV cache data type.
    e.g.: -kvd fp8_e4m3""",
)

if __name__ == "__main__":
    args = parser.parse_args()
    max_positions = args.max_positions
    # rope
    for is_neox_style in args.is_neox_styles:
        for num_token in args.token:
            for num_head, num_kv_head in args.head:
                k_scale = torch.zeros(
                    [args.num_blocks, num_kv_head, args.page_size],
                    dtype=torch.float32,
                    device="cuda",
                )
                v_scale = torch.zeros(
                    [args.num_blocks, num_kv_head, args.page_size],
                    dtype=torch.float32,
                    device="cuda",
                )
                for i, head_size in enumerate(args.head_sizes):
                    for kv_cache_dtype in args.kv_cache_dtypes:
                        if kv_cache_dtype == "fp8":
                            cache_dtype = get_dtype_fp8()
                        else:
                            cache_dtype = args.dtype
                        k_cache = torch.randn(
                            [args.num_blocks, args.page_size, num_kv_head, head_size],
                            dtype=args.dtype,
                            device="cuda",
                        ).to(cache_dtype)
                        v_cache = torch.randn(
                            [args.num_blocks, args.page_size, num_kv_head, head_size],
                            dtype=args.dtype,
                            device="cuda",
                        ).to(cache_dtype)
                        slot_mapping = torch.randperm(
                            num_token, dtype=torch.int64, device="cuda"
                        )
                        x = 16 // k_cache.element_size()
                        k_cache = (
                            k_cache.view(
                                [
                                    args.num_blocks,
                                    args.page_size,
                                    num_kv_head,
                                    head_size // x,
                                    x,
                                ]
                            )
                            .permute(0, 2, 3, 1, 4)
                            .contiguous()
                        )
                        v_cache = v_cache.permute(0, 2, 3, 1).contiguous()

                        test_qk_norm_rope_cache_quant(
                            args.dtype,
                            num_token,
                            num_head,
                            num_kv_head,
                            num_kv_head,
                            head_size,
                            is_neox_style,
                            1e-6,
                            k_cache,
                            v_cache,
                            slot_mapping,
                            kv_cache_dtype,
                            k_scale,
                            v_scale,
                        )

    print("done")
