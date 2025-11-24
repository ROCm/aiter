# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter.test_common import checkAllclose, perftest, benchmark
import itertools
from enum import IntEnum
import argparse
from aiter import dtypes
from typing import Tuple
from test_kvcache import run_aiter, run_torch

class RotateStyle(IntEnum):
    NEOX = (0,)
    GPTJ = 1


def rotate_half_neox(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rotate_half_gptj(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def ref_rope_sbhd_fwd(
    x_,
    freqs_,
    rotate_style,
    reuse_freqs_front_part,
    nope_first,
    simulate_cached=False,
    comp_with_fp32=False,
):
    x = x_.to(dtype=torch.float32) if comp_with_fp32 else x_
    freqs = freqs_.to(dtype=torch.float32) if comp_with_fp32 else freqs_
    rotate_half = (
        rotate_half_neox if rotate_style == RotateStyle.NEOX else rotate_half_gptj
    )
    rotate_dim = freqs.shape[-1] * (2 if reuse_freqs_front_part else 1)
    if nope_first:
        d = x.shape[-1]
        x, x_forward = x[..., d - rotate_dim :], x[..., : d - rotate_dim]
    else:
        x, x_forward = x[..., :rotate_dim], x[..., rotate_dim:]
    if reuse_freqs_front_part:
        if rotate_style == RotateStyle.NEOX:
            freqs = freqs.repeat([1] * (freqs.dim() - 1) + [2])
        elif rotate_style == RotateStyle.GPTJ:
            freqs = freqs.repeat_interleave(2, dim=-1)
    cos = (
        torch.cos(freqs).to(dtype=freqs_.dtype).to(dtype=torch.float32)
        if simulate_cached and comp_with_fp32
        else torch.cos(freqs)
    )
    sin = (
        torch.sin(freqs).to(dtype=freqs_.dtype).to(dtype=torch.float32)
        if simulate_cached and comp_with_fp32
        else torch.sin(freqs)
    )
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return (
        torch.cat((x_forward, x_embed.to(dtype=x.dtype)), dim=-1).to(dtype=x_.dtype)
        if nope_first
        else torch.cat((x_embed.to(dtype=x.dtype), x_forward), dim=-1).to(
            dtype=x_.dtype
        )
    )

@perftest()
def hip_rope_cached_positions_offsets_2d_fwd_inplace(
    input_x,
    input_y,
    cos,
    sin,
    positions,
    offsets,
    rotate_style,
    reuse_freqs_front_part,
    nope_first,
):
    return aiter.rope_cached_positions_offsets_2c_fwd_inplace(
        input_x,
        input_y,
        cos,
        sin,
        positions,
        offsets,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
    )

def compare_rope_sbhd_2c_positions_with_legacy(
    input_x,
    input_y,
    freqs,
    positions,
    offsets,
    rotate_style,
    nope_first,
    check_correction=False,
):
    input_msg = f"""dtype: {input_x.dtype}, \
freq_dtype: {freqs.dtype}, \
dim_input: {str(input_x.shape):<20} - {str(input_y.shape):<20}, \
dim_freqs: {str(freqs.shape):<20}, \
dim_positions: {str(positions.shape):<20}, \
dim_offsets: {str(offsets.shape) if offsets is not None else 'None'}, \
rotate_style: {rotate_style.value}, \
nope_first: {nope_first}
"""

    s, b, h_x, d = input_x.shape

    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    # perftest cannot test correction of inplace operators
    if check_correction:
        ref_x = ref_rope_sbhd_fwd(
            input_x,
            freqs[
                positions if offsets is None else torch.add(positions, offsets)
            ].squeeze(-2),
            rotate_style,
            True,
            nope_first,
            True,
            True,
        )
        ref_y = ref_rope_sbhd_fwd(
            input_y,
            freqs[
                positions if offsets is None else torch.add(positions, offsets)
            ].squeeze(-2),
            rotate_style,
            True,
            nope_first,
            True,
            True,
        )
        h_y = input_y.shape[2]
        hip_input_x, hip_input_y = input_x, input_y
        aiter.rope_cached_positions_offsets_2c_fwd_inplace(
            hip_input_x,
            hip_input_y,
            cos,
            sin,
            positions,
            offsets,
            rotate_style,
            True,
            nope_first,
        )

        checkAllclose(
            ref_x,
            hip_input_x,
            rtol=1e-3,
            atol=1e-3,
            msg=f"correction: hip_fwd_x - {input_msg}\n",
        )
        checkAllclose(
            ref_y,
            hip_input_y,
            rtol=1e-3,
            atol=1e-3,
            msg=f"correction: hip_fwd_y - {input_msg}\n",
        )

    hip_cached_fwd_avg = 0.0001
    _, hip_cached_fwd_avg = hip_rope_cached_positions_offsets_2d_fwd_inplace(
        input_x,
        input_y,
        cos,
        sin,
        positions,
        offsets,
        rotate_style,
        True,
        nope_first,
    )

    color = "\033[92m"
    endc = "\033[0m"
    print(
        f"{color}{input_msg}hip: {hip_cached_fwd_avg:<8.2f} us.\n{endc}"
    )

@benchmark()
def test_reshape_and_cache(
    key,
    value,
    ctx_lens: int,
    bs: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    DType_KV: torch.dtype,
    DType_KVCache: torch.dtype,
):
    ret = {}
    MAX_TOKEN_SUPPORTED = 16384
    quantCfg = (
        {}
        if DType_KVCache in [dtypes.bf16, dtypes.fp16]
        else {"quant_dtype": DType_KVCache}
    )
    asm_layout = True
    kvhead = num_heads
    num_blocks = (MAX_TOKEN_SUPPORTED + block_size - 1) // block_size
    max_token_num_support = num_blocks * block_size
    x = 16 // DType_KVCache.itemsize
    if asm_layout:
        k_cache_shape = (bs * num_blocks, kvhead, head_size // x, block_size, x)
        v_cache_shape = (bs * num_blocks, kvhead, block_size // x, head_size, x)
        kv_scale_shape = (bs * num_blocks, kvhead, block_size)
    else:
        k_cache_shape = (bs * num_blocks, kvhead, head_size // x, block_size, x)
        v_cache_shape = (bs * num_blocks, kvhead, head_size, block_size)
        kv_scale_shape = (kvhead, bs * max_token_num_support)

    device = key.device
    k_cache = torch.empty(k_cache_shape, dtype=DType_KVCache, device=device)
    v_cache = torch.empty(v_cache_shape, dtype=DType_KVCache, device=device)
    if quantCfg:
        k_scale = torch.empty(kv_scale_shape, device=key.device)
        v_scale = torch.empty_like(k_scale)
        quantCfg["k_scale"] = k_scale.clone()
        quantCfg["v_scale"] = v_scale.clone()
    slot_mapping = torch.tensor(
        [
            bsID * max_token_num_support + i
            for bsID in range(bs)
            for i in range(ctx_lens)
        ]
    ).cuda()

    k_cache_a = k_cache.clone()
    v_cache_a = v_cache.clone()
    if quantCfg:
        quantCfg["k_scale"] = k_scale.clone()
        quantCfg["v_scale"] = v_scale.clone()
    out_a, us_a = run_aiter(
        key,
        value,
        k_cache_a,
        v_cache_a,
        slot_mapping,
        block_size,
        x,
        asm_layout,
        quantCfg,
    )
    ret["us_prefill"] = us_a

    return ret

@perftest()
def hip_rope_cached_positions_offsets_2c_fwd_inplace_cachekv(
    query_a,
    key_a,
    value_a,
    k_cache_a,
    v_cache_a,
    cos,
    sin,
    positions,
    offsets,
    rotate_style,
    reuse_freqs_front_part,
    nope_first,
    slot_mapping,
    asm_layout,
):
    return aiter.rope_cached_positions_offsets_2c_fwd_inplace_cachekv(
        query_a,
        key_a,
        value_a,
        k_cache_a,
        v_cache_a,
        cos,
        sin,
        positions,
        offsets,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        slot_mapping,
        asm_layout,
    )

@benchmark()
def test_rope_cached_positions_offsets_2c_fwd_inplace_cachekv(
    s: int,
    b: int,
    h: Tuple[int, int],
    d: int,
    block_size: int,
    DType_KV: torch.dtype,
    DType_KVCache: torch.dtype,
):
    MAX_TOKEN_SUPPORTED = 16384
    ret = {}
    quantCfg = (
        {}
        if DType_KVCache in [dtypes.bf16, dtypes.fp16]
        else {"quant_dtype": DType_KVCache}
    )
    # q, k, v
    qhead, kvhead = h
    query = torch.randn((s,b,qhead,d), dtype=dtype, device="cuda")
    key = torch.randn((s,b,kvhead,d), dtype=dtype, device="cuda")
    value = torch.randn((s,b,kvhead,d), dtype=dtype, device="cuda")

    # rope
    rotary_percent_and_reuse = (1.0, True, False)
    rotate_style = RotateStyle.NEOX
    rotary_percent = rotary_percent_and_reuse[0]
    reuse_freqs_front_part = rotary_percent_and_reuse[1]
    nope_first = (rotary_percent >= 1.0) and rotary_percent_and_reuse[2]
    freqs_ratio = 2 if reuse_freqs_front_part else 1
    freqs = torch.randn(
        (s * 2, 1, 1, int(d * rotary_percent) // freqs_ratio),
        dtype=dtype,
        device="cuda",
    )
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    positions = torch.randint(
        int(s * 0.25),
        int(s * 0.75),
        (
            s,
            b,
        ),
        device="cuda",
    )
    offsets = torch.randint(
        int(s * -0.25),
        int(s * 0.25),
        (
            s,
            b,
        ),
        device="cuda",
    )
    # kv cache
    num_blocks = (MAX_TOKEN_SUPPORTED + block_size - 1) // block_size
    max_token_num_support = num_blocks * block_size
    x = 16 // DType_KVCache.itemsize
    asm_layout = True
    if asm_layout:
        k_cache_shape = (b * num_blocks, kvhead, d // x, block_size, x)
        v_cache_shape = (b * num_blocks, kvhead, block_size // x, d, x)
        kv_scale_shape = (b * num_blocks, kvhead, block_size)
    else:
        k_cache_shape = (b * num_blocks, kvhead, d // x, block_size, x)
        v_cache_shape = (b * num_blocks, kvhead, d, block_size)
        kv_scale_shape = (kvhead, b * max_token_num_support)
    k_cache = torch.empty(k_cache_shape, dtype=DType_KVCache, device="cuda")
    v_cache = torch.empty(v_cache_shape, dtype=DType_KVCache, device="cuda")
    slot_mapping = torch.tensor(
        [
            bsID * max_token_num_support + i
            for i in range(s)
            for bsID in range(b)
        ]
    ).cuda()

    query_a = query.clone()
    key_a = key.clone()
    value_a = value.clone()
    k_cache_a = k_cache.clone()
    v_cache_a = v_cache.clone()

    hip_seperate_avg = 0.0001
    hip_fuse_avg = 0.0001

    _, hip_rope_avg = hip_rope_cached_positions_offsets_2d_fwd_inplace(
        query,
        key,
        cos,
        sin,
        positions,
        offsets,
        rotate_style,
        True,
        nope_first,
    )
    # print(hip_rope_avg)
    key_cache_kernel = key.permute(1,0,2,3).reshape(b*s,h_y,d)
    value_cache_kernel = value.permute(1,0,2,3).reshape(b*s,h_y,d)
    ret = test_reshape_and_cache(
                key_cache_kernel,
                value_cache_kernel,
                s,
                b,
                h_y,
                d,
                16,
                dtype,
                dtype,
            )
    hip_cache_avg = ret["us_prefill"]
    # print(hip_cache_avg)

    _, hip_fuse_avg = hip_rope_cached_positions_offsets_2c_fwd_inplace_cachekv(
        query_a,
        key_a,
        value_a,
        k_cache_a,
        v_cache_a,
        cos,
        sin,
        positions,
        offsets,
        rotate_style,
        True,
        nope_first,
        slot_mapping,
        asm_layout,
    )
    hip_seperate_avg = hip_rope_avg + hip_cache_avg
    # print(hip_fuse_avg, hip_seperate_avg)
    print(f"sep vs fuse {hip_seperate_avg:>8.2f}us vs {hip_fuse_avg:>8.2f}us")

if __name__ == "__main__":
    l_dtype = ("fp16", "bf16")
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--no_check",
        action="store_true",
        help="""Do not check correctness of ops. Default: False.
    --no_check    # True""",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="""Compare with legacy implementation. Default: False
    --compare    # True""",
    )
    parser.add_argument(
        "--compare_check",
        action="store_true",
        help="""Check correctness when compare with legacy implementation. Default: False
    --compare_check    # True""",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=str,
        choices=l_dtype,
        nargs="?",
        const=None,
        default=None,
        help="""Data type.
    e.g.: -d bf16""",
    )
    parser.add_argument(
        "-t",
        "--transpose_output",
        default=(False, True),
        nargs="*",
        type=dtypes.str2bool,
        help="""Transpose output. Default: (False, True).
    e.g.: -t f   # for False
       or -t t   # for True""",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=[4],
        nargs="*",
        help="""Batch sizes for testing. The default is 4, but you can choose from: 1, 2, 4.
    e.g.: -b 1""",
    )
    parser.add_argument(
        "-s",
        "--seq_size",
        type=int,
        default=[512],
        nargs="*",
        help="""Sequence sizes to test. Default: 2048, but you can choose from: 1024, 2048, 4096.
    e.g.: -s 1024""",
    )
    parser.add_argument(
        "-hs",
        "--head_size",
        type=int,
        default=[64],
        nargs="*",
        help="""Head sizes to test. Default is 64, but you can choose from: 32, 64.
    e.g.: -hs 32""",
    )
    parser.add_argument(
        "-hd",
        "--hidden_dim",
        type=int,
        default=[256],
        nargs="*",
        help="""Hidden dimensions to test. Default is 256, bui you can choose from: 128, 256.
    e.g.: -hd 128""",
    )
    parser.add_argument(
        "-ht",
        "--height",
        default=[64],
        nargs="*",
        type=int,
        help="""Height sizes to test. Default is 64, but you can choose from: 32, 64.
    e.g.: -ht 32""",
    )
    parser.add_argument(
        "-wd",
        "--width",
        default=[64],
        nargs="*",
        type=int,
        help="""Width sizes to test. Default is 64, but you can choose from: 32, 64.
    e.g.: -wd 32""",
    )
    parser.add_argument(
        "-m",
        "--margin",
        default=[0, 3],
        nargs="*",
        type=int,
        help="""Margin sizes to test. Default is [0,3].
    e.g.: -m 0""",
    )
    d_rs = {"neox": RotateStyle.NEOX, "gptj": RotateStyle.GPTJ}
    parser.add_argument(
        "-rs",
        "--rotate_style",
        default=list(d_rs.keys()),
        type=str,
        choices=list(d_rs.keys()),
        nargs="*",
        help="""Rotate style. Default is all combinations of neox and gptj.
    e.g.: -rs neox
          or -rs gptj""",
    )
    d_rr = {
        # [0]: rotary percentage, [1]: reuse front part, [2]: nope first
        0: (1.0, True, False),
        1: (1.0, False, False),
        2: (0.5, False, False),
        3: (0.5, True, False),
        4: (0.5, True, True),
        5: (0.5, False, True),
    }
    parser.add_argument(
        "-rr",
        "--rotary_percent_and_reuse",
        default=list(d_rr.keys()),
        type=int,
        nargs="*",
        choices=list(d_rr.keys()),
        help="""Rotary percentage and reuse front part. Default is all combinations of:
    e.g.: -rr 0     # for (1.0, True, False)
          or -rr 1  # for (1.0, False, False)
          or -rr 2  # for (0.5, False, False)
          or -rr 3  # for (0.5, True, False)
          or -rr 4  # for (0.5, True, True)
          or -rr 5  # for (0.5, False, True)""",
    )

    args = parser.parse_args()
    if args.dtype is None:
        l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
    else:
        l_dtype = [dtypes.d_dtypes[args.dtype]]

    args.rotate_style = [d_rs[rs] for rs in args.rotate_style]
    args.rotary_percent_and_reuse = [d_rr[rr] for rr in args.rotary_percent_and_reuse]

    # Compare new with legacy
    # if args.compare:
    if True:
        # [0]: rotary percentage, [1]: reuse front part, [2]: nope first
        # reuse front part should always be True here since legacy implementation doesn't support the opposite setting.
        rotary_percent_and_reuse_compare_ = (
            (1.0, True, False),
            (0.5, True, False),
        )
        for (
            dtype,
            rotate_style,
            rotary_percent_and_reuse,
            has_offsets,
            b,
            s,
            h_x,
            h_y,
            d,
        ) in itertools.product(
            l_dtype,  # legacy implementation doesn't support different scalar type between input/output and freqs/sin/cos
            args.rotate_style,
            rotary_percent_and_reuse_compare_,
            (True, True),
            args.batch_size,
            args.seq_size,
            args.head_size,
            args.head_size,
            args.hidden_dim,
        ):
            color, endc = "\033[95m", "\033[0m"
            print(
                f"{color}dtype: {dtype}, rotate_style: {rotate_style}, rpar: {rotary_percent_and_reuse}, (s,b,hx,hy,d): {s, b, h_x, h_y, d}, has_offsets: {has_offsets}{endc}"
            )
            test_rope_cached_positions_offsets_2c_fwd_inplace_cachekv(
                s,
                b,
                [h_x, h_y],
                d,
                16,
                dtype,
                dtype,
            )
            break
        print("test over!")