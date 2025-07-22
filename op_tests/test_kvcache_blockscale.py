# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter.test_common import checkAllclose, run_perftest, benchmark
from typing import Tuple
from aiter import dtypes
import functools
import argparse

MAX_TOKEN_SUPPORTED = 16384


def run_torch(
    key,
    value,
    k_cache,
    v_cache,
    k_scale,
    v_scale,
    slot_mapping,
    block_size,
    x,
    asm_layout,
    quantCfg={},
):
    num_batch, num_tokens, num_heads, head_size = key.shape
    num_blocks = k_cache.shape[0]
    k_cache_shape = k_cache.shape
    v_cache_shape = v_cache.shape

    key = key.to(dtypes.fp32).contiguous()
    value = value.to(dtypes.fp32).contiguous()

    if asm_layout:
        k_cache = k_cache.view(num_blocks, num_heads, -1).to(
            dtypes.fp32
        ) * k_scale.view(num_blocks, num_heads, 1)
        v_cache = v_cache.view(num_blocks, num_heads, -1).to(
            dtypes.fp32
        ) * v_scale.view(num_blocks, num_heads, 1)
    else:
        k_cache = k_cache.view(num_blocks, num_heads, -1).to(
            dtypes.fp32
        ) * k_scale.t().view(num_blocks, num_heads, 1)
        v_cache = v_cache.view(num_blocks, num_heads, -1).to(
            dtypes.fp32
        ) * v_scale.t().view(num_blocks, num_heads, 1)

    # [num_blocks, num_heads, head_size//x, block_size, x]
    k_cache = (
        k_cache.view(k_cache_shape)
        .permute(0, 3, 1, 2, 4)
        .contiguous()
        .view(-1, num_heads, head_size)
    )

    if asm_layout:
        # [num_blocks, num_heads, block_size//x, head_size, x]
        v_cache = (
            v_cache.view(v_cache_shape)
            .permute(0, 2, 4, 1, 3)
            .contiguous()
            .view(-1, num_heads, head_size)
        )
    else:
        # [num_blocks, num_heads, head_size, block_size]
        v_cache = (
            v_cache.view(v_cache_shape)
            .permute(0, 3, 1, 2)
            .contiguous()
            .view(-1, num_heads, head_size)
        )

    k_cache[slot_mapping] = key.view(-1, num_heads, head_size)
    k_cache = (
        k_cache.view(num_blocks, block_size, num_heads, head_size)
        .permute(0, 2, 1, 3)
        .view(num_blocks, num_heads, -1)
    )
    k_cache, k_scale = aiter.pertoken_quant(
        k_cache,
        scale_dtype=quantCfg["y_scale_dtype"],
        quant_dtype=quantCfg["quant_dtype"],
    )
    k_cache = (
        k_cache.view(num_blocks, num_heads, block_size, head_size // x, x)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    k_scale = k_scale.view(num_blocks, num_heads)

    v_cache[slot_mapping] = value.view(-1, num_heads, head_size)
    v_cache = (
        v_cache.view(num_blocks, block_size, num_heads, head_size)
        .permute(0, 2, 1, 3)
        .view(num_blocks, num_heads, -1)
    )
    v_cache, v_scale = aiter.pertoken_quant(
        v_cache,
        scale_dtype=quantCfg["y_scale_dtype"],
        quant_dtype=quantCfg["quant_dtype"],
    )
    v_scale = v_scale.view(num_blocks, num_heads)

    if asm_layout:
        v_cache = (
            v_cache.view(num_blocks, num_heads, block_size // x, x, head_size)
            .permute(0, 1, 2, 4, 3)
            .contiguous()
        )
    else:
        k_scale = k_scale.t().contiguous()
        v_scale = v_scale.t().contiguous()
        v_cache = (
            v_cache.view(num_blocks, num_heads, block_size, head_size)
            .permute(0, 1, 3, 2)
            .contiguous()
        )

    return k_cache, v_cache, k_scale, v_scale


def run_aiter(
    key,
    value,
    k_cache,
    v_cache,
    k_scale,
    v_scale,
    slot_mapping,
    block_size,
    x,
    asm_layout,
    quantCfg={},
):
    aiter.reshape_and_cache_with_block_quant(
        key, value, k_cache, v_cache, k_scale, v_scale, slot_mapping, asm_layout
    )
    return k_cache, v_cache, k_scale, v_scale


def run_torch_for_asmpa(
    key,
    value,
    k_cache,
    v_cache,
    k_scale,
    v_scale,
    slot_mapping,
    block_size,
    x,
    asm_layout,
    quantCfg={},
    ori_block_size=128,
):
    block_split = ori_block_size // block_size
    num_batch, num_tokens, num_heads, head_size = key.shape
    num_blocks = k_cache.shape[0]
    k_cache_shape = k_cache.shape
    v_cache_shape = v_cache.shape
    ori_num_blocks = num_blocks // block_split

    key = key.to(dtypes.fp32).contiguous()
    value = value.to(dtypes.fp32).contiguous()

    if asm_layout:
        k_cache = k_cache.view(num_blocks, num_heads, -1).to(
            dtypes.fp32
        ) * k_scale.view(ori_num_blocks, num_heads, 1).repeat(
            1, 1, block_split
        ).permute(
            0, 2, 1
        ).view(
            num_blocks, num_heads, 1
        )
        v_cache = v_cache.view(num_blocks, num_heads, -1).to(
            dtypes.fp32
        ) * v_scale.view(ori_num_blocks, num_heads, 1).repeat(
            1, 1, block_split
        ).permute(
            0, 2, 1
        ).view(
            num_blocks, num_heads, 1
        )
    else:
        k_cache = k_cache.view(num_blocks, num_heads, -1).to(
            dtypes.fp32
        ) * k_scale.t().view(ori_num_blocks, num_heads, 1).repeat(
            1, 1, block_split
        ).permute(
            0, 2, 1
        ).view(
            num_blocks, num_heads, 1
        )
        v_cache = v_cache.view(num_blocks, num_heads, -1).to(
            dtypes.fp32
        ) * v_scale.t().view(ori_num_blocks, num_heads, 1).repeat(
            1, 1, block_split
        ).permute(
            0, 2, 1
        ).view(
            num_blocks, num_heads, 1
        )

    # [num_blocks, num_heads, head_size//x, block_size, x]
    k_cache = (
        k_cache.view(k_cache_shape)
        .permute(0, 3, 1, 2, 4)
        .contiguous()
        .view(-1, num_heads, head_size)
    )

    if asm_layout:
        # [num_blocks, num_heads, block_size//x, head_size, x]
        v_cache = (
            v_cache.view(v_cache_shape)
            .permute(0, 2, 4, 1, 3)
            .contiguous()
            .view(-1, num_heads, head_size)
        )
    else:
        # [num_blocks, num_heads, head_size, block_size]
        v_cache = (
            v_cache.view(v_cache_shape)
            .permute(0, 3, 1, 2)
            .contiguous()
            .view(-1, num_heads, head_size)
        )

    k_cache[slot_mapping] = key.view(-1, num_heads, head_size)
    k_cache = (
        k_cache.view(ori_num_blocks, ori_block_size, num_heads, head_size)
        .permute(0, 2, 1, 3)
        .view(ori_num_blocks, num_heads, -1)
    )
    k_cache, k_scale = aiter.pertoken_quant(
        k_cache,
        scale_dtype=quantCfg["y_scale_dtype"],
        quant_dtype=quantCfg["quant_dtype"],
    )
    k_cache = (
        k_cache.view(ori_num_blocks, num_heads, block_split, block_size, -1)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )
    k_cache = (
        k_cache.view(num_blocks, num_heads, block_size, head_size // x, x)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    k_scale = k_scale.view(ori_num_blocks, num_heads)

    v_cache[slot_mapping] = value.view(-1, num_heads, head_size)
    v_cache = (
        v_cache.view(ori_num_blocks, ori_block_size, num_heads, head_size)
        .permute(0, 2, 1, 3)
        .view(ori_num_blocks, num_heads, -1)
    )
    v_cache, v_scale = aiter.pertoken_quant(
        v_cache,
        scale_dtype=quantCfg["y_scale_dtype"],
        quant_dtype=quantCfg["quant_dtype"],
    )
    v_cache = (
        v_cache.view(ori_num_blocks, num_heads, block_split, block_size, -1)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
        .view(num_blocks, num_heads, block_split, block_size, -1)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )
    v_scale = v_scale.view(ori_num_blocks, num_heads)

    if asm_layout:
        v_cache = (
            v_cache.view(num_blocks, num_heads, block_size // x, x, head_size)
            .permute(0, 1, 2, 4, 3)
            .contiguous()
        )
    else:
        k_scale = k_scale.t().contiguous()
        v_scale = v_scale.t().contiguous()
        v_cache = (
            v_cache.view(num_blocks, num_heads, block_size, head_size)
            .permute(0, 1, 3, 2)
            .contiguous()
        )

    return k_cache, v_cache, k_scale, v_scale


def run_aiter_for_asmpa(
    key,
    value,
    k_cache,
    v_cache,
    k_scale,
    v_scale,
    slot_mapping,
    block_size,
    x,
    asm_layout,
    quantCfg={},
    ori_block_size=128,
):
    aiter.reshape_and_cache_with_block_quant_for_asm_pa(
        key,
        value,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        slot_mapping,
        asm_layout,
        ori_block_size,
    )
    return k_cache, v_cache, k_scale, v_scale


@benchmark()
def test_reshape_and_cache(
    ctx_lens: int,
    bs: int,
    num_heads: Tuple[int, int],
    head_size: int,
    block_size: int,
    DTyoe_KV: torch.dtype,
    DTyoe_KVCache: torch.dtype,
    quantCfg: dict = {},
    ori_block_size=0,  # test for asm pa , only support 128 / 256, 0 mean off
):
    asm_layout = True
    qhead, kvhead = num_heads
    if ori_block_size:
        num_blocks = (
            (MAX_TOKEN_SUPPORTED + ori_block_size - 1)
            // ori_block_size
            * ori_block_size
            // block_size
        )
    else:
        num_blocks = (MAX_TOKEN_SUPPORTED + block_size - 1) // block_size
    # num_blocks = (ctx_lens+1+block_size-1)//block_size
    max_token_num_support = num_blocks * block_size
    x = 16 // DTyoe_KVCache.itemsize
    if asm_layout:
        k_cache_shape = (bs * num_blocks, kvhead, head_size // x, block_size, x)
        v_cache_shape = (bs * num_blocks, kvhead, block_size // x, head_size, x)
        if ori_block_size:
            kv_scale_shape = (bs * num_blocks // (ori_block_size // block_size), kvhead)
        else:
            kv_scale_shape = (bs * num_blocks, kvhead)
    else:
        k_cache_shape = (bs * num_blocks, kvhead, head_size // x, block_size, x)
        v_cache_shape = (bs * num_blocks, kvhead, head_size, block_size)
        if ori_block_size:
            kv_scale_shape = (kvhead, bs * num_blocks // (ori_block_size // block_size))
        else:
            kv_scale_shape = (kvhead, bs * num_blocks)

    # ##################################################### prefill part
    qkv = torch.randn(
        bs * ctx_lens, qhead + 2 * kvhead, head_size, dtype=DTyoe_KV, device="cuda"
    )
    _, key, value = torch.split(qkv, [qhead, kvhead, kvhead], dim=1)
    device = key.device
    k_cache = torch.zeros(k_cache_shape, dtype=DTyoe_KVCache, device=device)
    v_cache = torch.zeros(v_cache_shape, dtype=DTyoe_KVCache, device=device)
    k_scale = torch.ones(
        kv_scale_shape, dtype=quantCfg["y_scale_dtype"], device=key.device
    )
    v_scale = torch.ones_like(k_scale)
    slot_mapping = torch.tensor(
        [
            bsID * max_token_num_support + i
            for bsID in range(bs)
            for i in range(ctx_lens)
        ]
    ).cuda()

    torch_func = (
        functools.partial(run_torch_for_asmpa, ori_block_size=ori_block_size)
        if ori_block_size
        else run_torch
    )
    aiter_func = (
        functools.partial(run_aiter_for_asmpa, ori_block_size=ori_block_size)
        if ori_block_size
        else run_aiter
    )

    k_cache_ref = k_cache.clone()
    v_cache_ref = v_cache.clone()
    k_scale_ref = k_scale.clone()
    v_scale_ref = v_scale.clone()
    (k_cache_ref, v_cache_ref, k_scale_ref, v_scale_ref), us_ref = run_perftest(
        torch_func,
        key.view(bs, ctx_lens, kvhead, head_size),
        value.view(bs, ctx_lens, kvhead, head_size),
        k_cache_ref,
        v_cache_ref,
        k_scale_ref,
        v_scale_ref,
        slot_mapping,
        block_size,
        x,
        asm_layout,
        quantCfg,
        num_iters=3,
    )

    k_cache_a = k_cache.clone()
    v_cache_a = v_cache.clone()
    k_scale_a = k_scale.clone()
    v_scale_a = v_scale.clone()
    (k_cache_a, v_cache_a, k_scale_a, v_scale_a), us_a = run_perftest(
        aiter_func,
        key.view(bs, ctx_lens, kvhead, head_size),
        value.view(bs, ctx_lens, kvhead, head_size),
        k_cache_a,
        v_cache_a,
        k_scale_a,
        v_scale_a,
        slot_mapping,
        block_size,
        x,
        asm_layout,
        quantCfg,
    )

    print(f"prefill part: ref vs aiter {us_ref:.2f}us vs {us_a:.2f}us")
    slots_edit = torch.unique(slot_mapping // block_size)
    checkAllclose(
        k_cache_ref.to(dtypes.fp32)[slots_edit],
        k_cache_a.to(dtypes.fp32)[slots_edit],
        msg=f"k_cache {k_cache_ref.shape}",
    )
    checkAllclose(
        v_cache_ref.to(dtypes.fp32)[slots_edit],
        v_cache_a.to(dtypes.fp32)[slots_edit],
        msg=f"v_cache {v_cache_ref.shape}",
    )
    if ori_block_size:
        slots_edit = slots_edit // (ori_block_size // block_size)
    if not asm_layout:
        k_scale_ref = k_scale_ref.t()
        v_scale_ref = v_scale_ref.t()
        k_scale_a = k_scale_a.t()
        v_scale_a = v_scale_a.t()

    checkAllclose(
        k_scale_ref[slots_edit],
        k_scale_a[slots_edit],
        atol=0.001,
        msg=f"k_scale {k_scale_ref.shape}",
    )
    checkAllclose(
        v_scale_ref[slots_edit],
        v_scale_a[slots_edit],
        atol=0.001,
        msg=f"v_scale {v_scale_ref.shape}",
    )

    ##################################################### chunk-prefill part
    chunk_left_ctx_lens = 10

    qkv = (
        torch.randn(
            bs * chunk_left_ctx_lens,
            qhead + 2 * kvhead,
            head_size,
            dtype=DTyoe_KV,
            device="cuda",
        )
        * 2
    )
    _, key, value = torch.split(qkv, [qhead, kvhead, kvhead], dim=1)

    slot_mapping = torch.tensor(
        [
            bsID * max_token_num_support + ctx_lens + i
            for bsID in range(bs)
            for i in range(chunk_left_ctx_lens)
        ]
    ).cuda()

    if not asm_layout:
        k_scale_ref = k_scale_ref.t()
        v_scale_ref = v_scale_ref.t()
        k_scale_a = k_scale_a.t()
        v_scale_a = v_scale_a.t()

    (k_cache_ref, v_cache_ref, k_scale_ref, v_scale_ref), us_ref = run_perftest(
        torch_func,
        key.view(bs, chunk_left_ctx_lens, kvhead, head_size),
        value.view(bs, chunk_left_ctx_lens, kvhead, head_size),
        k_cache_ref,
        v_cache_ref,
        k_scale_ref,
        v_scale_ref,
        slot_mapping,
        block_size,
        x,
        asm_layout,
        quantCfg,
        num_iters=3,
    )

    (k_cache_a, v_cache_a, k_scale_a, v_scale_a), us_a = run_perftest(
        aiter_func,
        key.view(bs, chunk_left_ctx_lens, kvhead, head_size),
        value.view(bs, chunk_left_ctx_lens, kvhead, head_size),
        k_cache_a,
        v_cache_a,
        k_scale_a,
        v_scale_a,
        slot_mapping,
        block_size,
        x,
        asm_layout,
        quantCfg,
    )

    print(f"chunk-prefill part: ref vs aiter {us_ref:.2f}us vs {us_a:.2f}us")
    slots_edit = torch.unique(slot_mapping // block_size)
    checkAllclose(
        k_cache_ref.to(dtypes.fp32)[slots_edit],
        k_cache_a.to(dtypes.fp32)[slots_edit],
        msg=f"k_cache {k_cache_ref.shape}",
    )
    checkAllclose(
        v_cache_ref.to(dtypes.fp32)[slots_edit],
        v_cache_a.to(dtypes.fp32)[slots_edit],
        msg=f"v_cache {v_cache_ref.shape}",
    )
    if ori_block_size:
        slots_edit = slots_edit // (ori_block_size // block_size)
    if not asm_layout:
        k_scale_ref = k_scale_ref.t()
        v_scale_ref = v_scale_ref.t()
        k_scale_a = k_scale_a.t()
        v_scale_a = v_scale_a.t()
    checkAllclose(
        k_scale_ref[slots_edit],
        k_scale_a[slots_edit],
        atol=0.001,
        msg=f"k_scale {k_scale_ref.shape}",
    )
    checkAllclose(
        v_scale_ref[slots_edit],
        v_scale_a[slots_edit],
        atol=0.001,
        msg=f"v_scale {v_scale_ref.shape}",
    )

    ##################################################### decode part
    qkv = (
        torch.randn(bs, qhead + 2 * kvhead, head_size, dtype=DTyoe_KV, device="cuda")
        * 2
    )
    _, key, value = torch.split(qkv, [qhead, kvhead, kvhead], dim=1)

    slot_mapping = torch.tensor(
        [
            bsID * max_token_num_support + ctx_lens + chunk_left_ctx_lens
            for bsID in range(bs)
        ]
    ).cuda()

    if not asm_layout:
        k_scale_ref = k_scale_ref.t()
        v_scale_ref = v_scale_ref.t()
        k_scale_a = k_scale_a.t()
        v_scale_a = v_scale_a.t()

    (k_cache_ref, v_cache_ref, k_scale_ref, v_scale_ref), us_ref = run_perftest(
        torch_func,
        key.view(bs, 1, kvhead, head_size),
        value.view(bs, 1, kvhead, head_size),
        k_cache_ref,
        v_cache_ref,
        k_scale_ref,
        v_scale_ref,
        slot_mapping,
        block_size,
        x,
        asm_layout,
        quantCfg,
        num_iters=3,
    )

    (k_cache_a, v_cache_a, k_scale_a, v_scale_a), us_a = run_perftest(
        aiter_func,
        key.view(bs, 1, kvhead, head_size),
        value.view(bs, 1, kvhead, head_size),
        k_cache_a,
        v_cache_a,
        k_scale_a,
        v_scale_a,
        slot_mapping,
        block_size,
        x,
        asm_layout,
        quantCfg,
    )

    print(f"decode part: ref vs aiter {us_ref:.2f}us vs {us_a:.2f}us")
    slots_edit = torch.unique(slot_mapping // block_size)
    checkAllclose(
        k_cache_ref.to(dtypes.fp32)[slots_edit],
        k_cache_a.to(dtypes.fp32)[slots_edit],
        msg=f"k_cache {k_cache_ref.shape}",
    )
    checkAllclose(
        v_cache_ref.to(dtypes.fp32)[slots_edit],
        v_cache_a.to(dtypes.fp32)[slots_edit],
        msg=f"v_cache {v_cache_ref.shape}",
    )
    if ori_block_size:
        slots_edit = slots_edit // (ori_block_size // block_size)
    if not asm_layout:
        k_scale_ref = k_scale_ref.t()
        v_scale_ref = v_scale_ref.t()
        k_scale_a = k_scale_a.t()
        v_scale_a = v_scale_a.t()
    checkAllclose(
        k_scale_ref[slots_edit],
        k_scale_a[slots_edit],
        atol=0.001,
        msg=f"k_scale {k_scale_ref.shape}",
    )
    checkAllclose(
        v_scale_ref[slots_edit],
        v_scale_a[slots_edit],
        atol=0.001,
        msg=f"v_scale {v_scale_ref.shape}",
    )

    print(
        f"finish test {ctx_lens=} {bs=} {num_heads=} {head_size=} {block_size=} {DTyoe_KV=} {DTyoe_KVCache=}"
    )


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="selec run whicn test.",
)
parser.add_argument(
    "-t",
    "--test",
    type=str,
    choices=[
        "fp16tofp8",
        "fp16toi8",
        "bf16toi8",
        "bf16tofp8",
        "bf16tofp8_asm_pa128",
        "bf16tofp8_asm_pa256",
    ],
    nargs="*",
    default=[
        "fp16tofp8",
        "fp16toi8",
        "bf16toi8",
        "bf16tofp8",
        "bf16tofp8_asm_pa128",
        "bf16tofp8_asm_pa256",
    ],
    help="""select which test to run, default is all
    e.g.: -t fp16tofp8""",
)
args = parser.parse_args()
for test in args.test:
    if test == "fp16tofp8":
        print("\nstart quant fp16->fp8")
        test_reshape_and_cache(
            4097,
            128,
            (8, 1),
            128,
            16,
            dtypes.fp16,
            dtypes.fp8,
            quantCfg={"y_scale_dtype": dtypes.fp32, "quant_dtype": dtypes.fp8},
        )
    elif test == "fp16toi8":
        print("\nstart quant fp16->i8")
        test_reshape_and_cache(
            4097,
            128,
            (8, 1),
            128,
            16,
            dtypes.fp16,
            dtypes.i8,
            quantCfg={"y_scale_dtype": dtypes.fp32, "quant_dtype": dtypes.i8},
        )
    elif test == "bf16toi8":
        print("\nstart quant bf16->i8")
        test_reshape_and_cache(
            4097,
            128,
            (8, 1),
            128,
            16,
            dtypes.bf16,
            dtypes.i8,
            quantCfg={"y_scale_dtype": dtypes.fp32, "quant_dtype": dtypes.i8},
        )
    elif test == "bf16tofp8":
        print("\nstart quant bf16->fp8")
        test_reshape_and_cache(
            4097,
            128,
            (8, 1),
            128,
            128,
            dtypes.bf16,
            dtypes.fp8,
            quantCfg={"y_scale_dtype": dtypes.fp32, "quant_dtype": dtypes.fp8},
        )
    elif test == "bf16tofp8_asm_pa128":
        print("\nstart quant bf16->fp8, for asm pa ori_block_size=128")
        test_reshape_and_cache(
            4097,
            128,
            (8, 1),
            128,
            16,
            dtypes.bf16,
            dtypes.fp8,
            quantCfg={"y_scale_dtype": dtypes.fp32, "quant_dtype": dtypes.fp8},
            ori_block_size=128,
        )
    elif test == "bf16tofp8_asm_pa256":
        print("\nstart quant bf16->fp8, for asm pa ori_block_size=256")
        test_reshape_and_cache(
            4097,
            128,
            (8, 1),
            128,
            16,
            dtypes.bf16,
            dtypes.fp8,
            quantCfg={"y_scale_dtype": dtypes.fp32, "quant_dtype": dtypes.fp8},
            ori_block_size=256,
        )
    else:
        raise ValueError(f"unknown test {test}, please check the code.")
