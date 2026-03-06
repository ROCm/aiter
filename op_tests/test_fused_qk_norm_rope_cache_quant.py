# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
import aiter
from aiter.test_common import checkAllclose, perftest, benchmark
from aiter.utility.dtypes import get_dtype_fp8
from aiter.utility import dtypes
import argparse
import pandas as pd


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
    k_cache: Tensor,  # [num_blocks, num_heads_k, head_size // x, page_size, x]
    v_cache: Tensor,  # [num_blocks, num_heads_v, head_size, page_size]
    k_scale: Tensor,  # [num_blocks, page_size]
    v_scale: Tensor,  # [num_blocks, page_size]
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


@perftest(num_iters=2)
def run_torch_qk_norm_rope_cache_block_quant_shuffle(
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
    k_cache: Tensor,  # [num_blocks, num_heads_k, head_size // x, block_size, x]
    v_cache: Tensor,  # [num_blocks, num_heads_v, block_size // x, head_size, x]
    k_scale: Tensor,  # [num_blocks, num_kv_heads]
    v_scale: Tensor,  # [num_blocks, num_kv_heads]
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

    from aiter import reshape_and_cache

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
        # Block quant ref using pertoken_quant (same approach as test_kvcache_blockscale.py)
        # k_cache: [num_blocks, num_heads_k, head_size // x, block_size, x]
        # v_cache: [num_blocks, num_heads_v, block_size // x, head_size, x]
        num_blocks = k_cache.shape[0]
        block_size = k_cache.shape[-2]  # page_size
        x_val = k_cache.shape[-1]
        cache_dtype = k_cache.dtype

        # Step 1: Unflatten k_cache to [num_blocks, block_size, num_heads_k, head_size]
        # and DEQUANTIZE using old scale (multiply raw fp8 values by scale)
        # k_cache: [num_blocks, num_heads_k, head_size//x, block_size, x]
        #       -> permute(0, 3, 1, 2, 4) -> [num_blocks, block_size, num_heads_k, head_size//x, x]
        #       -> view [num_blocks, block_size, num_heads_k, head_size]
        k_cache_flat = (
            k_cache.float()
            .permute(0, 3, 1, 2, 4)
            .contiguous()
            .view(num_blocks, block_size, num_heads_k, head_size)
        )
        # Dequantize: k_cache_flat *= k_scale[num_blocks, num_heads_k] (broadcast over block_size, head_size)
        k_cache_flat = k_cache_flat * k_scale.view(num_blocks, 1, num_heads_k, 1)
        k_cache_flat = k_cache_flat.view(-1, num_heads_k, head_size)

        # v_cache: [num_blocks, num_heads_v, block_size//x, head_size, x]
        #       -> permute(0, 2, 4, 1, 3) -> [num_blocks, block_size//x, x, num_heads_v, head_size]
        #       -> view [num_blocks, block_size, num_heads_v, head_size]
        v_cache_flat = (
            v_cache.float()
            .permute(0, 2, 4, 1, 3)
            .contiguous()
            .view(num_blocks, block_size, num_heads_v, head_size)
        )
        # Dequantize: v_cache_flat *= v_scale[num_blocks, num_heads_v]
        v_cache_flat = v_cache_flat * v_scale.view(num_blocks, 1, num_heads_v, 1)
        v_cache_flat = v_cache_flat.view(-1, num_heads_v, head_size)

        # Step 2: Scatter K/V into dequantized cache
        k_flat = k.view(-1, num_heads_k, head_size)
        v_flat = v.view(-1, num_heads_v, head_size)
        k_cache_flat[slot_mapping] = k_flat.float()
        v_cache_flat[slot_mapping] = v_flat.float()

        # Step 3: Reshape to [num_blocks, num_heads, block_size*head_size] and pertoken_quant
        k_cache_for_quant = (
            k_cache_flat.view(num_blocks, block_size, num_heads_k, head_size)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(num_blocks, num_heads_k, -1)
        )
        k_cache_q, k_scale_new = aiter.pertoken_quant(
            k_cache_for_quant,
            scale_dtype=torch.float32,
            quant_dtype=cache_dtype,
        )
        k_scale_new = k_scale_new.view(num_blocks, num_heads_k)

        v_cache_for_quant = (
            v_cache_flat.view(num_blocks, block_size, num_heads_v, head_size)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(num_blocks, num_heads_v, -1)
        )
        v_cache_q, v_scale_new = aiter.pertoken_quant(
            v_cache_for_quant,
            scale_dtype=torch.float32,
            quant_dtype=cache_dtype,
        )
        v_scale_new = v_scale_new.view(num_blocks, num_heads_v)

        # Step 4: Reshape back to tiled layout
        # k_cache_q: [num_blocks, num_heads_k, block_size*head_size]
        #         -> view [num_blocks, num_heads_k, block_size, head_size//x, x]
        #         -> permute(0, 1, 3, 2, 4) -> [num_blocks, num_heads_k, head_size//x, block_size, x]
        k_cache.copy_(
            k_cache_q.view(
                num_blocks, num_heads_k, block_size, head_size // x_val, x_val
            )
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        # v_cache_q: [num_blocks, num_heads_v, block_size*head_size]
        #         -> view [num_blocks, num_heads_v, block_size, head_size]
        #         -> view [num_blocks, num_heads_v, block_size//x, x, head_size]
        #         -> permute(0, 1, 2, 4, 3) -> [num_blocks, num_heads_v, block_size//x, head_size, x]
        v_cache.copy_(
            v_cache_q.view(
                num_blocks, num_heads_v, block_size // x_val, x_val, head_size
            )
            .permute(0, 1, 2, 4, 3)
            .contiguous()
        )

        k_scale.copy_(k_scale_new)
        v_scale.copy_(v_scale_new)

    k = k.reshape(k_shape)
    v = v.reshape(k_shape)
    return q, k, v, k_cache, v_cache


@perftest(num_iters=31, num_rotate_args=31)
def run_aiter_qk_norm_rope_cache_block_quant_shuffle(
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
    cu_q_len: Tensor,
    kv_cache_dtype: str,
    k_scale: Tensor,
    v_scale: Tensor,
):
    qkv = qkv.clone()  # inplace op

    aiter.fused_qk_norm_rope_cache_block_quant_shuffle(
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
        cu_q_len,
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
    kv_cache_dtype,
    num_blocks,
    page_size,
):
    # Construct tensors inside the function
    if kv_cache_dtype == "fp8_e4m3":
        cache_dtype = get_dtype_fp8()
    else:
        cache_dtype = dtype

    k_cache = torch.randn(
        [num_blocks, page_size, num_heads_k, head_size],
        dtype=dtype,
        device="cuda",
    ).to(cache_dtype)
    v_cache = torch.randn(
        [num_blocks, page_size, num_heads_v, head_size],
        dtype=dtype,
        device="cuda",
    ).to(cache_dtype)

    x = 16 // k_cache.element_size()
    k_cache = (
        k_cache.view([num_blocks, page_size, num_heads_k, head_size // x, x])
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    v_cache = v_cache.permute(0, 2, 3, 1).contiguous()

    slot_mapping = torch.randperm(num_tokens, dtype=torch.int64, device="cuda")
    k_scale = torch.zeros(
        [num_blocks, num_heads_k, page_size], dtype=torch.float32, device="cuda"
    )
    v_scale = torch.zeros(
        [num_blocks, num_heads_v, page_size], dtype=torch.float32, device="cuda"
    )
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
    ret = {}
    ret["fused_qk_us"] = avg_cu
    ret["unfused_us"] = avg_torch
    ret["aiter_bw(TB/s)"] = (
        num_tokens
        * (num_heads_k + num_heads_v + num_heads_q)
        * head_size
        * (torch.finfo(dtype).bits // 8)
        + num_tokens * num_heads_q * head_size * (torch.finfo(dtype).bits // 8)
        + num_tokens * num_heads_k * head_size * (torch.finfo(cache_dtype).bits // 8)
        + num_tokens * num_heads_v * head_size * (torch.finfo(cache_dtype).bits // 8)
    ) / (avg_cu * 1e6)
    return ret


@perftest()
def run_torch_qk_norm_rope_2way(
    q0: Tensor,  # contiguous (batch_size * num_tokens0 * num_heads_q * head_size)
    k0: Tensor,  # contiguous (batch_size * num_tokens0 * num_heads_k * head_size)
    q1: Tensor,  # contiguous (batch_size * num_tokens1 * num_heads_q * head_size)
    k1: Tensor,  # contiguous (batch_size * num_tokens1 * num_heads_k * head_size)
    w_q0: Tensor,  # contiguous (head_size)
    w_k0: Tensor,  # contiguous (head_size)
    w_q1: Tensor,  # contiguous (head_size)
    w_k1: Tensor,  # contiguous (head_size)
    cos_sin0: Tensor,  # contiguous (num_tokens0 * head_size)
    cos_sin1: Tensor,  # contiguous (num_tokens1 * head_size)
    batch_size: int,
    num_tokens0: int,
    num_tokens1: int,
    num_heads_q: int,
    num_heads_k: int,
    head_size: int,
    is_interleaved: bool,
    eps: float,
):
    is_neox_style = not is_interleaved
    q0_shape = q0.shape
    k0_shape = k0.shape
    q1_shape = q1.shape
    k1_shape = k1.shape
    q0_by_head = rms_norm_forward(
        q0.view(batch_size, num_tokens0, num_heads_q, head_size), w_q0, eps
    )
    k0_by_head = rms_norm_forward(
        k0.view(batch_size, num_tokens0, num_heads_k, head_size), w_k0, eps
    )
    q1_by_head = rms_norm_forward(
        q1.view(batch_size, num_tokens1, num_heads_q, head_size), w_q1, eps
    )
    k1_by_head = rms_norm_forward(
        k1.view(batch_size, num_tokens1, num_heads_k, head_size), w_k1, eps
    )
    cos_sin0 = cos_sin0.view(num_tokens0, head_size)
    cos_sin1 = cos_sin1.view(num_tokens1, head_size)
    cos0, sin0 = cos_sin0.chunk(2, dim=-1)
    cos1, sin1 = cos_sin1.chunk(2, dim=-1)
    q0 = apply_rotary_emb_torch(q0_by_head, cos0, sin0, is_neox_style)
    k0 = apply_rotary_emb_torch(k0_by_head, cos0, sin0, is_neox_style)
    q1 = apply_rotary_emb_torch(q1_by_head, cos1, sin1, is_neox_style)
    k1 = apply_rotary_emb_torch(k1_by_head, cos1, sin1, is_neox_style)
    q0 = q0.reshape(q0_shape)
    k0 = k0.reshape(k0_shape)
    q1 = q1.reshape(q1_shape)
    k1 = k1.reshape(k1_shape)
    q01 = torch.cat([q0, q1], dim=1)
    k01 = torch.cat([k0, k1], dim=1)
    return q01, k01


@perftest()
def run_fused_qk_norm_rope_2way(
    q0: Tensor,  # contiguous (batch_size * num_tokens0 * num_heads_q * head_size)
    k0: Tensor,  # contiguous (batch_size * num_tokens0 * num_heads_k * head_size)
    q1: Tensor,  # contiguous (batch_size * num_tokens1 * num_heads_q * head_size)
    k1: Tensor,  # contiguous (batch_size * num_tokens1 * num_heads_k * head_size)
    w_q0: Tensor,  # contiguous (head_size)
    w_k0: Tensor,  # contiguous (head_size)
    w_q1: Tensor,  # contiguous (head_size)
    w_k1: Tensor,  # contiguous (head_size)
    cos_sin0: Tensor,  # contiguous (num_tokens0 * head_size)
    cos_sin1: Tensor,  # contiguous (num_tokens1 * head_size)
    batch_size: int,
    num_tokens0: int,
    num_tokens1: int,
    num_heads_q: int,
    num_heads_k: int,
    head_size: int,
    is_interleaved: bool,
    eps: float,
):
    q01 = torch.empty(
        (batch_size, num_tokens0 + num_tokens1, num_heads_q, head_size),
        dtype=q0.dtype,
        device=q0.device,
    )
    k01 = torch.empty(
        (batch_size, num_tokens0 + num_tokens1, num_heads_k, head_size),
        dtype=k0.dtype,
        device=k0.device,
    )
    aiter.fused_qk_norm_rope_2way(
        q0,
        k0,
        q1,
        k1,
        w_q0,
        w_k0,
        w_q1,
        w_k1,
        cos_sin0,
        cos_sin1,
        batch_size,
        num_tokens0,
        num_tokens1,
        num_heads_q,
        num_heads_k,
        head_size,
        is_interleaved,
        eps,
        q01,
        k01,
    )
    return q01, k01


@benchmark()
def test_qk_norm_rope_2way(
    dtype,
    batch_size,
    num_tokens0,
    num_tokens1,
    num_heads_q,
    num_heads_k,
    head_size,
    is_interleaved,
    eps=1e-6,
):
    q0 = torch.randn(
        (batch_size, num_tokens0, num_heads_q, head_size),
        dtype=dtype,
        device="cuda",
    )
    k0 = torch.randn(
        (batch_size, num_tokens0, num_heads_k, head_size),
        dtype=dtype,
        device="cuda",
    )
    q1 = torch.randn(
        (batch_size, num_tokens1, num_heads_q, head_size),
        dtype=dtype,
        device="cuda",
    )
    k1 = torch.randn(
        (batch_size, num_tokens1, num_heads_k, head_size),
        dtype=dtype,
        device="cuda",
    )
    w_q0 = torch.randn(head_size, dtype=dtype, device="cuda")
    w_k0 = torch.randn(head_size, dtype=dtype, device="cuda")
    w_q1 = torch.randn(head_size, dtype=dtype, device="cuda")
    w_k1 = torch.randn(head_size, dtype=dtype, device="cuda")
    cos_sin0 = torch.randn(
        (num_tokens0, head_size),
        dtype=dtype,
        device="cuda",
    )
    cos_sin1 = torch.randn(
        (num_tokens1, head_size),
        dtype=dtype,
        device="cuda",
    )
    (q01_ref, k01_ref), avg_torch = run_torch_qk_norm_rope_2way(
        q0,
        k0,
        q1,
        k1,
        w_q0,
        w_k0,
        w_q1,
        w_k1,
        cos_sin0,
        cos_sin1,
        batch_size,
        num_tokens0,
        num_tokens1,
        num_heads_q,
        num_heads_k,
        head_size,
        is_interleaved,
        eps,
    )
    (q01, k01), avg_cu = run_fused_qk_norm_rope_2way(
        q0,
        k0,
        q1,
        k1,
        w_q0,
        w_k0,
        w_q1,
        w_k1,
        cos_sin0,
        cos_sin1,
        batch_size,
        num_tokens0,
        num_tokens1,
        num_heads_q,
        num_heads_k,
        head_size,
        is_interleaved,
        eps,
    )

    info = f"dtype:{dtype}, batch_size:{batch_size}, num_tokens0:{num_tokens0}, num_tokens1:{num_tokens1}, num_heads_q:{num_heads_q}, num_heads_k:{num_heads_k}"
    info += f", head_size:{head_size}, is_interleaved:{is_interleaved}, eps:{eps}"
    msg = f"[perf] === {info} === torch avg: {avg_torch:<8.2f} us, cu avg: {avg_cu:<8.2f} us, uplift: {avg_torch/avg_cu-1:<5.1%}"
    checkAllclose(q01_ref, q01, msg="q01", rtol=1e-2, atol=0.05)
    checkAllclose(k01_ref, k01, msg="k01", rtol=1e-2, atol=0.05)
    print(msg, flush=True)

    ret = {}
    ret["dtype"] = dtype
    ret["batch_size"] = batch_size
    ret["num_tokens0"] = num_tokens0
    ret["num_tokens1"] = num_tokens1
    ret["num_heads_q"] = num_heads_q
    ret["num_heads_k"] = num_heads_k
    ret["head_size"] = head_size
    ret["is_interleaved"] = "1" if is_interleaved else "0"
    ret["avg_torch"] = avg_torch
    ret["avg_cu"] = avg_cu
    ret["speedup"] = avg_torch / avg_cu
    return ret


@benchmark()
def test_qk_norm_rope_cache_block_quant(
    dtype,
    num_tokens,
    num_heads_q,
    num_heads_k,
    num_heads_v,
    head_size,
    num_blocks,
    page_size,
    is_neox_style,
    eps,
    kv_cache_dtype,
    batch=1,
    decode_tokens_per_batch=1,
):
    torch.manual_seed(0)
    # Initialize cache to zeros so unused slots have 0 -> ref pertoken_quant
    # matches kernel (kernel only writes pages with tokens; randn would pollute scale)
    if kv_cache_dtype == "fp8_e4m3":
        cache_dtype = get_dtype_fp8()
    else:
        cache_dtype = dtype
    # Zeros init: unused slots stay 0 so ref's pertoken_quant max = kernel's block_max
    k_cache = torch.zeros(
        [num_blocks, page_size, num_heads_k, head_size],
        dtype=dtype,
        device="cuda",
    ).to(cache_dtype)
    v_cache = torch.zeros(
        [num_blocks, page_size, num_heads_v, head_size],
        dtype=dtype,
        device="cuda",
    ).to(cache_dtype)

    # Check for NaN values in k_cache and v_cache
    if torch.isnan(k_cache).any():
        aiter.logger.warning(f"k_cache contains NaN values! dtype={cache_dtype}")
    if torch.isnan(v_cache).any():
        aiter.logger.warning(f"v_cache contains NaN values! dtype={cache_dtype}")

    # slot_mapping built after cu_q_len (see below)
    x = 16 // k_cache.element_size()
    k_cache = (
        k_cache.view(
            [
                num_blocks,
                page_size,
                num_heads_k,
                head_size // x,
                x,
            ]
        )
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    # Value cache [num_blocks, num_kv_heads, block_size // x, head_size, x]
    v_cache = (
        v_cache.view(
            [
                num_blocks,
                page_size // x,
                num_heads_v,
                head_size,
                x,
            ]
        )
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )
    batch_size = batch
    base_len = num_tokens // batch_size
    remainder = num_tokens % batch_size
    seq_lens = [base_len + 1] * remainder + [base_len] * (batch_size - remainder)
    total_len = sum(seq_lens)
    if total_len > num_tokens:
        seq_lens[-1] -= total_len - num_tokens
    elif total_len < num_tokens:
        seq_lens[-1] += num_tokens - total_len
    #
    cu_q_len = torch.zeros(batch_size + 1, dtype=torch.int64, device="cuda")

    cu_q_len[0] = 0
    for i in range(batch_size):
        cu_q_len[i + 1] = cu_q_len[i] + seq_lens[i]
    #
    assert (
        cu_q_len[-1].item() == num_tokens
    ), f"cu_q_len[-1]={cu_q_len[-1].item()} != num_tokens={num_tokens}"
    #
    # slot_mapping: each batch maps to disjoint blocks (no cross-batch block sharing)
    slot_start_per_batch = []
    next_slot = 0
    for i in range(batch_size):
        slot_start_per_batch.append(next_slot)
        blocks_needed = (seq_lens[i] + page_size - 1) // page_size
        next_slot += blocks_needed * page_size
    prefill_slot_end = next_slot  # chunk/decode must allocate after this
    slot_mapping = torch.zeros(num_tokens, dtype=torch.int64, device="cuda")
    for i in range(batch_size):
        start = cu_q_len[i].item()
        end = cu_q_len[i + 1].item()
        slot_mapping[start:end] = torch.arange(
            slot_start_per_batch[i],
            slot_start_per_batch[i] + (end - start),
            dtype=torch.int64,
            device="cuda",
        )
    #
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
    k_scale = torch.zeros(
        [num_blocks, num_heads_k],
        dtype=torch.float32,
        device="cuda",
    )
    v_scale = torch.zeros(
        [num_blocks, num_heads_v],
        dtype=torch.float32,
        device="cuda",
    )
    k_scale_ref = k_scale.clone()
    v_scale_ref = v_scale.clone()
    ## Use separate caches so ref and aiter each write to their own (avoid self-compare)
    k_cache_ref = k_cache.clone()
    v_cache_ref = v_cache.clone()

    (q_ref, k_ref, v_ref, k_cache_ref, v_cache_ref), avg_torch = (
        run_torch_qk_norm_rope_cache_block_quant_shuffle(
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
            k_cache_ref,
            v_cache_ref,
            k_scale_ref,
            v_scale_ref,
            slot_mapping,
            kv_cache_dtype,
        )
    )
    (q, k, v, k_cache, v_cache), avg_cu = (
        run_aiter_qk_norm_rope_cache_block_quant_shuffle(
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
            cu_q_len,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
    )

    info = f"dtype:{dtype}, batch:{batch}, num_tokens:{num_tokens}, num_heads_q:{num_heads_q}, num_heads_k:{num_heads_k}, num_heads_v:{num_heads_v}, head_size:{head_size}, is_neox_style:{is_neox_style}"
    msg = f"[perf] === {info} === torch avg: {avg_torch:<8.2f} us, cu avg: {avg_cu:<8.2f} us, uplift: {avg_torch / avg_cu - 1:<5.1%}"
    checkAllclose(q_ref[:64,], q[:64,], msg="prefill q", rtol=1e-2, atol=0.05)
    checkAllclose(k_ref, k, msg="prefill k", rtol=1e-2, atol=0.05)
    checkAllclose(v_ref, v, msg=msg, rtol=1e-2, atol=0.05)
    # Only check pages that have actual token data (via slot_mapping)
    page_size = k_cache.shape[
        -2
    ]  # k_cache: [num_blocks, num_kv_heads, head_size//x, page_size, x]
    slots_edit = torch.unique(slot_mapping // page_size)
    checkAllclose(
        k_cache_ref.float()[slots_edit],
        k_cache.float()[slots_edit],
        msg="prefill k_cache",
        rtol=5e-2,
        atol=0.05,
    )
    checkAllclose(
        v_cache_ref.float()[slots_edit],
        v_cache.float()[slots_edit],
        msg="prefill v_cache",
        rtol=5e-2,
        atol=0.05,
    )

    checkAllclose(
        k_scale_ref[slots_edit],
        k_scale[slots_edit],
        msg="prefill k_scale",
        rtol=1e-2,
        atol=0.01,
    )
    checkAllclose(
        v_scale_ref[slots_edit],
        v_scale[slots_edit],
        msg="prefill v_scale",
        rtol=1e-2,
        atol=0.01,
    )

    ret = {}
    ret["fused_qk_us"] = avg_cu
    # ret["k_cache_err"] = k_cache_err
    # ret["v_cache_err"] = v_cache_err
    # ========== chunk-prefill part ==========
    # Chunk: (page_size-1) tokens per batch, K=0.001 + kw=1 -> k_scale_global small.
    # Decode fills last slot per block -> k_scale_val > k_scale_global triggers requantization.
    # batch_size = cu_q_len.size(0) - 1
    page_size = k_cache.shape[
        -2
    ]  # k_cache: [num_blocks, num_kv_heads, head_size//x, page_size, x]
    chunk_left_ctx_lens = page_size - 1
    chunk_total_tokens = batch_size * chunk_left_ctx_lens
    #
    chunk_qkv = torch.randn(
        (chunk_total_tokens, (num_heads_q + num_heads_k + num_heads_v) * head_size),
        dtype=dtype,
        device="cuda",
    )
    q_size, k_size, v_size = (
        num_heads_q * head_size,
        num_heads_k * head_size,
        num_heads_v * head_size,
    )
    ## to test requant in worst case
    # chunk_qkv[:, q_size : q_size + k_size] = 1e-6  # K ~0 -> k_scale_global ~2e-6
    # chunk_qkv[:, q_size + k_size : q_size + k_size + v_size] = (
    #    1  # V ~0 -> v_scale_global ~2e-6
    # )
    kw = torch.ones(
        head_size, dtype=dtype, device="cuda"
    )  # kw=1 so k_scale_global small
    # slot_mapping: each batch fills slots [0..page_size-2] of its chunk block
    chunk_slot_mapping = (
        (
            prefill_slot_end
            + torch.arange(batch_size, device="cuda").unsqueeze(1) * page_size
            + torch.arange(chunk_left_ctx_lens, device="cuda").unsqueeze(0)
        )
        .reshape(-1)
        .to(torch.int64)
    )
    chunk_cu_q_len = torch.zeros(batch_size + 1, dtype=torch.int64, device="cuda")
    for i in range(batch_size):
        chunk_cu_q_len[i + 1] = chunk_cu_q_len[i] + chunk_left_ctx_lens
    chunk_positions = torch.randint(
        0, max_positions, (chunk_total_tokens,), dtype=torch.int64, device="cuda"
    )

    k_scale_chunk_ref = k_scale_ref.clone()
    v_scale_chunk_ref = v_scale_ref.clone()
    k_scale_chunk = k_scale.clone()
    v_scale_chunk = v_scale.clone()
    #
    (
        q_chunk_ref,
        k_chunk_ref,
        v_chunk_ref,
        k_cache_ref,
        v_cache_ref,
    ), avg_torch_chunk = run_torch_qk_norm_rope_cache_block_quant_shuffle(
        chunk_qkv,
        qw,
        kw,
        cos_sin,
        chunk_positions,
        chunk_total_tokens,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_size,
        is_neox_style,
        eps,
        k_cache_ref,
        v_cache_ref,
        k_scale_chunk_ref,
        v_scale_chunk_ref,
        chunk_slot_mapping,
        kv_cache_dtype,
    )
    (q_chunk, k_chunk, v_chunk, k_cache, v_cache), avg_cu_chunk = (
        run_aiter_qk_norm_rope_cache_block_quant_shuffle(
            chunk_qkv,
            qw,
            kw,
            cos_sin,
            chunk_positions,
            chunk_total_tokens,
            num_heads_q,
            num_heads_k,
            num_heads_v,
            head_size,
            is_neox_style,
            eps,
            k_cache,
            v_cache,
            chunk_slot_mapping,
            chunk_cu_q_len,
            kv_cache_dtype,
            k_scale_chunk,
            v_scale_chunk,
        )
    )
    #
    print(
        f"chunk-prefill: torch avg: {avg_torch_chunk:.2f} us, cu avg: {avg_cu_chunk:.2f} us"
    )
    checkAllclose(q_chunk_ref, q_chunk, msg="chunk q", rtol=1e-2, atol=0.05)
    checkAllclose(k_chunk_ref, k_chunk, msg="chunk k", rtol=1e-2, atol=0.05)
    checkAllclose(v_chunk_ref, v_chunk, msg="chunk v", rtol=1e-2, atol=0.05)
    # Combine prefill + chunk slots to check all pages with data
    all_slots_so_far = torch.cat([slot_mapping, chunk_slot_mapping])
    chunk_slots_edit = torch.unique(all_slots_so_far // page_size)
    chunk_k_cache_err = checkAllclose(
        k_cache_ref.float()[chunk_slots_edit],
        k_cache.float()[chunk_slots_edit],
        msg="chunk k_cache",
        rtol=5e-2,
        atol=0.05,
    )
    chunk_v_cache_err = checkAllclose(
        v_cache_ref.float()[chunk_slots_edit],
        v_cache.float()[chunk_slots_edit],
        msg="chunk v_cache",
        rtol=5e-2,
        atol=0.05,
    )
    checkAllclose(
        k_scale_chunk_ref[chunk_slots_edit],
        k_scale_chunk[chunk_slots_edit],
        msg="chunk k_scale",
        rtol=1e-2,
        atol=0.01,
    )
    checkAllclose(
        v_scale_chunk_ref[chunk_slots_edit],
        v_scale_chunk[chunk_slots_edit],
        msg="chunk v_scale",
        rtol=1e-2,
        atol=0.01,
    )
    # ret["chunk_fused_qk_us"] = avg_cu_chunk
    # ret["chunk_k_cache_err"] = chunk_k_cache_err
    # ret["chunk_v_cache_err"] = chunk_v_cache_err
    #
    # ========== decode test 1: 1 token per batch (append after chunk's last slot) ==========
    dtpb = decode_tokens_per_batch
    decode1_total = batch_size
    decode1_slot_mapping = torch.tensor(
        [
            int(chunk_slot_mapping[(bsID + 1) * chunk_left_ctx_lens - 1].item()) + 1
            for bsID in range(batch_size)
        ],
        dtype=torch.int64,
        device="cuda",
    )
    decode1_qkv = torch.randn(
        (decode1_total, (num_heads_q + num_heads_k + num_heads_v) * head_size),
        dtype=dtype,
        device="cuda",
    )
    decode1_qkv[:, q_size : q_size + k_size] = torch.randn(
        (decode1_total, k_size), dtype=dtype, device="cuda"
    )
    decode1_qkv[:, q_size + k_size : q_size + k_size + v_size] = (
        torch.randn((decode1_total, v_size), dtype=dtype, device="cuda") * 2.0 + 2.0
    )
    decode1_cu_q_len = torch.zeros(batch_size + 1, dtype=torch.int64, device="cuda")
    for i in range(batch_size):
        decode1_cu_q_len[i + 1] = decode1_cu_q_len[i] + 1
    decode1_positions = torch.randint(
        0, max_positions, (decode1_total,), dtype=torch.int64, device="cuda"
    )
    k_scale_d1_ref = k_scale_chunk_ref.clone()
    v_scale_d1_ref = v_scale_chunk_ref.clone()
    k_scale_d1 = k_scale_chunk.clone()
    v_scale_d1 = v_scale_chunk.clone()

    print(
        f"decode1: 1 tok/batch, total={decode1_total}"
    )  # , slots={decode1_slot_mapping}
    (q_d1_ref, k_d1_ref, v_d1_ref, k_cache_ref, v_cache_ref), avg_torch_d1 = (
        run_torch_qk_norm_rope_cache_block_quant_shuffle(
            decode1_qkv,
            qw,
            kw,
            cos_sin,
            decode1_positions,
            decode1_total,
            num_heads_q,
            num_heads_k,
            num_heads_v,
            head_size,
            is_neox_style,
            eps,
            k_cache_ref,
            v_cache_ref,
            k_scale_d1_ref,
            v_scale_d1_ref,
            decode1_slot_mapping,
            kv_cache_dtype,
        )
    )
    (q_d1, k_d1, v_d1, k_cache, v_cache), avg_cu_d1 = (
        run_aiter_qk_norm_rope_cache_block_quant_shuffle(
            decode1_qkv,
            qw,
            kw,
            cos_sin,
            decode1_positions,
            decode1_total,
            num_heads_q,
            num_heads_k,
            num_heads_v,
            head_size,
            is_neox_style,
            eps,
            k_cache,
            v_cache,
            decode1_slot_mapping,
            decode1_cu_q_len,
            kv_cache_dtype,
            k_scale_d1,
            v_scale_d1,
        )
    )
    print(f"decode1: torch avg: {avg_torch_d1:.2f} us, cu avg: {avg_cu_d1:.2f} us")
    checkAllclose(q_d1_ref, q_d1, msg="decode1 q", rtol=1e-2, atol=0.05)
    checkAllclose(k_d1_ref, k_d1, msg="decode1 k", rtol=1e-2, atol=0.05)
    checkAllclose(v_d1_ref, v_d1, msg="decode1 v", rtol=1e-2, atol=0.05)
    all_slots_d1 = torch.cat([slot_mapping, chunk_slot_mapping, decode1_slot_mapping])
    d1_pages = torch.unique(all_slots_d1 // page_size)
    d1_k_err = checkAllclose(
        k_cache_ref.float()[d1_pages],
        k_cache.float()[d1_pages],
        msg="decode1 k_cache",
        rtol=5e-2,
        atol=0.05,
    )
    d1_v_err = checkAllclose(
        v_cache_ref.float()[d1_pages],
        v_cache.float()[d1_pages],
        msg="decode1 v_cache",
        rtol=5e-2,
        atol=0.05,
    )
    checkAllclose(
        k_scale_d1_ref[d1_pages],
        k_scale_d1[d1_pages],
        msg="decode1 k_scale",
        rtol=1e-2,
        atol=0.05,
    )
    checkAllclose(
        v_scale_d1_ref[d1_pages],
        v_scale_d1[d1_pages],
        msg="decode1 v_scale",
        rtol=1e-2,
        atol=0.05,
    )
    ret["decode1_fused_qk_us"] = avg_cu_d1
    ret["decode1_k_cache_err"] = d1_k_err
    ret["decode1_v_cache_err"] = d1_v_err

    # ========== decode test 2: dtpb tokens per batch (cross-page, only when dtpb > 1) ==========
    if dtpb > 1:
        decode2_total = batch_size * dtpb
        last_used_slot = int(chunk_slot_mapping[-1].item())
        decode2_page_base = (last_used_slot + page_size) // page_size * page_size
        num_blocks * page_size
        pages_needed = batch_size * 2 + (decode2_page_base // page_size)
        assert (
            pages_needed <= num_blocks
        ), f"decode2 needs {pages_needed} pages but num_blocks={num_blocks}. Increase -b."
        decode2_slots = []
        for bsID in range(batch_size):
            start_slot = (
                decode2_page_base + bsID * 2 * page_size + (page_size - 1)
            )  # (dtpb // 2)
            for t in range(dtpb):
                decode2_slots.append(start_slot + t)
        decode2_slot_mapping = torch.tensor(
            decode2_slots, dtype=torch.int64, device="cuda"
        )
        decode2_qkv = torch.randn(
            (decode2_total, (num_heads_q + num_heads_k + num_heads_v) * head_size),
            dtype=dtype,
            device="cuda",
        )
        decode2_qkv[:, q_size : q_size + k_size] = torch.randn(
            (decode2_total, k_size), dtype=dtype, device="cuda"
        )
        decode2_qkv[:, q_size + k_size : q_size + k_size + v_size] = (
            torch.randn((decode2_total, v_size), dtype=dtype, device="cuda") * 2.0 + 2.0
        )
        decode2_cu_q_len = torch.zeros(batch_size + 1, dtype=torch.int64, device="cuda")
        for i in range(batch_size):
            decode2_cu_q_len[i + 1] = decode2_cu_q_len[i] + dtpb
        decode2_positions = torch.randint(
            0, max_positions, (decode2_total,), dtype=torch.int64, device="cuda"
        )
        k_scale_d2_ref = k_scale_d1_ref.clone()
        v_scale_d2_ref = v_scale_d1_ref.clone()
        k_scale_d2 = k_scale_d1.clone()
        v_scale_d2 = v_scale_d1.clone()

        print(
            f"decode2: {dtpb} tok/batch (cross-page), total={decode2_total}"
        )  # , slots={decode2_slot_mapping}
        (q_d2_ref, k_d2_ref, v_d2_ref, k_cache_ref, v_cache_ref), avg_torch_d2 = (
            run_torch_qk_norm_rope_cache_block_quant_shuffle(
                decode2_qkv,
                qw,
                kw,
                cos_sin,
                decode2_positions,
                decode2_total,
                num_heads_q,
                num_heads_k,
                num_heads_v,
                head_size,
                is_neox_style,
                eps,
                k_cache_ref,
                v_cache_ref,
                k_scale_d2_ref,
                v_scale_d2_ref,
                decode2_slot_mapping,
                kv_cache_dtype,
            )
        )
        (q_d2, k_d2, v_d2, k_cache, v_cache), avg_cu_d2 = (
            run_aiter_qk_norm_rope_cache_block_quant_shuffle(
                decode2_qkv,
                qw,
                kw,
                cos_sin,
                decode2_positions,
                decode2_total,
                num_heads_q,
                num_heads_k,
                num_heads_v,
                head_size,
                is_neox_style,
                eps,
                k_cache,
                v_cache,
                decode2_slot_mapping,
                decode2_cu_q_len,
                kv_cache_dtype,
                k_scale_d2,
                v_scale_d2,
            )
        )
        print(f"decode2: torch avg: {avg_torch_d2:.2f} us, cu avg: {avg_cu_d2:.2f} us")
        checkAllclose(q_d2_ref, q_d2, msg="decode2 q", rtol=1e-2, atol=0.05)
        checkAllclose(k_d2_ref, k_d2, msg="decode2 k", rtol=1e-2, atol=0.05)
        checkAllclose(v_d2_ref, v_d2, msg="decode2 v", rtol=1e-2, atol=0.05)
        all_slots_d2 = torch.cat([all_slots_d1, decode2_slot_mapping])
        d2_pages = torch.unique(all_slots_d2 // page_size)
        checkAllclose(
            k_cache_ref.float()[d2_pages],
            k_cache.float()[d2_pages],
            msg="decode2 k_cache",
            rtol=5e-2,
            atol=0.05,
        )
        checkAllclose(
            v_cache_ref.float()[d2_pages],
            v_cache.float()[d2_pages],
            msg="decode2 v_cache",
            rtol=5e-2,
            atol=0.05,
        )
        checkAllclose(
            k_scale_d2_ref[d2_pages],
            k_scale_d2[d2_pages],
            msg="decode2 k_scale",
            rtol=1e-2,
            atol=0.05,
        )
        checkAllclose(
            v_scale_d2_ref[d2_pages],
            v_scale_d2[d2_pages],
            msg="decode2 v_scale",
            rtol=1e-2,
            atol=0.05,
        )
        ret["decode2_fused_qk_us"] = avg_cu_d2
        # ret["decode2_k_cache_err"] = d2_k_err
        # ret["decode2_v_cache_err"] = d2_v_err

    # ========== decode test 3: dtpb tokens per batch from block_offset=0 (fresh page) ==========
    if dtpb > 1:
        decode3_total = batch_size * dtpb
        last_used_slot = int(chunk_slot_mapping[-1].item())
        decode3_page_base = decode2_page_base + batch_size * 2 * page_size
        pages_needed_d3 = batch_size + (decode3_page_base // page_size)
        assert (
            pages_needed_d3 <= num_blocks
        ), f"decode3 needs {pages_needed_d3} pages but num_blocks={num_blocks}. Increase -b."
        decode3_slots = []
        for bsID in range(batch_size):
            base_slot = decode3_page_base + bsID * page_size
            for t in range(dtpb):
                decode3_slots.append(base_slot + t)
        decode3_slot_mapping = torch.tensor(
            decode3_slots, dtype=torch.int64, device="cuda"
        )
        decode3_qkv = torch.randn(
            (decode3_total, (num_heads_q + num_heads_k + num_heads_v) * head_size),
            dtype=dtype,
            device="cuda",
        )
        decode3_qkv[:, q_size : q_size + k_size] = torch.randn(
            (decode3_total, k_size), dtype=dtype, device="cuda"
        )
        decode3_qkv[:, q_size + k_size : q_size + k_size + v_size] = (
            torch.randn((decode3_total, v_size), dtype=dtype, device="cuda") * 2.0 + 2.0
        )
        decode3_cu_q_len = torch.zeros(batch_size + 1, dtype=torch.int64, device="cuda")
        for i in range(batch_size):
            decode3_cu_q_len[i + 1] = decode3_cu_q_len[i] + dtpb
        decode3_positions = torch.randint(
            0, max_positions, (decode3_total,), dtype=torch.int64, device="cuda"
        )
        k_scale_d3_ref = k_scale_d1_ref.clone()
        v_scale_d3_ref = v_scale_d1_ref.clone()
        k_scale_d3 = k_scale_d1.clone()
        v_scale_d3 = v_scale_d1.clone()

        print(
            f"decode3: {dtpb} tok/batch (offset=0, fresh page), total={decode3_total}, slots={decode3_slot_mapping[:16]}..."
        )
        (q_d3_ref, k_d3_ref, v_d3_ref, k_cache_ref, v_cache_ref), avg_torch_d3 = (
            run_torch_qk_norm_rope_cache_block_quant_shuffle(
                decode3_qkv,
                qw,
                kw,
                cos_sin,
                decode3_positions,
                decode3_total,
                num_heads_q,
                num_heads_k,
                num_heads_v,
                head_size,
                is_neox_style,
                eps,
                k_cache_ref,
                v_cache_ref,
                k_scale_d3_ref,
                v_scale_d3_ref,
                decode3_slot_mapping,
                kv_cache_dtype,
            )
        )
        (q_d3, k_d3, v_d3, k_cache, v_cache), avg_cu_d3 = (
            run_aiter_qk_norm_rope_cache_block_quant_shuffle(
                decode3_qkv,
                qw,
                kw,
                cos_sin,
                decode3_positions,
                decode3_total,
                num_heads_q,
                num_heads_k,
                num_heads_v,
                head_size,
                is_neox_style,
                eps,
                k_cache,
                v_cache,
                decode3_slot_mapping,
                decode3_cu_q_len,
                kv_cache_dtype,
                k_scale_d3,
                v_scale_d3,
            )
        )
        print(f"decode3: torch avg: {avg_torch_d3:.2f} us, cu avg: {avg_cu_d3:.2f} us")
        checkAllclose(q_d3_ref, q_d3, msg="decode3 q", rtol=1e-2, atol=0.05)
        checkAllclose(k_d3_ref, k_d3, msg="decode3 k", rtol=1e-2, atol=0.05)
        checkAllclose(v_d3_ref, v_d3, msg="decode3 v", rtol=1e-2, atol=0.05)
        all_slots_d3 = torch.cat([all_slots_d1, decode3_slot_mapping])
        d3_pages = torch.unique(all_slots_d3 // page_size)
        d3_k_err = checkAllclose(
            k_cache_ref.float()[d3_pages],
            k_cache.float()[d3_pages],
            msg="decode3 k_cache",
            rtol=5e-2,
            atol=0.05,
        )
        d3_v_err = checkAllclose(
            v_cache_ref.float()[d3_pages],
            v_cache.float()[d3_pages],
            msg="decode3 v_cache",
            rtol=5e-2,
            atol=0.05,
        )
        checkAllclose(
            k_scale_d3_ref[d3_pages],
            k_scale_d3[d3_pages],
            msg="decode3 k_scale",
            rtol=1e-2,
            atol=0.05,
        )
        checkAllclose(
            v_scale_d3_ref[d3_pages],
            v_scale_d3[d3_pages],
            msg="decode3 v_scale",
            rtol=1e-2,
            atol=0.05,
        )
        ret["decode3_fused_qk_us"] = avg_cu_d3
        ret["decode3_k_cache_err"] = d3_k_err
        ret["decode3_v_cache_err"] = d3_v_err

    return ret


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
    default=[3, 127, 513, 778, 1024, 1257],
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
    "--batch",
    type=int,
    default=1,
    help="""Batch size. num_tokens is split across batch batches for block quant only.
    e.g.: --batch 4""",
)
parser.add_argument(
    "--decode_tokens_per_batch",
    type=int,
    default=1,
    help="""Tokens per batch in decode stage (1 or 4) for block quant only.
    e.g.: --decode_tokens_per_batch 4""",
)
parser.add_argument(
    "-p",
    "--page_size",
    type=int,
    default=16,
    help="""Page size (for per_head quant).
    e.g.: -p 16""",
)
parser.add_argument(
    "--block_page_size",
    type=int,
    default=64,
    help="""Page size for block quant (default 64, more stable than 16) for block quant only.
    e.g.: --block_page_size 64""",
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

parser.add_argument(
    "-q",
    "--quant_type",
    type=str,
    nargs="*",
    choices=["block", "per_head"],
    default=["block", "per_head"],
    help="""Quantization type.
    block: prefill + chunk (page_size-1 per batch, K=0.001) + decode (last slot)""",
)

if __name__ == "__main__":
    args = parser.parse_args()
    max_positions = args.max_positions
    df = []
    # rope
    block_df = []

    for is_neox_style in args.is_neox_styles:
        for num_token in args.token:
            for num_head, num_kv_head in args.head:
                for i, head_size in enumerate(args.head_sizes):
                    for kv_cache_dtype in args.kv_cache_dtypes:
                        if kv_cache_dtype == "fp8_e4m3":
                            cache_dtype = get_dtype_fp8()
                        else:
                            cache_dtype = args.dtype
                        for quant_type in args.quant_type:
                            if quant_type == "block":
                                ret = test_qk_norm_rope_cache_block_quant(
                                    args.dtype,
                                    num_token,
                                    num_head,
                                    num_kv_head,
                                    num_kv_head,
                                    head_size,
                                    args.num_blocks,
                                    args.block_page_size,
                                    is_neox_style,
                                    1e-6,
                                    kv_cache_dtype,
                                    batch=args.batch,
                                    decode_tokens_per_batch=args.decode_tokens_per_batch,
                                )
                                block_df.append(ret)
                            else:
                                ret = test_qk_norm_rope_cache_quant(
                                    args.dtype,
                                    num_token,
                                    num_head,
                                    num_kv_head,
                                    num_kv_head,
                                    head_size,
                                    is_neox_style,
                                    1e-6,
                                    kv_cache_dtype,
                                    args.num_blocks,
                                    args.page_size,
                                )
                                df.append(ret)
    df = pd.DataFrame(df)
    block_df = pd.DataFrame(block_df)
    df_md = df.to_markdown(index=False)
    block_df_md = block_df.to_markdown(index=False)
    aiter.logger.info("qk_norm_rope_cache_quant summary (markdown):\n%s", df_md)
    aiter.logger.info(
        "qk_norm_rope_cache_block_quant summary (markdown):\n%s", block_df_md
    )

    dtype = torch.bfloat16
    batch_size = 2
    num_tokens1 = 3608
    num_heads_q = 24
    num_heads_k = 25
    df = []
    for head_size in args.head_sizes:
        for num_tokens0 in args.token:
            for is_neox_styles in args.is_neox_styles:
                ret = test_qk_norm_rope_2way(
                    dtype,
                    batch_size,
                    num_tokens0,
                    num_tokens1,
                    num_heads_q,
                    num_heads_k,
                    head_size,
                    not is_neox_styles,
                    eps=1e-6,
                )
                df.append(ret)
    df = pd.DataFrame(df)
    df_md = df.to_markdown(index=False)
    aiter.logger.info("qk_norm_rope_2way summary (markdown):\n%s", df_md)
