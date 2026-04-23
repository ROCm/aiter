# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import random
import pytest
import torch

from aiter.ops.triton.attention.mla import mla_decode_fwd
from aiter.ops.triton.attention.mla import mla_prefill_fwd
from aiter.ops.shuffle import shuffle_weight
from op_tests.triton_tests.quant.test_quant_mxfp4 import (
    torch_dynamic_mxfp4_quant,
    batched_swizzle_scales_gfx1250,
)
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils.types import e4m3_dtype
from typing import Optional

DEVICE_ARCH = arch_info.get_arch()

torch.set_default_device("cuda")


def shuffle_kv_buffer(
    kv_buffer: torch.Tensor,
    kv_lora_rank: int,
):
    """
    Shuffle key and value cache layout for optimized memory access.

        layout: (num_lanes, num_elements_per_thread)
            gfx1250: (16, 8) for BF16 and FP8.
            gfx950: (16, 8) for BF16 and (16, 16) for FP8.

        WMMA/MFMA instruction shape:
            BF16: 16x16x32
            FP8: 16x16x64
    """

    dtype = kv_buffer.dtype
    assert dtype in (torch.bfloat16, e4m3_dtype)

    if dtype == torch.bfloat16:
        layout = (16, 8)
    else:
        # Caution: in gfx1250, the 16-bit and 8-bit layout should both be (16, 8), however, in order to enable ds_load_b128 for 8-bit WMMA,
        # we use (16, 16) here, noted that you must set k_width to 16 in the corresponding DotOperandLayout, the math will be equivalent.
        layout = (16, 16)

    num_blocks, block_size, num_kv_heads, head_size = kv_buffer.shape

    assert block_size >= 16

    num_lanes, num_elements_per_thread = layout

    def shuffle(kv_lora_or_rope_buffer):
        d = kv_lora_or_rope_buffer.shape[-1]
        kv_lora_or_rope_buffer = kv_lora_or_rope_buffer.view(
            -1,
            num_kv_heads,
            block_size // num_lanes,
            num_lanes,
            d // (2 * num_elements_per_thread),
            2,  # there are 2 groups of threads, t0 ~ t15 and t16 ~ t31
            num_elements_per_thread,
        )
        kv_lora_or_rope_buffer = kv_lora_or_rope_buffer.permute(
            0, 1, 2, 4, 5, 3, 6
        ).contiguous()
        kv_lora_or_rope_buffer = kv_lora_or_rope_buffer.view(
            -1, num_kv_heads, block_size // 16, d * 16
        )
        return kv_lora_or_rope_buffer

    kv_buffer_shuffled = kv_buffer.view(
        -1, block_size, num_kv_heads, head_size
    ).permute(0, 2, 1, 3)
    kv_buffer_shuffled_lora = shuffle(kv_buffer_shuffled[..., :kv_lora_rank])
    kv_buffer_shuffled_rope = shuffle(kv_buffer_shuffled[..., kv_lora_rank:])
    kv_buffer_shuffled_lora = kv_buffer_shuffled_lora.view(
        -1, num_kv_heads, block_size * kv_lora_rank
    )
    kv_buffer_shuffled_rope = kv_buffer_shuffled_rope.view(
        -1, num_kv_heads, block_size * (head_size - kv_lora_rank)
    )
    kv_buffer_shuffled = torch.cat(
        [kv_buffer_shuffled_lora, kv_buffer_shuffled_rope], dim=-1
    ).contiguous()
    kv_buffer_shuffled = kv_buffer_shuffled.view(
        -1, num_kv_heads, block_size, head_size
    )

    return kv_buffer_shuffled


def dynamic_nvfp4_quant_kv_buffer(
    kv_buffer: torch.Tensor,
    kv_lora_rank: int,
):
    dtype = kv_buffer.dtype
    assert dtype == torch.bfloat16

    num_blocks, block_size, num_kv_heads, head_size = kv_buffer.shape

    assert block_size >= 128

    def quant_and_shuffle(kv_lora_or_rope_buffer):
        d = kv_lora_or_rope_buffer.shape[-1]
        quant_head_size = d // 2
        scale_width = d // 16
        cache_shuffled, cache_shuffled_scale = torch_dynamic_mxfp4_quant(
            kv_lora_or_rope_buffer, is_nvfp4=True
        )
        cache_shuffled_scale = cache_shuffled_scale.view(
            -1, num_kv_heads, block_size, scale_width
        )
        cache_shuffled = shuffle_weight(cache_shuffled).view(
            -1, num_kv_heads, block_size * quant_head_size
        )
        cache_shuffled_scale = batched_swizzle_scales_gfx1250(
            cache_shuffled_scale
        ).view(-1, num_kv_heads, block_size * scale_width)
        cache_shuffled = torch.cat(
            [
                cache_shuffled.view(torch.uint8),
                cache_shuffled_scale.view(torch.uint8),
            ],
            dim=-1,
        ).contiguous()
        cache_shuffled = cache_shuffled.view(
            -1, num_kv_heads, block_size, quant_head_size + scale_width
        )
        return cache_shuffled, quant_head_size, scale_width

    kv_buffer_shuffled = kv_buffer.view(
        -1, block_size, num_kv_heads, head_size
    ).permute(0, 2, 1, 3)
    kv_buffer_quant_and_shuffled_lora, quant_head_size_lora, scale_width_lora = (
        quant_and_shuffle(kv_buffer_shuffled[..., :kv_lora_rank])
    )
    kv_buffer_quant_and_shuffled_rope, quant_head_size_rope, scale_width_rope = (
        quant_and_shuffle(kv_buffer_shuffled[..., kv_lora_rank:])
    )
    kv_buffer_quant_and_shuffled_lora = kv_buffer_quant_and_shuffled_lora.view(
        -1, num_kv_heads, block_size * (quant_head_size_lora + scale_width_lora)
    )
    kv_buffer_quant_and_shuffled_rope = kv_buffer_quant_and_shuffled_rope.view(
        -1, num_kv_heads, block_size * (quant_head_size_rope + scale_width_rope)
    )
    kv_buffer_quant_and_shuffled = torch.cat(
        [kv_buffer_quant_and_shuffled_lora, kv_buffer_quant_and_shuffled_rope], dim=-1
    ).contiguous()
    kv_buffer_quant_and_shuffled = kv_buffer_quant_and_shuffled.view(
        -1,
        num_kv_heads,
        block_size,
        quant_head_size_lora
        + quant_head_size_rope
        + scale_width_lora
        + scale_width_rope,
    )

    return kv_buffer_quant_and_shuffled


def ref_masked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    q_descale: Optional[torch.Tensor] = None,
    kv_descale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    query_len = q.shape[0]
    kv_len = k.shape[0]
    if q.shape[1] != k.shape[1]:
        k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
        v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
    if q.dtype != torch.bfloat16:
        q = q.to(torch.bfloat16)
    k = k.to(q.dtype)
    attn = torch.einsum("qhd,khd->hqk", q, k).float()  # GEMM at q.dtype precision
    attn *= scale
    if q_descale is not None:
        attn *= q_descale
    if kv_descale is not None:
        attn *= kv_descale
    empty_mask = torch.ones(query_len, kv_len, device=q.device)
    mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
    attn.masked_fill_(mask, float("-inf"))
    attn = torch.softmax(attn, dim=-1)
    attn = attn.to(q.dtype)
    v = v.to(q.dtype)
    out = torch.einsum("hqk,khd->qhd", attn, v)  # GEMM at q.dtype precision
    if kv_descale is not None:
        out *= kv_descale

    return out


def torch_mla_extend(
    query,  # [total_q, num_query_heads, qk_lora_rank + qk_rope_head_dim]
    kv_buffer,  # [num_block, block_size, num_kv_heads, qk_lora_rank + qk_rope_head_dim]
    cu_seqlens_q,
    seq_lens_kv,
    block_tables,
    qk_lora_rank,
    scale: float,
    q_descale: Optional[torch.Tensor] = None,
    kv_descale: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
    o_dtype: Optional[torch.dtype] = torch.bfloat16,
):
    _, block_size, num_kv_heads, qk_head_dim = kv_buffer.shape
    num_seqs = cu_seqlens_q.shape[0] - 1

    outputs: list[torch.Tensor] = []
    for i in range(num_seqs):
        q = query[cu_seqlens_q[i] : cu_seqlens_q[i + 1]]

        kv_len = seq_lens_kv[i]
        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = kv_buffer[block_indices].view(-1, num_kv_heads, qk_head_dim)
        k = k[:kv_len]
        v = k[..., :qk_lora_rank]

        out = ref_masked_attention(q, k, v, scale, q_descale, kv_descale)

        outputs.append(out)

    out = torch.cat(outputs, dim=0)
    if out_scale is not None:
        out = out / out_scale
    return out.to(o_dtype)


# @pytest.mark.parametrize("batch_size", [1, 4, 8, 32])
# @pytest.mark.parametrize("decode_qlen", [1, 3])
# @pytest.mark.parametrize("ctx_lens", [200, 4371, 8192])
# @pytest.mark.parametrize("num_heads", [(16, 1)])
# @pytest.mark.parametrize("kv_lora_rank, qk_rope_head_dim", [(512, 64)])
# @pytest.mark.parametrize("block_size", [64])
# @pytest.mark.parametrize("num_blocks", [32768])
# @pytest.mark.parametrize("varlen", [True, False])
# @pytest.mark.parametrize(
#     "q_dtype, kv_dtype, out_dtype, use_out_scale",
#     [
#         (torch.bfloat16, torch.bfloat16, torch.bfloat16, False),
#         (torch.bfloat16, e4m3_dtype, torch.bfloat16, True),
#         (e4m3_dtype, e4m3_dtype, torch.bfloat16, True),
#     ],
# )
# @pytest.mark.parametrize("shuffled_kv_cache", [True, False])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("decode_qlen", [1])
@pytest.mark.parametrize("ctx_lens", [1328])
@pytest.mark.parametrize("num_heads", [(16, 1)])
@pytest.mark.parametrize("kv_lora_rank, qk_rope_head_dim", [(512, 64)])
@pytest.mark.parametrize("num_blocks", [128])
@pytest.mark.parametrize("varlen", [False])
@pytest.mark.parametrize(
    "q_dtype, kv_dtype, out_dtype, block_size, use_out_scale",
    [
        # (torch.bfloat16, torch.bfloat16, torch.bfloat16, 64, False),
        # (torch.bfloat16, e4m3_dtype, torch.bfloat16, 64, True),
        # (e4m3_dtype, e4m3_dtype, torch.bfloat16, 64, True),
        (e4m3_dtype, torch.uint8, torch.bfloat16, 128, True),
        (torch.uint8, torch.uint8, torch.bfloat16, 128, True),
    ],
)
@pytest.mark.parametrize("shuffled_kv_cache", [True])
@torch.inference_mode()
def test_mla_decode_fwd(
    batch_size: int,
    decode_qlen: int,
    ctx_lens: int,
    num_heads: tuple[int, int],
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_size: int,
    num_blocks: int,
    varlen: bool,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    out_dtype: torch.dtype,
    use_out_scale: bool,
    shuffled_kv_cache: bool,
):
    torch.cuda.empty_cache()
    num_query_heads, num_kv_heads = num_heads
    qk_head_dim = kv_lora_rank + qk_rope_head_dim

    if kv_dtype == torch.uint8:
        if not shuffled_kv_cache:
            pytest.skip("NVFP4 requires shuffled KV cache")

    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int, device="cuda")
    seq_lens_qo = torch.empty(batch_size, dtype=torch.int, device="cuda")
    seq_lens_kv = torch.empty(batch_size, dtype=torch.int, device="cuda")
    if varlen:
        for i in range(batch_size):
            seq_lens_kv[i] = max(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens)
    else:
        seq_lens_kv.fill_(ctx_lens)
    seq_lens_qo.fill_(decode_qlen)

    cu_seqlens_q[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)
    total_num_query_tokens = cu_seqlens_q[-1].item()

    max_seqlen_kv = seq_lens_kv.max().item()
    max_num_blocks_per_seq = (max_seqlen_kv + block_size - 1) // block_size
    block_tables = torch.randint(
        0,
        num_blocks,
        (batch_size, max_num_blocks_per_seq),
        dtype=torch.int32,
        device="cuda",
    )
    kv_buffer = torch.randn(
        (num_blocks, block_size, num_kv_heads, qk_head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    query = torch.randn(
        (total_num_query_tokens, num_query_heads, qk_head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    query_scales = None
    if q_dtype == torch.uint8:
        query = query / 10
        maybe_quant_query = query.view(-1, qk_head_dim)
        maybe_quant_query, query_scales = torch_dynamic_mxfp4_quant(
            maybe_quant_query, is_nvfp4=True
        )
        maybe_quant_query = maybe_quant_query.view(
            -1, num_query_heads, qk_head_dim // 2
        )
        query_scales = query_scales.view(-1, num_query_heads, qk_head_dim // 16)
        query = query.to(e4m3_dtype)
    else:
        query = query.to(q_dtype)
        maybe_quant_query = query

    # .to(q_dtype)
    sm_scale = 1.0 / (qk_head_dim**0.5)

    q_descale = None
    kv_descale = None
    if q_dtype != torch.bfloat16:
        q_descale = torch.rand((1,), dtype=torch.float32, device="cuda")

    if kv_dtype != torch.bfloat16:
        kv_descale = torch.rand((1,), dtype=torch.float32, device="cuda")

    out_scale = None
    if use_out_scale:
        out_scale = 1 / torch.rand(1, dtype=torch.float32, device="cuda")

    out = torch.empty(
        (total_num_query_tokens, num_query_heads, kv_lora_rank), dtype=out_dtype
    )

    if shuffled_kv_cache:
        if kv_dtype == torch.uint8:
            maybe_shuffled_kv_buffer = dynamic_nvfp4_quant_kv_buffer(
                kv_buffer, kv_lora_rank
            )
            kv_buffer = kv_buffer.to(e4m3_dtype)
        else:
            kv_buffer = kv_buffer.to(kv_dtype)
            maybe_shuffled_kv_buffer = shuffle_kv_buffer(kv_buffer, kv_lora_rank)
    else:
        kv_buffer = kv_buffer.to(kv_dtype)
        maybe_shuffled_kv_buffer = kv_buffer

    mla_decode_fwd(
        q=maybe_quant_query,
        kv_buffer=maybe_shuffled_kv_buffer,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seq_lens_kv,
        max_seqlen_kv=max_seqlen_kv,
        block_tables=block_tables,
        softmax_scale=sm_scale,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        causal=True,
        q_descale=q_descale,
        kv_descale=kv_descale,
        q_scales=query_scales,
        out_scale=out_scale,
        shuffled_kv_cache=shuffled_kv_cache,
    )

    out_ref = torch_mla_extend(
        query,
        kv_buffer,
        cu_seqlens_q,
        seq_lens_kv,
        block_tables,
        kv_lora_rank,
        sm_scale,
        q_descale=q_descale,
        kv_descale=kv_descale,
        out_scale=out_scale,
        o_dtype=out_dtype,
    )

    atol, rtol = 1.5e-2, 1e-2
    if q_dtype == e4m3_dtype or kv_dtype == e4m3_dtype:
        atol, rtol = 1.5e-1, 1.5e-1
    torch.testing.assert_close(
        out, out_ref, atol=atol, rtol=rtol
    ), f"{torch.max(torch.abs(out - out_ref))}"


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("ctx_lens", [200])
@pytest.mark.parametrize("num_heads", [(16, 1)])
@pytest.mark.parametrize("kv_lora_rank, qk_rope_head_dim", [(512, 64)])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("num_blocks", [16384])
@pytest.mark.parametrize("varlen", [True, False])
@pytest.mark.parametrize(
    "q_dtype, kv_dtype, out_dtype, use_out_scale",
    [
        (torch.bfloat16, torch.bfloat16, torch.bfloat16, False),
        (torch.bfloat16, e4m3_dtype, torch.bfloat16, True),
        (e4m3_dtype, e4m3_dtype, torch.bfloat16, True),
    ],
)
def test_mla_prefill_fwd(
    batch_size: int,
    ctx_lens: int,
    num_heads: tuple[int, int],
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_size: int,
    num_blocks: int,
    varlen: bool,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    out_dtype: torch.dtype,
    use_out_scale: bool,
):
    torch.cuda.empty_cache()
    num_query_heads, num_kv_heads = num_heads
    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int, device="cuda")
    seq_lens_qo = torch.empty(batch_size, dtype=torch.int, device="cuda")
    seq_lens_kv = torch.empty(batch_size, dtype=torch.int, device="cuda")
    if varlen:
        for i in range(batch_size):
            seq_lens_kv[i] = max(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens)
            seq_lens_qo[i] = max(
                min(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens), 1
            )
    else:
        seq_lens_kv.fill_(ctx_lens)
        seq_lens_qo.fill_(ctx_lens)

    cu_seqlens_q[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)
    total_num_query_tokens = cu_seqlens_q[-1].item()

    max_seqlen_kv = seq_lens_kv.max().item()
    max_num_blocks_per_seq = (max_seqlen_kv + block_size - 1) // block_size
    block_tables = torch.randint(
        0,
        num_blocks,
        (batch_size, max_num_blocks_per_seq),
        dtype=torch.int32,
        device="cuda",
    )
    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    kv_buffer = torch.randn(
        (num_blocks, block_size, num_kv_heads, qk_head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    ).to(kv_dtype)
    q = torch.randn(
        (total_num_query_tokens, num_query_heads, qk_head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    ).to(q_dtype)
    sm_scale = 1.0 / (qk_head_dim**0.5)

    q_descale = None
    kv_descale = None
    if q_dtype != torch.bfloat16:
        q_descale = torch.rand((1,), dtype=torch.float32, device="cuda")

    if kv_dtype != torch.bfloat16:
        kv_descale = torch.rand((1,), dtype=torch.float32, device="cuda")

    out_scale = None
    if use_out_scale:
        out_scale = 1 / torch.rand(1, dtype=torch.float32, device="cuda")

    out = torch.empty(
        (total_num_query_tokens, num_query_heads, kv_lora_rank), dtype=out_dtype
    )

    mla_prefill_fwd(
        q,
        kv_buffer,
        out,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seq_lens_kv,
        max_seqlen_kv=max_seqlen_kv,
        block_tables=block_tables,
        softmax_scale=sm_scale,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        causal=True,
        q_descale=q_descale,
        kv_descale=kv_descale,
        out_scale=out_scale,
        shuffled_kv_cache=False,
    )

    out_ref = torch_mla_extend(
        q,
        kv_buffer,
        cu_seqlens_q,
        seq_lens_kv,
        block_tables,
        kv_lora_rank,
        sm_scale,
        q_descale=q_descale,
        kv_descale=kv_descale,
        out_scale=out_scale,
        o_dtype=out_dtype,
    )

    atol, rtol = 1.5e-2, 1e-2
    if q_dtype == e4m3_dtype or kv_dtype == e4m3_dtype:
        atol, rtol = 1.5e-1, 1.5e-1
    torch.testing.assert_close(
        out, out_ref, atol=atol, rtol=rtol
    ), f"{torch.max(torch.abs(out - out_ref))}"
