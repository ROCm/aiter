# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import random
import pytest
import torch

import aiter
from aiter.ops.triton.attention.mla import mla_decode_fwd
from aiter.ops.triton.attention.mla import mla_prefill_fwd
import aiter.ops.triton.utils._triton.arch_info as arch_info

DEVICE_ARCH = arch_info.get_arch()
from aiter.ops.triton.utils.types import e4m3_dtype
from typing import Optional

torch.set_default_device("cuda")

# current supported case in decode MLA: mtp == 0, 1, 2, 3 (decode_qlen = 1, 2, 3, 4)


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

    def shuffle(kvb, h):
        kvb = kvb.view(
            -1,
            num_kv_heads,
            block_size // num_lanes,
            num_lanes,
            h // (2 * num_elements_per_thread),
            2,  # there are 2 groups of threads, t0 ~ t15 and t16 ~ t31
            num_elements_per_thread,
        )
        kvb = kvb.permute(0, 1, 2, 4, 5, 3, 6).contiguous()
        kvb = kvb.view(-1, num_kv_heads, block_size // 16, h * 16)
        return kvb

    kv_buffer_shuffled = kv_buffer.view(
        -1, block_size, num_kv_heads, head_size
    ).permute(0, 2, 1, 3)
    kv_buffer_shuffled_lora = shuffle(
        kv_buffer_shuffled[..., :kv_lora_rank], kv_lora_rank
    )
    kv_buffer_shuffled_rope = shuffle(
        kv_buffer_shuffled[..., kv_lora_rank:], head_size - kv_lora_rank
    )
    kv_buffer_shuffled = torch.cat(
        [kv_buffer_shuffled_lora, kv_buffer_shuffled_rope], dim=-1
    ).contiguous()

    return kv_buffer_shuffled


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


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("decode_qlen", [1, 2, 4])
@pytest.mark.parametrize("ctx_lens", [1024])
@pytest.mark.parametrize("num_heads", [(16, 1)])
@pytest.mark.parametrize("kv_lora_rank, qk_rope_head_dim", [(512, 64)])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("num_blocks", [128])
@pytest.mark.parametrize("varlen", [True, False])
@pytest.mark.parametrize(
    "q_dtype, kv_dtype, out_dtype, use_out_scale",
    [
        (torch.bfloat16, torch.bfloat16, torch.bfloat16, False),
        (torch.bfloat16, e4m3_dtype, torch.bfloat16, True),
        (e4m3_dtype, e4m3_dtype, torch.bfloat16, True),
    ],
)
@pytest.mark.parametrize(
    "backend, shuffled_kv_cache",
    [
        ("triton", False),
        ("gluon", True),
        ("gluon", False),
    ],
)
# @torch.inference_mode()
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
    backend: str,
    shuffled_kv_cache: bool,
):
    if backend == "triton" and DEVICE_ARCH in ("gfx1250",):
        pytest.skip("Triton is not supported on gfx1250")
    if backend == "gluon" and DEVICE_ARCH not in ("gfx1250",):
        pytest.skip("Gluon is only supported on gfx1250")
    torch.cuda.empty_cache()
    num_query_heads, num_kv_heads = num_heads
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

    maybe_shuffled_kv_buffer = (
        shuffle_kv_buffer(kv_buffer, kv_lora_rank) if shuffled_kv_cache else kv_buffer
    )
    mla_decode_fwd(
        q,
        maybe_shuffled_kv_buffer,
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
        use_gluon=(backend == "gluon"),
        shuffled_kv_cache=shuffled_kv_cache,
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


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("ctx_lens", [200])
@pytest.mark.parametrize("num_heads", [(16, 1)])
@pytest.mark.parametrize("kv_lora_rank, qk_rope_head_dim", [(512, 64)])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("num_blocks", [128])
@pytest.mark.parametrize("varlen", [True, False])
@pytest.mark.parametrize(
    "q_dtype, kv_dtype, out_dtype, use_out_scale",
    [
        (torch.bfloat16, torch.bfloat16, torch.bfloat16, False),
        (torch.bfloat16, e4m3_dtype, torch.bfloat16, True),
        (e4m3_dtype, e4m3_dtype, torch.bfloat16, True),
    ],
)
@pytest.mark.parametrize("backend", ["triton", "gluon"])
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
    backend: str,
):
    if backend == "triton" and DEVICE_ARCH in ("gfx1250",):
        pytest.skip("Triton is not supported on gfx1250")
    if backend == "gluon" and DEVICE_ARCH not in ("gfx1250",):
        pytest.skip("Gluon is only supported on gfx1250")
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
        use_gluon=(backend == "gluon"),
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
