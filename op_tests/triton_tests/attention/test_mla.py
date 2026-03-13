# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import random
import pytest
import torch

import aiter
from aiter.ops.triton.attention.mla import mla_decode_fwd as triton_mla_decode_fwd
from aiter.ops.triton.attention.mla import mla_prefill_fwd as triton_mla_prefill_fwd

from aiter.ops.triton.utils.types import e4m3_dtype
from typing import Optional

torch.set_default_device("cuda")

# current supported case in decode MLA: mtp == 0, 1, 2, 3 (decode_qlen = 1, 2, 3, 4)


def ref_masked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    query_len = q.shape[0]
    kv_len = k.shape[0]
    if q.shape[1] != k.shape[1]:
        k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
        v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
    attn = torch.einsum("qhd,khd->hqk", q, k).float()  # GEMM at q.dtype precision
    empty_mask = torch.ones(query_len, kv_len, device=q.device)
    mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
    attn.masked_fill_(mask, float("-inf"))
    attn = torch.softmax(attn, dim=-1).to(v.dtype)
    out = torch.einsum("hqk,khd->qhd", attn, v)  # GEMM at q.dtype precision

    return out


# def torch_mha_extend(
#     query,  # [total_q, num_query_heads, qk_lora_rank + qk_rope_head_dim]
#     key_cache,  # [num_block, block_size, num_kv_heads, qk_lora_rank + qk_rope_head_dim]
#     value_cache,  # [num_block, block_size, num_kv_heads, v_head_dim]
#     cu_seqlens_q,
#     kv_lens,
#     block_tables,
#     scale: float,
#     q_descale: Optional[torch.Tensor] = None,
#     kv_descale: Optional[torch.Tensor] = None,
#     o_dtype: Optional[torch.dtype] = torch.bfloat16,
# ) -> torch.Tensor:
#     _, block_size, num_kv_heads, qk_head_dim = key_cache.shape
#     v_head_dim = value_cache.shape[-1]
#     num_seqs = cu_seqlens_q.shape[0] - 1

#     query = query.to(torch.float32)
#     kv_buffer = kv_buffer.to(torch.float32)
#     query = query * scale
#     if q_descale is not None:
#         query = query * q_descale
#     if kv_descale is not None:
#         kv_buffer = kv_buffer * kv_descale

#     outputs: list[torch.Tensor] = []
#     for i in range(num_seqs):
#         q = query[cu_seqlens_q[i] : cu_seqlens_q[i + 1]]

#         kv_len = kv_lens[i]
#         num_kv_blocks = (kv_len + block_size - 1) // block_size
#         block_indices = block_tables[i, :num_kv_blocks]

#         k = key_cache[block_indices].view(-1, num_kv_heads, qk_head_dim)
#         k = k[:kv_len]
#         v = value_cache[block_indices].view(-1, num_kv_heads, v_head_dim)
#         v = v[:kv_len]

#         out = ref_masked_attention(q, k, v)

#         outputs.append(out)

#     return torch.cat(outputs, dim=0).to(o_dtype)


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

    query = query.to(torch.float32)
    kv_buffer = kv_buffer.to(torch.float32)
    query = query * scale
    if q_descale is not None:
        query = query * q_descale
    if kv_descale is not None:
        kv_buffer = kv_buffer * kv_descale

    outputs: list[torch.Tensor] = []
    for i in range(num_seqs):
        q = query[cu_seqlens_q[i] : cu_seqlens_q[i + 1]]

        kv_len = seq_lens_kv[i]
        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = kv_buffer[block_indices].view(-1, num_kv_heads, qk_head_dim)
        k = k[:kv_len]
        v = k[..., :qk_lora_rank]

        out = ref_masked_attention(q, k, v)

        outputs.append(out)

    out = torch.cat(outputs, dim=0)
    if out_scale is not None:
        out = out / out_scale
    return out.to(o_dtype)


@pytest.mark.parametrize("batch_size", [1, 4, 7, 8, 32])
@pytest.mark.parametrize("decode_qlen", [1, 2, 4, -1])
@pytest.mark.parametrize("ctx_lens", [1024, 8192])
@pytest.mark.parametrize("num_heads", [(16, 1)])
@pytest.mark.parametrize("kv_lora_rank, qk_rope_head_dim", [(512, 64)])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("num_blocks", [2048])
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
    "backend",
    [
        "triton",  # use triton
    ],
)
# @torch.inference_mode()
def test_mla(
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
):
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
    if decode_qlen > -1:
        seq_lens_qo.fill_(decode_qlen)
    else:
        seq_lens_qo.fill_(ctx_lens)
        if varlen:
            for i in range(batch_size):
                seq_lens_qo[i] = max(
                    min(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens), 1
                )
        else:
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

    if backend == "triton":
        if decode_qlen > -1:
            triton_mla_decode_fwd(
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
            )
        else:
            triton_mla_prefill_fwd(
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
