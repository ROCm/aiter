# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter.test_common import checkAllclose, benchmark, run_perftest
from aiter import dtypes
import random
import itertools
import argparse
import triton
import triton.language as tl
import pandas as pd

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)

# current supported case in ps decode MLA: mtp == 0, 1, 2, 3 (decode_qlen = 1, 2, 3, 4)
# qdtype bf16, kdtype bf16: nhead16
# qdtype fp8, kdtype fp8: nhead16, nhead128
# qdtype fp8, kdtype bf16: nhead16


def init_3buffer_kv_cache(
    num_page: int,
    page_size: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    scale_dim: int,
) -> tuple:
    """
    Initialize KV cache for 3BUFFER layout with FP8 quantization.

    Generates random KV cache data and applies per-channel quantization to the nope buffer.

    Args:
        num_page: Number of pages
        page_size: Size of each page (block size)
        kv_lora_rank: Rank of KV LoRA (nope dimension)
        qk_rope_head_dim: Dimension of RoPE (rope dimension)
        scale_dim: Number of scale factors per nope buffer

    Returns:
        tuple containing:
            - kv_buffer: Concatenated buffer (BF16), shape (num_page, page_size, 1, kv_lora_rank + qk_rope_head_dim)
            - kv_nope_buffer_fp8: Quantized nope buffer (FP8), shape (num_page, page_size, 1, kv_lora_rank)
            - kv_nope_scale_factors_fp32: Scale factors (FP32), shape (num_page, page_size, 1, scale_dim)
            - kv_rope_buffer_bf16: Rope buffer (BF16), shape (num_page, page_size, 1, qk_rope_head_dim)
            - kv_nope_buffer_fp32: Original nope buffer (FP32), shape (num_page, page_size, 1, kv_lora_rank)
    """
    assert (
        kv_lora_rank % scale_dim == 0
    ), f"kv_lora_rank ({kv_lora_rank}) must be divisible by scale_dim ({scale_dim})"

    kv_nope_buffer_fp32 = torch.ones(
        (num_page, page_size, 1, kv_lora_rank), dtype=torch.float32
    )
    kv_rope_buffer_bf16 = torch.ones(
        (num_page, page_size, 1, qk_rope_head_dim),
        dtype=torch.bfloat16,
    )

    # Create full KV buffer (for golden reference without quantization)
    kv_buffer = torch.cat(
        [kv_nope_buffer_fp32.to(torch.bfloat16), kv_rope_buffer_bf16], dim=-1
    )

    # Generate random scale factors
    scale_values = [1.0, 2.0, 4.0, 8.0]
    # scale_values = [1.0, 1.0, 1.0, 1.0]
    scale_indices = torch.randint(
        0, len(scale_values), size=(num_page, page_size, 1, scale_dim)
    )
    kv_nope_scale_factors_fp32 = torch.tensor(
        [scale_values[idx] for idx in scale_indices.flatten()], dtype=torch.float32
    ).reshape(num_page, page_size, 1, scale_dim)

    # Apply per-channel scaling and quantize to FP8
    kv_nope_scaled_buffer = kv_nope_buffer_fp32.reshape(
        num_page, page_size, 1, scale_dim, kv_lora_rank // scale_dim
    ) / kv_nope_scale_factors_fp32.reshape(num_page, page_size, 1, scale_dim, 1)

    kv_nope_buffer_fp8 = kv_nope_scaled_buffer.reshape(
        num_page, page_size, 1, kv_lora_rank
    ).to(dtypes.fp8)

    return (
        kv_buffer,
        kv_nope_buffer_fp8,
        kv_nope_scale_factors_fp32,
        kv_rope_buffer_bf16,
        kv_nope_buffer_fp32,
    )


def split_3buffer_kv_cache(
    kv_buffer_bytes: torch.Tensor,
    page_size: int,
    nhead_kv: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    scale_dim: int,
) -> tuple:
    """
    Split concatenated KV cache buffer back into 3 separate buffers.

    This is the inverse operation of concatenating after flattening last 3 dimensions.

    Args:
        kv_buffer_bytes: Concatenated buffer (uint8), shape (num_page, page_size*656)
                        where 656 = 512(nope) + 16(scale) + 128(rope)
        page_size: Size of each page (block size)
        nhead_kv: Number of heads in the KV cache
        kv_lora_rank: Rank of KV LoRA (nope dimension)
        qk_rope_head_dim: Dimension of RoPE (rope dimension)
        scale_dim: Number of scale factors per nope buffer

    Returns:
        tuple containing:
            - kv_nope_buffer_fp8: Quantized nope buffer (FP8), shape (num_page, page_size, 1, kv_lora_rank)
            - kv_nope_scale_factors_fp32: Scale factors (FP32), shape (num_page, page_size, 1, scale_dim)
            - kv_rope_buffer_bf16: Rope buffer (BF16), shape (num_page, page_size, 1, qk_rope_head_dim)
    """
    num_page = kv_buffer_bytes.shape[0]

    nope_total_bytes = page_size * nhead_kv * kv_lora_rank * 1  # FP8: 1 byte/elem
    scale_total_bytes = page_size * nhead_kv * scale_dim * 4  # FP32: 4 bytes/elem
    rope_total_bytes = page_size * nhead_kv * qk_rope_head_dim * 2  # BF16: 2 bytes/elem

    nope_flat = kv_buffer_bytes[:, 0:nope_total_bytes]
    scale_flat = kv_buffer_bytes[
        :, nope_total_bytes : nope_total_bytes + scale_total_bytes
    ]
    rope_flat = kv_buffer_bytes[
        :,
        nope_total_bytes
        + scale_total_bytes : nope_total_bytes
        + scale_total_bytes
        + rope_total_bytes,
    ]

    nope_bytes = nope_flat.reshape(num_page, page_size, nhead_kv, kv_lora_rank * 1)
    scale_bytes = scale_flat.reshape(num_page, page_size, nhead_kv, scale_dim * 4)
    rope_bytes = rope_flat.reshape(num_page, page_size, nhead_kv, qk_rope_head_dim * 2)

    # Convert bytes back to original dtypes
    kv_nope_buffer_fp8 = (
        nope_bytes.contiguous()
        .view(dtypes.fp8)
        .reshape(num_page, page_size, nhead_kv, kv_lora_rank)
    )

    kv_nope_scale_factors_fp32 = (
        scale_bytes.contiguous()
        .view(torch.float32)
        .reshape(num_page, page_size, nhead_kv, scale_dim)
    )

    kv_rope_buffer_bf16 = (
        rope_bytes.contiguous()
        .view(torch.bfloat16)
        .reshape(num_page, page_size, nhead_kv, qk_rope_head_dim)
    )

    return kv_nope_buffer_fp8, kv_nope_scale_factors_fp32, kv_rope_buffer_bf16


def check_support(dtype, kv_dtype, nhead):
    if dtype == dtypes.fp8 and kv_dtype == dtypes.bf16:
        return False
    return True


def cal_diff(
    x: torch.Tensor, y: torch.Tensor, name: str, use_fp8: bool = False
) -> None:
    x, y = x.double(), y.double()
    # RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    # amax_diff = (x - y).abs().max().item()
    # print(f"{name}: {cos_diff=}, {RMSE=}, {amax_diff=}")
    if use_fp8:
        assert cos_diff < 3e-2
    else:
        assert cos_diff < 1e-5


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    dtype,
    is_causal=True,
    is_fp8_q=False,
    is_fp8_kvc=False,
    q_scale=None,
    kv_scale=None,
    valid_mask=None,
):

    if is_fp8_q and q_scale is not None:
        scale *= q_scale
    if is_fp8_kvc and kv_scale is not None:
        scale *= kv_scale

    attn_weights = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * scale

    if valid_mask is not None:
        # valid_mask: [seq_k] -> broadcast to [1, 1, seq_k] -> [nhead, seq_q, seq_k]
        attn_bias = torch.zeros_like(attn_weights)
        attn_bias.masked_fill_(~valid_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weights += attn_bias

    if is_causal:
        s_q = query.shape[0]
        s_k = key.shape[0]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weights += attn_bias

    lse = attn_weights.logsumexp(dim=-1)

    m = attn_weights.max(-1).values

    attn_weights_exp = torch.exp(attn_weights - m.unsqueeze(-1))

    l = attn_weights_exp.sum(-1)  # noqa: E741

    if is_fp8_q:
        attn_weights_fp8 = attn_weights_exp.to(dtype)
        attn_weights_exp = attn_weights_fp8.to(torch.float)

    out = torch.einsum("hqk,khd->qhd", attn_weights_exp.float(), value.float())
    out = out / l.transpose(0, 1).unsqueeze(-1)
    if is_fp8_kvc and kv_scale is not None:
        out *= kv_scale
    return out.to(dtype), lse


def torch_mla_extend(
    q,  # [total_q, nheads, headdim_q]
    kvc_cache,  # [num_page, page_size, nhead_kv, qk_head_dim]
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    sm_scale,
    kv_lora_rank,
    qk_rope_head_dim,
    dtype,
    is_causal=True,
    q_scale=None,
    kv_scale=None,
    valid_mask=None,  # [batch_size, num_topk] - bool tensor
):

    num_page, page_size, nhead_kv, _ = kvc_cache.shape
    is_fp8_q = q.dtype == dtypes.fp8
    is_fp8_kvc = kvc_cache.dtype == dtypes.fp8

    if is_fp8_q:
        q = q.to(torch.float)

    if is_fp8_kvc:
        kvc_cache = kvc_cache.to(torch.float)

    qs = torch.tensor_split(q, qo_indptr.tolist()[1:])
    kvc = torch.index_select(kvc_cache, 0, kv_indices)
    # print(f"kvc[1,:,:] is {kvc[1,:,:]}")
    kvs = torch.tensor_split(kvc, kv_indptr.tolist()[1:])
    bs = qo_indptr.shape[0] - 1

    os = []
    lses = []
    for i in range(bs):
        cur_num_page = kvs[i].shape[0]
        real_kv_seq_len = (cur_num_page - 1) * page_size + kv_last_page_lens.tolist()[i]
        kvc = kvs[i].flatten(0, 1)[:real_kv_seq_len,]
        q = qs[i]
        k = kvc
        v, _ = torch.split(kvc, [kv_lora_rank, qk_rope_head_dim], dim=-1)
        valid_mask_i = None
        if valid_mask is not None:
            valid_mask_i = valid_mask[i, :real_kv_seq_len]

        o, lse = ref_masked_attention(
            q,
            k,
            v,
            sm_scale,
            dtype,
            is_causal=is_causal,
            is_fp8_q=is_fp8_q,
            is_fp8_kvc=is_fp8_kvc,
            q_scale=q_scale,
            kv_scale=kv_scale,
            valid_mask=valid_mask_i,
        )
        os.append(o)
        lses.append(lse)
    o = torch.concat(os)
    lse = torch.concat(lses).transpose(0, 1)
    return o, lse


def torch_mla_extend_3buffer(
    q,  # [total_q, nheads, headdim_q]
    kvc_cache,  # [num_page, page_size*(nhead_kv*(kv_lora_rank+scale_dim+qk_rope_head_dim))]
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    page_size,
    nhead_kv,
    sm_scale,
    kv_lora_rank,
    qk_rope_head_dim,
    dtype,
    is_causal=True,
    q_scale=None,
    kv_scale=None,
    scale_dim=4,
    valid_mask=None,
):
    num_page = kvc_cache.shape[0]
    kv_nope_buffer_fp8, kv_nope_scale_factors_fp32, kv_rope_buffer_bf16 = (
        split_3buffer_kv_cache(
            kvc_cache, page_size, nhead_kv, kv_lora_rank, qk_rope_head_dim, scale_dim
        )
    )
    kv_nope_buffer_fp32 = kv_nope_buffer_fp8.to(torch.float32).reshape(
        num_page, page_size, nhead_kv, scale_dim, -1
    ) * kv_nope_scale_factors_fp32.reshape(num_page, page_size, nhead_kv, scale_dim, 1)
    kvc_cache_bf16 = torch.cat(
        [
            kv_nope_buffer_fp32.reshape(num_page, page_size, nhead_kv, kv_lora_rank).to(
                torch.bfloat16
            ),
            kv_rope_buffer_bf16,
        ],
        dim=-1,
    )
    return torch_mla_extend(
        q,
        kvc_cache_bf16,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        sm_scale,
        kv_lora_rank,
        qk_rope_head_dim,
        dtype,
        is_causal,
        q_scale,
        kv_scale,
        valid_mask,
    )


def generate_topk_kv(
    kv_len_indptr: torch.Tensor,
    qo_len: int = 1,
    NUM_TOPK_TOKENS: int = 2048,
):
    batch_size = kv_len_indptr.shape[0] - 1
    batch_size = batch_size * qo_len
    token_indices = torch.empty([batch_size, NUM_TOPK_TOKENS], dtype=torch.int32)
    for i in range(batch_size):
        i_ori = i // qo_len
        kv_end = kv_len_indptr[i_ori + 1]
        kv_start = kv_len_indptr[i_ori]
        kv_len = kv_end - kv_start

        if kv_len < NUM_TOPK_TOKENS:
            token_indices[i, :kv_len] = torch.arange(0, kv_len, dtype=torch.int32)
        else:
            token_indices[i] = torch.randint(
                0, kv_len, (NUM_TOPK_TOKENS,), dtype=torch.int32
            )

    return token_indices


def sparse_kv_indptr_to_dense(
    kv_indptr: torch.Tensor,
    converted_indices: torch.Tensor,
    qo_len: int = 1,
    NUM_TOPK_TOKENS: int = 2048,
    page_size: int = 1,
):
    new_kv_indptr = [0]
    indices_list = []
    new_kv_last_page_lens = []
    batch_size = kv_indptr.shape[0] - 1
    batch_size = qo_len * batch_size
    for i in range(batch_size):
        i_ori = i // qo_len
        kv_len = kv_indptr[i_ori + 1] - kv_indptr[i_ori]
        kv_len = min(kv_len, NUM_TOPK_TOKENS)
        kv_num_pages = (kv_len + page_size - 1) // page_size
        indices_list.append(converted_indices[i, :kv_len])
        new_kv_indptr.append(kv_num_pages + new_kv_indptr[i])
        if kv_len % page_size == 0:
            new_kv_last_page_lens.append(page_size)
        else:
            new_kv_last_page_lens.append(kv_len % page_size)

    return (
        torch.arange(0, batch_size + 1, dtype=torch.int32),
        torch.tensor(new_kv_indptr, dtype=torch.int32),
        torch.tensor(new_kv_last_page_lens, dtype=torch.int32),
        torch.concat(indices_list),
    )


def gather_sparse_kv_buffer(
    kv_buffer: torch.Tensor,  # [num_page , page_size, nhead_kv, dim]
    converted_indices: torch.Tensor,  # [num_tokens, num_topk] - contains -1 for invalid tokens
    fill_value: float = 0.0,
):
    """
    Gather sparse KV buffer ??? valid_mask

    Returns:
        gathered_kv: [num_tokens, num_topk, nhead_kv, dim]
        valid_mask: [num_tokens, num_topk] - bool tensor
    """
    page_num, page_size, nhead_kv, head_dim = kv_buffer.shape
    total_kv_page_nums = page_num * page_size
    num_tokens, num_topk = converted_indices.shape
    valid_mask = (converted_indices >= 0) & (converted_indices < total_kv_page_nums)
    safe_indices = torch.where(valid_mask, converted_indices, 0)
    gathered_kv = (kv_buffer.reshape(total_kv_page_nums, nhead_kv, head_dim))[
        safe_indices
    ]
    gathered_kv = torch.where(
        valid_mask.unsqueeze(-1).unsqueeze(-1),
        gathered_kv,
        torch.full_like(gathered_kv, fill_value),
    )
    print(f"valid_mask.shape is {valid_mask.shape}")
    return gathered_kv.reshape(-1, page_size, nhead_kv, head_dim), valid_mask


def get_dense_kv_indptr_and_indices(
    converted_indices: torch.Tensor,
    new_kv_len_indptr: torch.Tensor,
    page_size: int = 1,
):
    num_tokens, topk = converted_indices.shape
    new_kv_indptr = [0]
    new_kv_last_page_lens = []
    total_kv_page_nums = 0
    for i in range(num_tokens):
        kv_num_pages = (topk + page_size - 1) // page_size
        total_kv_page_nums = total_kv_page_nums + kv_num_pages
        new_kv_indptr.append(kv_num_pages + new_kv_indptr[i])
        kv_len = new_kv_len_indptr[i + 1] - new_kv_len_indptr[i]
        kv_len = min(kv_len, topk)
        if kv_len % page_size == 0:
            new_kv_last_page_lens.append(page_size)
        else:
            new_kv_last_page_lens.append(kv_len % page_size)

    return (
        torch.tensor(new_kv_indptr, dtype=torch.int32),
        torch.tensor(new_kv_last_page_lens, dtype=torch.int32),
        torch.arange(0, total_kv_page_nums, dtype=torch.int32),
    )


@triton.jit
def _convert_req_index_to_global_index_kernel(
    kv_indptr,  # int32 [num_requests]
    kv_indices,  # int32 [num_requests * max_num_blocks_per_req]
    kv_last_page_lens,  # int32 [num_requests]
    token_indices_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    out_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    # shapes (compile-time where possible)
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,  # tile width along columns
    # strides (in elements)
    bt_stride0: tl.constexpr,
    ti_stride0: tl.constexpr,
    ti_stride1: tl.constexpr,
    out_stride0: tl.constexpr,
    out_stride1: tl.constexpr,
    qo_len: tl.constexpr,
):
    # program_id(0) -> token_id (row)
    # program_id(1) -> tile index along columns
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    # Each program covers BLOCK_N consecutive columns
    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
    batch_id = token_id // qo_len

    # Load request id for this token (no mask: grid is exact)
    kv_start = tl.load(kv_indptr + batch_id)
    kv_end = tl.load(kv_indptr + batch_id + 1)
    kv_remain = tl.load(kv_last_page_lens + batch_id)
    kv_len = (kv_end - kv_start - 1) * BLOCK_SIZE + kv_remain

    # Load token indices for this tile
    ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
    tok = tl.load(ti_ptr)  # int32

    # Only token == -1 should propagate as -1
    is_invalid_tok = tok < 0

    # Compute block id and in-block offset
    block_id = tok // BLOCK_SIZE
    inblock_off = tok % BLOCK_SIZE

    # Guard block_table access
    valid_block = indice_id < kv_len
    # tl.device_print("offset", valid_block)
    base = tl.load(
        kv_indices + kv_start + block_id * bt_stride0, mask=valid_block, other=0
    )

    # base = 0

    # If token == -1 OR block_id OOB, output -1; else base * BLOCK_SIZE + offset
    out_val = tl.where(
        is_invalid_tok | (~valid_block), -1, base * BLOCK_SIZE + inblock_off
    )

    # Store results
    out_ptr_ij = out_ptr + token_id * out_stride0 + indice_id * out_stride1
    tl.store(out_ptr_ij, out_val)


def triton_convert_req_index_to_global_index(
    kv_indptr: torch.Tensor,  # int32 [batch_size + 1]
    kv_indices: torch.Tensor,  # int32 [total_kv_page_nums]
    kv_last_page_lens: torch.Tensor,  # int32 [batch_size]
    token_indices: torch.Tensor,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    qo_len: int = 1,
    BLOCK_SIZE: int = 1,  # page_block_size = 1 for now
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,  # tile width along columns
):
    """
    out[token_id, indice_id] =
        block_table[req_id[token_id],
            token_indices[token_id, indice_id] // BLOCK_SIZE] * BLOCK_SIZE
        + token_indices[token_id, indice_id] % BLOCK_SIZE

    Only when token_indices[token_id, indice_id] == -1 do we output -1.
    For safety, we also output -1 if the derived block_id would be
        out-of-bounds.
    """
    assert kv_indices.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0, (
        f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible by"
        f"BLOCK_N ({BLOCK_N})"
    )

    # num_batches = kv_indptr.shape[0] - 1
    num_tokens = token_indices.shape[0]

    # num_requests, max_num_blocks_per_req = block_table.shape
    # max_num_blocks_per_req = 65536 * 32
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    # Ensure contiguous tensors on the same device
    kv_indptr_c = kv_indptr.contiguous()
    kv_indices_c = kv_indices.contiguous()
    kv_last_page_lens_c = kv_last_page_lens.contiguous()
    token_indices_c = token_indices.contiguous()
    out = torch.empty_like(token_indices_c)

    # Strides in elements
    bt_stride0 = kv_indices_c.stride()[0]
    ti_stride0, ti_stride1 = token_indices_c.stride()
    out_stride0, out_stride1 = out.stride()

    # Exact 2D grid: tokens x column tiles
    grid = (num_tokens, tiles_per_row)

    _convert_req_index_to_global_index_kernel[grid](
        kv_indptr_c,
        kv_indices_c,
        kv_last_page_lens_c,
        token_indices_c,
        out,
        # shapes / constexprs
        BLOCK_SIZE,
        BLOCK_N,
        # strides
        bt_stride0,
        ti_stride0,
        ti_stride1,
        out_stride0,
        out_stride1,
        qo_len,
    )
    return out


@benchmark()
def test_mla(
    ctx_lens,
    batch_size,
    nhead,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
    dtype,
    kvtype,
    page_size,
    varlen,
    decode_qlen,
    max_split_per_batch,
    paged_layout,
    scale_dim,
):
    ret = {}

    out_dtype = torch.bfloat16
    kv_max_sz = (
        65536 * 32
    )  # calculated by rest of mem after weight loaded in frameworks
    num_page = (kv_max_sz + page_size - 1) // page_size

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    kv_len_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    seq_lens_qo = torch.empty(batch_size, dtype=torch.int)
    seq_lens_kv = torch.empty(batch_size, dtype=torch.int)
    kv_block_nums = torch.empty(batch_size, dtype=torch.int)
    kv_last_page_lens = torch.empty(batch_size, dtype=torch.int)
    if varlen:
        for i in range(batch_size):
            # seq_lens_kv[i] = max(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens)
            seq_lens_kv[i] = random.uniform(6, ctx_lens)
            seq_lens_qo[i] = max(
                min(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens), 1
            )
            kv_block_nums[i] = (seq_lens_kv[i] + page_size - 1) // page_size
            if seq_lens_kv[i] % page_size == 0:
                kv_last_page_lens[i] = page_size
            else:
                kv_last_page_lens[i] = seq_lens_kv[i] % page_size
    else:
        seq_lens_kv.fill_(ctx_lens)
        seq_lens_qo.fill_(ctx_lens)
        kv_block_nums.fill_((ctx_lens + page_size - 1) // page_size)
        if ctx_lens % page_size == 0:
            kv_last_page_lens.fill_(page_size)
        else:
            kv_last_page_lens.fill_(ctx_lens % page_size)

    kv_len_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_kv, dim=0)
    kv_indptr[1 : batch_size + 1] = torch.cumsum(kv_block_nums, dim=0)
    print(
        f"seq_lens_kv is {seq_lens_kv}, kv_len_indptr is {kv_len_indptr}, kv_block_nums is {kv_block_nums}, kv_indptr is {kv_indptr}"
    )
    num_page = kv_indptr[-1].item()
    kv_indices = torch.randperm(num_page, dtype=torch.int)
    qo_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)
    max_seqlen_qo = seq_lens_qo.max().item()
    # max_seqlen_kv = seq_lens_kv.max().item()
    # total_qo = qo_indptr[-1].item()
    kv_buffer = torch.randn(
        (num_page, page_size, 1, kv_lora_rank + qk_rope_head_dim),
        dtype=torch.bfloat16,
    )

    kv_nope_scale_factors_fp32 = None
    kv_nope_buffer_fp8 = None
    kv_rope_buffer_bf16 = None

    if paged_layout == "3BUFFER":
        (
            kv_buffer,
            kv_nope_buffer_fp8,
            kv_nope_scale_factors_fp32,
            kv_rope_buffer_bf16,
            _,
        ) = init_3buffer_kv_cache(
            num_page, page_size, kv_lora_rank, qk_rope_head_dim, scale_dim
        )
    # for none absorb (mha)
    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    sm_scale = 1.0 / (qk_head_dim**0.5)

    # us_asm = None
    # if batch_size * ctx_lens * nhead < 32 * 8192 * 16:
    #     us_asm = test_absorb_prefill()
    torch.cuda.empty_cache()
    nhead_kv = 1

    # ############################## absorb: decode
    # seq_lens_qo = torch.randint(1, 5, (batch_size,), dtype=torch.int)
    # if nhead == 16 and decode_qlen != 1:
    #     return
    seq_lens_qo.fill_(decode_qlen)

    max_seqlen_qo = seq_lens_qo.max().item()
    qo_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)
    total_q = qo_indptr[-1].item()
    q = torch.randn((total_q, nhead, qk_head_dim), dtype=torch.bfloat16)

    # troch implementation
    out_ref, lse_ref = torch_mla_extend(
        q,
        kv_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        sm_scale,
        kv_lora_rank,
        qk_rope_head_dim,
        is_causal=True,
        dtype=dtype,
    )

    (
        (work_meta_data_size, work_meta_data_type),
        (work_indptr_size, work_indptr_type),
        (work_info_set_size, work_info_set_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = aiter.get_mla_metadata_info_v1(
        batch_size,
        max_seqlen_qo,
        nhead,
        dtype,
        kvtype,
        is_sparse=True,
        fast_mode=True,
        num_kv_splits=max_split_per_batch,
    )

    # aiter implementation
    # the tensor's meaning please refer aiter/ops/attention.py
    work_meta_data = torch.empty(
        work_meta_data_size, dtype=work_meta_data_type, device="cuda"
    )
    work_indptr = torch.empty(work_indptr_size, dtype=work_indptr_type, device="cuda")
    work_info_set = torch.empty(
        work_info_set_size,
        dtype=work_info_set_type,
        device="cuda",
    )
    reduce_indptr = torch.empty(
        reduce_indptr_size, dtype=reduce_indptr_type, device="cuda"
    )
    reduce_final_map = torch.empty(
        reduce_final_map_size, dtype=reduce_final_map_type, device="cuda"
    )
    reduce_partial_map = torch.empty(
        reduce_partial_map_size, dtype=reduce_partial_map_type, device="cuda"
    )
    topk = 2048
    aiter.get_mla_metadata_v1(
        qo_indptr,
        kv_indptr,
        kv_last_page_lens,
        nhead // nhead_kv,
        nhead_kv,
        True,
        work_meta_data,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        page_size=page_size,
        kv_granularity=max(1, 16 // page_size),
        max_seqlen_qo=1,
        uni_seqlen_qo=1,
        fast_mode=True,
        max_split_per_batch=max_split_per_batch,
        topk=topk,
        dtype_q=dtype,
        dtype_kv=kvtype,
    )
    num_works = work_indptr[-1].item()
    for i in range(num_works):
        print(
            f"work_info_set[{i}, 0]: {work_info_set[i, 0]}, [{i}, 1]: {work_info_set[i, 1]}, [{i}, 2]: {work_info_set[i, 2]}, [{i}, 3]: {work_info_set[i, 3]}, [{i}, 4]: {work_info_set[i, 4]}, [{i}, 5]: {work_info_set[i, 5]}, [{i}, 6]: {work_info_set[i, 6]}"
        )
    # generate kv topk per token & convert indices into per token
    token_indices = generate_topk_kv(kv_len_indptr, decode_qlen, topk)
    # ??????????
    converted_indices = triton_convert_req_index_to_global_index(
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        token_indices,
        decode_qlen,
        BLOCK_SIZE=page_size,
        NUM_TOPK_TOKENS=topk,
        BLOCK_N=128,
    )

    # convert kv indptr perbatch into pertoken and calc ref
    new_qo_indptr, new_kv_indptr, new_kv_last_page_lens, new_indices = (
        sparse_kv_indptr_to_dense(
            kv_len_indptr,
            converted_indices,
            decode_qlen,
            NUM_TOPK_TOKENS=topk,
            page_size=1,
        )
    )
    total_kv = new_kv_indptr[-1].item()  # change into pertoken total_kv
    print(
        f"new_qo_indptr shape is {new_qo_indptr.shape}, new_qo_indptr is {new_qo_indptr}, "
        f"new_kv_indptr shape is {new_kv_indptr.shape}, new_kv_indptr is {new_kv_indptr}, "
        f"new_kv_last_page_lens shape is {new_kv_last_page_lens.shape}, new_kv_last_page_lens is {new_kv_last_page_lens}, "
        f"new_indices shape is {new_indices.shape}, new_indices is {new_indices}"
    )

    out_ref, lse_ref = torch_mla_extend(
        q,
        kv_buffer.reshape(num_page * page_size, 1, nhead_kv, qk_head_dim),
        new_qo_indptr,
        new_kv_indptr,
        new_indices,
        new_kv_last_page_lens,
        sm_scale,
        kv_lora_rank,
        qk_rope_head_dim,
        is_causal=False,
        dtype=out_dtype,
    )

    def test_sparse_mla_bf16():
        kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
        out_asm = torch.empty((total_q, nhead, v_head_dim), dtype=out_dtype).fill_(-1)

        (attn_logits, attn_lse), us_asm_decode = run_perftest(
            aiter.mla.mla_decode_fwd,
            q,
            kv_buffer.view(num_page, page_size, nhead_kv, qk_head_dim),
            out_asm,
            qo_indptr,
            kv_indptr,
            # new_kv_indptr,
            converted_indices.view(-1),
            kv_last_page_lens,
            1,
            page_size,
            nhead_kv,
            sm_scale,
            num_kv_splits=max_split_per_batch,
            work_meta_data=work_meta_data,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
        )

        # print(f"{out_ref.view(total_q, -1)=}")
        # print(f"{out_asm.view(total_q, -1)=}")
        # checkAllclose(logits_ref, attn_logits,
        #               msg=f'attn_logits [golden vs aiter_asm]')
        # checkAllclose(lse_ref, attn_lse, msg="attn_lse    [golden vs aiter_asm]")
        err = checkAllclose(
            out_ref,
            out_asm,
            msg=f"mla_decode-absorb    [golden vs aiter_asm]: {us_asm_decode:>8.2f} us......",
        )
        return err, us_asm_decode

    def test_sparse_mla_fp8():
        if dtype != dtypes.fp8 and nhead == 128:
            aiter.logger.info("don't support this case:\n")
            return None, 1e12

        kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
        out_asm = torch.empty((total_q, nhead, v_head_dim), dtype=out_dtype).fill_(-1)

        q_fp8 = q.to(dtypes.fp8)
        q_scale = torch.ones([1], dtype=torch.float, device="cuda")

        kv_buffer_fp8 = kv_buffer.to(kvtype)
        kv_scale = torch.ones([1], dtype=torch.float, device="cuda")

        out_ref_fp8, lse_ref_fp8 = torch_mla_extend(
            q_fp8 if dtype == dtypes.fp8 else q,
            kv_buffer_fp8,
            new_qo_indptr,
            new_kv_indptr,
            new_indices,
            new_kv_last_page_lens,
            sm_scale,
            kv_lora_rank,
            qk_rope_head_dim,
            dtype=out_dtype,
            is_causal=True,
            q_scale=q_scale,
            kv_scale=kv_scale,
        )

        (attn_logits, attn_lse), us_asm_decode = run_perftest(
            aiter.mla.mla_decode_fwd,
            q_fp8 if dtype == dtypes.fp8 else q,
            kv_buffer_fp8.view(num_page, page_size, nhead_kv, qk_head_dim),
            out_asm,
            qo_indptr,
            kv_indptr,
            converted_indices.view(-1),
            kv_last_page_lens,
            1,
            page_size,
            nhead_kv,
            sm_scale,
            num_kv_splits=max_split_per_batch,
            q_scale=q_scale,
            kv_scale=kv_scale,
            work_meta_data=work_meta_data,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
        )

        # print(f"{out_ref.view(total_q, -1)=}")
        # print(f"{out_asm.view(total_q, -1)=}")
        # checkAllclose(logits_ref, attn_logits,
        #               msg=f'attn_logits [golden vs aiter_asm]')
        # checkAllclose(lse_ref, attn_lse, msg="attn_lse    [golden vs aiter_asm]")
        err = checkAllclose(
            out_ref,
            out_asm,
            msg=f"mla_decode-absorb_fp8    [golden vs aiter_asm]: {us_asm_decode:>8.2f} us......",
        )
        checkAllclose(
            out_ref_fp8,
            out_asm,
            msg=f"mla_decode-absorb_fp8    [golden fp8 vs aiter_asm]: {us_asm_decode:>8.2f} us......",
        )

        cal_diff(out_ref, out_asm, "out", True)
        return err, us_asm_decode

    def test_absorb_decode_3buffer():
        out_asm = torch.empty((total_q, nhead, v_head_dim), dtype=out_dtype).fill_(-1)
        new_kv_nope_buffer_fp8, valid_mask = gather_sparse_kv_buffer(
            kv_nope_buffer_fp8,
            converted_indices,
        )
        new_kv_rope_buffer_bf16, _ = gather_sparse_kv_buffer(
            kv_rope_buffer_bf16,
            converted_indices,
        )
        new_kv_scale_factors_fp32, _ = gather_sparse_kv_buffer(
            kv_nope_scale_factors_fp32,
            converted_indices,
        )
        # print(f"new_kv_nope_buffer_fp8[17,:,:] is {new_kv_nope_buffer_fp8[17,:,:]}")
        # convert to bytes
        nope_bytes = new_kv_nope_buffer_fp8.view(torch.uint8)
        scale_bytes = new_kv_scale_factors_fp32.view(torch.uint8)
        rope_bytes = new_kv_rope_buffer_bf16.view(torch.uint8)

        kv_buffer_bytes = torch.cat(
            [nope_bytes.flatten(1), scale_bytes.flatten(1), rope_bytes.flatten(1)],
            dim=-1,
        )

        dense_kv_indptr, dense_kv_last_page_lens, dense_kv_indices = (
            get_dense_kv_indptr_and_indices(
                converted_indices,
                new_kv_indptr,
                page_size,
            )
        )
        print(f"dense_kv_last_page_lens is {dense_kv_last_page_lens}")
        out_ref_fp8, lse_ref_fp8 = torch_mla_extend_3buffer(
            q,
            kv_buffer_bytes,
            new_qo_indptr,
            dense_kv_indptr,
            dense_kv_indices,
            dense_kv_last_page_lens,
            page_size,
            nhead_kv,
            sm_scale,
            kv_lora_rank,
            qk_rope_head_dim,
            dtype=out_dtype,
            is_causal=True,
            scale_dim=scale_dim,
            valid_mask=valid_mask,
        )

        checkAllclose(
            out_ref,
            out_ref_fp8,
            msg="mla_decode-absorb_fp8    [golden fp8 vs golden]:......",
        )

        (attn_logits, attn_lse), us_asm_decode = run_perftest(
            aiter.mla.mla_decode_fwd,
            q,
            kv_buffer_bytes,
            out_asm,
            new_qo_indptr,
            dense_kv_indptr,
            dense_kv_indices,
            dense_kv_last_page_lens,
            max_seqlen_qo,
            page_size,
            nhead_kv,
            sm_scale,
            num_kv_splits=max_split_per_batch,
            work_meta_data=work_meta_data,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
        )

        err = checkAllclose(
            out_ref,
            out_asm,
            msg=f"mla_decode-absorb_fp8    [golden vs aiter_asm]: {us_asm_decode:>8.2f} us......",
        )
        checkAllclose(
            out_ref_fp8,
            out_asm,
            msg=f"mla_decode-absorb_fp8    [golden fp8 vs aiter_asm]: {us_asm_decode:>8.2f} us......",
        )
        cal_diff(out_ref, out_asm, "out", True)
        return err, us_asm_decode

    err = None
    us_asm_decode = 1e12
    if paged_layout == "3BUFFER":
        err, us_asm_decode = test_absorb_decode_3buffer()
    elif dtype == torch.bfloat16:
        err, us_asm_decode = test_sparse_mla_bf16()
    elif kvtype == dtypes.fp8:
        err, us_asm_decode = test_sparse_mla_fp8()
    ret["decode:err"] = err
    ret["decode:asm_576"] = us_asm_decode

    flops = total_kv * nhead * (qk_head_dim + v_head_dim) * 2
    bytes = (
        total_kv * nhead_kv * qk_head_dim * (torch.finfo(kvtype).bits // 8)
        + total_q * nhead * qk_head_dim * (torch.finfo(dtype).bits // 8)
        + total_q * nhead * v_head_dim * (torch.finfo(out_dtype).bits // 8)
    )

    ret["decode:flops"] = flops
    ret["decode:bytes"] = bytes
    ret["decode:TFLOPS"] = flops / us_asm_decode / 1e6
    ret["decode:TB/s"] = bytes / us_asm_decode / 1e6

    return ret


kv_lora_rank = 512
qk_nope_head_dim = 128
qk_rope_head_dim = 64
v_head_dim = 128
block_size = 1
list_dtype = ["bf16", "fp8"]
l_kv_dtype = ["bf16", "fp8"]
list_nhead = [(16, 2), (48, 1), (128, 2)]
# list_nhead = [(16, 1), (16, 2), (16, 4), (16, 3)]
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-k",
    "--kv_lora_rank",
    type=int,
    default=512,
    help="""kv lora rank.
    e.g.: -k 512""",
)
parser.add_argument(
    "-qn",
    "--qk_nope_head_dim",
    type=int,
    default=128,
    help="""qk nope head dim.
    e.g.: -qn 512""",
)
parser.add_argument(
    "-qr",
    "--qk_rope_head_dim",
    type=int,
    default=64,
    help="""qk rope head dim.
    e.g.: -qr 64""",
)
parser.add_argument(
    "-vh",
    "--v_head_dim",
    type=int,
    default=512,
    help="""v head dim.
    e.g.: -vh 512""",
)
parser.add_argument(
    "-blk",
    "--block_size",
    type=int,
    default=1,
    help="""Block size.
    e.g.: -blk 1""",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=["bf16", "fp8"],
    nargs="*",
    default=["bf16", "fp8"],
    help="""Data type of Q.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-kvd",
    "--kv_dtype",
    type=str,
    choices=["bf16", "fp8"],
    nargs="*",
    default=["bf16", "fp8"],
    help="""Data type of KV.
    e.g.: -kvd bf16""",
)
parser.add_argument(
    "-c",
    "--ctxLen",
    type=int,
    nargs="*",
    default=[21, 64, 256, 512, 1200, 2048, 3200, 5200, 8192],
    help="""Context length.
    e.g.: -c 21""",
)
parser.add_argument(
    "-b",
    "--batchSize",
    type=int,
    nargs="*",
    default=[1, 3, 5, 16, 32, 64, 128, 256],
    help="""Batch size.
    e.g.: -b 16""",
)
parser.add_argument(
    "-n",
    "--nhead",
    type=dtypes.str2tuple,
    nargs="?",
    const=None,
    default=None,
    help="""Number of heads.
    e.g.: -n 16,1""",
)
parser.add_argument(
    "-ms",
    "--max_split_per_batch",
    type=int,
    nargs="*",
    default=[32],
    help="""kv seqlens max split num for per batch.
    e.g.: -ms 32""",
)
parser.add_argument(
    "--varlen",
    action="store_true",
    help="""variable kv seqlens per batch. Default: False.
    --varlen # True""",
)
parser.add_argument(
    "-pl",
    "--paged_layout",
    type=str,
    choices=["LEGACY", "3BUFFER"],
    default="LEGACY",
    help="""kv paged layout for persistent mode.
        LEGACY: kv buffer is common buffer with nope and rope parts.
        3BUFFER: kv buffer is 3-buffer with nope, kv_scale and rope parts.
        e.g.: -pl 3BUFFER""",
)
parser.add_argument(
    "-sd",
    "--scale_dim",
    type=int,
    default=4,
    help="""scale dim.
    e.g.: -sd 4""",
)

args = parser.parse_args()
list_dtype = [dtypes.d_dtypes[key] for key in args.dtype]
l_kv_dtype = [dtypes.d_dtypes[key] for key in args.kv_dtype]
if args.nhead is not None:
    list_nhead = [args.nhead]

for nhead, decode_qlen in list_nhead:
    df = []
    for dtype, kvtype, ctx_len, batch_size, max_split_per_batch in itertools.product(
        list_dtype, l_kv_dtype, args.ctxLen, args.batchSize, args.max_split_per_batch
    ):
        if check_support(dtype, kvtype, nhead):
            ret = test_mla(
                ctx_len,
                batch_size,
                nhead,
                args.kv_lora_rank,
                args.qk_nope_head_dim,
                args.qk_rope_head_dim,
                args.v_head_dim,
                dtype,
                kvtype,
                args.block_size,
                varlen=args.varlen,
                decode_qlen=decode_qlen,
                max_split_per_batch=max_split_per_batch,
                paged_layout=args.paged_layout,
                scale_dim=args.scale_dim,
            )
            df.append(ret)
    df = pd.DataFrame(df)
    # df.to_csv(f"mla_nhead{nhead}decode_qlen{decode_qlen}.csv")
    df_md = df.to_markdown(index=False)
    aiter.logger.info("mla_sparse summary (markdown):\n%s", df_md)
