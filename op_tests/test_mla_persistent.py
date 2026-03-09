# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter.test_common import checkAllclose, benchmark, run_perftest
from aiter import dtypes

# from aiter.jit.utils.chip_info import get_cu_num, get_gfx
import random
import itertools
import argparse
import pandas as pd
import math

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)

# current supported case in ps decode MLA: mtp == 0, 1, 2, 3 (decode_qlen = 1, 2, 3, 4)
# qdtype bf16, kdtype bf16: nhead16
# qdtype fp8, kdtype fp8: nhead16, nhead128
# qdtype fp8, kdtype fp8: nhead32, max_seqlen_qo=4
# qdtype fp8, kdtype bf16: nhead16


def check_support(dtype, kv_dtype, nhead):
    if dtype == dtypes.fp8 and kv_dtype == dtypes.bf16:
        return False
    if dtype == dtypes.bf16 and nhead == 32:
        return False
    return True


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

    kv_nope_buffer_fp32 = torch.randn(
        (num_page, page_size, 1, kv_lora_rank), dtype=torch.float32
    )
    kv_rope_buffer_bf16 = torch.randn(
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
    causal_diagonal=None,
):
    if is_fp8_q and q_scale is not None:
        scale *= q_scale
    if is_fp8_kvc and kv_scale is not None:
        scale *= kv_scale
    attn_weights = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * scale

    if is_causal:
        s_q = query.shape[0]
        s_k = key.shape[0]
        diagonal = causal_diagonal if causal_diagonal is not None else s_k - s_q
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=diagonal)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weights += attn_bias

    lse = attn_weights.logsumexp(dim=-1)
    m = attn_weights.max(-1).values
    attn_weights_exp = torch.exp(attn_weights - m.unsqueeze(-1))
    l = attn_weights_exp.sum(-1)  # noqa: E741
    if is_fp8_q:
        attn_weights_fp8 = attn_weights_exp.to(dtypes.fp8)
        attn_weights_exp = attn_weights_fp8.to(torch.float)

    out = torch.einsum("hqk,khd->qhd", attn_weights_exp.float(), value.float())
    out = out / l.transpose(0, 1).unsqueeze(-1)
    if is_fp8_kvc and kv_scale is not None:
        out *= kv_scale
    return out.to(dtype), lse


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
    )


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
        )
        os.append(o)
        lses.append(lse)
    o = torch.concat(os)
    lse = torch.concat(lses).transpose(0, 1)
    return o, lse


def torch_mla_extend_split_kv(
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
    work_meta_data,
    work_info_set,
    work_indptr,
    max_seqlen_q,
    is_causal=True,
    q_scale=None,
    kv_scale=None,
):

    num_page, page_size, nhead_kv, _ = kvc_cache.shape
    total_q, nheads, _ = q.shape
    dev = kvc_cache.device
    is_fp8_q = q.dtype == dtypes.fp8
    is_fp8_kvc = kvc_cache.dtype == dtypes.fp8

    if is_fp8_q:
        q = q.to(torch.float)

    if is_fp8_kvc:
        kvc_cache = kvc_cache.to(torch.float)

    kvc = torch.index_select(kvc_cache, 0, kv_indices)
    kvs = torch.tensor_split(kvc, kv_indptr.tolist()[1:])
    num_works = work_indptr[-1].item()
    partial_os = []
    partial_lses = []
    final_out = torch.empty(total_q, nheads, kv_lora_rank, dtype=dtype, device=dev)
    final_lse = torch.empty(total_q, nheads, dtype=torch.float32, device=dev)

    io_transformed = False
    q_ratio = 1
    if (
        nheads == 16
        or (nheads == 128 and is_fp8_q and is_fp8_kvc)
        or (
            # get_gfx() == "gfx950"
            nheads == 32
            and is_fp8_q
            and is_fp8_kvc
            and max_seqlen_q == 4
        )
    ):
        # Natively support cases
        pass
    elif nheads in range(32, 128 + 1, 16):
        # we use nhead=16 to simulate such cases by customized metadata
        # metadata also views qo's tensor as shape (total_s * (nhead // 16), 16, ...)
        q_ratio = nheads // 16
        total_s = total_q * q_ratio
        nheads = 16
        q = q.view(total_s, nheads, -1)
        final_out = final_out.view(total_s, nheads, -1)
        final_lse = final_lse.view(total_s, nheads)
        io_transformed = True

    for work_idx in range(num_works):
        row = work_info_set[work_idx]
        batch_idx = row[0].item() // q_ratio
        partial_qo_loc = row[1].item()
        qo_start = row[2].item()
        qo_end = row[3].item()
        kv_start = row[4].item()
        kv_end = row[5].item()
        kv_offset = row[6].item()
        print(
            f"batch_idx: {batch_idx}, partial_qo_loc: {partial_qo_loc}, qo_start: {qo_start}, qo_end: {qo_end}, kv_start: {kv_start}, kv_end: {kv_end}, kv_offset: {kv_offset}"
        )
        cur_num_page = kvs[batch_idx].shape[0]
        cur_real_kv_seq_len = (
            cur_num_page - 1
        ) * page_size + kv_last_page_lens.tolist()[batch_idx]
        real_sum_kv_seq_len = (
            kv_indptr.tolist()[batch_idx] * page_size + cur_real_kv_seq_len
        )
        # slice_k = kvs[batch_idx].flatten(0, 1)[(kv_start * page_size):(real_kv_seq_len - kv_offset)]
        print(
            f"slice_k start: {kv_start * page_size}, end: {real_sum_kv_seq_len - kv_offset}"
        )
        print(f"slice_q start: {qo_start}, end: {qo_end}")

        slice_k = kvc.flatten(0, 1)[
            kv_start * page_size : real_sum_kv_seq_len - kv_offset
        ]
        slice_q = q[qo_start:qo_end]

        v, _ = torch.split(slice_k, [kv_lora_rank, qk_rope_head_dim], dim=-1)
        print(f"slice_k shape: {slice_k.shape}")
        print(f"slice_q shape: {slice_q.shape}")
        print(f"v shape: {v.shape}")
        if partial_qo_loc != -1:
            out_dtype = torch.float32
        else:
            out_dtype = dtype

        # Compute correct causal diagonal for this split chunk
        # In the original full-batch attention, q[i] attends to k[j] if j <= i + (total_kv - total_q)
        # For a split chunk, we need to account for the position offsets of Q and KV within the batch
        causal_diagonal = None
        if is_causal:
            q_local_start = qo_start - qo_indptr[batch_idx].item()
            kv_local_start = (kv_start - kv_indptr[batch_idx].item()) * page_size
            total_q_len = qo_indptr[batch_idx + 1].item() - qo_indptr[batch_idx].item()
            causal_diagonal = (
                q_local_start - kv_local_start + cur_real_kv_seq_len - total_q_len
            )

        o, lse = ref_masked_attention(
            slice_q,
            slice_k,
            v,
            sm_scale,
            out_dtype,
            is_causal=is_causal,
            is_fp8_q=is_fp8_q,
            is_fp8_kvc=is_fp8_kvc,
            q_scale=q_scale,
            kv_scale=kv_scale,
            causal_diagonal=causal_diagonal,
        )

        if partial_qo_loc == -1:
            final_out[qo_start:qo_end, :, :] = o
            final_lse[qo_start:qo_end, :] = lse.transpose(0, 1)  # [seq_q, num_heads]
        else:
            partial_os.append(o)
            partial_lses.append(lse)

    partial_o = (
        torch.concat(partial_os)
        if partial_os
        else torch.empty(0, nheads, qk_rope_head_dim, dtype=torch.float32, device=dev)
    )
    partial_lse = (
        torch.concat(partial_lses, dim=1).transpose(0, 1)
        if partial_lses
        else torch.empty(0, nheads, dtype=torch.float32, device=dev)
    )
    partial_o = torch.where(
        torch.isnan(partial_o), torch.zeros_like(partial_o), partial_o
    )

    partial_lse = torch.where(
        torch.isnan(partial_lse),
        torch.full_like(partial_lse, float("-inf")),
        partial_lse,
    )

    return partial_o, partial_lse, final_out, final_lse, io_transformed


def torch_mla_reduce_v1(
    partial_output: torch.Tensor,  # [max(reduce_partial_map)+s, h, dv]
    partial_lse: torch.Tensor,  # [max(reduce_partial_map)+s, h]
    reduce_indptr: torch.Tensor,  # [#work + 1]
    reduce_final_map: torch.Tensor,  # [#work, 2] or None
    reduce_partial_map: torch.Tensor,  # [reduce_indptr[-1]]
    max_seqlen_q: int,
    final_output: torch.Tensor,  # [bs, h, dv]
    final_lse: torch.Tensor,  # [bs, h] or None
) -> None:
    device = partial_output.device
    dtype = partial_output.dtype

    # Check input types
    assert partial_output.dtype == torch.float32, "partial_output must be float32"
    assert partial_lse.dtype == torch.float32, "partial_lse must be float32"

    num_reduce_tile = reduce_indptr.shape[0] - 1
    num_heads = partial_output.shape[1]
    head_dim = final_output.shape[2]

    if num_reduce_tile == 0:
        return

    # Process each reduce tile
    for tile_idx in range(num_reduce_tile):
        reduce_tile_start = reduce_indptr[tile_idx].item()
        reduce_tile_end = reduce_indptr[tile_idx + 1].item()

        if reduce_tile_start == reduce_tile_end:
            continue

        num_splits = reduce_tile_end - reduce_tile_start

        # Get reduce partial map for this tile
        tile_reduce_partial_map = reduce_partial_map[reduce_tile_start:reduce_tile_end]

        # Determine final output location
        if reduce_final_map is not None:
            # Use provided final map
            q_start = reduce_final_map[tile_idx, 0].item()
            q_end = reduce_final_map[tile_idx, 1].item()
        else:
            # Compute from reduce_partial_map
            if num_splits >= 2:
                reduce_partial_map_0 = tile_reduce_partial_map[0].item()
                reduce_partial_map_1 = tile_reduce_partial_map[1].item()
                qo_len = reduce_partial_map_1 - reduce_partial_map_0
                q_start = tile_idx * qo_len
                q_end = (tile_idx + 1) * qo_len
            else:
                # Fallback: use max_seqlen_q
                q_start = tile_idx * max_seqlen_q
                q_end = (tile_idx + 1) * max_seqlen_q

        # Process each sequence position and head
        for seq_idx in range(q_start, q_end):
            for head_idx in range(num_heads):
                local_seq_idx = seq_idx - q_start
                partial_lses = []
                partial_outputs = []

                for split_idx in range(num_splits):
                    partial_qo_loc = tile_reduce_partial_map[split_idx].item()

                    partial_buffer_idx = partial_qo_loc + local_seq_idx

                    # Get LSE value
                    if (
                        partial_buffer_idx < partial_lse.shape[0]
                        and head_idx < partial_lse.shape[1]
                    ):
                        lse_val = partial_lse[partial_buffer_idx, head_idx].item()
                        # Handle NaN
                        if math.isnan(lse_val):
                            lse_val = float("-inf")
                    else:
                        lse_val = float("-inf")

                    # Get output values
                    if (
                        partial_buffer_idx < partial_output.shape[0]
                        and head_idx < partial_output.shape[1]
                    ):
                        out_vals = partial_output[
                            partial_buffer_idx, head_idx, :
                        ].clone()
                        # Handle NaN
                        out_vals = torch.where(
                            torch.isnan(out_vals), torch.zeros_like(out_vals), out_vals
                        )
                    else:
                        out_vals = torch.zeros(head_dim, dtype=dtype, device=device)

                    partial_lses.append(lse_val)
                    partial_outputs.append(out_vals)

                if len(partial_lses) == 0:
                    continue

                # Numerically stable online reduction (matching C++ implementation)
                # Start with first split
                if len(partial_lses) == 0:
                    continue

                # Initialize with first split
                max_lse = partial_lses[0]
                reg_out = partial_outputs[0].clone()  # [head_dim]
                sum_e_lse = 1.0

                # Online update for remaining splits
                for split_idx in range(1, num_splits):
                    lse = partial_lses[split_idx]
                    oaccu = partial_outputs[split_idx]  # [head_dim]

                    # Update max LSE
                    new_max_lse = max(max_lse, lse)

                    # Compute scales
                    old_scale = math.exp(max_lse - new_max_lse)
                    new_scale = math.exp(lse - new_max_lse)

                    # Update output: old_scale * reg_out + new_scale * oaccu
                    reg_out = old_scale * reg_out + new_scale * oaccu

                    # Update sum
                    max_lse = new_max_lse
                    sum_e_lse = sum_e_lse * old_scale + new_scale

                # Normalize by sum_e_lse
                if sum_e_lse > 0 and not math.isnan(sum_e_lse):
                    reg_out = reg_out / sum_e_lse
                else:
                    # Handle edge case
                    reg_out = torch.zeros_like(reg_out)

                # Write to final output
                final_output[seq_idx, head_idx, :] = reg_out.to(final_output.dtype)

                # Compute and write final LSE if needed
                if final_lse is not None:
                    if sum_e_lse > 0 and not math.isnan(sum_e_lse):
                        final_lse_val = max_lse + math.log(sum_e_lse)
                    else:
                        final_lse_val = float("inf")
                    final_lse[seq_idx, head_idx] = final_lse_val


def torch_mla_split_kv_and_reduce(
    q,
    kv_cache,
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    sm_scale,
    kv_lora_rank,
    qk_rope_head_dim,
    dtype,
    work_meta_data,
    work_info_set,
    work_indptr,
    reduce_indptr,
    reduce_final_map,
    reduce_partial_map,
    max_seqlen_q,
    is_causal=True,
    q_scale=None,
    kv_scale=None,
):
    total_q, nhead, _ = q.shape
    partial_out, partial_lse, split_out, split_lse, io_transformed = (
        torch_mla_extend_split_kv(
            q,
            kv_cache,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            sm_scale,
            kv_lora_rank,
            qk_rope_head_dim,
            dtype=dtype,
            work_meta_data=work_meta_data,
            work_info_set=work_info_set,
            work_indptr=work_indptr,
            max_seqlen_q=max_seqlen_q,
            is_causal=is_causal,
            q_scale=q_scale,
            kv_scale=kv_scale,
        )
    )

    torch_mla_reduce_v1(
        partial_out,
        partial_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        max_seqlen_q,
        split_out,
        split_lse,
    )

    if io_transformed:
        # import pdb; pdb.set_trace()
        # partial_out = partial_out.reshape(-1, nhead, kv_lora_rank)
        # partial_lse = partial_lse.reshape(-1, nhead)
        split_out = split_out.reshape(total_q, nhead, kv_lora_rank)
        split_lse = split_lse.reshape(total_q, nhead)

    return partial_out, partial_lse, split_out, split_lse


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
    non_persistent_mode,
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
    seq_lens_qo = torch.empty(batch_size, dtype=torch.int)
    seq_lens_kv = torch.empty(batch_size, dtype=torch.int)
    kv_block_nums = torch.empty(batch_size, dtype=torch.int)
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
    if varlen:
        for i in range(batch_size):
            # seq_lens_kv[i] = max(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens)
            seq_lens_kv[i] = random.uniform(5, ctx_lens)
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

    kv_indptr[1 : batch_size + 1] = torch.cumsum(kv_block_nums, dim=0)
    num_page = kv_indptr[-1].item()
    kv_indices = torch.randperm(num_page, dtype=torch.int)
    qo_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)
    max_seqlen_qo = seq_lens_qo.max().item()
    # max_seqlen_kv = seq_lens_kv.max().item()
    # total_qo = qo_indptr[-1].item()
    total_kv = seq_lens_kv.sum().item()

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
    # print(f"q shape: {q.shape}")
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
        dtype=out_dtype,
    )

    # It is necessary to limit the size of the tensor in the DP mode
    # so reduce the split_num in the DP mode.
    if nhead >= 128:
        gpu = torch.cuda.current_device()
        device_properties = torch.cuda.get_device_properties(gpu)
        cu_num = device_properties.multi_processor_count
        max_split_per_batch = min(
            (cu_num + batch_size - 1) // batch_size, max_split_per_batch
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
        is_sparse=False,
        fast_mode=True if not non_persistent_mode else False,
        num_kv_splits=max_split_per_batch,
        intra_batch_mode=non_persistent_mode,
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

    aiter.get_mla_metadata_v1(
        qo_indptr,
        kv_indptr,
        kv_last_page_lens,
        nhead // nhead_kv,
        nhead_kv,
        False,
        work_meta_data,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        kv_granularity=max(page_size, 16),  # for qh32 kv split is disabled
        max_seqlen_qo=int(max_seqlen_qo),
        uni_seqlen_qo=decode_qlen,
        fast_mode=True if not non_persistent_mode else False,
        max_split_per_batch=max_split_per_batch,
        intra_batch_mode=non_persistent_mode,
        dtype_q=dtype,
        dtype_kv=kvtype,
    )

    def test_absorb_decode_bf16_fp8():
        out_asm = torch.empty((total_q, nhead, v_head_dim), dtype=out_dtype).fill_(-1)
        kv_buffer_fp8 = kv_buffer.to(kvtype)
        kv_scale = torch.ones([1], dtype=torch.float, device="cuda")

        out_ref_fp8, lse_ref_fp8 = torch_mla_extend(
            q,
            kv_buffer_fp8,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            sm_scale,
            kv_lora_rank,
            qk_rope_head_dim,
            dtype=out_dtype,
            is_causal=True,
            q_scale=None,
            kv_scale=kv_scale,
        )

        (attn_logits, attn_lse), us_asm_decode = run_perftest(
            aiter.mla.mla_decode_fwd,
            q,
            kv_buffer_fp8.view(num_page, page_size, nhead_kv, qk_head_dim),
            out_asm,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
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
            intra_batch_mode=non_persistent_mode,
            kv_scale=kv_scale,
        )

        err = checkAllclose(
            out_ref,
            out_asm,
            msg=f"mla_decode-absorb    [golden vs aiter_asm]: {us_asm_decode:>8.2f} us......",
        )
        checkAllclose(
            out_ref_fp8,
            out_asm,
            msg=f"mla_decode-absorb_fp8    [golden fp8 vs aiter_asm]: {us_asm_decode:>8.2f} us......",
        )

        if not non_persistent_mode:
            partial_out_ref, partial_lse_ref, split_out_ref, split_lse_ref = (
                torch_mla_split_kv_and_reduce(
                    q,
                    kv_buffer_fp8,
                    qo_indptr,
                    kv_indptr,
                    kv_indices,
                    kv_last_page_lens,
                    sm_scale,
                    kv_lora_rank,
                    qk_rope_head_dim,
                    dtype=out_dtype,
                    work_meta_data=work_meta_data,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    max_seqlen_q=max_seqlen_qo,
                    is_causal=True,
                    q_scale=None,
                    kv_scale=kv_scale,
                )
            )

            checkAllclose(
                split_out_ref,
                out_asm,
                msg=f"mla_decode-absorb_fp8    [golden fp8 split_out_ref vs aiter_asm]: {us_asm_decode:>8.2f} us......",
            )
            if partial_out_ref.shape[0] > 0:
                checkAllclose(
                    partial_out_ref,
                    attn_logits[: partial_out_ref.shape[0]].flatten(0,1),
                    msg=f"mla_decode-absorb_fp8    [partial_out_ref vs attn_logits]: {us_asm_decode:>8.2f} us......",
                )
        return err, us_asm_decode

    def test_absorb_decode_bf16():
        kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
        out_asm = torch.empty((total_q, nhead, v_head_dim), dtype=out_dtype).fill_(-1)
        (attn_logits, attn_lse), us_asm_decode = run_perftest(
            aiter.mla.mla_decode_fwd,
            q,
            kv_buffer.view(num_page, page_size, nhead_kv, qk_head_dim),
            out_asm,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
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
            intra_batch_mode=non_persistent_mode,
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

        if not non_persistent_mode:
            partial_out_ref, partial_lse_ref, split_out_ref, split_lse_ref = (
                torch_mla_split_kv_and_reduce(
                    q,
                    kv_buffer,
                    qo_indptr,
                    kv_indptr,
                    kv_indices,
                    kv_last_page_lens,
                    sm_scale,
                    kv_lora_rank,
                    qk_rope_head_dim,
                    dtype=out_dtype,
                    work_meta_data=work_meta_data,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    max_seqlen_q=max_seqlen_qo,
                    is_causal=True,
                )
            )

            checkAllclose(
                split_out_ref,
                out_asm,
                msg=f"mla_decode-absorb_fp8    [golden fp8 split_out_ref vs aiter_asm]: {us_asm_decode:>8.2f} us......",
            )
            if partial_out_ref.shape[0] > 0:
                checkAllclose(
                    partial_out_ref,
                    attn_logits[: partial_out_ref.shape[0]].flatten(0,1),
                    msg=f"mla_decode-absorb_fp8    [partial_out_ref vs attn_logits]: {us_asm_decode:>8.2f} us......",
                )
        return err, us_asm_decode

    def test_absorb_decode_fp8():
        # print(f"work_indptr: {work_indptr}")
        # print(f"work_info_set shape: {work_info_set.shape}")
        num_works = work_indptr[-1].item()
        for i in range(num_works):
            print(
                f"work_info_set[{i}, 0]: {work_info_set[i, 0]}, [{i}, 1]: {work_info_set[i, 1]}, [{i}, 2]: {work_info_set[i, 2]}, [{i}, 3]: {work_info_set[i, 3]}, [{i}, 4]: {work_info_set[i, 4]}, [{i}, 5]: {work_info_set[i, 5]}, [{i}, 6]: {work_info_set[i, 6]}"
            )        
        kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
        out_asm = torch.empty((total_q, nhead, v_head_dim), dtype=out_dtype).fill_(-1)

        q_fp8 = q.to(dtypes.fp8)
        q_scale = torch.ones([1], dtype=torch.float, device="cuda")

        kv_buffer_fp8 = kv_buffer.to(dtypes.fp8)
        kv_scale = torch.ones([1], dtype=torch.float, device="cuda")

        out_ref_fp8, lse_ref_fp8 = torch_mla_extend(
            q_fp8 if dtype == dtypes.fp8 else q,
            kv_buffer_fp8,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            sm_scale,
            kv_lora_rank,
            qk_rope_head_dim,
            dtype=out_dtype,
            is_causal=True,
            q_scale=None,
            kv_scale=kv_scale,
        )

        (attn_logits, attn_lse), us_asm_decode = run_perftest(
            aiter.mla.mla_decode_fwd,
            q_fp8 if dtype == dtypes.fp8 else q,
            kv_buffer_fp8.view(num_page, page_size, nhead_kv, qk_head_dim),
            out_asm,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            max_seqlen_qo,
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
            intra_batch_mode=non_persistent_mode,
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

        err = checkAllclose(
            out_ref_fp8,
            out_asm,
            msg=f"mla_decode-absorb_fp8    [golden fp8 vs aiter_asm]: {us_asm_decode:>8.2f} us......",
        )

        if not non_persistent_mode:
            partial_out_ref, partial_lse_ref, split_out_ref, split_lse_ref = (
                torch_mla_split_kv_and_reduce(
                    q_fp8 if dtype == dtypes.fp8 else q,
                    kv_buffer_fp8,
                    qo_indptr,
                    kv_indptr,
                    kv_indices,
                    kv_last_page_lens,
                    sm_scale,
                    kv_lora_rank,
                    qk_rope_head_dim,
                    dtype=out_dtype,
                    work_meta_data=work_meta_data,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    max_seqlen_q=max_seqlen_qo,
                    is_causal=True,
                    q_scale=q_scale,
                    kv_scale=kv_scale,
                )
            )

            checkAllclose(
                split_out_ref,
                out_asm,
                msg=f"mla_decode-absorb_fp8    [golden fp8 split_out_ref vs aiter_asm]: {us_asm_decode:>8.2f} us......",
            )

            if partial_out_ref.shape[0] > 0:
                checkAllclose(
                    partial_out_ref,
                    attn_logits[: partial_out_ref.shape[0]].flatten(0,1),
                    msg=f"mla_decode-absorb_fp8    [partial_out_ref vs attn_logits]: {us_asm_decode:>8.2f} us......",
                )

        cal_diff(out_ref, out_asm, "out", True)
        return err, us_asm_decode

    def test_absorb_decode_3buffer():

        out_asm = torch.empty((total_q, nhead, v_head_dim), dtype=out_dtype).fill_(-1)

        # convert to bytes
        nope_bytes = kv_nope_buffer_fp8.view(torch.uint8)
        scale_bytes = kv_nope_scale_factors_fp32.view(torch.uint8)
        rope_bytes = kv_rope_buffer_bf16.view(torch.uint8)
        kv_buffer_bytes = torch.cat(
            [nope_bytes.flatten(1), scale_bytes.flatten(1), rope_bytes.flatten(1)],
            dim=-1,
        )

        out_ref_fp8, lse_ref_fp8 = torch_mla_extend_3buffer(
            q,
            kv_buffer_bytes,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            page_size,
            nhead_kv,
            sm_scale,
            kv_lora_rank,
            qk_rope_head_dim,
            dtype=out_dtype,
            is_causal=True,
            scale_dim=scale_dim,
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
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
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
            intra_batch_mode=non_persistent_mode,
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

    if paged_layout == "3BUFFER" and not non_persistent_mode:
        err, us_asm_decode = test_absorb_decode_3buffer()
    elif dtype == torch.bfloat16 and kvtype == dtypes.fp8:
        err, us_asm_decode = test_absorb_decode_bf16_fp8()
    elif dtype == torch.bfloat16:
        err, us_asm_decode = test_absorb_decode_bf16()
    elif kvtype == dtypes.fp8:
        err, us_asm_decode = test_absorb_decode_fp8()

    ret["decode:err"] = err
    ret["decode:asm_576"] = us_asm_decode

    flops = decode_qlen * total_kv * nhead * (qk_head_dim + v_head_dim) * 2
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
    default=512,
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
    type=dtypes.str2Dtype,
    choices=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp8"]],
    nargs="*",
    default="bf16,fp8",
    metavar="{bf16, fp8}",
    help="""Data type of Q.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-kvd",
    "--kv_dtype",
    type=dtypes.str2Dtype,
    choices=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp8"]],
    nargs="*",
    metavar="{bf16, fp8}",
    default="bf16,fp8",
    help="""Data type of KV.
    e.g.: -kvd bf16""",
)
parser.add_argument(
    "-c",
    "--ctxLen",
    type=int,
    nargs="*",
    default=[21, 64, 256, 512, 1200, 3200, 5200, 8192],
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
    nargs="*",
    const=None,
    default=[(48, 1), (128, 2)],
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
    "-nps",
    "--non_persistent_mode",
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
for nhead, decode_qlen in args.nhead:
    df = []
    for dtype, kvtype, ctx_len, batch_size, max_split_per_batch in itertools.product(
        args.dtype, args.kv_dtype, args.ctxLen, args.batchSize, args.max_split_per_batch
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
                non_persistent_mode=args.non_persistent_mode,
                paged_layout=args.paged_layout,
                scale_dim=args.scale_dim,
            )
            df.append(ret)
    df = pd.DataFrame(df)
    # df.to_csv(f"mla_nhead{nhead}decode_qlen{decode_qlen}.csv")
    df_md = df.to_markdown(index=False)
    aiter.logger.info("mla_persistent summary (markdown):\n%s", df_md)
