# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import math

import torch
import triton

from aiter.ops.triton._triton_kernels.attention.pa_decode import (
    _get_dispatch_config,
    _get_launch_config,
    _paged_attn_decode_v1_w_dot_kernel,
    _paged_attn_decode_v1_w_dot_kernel_per_token_quant,
    _paged_attn_decode_v1_wo_dot_kernel,
    _paged_attn_decode_v1_wo_dot_kernel_per_token_quant,
    _paged_attn_decode_v2_w_dot_kernel,
    _paged_attn_decode_v2_w_dot_kernel_per_token_quant,
    _paged_attn_decode_v2_w_dot_reduce_kernel,
    _paged_attn_decode_v2_w_dot_reduce_kernel_per_token_quant,
    _paged_attn_decode_v2_wo_dot_kernel,
    _paged_attn_decode_v2_wo_dot_kernel_per_token_quant,
    _paged_attn_decode_v2_wo_dot_reduce_kernel,
    _paged_attn_decode_v2_wo_dot_reduce_kernel_per_token_quant,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

try:
    from aiter.ops.triton._gluon_kernels.gfx1250.attention.pa_decode import (
        _pa_decode_gqa_tdm_kernel,
    )
except ImportError:
    _pa_decode_gqa_tdm_kernel = None

_LOGGER = AiterTritonLogger()

# This code is derived from sglang and FLASHNN projects
# https://github.com/AlibabaPAI/FLASHNN/blob/main/flashnn/triton_kernels/paged_attn.py

_SEQ_PARTITION_SIZE = 1024  # HIP

_KV_DTYPE_NAMES = {torch.bfloat16: "bf16", torch.float16: "fp16"}

_TDM_HEAD_SIZES = (56, 64, 128)
_TDM_PAGE_SIZE = 128
_TDM_NO_SPLIT = 1 << 30


def _select_tdm_partition_size(
    num_programs: int, max_seq_len: int, dispatch: dict
) -> int:
    """Tokens per program of the Gluon kernel, or `_TDM_NO_SPLIT` to keep a sequence whole."""
    target = dispatch.get("tdm_split_target_programs")
    if target is None or num_programs >= target:
        return _TDM_NO_SPLIT
    sizes = [
        s
        for s in sorted(dispatch["tdm_split_sizes"], reverse=True)
        if s < max_seq_len
        and triton.cdiv(max_seq_len, s) <= dispatch["tdm_split_max_partitions"]
    ]
    # Largest size that reaches the target; else the smallest allowed.
    for s in sizes:
        if num_programs * triton.cdiv(max_seq_len, s) >= target:
            return s
    return sizes[-1] if sizes else _TDM_NO_SPLIT


def _select_seq_partition_size(
    num_programs: int, max_seq_len: int, kv_blk_sz: int, dispatch: dict
) -> int:
    """Pick the sequence partition size of the split-K GQA kernel from the config.

    A return value of `_SEQ_PARTITION_SIZE` means: keep the default v1/v2 choice.
    """
    target = dispatch.get("split_target_programs")
    busy = num_programs >= dispatch.get("split_max_programs", 1 << 62)
    short = max_seq_len < dispatch.get(
        "split_short_seq", 0
    ) and num_programs >= dispatch.get("split_short_min_programs", 1 << 62)
    if target is None or busy or short:
        return _SEQ_PARTITION_SIZE
    sizes = [
        p
        for p in sorted(dispatch["split_sizes"], reverse=True)
        if p % kv_blk_sz == 0 and (p == _SEQ_PARTITION_SIZE or p <= max_seq_len)
    ]
    # Largest size that reaches the target; else the smallest size within the cap.
    allowed = [
        (p, triton.cdiv(max_seq_len, p))
        for p in sizes
        if triton.cdiv(max_seq_len, p) <= dispatch["split_max_partitions"]
    ]
    for p, num_parts in allowed:
        if num_programs * num_parts >= target:
            return p
    return allowed[-1][0] if allowed else _SEQ_PARTITION_SIZE


def _gqa_tdm_supported(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
) -> bool:
    """True when the shapes, dtypes and layouts fit the Gluon TDM kernel."""
    _, num_q_heads, head_sz = query.shape
    num_blks, num_kv_heads, kv_blk_sz, _ = key_cache.shape
    group = num_q_heads // num_kv_heads
    return (
        query.dtype in _KV_DTYPE_NAMES
        and key_cache.dtype == query.dtype
        and 1 <= group <= 16
        and num_q_heads == group * num_kv_heads
        and head_sz in _TDM_HEAD_SIZES
        and kv_blk_sz == _TDM_PAGE_SIZE
        and num_blks * num_kv_heads * kv_blk_sz < 1 << 31
        and query.stride(2) == 1
        and output.stride(2) == 1
        and key_cache.is_contiguous()
        and value_cache.is_contiguous()
    )


def _use_gqa_tdm(
    dispatch: dict | None,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    alibi_slopes: torch.Tensor | None,
) -> bool:
    """True when the config enables the gfx1250 TDM kernel and it supports this call."""
    return (
        dispatch is not None
        and dispatch.get("tdm_num_warps") is not None
        and _pa_decode_gqa_tdm_kernel is not None
        and alibi_slopes is None
        and _gqa_tdm_supported(query, key_cache, value_cache, output)
    )


def paged_attention_decode(
    output: torch.Tensor,  # [num_seqs, num_kv_heads*query_grp_sz, head_sz]
    query: torch.Tensor,  # [num_seqs, num_kv_heads*query_grp_sz, head_sz]
    key_cache: torch.Tensor,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz]
    value_cache: torch.Tensor,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz]
    seq_lens: torch.Tensor,  # [num_seqs]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blks_per_seq]
    attn_scale: float,
    max_seq_len: int,
    compute_type,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    num_seq_partitions: int = 0,  # TODO use this below
    alibi_slopes: torch.Tensor = None,
) -> None:
    """
    Paged attention decode with automatic V1/V2 dispatch and quantization support.

    If the arch has a PA-DECODE config, bf16/fp16 caches with scalar scales use V1 only
    up to ``v1_max_partitions`` partitions. Other cases use V1 up to 8192 tokens when
    there is one partition or num_seqs * num_q_heads > 512, and V2 otherwise.

    Args:
        output (torch.Tensor): Pre-allocated output with shape (num_seqs, num_q_heads, head_dim).
        query (torch.Tensor): Query tensor with shape (num_seqs, num_q_heads, head_dim).
        key_cache (torch.Tensor): Paged key cache with shape (num_blocks, num_kv_heads, block_size, head_dim).
        value_cache (torch.Tensor): Paged value cache with shape (num_blocks, num_kv_heads, block_size, head_dim).
        seq_lens (torch.Tensor): Sequence lengths with shape (num_seqs,).
        block_tables (torch.Tensor): Block table mapping with shape (num_seqs, max_blocks_per_seq).
        attn_scale (float): Attention scale, typically 1/sqrt(head_dim).
        max_seq_len (int): Maximum sequence length in batch.
        compute_type: Compute precision type.
        k_scale (torch.Tensor): Key quantization scale. Scalar for per-tensor,
            shape (num_blocks, num_kv_heads, block_size) for per-token.
        v_scale (torch.Tensor): Value quantization scale with same shape as k_scale.
        num_seq_partitions (int): Number of sequence partitions (not currently used).
        alibi_slopes (Optional[torch.Tensor]): ALiBi position bias slopes.

    Returns:
        None. Results written in-place to output.
    """

    _LOGGER.info(
        "PA_DECODE: q=%s key_cache=%s value_cache=%s",
        tuple(query.shape),
        tuple(key_cache.shape),
        tuple(value_cache.shape),
    )
    # get num_seqs, num_kv_heads, kv_blk_sz, head_sz and query_grp_sz
    num_seqs = query.shape[0]
    num_q_heads = query.shape[1]
    num_kv_heads = key_cache.shape[1]

    max_num_partitions = (max_seq_len + _SEQ_PARTITION_SIZE - 1) // _SEQ_PARTITION_SIZE

    dispatch = None
    if k_scale.numel() == 1 and key_cache.dtype in _KV_DTYPE_NAMES:
        dispatch = _get_dispatch_config(_KV_DTYPE_NAMES[key_cache.dtype])
    if _use_gqa_tdm(dispatch, query, key_cache, value_cache, output, alibi_slopes):
        k_scale_val = k_scale.item()
        v_scale_val = v_scale.item()
        # The TDM kernel does not rescale K or V.
        if k_scale_val == 1.0 and v_scale_val == 1.0:
            paged_attn_decode_gqa_tdm(
                output,
                query,
                key_cache,
                value_cache,
                block_tables,
                seq_lens,
                attn_scale,
                num_warps=dispatch["tdm_num_warps"],
                seq_partition_size=_select_tdm_partition_size(
                    num_seqs * num_kv_heads, max_seq_len, dispatch
                ),
                max_seq_len=max_seq_len,
            )
            return
    if dispatch is not None:
        use_v1 = max_num_partitions <= dispatch["v1_max_partitions"]
        # With enough programs the split-free v1 fills the GPU and avoids the reduce.
        busy_min = dispatch.get("v1_min_programs")
        if busy_min is not None and num_seqs * num_kv_heads >= busy_min:
            use_v1 = use_v1 or max_num_partitions <= dispatch["v1_busy_max_partitions"]
    else:
        use_v1 = max_seq_len <= 8192 and (
            max_num_partitions == 1 or num_seqs * num_q_heads > 512
        )
    seq_partition_size = _SEQ_PARTITION_SIZE
    if dispatch is not None and num_q_heads != num_kv_heads:  # GQA only
        seq_partition_size = _select_seq_partition_size(
            num_seqs * num_kv_heads, max_seq_len, key_cache.shape[2], dispatch
        )
        if seq_partition_size != _SEQ_PARTITION_SIZE:
            use_v1 = False
            max_num_partitions = triton.cdiv(max_seq_len, seq_partition_size)
    kv_blk_sz = key_cache.shape[2]
    if kv_blk_sz & (kv_blk_sz - 1) or (not use_v1 and kv_blk_sz > _SEQ_PARTITION_SIZE):
        raise ValueError(
            f"KV block size {kv_blk_sz} is not supported: it must be a power of two, "
            f"and at most {_SEQ_PARTITION_SIZE} when V2 is used"
        )
    if k_scale.numel() > 1:
        if use_v1:
            paged_attn_decode_v1_per_token_quant(
                output,
                query,
                key_cache,
                value_cache,
                block_tables,
                seq_lens,
                max_seq_len,
                compute_type,
                num_kv_heads,
                attn_scale,
                alibi_slopes,
                k_scale,
                v_scale,
            )
        else:
            paged_attn_decode_v2_per_token_quant(
                output,
                query,
                key_cache,
                value_cache,
                block_tables,
                seq_lens,
                max_seq_len,
                compute_type,
                num_kv_heads,
                attn_scale,
                alibi_slopes,
                k_scale,
                v_scale,
                max_num_partitions,
            )
    else:
        if use_v1:
            paged_attn_decode_v1(
                output,
                query,
                key_cache,
                value_cache,
                block_tables,
                seq_lens,
                max_seq_len,
                compute_type,
                num_kv_heads,
                attn_scale,
                alibi_slopes,
                k_scale.item(),
                v_scale.item(),
            )
        else:
            paged_attn_decode_v2(
                output,
                query,
                key_cache,
                value_cache,
                block_tables,
                seq_lens,
                max_seq_len,
                compute_type,
                num_kv_heads,
                attn_scale,
                alibi_slopes,
                k_scale.item(),
                v_scale.item(),
                max_num_partitions,
                seq_partition_size=seq_partition_size,
            )


def paged_attn_decode_v1(
    output: torch.Tensor,  # [num_seqs, num_kv_heads*query_grp_sz, head_sz]
    query: torch.Tensor,  # [num_seqs, num_kv_heads*query_grp_sz, head_sz]
    key_cache: torch.Tensor,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz]
    value_cache: torch.Tensor,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blks_per_seq]
    seq_lens: torch.Tensor,  # [num_seqs]
    max_seq_len: int,
    compute_type,
    num_kv_heads: int,
    scale: float,
    alibi_slopes: torch.Tensor | None,
    k_scale: float,
    v_scale: float,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
):
    """
    #TODO: Add Doc
    """

    num_seqs = query.shape[0]
    num_q_heads = query.shape[1]
    kv_blk_sz = key_cache.shape[2]
    head_sz = key_cache.shape[3]
    query_grp_sz = query.shape[1] // num_kv_heads
    query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)
    kv_blk_sz_pow2 = triton.next_power_of_2(kv_blk_sz)
    head_sz_pow2 = triton.next_power_of_2(head_sz)

    # MHA- Multi-Head Attention
    if query_grp_sz == 1:
        grid = (num_q_heads, num_seqs, 1)
        launch = _get_launch_config("v1_wo_dot", kv_blk_sz) or {}
        _paged_attn_decode_v1_wo_dot_kernel[grid](
            output,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            alibi_slopes,
            scale,
            k_scale,
            v_scale,
            query.stride(0),
            query.stride(1),
            output.stride(0),
            output.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            block_tables.stride(0),
            compute_type=compute_type,
            KV_BLK_SZ=kv_blk_sz,
            KV_BLK_SZ_POW2=kv_blk_sz_pow2,
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
            **launch,
        )
    # GQA - Grouped Query Attention
    else:
        grid = (num_seqs, num_kv_heads, 1)
        if query_grp_sz <= 16:
            query_grp_sz_pow2 = 16
        else:
            query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)
        launch = _get_launch_config("v1_w_dot", kv_blk_sz) or {}
        _paged_attn_decode_v1_w_dot_kernel[grid](
            output,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            alibi_slopes,
            scale,
            k_scale,
            v_scale,
            output.stride(0),
            output.stride(1),
            query.stride(0),
            query.stride(1),
            query.stride(2),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
            block_tables.stride(0),
            compute_type=compute_type,
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
            QUERY_GRP_SZ_POW2=query_grp_sz_pow2,
            KV_BLK_SZ=kv_blk_sz,
            KV_BLK_SZ_POW2=kv_blk_sz,
            **launch,
        )


def paged_attn_decode_v2(
    output: torch.Tensor,  # [num_seqs, num_kv_heads*query_grp_sz, head_sz],
    query: torch.Tensor,  # [num_seqs, num_kv_heads*query_grp_sz, head_sz],
    key_cache: torch.Tensor,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz] ,
    value_cache: torch.Tensor,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz] ,
    block_tables: torch.Tensor,  # [num_seqs, max_num_blks_per_seq],
    seq_lens: torch.Tensor,  # [num_seqs],
    max_seq_len: int,
    compute_type,
    num_kv_heads: int,
    scale: float,
    alibi_slopes: torch.Tensor | None,
    k_scale: float,
    v_scale: float,
    max_num_partitions: int,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
    seq_partition_size: int = _SEQ_PARTITION_SIZE,  # GQA only
):
    """
    #TODO: Add Doc
    """

    num_seqs = query.shape[0]
    num_q_heads = query.shape[1]
    kv_blk_sz = key_cache.shape[2]
    head_sz = key_cache.shape[3]
    query_grp_sz = num_q_heads // num_kv_heads
    query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)

    # Note: There is a bug in triton.next_power_of_2 function which causes it
    # to update the passed in arg, so that's why we have a workaround here
    # max_num_partitions_pow2 = triton.next_power_of_2(max_num_partitions)
    if max_num_partitions == 0:
        max_num_partitions_pow2 = 1
    else:
        max_num_partitions_pow2 = 2 ** math.ceil(math.log2(max_num_partitions))

    kv_blk_sz_pow2 = triton.next_power_of_2(kv_blk_sz)
    head_sz_pow2 = triton.next_power_of_2(head_sz)

    # MHA
    if query_grp_sz == 1:
        grid = (num_q_heads, num_seqs, max_num_partitions)
        shape_info = (num_seqs, num_q_heads, max_num_partitions)
        exp_sums = torch.empty(
            size=shape_info, dtype=torch.float32, device=output.device
        )
        max_logits = torch.empty(
            size=shape_info, dtype=torch.float32, device=output.device
        )
        tmp_output = torch.empty(
            (*shape_info, head_sz), dtype=output.dtype, device=output.device
        )
        launch = _get_launch_config("v2_wo_dot", kv_blk_sz) or {}
        _paged_attn_decode_v2_wo_dot_kernel[grid](
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            scale,
            k_scale,
            v_scale,
            alibi_slopes,
            exp_sums.stride(0),
            exp_sums.stride(1),
            tmp_output.stride(0),
            tmp_output.stride(1),
            tmp_output.stride(2),
            query.stride(0),
            query.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            block_tables.stride(0),
            block_tables.stride(1),
            compute_type=compute_type,
            KV_BLK_SZ=kv_blk_sz,
            KV_BLK_SZ_POW2=kv_blk_sz_pow2,
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
            SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
            **launch,
        )
        grid = (num_q_heads, num_seqs, 1)
        _paged_attn_decode_v2_wo_dot_reduce_kernel[grid](
            output,
            exp_sums,
            max_logits,
            tmp_output,
            seq_lens,
            output.stride(0),
            output.stride(1),
            exp_sums.stride(0),
            exp_sums.stride(1),
            tmp_output.stride(0),
            tmp_output.stride(1),
            tmp_output.stride(2),
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
            MAX_NUM_SEQ_PARTITIONS_POW2=int(max_num_partitions_pow2),
        )
    # GQA
    else:
        grid = (num_seqs, num_kv_heads, max_num_partitions)
        shape_info = (num_seqs, num_kv_heads, max_num_partitions, query_grp_sz)
        max_logits = torch.empty(shape_info, dtype=torch.float32, device=output.device)
        exp_sums = torch.empty(shape_info, dtype=torch.float32, device=output.device)
        tmp_output = torch.empty(
            *shape_info, head_sz, dtype=output.dtype, device=output.device
        )
        if query_grp_sz <= 16:
            query_grp_sz_pow2 = 16
        else:
            query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)
        launch = _get_launch_config("v2_w_dot", kv_blk_sz) or {}
        _paged_attn_decode_v2_w_dot_kernel[grid](
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            scale,
            k_scale,
            v_scale,
            alibi_slopes,
            exp_sums.stride(0),
            exp_sums.stride(1),
            exp_sums.stride(2),
            tmp_output.stride(0),
            tmp_output.stride(1),
            tmp_output.stride(2),
            tmp_output.stride(3),
            query.stride(0),
            query.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            block_tables.stride(0),
            compute_type=compute_type,
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
            QUERY_GRP_SZ_POW2=query_grp_sz_pow2,
            KV_BLK_SZ=kv_blk_sz,
            KV_BLK_SZ_POW2=kv_blk_sz_pow2,
            SEQ_PARTITION_SZ=seq_partition_size,
            **launch,
        )
        grid = (num_seqs, num_kv_heads, 1)
        _paged_attn_decode_v2_w_dot_reduce_kernel[grid](
            output,
            exp_sums,
            max_logits,
            tmp_output,
            seq_lens,
            output.stride(0),
            output.stride(1),
            exp_sums.stride(0),
            exp_sums.stride(1),
            exp_sums.stride(2),
            tmp_output.stride(0),
            tmp_output.stride(1),
            tmp_output.stride(2),
            tmp_output.stride(3),
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
            QUERY_GRP_SZ_POW2=query_grp_sz_pow2,
            SEQ_PARTITION_SZ=seq_partition_size,
            MAX_NUM_SEQ_PARTITIONS_POW2=int(triton.next_power_of_2(max_num_partitions)),
        )


def paged_attn_decode_gqa_tdm(
    output: torch.Tensor,  # [num_seqs, num_q_heads, head_sz]
    query: torch.Tensor,  # [num_seqs, num_q_heads, head_sz]
    key_cache: torch.Tensor,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz]
    value_cache: torch.Tensor,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blks_per_seq], int32
    seq_lens: torch.Tensor,  # [num_seqs], at least 1
    scale: float,
    num_warps: int,
    seq_partition_size: int = _TDM_NO_SPLIT,
    max_seq_len: int | None = None,
):
    """GQA decode on gfx1250 with TDM page loads into LDS. The result goes to `output`.

    Needs unit K/V scales and no ALiBi. A `seq_partition_size` below `_TDM_NO_SPLIT`
    cuts each sequence into partitions that the v2 reduce kernel combines.
    """
    assert _pa_decode_gqa_tdm_kernel is not None, "needs the gfx1250 Gluon kernel"
    assert _gqa_tdm_supported(query, key_cache, value_cache, output), (
        "the TDM kernel needs fp16/bf16 contiguous caches, head size 56, 64 or 128, "
        "page 128 and a group of at most 16"
    )
    num_seqs, num_q_heads, head_sz = query.shape
    num_blks, num_kv_heads, kv_blk_sz, _ = key_cache.shape
    group = num_q_heads // num_kv_heads

    split = seq_partition_size < _TDM_NO_SPLIT
    num_parts = 1
    if split:
        assert max_seq_len is not None and seq_partition_size % kv_blk_sz == 0
        num_parts = triton.cdiv(max_seq_len, seq_partition_size)
        shape_info = (num_seqs, num_kv_heads, num_parts, group)
        max_logits = torch.empty(shape_info, dtype=torch.float32, device=output.device)
        exp_sums = torch.empty(shape_info, dtype=torch.float32, device=output.device)
        tmp_output = torch.empty(
            *shape_info, head_sz, dtype=output.dtype, device=output.device
        )
    else:
        max_logits = exp_sums = tmp_output = output

    _pa_decode_gqa_tdm_kernel[(num_seqs, num_kv_heads, num_parts)](
        output,
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        num_blks * num_kv_heads * kv_blk_sz,
        scale,
        output.stride(0),
        output.stride(1),
        query.stride(0),
        query.stride(1),
        block_tables.stride(0),
        exp_sums,
        max_logits,
        tmp_output,
        *(
            (
                exp_sums.stride(0),
                exp_sums.stride(1),
                exp_sums.stride(2),
                tmp_output.stride(0),
                tmp_output.stride(1),
                tmp_output.stride(2),
                tmp_output.stride(3),
            )
            if split
            else (0,) * 7
        ),
        NUM_KV_HEADS=num_kv_heads,
        G=group,
        HEAD_SZ=head_sz,
        BLOCK_H=16,
        BLOCK_D=triton.next_power_of_2(head_sz),
        BLOCK_K=kv_blk_sz,
        SEQ_PARTITION_SZ=seq_partition_size,
        SPLIT=split,
        num_warps=num_warps,
    )
    if split:
        _paged_attn_decode_v2_w_dot_reduce_kernel[(num_seqs, num_kv_heads, 1)](
            output,
            exp_sums,
            max_logits,
            tmp_output,
            seq_lens,
            output.stride(0),
            output.stride(1),
            exp_sums.stride(0),
            exp_sums.stride(1),
            exp_sums.stride(2),
            tmp_output.stride(0),
            tmp_output.stride(1),
            tmp_output.stride(2),
            tmp_output.stride(3),
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=triton.next_power_of_2(head_sz),
            QUERY_GRP_SZ=group,
            QUERY_GRP_SZ_POW2=16,
            SEQ_PARTITION_SZ=seq_partition_size,
            MAX_NUM_SEQ_PARTITIONS_POW2=int(triton.next_power_of_2(num_parts)),
        )


def paged_attn_decode_v1_per_token_quant(
    output: torch.Tensor,  # [num_seqs, num_kv_heads*query_grp_sz, head_sz]
    query: torch.Tensor,  # [num_seqs, num_kv_heads*query_grp_sz, head_sz]
    key_cache: torch.Tensor,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz]
    value_cache: torch.Tensor,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blks_per_seq]
    seq_lens: torch.Tensor,  # [num_seqs]
    max_seq_len: int,
    compute_type,
    num_kv_heads: int,
    scale: float,
    alibi_slopes: torch.Tensor | None,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
):
    """
    #TODO: Add Doc
    """

    num_seqs = query.shape[0]
    num_q_heads = query.shape[1]
    kv_blk_sz = key_cache.shape[2]
    head_sz = key_cache.shape[3]
    query_grp_sz = query.shape[1] // num_kv_heads
    query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)
    kv_blk_sz_pow2 = triton.next_power_of_2(kv_blk_sz)
    head_sz_pow2 = triton.next_power_of_2(head_sz)

    # MHA- Multi-Head Attention
    if query_grp_sz == 1:
        grid = (num_q_heads, num_seqs, 1)
        _paged_attn_decode_v1_wo_dot_kernel_per_token_quant[grid](
            output,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            alibi_slopes,
            scale,
            k_scale,
            v_scale,
            query.stride(0),
            query.stride(1),
            output.stride(0),
            output.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            block_tables.stride(0),
            k_scale.stride(0),
            k_scale.stride(1),
            k_scale.stride(2),
            compute_type=compute_type,
            KV_BLK_SZ=kv_blk_sz,
            KV_BLK_SZ_POW2=kv_blk_sz_pow2,
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
        )
    # GQA - Grouped Query Attention
    else:
        grid = (num_seqs, num_kv_heads, 1)
        if query_grp_sz <= 16:
            query_grp_sz_pow2 = 16
        else:
            query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)
        _paged_attn_decode_v1_w_dot_kernel_per_token_quant[grid](
            output,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            alibi_slopes,
            scale,
            k_scale,
            v_scale,
            output.stride(0),
            output.stride(1),
            query.stride(0),
            query.stride(1),
            query.stride(2),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
            block_tables.stride(0),
            k_scale.stride(0),
            k_scale.stride(1),
            k_scale.stride(2),
            compute_type=compute_type,
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
            QUERY_GRP_SZ_POW2=query_grp_sz_pow2,
            KV_BLK_SZ=kv_blk_sz,
            KV_BLK_SZ_POW2=kv_blk_sz,
            waves_per_eu=3 if query_grp_sz > 1 else 0,
            num_stages=1,
        )


def paged_attn_decode_v2_per_token_quant(
    output: torch.Tensor,  # [num_seqs, num_kv_heads*query_grp_sz, head_sz],
    query: torch.Tensor,  # [num_seqs, num_kv_heads*query_grp_sz, head_sz],
    key_cache: torch.Tensor,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz] ,
    value_cache: torch.Tensor,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz] ,
    block_tables: torch.Tensor,  # [num_seqs, max_num_blks_per_seq],
    seq_lens: torch.Tensor,  # [num_seqs],
    max_seq_len: int,
    compute_type,
    num_kv_heads: int,
    scale: float,
    alibi_slopes: torch.Tensor | None,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    max_num_partitions: int,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
):
    """
    #TODO: Add Doc
    """

    num_seqs = query.shape[0]
    num_q_heads = query.shape[1]
    kv_blk_sz = key_cache.shape[2]
    head_sz = key_cache.shape[3]
    query_grp_sz = num_q_heads // num_kv_heads
    query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)

    # Note: There is a bug in triton.next_power_of_2 function which causes it
    # to update the passed in arg, so that's why we have a workaround here
    # max_num_partitions_pow2 = triton.next_power_of_2(max_num_partitions)
    if max_num_partitions == 0:
        max_num_partitions_pow2 = 1
    else:
        max_num_partitions_pow2 = 2 ** math.ceil(math.log2(max_num_partitions))

    kv_blk_sz_pow2 = triton.next_power_of_2(kv_blk_sz)
    head_sz_pow2 = triton.next_power_of_2(head_sz)

    # MHA
    if query_grp_sz == 1:
        grid = (num_q_heads, num_seqs, max_num_partitions)
        shape_info = (num_seqs, num_q_heads, max_num_partitions)
        exp_sums = torch.empty(
            size=shape_info, dtype=torch.float32, device=output.device
        )
        max_logits = torch.empty(
            size=shape_info, dtype=torch.float32, device=output.device
        )
        tmp_output = torch.empty(
            (*shape_info, head_sz), dtype=output.dtype, device=output.device
        )
        _paged_attn_decode_v2_wo_dot_kernel_per_token_quant[grid](
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            scale,
            k_scale,
            v_scale,
            alibi_slopes,
            exp_sums.stride(0),
            exp_sums.stride(1),
            tmp_output.stride(0),
            tmp_output.stride(1),
            tmp_output.stride(2),
            query.stride(0),
            query.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            block_tables.stride(0),
            block_tables.stride(1),
            k_scale.stride(0),
            k_scale.stride(1),
            k_scale.stride(2),
            compute_type=compute_type,
            KV_BLK_SZ=kv_blk_sz,
            KV_BLK_SZ_POW2=kv_blk_sz_pow2,
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
            SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
        )
        grid = (num_q_heads, num_seqs, 1)
        _paged_attn_decode_v2_wo_dot_reduce_kernel_per_token_quant[grid](
            output,
            exp_sums,
            max_logits,
            tmp_output,
            seq_lens,
            output.stride(0),
            output.stride(1),
            exp_sums.stride(0),
            exp_sums.stride(1),
            tmp_output.stride(0),
            tmp_output.stride(1),
            tmp_output.stride(2),
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
            MAX_NUM_SEQ_PARTITIONS_POW2=int(max_num_partitions_pow2),
        )
    # GQA
    else:
        grid = (num_seqs, num_kv_heads, max_num_partitions)
        shape_info = (num_seqs, num_kv_heads, max_num_partitions, query_grp_sz)
        max_logits = torch.empty(shape_info, dtype=torch.float32, device=output.device)
        exp_sums = torch.empty(shape_info, dtype=torch.float32, device=output.device)
        tmp_output = torch.empty(
            *shape_info, head_sz, dtype=output.dtype, device=output.device
        )
        if query_grp_sz <= 16:
            query_grp_sz_pow2 = 16
        else:
            query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)
        _paged_attn_decode_v2_w_dot_kernel_per_token_quant[grid](
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            scale,
            k_scale,
            v_scale,
            alibi_slopes,
            exp_sums.stride(0),
            exp_sums.stride(1),
            exp_sums.stride(2),
            tmp_output.stride(0),
            tmp_output.stride(1),
            tmp_output.stride(2),
            tmp_output.stride(3),
            query.stride(0),
            query.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            block_tables.stride(0),
            k_scale.stride(0),
            k_scale.stride(1),
            k_scale.stride(2),
            compute_type=compute_type,
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
            QUERY_GRP_SZ_POW2=query_grp_sz_pow2,
            KV_BLK_SZ=kv_blk_sz,
            KV_BLK_SZ_POW2=kv_blk_sz_pow2,
            SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
        )
        grid = (num_seqs, num_kv_heads, 1)
        _paged_attn_decode_v2_w_dot_reduce_kernel_per_token_quant[grid](
            output,
            exp_sums,
            max_logits,
            tmp_output,
            seq_lens,
            output.stride(0),
            output.stride(1),
            exp_sums.stride(0),
            exp_sums.stride(1),
            exp_sums.stride(2),
            tmp_output.stride(0),
            tmp_output.stride(1),
            tmp_output.stride(2),
            tmp_output.stride(3),
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
            QUERY_GRP_SZ_POW2=query_grp_sz_pow2,
            SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
            MAX_NUM_SEQ_PARTITIONS_POW2=int(triton.next_power_of_2(max_num_partitions)),
        )
