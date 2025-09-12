# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Lean Attention
===============
This is a Triton implementation of the Lean Attention algorithm from https://arxiv.org/abs/2405.10480
Lean Attention adopts streamK style tiling strategy, which efficiently utilize all available CUs in the system.
Lean Attention is for both decode and prefill attention of transformer based models.

It currently supports ragged batching decode and prefill attention with causal=1

TO be added features:
- Add GQA support
- Misc
    - N_CTX with non-integer number of BLOCK_SIZE_N (pad zeros or add mask)
    -
"""

import functools
import json
import triton
import triton.language as tl
from typing import Optional
from bisect import bisect_right
from ..utils._triton.pid_preprocessing import pid_grid, remap_xcd
from ..utils._triton import arch_info
from ..utils.core import AITER_TRITON_CONFIGS_PATH


LOG_TWO_E = 1.44269504  # log_2(e) value for softmax scaling
# Support tensor in [B, Seqlen, H, d] format. Taking tensors in [B*Seqlen, H, d] as inputs


@functools.lru_cache(maxsize=1024)
def _get_config(
    causal: bool,
    batch_size: int,
):
    if not hasattr(_get_config, "_config_dict"):
        dev = arch_info.get_device()
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/{dev}-LEANATTN-DEFAULT.json"
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_config._config_dict = config

    config = _get_config._config_dict["any"]
    return (
        config.copy()
    )  # return a copy to avoid mutation of stored config in LRU cache


@triton.jit
def find_group_sequential(x, MASKED_BLOCKS: tl.constexpr, num_m_blocks: tl.constexpr):
    i = tl.arange(0, num_m_blocks)
    q_block_idx = i  # Group indices: 0, 1, 2, ...

    task_size = (q_block_idx + 1) * MASKED_BLOCKS
    cumulative = tl.cumsum(task_size) - task_size
    mask = cumulative + task_size > x

    # Find only the first True occurrence in mask.
    mask_int = mask.to(tl.int32)
    prefix_sum = tl.cumsum(mask_int)
    one_hot = mask & (prefix_sum == 1)

    # Handle case where no group is found (e.g., x is out of range)
    found = tl.sum(one_hot)
    sentinel = -1

    final_q_block_idx = tl.where(found > 0, tl.sum(q_block_idx * one_hot), sentinel)
    final_task_size = tl.where(found > 0, tl.sum(task_size * one_hot), 0)
    final_total_blocks = tl.where(found > 0, tl.sum(cumulative * one_hot), 0)

    return final_q_block_idx, final_task_size, final_total_blocks


@triton.jit
def find_group_pingpong(x, MASKED_BLOCKS: tl.constexpr, num_m_blocks: tl.constexpr):
    i = tl.arange(0, num_m_blocks)
    pair_idx = i // 2
    q_block_idx = tl.where(i % 2 == 0, pair_idx, num_m_blocks - 1 - pair_idx)
    task_size = (q_block_idx + 1) * MASKED_BLOCKS
    cumulative = tl.cumsum(task_size) - task_size
    mask = cumulative + task_size > x

    # Use cumsum to construct a one-hot for the first True
    mask_int = mask.to(tl.int32)
    prefix_sum = tl.cumsum(mask_int)
    one_hot = mask & (prefix_sum == 1)

    final_q_block_idx = tl.sum(q_block_idx * one_hot)
    final_task_size = tl.sum(task_size * one_hot)
    final_total_blocks = tl.sum(cumulative * one_hot)

    return final_q_block_idx, final_task_size, final_total_blocks


@triton.jit
def _attention_inner(
    q,
    k_ptrs,
    v_ptrs,
    stride_vn,
    stride_kn,
    m_i,
    l_i,
    acc,
    qk_scale,
    causal,
    q_start_m,
    b_seq_size,
    offs_m,
    offs_n,
    BLOCK_M,
    BLOCK_N,
    HEAD_DIM,
    local_iter,
    local_iter_end,
):
    """
    Performs attention calculation for an (maybe partial) output tile
    """
    for l_iter in range(local_iter, local_iter_end):
        k = tl.load(k_ptrs)
        qk = tl.dot(q, k) * qk_scale

        if causal:
            # Get the starting column index of the current K block
            k_start_n = (b_seq_size + l_iter) * BLOCK_N
            # Create mask based on absolute sequence positions
            mask = (q_start_m + offs_m[:, None]) >= (k_start_n + offs_n[None, :])
            # Apply the mask
            qk = tl.where(mask, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))

        if causal:
            # Safe exp2 calculation to avoid NaNs
            p_arg = qk - m_ij[:, None]
            p_arg = tl.where(m_ij[:, None] == float("-inf"), float("-inf"), p_arg)
            p = tl.math.exp2(p_arg)

            # Safe scaling factor calculation
            alpha_arg = m_i - m_ij
            alpha_arg = tl.where(m_ij == float("-inf"), 0.0, alpha_arg)
            alpha = tl.math.exp2(alpha_arg)
        else:
            qk = qk - m_ij[:, None]
            p = tl.math.exp2(qk)  # p.shape = [BLOCK_M, BLOCK_N]
            alpha = tl.math.exp2(m_i - m_ij)

        # Update accumulator
        acc = acc * alpha[:, None]
        v = tl.load(v_ptrs)
        acc += tl.dot(p.to(v.dtype), v)

        # Update stats
        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        m_i = m_ij.to(m_i.dtype)

        # update k/v pointer
        v_ptrs += BLOCK_N * stride_vn
        k_ptrs += BLOCK_N * stride_kn
    return m_i, l_i, acc


@triton.jit
def remap_xcd(pid, GRID_MN: tl.constexpr, NUM_XCDS: tl.constexpr = 8):
    ## pid remapping on xcds
    # Number of pids per XCD in the new arrangement
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN cannot divide NUM_XCDS, some xcds will have
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
    # We calculate the number of xcds that have pids_per_xcd pids as
    # tall_xcds
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    # Compute current XCD and local pid within the XCD
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    # Calculate new pid based on the new grouping
    # Note that we need to consider the following two cases:
    # 1. the current pid is on a tall xcd
    # 2. the current pid is on a short xcd
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )

    return pid, pids_per_xcd


@triton.jit
def la_persistent(
    is_pod,
    pod_pid,
    Q,
    K,
    V,
    qk_scale,
    Mp,
    Lp,
    Op,
    Out,
    batch_num_block_n,
    locks,
    stride_qm,  # n_ctx_q
    stride_qh,  # Head
    stride_qk,  # head_dim
    stride_kn,
    stride_kh,
    stride_kk,
    stride_vn,
    stride_vh,
    stride_vk,
    stride_om,  # n_ctx_q
    stride_oh,  # Head
    stride_on,  # head_dim
    stride_oph,  # total_programs
    stride_opm,  # n_ctx_q
    stride_opn,  # head_dim
    HEADS_PER_XCD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MASKED_BLOCKS: tl.constexpr,
    XCD_REMAP: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    batch_size: tl.constexpr,
    causal: tl.constexpr,
    num_m_blocks: tl.constexpr,
    num_n_blocks: tl.constexpr,
    total_programs: tl.constexpr,
    # leanAttention params
    high_load_wgs: tl.constexpr,
    max_tiles_per_wg: tl.constexpr,
    tiles_per_head: tl.constexpr,
    num_splits: tl.constexpr,
    max_output_tile_cnt: tl.constexpr,
):
    if is_pod:
        current_pid = pod_pid
    else:
        current_pid = tl.program_id(0)

    if XCD_REMAP:
        # remap pid's so contiguous group of pid's reside on the same XCD
        current_pid, pids_per_xcd = remap_xcd(
            current_pid, GRID_MN=total_programs, NUM_XCDS=NUM_XCDS
        )
        # XCD_REMAP: high_load_wgs, max_tiles_per_wg are relative to 1 XCD
        xcd_pid = current_pid % pids_per_xcd
        xcd_id = current_pid // pids_per_xcd
    else:
        xcd_pid = current_pid
        xcd_id = 0

    if xcd_pid < high_load_wgs:
        iter = max_tiles_per_wg * xcd_pid
        cta_end_tile_gid = iter + max_tiles_per_wg
    else:
        iter = (max_tiles_per_wg - 1) * (
            xcd_pid - high_load_wgs
        ) + high_load_wgs * max_tiles_per_wg
        cta_end_tile_gid = iter + (max_tiles_per_wg - 1)

    tl.assume(stride_qm > 0)  # n_ctx_q
    tl.assume(stride_qh > 0)  # Head
    tl.assume(stride_qk > 0)  # head_dim
    tl.assume(stride_kn > 0)
    tl.assume(stride_kh > 0)
    tl.assume(stride_kk > 0)
    tl.assume(stride_vn > 0)
    tl.assume(stride_vh > 0)
    tl.assume(stride_vk > 0)
    tl.assume(stride_om > 0)  # n_ctx_q
    tl.assume(stride_oh > 0)  # Head
    tl.assume(stride_on > 0)  # head_dim
    tl.assume(stride_oph > 0)  # total_programs
    tl.assume(stride_opm > 0)  # n_ctx_q
    tl.assume(stride_opn > 0)  # head_dim

    for i in tl.static_range(max_output_tile_cnt + 1):
        if iter < cta_end_tile_gid:
            iter = la_persistent_inner(
                Q,
                K,
                V,
                qk_scale,
                Mp,
                Lp,
                Op,
                Out,
                batch_num_block_n,
                locks,
                stride_qm,  # n_ctx_q
                stride_qh,  # Head
                stride_qk,  # head_dim
                stride_kn,
                stride_kh,
                stride_kk,
                stride_vn,
                stride_vh,
                stride_vk,
                stride_om,  # n_ctx_q
                stride_oh,  # Head
                stride_on,  # head_dim
                stride_oph,  # total_programs
                stride_opm,  # n_ctx_q
                stride_opn,  # head_dim
                iter=iter,
                cta_end_tile_gid=cta_end_tile_gid,
                current_pid=current_pid,
                xcd_pid=xcd_pid,
                xcd_id=xcd_id,
                HEADS_PER_XCD=HEADS_PER_XCD,
                HEAD_DIM=HEAD_DIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                MASKED_BLOCKS=MASKED_BLOCKS,
                XCD_REMAP=XCD_REMAP,
                NUM_XCDS=NUM_XCDS,
                batch_size=batch_size,
                causal=causal,
                num_m_blocks=num_m_blocks,
                num_n_blocks=num_n_blocks,
                # leanAttention params
                high_load_wgs=high_load_wgs,
                max_tiles_per_wg=max_tiles_per_wg,
                tiles_per_head=tiles_per_head,
                num_splits=num_splits,
            )


@triton.jit
def la_persistent_inner(
    Q,
    K,
    V,
    qk_scale,
    Mp,
    Lp,
    Op,
    Out,
    batch_num_block_n,
    locks,
    stride_qm,  # n_ctx_q
    stride_qh,  # Head
    stride_qk,  # head_dim
    stride_kn,
    stride_kh,
    stride_kk,
    stride_vn,
    stride_vh,
    stride_vk,
    stride_om,  # n_ctx_q
    stride_oh,  # Head
    stride_on,  # head_dim
    stride_oph,  # total_programs
    stride_opm,  # n_ctx_q
    stride_opn,  # head_dim
    iter,
    cta_end_tile_gid,
    current_pid,  # SOC pid
    xcd_pid,  # XCD pid
    xcd_id,  # The XCD the pid belongs to
    HEADS_PER_XCD,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MASKED_BLOCKS: tl.constexpr,
    XCD_REMAP: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    batch_size: tl.constexpr,
    causal: tl.constexpr,
    num_m_blocks: tl.constexpr,
    num_n_blocks: tl.constexpr,
    # leanAttention params
    high_load_wgs: tl.constexpr,
    max_tiles_per_wg: tl.constexpr,
    tiles_per_head: tl.constexpr,
    num_splits: tl.constexpr,
):

    tl.assume(stride_qm > 0)  # n_ctx_q
    tl.assume(stride_qh > 0)  # Head
    tl.assume(stride_qk > 0)  # head_dim
    tl.assume(stride_kn > 0)
    tl.assume(stride_kh > 0)
    tl.assume(stride_kk > 0)
    tl.assume(stride_vn > 0)
    tl.assume(stride_vh > 0)
    tl.assume(stride_vk > 0)
    tl.assume(stride_om > 0)  # n_ctx_q
    tl.assume(stride_oh > 0)  # Head
    tl.assume(stride_on > 0)  # head_dim
    tl.assume(stride_oph > 0)  # total_programs
    tl.assume(stride_opm > 0)  # n_ctx_q
    tl.assume(stride_opn > 0)  # head_dim

    # Loop context length
    # while iter < cta_end_tile_gid:
    # Calculate index of current head output tile
    # The tiles_per_head is the sum of # BLOCK_N in K/V sequence of all batches
    # All the tiling indices calculated below are relative to 1 XCD when XCD_REMAP=True
    tile_head_idx = iter // tiles_per_head
    # To generate an otuput tile, a loop over [tile_iter, tile_iter_end) lean tiles is needed
    # [tile_iter, tile_iter_end) are in the form of global tile id
    if causal:
        tile_batch_idx = (iter % tiles_per_head) // (tiles_per_head // batch_size)
        # Does not support ragged batching. All requests in the batch have the same context length (per_head_tile_size)
        # tiles_per_head: total sum of # BLOCK_N in K/V sequence of all batches
        # per_head_tile_size: per head # BLOCK_N of each output tile

        per_head_tile_idx, per_head_tile_size, total_blocks = find_group_pingpong(
            iter
            - (tile_head_idx * tiles_per_head)
            - (tile_batch_idx * (tiles_per_head // batch_size)),
            MASKED_BLOCKS,
            num_m_blocks,
        )
        """
        per_head_tile_idx, per_head_tile_size, total_blocks = find_group_sequential(
            iter
            - (tile_head_idx * tiles_per_head)
            - (tile_batch_idx * (tiles_per_head // batch_size)),
            MASKED_BLOCKS,
            num_m_blocks,
        )
        """
        tile_iter = (
            tile_head_idx * tiles_per_head
            + (tile_batch_idx * (tiles_per_head // batch_size))
            + total_blocks
        )
        tile_iter_end = tile_iter + (per_head_tile_size)
        tile_idx = (
            tile_head_idx * batch_size + tile_batch_idx
        ) * num_m_blocks + per_head_tile_idx
    else:
        tile_idx = (
            tile_head_idx * batch_size
        )  # Output tile idx, 1 output tile per head per batch
        tile_iter = tile_head_idx * tiles_per_head
        if batch_size == 1:
            req_size = tiles_per_head
        else:
            req_size = tl.load(batch_num_block_n)
        tile_iter_end = tile_iter + req_size
        for b in range(1, batch_size):
            next_req_size = tl.load(batch_num_block_n + b)
            local_head_iter = iter % tiles_per_head
            if (local_head_iter < next_req_size) and (local_head_iter >= req_size):
                tile_iter = tile_iter + req_size
                tile_idx = tile_idx + b
                tile_iter_end = tile_iter + (next_req_size - req_size)
            req_size = next_req_size
    # Local lean tile ID within a loop of an output tile
    local_iter = iter - tile_iter
    local_iter_end = tl.minimum(tile_iter_end, cta_end_tile_gid) - tile_iter

    if iter == tile_iter:
        host_block = True
    else:
        host_block = False
    # finishing_block: the output tile is finished within this block
    if cta_end_tile_gid >= tile_iter_end:
        finishing_block = True
    else:
        finishing_block = False

    # Q/K/V/O offsets calculation needs global head index.
    # When XCD_REMAP=False, xcd_id=0
    tile_head_idx_global = HEADS_PER_XCD * xcd_id + tile_head_idx

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    if causal:
        b_seq_size = tile_batch_idx * num_n_blocks
    else:
        tile_batch_idx = tile_idx % batch_size
        b_seq_size = 0
        if tile_batch_idx > 0:
            b_seq_size = tl.load(
                batch_num_block_n + tile_batch_idx - 1
            )  # Previous batch size

    k_offs = (
        (b_seq_size + local_iter) * BLOCK_N * stride_kn
        + tile_head_idx_global * stride_kh
        + offs_n[None, :] * stride_kn
        + offs_k[:, None] * stride_kk
    )
    v_offs = (
        (b_seq_size + local_iter) * BLOCK_N * stride_vn
        + tile_head_idx_global * stride_vh
        + offs_n[:, None] * stride_vn
        + offs_k[None, :] * stride_vk
    )

    k_ptrs = K + k_offs
    k_ptrs = tl.multiple_of(k_ptrs, (16, 1))
    v_ptrs = V + v_offs
    v_ptrs = tl.multiple_of(v_ptrs, (1, 16))

    if causal:
        q_idx = per_head_tile_idx + tile_batch_idx * num_m_blocks
        q_start_m = q_idx * BLOCK_M
    else:
        q_idx = tile_batch_idx
        q_start_m = 0

    q_offs = (
        q_idx * BLOCK_M * stride_qm
        + tile_head_idx_global * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_k[None, :] * stride_qk
    )
    q_ptrs = Q + q_offs
    q_ptrs = tl.multiple_of(q_ptrs, (1, 16))

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q = tl.load(q_ptrs)

    m_i, l_i, acc = _attention_inner(
        q,
        k_ptrs,
        v_ptrs,
        stride_vn,
        stride_kn,
        m_i,
        l_i,
        acc,
        qk_scale,
        causal,
        q_start_m,
        b_seq_size,
        offs_m,
        offs_n,
        BLOCK_M,
        BLOCK_N,
        HEAD_DIM,
        local_iter,
        local_iter_end,
    )

    # initialize pointer to m and l
    m_cta = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_cta = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    # acc_cta = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # lean output tile epilogue
    if not host_block:
        # Update pointers of partial results Mp[cta], Lp[cta], Op[cta]
        mp_ptrs = Mp + current_pid * BLOCK_M + offs_m
        lp_ptrs = Lp + current_pid * BLOCK_M + offs_m
        op_ptrs = (
            Op
            + current_pid * stride_oph  # stride_oph is total_program dimension
            + offs_m[:, None] * stride_opm
            + offs_k[None, :] * stride_opn
        )

        tl.store(mp_ptrs, m_i, cache_modifier=".wt")
        tl.store(lp_ptrs, l_i, cache_modifier=".wt")
        tl.store(op_ptrs, acc, cache_modifier=".wt")
        tl.debug_barrier()
        tl.store(locks + current_pid, 1, cache_modifier=".wt")
        # According to streamK gemm, store + cache_modifier won't work universally
        # atomic_xchg is better solution but a less performant variant
        # tl.atomic_xchg(locks + current_pid, 1)

    if host_block:  # and finishing_block:
        # A host block that is also a finishing block completes all the LeanTile iterations for its output tile
        # in a single CTA and so can directly store its results from LeanTile() in global memory without any reduction
        acc_reshaped = tl.reshape(acc, (BLOCK_M, 2, HEAD_DIM // 2))
        acc_permuted = tl.permute(acc_reshaped, (0, 2, 1))
        acc0, acc1 = tl.split(acc_permuted)

        if not finishing_block:
            # if host not finishing_block: # another CTA is processing the end of the output tile and store partial results
            """
            last_cta = xcd_pid + 1
            temp_end_gid = cta_end_tile_gid
            split = 1
            while (split < num_splits) and (temp_end_gid < tile_iter_end):
                if last_cta < high_load_wgs:
                    if (tile_iter_end - temp_end_gid) < max_tiles_per_wg:
                        temp_end_gid += tile_iter_end - temp_end_gid
                    else:
                        temp_end_gid += max_tiles_per_wg
                else:
                    if (tile_iter_end - temp_end_gid) < (max_tiles_per_wg - 1):
                        temp_end_gid += tile_iter_end - temp_end_gid
                    else:
                        temp_end_gid += max_tiles_per_wg - 1

                last_cta += 1
                split += 1
            """

            # Calculate #CTAs that store partial result for this output tile
            zero_i = tl.full((), 0, dtype=tl.int32)
            start_cta = tl.cast(xcd_pid + 1, tl.int32)

            # remaining tiles to cover
            remaining = tl.maximum(
                tl.cast(tile_iter_end - cta_end_tile_gid, tl.int32), zero_i
            )

            # capacities (use int32)
            cap_high = tl.cast(max_tiles_per_wg, tl.int32)
            cap_low = tl.cast(max_tiles_per_wg - 1, tl.int32)
            cap_low = tl.where(cap_low > 0, cap_low, tl.full((), 1, dtype=tl.int32))

            # available high-load CTAs starting from start_cta
            ctas_high_avail = tl.maximum(
                tl.cast(high_load_wgs, tl.int32) - start_cta, zero_i
            )
            total_high_capacity = ctas_high_avail * cap_high
            need_high_only = (remaining + cap_high - 1) // cap_high

            # remaining after using all available high CTAs
            rem_after_high = tl.maximum(remaining - total_high_capacity, zero_i)

            # CTAs required in the low region (ceil)
            need_low_after_high = (rem_after_high + cap_low - 1) // cap_low
            ctas_needed = tl.where(
                remaining <= total_high_capacity,
                need_high_only,
                ctas_high_avail + need_low_after_high,
            )

            # Allowed at most (num_splits - 1) extra CTAs:
            max_ctas_allowed = tl.maximum(tl.cast(num_splits - 1, tl.int32), zero_i)
            ctas_to_use = tl.minimum(ctas_needed, max_ctas_allowed)

            # compute capacity provided by those ctas_to_use
            k = ctas_to_use
            cap_by_k = tl.where(
                k <= ctas_high_avail,
                k * cap_high,
                total_high_capacity + (k - ctas_high_avail) * cap_low,
            )
            covered = tl.minimum(remaining, cap_by_k)

            # final last_cta after loop is start_cta + number_of_iterations_performed
            last_cta = start_cta + ctas_to_use
            last_cta = tl.where(ctas_to_use == 0, start_cta - 1, last_cta)

            # Next, load nonHost partial restult
            temp_pid = current_pid

            for cta in range((xcd_pid + 1), last_cta):
                # last_cta is calculated using xcd local pid
                # locks, mp/lp/op are referenced using global pid
                temp_pid = temp_pid + 1
                # According to treamK gemm, atomic_cas is universal solution but less performant
                # while tl.atomic_cas(locks + cta, 1, 1) != 1:
                while (
                    tl.load(locks + temp_pid, cache_modifier=".cv", volatile=True) != 1
                ):
                    pass
                # Partial results are stored in [nonHost, Host-nonFinishing] layout
                offs_mplp = temp_pid * BLOCK_M + offs_m
                mp_ptrs = Mp + offs_mplp
                lp_ptrs = Lp + offs_mplp
                op_ptrs0 = (
                    Op
                    + temp_pid * stride_oph
                    + offs_m[:, None] * stride_opm
                    + tl.arange(0, HEAD_DIM // 2)[None, :] * stride_opn
                )
                op_ptrs1 = (
                    Op
                    + temp_pid * stride_oph
                    + offs_m[:, None] * stride_opm
                    + (tl.arange(0, HEAD_DIM // 2)[None, :] + HEAD_DIM // 2)
                    * stride_opn
                )

                m_cta = tl.load(mp_ptrs, cache_modifier=".cv")
                l_cta = tl.load(lp_ptrs, cache_modifier=".cv")
                # acc_cta = tl.load(op_ptrs)
                acc_cta0 = tl.load(
                    tl.multiple_of(op_ptrs0, (1, 16)), cache_modifier=".cv"
                )
                acc_cta1 = tl.load(
                    tl.multiple_of(op_ptrs1, (1, 16)), cache_modifier=".cv"
                )

                # m_i is the host CTA's m, m_cta is other nonHost CTA's m
                m_new = tl.maximum(m_cta, m_i)
                alpha = tl.math.exp2(m_cta - m_new)
                alpha1 = tl.math.exp2(m_i - m_new)
                l_new = alpha * l_cta + alpha1 * l_i
                # acc = acc_cta * alpha[:, None] + acc * alpha1[:, None]
                acc0 = acc_cta0 * alpha[:, None] + acc0 * alpha1[:, None]
                acc1 = acc_cta1 * alpha[:, None] + acc1 * alpha1[:, None]
                # update m, l
                m_i = m_new
                l_i = l_new

        # host CTA write final result to memory
        # acc = acc / l_i[:, None]
        # tl.store(o_ptrs, acc.to(Out.type.element_ty))
        o_ptrs0 = (
            Out
            + q_idx * BLOCK_M * stride_om
            + tile_head_idx_global * stride_oh
            + offs_m[:, None] * stride_om
            + tl.arange(0, HEAD_DIM // 2)[None, :] * stride_on
        )
        o_ptrs1 = (
            Out
            + q_idx * BLOCK_M * stride_om
            + tile_head_idx_global * stride_oh
            + offs_m[:, None] * stride_om
            + (tl.arange(0, HEAD_DIM // 2)[None, :] + HEAD_DIM // 2) * stride_on
        )

        acc0 = acc0 / l_i[:, None]
        acc1 = acc1 / l_i[:, None]
        tl.store(o_ptrs0, acc0.to(Out.type.element_ty))
        tl.store(o_ptrs1, acc1.to(Out.type.element_ty))

    # update iter
    iter = iter + (local_iter_end - local_iter)

    return iter
