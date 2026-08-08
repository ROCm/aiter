# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon MLA decode kernel for gfx942 (CDNA3)."""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.gluon.mla_gluon import _mla_softmax_reducev_kernel
from aiter.ops.triton.utils._triton import arch_info

LOG2E = tl.constexpr(1.4426950408889634)


@gluon.jit
def _mla_decode_gfx942_kernel(
    Q_nope,
    Q_pe,
    Kv_c_cache,
    K_pe_cache,
    Req_to_tokens,
    B_seq_len,
    O,
    sm_scale,
    kv_scale,
    stride_q_nope_bs,
    stride_q_nope_s,
    stride_q_nope_h,
    stride_q_pe_bs,
    stride_q_pe_s,
    stride_q_pe_h,
    stride_kv_c_bs,
    stride_k_pe_bs,
    stride_req_to_tokens_bs,
    stride_o_b,
    stride_o_s,
    stride_o_h,
    stride_o_split,
    Mid_lse,
    stride_mid_lse_b,
    stride_mid_lse_s,
    stride_mid_lse_h,
    stride_mid_lse_split,
    BLOCK_H: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_KV_SPLITS: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    HEAD_DIM_CKV: gl.constexpr,
    HEAD_DIM_KPE: gl.constexpr,
    KV_PE_OFFSET: gl.constexpr,
    USE_2D_VIEW: gl.constexpr,
    WITHIN_2GB: gl.constexpr,
    NHEAD: gl.constexpr,
    QLEN: gl.constexpr,
    HAS_PE: gl.constexpr,
    CAUSAL: gl.constexpr,
    FUSE_QH: gl.constexpr,
):
    # FUSE_QH flattens QLEN * NHEAD into the native 16-row MFMA dimension so
    # rows from adjacent query positions reuse the LDS KV tile.
    cur_batch = gl.program_id(0)
    split_kv_id = gl.program_id(1)
    cur_q_tile = gl.program_id(2)

    if USE_2D_VIEW:
        cur_batch_seq_len = gl.load(B_seq_len + cur_batch)
        batch_page_start = cur_batch * stride_req_to_tokens_bs
    else:
        batch_page_start = gl.load(B_seq_len + cur_batch)
        cur_batch_seq_len = gl.load(B_seq_len + cur_batch + 1) - batch_page_start

    kv_len_per_split = cur_batch_seq_len // NUM_KV_SPLITS
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = split_kv_start + kv_len_per_split
    if split_kv_id == NUM_KV_SPLITS - 1:
        split_kv_end = cur_batch_seq_len

    if split_kv_start >= split_kv_end:
        return

    num_iter = gl.cdiv(split_kv_end - split_kv_start, BLOCK_N)

    blocked_q_nope: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[1, 64],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    shared_q_nope: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=8, order=[1, 0]
    )
    blocked_q_pe: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[1, 64],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    shared_q_pe: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=2, max_phase=8, order=[1, 0]
    )

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3,
        instr_shape=[16, 16, 16],
        transposed=True,
        warps_per_cta=[1, 4],
    )

    blocked_kv: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[8, 1],
        threads_per_warp=[64, 1],
        warps_per_cta=[1, 4],
        order=[0, 1],
    )
    shared_kv: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=8, order=[0, 1]
    )
    blocked_kpe: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[8, 1],
        threads_per_warp=[64, 1],
        warps_per_cta=[1, 4],
        order=[0, 1],
    )
    shared_kpe: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=8, order=[0, 1]
    )
    blocked_page: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1],
        threads_per_warp=[64],
        warps_per_cta=[4],
        order=[0],
    )

    mfma_layout_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=4
    )
    mfma_layout_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=4
    )

    dtype = Q_nope.type.element_ty
    kvtype = Kv_c_cache.type.element_ty

    gl.static_assert(PAGE_SIZE == 1)

    # Q is loaded synchronously and remains resident for the KV loop.
    buf_q_nope = gl.allocate_shared_memory(
        dtype, shape=[BLOCK_H, HEAD_DIM_CKV], layout=shared_q_nope
    )
    offs_d_ckv = gl.arange(0, HEAD_DIM_CKV, layout=gl.SliceLayout(0, blocked_q_nope))
    if FUSE_QH:
        cur_qh = cur_q_tile * BLOCK_H + gl.arange(
            0, BLOCK_H, layout=gl.SliceLayout(1, blocked_q_nope)
        )
        q_pos = cur_qh // NHEAD
        cur_head = cur_qh % NHEAD
        q_mask = cur_qh < QLEN * NHEAD
        offs_q_nope = (
            cur_batch * stride_q_nope_bs
            + q_pos[:, None] * stride_q_nope_s
            + cur_head[:, None] * stride_q_nope_h
            + offs_d_ckv[None, :]
        )
    else:
        cur_head = gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blocked_q_nope))
        q_mask = cur_head < NHEAD
        offs_q_nope = (
            cur_batch * stride_q_nope_bs
            + cur_q_tile * stride_q_nope_s
            + cur_head[:, None] * stride_q_nope_h
            + offs_d_ckv[None, :]
        )
    q_nope_val = gl.amd.cdna3.buffer_load(
        ptr=Q_nope, offsets=offs_q_nope, mask=q_mask[:, None], other=0.0
    )
    buf_q_nope.store(q_nope_val)

    if HAS_PE:
        buf_q_pe = gl.allocate_shared_memory(
            dtype, shape=[BLOCK_H, HEAD_DIM_KPE], layout=shared_q_pe
        )
        offs_d_kpe = gl.arange(0, HEAD_DIM_KPE, layout=gl.SliceLayout(0, blocked_q_pe))
        if FUSE_QH:
            cur_qh_qpe = cur_q_tile * BLOCK_H + gl.arange(
                0, BLOCK_H, layout=gl.SliceLayout(1, blocked_q_pe)
            )
            q_pos_qpe = cur_qh_qpe // NHEAD
            cur_head_qpe = cur_qh_qpe % NHEAD
            q_pe_mask = cur_qh_qpe < QLEN * NHEAD
            offs_q_pe = (
                cur_batch * stride_q_pe_bs
                + q_pos_qpe[:, None] * stride_q_pe_s
                + cur_head_qpe[:, None] * stride_q_pe_h
                + offs_d_kpe[None, :]
            )
        else:
            cur_head_qpe = gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blocked_q_pe))
            q_pe_mask = cur_head_qpe < NHEAD
            offs_q_pe = (
                cur_batch * stride_q_pe_bs
                + cur_q_tile * stride_q_pe_s
                + cur_head_qpe[:, None] * stride_q_pe_h
                + offs_d_kpe[None, :]
            )
        q_pe_val = gl.amd.cdna3.buffer_load(
            ptr=Q_pe,
            offsets=offs_q_pe,
            mask=q_pe_mask[:, None],
            other=0.0,
        )
        buf_q_pe.store(q_pe_val)

    e_max = gl.zeros(
        [BLOCK_H], dtype=gl.float32, layout=gl.SliceLayout(1, mfma_layout)
    ) - float("inf")
    e_sum = gl.zeros([BLOCK_H], dtype=gl.float32, layout=gl.SliceLayout(1, mfma_layout))
    acc = gl.zeros([BLOCK_H, HEAD_DIM_CKV], dtype=gl.float32, layout=mfma_layout)

    qk_scale = sm_scale * kv_scale

    q_nope = buf_q_nope.load(mfma_layout_a)
    if HAS_PE:
        q_pe = buf_q_pe.load(mfma_layout_a)

    # CDNA3 uses one synchronous LDS tile. Loading and consuming each tile
    # serially keeps the allocation within the 64 KiB LDS budget.
    buf_kv = gl.allocate_shared_memory(
        kvtype, shape=[1, HEAD_DIM_CKV, BLOCK_N], layout=shared_kv
    )
    if HAS_PE:
        buf_kpe = gl.allocate_shared_memory(
            kvtype, shape=[1, HEAD_DIM_KPE, BLOCK_N], layout=shared_kpe
        )

    offs_page_raw = gl.arange(0, BLOCK_N, layout=blocked_page)
    offs_d_ckv_1 = gl.arange(0, HEAD_DIM_CKV, layout=gl.SliceLayout(1, blocked_kv))
    offs_n_kv = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked_kv))
    if HAS_PE:
        offs_d_kpe_1 = gl.arange(0, HEAD_DIM_KPE, layout=gl.SliceLayout(1, blocked_kpe))
        offs_n_pe = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked_kpe))

    for i in range(num_iter):
        start_n = split_kv_start + i * BLOCK_N

        offs_n_page = start_n + offs_page_raw
        offs_page = batch_page_start + offs_n_page // PAGE_SIZE
        kv_loc = gl.amd.cdna3.buffer_load(
            ptr=Req_to_tokens,
            offsets=offs_page,
            mask=offs_n_page < split_kv_end,
            other=0,
        )
        kv_loc_kv = gl.convert_layout(kv_loc, gl.SliceLayout(0, blocked_kv))

        offs_k_c = kv_loc_kv[None, :] * stride_kv_c_bs + offs_d_ckv_1[:, None]
        kv_mask = (start_n + offs_n_kv)[None, :] < split_kv_end
        if WITHIN_2GB:
            k_c_val = gl.amd.cdna3.buffer_load(
                ptr=Kv_c_cache,
                offsets=offs_k_c,
                mask=kv_mask,
                other=0.0,
            )
        else:
            # buffer_load has a 32-bit byte offset. Use normal pointer
            # arithmetic when the cache can span more than 2 GiB.
            k_c_val = gl.load(Kv_c_cache + offs_k_c, mask=kv_mask, other=0.0)
        buf_kv.index(0).store(k_c_val)

        if HAS_PE:
            kv_loc_pe = gl.convert_layout(kv_loc, gl.SliceLayout(0, blocked_kpe))
            offs_k_pe = (
                kv_loc_pe[None, :] * stride_k_pe_bs
                + offs_d_kpe_1[:, None]
                + KV_PE_OFFSET
            )
            kpe_mask = (start_n + offs_n_pe)[None, :] < split_kv_end
            if WITHIN_2GB:
                k_pe_val = gl.amd.cdna3.buffer_load(
                    ptr=K_pe_cache,
                    offsets=offs_k_pe,
                    mask=kpe_mask,
                    other=0.0,
                )
            else:
                k_pe_val = gl.load(K_pe_cache + offs_k_pe, mask=kpe_mask, other=0.0)
            buf_kpe.index(0).store(k_pe_val)

        k_c = buf_kv.index(0).load(mfma_layout_b)
        zeros = gl.zeros([BLOCK_H, BLOCK_N], dtype=gl.float32, layout=mfma_layout)
        qk = gl.amd.cdna3.mfma(q_nope, k_c.to(dtype), zeros)
        if HAS_PE:
            k_pe = buf_kpe.index(0).load(mfma_layout_b)
            qk = gl.amd.cdna3.mfma(q_pe, k_pe.to(dtype), qk)

        qk = qk * qk_scale

        offs_n_score = start_n + gl.arange(
            0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout)
        )
        if QLEN > 1 and CAUSAL:
            if FUSE_QH:
                cur_qh_score = cur_q_tile * BLOCK_H + gl.arange(
                    0, BLOCK_H, layout=gl.SliceLayout(1, mfma_layout)
                )
                q_pos_score = cur_qh_score // NHEAD
                score_end = gl.minimum(
                    split_kv_end, cur_batch_seq_len - QLEN + q_pos_score + 1
                )
                qk = gl.where(
                    offs_n_score[None, :] < score_end[:, None],
                    qk,
                    float("-inf"),
                )
            else:
                score_end = gl.minimum(
                    split_kv_end,
                    cur_batch_seq_len - QLEN + cur_q_tile + 1,
                )
                qk = gl.where(offs_n_score[None, :] < score_end, qk, float("-inf"))
        else:
            qk = gl.where(offs_n_score[None, :] < split_kv_end, qk, float("-inf"))

        tile_max = gl.max(qk, 1)
        tile_has_values = tile_max != float("-inf")
        n_e_max = gl.where(tile_has_values, gl.maximum(tile_max, e_max), e_max)
        re_scale = gl.where(tile_has_values, gl.exp2((e_max - n_e_max) * LOG2E), 1.0)
        p = gl.where(
            tile_has_values[:, None],
            gl.exp2((qk - n_e_max[:, None]) * LOG2E),
            0.0,
        )
        e_sum = e_sum * re_scale + gl.sum(p, 1)
        acc = acc * re_scale[:, None]
        e_max = n_e_max

        # The cached latent serves as both K_nope and V. Read the LDS tile
        # through a transposed view for P @ V.
        v_c = buf_kv.index(0).permute((1, 0)).load(mfma_layout_b)
        p_operand = gl.convert_layout(p.to(dtype), mfma_layout_a)
        acc = gl.amd.cdna3.mfma(p_operand, v_c.to(dtype), acc)

    offs_d_out = gl.arange(0, HEAD_DIM_CKV, layout=gl.SliceLayout(0, mfma_layout))
    out = gl.where(
        e_sum[:, None] > 0.0,
        acc / e_sum[:, None],
        gl.zeros([BLOCK_H, HEAD_DIM_CKV], dtype=gl.float32, layout=mfma_layout),
    )

    if FUSE_QH:
        cur_qh_out = cur_q_tile * BLOCK_H + gl.arange(
            0, BLOCK_H, layout=gl.SliceLayout(1, mfma_layout)
        )
        q_pos_out = cur_qh_out // NHEAD
        cur_head_out = cur_qh_out % NHEAD
        out_mask = cur_qh_out < QLEN * NHEAD
        offs_o = (
            cur_batch * stride_o_b
            + q_pos_out[:, None] * stride_o_s
            + cur_head_out[:, None] * stride_o_h
            + split_kv_id * stride_o_split
            + offs_d_out[None, :]
        )
    else:
        cur_head_out = gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, mfma_layout))
        out_mask = cur_head_out < NHEAD
        offs_o = (
            cur_batch * stride_o_b
            + cur_q_tile * stride_o_s
            + cur_head_out[:, None] * stride_o_h
            + split_kv_id * stride_o_split
            + offs_d_out[None, :]
        )
    gl.amd.cdna3.buffer_store(
        out.to(O.type.element_ty), O, offs_o, mask=out_mask[:, None]
    )

    if NUM_KV_SPLITS > 1:
        lse = gl.where(e_sum > 0.0, e_max + gl.log(e_sum), float("-inf"))
        if FUSE_QH:
            offs_lse = (
                cur_batch * stride_mid_lse_b
                + q_pos_out * stride_mid_lse_s
                + cur_head_out * stride_mid_lse_h
                + split_kv_id * stride_mid_lse_split
            )
        else:
            offs_lse = (
                cur_batch * stride_mid_lse_b
                + cur_q_tile * stride_mid_lse_s
                + cur_head_out * stride_mid_lse_h
                + split_kv_id * stride_mid_lse_split
            )
        gl.amd.cdna3.buffer_store(lse, Mid_lse, offs_lse, mask=out_mask)


def mla_gluon_gfx942(
    q_nope,
    q_pe,
    kv_c_cache,
    k_pe_cache,
    req_to_tokens,
    b_seq_len,
    o,
    sm_scale,
    kv_scale=1.0,
    num_kv_splits=8,
    page_size=1,
    kv_pe_offset=0,
    block_n=32,
    num_warps=4,
    waves_per_eu=0,
    mid_o=None,
    mid_lse=None,
    use_2d_view=True,
    within_2gb_override=None,
    causal=True,
    reduce_num_warps=8,
    fuse_qlen_heads=False,
):
    """Run the gfx942 MLA decode kernel.

    Query and output tensors may be shaped either ``[B, H, D]`` or
    ``[B, QLEN, H, D]``. With ``use_2d_view=False``, ``req_to_tokens`` is a
    ragged token-index array and ``b_seq_len`` is its ``[B + 1]`` indptr.
    """
    assert q_pe is not None, "q_pe must not be None"
    assert q_pe.dim() == q_nope.dim(), (
        f"q_pe and q_nope must have the same rank, got "
        f"{q_pe.dim()}-D and {q_nope.dim()}-D"
    )

    original_o = o
    if q_nope.dim() == 3:
        q_nope = q_nope.unsqueeze(1)
        q_pe = q_pe.unsqueeze(1)
    if o.dim() == 3:
        o = o.unsqueeze(1)
        if mid_o is not None and mid_o.dim() == 4:
            mid_o = mid_o.unsqueeze(1)
        if mid_lse is not None and mid_lse.dim() == 3:
            mid_lse = mid_lse.unsqueeze(1)

    assert (
        arch_info.get_arch() == "gfx942"
    ), f"mla_gluon_gfx942 requires gfx942, got {arch_info.get_arch()}"
    assert (
        q_nope.dim() == 4
    ), f"q_nope must be 4-D [B, QLEN, H, D], got {q_nope.dim()}-D"
    assert q_pe.shape[:-1] == q_nope.shape[:-1], (
        "q_pe leading dimensions must match q_nope after normalization, got "
        f"{q_pe.shape[:-1]} and {q_nope.shape[:-1]}"
    )
    batch_size, qlen, nhead, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    assert (
        head_dim_ckv == 512
    ), f"mla_gluon_gfx942 requires head_dim_ckv=512, got {head_dim_ckv}"
    assert head_dim_kpe == 64, f"expected head_dim_kpe=64, got {head_dim_kpe}"
    assert page_size == 1, "mla_gluon_gfx942 requires page_size=1"

    block_h = 16
    assert (
        nhead <= block_h
    ), f"mla_gluon_gfx942 handles at most {block_h} heads, got {nhead}"

    has_pe = q_pe is not None

    if num_kv_splits > 1 and mid_o is None:
        mid_o = torch.empty(
            (batch_size, qlen, nhead, num_kv_splits, head_dim_ckv),
            dtype=o.dtype,
            device=o.device,
        )
    if num_kv_splits > 1 and mid_lse is None:
        mid_lse = torch.empty(
            (batch_size, qlen, nhead, num_kv_splits),
            dtype=torch.float32,
            device=o.device,
        )

    out = o if num_kv_splits == 1 else mid_o
    stride_o_split = 0 if num_kv_splits == 1 else out.stride(-2)
    max_kv_bytes = (
        kv_c_cache.shape[0] * kv_c_cache.stride(0) * kv_c_cache.element_size()
    )
    within_2gb = (
        max_kv_bytes <= 0x80000000
        if within_2gb_override is None
        else within_2gb_override
    )

    grid_q = triton.cdiv(qlen * nhead, block_h) if fuse_qlen_heads else qlen
    grid = (batch_size, num_kv_splits, grid_q)

    _mla_decode_gfx942_kernel[grid](
        q_nope,
        q_pe,
        kv_c_cache,
        k_pe_cache,
        req_to_tokens,
        b_seq_len,
        out,
        sm_scale,
        kv_scale,
        q_nope.stride(0),
        q_nope.stride(1) if qlen > 1 else 0,
        q_nope.stride(2),
        q_pe.stride(0),
        q_pe.stride(1) if qlen > 1 else 0,
        q_pe.stride(2),
        kv_c_cache.stride(0),
        k_pe_cache.stride(0),
        req_to_tokens.stride(0) if use_2d_view else 0,
        out.stride(0),
        out.stride(1) if qlen > 1 else 0,
        out.stride(-3) if num_kv_splits > 1 else out.stride(2),
        stride_o_split,
        mid_lse,
        mid_lse.stride(0) if mid_lse is not None else 0,
        (mid_lse.stride(1) if qlen > 1 else 0) if mid_lse is not None else 0,
        mid_lse.stride(2) if mid_lse is not None else 0,
        mid_lse.stride(3) if mid_lse is not None else 0,
        BLOCK_H=block_h,
        BLOCK_N=block_n,
        NUM_KV_SPLITS=num_kv_splits,
        PAGE_SIZE=page_size,
        HEAD_DIM_CKV=head_dim_ckv,
        HEAD_DIM_KPE=head_dim_kpe,
        KV_PE_OFFSET=kv_pe_offset,
        USE_2D_VIEW=use_2d_view,
        WITHIN_2GB=within_2gb,
        NHEAD=nhead,
        QLEN=qlen,
        HAS_PE=has_pe,
        CAUSAL=causal,
        FUSE_QH=fuse_qlen_heads,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
        num_stages=1,
    )

    if num_kv_splits > 1:
        _mla_softmax_reducev_kernel[(batch_size, nhead, qlen)](
            mid_o,
            mid_lse,
            o,
            None,
            b_seq_len,
            mid_o.stride(0),
            mid_o.stride(1) if qlen > 1 else 0,
            mid_o.stride(2),
            mid_o.stride(3),
            mid_lse.stride(0),
            mid_lse.stride(1) if qlen > 1 else 0,
            mid_lse.stride(2),
            mid_lse.stride(3),
            o.stride(0),
            o.stride(1) if qlen > 1 else 0,
            o.stride(2),
            0,
            0,
            0,
            NUM_KV_SPLITS=num_kv_splits,
            HEAD_DIM_CKV=head_dim_ckv,
            HAS_FINAL_LSE=False,
            USE_2D_VIEW=use_2d_view,
            num_warps=reduce_num_warps,
        )
    return original_o


def _graph_launch_config(batch_size: int, qlen: int) -> tuple[int, int, int]:
    """Return ``(num_kv_splits, block_n, waves_per_eu)`` for a graph shape."""
    if qlen == 1:
        if batch_size == 1:
            return 64, 16, 2
        if batch_size <= 4:
            return 64, 32, 0
        if batch_size <= 8:
            return 32, 32, 2
        if batch_size <= 16:
            return 16, 32, 0
        return 9, 32, 2

    if qlen == 4:
        if batch_size == 1:
            return 24, 32, 2
        if batch_size >= 16:
            return 6, 32, 2
        if batch_size >= 8:
            return 10, 32, 2
    if batch_size == 1:
        return 16, 32, 2
    effective_batch = batch_size * qlen
    splits = max(1, min(64, (256 + effective_batch - 1) // effective_batch))
    return splits, 32, 2


def mla_gluon_gfx942_graph(
    q_nope,
    q_pe,
    kv_buffer,
    o,
    page_table,
    seq_info,
    sm_scale,
    within_2gb_override=None,
):
    """Run the graph-capturable combined-cache adapter with ragged metadata."""
    assert q_pe is not None, "q_pe must not be None"
    if q_nope.dim() == 3:
        batch_size, nhead, _ = q_nope.shape
        qlen = 1
    else:
        batch_size, qlen, nhead, _ = q_nope.shape

    assert nhead == 12, f"gfx942 graph path is tuned for 12 heads, got {nhead}"
    assert q_nope.dtype == torch.bfloat16 and q_pe.dtype == torch.bfloat16
    assert kv_buffer.dtype == torch.bfloat16
    assert kv_buffer.shape[-1] == 576

    num_kv_splits, block_n, waves_per_eu = _graph_launch_config(batch_size, qlen)
    fuse_qlen_heads = qlen == 4 and batch_size >= 8
    kv_c = kv_buffer[:, :512]
    k_pe = kv_buffer[:, 512:]
    return mla_gluon_gfx942(
        q_nope,
        q_pe,
        kv_c,
        k_pe,
        page_table,
        seq_info,
        o,
        sm_scale,
        num_kv_splits=num_kv_splits,
        page_size=1,
        kv_pe_offset=0,
        block_n=block_n,
        num_warps=4,
        waves_per_eu=waves_per_eu,
        use_2d_view=False,
        within_2gb_override=within_2gb_override,
        causal=True,
        fuse_qlen_heads=fuse_qlen_heads,
    )
