# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx950 H32/H64 D128/KVB64 preshuffled paged FP8 MQA-logits kernel."""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import arith, range_constexpr, rocdl
from flydsl.expr.typing import T

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype

from .. import buffer_ops
from ..tensor_shim import GTensor, _run_compiled, _to_raw
from ._fp8_paged_mqa_logits_gfx950 import (
    B_RING,
    DREG,
    HEAD_DIM,
    MFMA_M,
    MFMA_N,
    NEXT_N_MAX,
    KV_BLOCK_SIZE,
    PAGE_VMEM_LOADS,
    PAGE_VMEM_STORES,
    SUPPORTED_HEADS,
    guarded_store,
    imin,
    load_kv_scale,
    load_preshuffled_k_pack,
    load_q_pack,
    mfma_scores,
    reduce_scores,
    schedule_mfma_valu_pairs,
    uceildiv,
    wait_vmcnt,
)
from ._mqa_logits_common import (
    DEFAULT_COMPILE_HINTS,
    device_cu_count,
    udiv,
    umod,
)

_GFX950 = "gfx950"


def _unwrap(value):
    return value.ir_value() if hasattr(value, "ir_value") else value


def _build_kernel(*, index_dim: int, num_heads: int):
    m_tiles = num_heads // MFMA_M
    kernel_name = (
        f"fp8_paged_mqa_logits_gfx950_H{num_heads}_D{HEAD_DIM}_"
        f"bkv64_kvb{KV_BLOCK_SIZE}_r{B_RING}_nq{NEXT_N_MAX}_w1_nt_ps_flydsl"
    )

    @flyc.kernel(name=kernel_name, known_block_size=[64, 1, 1])
    def kernel(
        Q: fx.Tensor,
        KV_cache: fx.Tensor,
        weights: fx.Tensor,
        out_logits: fx.Tensor,
        context_lens: fx.Tensor,
        kv_indices: fx.Tensor,
        next_n: fx.Int32,
        batch_size: fx.Int32,
        split_kv: fx.Int32,
        stride_q_batch: fx.Int32,
        stride_q_next_n: fx.Int32,
        stride_q_heads: fx.Int32,
        max_block_len: fx.Int32,
        stride_out: fx.Int32,
    ):
        bid = fx.block_idx.x
        pid_batch = umod(bid, batch_size)
        pid_split = udiv(bid, batch_size)
        lane = umod(fx.thread_idx.x, 64)
        lane_mod_16 = umod(lane, MFMA_N)
        lane_div_16 = udiv(lane, MFMA_N)

        q_i32 = GTensor(Q, dtype=T.i32, shape=(-1,))
        kv_i32 = GTensor(KV_cache, dtype=T.i32, shape=(-1,))
        weight_t = GTensor(weights, dtype=T.f32, shape=(-1,))
        context_t = GTensor(context_lens, dtype=T.i32, shape=(-1,))
        table_t = GTensor(kv_indices, dtype=T.i32, shape=(-1,))
        out_t = GTensor(out_logits, dtype=T.f32, shape=(-1,), cache_modifier=2)

        context_len = fx.Int32(context_t[pid_batch])
        page_count = uceildiv(context_len, fx.Int32(KV_BLOCK_SIZE))
        pages_per_split = uceildiv(page_count, split_kv)
        page_lo = pid_split * pages_per_split
        page_hi = imin(page_lo + pages_per_split, page_count)
        col_lo = page_lo * KV_BLOCK_SIZE
        col_hi = page_hi * KV_BLOCK_SIZE

        a_rows = [
            [None for _ in range_constexpr(m_tiles)]
            for _ in range_constexpr(NEXT_N_MAX)
        ]
        w_rows = [
            [[None for _ in range_constexpr(DREG)] for _ in range_constexpr(m_tiles)]
            for _ in range_constexpr(NEXT_N_MAX)
        ]
        for row in range_constexpr(NEXT_N_MAX):
            q_row = imin(fx.Int32(row), next_n - 1)
            q_base = pid_batch * stride_q_batch + q_row * stride_q_next_n
            out_row = pid_batch * next_n + q_row
            for mi in range_constexpr(m_tiles):
                h = mi * MFMA_M + lane_mod_16
                byte_base = q_base + h * stride_q_heads
                a_rows[row][mi] = load_q_pack(q_i32, byte_base, lane_div_16)
                weight_vec = fx.Vector(
                    buffer_ops.buffer_load(
                        weight_t.rsrc,
                        out_row * num_heads + mi * MFMA_M + lane_div_16 * DREG,
                        vec_width=DREG,
                        dtype=T.f32,
                    )
                )
                for ii in range_constexpr(DREG):
                    w_rows[row][mi][ii] = fx.Float32(weight_vec[ii])

        def _issue_page(page_col):
            table_idx = pid_batch * max_block_len + udiv(page_col, fx.Int32(KV_BLOCK_SIZE))
            physical = fx.Int32(
                buffer_ops.buffer_load(
                    table_t.rsrc, table_idx, vec_width=1, dtype=T.i32, is_scalar=True
                )
            )
            b_slots, scale_slots = [], []
            for slot in range_constexpr(B_RING):
                token = slot * MFMA_N + lane_mod_16
                b_slots.append(
                    load_preshuffled_k_pack(
                        kv_i32,
                        physical,
                        slot,
                        lane_mod_16,
                        lane_div_16,
                        index_dim=index_dim,
                    )
                )
                scale_slots.append(
                    load_kv_scale(kv_i32, physical, token, index_dim=index_dim)
                )
            return b_slots, scale_slots

        def _pack_state(b_slots, scale_slots):
            return [_unwrap(v) for v in b_slots + scale_slots]

        def _unpack_state(state):
            b_slots = [fx.Vector(state[i]) for i in range_constexpr(B_RING)]
            scale_slots = [
                fx.Float32(state[B_RING + i]) for i in range_constexpr(B_RING)
            ]
            return b_slots, scale_slots

        def _store_row(row, col, scores, scale):
            q_row = fx.Int32(row)
            q_limit = context_len - next_n + q_row
            out_row = pid_batch * next_n + q_row
            logit = reduce_scores(scores, w_rows[row], scale, m_tiles=m_tiles)
            writer = (
                (lane_div_16 == 0)
                & (q_row < next_n)
                & (col < context_len)
                & (col <= q_limit)
            )
            def _write(_row=out_row, _col=col, _value=logit):
                out_t[_row * stride_out + _col] = _value

            guarded_store(writer, _write)

        def _compute_page(page_col, b_slots, scale_slots):
            rocdl.sched_barrier(0)
            prev_scores = None
            prev_col = None
            prev_scale = None
            for slot in range_constexpr(B_RING):
                col = page_col + slot * MFMA_N + lane_mod_16
                scores = [
                    mfma_scores(a_rows[row], b_slots[slot], m_tiles=m_tiles)
                    for row in range_constexpr(NEXT_N_MAX)
                ]
                if slot > 0:
                    schedule_mfma_valu_pairs(m_tiles=m_tiles)
                    for row in range_constexpr(NEXT_N_MAX):
                        _store_row(row, prev_col, prev_scores[row], prev_scale)
                prev_scores = scores
                prev_col = col
                prev_scale = scale_slots[slot]
            schedule_mfma_valu_pairs(m_tiles=m_tiles)
            for row in range_constexpr(NEXT_N_MAX):
                _store_row(row, prev_col, prev_scores[row], prev_scale)
            rocdl.sched_barrier(0)

        if page_lo < page_hi:
            b_init, scale_init = _issue_page(col_lo)
            if page_lo + 1 < page_hi:
                stop = col_hi - KV_BLOCK_SIZE
                init_state = _pack_state(b_init, scale_init)
                for page_col, state in range(
                    col_lo, stop, fx.Int32(KV_BLOCK_SIZE), init=init_state
                ):
                    current_b, current_scale = _unpack_state(state)
                    next_b, next_scale = _issue_page(
                        fx.Int32(page_col) + fx.Int32(KV_BLOCK_SIZE)
                    )
                    # 12 next-page loads in flight: wait until those 12 remain so
                    # the current page (or leftover stores) has retired.
                    wait_vmcnt(PAGE_VMEM_LOADS)
                    _compute_page(fx.Int32(page_col), current_b, current_scale)
                    wait_vmcnt(PAGE_VMEM_STORES)
                    results = yield _pack_state(next_b, next_scale)
                final_b, final_scale = _unpack_state(results)
                _compute_page(stop, final_b, final_scale)
            else:
                wait_vmcnt(0)
                _compute_page(col_lo, b_init, scale_init)

    @flyc.jit
    def launch(
        Q,
        KV_cache,
        weights,
        out_logits,
        context_lens,
        kv_indices,
        grid_blocks,
        next_n,
        batch_size,
        split_kv,
        stride_q_batch,
        stride_q_next_n,
        stride_q_heads,
        max_block_len,
        stride_out,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel._func.__name__ = kernel_name
        kernel(
            Q,
            KV_cache,
            weights,
            out_logits,
            context_lens,
            kv_indices,
            next_n,
            batch_size,
            split_kv,
            stride_q_batch,
            stride_q_next_n,
            stride_q_heads,
            max_block_len,
            stride_out,
        ).launch(
            grid=(arith.index_cast(T.index, _to_raw(grid_blocks)), 1, 1),
            block=(64, 1, 1),
            stream=stream,
        )

    launch.compile_hints = dict(DEFAULT_COMPILE_HINTS)
    launch.kernel_name = kernel_name
    return launch


@lru_cache(maxsize=8)
def _compile(*, index_dim: int, num_heads: int):
    return _build_kernel(index_dim=index_dim, num_heads=num_heads)


def flydsl_fp8_paged_mqa_logits_gfx950(
    q_fp8,
    kv_cache,
    weights,
    out_logits,
    context_lens,
    kv_indices,
    max_model_len,
    *,
    KVBlockSize=KV_BLOCK_SIZE,
    SplitKV=None,
    TotalCuCount=None,
    stream=None,
):
    """Run the H32/H64 D128/KVB64/preshuffle gfx950 mapping."""
    if get_gfx() != _GFX950:
        raise RuntimeError(f"gfx950 kernel requested on {get_gfx()}")
    batch_size, next_n, heads, head_dim = q_fp8.shape
    if heads not in SUPPORTED_HEADS or (head_dim, int(KVBlockSize)) != (
        HEAD_DIM,
        KV_BLOCK_SIZE,
    ):
        raise ValueError("requires H in {32, 64}, D=128, KVBlockSize=64")
    if next_n not in (1, 2):
        raise ValueError("requires next_n in {1, 2}")
    if q_fp8.dtype != get_fp8_e4m3_dtype():
        raise ValueError(f"q_fp8 must be native FP8 E4M3, got {q_fp8.dtype}")
    if kv_cache.dtype != torch.uint8:
        raise ValueError("kv_cache must contain preshuffled uint8 FP8 data")
    num_blocks, block_size, one, index_dim = kv_cache.shape
    if block_size != KV_BLOCK_SIZE or one != 1 or index_dim != HEAD_DIM + 4:
        raise ValueError(f"unexpected KV cache shape {tuple(kv_cache.shape)}")

    context_lens = context_lens.reshape(batch_size)
    weights = weights.reshape(batch_size * next_n, heads)
    max_block_len = kv_indices.shape[-1]
    kv_indices = kv_indices.reshape(batch_size, max_block_len)
    total_cu = (
        device_cu_count(q_fp8.device.index)
        if TotalCuCount is None
        else int(TotalCuCount)
    )
    max_pages = max(1, (int(max_model_len) + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE)
    split_kv = (
        max(
            1,
            min(
                max_pages,
                (total_cu * 4 * 2 + batch_size - 1) // batch_size,
            ),
        )
        if SplitKV is None or int(SplitKV) <= 0
        else max(1, min(max_pages, int(SplitKV)))
    )
    grid_blocks = batch_size * split_kv
    launcher = _compile(index_dim=int(index_dim), num_heads=int(heads))
    launcher.compile_hints = {**DEFAULT_COMPILE_HINTS, "waves_per_eu": 2}
    stream = stream or torch.cuda.current_stream(q_fp8.device)
    with torch.cuda.device(q_fp8.device.index):
        _run_compiled(
            launcher,
            q_fp8,
            kv_cache.reshape(-1),
            weights,
            out_logits,
            context_lens,
            kv_indices,
            int(grid_blocks),
            int(next_n),
            int(batch_size),
            int(split_kv),
            int(q_fp8.stride(0)),
            int(q_fp8.stride(1)),
            int(q_fp8.stride(2)),
            int(max_block_len),
            int(out_logits.stride(0)),
            stream,
        )
    return out_logits
