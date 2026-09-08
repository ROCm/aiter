# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx950 H32/H64 D128/KVB64 preshuffled paged FP8 MQA-logits kernel.

Public decode path for ``flydsl_fp8_paged_mqa_logits``. Supports H in {32, 64},
``next_n`` in {1, 2}, and KVBlockSize=64 with a preshuffled KV cache.
"""

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

Vec = fx.Vector

SUPPORTED_HEADS = (32, 64)
HEAD_DIM = 128
INDEX_DIM = HEAD_DIM + 4
KV_BLOCK_SIZE = 64
MFMA_M = MFMA_N = 16
DREG = 4
B_RING = KV_BLOCK_SIZE // MFMA_N
BLOCK_I32 = KV_BLOCK_SIZE * INDEX_DIM // 4
SCALE_I32 = KV_BLOCK_SIZE * HEAD_DIM // 4
TILE_I32 = HEAD_DIM * MFMA_N // 4
HALF_I32 = MFMA_N * 16 // 4
LANE_K_I32 = (HEAD_DIM // (64 // MFMA_N)) * MFMA_N // 4
# Per physical page: 4 tiles × (2× dwordx4 K + 1× f32 scale).
PAGE_VMEM_LOADS = B_RING * 3
_NEUTRAL_E8M0 = 0x7F7F7F7F

_GFX950 = "gfx950"
DEFAULT_COMPILE_HINTS = {
    "waves_per_eu": 2,
    "fast_fp_math": True,
}


def wait_vmcnt(n):
    """Explicit s_waitcnt vmcnt(n). Leaves n VM ops outstanding."""
    rocdl.sched_barrier(0)
    rocdl.s_waitcnt(vmcnt=int(n))
    rocdl.sched_barrier(0)


# Via Uint32 because `//` on a signed Int32 lowers to floordivsi.
def udiv(a, b):
    return fx.Int32(fx.Uint32(a) // fx.Uint32(b))


def umod(a, b):
    return fx.Int32(fx.Uint32(a) % fx.Uint32(b))


def uceildiv(a, b):
    a, b = fx.Int32(a), fx.Int32(b)
    return fx.Int32((fx.Uint32(a) + fx.Uint32(b) - 1) // fx.Uint32(b))


def imin(a, b):
    a, b = fx.Int32(a), fx.Int32(b)
    return (a <= b).select(a, b)


def guarded_store(pred, store_fn):
    """Predicated side effect; masked buffer stores can fault on gfx950."""

    @flyc.jit
    def _guarded(_pred=pred, _store=store_fn):
        if _pred:
            _store()

    _guarded()


def _concat_i32x4(lo, hi):
    return Vec(lo).shuffle(Vec(hi), list(range(DREG * 2)))


def load_q_pack(q_i32, byte_base, lane_div_16):
    """Load one row's lane-owned contiguous K32 segment as a 256-bit operand."""
    off = (byte_base + lane_div_16 * 32) // fx.Int32(4)
    lo = buffer_ops.buffer_load(q_i32.rsrc, off, vec_width=4, dtype=T.i32)
    hi = buffer_ops.buffer_load(q_i32.rsrc, off + 4, vec_width=4, dtype=T.i32)
    return _concat_i32x4(lo, hi)


def load_preshuffled_k_pack(kv_i32, physical, tile_in_page, lane_mod_16, lane_div_16):
    """Load a shuffle_weight(16,16) K column as an i32x8 MFMA operand."""
    base = (
        physical * BLOCK_I32
        + fx.Int32(tile_in_page * TILE_I32)
        + lane_div_16 * LANE_K_I32
        + lane_mod_16 * 4
    )
    lo = buffer_ops.buffer_load(
        kv_i32.rsrc, base, vec_width=4, dtype=T.i32, cache_modifier=2
    )
    hi = buffer_ops.buffer_load(
        kv_i32.rsrc,
        base + HALF_I32,
        vec_width=4,
        dtype=T.i32,
        cache_modifier=2,
    )
    return _concat_i32x4(lo, hi)


def load_kv_scale(kv_i32, physical, token_in_page):
    off = physical * BLOCK_I32 + SCALE_I32 + token_in_page
    return fx.Float32(
        buffer_ops.buffer_load(
            kv_i32.rsrc, off, vec_width=1, dtype=T.f32, cache_modifier=2
        )
    )


def mfma_scores(mma, a_tiles, b_pack, *, m_tiles):
    """Return one 4-f32 score fragment per 16-head M tile."""
    neutral = fx.Int32(_NEUTRAL_E8M0)
    b_frag = fx.make_rmem_tensor(DREG * 2, fx.Int32)
    b_frag.store(b_pack)
    scores = []
    for mi in range_constexpr(m_tiles):
        a_frag = fx.make_rmem_tensor(DREG * 2, fx.Int32)
        acc = fx.make_rmem_tensor(DREG, fx.Float32)
        a_frag.store(a_tiles[mi])
        acc.store(Vec.filled(DREG, 0.0, fx.Float32))
        fx.gemm(
            mma,
            acc,
            a_frag,
            b_frag,
            acc,
            scale_a=neutral,
            scale_b=neutral,
        )
        scores.append(acc.load())
    return scores


def reduce_scores(scores, weights, kv_scale, *, m_tiles):
    """ReLU, weighted H reduction, positive KV scale, then wave reduction."""
    zero = fx.Float32(0.0)
    total = zero
    for mi in range_constexpr(m_tiles):
        frag = Vec(scores[mi])
        for ii in range_constexpr(DREG):
            total = total + fx.Float32(frag[ii]).maximumf(zero) * weights[mi][ii]
    total = total * kv_scale
    total = total + total.shuffle_xor(16, 64)
    return total + total.shuffle_xor(32, 64)


def schedule_mfma_valu_pairs(*, m_tiles):
    """Pair each 32-cycle MFMA with the prior tile's 12-op VALU fragment."""
    for _ in range_constexpr(m_tiles):
        rocdl.sched_mfma(1)
        rocdl.sched_group_barrier("valu", DREG * 3, 0)


@lru_cache(maxsize=8)
def device_cu_count(device_index: int) -> int:
    try:
        return torch.cuda.get_device_properties(device_index).multi_processor_count
    except Exception:  # noqa: BLE001
        return 256


def _make_out_row_view(logits, stride_out, row):
    """Fold the row byte offset into the pointer using i64 arithmetic."""
    byte = fx.Int64(fx.Uint32(row)) * fx.Int64(fx.Uint32(stride_out)) * 4
    return GTensor(
        logits,
        dtype=T.f32,
        shape=(-1,),
        cache_modifier=2,
        static_bytes_offset_i64=byte,
    )


def _build_kernel(*, num_heads: int, next_n: int):
    m_tiles = num_heads // MFMA_M
    n_rows = int(next_n)
    store_vmem = n_rows
    kernel_name = (
        f"fp8_paged_mqa_logits_gfx950_H{num_heads}_D{HEAD_DIM}_"
        f"bkv64_kvb{KV_BLOCK_SIZE}_r{B_RING}_nq{n_rows}_db2_ws1_pfp_sb0_nt_ps_flydsl"
    )

    @flyc.kernel(name=kernel_name, known_block_size=[64, 1, 1])
    def kernel(
        Q: fx.Tensor,
        KV_cache: fx.Tensor,
        weights: fx.Tensor,
        out_logits: fx.Tensor,
        context_lens: fx.Tensor,
        kv_indices: fx.Tensor,
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
        mma = fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(MFMA_M, MFMA_N, HEAD_DIM, fx.Float8E4M3FN)
        )

        q_i32 = GTensor(Q, dtype=T.i32, shape=(-1,))
        kv_i32 = GTensor(KV_cache, dtype=T.i32, shape=(-1,))
        weight_t = GTensor(weights, dtype=T.f32, shape=(-1,))
        context_t = GTensor(context_lens, dtype=T.i32, shape=(-1,))
        table_t = GTensor(kv_indices, dtype=T.i32, shape=(-1,))

        context_len = fx.Int32(context_t[pid_batch])
        page_count = uceildiv(context_len, fx.Int32(KV_BLOCK_SIZE))
        pages_per_split = uceildiv(page_count, split_kv)
        page_lo = pid_split * pages_per_split
        page_hi = imin(page_lo + pages_per_split, page_count)
        col_lo = page_lo * KV_BLOCK_SIZE
        col_hi = page_hi * KV_BLOCK_SIZE

        next_n_c = fx.Int32(n_rows)
        out_rows = [
            _make_out_row_view(
                out_logits, stride_out, pid_batch * next_n_c + fx.Int32(row)
            )
            for row in range_constexpr(n_rows)
        ]
        a_rows = [
            [None for _ in range_constexpr(m_tiles)] for _ in range_constexpr(n_rows)
        ]
        w_rows = [
            [None for _ in range_constexpr(m_tiles)] for _ in range_constexpr(n_rows)
        ]
        for row in range_constexpr(n_rows):
            q_row = fx.Int32(row)
            q_base = pid_batch * stride_q_batch + q_row * stride_q_next_n
            out_row = pid_batch * next_n_c + q_row
            for mi in range_constexpr(m_tiles):
                h = mi * MFMA_M + lane_mod_16
                byte_base = q_base + h * stride_q_heads
                a_rows[row][mi] = load_q_pack(q_i32, byte_base, lane_div_16)
                w_rows[row][mi] = fx.Vector(
                    buffer_ops.buffer_load(
                        weight_t.rsrc,
                        out_row * num_heads + mi * MFMA_M + lane_div_16 * DREG,
                        vec_width=DREG,
                        dtype=T.f32,
                    )
                )

        def _load_physical(page_col):
            table_idx = pid_batch * max_block_len + udiv(
                page_col, fx.Int32(KV_BLOCK_SIZE)
            )
            return fx.Int32(
                buffer_ops.buffer_load(
                    table_t.rsrc, table_idx, vec_width=1, dtype=T.i32, is_scalar=True
                )
            )

        def _issue_physical(physical):
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
                    )
                )
                scale_slots.append(load_kv_scale(kv_i32, physical, token))
            return b_slots, scale_slots

        def _make_page_bank():
            return (
                [
                    fx.make_rmem_tensor(DREG * 2, fx.Int32)
                    for _ in range_constexpr(B_RING)
                ],
                [fx.make_rmem_tensor(1, fx.Float32) for _ in range_constexpr(B_RING)],
            )

        def _store_page_bank(bank, b_slots, scale_slots):
            b_bank, scale_bank = bank
            for slot in range_constexpr(B_RING):
                b_bank[slot].store(b_slots[slot])
                scale_bank[slot].store(
                    fx.Vector.from_elements([scale_slots[slot].ir_value()], fx.Float32)
                )

        def _load_page_bank(bank):
            b_bank, scale_bank = bank
            return (
                [fx.Vector(b_bank[slot].load()) for slot in range_constexpr(B_RING)],
                [
                    fx.Float32(fx.Vector(scale_bank[slot].load())[0])
                    for slot in range_constexpr(B_RING)
                ],
            )

        def _make_physical_bank():
            return fx.make_rmem_tensor(1, fx.Int32)

        def _store_physical(bank, physical):
            bank.store(fx.Vector.from_elements([physical.ir_value()], fx.Int32))

        def _load_physical_bank(bank):
            return fx.Int32(fx.Vector(bank.load())[0])

        def _store_page_row(row, page_col, logits):
            q_row = fx.Int32(row)
            q_limit = context_len - next_n_c + q_row
            col = page_col + lane
            logit = logits[0]
            for slot in range_constexpr(1, B_RING):
                logit = (lane_div_16 == slot).select(logits[slot], logit)
            writer = (col < context_len) & (col <= q_limit)

            def _write(_out=out_rows[row], _col=col, _value=logit):
                _out[_col] = _value

            guarded_store(writer, _write)

        def _compute_page(page_col, b_slots, scale_slots):
            prev_scores = None
            prev_scale = None
            page_logits = [
                [None for _ in range_constexpr(B_RING)] for _ in range_constexpr(n_rows)
            ]
            for slot in range_constexpr(B_RING):
                scores = [
                    mfma_scores(mma, a_rows[row], b_slots[slot], m_tiles=m_tiles)
                    for row in range_constexpr(n_rows)
                ]
                if slot > 0:
                    schedule_mfma_valu_pairs(m_tiles=m_tiles)
                    for row in range_constexpr(n_rows):
                        page_logits[row][slot - 1] = reduce_scores(
                            prev_scores[row],
                            w_rows[row],
                            prev_scale,
                            m_tiles=m_tiles,
                        )
                prev_scores = scores
                prev_scale = scale_slots[slot]
            schedule_mfma_valu_pairs(m_tiles=m_tiles)
            for row in range_constexpr(n_rows):
                page_logits[row][B_RING - 1] = reduce_scores(
                    prev_scores[row],
                    w_rows[row],
                    prev_scale,
                    m_tiles=m_tiles,
                )
                _store_page_row(row, page_col, page_logits[row])

        if page_lo < page_hi:
            current_bank = _make_page_bank()
            next_bank = _make_page_bank()
            b_init, scale_init = _issue_physical(_load_physical(col_lo))
            _store_page_bank(current_bank, b_init, scale_init)
            current_physical = _make_physical_bank()
            next_physical = _make_physical_bank()
            if col_lo + fx.Int32(KV_BLOCK_SIZE) < col_hi:
                _store_physical(
                    next_physical,
                    _load_physical(col_lo + fx.Int32(KV_BLOCK_SIZE)),
                )
            page_col = col_lo
            while page_col + fx.Int32(KV_BLOCK_SIZE) < col_hi:
                next_b, next_scale = _issue_physical(_load_physical_bank(next_physical))
                _store_page_bank(next_bank, next_b, next_scale)
                has_following = page_col + fx.Int32(2 * KV_BLOCK_SIZE) < col_hi
                if has_following:
                    _store_physical(
                        current_physical,
                        _load_physical(page_col + fx.Int32(2 * KV_BLOCK_SIZE)),
                    )
                # Leave the next-page gather outstanding while consuming current.
                wait_vmcnt(PAGE_VMEM_LOADS)
                current_b, current_scale = _load_page_bank(current_bank)
                _compute_page(page_col, current_b, current_scale)

                if has_following:
                    following_b, following_scale = _issue_physical(
                        _load_physical_bank(current_physical)
                    )
                    _store_page_bank(current_bank, following_b, following_scale)
                    if page_col + fx.Int32(3 * KV_BLOCK_SIZE) < col_hi:
                        _store_physical(
                            next_physical,
                            _load_physical(page_col + fx.Int32(3 * KV_BLOCK_SIZE)),
                        )
                    # following-page loads plus current-page stores can remain.
                    wait_vmcnt(PAGE_VMEM_LOADS + store_vmem)
                else:
                    # Only current-page stores may remain.
                    wait_vmcnt(store_vmem)
                next_b, next_scale = _load_page_bank(next_bank)
                _compute_page(page_col + fx.Int32(KV_BLOCK_SIZE), next_b, next_scale)
                page_col = page_col + fx.Int32(2 * KV_BLOCK_SIZE)

            if page_col < col_hi:
                wait_vmcnt(0)
                current_b, current_scale = _load_page_bank(current_bank)
                _compute_page(page_col, current_b, current_scale)

    @flyc.jit
    def launch(
        Q,
        KV_cache,
        weights,
        out_logits,
        context_lens,
        kv_indices,
        grid_blocks,
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
def _compile(*, num_heads: int, next_n: int):
    return _build_kernel(num_heads=num_heads, next_n=next_n)


def flydsl_fp8_paged_mqa_logits(
    q_fp8,
    kv_cache,
    weights,
    out_logits,
    context_lens,
    kv_indices,
    max_model_len,
    *,
    Preshuffle=True,
    KVBlockSize=KV_BLOCK_SIZE,
    SplitKV=None,
    TotalCuCount=None,
    stream=None,
):
    """Paged FP8 MQA logits (decode) for H in {32, 64}, next_n in {1, 2}, KVB=64.

    Drop-in for the Triton ``deepgemm_fp8_paged_mqa_logits`` tensor contract on
    the gfx950 16x16x128 mapping. Requires a preshuffled ``shuffle_weight(16,16)``
    KV cache.
    """
    if get_gfx() != _GFX950:
        raise RuntimeError(f"gfx950 kernel requested on {get_gfx()}")
    if not Preshuffle:
        raise ValueError("requires Preshuffle=True")
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
    _, block_size, one, index_dim = kv_cache.shape
    if block_size != KV_BLOCK_SIZE or one != 1 or index_dim != INDEX_DIM:
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
    launcher = _compile(num_heads=int(heads), next_n=int(next_n))
    launcher.compile_hints = dict(DEFAULT_COMPILE_HINTS)
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


flydsl_fp8_paged_mqa_logits_gfx950 = flydsl_fp8_paged_mqa_logits
