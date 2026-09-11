# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx950 H32/D128/KVB64 ragged paged FP8 MQA-logits kernel.

One 8-wave workgroup owns one ``(sequence, SplitKV slice)``. Waves 0..3
compute the four N=16 tiles in one page and replay every live query row.
Waves 4..7 stream the next page directly into the idle LDS bank.
"""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import arith, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype

from .. import buffer_ops
from ..mxfp4_gemm_common import _lds_ptr3
from ..tensor_shim import GTensor, _run_compiled, _to_raw

Vec = fx.Vector

MAX_NN = 8
NUM_HEADS = 32
HEAD_DIM = 128
INDEX_DIM = HEAD_DIM + 4
KV_BLOCK_SIZE = 64
MFMA_M = MFMA_N = 16
MFMA_K = HEAD_DIM
M_TILES = NUM_HEADS // MFMA_M
DREG = 4
B_RING = KV_BLOCK_SIZE // MFMA_N
TILE_I32 = HEAD_DIM * MFMA_N // 4
HALF_I32 = MFMA_N * 16 // 4
LANE_K_I32 = (HEAD_DIM // (64 // MFMA_N)) * MFMA_N // 4

THREADS = 512
WAVES = THREADS // 64
COMPUTE_WAVES = 4

PAGE_BYTES = KV_BLOCK_SIZE * INDEX_DIM
PAGE_CHUNKS = PAGE_BYTES // 16
K_BANKS = 2
K_LDS_BYTES = K_BANKS * PAGE_BYTES
Q_ROW_BYTES = NUM_HEADS * HEAD_DIM
Q_LDS_BYTES = MAX_NN * Q_ROW_BYTES
W_ROW_BYTES = NUM_HEADS * 4
W_LDS_BYTES = MAX_NN * W_ROW_BYTES
LDS_BYTES = K_LDS_BYTES + Q_LDS_BYTES + W_LDS_BYTES
Q_LDS_OFF = K_LDS_BYTES
W_LDS_OFF = Q_LDS_OFF + Q_LDS_BYTES

_NEUTRAL_E8M0 = 0x7F7F7F7F
_GFX950 = "gfx950"
DEFAULT_COMPILE_HINTS = {"waves_per_eu": 2, "fast_fp_math": True}


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


def _wait_all():
    rocdl.sched_barrier(0)
    rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
    rocdl.sched_barrier(0)


def _guarded(pred, fn):
    @flyc.jit
    def _do(_pred=pred, _fn=fn):
        if _pred:
            _fn()

    _do()


def _lds_f32x4(base_i32, byte_off):
    ptr = fx.inttoptr(
        fx.PointerType.get(T.f32, fx.AddressSpace.Shared, 16),
        fx.Int32(base_i32 + byte_off),
    )
    return fx.ptr_load(ptr, result_type=T.vec(4, T.f32))


def _lds_f32(base_i32, byte_off):
    ptr = fx.inttoptr(
        fx.PointerType.get(T.f32, fx.AddressSpace.Shared, 4),
        fx.Int32(base_i32 + byte_off),
    )
    return fx.Float32(fx.ptr_load(ptr))


def _concat_i32x4(lo, hi):
    return Vec(lo).shuffle(Vec(hi), list(range(8)))


def _lds_i32x4(base_i32, byte_off):
    ptr = fx.inttoptr(
        fx.PointerType.get(T.i32, fx.AddressSpace.Shared, 16),
        fx.Int32(base_i32 + byte_off),
    )
    return fx.ptr_load(ptr, result_type=T.vec(4, T.i32))


def _load_q_pack_lds(q_base, row, head, lane_div_16):
    """Lane-owned contiguous K32 segment for a 16x16x128 A operand."""
    base = q_base + row * Q_ROW_BYTES + head * HEAD_DIM + lane_div_16 * 32
    return _concat_i32x4(_lds_i32x4(base, 0), _lds_i32x4(base, 16))


def _load_k_pack_lds(k_base, tile, lane_mod_16, lane_div_16):
    """Lane-owned B operand from one shuffle_weight(16,16) N=16 tile."""
    base_i32 = tile * TILE_I32 + lane_div_16 * LANE_K_I32 + lane_mod_16 * 4
    lo = _lds_i32x4(k_base, base_i32 * 4)
    hi = _lds_i32x4(k_base, (base_i32 + HALF_I32) * 4)
    return _concat_i32x4(lo, hi)


def _load_weight_frag_lds(w_base, row, mi, lane_div_16):
    return Vec(
        _lds_f32x4(
            w_base,
            row * W_ROW_BYTES + (mi * MFMA_M + lane_div_16 * DREG) * 4,
        )
    )


def _mfma_score(a_pack, b_pack):
    result_ty = Vec.make_type(DREG, fx.Float32)
    neutral = arith.constant(_NEUTRAL_E8M0, type=T.i32)
    return Vec(
        rocdl.mfma_scale_f32_16x16x128_f8f6f4(
            result_ty,
            [
                a_pack,
                b_pack,
                Vec.filled(DREG, 0.0, fx.Float32),
                0,
                0,
                0,
                neutral,
                0,
                neutral,
            ],
        )
    )


def _reduce_scores(scores, weights, kv_scale):
    zero = fx.Float32(0.0)
    total = zero
    for mi in range_constexpr(M_TILES):
        for ii in range_constexpr(DREG):
            total = total + fx.Float32(scores[mi][ii]).maximumf(zero) * weights[mi][ii]
    total = total * kv_scale
    total = total + total.shuffle_xor(16, 64)
    return total + total.shuffle_xor(32, 64)


def _make_out_row_view(logits, stride_out, row):
    byte = fx.Int64(fx.Uint32(row)) * fx.Int64(fx.Uint32(stride_out)) * 4
    return GTensor(
        logits,
        dtype=T.f32,
        shape=(-1,),
        cache_modifier=2,
        static_bytes_offset_i64=byte,
    )


@lru_cache(maxsize=8)
def device_cu_count(device_index: int) -> int:
    try:
        return torch.cuda.get_device_properties(device_index).multi_processor_count
    except Exception:  # noqa: BLE001
        return 256


def _build_kernel():
    kernel_name = (
        "fp8_paged_mqa_logits_gfx950_H32_D128_bkv64_kvb64_"
        "nn8_w8_lds2_mfma16_ntg2l_flydsl"
    )

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, LDS_BYTES, 16]

    @flyc.kernel(name=kernel_name, known_block_size=[THREADS, 1, 1])
    def kernel(
        Q: fx.Tensor,
        KV_cache: fx.Tensor,
        weights: fx.Tensor,
        out_logits: fx.Tensor,
        context_lens: fx.Tensor,
        kv_indices: fx.Tensor,
        next_n_lens: fx.Tensor,
        batch_size: fx.Int32,
        split_kv: fx.Int32,
        stride_q_batch: fx.Int32,
        max_block_len: fx.Int32,
        stride_out: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        wave = udiv(tid, 64)
        lane = umod(tid, 64)
        lane_div_16 = udiv(lane, 16)
        lane_mod_16 = umod(lane, 16)

        bid = fx.Int32(fx.block_idx.x)
        pid_batch = umod(bid, batch_size)
        pid_split = udiv(bid, batch_size)

        q_t = GTensor(Q, dtype=T.i32, shape=(-1,))
        kv_t = GTensor(KV_cache, dtype=T.i32, shape=(-1,))
        w_t = GTensor(weights, dtype=T.i32, shape=(-1,))
        context_t = GTensor(context_lens, dtype=T.i32, shape=(-1,))
        table_t = GTensor(kv_indices, dtype=T.i32, shape=(-1,))
        nn_t = GTensor(next_n_lens, dtype=T.i32, shape=(-1,))

        nn = fx.Int32(nn_t[pid_batch])
        context_len = fx.Int32(context_t[pid_batch])
        page_count = uceildiv(context_len, fx.Int32(KV_BLOCK_SIZE))
        pages_per_split = uceildiv(page_count, split_kv)
        page_lo = pid_split * pages_per_split
        page_hi = imin(page_lo + pages_per_split, page_count)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_base = fx.Int32(fx.ptrtoint(lds.raw.ptr))
        q_lds_base = lds_base + Q_LDS_OFF
        w_lds_base = lds_base + W_LDS_OFF

        def _issue_dma(rsrc, dst_byte, src_byte, *, aux):
            rocdl.raw_ptr_buffer_load_lds(
                rsrc,
                _lds_ptr3(lds_base, dst_byte),
                fx.Int32(16),
                fx.Int32(src_byte),
                fx.Int32(0),
                fx.Int32(0),
                aux,
            )

        # Cached prologue: live Q rows and their f32 head weights.
        q_batch_base = pid_batch * stride_q_batch
        for rep in range_constexpr(Q_LDS_BYTES // (THREADS * 16)):
            ci = tid + rep * THREADS
            row = udiv(ci * 16, Q_ROW_BYTES)
            pred = row < nn

            def _copy_q(_ci=ci):
                byte = _ci * 16
                _issue_dma(
                    q_t.rsrc,
                    Q_LDS_OFF + byte,
                    q_batch_base + byte,
                    aux=0,
                )

            _guarded(pred, _copy_q)

        w_batch_base = pid_batch * (MAX_NN * W_ROW_BYTES)
        ci_w = tid
        w_row = udiv(ci_w * 16, W_ROW_BYTES)

        def _copy_w(_ci=ci_w):
            byte = _ci * 16
            _issue_dma(
                w_t.rsrc,
                W_LDS_OFF + byte,
                w_batch_base + byte,
                aux=0,
            )

        _guarded((ci_w < W_LDS_BYTES // 16) & (w_row < nn), _copy_w)
        _wait_all()
        gpu.barrier()

        def _load_physical(page):
            safe_page = imin(page, page_hi - fx.Int32(1))
            return fx.Int32(
                buffer_ops.buffer_load(
                    table_t.rsrc,
                    pid_batch * max_block_len + safe_page,
                    vec_width=1,
                    dtype=T.i32,
                    is_scalar=True,
                )
            )

        def _copy_page(bank_base, page, copy_tid, copy_threads, reps):
            physical = _load_physical(page)
            for rep in range_constexpr(reps):
                ci = copy_tid + rep * copy_threads
                pred = ci < PAGE_CHUNKS

                def _copy(_chunk=ci, _physical=physical):
                    _issue_dma(
                        kv_t.rsrc,
                        bank_base + _chunk * 16,
                        _physical * PAGE_BYTES + _chunk * 16,
                        aux=2,
                    )

                _guarded(pred, _copy)

        def _store_logit(row, col, value):
            out = _make_out_row_view(out_logits, stride_out, pid_batch * MAX_NN + row)

            def _write(_out=out, _col=col, _value=value):
                _out[_col] = _value

            _guarded(
                (lane_div_16 == 0)
                & (col < context_len)
                & (col <= context_len - nn + row),
                _write,
            )

        def _compute_tile(bank_base, page, tile):
            k_base = lds_base + bank_base
            token_base = tile * MFMA_N
            b_pack = _load_k_pack_lds(k_base, tile, lane_mod_16, lane_div_16)
            scale = _lds_f32(
                k_base,
                KV_BLOCK_SIZE * HEAD_DIM + (token_base + lane_mod_16) * 4,
            )
            row = fx.Int32(0)
            while row < nn:
                a_packs = [
                    _load_q_pack_lds(
                        q_lds_base,
                        row,
                        mi * MFMA_M + lane_mod_16,
                        lane_div_16,
                    )
                    for mi in range_constexpr(M_TILES)
                ]
                weights_frag = [
                    _load_weight_frag_lds(w_lds_base, row, mi, lane_div_16)
                    for mi in range_constexpr(M_TILES)
                ]
                scores = [
                    _mfma_score(a_packs[mi], b_pack) for mi in range_constexpr(M_TILES)
                ]
                value = _reduce_scores(scores, weights_frag, scale)
                _store_logit(
                    row,
                    page * KV_BLOCK_SIZE + token_base + lane_mod_16,
                    value,
                )
                row = row + fx.Int32(1)

        if page_lo < page_hi:
            # All 512 threads fill the first page (two 16-byte issues max).
            _copy_page(
                fx.Int32(0),
                page_lo,
                tid,
                fx.Int32(THREADS),
                2,
            )
            _wait_all()
            gpu.barrier()

            init_state = [fx.Int32(0)]
            for page, state in range(page_lo, page_hi, fx.Int32(1), init=init_state):
                page_i = fx.Int32(page)
                bank = state[0]
                bank_base = bank * PAGE_BYTES
                next_bank = fx.Int32(1) - bank
                next_bank_base = next_bank * PAGE_BYTES

                compute_pred = wave < COMPUTE_WAVES

                def _compute(_bank_base=bank_base, _page=page_i):
                    _compute_tile(_bank_base, _page, wave)

                _guarded(compute_pred, _compute)

                helper_tid = tid - COMPUTE_WAVES * 64
                next_page = page_i + fx.Int32(1)

                def _prefetch(
                    _bank_base=next_bank_base,
                    _page=next_page,
                    _helper_tid=helper_tid,
                ):
                    _copy_page(
                        _bank_base,
                        _page,
                        _helper_tid,
                        fx.Int32((WAVES - COMPUTE_WAVES) * 64),
                        3,
                    )

                _guarded(
                    (wave >= COMPUTE_WAVES) & (next_page < page_hi),
                    _prefetch,
                )
                _wait_all()
                gpu.barrier()
                yield [next_bank]

    @flyc.jit
    def launch(
        Q,
        KV_cache,
        weights,
        out_logits,
        context_lens,
        kv_indices,
        next_n_lens,
        grid_blocks,
        batch_size,
        split_kv,
        stride_q_batch,
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
            next_n_lens,
            batch_size,
            split_kv,
            stride_q_batch,
            max_block_len,
            stride_out,
        ).launch(
            grid=(arith.index_cast(T.index, _to_raw(grid_blocks)), 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    launch.compile_hints = dict(DEFAULT_COMPILE_HINTS)
    launch.kernel_name = kernel_name
    return launch


@lru_cache(maxsize=1)
def _compile():
    return _build_kernel()


def flydsl_fp8_paged_mqa_logits(
    q_fp8,
    kv_cache,
    weights,
    out_logits,
    context_lens,
    kv_indices,
    max_model_len,
    *,
    next_n_lens=None,
    Preshuffle=True,
    KVBlockSize=KV_BLOCK_SIZE,
    SplitKV=None,
    TotalCuCount=None,
    stream=None,
):
    """Ragged paged FP8 MQA logits for padded Q ``[B, 8, 32, 128]``."""
    if get_gfx() != _GFX950:
        raise RuntimeError(f"gfx950 kernel requested on {get_gfx()}")
    if not Preshuffle:
        raise ValueError("requires Preshuffle=True")
    if q_fp8.ndim != 4:
        raise ValueError(f"q_fp8 must be rank 4, got shape {tuple(q_fp8.shape)}")
    batch_size, max_nn, heads, head_dim = q_fp8.shape
    if (max_nn, heads, head_dim, int(KVBlockSize)) != (
        MAX_NN,
        NUM_HEADS,
        HEAD_DIM,
        KV_BLOCK_SIZE,
    ):
        raise ValueError("requires Q shape [B, 8, 32, 128] and KVBlockSize=64")
    if not q_fp8.is_contiguous():
        raise ValueError("q_fp8 must be contiguous")
    if q_fp8.dtype != get_fp8_e4m3_dtype():
        raise ValueError(f"q_fp8 must be native FP8 E4M3, got {q_fp8.dtype}")
    if kv_cache.dtype != torch.uint8 or not kv_cache.is_contiguous():
        raise ValueError("kv_cache must be contiguous preshuffled uint8 FP8 data")
    _, block_size, one, index_dim = kv_cache.shape
    if (block_size, one, index_dim) != (KV_BLOCK_SIZE, 1, INDEX_DIM):
        raise ValueError(f"unexpected KV cache shape {tuple(kv_cache.shape)}")
    if weights.shape != (batch_size * MAX_NN, NUM_HEADS):
        raise ValueError(f"weights must have shape {(batch_size * MAX_NN, NUM_HEADS)}")
    if not weights.is_contiguous() or weights.dtype != torch.float32:
        raise ValueError("weights must be contiguous float32")

    context_lens = context_lens.reshape(batch_size)
    max_block_len = kv_indices.shape[-1]
    kv_indices = kv_indices.reshape(batch_size, max_block_len)
    if next_n_lens is None:
        next_n_lens = torch.full(
            (batch_size,),
            MAX_NN,
            dtype=torch.int32,
            device=q_fp8.device,
        )
    next_n_lens = next_n_lens.reshape(batch_size)
    if next_n_lens.dtype != torch.int32 or next_n_lens.device != q_fp8.device:
        raise ValueError("next_n_lens must be int32 on the same device as q_fp8")

    # One page fills all four compute SIMDs. Target three resident WGs/CU,
    # matching the 50 KiB LDS allocation. Auto mode reads the runtime context
    # maximum once; explicit SplitKV is graph-capture safe.
    if SplitKV is None or int(SplitKV) <= 0:
        real_pages = max(
            1,
            (int(context_lens.max().item()) + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE,
        )
        total_cu = (
            device_cu_count(q_fp8.device.index)
            if TotalCuCount is None
            else int(TotalCuCount)
        )
        split_kv = max(
            1,
            min(real_pages, (total_cu * 3 + batch_size - 1) // batch_size),
        )
    else:
        split_kv = max(1, int(SplitKV))

    grid_blocks = batch_size * split_kv
    launcher = _compile()
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
            next_n_lens,
            int(grid_blocks),
            int(batch_size),
            int(split_kv),
            int(q_fp8.stride(0)),
            int(max_block_len),
            int(out_logits.stride(0)),
            stream,
        )
    return out_logits


flydsl_fp8_paged_mqa_logits_gfx950 = flydsl_fp8_paged_mqa_logits
