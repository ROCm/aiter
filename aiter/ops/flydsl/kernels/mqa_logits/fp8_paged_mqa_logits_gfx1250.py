# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 paged FP8 MQA logits (decode / MTP), WMMA 16x16x128.

Drop-in for ``deepgemm_fp8_paged_mqa_logits(..., Preshuffle=True)`` on the
packed preshuffled cache the Triton/Gluon path reads::

    out[b * nn + j, t] = sum_h relu((q[b, j, h] . k_t) * scale_t) * w[b * nn + j, h]

for ``t < ctx_b``; columns ``t > ctx_b - nn + j`` (still below ``ctx_b``) get
``-inf``, columns at or past ``ctx_b`` are not written.

Cache page ``p`` is ``KVBlockSize * 128`` FP8 bytes shuffled with
``shuffle_weight(kv, layout=(min(KVBlockSize, 16), 16))`` followed by
``KVBlockSize`` fp32 scales. Within a group of ``BN = min(KVBlockSize, 16)``
rows the shuffle stores, for every 32-wide K chunk ``kc`` and half ``hh``, the
16 bytes ``d = kc * 32 + hh * 16 + [0, 16)`` of each row next to each other.
That is the wave32 WMMA 16x16x128 FP8 operand layout: lane ``n + 16 * hh``
holds ``d = kc * 32 + hh * 16 + [0, 16)`` for ``kc = 0..3``. So every lane
fetches its key column with four 16-byte loads straight into registers, no
LDS, and a 16-token tile may mix pages freely -- page 8 just means lanes 0-7
and 8-15 read two different pages.

Mapping: the M side of the WMMA is 16 heads of one query row (Q held in
registers for the whole workgroup), the N side 16 tokens. The f32 result puts
token ``lane % 16`` and heads ``8 * (lane // 16) + [0, 8)`` in each lane, so
the head reduction is eight in-lane FMAs plus one ``permlanex16``.

Work split: one workgroup (4 waves by default) per ``(sequence, split)``. A
sequence's 16-token tiles are divided evenly over ``SplitKV`` workgroups, so
split boundaries are multiples of the 16-column tile whatever the page size,
and no two workgroups write the same column. The waves of a workgroup take
interleaved tiles, ``Unroll`` each per step, in a two-stage register pipeline:
a step scores the keys loaded by the previous one while its successor's keys
and the page-table entries three steps ahead are in flight. ``SplitKV`` comes
from ``max_model_len`` (no device sync, graph-capture safe) unless given.

Caches below 2 GiB are read with buffer loads (32-bit offsets); larger ones
compile a variant with 64-bit global loads.
"""

from functools import cache, lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype

from .. import buffer_ops
from ..mla_gfx1250.mla_common import _xor16_f32, make_fp8_wmma_atom
from ..tensor_shim import GTensor, _run_compiled, buf_base_i64

Vec = fx.Vector

_GFX1250 = "gfx1250"
HEAD_DIM = 128
INDEX_DIM = HEAD_DIM + 4
WMMA_M = WMMA_N = 16
TILE = WMMA_N
WAVE = 32
DEFAULT_WAVES = 4
ACC = 8  # f32 accumulator VGPRs per lane of a 16x16 WMMA tile
OPND = 16  # i32 VGPRs per lane of a 16x128 FP8 WMMA operand
SUPPORTED_BLOCK_SIZES = (8, 16, 32, 64, 128, 256)
MAX_NEXT_N = 8
# Q fragments live in VGPRs: next_n * heads i32 per lane.
MAX_Q_VGPRS = 256
DEFAULT_UNROLL = 2


def udiv(a, b):
    return fx.Int32(fx.Uint32(a) // fx.Uint32(b))


def umod(a, b):
    return fx.Int32(fx.Uint32(a) % fx.Uint32(b))


def uceildiv(a, b):
    a, b = fx.Int32(a), fx.Int32(b)
    return fx.Int32((fx.Uint32(a) + fx.Uint32(b) - 1) // fx.Uint32(b))


def _imin(a, b):
    return (a <= b).select(a, b)


def _guarded(pred, fn):
    @flyc.jit
    def _do(_pred=pred, _fn=fn):
        if _pred:
            _fn()

    _do()


def _concat4(chunks):
    v01 = Vec(chunks[0]).shuffle(Vec(chunks[1]), list(range(8)))
    v23 = Vec(chunks[2]).shuffle(Vec(chunks[3]), list(range(8)))
    return v01.shuffle(v23, list(range(16)))


def _rmem(n, dtype, value):
    frag = fx.make_rmem_tensor(n, dtype)
    frag.store(value)
    return frag


def _global_load(addr_i64, n, elem):
    ptr = fx.inttoptr(fx.PointerType.get(elem, fx.AddressSpace.Global, 4 * n), addr_i64)
    if n == 1:
        return fx.ptr_load(ptr)
    return fx.ptr_load(ptr, result_type=T.vec(n, elem))


def _make_out_row_view(logits, stride_out, row):
    # 64-bit row base: rows * stride can pass 2**31 bytes for wide outputs.
    byte = fx.Int64(fx.Uint32(row)) * fx.Int64(fx.Uint32(stride_out)) * 4
    return GTensor(logits, dtype=T.f32, shape=(-1,), static_bytes_offset_i64=byte)


@cache
def device_cu_count(device_index: int) -> int:
    try:
        return torch.cuda.get_device_properties(device_index).multi_processor_count
    except Exception:  # noqa: BLE001
        return 256


@lru_cache(maxsize=64)
def _build_kernel(heads, next_n, block_size, ctx_stride, unroll, waves, wide_cache):
    threads = WAVE * waves
    m_tiles = heads // WMMA_M
    bn = min(block_size, 16)
    page_bytes = block_size * INDEX_DIM
    scale_off = block_size * HEAD_DIM
    wide = "_wide" if wide_cache else ""
    kernel_name = (
        f"fp8_paged_mqa_logits_gfx1250_H{heads}_D{HEAD_DIM}_nn{next_n}"
        f"_kvb{block_size}_cs{ctx_stride}_u{unroll}_w{waves}{wide}_flydsl"
    )

    @flyc.kernel(name=kernel_name, known_block_size=[threads, 1, 1])
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
        has_next_n_lens: fx.Int32,
        stride_q_batch: fx.Int32,
        stride_q_row: fx.Int32,
        stride_q_head: fx.Int32,
        stride_w: fx.Int32,
        stride_table: fx.Int32,
        stride_out: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        wave = udiv(tid, WAVE)
        lane = umod(tid, WAVE)
        lane16 = umod(lane, 16)
        half = udiv(lane, 16)

        bid = fx.Int32(fx.block_idx.x)
        pid_batch = umod(bid, batch_size)
        pid_split = udiv(bid, batch_size)

        q_t = GTensor(Q, dtype=T.i32, shape=(-1,))
        kv_t = GTensor(KV_cache, dtype=T.i32, shape=(-1,))
        w_t = GTensor(weights, dtype=T.f32, shape=(-1,))
        context_t = GTensor(context_lens, dtype=T.i32, shape=(-1,))
        table_t = GTensor(kv_indices, dtype=T.i32, shape=(-1,))
        nn_t = GTensor(next_n_lens, dtype=T.i32, shape=(-1,))
        kv_rsrc = kv_t.rsrc
        kv_base = buf_base_i64(KV_cache)
        table_rsrc = table_t.rsrc

        nn = (has_next_n_lens != 0).select(
            _imin(fx.Int32(nn_t[pid_batch]), fx.Int32(next_n)), fx.Int32(next_n)
        )
        context_len = fx.Int32(context_t[pid_batch * ctx_stride + (ctx_stride - 1)])
        tile_count = uceildiv(context_len, fx.Int32(TILE))
        tiles_per_split = uceildiv(tile_count, split_kv)
        tile_lo = pid_split * tiles_per_split
        tile_hi = _imin(tile_lo + tiles_per_split, tile_count)
        last_token = context_len - fx.Int32(1)
        table_base = pid_batch * stride_table
        step = fx.Int32(waves * unroll)

        def _tile_of(it, u):
            return tile_lo + it * step + fx.Int32(u * waves) + wave

        def _token_of(tile):
            return _imin(tile * TILE + lane16, last_token)

        def _load_physical(tile):
            slot = udiv(_token_of(tile), block_size)
            return fx.Int32(
                buffer_ops.buffer_load(
                    table_rsrc, table_base + slot, vec_width=1, dtype=T.i32
                )
            )

        def _load_key(tile, physical):
            in_page = umod(_token_of(tile), block_size)
            byte = (
                udiv(in_page, bn) * (bn * HEAD_DIM)
                + half * (bn * 16)
                + umod(in_page, bn) * 16
            )
            scale_byte = scale_off + in_page * 4
            if const_expr(wide_cache):
                # Caches past 2 GiB: 64-bit page base, global loads.
                page = kv_base + fx.Int64(fx.Uint32(physical)) * page_bytes
                chunks = [
                    _global_load(page + fx.Int64(byte + kc * (bn * 32)), 4, T.i32)
                    for kc in range_constexpr(4)
                ]
                scale = fx.Float32(_global_load(page + fx.Int64(scale_byte), 1, T.f32))
                return _concat4(chunks), scale
            page = physical * page_bytes
            chunks = [
                buffer_ops.buffer_load(
                    kv_rsrc,
                    udiv(page + byte + kc * (bn * 32), 4),
                    vec_width=4,
                    dtype=T.i32,
                )
                for kc in range_constexpr(4)
            ]
            scale = fx.Float32(
                buffer_ops.buffer_load(
                    kv_rsrc, udiv(page + scale_byte, 4), vec_width=1, dtype=T.f32
                )
            )
            return _concat4(chunks), scale

        def _store_logit(row, col, value):
            out = _make_out_row_view(out_logits, stride_out, pid_batch * next_n + row)
            masked = col > context_len - nn + row
            final = masked.select(fx.Float32(float("-inf")), value)

            def _write(_out=out, _col=col, _value=final):
                _out[_col] = _value

            _guarded((half == 0) & (col < context_len) & (row < nn), _write)

        if tile_lo < tile_hi:
            # Q fragments and head weights for every row, kept in VGPRs.
            q_frags = []
            w_frags = []
            for j in range_constexpr(next_n):
                q_row = []
                w_row = []
                for mi in range_constexpr(m_tiles):
                    head = mi * WMMA_M + lane16
                    byte = (
                        pid_batch * stride_q_batch
                        + j * stride_q_row
                        + head * stride_q_head
                        + half * 16
                    )
                    chunks = [
                        buffer_ops.buffer_load(
                            q_t.rsrc,
                            udiv(byte + kc * 32, 4),
                            vec_width=4,
                            dtype=T.i32,
                        )
                        for kc in range_constexpr(4)
                    ]
                    q_row.append(_concat4(chunks))
                    w_off = (
                        (pid_batch * next_n + j) * stride_w + mi * WMMA_M + half * ACC
                    )
                    lo = Vec(
                        buffer_ops.buffer_load(
                            w_t.rsrc, w_off, vec_width=4, dtype=T.f32
                        )
                    )
                    hi = Vec(
                        buffer_ops.buffer_load(
                            w_t.rsrc, w_off + 4, vec_width=4, dtype=T.f32
                        )
                    )
                    w_row.append(lo.shuffle(hi, list(range(8))))
                q_frags.append(q_row)
                w_frags.append(w_row)

            atom = make_fp8_wmma_atom(HEAD_DIM)
            zero = fx.Float32(0.0)

            def _score(key, scale, j):
                b_frag = _rmem(OPND, fx.Int32, key)
                total = zero
                for mi in range_constexpr(m_tiles):
                    a_frag = _rmem(OPND, fx.Int32, q_frags[j][mi])
                    c_frag = _rmem(ACC, fx.Float32, Vec.filled(ACC, 0.0, fx.Float32))
                    fx.gemm(atom, c_frag, a_frag, b_frag, c_frag)
                    acc = Vec(c_frag.load())
                    for i in range_constexpr(ACC):
                        s = (fx.Float32(acc[i]) * scale).maximumf(zero)
                        total = total + s * fx.Float32(w_frags[j][mi][i])
                return total + _xor16_f32(total)

            iters = uceildiv(tile_hi - tile_lo, step)

            def _score_tiles(tiles, keys):
                for u in range_constexpr(unroll):
                    key, scale = keys[u]
                    col = tiles[u] * TILE + lane16
                    live = tiles[u] < tile_hi
                    for j in range_constexpr(next_n):
                        value = _score(key, scale, j)

                        def _st(_j=j, _col=col, _value=value):
                            _store_logit(fx.Int32(_j), _col, _value)

                        _guarded(live, _st)

            # Software pipeline over half-steps of ``unroll`` tiles per wave:
            # half-step ``h`` scores keys loaded during ``h - 1``, while the
            # keys of ``h + 1`` and the page-table entries of ``h + 3`` are in
            # flight. A loop iteration runs two half-steps (A then B), so the
            # loads of the next A land straight in the loop-carried registers
            # and the back edge needs no copy (a copy would wait on them).
            # Loads past the split are clamped onto the last token.
            def _phys(h):
                return [_load_physical(_tile_of(h, u)) for u in range_constexpr(unroll)]

            def _keys(h, phys):
                return [
                    _load_key(_tile_of(h, u), phys[u]) for u in range_constexpr(unroll)
                ]

            def _flat(keys):
                return [k for k, _ in keys] + [sc for _, sc in keys]

            h0 = fx.Int32(0)
            keys_a = _keys(h0, _phys(h0))
            init = _phys(h0 + 1) + _phys(h0 + 2) + _flat(keys_a)
            pairs = uceildiv(iters, 2)
            for it, state in range(fx.Int32(0), pairs, fx.Int32(1), init=init):
                h = fx.Int32(it) * 2
                phys_b = [fx.Int32(state[u]) for u in range_constexpr(unroll)]
                phys_a = [fx.Int32(state[unroll + u]) for u in range_constexpr(unroll)]
                cur_a = [
                    (Vec(state[2 * unroll + u]), fx.Float32(state[3 * unroll + u]))
                    for u in range_constexpr(unroll)
                ]
                keys_b = _keys(h + 1, phys_b)
                nxt_phys_b = _phys(h + 3)
                _score_tiles([_tile_of(h, u) for u in range_constexpr(unroll)], cur_a)
                nxt_keys_a = _keys(h + 2, phys_a)
                nxt_phys_a = _phys(h + 4)
                _score_tiles(
                    [_tile_of(h + 1, u) for u in range_constexpr(unroll)], keys_b
                )
                yield nxt_phys_b + nxt_phys_a + _flat(nxt_keys_a)

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
        has_next_n_lens,
        stride_q_batch,
        stride_q_row,
        stride_q_head,
        stride_w,
        stride_table,
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
            has_next_n_lens,
            stride_q_batch,
            stride_q_row,
            stride_q_head,
            stride_w,
            stride_table,
            stride_out,
        ).launch(
            grid=(fx.Int64(grid_blocks), 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    launch.kernel_name = kernel_name
    return launch


def _context_lens_stride(context_lens, batch_size, next_n):
    if context_lens.dtype != torch.int32 or not context_lens.is_contiguous():
        raise ValueError("context_lens must be contiguous int32")
    if context_lens.ndim == 1 or (
        context_lens.ndim == 2 and context_lens.shape[1] == 1
    ):
        if context_lens.numel() != batch_size:
            raise ValueError(f"context_lens must have {batch_size} entries")
        return 1
    if context_lens.ndim == 2 and context_lens.shape == (batch_size, next_n):
        return next_n
    raise ValueError(
        f"context_lens must be [B] or [B, next_n], got {tuple(context_lens.shape)}"
    )


def default_split_kv(batch_size, max_model_len, total_cu):
    """Splits per sequence: about two workgroups per CU in total, each with at
    least a dozen 16-token tiles. ``max_model_len`` bounds the context, so no
    device sync is needed (graph-capture safe)."""
    tiles = max(1, (int(max_model_len) + TILE - 1) // TILE)
    target = max(1, (2 * total_cu + batch_size - 1) // batch_size)
    return max(1, min(target, tiles // 12))


def flydsl_fp8_paged_mqa_logits_gfx1250(
    q_fp8,
    kv_cache,
    weights,
    out_logits,
    context_lens,
    kv_indices,
    max_model_len,
    Preshuffle=True,
    KVBlockSize=8,
    *,
    next_n_lens=None,
    SplitKV=None,
    TotalCuCount=None,
    Unroll=DEFAULT_UNROLL,
    Waves=DEFAULT_WAVES,
    _wide_cache=False,
    stream=None,
    **_ignored,
):
    """Paged FP8 MQA logits on gfx1250; ``deepgemm_fp8_paged_mqa_logits`` args.

    ``q_fp8`` ``[B, next_n, H, 128]`` (H a multiple of 16, last dim
    contiguous), ``kv_cache`` ``[num_blocks, KVBlockSize, 1, 132]`` packed and
    preshuffled, ``weights`` ``[B * next_n, H]`` fp32, ``out_logits``
    ``[B * next_n, >= max context]`` fp32 (columns past a row's context are
    not written), ``context_lens`` ``[B]`` or ``[B, next_n]`` (last column),
    ``kv_indices`` ``[B, max_blocks]`` int32. ``next_n_lens`` optionally
    limits the live rows of each sequence. Extra Triton tuning kwargs
    (``ChunkK``, ``WavePerEU``) are accepted and ignored.
    """
    if get_gfx() != _GFX1250:
        raise RuntimeError(f"gfx1250 kernel requested on {get_gfx()}")
    if not Preshuffle:
        raise ValueError("requires Preshuffle=True")
    block_size = int(KVBlockSize)
    if block_size not in SUPPORTED_BLOCK_SIZES:
        raise ValueError(
            f"KVBlockSize must be one of {SUPPORTED_BLOCK_SIZES}, got {block_size}"
        )
    if q_fp8.ndim != 4:
        raise ValueError(f"q_fp8 must be rank 4, got shape {tuple(q_fp8.shape)}")
    batch_size, next_n, heads, head_dim = q_fp8.shape
    if head_dim != HEAD_DIM or heads % WMMA_M or heads <= 0:
        raise ValueError(
            f"requires q_fp8 [B, next_n, 16k, {HEAD_DIM}], got {tuple(q_fp8.shape)}"
        )
    if not 1 <= next_n <= MAX_NEXT_N or next_n * heads > MAX_Q_VGPRS:
        raise ValueError(
            f"next_n * heads must be <= {MAX_Q_VGPRS} with next_n <= {MAX_NEXT_N}"
        )
    if q_fp8.dtype != get_fp8_e4m3_dtype():
        raise ValueError(f"q_fp8 must be native FP8 E4M3, got {q_fp8.dtype}")
    if (
        q_fp8.stride(3) != 1
        or any(q_fp8.stride(i) % 16 for i in range(3))
        or q_fp8.data_ptr() % 16
    ):
        raise ValueError("q_fp8 needs a contiguous, 16-byte aligned head dim")
    if not kv_cache.is_contiguous():
        raise ValueError("kv_cache must be contiguous")
    if kv_cache.shape[1:] != (block_size, 1, INDEX_DIM):
        raise ValueError(f"unexpected KV cache shape {tuple(kv_cache.shape)}")
    if weights.shape != (batch_size * next_n, heads) or weights.dtype != torch.float32:
        raise ValueError(
            f"weights must be float32 {(batch_size * next_n, heads)}, "
            f"got {weights.dtype} {tuple(weights.shape)}"
        )
    if weights.stride(1) != 1 or weights.stride(0) % 4 or weights.data_ptr() % 16:
        raise ValueError("weights rows must be contiguous and 16-byte aligned")
    if out_logits.shape[0] != batch_size * next_n or out_logits.stride(1) != 1:
        raise ValueError(f"out_logits must be [{batch_size * next_n}, N] row-major")
    if out_logits.dtype != torch.float32:
        raise ValueError("out_logits must be float32")
    ctx_stride = _context_lens_stride(context_lens, batch_size, next_n)
    if kv_indices.ndim != 2 or kv_indices.shape[0] != batch_size:
        raise ValueError(f"kv_indices must be [{batch_size}, max_blocks]")
    if kv_indices.dtype != torch.int32 or kv_indices.stride(1) != 1:
        raise ValueError("kv_indices must be int32 with contiguous rows")
    tensors = [kv_cache, weights, out_logits, context_lens, kv_indices, next_n_lens]
    if any(t is not None and t.device != q_fp8.device for t in tensors):
        raise ValueError("all tensors must be on q_fp8's device")
    has_next_n_lens = next_n_lens is not None
    if next_n_lens is None:
        next_n_lens = context_lens
    elif next_n_lens.dtype != torch.int32 or next_n_lens.numel() != batch_size:
        raise ValueError(f"next_n_lens must be int32 [{batch_size}]")

    if SplitKV is None or int(SplitKV) <= 0:
        total_cu = (
            device_cu_count(q_fp8.device.index)
            if TotalCuCount is None
            else int(TotalCuCount)
        )
        split_kv = default_split_kv(batch_size, max_model_len, total_cu)
    else:
        split_kv = int(SplitKV)

    launcher = _build_kernel(
        heads,
        next_n,
        block_size,
        ctx_stride,
        int(Unroll),
        int(Waves),
        bool(_wide_cache) or kv_cache.numel() * kv_cache.element_size() >= 2**31,
    )
    stream = stream or torch.cuda.current_stream(q_fp8.device)
    with torch.cuda.device(q_fp8.device.index):
        _run_compiled(
            launcher,
            q_fp8,
            # 2-D: FlyDSL passes tensor extents as int32.
            kv_cache.view(torch.uint8).view(kv_cache.shape[0], -1),
            weights,
            out_logits,
            context_lens,
            kv_indices,
            next_n_lens,
            int(batch_size * split_kv),
            int(batch_size),
            int(split_kv),
            int(has_next_n_lens),
            int(q_fp8.stride(0)),
            int(q_fp8.stride(1)),
            int(q_fp8.stride(2)),
            int(weights.stride(0)),
            int(kv_indices.stride(0)),
            int(out_logits.stride(0)),
            stream,
        )
    return out_logits
