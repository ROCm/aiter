# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Paged FP8 MQA logits (DeepSeek lightning indexer, decode) -- FlyDSL gfx950.

Compute for each decode batch element ``b`` and next-n query slot ``n`` (query
row ``b*next_n + n``) and KV logical position ``p``::

    logits[b*next_n+n, p] = sum_h ReLU(<Q[b,n,h,:], K_phys(p)>) * weights[b*next_n+n, h] * kv_scale_phys(p)

masked by the causal rule ``p <= context_lens[b] - next_n + n`` (positions past
the query token and ``p >= context_lens[b]`` stay ``-inf``). ``K_phys(p)`` /
``kv_scale_phys(p)`` are gathered from a **paged** cache: logical position ``p``
maps through the block table (``kv_indices``) to a physical block, and each
token's fp8 K bytes and its f32 dequant scale are **co-packed** in that block.

Supports ``KVBlockSize >= 1`` with SplitKV KV-column parallelism, and mirrors
the tensor contract of the Triton ``deepgemm_fp8_paged_mqa_logits`` so the two
are interchangeable. ``Preshuffle=True`` consumes the production
``shuffle_weight(layout=(16,16))`` KV layout (needs ``KVBlockSize % 16 == 0``).

Cache layout, per physical block of ``KVBlockSize`` tokens: the fp8 key rows
(``D`` bytes each) grouped first, then the f32 dequant scales. Each lane
resolves its own column through ``kv_indices[b, p // KVBlockSize]``, so a column
tile may straddle blocks; ``KVBlockSize == 1`` degenerates to per-token
``[D fp8 | 4 scale]`` slots.

The MFMA compute, ReLU*weight head-sum, head reduce, dword-pack load and output
view are shared with the dense kernel via ``._mqa_logits_common``.
"""

# No `from __future__ import annotations`: FlyDSL arg typing needs real
# annotation objects, not PEP 563 strings.

import os
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import arith, range_constexpr
from flydsl.expr.typing import T

from aiter.jit.utils.chip_info import get_gfx

from ..tensor_shim import GTensor, _run_compiled, _to_raw
from ._mqa_logits_common import (
    DEFAULT_COMPILE_HINTS,
    MFMA_K,
    MFMA_M,
    MFMA_N,
    device_cu_count,
    load_pack_kv,
    load_pack_v8i32,
    make_kv_key_view,
    make_kv_scale_view,
    make_out_row_view,
    mfma_head_reduce,
    udiv,
    umod,
)

_SIMDS_PER_CU = 4


def _auto_split_kv(
    batch_size, next_n, wave_per_eu, device_index, total_cu=None, waves_per_block=1
):
    """Pick SplitKV so the dispatch lands ``wave_per_eu`` waves on every SIMD:

        SplitKV = round(WavePerEU * 4 * TotalCuCount / (batch*next_n*WavePerBlock))

    Resident wave count is the lever that matters here: the kernel is bound by
    memory throughput rather than per-wave latency, so too few waves leaves the
    memory system idle and too many just queue. Accounting for the CTA's wave
    count keeps the target fixed across shapes. Returns >= 1.
    """
    tile_q_count = max(1, batch_size * next_n)
    if total_cu is None:
        total_cu = device_cu_count(device_index)
    target_waves = max(1, wave_per_eu) * _SIMDS_PER_CU * total_cu
    split_kv = round(target_waves / (tile_q_count * max(1, waves_per_block)))
    return max(1, int(split_kv))


# Default KV tile width (columns processed per MFMA inner-loop iteration).
_BLOCK_KV = 128


def _uceildiv(a, b):
    a = fx.Int32(a)
    b = fx.Int32(b)
    return fx.Int32((fx.Uint32(a) + fx.Uint32(b) - 1) // fx.Uint32(b))


def _imin(a, b):
    a = fx.Int32(a)
    b = fx.Int32(b)
    return (a <= b).select(a, b)


def _build_paged_kernel(
    *,
    num_heads: int,
    head_size: int,
    block_kv: int,
    waves_per_block: int,
    kv_block_size: int = 1,
    preshuffle: bool = False,
):
    """Paged MQA-logits kernel with SplitKV column parallelism.

    One thread block owns one ``(batch, next_n)`` query row and walks its KV
    window in ``BKV``-wide tiles; the ``waves_per_block`` waves each take a
    disjoint slice of the ``BKV/MFMA_N`` column tiles, with no barrier between
    them.

    ``preshuffle`` only changes the intra-block key-byte offset -- the
    block-table gather, scale gather, MFMA and mask are identical either way.
    """
    H = num_heads
    D = head_size
    BKV = block_kv
    WPB = waves_per_block
    KVB = kv_block_size
    assert KVB >= 1, f"kv_block_size must be >= 1; got {KVB}"
    MR_BLOCK_THREADS = 64 * WPB

    assert H % MFMA_M == 0, f"num_heads={H} must be a multiple of MFMA_M={MFMA_M}"
    assert BKV % MFMA_N == 0, f"block_kv={BKV} must be a multiple of MFMA_N={MFMA_N}"
    assert D % MFMA_K == 0, f"head_size={D} must be a multiple of MFMA_K={MFMA_K}"
    assert WPB >= 1, "waves_per_block must be >= 1"
    N_TILES = BKV // MFMA_N
    assert (
        N_TILES % WPB == 0
    ), f"BKV/MFMA_N={N_TILES} must be divisible by waves_per_block={WPB}"
    M_TILES = H // MFMA_M
    K_STEPS = D // MFMA_K
    N_TILES_PER_WAVE = N_TILES // WPB

    _kname = (
        f"fp8_paged_mqa_logits_H{H}_D{D}_bkv{BKV}_kvb{KVB}_w{WPB}"
        f"{'_ps' if preshuffle else ''}_flydsl"
    )

    @flyc.kernel(name=_kname, known_block_size=[MR_BLOCK_THREADS, 1, 1])
    def kernel(
        Q: fx.Tensor,  # [batch, next_n, H, D]       fp8 (bytes passed raw)
        KV_cache: fx.Tensor,  # [num_blocks, KVB, 1, index_dim] uint8 block-flat
        weights: fx.Tensor,  # [batch*next_n, H]           f32
        out_logits: fx.Tensor,  # [batch*next_n, max_model_len] f32 (-inf prefilled)
        context_lens: fx.Tensor,  # [batch]                     i32
        kv_indices: fx.Tensor,  # [batch, max_block_len]        i32 (block table)
        next_n: fx.Int32,
        batch_size: fx.Int32,
        split_kv: fx.Int32,  # KV-column splits (grid = split_kv*batch*next_n)
        stride_q_batch: fx.Int32,  # fp8 elems (== bytes)
        stride_q_next_n: fx.Int32,
        stride_q_heads: fx.Int32,
        index_dim: fx.Int32,  # per-token slot bytes (D+4+pad); block stride = KVB*index_dim
        max_block_len: fx.Int32,  # kv_indices row width
        stride_out: fx.Int32,  # out_logits.stride(0) == max_model_len
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x

        # ChunkQ==H  =>  grid = split_kv * batch * next_n (split outermost),
        # mirroring the Triton _deepgemm_fp8_paged_mqa_logits pid decomposition.
        pid_next_n = umod(bid, next_n)
        _rem = udiv(bid, next_n)
        pid_batch = umod(_rem, batch_size)
        pid_split_kv = udiv(_rem, batch_size)

        wave = udiv(tid, 64)
        lane = umod(tid, 64)
        lane_div_N = udiv(lane, MFMA_N)
        lane_mod_N = umod(lane, MFMA_N)
        lane8 = lane_div_N * 8

        q_i32 = GTensor(Q, dtype=T.i32, shape=(-1,))
        # Uniform-base views: the per-token gather rides the per-lane byte
        # offset, not the buffer base, since a per-lane base cannot ride a
        # scalar buffer descriptor. That offset stays i32, hence the host's
        # pool-size assert.
        kv_key = make_kv_key_view(KV_cache, D, KVB, index_dim, preshuffle)
        kv_scale_v = make_kv_scale_view(KV_cache, D, KVB, index_dim)
        w_t = GTensor(weights, dtype=T.f32, shape=(-1, H))
        cl_t = GTensor(context_lens, dtype=T.i32, shape=(-1,))
        ind_t = GTensor(kv_indices, dtype=T.i32, shape=(-1,))

        context_length = fx.Int32(cl_t[pid_batch])
        # Inclusive causal upper bound: p <= context_length - next_n + pid_next_n.
        q_limit = context_length - next_n + pid_next_n

        out_row = pid_batch * next_n + pid_next_n
        out_row_t = make_out_row_view(out_logits, stride_out, out_row)

        # ---- Preload Q frags + weights for the single query row ----
        # The A-operand layout is per in-wave lane, so `lane` indexes Q, not
        # `tid`.
        q_row_base = pid_batch * stride_q_batch + pid_next_n * stride_q_next_n
        a_pack = [[None] * K_STEPS for _ in range_constexpr(M_TILES)]
        for mi in range_constexpr(M_TILES):
            h_a = mi * MFMA_M + lane_mod_N
            base_a = q_row_base + h_a * stride_q_heads
            for kk in range_constexpr(K_STEPS):
                byte_base = base_a + kk * MFMA_K
                a_pack[mi][kk] = load_pack_v8i32(q_i32, byte_base, lane8)

        # weights[out_row, h] per (mi, ii): head = mi*MFMA_M + h_off(lane_div_N, ii)
        # For the 32x32x64 MFMA (16 f32 D-regs), the D-reg ii at lane_div_N encodes
        # head:  h_off = (ii % 4) + lane_div_N * 4 + (ii // 4) * 8
        # -- the same 4-interleave as the 16x16x32 layout, extended to 16 D-regs.
        _DREG = 16
        w_frag = [[None] * _DREG for _ in range_constexpr(M_TILES)]
        for mi in range_constexpr(M_TILES):
            for ii in range_constexpr(_DREG):
                h_off = (ii % 4) + lane_div_N * 4 + (ii // 4) * 8
                h_w = mi * MFMA_M + h_off
                w_frag[mi][ii] = fx.Float32(w_t[out_row, h_w])

        # ---- SplitKV: each split owns ceil(total_tiles/split_kv) contiguous
        # BKV-aligned tiles, so the slices are gap-free and disjoint and every
        # logits column has exactly one writer -- no cross-CTA reduction. A
        # split past ctx runs zero iterations. ----
        context_chunk_num = _uceildiv(context_length, fx.Int32(BKV))
        split_chunk_num = _uceildiv(context_chunk_num, split_kv)
        full_end = context_chunk_num * BKV
        split_cols = split_chunk_num * BKV
        tile_lo_col = pid_split_kv * split_cols
        tile_hi_col = _imin(tile_lo_col + split_cols, full_end)
        ctx_m1 = context_length - 1

        for col0 in range(tile_lo_col, tile_hi_col, fx.Int32(BKV)):
            wave_ni_base = wave * N_TILES_PER_WAVE

            # Helpers close over this chunk's SSA values; default args silence B023.
            def _col_for(abs_ni, _col0=col0):
                return _col0 + abs_ni * MFMA_N + lane_mod_N

            def _col_c_for(abs_ni, _ctx_m1=ctx_m1):
                return _imin(_col_for(abs_ni), _ctx_m1)

            def _ind_off_for(abs_ni):
                return pid_batch * max_block_len + udiv(_col_c_for(abs_ni), KVB)

            def _tok_in_block_for(abs_ni):
                return umod(_col_c_for(abs_ni), KVB)

            def _load_physical(ind_off):
                return fx.Int32(ind_t[ind_off])

            def _prefetch_physical(physical_cur, ind_off_cur, ind_off_next):
                if ind_off_next == ind_off_cur:
                    return physical_cur
                return _load_physical(ind_off_next)

            def _b_col_for_tile(physical, tok_in_block, b0_carried):
                return [
                    b0_carried,
                    load_pack_kv(kv_key, physical, tok_in_block, 1, lane_div_N),
                ]

            def _process_tile_with_b0(abs_ni, physical, b0_carried):
                col = _col_for(abs_ni)
                tok_in_block = _tok_in_block_for(abs_ni)
                b_col = _b_col_for_tile(physical, tok_in_block, b0_carried)
                kv_scale = fx.Float32(kv_scale_v[physical, tok_in_block])
                col_sum = mfma_head_reduce(
                    a_pack, b_col, w_frag, kv_scale, m_tiles=M_TILES, k_steps=K_STEPS
                )
                is_writer = (lane_div_N == 0) & (col <= q_limit)

                def _do_write(_t=out_row_t, _c=col, _v=col_sum):
                    _t[_c] = _v

                @flyc.jit
                def _guarded_write(_pred=is_writer, _w=_do_write):
                    if _pred:
                        _w()

                _guarded_write()

            def _prefetch_k0(abs_ni, physical):
                return load_pack_kv(
                    kv_key, physical, _tok_in_block_for(abs_ni), 0, lane_div_N
                )

            physical_carry = _load_physical(_ind_off_for(wave_ni_base + 0))
            b0_carry = _prefetch_k0(wave_ni_base + 0, physical_carry)
            for ni in range_constexpr(N_TILES_PER_WAVE):
                abs_ni = wave_ni_base + ni
                physical = physical_carry
                if ni + 1 < N_TILES_PER_WAVE:
                    abs_ni_next = wave_ni_base + ni + 1
                    physical_carry = _prefetch_physical(
                        physical,
                        _ind_off_for(abs_ni),
                        _ind_off_for(abs_ni_next),
                    )
                _process_tile_with_b0(abs_ni, physical, b0_carry)
                if ni + 1 < N_TILES_PER_WAVE:
                    abs_ni_next = wave_ni_base + ni + 1
                    b0_carry = _prefetch_k0(abs_ni_next, physical_carry)

    @flyc.jit
    def launch_fp8_paged_mqa_logits(
        Q: fx.Tensor,
        KV_cache: fx.Tensor,
        weights: fx.Tensor,
        out_logits: fx.Tensor,
        context_lens: fx.Tensor,
        kv_indices: fx.Tensor,
        grid_blocks: fx.Int32,
        next_n: fx.Int32,
        batch_size: fx.Int32,
        split_kv: fx.Int32,
        stride_q_batch: fx.Int32,
        stride_q_next_n: fx.Int32,
        stride_q_heads: fx.Int32,
        index_dim: fx.Int32,
        max_block_len: fx.Int32,
        stride_out: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        gx = arith.index_cast(T.index, _to_raw(grid_blocks))
        kernel._func.__name__ = _kname
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
            index_dim,
            max_block_len,
            stride_out,
        ).launch(grid=(gx, 1, 1), block=(MR_BLOCK_THREADS, 1, 1), stream=stream)

    return launch_fp8_paged_mqa_logits


# Kernel variants: single-token-per-block, wave-split only ("paged_w<WPB>").
# WPB must divide the column-tile count BKV/16 (=8 at the default BKV=128).
#
# WPB=1 (one wave per CTA) is the default: a 1-wave CTA schedules more freely
# and pays no whole-CTA granularity cost, and the waves have no work to share
# anyway, each owning a disjoint set of KV columns with no barrier or LDS
# between them.
KERNEL_VARIANTS = tuple(f"paged_w{w}" for w in (1, 2, 4))
DEFAULT_VARIANT = "paged_w1"


def _variant_wpb(variant):
    """Parse the WPB int out of a ``paged_w<WPB>`` tag (single validation point)."""
    if variant not in KERNEL_VARIANTS:
        raise ValueError(
            f"unknown fp8_paged_mqa_logits variant {variant!r}; "
            f"available: {list(KERNEL_VARIANTS)}"
        )
    return int(variant.removeprefix("paged_w"))


def _resolve_variant(variant):
    """Pick the variant tag: explicit arg, then env override, then default.
    Shape-adaptive selection is not implemented."""
    return (
        variant
        or os.environ.get("FLYDSL_FP8_PAGED_MQA_LOGITS_VARIANT")
        or DEFAULT_VARIANT
    )


@lru_cache(maxsize=32)
def compile_fp8_paged_mqa_logits(
    *,
    num_heads: int,
    head_size: int,
    block_kv: int = _BLOCK_KV,
    kv_block_size: int = 1,
    preshuffle: bool = False,
    variant: str = DEFAULT_VARIANT,
):
    """Return a cached, compiled FlyDSL paged launcher for the given config.

    ``num_heads``/``head_size``/``kv_block_size``/``preshuffle`` are compile-time
    constants and ``variant`` is a ``paged_w<WPB>`` tag.
    """
    launcher = _build_paged_kernel(
        num_heads=num_heads,
        head_size=head_size,
        block_kv=block_kv,
        waves_per_block=_variant_wpb(variant),
        kv_block_size=kv_block_size,
        preshuffle=preshuffle,
    )
    launcher.compile_hints = dict(DEFAULT_COMPILE_HINTS)
    return launcher


def flydsl_fp8_paged_mqa_logits(
    q_fp8,
    kv_cache,
    weights,
    out_logits,
    context_lens,
    kv_indices,
    max_model_len,
    *,
    Preshuffle=False,
    KVBlockSize=1,
    ChunkK=_BLOCK_KV,
    SplitKV=None,
    WavePerEU=2,
    TotalCuCount=None,
    variant=None,
    stream=None,
):
    """FlyDSL paged FP8 MQA logits (decode) -- KVBlockSize>=1 with SplitKV.

    Drop-in for the Triton ``deepgemm_fp8_paged_mqa_logits`` tensor contract.

    q_fp8:        [batch, next_n, heads, hidden_dim], float8 e4m3fn (gfx950 native)
    kv_cache:     [num_blocks, KVBlockSize, 1, index_dim] uint8, co-packed per
                  block as KVBlockSize*hidden_dim fp8 K bytes then KVBlockSize
                  f32 scales; index_dim == hidden_dim + 4 (+ optional padding).
    Preshuffle:   consume the shuffle_weight(16x16) layout for the fp8 data
                  section (the scale tail stays token-ordered). Requires
                  ``KVBlockSize % 16 == 0``.
    weights:      [batch*next_n, heads], float32
    out_logits:   [batch*next_n, max_model_len], float32. MUST be prefilled with
                  -inf: the kernel writes only in-window positions.
    context_lens: [batch], int32
    kv_indices:   [batch, max_block_len], int32 block table indexed by
                  ``p // KVBlockSize``.
    max_model_len: out_logits column count.
    SplitKV:      KV-column split count (grid = split_kv*batch*next_n). None =>
                  sized for WavePerEU waves per SIMD; 1 disables splitting.
    WavePerEU:    target resident waves per SIMD for the auto-SplitKV formula.
    TotalCuCount: CU count for that formula; None => query the device.
    ChunkK:       KV tile width, a multiple of MFMA_N=32. Defaults to 128.

    Returns the same ``out_logits`` tensor (written in place).
    """
    if Preshuffle:
        assert KVBlockSize % 16 == 0, (
            f"Preshuffle mode only supports KVBlockSize aligned to 16; "
            f"got KVBlockSize={KVBlockSize}."
        )
    if ChunkK % MFMA_N != 0:
        raise ValueError(f"ChunkK={ChunkK} must be a multiple of MFMA_N={MFMA_N}.")

    batch_size, next_n, num_heads, head_size = q_fp8.shape
    assert num_heads & (num_heads - 1) == 0, "num q. heads should be power of 2."
    assert head_size & (head_size - 1) == 0, "head size should be power of 2."

    num_blocks, block_size, one, index_dim = kv_cache.shape
    assert block_size == KVBlockSize, (
        f"kv_cache KVBlockSize dim ({block_size}) must equal KVBlockSize "
        f"({KVBlockSize})."
    )
    assert KVBlockSize >= 1, f"KVBlockSize must be >= 1; got {KVBlockSize}."
    assert one == 1, f"kv_cache head dim must be 1; got {one}."
    assert (
        index_dim >= head_size + 4
    ), f"index_dim={index_dim} must hold {head_size} fp8 bytes + 4 scale bytes."
    assert (
        kv_cache.dtype == torch.uint8
    ), f"kv_cache must be uint8 co-packed bytes; got {kv_cache.dtype}."

    # i32 gather-offset ceiling: the per-token byte base (physical block byte base
    # + intra-block offset) is computed in i32 (it rides the buffer voffset), so
    # the whole cache pool must be addressable in i32 bytes. A larger pool would
    # need an i64/global gather.
    pool_bytes = num_blocks * block_size * index_dim
    assert pool_bytes < 2**31, (
        f"num_blocks*KVBlockSize*index_dim={pool_bytes} exceeds the i32 "
        f"gather-offset limit (2^31); the paged kernel needs an i64 gather path "
        f"for a cache pool this large."
    )

    _, max_block_len = kv_indices.shape

    # FlyDSL's DLPack adaptor rejects 0-dim tensors; keep logical ranks.
    context_lens = context_lens.reshape(batch_size)
    weights = weights.reshape(batch_size * next_n, num_heads)
    kv_indices = kv_indices.reshape(batch_size, max_block_len)

    _fnuz = torch.float8_e4m3fnuz
    _fn = torch.float8_e4m3fn
    assert q_fp8.dtype in (
        _fnuz,
        _fn,
    ), f"q_fp8 must be e4m3 fp8 (fnuz or fn); got {q_fp8.dtype}"
    assert (
        get_gfx() == "gfx950"
    ), f"flydsl_fp8_paged_mqa_logits targets gfx950 (32x32x64 MFMA); got {get_gfx()}"

    variant = _resolve_variant(variant)

    launcher = compile_fp8_paged_mqa_logits(
        num_heads=num_heads,
        head_size=head_size,
        block_kv=ChunkK,
        kv_block_size=int(KVBlockSize),
        preshuffle=bool(Preshuffle),
        variant=variant,
    )

    # Co-packed cache -> raw uint8 byte view [num_blocks * KVBlockSize * index_dim].
    kv_bytes = kv_cache.reshape(-1)

    # KV-column splits (one lever): fill the device when the batch*next_n row
    # grid is small. Auto targets WavePerEU waves per SIMD; overridable.
    if SplitKV is None:
        split_kv = _auto_split_kv(
            batch_size,
            next_n,
            WavePerEU,
            q_fp8.device.index,
            TotalCuCount,
            waves_per_block=_variant_wpb(variant),
        )
    else:
        split_kv = max(1, int(SplitKV))

    grid_blocks = batch_size * next_n * split_kv

    if stream is None:
        stream = torch.cuda.current_stream()

    with torch.cuda.device(q_fp8.device.index):
        _run_compiled(
            launcher,
            q_fp8,
            kv_bytes,
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
            int(index_dim),
            int(max_block_len),
            int(out_logits.stride(0)),
            stream,
        )

    return out_logits
