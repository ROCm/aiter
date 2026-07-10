# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Paged FP8 MQA logits (DeepSeek lightning indexer, decode) -- FlyDSL gfx942/gfx950.

Compute for each decode batch element ``b`` and next-n query slot ``n`` (query
row ``b*next_n + n``) and KV logical position ``p``::

    logits[b*next_n+n, p] = sum_h ReLU(<Q[b,n,h,:], K_phys(p)>) * weights[b*next_n+n, h] * kv_scale_phys(p)

masked by the causal rule ``p <= context_lens[b] - next_n + n`` (positions past
the query token and ``p >= context_lens[b]`` stay ``-inf``). ``K_phys(p)`` /
``kv_scale_phys(p)`` are gathered from a **paged** cache: logical position ``p``
maps through the block table (``kv_indices``) to a physical block, and each
token's fp8 K bytes and its f32 dequant scale are **co-packed** in that block.

Supported scope (this file): ``KVBlockSize == 1`` with SplitKV KV-column
parallelism. The public ``flydsl_fp8_paged_mqa_logits`` mirrors the tensor
contract of the Triton
``aiter.ops.triton.attention.pa_mqa_logits.deepgemm_fp8_paged_mqa_logits`` so the
two are interchangeable in tests/benchmarks. ``KVBlockSize > 1`` and preshuffle
are not supported and raise ``NotImplementedError``.

The fp8 16x16x32 MFMA compute, ReLU*weight head-sum + kv-scale hoist, in-wave
``shuffle_xor`` head reduce, the fp8 dword-pack load, the FN->FNUZ byte patch, and
the i64 byte-base per-row output view are all shared with the dense kernel via
``._mqa_logits_common`` (no duplication).
"""

# No `from __future__ import annotations`: FlyDSL arg typing needs real
# annotation objects, not PEP 563 strings.

import os
from functools import lru_cache

import torch

from aiter.jit.utils.chip_info import get_gfx

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, range_constexpr
from flydsl.expr.typing import T
from flydsl._mlir.dialects import scf
from flydsl._mlir import ir

from .tensor_shim import GTensor, _run_compiled, _to_raw
from ._mqa_logits_common import (
    MFMA_M,
    MFMA_N,
    MFMA_K,
    DEFAULT_COMPILE_HINTS,
    Vec,
    device_cu_count,
    load_pack_i64,
    fn_to_fnuz_i64,
    make_out_row_view,
    mfma_head_reduce,
)

def _auto_split_kv(batch_size, next_n, wave_per_eu, device_index, total_cu=None):
    """Production host SplitKV formula (mirrors deepgemm_fp8_paged_mqa_logits):

        SplitKV = ((max(1, TotalCuCount // (batch*next_n)) + 4) // 5 * 5) * WavePerEU

    Fills the device for small decode grids where the batch*next_n row grid alone
    leaves most CUs idle. Returns >= 1.
    """
    tile_q_count = max(1, batch_size * next_n)
    if total_cu is None:
        total_cu = device_cu_count(device_index)
    split_kv = ((max(1, total_cu // tile_q_count) + 4) // 5 * 5) * wave_per_eu
    return max(1, int(split_kv))

# Default KV tile width (columns processed per MFMA inner-loop iteration).
_BLOCK_KV = 128


def _build_paged_kernel(
    *,
    num_heads: int,
    head_size: int,
    block_kv: int,
    waves_per_block: int,
    scale_mul: float = 1.0,
    convert_q_fn: bool = False,
    convert_kv_fn: bool = False,
):
    """Paged MQA-logits kernel (KVBlockSize==1) with SplitKV column parallelism.

    One thread block owns one ``(batch, next_n)`` query row and walks the whole
    ``[0, context_length)`` KV window in ``BKV``-wide tiles. ``waves_per_block``
    waves each own a disjoint slice of the ``BKV/MFMA_N`` column tiles (no
    cross-wave sharing / barrier), mirroring the dense kernel's wave split.

    ``scale_mul`` folds the FN->FNUZ 2x-per-converted-operand compensation into
    the (co-packed) per-token scale at compile time (ReLU positive-homogeneity).
    """
    H = num_heads
    D = head_size
    BKV = block_kv
    WPB = waves_per_block
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

    fm_fast = arith.FastMathFlags.fast

    _cvt_tag = ""
    if convert_q_fn:
        _cvt_tag += "_cq"
    if convert_kv_fn:
        _cvt_tag += "_ck"
    _kname = f"fp8_paged_mqa_logits_H{H}_D{D}_bkv{BKV}_w{WPB}{_cvt_tag}_flydsl"

    @flyc.kernel(name=_kname, known_block_size=[MR_BLOCK_THREADS, 1, 1])
    def kernel(
        Q: fx.Tensor,  # [batch, next_n, H, D]       fp8 (bytes passed raw)
        KV_cache: fx.Tensor,  # [num_blocks, 1, 1, index_dim] uint8 co-packed
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
        index_dim: fx.Int32,  # cache row width in bytes (D + 4 + pad)
        max_block_len: fx.Int32,  # kv_indices row width
        stride_out: fx.Int32,  # out_logits.stride(0) == max_model_len
    ):
        f32_0 = arith.constant(0.0, type=T.f32)
        mfma_res_ty = Vec.make_type(4, fx.Float32)
        scale_mul_c = arith.constant(float(scale_mul), type=T.f32)

        tid = fx.thread_idx.x
        bid = fx.block_idx.x

        # ChunkQ==H  =>  grid = split_kv * batch * next_n (split outermost),
        # mirroring the Triton _deepgemm_fp8_paged_mqa_logits pid decomposition.
        pid_next_n = fx.Int32(arith.remui(_to_raw(bid), _to_raw(next_n)))
        _rem = fx.Int32(arith.divui(_to_raw(bid), _to_raw(next_n)))
        pid_batch = fx.Int32(arith.remui(_to_raw(_rem), _to_raw(batch_size)))
        pid_split_kv = fx.Int32(arith.divui(_to_raw(_rem), _to_raw(batch_size)))

        wave = fx.Int32(arith.divui(_to_raw(tid), _to_raw(fx.Int32(64))))
        lane = fx.Int32(arith.remui(_to_raw(tid), _to_raw(fx.Int32(64))))
        lane_div_N = fx.Int32(arith.divui(_to_raw(lane), _to_raw(fx.Int32(MFMA_N))))
        lane_mod_N = fx.Int32(arith.remui(_to_raw(lane), _to_raw(fx.Int32(MFMA_N))))
        lane8 = lane_div_N * 8

        # fp8 operands read 8 bytes at a time as 2 i32 dwords (v8i8 buffer_load
        # fails to lower on gfx942), then bitcast to i64 for the MFMA.
        q_i32 = GTensor(Q, dtype=T.i32, shape=(-1,))
        # Uniform-base views over the co-packed cache: per-token gather is done
        # via the (per-lane) byte offset, NOT the buffer base -- a per-lane base
        # can't ride a scalar buffer descriptor. The byte offset stays i32
        # (assumes num_blocks*index_dim < 2^31; see host assert); a shared cache
        # pool large enough to exceed that would need an i64/global-load gather.
        kv_i32 = GTensor(KV_cache, dtype=T.i32, shape=(-1,))
        kv_f32 = GTensor(KV_cache, dtype=T.f32, shape=(-1,))
        w_t = GTensor(weights, dtype=T.f32, shape=(-1, H))
        cl_t = GTensor(context_lens, dtype=T.i32, shape=(-1,))
        ind_t = GTensor(kv_indices, dtype=T.i32, shape=(-1,))

        context_length = fx.Int32(cl_t[pid_batch])
        # Inclusive causal upper bound: p <= context_length - next_n + pid_next_n.
        q_limit = context_length - next_n + pid_next_n

        out_row = pid_batch * next_n + pid_next_n
        stride_out_i64 = arith.extui(T.i64, _to_raw(stride_out))
        out_row_t = make_out_row_view(out_logits, stride_out_i64, out_row)

        # ---- Preload Q frags + weights for the single query row ----
        # A-operand layout is per in-wave lane, so `lane` (not `tid`) indexes Q.
        q_row_base = pid_batch * stride_q_batch + pid_next_n * stride_q_next_n
        a_pack = [[None] * K_STEPS for _ in range_constexpr(M_TILES)]
        for mi in range_constexpr(M_TILES):
            h_a = mi * MFMA_M + lane_mod_N
            base_a = q_row_base + h_a * stride_q_heads
            for kk in range_constexpr(K_STEPS):
                d_a = kk * MFMA_K + lane8
                raw = load_pack_i64(q_i32, base_a + d_a)
                a_pack[mi][kk] = fn_to_fnuz_i64(raw) if convert_q_fn else raw

        # weights[out_row, h] per (mi, ii): head = mi*MFMA_M + lane_div_N*4 + ii
        w_frag = [[None] * 4 for _ in range_constexpr(M_TILES)]
        for mi in range_constexpr(M_TILES):
            for ii in range_constexpr(4):
                h_w = mi * MFMA_M + lane_div_N * 4 + ii
                w_frag[mi][ii] = _to_raw(fx.Float32(w_t[out_row, h_w]))

        # ---- SplitKV: this CTA owns a disjoint, BKV-aligned slice of the KV
        # window. total_tiles = ceil(ctx/BKV); each split owns
        # ceil(total_tiles/split_kv) contiguous tiles => slices are gap-free and
        # disjoint, so every logits column has exactly one writer (pure
        # parallelism across n, no cross-CTA reduction). split_kv==1 collapses to
        # the full [0, ceil(ctx/BKV)*BKV) window. Splits past ctx early-exit
        # (tile_lo_col >= tile_hi_col => the loop runs 0 iterations). ----
        context_chunk_num = arith.ceildivui(
            _to_raw(context_length), _to_raw(fx.Int32(BKV))
        )
        split_chunk_num = arith.ceildivui(context_chunk_num, _to_raw(split_kv))
        full_end = context_chunk_num * BKV
        split_cols = split_chunk_num * BKV
        tile_lo_col = pid_split_kv * split_cols
        tile_hi_col = arith.minsi(_to_raw(tile_lo_col + split_cols), _to_raw(full_end))
        ctx_m1 = context_length - 1

        tile_lo = _to_raw(fx.Index(tile_lo_col))
        tile_hi = _to_raw(fx.Index(fx.Int32(tile_hi_col)))
        tile_step = _to_raw(fx.Index(fx.Int32(BKV)))
        tile_loop = scf.ForOp(tile_lo, tile_hi, tile_step, [])
        with ir.InsertionPoint(tile_loop.body):
            col0 = fx.Int32(arith.index_cast(T.i32, tile_loop.induction_variable))
            wave_ni_base = wave * N_TILES_PER_WAVE
            for ni in range_constexpr(N_TILES_PER_WAVE):
                abs_ni = wave_ni_base + ni
                col = col0 + abs_ni * MFMA_N + lane_mod_N
                # Clamp the gather index to a valid token (mask handles the rest).
                col_c = fx.Int32(arith.minsi(_to_raw(col), _to_raw(ctx_m1)))
                ind_off = pid_batch * max_block_len + col_c
                physical = fx.Int32(ind_t[ind_off])
                tok_byte = physical * index_dim

                b_col = [None] * K_STEPS
                for kk in range_constexpr(K_STEPS):
                    d_b = kk * MFMA_K + lane8
                    raw = load_pack_i64(kv_i32, tok_byte + d_b)
                    b_col[kk] = fn_to_fnuz_i64(raw) if convert_kv_fn else raw

                # Co-packed per-token scale: f32 at byte tok_byte + D.
                scale_dword = fx.Int32(
                    arith.divui(_to_raw(tok_byte + D), _to_raw(fx.Int32(4)))
                )
                kv_scale = _to_raw(fx.Float32(kv_f32[scale_dword]))
                if scale_mul != 1.0:
                    kv_scale = arith.MulFOp(
                        kv_scale, scale_mul_c, fastmath=fm_fast
                    ).result

                col_sum = mfma_head_reduce(
                    a_pack,
                    b_col,
                    w_frag,
                    kv_scale,
                    m_tiles=M_TILES,
                    k_steps=K_STEPS,
                    res_ty=mfma_res_ty,
                    f32_0=f32_0,
                    fm_fast=fm_fast,
                )

                # Only lane_div_N==0 lanes hold the MFMA_N distinct columns.
                # Write only in-causal columns; masked positions keep the
                # caller's -inf prefill (matches clean_logits semantics).
                is_writer = arith.andi(
                    _to_raw(
                        arith.cmpi(
                            arith.CmpIPredicate.eq,
                            _to_raw(lane_div_N),
                            _to_raw(fx.Int32(0)),
                        )
                    ),
                    _to_raw(
                        arith.cmpi(
                            arith.CmpIPredicate.sle,
                            _to_raw(col),
                            _to_raw(q_limit),
                        )
                    ),
                )
                with ir.InsertionPoint(scf.IfOp(is_writer).then_block):
                    out_row_t[col] = fx.Float32(col_sum)
                    scf.YieldOp([])

            scf.YieldOp([])

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
        stream: fx.Stream = fx.Stream(None),
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
        ).launch(
            grid=(gx, 1, 1), block=(MR_BLOCK_THREADS, 1, 1), stream=stream
        )

    return launch_fp8_paged_mqa_logits


# Kernel variants: single-token-per-block, wave-split only ("paged_w<WPB>").
# WPB must divide the column-tile count BKV/16 (=8 at the default BKV=128).
KERNEL_VARIANTS = tuple(f"paged_w{w}" for w in (1, 2, 4))
DEFAULT_VARIANT = "paged_w4"


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
    variant: str = DEFAULT_VARIANT,
    scale_mul: float = 1.0,
    convert_q_fn: bool = False,
    convert_kv_fn: bool = False,
):
    """Return a cached, compiled FlyDSL paged launcher for the given config.

    ``num_heads``/``head_size`` are compile-time constants; ``variant`` is a
    ``paged_w<WPB>`` tag; ``convert_q_fn``/``convert_kv_fn`` mark an FP8 FN
    operand whose -0 (0x80) byte the kernel patches to FNUZ +0, and ``scale_mul``
    folds the 2x-per-converted-operand compensation into the co-packed scale.
    """
    launcher = _build_paged_kernel(
        num_heads=num_heads,
        head_size=head_size,
        block_kv=block_kv,
        waves_per_block=_variant_wpb(variant),
        scale_mul=scale_mul,
        convert_q_fn=convert_q_fn,
        convert_kv_fn=convert_kv_fn,
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
    ChunkK=256,
    SplitKV=None,
    WavePerEU=2,
    TotalCuCount=None,
    variant=None,
    stream=None,
):
    """FlyDSL paged FP8 MQA logits (decode) -- KVBlockSize==1 with SplitKV.

    Drop-in for the Triton ``deepgemm_fp8_paged_mqa_logits`` tensor contract.

    q_fp8:        [batch, next_n, heads, hidden_dim], dtype float8 (e4m3 fn/fnuz)
    kv_cache:     [num_blocks, KVBlockSize, 1, index_dim], dtype uint8, co-packed
                  fp8 K bytes (first KVBlockSize*hidden_dim) then f32 dequant
                  scales; index_dim == hidden_dim + 4 (+ optional 16B padding).
    weights:      [batch*next_n, heads], dtype float32
    out_logits:   [batch*next_n, max_model_len], dtype float32. MUST be prefilled
                  with -inf by the caller; the kernel writes only in-window
                  (causal) positions and leaves masked positions untouched.
    context_lens: [batch], dtype int32
    kv_indices:   [batch, max_block_len], dtype int32 block table (physical block
                  per logical position; KVBlockSize==1 => per-token page table).
    max_model_len: int, out_logits column count.
    SplitKV:      KV-column split count (grid = split_kv*batch*next_n). None =>
                  the production host formula (fills the device on small decode
                  grids); pass an int to override, 1 disables splitting. Splits
                  own disjoint, gap-free BKV-aligned column ranges (one writer
                  per column, no cross-CTA reduction).
    WavePerEU/TotalCuCount: inputs to the auto-SplitKV formula.
    ChunkK:       accepted for Triton-contract parity; ignored here (the FlyDSL
                  kernel tiles the KV window at a fixed ``block_kv`` width).

    Returns the same ``out_logits`` tensor (written in place).
    """
    if Preshuffle:
        raise NotImplementedError(
            "Preshuffle is not supported by the FlyDSL paged fp8_mqa_logits kernel."
        )
    if KVBlockSize != 1:
        raise NotImplementedError(
            "FlyDSL paged fp8_mqa_logits supports KVBlockSize==1 only "
            f"(got KVBlockSize={KVBlockSize})."
        )

    batch_size, next_n, num_heads, head_size = q_fp8.shape
    assert num_heads & (num_heads - 1) == 0, "num q. heads should be power of 2."
    assert head_size & (head_size - 1) == 0, "head size should be power of 2."

    num_blocks, block_size, one, index_dim = kv_cache.shape
    assert block_size == 1, (
        f"kv_cache KVBlockSize dim must be 1; got {block_size}."
    )
    assert one == 1, f"kv_cache head dim must be 1; got {one}."
    assert index_dim >= head_size + 4, (
        f"index_dim={index_dim} must hold {head_size} fp8 bytes + 4 scale bytes."
    )
    assert kv_cache.dtype == torch.uint8, (
        f"kv_cache must be uint8 co-packed bytes; got {kv_cache.dtype}."
    )

    # i32 gather-offset ceiling: the per-token byte base physical*index_dim is
    # computed in i32 (it rides the buffer voffset), so the whole cache pool must
    # be addressable in i32 bytes. A larger pool would need an i64/global gather.
    assert num_blocks * index_dim < 2**31, (
        f"num_blocks*index_dim={num_blocks * index_dim} exceeds the i32 "
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
    assert q_fp8.dtype in (_fnuz, _fn), (
        f"q_fp8 must be e4m3 fp8 (fnuz or fn); got {q_fp8.dtype}"
    )

    # gfx942 fp8 MFMA reads operands as e4m3 FNUZ (bias 8). Q may arrive as FN
    # (OCP, bias 7); patch its 0x80 byte and undo the implied 2x via scale_mul.
    # The co-packed KV in the cache is arch-native fp8 (fnuz on gfx942, fn on
    # gfx950), so it is never converted.
    gfx = get_gfx()
    convert_q_fn = gfx == "gfx942" and q_fp8.dtype == _fn
    convert_kv_fn = False
    scale_mul = 2.0 if convert_q_fn else 1.0

    variant = _resolve_variant(variant)

    launcher = compile_fp8_paged_mqa_logits(
        num_heads=num_heads,
        head_size=head_size,
        block_kv=_BLOCK_KV,
        variant=variant,
        scale_mul=scale_mul,
        convert_q_fn=convert_q_fn,
        convert_kv_fn=convert_kv_fn,
    )

    # Co-packed cache -> raw uint8 byte view [num_blocks * index_dim].
    kv_bytes = kv_cache.reshape(-1)

    # KV-column splits (one lever): fill the device when the batch*next_n row
    # grid is small. Auto == production host formula; overridable.
    if SplitKV is None:
        split_kv = _auto_split_kv(
            batch_size, next_n, WavePerEU, q_fp8.device.index, TotalCuCount
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
