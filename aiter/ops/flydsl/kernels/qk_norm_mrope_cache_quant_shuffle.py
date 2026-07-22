# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused RMSNorm + interleaved 3D-mrope RoPE + per-tensor FP8 quant, with a
*coalesced* write into a shuffle-layout paged KV cache (FlyDSL, wave64).

This is the production counterpart of
``aiter.fused_qk_norm_mrope_3d_cache_pts_quant_shuffle`` (the HIP kernel in
``module_fused_qk_norm_mrope_cache_quant_shuffle``): same call signature and
semantics for the configuration this FlyDSL implementation supports, so it
can be dropped in wherever that op is called.

Two kernel launches, not one -- see the design note below for why.

Grid / thread mapping
----------------------
Q (built by ``_build_q_kernel``): grid = (num_heads_q, num_tokens), block =
one wave (``WAVE`` threads). Thread ``t`` in ``[0, WAVE)`` owns the NEOX
pair columns ``{t + k*WAVE, t + k*WAVE + head_size//2}`` for
``k in [0, VEC_PAIRS)``, where ``VEC_PAIRS = (head_size // 2) // WAVE``. No
cache write, so no write-amplification problem -- this stays a plain
one-wave-per-(token, head) kernel.

KV (built by ``_build_kv_kernel``): grid = (num_kv_heads, num_page_blocks),
block = ``KV_THREADS`` threads (``KV_THREADS // WAVE`` waves). K
*does* have a write-amplification problem: the shuffle-layout fp8 KV cache
interleaves ``x`` consecutive tokens in its innermost dim
(``K: [num_blocks, num_kv_heads, head_size/x, block_size, x]``,
``V: [num_blocks, num_kv_heads, block_size/x, head_size, x]``), so a
per-token kernel can only ever write isolated ``x``-strided bytes (see
``rope_common.h``'s ``get_shuffle_layout_v_base``). This kernel instead:

  1. computes RMSNorm+RoPE+quant (K) / raw+quant (V) for one page-block's
     ``block_size`` tokens x one kv-head, staging the resulting fp8 bytes
     into an LDS tile ``[block_size, head_size]`` per K/V (looped over
     ``block_size // (KV_THREADS // WAVE)`` compile-time iterations);
  2. after one ``gpu.barrier()``, re-purposes the same threads to copy the
     whole page-block's-worth of bytes from LDS to the cache as a run of
     16 B (``dwordx4``) coalesced global stores -- the exact granularity
     the shuffle layout's contiguous runs support.

Why two launches instead of one: Q's output is a plain contiguous
``[T, H_q, D]`` tensor with no write-amplification concern, so forcing it
through the KV kernel's page-block grid would only add complexity with no
HBM-traffic benefit. See ``op_tests/flydsl-best-practices.md`` S6.

Scope of this implementation (validated against the Qwen3-VL worst-case
config: ``H_q=64, H_k=H_v=4, D=128, mrope_section=[24,20,20]``): NEOX-style
interleaved 3D-mrope, per-tensor FP8 quant, shuffle-layout cache, full
(non-partial) rotary, non-Gemma RMSNorm, wave64 (gfx942/gfx950). Unsupported
combinations raise ``ValueError``/``NotImplementedError`` with a clear
message rather than silently producing wrong results -- see
``flydsl_fused_qk_norm_mrope_3d_cache_pts_quant_shuffle``'s validation.

``num_tokens`` need not be a multiple of ``block_size``: the KV kernel grids
``ceil_div(num_tokens, block_size)`` page-blocks and guards every per-token
read/write in the last (possibly ragged) block behind ``tok < num_tokens``,
zero-filling that block's unused LDS staging rows so the unconditional
Phase-2 coalesced copy writes deterministic zero padding rather than
uninitialized shared memory into the tail page's unused slots.
"""

import math
from functools import lru_cache
from typing import List, Optional

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr, vector
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from .tensor_shim import GTensor, _run_compiled

# --- fixed HW assumption: wave64 (gfx942 / gfx950 CDNA). gfx1250 is wave32
# and needs a dedicated variant (mirrors qk_norm_rope_quant_gfx1250.py) --
# not implemented here; the public API raises clearly on that arch.
WAVE = 64
_LOG2_WAVE = int(math.log2(WAVE))

# Threads per KV-fusion block: KV_THREADS // WAVE waves, each looping over
# the page block's tokens. 512 gives good occupancy and, for the common
# block_size=64/x=16/head_size=128 config, exactly matches both coalesced
# write grains (no tail iteration) -- see _build_kv_kernel.
KV_THREADS = 512

# fp8 = 1 byte/elem -> a HW dwordx4 (16 B) coalesced run covers 16 elements.
_RUN_BYTES = 16

# Typical LDS budget per CU on gfx942/gfx950 (64 KB); used only as a sanity
# check on (head_size, block_size) so oversized configs fail fast with a
# clear message instead of an opaque compiler/allocator error.
_MAX_LDS_BYTES = 65536


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


# ============================================================================
# Shuffle-layout byte-offset helpers (pure Python -- compile-time constants
# given (head_size, block_size, x, num_kv_heads); mirror
# ``set_kv_cache_shuffle_kernel`` / ``rope_common.h``).
# ============================================================================
def _k_head_stride(head_size, block_size):
    return head_size * block_size


def _k_per_block(head_size, block_size, num_kv_heads):
    return num_kv_heads * _k_head_stride(head_size, block_size)


def _v_head_stride(head_size, block_size, x):
    return (block_size // x) * head_size * x


def _v_per_block(head_size, block_size, x, num_kv_heads):
    return num_kv_heads * _v_head_stride(head_size, block_size, x)


# ============================================================================
# Q kernel builder
# ============================================================================
def _build_q_kernel(
    *,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_size: int,
    mrope_section: List[int],
    eps: float,
):
    H_Q, H_K, H_V, D = num_heads_q, num_heads_k, num_heads_v, head_size
    HALF = D // 2
    VEC_PAIRS = HALF // WAVE

    kname = f"qk_norm_mrope_q_H{H_Q}_D{D}_flydsl"

    @flyc.kernel(name=kname)
    def kernel(
        qkv: fx.Pointer,  # [T, H_Q+H_K+H_V, D] bf16, contig
        positions: fx.Pointer,  # [3, T] i64, contig (flat mid*T + tok)
        cos_sin: fx.Pointer,  # [max_pos, D] bf16 (cos=[:, :D/2], sin=[:, D/2:])
        q_norm_w: fx.Pointer,  # [D] bf16
        q_out: fx.Pointer,  # [T, H_Q, D] bf16
        num_tokens: fx.Int32,
    ):
        # NOTE: helpers are nested inside the @flyc.kernel body, not sibling
        # functions in _build_q_kernel -- FlyDSL's AST rewriter (dynamic
        # if/elif/else -> structured dispatch, `and`/`or` ->
        # __dsl_and__/__dsl_or__) only rewrites the decorated function's own
        # source text; nested `def`s are included, plain sibling functions
        # merely *called* from it are not. See flydsl-best-practices.md S1.
        fm_fast = arith.FastMathFlags.fast

        def wave_reduce_add(x):
            v = x
            for sh_exp in range_constexpr(_LOG2_WAVE):
                off = WAVE // (2 << sh_exp)
                peer = v.shuffle_xor(off, WAVE)
                v = v.addf(peer, fastmath=fm_fast)
            return v

        def mrope_cos_sin(col, tok, positions_t, cos_sin_t):
            """Interleaved 3D-mrope cos/sin gather for column ``col`` in
            [0, HALF). Mirrors ``apply_interleaved_rope``: column ``col``
            takes its position from section ``col % 3`` if that section
            "owns" the column (``col < mrope_section[col % 3] * 3``), else
            falls back to section 0 (temporal). Two ``select()`` ternaries,
            no branching -- valid for any ``col``, not just ``col < WAVE``
            (this is a pure function of the column value)."""
            mid = col % fx.Int32(3)
            is_mid1 = mid == fx.Int32(1)
            boundary = is_mid1.select(
                fx.Int32(mrope_section[1] * 3), fx.Int32(mrope_section[2] * 3)
            )
            in_range = col < boundary
            use_mid = (mid != fx.Int32(0)) and in_range
            sect_idx = use_mid.select(mid, fx.Int32(0))

            pos_i64 = fx.Int64(positions_t[sect_idx * num_tokens + tok])
            pos = pos_i64.to(fx.Int32)
            cos_v = fx.BFloat16(cos_sin_t[pos, col]).to(fx.Float32)
            sin_v = fx.BFloat16(cos_sin_t[pos, col + HALF]).to(fx.Float32)
            return cos_v, sin_v

        bid_x = fx.block_idx.x  # 0..H_Q-1
        bid_t = fx.block_idx.y  # token id
        tid = fx.thread_idx.x  # 0..WAVE-1

        qkv_t = GTensor(qkv, dtype=T.bf16, shape=(-1, H_Q + H_K + H_V, D))
        positions_t = GTensor(positions, dtype=T.i64, shape=(1,))
        cos_sin_t = GTensor(cos_sin, dtype=T.bf16, shape=(1, D))
        qw_t = GTensor(q_norm_w, dtype=T.bf16, shape=(D,))
        q_out_t = GTensor(q_out, dtype=T.bf16, shape=(-1, H_Q, D))

        # ---- Pass 1: load this thread's VEC_PAIRS pairs, reduce sum-sq over
        # the full D-wide row (RMSNorm scope) via one wave butterfly. ----
        x0s, x1s = [], []
        sumsq_local = fx.Float32(0.0)
        for k in range_constexpr(VEC_PAIRS):
            col = tid + WAVE * k
            x0 = fx.BFloat16(qkv_t[bid_t, bid_x, col]).to(fx.Float32)
            x1 = fx.BFloat16(qkv_t[bid_t, bid_x, col + HALF]).to(fx.Float32)
            x0s.append(x0)
            x1s.append(x1)
            sumsq_local = sumsq_local + x0 * x0 + x1 * x1
        sumsq = wave_reduce_add(sumsq_local)
        rstd = fmath.rsqrt(sumsq * (1.0 / D) + eps, fastmath=fm_fast)

        # ---- Pass 2: per-pair weight + RoPE + store. ----
        for k in range_constexpr(VEC_PAIRS):
            col = tid + WAVE * k
            w0 = fx.BFloat16(qw_t[col]).to(fx.Float32)
            w1 = fx.BFloat16(qw_t[col + HALF]).to(fx.Float32)
            xn0 = x0s[k] * rstd * w0
            xn1 = x1s[k] * rstd * w1
            cos_v, sin_v = mrope_cos_sin(col, bid_t, positions_t, cos_sin_t)
            o0 = xn0 * cos_v - xn1 * sin_v
            o1 = xn1 * cos_v + xn0 * sin_v
            q_out_t[bid_t, bid_x, col] = o0.to(fx.BFloat16)
            q_out_t[bid_t, bid_x, col + HALF] = o1.to(fx.BFloat16)

    @flyc.jit
    def launch(
        qkv: fx.Pointer,
        positions: fx.Pointer,
        cos_sin: fx.Pointer,
        q_norm_w: fx.Pointer,
        q_out: fx.Pointer,
        num_tokens: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        k = kernel(qkv, positions, cos_sin, q_norm_w, q_out, num_tokens)
        k.launch(
            grid=(H_Q, fx.Index(num_tokens), 1),
            block=(WAVE, 1, 1),
            stream=stream,
        )

    return launch


# ============================================================================
# Fused KV kernel builder: compute (RMSNorm+RoPE+quant for K, raw+quant for
# V) into LDS, barrier, then coalesced write from LDS to the shuffle-layout
# cache. Grid = (kv_head, page_block); block = KV_THREADS threads.
# ============================================================================
def _build_kv_kernel(
    *,
    num_heads_q: int,
    num_heads_kv: int,  # num_heads_k == num_heads_v (validated by caller)
    head_size: int,
    mrope_section: List[int],
    eps: float,
    block_size: int,
    x: int,
    emit_flat_kv: bool,
):
    H_Q, H_KV, D = num_heads_q, num_heads_kv, head_size
    HALF = D // 2
    VEC_PAIRS = HALF // WAVE
    WAVES_PER_BLOCK = KV_THREADS // WAVE
    PHASE1_ITERS = _ceil_div(block_size, WAVES_PER_BLOCK)

    STAGE_ELEMS = block_size * D  # bytes (fp8 = 1 B/elem) per K or V tile
    K_TOTAL_RUNS = (D * block_size) // _RUN_BYTES  # dwordx4 runs to write (K)
    K_ITERS = _ceil_div(K_TOTAL_RUNS, KV_THREADS)
    V_TOTAL_RUNS = (block_size // x) * D  # dwordx4 runs to write (V)
    V_ITERS = _ceil_div(V_TOTAL_RUNS, KV_THREADS)

    @fx.struct
    class SharedStorage:
        k_lds: fx.Array[fx.Int8, STAGE_ELEMS, 16]
        v_lds: fx.Array[fx.Int8, STAGE_ELEMS, 16]

    _name_parts = [
        "qk_norm_mrope_kv_shuffle",
        f"H{H_KV}",
        f"D{D}",
        f"bs{block_size}",
        f"x{x}",
    ]
    if emit_flat_kv:
        _name_parts.append("kvout")
    _name_parts.append("flydsl")
    kname = "_".join(_name_parts)

    @flyc.kernel(name=kname, known_block_size=[KV_THREADS, 1, 1])
    def kernel(
        qkv: fx.Pointer,  # [T, H_Q+H_K+H_V, D] bf16, contig
        positions: fx.Pointer,  # [3, T] i64, contig (flat mid*T + tok)
        cos_sin: fx.Pointer,  # [max_pos, D] bf16
        k_norm_w: fx.Pointer,  # [D] bf16
        k_cache: fx.Pointer,  # shuffle-layout fp8 K cache (flat u8)
        v_cache: fx.Pointer,  # shuffle-layout fp8 V cache (flat u8)
        slot_mapping: fx.Pointer,  # [T] i64
        k_scale_ptr: fx.Pointer,  # [1] f32 (per-tensor scale; no host sync)
        v_scale_ptr: fx.Pointer,  # [1] f32
        k_out: fx.Pointer,  # [T, H_K, D] u8 flat (dummy unless emit_flat_kv)
        v_out: fx.Pointer,  # [T, H_V, D] u8 flat (dummy unless emit_flat_kv)
        num_tokens: fx.Int32,
    ):
        fm_fast = arith.FastMathFlags.fast

        def wave_reduce_add(v):
            for sh_exp in range_constexpr(_LOG2_WAVE):
                off = WAVE // (2 << sh_exp)
                peer = v.shuffle_xor(off, WAVE)
                v = v.addf(peer, fastmath=fm_fast)
            return v

        def mrope_cos_sin(col, tok, positions_t, cos_sin_t):
            mid = col % fx.Int32(3)
            is_mid1 = mid == fx.Int32(1)
            boundary = is_mid1.select(
                fx.Int32(mrope_section[1] * 3), fx.Int32(mrope_section[2] * 3)
            )
            in_range = col < boundary
            use_mid = (mid != fx.Int32(0)) and in_range
            sect_idx = use_mid.select(mid, fx.Int32(0))

            pos_i64 = fx.Int64(positions_t[sect_idx * num_tokens + tok])
            pos = pos_i64.to(fx.Int32)
            cos_v = fx.BFloat16(cos_sin_t[pos, col]).to(fx.Float32)
            sin_v = fx.BFloat16(cos_sin_t[pos, col + HALF]).to(fx.Float32)
            return cos_v, sin_v

        def quant_pair_fp8(v0, v1, scale):
            s0 = v0 / scale
            s1 = v1 / scale
            packed = fx.Int32(fx.rocdl.cvt_pk_fp8_f32(T.i32, s0, s1, fx.Int32(0), 0))
            byte0 = packed.to(fx.Int8)
            byte1 = (packed >> fx.Int32(8)).to(fx.Int8)
            return byte0, byte1

        def slot_i32(slot_tensor, tok):
            return fx.Int32(arith.trunci(T.i32, slot_tensor[tok]))

        head = fx.block_idx.x  # kv head 0..H_KV-1
        blk = fx.block_idx.y  # page-block index (contiguous-slot assumption)
        t = fx.thread_idx.x  # 0..KV_THREADS-1

        qkv_t = GTensor(qkv, dtype=T.bf16, shape=(-1, H_Q + 2 * H_KV, D))
        positions_t = GTensor(positions, dtype=T.i64, shape=(1,))
        cos_sin_t = GTensor(cos_sin, dtype=T.bf16, shape=(1, D))
        kw_t = GTensor(k_norm_w, dtype=T.bf16, shape=(D,))
        slot_t = GTensor(slot_mapping, dtype=T.i64, shape=(-1,))
        kscale_t = GTensor(k_scale_ptr, dtype=T.f32, shape=(1,))
        vscale_t = GTensor(v_scale_ptr, dtype=T.f32, shape=(1,))
        k_scale = fx.Float32(kscale_t[0])
        v_scale = fx.Float32(vscale_t[0])
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        k_lds = lds.k_lds
        v_lds = lds.v_lds

        tok0 = blk * block_size

        # ---------------- Phase 1: compute -> LDS stage ----------------
        # `num_tokens` need not be a multiple of `block_size` -- the very
        # last page-block (blk == num_page_blocks - 1) may have fewer than
        # `block_size` real tokens. `tok0` itself is always a valid row
        # (num_page_blocks = ceil_div(num_tokens, block_size) guarantees
        # tok0 = blk*block_size < num_tokens), but individual `tok` values
        # within that last block's `token_local` range can run past
        # `num_tokens`. Guard every `qkv`/`positions` read (and, in the
        # return_kv path, every `k_out`/`v_out` write, which are sized to
        # exactly `num_tokens` rows) behind `tok < num_tokens`, and zero-fill
        # the LDS staging row for out-of-range tokens instead of leaving it
        # uninitialized, so the unconditional Phase-2 coalesced copy (which
        # always moves a full `block_size` worth of bytes per page, since
        # that's the fixed physical page granularity) writes deterministic
        # zero padding into the tail page's unused slots rather than
        # propagating garbage shared memory.
        wid = t // WAVE
        lane = t % WAVE
        for it in range_constexpr(PHASE1_ITERS):
            token_local = wid + WAVES_PER_BLOCK * it
            if token_local < block_size:
                tok = tok0 + token_local
                k_row = token_local * D
                v_row = token_local * D
                if tok < num_tokens:
                    # GTensor is a Python-side wrapper, not a DSL value (see
                    # flydsl-best-practices.md S3) -- construct it fresh
                    # inside this dynamic branch (cheap: pointer + descriptor
                    # only) so the AST rewriter never has to thread it as
                    # if/else state.
                    if const_expr(emit_flat_kv):
                        k_out_t = GTensor(k_out, dtype=T.i8, shape=(-1, H_KV, D))
                        v_out_t = GTensor(v_out, dtype=T.i8, shape=(-1, H_KV, D))

                    # ---- K: RMSNorm (over full D) + interleaved-mrope RoPE +
                    # per-tensor fp8 quant, one wave butterfly + VEC_PAIRS
                    # loop per thread (see _build_q_kernel for the same
                    # pattern). ----
                    k0s, k1s = [], []
                    sumsq_local = fx.Float32(0.0)
                    for p in range_constexpr(VEC_PAIRS):
                        col = lane + WAVE * p
                        k0 = fx.BFloat16(qkv_t[tok, H_Q + head, col]).to(fx.Float32)
                        k1 = fx.BFloat16(qkv_t[tok, H_Q + head, col + HALF]).to(
                            fx.Float32
                        )
                        k0s.append(k0)
                        k1s.append(k1)
                        sumsq_local = sumsq_local + k0 * k0 + k1 * k1
                    sumsq = wave_reduce_add(sumsq_local)
                    rstd = fmath.rsqrt(sumsq * (1.0 / D) + eps, fastmath=fm_fast)

                    for p in range_constexpr(VEC_PAIRS):
                        col = lane + WAVE * p
                        w0 = fx.BFloat16(kw_t[col]).to(fx.Float32)
                        w1 = fx.BFloat16(kw_t[col + HALF]).to(fx.Float32)
                        xn0 = k0s[p] * rstd * w0
                        xn1 = k1s[p] * rstd * w1
                        cos_v, sin_v = mrope_cos_sin(col, tok, positions_t, cos_sin_t)
                        o0 = xn0 * cos_v - xn1 * sin_v
                        o1 = xn1 * cos_v + xn0 * sin_v
                        kb0, kb1 = quant_pair_fp8(o0, o1, k_scale)
                        k_lds[k_row + col] = kb0
                        k_lds[k_row + col + HALF] = kb1
                        if const_expr(emit_flat_kv):
                            k_out_t[tok, head, col] = kb0
                            k_out_t[tok, head, col + HALF] = kb1

                    # ---- V: raw + per-tensor fp8 quant (no norm/rope) ----
                    for p in range_constexpr(VEC_PAIRS):
                        col = lane + WAVE * p
                        v0 = fx.BFloat16(qkv_t[tok, H_Q + H_KV + head, col]).to(
                            fx.Float32
                        )
                        v1 = fx.BFloat16(
                            qkv_t[tok, H_Q + H_KV + head, col + HALF]
                        ).to(fx.Float32)
                        vb0, vb1 = quant_pair_fp8(v0, v1, v_scale)
                        v_lds[v_row + col] = vb0
                        v_lds[v_row + col + HALF] = vb1
                        if const_expr(emit_flat_kv):
                            v_out_t[tok, head, col] = vb0
                            v_out_t[tok, head, col + HALF] = vb1
                else:
                    # Tail page-block padding row: no real token here (would
                    # be OOB on qkv/positions, and on k_out/v_out which are
                    # sized to exactly num_tokens rows) -- zero-fill instead.
                    for p in range_constexpr(VEC_PAIRS):
                        col = lane + WAVE * p
                        k_lds[k_row + col] = fx.Int8(0)
                        k_lds[k_row + col + HALF] = fx.Int8(0)
                        v_lds[v_row + col] = fx.Int8(0)
                        v_lds[v_row + col + HALF] = fx.Int8(0)

        gpu.barrier()

        # ---------------- Phase 2a: K coalesced write, LDS -> global ----
        # Whole [head_size/x, block_size, x] region for this (head, block)
        # is HD*BS contiguous bytes; thread run r (0..K_TOTAL_RUNS-1) owns
        # 16 contiguous output bytes, which map to ONE token's contiguous
        # x-elem head chunk on the src side (see proto_flydsl_shuffle_write
        # .py's k_coalesced for the derivation). Looped so K_TOTAL_RUNS need
        # not be an exact multiple of KV_THREADS.
        block_id = slot_i32(slot_t, tok0) // block_size
        head_base = block_id * _k_per_block(D, block_size, H_KV) + head * _k_head_stride(
            D, block_size
        )
        k_lds_i32 = fx.recast_iter(fx.Int32, k_lds.ptr)
        for it in range_constexpr(K_ITERS):
            r = t + KV_THREADS * it
            if r < K_TOTAL_RUNS:
                # GTensor is a Python-side wrapper, not a DSL value -- build
                # it fresh inside this dynamic branch (see S3 note above).
                cache_k_i32 = GTensor(k_cache, dtype=T.i32, shape=(-1,))
                f0 = r * _RUN_BYTES
                chunk_k = f0 // (block_size * x)
                block_off = (f0 % (block_size * x)) // x
                d0 = chunk_k * x
                src_elem_i32 = (block_off * D + d0) // 4
                vec4_k = fx.ptr_load(
                    k_lds_i32 + src_elem_i32,
                    result_type=fx.Vector.make_type(_RUN_BYTES // 4, fx.Int32),
                )
                cache_k_i32.vec_store(
                    ((head_base + f0) // 4,), vec4_k, _RUN_BYTES // 4
                )

        # ---------------- Phase 2b: V coalesced write, LDS -> global -----
        # V_TOTAL_RUNS = (block_size/x) tiles * head_size positions; thread
        # run r gathers x LDS bytes at stride D (one per token in the tile)
        # and packs them into one 16 B contiguous cache run.
        v_head_base = block_id * _v_per_block(D, block_size, x, H_KV) + head * _v_head_stride(
            D, block_size, x
        )
        for it in range_constexpr(V_ITERS):
            r = t + KV_THREADS * it
            if r < V_TOTAL_RUNS:
                cache_v_i32 = GTensor(v_cache, dtype=T.i32, shape=(-1,))
                tile = r // D
                d = r % D
                v_dst_base = v_head_base + tile * (D * x) + d * x
                vals = [v_lds[(tile * x + j) * D + d] for j in range_constexpr(x)]
                vec_x = vector.from_elements(T.vec(x, T.i8), vals)
                vec4_v = vector.bitcast(T.vec(x // 4, T.i32), vec_x)
                cache_v_i32.vec_store((v_dst_base // 4,), vec4_v, x // 4)

    @flyc.jit
    def launch(
        qkv: fx.Pointer,
        positions: fx.Pointer,
        cos_sin: fx.Pointer,
        k_norm_w: fx.Pointer,
        k_cache: fx.Pointer,
        v_cache: fx.Pointer,
        slot_mapping: fx.Pointer,
        k_scale_ptr: fx.Pointer,
        v_scale_ptr: fx.Pointer,
        k_out: fx.Pointer,
        v_out: fx.Pointer,
        num_tokens: fx.Int32,
        num_page_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        k = kernel(
            qkv,
            positions,
            cos_sin,
            k_norm_w,
            k_cache,
            v_cache,
            slot_mapping,
            k_scale_ptr,
            v_scale_ptr,
            k_out,
            v_out,
            num_tokens,
        )
        k.launch(
            grid=(H_KV, fx.Index(num_page_blocks), 1),
            block=(KV_THREADS, 1, 1),
            stream=stream,
        )

    return launch


# ============================================================================
# Cached compile + public API
# ============================================================================
@lru_cache(maxsize=32)
def _compile_q(
    *, num_heads_q, num_heads_k, num_heads_v, head_size, mrope_section, eps
):
    return _build_q_kernel(
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        num_heads_v=num_heads_v,
        head_size=head_size,
        mrope_section=list(mrope_section),
        eps=eps,
    )


@lru_cache(maxsize=32)
def _compile_kv(
    *,
    num_heads_q,
    num_heads_kv,
    head_size,
    mrope_section,
    eps,
    block_size,
    x,
    emit_flat_kv,
):
    return _build_kv_kernel(
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_size=head_size,
        mrope_section=list(mrope_section),
        eps=eps,
        block_size=block_size,
        x=x,
        emit_flat_kv=emit_flat_kv,
    )


def flydsl_fused_qk_norm_mrope_3d_cache_pts_quant_shuffle(
    qkv: torch.Tensor,
    qw: torch.Tensor,
    kw: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    num_tokens: int,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_size: int,
    is_neox_style: bool,
    mrope_section_: List[int],
    is_interleaved: bool,
    eps: float,
    q_out: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    per_tensor_k_scale: torch.Tensor,
    per_tensor_v_scale: torch.Tensor,
    k_out: Optional[torch.Tensor],
    v_out: Optional[torch.Tensor],
    return_kv: bool,
    use_shuffle_layout: bool,
    block_size: int,
    x: int,
    rotary_dim: int = 0,
    gemma_norm: bool = False,
    stream: Optional[torch.cuda.Stream] = None,
) -> None:
    """FlyDSL drop-in for ``aiter.fused_qk_norm_mrope_3d_cache_pts_quant_shuffle``.

    Same signature and semantics as the HIP op for the configuration this
    kernel supports (see module docstring); writes results into the
    caller-provided ``q_out`` / ``k_cache`` / ``v_cache`` (and, when
    ``return_kv=True``, ``k_out`` / ``v_out``) in place, matching the HIP
    op's ``-> None`` convention. Unsupported ``(is_neox_style,
    is_interleaved, gemma_norm, use_shuffle_layout, rotary_dim, x)``
    combinations raise a clear error instead of silently mis-computing.

    Args:
        qkv: ``[T, H_q+H_k+H_v, D]`` (or ``[T, (H_q+H_k+H_v)*D]``) bf16,
            contiguous in the trailing ``(head, D)`` block.
        qw, kw: RMSNorm weights for Q / K, shape ``[D]``, bf16.
        cos_sin: ``[max_pos, D]`` bf16 (``cos = [:, :D/2]``, ``sin =
            [:, D/2:]``).
        positions: 3D-mrope position ids, shape ``[3, T]``, int64, contig.
        num_tokens: T (must equal ``qkv.shape[0]``).
        num_heads_q/k/v: per-rank head counts. ``num_heads_k`` must equal
            ``num_heads_v`` (every real GQA/MQA config satisfies this --
            K and V always share the same head grouping).
        head_size: D; must be a multiple of ``2*WAVE`` (=128 on wave64), so
            each of the ``head_size//2`` NEOX-pair columns is covered by an
            integer number of full-wave passes.
        is_neox_style: must be ``True`` (NEOX pair layout ``(i, i+D/2)``;
            GPT-J interleaved-pair layout is not implemented).
        mrope_section_: 3-entry list summing to ``head_size // 2``
            (temporal / height / width sections for interleaved 3D-mrope).
        is_interleaved: must be ``True`` (the interleaved column-reassignment
            scheme from ``apply_interleaved_rope``; the alternative
            "blocked" section layout is not implemented).
        eps: RMSNorm epsilon.
        q_out: ``[T, H_q, D]`` bf16 output (always bf16 -- Q is never
            quantized by this op, matching the HIP kernel).
        k_cache, v_cache: shuffle-layout fp8 paged KV cache buffers. Treated
            as flat byte buffers (addressed via ``block_size``/``x``/
            ``head_size``/``num_heads_k`` regardless of the tensor's
            declared torch shape, exactly like the HIP kernel with
            ``use_shuffle_layout=True``).
        slot_mapping: ``[T]`` int64, physical cache slot per token. Assumed
            *page-block contiguous* for a run of ``block_size`` tokens
            (i.e. token ``p*block_size + i`` for ``i in [0, block_size)``
            all land in one page, at consecutive in-page offsets) -- the
            standard prefill scatter pattern; this is what makes the
            coalesced write possible. Decode-style scattered slot mappings
            are not supported by this coalesced path. ``num_tokens`` need
            not be a multiple of ``block_size``: the final page-block may be
            a ragged tail (fewer than ``block_size`` real tokens); the
            kernel writes deterministic zero bytes into that page's unused
            trailing slots rather than reading/writing out of bounds.
        per_tensor_k_scale, per_tensor_v_scale: 0-d/1-elem float32 CUDA
            tensors (read on-device inside the kernel -- no host sync).
        k_out, v_out: optional ``[T, H_k, D]`` / ``[T, H_v, D]`` uint8 (fp8
            bit pattern) flat outputs, populated additionally when
            ``return_kv=True`` (debugging / testing parity with the
            non-cached path), from the same LDS-staged bytes as the cache
            write (no extra compute).
        return_kv: see ``k_out``/``v_out``.
        use_shuffle_layout: must be ``True`` (the only cache layout this
            kernel writes).
        block_size: KV cache page size (tokens/page).
        x: shuffle-layout innermost dim; must be 16 (fp8 -> 16 B / 16 elems,
            the ``dwordx4`` HW granularity this kernel is built around).
        rotary_dim: must be ``0`` or ``head_size`` (full rotary only;
            partial rotary -- rotating only a leading sub-range of the head
            -- is not implemented).
        gemma_norm: must be ``False`` (Gemma's ``(1+weight)`` RMSNorm gamma
            convention is not implemented).
        stream: torch CUDA stream to launch on; defaults to the current
            stream.
    """
    from aiter.jit.utils.chip_info import get_gfx as _get_gfx

    if _get_gfx() == "gfx1250":
        raise NotImplementedError(
            "flydsl_fused_qk_norm_mrope_3d_cache_pts_quant_shuffle is wave64-"
            "only (gfx942/gfx950); gfx1250 (wave32) needs a dedicated "
            "variant, analogous to qk_norm_rope_quant_gfx1250.py, which does "
            "not exist yet."
        )

    # ---- validate the flag combination against this kernel's scope ----
    if not is_neox_style:
        raise NotImplementedError(
            "is_neox_style=False (GPT-J interleaved-pair RoPE) is not "
            "implemented; only the NEOX pair layout (i, i+D/2) is supported."
        )
    if not is_interleaved:
        raise NotImplementedError(
            "is_interleaved=False (non-interleaved / 'blocked' 3D-mrope "
            "section layout) is not implemented."
        )
    if gemma_norm:
        raise NotImplementedError(
            "gemma_norm=True ((1+weight) RMSNorm gamma) is not implemented."
        )
    if not use_shuffle_layout:
        raise NotImplementedError(
            "use_shuffle_layout=False is not implemented -- this kernel only "
            "writes the shuffle-layout KV cache."
        )
    if rotary_dim not in (0, head_size):
        raise NotImplementedError(
            f"partial rotary (rotary_dim={rotary_dim} != head_size="
            f"{head_size}, and != 0) is not implemented; only full rotary "
            "is supported."
        )
    if x != 16:
        raise NotImplementedError(
            f"x={x} is not implemented; this kernel is built around the "
            "fp8 shuffle innermost dim x=16 (16 B dwordx4 runs)."
        )
    if num_heads_k != num_heads_v:
        raise ValueError(
            f"num_heads_k ({num_heads_k}) must equal num_heads_v "
            f"({num_heads_v}) -- K and V share one grid_x in the fused "
            "KV-cache-write kernel."
        )
    if head_size % (2 * WAVE) != 0:
        raise ValueError(
            f"head_size ({head_size}) must be a multiple of 2*WAVE "
            f"(={2 * WAVE}); got head_size={head_size}."
        )
    if len(mrope_section_) != 3:
        raise ValueError(
            f"mrope_section_ must have exactly 3 entries (3D-mrope), got "
            f"{len(mrope_section_)}"
        )
    if sum(mrope_section_) != head_size // 2:
        raise ValueError(
            f"sum(mrope_section_)={sum(mrope_section_)} must equal "
            f"head_size//2={head_size // 2}"
        )
    if block_size % x != 0:
        raise ValueError(f"block_size ({block_size}) must be a multiple of x ({x})")
    if (head_size * block_size) % 16 != 0:
        raise ValueError(
            f"head_size*block_size ({head_size * block_size}) must be a "
            "multiple of 16 (dwordx4 K-cache run size)"
        )
    if 2 * head_size * block_size > _MAX_LDS_BYTES:
        raise ValueError(
            f"head_size={head_size} x block_size={block_size} needs "
            f"{2 * head_size * block_size} B of LDS staging (K + V tiles), "
            f"exceeding the {_MAX_LDS_BYTES} B budget assumed here."
        )
    if num_tokens > 65535:
        raise NotImplementedError(
            f"num_tokens ({num_tokens}) exceeds the HW grid-Y limit (65535) "
            "used directly by the Q launch; chunking across multiple "
            "launches (as qk_norm_rope_quant.py does for its 1-D positions) "
            "is not implemented for the 3-D mrope position layout here."
        )
    if qkv.dtype != torch.bfloat16:
        raise TypeError(f"qkv must be bf16, got {qkv.dtype}")
    if qw.dtype != torch.bfloat16 or kw.dtype != torch.bfloat16:
        raise TypeError("qw/kw must be bf16")
    if positions.dtype != torch.int64:
        raise TypeError(f"positions must be int64, got {positions.dtype}")
    if positions.shape != (3, num_tokens):
        raise ValueError(f"positions shape {tuple(positions.shape)} != (3, {num_tokens})")
    if slot_mapping.dtype != torch.int64:
        raise TypeError(f"slot_mapping must be int64, got {slot_mapping.dtype}")
    if per_tensor_k_scale.dtype != torch.float32 or per_tensor_v_scale.dtype != torch.float32:
        raise TypeError("per_tensor_k_scale/per_tensor_v_scale must be float32")
    if return_kv and (k_out is None or v_out is None):
        raise ValueError("return_kv=True requires k_out and v_out to be provided")

    H_Q, H_K, H_V, D = num_heads_q, num_heads_k, num_heads_v, head_size
    qkv_flat = qkv.view(num_tokens, H_Q + H_K + H_V, D)
    if not (qkv_flat.stride(-1) == 1 and qkv_flat.stride(-2) == D):
        raise ValueError("qkv must be contiguous in the trailing (head, D) block")

    if stream is None:
        stream = torch.cuda.current_stream()
    fx_stream = fx.Stream(stream)

    def _ptr(t):
        return flyc.from_c_void_p(fx.Uint8, t.data_ptr())

    q_launch = _compile_q(
        num_heads_q=H_Q,
        num_heads_k=H_K,
        num_heads_v=H_V,
        head_size=D,
        mrope_section=tuple(mrope_section_),
        eps=eps,
    )
    _run_compiled(
        q_launch,
        _ptr(qkv_flat),
        _ptr(positions),
        _ptr(cos_sin),
        _ptr(qw),
        _ptr(q_out),
        num_tokens,
        fx_stream,
    )

    kv_launch = _compile_kv(
        num_heads_q=H_Q,
        num_heads_kv=H_K,
        head_size=D,
        mrope_section=tuple(mrope_section_),
        eps=eps,
        block_size=block_size,
        x=x,
        emit_flat_kv=return_kv,
    )
    # num_tokens need not be a multiple of block_size -- the last page-block
    # may be a ragged tail with fewer than block_size real tokens; the
    # kernel's Phase 1 guards every per-token read/write behind
    # `tok < num_tokens` and zero-fills the corresponding LDS staging rows
    # for the out-of-range tail positions (see _build_kv_kernel).
    num_page_blocks = _ceil_div(num_tokens, block_size)
    k_out_arg = k_out if return_kv else k_cache.new_empty(1)
    v_out_arg = v_out if return_kv else v_cache.new_empty(1)
    _run_compiled(
        kv_launch,
        _ptr(qkv_flat),
        _ptr(positions),
        _ptr(cos_sin),
        _ptr(kw),
        _ptr(k_cache),
        _ptr(v_cache),
        _ptr(slot_mapping),
        _ptr(per_tensor_k_scale),
        _ptr(per_tensor_v_scale),
        _ptr(k_out_arg),
        _ptr(v_out_arg),
        num_tokens,
        num_page_blocks,
        fx_stream,
    )
