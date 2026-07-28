# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Readable tile-programming reference for paged-attention fp8 decode.

K/V are fp8 e4m3 (FNUZ on gfx942, OCP on gfx950) fed straight into
``mfma_f32_16x16x32_fp8_fp8``; Q (bf16/f16) and the softmax probabilities P are
quantized to fp8 too. Scales fold out of the matmuls (q/key scale into the QK
score, value scale + 1/FP8_MAX into the epilogue); softmax max/sum stay f32.
``key_scale``/``value_scale`` are either a ``[1]`` per-tensor scalar or a
``[num_blocks, num_kv_heads, block_size]`` per-token tensor (chosen by rank).

``block_size`` (16/64) and ``head_dim`` (multiple of 64) are compile-time
constants. Layouts are logical, not production's preshuffle.

* ``query``        [num_seqs, num_q_heads, head_dim]  f16/bf16 (head_dim contiguous)
* ``key_cache``    [num_blocks, num_kv_heads, head_dim//16, block_size, 16]  fp8
* ``value_cache``  [num_blocks, num_kv_heads, block_size//16, head_dim, 16] (trans_v)
                   or [num_blocks, num_kv_heads, head_dim, block_size] (plain), by rank
* ``block_tables`` [num_seqs, max_blocks_per_seq]  int32
* ``context_lengths`` [num_seqs]  int32
* ``output``       [num_seqs, num_q_heads, head_dim]  same dtype as query

One CTA (4 waves) per (seq, kv_head) runs a flash-style online softmax over
256-token blocks; the 4 waves split tokens for Q.KT and head-dim for P.V, with
an LDS round-trip on P transposing ownership between the two MMAs.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.protocol import dsl_size_of
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, vector
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from flydsl.expr.vector import ReductionOp
from flydsl.runtime.device import get_rocm_arch

from . import dpp_utils
from .utils import cdiv, exp2_amdgcn_scalar, exp2_f32_fast, rcp_f32

MFMA_MNK = (
    16  # M = N = 16 for the MMA atom; also query rows handled per CTA (padded to 16)
)
MFMA_K = 32  # fp8 MFMA contracts K = 32 per instruction (mfma_f32_16x16x32_fp8_fp8)
WAVE = 64
LOG2E = 1.4426950408889634
KV_COMPUTE_BLOCK = 256


@functools.cache
def compile_pa_decode_tile(
    *,
    head_dim: int,
    query_group_size: int,
    block_size: int,
    num_partitions: int = 1,
    softmax_scale: float | None = None,
    query_dtype: str = "f16",
    per_token_kv: bool = False,
    query_length: int = 1,
    trans_v: bool = True,
    device_index: int | None = None,
):
    """Build the tile-programming PA-decode kernel + launch wrapper.

    ``block_size``, ``head_dim``, and ``query_dtype`` are compile-time
    constants. ``device_index`` keeps the cached launch/compiled function
    device-local. ``query_length`` (MTP) and ``query_group_size`` flatten into
    ``TOTAL_ROWS = query_length * query_group_size``, tiled into
    ``M_TILES = ceil(TOTAL_ROWS / 16)`` independent 16-row MFMA tiles; each
    extra M-tile duplicates loop-carried state, so VGPR/LDS/occupancy scale
    roughly linearly with ``M_TILES``.
    """
    del device_index
    is_gfx950 = "gfx95" in get_rocm_arch()
    FP8 = fx.Float8E4M3FN if is_gfx950 else fx.Float8E4M3FNUZ
    FP8_MAX = (
        448.0 if is_gfx950 else 240.0
    )  # max representable magnitude of the format above

    assert (
        head_dim % MFMA_MNK == 0
    ), f"head_dim {head_dim} must be a multiple of {MFMA_MNK}"
    assert block_size in (
        16,
        64,
    ), f"pa_decode_tile only supports block_size in (16, 64), got {block_size}"
    assert query_dtype in (
        "f16",
        "bf16",
    ), f"pa_decode_tile only supports query_dtype in ('f16', 'bf16'), got {query_dtype}"
    Q_DTYPE = fx.BFloat16 if query_dtype == "bf16" else fx.Float16

    assert (
        head_dim % 64 == 0
    ), f"pa_decode_tile only supports head_dim that's a multiple of 64, got {head_dim}"
    assert query_length >= 1, f"query_length must be >= 1, got {query_length}"
    # Flattened query-row axis (MTP outer, GQA head inner), tiled into 16-row M-tiles.
    TOTAL_ROWS = query_length * query_group_size
    M_TILES = cdiv(TOTAL_ROWS, MFMA_MNK)
    ROWS_PADDED = M_TILES * MFMA_MNK
    # PV layout: V=A, P=B -> output [head-dim (row), query-row (col=lane16)],
    # generalized over head_dim via the VHE_CHUNKS loop.
    NWARP = 4  # 4 waves / CTA
    TOK_PER_WARP = 64  # tokens each warp owns per compute block (matches production KV_COMPUTE_BLOCK)
    TILE_TOK = NWARP * TOK_PER_WARP  # 256 tokens / compute block
    PAGES_PER_CHUNK = (
        TOK_PER_WARP // block_size
    )  # pages spanned by one 64-token warp-chunk: 1 (bs=64) or 4 (bs=16)
    assert (
        head_dim % (NWARP * MFMA_MNK) == 0
    ), "head_dim must split across the 4 warps for PV"

    # head_dim splits into 16-element chunks (QK_CHUNK_ELEMS, one dwordx4 load);
    # RGROUP_QUARTERS of them make a 64-element fetch group, QKHE_LOOP groups total.
    RGROUP_QUARTERS = 4
    QK_CHUNK_ELEMS = 16
    QKHE_LOOP = head_dim // (RGROUP_QUARTERS * QK_CHUNK_ELEMS)
    assert (
        QKHE_LOOP >= 1
    ), f"head_dim {head_dim} must be at least {RGROUP_QUARTERS * QK_CHUNK_ELEMS}"
    N_SUBCHUNKS = QKHE_LOOP * (QK_CHUNK_ELEMS // 8)

    # Q-quant chunk width: NQCHUNK stays fixed at 16 (tied to `lane16`'s role
    # as the absmax butterfly width); QCHUNK scales with head_dim instead.
    NQCHUNK = 16
    QCHUNK = (
        head_dim // NQCHUNK
    )  # f16 elements per lane's load chunk (8 for head_dim=128, 4 for head_dim=64)

    VHE_CHUNKS = head_dim // (NWARP * MFMA_MNK)  # 2 for head_dim=128, 1 for head_dim=64

    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim**0.5)
    NP = int(num_partitions)  # context partitions (grid.z); compile-time constant

    BLOCK_THREADS = NWARP * WAVE  # 256

    # -- LDS layout (shared across the 4 warps) --
    # sQ: fp8[ROWS_PADDED,head_dim] staged+quantized query. sP: fp8[16,TILE_TOK]
    # quantized probs. sQscale: f32[ROWS_PADDED]. sLmax/sLsum: cross-warp scratch.
    # sVPage: V page-index broadcast. sKScale/sVScale/sVScaleMax: per-token K/V
    # scale staging. No sO/sM/sL/sCorr: PV output is register-resident/loop-carried
    # and stored straight to global (V=A/P=B swap).
    f32 = 4
    sQ_bytes = ROWS_PADDED * head_dim * 1  # fp8
    sP_off = sQ_bytes
    # +16B row padding breaks a 32-bank LDS conflict on the P-pack writes while
    # keeping the row 16B-aligned for PV's ds_read_b128.
    SP_ROW_BYTES = TILE_TOK + 16
    sP_bytes = MFMA_MNK * SP_ROW_BYTES  # fp8, padded rows
    sQscale_off = sP_off + sP_bytes
    NWARP_PAD = (
        NWARP + 1
    )  # coprime with 32 banks -> no LDS bank conflict on the cross-warp scratch
    # Phase-split slices sLmax per M-tile so all pass-1 writes share one barrier.
    sLmax_off = sQscale_off + ROWS_PADDED * f32
    sLsum_off = sLmax_off + M_TILES * MFMA_MNK * NWARP_PAD * f32
    # V page-index broadcast (V's page depends on `rgroup`, shared across warps).
    sVPage_off = sLsum_off + MFMA_MNK * NWARP_PAD * f32
    sVPage_bytes = NWARP * PAGES_PER_CHUNK * 4  # i32
    # Per-token KV-scale staging (per_token only), double-buffered so the tt+1
    # prefetch (into the other buffer) doesn't clobber the current tile's scales.
    KV_BUFS = 2 if per_token_kv else 1
    KV_BUF_STRIDE = 2 * NWARP * TOK_PER_WARP * f32  # k-region + v-region, one buffer
    sKScale_off = sVPage_off + sVPage_bytes
    sVScale_off = sKScale_off + NWARP * TOK_PER_WARP * f32
    sKVScale_bytes = KV_BUFS * KV_BUF_STRIDE if per_token_kv else 0
    sVScaleMax_off = sKScale_off + sKVScale_bytes
    sVScaleMax_bytes = (
        NWARP_PAD * f32 if per_token_kv else 0
    )  # m-independent: one cross-warp slot
    total_bytes = sVScaleMax_off + sVScaleMax_bytes

    # LDS blob: one i32 array carved into typed byte-offset views (see the
    # `_lds_*` helpers in the kernel). Every region size above is 4-byte
    # aligned, so `total_bytes // 4` covers the blob exactly.
    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Int32, total_bytes // 4, 16]

    @flyc.kernel(known_block_size=(BLOCK_THREADS, 1, 1))
    def pa_decode_tile_kernel(
        output_ptr: fx.Tensor,  # [num_seqs*query_length, num_q_heads, head_dim]  (written directly when NP==1)
        # per-partition partial outputs (combined by the reduce kernel when NP>1):
        pmax_ptr: fx.Tensor,  # [num_seqs*query_length, num_kv_heads, num_partitions, query_group_size]   row max
        psum_ptr: fx.Tensor,  # [num_seqs*query_length, num_kv_heads, num_partitions, query_group_size]   row sum
        pout_ptr: fx.Tensor,  # [num_seqs*query_length, num_kv_heads, num_partitions, query_group_size, head_dim] Q_DTYPE, normalized O_p/l_p
        query_ptr: fx.Tensor,  # [num_seqs*query_length, num_q_heads, head_dim] -- row = seq*query_length + qi (MTP position)
        key_cache_ptr: fx.Tensor,  # [num_blocks, num_kv_heads, head_dim//16, block_size, 16] (blocked, see module docstring)
        value_cache_ptr: fx.Tensor,  # [num_blocks, num_kv_heads, block_size//16, head_dim, 16] (blocked, see module docstring)
        block_tables_ptr: fx.Tensor,  # [num_seqs, max_blocks_per_seq]
        context_lengths_ptr: fx.Tensor,  # [num_seqs]
        key_scale_ptr: fx.Tensor,  # [1] per-tensor OR [num_blocks, num_kv_heads, block_size] per-token
        value_scale_ptr: fx.Tensor,  # same shape as key_scale_ptr
        max_blocks_per_seq: fx.Int32,
        stride_ks_block: fx.Int32,
        stride_ks_head: fx.Int32,
        stride_q_row: fx.Int32,
        stride_q_head: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        warp = tid // WAVE  # 0..NWARP-1
        lane = tid - warp * WAVE  # 0..63
        seq = fx.Int32(gpu.block_id("x"))
        kv_h = fx.Int32(gpu.block_id("y"))
        part = fx.Int32(gpu.block_id("z"))  # context partition handled by this CTA
        n_kv = fx.Int32(gpu.grid_dim.y)  # num_kv_heads == gridDim.y

        # fx.copy-based flat loaders: K/V/context_len over a raw pointer, Q over
        # a buffer resource.
        def _make_flat_loader(tensor_ptr, elem_ty, reg_width, copy_op):
            use_buffer_resource = isinstance(
                copy_op, fx.rocdl.CopyOpCDNA3BufferCopyType
            )
            copy_atom = fx.make_copy_atom(copy_op, elem_ty)
            reg = fx.make_rmem_tensor(fx.make_layout(reg_width, 1), elem_ty)
            base_iter = (
                fx.get_iter(fx.rocdl.make_buffer_tensor(tensor_ptr, max_size=True))
                if use_buffer_resource
                else fx.get_iter(tensor_ptr)
            )
            flat = fx.Tensor(fx.make_view(base_iter, fx.make_layout(1 << 30, 1)))
            div = fx.logical_divide(flat, fx.make_layout(1, 1))

            def _load(elem_idx):
                fx.copy(copy_atom, fx.slice(div, (None, elem_idx)), reg)
                return fx.Vector(fx.memref_load_vec(reg))

            return _load

        _k_load_fp8x16 = _make_flat_loader(
            key_cache_ptr, FP8, 16, fx.UniversalCopy128b()
        )
        _v_load_fp8x16 = _make_flat_loader(
            value_cache_ptr, FP8, 16, fx.UniversalCopy128b()
        )
        # Per-lane Q chunk (QCHUNK 16-bit elems) fetched in QLOAD_UNIT-wide
        # pieces (128b max per buffer load): head_dim=256 needs 2 pieces.
        QLOAD_UNIT = min(8, QCHUNK)
        N_QLOADS = QCHUNK // QLOAD_UNIT
        _q_copy_op = (
            fx.rocdl.BufferCopy128b() if QLOAD_UNIT == 8 else fx.rocdl.BufferCopy64b()
        )
        _q_load_chunk = _make_flat_loader(query_ptr, Q_DTYPE, QLOAD_UNIT, _q_copy_op)
        _ctxlen_load = _make_flat_loader(
            context_lengths_ptr, fx.Int32, 1, fx.rocdl.BufferCopy32b()
        )

        def _k_load16(byte_off):
            return _k_load_fp8x16(byte_off).bitcast(fx.Int64)

        def _v_load16(byte_off):
            return _v_load_fp8x16(byte_off).bitcast(fx.Int64)

        context_len = fx.Int32(_ctxlen_load(seq)[0])
        # Bound block_tables to its real extent: the last (partial) 256-token tile
        # can index a page past ceil(context/block_size); the bounded resource
        # returns page 0 for that out-of-range read instead of faulting (those
        # tail tokens are masked out anyway).
        bt_num_records_bytes = (
            fx.Index(gpu.grid_dim.x) * fx.Index(max_blocks_per_seq) * 4
        )  # int32 entries
        bt_rsrc = buffer_ops.create_buffer_resource(
            block_tables_ptr, max_size=False, num_records_bytes=bt_num_records_bytes
        )
        ks_rsrc = buffer_ops.create_buffer_resource(key_scale_ptr, max_size=True)
        vs_rsrc = buffer_ops.create_buffer_resource(value_scale_ptr, max_size=True)
        # Per-tensor: a single global scale, read once. Per-token: read
        # per-token instead (see _kv_scale_ops/_stage_kv_scale_to_lds below).
        if const_expr(not per_token_kv):
            key_scale = fx.Int32(
                buffer_ops.buffer_load(
                    ks_rsrc, arith.constant(0, type=T.i32), vec_width=1, is_scalar=True
                )
            ).bitcast(fx.Float32)
            value_scale = fx.Int32(
                buffer_ops.buffer_load(
                    vs_rsrc, arith.constant(0, type=T.i32), vec_width=1, is_scalar=True
                )
            ).bitcast(fx.Float32)

        num_tiles = cdiv(context_len, TILE_TOK)
        tiles_per_part = cdiv(num_tiles, NP)
        part_start = part * tiles_per_part
        part_end_raw = part_start + tiles_per_part
        part_end = arith.select(part_end_raw < num_tiles, part_end_raw, num_tiles)

        # One i8 blob carved into typed byte-offset pointers. `lds_base` is an
        # ir.Value pointer (safe inside scf control flow); the Python `lds`
        # handle is not.
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_base = fx.recast_iter(fx.Uint8, lds.buf.ptr)  # byte-addressed base

        def _lds_ptr(byte_off, elem_ty):
            # Pin the element-size alignment explicitly (a bare recast off the
            # byte base would gcd the alignment down to 1).
            p = fx.add_offset(lds_base, fx.make_int_tuple(byte_off))
            ptr_ty = fx.PointerType.get(
                elem_ty.ir_type, fx.AddressSpace.Shared, dsl_size_of(elem_ty)
            )
            return fx.recast_iter(ptr_ty, p)

        def _lds_load(byte_off, elem_ty, n):
            return fx.ptr_load(
                _lds_ptr(byte_off, elem_ty), result_type=fx.Vector.make_type(n, elem_ty)
            )

        def _lds_store(byte_off, elem_ty, vec):
            fx.ptr_store(vec, _lds_ptr(byte_off, elem_ty))

        c16 = 16
        rgroup = lane // c16  # 0..3: which quarter-wave (paired with warp -> query row)
        lane16 = lane - rgroup * c16  # 0..15: this row's head-dim chunk index

        TOK_CHUNK = NWARP * MFMA_MNK  # 64
        NCHUNK = TILE_TOK // TOK_CHUNK  # 4

        def _load_phys_scalar(page, vec_width=1):
            result = buffer_ops.buffer_load(
                bt_rsrc,
                seq * max_blocks_per_seq + page,
                vec_width=vec_width,
                is_scalar=True,
            )
            return fx.Int32(result) if vec_width == 1 else result

        def _v_page_fetch_and_stage(tt_i32):
            # V's page depends on `rgroup` (shared across warps): warp w fetches
            # its rgroup row and broadcasts via LDS (read back by _v_page_read_row).
            base_page = tt_i32 * TILE_TOK // block_size  # tile start is page-aligned
            fetched = _load_phys_scalar(
                base_page + warp * PAGES_PER_CHUNK, PAGES_PER_CHUNK
            )
            if lane == 0:
                fetched_vec = (
                    fx.Vector.from_elements([fx.Int32(fetched)], dtype=fx.Int32)
                    if const_expr(PAGES_PER_CHUNK == 1)
                    else fx.Vector(fetched)
                )
                _lds_store(
                    sVPage_off + warp * (PAGES_PER_CHUNK * 4), fx.Int32, fetched_vec
                )

        def _v_page_read_row():
            off = sVPage_off + rgroup * (PAGES_PER_CHUNK * 4)
            return _lds_load(off, fx.Int32, PAGES_PER_CHUNK)

        def _kv_buf_off(tt_val):
            # ping-pong buffer byte offset (0 when single-buffered).
            if const_expr(per_token_kv):
                return (tt_val & fx.Int32(1)) * KV_BUF_STRIDE
            return 0

        def _stage_kv_scale_to_lds(phys_vec, buf_off=0):
            if const_expr(block_size == 64):
                phys = fx.Int32(phys_vec[0])
                base_tok = lane16 * NCHUNK
                scale_idx = phys * stride_ks_block + kv_h * stride_ks_head + base_tok
                k_scale_vec = fx.Vector(
                    buffer_ops.buffer_load(
                        ks_rsrc, scale_idx, vec_width=NCHUNK, dtype=fx.Float32
                    )
                )
                v_scale_vec = fx.Vector(
                    buffer_ops.buffer_load(
                        vs_rsrc, scale_idx, vec_width=NCHUNK, dtype=fx.Float32
                    )
                )
                slot = (warp * TOK_PER_WARP + base_tok) * f32
                _lds_store(sKScale_off + buf_off + slot, fx.Float32, k_scale_vec)
                _lds_store(sVScale_off + buf_off + slot, fx.Float32, v_scale_vec)
            else:
                # block_size==16: each lane stages its own rgroup's page/token,
                # so the 4 sub-blocks stage in parallel across rgroup-groups.
                phys = fx.Int32(
                    vector.extract(
                        arith.unwrap(phys_vec),
                        static_position=[],
                        dynamic_position=[fx.Index(rgroup)],
                    )
                )
                scale_idx = phys * stride_ks_block + kv_h * stride_ks_head + lane16
                k_scale_scalar = fx.Float32(
                    buffer_ops.buffer_load(
                        ks_rsrc, scale_idx, vec_width=1, dtype=fx.Float32
                    )
                )
                v_scale_scalar = fx.Float32(
                    buffer_ops.buffer_load(
                        vs_rsrc, scale_idx, vec_width=1, dtype=fx.Float32
                    )
                )
                fx.rocdl.sched_barrier(fx.rocdl.mask_vmem_rd)
                slot = (warp * TOK_PER_WARP + rgroup * c16 + lane16) * f32
                _lds_store(
                    sKScale_off + buf_off + slot,
                    fx.Float32,
                    fx.Vector.from_elements([k_scale_scalar], dtype=fx.Float32),
                )
                _lds_store(
                    sVScale_off + buf_off + slot,
                    fx.Float32,
                    fx.Vector.from_elements([v_scale_scalar], dtype=fx.Float32),
                )

        def _load_scale_vec(base_off, a, buf_off=0):
            # This lane's 4 per-token scales for chunk `a` from an LDS scale region.
            slot = (warp * TOK_CHUNK + a * c16 + rgroup * 4) * f32
            return _lds_load(base_off + buf_off + slot, fx.Float32, 4)

        def _load_kv_scale_vecs(a, buf_off=0):
            return _load_scale_vec(sKScale_off, a, buf_off), _load_scale_vec(
                sVScale_off, a, buf_off
            )

        # -- raw dwordx4 K load (A operand) --
        # token = warp*TOK_CHUNK + a*c16 + lane16 (the softmax mask and P-pack
        # write position below must encode this same formula).
        def _k_ops(phys, a):
            within_page_tok = (a * c16 + lane16) % block_size
            ops = []
            for qkhe in range_constexpr(QKHE_LOOP):
                he_idx = qkhe * RGROUP_QUARTERS + rgroup
                base = (
                    ((phys * n_kv + kv_h) * QCHUNK + he_idx) * block_size
                    + within_page_tok
                ) * QK_CHUNK_ELEMS
                w = _k_load16(base)  # head[he_idx*16 : +16] -> k_step 2*qkhe, 2*qkhe+1
                if const_expr(block_size == 16):
                    # help the scheduler overlap the PAGES_PER_CHUNK gathered loads
                    fx.rocdl.sched_barrier(fx.rocdl.mask_vmem_rd)
                ops.extend([w[0], w[1]])
            return ops  # N_SUBCHUNKS i64 operands

        def _k_ops_flat(tt_i32):
            base_page = tt_i32 * TILE_TOK // block_size  # tile start is page-aligned
            fetched = _load_phys_scalar(
                base_page + warp * PAGES_PER_CHUNK, PAGES_PER_CHUNK
            )
            phys_vec = (
                fx.Vector.from_elements([fx.Int32(fetched)], dtype=fx.Int32)
                if const_expr(PAGES_PER_CHUNK == 1)
                else fx.Vector(fetched)
            )
            flat = []
            for a in range_constexpr(NCHUNK):
                phys = fx.Int32(phys_vec[(a * c16) // block_size])
                flat.extend(_k_ops(phys, a))
            if const_expr(head_dim == 64):
                fx.rocdl.sched_vmem(len(flat) // 2)

            return fx.Vector.from_elements(flat, dtype=fx.Int64), phys_vec

        # -- prologue: prefetch the first tile's K --
        num_tiles_m1 = num_tiles - 1
        start_safe = arith.select(part_start < num_tiles, part_start, num_tiles_m1)
        k_pf0, phys_vec0 = _k_ops_flat(start_safe)
        # V page-index prefetch, issued here too for the same overlap; the
        # LDS write is visible after the barrier below.
        _v_page_fetch_and_stage(start_safe)
        if const_expr(per_token_kv):
            _stage_kv_scale_to_lds(phys_vec0, _kv_buf_off(fx.Int32(start_safe)))

        # per_token_kv has no single global key_scale/value_scale: scale_qk
        # drops the key_scale factor (folded in per-token, see masked_chunks
        # below) and v_scale_f is unused (replaced by v_max_scaled).
        if const_expr(per_token_kv):
            scale_qk = fx.Float32(softmax_scale * LOG2E)
        else:
            scale_qk = fx.Float32(softmax_scale * LOG2E) * fx.Float32(key_scale)
            v_scale_f = fx.Float32(value_scale)
        NEG_INF = fx.Float32(float("-inf"))
        ZERO_F = fx.Float32(0.0)
        fm_contract = arith.FastMathFlags.contract
        # Softmax scores are finite or the -inf mask sentinel -- never NaN -- so
        # nnan lets maxnum lower to a bare v_max (no v_cmp_u NaN check + its s_nop
        # hazard) and fuse to v_max3. (ninf must NOT be set: -inf is load-bearing.)
        fm_nnan = arith.FastMathFlags.nnan

        def _row_off(byte_off, m_idx, width, elem_ty):
            return byte_off + m_idx * (width * dsl_size_of(elem_ty))

        def _ld1(byte_off, m_idx):
            return _lds_load(_row_off(byte_off, m_idx, 1, fx.Float32), fx.Float32, 1)[0]

        def _st1(byte_off, m_idx, val):
            _lds_store(
                _row_off(byte_off, m_idx, 1, fx.Float32),
                fx.Float32,
                fx.Vector.from_elements([val], dtype=fx.Float32),
            )

        # f32[16, NWARP] cross-warp scratch: scalar write at (row, warp), vec read of a row's NWARP valid slots.
        def _st_lw(base_off, row, w, val):
            off = base_off + (row * NWARP_PAD + w) * 4
            _lds_store(
                off, fx.Float32, fx.Vector.from_elements([val], dtype=fx.Float32)
            )

        def _ld_lw_row(base_off, row):
            off = base_off + row * (NWARP_PAD * 4)
            return _lds_load(off, fx.Float32, NWARP)

        def _f32_to_fp8_words(vf32):
            # f32 -> fp8 must use the HW cvt (arith.truncf to fp8 doesn't lower);
            # pack 4 f32 -> 1 i32 (4 fp8) via two cvt_pk_fp8_f32 calls.
            n = vf32.shape[0]
            words = []
            for i in range_constexpr(n // 4):
                b = i * 4
                lo = fx.rocdl.cvt_pk_fp8_f32(T.i32, vf32[b], vf32[b + 1], 0, False)
                words.append(
                    fx.rocdl.cvt_pk_fp8_f32(T.i32, vf32[b + 2], vf32[b + 3], lo, True)
                )
            return fx.Vector.from_elements(words, dtype=fx.Int32)

        def _st_words(byte_off, words):
            _lds_store(byte_off, fx.Int32, words)

        qh_local = warp * 4 + rgroup  # 0..15: this thread's query row within an M-tile

        # Each M-tile quantizes 16 rows of the flattened (MTP position, GQA
        # head) axis: `flat_idx = m*16 + qh_local`, decomposed as
        # `qi = flat_idx // query_group_size`, `gs_head = flat_idx %
        # query_group_size` (same convention as `_mtp_groups`). No
        # cross-M-tile dependency, so no barriers needed between iterations.
        def _quant_q_row(m, qi, gs_head, q_row_off):
            qh0 = kv_h * query_group_size + gs_head
            row_byte0 = (
                (seq * query_length + qi) * stride_q_row + qh0 * stride_q_head
            ) * 2  # 16-bit float = 2B/elem
            base_elem = (
                row_byte0 + lane16 * (QCHUNK * 2)
            ) // 2  # byte offset -> element index
            # QCHUNK elements, loaded as N_QLOADS contiguous QLOAD_UNIT pieces
            # (a buffer load is 128b max); head_dim=256 splits into 2 pieces.
            q_units = [
                _q_load_chunk(base_elem + u * QLOAD_UNIT)
                for u in range_constexpr(N_QLOADS)
            ]

            absmax = fmath.absf(q_units[0]).reduce(ReductionOp.MAX).to(fx.Float32)
            for u in range_constexpr(1, N_QLOADS):
                absmax = fx.maxnumf(
                    absmax,
                    fmath.absf(q_units[u]).reduce(ReductionOp.MAX).to(fx.Float32),
                )
            for sh in (8, 4, 2, 1):
                absmax = fx.maxnumf(absmax, dpp_utils.dpp_xor_f32(absmax, sh))

            q_scale = absmax * fx.Float32(1.0 / FP8_MAX)
            inv = fx.Float32(rcp_f32(fx.maxnumf(q_scale, fx.Float32(1e-20))))
            inv_b = fx.Vector.from_elements([inv], dtype=fx.Float32).broadcast_to(
                QLOAD_UNIT
            )

            for u in range_constexpr(N_QLOADS):
                q_scaled_unit = q_units[u].to(fx.Float32) * inv_b
                _st_words(
                    q_row_off + qh_local * head_dim + lane16 * QCHUNK + u * QLOAD_UNIT,
                    _f32_to_fp8_words(q_scaled_unit),
                )
            if lane16 == 0:
                # Transposed [qh][m] (not [m][qh]) so the whole M_TILES-wide
                # row for a fixed qh is contiguous, letting the KV-loop read
                # it back in one wide load instead of M_TILES separate
                # narrow ones -- see the read site below.
                _st1(sQscale_off, qh_local * M_TILES + m, q_scale)

        for m in range_constexpr(M_TILES):
            flat_idx = m * MFMA_MNK + qh_local
            qi = flat_idx // query_group_size
            gs_head = flat_idx - qi * query_group_size
            q_row_off = m * MFMA_MNK * head_dim
            # `flat_idx < TOTAL_ROWS` is statically true for every lane except
            # possibly on the last M-tile (only it can be a partial tile), so
            # skip the runtime branch (and its EXEC-mask overhead) entirely
            # for every other M-tile.
            if const_expr((m + 1) * MFMA_MNK <= TOTAL_ROWS):  # noqa: SIM114
                _quant_q_row(m, qi, gs_head, q_row_off)
            elif flat_idx < TOTAL_ROWS:
                _quant_q_row(m, qi, gs_head, q_row_off)
            else:
                _st_words(
                    q_row_off + qh_local * head_dim + lane16 * QCHUNK,
                    fx.Vector.filled(QCHUNK // 4, 0, fx.Int32),
                )
                if lane16 == 0:
                    _st1(sQscale_off, qh_local * M_TILES + m, ZERO_F)

        gpu.barrier()

        # First tile's V page-index row, now visible after the barrier above
        # (the fetch+LDS-store was issued earlier, alongside k_pf0).
        v_page_pf0 = _v_page_read_row()

        # Q is the B operand, read raw from sQ once per M-tile and held in
        # registers. MUST use the exact same (qkhe, rgroup, qkr) -> head_dim
        # permutation as K's `_k_ops`.
        q_ops_all = []
        for m in range_constexpr(M_TILES):
            q_row_off = m * MFMA_MNK * head_dim
            for qkhe in range_constexpr(QKHE_LOOP):
                he_idx = qkhe * RGROUP_QUARTERS + rgroup
                chunk = _lds_load(
                    q_row_off + lane16 * head_dim + he_idx * QK_CHUNK_ELEMS, fx.Int64, 2
                )
                q_ops_all.extend([chunk[0], chunk[1]])
        # q_ops_all[m*N_SUBCHUNKS+s] for s=0..N_SUBCHUNKS-1, s = qkhe*2+qkr,
        # = M-tile m's head[he_idx*16+qkr*8 : +8] of qhead=lane16

        # QK in NCHUNK chunks of 4 tokens: each chunk yields a f32x4
        # C-fragment, so softmax processes 4 scores at a time (low VGPR peak).
        _ct = [
            fx.Vector.from_elements([float(a * c16 + r) for r in range_constexpr(4)])
            for a in range_constexpr(NCHUNK)
        ]
        # P.V is loop-tiled over head-dim (like production's VHELOOP): each
        # step computes O[:, vh*VHE_SIZE:+VHE_SIZE] instead of materializing
        # the full [16, head_dim] at once.
        VHE_SIZE = head_dim // VHE_CHUNKS
        OP_ELEMS = (
            MFMA_MNK * VHE_SIZE // (NWARP * WAVE)
        )  # PV C-fragment elements/lane/chunk (probed = 4)

        # -- raw dwordx4 V load (B operand) --
        # One dwordx4 load per (16-token sub-block, head_elem); trans_v only
        # changes the offset formula. `sub`/`step` walk pages/16-token sub-blocks.
        NVOPS = TILE_TOK // MFMA_K  # 8 PV k_steps (256 tokens / K=32)
        STEPS_PER_PAGE = block_size // 16

        def _v_ops(phys_row, vh):
            head_group = ((vh * VHE_SIZE) // 16) + warp
            head_element = head_group * 16 + lane16
            ops = []
            for sub in range_constexpr(PAGES_PER_CHUNK):
                for step in range_constexpr(STEPS_PER_PAGE):
                    if const_expr(trans_v):
                        base = (
                            ((phys_row[sub] * n_kv + kv_h) * STEPS_PER_PAGE + step)
                            * head_dim
                            + head_element
                        ) * 16
                    else:
                        base = (
                            (phys_row[sub] * n_kv + kv_h) * head_dim + head_element
                        ) * block_size + step * 16
                    w = _v_load16(base)
                    if const_expr(block_size == 16):
                        # help the scheduler overlap the per-page gathered loads (see _k_ops)
                        fx.rocdl.sched_barrier(fx.rocdl.mask_vmem_rd)
                    ops.extend([w[0], w[1]])
            if const_expr(head_dim == 64):
                fx.rocdl.sched_vmem(len(ops) // 2)
            return ops  # NVOPS i64, the 64-token contiguous run for this head

        # query_length==1: plain wave-uniform `context_len` (folding in the
        # per-lane `lane16` would cost a live VGPR for no behavioral change).
        if const_expr(query_length == 1):
            causal_bound = [context_len for _m in range_constexpr(M_TILES)]
        else:
            causal_bound = [
                context_len
                - (query_length - 1)
                + (m * MFMA_MNK + lane16) // query_group_size
                for m in range_constexpr(M_TILES)
            ]

        K_SLOT, V_SLOT = 0, 1
        # Per-M-tile loop-carried state after the K/V slots: one output chunk
        # per VHE_CHUNKS (2 for head_dim=128, 4 for head_dim=256, 1 for 64),
        # then running-max + running-denom.
        STATE_PER_M = VHE_CHUNKS + 2

        def _o_slot(m, vh):
            return 2 + STATE_PER_M * m + vh

        def _m_slot(m):
            return 2 + STATE_PER_M * m + VHE_CHUNKS

        def _l_slot(m):
            return 2 + STATE_PER_M * m + VHE_CHUNKS + 1

        o_zero = fx.Vector.filled(OP_ELEMS, 0.0, fx.Float32)
        init_state = [k_pf0, v_page_pf0]
        for _m in range_constexpr(M_TILES):
            init_state.extend([o_zero] * VHE_CHUNKS + [NEG_INF, ZERO_F])
        for tt, ostate in range(part_start, part_end, 1, init=init_state):
            k_cur = ostate[
                K_SLOT
            ]  # this tile's prefetched K, as one (NCHUNK*N_SUBCHUNKS,) i64 vector
            v_page_cur = ostate[
                V_SLOT
            ]  # this tile's V pages, as one PAGES_PER_CHUNK-wide i32 vector
            tt = fx.Int32(tt)
            tok0 = tt * TILE_TOK
            # per_tensor phase-split: let IGLP interleave MFMA with softmax
            # VALU/LDS to hide the MFMA-hazard s_nop (per_token is VGPR-cliffed).
            if const_expr(not per_token_kv and M_TILES > 1):
                fx.rocdl.iglp_opt(0)

            tt1 = tt + 1

            next_state = [
                None,
                None,
            ]  # slots 0/1 (K_SLOT/V_SLOT) filled in at m==0 below

            # This tile's ping-pong scale buffer; the tt+1 prefetch stages the other.
            cur_kv_buf = _kv_buf_off(tt)

            # V data doesn't depend on `m`; hoist once for the phase-split
            # instead of reloading per M-tile.
            v_vh_shared = None
            if const_expr(M_TILES > 1):
                v_vh_shared = [
                    _v_ops(v_page_cur, vh) for vh in range_constexpr(VHE_CHUNKS)
                ]

            # q_scale doesn't depend on `m`; read the whole M_TILES-wide row once
            # (contiguous via the transposed [qh][m] sQscale layout).
            q_scale_vec = None
            if const_expr(M_TILES > 1):
                q_scale_vec = _lds_load(
                    sQscale_off + lane16 * (M_TILES * f32), fx.Float32, M_TILES
                )

            def _lmax_off_m(m):
                return sLmax_off + (
                    m * MFMA_MNK * NWARP_PAD * f32 if const_expr(M_TILES > 1) else 0
                )

            # Phase-split (M_TILES>1): pass-1 (QK+mask+max) loops all M-tiles into
            # per-M-tile LDS slices so they share ONE barrier. M_TILES==1: else below.
            if const_expr(M_TILES > 1):
                masked_chunks_saved = [None] * M_TILES
                scale_saved = [None] * M_TILES

                # per_token: compute the m-independent pv_max once, hold only
                # k_scale across Phase A, re-read v_scale for Phase B after the
                # barrier (peak scale liveness 16, not 32).
                k_scale_shared = None
                if const_expr(per_token_kv):
                    v_scale_A = [
                        _load_scale_vec(sVScale_off, a, cur_kv_buf)
                        for a in range_constexpr(NCHUNK)
                    ]
                    pv_max = fx.Float32(0.0)
                    for a in range_constexpr(NCHUNK):
                        pv_max = fx.maxnumf(
                            pv_max, v_scale_A[a].reduce(ReductionOp.MAX)
                        )
                    for sh in (16, 32):
                        pv_max = fx.maxnumf(pv_max, pv_max.shuffle_xor(sh, WAVE))
                    _st_lw(sVScaleMax_off, 0, warp, pv_max)
                    k_scale_shared = [
                        _load_scale_vec(sKScale_off, a, cur_kv_buf)
                        for a in range_constexpr(NCHUNK)
                    ]

                for m in range_constexpr(M_TILES):
                    frag_Ss = []
                    for a in range_constexpr(NCHUNK):
                        acc = arith.constant_vector(0.0, T.f32x4)
                        for s in range_constexpr(N_SUBCHUNKS):
                            acc = fx.rocdl.mfma_f32_16x16x32_fp8_fp8(
                                T.f32x4,
                                [
                                    k_cur[a * N_SUBCHUNKS + s],
                                    q_ops_all[m * N_SUBCHUNKS + s],
                                    acc,
                                    0,
                                    0,
                                    0,
                                ],
                            )
                        frag_Ss.append(fx.Vector(acc))

                    scale = scale_qk * fx.Float32(q_scale_vec[m])
                    n_valid_tile = (causal_bound[m] - tok0).to(fx.Float32)
                    base_tok_f = fx.Int32(warp * TOK_CHUNK + rgroup * 4).to(fx.Float32)
                    thr = fx.Vector.from_elements(
                        [n_valid_tile - base_tok_f], dtype=fx.Float32
                    ).broadcast_to(4)
                    neg4 = fx.Vector.filled(4, float("-inf"), fx.Float32)

                    if const_expr(per_token_kv):
                        scaled_frags = [
                            frag_Ss[a] * k_scale_shared[a]
                            for a in range_constexpr(NCHUNK)
                        ]
                    else:
                        scaled_frags = frag_Ss

                    masked_chunks = [
                        (_ct[a] < thr).select(scaled_frags[a], neg4)
                        for a in range_constexpr(NCHUNK)
                    ]

                    pm = fx.Float32(float("-inf"))
                    for a in range_constexpr(NCHUNK):
                        pm = fx.maxnumf(
                            pm,
                            masked_chunks[a].reduce(ReductionOp.MAX, fastmath=fm_nnan),
                            fastmath=fm_nnan,
                        )
                    for sh in (16, 32):
                        pm = fx.maxnumf(pm, pm.shuffle_xor(sh, WAVE), fastmath=fm_nnan)
                    _st_lw(_lmax_off_m(m), lane16, warp, pm * scale)

                    masked_chunks_saved[m] = masked_chunks
                    scale_saved[m] = scale

                # tt+1 K/V/scale prefetch, issued once here so the V-page read
                # reuses this barrier.
                k_next = k_cur
                if tt1 < part_end:
                    k_next, phys_vec1 = _k_ops_flat(tt1)
                    _v_page_fetch_and_stage(tt1)
                    if const_expr(per_token_kv):
                        # stage tt+1 into the OTHER buffer (current v_scale re-read below)
                        _stage_kv_scale_to_lds(phys_vec1, _kv_buf_off(tt1))
                next_state[K_SLOT] = k_next

                gpu.barrier()

                v_page_next = v_page_cur
                if tt1 < part_end:
                    v_page_next = _v_page_read_row()
                next_state[V_SLOT] = v_page_next

                # Phase B v_scale: M_TILES>=4 re-reads per chunk (VGPR cliff);
                # smaller M_TILES hold all NCHUNK. Both read the current buffer.
                v_scale_shared = None
                if const_expr(per_token_kv and M_TILES < 4):
                    v_scale_shared = [
                        _load_scale_vec(sVScale_off, a, cur_kv_buf)
                        for a in range_constexpr(NCHUNK)
                    ]

                for m in range_constexpr(M_TILES):
                    o_acc = [
                        ostate[_o_slot(m, vh)] for vh in range_constexpr(VHE_CHUNKS)
                    ]
                    m_prev = ostate[
                        _m_slot(m)
                    ]  # this thread's own running max, carried from last tile
                    l_prev = ostate[
                        _l_slot(m)
                    ]  # this thread's own running denom, carried from last tile

                    masked_chunks = masked_chunks_saved[m]
                    scale = scale_saved[m]

                    v_max_scaled = None
                    norm_factor_b = None
                    if const_expr(per_token_kv):
                        v_max_global = _ld_lw_row(sVScaleMax_off, 0).reduce(
                            ReductionOp.MAX
                        )
                        v_max_scaled = v_max_global * fx.Float32(1.0 / FP8_MAX)
                        v_max_safe = v_max_scaled + fx.Float32(1e-8 / FP8_MAX)
                        norm_factor = fx.Float32(rcp_f32(v_max_safe))
                        norm_factor_b = fx.Vector.from_elements(
                            [norm_factor], dtype=fx.Float32
                        ).broadcast_to(4)

                    m_new = fx.maxnumf(
                        m_prev,
                        _ld_lw_row(_lmax_off_m(m), lane16).reduce(
                            ReductionOp.MAX, fastmath=fm_nnan
                        ),
                        fastmath=fm_nnan,
                    )
                    # Fully-invalid row: use 0 as the effective max so masked lanes
                    # give exp2(-inf-0)==0 (avoids the -inf-(-inf) cancellation).
                    safe_max = arith.select(m_new > NEG_INF, m_new, ZERO_F)
                    m_new_b = fx.Vector.from_elements(
                        [safe_max], dtype=fx.Float32
                    ).broadcast_to(4)
                    ls = fx.Float32(0.0)
                    words = []
                    for a in range_constexpr(NCHUNK):
                        Pa = fx.Vector(
                            exp2_f32_fast(masked_chunks[a] * scale - m_new_b)
                        )
                        ls = ls + Pa.reduce(ReductionOp.ADD)
                        if const_expr(per_token_kv):
                            v_sc = (
                                _load_scale_vec(sVScale_off, a, cur_kv_buf)
                                if const_expr(M_TILES >= 4)
                                else v_scale_shared[a]
                            )
                            p_scaled = Pa * v_sc * norm_factor_b
                        else:
                            p_scaled = Pa * fx.Vector.filled(4, FP8_MAX, fx.Float32)
                        words.append(_f32_to_fp8_words(p_scaled)[0])

                    p_off0 = (
                        sP_off + lane16 * SP_ROW_BYTES + warp * TOK_CHUNK + rgroup * 4
                    )
                    # The NCHUNK P words scatter across the row at stride c16//4
                    # i32 (the token->fp8-lane interleave the PV ds_read_b128
                    # expects); one strided store per chunk.
                    for a in range_constexpr(NCHUNK):
                        _lds_store(
                            p_off0 + a * (c16 // 4) * f32,
                            fx.Int32,
                            fx.Vector.from_elements([words[a]], dtype=fx.Int32),
                        )
                    for sh in (16, 32):
                        ls = ls.addf(ls.shuffle_xor(sh, WAVE), fastmath=fm_contract)
                    # PV output is [head-dim, query-row=lane16] after the operand
                    # swap, so correction/denominator are per-lane scalars (no sCorr).
                    safe_prev = arith.select(m_prev > NEG_INF, m_prev, ZERO_F)
                    corr_reg = fx.Float32(exp2_amdgcn_scalar(safe_prev - safe_max))
                    if rgroup == 0:
                        _st_lw(sLsum_off, lane16, warp, ls)
                    gpu.barrier()
                    gsum = _ld_lw_row(sLsum_off, lane16).reduce(ReductionOp.ADD)
                    l_new = fx.Float32(
                        arith.mulf(
                            arith.unwrap(l_prev),
                            arith.unwrap(corr_reg),
                            fastmath=fm_contract,
                        )
                    ).addf(gsum, fastmath=fm_contract)

                    p_ops = _lds_load(
                        sP_off + lane16 * SP_ROW_BYTES + rgroup * 64, fx.Int64, NVOPS
                    )

                    corr_b = fx.Vector.from_elements(
                        [corr_reg], dtype=fx.Float32
                    ).broadcast_to(OP_ELEMS)
                    for vh in range_constexpr(VHE_CHUNKS):
                        v_vh = v_vh_shared[vh]
                        acc = arith.constant_vector(0.0, T.f32x4)
                        for s in range_constexpr(NVOPS):
                            # SWAPPED operands (V=A, P=B): output row =
                            # head-dim, output col = query-row=lane16.
                            acc = fx.rocdl.mfma_f32_16x16x32_fp8_fp8(
                                T.f32x4, [v_vh[s], p_ops[s], acc, 0, 0, 0]
                            )
                        op = fx.Vector(acc)
                        if const_expr(per_token_kv):
                            op = op * fx.Vector.from_elements(
                                [v_max_scaled], dtype=fx.Float32
                            ).broadcast_to(OP_ELEMS)
                        o_acc[vh] = o_acc[vh] * corr_b + op
                    next_state.extend([*o_acc, m_new, l_new])
                    # sP and sLsum are reused by the next M-tile. Synchronize
                    # all waves after their LDS reads before any wave overwrites
                    # those regions, then retire this tile's dependency chain.
                    if const_expr(m < M_TILES - 1):
                        gpu.barrier()
                        fx.rocdl.sched_barrier(0)
            else:
                # M_TILES==1 single tile (m==0 for the _o_slot/_m_slot/_l_slot helpers).
                o_acc = [ostate[_o_slot(0, vh)] for vh in range_constexpr(VHE_CHUNKS)]
                m_prev = ostate[_m_slot(0)]  # running max, carried from last tile
                l_prev = ostate[_l_slot(0)]  # running denom, carried from last tile
                # QK: each NCHUNK chunk accumulates N_SUBCHUNKS k_steps into an f32x4.
                frag_Ss = []
                for a in range_constexpr(NCHUNK):
                    acc = arith.constant_vector(0.0, T.f32x4)
                    for s in range_constexpr(N_SUBCHUNKS):
                        acc = fx.rocdl.mfma_f32_16x16x32_fp8_fp8(
                            T.f32x4,
                            [k_cur[a * N_SUBCHUNKS + s], q_ops_all[s], acc, 0, 0, 0],
                        )
                    frag_Ss.append(fx.Vector(acc))
                # tt+1 K/V/scale prefetch, issued here to reuse the pass-1 barrier.
                k_next = k_cur
                if tt1 < part_end:
                    k_next, phys_vec1 = _k_ops_flat(tt1)
                    _v_page_fetch_and_stage(tt1)
                    if const_expr(per_token_kv):
                        _stage_kv_scale_to_lds(phys_vec1, _kv_buf_off(tt1))
                next_state[K_SLOT] = k_next
                # Softmax: each lane owns one qhead (lane%16); register reduce + shuffle_xor.
                scale = scale_qk * _ld1(
                    sQscale_off, lane16
                )  # per-qhead positive score scale
                n_valid_tile = (causal_bound[0] - tok0).to(fx.Float32)
                base_tok_f = fx.Int32(warp * TOK_CHUNK + rgroup * 4).to(fx.Float32)
                thr = fx.Vector.from_elements(
                    [n_valid_tile - base_tok_f], dtype=fx.Float32
                ).broadcast_to(4)
                neg4 = fx.Vector.filled(4, -1e30, fx.Float32)
                # per_token_kv: K-scale varies per token, so fold it in BEFORE the max-reduce.
                v_scale_vecs = None
                if const_expr(per_token_kv):
                    v_scale_vecs = []
                    scaled_frags = []
                    for a in range_constexpr(NCHUNK):
                        k_scale_vec, v_scale_vec = _load_kv_scale_vecs(a, cur_kv_buf)
                        v_scale_vecs.append(v_scale_vec)
                        scaled_frags.append(frag_Ss[a] * k_scale_vec)
                else:
                    scaled_frags = frag_Ss
                # Reused in pass 2 below, halving the mask instruction count.
                masked_chunks = [
                    (_ct[a] < thr).select(scaled_frags[a], neg4)
                    for a in range_constexpr(NCHUNK)
                ]
                # pass 1: per-warp max for this qhead
                pm = fx.Float32(float("-inf"))
                for a in range_constexpr(NCHUNK):
                    pm = fx.maxnumf(pm, masked_chunks[a].reduce(ReductionOp.MAX))
                for sh in (16, 32):
                    pm = fx.maxnumf(pm, pm.shuffle_xor(sh, WAVE))
                _st_lw(
                    sLmax_off, lane16, warp, pm * scale
                )  # redundant across the 4 lanes sharing this qhead
                # per_token_kv: max V-scale for the per-tile fp8 normalization
                # (any positive value is correct, so skip the causal mask).
                if const_expr(per_token_kv):
                    pv_max = fx.Float32(0.0)
                    for a in range_constexpr(NCHUNK):
                        pv_max = fx.maxnumf(
                            pv_max, v_scale_vecs[a].reduce(ReductionOp.MAX)
                        )
                    for sh in (16, 32):
                        pv_max = fx.maxnumf(pv_max, pv_max.shuffle_xor(sh, WAVE))
                    _st_lw(sVScaleMax_off, 0, warp, pv_max)
                gpu.barrier()
                # V page-index row for next tile, now visible after the barrier.
                v_page_next = v_page_cur
                if tt1 < part_end:
                    v_page_next = _v_page_read_row()
                next_state[V_SLOT] = v_page_next
                # per_token_kv: combine warps' max V-scale into the tile's
                # normalization factor (also the PV correction below).
                v_max_scaled = None
                norm_factor_b = None
                if const_expr(per_token_kv):
                    v_max_global = _ld_lw_row(sVScaleMax_off, 0).reduce(ReductionOp.MAX)
                    v_max_scaled = v_max_global * fx.Float32(1.0 / FP8_MAX)
                    v_max_safe = v_max_scaled + fx.Float32(1e-8 / FP8_MAX)
                    norm_factor = fx.Float32(rcp_f32(v_max_safe))
                    norm_factor_b = fx.Vector.from_elements(
                        [norm_factor], dtype=fx.Float32
                    ).broadcast_to(4)
                # pass 2: global max over warps -> exp -> fp8 P pack (-> sP) -> sum
                m_new = fx.maxnumf(
                    m_prev, _ld_lw_row(sLmax_off, lane16).reduce(ReductionOp.MAX)
                )
                m_new_b = fx.Vector.from_elements(
                    [m_new], dtype=fx.Float32
                ).broadcast_to(4)
                ls = fx.Float32(0.0)
                words = []
                zero4_p = fx.Vector.filled(4, 0.0, fx.Float32)
                for a in range_constexpr(NCHUNK):
                    # re-mask Pa so a fully-masked chunk contributes exactly 0
                    valid_a = masked_chunks[a] > fx.Vector.filled(4, -1e29, fx.Float32)
                    Pa = valid_a.select(
                        fx.Vector(exp2_f32_fast(masked_chunks[a] * scale - m_new_b)),
                        zero4_p,
                    )
                    ls = ls + Pa.reduce(ReductionOp.ADD)
                    if const_expr(per_token_kv):
                        v_scale_this = (
                            _load_scale_vec(sVScale_off, a, cur_kv_buf)
                            if const_expr(head_dim == 64)
                            else v_scale_vecs[a]
                        )
                        p_scaled = Pa * v_scale_this * norm_factor_b
                    else:
                        p_scaled = Pa * fx.Vector.filled(4, FP8_MAX, fx.Float32)
                    words.append(_f32_to_fp8_words(p_scaled)[0])
                p_off0 = sP_off + lane16 * SP_ROW_BYTES + warp * TOK_CHUNK + rgroup * 4
                # NCHUNK P words scatter at stride c16//4 i32 (see phase-split).
                for a in range_constexpr(NCHUNK):
                    _lds_store(
                        p_off0 + a * (c16 // 4) * f32,
                        fx.Int32,
                        fx.Vector.from_elements([words[a]], dtype=fx.Int32),
                    )
                if const_expr(head_dim == 64):
                    fx.rocdl.sched_dswr(NCHUNK)
                for sh in (16, 32):
                    ls = ls.addf(ls.shuffle_xor(sh, WAVE), fastmath=fm_contract)
                # PV (V=A, P=B) -> output [head-dim, query-row=lane16]; same as
                # the phase-split path.
                corr_reg = fx.Float32(exp2_amdgcn_scalar(m_prev - m_new))
                if rgroup == 0:
                    _st_lw(sLsum_off, lane16, warp, ls)
                gpu.barrier()
                gsum = _ld_lw_row(sLsum_off, lane16).reduce(ReductionOp.ADD)
                l_new = fx.Float32(
                    arith.mulf(
                        arith.unwrap(l_prev),
                        arith.unwrap(corr_reg),
                        fastmath=fm_contract,
                    )
                ).addf(gsum, fastmath=fm_contract)
                p_ops = _lds_load(
                    sP_off + lane16 * SP_ROW_BYTES + rgroup * 64, fx.Int64, NVOPS
                )
                corr_b = fx.Vector.from_elements(
                    [corr_reg], dtype=fx.Float32
                ).broadcast_to(OP_ELEMS)
                # Single tile: batch both vh's V loads upfront (no sibling chain
                # to hide the latency behind).
                v_vh_batch = [
                    _v_ops(v_page_cur, vh) for vh in range_constexpr(VHE_CHUNKS)
                ]
                for vh in range_constexpr(VHE_CHUNKS):
                    v_vh = v_vh_batch[vh]
                    acc = arith.constant_vector(0.0, T.f32x4)
                    for s in range_constexpr(NVOPS):
                        acc = fx.rocdl.mfma_f32_16x16x32_fp8_fp8(
                            T.f32x4, [v_vh[s], p_ops[s], acc, 0, 0, 0]
                        )
                    op = fx.Vector(acc)
                    if const_expr(per_token_kv):
                        op = op * fx.Vector.from_elements(
                            [v_max_scaled], dtype=fx.Float32
                        ).broadcast_to(OP_ELEMS)
                    o_acc[vh] = o_acc[vh] * corr_b + op
                next_state.extend([*o_acc, m_new, l_new])
            results = yield next_state
        o_final = results

        # Direct-store epilogue: after the PV swap each lane holds its 4 head-dim
        # values for one query-row and writes them straight to global.
        inv_fp8 = fx.Float32(1.0 / FP8_MAX)
        for m in range_constexpr(M_TILES):
            row = m * MFMA_MNK + lane16  # flat (mtp, gqa) query-row for this lane
            l_row = o_final[_l_slot(m)]
            safe_l = arith.select(l_row > ZERO_F, l_row, fx.Float32(1.0))
            inv_l = fx.Float32(rcp_f32(safe_l))
            if const_expr(per_token_kv):
                o_scale = inv_l
            else:
                o_scale = fx.Float32(
                    arith.mulf(
                        arith.unwrap(inv_l),
                        arith.unwrap(v_scale_f * inv_fp8),
                        fastmath=fm_contract,
                    )
                )
            o_scale_b = fx.Vector.from_elements(
                [o_scale], dtype=fx.Float32
            ).broadcast_to(OP_ELEMS)
            qi_e = row // query_group_size
            gs_head_e = row - qi_e * query_group_size
            qh = kv_h * query_group_size + gs_head_e

            def _emit(o_norm, sub):
                if const_expr(NP == 1):
                    out_row = output_ptr[
                        seq * query_length + qi_e, qh, None  # noqa: B023
                    ]
                    out_chunk = fx.slice(
                        fx.logical_divide(out_row, fx.make_layout(OP_ELEMS, 1)),
                        (None, sub),
                    )
                    out_chunk.store(o_norm)
                else:
                    base = (
                        (seq * n_kv + kv_h) * NP + part
                    ) * TOTAL_ROWS + row  # noqa: B023
                    pout_div = fx.logical_divide(pout_ptr, fx.make_layout(OP_ELEMS, 1))
                    pout_chunk = fx.slice(
                        pout_div, (None, base * (head_dim // OP_ELEMS) + sub)
                    )
                    pout_chunk.store(o_norm)

            for vh in range_constexpr(VHE_CHUNKS):
                o_slot = _o_slot(m, vh)
                o_norm = (o_final[o_slot] * o_scale_b).to(Q_DTYPE)
                head_base = (
                    vh * (NWARP * MFMA_MNK) + warp * MFMA_MNK + rgroup * OP_ELEMS
                )
                sub = head_base // OP_ELEMS
                # Guard the partial last tile's out-of-range rows (folded away for full tiles).
                if row < TOTAL_ROWS:
                    _emit(o_norm, sub)

            if const_expr(NP > 1):  # noqa: SIM102
                if warp == 0 and rgroup == 0:
                    base = ((seq * n_kv + kv_h) * NP + part) * TOTAL_ROWS + row
                    if row < TOTAL_ROWS:
                        # Convert the running max from log2 units (scale_qk folds
                        # in LOG2E) to natural-log units: the shared reduce
                        # re-applies LOG2E itself when combining partitions.
                        pmax_ptr[base] = o_final[_m_slot(m)] * fx.Float32(1.0 / LOG2E)
                        psum_ptr[base] = l_row

    @flyc.jit
    def pa_decode_tile_launch(
        output: fx.Tensor,
        pmax: fx.Tensor,
        psum: fx.Tensor,
        pout: fx.Tensor,
        query: fx.Tensor,
        key_cache: fx.Tensor,
        value_cache: fx.Tensor,
        block_tables: fx.Tensor,
        context_lengths: fx.Tensor,
        key_scale: fx.Tensor,  # [1] per-tensor OR [num_blocks, num_kv_heads, block_size] per-token
        value_scale: fx.Tensor,  # same shape as key_scale
        max_blocks_per_seq: fx.Int32,
        num_seqs: fx.Int32,
        num_kv_heads: fx.Int32,
        stride_ks_block: fx.Int32,
        stride_ks_head: fx.Int32,
        stride_q_row: fx.Int32,
        stride_q_head: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        pa_decode_tile_kernel(
            output,
            pmax,
            psum,
            pout,
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            key_scale,
            value_scale,
            max_blocks_per_seq,
            stride_ks_block,
            stride_ks_head,
            stride_q_row,
            stride_q_head,
        ).launch(
            grid=(num_seqs, num_kv_heads, NP),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return {"launch": pa_decode_tile_launch, "kernel": pa_decode_tile_kernel}
