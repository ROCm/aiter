# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.layout_utils import crd2idx

from .stage_utils import (
    A16WI4_GROUP_SIZE,
    _a16w4_swizzle_xor16,
    _buffer_i32_scalar_read,
    _e8m0_byte_to_f32,
    _gep1,
    _gep3,
    _global_base_ptr1,
    _global_i32_at,
    _global_i32_buffer_tiles,
    _global_i32_buffer_view,
    _int4_nibble_to_bf16x8,
    _lds_ptr3,
    _raw,
    _udiv,
    _umod,
    a16wmix_use_k16,
    kmchunks_for,
    lds_acc_bytes_for,
)

# gfx950 CU count; caps the persistent gemm2 grid so high-expert launches (E896) do
# not over-launch ~max_m_blocks empty CTAs.
NUM_CU = 256
_GFX950_LDS_BYTES = 160 * 1024


# @flyc.jit is LOAD-BEARING: it AST-rewrites ``if token_id < i32_M`` into an scf.if.
# Without it the guard runs as a plain Python if (dropped at trace), so the output
# scatter fires on padded/OOB rows -- ~13x s2 regression (39us -> ~490us at E896).
@flyc.jit
def _weighted_a16_epilog(
    lds_acc_base_i32,
    accm,
    arg_out,
    arg_stids,
    arg_sweights,
    m_row,
    n_block_idx,
    wave,
    lane,
    i32_M,
    BM,
    N_OUT,
    BN,
    out_dtype,
    output_mode,
    TOPK,
):
    elem_dtype = fx.Float16 if const_expr(out_dtype == "fp16") else fx.BFloat16
    _kMChunks = kmchunks_for(BM)
    M_REPS = BM // 8
    # 4 waves split the BN(=TILE_N) tile (generic over BN, e.g. int4 tile_n=128).
    _n_per_wave = BN // 4
    num_acc_n = _n_per_wave // 16
    _s_count = BN // 64  # readback: each s-iter covers 64 cols (32 lanes x vec2)
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lds_base = _lds_ptr3(lds_acc_base_i32, fx.Int32(0))

    tx_i32 = fx.Int32(gpu.thread_id("x"))
    m_lane = tx_i32 // fx.Int32(32)
    n_lane = tx_i32 % fx.Int32(32)
    col_start = n_lane * fx.Int32(2)
    stids_base = _global_base_ptr1(arg_stids)
    sweights_base = _global_base_ptr1(arg_sweights)
    out_base = _global_base_ptr1(arg_out)

    packed = []
    weight = []
    for mr in range_constexpr(M_REPS):
        sorted_pos = m_row + fx.Int32(mr * 8) + m_lane
        packed.append(
            llvm.load(
                T.i32, _gep1(stids_base, sorted_pos * fx.Int32(4)), invariant=True
            )
        )
        weight.append(
            llvm.load(
                T.f32, _gep1(sweights_base, sorted_pos * fx.Int32(4)), invariant=True
            )
        )

    for i in range_constexpr(_kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for J in range_constexpr(num_acc_n):
            col = wave * fx.Int32(_n_per_wave) + fx.Int32(J * 16) + lane_mod_16
            vec = Vec(accm[i][J])
            for v in range_constexpr(4):
                idx = (row_base + fx.Int32(v)) * fx.Int32(BN) + col
                llvm.StoreOp(_raw(vec[v]), _gep3(lds_base, idx * fx.Int32(4)))

    gpu.barrier()

    for mr in range_constexpr(M_REPS):
        row_in_block = fx.Int32(mr * 8) + m_lane
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        if token_id < i32_M:
            if const_expr(output_mode == "reduce"):
                # Fixed-K sorting preserves the original top-k slot in the high
                # byte.  Each (token, slot) pair is unique, so this path can use
                # ordinary stores into [tokens, TOPK, N_OUT] without atomics.
                slot = (packed[mr] >> fx.Int32(24)) & fx.Int32(0xFF)
                row_base_addr = (
                    fx.Int64(token_id) * fx.Int64(TOPK) + fx.Int64(slot)
                ) * fx.Int64(N_OUT) + fx.Int64(n_block_idx * fx.Int32(BN) + col_start)
            elif const_expr(output_mode == "identity"):
                row_base_addr = fx.Int64(token_id) * fx.Int64(N_OUT) + fx.Int64(
                    n_block_idx * fx.Int32(BN) + col_start
                )
            else:
                row_base_addr = (
                    token_id * fx.Int32(N_OUT) + n_block_idx * fx.Int32(BN) + col_start
                )
            for s in range_constexpr(_s_count):
                idx0 = row_in_block * fx.Int32(BN) + col_start + fx.Int32(s * 64)
                v2 = Vec(
                    llvm.load(T.vec(2, T.f32), _gep3(lds_base, idx0 * fx.Int32(4)))
                )
                pk = Vec.from_elements(
                    [v2[0] * weight[mr], v2[1] * weight[mr]], fx.Float32
                ).to(elem_dtype)
                if const_expr(output_mode != "atomic"):
                    # ``identity`` flat routing has exactly one route per
                    # token and ``token_id == route_id``.  It therefore shares
                    # the ordinary-store epilogue with fixed-slot reduction,
                    # but addresses the final [tokens, N_OUT] output directly.
                    off = (row_base_addr + fx.Int64(s * 64)) * fx.Int64(2)
                    llvm.StoreOp(_raw(pk), _gep1(out_base, off), alignment=4)
                else:
                    off = (row_base_addr + fx.Int32(s * 64)) * fx.Int32(2)
                    out_ptr = _gep1(out_base, off)
                    llvm.AtomicRMWOp(
                        llvm.AtomicBinOp.fadd,
                        out_ptr,
                        _raw(pk),
                        llvm.AtomicOrdering.monotonic,
                        syncscope="agent",
                        alignment=4,
                    )


@flyc.jit
def _sorted_a16_epilog(
    lds_acc_base_i32,
    accm,
    arg_out,
    arg_stids,
    m_row,
    n_block_idx,
    wave,
    lane,
    i32_M,
    BM,
    N_OUT,
    BN,
    out_dtype,
):
    """Transpose MFMA accumulators and store an unweighted sorted BF16 row."""

    elem_dtype = fx.Float16 if const_expr(out_dtype == "fp16") else fx.BFloat16
    _kMChunks = kmchunks_for(BM)
    M_REPS = BM // 8
    _n_per_wave = BN // 4
    num_acc_n = _n_per_wave // 16
    _s_count = BN // 64
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lds_base = _lds_ptr3(lds_acc_base_i32, fx.Int32(0))

    tx_i32 = fx.Int32(gpu.thread_id("x"))
    m_lane = tx_i32 // fx.Int32(32)
    n_lane = tx_i32 % fx.Int32(32)
    col_start = n_lane * fx.Int32(2)
    stids_base = _global_base_ptr1(arg_stids)
    out_base = _global_base_ptr1(arg_out)

    packed = []
    for mr in range_constexpr(M_REPS):
        sorted_pos = m_row + fx.Int32(mr * 8) + m_lane
        packed.append(
            llvm.load(
                T.i32, _gep1(stids_base, sorted_pos * fx.Int32(4)), invariant=True
            )
        )

    for i in range_constexpr(_kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for J in range_constexpr(num_acc_n):
            col = wave * fx.Int32(_n_per_wave) + fx.Int32(J * 16) + lane_mod_16
            vec = Vec(accm[i][J])
            for v in range_constexpr(4):
                idx = (row_base + fx.Int32(v)) * fx.Int32(BN) + col
                llvm.StoreOp(_raw(vec[v]), _gep3(lds_base, idx * fx.Int32(4)))

    gpu.barrier()

    for mr in range_constexpr(M_REPS):
        row_in_block = fx.Int32(mr * 8) + m_lane
        sorted_pos = m_row + row_in_block
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        if token_id < i32_M:
            row_base_addr = fx.Int64(sorted_pos) * fx.Int64(N_OUT) + fx.Int64(
                n_block_idx * fx.Int32(BN) + col_start
            )
            for s in range_constexpr(_s_count):
                idx0 = row_in_block * fx.Int32(BN) + col_start + fx.Int32(s * 64)
                v2 = Vec(
                    llvm.load(T.vec(2, T.f32), _gep3(lds_base, idx0 * fx.Int32(4)))
                )
                packed_out = v2.to(elem_dtype)
                off = (row_base_addr + fx.Int64(s * 64)) * fx.Int64(2)
                llvm.StoreOp(_raw(packed_out), _gep1(out_base, off), alignment=4)


def _gemm2_body_a16w4(
    lds_raw_ptr,
    arg_a,
    arg_bq,
    arg_bscale,
    arg_bias,
    arg_eids,
    arg_stids,
    arg_sweights,
    arg_out,
    bx_i32,
    lane,
    wave,
    i32_M,
    *,
    BM,
    SORTED_BM,
    TILE_N,
    TILE_K,
    N_OUT,
    INTER,
    NE,
    b_cache_mod=2,
    w_dtype="mxfp4",
    a_dtype="bf16",
    use_k16=False,
    has_bias=False,
    round_projection_bf16=False,
    output_mode="atomic",
    TOPK=1,
    logical_dense_weight=False,
    store_sorted_projection=False,
    stages=1,
):
    """A16W4/A16W16 stage2 body. K=inter_dim, N=model_dim.

    A is the BF16/FP16 stage1 intermediate by sorted position. W2 is
    mxfp4/int4 (BF16 activation only) or matching dense BF16/FP16. Output either
    uses packed A16 atomic-fadd routing-weighted scatter to [tokens, model_dim],
    or ordinary weighted A16 stores to [tokens, TOPK, model_dim] for a following
    FP32 top-k reduction.  The backward recompute specialization consumes the
    public row-major ``[E, N_OUT, INTER]`` weight directly and stores the
    unweighted BF16 projection by sorted row.
    """
    _is_int4 = w_dtype == "int4"
    _is_dense = w_dtype in ("bf16", "fp16")
    _is_fp16 = a_dtype == "fp16"
    elem_dtype = fx.Float16 if const_expr(_is_fp16) else fx.BFloat16
    elem_ir_type = T.f16 if const_expr(_is_fp16) else T.bf16
    elem_bytes = 2
    KH_TILE_BYTES = TILE_K * elem_bytes
    LDS_STRIDE = TILE_K
    K = INTER
    K_HALF = K // 2
    K_TILES_TOTAL = K // TILE_K
    # The gfx950 path can overlap the next A direct-to-LDS copy and B register
    # loads with the current tile's MFMA cluster.  Keep the historical serial
    # loop for one-tile contractions and for gfx942's VGPR-staged A copy.
    _PIPE = stages == 2 and K_TILES_TOTAL > 1 and not use_k16
    A_LDS_STAGES = 2 if _PIPE else 1
    A_SLOT_BYTES = BM * KH_TILE_BYTES
    m_repeat = BM // 16
    k_unroll = KH_TILE_BYTES // 64
    _k0_count = TILE_K // 128
    # 4 waves split the TILE_N tile (matches the atomic-epilog wave-split).
    _n_per_wave = TILE_N // 4
    num_acc_n = _n_per_wave // 16
    k_blocks16 = KH_TILE_BYTES // 16
    _num_n_blocks = N_OUT // TILE_N

    # W2 (mxfp4) preshuffle layout (make_preshuffle_b_layout, N-major, fp4).
    bl_k0 = K_HALF // 64
    bl_stride_klane = 256
    bl_stride_k0 = 1024
    bl_stride_n0 = bl_k0 * bl_stride_k0
    layout_b = fx.make_layout(
        (N_OUT // 16, bl_k0, 4, 16, 16),
        (bl_stride_n0, bl_stride_k0, bl_stride_klane, 16, 1),
    )
    # W2 (raw bf16) preshuffle layout (N-major == shuffle_weight (16,16)), bf16-elem units:
    #   shape (N_OUT/16, K/32, 4, 16, 8). One kpack=8 bf16=one MFMA K32 fragment; K
    #   reindexed to the fp4 (klane_hw, ku)->K order (see load_b_raw_bf16).
    bfl_k0 = K // 32
    bfl_stride_klane = 128
    bfl_stride_k0 = 512
    bfl_stride_n0 = bfl_k0 * bfl_stride_k0
    layout_b_bf16 = fx.make_layout(
        (N_OUT // 16, bfl_k0, 4, 16, 8),
        (bfl_stride_n0, bfl_stride_k0, bfl_stride_klane, 8, 1),
    )
    scale_k_padded = ((K + 255) // 256) * 256
    sc_k1 = ((scale_k_padded // 32) // 4) // 2
    sc_stride_klane = 16
    sc_stride_k0 = 64
    sc_stride_n0 = sc_k1 * sc_stride_k0

    # a16wi4 groupwise scale: bf16 pairs (E, N_OUT, num_groups//2, 2), K = inter_dim.
    _num_groups = K // A16WI4_GROUP_SIZE
    _g_half = _num_groups // 2

    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)

    m_block_idx = bx_i32 // fx.Int32(_num_n_blocks)
    n_block_idx = bx_i32 % fx.Int32(_num_n_blocks)
    # The shared route layout stores one expert id per SORTED_BM rows. A
    # smaller stage-2 compute BM subdivides that metadata block.
    metadata_block_idx = m_block_idx // fx.Int32(SORTED_BM // BM)
    e = rocdl.readfirstlane(T.i32, _raw(_global_i32_at(arg_eids, metadata_block_idx)))
    m_row = m_block_idx * fx.Int32(BM)  # first sorted row of this m-block
    by_n = n_block_idx * fx.Int32(TILE_N)
    expert_off = e * fx.Int32(N_OUT)

    # Standard-layout W is contiguous per expert. Fold that expert's byte offset into
    # the 64-bit resource base so large expert sets do not overflow a whole-tensor
    # 32-bit buffer offset. (Stage 2 has no GUGU layout, so every current mode folds.)
    _w_per_expert_bytes = N_OUT * (K * 2 if _is_dense else K_HALF)
    w_base_i64 = fx.Int64(arg_bq) + fx.Int64(e) * fx.Int64(_w_per_expert_bytes)
    w_tiles = _global_i32_buffer_tiles(
        w_base_i64, min(_w_per_expert_bytes, 0xFFFFFFFF), 4
    )
    # W dwordx4 load via BufferCopy128b atom (cache modifier in the aux field).
    w_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(b_cache_mod), fx.Int32)
    w_reg_lay = fx.make_layout(4, 1)
    if _is_int4:
        _sw_bytes = NE * N_OUT * _g_half * 4
    else:
        _sw_bytes = NE * N_OUT * (scale_k_padded // 32)
    # Per-lane scalar scale gather via make_buffer_tensor 1-dword tiles + BufferCopy32b
    # scalar read (see gemm1._buffer_i32_scalar_read), replacing raw buffer_ops.
    sw_tiles = (
        None
        if _is_dense
        else _global_i32_buffer_tiles(arg_bscale, min(_sw_bytes, 0xFFFFFFFF), 1)
    )
    sw_read_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(0), fx.Int32)

    # ---- A gather (per-thread) -> LDS. A row = SORTED position m_row + row_local.
    total_threads = 256
    bytes_per_thread = (BM * TILE_K * elem_bytes) // total_threads
    x_load_bytes = 16
    num_x_loads = bytes_per_thread // x_load_bytes
    tile_k_dwords = (TILE_K * elem_bytes) // 4
    c_k_div4 = (K * elem_bytes) // 4
    tx_i32 = fx.Int32(gpu.thread_id("x"))
    chunk_i32 = x_load_bytes // 4
    tx_base = tx_i32 * fx.Int32(chunk_i32)

    x_row_local = []
    x_col_dw = []
    x_row_base_div4 = []
    for i in range_constexpr(num_x_loads):
        tile_idx = tx_base + fx.Int32(i * total_threads * chunk_i32)
        row_local = tile_idx // fx.Int32(tile_k_dwords)
        col_dw = tile_idx % fx.Int32(tile_k_dwords)
        x_row_local.append(row_local)
        x_col_dw.append(col_dw)
        sorted_row = m_row + row_local
        x_row_base_div4.append(sorted_row * fx.Int32(c_k_div4))

    x_buf = _global_i32_buffer_view(arg_a, fx.Int64(0xFFFFFFFF))
    x_dma_tiles4 = fx.logical_divide(x_buf, fx.make_layout(4, 1))
    # gfx950 (K=32): BufferCopyLDS128b direct-to-LDS async copy. gfx942 (use_k16): CDNA3
    # direct-to-LDS is 4 B/lane only (the 16 B form fails LLVM ISA lowering), so stage via
    # VGPRs like the legacy kernel: buffer_load 16 B gmem->regs then ds_write 16 B regs->
    # LDS (both b128, valid on CDNA3), preserving the same swizzled-src / linear-LDS layout.
    if const_expr(use_k16):
        x_dma_atom = fx.make_copy_atom(
            fx.rocdl.BufferCopy128b(b_cache_mod), fx.Int32
        )  # gmem->regs
        x_lds_store_atom = fx.make_copy_atom(
            fx.UniversalCopy128b(), fx.Int32
        )  # regs->LDS
    else:
        x_dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), fx.Int32)

    s_x_i32_flat = fx.make_view(
        fx.recast_iter(fx.Int32, lds_raw_ptr),
        fx.make_layout(A_LDS_STAGES * BM * LDS_STRIDE // 2, 1),
    )
    s_x_i32x4_tiles = fx.logical_divide(s_x_i32_flat, fx.make_layout(4, 1))
    a_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)

    def dma_a_tile_to_lds(base_k, slot=0):
        base_k_div4 = (base_k * fx.Int32(elem_bytes)) // fx.Int32(4)
        slot_byte = fx.Int32(slot * A_SLOT_BYTES)
        for i in range_constexpr(num_x_loads):
            col_bytes = x_col_dw[i] * fx.Int32(4)
            # A-LDS bank-conflict XOR swizzle: LDS dest stays LINEAR (buffer_load_lds
            # ignores an arbitrary swizzled per-lane dest -> NaN); swizzle the GMEM
            # source col instead, and lds_load_a applies the SAME swizzle on read.
            col_sw = _a16w4_swizzle_xor16(
                x_row_local[i], col_bytes, fx.Int32(k_blocks16), enable=True
            )
            row_k_dw = x_row_base_div4[i] + base_k_div4
            global_byte = row_k_dw * fx.Int32(4) + col_sw
            lds_byte = slot_byte + x_row_local[i] * fx.Int32(KH_TILE_BYTES) + col_bytes
            if const_expr(use_k16):
                # gfx942: buffer_load 16 B gmem->regs, then ds_write 16 B regs->LDS.
                r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                fx.copy(
                    x_dma_atom,
                    fx.slice(x_dma_tiles4, (None, global_byte // fx.Int32(16))),
                    r,
                )
                fx.copy(
                    x_lds_store_atom,
                    r,
                    fx.slice(s_x_i32x4_tiles, (None, lds_byte // fx.Int32(16))),
                )
            else:
                fx.copy(
                    x_dma_atom,
                    fx.slice(x_dma_tiles4, (None, global_byte // fx.Int32(16))),
                    fx.slice(s_x_i32x4_tiles, (None, lds_byte // fx.Int32(16))),
                )

    row_a_lds = lane_mod_16
    # The original mapping assigns each of the four 16-lane groups a 32-element
    # slice inside a 128-wide K chunk.  A dense TILE_K=64 tile instead has only
    # 16 elements per lane group; keep the A and B fragment maps in lockstep so
    # all four groups cover the two K32 instructions without reading past LDS.
    if const_expr(TILE_K < 128):
        col_base_bytes_L = lane_div_16 * fx.Int32(KH_TILE_BYTES // 4)
    else:
        col_base_bytes_L = lane_div_16 * fx.Int32(64)

    def _a_col_bytes_for_ku(ku):
        _k0_blk = ku // 4
        _ku_in = ku % 4
        return col_base_bytes_L + fx.Int32(_ku_in * 16 + _k0_blk * 256)

    def lds_load_a(mi, ku, slot=0):
        row = row_a_lds + fx.Int32(mi * 16)
        # Same XOR swizzle as the DMA write (16 B-multiple cols/mask keep alignment).
        col_swz_bytes = _a16w4_swizzle_xor16(
            row, _a_col_bytes_for_ku(ku), fx.Int32(k_blocks16), enable=True
        )
        byte_off = (
            fx.Int32(slot * A_SLOT_BYTES)
            + row * fx.Int32(KH_TILE_BYTES)
            + col_swz_bytes
        )
        r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
        fx.copy_atom_call(
            a_copy_atom, fx.slice(s_x_i32x4_tiles, (None, byte_off // fx.Int32(16))), r
        )
        return fx.Vector(fx.memref_load_vec(r)).bitcast(elem_dtype)

    def load_b_raw(base_k, n_blk, n_intra):
        raw = []
        for k0i in range_constexpr(_k0_count):
            k0 = (base_k + fx.Int32(k0i * 128)) // fx.Int32(128)
            idx_pack = fx.Int32(
                crd2idx(
                    [
                        fx.Int64(n_blk),
                        fx.Int64(k0),
                        fx.Int64(lane_div_16),
                        fx.Int64(n_intra),
                        fx.Int64(0),
                    ],
                    layout_b,
                )
            )
            # idx_pack is a fp4-byte offset; the dwordx4 tile index = (idx_pack/4 dwords)/4.
            r = fx.make_rmem_tensor(w_reg_lay, fx.Int32)
            fx.copy(w_copy_atom, fx.slice(w_tiles, (None, idx_pack // fx.Int32(16))), r)
            v4 = fx.Vector(fx.memref_load_vec(r))
            raw.append([fx.Int32(v4[j]) for j in range(4)])
        return raw

    def load_b_raw_bf16(base_k, n_blk, n_intra):
        # Raw bf16 W: one dwordx4 (8 bf16) per ku = one MFMA K32 fragment. Index to
        # match the fp4 (klane_hw=lane_div_16, ku)->K order (see gemm1 counterpart).
        raw = []
        base_k0 = base_k // fx.Int32(32)
        for ku in range_constexpr(k_unroll):
            if const_expr(TILE_K < 128):
                # Flatten the TILE_K/8 fragments across four lane groups.  For
                # TILE_K=64 this maps (lane_group, ku) to logical fragments
                # 0..7 and therefore to exactly two K32 preshuffle blocks.
                frag = lane_div_16 * fx.Int32(k_unroll) + fx.Int32(ku)
                bf_k0 = base_k0 + frag // fx.Int32(4)
                bf_klane = frag % fx.Int32(4)
            else:
                _k0_blk = ku // 4
                bf_k0 = base_k0 + fx.Int32(_k0_blk * 4) + lane_div_16
                bf_klane = fx.Int32(ku % 4)
            if const_expr(logical_dense_weight):
                # Public backward W2 is contiguous [E, N_OUT, INTER].  The
                # preshuffle coordinates map to eight naturally contiguous K
                # values in one logical output row.
                logical_n = n_blk * fx.Int32(16) + n_intra
                logical_k = bf_k0 * fx.Int32(32) + bf_klane * fx.Int32(8)
                elem_idx = logical_n * fx.Int32(K) + logical_k
            else:
                elem_idx = fx.Int32(
                    crd2idx(
                        [
                            fx.Int64(n_blk),
                            fx.Int64(bf_k0),
                            fx.Int64(bf_klane),
                            fx.Int64(n_intra),
                            fx.Int64(0),
                        ],
                        layout_b_bf16,
                    )
                )
            # elem_idx is a bf16-elem offset; dword index = elem_idx*2/4, tile index = /4.
            r = fx.make_rmem_tensor(w_reg_lay, fx.Int32)
            fx.copy(w_copy_atom, fx.slice(w_tiles, (None, elem_idx // fx.Int32(8))), r)
            raw.append(fx.Vector(fx.memref_load_vec(r)).bitcast(elem_dtype))
        return raw

    def load_b_scale(base_k, mni, n_pack):
        # Per-lane scalar e8m0 gather (dict-cached across ku).
        scales = []
        cache = {}
        for ku in range_constexpr(k_unroll):
            _k0_blk = ku // 4
            adj_ku = base_k // fx.Int32(32) + fx.Int32(_k0_blk * 4) + lane_div_16
            k_pack_sub = (adj_ku // fx.Int32(4)) % fx.Int32(2)
            s_ku = adj_ku // fx.Int32(8)
            if _k0_blk not in cache:
                idx = (
                    mni * fx.Int32(sc_stride_n0)
                    + s_ku * fx.Int32(sc_stride_k0)
                    + lane_div_16 * fx.Int32(sc_stride_klane)
                    + lane_mod_16
                )
                cache[_k0_blk] = _buffer_i32_scalar_read(sw_tiles, idx, sw_read_atom)
            packed = cache[_k0_blk]
            byte_even = k_pack_sub * fx.Int32(2)
            byte_odd = byte_even + fx.Int32(1)
            se = _e8m0_byte_to_f32(packed, byte_even)
            so = _e8m0_byte_to_f32(packed, byte_odd)
            scales.append((n_pack == fx.Int32(0)).select(se, so))
        return scales

    def load_b_scale_int4(base_k, col_g):
        # int4 groupwise (bf16-pair) scale, per-lane N = col_g. See gemm1 counterpart.
        scales = []
        base_dword = col_g * fx.Int32(_g_half)
        for ku in range_constexpr(k_unroll):
            _k0_blk = ku // 4
            adj_ku = base_k // fx.Int32(32) + fx.Int32(_k0_blk * 4) + lane_div_16
            pair_idx = adj_ku // fx.Int32(2)
            packed = _buffer_i32_scalar_read(
                sw_tiles, base_dword + pair_idx, sw_read_atom
            )
            lo = fx.Float32(_raw(packed << fx.Int32(16)).bitcast(T.f32))
            hi = fx.Float32(_raw(packed & fx.Int32(0xFFFF0000)).bitcast(T.f32))
            scales.append((adj_ku % fx.Int32(2) == fx.Int32(0)).select(lo, hi))
        return scales

    vec2_bf16 = ir.Type.parse("vector<2xbf16>")

    def upconvert_b(raw, ku, scale_f32):
        if const_expr(_is_dense):
            return raw[ku]  # already the matching A16 MMA operand
        i32_val = _raw(raw[ku // 4][ku % 4])
        if const_expr(_is_int4):
            return _int4_nibble_to_bf16x8(fx.Int32(i32_val), scale_f32, use_k16=use_k16)
        s_raw = _raw(scale_f32)
        i32s = []
        for sel in range_constexpr(4):
            pp = rocdl.cvt_scalef32_pk_bf16_fp4(vec2_bf16, i32_val, s_raw, sel)
            i32s.append(fx.Int32(fx.Vector(pp).bitcast(fx.Int32)[0]))
        v4i32 = fx.Vector.from_elements([_raw(x) for x in i32s], fx.Int32)
        return v4i32.bitcast(fx.BFloat16)

    # ---- N-column addressing (W2 cols of model_dim; wave owns _n_per_wave) ------
    n_tile_base = wave * fx.Int32(_n_per_wave)
    col_g_list = []
    n_blk_list, n_intra_list, scale_mni_list, scale_np_list = [], [], [], []
    for ni in range_constexpr(num_acc_n):
        col_g = by_n + n_tile_base + fx.Int32(ni * 16) + lane_mod_16
        col_g_list.append(col_g)
        # The resource base already selects the expert (see w_tiles).
        row_w = col_g
        n_blk_list.append(row_w // fx.Int32(16))
        n_intra_list.append(row_w % fx.Int32(16))
        ng = expert_off + by_n + n_tile_base + fx.Int32(ni * 16)
        scale_mni_list.append(ng // fx.Int32(32))
        scale_np_list.append((ng // fx.Int32(16)) % fx.Int32(2))
    # int4 groupwise scale: per-lane N = expert_off + col_g (row_w already computed).
    if const_expr(_is_int4):
        scale_n_list = [
            expert_off + col_g_list[ni] for ni in range_constexpr(num_acc_n)
        ]

    # ---- accumulators: accm[mi][ni] f32[4] (layout the atomic epilog expects) --
    acc_layout = fx.make_layout(4, 1)
    accm = [
        [fx.make_rmem_tensor(acc_layout, fx.Float32) for _ in range(num_acc_n)]
        for _ in range(m_repeat)
    ]
    zero4 = Vec.filled(4, 0.0, fx.Float32)
    for mi in range_constexpr(m_repeat):
        for ni in range_constexpr(num_acc_n):
            accm[mi][ni].store(zero4)

    # Arch-gate: gfx950 K=32 (one MFMA/K-step); gfx942 (use_k16) splits each v8bf16 into
    # two v4bf16 halves -> TWO 16x16x16 MFMAs into the same acc (no 16x16x32 on gfx942).
    if const_expr(use_k16):
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, elem_dtype))
    else:
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, elem_dtype))

    def _a16_frag(v8):
        t = fx.make_rmem_tensor(fx.make_layout(8, 1), elem_dtype)
        t.store(v8)
        return t

    def _a16_frag4(v8, half):
        t = fx.make_rmem_tensor(fx.make_layout(4, 1), elem_dtype)
        t.store(
            fx.Vector.from_elements(
                [_raw(v8[half * 4 + j]) for j in range_constexpr(4)], elem_dtype
            )
        )
        return t

    def _mma(acc, a8, b8):
        if const_expr(use_k16):
            for h in range_constexpr(2):
                fx.gemm(mma_atom, acc, _a16_frag4(a8, h), _a16_frag4(b8, h), acc)
        else:
            fx.gemm(mma_atom, acc, _a16_frag(a8), _a16_frag(b8), acc)

    def load_b_subtile(base_k, ni):
        if const_expr(_is_dense):
            return load_b_raw_bf16(base_k, n_blk_list[ni], n_intra_list[ni]), None
        if const_expr(_is_int4):
            b_sc = load_b_scale_int4(base_k, scale_n_list[ni])
        else:
            b_sc = load_b_scale(base_k, scale_mni_list[ni], scale_np_list[ni])
        return load_b_raw(base_k, n_blk_list[ni], n_intra_list[ni]), b_sc

    def load_b_tile(base_k):
        return [load_b_subtile(base_k, ni) for ni in range_constexpr(num_acc_n)]

    def preload_a(read_slot):
        # Retire all current-slot LDS reads before issuing the next direct-to-LDS
        # copy.  Besides making ping/pong reuse safe, each A fragment is then
        # shared by every N subtile instead of being reloaded from LDS.
        return [
            [lds_load_a(mi, ku, slot=read_slot) for ku in range_constexpr(k_unroll)]
            for mi in range_constexpr(m_repeat)
        ]

    def compute_b_subtile(b_subtile, a_frags, ni):
        b_raw, b_sc = b_subtile
        for ku in range_constexpr(k_unroll):
            _bsc = None if const_expr(_is_dense) else b_sc[ku]
            bb = upconvert_b(b_raw, ku, _bsc)
            for mi in range_constexpr(m_repeat):
                a8 = a_frags[mi][ku]
                _mma(accm[mi][ni], a8, bb)

    def compute_tile(b_tile, a_frags):
        for ni in range_constexpr(num_acc_n):
            b_raw, b_sc = b_tile[ni]
            for ku in range_constexpr(k_unroll):
                _bsc = None if const_expr(_is_dense) else b_sc[ku]
                bb = upconvert_b(b_raw, ku, _bsc)
                for mi in range_constexpr(m_repeat):
                    _mma(accm[mi][ni], a_frags[mi][ku], bb)

    if const_expr(not _PIPE):
        # Preserve the established serial path byte-for-byte for K_tiles==1 and
        # for callers that leave ``stages`` at its backwards-compatible default.
        for kt in range_constexpr(K_TILES_TOTAL):
            base_k = fx.Int32(kt * TILE_K)
            dma_a_tile_to_lds(base_k, slot=0)
            if const_expr(_is_dense):
                b_raw = [
                    load_b_raw_bf16(base_k, n_blk_list[ni], n_intra_list[ni])
                    for ni in range_constexpr(num_acc_n)
                ]
                b_sc = None
            else:
                b_raw = [
                    load_b_raw(base_k, n_blk_list[ni], n_intra_list[ni])
                    for ni in range_constexpr(num_acc_n)
                ]
                if const_expr(_is_int4):
                    b_sc = [
                        load_b_scale_int4(base_k, scale_n_list[ni])
                        for ni in range_constexpr(num_acc_n)
                    ]
                else:
                    b_sc = [
                        load_b_scale(base_k, scale_mni_list[ni], scale_np_list[ni])
                        for ni in range_constexpr(num_acc_n)
                    ]
            gpu.barrier()
            for ni in range_constexpr(num_acc_n):
                for ku in range_constexpr(k_unroll):
                    _bsc = None if const_expr(_is_dense) else b_sc[ni][ku]
                    bb = upconvert_b(b_raw[ni], ku, _bsc)
                    for mi in range_constexpr(m_repeat):
                        a8 = lds_load_a(mi, ku, slot=0)
                        _mma(accm[mi][ni], a8, bb)
            gpu.barrier()
    else:
        # Two-stage gfx950 software pipeline.  Tile 0 is primed before the
        # loop.  Each iteration snapshots the current A slot into registers,
        # starts the next A DMA into the other slot, then rotates next-tile B
        # fragments into registers released by the current N subtile.
        dma_a_tile_to_lds(fx.Int32(0), slot=0)
        b_cur = load_b_tile(fx.Int32(0))
        for kt in range_constexpr(K_TILES_TOTAL):
            cur_slot = kt % A_LDS_STAGES
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()
            a_frags = preload_a(cur_slot)
            if const_expr(kt + 1 < K_TILES_TOTAL):
                next_k = fx.Int32((kt + 1) * TILE_K)
                dma_a_tile_to_lds(next_k, slot=(kt + 1) % A_LDS_STAGES)
                if const_expr(num_acc_n == 1):
                    b_nxt = load_b_tile(next_k)
                    compute_tile(b_cur, a_frags)
                    b_cur = b_nxt
                else:
                    for ni in range_constexpr(num_acc_n):
                        compute_b_subtile(b_cur[ni], a_frags, ni)
                        b_cur[ni] = load_b_subtile(next_k, ni)
            else:
                compute_tile(b_cur, a_frags)

    # ---- epilogue: sorted projection, routing-weighted atomic scatter, or
    # fixed-slot store.  All modes use an LDS transpose for coalesced writes.
    if const_expr(has_bias):
        bias_base = _global_base_ptr1(
            fx.Int64(arg_bias) + fx.Int64(e) * fx.Int64(N_OUT * 2)
        )
        bias_values = []
        for J in range_constexpr(num_acc_n):
            col = wave * fx.Int32(_n_per_wave) + fx.Int32(J * 16) + lane_mod_16
            bias_idx = by_n + col
            bias_raw = llvm.load(
                elem_ir_type,
                _gep1(bias_base, bias_idx * fx.Int32(2)),
                invariant=True,
            )
            bias_values.append(fx.Float32(elem_dtype(bias_raw)))

    accm_v = []
    for i in range_constexpr(m_repeat):
        row = []
        for J in range_constexpr(num_acc_n):
            vec = Vec(accm[i][J].load().ir_value())
            if const_expr(has_bias):
                vec = Vec.from_elements(
                    [vec[v] + bias_values[J] for v in range_constexpr(4)],
                    fx.Float32,
                )
            if const_expr(round_projection_bf16 and not store_sorted_projection):
                # Legacy Sonic stores grouped-GEMM output in BF16 before the
                # FP32 route-score reduction reloads it.
                vec = vec.to(elem_dtype).to(fx.Float32)
            row.append(vec.ir_value())
        accm_v.append(row)
    # Both epilogues transpose accumulator ownership through LDS for coalesced
    # vec2 stores.  Sorted mode skips route weights and atomics after the same
    # proven MFMA-fragment mapping.
    gpu.barrier()
    lds_acc_base_i32 = fx.Int32(fx.ptrtoint(lds_raw_ptr))
    if const_expr(store_sorted_projection):
        _sorted_a16_epilog(
            lds_acc_base_i32,
            accm_v,
            arg_out,
            arg_stids,
            m_row,
            n_block_idx,
            wave,
            lane,
            i32_M,
            BM,
            N_OUT,
            TILE_N,
            a_dtype,
        )
        # Expert-grid may immediately reuse this LDS region for its next M
        # subtile.  Ensure every lane finished the transpose read first.
        gpu.barrier()
    else:
        _weighted_a16_epilog(
            lds_acc_base_i32,
            accm_v,
            arg_out,
            arg_stids,
            arg_sweights,
            m_row,
            n_block_idx,
            wave,
            lane,
            i32_M,
            BM,
            N_OUT,
            TILE_N,
            a_dtype,
            output_mode,
            TOPK,
        )


def gemm2_a16w4_grid(BM, *, N_OUT, TILE_N, max_m_blocks, persist=False):
    """Flattened launch grid for a16w4 gemm2.

    Non-persistent (default): one CTA per (m-block x n-block) tile over padded
    ``max_m_blocks``. Persistent: cap to ``min(total_work, NUM_CU)`` CTAs (only when
    padded work > ``NUM_CU*4``); each CTA loops over its real work-tiles.
    """
    total_work = int(max_m_blocks) * (N_OUT // TILE_N)
    if persist and total_work > NUM_CU * 4:
        return min(total_work, NUM_CU)
    return total_work


def compile_gemm2_a16w4_port(
    BM=32,
    *,
    SORTED_BM=None,
    NE,
    N_OUT,
    D_INTER,
    TILE_N=256,
    TILE_K=256,
    xcd_swizzle=1,
    b_cache_mod=2,
    waves_per_eu=None,
    w_dtype="mxfp4",
    a_dtype="bf16",
    persist=False,
    has_bias=False,
    round_projection_bf16=False,
    output_mode="atomic",
    TOPK=1,
    logical_dense_weight=False,
    store_sorted_projection=False,
    expert_grid=False,
    stages=1,
):
    """A16W4/A16W16 grouped down-projection builder.

    N_OUT = model_dim (down-proj output). D_INTER = inter_dim (contraction).
    ``output_mode='atomic'`` uses packed A16 routing-weighted atomic scatter;
    ``output_mode='reduce'`` writes one weighted A16 row per original fixed-K
    route for a subsequent FP32 top-k reduction; and ``output_mode='identity'``
    directly stores one weighted row per token for the flat K=1 identity-route
    contract. ``TOPK`` is compile-time.

    ``SORTED_BM`` is the route-sort padding and expert-metadata granularity. It
    defaults to ``BM`` and may be a multiple when this GEMM subdivides a route
    tile shared with stage 1.

    ``xcd_swizzle`` (>0) bijectively round-robins the launch index across the 8 XCDs to
    balance per-XCD/HBM traffic (gemm2 is HBM-bound), + optional M-group swizzle for
    per-XCD L2 locality (group = xcd_swizzle m-blocks).
    """
    SORTED_BM = BM if SORTED_BM is None else SORTED_BM
    assert w_dtype in ("mxfp4", "int4", "bf16", "fp16"), (
        "w_dtype must be 'mxfp4', 'int4', 'bf16' or 'fp16', " f"got {w_dtype!r}"
    )
    assert a_dtype in (
        "bf16",
        "fp16",
    ), f"a_dtype must be 'bf16' or 'fp16', got {a_dtype!r}"
    if w_dtype in ("bf16", "fp16"):
        assert w_dtype == a_dtype, "dense w_dtype must match a_dtype"
    else:
        assert a_dtype == "bf16", "quantized weights currently require a_dtype='bf16'"
    assert isinstance(has_bias, bool), "has_bias must be bool"
    assert isinstance(round_projection_bf16, bool), "round_projection_bf16 must be bool"
    assert isinstance(logical_dense_weight, bool), "logical_dense_weight must be bool"
    assert isinstance(
        store_sorted_projection, bool
    ), "store_sorted_projection must be bool"
    assert isinstance(expert_grid, bool), "expert_grid must be bool"
    assert (
        isinstance(stages, int) and not isinstance(stages, bool) and stages in (1, 2)
    ), f"stages must be the integer 1 or 2, got {stages!r}"
    assert output_mode in (
        "atomic",
        "reduce",
        "identity",
    ), "output_mode must be 'atomic', 'reduce', or 'identity'"
    assert (
        isinstance(TOPK, int) and 0 < TOPK <= 255
    ), "TOPK must be an integer in [1, 255]"
    assert not logical_dense_weight or w_dtype in (
        "bf16",
        "fp16",
    ), "logical_dense_weight is valid only for dense A16 weights"
    assert (
        not store_sorted_projection or logical_dense_weight
    ), "store_sorted_projection requires logical dense weights"
    assert (
        not expert_grid or store_sorted_projection
    ), "expert_grid requires the backward sorted-projection mode"
    assert (
        not expert_grid or not persist
    ), "expert_grid and persistent route-grid scheduling are mutually exclusive"
    # Arch-gate K=16 (gfx942) vs K=32 (gfx950); see a16wmix_use_k16.
    _use_k16 = a16wmix_use_k16()
    _K = D_INTER
    _effective_stages = 2 if stages == 2 and (_K // TILE_K) > 1 and not _use_k16 else 1
    assert _K % TILE_K == 0, f"D_INTER (K) must be a multiple of {TILE_K}, got {_K}"
    assert (
        N_OUT % TILE_N == 0
    ), f"model_dim (N_OUT) must be a multiple of {TILE_N}, got {N_OUT}"
    assert BM % 16 == 0, f"BM must be a multiple of 16, got {BM}"
    assert TILE_N % 64 == 0, f"TILE_N must be a multiple of 64, got {TILE_N}"
    assert (BM * TILE_K) % 2048 == 0, (
        "BM * TILE_K must be a multiple of 2048 A16 elements for the "
        f"256-thread direct-to-LDS copy, got BM={BM}, TILE_K={TILE_K}"
    )
    assert (
        SORTED_BM % BM == 0
    ), f"SORTED_BM ({SORTED_BM}) must be a multiple of BM ({BM})"
    _num_n_blocks = N_OUT // TILE_N
    KH_TILE_BYTES = TILE_K * 2

    # A and epilogue accumulator staging use LDS in disjoint phases, so reserve
    # their maximum rather than their sum.
    _a_bytes = _effective_stages * BM * KH_TILE_BYTES
    _acc_bytes = lds_acc_bytes_for(BM, TILE_N)
    _lds_bytes = max(_a_bytes, _acc_bytes)
    assert _lds_bytes <= _GFX950_LDS_BYTES, (
        f"gemm2 needs {_lds_bytes} LDS bytes, exceeding gfx950's "
        f"{_GFX950_LDS_BYTES} bytes (BM={BM}, TILE_N={TILE_N}, TILE_K={TILE_K}, "
        f"stages={_effective_stages})"
    )

    _wd_tag = "" if w_dtype == "mxfp4" else f"_{w_dtype}"
    _ad_tag = "" if a_dtype == "bf16" else f"_a{a_dtype}"
    _output_tag = (
        "atomic"
        if output_mode == "atomic"
        else (f"reduce_tk{TOPK}" if output_mode == "reduce" else "identity")
    )
    _sorted_tag = f"_sbm{SORTED_BM}" if SORTED_BM != BM else ""
    _name = (
        f"gemm2_a16w4{_wd_tag}{_ad_tag}_port_ne{NE}_h{N_OUT}_i{_K}_bm{BM}_tn{TILE_N}"
        f"_tk{TILE_K}{_sorted_tag}_{_output_tag}"
    )
    if b_cache_mod != 2:
        _name += f"_bcm{b_cache_mod}"
    if xcd_swizzle > 0:
        _name += f"_xcd{xcd_swizzle}"
    if waves_per_eu:
        _name += f"_w{waves_per_eu}"
    if persist:
        _name += "_persist"
    if has_bias:
        _name += "_biasv1"
    if round_projection_bf16:
        _name += "_projbf16v1"
    if logical_dense_weight:
        _name += "_logicalw"
    if store_sorted_projection:
        _name += "_storesorted"
    if expert_grid:
        _name += "_egrid"
    if _effective_stages > 1:
        _name += f"_s{_effective_stages}"

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, _lds_bytes, 16]

    @flyc.kernel(name=_name, known_block_size=[256, 1, 1])
    def gemm2_kernel(
        arg_a: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_bias: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        arg_out: fx.Int64,
    ):
        lds_raw_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        tx_i32 = fx.Int32(gpu.thread_id("x"))
        bx_i32 = fx.Int32(gpu.block_id("x"))
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))
        cumsum0 = _global_i32_at(arg_cumsum, fx.Int32(0))
        total_m_blocks = cumsum0 // fx.Int32(BM)
        bound = total_m_blocks * fx.Int32(_num_n_blocks)

        # Bijective XCD round-robin over valid tiles [0, bound) to balance per-XCD/HBM
        # traffic; xcd_swizzle>0 also M-group-swizzles for per-XCD L2 locality.
        _NXCD = 8
        _xq = _udiv(bound, _NXCD)
        _xr = _umod(bound, _NXCD)
        _SW = xcd_swizzle

        def _xcd_np(pid):
            xc = _umod(pid, _NXCD)
            wgid = (
                xc * _xq
                + fx.Int32(arith.minsi(_raw(xc), _raw(_xr)))
                + _udiv(pid, _NXCD)
            )
            if const_expr(_SW <= 0):
                return wgid
            _ng = fx.Int32(_SW * _num_n_blocks)
            group_id = wgid // _ng
            first_pid_m = group_id * fx.Int32(_SW)
            remaining_m = total_m_blocks - first_pid_m
            group_size_m = fx.Int32(arith.minsi(_raw(remaining_m), _raw(fx.Int32(_SW))))
            wig = wgid % _ng
            m_block = first_pid_m + (wig % group_size_m)
            n_block = wig // group_size_m
            return m_block * fx.Int32(_num_n_blocks) + n_block

        def _run_tile(tile):
            _gemm2_body_a16w4(
                lds_raw_ptr,
                arg_a,
                arg_bq,
                arg_bscale,
                arg_bias,
                arg_eids,
                arg_stids,
                arg_sweights,
                arg_out,
                tile,
                lane,
                wave,
                i32_M,
                BM=BM,
                SORTED_BM=SORTED_BM,
                TILE_N=TILE_N,
                TILE_K=TILE_K,
                N_OUT=N_OUT,
                INTER=_K,
                NE=NE,
                b_cache_mod=b_cache_mod,
                w_dtype=w_dtype,
                a_dtype=a_dtype,
                use_k16=_use_k16,
                has_bias=has_bias,
                round_projection_bf16=round_projection_bf16,
                output_mode=output_mode,
                TOPK=TOPK,
                logical_dense_weight=logical_dense_weight,
                store_sorted_projection=store_sorted_projection,
                stages=_effective_stages,
            )

        if const_expr(expert_grid):
            # Backward-only device-driven schedule.  One workgroup owns an
            # (expert, N-tile) pair and loops over ceil(real_rows/BM) tiles.
            # A fixed-trip lower_bound finds the expert's first SORTED_BM block
            # without copying prefix sums back to the host.
            expert_bound = fx.Int32(NE * _num_n_blocks)
            if bx_i32 < expert_bound:
                expert = bx_i32 // fx.Int32(_num_n_blocks)
                n_block = bx_i32 % fx.Int32(_num_n_blocks)
                frequency = rocdl.readfirstlane(
                    T.i32, _raw(_global_i32_at(arg_bscale, expert))
                )
                if frequency > fx.Int32(0):
                    lo = fx.Int32(0)
                    hi = cumsum0 // fx.Int32(SORTED_BM)
                    for _ in range_constexpr(25):
                        searching = lo < hi
                        mid = (lo + hi) // fx.Int32(2)
                        safe_mid = searching.select(mid, fx.Int32(0))
                        mid_expert = fx.Int32(_global_i32_at(arg_eids, safe_mid))
                        move_right = searching & (mid_expert < expert)
                        lo = move_right.select(mid + fx.Int32(1), lo)
                        move_left = searching & (mid_expert >= expert)
                        hi = move_left.select(mid, hi)
                    first_m_block = lo * fx.Int32(SORTED_BM // BM)
                    num_m_blocks = (frequency + fx.Int32(BM - 1)) // fx.Int32(BM)
                    for subtile in range(0, num_m_blocks, 1):
                        tile = (first_m_block + fx.Int32(subtile)) * fx.Int32(
                            _num_n_blocks
                        ) + n_block
                        _run_tile(tile)
        elif const_expr(persist):
            # Persistent CU-limited grid (~NUM_CU CTAs): each CTA does tile bx_i32 then
            # strides by grid size over [0, bound); _xcd_np maps every visited index, so
            # each tile runs once (same mapping as non-persistent). Loop-top barrier
            # separates the prev tile's epilog LDS from the next tile's A-DMA.
            grid_nb = fx.Int32(gpu.grid_dim.x)
            if bx_i32 < bound:
                _run_tile(_xcd_np(bx_i32))
            for iv in range(bx_i32 + grid_nb, bound, gpu.grid_dim.x):
                gpu.barrier()
                _run_tile(_xcd_np(fx.Int32(iv)))
        else:
            if bx_i32 < bound:
                _run_tile(_xcd_np(bx_i32))

    @flyc.jit
    def launch_gemm2(
        arg_a: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_bias: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        i32_grid: fx.Int32,
        arg_out: fx.Int64,
        stream: fx.Stream,
    ):
        grid_x = fx.Int64(i32_grid)
        gemm2_kernel(
            arg_a,
            arg_bq,
            arg_bscale,
            arg_bias,
            arg_eids,
            arg_cumsum,
            arg_stids,
            arg_sweights,
            i32_M,
            i32_max_m_blocks,
            arg_out,
            value_attrs={"rocdl.waves_per_eu": waves_per_eu} if waves_per_eu else None,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm2
