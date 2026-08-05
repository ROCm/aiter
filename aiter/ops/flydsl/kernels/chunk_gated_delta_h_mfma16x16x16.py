# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Gated Delta Net K5 hidden-state recurrence kernel using the @flyc.kernel API.

mfma16 / HIP-aligned fork (formerly the "vk" fork): the compute path uses the
16x16x16 bf16 MFMA (``mfma_f32_16x16x16bf16_1k``) -- the SAME instruction as the
hand-tuned HIP/C++ K5 kernel -- and the SAME warp partition (BT split-M, K split
across waves, V not split across warps). This fork is NON-VWARP / split-M ONLY
(the alternative OPT-VWARP layout has been removed). It writes the public VK
layout [..., V, K] via a [V][K] transpose buffer + b128 store (HIP-aligned).

For each chunk t (serial over NT chunks):
  1. Store h snapshot for downstream K6
  2. v_new = u - w @ h   (delta correction via MFMA)
  3. Gated decay + state update:
       v_new *= exp(g_last - g_cumsum)
       h = h * exp(g_last) + k^T @ v_new
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .tensor_shim import GTensor, _to_raw

_LOG2E = math.log2(math.e)  # 1.4426950408889634


def _make_fast_exp(g_is_log2_scaled: bool):
    """``exp(x)`` lowered via ``exp2``. If ``g`` is already log2(e)-prescaled the
    ``* LOG2E`` is dropped (single ``v_exp_f32``); otherwise it is folded in."""
    if g_is_log2_scaled:
        return lambda x: rocdl.exp2(T.f32, x)
    return lambda x: rocdl.exp2(T.f32, x * _LOG2E)


def _mfma_bf16_16x16x16(a_bf16x4, b_bf16x4, acc_f32x4):
    """Single mfma_f32_16x16x16_bf16 instruction (gfx950 / CDNA bf16 1k form).

    K-tile is 16 (half of the 16x16x32 variant), so A and B are bf16x4
    (one ds_read_tr16_b64 worth) instead of bf16x8. Output C is f32x4,
    identical layout to the 16x16x32 form.

    The ``rocdl.mfma.f32.16x16x16bf16.1k`` op takes its A/B operands as
    ``vector<4xi16>`` (bf16 bit-pattern as signless i16), so bitcast the
    bf16x4 fragments before handing them over.
    """
    a_i16x4 = a_bf16x4.bitcast(fx.Int16)
    b_i16x4 = b_bf16x4.bitcast(fx.Int16)
    raw = rocdl.mfma_f32_16x16x16bf16_1k(
        T.f32x4, [a_i16x4, b_i16x4, acc_f32x4, 0, 0, 0]
    )
    # Wrap the intrinsic result so accumulators stay on the fx.Vector method
    # API (arithmetic operators, .to(), .bitcast()) all the way to the epilogue.
    return fx.Vector(raw, (4,), fx.Float32)


def _f32x4_to_bf16x4_rne(vec_f32x4):
    """Round-to-nearest-even f32x4 -> bf16x4. LLVM lowers this to the native
    ``v_cvt_pk_bf16_f32`` on gfx950 and to a software RNE sequence (bias +
    NaN->qNaN fix-up) on gfx942, both matching torch/HIP
    ``__float2bfloat16_rn`` (verified bit-exact on gfx942)."""
    return vec_f32x4.to(fx.BFloat16)


def _f32x4_to_bf16x4_trunc(vec_f32x4):
    """Truncating f32x4 -> bf16x4 (keep high 16 bits, no rounding bias). Bit-
    identical to HIP ``float_to_bf16`` (``bit_cast<u32>(x) >> 16``); arch-neutral.
    """
    hi = vec_f32x4.bitcast(fx.Uint32) >> fx.Vector.filled(4, 16, fx.Uint32)
    return hi.to(fx.Uint16).bitcast(fx.BFloat16)


# Default fp32->bf16 output-conversion mode. When True, use bit-truncation to
# match HIP's ``float_to_bf16`` (``bit_cast<u32>(x) >> 16``) so flydsl-hip
# outputs are bit-identical to the HIP/C++ K5 kernel. When False, use RNE
# (~0.5 ulp closer to the FP32 reference, but NOT bit-matching HIP). This is
# only the default; the actual mode is a compile option (``BF16_CONVERT_TRUNC``)
# threaded through ``compile_chunk_gated_delta_h_mfma16_hip``.
_BF16_CONVERT_TRUNC_DEFAULT = True


def _make_bf16_converter(trunc: bool):
    """Return the fp32x4 -> bf16x4 output converter for this compile:
    bit-truncation to match HIP ``float_to_bf16`` (``trunc=True``, default), or
    RNE (arch-optimal: native cvt on gfx950, software RNE on gfx942)."""
    return _f32x4_to_bf16x4_trunc if trunc else _f32x4_to_bf16x4_rne


# -- Compile the kernel ---------------------------------------------------


def compile_chunk_gated_delta_h_mfma16_hip(
    *,
    K: int,
    V: int,
    BT: int = 64,
    BV: int = 32,
    H: int,
    Hg: int,
    USE_G: bool = True,
    USE_GK: bool = False,
    USE_INITIAL_STATE: bool = True,
    STORE_FINAL_STATE: bool = True,
    SAVE_NEW_VALUE: bool = True,
    IS_VARLEN: bool = True,
    WU_CONTIGUOUS: bool = True,
    STATE_DTYPE_BF16: bool = False,
    G_IS_LOG2_SCALED: bool = False,
    USE_STATE_INDICES: bool = False,
    SCHED_GFX942: bool = False,
    G_HEAD_MAJOR: bool = False,
    BF16_CONVERT_TRUNC: bool = _BF16_CONVERT_TRUNC_DEFAULT,
):
    """Compile the GDN K5 kernel.

    Returns a @flyc.jit function:
        launch_fn(k, v, w, v_new, g, gk, h, h0, ht,
                  cu_seqlens, chunk_offsets,
                  T_val, T_flat, N_val, stream)

    When ``STATE_DTYPE_BF16=False`` (default) the SSM state tensors ``h0`` /
    ``ht`` are ``float32``. When ``STATE_DTYPE_BF16=True`` they are
    ``bfloat16``: ``h0`` is ``extf``-promoted to f32 right after each load,
    and ``ht`` is ``truncf``-demoted to bf16 right before each store. The
    f32 accumulator (``h_accs``) and all intermediate LDS layouts are
    unchanged, so this only affects HBM bandwidth / footprint of the SSM
    state. Mirrors the pattern used by ``kernels/gdr_decode.py``.
    """
    # This kernel's wave mapping (wid*16, 4 waves cover 64 rows), the
    # cooperative-load batching, and BT_STEPS all hardcode BT=64; gated_v
    # alias-reuses h_state panel1 (needs NUM_K_BLOCKS>=2; for K=64 panel1 does
    # not exist at all -> out-of-bounds store), and the LDS layout is only
    # validated for K=V=128. Other BT/K would trigger LDS aliasing OOB,
    # out-of-bounds stores, or excessive LDS usage, so reject them explicitly
    # here instead of silently producing wrong results.
    assert BT == 64, f"chunk_gated_delta_h_mfma16_hip only supports BT=64, got BT={BT}"
    assert K == 128, f"chunk_gated_delta_h_mfma16_hip only supports K=128, got K={K}"
    assert BV % 16 == 0
    NUM_K_BLOCKS = K // 64

    # Tensor slots use the ``fx.Pointer`` ABI (raw data pointer). The kernel
    # body wraps every slot as ``GTensor(..., shape=(-1,))`` and never reads
    # the FlyDSL-injected memref shape/stride, so passing a bare pointer
    # produces identical device code while skipping the per-launch DLPack
    # export + layout-buffer packing that the default layout-dynamic
    # ``fx.Tensor`` memref incurs under flydsl >=0.2.0. The host side wraps
    # each tensor with ``flyc.from_c_void_p`` (see ``_as_ptr`` in the host
    # wrapper module), which requires flydsl >=0.2.0.

    _fast_exp = _make_fast_exp(G_IS_LOG2_SCALED)
    # fp32->bf16 output converter selected by the compile option (arch-aware).
    _f32x4_to_bf16x4 = _make_bf16_converter(BF16_CONVERT_TRUNC)

    WARP_SIZE = 64
    NUM_WARPS = 4
    BLOCK_THREADS = NUM_WARPS * WARP_SIZE

    WMMA_N = 16
    WMMA_K = 32
    N_REPEAT = BV // WMMA_N

    NUM_H_ACCS = NUM_K_BLOCKS * N_REPEAT

    # Warp partition (split-M only): this fork uses ONLY the "wid owns
    # 16 BT rows, all V" split-M warp partition -- the SAME warp scheme as the
    # hand-tuned HIP/C++ K5 kernel (BT split-M in GEMM1, K split across waves in
    # GEMM2, V not split across warps) with the SAME 16x16x16 bf16 MFMA. The
    # alternative OPT-VWARP layout has been removed entirely from this fork.

    # w panels (w_panel0/1), one [BT][64] panel per 64-K block,
    # written with HIP's ``w_panel_swizzle`` (bank-conflict-free) and read by
    # GEMM1 via ``load_a_w_fragment_swizzled`` (plain b64, contiguous 16-K).
    LDS_WP_PANEL_ELEMS = BT * 64
    LDS_WP_ELEMS = NUM_K_BLOCKS * LDS_WP_PANEL_ELEMS

    # k in rotating-pair swizzled panels (one per 64-K block).
    # Mirrors HIP's k_panel0[64*BT] + k_panel_rotating_pair_addr_bytes layout
    # exactly: global load reads b128 (8 bf16) for adjacent token pairs (t0,t1),
    # then scatters b16x2 packed writes to swizzled LDS addresses; GEMM2 reads
    # the MFMA A frag via load_a_k_fragment_rotating (plain b64 from the
    # swizzled address, no ds_read_tr16). Panel size = 64 K-rows * BT tokens
    # = 64*BT bf16 elements per panel.
    LDS_KP_PANEL_ELEMS = 64 * BT
    LDS_KP_ELEMS = NUM_K_BLOCKS * LDS_KP_PANEL_ELEMS

    # gated v_new in a shared2 panel (like h / HIP gated_v_panel);
    # GEMM2 B read via load_shared2 (contiguous BT, matches the k A frag).
    LDS_GV_ELEMS = (BT // 4) * BV * 4  # 16 bt_groups x BV cols x 4 BT

    # h_state panels (shared2 layout) for the GEMM1 B
    # operand. One [row_block][V] panel per 64-K block (like HIP's
    # h_state_panel0/1); GEMM1 reads the MFMA B fragment with a plain b64 load
    # (load_b_shared2) instead of the hardware-transpose ds_read_tr16. Each
    # cell holds 4 K (a k_group): 64/4=16 row_blocks x BV cols x 4 K.
    LDS_HP_PANEL_ELEMS = (64 // 4) * BV * 4  # = 64 * BV per 64-K block
    LDS_HP_ELEMS = NUM_K_BLOCKS * LDS_HP_PANEL_ELEMS
    # gated_v aliases h_state panel 1, so it must fit that panel exactly.
    assert LDS_GV_ELEMS == LDS_HP_PANEL_ELEMS

    # [V][K] transpose buffer for the h snapshot HBM store.
    # h_accs (f32) is written here in V-major / K-innermost order so 8 adjacent
    # K land contiguously -> a single b128 (ds_read_b128 + buffer_store b128),
    # matching HIP's ``h_transpose_buf`` + ``coalesced_vk_store_from_transpose``
    # (fewer store instructions than the per-element bf16 readout). K-contiguous
    # (no pad) so the 8-wide vec load/store stays 16 B aligned.
    LDS_HT_STRIDE = K
    LDS_HT_ELEMS = BV * LDS_HT_STRIDE

    # ``hp`` must stay a single Array: GEMM1 indexes across both 64-K panels as
    # ``kb * LDS_HP_PANEL_ELEMS + ...``, which requires the two panels to be
    # contiguous. Separate struct fields are independent static LDS symbols with
    # no guaranteed adjacency.
    @fx.struct
    class SharedStorage:
        wp: fx.Array[fx.BFloat16, LDS_WP_ELEMS, 16]
        kp: fx.Array[fx.BFloat16, LDS_KP_ELEMS, 16]
        hp: fx.Array[fx.BFloat16, LDS_HP_ELEMS, 16]
        ht: fx.Array[fx.BFloat16, LDS_HT_ELEMS, 16]

    # Cooperative load parameters
    LOAD_VEC_WIDTH = 8  # 8 bf16 = 16 bytes = buffer_load_dwordx4
    THREADS_PER_ROW_64 = 64 // LOAD_VEC_WIDTH  # 8
    ROWS_PER_BATCH_64 = BLOCK_THREADS // THREADS_PER_ROW_64  # 32
    NUM_LOAD_BATCHES_64 = BT // ROWS_PER_BATCH_64  # 2

    # GEMM1 issues K_STEPS_PER_BLOCK 16x16x16 MFMA steps per 64-K block; the
    # u/g/gk/w_next prefetch is emitted inline in the kernel body (before/within
    # GEMM1) so the MFMA chain hides its HBM latency.
    K_STEPS_PER_BLOCK = 64 // WMMA_K

    @flyc.kernel(name="chunk_gdn_fwd_h_flydsl_mfma16_hip")
    def gdn_h_kernel(
        k_tensor: fx.Pointer,
        v_tensor: fx.Pointer,
        w_tensor: fx.Pointer,
        v_new_tensor: fx.Pointer,
        g_tensor: fx.Pointer,
        gk_tensor: fx.Pointer,
        h_tensor: fx.Pointer,
        h0_tensor: fx.Pointer,
        ht_tensor: fx.Pointer,
        cu_seqlens_tensor: fx.Pointer,
        chunk_offsets_tensor: fx.Pointer,
        state_indices_tensor: fx.Pointer,
        T_val: fx.Int32,
        T_flat: fx.Int32,
        N_val: fx.Int32,
    ):
        i_v = fx.Int32(gpu.block_id("x"))
        i_nh = fx.Int32(gpu.block_id("y"))
        i_n = i_nh // fx.Int32(H)
        i_h = i_nh % fx.Int32(H)

        # Indexed state-pool gather: when USE_STATE_INDICES, the SSM state slot
        # for sequence ``i_n`` is ``state_indices[i_n]`` (addressing a pool
        # ``[pool_size, H, V, K]``) rather than ``i_n`` itself (dense
        # ``[N, H, V, K]``). Only h0 (read) and ht (in-place write-back) use this
        # slot; the per-chunk h snapshot stays dense (i_n-indexed).
        if const_expr(USE_STATE_INDICES):
            si_ = GTensor(state_indices_tensor, dtype=T.i32, shape=(-1,))
            state_n = si_[fx.Int64(i_n)]
        else:
            state_n = i_n
        state_nh = state_n * fx.Int32(H) + i_h

        tid = fx.Int32(gpu.thread_id("x"))
        wid = tid // fx.Int32(WARP_SIZE)
        lane = tid % fx.Int32(WARP_SIZE)

        k_ = GTensor(k_tensor, dtype=T.bf16, shape=(-1,))
        v_ = GTensor(v_tensor, dtype=T.bf16, shape=(-1,))
        w_ = GTensor(w_tensor, dtype=T.bf16, shape=(-1,))
        h_ = GTensor(h_tensor, dtype=T.bf16, shape=(-1,))
        g_ = GTensor(g_tensor, dtype=T.f32, shape=(-1,))
        if const_expr(USE_GK):
            gk_ = GTensor(gk_tensor, dtype=T.f32, shape=(-1,))

        vn_ = GTensor(v_new_tensor, dtype=T.bf16, shape=(-1,))
        # SSM-state dtype is selected by the compile-time flag; ``T.f32`` /
        # ``T.bf16`` must be evaluated *inside* the kernel body where an MLIR
        # context is active (mirrors how ``gdr_decode.py`` resolves
        # ``state_dtype_`` from inside its kernel function).
        state_t = T.bf16 if STATE_DTYPE_BF16 else T.f32
        if const_expr(USE_INITIAL_STATE):
            h0_ = GTensor(h0_tensor, dtype=state_t, shape=(-1,))
        if const_expr(STORE_FINAL_STATE):
            ht_ = GTensor(ht_tensor, dtype=state_t, shape=(-1,))

        if const_expr(IS_VARLEN):
            cu_ = GTensor(cu_seqlens_tensor, dtype=T.i32, shape=(-1,))
            co_ = GTensor(chunk_offsets_tensor, dtype=T.i32, shape=(-1,))

        # -- LDS --
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_wp_ptr = lds.wp.ptr  # w panels (HIP w_panel swizzle)
        lds_kp_ptr = lds.kp.ptr  # k panels (rotating-pair swizzle)
        lds_hp_ptr = lds.hp.ptr  # h_state panels (shared2, GEMM1 B operand)
        lds_ht_ptr = lds.ht.ptr  # [V][K] transpose buffer for the b128 h store
        # gated_v ALIASES h_state panel 1 (like HIP's ``gated_v_panel =
        # h_state_panel1``): gated_v is written only AFTER GEMM1 has finished
        # reading the h_state panels, and a WAR barrier before the gated_v store
        # enforces that ordering across warps. Aliasing avoids a separate
        # LDS_GV buffer, which would cost an occupancy step.
        lds_gv_ptr = lds_hp_ptr + fx.Int32(LDS_HP_PANEL_ELEMS)

        v4bf16 = T.vec(4, T.bf16)

        # Plain b64 read of a shared2 panel cell (GEMM1 / GEMM2 B frag).
        def _lds_read_hp_bf16x4(elem_idx):
            return fx.ptr_load(lds_hp_ptr + fx.Int32(elem_idx), result_type=v4bf16)

        def _lds_read_gv_bf16x4(elem_idx):
            return fx.ptr_load(lds_gv_ptr + fx.Int32(elem_idx), result_type=v4bf16)

        # -- Cooperative load decomposition --
        load_row_in_batch = tid // fx.Int32(THREADS_PER_ROW_64)
        load_col_base = (tid % fx.Int32(THREADS_PER_ROW_64)) * fx.Int32(LOAD_VEC_WIDTH)

        # w_panel swizzle (returns ELEMENT offset within a panel).
        # Port of w_panel_swizzle_base_bytes >> 1 (all bf16 = 2 B).
        #   row_in_half = row & 31; col_group = col_base >> 3
        #   tid = row_in_half*8 + col_group
        #   base_bytes = ((tid<<4)&4080) ^ (tid&120); if row&32: base|=4096
        def _w_panel_swz_elems(row, col_base):
            row_in_half = row & fx.Int32(31)
            col_group = col_base >> fx.Int32(3)
            tid_like = row_in_half * fx.Int32(8) + col_group
            base = ((tid_like << fx.Int32(4)) & fx.Int32(4080)) ^ (
                tid_like & fx.Int32(120)
            )
            base = base | ((row & fx.Int32(32)) << fx.Int32(7))  # 32<<7 = 4096
            return base >> fx.Int32(1)  # bytes -> bf16 elements

        # load_a_w_fragment_swizzled (contiguous 16-K A frag).
        # Caller passes row = row_base + lane&15 and k0 = k_base + (lane>>4)*4;
        # the ^4-elem (^8-byte) toggle picks the low/high 4 within an 8-group.
        def _load_a_w_swizzled(panel_base_elems, row, k0):
            col_base = k0 & fx.Int32(~7)
            elem = _w_panel_swz_elems(row, col_base) ^ (k0 & fx.Int32(4))
            return fx.ptr_load(
                lds_wp_ptr + fx.Int32(panel_base_elems + elem), result_type=v4bf16
            )

        # k_panel rotating-pair swizzle address computation.
        # Port of HIP's k_panel_rotating_pair_base_bytes / _addr_bytes.
        # Returns a BYTE offset within a panel; caller converts to element
        # offset (>> 1) before indexing the bf16 LDS pointer.
        def _k_panel_rotating_pair_addr_bytes(row, pair_col):
            row_block = row >> fx.Int32(3)
            row_in_block = row & fx.Int32(7)
            # k_panel_rotating_pair_base_bytes(row_block, pair_col)
            tid_like = (pair_col << fx.Int32(3)) | row_block
            lane_1_2 = tid_like & fx.Int32(6)
            base = lane_1_2 << fx.Int32(10)
            low = (lane_1_2 << fx.Int32(2)) ^ (
                (tid_like & fx.Int32(0xF8)) >> fx.Int32(1)
            )
            toggle = (tid_like & fx.Int32(1)) * fx.Int32(0x440)
            low = low ^ toggle
            base_bytes = base | low
            return (base_bytes ^ (row_in_block << fx.Int32(3))) + (
                row_in_block << fx.Int32(7)
            )

        # load_a_k_fragment_rotating -- GEMM2 A operand read.
        # Mirrors HIP's load_a_k_fragment_rotating(base, row_base, t_base, lane).
        def _load_a_k_rotating(panel_base_elems, row_base, t_base):
            row = row_base + lane_n
            t0 = t_base + lane_m_base * fx.Int32(4)
            pair_col = t0 >> fx.Int32(1)
            byte_off = _k_panel_rotating_pair_addr_bytes(row, pair_col)
            elem_off = byte_off >> fx.Int32(1)
            return fx.ptr_load(
                lds_kp_ptr + fx.Int32(panel_base_elems + elem_off), result_type=v4bf16
            )

        # -- Prologue: compute bos, T_local, NT, boh --
        if const_expr(IS_VARLEN):
            bos = cu_[fx.Int64(i_n)]
            eos = cu_[fx.Int64(i_n) + fx.Int64(1)]
            T_local = eos - bos
            NT = (T_local + fx.Int32(BT - 1)) // fx.Int32(BT)
            boh = co_[fx.Int64(i_n)]
        else:
            bos = i_n * T_val
            T_local = T_val
            NT = (T_local + fx.Int32(BT - 1)) // fx.Int32(BT)
            boh = i_n * NT

        # -- Base pointer offsets (element counts) --
        # Keep all global-memory address arithmetic in i64. Long-context
        # snapshot tensors can exceed 2^31 elements even though per-chunk and
        # LDS coordinates remain comfortably within i32.
        i_n64 = fx.Int64(i_n)
        i_h64 = fx.Int64(i_h)
        bos64 = fx.Int64(bos)
        boh64 = fx.Int64(boh)
        t_flat64 = fx.Int64(T_flat)

        # h: [B, NT, H, V, K] (VK) -- base = (boh*H + i_h) * V * K
        h_base = (boh64 * fx.Int64(H) + i_h64) * fx.Int64(V * K)
        stride_h = fx.Int64(H * V * K)

        # k: [B, T, Hg, K] -- base = (bos*Hg + i_h//(H//Hg)) * K
        gqa_ratio = H // Hg
        k_base = (
            bos64 * fx.Int64(Hg) + fx.Int64(i_h // fx.Int32(gqa_ratio))
        ) * fx.Int64(K)
        stride_k = fx.Int64(Hg * K)

        if const_expr(WU_CONTIGUOUS):
            if const_expr(IS_VARLEN):
                v_base = (i_h64 * t_flat64 + bos64) * fx.Int64(V)
                w_base = (i_h64 * t_flat64 + bos64) * fx.Int64(K)
            else:
                v_base = ((i_n64 * fx.Int64(H) + i_h64) * t_flat64) * fx.Int64(V)
                w_base = ((i_n64 * fx.Int64(H) + i_h64) * t_flat64) * fx.Int64(K)
            stride_v = fx.Int64(V)
            stride_w = fx.Int64(K)
        else:
            v_base = (bos64 * fx.Int64(H) + i_h64) * fx.Int64(V)
            w_base = (bos64 * fx.Int64(H) + i_h64) * fx.Int64(K)
            stride_v = fx.Int64(H * V)
            stride_w = fx.Int64(H * K)

        if const_expr(IS_VARLEN):
            vn_base = (i_h64 * t_flat64 + bos64) * fx.Int64(V)
        else:
            vn_base = ((i_n64 * fx.Int64(H) + i_h64) * t_flat64) * fx.Int64(V)

        if const_expr(USE_INITIAL_STATE):
            h0_base = fx.Int64(state_nh) * fx.Int64(V * K)
        if const_expr(STORE_FINAL_STATE):
            ht_base = fx.Int64(state_nh) * fx.Int64(V * K)

        # g gate offset. Mirrors the HIP kernel's load_g_value:
        #   idx = i_n*g_stride_b + i_h*g_stride_h + token*g_stride_t
        # with the strides derived from the layout:
        #   head-major  [B, H, T_flat] -> g_stride_h=T_flat, g_stride_t=1
        #   token-major [B, T_flat, H] -> g_stride_h=1,      g_stride_t=H
        # varlen (B==1, flattened): batch stride is 0 and token is the global
        #   token (bos + rel_row), so bos*g_stride_t is folded into the base.
        #   dense: batch stride is H*T_flat (same element count for both
        #   layouts) and token is the in-sequence relative row.
        # The base below excludes the per-token term; callers add
        #   token * g_stride_t (token = last_idx_raw / safe_row, relative).
        if const_expr(USE_G):
            if const_expr(G_HEAD_MAJOR):
                g_stride_h = t_flat64
                g_stride_t = fx.Int64(1)
            else:
                g_stride_h = fx.Int64(1)
                g_stride_t = fx.Int64(H)
            if const_expr(IS_VARLEN):
                g_sh_base = i_h64 * g_stride_h + bos64 * g_stride_t
            else:
                g_sh_base = i_n64 * fx.Int64(H) * t_flat64 + i_h64 * g_stride_h

        # -- MFMA lane mapping for 16x16 tiles --
        lane_n = lane % fx.Int32(16)
        lane_m_base = lane // fx.Int32(16)

        # -- Initialize h accumulators --
        acc_zero = fx.full(4, 0.0, fx.Float32)

        # h_accs[kb][nr] = f32x4 accumulator for k-block kb, v-repeat nr
        h_accs = []
        for _kb in range_constexpr(NUM_K_BLOCKS):
            for _nr in range_constexpr(N_REPEAT):
                h_accs.append(acc_zero)

        # -- Load initial state if provided --
        # OPT-F: 4 x scalar f32 load -> 1 x buffer_load_dwordx4 (16 B).
        # h0 is [V, K] so K is innermost; 4 consecutive K positions are
        # contiguous in memory -> a single vec_load(4) covers them.
        if const_expr(USE_INITIAL_STATE):
            for kb in range_constexpr(NUM_K_BLOCKS):
                for slot in range_constexpr(N_REPEAT):
                    h0_col = i_v * fx.Int32(BV) + fx.Int32(slot * 16) + lane_n
                    h0_row_base = (
                        fx.Int32(kb * 64)
                        + wid * fx.Int32(16)
                        + lane_m_base * fx.Int32(4)
                    )
                    h0_off_base = (
                        h0_base + fx.Int64(h0_col) * fx.Int64(K) + fx.Int64(h0_row_base)
                    )
                    loaded_vec = h0_.vec_load((h0_off_base,), 4)
                    if const_expr(STATE_DTYPE_BF16):
                        loaded_vec = loaded_vec.extf(T.f32x4)
                    acc_idx = kb * N_REPEAT + slot
                    h_accs[acc_idx] = h_accs[acc_idx] + loaded_vec

        # -- HIP-aligned pipelined main chunk loop --
        # Prologue loads w/k for chunk 0 to LDS; each iteration prefetches
        # NEXT chunk's w (during GEMM1) and k (after gating) into VGPRs,
        # then writes them to LDS at GEMM2 end (mirrors HIP .cu:1125-1194).
        GEMM1_PF_SPLIT = min(1, NUM_K_BLOCKS)
        k_row_base_pf = (lane & fx.Int32(7)) * fx.Int32(8)
        k_pair_col_pf = wid * fx.Int32(8) + (lane >> fx.Int32(3))
        k_t0_pf = k_pair_col_pf * fx.Int32(2)
        k_t1_pf = k_t0_pf + fx.Int32(1)

        init_state = [_to_raw(v) for v in h_accs]
        c_zero = fx.Int64(0)
        c_one = fx.Int64(1)
        nt_idx = fx.Int64(NT)

        # -- PROLOGUE: load w/k for chunk 0 to LDS (HIP .cu:1125-1141) --
        # Empty varlen sequences (T_local==0 -> NT==0) need no prefetch; if not
        # skipped, safe_row is clamped to 0 while a trailing empty sequence has
        # bos==T_flat, so the w/k global address runs past the end of the input
        # buffer -> out-of-bounds read. Use a block-uniform NT>0 guard to skip
        # the whole prologue (including its barrier). For an empty sequence the
        # main loop also runs 0 times, results simply carry init_state(=h0)
        # through, and the epilogue writes h0 back to ht -- state passes through
        # correctly.
        #
        # The body lives in the _load_chunk0_to_lds closure so the ``if`` body
        # is just "call closure + barrier": the GTensor wrappers (w_/k_) are
        # captured as free variables rather than appearing inside the if, which
        # would make the FlyDSL AST rewriter try to carry them as scf.if state
        # and fail with "GTensor is not an MLIR Value".
        def _load_chunk0_to_lds():
            _prol_it = fx.Int32(0)
            for kb in range_constexpr(NUM_K_BLOCKS):
                wp_panel_base = fx.Int32(kb * LDS_WP_PANEL_ELEMS)
                for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                    row = fx.Int32(batch * ROWS_PER_BATCH_64) + load_row_in_batch
                    abs_row = _prol_it * fx.Int32(BT) + row
                    safe_row = (abs_row < T_local).select(abs_row, fx.Int32(0))
                    w_g_off = (
                        w_base
                        + fx.Int64(safe_row) * stride_w
                        + fx.Int64(kb * 64)
                        + fx.Int64(load_col_base)
                    )
                    wvec = w_.vec_load((w_g_off,), LOAD_VEC_WIDTH)
                    swz = wp_panel_base + _w_panel_swz_elems(row, load_col_base)
                    fx.ptr_store(
                        wvec.shuffle(wvec, [0, 1, 2, 3]), lds_wp_ptr + fx.Int32(swz)
                    )
                    fx.ptr_store(
                        wvec.shuffle(wvec, [4, 5, 6, 7]),
                        lds_wp_ptr + fx.Int32(swz ^ fx.Int32(4)),
                    )
            k_abs_t0_prol = _prol_it * fx.Int32(BT) + k_t0_pf
            k_abs_t1_prol = _prol_it * fx.Int32(BT) + k_t1_pf
            k_safe_t0_prol = (k_abs_t0_prol < T_local).select(
                k_abs_t0_prol, fx.Int32(0)
            )
            k_safe_t1_prol = (k_abs_t1_prol < T_local).select(
                k_abs_t1_prol, fx.Int32(0)
            )
            for kb in range_constexpr(NUM_K_BLOCKS):
                kp_pbase = fx.Int32(kb * LDS_KP_PANEL_ELEMS)
                k_col_off = fx.Int32(kb * 64) + k_row_base_pf
                k_g_off_t0 = (
                    k_base + fx.Int64(k_safe_t0_prol) * stride_k + fx.Int64(k_col_off)
                )
                k_g_off_t1 = (
                    k_base + fx.Int64(k_safe_t1_prol) * stride_k + fx.Int64(k_col_off)
                )
                kvec_t0 = k_.vec_load((k_g_off_t0,), LOAD_VEC_WIDTH)
                kvec_t1 = k_.vec_load((k_g_off_t1,), LOAD_VEC_WIDTH)
                for i in range_constexpr(LOAD_VEC_WIDTH):
                    row_i = k_row_base_pf + fx.Int32(i)
                    byte_off = _k_panel_rotating_pair_addr_bytes(row_i, k_pair_col_pf)
                    elem_off = byte_off >> fx.Int32(1)
                    fx.ptr_store(kvec_t0[i], lds_kp_ptr + fx.Int32(kp_pbase + elem_off))
                    fx.ptr_store(
                        kvec_t1[i],
                        lds_kp_ptr + fx.Int32(kp_pbase + elem_off + fx.Int32(1)),
                    )

        # NT comes from integer division and is a raw ArithValue; wrap it in
        # fx.Int32 before comparing so the result is an fx boolean.
        has_work = fx.Int32(NT) > fx.Int32(0)
        if has_work:
            _load_chunk0_to_lds()
            gpu.barrier()

        for i_t, state in range(c_zero, nt_idx, c_one, init=init_state):
            # scf.for hands back its iter_args as raw ir.Values; re-type them so
            # the loop body stays on the fx.Vector method API.
            h_accs_in = [fx.Vector(v, (4,), fx.Float32) for v in state[:NUM_H_ACCS]]
            i_t_i32 = fx.Int32(i_t)

            # Stage h_accs into (a) the h_state panels [row_block][V] for the
            # GEMM1 B operand (HIP shared2 layout, plain b64 read) and (b) the
            # [V][K] transpose buffer for the b128 HBM store. split-M mapping:
            # wid -> K sub-tile, acc_j(nr) -> V-tile.
            for kb in range_constexpr(NUM_K_BLOCKS):
                for acc_j in range_constexpr(N_REPEAT):
                    acc_idx = kb * N_REPEAT + acc_j
                    acc_val = h_accs_in[acc_idx]
                    # acc_j == nr (V-tile); K sub-tile = wid*16.
                    hp_col = fx.Int32(acc_j * 16) + lane_n

                    # Write the h_state panel cell (shared2). This
                    # lane owns k_group (row_block = wid*4+lane_m_base) at V-col
                    # = nr*16+lane_n; 4 warps together fill all 16 row_blocks.
                    hp_row_block = wid * fx.Int32(4) + lane_m_base
                    hp_cell = fx.Int32(kb * LDS_HP_PANEL_ELEMS) + (
                        hp_row_block * fx.Int32(BV) + hp_col
                    ) * fx.Int32(4)
                    fx.ptr_store(
                        _f32x4_to_bf16x4(acc_val), lds_hp_ptr + fx.Int32(hp_cell)
                    )

                    # [V][K/4-group] transpose buffer, XOR-swizzled
                    # (k_group ^ (v & 0xF)) to break bank conflicts on the
                    # scatter write -- mirrors HIP ``h_transpose_buf_offset``.
                    # The 4 elem_i are one k_group (4 contiguous K) -> one b64.
                    ht_kg = fx.Int32(kb * 16) + wid * fx.Int32(4) + lane_m_base
                    ht_idx = (
                        hp_col * fx.Int32(K // 4) + (ht_kg ^ (hp_col & fx.Int32(0xF)))
                    ) * fx.Int32(4)
                    fx.ptr_store(
                        _f32x4_to_bf16x4(acc_val), lds_ht_ptr + fx.Int32(ht_idx)
                    )

            # w/k for this chunk already in LDS (prologue or prev GEMM2 end).
            gpu.barrier()

            # last_idx for the current chunk (gating).
            next_chunk_end = (i_t_i32 + fx.Int32(1)) * fx.Int32(BT)
            last_idx_raw = (next_chunk_end < T_local).select(
                next_chunk_end, T_local
            ) - fx.Int32(1)

            # >>> PREFETCH u + g BEFORE GEMM1: issue all HBM loads for u
            # (N_REPEAT × 4 ushort) and g (4 rows + 1 g_last = 5 dword) now,
            # so the full 64-MFMA GEMM1 chain hides their HBM latency.
            # Without this, LLVM hoists the loads into the GEMM1 middle where
            # only ~2 MFMA can hide them → 4.2M cycle vmcnt stall (39% of stalls).
            u_prefetch = []  # N_REPEAT × 4 bf16 scalars
            for idx in range_constexpr(N_REPEAT):
                u_col_pf = i_v * fx.Int32(BV) + fx.Int32(idx * 16) + lane_n
                for elem_i in range_constexpr(4):
                    u_bt_row_raw = (
                        i_t_i32 * fx.Int32(BT)
                        + wid * fx.Int32(16)
                        + lane_m_base * fx.Int32(4)
                        + fx.Int32(elem_i)
                    )
                    safe_u_row = (u_bt_row_raw < T_local).select(
                        u_bt_row_raw, fx.Int32(0)
                    )
                    u_off = (
                        v_base + fx.Int64(safe_u_row) * stride_v + fx.Int64(u_col_pf)
                    )
                    u_prefetch.append(v_[u_off])

            if const_expr(USE_G):
                g_last = g_[g_sh_base + fx.Int64(last_idx_raw) * g_stride_t]
                g_row_pf = []
                for elem_i in range_constexpr(4):
                    abs_row = (
                        i_t_i32 * fx.Int32(BT)
                        + wid * fx.Int32(16)
                        + lane_m_base * fx.Int32(4)
                        + fx.Int32(elem_i)
                    )
                    in_bounds = abs_row < T_local
                    safe_row = in_bounds.select(abs_row, fx.Int32(0))
                    g_row_pf.append(
                        (
                            g_[g_sh_base + fx.Int64(safe_row) * g_stride_t],
                            in_bounds,
                        )
                    )

            # -- GEMM1: b_v = w @ h_state, with w_next prefetch interleaved.
            # u/g/w_next loads are all in flight; 64 MFMA hide HBM latency.
            bv_accs = []
            for _i in range_constexpr(N_REPEAT):
                bv_accs.append(fx.full(4, 0.0, fx.Float32))

            # GEMM1 first K-block(s) -- before w prefetch.
            for kb in range_constexpr(GEMM1_PF_SPLIT):
                for ks in range_constexpr(K_STEPS_PER_BLOCK):
                    wp_pbase = fx.Int32(kb * LDS_WP_PANEL_ELEMS)
                    a_row = wid * fx.Int32(16) + lane_n
                    a_frag_lo = _load_a_w_swizzled(
                        wp_pbase,
                        a_row,
                        fx.Int32(ks * WMMA_K) + lane_m_base * fx.Int32(4),
                    )
                    a_frag_hi = _load_a_w_swizzled(
                        wp_pbase,
                        a_row,
                        fx.Int32(ks * WMMA_K + 16) + lane_m_base * fx.Int32(4),
                    )
                    for nr in range_constexpr(N_REPEAT):
                        hp_base = fx.Int32(kb * LDS_HP_PANEL_ELEMS)
                        hp_col_b = fx.Int32(nr * 16) + lane_n
                        rb_lo = fx.Int32((ks * WMMA_K) >> 2) + lane_m_base
                        rb_hi = fx.Int32(((ks * WMMA_K) + 16) >> 2) + lane_m_base
                        idx_lo = hp_base + (rb_lo * fx.Int32(BV) + hp_col_b) * fx.Int32(
                            4
                        )
                        idx_hi = hp_base + (rb_hi * fx.Int32(BV) + hp_col_b) * fx.Int32(
                            4
                        )
                        b_frag_lo = _lds_read_hp_bf16x4(idx_lo)
                        b_frag_hi = _lds_read_hp_bf16x4(idx_hi)
                        bv_accs[nr] = _mfma_bf16_16x16x16(
                            a_frag_lo, b_frag_lo, bv_accs[nr]
                        )
                        bv_accs[nr] = _mfma_bf16_16x16x16(
                            a_frag_hi, b_frag_hi, bv_accs[nr]
                        )
                    # gfx942 scheduling: insert sched_barrier(mask_mfma) at each
                    # ks boundary to stop LLVM from hoisting the per-ks b_frag
                    # ds_read across iterations into one big cluster (the main
                    # cause of LDS port back-pressure), while still allowing MFMA
                    # to overlap across ks to keep pipeline latency hidden.
                    # sched_barrier only constrains instruction ordering, not
                    # addresses/values, so it is correctness-safe.
                    if const_expr(SCHED_GFX942):
                        rocdl.sched_barrier(rocdl.mask_mfma)

            # >>> PREFETCH w_next: HBM loads for next chunk (HIP .cu:670-672).
            next_i_t = i_t_i32 + fx.Int32(1)
            w_next_vecs = []
            w_next_swz = []
            for kb in range_constexpr(NUM_K_BLOCKS):
                wp_pb_next = fx.Int32(kb * LDS_WP_PANEL_ELEMS)
                for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                    row = fx.Int32(batch * ROWS_PER_BATCH_64) + load_row_in_batch
                    abs_row_next = next_i_t * fx.Int32(BT) + row
                    safe_row_next = (abs_row_next < T_local).select(
                        abs_row_next, fx.Int32(0)
                    )
                    w_g_off_next = (
                        w_base
                        + fx.Int64(safe_row_next) * stride_w
                        + fx.Int64(kb * 64)
                        + fx.Int64(load_col_base)
                    )
                    w_next_vecs.append(w_.vec_load((w_g_off_next,), LOAD_VEC_WIDTH))
                    w_next_swz.append(
                        wp_pb_next + _w_panel_swz_elems(row, load_col_base)
                    )

            # GEMM1 remaining K-blocks -- MFMA hides u/g/w_next HBM latency.
            for kb in range_constexpr(GEMM1_PF_SPLIT, NUM_K_BLOCKS):
                for ks in range_constexpr(K_STEPS_PER_BLOCK):
                    wp_pbase = fx.Int32(kb * LDS_WP_PANEL_ELEMS)
                    a_row = wid * fx.Int32(16) + lane_n
                    a_frag_lo = _load_a_w_swizzled(
                        wp_pbase,
                        a_row,
                        fx.Int32(ks * WMMA_K) + lane_m_base * fx.Int32(4),
                    )
                    a_frag_hi = _load_a_w_swizzled(
                        wp_pbase,
                        a_row,
                        fx.Int32(ks * WMMA_K + 16) + lane_m_base * fx.Int32(4),
                    )
                    for nr in range_constexpr(N_REPEAT):
                        hp_base = fx.Int32(kb * LDS_HP_PANEL_ELEMS)
                        hp_col_b = fx.Int32(nr * 16) + lane_n
                        rb_lo = fx.Int32((ks * WMMA_K) >> 2) + lane_m_base
                        rb_hi = fx.Int32(((ks * WMMA_K) + 16) >> 2) + lane_m_base
                        idx_lo = hp_base + (rb_lo * fx.Int32(BV) + hp_col_b) * fx.Int32(
                            4
                        )
                        idx_hi = hp_base + (rb_hi * fx.Int32(BV) + hp_col_b) * fx.Int32(
                            4
                        )
                        b_frag_lo = _lds_read_hp_bf16x4(idx_lo)
                        b_frag_hi = _lds_read_hp_bf16x4(idx_hi)
                        bv_accs[nr] = _mfma_bf16_16x16x16(
                            a_frag_lo, b_frag_lo, bv_accs[nr]
                        )
                        bv_accs[nr] = _mfma_bf16_16x16x16(
                            a_frag_hi, b_frag_hi, bv_accs[nr]
                        )
                    # gfx942 scheduling: same as above, break up the remaining
                    # K-block's cross-ks ds_read clustering while allowing MFMA.
                    if const_expr(SCHED_GFX942):
                        rocdl.sched_barrier(rocdl.mask_mfma)

            # WAR barrier (.cu:692).
            gpu.barrier()

            # -- FUSED v_new + gating + gated_v store --
            # Consume prefetched u/g (already in VGPRs from before GEMM1).
            if const_expr(USE_G):
                exp_g_last = _fast_exp(g_last)
                gate_elems = []
                for elem_i in range_constexpr(4):
                    g_row_val, in_bounds = g_row_pf[elem_i]
                    gate = _fast_exp(g_last - g_row_val)
                    gate_elems.append(in_bounds.select(gate, fx.Float32(0.0)))
                gate_vec = fx.Vector.from_elements(gate_elems, dtype=fx.Float32)
            else:
                # Even without g we must mask the last chunk's padding rows
                # (abs_row >= T_local). Otherwise these invalid tokens' v_new
                # would flow through gated_v into GEMM2 (h += k^T @ v_new_gated)
                # and corrupt the state update. The USE_G path already masks via
                # gate=0; here a 0/1 mask achieves the same effect.
                mask_elems = []
                for elem_i in range_constexpr(4):
                    abs_row = (
                        i_t_i32 * fx.Int32(BT)
                        + wid * fx.Int32(16)
                        + lane_m_base * fx.Int32(4)
                        + fx.Int32(elem_i)
                    )
                    in_bounds = abs_row < T_local
                    mask_elems.append(
                        in_bounds.select(fx.Float32(1.0), fx.Float32(0.0))
                    )
                gate_vec = fx.Vector.from_elements(mask_elems, dtype=fx.Float32)

            if const_expr(SAVE_NEW_VALUE):
                # Wrapping the store in a bare function call hides it from the
                # ``ReplaceIfWithDispatch`` AST rewriter, which scans subscript
                # stores and ``obj.method()`` calls inside dynamic ``if`` bodies
                # and demands MLIR-Value state for any name written to. ``vn_``
                # is a GTensor (HBM tensor wrapper), not an MLIR Value; an
                # ast.Name call makes the analyzer skip it.
                def _emit_vn_store(off, value):
                    vn_[off] = value

            for idx in range_constexpr(N_REPEAT):
                bv_val = bv_accs[idx]
                u_col = i_v * fx.Int32(BV) + fx.Int32(idx * 16) + lane_n
                u_f32_elems = []
                for elem_i in range_constexpr(4):
                    u_raw = u_prefetch[idx * 4 + elem_i]
                    u_bf16 = fx.BFloat16(u_raw)
                    u_f32_elems.append(u_bf16.to(fx.Float32))
                u_f32 = fx.Vector.from_elements(u_f32_elems, dtype=fx.Float32)
                vn_val = u_f32 - bv_val

                if const_expr(SAVE_NEW_VALUE):
                    vn_bf16 = fx.Vector(_f32x4_to_bf16x4(vn_val), (4,), fx.BFloat16)
                    bt_tile_base = wid * fx.Int32(16)
                    for elem_i in range_constexpr(4):
                        vn_bt_row = (
                            i_t_i32 * fx.Int32(BT)
                            + bt_tile_base
                            + lane_m_base * fx.Int32(4)
                            + fx.Int32(elem_i)
                        )
                        if vn_bt_row < T_local:
                            bf16_v = vn_bf16[elem_i]
                            vn_off = (
                                vn_base
                                + fx.Int64(vn_bt_row) * fx.Int64(V)
                                + fx.Int64(u_col)
                            )
                            _emit_vn_store(vn_off, bf16_v)

                # gate_vec: with USE_G it is the decay gate (with the OOB mask
                # already folded in); otherwise it is a pure 0/1 padding mask.
                # Both paths multiply by gate_vec so OOB rows always contribute 0.
                gated_val = vn_val * gate_vec
                gv_col = fx.Int32(idx * 16) + lane_n
                gv_row_block = wid * fx.Int32(4) + lane_m_base
                gv_cell = (gv_row_block * fx.Int32(BV) + gv_col) * fx.Int32(4)
                fx.ptr_store(
                    _f32x4_to_bf16x4(gated_val), lds_gv_ptr + fx.Int32(gv_cell)
                )

            # >>> PREFETCH k_next: HBM loads for next chunk (HIP .cu:727-731).
            # Overlaps with the barrier + h store below.
            k_abs_t0_next = next_i_t * fx.Int32(BT) + k_t0_pf
            k_abs_t1_next = next_i_t * fx.Int32(BT) + k_t1_pf
            k_safe_t0_next = (k_abs_t0_next < T_local).select(
                k_abs_t0_next, fx.Int32(0)
            )
            k_safe_t1_next = (k_abs_t1_next < T_local).select(
                k_abs_t1_next, fx.Int32(0)
            )
            k_next_vecs_t0 = []
            k_next_vecs_t1 = []
            for kb in range_constexpr(NUM_K_BLOCKS):
                k_col_off_pf = fx.Int32(kb * 64) + k_row_base_pf
                k_next_vecs_t0.append(
                    k_.vec_load(
                        (
                            k_base
                            + fx.Int64(k_safe_t0_next) * stride_k
                            + fx.Int64(k_col_off_pf),
                        ),
                        LOAD_VEC_WIDTH,
                    )
                )
                k_next_vecs_t1.append(
                    k_.vec_load(
                        (
                            k_base
                            + fx.Int64(k_safe_t1_next) * stride_k
                            + fx.Int64(k_col_off_pf),
                        ),
                        LOAD_VEC_WIDTH,
                    )
                )

            # Apply exp(g_last) decay to h_accs (scalar broadcast).
            if const_expr(USE_G):
                exp_g_last_s = fx.Float32(exp_g_last)
                for kb in range_constexpr(NUM_K_BLOCKS):
                    for nr in range_constexpr(N_REPEAT):
                        acc_idx = kb * N_REPEAT + nr
                        h_accs_in[acc_idx] = h_accs_in[acc_idx] * exp_g_last_s

            # Per-K decay: h[v, k] *= exp(gk_last[k]) at chunk end.
            if const_expr(USE_GK):
                gk_chunk_base = (bos64 + fx.Int64(last_idx_raw)) * fx.Int64(
                    H * K
                ) + i_h64 * fx.Int64(K)
                for kb in range_constexpr(NUM_K_BLOCKS):
                    gk_elems = []
                    for elem_i in range_constexpr(4):
                        global_k = (
                            fx.Int32(kb * 64)
                            + wid * fx.Int32(16)
                            + lane_m_base * fx.Int32(4)
                            + fx.Int32(elem_i)
                        )
                        gk_raw = gk_[gk_chunk_base + fx.Int64(global_k)]
                        gk_elems.append(_fast_exp(gk_raw))
                    gk_vec = fx.Vector.from_elements(gk_elems, dtype=fx.Float32)
                    for nr in range_constexpr(N_REPEAT):
                        acc_idx = kb * N_REPEAT + nr
                        h_accs_in[acc_idx] = h_accs_in[acc_idx] * gk_vec

            gpu.barrier()

            # -- 5. h store from XOR-swizzled transpose buffer (HIP-aligned:
            # after GEMM1, before GEMM2 -- mirrors HIP's
            # coalesced_vk_store_from_transpose called between run_gemm1 and
            # run_gemm2). The transpose buffer was populated during staging
            # above and is read-only hereafter; gated_v was written to a
            # different LDS region, so no conflict.
            K_VECS = K // LOAD_VEC_WIDTH
            NUM_HT_VECS = BV * K_VECS
            for vbase in range_constexpr(0, NUM_HT_VECS, BLOCK_THREADS):
                vec_idx = fx.Int32(vbase) + tid
                kv = vec_idx % fx.Int32(K_VECS)
                v_loc = vec_idx // fx.Int32(K_VECS)
                k8 = kv * fx.Int32(LOAD_VEC_WIDTH)
                v_xor = v_loc & fx.Int32(0xF)
                kg_lo = kv * fx.Int32(2)
                kg_hi = kg_lo + fx.Int32(1)
                off_lo = (v_loc * fx.Int32(K // 4) + (kg_lo ^ v_xor)) * fx.Int32(4)
                off_hi = (v_loc * fx.Int32(K // 4) + (kg_hi ^ v_xor)) * fx.Int32(4)
                val_lo = fx.Vector(
                    fx.ptr_load(lds_ht_ptr + fx.Int32(off_lo), result_type=v4bf16),
                    (4,),
                    fx.BFloat16,
                )
                val_hi = fx.Vector(
                    fx.ptr_load(lds_ht_ptr + fx.Int32(off_hi), result_type=v4bf16),
                    (4,),
                    fx.BFloat16,
                )
                vec8 = val_lo.shuffle(val_hi, [0, 1, 2, 3, 4, 5, 6, 7])
                v_global = i_v * fx.Int32(BV) + v_loc
                h_off = (
                    h_base
                    + fx.Int64(i_t_i32) * stride_h
                    + fx.Int64(v_global) * fx.Int64(K)
                    + fx.Int64(k8)
                )
                h_.vec_store((h_off,), vec8, LOAD_VEC_WIDTH)

            # -- 6. GEMM2: h += k^T @ v_new_gated (no w prefetch/interleave).
            BT_STEPS = BT // WMMA_K
            for kb in range_constexpr(NUM_K_BLOCKS):
                for bt_s in range_constexpr(BT_STEPS):
                    # A = k load_a_k_fragment_rotating.
                    # row_base = wid*16 within the panel; t_base lo/hi split.
                    kp_pbase = fx.Int32(kb * LDS_KP_PANEL_ELEMS)
                    k_a_lo = _load_a_k_rotating(
                        kp_pbase, wid * fx.Int32(16), fx.Int32(bt_s * WMMA_K)
                    )
                    k_a_hi = _load_a_k_rotating(
                        kp_pbase, wid * fx.Int32(16), fx.Int32(bt_s * WMMA_K + 16)
                    )

                    # gated_v B: shared2 layout (unchanged).
                    gv_rb_lo = fx.Int32((bt_s * WMMA_K) >> 2) + lane_m_base
                    gv_rb_hi = fx.Int32(((bt_s * WMMA_K) + 16) >> 2) + lane_m_base
                    for nr in range_constexpr(N_REPEAT):
                        gv_col = fx.Int32(nr * 16) + lane_n
                        vn_b_lo = _lds_read_gv_bf16x4(
                            (gv_rb_lo * fx.Int32(BV) + gv_col) * fx.Int32(4)
                        )
                        vn_b_hi = _lds_read_gv_bf16x4(
                            (gv_rb_hi * fx.Int32(BV) + gv_col) * fx.Int32(4)
                        )

                        acc_idx = kb * N_REPEAT + nr
                        h_accs_in[acc_idx] = _mfma_bf16_16x16x16(
                            k_a_lo, vn_b_lo, h_accs_in[acc_idx]
                        )
                        h_accs_in[acc_idx] = _mfma_bf16_16x16x16(
                            k_a_hi, vn_b_hi, h_accs_in[acc_idx]
                        )

            # >>> WRITE prefetched w_next/k_next to LDS for next iteration.
            # Barrier ensures GEMM2 done reading old panels (HIP .cu:794-798).
            has_next = next_i_t * fx.Int32(BT) < T_local
            if has_next:
                gpu.barrier()
                _pf_idx = 0
                for kb in range_constexpr(NUM_K_BLOCKS):
                    for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                        wvec_pf = w_next_vecs[_pf_idx]
                        swz_pf = w_next_swz[_pf_idx]
                        fx.ptr_store(
                            wvec_pf.shuffle(wvec_pf, [0, 1, 2, 3]),
                            lds_wp_ptr + fx.Int32(swz_pf),
                        )
                        fx.ptr_store(
                            wvec_pf.shuffle(wvec_pf, [4, 5, 6, 7]),
                            lds_wp_ptr + fx.Int32(swz_pf ^ fx.Int32(4)),
                        )
                        _pf_idx += 1
                for kb in range_constexpr(NUM_K_BLOCKS):
                    kp_pbase = fx.Int32(kb * LDS_KP_PANEL_ELEMS)
                    kvec_t0_pf = k_next_vecs_t0[kb]
                    kvec_t1_pf = k_next_vecs_t1[kb]
                    for i in range_constexpr(LOAD_VEC_WIDTH):
                        row_i = k_row_base_pf + fx.Int32(i)
                        byte_off = _k_panel_rotating_pair_addr_bytes(
                            row_i, k_pair_col_pf
                        )
                        elem_off = byte_off >> fx.Int32(1)
                        fx.ptr_store(
                            kvec_t0_pf[i], lds_kp_ptr + fx.Int32(kp_pbase + elem_off)
                        )
                        fx.ptr_store(
                            kvec_t1_pf[i],
                            lds_kp_ptr + fx.Int32(kp_pbase + elem_off + fx.Int32(1)),
                        )

            results = yield [_to_raw(v) for v in h_accs_in]

        h_accs_final = [fx.Vector(v, (4,), fx.Float32) for v in results[:NUM_H_ACCS]]

        # -- Epilogue: store final state --
        # OPT-7: 4 x scalar f32 store -> 1 x buffer_store_dwordx4 (16 B).
        # acc_val is already f32x4 with element i at K offset i -> vec_store
        # directly (no extract + from_elements needed).
        if const_expr(STORE_FINAL_STATE):
            for kb in range_constexpr(NUM_K_BLOCKS):
                for slot in range_constexpr(N_REPEAT):
                    acc_idx = kb * N_REPEAT + slot
                    acc_val = h_accs_final[acc_idx]

                    ht_col = i_v * fx.Int32(BV) + fx.Int32(slot * 16) + lane_n
                    ht_row_base = (
                        fx.Int32(kb * 64)
                        + wid * fx.Int32(16)
                        + lane_m_base * fx.Int32(4)
                    )
                    ht_off_base = (
                        ht_base + fx.Int64(ht_col) * fx.Int64(K) + fx.Int64(ht_row_base)
                    )
                    if const_expr(STATE_DTYPE_BF16):
                        out_vec = _f32x4_to_bf16x4(acc_val)
                    else:
                        out_vec = acc_val
                    ht_.vec_store((ht_off_base,), out_vec, 4)

    # -- Host launcher ------------------------------------------------------
    @flyc.jit
    def launch_gdn_h(
        k_tensor: fx.Pointer,
        v_tensor: fx.Pointer,
        w_tensor: fx.Pointer,
        v_new_tensor: fx.Pointer,
        g_tensor: fx.Pointer,
        gk_tensor: fx.Pointer,
        h_tensor: fx.Pointer,
        h0_tensor: fx.Pointer,
        ht_tensor: fx.Pointer,
        cu_seqlens_tensor: fx.Pointer,
        chunk_offsets_tensor: fx.Pointer,
        state_indices_tensor: fx.Pointer,
        T_val: fx.Int32,
        T_flat: fx.Int32,
        N_val: fx.Int32,
        grid_v: fx.Int32,
        grid_nh: fx.Int32,
        stream: fx.Stream,
    ):
        launcher = gdn_h_kernel(
            k_tensor,
            v_tensor,
            w_tensor,
            v_new_tensor,
            g_tensor,
            gk_tensor,
            h_tensor,
            h0_tensor,
            ht_tensor,
            cu_seqlens_tensor,
            chunk_offsets_tensor,
            state_indices_tensor,
            T_val,
            T_flat,
            N_val,
        )
        launcher.launch(
            grid=(grid_v, grid_nh, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_gdn_h


# NOTE: The Python host wrapper, BV autotune, and kernel cache live in
# ``aiter.ops.flydsl.linear_attention_prefill_kernels`` to keep this module
# free of any ``torch`` / ``triton`` dependency (mirrors the layering used
# by ``aiter.ops.flydsl.kernels.gdr_decode``).


__all__ = [
    "compile_chunk_gated_delta_h_mfma16_hip",
]
