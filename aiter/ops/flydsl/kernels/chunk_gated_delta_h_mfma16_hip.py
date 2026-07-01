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
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from .tensor_shim import GTensor, STensor, _to_raw

_LOG2E = math.log2(math.e)  # 1.4426950408889634
_LLVM_GEP_DYNAMIC = -2147483648


def _llvm_lds_ptr_ty():
    return ir.Type.parse("!llvm.ptr<3>")


def _make_fast_exp(g_is_log2_scaled: bool):
    """Return the ``exp`` helper for this kernel compile.

    If ``g_is_log2_scaled`` is False (default), ``g_cumsum`` is in the natural
    log domain (matches upstream K12) and we lower ``exp(x)`` as
    ``exp2(x * log2(e))`` so the multiplier merges into one ``v_exp_f32`` plus
    one ``v_mul_f32`` on AMD.

    If True, the caller has pre-scaled ``g_cumsum`` by ``log2(e)`` already
    (the K12 prescale optimization), so we can drop the per-call ``* LOG2E``
    multiply and lower directly to a single ``v_exp_f32``. NOTE: enabling
    this flag without the matching K12 prescale produces incorrect outputs;
    it exists for ISA-level perf probing of the prescale upper bound.
    """
    if g_is_log2_scaled:

        def _fast_exp(x):
            return rocdl.exp2(T.f32, x)

    else:

        def _fast_exp(x):
            return rocdl.exp2(T.f32, x * _LOG2E)

    return _fast_exp


def _mfma_bf16_16x16x32(a_bf16x8, b_bf16x8, acc_f32x4):
    """Single mfma_f32_16x16x32_bf16 instruction."""
    return rocdl.mfma_f32_16x16x32_bf16(
        T.f32x4, [a_bf16x8, b_bf16x8, acc_f32x4, 0, 0, 0]
    )


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
    return rocdl.mfma_f32_16x16x16bf16_1k(
        T.f32x4, [a_i16x4, b_i16x4, acc_f32x4, 0, 0, 0]
    )


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
    assert K <= 256
    assert K % 64 == 0
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

    WARP_SIZE = 64
    NUM_WARPS = 4
    BLOCK_THREADS = NUM_WARPS * WARP_SIZE

    WMMA_N = 16
    WMMA_K = 32
    N_REPEAT = BV // WMMA_N

    NUM_H_ACCS = NUM_K_BLOCKS * N_REPEAT

    # HIP-ALIGNED (non-VWARP / split-M only): this fork uses ONLY the "wid owns
    # 16 BT rows, all V" split-M warp partition -- the SAME warp scheme as the
    # hand-tuned HIP/C++ K5 kernel (BT split-M in GEMM1, K split across waves in
    # GEMM2, V not split across warps) with the SAME 16x16x16 bf16 MFMA. The
    # alternative OPT-VWARP layout has been removed entirely from this fork.

    # HIP-ALIGN 2a: w panels (w_panel0/1), one [BT][64] panel per 64-K block,
    # written with HIP's ``w_panel_swizzle`` (bank-conflict-free) and read by
    # GEMM1 via ``load_a_w_fragment_swizzled`` (plain b64, contiguous 16-K).
    LDS_WP_PANEL_ELEMS = BT * 64
    LDS_WP_ELEMS = NUM_K_BLOCKS * LDS_WP_PANEL_ELEMS
    LDS_WP_BYTES = LDS_WP_ELEMS * 2

    # HIP-ALIGN 2b: k in K-major shared2 panels (one per 64-K block); GEMM2
    # reads the MFMA A frag via load_shared2 (contiguous BT) -- eliminates
    # ds_read_tr16. cell = 4 consecutive tokens (BT): BT/4=16 bt_groups x 64
    # K_out cols x 4. NOTE: HIP's exact rotating-pair bank-swizzle is a
    # micro-opt and is NOT bit-replicated; the K-major panel + plain b64
    # fragment read match HIP's access pattern (K-major, no HW transpose).
    # k K-major shared2 panel row stride = 64 (K_out cols), no padding -- matches
    # HIP's compact k_panel0[64*BT]. (A padded stride was tried to break the
    # GEMM2 k-read bank conflict but a trace showed the bank conflict is NOT the
    # bottleneck -- the LDS stall is dominated by op count/scheduling -- so the
    # pad was reverted.)
    LDS_KP_ROW_STRIDE = 64
    LDS_KP_PANEL_ELEMS = (BT // 4) * LDS_KP_ROW_STRIDE * 4
    LDS_KP_ELEMS = NUM_K_BLOCKS * LDS_KP_PANEL_ELEMS
    LDS_KP_BYTES = LDS_KP_ELEMS * 2

    # HIP-ALIGN 3: gated v_new in a shared2 panel (like h / HIP gated_v_panel);
    # GEMM2 B read via load_shared2 (contiguous BT, matches the k A frag).
    LDS_GV_ELEMS = (BT // 4) * BV * 4  # 16 bt_groups x BV cols x 4 BT
    LDS_GV_BYTES = LDS_GV_ELEMS * 2

    # HIP-ALIGN phase 1b: h_state panels (shared2 layout) for the GEMM1 B
    # operand. One [row_block][V] panel per 64-K block (like HIP's
    # h_state_panel0/1); GEMM1 reads the MFMA B fragment with a plain b64 load
    # (load_b_shared2) instead of the hardware-transpose ds_read_tr16. Each
    # cell holds 4 K (a k_group): 64/4=16 row_blocks x BV cols x 4 K.
    LDS_HP_PANEL_ELEMS = (64 // 4) * BV * 4  # = 64 * BV per 64-K block
    LDS_HP_ELEMS = NUM_K_BLOCKS * LDS_HP_PANEL_ELEMS
    LDS_HP_BYTES = LDS_HP_ELEMS * 2

    # HIP-ALIGN phase 1a: [V][K] transpose buffer for the h snapshot HBM store.
    # h_accs (f32) is written here in V-major / K-innermost order so 8 adjacent
    # K land contiguously -> a single b128 (ds_read_b128 + buffer_store b128),
    # matching HIP's ``h_transpose_buf`` + ``coalesced_vk_store_from_transpose``
    # (fewer store instructions than the per-element bf16 readout). K-contiguous
    # (no pad) so the 8-wide vec load/store stays 16 B aligned.
    LDS_HT_STRIDE = K
    LDS_HT_ELEMS = BV * LDS_HT_STRIDE
    LDS_HT_BYTES = LDS_HT_ELEMS * 2

    # Bump revision so the FlyDSL JIT disk cache (~/.flydsl/cache/) invalidates
    # on revision change (port of FlyDSL commit d4643e0e).
    _K5_KERNEL_REVISION = 110  # LDS-eff: lds_ht transpose-buffer store 4x scalar -> 1 b64 (cut ds_write op count)

    GPU_ARCH = get_rocm_arch()
    allocator = SmemAllocator(
        None,
        arch=GPU_ARCH,
        global_sym_name=f"gdn_h_mfma16_hip_smem_v{_K5_KERNEL_REVISION}",
    )
    lds_wp_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_wp_offset + LDS_WP_BYTES
    lds_kp_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_kp_offset + LDS_KP_BYTES
    lds_gv_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_gv_offset + LDS_GV_BYTES
    # HIP-ALIGN 1b: h_state panels (GEMM1 B operand, shared2 layout).
    lds_hp_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_hp_offset + LDS_HP_BYTES
    # HIP-ALIGN phase 1a: [V][K] transpose buffer for the b128 h store.
    lds_ht_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_ht_offset + LDS_HT_BYTES

    # Cooperative load parameters
    LOAD_VEC_WIDTH = 8  # 8 bf16 = 16 bytes = buffer_load_dwordx4
    THREADS_PER_ROW_64 = 64 // LOAD_VEC_WIDTH  # 8
    ROWS_PER_BATCH_64 = BLOCK_THREADS // THREADS_PER_ROW_64  # 32
    NUM_LOAD_BATCHES_64 = BT // ROWS_PER_BATCH_64  # 2

    # ---- OPT-VC: precompute the GEMM1 prefetch interleaving schedule.
    # All quantities here depend ONLY on compile-time constants
    # (K, BV, USE_G, USE_GK) and live in the outer compile_*-function
    # scope so they are pure Python ints/lists -- the FlyDSL AST rewriter
    # only touches the @flyc.kernel body below, so any control flow here
    # is safe to mix as ordinary Python.
    # OPT-VC enablement gate: only spread prefetch into GEMM1 when N_REPEAT
    # == 1 (i.e. BV == WMMA_N == 16). When OPT_VC_ENABLED is False (BV>=32),
    # emit all g/gk/u prefetch in a BATCH BEFORE GEMM1 starts (via
    # PROLOGUE_EMITTER_CT), exactly matching the pre-OPT-VC (rev5) layout --
    # this leaves the full GEMM1 MFMA chain to overlap the HBM latency.
    # An earlier attempt (rev21) routed disabled-BV prefetch to the GEMM1
    # tail (TAIL_EMITTER_CT), which empirically lost 9-14% on BV>=32 shapes
    # because the prefetched values had no MFMA to hide behind before being
    # consumed by the gating / vn = u - bv computation.
    K_STEPS_PER_BLOCK = 64 // WMMA_K
    OPT_VC_ENABLED = N_REPEAT == 1
    # OPT-W is gated together with OPT-VC. On BV>=32 (N_REPEAT>=2) the GEMM2
    # inner loop is also thin enough that interleaving w_next vec_loads into
    # it causes the SIMD's single VMEM port to bottleneck on certain varlen
    # shapes. Disabling the interleave on BV>=32 falls back to the rev5-style
    # batched issue right before GEMM2, where the full MFMA chain hides the
    # HBM latency.
    OPT_W_ENABLED = N_REPEAT == 1
    NUM_INNER_SLOTS = NUM_K_BLOCKS * K_STEPS_PER_BLOCK * N_REPEAT
    NUM_GK_LOADS_CT = (NUM_K_BLOCKS * 4) if USE_GK else 0
    # g_last + g_row are batched through the emitter queue (5 loads).
    NUM_G_LOADS_CT = 5 if USE_G else 0
    NUM_U_LOADS_CT = N_REPEAT * 4
    _NUM_U_QUEUE_CT = NUM_U_LOADS_CT
    NUM_EXTRA_LOADS_CT = NUM_GK_LOADS_CT + NUM_G_LOADS_CT + _NUM_U_QUEUE_CT
    if OPT_VC_ENABLED and NUM_INNER_SLOTS > 0 and NUM_EXTRA_LOADS_CT > 0:
        EXTRAS_PER_SLOT_CT = (
            NUM_EXTRA_LOADS_CT + NUM_INNER_SLOTS - 1
        ) // NUM_INNER_SLOTS
    else:
        EXTRAS_PER_SLOT_CT = 0
    # Map each emitter idx (0..NUM_EXTRA_LOADS_CT-1) to one of three buckets:
    #   * SLOT_ASSIGN_CT[slot_idx] -- emitted inside GEMM1 at (kb,ks,nr) slot
    #     (used when OPT_VC_ENABLED is True, BV=16 path)
    #   * PROLOGUE_EMITTER_CT     -- emitted right BEFORE GEMM1 main loop
    #     (used when OPT_VC_ENABLED is False, BV>=32 path; matches rev5)
    #   * TAIL_EMITTER_CT         -- emitted AFTER GEMM1 (kept as future-
    #     facing safety net; not used by the current schedule).
    SLOT_ASSIGN_CT: list[list[int]] = [[] for _ in range(NUM_INNER_SLOTS)]
    PROLOGUE_EMITTER_CT: list[int] = []
    TAIL_EMITTER_CT: list[int] = []
    for _e_idx in range(NUM_EXTRA_LOADS_CT):
        if OPT_VC_ENABLED and NUM_INNER_SLOTS > 0:
            _slot = min(_e_idx // max(EXTRAS_PER_SLOT_CT, 1), NUM_INNER_SLOTS - 1)
            SLOT_ASSIGN_CT[_slot].append(_e_idx)
        else:
            PROLOGUE_EMITTER_CT.append(_e_idx)

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
        T_val: fx.Int32,
        T_flat: fx.Int32,
        N_val: fx.Int32,
    ):
        i_v = fx.block_idx.x
        i_nh = fx.block_idx.y
        i_n = i_nh // fx.Int32(H)
        i_h = i_nh % fx.Int32(H)

        tid = fx.thread_idx.x
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

        # -- LDS views --
        lds_base_ptr = allocator.get_base()

        # w panels (bf16, HIP w_panel swizzle) -- separate from k
        lds_wp_ptr = SmemPtr(lds_base_ptr, lds_wp_offset, T.bf16, shape=(LDS_WP_ELEMS,))
        lds_wp = STensor(lds_wp_ptr, dtype=T.bf16, shape=(LDS_WP_ELEMS,))
        lds_wp_memref = lds_wp_ptr.get()

        # k panels (bf16, K-major shared2) -- separate from w
        lds_kp_ptr = SmemPtr(lds_base_ptr, lds_kp_offset, T.bf16, shape=(LDS_KP_ELEMS,))
        lds_kp = STensor(lds_kp_ptr, dtype=T.bf16, shape=(LDS_KP_ELEMS,))
        lds_kp_memref = lds_kp_ptr.get()

        # gated v_new (bf16, shared2)
        lds_gv_ptr = SmemPtr(lds_base_ptr, lds_gv_offset, T.bf16, shape=(LDS_GV_ELEMS,))
        lds_gv = STensor(lds_gv_ptr, dtype=T.bf16, shape=(LDS_GV_ELEMS,))
        lds_gv_memref = lds_gv_ptr.get()

        # h_state panels (shared2 layout) for the GEMM1 B operand
        lds_hp_ptr = SmemPtr(lds_base_ptr, lds_hp_offset, T.bf16, shape=(LDS_HP_ELEMS,))
        lds_hp = STensor(lds_hp_ptr, dtype=T.bf16, shape=(LDS_HP_ELEMS,))
        lds_hp_memref = lds_hp_ptr.get()

        # h snapshot transpose buffer [V][K] (bf16) for the b128 HBM store
        lds_ht_ptr = SmemPtr(lds_base_ptr, lds_ht_offset, T.bf16, shape=(LDS_HT_ELEMS,))
        lds_ht = STensor(lds_ht_ptr, dtype=T.bf16, shape=(LDS_HT_ELEMS,))
        lds_ht_memref = lds_ht_ptr.get()

        # HIP-ALIGN 1b: plain b64 read of a shared2 panel cell (GEMM1 B frag).
        v4bf16_hp_type = T.vec(4, T.bf16)

        def _lds_read_hp_bf16x4(elem_idx):
            return vector.load_op(v4bf16_hp_type, lds_hp_memref, [fx.Index(elem_idx)])

        # HIP-ALIGN 2b/3: plain b64 reads of the k / gated_v shared2 panels.
        def _lds_read_kp_bf16x4(elem_idx):
            return vector.load_op(v4bf16_hp_type, lds_kp_memref, [fx.Index(elem_idx)])

        def _lds_read_gv_bf16x4(elem_idx):
            return vector.load_op(v4bf16_hp_type, lds_gv_memref, [fx.Index(elem_idx)])

        # -- Cooperative load decomposition --
        load_row_in_batch = tid // fx.Int32(THREADS_PER_ROW_64)
        load_col_base = (tid % fx.Int32(THREADS_PER_ROW_64)) * fx.Int32(LOAD_VEC_WIDTH)

        # -- LDS vector read helpers (generates ds_read_b128 for 8xbf16) --
        v8bf16_type = T.vec(8, T.bf16)

        # HIP-ALIGN 2a: w_panel swizzle (returns ELEMENT offset within a panel).
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

        v4bf16_w_type = T.vec(4, T.bf16)

        # HIP-ALIGN 2a: load_a_w_fragment_swizzled (contiguous 16-K A frag).
        # Caller passes row = row_base + lane&15 and k0 = k_base + (lane>>4)*4;
        # the ^4-elem (^8-byte) toggle picks the low/high 4 within an 8-group.
        def _load_a_w_swizzled(panel_base_elems, row, k0):
            col_base = k0 & fx.Int32(~7)
            elem = _w_panel_swz_elems(row, col_base) ^ (k0 & fx.Int32(4))
            return vector.load_op(
                v4bf16_w_type,
                lds_wp_memref,
                [fx.Index(panel_base_elems + elem)],
            )

        # -- Prologue: compute bos, T_local, NT, boh --
        if const_expr(IS_VARLEN):
            bos = cu_[fx.Index(i_n)]
            eos = cu_[fx.Index(i_n) + fx.Index(1)]
            T_local = eos - bos
            NT = (T_local + fx.Int32(BT - 1)) // fx.Int32(BT)
            boh = co_[fx.Index(i_n)]
        else:
            bos = i_n * T_val
            T_local = T_val
            NT = (T_local + fx.Int32(BT - 1)) // fx.Int32(BT)
            boh = i_n * NT

        # -- Base pointer offsets (element counts) --
        # h: [B, NT, H, V, K] (VK) -- base = (boh*H + i_h) * V * K
        h_base = (boh * fx.Int32(H) + i_h) * fx.Int32(V * K)
        stride_h = fx.Int32(H * V * K)

        # k: [B, T, Hg, K] -- base = (bos*Hg + i_h//(H//Hg)) * K
        gqa_ratio = H // Hg
        k_base = (bos * fx.Int32(Hg) + i_h // fx.Int32(gqa_ratio)) * fx.Int32(K)
        stride_k = fx.Int32(Hg * K)

        if const_expr(WU_CONTIGUOUS):
            if const_expr(IS_VARLEN):
                v_base = (i_h * T_flat + bos) * fx.Int32(V)
                w_base = (i_h * T_flat + bos) * fx.Int32(K)
            else:
                v_base = ((i_n * fx.Int32(H) + i_h) * T_flat) * fx.Int32(V)
                w_base = ((i_n * fx.Int32(H) + i_h) * T_flat) * fx.Int32(K)
            stride_v = fx.Int32(V)
            stride_w = fx.Int32(K)
        else:
            v_base = (bos * fx.Int32(H) + i_h) * fx.Int32(V)
            w_base = (bos * fx.Int32(H) + i_h) * fx.Int32(K)
            stride_v = fx.Int32(H * V)
            stride_w = fx.Int32(H * K)

        if const_expr(IS_VARLEN):
            vn_base = (i_h * T_flat + bos) * fx.Int32(V)
        else:
            vn_base = ((i_n * fx.Int32(H) + i_h) * T_flat) * fx.Int32(V)

        if const_expr(USE_INITIAL_STATE):
            h0_base = i_nh * fx.Int32(V * K)
        if const_expr(STORE_FINAL_STATE):
            ht_base = i_nh * fx.Int32(V * K)

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
                    h0_off_base = h0_base + h0_col * fx.Int32(K) + h0_row_base
                    loaded_vec = h0_.vec_load((fx.Index(h0_off_base),), 4)
                    if const_expr(STATE_DTYPE_BF16):
                        loaded_vec = loaded_vec.extf(T.f32x4)
                    acc_idx = kb * N_REPEAT + slot
                    h_accs[acc_idx] = h_accs[acc_idx] + loaded_vec

        # -- Software-pipelined main chunk loop --

        # -- Prologue: pre-load first chunk's w data --
        i_t0_i32 = fx.Int32(0)
        w_prefetch_init = []
        for kb in range_constexpr(NUM_K_BLOCKS):
            for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                row = fx.Int32(batch * ROWS_PER_BATCH_64) + load_row_in_batch
                abs_row = i_t0_i32 * fx.Int32(BT) + row
                safe_row = (abs_row < T_local).select(abs_row, fx.Int32(0))
                g_off = w_base + safe_row * stride_w + fx.Int32(kb * 64) + load_col_base
                w_prefetch_init.append(w_.vec_load((fx.Index(g_off),), LOAD_VEC_WIDTH))

        init_state = [_to_raw(v) for v in h_accs] + [
            _to_raw(v) for v in w_prefetch_init
        ]
        c_zero = fx.Index(0)
        c_one = fx.Index(1)
        nt_idx = fx.Index(NT)

        for i_t, state in range(c_zero, nt_idx, c_one, init=init_state):
            h_accs_in = list(state[:NUM_H_ACCS])
            w_prefetch_all = list(state[NUM_H_ACCS:])
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
                    k_tile_base = fx.Int32(kb * 64) + wid * fx.Int32(16)

                    # HIP-ALIGN 1b: write the h_state panel cell (shared2). This
                    # lane owns k_group (row_block = wid*4+lane_m_base) at V-col
                    # = nr*16+lane_n; 4 warps together fill all 16 row_blocks.
                    hp_row_block = wid * fx.Int32(4) + lane_m_base
                    hp_cell = fx.Int32(kb * LDS_HP_PANEL_ELEMS) + (
                        hp_row_block * fx.Int32(BV) + hp_col
                    ) * fx.Int32(4)
                    lds_hp.vec_store(
                        (fx.Index(hp_cell),),
                        acc_val.truncf(T.vec(4, T.bf16)),
                        4,
                    )

                    # HIP-ALIGN 1a: [V][K] transpose buffer for the b128 store.
                    # The 4 elem_i are 4 consecutive K (k_tile_base+lm*4+[0..3])
                    # so write them as ONE b64 (ds_write_b64) instead of 4 scalar
                    # ds_write_b16 -- cuts the transpose-buffer store op count 4x.
                    ht_base_idx = (
                        hp_col * fx.Int32(LDS_HT_STRIDE)
                        + k_tile_base
                        + lane_m_base * fx.Int32(4)
                    )
                    lds_ht.vec_store(
                        (fx.Index(ht_base_idx),),
                        acc_val.truncf(T.vec(4, T.bf16)),
                        4,
                    )

            gpu.barrier()

            # HIP-ALIGN 1a: b128 h store. Each thread reads 8 contiguous K from
            # lds_ht[v][k8:k8+8] (ds_read_b128) and writes them to HBM
            # h[chunk][v][k8:k8+8] (buffer_store b128). K_VECS = K/8 vectors per
            # v row; adjacent threads cover adjacent (v, k8) -> coalesced wide
            # stores (1/8 the store instructions of the per-element readout).
            K_VECS = K // LOAD_VEC_WIDTH
            NUM_HT_VECS = BV * K_VECS
            for vbase in range_constexpr(0, NUM_HT_VECS, BLOCK_THREADS):
                vec_idx = fx.Int32(vbase) + tid
                k8 = (vec_idx % fx.Int32(K_VECS)) * fx.Int32(LOAD_VEC_WIDTH)
                v_loc = vec_idx // fx.Int32(K_VECS)
                lds_ht_read = v_loc * fx.Int32(LDS_HT_STRIDE) + k8
                vec8 = vector.load_op(
                    v8bf16_type, lds_ht_memref, [fx.Index(lds_ht_read)]
                )
                v_global = i_v * fx.Int32(BV) + v_loc
                h_off = (
                    h_base + i_t_i32 * stride_h + v_global * fx.Int32(K) + k8
                )
                h_.vec_store((fx.Index(h_off),), vec8, LOAD_VEC_WIDTH)

            # HIP-ALIGN 2a: write prefetched w to the swizzled panels.
            # w_prefetch_all is ordered [kb][batch] = b16x8 (8 K) for
            # row = batch*32 + tid//8, col_base = (tid%8)*8. Split into
            # low4/high4 and store at the w_panel swizzle base and base^4 (elem)
            # in panel kb -- matching load_a_w_fragment_swizzled reads below.
            for kb in range_constexpr(NUM_K_BLOCKS):
                wp_panel_base = fx.Int32(kb * LDS_WP_PANEL_ELEMS)
                for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                    row = fx.Int32(batch * ROWS_PER_BATCH_64) + load_row_in_batch
                    swz = wp_panel_base + _w_panel_swz_elems(row, load_col_base)
                    wvec = w_prefetch_all[kb * NUM_LOAD_BATCHES_64 + batch]
                    lds_wp.vec_store(
                        (fx.Index(swz),),
                        wvec.shuffle(wvec, [0, 1, 2, 3]),
                        4,
                    )
                    lds_wp.vec_store(
                        (fx.Index(swz ^ fx.Int32(4)),),
                        wvec.shuffle(wvec, [4, 5, 6, 7]),
                        4,
                    )

            gpu.barrier()

            # -- 2. Delta correction: b_v = w @ h, then v_new = u - b_v --
            # OPT-K: k prefetch is interleaved into the GEMM1 main loop below
            # so the 4 buffer_load_dwordx4 are issued one per (mfma_kb, ks)
            # iteration and their HBM latency is hidden by the MFMA chain.
            # Here we only precompute the per-batch HBM byte offsets and the
            # LDS write offsets; the actual vec_load is emitted inside the
            # GEMM1 loop.
            k_prefetch_off = []
            for kb in range_constexpr(NUM_K_BLOCKS):
                for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                    row = fx.Int32(batch * ROWS_PER_BATCH_64) + load_row_in_batch
                    abs_row = i_t_i32 * fx.Int32(BT) + row
                    safe_row = (abs_row < T_local).select(abs_row, fx.Int32(0))
                    k_off = (
                        k_base + safe_row * stride_k + fx.Int32(kb * 64) + load_col_base
                    )
                    k_prefetch_off.append(k_off)

            # k_prefetch results are filled inside the GEMM1 main loop below.
            k_prefetch = [None] * len(k_prefetch_off)

            # Compute last_idx for the current chunk. The offset precompute
            # below is intentionally unconditional, even for ungated kernels.
            next_chunk_end = (i_t_i32 + fx.Int32(1)) * fx.Int32(BT)
            last_idx_raw = (next_chunk_end < T_local).select(
                next_chunk_end, T_local
            ) - fx.Int32(1)

            # OPT-VC (vmcnt-spread): precompute HBM offsets for g/gk/u prefetch
            # but DEFER the actual vec_load/scalar load until interleaved into
            # the GEMM1 main loop below. Hotspot report (35B/TP2/60K) shows the
            # original "load-all-before-GEMM1" pattern piles up ~17 in-flight
            # VMEM ops and triggers vmcnt(7) reverse-pressure (34% of total
            # stall). Spreading them across the MFMA chain drops the steady-
            # state vmcnt threshold to ~3-4 and unblocks GEMM1 entry.
            # OPT-VC: precompute offsets for g/gk/u prefetch but defer the
            # actual vec_load until interleaved into GEMM1 below. All Python
            # bookkeeping (slot_assignments, EXTRAS_PER_SLOT, etc.) was done
            # at compile-time in the enclosing compile_chunk_gated_delta_h
            # scope to avoid AST-rewriter interference.
            # G layout: head-major [B, H, T_flat] (matches Triton VK / HIP).
            # Each head's gate values are contiguous in HBM (stride=1):
            #     g[i_h * T_flat + (bos + row)]
            g_last_off = i_h * T_flat + (bos + last_idx_raw)
            g_row_off_list = []
            g_row_in_bounds = []
            for elem_i in range_constexpr(4):
                abs_row = (
                    i_t_i32 * fx.Int32(BT)
                    + wid * fx.Int32(16)
                    + lane_m_base * fx.Int32(4)
                    + fx.Int32(elem_i)
                )
                in_bounds = abs_row < T_local
                safe_row = in_bounds.select(abs_row, fx.Int32(0))
                g_row_off = i_h * T_flat + (bos + safe_row)
                g_row_off_list.append(g_row_off)
                g_row_in_bounds.append(in_bounds)
            g_last_prefetch_cell = [None]
            g_row_prefetch = [None] * 4

            gk_chunk_base = (bos + last_idx_raw) * fx.Int32(H * K) + i_h * fx.Int32(K)
            gk_off_flat = []
            for kb in range_constexpr(NUM_K_BLOCKS):
                for elem_i in range_constexpr(4):
                    global_k = (
                        fx.Int32(kb * 64)
                        + wid * fx.Int32(16)
                        + lane_m_base * fx.Int32(4)
                        + fx.Int32(elem_i)
                    )
                    gk_off_flat.append(gk_chunk_base + global_k)
            gk_raw_prefetch = [None] * NUM_GK_LOADS_CT

            u_off_list = []
            for nr in range_constexpr(N_REPEAT):
                u_col = i_v * fx.Int32(BV) + fx.Int32(nr * 16) + lane_n
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
                    u_off = v_base + safe_u_row * stride_v + u_col
                    u_off_list.append(u_off)
            u_prefetch = [None] * NUM_U_LOADS_CT

            # bv_accs: indexed by nr (V-tile). 4 accumulators.
            bv_accs = []
            for _i in range_constexpr(N_REPEAT):
                bv_accs.append(fx.full(4, 0.0, fx.Float32))

            K_STEPS_PER_BLOCK = 64 // WMMA_K
            NUM_K_LOADS = NUM_K_BLOCKS * NUM_LOAD_BATCHES_64

            # OPT-VC: Build a flat queue of "extra" prefetches to inject one-
            # per-(nr-step) into GEMM1 so that g_last/g_row/gk/u VMEM loads are
            # spread across the entire MFMA chain instead of bursting into a
            # single vmcnt(7) wall just before GEMM1. Order matters: items at
            # the front issue earliest -> longest HBM latency hiding window;
            # items at the back issue latest. Place gk first (it also needs a
            # follow-up _fast_exp ALU op so earlier issue = more ALU overlap),
            # then g_last / g_row (short scalar loads, ALU follow-up), then u
            # (consumed right after GEMM1 with no ALU between).
            # OPT-VC: emitter factories return zero-arg lambdas that bind all
            # captured Python values via DEFAULT ARGUMENTS (not via implicit
            # closures, which FlyDSL's AST rewriter does not preserve across
            # its exec()-based function regeneration). The lambdas themselves
            # are AST.Lambda nodes which the rewriter never visits, so their
            # bodies execute unchanged at trace time.
            _gk_local = gk_ if USE_GK else g_  # safe placeholder when USE_GK=False

            def _make_emit_g_last(_g=g_, _off=g_last_off, _cell=g_last_prefetch_cell):
                return lambda: _cell.__setitem__(0, _g[fx.Index(_off)])

            def _make_emit_g_row(
                idx,
                _g=g_,
                _offs=g_row_off_list,
                _bnds=g_row_in_bounds,
                _arr=g_row_prefetch,
            ):
                _off_i = _offs[idx]
                _bnd_i = _bnds[idx]
                return lambda: _arr.__setitem__(idx, (_g[fx.Index(_off_i)], _bnd_i))

            def _make_emit_gk(
                idx, _gk=_gk_local, _offs=gk_off_flat, _arr=gk_raw_prefetch
            ):
                _off_i = _offs[idx]
                return lambda: _arr.__setitem__(idx, _gk[fx.Index(_off_i)])

            def _make_emit_u(idx, _v=v_, _offs=u_off_list, _arr=u_prefetch):
                _off_i = _offs[idx]
                return lambda: _arr.__setitem__(idx, _v[fx.Index(_off_i)])

            # OPT-VC: assemble emitter queue using plain Python ``for`` loops
            # (not ``range_constexpr``). These emitter objects are pure Python
            # callables built at trace time -- the actual MLIR ops are emitted
            # only when the emitter is invoked inside the GEMM1 loop below.
            # Avoid ``range_constexpr`` here because FlyDSL's AST rewriter
            # rebinds local names captured inside ``range_constexpr`` bodies
            # in ways that can hide subsequent plain-Python locals (e.g.
            # ``EXTRAS_PER_SLOT`` derived from the queue length).
            extra_load_emitters = []
            if const_expr(USE_GK):
                for i in range_constexpr(NUM_GK_LOADS_CT):
                    extra_load_emitters.append(_make_emit_gk(i))
            if const_expr(USE_G):
                extra_load_emitters.append(_make_emit_g_last())
                for i in range_constexpr(4):
                    extra_load_emitters.append(_make_emit_g_row(i))
            for i in range_constexpr(NUM_U_LOADS_CT):
                extra_load_emitters.append(_make_emit_u(i))

            # OPT-VC: the prefetch slot-assignment schedule lives in the
            # outer compile_chunk_gated_delta_h scope as SLOT_ASSIGN_CT /
            # PROLOGUE_EMITTER_CT / TAIL_EMITTER_CT (pure Python lists) so
            # we don't run any Python control flow here that the AST
            # rewriter would clobber. ``extra_load_emitters`` is populated
            # above and is index-compatible with the static schedule.
            #
            # OPT-VC prologue path (BV>=32): when OPT_VC_ENABLED is False
            # the schedule routes every emitter into PROLOGUE_EMITTER_CT,
            # so the entire batch of g/gk/u prefetch is issued HERE -- right
            # before the GEMM1 main loop begins. This matches the original
            # pre-OPT-VC (rev5) placement and lets the full MFMA chain hide
            # the HBM latency of these scalar / dwordx4 loads.
            for _eidx in PROLOGUE_EMITTER_CT:
                extra_load_emitters[_eidx]()

            for kb in range_constexpr(NUM_K_BLOCKS):
                for ks in range_constexpr(K_STEPS_PER_BLOCK):
                    # OPT-K: issue one k_prefetch vec_load per (kb, ks) slot
                    # to spread the 4 buffer_load_dwordx4 across the MFMA
                    # chain so HBM latency is hidden by the MFMA chain.
                    mfma_slot = kb * K_STEPS_PER_BLOCK + ks
                    if mfma_slot < NUM_K_LOADS:
                        k_prefetch[mfma_slot] = k_.vec_load(
                            (fx.Index(k_prefetch_off[mfma_slot]),),
                            LOAD_VEC_WIDTH,
                        )

                    # HIP-ALIGN 2a: A = load_a_w_fragment_swizzled. Two
                    # contiguous 16-K tiles (lo=[ks*32,+16), hi=[+16,+32));
                    # k0 = tile_base + lane_m_base*4 (standard A layout).
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
                        slot_idx = (
                            kb * (K_STEPS_PER_BLOCK * N_REPEAT)
                            + ks * N_REPEAT
                            + nr
                        )
                        for _eidx in SLOT_ASSIGN_CT[slot_idx]:
                            extra_load_emitters[_eidx]()

                        # HIP-ALIGN 1b/2a: B = load_b_shared2(panel_kb, k_base,
                        # lane, nr*16), CONTIGUOUS to match the (now HIP-style)
                        # contiguous w A read. col = (lane&15)+nr*16;
                        # row_block = (k_base>>2)+(lane>>4). lo tile K in
                        # [ks*32, ks*32+16), hi in [ks*32+16, ks*32+32).
                        hp_base = fx.Int32(kb * LDS_HP_PANEL_ELEMS)
                        hp_col_b = fx.Int32(nr * 16) + lane_n
                        rb_lo = fx.Int32((ks * WMMA_K) >> 2) + lane_m_base
                        rb_hi = fx.Int32(((ks * WMMA_K) + 16) >> 2) + lane_m_base
                        idx_lo = hp_base + (
                            rb_lo * fx.Int32(BV) + hp_col_b
                        ) * fx.Int32(4)
                        idx_hi = hp_base + (
                            rb_hi * fx.Int32(BV) + hp_col_b
                        ) * fx.Int32(4)
                        b_frag_lo = _lds_read_hp_bf16x4(idx_lo)
                        b_frag_hi = _lds_read_hp_bf16x4(idx_hi)

                        bv_accs[nr] = _mfma_bf16_16x16x16(
                            a_frag_lo, b_frag_lo, bv_accs[nr]
                        )
                        bv_accs[nr] = _mfma_bf16_16x16x16(
                            a_frag_hi, b_frag_hi, bv_accs[nr]
                        )

            # OPT-VC: tail-emit any extras that did not fit (rare path).
            for _eidx in TAIL_EMITTER_CT:
                extra_load_emitters[_eidx]()

            # OPT-VC: apply _fast_exp on the gk raw loads to build the
            # gk_last_prefetch[kb][elem_i] structure expected downstream.
            if const_expr(USE_GK):
                gk_last_prefetch = []
                for kb in range_constexpr(NUM_K_BLOCKS):
                    kb_elems = []
                    for elem_i in range_constexpr(4):
                        kb_elems.append(_fast_exp(gk_raw_prefetch[kb * 4 + elem_i]))
                    gk_last_prefetch.append(kb_elems)

            # v_new = u - b_v (u values already prefetched). Indexed by nr
            # (V-tile).
            vn_frags = []
            for idx in range_constexpr(N_REPEAT):
                bv_val = bv_accs[idx]
                u_f32_elems = []
                for elem_i in range_constexpr(4):
                    # u_prefetch entries are ordered (idx outer, elem_i
                    # inner), so the flat index is the same expression.
                    u_raw = u_prefetch[idx * 4 + elem_i]
                    u_bf16 = fx.BFloat16(u_raw)
                    u_f32_elems.append(u_bf16.to(fx.Float32))
                u_f32 = vector.from_elements(T.f32x4, u_f32_elems)

                vn_frags.append(u_f32 - bv_val)

            # -- 2b. Store v_new (pre-gating) for output --
            if const_expr(SAVE_NEW_VALUE):
                # Closure wrapper to hide ``vn_`` from FlyDSL 0.1.5+
                # ``ReplaceIfWithDispatch`` ast rewriter: it scans
                # subscript-store and ``obj.method()`` calls inside dynamic
                # ``if`` bodies and demands MLIR-Value state for any name
                # written to / invoked on.  ``vn_`` is a GTensor (HBM tensor
                # wrapper), not an MLIR Value.  Wrapping the store in a bare
                # function call (ast.Name, not ast.Attribute / Subscript)
                # makes the analyzer skip it.
                def _emit_vn_store(off, value):
                    vn_[fx.Index(off)] = value

                for idx in range_constexpr(N_REPEAT):
                    vn_val = vn_frags[idx]
                    vn_col = i_v * fx.Int32(BV) + fx.Int32(idx * 16) + lane_n
                    bt_tile_base = wid * fx.Int32(16)
                    for elem_i in range_constexpr(4):
                        vn_bt_row = (
                            i_t_i32 * fx.Int32(BT)
                            + bt_tile_base
                            + lane_m_base * fx.Int32(4)
                            + fx.Int32(elem_i)
                        )
                        if (vn_bt_row < T_local).ir_value():
                            f32_v = vn_val[elem_i]
                            bf16_v = f32_v.to(fx.BFloat16)
                            vn_off = vn_base + vn_bt_row * fx.Int32(V) + vn_col
                            _emit_vn_store(vn_off, bf16_v)

            # -- 3. Gating -- g values prefetched before MFMA --
            if const_expr(USE_G):
                g_last = g_last_prefetch_cell[0]
                exp_g_last = _fast_exp(g_last)

                # Build the 4-lane gate vector via a single from_elements.
                gate_elems = []
                for elem_i in range_constexpr(4):
                    g_row, in_bounds = g_row_prefetch[elem_i]
                    gate = _fast_exp(g_last - g_row)
                    gate_elems.append(in_bounds.select(gate, fx.Float32(0.0)))
                gate_vec = vector.from_elements(T.f32x4, gate_elems)

                for nr in range_constexpr(N_REPEAT):
                    vn_frags[nr] = vn_frags[nr] * gate_vec

                # Scalar broadcast multiply (no explicit f32x4 temp) to keep
                # VGPR pressure down -- helps reach the 3 waves/SIMD threshold.
                exp_g_last_s = fx.Float32(exp_g_last)

                for kb in range_constexpr(NUM_K_BLOCKS):
                    for nr in range_constexpr(N_REPEAT):
                        acc_idx = kb * N_REPEAT + nr
                        h_accs_in[acc_idx] = h_accs_in[acc_idx] * exp_g_last_s

            # Per-K decay: h[v, k] *= exp(gk_last[k]) at chunk end.
            # Each lane's v4f32 spans 4 different K positions (one per elem_i),
            # so we build a per-kb gate vector and multiply h_accs accordingly.
            if const_expr(USE_GK):
                for kb in range_constexpr(NUM_K_BLOCKS):
                    # Same simplification as gate_vec above: one
                    # from_elements instead of fx.full(0.0) + 4x insert.
                    gk_vec = vector.from_elements(
                        T.f32x4,
                        [gk_last_prefetch[kb][elem_i] for elem_i in range_constexpr(4)],
                    )
                    for nr in range_constexpr(N_REPEAT):
                        acc_idx = kb * N_REPEAT + nr
                        h_accs_in[acc_idx] = h_accs_in[acc_idx] * gk_vec

            # -- 4. State update: h += k^T @ v_new_gated --
            BT_STEPS = BT // WMMA_K

            # HIP-ALIGN 3: store gated v_new to the shared2 panel (like h): each
            # lane owns bt_group = wid*4+lane_m_base at V-col = nr*16+lane_n; the
            # f32x4 = 4 consecutive BT rows. Plain b64 store.
            for idx in range_constexpr(N_REPEAT):
                gv_col = fx.Int32(idx * 16) + lane_n
                gv_row_block = wid * fx.Int32(4) + lane_m_base
                gv_cell = (gv_row_block * fx.Int32(BV) + gv_col) * fx.Int32(4)
                lds_gv.vec_store(
                    (fx.Index(gv_cell),),
                    vn_frags[idx].truncf(T.vec(4, T.bf16)),
                    4,
                )

            # HIP-ALIGN 2b: transpose-scatter k into the K-major shared2 panels.
            # k_prefetch[kb][batch] = b16x8 (8 K_out) for token t = batch*32 +
            # tid//8, K_out = (tid%8)*8 + i. Write each to k_panel[kb] cell
            # (bt_group = t>>2, K_out) at slot (t&3) -> 4 consecutive tokens per
            # cell, matching the GEMM2 load_shared2 A read.
            for kb in range_constexpr(NUM_K_BLOCKS):
                kp_pbase = fx.Int32(kb * LDS_KP_PANEL_ELEMS)
                for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                    t_local = fx.Int32(batch * ROWS_PER_BATCH_64) + load_row_in_batch
                    bt_group = t_local >> fx.Int32(2)
                    t_slot = t_local & fx.Int32(3)
                    kvec = k_prefetch[kb * NUM_LOAD_BATCHES_64 + batch]
                    cell0 = (
                        kp_pbase
                        + (bt_group * fx.Int32(LDS_KP_ROW_STRIDE) + load_col_base)
                        * fx.Int32(4)
                        + t_slot
                    )
                    for i in range_constexpr(LOAD_VEC_WIDTH):
                        lds_kp[fx.Index(cell0 + fx.Int32(i * 4))] = kvec[i]

            gpu.barrier()

            # -- OPT-W: precompute NEXT iteration's w prefetch offsets only.
            # The actual buffer_load vec_load calls are interleaved into the
            # GEMM2 (k @ v_new) main loop below so the HBM latency of each
            # buffer_load_dwordx4 is hidden behind the MFMA dependency chain
            # (same idea as OPT-K for k). Without this, the 4 dwordx4 loads
            # all issue back-to-back before GEMM2 and pile up at vmcnt(7),
            # which is the #1 hotspot per ATT trace (~34% of total stall).
            next_i_t_i32 = i_t_i32 + fx.Int32(1)
            w_next_prefetch_off = []
            for kb in range_constexpr(NUM_K_BLOCKS):
                for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                    row = fx.Int32(batch * ROWS_PER_BATCH_64) + load_row_in_batch
                    abs_row = next_i_t_i32 * fx.Int32(BT) + row
                    safe_row = (abs_row < T_local).select(abs_row, fx.Int32(0))
                    g_off = (
                        w_base + safe_row * stride_w + fx.Int32(kb * 64) + load_col_base
                    )
                    w_next_prefetch_off.append(g_off)

            NUM_W_NEXT_LOADS = NUM_K_BLOCKS * NUM_LOAD_BATCHES_64
            w_next_prefetch = [None] * NUM_W_NEXT_LOADS

            # OPT-W prologue path (BV>=32): issue all w_next vec_loads as a
            # BATCH right before GEMM2 starts, matching the rev5 scheduling.
            # The interleaved per-(kb,bt_s) issue inside GEMM2 below is then
            # skipped. ``const_expr`` ensures the FlyDSL AST rewriter treats
            # this branch as a compile-time const (no dispatch wrapper).
            if const_expr(not OPT_W_ENABLED):
                for _i in range_constexpr(NUM_W_NEXT_LOADS):
                    w_next_prefetch[_i] = w_.vec_load(
                        (fx.Index(w_next_prefetch_off[_i]),), LOAD_VEC_WIDTH
                    )

            for kb in range_constexpr(NUM_K_BLOCKS):
                for bt_s in range_constexpr(BT_STEPS):
                    # OPT-W: issue one w-next vec_load per (kb, bt_s) slot.
                    # NUM_K_BLOCKS * BT_STEPS == NUM_W_NEXT_LOADS for the
                    # current (K=128, BT=64) config (4 == 4), so every slot
                    # gets exactly one load. Skipped when OPT_W_ENABLED is
                    # False (BV>=32) since the batch was already issued above.
                    w_slot = kb * BT_STEPS + bt_s
                    if const_expr(OPT_W_ENABLED):
                        if w_slot < NUM_W_NEXT_LOADS:
                            w_next_prefetch[w_slot] = w_.vec_load(
                                (fx.Index(w_next_prefetch_off[w_slot]),),
                                LOAD_VEC_WIDTH,
                            )

                    # HIP-ALIGN 2b: A = k load_shared2 (K-major panel). K_out
                    # tile = wid*16 (this warp's K rows); contiguous BT lo/hi.
                    kp_pbase = fx.Int32(kb * LDS_KP_PANEL_ELEMS)
                    k_col = fx.Int32(wid * 16) + lane_n
                    k_rb_lo = fx.Int32((bt_s * WMMA_K) >> 2) + lane_m_base
                    k_rb_hi = fx.Int32(((bt_s * WMMA_K) + 16) >> 2) + lane_m_base
                    k_a_lo = _lds_read_kp_bf16x4(
                        kp_pbase
                        + (k_rb_lo * fx.Int32(LDS_KP_ROW_STRIDE) + k_col) * fx.Int32(4)
                    )
                    k_a_hi = _lds_read_kp_bf16x4(
                        kp_pbase
                        + (k_rb_hi * fx.Int32(LDS_KP_ROW_STRIDE) + k_col) * fx.Int32(4)
                    )

                    for nr in range_constexpr(N_REPEAT):
                        # HIP-ALIGN 3: B = gated_v load_shared2 (contiguous BT,
                        # same bt_group as k A). col = nr*16+lane_n = V.
                        gv_col = fx.Int32(nr * 16) + lane_n
                        vn_b_lo = _lds_read_gv_bf16x4(
                            (k_rb_lo * fx.Int32(BV) + gv_col) * fx.Int32(4)
                        )
                        vn_b_hi = _lds_read_gv_bf16x4(
                            (k_rb_hi * fx.Int32(BV) + gv_col) * fx.Int32(4)
                        )

                        acc_idx = kb * N_REPEAT + nr
                        h_accs_in[acc_idx] = _mfma_bf16_16x16x16(
                            k_a_lo, vn_b_lo, h_accs_in[acc_idx]
                        )
                        h_accs_in[acc_idx] = _mfma_bf16_16x16x16(
                            k_a_hi, vn_b_hi, h_accs_in[acc_idx]
                        )

            # OPT-W: emit any remaining w_next loads that didn't fit into the
            # GEMM2 main loop (only possible if NUM_K_BLOCKS*BT_STEPS <
            # NUM_W_NEXT_LOADS for an exotic config). Const-expr loop, all
            # slots resolved at trace time.
            for i_wp in range_constexpr(NUM_W_NEXT_LOADS):
                if w_next_prefetch[i_wp] is None:
                    w_next_prefetch[i_wp] = w_.vec_load(
                        (fx.Index(w_next_prefetch_off[i_wp]),), LOAD_VEC_WIDTH
                    )

            results = yield [_to_raw(v) for v in h_accs_in] + [
                _to_raw(v) for v in w_next_prefetch
            ]

        h_accs_final = list(results[:NUM_H_ACCS])

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
                    ht_off_base = ht_base + ht_col * fx.Int32(K) + ht_row_base
                    if const_expr(STATE_DTYPE_BF16):
                        out_vec = acc_val.truncf(T.vec(4, T.bf16))
                    else:
                        out_vec = acc_val
                    ht_.vec_store((fx.Index(ht_off_base),), out_vec, 4)

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
        T_val: fx.Int32,
        T_flat: fx.Int32,
        N_val: fx.Int32,
        grid_v: fx.Int32,
        grid_nh: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

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
