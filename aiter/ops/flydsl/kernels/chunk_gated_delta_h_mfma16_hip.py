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


def _f32x4_to_bf16x4_rne_gfx950(vec_f32x4):
    """Round-to-nearest-even f32x4 -> bf16x4 via the gfx950 native convert.

    Mirrors HIP's ``float_to_bf16`` which uses the gfx950 native RNE convert
    ``v_cvt_pk_bf16_f32`` (``static_cast<__bf16>``), NOT the FlyDSL default
    ``arith.truncf`` (bit-truncation, differs from torch/HIP by ~1 ulp). Packs
    the 4 lanes with two ``cvt_pk_bf16_f32`` (each 2xf32 -> i32 holding 2xbf16),
    then bitcasts the 2xi32 back to a vector<4xbf16>. Returns a raw
    vector<4xbf16> ir.Value (drop-in for the previous ``.truncf(vec4 bf16)``).

    gfx950 (CDNA4) ONLY -- ``v_cvt_pk_bf16_f32`` does not exist on gfx942.
    """
    lo = rocdl.cvt_pk_bf16_f32(vec_f32x4[0], vec_f32x4[1])  # i32: [bf16(0), bf16(1)]
    hi = rocdl.cvt_pk_bf16_f32(vec_f32x4[2], vec_f32x4[3])  # i32: [bf16(2), bf16(3)]
    packed = vector.from_elements(T.vec(2, T.i32), [lo, hi])
    return vector.bitcast(T.vec(4, T.bf16), packed)


def _f32x4_to_bf16x4_rne_portable(vec_f32x4):
    """Round-to-nearest-even f32x4 -> bf16x4 without ``v_cvt_pk_bf16_f32``.

    Portable software RNE for architectures lacking the gfx950 native
    ``v_cvt_pk_bf16_f32`` convert (e.g. gfx942 / CDNA3). Emulates HIP's
    ``__float2bfloat16_rn`` software fallback with the standard "add a
    round-to-nearest-even bias, then keep the high 16 bits" integer
    sequence, applied lane-wise over the whole vector<4xf32>:

        x            = bitcast<i32>(f)
        rounding_bias = 0x7FFF + ((x >> 16) & 1)   # even-tie -> +0x7FFF, odd -> +0x8000
        bf16_bits    = (x + rounding_bias) >> 16

    This matches torch/HIP RNE (not FlyDSL's default ``truncf`` truncation).
    Inf/NaN survive: the bias never carries a finite value up to Inf, and a
    NaN keeps a non-zero mantissa. Returns a vector<4xbf16> ir.Value, a
    drop-in replacement for the gfx950 path.
    """
    i32x4 = T.vec(4, T.i32)
    x = vector.bitcast(i32x4, vec_f32x4)
    c16 = arith.constant_vector(16, i32x4)
    c1 = arith.constant_vector(1, i32x4)
    c7fff = arith.constant_vector(0x7FFF, i32x4)
    lsb = arith.andi(arith.shrui(x, c16), c1)
    rounding_bias = arith.addi(lsb, c7fff)
    rounded = arith.addi(x, rounding_bias)
    hi = arith.shrui(rounded, c16)
    hi16 = arith.trunci(T.vec(4, T.i16), hi)
    return vector.bitcast(T.vec(4, T.bf16), hi16)


def _f32x4_to_bf16x4_rne(vec_f32x4):
    """Arch-aware RNE f32x4 -> bf16x4.

    Uses the native ``v_cvt_pk_bf16_f32`` convert on gfx950 (CDNA4) and a
    portable integer software RNE everywhere else (gfx942 / CDNA3 has no
    ``v_cvt_pk_bf16_f32``). Called at trace time, so dispatching on
    ``get_rocm_arch()`` here selects the right lowering per compile.
    """
    if "gfx950" in get_rocm_arch():
        return _f32x4_to_bf16x4_rne_gfx950(vec_f32x4)
    return _f32x4_to_bf16x4_rne_portable(vec_f32x4)


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

    # HIP-ALIGN 2b: k in rotating-pair swizzled panels (one per 64-K block).
    # Mirrors HIP's k_panel0[64*BT] + k_panel_rotating_pair_addr_bytes layout
    # exactly: global load reads b128 (8 bf16) for adjacent token pairs (t0,t1),
    # then scatters b16x2 packed writes to swizzled LDS addresses; GEMM2 reads
    # the MFMA A frag via load_a_k_fragment_rotating (plain b64 from the
    # swizzled address, no ds_read_tr16). Panel size = 64 K-rows * BT tokens
    # = 64*BT bf16 elements per panel.
    LDS_KP_PANEL_ELEMS = 64 * BT
    LDS_KP_ELEMS = NUM_K_BLOCKS * LDS_KP_PANEL_ELEMS
    LDS_KP_BYTES = LDS_KP_ELEMS * 2

    # HIP-ALIGN 3: gated v_new in a shared2 panel (like h / HIP gated_v_panel);
    # GEMM2 B read via load_shared2 (contiguous BT, matches the k A frag).
    LDS_GV_ELEMS = (BT // 4) * BV * 4  # 16 bt_groups x BV cols x 4 BT

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
    _K5_KERNEL_REVISION = (
        119  # +state_indices slot (indexed state-pool gather / in-place write-back)
    )

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
    # HIP-ALIGN 1b: h_state panels (GEMM1 B operand, shared2 layout).
    lds_hp_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_hp_offset + LDS_HP_BYTES
    # HIP-ALIGN 3: gated_v ALIASES h_state_panel1 (the kb=1 h_state panel), like
    # HIP's ``gated_v_panel = h_state_panel1``. LDS_GV_ELEMS == LDS_HP_PANEL_ELEMS
    # (both 64*BV) so it fits exactly, and no separate buffer is allocated
    # (saves LDS_GV_BYTES). Correct because gated_v is written only AFTER GEMM1
    # has finished reading the h_state panels -- a WAR barrier is inserted right
    # before the gated_v store to enforce that ordering across warps.
    lds_gv_offset = lds_hp_offset + LDS_HP_PANEL_ELEMS * 2  # panel 1 (byte offset)
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
        state_indices_tensor: fx.Pointer,
        T_val: fx.Int32,
        T_flat: fx.Int32,
        N_val: fx.Int32,
    ):
        i_v = fx.block_idx.x
        i_nh = fx.block_idx.y
        i_n = i_nh // fx.Int32(H)
        i_h = i_nh % fx.Int32(H)

        # Indexed state-pool gather: when USE_STATE_INDICES, the SSM state slot
        # for sequence ``i_n`` is ``state_indices[i_n]`` (addressing a pool
        # ``[pool_size, H, V, K]``) rather than ``i_n`` itself (dense
        # ``[N, H, V, K]``). Only h0 (read) and ht (in-place write-back) use this
        # slot; the per-chunk h snapshot stays dense (i_n-indexed).
        if const_expr(USE_STATE_INDICES):
            si_ = GTensor(state_indices_tensor, dtype=T.i32, shape=(-1,))
            state_n = si_[fx.Index(i_n)]
        else:
            state_n = i_n
        state_nh = state_n * fx.Int32(H) + i_h

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

        # k panels (bf16, rotating-pair swizzle) -- separate from w
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

        # HIP-ALIGN 2b: rotating-pair swizzled reads of k panel / plain b64 gated_v.
        def _lds_read_kp_bf16x4(elem_idx):
            return vector.load_op(v4bf16_hp_type, lds_kp_memref, [fx.Index(elem_idx)])

        def _lds_read_gv_bf16x4(elem_idx):
            return vector.load_op(v4bf16_hp_type, lds_gv_memref, [fx.Index(elem_idx)])

        # -- Cooperative load decomposition --
        load_row_in_batch = tid // fx.Int32(THREADS_PER_ROW_64)
        load_col_base = (tid % fx.Int32(THREADS_PER_ROW_64)) * fx.Int32(LOAD_VEC_WIDTH)

        # -- LDS vector read helpers (generates ds_read_b128 for 8xbf16) --

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

        # HIP-ALIGN 2b: k_panel rotating-pair swizzle address computation.
        # Port of HIP's k_panel_rotating_pair_base_bytes / _addr_bytes.
        # Returns a BYTE offset within a panel; caller converts to element
        # offset (>> 1) before indexing the bf16 memref.
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

        # HIP-ALIGN 2b: load_a_k_fragment_rotating -- GEMM2 A operand read.
        # Mirrors HIP's load_a_k_fragment_rotating(base, row_base, t_base, lane).
        def _load_a_k_rotating(panel_base_elems, row_base, t_base):
            row = row_base + lane_n
            t0 = t_base + lane_m_base * fx.Int32(4)
            pair_col = t0 >> fx.Int32(1)
            byte_off = _k_panel_rotating_pair_addr_bytes(row, pair_col)
            elem_off = byte_off >> fx.Int32(1)
            return vector.load_op(
                v4bf16_w_type,
                lds_kp_memref,
                [fx.Index(panel_base_elems + elem_off)],
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
            h0_base = state_nh * fx.Int32(V * K)
        if const_expr(STORE_FINAL_STATE):
            ht_base = state_nh * fx.Int32(V * K)

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
        c_zero = fx.Index(0)
        c_one = fx.Index(1)
        nt_idx = fx.Index(NT)

        # -- PROLOGUE: load w/k for chunk 0 to LDS (HIP .cu:1125-1141) --
        _prol_it = fx.Int32(0)
        for kb in range_constexpr(NUM_K_BLOCKS):
            wp_panel_base = fx.Int32(kb * LDS_WP_PANEL_ELEMS)
            for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                row = fx.Int32(batch * ROWS_PER_BATCH_64) + load_row_in_batch
                abs_row = _prol_it * fx.Int32(BT) + row
                safe_row = (abs_row < T_local).select(abs_row, fx.Int32(0))
                w_g_off = (
                    w_base + safe_row * stride_w + fx.Int32(kb * 64) + load_col_base
                )
                wvec = w_.vec_load((fx.Index(w_g_off),), LOAD_VEC_WIDTH)
                swz = wp_panel_base + _w_panel_swz_elems(row, load_col_base)
                lds_wp.vec_store((fx.Index(swz),), wvec.shuffle(wvec, [0, 1, 2, 3]), 4)
                lds_wp.vec_store(
                    (fx.Index(swz ^ fx.Int32(4)),),
                    wvec.shuffle(wvec, [4, 5, 6, 7]),
                    4,
                )
        k_abs_t0_prol = _prol_it * fx.Int32(BT) + k_t0_pf
        k_abs_t1_prol = _prol_it * fx.Int32(BT) + k_t1_pf
        k_safe_t0_prol = (k_abs_t0_prol < T_local).select(k_abs_t0_prol, fx.Int32(0))
        k_safe_t1_prol = (k_abs_t1_prol < T_local).select(k_abs_t1_prol, fx.Int32(0))
        for kb in range_constexpr(NUM_K_BLOCKS):
            kp_pbase = fx.Int32(kb * LDS_KP_PANEL_ELEMS)
            k_col_off = fx.Int32(kb * 64) + k_row_base_pf
            k_g_off_t0 = k_base + k_safe_t0_prol * stride_k + k_col_off
            k_g_off_t1 = k_base + k_safe_t1_prol * stride_k + k_col_off
            kvec_t0 = k_.vec_load((fx.Index(k_g_off_t0),), LOAD_VEC_WIDTH)
            kvec_t1 = k_.vec_load((fx.Index(k_g_off_t1),), LOAD_VEC_WIDTH)
            for i in range_constexpr(LOAD_VEC_WIDTH):
                row_i = k_row_base_pf + fx.Int32(i)
                byte_off = _k_panel_rotating_pair_addr_bytes(row_i, k_pair_col_pf)
                elem_off = byte_off >> fx.Int32(1)
                lds_kp[fx.Index(kp_pbase + elem_off)] = kvec_t0[i]
                lds_kp[fx.Index(kp_pbase + elem_off + fx.Int32(1))] = kvec_t1[i]
        gpu.barrier()

        for i_t, state in range(c_zero, nt_idx, c_one, init=init_state):
            h_accs_in = list(state[:NUM_H_ACCS])
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

                    # HIP-ALIGN 1b: write the h_state panel cell (shared2). This
                    # lane owns k_group (row_block = wid*4+lane_m_base) at V-col
                    # = nr*16+lane_n; 4 warps together fill all 16 row_blocks.
                    hp_row_block = wid * fx.Int32(4) + lane_m_base
                    hp_cell = fx.Int32(kb * LDS_HP_PANEL_ELEMS) + (
                        hp_row_block * fx.Int32(BV) + hp_col
                    ) * fx.Int32(4)
                    lds_hp.vec_store(
                        (fx.Index(hp_cell),),
                        _f32x4_to_bf16x4_rne(acc_val),
                        4,
                    )

                    # HIP-ALIGN 1a: [V][K/4-group] transpose buffer, XOR-swizzled
                    # (k_group ^ (v & 0xF)) to break bank conflicts on the
                    # scatter write -- mirrors HIP ``h_transpose_buf_offset``.
                    # The 4 elem_i are one k_group (4 contiguous K) -> one b64.
                    ht_kg = fx.Int32(kb * 16) + wid * fx.Int32(4) + lane_m_base
                    ht_idx = (
                        hp_col * fx.Int32(K // 4) + (ht_kg ^ (hp_col & fx.Int32(0xF)))
                    ) * fx.Int32(4)
                    lds_ht.vec_store(
                        (fx.Index(ht_idx),),
                        _f32x4_to_bf16x4_rne(acc_val),
                        4,
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
                    u_off = v_base + safe_u_row * stride_v + u_col_pf
                    u_prefetch.append(v_[fx.Index(u_off)])

            if const_expr(USE_G):
                g_last = g_[fx.Index(i_h * T_flat + (bos + last_idx_raw))]
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
                        (g_[fx.Index(i_h * T_flat + (bos + safe_row))], in_bounds)
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
                        + safe_row_next * stride_w
                        + fx.Int32(kb * 64)
                        + load_col_base
                    )
                    w_next_vecs.append(
                        w_.vec_load((fx.Index(w_g_off_next),), LOAD_VEC_WIDTH)
                    )
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
                gate_vec = vector.from_elements(T.f32x4, gate_elems)

            if const_expr(SAVE_NEW_VALUE):

                def _emit_vn_store(off, value):
                    vn_[fx.Index(off)] = value

            for idx in range_constexpr(N_REPEAT):
                bv_val = bv_accs[idx]
                u_col = i_v * fx.Int32(BV) + fx.Int32(idx * 16) + lane_n
                u_f32_elems = []
                for elem_i in range_constexpr(4):
                    u_raw = u_prefetch[idx * 4 + elem_i]
                    u_bf16 = fx.BFloat16(u_raw)
                    u_f32_elems.append(u_bf16.to(fx.Float32))
                u_f32 = vector.from_elements(T.f32x4, u_f32_elems)
                vn_val = u_f32 - bv_val

                if const_expr(SAVE_NEW_VALUE):
                    vn_bf16 = fx.Vector(_f32x4_to_bf16x4_rne(vn_val), (4,), fx.BFloat16)
                    bt_tile_base = wid * fx.Int32(16)
                    for elem_i in range_constexpr(4):
                        vn_bt_row = (
                            i_t_i32 * fx.Int32(BT)
                            + bt_tile_base
                            + lane_m_base * fx.Int32(4)
                            + fx.Int32(elem_i)
                        )
                        if (vn_bt_row < T_local).ir_value():
                            bf16_v = vn_bf16[elem_i]
                            vn_off = vn_base + vn_bt_row * fx.Int32(V) + u_col
                            _emit_vn_store(vn_off, bf16_v)

                if const_expr(USE_G):
                    gated_val = vn_val * gate_vec
                else:
                    gated_val = vn_val
                gv_col = fx.Int32(idx * 16) + lane_n
                gv_row_block = wid * fx.Int32(4) + lane_m_base
                gv_cell = (gv_row_block * fx.Int32(BV) + gv_col) * fx.Int32(4)
                lds_gv.vec_store(
                    (fx.Index(gv_cell),),
                    _f32x4_to_bf16x4_rne(gated_val),
                    4,
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
                        (fx.Index(k_base + k_safe_t0_next * stride_k + k_col_off_pf),),
                        LOAD_VEC_WIDTH,
                    )
                )
                k_next_vecs_t1.append(
                    k_.vec_load(
                        (fx.Index(k_base + k_safe_t1_next * stride_k + k_col_off_pf),),
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
                gk_chunk_base = (bos + last_idx_raw) * fx.Int32(H * K) + i_h * fx.Int32(
                    K
                )
                for kb in range_constexpr(NUM_K_BLOCKS):
                    gk_elems = []
                    for elem_i in range_constexpr(4):
                        global_k = (
                            fx.Int32(kb * 64)
                            + wid * fx.Int32(16)
                            + lane_m_base * fx.Int32(4)
                            + fx.Int32(elem_i)
                        )
                        gk_raw = gk_[fx.Index(gk_chunk_base + global_k)]
                        gk_elems.append(_fast_exp(gk_raw))
                    gk_vec = vector.from_elements(T.f32x4, gk_elems)
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
                    vector.load_op(v4bf16_w_type, lds_ht_memref, [fx.Index(off_lo)]),
                    (4,),
                    fx.BFloat16,
                )
                val_hi = fx.Vector(
                    vector.load_op(v4bf16_w_type, lds_ht_memref, [fx.Index(off_hi)]),
                    (4,),
                    fx.BFloat16,
                )
                vec8 = val_lo.shuffle(val_hi, [0, 1, 2, 3, 4, 5, 6, 7])
                v_global = i_v * fx.Int32(BV) + v_loc
                h_off = h_base + i_t_i32 * stride_h + v_global * fx.Int32(K) + k8
                h_.vec_store((fx.Index(h_off),), vec8, LOAD_VEC_WIDTH)

            # -- 6. GEMM2: h += k^T @ v_new_gated (no w prefetch/interleave).
            BT_STEPS = BT // WMMA_K
            for kb in range_constexpr(NUM_K_BLOCKS):
                for bt_s in range_constexpr(BT_STEPS):
                    # HIP-ALIGN 2b: A = k load_a_k_fragment_rotating.
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
            # Closures hide lds_wp/lds_kp (STensor) from the FlyDSL AST
            # rewriter which otherwise tries to yield them through scf.if.
            # Closures hide lds_wp/lds_kp (STensor) from the FlyDSL AST
            # rewriter which otherwise tries to yield them through scf.if.
            def _emit_wp_vec_store(idx_tuple, val, width):
                lds_wp.vec_store(idx_tuple, val, width)

            def _emit_kp_scalar_store(idx, val):
                lds_kp[fx.Index(idx)] = val

            has_next = next_i_t * fx.Int32(BT) < T_local
            if has_next.ir_value():
                gpu.barrier()
                _pf_idx = 0
                for kb in range_constexpr(NUM_K_BLOCKS):
                    for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                        wvec_pf = w_next_vecs[_pf_idx]
                        swz_pf = w_next_swz[_pf_idx]
                        _emit_wp_vec_store(
                            (fx.Index(swz_pf),),
                            wvec_pf.shuffle(wvec_pf, [0, 1, 2, 3]),
                            4,
                        )
                        _emit_wp_vec_store(
                            (fx.Index(swz_pf ^ fx.Int32(4)),),
                            wvec_pf.shuffle(wvec_pf, [4, 5, 6, 7]),
                            4,
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
                        _emit_kp_scalar_store(kp_pbase + elem_off, kvec_t0_pf[i])
                        _emit_kp_scalar_store(
                            kp_pbase + elem_off + fx.Int32(1), kvec_t1_pf[i]
                        )

            results = yield [_to_raw(v) for v in h_accs_in]

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
                        out_vec = _f32x4_to_bf16x4_rne(acc_val)
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
        state_indices_tensor: fx.Pointer,
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
