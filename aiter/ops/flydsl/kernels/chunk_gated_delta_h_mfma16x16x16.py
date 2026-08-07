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

_LOG2E = math.log2(math.e)  # 1.4426950408889634


def _gview(tensor, base, shape, stride):
    """Buffer-resource view of the global slot ``tensor``, rooted at ELEMENT
    offset ``base``.

    The slot's own memref layout is discarded and replaced by ``shape`` /
    ``stride``, so the ``make_buffer_tensor`` that turns it into an
    (unbounded, ``max_size``) buffer resource belongs here: every slot is
    described by exactly one view.

    ``base`` is the per-block runtime scalar (varlen / head / state-slot origin)
    that carries no tile structure and so stays an iterator shift; ``shape`` and
    ``stride`` describe the regular part addressed by coordinate. The innermost
    mode is always the element count of one vector access, so slicing it off
    with ``None`` yields exactly the copy-atom tile. ``base`` may be None for a
    slot that is already rooted at the tensor origin.
    """
    it = fx.get_iter(fx.rocdl.make_buffer_tensor(tensor, max_size=True))
    if base is not None:
        it = fx.add_offset(it, base)
    return fx.Tensor(fx.make_view(it, fx.make_layout(shape, stride)))


def _load_vec(atom, tile, width, numeric):
    """Load ``width`` elements from the coordinate ``tile`` through a register
    fragment. ``atom`` picks the access -- buffer_load from a global view,
    ds_read from an LDS one. The fragment is folded away by
    fly-promote-regmem-to-vectorssa."""
    frag = fx.make_rmem_tensor(width, numeric)
    fx.copy(atom, tile, frag)
    vec = frag.load()
    return vec[0] if width == 1 else vec


def _store_vec(atom, tile, value, width, numeric):
    """Store ``value`` into the coordinate ``tile``, the inverse of
    ``_load_vec``."""
    frag = fx.make_rmem_tensor(width, numeric)
    frag.store(fx.Vector.from_elements([value], dtype=numeric) if width == 1 else value)
    fx.copy(atom, frag, tile)


def _make_fast_exp(g_is_log2_scaled: bool):
    """``exp(x)`` lowered via ``exp2``. If ``g`` is already log2(e)-prescaled the
    ``* LOG2E`` is dropped (single ``v_exp_f32``); otherwise it is folded in.

    Deliberately the raw ``rocdl.exp2`` (``llvm.amdgcn.exp2.f32``) rather than
    the ``fx.exp2`` wrapper: both lower to ``v_exp_f32``, but ``fx.exp2`` goes
    through the math dialect, whose IEEE-correct lowering appends a denormal
    rescale guard around every call. Measured on gfx942 (5 exp sites in this
    kernel): +5 v_cmp_gt_f32, +5 v_add_f32, +5 v_ldexp_f32, +9 v_cndmask,
    i.e. 819 -> 849 instructions and 146 -> 148 VGPRs, for a fix-up this
    kernel does not need (the gates underflow to 0 either way).

    The raw intrinsic takes / returns an untyped SSA value, so ``x`` is
    unwrapped and the result re-wrapped here, keeping callers on the typed
    fx API.
    """
    if g_is_log2_scaled:
        return lambda x: fx.Float32(rocdl.exp2(T.f32, x.ir_value()))
    return lambda x: fx.Float32(rocdl.exp2(T.f32, (x * _LOG2E).ir_value()))


# Default fp32->bf16 output-conversion mode. When True, use bit-truncation to
# match HIP's ``float_to_bf16`` (``bit_cast<u32>(x) >> 16``) so flydsl-hip
# outputs are bit-identical to the HIP/C++ K5 kernel. When False, use RNE
# (~0.5 ulp closer to the FP32 reference, but NOT bit-matching HIP). This is
# only the default; the actual mode is a compile option (``BF16_CONVERT_TRUNC``)
# threaded through ``compile_chunk_gated_delta_h_mfma16_hip``.
_BF16_CONVERT_TRUNC_DEFAULT = True


def _make_bf16_converter(trunc: bool):
    """Return the fp32x4 -> bf16x4 output converter for this compile.

    ``trunc=True`` (the default) keeps the high 16 bits with no rounding bias,
    which is bit-identical to HIP ``float_to_bf16``
    (``bit_cast<u32>(x) >> 16``) and arch-neutral. ``trunc=False`` rounds to
    nearest even instead: LLVM lowers it to the native ``v_cvt_pk_bf16_f32`` on
    gfx950 and to a software RNE sequence (bias + NaN->qNaN fix-up) on gfx942,
    both matching torch/HIP ``__float2bfloat16_rn`` (verified bit-exact on
    gfx942).
    """
    if trunc:
        return lambda v: (
            (v.bitcast(fx.Uint32) >> fx.full(4, 16, fx.Uint32))
            .to(fx.Uint16)
            .bitcast(fx.BFloat16)
        )
    return lambda v: v.to(fx.BFloat16)


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
    assert BV % 16 == 0, f"BV must be a multiple of the MFMA N of 16, got BV={BV}"
    NUM_K_BLOCKS = K // 64

    # Tensor slots use the ``fx.Tensor`` ABI, so each slot arrives as a memref
    # carrying its element type, and ``_gview`` turns it into a buffer resource
    # plus the per-block view the access sites index by coordinate. The
    # shape/stride of the incoming memref is never read: the per-block origin
    # (varlen bos, chunk offset, head, state slot) is a runtime scalar that no
    # layout coordinate can express, so it stays an iterator shift and the
    # regular part is re-described by the view. The buffer descriptors are built
    # with ``max_size`` (unbounded), matching the in-kernel clamps (safe_row /
    # in_bounds selects) that already keep every access in range. Element types
    # therefore come from the host tensors -- the placeholder tensors the host
    # passes for unused slots must match the dtype the body assumes for that
    # slot.

    _fast_exp = _make_fast_exp(G_IS_LOG2_SCALED)
    # fp32->bf16 output converter selected by the compile option (arch-aware).
    _f32x4_to_bf16x4 = _make_bf16_converter(BF16_CONVERT_TRUNC)

    WARP_SIZE = 64
    NUM_WARPS = 4
    BLOCK_THREADS = NUM_WARPS * WARP_SIZE

    # MFMA tile is 16x16x16; each k-step issues a lo/hi MFMA pair and so
    # advances K by 32 (K_STEP), which is why K_STEP != the MFMA K of 16.
    MFMA_N = 16
    K_STEP = 32
    N_REPEAT = BV // MFMA_N

    # Warp partition: wid owns 16 BT rows and all of V (see module docstring).

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
    assert LDS_GV_ELEMS == LDS_HP_PANEL_ELEMS, (
        f"gated_v aliases h_state panel 1 and must match it exactly, got "
        f"{LDS_GV_ELEMS} vs {LDS_HP_PANEL_ELEMS}"
    )

    # [V][K] transpose buffer for the h snapshot HBM store.
    # h_accs (f32) is written here in V-major / K-innermost order so 8 adjacent
    # K land contiguously -> a single b128 (ds_read_b128 + buffer_store b128),
    # matching HIP's ``h_transpose_buf`` + ``coalesced_vk_store_from_transpose``
    # (fewer store instructions than the per-element bf16 readout). K-contiguous
    # (no pad) so the 8-wide vec load/store stays 16 B aligned.
    LDS_HT_STRIDE = K
    LDS_HT_ELEMS = BV * LDS_HT_STRIDE

    # ``hp`` must stay a single Array: its view spans both 64-K panels with a
    # panel stride of LDS_HP_PANEL_ELEMS, which requires the two panels to be
    # contiguous. Separate struct fields are independent static LDS symbols with
    # no guaranteed adjacency. Same for ``wp``.
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
    K_STEPS_PER_BLOCK = 64 // K_STEP

    @flyc.kernel(name="chunk_gdn_fwd_h_flydsl_mfma16_hip")
    def gdn_h_kernel(
        k_tensor: fx.Tensor,
        v_tensor: fx.Tensor,
        w_tensor: fx.Tensor,
        v_new_tensor: fx.Tensor,
        g_tensor: fx.Tensor,
        gk_tensor: fx.Tensor,
        h_tensor: fx.Tensor,
        h0_tensor: fx.Tensor,
        ht_tensor: fx.Tensor,
        cu_seqlens_tensor: fx.Tensor,
        chunk_offsets_tensor: fx.Tensor,
        state_indices_tensor: fx.Tensor,
        T_val: fx.Int32,
        T_flat: fx.Int32,
        # Unused by the kernel body: N is already encoded in the grid.y extent.
        # Kept so the launch signature stays aligned with the baseline fork's.
        N_val: fx.Int32,
    ):
        i_v = fx.Int32(gpu.block_id("x"))
        i_nh = fx.Int32(gpu.block_id("y"))
        i_n = i_nh // H
        i_h = i_nh % H

        # Indexed state-pool gather: when USE_STATE_INDICES, the SSM state slot
        # for sequence ``i_n`` is ``state_indices[i_n]`` (addressing a pool
        # ``[pool_size, H, V, K]``) rather than ``i_n`` itself (dense
        # ``[N, H, V, K]``). Only h0 (read) and ht (in-place write-back) use this
        # slot; the per-chunk h snapshot stays dense (i_n-indexed).
        # Buffer-copy atoms shared by every global access below. The 128b/64b
        # ones carry the cooperative vector loads (8 bf16 / 4 state elements);
        # the 32b/16b ones the per-lane scalars.
        cp_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        cp_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        cp_bf16 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        cp_bf16x8 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)

        if const_expr(USE_STATE_INDICES):
            si_view = _gview(state_indices_tensor, None, (N_val, 1), (1, 1))
            state_n = _load_vec(cp_i32, fx.slice(si_view, (i_n, None)), 1, fx.Int32)
        else:
            state_n = i_n
        state_nh = state_n * H + i_h

        tid = fx.Int32(gpu.thread_id("x"))
        wid = tid // WARP_SIZE
        lane = tid % WARP_SIZE

        # SSM-state dtype is selected by the compile-time flag; the h0/ht slots
        # carry it as their memref element type. The accesses are 4 contiguous
        # K, so the copy atom is sized to match (b64 in bf16, b128 in f32).
        state_num = fx.BFloat16 if STATE_DTYPE_BF16 else fx.Float32
        cp_state_x4 = fx.make_copy_atom(
            fx.rocdl.BufferCopy64b() if STATE_DTYPE_BF16 else fx.rocdl.BufferCopy128b(),
            state_num,
        )

        if const_expr(IS_VARLEN):
            cu_view = _gview(cu_seqlens_tensor, None, (N_val + 1, 1), (1, 1))
            co_view = _gview(chunk_offsets_tensor, None, (N_val, 1), (1, 1))

        # -- LDS --
        # Each MMA operand panel is a layout view in the shape the tiled MMA
        # expects -- A as (M, K), B as (N, K) -- so the per-lane addresses come
        # out of partition_S once, outside the chunk loop, instead of being
        # recomputed per access. The K mode is nested ``(4, k_groups, panels)``:
        # the innermost 4 elements are one k_group, which is exactly one MFMA
        # operand and one b64 access. The swizzles use base=2 (elements -> 8 B),
        # so those 4 elements always stay contiguous.
        # The k panel is the one exception and keeps hand-written bit math: its
        # rotating-pair address needs three overlapping XOR terms, while a
        # layout carries a single swizzle.
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_kp_ptr = lds.kp.ptr  # k panels (rotating-pair swizzle)

        KG_PER_BLOCK = 64 // 4  # k_groups per 64-K panel

        # GEMM1 A -- w panels as (BT, K). Per panel this is [BT][64] row-major
        # + S<4,2,4>, the layout form of HIP's ``w_panel_swizzle`` (verified
        # bit-exact on all BT*64 addresses). The panel stride is far above the
        # swizzled bits, so both panels share one view.
        sW = fx.make_view(
            lds.wp.ptr,
            fx.make_composed_layout(
                fx.static(fx.SwizzleType.get(4, 2, 4)),
                fx.make_layout(
                    (BT, (4, KG_PER_BLOCK, NUM_K_BLOCKS)),
                    (64, (1, 4, LDS_WP_PANEL_ELEMS)),
                ),
            ),
        )
        # GEMM1 B -- h_state panels (shared2) as (BV, K). No swizzle needed.
        sH = fx.make_view(
            lds.hp.ptr,
            fx.make_layout(
                (BV, (4, KG_PER_BLOCK, NUM_K_BLOCKS)),
                (4, (1, BV * 4, LDS_HP_PANEL_ELEMS)),
            ),
        )
        # GEMM2 B -- gated_v as (BV, BT), same shared2 cell layout.
        # gated_v ALIASES h_state panel 1 (like HIP's ``gated_v_panel =
        # h_state_panel1``): gated_v is written only AFTER GEMM1 has finished
        # reading the h_state panels, and a WAR barrier before the gated_v store
        # enforces that ordering across warps. Aliasing avoids a separate
        # LDS_GV buffer, which would cost an occupancy step.
        sGV = fx.make_view(
            lds.hp.ptr + LDS_HP_PANEL_ELEMS,
            fx.make_layout((BV, (4, BT // 4)), (4, (1, BV * 4))),
        )
        # [V][K] transpose buffer: [BV][K] row-major + S<4,2,5>, the layout form
        # of the hand-written ``k_group ^ (v & 0xF)`` scatter. Not an MMA
        # operand -- it only stages the h snapshot for the b128 HBM store -- so
        # it keeps a plain (V, k_group, 4) shape.
        sHT = fx.make_view(
            lds.ht.ptr,
            fx.make_composed_layout(
                fx.static(fx.SwizzleType.get(4, 2, 5)),
                fx.make_layout((BV, K // 4, 4), (K, 4, 1)),
            ),
        )

        # LDS <-> register moves go through the same _load_vec / _store_vec as
        # the global accesses above, just with an LDS copy atom. Every one of
        # them moves a single k_group (4 bf16 = one b64), so the two wrappers
        # below pin the atom / width / dtype and leave only the tile.
        cp_lds_x4 = fx.make_copy_atom(fx.UniversalCopy64b(), fx.BFloat16)

        def _lds_read_x4(tile):
            return _load_vec(cp_lds_x4, tile, 4, fx.BFloat16)

        def _lds_write_x4(tile, vec4):
            _store_vec(cp_lds_x4, tile, vec4, 4, fx.BFloat16)

        v4bf16 = T.vec(4, T.bf16)

        # 16x16x16 bf16 MFMA (K-tile 16, so A/B are bf16x4 -- one b64 read --
        # and C is f32x4), tiled over the 4 waves along M. That atom's TV layout
        # is exactly this kernel's wave/lane mapping: A gives lane l element
        # A[wid*16 + l%16][(l//16)*4 + v], B gives B[l%16][(l//16)*4 + v] with
        # a zero wave stride (B is shared by all 4 waves), and C gives
        # C[wid*16 + (l//16)*4 + v][l%16]. So the operand addresses, the
        # accumulator layout and the h/v staging below all keep their previous
        # meaning -- they are just derived from the layout algebra now.
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16, fx.Float32))
        tiled_mma = fx.make_tiled_mma(
            mma_atom, fx.make_layout((NUM_WARPS, 1, 1), (1, 0, 0))
        )
        thr_cp_a = fx.make_tiled_copy_A(cp_lds_x4, tiled_mma).get_slice(tid)
        thr_cp_b = fx.make_tiled_copy_B(cp_lds_x4, tiled_mma).get_slice(tid)

        # Per-lane source partitions. These fold the whole swizzle + lane
        # mapping into one address per K-tile, computed once here rather than
        # per access inside the chunk loop.
        pS_w = thr_cp_a.partition_S(sW)  # GEMM1 A: ((4,1), 1, (4, NUM_K_BLOCKS))
        pS_h = thr_cp_b.partition_S(sH)  # GEMM1 B: ((4,1), N_REPEAT, K_TILES)
        pS_gv = thr_cp_b.partition_S(sGV)  # GEMM2 B: ((4,1), N_REPEAT, BT_TILES)

        # Register fragments. ``retile`` gives the copy-side view of the same
        # registers, which is what fx.copy writes into; fx.gemm consumes the
        # mma-side view. A's K mode stays nested as (k_tile_in_panel, panel)
        # while the copy/B side is flat, so the two index forms are related by
        # ``k_tile = k_tile_in_panel + panel * K_TILES_PER_BLOCK``.
        K_TILES_PER_BLOCK = 64 // 16
        frag_w = tiled_mma.make_fragment_A(sW)
        frag_h = tiled_mma.make_fragment_B(sH)
        frag_gv = tiled_mma.make_fragment_B(sGV)
        frag_w_rt = thr_cp_a.retile(frag_w)
        frag_h_rt = thr_cp_b.retile(frag_h)
        frag_gv_rt = thr_cp_b.retile(frag_gv)

        # GEMM2's A operand is the rotating-pair k panel, whose address is not
        # expressible as a layout, so its fragment is filled by hand below
        # instead of by a partitioned fx.copy.
        frag_k = fx.make_rmem_tensor(
            fx.tiled_mma_partition_shape(fx.MmaOperand.A, tiled_mma, (64, BT)),
            fx.BFloat16,
        )
        # Accumulators: b_v (GEMM1) over the full (BT, BV) tile, and one h state
        # accumulator per 64-K block (GEMM2's M tile is a 64-K block).
        frag_bv = fx.make_rmem_tensor(
            fx.tiled_mma_partition_shape(fx.MmaOperand.C, tiled_mma, (BT, BV)),
            fx.Float32,
        )
        frag_h_accs = [
            fx.make_rmem_tensor(
                fx.tiled_mma_partition_shape(fx.MmaOperand.C, tiled_mma, (64, BV)),
                fx.Float32,
            )
            for _ in range(NUM_K_BLOCKS)
        ]

        # -- Cooperative load decomposition --
        load_row_in_batch = tid // THREADS_PER_ROW_64
        # Index of this thread's LOAD_VEC_WIDTH-wide K vector within a 64-K
        # block, i.e. the global-view column coordinate.
        load_vec_group = tid % THREADS_PER_ROW_64
        load_col_group = load_vec_group * (LOAD_VEC_WIDTH // 4)

        # k_panel rotating-pair swizzle address computation.
        # Port of HIP's k_panel_rotating_pair_base_bytes / _addr_bytes.
        # Returns a BYTE offset within a panel; caller converts to element
        # offset (>> 1) before indexing the bf16 LDS pointer.
        def _k_panel_rotating_pair_addr_bytes(row, pair_col):
            row_block = row >> 3
            row_in_block = row & 7
            # k_panel_rotating_pair_base_bytes(row_block, pair_col)
            tid_like = (pair_col << 3) | row_block
            lane_1_2 = tid_like & 6
            base = lane_1_2 << 10
            low = (lane_1_2 << 2) ^ ((tid_like & 0xF8) >> 1)
            toggle = (tid_like & 1) * 0x440
            low = low ^ toggle
            base_bytes = base | low
            return (base_bytes ^ (row_in_block << 3)) + (row_in_block << 7)

        # load_a_k_fragment_rotating -- GEMM2 A operand read.
        # Mirrors HIP's load_a_k_fragment_rotating(base, row_base, t_base, lane).
        def _load_a_k_rotating(panel_base_elems, row_base, t_base):
            row = row_base + lane_n
            t0 = t_base + lane_m_base * 4
            pair_col = t0 >> 1
            byte_off = _k_panel_rotating_pair_addr_bytes(row, pair_col)
            elem_off = byte_off >> 1
            return fx.ptr_load(
                lds_kp_ptr + (panel_base_elems + elem_off), result_type=v4bf16
            )

        # -- Prologue: compute bos, T_local, NT, boh --
        if const_expr(IS_VARLEN):
            bos = _load_vec(cp_i32, fx.slice(cu_view, (i_n, None)), 1, fx.Int32)
            eos = _load_vec(cp_i32, fx.slice(cu_view, (i_n + 1, None)), 1, fx.Int32)
            T_local = eos - bos
            NT = (T_local + (BT - 1)) // BT
            boh = _load_vec(cp_i32, fx.slice(co_view, (i_n, None)), 1, fx.Int32)
        else:
            bos = i_n * T_val
            T_local = T_val
            NT = (T_local + (BT - 1)) // BT
            boh = i_n * NT

        # -- Global tensor views --
        # Every slot is addressed as ``base + coordinate``. The base is the
        # per-block runtime origin (varlen bos, chunk offset, head, state slot),
        # a scalar with no tile structure, so it stays an iterator shift; the
        # rest is a static-stride layout that the access sites index by
        # coordinate. Bases stay i64 because long-context snapshot tensors can
        # exceed 2^31 elements; the coordinates are per-chunk and fit in i32.
        i_n64 = fx.Int64(i_n)
        i_h64 = fx.Int64(i_h)
        bos64 = fx.Int64(bos)
        boh64 = fx.Int64(boh)
        t_flat64 = fx.Int64(T_flat)

        VEC = LOAD_VEC_WIDTH

        # h: [B, NT, H, V, K] (VK) -- base = (boh*H + i_h) * V * K
        h_view = _gview(
            h_tensor,
            (boh64 * fx.Int64(H) + i_h64) * fx.Int64(V * K),
            (NT, V, K // VEC, VEC),
            (H * V * K, K, VEC, 1),
        )

        # k: [B, T, Hg, K] -- base = (bos*Hg + i_h//(H//Hg)) * K
        gqa_ratio = H // Hg
        k_view = _gview(
            k_tensor,
            (bos64 * fx.Int64(Hg) + fx.Int64(i_h // gqa_ratio)) * fx.Int64(K),
            (T_local, K // VEC, VEC),
            (Hg * K, VEC, 1),
        )

        if const_expr(WU_CONTIGUOUS):
            if const_expr(IS_VARLEN):
                v_base = (i_h64 * t_flat64 + bos64) * fx.Int64(V)
                w_base = (i_h64 * t_flat64 + bos64) * fx.Int64(K)
            else:
                v_base = ((i_n64 * fx.Int64(H) + i_h64) * t_flat64) * fx.Int64(V)
                w_base = ((i_n64 * fx.Int64(H) + i_h64) * t_flat64) * fx.Int64(K)
            stride_v = V
            stride_w = K
        else:
            v_base = (bos64 * fx.Int64(H) + i_h64) * fx.Int64(V)
            w_base = (bos64 * fx.Int64(H) + i_h64) * fx.Int64(K)
            stride_v = H * V
            stride_w = H * K
        w_view = _gview(w_tensor, w_base, (T_local, K // VEC, VEC), (stride_w, VEC, 1))
        # u and v_new are read/written one element at a time, so their innermost
        # mode is a single element rather than a vector.
        u_view = _gview(v_tensor, v_base, (T_local, V, 1), (stride_v, 1, 1))

        if const_expr(IS_VARLEN):
            vn_base = (i_h64 * t_flat64 + bos64) * fx.Int64(V)
        else:
            vn_base = ((i_n64 * fx.Int64(H) + i_h64) * t_flat64) * fx.Int64(V)
        vn_view = _gview(v_new_tensor, vn_base, (T_local, V, 1), (V, 1, 1))

        # h0/ht: the [V, K] state slot for this (state, head), K split into the
        # 4 contiguous elements of one state vector access.
        state_slot_base = fx.Int64(state_nh) * fx.Int64(V * K)
        if const_expr(USE_INITIAL_STATE):
            h0_view = _gview(h0_tensor, state_slot_base, (V, K // 4, 4), (K, 4, 1))
        if const_expr(STORE_FINAL_STATE):
            ht_view = _gview(ht_tensor, state_slot_base, (V, K // 4, 4), (K, 4, 1))

        # g gate offset. Mirrors the HIP kernel's load_g_value:
        #   idx = i_n*g_stride_b + i_h*g_stride_h + token*g_stride_t
        # with the strides derived from the layout:
        #   head-major  [B, H, T_flat] -> g_stride_h=T_flat, g_stride_t=1
        #   token-major [B, T_flat, H] -> g_stride_h=1,      g_stride_t=H
        # varlen (B==1, flattened): batch stride is 0 and token is the global
        #   token (bos + rel_row), so bos*g_stride_t is folded into the base.
        #   dense: batch stride is H*T_flat (same element count for both
        #   layouts) and token is the in-sequence relative row.
        # The base below excludes the per-token term; the view's leading mode is
        #   token (= last_idx_raw / safe_row, relative).
        if const_expr(USE_G):
            if const_expr(G_HEAD_MAJOR):
                g_stride_h = t_flat64
                g_stride_t = 1
            else:
                g_stride_h = fx.Int64(1)
                g_stride_t = H
            if const_expr(IS_VARLEN):
                g_sh_base = i_h64 * g_stride_h + bos64 * fx.Int64(g_stride_t)
            else:
                g_sh_base = i_n64 * fx.Int64(H) * t_flat64 + i_h64 * g_stride_h
            g_view = _gview(g_tensor, g_sh_base, (T_local, 1), (g_stride_t, 1))

        # gk: [B, T, H, K] token-major. The chunk's last token is a coordinate
        # rather than part of the base, so the view hoists out of the loop.
        if const_expr(USE_GK):
            gk_view = _gview(
                gk_tensor,
                bos64 * fx.Int64(H * K) + i_h64 * fx.Int64(K),
                (T_local, K, 1),
                (H * K, 1, 1),
            )

        # -- MFMA lane mapping for 16x16 tiles --
        lane_n = lane % 16
        lane_m_base = lane // 16

        # -- Initialize h accumulators --
        # One accumulator fragment per 64-K block; within a fragment the
        # per-lane element for v-repeat ``nr`` is ``frag[None, None, nr]``.
        for kb in range_constexpr(NUM_K_BLOCKS):
            frag_h_accs[kb].fill(0.0)

        # -- Load initial state if provided --
        # h0 is [V, K] so K is innermost; 4 consecutive K positions are
        # contiguous in memory -> one buffer_load_dwordx4 instead of 4 scalar
        # f32 loads.
        if const_expr(USE_INITIAL_STATE):
            for kb in range_constexpr(NUM_K_BLOCKS):
                for slot in range_constexpr(N_REPEAT):
                    h0_col = i_v * BV + (slot * 16) + lane_n
                    h0_kgroup = kb * 16 + wid * 4 + lane_m_base
                    loaded_vec = _load_vec(
                        cp_state_x4,
                        fx.slice(h0_view, (h0_col, h0_kgroup, None)),
                        4,
                        state_num,
                    )
                    if const_expr(STATE_DTYPE_BF16):
                        loaded_vec = loaded_vec.to(fx.Float32)
                    _acc_cell = frag_h_accs[kb][None, None, slot]
                    _acc_cell.store(fx.Vector(_acc_cell.load()) + loaded_vec)

        # -- HIP-aligned pipelined main chunk loop --
        # Prologue loads w/k for chunk 0 to LDS; each iteration prefetches
        # NEXT chunk's w (during GEMM1) and k (after gating) into VGPRs,
        # then writes them to LDS at GEMM2 end (mirrors the HIP chunk loop
        # around run_gemm1_fulltile_bvp / run_gemm2_fulltile_bvp).
        # GEMM1 k-blocks issued before the w_next prefetch; the remaining
        # NUM_K_BLOCKS - 1 blocks hide that prefetch's HBM latency.
        GEMM1_PF_SPLIT = 1
        k_vec_group_pf = lane & 7
        k_row_base_pf = k_vec_group_pf * 8
        k_pair_col_pf = wid * 8 + (lane >> 3)
        k_t0_pf = k_pair_col_pf * 2
        k_t1_pf = k_t0_pf + 1

        c_zero = fx.Int64(0)
        c_one = fx.Int64(1)
        nt_idx = fx.Int64(NT)

        # -- PROLOGUE: load w/k for chunk 0 to LDS (HIP
        # load_w_panels_from_global / load_k_panels_from_global) --
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
        # is just "call closure + barrier": the buffer views (w_view/k_view) and copy
        # atoms are captured as free variables rather than appearing inside the
        # if, which would make the FlyDSL AST rewriter try to carry them as
        # scf.if state.
        def _load_chunk0_to_lds():
            # Chunk 0, so the absolute row is just the in-chunk row.
            for kb in range_constexpr(NUM_K_BLOCKS):
                for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                    row = batch * ROWS_PER_BATCH_64 + load_row_in_batch
                    safe_row = (row < T_local).select(row, 0)
                    wvec = _load_vec(
                        cp_bf16x8,
                        fx.slice(w_view, (safe_row, kb * 8 + load_vec_group, None)),
                        LOAD_VEC_WIDTH,
                        fx.BFloat16,
                    )
                    _lds_write_x4(
                        fx.slice(sW, (row, (None, load_col_group, kb))),
                        wvec.shuffle(wvec, [0, 1, 2, 3]),
                    )
                    _lds_write_x4(
                        fx.slice(sW, (row, (None, load_col_group + 1, kb))),
                        wvec.shuffle(wvec, [4, 5, 6, 7]),
                    )
            k_safe_t0_prol = (k_t0_pf < T_local).select(k_t0_pf, 0)
            k_safe_t1_prol = (k_t1_pf < T_local).select(k_t1_pf, 0)
            for kb in range_constexpr(NUM_K_BLOCKS):
                kp_pbase = kb * LDS_KP_PANEL_ELEMS
                k_vec_col = kb * 8 + k_vec_group_pf
                kvec_t0 = _load_vec(
                    cp_bf16x8,
                    fx.slice(k_view, (k_safe_t0_prol, k_vec_col, None)),
                    LOAD_VEC_WIDTH,
                    fx.BFloat16,
                )
                kvec_t1 = _load_vec(
                    cp_bf16x8,
                    fx.slice(k_view, (k_safe_t1_prol, k_vec_col, None)),
                    LOAD_VEC_WIDTH,
                    fx.BFloat16,
                )
                for i in range_constexpr(LOAD_VEC_WIDTH):
                    row_i = k_row_base_pf + i
                    byte_off = _k_panel_rotating_pair_addr_bytes(row_i, k_pair_col_pf)
                    elem_off = byte_off >> 1
                    fx.ptr_store(kvec_t0[i], lds_kp_ptr + (kp_pbase + elem_off))
                    fx.ptr_store(
                        kvec_t1[i],
                        lds_kp_ptr + (kp_pbase + elem_off + 1),
                    )

        has_work = NT > 0
        if has_work:
            _load_chunk0_to_lds()
            gpu.barrier()

        # The accumulator fragments are the loop-carried state; scf.for wants
        # plain values, so each fragment enters and leaves the body as one
        # flat vector.
        h_accs_init = [frag_h_accs[kb].load() for kb in range_constexpr(NUM_K_BLOCKS)]
        H_ACC_ELEMS = N_REPEAT * 4

        for i_t, state in range(c_zero, nt_idx, c_one, init=h_accs_init):
            for kb in range_constexpr(NUM_K_BLOCKS):
                frag_h_accs[kb].store(fx.Vector(state[kb], (H_ACC_ELEMS,), fx.Float32))
            # ``i_t`` is the raw scf.for induction variable (an untyped i64
            # ir.Value), so it must be wrapped before any arithmetic. Chunk-local
            # coordinates use the i32 narrowing; the one i64 use (the h snapshot
            # offset) wraps it as fx.Int64 at the use site.
            i_t_i32 = fx.Int32(i_t)

            # Stage h_accs into (a) the h_state panels [row_block][V] for the
            # GEMM1 B operand (HIP shared2 layout, plain b64 read) and (b) the
            # [V][K] transpose buffer for the b128 HBM store. split-M mapping:
            # wid -> K sub-tile, acc_j(nr) -> V-tile.
            for kb in range_constexpr(NUM_K_BLOCKS):
                for acc_j in range_constexpr(N_REPEAT):
                    acc_val = fx.Vector(frag_h_accs[kb][None, None, acc_j].load())
                    # acc_j == nr (V-tile); K sub-tile = wid*16.
                    hp_col = acc_j * 16 + lane_n

                    # Write the h_state panel cell (shared2). This
                    # lane owns k_group (row_block = wid*4+lane_m_base) at V-col
                    # = nr*16+lane_n; 4 warps together fill all 16 row_blocks.
                    hp_row_block = wid * 4 + lane_m_base
                    _lds_write_x4(
                        fx.slice(sH, (hp_col, (None, hp_row_block, kb))),
                        _f32x4_to_bf16x4(acc_val),
                    )

                    # [V][K/4-group] transpose buffer. The swizzle in sHT breaks
                    # bank conflicts on this scatter write; the 4 elements of a
                    # k_group stay contiguous, so it is one b64.
                    ht_kg = kb * 16 + wid * 4 + lane_m_base
                    _lds_write_x4(
                        fx.slice(sHT, (hp_col, ht_kg, None)),
                        _f32x4_to_bf16x4(acc_val),
                    )

            # w/k for this chunk already in LDS (prologue or prev GEMM2 end).
            gpu.barrier()

            # last_idx for the current chunk (gating).
            next_chunk_end = (i_t_i32 + 1) * BT
            last_idx_raw = (next_chunk_end < T_local).select(
                next_chunk_end, T_local
            ) - 1

            # >>> PREFETCH u + g BEFORE GEMM1: issue all HBM loads for u
            # (N_REPEAT × 4 ushort) and g (4 rows + 1 g_last = 5 dword) now,
            # so the full 64-MFMA GEMM1 chain hides their HBM latency.
            # Without this, LLVM hoists the loads into the GEMM1 middle where
            # only ~2 MFMA can hide them → 4.2M cycle vmcnt stall (39% of stalls).
            u_prefetch = []  # N_REPEAT × 4 bf16 scalars
            for idx in range_constexpr(N_REPEAT):
                u_col_pf = i_v * BV + (idx * 16) + lane_n
                for elem_i in range_constexpr(4):
                    u_bt_row_raw = i_t_i32 * BT + wid * 16 + lane_m_base * 4 + elem_i
                    safe_u_row = (u_bt_row_raw < T_local).select(u_bt_row_raw, 0)
                    u_prefetch.append(
                        _load_vec(
                            cp_bf16,
                            fx.slice(u_view, (safe_u_row, u_col_pf, None)),
                            1,
                            fx.BFloat16,
                        )
                    )

            if const_expr(USE_G):
                g_last = _load_vec(
                    cp_f32, fx.slice(g_view, (last_idx_raw, None)), 1, fx.Float32
                )
                g_row_pf = []
                for elem_i in range_constexpr(4):
                    abs_row = i_t_i32 * BT + wid * 16 + lane_m_base * 4 + elem_i
                    in_bounds = abs_row < T_local
                    safe_row = in_bounds.select(abs_row, 0)
                    g_row_pf.append(
                        (
                            _load_vec(
                                cp_f32,
                                fx.slice(g_view, (safe_row, None)),
                                1,
                                fx.Float32,
                            ),
                            in_bounds,
                        )
                    )

            # -- GEMM1: b_v = w @ h_state, with w_next prefetch interleaved.
            # u/g/w_next loads are all in flight; 64 MFMA hide HBM latency.
            frag_bv.fill(0.0)

            # One 64-K block of GEMM1, issued around the w_next prefetch below so
            # the remaining MFMA chain hides that prefetch's HBM latency.
            # Each fx.gemm covers one 16-K tile over all N_REPEAT v-tiles.
            # The per-k-tile loop is deliberate and must NOT be folded into a
            # single whole-partition fx.copy + fx.gemm: the K-tiles have to stay
            # individually addressable so the w_next prefetch can split the
            # k-blocks (GEMM1_PF_SPLIT) and so the sched_barrier can land on
            # every ks boundary.
            def _gemm1_kblock(kb):
                for ks in range_constexpr(K_STEPS_PER_BLOCK):
                    for half in range_constexpr(K_STEP // 16):
                        kj = ks * (K_STEP // 16) + half
                        kt = kb * K_TILES_PER_BLOCK + kj
                        fx.copy(
                            cp_lds_x4,
                            pS_w[None, None, (kj, kb)],
                            frag_w_rt[None, None, kt],
                        )
                        fx.copy(
                            cp_lds_x4, pS_h[None, None, kt], frag_h_rt[None, None, kt]
                        )
                        fx.gemm(
                            tiled_mma,
                            frag_bv,
                            frag_w[None, None, (kj, kb)],
                            frag_h[None, None, kt],
                            frag_bv,
                        )
                    # gfx942 scheduling: a sched_barrier(mask_mfma) at each ks
                    # boundary stops LLVM from hoisting the per-ks b_frag ds_read
                    # across iterations into one big cluster (the main cause of
                    # LDS port back-pressure), while still letting MFMA overlap
                    # across ks to keep pipeline latency hidden. sched_barrier
                    # only constrains instruction ordering, not addresses or
                    # values, so it is correctness-safe.
                    if const_expr(SCHED_GFX942):
                        rocdl.sched_barrier(rocdl.mask_mfma)

            # GEMM1 first K-block(s) -- before w prefetch.
            for kb in range_constexpr(GEMM1_PF_SPLIT):
                _gemm1_kblock(kb)

            # >>> PREFETCH w_next: HBM loads for next chunk (HIP
            # load_w_panels_from_global_full inside run_gemm1_fulltile_bvp).
            next_i_t = i_t_i32 + 1
            w_next_vecs = []
            w_next_coord = []
            for kb in range_constexpr(NUM_K_BLOCKS):
                for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                    row = batch * ROWS_PER_BATCH_64 + load_row_in_batch
                    abs_row_next = next_i_t * BT + row
                    safe_row_next = (abs_row_next < T_local).select(abs_row_next, 0)
                    w_next_vecs.append(
                        _load_vec(
                            cp_bf16x8,
                            fx.slice(
                                w_view, (safe_row_next, kb * 8 + load_vec_group, None)
                            ),
                            LOAD_VEC_WIDTH,
                            fx.BFloat16,
                        )
                    )
                    w_next_coord.append((kb, row))

            # GEMM1 remaining K-blocks -- MFMA hides u/g/w_next HBM latency.
            for kb in range_constexpr(GEMM1_PF_SPLIT, NUM_K_BLOCKS):
                _gemm1_kblock(kb)

            # WAR barrier: GEMM1 is done reading the h_state panels, so gated_v
            # may now overwrite panel 1.
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
                    abs_row = i_t_i32 * BT + wid * 16 + lane_m_base * 4 + elem_i
                    in_bounds = abs_row < T_local
                    mask_elems.append(
                        in_bounds.select(fx.Float32(1.0), fx.Float32(0.0))
                    )
                gate_vec = fx.Vector.from_elements(mask_elems, dtype=fx.Float32)

            for idx in range_constexpr(N_REPEAT):
                bv_val = fx.Vector(frag_bv[None, None, idx].load())
                u_col = i_v * BV + (idx * 16) + lane_n
                u_f32_elems = []
                for elem_i in range_constexpr(4):
                    u_f32_elems.append(u_prefetch[idx * 4 + elem_i].to(fx.Float32))
                u_f32 = fx.Vector.from_elements(u_f32_elems, dtype=fx.Float32)
                vn_val = u_f32 - bv_val

                if const_expr(SAVE_NEW_VALUE):
                    vn_bf16 = _f32x4_to_bf16x4(vn_val)
                    bt_tile_base = wid * 16
                    for elem_i in range_constexpr(4):
                        vn_bt_row = (
                            i_t_i32 * BT + bt_tile_base + lane_m_base * 4 + elem_i
                        )
                        if vn_bt_row < T_local:
                            bf16_v = vn_bf16[elem_i]
                            _store_vec(
                                cp_bf16,
                                fx.slice(vn_view, (vn_bt_row, u_col, None)),
                                bf16_v,
                                1,
                                fx.BFloat16,
                            )

                # gate_vec: with USE_G it is the decay gate (with the OOB mask
                # already folded in); otherwise it is a pure 0/1 padding mask.
                # Both paths multiply by gate_vec so OOB rows always contribute 0.
                gated_val = vn_val * gate_vec
                gv_col = idx * 16 + lane_n
                gv_row_block = wid * 4 + lane_m_base
                _lds_write_x4(
                    fx.slice(sGV, (gv_col, (None, gv_row_block))),
                    _f32x4_to_bf16x4(gated_val),
                )

            # >>> PREFETCH k_next: HBM loads for next chunk (HIP
            # load_k_panels_from_global_full at the run_gemm1 tail).
            # Overlaps with the barrier + h store below.
            k_abs_t0_next = next_i_t * BT + k_t0_pf
            k_abs_t1_next = next_i_t * BT + k_t1_pf
            k_safe_t0_next = (k_abs_t0_next < T_local).select(k_abs_t0_next, 0)
            k_safe_t1_next = (k_abs_t1_next < T_local).select(k_abs_t1_next, 0)
            k_next_vecs_t0 = []
            k_next_vecs_t1 = []
            for kb in range_constexpr(NUM_K_BLOCKS):
                k_vec_col_pf = kb * 8 + k_vec_group_pf
                k_next_vecs_t0.append(
                    _load_vec(
                        cp_bf16x8,
                        fx.slice(k_view, (k_safe_t0_next, k_vec_col_pf, None)),
                        LOAD_VEC_WIDTH,
                        fx.BFloat16,
                    )
                )
                k_next_vecs_t1.append(
                    _load_vec(
                        cp_bf16x8,
                        fx.slice(k_view, (k_safe_t1_next, k_vec_col_pf, None)),
                        LOAD_VEC_WIDTH,
                        fx.BFloat16,
                    )
                )

            # Apply exp(g_last) decay to h_accs (scalar broadcast).
            if const_expr(USE_G):
                for kb in range_constexpr(NUM_K_BLOCKS):
                    for nr in range_constexpr(N_REPEAT):
                        acc_cell = frag_h_accs[kb][None, None, nr]
                        acc_cell.store(fx.Vector(acc_cell.load()) * exp_g_last)

            # Per-K decay: h[v, k] *= exp(gk_last[k]) at chunk end.
            if const_expr(USE_GK):
                for kb in range_constexpr(NUM_K_BLOCKS):
                    gk_elems = []
                    for elem_i in range_constexpr(4):
                        global_k = kb * 64 + wid * 16 + lane_m_base * 4 + elem_i
                        gk_raw = _load_vec(
                            cp_f32,
                            fx.slice(gk_view, (last_idx_raw, global_k, None)),
                            1,
                            fx.Float32,
                        )
                        gk_elems.append(_fast_exp(gk_raw))
                    gk_vec = fx.Vector.from_elements(gk_elems, dtype=fx.Float32)
                    for nr in range_constexpr(N_REPEAT):
                        acc_cell = frag_h_accs[kb][None, None, nr]
                        acc_cell.store(fx.Vector(acc_cell.load()) * gk_vec)

            gpu.barrier()

            # -- h store from XOR-swizzled transpose buffer (HIP-aligned:
            # after GEMM1, before GEMM2 -- mirrors HIP's
            # coalesced_vk_store_from_transpose called between run_gemm1 and
            # run_gemm2). The transpose buffer was populated during staging
            # above and is read-only hereafter; gated_v was written to a
            # different LDS region, so no conflict.
            K_VECS = K // LOAD_VEC_WIDTH
            NUM_HT_VECS = BV * K_VECS
            for vbase in range_constexpr(0, NUM_HT_VECS, BLOCK_THREADS):
                vec_idx = vbase + tid
                kv = vec_idx % K_VECS
                v_loc = vec_idx // K_VECS
                kg_lo = kv * 2
                val_lo = _lds_read_x4(fx.slice(sHT, (v_loc, kg_lo, None)))
                val_hi = _lds_read_x4(fx.slice(sHT, (v_loc, kg_lo + 1, None)))
                vec8 = val_lo.shuffle(val_hi, [0, 1, 2, 3, 4, 5, 6, 7])
                v_global = i_v * BV + v_loc
                _store_vec(
                    cp_bf16x8,
                    fx.slice(h_view, (i_t_i32, v_global, kv, None)),
                    vec8,
                    LOAD_VEC_WIDTH,
                    fx.BFloat16,
                )

            # -- GEMM2: h += k^T @ v_new_gated (no w prefetch/interleave).
            # The A operand comes from the rotating-pair k panel, which has no
            # layout form, so it is read by hand into frag_k; B is a normal
            # partitioned copy. Each 64-K block is its own M tile, hence its own
            # accumulator fragment. The hand-filled A is also why this stays a
            # per-k-tile loop rather than one whole-partition fx.gemm.
            BT_STEPS = BT // K_STEP
            for kb in range_constexpr(NUM_K_BLOCKS):
                for bt_s in range_constexpr(BT_STEPS):
                    for half in range_constexpr(K_STEP // 16):
                        kt = bt_s * (K_STEP // 16) + half
                        frag_k[None, None, kt].store(
                            _load_a_k_rotating(
                                kb * LDS_KP_PANEL_ELEMS,
                                wid * 16,
                                bt_s * K_STEP + half * 16,
                            )
                        )
                        fx.copy(
                            cp_lds_x4,
                            pS_gv[None, None, kt],
                            frag_gv_rt[None, None, kt],
                        )
                        fx.gemm(
                            tiled_mma,
                            frag_h_accs[kb],
                            frag_k[None, None, kt],
                            frag_gv[None, None, kt],
                            frag_h_accs[kb],
                        )

            # >>> WRITE prefetched w_next/k_next to LDS for next iteration.
            # Barrier ensures GEMM2 is done reading the old panels (HIP
            # run_gemm2_fulltile_bvp tail).
            has_next = next_i_t * BT < T_local
            if has_next:
                gpu.barrier()
                for pf_idx in range_constexpr(NUM_K_BLOCKS * NUM_LOAD_BATCHES_64):
                    wvec_pf = w_next_vecs[pf_idx]
                    kb_pf, row_pf = w_next_coord[pf_idx]
                    _lds_write_x4(
                        fx.slice(sW, (row_pf, (None, load_col_group, kb_pf))),
                        wvec_pf.shuffle(wvec_pf, [0, 1, 2, 3]),
                    )
                    _lds_write_x4(
                        fx.slice(sW, (row_pf, (None, load_col_group + 1, kb_pf))),
                        wvec_pf.shuffle(wvec_pf, [4, 5, 6, 7]),
                    )
                for kb in range_constexpr(NUM_K_BLOCKS):
                    kp_pbase = kb * LDS_KP_PANEL_ELEMS
                    kvec_t0_pf = k_next_vecs_t0[kb]
                    kvec_t1_pf = k_next_vecs_t1[kb]
                    for i in range_constexpr(LOAD_VEC_WIDTH):
                        row_i = k_row_base_pf + i
                        byte_off = _k_panel_rotating_pair_addr_bytes(
                            row_i, k_pair_col_pf
                        )
                        elem_off = byte_off >> 1
                        fx.ptr_store(kvec_t0_pf[i], lds_kp_ptr + (kp_pbase + elem_off))
                        fx.ptr_store(
                            kvec_t1_pf[i],
                            lds_kp_ptr + (kp_pbase + elem_off + 1),
                        )

            results = yield [
                frag_h_accs[kb].load() for kb in range_constexpr(NUM_K_BLOCKS)
            ]

        for kb in range_constexpr(NUM_K_BLOCKS):
            frag_h_accs[kb].store(fx.Vector(results[kb], (H_ACC_ELEMS,), fx.Float32))

        # -- Epilogue: store final state --
        # acc_val is already f32x4 with element i at K offset i -> store the
        # vector directly (no extract + from_elements needed), giving one
        # buffer_store_dwordx4 instead of 4 scalar f32 stores.
        if const_expr(STORE_FINAL_STATE):
            for kb in range_constexpr(NUM_K_BLOCKS):
                for slot in range_constexpr(N_REPEAT):
                    acc_val = fx.Vector(frag_h_accs[kb][None, None, slot].load())

                    ht_col = i_v * BV + (slot * 16) + lane_n
                    ht_kgroup = kb * 16 + wid * 4 + lane_m_base
                    if const_expr(STATE_DTYPE_BF16):
                        out_vec = _f32x4_to_bf16x4(acc_val)
                    else:
                        out_vec = acc_val
                    _store_vec(
                        cp_state_x4,
                        fx.slice(ht_view, (ht_col, ht_kgroup, None)),
                        out_vec,
                        4,
                        state_num,
                    )

    # -- Host launcher ------------------------------------------------------
    @flyc.jit
    def launch_gdn_h(
        k_tensor: fx.Tensor,
        v_tensor: fx.Tensor,
        w_tensor: fx.Tensor,
        v_new_tensor: fx.Tensor,
        g_tensor: fx.Tensor,
        gk_tensor: fx.Tensor,
        h_tensor: fx.Tensor,
        h0_tensor: fx.Tensor,
        ht_tensor: fx.Tensor,
        cu_seqlens_tensor: fx.Tensor,
        chunk_offsets_tensor: fx.Tensor,
        state_indices_tensor: fx.Tensor,
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
