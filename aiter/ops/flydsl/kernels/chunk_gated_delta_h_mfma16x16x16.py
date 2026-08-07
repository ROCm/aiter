# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Gated Delta Net K5 hidden-state recurrence kernel using the @flyc.kernel API.

HIP-aligned fork: same instruction as the hand-tuned HIP/C++ K5 kernel
(``mfma_f32_16x16x16bf16_1k``) and the same warp partition (BT split-M, K split
across waves, V not split across warps). Writes the public VK layout
[..., V, K] through a [V][K] transpose buffer + b128 store.

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

_LOG2E = math.log2(math.e)


def _gview(tensor, base, shape, stride):
    """Buffer-resource view of the global slot ``tensor``, rooted at ELEMENT
    offset ``base`` (None = tensor origin). The slot's own memref layout is
    discarded and replaced by ``shape`` / ``stride``.

    ``base`` is the per-block runtime scalar (varlen / head / state-slot origin)
    that carries no tile structure, so it stays an iterator shift while
    ``shape`` / ``stride`` describe the part addressed by coordinate. The
    innermost mode is always one vector access, so slicing it off with ``None``
    yields exactly the copy-atom tile.
    """
    it = fx.get_iter(fx.rocdl.make_buffer_tensor(tensor, max_size=True))
    if base is not None:
        it = fx.add_offset(it, base)
    return fx.Tensor(fx.make_view(it, fx.make_layout(shape, stride)))


def _load_vec(atom, tile, width, numeric):
    """Load ``width`` elements from the coordinate ``tile``. ``atom`` picks the
    access: buffer_load from a global view, ds_read from an LDS one. The
    staging fragment is folded away by fly-promote-regmem-to-vectorssa."""
    frag = fx.make_rmem_tensor(width, numeric)
    fx.copy(atom, tile, frag)
    vec = frag.load()
    return vec[0] if width == 1 else vec


def _store_vec(atom, tile, value, width, numeric):
    """Inverse of ``_load_vec``: store ``value`` into the coordinate ``tile``."""
    frag = fx.make_rmem_tensor(width, numeric)
    frag.store(fx.Vector.from_elements([value], dtype=numeric) if width == 1 else value)
    fx.copy(atom, frag, tile)


def _make_fast_exp(g_is_log2_scaled: bool):
    """``exp(x)`` lowered via ``exp2``; the ``* LOG2E`` is dropped when ``g`` is
    already log2(e)-prescaled.

    Raw ``rocdl.exp2`` rather than the ``fx.exp2`` wrapper: both lower to
    ``v_exp_f32``, but ``fx.exp2`` goes through the math dialect, whose
    IEEE-correct lowering wraps every call in a denormal rescale guard this
    kernel does not need (the gates underflow to 0 either way). Measured on
    gfx942 over the 5 exp sites: 819 -> 849 instructions, 146 -> 148 VGPRs.
    The intrinsic is untyped, hence the ``ir_value()`` / ``fx.Float32`` hop.
    """
    if g_is_log2_scaled:
        return lambda x: fx.Float32(rocdl.exp2(T.f32, x.ir_value()))
    return lambda x: fx.Float32(rocdl.exp2(T.f32, (x * _LOG2E).ir_value()))


# Only the default for the ``BF16_CONVERT_TRUNC`` compile option; truncation
# keeps outputs bit-identical to the HIP/C++ K5 kernel, RNE is ~0.5 ulp closer
# to the FP32 reference but does NOT bit-match HIP.
_BF16_CONVERT_TRUNC_DEFAULT = True


def _make_bf16_converter(trunc: bool):
    """Return the fp32x4 -> bf16x4 output converter for this compile.

    ``trunc=True`` keeps the high 16 bits with no rounding bias, matching HIP
    ``float_to_bf16`` (``bit_cast<u32>(x) >> 16``) and arch-neutral.
    ``trunc=False`` rounds to nearest even, which LLVM lowers to the native
    ``v_cvt_pk_bf16_f32`` on gfx950 and to a software RNE sequence on gfx942 --
    both bit-exact with torch/HIP ``__float2bfloat16_rn``.
    """
    if trunc:
        return lambda v: (
            (v.bitcast(fx.Uint32) >> 16).to(fx.Uint16).bitcast(fx.BFloat16)
        )
    return lambda v: v.to(fx.BFloat16)


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

    ``STATE_DTYPE_BF16`` switches the SSM state tensors ``h0`` / ``ht`` from
    f32 to bf16, promoting on load and demoting on store. The f32 accumulator
    and every intermediate LDS layout are unchanged, so it only affects HBM
    bandwidth / footprint of the state.
    """
    # The wave mapping (wid*16, 4 waves cover 64 rows), the cooperative-load
    # batching and BT_STEPS all hardcode BT=64, gated_v alias-reuses h_state
    # panel1 (which does not exist at K=64 -> out-of-bounds store), and the LDS
    # layout is only validated at K=V=128. Reject anything else explicitly
    # rather than silently producing wrong results.
    assert BT == 64, f"chunk_gated_delta_h_mfma16_hip only supports BT=64, got BT={BT}"
    assert K == 128, f"chunk_gated_delta_h_mfma16_hip only supports K=128, got K={K}"
    assert BV % 16 == 0, f"BV must be a multiple of the MFMA N of 16, got BV={BV}"
    NUM_K_BLOCKS = K // 64

    # Every slot's shape/stride is re-described by ``_gview``, so only its
    # memref ELEMENT TYPE is read from the incoming tensor: the placeholder
    # tensors the host passes for unused slots must still match the dtype the
    # body assumes. Buffer descriptors are ``max_size`` (unbounded) because the
    # in-kernel clamps (safe_row / in_bounds selects) already keep every access
    # in range.

    _fast_exp = _make_fast_exp(G_IS_LOG2_SCALED)
    _f32x4_to_bf16x4 = _make_bf16_converter(BF16_CONVERT_TRUNC)

    WARP_SIZE = 64
    NUM_WARPS = 4
    BLOCK_THREADS = NUM_WARPS * WARP_SIZE

    # MFMA tile is 16x16x16; each k-step issues a lo/hi MFMA pair and so
    # advances K by 32 (K_STEP), which is why K_STEP != the MFMA K of 16.
    MFMA_N = 16
    K_STEP = 32
    N_REPEAT = BV // MFMA_N

    # w panels, one [BT][64] per 64-K block, written with HIP's
    # ``w_panel_swizzle`` and read by GEMM1 as a plain contiguous-16-K b64.
    LDS_WP_PANEL_ELEMS = BT * 64
    LDS_WP_ELEMS = NUM_K_BLOCKS * LDS_WP_PANEL_ELEMS

    # k in rotating-pair swizzled panels, one [64 K-rows][BT tokens] per 64-K
    # block. Mirrors HIP's ``k_panel_rotating_pair_addr_bytes``: the global load
    # reads b128 for an adjacent token pair and scatters b16x2 writes to
    # swizzled addresses, so GEMM2 can read the MFMA A frag as a plain b64
    # (no ds_read_tr16).
    LDS_KP_PANEL_ELEMS = 64 * BT
    LDS_KP_ELEMS = NUM_K_BLOCKS * LDS_KP_PANEL_ELEMS

    # gated v_new in a shared2 panel; GEMM2 B read is contiguous over BT so it
    # matches the k A frag.
    LDS_GV_ELEMS = (BT // 4) * BV * 4  # 16 bt_groups x BV cols x 4 BT

    # h_state panels (shared2), one [row_block][V] per 64-K block, so GEMM1
    # reads the MFMA B fragment as a plain b64 instead of ds_read_tr16. Each
    # cell holds one k_group of 4 K: 64/4=16 row_blocks x BV cols x 4 K.
    LDS_HP_PANEL_ELEMS = (64 // 4) * BV * 4  # = 64 * BV per 64-K block
    LDS_HP_ELEMS = NUM_K_BLOCKS * LDS_HP_PANEL_ELEMS
    # gated_v aliases h_state panel 1, so it must fit that panel exactly.
    assert LDS_GV_ELEMS == LDS_HP_PANEL_ELEMS, (
        f"gated_v aliases h_state panel 1 and must match it exactly, got "
        f"{LDS_GV_ELEMS} vs {LDS_HP_PANEL_ELEMS}"
    )

    # [V][K] transpose buffer for the h snapshot HBM store: h_accs is staged
    # V-major / K-innermost so 8 adjacent K land contiguously and the readout
    # becomes one ds_read_b128 + one buffer_store b128 instead of per-element
    # stores. K-contiguous (no pad) keeps the 8-wide access 16 B aligned.
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
        # Only bounds the cu_seqlens / chunk_offsets / state_indices views; the
        # sequence count itself is carried by the grid.y extent.
        N_val: fx.Int32,
    ):
        i_v = fx.block_idx.x
        i_nh = fx.block_idx.y
        i_n = i_nh // H
        i_h = i_nh % H

        # Buffer-copy atoms shared by every global access below: 128b/64b for
        # the cooperative vector loads, 32b/16b for the per-lane scalars.
        cp_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        cp_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        cp_bf16 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        cp_bf16x8 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)

        # State-pool gather: the SSM slot is ``state_indices[i_n]`` into a
        # ``[pool_size, H, V, K]`` pool instead of ``i_n`` into a dense
        # ``[N, H, V, K]``. Only h0 / ht use it; the h snapshot stays dense.
        if const_expr(USE_STATE_INDICES):
            si_view = _gview(state_indices_tensor, None, (N_val, 1), (1, 1))
            state_n = _load_vec(cp_i32, fx.slice(si_view, (i_n, None)), 1, fx.Int32)
        else:
            state_n = i_n
        state_nh = state_n * H + i_h

        tid = fx.thread_idx.x
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
        # out of partition_S once, outside the chunk loop. The K mode is nested
        # ``(4, k_groups, panels)``: the innermost 4 elements are one k_group,
        # which is exactly one MFMA operand and one b64 access. All swizzles use
        # base=2 (elements -> 8 B) so those 4 stay contiguous.
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        KG_PER_BLOCK = 64 // 4  # k_groups per 64-K panel

        # GEMM1 A -- w panels as (BT, K): [BT][64] row-major + S<4,2,4>, the
        # layout form of HIP's ``w_panel_swizzle`` (verified bit-exact on all
        # BT*64 addresses). The panel stride is far above the swizzled bits, so
        # both panels share one view.
        sW = lds.wp.view(
            fx.make_composed_layout(
                fx.static(fx.SwizzleType.get(4, 2, 4)),
                fx.make_layout(
                    (BT, (4, KG_PER_BLOCK, NUM_K_BLOCKS)),
                    (64, (1, 4, LDS_WP_PANEL_ELEMS)),
                ),
            ),
        )
        # GEMM1 B -- h_state panels (shared2) as (BV, K). No swizzle needed.
        sH = lds.hp.view(
            fx.make_layout(
                (BV, (4, KG_PER_BLOCK, NUM_K_BLOCKS)),
                (4, (1, BV * 4, LDS_HP_PANEL_ELEMS)),
            )
        )
        # GEMM2 B -- gated_v as (BV, BT), same shared2 cell layout. It ALIASES
        # h_state panel 1 to avoid a separate LDS buffer that would cost an
        # occupancy step; the write happens only after GEMM1 has finished
        # reading the panels, enforced by a WAR barrier before the store.
        sGV = fx.make_view(
            lds.hp.ptr + LDS_HP_PANEL_ELEMS,
            fx.make_layout((BV, (4, BT // 4)), (4, (1, BV * 4))),
        )

        # GEMM2 A -- k panels as (64, BT), one panel being k^T for a 64-K block:
        # [64][BT] row-major + S<2,2,8> + S<4,2,4>, the layout form of HIP's
        # ``k_panel_rotating_pair_addr_bytes`` (verified bit-exact on all 64*BT
        # addresses of both panels).
        #
        # k is the only panel written along one axis and read along the other,
        # and both sides must stay conflict-free: the store varies K-row bits
        # 3..5 across lanes while the MFMA A read varies bits 0..3, so all six
        # row bits fold into the token field. Only address bits 3..6 are
        # available (bit 2 must stay put, or a k_group's 4 elements stop being
        # contiguous and the b64 operand read is lost), so 6 bits fold into 4
        # and the XOR windows overlap -- the "rotating" in the HIP name. One
        # SwizzleType is a single XOR of one window at one shift, hence two:
        # row bits 0..3 at shift 4, bits 4..5 at shift 8.
        #
        # The panel base stays an iterator shift because 64*BT is a multiple of
        # both swizzle periods, so swizzle(base + x) == base + swizzle(x) and
        # one layout serves both panels.
        #
        # NOTE: index this view with fx.slice, NOT partition_S /
        # make_tiled_copy. The partition path mishandles a NESTED composed
        # layout and silently produces wrong addresses (single-swizzle views
        # are fine either way), hence the explicit per-k-tile fx.slice below.
        def _k_panel_view(kb, group):
            """k panel ``kb`` as (64, (group, BT/group)) -- same addresses for
            every ``group``, only the token mode is regrouped: ``group=4`` is
            the MFMA A operand (one k_group per b64), ``group=2`` the
            cooperative store (one token pair per b32)."""
            return fx.make_view(
                lds.kp.ptr + kb * LDS_KP_PANEL_ELEMS,
                fx.make_composed_layout(
                    fx.static(fx.SwizzleType.get(4, 2, 4)),
                    fx.make_composed_layout(
                        fx.static(fx.SwizzleType.get(2, 2, 8)),
                        fx.make_layout((64, (group, BT // group)), (BT, (1, group))),
                    ),
                ),
            )

        sK = [_k_panel_view(kb, 4) for kb in range(NUM_K_BLOCKS)]
        sKw = [_k_panel_view(kb, 2) for kb in range(NUM_K_BLOCKS)]
        # [V][K] transpose buffer: [BV][K] row-major + S<4,2,5>, the layout form
        # of the hand-written ``k_group ^ (v & 0xF)`` scatter. Not an MMA
        # operand, so it keeps a plain (V, k_group, 4) shape.
        sHT = lds.ht.view(
            fx.make_composed_layout(
                fx.static(fx.SwizzleType.get(4, 2, 5)),
                fx.make_layout((BV, K // 4, 4), (K, 4, 1)),
            )
        )

        # Every LDS <-> register move here is a single k_group (4 bf16 = one
        # b64), so these wrappers pin the atom / width / dtype and leave only
        # the tile.
        cp_lds_x4 = fx.make_copy_atom(fx.UniversalCopy64b(), fx.BFloat16)

        def _lds_read_x4(tile):
            return _load_vec(cp_lds_x4, tile, 4, fx.BFloat16)

        def _lds_write_x4(tile, vec4):
            _store_vec(cp_lds_x4, tile, vec4, 4, fx.BFloat16)

        cp_lds_x2 = fx.make_copy_atom(fx.UniversalCopy32b(), fx.BFloat16)

        # Scatter the adjacent token pair (2*pc, 2*pc+1) of K-row ``row`` in
        # panel ``kb``, through the same swizzled layout the GEMM2 A read uses.
        def _k_panel_store_pair(kb, row, pc, v0, v1):
            _store_vec(
                cp_lds_x2,
                fx.slice(sKw[kb], (row, (None, pc))),
                fx.Vector.from_elements([v0, v1], dtype=fx.BFloat16),
                2,
                fx.BFloat16,
            )

        # 16x16x16 bf16 MFMA (K-tile 16, so A/B are bf16x4 -- one b64 read --
        # and C is f32x4), tiled over the 4 waves along M. That atom's TV layout
        # is exactly this kernel's wave/lane mapping: A gives lane l element
        # A[wid*16 + l%16][(l//16)*4 + v], B gives B[l%16][(l//16)*4 + v] with
        # a zero wave stride (B is shared by all 4 waves), and C gives
        # C[wid*16 + (l//16)*4 + v][l%16] -- which is what the operand
        # addresses and the h/v staging below assume.
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16, fx.Float32))
        tiled_mma = fx.make_tiled_mma(
            mma_atom, fx.make_layout((NUM_WARPS, 1, 1), (1, 0, 0))
        )
        thr_cp_a = fx.make_tiled_copy_A(cp_lds_x4, tiled_mma).get_slice(tid)
        thr_cp_b = fx.make_tiled_copy_B(cp_lds_x4, tiled_mma).get_slice(tid)

        # Per-lane source partitions: the whole swizzle + lane mapping folded
        # into one address per K-tile, computed once instead of per access.
        pS_w = thr_cp_a.partition_S(sW)  # GEMM1 A: ((4,1), 1, (4, NUM_K_BLOCKS))
        pS_h = thr_cp_b.partition_S(sH)  # GEMM1 B: ((4,1), N_REPEAT, K_TILES)
        pS_gv = thr_cp_b.partition_S(sGV)  # GEMM2 B: ((4,1), N_REPEAT, BT_TILES)

        # ``retile`` gives the copy-side view of the same registers (what
        # fx.copy writes into); fx.gemm consumes the mma-side view. A's K mode
        # stays nested as (k_tile_in_panel, panel) while the copy/B side is
        # flat: ``k_tile = k_tile_in_panel + panel * K_TILES_PER_BLOCK``.
        K_TILES_PER_BLOCK = 64 // 16
        frag_w = tiled_mma.make_fragment_A(sW)
        frag_h = tiled_mma.make_fragment_B(sH)
        frag_gv = tiled_mma.make_fragment_B(sGV)
        frag_k = [tiled_mma.make_fragment_A(v) for v in sK]
        frag_w_rt = thr_cp_a.retile(frag_w)
        frag_h_rt = thr_cp_b.retile(frag_h)
        frag_gv_rt = thr_cp_b.retile(frag_gv)
        # Accumulators: b_v (GEMM1) over the full (BT, BV) tile, and one h state
        # accumulator per 64-K block (GEMM2's M tile is a 64-K block).
        #
        # A fragment's ``load()`` keeps the partition's NESTED shape (e.g.
        # ((4,1),1,1)) while every vector built by hand below is flat (4,).
        # Vector arithmetic broadcasts on that Python-side shape, so mixing the
        # two expands 4 elements to 16 and raises. ``fx.Vector(frag.load())``
        # re-derives the shape from the IR type: a reshape, NOT a redundant
        # wrap -- do not strip it.
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

        # Cooperative load decomposition. load_vec_group is the index of this
        # thread's LOAD_VEC_WIDTH-wide K vector within a 64-K block, i.e. the
        # global-view column coordinate.
        load_row_in_batch = tid // THREADS_PER_ROW_64
        load_vec_group = tid % THREADS_PER_ROW_64
        load_col_group = load_vec_group * (LOAD_VEC_WIDTH // 4)

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

        # A row past the sequence end is clamped to row 0 rather than masked:
        # the buffer descriptors are max_size, so this clamp is what keeps the
        # w / k / u accesses in range. The padding rows it aliases contribute
        # nothing downstream, being zeroed by the gate.
        def _clamp_row(row):
            return (row < T_local).select(row, 0)

        # -- Global tensor views --
        # Bases stay i64 because long-context snapshot tensors can exceed 2^31
        # elements; the coordinates are per-chunk and fit in i32.
        i_n64 = fx.Int64(i_n)
        i_h64 = fx.Int64(i_h)
        bos64 = fx.Int64(bos)
        boh64 = fx.Int64(boh)
        t_flat64 = fx.Int64(T_flat)

        VEC = LOAD_VEC_WIDTH

        # h: [B, NT, H, V, K] (VK) -- base = (boh*H + i_h) * V * K
        h_view = _gview(
            h_tensor,
            (boh64 * H + i_h64) * (V * K),
            (NT, V, K // VEC, VEC),
            (H * V * K, K, VEC, 1),
        )

        # k: [B, T, Hg, K] -- base = (bos*Hg + i_h//(H//Hg)) * K
        gqa_ratio = H // Hg
        k_view = _gview(
            k_tensor,
            (bos64 * Hg + fx.Int64(i_h // gqa_ratio)) * K,
            (T_local, K // VEC, VEC),
            (Hg * K, VEC, 1),
        )

        if const_expr(WU_CONTIGUOUS):
            if const_expr(IS_VARLEN):
                v_base = (i_h64 * t_flat64 + bos64) * V
                w_base = (i_h64 * t_flat64 + bos64) * K
            else:
                v_base = ((i_n64 * H + i_h64) * t_flat64) * V
                w_base = ((i_n64 * H + i_h64) * t_flat64) * K
            stride_v = V
            stride_w = K
        else:
            v_base = (bos64 * H + i_h64) * V
            w_base = (bos64 * H + i_h64) * K
            stride_v = H * V
            stride_w = H * K
        w_view = _gview(w_tensor, w_base, (T_local, K // VEC, VEC), (stride_w, VEC, 1))
        # u and v_new are read/written one element at a time, so their innermost
        # mode is a single element rather than a vector.
        u_view = _gview(v_tensor, v_base, (T_local, V, 1), (stride_v, 1, 1))

        if const_expr(IS_VARLEN):
            vn_base = (i_h64 * t_flat64 + bos64) * V
        else:
            vn_base = ((i_n64 * H + i_h64) * t_flat64) * V
        vn_view = _gview(v_new_tensor, vn_base, (T_local, V, 1), (V, 1, 1))

        # h0/ht: the [V, K] state slot for this (state, head), K split into the
        # 4 contiguous elements of one state vector access.
        state_slot_base = fx.Int64(state_nh) * (V * K)
        if const_expr(USE_INITIAL_STATE):
            h0_view = _gview(h0_tensor, state_slot_base, (V, K // 4, 4), (K, 4, 1))
        if const_expr(STORE_FINAL_STATE):
            ht_view = _gview(ht_tensor, state_slot_base, (V, K // 4, 4), (K, 4, 1))

        # g gate offset = i_n*g_stride_b + i_h*g_stride_h + token*g_stride_t,
        # with the strides read off the layout:
        #   head-major  [B, H, T_flat] -> g_stride_h=T_flat, g_stride_t=1
        #   token-major [B, T_flat, H] -> g_stride_h=1,      g_stride_t=H
        # varlen is flattened to B==1, so the batch stride is 0 and the token is
        # global -- bos*g_stride_t folds into the base. Dense keeps a H*T_flat
        # batch stride and an in-sequence relative token. Either way the base
        # excludes the per-token term, which is the view's leading coordinate.
        if const_expr(USE_G):
            if const_expr(G_HEAD_MAJOR):
                g_stride_h = t_flat64
                g_stride_t = 1
            else:
                g_stride_h = 1
                g_stride_t = H
            if const_expr(IS_VARLEN):
                g_sh_base = i_h64 * g_stride_h + bos64 * g_stride_t
            else:
                g_sh_base = i_n64 * H * t_flat64 + i_h64 * g_stride_h
            g_view = _gview(g_tensor, g_sh_base, (T_local, 1), (g_stride_t, 1))

        # gk: [B, T, H, K] token-major. The chunk's last token is a coordinate
        # rather than part of the base, so the view hoists out of the loop.
        if const_expr(USE_GK):
            gk_view = _gview(
                gk_tensor,
                bos64 * (H * K) + i_h64 * K,
                (T_local, K, 1),
                (H * K, 1, 1),
            )

        # -- MFMA lane mapping for 16x16 tiles --
        lane_n = lane % 16
        lane_m_base = lane // 16
        # The k_group this lane owns within a 64-K panel; the 4 warps together
        # cover all 16 row_blocks.
        row_block = wid * 4 + lane_m_base

        # The [V, K] state cell of v-repeat ``slot`` in 64-K block ``kb``. h0
        # load and ht store must address the same cell, so they share this.
        def _state_coord(kb, slot):
            return i_v * BV + slot * 16 + lane_n, kb * 16 + row_block

        # One accumulator fragment per 64-K block; the per-lane element for
        # v-repeat ``nr`` is ``frag[None, None, nr]``.
        for kb in range_constexpr(NUM_K_BLOCKS):
            frag_h_accs[kb].fill(0.0)

        # h0 is [V, K] so K is innermost and 4 consecutive K are contiguous ->
        # one buffer_load_dwordx4 instead of 4 scalar loads.
        if const_expr(USE_INITIAL_STATE):
            for kb in range_constexpr(NUM_K_BLOCKS):
                for slot in range_constexpr(N_REPEAT):
                    h0_col, h0_kgroup = _state_coord(kb, slot)
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

        # Pipelined main chunk loop: the prologue stages chunk 0's w/k in LDS,
        # then each iteration prefetches the NEXT chunk's w (during GEMM1) and k
        # (after gating) into VGPRs and writes them to LDS at the GEMM2 end.
        # GEMM1 k-blocks issued before the w_next prefetch; the remaining
        # NUM_K_BLOCKS - GEMM1_PF_SPLIT hide that prefetch's HBM latency.
        GEMM1_PF_SPLIT = 1
        k_vec_group_pf = lane & 7
        k_row_base_pf = k_vec_group_pf * 8
        k_pair_col_pf = wid * 8 + (lane >> 3)
        k_t0_pf = k_pair_col_pf * 2
        k_t1_pf = k_t0_pf + 1

        # Cooperative staging of one chunk's w / k. Load and store are split so
        # the main loop can issue the loads early (hidden behind GEMM1 / the h
        # store) and publish to LDS only at the GEMM2 end, while the prologue
        # runs the two back to back. ``row_base=None`` is chunk 0, where the
        # absolute row IS the in-chunk row -- passing 0 would emit a dead add.
        def _load_w_rows(row_base=None):
            rows = []
            for kb in range_constexpr(NUM_K_BLOCKS):
                for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                    row = batch * ROWS_PER_BATCH_64 + load_row_in_batch
                    abs_row = row if row_base is None else row_base + row
                    wvec = _load_vec(
                        cp_bf16x8,
                        fx.slice(
                            w_view, (_clamp_row(abs_row), kb * 8 + load_vec_group, None)
                        ),
                        LOAD_VEC_WIDTH,
                        fx.BFloat16,
                    )
                    rows.append((kb, row, wvec))
            return rows

        def _store_w_rows(rows):
            for kb, row, wvec in rows:
                _lds_write_x4(
                    fx.slice(sW, (row, (None, load_col_group, kb))),
                    wvec.shuffle(wvec, [0, 1, 2, 3]),
                )
                _lds_write_x4(
                    fx.slice(sW, (row, (None, load_col_group + 1, kb))),
                    wvec.shuffle(wvec, [4, 5, 6, 7]),
                )

        def _load_k_pairs(row_base=None):
            t0 = k_t0_pf if row_base is None else row_base + k_t0_pf
            t1 = k_t1_pf if row_base is None else row_base + k_t1_pf
            safe_t0 = _clamp_row(t0)
            safe_t1 = _clamp_row(t1)
            pairs = []
            for kb in range_constexpr(NUM_K_BLOCKS):
                k_vec_col = kb * 8 + k_vec_group_pf
                kvec_t0 = _load_vec(
                    cp_bf16x8,
                    fx.slice(k_view, (safe_t0, k_vec_col, None)),
                    LOAD_VEC_WIDTH,
                    fx.BFloat16,
                )
                kvec_t1 = _load_vec(
                    cp_bf16x8,
                    fx.slice(k_view, (safe_t1, k_vec_col, None)),
                    LOAD_VEC_WIDTH,
                    fx.BFloat16,
                )
                pairs.append((kb, kvec_t0, kvec_t1))
            return pairs

        def _store_k_pairs(pairs):
            for kb, kvec_t0, kvec_t1 in pairs:
                for i in range_constexpr(LOAD_VEC_WIDTH):
                    _k_panel_store_pair(
                        kb, k_row_base_pf + i, k_pair_col_pf, kvec_t0[i], kvec_t1[i]
                    )

        c_zero = fx.Int64(0)
        c_one = fx.Int64(1)
        nt_idx = fx.Int64(NT)

        # PROLOGUE: stage chunk 0's w/k in LDS, under a block-uniform NT>0
        # guard. An empty varlen sequence (T_local==0) would clamp safe_row to 0
        # while its bos is already T_flat, so the w/k address would run past the
        # end of the input buffer; skipping the whole prologue (barrier
        # included) is also correct because the main loop then runs 0 times and
        # h0 simply passes through to ht.
        #
        # The staging helpers are closures so the ``if`` holds only calls and a
        # barrier: the views and copy atoms stay free variables instead of
        # appearing inside the if, where the AST rewriter would try to carry
        # them as scf.if state.
        has_work = NT > 0
        if has_work:
            _store_w_rows(_load_w_rows())
            _store_k_pairs(_load_k_pairs())
            gpu.barrier()

        # The accumulator fragments are the loop-carried state; scf.for wants
        # plain values, so each fragment enters and leaves the body as one
        # flat vector.
        h_accs_init = [frag_h_accs[kb].load() for kb in range_constexpr(NUM_K_BLOCKS)]
        H_ACC_ELEMS = N_REPEAT * 4

        for i_t, state in range(c_zero, nt_idx, c_one, init=h_accs_init):
            for kb in range_constexpr(NUM_K_BLOCKS):
                frag_h_accs[kb].store(fx.Vector(state[kb], (H_ACC_ELEMS,), fx.Float32))
            # ``i_t`` is the raw scf.for induction variable (untyped i64), so it
            # has to be wrapped before any arithmetic; chunk-local coordinates
            # take the i32 narrowing and the one i64 use (the h snapshot offset)
            # re-wraps at its use site.
            i_t_i32 = fx.Int32(i_t)

            # Absolute token row of MFMA C element ``elem_i`` for this lane:
            # C hands lane l the rows wid*16 + (l//16)*4 + elem_i (see the
            # mma_atom comment), and the chunk contributes i_t*BT. The chunk
            # index is bound as a default argument because the closure is
            # defined inside the chunk loop (ruff B023); it is only ever called
            # while tracing that same iteration.
            def _bt_abs_row(elem_i, i_t_i32=i_t_i32):
                return i_t_i32 * BT + wid * 16 + lane_m_base * 4 + elem_i

            # Stage h_accs into (a) the h_state panels for the GEMM1 B operand
            # and (b) the [V][K] transpose buffer for the b128 HBM store.
            # split-M mapping: wid -> K sub-tile, acc_j -> V-tile.
            for kb in range_constexpr(NUM_K_BLOCKS):
                for acc_j in range_constexpr(N_REPEAT):
                    acc_bf16 = _f32x4_to_bf16x4(
                        fx.Vector(frag_h_accs[kb][None, None, acc_j].load())
                    )
                    hp_col = acc_j * 16 + lane_n
                    _lds_write_x4(
                        fx.slice(sH, (hp_col, (None, row_block, kb))), acc_bf16
                    )

                    # The sHT swizzle breaks bank conflicts on this scatter
                    # write while keeping a k_group contiguous -> one b64.
                    _lds_write_x4(
                        fx.slice(sHT, (hp_col, kb * 16 + row_block, None)), acc_bf16
                    )

            # w/k for this chunk already in LDS (prologue or prev GEMM2 end).
            gpu.barrier()

            next_chunk_end = (i_t_i32 + 1) * BT
            last_idx_raw = (next_chunk_end < T_local).select(
                next_chunk_end, T_local
            ) - 1

            # Issue every u / g HBM load before GEMM1 so the full 64-MFMA chain
            # hides their latency. Left to itself LLVM sinks them into the
            # middle of GEMM1, where only ~2 MFMA can cover them -- measured at
            # a 4.2M cycle vmcnt stall, 39% of all stalls.
            u_cols = [i_v * BV + idx * 16 + lane_n for idx in range(N_REPEAT)]
            u_prefetch = []  # N_REPEAT x 4 bf16 scalars
            for idx in range_constexpr(N_REPEAT):
                for elem_i in range_constexpr(4):
                    safe_u_row = _clamp_row(_bt_abs_row(elem_i))
                    u_prefetch.append(
                        _load_vec(
                            cp_bf16,
                            fx.slice(u_view, (safe_u_row, u_cols[idx], None)),
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
                    abs_row = _bt_abs_row(elem_i)
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

            # -- GEMM1: b_v = w @ h_state, with the w_next prefetch interleaved.
            frag_bv.fill(0.0)

            # One 64-K block, each fx.gemm covering one 16-K tile over all
            # N_REPEAT v-tiles. The per-k-tile loop must NOT be folded into a
            # single whole-partition fx.copy + fx.gemm: the K-tiles have to stay
            # individually addressable so the w_next prefetch can split the
            # k-blocks (GEMM1_PF_SPLIT) and the sched_barrier can land on every
            # ks boundary.
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
                    # gfx942: a sched_barrier at each ks boundary stops LLVM
                    # from hoisting the per-ks b_frag ds_read across iterations
                    # into one cluster (the main source of LDS port
                    # back-pressure), while mask_mfma still lets MFMA overlap
                    # across ks. It reorders instructions only -- no address or
                    # value changes -- so it is correctness-safe.
                    if const_expr(SCHED_GFX942):
                        rocdl.sched_barrier(rocdl.mask_mfma)

            for kb in range_constexpr(GEMM1_PF_SPLIT):
                _gemm1_kblock(kb)

            # Prefetch the next chunk's w.
            next_i_t = i_t_i32 + 1
            next_chunk_base = next_i_t * BT
            w_next = _load_w_rows(next_chunk_base)

            # Remaining K-blocks; their MFMA hide the u/g/w_next HBM latency.
            for kb in range_constexpr(GEMM1_PF_SPLIT, NUM_K_BLOCKS):
                _gemm1_kblock(kb)

            # WAR barrier: GEMM1 is done reading the h_state panels, so gated_v
            # may now overwrite panel 1.
            gpu.barrier()

            # -- FUSED v_new + gating + gated_v store, consuming the u/g already
            # prefetched into VGPRs --
            if const_expr(USE_G):
                exp_g_last = _fast_exp(g_last)
                gate_elems = []
                for elem_i in range_constexpr(4):
                    g_row_val, in_bounds = g_row_pf[elem_i]
                    gate = _fast_exp(g_last - g_row_val)
                    gate_elems.append(in_bounds.select(gate, fx.Float32(0.0)))
                gate_vec = fx.Vector.from_elements(gate_elems, dtype=fx.Float32)
            else:
                # The last chunk's padding rows still have to be masked, or
                # those invalid tokens' v_new flows through gated_v into GEMM2
                # and corrupts the state update. The USE_G path gets this for
                # free via gate=0; here a 0/1 mask does the same.
                mask_elems = []
                for elem_i in range_constexpr(4):
                    in_bounds = _bt_abs_row(elem_i) < T_local
                    mask_elems.append(
                        in_bounds.select(fx.Float32(1.0), fx.Float32(0.0))
                    )
                gate_vec = fx.Vector.from_elements(mask_elems, dtype=fx.Float32)

            for idx in range_constexpr(N_REPEAT):
                bv_val = fx.Vector(frag_bv[None, None, idx].load())
                u_f32_elems = []
                for elem_i in range_constexpr(4):
                    u_f32_elems.append(u_prefetch[idx * 4 + elem_i].to(fx.Float32))
                u_f32 = fx.Vector.from_elements(u_f32_elems, dtype=fx.Float32)
                vn_val = u_f32 - bv_val

                if const_expr(SAVE_NEW_VALUE):
                    vn_bf16 = _f32x4_to_bf16x4(vn_val)
                    for elem_i in range_constexpr(4):
                        vn_bt_row = _bt_abs_row(elem_i)
                        if vn_bt_row < T_local:
                            bf16_v = vn_bf16[elem_i]
                            _store_vec(
                                cp_bf16,
                                fx.slice(vn_view, (vn_bt_row, u_cols[idx], None)),
                                bf16_v,
                                1,
                                fx.BFloat16,
                            )

                # gate_vec is the decay gate with the OOB mask folded in, or a
                # pure 0/1 padding mask -- either way OOB rows contribute 0.
                gated_val = vn_val * gate_vec
                gv_col = idx * 16 + lane_n
                _lds_write_x4(
                    fx.slice(sGV, (gv_col, (None, row_block))),
                    _f32x4_to_bf16x4(gated_val),
                )

            # Prefetch the next chunk's k; overlaps the barrier + h store below.
            k_next = _load_k_pairs(next_chunk_base)

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

            # h store out of the swizzled transpose buffer, between GEMM1 and
            # GEMM2. The buffer was filled during staging above and is
            # read-only from here on, and gated_v lives in a different LDS
            # region, so the two do not conflict.
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

            # -- GEMM2: h += k^T @ v_new_gated. Each 64-K block is its own M
            # tile, hence its own k panel view and accumulator fragment. The A
            # operand uses an explicit per-k-tile fx.slice (see the
            # nested-swizzle note on _k_panel_view) at the MFMA A coordinate,
            # lane l holding A[wid*16 + l%16][k_tile*16 + (l//16)*4].
            BT_STEPS = BT // K_STEP
            for kb in range_constexpr(NUM_K_BLOCKS):
                for bt_s in range_constexpr(BT_STEPS):
                    for half in range_constexpr(K_STEP // 16):
                        kt = bt_s * (K_STEP // 16) + half
                        frag_k[kb][None, None, kt].store(
                            _lds_read_x4(
                                fx.slice(
                                    sK[kb],
                                    (wid * 16 + lane_n, (None, kt * 4 + lane_m_base)),
                                )
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
                            frag_k[kb][None, None, kt],
                            frag_gv[None, None, kt],
                            frag_h_accs[kb],
                        )

            # Publish the prefetched w_next/k_next to LDS for the next
            # iteration; the barrier ensures GEMM2 is done with the old panels.
            has_next = next_chunk_base < T_local
            if has_next:
                gpu.barrier()
                _store_w_rows(w_next)
                _store_k_pairs(k_next)

            results = yield [
                frag_h_accs[kb].load() for kb in range_constexpr(NUM_K_BLOCKS)
            ]

        for kb in range_constexpr(NUM_K_BLOCKS):
            frag_h_accs[kb].store(fx.Vector(results[kb], (H_ACC_ELEMS,), fx.Float32))

        # -- Epilogue: store final state --
        # acc_val is already f32x4 with element i at K offset i, so it stores
        # directly as one buffer_store_dwordx4 rather than 4 scalar stores.
        if const_expr(STORE_FINAL_STATE):
            for kb in range_constexpr(NUM_K_BLOCKS):
                for slot in range_constexpr(N_REPEAT):
                    acc_val = fx.Vector(frag_h_accs[kb][None, None, slot].load())
                    ht_col, ht_kgroup = _state_coord(kb, slot)
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
