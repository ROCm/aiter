# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""GDN K5 inter-chunk state scan — gfx942 (CDNA3 / MI300X) FlyDSL kernel.

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
from flydsl.expr import as_ir_value, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import vector as _vector

from .tensor_shim import GTensor, _to_raw

_LOG2E = math.log2(math.e)  # 1.4426950408889634


def _make_fast_exp(g_is_log2_scaled: bool):
    """Return the ``exp`` helper (see gfx950 kernel for the rationale)."""
    if g_is_log2_scaled:

        def _fast_exp(x):
            return rocdl.exp2(T.f32, x)

    else:

        def _fast_exp(x):
            return rocdl.exp2(T.f32, x * _LOG2E)

    return _fast_exp

def _to_bf16_fast(val, n=1):
    """f32 -> bf16 as ``(bitcast<u32>(x) + 0x8000) >> 16`` (round-half-away).

    ``n`` is the element count: 1 for a scalar ``Float32``, N for an f32xN
    ``Vector``. Returns a raw ``ir.Value`` (accepted by ``fx.ptr_store`` and by
    the ``GTensor`` store paths).

    Why not ``.truncf()`` / ``.to(BFloat16)``: those emit ``arith.truncf`` with
    no rounding-mode attribute, which MLIR defines as IEEE round-to-nearest-EVEN.
    gfx942 has no ``v_cvt_pk_bf16_f32`` (gfx950-only), so the backend expands RNE
    into ~6 VALU per element -- extract lsb, add the 0x7FFF bias, ``v_cmp_u_f32``
    NaN test, ``v_cndmask``, then pack with ``v_perm_b32``. At 64 conversions per
    chunk that was ~90 of the ~193 non-MFMA VALU instructions in the chunk loop,
    the single largest term. Asking for ``rounding_mode=toward_zero`` instead is
    NOT an option: it hard-aborts inside MLIR (uncatchable) on this path.

    Pure truncation (what the HIP reference does --
    ``bit_cast<uint32_t>(x) >> 16``, csrc/kernels/chunk_gated_delta_rule_fwd_h.cu:63)
    is cheaper still at ~2 VALU, but it was measured to be marginally too lossy
    here: its one-sided bias, accumulated over the serial chunk scan, put 13 of
    25.4M h elements past the 5e-2 tolerance (max_abs 0.104). Adding the 0x8000
    bias first costs one more add and restores <=0.5 ulp symmetric error, which
    matches RNE except on exact ties.

    Ties round away from zero rather than to even. Values are sign-magnitude, so
    the same bias works for both signs; the carry can only perturb the exponent
    within ~1 ulp of FLT_MAX, far outside the range this kernel produces.
    """
    is_vec = n > 1
    i32_ty = T.vec(n, T.i32) if is_vec else T.i32
    i16_ty = T.vec(n, T.i16) if is_vec else T.i16
    bf16_ty = T.vec(n, T.bf16) if is_vec else T.bf16

    def _splat(c):
        return as_ir_value(fx.full(n, c, fx.Int32) if is_vec else fx.Int32(c))

    bits = _arith.bitcast(i32_ty, as_ir_value(val))
    # The shift may be signed or unsigned: the following trunci keeps only the
    # low 16 bits, which are bits 16..31 of the input either way.
    hi = _arith.shrui(_arith.addi(bits, _splat(0x8000)), _splat(16))
    narrowed = _arith.trunci(i16_ty, hi)
    cast = _vector.bitcast if is_vec else _arith.bitcast
    return cast(bf16_ty, narrowed)


def _mfma_bf16_16x16x16(a_bf16x4, b_bf16x4, acc_f32x4):
    """Single ``mfma_f32_16x16x16bf16_1k`` (gfx942 bf16 K=16 MFMA).

    The MFMA fragment ABI:
    * A operand (16x16x16): lane holds bf16x4 with element e = A[m=lane_n, k=grp*4+e],
        where grp = lane_m_base (lane//16).
    * B operand: lane holds bf16x4 with element e = B[k=grp*4+e, n=lane_n].
    * C/D accumulator: lane holds f32x4 with element e = C[m=grp*4+e, n=lane_n].
    
    Operands are bitcast bf16x4 -> vec<4xi16> (the intrinsic's operand type).
    """
    a_i16 = _vector.bitcast(T.vec(4, T.i16), as_ir_value(a_bf16x4))
    b_i16 = _vector.bitcast(T.vec(4, T.i16), as_ir_value(b_bf16x4))
    return rocdl.mfma_f32_16x16x16bf16_1k(T.f32x4, [a_i16, b_i16, acc_f32x4, 0, 0, 0])

def compile_chunk_gated_delta_h_gfx942(
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
    """Build the gfx942 GDN K5 launcher for one compile-time configuration.

    Signature matches ``compile_chunk_gated_delta_h`` so ``_get_or_compile`` in
    ``linear_attention_prefill_kernels`` can call either implementation without modification.
    """
    assert K <= 256
    assert K % 64 == 0
    assert BV % 16 == 0
    # gfx942 LDS budget: after the lds_vnt reclaim (Gap 4, sized to BV not V), the
    # 4 LDS buffers total ~58 KiB at BV=64 (< 64 KiB/CU), so BV=64 now fits. The
    # previous cap of 32 was due to the old V-sized lds_vnt (66.5 KiB at BV=64).
    assert BV <= 64, (
        f"gfx942 LDS budget caps BV at 64 (got BV={BV}); "
        "BV>64 overflows the 64 KiB/CU LDS limit at K=128, BT=64."
    )
    NUM_K_BLOCKS = K // 64

    _fast_exp = _make_fast_exp(G_IS_LOG2_SCALED)

    WARP_SIZE = 64
    NUM_WARPS = 4
    BLOCK_THREADS = NUM_WARPS * WARP_SIZE

    WMMA_N = 16
    WMMA_K = 16  # gfx942: K=16
    N_REPEAT = BV // WMMA_N

    NUM_H_ACCS = NUM_K_BLOCKS * N_REPEAT

    # -- LDS layout --
    # All four buffers use the same GROUP-MAJOR + XOR scheme (see _grp_idx in the
    # kernel body): a logical [R, C] tile is stored as [R][C/4][4], and the group
    # index is XOR-swizzled by the row. 4 bf16 = 8 B = one MFMA fragment, so every
    # fragment access is a single conflict-free ds_read_b64/ds_write_b64 and no
    # buffer needs padding.
    assert BT % 4 == 0 and K % 4 == 0
    # The XOR is a bank bijection only if a row has >= 16 groups (one per lane of
    # an MFMA fragment); both K/4 and BT/4 are >= 16 for the supported shapes.
    assert K // 4 >= 16 and BT // 4 >= 16, "group-XOR needs >=16 groups per row"

    # lds_w: w tile [BT, K] (A-frag for GEMM1). Single stage.
    # The old row-major [BT, K] pitch was 256 B == 0 (mod 32 banks), so all 16
    # lanes of an A-frag hit the SAME bank -- a 16-way conflict on the highest
    # traffic read in the kernel. The XOR fixes exactly that.
    LDS_W_NG = K // 4
    LDS_W_ELEMS = BT * K

    # lds_k: k tile stored TRANSPOSED as [K, BT] so GEMM2's k A-frag (a run over BT
    # for fixed K) is one group.
    LDS_KT_NG = BT // 4
    LDS_KT_ELEMS = K * BT

    # lds_vn: v_new stored TRANSPOSED as [BV, BT] -> GEMM2 B-frag = run over BT
    # (contraction) at fixed V. Each CTA only handles a BV-wide V-slice, and every
    # vnt access uses v_local = nr*16 + lane_n in [0, BV) -- NOT the full V, so the
    # buffer is sized to BV rows (this is what lets BV=64 fit the 64 KiB budget).
    LDS_VNT_NG = BT // 4
    LDS_VNT_ELEMS = BV * BT

    # lds_h: h snapshot, logically [BV, K] (v_local, k) -- GEMM1's B-frag is a run
    # over K (the contraction) at fixed V, and the HBM snapshot wants K contiguous
    # too, so both consumers want the same K-major order.
    #
    # Layout is GROUP-MAJOR + XOR (see _grp_idx below): the row is split into
    # NG = K/4 groups of 4 bf16, and the group index is XOR-swizzled by the row.
    # 4 bf16 = 8 bytes = exactly one MFMA fragment, so every access is a single
    # aligned ds_read_b64/ds_write_b64 instead of 4 scalar ds_*_u16.
    #
    # Why this replaces the old ``[BV, K+4]`` padded layout: with a 132-element
    # pitch the row stride is 264 B = 66 dwords == 2 (mod 32 banks), so the 16
    # lanes of a fragment land on only the 16 EVEN banks -- 2-way conflicted no
    # matter how wide the access is. That is why widening the old layout to vec4
    # and adding an XOR on top of the pitch both measured as no-ops. Here the row
    # term is a multiple of 128 B, so the bank pair is decided purely by the
    # swizzled group index, and XOR-ing by the row makes it a bijection across the
    # 16 lanes -> all 32 banks hit exactly once. Padding is no longer needed.
    LDS_H_NG = K // 4
    LDS_H_ELEMS = BV * K  # BV rows of V per CTA, no padding

    @fx.struct
    class SharedStorage:
        lds_w: fx.Array[fx.BFloat16, LDS_W_ELEMS, 16]
        lds_kt: fx.Array[fx.BFloat16, LDS_KT_ELEMS, 16]
        lds_vnt: fx.Array[fx.BFloat16, LDS_VNT_ELEMS, 16]
        lds_h: fx.Array[fx.BFloat16, LDS_H_ELEMS, 16]

    # Cooperative load parameters (bf16x8 = dwordx4)
    LOAD_VEC_WIDTH = 8
    THREADS_PER_ROW_64 = 64 // LOAD_VEC_WIDTH  # 8
    ROWS_PER_BATCH_64 = BLOCK_THREADS // THREADS_PER_ROW_64  # 32
    NUM_LOAD_BATCHES_64 = BT // ROWS_PER_BATCH_64  # 2

    K_STEPS_PER_BLOCK = 64 // WMMA_K  # 4
    BT_STEPS = BT // WMMA_K  # 4

    # -- k store-transpose decomposition --
    # k arrives from HBM as runs along K, but lds_kt wants runs along BT, so the
    # store is a genuine transpose. With the default load mapping (1 row x 8 k-cols
    # per thread) the 8 elements land in 8 different lds_kt rows -> 8 scalar
    # ds_write_b16. Instead give each thread 4 BT-CONSECUTIVE rows at the same 8
    # k-cols: then for each k-col the 4 values are one bt-group, so an in-register
    # transpose turns 8 scalar writes into one packed ds_write_b64 each.
    # A "slot" is one (row-quad, k-col-group) pair; each slot is 4 vec8 loads.
    K_COL_GROUPS = K // LOAD_VEC_WIDTH
    K_ROW_QUADS = BT // 4
    K_XPOSE_SLOTS = K_ROW_QUADS * K_COL_GROUPS
    # Only take this path when the slots tile the block exactly (true for K=128
    # and K=256); otherwise fall back to the scalar transpose store.
    K_PACKED_XPOSE = K_XPOSE_SLOTS % BLOCK_THREADS == 0
    K_SLOTS_PER_THREAD = K_XPOSE_SLOTS // BLOCK_THREADS if K_PACKED_XPOSE else 0
    K_ROW_QUAD_STRIDE = BLOCK_THREADS // K_COL_GROUPS if K_PACKED_XPOSE else 0

    @flyc.kernel(name="chunk_gdn_fwd_h_flydsl_vk")
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
        state_t = T.bf16 if STATE_DTYPE_BF16 else T.f32
        if const_expr(USE_INITIAL_STATE):
            h0_ = GTensor(h0_tensor, dtype=state_t, shape=(-1,))
        if const_expr(STORE_FINAL_STATE):
            ht_ = GTensor(ht_tensor, dtype=state_t, shape=(-1,))

        if const_expr(IS_VARLEN):
            cu_ = GTensor(cu_seqlens_tensor, dtype=T.i32, shape=(-1,))
            co_ = GTensor(chunk_offsets_tensor, dtype=T.i32, shape=(-1,))

        # -- LDS views --
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_w_ptr = lds.lds_w.ptr
        lds_kt_ptr = lds.lds_kt.ptr
        lds_vnt_ptr = lds.lds_vnt.ptr
        lds_h_ptr = lds.lds_h.ptr

        # -- Group-major + XOR LDS addressing --
        # A buffer of R rows x C columns is stored as [R][C/4][4]: each row is
        # NG = C/4 groups of 4 bf16 (8 B = one MFMA fragment), and the group index
        # is XOR-swizzled by the row so that the 16 lanes of a fragment (whose row
        # indices are 16 consecutive values) map to 16 distinct groups -- covering
        # all 32 banks exactly once. Returns the element index of the group base.
        def _grp_idx(row, grp, cols, ng):
            # The mask folds the row's bits 3+ down onto its low bits before the
            # XOR. Most sites vary the row's low bits across lanes (all four MFMA
            # fragment reads use row = ...*16 + lane_n), so a plain ``row & (ng-1)``
            # spreads them sufficiently. But the k store-transpose writes rows
            # ``(tid%16)*8 + e`` -- across its 16 lanes ``row & 15`` takes only two
            # distinct values, so a plain mask cannot spread it and that causes
            # 8-way bank multiplicity.
            # Folding in ``row >> 3`` keys the swizzle on the bits that site does
            # vary, and a sweep over all nine LDS sites puts every one at minimum 
            # bank conflict rate.
            #
            # Safe by construction: this only permutes group slots within a row, XOR
            # by a fixed per-row value is a bijection on the group index, and writes
            # and reads derive the mask from the same row. Each (row, grp) still
            # maps to a unique slot.
            mask = (row ^ (row >> fx.Int32(3))) & fx.Int32(ng - 1)
            return row * fx.Int32(cols) + ((grp ^ mask) * fx.Int32(4))

        # 4 bf16 = 8 B: one ds_read_b64 / ds_write_b64, and one MFMA A/B fragment.
        v4bf16_type = T.vec(4, T.bf16)

        def _lds_h_idx(v_local, k_grp):
            return _grp_idx(v_local, k_grp, K, LDS_H_NG)

        def _lds_w_idx(bt_row, k_grp):
            return _grp_idx(bt_row, k_grp, K, LDS_W_NG)

        def _lds_kt_idx(k_row, bt_grp):
            return _grp_idx(k_row, bt_grp, BT, LDS_KT_NG)

        def _lds_vnt_idx(v_local, bt_grp):
            return _grp_idx(v_local, bt_grp, BT, LDS_VNT_NG)

        # -- Cooperative load decomposition --
        load_row_in_batch = tid // fx.Int32(THREADS_PER_ROW_64)
        load_col_base = (tid % fx.Int32(THREADS_PER_ROW_64)) * fx.Int32(LOAD_VEC_WIDTH)

        # k uses its own mapping so the transpose store can be packed: thread ->
        # (row-quad, k-col-group). Consecutive tids walk k-col groups, so a full
        # K-row is covered by K_COL_GROUPS consecutive threads (contiguous HBM).
        if const_expr(K_PACKED_XPOSE):
            kx_col_base = (tid % fx.Int32(K_COL_GROUPS)) * fx.Int32(LOAD_VEC_WIDTH)
            kx_row_quad = tid // fx.Int32(K_COL_GROUPS)

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
        h_base = (boh * fx.Int32(H) + i_h) * fx.Int32(V * K)
        stride_h = fx.Int32(H * V * K)

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
        h_accs = []
        for _kb in range_constexpr(NUM_K_BLOCKS):
            for _nr in range_constexpr(N_REPEAT):
                h_accs.append(acc_zero)

        # -- Load initial state if provided --
        # h_accs[kb][nr] element e = h[v = i_v*BV + nr*16 + lane_n,
        #                              k = kb*64 + wid*16 + lane_m_base*4 + e]
        if const_expr(USE_INITIAL_STATE):
            for kb in range_constexpr(NUM_K_BLOCKS):
                for nr in range_constexpr(N_REPEAT):
                    h0_col = i_v * fx.Int32(BV) + fx.Int32(nr * 16) + lane_n
                    h0_row_base = (
                        fx.Int32(kb * 64)
                        + wid * fx.Int32(16)
                        + lane_m_base * fx.Int32(4)
                    )
                    h0_off_base = h0_base + h0_col * fx.Int32(K) + h0_row_base
                    loaded_vec = h0_.vec_load((fx.Int64(h0_off_base),), 4)
                    if const_expr(STATE_DTYPE_BF16):
                        loaded_vec = loaded_vec.extf(T.f32x4)
                    acc_idx = kb * N_REPEAT + nr
                    h_accs[acc_idx] = h_accs[acc_idx] + loaded_vec

        NUM_W_LOADS = NUM_K_BLOCKS * NUM_LOAD_BATCHES_64

        # -- Prologue: pre-load first chunk's w data --
        i_t0_i32 = fx.Int32(0)
        w_prefetch_init = []
        for kb in range_constexpr(NUM_K_BLOCKS):
            for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                row = fx.Int32(batch * ROWS_PER_BATCH_64) + load_row_in_batch
                abs_row = i_t0_i32 * fx.Int32(BT) + row
                safe_row = (abs_row < T_local).select(abs_row, fx.Int32(0))
                g_off = w_base + safe_row * stride_w + fx.Int32(kb * 64) + load_col_base
                w_prefetch_init.append(w_.vec_load((fx.Int64(g_off),), LOAD_VEC_WIDTH))

        init_state = [_to_raw(v) for v in h_accs] + [
            _to_raw(v) for v in w_prefetch_init
        ]
        c_zero = fx.Int64(0)
        c_one = fx.Int64(1)
        nt_idx = fx.Int64(NT)

        for i_t, state in range(c_zero, nt_idx, c_one, init=init_state):
            h_accs_in = list(state[:NUM_H_ACCS])
            w_prefetch_all = list(state[NUM_H_ACCS:])
            i_t_i32 = fx.Int32(i_t)

            # -- w LDS write offsets (group-major [BT][K/4][4] + XOR) --
            # Each thread holds a bf16x8 run = two adjacent k-groups, whose
            # swizzled positions are NOT adjacent, so the single ds_write_b128
            # becomes two ds_write_b64. That is the price of making the far
            # hotter A-frag read (below) conflict-free, and matches what the HIP
            # reference does for its w panels.
            w_prefetch_lds_all = []
            for kb in range_constexpr(NUM_K_BLOCKS):
                for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                    row = fx.Int32(batch * ROWS_PER_BATCH_64) + load_row_in_batch
                    grp = fx.Int32(kb * (64 // 4)) + load_col_base // fx.Int32(4)
                    w_prefetch_lds_all.append(
                        (_lds_w_idx(row, grp), _lds_w_idx(row, grp + fx.Int32(1)))
                    )

            # -- Store h snapshot to LDS (group-major [BV][K/4][4] + XOR) --
            # h_accs element e = h[v_local = nr*16 + lane_n, k = kb*64 + wid*16 +
            #                      lane_m_base*4 + e].  The four e's are one
            # k-group, so the whole f32x4 accumulator packs into a single bf16x4
            # (one ds_write_b64) instead of 4 scalar ds_write_b16.
            for kb in range_constexpr(NUM_K_BLOCKS):
                for nr in range_constexpr(N_REPEAT):
                    acc_idx = kb * N_REPEAT + nr
                    acc_val = h_accs_in[acc_idx]
                    lds_h_v = fx.Int32(nr * 16) + lane_n
                    lds_h_g = fx.Int32(kb * 16) + wid * fx.Int32(4) + lane_m_base
                    fx.ptr_store(
                        _to_bf16_fast(acc_val, 4),
                        lds_h_ptr + _lds_h_idx(lds_h_v, lds_h_g),
                    )

            gpu.barrier()

            # -- LDS -> HBM h snapshot, one k-group (4 bf16) per thread. --
            # Consecutive tids walk consecutive k-groups at fixed v, so the XOR
            # term is constant across the wave (conflict-free 8 B/lane) and the
            # HBM side is both coalesced and vectorized (it was scalar bf16).
            VG_TOTAL = BV * LDS_H_NG
            for vg_base in range_constexpr(0, VG_TOTAL, BLOCK_THREADS):
                linear = fx.Int32(vg_base) + tid
                g_idx = linear % fx.Int32(LDS_H_NG)
                v_loc = linear // fx.Int32(LDS_H_NG)
                bf16_tile = fx.ptr_load(
                    lds_h_ptr + _lds_h_idx(v_loc, g_idx), result_type=v4bf16_type
                )
                v_global = i_v * fx.Int32(BV) + v_loc
                h_off = (
                    h_base
                    + i_t_i32 * stride_h
                    + v_global * fx.Int32(K)
                    + g_idx * fx.Int32(4)
                )
                h_.vec_store((fx.Int64(h_off),), bf16_tile, 4)

            # -- Store prefetched w to LDS (two b64 halves per bf16x8) --
            for i_wp in range_constexpr(NUM_W_LOADS):
                wvec = w_prefetch_all[i_wp]
                off_lo, off_hi = w_prefetch_lds_all[i_wp]
                lo = fx.Vector.from_elements(
                    [wvec[e] for e in range_constexpr(4)], dtype=fx.BFloat16
                )
                hi = fx.Vector.from_elements(
                    [wvec[4 + e] for e in range_constexpr(4)], dtype=fx.BFloat16
                )
                fx.ptr_store(lo, lds_w_ptr + off_lo)
                fx.ptr_store(hi, lds_w_ptr + off_hi)

            gpu.barrier()

            # -- k prefetch (issued now, stored transposed after GEMM1) --
            k_prefetch = []
            k_prefetch_lds_t = []  # transposed store offsets: lds_kt[k, bt]
            if const_expr(K_PACKED_XPOSE):
                # Each thread owns K_SLOTS_PER_THREAD slots; a slot is 4
                # BT-consecutive rows at one 8-wide k-col group.
                for s in range_constexpr(K_SLOTS_PER_THREAD):
                    row_quad = kx_row_quad + fx.Int32(s * K_ROW_QUAD_STRIDE)
                    quad_rows = []
                    for j in range_constexpr(4):
                        row = row_quad * fx.Int32(4) + fx.Int32(j)
                        abs_row = i_t_i32 * fx.Int32(BT) + row
                        safe_row = (abs_row < T_local).select(abs_row, fx.Int32(0))
                        k_off = k_base + safe_row * stride_k + kx_col_base
                        quad_rows.append(
                            k_.vec_load((fx.Int64(k_off),), LOAD_VEC_WIDTH)
                        )
                    k_prefetch.append(quad_rows)
                    k_prefetch_lds_t.append(row_quad)
            else:
                for kb in range_constexpr(NUM_K_BLOCKS):
                    for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                        row = fx.Int32(batch * ROWS_PER_BATCH_64) + load_row_in_batch
                        abs_row = i_t_i32 * fx.Int32(BT) + row
                        safe_row = (abs_row < T_local).select(abs_row, fx.Int32(0))
                        k_off = (
                            k_base
                            + safe_row * stride_k
                            + fx.Int32(kb * 64)
                            + load_col_base
                        )
                        k_prefetch.append(
                            k_.vec_load((fx.Int64(k_off),), LOAD_VEC_WIDTH)
                        )
                        # this vec holds k[row, kb*64 + load_col_base + (0..7)];
                        # store each element transposed to lds_kt[kcol, row].
                        k_prefetch_lds_t.append(
                            (row, fx.Int32(kb * 64) + load_col_base)
                        )

            # last_idx for gating
            next_chunk_end = (i_t_i32 + fx.Int32(1)) * fx.Int32(BT)
            last_idx_raw = (next_chunk_end < T_local).select(
                next_chunk_end, T_local
            ) - fx.Int32(1)

            # -- g / gk / u prefetch (simple batched, no OPT-VC interleave) --
            if const_expr(USE_G):
                g_last_off = i_h * T_flat + (bos + last_idx_raw)
                g_last_val = g_[fx.Int64(g_last_off)]
                g_row_vals = []
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
                    g_row_vals.append((g_[fx.Int64(g_row_off)], in_bounds))

            if const_expr(USE_GK):
                gk_chunk_base = (bos + last_idx_raw) * fx.Int32(H * K) + i_h * fx.Int32(K)
                gk_last_prefetch = []
                for kb in range_constexpr(NUM_K_BLOCKS):
                    kb_elems = []
                    for elem_i in range_constexpr(4):
                        global_k = (
                            fx.Int32(kb * 64)
                            + wid * fx.Int32(16)
                            + lane_m_base * fx.Int32(4)
                            + fx.Int32(elem_i)
                        )
                        gk_raw = gk_[fx.Int64(gk_chunk_base + global_k)]
                        kb_elems.append(_fast_exp(gk_raw))
                    gk_last_prefetch.append(kb_elems)

            u_prefetch = []
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
                    u_prefetch.append(v_.vec_load((fx.Int64(u_off),), 1))

            # -- GEMM1: bv = w @ h  (contraction over K) --
            # A-frag (w): lane holds w[m=BT row, k]; plain read of lds_w.
            # B-frag (h): lane holds h[k, n=V]; read from lds_h[v, k] with the
            #   transposed access = 4 contiguous k for fixed v (since lds_h is
            #   [v, k] with k contiguous, a run over k IS contiguous).
            bv_accs = []
            for _nr in range_constexpr(N_REPEAT):
                bv_accs.append(fx.full(4, 0.0, fx.Float32))

            for kb in range_constexpr(NUM_K_BLOCKS):
                for ks in range_constexpr(K_STEPS_PER_BLOCK):
                    # w A-frag: 4 bf16 K-elems for this lane's BT row.
                    # A[m=BT row=wid*16+lane_n, k=kb*64+ks*16 + lane_m_base*4 + e]
                    w_row = wid * fx.Int32(16) + lane_n
                    w_g = fx.Int32(kb * 16 + ks * (WMMA_K // 4)) + lane_m_base
                    a_frag = fx.ptr_load(
                        lds_w_ptr + _lds_w_idx(w_row, w_g), result_type=v4bf16_type
                    )

                    for nr in range_constexpr(N_REPEAT):
                        # h B-frag: B[k=kb*64+ks*16 + lane_m_base*4 + e, n=V=nr*16+lane_n]
                        # The 4 k-elements are exactly one k-group, so the whole
                        # fragment is a single ds_read_b64.
                        h_v = fx.Int32(nr * 16) + lane_n
                        h_g = fx.Int32(kb * 16 + ks * (WMMA_K // 4)) + lane_m_base
                        b_frag = fx.ptr_load(
                            lds_h_ptr + _lds_h_idx(h_v, h_g), result_type=v4bf16_type
                        )
                        bv_accs[nr] = _mfma_bf16_16x16x16(a_frag, b_frag, bv_accs[nr])

            # -- v_new = u - bv --
            vn_frags = []
            for nr in range_constexpr(N_REPEAT):
                bv_val = bv_accs[nr]
                u_f32_elems = []
                for elem_i in range_constexpr(4):
                    u_bf16 = fx.BFloat16(u_prefetch[nr * 4 + elem_i])
                    u_f32_elems.append(u_bf16.to(fx.Float32))
                u_f32 = fx.Vector.from_elements(u_f32_elems, dtype=fx.Float32)
                vn_frags.append(u_f32 - bv_val)

            # -- Tail-chunk row mask --
            # On the final chunk, BT rows beyond T_local are padding whose w/u/k
            # loads were clamped to row 0 (garbage). They must be zeroed in v_new
            # before the k^T @ v_new state update, or ``final_state`` is corrupted.
            # The USE_G gate below already zeroes out-of-range rows, but the
            # USE_GK path does no v_new gating -- so mask here unconditionally so
            # both gate ranks are correct. Each lane's f32x4 spans 4 BT rows (one
            # per elem_i); the row is the same across all nr.
            row_mask_elems = []
            for elem_i in range_constexpr(4):
                bt_row = (
                    i_t_i32 * fx.Int32(BT)
                    + wid * fx.Int32(16)
                    + lane_m_base * fx.Int32(4)
                    + fx.Int32(elem_i)
                )
                in_bounds = bt_row < T_local
                row_mask_elems.append(
                    in_bounds.select(fx.Float32(1.0), fx.Float32(0.0))
                )
            row_mask_vec = fx.Vector.from_elements(row_mask_elems, dtype=fx.Float32)
            for nr in range_constexpr(N_REPEAT):
                vn_frags[nr] = vn_frags[nr] * row_mask_vec

            # -- 2b. Store v_new (pre-gating) for output --
            if const_expr(SAVE_NEW_VALUE):

                def _emit_vn_store(off, value):
                    vn_[fx.Int64(off)] = value

                for nr in range_constexpr(N_REPEAT):
                    vn_val = vn_frags[nr]
                    vn_col = i_v * fx.Int32(BV) + fx.Int32(nr * 16) + lane_n
                    for elem_i in range_constexpr(4):
                        vn_bt_row = (
                            i_t_i32 * fx.Int32(BT)
                            + wid * fx.Int32(16)
                            + lane_m_base * fx.Int32(4)
                            + fx.Int32(elem_i)
                        )
                        if (vn_bt_row < T_local).ir_value():
                            f32_v = vn_val[elem_i]
                            bf16_v = _to_bf16_fast(f32_v)
                            vn_off = vn_base + vn_bt_row * fx.Int32(V) + vn_col
                            _emit_vn_store(vn_off, bf16_v)

            # -- 3. Gating --
            if const_expr(USE_G):
                exp_g_last = _fast_exp(g_last_val)
                gate_elems = []
                for elem_i in range_constexpr(4):
                    g_row, in_bounds = g_row_vals[elem_i]
                    gate = _fast_exp(g_last_val - g_row)
                    gate_elems.append(in_bounds.select(gate, fx.Float32(0.0)))
                gate_vec = fx.Vector.from_elements(gate_elems, dtype=fx.Float32)
                for nr in range_constexpr(N_REPEAT):
                    vn_frags[nr] = vn_frags[nr] * gate_vec
                exp_g_last_vec = fx.full(4, fx.Float32(exp_g_last), fx.Float32)
                for kb in range_constexpr(NUM_K_BLOCKS):
                    for nr in range_constexpr(N_REPEAT):
                        acc_idx = kb * N_REPEAT + nr
                        h_accs_in[acc_idx] = h_accs_in[acc_idx] * exp_g_last_vec

            if const_expr(USE_GK):
                for kb in range_constexpr(NUM_K_BLOCKS):
                    gk_vec = fx.Vector.from_elements(
                        [gk_last_prefetch[kb][elem_i] for elem_i in range_constexpr(4)],
                        dtype=fx.Float32,
                    )
                    for nr in range_constexpr(N_REPEAT):
                        acc_idx = kb * N_REPEAT + nr
                        h_accs_in[acc_idx] = h_accs_in[acc_idx] * gk_vec

            # -- 4. State update: h += k^T @ v_new_gated --
            # Store gated v_new transposed as [V, BT] so GEMM2 B-frag (run over
            # BT for fixed V) is contiguous. v_new element e is at
            # BT row = wid*16 + lane_m_base*4 + e, V col = nr*16 + lane_n.
            # The 4 accumulator elements are 4 consecutive BT = one bt-group, so
            # the fragment packs into a single ds_write_b64.
            for nr in range_constexpr(N_REPEAT):
                vnt_v = fx.Int32(nr * 16) + lane_n
                vnt_g = wid * fx.Int32(4) + lane_m_base
                fx.ptr_store(
                    _to_bf16_fast(vn_frags[nr], 4),
                    lds_vnt_ptr + _lds_vnt_idx(vnt_v, vnt_g),
                )

            # Store k transposed as [K, BT].
            if const_expr(K_PACKED_XPOSE):
                # In-register transpose: for each k-col, gather the 4 BT-consecutive
                # rows this thread loaded into one bt-group -> one ds_write_b64.
                # No cross-lane movement is needed; the 4 rows are already local.
                for s in range_constexpr(K_SLOTS_PER_THREAD):
                    quad_rows = k_prefetch[s]
                    row_quad = k_prefetch_lds_t[s]
                    for e in range_constexpr(LOAD_VEC_WIDTH):
                        bt_grp = fx.Vector.from_elements(
                            [quad_rows[j][e] for j in range_constexpr(4)],
                            dtype=fx.BFloat16,
                        )
                        fx.ptr_store(
                            bt_grp,
                            lds_kt_ptr
                            + _lds_kt_idx(kx_col_base + fx.Int32(e), row_quad),
                        )
            else:
                # k_prefetch[i] holds k[row, kcol+(0..7)]; scatter each element to
                # lds_kt[kcol+e, row] (scalar b16 writes).
                for i_kp in range_constexpr(NUM_W_LOADS):
                    kvec = k_prefetch[i_kp]
                    row, kcol = k_prefetch_lds_t[i_kp]
                    row_g = row // fx.Int32(4)
                    row_e = row % fx.Int32(4)
                    for e in range_constexpr(LOAD_VEC_WIDTH):
                        kt_idx = _lds_kt_idx(kcol + fx.Int32(e), row_g) + row_e
                        fx.ptr_store(kvec[e], lds_kt_ptr + kt_idx)

            gpu.barrier()

            # -- next iteration's w prefetch (batched) --
            next_i_t_i32 = i_t_i32 + fx.Int32(1)
            w_next_prefetch = []
            for kb in range_constexpr(NUM_K_BLOCKS):
                for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                    row = fx.Int32(batch * ROWS_PER_BATCH_64) + load_row_in_batch
                    abs_row = next_i_t_i32 * fx.Int32(BT) + row
                    safe_row = (abs_row < T_local).select(abs_row, fx.Int32(0))
                    g_off = (
                        w_base + safe_row * stride_w + fx.Int32(kb * 64) + load_col_base
                    )
                    w_next_prefetch.append(
                        w_.vec_load((fx.Int64(g_off),), LOAD_VEC_WIDTH)
                    )

            # -- GEMM2: h += k^T @ v_new  (contraction over BT) --
            # A-frag (k): lane holds k[m=V head dim? no] -> k is [BT, K]; we want
            #   k^T = [K, BT] as A so output is [K, V]. A[m=K, contraction=BT].
            #   lds_kt[k, bt]: read 4 contiguous BT for fixed k.
            #   A[m=K=wid*16+lane_n? ] -- see ABI: A[m=lane_n(+row grp), k=grp*4+e].
            #   Here MFMA "m" = K output row, "contraction" = BT.
            for kb in range_constexpr(NUM_K_BLOCKS):
                for bt_s in range_constexpr(BT_STEPS):
                    # A-frag k: m = K row = kb*64 + wid*16 + lane_n,
                    #           contraction bt = bt_s*16 + lane_m_base*4 + e
                    k_m = fx.Int32(kb * 64) + wid * fx.Int32(16) + lane_n
                    k_g = fx.Int32(bt_s * (WMMA_K // 4)) + lane_m_base
                    k_a_frag = fx.ptr_load(
                        lds_kt_ptr + _lds_kt_idx(k_m, k_g), result_type=v4bf16_type
                    )

                    for nr in range_constexpr(N_REPEAT):
                        # B-frag v_new: n = V = nr*16 + lane_n,
                        #               contraction bt = bt_s*16 + lane_m_base*4 + e
                        vn_v = fx.Int32(nr * 16) + lane_n
                        vn_g = fx.Int32(bt_s * (WMMA_K // 4)) + lane_m_base
                        vn_b_frag = fx.ptr_load(
                            lds_vnt_ptr + _lds_vnt_idx(vn_v, vn_g),
                            result_type=v4bf16_type,
                        )
                        acc_idx = kb * N_REPEAT + nr
                        h_accs_in[acc_idx] = _mfma_bf16_16x16x16(
                            k_a_frag, vn_b_frag, h_accs_in[acc_idx]
                        )

            results = yield [_to_raw(v) for v in h_accs_in] + [
                _to_raw(v) for v in w_next_prefetch
            ]

        h_accs_final = list(results[:NUM_H_ACCS])

        # -- Epilogue: store final state --
        if const_expr(STORE_FINAL_STATE):
            for kb in range_constexpr(NUM_K_BLOCKS):
                for nr in range_constexpr(N_REPEAT):
                    acc_idx = kb * N_REPEAT + nr
                    acc_val = h_accs_final[acc_idx]
                    ht_col = i_v * fx.Int32(BV) + fx.Int32(nr * 16) + lane_n
                    ht_row_base = (
                        fx.Int32(kb * 64)
                        + wid * fx.Int32(16)
                        + lane_m_base * fx.Int32(4)
                    )
                    ht_off_base = ht_base + ht_col * fx.Int32(K) + ht_row_base
                    if const_expr(STATE_DTYPE_BF16):
                        out_vec = _to_bf16_fast(acc_val, 4)
                    else:
                        out_vec = acc_val
                    ht_.vec_store((fx.Int64(ht_off_base),), out_vec, 4)

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


__all__ = [
    "compile_chunk_gated_delta_h_gfx942",
]
