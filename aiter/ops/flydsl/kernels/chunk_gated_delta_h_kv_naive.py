# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Gated Delta Net K5 hidden-state recurrence kernel -- stripped (no-lds_g) KV fork.

Exports ``compile_chunk_gated_delta_h_kv_naive`` / emits
``chunk_gdn_fwd_h_flydsl_kv_naive`` and backs the host ``_fork="kv_naive"`` path.
This is the NO-lds_g variant (gating reads g inline from HBM per lane). Its
sibling ``chunk_gated_delta_h_kv.py`` is the with-lds_g variant (``_fork="kv"``).

ALL prefetch / software-pipeline scheduling is removed, AND the per-chunk g LDS
staging (lds_g) is removed -- gating reads g inline from HBM per lane. A trace
therefore shows the raw bottleneck structure with no hand-scheduling and no
g-staging tweak:

  * NO cross-chunk w prefetch (no init=/yield carry of w).
  * NO OPT-K k-prefetch interleave (k is loaded straight to LDS before GEMM1).
  * NO OPT-W next-w interleave into GEMM2.

PHASE-4 (rev219): lds_g g-staging IS now enabled (OPT-C(g), ported from the
sibling chunk_gated_delta_h_kv.py). The chunk's 64 per-row g values are staged
once into a tiny 256 B LDS buffer (branchless, published on the existing
w-barrier) and read from LDS in gating, replacing the ~17 scalar
buffer_load_dword/lane that the ISA proved were the #1 VMEM-wait (a vmcnt(16)->0
drain in gating). +256 B LDS, 3-wave safe. The w/u/k prefetch experiments
(rev216-218) were reverted first -- they were net-neutral because those wide
dwordx4 loads were not the binding wait; the scalar g loads were.

What is KEPT (these are layout/correctness, not pipeline scheduling):
  * OPT-VWARP layout (wid -> one 16-wide V-tile, full K). VWARP-only file:
    the non-VWARP path is dropped entirely (this fork is BV==64 only).
  * KV h snapshot store: h is written DIRECTLY to HBM in [..., K, V] layout
    (V innermost) straight from the h_accs registers -- adjacent lanes (V)
    write adjacent HBM addresses => coalesced, no lds_h round-trip. The host
    wrapper returns a transposed VK view so callers see the usual VK layout.
  * h_accs recurrence state is still carried across chunks via the dynamic
    ``for ... init=/yield`` -- that is the SSM recurrence, not a prefetch.

For each chunk t (serial over NT chunks):
  1. Store h snapshot for downstream K6 (h_accs -> HBM, KV [K][V] coalesced).
  2. v_new = u - w @ h   (delta correction via MFMA, h from registers).
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


def _llvm_lds_ptr_ty():
    return ir.Type.parse("!llvm.ptr<3>")


def _make_fast_exp(g_is_log2_scaled: bool):
    if g_is_log2_scaled:

        def _fast_exp(x):
            return rocdl.exp2(T.f32, x)

    else:

        def _fast_exp(x):
            return rocdl.exp2(T.f32, x * _LOG2E)

    return _fast_exp


def _mfma_bf16_16x16x16(a_bf16x4, b_bf16x4, acc_f32x4):
    """Single mfma_f32_16x16x16_bf16 instruction (gfx950 / CDNA bf16 1k form)."""
    a_i16x4 = a_bf16x4.bitcast(fx.Int16)
    b_i16x4 = b_bf16x4.bitcast(fx.Int16)
    return rocdl.mfma_f32_16x16x16bf16_1k(
        T.f32x4, [a_i16x4, b_i16x4, acc_f32x4, 0, 0, 0]
    )


# -- Compile the kernel ---------------------------------------------------


def compile_chunk_gated_delta_h_kv_naive(
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
    """Compile the NAIVE (un-pipelined, no lds_g) GDN K5 KV-fork kernel.

    VWARP-only: requires BV == 64, K % 64 == 0, not USE_GK. The host wrapper
    only routes BV==64 to this kernel; other shapes fall back to the baseline.
    h is stored in KV [..., K, V] layout; the host returns a transposed VK view.
    """
    assert K <= 256
    assert K % 64 == 0
    assert BV % 16 == 0
    NUM_K_BLOCKS = K // 64

    assert BV == 64, "naive KV fork is VWARP-only (BV must be 64)"
    assert not USE_GK, "naive KV fork does not support per-K gates (USE_GK)"

    _fast_exp = _make_fast_exp(G_IS_LOG2_SCALED)

    WARP_SIZE = 64
    NUM_WARPS = 4
    BLOCK_THREADS = NUM_WARPS * WARP_SIZE

    WMMA_N = 16
    WMMA_K = 32  # ds_read_tr16 bt-row span granularity (2 x 16)
    N_REPEAT = BV // WMMA_N  # 4

    NUM_H_ACCS = NUM_K_BLOCKS * N_REPEAT

    K_SUB_PER_BLOCK = 64 // WMMA_N  # 16-wide K sub-tiles per 64 block (=4)
    BT_MTILES = BT // WMMA_N  # 16-row BT M-tiles a warp spans in GEMM1 (=4)

    # EXPERIMENT toggle: stage u via lds_u (True, rev211+) vs read u as scalar
    # from HBM (False, rev210 path). MEASURED (varlen-32k-aws, T1000): USE_LDS_U
    # =False reaches 4 waves/SIMD (VGPR 136->128, LDS 44->35KB) but is ~20%
    # SLOWER (578us vs 480us) because u reverts to 16 scalar buffer_load_ushort
    # -- higher occupancy does NOT compensate the extra VMEM traffic. Keep True.
    USE_LDS_U = True

    # EXPERIMENT toggle: v_new output store. True = stage to lds_vn [BT,V] then
    # coalesced b128 read-out (independent buffer, ~8.5KB, keeps 3 waves);
    # False = original 16 scalar buffer_store_short straight to HBM.
    USE_LDS_VN_STORE = True

    # EXPERIMENT toggle (OPT-C(g), rev219+): stage the chunk's 64 per-row g into
    # lds_g (one coalesced load/thread, +256 B LDS) and read g from LDS in
    # gating. False = rev215 inline path: every lane re-loads its g as ~17
    # scalar buffer_load_dword from HBM. MEASURED (varlen-32k-aws T1000, same
    # idle card): ON vs OFF is wall-clock NEUTRAL (~484us both) -- the inline-g
    # vmcnt(16) drain is fully hidden by 3-wave overlap, so it is not the
    # binding wait -- but ON cuts buffer_load 41->21 / removes the redundant HBM
    # g traffic, so kept True as the cleaner / lower-traffic default.
    USE_LDS_G = True

    # EXPERIMENT toggle (w-prefetch-interleave, rev224): cross-chunk w register
    # prefetch carried via the scf.for iter_args, with the NEXT chunk's w
    # vec_loads issued INTERLEAVED into the GEMM2 MFMA loop (HBM latency hidden
    # behind the MFMA chain, mirroring baseline chunk_gated_delta_h.py OPT-W).
    # False = rev223 naive path (w loaded fresh into lds_w at the top of each
    # chunk). Defined here (outer scope) so the smem sym_name can key on it.
    W_PF_INTERLEAVE = True

    # EXPERIMENT toggle (k-prefetch-interleave, rev225): same as W_PF_INTERLEAVE
    # but for k -- carry t+1's k in registers (issued interleaved in GEMM2) and
    # store to lds_k at the next chunk's step-3 (data from registers, no fresh
    # HBM load). Requires W_PF_INTERLEAVE (shares GEMM2 interleave + iter_arg
    # ordering: state = [h_accs] + [w_pf] + [k_pf]). Goal: take k off the
    # per-chunk synchronous vmcnt drain (the binding wait once w is prefetched).
    K_PF_INTERLEAVE = True

    # EXPERIMENT toggle (HIP-align, store-overlap rev227): move the next-chunk
    # w/k vec_loads from the GEMM2 interleave INTO the GEMM1 MFMA loop, so GEMM2
    # is freed of load traffic and can hide the h-snapshot store instead (see
    # HT_STORE_OVERLAP_GEMM2). Mirrors HIP, which loads next-chunk w/k during
    # run_gemm1 and lets run_gemm2 hide the h store. Requires W_PF_INTERLEAVE.
    # The loaded w_next/k_next live in registers GEMM1->yield (longer than the
    # GEMM2 path), +~32 VGPR, fine at 2 waves (budget 256). False = rev226
    # (loads stay in GEMM2).
    W_PF_INTERLEAVE_GEMM1 = True

    # EXPERIMENT toggle (HIP-align, store-overlap rev227): DECOUPLE the
    # h-snapshot store. Keep only the stage (h_accs -> lds_ht, bf16) at the top
    # of the chunk; move the b128 readout + HBM store to BETWEEN the step-7
    # barrier and GEMM2 so GEMM2's MFMA chain hides the HBM store latency
    # (mirrors HIP coalesced_vk_store_from_transpose placed before run_gemm2).
    # The staged lds_ht copy is pre-gating and independent of h_accs, so GEMM2's
    # in-place h_accs update does not corrupt the snapshot.
    # Requires LDS_HT_INDEPENDENT (lds_ht must survive step1->preGEMM2 unaliased)
    # and is only beneficial with W_PF_INTERLEAVE_GEMM1 (else GEMM2 still busy
    # with loads). Barrier restructure: DROP B2 (stage->readout RAW now covered
    # by the step2 w/u/g barrier + step5 vn barrier + step7 barrier), and ENABLE
    # a top-of-chunk WAR barrier so the NEXT chunk's stage-write does not clobber
    # lds_ht before this chunk's (now late) readout finishes. False = rev226
    # (readout at top of chunk).
    HT_STORE_OVERLAP_GEMM2 = True

    # -- LDS layout (no lds_h: KV h-store goes reg -> HBM directly) --
    # lds_w / lds_k row pads break LDS bank conflicts. K (=128 bf16 = 256 B) is
    # a multiple of the 32-bank LDS period (128 B), so a plain row stride makes
    # the GEMM ds reads (which stride across rows by the LDS stride) land on the
    # same banks. Padding the row stride breaks that integer-multiple relation
    # and offsets the banks WITHOUT any XOR swizzle, so a tile address stays
    # "base + const" and NO extra address-chain / VGPR is introduced (a
    # row-dependent swizzle, by contrast, makes the GEMM2 hi/lo k tiles use
    # different column permutations -> two independent address chains -> +~20
    # VGPR -> drops to 2 waves). The pads were chosen by a counter sweep
    # (SQ_LDS_BANK_CONFLICT) on K128/BT64/BV64: lds_w pad=4 + lds_k pad=8 gives
    # 67.6M conflict cycles vs 470M unpadded (-86%); they are coupled (lds_w's
    # pad shifts lds_k's bank alignment), and conflict is non-monotonic in the
    # pad value, so these are tuned for this shape.
    LDS_W_PAD = 4
    LDS_W_STRIDE = K + LDS_W_PAD
    LDS_W_ELEMS = BT * LDS_W_STRIDE
    LDS_W_BYTES = LDS_W_ELEMS * 2

    LDS_K_PAD = 8
    LDS_K_STRIDE = BT + LDS_K_PAD   # EXPERIMENT (vn-direct): lds_k = [K, BT]
    LDS_K_ELEMS = K * LDS_K_STRIDE
    LDS_K_BYTES = LDS_K_ELEMS * 2

    # EXPERIMENT (h-b128): transpose buffer for the h snapshot store. h is now
    # written to HBM in the public VK layout [..., V, K] (K innermost), and to
    # get a coalesced b128 (8 bf16) HBM write each thread must hold 8 CONTIGUOUS
    # K for one V. The MFMA accumulators put a lane's 4 elems along K (V-tile
    # fixed), so we first stage h_accs into this [V, K] LDS buffer (K innermost,
    # contiguous), barrier, then cooperatively read 8 contiguous K per thread
    # and vec_store(8) -> buffer_store_dwordx4. Row pad on the V axis breaks
    # bank conflicts on the staging write.
    #
    # OCCUPANCY: gfx950 has 160KB LDS/CU; 3 waves/SIMD needs <= 160/3 = 53.3KB
    # per WG. A SEPARATE 17KB transpose buffer would push 44KB -> 60KB and drop
    # to 2 waves/SIMD (LDS-bound), which measured ~3.6% SLOWER despite halving
    # buffer_store. So lds_ht ALIASES lds_k (18KB >= 17KB needed): the snapshot
    # runs at the very start of the chunk, BEFORE this chunk's k is loaded into
    # lds_k (step 3), and AFTER the previous chunk's GEMM2 finished reading
    # lds_k -- a top-of-step-1 barrier guards that WAR. Net LDS stays 44KB ->
    # 3 waves preserved. (Same aliasing trick HIP uses for gated_v/h_state.)
    LDS_HT_PAD = 8  # pad on K axis (bf16); keeps 8-K vec aligned, breaks banks
    LDS_HT_STRIDE = K + LDS_HT_PAD  # [V, K] : K innermost
    LDS_HT_ELEMS = BV * LDS_HT_STRIDE
    LDS_HT_BYTES = LDS_HT_ELEMS * 2

    # EXPERIMENT toggle (ht-independent): when False (default), lds_ht ALIASES
    # lds_k (no extra LDS, 3 waves preserved) -- the snapshot runs at the top of
    # the chunk before this chunk's k is loaded and after the prev chunk's GEMM2
    # read lds_k, with a top-of-step-1 barrier guarding that WAR. When True,
    # lds_ht gets its OWN buffer (+LDS_HT_BYTES ~17KB), HIP-style.
    # MEASURED (2026-06-29, varlen-32k-aws T1000, pytest profiler kernel-time):
    #   - True  (independent): LDS 70400 B/wg -> 2 waves/SIMD (LDS-bound, ATT
    #     confirmed), FlyDSL_kvn ~533 us.
    #   - False (aliased):     LDS 52992 B/wg -> 3 waves/SIMD, FlyDSL_kvn ~495 us.
    # The independent layout is a ~38us (+~8%) REGRESSION (occupancy 3->2); set
    # to True here for direct re-measurement / A/B comparison.
    LDS_HT_INDEPENDENT = True
    assert (LDS_HT_INDEPENDENT or LDS_HT_ELEMS <= LDS_K_ELEMS), \
        "aliased lds_ht must fit inside lds_k"
    assert (not W_PF_INTERLEAVE_GEMM1 or W_PF_INTERLEAVE), \
        "W_PF_INTERLEAVE_GEMM1 requires W_PF_INTERLEAVE"
    assert (not HT_STORE_OVERLAP_GEMM2 or LDS_HT_INDEPENDENT), \
        "HT_STORE_OVERLAP_GEMM2 requires LDS_HT_INDEPENDENT (lds_ht must survive step1->preGEMM2 unaliased)"

    # EXPERIMENT (vn-b128): transpose buffer for the v_new OUTPUT store. v_new is
    # [B,H,T,V] (V innermost); the MFMA-C layout makes a lane's 4 elems stride
    # along BT (by V) -> 16 scalar buffer_store_short. Stage vn into lds_vn
    # [BT, V] (V innermost, contiguous), barrier, then read 8 contiguous V per
    # thread and vec_store(8) -> buffer_store_dwordx4. INDEPENDENT buffer (not
    # aliased): total LDS 44KB + ~8.5KB = ~51.5KB < 160/3 = 53.3KB, so 3 waves
    # are preserved (verified). Tail chunks need a per-row bounds check on the
    # readout (v_new is token-indexed, unlike the chunk-indexed h snapshot).
    LDS_VN_PAD = 4
    LDS_VN_STRIDE = BV + LDS_VN_PAD  # [BT, BV] : V innermost
    LDS_VN_ELEMS = BT * LDS_VN_STRIDE
    LDS_VN_BYTES = LDS_VN_ELEMS * 2

    # EXPERIMENT (u-prefetch): stage u cooperatively into lds_u [BT, BV] so the
    # per-lane v_new reads come from LDS instead of 16 scalar buffer_load_ushort
    # from HBM (u was the #2 VMEM-wait hotspot). u is V-contiguous in HBM, so the
    # cooperative load mirrors the w-load (8 bf16/thread dwordx4, warp coalesced).
    # Row pad to break LDS bank conflicts on the v_new read (mirrors lds_w pad).
    LDS_U_PAD = 4
    LDS_U_STRIDE = BV + LDS_U_PAD
    LDS_U_ELEMS = BT * LDS_U_STRIDE
    LDS_U_BYTES = LDS_U_ELEMS * 2

    # NOTE: lds_vn removed -- GEMM2 now feeds v_new (B operand) straight from
    # the vn_frags registers (vn-direct), so there is no gated-v_new LDS
    # round-trip. This frees BT*(BV+pad)*2 bytes of LDS.

    # OPT-C(g) (PHASE-4): tiny staging buffer for this chunk's per-row g gate
    # values (one f32 per BT row). g depends only on the BT row, not on
    # V/lane/wid, so reading it inline per lane made all 256 lanes re-load the
    # SAME 64 g values as ~17 scalar buffer_load_dword from HBM (the ISA-proven
    # #1 VMEM-wait: a vmcnt(16)->0 drain in gating). Stage once into LDS (one
    # coalesced load per thread, published on the existing w-barrier) -> read
    # from LDS in gating. BT*4 = 256 B (negligible vs the 53.3 KB = 160/3 LDS
    # budget for 3 waves). Ported from the sibling chunk_gated_delta_h_kv.py.
    LDS_G_ELEMS = BT
    LDS_G_BYTES = LDS_G_ELEMS * 4

    _K5_KERNEL_REVISION = 227  # HIP-align: w/k load -> GEMM1, h-snapshot store decoupled -> before GEMM2 (overlap)

    GPU_ARCH = get_rocm_arch()
    allocator = SmemAllocator(
        None,
        arch=GPU_ARCH,
        global_sym_name=f"gdn_h_kv_naive_vkb128alias_u{int(USE_LDS_U)}vn{int(USE_LDS_VN_STORE)}g{int(USE_LDS_G)}ht{int(LDS_HT_INDEPENDENT)}wpf{int(W_PF_INTERLEAVE)}kpf{int(K_PF_INTERLEAVE)}wg1{int(W_PF_INTERLEAVE_GEMM1)}hov{int(HT_STORE_OVERLAP_GEMM2)}_smem_v{_K5_KERNEL_REVISION}",
    )
    lds_w_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_w_offset + LDS_W_BYTES
    lds_k_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_k_offset + LDS_K_BYTES
    lds_u_offset = allocator._align(allocator.ptr, 16)
    if USE_LDS_U:
        allocator.ptr = lds_u_offset + LDS_U_BYTES  # only reserve when used
    # vn-b128: independent lds_vn (margin is enough, no aliasing needed).
    lds_vn_offset = allocator._align(allocator.ptr, 16)
    if USE_LDS_VN_STORE:
        allocator.ptr = lds_vn_offset + LDS_VN_BYTES  # only reserve when used
    # OPT-C(g): tiny per-chunk g staging buffer (256 B), independent.
    lds_g_offset = allocator._align(allocator.ptr, 16)
    if USE_LDS_G:
        allocator.ptr = lds_g_offset + LDS_G_BYTES  # only reserve when used
    # h-b128: lds_ht either ALIASES lds_k (default, no extra allocation, keeps
    # 3 waves) or gets its OWN buffer (LDS_HT_INDEPENDENT=True, HIP-style, ~17KB
    # extra -> expected 2 waves). See the LDS_HT comment above.
    if LDS_HT_INDEPENDENT:
        lds_ht_offset = allocator._align(allocator.ptr, 16)
        allocator.ptr = lds_ht_offset + LDS_HT_BYTES
    else:
        lds_ht_offset = lds_k_offset

    # Cooperative load parameters
    LOAD_VEC_WIDTH = 8  # 8 bf16 = 16 bytes = buffer_load_dwordx4
    THREADS_PER_ROW_64 = 64 // LOAD_VEC_WIDTH  # 8
    ROWS_PER_BATCH_64 = BLOCK_THREADS // THREADS_PER_ROW_64  # 32
    NUM_LOAD_BATCHES_64 = BT // ROWS_PER_BATCH_64  # 2

    @flyc.kernel(name="chunk_gdn_fwd_h_flydsl_kv_naive")
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
        lds_base_ptr = allocator.get_base()
        lds_w_ptr = SmemPtr(lds_base_ptr, lds_w_offset, T.bf16, shape=(LDS_W_ELEMS,))
        lds_w = STensor(lds_w_ptr, dtype=T.bf16, shape=(LDS_W_ELEMS,))
        lds_k_ptr = SmemPtr(lds_base_ptr, lds_k_offset, T.bf16, shape=(LDS_K_ELEMS,))
        lds_k = STensor(lds_k_ptr, dtype=T.bf16, shape=(LDS_K_ELEMS,))
        lds_u_ptr = SmemPtr(lds_base_ptr, lds_u_offset, T.bf16, shape=(LDS_U_ELEMS,))
        lds_u = STensor(lds_u_ptr, dtype=T.bf16, shape=(LDS_U_ELEMS,))
        lds_ht_ptr = SmemPtr(lds_base_ptr, lds_ht_offset, T.bf16, shape=(LDS_HT_ELEMS,))
        lds_ht = STensor(lds_ht_ptr, dtype=T.bf16, shape=(LDS_HT_ELEMS,))
        lds_vn_ptr = SmemPtr(lds_base_ptr, lds_vn_offset, T.bf16, shape=(LDS_VN_ELEMS,))
        lds_vn = STensor(lds_vn_ptr, dtype=T.bf16, shape=(LDS_VN_ELEMS,))
        lds_g_ptr = SmemPtr(lds_base_ptr, lds_g_offset, T.f32, shape=(LDS_G_ELEMS,))
        lds_g = STensor(lds_g_ptr, dtype=T.f32, shape=(LDS_G_ELEMS,))

        # -- Cooperative load decomposition --
        load_row_in_batch = tid // fx.Int32(THREADS_PER_ROW_64)
        load_col_base = (tid % fx.Int32(THREADS_PER_ROW_64)) * fx.Int32(LOAD_VEC_WIDTH)

        v8bf16_type = T.vec(8, T.bf16)
        lds_w_memref = lds_w_ptr.get()
        lds_k_memref = lds_k_ptr.get()
        lds_u_memref = lds_u_ptr.get()

        v4bf16_w_type = T.vec(4, T.bf16)

        def _lds_vec_read_w_bf16x4(elem_idx):
            return vector.load_op(v4bf16_w_type, lds_w_memref, [elem_idx])

        def _lds_vec_read_k_bf16x4(elem_idx):  # EXPERIMENT (vn-direct) std-A read from lds_k[K,BT]
            return vector.load_op(v4bf16_w_type, lds_k_memref, [elem_idx])

        def _lds_read_u_scalar(elem_idx):  # EXPERIMENT (u-prefetch) per-lane u read from lds_u[BT,BV]
            return vector.load_op(T.vec(1, T.bf16), lds_u_memref, [elem_idx])[0]

        v4bf16_type = T.vec(4, T.bf16)

        def _ds_read_tr_bf16x4(lds_byte_offset):
            byte_idx = arith.index_cast(T.index, lds_byte_offset)
            byte_i64 = arith.index_cast(T.i64, byte_idx)
            ptr = _llvm.IntToPtrOp(_llvm_lds_ptr_ty(), byte_i64).result
            raw = rocdl.ds_read_tr16_b64(v4bf16_type, ptr).result
            return fx.Vector(raw, (4,), fx.BFloat16)

        tr_k_group = (lane % fx.Int32(16)) // fx.Int32(4)
        tr_col_sub = lane % fx.Int32(4)

        # -- Prologue: bos, T_local, NT, boh --
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

        # -- Base pointer offsets (element counts). h is [B, NT, H, K, V] (KV)
        # -- here; V*K == K*V so the base/stride math is identical to VK. --
        h_base = (boh * fx.Int32(H) + i_h) * fx.Int32(V * K)
        stride_h = fx.Int32(H * V * K)

        gqa_ratio = H // Hg
        # EXPERIMENT (vn-direct): k pre-transposed to [B, Hg, K, T_flat].
        i_hg_kv = i_h // fx.Int32(gqa_ratio)
        k_base = i_hg_kv * fx.Int32(K) * T_flat + bos

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

        lane_n = lane % fx.Int32(16)
        lane_m_base = lane // fx.Int32(16)

        wid_idx = fx.Index(wid)
        lane_n_idx = fx.Index(lane_n)
        lane_m_base_idx = fx.Index(lane_m_base)

        # -- Initialize h accumulators --
        acc_zero = fx.full(4, 0.0, fx.Float32)
        h_accs = []
        for _kb in range_constexpr(NUM_K_BLOCKS):
            for _nr in range_constexpr(N_REPEAT):
                h_accs.append(acc_zero)

        # -- Load initial state (VWARP: wid -> V-tile, slot -> K sub-tile).
        # h0 is the VK reference [V, K] layout (K innermost), same as the VK
        # fork -- only the snapshot STORE differs between the KV/VK forks. --
        if const_expr(USE_INITIAL_STATE):
            for kb in range_constexpr(NUM_K_BLOCKS):
                for slot in range_constexpr(N_REPEAT):
                    h0_col = i_v * fx.Int32(BV) + wid * fx.Int32(16) + lane_n
                    h0_row_base = (
                        fx.Int32(kb * 64)
                        + fx.Int32(slot * 16)
                        + lane_m_base * fx.Int32(4)
                    )
                    h0_off_base = h0_base + h0_col * fx.Int32(K) + h0_row_base
                    loaded_vec = h0_.vec_load((fx.Index(h0_off_base),), 4)
                    if const_expr(STATE_DTYPE_BF16):
                        loaded_vec = loaded_vec.extf(T.f32x4)
                    acc_idx = kb * N_REPEAT + slot
                    h_accs[acc_idx] = h_accs[acc_idx] + loaded_vec

        NUM_W_LOADS = NUM_K_BLOCKS * NUM_LOAD_BATCHES_64
        NUM_K_LOADS = NUM_K_BLOCKS * NUM_LOAD_BATCHES_64

        # EXPERIMENT (w-prefetch-interleave, rev224): hoist the cooperative w HBM
        # load out of step-2 into a cross-chunk register prefetch carried via the
        # scf.for init=/yield iter_args, AND issue the NEXT chunk's w vec_loads
        # INTERLEAVED into the GEMM2 MFMA loop so each buffer_load's HBM latency
        # is hidden behind the MFMA dependency chain (mirrors baseline
        # chunk_gated_delta_h.py OPT-W; the prior phase1 carried w but issued the
        # load AFTER GEMM2 -- no MFMA overlap -- which did not move VMEM-wait).
        # step-2 then only STORES the already-in-register w to lds_w.
        # When False: step-2 loads w fresh each chunk (the rev223 naive path).
        # (W_PF_INTERLEAVE is defined at outer scope so sym_name can key on it.)

        def _w_load_off(it_i32, kb, batch):
            row = fx.Int32(batch * ROWS_PER_BATCH_64) + load_row_in_batch
            abs_row = it_i32 * fx.Int32(BT) + row
            safe_row = (abs_row < T_local).select(abs_row, fx.Int32(0))
            return w_base + safe_row * stride_w + fx.Int32(kb * 64) + load_col_base

        def _w_lds_off(kb, batch):
            row = fx.Int32(batch * ROWS_PER_BATCH_64) + load_row_in_batch
            col = fx.Int32(kb * 64) + load_col_base
            return row * fx.Int32(LDS_W_STRIDE) + col

        # k-load params hoisted out of step-3 so the prologue + k prefetch helpers
        # can reuse them (k is pre-transposed [B,Hg,K,T_flat], token innermost).
        K_TOK_THREADS = BT // LOAD_VEC_WIDTH               # 8
        K_ROWS_PER_BATCH = BLOCK_THREADS // K_TOK_THREADS  # 32
        K_LOAD_BATCHES = K // K_ROWS_PER_BATCH             # 4
        k_load_krow = tid // fx.Int32(K_TOK_THREADS)
        k_load_tokbase = (tid % fx.Int32(K_TOK_THREADS)) * fx.Int32(LOAD_VEC_WIDTH)

        def _k_load_off(it_i32, batch):
            k_row = fx.Int32(batch * K_ROWS_PER_BATCH) + k_load_krow
            abs_tok = it_i32 * fx.Int32(BT) + k_load_tokbase
            in_b = abs_tok < T_local
            k_off = k_base + k_row * T_flat + abs_tok
            return in_b.select(k_off, k_base)

        def _k_lds_off(batch):
            k_row = fx.Int32(batch * K_ROWS_PER_BATCH) + k_load_krow
            return k_row * fx.Int32(LDS_K_STRIDE) + k_load_tokbase

        c_zero = fx.Index(0)
        c_one = fx.Index(1)
        nt_idx = fx.Index(NT)

        # h_accs is the SSM recurrence state -- carried across chunks via yield.
        # w_pf (W_PF_INTERLEAVE) = chunk-0's w prefetched into registers; also
        # carried via the iter_args so each chunk consumes the prefetch issued by
        # the previous chunk's GEMM2.
        init_state = [_to_raw(v) for v in h_accs]
        if const_expr(W_PF_INTERLEAVE):
            w_pf_init = []
            for kb in range_constexpr(NUM_K_BLOCKS):
                for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                    w_pf_init.append(
                        w_.vec_load(
                            (fx.Index(_w_load_off(fx.Int32(0), kb, batch)),),
                            LOAD_VEC_WIDTH,
                        )
                    )
            init_state = init_state + [_to_raw(v) for v in w_pf_init]
            # k prefetch (K_PF_INTERLEAVE requires W_PF_INTERLEAVE): chunk-0's k
            # into registers, carried after the w_pf block in the iter_args.
            if const_expr(K_PF_INTERLEAVE):
                k_pf_init = []
                for batch in range_constexpr(K_LOAD_BATCHES):
                    k_pf_init.append(
                        k_.vec_load(
                            (fx.Index(_k_load_off(fx.Int32(0), batch)),), LOAD_VEC_WIDTH
                        )
                    )
                init_state = init_state + [_to_raw(v) for v in k_pf_init]

        for i_t, state in range(c_zero, nt_idx, c_one, init=init_state):
            h_accs_in = list(state[:NUM_H_ACCS])
            if const_expr(W_PF_INTERLEAVE):
                w_pf = list(state[NUM_H_ACCS:NUM_H_ACCS + NUM_W_LOADS])
                if const_expr(K_PF_INTERLEAVE):
                    k_pf = list(
                        state[
                            NUM_H_ACCS + NUM_W_LOADS : NUM_H_ACCS
                            + NUM_W_LOADS
                            + K_LOAD_BATCHES
                        ]
                    )
            i_t_i32 = fx.Int32(i_t)

            # ============================================================
            # 1. h snapshot: h_accs -> lds_ht [V, K] -> coalesced b128 HBM
            # (public VK layout [..., V, K], K innermost). EXPERIMENT (h-b128):
            #   (a) stage: each lane owns V col (wid*16+lane_n); its 4 acc elems
            #       are 4 CONTIGUOUS K, written as one vec_store(4) into lds_ht.
            #   (b) barrier.
            #   (c) coalesced store: re-map threads so each thread reads 8
            #       CONTIGUOUS K for one V from lds_ht and vec_store(8) to HBM
            #       -> buffer_store_dwordx4 (was 4 scalar buffer_store_short).
            # WAR barrier at top-of-chunk:
            #   - When lds_ht ALIASES lds_k (LDS_HT_INDEPENDENT=False): ensures
            #     the PREVIOUS chunk's GEMM2 finished reading lds_k before the
            #     staging below overwrites those bytes via lds_ht.
            #   - When lds_ht is INDEPENDENT (True): the alias WAR is gone. The
            #     only remaining hazard is lds_ht's OWN cross-chunk WAR (prev
            #     chunk's step-1c read-out vs this chunk's step-1a stage-write),
            #     but that is already covered by the prev chunk's B3 (line ~595)
            #     and B7 (line ~837) -- both sit between the read-out and this
            #     write -- so B1 is REDUNDANT here and is skipped. (Verified by
            #     time-ordering; kept guarded so the aliased mode still has it.)
            # ============================================================
            # B1 (top-of-chunk WAR):
            #  - aliased lds_ht (LDS_HT_INDEPENDENT=False): guards the prev
            #    chunk's GEMM2 lds_k reads vs this stage-write (lds_ht aliases
            #    lds_k).
            #  - HT_STORE_OVERLAP_GEMM2: the readout is deferred to before GEMM2,
            #    so this barrier guards the prev chunk's (now late) readout-read
            #    of lds_ht vs this chunk's stage-write (lds_ht's own cross-chunk
            #    WAR; rev226 covered it with intervening barriers, but deferring
            #    the readout removes that coverage).
            if const_expr(not LDS_HT_INDEPENDENT or HT_STORE_OVERLAP_GEMM2):
                gpu.barrier()
            ht_v_col = wid * fx.Int32(16) + lane_n
            for kb in range_constexpr(NUM_K_BLOCKS):
                for slot in range_constexpr(N_REPEAT):
                    acc_val = h_accs_in[kb * N_REPEAT + slot]
                    k_tile_base = (
                        fx.Int32(kb * 64)
                        + fx.Int32(slot * 16)
                        + lane_m_base * fx.Int32(4)
                    )
                    ht_vec = acc_val.truncf(T.vec(4, T.bf16))
                    ht_idx = ht_v_col * fx.Int32(LDS_HT_STRIDE) + k_tile_base
                    lds_ht.vec_store((fx.Index(ht_idx),), ht_vec, 4)

            if const_expr(not HT_STORE_OVERLAP_GEMM2):
                # rev226: B2 + coalesced b128 read-out at the TOP of the chunk.
                # V*K = BV*K elems; each thread reads 8 contiguous K (one b128).
                # Groups = BV*(K/8); iters = groups/BLOCK.
                gpu.barrier()
                HT_K8 = K // 8
                HT_GROUPS = BV * HT_K8
                HT_ITERS = HT_GROUPS // BLOCK_THREADS
                for it in range_constexpr(HT_ITERS):
                    grp = fx.Int32(it * BLOCK_THREADS) + tid
                    v_loc = grp // fx.Int32(HT_K8)
                    k8 = (grp % fx.Int32(HT_K8)) * fx.Int32(8)
                    ht_read_idx = v_loc * fx.Int32(LDS_HT_STRIDE) + k8
                    tile8 = lds_ht.vec_load((fx.Index(ht_read_idx),), 8)
                    v_global = i_v * fx.Int32(BV) + v_loc
                    h_off = h_base + i_t_i32 * stride_h + v_global * fx.Int32(K) + k8
                    h_.vec_store((fx.Index(h_off),), tile8, 8)
            # else (HT_STORE_OVERLAP_GEMM2): readout+store deferred to before
            # GEMM2 (step 8) so GEMM2's MFMA hides the HBM store latency.

            # NOTE: no barrier here. readout reads lds_ht(=lds_k); step 2 writes
            # the independent lds_w/lds_u (no conflict), and step 3's lds_k
            # overwrite is guarded by the barrier after the u-load below. Adding
            # a barrier here would be redundant (and costs occupancy-sensitive
            # sync time over the full NT-chunk loop).

            # ============================================================
            # 2. Load w -> lds_w (NO prefetch; load now, use after barrier)
            # ============================================================
            # Pad-based bank-conflict avoidance (no XOR swizzle): the column is the
            # plain logical column; the row pad in LDS_W_STRIDE offsets the banks.
            # GEMM1 read mirrors this.
            if const_expr(W_PF_INTERLEAVE):
                # w already in registers (prefetched in prev chunk's GEMM2 /
                # prologue) -- store-only, no HBM load here.
                i_wp = 0
                for kb in range_constexpr(NUM_K_BLOCKS):
                    for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                        lds_w.vec_store(
                            (fx.Index(_w_lds_off(kb, batch)),), w_pf[i_wp], LOAD_VEC_WIDTH
                        )
                        i_wp += 1
            else:
                for kb in range_constexpr(NUM_K_BLOCKS):
                    for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                        row = fx.Int32(batch * ROWS_PER_BATCH_64) + load_row_in_batch
                        abs_row = i_t_i32 * fx.Int32(BT) + row
                        safe_row = (abs_row < T_local).select(abs_row, fx.Int32(0))
                        w_g_off = (
                            w_base + safe_row * stride_w + fx.Int32(kb * 64) + load_col_base
                        )
                        w_vec = w_.vec_load((fx.Index(w_g_off),), LOAD_VEC_WIDTH)
                        col = fx.Int32(kb * 64) + load_col_base
                        w_lds_off = row * fx.Int32(LDS_W_STRIDE) + col
                        lds_w.vec_store((fx.Index(w_lds_off),), w_vec, LOAD_VEC_WIDTH)

            # EXPERIMENT (u-prefetch): cooperatively load u [BT, BV] -> lds_u,
            # mirroring the w-load (u is V-contiguous in HBM, BV=64 per row =
            # same decomposition as the 64-wide w rows). Replaces the 16 scalar
            # buffer_load_ushort that were the #2 VMEM-wait hotspot. Published
            # on the SAME barrier below as w (no extra barrier).
            if const_expr(USE_LDS_U):
              for batch in range_constexpr(NUM_LOAD_BATCHES_64):
                row = fx.Int32(batch * ROWS_PER_BATCH_64) + load_row_in_batch
                abs_row = i_t_i32 * fx.Int32(BT) + row
                safe_row = (abs_row < T_local).select(abs_row, fx.Int32(0))
                u_g_off = (
                    v_base + safe_row * stride_v
                    + i_v * fx.Int32(BV) + load_col_base
                )
                u_vec = v_.vec_load((fx.Index(u_g_off),), LOAD_VEC_WIDTH)
                u_lds_off = row * fx.Int32(LDS_U_STRIDE) + load_col_base
                lds_u.vec_store((fx.Index(u_lds_off),), u_vec, LOAD_VEC_WIDTH)

            # OPT-C(g) (PHASE-4): cooperatively stage this chunk's per-row g into
            # lds_g. BRANCHLESS: every thread maps to a row via ``tid % BT`` and
            # stores that slot unconditionally (threads mapping to the same slot
            # write the SAME value -> benign race). No ``if tid < BT`` -> no
            # scf.if in the dynamic chunk loop. The w-barrier below publishes it;
            # gating then reads g from LDS instead of ~17 scalar HBM loads/lane.
            # Gated by USE_LDS_G: when False, gating reads g inline from HBM.
            if const_expr(USE_G and USE_LDS_G):
                g_stage_row = tid % fx.Int32(BT)
                g_stage_abs = i_t_i32 * fx.Int32(BT) + g_stage_row
                g_stage_in_bounds = g_stage_abs < T_local
                g_stage_safe = g_stage_in_bounds.select(g_stage_abs, fx.Int32(0))
                g_stage_off = i_h * T_flat + (bos + g_stage_safe)
                g_stage_val = g_[fx.Index(g_stage_off)]
                g_stage_val = g_stage_in_bounds.select(g_stage_val, fx.Float32(0.0))
                lds_g[fx.Index(g_stage_row)] = g_stage_val

            gpu.barrier()

            # ============================================================
            # 3. k -> lds_k [K, BT]. K_PF_INTERLEAVE: k already in registers
            # (prefetched in prev chunk's GEMM2 / prologue) -- store-only. Else:
            # load 8 contiguous tokens (BT, row-contiguous) per thread, store to
            # lds_k. (k-load params hoisted to outer scope.)
            # ============================================================
            if const_expr(W_PF_INTERLEAVE and K_PF_INTERLEAVE):
                for batch in range_constexpr(K_LOAD_BATCHES):
                    lds_k.vec_store(
                        (fx.Index(_k_lds_off(batch)),), k_pf[batch], LOAD_VEC_WIDTH
                    )
            else:
              for batch in range_constexpr(K_LOAD_BATCHES):
                k_row = fx.Int32(batch * K_ROWS_PER_BATCH) + k_load_krow
                abs_tok = i_t_i32 * fx.Int32(BT) + k_load_tokbase
                in_b = abs_tok < T_local
                # k is [B, Hg, K, T_flat] (token innermost); the absolute token
                # within the sequence is i_t*BT + tokbase. The chunk offset
                # i_t*BT MUST be in the address (previously only used for the
                # bounds check -> chunks i_t>0 read chunk-0's k -> GEMM2 wrong).
                k_off = k_base + k_row * T_flat + abs_tok
                k_off = in_b.select(k_off, k_base)
                k_vec = k_.vec_load((fx.Index(k_off),), LOAD_VEC_WIDTH)
                k_lds_off = k_row * fx.Int32(LDS_K_STRIDE) + k_load_tokbase
                lds_k.vec_store((fx.Index(k_lds_off),), k_vec, LOAD_VEC_WIDTH)

            # NOTE (barrier-trim): removed the post-k-load barrier. It guarded
            # the lds_k RAW (step 3 cooperative write -> step 8 GEMM2 read), but
            # nothing reads or overwrites lds_k between here and GEMM2 (GEMM1
            # reads lds_w, v_new reads lds_u, gating reads lds_g), and the
            # step-7 cross-chunk barrier (B7, line ~801) sits right before GEMM2
            # and already publishes this write as a pre-read fence. lds_k's
            # lds_ht alias WAR for the NEXT chunk is covered by B1 (step 1).

            # ============================================================
            # 4. GEMM1: b_v = w @ h^T (h from registers). VWARP.
            # ============================================================
            bv_accs = []
            for _i in range_constexpr(N_REPEAT):
                bv_accs.append(fx.full(4, 0.0, fx.Float32))

            # W_PF_INTERLEAVE_GEMM1: issue NEXT chunk's w/k vec_loads INTERLEAVED
            # into the GEMM1 (m_bt,kb,slot) iterations -- one load on each of the
            # first NUM_W_LOADS+K_LOAD_BATCHES flat slots -- so each
            # buffer_load_dwordx4's HBM latency hides behind GEMM1's MFMA chain
            # AND GEMM2 is left free to hide the deferred h-snapshot store. The
            # loaded values are carried in registers to the yield (mirrors HIP's
            # next-chunk load in run_gemm1).
            if const_expr(W_PF_INTERLEAVE and W_PF_INTERLEAVE_GEMM1):
                next_i_t_i32 = i_t_i32 + fx.Int32(1)
                w_next_pf = [None] * NUM_W_LOADS
                if const_expr(K_PF_INTERLEAVE):
                    k_next_pf = [None] * K_LOAD_BATCHES

            for m_bt in range_constexpr(BT_MTILES):
                for kb in range_constexpr(NUM_K_BLOCKS):
                    for slot in range_constexpr(K_SUB_PER_BLOCK):
                        if const_expr(W_PF_INTERLEAVE and W_PF_INTERLEAVE_GEMM1):
                            g1_slot = (
                                (m_bt * NUM_K_BLOCKS + kb) * K_SUB_PER_BLOCK + slot
                            )
                            # slots [0, NUM_W_LOADS) carry w; the next
                            # K_LOAD_BATCHES slots carry k.
                            if const_expr(g1_slot < NUM_W_LOADS):
                                kb_w = g1_slot // NUM_LOAD_BATCHES_64
                                batch_w = g1_slot % NUM_LOAD_BATCHES_64
                                w_next_pf[g1_slot] = w_.vec_load(
                                    (fx.Index(_w_load_off(next_i_t_i32, kb_w, batch_w)),),
                                    LOAD_VEC_WIDTH,
                                )
                            elif const_expr(
                                K_PF_INTERLEAVE
                                and g1_slot < NUM_W_LOADS + K_LOAD_BATCHES
                            ):
                                batch_k = g1_slot - NUM_W_LOADS
                                k_next_pf[batch_k] = k_.vec_load(
                                    (fx.Index(_k_load_off(next_i_t_i32, batch_k)),),
                                    LOAD_VEC_WIDTH,
                                )
                        w_lds_row_idx = fx.Index(m_bt * 16) + lane_n_idx
                        w_lds_col_idx = fx.Index(
                            kb * 64 + slot * 16
                        ) + lane_m_base_idx * fx.Index(4)
                        # Plain logical column; bank conflicts are avoided by the
                        # LDS_W_STRIDE row pad (matches the cooperative store).
                        w_lds_idx = (
                            w_lds_row_idx * fx.Index(LDS_W_STRIDE) + w_lds_col_idx
                        )
                        a_frag = _lds_vec_read_w_bf16x4(w_lds_idx)
                        b_frag = h_accs_in[kb * N_REPEAT + slot].to(fx.BFloat16)
                        bv_accs[m_bt] = _mfma_bf16_16x16x16(
                            a_frag, b_frag, bv_accs[m_bt]
                        )

            # ============================================================
            # 5. v_new = u - b_v  (u loaded inline, no prefetch)
            # ============================================================
            # EXPERIMENT (u-prefetch): read u from lds_u [BT, BV] (staged above)
            # instead of 16 scalar HBM loads. lds_u is chunk-local (rows 0..BT)
            # and block-local in V (cols 0..BV), so the indices drop the
            # i_t*BT / i_v*BV offsets used for the HBM address.
            u_lds_col = wid * fx.Int32(16) + lane_n
            u_hbm_col = i_v * fx.Int32(BV) + wid * fx.Int32(16) + lane_n
            vn_frags = []
            for m_bt in range_constexpr(BT_MTILES):
                bv_val = bv_accs[m_bt]
                u_f32_elems = []
                for elem_i in range_constexpr(4):
                    if const_expr(USE_LDS_U):
                        u_lds_row = (
                            fx.Int32(m_bt * 16)
                            + lane_m_base * fx.Int32(4)
                            + fx.Int32(elem_i)
                        )
                        u_lds_idx = u_lds_row * fx.Int32(LDS_U_STRIDE) + u_lds_col
                        u_raw = _lds_read_u_scalar(fx.Index(u_lds_idx))
                    else:
                        # rev210 path: scalar u read straight from HBM.
                        u_bt_row_raw = (
                            i_t_i32 * fx.Int32(BT)
                            + fx.Int32(m_bt * 16)
                            + lane_m_base * fx.Int32(4)
                            + fx.Int32(elem_i)
                        )
                        safe_u_row = (u_bt_row_raw < T_local).select(
                            u_bt_row_raw, fx.Int32(0)
                        )
                        u_off = v_base + safe_u_row * stride_v + u_hbm_col
                        u_raw = v_[fx.Index(u_off)]
                    u_bf16 = fx.BFloat16(u_raw)
                    u_f32_elems.append(u_bf16.to(fx.Float32))
                u_f32 = vector.from_elements(T.f32x4, u_f32_elems)
                vn_frags.append(u_f32 - bv_val)

            # ============================================================
            # 5b. Store v_new (pre-gating). EXPERIMENT (vn-b128): stage to
            # lds_vn [BT, V] (V innermost) then coalesced b128 read-out, instead
            # of 16 scalar buffer_store_short. Must run BEFORE gating (step 6)
            # which overwrites vn_frags in registers.
            #   (a) stage: each lane writes its 4 m_bt x 4 elem to lds_vn at
            #       row = m_bt*16 + lane_m_base*4 + elem (BT), col = wid*16+lane_n
            #       (V). Scalar ds_write (cheap, on-chip).
            #   (b) barrier.
            #   (c) read 8 contiguous V per thread, vec_store(8) -> dwordx4.
            #       v_new is token-indexed so guard each row with abs_row<T_local.
            # ============================================================
            if const_expr(SAVE_NEW_VALUE and USE_LDS_VN_STORE):
                vn_stage_col = wid * fx.Int32(16) + lane_n
                for m_bt in range_constexpr(BT_MTILES):
                    vn_val = vn_frags[m_bt]
                    for elem_i in range_constexpr(4):
                        vn_lds_row = (
                            fx.Int32(m_bt * 16)
                            + lane_m_base * fx.Int32(4)
                            + fx.Int32(elem_i)
                        )
                        vn_lds_idx = vn_lds_row * fx.Int32(LDS_VN_STRIDE) + vn_stage_col
                        lds_vn[fx.Index(vn_lds_idx)] = vn_val[elem_i].to(fx.BFloat16)

                gpu.barrier()

                # Bare-function wrapper hides the GTensor vec_store from FlyDSL's
                # dynamic-if AST rewriter (same trick as the scalar path's
                # _emit_vn_store): the if body must not contain obj.method()/
                # subscript on a non-MLIR-Value name.
                def _emit_vn_vstore(off, vec):
                    vn_.vec_store((fx.Index(off),), vec, 8)

                VN_V8 = BV // 8
                VN_GROUPS = BT * VN_V8
                VN_ITERS = VN_GROUPS // BLOCK_THREADS
                for it in range_constexpr(VN_ITERS):
                    grp = fx.Int32(it * BLOCK_THREADS) + tid
                    bt_row_local = grp // fx.Int32(VN_V8)
                    v8 = (grp % fx.Int32(VN_V8)) * fx.Int32(8)
                    # vec_load from lds_vn is unconditional (bt_row_local in
                    # [0,BT) is always a valid LDS index); only the HBM store is
                    # row-bounds-guarded for tail chunks.
                    vn_read_idx = bt_row_local * fx.Int32(LDS_VN_STRIDE) + v8
                    vn_tile8 = lds_vn.vec_load((fx.Index(vn_read_idx),), 8)
                    abs_bt_row = i_t_i32 * fx.Int32(BT) + bt_row_local
                    vn_off = (
                        vn_base + abs_bt_row * fx.Int32(V)
                        + i_v * fx.Int32(BV) + v8
                    )
                    if (abs_bt_row < T_local).ir_value():
                        _emit_vn_vstore(vn_off, vn_tile8)

                # NOTE (barrier-trim): removed the post-readout lds_vn WAR
                # barrier. Its only job was to keep this chunk's lds_vn read-out
                # ahead of the NEXT chunk's lds_vn stage-write (step 5b), but
                # that next write is separated by 6 intervening barriers
                # (B7 step7, B1 step1, B2, B3, B4-or-B7, B5), so the WAR is
                # already covered. gpu.barrier() is LDS/WG-only and does not
                # gate the HBM buffer_store, so dropping it cannot affect v_new
                # correctness.
            elif const_expr(SAVE_NEW_VALUE):
                # Original path: 16 scalar buffer_store_short straight to HBM.
                def _emit_vn_store(off, value):
                    vn_[fx.Index(off)] = value

                for m_bt in range_constexpr(BT_MTILES):
                    vn_val = vn_frags[m_bt]
                    vn_col = i_v * fx.Int32(BV) + wid * fx.Int32(16) + lane_n
                    bt_tile_base = fx.Int32(m_bt * 16)
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

            # ============================================================
            # 6. Gating. OPT-C(g) (PHASE-4, gated by USE_LDS_G): read g from the
            # staged lds_g (published at the w-barrier above) instead of ~17
            # scalar HBM loads/lane. g is chunk-local in lds_g (rows 0..BT-1).
            # USE_LDS_G=False falls back to the rev215 inline-from-HBM path.
            # ============================================================
            next_chunk_end = (i_t_i32 + fx.Int32(1)) * fx.Int32(BT)
            last_idx_raw = (next_chunk_end < T_local).select(
                next_chunk_end, T_local
            ) - fx.Int32(1)

            if const_expr(USE_G):
                if const_expr(USE_LDS_G):
                    # g_last: chunk's last in-bounds row, as a chunk-local index.
                    g_last_in_chunk = last_idx_raw - i_t_i32 * fx.Int32(BT)
                    g_last = lds_g[fx.Index(g_last_in_chunk)]
                else:
                    # rev215 inline: load g_last directly from HBM (head-major
                    # [H, T_flat]; absolute index i_h*T_flat + bos + last_idx).
                    g_last = g_[fx.Index(i_h * T_flat + bos + last_idx_raw)]
                exp_g_last = _fast_exp(g_last)

                for m_bt in range_constexpr(BT_MTILES):
                    gate_elems = []
                    for elem_i in range_constexpr(4):
                        in_chunk_row = (
                            fx.Int32(m_bt * 16)
                            + lane_m_base * fx.Int32(4)
                            + fx.Int32(elem_i)
                        )
                        abs_row = i_t_i32 * fx.Int32(BT) + in_chunk_row
                        in_bounds = abs_row < T_local
                        if const_expr(USE_LDS_G):
                            # in_chunk_row in [0,BT) is always a valid lds_g
                            # index; OOB rows were staged as 0.0, masked below.
                            g_row = lds_g[fx.Index(in_chunk_row)]
                        else:
                            # rev215 inline: per-lane g_row straight from HBM.
                            safe_abs_row = in_bounds.select(abs_row, fx.Int32(0))
                            g_row = g_[fx.Index(i_h * T_flat + bos + safe_abs_row)]
                        gate = _fast_exp(g_last - g_row)
                        gate_elems.append(in_bounds.select(gate, fx.Float32(0.0)))
                    gate_vec = vector.from_elements(T.f32x4, gate_elems)
                    vn_frags[m_bt] = vn_frags[m_bt] * gate_vec

                exp_g_last_s = fx.Float32(exp_g_last)
                for kb in range_constexpr(NUM_K_BLOCKS):
                    for nr in range_constexpr(N_REPEAT):
                        acc_idx = kb * N_REPEAT + nr
                        h_accs_in[acc_idx] = h_accs_in[acc_idx] * exp_g_last_s

            # ============================================================
            # 7. Cross-chunk WAR barrier. GEMM2 below feeds vn straight from
            # registers (vn_frags) -- no lds_vn round-trip -- so there is no
            # vn LDS publish to wait on. This barrier still guards the lds_w
            # WAR hazard: the current chunk's GEMM1 reads of lds_w must finish
            # before the NEXT chunk's cooperative w-load overwrites lds_w.
            # (lds_k's WAR is covered by the post-w-load barrier.)
            # ============================================================
            gpu.barrier()

            # HT_STORE_OVERLAP_GEMM2: deferred h-snapshot readout + HBM store,
            # issued HERE (after step-7 barrier, before GEMM2) so GEMM2's MFMA
            # chain below hides the buffer_store latency. lds_ht still holds this
            # chunk's PRE-gating snapshot (staged at step 1, independent buffer,
            # untouched since -- gating only mutated the h_accs REGISTERS). The
            # buffer_store is fire-and-forget; with loads moved to GEMM1, GEMM2
            # has no vmcnt wait to drain it early. Mirrors HIP's
            # coalesced_vk_store_from_transpose placed before run_gemm2.
            if const_expr(HT_STORE_OVERLAP_GEMM2):
                HT_K8 = K // 8
                HT_GROUPS = BV * HT_K8
                HT_ITERS = HT_GROUPS // BLOCK_THREADS
                for it in range_constexpr(HT_ITERS):
                    grp = fx.Int32(it * BLOCK_THREADS) + tid
                    v_loc = grp // fx.Int32(HT_K8)
                    k8 = (grp % fx.Int32(HT_K8)) * fx.Int32(8)
                    ht_read_idx = v_loc * fx.Int32(LDS_HT_STRIDE) + k8
                    tile8 = lds_ht.vec_load((fx.Index(ht_read_idx),), 8)
                    v_global = i_v * fx.Int32(BV) + v_loc
                    h_off = h_base + i_t_i32 * stride_h + v_global * fx.Int32(K) + k8
                    h_.vec_store((fx.Index(h_off),), tile8, 8)

            # ============================================================
            # 8. GEMM2: h[K,V] += k[K,BT] @ v_new_gated[BT,V]. VWARP.
            # EXPERIMENT (vn-direct): mirror GEMM1 -- A=k standard read from
            # lds_k[K,BT] (BT reduction row-contiguous), B=vn fed straight from
            # registers (vn_frags, standard B). Reduction over BT split into
            # BT_MTILES 16-row tiles; vn_frags[m_bt] is each reduction tile.
            # (numpy-verified index formulas, err ~1e-5.)
            # ============================================================
            # W_PF_INTERLEAVE: issue NEXT chunk's w vec_loads INTERLEAVED into the
            # GEMM2 outer (kb,slot) iterations -- one load on each of the first
            # NUM_W_LOADS slots -- so each buffer_load_dwordx4's HBM latency is
            # hidden behind the following MFMA chain (mirrors baseline OPT-W).
            # When W_PF_INTERLEAVE_GEMM1, the loads were already issued in GEMM1
            # (above) so this GEMM2 interleave is skipped.
            if const_expr(W_PF_INTERLEAVE and not W_PF_INTERLEAVE_GEMM1):
                next_i_t_i32 = i_t_i32 + fx.Int32(1)
                w_next_pf = [None] * NUM_W_LOADS
                if const_expr(K_PF_INTERLEAVE):
                    k_next_pf = [None] * K_LOAD_BATCHES
            for kb in range_constexpr(NUM_K_BLOCKS):
                for slot in range_constexpr(N_REPEAT):
                    acc_idx = kb * N_REPEAT + slot
                    if const_expr(W_PF_INTERLEAVE and not W_PF_INTERLEAVE_GEMM1):
                        g2_slot = kb * N_REPEAT + slot
                        # slots [0, NUM_W_LOADS) carry the w prefetch loads;
                        # slots [NUM_W_LOADS, NUM_W_LOADS+K_LOAD_BATCHES) carry k.
                        if const_expr(g2_slot < NUM_W_LOADS):
                            kb_w = g2_slot // NUM_LOAD_BATCHES_64
                            batch_w = g2_slot % NUM_LOAD_BATCHES_64
                            w_next_pf[g2_slot] = w_.vec_load(
                                (fx.Index(_w_load_off(next_i_t_i32, kb_w, batch_w)),),
                                LOAD_VEC_WIDTH,
                            )
                        elif const_expr(
                            K_PF_INTERLEAVE
                            and g2_slot < NUM_W_LOADS + K_LOAD_BATCHES
                        ):
                            batch_k = g2_slot - NUM_W_LOADS
                            k_next_pf[batch_k] = k_.vec_load(
                                (fx.Index(_k_load_off(next_i_t_i32, batch_k)),),
                                LOAD_VEC_WIDTH,
                            )
                    for m_bt in range_constexpr(BT_MTILES):
                        k_lds_row = fx.Index(kb * 64 + slot * 16) + lane_n_idx
                        k_lds_col = fx.Index(m_bt * 16) + lane_m_base_idx * fx.Index(4)
                        k_lds_idx = k_lds_row * fx.Index(LDS_K_STRIDE) + k_lds_col
                        k_a = _lds_vec_read_k_bf16x4(k_lds_idx)
                        vn_b = vn_frags[m_bt].to(fx.BFloat16)
                        h_accs_in[acc_idx] = _mfma_bf16_16x16x16(
                            k_a, vn_b, h_accs_in[acc_idx]
                        )

            # Single yield statement (FlyDSL's scf.for rewriter forbids two yield
            # statements in a loop body, even under a const_expr branch): build
            # the iter_args list conditionally, then yield once.
            yield_list = [_to_raw(v) for v in h_accs_in]
            if const_expr(W_PF_INTERLEAVE):
                yield_list = yield_list + [_to_raw(v) for v in w_next_pf]
                if const_expr(K_PF_INTERLEAVE):
                    yield_list = yield_list + [_to_raw(v) for v in k_next_pf]
            results = yield yield_list

        h_accs_final = list(results[:NUM_H_ACCS])

        # -- Epilogue: store final state (VK [V, K], same as VK fork) --
        if const_expr(STORE_FINAL_STATE):
            for kb in range_constexpr(NUM_K_BLOCKS):
                for slot in range_constexpr(N_REPEAT):
                    acc_idx = kb * N_REPEAT + slot
                    acc_val = h_accs_final[acc_idx]
                    ht_col = i_v * fx.Int32(BV) + wid * fx.Int32(16) + lane_n
                    ht_row_base = (
                        fx.Int32(kb * 64)
                        + fx.Int32(slot * 16)
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


__all__ = [
    "compile_chunk_gated_delta_h_kv_naive",
]
