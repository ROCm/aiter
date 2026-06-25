# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""FlyDSL port of aiter ``gemm1_a4w4`` (MXFP4 MoE up/gate-proj, gfx950).

A 1:1 port of HIP ``gemm1_a4w4.cuh`` + ``mxfp4_epilogs.hpp``, bit-exact vs
``aiter.mxfp4_moe_gemm1_a4w4``. The module now implements all 4 Kimi-K2.5
gemm1 variants:
  * ``(32, True, False)``  -> BM32_NT
  * ``(32, False, False)`` -> BM32_CACHED
  * ``(128, False, False)`` -> BM128
  * ``(16, True, True)``   -> BM16_INLINEQUANT
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import memref as memref_dialect
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from . import dpp_utils

# -- compile-time constants (BM-independent; BM-derived ones live in _bm_constants) --
MAX_M = 655360
NE = 385  # experts (DEFAULT / KIMI; per-shape value comes from the compile arg NE.
#           Threaded through the body so non-KIMI expert counts work; these module
#           globals are the KIMI defaults so existing importers keep working.)
K = 7168  # gemm1 contraction = hidden_size (DEFAULT / KIMI; per-shape value comes
#           from the compile arg D_HIDDEN. All K-derived sizes are computed from the
#           arg via the *_for() helpers below; these module globals are the KIMI
#           defaults so existing importers (and the test) keep working unchanged.)
INTER = 512  # per-shard MLP intermediate (OUTPUT side; DEFAULT / KIMI; per-shape
#              value comes from the compile arg D_INTER. All INTER-derived sizes
#              are computed from the arg via the *_for() helpers below.)
TOPK = 9  # DEFAULT / KIMI; per-shape value comes from the compile arg TOPK (only
#           used host-side in gemm1_grid for the launch bound).

BN = 256
BK = 256
KH_TILE = BK // 2  # 128 packed bytes per K-tile (BK-derived, K-independent)
kStages = 2

# B-scale layout (OUTPUT side + K-independent strides)
kBS_stride_k0_dw = 64


# -- INTER-derived sizes (parametrized over the OUTPUT/intermediate dim INTER) --
# 2*INTER (=N_OUT) MUST be a multiple of BN (256), i.e. INTER % 128 == 0. The
# *_for() helpers mirror gemm2's pattern so the KIMI default (INTER=512) path is
# byte-for-byte identical to the old globals.
def n_out_for(inter):
    return 2 * inter  # KIMI: 1024


def num_n_blocks_for(inter):
    return n_out_for(inter) // 256  # KIMI: 4 (OUTPUT side; INTER/N_OUT-derived)


def kbs_c_n1_for(inter):
    return n_out_for(inter) // 16 // 2  # KIMI: 32 (N_OUT-derived)


# gemm2-A-scale chunk stride (dwords) the epilog writes with; equals gemm2's
# kAS_per_chunk_dw for K(=gemm2 contraction)=INTER: ((INTER//32)//4//2)*64.
def out_as_per_chunk_dw_for(inter):
    return ((inter // 32) // 4 // 2) * 64  # KIMI: 128


# gemm2-A packed-fp4 bytes per row (= K_G2_HALF), and the per-N-block silu_mul
# intermediate width (= BN//2, BN-derived so INTER-independent).
def k_g2_half_for(inter):
    return inter // 2  # KIMI: 256


BN_INT = BN // 2  # 128 (per-N-block intermediate width; BN-derived, INTER-indep)


# -- OUTPUT-side defaults (back-compat module globals = the *_for() helpers). --
N_OUT = n_out_for(INTER)  # 1024
NUM_N_BLOCKS = num_n_blocks_for(INTER)  # 4
OUT_AS_PER_CHUNK_DW = out_as_per_chunk_dw_for(INTER)  # 128
K_G2_HALF = k_g2_half_for(INTER)  # 256


# -- K-derived sizes (parametrized over the contraction dim K = D_HIDDEN) ------
# K MUST be a multiple of BK (256). The *_for() helpers mirror gemm2's pattern so
# the KIMI default (K=7168) path is byte-for-byte identical to the old globals.
def k_half_for(k):
    return k // 2  # packed-fp4 bytes along K (KIMI: 3584)


def k_tiles_total_for(k):
    return k // BK  # KIMI: 28


def kunroll_for(k):
    # main-loop trip count. The K-loop is: prologue issues kStages tiles, the main
    # loop processes tiles [0, kUnroll) while issuing tiles [kStages, kStages+kUnroll),
    # and the drain processes the final kStages tiles. Full, gap-free coverage of all
    # K_TILES_TOTAL tiles therefore requires EXACTLY kUnroll = K_TILES_TOTAL - kStages
    # (KIMI: 28-2 = 26, matching the old hand-tuned value). This is the correct rule
    # for arbitrary K (e.g. K=3072 -> K_TILES_TOTAL=12 -> kUnroll=10), not just a tune.
    return k_tiles_total_for(k) - kStages


def kas_c_k1_for(k):
    return (k // 32) // 4 // 2  # KIMI: 28


def kas_per_chunk_dw_for(k):
    return 1 * kas_c_k1_for(k) * 64  # KIMI: 1792


def kbs_c_k1_for(k):
    return (k // 32) // 4 // 2  # KIMI: 28


def kbs_stride_n0_dw_for(k):
    return kbs_c_k1_for(k) * 64  # KIMI: 1792


def kbs_per_expert_dw_for(k, inter=INTER):
    # depends on BOTH INTER (via kbs_c_n1_for) AND K (via kbs_stride_n0_dw_for).
    return kbs_c_n1_for(inter) * kbs_stride_n0_dw_for(k)  # KIMI: 57344


def bq_bytes_for(k, inter=INTER, ne=NE):
    return ne * n_out_for(inter) * k_half_for(k)  # KIMI: 1412956160


def ascale_bytes_for(k, max_m=MAX_M):
    return (max_m // 32) * kas_per_chunk_dw_for(k) * 4  # KIMI: 146800640


def bscale_bytes_for(k, inter=INTER, ne=NE):
    return ne * kbs_per_expert_dw_for(k, inter) * 4  # KIMI: 88309760


# KIMI defaults (back-compat module globals; equal to the *_for(K) helpers).
K_HALF = k_half_for(K)  # 3584
K_TILES_TOTAL = k_tiles_total_for(K)  # 28
kUnroll = kunroll_for(K)  # 26
kAS_per_chunk_dw = kas_per_chunk_dw_for(K)  # 1792
kBS_stride_n0_dw = kbs_stride_n0_dw_for(K)  # 1792
kBS_per_expert_dw = kbs_per_expert_dw_for(K)  # 57344
BQ_BYTES = bq_bytes_for(K)  # 1412956160
ASCALE_BYTES = ascale_bytes_for(K)  # 146800640
BSCALE_BYTES = bscale_bytes_for(K)  # 88309760
# (K_G2_HALF/BN_INT/OUT_AS_PER_CHUNK_DW are the INTER-derived globals
#  defined above; the gemm2-A-scale epilog layout lives with the OUTPUT-side block.)

LOG2E = 1.4426950408889634


_PTR3 = "!llvm.ptr<3>"


def _raw(v):
    """Unwrap an fx value to a raw ir.Value for raw llvm/arith ops."""
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _lds_ptr3(base_i32, byte_off_i32):
    """ptr<3> = inttoptr(i64(base_i32 + byte_off_i32))."""
    addr_i64 = fx.Int64(base_i32 + byte_off_i32)
    return llvm.inttoptr(ir.Type.parse(_PTR3), _raw(addr_i64))


def _lds_base_ptr3(lds_view):
    """One ptr<3> for the LDS base; offsets via GEP. (extract_aligned_pointer ->
    inttoptr is forced by FlyDSL's memref.global LDS model.)"""
    base_i32 = fx.Int32(memref_dialect.extract_aligned_pointer_as_index(lds_view))
    return llvm.inttoptr(ir.Type.parse(_PTR3), _raw(fx.Int64(base_i32)))


def _gep3(base_ptr, byte_off_i32):
    """getelementptr i8, base_ptr, byte_off_i32  (ptr<3>)."""
    return buffer_ops.get_element_ptr(
        base_ptr, byte_offset=_raw(byte_off_i32), elem_type=T.i8
    )


def _global_base_ptr1(addr_i64):
    """ptr<1> base from a raw i64 device address.

    Global args are passed as bare ``data_ptr()`` (fx.Int64) rather than full
    memref descriptors: this kernel only ever needs the base pointer (it assumes
    contiguity and derives all sizes from i32_ntok / compile-time constants), so
    the dynamic-memref shape/stride layout buffer was dead weight that scattered
    the kernarg pointers and blocked LLVM from coalescing the scalar loads."""
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), _raw(fx.Int64(addr_i64)))


def _gep1(base_ptr, byte_off_i32):
    """getelementptr i8, base_ptr, byte_off_i32  (ptr<1>)."""
    return buffer_ops.get_element_ptr(
        base_ptr, byte_offset=_raw(byte_off_i32), elem_type=T.i8
    )


def _global_ptr1(arg, byte_off_i32):
    return _gep1(_global_base_ptr1(arg), byte_off_i32)


def _lds_swizzle_mask(row):
    """lds_swizzle_mask<ROW_BYTES=BK/2=128>(row): mask = (row & 14) << 3."""
    return (row & fx.Int32(14)) << fx.Int32(3)


# -- epilog math helpers (HIP mxfp4_epilogs.hpp 87-122 + mxfp4_gemm_common.hpp) -
def _silu_mul(g, u):
    """silu(g)*u, matching HIP silu_mul_fast (mxfp4_gemm_common.hpp:62-65):
    e = __expf(-g) = exp2(-g*log2e); sig = rcpf(1+e); return g*sig*u."""
    e = fx.Float32(rocdl.exp2(T.f32, _raw(g * fx.Float32(-LOG2E))))
    sig = fx.Float32(rocdl.rcp(T.f32, _raw(fx.Float32(1.0) + e)))
    return g * sig * u


def _silu_mul_batch(gs, us):
    e = [fx.Float32(rocdl.exp2(T.f32, _raw(g * fx.Float32(-LOG2E)))) for g in gs]
    sig = [fx.Float32(rocdl.rcp(T.f32, _raw(fx.Float32(1.0) + ei))) for ei in e]
    return [gs[i] * sig[i] * us[i] for i in range(len(gs))]


def _fabs_f32(x):
    """fabsf via bit-mask (FlyDSL has no arith.absf): clear the sign bit."""
    bits = _raw(x).bitcast(T.i32)
    abs_bits = bits & _raw(fx.Int32(0x7FFFFFFF))
    return fx.Float32(abs_bits.bitcast(T.f32))


def _e8m0_roundup(amax_f32):
    """RoundUp e8m0 = ceil_pow2(amax / 6), clamped to 254."""
    wi = fx.Int32(_raw(amax_f32 * fx.Float32(1.0 / 6.0)).bitcast(T.i32))
    bexp = (wi + fx.Int32(0x7FFFFF)).shrui(fx.Int32(23)) & fx.Int32(0xFF)
    lt = arith.cmpi(arith.CmpIPredicate.ult, _raw(bexp), _raw(fx.Int32(254)))
    return fx.Int32(arith.select(lt, _raw(bexp), _raw(fx.Int32(254))))


def _e8m0_from_amax(amax_f32):
    """Returns (e8m0_i32, quant_scale_f32)."""
    e8m0 = _e8m0_roundup(amax_f32)
    qscale = fx.Float32(_raw(e8m0 << fx.Int32(23)).bitcast(T.f32))
    return e8m0, qscale


# -- inline-quant helpers (HIP mxfp4_gemm_common.hpp:76-94) -------------------
def _pkmax_u16(a_i32, b_i32):
    """v_pk_max_u16 a, b -- pairwise u16 max of two packed-u16 dwords.

    No FlyDSL builtin, but no inline asm needed: bitcast each dword to
    vector<2xi16> and let arith.maxui lower through MLIR -> LLVM. The AMDGPU
    backend ISel-pattern-matches `umax <2 x i16>` to a single v_pk_max_u16,
    so this emits the same instruction with zero inline asm."""
    _v2i16 = ir.Type.parse("vector<2xi16>")
    va = llvm.BitcastOp(_v2i16, _raw(a_i32)).result
    vb = llvm.BitcastOp(_v2i16, _raw(b_i32)).result
    vm = arith.MaxUIOp(va, vb).result
    out = llvm.BitcastOp(T.i32, vm).result
    return fx.Int32(out)


def _inline_dpp_quad_amax(a32):
    """inline_quant_dpp_quad_amax (mxfp4_gemm_common.hpp:82-87): mov_dpp 0xB1
    (xor-1) then 0x4E (xor-2) with full row/bank masks + bound_ctrl, max-reduced.
    update.dpp with old==src + full masks + bound_ctrl is equivalent to mov_dpp
    here (every lane participates; result is the umax over the quad)."""
    a32 = fx.Int32(_raw(a32))
    s1 = fx.Int32(dpp_utils.update_dpp_i32(_raw(a32), _raw(a32), 0xB1, 0xF, 0xF, True))
    a32 = _umax_i32(a32, s1)
    s2 = fx.Int32(dpp_utils.update_dpp_i32(_raw(a32), _raw(a32), 0x4E, 0xF, 0xF, True))
    return _umax_i32(a32, s2)


def _umax_i32(a, b):
    """unsigned max of two i32 (used for u16-amax reduction)."""
    is_gt = arith.cmpi(arith.CmpIPredicate.ugt, _raw(a), _raw(b))
    return fx.Int32(arith.select(is_gt, _raw(a), _raw(b)))


def _inline_e8m0(amax_u16_i32):
    """RoundUp e8m0 from a packed u16 (bf16) amax. Returns e8m0 as i32 (0..254)."""
    f32 = fx.Float32(
        _raw((fx.Int32(_raw(amax_u16_i32)) & fx.Int32(0xFFFF)) << fx.Int32(16)).bitcast(
            T.f32
        )
    )
    return _e8m0_roundup(f32)


def gemm1_grid(n_tokens, BM=32, NE=NE, TOPK=TOPK, INTER=INTER):
    """Host-side grid size, mirroring gemm1_a4w4.cuh launch (:611-619).

    BM==128 uses the full-NE bound; else the active-experts bound. (BM=32 path
    is unchanged from Phase 1.) NE/TOPK/INTER default to the KIMI module globals.
    """
    num_n_blocks = num_n_blocks_for(INTER)
    if BM == 128:
        max_m_blocks = (n_tokens * TOPK + NE * (BM - 1) + BM - 1) // BM
    else:
        active = min(n_tokens * TOPK, NE)
        max_m_blocks = (n_tokens * TOPK + active * (BM - 1) + BM - 1) // BM
    return max_m_blocks * num_n_blocks


@flyc.jit
def _gemm1_body(
    allocator,
    lds_off,
    arg_aq,
    arg_ascale,
    arg_bq,
    arg_bscale,
    arg_eids,
    arg_mind,
    arg_aqout,
    arg_ascaleout,
    arg_hidden,
    bx_i32,
    lane,
    wave,
    use_nt,
    i32_ntok,
    i32_total_m_blocks,
    *,
    BM,
    kAStages,
    kSubBlocks,
    kMChunks,
    inline_quant=False,
    K=K,
    K_HALF=K_HALF,
    K_TILES_TOTAL=K_TILES_TOTAL,
    kUnroll=kUnroll,
    kAS_per_chunk_dw=kAS_per_chunk_dw,
    kBS_stride_n0_dw=kBS_stride_n0_dw,
    kBS_per_expert_dw=kBS_per_expert_dw,
    BQ_BYTES=BQ_BYTES,
    ASCALE_BYTES=ASCALE_BYTES,
    BSCALE_BYTES=BSCALE_BYTES,
    N_OUT=N_OUT,
    NUM_N_BLOCKS=NUM_N_BLOCKS,
    OUT_AS_PER_CHUNK_DW=OUT_AS_PER_CHUNK_DW,
    K_G2_HALF=K_G2_HALF,
    interleave=True,
):
    # All K- and INTER/NE-derived sizes arrive as params (KIMI defaults bound from
    # the module globals above so the default call path is unchanged).
    b_aux = 2 if use_nt else 0  # NT: B_q loads carry aux=2 (non-temporal hint)
    M_REPS = BM // 16

    # block -> (m_block_idx, n_block_idx) ; e = sorted_expert_ids[m_block_idx]
    n_block_idx = bx_i32 % fx.Int32(NUM_N_BLOCKS)
    m_block_idx = bx_i32 // fx.Int32(NUM_N_BLOCKS)
    e = rocdl.readfirstlane(
        T.i32, llvm.load(T.i32, _global_ptr1(arg_eids, m_block_idx * fx.Int32(4)))
    )
    m_row = m_block_idx * fx.Int32(BM)

    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lane_div_8 = lane // fx.Int32(8)
    lane_mod_8 = lane % fx.Int32(8)

    # -- buffer resources (exact num_bytes) ----------------------------------
    # A_q rsrc must be sized to the LOGICAL a_quant extent (n_tokens*K_HALF), NOT
    # max_size: padding rows carry m_indices == M (out of range), and the kernel
    # relies on hardware buffer-bounds returning 0 for those OOB row loads so
    # padding output rows stay zero (HIP gemm1_a4w4.cuh:82). max_size (and the
    # max_size=False memref fallback for DLPack's dynamic-shape a_quant) would read
    # garbage past the logical extent into the padding rows. Pass the exact byte
    # count = n_tokens * K_HALF (a_quant is uint8, 1 byte/elem).
    # args arrive as raw i64 device addresses (data_ptr()); build buffer resources
    # straight from the address -- no memref descriptor needed (see _global_base_ptr1).
    aq_num_records = arith.index_cast(T.index, _raw(i32_ntok * fx.Int32(K_HALF)))
    aq_rsrc = buffer_ops.create_buffer_resource_from_addr(
        _raw(fx.Int64(arg_aq)), num_records_bytes=aq_num_records
    )
    _asc_per_mb = max(BM // 32, 1) * kAS_per_chunk_dw * 4
    ascale_num = arith.index_cast(T.index, _raw(i32_total_m_blocks)) * fx.Index(
        _asc_per_mb
    )
    ascale_rsrc = buffer_ops.create_buffer_resource_from_addr(
        _raw(fx.Int64(arg_ascale)), num_records_bytes=ascale_num
    )
    bq_rsrc = buffer_ops.create_buffer_resource_from_addr(
        _raw(fx.Int64(arg_bq)), num_records_bytes=BQ_BYTES
    )
    bscale_rsrc = buffer_ops.create_buffer_resource_from_addr(
        _raw(fx.Int64(arg_bscale)), num_records_bytes=BSCALE_BYTES
    )
    # hidden_states rsrc (inline-quant only): n_tokens*K*sizeof(bf16) bytes (HIP :92-97).
    # Non-inline keeps arg_hidden unused.
    hidden_rsrc = None
    if const_expr(inline_quant):
        hidden_num = arith.index_cast(T.index, _raw(i32_ntok * fx.Int32(K * 2)))
        hidden_rsrc = buffer_ops.create_buffer_resource_from_addr(
            _raw(fx.Int64(arg_hidden)), num_records_bytes=hidden_num
        )

    # -- LDS views (s_aq / s_asc, union-overlapping lds_acc) ------------------
    lds_base = allocator.get_base()
    s_aq = SmemPtr(lds_base, lds_off, T.i8, shape=(kAStages * BM * KH_TILE,))  # 12288
    s_asc = SmemPtr(
        lds_base,
        lds_off + kAStages * BM * KH_TILE,
        T.i8,
        shape=(kSubBlocks * K_TILES_TOTAL * 256,),
    )
    lds_acc = SmemPtr(
        lds_base, lds_off, T.f32, shape=(BM * BN,)
    )  # union overlap with s_aq/s_asc

    # -- cached A rows ---------------------------------------------------------
    # Non-inline (HIP run_one 385-399): cached_actual_row per kSubBlocks, row_off=lane/8.
    # NB: for BM16 non-inline HIP guards on wave<2, but BM16 is inline-only here.
    # Inline (HIP run_one 402-409, kCachedInline=1 for BM16): cached_row_inline[s] =
    #   m_indices[m_row + s*16 + (wave*4 + lane/16)].
    cached_actual_row = []
    cached_row_inline = []
    if const_expr(inline_quant):
        # BM16 inline-quant: single cached row (kCachedInline==1; index [1] never read).
        rcls = wave * fx.Int32(4) + lane_div_16  # wave*4 + lane/16
        cached_row_inline = [
            llvm.load(T.i32, _global_ptr1(arg_mind, (m_row + rcls) * fx.Int32(4)))
        ]
    else:
        for sub in range_constexpr(kSubBlocks):
            idx = m_row + wave * fx.Int32(BM // 4) + fx.Int32(sub * 8) + lane_div_8
            cached_actual_row.append(
                llvm.load(T.i32, _global_ptr1(arg_mind, idx * fx.Int32(4)))
            )

    # -- b_load_s_base[j] (HIP 412-416), readfirstlane'd uniform per wave ------
    N0_HALF = N_OUT // 32
    b_load_s_base = []
    for j in range_constexpr(4):
        if const_expr(interleave):
            col = (
                n_block_idx * fx.Int32(BN) + wave * fx.Int32(BN // 4) + fx.Int32(j * 16)
            )
        else:
            tile_il = n_block_idx * fx.Int32(16) + wave * fx.Int32(4) + fx.Int32(j)
            g = tile_il & fx.Int32(1)
            n0 = tile_il >> fx.Int32(1)
            col = (g * fx.Int32(N0_HALF) + n0) * fx.Int32(16)
        v = (e * fx.Int32(N_OUT) + col) * fx.Int32(K_HALF)
        b_load_s_base.append(rocdl.readfirstlane(T.i32, v))

    # -- b_scale_s_base / _hi (HIP 418-429) -----------------------------------
    if const_expr(interleave):
        mni_base = n_block_idx * fx.Int32(BN // 32) + wave * fx.Int32(BN // 128)
        np_list = [mni_base, mni_base + fx.Int32(1)]
    else:
        np_gate = n_block_idx * fx.Int32(BN // 64) + wave
        np_list = [np_gate, np_gate + fx.Int32(N_OUT // 64)]
    b_scale_s_base, b_scale_s_base_hi = [], []
    for mw in range_constexpr(2):
        base = (
            e * fx.Int32(kBS_per_expert_dw) + np_list[mw] * fx.Int32(kBS_stride_n0_dw)
        ) * fx.Int32(4)
        base = rocdl.readfirstlane(T.i32, base)
        b_scale_s_base.append(base)
        b_scale_s_base_hi.append(base + fx.Int32(16 * kBS_stride_k0_dw * 4))

    # -- register buffers (accumulators + per-stage B / B-scale) ---------------
    accm = [[None] * 4 for _ in range(kMChunks)]  # kMChunks = BM//16
    b = [[[None, None] for _ in range(4)] for _ in range(kStages)]  # kStages=2
    b_scale_v = [[None, None] for _ in range(kStages)]

    # -- A-side data movement helpers (HIP 121-195) ---------------------------
    def issue_a_load_lds(slot, kt):
        for sub in range_constexpr(kSubBlocks):
            lds_row = wave * fx.Int32(BM // 4) + fx.Int32(sub * 8)
            mask = _lds_swizzle_mask(lds_row + lane_div_8)  # ROW_BYTES=BK/2=128
            voffset = ((lane_mod_8 * fx.Int32(16)) ^ mask) + cached_actual_row[
                sub
            ] * fx.Int32(K_HALF)
            base_i32 = fx.Int32(
                memref_dialect.extract_aligned_pointer_as_index(s_aq.get())
            )
            off = fx.Int32(slot * (BM * KH_TILE)) + lds_row * fx.Int32(KH_TILE)
            rocdl.raw_ptr_buffer_load_lds(
                aq_rsrc,
                _lds_ptr3(base_i32, off),
                fx.Int32(16),
                voffset,
                fx.Int32(kt * KH_TILE),
                fx.Int32(0),
                fx.Int32(0),
            )

    def issue_a_ds_read(slot):
        mask = _lds_swizzle_mask(lane_mod_16)
        base_ptr = _lds_base_ptr3(s_aq.get())
        a = [[None, None] for _ in range(kMChunks)]
        for k in range_constexpr(2):
            lds_col = (lane_div_16 * fx.Int32(16) + fx.Int32(k * 64)) ^ mask
            for i in range_constexpr(kMChunks):
                lds_row = lane_mod_16 + fx.Int32(i * 16)
                off = (
                    fx.Int32(slot * (BM * KH_TILE))
                    + lds_row * fx.Int32(KH_TILE)
                    + lds_col
                )
                a[i][k] = llvm.load(T.vec(4, T.i32), _gep3(base_ptr, off))
        return a

    def issue_a_scale_load():
        chunk_base = m_row // fx.Int32(32)
        v16 = (wave * fx.Int32(64) + lane) * fx.Int32(16)
        v4 = (wave * fx.Int32(64) + lane) * fx.Int32(4)
        asc_base = fx.Int32(
            memref_dialect.extract_aligned_pointer_as_index(s_asc.get())
        )
        for sub in range_constexpr(kSubBlocks):
            s_chunk = rocdl.readfirstlane(
                T.i32, (chunk_base + fx.Int32(sub)) * fx.Int32(kAS_per_chunk_dw * 4)
            )
            lds_sub = fx.Int32(sub * kAS_per_chunk_dw * 4)
            rocdl.raw_ptr_buffer_load_lds(
                ascale_rsrc,
                _lds_ptr3(asc_base, lds_sub + wave * fx.Int32(1024)),
                fx.Int32(16),
                v16,
                s_chunk,
                fx.Int32(0),
                fx.Int32(0),
            )
            for d in range_constexpr(3):
                byte_off = 4096 + d * 1024
                s_off = rocdl.readfirstlane(T.i32, s_chunk + fx.Int32(byte_off))
                rocdl.raw_ptr_buffer_load_lds(
                    ascale_rsrc,
                    _lds_ptr3(
                        asc_base, lds_sub + fx.Int32(byte_off) + wave * fx.Int32(256)
                    ),
                    fx.Int32(4),
                    v4,
                    s_off,
                    fx.Int32(0),
                    fx.Int32(0),
                )

    def issue_a_scale_ds_read(kt):
        base_ptr = _lds_base_ptr3(s_asc.get())
        out = []
        for sub in range_constexpr(kSubBlocks):
            lds_dw = (
                fx.Int32(sub * kAS_per_chunk_dw)
                + fx.Int32(kt * 64)
                + lane_div_16 * fx.Int32(16)
                + lane_mod_16
            )
            out.append(llvm.load(T.i32, _gep3(base_ptr, lds_dw * fx.Int32(4))))
        return out  # a_scale_aiter[sub]

    # -- inline-quant A path helpers (HIP 197-310) ----------------------------
    # Used only on the BM16 inline-quant path. SUB is always 0 for BM16.
    # Common per-lane indices (HIP :241-246).
    lib = lane & fx.Int32(3)  # lane & 3
    lane_shr2_and3 = (lane >> fx.Int32(2)) & fx.Int32(3)  # (lane>>2)&3
    r_in_chunk = wave * fx.Int32(4) + lane_div_16  # wave*4 + lane/16

    def inline_quant_load_kt(B128_IDX, kt, row_token):
        """Load 8 bf16 (4xi32) from hidden_rsrc (HIP :197-207)."""
        v_voff = (
            row_token * fx.Int32(K * 2)
            + lane_shr2_and3 * fx.Int32(64)
            + lib * fx.Int32(16)
        )
        s_soff = rocdl.readfirstlane(T.i32, fx.Int32(kt * (BK * 2) + B128_IDX * 256))
        frag = buffer_ops.buffer_load(
            hidden_rsrc,
            v_voff // fx.Int32(4),  # dword offset
            vec_width=4,
            dtype=T.i32,
            soffset_bytes=s_soff,
        )
        return Vec(frag)

    def _inline_quant_core(B128_IDX, SUB, slot, kt, h_v, scale_accum):
        """Shared body of inline_quant_kt / inline_quant_finish_kt (HIP :209-310).
        h_v is a Vec(4,i32). Quantizes 8 bf16 -> 4 packed-fp4 bytes (1 dword) and
        writes to s_Aq[slot]; folds the e8m0 scale into scale_accum (packed path)."""
        h_dw = [fx.Int32(_raw(h_v[j])) for j in range_constexpr(4)]
        # 0x7FFF7FFF clears both bf16 sign bits in the packed dword (-> |bf16|).
        hm = [h_dw[j] & fx.Int32(0x7FFF7FFF) for j in range_constexpr(4)]
        m01 = _pkmax_u16(hm[0], hm[1])
        m23 = _pkmax_u16(hm[2], hm[3])
        m0123 = _pkmax_u16(m01, m23)
        lo = m0123 & fx.Int32(0xFFFF)
        hi = m0123.shrui(fx.Int32(16)) & fx.Int32(0xFFFF)
        local_amax = _umax_i32(lo, hi)  # u16 amax in low 16 bits
        amax_u32 = _inline_dpp_quad_amax(local_amax)
        e8m0 = _inline_e8m0(amax_u32)
        qs = fx.Float32(
            _raw(e8m0 << fx.Int32(23)).bitcast(T.f32)
        )  # uint_as_float(e8m0<<23)
        # pack 8 bf16 -> 4 fp4 bytes (1 u32) via cvt_scalef32_pk_fp4_bf16.
        # src is vector<2xbf16>: build from each i32 dword via a vector bitcast.
        pk = _raw(fx.Int32(0))
        qs_raw = _raw(qs)
        for j in range_constexpr(4):
            src_bf16x2 = _raw(
                Vec.from_elements([h_dw[j]], fx.Int32).bitcast(fx.BFloat16)
            )
            pk = rocdl.cvt_scalef32_pk_fp4_bf16(T.i32, pk, src_bf16x2, qs_raw, j)
        pk = fx.Int32(pk)
        # write pk -> s_Aq[slot][r][((kb_in_kt*16)^mask_r)+b_off] (HIP :247-248).
        r = fx.Int32(SUB * 16) + r_in_chunk
        kb_in_kt = fx.Int32(B128_IDX * 4) + lane_shr2_and3
        mask_r = _lds_swizzle_mask(r)
        b_off = lib * fx.Int32(4)
        aq_base = fx.Int32(memref_dialect.extract_aligned_pointer_as_index(s_aq.get()))
        off = (
            fx.Int32(slot * (BM * KH_TILE))
            + r * fx.Int32(KH_TILE)
            + ((kb_in_kt * fx.Int32(16)) ^ mask_r)
            + b_off
        )
        llvm.StoreOp(_raw(pk), _lds_ptr3(aq_base, off))
        # packed-scale path: fold e8m0 into the accumulated dword (BM16 only path).
        pack_byte = B128_IDX * 2 + SUB
        scale_accum[0] = scale_accum[0] | (e8m0 << fx.Int32(pack_byte * 8))

    def _inline_quant_core_pair(specs, slot, kt, scale_accum):
        """Interleaved N-row variant of _inline_quant_core: emit each pipeline
        stage (abs+pkmax, cross-lane DPP quad-reduce, e8m0) for ALL rows
        back-to-back so the scheduler can hide one row's high-latency DPP /
        umax behind the other row's independent work -- matching HIP gemm1
        BM16's quant interleave (flydsl's row-by-row order leaves s_nop bubbles
        in the DPP latency window). specs = [(B128_IDX, SUB, h_v), ...]."""
        n = len(specs)
        h_dw = [
            [fx.Int32(_raw(h_v[j])) for j in range_constexpr(4)]
            for (_b, _s, h_v) in specs
        ]
        # stage 1: |bf16| + pkmax tree -> per-row local u16 amax
        la = [None] * n
        for i in range_constexpr(n):
            hm = [h_dw[i][j] & fx.Int32(0x7FFF7FFF) for j in range_constexpr(4)]
            m01 = _pkmax_u16(hm[0], hm[1])
            m23 = _pkmax_u16(hm[2], hm[3])
            m0123 = _pkmax_u16(m01, m23)
            lo = m0123 & fx.Int32(0xFFFF)
            hi = m0123.shrui(fx.Int32(16)) & fx.Int32(0xFFFF)
            la[i] = _umax_i32(lo, hi)
        # stage 2: DPP quad-reduce, INTERLEAVED across rows (hide DPP latency)
        a = [fx.Int32(_raw(la[i])) for i in range_constexpr(n)]
        s1 = [
            fx.Int32(
                dpp_utils.update_dpp_i32(_raw(a[i]), _raw(a[i]), 0xB1, 0xF, 0xF, True)
            )
            for i in range_constexpr(n)
        ]
        a = [_umax_i32(a[i], s1[i]) for i in range_constexpr(n)]
        s2 = [
            fx.Int32(
                dpp_utils.update_dpp_i32(_raw(a[i]), _raw(a[i]), 0x4E, 0xF, 0xF, True)
            )
            for i in range_constexpr(n)
        ]
        a = [_umax_i32(a[i], s2[i]) for i in range_constexpr(n)]
        # stage 3: e8m0 per row
        e8 = [_inline_e8m0(a[i]) for i in range_constexpr(n)]
        # stage 4: cvt pack + LDS store + scale fold per row
        for i in range_constexpr(n):
            B128_IDX, SUB, _hv = specs[i]
            qs_raw = _raw(fx.Float32(_raw(e8[i] << fx.Int32(23)).bitcast(T.f32)))
            pk = _raw(fx.Int32(0))
            for j in range_constexpr(4):
                src_bf16x2 = _raw(
                    Vec.from_elements([h_dw[i][j]], fx.Int32).bitcast(fx.BFloat16)
                )
                pk = rocdl.cvt_scalef32_pk_fp4_bf16(T.i32, pk, src_bf16x2, qs_raw, j)
            pk = fx.Int32(pk)
            r = fx.Int32(SUB * 16) + r_in_chunk
            kb_in_kt = fx.Int32(B128_IDX * 4) + lane_shr2_and3
            mask_r = _lds_swizzle_mask(r)
            b_off = lib * fx.Int32(4)
            aq_base = fx.Int32(
                memref_dialect.extract_aligned_pointer_as_index(s_aq.get())
            )
            off = (
                fx.Int32(slot * (BM * KH_TILE))
                + r * fx.Int32(KH_TILE)
                + ((kb_in_kt * fx.Int32(16)) ^ mask_r)
                + b_off
            )
            llvm.StoreOp(_raw(pk), _lds_ptr3(aq_base, off))
            pack_byte = B128_IDX * 2 + SUB
            scale_accum[0] = scale_accum[0] | (e8[i] << fx.Int32(pack_byte * 8))

    def inline_quant_kt(B128_IDX, SUB, slot, kt, row_token, scale_accum):
        h_v = inline_quant_load_kt(B128_IDX, kt, row_token)
        _inline_quant_core(B128_IDX, SUB, slot, kt, h_v, scale_accum)

    def inline_quant_finish_kt(B128_IDX, SUB, slot, kt, h_v, scale_accum):
        _inline_quant_core(B128_IDX, SUB, slot, kt, h_v, scale_accum)

    def inline_quant_pack_write(kt, scale_accum):
        """Write packed scale dword to s_Ascale (HIP :261-266)."""
        lane_tgt = lane_shr2_and3 * fx.Int32(16) + r_in_chunk
        asc_base = fx.Int32(
            memref_dialect.extract_aligned_pointer_as_index(s_asc.get())
        )
        off = fx.Int32(kt * 256) + lane_tgt * fx.Int32(4)
        llvm.StoreOp(_raw(scale_accum[0]), _lds_ptr3(asc_base, off))

    # -- B-side data movement helpers (HIP 312-332) ---------------------------
    def issue_b_load_j(b_slot, K_C, j):
        v = (
            (lane_div_16 * fx.Int32(256))
            + (lane_mod_16 * fx.Int32(16))
            + fx.Int32(K_C * 2048)
        )
        for half in range_constexpr(2):
            frag = buffer_ops.buffer_load(
                bq_rsrc,
                (v + fx.Int32(half * 1024)) // fx.Int32(4),
                vec_width=4,
                dtype=T.i32,
                cache_modifier=b_aux,
                soffset_bytes=b_load_s_base[j],
            )
            b_slot[j][half] = Vec(frag)

    def issue_b_scale_load(bs_slot, K_C):
        v = ((lane_div_16 * fx.Int32(16)) + lane_mod_16) * fx.Int32(4)
        K_C_HI = K_C // 16
        imm = (K_C - K_C_HI * 16) * (
            kBS_stride_k0_dw * 4
        )  # Python int (K_C is constexpr)
        for mw in range_constexpr(2):
            s_off = b_scale_s_base[mw] if K_C_HI == 0 else b_scale_s_base_hi[mw]
            bs_slot[mw] = buffer_ops.buffer_load(
                bscale_rsrc,
                (v + fx.Int32(imm)) // fx.Int32(4),
                vec_width=1,
                dtype=T.i32,
                soffset_bytes=s_off,
            )

    # -- MFMA cluster (HIP 334-375, BM!=16 non-AGPR else branch) --------------
    # Generalized over kSubBlocks (BM32 -> 1 sub, BM128 -> 4 subs).
    # HIP uses an AGPR init form for BM128's zero-init mfmas (kUseAGPR=BM==128) to
    # relieve register pressure; the rest stay VGPR. FlyDSL emits plain
    # rocdl.mfma_scale throughout and lets the backend allocate AGPRs under pressure
    # -- numerically identical.
    mfma_ty = T.f32x4
    zero4 = Vec.filled(4, 0.0, fx.Float32)

    # Unlike gemm2's internal J-loop, gemm1 issues one J per call so the K-loop
    # pipeline can interleave issue_b_load_j between J iterations (HIP run_one 465-552).
    def mfma_cluster(b_slot, a, a_scale, bs_slot, J, init):
        if const_expr(interleave):
            mni = J // 2
            in_b = J % 2
        else:
            mni = J % 2
            in_b = J // 2
        sb = bs_slot[mni]
        bJ0, bJ1 = b_slot[J][0], b_slot[J][1]
        # BM16 single-chunk (HIP :338-345): kMChunks=1, only i0 exists. Two mfma on
        # chunk 0 (op_sel_a 0 for k0, 2 for k1); the i1 ops would index out of range.
        if const_expr(kMChunks == 1):
            sa = a_scale[0]
            if const_expr(init):
                accm[0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    mfma_ty, [a[0][0], bJ0, zero4, 4, 4, 0, sa, 0 + in_b, sb]
                )
            else:
                accm[0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    mfma_ty, [a[0][0], bJ0, accm[0][J], 4, 4, 0, sa, 0 + in_b, sb]
                )
            accm[0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                mfma_ty, [a[0][1], bJ1, accm[0][J], 4, 4, 2, sa, 2 + in_b, sb]
            )
            return
        # Per sub: i0=sub*2, i1=sub*2+1; sa=a_scale[sub]. op_sel_a is the FIXED
        # within-128 K-fragment selector {0,1,2,3}, NOT offset by sub (HIP :350-371).
        for sub in range_constexpr(kSubBlocks):
            i0 = sub * 2 + 0
            i1 = sub * 2 + 1
            sa = a_scale[sub]
            if const_expr(init):
                accm[i0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    mfma_ty, [a[i0][0], bJ0, zero4, 4, 4, 0, sa, 0 + in_b, sb]
                )
                accm[i1][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    mfma_ty, [a[i1][0], bJ0, zero4, 4, 4, 1, sa, 0 + in_b, sb]
                )
            else:
                accm[i0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    mfma_ty, [a[i0][0], bJ0, accm[i0][J], 4, 4, 0, sa, 0 + in_b, sb]
                )
                accm[i1][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    mfma_ty, [a[i1][0], bJ0, accm[i1][J], 4, 4, 1, sa, 0 + in_b, sb]
                )
            accm[i0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                mfma_ty, [a[i0][1], bJ1, accm[i0][J], 4, 4, 2, sa, 2 + in_b, sb]
            )
            accm[i1][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                mfma_ty, [a[i1][1], bJ1, accm[i1][J], 4, 4, 3, sa, 2 + in_b, sb]
            )

    # ---- prologue: stages 0,1 (HIP 431-463) ----
    _relax_prologue = (BM == 128) and not inline_quant
    if const_expr(not inline_quant):
        issue_a_scale_load()
    for K_C in range_constexpr(kStages):
        if const_expr(inline_quant):
            # HIP :447-455 (BM16 inline branch): 2x inline_quant_kt<B128,0> (packed scale)
            # interleaved with 4x issue_b_load_j, then inline_quant_pack_write.
            scale_accum = [fx.Int32(0)]
            inline_quant_kt(0, 0, K_C, K_C, cached_row_inline[0], scale_accum)
            issue_b_load_j(b[K_C], K_C, 0)
            issue_b_load_j(b[K_C], K_C, 1)
            inline_quant_kt(1, 0, K_C, K_C, cached_row_inline[0], scale_accum)
            issue_b_load_j(b[K_C], K_C, 2)
            issue_b_load_j(b[K_C], K_C, 3)
            inline_quant_pack_write(K_C, scale_accum)
        else:
            issue_a_load_lds(K_C, K_C)
            if const_expr(not _relax_prologue):
                for j in range_constexpr(4):
                    issue_b_load_j(b[K_C], K_C, j)
        if const_expr(not _relax_prologue):
            issue_b_scale_load(b_scale_v[K_C], K_C)
    if const_expr(_relax_prologue):
        rocdl.sched_barrier(0)
        for K_C in range_constexpr(kStages):
            for j in range_constexpr(4):
                issue_b_load_j(b[K_C], K_C, j)
            issue_b_scale_load(b_scale_v[K_C], K_C)

    # ---- main loop: OFFSET in [0,26) (HIP 465-552 non-inline else) ----
    for OFFSET in range_constexpr(kUnroll):
        K_C = kStages + OFFSET
        read_slot = OFFSET % kAStages
        write_slot = K_C % kAStages
        slot_b = OFFSET % kStages
        gpu.barrier()  # __syncthreads (HIP 472)
        if const_expr(BM == 128):
            asc_cur = issue_a_scale_ds_read(K_C - kStages)
            a_cur = issue_a_ds_read(read_slot)
        else:
            a_cur = issue_a_ds_read(read_slot)
            asc_cur = issue_a_scale_ds_read(K_C - kStages)
        if const_expr(not inline_quant):
            issue_a_load_lds(write_slot, K_C)
        # Inline path (HIP :523-528): pre-load the next K-tile's hidden into regs so
        # the quant (inline_quant_finish_kt) can overlap with the mfma cluster below.
        if const_expr(inline_quant):
            h_v0 = inline_quant_load_kt(0, K_C, cached_row_inline[0])
            h_v1 = inline_quant_load_kt(1, K_C, cached_row_inline[0])
            rocdl.sched_barrier(0)
        for J in range_constexpr(4):
            # HIP :531-539 gates the sched_barrier/s_setprio scheduling hints
            # around the mfma with `if constexpr(BM != 128)`; the post-mfma and
            # post-b-load sched_barrier(0) (HIP :540, :542) are unconditional.
            if const_expr(BM != 128):
                rocdl.sched_barrier(0)
                rocdl.s_setprio(1)
            mfma_cluster(
                b[slot_b], a_cur, asc_cur, b_scale_v[slot_b], J, init=(OFFSET == 0)
            )
            if const_expr(BM != 128):
                rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
            issue_b_load_j(b[slot_b], K_C, J)
            rocdl.sched_barrier(0)
        issue_b_scale_load(b_scale_v[slot_b], K_C)
        # Inline quant of the pre-loaded h_v0/h_v1 into write_slot (HIP :545-550).
        if const_expr(inline_quant):
            scale_accum = [fx.Int32(0)]
            _inline_quant_core_pair(
                [(0, 0, h_v0), (1, 0, h_v1)], write_slot, K_C, scale_accum
            )
            inline_quant_pack_write(K_C, scale_accum)

    # ---- drain: S in [0,2) (HIP 554-565) ----
    for S in range_constexpr(kStages):
        kt = K_TILES_TOTAL - kStages + S
        gpu.barrier()
        if const_expr(BM == 128):
            asc_cur = issue_a_scale_ds_read(kt)
            a_cur = issue_a_ds_read(kt % kAStages)
        else:
            a_cur = issue_a_ds_read(kt % kAStages)
            asc_cur = issue_a_scale_ds_read(kt)
        for J in range_constexpr(4):
            mfma_cluster(
                b[kt % kStages], a_cur, asc_cur, b_scale_v[kt % kStages], J, init=False
            )

    gpu.barrier()
    s_aq._view_cache = None
    s_asc._view_cache = None
    lds_acc._view_cache = None

    # -- epilog: cshuffle -> SwiGLU -> fp4 + e8m0 requant (HIP apply_cshuffle_quant_epilog 39-122) --
    # NOTE: the e8m0 scale is computed here and stored to arg_ascaleout in the kk==0 block below.
    wave_n = (
        wave  # NUM_N_BLOCKS path: wave_n == wave (HIP run_one passes wave as wave_n)
    )
    lds_acc_base = _lds_base_ptr3(lds_acc.get())

    # cshuffle store (epilog 39-53): scalar f32 stores of accm into lds_acc.
    for i in range_constexpr(kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for J in range_constexpr(4):
            is_up = (J % 2) == 1  # Python const
            J_local = J // 2
            col_local = wave_n * fx.Int32(32) + fx.Int32(J_local * 16) + lane_mod_16
            lds_col = (fx.Int32(128) + col_local) if is_up else col_local
            vec = Vec(accm[i][J])
            for v in range_constexpr(4):
                idx = (row_base + fx.Int32(v)) * fx.Int32(BN) + lds_col
                llvm.StoreOp(_raw(vec[v]), _gep3(lds_acc_base, idx * fx.Int32(4)))

    gpu.barrier()

    # per-mr SwiGLU + amax + e8m0 + fp4 pack + store (epilog 57-122).
    tx_i32 = arith.index_cast(T.i32, gpu.thread_id("x"))
    m_lane = tx_i32 // fx.Int32(16)
    n_lane = tx_i32 % fx.Int32(16)
    wave_grp = n_lane // fx.Int32(4)
    kk = n_lane % fx.Int32(4)

    aqout_base = _global_base_ptr1(arg_aqout)
    scales_per_mr = [None] * M_REPS

    for mr in range_constexpr(M_REPS):
        row_local = fx.Int32(mr * 16) + m_lane

        # read 8 gate + 8 up f32 from lds_acc (epilog 75-83).
        gate_vs = [None] * 8
        up_vs = [None] * 8
        for ee in range_constexpr(8):
            col_in_grp = fx.Int32(8) * kk + fx.Int32(ee)
            gate_col = wave_grp * fx.Int32(32) + col_in_grp
            up_col = fx.Int32(128) + gate_col
            gate_off = (row_local * fx.Int32(BN) + gate_col) * fx.Int32(4)
            up_off = (row_local * fx.Int32(BN) + up_col) * fx.Int32(4)
            gate_vs[ee] = fx.Float32(llvm.load(T.f32, _gep3(lds_acc_base, gate_off)))
            up_vs[ee] = fx.Float32(llvm.load(T.f32, _gep3(lds_acc_base, up_off)))
        result = _silu_mul_batch(gate_vs, up_vs)

        # local amax over the 8 results (epilog 91-95).
        local_max = _fabs_f32(result[0])
        for ee in range_constexpr(1, 8):
            local_max = local_max.maximumf(_fabs_f32(result[ee]))
        # quad amax reduce across 4 lanes (epilog 96-101: DPP 0xB1 then 0x4E).
        # amax is order-independent => xor shuffles over offsets 1,2 are equivalent.
        local_max = local_max.maximumf(local_max.shuffle_xor(fx.Int32(1), fx.Int32(64)))
        local_max = local_max.maximumf(local_max.shuffle_xor(fx.Int32(2), fx.Int32(64)))

        e8m0, qscale = _e8m0_from_amax(local_max)
        scales_per_mr[mr] = e8m0  # consumed by the kk==0 scale store below

        # pack 4 fp4 bytes via the HW cvt intrinsic (epilog 108-116):
        #   packed = cvt_scalef32_pk_fp4_f32(packed, result[2w], result[2w+1], qscale, w)
        # rocdl.cvt_scalef32_pk_fp4_f32 maps 1:1 to
        # __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(old_u32, a, b, scale, word_sel).
        packed_i32 = _raw(fx.Int32(0))
        qscale_raw = _raw(qscale)
        for w in range_constexpr(4):
            packed_i32 = rocdl.cvt_scalef32_pk_fp4_f32(
                T.i32,
                packed_i32,
                _raw(result[2 * w]),
                _raw(result[2 * w + 1]),
                qscale_raw,
                w,
            )
        packed = fx.Int32(packed_i32)

        # nontemporal store to inter_sorted_quant (epilog 118-121).
        byte_pos = (
            n_block_idx * fx.Int32(BN_INT // 2)
            + wave_grp * fx.Int32(16)
            + kk * fx.Int32(4)
        )
        out_row = m_row + row_local
        store_off = out_row * fx.Int32(K_G2_HALF) + byte_pos
        llvm.StoreOp(
            _raw(packed),
            _gep1(aqout_base, store_off),
            alignment=4,
            nontemporal=True,
        )

    # -- e8m0 scale store (epilog 124-144, BM != 16 branch; over kSubBlocks) ---
    # Only lane-group kk==0 writes. Per sub: pack scales_per_mr[2*sub], [2*sub+1]
    # into a u16 and store at byte offset dword_off*4 + ikxdl*2 of a_scale_out,
    # using the OUTPUT-scale chunk stride OUT_AS_PER_CHUNK_DW(=128).
    ascaleout_base = _global_base_ptr1(arg_ascaleout)
    # Native Python `if` on the runtime kk==0 lane predicate. _gemm1_body is
    # @flyc.jit, so its AST is rewritten: this lowers to the same
    # scf.if(cmpi(eq), [], has_else=False) the manual scf.IfOp built. (When
    # called inside the kernel trace, @flyc.jit just runs the rewritten body
    # inline -- ir.Context.current is set, so no nested compilation.)
    if kk == fx.Int32(0):
        ku = n_block_idx >> fx.Int32(1)
        ikxdl = n_block_idx & fx.Int32(1)
        if const_expr(BM == 16):
            # BM16 (HIP mxfp4_epilogs.hpp:127-132): write a SINGLE LOW byte
            # scales_per_mr[0]; chunk=m_block_idx, dword_off uses kAS_per_chunk_dw.
            chunk = m_block_idx
            dword_off = (
                chunk * fx.Int32(OUT_AS_PER_CHUNK_DW)
                + ku * fx.Int32(64)
                + wave_grp * fx.Int32(16)
                + m_lane
            )
            addr = dword_off * fx.Int32(4) + ikxdl * fx.Int32(2)
            byte_i8 = arith.TruncIOp(T.i8, _raw(scales_per_mr[0])).result
            llvm.StoreOp(byte_i8, _gep1(ascaleout_base, addr), alignment=1)
        else:
            for sub in range_constexpr(kSubBlocks):
                chunk = m_block_idx * fx.Int32(kSubBlocks) + fx.Int32(sub)
                dword_off = (
                    chunk * fx.Int32(OUT_AS_PER_CHUNK_DW)
                    + ku * fx.Int32(64)
                    + wave_grp * fx.Int32(16)
                    + m_lane
                )
                pair_i32 = scales_per_mr[sub * 2 + 0] | (
                    scales_per_mr[sub * 2 + 1] << fx.Int32(8)
                )
                pair_i16 = arith.TruncIOp(T.i16, _raw(pair_i32)).result
                addr = dword_off * fx.Int32(4) + ikxdl * fx.Int32(2)
                llvm.StoreOp(
                    pair_i16,
                    _gep1(ascaleout_base, addr),
                    alignment=2,
                )


def _bm_constants(BM, K_TILES_TOTAL=K_TILES_TOTAL):
    """BM-derived compile-time constants (mirror gemm1_a4w4.cuh:49-72).

    s_Ascale (and thus the LDS union) depends on K via K_TILES_TOTAL, so K is a
    param (KIMI default keeps the old sizes: BM32->32768, BM128->131072, BM16->16384).
    """
    kAStages = 2 if BM == 128 else 3
    kSubBlocks = 1 if BM < 32 else BM // 32  # HIP :61
    kMChunks = BM // 16  # HIP :62
    # LDS union (bytes): max(s_Aq[kAStages][BM][KH_TILE] + s_Ascale[kSubBlocks*K_TILES_TOTAL*256],
    #                        lds_acc[BM*BN] f32).
    s_aq_bytes = kAStages * BM * KH_TILE
    s_asc_bytes = kSubBlocks * K_TILES_TOTAL * 256
    lds_acc_bytes = BM * BN * 4
    lds_bytes = max(s_aq_bytes + s_asc_bytes, lds_acc_bytes)
    return kAStages, kSubBlocks, kMChunks, lds_bytes


def compile_gemm1_a4w4_port(
    BM=32,
    use_nt=True,
    inline_quant=False,
    D_HIDDEN=K,
    D_INTER=INTER,
    NE=NE,
    TOPK=TOPK,
    interleave=True,
):
    print(
        f"[PORT-FLYDSL-GEMM1] compile_gemm1_a4w4_port ENTERED "
        f"BM={BM} use_nt={use_nt} inline_quant={inline_quant} "
        f"D_HIDDEN={D_HIDDEN} D_INTER={D_INTER} NE={NE} interleave={interleave}",
        flush=True,
    )
    # Supported Phase 2 combos.
    if (BM, use_nt, inline_quant) not in {
        (32, True, False),
        (32, False, False),
        (64, False, False),
        (128, False, False),
        (16, True, True),
    }:
        raise AssertionError(
            f"unsupported gemm1 variant (BM={BM}, use_nt={use_nt}, inline_quant={inline_quant})"
        )

    # K = contraction dim = hidden_size. Parametrized; defaults to KIMI's 7168.
    _K = D_HIDDEN
    assert _K % BK == 0, f"D_HIDDEN (K) must be a multiple of {BK}, got {_K}"
    # INTER = OUTPUT/intermediate dim. Parametrized; defaults to KIMI's 512.
    # 2*INTER (=N_OUT) must be a multiple of BN(256), i.e. INTER % 128 == 0.
    _INTER = D_INTER
    _N_OUT = n_out_for(_INTER)
    assert (
        _N_OUT % BN == 0
    ), f"2*D_INTER (N_OUT) must be a multiple of {BN}, got {_N_OUT}"
    # NE (experts) and TOPK are parametrized; TOPK is host-side only (grid bound).
    _NE = NE
    # Derive ALL K-dependent compile constants from the arg (KIMI default -> old values).
    _K_HALF = k_half_for(_K)
    _K_TILES_TOTAL = k_tiles_total_for(_K)
    _kUnroll = kunroll_for(_K)
    _kAS_per_chunk_dw = kas_per_chunk_dw_for(_K)
    _kBS_stride_n0_dw = kbs_stride_n0_dw_for(_K)
    # INTER/NE-dependent compile constants (kBS_per_expert_dw / BQ / BSCALE depend
    # on INTER and NE too; the OUTPUT-side block is purely INTER-derived).
    _kBS_per_expert_dw = kbs_per_expert_dw_for(_K, _INTER)
    _BQ_BYTES = bq_bytes_for(_K, _INTER, _NE)
    _ASCALE_BYTES = ascale_bytes_for(_K)
    _BSCALE_BYTES = bscale_bytes_for(_K, _INTER, _NE)
    _NUM_N_BLOCKS = num_n_blocks_for(_INTER)
    _OUT_AS_PER_CHUNK_DW = out_as_per_chunk_dw_for(_INTER)
    _K_G2_HALF = k_g2_half_for(_INTER)

    kAStages, kSubBlocks, kMChunks, lds_bytes = _bm_constants(BM, _K_TILES_TOTAL)

    variant_tag = "iq" if inline_quant else ("nt" if use_nt else "cached")
    # Tag with H/INTER/NE so different shape specializations get distinct
    # kernel/smem symbols (so KIMI and non-KIMI instances never collide).
    gu_tag = "il" if interleave else "sep"
    name_suffix = f"h{_K}_i{_INTER}_ne{_NE}_bm{BM}_{variant_tag}_{gu_tag}"

    allocator = SmemAllocator(
        None, arch="gfx950", global_sym_name=f"gemm1port_smem_{name_suffix}"
    )
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = (
        lds_off + lds_bytes
    )  # lds_bytes from _bm_constants (KIMI: BM32->32768, BM128->131072, BM16->16384)

    @flyc.kernel(name=f"gemm1_a4w4_port_{name_suffix}", known_block_size=[256, 1, 1])
    def gemm1_kernel(
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_mind: fx.Int64,
        i32_ntok: fx.Int32,
        arg_aqout: fx.Int64,
        arg_ascaleout: fx.Int64,
        arg_hidden: fx.Int64,
    ):
        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        tx_i32 = arith.index_cast(T.i32, tx)
        bx_i32 = arith.index_cast(T.i32, bx)
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))
        # bound: total_m_blocks = cumsum[0]//BM ; bound = total_m_blocks*NUM_N_BLOCKS
        cumsum0 = llvm.load(T.i32, _global_ptr1(arg_cumsum, fx.Int32(0)))
        total_m_blocks = cumsum0 // fx.Int32(BM)
        bound = total_m_blocks * fx.Int32(_NUM_N_BLOCKS)
        # Native Python `if` + comparison: the @flyc.kernel AST rewriter lowers
        # `fx.Int32 < fx.Int32` to arith.cmpi(slt) (Int32 is signed) and the
        # guard to scf.if(cond, [], has_else=False) -- identical IR to the
        # manual arith.cmpi/scf.IfOp/InsertionPoint/YieldOp.
        if fx.Int32(bx_i32) < bound:
            _gemm1_body(
                allocator,
                lds_off,
                arg_aq,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_mind,
                arg_aqout,
                arg_ascaleout,
                arg_hidden,
                bx_i32,
                lane,
                wave,
                use_nt,
                i32_ntok,
                total_m_blocks,
                BM=BM,
                kAStages=kAStages,
                kSubBlocks=kSubBlocks,
                kMChunks=kMChunks,
                inline_quant=inline_quant,
                K=_K,
                K_HALF=_K_HALF,
                K_TILES_TOTAL=_K_TILES_TOTAL,
                kUnroll=_kUnroll,
                kAS_per_chunk_dw=_kAS_per_chunk_dw,
                kBS_stride_n0_dw=_kBS_stride_n0_dw,
                kBS_per_expert_dw=_kBS_per_expert_dw,
                BQ_BYTES=_BQ_BYTES,
                ASCALE_BYTES=_ASCALE_BYTES,
                BSCALE_BYTES=_BSCALE_BYTES,
                N_OUT=_N_OUT,
                NUM_N_BLOCKS=_NUM_N_BLOCKS,
                OUT_AS_PER_CHUNK_DW=_OUT_AS_PER_CHUNK_DW,
                K_G2_HALF=_K_G2_HALF,
                interleave=interleave,
            )

    @flyc.jit
    def launch_gemm1(
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_mind: fx.Int64,
        i32_ntok: fx.Int32,
        i32_grid: fx.Int32,
        arg_aqout: fx.Int64,
        arg_ascaleout: fx.Int64,
        arg_hidden: fx.Int64,
        stream: fx.Stream,
    ):
        from flydsl.compiler.kernel_function import CompilationContext

        ctx = CompilationContext.get_current()
        allocator.finalized = False
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        grid_x = arith.index_cast(T.index, i32_grid)
        gemm1_kernel(
            arg_aq,
            arg_ascale,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_cumsum,
            arg_mind,
            i32_ntok,
            arg_aqout,
            arg_ascaleout,
            arg_hidden,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm1
