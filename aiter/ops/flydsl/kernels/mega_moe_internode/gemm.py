# SPDX-License-Identifier: MIT
"""GEMM bodies of MegaMoEInternode.

* GMM1 (``_gemm1_body``, run inside stage1): gate/up + activation + MXFP4 quant
  of h1, one m-block of ``32 * tile_group`` rows per job, BN = 256.
* GMM2 (``gemm2_compute_v2``, run inside stage2 kernel1): MegaMoEV2's GEMM2
  reduced to the one path this operator runs -- fp4 A through LDS DMA, fp4 B
  streamed through a 2-stage register pipeline, BM = 32, BK = 256.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Float4E2M1FN, Float8E4M3FN, Float32, Int32, T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import (
    batched_silu_swiglu,
    batched_situv2,
    fclamp_f32,
    fmin_f32,
    situv2_consts,
)
from aiter.ops.flydsl.kernels.layout_utils import crd2idx

from .. import buffer_ops
from ..mxfp4_gemm_common import (
    _buffer_rsrc,
    _fabs_f32,
    _inline_dpp_quad_amax,
    _lds_ptr3,
    _lds_swizzle_mask,
    _raw,
    bq_bytes_for,
    flat_buffer_view,
    global_typed_ptr,
    k_half_for,
    k_tiles_total_for,
    kas_per_chunk_dw_for,
    kBS_stride_k0_dw,
    kmchunks_for,
    kStages,
    lds_acc_bytes_for,
    lds_dma_atom_128,
    lds_dma_dst,
    lds_swizzle_mask_f8,
    lds_vec_load,
    num_n_blocks_for,
)
from ..tensor_shim import buf_copy_load, ptr_buf_tensor
from .common import (
    _e8m0_from_amax,
    bscale_bytes_for,
    kbs_per_expert_dw_for,
    kbs_stride_n0_dw_for,
)


def _global_i32_ptr(addr_i64):
    ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=4
    )
    return fx.inttoptr(ptr_ty, fx.Int64(addr_i64))


def _global_i32_at(addr_i64, idx):
    # fx.ptr_load/add_offset for a plain scalar read -- no tiling needed, so
    # skip the Tensor/tile/register-fragment machinery fx.copy requires.
    return _global_i32_ptr(addr_i64)[idx]


def _global_scalar_tiles(addr_i64, numeric_cls, num_elems):
    ptr_ty = fx.PointerType.get(
        numeric_cls.ir_type,
        address_space=fx.AddressSpace.Global,
        alignment=numeric_cls.width // 8,
    )
    ptr = fx.inttoptr(ptr_ty, fx.Int64(addr_i64))
    flat = fx.make_view(ptr, fx.make_layout(num_elems, 1))
    return fx.logical_divide(flat, fx.make_layout(1, 1))


def _scalar_store(tiles, idx, value, numeric_cls):
    atom = fx.make_copy_atom(fx.UniversalCopy(numeric_cls.width), numeric_cls)
    reg_lay = fx.make_layout(1, 1)
    r = fx.make_rmem_tensor(reg_lay, numeric_cls)
    r.store(fx.Vector.from_elements([numeric_cls(value)], numeric_cls))
    fx.copy(atom, r, fx.slice(tiles, (None, idx)))


def _layout_idx(layout, *coords):
    idx_coords = [fx.Int64(c) for c in coords]
    flat = crd2idx(idx_coords, layout)
    return fx.Int32(flat)


LOG2E = 1.4426950408889634


def _silu_mul_batch(gs, us):
    e = [fx.Float32(rocdl.exp2(T.f32, _raw(g * fx.Float32(-LOG2E)))) for g in gs]
    sig = [fx.Float32(rocdl.rcp(T.f32, _raw(fx.Float32(1.0) + ei))) for ei in e]
    return [gs[i] * sig[i] * us[i] for i in range(len(gs))]


def _gate_up_batch(
    gs,
    us,
    *,
    act="silu",
    swiglu_limit=None,
    situ_beta=4.0,
    situ_linear_beta=25.0,
):
    """Compile-time activation epilogue for the fused A4W4 port.

    Keep the original SiLU implementation as the default fast path so existing
    code objects remain bitwise stable.  SwiGLU and SiTUv2 reuse the same
    batched transcendental helpers as AITER's generic mixed-MoE kernel.
    """

    if act == "silu" and swiglu_limit is None:
        return _silu_mul_batch(gs, us)
    if act == "silu":
        # SiLU with DSV4 clamping: gate <= L and -L <= up <= L, then exp2 + rcp.
        # batched_silu_swiglu uses v_tanh, available only on gfx1250; gfx950 compilation fails.
        limit = float(swiglu_limit)
        if math.isinf(limit):
            return _silu_mul_batch(gs, us)
        hi, lo = fx.Float32(limit), fx.Float32(-limit)
        return _silu_mul_batch(
            [fmin_f32(g, hi) for g in gs], [fclamp_f32(u, lo, hi) for u in us]
        )

    pairs = list(zip(gs, us))
    if act in ("silu", "swiglu"):
        limit = (
            7.0
            if act == "swiglu" and swiglu_limit is None
            else float("inf") if swiglu_limit is None else float(swiglu_limit)
        )
        return batched_silu_swiglu(
            pairs,
            swiglu=act == "swiglu",
            limit_f32=fx.Float32(limit),
            neg_limit_f32=fx.Float32(-limit),
            range_constexpr=range_constexpr,
        )
    if act == "situv2":
        consts = situv2_consts(
            fx.Float32(float(situ_beta)),
            fx.Float32(float(situ_linear_beta)),
        )
        return batched_situv2(pairs, consts=consts, range_constexpr=range_constexpr)
    raise ValueError(f"unsupported A4W4 GMM1 activation {act!r}")


@flyc.jit
def _gemm1_body_m16(
    lds_raw_ptr,
    arg_aq,
    arg_ascale,
    arg_bq,
    arg_bscale,
    arg_eids,
    arg_mind,
    arg_aqout,
    arg_ascaleout,
    bx_i32,
    lane,
    wave,
    i32_ntok,
    *,
    BM,
    BN,
    BK,
    K,
    N_OUT,
    NE,
    act="silu",
    swiglu_limit=None,
    situ_beta=4.0,
    situ_linear_beta=25.0,
    aqin_row_base=False,
    aqout_row_base=False,
    NW=4,
    h1_wt=False,
    a_dtype="fp4",
):
    # h1_wt: store h1 and its scales write-through (SC0|NT|SC1, as Stage2's
    # push) so a GPU-local reader on another XCD sees them once the wave's
    # vmcnt retires, without an L2 write-back (single fused kernel: GEMM2
    # starts on a tile group as soon as its GMM1 jobs are counted).
    # aqout_row_base: the h1 output exceeds the i32 byte range (mtpr > 4096), so
    # the m-block's row offset goes into the 64-bit base and the store index only
    # covers the BM rows of this block. Off: the original single-base addressing.
    # aqin_row_base: compact EP32 inputs can exceed signed-i32 byte offsets.
    # Use 64-bit global addresses for those loads; smaller inputs retain the
    # direct buffer-to-LDS DMA path and its existing instruction sequence.
    is_f8_a = a_dtype == "fp8"
    KH_TILE = BK if is_f8_a else BK // 2
    K_HALF = k_half_for(K)
    A_ROW_BYTES = K if is_f8_a else K_HALF
    K_TILES_TOTAL = k_tiles_total_for(K, BK)
    kAS_per_chunk_dw = kas_per_chunk_dw_for(K)
    kBS_stride_n0_dw = kbs_stride_n0_dw_for(K)
    kBS_per_expert_dw = kbs_per_expert_dw_for(N_OUT, K)
    BQ_BYTES = bq_bytes_for(NE, N_OUT, K)
    BSCALE_BYTES = bscale_bytes_for(NE, N_OUT, K)
    NUM_N_BLOCKS = num_n_blocks_for(N_OUT, BN)
    inter = N_OUT // 2
    OUT_AS_PER_CHUNK_DW = kas_per_chunk_dw_for(inter)
    K_G2_HALF = inter if is_f8_a else k_half_for(inter)
    kAStages, kSubBlocks, kMChunks, _ = _bm_constants(BM, BN, KH_TILE, K_TILES_TOTAL)

    BN_INT = BN // 2
    b_aux = 2
    M_REPS = BM // 16
    # Four 16-column N subtiles per wave: two each for gate and up.
    NJ = BN // 64
    assert BN == 256, f"GMM1 BN must be 256, got {BN}"
    assert BM in (32, 64, 128), f"GMM1 BM must be 32, 64 or 128, got {BM}"
    # NW = waves per CTA.  4: each wave computes all BM rows of a 64-column
    # slice (32 gate + 32 up columns).  8 (PR6881 layout, two waves per SIMD at
    # <= 256 registers): each wave takes all BM rows of a 32-column slice
    # (16 gate + 16 up), so the CTA still reads B once; the B-scale word holds an
    # n0 pair, the odd wave shifts its bytes down so the MFMA byte select stays
    # compile-time.  Every output element keeps its lane and K order: results
    # are bit-identical to 4 waves.
    assert NW in (4, 8), f"GMM1 NW must be 4 or 8, got {NW}"
    W8 = NW == 8
    NJW = NJ // 2 if W8 else NJ  # 16-column B tiles per wave
    # A LDS DMA: 8-row calls, so at most BM // 8 waves share it (BM 32 x 8
    # waves: waves 0-3 only; the others' vmcnt waits just get looser).
    DMA_ROWS = 4 if is_f8_a else 8
    A_NW = min(NW, BM // DMA_ROWS)
    A_ROWS_W = BM // A_NW  # rows a DMA wave moves into LDS each K step
    kDmaW = A_ROWS_W // DMA_ROWS
    wave_a = (
        wave % fx.Int32(A_NW) if A_NW < NW else wave
    )  # in-range row index for every wave

    n_block_idx = bx_i32 % fx.Int32(NUM_N_BLOCKS)
    m_block_idx = bx_i32 // fx.Int32(NUM_N_BLOCKS)
    e = rocdl.readfirstlane(T.i32, _raw(_global_i32_at(arg_eids, m_block_idx)))
    m_row = m_block_idx * fx.Int32(BM)
    # h1 is written at the physical tile the m-block reads from.
    m_block_out = m_block_idx
    m_row_out = m_block_out * fx.Int32(BM)

    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lane_div_8 = lane // fx.Int32(8)
    lane_mod_8 = lane % fx.Int32(8)

    aq_num_records = fx.Int64(i32_ntok) * fx.Int64(A_ROW_BYTES)

    # fx.copy's BufferCopy/BufferCopyLDS atoms take soffset as an element count,
    # not the bytes buffer_ops.buffer_load's soffset_bytes expects.
    def _global_i32_buffer_view(addr_i64, num_bytes):
        # make_layout's dynamic-shape leaf must be i32/i64, not fx.Index.
        num_bytes_i64 = fx.Int64(num_bytes)
        ptr_ty = fx.PointerType.get(
            T.i32, address_space=fx.AddressSpace.Global, alignment=4
        )
        ptr = fx.inttoptr(ptr_ty, fx.Int64(addr_i64))
        view = fx.Tensor(
            fx.make_view(ptr, fx.make_layout(num_bytes_i64 // fx.Int64(4), 1))
        )
        return fx.rocdl.make_buffer_tensor(
            view, max_size=False, num_records_bytes=num_bytes_i64
        )

    def _global_i32_buffer_tiles(addr_i64, num_bytes, tile_elems):
        return fx.logical_divide(
            _global_i32_buffer_view(addr_i64, num_bytes), fx.make_layout(tile_elems, 1)
        )

    bq_tiles = _global_i32_buffer_tiles(arg_bq, BQ_BYTES, 4)
    bq_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(b_aux), fx.Int32)
    bq_reg_lay = fx.make_layout(4, 1)

    bscale_tiles = _global_i32_buffer_tiles(arg_bscale, BSCALE_BYTES, 1)
    bscale_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
    bscale_reg_lay = fx.make_layout(1, 1)

    # LDS alias scopes: A's LDS DMA contributes to vmcnt. Without alias.scope,
    # LLVM conservatively waits for all outstanding LDS DMA before LDS accesses.
    # This inserts vmcnt(0) before each A ds_read (41% of GMM1 cycles in ATT),
    # defeating the two-step A prefetch. Give each A slot and the A-scale region
    # a separate scope on both reads and writes, so LLVM waits only for that slot's DMA.
    from flydsl._mlir import ir as _ir

    _dom = '#llvm.alias_scope_domain<id = distinct[0]<>, description = "gmm1_lds">'
    _sc = _ir.ArrayAttr(
        _ir.Attribute.parse(
            "["
            + ", ".join(
                f"#llvm.alias_scope<id = distinct[{i + 1}]<>, domain = {_dom}>"
                for i in range(kAStages + 1)
            )
            + "]"
        )
    )
    _scopes = [_sc[i] for i in range(kAStages + 1)]  # [A slots 0..kAStages-1, A-scale]

    def _scope_attrs(idx):
        return {
            "alias_scopes": _ir.ArrayAttr.get([_scopes[idx]]),
            "noalias_scopes": _ir.ArrayAttr.get(
                [_scopes[j] for j in range(kAStages + 1) if j != idx]
            ),
        }

    _lds_base_i32 = fx.Int32(fx.ptrtoint(lds_raw_ptr))
    _asc_base_i32 = _lds_base_i32 + fx.Int32(kAStages * BM * KH_TILE)
    aq_rsrc = _buffer_rsrc(arg_aq, aq_num_records)

    # Union LDS region [s_aq | s_asc], reused as lds_acc (f32 accumulator) in
    # the epilogue. s_aq/lds_acc start at lds_raw_ptr; s_asc follows at
    # +kAStages*BM*KH_TILE.

    cached_actual_row = []
    for sub in range_constexpr(kDmaW):
        a_lane_row = lane_div_16 if is_f8_a else lane_div_8
        idx = (
            m_row + wave_a * fx.Int32(A_ROWS_W) + fx.Int32(sub * DMA_ROWS) + a_lane_row
        )
        cached_actual_row.append(_global_i32_at(arg_mind, idx))

    # -- b_load_s_base[j] (HIP 412-416), readfirstlane'd uniform per wave ------
    N0_HALF = N_OUT // 32
    b_load_s_base = []
    for j in range_constexpr(NJW):
        tile_il = n_block_idx * fx.Int32(BN // 16) + wave * fx.Int32(NJW) + fx.Int32(j)
        g = tile_il & fx.Int32(1)
        n0 = tile_il >> fx.Int32(1)
        col = (g * fx.Int32(N0_HALF) + n0) * fx.Int32(16)
        v = (e * fx.Int32(N_OUT) + col) * fx.Int32(K_HALF)
        b_load_s_base.append(rocdl.readfirstlane(T.i32, v))

    # -- b_scale_s_base / _hi (HIP 418-429) -----------------------------------
    # One B-scale word covers two adjacent 16-column n0 blocks, exactly this wave's pair.
    np_gate = n_block_idx * fx.Int32(BN // 64) + (wave // fx.Int32(2) if W8 else wave)
    bs_shift = (wave % fx.Int32(2)) * fx.Int32(8) if W8 else None
    np_list = [np_gate, np_gate + fx.Int32(N_OUT // 64)]
    b_scale_s_base, b_scale_s_base_hi = [], []
    for mw in range_constexpr(2):
        base = (
            e * fx.Int32(kBS_per_expert_dw) + np_list[mw] * fx.Int32(kBS_stride_n0_dw)
        ) * fx.Int32(4)
        base = rocdl.readfirstlane(T.i32, base)
        b_scale_s_base.append(base)
        b_scale_s_base_hi.append(base + fx.Int32(16 * kBS_stride_k0_dw * 4))

    accm = [[None] * NJW for _ in range(kMChunks)]
    b = [[[None, None] for _ in range(NJW)] for _ in range(kStages)]
    b_scale_v = [[None, None] for _ in range(kStages)]

    def issue_a_load_lds(slot, kt):
        if const_expr(A_NW < NW):
            if wave < fx.Int32(A_NW):
                _issue_a_load_lds(slot, kt)
        else:
            _issue_a_load_lds(slot, kt)

    def _issue_a_load_lds(slot, kt):
        for sub in range_constexpr(kDmaW):
            lds_row = wave * fx.Int32(A_ROWS_W) + fx.Int32(sub * DMA_ROWS)
            if const_expr(is_f8_a):
                mask = lds_swizzle_mask_f8(lds_row + lane_div_16, KH_TILE)
                lane_col = lane_mod_16 * fx.Int32(16)
            else:
                mask = _lds_swizzle_mask(lds_row + lane_div_8)
                lane_col = lane_mod_8 * fx.Int32(16)
            off = fx.Int32(slot * (BM * KH_TILE)) + lds_row * fx.Int32(KH_TILE)
            if const_expr(aqin_row_base):
                actual_row = cached_actual_row[sub]
                valid_row = (actual_row >= fx.Int32(0)) & (actual_row < i32_ntok)
                safe_row = valid_row.select(actual_row, fx.Int32(0))
                address = (
                    fx.Int64(arg_aq)
                    + fx.Int64(safe_row) * fx.Int64(A_ROW_BYTES)
                    + fx.Int64(lane_col ^ mask)
                    + fx.Int64(kt * KH_TILE)
                )
                pointer_type = fx.PointerType.get(
                    fx.Int32.ir_type, fx.AddressSpace.Global, 16
                )
                values = fx.Vector(
                    fx.ptr_load(
                        fx.inttoptr(pointer_type, address),
                        result_type=fx.Vector.make_type(4, fx.Int32),
                    )
                )
                values = valid_row.select(values, fx.Vector.filled(4, 0, fx.Int32))
                destination = fx.add_offset(
                    fx.recast_iter(fx.Int32, lds_raw_ptr),
                    (off + lane * fx.Int32(16)) // fx.Int32(4),
                )
                fx.ptr_store(values, destination)
            else:
                voffset = (lane_col ^ mask) + cached_actual_row[sub] * fx.Int32(
                    A_ROW_BYTES
                )
                # A wave-uniform LDS base, with 16B per lane.
                rocdl.raw_ptr_buffer_load_lds(
                    aq_rsrc,
                    _lds_ptr3(_lds_base_i32, off),
                    fx.Int32(16),
                    voffset,
                    fx.Int32(kt * KH_TILE),
                    fx.Int32(0),
                    fx.Int32(0),
                    **_scope_attrs(slot),
                )

    def issue_a_ds_read(slot, first_m=0, count_m=kMChunks):
        mask = (
            lds_swizzle_mask_f8(lane_mod_16, KH_TILE)
            if is_f8_a
            else _lds_swizzle_mask(lane_mod_16)
        )
        a = [[None, None] for _ in range(kMChunks)]
        for k in range_constexpr(2):
            col0 = lane_div_16 * fx.Int32(16) + fx.Int32(k * (128 if is_f8_a else 64))
            lds_col = col0 ^ mask
            # Constant block offsets fold into ds_read immediates. Keeping the
            # lane address separate avoids hoisting one VGPR per row and slot.
            lane_ptr = fx.add_offset(
                fx.recast_iter(fx.Uint8, lds_raw_ptr),
                lane_mod_16 * fx.Int32(KH_TILE) + lds_col,
            )
            for i in range_constexpr(first_m, first_m + count_m):
                block_off = (slot * BM + i * 16) * KH_TILE
                lo = Vec(
                    llvm.LoadOp(
                        Vec.make_type(4, fx.Int32),
                        fx.to_llvm_ptr(fx.add_offset(lane_ptr, block_off)),
                        alignment=16,
                        **_scope_attrs(slot),
                    ).result
                )
                if const_expr(is_f8_a):
                    hi_ptr = fx.add_offset(
                        fx.recast_iter(fx.Uint8, lds_raw_ptr),
                        lane_mod_16 * fx.Int32(KH_TILE)
                        + ((col0 + fx.Int32(64)) ^ mask),
                    )
                    hi = Vec(
                        llvm.LoadOp(
                            Vec.make_type(4, fx.Int32),
                            fx.to_llvm_ptr(fx.add_offset(hi_ptr, block_off)),
                            alignment=16,
                            **_scope_attrs(slot),
                        ).result
                    )
                    a[i][k] = Vec.from_elements(
                        [lo[j] for j in range_constexpr(4)]
                        + [hi[j] for j in range_constexpr(4)],
                        fx.Int32,
                    )
                else:
                    a[i][k] = lo
        return a

    def issue_a_scale_ds_read(kt):
        out = []
        for sub in range_constexpr(kSubBlocks):
            if const_expr(W8):
                # Keep constant terms (sub, kt) outside *4 so they fold into ds_read immediates.
                # Otherwise each (kt, sub) address needs a VGPR and spills at the 8-wave limit of 256.
                # Each scratch reload also inserts vmcnt(0), draining the B prefetch pipeline.
                lds_b = (lane_div_16 * fx.Int32(16) + lane_mod_16) * fx.Int32(
                    4
                ) + fx.Int32((sub * kAS_per_chunk_dw + kt * 64) * 4)
            else:
                lds_b = (
                    fx.Int32(sub) * fx.Int32(kAS_per_chunk_dw)
                    + fx.Int32(kt * 64)
                    + lane_div_16 * fx.Int32(16)
                    + lane_mod_16
                ) * fx.Int32(4)
            out.append(
                llvm.LoadOp(
                    T.i32,
                    _lds_ptr3(_asc_base_i32, lds_b),
                    alignment=4,
                    **_scope_attrs(kAStages),
                ).result
            )
        return out

    # ascale_gather: A-scale is stored in source-token row-major order, K/32 bytes
    # per row, written contiguously once per (token, dest) by Stage1. Use arg_mind
    # to fetch 8 bytes from each of two rows (n_lane, n_lane+16) and form the DMA
    # path's BM32 LDS layout:
    #   dword(ku, k_lane, n_lane) = [A[8ku+k], B[8ku+k], A[8ku+4+k], B[8ku+4+k]]
    # Byte order is ikxdl*2 + im_a, matching Stage1 fanout's scatter formula.
    # Issue the global load at the start of the prologue and write LDS afterward.
    # The final _wait_lds_barrier(lgkmcnt(0)+barrier) orders writes before reads.
    _GS_ITEMS = kSubBlocks * 16 * K_TILES_TOTAL
    _GS_THREADS = NW * 64
    _GS_PASS = (_GS_ITEMS + _GS_THREADS - 1) // _GS_THREADS
    _GS_ROW_DW = K // 32 // 4

    def gather_a_scale_issue():
        tid = wave * fx.Int32(64) + lane
        loads = []
        for p in range_constexpr(_GS_PASS):
            w = tid + fx.Int32(p * _GS_THREADS)
            w = (w < fx.Int32(_GS_ITEMS)).select(w, fx.Int32(0))
            sub = w // fx.Int32(16 * K_TILES_TOTAL)
            rem = w % fx.Int32(16 * K_TILES_TOTAL)
            n_lane = rem % fx.Int32(16)
            ku = rem // fx.Int32(16)
            row0 = m_row + sub * fx.Int32(32) + n_lane
            ra = _global_i32_at(arg_mind, row0)
            rb = _global_i32_at(arg_mind, row0 + fx.Int32(16))
            da = ra * fx.Int32(_GS_ROW_DW) + ku * fx.Int32(2)
            db = rb * fx.Int32(_GS_ROW_DW) + ku * fx.Int32(2)
            if const_expr(W8):
                # 8-wave: carry one LDS byte base through the prologue; k*16 dwords is immediate.
                # This saves two VGPRs per iteration.
                sub = (
                    sub * fx.Int32(kAS_per_chunk_dw) + ku * fx.Int32(64) + n_lane
                ) * fx.Int32(4)
            loads.append(
                (
                    sub,
                    n_lane,
                    ku,
                    _global_i32_at(arg_ascale, da),
                    _global_i32_at(arg_ascale, da + fx.Int32(1)),
                    _global_i32_at(arg_ascale, db),
                    _global_i32_at(arg_ascale, db + fx.Int32(1)),
                )
            )
        return loads

    def gather_a_scale_write(loads):
        m8 = fx.Int32(0xFF)
        # Out-of-range work items clamp to item 0 and write the same value without a branch.
        for sub, n_lane, ku, a_lo, a_hi, b_lo, b_hi in loads:
            for k in range_constexpr(4):
                sh = fx.Int32(8 * k)
                v = (
                    ((a_lo >> sh) & m8)
                    | (((b_lo >> sh) & m8) << fx.Int32(8))
                    | (((a_hi >> sh) & m8) << fx.Int32(16))
                    | (((b_hi >> sh) & m8) << fx.Int32(24))
                )
                if const_expr(W8):
                    lds_b = sub + fx.Int32(k * 64)
                else:
                    lds_b = (
                        sub * fx.Int32(kAS_per_chunk_dw)
                        + ku * fx.Int32(64)
                        + fx.Int32(k * 16)
                        + n_lane
                    ) * fx.Int32(4)
                llvm.StoreOp(
                    _raw(v),
                    _lds_ptr3(_asc_base_i32, lds_b),
                    alignment=4,
                    **_scope_attrs(kAStages),
                )

    def issue_b_load_j(b_slot, K_C, j):
        v = (
            (lane_div_16 * fx.Int32(256))
            + (lane_mod_16 * fx.Int32(16))
            + fx.Int32(K_C * 2048)
        )
        for half in range_constexpr(2):
            tile_idx = (v + fx.Int32(half * 1024)) // fx.Int32(16)
            r = fx.make_rmem_tensor(bq_reg_lay, fx.Int32)
            fx.copy(
                bq_copy_atom,
                fx.slice(bq_tiles, (None, tile_idx)),
                r,
                soffset=b_load_s_base[j] // fx.Int32(4),
            )
            b_slot[j][half] = r.load()

    def issue_b_scale_load(bs_slot, K_C):
        v = ((lane_div_16 * fx.Int32(16)) + lane_mod_16) * fx.Int32(4)
        K_C_HI = K_C // 16
        imm = (K_C - K_C_HI * 16) * (kBS_stride_k0_dw * 4)
        for mw in range_constexpr(2):
            s_off = b_scale_s_base[mw] if K_C_HI == 0 else b_scale_s_base_hi[mw]
            idx = (v + fx.Int32(imm)) // fx.Int32(4)
            r = fx.make_rmem_tensor(bscale_reg_lay, fx.Int32)
            fx.copy(
                bscale_copy_atom,
                fx.slice(bscale_tiles, (None, idx)),
                r,
                soffset=s_off // fx.Int32(4),
            )
            bs_slot[mw] = r.load()[0]
            if const_expr(W8):
                # For the odd n0 in the pair, move bytes 1/3 to 0/2; in_b is always 0.
                bs_slot[mw] = fx.Int32(bs_slot[mw]) >> bs_shift

    mfma_ty = T.f32x4
    zero4 = fx.Vector.filled(4, 0.0, fx.Float32)

    def mfma_cluster(
        b_slot, a, a_scale, bs_slot, J, init, first_sub=0, count_sub=kSubBlocks
    ):
        mni = J % 2
        in_b = J // 2
        sb = bs_slot[mni]
        bJ0, bJ1 = b_slot[J][0], b_slot[J][1]
        for sub in range_constexpr(first_sub, first_sub + count_sub):
            i0 = sub * 2 + 0
            i1 = sub * 2 + 1
            sa = a_scale[sub]
            if const_expr(init):
                accm[i0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    mfma_ty,
                    [a[i0][0], bJ0, zero4, 0 if is_f8_a else 4, 4, 0, sa, 0 + in_b, sb],
                )
                accm[i1][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    mfma_ty,
                    [a[i1][0], bJ0, zero4, 0 if is_f8_a else 4, 4, 1, sa, 0 + in_b, sb],
                )
            else:
                accm[i0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    mfma_ty,
                    [
                        a[i0][0],
                        bJ0,
                        accm[i0][J],
                        0 if is_f8_a else 4,
                        4,
                        0,
                        sa,
                        0 + in_b,
                        sb,
                    ],
                )
                accm[i1][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    mfma_ty,
                    [
                        a[i1][0],
                        bJ0,
                        accm[i1][J],
                        0 if is_f8_a else 4,
                        4,
                        1,
                        sa,
                        0 + in_b,
                        sb,
                    ],
                )
            accm[i0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                mfma_ty,
                [
                    a[i0][1],
                    bJ1,
                    accm[i0][J],
                    0 if is_f8_a else 4,
                    4,
                    2,
                    sa,
                    2 + in_b,
                    sb,
                ],
            )
            accm[i1][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                mfma_ty,
                [
                    a[i1][1],
                    bJ1,
                    accm[i1][J],
                    0 if is_f8_a else 4,
                    4,
                    3,
                    sa,
                    2 + in_b,
                    sb,
                ],
            )

    def mfma_row(b_slot, a, a_scale, bs_slot, J, i, init):
        in_b = J // 2
        sb = bs_slot[J % 2]
        sa = a_scale[i // 2]
        previous = zero4 if init else accm[i][J]
        accm[i][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
            mfma_ty, [a[i][0], b_slot[J][0], previous, 0, 4, i % 2, sa, in_b, sb]
        )
        accm[i][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
            mfma_ty,
            [a[i][1], b_slot[J][1], accm[i][J], 0, 4, 2 + i % 2, sa, 2 + in_b, sb],
        )

    # B vmem instructions per K step: 4 j x 2 halves of dwordx4, plus 2 B-scale loads.
    _B_VMEM_PER_STEP = NJW * 2 + 2

    # Use bare s_barrier only on the scoped path. gpu.barrier() lowers to an LDS
    # workgroup release fence, s_barrier, and acquire fence. SIMemoryLegalizer emits
    # S_WAITCNT_lds_direct for the release fence; SIInsertWaitcnts turns this into
    # a wait for all outstanding LDS DMA, ignoring alias scopes. Merging it with
    # the explicit s_waitcnt tightens vmcnt(24) to vmcnt(10), forcing A(kt+2) to
    # finish one step early (28 ISA sites become vmcnt(10) lgkmcnt(0)). Explicit
    # vmcnt(N)+lgkmcnt(0) already provides cross-wave visibility; no extra fence is needed.

    def _wait_lds_barrier(vmcnt):
        # As in mega_moe/gemm_util.wait_lds_barrier: lgkmcnt(0) completes this wave's
        # ds_read, and vmcnt(N) completes the A DMA issued at this step's start.
        # The N later B prefetch instructions remain in flight; then synchronize waves.
        fx.rocdl.s_waitcnt(vmcnt=vmcnt, lgkmcnt=0)
        rocdl.s_barrier()

    # Follow MegaMoEv2 do_tile: wait_lds_barrier at each step's end, three A stages,
    # and DMA two steps ahead. At step kt, load A(kt+2) into slot (kt+2)%3, which
    # all waves finished reading at the previous step's lgkmcnt(0)+barrier.
    # At step kt's end, wait only for A(kt+1); the later B(kt+1), A(kt+2), and
    # B(kt+2) remain in flight. Each wave loads one quarter of the rows via LDS DMA
    # but reads all rows, so both completion and cross-wave synchronization are required.
    _A_DMA_PER_STEP = kDmaW

    def _vmcnt_after_a(kt_next):
        # vmem issued after A(kt_next): B(kt_next), plus A(kt_next+1) and B(kt_next+1) if present.
        n = _B_VMEM_PER_STEP
        if kt_next + 1 < K_TILES_TOTAL:
            n += _A_DMA_PER_STEP + _B_VMEM_PER_STEP
        return n

    gs_loads = gather_a_scale_issue()
    for K_C in range_constexpr(kStages):
        if const_expr(K_C < K_TILES_TOTAL):
            rocdl.sched_barrier(0)
            issue_a_load_lds(K_C % 3, K_C)
            rocdl.sched_barrier(0)
            for j in range_constexpr(NJW):
                issue_b_load_j(b[K_C], K_C, j)
            issue_b_scale_load(b_scale_v[K_C], K_C)
            rocdl.sched_barrier(0)
    gather_a_scale_write(gs_loads)
    _wait_lds_barrier(_vmcnt_after_a(0))
    for kt in range_constexpr(K_TILES_TOTAL):
        has_next = kt + 1 < K_TILES_TOTAL
        refill = kt + kStages < K_TILES_TOTAL
        slot_b = kt % kStages
        if const_expr(refill):
            rocdl.sched_barrier(0)
            issue_a_load_lds((kt + kStages) % 3, kt + kStages)
            rocdl.sched_barrier(0)
        if const_expr(is_f8_a and BM == 128):
            asc_cur = issue_a_scale_ds_read(kt)
            # Retain every B operand until both M halves consume it. No A
            # fragment survives across halves; B prefetch happens only once.
            for first_m in range_constexpr(kMChunks):
                rocdl.sched_barrier(0)
                a_cur = issue_a_ds_read(kt % 3, first_m, 1)
                for J in range_constexpr(NJW):
                    mfma_row(
                        b[slot_b],
                        a_cur,
                        asc_cur,
                        b_scale_v[slot_b],
                        J,
                        first_m,
                        init=(kt == 0),
                    )
                    rocdl.sched_barrier(0)
                    if const_expr(refill and first_m + 1 == kMChunks):
                        issue_b_load_j(b[slot_b], kt + kStages, J)
                    rocdl.sched_barrier(0)
        else:
            if const_expr(BM == 128):
                asc_cur = issue_a_scale_ds_read(kt)
                a_cur = issue_a_ds_read(kt % 3)
            else:
                a_cur = issue_a_ds_read(kt % 3)
                asc_cur = issue_a_scale_ds_read(kt)
            for J in range_constexpr(NJW):
                if const_expr(BM != 128):
                    rocdl.sched_barrier(0)
                    rocdl.s_setprio(1)
                mfma_cluster(
                    b[slot_b], a_cur, asc_cur, b_scale_v[slot_b], J, init=(kt == 0)
                )
                if const_expr(BM != 128):
                    rocdl.s_setprio(0)
                rocdl.sched_barrier(0)
                if const_expr(refill):
                    issue_b_load_j(b[slot_b], kt + kStages, J)
                rocdl.sched_barrier(0)
        if const_expr(refill):
            issue_b_scale_load(b_scale_v[slot_b], kt + kStages)
        if const_expr(has_next):
            _wait_lds_barrier(_vmcnt_after_a(kt + 1))

    gpu.barrier()

    if const_expr(is_f8_a and BM == 128):
        # Reconstruct physical wave lanes after the MFMA chain. The explicit
        # hardware count avoids keeping epilogue lane coordinates live across
        # GEMM. Counting an all-ones mask returns lane0..63 independently of
        # the current execution mask; this does not change the output mapping.
        epi_lane = fx.Int32(
            llvm.inline_asm(
                T.i32,
                [],
                "v_mbcnt_lo_u32_b32 $0, -1, 0\n\tv_mbcnt_hi_u32_b32 $0, -1, $0",
                "=v",
                has_side_effects=True,
            )
        )
        lane_div_16 = epi_lane // fx.Int32(16)
        lane_mod_16 = epi_lane % fx.Int32(16)

    # lds_acc reuses the s_aq region (offset 0) as an f32 accumulator.
    # A BN+4 row stride without column XOR removes epilogue LDS bank conflicts.
    # _bm_constants already sizes LDS for BN+4.
    _epi_swz_selfcheck(BM, BN)
    ACC_STRIDE = BN + _EPI_ACC_PAD
    acc_layout = fx.make_layout((BM, BN), (ACC_STRIDE, 1))

    def acc_idx(row, col):
        return _layout_idx(acc_layout, row, col)

    acc_copy_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    acc_reg_lay = fx.make_layout(1, 1)
    acc_flat_view = fx.make_view(
        fx.recast_iter(fx.Float32, lds_raw_ptr), fx.make_layout(BM * ACC_STRIDE, 1)
    )
    acc_flat_tiles = fx.logical_divide(acc_flat_view, fx.make_layout(1, 1))

    def acc_store(idx, value):
        r = fx.make_rmem_tensor(acc_reg_lay, fx.Float32)
        r.store(fx.Vector.from_elements([fx.Float32(value)], fx.Float32))
        fx.copy(acc_copy_atom, r, fx.slice(acc_flat_tiles, (None, idx)))

    def acc_load(idx):
        r = fx.make_rmem_tensor(acc_reg_lay, fx.Float32)
        fx.copy(acc_copy_atom, fx.slice(acc_flat_tiles, (None, idx)), r)
        return r.load()[0]

    for i in range_constexpr(kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for J in range_constexpr(NJW):
            is_up = (J % 2) == 1
            J_local = J // 2
            col_local = (
                wave * fx.Int32(NJW // 2 * 16) + fx.Int32(J_local * 16) + lane_mod_16
            )
            lds_col = (fx.Int32(BN // 2) + col_local) if is_up else col_local
            vec = fx.Vector(accm[i][J])
            for v in range_constexpr(4):
                idx = acc_idx(row_base + fx.Int32(v), lds_col)
                acc_store(idx, vec[v])

    gpu.barrier()

    if const_expr(is_f8_a and BM == 128):
        quant_lane = fx.Int32(
            llvm.inline_asm(
                T.i32,
                [],
                "v_mbcnt_lo_u32_b32 $0, -1, 0\n\tv_mbcnt_hi_u32_b32 $0, -1, $0",
                "=v",
                has_side_effects=True,
            )
        )
        tx_i32 = wave * fx.Int32(64) + quant_lane
    else:
        tx_i32 = fx.Int32(gpu.thread_id("x"))
    # 16 threads per row; 4-wave: tx // 16 covers 16 rows per pass.  8-wave: the
    # second 256 threads (half = 1) take the next 32-row block in the same pass.
    m_lane = (tx_i32 % fx.Int32(256)) // fx.Int32(16) if W8 else tx_i32 // fx.Int32(16)
    n_lane = tx_i32 % fx.Int32(16)
    wave_grp = n_lane // fx.Int32(4)
    kk = n_lane % fx.Int32(4)

    aqout_layout = fx.make_layout((BM, K_G2_HALF), (K_G2_HALF, 1))
    # UniversalCopy has no nontemporal/cache-hint knob; dropped (perf-neutral).
    if const_expr(aqout_row_base):
        aqout_tiles = _global_scalar_tiles(
            fx.Int64(arg_aqout) + fx.Int64(m_row_out) * fx.Int64(K_G2_HALF),
            fx.Int32,
            1 << 24,
        )
        aqout_row0 = fx.Int32(0)
        aqout_base = fx.Int64(arg_aqout) + fx.Int64(m_row_out) * fx.Int64(K_G2_HALF)
    else:
        aqout_tiles = _global_scalar_tiles(arg_aqout, fx.Int32, 1 << 24)
        aqout_row0 = m_row_out
        aqout_base = fx.Int64(arg_aqout)
    if const_expr(h1_wt):
        _H1_CM = 17  # SC0 | SC1 (no NT)
        aqout_rsrc = buffer_ops.create_buffer_resource_from_addr(aqout_base)
        ascaleout_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(arg_ascaleout)
        )

    def _epi_row(row_local):
        """SwiGLU + FP4 quant of this thread's 8 columns of one h1 row; returns its e8m0."""
        gate_vs = [None] * 8
        up_vs = [None] * 8
        for ee in range_constexpr(8):
            col_in_grp = fx.Int32(8) * kk + fx.Int32(ee)
            gate_col = wave_grp * fx.Int32(32) + col_in_grp
            up_col = fx.Int32(128) + gate_col
            gate_vs[ee] = acc_load(acc_idx(row_local, gate_col))
            up_vs[ee] = acc_load(acc_idx(row_local, up_col))
        result = _gate_up_batch(
            gate_vs,
            up_vs,
            act=act,
            swiglu_limit=swiglu_limit,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
        )
        if const_expr(is_f8_a):
            # The established A8W4 path materializes activation in BF16
            # before MXFP8 requantization; retain that rounding boundary.
            result = [value.to(fx.BFloat16).to(fx.Float32) for value in result]

        local_max = _fabs_f32(result[0])
        for ee in range_constexpr(1, 8):
            local_max = local_max.maximumf(_fabs_f32(result[ee]))
        lm_i = _inline_dpp_quad_amax(fx.Int32(_raw(local_max).bitcast(T.i32)))
        local_max = fx.Float32(_raw(lm_i).bitcast(T.f32))

        e8m0, qscale = _e8m0_from_amax(local_max, max_norm=448.0 if is_f8_a else 6.0)
        qscale_raw = _raw(qscale)
        words = []
        if const_expr(is_f8_a):
            for word in range_constexpr(2):
                packed = fx.full(2, 0, fx.Int16)
                for pair in range_constexpr(2):
                    idx = word * 4 + pair * 2
                    packed = rocdl.cvt_scalef32_pk_fp8_f32(
                        Vec.make_type(2, fx.Int16),
                        packed,
                        _raw(result[idx]),
                        _raw(result[idx + 1]),
                        qscale_raw,
                        pair,
                    )
                words.append(Vec(packed).bitcast(fx.Int32)[0])
            byte_pos = (
                n_block_idx * fx.Int32(BN_INT)
                + wave_grp * fx.Int32(32)
                + kk * fx.Int32(8)
            )
        else:
            packed_i32 = _raw(fx.Int32(0))
            for w in range_constexpr(4):
                packed_i32 = rocdl.cvt_scalef32_pk_fp4_f32(
                    T.i32,
                    packed_i32,
                    _raw(result[2 * w]),
                    _raw(result[2 * w + 1]),
                    qscale_raw,
                    w,
                )
            words.append(fx.Int32(packed_i32))
            byte_pos = (
                n_block_idx * fx.Int32(BN_INT // 2)
                + wave_grp * fx.Int32(16)
                + kk * fx.Int32(4)
            )
        out_row = aqout_row0 + row_local
        store_off = _layout_idx(aqout_layout, out_row, byte_pos)
        for word in range_constexpr(2 if is_f8_a else 1):
            if const_expr(h1_wt):
                buffer_ops.buffer_store(
                    words[word],
                    aqout_rsrc,
                    store_off + fx.Int32(word * 4),
                    cache_modifier=_H1_CM,
                    offset_is_bytes=True,
                )
            else:
                _scalar_store(
                    aqout_tiles,
                    store_off // fx.Int32(4) + fx.Int32(word),
                    words[word],
                    fx.Int32,
                )
        return e8m0

    # (chunk, ku, wave_grp, m_lane) -> dword index; shape is a placeholder.
    ascaleout_layout = fx.make_layout(
        (1 << 20, 2, 4, 16), (OUT_AS_PER_CHUNK_DW, 64, 16, 1)
    )
    ascaleout_i16_tiles = _global_scalar_tiles(arg_ascaleout, fx.Int16, 1 << 25)
    ku_o = n_block_idx >> fx.Int32(1)
    ikxdl = n_block_idx & fx.Int32(1)

    def _store_scale_pair(sub, s_lo, s_hi):
        # rows sub*32 + m_lane and sub*32 + 16 + m_lane share one i16.
        chunk = m_block_out * fx.Int32(kSubBlocks) + sub
        dword_off = _layout_idx(ascaleout_layout, chunk, ku_o, wave_grp, m_lane)
        pair_i32 = s_lo | (s_hi << fx.Int32(8))
        addr = dword_off * fx.Int32(4) + ikxdl * fx.Int32(2)
        if const_expr(h1_wt):
            buffer_ops.buffer_store(
                pair_i32.to(fx.Int16),
                ascaleout_rsrc,
                addr,
                cache_modifier=_H1_CM,
                offset_is_bytes=True,
            )
        else:
            _scalar_store(ascaleout_i16_tiles, addr // fx.Int32(2), pair_i32, fx.Int16)

    if const_expr(not W8):
        scales_per_mr = [None] * M_REPS
        for mr in range_constexpr(M_REPS):
            scales_per_mr[mr] = _epi_row(fx.Int32(mr * 16) + m_lane)
        if kk == fx.Int32(0):
            for sub in range_constexpr(kSubBlocks):
                _store_scale_pair(
                    fx.Int32(sub),
                    scales_per_mr[sub * 2 + 0],
                    scales_per_mr[sub * 2 + 1],
                )
    elif const_expr(kSubBlocks == 1):
        # BM 32 x 8 waves: one 32-row block; the first 256 threads take it.
        if tx_i32 < fx.Int32(256):
            s_lo = _epi_row(m_lane)
            s_hi = _epi_row(fx.Int32(16) + m_lane)
            if kk == fx.Int32(0):
                _store_scale_pair(fx.Int32(0), s_lo, s_hi)
    else:
        epi_half = tx_i32 // fx.Int32(256)
        for ps in range_constexpr(kSubBlocks // 2):
            sub = fx.Int32(ps * 2) + epi_half
            s_lo = _epi_row(sub * fx.Int32(32) + m_lane)
            s_hi = _epi_row(sub * fx.Int32(32) + fx.Int32(16) + m_lane)
            if kk == fx.Int32(0):
                _store_scale_pair(sub, s_lo, s_hi)


# Preserve the private import spelling used by the fused wrappers while making
# the JIT function name itself layout-versioned.  FlyDSL keys its persistent
# cache by JIT function, not only by the emitted GPU symbol.
_gemm1_body = _gemm1_body_m16


# Bank-conflict-free LDS layout for the f32 epilogue accumulator (BN=256 only).
# The old (BM,BN):(BN,1) layout writes one f32 per lane across 16 columns x 4 rows.
# Four rows hit the same bank, causing four-way conflicts with 64 banks.
# ds_read_b128 has two-way conflicts in CDNA4's four documented phases:
# 16 lanes per phase (T0-3,T12-15,T20-23,T24-27, etc.), two m rows and two wave_grp.
# The new layout uses a BN+4 dword row stride without column XOR. Adjacent
# rows shift by four banks (one 16B block). Write row 4k shifts by 16k banks,
# so 64 lanes cover all 64 banks. Each read phase's two rows occupy even/odd
# 16B blocks, covering 64 banks; CDNA3's eight phases on 32 banks are also conflict-free.
# XOR by wave_grp would reintroduce two-way read conflicts under these phase
# groups, as verified by the self-check model. Logical values remain bitwise
# identical, and addresses remain lane base + compile-time constant. Extra
# LDS is BM*4*4 bytes (2KB for BM=128).
_EPI_ACC_PAD = 4

# ds_read_b128 lane phases from nod-ai amdgpu_kernel_optimization_guide.md measurements.
_B128_PHASES_CDNA4 = (
    (0, 1, 2, 3, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 26, 27),
    (32, 33, 34, 35, 44, 45, 46, 47, 52, 53, 54, 55, 56, 57, 58, 59),
    (4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 28, 29, 30, 31),
    (36, 37, 38, 39, 40, 41, 42, 43, 48, 49, 50, 51, 60, 61, 62, 63),
)
_B128_PHASES_CDNA3 = tuple(
    tuple(b + a + i for i in range(4)) + tuple(b + c + i for i in range(4))
    for b in (0, 32)
    for a, c in ((0, 20), (4, 16), (8, 28), (12, 24))
)


_EPI_SWZ_CHECKED = set()


def _epi_swz_selfcheck(BM, BN):
    """Check the _gemm1_body epilogue addresses using compile-time Python integers.
    With a BN+4 stride, b128 reads are 16B-aligned. Writes (b32, one phase of 64
    lanes or two phases of 32 lanes) and reads (four CDNA4 phases or eight CDNA3
    phases) have no bank conflicts.
    """
    key = (BM, BN)
    if key in _EPI_SWZ_CHECKED:
        return
    stride = BN + _EPI_ACC_PAD
    assert BN == 256 and BM % 16 == 0, (BM, BN)
    assert (stride * 4) % 16 == 0

    def _ways(addrs, width, banks, groups):
        worst = 1
        for g in groups:
            cnt = {}
            for l in g:
                for d in range(width):
                    a = addrs[l] + d
                    cnt.setdefault(a % banks, set()).add(a)
            worst = max(worst, max(len(v) for v in cnt.values()))
        return worst

    for wave in range(4):
        for J in range(BN // 64):
            up = BN // 2 if J % 2 == 1 else 0
            for v in range(4):
                addrs = [
                    ((lane // 16) * 4 + v) * stride
                    + wave * (BN // 8)
                    + (J // 2) * 16
                    + lane % 16
                    + up
                    for lane in range(64)
                ]
                assert _ways(addrs, 1, 64, (range(64),)) == 1, "epi store conflict 64"
                assert (
                    _ways(addrs, 1, 32, (range(32), range(32, 64))) == 1
                ), "epi store conflict 32"
        for up in (0, 128):
            for h in (0, 1):
                addrs = []
                for lane in range(64):
                    tx = wave * 64 + lane
                    m, n = tx // 16, tx % 16
                    wg, kk = n // 4, n % 4
                    a = m * stride + up + wg * 32 + kk * 8 + 4 * h
                    assert a % 4 == 0
                    addrs.append(a)
                assert (
                    _ways(addrs, 4, 64, _B128_PHASES_CDNA4) == 1
                ), "epi load conflict cdna4"
                assert (
                    _ways(addrs, 4, 32, _B128_PHASES_CDNA3) == 1
                ), "epi load conflict cdna3"
    _EPI_SWZ_CHECKED.add(key)


def _bm_constants(BM, BN, KH_TILE, K_TILES_TOTAL):
    # The production pipeline uses three A stages with DMA two steps ahead.
    # LDS does not grow: three A stages plus scales use BM*496B, below BM*BN*4 reuse space.
    kAStages = 3
    kSubBlocks = BM // 32
    kMChunks = kmchunks_for(BM)
    s_aq_bytes = kAStages * BM * KH_TILE
    s_asc_bytes = kSubBlocks * K_TILES_TOTAL * 256
    lds_acc_bytes = lds_acc_bytes_for(BM, BN + _EPI_ACC_PAD)
    lds_bytes = max(s_aq_bytes + s_asc_bytes, lds_acc_bytes)
    return kAStages, kSubBlocks, kMChunks, lds_bytes


def scale_view(
    arg_scale, base_dw, K_TILES_TOTAL, k0_stride_dw=64, num_records_bytes=None
):
    """View one e8m0 scale word, optionally bounded to the real buffer extent."""
    base_dw = rocdl.readfirstlane(T.i32, base_dw)
    i32_ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=4
    )
    off_i64 = fx.Int64(base_dw)
    base_iter = fx.inttoptr(i32_ptr_ty, fx.Int64(arg_scale) + off_i64 * 4)
    shape = (4, 16, K_TILES_TOTAL, 1)
    stride = (16, 1, k0_stride_dw, 1)
    view = fx.Tensor(fx.make_view(base_iter, fx.make_layout(shape, stride)))
    if num_records_bytes is not None:
        return fx.rocdl.make_buffer_tensor(view, num_records_bytes=num_records_bytes)
    return fx.rocdl.make_buffer_tensor(view, max_size=False)


def scale_mma_atoms(a_dtype="fp4"):
    """Scaled-MFMA atoms for FP4 or FP8 activation and FP4 weights."""
    elem_a = Float8E4M3FN if a_dtype == "fp8" else Float4E2M1FN
    return {
        (osa, osb): fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(
                16, 16, 128, elem_a, Float4E2M1FN, opsel_a=osa, opsel_b=osb
            )
        )
        for osa in range(4)
        for osb in range(4)
    }


def mma_one_j(
    J,
    in_b,
    sa,
    sb,
    bq_frags_kt,
    a_frags,
    c_frags,
    atoms,
    i0=0,
    k_halves=2,
):
    """Run one scaled-MFMA cluster over a 32-row A-scale group."""
    steps = tuple(
        (2 * k + im, k, i0 + im, bq_frags_kt[J][k])
        for k in range(k_halves)
        for im in range(2)
    )
    for osa, k, i, bJ in steps:
        osb = 2 * k + in_b
        fx.gemm(
            atoms[(osa, osb)],
            c_frags[i][J],
            a_frags[i][k],
            bJ,
            c_frags[i][J],
            scale_a=sa,
            scale_b=sb,
        )


def issue_a_load_lds_dt(
    arg_aq,
    s_aq_base,
    slot,
    kt,
    m_row,
    wave,
    lane,
    KH_TILE_A,
    K_BYTES,
    BM=32,
    NW=4,
    a_dtype="fp4",
):
    """Load one A tile through a tile-local descriptor so allocations may span the 4 GiB buffer ABI.

    NW waves share the rows; with fewer than 8 rows per wave the caller lets
    only the first BM // 8 waves call this with NW = BM // 8."""
    lanes_per_row = KH_TILE_A // 16  # 8 (fp4)
    rows_per_call = 64 // lanes_per_row  # 8 (fp4)
    a_lane_row = lane // lanes_per_row
    rows_per_wave = BM // NW  # rows each wave loads (BM32 x 4 waves: 8)
    assert (
        rows_per_wave >= rows_per_call
    ), "every wave loads whole row groups (BM >= 32)"
    gather_base_row = wave * rows_per_wave
    n_row_groups = rows_per_wave // rows_per_call
    lane_col = (lane % lanes_per_row) * 16
    base_i32 = s_aq_base
    atom = lds_dma_atom_128()
    tile_base = fx.Int64(arg_aq) + fx.Int64(m_row) * fx.Int64(K_BYTES)
    src = flat_buffer_view(
        tile_base,
        None,
        T.i32,
        align=16,
        elem_bytes=4,
        fold=False,
        num_records_bytes=fx.Int64(BM) * fx.Int64(K_BYTES),
    )
    for g in range_constexpr(n_row_groups):
        lds_row = gather_base_row + g * rows_per_call
        mask = (
            lds_swizzle_mask_f8(lds_row + a_lane_row, KH_TILE_A)
            if a_dtype == "fp8"
            else _lds_swizzle_mask(lds_row + a_lane_row)
        )
        tile_row = lds_row + a_lane_row
        voffset = (lane_col ^ mask) + tile_row * K_BYTES
        off = fx.Int32(slot * (BM * KH_TILE_A)) + lds_row * KH_TILE_A
        v_e = (voffset + kt * KH_TILE_A) // 4  # per-lane i32-elem index
        fx.copy(
            atom, src[v_e, None], lds_dma_dst(base_i32, off, elem_ty=T.i32, align=16)
        )


@flyc.jit
def gemm2_compute_v2(
    lds_base_i32,
    arg_ascale,
    arg_bq,
    arg_bscale,
    arg_eids,
    arg_aq,
    i32_max_m_blocks,
    bx_i32,
    lane,
    wave,
    i32_inter,
    i32_hidden,
    *,
    BM,
    BN=256,
    BK=256,
    INTER_MAX,
    aStages,
    explicit_a_row=None,
    eids_stride=1,
    NW=4,
    a_dtype="fp4",
):
    """GEMM2 of one (m_block, n_block): fp4 A (LDS DMA, kStages+1 slots), fp4 B
    streamed through a 2-stage register pipeline with the A-scale carried alongside.

    explicit_a_row: A/A-scale rows live at this (physical) row instead of the
    sorted m_row; expert id and the returned m_row stay on the sorted row.
    eids_stride: entries of ``arg_eids`` per m_block (``arg_eids`` is per
    32-row tile; a BM = 32*G block reads its first tile's expert).
    NW=8: an 8-wave CTA runs two n_blocks of one m_block side by side (waves
    0-3 the n_block of ``bx_i32``, waves 4-7 the next one, which the caller
    folds into ``bx_i32`` per wave); the A tile in LDS is shared, so all 8
    waves split its DMA (only waves 0-3 at BM 32, 8 rows per wave minimum).
    """
    assert aStages == kStages + 1
    assert BK == 256, "one 256-K E8M0 scale word per K-tile"
    assert BM >= 32
    assert NW in (4, 8)
    wave_c = wave % fx.Int32(4) if NW == 8 else wave  # column slice within the n_block
    is_f8_a = a_dtype == "fp8"
    stream_a = is_f8_a and BM == 128
    A_NW = min(NW, BM // (4 if is_f8_a else 8))
    kMChunks = BM // 16  # 16-row MFMA row-groups
    kHalves = BK // 128  # 16x16x128 MFMA K-steps per K-tile
    numAccN = (BN // 4) // 16  # 16-column MFMA subblocks per wave
    nPairs = max(1, numAccN // 2)  # one B-scale per two 16-column subblocks
    kScaleSubBlocks = kMChunks // 2
    a_pack = 1 if is_f8_a else 2
    KH_TILE_A = BK // a_pack
    slot_bytes = BM * KH_TILE_A
    # Contraction K = inter_dim runtime (i32_inter); INTER_MAX caps compile-time view/fragment bounds.
    K_rt = fx.Int32(i32_inter)
    K_BYTES = K_rt // fx.Int32(a_pack)  # A row stride bytes (runtime)
    kc_rt = K_rt // 256  # (K//32)//4//2
    K_TILES_RT = K_rt // fx.Int32(BK)  # runtime K-tile trip count
    kAS_per_chunk_dw = kc_rt * 64
    kBS_stride_n0_dw = kc_rt * 64
    # N_OUT = model_dim/hidden is the gemm2 output N dim; runtime via i32_hidden (no K-loop dependency).
    N_OUT_rt = fx.Int32(i32_hidden)
    kbs_per_expert_dw = (N_OUT_rt // 32) * kBS_stride_n0_dw  # (N_OUT//16//2)*stride
    num_n_blocks = N_OUT_rt // fx.Int32(BN)
    KH4 = K_rt // 8  # i32 col stride (= K_HALF//4)
    K_SCALE_CHUNKS_MAX = INTER_MAX // 256

    # Map each compute block to its expert metadata row.
    m_block_idx = bx_i32 // num_n_blocks
    n_block_idx = bx_i32 - m_block_idx * num_n_blocks
    eids_ptr = global_typed_ptr(arg_eids, T.i32)
    e_idx = m_block_idx if eids_stride == 1 else m_block_idx * fx.Int32(eids_stride)
    e = rocdl.readfirstlane(T.i32, eids_ptr[e_idx])
    m_row = m_block_idx * BM
    a_row = m_row if explicit_a_row is None else fx.Int32(explicit_a_row)

    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16

    s_aq_base = lds_base_i32
    mma_atoms = scale_mma_atoms(a_dtype)

    # A activation: global->LDS DMA (issue_a_load_lds), then LDS->reg ds-read (issue_a_ds_read).
    A_NDW = 8 if is_f8_a else 4
    a_frags = [
        [fx.make_rmem_tensor(A_NDW, Int32) for _ in range_constexpr(kHalves)]
        for _ in range_constexpr(kMChunks)
    ]

    def issue_a_load_lds(slot, kt):
        if const_expr(A_NW < NW):
            if wave < fx.Int32(A_NW):
                issue_a_load_lds_dt(
                    arg_aq,
                    s_aq_base,
                    slot,
                    kt,
                    a_row,
                    wave,
                    lane,
                    KH_TILE_A,
                    K_BYTES,
                    BM=BM,
                    NW=A_NW,
                    a_dtype=a_dtype,
                )
        else:
            issue_a_load_lds_dt(
                arg_aq,
                s_aq_base,
                slot,
                kt,
                a_row,
                wave,
                lane,
                KH_TILE_A,
                K_BYTES,
                BM=BM,
                NW=NW,
                a_dtype=a_dtype,
            )

    def issue_a_ds_read(slot, first_m=0, count_m=kMChunks):
        # A ds-read for one slot into a_frags: fp8 -> i32<8:1> (two 128-K halves), fp4 -> i32<4:1>.
        for k in range_constexpr(kHalves):
            for i in range_constexpr(first_m, first_m + count_m):
                lds_row = lane_mod_16 + i * 16
                row_off = fx.Int32(slot * slot_bytes) + lds_row * KH_TILE_A
                mask = (
                    lds_swizzle_mask_f8(lane_mod_16, KH_TILE_A)
                    if is_f8_a
                    else _lds_swizzle_mask(lane_mod_16)
                )
                col0 = lane_div_16 * 16 + k * (128 if is_f8_a else 64)
                lds_col = col0 ^ mask
                vec = lds_vec_load(
                    s_aq_base,
                    row_off + lds_col,
                    Vec.make_type(4, fx.Int32),
                    fx.Int32,
                    align=16,
                )
                if const_expr(is_f8_a):
                    hi = Vec(
                        lds_vec_load(
                            s_aq_base,
                            row_off + ((col0 + 64) ^ mask),
                            Vec.make_type(4, fx.Int32),
                            fx.Int32,
                            align=16,
                        )
                    )
                    lo = Vec(vec)
                    a_frags[i][k].store(
                        Vec.from_elements(
                            [lo[j] for j in range_constexpr(4)]
                            + [hi[j] for j in range_constexpr(4)],
                            fx.Int32,
                        )
                    )
                else:
                    a_frags[i][k].store(Vec(vec))

    # The shared e8m0 scale layout bounds each A-scale view to its remaining bytes.
    sc_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

    asc_per_mb = fx.Int32(kScaleSubBlocks) * kAS_per_chunk_dw * 4
    asc_num = fx.Int64(i32_max_m_blocks) * fx.Int64(asc_per_mb)
    scale_chunk0 = a_row // 32

    def make_ascale_view(sub):
        base_dw = (scale_chunk0 + fx.Int32(sub)) * kAS_per_chunk_dw
        nrec = asc_num - fx.Int64(base_dw) * 4
        return scale_view(
            arg_ascale,
            base_dw,
            K_SCALE_CHUNKS_MAX,
            k0_stride_dw=64,
            num_records_bytes=nrec,
        )

    ascale_views = [make_ascale_view(sub) for sub in range_constexpr(kScaleSubBlocks)]
    sc_frag_tmpl = ascale_views[0][0, 0, 0, None]  # i32<1:1> (one e8m0 word)

    def load_a_scale_tile(kt):
        # One i32 A-scale register per 32-row chunk (kScaleSubBlocks).
        chunk_kt = kt
        out = []
        for sub in range_constexpr(kScaleSubBlocks):
            saf = fx.make_fragment_like(sc_frag_tmpl)
            fx.copy(
                sc_copy_atom,
                ascale_views[sub][lane_div_16, lane_mod_16, chunk_kt, None],
                saf,
            )
            out.append(Vec(saf.load())[0])
        return out

    # Stream B weights and scales through registers.
    bq_buffer = ptr_buf_tensor(arg_bq, fx.Int32, unit_elems=4)

    bq_base_dw = [
        rocdl.readfirstlane(
            T.i32,
            (e * N_OUT_rt + n_block_idx * BN + wave_c * (BN // 4) + j * 16) * KH4,
        )
        for j in range_constexpr(numAccN)
    ]

    mni_base = n_block_idx * (BN // 16 // 2) + wave_c * (BN // 64 // 2)
    bscale_views = [
        scale_view(
            arg_bscale,
            e * kbs_per_expert_dw + (mni_base + mw) * kBS_stride_n0_dw,
            K_SCALE_CHUNKS_MAX,
            k0_stride_dw=kBS_stride_k0_dw,
        )
        for mw in range_constexpr(nPairs)
    ]

    frag_tmpl = fx.make_rmem_tensor(4, Int32)
    # B-scale word template shares the A-scale layout (sc_frag_tmpl).

    def issue_b_load_into(bqf, bsf, kt_rt):
        # Issue B-weight + B-scale vmem loads for K-tile kt_rt into the given (per-stage) fragments.
        for j in range_constexpr(numAccN):
            for half in range_constexpr(kHalves):
                bq_off_dw = (
                    bq_base_dw[j]
                    + lane_div_16 * fx.Int32(64)
                    + lane_mod_16 * fx.Int32(4)
                    + kt_rt * fx.Int32(kHalves * 256)
                    + fx.Int32(half * 256)
                )
                loaded_bq = buf_copy_load(
                    bq_buffer,
                    bq_off_dw // fx.Int32(4),
                    fx.Int32,
                    unit_elems=4,
                    cache_modifier=0,
                )
                bqf[j][half].store(Vec(loaded_bq))
        for mw in range_constexpr(nPairs):
            fx.copy(
                sc_copy_atom,
                bscale_views[mw][lane_div_16, lane_mod_16, kt_rt, None],
                bsf[mw],
            )

    # Scaled-MFMA clusters over the loaded A / B / scale fragments (one E8M0 word per K-tile).
    def mfma_cluster(bqf, bsf, sa, kt_rt):
        # opsel (no gate/up split): mni=J//2, in_b=J%2; sa is a per-32-row-chunk list.
        sb_words = [Vec(bsf[mni].load())[0] for mni in range_constexpr(nPairs)]
        for J in range_constexpr(numAccN):
            mni, in_b = J // 2, J % 2
            sb = sb_words[mni]
            for sub in range_constexpr(kScaleSubBlocks):
                mma_one_j(
                    J,
                    in_b,
                    sa[sub],
                    sb,
                    bqf,
                    a_frags,
                    c_frags,
                    mma_atoms,
                    i0=2 * sub,
                    k_halves=kHalves,
                )

    def mfma_row(bqf, bsf, sa, i):
        # Preserve each row's K-half order and E8M0 operand selection.
        for J in range_constexpr(numAccN):
            sb = Vec(bsf[J // 2].load())[0]
            for half in range_constexpr(kHalves):
                osa = 2 * half + i % 2
                osb = 2 * half + J % 2
                fx.gemm(
                    mma_atoms[(osa, osb)],
                    c_frags[i][J],
                    a_frags[i][half],
                    bqf[J][half],
                    c_frags[i][J],
                    scale_a=sa[i // 2],
                    scale_b=sb,
                )

    # C accumulator: register fragments, zeroed then accumulated in place; (un)packed to K-loop carry.
    zero4 = fx.full(4, 0.0, Float32)
    c_frags = [
        [fx.make_rmem_tensor(4, Float32) for _ in range_constexpr(numAccN)]
        for _ in range_constexpr(kMChunks)
    ]
    for i in range_constexpr(kMChunks):
        for J in range_constexpr(numAccN):
            c_frags[i][J].store(zero4)

    def load_c_carry():
        return [c_frags[i][J].load() for i in range(kMChunks) for J in range(numAccN)]

    def store_c_carry(state):
        n = 0
        for i in range_constexpr(kMChunks):
            for J in range_constexpr(numAccN):
                c_frags[i][J].store(state[n])
                n += 1
        return n

    # 2-stage B pipeline: consume carried "current" B, prefetch next tile into the same fragments via scf.for state.
    cur_bqf = [
        [fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(kHalves)]
        for _ in range_constexpr(numAccN)
    ]
    cur_bsf = [fx.make_fragment_like(sc_frag_tmpl) for _ in range_constexpr(nPairs)]
    nxt_bqf = [
        [fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(kHalves)]
        for _ in range_constexpr(numAccN)
    ]
    nxt_bsf = [fx.make_fragment_like(sc_frag_tmpl) for _ in range_constexpr(nPairs)]
    # g2_ascale_pf: carry the A-scale through scf.for state, same rotating-buffer model as B.
    cur_saf = nxt_saf = None
    cur_saf = [
        fx.make_fragment_like(sc_frag_tmpl) for _ in range_constexpr(kScaleSubBlocks)
    ]
    nxt_saf = [
        fx.make_fragment_like(sc_frag_tmpl) for _ in range_constexpr(kScaleSubBlocks)
    ]

    def load_b_carry():
        # Flat CURRENT (to-consume) B-weight, B-scale, then (opt) A-scale values.
        out = []
        for j in range_constexpr(numAccN):
            for half in range_constexpr(kHalves):
                out.append(cur_bqf[j][half].load())
        for mw in range_constexpr(nPairs):
            out.append(cur_bsf[mw].load())
        for sub in range_constexpr(kScaleSubBlocks):
            out.append(cur_saf[sub].load())
        return out

    def store_b_carry(state, base):
        n = base
        for j in range_constexpr(numAccN):
            for half in range_constexpr(kHalves):
                cur_bqf[j][half].store(state[n])
                n += 1
        for mw in range_constexpr(nPairs):
            cur_bsf[mw].store(state[n])
            n += 1
        for sub in range_constexpr(kScaleSubBlocks):
            cur_saf[sub].store(state[n])
            n += 1
        return n

    def rotate_b_carry():
        # Yield the PREFETCHED (next-tile) values -> become "current" next iteration.
        out = []
        for j in range_constexpr(numAccN):
            for half in range_constexpr(kHalves):
                out.append(nxt_bqf[j][half].load())
        for mw in range_constexpr(nPairs):
            out.append(nxt_bsf[mw].load())
        for sub in range_constexpr(kScaleSubBlocks):
            out.append(nxt_saf[sub].load())
        return out

    def issue_a_scale_load_into(saf, kt_rt):
        # A-scale vmem load(s) for K-tile kt_rt into the given (per-stage) fragment(s).
        sa = load_a_scale_tile(kt_rt)
        for sub in range_constexpr(kScaleSubBlocks):
            saf[sub].store(fx.full(1, sa[sub], Int32))

    def load_carry():
        return load_c_carry() + load_b_carry()

    def store_carry(state):
        base = store_c_carry(state)
        store_b_carry(state, base)

    def yield_carry():
        return load_c_carry() + rotate_b_carry()

    # Prologue: prefetch tile 0's B/B-scale into "current" (VALUES enter via init=load_carry()).
    issue_b_load_into(cur_bqf, cur_bsf, fx.Int32(0))
    issue_a_scale_load_into(cur_saf, fx.Int32(0))
    rocdl.sched_barrier(0)

    def prefetch_next_b(kt_rt):
        # Prefetch NEXT tile's B; if none, copy current through (rotate_b_carry state, unused after loop).
        nxt_b = kt_rt + fx.Int32(1)
        if nxt_b < K_TILES_RT:
            issue_b_load_into(nxt_bqf, nxt_bsf, nxt_b)
            issue_a_scale_load_into(nxt_saf, nxt_b)
        else:
            for j in range_constexpr(numAccN):
                for half in range_constexpr(kHalves):
                    nxt_bqf[j][half].store(cur_bqf[j][half].load())
            for mw in range_constexpr(nPairs):
                nxt_bsf[mw].store(cur_bsf[mw].load())
            for sub in range_constexpr(kScaleSubBlocks):
                nxt_saf[sub].store(cur_saf[sub].load())

    for kt_iv, state in range(
        fx.Int32(0),
        K_TILES_RT,
        fx.Int32(1),
        init=load_carry(),
    ):
        store_carry(state)
        kt_rt = fx.Int32(kt_iv)
        if const_expr(not stream_a):
            prefetch_next_b(kt_rt)
        gpu.barrier()
        nxt_a = kt_rt + fx.Int32(kStages)
        if const_expr(not stream_a):
            issue_a_ds_read(kt_rt % fx.Int32(aStages))
        if nxt_a < K_TILES_RT:
            issue_a_load_lds(nxt_a % fx.Int32(aStages), nxt_a)
        # A-scale from the prefetch carry (g2_ascale_pf) or loaded synchronously here.
        sa = [Vec(cur_saf[sub].load())[0] for sub in range_constexpr(kScaleSubBlocks)]
        # Fence the MFMA chain from the B vmem loads (next-tile loads ride ahead of compute).
        rocdl.sched_barrier(0)
        rocdl.s_setprio(1)
        if const_expr(stream_a):
            # A row is consumed before reading the next. B is reused over M;
            # refill its next K tile only when the final M row starts.
            for i in range_constexpr(kMChunks):
                issue_a_ds_read(kt_rt % fx.Int32(aStages), i, 1)
                if const_expr(i == kMChunks - 1):
                    prefetch_next_b(kt_rt)
                mfma_row(cur_bqf, cur_bsf, sa, i)
        else:
            mfma_cluster(cur_bqf, cur_bsf, sa, kt_rt)
        rocdl.s_setprio(0)
        rocdl.sched_barrier(0)
        results = yield yield_carry()
    store_carry(results)

    # Load the C fragments (fp8/fp4 unified onto the same fx.gemm path) and hand them to the epilog.
    accm_vecs = [
        [c_frags[i][J].load() for J in range(numAccN)] for i in range(kMChunks)
    ]
    return accm_vecs, m_row, n_block_idx, N_OUT_rt
