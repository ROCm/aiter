# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import dpp_utils
from aiter.ops.flydsl.kernels.communication_ops_utils import (
    atomic_add_agent as _nc_atomic_add_agent,
)
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import (
    batched_silu_swiglu,
    batched_situv2,
    fclamp_f32,
    fmin_f32,
    situv2_consts,
)
from aiter.ops.flydsl.kernels.layout_utils import crd2idx
from .gemm_common import (
    MXFP4_SCALE_LAYOUT_TAG,
    _buffer_rsrc,
    _lds_ptr3,
    _e8m0_from_amax,
    _e8m0_roundup,
    _fabs_f32,
    _inline_dpp_quad_amax,
    _lds_swizzle_mask,
    _raw,
    _umax_i32,
    bq_bytes_for,
    bscale_bytes_for,
    k_half_for,
    k_tiles_total_for,
    kas_per_chunk_dw_for,
    kbs_per_expert_dw_for,
    kBS_stride_k0_dw,
    kbs_stride_n0_dw_for,
    kmchunks_for,
    kStages,
    kunroll_for,
    lds_acc_bytes_for,
    num_n_blocks_for,
)


def _udiv(a, c):
    return fx.Int32(fx.Uint32(a) // fx.Uint32(c))


def _umod(a, c):
    return fx.Int32(fx.Uint32(a) % fx.Uint32(c))


def _global_i32_ptr(addr_i64):
    ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=4
    )
    return fx.inttoptr(ptr_ty, fx.Int64(addr_i64))


def _global_i32_at(addr_i64, idx):
    # fx.ptr_load/add_offset for a plain scalar read -- no tiling needed, so
    # skip the Tensor/tile/register-fragment machinery fx.copy requires.
    return _global_i32_ptr(addr_i64)[idx]


def _global_i32_load(tiles, idx):
    # Must build atom/types here, not as module globals -- they need an active
    # MLIR trace context.
    atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
    reg_lay = fx.make_layout(1, 1)
    r = fx.make_rmem_tensor(reg_lay, fx.Int32)
    fx.copy_atom_call(atom, fx.slice(tiles, (None, idx)), r)
    return r.load()[0]


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
    fx.copy_atom_call(atom, r, fx.slice(tiles, (None, idx)))


def _layout_idx(layout, *coords):
    idx_coords = [fx.Int64(c) for c in coords]
    flat = crd2idx(idx_coords, layout)
    return fx.Int32(flat)


def n_out_for(inter):
    return 2 * inter


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
        # silu + clamp(DSV4):gate<=L, -L<=up<=L 后走同一个 exp2+rcp 的 SiLU。
        # batched_silu_swiglu 用 v_tanh(gfx1250 才有),gfx950 编译失败。
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
            7.0 if act == "swiglu" and swiglu_limit is None
            else float("inf") if swiglu_limit is None
            else float(swiglu_limit)
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
        return batched_situv2(
            pairs, consts=consts, range_constexpr=range_constexpr
        )
    raise ValueError(f"unsupported A4W4 GMM1 activation {act!r}")


def _pkmax_u16(a_i32, b_i32):
    _v2i16 = T.vec(2, T.i16)
    va = llvm.BitcastOp(_v2i16, _raw(a_i32)).result
    vb = llvm.BitcastOp(_v2i16, _raw(b_i32)).result
    vm = arith.MaxUIOp(va, vb).result
    out = llvm.BitcastOp(T.i32, vm).result
    return fx.Int32(out)


def _inline_e8m0(amax_u16_i32):
    f32 = fx.Float32(
        _raw((fx.Int32(_raw(amax_u16_i32)) & fx.Int32(0xFFFF)) << fx.Int32(16)).bitcast(
            T.f32
        )
    )
    return _e8m0_roundup(f32)


def gemm1_grid(n_tokens, BM, *, NE, TOPK, INTER, BN=256):
    num_n_blocks = num_n_blocks_for(n_out_for(INTER), BN)
    if BM == 128:
        max_m_blocks = (n_tokens * TOPK + NE * (BM - 1) + BM - 1) // BM
    else:
        active = min(n_tokens * TOPK, NE)
        max_m_blocks = (n_tokens * TOPK + active * (BM - 1) + BM - 1) // BM
    return max_m_blocks * num_n_blocks


@flyc.jit
def _gemm1_body_sc2(
    lds_raw_ptr,
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
    arg_tile_perm,
    arg_nc_head,
    arg_nc_list,
    i32_nc_bound,
    i32_nc_off,
    *,
    BM,
    BN,
    BK,
    expert_major=False,
    inline_quant=False,
    ascale_gather=False,
    K,
    N_OUT,
    NE,
    interleave=False,
    act="silu",
    swiglu_limit=None,
    situ_beta=4.0,
    situ_linear_beta=25.0,
    next_claim=False,
    next_claim_lds_dw=0,
    next_claim_lead=2,
):
    KH_TILE = BK // 2
    K_HALF = k_half_for(K)
    K_TILES_TOTAL = k_tiles_total_for(K, BK)
    kUnroll = kunroll_for(K, BK)
    kAS_per_chunk_dw = kas_per_chunk_dw_for(K)
    kBS_stride_n0_dw = kbs_stride_n0_dw_for(K)
    kBS_per_expert_dw = kbs_per_expert_dw_for(N_OUT, K)
    BQ_BYTES = bq_bytes_for(NE, N_OUT, K)
    BSCALE_BYTES = bscale_bytes_for(NE, N_OUT, K)
    NUM_N_BLOCKS = num_n_blocks_for(N_OUT, BN)
    inter = N_OUT // 2
    OUT_AS_PER_CHUNK_DW = kas_per_chunk_dw_for(inter)
    K_G2_HALF = k_half_for(inter)
    kAStages, kSubBlocks, kMChunks, _ = _bm_constants(BM, BN, KH_TILE, K_TILES_TOTAL)

    BN_INT = BN // 2
    b_aux = 2 if use_nt else 0
    M_REPS = BM // 16
    # 每 wave 的 16 列 N 子块数:BN=256 为 4(gate/up 各两个),BN=128 为 2(gate/up 各一个,
    # 与小算子 t128 同为每 wave 32 列)。BN=128 只支持生产路径(非 interleave、非 inline_quant)。
    NJ = BN // 64
    assert BN in (128, 256), f"GMM1 BN must be 128 or 256, got {BN}"
    assert BN == 256 or not (interleave or inline_quant), "BN=128 needs the non-interleave, non-inline path"

    # next_claim(MEGAMOE_TK_GMM1_NEXT_CLAIM,stage1 early_local_gmm 专用):在本 job 的第
    # K_TILES_TOTAL-lead 步开头由 tx0 发下一个 job 的领位原子(arg_nc_head),结果留在 VGPR
    # 里跨过最后 lead 步;K 循环后的 barrier 之后查组表 arg_nc_list,epilogue 末尾把映射好的
    # 物理 job 号(越界 = -1)写进 LDS 信箱 dword next_claim_lds_dw。调用方在 job 后的 barrier
    # 之后直接读信箱,省掉逐 job 的「barrier + 原子 + 系统 load + LDS 广播 + barrier + 组表 load」。
    # 信箱必须在 A 环/scale/累加器复用区之外(调用方多分配 16B)。
    # 原子是 monotonic、只在 wave0 发:它排在已计数的 vmem 之后,_wait_lds_barrier 的 vmcnt
    # 只会变松不会变错(in-order 返回,A(kt+1) 之后的条数只多不少)。
    if const_expr(next_claim):
        assert not inline_quant, "next_claim only on the production (non-inline) pipeline"
        _nc_lds_min = _bm_constants(BM, BN, KH_TILE, K_TILES_TOTAL)[3]
        assert next_claim_lds_dw * 4 >= _nc_lds_min, "next_claim mailbox aliases the GMM1 LDS"
        assert next_claim_lead >= 1
        _NC_STEP = max(0, K_TILES_TOTAL - int(next_claim_lead))

        nc_tx0 = fx.Int32(gpu.thread_id("x")) == fx.Int32(0)
        # 非 tx0 线程的占位值:保证 < bound 为假,组表只读第 0 项(不越界)。
        nc_raw = fx.Int32(0x7FFFFFFF)

    n_block_idx = bx_i32 % fx.Int32(NUM_N_BLOCKS)
    m_block_idx = bx_i32 // fx.Int32(NUM_N_BLOCKS)
    e = rocdl.readfirstlane(T.i32, _raw(_global_i32_at(arg_eids, m_block_idx)))
    m_row = m_block_idx * fx.Int32(BM)
    # Stage1 allocates physical tiles in arrival order, which scatters
    # experts across consecutive m-blocks and makes GEMM2 reload one
    # expert's weights per block.  Under expert_major this body still
    # reads its source tile but stores the result at the expert-major
    # slot, so Stage2 sees contiguous runs with no data movement.
    m_block_out = (
        rocdl.readfirstlane(
            T.i32, _raw(_global_i32_at(arg_tile_perm, m_block_idx))
        )
        if const_expr(expert_major)
        else m_block_idx
    )
    m_row_out = m_block_out * fx.Int32(BM)

    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lane_div_8 = lane // fx.Int32(8)
    lane_mod_8 = lane % fx.Int32(8)

    aq_num_records = fx.Int64(i32_ntok * fx.Int32(K_HALF))
    _asc_per_mb = max(BM // 32, 1) * kAS_per_chunk_dw * 4
    ascale_num = fx.Int64(i32_total_m_blocks) * fx.Int64(_asc_per_mb)

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

    # LDS alias scopes(生产路径:ascale_gather 且非 inline_quant)。A 的 LDS DMA 计入 vmcnt;
    # LLVM 在任何 LDS 读写前,对没有 alias.scope 的访问保守地等全部在途的 LDS DMA,
    # 于是每步 A 的 ds_read 前被插 vmcnt(0)(ATT:GMM1 41% 周期),A 两步预取形同虚设。
    # 每个 A 槽一个 scope、A-scale 区一个 scope,读写都带上,LLVM 只等同槽的 DMA。
    _lds_scoped = (
        ascale_gather
        and not inline_quant
        and __import__("os").environ.get("MEGAMOE_TK_GMM1_LDS_SCOPES", "1") != "0"
    )
    if const_expr(_lds_scoped):
        from flydsl._mlir import ir as _ir

        _dom = "#llvm.alias_scope_domain<id = distinct[0]<>, description = \"gmm1_lds\">"
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
        _scopes = [_sc[i] for i in range(kAStages + 1)]  # [A 槽 0..kAStages-1, A-scale]

        def _scope_attrs(idx):
            return dict(
                alias_scopes=_ir.ArrayAttr.get([_scopes[idx]]),
                noalias_scopes=_ir.ArrayAttr.get(
                    [_scopes[j] for j in range(kAStages + 1) if j != idx]
                ),
            )

        _lds_base_i32 = fx.Int32(fx.ptrtoint(lds_raw_ptr))
        _asc_base_i32 = _lds_base_i32 + fx.Int32(kAStages * BM * KH_TILE)
        aq_rsrc = _buffer_rsrc(arg_aq, aq_num_records)

    # aq/ascale: global->LDS async DMA (no register fragment), via BufferCopyLDS.
    aq_buf = _global_i32_buffer_view(arg_aq, aq_num_records)
    aq_dma_tiles4 = fx.logical_divide(aq_buf, fx.make_layout(4, 1))
    aq_dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), fx.Int32)

    ascale_buf = _global_i32_buffer_view(arg_ascale, ascale_num)
    ascale_dma_tiles4 = fx.logical_divide(ascale_buf, fx.make_layout(4, 1))
    ascale_dma_tiles1 = fx.logical_divide(ascale_buf, fx.make_layout(1, 1))
    ascale_dma_atom16 = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), fx.Int32)
    ascale_dma_atom4 = fx.make_copy_atom(fx.rocdl.BufferCopyLDS32b(), fx.Int32)

    if const_expr(inline_quant):
        hidden_num = fx.Int64(i32_ntok * fx.Int32(K * 2))
        hidden_tiles = _global_i32_buffer_tiles(arg_hidden, hidden_num, 4)
        hidden_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        hidden_reg_lay = fx.make_layout(4, 1)

    # Union LDS region [s_aq | s_asc], reused as lds_acc (f32 accumulator) in
    # the epilogue. s_aq/lds_acc start at lds_raw_ptr; s_asc follows at
    # +kAStages*BM*KH_TILE.

    cached_actual_row = []
    cached_row_inline = None
    if const_expr(inline_quant):
        rcls = wave * fx.Int32(4) + lane_div_16
        cached_row_inline = _global_i32_at(arg_mind, m_row + rcls)
    else:
        for sub in range_constexpr(kSubBlocks):
            idx = m_row + wave * fx.Int32(BM // 4) + fx.Int32(sub * 8) + lane_div_8
            cached_actual_row.append(_global_i32_at(arg_mind, idx))

    # -- b_load_s_base[j] (HIP 412-416), readfirstlane'd uniform per wave ------
    N0_HALF = N_OUT // 32
    b_load_s_base = []
    for j in range_constexpr(NJ):
        if const_expr(interleave):
            col = (
                n_block_idx * fx.Int32(BN) + wave * fx.Int32(BN // 4) + fx.Int32(j * 16)
            )
        else:
            tile_il = n_block_idx * fx.Int32(BN // 16) + wave * fx.Int32(NJ) + fx.Int32(j)
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
        # B-scale 一个字覆盖相邻两个 16 列 n0。BN=256:本 wave 的两个 n0 正好一对;
        # BN=128:本 wave 只有一个 n0 = n_block*4+wave,取所在的那一对,字节在 issue_b_scale_load 里按 wave&1 移位。
        np_gate = n_block_idx * fx.Int32(BN // 64) + (
            wave if BN == 256 else wave // fx.Int32(2)
        )
        np_list = [np_gate, np_gate + fx.Int32(N_OUT // 64)]
    b_scale_s_base, b_scale_s_base_hi = [], []
    for mw in range_constexpr(2):
        base = (
            e * fx.Int32(kBS_per_expert_dw) + np_list[mw] * fx.Int32(kBS_stride_n0_dw)
        ) * fx.Int32(4)
        base = rocdl.readfirstlane(T.i32, base)
        b_scale_s_base.append(base)
        b_scale_s_base_hi.append(base + fx.Int32(16 * kBS_stride_k0_dw * 4))

    accm = [[None] * NJ for _ in range(kMChunks)]
    b = [[[None, None] for _ in range(NJ)] for _ in range(kStages)]
    b_scale_v = [[None, None] for _ in range(kStages)]

    def issue_a_load_lds(slot, kt):
        for sub in range_constexpr(kSubBlocks):
            lds_row = wave * fx.Int32(BM // 4) + fx.Int32(sub * 8)
            mask = _lds_swizzle_mask(lds_row + lane_div_8)
            voffset = ((lane_mod_8 * fx.Int32(16)) ^ mask) + cached_actual_row[
                sub
            ] * fx.Int32(K_HALF)
            off = fx.Int32(slot * (BM * KH_TILE)) + lds_row * fx.Int32(KH_TILE)
            if const_expr(_lds_scoped):
                # 同 mxfp4_gemm2._issue_a_load_lds:LDS 基址 wave 一致,每 lane 16B 依次落位。
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
            else:
                fx.copy(
                    aq_dma_atom,
                    fx.slice(aq_dma_tiles4, (None, voffset // fx.Int32(16))),
                    fx.slice(s_aq_i32x4_tiles, (None, off // fx.Int32(16))),
                    soffset=fx.Int32(kt * KH_TILE) // fx.Int32(4),
                )

    # s_aq as flat i32, divided into 4-element (128-bit) and 1-element tiles.
    s_aq_i32_flat = fx.make_view(
        fx.recast_iter(fx.Int32, lds_raw_ptr),
        fx.make_layout(kAStages * BM * KH_TILE // 4, 1),
    )
    s_aq_i32x4_tiles = fx.logical_divide(s_aq_i32_flat, fx.make_layout(4, 1))
    s_aq_i32x1_tiles = fx.logical_divide(s_aq_i32_flat, fx.make_layout(1, 1))
    i32x4_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
    i32x4_reg_lay = fx.make_layout(4, 1)

    def _lds_i32x4_load(tile_idx):
        r = fx.make_rmem_tensor(i32x4_reg_lay, fx.Int32)
        fx.copy_atom_call(
            i32x4_copy_atom, fx.slice(s_aq_i32x4_tiles, (None, tile_idx)), r
        )
        return r.load()

    def issue_a_ds_read(slot):
        mask = _lds_swizzle_mask(lane_mod_16)
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
                if const_expr(_lds_scoped):
                    a[i][k] = Vec(
                        llvm.LoadOp(
                            Vec.make_type(4, fx.Int32),
                            _lds_ptr3(_lds_base_i32, off),
                            alignment=16,
                            **_scope_attrs(slot),
                        ).result
                    )
                else:
                    a[i][k] = _lds_i32x4_load(off // fx.Int32(16))
        return a

    def issue_a_scale_load():
        chunk_base = m_row // fx.Int32(32)
        v16 = (wave * fx.Int32(64) + lane) * fx.Int32(16)
        v4 = (wave * fx.Int32(64) + lane) * fx.Int32(4)
        for sub in range_constexpr(kSubBlocks):
            s_chunk = rocdl.readfirstlane(
                T.i32, (chunk_base + fx.Int32(sub)) * fx.Int32(kAS_per_chunk_dw * 4)
            )
            lds_sub = fx.Int32(sub * kAS_per_chunk_dw * 4)
            fx.copy(
                ascale_dma_atom16,
                fx.slice(ascale_dma_tiles4, (None, v16 // fx.Int32(16))),
                fx.slice(
                    asc_i32x4_tiles,
                    (None, (lds_sub + wave * fx.Int32(1024)) // fx.Int32(16)),
                ),
                soffset=s_chunk // fx.Int32(4),
            )
            for d in range_constexpr(3):
                byte_off = 4096 + d * 1024
                s_off = rocdl.readfirstlane(T.i32, s_chunk + fx.Int32(byte_off))
                fx.copy(
                    ascale_dma_atom4,
                    fx.slice(ascale_dma_tiles1, (None, v4 // fx.Int32(4))),
                    fx.slice(
                        asc_i32_tiles,
                        (
                            None,
                            (lds_sub + fx.Int32(byte_off) + wave * fx.Int32(256))
                            // fx.Int32(4),
                        ),
                    ),
                    soffset=s_off // fx.Int32(4),
                )

    # s_asc as flat i32.
    s_asc_i32_flat = fx.make_view(
        fx.recast_iter(fx.Int32, fx.add_offset(lds_raw_ptr, kAStages * BM * KH_TILE)),
        fx.make_layout(kSubBlocks * K_TILES_TOTAL * 64, 1),
    )
    asc_i32_tiles = fx.logical_divide(s_asc_i32_flat, fx.make_layout(1, 1))
    asc_i32x4_tiles = fx.logical_divide(s_asc_i32_flat, fx.make_layout(4, 1))

    def issue_a_scale_ds_read(kt):
        out = []
        for sub in range_constexpr(kSubBlocks):
            lds_dw = (
                fx.Int32(sub * kAS_per_chunk_dw)
                + fx.Int32(kt * 64)
                + lane_div_16 * fx.Int32(16)
                + lane_mod_16
            )
            if const_expr(_lds_scoped):
                out.append(
                    llvm.LoadOp(
                        T.i32,
                        _lds_ptr3(_asc_base_i32, lds_dw * fx.Int32(4)),
                        alignment=4,
                        **_scope_attrs(kAStages),
                    ).result
                )
            else:
                out.append(_raw(_global_i32_load(asc_i32_tiles, lds_dw)))
        return out

    # ascale_gather:A-scale 按源 token 行主序存放(每行 K/32 字节,由 stage1
    # 每 (token, dest) 连续写一次),这里按 arg_mind 取两行(n_lane、n_lane+16)
    # 各 8 字节,拼成 DMA 版本在 LDS 里的 BM32 预排布:
    #   dword(ku, k_lane, n_lane) = [A[8ku+k], B[8ku+k], A[8ku+4+k], B[8ku+4+k]]
    # (字节序 = ikxdl*2 + im_a,与 stage1 fanout 的散写同一公式)。
    # global load 在序章最前面发、LDS 写在序章之后,由序章末尾的
    # _wait_lds_barrier(lgkmcnt(0)+barrier)保证先写后读。
    _GS_ITEMS = kSubBlocks * 16 * K_TILES_TOTAL
    _GS_PASS = (_GS_ITEMS + 255) // 256
    _GS_ROW_DW = K // 32 // 4

    def gather_a_scale_issue():
        tid = wave * fx.Int32(64) + lane
        loads = []
        for p in range_constexpr(_GS_PASS):
            w = tid + fx.Int32(p * 256)
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
            loads.append(
                (
                    sub, n_lane, ku,
                    _global_i32_at(arg_ascale, da),
                    _global_i32_at(arg_ascale, da + fx.Int32(1)),
                    _global_i32_at(arg_ascale, db),
                    _global_i32_at(arg_ascale, db + fx.Int32(1)),
                )
            )
        return loads

    def gather_a_scale_write(loads):
        m8 = fx.Int32(0xFF)
        # 越界的工作项已夹到第 0 项:写同一位置同一值,无分支。
        for sub, n_lane, ku, a_lo, a_hi, b_lo, b_hi in loads:
            if True:
                for k in range_constexpr(4):
                    sh = fx.Int32(8 * k)
                    v = (
                        ((a_lo >> sh) & m8)
                        | (((b_lo >> sh) & m8) << fx.Int32(8))
                        | (((a_hi >> sh) & m8) << fx.Int32(16))
                        | (((b_hi >> sh) & m8) << fx.Int32(24))
                    )
                    lds_dw = (
                        sub * fx.Int32(kAS_per_chunk_dw)
                        + ku * fx.Int32(64)
                        + fx.Int32(k * 16)
                        + n_lane
                    )
                    if const_expr(_lds_scoped):
                        llvm.StoreOp(
                            _raw(v),
                            _lds_ptr3(_asc_base_i32, lds_dw * fx.Int32(4)),
                            alignment=4,
                            **_scope_attrs(kAStages),
                        )
                    else:
                        fx.ptr_store(
                            fx.Vector.from_elements([v], fx.Int32),
                            fx.add_offset(
                                fx.recast_iter(
                                    fx.Int32,
                                    fx.add_offset(lds_raw_ptr, kAStages * BM * KH_TILE),
                                ),
                                lds_dw,  # add_offset 按元素(i32)计,不是字节
                            ),
                        )

    lib = lane & fx.Int32(3)
    lane_shr2_and3 = (lane >> fx.Int32(2)) & fx.Int32(3)
    r_in_chunk = wave * fx.Int32(4) + lane_div_16

    def inline_quant_load_kt(B128_IDX, kt, row_token):
        v_voff = (
            row_token * fx.Int32(K * 2)
            + lane_shr2_and3 * fx.Int32(64)
            + lib * fx.Int32(16)
        )
        s_soff = rocdl.readfirstlane(T.i32, fx.Int32(kt * (BK * 2) + B128_IDX * 256))
        r = fx.make_rmem_tensor(hidden_reg_lay, fx.Int32)
        fx.copy(
            hidden_copy_atom,
            fx.slice(hidden_tiles, (None, v_voff // fx.Int32(16))),
            r,
            soffset=s_soff // fx.Int32(4),
        )
        return r.load()

    def _inline_quant_core_batch(specs, slot, scale_accum):
        n = len(specs)
        h_dw = [
            [fx.Int32(_raw(h_v[j])) for j in range_constexpr(4)]
            for (_b, _s, h_v) in specs
        ]
        la = [None] * n
        for i in range_constexpr(n):
            hm = [h_dw[i][j] & fx.Int32(0x7FFF7FFF) for j in range_constexpr(4)]
            m01 = _pkmax_u16(hm[0], hm[1])
            m23 = _pkmax_u16(hm[2], hm[3])
            m0123 = _pkmax_u16(m01, m23)
            lo = m0123 & fx.Int32(0xFFFF)
            hi = m0123.shrui(fx.Int32(16)) & fx.Int32(0xFFFF)
            la[i] = _umax_i32(lo, hi)
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
        e8 = [_inline_e8m0(a[i]) for i in range_constexpr(n)]
        for i in range_constexpr(n):
            B128_IDX, SUB, _hv = specs[i]
            qs_raw = _raw(fx.Float32(_raw(e8[i] << fx.Int32(23)).bitcast(T.f32)))
            pk = _raw(fx.Int32(0))
            for j in range_constexpr(4):
                src_bf16x2 = _raw(
                    fx.Vector.from_elements([h_dw[i][j]], fx.Int32).bitcast(fx.BFloat16)
                )
                pk = rocdl.cvt_scalef32_pk_fp4_bf16(T.i32, pk, src_bf16x2, qs_raw, j)
            pk = fx.Int32(pk)
            r = fx.Int32(SUB * 16) + r_in_chunk
            kb_in_kt = fx.Int32(B128_IDX * 4) + lane_shr2_and3
            mask_r = _lds_swizzle_mask(r)
            b_off = lib * fx.Int32(4)
            off = (
                fx.Int32(slot * (BM * KH_TILE))
                + r * fx.Int32(KH_TILE)
                + ((kb_in_kt * fx.Int32(16)) ^ mask_r)
                + b_off
            )
            _scalar_store(s_aq_i32x1_tiles, off // fx.Int32(4), pk, fx.Int32)
            pack_byte = B128_IDX * 2 + SUB
            scale_accum = scale_accum | (e8[i] << fx.Int32(pack_byte * 8))
        return scale_accum

    def inline_quant_kt(B128_IDX, SUB, slot, kt, row_token, scale_accum):
        h_v = inline_quant_load_kt(B128_IDX, kt, row_token)
        return _inline_quant_core_batch([(B128_IDX, SUB, h_v)], slot, scale_accum)

    def inline_quant_pack_write(kt, scale_accum):
        lane_tgt = lane_shr2_and3 * fx.Int32(16) + r_in_chunk
        off = fx.Int32(kt * 256) + lane_tgt * fx.Int32(4)
        _scalar_store(asc_i32_tiles, off // fx.Int32(4), scale_accum, fx.Int32)

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
            if const_expr(BN == 128):
                # 本 wave 的 n0 是这一对里的第 (wave&1) 个:右移一个字节后按 in_b=0 取。
                bs_slot[mw] = r.load()[0].shrui((wave & fx.Int32(1)) * fx.Int32(8))
            else:
                bs_slot[mw] = r.load()[0]

    mfma_ty = T.f32x4
    zero4 = fx.Vector.filled(4, 0.0, fx.Float32)

    def mfma_cluster(b_slot, a, a_scale, bs_slot, J, init):
        if const_expr(interleave):
            mni = J // 2
            in_b = J % 2
        elif const_expr(BN == 128):
            mni = J
            in_b = 0
        else:
            mni = J % 2
            in_b = J // 2
        sb = bs_slot[mni]
        bJ0, bJ1 = b_slot[J][0], b_slot[J][1]
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
        else:
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

    # 每个 K 步 B 发出的 vmem 条数:4 个 j x 2 个 half 的 dwordx4 + 2 条 B-scale。
    _B_VMEM_PER_STEP = NJ * 2 + 2

    # 裸 s_barrier(MEGAMOE_TK_GMM1_NOFENCE_BAR=1,仅 scoped 路径)。gpu.barrier() 降成
    # fence release(workgroup, LDS) + s_barrier + fence acquire;SIMemoryLegalizer 在该
    # release fence 上发 S_WAITCNT_lds_direct,SIInsertWaitcnts 把它换成「等全部在途 LDS DMA」
    # (通用 LDS 槽,不看 alias scope),并入前面那条显式 s_waitcnt:vmcnt(24) 被收紧成
    # vmcnt(10),即 A(kt+2) 被迫提前一步落地(ISA 28 处全是 vmcnt(10) lgkmcnt(0))。
    # 这里跨 wave 可见性已由显式 vmcnt(N)+lgkmcnt(0) 保证,fence 不提供额外必需的顺序。
    _nofence_bar = (
        _lds_scoped
        and __import__("os").environ.get("MEGAMOE_TK_GMM1_NOFENCE_BAR", "1") != "0"
    )

    def _wait_lds_barrier(vmcnt):
        # 同 mega_moe/gemm_util.wait_lds_barrier:lgkmcnt(0) 等本 wave 的 ds_read 读完,
        # vmcnt(N) 等本步开头发出的 A DMA 落地(之后发的 N 条 B 预取继续飞),再 barrier。
        rocdl.s_waitcnt((vmcnt & 0xF) | ((vmcnt & 0x30) << 10) | (7 << 4))
        if const_expr(_nofence_bar):
            rocdl.s_barrier()
        else:
            gpu.barrier()

    if const_expr(not inline_quant):
        # MegaMoEv2 do_tile 的同步写法(每步末尾一把 wait_lds_barrier),A 三级、DMA 提前两步:
        #   第 kt 步开头把 A(kt+2) 搬进槽 (kt+2)%3 —— 该槽上一步已被全体 wave 读完
        #   (上一步末尾的 lgkmcnt(0)+barrier 保证);
        #   第 kt 步末尾只等 A(kt+1) 落地,它之后发出的 B(kt+1)、A(kt+2)、B(kt+2) 继续飞。
        # A 走 LDS DMA,每个 wave 只搬 1/4 行却读全部行,两个条件缺一不可。
        _A_DMA_PER_STEP = kSubBlocks

        def _vmcnt_after_a(kt_next):
            # A(kt_next) 之后按程序序发出的 vmem 条数:B(kt_next),以及(若存在)A(kt_next+1)、B(kt_next+1)
            n = _B_VMEM_PER_STEP
            if kt_next + 1 < K_TILES_TOTAL:
                n += _A_DMA_PER_STEP + _B_VMEM_PER_STEP
            return n

        if const_expr(ascale_gather):
            gs_loads = gather_a_scale_issue()
        else:
            issue_a_scale_load()
        for K_C in range_constexpr(kStages):
            if const_expr(K_C < K_TILES_TOTAL):
                rocdl.sched_barrier(0)
                issue_a_load_lds(K_C % 3, K_C)
                rocdl.sched_barrier(0)
                for j in range_constexpr(NJ):
                    issue_b_load_j(b[K_C], K_C, j)
                issue_b_scale_load(b_scale_v[K_C], K_C)
                rocdl.sched_barrier(0)
        if const_expr(ascale_gather):
            gather_a_scale_write(gs_loads)
        _wait_lds_barrier(_vmcnt_after_a(0))
        for kt in range_constexpr(K_TILES_TOTAL):
            has_next = kt + 1 < K_TILES_TOTAL
            refill = kt + kStages < K_TILES_TOTAL
            slot_b = kt % kStages
            if const_expr(next_claim and kt == _NC_STEP):
                rocdl.sched_barrier(0)
                if nc_tx0:
                    nc_raw = fx.Int32(
                        _nc_atomic_add_agent(arg_nc_head, fx.Int32(1))
                    )
                rocdl.sched_barrier(0)
            if const_expr(refill):
                rocdl.sched_barrier(0)
                issue_a_load_lds((kt + kStages) % 3, kt + kStages)
                rocdl.sched_barrier(0)
            if const_expr(BM == 128):
                asc_cur = issue_a_scale_ds_read(kt)
                a_cur = issue_a_ds_read(kt % 3)
            else:
                a_cur = issue_a_ds_read(kt % 3)
                asc_cur = issue_a_scale_ds_read(kt)
            for J in range_constexpr(NJ):
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
    else:
        # inline_quant 变体(独立 gemm1)保留原流水。
        _relax_prologue = (BM == 128) and not inline_quant
        if const_expr(not inline_quant):
            issue_a_scale_load()
        for K_C in range_constexpr(kStages):
            if const_expr(inline_quant):
                scale_accum = fx.Int32(0)
                scale_accum = inline_quant_kt(
                    0, 0, K_C, K_C, cached_row_inline, scale_accum
                )
                issue_b_load_j(b[K_C], K_C, 0)
                issue_b_load_j(b[K_C], K_C, 1)
                scale_accum = inline_quant_kt(
                    1, 0, K_C, K_C, cached_row_inline, scale_accum
                )
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

        for OFFSET in range_constexpr(kUnroll):
            K_C = kStages + OFFSET
            read_slot = OFFSET % kAStages
            write_slot = K_C % kAStages
            slot_b = OFFSET % kStages
            gpu.barrier()
            if const_expr(BM == 128):
                asc_cur = issue_a_scale_ds_read(K_C - kStages)
                a_cur = issue_a_ds_read(read_slot)
            else:
                a_cur = issue_a_ds_read(read_slot)
                asc_cur = issue_a_scale_ds_read(K_C - kStages)
            if const_expr(not inline_quant):
                issue_a_load_lds(write_slot, K_C)
            if const_expr(inline_quant):
                h_v0 = inline_quant_load_kt(0, K_C, cached_row_inline)
                h_v1 = inline_quant_load_kt(1, K_C, cached_row_inline)
                rocdl.sched_barrier(0)
            for J in range_constexpr(4):
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
            if const_expr(inline_quant):
                scale_accum = _inline_quant_core_batch(
                    [(0, 0, h_v0), (1, 0, h_v1)], write_slot, fx.Int32(0)
                )
                inline_quant_pack_write(K_C, scale_accum)

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

    if const_expr(next_claim):
        # 下一个 job 的组表项在 epilogue 期间飞;原子此时早已返回(K 循环末尾 vmem 已排空)。
        nc_valid = nc_raw < i32_nc_bound
        nc_j = nc_raw + i32_nc_off
        nc_g = nc_valid.select(nc_j // fx.Int32(NUM_N_BLOCKS), fx.Int32(0))
        nc_p = _global_i32_at(arg_nc_list, nc_g)

    # lds_acc reuses the s_aq region (offset 0) as an f32 accumulator.
    _epi_swz = _epi_swz_on(BN)
    if const_expr(_epi_swz):
        # 见 _epi_swz_on 上方注释:只把行距改成 BN+4(不做列异或)。_bm_constants 已按 BN+4 定 LDS。
        _epi_swz_selfcheck(BM, BN)
        ACC_STRIDE = BN + _EPI_ACC_PAD
    else:
        ACC_STRIDE = BN
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
        fx.copy_atom_call(acc_copy_atom, r, fx.slice(acc_flat_tiles, (None, idx)))

    def acc_load(idx):
        r = fx.make_rmem_tensor(acc_reg_lay, fx.Float32)
        fx.copy_atom_call(acc_copy_atom, fx.slice(acc_flat_tiles, (None, idx)), r)
        return r.load()[0]

    for i in range_constexpr(kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for J in range_constexpr(NJ):
            is_up = (J % 2) == 1
            J_local = J // 2
            col_local = wave * fx.Int32(BN // 8) + fx.Int32(J_local * 16) + lane_mod_16
            lds_col = (fx.Int32(BN // 2) + col_local) if is_up else col_local
            vec = fx.Vector(accm[i][J])
            for v in range_constexpr(4):
                idx = acc_idx(row_base + fx.Int32(v), lds_col)
                acc_store(idx, vec[v])

    gpu.barrier()

    if const_expr(BN == 128):
        # BN=128:每块 64 个输出列 = 2 个 1x32 scale 组。256 线程 = 32 行 x 8 线程,
        # 线程 (wave_grp_b, kk_b) 管 [wave_grp_b*32 + kk_b*8, +8) 列;每遍 32 行 = 一个 scale chunk_b。
        tx_i32_b = fx.Int32(gpu.thread_id("x"))
        m_lane_b = tx_i32_b // fx.Int32(8)
        n_lane_b = tx_i32_b % fx.Int32(8)
        wave_grp_b = n_lane_b // fx.Int32(4)
        kk_b = n_lane_b % fx.Int32(4)
        aqout_layout_b = fx.make_layout((BM, K_G2_HALF), (K_G2_HALF, 1))
        aqout_tiles_b = _global_scalar_tiles(arg_aqout, fx.Int32, 1 << 24)
        ascaleout_layout_b = fx.make_layout(
            (1 << 20, 2, 4, 16), (OUT_AS_PER_CHUNK_DW, 64, 16, 1)
        )
        ascaleout_i8_tiles_b = _global_scalar_tiles(arg_ascaleout, fx.Int8, 1 << 26)
        # 按 BN=256 的预排布定位:全局 32 列组 sg_b = n_block*2 + wave_grp_b,
        # 对应旧块 old_nb_b = sg_b//4、旧 wave_grp_b = sg_b%4;ku_b/ikxdl_b 同 BN=256 的公式。
        sg_b = n_block_idx * fx.Int32(2) + wave_grp_b
        old_nb_b = sg_b // fx.Int32(4)
        old_wg_b = sg_b % fx.Int32(4)
        ku_b = old_nb_b >> fx.Int32(1)
        ikxdl_b = old_nb_b & fx.Int32(1)
        for mr in range_constexpr(BM // 32):
            row_local_b = fx.Int32(mr * 32) + m_lane_b
            gate_vs_b = [None] * 8
            up_vs_b = [None] * 8
            for ee in range_constexpr(8):
                gate_col_b = wave_grp_b * fx.Int32(32) + fx.Int32(8) * kk_b + fx.Int32(ee)
                gate_vs_b[ee] = acc_load(acc_idx(row_local_b, gate_col_b))
                up_vs_b[ee] = acc_load(acc_idx(row_local_b, fx.Int32(64) + gate_col_b))
            result_b = _gate_up_batch(
                gate_vs_b,
                up_vs_b,
                act=act,
                swiglu_limit=swiglu_limit,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
            )
            local_max_b = _fabs_f32(result_b[0])
            for ee in range_constexpr(1, 8):
                local_max_b = local_max_b.maximumf(_fabs_f32(result_b[ee]))
            lm_i_b = _inline_dpp_quad_amax(fx.Int32(_raw(local_max_b).bitcast(T.i32)))
            local_max_b = fx.Float32(_raw(lm_i_b).bitcast(T.f32))
            e8m0_b, qscale_b = _e8m0_from_amax(local_max_b)
            packed_i32_b = _raw(fx.Int32(0))
            qscale_raw_b = _raw(qscale_b)
            for w in range_constexpr(4):
                packed_i32_b = rocdl.cvt_scalef32_pk_fp4_f32(
                    T.i32,
                    packed_i32_b,
                    _raw(result_b[2 * w]),
                    _raw(result_b[2 * w + 1]),
                    qscale_raw_b,
                    w,
                )
            byte_pos_b = (
                n_block_idx * fx.Int32(BN_INT // 2)
                + wave_grp_b * fx.Int32(16)
                + kk_b * fx.Int32(4)
            )
            store_off_b = _layout_idx(aqout_layout_b, m_row_out + row_local_b, byte_pos_b)
            _scalar_store(aqout_tiles_b, store_off_b // fx.Int32(4), fx.Int32(packed_i32_b), fx.Int32)
            if kk_b == fx.Int32(0):
                chunk_b = m_block_out * fx.Int32(kSubBlocks) + fx.Int32(mr)
                dword_off_b = _layout_idx(
                    ascaleout_layout_b, chunk_b, ku_b, old_wg_b, m_lane_b % fx.Int32(16)
                )
                addr_b = (
                    dword_off_b * fx.Int32(4)
                    + ikxdl_b * fx.Int32(2)
                    + m_lane_b // fx.Int32(16)
                )
                _scalar_store(ascaleout_i8_tiles_b, addr_b, e8m0_b, fx.Int8)
    else:
        tx_i32 = fx.Int32(gpu.thread_id("x"))
        m_lane = tx_i32 // fx.Int32(16)
        n_lane = tx_i32 % fx.Int32(16)
        wave_grp = n_lane // fx.Int32(4)
        kk = n_lane % fx.Int32(4)

        aqout_layout = fx.make_layout((BM, K_G2_HALF), (K_G2_HALF, 1))
        # UniversalCopy has no nontemporal/cache-hint knob; dropped (perf-neutral).
        aqout_tiles = _global_scalar_tiles(arg_aqout, fx.Int32, 1 << 24)
        scales_per_mr = [None] * M_REPS

        for mr in range_constexpr(M_REPS):
            row_local = fx.Int32(mr * 16) + m_lane

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

            local_max = _fabs_f32(result[0])
            for ee in range_constexpr(1, 8):
                local_max = local_max.maximumf(_fabs_f32(result[ee]))
            lm_i = _inline_dpp_quad_amax(fx.Int32(_raw(local_max).bitcast(T.i32)))
            local_max = fx.Float32(_raw(lm_i).bitcast(T.f32))

            e8m0, qscale = _e8m0_from_amax(local_max)
            scales_per_mr[mr] = e8m0

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

            byte_pos = (
                n_block_idx * fx.Int32(BN_INT // 2)
                + wave_grp * fx.Int32(16)
                + kk * fx.Int32(4)
            )
            out_row = m_row_out + row_local
            store_off = _layout_idx(aqout_layout, out_row, byte_pos)
            _scalar_store(aqout_tiles, store_off // fx.Int32(4), packed, fx.Int32)

        # (chunk, ku, wave_grp, m_lane) -> dword index; shape is a placeholder.
        ascaleout_layout = fx.make_layout(
            (1 << 20, 2, 4, 16), (OUT_AS_PER_CHUNK_DW, 64, 16, 1)
        )
        ascaleout_i8_tiles = _global_scalar_tiles(arg_ascaleout, fx.Int8, 1 << 26)
        ascaleout_i16_tiles = _global_scalar_tiles(arg_ascaleout, fx.Int16, 1 << 25)
        if kk == fx.Int32(0):
            ku = n_block_idx >> fx.Int32(1)
            ikxdl = n_block_idx & fx.Int32(1)
            if const_expr(BM == 16):
                chunk = m_block_out
                dword_off = _layout_idx(ascaleout_layout, chunk, ku, wave_grp, m_lane)
                addr = dword_off * fx.Int32(4) + ikxdl * fx.Int32(2)
                _scalar_store(ascaleout_i8_tiles, addr, scales_per_mr[0], fx.Int8)
            else:
                for sub in range_constexpr(kSubBlocks):
                    chunk = m_block_out * fx.Int32(kSubBlocks) + fx.Int32(sub)
                    dword_off = _layout_idx(ascaleout_layout, chunk, ku, wave_grp, m_lane)
                    pair_i32 = scales_per_mr[sub * 2 + 0] | (
                        scales_per_mr[sub * 2 + 1] << fx.Int32(8)
                    )
                    addr = dword_off * fx.Int32(4) + ikxdl * fx.Int32(2)
                    _scalar_store(
                        ascaleout_i16_tiles, addr // fx.Int32(2), pair_i32, fx.Int16
                    )

    if const_expr(next_claim):
        nc_job = nc_valid.select(
            fx.Int32(nc_p) * fx.Int32(NUM_N_BLOCKS)
            + nc_j
            - nc_g * fx.Int32(NUM_N_BLOCKS),
            fx.Int32(-1),
        )
        if nc_tx0:
            fx.ptr_store(
                Vec.from_elements([nc_job], fx.Int32),
                fx.add_offset(
                    fx.recast_iter(fx.Int32, lds_raw_ptr),
                    next_claim_lds_dw,  # add_offset 按元素(i32)计
                ),
            )


# Preserve the private import spelling used by the fused wrappers while making
# the JIT function name itself layout-versioned.  FlyDSL keys its persistent
# cache by JIT function, not only by the emitted GPU symbol.
_gemm1_body = _gemm1_body_sc2


# 尾声 f32 累加器的 LDS 去 bank 冲突排布(MEGAMOE_TK_GMM1_EPI_SWZ=1,默认开,=0 关;只作用于 BN=256)。
# 旧排布 (BM,BN):(BN,1):写(每 lane 1 个 f32,16 列 x 4 行、行距 4)四行同 bank -> 64 bank 下 4 路;
# 读(ds_read_b128)按 CDNA4 文档的 4 相位(每相位 16 lane:T0-3,T12-15,T20-23,T24-27 等,
# 即两行 m、各两个 wave_grp)-> 2 路。
# 新排布:只把行距改成 BN+4 dword,不做列异或。行 r 相对 r+1 错开 4 bank(一个 16B 块):
#   写:行 4k 错开 16k bank,64 lane 覆盖 64 个不同 bank;
#   读:每相位两行各占偶数/奇数 16B 块,正好铺满 64 bank(CDNA3 32 bank 的 8 相位分组也无冲突)。
# 注意:按 wave_grp 做 16B 块异或在文档相位分组下会把读冲突重新引回 2 路(已用 selfcheck 模型核过)。
# 逻辑位置不变,数值逐位不变;地址仍是「lane 基址 + 编译期常量」。LDS 多 BM*4*4 字节(BM=128: +2KB)。
_EPI_ACC_PAD = 4

# ds_read_b128 的相位 lane 分组(nod-ai amdgpu_kernel_optimization_guide.md,实测所得)。
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


def _epi_swz_on(BN):
    return BN == 256 and __import__("os").environ.get("MEGAMOE_TK_GMM1_EPI_SWZ", "1") != "0"


_EPI_SWZ_CHECKED = set()


def _epi_swz_selfcheck(BM, BN):
    """编译期(纯 python 整数)核对:按 _gemm1_body_sc2 尾声的写/读地址公式、行距 BN+4,
    b128 读 16B 对齐,写(b32,64 lane 单相位 / 32 lane 两相位)与读(CDNA4 4 相位 / CDNA3 8 相位)无冲突。"""
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
                    ((lane // 16) * 4 + v) * stride + wave * (BN // 8) + (J // 2) * 16 + lane % 16 + up
                    for lane in range(64)
                ]
                assert _ways(addrs, 1, 64, (range(64),)) == 1, "epi store conflict 64"
                assert _ways(addrs, 1, 32, (range(32), range(32, 64))) == 1, "epi store conflict 32"
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
                assert _ways(addrs, 4, 64, _B128_PHASES_CDNA4) == 1, "epi load conflict cdna4"
                assert _ways(addrs, 4, 32, _B128_PHASES_CDNA3) == 1, "epi load conflict cdna3"
    _EPI_SWZ_CHECKED.add(key)


def _bm_constants(BM, BN, KH_TILE, K_TILES_TOTAL):
    # 生产流水 A 三级(DMA 提前两步)。LDS 不涨:A 三级+scale = BM*496B < 累加器复用区 BM*BN*4。
    kAStages = 3
    kSubBlocks = 1 if BM < 32 else BM // 32
    kMChunks = kmchunks_for(BM)
    s_aq_bytes = kAStages * BM * KH_TILE
    s_asc_bytes = kSubBlocks * K_TILES_TOTAL * 256
    if _epi_swz_on(BN):
        lds_acc_bytes = lds_acc_bytes_for(BM, BN + _EPI_ACC_PAD)
    else:
        lds_acc_bytes = lds_acc_bytes_for(BM, BN)
    lds_bytes = max(s_aq_bytes + s_asc_bytes, lds_acc_bytes)
    return kAStages, kSubBlocks, kMChunks, lds_bytes


def compile_gemm1_a4w4_port(
    BM=32,
    use_nt=True,
    inline_quant=False,
    *,
    D_HIDDEN,
    D_INTER,
    NE,
    TOPK,
    BN=256,
    BK=256,
    interleave=False,
    xcd_swizzle=0,
    act="silu",
    swiglu_limit=None,
    situ_beta=4.0,
    situ_linear_beta=25.0,
    persistent=False,
):
    if (BM, use_nt, inline_quant) not in {
        (32, True, False),
        (32, False, False),
        (64, False, False),
        (64, True, False),
        (128, False, False),
        (128, True, False),
        (16, True, True),
    }:
        raise AssertionError(
            f"unsupported gemm1 variant (BM={BM}, use_nt={use_nt}, inline_quant={inline_quant})"
        )
    if act not in ("silu", "swiglu", "situv2"):
        raise ValueError(
            f"activation must be silu, swiglu or situv2, got {act!r}"
        )
    if (
        not math.isfinite(float(situ_beta))
        or float(situ_beta) <= 0.0
        or not math.isfinite(float(situ_linear_beta))
        or float(situ_linear_beta) <= 0.0
    ):
        raise ValueError("SiTUv2 beta parameters must be positive")
    if swiglu_limit is not None and (
        math.isnan(float(swiglu_limit)) or float(swiglu_limit) <= 0.0
    ):
        raise ValueError("swiglu_limit must be positive or +inf when specified")
    if act == "situv2" and swiglu_limit is not None:
        raise ValueError("swiglu_limit does not apply to situv2")

    assert BN == 256 and BK == 256, f"only BN==BK==256 supported, got BN={BN} BK={BK}"
    KH_TILE = BK // 2
    _K = D_HIDDEN
    assert _K % BK == 0, f"D_HIDDEN (K) must be a multiple of {BK}, got {_K}"
    _INTER = D_INTER
    _N_OUT = n_out_for(_INTER)
    assert (
        _N_OUT % BN == 0
    ), f"2*D_INTER (N_OUT) must be a multiple of {BN}, got {_N_OUT}"
    _NE = NE
    _K_TILES_TOTAL = k_tiles_total_for(_K, BK)
    _NUM_N_BLOCKS = num_n_blocks_for(_N_OUT, BN)

    _, _, _, lds_bytes = _bm_constants(BM, BN, KH_TILE, _K_TILES_TOTAL)

    variant_tag = "iq" if inline_quant else ("nt" if use_nt else "cached")
    # Tag with H/INTER/NE so different shape specializations get distinct
    # kernel/smem symbols (so KIMI and non-KIMI instances never collide).
    gu_tag = "il" if interleave else "sep"
    name_suffix = (
        f"h{_K}_i{_INTER}_ne{_NE}_bm{BM}_{variant_tag}_{gu_tag}_"
        f"{MXFP4_SCALE_LAYOUT_TAG}"
    )

    def _float_tag(value):
        return (
            str(float(value))
            .replace("-", "m")
            .replace("+", "p")
            .replace(".", "p")
        )

    if act == "swiglu":
        limit = 7.0 if swiglu_limit is None else float(swiglu_limit)
        name_suffix += f"_swiglu_l{_float_tag(limit)}"
    elif act == "situv2":
        beta_tag = _float_tag(situ_beta)
        linear_tag = _float_tag(situ_linear_beta)
        name_suffix += f"_situv2_b{beta_tag}_lb{linear_tag}"
    elif swiglu_limit is not None:
        limit_tag = _float_tag(swiglu_limit)
        name_suffix += f"_silu_l{limit_tag}"
    if xcd_swizzle > 0:
        name_suffix += f"_xcd{xcd_swizzle}"
    if persistent:
        name_suffix += "_persistent"
    if _epi_swz_on(BN):
        name_suffix += "_esw"

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, lds_bytes, 16]

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
        lds_raw_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        tx_i32 = fx.Int32(tx)
        bx_i32 = fx.Int32(bx)
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))
        cumsum0 = _global_i32_at(arg_cumsum, fx.Int32(0))
        total_m_blocks = cumsum0 // fx.Int32(BM)
        bound = total_m_blocks * fx.Int32(_NUM_N_BLOCKS)

        _NXCD = 8
        _xq = _udiv(bound, _NXCD)
        _xr = _umod(bound, _NXCD)
        _SW = xcd_swizzle

        def _xcd(pid):
            xc = _umod(pid, _NXCD)
            wgid = (
                xc * _xq
                + fx.Int32(arith.minsi(_raw(xc), _raw(_xr)))
                + _udiv(pid, _NXCD)
            )
            _ng = fx.Int32(_SW * _NUM_N_BLOCKS)
            group_id = wgid // _ng
            first_pid_m = group_id * fx.Int32(_SW)
            remaining_m = total_m_blocks - first_pid_m
            group_size_m = fx.Int32(arith.minsi(_raw(remaining_m), _raw(fx.Int32(_SW))))
            wig = wgid % _ng
            m_block = first_pid_m + (wig % group_size_m)
            n_block = wig // group_size_m
            return m_block * fx.Int32(_NUM_N_BLOCKS) + n_block

        def _run_tile(pid):
            if const_expr(_SW > 0):
                _tile = _xcd(pid)
            else:
                _tile = pid
            _gemm1_body(
                lds_raw_ptr,
                arg_aq,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_mind,
                arg_aqout,
                arg_ascaleout,
                arg_hidden,
                _tile,
                lane,
                wave,
                use_nt,
                i32_ntok,
                total_m_blocks,
                fx.Int64(0),
                fx.Int64(0),
                fx.Int64(0),
                fx.Int32(0),
                fx.Int32(0),
                BM=BM,
                BN=BN,
                BK=BK,
                inline_quant=inline_quant,
                K=_K,
                N_OUT=_N_OUT,
                NE=_NE,
                interleave=interleave,
                act=act,
                swiglu_limit=swiglu_limit,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
            )

        if const_expr(persistent):
            if bx_i32 < bound:
                _run_tile(bx_i32)
            for iv in range(
                bx_i32 + fx.Int32(gpu.grid_dim.x),
                bound,
                gpu.grid_dim.x,
            ):
                # The A4 body has no final CTA barrier. Separate its prior
                # epilogue LDS reads from the next tile's accumulator staging.
                gpu.barrier()
                _run_tile(fx.Int32(iv))
        else:
            if bx_i32 < bound:
                _run_tile(bx_i32)

    @flyc.jit
    def launch_gemm1_sc2(
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
        grid_x = fx.Int64(i32_grid)
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

    return launch_gemm1_sc2
