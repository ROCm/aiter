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
from flydsl.expr.typing import T, Vector as Vec, Float4E2M1FN, Float32, Int32
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
from .common import (
    _e8m0_from_amax,
    bscale_bytes_for,
    kbs_per_expert_dw_for,
    kbs_stride_n0_dw_for,
)
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
    lds_vec_load,
    num_n_blocks_for,
)
from ..tensor_shim import buf_copy_load, ptr_buf_tensor


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
    fx.copy_atom_call(atom, r, fx.slice(tiles, (None, idx)))


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
    K,
    N_OUT,
    NE,
    act="silu",
    swiglu_limit=None,
    situ_beta=4.0,
    situ_linear_beta=25.0,
    next_claim=False,
    next_claim_lds_dw=0,
    next_claim_lead=2,
    aqout_row_base=False,
):
    # aqout_row_base: the h1 output exceeds the i32 byte range (mtpr > 4096), so
    # the m-block's row offset goes into the 64-bit base and the store index only
    # covers the BM rows of this block. Off: the original single-base addressing.
    KH_TILE = BK // 2
    K_HALF = k_half_for(K)
    K_TILES_TOTAL = k_tiles_total_for(K, BK)
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
    b_aux = 2
    M_REPS = BM // 16
    # 每 wave 的 16 列 N 子块数:4(gate/up 各两个)。
    NJ = BN // 64
    assert BN == 256, f"GMM1 BN must be 256, got {BN}"
    assert BM in (32, 64, 128), f"GMM1 BM must be 32, 64 or 128, got {BM}"

    # next_claim(stage1 early_local_gmm 专用):在本 job 的第
    # K_TILES_TOTAL-lead 步开头由 tx0 发下一个 job 的领位原子(arg_nc_head),结果留在 VGPR
    # 里跨过最后 lead 步;K 循环后的 barrier 之后查组表 arg_nc_list,epilogue 末尾把映射好的
    # 物理 job 号(越界 = -1)写进 LDS 信箱 dword next_claim_lds_dw。调用方在 job 后的 barrier
    # 之后直接读信箱,省掉逐 job 的「barrier + 原子 + 系统 load + LDS 广播 + barrier + 组表 load」。
    # 信箱必须在 A 环/scale/累加器复用区之外(调用方多分配 16B)。
    # 原子是 monotonic、只在 wave0 发:它排在已计数的 vmem 之后,_wait_lds_barrier 的 vmcnt
    # 只会变松不会变错(in-order 返回,A(kt+1) 之后的条数只多不少)。
    if const_expr(next_claim):
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
    # h1 is written at the physical tile the m-block reads from.
    m_block_out = m_block_idx
    m_row_out = m_block_out * fx.Int32(BM)

    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lane_div_8 = lane // fx.Int32(8)
    lane_mod_8 = lane % fx.Int32(8)

    aq_num_records = fx.Int64(i32_ntok * fx.Int32(K_HALF))

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

    # LDS alias scopes。A 的 LDS DMA 计入 vmcnt;
    # LLVM 在任何 LDS 读写前,对没有 alias.scope 的访问保守地等全部在途的 LDS DMA,
    # 于是每步 A 的 ds_read 前被插 vmcnt(0)(ATT:GMM1 41% 周期),A 两步预取形同虚设。
    # 每个 A 槽一个 scope、A-scale 区一个 scope,读写都带上,LLVM 只等同槽的 DMA。
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




    # Union LDS region [s_aq | s_asc], reused as lds_acc (f32 accumulator) in
    # the epilogue. s_aq/lds_acc start at lds_raw_ptr; s_asc follows at
    # +kAStages*BM*KH_TILE.

    cached_actual_row = []
    for sub in range_constexpr(kSubBlocks):
        idx = m_row + wave * fx.Int32(BM // 4) + fx.Int32(sub * 8) + lane_div_8
        cached_actual_row.append(_global_i32_at(arg_mind, idx))

    # -- b_load_s_base[j] (HIP 412-416), readfirstlane'd uniform per wave ------
    N0_HALF = N_OUT // 32
    b_load_s_base = []
    for j in range_constexpr(NJ):
        tile_il = n_block_idx * fx.Int32(BN // 16) + wave * fx.Int32(NJ) + fx.Int32(j)
        g = tile_il & fx.Int32(1)
        n0 = tile_il >> fx.Int32(1)
        col = (g * fx.Int32(N0_HALF) + n0) * fx.Int32(16)
        v = (e * fx.Int32(N_OUT) + col) * fx.Int32(K_HALF)
        b_load_s_base.append(rocdl.readfirstlane(T.i32, v))

    # -- b_scale_s_base / _hi (HIP 418-429) -----------------------------------
    # B-scale 一个字覆盖相邻两个 16 列 n0:本 wave 的两个 n0 正好一对。
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
    # s_aq as flat i32, divided into 4-element (128-bit) and 1-element tiles.
    s_aq_i32_flat = fx.make_view(
        fx.recast_iter(fx.Int32, lds_raw_ptr),
        fx.make_layout(kAStages * BM * KH_TILE // 4, 1),
    )
    s_aq_i32x4_tiles = fx.logical_divide(s_aq_i32_flat, fx.make_layout(4, 1))
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
                a[i][k] = Vec(
                    llvm.LoadOp(
                        Vec.make_type(4, fx.Int32),
                        _lds_ptr3(_lds_base_i32, off),
                        alignment=16,
                        **_scope_attrs(slot),
                    ).result
                )
        return a



    def issue_a_scale_ds_read(kt):
        out = []
        for sub in range_constexpr(kSubBlocks):
            lds_dw = (
                fx.Int32(sub * kAS_per_chunk_dw)
                + fx.Int32(kt * 64)
                + lane_div_16 * fx.Int32(16)
                + lane_mod_16
            )
            out.append(
                llvm.LoadOp(
                    T.i32,
                    _lds_ptr3(_asc_base_i32, lds_dw * fx.Int32(4)),
                    alignment=4,
                    **_scope_attrs(kAStages),
                ).result
            )
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
                llvm.StoreOp(
                    _raw(v),
                    _lds_ptr3(_asc_base_i32, lds_dw * fx.Int32(4)),
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

    mfma_ty = T.f32x4
    zero4 = fx.Vector.filled(4, 0.0, fx.Float32)

    def mfma_cluster(b_slot, a, a_scale, bs_slot, J, init):
        mni = J % 2
        in_b = J // 2
        sb = bs_slot[mni]
        bJ0, bJ1 = b_slot[J][0], b_slot[J][1]
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

    # 裸 s_barrier(仅 scoped 路径)。gpu.barrier() 降成
    # fence release(workgroup, LDS) + s_barrier + fence acquire;SIMemoryLegalizer 在该
    # release fence 上发 S_WAITCNT_lds_direct,SIInsertWaitcnts 把它换成「等全部在途 LDS DMA」
    # (通用 LDS 槽,不看 alias scope),并入前面那条显式 s_waitcnt:vmcnt(24) 被收紧成
    # vmcnt(10),即 A(kt+2) 被迫提前一步落地(ISA 28 处全是 vmcnt(10) lgkmcnt(0))。
    # 这里跨 wave 可见性已由显式 vmcnt(N)+lgkmcnt(0) 保证,fence 不提供额外必需的顺序。

    def _wait_lds_barrier(vmcnt):
        # 同 mega_moe/gemm_util.wait_lds_barrier:lgkmcnt(0) 等本 wave 的 ds_read 读完,
        # vmcnt(N) 等本步开头发出的 A DMA 落地(之后发的 N 条 B 预取继续飞),再 barrier。
        rocdl.s_waitcnt((vmcnt & 0xF) | ((vmcnt & 0x30) << 10) | (7 << 4))
        rocdl.s_barrier()
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

    gs_loads = gather_a_scale_issue()
    for K_C in range_constexpr(kStages):
        if const_expr(K_C < K_TILES_TOTAL):
            rocdl.sched_barrier(0)
            issue_a_load_lds(K_C % 3, K_C)
            rocdl.sched_barrier(0)
            for j in range_constexpr(NJ):
                issue_b_load_j(b[K_C], K_C, j)
            issue_b_scale_load(b_scale_v[K_C], K_C)
            rocdl.sched_barrier(0)
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

    gpu.barrier()

    if const_expr(next_claim):
        # 下一个 job 的组表项在 epilogue 期间飞;原子此时早已返回(K 循环末尾 vmem 已排空)。
        nc_valid = nc_raw < i32_nc_bound
        nc_j = nc_raw + i32_nc_off
        nc_g = nc_valid.select(nc_j // fx.Int32(NUM_N_BLOCKS), fx.Int32(0))
        nc_p = _global_i32_at(arg_nc_list, nc_g)

    # lds_acc reuses the s_aq region (offset 0) as an f32 accumulator.
    # 累加器行距 BN+4(不做列异或),消掉尾声 LDS 的 bank 冲突;_bm_constants 已按 BN+4 定 LDS。
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

    tx_i32 = fx.Int32(gpu.thread_id("x"))
    m_lane = tx_i32 // fx.Int32(16)
    n_lane = tx_i32 % fx.Int32(16)
    wave_grp = n_lane // fx.Int32(4)
    kk = n_lane % fx.Int32(4)

    aqout_layout = fx.make_layout((BM, K_G2_HALF), (K_G2_HALF, 1))
    # UniversalCopy has no nontemporal/cache-hint knob; dropped (perf-neutral).
    if const_expr(aqout_row_base):
        aqout_tiles = _global_scalar_tiles(
            fx.Int64(arg_aqout) + fx.Int64(m_row_out) * fx.Int64(K_G2_HALF),
            fx.Int32, 1 << 24)
        aqout_row0 = fx.Int32(0)
    else:
        aqout_tiles = _global_scalar_tiles(arg_aqout, fx.Int32, 1 << 24)
        aqout_row0 = m_row_out
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
        out_row = aqout_row0 + row_local
        store_off = _layout_idx(aqout_layout, out_row, byte_pos)
        _scalar_store(aqout_tiles, store_off // fx.Int32(4), packed, fx.Int32)

    # (chunk, ku, wave_grp, m_lane) -> dword index; shape is a placeholder.
    ascaleout_layout = fx.make_layout(
        (1 << 20, 2, 4, 16), (OUT_AS_PER_CHUNK_DW, 64, 16, 1)
    )
    ascaleout_i16_tiles = _global_scalar_tiles(arg_ascaleout, fx.Int16, 1 << 25)
    if kk == fx.Int32(0):
        ku = n_block_idx >> fx.Int32(1)
        ikxdl = n_block_idx & fx.Int32(1)
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


# 尾声 f32 累加器的 LDS 去 bank 冲突排布(只作用于 BN=256)。
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


def scale_mma_atoms():
    """16 (opselA,opselB) scaled-MFMA atoms; A and B are fp4."""
    return {
        (osa, osb): fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(
                16, 16, 128, Float4E2M1FN, Float4E2M1FN, opsel_a=osa, opsel_b=osb
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
):
    """Load one A tile through a tile-local descriptor so allocations may span the 4 GiB buffer ABI."""
    lanes_per_row = KH_TILE_A // 16  # 8 (fp4)
    rows_per_call = 64 // lanes_per_row  # 8 (fp4)
    a_lane_row = lane // lanes_per_row
    rows_per_wave = BM // 4  # rows each wave loads (BM32: 8)
    assert rows_per_wave >= rows_per_call, "every wave loads whole row groups (BM >= 32)"
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
        mask = _lds_swizzle_mask(lds_row + a_lane_row)
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
):
    """GEMM2 of one (m_block, n_block): fp4 A (LDS DMA, kStages+1 slots), fp4 B
    streamed through a 2-stage register pipeline with the A-scale carried alongside.

    explicit_a_row: A/A-scale rows live at this (physical) row instead of the
    sorted m_row; expert id and the returned m_row stay on the sorted row.
    """
    assert aStages == kStages + 1
    assert BK == 256, "one 256-K E8M0 scale word per K-tile"
    assert BM >= 32
    kMChunks = BM // 16  # 16-row MFMA row-groups
    kHalves = BK // 128  # 16x16x128 MFMA K-steps per K-tile
    numAccN = (BN // 4) // 16  # 16-column MFMA subblocks per wave
    nPairs = max(1, numAccN // 2)  # one B-scale per two 16-column subblocks
    kScaleSubBlocks = kMChunks // 2
    a_pack = 2  # fp4
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
    e = rocdl.readfirstlane(T.i32, eids_ptr[m_block_idx])
    m_row = m_block_idx * BM
    a_row = m_row if explicit_a_row is None else fx.Int32(explicit_a_row)

    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16

    s_aq_base = lds_base_i32
    mma_atoms = scale_mma_atoms()

    # A activation: global->LDS DMA (issue_a_load_lds), then LDS->reg ds-read (issue_a_ds_read).
    A_NDW = 4  # fp4 -> i32<4:1>
    a_frags = [
        [fx.make_rmem_tensor(A_NDW, Int32) for _ in range_constexpr(kHalves)]
        for _ in range_constexpr(kMChunks)
    ]

    def issue_a_load_lds(slot, kt):
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
        )

    def issue_a_ds_read(slot):
        # A ds-read for one slot into a_frags: fp8 -> i32<8:1> (two 128-K halves), fp4 -> i32<4:1>.
        for k in range_constexpr(kHalves):
            for i in range_constexpr(kMChunks):
                lds_row = lane_mod_16 + i * 16
                row_off = fx.Int32(slot * slot_bytes) + lds_row * KH_TILE_A
                mask = _lds_swizzle_mask(lane_mod_16)
                lds_col = (lane_div_16 * 16 + k * 64) ^ mask
                vec = lds_vec_load(
                    s_aq_base,
                    row_off + lds_col,
                    Vec.make_type(4, fx.Int32),
                    fx.Int32,
                    align=16,
                )
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
            (e * N_OUT_rt + n_block_idx * BN + wave * (BN // 4) + j * 16) * KH4,
        )
        for j in range_constexpr(numAccN)
    ]

    mni_base = n_block_idx * (BN // 16 // 2) + wave * (BN // 64 // 2)
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
        fx.make_fragment_like(sc_frag_tmpl)
        for _ in range_constexpr(kScaleSubBlocks)
    ]
    nxt_saf = [
        fx.make_fragment_like(sc_frag_tmpl)
        for _ in range_constexpr(kScaleSubBlocks)
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
        prefetch_next_b(kt_rt)
        gpu.barrier()
        nxt_a = kt_rt + fx.Int32(kStages)
        issue_a_ds_read(kt_rt % fx.Int32(aStages))
        if nxt_a < K_TILES_RT:
            issue_a_load_lds(nxt_a % fx.Int32(aStages), nxt_a)
        # A-scale from the prefetch carry (g2_ascale_pf) or loaded synchronously here.
        sa = [
            Vec(cur_saf[sub].load())[0]
            for sub in range_constexpr(kScaleSubBlocks)
        ]
        # Fence the MFMA chain from the B vmem loads (next-tile loads ride ahead of compute).
        rocdl.sched_barrier(0)
        rocdl.s_setprio(1)
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
