# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""FlyDSL port of aiter PR #3470 ``gemm2_a4w4`` (MXFP4 MoE down-proj, gfx950).

Parametrized over the launch_atomic specialization:
    ``launch_atomic<MAX_M=655360, NE=385, K=512, N_OUT=7168, TOPK=9, BM, kUseNT>``
Supported instances (atomic path):
  * BM=32, kUseNT=false -> ``...TOPK9_BM32_ATOMIC``        (compile_gemm2_a4w4_port(BM=32))
  * BM=16, kUseNT=true  -> ``...TOPK9_BM16_ATOMIC_NT``     (compile_gemm2_a4w4_port(BM=16, use_nt=True))

The port mirrors gemm2_a4w4.cuh's atomic path instruction-for-instruction:
  * 4 ``make.buffer.rsrc`` (A_q, A_scale, B_q, B_scale) with exact num_bytes.
  * A -> LDS via ``raw.ptr.buffer.load.lds`` (2 slots), swizzled (BM16: 2 waves).
  * B / scales via ``raw.ptr.buffer.load.v4i32`` / ``.i32`` (NT: B aux=2).
  * ``s_waitcnt vmcnt(23/22)`` + ``s_barrier`` cross-wave fences.
  * K=512 = 2 K-tiles fully unrolled; 32 (BM32) / 16 (BM16) MFMAs.
  * atomic bf16 epilog: LDS cshuffle -> ``global.atomic.fadd.v2bf16`` * topk weight.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl._mlir.dialects import memref as memref_dialect
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

# -- shape constants (BM-independent) -----------------------------------------
MAX_M = 655360
NE = 385
K = 512  # gemm2 contraction = inter_dim
N_OUT = 7168  # gemm2 output dim = model_dim
TOPK = 9

BN = 256
BK = 256
K_HALF = K // 2  # 256 packed-fp4 bytes along K
KH_TILE = BK // 2  # 128 packed bytes per K-tile
NUM_N_BLOCKS = N_OUT // 256  # 28
K_TILES_TOTAL = K // BK  # 2
kStages = 2
NUM_CU = 256  # persistent-grid workgroup count (matches gemm2_a4w4.cuh NUM_CU).
# Measured: 1 workgroup/CU (grid == NUM_CU) is optimal; over-subscribing the
# persistent grid only adds L2/memory-queue contention (the kernel has enough
# memory-level parallelism at 1 wg/CU), so the grid is capped at NUM_CU.
# Explicit hand-tuned vmcnt for the nonatomic K-loop ds_read fence (inline asm).
# None -> let the backend choose (rocdl.barrier); an int forces s_waitcnt vmcnt(N).
_NONATOMIC_KLOOP_VMCNT = 16

# scale-layout consts (mirror gemm2_a4w4.cuh)
kBS_c_k1 = (K // 32) // 4 // 2  # 2
kBS_stride_k0_dw = 64
kBS_stride_n0_dw = kBS_c_k1 * 64  # 128
kBS_c_n1 = N_OUT // 16 // 2  # 224
kBS_per_expert_dw = kBS_c_n1 * kBS_stride_n0_dw  # 28672
kAS_c_k1 = (K // 32) // 4 // 2  # 2
kAS_per_chunk_dw = kAS_c_k1 * 64  # 128


# -- shape-parametrized sizes (K=512 fixed; NE/N_OUT/MAX_M vary per instance) --
# N_OUT must be a multiple of 256 (BN); the mxfp4 gemm2 hard-requires K==512.
def num_n_blocks_for(n_out):
    return n_out // 256


def kbs_per_expert_dw_for(n_out):
    return (n_out // 16 // 2) * kBS_stride_n0_dw  # kBS_c_n1 * kBS_stride_n0_dw


def aq_bytes_for(max_m):
    return max_m * K_HALF


def bq_bytes_for(ne, n_out):
    return ne * n_out * K_HALF


def bscale_bytes_for(ne, n_out):
    return ne * kbs_per_expert_dw_for(n_out) * 4


def ascale_bytes(BM, max_m=MAX_M):
    """A_scale buffer-resource num_bytes (kAS_bound_div=BM, atomic mode):
    (max_m/BM) * kAS_per_chunk_dw * 4."""
    return (max_m // BM) * kAS_per_chunk_dw * 4


# Back-compat KIMI defaults (NE=385, N_OUT=7168, MAX_M=655360).
AQ_BYTES = aq_bytes_for(MAX_M)  # 167772160
BQ_BYTES = bq_bytes_for(NE, N_OUT)  # 706478080
BSCALE_BYTES = bscale_bytes_for(NE, N_OUT)  # 44154880


def saq_slot_bytes(BM):
    return BM * KH_TILE  # s_Aq[slot] = BM rows x KH_TILE bytes


def lds_bytes(BM):
    return BM * BN * 4  # union max: lds_acc[BM*BN] f32 (>= 2*saq_slot_bytes)


def kmchunks(BM):
    return 1 if BM == 16 else BM // 16


def tiling(BM):
    """A-load tiling: (n_load_waves, rows_per_wave, kSubBlocks). Each loading
    wave streams ``rows_per_wave`` A rows split into ``kSubBlocks`` 8-row chunks.
    BM16 -> (2,8,1); BM32 -> (4,8,1); BM64 -> (4,16,2)."""
    n_load_waves = min(4, BM // 8)
    rows_per_wave = BM // n_load_waves
    return n_load_waves, rows_per_wave, rows_per_wave // 8


# Back-compat module constants (BM32 defaults; the test imports BM/ASCALE_BYTES).
BM = 32
kMChunks = kmchunks(BM)
SAQ_SLOT_BYTES = saq_slot_bytes(BM)
LDS_ACC_FLOATS = BM * BN
LDS_BYTES = lds_bytes(BM)
ASCALE_BYTES = ascale_bytes(BM)


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


def _s_barrier_bare():
    """Bare ``s_barrier`` (no surrounding memory fence), matching HIP's K-loop
    ``__builtin_amdgcn_s_barrier()`` cross-wave fence after the vmcnt wait."""
    llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="s_barrier",
        constraints="",
        has_side_effects=True,
    )


def _global_base_ptr1(arg):
    """One ptr<1> base for a global tensor (single memref->ptr conversion)."""
    base_idx = buffer_ops.extract_base_index(arg, address_space=1)
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), _raw(fx.Int64(base_idx)))


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


def _issue_a_load_lds(aq_rsrc, saq, slot, kt, car, lane, slot_bytes, lds_row):
    """Issue one A->LDS chunk load (one wave's 8 rows for one (K-tile=slot,
    M-subblock)) via ``raw.ptr.buffer.load.lds`` into s_Aq[slot][lds_row]. ``car``
    is the cached actual row, ``lds_row = wave*rows_per_wave + sub*8``. Caller
    loops slot (K-tile) outer, sub inner (matching HIP), and gates on
    ``wave < n_load_waves``. Side-effecting -> not sunk past the cumsum branch."""
    lane_mod_8 = lane % fx.Int32(8)
    mask = _lds_swizzle_mask(lds_row + (lane // fx.Int32(8)))
    voffset = ((lane_mod_8 * fx.Int32(16)) ^ mask) + car * fx.Int32(K // 2)
    base_i32 = fx.Int32(memref_dialect.extract_aligned_pointer_as_index(saq.get()))
    off_i32 = fx.Int32(slot * slot_bytes) + lds_row * fx.Int32(KH_TILE)
    lds_ptr = _lds_ptr3(base_i32, off_i32)
    rocdl.raw_ptr_buffer_load_lds(
        aq_rsrc,
        lds_ptr,
        fx.Int32(16),
        voffset,
        fx.Int32(kt * KH_TILE),
        fx.Int32(0),
        fx.Int32(0),
    )


def compile_gemm2_a4w4_port(
    BM=32, use_nt=False, NE=NE, N_OUT=N_OUT, MAX_M=MAX_M, epilog="atomic"
):
    """Compile the gemm2 a4w4 port for a given shape / specialization / epilog.

    Shape params (K=512 is fixed by the mxfp4 gemm2 kernel; TOPK is upstream and
    not used in the gemm2 body): NE (experts), N_OUT (model_dim, %256), MAX_M.
    Specialization: BM in {16,32,64} (atomic) or 128 (nonatomic), kUseNT.
    epilog:
      "atomic"          -> LDS cshuffle + global_atomic_fadd x sorted_weights (BM16/32/64)
      "nonatomic"       -> flat per-sorted-row bf16 write, no atomic (BM128); a
                           separate scatter_reduce sums topk afterwards
      "nonatomic_mxfp4" -> flat per-sorted-row fp4 (q + e8m0 scale) write (BM128)
    """
    _atomic = epilog == "atomic"
    # The BM128 non-atomic bf16 epilog uses a hybrid persistent grid (one-shot for
    # small launches, NUM_CU workgroups grid-striding for large) -- its cheap,
    # high-occupancy epilog lets cross-tile pipelining hide the next tile's loads.
    # The atomic and mxfp4out paths stay one-shot: atomic is already faster than
    # HIP, and mxfp4out's heavy LDS-cshuffle epilog (130KB LDS, 1 block/CU) has no
    # spare occupancy to benefit from persistence.
    _persistent = epilog == "nonatomic"
    kmchunks(BM)
    _slot_bytes = saq_slot_bytes(BM)
    BM * BN
    # atomic / mxfp4 epilog reuses LDS for the cshuffle (BM*BN f32); nonatomic
    # bf16 writes direct, so only s_Aq (kStages slots) is needed.
    _lds_bytes = lds_bytes(BM) if epilog != "nonatomic" else kStages * _slot_bytes
    _aq_bytes = aq_bytes_for(MAX_M)
    _num_n_blocks = num_n_blocks_for(N_OUT)
    _n_load_waves, _rows_per_wave, _kSubBlocks = tiling(
        BM
    )  # BM16/32:1, BM64:2, BM128:4
    _epi_tag = {
        "atomic": "atomic",
        "nonatomic": "nonatomic",
        "nonatomic_mxfp4": "nonatomic_mxfp4",
    }[epilog]
    _tag = f"ne{NE}_h{N_OUT}_bm{BM}{'_nt' if use_nt else ''}_{_epi_tag}"
    _name = f"gemm2_a4w4_port_{_tag}"

    allocator = SmemAllocator(
        None, arch="gfx950", global_sym_name=f"gemm2port_smem_{_tag}"
    )
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + _lds_bytes

    @flyc.kernel(name=_name, known_block_size=[256, 1, 1])
    def gemm2_kernel(
        arg_aq: fx.Tensor,
        arg_ascale: fx.Tensor,
        arg_bq: fx.Tensor,
        arg_bscale: fx.Tensor,
        arg_eids: fx.Tensor,
        arg_cumsum: fx.Tensor,
        arg_stids: fx.Tensor,
        arg_sweights: fx.Tensor,
        i32_M: fx.Int32,
        arg_out: fx.Tensor,
        arg_out_scale: fx.Tensor,  # flat_out_scale (mxfp4 epilog only; dummy otherwise)
    ):
        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        tx_i32 = arith.index_cast(T.i32, tx)
        bx_i32 = arith.index_cast(T.i32, bx)

        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))  # wave == wave_n

        aq_rsrc = buffer_ops.create_buffer_resource(
            arg_aq, max_size=False, num_records_bytes=fx.Index(_aq_bytes)
        )
        saq = SmemPtr(
            allocator.get_base(), lds_off, T.i8, shape=(kStages * _slot_bytes,)
        )

        # Issue A->LDS for a tile's m_block. raw.ptr.buffer.load.lds is
        # side-effecting (writes LDS). Loop K-tile (slot) outer, M-subblock (sub)
        # inner, matching HIP.
        def _issue_all_a_loads(m_row0):
            for slot in range_constexpr(kStages):  # slot == K-tile index for preload
                for sub in range_constexpr(_kSubBlocks):
                    lds_row = wave * fx.Int32(_rows_per_wave) + fx.Int32(sub * 8)
                    car = m_row0 + lds_row + (lane // fx.Int32(8))
                    _issue_a_load_lds(
                        aq_rsrc, saq, slot, slot, car, lane, _slot_bytes, lds_row
                    )

        def _run_tile(tile_i32):
            _gemm2_body(
                allocator,
                lds_off,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_stids,
                arg_sweights,
                i32_M,
                arg_out,
                arg_out_scale,
                tile_i32,
                lane,
                wave,
                BM,
                use_nt,
                NE,
                N_OUT,
                MAX_M,
                epilog,
            )

        # total_m_blocks = cumsum[0] / BM ; bound = total_m_blocks * NUM_N_BLOCKS
        if const_expr(_persistent):
            # Persistent grid (BM128 non-atomic): NUM_CU workgroups grid-stride over
            # tiles. Peel tile 0 (keeps its sched_barrier so the A->LDS issue is
            # pinned early -- matters for one-shot-sized launches); remaining tiles
            # run without it so the compiler overlaps each tile's loads with the
            # previous tile's epilog.
            cumsum0 = llvm.load(T.i32, _global_ptr1(arg_cumsum, fx.Int32(0)))
            total_m_blocks = cumsum0 // fx.Int32(BM)
            bound = total_m_blocks * fx.Int32(_num_n_blocks)
            ub = arith.index_cast(T.index, _raw(bound))
            step = arith.index_cast(T.index, _raw(gpu.grid_dim.x))
            grid_nb = arith.index_cast(T.i32, _raw(gpu.grid_dim.x))

            # XCD-grouped interleave (mirrors xcd_remap.hpp swizzle=-1): remap the
            # raw persistent index -> wgid so consecutive indices spread across the
            # 8 XCDs and same-XCD tiles reuse B in that XCD's private L2 slice. Only
            # constant divisors (cheap). HIP's plain NONATOMIC baseline omits this.
            _NXCD = 8
            _xq = bound // fx.Int32(_NXCD)
            _xr = bound % fx.Int32(_NXCD)

            def _xcd(pid):
                xc = pid % fx.Int32(_NXCD)
                return (
                    xc * _xq
                    + fx.Int32(arith.minsi(_raw(xc), _raw(_xr)))
                    + pid // fx.Int32(_NXCD)
                )

            peel_if = scf.IfOp(
                arith.cmpi(arith.CmpIPredicate.slt, bx_i32, bound), [], has_else=False
            )
            with ir.InsertionPoint(peel_if.then_block):
                tile = _xcd(bx_i32)
                _issue_all_a_loads((tile // fx.Int32(_num_n_blocks)) * fx.Int32(BM))
                rocdl.sched_barrier(0)
                _run_tile(tile)
                scf.YieldOp([])

            saq._view_cache = None
            start2 = arith.index_cast(T.index, _raw(bx_i32 + grid_nb))
            for_op = scf.ForOp(start2, ub, step)
            with ir.InsertionPoint(for_op.body):
                wu = arith.index_cast(T.i32, for_op.induction_variable)
                # iter-boundary fence: prev tile's LDS reads must finish before
                # this tile overwrites the s_Aq slots (persistent-grid reuse race).
                rocdl.barrier()
                saq._view_cache = None
                tile = _xcd(wu)
                _issue_all_a_loads((tile // fx.Int32(_num_n_blocks)) * fx.Int32(BM))
                _run_tile(tile)
                scf.YieldOp([])
        else:
            # One-shot grid (atomic): issue A->LDS BEFORE the cumsum load so the
            # A->LDS HBM latency overlaps the cumsum load + bound check (A->LDS
            # depends only on bx/lane). Only the first n_load_waves hold A rows
            # (BM16: waves 0,1), so gate on wave < n_load_waves.
            m_row0 = (bx_i32 // fx.Int32(_num_n_blocks)) * fx.Int32(BM)
            if const_expr(_n_load_waves < 4):  # BM16: only waves 0,1 hold A rows
                a_pred = arith.cmpi(
                    arith.CmpIPredicate.slt, wave, fx.Int32(_n_load_waves)
                )
                a_if = scf.IfOp(a_pred, [], has_else=False)
                with ir.InsertionPoint(a_if.then_block):
                    _issue_all_a_loads(m_row0)
                    scf.YieldOp([])
            else:
                _issue_all_a_loads(m_row0)
            rocdl.sched_barrier(0)

            cumsum0 = llvm.load(T.i32, _global_ptr1(arg_cumsum, fx.Int32(0)))
            total_m_blocks = cumsum0 // fx.Int32(BM)
            bound = total_m_blocks * fx.Int32(_num_n_blocks)

            in_range = arith.cmpi(arith.CmpIPredicate.slt, bx_i32, bound)
            if_op = scf.IfOp(in_range, [], has_else=False)
            with ir.InsertionPoint(if_op.then_block):
                _run_tile(bx_i32)
                scf.YieldOp([])

    @flyc.jit
    def launch_gemm2(
        arg_aq: fx.Tensor,
        arg_ascale: fx.Tensor,
        arg_bq: fx.Tensor,
        arg_bscale: fx.Tensor,
        arg_eids: fx.Tensor,
        arg_cumsum: fx.Tensor,
        arg_stids: fx.Tensor,
        arg_sweights: fx.Tensor,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        arg_out: fx.Tensor,
        arg_out_scale: fx.Tensor,  # flat_out_scale (mxfp4 epilog only; dummy otherwise)
        stream: fx.Stream,
    ):
        from flydsl.compiler.kernel_function import CompilationContext

        ctx = CompilationContext.get_current()
        allocator.finalized = False
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        if const_expr(_persistent):
            # Hybrid grid. The persistent kernel grid-strides over tiles, so a grid
            # of total_work runs ~1 tile/wg (== one-shot: more wavefronts in flight,
            # best latency hiding for small launches), while a grid of NUM_CU
            # amortizes the prologue + pipelines tiles for large launches. Persist
            # only past ~4*NUM_CU tiles, where tiles/wg is high enough to pay off.
            tw = i32_max_m_blocks * fx.Int32(_num_n_blocks)
            persist = arith.cmpi(
                arith.CmpIPredicate.sgt, _raw(tw), _raw(fx.Int32(NUM_CU * 4))
            )
            grid_i32 = arith.select(persist, _raw(fx.Int32(NUM_CU)), _raw(tw))
            grid_x = arith.index_cast(T.index, grid_i32)
        else:
            grid_x = arith.index_cast(T.index, i32_max_m_blocks) * fx.Index(
                _num_n_blocks
            )
        gemm2_kernel(
            arg_aq,
            arg_ascale,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_cumsum,
            arg_stids,
            arg_sweights,
            i32_M,
            arg_out,
            arg_out_scale,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm2


def _gemm2_body(
    allocator,
    lds_off,
    arg_ascale,
    arg_bq,
    arg_bscale,
    arg_eids,
    arg_stids,
    arg_sweights,
    i32_M,
    arg_out,
    arg_out_scale,
    bx_i32,
    lane,
    wave,
    BM,
    use_nt,
    NE,
    N_OUT,
    MAX_M,
    epilog,
):
    _atomic = epilog == "atomic"
    _kMChunks = kmchunks(BM)
    _slot_bytes = saq_slot_bytes(BM)
    _lds_acc_floats = BM * BN
    _ascale_bytes = ascale_bytes(BM, MAX_M)
    _bq_bytes = bq_bytes_for(NE, N_OUT)
    _bscale_bytes = bscale_bytes_for(NE, N_OUT)
    _kbs_per_expert_dw = kbs_per_expert_dw_for(N_OUT)
    _num_n_blocks = num_n_blocks_for(N_OUT)
    _kSubBlocks = tiling(BM)[2]  # BM16/32: 1, BM64: 2, BM128: 4
    b_aux = 2 if use_nt else 0  # NT: B_q loads carry aux=2 (non-temporal hint)

    # block -> (m_block_idx, n_block_idx) ; e = sorted_expert_ids[m_block_idx]
    n_block_idx = bx_i32 % fx.Int32(_num_n_blocks)
    m_block_idx = bx_i32 // fx.Int32(_num_n_blocks)
    e = llvm.load(T.i32, _global_ptr1(arg_eids, m_block_idx * fx.Int32(4)))
    e = rocdl.readfirstlane(T.i32, e)
    m_row = m_block_idx * fx.Int32(BM)

    # -- buffer resources (exact num_bytes) ----------------------------------
    # (A_q resource + A->LDS loads are issued by the kernel before the branch.)
    ascale_rsrc = buffer_ops.create_buffer_resource(
        arg_ascale, max_size=False, num_records_bytes=fx.Index(_ascale_bytes)
    )
    bq_rsrc = buffer_ops.create_buffer_resource(
        arg_bq, max_size=False, num_records_bytes=fx.Index(_bq_bytes)
    )
    bscale_rsrc = buffer_ops.create_buffer_resource(
        arg_bscale, max_size=False, num_records_bytes=fx.Index(_bscale_bytes)
    )

    # -- LDS base ------------------------------------------------------------
    lds_base = allocator.get_base()
    saq = SmemPtr(lds_base, lds_off, T.i8, shape=(kStages * _slot_bytes,))
    # lds_acc (cshuffle scratch) only used by atomic / mxfp4 epilogs.
    lds_acc = (
        SmemPtr(lds_base, lds_off, T.f32, shape=(_lds_acc_floats,))
        if epilog != "nonatomic"
        else None
    )

    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)

    # -- s_base computations (readfirstlane'd, uniform per wave) --------------
    b_load_s_base = []
    for j in range_constexpr(4):
        v = (
            e * fx.Int32(N_OUT)
            + n_block_idx * fx.Int32(BN)
            + wave * fx.Int32(BN // 4)
            + fx.Int32(j * 16)
        ) * fx.Int32(K_HALF)
        b_load_s_base.append(rocdl.readfirstlane(T.i32, v))

    mni_base = n_block_idx * fx.Int32(BN // 16 // 2) + wave * fx.Int32(BN // 64 // 2)
    b_scale_s_base = []
    for mw in range_constexpr(2):
        v = (
            e * fx.Int32(_kbs_per_expert_dw)
            + (mni_base + fx.Int32(mw)) * fx.Int32(kBS_stride_n0_dw)
        ) * fx.Int32(4)
        b_scale_s_base.append(rocdl.readfirstlane(T.i32, v))

    # a_scale_s_base[sub]: chunk_base = m_row / (16 if BM==16 else 32); sub in kSubBlocks
    chunk_base = m_row // fx.Int32(16 if BM == 16 else 32)
    a_scale_s_base = [
        rocdl.readfirstlane(
            T.i32,
            (chunk_base + fx.Int32(sub)) * fx.Int32(kAS_per_chunk_dw) * fx.Int32(4),
        )
        for sub in range_constexpr(_kSubBlocks)
    ]

    # -- a_scale (atomic) : v_voff = ((lane/16)*16 + lane%16)*4 ; per sub-block -
    v_voff_scale = ((lane_div_16 * fx.Int32(16)) + lane_mod_16) * fx.Int32(4)
    a_scale_v = [[None, None] for _ in range(_kSubBlocks)]
    for sub in range_constexpr(_kSubBlocks):
        for ku in range_constexpr(2):
            a_scale_v[sub][ku] = buffer_ops.buffer_load(
                ascale_rsrc,
                (v_voff_scale + fx.Int32(ku * 256)) // fx.Int32(4),
                vec_width=1,
                dtype=T.i32,
                soffset_bytes=a_scale_s_base[sub],
            )

    # -- b_scale ku0/ku1 ------------------------------------------------------
    b_scale_v = [[None, None], [None, None]]
    for ku in range_constexpr(2):
        imm = ku * (kBS_stride_k0_dw * 4)
        for mw in range_constexpr(2):
            v = buffer_ops.buffer_load(
                bscale_rsrc,
                (v_voff_scale + fx.Int32(imm)) // fx.Int32(4),
                vec_width=1,
                dtype=T.i32,
                soffset_bytes=b_scale_s_base[mw],
            )
            b_scale_v[ku][mw] = v

    # -- B loads (NT: cache_modifier=2) : v_voff = (lane/16)*256 + (lane%16)*16 + K_BYTE
    b = [[[None, None] for _ in range(4)] for _ in range(2)]
    for kc in range_constexpr(2):
        k_byte = kc * 2048
        v_voff_b = (
            (lane_div_16 * fx.Int32(256))
            + (lane_mod_16 * fx.Int32(16))
            + fx.Int32(k_byte)
        )
        for j in range_constexpr(4):
            for half in range_constexpr(2):
                imm = half * 1024
                frag = buffer_ops.buffer_load(
                    bq_rsrc,
                    (v_voff_b + fx.Int32(imm)) // fx.Int32(4),
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=b_aux,
                    soffset_bytes=b_load_s_base[j],
                )
                b[kc][j][half] = Vec(frag)

    # -- ds_read(slot) -> a[i][k] (i32x4) ; i in [0,kMChunks) -----------------
    def issue_a_ds_read(slot):
        lane_row = lane_mod_16
        lane_col = lane_div_16 * fx.Int32(16)
        mask = _lds_swizzle_mask(lane_row)
        base_ptr = _lds_base_ptr3(saq.get())
        a = [[None, None] for _ in range(_kMChunks)]
        for k in range_constexpr(2):
            lds_col = (lane_col + fx.Int32(k * 64)) ^ mask
            for i in range_constexpr(_kMChunks):
                lds_row = lane_row + fx.Int32(i * 16)
                byte_off = (
                    fx.Int32(slot * _slot_bytes) + lds_row * fx.Int32(KH_TILE) + lds_col
                )
                a[i][k] = llvm.load(
                    T.vec(4, T.i32), _gep3(base_ptr, byte_off)
                )  # ds_read_b128
        return a

    # -- MFMA cluster (per M-subblock; BM16: kMChunks=1 -> i0 only) -----------
    # opselA encodes (16-row half, K-half) = 0,1,2,3; sub picks accm[sub*2+{0,1}]
    # and the per-subblock A scale a_scale_sub[sub]. BM64 has kSubBlocks=2.
    mfma_res_ty = T.f32x4
    zero4 = Vec.filled(4, 0.0, fx.Float32)
    accm = [[None, None, None, None] for _ in range(_kMChunks)]

    def mfma_cluster(slot, a, a_scale_sub, b_scale_slot, init):
        for J in range_constexpr(4):
            mni = J // 2
            in_b = J % 2
            sb = b_scale_slot[mni]
            b_J0 = b[slot][J][0]
            b_J1 = b[slot][J][1]
            for sub in range_constexpr(_kSubBlocks):
                sa = a_scale_sub[sub]
                i0 = sub * 2
                i1 = sub * 2 + 1
                if const_expr(init):
                    accm[i0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_res_ty, [a[i0][0], b_J0, zero4, 4, 4, 0, sa, 0 + in_b, sb]
                    )
                    if const_expr(_kMChunks > 1):
                        accm[i1][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            mfma_res_ty,
                            [a[i1][0], b_J0, zero4, 4, 4, 1, sa, 0 + in_b, sb],
                        )
                else:
                    accm[i0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_res_ty,
                        [a[i0][0], b_J0, accm[i0][J], 4, 4, 0, sa, 0 + in_b, sb],
                    )
                    if const_expr(_kMChunks > 1):
                        accm[i1][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            mfma_res_ty,
                            [a[i1][0], b_J0, accm[i1][J], 4, 4, 1, sa, 0 + in_b, sb],
                        )
                accm[i0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    mfma_res_ty,
                    [a[i0][1], b_J1, accm[i0][J], 4, 4, 2, sa, 2 + in_b, sb],
                )
                if const_expr(_kMChunks > 1):
                    accm[i1][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_res_ty,
                        [a[i1][1], b_J1, accm[i1][J], 4, 4, 3, sa, 2 + in_b, sb],
                    )

    # -- K loop (2 stages, fully unrolled) ------------------------------------
    for S in range_constexpr(kStages):
        kt = K_TILES_TOTAL - kStages + S
        slot = kt % kStages
        if const_expr(_atomic):
            # atomic: explicit vmcnt-tuned cross-wave fence (loads land before ds_read).
            vmcnt = 23 if S == 0 else 22
            llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string=f"s_waitcnt vmcnt({vmcnt})",
                constraints="",
                has_side_effects=True,
            )
            _s_barrier_bare()
        elif const_expr(_NONATOMIC_KLOOP_VMCNT is None):
            # nonatomic: plain barrier (== HIP __syncthreads); the backend inserts
            # the buffer_load_lds->ds_read vmcnt wait.
            rocdl.barrier()
        else:
            # nonatomic: explicit hand-tuned fence (inline asm) -- replaces the
            # backend's auto waitcnt before the ds_read with a less-conservative one.
            _v = _NONATOMIC_KLOOP_VMCNT
            llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string=f"s_waitcnt vmcnt({_v}) lgkmcnt(0)",
                constraints="",
                has_side_effects=True,
            )
            _s_barrier_bare()
        a = issue_a_ds_read(slot)
        a_scale_sub = [a_scale_v[sub][kt] for sub in range_constexpr(_kSubBlocks)]
        mfma_cluster(slot, a, a_scale_sub, b_scale_v[slot], init=(S == 0))

    # -- epilog ---------------------------------------------------------------
    saq._view_cache = None
    if epilog == "nonatomic":
        # flat per-sorted-row bf16 write (no LDS, no atomic, no weight); a
        # separate scatter_reduce sums the TOPK contributions per token.
        out_base = _global_base_ptr1(arg_out)
        _flat_bf16_epilog(
            accm, out_base, m_row, n_block_idx, wave, lane, N_OUT, _kMChunks
        )
    elif epilog == "nonatomic_mxfp4":
        # flat per-sorted-row fp4 (packed q + e8m0 scale) write.
        out_q_base = _global_base_ptr1(arg_out)
        out_scale_base = _global_base_ptr1(arg_out_scale)
        tid_i32 = arith.index_cast(T.i32, gpu.thread_id("x"))
        _flat_mxfp4_epilog(
            accm,
            out_q_base,
            out_scale_base,
            m_row,
            n_block_idx,
            wave,
            lane,
            tid_i32,
            N_OUT,
            lds_acc,
            _kMChunks,
        )
    else:
        lds_acc._view_cache = None
        _atomic_bf16_epilog(
            lds_acc,
            accm,
            arg_out,
            arg_stids,
            arg_sweights,
            m_row,
            n_block_idx,
            wave,
            lane,
            i32_M,
            BM,
            N_OUT,
        )


def _flat_bf16_epilog(accm, out_base, m_row, n_block_idx, wave, lane, N_OUT, kMChunks):
    """Nonatomic flat epilog (BM128): write each computed sorted-row element
    directly to flat_out[(m_row+row)*N_OUT + gn] as bf16 -- no LDS cshuffle, no
    atomic, no sorted_weights (a later scatter_reduce sums the TOPK rows per
    token). i64 element index (rows can exceed the i32 byte range)."""
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    for i in range_constexpr(kMChunks):
        for J in range_constexpr(4):
            gn = (
                n_block_idx * fx.Int32(BN)
                + wave * fx.Int32(BN // 4)
                + fx.Int32(J * 16)
                + lane_mod_16
            )
            vec = Vec(accm[i][J])
            for v in range_constexpr(4):
                row = m_row + fx.Int32(i * 16) + lane_div_16 * fx.Int32(4) + fx.Int32(v)
                elem = fx.Int64(row) * fx.Int64(N_OUT) + fx.Int64(
                    gn
                )  # i64 element index
                bf = Vec.from_elements([vec[v]], fx.Float32).to(fx.BFloat16)
                llvm.StoreOp(_raw(bf), _gep1(out_base, elem * fx.Int64(2)))


def _flat_mxfp4_epilog(
    accm,
    out_q_base,
    out_scale_base,
    m_row,
    n_block_idx,
    wave,
    lane,
    tid_i32,
    N_OUT,
    lds_acc,
    kMChunks,
):
    """Nonatomic MXFP4 epilog (BM128): cshuffle accm -> lds_acc, then per 32-elem
    block quantize to fp4 -- e8m0 block scale via DPP quad-amax -- and write packed
    fp4 (flat_out_q, u32 = 8 fp4) + e8m0 scale (flat_out_scale). Mirrors
    apply_mxfp4_flat_epilog_bm128."""
    lds_base = _lds_base_ptr3(lds_acc.get())
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    for i in range_constexpr(kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for J in range_constexpr(4):
            col = wave * fx.Int32(BN // 4) + fx.Int32(J * 16) + lane_mod_16
            vec = Vec(accm[i][J])
            for v in range_constexpr(4):
                idx = (row_base + fx.Int32(v)) * fx.Int32(BN) + col
                llvm.StoreOp(_raw(vec[v]), _gep3(lds_base, idx * fx.Int32(4)))
    rocdl.barrier()

    NBLK = BN // 32  # 8
    m_lane = tid_i32 // fx.Int32(16)
    n_lane = tid_i32 % fx.Int32(16)
    wave_grp = n_lane // fx.Int32(4)
    kk = n_lane % fx.Int32(4)
    i7fff = _raw(fx.Int32(0x7FFFFFFF))
    for mr in range_constexpr(kMChunks):  # BM/16
        row_local = fx.Int32(mr * 16) + m_lane
        out_row = m_row + row_local
        for half in range_constexpr(NBLK // 4):  # 2
            group = wave_grp + fx.Int32(half * 4)
            col0 = group * fx.Int32(32) + kk * fx.Int32(8)
            r = []
            for e in range_constexpr(8):
                idx = row_local * fx.Int32(BN) + col0 + fx.Int32(e)
                r.append(llvm.load(T.f32, _gep3(lds_base, idx * fx.Int32(4))))
            # block amax over |r[0..7]| (positive-float bits) -> bf16-bits
            maxb = arith.andi(arith.bitcast(T.i32, _raw(r[0])), i7fff)
            for e in range_constexpr(1, 8):
                maxb = arith.maxui(
                    maxb, arith.andi(arith.bitcast(T.i32, _raw(r[e])), i7fff)
                )
            amax = arith.shrui(maxb, _raw(fx.Int32(16)))
            # DPP quad-amax (reduce across the 4 kk-lanes of the block)
            s1 = rocdl.update_dpp(T.i32, amax, amax, 0xB1, 0xF, 0xF, True)
            a = arith.maxui(amax, s1)
            s2 = rocdl.update_dpp(T.i32, a, a, 0x4E, 0xF, 0xF, True)
            amax_dpp = arith.maxui(a, s2)
            # encode e8m0: bexp = ((amax<<16)+0x200000>>23)&0xFF ; e8 = clamp(bexp-2,0,254)
            f32b = arith.shli(amax_dpp, _raw(fx.Int32(16)))
            bexp = arith.andi(
                arith.shrui(
                    arith.addi(f32b, _raw(fx.Int32(0x200000))), _raw(fx.Int32(23))
                ),
                _raw(fx.Int32(0xFF)),
            )
            e8 = arith.minsi(
                _raw(fx.Int32(254)),
                arith.maxsi(_raw(fx.Int32(0)), arith.subi(bexp, _raw(fx.Int32(2)))),
            )
            qscale = arith.bitcast(T.f32, arith.shli(e8, _raw(fx.Int32(23))))
            packed = _raw(fx.Int32(0))
            packed = rocdl.cvt_scalef32_pk_fp4_f32(
                T.i32, packed, _raw(r[0]), _raw(r[1]), qscale, 0
            )
            packed = rocdl.cvt_scalef32_pk_fp4_f32(
                T.i32, packed, _raw(r[2]), _raw(r[3]), qscale, 1
            )
            packed = rocdl.cvt_scalef32_pk_fp4_f32(
                T.i32, packed, _raw(r[4]), _raw(r[5]), qscale, 2
            )
            packed = rocdl.cvt_scalef32_pk_fp4_f32(
                T.i32, packed, _raw(r[6]), _raw(r[7]), qscale, 3
            )
            global_col = n_block_idx * fx.Int32(BN) + col0
            q_byte = fx.Int64(out_row) * fx.Int64(N_OUT // 2) + fx.Int64(
                global_col // fx.Int32(2)
            )
            llvm.StoreOp(packed, _gep1(out_q_base, q_byte), nontemporal=True)
            blk = n_block_idx * fx.Int32(NBLK) + group
            s_byte = fx.Int64(out_row) * fx.Int64(N_OUT // 32) + fx.Int64(blk)
            pred = arith.cmpi(arith.CmpIPredicate.eq, _raw(kk), _raw(fx.Int32(0)))
            if_op = scf.IfOp(pred, [], has_else=False)
            with ir.InsertionPoint(if_op.then_block):
                llvm.StoreOp(arith.trunci(T.i8, e8), _gep1(out_scale_base, s_byte))
                scf.YieldOp([])


def _atomic_bf16_epilog(
    lds_acc,
    accm,
    arg_out,
    arg_stids,
    arg_sweights,
    m_row,
    n_block_idx,
    wave,
    lane,
    i32_M,
    BM,
    N_OUT,
):
    _kMChunks = kmchunks(BM)
    M_REPS = BM // 8  # BM32: 4, BM16: 2
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lds_base = _lds_base_ptr3(lds_acc.get())

    tx_i32 = arith.index_cast(T.i32, gpu.thread_id("x"))
    m_lane = tx_i32 // fx.Int32(32)
    n_lane = tx_i32 % fx.Int32(32)
    col_start = n_lane * fx.Int32(2)
    stids_base = _global_base_ptr1(arg_stids)
    sweights_base = _global_base_ptr1(arg_sweights)
    out_base = _global_base_ptr1(arg_out)

    # Prefetch sorted_token_ids / sorted_weights BEFORE the cshuffle stores and
    # both LDS barriers (invariant => freely hoistable), overlapping their global
    # latency with the store + barriers instead of exposing it in the atomic loop.
    packed = []
    weight = []
    for mr in range_constexpr(M_REPS):
        sorted_pos = m_row + fx.Int32(mr * 8) + m_lane
        packed.append(
            llvm.load(
                T.i32, _gep1(stids_base, sorted_pos * fx.Int32(4)), invariant=True
            )
        )
        weight.append(
            llvm.load(
                T.f32, _gep1(sweights_base, sorted_pos * fx.Int32(4)), invariant=True
            )
        )

    # pre-store fence+barrier (HIP run_one __syncthreads() before the epilog).
    rocdl.barrier()

    # write accm -> lds_acc cshuffle (scalar f32 stores, as HIP does)
    for i in range_constexpr(_kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for J in range_constexpr(4):
            col = wave * fx.Int32(64) + fx.Int32(J * 16) + lane_mod_16
            vec = Vec(accm[i][J])
            for v in range_constexpr(4):
                idx = (row_base + fx.Int32(v)) * fx.Int32(BN) + col
                llvm.StoreOp(_raw(vec[v]), _gep3(lds_base, idx * fx.Int32(4)))

    rocdl.barrier()

    # read back + weighted atomic add (token_id / weight prefetched above)
    for mr in range_constexpr(M_REPS):
        row_in_block = fx.Int32(mr * 8) + m_lane
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        valid = arith.cmpi(arith.CmpIPredicate.slt, token_id, i32_M)
        if_op = scf.IfOp(valid, [], has_else=False)
        with ir.InsertionPoint(if_op.then_block):
            row_base_addr = (
                token_id * fx.Int32(N_OUT) + n_block_idx * fx.Int32(BN) + col_start
            )
            for s in range_constexpr(4):
                # adjacent ee=0,1 are contiguous -> one <2xf32> load (as HIP vectorizes)
                idx0 = row_in_block * fx.Int32(BN) + col_start + fx.Int32(s * 64)
                v2 = Vec(
                    llvm.load(T.vec(2, T.f32), _gep3(lds_base, idx0 * fx.Int32(4)))
                )
                pk = Vec.from_elements(
                    [v2[0] * weight[mr], v2[1] * weight[mr]], fx.Float32
                ).to(fx.BFloat16)
                off = (row_base_addr + fx.Int32(s * 64)) * fx.Int32(
                    2
                )  # bf16 byte offset
                out_ptr = _gep1(out_base, off)
                llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.fadd,
                    out_ptr,
                    _raw(pk),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                )
            scf.YieldOp([])
