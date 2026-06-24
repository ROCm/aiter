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
  * ``gpu.barrier`` cross-wave fences (the compiler inserts the vmcnt).
  * K=512 = 2 K-tiles fully unrolled; 32 (BM32) / 16 (BM16) MFMAs.
  * atomic bf16 epilog: LDS cshuffle -> ``global.atomic.fadd.v2bf16`` * topk weight.

The contraction dim K(=inter_dim=D_INTER) is parametrized via ``compile_gemm2_a4w4_port(D_INTER=...)``.
KIMI/DSR (D_INTER=512 -> K_TILES_TOTAL=2) keeps the original fully-unrolled, all-
tiles-preloaded fast path byte-for-byte. For D_INTER>512 (K_TILES_TOTAL>2; e.g.
minimax 768->3, 2048->8) the kernel switches to a streaming double-buffered K-loop
(prologue preloads kStages=2 tiles, a main loop streams tiles [kStages,
K_TILES_TOTAL) while MFMA-ing, a drain processes the last kStages) -- mirroring
mxfp4_gemm1.py. Constraint: D_INTER % BK(256) == 0 (covers 512/768/1024/1536/
2048/3072); inter_dim not divisible by 256 (e.g. 384/192) is NOT supported by this
BK=256 kernel.
"""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import memref as memref_dialect
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

# -- shape constants (BM-independent) -----------------------------------------
MAX_M = 655360
NE = 385
K = 512  # gemm2 contraction = inter_dim (DEFAULT / KIMI; per-shape value comes
#          from the compile arg D_INTER. All K-derived sizes are computed from the
#          arg via the *_for() helpers below; these module globals are the KIMI
#          defaults so existing importers (and the test) keep working unchanged.)
N_OUT = 7168  # default gemm2 output dim = model_dim (per-shape value comes from the compile arg)

BN = 256
BK = 256
KH_TILE = BK // 2  # 128 packed bytes per K-tile (BK-derived, K-independent)
kStages = 2
NUM_CU = 256  # persistent-grid workgroup count (matches gemm2_a4w4.cuh NUM_CU).
# XCD remap mode for the persistent (cshuffle/nonatomic) grid, read at build time.
#   <=0 : step-1-only contiguous-per-XCD (mirrors xcd_remap.hpp swizzle=-1; the
#         original port behavior).
#   N>0 : flydsl-style 2-step remap with M-major group size N (mirrors
#         xcd_remap.hpp positive swizzle) -- same-XCD tiles share M-row panels so
#         the stationary operand stays hot in that XCD's private L2 slice.
PORT_XCD_SWIZZLE = int(os.environ.get("PORT_XCD_SWIZZLE", "-1"))
# Measured: 1 workgroup/CU (grid == NUM_CU) is optimal; over-subscribing the
# persistent grid only adds L2/memory-queue contention (the kernel has enough
# memory-level parallelism at 1 wg/CU), so the grid is capped at NUM_CU.

# scale-layout consts (mirror gemm2_a4w4.cuh). K-independent stride:
kBS_stride_k0_dw = 64


# -- K-derived sizes (parametrized over the contraction dim K = inter_dim = D_INTER)
# K MUST be a multiple of BK(256). The *_for() helpers mirror gemm1's pattern so the
# KIMI default (K=512) path is byte-for-byte identical to the old module globals.
def k_half_for(k):
    return k // 2  # packed-fp4 bytes along K (KIMI: 256)


def k_tiles_total_for(k):
    return k // BK  # KIMI: 2


def kunroll_for(k):
    # streaming main-loop trip count (K_TILES_TOTAL>2 path): prologue issues kStages
    # tiles, the main loop processes tiles [0, kUnroll) while streaming tiles
    # [kStages, kStages+kUnroll), and the drain processes the final kStages tiles.
    # Full gap-free coverage requires EXACTLY kUnroll = K_TILES_TOTAL - kStages.
    return k_tiles_total_for(k) - kStages


def kbs_c_k1_for(k):
    return (k // 32) // 4 // 2  # KIMI: 2


def kbs_stride_n0_dw_for(k):
    return kbs_c_k1_for(k) * 64  # KIMI: 128


def kas_c_k1_for(k):
    return (k // 32) // 4 // 2  # KIMI: 2


def kas_per_chunk_dw_for(k):
    return kas_c_k1_for(k) * 64  # KIMI: 128


# -- shape-parametrized sizes (NE/N_OUT/MAX_M/K vary per instance) --
# N_OUT must be a multiple of 256 (BN).
def num_n_blocks_for(n_out):
    return n_out // 256


def kbs_per_expert_dw_for(n_out, k=K):
    # depends on BOTH N_OUT (via N_OUT//16//2) AND K (via kBS_stride_n0_dw).
    return (n_out // 16 // 2) * kbs_stride_n0_dw_for(k)


def aq_bytes_for(max_m, k=K):
    return max_m * k_half_for(k)


def bq_bytes_for(ne, n_out, k=K):
    return ne * n_out * k_half_for(k)


def bscale_bytes_for(ne, n_out, k=K):
    return ne * kbs_per_expert_dw_for(n_out, k) * 4


def ascale_bytes(BM, max_m=MAX_M, k=K):
    """A_scale buffer-resource num_bytes.

    The A_scale read stride is one ``kAS_per_chunk_dw*4`` chunk per ``chunk_div``
    A rows, where ``chunk_div = 16 if BM==16 else 32`` (see the
    ``chunk_base = m_row // chunk_div`` addressing in ``_gemm2_body``). The
    resource bound MUST divide ``max_m`` by that read granularity -- NOT by BM.

    For BM in {64,128} the chunk granularity (32) is smaller than BM, so the
    old ``max_m // BM`` under-sized the resource by ``BM/32x`` and clamped
    A_scale reads (-> 0) for every sorted row past ``max_m*32/BM``. That
    silently zeroed the gemm2 output of the trailing sorted rows -- i.e. the
    high-id experts incl. the always-on shared expert -- so large-M MoE
    (BM128 nonatomic / mxfp4out) lost accuracy (e.g. KIMI cos 0.68 @ M=16384,
    0.05 @ M=32768) while smaller M that stayed under the bound looked fine."""
    chunk_div = 16 if const_expr(BM == 16) else 32
    return (max_m // chunk_div) * kas_per_chunk_dw_for(k) * 4


# KIMI default kept as a module global (used as a param default below). All other
# K-derived sizes are computed per-shape from the compile arg via the *_for(K)
# helpers (local _K_* in compile_gemm2_a4w4_port), so they need no module globals.
K_HALF = k_half_for(K)  # 256


def saq_slot_bytes(BM):
    return BM * KH_TILE  # s_Aq[slot] = BM rows x KH_TILE bytes


def lds_bytes(BM):
    return BM * BN * 4  # union max: lds_acc[BM*BN] f32 (>= 2*saq_slot_bytes)


def kmchunks(BM):
    return 1 if const_expr(BM == 16) else BM // 16


def tiling(BM):
    """A-load tiling: (n_load_waves, rows_per_wave, kSubBlocks). Each loading
    wave streams ``rows_per_wave`` A rows split into ``kSubBlocks`` 8-row chunks.
    BM16 -> (2,8,1); BM32 -> (4,8,1); BM64 -> (4,16,2)."""
    n_load_waves = min(4, BM // 8)
    rows_per_wave = BM // n_load_waves
    return n_load_waves, rows_per_wave, rows_per_wave // 8


_PTR3 = "!llvm.ptr<3>"


def _raw(v):
    """Unwrap an fx value to a raw ir.Value for raw llvm/arith ops."""
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _udiv(a, c):
    cc = fx.Int32(c) if isinstance(c, int) else c
    return fx.Int32(arith.divui(_raw(a), _raw(cc)))


def _umod(a, c):
    cc = fx.Int32(c) if isinstance(c, int) else c
    return fx.Int32(arith.remui(_raw(a), _raw(cc)))


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
    """One ptr<1> base from a raw i64 device address.

    Global args are passed as bare ``data_ptr()`` (fx.Int64) rather than full
    memref descriptors (ported from gemm1): the kernel only needs base pointers
    (it assumes contiguity + derives sizes from compile-time consts), so raw i64
    addresses pack contiguously into kernargs -> coalesced s_load prologue."""
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


def _issue_a_load_lds(
    aq_rsrc, saq, slot, kt, car, lane, slot_bytes, lds_row, k_half=K_HALF
):
    """Issue one A->LDS chunk load (one wave's 8 rows for one (K-tile=slot,
    M-subblock)) via ``raw.ptr.buffer.load.lds`` into s_Aq[slot][lds_row]. ``car``
    is the cached actual row, ``lds_row = wave*rows_per_wave + sub*8``. ``k_half``
    is the per-row A byte stride (= K//2, parametrized over the contraction dim).
    Caller loops slot (K-tile) outer, sub inner (matching HIP), and gates on
    ``wave < n_load_waves``. Side-effecting -> not sunk past the cumsum branch."""
    lane_mod_8 = lane % fx.Int32(8)
    mask = _lds_swizzle_mask(lds_row + (lane // fx.Int32(8)))
    voffset = ((lane_mod_8 * fx.Int32(16)) ^ mask) + car * fx.Int32(k_half)
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
    BM=32,
    use_nt=False,
    NE=NE,
    N_OUT=N_OUT,
    MAX_M=MAX_M,
    epilog="atomic",
    D_INTER=K,
    D_INTER_REAL=None,
):
    """Compile the gemm2 a4w4 port for a given shape / specialization / epilog.

    Shape params (TOPK is upstream and not used in the gemm2 body):
      NE (experts), N_OUT (model_dim, %256), MAX_M,
      D_INTER (= contraction dim K = inter_dim, %BK(256); KIMI/DSR default 512).
    For D_INTER==512 (K_TILES_TOTAL==2) the original fully-unrolled all-tiles-
    preloaded fast path is used (byte-for-byte identical). For D_INTER>512 the
    streaming double-buffered K-loop (prologue/main/drain) is used.
    Specialization: BM in {16,32,64} (atomic) or 128 (nonatomic), kUseNT.
    epilog:
      "atomic"          -> LDS cshuffle + global_atomic_fadd x sorted_weights (BM16/32/64)
      "nonatomic"       -> flat per-sorted-row bf16 write, no atomic (BM128); a
                           separate scatter_reduce sums topk afterwards
      "nonatomic_mxfp4" -> flat per-sorted-row fp4 (q + e8m0 scale) write (BM128)
    """
    print(
        f"[PORT-FLYDSL-GEMM2] compile_gemm2_a4w4_port ENTERED "
        f"BM={BM} use_nt={use_nt} NE={NE} N_OUT={N_OUT} epilog={epilog} D_INTER={D_INTER}",
        flush=True,
    )
    # K = contraction dim = inter_dim. Parametrized; defaults to KIMI/DSR's 512.
    # For a non-256-aligned shard (dsv4 TP8: real inter=384) the caller pads weights/
    # inter/scales to D_INTER (the next %256, e.g. 512) and passes D_INTER_REAL=384;
    # the K-loop tiles over the padded _K but skips loading/MFMA-ing the pad-tail
    # 128-K half-steps (k >= _K_REAL) -> the real ~25% weight bandwidth is not read.
    _K = D_INTER
    _K_REAL = D_INTER if D_INTER_REAL is None else D_INTER_REAL
    assert _K % BK == 0, (
        f"D_INTER (gemm2 contraction K = inter_dim) must be a multiple of {BK}, "
        f"got {_K}; inter_dim not divisible by {BK} (e.g. 384/192) is not "
        f"supported by this BK={BK} kernel"
    )
    assert (
        _K_REAL % 128 == 0 and 0 < _K_REAL <= _K
    ), f"D_INTER_REAL={_K_REAL} must be a multiple of 128 and in (0, {_K}]"
    _K_HALF = k_half_for(_K)
    _K_TILES_TOTAL = k_tiles_total_for(_K)
    # The BM128 non-atomic epilogs use a hybrid persistent grid (one-shot for small
    # launches, NUM_CU workgroups grid-striding for large). Even though the heavy
    # mxfp4-out epilog is LDS-bound to 1 block/CU (no occupancy slack), persistence
    # still wins at large M: it collapses the launch from tens of thousands of
    # one-shot WGs (each re-running the prologue) to NUM_CU WGs, and lets the
    # compiler overlap the next tile's A->LDS HBM loads with the current tile's
    # epilog. This mirrors the prebuilt HIP gemm2 (256 persistent WGs); profiling
    # showed the old one-shot mxfp4out path ~8-9% slower than HIP at M>=16384.
    _persistent = epilog in ("nonatomic", "nonatomic_mxfp4")
    _slot_bytes = saq_slot_bytes(BM)
    # Number of LDS A-slots (s_Aq stages). The K_TILES_TOTAL==2 fast path preloads
    # all 2 tiles into 2 slots (double buffer). The streaming K_TILES_TOTAL>2 path
    # uses 3 slots (triple buffer) so the per-iteration read-slot (tile kt) and
    # write-slot (tile kt+kStages, streamed in) never alias -- the read tile maps to
    # kt%3 and the streamed tile to (kt+2)%3, which differ for all kt. (gemm1 uses
    # the same kAStages=3 triple-buffer for its streaming non-128 path.)
    _aStages = kStages if _K_TILES_TOTAL <= kStages else 3
    # atomic / mxfp4 epilog reuses LDS for the cshuffle (BM*BN f32); nonatomic
    # bf16 writes direct, so only s_Aq (_aStages slots) is needed. nonatomic_cshuffle
    # cshuffles in 64-row passes -> only min(BM,64)*BN f32 (BM128 -> 64KB not 128KB).
    _acc_rows = min(BM, 64) if epilog == "nonatomic_cshuffle" else BM
    _lds_bytes = (
        max(lds_bytes(_acc_rows), _aStages * _slot_bytes)
        if epilog != "nonatomic"
        else _aStages * _slot_bytes
    )
    _aq_bytes = aq_bytes_for(MAX_M, _K)
    _num_n_blocks = num_n_blocks_for(N_OUT)
    _n_load_waves, _rows_per_wave, _kSubBlocks = tiling(
        BM
    )  # BM16/32:1, BM64:2, BM128:4
    _epi_tag = {
        "atomic": "atomic",
        "nonatomic": "nonatomic",
        "nonatomic_mxfp4": "nonatomic_mxfp4",
        "nonatomic_cshuffle": "nonatomic_cshuffle",
    }[epilog]
    # Tag with the inter (K) so specializations with different contraction dims get
    # distinct kernel/smem symbols (KIMI i512 keeps the original numeric layout).
    _rtag = "" if _K_REAL == _K else f"r{_K_REAL}"
    _tag = f"ne{NE}_h{N_OUT}_i{_K}{_rtag}_bm{BM}{'_nt' if use_nt else ''}_{_epi_tag}"
    _name = f"gemm2_a4w4_port_{_tag}"

    allocator = SmemAllocator(
        None, arch="gfx950", global_sym_name=f"gemm2port_smem_{_tag}"
    )
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + _lds_bytes

    @flyc.kernel(name=_name, known_block_size=[256, 1, 1])
    def gemm2_kernel(
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        i32_M: fx.Int32,
        arg_out: fx.Int64,
        arg_out_scale: fx.Int64,  # flat_out_scale (mxfp4 epilog only; dummy otherwise)
    ):
        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        tx_i32 = fx.Int32(tx)
        bx_i32 = fx.Int32(bx)

        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))  # wave == wave_n

        aq_rsrc = buffer_ops.create_buffer_resource_from_addr(
            _raw(fx.Int64(arg_aq)), num_records_bytes=fx.Index(_aq_bytes)
        )
        saq = SmemPtr(
            allocator.get_base(), lds_off, T.i8, shape=(_aStages * _slot_bytes,)
        )

        # Issue A->LDS for a tile's m_block. raw.ptr.buffer.load.lds is
        # side-effecting (writes LDS). Loop K-tile (slot) outer, M-subblock (sub)
        # inner, matching HIP.
        # Preload the first kStages K-tiles (== ALL tiles when K_TILES_TOTAL==2 /
        # KIMI; == prologue for the streaming K_TILES_TOTAL>2 path). slot == kt here
        # because the prologue tiles are 0..kStages-1.
        def _issue_all_a_loads(m_row0):
            for slot in range_constexpr(kStages):  # slot == K-tile index for preload
                for sub in range_constexpr(_kSubBlocks):
                    lds_row = wave * fx.Int32(_rows_per_wave) + fx.Int32(sub * 8)
                    car = m_row0 + lds_row + (lane // fx.Int32(8))
                    _issue_a_load_lds(
                        aq_rsrc,
                        saq,
                        slot,
                        slot,
                        car,
                        lane,
                        _slot_bytes,
                        lds_row,
                        k_half=_K_HALF,
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
                aq_rsrc=aq_rsrc,
                D_INTER=_K,
                D_INTER_REAL=_K_REAL,
                aStages=_aStages,
            )

        # total_m_blocks = cumsum[0] / BM ; bound = total_m_blocks * _num_n_blocks
        if const_expr(_persistent):
            # Persistent grid (BM128 non-atomic): NUM_CU workgroups grid-stride over
            # tiles. Peel tile 0 (keeps its sched_barrier so the A->LDS issue is
            # pinned early -- matters for one-shot-sized launches); remaining tiles
            # run without it so the compiler overlaps each tile's loads with the
            # previous tile's epilog.
            cumsum0 = llvm.load(T.i32, _global_ptr1(arg_cumsum, fx.Int32(0)))
            total_m_blocks = _udiv(cumsum0, BM)
            bound = total_m_blocks * fx.Int32(_num_n_blocks)
            grid_nb = fx.Int32(gpu.grid_dim.x)

            # XCD-grouped interleave: remap the raw persistent index -> wgid so
            # consecutive indices spread across the 8 XCDs and same-XCD tiles reuse
            # B in that XCD's private L2 slice. HIP's plain NONATOMIC baseline omits
            # this. Returns a row-major linear tile (m_block*_num_n_blocks+n_block),
            # which _gemm2_body / _issue_all_a_loads split back into (m,n).
            #   PORT_XCD_SWIZZLE<=0: step-1-only (mirrors xcd_remap.hpp swizzle=-1).
            #   PORT_XCD_SWIZZLE>0 : 2-step M-major grouping (positive swizzle).
            _NXCD = 8
            _xq = _udiv(bound, _NXCD)
            _xr = _umod(bound, _NXCD)
            _SW = PORT_XCD_SWIZZLE

            def _xcd(pid):
                xc = _umod(pid, _NXCD)
                wgid = (
                    xc * _xq
                    + fx.Int32(arith.minsi(_raw(xc), _raw(_xr)))
                    + _udiv(pid, _NXCD)
                )
                if const_expr(_SW <= 0):
                    return wgid
                # 2-step M-major grouping (mirrors remap_xcd_grouped positive branch).
                # num_wgid_in_group is a build-time constant; group_size_m is runtime
                # (clamped for the partial trailing group), same as the HIP kernel.
                _ng = fx.Int32(_SW * _num_n_blocks)  # num_wgid_in_group (const)
                group_id = wgid // _ng
                first_pid_m = group_id * fx.Int32(_SW)
                remaining_m = total_m_blocks - first_pid_m
                group_size_m = fx.Int32(
                    arith.minsi(_raw(remaining_m), _raw(fx.Int32(_SW)))
                )
                wig = wgid % _ng
                m_block = first_pid_m + (wig % group_size_m)
                n_block = wig // group_size_m
                return m_block * fx.Int32(_num_n_blocks) + n_block

            if bx_i32 < bound:
                tile = _xcd(bx_i32)
                _issue_all_a_loads(_udiv(tile, _num_n_blocks) * fx.Int32(BM))
                rocdl.sched_barrier(0)
                _run_tile(tile)

            saq._view_cache = None
            for iv in range(bx_i32 + grid_nb, bound, gpu.grid_dim.x):
                wu = fx.Int32(iv)
                # iter-boundary fence: prev tile's LDS reads must finish before
                # this tile overwrites the s_Aq slots (persistent-grid reuse race).
                gpu.barrier()
                # setattr (not `saq._view_cache = None`): an `=` assignment would make
                # the AST for-rewriter treat saq (a SmemPtr, active before the loop) as
                # a loop-carried iter_arg, which only MLIR values can be.
                setattr(saq, "_view_cache", None)
                tile = _xcd(wu)
                _issue_all_a_loads(_udiv(tile, _num_n_blocks) * fx.Int32(BM))
                _run_tile(tile)
        else:
            # One-shot grid (atomic): issue A->LDS BEFORE the cumsum load so the
            # A->LDS HBM latency overlaps the cumsum load + bound check (A->LDS
            # depends only on bx/lane). Only the first n_load_waves hold A rows
            # (BM16: waves 0,1), so gate on wave < n_load_waves.
            m_row0 = _udiv(bx_i32, _num_n_blocks) * fx.Int32(BM)
            if const_expr(_n_load_waves < 4):  # BM16: only waves 0,1 hold A rows
                if wave < fx.Int32(_n_load_waves):
                    _issue_all_a_loads(m_row0)
            else:
                _issue_all_a_loads(m_row0)
            rocdl.sched_barrier(0)

            cumsum0 = llvm.load(T.i32, _global_ptr1(arg_cumsum, fx.Int32(0)))
            total_m_blocks = _udiv(cumsum0, BM)
            bound = total_m_blocks * fx.Int32(_num_n_blocks)

            if bx_i32 < bound:
                _run_tile(bx_i32)

    @flyc.jit
    def launch_gemm2(
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        arg_out: fx.Int64,
        arg_out_scale: fx.Int64,  # flat_out_scale (mxfp4 epilog only; dummy otherwise)
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
            persist = _raw(tw > fx.Int32(NUM_CU * 4))
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


@flyc.jit
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
    *,
    aq_rsrc=None,
    D_INTER=K,
    D_INTER_REAL=None,
    aStages=kStages,
):
    _aStages = aStages
    _kMChunks = kmchunks(BM)
    _slot_bytes = saq_slot_bytes(BM)
    _lds_acc_floats = (min(BM, 64) if epilog == "nonatomic_cshuffle" else BM) * BN
    # K-derived sizes (parametrized over the contraction dim K = inter_dim = D_INTER).
    _K = D_INTER
    _K_HALF = k_half_for(_K)
    _K_TILES_TOTAL = k_tiles_total_for(_K)  # KIMI: 2
    # Real (unpadded) contraction. For a non-256-aligned shard (dsv4 TP8: 384) the
    # weights/inter/scales are zero-padded to D_INTER (=512) so the layout/shuffle
    # is the proven %256 one, but the last 128-K MFMA half-step(s) that fall in the
    # pad region (k in [384,512)) are NOT issued: we skip both the B(weight) HBM
    # load and the MFMA for them. The skipped step would add inter(0)*w(0)=0, so the
    # result is unchanged while ~25% of the last tile's weight bandwidth is saved.
    _K_REAL = D_INTER if D_INTER_REAL is None else D_INTER_REAL
    _n_real_half = (
        _K_REAL + 127
    ) // 128  # valid 128-K MFMA half-steps (512->4, 384->3)
    _kUnroll = kunroll_for(_K)  # streaming main-loop trips (KIMI: 0)
    _kAS_per_chunk_dw = kas_per_chunk_dw_for(_K)
    _kBS_stride_n0_dw = kbs_stride_n0_dw_for(_K)
    _ascale_bytes = ascale_bytes(BM, MAX_M, _K)
    _bq_bytes = bq_bytes_for(NE, N_OUT, _K)
    _bscale_bytes = bscale_bytes_for(NE, N_OUT, _K)
    _kbs_per_expert_dw = kbs_per_expert_dw_for(N_OUT, _K)
    _num_n_blocks = num_n_blocks_for(N_OUT)
    _n_load_waves, _rows_per_wave, _kSubBlocks = tiling(
        BM
    )  # BM16/32:1, BM64:2, BM128:4
    b_aux = 2 if use_nt else 0  # NT: B_q loads carry aux=2 (non-temporal hint)

    # block -> (m_block_idx, n_block_idx) ; e = sorted_expert_ids[m_block_idx]
    m_block_idx = _udiv(bx_i32, _num_n_blocks)
    n_block_idx = bx_i32 - m_block_idx * fx.Int32(_num_n_blocks)
    e = llvm.load(T.i32, _global_ptr1(arg_eids, m_block_idx * fx.Int32(4)))
    e = rocdl.readfirstlane(T.i32, e)
    m_row = m_block_idx * fx.Int32(BM)

    # -- buffer resources (exact num_bytes) ----------------------------------
    # (A_q resource + A->LDS loads are issued by the kernel before the branch.)
    ascale_rsrc = buffer_ops.create_buffer_resource_from_addr(
        _raw(fx.Int64(arg_ascale)), num_records_bytes=fx.Index(_ascale_bytes)
    )
    bq_rsrc = buffer_ops.create_buffer_resource_from_addr(
        _raw(fx.Int64(arg_bq)), num_records_bytes=fx.Index(_bq_bytes)
    )
    bscale_rsrc = buffer_ops.create_buffer_resource_from_addr(
        _raw(fx.Int64(arg_bscale)), num_records_bytes=fx.Index(_bscale_bytes)
    )

    # -- LDS base ------------------------------------------------------------
    lds_base = allocator.get_base()
    saq = SmemPtr(lds_base, lds_off, T.i8, shape=(_aStages * _slot_bytes,))
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
        ) * fx.Int32(_K_HALF)
        b_load_s_base.append(rocdl.readfirstlane(T.i32, v))

    mni_base = n_block_idx * fx.Int32(BN // 16 // 2) + wave * fx.Int32(BN // 64 // 2)
    b_scale_s_base = []
    for mw in range_constexpr(2):
        v = (
            e * fx.Int32(_kbs_per_expert_dw)
            + (mni_base + fx.Int32(mw)) * fx.Int32(_kBS_stride_n0_dw)
        ) * fx.Int32(4)
        b_scale_s_base.append(rocdl.readfirstlane(T.i32, v))

    # a_scale_s_base[sub]: chunk_base = m_row / (16 if BM==16 else 32); sub in kSubBlocks
    chunk_base = m_row // fx.Int32(16 if const_expr(BM == 16) else 32)
    a_scale_s_base = [
        rocdl.readfirstlane(
            T.i32,
            (chunk_base + fx.Int32(sub)) * fx.Int32(_kAS_per_chunk_dw) * fx.Int32(4),
        )
        for sub in range_constexpr(_kSubBlocks)
    ]

    v_voff_scale = ((lane_div_16 * fx.Int32(16)) + lane_mod_16) * fx.Int32(4)

    # -- per-K-tile scale / B load helpers (K-tile offset is K-independent:
    #    A-scale & B-scale = kt*256 bytes; B_q = kt*2048 bytes) -----------------
    def load_a_scale_tile(kt):
        # a_scale_v[sub] for K-tile kt (atomic): ((lane/16)*16 + lane%16)*4 + kt*256
        out = [None] * _kSubBlocks
        for sub in range_constexpr(_kSubBlocks):
            out[sub] = buffer_ops.buffer_load(
                ascale_rsrc,
                (v_voff_scale + fx.Int32(kt * 256)) // fx.Int32(4),
                vec_width=1,
                dtype=T.i32,
                soffset_bytes=a_scale_s_base[sub],
            )
        return out

    def load_b_scale_tile(kt):
        # b_scale[mw] for K-tile kt: v_voff + kt*(kBS_stride_k0_dw*4)
        imm = kt * (kBS_stride_k0_dw * 4)
        out = [None, None]
        for mw in range_constexpr(2):
            out[mw] = buffer_ops.buffer_load(
                bscale_rsrc,
                (v_voff_scale + fx.Int32(imm)) // fx.Int32(4),
                vec_width=1,
                dtype=T.i32,
                soffset_bytes=b_scale_s_base[mw],
            )
        return out

    def load_b_tile(kt):
        # b[j][half] for K-tile kt: (lane/16)*256 + (lane%16)*16 + kt*2048 (+ half*1024)
        v_voff_b = (
            (lane_div_16 * fx.Int32(256))
            + (lane_mod_16 * fx.Int32(16))
            + fx.Int32(kt * 2048)
        )
        out = [[None, None] for _ in range(4)]
        for j in range_constexpr(4):
            for half in range_constexpr(2):
                # Skip the weight HBM load for half-steps in the zero-pad tail
                # (k >= _K_REAL): the MFMA for them is skipped too, so they are dead.
                if const_expr(kt * 2 + half >= _n_real_half):
                    continue
                frag = buffer_ops.buffer_load(
                    bq_rsrc,
                    (v_voff_b + fx.Int32(half * 1024)) // fx.Int32(4),
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=b_aux,
                    soffset_bytes=b_load_s_base[j],
                )
                out[j][half] = Vec(frag)
        return out

    # -- streaming A->LDS: issue K-tile `kt` into LDS slot `slot` (K_TILES_TOTAL>2
    #    only; the K_TILES_TOTAL==2 prologue is preloaded by the kernel). Mirrors
    #    gemm1.issue_a_load_lds: m_row-derived cached rows + lds_swizzle. -------
    def issue_a_load_lds(slot, kt):
        for sub in range_constexpr(_kSubBlocks):
            lds_row = wave * fx.Int32(_rows_per_wave) + fx.Int32(sub * 8)
            car = m_row + lds_row + (lane // fx.Int32(8))
            _issue_a_load_lds(
                aq_rsrc, saq, slot, kt, car, lane, _slot_bytes, lds_row, k_half=_K_HALF
            )

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

    def mfma_cluster(b_tile, a, a_scale_sub, b_scale_slot, init, kt=0):
        # half0 = global K-half-step kt*2, half1 = kt*2+1. Skip the half1 MFMAs when
        # that step is in the zero-pad tail (k >= _K_REAL): b_tile[J][1] was not even
        # loaded, and the step would only add inter(0)*w(0)=0.
        _skip_h1 = (kt * 2 + 1) >= _n_real_half
        for J in range_constexpr(4):
            mni = J // 2
            in_b = J % 2
            sb = b_scale_slot[mni]
            b_J0 = b_tile[J][0]
            b_J1 = None if const_expr(_skip_h1) else b_tile[J][1]
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
                if const_expr(not _skip_h1):
                    accm[i0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_res_ty,
                        [a[i0][1], b_J1, accm[i0][J], 4, 4, 2, sa, 2 + in_b, sb],
                    )
                    if const_expr(_kMChunks > 1):
                        accm[i1][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            mfma_res_ty,
                            [a[i1][1], b_J1, accm[i1][J], 4, 4, 3, sa, 2 + in_b, sb],
                        )

    def _kloop_fence():
        gpu.barrier()

    if const_expr(_K_TILES_TOTAL <= kStages):
        # -- KIMI/DSR fast path: K_TILES_TOTAL <= 2 (D_INTER <= 512), fully
        #    unrolled, ALL tiles preloaded by the kernel. Byte-for-byte identical
        #    to the original K=512 port at K_TILES_TOTAL==2: load all B + scales
        #    upfront, then the unrolled K-loop (vmcnt 23 then 22) with no streaming
        #    A->LDS. K_TILES_TOTAL==1 (D_INTER==256) iterates the single tile here
        #    (init=True on it) -- the streaming path's kUnroll = K_TILES-kStages
        #    would be negative and skip the init mfma, leaving accm uninitialized. -
        a_scale_v = [load_a_scale_tile(kt) for kt in range_constexpr(_K_TILES_TOTAL)]
        b_scale_v = [load_b_scale_tile(kt) for kt in range_constexpr(_K_TILES_TOTAL)]
        b = [load_b_tile(kt) for kt in range_constexpr(_K_TILES_TOTAL)]
        for S in range_constexpr(_K_TILES_TOTAL):
            kt = S
            slot = kt % kStages
            _kloop_fence()
            a = issue_a_ds_read(slot)
            a_scale_sub = [a_scale_v[kt][sub] for sub in range_constexpr(_kSubBlocks)]
            mfma_cluster(b[slot], a, a_scale_sub, b_scale_v[slot], init=(S == 0), kt=kt)
    else:
        # -- streaming double-buffered K-loop (K_TILES_TOTAL>2): the kernel
        #    preloaded tiles 0..kStages-1 (prologue) into the kStages LDS slots.
        #    B-q + scales for ALL tiles are loaded into registers up front (they
        #    are not LDS-bound; matches the fast path's upfront B/scale loads).
        #    The main loop processes tiles [0, kUnroll) while streaming the next
        #    tile [kStages, K_TILES_TOTAL) into the freed LDS slot; the drain
        #    processes the final kStages tiles. Mirrors mxfp4_gemm1's K-loop. ----
        a_scale_v = [load_a_scale_tile(kt) for kt in range_constexpr(_K_TILES_TOTAL)]
        b_scale_v = [load_b_scale_tile(kt) for kt in range_constexpr(_K_TILES_TOTAL)]
        b = [load_b_tile(kt) for kt in range_constexpr(_K_TILES_TOTAL)]

        # main loop: OFFSET in [0, kUnroll). Process tile kt=OFFSET (read from LDS
        # slot kt%_aStages), and stream the next tile next_kt=kStages+OFFSET into
        # slot next_kt%_aStages. With _aStages=3 (triple buffer) read-slot and
        # write-slot never alias, so the streamed load cannot clobber the tile this
        # iteration is reading. A loop-top barrier (inside _kloop_fence) guards the
        # cross-iteration reuse of each slot.
        for OFFSET in range_constexpr(_kUnroll):
            kt = OFFSET
            slot = kt % _aStages
            next_kt = kStages + OFFSET
            write_slot = next_kt % _aStages
            _kloop_fence()
            a = issue_a_ds_read(slot)
            # stream next tile's A into the freed slot (overlaps with the mfma).
            issue_a_load_lds(write_slot, next_kt)
            a_scale_sub = [a_scale_v[kt][sub] for sub in range_constexpr(_kSubBlocks)]
            mfma_cluster(b[kt], a, a_scale_sub, b_scale_v[kt], init=(OFFSET == 0))

        # drain: final kStages tiles (already in LDS, no further streaming).
        for S in range_constexpr(kStages):
            kt = _K_TILES_TOTAL - kStages + S
            slot = kt % _aStages
            _kloop_fence()
            a = issue_a_ds_read(slot)
            a_scale_sub = [a_scale_v[kt][sub] for sub in range_constexpr(_kSubBlocks)]
            mfma_cluster(b[kt], a, a_scale_sub, b_scale_v[kt], init=False)

    # -- epilog ---------------------------------------------------------------
    saq._view_cache = None
    if epilog == "nonatomic":
        # flat per-sorted-row bf16 write (no LDS, no atomic, no weight); a
        # separate scatter_reduce sums the TOPK contributions per token.
        out_base = _global_base_ptr1(arg_out)
        _flat_bf16_epilog(
            accm, out_base, m_row, n_block_idx, wave, lane, N_OUT, _kMChunks
        )
    elif epilog == "nonatomic_cshuffle":
        # cshuffle -> coalesced flat per-sorted-row bf16 write (BM<=64); scatter
        # follows (fly's mfma_moe2_cshuffle recipe). One-shot grid (no _persistent)
        # to avoid the lds_acc reuse race.
        lds_acc._view_cache = None
        _cshuffle_flat_bf16_epilog(
            lds_acc, accm, arg_out, m_row, n_block_idx, wave, lane, BM, N_OUT
        )
    elif epilog == "nonatomic_mxfp4":
        # flat per-sorted-row fp4 (packed q + e8m0 scale) write.
        out_q_base = _global_base_ptr1(arg_out)
        out_scale_base = _global_base_ptr1(arg_out_scale)
        tid_i32 = fx.Int32(gpu.thread_id("x"))
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
    token). i64 element index (rows can exceed the i32 byte range). Writes all BM
    rows unconditionally: a per-row padding skip was tried and lost (the scf.if
    breaks store coalescing, costing more than the ~37% saved padding writes)."""
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    row_base = m_row + lane_div_16 * fx.Int32(4)
    gn_base = n_block_idx * fx.Int32(BN) + wave * fx.Int32(BN // 4) + lane_mod_16
    byte_base = (fx.Int64(row_base) * fx.Int64(N_OUT) + fx.Int64(gn_base)) * fx.Int64(2)
    for i in range_constexpr(kMChunks):
        for J in range_constexpr(4):
            vec = Vec(accm[i][J])
            for v in range_constexpr(4):
                const_off = ((i * 16 + v) * N_OUT + J * 16) * 2
                bf = Vec.from_elements([vec[v]], fx.Float32).to(fx.BFloat16)
                llvm.StoreOp(_raw(bf), _gep1(out_base, byte_base + fx.Int64(const_off)))


def _cshuffle_flat_bf16_epilog(
    lds_acc, accm, arg_out, m_row, n_block_idx, wave, lane, BM, N_OUT
):
    """Nonatomic flat epilog WITH cshuffle: cshuffle accm -> lds_acc (the SAME reorg
    the atomic epilog does) so the flat_out write is a COALESCED <2xbf16> store, then
    write per-sorted-row to flat_out[(m_row+row)*N_OUT + n] (no weight/atomic; a
    scatter_reduce sums the TOPK rows). Mirrors fly's mfma_moe2 cshuffle gemm2.

    The LDS scratch holds bf16 (the output dtype), NOT f32: BM*BN*2 <= 64KB for
    BM<=128, so the cshuffle is SINGLE-PASS even at BM128 -- vs an f32 scratch, where
    128*BN*4=128KB forces a 2-pass 64-row split (2x the LDS barriers). Storing bf16 is
    lossless: the f32->bf16 convert just moves from the readback to the LDS store, and
    the readback then reads 2 adjacent bf16 as one <2xbf16> dword straight to flat_out.
    Halving the barriers is the win on the epilog-bound INTER=256 shapes (qwen/kimik2_b).
    """
    _iC = BM // 16  # accm i-chunks (ALL BM rows -- single pass)
    _REPS = BM // 8
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lds_base = _lds_base_ptr3(lds_acc.get())
    tx_i32 = fx.Int32(gpu.thread_id("x"))
    m_lane = tx_i32 // fx.Int32(32)
    n_lane = tx_i32 % fx.Int32(32)
    col_start = n_lane * fx.Int32(2)
    out_base = _global_base_ptr1(arg_out)

    gpu.barrier()  # pre-store fence (K-loop s_Aq reads done; lds_acc union reuse)
    for i in range_constexpr(_iC):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for J in range_constexpr(4):
            col = wave * fx.Int32(64) + fx.Int32(J * 16) + lane_mod_16
            bf4 = Vec(accm[i][J]).to(fx.BFloat16)  # 4 rows -> bf16 (the store dtype)
            for v in range_constexpr(4):
                idx = (row_base + fx.Int32(v)) * fx.Int32(BN) + col
                llvm.StoreOp(_raw(bf4[v]), _gep3(lds_base, idx * fx.Int32(2)))
    gpu.barrier()
    for mr in range_constexpr(_REPS):
        row_local = fx.Int32(mr * 8) + m_lane
        sorted_row = m_row + row_local
        for s in range_constexpr(4):
            idx0 = row_local * fx.Int32(BN) + col_start + fx.Int32(s * 64)
            # 2 adjacent bf16 = one dword, already the output dtype -> straight write
            pk = Vec(llvm.load(T.vec(2, T.bf16), _gep3(lds_base, idx0 * fx.Int32(2))))
            n_col = n_block_idx * fx.Int32(BN) + col_start + fx.Int32(s * 64)
            elem = fx.Int64(sorted_row) * fx.Int64(N_OUT) + fx.Int64(n_col)
            llvm.StoreOp(_raw(pk), _gep1(out_base, elem * fx.Int64(2)))


@flyc.jit
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
    gpu.barrier()

    NBLK = BN // 32  # 8
    m_lane = tid_i32 // fx.Int32(16)
    n_lane = tid_i32 % fx.Int32(16)
    wave_grp = n_lane // fx.Int32(4)
    kk = n_lane % fx.Int32(4)
    _m_base = m_row + m_lane
    _q_row0 = fx.Int64(_m_base) * fx.Int64(N_OUT // 2)
    _s_row0 = fx.Int64(_m_base) * fx.Int64(N_OUT // 32)
    # Each (mr, half) block needs 8 contiguous f32 from LDS = 2x ds_read_b128.
    # The blocks are independent, so software-pipeline the LDS reads: issue the
    # NEXT block's ds_read before computing the CURRENT block's amax/DPP/pack, so
    # the ds_read latency (the epilog's top lgkmcnt(1) stall) is hidden behind the
    # ~16 ALU ops of amax + 4 DPP + 4 cvt_fp4. Distance-1 prefetch keeps only one
    # extra block of registers live (matters: this kernel is 1-wave / 440 VGPR).
    _blocks = [(mr, half) for mr in range(kMChunks) for half in range(NBLK // 4)]

    def _issue_load(mr, half):
        row_local = fx.Int32(mr * 16) + m_lane
        group = wave_grp + fx.Int32(half * 4)
        col0 = group * fx.Int32(32) + kk * fx.Int32(8)
        base_idx = row_local * fx.Int32(BN) + col0
        v0 = Vec(llvm.load(T.vec(4, T.f32), _gep3(lds_base, base_idx * fx.Int32(4))))
        v1 = Vec(
            llvm.load(
                T.vec(4, T.f32),
                _gep3(lds_base, (base_idx + fx.Int32(4)) * fx.Int32(4)),
            )
        )
        return [v0[0], v0[1], v0[2], v0[3], v1[0], v1[1], v1[2], v1[3]], group, col0

    # prologue: issue first block's loads
    _r_next, _grp_next, _col0_next = _issue_load(*_blocks[0])
    for _bi in range_constexpr(len(_blocks)):
        mr, half = _blocks[_bi]
        r, group, col0 = _r_next, _grp_next, _col0_next
        # prefetch next block's LDS reads before consuming the current block
        if _bi + 1 < len(_blocks):
            _r_next, _grp_next, _col0_next = _issue_load(*_blocks[_bi + 1])
        if True:
            # block amax over |r[0..7]| (positive-float bits) -> bf16-bits
            amax_f = llvm.call_intrinsic(T.f32, "llvm.fabs.f32", [_raw(r[0])], [], [])
            for e in range_constexpr(1, 8):
                abs_e = llvm.call_intrinsic(
                    T.f32, "llvm.fabs.f32", [_raw(r[e])], [], []
                )
                amax_f = arith.maxnumf(amax_f, abs_e)
            amax = arith.shrui(arith.bitcast(T.i32, amax_f), _raw(fx.Int32(16)))
            # DPP quad-amax (reduce across the 4 kk-lanes of the block)
            s1 = rocdl.update_dpp(T.i32, amax, amax, 0xB1, 0xF, 0xF, True)
            a = arith.maxui(amax, s1)
            s2 = rocdl.update_dpp(T.i32, a, a, 0x4E, 0xF, 0xF, True)
            amax_dpp = arith.maxui(a, s2)
            # encode e8m0 RoundUp: e8 = ceil_pow2(amax/6)
            f32b = arith.shli(amax_dpp, _raw(fx.Int32(16)))
            working_i = arith.bitcast(
                T.i32, arith.mulf(arith.bitcast(T.f32, f32b), _raw(fx.Float32(1.0 / 6.0)))
            )
            bexp = arith.andi(
                arith.shrui(
                    arith.addi(working_i, _raw(fx.Int32(0x7FFFFF))), _raw(fx.Int32(23))
                ),
                _raw(fx.Int32(0xFF)),
            )
            e8 = arith.minsi(
                _raw(fx.Int32(254)),
                arith.maxsi(_raw(fx.Int32(0)), bexp),
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
            blk = n_block_idx * fx.Int32(NBLK) + group
            q_byte = (
                _q_row0
                + fx.Int64(mr * 16 * (N_OUT // 2))
                + fx.Int64(global_col // fx.Int32(2))
            )
            s_byte = _s_row0 + fx.Int64(mr * 16 * (N_OUT // 32)) + fx.Int64(blk)
            llvm.StoreOp(packed, _gep1(out_q_base, q_byte), nontemporal=True)
            if kk == fx.Int32(0):
                llvm.StoreOp(arith.trunci(T.i8, e8), _gep1(out_scale_base, s_byte))


@flyc.jit
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

    tx_i32 = fx.Int32(gpu.thread_id("x"))
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
    gpu.barrier()

    # write accm -> lds_acc cshuffle (scalar f32 stores, as HIP does)
    for i in range_constexpr(_kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for J in range_constexpr(4):
            col = wave * fx.Int32(64) + fx.Int32(J * 16) + lane_mod_16
            vec = Vec(accm[i][J])
            for v in range_constexpr(4):
                idx = (row_base + fx.Int32(v)) * fx.Int32(BN) + col
                llvm.StoreOp(_raw(vec[v]), _gep3(lds_base, idx * fx.Int32(4)))

    gpu.barrier()

    # read back + weighted atomic add (token_id / weight prefetched above)
    for mr in range_constexpr(M_REPS):
        row_in_block = fx.Int32(mr * 8) + m_lane
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        if token_id < i32_M:
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
