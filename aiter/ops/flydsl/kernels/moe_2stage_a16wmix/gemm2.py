# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .utils import (
    _gep,
    _global_base_ptr1,
    _global_i32_at,
    _lds_ptr3,
    _mma_bf16,
    _raw,
    _udiv,
    _umod,
    make_a_loader,
    make_b_loader,
)

# gfx950 CU count; caps the persistent gemm2 grid so high-expert launches (E896) do
# not over-launch ~max_m_blocks empty CTAs.
NUM_CU = 256


# @flyc.jit is LOAD-BEARING: it AST-rewrites ``if token_id < i32_M`` into an scf.if.
# Without it the guard runs as a plain Python if (dropped at trace), so the atomic-fadd
# scatter fires on padded/OOB rows -- ~13x s2 regression (39us -> ~490us at E896).
@flyc.jit
def _atomic_bf16_epilog(
    lds_acc_base_i32,
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
    BN,
):
    _kMChunks = BM // 16
    M_REPS = BM // 8
    # 4 waves split the BN(=TILE_N) tile (generic over BN, e.g. int4 tile_n=128).
    _n_per_wave = BN // 4
    num_acc_n = _n_per_wave // 16
    _s_count = BN // 64  # each s-iter covers 64 cols (32 lanes x vec2)
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lds_base = _lds_ptr3(lds_acc_base_i32, fx.Int32(0))

    tx_i32 = fx.Int32(gpu.thread_id("x"))
    m_lane = tx_i32 // fx.Int32(32)
    n_lane = tx_i32 % fx.Int32(32)
    col_start = n_lane * fx.Int32(2)
    stids_base = _global_base_ptr1(arg_stids)
    sweights_base = _global_base_ptr1(arg_sweights)
    out_base = _global_base_ptr1(arg_out)

    packed = []
    weight = []
    for mr in range_constexpr(M_REPS):
        sorted_pos = m_row + fx.Int32(mr * 8) + m_lane
        packed.append(
            llvm.load(T.i32, _gep(stids_base, sorted_pos * fx.Int32(4)), invariant=True)
        )
        weight.append(
            llvm.load(
                T.f32, _gep(sweights_base, sorted_pos * fx.Int32(4)), invariant=True
            )
        )

    for i in range_constexpr(_kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for J in range_constexpr(num_acc_n):
            col = wave * fx.Int32(_n_per_wave) + fx.Int32(J * 16) + lane_mod_16
            vec = Vec(accm[i][J])
            for v in range_constexpr(4):
                idx = (row_base + fx.Int32(v)) * fx.Int32(BN) + col
                llvm.StoreOp(_raw(vec[v]), _gep(lds_base, idx * fx.Int32(4)))

    gpu.barrier()

    for mr in range_constexpr(M_REPS):
        row_in_block = fx.Int32(mr * 8) + m_lane
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        if token_id < i32_M:
            row_base_addr = (
                token_id * fx.Int32(N_OUT) + n_block_idx * fx.Int32(BN) + col_start
            )
            for s in range_constexpr(_s_count):
                idx0 = row_in_block * fx.Int32(BN) + col_start + fx.Int32(s * 64)
                v2 = Vec(llvm.load(T.vec(2, T.f32), _gep(lds_base, idx0 * fx.Int32(4))))
                pk = Vec.from_elements(
                    [v2[0] * weight[mr], v2[1] * weight[mr]], fx.Float32
                ).to(fx.BFloat16)
                off = (row_base_addr + fx.Int32(s * 64)) * fx.Int32(2)
                out_ptr = _gep(out_base, off)
                llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.fadd,
                    out_ptr,
                    _raw(pk),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                )


def _gemm2_body_a16w4(
    lds_raw_ptr,
    arg_a,
    arg_bq,
    arg_bscale,
    arg_eids,
    arg_stids,
    arg_sweights,
    arg_out,
    bx_i32,
    lane,
    wave,
    i32_M,
    *,
    BM,
    TILE_N,
    TILE_K,
    N_OUT,
    INTER,
    NE,
    b_cache_mod=2,
    w_dtype="fp4",
    use_k16=False,
):
    """a16w4/a16wi4/a16w16 stage2 body. K=inter_dim (contraction), N=model_dim (N_OUT).

    A = bf16 stage1 intermediate by SORTED position. W2 = mxfp4/int4/bf16 (see gemm1).
    Output = bf16 atomic-fadd (routing-weighted) scatter to [tokens, model_dim].
    """
    elem_bytes = 2
    KH_TILE_BYTES = TILE_K * elem_bytes
    LDS_STRIDE = TILE_K
    K = INTER
    K_TILES_TOTAL = K // TILE_K
    m_repeat = BM // 16
    k_unroll = KH_TILE_BYTES // 64
    # 4 waves split the TILE_N tile (matches the atomic-epilog wave-split).
    _n_per_wave = TILE_N // 4
    num_acc_n = _n_per_wave // 16
    k_blocks16 = KH_TILE_BYTES // 16
    _num_n_blocks = N_OUT // TILE_N

    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)

    m_block_idx = bx_i32 // fx.Int32(_num_n_blocks)
    n_block_idx = bx_i32 % fx.Int32(_num_n_blocks)
    e = rocdl.readfirstlane(T.i32, _raw(_global_i32_at(arg_eids, m_block_idx)))
    m_row = m_block_idx * fx.Int32(BM)  # first sorted row of this m-block
    by_n = n_block_idx * fx.Int32(TILE_N)

    # ---- B (weight) operand path: layouts + buffer resources + load closures ----
    # Shared verbatim with gemm1 (see utils.make_b_loader); stage2's N is model_dim and
    # its K is inter_dim (the contraction).
    b_loader = make_b_loader(
        arg_bq,
        arg_bscale,
        N_OUT=N_OUT,
        K=K,
        NE=NE,
        e=e,
        lane_div_16=lane_div_16,
        lane_mod_16=lane_mod_16,
        TILE_K=TILE_K,
        w_dtype=w_dtype,
        b_cache_mod=b_cache_mod,
        use_k16=use_k16,
    )

    # ---- A path (shared with gemm1, see utils.make_a_loader) -------------------
    # A row = SORTED position m_row + row_local; the whole block (256 threads) stages one
    # BM x TILE_K tile. stage2 has a single unslotted A region and XOR-swizzles it to
    # kill LDS bank conflicts.
    c_k_div4 = (K * elem_bytes) // 4
    a_loader = make_a_loader(
        lds_raw_ptr,
        num_i32=BM * LDS_STRIDE // 2,
        BM=BM,
        TILE_K=TILE_K,
        KH_TILE_BYTES=KH_TILE_BYTES,
        k_blocks16=k_blocks16,
        lane_div_16=lane_div_16,
        lane_mod_16=lane_mod_16,
        swizzle=True,
        a_ptr=arg_a,
        a_num_bytes=fx.Int64(0xFFFFFFFF),
        a_load_threads=256,
        row_base_dwords=lambda row_local: (m_row + row_local) * fx.Int32(c_k_div4),
        dma_cache_mod=b_cache_mod,
        dma_via_vgpr=use_k16,
    )

    # ---- N-column addressing (W2 cols of model_dim; wave owns _n_per_wave) ------
    n_tile_base = wave * fx.Int32(_n_per_wave)
    cols = [
        b_loader.col(by_n, n_tile_base, fx.Int32(ni * 16))
        for ni in range_constexpr(num_acc_n)
    ]

    # ---- accumulators: accm[mi][ni] f32[4] (layout the atomic epilog expects) --
    acc_layout = fx.make_layout(4, 1)
    accm = [
        [fx.make_rmem_tensor(acc_layout, fx.Float32) for _ in range(num_acc_n)]
        for _ in range(m_repeat)
    ]
    zero4 = Vec.filled(4, 0.0, fx.Float32)
    for mi in range_constexpr(m_repeat):
        for ni in range_constexpr(num_acc_n):
            accm[mi][ni].store(zero4)

    # Arch-gate: gfx950 K=32 (one MFMA/K-step); gfx942 (use_k16) splits each v8bf16 into
    # two v4bf16 halves -> TWO 16x16x16 MFMAs into the same acc (no 16x16x32 on gfx942).
    if const_expr(use_k16):
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
    else:
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))

    _mma = functools.partial(_mma_bf16, mma_atom, use_k16)

    for kt in range_constexpr(K_TILES_TOTAL):
        base_k = fx.Int32(kt * TILE_K)
        a_loader.store_tile(base_k)
        b_raw = [b_loader.load_raw(base_k, c) for c in cols]
        b_sc = [b_loader.load_scale(base_k, c) for c in cols]
        gpu.barrier()
        for ni in range_constexpr(num_acc_n):
            for ku in range_constexpr(k_unroll):
                bb = b_loader.upconvert(b_raw[ni], ku, b_sc[ni][ku])
                for mi in range_constexpr(m_repeat):
                    a8 = a_loader.load(mi, ku)
                    _mma(accm[mi][ni], a8, bb)
        gpu.barrier()

    # ---- epilogue: atomic bf16 scatter (routing-weighted). K-loop done, so the A-LDS
    # region (offset 0) is reused for the epilog's f32 acc staging.
    gpu.barrier()
    lds_acc_base_i32 = fx.Int32(fx.ptrtoint(lds_raw_ptr))
    accm_v = [
        [accm[i][J].load().ir_value() for J in range(num_acc_n)]
        for i in range(m_repeat)
    ]
    _atomic_bf16_epilog(
        lds_acc_base_i32,
        accm_v,
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
        TILE_N,
    )


def gemm2_a16w4_grid(BM, *, N_OUT, TILE_N, max_m_blocks, persist=False):
    """Flattened launch grid for a16w4 gemm2.

    Non-persistent (default): one CTA per (m-block x n-block) tile over padded
    ``max_m_blocks``. Persistent: cap to ``min(total_work, NUM_CU)`` CTAs (only when
    padded work > ``NUM_CU*4``); each CTA loops over its real work-tiles.
    """
    total_work = int(max_m_blocks) * (N_OUT // TILE_N)
    if persist and total_work > NUM_CU * 4:
        return min(total_work, NUM_CU)
    return total_work


@functools.cache
def compile_gemm2_a16w4_port(
    BM=32,
    *,
    NE,
    N_OUT,
    D_INTER,
    TILE_N=256,
    TILE_K=256,
    xcd_swizzle=1,
    b_cache_mod=2,
    waves_per_eu=None,
    w_dtype="fp4",
    persist=False,
    use_k16,
):
    """a16w4/a16wi4/a16w16 (bf16 intermediate A x mxfp4/int4/bf16 W2) stage2 builder.

    N_OUT = model_dim (down-proj output). D_INTER = inter_dim (contraction). Output
    bf16 [tokens, model_dim] via atomic (routing-weighted) scatter.

    ``xcd_swizzle`` (>0) bijectively round-robins the launch index across the 8 XCDs to
    balance per-XCD/HBM traffic (gemm2 is HBM-bound), + optional M-group swizzle for
    per-XCD L2 locality (group = xcd_swizzle m-blocks).
    """
    assert w_dtype in (
        "fp4",
        "int4",
        "bf16",
    ), f"w_dtype must be 'mxfp4', 'int4' or 'bf16', got {w_dtype!r}"
    # Arch-gate K=16 (gfx942) vs K=32 (gfx950); resolved by the caller and passed in.
    _use_k16 = use_k16
    _K = D_INTER
    assert _K % TILE_K == 0, f"D_INTER (K) must be a multiple of {TILE_K}, got {_K}"
    assert (
        N_OUT % TILE_N == 0
    ), f"model_dim (N_OUT) must be a multiple of {TILE_N}, got {N_OUT}"
    # 4 waves split TILE_N (TILE_N//4 cols each) -> num_acc_n = (TILE_N//4)//16.
    # num_acc_n==0 makes every accumulate/store loop empty -> silent all-zero
    # output that times fast (e.g. TILE_N=32). Require TILE_N >= 64.
    assert (
        TILE_N // 4
    ) >= 16, f"TILE_N//4 must be >= 16 (num_acc_n>=1), got TILE_N={TILE_N}"
    assert BM % 16 == 0, f"BM must be a multiple of 16, got {BM}"
    _num_n_blocks = N_OUT // TILE_N
    KH_TILE_BYTES = TILE_K * 2

    # LDS: A tile (BM x TILE_K bf16) then f32 accumulator region (BM x TILE_N f32).
    _a_bytes = BM * KH_TILE_BYTES
    _acc_bytes = BM * TILE_N * 4  # f32 accumulator region
    _lds_bytes = _a_bytes + _acc_bytes

    _wd_tag = "" if w_dtype == "fp4" else f"_{w_dtype}"
    _name = f"gemm2_a16w4{_wd_tag}_port_ne{NE}_h{N_OUT}_i{_K}_bm{BM}_tn{TILE_N}"
    if b_cache_mod != 2:
        _name += f"_bcm{b_cache_mod}"
    if xcd_swizzle > 0:
        _name += f"_xcd{xcd_swizzle}"
    if waves_per_eu:
        _name += f"_w{waves_per_eu}"
    if persist:
        _name += "_persist"

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, _lds_bytes, 16]

    @flyc.kernel(name=_name, known_block_size=[256, 1, 1])
    def gemm2_kernel(
        arg_a: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        arg_out: fx.Int64,
    ):
        lds_raw_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        tx_i32 = fx.Int32(gpu.thread_id("x"))
        bx_i32 = fx.Int32(gpu.block_id("x"))
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))
        cumsum0 = _global_i32_at(arg_cumsum, fx.Int32(0))
        total_m_blocks = cumsum0 // fx.Int32(BM)
        bound = total_m_blocks * fx.Int32(_num_n_blocks)

        # Bijective XCD round-robin over valid tiles [0, bound) to balance per-XCD/HBM
        # traffic; xcd_swizzle>0 also M-group-swizzles for per-XCD L2 locality.
        _NXCD = 8
        _xq = _udiv(bound, _NXCD)
        _xr = _umod(bound, _NXCD)
        _SW = xcd_swizzle

        def _xcd_np(pid):
            xc = _umod(pid, _NXCD)
            wgid = (
                xc * _xq
                + fx.Int32(arith.minsi(_raw(xc), _raw(_xr)))
                + _udiv(pid, _NXCD)
            )
            if const_expr(_SW <= 0):
                return wgid
            _ng = fx.Int32(_SW * _num_n_blocks)
            group_id = wgid // _ng
            first_pid_m = group_id * fx.Int32(_SW)
            remaining_m = total_m_blocks - first_pid_m
            group_size_m = fx.Int32(arith.minsi(_raw(remaining_m), _raw(fx.Int32(_SW))))
            wig = wgid % _ng
            m_block = first_pid_m + (wig % group_size_m)
            n_block = wig // group_size_m
            return m_block * fx.Int32(_num_n_blocks) + n_block

        def _run_tile(tile):
            _gemm2_body_a16w4(
                lds_raw_ptr,
                arg_a,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_stids,
                arg_sweights,
                arg_out,
                tile,
                lane,
                wave,
                i32_M,
                BM=BM,
                TILE_N=TILE_N,
                TILE_K=TILE_K,
                N_OUT=N_OUT,
                INTER=_K,
                NE=NE,
                b_cache_mod=b_cache_mod,
                w_dtype=w_dtype,
                use_k16=_use_k16,
            )

        if const_expr(persist):
            # Persistent CU-limited grid (~NUM_CU CTAs): each CTA does tile bx_i32 then
            # strides by grid size over [0, bound); _xcd_np maps every visited index, so
            # each tile runs once (same mapping as non-persistent). Loop-top barrier
            # separates the prev tile's epilog LDS from the next tile's A-DMA.
            grid_nb = fx.Int32(gpu.grid_dim.x)
            if bx_i32 < bound:
                _run_tile(_xcd_np(bx_i32))
            for iv in range(bx_i32 + grid_nb, bound, gpu.grid_dim.x):
                gpu.barrier()
                _run_tile(_xcd_np(fx.Int32(iv)))
        else:
            if bx_i32 < bound:
                _run_tile(_xcd_np(bx_i32))

    @flyc.jit
    def launch_gemm2(
        arg_a: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        i32_grid: fx.Int32,
        arg_out: fx.Int64,
        stream: fx.Stream,
    ):
        grid_x = fx.Int64(i32_grid)
        gemm2_kernel(
            arg_a,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_cumsum,
            arg_stids,
            arg_sweights,
            i32_M,
            i32_max_m_blocks,
            arg_out,
            value_attrs={"rocdl.waves_per_eu": waves_per_eu} if waves_per_eu else None,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm2
