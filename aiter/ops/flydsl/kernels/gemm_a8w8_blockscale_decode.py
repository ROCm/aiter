# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Small-M (decode) FP8 blockscale GEMM for gfx950.

Each wave owns a 16 (M) x 16 (N) output tile and streams the weights with a
register software pipeline.  Workgroups tile N, so a decode step (M <= 32) gets
N/BN workgroups instead of the single 256x256 tile the stock prefill kernel
launches.  The 128-wide K blocks carry the 1x128 activation scales and the
128x128 weight scales: every block's 16x16 MFMA result is scaled per output
element and accumulated in fp32, which is exactly the blockscale definition.

A and B operand fragments use the 32-consecutive-K MFMA layout, so one lane
loads two 16-byte chunks per operand per K block from the same output row.  B
is the plain row-major weight; the (16, 16)-preshuffled layout is left to the
``gemm_a8w8_blockscale_bpreshuffle`` path.
The C fragment holds four consecutive M rows for one N column, so the x scale
is a contiguous four-float slice and the epilogue writes four coalesced bf16
columns.  Weight fragments are shared by both 16-row activation groups.

A workgroup may put several waves on the same output tile, splitting K between
them (``k_waves``).  Short N grids then keep every CU busy and multiply the
memory-level parallelism of a bandwidth-latency-bound decode kernel.  The
per-wave partial sums meet in LDS and one cooperative pass writes bf16.

``argScaleA`` is the caller's row-major ``[M, K/128]`` scale, indexed
``m * KB + kb``.  The public op passes vLLM's layout through unchanged; the
launcher limits the number of K-split waves to what the CU register file can
hold and the adapter passes the matching ``waves_per_eu`` compile hint, so a
specialization cannot be built that fails to launch.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.kernels_common import ceildiv
from aiter.ops.flydsl.kernels.tensor_shim import (
    buf_base_i64,
    buf_copy_load,
    ptr_buf_tensor,
)


def _fragment_load(buf, unit_index, elem=fx.Int32):
    """One 16-byte chunk of an MFMA 32-consecutive-K operand fragment."""
    return buf_copy_load(buf, unit_index, elem=elem, unit_elems=4)


#: gfx950 has four SIMDs per CU and 16,384 32-bit VGPRs per SIMD.  Waves of a
#: workgroup are spread over the SIMDs, so ``ceil(WAVES / 4)`` of them share
#: one SIMD's file.  A workgroup of at most 16 waves is the largest that still
#: fits one 1024-thread launch.
MAX_WAVES = 16


def decode_k_waves(N: int, K: int, BM: int, BN: int) -> int:
    """K-split waves the heuristic asks for, before the occupancy clamp."""
    _kb = K // 128
    _grid = (N + BN - 1) // BN
    _cap = min(MAX_WAVES, max(1, _kb // 2), max(1, 256 // _grid))
    # A 32-row tile keeps two activation fragments live per K block.  At
    # ``KCH >= 4`` (e.g. N=320, K=4096, 8 waves) the compiled kernel needs
    # ~142 VGPRs, i.e. more than two waves per SIMD can hold; four waves keep
    # one wave per SIMD and the full 256-VGPR budget.  A 16-row tile needs
    # ~54-152 VGPRs, and the ``waves_per_eu`` hint below keeps every split the
    # heuristic picks launchable without giving up the N=320, K=2560 10-way
    # split (54 VGPRs at 3 waves per EU).
    _cap = min(_cap, 4 if BM == 32 else MAX_WAVES)
    waves = 1
    for _d in range(1, _cap + 1):
        if _kb % _d == 0:
            waves = _d
    return waves


def decode_waves_per_eu(waves: int) -> int:
    """Occupancy hint that makes ``waves`` waves of one workgroup fit.

    Waves spread over the CU's four SIMDs, so ``ceil(waves / 4)`` waves share
    one SIMD's 16,384 VGPR file.  FlyDSL's supported ``waves_per_eu`` compile
    hint (``--amdgpu-waves-per-eu``) makes the compiler allocate at most
    ``16384 / (64 * ceil(waves / 4))`` VGPRs per lane, spilling if a
    specialization would otherwise need more.
    """
    return (waves + 3) // 4


def compile_gemm_fp8_decode(
    N: int,
    K: int,
    BM: int,
    BN: int,
    out_dtype,
    num_stages: int = 4,
    k_waves: int = 0,
):
    """Build the small-M blockscale kernel launcher.

    BM is 16 or 32 (one or two MFMA row groups); BN is a positive multiple of 16.
    ``k_waves`` waves of a workgroup each reduce a disjoint K slice of the same
    output tile and combine through LDS.  Wave counts above the CU register
    budget are clamped (32-row tiles: at most 4; any tile: at most 16).
    """
    assert K % 256 == 0, K
    assert N % 16 == 0, N
    assert BM in (16, 32), BM
    assert BN % 16 == 0 and BN <= 64, BN
    assert out_dtype in (fx.BFloat16, fx.Float16)
    NW = BN // 16
    NG = BM // 16
    if k_waves <= 0:
        k_waves = decode_k_waves(N, K, BM, BN)
    # A 32-row tile's two live activation fragments need more registers than a
    # 16-row one; four waves keep it at one wave per SIMD.  Both branches also
    # respect MAX_WAVES / the 1024-thread launch limit.
    k_waves = min(k_waves, 4 if BM == 32 else MAX_WAVES)
    k_waves = max(1, min(k_waves, MAX_WAVES // NW))
    WS = k_waves
    assert WS >= 1 and (K // 128) % WS == 0
    WAVES = NW * WS
    THREADS = 64 * WAVES
    assert THREADS <= 1024
    KB = K // 128
    KCH = KB // WS
    D = min(num_stages, KCH)
    assert D >= 1
    KWU = K // 16  # 16-byte units per row
    scaleB_rows = (N + 127) // 128
    TILE_ELEMS = BM * BN

    if WS > 1:

        @fx.struct
        class SharedStorage:
            psum: fx.Array[fx.Float32, TILE_ELEMS * WS, 4]

    kname = (
        f"flydsl_decode_v8_{N}_{K}_{BM}_{BN}"
        f"_d{D}_k{WS}_{int(out_dtype == fx.Float16)}"
    )

    @flyc.kernel(name=kname, known_block_size=[THREADS, 1, 1])
    def gemm_decode_kernel(
        argA: fx.Tensor,
        argB: fx.Tensor,
        argC: fx.Tensor,
        argScaleA: fx.Tensor,
        argScaleB: fx.Tensor,
        M: fx.Int32,
    ):
        tid = fx.thread_idx.x
        wave = tid // 64
        lane = tid % 64
        idx_n = wave % NW
        k_wave = wave // NW
        lane_row = lane % 16
        m_rep = lane // 16
        n0 = fx.block_idx.x * BN

        # ---- rebased buffer views -------------------------------------------------
        b_left = N - n0
        b_rows = (b_left < BN).select(b_left, BN)
        wt = ptr_buf_tensor(
            buf_base_i64(argB) + fx.Int64(n0) * fx.Int64(K),
            elem=fx.Int32,
            unit_elems=4,
            num_records_bytes=fx.Int64(b_rows) * fx.Int64(K),
        )
        xt = ptr_buf_tensor(
            buf_base_i64(argA),
            elem=fx.Int32,
            unit_elems=4,
            num_records_bytes=fx.Int64(M) * fx.Int64(K),
        )
        sat = ptr_buf_tensor(
            buf_base_i64(argScaleA),
            elem=fx.Float32,
            unit_elems=1,
            num_records_bytes=fx.Int64(M) * fx.Int64(KB) * 4,
        )
        sb_rsrc = rocdl.get_buffer_rsrc(
            fx.get_iter(
                ptr_buf_tensor(
                    buf_base_i64(argScaleB),
                    elem=fx.Float32,
                    unit_elems=1,
                    num_records_bytes=fx.Int64(scaleB_rows) * fx.Int64(KB) * 4,
                )
            )
        )
        ct = ptr_buf_tensor(
            buf_base_i64(argC),
            elem=out_dtype,
            unit_elems=1,
            num_records_bytes=fx.Int64(M) * fx.Int64(N) * 2,
        )
        c_rsrc = rocdl.get_buffer_rsrc(fx.get_iter(ct))

        nb = n0 // 128
        kbase = k_wave * KCH

        # Per-thread source unit offsets.  One lane owns 32 consecutive K
        # values, k = 32*(lane//16) + [0,32).
        def x_unit_index(gk, local_row):
            return local_row * KWU + gk * 8 + (lane // 16) * 2

        w_row = idx_n * 16 + lane_row

        def load_w(gk):
            idx = x_unit_index(gk, w_row)
            lo = _fragment_load(wt, idx)
            hi = _fragment_load(wt, idx + 1)
            return lo.shuffle(hi, list(range(8))).bitcast(fx.Int32)

        def load_x(gk, g):
            idx = x_unit_index(gk, g * 16 + lane_row)
            lo = _fragment_load(xt, idx)
            hi = _fragment_load(xt, idx + 1)
            return lo.shuffle(hi, list(range(8))).bitcast(fx.Int32)

        def load_sa(gk, g):
            # x_scale is the caller's row-major [M, KB]: one scalar per row.
            base = (g * 16 + m_rep * 4) * KB + gk
            return [
                buf_copy_load(sat, base + i * KB, elem=fx.Float32, unit_elems=1)
                for i in range_constexpr(4)
            ]

        def load_sb(gk):
            v = buffer_ops.buffer_load(
                sb_rsrc, nb * KB + gk, vec_width=1, is_scalar=True
            )
            return fx.Int32(v).bitcast(fx.Float32)

        ring_x = [[None] * D for _ in range_constexpr(NG)]
        ring_w = [None] * D
        ring_sa = [[None] * D for _ in range_constexpr(NG)]
        ring_sb = [None] * D
        for s in range_constexpr(D):
            ring_w[s] = load_w(kbase + s)
            ring_sb[s] = load_sb(kbase + s)
            for g in range_constexpr(NG):
                ring_x[g][s] = load_x(kbase + s, g)
                ring_sa[g][s] = load_sa(kbase + s, g)

        p = [[fx.Float32(0.0) for _ in range_constexpr(4)] for _ in range_constexpr(NG)]
        zero4 = Vec.filled(4, 0.0, fx.Float32)
        for kb in range_constexpr(KCH):
            slot = kb % D
            wf = ring_w[slot]
            sb = ring_sb[slot]
            xfs = [ring_x[g][slot] for g in range_constexpr(NG)]
            sas = [ring_sa[g][slot] for g in range_constexpr(NG)]
            nxt = kb + D
            if const_expr(nxt < KCH):
                ring_w[slot] = load_w(kbase + nxt)
                ring_sb[slot] = load_sb(kbase + nxt)
                for g in range_constexpr(NG):
                    ring_x[g][slot] = load_x(kbase + nxt, g)
                    ring_sa[g][slot] = load_sa(kbase + nxt, g)
            for g in range_constexpr(NG):
                c = Vec(
                    rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        T.vec(4, T.f32),
                        [
                            xfs[g],
                            wf,
                            zero4,
                            0,
                            0,
                            0,
                            fx.Int32(0),
                            0,
                            fx.Int32(0),
                        ],
                    )
                )
                p[g] = [
                    fx.fma(c[i], sas[g][i] * sb, p[g][i]) for i in range_constexpr(4)
                ]

        # ---- epilogue -------------------------------------------------------------
        if const_expr(WS == 1):
            n = n0 + idx_n * 16 + lane_row
            for g in range_constexpr(NG):
                for i in range_constexpr(4):
                    m = g * 16 + m_rep * 4 + i
                    valid = (m < M) & (n < N)
                    buffer_ops.buffer_store(
                        p[g][i].to(out_dtype).ir_value(),
                        c_rsrc,
                        fx.Int32(m * N + n),
                        mask=valid,
                    )
        else:
            lds = fx.SharedAllocator().allocate(SharedStorage).peek()
            psum = fx.make_view(lds.psum.ptr, fx.make_layout(TILE_ELEMS * WS, 1))
            n_local = idx_n * 16 + lane_row
            for g in range_constexpr(NG):
                for i in range_constexpr(4):
                    m_local = g * 16 + m_rep * 4 + i
                    fx.memref_store(
                        p[g][i],
                        psum,
                        fx.Int32((k_wave * BM + m_local) * BN + n_local),
                    )
            rocdl.s_barrier()
            n_iters = (TILE_ELEMS + THREADS - 1) // THREADS
            for it in range_constexpr(n_iters):
                idx = tid + it * THREADS
                # Every thread of the last iteration may be past the tile
                # (TILE_ELEMS is not a multiple of THREADS).  Fold such a
                # thread onto element 0 before the LDS load: its store below
                # is masked, but the read must not leave the psum block.
                safe_idx = (idx < TILE_ELEMS).select(idx, fx.Int32(0))
                m = safe_idx // BN
                n = safe_idx % BN
                total = fx.Float32(0.0)
                for w in range_constexpr(WS):
                    total = total + fx.memref_load(
                        psum, fx.Int32(w * TILE_ELEMS + safe_idx)
                    )
                valid = (idx < TILE_ELEMS) & (m < M) & (n0 + n < N)
                buffer_ops.buffer_store(
                    total.to(out_dtype).ir_value(),
                    c_rsrc,
                    (idx < TILE_ELEMS).select(
                        fx.Int32(m * N + n0 + n), fx.Int32(0x7FFFFFFF)
                    ),
                    mask=valid,
                )

    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        scaleA: fx.Tensor,
        scaleB: fx.Tensor,
        M: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        value_attrs = {"llvm.passthrough": [["target-features", "-packed-fp32-ops"]]}
        gemm_decode_kernel(A, B, C, scaleA, scaleB, M, value_attrs=value_attrs).launch(
            grid=(ceildiv(N, BN), 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    return launch_gemm
