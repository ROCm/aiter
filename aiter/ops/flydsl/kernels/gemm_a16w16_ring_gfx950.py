# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Adapted from the amd-kernels project's gemm/bf16_n50016_k6144 kernel (revision fd233ef; SPDX-License-Identifier:
# Apache-2.0, SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project): its
# compile_bf16_gemm_n50016_k6144, generalized in N and K, with the M-halves (mh) and ring-depth (r) parameters kept
# free as amd-kernels' own host.py dispatches them.

"""BF16 small-M ring GEMM for gfx950: ``out[M, N] = x[M, K] @ w[N, K]^T``, 1 <= M <= 128.

One launch, 256 tiles of 16-row weight fragments (XCD-interleaved), no split-K and no cross-tile reduction, so the
result is deterministic and CUDA-graph safe. Weight fragments (``w``) and the activation rows (``x``) stream through
an LDS ring of ``r`` K=64 stages by buffer-load-to-LDS DMA, and each wave runs ``mfma_f32_16x16x32_bf16`` over its
fragments with the K order fixed (ascending 32-wide MFMA steps, fp32 accumulate), so the logits are bit-identical
for every legal ``(mh, r)`` choice at the same ``mt``.

``shuffled=True`` reads the weight in the ``[N/16, K/64, 16, 64]`` tile layout (``shuffle_weight_ring``), which
makes each fragment's stage one contiguous 2 KiB read; ``shuffled=False`` reads the row-major weight.
``shuffled=2`` (``LAYOUT_AITER``) reads AITER's ``shuffle_weight(w, (16, 16))`` layout
``[N/16, K/32, 4, 16, 8]`` in place, so a weight that is already preshuffled for the other bpreshuffle GEMMs
needs no second copy: a stage is the same contiguous 2 KiB, only the 16-byte chunks sit at a different offset.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ods
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from . import buffer_ops

LAYOUT_ROW, LAYOUT_RING, LAYOUT_AITER = 0, 1, 2  # weight layouts (``shuffled``)
TILES = 256  # one workgroup column per CU of gfx950
NWV = 4  # waves per M half
W_CM = 2  # cache-policy aux of the weight DMA (streamed once)
LDS_BYTES = 160 * 1024


def ring_depth(mt: int) -> int:
    """Ring slots that fit the 160 KB LDS at one M half (amd-kernels' ring_depth)."""
    return 4 if mt <= 6 else 3


def _sb():
    fx.rocdl.sched_barrier(0)


def _lds_ptr(byte_addr):
    # The backend-resolved LDS pointer (no hand-built address space), as ``issue`` below builds it.
    return fx.to_llvm_ptr(
        fx.inttoptr(
            fx.PointerType.get(T.i8, fx.AddressSpace.Shared), fx.Int64(byte_addr)
        )
    )


def _barrier():
    # Raw s_barrier: the explicit per-wave vmcnt/lgkmcnt waits below order the DMA ring and the barrier pairs with
    # them, which the typed barrier API (it adds its own waitcnt) would reorder.
    llvm.inline_asm(None, [], "s_barrier", "~{memory}", has_side_effects=True)


# Weight DMA offsets are 32-bit; 0x70000000 is the out-of-range sentinel of a fragment past the tile's rows, so the
# whole weight (N * K * 2 bytes) must end below it.
W_MAX_BYTES = 0x70000000


def supported_shape(n: int, k: int) -> bool:
    return (
        n % 16 == 0
        and (n // 16) >= TILES
        and k % 64 == 0
        and k >= 64 * 4
        and n * k * 2 < W_MAX_BYTES
    )


def fits(n: int, k: int, mt: int, mh: int, r: int) -> bool:
    """Whether the ring (LDS size, outstanding-DMA counter, K stages) is legal for this shape and tiling."""
    if not (supported_shape(n, k) and 1 <= mt <= 8 and mh in (1, 2) and r >= 2):
        return False
    nr = (n // 16) // TILES + 1
    nrw = (nr + NWV - 1) // NWV
    nwt = NWV * mh
    xd = mt * mh * 2
    per = nrw * 2 // mh + (xd + nwt - 1) // nwt
    return (
        r * (nr + 1 + mt * mh) * 2048 <= LDS_BYTES
        and (r - 1) * per <= 63
        and k // 64 >= r
    )


@functools.cache
def compile_ring(
    n: int,
    k: int,
    mt: int,
    mh: int,
    r: int,
    shuffled: int = LAYOUT_RING,
    device: int = 0,
):
    """Launcher for N, K and ``mt`` 16-row fragments per wave, ``mh`` M halves, ring depth ``r``.

    ``device`` only keys the cache: a compiled handle is bound to the device it was built on.
    """
    assert fits(n, k, mt, mh, r), (n, k, mt, mh, r)
    G16 = n // 16
    NRB = G16 // TILES
    NBIG = G16 - NRB * TILES
    NR = NRB + 1
    NRW = (NR + NWV - 1) // NWV
    NB = k // 64
    MT, SHUF, MH = mt, int(shuffled), mh
    assert SHUF in (LAYOUT_ROW, LAYOUT_RING, LAYOUT_AITER), shuffled
    NWT = NWV * MH
    MTT = MT * MH
    TM = 16 * MTT
    XD = MTT * 2
    XPW = (XD + NWT - 1) // NWT
    PER = NRW * 2 // MH + XPW
    WF = NR + 1
    SLOT = (WF + MTT) * 2048
    LDS = r * SLOT

    @fx.struct
    class Shared:
        buf: fx.Array[fx.Uint8, LDS, 16]

    name = f"aiter_ring_bf16_gemm_n{n}_k{k}_tm{TM}_r{r}_mh{MH}_s{SHUF}"

    @flyc.kernel(name=name, known_block_size=[64 * NWT, 1, 1])
    def kernel(
        arg_x: fx.Int64,
        arg_w: fx.Int64,
        arg_y: fx.Int64,
        i32_m: fx.Int32,
        i32_mtiles: fx.Int32,
    ):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        smem = fx.SharedAllocator().allocate(Shared).peek()
        lane = tx % 64
        wave_all = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        wave = wave_all % NWV
        mhalf = wave_all // NWV
        xcd = bid % 8
        j = bid // 8
        t = (j // i32_mtiles) * 8 + xcd
        m0 = (j % i32_mtiles) * TM
        big = t < NBIG
        n0 = big.select(t * (16 * NR), t * (16 * NRB) + NBIG * 16)
        nfrag = big.select(fx.Int32(NR), fx.Int32(NRB))
        g = lane // 16
        l16 = lane % 16
        r8 = lane // 8

        xr = buffer_ops.create_buffer_resource_from_addr(
            arg_x, num_records_bytes=i32_m * (k * 2)
        )
        wr = buffer_ops.create_buffer_resource_from_addr(
            arg_w, num_records_bytes=n * k * 2
        )
        yr = buffer_ops.create_buffer_resource_from_addr(
            arg_y, num_records_bytes=i32_m * (n * 2)
        )
        lds_base = fx.Int32(fx.ptrtoint(smem.buf.ptr))

        w_src, w_dst = [], []
        for i in range_constexpr(NRW):
            rn = wave + i * NWV
            ok = rn < nfrag
            for jq in range_constexpr(2 // MH):
                jj = mhalf + jq if MH == 2 else jq
                pc = (lane % 8) ^ ((r8 // 2) + jj * 4)
                if const_expr(SHUF == LAYOUT_RING):
                    off = (n0 // 16 + rn) * (k * 32) + (jj * 8 + r8) * 128 + pc * 16
                elif const_expr(SHUF == LAYOUT_AITER):
                    off = (
                        (n0 // 16 + rn) * (k * 32)
                        + (pc // 4) * 1024
                        + (pc % 4) * 256
                        + (jj * 8 + r8) * 16
                    )
                else:
                    off = (n0 + rn * 16 + jj * 8 + r8) * (k * 2) + pc * 16
                w_src.append(ok.select(off, fx.Int32(0x70000000)))
                w_dst.append((ok.select(rn, fx.Int32(NR)) * 2 + jj) * 1024)
        x_src, x_dst = [], []
        for i in range_constexpr(XPW):
            d = wave_all + i * NWT
            d = (d < XD).select(d, fx.Int32(XD - 1))
            jx = d % 2
            row = m0 + (d // 2) * 16 + jx * 8 + r8
            row = (row < i32_m).select(row, i32_m - 1)
            px = (lane % 8) ^ ((r8 // 2) + jx * 4)
            x_src.append(row * (k * 2) + px * 16)
            x_dst.append((WF * 2 + d) * 1024)
        aux_w = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), W_CM)
        aux_x = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)

        # Buffer-load-to-LDS DMA is only exposed as a raw rocdl op (no typed fx.copy form with a per-wave
        # LDS base and cache aux); the explicit vmcnt waits below depend on its exact issue order.
        def issue(rs, dst, src, b, soff, aux):
            d = fx.Int32(fx.rocdl.readfirstlane(T.i32, lds_base + (b % r) * SLOT + dst))
            lp = fx.to_llvm_ptr(
                fx.inttoptr(
                    fx.PointerType.get(T.i8, fx.AddressSpace.Shared), fx.Int64(d)
                )
            )
            _rocdl_ods.raw_ptr_buffer_load_async_lds(
                rs,
                lp,
                fx.Int32(16).ir_value(),
                fx.Int32(src).ir_value(),
                fx.Int32(soff).ir_value(),
                fx.Int32(0).ir_value(),
                aux=aux,
            )

        def dma(b):
            for q in range_constexpr(NRW * 2 // MH):
                issue(
                    wr,
                    w_dst[q],
                    w_src[q],
                    b,
                    b * (128 if SHUF == LAYOUT_ROW else 2048),
                    aux_w,
                )
            for q in range_constexpr(XPW):
                issue(xr, x_dst[q], x_src[q], b, b * 128, aux_x)

        wpos = [
            (wave + i * NWV < NR).select(wave + i * NWV, fx.Int32(NR))
            for i in range_constexpr(NRW)
        ]
        r8r = l16 % 8
        hh = l16 // 8
        rd_off = [
            hh * 1024 + (r8r * 8 + ((kk * 4 + g) ^ ((r8r // 2) + hh * 4))) * 16
            for kk in range_constexpr(2)
        ]

        def rd(b, pos, kk):
            p = _lds_ptr(lds_base + (b % r) * SLOT + pos * 2048 + rd_off[kk])
            return Vec(llvm.load(T.vec(4, T.i32), p, alignment=16)).bitcast(fx.BFloat16)

        acc = [
            [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(MT)]
            for _ in range_constexpr(NRW)
        ]

        def read_ops(b, kk):
            return (
                [rd(b, wpos[i], kk) for i in range_constexpr(NRW)],
                [rd(b, WF + mhalf * MT + m, kk) for m in range_constexpr(MT)],
            )

        def mfmas(ops):
            wv, xv = ops
            for i in range_constexpr(NRW):
                for m in range_constexpr(MT):
                    # Raw MFMA, not fx.make_mma_atom + fx.gemm: the s_waitcnt vmcnt / lgkmcnt counts and the
                    # sched_barrier groups of this ring are written against this exact MFMA issue order (A/B
                    # operands as register vectors); the atom form is not yet ISA-diffed against it.
                    acc[i][m] = Vec(
                        fx.rocdl.mfma_f32_16x16x32_bf16(
                            Vec.make_type(4, fx.Float32),
                            [wv[i], xv[m], acc[i][m], 0, 0, 0],
                        )
                    )

        for b in range_constexpr(r):
            dma(b)
        _sb()
        fx.rocdl.s_waitcnt(vmcnt=(r - 1) * PER)
        _barrier()
        _sb()
        ops_a = read_ops(0, 0)
        for b in range_constexpr(NB):
            _sb()
            ops_b = read_ops(b, 1)
            _sb()
            mfmas(ops_a)
            _sb()
            if b + 1 < NB:
                fx.rocdl.s_waitcnt(
                    lgkmcnt=0, vmcnt=(min(b + r - 1, NB - 1) - (b + 1)) * PER
                )
                _barrier()
                _sb()
                ops_a = read_ops(b + 1, 0)
                _sb()
                if b + r < NB:
                    dma(b + r)
                _sb()
            mfmas(ops_b)

        for i in range_constexpr(NRW):
            rn = wave + i * NWV
            nn = n0 + rn * 16 + g * 4
            for m in range_constexpr(MT):
                row = m0 + (mhalf * MT + m) * 16 + l16
                pk = acc[i][m].to(fx.BFloat16).bitcast(fx.Int32)
                buffer_ops.buffer_store(
                    pk, yr, row * (n // 2) + nn // 2, mask=(rn < nfrag) & (row < i32_m)
                )

    @flyc.jit
    def launch(
        arg_x: fx.Int64,
        arg_w: fx.Int64,
        arg_y: fx.Int64,
        i32_m: fx.Int32,
        i32_mtiles: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(arg_x, arg_w, arg_y, i32_m, i32_mtiles).launch(
            grid=(TILES * i32_mtiles, 1, 1), block=(64 * NWT, 1, 1), stream=stream
        )

    return launch
