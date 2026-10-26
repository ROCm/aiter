# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen gated-residual HC mixing for gfx950 (BF16, decode-sized M).

Like SGLang's CuTe path, keep GEMM accumulators and gate arithmetic in FP32,
rounding only the SiLU intermediate and final output to BF16. Weights are packed
once into the CDNA4 MFMA operand layout. Up-weight columns interleave HC
branches, allowing the gate/weighted reduction in wave shuffles.

Rows are covered by ceil(M/16) MFMA tiles that each wave walks against every B
fragment it loads, so the weights -- the dominant traffic -- are read once per
call whatever M is.

The down projection writes explicit split-K partials ``P``, a small second
kernel reduces them once and applies the SiLU into a BF16 ``T``, and each up
projection CTA copies ``T`` into LDS once so its waves share it rather than each
re-reading it from L2. No cross-CTA spinning is needed.

``fuse_reduce`` instead folds the reduction into the up projection's prologue
and stages ``T`` in LDS (single row tile only), saving a dispatch. Every up CTA
then re-reads all of ``P``, so the down grid must keep ``split_k`` small; with
the reduction separate, ``split_k`` can grow until the down grid covers the GPU,
which measured faster at every M.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, math, range_constexpr, rocdl

from .kernels.mxfp4_gemm_common import lds_typed_ptr, lds_vec_load

# Rows in a 16x16x32 MFMA tile. The A operand is indexed by ``lane % 16``
# whatever M is, so the staged T tile carries whole 16-row tiles.
_MFMA_M = 16
# LDS row-stride pad for that tile, in BF16 elements. Unpadded, consecutive rows
# sit R*2 bytes apart, a multiple of 128 for the production R, which lands all 16
# rows in one LDS bank group -- a 16-way conflict on the ds_read_b128 feeding the
# A operand. R % 32 == 0 makes (R + 8) * 2 an odd multiple of 16B, so rows walk
# the banks and the read drops to its 2-way floor (16 lanes x 16B cannot clear
# 128B/cycle anyway).
_LDS_ROW_PAD = 8
# gfx950 LDS per workgroup.
_LDS_BYTES = 160 * 1024
# Speculative verify passes carry requests x draft tokens rows (32 x 4 for the
# Qwen serving config). Past this a regular GEMM is the better tool, and every
# distinct M is a separate compile.
_MAX_M = 128
# Threads per CTA of the split-K reduce kernel.
_REDUCE_THREADS = 256
# K steps whose B fragments are loaded before any of their MFMAs issue. A rolled
# loop kept one 1 KB weight load in flight per wave, so each step paid a full
# HBM round trip; batching hides that latency at 4 VGPRs per step.
_PREFETCH = 16


def _lds_tile_bytes(m, r):
    """LDS the up projection needs to stage the whole padded [M, R] T operand."""
    return -(-m // _MFMA_M) * _MFMA_M * (r + _LDS_ROW_PAD) * 2


def pack_hc_weights(down, up, hc=4):
    """Pack contiguous [R,K] and [HC*HS,R] weights; call after weight loading."""
    if down.ndim != 2 or up.ndim != 2:
        raise ValueError("HC weights must be matrices")
    if (
        down.dtype != torch.bfloat16
        or up.dtype != down.dtype
        or down.device != up.device
    ):
        raise ValueError("HC weights must be BF16 on the same device")
    r, k = down.shape
    if hc != 4 or k <= 0 or r <= 0 or k % 64 or r % 32 or up.shape != (k, r):
        raise ValueError("HC mix requires HC=4, K divisible by 64 and R by 32")

    def pack(w):
        n, kk = w.shape
        return (
            w.reshape(n // 16, 16, kk // 32, 4, 8).permute(0, 2, 3, 1, 4).contiguous()
        )

    interleaved = up.reshape(hc, k // hc, r).permute(1, 0, 2).reshape(k, r)
    return pack(down), pack(interleaved)


def _load(tensor, offset, width, dtype):
    """Buffer load of `width` elements at `offset`; OOB reads return zero."""
    if width == 8:
        kind = fx.rocdl.BufferCopy128b()
    elif dtype == fx.Float32:
        kind = fx.rocdl.BufferCopy32b()
    else:
        kind = fx.rocdl.BufferCopy16b()
    view = fx.make_view(
        fx.get_iter(tensor) + fx.Int32(offset), fx.make_layout(width, 1)
    )
    frag = fx.make_rmem_tensor(width, dtype)
    fx.copy(fx.make_copy_atom(kind, dtype), view, frag)
    return fx.Vector(frag.load())


def _store(tensor, offset, value, dtype):
    view = fx.make_view(fx.get_iter(tensor) + fx.Int32(offset), fx.make_layout(1, 1))
    atom = fx.make_copy_atom(
        fx.rocdl.BufferCopy32b() if dtype == fx.Float32 else fx.rocdl.BufferCopy16b(),
        dtype,
    )
    frag = fx.make_rmem_tensor(1, dtype)
    frag.store(fx.Vector.from_elements([value], dtype))
    fx.copy(atom, frag, view)


def _mfma(av, bv, acc):
    return fx.Vector(
        rocdl.mfma_f32_16x16x32_bf16(
            acc.ir_value().type,
            [av.ir_value(), bv.ir_value(), acc.ir_value(), 0, 0, 0],
        )
    )


@functools.cache
def _up_shared(rows, rpad):
    """LDS storage for the staged [rows, rpad] BF16 T tile, declared as i32 words."""

    @fx.struct
    class UpShared:
        t: fx.Array[fx.Int32, rows * rpad // 2, 16]

    return UpShared


@flyc.kernel
def _hc_down(
    A: fx.Tensor,
    W: fx.Tensor,
    P: fx.Tensor,
    M: fx.Constexpr[int],
    R: fx.Constexpr[int],
    K: fx.Constexpr[int],
    SPLIT: fx.Constexpr[int],
    WAVES: fx.Constexpr[int],
):
    """X[M,K] against packed down[R,K] -> P[SPLIT,M,R] FP32 split-K partials."""
    tid = gpu.thread_id("x")
    lane = tid % 64
    nt = gpu.block_id("x") * WAVES + tid // 64
    sp = gpu.block_id("y")
    a = fx.rocdl.make_buffer_tensor(A, max_size=False)
    w = fx.rocdl.make_buffer_tensor(W, max_size=False)
    out = fx.rocdl.make_buffer_tensor(P, max_size=False)
    # CDNA4 16x16x32 BF16 MFMA: each lane supplies 8 K values;
    # lane%16 selects A's row / B's column, lane//16 selects the K group.
    # Packed B follows AITER's preshuffle GEMM register layout.
    row_a = lane % 16
    mt = (M + _MFMA_M - 1) // _MFMA_M
    steps = K // SPLIT // 32
    k0 = sp * steps
    accs = [fx.Vector.filled(4, 0.0, fx.Float32)] * mt
    for g in range_constexpr(0, steps, _PREFETCH):
        bvs = []
        for ki in range_constexpr(g, min(g + _PREFETCH, steps)):
            bvs.append(
                _load(w, (nt * (K // 32) + k0 + ki) * 512 + lane * 8, 8, fx.BFloat16)
            )
        for j in range_constexpr(len(bvs)):
            kk = k0 + g + j
            for t in range_constexpr(mt):
                row = t * _MFMA_M + row_a
                # Descriptor-bound OOB loads return zero for the padded M rows.
                ao = (row < M).select(
                    fx.Int32(row * K + kk * 32 + lane // 16 * 8), fx.Int32(M * K)
                )
                accs[t] = _mfma(_load(a, ao, 8, fx.BFloat16), bvs[j], accs[t])
    for t in range_constexpr(mt):
        for i in range_constexpr(4):
            row = t * _MFMA_M + lane // 16 * 4 + i
            col = nt * 16 + lane % 16
            if row < M:
                _store(out, (sp * M + row) * R + col, accs[t][i], fx.Float32)


@flyc.kernel
def _hc_reduce_silu(
    P: fx.Tensor,
    T: fx.Tensor,
    M: fx.Constexpr[int],
    R: fx.Constexpr[int],
    SPLIT: fx.Constexpr[int],
):
    """Sum P[SPLIT,M,R] over split-K, scale by 1/HC and SiLU into T[M,R] BF16."""
    i = gpu.block_id("x") * _REDUCE_THREADS + gpu.thread_id("x")
    p = fx.rocdl.make_buffer_tensor(P, max_size=False)
    t = fx.rocdl.make_buffer_tensor(T, max_size=False)
    acc = fx.Float32(0.0)
    for s in range_constexpr(SPLIT):
        acc = acc + _load(p, fx.Int32(s * M * R) + fx.Int32(i), 1, fx.Float32)[0]
    v = acc * fx.Float32(0.25)
    v = v / (fx.Float32(1.0) + math.exp(-v))
    if i < M * R:
        _store(t, i, v.to(fx.BFloat16), fx.BFloat16)


@flyc.kernel
def _hc_up_silu(
    A: fx.Tensor,
    W: fx.Tensor,
    X: fx.Tensor,
    O: fx.Tensor,
    M: fx.Constexpr[int],
    N: fx.Constexpr[int],
    R: fx.Constexpr[int],
    SPLIT: fx.Constexpr[int],
    WAVES: fx.Constexpr[int],
    FUSED: fx.Constexpr[bool],
    STAGE: fx.Constexpr[bool],
):
    """T[M,R] against up[N,R], then the HC gate and branch mean.

    FUSED: A is P; reduce its split-K partials and SiLU them into LDS first
    (single row tile only). Otherwise A is the T that ``_hc_reduce_silu`` wrote,
    copied into LDS once per CTA when STAGE, else read per wave from global.
    """
    tid = gpu.thread_id("x")
    lane = tid % 64
    nt = gpu.block_id("x") * WAVES + tid // 64
    a = fx.rocdl.make_buffer_tensor(A, max_size=False)
    w = fx.rocdl.make_buffer_tensor(W, max_size=False)
    x = fx.rocdl.make_buffer_tensor(X, max_size=False)
    out = fx.rocdl.make_buffer_tensor(O, max_size=False)
    mt = (M + _MFMA_M - 1) // _MFMA_M

    # Neither the weights nor the gate's X operand depend on T, so put their
    # loads in flight before T is built or read.
    bvs = []
    for ki in range_constexpr(min(_PREFETCH, R // 32)):
        bvs.append(_load(w, (nt * (R // 32) + ki) * 512 + lane * 8, 8, fx.BFloat16))
    col = nt * 16 + lane % 16
    xvs = []
    for t in range_constexpr(mt):
        for i in range_constexpr(4):
            row = t * _MFMA_M + lane // 16 * 4 + i
            # Rows >= M sit past the end of X, so these loads return zero.
            xvs.append(
                _load(x, row * N + (col % 4) * (N // 4) + col // 4, 1, fx.BFloat16)[
                    0
                ].to(fx.Float32)
            )

    stage = FUSED or STAGE
    if const_expr(stage):
        rows = mt * _MFMA_M
        rpad = R + _LDS_ROW_PAD
        threads = 64 * WAVES
        lds = fx.SharedAllocator().allocate(_up_shared(rows, rpad)).peek()
        base = fx.Int32(fx.ptrtoint(lds.t.ptr))

    if const_expr(FUSED):
        # Prologue, replacing the old standalone reduce+SiLU dispatch. Every CTA
        # needs all of T, so each rebuilds it from the partials rather than
        # waiting on a kernel whose 2-workgroup grid was pure launch overhead.
        # Rows >= M are zeroed instead of left as whatever LDS held: MFMA keeps
        # A rows independent so they only reach output rows the epilogue drops,
        # but zero keeps the tile's arithmetic defined.
        for row in range_constexpr(_MFMA_M):
            for c0 in range_constexpr(0, R, threads):
                # Cast tid explicitly: Int32 + raw index promotes to fx.Index,
                # which FlyDSL 0.3.4.1 passes unconverted into int-tuple
                # inference and segfaults on (fixed upstream by ROCm/FlyDSL#1294).
                c = fx.Int32(c0) + fx.Int32(tid)
                if const_expr(row < M):
                    acc = fx.Float32(0.0)
                    for s in range_constexpr(SPLIT):
                        acc = (
                            acc
                            + _load(
                                a, fx.Int32(s * M * R + row * R) + c, 1, fx.Float32
                            )[0]
                        )
                    v = acc * fx.Float32(0.25)
                    v = v / (fx.Float32(1.0) + math.exp(-v))
                else:
                    v = fx.Float32(0.0)
                # R need not be a multiple of `threads`. Overflow lanes read the
                # next row (or descriptor zero) and fold into the row pad, which
                # is never read back: the A operand only touches columns < R.
                slot = (c < fx.Int32(R)).select(c, fx.Int32(R))
                fx.ptr_store(
                    v.to(fx.BFloat16),
                    lds_typed_ptr(
                        base,
                        fx.BFloat16.ir_type,
                        align=2,
                        byte_offset=(fx.Int32(row * rpad) + slot) * fx.Int32(2),
                    ),
                )
    elif const_expr(STAGE):
        # Copy T into LDS in 16-byte chunks, all loads first. Chunk c holds
        # T.flat[8c:8c+8], i.e. row c // (R/8); the row pad shifts its LDS
        # address by 16 bytes per row. Chunks of rows >= M read past the end of
        # T and stage zeros, for the same reason the fused prologue zeroes them.
        chunks = rows * (R // 8)
        vals = []
        for c0 in range_constexpr(0, chunks, threads):
            vals.append(_load(a, (fx.Int32(c0) + fx.Int32(tid)) * 8, 8, fx.BFloat16))
        for j in range_constexpr(len(vals)):
            c = fx.Int32(j * threads) + fx.Int32(tid)
            ptr = lds_typed_ptr(
                base,
                fx.BFloat16.ir_type,
                align=16,
                byte_offset=(c + c // (R // 8)) * fx.Int32(16),
            )
            # Compile-time full chunk vs. runtime-guarded tail; keep separate.
            if const_expr((j + 1) * threads <= chunks):  # noqa: SIM114
                fx.ptr_store(vals[j].ir_value(), ptr)
            elif c < chunks:
                fx.ptr_store(vals[j].ir_value(), ptr)
    if const_expr(stage):
        gpu.barrier()

    row_a = lane % 16
    steps = R // 32
    accs = [fx.Vector.filled(4, 0.0, fx.Float32)] * mt
    for g in range_constexpr(0, steps, _PREFETCH):
        if const_expr(g > 0):
            bvs = []
            for ki in range_constexpr(g, min(g + _PREFETCH, steps)):
                bvs.append(_load(w, (nt * steps + ki) * 512 + lane * 8, 8, fx.BFloat16))
        for j in range_constexpr(len(bvs)):
            ki = g + j
            for t in range_constexpr(mt):
                # Same operand layout as the down pass.
                row = t * _MFMA_M + row_a
                if const_expr(stage):
                    # Every term of the byte offset is a multiple of 16, so the
                    # read stays b128.
                    av = fx.Vector(
                        lds_vec_load(
                            base,
                            fx.Int32(row * rpad + ki * 32 + lane // 16 * 8)
                            * fx.Int32(2),
                            fx.Vector.make_type(8, fx.BFloat16),
                            fx.BFloat16,
                            align=16,
                        )
                    )
                else:
                    ao = (row < M).select(
                        fx.Int32(row * R + ki * 32 + lane // 16 * 8),
                        fx.Int32(M * R),
                    )
                    av = _load(a, ao, 8, fx.BFloat16)
                accs[t] = _mfma(av, bvs[j], accs[t])
    for t in range_constexpr(mt):
        for i in range_constexpr(4):
            row = t * _MFMA_M + lane // 16 * 4 + i
            value = xvs[t * 4 + i] / (fx.Float32(1.0) + math.exp(-accs[t][i]))
            # Four adjacent lanes hold the four branches of one hidden unit.
            # Keep the whole wave active until both shuffles have completed.
            for shift in range_constexpr(2):
                peer = gpu.shuffle(
                    value, fx.Int32(1 << shift), fx.Int32(64), mode="xor"
                )
                value = value + fx.Float32(peer)
            if (row < M) & (lane % 4 == 0):
                _store(
                    out,
                    row * (N // 4) + col // 4,
                    (value * fx.Float32(0.25)).to(fx.BFloat16),
                    fx.BFloat16,
                )


@flyc.jit
def _launch(
    X: fx.Tensor,
    D: fx.Tensor,
    U: fx.Tensor,
    P: fx.Tensor,
    T: fx.Tensor,
    O: fx.Tensor,
    M: fx.Constexpr[int],
    K: fx.Constexpr[int],
    R: fx.Constexpr[int],
    S: fx.Constexpr[int],
    W: fx.Constexpr[int],
    FUSED: fx.Constexpr[bool],
    STAGE: fx.Constexpr[bool],
    stream: fx.Stream,
):
    _hc_down(X, D, P, M, R, K, S, 1).launch(
        grid=(R // 16, S), block=(64,), stream=stream
    )
    if const_expr(FUSED):
        _hc_up_silu(P, U, X, O, M, K, R, S, W, True, True).launch(
            grid=(K // (16 * W),), block=(64 * W,), stream=stream
        )
    else:
        _hc_reduce_silu(P, T, M, R, S).launch(
            grid=((M * R + _REDUCE_THREADS - 1) // _REDUCE_THREADS,),
            block=(_REDUCE_THREADS,),
            stream=stream,
        )
        _hc_up_silu(T, U, X, O, M, K, R, S, W, False, STAGE).launch(
            grid=(K // (16 * W),), block=(64 * W,), stream=stream
        )


def hc_mix(x, down_packed, up_packed, *, split_k=None, up_waves=4, fuse_reduce=False):
    """Return [M,K/4], for BF16 gfx950 inference with 1 <= M <= 128.

    Scratch belongs to each call, including captured graphs. No zeroing or
    semaphore reset is required. ``split_k``, ``up_waves`` and ``fuse_reduce``
    are tuning knobs; defaults were measured for Qwen K=10240, R=320 with
    rotating layer weights. ``fuse_reduce`` folds the split-K reduction into the
    up projection (two dispatches instead of three) and needs M <= 16. It
    measured slower than the three-dispatch default at every M, since it caps
    ``split_k``; it stays for comparison.
    """
    if x.ndim != 2 or down_packed.ndim != 5 or up_packed.ndim != 5:
        raise ValueError("Expected a matrix input and 5D packed weights")
    m, k = x.shape
    r = down_packed.shape[0] * 16
    if split_k is None:
        # The fused prologue re-reads every partial in each up CTA, so it wants
        # a small split; the reduce kernel reads them once, which lets the down
        # grid grow to cover the GPU. Past 32, single-wave down CTAs get too
        # short to amortize their launch and P outgrows what the reduce hides.
        choices = (8, 4) if fuse_reduce else (32, 16, 8, 4)
        split_k = next((s for s in choices if k % (32 * s) == 0), 1)
    if up_waves not in (1, 2, 4, 8) or k % (16 * up_waves):
        raise ValueError("up_waves must divide the output into full MFMA tiles")
    if not isinstance(split_k, int) or split_k < 1:
        raise ValueError("split_k must be a positive integer")
    if (
        not x.is_cuda
        or torch.version.hip is None
        or torch.cuda.get_device_properties(x.device).gcnArchName.split(":")[0]
        != "gfx950"
    ):
        raise ValueError("HC mix requires gfx950")
    if (
        x.dtype != torch.bfloat16
        or not x.is_contiguous()
        or x.data_ptr() % 16
        or not 1 <= m <= _MAX_M
        or r <= 0
        or k <= 0
        or k % (32 * split_k)
        or r % 32
        or k % 64
    ):
        raise ValueError("Unsupported HC mix input shape/dtype/layout")
    if fuse_reduce and m > _MFMA_M:
        raise ValueError("fuse_reduce stages a single 16-row tile; needs M <= 16")
    # The staged T tile is the up kernel's only LDS consumer. Unfused, a tile
    # that does not fit is read from global memory instead.
    stage = _lds_tile_bytes(m, r) <= _LDS_BYTES
    if fuse_reduce and not stage:
        raise ValueError("R too large to stage the HC mix operand in LDS")
    for w in (down_packed, up_packed):
        if (
            w.device != x.device
            or w.dtype != x.dtype
            or not w.is_contiguous()
            or w.data_ptr() % 16
        ):
            raise ValueError("Packed weights must match input device and dtype")
    if down_packed.shape != (r // 16, k // 32, 4, 16, 8) or up_packed.shape != (
        k // 16,
        r // 32,
        4,
        16,
        8,
    ):
        raise ValueError("Invalid packed HC weight shape")
    p = torch.empty((split_k, m, r), device=x.device, dtype=torch.float32)
    # The fused path never touches T; P only fills the argument slot.
    t = p if fuse_reduce else torch.empty((m, r), device=x.device, dtype=x.dtype)
    out = torch.empty((m, k // 4), device=x.device, dtype=x.dtype)
    _launch(
        x,
        down_packed,
        up_packed,
        p,
        t,
        out,
        m,
        k,
        r,
        split_k,
        up_waves,
        bool(fuse_reduce),
        stage,
        torch.cuda.current_stream(x.device),
    )
    return out
