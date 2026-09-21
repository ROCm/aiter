# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen gated-residual HC mixing for gfx950 (BF16, small M).

Like SGLang's CuTe path, keep GEMM accumulators and gate arithmetic in FP32,
rounding only the SiLU intermediate and final output to BF16. Down projection
uses explicit split-K partials; a separate reduction avoids cross-CTA spinning.
Weights are packed once into the CDNA4 MFMA operand layout. Up-weight columns
interleave HC branches, allowing the gate/weighted reduction in wave shuffles.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, math, range_constexpr, rocdl


def pack_hc_weights(down, up, hc=4):
    """Pack contiguous [R,K] and [HC*HS,R] weights; call after weight loading."""
    if down.ndim != 2 or up.ndim != 2:
        raise ValueError("HC weights must be matrices")
    if down.dtype != torch.bfloat16 or up.dtype != down.dtype or down.device != up.device:
        raise ValueError("HC weights must be BF16 on the same device")
    r, k = down.shape
    if hc != 4 or k <= 0 or r <= 0 or k % 64 or r % 32 or up.shape != (k, r):
        raise ValueError("HC mix requires HC=4, K divisible by 64 and R by 32")

    def pack(w):
        n, kk = w.shape
        return w.reshape(n // 16, 16, kk // 32, 4, 8).permute(0, 2, 3, 1, 4).contiguous()

    interleaved = up.reshape(hc, k // hc, r).permute(1, 0, 2).reshape(k, r)
    return pack(down), pack(interleaved)


def _load(tensor, offset, width, dtype):
    view = fx.make_view(fx.get_iter(tensor) + offset, fx.make_layout(width, 1))
    atom = fx.make_copy_atom(
        fx.rocdl.BufferCopy128b() if width == 8 else fx.rocdl.BufferCopy16b(),
        dtype,
    )
    frag = fx.make_rmem_tensor(width, dtype)
    fx.copy(atom, view, frag)
    return fx.Vector(frag.load())


def _store(tensor, offset, value, dtype):
    view = fx.make_view(fx.get_iter(tensor) + offset, fx.make_layout(1, 1))
    atom = fx.make_copy_atom(
        fx.rocdl.BufferCopy32b() if dtype == fx.Float32 else fx.rocdl.BufferCopy16b(),
        dtype,
    )
    frag = fx.make_rmem_tensor(1, dtype)
    frag.store(fx.Vector.from_elements([value], dtype))
    fx.copy(atom, frag, view)


@flyc.kernel
def _hc_project(
    A: fx.Tensor,
    W: fx.Tensor,
    X: fx.Tensor,
    O: fx.Tensor,
    M: fx.Constexpr[int],
    N: fx.Constexpr[int],
    K: fx.Constexpr[int],
    SPLIT: fx.Constexpr[int],
    UP: fx.Constexpr[bool],
    WAVES: fx.Constexpr[int],
):
    tid = gpu.thread_id("x")
    lane = tid % 64
    nt = gpu.block_id("x") * WAVES + tid // 64
    sp = gpu.block_id("y")
    a = fx.rocdl.make_buffer_tensor(A, max_size=False)
    w = fx.rocdl.make_buffer_tensor(W, max_size=False)
    x = fx.rocdl.make_buffer_tensor(X, max_size=False)
    out = fx.rocdl.make_buffer_tensor(O, max_size=False)
    # CDNA4 16x16x32 BF16 MFMA: each lane supplies 8 K values;
    # lane%16 selects A's row / B's column, lane//16 selects the K group.
    # Packed B follows AITER's preshuffle GEMM register layout.
    row_a = lane % 16
    acc0 = fx.Vector.filled(4, 0.0, fx.Float32)
    for ki, state in range(
        fx.Int64(0), fx.Int64(K // SPLIT // 32), fx.Int64(1), init=[acc0]
    ):
        kk = sp * (K // SPLIT // 32) + ki
        # Descriptor-bound OOB loads return zero for the padded M rows.
        ao = (row_a < M).select(
            fx.Int32(row_a * K + kk * 32 + lane // 16 * 8), fx.Int32(M * K)
        )
        av = _load(a, ao, 8, fx.BFloat16)
        bv = _load(w, (nt * (K // 32) + kk) * 512 + lane * 8, 8, fx.BFloat16)
        c = state[0]
        v = rocdl.mfma_f32_16x16x32_bf16(
            c.ir_value().type,
            [av.ir_value(), bv.ir_value(), c.ir_value(), 0, 0, 0],
        )
        result = yield [fx.Vector(v)]
    acc = result
    for i in range_constexpr(4):
        row = lane // 16 * 4 + i
        col = nt * 16 + lane % 16
        if const_expr(UP):
            xv = _load(
                x, row * N + (col % 4) * (N // 4) + col // 4, 1, fx.BFloat16
            )[0].to(fx.Float32)
            value = xv / (fx.Float32(1.0) + math.exp(-acc[i]))
            # Four adjacent lanes hold the four branches of one hidden unit.
            # Keep the whole wave active until both shuffles have completed.
            for shift in range_constexpr(2):
                peer = gpu.shuffle(value, fx.Int32(1 << shift), fx.Int32(64), mode="xor")
                value = value + fx.Float32(peer)
            if (row < M) & (lane % 4 == 0):
                _store(
                    out, row * (N // 4) + col // 4,
                    (value * fx.Float32(0.25)).to(fx.BFloat16), fx.BFloat16,
                )
        else:
            if row < M:
                _store(out, (sp * M + row) * N + col, acc[i], fx.Float32)


@flyc.kernel
def _hc_reduce_silu(
    P: fx.Tensor, T: fx.Tensor,
    M: fx.Constexpr[int], R: fx.Constexpr[int], S: fx.Constexpr[int],
):
    idx = gpu.block_id("x") * 256 + gpu.thread_id("x")
    p = fx.rocdl.make_buffer_tensor(P, max_size=False)
    t = fx.rocdl.make_buffer_tensor(T, max_size=False)
    if idx < M * R:
        acc = fx.Float32(0.0)
        for s in range_constexpr(S):
            view = fx.make_view(fx.get_iter(p) + s * M * R + idx, fx.make_layout(1, 1))
            frag = fx.make_rmem_tensor(1, fx.Float32)
            fx.copy(fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32), view, frag)
            acc = acc + fx.Vector(frag.load())[0]
        v = acc * fx.Float32(0.25)
        v = v / (fx.Float32(1.0) + math.exp(-v))
        _store(t, idx, v.to(fx.BFloat16), fx.BFloat16)


@flyc.jit
def _launch(
    X: fx.Tensor, D: fx.Tensor, U: fx.Tensor,
    P: fx.Tensor, T: fx.Tensor, O: fx.Tensor,
    M: fx.Constexpr[int], K: fx.Constexpr[int], R: fx.Constexpr[int],
    S: fx.Constexpr[int], W: fx.Constexpr[int], stream: fx.Stream,
):
    _hc_project(X, D, X, P, M, R, K, S, False, 1).launch(
        grid=(R // 16, S), block=(64,), stream=stream
    )
    _hc_reduce_silu(P, T, M, R, S).launch(
        grid=((M * R + 255) // 256,), block=(256,), stream=stream
    )
    _hc_project(T, U, X, O, M, K, R, 1, True, W).launch(
        grid=(K // (16 * W),), block=(64 * W,), stream=stream
    )


def hc_mix(x, down_packed, up_packed, *, split_k=None, up_waves=4):
    """Return [M,K/4], for BF16 gfx950 inference with 1 <= M <= 16.

    Scratch belongs to each call, including captured graphs. No zeroing or
    semaphore reset is required. ``split_k`` and ``up_waves`` are tuning knobs;
    defaults were measured for Qwen K=10240, R=320 with rotating layer weights.
    """
    if x.ndim != 2 or down_packed.ndim != 5 or up_packed.ndim != 5:
        raise ValueError("Expected a matrix input and 5D packed weights")
    m, k = x.shape
    r = down_packed.shape[0] * 16
    if split_k is None:
        split_k = 64 if k % 2048 == 0 else 32
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
        or not 1 <= m <= 16
        or r <= 0
        or k <= 0
        or k % (32 * split_k)
        or r % 32
        or k % 64
    ):
        raise ValueError("Unsupported HC mix input shape/dtype/layout")
    for w in (down_packed, up_packed):
        if (
            w.device != x.device or w.dtype != x.dtype
            or not w.is_contiguous() or w.data_ptr() % 16
        ):
            raise ValueError("Packed weights must match input device and dtype")
    if (
        down_packed.shape != (r // 16, k // 32, 4, 16, 8)
        or up_packed.shape != (k // 16, r // 32, 4, 16, 8)
    ):
        raise ValueError("Invalid packed HC weight shape")
    p = torch.empty((split_k, m, r), device=x.device, dtype=torch.float32)
    t = torch.empty((m, r), device=x.device, dtype=x.dtype)
    out = torch.empty((m, k // 4), device=x.device, dtype=x.dtype)
    _launch(
        x, down_packed, up_packed, p, t, out, m, k, r, split_k, up_waves,
        torch.cuda.current_stream(x.device),
    )
    return out
