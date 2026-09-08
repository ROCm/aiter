# SPDX-License-Identifier: Apache-2.0
"""Two-pass row softmax for a few very long rows on gfx950.

Pass 1 computes a stable (maximum, exponential sum) per chunk. Pass 2
combines those statistics and independently normalizes every chunk. The
scratch buffer is FP32 and allocated once, outside steady-state launches.
Each launcher owns one workspace; concurrent streams need separate launchers.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import arith, const_expr, gpu, range_constexpr

from aiter.jit.utils.chip_info import get_gfx


def build_softmax_split(M, N, dtype_str="bf16", *, device=None):
    """Build an opt-in long-row launcher: ``launch(x, out, stream=None)``.

    Requires contiguous, non-overlapping finite input/output, 1 <= M <= 4,
    32768 <= N <= 262144. Allocate one launcher per concurrent stream.
    The launcher owns its workspace and must outlive captured graphs.
    """
    CHUNK = 4096
    if not 1 <= M <= 4 or not 32768 <= N <= CHUNK * 64:
        raise ValueError("split softmax requires 1 <= M <= 4 and 32768 <= N <= 262144")
    if dtype_str not in ("bf16", "f16", "f32"):
        raise ValueError("dtype_str must be bf16, f16, or f32")
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if device is None
        else torch.device(device)
    )
    if device.type != "cuda":
        raise ValueError("split softmax requires a CUDA/HIP device")
    with torch.cuda.device(device):
        if get_gfx() != "gfx950":
            raise ValueError("split softmax currently supports gfx950 only")
    chunks = (N + CHUNK - 1) // CHUNK
    elem = {"bf16": fx.BFloat16, "f16": fx.Float16, "f32": fx.Float32}[dtype_str]
    scratch = torch.empty((M, chunks, 2), device=device, dtype=torch.float32)

    @flyc.kernel
    def partials(X: fx.Tensor, P: fx.Tensor):
        tid = fx.thread_idx.x
        wave = tid // 64
        lane = tid % 64
        block = fx.block_idx.x
        row = block // chunks
        chunk = block % chunks
        xb = fx.rocdl.make_buffer_tensor(X)
        pb = fx.rocdl.make_buffer_tensor(P)

        @fx.struct
        class Storage:
            values: fx.Array[fx.Float32, 8, 16]

        sm = (
            fx.SharedAllocator()
            .allocate(Storage)
            .peek()
            .values.view(fx.make_layout(8, 1))
        )

        def reduce(v, maximum, width):
            for i in range_constexpr(int(math.log2(width))):
                peer = v.shuffle_xor(width // (2 << i), width)
                if const_expr(maximum):
                    v = fx.max(v, peer)
                else:
                    v = v + peer
            return v

        vals = []
        mx = fx.Float32(float("-inf"))
        for i in range_constexpr(CHUNK // 256):
            col = chunk * CHUNK + tid + i * 256
            valid = col < N
            v = xb[row, valid.select(col, 0)].to(fx.Float32)
            v = valid.select(v, fx.Float32(float("-inf")))
            vals.append(v)
            mx = fx.max(mx, v)
        mx = reduce(mx, True, 64)
        if lane == 0:
            fx.memref_store(mx, sm, wave)
        gpu.barrier()
        mx = reduce(fx.memref_load(sm, lane % 4), True, 4)
        total = fx.Float32(0.0)
        for v in vals:
            total = total + ((v - mx) * 1.4426950408889634).exp2(
                fastmath=arith.FastMathFlags.fast
            )
        total = reduce(total, False, 64)
        if lane == 0:
            fx.memref_store(total, sm, 4 + wave)
        gpu.barrier()
        total = reduce(fx.memref_load(sm, 4 + lane % 4), False, 4)
        if tid == 0:
            pb[row, chunk, 0] = mx
            pb[row, chunk, 1] = total

    @flyc.kernel
    def normalize(X: fx.Tensor, P: fx.Tensor, Y: fx.Tensor):
        tid = fx.thread_idx.x
        lane = tid % 64
        block = fx.block_idx.x
        row = block // chunks
        chunk = block % chunks
        xb = fx.rocdl.make_buffer_tensor(X)
        pb = fx.rocdl.make_buffer_tensor(P)
        yb = fx.rocdl.make_buffer_tensor(Y)
        valid = lane < chunks
        stat_idx = valid.select(lane, 0)
        local_max = pb[row, stat_idx, 0]
        local_sum = pb[row, stat_idx, 1]
        mx = valid.select(local_max, fx.Float32(float("-inf")))
        for i in range_constexpr(6):
            mx = fx.max(mx, mx.shuffle_xor(32 >> i, 64))
        weighted = local_sum * ((local_max - mx) * 1.4426950408889634).exp2(
            fastmath=arith.FastMathFlags.fast
        )
        total = valid.select(weighted, fx.Float32(0.0))
        for i in range_constexpr(6):
            total = total + total.shuffle_xor(32 >> i, 64)
        inv = 1.0 / total
        for i in range_constexpr(CHUNK // 256):
            col = chunk * CHUNK + tid + i * 256
            if col < N:
                x = xb[row, col].to(fx.Float32)
                value = ((x - mx) * 1.4426950408889634).exp2(
                    fastmath=arith.FastMathFlags.fast
                ) * inv
                yb[row, col] = value.to(elem)

    @flyc.jit
    def launch(
        X: fx.Tensor,
        P: fx.Tensor,
        Y: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008 - FlyDSL stream annotation
    ):
        partials(X, P).launch(grid=(M * chunks, 1, 1), block=(256, 1, 1), stream=stream)
        normalize(X, P, Y).launch(
            grid=(M * chunks, 1, 1), block=(256, 1, 1), stream=stream
        )

    dtype = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}[
        dtype_str
    ]

    def call(x, out, stream=None):
        for tensor in (x, out):
            if (
                tensor.shape != (M, N)
                or tensor.dtype != dtype
                or tensor.device != scratch.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    "split softmax requires matching contiguous tensors on the launcher's device"
                )
        if x.untyped_storage().data_ptr() == out.untyped_storage().data_ptr():
            raise ValueError("split softmax does not support input/output aliasing")
        with torch.cuda.device(scratch.device):
            stream = torch.cuda.current_stream() if stream is None else stream
            if stream.device != scratch.device:
                raise ValueError("stream must belong to the launcher's device")
            launch(x, scratch, out, stream=stream)
        return out

    return call
