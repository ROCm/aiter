# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Technique source: amd-kernels moe/mxfp4_h6144_i768_e128_top4/common/reduce_slots_shared.py

"""MoE routed-slot reduction plus the shared expert, in one pass (FlyDSL).

``out[t, :] = bf16(sum_k fp32(parts[t, k, :]) + fp32(shared[t, :]))`` with the slots accumulated in order
``0 .. topk-1`` and the shared expert added last, all in fp32.  ``parts`` are the per-(token, slot) route rows of a
route-reduce stage2 (``[M, topk, model_dim]`` bf16), ``shared`` the shared expert's ``[M, model_dim]`` bf16 output.
Same bits as ``compile_moe_reduction`` over a ``[M, topk + 1, model_dim]`` buffer whose last slot is the shared
output, without needing the shared expert written into the routed buffer.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Vector as Vec

from . import buffer_ops
from .tensor_shim import _run_compiled

THREADS = 256
VEC = 8  # bf16 values per thread (one 16 B load)
CHUNK = THREADS * VEC  # columns per workgroup


@functools.lru_cache(maxsize=32)
def compile_moe_slots_shared_reduce(*, topk: int, model_dim: int):
    assert model_dim % CHUNK == 0, f"model_dim must be a multiple of {CHUNK}"
    chunks = model_dim // CHUNK

    @flyc.kernel(
        name=f"moe_slots_shared_reduce_top{topk}_h{model_dim}",
        known_block_size=[THREADS, 1, 1],
    )
    def kernel(P: fx.Tensor, S: fx.Tensor, O: fx.Tensor, m: fx.Int32):
        tx = fx.thread_idx.x
        row, chunk = fx.block_idx.x // chunks, fx.block_idx.x % chunks
        col = chunk * CHUNK + tx * VEC
        pr = buffer_ops.create_buffer_resource(
            P, max_size=False, num_records_bytes=m * topk * model_dim * 2
        )
        sr = buffer_ops.create_buffer_resource(
            S, max_size=False, num_records_bytes=m * model_dim * 2
        )
        out = buffer_ops.create_buffer_resource(
            O, max_size=False, num_records_bytes=m * model_dim * 2
        )
        acc = Vec.filled(VEC, 0.0, fx.Float32)
        for k in range_constexpr(topk):
            v = (
                Vec(
                    buffer_ops.buffer_load(
                        pr,
                        ((row * topk + k) * model_dim + col) // 2,
                        vec_width=4,
                        dtype=fx.Int32,
                    )
                )
                .bitcast(fx.BFloat16)
                .to(fx.Float32)
            )
            acc = acc + v
        sh = (
            Vec(
                buffer_ops.buffer_load(
                    sr, (row * model_dim + col) // 2, vec_width=4, dtype=fx.Int32
                )
            )
            .bitcast(fx.BFloat16)
            .to(fx.Float32)
        )
        buffer_ops.buffer_store(
            (acc + sh).to(fx.BFloat16).bitcast(fx.Int32),
            out,
            (row * model_dim + col) // 2,
        )

    @flyc.jit
    def launch(
        P: fx.Tensor, S: fx.Tensor, O: fx.Tensor, m: fx.Int32, stream: fx.Stream
    ):
        kernel(P, S, O, m).launch(
            grid=(m * chunks, 1, 1), block=(THREADS, 1, 1), stream=stream
        )

    return launch


def moe_slots_shared_reduce(parts, shared, out=None):
    """``parts`` [M, topk, model_dim] bf16 and ``shared`` [M, model_dim] bf16 -> [M, model_dim] bf16."""
    M, topk, model_dim = parts.shape
    assert parts.dtype == shared.dtype == torch.bfloat16
    assert (
        parts.is_contiguous()
        and shared.is_contiguous()
        and shared.shape == (M, model_dim)
    )
    if out is None:
        out = torch.empty_like(shared)
    assert (
        out.dtype == torch.bfloat16
        and out.is_contiguous()
        and out.shape == shared.shape
    )
    if M == 0:
        return out
    _run_compiled(
        compile_moe_slots_shared_reduce(topk=topk, model_dim=model_dim),
        parts.view(-1),
        shared.view(-1),
        out.view(-1),
        M,
        torch.cuda.current_stream(parts.device),
    )
    return out
