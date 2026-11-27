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
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec

from . import buffer_ops
from .tensor_shim import _run_compiled

THREADS = 256
VEC = 8  # bf16 values per thread (one 16 B load)
CHUNK = THREADS * VEC  # columns per workgroup


@functools.lru_cache(maxsize=32)
def compile_moe_slots_shared_reduce(
    *, topk: int, model_dim: int, num_experts: int = 0, device: int = 0
):
    # ``device`` only keys the cache: a compiled handle is bound to the device it was built on.
    # ``num_experts`` > 0: the kernel also takes the routed ids and a slot whose id is outside [0, num_experts) (a
    # dropped route, which the routed GEMM2 never writes) adds zero instead of its buffer contents.
    assert model_dim % CHUNK == 0, f"model_dim must be a multiple of {CHUNK}"
    chunks = model_dim // CHUNK

    @flyc.kernel(
        name=f"moe_slots_shared_reduce_top{topk}_h{model_dim}"
        + (f"_e{num_experts}" if num_experts else ""),
        known_block_size=[THREADS, 1, 1],
    )
    def kernel(P: fx.Tensor, S: fx.Tensor, O: fx.Tensor, ids: fx.Tensor, m: fx.Int32):
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
        if const_expr(num_experts):
            ir = buffer_ops.create_buffer_resource(
                ids, max_size=False, num_records_bytes=m * topk * 4
            )
        acc = Vec.filled(VEC, 0.0, fx.Float32)
        for k in range_constexpr(topk):
            valid = None
            if const_expr(num_experts):
                eid = fx.Int32(
                    buffer_ops.buffer_load(
                        ir, row * topk + k, vec_width=1, dtype=fx.Int32
                    )
                )
                # a masked load reads zeros (out-of-range offset on a bounded descriptor)
                valid = (eid >= 0) & (eid < num_experts)
            v = (
                Vec(
                    buffer_ops.buffer_load(
                        pr,
                        ((row * topk + k) * model_dim + col) // 2,
                        vec_width=4,
                        dtype=fx.Int32,
                        mask=valid,
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
        P: fx.Tensor,
        S: fx.Tensor,
        O: fx.Tensor,
        ids: fx.Tensor,
        m: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(P, S, O, ids, m).launch(
            grid=(m * chunks, 1, 1), block=(THREADS, 1, 1), stream=stream
        )

    return launch


def moe_slots_shared_reduce(parts, shared, out=None, *, topk_ids=None, num_experts=0):
    """``parts`` [M, topk, model_dim] bf16 and ``shared`` [M, model_dim] bf16 -> [M, model_dim] bf16.

    ``topk_ids`` [M, topk] int32 with ``num_experts``: a slot whose id is outside ``[0, num_experts)`` (a dropped
    route: the routed GEMM2 never writes its row) contributes zero instead of whatever its buffer holds.
    """
    M, topk, model_dim = parts.shape
    assert parts.is_cuda and parts.device == shared.device
    assert parts.dtype == shared.dtype == torch.bfloat16
    # the buffer resources and element offsets are 32-bit bytes
    assert parts.numel() * 2 < 2**31, "parts must be smaller than 2 GiB"
    assert (
        parts.is_contiguous()
        and shared.is_contiguous()
        and shared.shape == (M, model_dim)
    )
    if out is None:
        out = torch.empty_like(shared)
    assert (
        out.device == parts.device
        and out.dtype == torch.bfloat16
        and out.is_contiguous()
        and out.shape == shared.shape
    )
    if topk_ids is None:
        num_experts = 0
        ids = shared  # unused by the unmasked kernel
    else:
        assert num_experts > 0, "topk_ids needs num_experts"
        assert (
            topk_ids.device == parts.device
            and topk_ids.dtype == torch.int32
            and topk_ids.is_contiguous()
            and tuple(topk_ids.shape) == (M, topk)
        )
        ids = topk_ids
    if M == 0:
        return out
    with torch.cuda.device(parts.device):
        _run_compiled(
            compile_moe_slots_shared_reduce(
                topk=topk,
                model_dim=model_dim,
                num_experts=num_experts,
                device=torch.cuda.current_device(),
            ),
            parts.view(-1),
            shared.view(-1),
            out.view(-1),
            ids.view(-1),
            M,
            torch.cuda.current_stream(parts.device),
        )
    return out
