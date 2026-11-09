# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Validated BF16 dense-projection configurations and launch contract."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import range_constexpr

from aiter.jit.utils.chip_info import get_gfx
from aiter.utility import graph_alloc

from .kernels.tensor_shim import _run_compiled, ptr_arg

# (M,N,K): (BM,BN,BK,stages,M waves,N waves,split-K,stagger,unroll,
#           scheduling barrier,reverse MFMA order,stagger by N,expert scheduling)
# Shapes come from the supplied B300/MI355X dense-projection benchmark.
_PROJECTIONS = {
    (256, 3584, 7168): (128, 64, 64, 4, 2, 2, 2, 1, 2, True, True, 1, False),
    (256, 7168, 3584): (64, 128, 64, 4, 2, 2, 1, 4, 4, True, True, 0, True),
    (256, 4480, 7168): (128, 80, 128, 3, 4, 1, 2, 1, 1, False, False, 0, False),
}


def projection_config(m, n, k):
    row = _PROJECTIONS.get((m, n, k))
    if row is None:
        return None
    bm, bn, bk, stages, wm, wn, sk, stagger = row[:8]
    return {
        "block_m": bm,
        "block_n": bn,
        "block_k": bk,
        "stages": stages,
        "m_waves": wm,
        "n_waves": wn,
        "k_waves": 1,
        "split_k": sk,
        "k_stagger": stagger,
        "group_m": 0,
        "b_non_temporal": False,
        "use_half_tile_interleaved": False,
        "policy": "pr",
    }


def projection_name_supported(params):
    if (
        params["target_gfx"] != "gfx950"
        or params["dtype"] != "bf16"
        or params["out_dtype"] != "bf16"
        or params["has_bias"]
    ):
        return False
    keys = (
        "block_m",
        "block_n",
        "block_k",
        "stages",
        "m_waves",
        "n_waves",
        "k_waves",
        "split_k",
        "k_stagger",
        "group_m",
        "b_non_temporal",
    )
    return any(
        all(params[key] == projection_config(*shape)[key] for key in keys)
        for shape in _PROJECTIONS
    )


@functools.cache
def _workspace(device, stream, capture_id, m, n, splits):
    # Fully overwritten, no initialization or semaphore. Never evict: graphs
    # may retain pointers. Distinct captures can replay concurrently, even if
    # captured on the same stream; they must not share mutable partials.
    with graph_alloc.persistent_alloc(device):
        return torch.empty((splits, m, n), device=device, dtype=torch.float32)


@functools.lru_cache(maxsize=16)
def _build_reduce(m, n, splits):
    threads = 128 if n == 3584 else 256
    vec, iters = 4, 2

    @flyc.kernel
    def reduce_projection(out: fx.Pointer, partial: fx.Pointer):
        for i in range_constexpr(iters):
            idx = (
                fx.block_idx.x * threads * iters + i * threads + fx.thread_idx.x
            ) * vec
            if idx < m * n:
                value = fx.ptr_load(
                    partial + idx, result_type=fx.Vector.make_type(vec, fx.Float32)
                )
                for part in range_constexpr(1, splits):
                    value = value + fx.ptr_load(
                        partial + part * m * n + idx,
                        result_type=fx.Vector.make_type(vec, fx.Float32),
                    )
                fx.ptr_store(value.to(fx.BFloat16), out + idx)

    @flyc.jit
    def launch(out: fx.Pointer, partial: fx.Pointer, stream: fx.Stream):
        reduce_projection(out, partial).launch(
            grid=((m * n + threads * vec * iters - 1) // (threads * vec * iters),),
            block=(threads,),
            stream=stream,
        )

    return launch


def gemm_a16w16_projection(a, b, out, bias, out_dtype, stream, config):
    from .kernels.gemm_a16w16_projection_gfx950 import build_projection

    if (
        get_gfx() != "gfx950"
        or torch.cuda.get_device_properties(a.device).multi_processor_count != 256
    ):
        raise ValueError("projection HGEMM requires gfx950 with 256 CUs")
    if (
        a.ndim != 2
        or b.ndim != 2
        or a.device.type != "cuda"
        or b.device != a.device
        or a.dtype != torch.bfloat16
        or b.dtype != a.dtype
        or not a.is_contiguous()
        or not b.is_contiguous()
        or bias is not None
        or out_dtype not in (None, torch.bfloat16)
    ):
        raise ValueError(
            "projection HGEMM requires contiguous BF16 A/B and BF16 output without bias"
        )
    m, k = a.shape
    n, bk = b.shape
    expected = projection_config(m, n, k)
    if expected is None or bk != k:
        raise ValueError("unsupported projection M/N/K")
    if any(
        config.get(key) != value
        for key, value in expected.items()
        if key not in ("use_half_tile_interleaved", "policy")
    ):
        raise ValueError("unsupported projection kernel configuration")
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    if (
        out.shape != (m, n)
        or out.dtype != a.dtype
        or out.device != a.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "projection output must be contiguous BF16 [M,N] on A's device"
        )
    if any(t.data_ptr() % 16 for t in (a, b, out)):
        raise ValueError("projection A/B/output must be 16-byte aligned")
    row = _PROJECTIONS[m, n, k]
    splits = row[6]
    with torch.cuda.stream(stream):
        partial = (
            _workspace(
                a.device,
                stream,
                graph_alloc._capture_id(stream.cuda_stream),
                m,
                n,
                splits,
            )
            if splits > 1
            else out
        )
        exe = build_projection(m, n, k, *row)
        _run_compiled(
            exe,
            ptr_arg(partial, fx.Float32 if splits > 1 else fx.BFloat16),
            ptr_arg(a, fx.BFloat16),
            ptr_arg(b, fx.BFloat16),
            fx.Stream(stream),
        )
        if splits > 1:
            _run_compiled(
                _build_reduce(m, n, splits),
                ptr_arg(out, fx.BFloat16),
                ptr_arg(partial, fx.Float32),
                fx.Stream(stream),
            )
    return out
