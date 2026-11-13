# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Opt-in experimental loader/compute wave specialization, gfx950 only."""

import flydsl.expr as fx
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

from .kernels.tensor_shim import _run_compiled, ptr_arg


def _validate_ws_config(
    block_m,
    block_n,
    block_k,
    stages,
    producer_waves,
    m_waves,
    n_waves,
    split_k,
    prefetch,
    counter_handshake=False,
):
    params = (
        block_m,
        block_n,
        block_k,
        stages,
        producer_waves,
        m_waves,
        n_waves,
        split_k,
    )
    if any(type(x) is not int or x <= 0 for x in params):
        raise ValueError("WS configuration fields must be positive integers")
    if (
        block_k not in (64, 128, 256)
        or not 2 <= stages <= 5
        or producer_waves not in (1, 2, 4)
        or m_waves * n_waves not in (2, 4, 8)
        or producer_waves + m_waves * n_waves > 16
        or split_k > 16
        or block_m % (m_waves * 16)
        or block_n % (n_waves * 16)
    ):
        raise ValueError("unsupported WS tile/wave configuration")
    if type(prefetch) is not bool or type(counter_handshake) is not bool:
        raise ValueError("prefetch and counter_handshake must be bool")
    loaders = producer_waves * 64
    if block_m * block_k % (loaders * 8) or block_n * block_k % (loaders * 8):
        raise ValueError("loader waves must exactly cover the DMA tiles")
    if (block_m + block_n) * block_k // (loaders * 8) * (
        stages - 1 if prefetch else 1
    ) >= 63:
        raise ValueError("too many outstanding DMA operations per loader")
    lds = (
        stages * (block_m + block_n) * block_k * 2
        + stages * (2 if counter_handshake else producer_waves + m_waves * n_waves) * 4
    )
    if lds > 163840:
        raise ValueError("WS LDS capacity exceeded")
    return params


def gemm_a16w16_wave_specialized(
    a,
    weight,
    out=None,
    *,
    block_m=64,
    block_n=64,
    block_k=64,
    stages=3,
    producer_waves=1,
    m_waves=2,
    n_waves=2,
    split_k=1,
    prefetch=False,
    counter_handshake=False,
    stream=None,
):
    """BF16 A[M,K] @ weight[N,K]. Explicit path: never selected by old CSVs.

    Split-K uses per-call FP32 partials and a separate deterministic reduction.
    Initialization of ring state is inside each workgroup; no global counters.
    counter_handshake=True coalesces per-wave flags into cumulative LDS counters.
    It is experimental and does not imply a speedup over the default GEMM path.
    """
    from .kernels.gemm_a16w16_ws_gfx950 import build_ws

    if get_gfx_runtime() != "gfx950":
        raise ValueError("wave-specialized HGEMM requires gfx950")
    params = _validate_ws_config(
        block_m,
        block_n,
        block_k,
        stages,
        producer_waves,
        m_waves,
        n_waves,
        split_k,
        prefetch,
        counter_handshake,
    )
    if (
        a.ndim != 2
        or weight.ndim != 2
        or a.device.type != "cuda"
        or weight.device != a.device
        or a.dtype != torch.bfloat16
        or weight.dtype != a.dtype
    ):
        raise ValueError("WS requires BF16 CUDA matrices on the same device")
    m, k = a.shape
    n, wk = weight.shape
    if min(m, n, k) <= 0 or wk != k or k % (split_k * block_k) or n % 4:
        raise ValueError("WS requires positive MNK, N%4=0 and K%(split_k*block_k)=0")
    if (
        min(a.stride(0), weight.stride(0)) <= 0
        or a.stride(1) != 1
        or weight.stride(1) != 1
        or a.stride(0) * 2 % 16
        or weight.stride(0) * 2 % 16
        or a.data_ptr() % 16
        or weight.data_ptr() % 16
    ):
        raise ValueError("WS requires DMA-aligned row-major A and weight")
    if (
        max(
            ((m - 1) * a.stride(0) + k) * 2,
            ((n - 1) * weight.stride(0) + k) * 2,
            split_k * m * n * 4,
        )
        >= 1 << 31
    ):
        raise ValueError("WS spans must fit signed i32 bytes")
    if stream is None:
        stream = torch.cuda.current_stream(a.device)
    if stream.device != a.device:
        raise ValueError("WS launch stream must match device")
    with torch.cuda.stream(stream):
        if out is None:
            out = torch.empty((m, n), device=a.device, dtype=a.dtype)
        if (
            out.shape != (m, n)
            or out.device != a.device
            or out.dtype != a.dtype
            or not out.is_contiguous()
            or out.data_ptr() % 8
        ):
            raise ValueError(
                "WS out must be aligned contiguous BF16 [M,N] on the launch device"
            )
        partial = (
            torch.empty((split_k, m, n), device=a.device, dtype=torch.float32)
            if split_k > 1
            else out
        )
        exe = build_ws(*params, prefetch, counter_handshake)
        _run_compiled(
            exe,
            ptr_arg(out, fx.BFloat16),
            ptr_arg(a, fx.BFloat16),
            ptr_arg(weight, fx.BFloat16),
            ptr_arg(partial, fx.Float32),
            m,
            n,
            k,
            a.stride(0),
            weight.stride(0),
            fx.Stream(stream),
            specialization_key=(a.device.index, a.dtype, out.dtype),
        )
    return out
