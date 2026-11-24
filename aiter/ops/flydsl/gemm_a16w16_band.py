# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Validated band-stream decode configurations and launch contract."""

import flydsl.expr as fx
import torch

from aiter.jit.utils.chip_info import get_gfx

from .kernels.tensor_shim import _run_compiled, ptr_arg

# (max M, BM, BN, stages, M waves, N waves, K stagger, output cache policy,
# skip waves with no live rows in the final A DMA).
_BANDS = (
    (16, 16, 96, 5, 1, 2, 3, 0, False),
    (32, 32, 96, 5, 2, 2, 3, 0, False),
    (48, 48, 96, 3, 1, 3, 1, 0, True),
    (64, 64, 96, 3, 2, 2, 1, 2, True),
    (96, 96, 96, 3, 2, 2, 1, 2, False),
    (128, 128, 80, 3, 4, 1, 1, 17, False),
)


def band_stream_config(m, n, k):
    if not (0 < m <= 128 and (n, k) in ((18432, 1536), (20480, 7168))):
        return None
    row = next(row for row in _BANDS if m <= row[0])
    _, bm, bn, stages, wm, wn, stagger, _, _ = row
    return {
        "block_m": bm,
        "block_n": bn,
        "block_k": 128,
        "stages": stages,
        "split_k": 1,
        "m_waves": wm,
        "n_waves": wn,
        "k_waves": 1,
        "group_m": 0,
        "use_half_tile_interleaved": False,
        "policy": "bs",
        "k_stagger": stagger,
        "b_non_temporal": True,
    }


def band_stream_name_supported(params):
    """Name fields independent of the runtime M/N/K, for tuned-name parsing."""
    return (
        params["dtype"] == params["out_dtype"] == "bf16"
        and params["target_gfx"] == "gfx950"
        and not params["has_bias"]
        and params["block_k"] == 128
        and params["split_k"] == params["k_waves"] == 1
        and params["group_m"] == 0
        and params["b_non_temporal"]
        and any(
            (
                params["block_m"],
                params["block_n"],
                params["stages"],
                params["m_waves"],
                params["n_waves"],
            )
            == row[1:6]
            for row in _BANDS
        )
    )


def gemm_a16w16_band(a, b, out, bias, out_dtype, stream, config):
    from .kernels.gemm_a16w16_band_gfx950 import build_band_stream

    if (
        get_gfx() != "gfx950"
        or torch.cuda.get_device_properties(a.device).multi_processor_count != 256
    ):
        raise ValueError("band-stream HGEMM requires gfx950 with 256 CUs")
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
        or out_dtype not in (None, a.dtype)
    ):
        raise ValueError(
            "band-stream requires contiguous BF16 CUDA A/B, BF16 output and no bias"
        )
    m, k = a.shape
    n, bk = b.shape
    expected = band_stream_config(m, n, k)
    if expected is None or bk != k:
        raise ValueError("unsupported band-stream M/N/K")
    row = next(
        (
            row
            for row in _BANDS
            if m <= row[0]
            and (
                config["block_m"],
                config["block_n"],
                config["stages"],
                config["m_waves"],
                config["n_waves"],
            )
            == row[1:6]
        ),
        None,
    )
    if row is None:
        raise ValueError("unsupported band-stream tile for M")
    for key in ("block_k", "split_k", "k_waves", "group_m", "b_non_temporal"):
        if config.get(key) != expected[key]:
            raise ValueError(f"band-stream requires {key}={expected[key]}")
    if any(t.data_ptr() % 16 for t in (a, b)):
        raise ValueError("band-stream A/B pointers must be 16-byte aligned")
    if out is None:
        out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    if (
        out.shape != (m, n)
        or out.dtype != a.dtype
        or out.device != a.device
        or not out.is_contiguous()
        or out.data_ptr() % 16
    ):
        raise ValueError(
            "band-stream out must be aligned contiguous BF16 [M,N] on A's device"
        )
    _, bm, bn, stages, wm, wn, _, c_policy, skip_a = row
    exe = build_band_stream(
        m, n, k, bm, bn, stages, wm, wn, config["k_stagger"], c_policy, skip_a
    )
    _run_compiled(
        exe,
        ptr_arg(out, fx.BFloat16),
        ptr_arg(a, fx.BFloat16),
        ptr_arg(b, fx.BFloat16),
        fx.Stream(stream),
    )
    return out
