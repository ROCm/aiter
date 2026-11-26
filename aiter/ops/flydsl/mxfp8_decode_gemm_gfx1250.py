# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Decode-shaped gfx1250 MXFP8 1x32 GEMM (kernels/mxfp8_decode_gemm_gfx1250.py) as
a tuned kernel.

A tuned row selects it by kernelName (``flydsl_mxfp8_32_bpreshuffle_decode_*``)
and the B = 1 batched runner launches it, as it does the streaming kernel. The
name carries the tile: ``t`` rows a workgroup, ``nb`` 16-column blocks, ``ks``
split-K workgroups, ``kw`` K-waves, ``wn`` N-waves and ``d`` K steps of weight
loads in flight. Weight loads are non-temporal and the activation goes to LDS by
TDM. The values differ from the other kernels' only by the FP32 K order.
"""

from __future__ import annotations

import functools
import re

import torch

from aiter.utility.graph_alloc import capture_scoped_cache, persistent_alloc

MX32_DECODE_NAME_PREFIX = "flydsl_mxfp8_32_bpreshuffle_decode"
_KERNEL_NAME_RE = re.compile(
    rf"^{re.escape(MX32_DECODE_NAME_PREFIX)}_t(?P<tile_m>\d+)_nb(?P<nb>\d+)"
    r"_ks(?P<ks>\d+)_kw(?P<kw>\d+)_wn(?P<wn>\d+)_d(?P<wdepth>\d+)(?P<kint>_ki)?$"
)
# Tuner candidates: decode token counts, and at most 16 K-waves (with one 128-wide
# K step each, every weight load of a workgroup is in flight at once).
_CANDIDATE_MAX_M = 128
_MAX_K_WAVES = 16


def mxfp8_32_decode_kernel_name(tile_m, nb, ks, kw, wn, wdepth, kint=0) -> str:
    """kint (``_ki``): a K-wave takes every kw-th K step instead of a contiguous run."""
    return (
        f"{MX32_DECODE_NAME_PREFIX}_t{tile_m}_nb{nb}_ks{ks}_kw{kw}_wn{wn}_d{wdepth}"
        f"{'_ki' if kint else ''}"
    )


def parse_mxfp8_32_decode_kernel_name(name: str) -> dict | None:
    m = _KERNEL_NAME_RE.fullmatch(name)
    if m is None:
        return None
    cfg = {key: int(v) for key, v in m.groupdict().items() if key != "kint"}
    cfg["kint"] = int(m.group("kint") is not None)
    return cfg


@functools.lru_cache(maxsize=4096)
def mxfp8_32_decode_kernel_launchable(kernel_name: str, M: int, N: int, K: int) -> bool:
    cfg = parse_mxfp8_32_decode_kernel_name(kernel_name)
    if cfg is None or M <= 0 or N % 32 or K % 128:
        return False
    from .kernels.mxfp8_decode_gemm_gfx1250 import decode_gemm_config_error

    return decode_gemm_config_error(N, K, **cfg, atdm=1) is None


def mxfp8_32_decode_candidates(M: int, N: int, K: int) -> list[str]:
    """Decode kernelNames the tuner times: 16- or 32-row tiles, 1 to 8 column
    blocks, the most K-waves (up to 16) that split the K steps evenly, and one
    K step or all of a wave's K steps of weight loads in flight; plus 8 K-waves
    with a 2- or 4-workgroup cluster split of K and one or two n-waves, which
    put more workgroups and loads in flight at small N or larger M."""
    if not 0 < M <= _CANDIDATE_MAX_M or N % 32 or K % 128:
        return []
    steps = K // 128
    kw = max(w for w in range(1, _MAX_K_WAVES + 1) if steps % w == 0)
    names = [
        mxfp8_32_decode_kernel_name(tile_m, nb, 1, kw, 1, wdepth)
        for tile_m in (16, 32)
        for nb in (1, 2, 3, 4, 6, 8)
        for wdepth in sorted({1, steps // kw})
    ]
    names += [
        mxfp8_32_decode_kernel_name(tile_m, nb, ks, 8, wn, wdepth)
        for tile_m in (16, 32)
        for nb in (2, 4, 8)
        for ks in (1, 2, 4)
        for wn in (1, 2)
        if steps % (8 * ks) == 0
        for wdepth in sorted({1, steps // (8 * ks)})
    ]

    # Interleaved K steps (``_ki``) wherever a wave owns two or more steps: the
    # contiguous runs of the K-waves start a power of two apart when K is one.
    def wave_steps(name):
        cfg = parse_mxfp8_32_decode_kernel_name(name)
        return steps // (cfg["ks"] * cfg["kw"])

    names += [f"{name}_ki" for name in names if wave_steps(name) > 1]
    names = list(dict.fromkeys(names))
    return [name for name in names if mxfp8_32_decode_kernel_launchable(name, M, N, K)]


@functools.cache
def _launcher():
    from .kernels.mxfp8_decode_gemm_gfx1250 import launch_mxfp8_decode_gemm

    return launch_mxfp8_decode_gemm


@capture_scoped_cache
def _slab_cached(device: torch.device, stream_id: int, numel: int) -> torch.Tensor:
    """FP32 split partials, one per (device, stream); a capture gets its own."""
    del stream_id
    with persistent_alloc(device):
        return torch.empty(numel, dtype=torch.float32, device=device)


def _slab(device, numel):
    bucket = 1 << max(20, (numel - 1).bit_length())
    return _slab_cached(device, torch.cuda.current_stream(device).cuda_stream, bucket)


def run_gemm_a8w8_mxfp8_32_decode_gfx1250(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    kernel_name: str,
) -> torch.Tensor:
    """Out [M, N] bf16 = XQ [M, K] (e4m3, unit column stride, 16-byte row stride)
    with x_scale [M, K / 32] (e8m0, row-major) times WQ [N, K] ((16, 16)-preshuffled
    e4m3) with w_scale [N, K / 32] (32x4-shuffled e8m0)."""
    import flydsl.expr as fx

    from .kernels.tensor_shim import _run_compiled, ptr_arg

    m, k = XQ.shape
    n = WQ.shape[0]
    if not mxfp8_32_decode_kernel_launchable(kernel_name, max(m, 1), n, k):
        raise ValueError(
            f"[FlyDSL gfx1250 decode] {kernel_name!r} cannot run N={n} K={k}"
        )
    cfg = parse_mxfp8_32_decode_kernel_name(kernel_name)
    if (
        XQ.element_size() != 1
        or tuple(WQ.shape) != (n, k)
        or XQ.stride(1) != 1
        or XQ.stride(0) % 16
        or tuple(x_scale.shape) != (m, k // 32)
        or not x_scale.is_contiguous()
        or tuple(w_scale.shape) != (n, k // 32)
        or not (WQ.is_contiguous() and w_scale.is_contiguous())
        or Out.dtype != torch.bfloat16
        or tuple(Out.shape) != (m, n)
        or not Out.is_contiguous()
    ):
        raise ValueError(
            "[FlyDSL gfx1250 decode] needs an fp8 A with a 16-byte row stride, "
            "contiguous B and scales, and a contiguous bf16 C"
        )
    if m == 0:
        return Out
    tile_m, ks = cfg["tile_m"], cfg["ks"]
    mpad = (m + tile_m - 1) // tile_m * tile_m
    slab = _slab(XQ.device, ks * mpad * n) if ks > 1 else Out
    stream = torch.cuda.current_stream(XQ.device)
    # (N, K, tile_m, nb, ks, kw, wn, wdepth, wcpol, atdm, kint)
    spec = (
        n,
        k,
        tile_m,
        cfg["nb"],
        ks,
        cfg["kw"],
        cfg["wn"],
        cfg["wdepth"],
        1,
        1,
        cfg["kint"],
    )
    _run_compiled(
        _launcher(),
        ptr_arg(XQ), ptr_arg(x_scale), ptr_arg(WQ), ptr_arg(w_scale), ptr_arg(Out), ptr_arg(slab),
        m, XQ.stride(0), mpad, fx.Stream(stream),
        *spec,
        specialization_key=("mxfp8_32_decode", XQ.device.index) + spec,
    )  # fmt: skip
    return Out
