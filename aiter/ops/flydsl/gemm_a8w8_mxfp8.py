# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL mxfp8 GEMM on a (16, 16)-preshuffled weight, on whichever arch has a
kernel for the operands' w_scale block and scale layouts -- the 2-D counterpart
of batched_gemm_a8w8.

``XQ [M, K]`` fp8, ``WQ [N, K]`` fp8 preshuffled, ``x_scale`` / ``w_scale``
e8m0, ``Out [M, N]`` bf16 / fp16. Each scale's layout is read off its rank: a
2-D scale is the unshuffled row-major one ("row": ``x_scale [M, K / 32]``,
``w_scale [N / WN, K / WK]``), a 1-D one a pre-shuffled flat buffer
("mfma_tile"). The w_scale block is read off a row w_scale's shape. The per-arch
modules hold the kernels; a tuned kernelName names one of them, so it is only
valid on its arch.
"""

from __future__ import annotations

import functools
import importlib
from collections.abc import Callable

from torch import Tensor

from aiter.jit.utils.chip_info import get_gfx

from ..gemm_op_common import mxscale_w_scale_block

_GFX950 = ("mxscale_preshuffle_kernels", "run_gemm_a8w8_mxfp8_gfx950")
# (gfx, w_scale block, x_scale layout, w_scale layout) -> (module, runner, its
# layout arguments). Only the running arch's module is ever imported.
_RUNNERS = {
    ("gfx950", "1x32", "row", "row"): (*_GFX950, {}),
}


@functools.cache
def _runner(
    gfx: str, w_scale_block: str, x_layout: str, w_layout: str
) -> Callable | None:
    entry = _RUNNERS.get((gfx, w_scale_block, x_layout, w_layout))
    if entry is None:
        return None
    module, name, layout = entry
    run = getattr(importlib.import_module(f".{module}", __package__), name)
    return functools.partial(run, **layout) if layout else run


def _scale_layout(scale: Tensor) -> str:
    return "row" if scale.dim() == 2 else "mfma_tile"


def gemm_a8w8_mxfp8_supported(
    w_scale_block: str, x_layout: str = "row", w_layout: str = "row"
) -> bool:
    """Whether this arch has an mxfp8 GEMM kernel for a w_scale block and the
    two scale layouts."""
    return _runner(get_gfx(), w_scale_block, x_layout, w_layout) is not None


def run_gemm_a8w8_mxfp8(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    kernel_name: str | None = None,
) -> Tensor:
    """``Out = dequant(XQ) @ dequant(WQ).T`` on this arch's kernel for the
    w_scale block and the two scale layouts; writes into ``Out`` and returns it.
    ``kernel_name`` is a tuned row's kernelName, else the arch's heuristic picks
    one."""
    n, k = WQ.shape[-2], WQ.shape[-1]
    x_layout, w_layout = _scale_layout(x_scale), _scale_layout(w_scale)
    if w_layout != "row":
        # A flat buffer's block is not readable off its shape.
        raise NotImplementedError(
            f"no preshuffled mxfp8 GEMM kernel reads a pre-shuffled (1-D) w_scale "
            f"on {get_gfx()}; pass the unshuffled 2-D [N / WN, K / WK] scale"
        )
    w_scale_block = mxscale_w_scale_block(tuple(w_scale.shape), n, k)
    run = _runner(get_gfx(), w_scale_block, x_layout, w_layout)
    if run is None:
        raise NotImplementedError(
            f"no preshuffled mxfp8 GEMM kernel reads a {w_scale_block} w_scale "
            f"with {x_layout} x_scale / {w_layout} w_scale layouts on {get_gfx()}"
        )
    return run(XQ, WQ, x_scale, w_scale, Out, kernel_name=kernel_name)


__all__ = ["gemm_a8w8_mxfp8_supported", "run_gemm_a8w8_mxfp8"]
