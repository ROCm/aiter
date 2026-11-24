# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host op for the gfx950 small-M BF16 ring GEMM (``out = x @ w^T``, 1 <= M <= 128), e.g. a vocabulary-shard
``lm_head`` where the weight is read once per call and the GEMM is HBM-bound.

Typical use for a weight that is fixed for the life of the model::

    w_ring = shuffle_weight_ring(w)            # once, before CUDA-graph capture
    ...
    if ring_supported(x.shape[0], *w.shape):
        out = gemm_a16w16_ring(x, w_ring)      # per call, graph safe
"""

import functools

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

from .kernels.gemm_a16w16_ring_gfx950 import compile_ring, fits, ring_depth
from .kernels.tensor_shim import _run_compiled

MAX_ROWS = 128
_MH1_MAX_ROWS = 64


def ring_tiling(m: int):
    """``(mt, mh, r)`` for ``m`` rows: one M half (4 waves) up to 64 rows at the deepest ring that fits (amd-kernels'
    own dispatch), two M halves (8 waves, ring depth 3) for 65-128 rows."""
    if m <= _MH1_MAX_ROWS:
        mt = (m + 15) // 16
        return mt, 1, ring_depth(mt)
    return (m + 31) // 32, 2, 3


@functools.cache
def _device_ok(index: int) -> bool:
    return (
        get_gfx_runtime() == "gfx950"
        and torch.cuda.get_device_properties(index).multi_processor_count == 256
    )


def ring_supported(m: int, n: int, k: int, device=None) -> bool:
    """Whether ``gemm_a16w16_ring`` takes this M/N/K on this GPU (gfx950, 256 CUs)."""
    if not 1 <= m <= MAX_ROWS or not torch.cuda.is_available():
        return False
    index = (
        torch.cuda.current_device() if device is None else torch.device(device).index
    )
    if index is None:
        index = torch.cuda.current_device()
    return _device_ok(index) and fits(n, k, *ring_tiling(m))


def shuffle_weight_ring(w: torch.Tensor) -> torch.Tensor:
    """The ``[N/16, K/64, 16, 64]`` tile layout of the row-major ``w [N, K]``, returned as ``[N, K]`` (a bit-exact
    permutation, new storage). ``gemm_a16w16_ring`` reads it with ``shuffled=True``."""
    n, k = w.shape
    assert w.dtype == torch.bfloat16 and n % 16 == 0 and k % 64 == 0
    return w.view(n // 16, 16, k // 64, 64).permute(0, 2, 1, 3).contiguous().view(n, k)


def unshuffle_weight_ring(ws: torch.Tensor) -> torch.Tensor:
    """The inverse of ``shuffle_weight_ring``."""
    n, k = ws.shape
    return ws.view(n // 16, k // 64, 16, 64).permute(0, 2, 1, 3).contiguous().view(n, k)


def gemm_a16w16_ring(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor | None = None,
    shuffled: bool = True,
) -> torch.Tensor:
    """``out [M, N] bf16 = x [M, K] bf16 @ w^T`` with fp32 accumulation; ``w`` is ``shuffle_weight_ring``'s layout
    (``shuffled=True``) or the row-major weight. Deterministic, no allocation when ``out`` is given, graph safe
    once the first call of each ``(M tier, N, K, layout)`` has compiled (warm up before capture).
    """
    m, k = x.shape
    n = w.shape[0]
    if not ring_supported(m, n, k, x.device):
        raise ValueError(f"ring GEMM does not take M={m} N={n} K={k} on this device")
    if (
        x.dtype != torch.bfloat16
        or w.dtype != torch.bfloat16
        or w.shape[1] != k
        or not x.is_contiguous()
        or not w.is_contiguous()
        or x.data_ptr() % 16
        or w.data_ptr() % 16
    ):
        raise ValueError(
            "ring GEMM takes contiguous 16-byte-aligned BF16 x [M, K] and w [N, K]"
        )
    if out is None:
        out = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
    elif (
        out.shape != (m, n)
        or out.dtype != torch.bfloat16
        or not out.is_contiguous()
        or out.data_ptr() % 16
    ):
        raise ValueError("out must be a contiguous 16-byte-aligned BF16 [M, N]")
    mt, mh, r = ring_tiling(m)
    exe = compile_ring(n, k, mt, mh, r, shuffled)
    _run_compiled(
        exe,
        x.data_ptr(),
        w.data_ptr(),
        out.data_ptr(),
        m,
        1,
        torch.cuda.current_stream(),
    )
    return out


def warmup_ring(n: int, k: int, device, w: torch.Tensor, shuffled: bool = True) -> None:
    """Compile every row tier (call before CUDA-graph capture)."""
    for m in (16, 32, 48, 64, 96, 128):
        if ring_supported(m, n, k, device):
            gemm_a16w16_ring(
                torch.zeros(m, k, dtype=torch.bfloat16, device=device),
                w,
                shuffled=shuffled,
            )
    torch.cuda.synchronize(device)
