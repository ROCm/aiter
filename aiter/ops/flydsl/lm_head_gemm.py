# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL bf16 lm_head GEMM for a 129280 x 5120 vocabulary head on gfx950.

``out [M, 129280] = x [M, 5120] @ weight^T``: bf16 operands, fp32 sums, one
bf16 rounding, no bias, no split-K, no output atomics. The weight is the full
(replicated) lm_head of a rank, 1.32 GB, read once per call. M is any row
count (calls above 2048 rows run as launches of 2048 rows).

Shape specialization is explicit: :func:`lm_head_gemm_unsupported_reason`
names why a call is not served (anything but a contiguous [129280, 5120] bf16
weight without bias on gfx950), so a caller falls back to its own GEMM.
:func:`lm_head_gemm_preferred` is the measured dispatch rule: the row counts
at which this kernel beat the stock path (hipBLASLt / vLLM's skinny GEMMs) on
MI355X with the row-major weight. Outside them the stock GEMM is as fast or
faster.

Weight layouts: the checkpoint's row-major weight, or :func:`shuffle_weight`'s
load-time layout ``[8080, 80, 16, 64]`` (one permutation, bit-exact, both give
the same output bits). The shuffled layout is only useful to a caller that
sends every M to this kernel (the stock GEMM cannot read it).
"""

import functools

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

from .kernels.gemm_bf16_n129280_k5120 import (
    NB,
    K,
    N,
    compile_large,
    compile_small,
)
from .kernels.tensor_shim import _run_compiled

MAX_M = 2048  # rows of one launch
SMALL_MAX = 64  # rows up to which one four-wave workgroup covers every row
ROT_MAX = 256  # rows up to which vocabulary tiles start their K loops apart
# Eight-wave tiles of 32 mt rows, in the order a tie is broken.
LARGE_MTS = (8, 6, 5, 7, 4, 3)
SUPPORTED_GFX = ("gfx950",)
# Row ranges (inclusive) where this kernel measured faster than the stock
# bf16 GEMM path (hipBLASLt) on MI355X with the row-major weight, 1.04-1.10x
# (op_tests/op_benchmarks/flydsl/bench_lm_head_markov.py).
PREFERRED_ROWS = ((113, 128), (481, 512))


def m_tile_fragments(m: int) -> int:
    """16-row fragments per wave. Up to 64 rows four waves take every row;
    above that, eight waves take two M halves and the M tile with the fewest
    padded rows wins."""
    if m <= SMALL_MAX:
        return (m + 15) // 16
    return min(
        LARGE_MTS,
        key=lambda mt: (-(-m // (32 * mt)) * 32 * mt, LARGE_MTS.index(mt)),
    )


def k_starts(m: int) -> int:
    """Distinct first K blocks of the eight-wave launch's vocabulary tiles: all
    80 while the activations fit L2, else 1 (lockstep)."""
    return NB if m <= ROT_MAX else 1


def m_rows(m: int) -> int:
    """Rows of one M tile."""
    return 16 * m_tile_fragments(m) * (1 if m <= SMALL_MAX else 2)


@functools.cache
def _kernel(mt: int, shuf: bool, large: bool, starts: int):
    if large:
        return compile_large(mt, int(shuf), starts)
    return compile_small(mt, int(shuf))


def _variant(m: int, shuffled: bool):
    large = m > SMALL_MAX
    return (m_tile_fragments(m), bool(shuffled), large, k_starts(m) if large else 0)


def lm_head_gemm_unsupported_reason(
    weight: torch.Tensor, bias: torch.Tensor | None = None
) -> str | None:
    """Why :func:`flydsl_lm_head_gemm` cannot serve this weight, or None.

    Fixed once the weights are loaded: a caller can ask once and keep the
    answer. The activation only has to be a contiguous bf16 [M, 5120] tensor
    on the weight's device (checked per call).
    """
    if bias is not None:
        return "a bias is not supported"
    if tuple(weight.shape) != (N, K):
        return f"weight must be [{N}, {K}], got {tuple(weight.shape)}"
    if weight.dtype != torch.bfloat16:
        return f"weight must be bf16, got {weight.dtype}"
    if not weight.is_contiguous():
        return "weight must be contiguous"
    if not weight.is_cuda:
        return "weight must be on a GPU"
    gfx = get_gfx_runtime()
    if gfx not in SUPPORTED_GFX:
        return f"{gfx} is not supported (gfx950 only)"
    return None


def lm_head_gemm_supported(
    weight: torch.Tensor, bias: torch.Tensor | None = None
) -> bool:
    """Whether :func:`flydsl_lm_head_gemm` serves this weight (and bias)."""
    return lm_head_gemm_unsupported_reason(weight, bias) is None


def lm_head_gemm_preferred(m: int) -> bool:
    """The measured dispatch rule: True for the row counts at which this kernel
    beat the stock bf16 GEMM path on MI355X (row-major weight)."""
    return any(lo <= m <= hi for lo, hi in PREFERRED_ROWS)


def shuffle_weight(w: torch.Tensor) -> torch.Tensor:
    """The load-time layout [8080 groups, 80 K blocks, 16 rows, 64 K] of the
    [129280, 5120] weight, returned as [129280, 5120]. A permutation: replace
    the parameter once, before graph capture, and pass ``shuffled=True``."""
    assert w.shape == (N, K) and w.dtype == torch.bfloat16
    return w.view(N // 16, 16, K // 64, 64).permute(0, 2, 1, 3).contiguous().view(N, K)


def unshuffle_weight(ws: torch.Tensor) -> torch.Tensor:
    """The inverse of :func:`shuffle_weight` (row-major [129280, 5120])."""
    assert ws.shape == (N, K) and ws.dtype == torch.bfloat16
    return ws.view(N // 16, K // 64, 16, 64).permute(0, 2, 1, 3).contiguous().view(N, K)


def flydsl_lm_head_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    shuffled: bool = False,
) -> torch.Tensor:
    """``out [M, 129280] bf16 = x [M, 5120] bf16 @ weight^T`` (fp32 accumulate).

    ``weight`` is row-major or :func:`shuffle_weight`'s layout (``shuffled``).
    Deterministic; a row's bits do not depend on M within one kernel variant
    (up to 64 rows, 65-256 rows, 257 rows and above). CUDA-graph safe: no host
    sync, no allocation when ``out`` is given. Compile every variant before
    graph capture with :func:`warmup_lm_head_gemm`.
    """
    reason = lm_head_gemm_unsupported_reason(weight)
    if reason is not None:
        raise ValueError(f"[FlyDSL lm_head_gemm] {reason}")
    m = x.shape[0]
    assert x.ndim == 2 and x.shape[1] == K and x.dtype == torch.bfloat16
    assert x.is_contiguous() and x.device == weight.device
    if out is None:
        out = torch.empty(m, N, dtype=torch.bfloat16, device=x.device)
    assert out.shape == (m, N) and out.dtype == torch.bfloat16
    assert out.is_contiguous() and out.device == x.device
    if m > MAX_M:
        for i in range(0, m, MAX_M):
            flydsl_lm_head_gemm(
                x[i : i + MAX_M], weight, out[i : i + MAX_M], shuffled=shuffled
            )
        return out
    if m:
        _run_compiled(
            _kernel(*_variant(m, shuffled)),
            x.data_ptr(),
            weight.data_ptr(),
            out.data_ptr(),
            m,
            -(-m // m_rows(m)),
            torch.cuda.current_stream(x.device),
            specialization_key=x.device.index,
        )
    return out


def _warmup_rows() -> list[int]:
    """One row count per kernel variant up to MAX_M rows."""
    first = {}
    for m in range(1, MAX_M + 1):
        first.setdefault(_variant(m, False), m)
    return sorted(first.values())


# one row count per kernel variant; every other row count reuses one of these
WARMUP_ROWS = _warmup_rows()


def warmup_lm_head_gemm(
    device: torch.device, shuffled: bool = False, rows=None
) -> None:
    """Compile (and run once) the variants of ``rows`` (default: every
    variant) for one weight layout, before graph capture."""
    rows = WARMUP_ROWS if rows is None else sorted({_variant_row(m) for m in rows})
    w = torch.zeros(N, K, dtype=torch.bfloat16, device=device)
    x = torch.zeros(max(rows), K, dtype=torch.bfloat16, device=device)
    for m in rows:
        flydsl_lm_head_gemm(x[:m], w, shuffled=shuffled)
    torch.cuda.synchronize(device)


def _variant_row(m: int) -> int:
    """The WARMUP_ROWS entry that compiles the variant of ``m`` rows."""
    key = _variant(min(m, MAX_M), False)
    return next(r for r in WARMUP_ROWS if _variant(r, False) == key)
