# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""MX GEMMs on the tilescale layout (``aiter.ops.tilescale``): ``out[M, N] = A @ B^T (+ bias)``, bf16 out.

``a_fmt`` / ``b_fmt``: 4 (E2M1) or 6 (E2M3). FP6 operands are tilescale FP6 buffers (C0 then C1 planes), FP4
operands tilescale FP4 bytes in the ``b_codes`` layout (0 = "row", 1 = "k128"; A operands are "row"). Scales are
tilescale slabs, role B with interleave ``b_ilv``. Kernels are selected from the ``tsgemm`` manifest;
``tilescale_supported`` says whether a call has one (pure Python on ints, safe inside compiled regions).
"""

import csv
import functools
import os

from torch import Tensor

from ..jit.core import AITER_META_DIR, compile_ops
from .tilescale import TILESCALE_VERSION

FP4_CODES = {"row": 0, "k128": 1}


@compile_ops("module_gemm_tilescale_asm", fc_name="gemm_tilescale_asm", ffi_type="ctypes")
def _gemm_tilescale_asm(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    out: Tensor,
    K: int,
    bias: Tensor | None,
    a_fmt: int,
    b_fmt: int,
    b_codes: int,
    b_ilv: int,
) -> None: ...


@functools.lru_cache(maxsize=None)
def _rows(arch: str = "gfx950"):
    path = os.path.join(AITER_META_DIR, "hsa", arch, "tsgemm", "tsgemm_bf16_per1x32.csv")
    if not os.path.exists(path):
        return frozenset()
    out = set()
    with open(path) as f:
        for r in csv.DictReader(f):
            if int(r["ts_ver"]) != TILESCALE_VERSION:
                continue
            out.add(tuple(int(r[k]) for k in ("a_fmt", "b_fmt", "b_codes", "b_ilv", "bias", "M", "N", "K")))
    return frozenset(out)


def tilescale_supported(M: int, N: int, K: int, a_fmt: int, b_fmt: int, bias: bool = False, b_codes: int = 0,
                        b_ilv: int = 0) -> bool:
    return (a_fmt, b_fmt, b_codes, b_ilv, int(bias), M, N, K) in _rows()


def gemm_mx_tilescale(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    out: Tensor,
    a_fmt: int,
    b_fmt: int,
    K: int,
    bias: Tensor | None = None,
    b_codes: int = 0,
    b_ilv: int = 0,
) -> Tensor:
    """``out[M, N] = A @ B^T (+ bias)``; raises if the manifest has no kernel for the call."""
    if out.ndim != 2:
        raise ValueError(f"gemm_mx_tilescale expects a 2D output, got {out.ndim}D")
    _gemm_tilescale_asm(A, B, A_scale, B_scale, out, K, bias, a_fmt, b_fmt, b_codes, b_ilv)
    return out


def gemm_a6w4_tilescale(A, B, A_scale, B_scale, out, K, bias=None, b_codes=FP4_CODES["k128"]):
    return gemm_mx_tilescale(A, B, A_scale, B_scale, out, 6, 4, K, bias, b_codes, 0)


def gemm_a6w6_tilescale(A, B, A_scale, B_scale, out, K, bias=None):
    return gemm_mx_tilescale(A, B, A_scale, B_scale, out, 6, 6, K, bias, 0, 0)


def gemm_a4w4_tilescale(A, B, A_scale, B_scale, out, K, b_ilv=0):
    return gemm_mx_tilescale(A, B, A_scale, B_scale, out, 4, 4, K, None, 0, b_ilv)
