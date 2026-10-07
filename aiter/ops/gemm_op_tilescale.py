# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""MX GEMMs on the tilescale layout (``aiter.ops.tilescale``): ``out[M, N] = A @ B^T (+ bias)``, bf16 out.

``a_fmt`` / ``b_fmt``: 4 (E2M1) or 6 (E2M3). FP6 operands are tilescale FP6 buffers (C0 then C1 planes), FP4
operands tilescale FP4 bytes in the ``b_codes`` layout (0 = "row", 1 = "k128", 2 = "kouter" with its K256-outer
scale slab; A operands are "row"). Scales are tilescale slabs, role B with interleave ``b_ilv``. Kernels are selected from the ``tsgemm`` manifest;
``tilescale_supported`` says whether a call has one (pure Python on ints, safe inside compiled regions).

A4W4 rows may also emit attention's softmax_d (the ``fmha_v3_bwd`` ``softmax_d`` input) from the GEMM: with ``out``
the attention output's gradient dO in sbhd rows (row = s * B + b) and ``epi_o`` the attention output O in the same
layout, the kernel adds, per row and 128-column head, the fp32 sum of out * O into ``epi_delta`` [B, N/128, S]
(zeroed by the caller) from sequence position ``epi_s0`` on. The sum order is the kernel's own, not the odo
kernel's. ``tilescale_softmax_d_supported`` says whether a kernel exists.
"""

import csv
import functools
import os

from torch import Tensor

from ..jit.core import AITER_META_DIR, compile_ops
from .tilescale import TILESCALE_VERSION

FP4_CODES = {"row": 0, "k128": 1, "kouter": 2}


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
    B_c1: Tensor | None,
    epi_o: Tensor | None,
    epi_delta: Tensor | None,
    epi_s0: int,
) -> None: ...


@functools.lru_cache(maxsize=None)
def _manifest(arch: str = "gfx950"):
    path = os.path.join(AITER_META_DIR, "hsa", arch, "tsgemm", "tsgemm_bf16_per1x32.csv")
    if not os.path.exists(path):
        return ()
    with open(path) as f:
        return tuple(
            {k: (v if k in ("knl_name", "co_name") else int(v or 0)) for k, v in r.items()}
            for r in csv.DictReader(f)
            if int(r["ts_ver"]) == TILESCALE_VERSION
        )


@functools.lru_cache(maxsize=None)
def _generic(arch: str = "gfx950"):
    """{(a_fmt, b_fmt, b_codes, b_ilv, bias, kcls): kmin} of the shape-generic rows."""
    return {(r["a_fmt"], r["b_fmt"], r["b_codes"], r["b_ilv"], r["bias"], r["kcls"]): r["kmin"]
            for r in _manifest(arch) if r["abi"] in (2, 3)}


def _key(r):
    return tuple(r.get(k, 0) for k in ("a_fmt", "b_fmt", "b_codes", "b_ilv", "bias", "M", "N", "K", "epi_b", "epi_s"))


@functools.lru_cache(maxsize=None)
def _rows(arch: str = "gfx950"):
    return frozenset(_key(r) for r in _manifest(arch))


def tilescale_supported(M: int, N: int, K: int, a_fmt: int, b_fmt: int, bias: bool = False, b_codes: int = 0,
                        b_ilv: int = 0) -> bool:
    """Whether a kernel exists for the call: an exact-shape row, or the shape-generic row of K's K-loop class
    (M, N multiples of 256, K a multiple of 512 and at least the row's kmin)."""
    if (a_fmt, b_fmt, b_codes, b_ilv, int(bias), M, N, K, 0, 0) in _rows():
        return True
    if M % 256 or N % 256 or K % 512:
        return False
    g = _generic()
    for kcls in ((K // 128) % 12, 12):  # 12: a row serving every K-loop class
        kmin = g.get((a_fmt, b_fmt, b_codes, b_ilv, int(bias), kcls))
        if kmin is not None and K >= kmin:
            return True
    return False


def tilescale_softmax_d_supported(M: int, N: int, K: int, B: int, S: int, b_codes: int = 0, b_ilv: int = 0) -> bool:
    """Whether an A4W4 kernel emitting softmax_d [B, N/128, S] exists for (M, N, K) (exact rows only)."""
    return (4, 4, b_codes, b_ilv, 0, M, N, K, B, S) in _rows()


def a4w4_b_ilv(M: int, N: int, K: int) -> int:
    """Role B's scale interleave of the A4W4 tilescale kernel for (M, N, K), or ``ts_b_ilv(K)`` if none exists."""
    for r in _rows():
        if r[:2] == (4, 4) and r[5:8] == (M, N, K) and r[8] == 0:
            return r[3]
    from .tilescale import ts_b_ilv

    return ts_b_ilv(K)


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
    b_c1: Tensor | None = None,
    epi_o: Tensor | None = None,
    epi_delta: Tensor | None = None,
    epi_s0: int = 0,
) -> Tensor:
    """``out[M, N] = A @ B^T (+ bias)``; raises if the manifest has no kernel for the call. ``b_c1``: an FP6 B's
    C1 plane in its own buffer (``B`` then holds the C0 plane only). ``epi_o`` / ``epi_delta`` / ``epi_s0``: the
    softmax_d epilogue (module docstring)."""
    if out.ndim != 2:
        raise ValueError(f"gemm_mx_tilescale expects a 2D output, got {out.ndim}D")
    _gemm_tilescale_asm(A, B, A_scale, B_scale, out, K, bias, a_fmt, b_fmt, b_codes, b_ilv, b_c1, epi_o, epi_delta,
                        epi_s0)
    return out


def gemm_a6w4_tilescale(A, B, A_scale, B_scale, out, K, bias=None, b_codes=FP4_CODES["k128"]):
    return gemm_mx_tilescale(A, B, A_scale, B_scale, out, 6, 4, K, bias, b_codes, 0)


def gemm_a6w6_tilescale(A, B, A_scale, B_scale, out, K, bias=None, b_c1=None):
    return gemm_mx_tilescale(A, B, A_scale, B_scale, out, 6, 6, K, bias, 0, 0, b_c1)


def gemm_a4w4_tilescale(A, B, A_scale, B_scale, out, K, b_ilv=0, b_codes=FP4_CODES["row"], epi_o=None,
                        epi_delta=None, epi_s0=0):
    return gemm_mx_tilescale(A, B, A_scale, B_scale, out, 4, 4, K, None, b_codes, b_ilv, None, epi_o, epi_delta,
                             epi_s0)
