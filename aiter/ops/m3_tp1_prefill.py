# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3.1 TP1 decoder-layer kernels for gfx950, byte for byte the aiter ops
they replace (op_tests/test_m3_tp1_norm.py).

``norm_mx``: aiter's Gemma add+RMSNorm (the fused_qk_rmsnorm_group_quant kernel
for 4096 < n <= 8192) with MXFP8 (MX8; ``shuffle_scale_w4`` scales with
``s8_w4``) and MXFP4 (MX4) epilogues. ADD3 makes the input ``bf16(x * 2 +
s_add)``, RED4 first sums ``x`` [m, 4, n] as moe_reduction does, HOUT writes
that input and stops. aiter's build flushes fp32 denormals, so bf16-subnormal
route or ``s_add`` values count as zero.
``mx8_quant``: MXFP8 of a [rows, K] bf16 tensor, ``code = ceil_pow2(max(amax,
1e-10) / 448)``, ``q = clamp(x * 2^(127 - code), +-448)``; a NaN gives -448.
w4 scales leave the pad rows unwritten; the GEMM drops those rows' outputs.
Launches run on the current stream. The entry points raise off gfx950;
``is_supported()`` tells callers in advance.
"""

from torch import Tensor

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx_runtime

__all__ = [
    "NORM_FOLD_H",
    "NORM_FOLD_MX8",
    "NORM_INPUT_MX8",
    "NORM_POST_BF16_MX8",
    "NORM_POST_BF16_MX8_MX4",
    "NORM_RED4_FOLD_H",
    "NORM_RED4_FOLD_MX8",
    "mx8_quant",
    "norm_mx",
]

_ADD3, _OUT_BF16, _MX8, _MX4, _RED4, _HOUT = 1, 2, 4, 8, 16, 32
NORM_INPUT_MX8 = _MX8  # add + norm -> MXFP8
NORM_FOLD_MX8 = _ADD3 | _MX8  # bf16(x * 2 + s_add) + add + norm -> MXFP8
NORM_RED4_FOLD_MX8 = _RED4 | _ADD3 | _MX8  # top-4 sum, then as NORM_FOLD_MX8
NORM_POST_BF16_MX8 = _OUT_BF16 | _MX8  # add + norm -> bf16 and MXFP8
NORM_POST_BF16_MX8_MX4 = _OUT_BF16 | _MX8 | _MX4  # ... and MXFP4
NORM_FOLD_H = _ADD3 | _HOUT  # bf16(x * 2 + s_add) -> out_bf16
NORM_RED4_FOLD_H = _RED4 | _ADD3 | _HOUT  # top-4 sum, then as NORM_FOLD_H


def is_supported() -> bool:
    """True when this device runs these kernels (False on a host without a GPU)."""
    try:
        return get_gfx_runtime() == "gfx950"
    except RuntimeError:
        return False


def _require_gfx950(what: str) -> None:
    if not is_supported():
        raise RuntimeError(f"{what} requires gfx950, got {get_gfx_runtime()}")


@compile_ops("module_m3_tp1_norm", fc_name="m3_tp1_norm_mx", develop=True)
def _m3_tp1_norm_mx(
    x: Tensor,
    flags: int,
    s_add: Tensor | None,
    res: Tensor | None,
    w: Tensor | None,
    res_out: Tensor | None,
    out_bf16: Tensor | None,
    q8: Tensor | None,
    s8: Tensor | None,
    q4: Tensor | None,
    s4: Tensor | None,
    eps: float,
    m: int,
    n: int,
    s8_w4: bool,
) -> None: ...


@compile_ops("module_m3_tp1_mx8_quant", fc_name="m3_tp1_mx8_quant", develop=True)
def _m3_tp1_mx8_quant(x: Tensor, q: Tensor, s: Tensor, w4: bool) -> None: ...


def norm_mx(
    flags: int,
    x: Tensor,  # [m, n] bf16; [m, 4, n] under RED4
    s_add: Tensor | None,  # [m, n] bf16, ADD3
    res: Tensor | None,  # [m, n] bf16 residual (not HOUT)
    w: Tensor | None,  # [n] bf16 Gemma weight (not HOUT)
    res_out: Tensor | None,  # [m, n] bf16 = input + res (not HOUT)
    out_bf16: Tensor | None,  # [m, n] bf16: the norm output, or the HOUT input
    q8: Tensor | None,  # [m, n] e4m3, MX8
    s8: Tensor | None,  # e8m0 bytes, [>= m, n / 32] or w4 over 256-row padding
    q4: Tensor | None,  # [m, n / 2] fp4x2, MX4
    s4: Tensor | None,  # [m, n / 32] e8m0, MX4
    eps: float,
    m: int,
    n: int,
    s8_w4: bool,
) -> None:
    """One launch of the fused norm; ``flags`` is one of the NORM_* sets. Every
    tensor is contiguous; outputs the flags do not write may be None."""
    _require_gfx950("norm_mx")
    tensors = (s_add, res, w, res_out, out_bf16, q8, s8, q4, s4)
    _m3_tp1_norm_mx(x, int(flags), *tensors, float(eps), int(m), int(n), bool(s8_w4))


def mx8_quant(x: Tensor, q: Tensor, s: Tensor, w4: bool) -> None:
    """MXFP8 of a contiguous ``[rows, K]`` bf16 ``x`` into ``q`` (e4m3) and ``s``
    (e8m0 bytes, row-major ``[rows, K / 32]``, or ``shuffle_scale_w4`` over rows
    padded to 256 when ``w4``)."""
    _require_gfx950("mx8_quant")
    _m3_tp1_mx8_quant(x, q, s, bool(w4))
