# SPDX-License-Identifier: MIT

"""Packed group-64 signed INT4 GEMV for gfx1201.

The packed weight layout is ``[N / 32, K / 64, 1088]`` bytes per tile.  Each
tile starts with 32 little-endian FP16 row scales followed by 32 pairs of
signed INT4 values for each of the 32 rows.
"""

from __future__ import annotations

from functools import cache

import torch
from torch import Tensor

from ..jit.core import compile_ops, is_experimental_enabled

__all__ = ["q4_group64_gemv"]

_MODULE = "module_q4_group64_gemv"
_MAPPING_IDS = {
    "auto": 0,
    "old": 1,
    "split2": 2,
    "split4": 3,
    "split8": 4,
    "small8x8": 5,
    "small8x16": 6,
    "small8x32": 7,
    "small16x16": 8,
    "small16x32": 9,
    "small32x32": 10,
}

# Dispatch choices measured with rotating inputs on gfx1201.  Shape is the
# complete key; unlisted shapes deliberately retain the conservative kernel.
_AUTO_MAPPING = {
    (512, 3584): "small32x32",
    (1024, 3072): "small32x32",
    (1024, 4096): "small32x32",
    (3072, 3072): "split8",
    (3072, 8192): "split8",
    (3584, 3584): "split8",
    (3584, 18944): "split8",
    (4096, 4096): "split8",
    (4096, 12288): "split8",
    (4096, 14336): "split8",
    (8192, 3072): "split4",
    (12288, 4096): "split8",
    (14336, 4096): "split8",
    (18944, 3584): "split8",
}


@compile_ops(_MODULE, fc_name="q4_group64_gemv_out", develop=True)
def _q4_group64_gemv_out(
    x: Tensor,
    packed_weight: Tensor,
    out: Tensor,
    mapping: int,
) -> None: ...


@cache
def _gfx_arch_for_index(index: int) -> str:
    arch = getattr(torch.cuda.get_device_properties(index), "gcnArchName", "")
    return arch.lower().split(":", maxsplit=1)[0]


def _gfx_arch(device: torch.device) -> str:
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    return _gfx_arch_for_index(index)


def _require_experimental_enabled() -> None:
    if not is_experimental_enabled():
        raise RuntimeError(
            "q4_group64_gemv is experimental; set "
            "AITER_ENABLE_EXPERIMENTAL=1 before calling it"
        )


def _validate_inputs(x: Tensor, packed_weight: Tensor, out: Tensor | None) -> int:
    if not x.is_cuda or not packed_weight.is_cuda:
        raise ValueError("x and packed_weight must be CUDA/HIP tensors")
    if x.device != packed_weight.device:
        raise ValueError("x and packed_weight must be on the same device")
    arch = _gfx_arch(x.device)
    if arch != "gfx1201":
        raise RuntimeError(f"q4_group64_gemv requires gfx1201, got {arch!r}")

    if x.dtype != torch.float32 or x.ndim != 1 or not x.is_contiguous():
        raise ValueError("x must be a contiguous FP32 vector [K]")
    k = x.numel()
    if k <= 0 or k % 64:
        raise ValueError(f"K must be positive and divisible by 64, got {k}")
    if packed_weight.dtype != torch.uint8:
        raise ValueError("packed_weight must have dtype uint8")
    if packed_weight.ndim != 3 or not packed_weight.is_contiguous():
        raise ValueError("packed_weight must be contiguous [N/32,K/64,1088]")
    if packed_weight.shape[0] <= 0 or packed_weight.shape[2] != 1088:
        raise ValueError("packed_weight must have shape [N/32,K/64,1088]")
    if packed_weight.shape[1] * 64 != k:
        raise ValueError(
            "packed_weight K dimension does not match x: "
            f"{packed_weight.shape[1] * 64} != {k}"
        )
    if packed_weight.data_ptr() % 2:
        raise ValueError("packed_weight must be 2-byte aligned for FP16 scales")

    n = packed_weight.shape[0] * 32
    if out is not None:
        if out.device != x.device:
            raise ValueError("out must be on the same device as x")
        if out.dtype != torch.float32 or out.shape != (n,) or not out.is_contiguous():
            raise ValueError(f"out must be a contiguous FP32 vector [{n}]")
    return n


def _selected_mapping(n: int, k: int) -> str:
    """Return the private RX 9070 XT candidate; unseen shapes use ``old``."""

    return _AUTO_MAPPING.get((n, k), "old")


def _q4_group64_gemv(
    x: Tensor,
    packed_weight: Tensor,
    *,
    mapping: str = "auto",
    out: Tensor | None = None,
) -> Tensor:
    """Private correctness/ablation entry point with an explicit mapping knob."""

    _require_experimental_enabled()
    n = _validate_inputs(x, packed_weight, out)
    try:
        mapping_id = _MAPPING_IDS[mapping]
    except KeyError as exc:
        choices = ", ".join(_MAPPING_IDS)
        raise ValueError(
            f"unknown mapping {mapping!r}; expected one of {choices}"
        ) from exc
    if mapping == "split2" and n % 128:
        raise ValueError(f"split2 requires N divisible by 128, got {n}")
    if mapping == "split4" and n % 64:
        raise ValueError(f"split4 requires N divisible by 64, got {n}")
    if out is None:
        out = torch.empty(n, dtype=torch.float32, device=x.device)
    with torch.cuda.device(x.device):
        _q4_group64_gemv_out(x, packed_weight, out, mapping_id)
    return out


def q4_group64_gemv(x: Tensor, packed_weight: Tensor) -> Tensor:
    """Multiply a packed group-64 signed INT4 matrix by an FP32 vector.

    Args:
        x: Contiguous FP32 vector with shape ``[K]``; ``K`` is divisible by 64.
        packed_weight: Contiguous uint8 tensor with shape
            ``[N / 32, K / 64, 1088]``.  The first 64 bytes of each tile are
            32 FP16 row scales and the remaining 1024 bytes contain signed
            low/high INT4 pairs in row-interleaved order.

    Returns:
        A contiguous FP32 vector with shape ``[N]``.
    """

    return _q4_group64_gemv(x, packed_weight)
