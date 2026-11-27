# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Runtime wrapper for the gfx950 4-wave MXFP6 (A6W6) GEMM.

The operands are the ASM A6W6 kernels' packed blobs from
``aiter.ops.gemm_op_a6w6.quant_mxfp6_gemm``; the output is the padded
``[ceil(M, 256), ceil(N, 256)]`` BF16 tensor that ``gemm_a6w6`` allocates.
"""

from __future__ import annotations

import functools
import re

import torch
from torch import Tensor

from aiter.jit.utils.chip_info import get_cdna4_num_xcds, get_gfx_runtime

BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 128
MIN_K = BLOCK_K
PADK = 2
TILE_BYTES = BLOCK_M * BLOCK_K * 6 // 8
SCALE_TILE_BYTES = BLOCK_M * BLOCK_K // 32
DEFAULT_XCD_SWIZZLE = 4

_compile = None
_run_compiled = None
_fx = None


def _lazy_import() -> None:
    global _compile, _run_compiled, _fx
    if _compile is not None:
        return
    import flydsl.expr as fx_mod

    from .kernels.gemm_a6w6_4wave import compile_a6w6_gemm_4w
    from .kernels.tensor_shim import _run_compiled as run_compiled

    _compile = compile_a6w6_gemm_4w
    _run_compiled = run_compiled
    _fx = fx_mod


KERNEL_PREFIX = "flydsl_a6w6_4w_"
_KERNEL_NAME_RE = re.compile(
    rf"^{KERNEL_PREFIX}{BLOCK_M}x{BLOCK_N}x{BLOCK_K}_m8(?P<mblk_n>[24])b(?P<bar>\d+)"
    r"_1x(?P<xcd_swizzle>\d+)$"
)


def make_kernel_name(
    *,
    xcd_swizzle: int = DEFAULT_XCD_SWIZZLE,
    mblk_n: int = 2,
    bar: int | None = 8,
) -> str:
    """The tuned-table name of one variant. A name leaves K open, as tuned rows key
    on it and the launch passes it, so the untuned default name is a constant
    under torch.compile with dynamic shapes."""
    from .kernels.gemm_a6w6_4wave import Schedule

    tag = Schedule(int(mblk_n), bar).tag
    return f"{KERNEL_PREFIX}{BLOCK_M}x{BLOCK_N}x{BLOCK_K}_{tag}_1x{int(xcd_swizzle)}"


def parse_kernel_name(name: str) -> dict[str, int] | None:
    """The variant a canonical kernel name selects, or ``None``."""
    match = _KERNEL_NAME_RE.fullmatch(name)
    if match is None:
        return None
    return {key: int(value) for key, value in match.groupdict().items()}


def compile_a6w6_gemm(
    *,
    K: int,
    xcd_swizzle: int = DEFAULT_XCD_SWIZZLE,
    mblk_n: int = 2,
    bar: int | None = 8,
):
    """Compile and memoize the current device's launcher for the padded contraction
    dimension ``K``."""
    return _compile_a6w6_gemm(
        torch.cuda.current_device(), int(K), int(xcd_swizzle), int(mblk_n), bar
    )


@functools.lru_cache(maxsize=256)
def _compile_a6w6_gemm(device, K, xcd_swizzle, mblk_n, bar):
    # device only keys the cache: a launcher runs on the device that first ran it.
    _lazy_import()
    if K < MIN_K or K % BLOCK_K:
        raise ValueError(
            f"[FlyDSL A6W6 4wave] padded K must be a multiple of {BLOCK_K} of at "
            f"least {MIN_K}, got {K}"
        )
    num_xcds = get_cdna4_num_xcds() if xcd_swizzle > 0 else 8
    return _compile(
        K=int(K),
        xcd_swizzle=int(xcd_swizzle),
        num_xcds=int(num_xcds),
        mblk_n=int(mblk_n),
        bar=bar,
    )


def run_gemm_a6w6_4wave(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    out: Tensor,
    K: int,
    xcd_swizzle: int = DEFAULT_XCD_SWIZZLE,
    mblk_n: int = 2,
    bar: int | None = 8,
) -> Tensor:
    """``out = A * B^T`` for packed operands; ``K`` is the padded K."""
    if get_gfx_runtime() != "gfx950":
        raise NotImplementedError(
            "[FlyDSL A6W6 4wave] kernel is supported only on gfx950"
        )
    if (
        not out.is_cuda
        or out.dtype != torch.bfloat16
        or out.ndim != 2
        or not out.is_contiguous()
    ):
        raise ValueError(
            "[FlyDSL A6W6 4wave] out must be a contiguous 2-D BF16 GPU tensor"
        )
    device = out.get_device()
    pad_m, pad_n = out.shape
    if pad_m % BLOCK_M or pad_n % BLOCK_N:
        raise ValueError(
            f"[FlyDSL A6W6 4wave] out must be padded to {BLOCK_M}x{BLOCK_N}, got "
            f"{tuple(out.shape)}"
        )
    steps = K // BLOCK_K + PADK
    for name, tensor, rows, tile in (
        ("A", A, pad_m, TILE_BYTES),
        ("B", B, pad_n, TILE_BYTES),
        ("A_scale", A_scale, pad_m, SCALE_TILE_BYTES),
        ("B_scale", B_scale, pad_n, SCALE_TILE_BYTES),
    ):
        expected = rows // BLOCK_M * steps * tile
        if tensor.dtype != torch.uint8 or tensor.numel() != expected:
            raise ValueError(
                f"[FlyDSL A6W6 4wave] {name} must be {expected} packed uint8 bytes, got "
                f"{tensor.numel()} {tensor.dtype}"
            )
        if tensor.get_device() != device or not tensor.is_contiguous():
            raise ValueError(
                f"[FlyDSL A6W6 4wave] {name} must be a contiguous tensor on {out.device}"
            )
    # Byte offsets are signed 32-bit: a buffer may hold up to 2**31 bytes.
    if max(A.numel(), B.numel(), out.numel() * 2) > 2**31:
        raise ValueError("[FlyDSL A6W6 4wave] operands exceed 32-bit byte offsets")
    if device != torch.cuda.current_device():
        with torch.cuda.device(device):
            return run_gemm_a6w6_4wave(
                A, B, A_scale, B_scale, out, K, xcd_swizzle, mblk_n, bar
            )
    executable = _compile_a6w6_gemm(device, int(K), int(xcd_swizzle), int(mblk_n), bar)
    _run_compiled(
        executable,
        A.view(-1),
        B.view(-1),
        out.view(-1),
        A_scale.view(-1),
        B_scale.view(-1),
        pad_m,
        pad_n,
        _fx.Stream(torch.cuda.current_stream(device)),
    )
    return out


def run_named_gemm_a6w6_4wave(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    out: Tensor,
    K: int,
    kernel_name: str,
) -> Tensor:
    """Run the variant ``kernel_name`` names; ``K`` is the padded K."""
    config = parse_kernel_name(kernel_name)
    if config is None:
        raise ValueError(
            f"[FlyDSL A6W6 4wave] unrecognized kernelName: {kernel_name!r}"
        )
    return run_gemm_a6w6_4wave(A, B, A_scale, B_scale, out, K, **config)


__all__ = [
    "KERNEL_PREFIX",
    "compile_a6w6_gemm",
    "make_kernel_name",
    "parse_kernel_name",
    "run_gemm_a6w6_4wave",
    "run_named_gemm_a6w6_4wave",
]
