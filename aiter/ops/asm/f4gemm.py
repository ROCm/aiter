# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# gfx1250 F4GEMM (a4w4) -- ASM, kernarg preload mode. benchmark-only, private.
# MXFP4/NVFP4 x FP4 GEMM. See csrc/py_itfs_cu/asm_f4gemm.cu.

import torch
from torch import Tensor

from aiter import logger
from aiter.jit.core import compile_ops
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.utility import dtypes


@compile_ops(
    "module_f4gemm_asm",
    fc_name="mxfp4_gemm_asm",
    ffi_type="ctypes",
)
def _mxfp4_gemm_asm(
    A: Tensor,  # A:[M, K/2] fp4x2 (preshuffled if a_preshuffle=1)
    B: Tensor,  # B:[N, K/2] fp4x2 (preshuffled)
    ScaleA: Tensor,  # ScaleA:[M, K/32] e8m0 (shuffled)
    ScaleB: Tensor,  # ScaleB:[N, K/32] e8m0 (shuffled)
    out: Tensor,  # Out: bf16 [M,N] / fp4x2 [M,N/2] / fp8 [M,N]
    out_scale: Tensor | None = None,  # mxfp8 only: E8M0 [M, N/128] (None otherwise)
    kernelName: str | None = None,
    a_preshuffle: int = 1,
) -> None: ...


@compile_ops(
    "module_f4gemm_asm",
    fc_name="nvfp4_gemm_asm",
    ffi_type="ctypes",
)
def _nvfp4_gemm_asm(
    A: Tensor,
    B: Tensor,
    ScaleA: Tensor,  # e4m3 (shuffled)
    ScaleB: Tensor,  # e4m3 (shuffled)
    GlobalScaleA: float,
    GlobalScaleB: float,
    out: Tensor,  # Out: bf16 [M,N] / fp4x2 [M,N/2] / fp8 [M,N]
    out_scale: Tensor | None = None,  # mxfp8 only: E8M0 [M, N/128] (None otherwise)
    kernelName: str | None = None,
    a_preshuffle: int = 1,
) -> None: ...


# gfx1250 f4gemm mxfp8-output block size along N: the kernel dynamically
# quantizes each 128-wide output block to fp8 e4m3 and emits one E8M0 scale.
MXFP8_OUT_SCALE_BLOCK = 128


def _is_mxfp8_out(dtype: torch.dtype) -> bool:
    """mxfp8 output == fp8 e4m3 data + a per-block E8M0 scale (dtypes.fp8)."""
    return dtype == dtypes.fp8


def _alloc_f4gemm_out(M: int, N: int, dtype: torch.dtype, device) -> Tensor:
    """Allocate the F4GEMM output. Packed FP4 (e2m1) output is 2 values/byte, so
    it is a ``[M, N//2]`` ``fp4x2`` tensor; bf16 output is a plain ``[M, N]``;
    mxfp8 output is a plain ``[M, N]`` fp8 (its E8M0 scale is a separate tensor,
    see :func:`_alloc_f4gemm_out_scale`)."""
    if dtype == dtypes.fp4x2:
        assert N % 2 == 0, "packed fp4 output requires even N"
        return torch.empty((M, N // 2), dtype=dtypes.fp4x2, device=device)
    return torch.empty((M, N), dtype=dtype, device=device)


def _alloc_f4gemm_out_scale(
    M: int, N: int, dtype: torch.dtype, device
) -> Tensor | None:
    """Allocate the mxfp8 output E8M0 block-scale buffer (``None`` for bf16/fp4).

    The kernel fills it in the packed ``(Mpad/64, scaleN, 16, 4)`` layout with
    ``Mpad = ceil(M/64)*64`` (POC host Mpad64) and ``scaleN = ceil(N/128)``, so
    the buffer is ``[Mpad, scaleN]``; unpack via :func:`unpack_mxfp8_out_scale`."""
    if not _is_mxfp8_out(dtype):
        return None
    Mpad = (M + 63) // 64 * 64
    scaleN = (N + MXFP8_OUT_SCALE_BLOCK - 1) // MXFP8_OUT_SCALE_BLOCK
    return torch.empty((Mpad, scaleN), dtype=dtypes.fp8_e8m0, device=device)


def unpack_mxfp8_out_scale(packed: Tensor, M: int, N: int) -> Tensor:
    """Unpack the mxfp8 output E8M0 scale from the kernel's packed
    ``(Mpad/64, scaleN, 16, 4)`` layout to row-major ``[M, ceil(N/128)]``
    (``Mpad = ceil(M/64)*64``; padding rows are dropped)."""
    scaleN = (N + MXFP8_OUT_SCALE_BLOCK - 1) // MXFP8_OUT_SCALE_BLOCK
    Mpad = (M + 63) // 64 * 64
    u8 = packed.reshape(-1).view(torch.uint8)[: (Mpad // 64) * scaleN * 16 * 4]
    rm = u8.reshape(Mpad // 64, scaleN, 16, 4).permute(0, 3, 2, 1).reshape(Mpad, scaleN)
    return rm[:M].contiguous().view(dtypes.fp8_e8m0)


def _gemm_mxfp4_asm(
    A: Tensor,  # A:[M, K/2] fp4x2
    B: Tensor,  # B:[N, K/2] fp4x2
    ScaleA: Tensor,  # ScaleA:[M, K/32] e8m0
    ScaleB: Tensor,  # ScaleB:[N, K/32] e8m0
    dtype: torch.dtype = dtypes.bf16,  # output dtype: bf16, fp4x2 (packed e2m1), or fp8 (mxfp8)
    a_preshuffle: bool = True,
    kernelName: str = "",
) -> Tensor | tuple[Tensor, Tensor]:
    """MXFP4 GEMM (preload SGPR mode). D = A * B with e8m0 scales. ``dtype``:
    bf16 ``[M,N]``, packed FP4 ``[M,N//2]`` fp4x2 (cvt_scale=1), or mxfp8
    (``dtypes.fp8``) returning ``(out_fp8 [M,N], scale_e8m0)``. ``scale_e8m0`` is
    in the PACKED ``(M/64,N//128,16,4)`` layout -- unpack via
    :func:`unpack_mxfp8_out_scale`."""
    M = A.shape[0]
    N = B.shape[0]
    out = _alloc_f4gemm_out(M, N, dtype, A.device)
    out_scale = _alloc_f4gemm_out_scale(M, N, dtype, A.device)
    _mxfp4_gemm_asm(
        A,
        B,
        ScaleA,
        ScaleB,
        out,
        out_scale,
        kernelName if kernelName else None,
        int(bool(a_preshuffle)),
    )
    return (out, out_scale) if out_scale is not None else out


def _gemm_nvfp4_asm(
    A: Tensor,
    B: Tensor,
    ScaleA: Tensor,  # e4m3
    ScaleB: Tensor,  # e4m3
    GlobalScaleA: float,
    GlobalScaleB: float,
    dtype: torch.dtype = dtypes.bf16,  # output dtype: bf16, fp4x2 (packed e2m1), or fp8 (mxfp8)
    a_preshuffle: bool = True,
    kernelName: str = "",
) -> Tensor | tuple[Tensor, Tensor]:
    """NVFP4 GEMM (preload SGPR mode). D = A * B with e4m3 scales + global alphas.
    ``dtype``: bf16 ``[M,N]``, packed FP4 ``[M,N//2]`` fp4x2 (cvt_scale=1), or
    mxfp8 (``dtypes.fp8``) returning ``(out_fp8 [M,N], scale_e8m0)``.
    ``scale_e8m0`` is in the PACKED ``(M/64,N//128,16,4)`` layout -- unpack via
    :func:`unpack_mxfp8_out_scale`."""
    M = A.shape[0]
    N = B.shape[0]
    out = _alloc_f4gemm_out(M, N, dtype, A.device)
    out_scale = _alloc_f4gemm_out_scale(M, N, dtype, A.device)
    _nvfp4_gemm_asm(
        A,
        B,
        ScaleA,
        ScaleB,
        float(GlobalScaleA),
        float(GlobalScaleB),
        out,
        out_scale,
        kernelName if kernelName else None,
        int(bool(a_preshuffle)),
    )
    return (out, out_scale) if out_scale is not None else out


def _as_global_scale(scale) -> float:
    """Normalize an NVFP4 per-tensor global scale (float or 0-d/1-elem Tensor) to a float."""
    if scale is None:
        return 1.0
    if torch.is_tensor(scale):
        return float(scale.detach().reshape(-1)[0].item())
    return float(scale)


_GFX1250 = ["gfx1250"]


def _f4gemm_asm_dispatch(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    *,
    dtype: torch.dtype,
    apreshuffle: bool,
    global_A_scale: Tensor | None,
    global_B_scale: Tensor | None,
    bias: Tensor | None,
    alpha: float | None,
    beta: float | None,
):
    """Shared gfx1250 F4GEMM dispatch: MXFP4 vs NVFP4 by global-scale presence.
    Returns the raw asm result (single Tensor, or (data, scale) tuple for mxfp8).
    B is always preshuffled; bias/alpha/beta are not plumbed through these kernels."""
    if (
        bias is not None
        or (alpha is not None and alpha != 1.0)
        or (beta is not None and beta != 0.0)
    ):
        logger.warning(
            "gemm_a4w4* on gfx1250 ignores bias/alpha/beta: not supported by the "
            "F4GEMM kernels."
        )
    m = A.numel() // A.shape[-1]
    A2 = A.view(m, A.shape[-1])
    if global_A_scale is not None or global_B_scale is not None:
        return _gemm_nvfp4_asm(
            A2,
            B,
            A_scale,
            B_scale,
            _as_global_scale(global_A_scale),
            _as_global_scale(global_B_scale),
            dtype=dtype,
            a_preshuffle=bool(apreshuffle),
        )
    return _gemm_mxfp4_asm(
        A2, B, A_scale, B_scale, dtype=dtype, a_preshuffle=bool(apreshuffle)
    )


def _gemm_a4w4o4_asm_fake(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    bias: Tensor | None = None,
    alpha: float | None = 1.0,
    beta: float | None = 0.0,
    bpreshuffle: bool | None = True,
    apreshuffle: bool | None = False,
    global_A_scale: Tensor | None = None,
    global_B_scale: Tensor | None = None,
) -> torch.Tensor:
    m = A.numel() // A.shape[-1]
    n = B.shape[0]
    out = _alloc_f4gemm_out(m, n, dtypes.fp4x2, A.device)
    return out.view(*A.shape[:-1], out.shape[-1])


@torch_compile_guard(gen_fake=_gemm_a4w4o4_asm_fake)
def _gemm_a4w4o4_asm(
    A: Tensor,  # A:[M, K/2] f4x2
    B: Tensor,  # B:[N, K/2] f4x2
    A_scale: Tensor,  # A_scale:[M, K/block] e8m0 (MXFP4) / e4m3 (NVFP4)
    B_scale: Tensor,  # B_scale:[N, K/block]
    bias: Tensor | None = None,
    alpha: float | None = 1.0,
    beta: float | None = 0.0,
    bpreshuffle: bool | None = True,
    apreshuffle: bool | None = False,
    global_A_scale: Tensor | None = None,  # NVFP4 per-tensor
    global_B_scale: Tensor | None = None,  # NVFP4 per-tensor
) -> torch.Tensor:
    """A4W4 GEMM with packed FP4 (e2m1, cvt_scale=1) output ``[*lead, N//2]``.
    gfx1250 only. MXFP4 vs NVFP4 by global-scale presence."""
    assert (
        get_gfx() in _GFX1250
    ), f"_gemm_a4w4o4_asm (packed FP4 output) is only supported on gfx1250, got {get_gfx()}"
    out = _f4gemm_asm_dispatch(
        A,
        B,
        A_scale,
        B_scale,
        dtype=dtypes.fp4x2,
        apreshuffle=bool(apreshuffle),
        global_A_scale=global_A_scale,
        global_B_scale=global_B_scale,
        bias=bias,
        alpha=alpha,
        beta=beta,
    )
    return out.view(*A.shape[:-1], out.shape[-1])


def _gemm_a4w4o8_asm_fake(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    bias: Tensor | None = None,
    alpha: float | None = 1.0,
    beta: float | None = 0.0,
    bpreshuffle: bool | None = True,
    apreshuffle: bool | None = False,
    global_A_scale: Tensor | None = None,
    global_B_scale: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    m = A.numel() // A.shape[-1]
    n = B.shape[0]
    lead = A.shape[:-1]
    out = _alloc_f4gemm_out(m, n, dtypes.fp8, A.device)
    # scale is a packed [Mpad, scaleN] buffer (not row-aligned to M): returned
    # as-is, unpack via unpack_mxfp8_out_scale.
    scale = _alloc_f4gemm_out_scale(m, n, dtypes.fp8, A.device)
    return out.view(*lead, out.shape[-1]), scale


@torch_compile_guard(gen_fake=_gemm_a4w4o8_asm_fake)
def _gemm_a4w4o8_asm(
    A: Tensor,  # A:[M, K/2] f4x2
    B: Tensor,  # B:[N, K/2] f4x2
    A_scale: Tensor,  # A_scale:[M, K/block] e8m0 (MXFP4) / e4m3 (NVFP4)
    B_scale: Tensor,  # B_scale:[N, K/block]
    bias: Tensor | None = None,
    alpha: float | None = 1.0,
    beta: float | None = 0.0,
    bpreshuffle: bool | None = True,
    apreshuffle: bool | None = False,
    global_A_scale: Tensor | None = None,  # NVFP4 per-tensor
    global_B_scale: Tensor | None = None,  # NVFP4 per-tensor
) -> tuple[Tensor, Tensor]:
    """A4W4 GEMM with mxfp8 output: returns ``(fp8 e4m3 data [*lead, N], E8M0
    scale)``. The scale is in the PACKED ``(M/64, N//128, 16, 4)`` layout --
    unpack via :func:`unpack_mxfp8_out_scale`. gfx1250 only. MXFP4 vs NVFP4 by
    global-scale presence."""
    assert (
        get_gfx() in _GFX1250
    ), f"_gemm_a4w4o8_asm (mxfp8 output) is only supported on gfx1250, got {get_gfx()}"
    o, s = _f4gemm_asm_dispatch(
        A,
        B,
        A_scale,
        B_scale,
        dtype=dtypes.fp8,
        apreshuffle=bool(apreshuffle),
        global_A_scale=global_A_scale,
        global_B_scale=global_B_scale,
        bias=bias,
        alpha=alpha,
        beta=beta,
    )
    lead = A.shape[:-1]
    # s is the packed [Mpad, scaleN] scale buffer; return as-is (see o8_fake).
    return o.view(*lead, o.shape[-1]), s
