# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
from typing import Optional, Tuple, Union

import pandas as pd
import torch
from torch import Tensor

from aiter import logger
from aiter.jit.utils.torch_guard import torch_compile_guard

from ..jit.core import AITER_CONFIGS, AITER_LOG_TUNED_CONFIG, compile_ops
from ..jit.utils.chip_info import get_cu_num
from ..jit.utils.chip_info import get_gfx_runtime as get_gfx
from ..ops.gemm_op_common import get_padded_m
from ..utility import dtypes


@functools.lru_cache(maxsize=1024)
def compute_gemm_SplitK(M: int, N: int, K: int, tile_m: int, tile_n: int, tile_k: int):
    cu_num = get_cu_num()
    tile_num = ((M + tile_m - 1) // tile_m) * ((N + tile_n - 1) // tile_n)
    cusPerTile = cu_num / tile_num
    splitK = 0
    while cusPerTile >= pow(2, splitK + 1) and (pow(2, splitK + 1) * tile_k) < 2 * K:
        splitK += 1
    ## to make sure the precision is not lost, max is 4
    # return min(splitK, 4)
    return 3


@functools.lru_cache(maxsize=1024)
def get_GEMM_config(M: int, N: int, K: int):
    tuned_file = AITER_CONFIGS.AITER_CONFIG_GEMM_A4W4_FILE
    if not hasattr(get_GEMM_config, "gemm_dict"):
        gemm_dict = pd.read_csv(
            AITER_CONFIGS.AITER_CONFIG_GEMM_A4W4_FILE
        ).drop_duplicates()
        # Use (gfx, cu_num, M, N, K) key when the CSV has a gfx column (new schema).
        # Fall back to (cu_num, M, N, K) for old CSVs that pre-date the gfx column.
        if "gfx" in gemm_dict.columns:
            get_GEMM_config.gemm_dict = gemm_dict.set_index(
                ["gfx", "cu_num", "M", "N", "K"]
            ).to_dict("index")
            get_GEMM_config.has_gfx = True
        else:
            logger.warning(
                f"{AITER_CONFIGS.AITER_CONFIG_GEMM_A4W4_FILE} has no 'gfx' column -- "
                "falling back to cu_num-only key. Re-run the tuner or migrate the CSV."
            )
            get_GEMM_config.gemm_dict = gemm_dict.set_index(
                ["cu_num", "M", "N", "K"]
            ).to_dict("index")
            get_GEMM_config.has_gfx = False
    gfx = get_gfx()
    cu_num = get_cu_num()
    padded_M = M
    config = None
    for gl in [None, 0, 1]:
        padded_M = M if gl is None else get_padded_m(M, N, K, gl)
        key = (
            (gfx, cu_num, padded_M, N, K)
            if get_GEMM_config.has_gfx
            else (cu_num, padded_M, N, K)
        )
        config = get_GEMM_config.gemm_dict.get(key, None)
        if config is not None:
            if AITER_LOG_TUNED_CONFIG:
                logger.info(
                    f"shape is M:{M}, N:{N}, K:{K}, found padded_M: {padded_M}, N:{N}, K:{K} is tuned on cu_num = {cu_num} in {AITER_CONFIGS.AITER_CONFIG_GEMM_A4W4_FILE}, kernel name is {config['kernelName']}, splitK is {config['splitK']}!"
                )
            break
    else:
        logger.info(
            f"shape is M:{M}, N:{N}, K:{K}, not found tuned config in {tuned_file}, will use default config!"
        )
    return config


def _f4gemm_asm_dispatch(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    *,
    dtype: torch.dtype,
    apreshuffle: bool,
    global_A_scale: Optional[Tensor],
    global_B_scale: Optional[Tensor],
    bias: Optional[Tensor],
    alpha: Optional[float],
    beta: Optional[float],
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
        return gemm_nvfp4_asm(
            A2,
            B,
            A_scale,
            B_scale,
            _as_global_scale(global_A_scale),
            _as_global_scale(global_B_scale),
            dtype=dtype,
            a_preshuffle=bool(apreshuffle),
        )
    return gemm_mxfp4_asm(
        A2, B, A_scale, B_scale, dtype=dtype, a_preshuffle=bool(apreshuffle)
    )


def gemm_a4w4_fake(
    A: Tensor,  # A:[M, K/2] f4x2
    B: Tensor,  # B:[N, K/2] f4x2
    A_scale: Tensor,  # A_scale:[M, K/block_size] MXFP4: block_size=32 e8m0 padded, NVFP4: block_size=16 e4m3 padded
    B_scale: Tensor,  # B_scale:[N, K/block_size] MXFP4: block_size=32 e8m0 padded, NVFP4: block_size=16 e4m3 padded
    bias: Tensor | None = None,  # bias:[1, N] f32
    dtype: torch.dtype = dtypes.bf16,
    alpha: float | None = 1.0,
    beta: float | None = 0.0,
    bpreshuffle: bool | None = True,
    apreshuffle: bool | None = False,
    global_A_scale: Tensor | None = None,  # NVFP4 per-tensor
    global_B_scale: Tensor | None = None,  # NVFP4 per-tensor
) -> torch.Tensor:
    if dtype in (dtypes.fp4x2, dtypes.fp8):
        raise NotImplementedError(
            "gemm_a4w4 returns one plain-dtype tensor; use gemm_a4w4o4 / gemm_a4w4o8"
        )
    n = B.shape[0]
    return torch.empty((*A.shape[:-1], n), dtype=dtype, device=A.device)


@torch_compile_guard(gen_fake=gemm_a4w4_fake)
def gemm_a4w4(
    A: Tensor,  # A:[M, K/2] f4x2
    B: Tensor,  # B:[N, K/2] f4x2
    A_scale: Tensor,  # A_scale:[M, K/block_size] MXFP4: block_size=32 e8m0 padded, NVFP4: block_size=16 e4m3 padded
    B_scale: Tensor,  # B_scale:[N, K/block_size] MXFP4: block_size=32 e8m0 padded, NVFP4: block_size=16 e4m3 padded
    bias: Tensor | None = None,  # bias:[1, N] f32
    dtype: torch.dtype = dtypes.bf16,
    alpha: float | None = 1.0,
    beta: float | None = 0.0,
    bpreshuffle: bool | None = True,
    apreshuffle: bool | None = False,
    global_A_scale: Tensor | None = None,  # NVFP4 per-tensor
    global_B_scale: Tensor | None = None,  # NVFP4 per-tensor
) -> torch.Tensor:
    """A4W4 GEMM (4-bit quantized matmul) returning one plain-dtype tensor.

    On gfx1250 the call is dispatched to the dedicated F4GEMM asm path.
    MXFP4 vs NVFP4 is selected by the presence of ``global_A_scale``/
    ``global_B_scale`` (NVFP4 per-tensor global scales).
    """
    # Low-precision output has a different return arity/shape (fp4 is [M,N//2],
    # mxfp8 is a (data, scale) tuple), so it lives on separate fixed-arity ops.
    if dtype in (dtypes.fp4x2, dtypes.fp8):
        raise NotImplementedError(
            "gemm_a4w4 returns one plain-dtype tensor; use gemm_a4w4o4 (packed "
            "fp4) or gemm_a4w4o8 (mxfp8) for low-precision output"
        )
    m = A.numel() // A.shape[-1]
    n = B.shape[0]
    k = A.shape[-1] * 2
    gfx_arch = get_gfx()
    if gfx_arch in ["gfx1250"]:
        # F4GEMM is kept on a separate dispatch (preload kargs layout).
        out = _f4gemm_asm_dispatch(
            A,
            B,
            A_scale,
            B_scale,
            dtype=dtype,
            apreshuffle=bool(apreshuffle),
            global_A_scale=global_A_scale,
            global_B_scale=global_B_scale,
            bias=bias,
            alpha=alpha,
            beta=beta,
        )
        return out.view(*A.shape[:-1], out.shape[-1])
    out = torch.empty(((m + 31) // 32 * 32, n), dtype=dtype, device=A.device)
    if gfx_arch in ["gfx942"]:
        raise RuntimeError(
            f"A4W4 GEMM kernel is not supported on gfx942, but got {gfx_arch}!"
        )
    ck_config = get_GEMM_config(m, n, k)
    # splitK = None
    splitK = 0
    kernelName = ""
    if ck_config is not None:
        splitK = ck_config.get("splitK", None)
        kernelName = ck_config["kernelName"]
    if (
        ck_config is not None
        and kernelName.find("_ZN") == -1
        # or bias is None
    ):
        splitK = 0 if splitK is None else splitK
        return gemm_a4w4_blockscale(
            A.view(m, k // 2),
            B,
            A_scale,
            B_scale,
            out,
            splitK=splitK,
            kernelName=kernelName,
        )[:m]
    assert (
        out.shape[0] % 32 == 0
    ), "Dim0 of gemm_a4w4_asm output needs to be padded to multiples of 32!"
    gemm_a4w4_asm(
        A.view(m, k // 2),
        B,
        A_scale,
        B_scale,
        out,
        kernelName,
        bias,
        alpha,
        beta,
        bpreshuffle,
        log2_k_split=splitK,
    )
    return out[:m].view(*A.shape[:-1], n)


@compile_ops(
    "module_gemm_a4w4_asm",
    fc_name="gemm_a4w4_asm",
    ffi_type="ctypes",
)
def _gemm_a4w4_asm(
    A: Tensor,  # A:[M, K/2] f4x2
    B: Tensor,  # B:[N, K/2] f4x2
    A_scale: Tensor,  # A_scale:[M, K/32] e8m0 paded
    B_scale: Tensor,  # B_scale:[N, K/32] e8m0 paded
    out: Tensor,  # Out:[M, N] bf16
    kernelName: str | None = None,
    bias: Tensor | None = None,  # bias:[1, N] f32
    alpha: float = 1.0,
    beta: float = 0.0,
    bpreshuffle: int = 1,
    log2_k_split: int = 0,
) -> None: ...


def gemm_a4w4_asm(
    A: Tensor,  # A:[M, K/2] f4x2
    B: Tensor,  # B:[N, K/2] f4x2
    A_scale: Tensor,  # A_scale:[M, K/32] e8m0 paded
    B_scale: Tensor,  # B_scale:[N, K/32] e8m0 paded
    out: Tensor,  # Out:[M, N] bf16
    kernelName: str = "",
    bias: Tensor | None = None,  # bias:[1, N] f32
    alpha: float | None = 1.0,
    beta: float | None = 0.0,
    bpreshuffle: bool | None = True,
    log2_k_split: int | None = None,
) -> Tensor:
    _gemm_a4w4_asm(
        A,
        B,
        A_scale,
        B_scale,
        out,
        kernelName if kernelName else None,
        bias,
        alpha if alpha is not None else 1.0,
        beta if beta is not None else 0.0,
        int(bpreshuffle) if bpreshuffle is not None else 1,
        log2_k_split if log2_k_split is not None else 0,
    )
    return out


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
    out_scale: Optional[Tensor] = None,  # mxfp8 only: E8M0 [M, N/128] (None otherwise)
    kernelName: Optional[str] = None,
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
    out_scale: Optional[Tensor] = None,  # mxfp8 only: E8M0 [M, N/128] (None otherwise)
    kernelName: Optional[str] = None,
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
) -> Optional[Tensor]:
    """Allocate the mxfp8 output E8M0 block-scale buffer (``None`` for bf16/fp4).

    Shaped ``[M, N//128]`` for the byte count; the kernel fills it in the packed
    ``(M/64, N//128, 16, 4)`` layout -- unpack via :func:`unpack_mxfp8_out_scale`."""
    if not _is_mxfp8_out(dtype):
        return None
    assert (
        N % MXFP8_OUT_SCALE_BLOCK == 0
    ), f"mxfp8 output requires N % {MXFP8_OUT_SCALE_BLOCK} == 0, got N={N}"
    assert M % 64 == 0, f"mxfp8 output-scale packing requires M % 64 == 0, got M={M}"
    return torch.empty(
        (M, N // MXFP8_OUT_SCALE_BLOCK), dtype=dtypes.fp8_e8m0, device=device
    )


def unpack_mxfp8_out_scale(packed: Tensor, M: int, N: int) -> Tensor:
    """Unpack the mxfp8 output E8M0 scale from the kernel's packed
    ``(M/64, N//128, 16, 4)`` layout to row-major ``[M, N//128]`` (requires
    ``M % 64 == 0``)."""
    assert M % 64 == 0, f"mxfp8 output-scale packing requires M % 64 == 0, got M={M}"
    assert (
        N % MXFP8_OUT_SCALE_BLOCK == 0
    ), f"mxfp8 output requires N % {MXFP8_OUT_SCALE_BLOCK} == 0, got N={N}"
    scaleN = N // MXFP8_OUT_SCALE_BLOCK
    u8 = packed.reshape(-1).view(torch.uint8)[: (M // 64) * scaleN * 16 * 4]
    rm = u8.reshape(M // 64, scaleN, 16, 4).permute(0, 3, 2, 1).reshape(M, scaleN)
    return rm.contiguous().view(dtypes.fp8_e8m0)


def gemm_mxfp4_asm(
    A: Tensor,  # A:[M, K/2] fp4x2
    B: Tensor,  # B:[N, K/2] fp4x2
    ScaleA: Tensor,  # ScaleA:[M, K/32] e8m0
    ScaleB: Tensor,  # ScaleB:[N, K/32] e8m0
    dtype: torch.dtype = dtypes.bf16,  # output dtype: bf16, fp4x2 (packed e2m1), or fp8 (mxfp8)
    a_preshuffle: bool = True,
    kernelName: str = "",
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
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


def gemm_nvfp4_asm(
    A: Tensor,
    B: Tensor,
    ScaleA: Tensor,  # e4m3
    ScaleB: Tensor,  # e4m3
    GlobalScaleA: float,
    GlobalScaleB: float,
    dtype: torch.dtype = dtypes.bf16,  # output dtype: bf16, fp4x2 (packed e2m1), or fp8 (mxfp8)
    a_preshuffle: bool = True,
    kernelName: str = "",
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
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


def gemm_a4w4o4_fake(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    bias: Optional[Tensor] = None,
    alpha: Optional[float] = 1.0,
    beta: Optional[float] = 0.0,
    bpreshuffle: Optional[bool] = True,
    apreshuffle: Optional[bool] = False,
    global_A_scale: Optional[Tensor] = None,
    global_B_scale: Optional[Tensor] = None,
) -> torch.Tensor:
    m = A.numel() // A.shape[-1]
    n = B.shape[0]
    out = _alloc_f4gemm_out(m, n, dtypes.fp4x2, A.device)
    return out.view(*A.shape[:-1], out.shape[-1])


@torch_compile_guard(gen_fake=gemm_a4w4o4_fake)
def gemm_a4w4o4(
    A: Tensor,  # A:[M, K/2] f4x2
    B: Tensor,  # B:[N, K/2] f4x2
    A_scale: Tensor,  # A_scale:[M, K/block] e8m0 (MXFP4) / e4m3 (NVFP4)
    B_scale: Tensor,  # B_scale:[N, K/block]
    bias: Optional[Tensor] = None,
    alpha: Optional[float] = 1.0,
    beta: Optional[float] = 0.0,
    bpreshuffle: Optional[bool] = True,
    apreshuffle: Optional[bool] = False,
    global_A_scale: Optional[Tensor] = None,  # NVFP4 per-tensor
    global_B_scale: Optional[Tensor] = None,  # NVFP4 per-tensor
) -> torch.Tensor:
    """A4W4 GEMM with packed FP4 (e2m1, cvt_scale=1) output ``[*lead, N//2]``.
    gfx1250 only. MXFP4 vs NVFP4 by global-scale presence (see :func:`gemm_a4w4`)."""
    assert (
        get_gfx() in _GFX1250
    ), f"gemm_a4w4o4 (packed FP4 output) is only supported on gfx1250, got {get_gfx()}"
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


def gemm_a4w4o8_fake(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    bias: Optional[Tensor] = None,
    alpha: Optional[float] = 1.0,
    beta: Optional[float] = 0.0,
    bpreshuffle: Optional[bool] = True,
    apreshuffle: Optional[bool] = False,
    global_A_scale: Optional[Tensor] = None,
    global_B_scale: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    m = A.numel() // A.shape[-1]
    n = B.shape[0]
    lead = A.shape[:-1]
    out = _alloc_f4gemm_out(m, n, dtypes.fp8, A.device)
    scale = _alloc_f4gemm_out_scale(m, n, dtypes.fp8, A.device)
    return out.view(*lead, out.shape[-1]), scale.view(*lead, scale.shape[-1])


@torch_compile_guard(gen_fake=gemm_a4w4o8_fake)
def gemm_a4w4o8(
    A: Tensor,  # A:[M, K/2] f4x2
    B: Tensor,  # B:[N, K/2] f4x2
    A_scale: Tensor,  # A_scale:[M, K/block] e8m0 (MXFP4) / e4m3 (NVFP4)
    B_scale: Tensor,  # B_scale:[N, K/block]
    bias: Optional[Tensor] = None,
    alpha: Optional[float] = 1.0,
    beta: Optional[float] = 0.0,
    bpreshuffle: Optional[bool] = True,
    apreshuffle: Optional[bool] = False,
    global_A_scale: Optional[Tensor] = None,  # NVFP4 per-tensor
    global_B_scale: Optional[Tensor] = None,  # NVFP4 per-tensor
) -> Tuple[Tensor, Tensor]:
    """A4W4 GEMM with mxfp8 output: returns ``(fp8 e4m3 data [*lead, N], E8M0
    scale)``. The scale is in the PACKED ``(M/64, N//128, 16, 4)`` layout --
    unpack via :func:`unpack_mxfp8_out_scale`. gfx1250 only. MXFP4 vs NVFP4 by
    global-scale presence (see :func:`gemm_a4w4`)."""
    assert (
        get_gfx() in _GFX1250
    ), f"gemm_a4w4o8 (mxfp8 output) is only supported on gfx1250, got {get_gfx()}"
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
    return o.view(*lead, o.shape[-1]), s.view(*lead, s.shape[-1])


def gen_gemm_a4w4_blockscale_fake_tensors(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    splitK: int = 0,
) -> torch.Tensor:
    return Out


@compile_ops(
    "module_gemm_a4w4_blockscale", gen_fake=gen_gemm_a4w4_blockscale_fake_tensors
)
def gemm_a4w4_blockscale(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    splitK: int = 0,
    kernelName: str = "",
) -> Tensor: ...


@compile_ops(
    "module_gemm_a4w4_blockscale_tune",
    fc_name="gemm_a4w4_blockscale_tune",
    gen_fake=gen_gemm_a4w4_blockscale_fake_tensors,
)
def gemm_a4w4_blockscale_tune(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    kernelId: int,
    splitK: int = 0,
) -> Tensor: ...
