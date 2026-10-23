# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
import importlib.util

import pandas as pd
import torch
from torch import Tensor

from aiter import logger
from aiter.jit.utils.torch_guard import torch_compile_guard

from ..jit.core import AITER_CONFIGS, AITER_LOG_TUNED_CONFIG, compile_ops
from ..jit.utils.asm_guard import require_gfx1250_asm
from ..jit.utils.chip_info import get_cu_num
from ..jit.utils.chip_info import get_gfx_runtime as get_gfx
from ..ops.gemm_op_common import get_padded_m
from ..utility import dtypes

_FLYDSL_AVAILABLE = importlib.util.find_spec("flydsl") is not None


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


# gfx1250 MXFP4 rows (1x32 e8m0 scales in the 32x4 layout), kept apart from the
# a4w4_blockscale table above: its kernels read a different scale layout.
@functools.lru_cache(maxsize=1)
def _load_a4w4_mxscale_tuned(tuned_file: str) -> dict:
    """{(gfx, cu_num, M, N, K): (libtype, kernelName)} of the a4w4_mxscale_bpreshuffle table."""
    try:
        df = pd.read_csv(tuned_file).drop_duplicates()
    except (OSError, pd.errors.EmptyDataError):
        return {}
    return {
        (r.gfx, r.cu_num, r.M, r.N, r.K): (r.libtype, str(r.kernelName))
        for r in df.itertuples(index=False)
    }


@functools.lru_cache(maxsize=1024)
def flydsl_a4w4_fallback_kernel(M: int, N: int, K: int, cu_num: int) -> str | None:
    """FlyDSL kernelName for an untuned shape."""
    from .flydsl.a4w4_mxscale_bpreshuffle_gemm_gfx1250 import (
        a4w4_kernel_launchable,
        a4w4_kernel_name,
    )
    from .flydsl.gemm_tune.flydsl_gemm_a4w4_mxscale_bpreshuffle_wmma_common import (
        kernel_fits_shape,
        kernels_list,
    )

    cdiv = lambda a, b: -(-a // b)
    fill = 0.6 * cu_num
    if M >= 256 and K >= 1024 and fill <= cdiv(M, 256) * cdiv(N, 256) <= cu_num:
        # cluster_m 4 shrinks at launch to the largest that tiles M in whole cluster rows
        for cn in (4, 2, 1):
            name = a4w4_kernel_name(256, 256, 256, cluster_m=4, cluster_n=cn)
            if a4w4_kernel_launchable(name, M, N, K):
                return name

    wgs = lambda ki: cdiv(M, ki.tile_m) * cdiv(N, ki.tile_n) * ki.split_k
    pow2_m = 1 << (M - 1).bit_length()
    general = [
        ki
        for ki in kernels_list.values()
        if not ki.compute_bound
        and min(64, pow2_m) <= ki.tile_m <= min(256, max(16, pow2_m))
        and ki.tile_n >= 64
        and ki.cluster_m == cdiv(M, ki.tile_m)
        and wgs(ki) <= cu_num
        and K // (ki.split_k * ki.tile_k) >= 2
        and ki.cluster_n <= 4
        and kernel_fits_shape(ki, M, N, K)  # the costly check, last
    ]

    def rank(ki):
        full = wgs(ki) >= fill
        warps = (1, 4) if ki.tile_m <= 64 and ki.tile_n >= 128 else (2, 2)
        return (
            not full,
            ki.split_k if full else 0,
            -wgs(ki),
            -ki.tile_m * ki.tile_n / (ki.tile_m + ki.tile_n),
            -ki.tile_m,
            (512, 256, 128, 1024).index(ki.tile_k),
            (ki.m_warp, ki.n_warp) != warps,
            (3, 4, 2).index(ki.num_buffers),
            -ki.cluster_n,
        )

    return min(general, key=rank).name if general else None


@functools.lru_cache(maxsize=1024)
def _a4w4_mxscale_config(M: int, N: int, K: int) -> tuple[str, str] | None:
    """(libtype, kernelName) for a gfx1250 shape: the tuned row at M or a padded M (a FlyDSL
    row that cannot launch for the real M is skipped), else the FlyDSL fallback."""
    if not _FLYDSL_AVAILABLE:  # optional dependency: asm serves all
        return None
    from .flydsl.a4w4_mxscale_bpreshuffle_gemm_gfx1250 import a4w4_kernel_launchable

    gfx, cu_num = get_gfx(), get_cu_num()
    rows = _load_a4w4_mxscale_tuned(
        AITER_CONFIGS.AITER_CONFIG_GEMM_A4W4_MXSCALE_BPRESHUFFLE_FILE
    )
    for gl in (None, 0, 1):
        row = rows.get(
            (gfx, cu_num, M if gl is None else get_padded_m(M, N, K, gl), N, K)
        )
        if row is not None and (
            row[0] != "flydsl" or a4w4_kernel_launchable(row[1], M, N, K)
        ):
            return row
    name = flydsl_a4w4_fallback_kernel(M, N, K, cu_num)
    if name is None:
        return None
    logger.warning(
        f"[gfx1250] gemm_a4w4 untuned M={M}, N={N}, K={K}; "
        f"falling back to flydsl kernel '{name}'."
    )
    return "flydsl", name


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

    is_nvfp4 = global_A_scale is not None or global_B_scale is not None
    N = B.shape[0]
    K = A.shape[-1] * 2  # A is packed fp4x2 -> 2 values per column
    if dtype not in (dtypes.bf16, dtypes.fp8):
        raise NotImplementedError(f"gfx1250 F4GEMM: unsupported output dtype {dtype}")
    if K % 32 != 0:  # B 16x16 preshuffle
        raise NotImplementedError(f"gfx1250 F4GEMM requires K%32==0, got K={K}")
    if N % 16 != 0:  # B 16x16 preshuffle
        raise NotImplementedError(f"gfx1250 F4GEMM requires N%16==0, got N={N}")
    if apreshuffle and m % 16 != 0:  # A 16x16 preshuffle
        raise NotImplementedError(
            f"gfx1250 F4GEMM apreshuffle requires M%16==0, got M={m}"
        )
    if is_nvfp4:
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
    if dtype == dtypes.fp8:
        raise NotImplementedError(
            "gemm_a4w4 returns one plain-dtype tensor; use gemm_a4w4o8"
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

    On gfx1250 the a4w4_mxscale_bpreshuffle table picks FlyDSL or the F4GEMM asm
    kernels per shape. MXFP4 vs NVFP4 is selected by the presence of ``global_A_scale``/
    ``global_B_scale`` (NVFP4 per-tensor global scales).
    """
    # Low-precision output has a different return arity/shape (mxfp8 is a
    # (data, scale) tuple), so it lives on a separate fixed-arity op.
    if dtype == dtypes.fp8:
        raise NotImplementedError(
            "gemm_a4w4 returns one plain-dtype tensor; use gemm_a4w4o8 (mxfp8) "
            "for low-precision output"
        )
    m = A.numel() // A.shape[-1]
    n = B.shape[0]
    k = A.shape[-1] * 2
    gfx_arch = get_gfx()
    if gfx_arch in ["gfx1250"]:
        # FlyDSL serves plain MXFP4 with row-major A and a bf16 output; the rest is asm.
        plain_mxfp4 = (
            dtype == dtypes.bf16
            and not apreshuffle
            and global_A_scale is None
            and global_B_scale is None
            and bias is None
            and alpha in (None, 1.0)
            and beta in (None, 0.0)
        )
        config = _a4w4_mxscale_config(m, n, k) if plain_mxfp4 else None
        if config is not None and config[0] == "flydsl":
            from .flydsl.a4w4_mxscale_bpreshuffle_gemm_gfx1250 import (
                a4w4_operands_aligned,
                run_gemm_a4w4_mxscale_bpreshuffle_gfx1250,
            )

            A2 = A.view(m, k // 2)
            if a4w4_operands_aligned(A2, B, A_scale, B_scale):  # else asm
                out = torch.empty((m, n), dtype=dtype, device=A.device)
                run_gemm_a4w4_mxscale_bpreshuffle_gfx1250(
                    A2, B, A_scale, B_scale, out, config[1]
                )
                return out.view(*A.shape[:-1], n)
        require_gfx1250_asm("gemm_a4w4")
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
    out: Tensor,  # Out: bf16 [M,N] / fp8 [M,N]
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
    out: Tensor,  # Out: bf16 [M,N] / fp8 [M,N]
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


def _alloc_f4gemm_out(
    M: int, N: int, dtype: torch.dtype, device, y: Tensor | None = None
) -> Tensor:
    """Allocate or reuse the F4GEMM output. bf16 output is a plain ``[M, N]``; mxfp8 output
    is a plain ``[M, N]`` fp8 (its E8M0 scale is a separate tensor, see
    :func:`_alloc_f4gemm_out_scale`)."""
    if y is not None:
        if (
            y.shape != (M, N)
            or y.dtype != dtype
            or y.device != device
            or not y.is_contiguous()
        ):
            raise ValueError(
                f"y must be contiguous with shape {(M, N)}, dtype {dtype}, device {device}"
            )
        return y
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


def gemm_mxfp4_asm(
    A: Tensor,  # A:[M, K/2] fp4x2
    B: Tensor,  # B:[N, K/2] fp4x2
    ScaleA: Tensor,  # ScaleA:[M, K/32] e8m0
    ScaleB: Tensor,  # ScaleB:[N, K/32] e8m0
    dtype: torch.dtype = dtypes.bf16,  # output dtype: bf16 or fp8 (mxfp8)
    a_preshuffle: bool = True,
    kernelName: str = "",
    *,
    y: Tensor | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    """MXFP4 GEMM (preload SGPR mode). D = A * B with e8m0 scales. ``dtype``:
    bf16 ``[M,N]`` or mxfp8 (``dtypes.fp8``) returning
    ``(out_fp8 [M,N], scale_e8m0)``. ``scale_e8m0`` is in the PACKED
    ``(M/64,N//128,16,4)`` layout -- unpack via
    :func:`unpack_mxfp8_out_scale`. Optional ``y`` supplies a contiguous
    ``[M,N]`` output with the requested dtype on A's device, as in Gluon.
    An empty batch returns empty output buffers without launching a kernel."""
    require_gfx1250_asm("gemm_mxfp4_asm")
    M = A.shape[0]
    N = B.shape[0]
    out = _alloc_f4gemm_out(M, N, dtype, A.device, y)
    out_scale = _alloc_f4gemm_out_scale(M, N, dtype, A.device)
    if M == 0:
        return (out, out_scale) if out_scale is not None else out
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
    dtype: torch.dtype = dtypes.bf16,  # output dtype: bf16 or fp8 (mxfp8)
    a_preshuffle: bool = True,
    kernelName: str = "",
    *,
    y: Tensor | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    """NVFP4 GEMM (preload SGPR mode). D = A * B with e4m3 scales + global alphas.
    ``dtype``: bf16 ``[M,N]`` or mxfp8 (``dtypes.fp8``) returning
    ``(out_fp8 [M,N], scale_e8m0)``. ``scale_e8m0`` is in the PACKED
    ``(M/64,N//128,16,4)`` layout -- unpack via
    :func:`unpack_mxfp8_out_scale`. Optional ``y`` supplies a contiguous
    ``[M,N]`` output with the requested dtype on A's device.
    An empty batch returns empty output buffers without launching a kernel."""
    require_gfx1250_asm("gemm_nvfp4_asm")
    M = A.shape[0]
    N = B.shape[0]
    out = _alloc_f4gemm_out(M, N, dtype, A.device, y)
    out_scale = _alloc_f4gemm_out_scale(M, N, dtype, A.device)
    if M == 0:
        return (out, out_scale) if out_scale is not None else out
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


def gemm_a4w4o8_fake(
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


@torch_compile_guard(gen_fake=gemm_a4w4o8_fake)
def gemm_a4w4o8(
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
    global-scale presence (see :func:`gemm_a4w4`)."""
    assert (
        get_gfx() in _GFX1250
    ), f"gemm_a4w4o8 (mxfp8 output) is only supported on gfx1250, got {get_gfx()}"
    require_gfx1250_asm("gemm_a4w4o8")
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


# torch_compile_guard infers each op's schema from the decorated function, then
# hands the fake the very same arguments, so a fake has to mirror its own op's
# signature -- the two ops below disagree past `Out` and cannot share one.
def gen_gemm_a4w4_blockscale_fake_tensors(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    splitK: int = 0,
    kernelName: str = "",
) -> None:
    return None


def gen_gemm_a4w4_blockscale_tune_fake_tensors(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    kernelId: int = 0,
    splitK: int = 0,
) -> None:
    return None


# The C++ side of both ops is torch-free: it takes pybind aiter_tensor_t and
# writes into Out in place.  `develop=True` makes core.py convert every
# torch.Tensor argument via torch_to_aiter_pybind() and push the current stream
# through _set_current_hip_stream() before the call.  The public wrappers below
# return Out so callers still get a torch.Tensor.
@compile_ops(
    "module_gemm_a4w4_blockscale",
    fc_name="gemm_a4w4_blockscale",
    gen_fake=gen_gemm_a4w4_blockscale_fake_tensors,
    develop=True,
)
def _gemm_a4w4_blockscale(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    splitK: int = 0,
    kernelName: str = "",
) -> None: ...


def gemm_a4w4_blockscale(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    splitK: int = 0,
    kernelName: str = "",
) -> Tensor:
    _gemm_a4w4_blockscale(XQ, WQ, x_scale, w_scale, Out, splitK, kernelName)
    return Out


@compile_ops(
    "module_gemm_a4w4_blockscale_tune",
    fc_name="gemm_a4w4_blockscale_tune",
    gen_fake=gen_gemm_a4w4_blockscale_tune_fake_tensors,
    develop=True,
)
def _gemm_a4w4_blockscale_tune(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    kernelId: int,
    splitK: int = 0,
) -> None: ...


def gemm_a4w4_blockscale_tune(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    kernelId: int,
    splitK: int = 0,
) -> Tensor:
    _gemm_a4w4_blockscale_tune(XQ, WQ, x_scale, w_scale, Out, kernelId, splitK)
    return Out
