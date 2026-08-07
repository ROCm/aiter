# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host dispatch for the gfx950 FlyDSL MX-scale preshuffle GEMM."""

from __future__ import annotations

import torch

from aiter.ops.flydsl.utils import is_flydsl_available

_OUT_DTYPE_STR = {torch.bfloat16: "bf16", torch.float16: "fp16"}


def flydsl_mxscale_preshuffle_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    Out: torch.Tensor,
    *,
    a_dtype: str,
    b_dtype: str = "fp4",
    tile_m: int,
    tile_n: int,
    tile_k: int,
    waves_per_eu: int = 0,
    xcd_swizzle: int = 0,
    split_k: int = 1,
    splitk_workspace: torch.Tensor | None = None,
    stream=None,
) -> torch.Tensor:
    """Run MXFP4/6/8 A by preshuffled MXFP4/8 B on gfx950."""
    if not is_flydsl_available():
        raise RuntimeError(
            "flydsl is not available; cannot run mxscale_preshuffle GEMM"
        )

    from .kernels.mxscale_preshuffle import launch_gemm
    from .kernels.tensor_shim import ptr_arg

    if a_dtype not in ("fp4", "fp6", "fp8"):
        raise ValueError(
            f"unsupported a_dtype {a_dtype!r}; expected 'fp4', 'fp6', or 'fp8'"
        )
    if b_dtype not in ("fp4", "fp8"):
        raise ValueError(f"unsupported b_dtype {b_dtype!r}; expected 'fp4' or 'fp8'")

    M = int(A.shape[0])
    K = int(A.shape[-1]) * (2 if a_dtype == "fp4" else 1)
    N = int(Out.shape[-1])
    if N % int(tile_n) != 0:
        raise ValueError(f"N ({N}) is not a multiple of tile_n ({tile_n})")
    if K % int(tile_k) != 0:
        raise ValueError(f"K ({K}) is not a multiple of tile_k ({tile_k})")
    if K % 128 != 0:
        raise ValueError(
            f"K ({K}) must be a multiple of 128 for MXFP microscale; got {K}"
        )
    out_dtype = _OUT_DTYPE_STR.get(Out.dtype)
    if out_dtype is None:
        raise ValueError(
            f"unsupported Out dtype {Out.dtype}; expected bfloat16 or float16"
        )

    st = stream if stream is not None else torch.cuda.current_stream()
    split_k = int(split_k)
    if split_k < 1:
        raise ValueError(f"split_k must be positive, got {split_k}")
    if split_k > 1:
        k_per_split = K // split_k
        if (
            K % split_k != 0
            or k_per_split % int(tile_k) != 0
            or k_per_split % 256 != 0
        ):
            raise ValueError(
                f"illegal split_k={split_k} for K={K}, tile_k={tile_k}: "
                f"K/split_k ({k_per_split}) must be a multiple of tile_k and 256"
            )

    launch_args = (
        ptr_arg(A),
        ptr_arg(B),
        ptr_arg(a_scale),
        ptr_arg(b_scale),
        M,
        N,
        st,
        N,
        K,
        int(tile_m),
        int(tile_n),
        int(tile_k),
        a_dtype,
        out_dtype,
        b_dtype,
        1,  # batch
        -1,  # a_row_stride
        -1,  # a_batch_stride
        -1,  # sca_row_stride
        -1,  # sca_batch_stride
        -1,  # c_row_stride
        -1,  # c_batch_stride
        int(waves_per_eu),
        int(xcd_swizzle),
        split_k,
    )
    if split_k == 1:
        launch_gemm(ptr_arg(Out), *launch_args)
        return Out

    from .kernels.mxscale_preshuffle import launch_splitk_reduce

    workspace_shape = (split_k, M, N)
    if splitk_workspace is None:
        tmp = torch.empty(workspace_shape, dtype=torch.float32, device=A.device)
    else:
        if (
            tuple(splitk_workspace.shape) != workspace_shape
            or splitk_workspace.dtype != torch.float32
            or splitk_workspace.device != A.device
            or not splitk_workspace.is_contiguous()
        ):
            raise ValueError(
                "splitk_workspace must be contiguous fp32 on A.device with "
                f"shape {workspace_shape}; got shape={tuple(splitk_workspace.shape)}, "
                f"dtype={splitk_workspace.dtype}, device={splitk_workspace.device}, "
                f"contiguous={splitk_workspace.is_contiguous()}"
            )
        tmp = splitk_workspace
    launch_gemm(ptr_arg(tmp), *launch_args)
    launch_splitk_reduce(
        ptr_arg(tmp),
        ptr_arg(Out),
        (M * N) // 2,
        M * N,
        st,
        split_k,
        out_dtype,
    )
    return Out


_TUNED_CACHE = {}


def _lookup_tuned(M, N, K, a_dtype, b_dtype, tuned_file=None):
    """Look up an exact (gfx, CU, shape, operand dtype) row."""
    import pandas as pd

    from aiter.jit.core import AITER_CONFIGS
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

    tune_file = (
        tuned_file or AITER_CONFIGS.AITER_CONFIG_GEMM_MXSCALE_PRESHUFFLE_FILE
    )
    if tune_file not in _TUNED_CACHE:
        try:
            frame = pd.read_csv(tune_file).drop_duplicates()
            _TUNED_CACHE[tune_file] = frame.set_index(
                ["gfx", "cu_num", "M", "N", "K", "a_dtype", "b_dtype"]
            ).to_dict("index")
        except (FileNotFoundError, KeyError, ValueError, pd.errors.EmptyDataError):
            _TUNED_CACHE[tune_file] = None
    table = _TUNED_CACHE[tune_file]
    if not table:
        return None
    return table.get(
        (get_gfx_runtime(), get_cu_num(), M, N, K, a_dtype, b_dtype)
    )


def get_mxscale_preshuffle_config(
    M: int,
    N: int,
    K: int,
    *,
    a_dtype: str = "fp8",
    b_dtype: str = "fp8",
    tuned_file=None,
):
    """Return only an exact runtime config row; never approximate a signature."""
    return _lookup_tuned(
        int(M), int(N), int(K), a_dtype, b_dtype, tuned_file=tuned_file
    )


def _heuristic_tile(a_dtype, b_dtype, M, N, K):
    from .gemm_tune.flydsl_gemm_mxscale_preshuffle_common import candidates_for

    candidates = [
        instance for _, instance in candidates_for(a_dtype, b_dtype, M, N, K)
    ]
    if not candidates:
        return None
    target_m = min(max((M + 31) // 32 * 32, 32), 128)
    return max(
        candidates,
        key=lambda instance: (
            instance.tile_k,
            instance.tile_n,
            -abs(instance.tile_m - target_m),
            -instance.waves_per_eu,
        ),
    )


def gemm_mxscale_preshuffle(
    A,
    B,
    a_scale,
    b_scale,
    Out,
    *,
    a_dtype,
    b_dtype,
    tile_m=None,
    tile_n=None,
    tile_k=None,
    waves_per_eu=None,
    xcd_swizzle=None,
    split_k=None,
    config=None,
    require_tuned=False,
    splitk_workspace=None,
    stream=None,
):
    """Dispatch explicit config, exact tuned row, then a legal heuristic."""
    M = int(A.shape[0])
    N = int(Out.shape[-1])
    K = int(A.shape[-1]) * (2 if a_dtype == "fp4" else 1)

    explicit_tiles = (tile_m, tile_n, tile_k)
    if any(value is not None for value in explicit_tiles) and not all(
        value is not None for value in explicit_tiles
    ):
        raise ValueError("tile_m, tile_n, and tile_k must be provided together")

    if tile_m is None:
        cfg = (
            config
            if config is not None
            else _lookup_tuned(M, N, K, a_dtype, b_dtype)
        )
        if cfg is not None and cfg.get("kernelName"):
            from .gemm_tune.flydsl_gemm_mxscale_preshuffle_common import (
                parse_kernel_name,
            )

            parsed = parse_kernel_name(cfg["kernelName"])
            if parsed is not None:
                out_dtype = _OUT_DTYPE_STR.get(Out.dtype)
                encoded_signature = (
                    parsed["a_dtype"],
                    parsed["b_dtype"],
                    parsed["out_dtype"],
                )
                runtime_signature = (a_dtype, b_dtype, out_dtype)
                if encoded_signature != runtime_signature:
                    raise ValueError(
                        f"kernelName {cfg['kernelName']!r} encodes "
                        f"{encoded_signature}, expected {runtime_signature}"
                    )
                tile_m = parsed["tile_m"]
                tile_n = parsed["tile_n"]
                tile_k = parsed["tile_k"]
                if waves_per_eu is None:
                    waves_per_eu = parsed["waves_per_eu"]
                if xcd_swizzle is None:
                    xcd_swizzle = parsed["xcd_swizzle"]
                if split_k is None:
                    split_k = parsed["split_k"]
        if tile_m is None and require_tuned:
            raise RuntimeError(
                "no exact mxscale_preshuffle tune for "
                f"M={M}, N={N}, K={K}, a_dtype={a_dtype}, b_dtype={b_dtype}"
            )
        if tile_m is None:
            instance = _heuristic_tile(a_dtype, b_dtype, M, N, K)
            if instance is None:
                raise ValueError(
                    f"no legal tile for M={M} N={N} K={K} "
                    f"{a_dtype}/{b_dtype}; pass tile_m/n/k explicitly"
                )
            tile_m = instance.tile_m
            tile_n = instance.tile_n
            tile_k = instance.tile_k
            if waves_per_eu is None:
                waves_per_eu = instance.waves_per_eu
            if xcd_swizzle is None:
                xcd_swizzle = instance.xcd_swizzle
            if split_k is None:
                split_k = instance.split_k

    return flydsl_mxscale_preshuffle_gemm(
        A,
        B,
        a_scale,
        b_scale,
        Out,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        waves_per_eu=0 if waves_per_eu is None else waves_per_eu,
        xcd_swizzle=0 if xcd_swizzle is None else xcd_swizzle,
        split_k=1 if split_k is None else split_k,
        splitk_workspace=splitk_workspace,
        stream=stream,
    )
