# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host dispatch for the gfx950 FlyDSL MX-scale preshuffle GEMM."""

from __future__ import annotations

import torch

from aiter.utility.graph_alloc import persistent_alloc

_OUT_DTYPE_STR = {torch.bfloat16: "bf16", torch.float16: "fp16"}
_SUPPORTED_A = ("fp4", "fp8")
_SUPPORTED_B = ("fp4", "fp8")
# Matches kernels.mxscale_preshuffle.PRESHUFFLE_M_MAX.
PRESHUFFLE_M_MAX = 65536


def _packed_row_bytes(dtype: str, k: int) -> int:
    return k // 2 if dtype == "fp4" else k


def _scale_nbytes(rows: int, k: int) -> int:
    return ((rows + 31) // 32 * 32) * (k // 32)


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _require_gfx950() -> None:
    from flydsl.runtime.device import get_rocm_arch

    architecture = str(get_rocm_arch() or "").split(":", 1)[0]
    if architecture != "gfx950":
        raise RuntimeError(
            f"MX-scale preshuffle GEMM requires gfx950, got {architecture}"
        )


def _check_operand_layouts(
    A: torch.Tensor,
    B: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    Out: torch.Tensor,
    *,
    a_dtype: str,
    b_dtype: str,
    m: int,
    n: int,
    k: int,
) -> None:
    a_bytes = _packed_row_bytes(a_dtype, k)
    b_bytes = _packed_row_bytes(b_dtype, k)
    expected_a = m * a_bytes
    expected_b = n * b_bytes
    expected_a_scale = _scale_nbytes(m, k)
    expected_b_scale = _scale_nbytes(n, k)
    got_a = _tensor_nbytes(A)
    got_b = _tensor_nbytes(B)
    got_a_scale = _tensor_nbytes(a_scale)
    got_b_scale = _tensor_nbytes(b_scale)
    if got_a != expected_a:
        raise ValueError(
            f"A nbytes {got_a} != packed {m}x{k} {a_dtype} ({expected_a} bytes)"
        )
    if got_b != expected_b:
        raise ValueError(
            f"B nbytes {got_b} != shuffled {n}x{k} {b_dtype} ({expected_b} bytes)"
        )
    if got_a_scale != expected_a_scale:
        raise ValueError(
            f"a_scale nbytes {got_a_scale} != ceil(M/32)*32*(K/32) "
            f"({expected_a_scale} bytes)"
        )
    if got_b_scale != expected_b_scale:
        raise ValueError(
            f"b_scale nbytes {got_b_scale} != ceil(N/32)*32*(K/32) "
            f"({expected_b_scale} bytes)"
        )
    if Out.shape != (m, n):
        raise ValueError(f"Out shape {tuple(Out.shape)} != ({m}, {n})")


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
    """Run MXFP4/8 A by preshuffled MXFP4/8 B on gfx950."""
    from .kernels.mxscale_preshuffle import launch_gemm, launch_splitk_reduce
    from .kernels.tensor_shim import ptr_arg

    if a_dtype not in _SUPPORTED_A:
        raise ValueError(f"unsupported a_dtype {a_dtype!r}; expected 'fp4' or 'fp8'")
    if b_dtype not in _SUPPORTED_B:
        raise ValueError(f"unsupported b_dtype {b_dtype!r}; expected 'fp4' or 'fp8'")

    M = int(A.shape[0])
    K = int(A.shape[-1]) * (2 if a_dtype == "fp4" else 1)
    N = int(Out.shape[-1])
    if A.is_cuda:
        _require_gfx950()
    if M > PRESHUFFLE_M_MAX:
        raise ValueError(
            f"M ({M}) exceeds {PRESHUFFLE_M_MAX}; the mxscale kernel views A "
            "through a layout bounded by that many rows"
        )
    if N % int(tile_n) != 0:
        raise ValueError(f"N ({N}) is not a multiple of tile_n ({tile_n})")
    if K % int(tile_k) != 0:
        raise ValueError(f"K ({K}) is not a multiple of tile_k ({tile_k})")
    if K % 128 != 0:
        raise ValueError(
            f"K ({K}) must be a multiple of 128 for MXFP microscale; got {K}"
        )
    _check_operand_layouts(
        A,
        B,
        a_scale,
        b_scale,
        Out,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        m=M,
        n=N,
        k=K,
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
        if K % split_k != 0 or k_per_split % int(tile_k) != 0 or k_per_split % 256 != 0:
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
        int(waves_per_eu),
        int(xcd_swizzle),
        split_k,
    )
    if split_k == 1:
        launch_gemm(ptr_arg(Out), *launch_args)
        return Out

    workspace_shape = (split_k, M, N)
    if splitk_workspace is None:
        tmp = _splitk_workspace(A.device, workspace_shape)
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


def _splitk_workspace(
    device: torch.device, shape: tuple[int, int, int]
) -> torch.Tensor:
    if device.type != "cuda":
        return torch.empty(shape, dtype=torch.float32, device=device)
    with persistent_alloc(device):
        return torch.empty(shape, dtype=torch.float32, device=device)


_TUNED_CACHE = {}


def _lookup_tuned(M, N, K, a_dtype, b_dtype, tuned_file=None):
    """Look up an exact (gfx, CU, shape, operand dtype) row."""
    import pandas as pd

    from aiter.jit.core import AITER_CONFIGS
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

    tune_file = tuned_file or AITER_CONFIGS.AITER_CONFIG_GEMM_MXSCALE_PRESHUFFLE_FILE
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
    return table.get((get_gfx_runtime(), get_cu_num(), M, N, K, a_dtype, b_dtype))


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

    candidates = [instance for _, instance in candidates_for(a_dtype, b_dtype, M, N, K)]
    if not candidates:
        return None
    untuned = [instance for instance in candidates if instance.split_k == 1]
    pool = untuned if untuned else candidates
    target_m = min(max((M + 31) // 32 * 32, 32), 128)
    return max(
        pool,
        key=lambda instance: (
            instance.tile_k,
            instance.tile_n,
            -abs(instance.tile_m - target_m),
            -instance.waves_per_eu,
            -instance.xcd_swizzle,
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
        cfg = config if config is not None else _lookup_tuned(M, N, K, a_dtype, b_dtype)
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
