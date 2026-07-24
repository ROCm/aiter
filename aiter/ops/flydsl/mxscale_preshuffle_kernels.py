# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host dispatcher for the FlyDSL MXFP4/MXFP6/MXFP8 preshuffle GEMM (gfx950 MFMA).

Wraps the self-contained ``@flyc.jit launch_gemm`` in
``aiter/ops/flydsl/kernels/mxscale_preshuffle.py``. ``launch_gemm`` already caches
and dispatches per distinct Constexpr config (N, K, tile_*, a_dtype, b_dtype,
out_dtype, batch, strides, waves_per_eu), so this layer only marshals torch
tensors to raw pointers via ``tensor_shim.ptr_arg`` and forwards the config.

Operand convention (matches the kernel's host preshuffle):
    A       : [M, K]   row-major, NOT preshuffled  (fp8/fp6 = 1 byte/code, fp4 = 2 codes/byte)
    B       : preshuffled via aiter.ops.shuffle.shuffle_weight(., (16, 16))  (fp4 or fp8 weight)
    a_scale : per-1x32 E8M0, shuffle_scale_a16w4'd
    b_scale : per-1x32 E8M0, shuffle_scale_a16w4'd
    Out     : [M, N]   bf16 / fp16
"""

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
    stream=None,
) -> torch.Tensor:
    """Run the gfx950 MXFP4/6/8 preshuffle GEMM. a8w8 = a_dtype="fp8", b_dtype="fp8".

    A is [M, K]; N is taken from Out ([M, N]); K from A. Returns Out.

    split_k>1 splits the K reduction across grid.z: each split writes an fp32
    partial slab to a scratch tmp[split_k, M, N], then a reduce kernel sums the
    slabs into Out (bf16/fp16). Helps small-M / large-K (low-occupancy) shapes.
    """
    if not is_flydsl_available():
        raise RuntimeError(
            "flydsl is not available; cannot run mxscale_preshuffle GEMM"
        )

    from .kernels.mxscale_preshuffle import launch_gemm
    from .kernels.tensor_shim import ptr_arg

    # Logical K: fp4 A packs 2 codes/byte (A last dim = K//2); fp6/fp8 A = 1 byte/code.
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
    if split_k > 1:
        # split-K legality (same constraints the tuner enforces in fits_shape):
        # per-split K must be a whole number of tile_k tiles AND 256-K scale chunks.
        k_per_split = K // split_k
        if K % split_k != 0 or k_per_split % int(tile_k) != 0 or k_per_split % 256 != 0:
            raise ValueError(
                f"illegal split_k={split_k} for K={K}, tile_k={tile_k}: "
                f"K/split_k ({k_per_split}) must be a multiple of tile_k and 256"
            )

    if split_k == 1:
        launch_gemm(
            ptr_arg(Out),
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
            1,  # k_batch
        )
        return Out

    # split-K: GEMM -> fp32 partial slabs tmp[split_k, M, N] -> fused fp32 reduce -> Out.
    from .kernels.mxscale_preshuffle import launch_splitk_reduce

    tmp = torch.empty((split_k, M, N), dtype=torch.float32, device=A.device)
    launch_gemm(
        ptr_arg(tmp),
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
        split_k,  # k_batch
    )
    launch_splitk_reduce(
        ptr_arg(tmp),
        ptr_arg(Out),
        (M * N) // 2,  # n_out_dw (2 out elems per dword)
        M * N,  # slab_stride_dw (fp32: 1 dword/elem)
        st,
        split_k,
        out_dtype,
    )
    return Out


# ── Tuned dispatch: explicit tile args > tuned CSV > heuristic default ─────────

_TUNED_CACHE = {}


def _lookup_tuned(M, N, K, a_dtype, b_dtype, tuned_file=None):
    """Look up a tuned row keyed by (gfx, cu_num, M, N, K, a_dtype, b_dtype)."""
    import pandas as pd

    from aiter.jit.core import AITER_CONFIGS
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime as get_gfx

    tf = tuned_file or AITER_CONFIGS.AITER_CONFIG_GEMM_MXSCALE_PRESHUFFLE_FILE
    if tf not in _TUNED_CACHE:
        try:
            df = pd.read_csv(tf).drop_duplicates()
            _TUNED_CACHE[tf] = df.set_index(
                ["gfx", "cu_num", "M", "N", "K", "a_dtype", "b_dtype"]
            ).to_dict("index")
        except Exception:
            # missing / empty / malformed / missing-column CSV -> no tuned config
            _TUNED_CACHE[tf] = None
    tbl = _TUNED_CACHE[tf]
    if not tbl:
        return None
    return tbl.get((get_gfx(), get_cu_num(), M, N, K, a_dtype, b_dtype))


def _heuristic_tile(a_dtype, b_dtype, M, N, K):
    """Pick a legal tile from the catalog when no tune/explicit config is given."""
    from .gemm_tune.flydsl_gemm_mxscale_preshuffle_common import candidates_for

    cands = [ki for _, ki in candidates_for(a_dtype, b_dtype, M, N, K)]
    if not cands:
        return None
    target_m = min(max((M + 31) // 32 * 32, 32), 128)
    return max(
        cands,
        key=lambda ki: (
            ki.tile_k,
            ki.tile_n,
            -abs(ki.tile_m - target_m),
            -ki.waves_per_eu,
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
    stream=None,
):
    """Auto-dispatched MXFP4/MXFP8 preshuffle GEMM.

    Tile selection precedence: explicit tile_m/n/k args > tuned CSV
    (config or lookup by (gfx,cu_num,M,N,K,a_dtype,b_dtype)) > heuristic default.
    """
    M = int(A.shape[0])
    N = int(Out.shape[-1])
    K = int(A.shape[-1]) * (2 if a_dtype == "fp4" else 1)

    if tile_m is None or tile_n is None or tile_k is None:
        cfg = config if config is not None else _lookup_tuned(M, N, K, a_dtype, b_dtype)
        if cfg is not None and cfg.get("kernelName"):
            from .gemm_tune.flydsl_gemm_mxscale_preshuffle_common import (
                parse_kernel_name,
            )

            p = parse_kernel_name(cfg["kernelName"])
            if p is not None:
                tile_m, tile_n, tile_k = p["tile_m"], p["tile_n"], p["tile_k"]
                if waves_per_eu is None:
                    waves_per_eu = p["waves_per_eu"]
                if xcd_swizzle is None:
                    xcd_swizzle = p["xcd_swizzle"]
                if split_k is None:
                    split_k = p["split_k"]
        if tile_m is None:  # still unresolved -> heuristic
            ki = _heuristic_tile(a_dtype, b_dtype, M, N, K)
            if ki is None:
                raise ValueError(
                    f"no legal tile for M={M} N={N} K={K} {a_dtype}/{b_dtype}; pass tile_m/n/k explicitly"
                )
            tile_m, tile_n, tile_k = ki.tile_m, ki.tile_n, ki.tile_k
            if waves_per_eu is None:
                waves_per_eu = ki.waves_per_eu
            if xcd_swizzle is None:
                xcd_swizzle = ki.xcd_swizzle
            if split_k is None:
                split_k = ki.split_k

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
        waves_per_eu=(0 if waves_per_eu is None else waves_per_eu),
        xcd_swizzle=(0 if xcd_swizzle is None else xcd_swizzle),
        split_k=(1 if split_k is None else split_k),
        stream=stream,
    )
