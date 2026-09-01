# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Two-pass fp32 split-K preshuffle GEMM driver.

Production dispatch is ``aiter.ops.flydsl.splitk_bpreshuffle_common.dispatch_flydsl_splitk``,
used by the a8w8 and a4w4 tuned-config paths. Direct calls remain for the
correctness tests and isolated perf runs.
"""

import os

import flydsl.expr as fx
import torch

from aiter.ops.flydsl.kernels.preshuffle_gemm_splitk import (
    compile_preshuffle_gemm_splitk,
)
from aiter.ops.flydsl.kernels.preshuffle_gemm_splitk_reduce import (
    compile_preshuffle_gemm_splitk_reduce,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, unused_tensor_arg

# Predicate the partial-tile store to the real M so a ragged block skips its padding
# rows. Off by default: the reduce ignores those rows either way, so this is a pure
# store-traffic optimization whose payoff is shape-dependent.
USE_M_BOUNDED_STORE = os.environ.get("AITER_SPLITK_M_BOUNDED_STORE", "0") == "1"


def splitk_workspace_shape(m: int, n: int, tile_m: int, split_k: int):
    """(split_k, m_pad, N) — M padded to whole GEMM tiles; see the kernel docstring."""
    m_pad = ((m + tile_m - 1) // tile_m) * tile_m
    return (split_k, m_pad, n)


def flydsl_preshuffle_gemm_splitk_a8(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    split_k: int,
    *,
    workspace: torch.Tensor | None = None,
    use_async_copy: int = 0,
    waves_per_eu: int = 0,
    xcd_swizzle: int = 0,
    lds_stage: int = 2,
    enable_scheduler: bool = True,
    scale_mode: str = "epilogue",
    use_m_bounded_store: bool | None = None,
    in_dtype: str | None = None,
    stage_a_scales: bool = False,
    sched_dsrd: int | None = None,
) -> torch.Tensor:
    """Split-K fp8 preshuffle GEMM: partial pass into an fp32 workspace, then reduce.

    With ``scale_mode="epilogue"``, ``x_scale``/``w_scale`` are the per-row/per-col
    fp32 scale tensors the preshuffle path expects (broadcast copies of the
    per-tensor scalars for per-tensor quant). With ``scale_mode="blockscale"`` they
    are the 128-block scales instead: ``x_scale`` fp32 ``[K/128, M]`` (transposed)
    and ``w_scale`` fp32 ``[N/128, K/128]``.
    """
    m, k = XQ.shape[0], XQ.shape[-1]
    n = WQ.shape[0]
    if in_dtype is None:
        in_dtype = "int8" if XQ.dtype == torch.int8 else "fp8"
    if in_dtype == "fp4":
        # XQ/WQ carry two fp4 codes per byte, so the stored extent is K/2.
        k *= 2

    if n % tile_n != 0 or k % tile_k != 0:
        raise ValueError(
            f"[FlyDSL] N={n} must be a multiple of tile_n={tile_n} and "
            f"K={k} of tile_k={tile_k}"
        )
    if Out.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"[FlyDSL] unsupported output dtype {Out.dtype}")
    out_dtype = "bf16" if Out.dtype == torch.bfloat16 else "fp16"

    # One split means the reduce would only copy-and-downcast a single slice, so the
    # GEMM writes the final output itself and the workspace is not needed at all.
    direct_out = split_k == 1

    ws_shape = splitk_workspace_shape(m, n, tile_m, split_k)
    if direct_out:
        workspace = None
    elif workspace is None:
        workspace = torch.empty(ws_shape, device=XQ.device, dtype=torch.float32)
    elif workspace.shape != ws_shape or workspace.dtype != torch.float32:
        raise ValueError(
            f"[FlyDSL] workspace must be fp32 {ws_shape}, got "
            f"{tuple(workspace.shape)} {workspace.dtype}"
        )

    gemm_exe = compile_preshuffle_gemm_splitk(
        N=n,
        K=k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        split_k=split_k,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        use_async_copy=bool(use_async_copy),
        waves_per_eu=None if waves_per_eu <= 0 else waves_per_eu,
        enable_scheduler=bool(enable_scheduler),
        xcd_swizzle=int(xcd_swizzle),
        lds_stage=int(lds_stage),
        scale_mode=scale_mode,
        scale_block_k=32 if scale_mode == "mxfp4" else 128,
        use_m_bounded_store=(
            USE_M_BOUNDED_STORE if use_m_bounded_store is None else use_m_bounded_store
        ),
        direct_out=direct_out,
        stage_a_scales=stage_a_scales,
        sched_dsrd=sched_dsrd,
    )
    reduce_exe = (
        None
        if direct_out
        else compile_preshuffle_gemm_splitk_reduce(
            N=n, split_k=split_k, out_dtype=out_dtype
        )
    )

    def _as_i8(t):
        return t.view(torch.int8) if "float8" in str(t.dtype) else t

    stream = fx.Stream(torch.cuda.current_stream())
    out_contig = Out.contiguous()
    _run_compiled(
        gemm_exe,
        out_contig.view(-1) if direct_out else workspace.view(-1),
        _as_i8(XQ.contiguous()).view(-1),
        _as_i8(WQ.contiguous()).view(-1),
        x_scale.contiguous().view(-1),
        w_scale.contiguous().view(-1),
        unused_tensor_arg(None, torch.empty(0, dtype=Out.dtype, device=Out.device)),
        m,
        n,
        stream,
    )
    if not direct_out:
        _run_compiled(
            reduce_exe,
            out_contig.view(-1),
            workspace.view(-1),
            m,
            ws_shape[1],
            stream,
        )
    if out_contig is not Out:
        Out.copy_(out_contig)
    return Out
