# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 (WMMA) backend for the FlyDSL a8w8 bpreshuffle GEMM.

aiter.gemm_a8w8_bpreshuffle routes here when its tuned kernelName starts with
``flydsl_bpreshuffle_wmma_`` (gfx1250 has no MFMA preshuffle kernel). Runs the
vendored gemm_fp8fp4_gfx1250 WMMA kernel in ptpc scale mode: C = (A*sa) @ (B*sb)^T
with fp32 per-token sa[M] / per-channel sb[N] applied in the epilogue. N/K are not
padded (must divide the tile); M is padded to tile_m when ragged, since the kernel
reads a full tile_m rows per workgroup (M=1 would otherwise read past A/sa -> NaN).
"""

from __future__ import annotations

import re

import torch
from torch import Tensor

# Lazily bound flydsl symbols (kept out of import path when flydsl is absent).
_compile_ptpc_gemm = None
_run_compiled = None
_fx = None

_WMMA_K = 128
_SUPPORTED_NUM_BUFFERS = (2, 3, 4)
_OUT_DTYPE_NAME = {torch.bfloat16: "bf16", torch.float16: "f16"}


def _lazy_import():
    global _compile_ptpc_gemm, _run_compiled, _fx
    if _compile_ptpc_gemm is not None:
        return
    import flydsl.expr as fx_mod

    from .kernels.gemm_fp8fp4_gfx1250 import compile_ptpc_gemm
    from .kernels.tensor_shim import _run_compiled as runner

    _compile_ptpc_gemm = compile_ptpc_gemm
    _run_compiled = runner
    _fx = fx_mod


def _as_1d_fp32(scale: Tensor, length: int, name: str) -> Tensor:
    flat = scale.reshape(-1)
    if flat.numel() != length:
        raise RuntimeError(
            f"[FlyDSL gfx1250] {name} must have {length} elements, "
            f"got {tuple(scale.shape)}"
        )
    if flat.dtype != torch.float32:
        flat = flat.to(torch.float32)
    return flat.contiguous()


def _to_uint8(t: Tensor) -> Tensor:
    return t.contiguous().view(torch.uint8).view(-1)


def run_preshuffle_gemm_a8_gfx1250(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    *,
    num_buffers: int = 2,
    waves_per_eu: int = 0,
    m_warp: int = 2,
    n_warp: int = 2,
    cluster_m: int = 1,
    cluster_n: int = 1,
    split_k: int = 1,
) -> Tensor:
    """Run the gfx1250 WMMA a8w8 bpreshuffle (ptpc) GEMM; writes into ``Out``.

    XQ: (M, K) FP8 E4M3. WQ: (N, K) FP8 E4M3, 16x16 ``shuffle_weight``.
    x_scale: per-token fp32 ``(M,)`` or ``(M, 1)``. w_scale: per-channel fp32
    ``(N,)`` or ``(N, 1)``. ``Out``: (M, N) bf16/f16.
    """
    _lazy_import()

    if XQ.dim() != 2 or WQ.dim() != 2:
        raise RuntimeError(
            f"[FlyDSL gfx1250] A/B must be 2-D, got {tuple(XQ.shape)}, {tuple(WQ.shape)}"
        )
    if XQ.element_size() != 1 or WQ.element_size() != 1:
        raise RuntimeError("[FlyDSL gfx1250] A/B must be 1-byte fp8 storage")

    M, K = XQ.shape
    N = WQ.shape[0]
    if K != WQ.shape[1]:
        raise RuntimeError(f"[FlyDSL gfx1250] K mismatch: A.K={K} vs B.K={WQ.shape[1]}")
    if N % tile_n != 0:
        raise RuntimeError(f"[FlyDSL gfx1250] N={N} not a multiple of tile_n={tile_n}")
    if K % _WMMA_K != 0 or K % tile_k != 0:
        raise RuntimeError(
            f"[FlyDSL gfx1250] K={K} must be a multiple of WMMA_K={_WMMA_K} and "
            f"tile_k={tile_k}"
        )

    out_dtype = _OUT_DTYPE_NAME.get(Out.dtype)
    if out_dtype is None:
        raise RuntimeError(
            f"[FlyDSL gfx1250] unsupported out dtype {Out.dtype}; expected bf16/fp16"
        )

    split_k = max(1, int(split_k))
    cluster_m = max(1, int(cluster_m))
    cluster_n = max(1, int(cluster_n))

    accumulate_fp32 = split_k > 1
    kernel_out_dtype = "f32" if accumulate_fp32 else out_dtype

    # Pipeline depth needs >= 1 K tile per buffer (per split-k chunk).
    num_k_tiles = (K // split_k) // tile_k
    nb = max(2, min(int(num_buffers), num_k_tiles))
    if nb not in _SUPPORTED_NUM_BUFFERS:
        nb = max(b for b in _SUPPORTED_NUM_BUFFERS if b <= nb)

    sa = _as_1d_fp32(x_scale, M, "x_scale")
    sb = _as_1d_fp32(w_scale, N, "w_scale")

    # M padded to tile_m when ragged (kernel reads a full tile_m rows/wg).
    padded_m = ((M + tile_m - 1) // tile_m) * tile_m
    if padded_m == M:
        a_dev = XQ.contiguous()
        sa_dev = sa
    else:
        a_dev = torch.zeros((padded_m, K), dtype=XQ.dtype, device=XQ.device)
        a_dev[:M] = XQ
        sa_dev = torch.ones((padded_m,), dtype=torch.float32, device=sa.device)
        sa_dev[:M] = sa

    b_dev = WQ.contiguous()

    exe = _compile_ptpc_gemm(
        M=padded_m,
        N=N,
        K=K,
        data_format="fp8",
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=m_warp,
        n_warp=n_warp,
        num_buffers=nb,
        waves_per_eu=(None if waves_per_eu <= 0 else waves_per_eu),
        cluster_m=cluster_m,
        cluster_n=cluster_n,
        out_dtype=kernel_out_dtype,
        split_k=split_k,
    )

    if accumulate_fp32:
        # fp32 atomic-accumulation scratch (zeroed; narrowed into Out below).
        out_buf = torch.zeros((padded_m, N), dtype=torch.float32, device=Out.device)
    elif padded_m == M:
        out_buf = Out.contiguous()
    else:
        out_buf = torch.empty((padded_m, N), dtype=Out.dtype, device=Out.device)

    stream = _fx.Stream(torch.cuda.current_stream(device=a_dev.device))
    _run_compiled(
        exe,
        out_buf.view(-1),
        _to_uint8(a_dev),
        _to_uint8(b_dev),
        sa_dev.contiguous().view(-1),
        sb.contiguous().view(-1),
        padded_m,
        N,
        stream,
    )

    if out_buf.data_ptr() != Out.data_ptr():
        Out.copy_(out_buf[:M])
    return Out


# flydsl_bpreshuffle_wmma_t{tm}x{tn}x{tk}_nb{nb}_sk{sk}_cm{cm}_cn{cn}
_KERNEL_NAME_RE = re.compile(
    r"^flydsl_bpreshuffle_wmma_"
    r"t(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+)_"
    r"nb(?P<num_buffers>\d+)_sk(?P<split_k>\d+)_"
    r"cm(?P<cluster_m>\d+)_cn(?P<cluster_n>\d+)$"
)


def wmma_kernel_name(
    *,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    num_buffers: int,
    split_k: int,
    cluster_m: int,
    cluster_n: int,
) -> str:
    return (
        f"flydsl_bpreshuffle_wmma_t{tile_m}x{tile_n}x{tile_k}_"
        f"nb{num_buffers}_sk{split_k}_cm{cluster_m}_cn{cluster_n}"
    )


def parse_wmma_kernel_name(name: str):
    """Parse a flydsl_bpreshuffle_wmma_ kernelName into its config dict, or None."""
    m = _KERNEL_NAME_RE.fullmatch(name)
    return {k: int(v) for k, v in m.groupdict().items()} if m else None


def run_gemm_a8w8_bpreshuffle_gfx1250(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    kernel_name: str,
) -> Tensor:
    """Dispatch entry: decode a tuned wmma kernelName and run the kernel."""
    cfg = parse_wmma_kernel_name(kernel_name)
    if cfg is None:
        raise ValueError(f"[FlyDSL gfx1250] unrecognised kernelName: {kernel_name!r}")
    return run_preshuffle_gemm_a8_gfx1250(
        XQ,
        WQ,
        x_scale,
        w_scale,
        Out,
        cfg["tile_m"],
        cfg["tile_n"],
        cfg["tile_k"],
        num_buffers=cfg["num_buffers"],
        split_k=cfg["split_k"],
        cluster_m=cfg["cluster_m"],
        cluster_n=cfg["cluster_n"],
    )
