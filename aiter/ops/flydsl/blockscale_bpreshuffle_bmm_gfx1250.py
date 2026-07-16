# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 (WMMA) backend for the FlyDSL a8w8 blockscale bpreshuffle *batched* GEMM.

Wraps ``kernels.a8w8_bmm_bpreshuffle_gfx1250.launch_gemm_a8w8_tdm`` (strided-batched A8W8
blockwise-scaled preshuffle GEMM) with a tuned-``kernelName`` dispatcher, mirroring
``bpreshuffle_gemm_gfx1250.py`` for the non-batched case.

The tuned kernelName encodes the whole tile config::

    flydsl_blockscale_bpreshuffle_bmm_t{tm}x{tn}x{tk}_mw{mw}_nw{nw}_nb{nb}_sk{sk}_cm{cm}_cn{cn}

Data contract (caller pre-processes, matching the tuned-kernel + tune script
convention -- see FlyDSL/tests/kernels/test_a8w8_bmm.py):

  A          FP8 E4M3 bytes, layout ``[B, M, K]`` (BMN) or ``[M, B, K]`` (MBN).
  B          FP8 E4M3 bytes, already ``preshuffle_b_16x16``-ed, layout ``[B, N, K]``.
  scale_a    E8M0 uint8, 1x128 granularity, ``[B, M, K//128]`` (BMN) or
             ``[M, B, K//128]`` (MBN).
  scale_b    E8M0 uint8, 128x128 granularity, ``[B, N//128, K//128]``.
  Out        bf16/f16, layout ``[B, M, N]`` (BMN) or ``[M, B, N]`` (MBN).

Constraints (asserted): N % tile_n == 0, K % tile_k == 0, tile_k/tile_n multiple
of 128, tile_m multiple of m_warp*16, K // tile_k >= num_buffers. M may be ragged
(hardware OOB clipping). ``split_k`` / ``cluster_m`` / ``cluster_n`` are parsed
from the kernelName for forward-compat but the current kernel only supports 1/1/1.
"""

from __future__ import annotations

import re

import torch
from torch import Tensor

# Lazily bound flydsl symbols (kept out of import path when flydsl is absent).
_launch_gemm_a8w8_tdm = None
_ptr_arg = None
_run_compiled = None

_WMMA_K = 128
_SCALE_BLOCK = 128
_OUT_IS_F16 = {torch.bfloat16: 0, torch.float16: 1}


def _lazy_import():
    global _launch_gemm_a8w8_tdm, _run_compiled, _ptr_arg
    if _launch_gemm_a8w8_tdm is not None:
        return
    from .kernels.a8w8_bmm_bpreshuffle_gfx1250 import launch_gemm_a8w8_tdm
    from .kernels.tensor_shim import _run_compiled as runner, ptr_arg

    _launch_gemm_a8w8_tdm = launch_gemm_a8w8_tdm
    _run_compiled = runner
    _ptr_arg = ptr_arg


# flydsl_blockscale_bpreshuffle_bmm_t{tm}x{tn}x{tk}_mw{mw}_nw{nw}_nb{nb}_sk{sk}_cm{cm}_cn{cn}
_KERNEL_NAME_RE = re.compile(
    r"^flydsl_blockscale_bpreshuffle_bmm_"
    r"t(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+)_"
    r"mw(?P<m_warp>\d+)_nw(?P<n_warp>\d+)_"
    r"nb(?P<num_buffers>\d+)_sk(?P<split_k>\d+)_"
    r"cm(?P<cluster_m>\d+)_cn(?P<cluster_n>\d+)$"
)


def bmm_kernel_name(
    *,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    m_warp: int,
    n_warp: int,
    num_buffers: int,
    split_k: int = 1,
    cluster_m: int = 1,
    cluster_n: int = 1,
) -> str:
    return (
        f"flydsl_blockscale_bpreshuffle_bmm_t{tile_m}x{tile_n}x{tile_k}_"
        f"mw{m_warp}_nw{n_warp}_nb{num_buffers}_sk{split_k}_"
        f"cm{cluster_m}_cn{cluster_n}"
    )


def parse_bmm_kernel_name(name: str):
    """Parse a flydsl_blockscale_bpreshuffle_bmm_ kernelName into a config dict, or None."""
    m = _KERNEL_NAME_RE.fullmatch(name)
    return {k: int(v) for k, v in m.groupdict().items()} if m else None


def run_blockscale_bpreshuffle_bmm_gfx1250(
    A: Tensor,
    B: Tensor,
    scale_a: Tensor,
    scale_b: Tensor,
    Out: Tensor,
    *,
    M: int,
    N: int,
    K: int,
    batch: int,
    layout_mbn: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    m_warp: int,
    n_warp: int,
    num_buffers: int,
    split_k: int = 1,
    cluster_m: int = 1,
    cluster_n: int = 1,
) -> Tensor:
    """Run the gfx1250 WMMA a8w8 blockscale bpreshuffle batched GEMM; writes into ``Out``.

    See the module docstring for the input data contract. ``Out`` must be a
    pre-allocated, contiguous [B, M, N] (BMN) or [M, B, N] (MBN) bf16/f16 tensor.
    """
    _lazy_import()

    out_is_f16 = _OUT_IS_F16.get(Out.dtype)
    if out_is_f16 is None:
        raise RuntimeError(
            f"[FlyDSL bmm gfx1250] unsupported out dtype {Out.dtype}; expected bf16/fp16"
        )
    if A.element_size() != 1 or B.element_size() != 1:
        raise RuntimeError("[FlyDSL bmm gfx1250] A/B must be 1-byte fp8 storage")

    if tile_k < _WMMA_K or tile_k % _WMMA_K != 0:
        raise RuntimeError(
            f"[FlyDSL bmm gfx1250] tile_k={tile_k} must be a positive multiple of {_WMMA_K}"
        )
    if tile_n < _SCALE_BLOCK or tile_n % _SCALE_BLOCK != 0:
        raise RuntimeError(
            f"[FlyDSL bmm gfx1250] tile_n={tile_n} must be a positive multiple of {_SCALE_BLOCK}"
        )
    if tile_m % (m_warp * 16) != 0:
        raise RuntimeError(
            f"[FlyDSL bmm gfx1250] tile_m={tile_m} must be a multiple of m_warp*16={m_warp * 16}"
        )
    if N % tile_n != 0:
        raise RuntimeError(
            f"[FlyDSL bmm gfx1250] N={N} not a multiple of tile_n={tile_n}"
        )
    if K % tile_k != 0:
        raise RuntimeError(
            f"[FlyDSL bmm gfx1250] K={K} not a multiple of tile_k={tile_k}"
        )
    if K // tile_k < num_buffers:
        raise RuntimeError(
            f"[FlyDSL bmm gfx1250] K//tile_k={K // tile_k} must be >= num_buffers={num_buffers}"
        )
    if split_k != 1 or cluster_m != 1 or cluster_n != 1:
        raise RuntimeError(
            "[FlyDSL bmm gfx1250] kernel only supports split_k=cluster_m=cluster_n=1, "
            f"got sk={split_k} cm={cluster_m} cn={cluster_n}"
        )

    stream = torch.cuda.current_stream(device=A.device)
    # arg_a / arg_b are declared fx.Pointer in the kernel: wrap the torch tensors
    # as PointerJitArg so the cached fast-dispatch path (cf(*args)) binds them as
    # raw device pointers. arg_c / scale_a / scale_b stay fx.Tensor (passed raw).
    _run_compiled(
        _launch_gemm_a8w8_tdm,
        Out,
        _ptr_arg(A),
        _ptr_arg(B),
        scale_a,
        scale_b,
        int(M),
        stream,
        int(N),
        int(K),
        int(tile_m),
        int(tile_n),
        int(tile_k),
        int(m_warp),
        int(n_warp),
        int(out_is_f16),
        int(batch),
        int(layout_mbn),
        int(num_buffers),
    )
    return Out


def run_from_kernel_name(
    A: Tensor,
    B: Tensor,
    scale_a: Tensor,
    scale_b: Tensor,
    Out: Tensor,
    *,
    M: int,
    N: int,
    K: int,
    batch: int,
    layout_mbn: int,
    kernel_name: str,
) -> Tensor:
    """Dispatch entry: decode a tuned bmm kernelName and run the kernel."""
    cfg = parse_bmm_kernel_name(kernel_name)
    if cfg is None:
        raise ValueError(
            f"[FlyDSL bmm gfx1250] unrecognised kernelName: {kernel_name!r}"
        )
    return run_blockscale_bpreshuffle_bmm_gfx1250(
        A,
        B,
        scale_a,
        scale_b,
        Out,
        M=M,
        N=N,
        K=K,
        batch=batch,
        layout_mbn=layout_mbn,
        tile_m=cfg["tile_m"],
        tile_n=cfg["tile_n"],
        tile_k=cfg["tile_k"],
        m_warp=cfg["m_warp"],
        n_warp=cfg["n_warp"],
        num_buffers=cfg["num_buffers"],
        split_k=cfg["split_k"],
        cluster_m=cfg["cluster_m"],
        cluster_n=cfg["cluster_n"],
    )
