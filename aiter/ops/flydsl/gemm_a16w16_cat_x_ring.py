# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host op: ``out = concat(xs, dim=-1) @ w^T`` for five equal-width activation taps and a small M (1..128) on gfx950,
without materializing the concat (the DSpark draft ``fc``: five taps of 6144, N=1536, K=30720).

    w_ring = shuffle_weight_ring(w)                       # once, before CUDA-graph capture
    if cat_x_ring_supported(m, n, seg_k, device):
        out = gemm_a16w16_cat_x_ring(xs, w_ring)

K is split ``ks`` ways (default 10: two splits per tap), each split's fp32 partial is stored, and the partials are
summed in ascending split order by aiter's Triton split-K reduce kernel with one bf16 rounding, so the result is
deterministic. It is not bit-identical to a single-pass K order (or to hipBLASLt), only to bf16 rounding.
"""

import torch
import triton

from aiter.ops.triton._triton_kernels.common.splitk_reduce import (
    _gemm_splitk_reduce_kernel,
)

from .gemm_a16w16_ring import MAX_ROWS, _device_ok
from .kernels.gemm_a16w16_cat_x_ring_gfx950 import NSEG, compile_cat_ring
from .kernels.tensor_shim import _run_compiled

NWAVE_FRAGS = 4  # 16-row weight fragments per workgroup column tile (one per wave)


def _ring_depth(stages: int) -> int:
    return 6 if stages >= 6 else 4


def cat_x_ring_supported(m: int, n: int, seg_k: int, device=None, ks: int = 10) -> bool:
    """Whether ``gemm_a16w16_cat_x_ring`` takes M rows of five ``seg_k``-wide taps and an ``[n, 5 * seg_k]`` weight."""
    if not 1 <= m <= MAX_ROWS or not torch.cuda.is_available():
        return False
    index = (
        torch.cuda.current_device() if device is None else torch.device(device).index
    )
    if index is None:
        index = torch.cuda.current_device()
    stages = NSEG * seg_k // 64
    return (
        _device_ok(index)
        and n % (16 * NWAVE_FRAGS) == 0
        and seg_k % 64 == 0
        and ks % NSEG == 0
        and stages % ks == 0
        and stages // ks >= 4
    )


def gemm_a16w16_cat_x_ring(
    xs: list[torch.Tensor],
    w: torch.Tensor,
    out: torch.Tensor | None = None,
    ks: int = 10,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """``out [M, N] bf16 = concat(xs, -1) @ w^T``; ``xs`` are five bf16 ``[M, seg_k]`` taps (unit column stride, one
    common 16-byte-aligned row stride) and ``w`` is ``shuffle_weight_ring(weight [N, 5 * seg_k])``. ``workspace`` is an
    optional fp32 ``[ks, M, N]`` buffer."""
    if len(xs) != NSEG:
        raise ValueError(f"expected {NSEG} taps, got {len(xs)}")
    m, seg_k = xs[0].shape
    n = w.shape[0]
    if (
        not cat_x_ring_supported(m, n, seg_k, xs[0].device, ks)
        or w.shape[1] != NSEG * seg_k
    ):
        raise ValueError(
            f"cat-x ring GEMM does not take M={m} N={n} seg_k={seg_k} ks={ks} on this device"
        )
    stride = xs[0].stride(0)
    for x in xs:
        if (
            x.dtype != torch.bfloat16
            or x.shape != (m, seg_k)
            or x.stride(1) != 1
            or x.stride(0) != stride
            or (stride * 2) % 16
            or x.data_ptr() % 16
        ):
            raise ValueError(
                "taps must be bf16 [M, seg_k] with unit column stride, one aligned row stride"
            )
    if w.dtype != torch.bfloat16 or not w.is_contiguous() or w.data_ptr() % 16:
        raise ValueError(
            "w must be a contiguous aligned bf16 shuffle_weight_ring layout"
        )
    if out is None:
        out = torch.empty(m, n, dtype=torch.bfloat16, device=w.device)
    elif out.shape != (m, n) or out.dtype != torch.bfloat16 or out.stride(1) != 1:
        raise ValueError("out must be a bf16 [M, N] with unit column stride")
    if workspace is None:
        workspace = torch.empty(ks, m, n, dtype=torch.float32, device=w.device)
    elif (
        workspace.shape != (ks, m, n)
        or workspace.dtype != torch.float32
        or not workspace.is_contiguous()
    ):
        raise ValueError("workspace must be a contiguous fp32 [ks, M, N]")
    exe = compile_cat_ring(
        n, seg_k, (m + 15) // 16, ks, _ring_depth(NSEG * seg_k // 64 // ks)
    )
    _run_compiled(
        exe,
        *[x.data_ptr() for x in xs],
        w.data_ptr(),
        workspace.data_ptr(),
        m,
        stride * 2,
        torch.cuda.current_stream(),
    )
    _gemm_splitk_reduce_kernel[(triton.cdiv(m, 32), triton.cdiv(n, 32))](
        workspace,
        out,
        None,
        m,
        n,
        workspace.stride(0),
        workspace.stride(1),
        workspace.stride(2),
        out.stride(0),
        out.stride(1),
        32,
        32,
        ks,
        triton.next_power_of_2(ks),
        ADD_BIAS=False,
        activation=None,
        use_activation=False,
    )
    return out


def warmup_cat_x_ring(
    m_max: int, n: int, seg_k: int, w: torch.Tensor, ks: int = 10
) -> None:
    """Compile every row tier (call before CUDA-graph capture)."""
    x = [
        torch.zeros(m_max, seg_k, dtype=torch.bfloat16, device=w.device)
        for _ in range(NSEG)
    ]
    for m in range(16, m_max + 1, 16):
        gemm_a16w16_cat_x_ring([t[:m] for t in x], w, ks=ks)
    torch.cuda.synchronize(w.device)
