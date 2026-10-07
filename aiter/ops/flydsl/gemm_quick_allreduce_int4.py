# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Serial FlyDSL HGEMM followed by FlyDSL INT4 QuickReduce.

One implementation, two kernels, one stream:

    partial[M, N] = A[M, K] @ B[N, K].T
    out = quick_allreduce_int4(partial)

``partial`` is bf16. :meth:`GemmQuickAllReduceInt4.handoff` is the seam
between the GEMM store and the reduce load. The next iteration should
return the INT4 packets QuickReduce already builds, so the reduce does
not reload this bf16 tensor.

This is the row-parallel projection at TP8. ``N`` is the full hidden
size. ``K`` is the per-rank shard:

* KDA and MLA ``o_proj``: ``M x 1536 x 7168`` (93 launches per 16k chunk)
* dense ``down_proj``: ``M x 4224 x 7168`` (1 launch per chunk)

The GEMM and the reduce are the existing kernels. This module does not
retune either one. The default tile is the gfx950 256x256x64 half-tile
used for those shapes.
"""

from __future__ import annotations

import torch

from .gemm_kernels import flydsl_hgemm
from .quick_allreduce_int4 import QuickAllReduceInt4

# gfx950 half-tile kernel: t256x256x64, stages 2, waves 2x4x1, policy ht.
DEFAULT_HGEMM = {
    "block_m": 256,
    "block_n": 256,
    "block_k": 64,
    "stages": 2,
    "split_k": 1,
    "m_waves": 2,
    "n_waves": 4,
    "k_waves": 1,
    "group_m": 0,
    "policy": "ht",
}


def _overlaps(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.device != b.device or a.numel() == 0 or b.numel() == 0:
        return False
    a0 = int(a.data_ptr())
    b0 = int(b.data_ptr())
    a1 = a0 + int(a.numel()) * int(a.element_size())
    b1 = b0 + int(b.numel()) * int(b.element_size())
    return max(a0, b0) < min(a1, b1)


class GemmQuickAllReduceInt4:
    """Local bf16 HGEMM, then the existing INT4 two-shot all-reduce.

    ``B`` uses AITER's ``[N, K]`` layout, the same one ``flydsl_hgemm``
    takes. Each rank multiplies its own ``K`` shard. The reduce sums the
    partials across the QuickReduce group.
    """

    def __init__(self, qr: QuickAllReduceInt4, *, owns_qr: bool = False):
        if not isinstance(qr, QuickAllReduceInt4):
            raise TypeError(f"qr must be QuickAllReduceInt4, got {type(qr)!r}")
        self.qr = qr
        self._owns_qr = owns_qr
        self._partial: torch.Tensor | None = None

    @classmethod
    def from_group(cls, group, device, rank: int, world_size: int = 8, **qr_kwargs):
        qr = QuickAllReduceInt4(
            group=group,
            device=device,
            rank=rank,
            world_size=world_size,
            **qr_kwargs,
        )
        return cls(qr, owns_qr=True)

    def close(self) -> None:
        self._partial = None
        if self._owns_qr:
            self.qr.close()
            self._owns_qr = False

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            return

    def partial_buffer(self, out: torch.Tensor) -> torch.Tensor:
        """bf16 workspace the GEMM writes and the reduce reads."""
        cached = self._partial
        if (
            cached is None
            or cached.shape != out.shape
            or cached.dtype != out.dtype
            or cached.device != out.device
        ):
            self._partial = torch.empty(out.shape, dtype=out.dtype, device=out.device)
        return self._partial

    def handoff(self, partial: torch.Tensor) -> torch.Tensor:
        """Tensor the reduce consumes.

        Identity today. The next iteration returns INT4 packets instead
        of this bf16 partial.
        """
        return partial

    def gemm(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        partial: torch.Tensor,
        *,
        gemm: dict | None = None,
        stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        cfg = dict(DEFAULT_HGEMM)
        if gemm:
            cfg.update(gemm)
        return flydsl_hgemm(a, b, out=partial, stream=stream, **cfg)

    def reduce(
        self,
        partial: torch.Tensor,
        out: torch.Tensor,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        self.qr.allreduce(self.handoff(partial), out, stream=stream)

    def _check(self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
        if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
            raise ValueError("GemmQuickAllReduceInt4 requires bf16 A and B")
        if out.dtype != torch.bfloat16:
            raise ValueError("GemmQuickAllReduceInt4 requires a bf16 output")
        if a.ndim != 2 or b.ndim != 2 or out.ndim != 2:
            raise ValueError("A, B, and out must be rank-2")
        m, k = a.shape
        n, bk = b.shape
        if bk != k:
            raise ValueError(f"B is [N, K]={tuple(b.shape)}, A K is {k}")
        if tuple(out.shape) != (m, n):
            raise ValueError(f"out shape {tuple(out.shape)} != {(m, n)}")
        if not a.is_contiguous() or not b.is_contiguous() or not out.is_contiguous():
            raise ValueError("A, B, and out must be contiguous")
        if a.device != b.device or a.device != out.device:
            raise ValueError("A, B, and out must be on the same device")

    def apply(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        out: torch.Tensor,
        *,
        gemm: dict | None = None,
        stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        """``out = allreduce(A @ B)`` on ``stream``.

        The GEMM finishes before the reduce starts. Both use ``stream``,
        or the current stream when ``stream`` is omitted.
        """
        self._check(a, b, out)
        if stream is None:
            stream = torch.cuda.current_stream(device=a.device)
        partial = self.partial_buffer(out)
        if _overlaps(partial, out) or _overlaps(partial, a) or _overlaps(partial, b):
            raise ValueError("GEMM partial overlaps an input or the output")
        self.gemm(a, b, partial, gemm=gemm, stream=stream)
        self.reduce(partial, out, stream=stream)
        return out

    def compile(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        out: torch.Tensor,
        *,
        gemm: dict | None = None,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """JIT the GEMM and every QuickReduce super-tile binary."""
        self._check(a, b, out)
        partial = self.partial_buffer(out)
        self.qr.compile(partial, out, stream=stream)
        self.apply(a, b, out, gemm=gemm, stream=stream)
