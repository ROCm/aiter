# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from aiter.ops.mesh_lend_all_reduce import (
    mesh_lend_all_reduce,
    mesh_lend_connect,
    mesh_lend_create,
    mesh_lend_destroy,
    mesh_lend_export,
    mesh_lend_export_size,
    mesh_lend_num_lenders,
)

MB = 1024 * 1024


class MeshLendAllReduce:
    """bfloat16 all-reduce for a TP group of 2 or 4 GPUs on an xGMI full mesh.

    The payload moves on SDMA engines only: over the group's own links and
    through the GPUs outside the group (lenders), whose links to the group
    are otherwise idle. Nothing runs and nothing is stored on a lender.

    Two ranks reduce in one shot, exactly (``bits=0``) or with each input
    quantized once to 4 or 8 bits per value. Four ranks reduce exactly:
    reduce-scatter, then all-gather.

    One instance per rank process. Calls must come in the same order on every
    rank, on one stream, and outside CUDA graph capture: each call
    synchronizes the caller's stream and posts packets from the host.

    Needs bare metal and KFD ioctl 1.17. The lenders are the GPUs outside the
    group that ROCr sees in every rank (``ROCR_VISIBLE_DEVICES``) and that
    have an xGMI link to every rank; they need not be visible to HIP.

    Args:
        group: CPU process group of the ranks, used once to exchange handles.
        device: this rank's GPU.
        max_bytes: largest payload. A two-rank instance holds 2.5 times this
            in device memory, a four-rank one 2.25 times.
        max_lenders: use at most this many lenders; 0 keeps the payload on
            the group's own links.

    Raises:
        RuntimeError: if any rank cannot set the transport up. Every rank
            raises, so the group can fall back together.
    """

    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        *,
        max_bytes: int = 256 * MB,
        max_lenders: int = 7,
    ) -> None:
        self._ptr = 0
        device = torch.device(f"cuda:{device}" if isinstance(device, int) else device)
        if device.type != "cuda" or device.index is None:
            raise ValueError(f"MeshLendAllReduce needs an indexed GPU, got {device}")
        self.device = device
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.max_bytes = max_bytes

        # Host memory: the library copies its export into data_ptr().
        export, error = None, None
        try:
            self._ptr = mesh_lend_create(
                device.index, self.rank, self.world_size, max_bytes, max_lenders
            )
            export = torch.empty(mesh_lend_export_size(), dtype=torch.uint8)
            mesh_lend_export(self._ptr, export.data_ptr())
        except (RuntimeError, ValueError) as e:
            export, error = None, f"rank {self.rank}: {e}"
        exports = self._gather(group, export)
        if error is None and all(e is not None for e in exports):
            try:
                mesh_lend_connect(self._ptr, [e.data_ptr() for e in exports])
            except RuntimeError as e:
                error = f"rank {self.rank}: {e}"
        errors = [e for e in self._gather(group, error) if e]
        if errors:
            self.close()
            raise RuntimeError(
                "mesh-lending all-reduce unavailable: " + "; ".join(errors)
            )
        self.num_lenders = mesh_lend_num_lenders(self._ptr)

    def _gather(self, group: ProcessGroup, mine):
        out = [None] * self.world_size
        dist.all_gather_object(out, mine, group=group)
        return out

    def supports(self, inp: torch.Tensor, bits: int = 0) -> bool:
        """Whether ``all_reduce`` takes this tensor. Depends only on what all
        ranks share (dtype, layout, size), so they decide alike."""
        return (
            inp.dtype == torch.bfloat16
            and inp.is_contiguous()
            and 0 < inp.numel() * 2 <= self.max_bytes
            and (bits == 0 or (bits in (4, 8) and self.world_size == 2))
        )

    def all_reduce(
        self, inp: torch.Tensor, out: torch.Tensor | None = None, *, bits: int = 0
    ) -> torch.Tensor:
        """Sum of ``inp`` over the ranks, complete when the current stream is.

        ``bits=0`` is exact: the float32 sum in rank order, rounded once.
        ``bits=4`` or ``8`` (two ranks) adds the two dequantized inputs; both
        ranks get the same bits. ``out`` may be ``inp``.
        """
        if out is None:
            out = torch.empty_like(inp)
        mesh_lend_all_reduce(self._ptr, inp, out, bits)
        return out

    def close(self) -> None:
        if self._ptr:
            mesh_lend_destroy(self._ptr)
            self._ptr = 0

    def __del__(self) -> None:
        self.close()
