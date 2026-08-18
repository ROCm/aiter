# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import TransportKind


@dataclass
class TransportStats:
    logical_puts: int = 0
    bytes: int = 0
    signals: int = 0


class CopyTransport:
    """Synchronous local-copy implementation of the future SHMEM contract."""

    kind = TransportKind.COPY_STUB

    def __init__(self) -> None:
        self.stats = TransportStats()

    @torch.no_grad()
    def put_signal(
        self,
        destination: torch.Tensor,
        source: torch.Tensor,
        nbytes: int,
        signal: torch.Tensor,
        signal_value: int,
    ) -> None:
        if nbytes < 0:
            raise ValueError("nbytes must be non-negative")
        src = source.view(torch.uint8).flatten()
        dst = destination.view(torch.uint8).flatten()
        if nbytes > src.numel() or nbytes > dst.numel():
            raise ValueError("copy exceeds source/destination capacity")
        dst[:nbytes].copy_(src[:nbytes], non_blocking=False)
        signal.view(-1)[0] = int(signal_value)
        self.stats.logical_puts += 1
        self.stats.bytes += int(nbytes)
        self.stats.signals += 1


class MoriShmemTransport:
    """Host launcher for the real FlyDSL MORI SHMEM transport ABI."""

    kind = TransportKind.MORI_SHMEM

    def __init__(self) -> None:
        from .kernels import (
            build_mori_eos_module,
            build_mori_put_signal_module,
            build_mori_quiet_module,
        )

        self._put = build_mori_put_signal_module()
        self._quiet = build_mori_quiet_module()
        self._eos = build_mori_eos_module()
        self.stats = TransportStats()

    @staticmethod
    def _require_symmetric(*tensors: torch.Tensor) -> None:
        for tensor in tensors:
            if not getattr(tensor, "__symm_tensor__", False):
                raise ValueError("MORI address transport requires symmetric heap tensors")

    def post_chunk(
        self,
        destination_same_offset: torch.Tensor,
        source: torch.Tensor,
        nbytes: int,
        signal_same_offset: torch.Tensor,
        generation: int,
        remote_pe: int,
        qp_id: int,
        *,
        stream=None,
    ) -> None:
        self._require_symmetric(destination_same_offset, source, signal_same_offset)
        if nbytes <= 0:
            raise ValueError("data put must be non-empty; use publish_eos for empty peers")
        stream = torch.cuda.current_stream() if stream is None else stream
        self._put(
            destination_same_offset.data_ptr(),
            source.data_ptr(),
            int(nbytes),
            signal_same_offset.data_ptr(),
            int(generation),
            int(remote_pe),
            int(qp_id),
            stream=stream,
        )
        self.stats.logical_puts += 1
        self.stats.bytes += int(nbytes)
        self.stats.signals += 1

    def publish_eos(
        self,
        eos_same_offset: torch.Tensor,
        value: int,
        remote_pe: int,
        qp_id: int,
        *,
        stream=None,
    ) -> None:
        self._require_symmetric(eos_same_offset)
        stream = torch.cuda.current_stream() if stream is None else stream
        self._eos(
            eos_same_offset.data_ptr(),
            int(value),
            int(remote_pe),
            int(qp_id),
            stream=stream,
        )
        self.stats.signals += 1

    def drain(self, remote_pe: int, qp_id: int, *, stream=None) -> None:
        stream = torch.cuda.current_stream() if stream is None else stream
        self._quiet(int(remote_pe), int(qp_id), stream=stream)
