# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The IPC buffers and launch colours of the gfx950 Gluon multi-GPU ops."""

import ctypes

import torch
import torch.distributed as dist
import triton

from aiter.ops.triton.utils._triton.arch_info import get_arch

ERROR_BYTES = 256  # the error record [flag, rank, colour, site], then padding
SUPPORTED_WORLDS = (2, 4, 8)
_ALIGN = 2 << 20


class IpcCommBuffers:
    """
    The base class of the comm objects of the gfx950 Gluon multi-GPU ops, for one group
    of 2, 4 or 8 ranks.

    Each rank has one uncached IPC buffer. The buffer starts with the error record
    (ERROR_BYTES). The subclass sets the layout of the rest. ``colors`` holds one launch
    colour for each CTA. The constructor, ``reset`` and ``close`` are collective: all
    ranks of the group must call them.

    A subclass calls ``__init__``, then sets its layout, then calls ``_open``.

    Args:
        group: The CPU (gloo) process group of the ranks.
        timeout_s (float): The time limit of each wait in the kernel.
    """

    def __init__(self, group, timeout_s):
        assert get_arch() == "gfx950", "needs gfx950"
        self.world = dist.get_world_size(group)
        assert (
            self.world in SUPPORTED_WORLDS
        ), f"the group size must be one of {SUPPORTED_WORLDS}"
        self.timeout_us = max(1, int(timeout_s * 1e6))
        assert self.timeout_us < 2**31, "timeout_s is too large"
        self.group = group
        self.rank = dist.get_rank(group)
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.num_cus = torch.cuda.get_device_properties(
            self.device
        ).multi_processor_count
        self._pool = None

    def _open(self, nbytes, blocks, fill=0xFF, fill_bytes=None):
        """
        Allocates the buffer of this rank (nbytes, rounded up to 2 MiB), maps the
        buffers of the peers and makes the colours of blocks CTAs. Then it clears them:
        see ``_clear``.

        Args:
            nbytes (int): The bytes of the layout, from the error record to the end.
            blocks (int): The CTAs of a launch.
            fill (int): The byte that a clear writes into the buffer.
            fill_bytes (int, optional): The bytes that a clear fills. Default: all.
        """
        from aiter.dist.device_communicators.custom_all_reduce import IPCBufferPool

        self.nbytes = triton.cdiv(nbytes, _ALIGN) * _ALIGN
        self._fill = fill
        self._fill_bytes = self.nbytes if fill_bytes is None else fill_bytes
        assert ERROR_BYTES <= self._fill_bytes <= self.nbytes
        self._pool = IPCBufferPool(self.device, self.group)
        self._pool.create("inbox", self.nbytes, uncached=True)
        addrs = self._pool.open_peer_buffers("inbox")
        # The buffers of ranks rank, rank + 1, ..., rank + world - 1.
        self.peers = tuple(
            addrs[(self.rank + d) % self.world] for d in range(self.world)
        )
        self.colors = torch.empty(blocks, dtype=torch.int32, device=self.device)
        self._clear()
        dist.barrier(group=self.group)

    def _clear(self):
        """Fills the first fill_bytes of the buffer of this rank with fill, then clears
        the error record and the colours. A subclass with more state clears it too."""
        from aiter.dist.device_communicators.custom_all_reduce import (
            _loaded_hip_runtime,
        )

        rt = _loaded_hip_runtime()
        torch.cuda.synchronize(self.device)
        own = ctypes.c_void_p(self.peers[0])
        err = rt.hipMemset(own, self._fill, ctypes.c_size_t(self._fill_bytes))
        err = err or rt.hipMemset(own, 0, ctypes.c_size_t(ERROR_BYTES))
        assert err == 0, f"hipMemset failed: {err}"
        self.colors.zero_()
        torch.cuda.synchronize(self.device)

    def error(self):
        """
        Returns the error record of this rank: [0, 0, 0, 0], or [1, rank, colour,
        site] after a wait timed out. The op lists its sites. Synchronizes the device.
        """
        from aiter.dist.device_communicators.custom_all_reduce import (
            _loaded_hip_runtime,
        )

        record = (ctypes.c_int32 * 4)()
        torch.cuda.synchronize(self.device)
        err = _loaded_hip_runtime().hipMemcpy(
            record, ctypes.c_void_p(self.peers[0]), ctypes.c_size_t(16), 2
        )
        assert err == 0, f"hipMemcpy failed: {err}"
        return list(record)

    def reset(self):
        """Clears the buffers and the colours, for example after a wait timed out. All
        ranks must call it with no launch in flight."""
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        self._clear()
        dist.barrier(group=self.group)

    def close(self):
        """Unmaps the peer buffers and frees the buffer of this rank. All ranks must
        call it."""
        if self._pool is None:
            return
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        self._pool.close_peer_buffers()
        dist.barrier(group=self.group)  # no peer uses the buffer of this rank now
        self._pool.close()
        self._pool = None
