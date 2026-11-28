# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host side of the MegaMoEGfx942 kernels: one compiled launcher per (builder, device,
compile-time key), launched on the current stream through ``tensor_shim._run_compiled``.

Every launcher takes raw i64 addresses (``fx.Int64``), i32 / f32 scalars and the
stream, so its compiled form depends on the device and the compile-time key only."""

import functools

import torch

from ..tensor_shim import _run_compiled, current_stream_arg


@functools.cache
def launcher(builder, device, **key):
    """``builder(**key)``, one per device: a compiled launcher runs on one device."""
    return builder(**key)


def launch(fn, *args):
    """Launch ``fn`` on the current stream; a tensor goes in as its address, None as 0."""
    raw = [
        a.data_ptr() if isinstance(a, torch.Tensor) else 0 if a is None else a
        for a in args
    ]
    _run_compiled(fn, *raw, current_stream_arg())


def symmetric(shape, dtype):
    """A zeroed tensor in the MORI symmetric heap (collective: every rank allocates the
    same tensors in the same order)."""
    from mori.shmem import mori_shmem_create_tensor

    t = mori_shmem_create_tensor(shape, dtype)
    t.zero_()
    return t


def peer_table(t, rank, world_size):
    """int64 [world_size]: every rank's address of the symmetric tensor ``t``."""
    import mori.shmem as ms

    return torch.tensor(
        [ms.shmem_ptr_p2p(t.data_ptr(), rank, p) for p in range(world_size)],
        dtype=torch.int64,
        device=t.device,
    )
