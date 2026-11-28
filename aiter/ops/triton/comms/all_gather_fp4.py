# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP4 row gather of packed FP4 codes and scales/routing metadata on gfx1250."""

import ctypes

import torch
import torch.distributed as dist
import triton

from aiter.ops.triton._gluon_kernels.gfx1250.comms.all_gather_fp4 import (
    _gather_kernel,
    _publish_kernel,
    _ready_kernel,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


class Fp4AllGather:
    """Own the IPC buffers of a TP4 FP4 gather.

    Each rank supplies at most max_rows rows of codes and metadata. Row widths
    are bytes, divisible by four. Init and close are collective on a CPU group.
    Use one CUDA stream and the same call order and shapes on every rank.
    HIP graphs are supported. Recreate before 2**31 - 1 calls.
    """

    def __init__(self, group, max_rows, code_bytes, metadata_bytes=0, device=None):
        from aiter.dist.device_communicators.custom_all_reduce import (
            IPCBufferPool,
            _loaded_hip_runtime,
            _should_use_vmm,
        )

        if get_arch() != "gfx1250" or dist.get_world_size(group) != 4:
            raise ValueError("Fp4AllGather requires four gfx1250 ranks")
        if max_rows <= 0 or code_bytes <= 0 or metadata_bytes < 0:
            raise ValueError("row count and code width must be positive")
        if code_bytes % 4 or metadata_bytes % 4:
            raise ValueError("row byte widths must be divisible by four")
        if _should_use_vmm(True):
            raise RuntimeError("Fp4AllGather requires HIP IPC (ROCm 7.14 or later)")
        self.group = group
        self.rank = dist.get_rank(group)
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda", torch.cuda.current_device())
        )
        self.max_rows = max_rows
        self.code_bytes = code_bytes
        self.metadata_bytes = metadata_bytes
        shape = (max_rows, code_bytes, metadata_bytes)
        shapes = [None] * 4
        dist.all_gather_object(shapes, shape, group=group)
        if any(item != shape for item in shapes):
            raise ValueError("all ranks must use the same buffer limits")
        self.slot_words = (
            triton.cdiv(max_rows * (code_bytes + metadata_bytes), 256) * 64
        )
        self.nbytes = 256 + 2 * self.slot_words * 4
        if self.nbytes >= 2**31:
            raise ValueError(
                "IPC storage must fit the buffer instructions' 2 GiB range"
            )
        self.pool = IPCBufferPool(self.device, group)
        inbox = self.pool.create("fp4_gather", self.nbytes, raw_cached=True)
        err = _loaded_hip_runtime().hipMemset(
            ctypes.c_void_p(inbox.data_ptr), 0, ctypes.c_size_t(256)
        )
        if err:
            self.pool.close()
            raise RuntimeError(f"hipMemset failed: {err}")
        torch.cuda.synchronize(self.device)
        self.peers = tuple(self.pool.open_peer_buffers("fp4_gather"))
        self.calls = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.errors = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.empty = torch.empty((0,), dtype=torch.uint8, device=self.device)
        path = resolve_config_dir("comms", "ALL-GATHER-FP4", backend="gluon")
        self.config = load_config_json(f"{path}/DEFAULT.json").copy()
        self.closed = False

    def close(self):
        """Wait for all ranks, unmap peers, then free this rank's IPC storage."""
        if self.closed:
            return
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        self.pool.close_peer_buffers()
        dist.barrier(group=self.group)
        self.pool.close()
        self.closed = True


def all_gather_fp4(codes, metadata, comm, out=None):
    """Gather packed uint8 codes and metadata rows in rank order, without quantization.

    codes is [local_rows, code_bytes]. metadata is uint8
    [local_rows, metadata_bytes], or None when metadata_bytes is zero.
    out can hold two preallocated contiguous uint8 outputs. The result shapes
    are [4 * local_rows, code_bytes] and [4 * local_rows, metadata_bytes].
    comm.errors counts peer timeouts; check it outside graph capture.
    """
    if comm.closed:
        raise RuntimeError("Fp4AllGather is closed")
    if codes.ndim != 2 or codes.dtype != torch.uint8 or not codes.is_contiguous():
        raise ValueError("codes must be a contiguous uint8 matrix")
    rows, width = codes.shape
    if not 0 < rows <= comm.max_rows or width != comm.code_bytes:
        raise ValueError("codes exceed the communication buffer limits")
    if codes.device != comm.device or codes.data_ptr() % 16:
        raise ValueError("codes must be aligned to 16 bytes on the communicator device")
    if metadata is None:
        if comm.metadata_bytes:
            raise ValueError("metadata is required")
        metadata = comm.empty.view(0, 0)
    elif (
        metadata.shape != (rows, comm.metadata_bytes)
        or metadata.dtype != torch.uint8
        or not metadata.is_contiguous()
        or metadata.device != codes.device
        or metadata.data_ptr() % 16
    ):
        raise ValueError("invalid metadata shape, dtype, device, or alignment")
    shapes = ((4 * rows, width), (4 * rows, comm.metadata_bytes))
    if out is None:
        out = tuple(
            torch.empty(s, dtype=torch.uint8, device=codes.device) for s in shapes
        )
    if len(out) != 2 or any(
        t.shape != s
        or t.dtype != torch.uint8
        or not t.is_contiguous()
        or t.device != codes.device
        or t.data_ptr() % 16
        for t, s in zip(out, shapes)
    ):
        raise ValueError("invalid output tensors")
    qwords = codes.numel() // 4
    mwords = metadata.numel() // 4
    block = comm.config["BLOCK_WORDS"]
    warps = comm.config["num_warps"]
    peer_group = comm.config["PEER_GROUP"]
    local_copy = comm.config["LOCAL_COPY"]
    programs = triton.cdiv(qwords, block) + triton.cdiv(mwords, block)
    _LOGGER.info(
        "ALL_GATHER_FP4 [gluon]: codes=%s metadata=%s", codes.shape, metadata.shape
    )
    _publish_kernel[(programs,)](
        codes.view(-1).view(torch.int32),
        metadata.view(-1).view(torch.int32),
        out[0].view(-1).view(torch.int32),
        out[1].view(-1).view(torch.int32),
        comm.peers,
        comm.calls,
        qwords,
        mwords,
        RANK=comm.rank,
        SLOT_WORDS=comm.slot_words,
        BLOCK_WORDS=block,
        LOCAL_COPY=local_copy,
        num_warps=warps,
    )
    _ready_kernel[(1,)](
        comm.peers,
        comm.calls,
        comm.errors,
        RANK=comm.rank,
        MAXIT=20_000_000,
        num_warps=1,
    )
    # Two banks prevent a fast rank from overwriting a peer's current read.
    # A rank cannot finish the next ready barrier until every peer has finished this read.
    peers_count = 3 if local_copy else 4
    tiles = triton.cdiv(programs, peer_group) * peer_group if peer_group else programs
    _gather_kernel[(tiles * peers_count,)](
        comm.peers,
        comm.calls,
        out[0].view(-1).view(torch.int32),
        out[1].view(-1).view(torch.int32),
        qwords,
        mwords,
        RANK=comm.rank,
        SLOT_WORDS=comm.slot_words,
        BLOCK_WORDS=block,
        PEER_GROUP=peer_group,
        LOCAL_COPY=local_copy,
        num_warps=warps,
    )
    return out
