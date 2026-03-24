# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
from typing import Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)


def _iris_available() -> bool:
    try:
        import iris  # noqa: F401
        return True
    except ImportError:
        return False


def _rocm_arch_available() -> bool:
    try:
        props = torch.cuda.get_device_properties(0)
        gcn_arch = getattr(props, "gcnArchName", "")
        return any(gfx in gcn_arch for gfx in ["gfx94", "gfx95"])
    except Exception:
        return False


class AiterCommunicator:
    """
    Aiter communicator using Iris CCL GPU-initiated communication.

    Uses the Iris symmetric heap for collective operations.
    """

    _SUPPORTED_WORLD_SIZES = [2, 4, 8]
    _SUPPORTED_DTYPES = [torch.float16, torch.bfloat16]
    _HEAP_SIZE = 2**31  # 2 GB

    def __init__(
        self, group: ProcessGroup, device: Union[int, str, torch.device]
    ) -> None:
        self.disabled = True
        self._shmem = None
        self._workspace = None
        self._input_buf = None
        self._output_buf = None
        self._buf_shape = None
        self._buf_dtype = None
        self._group = group

        if not _rocm_arch_available():
            logger.debug("Allreduce only supported on ROCm MI300/MI350 series.")
            return

        if not _iris_available():
            logger.warning("Iris library not available. Allreduce disabled.")
            return

        world_size = dist.get_world_size(group)
        if world_size not in self._SUPPORTED_WORLD_SIZES:
            logger.debug(
                "Allreduce not supported for world_size=%d", world_size
            )
            return

        try:
            import iris
            self._shmem = iris.iris(heap_size=self._HEAP_SIZE, coord_backend="gloo")
            self.disabled = False
            rank = dist.get_rank(group)
            logger.info(
                "Allreduce initialized: rank %d/%d coord_backend=gloo",
                rank, world_size,
            )
        except Exception as e:
            logger.warning("Failed to initialize Allreduce: %s", e)

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        if self.disabled or self._shmem is None:
            return False
        if inp.dtype not in self._SUPPORTED_DTYPES:
            return False
        if not inp.is_contiguous():
            return False
        # Need 2 buffers (input + output) from the heap
        buf_size = inp.numel() * inp.element_size()
        if buf_size * 2 > self._HEAP_SIZE:
            return False
        return True

    def _get_buffers(self, shape, dtype):
        """Get or allocate symmetric heap buffers. Reuses if shape/dtype match."""
        if self._buf_shape != shape or self._buf_dtype != dtype:
            shmem = self._shmem
            self._input_buf = shmem.empty(shape, dtype=dtype)
            self._output_buf = shmem.empty(shape, dtype=dtype)
            self._buf_shape = shape
            self._buf_dtype = dtype
            self._workspace = None
        return self._input_buf, self._output_buf

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        assert self._shmem is not None

        input_buf, output_buf = self._get_buffers(inp.shape, inp.dtype)

        input_buf.copy_(inp)

        self._workspace = self._shmem.ccl.all_reduce(
            output_buf, input_buf, workspace=self._workspace
        )

        out = torch.empty_like(inp)
        out.copy_(output_buf)
        return out
