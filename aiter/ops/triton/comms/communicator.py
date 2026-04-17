# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
import os
from typing import Union

import torch

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

    Uses the Iris symmetric heap for collective operations. Input tensors
    are copied into symmetric heap buffers, reduced, and copied back.
    """

    _SUPPORTED_WORLD_SIZES = [2, 4, 8]
    _SUPPORTED_DTYPES = [torch.float16, torch.bfloat16]
    _HEAP_SIZE = 2**33  # 8 GB
    _MAX_NUM_TOKENS = 512

    def __init__(
        self,
        device: Union[int, str, torch.device],
    ) -> None:
        self.disabled = True
        self._shmem = None
        self._workspace = None

        # Copy-path buffers
        self._input_buf = None
        self._buf_shape = None
        self._buf_dtype = None

        if not _rocm_arch_available():
            logger.debug("Allreduce only supported on ROCm MI300/MI350 series.")
            return

        if not _iris_available():
            logger.warning("Iris library not available. Allreduce disabled.")
            return

        try:
            import iris

            self._shmem = iris.iris(
                heap_size=self._HEAP_SIZE,
                coord_backend="gloo",
            )
        except Exception as e:
            logger.warning("Failed to initialize Allreduce: %s", e)
            return

        world_size = self._shmem.num_ranks

        if world_size not in self._SUPPORTED_WORLD_SIZES:
            logger.debug("Allreduce not supported for world_size=%d", world_size)
            return

        self.disabled = False

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        if self.disabled or self._shmem is None:
            logger.critical("[AiterCommunicator] reject: disabled=%s shmem=%s", self.disabled, self._shmem is not None)
            return False
        if os.environ.get("AITER_COMMS_FORCE_FALLBACK") == "1":
            logger.critical("[AiterCommunicator] reject: AITER_COMMS_FORCE_FALLBACK=1")
            return False
        if inp.dtype not in self._SUPPORTED_DTYPES:
            logger.critical("[AiterCommunicator] reject: dtype=%s", inp.dtype)
            return False
        if not inp.is_contiguous():
            logger.critical("[AiterCommunicator] reject: noncontig shape=%s", tuple(inp.shape))
            return False
        # if inp.shape[0] > self._MAX_NUM_TOKENS:
        #     return False
        buf_size = inp.numel() * inp.element_size()
        if buf_size * 2 > self._HEAP_SIZE:
            logger.critical("[AiterCommunicator] reject: oversize buf=%d heap=%d", buf_size, self._HEAP_SIZE)
            return False
        return True

    def _validate_input(self, inp: torch.Tensor) -> None:
        """Validate tensor before allreduce."""
        assert self._shmem is not None, "Communicator not initialized"
        assert inp.is_cuda, f"Input must be on GPU, got {inp.device}"
        assert inp.is_contiguous(), "Input must be contiguous"
        assert inp.dtype in self._SUPPORTED_DTYPES, f"Unsupported dtype: {inp.dtype}"

    def _get_buffers(self, shape, dtype):
        """Get or allocate symmetric heap buffers. Reuses if shape/dtype match."""
        if self._buf_shape != shape or self._buf_dtype != dtype:
            self._input_buf = self._shmem.empty(shape, dtype=dtype)
            self._buf_shape = shape
            self._buf_dtype = dtype
            self._workspace = None
        return self._input_buf

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        self._validate_input(inp)
        try:
            input_buf = self._get_buffers(inp.shape, inp.dtype)
            input_buf.copy_(inp)

            if self._workspace is None:
                self._workspace = self._shmem.ccl.all_reduce_preamble(
                    input_buf,
                    input_buf,
                )
            self._workspace = self._shmem.ccl.all_reduce(
                input_buf, input_buf, workspace=self._workspace
            )

            inp.copy_(input_buf)
            return inp
        except Exception as e:
            logger.error(
                "[AiterCommunicator] all_reduce raised shape=%s dtype=%s capturing=%s: %s",
                tuple(inp.shape),
                inp.dtype,
                torch.cuda.is_current_stream_capturing(),
                e,
            )
            raise
