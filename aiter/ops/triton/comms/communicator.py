# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
from typing import Optional, Union

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
    When ``allocator_type="vmem"`` and the input tensor's data pointer is
    stable across calls (e.g. CUDA-graph replay), the communicator imports
    the tensor once via DMA-BUF and skips all host-side copies on
    subsequent invocations (zero-copy path).
    """

    _SUPPORTED_WORLD_SIZES = [2, 4, 8]
    _SUPPORTED_DTYPES = [torch.float16, torch.bfloat16]
    _HEAP_SIZE = 2**33  # 8 GB
    _MAX_NUM_TOKENS = 512

    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
        allocator_type: str = "vmem",
    ) -> None:
        self.disabled = True
        self._shmem = None
        self._workspace = None
        self._group = group

        # Legacy copy-path buffers (torch allocator fallback)
        self._input_buf = None
        self._output_buf = None
        self._buf_shape = None
        self._buf_dtype = None

        # Zero-copy cache: data_ptr -> (sym_input, sym_output, workspace)
        self._sym_cache: dict[int, tuple[torch.Tensor, torch.Tensor,
                                         Optional[object]]] = {}
        self._allocator_type: str = allocator_type

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
            self._shmem = iris.iris(
                heap_size=self._HEAP_SIZE,
                allocator_type=self._allocator_type,
                coord_backend="gloo",
            )
            self.disabled = False
            rank = dist.get_rank(group)
            logger.info(
                "Allreduce initialized: rank %d/%d allocator=%s",
                rank, world_size, self._allocator_type,
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
        # Skip large tensors (e.g. prefill) where per-allreduce host barriers
        # serialize the GPU pipeline.
        if inp.shape[0] > self._MAX_NUM_TOKENS:
            return False
        buf_size = inp.numel() * inp.element_size()
        if buf_size * 2 > self._HEAP_SIZE:
            return False
        return True

    # ------------------------------------------------------------------
    # Zero-copy path (vmem allocator)
    # ------------------------------------------------------------------

    def _get_or_import(self, inp: torch.Tensor):
        """Return cached (sym_input, sym_output, workspace) for *inp*.

        On first call for a given ``data_ptr``, imports *inp* into the
        symmetric heap via DMA-BUF (``as_symmetric``) and allocates an
        output buffer on the heap.  ``refresh_peer_access()`` is called
        once inside ``as_symmetric`` — this is a collective, so all ranks
        must hit this together (safe because every rank executes the same
        allreduce in lockstep).
        """
        ptr = inp.data_ptr()
        entry = self._sym_cache.get(ptr)
        if entry is not None:
            return entry

        shmem = self._shmem
        sym_input = shmem.as_symmetric(inp)
        sym_output = shmem.empty(inp.shape, dtype=inp.dtype)
        entry = (sym_input, sym_output, None)
        self._sym_cache[ptr] = entry
        logger.debug(
            "Imported tensor ptr=%#x shape=%s dtype=%s into symmetric heap",
            ptr, inp.shape, inp.dtype,
        )
        return entry

    def _all_reduce_zerocopy(self, inp: torch.Tensor) -> torch.Tensor:
        sym_input, sym_output, workspace = self._get_or_import(inp)

        workspace = self._shmem.ccl.all_reduce(
            sym_output, sym_input, workspace=workspace
        )

        # Update cached workspace for next call
        ptr = inp.data_ptr()
        self._sym_cache[ptr] = (sym_input, sym_output, workspace)

        # Copy result back into the original tensor so the caller's
        # reference is updated in-place (mirrors the contract of the
        # copy path where a new tensor is returned, but avoids an extra
        # allocation — the caller already holds *inp*).
        inp.copy_(sym_output)
        return inp

    # ------------------------------------------------------------------
    # Legacy copy path (torch allocator fallback)
    # ------------------------------------------------------------------

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

    def _all_reduce_copy(self, inp: torch.Tensor) -> torch.Tensor:
        input_buf, output_buf = self._get_buffers(inp.shape, inp.dtype)
        input_buf.copy_(inp)

        self._workspace = self._shmem.ccl.all_reduce(
            output_buf, input_buf, workspace=self._workspace
        )

        out = torch.empty_like(inp)
        out.copy_(output_buf)
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        assert self._shmem is not None

        if self._allocator_type == "vmem":
            return self._all_reduce_zerocopy(inp)
        return self._all_reduce_copy(inp)
