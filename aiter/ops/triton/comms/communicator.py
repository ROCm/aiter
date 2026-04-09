# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
import sys
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)


def _dbg(msg: str) -> None:
    """Debug print to stderr with flush for crash debugging."""
    rank = dist.get_rank() if dist.is_initialized() else "?"
    print(f"[aiter rank={rank}] {msg}", file=sys.stderr, flush=True)


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

        # Zero-copy cache: data_ptr -> (sym_tensor, workspace)
        self._sym_cache: dict[int, tuple[torch.Tensor,
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
            _dbg(f"iris.iris(heap_size={self._HEAP_SIZE}, allocator_type={self._allocator_type!r}, coord_backend='gloo')")
            self._shmem = iris.iris(
                heap_size=self._HEAP_SIZE,
                allocator_type=self._allocator_type,
                coord_backend="gloo",
            )
            self.disabled = False
            rank = dist.get_rank(group)
            _dbg(f"iris init OK: rank={rank}/{world_size}")
        except Exception as e:
            _dbg(f"iris init FAILED: {e}")
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
        """Return cached (sym_tensor, workspace) for *inp*.

        On first call for a given ``data_ptr``, imports *inp* into the
        symmetric heap via DMA-BUF (``as_symmetric``).
        ``refresh_peer_access()`` is called once inside ``as_symmetric``
        — this is a collective, so all ranks must hit this together
        (safe because every rank executes the same allreduce in lockstep).
        """
        ptr = inp.data_ptr()
        entry = self._sym_cache.get(ptr)
        if entry is not None:
            return entry

        _dbg(f"as_symmetric: ptr={ptr:#x} shape={tuple(inp.shape)} dtype={inp.dtype} numel={inp.numel()} device={inp.device}")
        sym_tensor = self._shmem.as_symmetric(inp)
        _dbg(f"as_symmetric OK: sym_ptr={sym_tensor.data_ptr():#x} on_heap={self._shmem.is_symmetric(sym_tensor)}")
        _dbg(f"all_reduce_preamble: sym shape={tuple(sym_tensor.shape)} dtype={sym_tensor.dtype}")
        workspace = self._shmem.ccl.all_reduce_preamble(
            sym_tensor, sym_tensor,
        )
        _dbg(f"all_reduce_preamble OK")
        entry = (sym_tensor, workspace)
        self._sym_cache[ptr] = entry
        return entry

    def _all_reduce_zerocopy(self, inp: torch.Tensor) -> torch.Tensor:
        sym_tensor, workspace = self._get_or_import(inp)

        _dbg(f"all_reduce zerocopy: shape={tuple(sym_tensor.shape)} dtype={sym_tensor.dtype} ptr={sym_tensor.data_ptr():#x}")
        workspace = self._shmem.ccl.all_reduce(
            sym_tensor, sym_tensor, workspace=workspace
        )
        _dbg(f"all_reduce zerocopy OK")

        self._sym_cache[inp.data_ptr()] = (sym_tensor, workspace)
        return inp

    # ------------------------------------------------------------------
    # Legacy copy path (torch allocator fallback)
    # ------------------------------------------------------------------

    def _get_buffers(self, shape, dtype):
        """Get or allocate symmetric heap buffers. Reuses if shape/dtype match."""
        if self._buf_shape != shape or self._buf_dtype != dtype:
            shmem = self._shmem
            _dbg(f"_get_buffers: allocating shape={shape} dtype={dtype}")
            self._input_buf = shmem.empty(shape, dtype=dtype)
            self._output_buf = shmem.empty(shape, dtype=dtype)
            _dbg(f"_get_buffers OK: inp_ptr={self._input_buf.data_ptr():#x} out_ptr={self._output_buf.data_ptr():#x}")
            self._buf_shape = shape
            self._buf_dtype = dtype
            self._workspace = None
        return self._input_buf, self._output_buf

    def _all_reduce_copy(self, inp: torch.Tensor) -> torch.Tensor:
        input_buf, _ = self._get_buffers(inp.shape, inp.dtype)
        input_buf.copy_(inp)

        if self._workspace is None:
            _dbg(f"all_reduce_preamble copy: shape={tuple(input_buf.shape)} dtype={input_buf.dtype}")
            self._workspace = self._shmem.ccl.all_reduce_preamble(
                input_buf, input_buf,
            )
            _dbg(f"all_reduce_preamble copy OK")
        _dbg(f"all_reduce copy: shape={tuple(input_buf.shape)} ptr={input_buf.data_ptr():#x}")
        self._workspace = self._shmem.ccl.all_reduce(
            input_buf, input_buf, workspace=self._workspace
        )
        _dbg(f"all_reduce copy OK")

        inp.copy_(input_buf)
        return inp

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        assert self._shmem is not None

        _dbg(f"all_reduce: path={self._allocator_type} inp shape={tuple(inp.shape)} dtype={inp.dtype} ptr={inp.data_ptr():#x} contiguous={inp.is_contiguous()}")
        if self._allocator_type == "vmem":
            return self._all_reduce_zerocopy(inp)
        return self._all_reduce_copy(inp)
