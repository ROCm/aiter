# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
import sys
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)


def _dbg(msg: str, rank: object = "?") -> None:
    """Debug print to stderr with flush for crash debugging."""
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
        device: Union[int, str, torch.device],
        allocator_type: str = "vmem",
    ) -> None:
        self.disabled = True
        self._shmem = None
        self._workspace = None

        # Legacy copy-path buffers (torch allocator fallback)
        self._input_buf = None
        self._output_buf = None
        self._buf_shape = None
        self._buf_dtype = None

        # Zero-copy cache: data_ptr -> (sym_tensor, workspace)
        self._sym_cache: dict[int, tuple[torch.Tensor, Optional[object]]] = {}
        self._allocator_type: str = allocator_type

        if not _rocm_arch_available():
            logger.debug("Allreduce only supported on ROCm MI300/MI350 series.")
            return

        if not _iris_available():
            logger.warning("Iris library not available. Allreduce disabled.")
            return

        try:
            import iris

            _dbg(
                f"iris.iris(heap_size={self._HEAP_SIZE}, allocator_type={self._allocator_type!r}, coord_backend='gloo')"
            )
            self._shmem = iris.iris(
                heap_size=self._HEAP_SIZE,
                allocator_type=self._allocator_type,
                coord_backend="gloo",
            )
        except Exception as e:
            _dbg(f"iris init FAILED: {e}")
            logger.warning("Failed to initialize Allreduce: %s", e)
            return

        world_size = self._shmem.num_ranks
        rank = self._shmem.cur_rank

        if world_size not in self._SUPPORTED_WORLD_SIZES:
            logger.debug("Allreduce not supported for world_size=%d", world_size)
            return

        self.disabled = False
        _dbg(f"iris init OK: rank={rank}/{world_size}", rank=rank)

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

    def _validate_input(self, inp: torch.Tensor) -> None:
        """Validate tensor before allreduce."""
        assert self._shmem is not None, "Communicator not initialized"
        assert inp.is_cuda, f"Input must be on GPU, got {inp.device}"
        assert inp.is_contiguous(), "Input must be contiguous"
        assert inp.dtype in self._SUPPORTED_DTYPES, f"Unsupported dtype: {inp.dtype}"

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
            sym_tensor, _ = entry
            # Detect stale cache: data_ptr reused with different shape/dtype
            assert sym_tensor.shape == inp.shape and sym_tensor.dtype == inp.dtype, (
                f"Stale sym_cache: cached shape={tuple(sym_tensor.shape)} dtype={sym_tensor.dtype} "
                f"vs input shape={tuple(inp.shape)} dtype={inp.dtype} at ptr={ptr:#x}"
            )
            return entry

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"DMA-BUF import for ptr={ptr:#x} during CUDA graph recording. "
                f"allreduce must be a splitting op so it runs in eager mode. "
                f"Add 'vllm::all_reduce' to compilation_config.splitting_ops."
            )

        rank = self._shmem.cur_rank
        _dbg(
            f"as_symmetric: ptr={ptr:#x} shape={tuple(inp.shape)} dtype={inp.dtype} numel={inp.numel()} device={inp.device}",
            rank=rank,
        )
        sym_tensor = self._shmem.as_symmetric(inp)
        assert self._shmem.is_symmetric(
            sym_tensor
        ), f"as_symmetric returned non-heap tensor: ptr={sym_tensor.data_ptr():#x}"
        _dbg(f"as_symmetric OK: sym_ptr={sym_tensor.data_ptr():#x}", rank=rank)
        _dbg(
            f"all_reduce_preamble: sym shape={tuple(sym_tensor.shape)} dtype={sym_tensor.dtype}",
            rank=rank,
        )
        workspace = self._shmem.ccl.all_reduce_preamble(
            sym_tensor,
            sym_tensor,
        )
        _dbg("all_reduce_preamble OK", rank=rank)
        entry = (sym_tensor, workspace)
        self._sym_cache[ptr] = entry
        return entry

    def _all_reduce_zerocopy(self, inp: torch.Tensor) -> torch.Tensor:
        sym_tensor, workspace = self._get_or_import(inp)

        rank = self._shmem.cur_rank
        _dbg(
            f"all_reduce zerocopy: shape={tuple(sym_tensor.shape)} dtype={sym_tensor.dtype} ptr={sym_tensor.data_ptr():#x}",
            rank=rank,
        )
        workspace = self._shmem.ccl.all_reduce(
            sym_tensor, sym_tensor, workspace=workspace
        )
        _dbg("all_reduce zerocopy OK", rank=rank)

        self._sym_cache[inp.data_ptr()] = (sym_tensor, workspace)
        return inp

    # ------------------------------------------------------------------
    # Legacy copy path (torch allocator fallback)
    # ------------------------------------------------------------------

    def _get_buffers(self, shape, dtype):
        """Get or allocate symmetric heap buffers. Reuses if shape/dtype match."""
        if self._buf_shape != shape or self._buf_dtype != dtype:
            shmem = self._shmem
            rank = shmem.cur_rank
            _dbg(f"_get_buffers: allocating shape={shape} dtype={dtype}", rank=rank)
            self._input_buf = shmem.empty(shape, dtype=dtype)
            self._output_buf = shmem.empty(shape, dtype=dtype)
            _dbg(
                f"_get_buffers OK: inp_ptr={self._input_buf.data_ptr():#x} out_ptr={self._output_buf.data_ptr():#x}",
                rank=rank,
            )
            self._buf_shape = shape
            self._buf_dtype = dtype
            self._workspace = None
        return self._input_buf, self._output_buf

    def _all_reduce_copy(self, inp: torch.Tensor) -> torch.Tensor:
        input_buf, _ = self._get_buffers(inp.shape, inp.dtype)
        input_buf.copy_(inp)

        rank = self._shmem.cur_rank
        if self._workspace is None:
            _dbg(
                f"all_reduce_preamble copy: shape={tuple(input_buf.shape)} dtype={input_buf.dtype}",
                rank=rank,
            )
            self._workspace = self._shmem.ccl.all_reduce_preamble(
                input_buf,
                input_buf,
            )
            _dbg("all_reduce_preamble copy OK", rank=rank)
        _dbg(
            f"all_reduce copy: shape={tuple(input_buf.shape)} ptr={input_buf.data_ptr():#x}",
            rank=rank,
        )
        self._workspace = self._shmem.ccl.all_reduce(
            input_buf, input_buf, workspace=self._workspace
        )
        _dbg("all_reduce copy OK", rank=rank)

        inp.copy_(input_buf)
        return inp

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        self._validate_input(inp)

        _dbg(
            f"all_reduce: path={self._allocator_type} inp shape={tuple(inp.shape)} dtype={inp.dtype} ptr={inp.data_ptr():#x} contiguous={inp.is_contiguous()}",
            rank=self._shmem.cur_rank,
        )
        if self._allocator_type == "vmem":
            return self._all_reduce_zerocopy(inp)
        return self._all_reduce_copy(inp)
