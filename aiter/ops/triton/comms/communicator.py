# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, contextmanager
from typing import Iterator, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)

# Match CustomAllreduce default (8 MB).
_DEFAULT_MAX_SIZE = 8 * 1024 * 1024


def _iris_available() -> bool:
    try:
        import iris  # noqa: F401

        return True
    except ImportError:
        return False


def _is_weak_contiguous(inp: torch.Tensor) -> bool:
    return inp.is_contiguous() or (
        inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )


def _rocm_arch_available() -> bool:
    try:
        props = torch.cuda.get_device_properties(0)
        gcn_arch = getattr(props, "gcnArchName", "")
        return any(gfx in gcn_arch for gfx in ["gfx94", "gfx95"])
    except Exception:
        return False


class Communicator(ABC):
    """Interface for a TP all-reduce / all-gather backend behind vLLM's
    CudaCommunicator.

    Two implementations select at one factory (make_communicator):
    IrisCommunicator (production — iris gluon GPU-initiated CCL) and
    TorchCommunicator (a torch.distributed reference, the known-good control the
    iris path is measured and checked against). The surface mirrors what
    CudaCommunicator calls: a ``disabled`` flag, the should_*/all_* pairs
    (collectives are out-of-place — input untouched, new tensor returned), and a
    capture() context entered around cudagraph capture.
    """

    disabled: bool

    @abstractmethod
    def should_allreduce(self, inp: torch.Tensor) -> bool: ...

    @abstractmethod
    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def should_allgather(self, inp: torch.Tensor) -> bool: ...

    @abstractmethod
    def all_gather(self, inp: torch.Tensor, dim: int = -1) -> torch.Tensor: ...

    @abstractmethod
    def capture(self) -> AbstractContextManager[None]: ...


class IrisCommunicator(Communicator):
    """Communicator using Iris CCL GPU-initiated communication.

    API mirrors CustomAllreduce: __init__(group, device, max_size),
    should_allreduce, all_reduce (out-of-place), capture, plus disabled.
    """

    _SUPPORTED_WORLD_SIZES = [2, 4, 8]
    _SUPPORTED_DTYPES = [torch.float16, torch.bfloat16]
    _HEAP_SIZE = 2**33  # 8 GB
    _AG_SLAB_SIZE = 2**25  # 32 MB per rank

    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
        max_size: int = _DEFAULT_MAX_SIZE,
    ) -> None:
        self.disabled = True
        self.group = group
        self.max_size = max_size
        self._IS_CAPTURING = False
        self._shmem = None
        self._workspace = None
        self._input_buf = None
        self._buf_shape = None
        self._buf_dtype = None
        self._ag_input_slab = None
        self._ag_output_slab = None

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self.device = device

        if not _rocm_arch_available():
            logger.debug("IrisCommunicator disabled: unsupported ROCm arch")
            return

        if not _iris_available():
            logger.warning("Iris library not available. Allreduce disabled.")
            return

        try:
            import iris
            from iris.ccl.config import Config

            self._shmem = iris.iris(heap_size=self._HEAP_SIZE)
            self._gluon_config = Config(use_gluon=True)
        except Exception as e:
            logger.warning("Failed to initialize Allreduce: %s", e)
            return

        world_size = self._shmem.num_ranks
        if world_size not in self._SUPPORTED_WORLD_SIZES:
            logger.debug(
                "IrisCommunicator disabled: world_size=%d not in %s",
                world_size,
                self._SUPPORTED_WORLD_SIZES,
            )
            return

        self.disabled = False
        logger.info(
            "IrisCommunicator ready: world_size=%d heap=%dGB max_size=%dMB",
            world_size,
            self._HEAP_SIZE >> 30,
            self.max_size >> 20,
        )

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        if self.disabled or self._shmem is None:
            return False
        if not _is_weak_contiguous(inp):
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 16 != 0:
            return False
        if inp_size >= self.max_size:
            return False
        if inp.dtype not in self._SUPPORTED_DTYPES:
            return False
        if inp_size * 2 > self._HEAP_SIZE:
            return False
        return True

    def _get_buffers(self, shape, dtype):
        if self._buf_shape != shape or self._buf_dtype != dtype:
            assert self._shmem is not None
            self._input_buf = self._shmem.empty(shape, dtype=dtype)
            self._buf_shape = shape
            self._buf_dtype = dtype
            self._workspace = None
        return self._input_buf

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        assert self._shmem is not None
        try:
            out = torch.empty_like(inp)
            input_buf = self._get_buffers(inp.shape, inp.dtype)
            input_buf.copy_(inp)

            if self._workspace is None:
                self._workspace = self._shmem.ccl.all_reduce_preamble(
                    out, input_buf, config=self._gluon_config
                )
            self._workspace = self._shmem.ccl.all_reduce(
                out,
                input_buf,
                workspace=self._workspace,
                config=self._gluon_config,
                async_op=True,
            )

            return out
        except Exception as e:
            logger.error(
                "IrisCommunicator.all_reduce failed: shape=%s dtype=%s "
                "capturing=%s err=%s",
                tuple(inp.shape),
                inp.dtype,
                torch.cuda.is_current_stream_capturing(),
                e,
            )
            raise

    def should_allgather(self, inp: torch.Tensor) -> bool:
        """Replace every NCCL all_gather; only slab capacity falls back."""
        if self.disabled or self._shmem is None:
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size > self._AG_SLAB_SIZE:
            logger.warning(
                "IrisCommunicator.all_gather fallback to NCCL: %d bytes "
                "exceeds slab",
                inp_size,
            )
            return False
        return True

    def _get_allgather_buffers(self, numel, dtype):
        # Fixed byte slabs allocated once; per-call views avoid heap churn
        # (the symmetric heap never frees).
        if self._ag_input_slab is None:
            assert self._shmem is not None
            world_size = self._shmem.num_ranks
            self._ag_input_slab = self._shmem.empty(
                (self._AG_SLAB_SIZE,), dtype=torch.uint8
            )
            self._ag_output_slab = self._shmem.empty(
                (world_size, self._AG_SLAB_SIZE), dtype=torch.uint8
            )
        input_buf = self._ag_input_slab.view(dtype)[:numel].view(1, numel)
        output_buf = self._ag_output_slab.view(dtype)[:, :numel]
        return input_buf, output_buf

    def all_gather(self, inp: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert self._shmem is not None
        try:
            if dim < 0:
                dim += inp.dim()
            world_size = self._shmem.num_ranks
            input_size = inp.size()

            input_buf, output_buf = self._get_allgather_buffers(inp.numel(), inp.dtype)
            input_buf.view(-1).copy_(inp.reshape(-1))

            self._shmem.ccl.all_gather(
                output_buf,
                input_buf,
                config=self._gluon_config,
                async_op=True,
            )

            # Same reshape contract as vLLM's DeviceCommunicatorBase.all_gather.
            # output_buf is a non-contiguous slab view, so reshape always
            # copies; the result never aliases the symmetric heap.
            output = output_buf.reshape((world_size,) + input_size).movedim(0, dim)
            return output.reshape(
                input_size[:dim]
                + (world_size * input_size[dim],)
                + input_size[dim + 1 :]
            )

        except Exception as e:
            logger.error(
                "IrisCommunicator.all_gather failed: shape=%s dtype=%s "
                "capturing=%s err=%s",
                tuple(inp.shape),
                inp.dtype,
                torch.cuda.is_current_stream_capturing(),
                e,
            )
            raise

    @contextmanager
    def capture(self) -> Iterator[None]:
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False


class TorchCommunicator(Communicator):
    """torch.distributed reference with the Communicator interface — the
    known-good control IrisCommunicator is measured and checked against.

    Same output contract as IrisCommunicator: all_reduce returns a new SUM tensor
    (input untouched); all_gather returns the per-rank inputs concatenated along
    ``dim``, rank-ordered. The collective runs over the process group it is given,
    so the caller must pass the device (nccl/rccl) group, not a gloo cpu group.

    The should_* gates accept everything (torch.distributed is correct at any
    size/dtype), so this routes the same calls the iris path would plus the larger
    ones iris gates to NCCL. That is fine for the correctness control; aligning the
    gates with iris for routing parity is a perf-decomposition concern, not needed
    here.
    """

    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
        max_size: int = _DEFAULT_MAX_SIZE,
    ) -> None:
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self.group = group
        self.device = device
        self.max_size = max_size
        self.world_size = dist.get_world_size(group)
        self.disabled = False

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        return not self.disabled

    def should_allgather(self, inp: torch.Tensor) -> bool:
        return not self.disabled

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        out = inp.clone()
        dist.all_reduce(out, group=self.group)  # SUM
        return out

    def all_gather(self, inp: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim < 0:
            dim += inp.dim()
        input_size = inp.size()
        out = torch.empty(
            (self.world_size,) + tuple(input_size),
            dtype=inp.dtype,
            device=inp.device,
        )
        dist.all_gather_into_tensor(out, inp.contiguous(), group=self.group)
        return out.movedim(0, dim).reshape(
            input_size[:dim]
            + (self.world_size * input_size[dim],)
            + input_size[dim + 1 :]
        )

    @contextmanager
    def capture(self) -> Iterator[None]:
        # torch.distributed collectives need no special capture handling.
        yield


def make_communicator(
    backend: str,
    group: ProcessGroup,
    device: Union[int, str, torch.device],
    max_size: int = _DEFAULT_MAX_SIZE,
) -> Communicator:
    """Construct the TP collective backend at the one branching point.

    'iris' is production (gluon GPU-initiated CCL); 'torch' is the
    torch.distributed reference/control. Returns the communicator without raising
    on unavailability — the caller checks ``.disabled`` (IrisCommunicator
    self-disables on unsupported arch / missing iris / unsupported world size).
    """
    if backend == "iris":
        return IrisCommunicator(group, device, max_size)
    if backend == "torch":
        return TorchCommunicator(group, device, max_size)
    raise ValueError(f"unknown communicator backend {backend!r}")
