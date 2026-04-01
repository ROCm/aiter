"""
* Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
* Copyright (C) 2024-2026, The vLLM team.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
"""

import pickle
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

# import vllm.envs as envs
# from vllm import _custom_ops as ops
import aiter as ops
from aiter.dist.parallel_state import in_the_same_node_as
from aiter import logger
from aiter.utility.dtypes import fp8, _torch_to_aiter_dtype, torch_to_aiter_pybind

try:
    ops.meta_size()
    custom_ar = True
except Exception as e:
    # For CPUs
    custom_ar = False
    logger.warning(f"Custom allreduce is disabled: {e}")


def is_weak_contiguous(inp: torch.Tensor):
    return inp.is_contiguous() or (
        inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )


_torch_to_aiter = torch_to_aiter_pybind


def _current_stream_ptr() -> int:
    return torch.cuda.current_stream().cuda_stream


class IPCBuffer:
    """A single IPC-accessible device buffer.

    Pure data container — owns a pre-allocated GPU allocation with a fixed
    device address.  All IPC handle / broadcast / registration logic lives
    in IPCBufferPool.

    When *uncached* is False (default), memory is allocated through PyTorch's
    caching allocator (torch.empty).  When True, memory is allocated via
    hipExtMallocWithFlags with hipDeviceMallocUncached, bypassing the cache.
    Uncached buffers are suitable for cross-GPU synchronization metadata and
    signal buffers where cache coherence overhead is undesirable.
    """

    def __init__(
        self,
        size: int,
        device: torch.device,
        uncached: bool = False,
    ):
        self._size = size
        self._uncached = uncached
        if uncached:
            self._buffer = None
            self._raw_ptr = ops.allocate_meta_buffer(size, _current_stream_ptr())
        else:
            self._buffer = torch.empty(size, dtype=torch.uint8, device=device)
            self._raw_ptr = self._buffer.data_ptr()

    @property
    def data_ptr(self) -> int:
        return self._raw_ptr

    @property
    def tensor(self) -> torch.Tensor:
        if self._buffer is None:
            raise RuntimeError(
                "Uncached IPCBuffer has no backing tensor; use .data_ptr"
            )
        return self._buffer

    @property
    def max_size(self) -> int:
        return self._size

    @property
    def uncached(self) -> bool:
        return self._uncached

    def __del__(self):
        if self._uncached and self._raw_ptr:
            ops.free_meta_buffer(self._raw_ptr)
            self._raw_ptr = 0


class IPCBufferPool:
    """Manages a collection of named IPCBuffers and provides IPC broadcast
    infrastructure for cross-GPU communication.

    Buffers are stored in an internal dict and accessed by string key.

    Two sets of operations:

    Eager mode (named internal buffers):
        create(key, size) allocates a buffer and stores it under *key*.
        get_ipc_meta(key) broadcasts IPC handles for that buffer.

    Graph mode (arbitrary external tensors):
        get_external_ipc_meta(tensor) broadcasts IPC handles for any tensor.
        flush_graph_buffers(ar_ptr) batch-registers addresses that the C++
        backend collected during CUDA graph capture.
    """

    _pool_seq: int = 0

    def __init__(self, device: torch.device, group: ProcessGroup):
        self._device = device
        self._group = group
        self._rank = dist.get_rank(group=group)
        self._world_size = dist.get_world_size(group=group)
        self._buffers: Dict[str, IPCBuffer] = {}

        self._store = dist.distributed_c10d._get_default_store()
        self._assert_pure_tcp_store(self._store)

        ranks_tag = "_".join(map(str, sorted(dist.get_process_group_ranks(group))))
        self._store_key_prefix = f"aiter_ipc/p{IPCBufferPool._pool_seq}/g{ranks_tag}"
        IPCBufferPool._pool_seq += 1
        self._ipc_seq = 0

    @staticmethod
    def _assert_pure_tcp_store(store) -> None:
        """Verify the store is a pure-TCP KV store, free from any collective
        communication backend (RCCL / gloo / MPI)."""
        s = store
        while isinstance(s, dist.PrefixStore):
            s = s.underlying_store
        assert isinstance(s, dist.TCPStore), (
            f"IPC metadata exchange requires a pure-TCP KV store "
            f"(torch.distributed.TCPStore), got {type(s).__name__}. "
            f"This ensures the exchange is backend-free — no RCCL, "
            f"gloo, or MPI collective is involved."
        )

    # ---- Buffer lifecycle ----

    def create(self, key: str, size: int, uncached: bool = False) -> IPCBuffer:
        """Allocate a new IPCBuffer and store it under *key*.

        Args:
            key: unique name for this buffer in the pool.
            size: buffer size in bytes.
            uncached: if True, allocate via hipMalloc (uncached);
                      if False (default), allocate via torch.empty (cached).
        """
        if key in self._buffers:
            raise KeyError(f"IPCBuffer '{key}' already exists in the pool")
        buf = IPCBuffer(size, self._device, uncached=uncached)
        self._buffers[key] = buf
        return buf

    def __getitem__(self, key: str) -> IPCBuffer:
        return self._buffers[key]

    def __contains__(self, key: str) -> bool:
        return key in self._buffers

    # ---- Eager mode: named buffer IPC meta ----

    def get_ipc_meta(self, key: str) -> Tuple[List, List]:
        """Broadcast IPC handles for the named buffer across all ranks."""
        buf = self._buffers[key]
        return self._broadcast_ipc(buf.data_ptr)

    # ---- Graph mode: external buffer IPC meta ----

    def get_external_ipc_meta(self, tensor: torch.Tensor) -> Tuple[List, List]:
        """Broadcast IPC handles for an arbitrary external tensor."""
        return self._broadcast_ipc(tensor.data_ptr())

    def flush_graph_buffers(self, ar_ptr):
        """Batch-register buffer addresses collected during CUDA graph capture.

        During graph capture the C++ backend records addresses of buffers that
        are not yet IPC-registered.  After capture ends this method exchanges
        their IPC handles across all ranks and completes registration.
        """
        count = ops.get_graph_buffer_count(ar_ptr)
        if count == 0:
            return
        handle_sz = 64  # sizeof(hipIpcMemHandle_t)
        handle = torch.empty(count * handle_sz, dtype=torch.uint8)
        offset = torch.empty(count, dtype=torch.int64)
        ops.get_graph_buffer_ipc_meta(ar_ptr, handle.data_ptr(), offset.data_ptr())
        handles, offsets = self._gather_ipc_meta((handle, offset))
        logger.info("Registering %d cuda graph addresses", count)
        ops.register_graph_buffers(
            ar_ptr,
            [h.data_ptr() for h in handles],
            [o.data_ptr() for o in offsets],
        )

    # ---- Private IPC primitives ----

    def _broadcast_ipc(self, data_ptr: int) -> Tuple[List, List]:
        """Get IPC handle for *data_ptr* and broadcast across all ranks."""
        handle = torch.empty(64, dtype=torch.uint8)  # sizeof(hipIpcMemHandle_t)
        ops.get_meta_buffer_ipc_handle(data_ptr, handle.data_ptr())
        return self._gather_ipc_meta((handle, 0))

    def _gather_ipc_meta(self, shard_data) -> Tuple[List, List]:
        """Exchange IPC metadata (handle + offset) across all ranks via TCP store.

        Each rank writes its serialised *shard_data* under a unique key, then
        reads every other rank's data.  ``store.get()`` blocks until the key
        is available, providing natural barrier semantics without involving any
        collective communication backend.
        """
        seq = self._ipc_seq
        self._ipc_seq += 1
        prefix = f"{self._store_key_prefix}/{seq}"

        self._store.set(f"{prefix}/r{self._rank}", pickle.dumps(shard_data))

        handles = []
        offsets = []
        for r in range(self._world_size):
            raw = self._store.get(f"{prefix}/r{r}")
            h, o = pickle.loads(raw)
            handles.append(h)
            offsets.append(o)
        return handles, offsets


class CustomAllreduce:

    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]

    # max_size: max supported allreduce size
    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
        max_size=1024 * 1024 * 1024,  # 2GB bf16/half
        enable_register_for_capturing: bool = True,
    ) -> None:
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the CustomAllreduce to. If None,
                it will be bind to f"cuda:{local_rank}".
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same node.
        """
        self._IS_CAPTURING = False
        self.disabled = True

        if not custom_ar:
            # disable because of missing custom allreduce library
            # e.g. in a non-cuda environment
            return

        self.group = group

        assert (
            dist.get_backend(group) != dist.Backend.NCCL
        ), "CustomAllreduce should be attached to a non-NCCL group."

        if not all(in_the_same_node_as(group, source_rank=0)):
            # No need to initialize custom allreduce for multi-node case.
            logger.warning(
                "Custom allreduce is disabled because this process group"
                " spans across nodes."
            )
            return

        rank = dist.get_rank(group=self.group)
        world_size = dist.get_world_size(group=self.group)
        if world_size == 1:
            # No need to initialize custom allreduce for single GPU case.
            return

        if world_size not in CustomAllreduce._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "Custom allreduce is disabled due to an unsupported world"
                " size: %d. Supported world sizes: %s. To silence this "
                "warning, specify disable_custom_all_reduce=True explicitly.",
                world_size,
                str(CustomAllreduce._SUPPORTED_WORLD_SIZES),
            )
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device

        # device_ids = get_cuda_visible_devices()

        # physical_device_id = device_ids[device.index]
        # tensor = torch.tensor([physical_device_id], dtype=torch.int, device="cpu")
        # gather_list = [
        #     torch.tensor([0], dtype=torch.int, device="cpu") for _ in range(world_size)
        # ]
        # dist.all_gather(gather_list, tensor, group=self.group)
        # physical_device_ids = [t.item() for t in gather_list]

        # test nvlink first, this will filter out most of the cases
        # where custom allreduce is not supported
        # this checks hardware and driver support for NVLink
        # assert current_platform.is_cuda() or current_platform.is_rocm()
        # fully_connected = current_platform.is_full_nvlink(physical_device_ids)
        fully_connected = True
        if world_size > 2 and not fully_connected:
            logger.warning(
                "Custom allreduce is disabled because it's not supported on"
                " more than two PCIe-only GPUs. To silence this warning, "
                "specify disable_custom_all_reduce=True explicitly."
            )
            return
        # test P2P capability, this checks software/cudaruntime support
        # this is expensive to compute at the first time
        # then we cache the result
        # On AMD GPU, p2p is always enabled between XGMI connected GPUs
        # if not current_platform.is_rocm() and not _can_p2p(rank, world_size):
        #     logger.warning(
        #         "Custom allreduce is disabled because your platform lacks "
        #         "GPU P2P capability or P2P test failed. To silence this "
        #         "warning, specify disable_custom_all_reduce=True explicitly.")
        #     return

        self.disabled = False
        self.enable_register_for_capturing = enable_register_for_capturing
        # This is a buffer for storing the tuples of pointers pointing to
        # IPC buffers from all ranks. Each registered tuple has size of
        # 8*world_size bytes where world_size is at most 8. Allocating 8MB
        # is enough for 131072 such tuples. The largest model I've seen only
        # needs less than 10000 of registered tuples.
        self.rank_data = torch.empty(
            8 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        self.max_size = max_size
        self.rank = rank
        self.world_size = world_size

        # Create IPC buffer pool and allocate all named buffers.
        # "meta" uses hipAlloc (uncached) for synchronization metadata +
        # intermediate allreduce temp storage.
        # "input" / "output" use torchAlloc (cached) for D2D relay in
        # eager mode.
        self._pool = IPCBufferPool(self.device, self.group)
        self._pool.create("meta", ops.meta_size() + max_size * 2, uncached=True)
        self._pool.create("input", max_size)
        self._pool.create("output", max_size)

        # Exchange meta buffer IPC handles to initialize C++ backend
        handles, offsets = self._pool.get_ipc_meta("meta")

        self.fully_connected = fully_connected
        self._ptr = ops.init_custom_ar(
            self._pool["meta"].data_ptr,
            self.rank_data.data_ptr(),
            self.rank_data.numel(),
            [h.data_ptr() for h in handles],
            offsets,
            rank,
            self.fully_connected,
        )

        # Register input/output IPC buffers with the C++ backend
        handles, offsets = self._pool.get_ipc_meta("input")
        ops.register_input_buffer(
            self._ptr,
            self._pool["input"].data_ptr,
            [h.data_ptr() for h in handles],
            offsets,
        )

        handles, offsets = self._pool.get_ipc_meta("output")
        ops.register_output_buffer(
            self._ptr,
            self._pool["output"].data_ptr,
            [h.data_ptr() for h in handles],
            offsets,
        )

    @contextmanager
    def capture(self):
        """
        The main responsibility of this context manager is the
        flush_graph_buffers call at the end of the context.
        It records all the buffer addresses used in the CUDA graph.
        """
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            if not self.disabled:
                self._pool.flush_graph_buffers(self._ptr)

    def register_input_buffer(self, inp: torch.Tensor):
        """Register an external tensor as an IPC input buffer."""
        handles, offsets = self._pool.get_external_ipc_meta(inp)
        ops.register_input_buffer(
            self._ptr, inp.data_ptr(), [h.data_ptr() for h in handles], offsets
        )

    def register_output_buffer(self, out: torch.Tensor):
        """Register an external tensor as an IPC output buffer."""
        handles, offsets = self._pool.get_external_ipc_meta(out)
        ops.register_output_buffer(
            self._ptr, out.data_ptr(), [h.data_ptr() for h in handles], offsets
        )

    def register_graph_buffers(self):
        """Batch-register graph-captured buffer addresses."""
        self._pool.flush_graph_buffers(self._ptr)

    def should_custom_ar(self, inp: torch.Tensor):
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        # for 4 or more non NVLink-capable GPUs, custom allreduce provides
        # little performance improvement over NCCL.
        # In allreduce 2stage writemode, use 2x tmp buffer
        if self.world_size == 2 or self.fully_connected:
            return inp_size <= (self.max_size / 2)
        return False

    def should_custom_ag(self, inp: torch.Tensor):
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        # all_gather output = input * world_size, so the per-rank input
        # must fit within max_size / world_size
        if self.world_size == 2 or self.fully_connected:
            return inp_size <= (self.max_size / (self.world_size * 2))
        return False

    def all_reduce(
        self,
        inp: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        use_new: bool = True,
        open_fp8_quant: bool = False,
        registered_input: bool = False,
        registered_output: bool = False,
    ):
        """Performs an out-of-place all reduce.

        If registered is True, this assumes inp's pointer is already
        IPC-registered. Otherwise, inp is first copied into a pre-registered
        buffer.
        """
        if out is None:
            out = torch.empty_like(inp)
        assert is_weak_contiguous(out), "output tensor is not weak-contiguous"
        stream = _current_stream_ptr()
        reg_inp = 0 if registered_input else self._pool["input"].data_ptr
        reg_inp_bytes = 0 if registered_input else self._pool["input"].max_size
        reg_out = 0 if registered_output else self._pool["output"].data_ptr
        reg_out_bytes = 0 if registered_output else self._pool["output"].max_size
        ops.all_reduce(
            self._ptr,
            _torch_to_aiter(inp),
            _torch_to_aiter(out),
            use_new,
            open_fp8_quant,
            reg_inp,
            reg_inp_bytes,
            reg_out,
            reg_out_bytes,
            stream,
        )
        return out

    def custom_all_reduce(
        self, input: torch.Tensor, use_new: bool = True, open_fp8_quant: bool = False
    ) -> Optional[torch.Tensor]:
        # when custom allreduce is disabled, this will be None
        if self.disabled or not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.all_reduce(
                    input,
                    use_new=use_new,
                    open_fp8_quant=open_fp8_quant,
                    registered_input=self.enable_register_for_capturing,
                    registered_output=self.enable_register_for_capturing,
                )
            else:
                # if warm up, mimic the allocation pattern
                # since custom allreduce is out-of-place
                return torch.zeros_like(input)
        else:
            # note: outside of cuda graph context,
            # custom allreduce incurs a cost of cudaMemcpy, which should
            # be small(<=1% of overall latency) compared to the performance
            # gains of using custom kernels
            return self.all_reduce(
                input,
                use_new=use_new,
                open_fp8_quant=open_fp8_quant,
                registered_input=False,
                registered_output=False,
            )

    def reduce_scatter(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        *,
        registered: bool = False,
    ):
        assert is_weak_contiguous(out), "output tensor is not weak-contiguous"
        stream = _current_stream_ptr()
        reg = 0 if registered else self._pool["input"].data_ptr
        reg_bytes = 0 if registered else self._pool["input"].max_size
        ops.reduce_scatter(
            self._ptr,
            _torch_to_aiter(inp),
            _torch_to_aiter(out),
            reg,
            reg_bytes,
            stream,
        )

    def custom_reduce_scatter(
        self, input: torch.Tensor, output: torch.Tensor
    ) -> Optional[torch.Tensor]:
        # when custom allreduce is disabled, this will be None
        if self.disabled or not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.reduce_scatter(input, output, registered=True)
        else:
            return self.reduce_scatter(input, output, registered=False)

    def _allgather_out_shape(self, inp: torch.Tensor, dim: int):
        ndim = inp.dim()
        if dim == 0:
            return (inp.shape[0] * self.world_size,) + inp.shape[1:]
        if dim == -1 or dim == ndim - 1:
            return inp.shape[:-1] + (inp.shape[-1] * self.world_size,)
        print(
            f"[aiter] allgather does not support dim={dim}, falling back to 1-D output"
        )
        return (inp.numel() * self.world_size,)

    def all_gather_reg(self, inp: torch.Tensor, out: torch.Tensor = None, dim: int = 0):
        if out is None:
            out = torch.empty(
                self._allgather_out_shape(inp, dim),
                dtype=inp.dtype,
                device=inp.device,
            )
        assert is_weak_contiguous(out), "output tensor is not weak-contiguous"
        stream = _current_stream_ptr()
        ops.all_gather_reg(
            self._ptr,
            _torch_to_aiter(inp),
            _torch_to_aiter(out),
            dim,
            stream,
        )
        return out

    def all_gather_unreg(
        self, inp: torch.Tensor, out: torch.Tensor = None, dim: int = 0
    ):
        if out is None:
            out = torch.empty(
                self._allgather_out_shape(inp, dim),
                dtype=inp.dtype,
                device=inp.device,
            )
        assert is_weak_contiguous(out), "output tensor is not weak-contiguous"
        stream = _current_stream_ptr()
        ops.all_gather_unreg(
            self._ptr,
            _torch_to_aiter(inp),
            self._pool["input"].data_ptr,
            _torch_to_aiter(out),
            self._pool["input"].max_size,
            dim,
            stream,
        )
        return out

    def custom_all_gather(
        self, inp: torch.Tensor, dim: int = 0
    ) -> Optional[torch.Tensor]:
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.all_gather_reg(inp, dim=dim)
            else:
                print("allgather capture hipgraph error")
                return torch.zeros_like(inp)
        else:
            return self.all_gather_unreg(inp, dim=dim)

    def fused_ar_rms(
        self,
        inp: torch.Tensor,
        res_inp: torch.Tensor,
        *,
        res_out: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        scale_out: Optional[torch.Tensor] = None,
        w: torch.Tensor,
        eps: float,
        registered: bool = False,
        use_1stage: bool = False,
        post_per_token_quant: bool = False,
    ):
        if res_out is None:
            res_out = torch.empty_like(inp)
        stream = _current_stream_ptr()
        reg = 0 if registered else self._pool["input"].data_ptr
        reg_bytes = 0 if registered else self._pool["input"].max_size
        if not post_per_token_quant:
            if out is None:
                out = torch.empty_like(inp)
            assert is_weak_contiguous(out), "output tensor is not weak-contiguous"
            ops.fused_allreduce_rmsnorm(
                self._ptr,
                _torch_to_aiter(inp),
                _torch_to_aiter(res_inp),
                _torch_to_aiter(res_out),
                _torch_to_aiter(out),
                _torch_to_aiter(w),
                eps,
                reg,
                reg_bytes,
                use_1stage,
                stream,
            )
            return out, res_out
        else:
            if out is None:
                out = torch.empty(inp.shape, dtype=fp8, device=inp.device)
            assert is_weak_contiguous(out), "output tensor is not weak-contiguous"
            if scale_out is None:
                scale_out = torch.empty(
                    inp.shape[:-1] + (1,), dtype=torch.float32, device=inp.device
                )
            ops.fused_allreduce_rmsnorm_quant(
                self._ptr,
                _torch_to_aiter(inp),
                _torch_to_aiter(res_inp),
                _torch_to_aiter(res_out),
                _torch_to_aiter(out),
                _torch_to_aiter(scale_out),
                _torch_to_aiter(w),
                eps,
                reg,
                reg_bytes,
                use_1stage,
                stream,
            )
            return out, res_out, scale_out

    def custom_fused_ar_rms(
        self,
        input: torch.Tensor,
        residual_inp: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        use_1stage: bool,
    ) -> Optional[torch.Tensor]:
        # when custom allreduce is disabled, this will be None
        if self.disabled or not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.fused_ar_rms(
                    input,
                    residual_inp,
                    w=weight,
                    eps=eps,
                    registered=True,
                    use_1stage=use_1stage,
                )
            else:
                return torch.zeros_like(input), torch.zeros_like(input)
        else:
            return self.fused_ar_rms(
                input,
                residual_inp,
                w=weight,
                eps=eps,
                registered=False,
                use_1stage=use_1stage,
            )

    def custom_fused_ar_rms_quant(
        self,
        input: torch.Tensor,
        residual_inp: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        use_1stage: bool,
    ):
        # when custom allreduce is disabled, this will be None
        if self.disabled or not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.fused_ar_rms(
                    input,
                    residual_inp,
                    w=weight,
                    eps=eps,
                    registered=True,
                    use_1stage=use_1stage,
                    post_per_token_quant=True,
                )
            else:
                dummy_out = torch.zeros(input.shape, dtype=fp8, device=input.device)
                dummy_scale_out = torch.zeros(
                    input.shape[:-1] + (1,), dtype=torch.float32, device=input.device
                )
                return dummy_out, torch.zeros_like(input), dummy_scale_out
        else:
            return self.fused_ar_rms(
                input,
                residual_inp,
                w=weight,
                eps=eps,
                registered=False,
                use_1stage=use_1stage,
                post_per_token_quant=True,
            )

    def close(self):
        if not self.disabled and self._ptr:
            ops.dispose(self._ptr)
            self._ptr = 0

    def __del__(self):
        self.close()
