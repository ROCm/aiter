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

from contextlib import contextmanager
from typing import Any, List, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

# import vllm.envs as envs
# from vllm import _custom_ops as ops
import aiter as ops
from aiter.dist.parallel_state import in_the_same_node_as
from aiter import logger
from aiter.utility.dtypes import fp8

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


class _GpuPtrBuffer:
    """Wraps a raw GPU pointer as a buffer with __cuda_array_interface__
    so that torch.as_tensor can create a tensor view without copying."""

    def __init__(self, ptr: int, nbytes: int):
        self.ptr = ptr
        self.nbytes = nbytes
        self.__cuda_array_interface__ = {
            "data": (ptr, False),
            "shape": (nbytes,),
            "typestr": "<u1",
            "strides": None,
            "version": 3,
        }


class CustomAllreduce:

    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]

    SHMEM_INPUT_BUFFER_SIZE = int(1.2 * 1024**3)  # 1.2 GB
    SHMEM_OUTPUT_BUFFER_SIZE = int(1.2 * 1024**3)  # 1.2 GB
    SHMEM_TMP_BUFFER_SIZE = int(2.4 * 1024**3)  # 2.4 GB
    _IPC_FALLBACK_MAX_SIZE = 8192 * 1024 * 8 * 2  # ~128 MB, IPC fallback only

    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
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

        Buffer allocation is automatic:
        - Primary path uses MoRI SHMEM symmetric memory (1.2 GB input/output,
          2.4 GB tmp) with no IPC handles needed for static buffers.
        - Falls back to IPC-based buffers if MoRI SHMEM is unavailable.
        - Eager mode: input address varies each call, so a d2d copy to the
          pre-registered static buffer is required before the allreduce kernel.
        - Graph mode: the input address is captured once and replayed at a
          fixed address, so the kernel operates directly on the captured input
          with zero d2d copy.  Graph-captured addresses are registered via IPC
          at the end of capture (they are not SHMEM allocations).
        """
        self._IS_CAPTURING = False
        self.disabled = True
        self._use_shmem = False
        self._shmem_ptrs = []

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

        fully_connected = True
        if world_size > 2 and not fully_connected:
            logger.warning(
                "Custom allreduce is disabled because it's not supported on"
                " more than two PCIe-only GPUs. To silence this warning, "
                "specify disable_custom_all_reduce=True explicitly."
            )
            return

        self.disabled = False
        self.rank = rank
        self.world_size = world_size
        self.fully_connected = fully_connected

        try:
            self._init_with_shmem(rank, world_size, fully_connected)
        except Exception as e:
            print(f"MoRI SHMEM unavailable ({e}), falling back to IPC path.")
            self._init_with_ipc(rank, world_size, fully_connected)

    def _init_with_shmem(self, rank, world_size, fully_connected):
        """Initialize using MoRI SHMEM symmetric memory (no IPC handles)."""
        import mori.shmem as shmem

        self._use_shmem = True
        self.enable_register_for_capturing = True

        # Ensure the SHMEM static heap is large enough for our buffers.
        # Total: meta(signal + tmp) + input + output ≈ 4.8 GB
        import os

        required = (
            ops.meta_size()
            + self.SHMEM_TMP_BUFFER_SIZE
            + self.SHMEM_INPUT_BUFFER_SIZE
            + self.SHMEM_OUTPUT_BUFFER_SIZE
        )
        # 10 % headroom for alignment
        required = int(required * 1.1)
        if "MORI_SHMEM_HEAP_SIZE" not in os.environ:
            os.environ["MORI_SHMEM_HEAP_SIZE"] = str(required)

        # Initialize MoRI SHMEM: bootstrap via UniqueId broadcast
        if rank == 0:
            uid_list = [shmem.shmem_get_unique_id()]
        else:
            uid_list = [None]
        dist.broadcast_object_list(uid_list, src=0, group=self.group)
        shmem.shmem_init_attr(
            shmem.MORI_SHMEM_INIT_WITH_UNIQUEID,
            rank,
            world_size,
            uid_list[0],
        )

        meta_signal_size = ops.meta_size()
        meta_total = meta_signal_size + self.SHMEM_TMP_BUFFER_SIZE

        # Allocate symmetric memory via MoRI SHMEM
        meta_ptr = shmem.shmem_malloc(meta_total)
        input_ptr = shmem.shmem_malloc(self.SHMEM_INPUT_BUFFER_SIZE)
        output_ptr = shmem.shmem_malloc(self.SHMEM_OUTPUT_BUFFER_SIZE)
        assert meta_ptr != 0, "shmem_malloc failed for meta buffer"
        assert input_ptr != 0, "shmem_malloc failed for input buffer"
        assert output_ptr != 0, "shmem_malloc failed for output buffer"
        self._shmem_ptrs = [meta_ptr, input_ptr, output_ptr]

        # Zero-initialize the Signal region (synchronization flags must start at 0)
        meta_view = torch.as_tensor(_GpuPtrBuffer(meta_ptr, meta_total), device="cuda")
        meta_view[:meta_signal_size].zero_()
        torch.cuda.synchronize(self.device)

        # Wrap SHMEM pointers as torch tensors for the existing C++ eager-mode
        # copy path (hipMemcpyAsync to/from registered buffers)
        self.meta = meta_view
        self.input_buffer = torch.as_tensor(
            _GpuPtrBuffer(input_ptr, self.SHMEM_INPUT_BUFFER_SIZE), device="cuda"
        )
        self.output_buffer = torch.as_tensor(
            _GpuPtrBuffer(output_ptr, self.SHMEM_OUTPUT_BUFFER_SIZE), device="cuda"
        )

        # Rank-data: GPU buffer for per-rank pointer tuples used by kernels
        self.rank_data = torch.empty(
            8 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )

        # Resolve P2P peer addresses for every buffer
        my_pe = shmem.shmem_mype()
        meta_peers = [
            shmem.shmem_ptr_p2p(meta_ptr, my_pe, pe) for pe in range(world_size)
        ]
        input_peers = [
            shmem.shmem_ptr_p2p(input_ptr, my_pe, pe) for pe in range(world_size)
        ]
        output_peers = [
            shmem.shmem_ptr_p2p(output_ptr, my_pe, pe) for pe in range(world_size)
        ]

        # Initialize C++ CustomAllreduce with pre-resolved peer pointers
        self._ptr = ops.init_custom_ar_with_peer_ptrs(
            meta_ptr,
            self.rank_data,
            meta_peers,
            rank,
            world_size,
            fully_connected,
        )

        # Register input/output buffers with pre-resolved peer pointers
        ops.register_input_buffer_with_peer_ptrs(
            self._ptr,
            input_ptr,
            input_peers,
        )
        ops.register_output_buffer_with_peer_ptrs(
            self._ptr,
            output_ptr,
            output_peers,
        )

        logger.info(
            "CustomAllreduce initialized with MoRI SHMEM: "
            "input=%.1fGB, output=%.1fGB, tmp=%.1fGB",
            self.SHMEM_INPUT_BUFFER_SIZE / 1024**3,
            self.SHMEM_OUTPUT_BUFFER_SIZE / 1024**3,
            self.SHMEM_TMP_BUFFER_SIZE / 1024**3,
        )

    def _init_with_ipc(self, rank, world_size, fully_connected):
        """Fallback: initialize using the original IPC handle exchange path."""
        self.enable_register_for_capturing = True
        max_size = self._IPC_FALLBACK_MAX_SIZE
        self.meta = ops.allocate_meta_buffer(ops.meta_size() + max_size)
        self.input_buffer = torch.empty(max_size, dtype=torch.uint8, device=self.device)
        self.output_buffer = torch.empty(
            max_size, dtype=torch.uint8, device=self.device
        )
        # This is a buffer for storing the tuples of pointers pointing to
        # IPC buffers from all ranks. Each registered tuple has size of
        # 8*world_size bytes where world_size is at most 8. Allocating 8MB
        # is enough for 131072 such tuples. The largest model I've seen only
        # needs less than 10000 of registered tuples.
        self.rank_data = torch.empty(
            8 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        # max_size = max input bytes the allreduce can handle.
        # In 2-stage write mode, tmp needs 2x, and input/tmp share the same
        # raw size in the IPC path, so effective max is raw_size // 2.
        self.max_size = max_size // 2
        handle = ops.get_meta_buffer_ipc_handle(self.meta)
        shard_data = (
            handle,  # ipc handle to base ptr
            0,  # offset of base ptr
        )
        handles, offsets = self._gather_ipc_meta(shard_data)

        self.fully_connected = fully_connected
        self._ptr = ops.init_custom_ar(
            self.meta, self.rank_data, handles, offsets, rank, self.fully_connected
        )
        self._register_input_buffer_ipc(self.input_buffer)
        self._register_output_buffer_ipc(self.output_buffer)

    @contextmanager
    def capture(self):
        """Context manager for CUDA graph capture.

        During capture the kernel operates directly on the captured input
        address (no d2d copy).  After capture completes, all captured
        addresses are registered via IPC so that peer GPUs can access them
        during graph replay.
        """
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            if not self.disabled:
                self._register_graph_buffers()

    # ---- IPC helpers ----
    # Used by _init_with_ipc for static buffers AND by _register_graph_buffers
    # for graph-captured addresses (which are regular PyTorch allocations, not
    # SHMEM, so IPC is still needed for P2P address resolution).

    def _get_ipc_meta(self, inp: torch.Tensor):
        handle = ops.get_meta_buffer_ipc_handle(inp)
        shard_data = (handle, 0)
        return self._gather_ipc_meta(shard_data)

    def _gather_ipc_meta(self, shard_data):
        all_data: List[Optional[Any]] = [[None] for i in range(self.world_size)]
        all_data[self.rank][0] = shard_data

        ranks = dist.get_process_group_ranks(group=self.group)
        ranks.sort()
        for i, rank in enumerate(ranks):
            dist.broadcast_object_list(
                all_data[i], src=rank, group=self.group, device="cpu"
            )

        handles = []
        offsets = []
        for i in range(len(all_data)):
            handles.append(all_data[i][0][0])  # type: ignore
            offsets.append(all_data[i][0][1])  # type: ignore
        return handles, offsets

    def _register_input_buffer_ipc(self, inp: torch.Tensor):
        handles, offsets = self._get_ipc_meta(inp)
        ops.register_input_buffer(self._ptr, inp, handles, offsets)

    def _register_output_buffer_ipc(self, out: torch.Tensor):
        handles, offsets = self._get_ipc_meta(out)
        ops.register_output_buffer(self._ptr, out, handles, offsets)

    def _register_graph_buffers(self):
        """Register graph-captured addresses via IPC after capture ends.

        Graph-captured tensor addresses come from PyTorch's caching allocator
        (not SHMEM), so IPC handles are still needed for cross-GPU access.
        """
        handle, offset = ops.get_graph_buffer_ipc_meta(self._ptr)
        handles, offsets = self._gather_ipc_meta((handle, offset))
        logger.info("Registering %d cuda graph addresses", len(offset))
        ops.register_graph_buffers(self._ptr, handles, offsets)

    def _max_input_bytes(self) -> int:
        """Max input bytes the custom allreduce can handle."""
        if self._use_shmem:
            return self.SHMEM_INPUT_BUFFER_SIZE
        return self.max_size

    def should_custom_ar(self, inp: torch.Tensor):
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        if self.world_size == 2 or self.fully_connected:
            return inp_size <= self._max_input_bytes()
        return False

    def should_custom_ag(self, inp: torch.Tensor):
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        # all_gather output = input * world_size
        if self.world_size == 2 or self.fully_connected:
            return inp_size <= (self._max_input_bytes() / self.world_size)
        return False

    def all_reduce(
        self,
        inp: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        use_new: bool = True,
        open_fp8_quant: bool = False,
        registered: bool = False,
    ):
        """Performs an out-of-place all reduce.

        If registered is False (eager mode), inp is first d2d-copied into the
        pre-registered static buffer before allreduce, and the result is copied
        back to *out*.
        If registered is True (graph mode), the kernel operates directly on
        *inp* whose address was captured and later IPC-registered, so no d2d
        copy takes place.
        """
        if out is None:
            out = torch.empty_like(inp)
        ops.all_reduce(
            self._ptr,
            inp,
            out,
            use_new,
            open_fp8_quant,
            None if registered else self.input_buffer,
            None if registered else self.output_buffer,
        )
        return out

    def custom_all_reduce(
        self, input: torch.Tensor, use_new: bool = True, open_fp8_quant: bool = False
    ) -> Optional[torch.Tensor]:
        if self.disabled or not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                # Graph capture: kernel uses the captured input address directly
                return self.all_reduce(
                    input,
                    use_new=use_new,
                    open_fp8_quant=open_fp8_quant,
                    registered=True,
                )
            else:
                # Warmup: mimic the allocation pattern (out-of-place)
                return torch.zeros_like(input)
        # Eager: d2d copy to static buffer → allreduce → copy back
        return self.all_reduce(input, use_new=use_new, open_fp8_quant=open_fp8_quant)

    def reduce_scatter(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        *,
        registered: bool = False,
    ):
        ops.reduce_scatter(
            self._ptr,
            inp,
            out,
            None if registered else self.input_buffer,
        )

    def custom_reduce_scatter(
        self, input: torch.Tensor, output: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if self.disabled or not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.reduce_scatter(input, output, registered=True)
            return None
        return self.reduce_scatter(input, output)

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
        ops.all_gather_reg(self._ptr, inp, out, inp.shape[-1], dim)
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
        ops.all_gather_unreg(self._ptr, inp, self.input_buffer, out, inp.shape[-1], dim)
        return out

    def custom_all_gather(
        self, inp: torch.Tensor, dim: int = 0
    ) -> Optional[torch.Tensor]:
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.all_gather_reg(inp, dim=dim)
            return torch.zeros_like(inp)
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
        if not post_per_token_quant:
            if out is None:
                out = torch.empty_like(inp)
            ops.fused_allreduce_rmsnorm(
                self._ptr,
                inp,
                res_inp,
                res_out,
                out,
                w,
                eps,
                None if registered else self.input_buffer,
                use_1stage,
            )
            return out, res_out
        else:
            if out is None:
                out = torch.empty(inp.shape, dtype=fp8, device=inp.device)
            if scale_out is None:
                scale_out = torch.empty(
                    inp.shape[:-1] + (1,), dtype=torch.float32, device=inp.device
                )
            ops.fused_allreduce_rmsnorm_quant(
                self._ptr,
                inp,
                res_inp,
                res_out,
                out,
                scale_out,
                w,
                eps,
                None if registered else self.input_buffer,
                use_1stage,
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
            return torch.zeros_like(input), torch.zeros_like(input)
        return self.fused_ar_rms(
            input,
            residual_inp,
            w=weight,
            eps=eps,
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
            dummy_out = torch.zeros(input.shape, dtype=fp8, device=input.device)
            dummy_scale_out = torch.zeros(
                input.shape[:-1] + (1,), dtype=torch.float32, device=input.device
            )
            return dummy_out, torch.zeros_like(input), dummy_scale_out
        return self.fused_ar_rms(
            input,
            residual_inp,
            w=weight,
            eps=eps,
            use_1stage=use_1stage,
            post_per_token_quant=True,
        )

    def close(self):
        if not self.disabled and self._ptr:
            ops.dispose(self._ptr)
            self._ptr = 0
        if self._shmem_ptrs:
            try:
                import mori.shmem as shmem

                for ptr in self._shmem_ptrs:
                    if ptr:
                        shmem.shmem_free(ptr)
                shmem.shmem_finalize()
            except Exception:
                pass
            self._shmem_ptrs.clear()

    def __del__(self):
        self.close()
