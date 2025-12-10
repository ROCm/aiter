# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
import torch.distributed as dist
from torch.distributed import ProcessGroup
from contextlib import contextmanager
from ..jit.core import compile_ops
from ..utility.dtypes import fp8


fp8_max_val_ = {
    torch.float8_e4m3fn: 240,
    torch.float8_e4m3fnuz: 120,
}
fp8_max_val = fp8_max_val_[fp8]
fp8_policy_id_ = {
    torch.float8_e4m3fn: 1,
    torch.float8_e4m3fnuz: 2,
}
fp8_policy_id = fp8_policy_id_[fp8]


@compile_ops("module_trtllm_all_reduce_fusion")
def trtllm_init_ar_fusion(
    device_id: int,
    rank: int,
    world_size: int,
    max_size_in_bytes: int,
    comm_ptrs_buf_len: int,
) -> int: ...


@compile_ops("module_trtllm_all_reduce_fusion")
def trtllm_destroy_ar_fusion(
    fptr_t: int,
) -> None: ...


@compile_ops("module_trtllm_all_reduce_fusion")
def trtllm_get_ar_fusion_barrier_handle(
    fptr_t: int,
) -> Tensor: ...


@compile_ops("module_trtllm_all_reduce_fusion")
def trtllm_get_ar_fusion_data_handle(
    fptr_t: int,
) -> Tensor: ...


@compile_ops("module_trtllm_all_reduce_fusion")
def trtllm_open_ar_fusion_barrier_handles(
    fptr_t: int,
    handles: list[Tensor],
) -> None: ...


@compile_ops("module_trtllm_all_reduce_fusion")
def trtllm_open_ar_fusion_data_handles(
    fptr_t: int,
    handles: list[Tensor],
) -> None: ...


@compile_ops("module_trtllm_all_reduce_fusion")
def trtllm_ar_fusion_capture_clear(
    fptr_t: int,
) -> None: ...


@compile_ops("module_trtllm_all_reduce_fusion")
def trtllm_open_ar_fusion_captured_handles(
    fptr_t: int,
    handles: list[Tensor],
    offsets: list[int],
    ptr_idx: int,
) -> None: ...


@compile_ops("module_trtllm_all_reduce_fusion")
def trtllm_get_ar_fusion_captured_handles(
    fptr_t: int,
) -> list[Tensor]: ...


@compile_ops("module_trtllm_all_reduce_fusion")
def trtllm_get_ar_fusion_captured_offsets(
    fptr_t: int,
) -> Tensor: ...


@compile_ops("module_trtllm_all_reduce_fusion")
def trtllm_allreduce_rms(
    fptr_t: int,
    allreduce_in: Tensor,
    residual_in: Tensor,
    rms_gamma: Tensor,
    residual_out: Tensor,
    norm_out: Tensor,
    scale_out: Tensor,
    eps: float,
    quant_type: int,
) -> None: ...


class AiterDistEnv:
    _SUPPORTED_WORLD_SIZES = [2, 4, 8]

    def __init__(
        self,
        group: ProcessGroup = None,
        device_id: int = None,
        max_size_in_bytes=16384 * 16384,
        comm_ptrs_buf_len=1024 * 256,
        dtype: torch.dtype=torch.bfloat16,
    ) -> None:
        self.group = group
        self.device_id = device_id
        self.rank = dist.get_rank(group=self.group)
        self.world_size = dist.get_world_size(group=self.group)
        self.fptr = None
        torch.cuda.set_device(self.device_id)

        if self.world_size == 1:
            return

        if self.world_size not in AiterDistEnv._SUPPORTED_WORLD_SIZES:
            return

        self.fptr = trtllm_init_ar_fusion(self.device_id, self.rank, self.world_size, max_size_in_bytes, comm_ptrs_buf_len)
        barrier_handle = trtllm_get_ar_fusion_barrier_handle(self.fptr)
        data_handle = trtllm_get_ar_fusion_data_handle(self.fptr)
        self.barrier()
        barrier_handle_list = [None] * self.world_size
        data_handle_list = [None] * self.world_size
        dist.all_gather_object(barrier_handle_list, barrier_handle, group=self.group)
        dist.all_gather_object(data_handle_list, data_handle, group=self.group)
        trtllm_open_ar_fusion_barrier_handles(self.fptr, barrier_handle_list)
        trtllm_open_ar_fusion_data_handles(self.fptr, data_handle_list)
        self.barrier()
        self._IS_CAPTURING = False
        self._IS_CAPTURED = False
        self.disabled = False

    def barrier(self):
        torch.cuda.set_device(self.device_id)
        torch.cuda.synchronize(self.device_id)
        dist.barrier(group=self.group)

    def consume_capture(self):
        self.barrier()
        handles = trtllm_get_ar_fusion_captured_handles(self.fptr)
        offsets = trtllm_get_ar_fusion_captured_offsets(self.fptr)
        for idx in range(len(handles)):
            handle_list = [None] * self.world_size
            offset_list = [None] * self.world_size
            dist.all_gather_object(handle_list, handles[idx], group=self.group)
            dist.all_gather_object(offset_list, int(offsets[idx].item()), group=self.group)
            self.barrier()
            trtllm_open_ar_fusion_captured_handles(self.fptr, handle_list, offset_list, idx)
        trtllm_ar_fusion_capture_clear(self.fptr)
        self.barrier()

    @contextmanager
    def capture(self):
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            if not self.disabled:
                self.consume_capture()

    def capture_(self, input: torch.Tensor):
        if torch.cuda.is_current_stream_capturing():
            pass
            self._IS_CAPTURED = True
        else:
            if self._IS_CAPTURED:
                self.consume_capture()
                self._IS_CAPTURED = False

    def __del__(self):
        if self.fptr:
            trtllm_destroy_ar_fusion(self.fptr)

    def allreduce_add_rms_native(
        self, allreduce_in, residual_in, rms_weight, eps, fp8_out=False
    ):
        def rms_norm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float):
            input_dtype = x.dtype
            variance = x.float().pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + eps)
            x = x.to(input_dtype)
            return weight * x
        dist.all_reduce(allreduce_in, group=self.group)
        residual_out = allreduce_in + residual_in
        norm_out = rms_norm_forward(residual_out, rms_weight, eps)
        if fp8_out:
            norm_out_scale, _ = norm_out.float().abs().max(dim=-1, keepdim=True)
            norm_out_scale = norm_out_scale / fp8_max_val
            norm_out = norm_out / norm_out_scale
            norm_out.clamp_(min=-fp8_max_val, max=fp8_max_val)
            norm_out = norm_out.to(fp8)
            return residual_out, norm_out, norm_out_scale
        else:
            scale_out = torch.empty(
                allreduce_in.shape[0],
                1,
                dtype=torch.float32,
                device=allreduce_in.device,
            )
            return residual_out, norm_out, scale_out

    def allreduce_add_rms_fused(
        self, allreduce_in, residual_in, rms_weight, eps, fp8_out=False
    ):
        self.capture_(allreduce_in)
        residual_out = torch.empty_like(residual_in)
        if fp8_out:
            norm_out = torch.empty_like(allreduce_in, dtype=fp8)
            scale_out = torch.empty(
                allreduce_in.shape[0],
                1,
                dtype=torch.float32,
                device=allreduce_in.device,
            )
        else:
            norm_out = torch.empty_like(allreduce_in)
            scale_out = torch.empty(1, dtype=torch.float32, device=allreduce_in.device)
        trtllm_allreduce_rms(
            self.fptr,
            allreduce_in,
            residual_in,
            rms_weight,
            residual_out,
            norm_out,
            scale_out,
            eps,
            fp8_policy_id if fp8_out else 0,
        )
        if fp8_out:
            return residual_out, norm_out, scale_out
        else:
            return residual_out, norm_out, scale_out
