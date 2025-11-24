# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
import torch.distributed as dist
from torch.distributed import ProcessGroup
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
    rank: int,
    world_size: int,
    max_size_in_bytes: int,
) -> int: ...


@compile_ops("module_trtllm_all_reduce_fusion")
def trtllm_destroy_ar_fusion(
    fptr_t: int,
) -> None: ...


@compile_ops("module_trtllm_all_reduce_fusion")
def trtllm_get_ar_fusion_handle(
    fptr_t: int,
) -> Tensor: ...


@compile_ops("module_trtllm_all_reduce_fusion")
def trtllm_open_ar_fusion_handles(
    fptr_t: int,
    handles: list[Tensor],
) -> None: ...


@compile_ops("module_trtllm_all_reduce_fusion")
def trtllm_get_ar_fusion_workspace(
    fptr_t: int,
    ref: Tensor,
) -> Tensor: ...


@compile_ops("module_trtllm_all_reduce_fusion")
def trtllm_allreduce_rms(
    rank: int,
    nranks: int,
    allreduce_in: Tensor,
    residual_in: Tensor,
    rms_gamma: Tensor,
    residual_out: Tensor,
    norm_out: Tensor,
    scale_out: Tensor,
    eps: float,
    quant_type: int,
    workspace: Tensor,
) -> None: ...


class _ARFusion:
    _SUPPORTED_WORLD_SIZES = [2, 4, 8]

    def __init__(
        self,
        group: ProcessGroup = None,
        max_size_in_bytes=16384 * 16384,
    ) -> None:
        self.group = group
        rank = dist.get_rank(group=self.group)
        torch.cuda.set_device(rank)
        self.rank = rank
        self.fptr = None
        world_size = dist.get_world_size(group=self.group)
        if world_size == 1:
            return

        if world_size not in _ARFusion._SUPPORTED_WORLD_SIZES:
            return

        torch.cuda.set_device(rank)
        self.fptr = trtllm_init_ar_fusion(rank, world_size, max_size_in_bytes)
        handle = trtllm_get_ar_fusion_handle(self.fptr)
        handle_list = [None] * world_size
        dist.all_gather_object(handle_list, handle, group=self.group)
        trtllm_open_ar_fusion_handles(self.fptr, handle_list)
        torch.cuda.synchronize(rank)
        dist.barrier(group=group)

    def get_workspace(self, ref: torch.Tensor):
        return trtllm_get_ar_fusion_workspace(self.fptr, ref)

    def __del__(self):
        if self.fptr:
            trtllm_destroy_ar_fusion(self.fptr)


class TRTLLMDistEnv:
    def __init__(self, rank, world_size, init_process_group=False, port=22339):
        torch.cuda.set_device(rank)
        if init_process_group:
            dist.init_process_group(
                backend="nccl",
                init_method=f"tcp://127.0.0.1:{port}",
                rank=rank,
                world_size=world_size,
            )
        self.rank = rank
        self.world_size = world_size
        self.group = dist.group.WORLD
        self.ar_fusion = _ARFusion(group=self.group)
        self.barrier()

    def __del__(self):
        if getattr(self, 'group', None):
            dist.destroy_process_group(self.group)
        else:
            dist.destroy_process_group(None)

    def barrier(self):
        torch.cuda.set_device(self.rank)
        dist.barrier(self.group)
        torch.cuda.synchronize()

    def allreduce_add_rms_native(
        self, allreduce_in, residual_in, rms_weight, eps, fp8_out=False
    ):
        def rms_norm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float):
            input_dtype = x.dtype
            variance = x.float().pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + eps)
            x = x.to(input_dtype)
            return weight * x
        dist.all_reduce(allreduce_in)
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
        residual_out = torch.empty_like(residual_in)
        norm_out = torch.empty_like(allreduce_in)
        if fp8_out:
            norm_out = norm_out.to(fp8)
            scale_out = torch.empty(
                allreduce_in.shape[0],
                1,
                dtype=torch.float32,
                device=allreduce_in.device,
            )
        else:
            scale_out = torch.empty(1, dtype=torch.float32, device=allreduce_in.device)
        trtllm_allreduce_rms(
            self.rank,
            self.world_size,
            allreduce_in,
            residual_in,
            rms_weight,
            residual_out,
            norm_out,
            scale_out,
            eps,
            fp8_policy_id if fp8_out else 0,
            self.ar_fusion.get_workspace(allreduce_in),
        )
        if fp8_out:
            return residual_out, norm_out, scale_out
        else:
            return residual_out, norm_out, scale_out
