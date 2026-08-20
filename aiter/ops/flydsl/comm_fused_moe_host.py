# SPDX-License-Identifier: MIT
"""FlyDSL host runner for communication-fused MoE."""

import torch
import torch.distributed._symmetric_memory as symm_mem
import flydsl.expr as fx
from mori.cco import Communicator

from aiter.ops.flydsl.kernels.comm_fused_moe.full_width import (
    FLAT_VA_RANK_STRIDE,
    compile_epoch_barrier,
    compile_stage2_compute,
    compile_stage2_fanout,
    compile_stage2_local_reduce,
    compile_stage2_peer_reduce,
)
from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg
from aiter.ops.flydsl.moe_kernels import _run_compiled


M = 2048
H = 7168
I = 384
E = 384
TOPK = 6
TP = 8
OWNER_ROWS = M // TP

_WORKSPACE = None


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class _Workspace:
    def __init__(self, tp_group) -> None:
        self.rank = int(tp_group.rank_in_group)
        self.device = torch.device(tp_group.device)

        route_bytes = H + H // 8
        self.partial_ready = _align(M * (H + H // 32), 8)
        partial_bytes = _align(self.partial_ready + 8, 256)
        owner_payload = OWNER_ROWS * H
        self.owner_ready = _align(owner_payload, 8)
        owner_bytes = _align(self.owner_ready + 8, 256)

        self.route = torch.empty(
            (M, TOPK, route_bytes), dtype=torch.uint8, device=self.device
        )
        self.partial = self._symmetric((partial_bytes,))
        self.owner_workspace = self._symmetric((owner_bytes,))
        self.owner_payload = self.owner_workspace[:owner_payload].view(OWNER_ROWS, H)
        self.owner_scale = self._symmetric((OWNER_ROWS, H // 32))
        self.output = torch.empty((M, H), dtype=torch.bfloat16, device=self.device)
        self.shared = torch.empty_like(self.output)
        self.empty_bias = torch.empty(0, dtype=torch.float32, device=self.device)

        uid = Communicator.get_unique_id() if self.rank == 0 else None
        self.comm = Communicator.init(
            TP,
            self.rank,
            tp_group.broadcast_object(uid),
            per_rank_vmm=FLAT_VA_RANK_STRIDE,
        )
        self.partial_window = self.comm.register_external_window(
            self.partial.data_ptr(), self.partial.nbytes
        )
        self.owner_window = self.comm.register_external_window(
            self.owner_workspace.data_ptr(), self.owner_workspace.nbytes
        )
        self.scale_window = self.comm.register_external_window(
            self.owner_scale.data_ptr(), self.owner_scale.nbytes
        )
        self.partial_flat_base = (
            self.partial_window.local_ptr - self.rank * FLAT_VA_RANK_STRIDE
        )
        self.owner_flat_base = (
            self.owner_window.local_ptr - self.rank * FLAT_VA_RANK_STRIDE
        )
        self.scale_flat_base = (
            self.scale_window.local_ptr - self.rank * FLAT_VA_RANK_STRIDE
        )
        owner_begin = self.rank * OWNER_ROWS
        self.owner_output = self.output[owner_begin : owner_begin + OWNER_ROWS]

    def shared_output_buffer(self) -> torch.Tensor:
        return self.shared

    def _symmetric(self, shape) -> torch.Tensor:
        tensor = symm_mem.empty(shape, dtype=torch.uint8, device=self.device)
        tensor.zero_()
        return tensor

    def __call__(self, *, stage2_args, stage2_kwargs, shared_partial):
        stream = torch.cuda.current_stream(self.device)
        inter_states, w2 = stage2_args[0], stage2_args[2]
        sorted_token_ids, sorted_expert_ids, num_valid_ids = stage2_args[3:6]
        common = (
            ptr_arg(inter_states),
            ptr_arg(w2),
            ptr_arg(stage2_kwargs["a2_scale"].view(-1)),
            ptr_arg(stage2_kwargs["w2_scale"].view(-1)),
            ptr_arg(sorted_token_ids),
            ptr_arg(sorted_expert_ids),
            ptr_arg(stage2_kwargs["sorted_weights"]),
            ptr_arg(num_valid_ids),
            ptr_arg(self.empty_bias),
            M,
            H,
            I,
            int(sorted_expert_ids.shape[0]) * 2,
        )

        _run_compiled(
            compile_stage2_compute(),
            (ptr_arg(self.route), *common, stream),
        )
        _run_compiled(
            compile_stage2_local_reduce(),
            (
                ptr_arg(self.route),
                ptr_arg(self.partial),
                ptr_arg(shared_partial),
                stream,
            ),
        )
        self._barrier(
            self.partial, self.partial_flat_base, self.partial_ready, stream
        )
        _run_compiled(
            compile_stage2_peer_reduce(),
            (
                fx.Int64(self.partial_flat_base),
                ptr_arg(self.owner_output),
                ptr_arg(self.owner_payload),
                ptr_arg(self.owner_scale),
                self.rank,
                stream,
            ),
        )
        self._barrier(
            self.owner_workspace, self.owner_flat_base, self.owner_ready, stream
        )
        _run_compiled(
            compile_stage2_fanout(),
            (
                fx.Int64(self.owner_flat_base),
                fx.Int64(self.scale_flat_base),
                ptr_arg(self.output),
                self.rank,
                stream,
            ),
        )
        return self.output

    @staticmethod
    def _barrier(workspace, flat_base, ready_offset, stream) -> None:
        _run_compiled(
            compile_epoch_barrier(),
            (
                ptr_arg(workspace),
                fx.Int64(flat_base),
                fx.Int64(ready_offset),
                stream,
            ),
        )


def create_flydsl_comm_fused_runners(*, tp_group, model_dim, inter_dim, experts, topk):
    shape = (model_dim, inter_dim, experts, topk, int(tp_group.world_size))
    if shape != (H, I, E, TOPK, TP):
        raise KeyError(f"unsupported comm_fused shape {shape}")
    global _WORKSPACE
    if _WORKSPACE is None:
        _WORKSPACE = _Workspace(tp_group)
    return {M: _WORKSPACE}
