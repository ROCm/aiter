# SPDX-License-Identifier: MIT
"""FlyDSL host runner for communication-fused MoE."""

import torch
import torch.distributed._symmetric_memory as symm_mem
import flydsl.expr as fx
from mori.cco import Communicator

from aiter.ops.flydsl.kernels.comm_fused_moe import full_width as full_width_kernels
from aiter.ops.flydsl.kernels.comm_fused_moe import persistent as persistent_kernels
from aiter.ops.flydsl.kernels.comm_fused_moe import windowed as windowed_kernels
from aiter.ops.flydsl.kernels.comm_fused_moe.sync import (
    FLAT_VA_RANK_STRIDE,
    compile_epoch_barrier,
)
from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg
from aiter.ops.flydsl.moe_kernels import _run_compiled


_RUNNERS = None


def _symmetric(device, shape) -> torch.Tensor:
    return symm_mem.empty(shape, dtype=torch.uint8, device=device)


def _workspace(device, payload_bytes: int) -> torch.Tensor:
    tensor = _symmetric(device, ((payload_bytes + 8 + 255) // 256 * 256,))
    tensor[payload_bytes : payload_bytes + 8].zero_()
    return tensor


def _register(tp_group, rank: int, tp: int, tensors):
    uid = Communicator.get_unique_id() if rank == 0 else None
    comm = Communicator.init(
        tp, rank, tp_group.broadcast_object(uid), per_rank_vmm=FLAT_VA_RANK_STRIDE
    )
    windows = tuple(
        comm.register_external_window(tensor.data_ptr(), tensor.nbytes)
        for tensor in tensors
    )
    bases = tuple(w.local_ptr - rank * FLAT_VA_RANK_STRIDE for w in windows)
    return comm, windows, bases


def _barrier(workspace, flat_base, ready_offset, stream) -> None:
    args = (
        ptr_arg(workspace), fx.Int64(flat_base), fx.Int64(ready_offset), stream
    )
    _run_compiled(compile_epoch_barrier(), args)


def _stage2_args(stage2_args, stage2_kwargs, kernels):
    inter_states, w2 = stage2_args[0], stage2_args[2]
    sorted_token_ids, sorted_expert_ids, num_valid_ids = stage2_args[3:6]
    return (
        ptr_arg(inter_states),
        ptr_arg(w2),
        ptr_arg(stage2_kwargs["a2_scale"].view(-1)),
        ptr_arg(stage2_kwargs["w2_scale"].view(-1)),
        ptr_arg(sorted_token_ids),
        ptr_arg(sorted_expert_ids),
        ptr_arg(stage2_kwargs["sorted_weights"]),
        ptr_arg(num_valid_ids),
        ptr_arg(inter_states),
        kernels.M,
        kernels.H,
        kernels.I,
        int(sorted_expert_ids.shape[0]) * kernels.SORT_BLOCK_M // kernels.TILE_M,
    )


class _FullWidthRunner:
    def __init__(self, tp_group) -> None:
        kernels = full_width_kernels
        self.rank = int(tp_group.rank_in_group)
        self.device = torch.device(tp_group.device)
        self.route = torch.empty(
            (kernels.M, kernels.TOPK, kernels.H + kernels.H // 8),
            dtype=torch.uint8,
            device=self.device,
        )
        self.partial_ready = kernels.M * (kernels.H + kernels.H // 32)
        self.partial = _workspace(self.device, self.partial_ready)
        self.reduced_ready = kernels.SHARD_ROWS * kernels.H
        self.reduced_payload = _workspace(self.device, self.reduced_ready)
        self.reduced_scale = _symmetric(
            self.device, (kernels.SHARD_ROWS, kernels.H // 32)
        )
        self.output = torch.empty(
            (kernels.M, kernels.H), dtype=torch.bfloat16, device=self.device
        )

        tensors = self.partial, self.reduced_payload, self.reduced_scale
        self.comm, self.windows, bases = _register(
            tp_group, self.rank, kernels.TP, tensors
        )
        (
            self.partial_flat_base,
            self.reduced_payload_base,
            self.reduced_scale_base,
        ) = bases
        shard_begin = self.rank * kernels.SHARD_ROWS
        self.reduced_shard = self.output[
            shard_begin : shard_begin + kernels.SHARD_ROWS
        ]

    def __call__(self, *, stage2_args, stage2_kwargs, shared_partial):
        k = full_width_kernels
        stream = torch.cuda.current_stream(self.device)
        common = _stage2_args(stage2_args, stage2_kwargs, k)
        _run_compiled(k.compile_stage2_compute(), (ptr_arg(self.route), *common, stream))
        _run_compiled(
            k.compile_stage2_local_reduce(),
            (ptr_arg(self.route), ptr_arg(self.partial), ptr_arg(shared_partial), stream),
        )
        _barrier(self.partial, self.partial_flat_base, self.partial_ready, stream)
        _run_compiled(
            k.compile_stage2_tp_reduce_scatter(),
            (
                fx.Int64(self.partial_flat_base),
                ptr_arg(self.reduced_shard),
                ptr_arg(self.reduced_payload),
                ptr_arg(self.reduced_scale),
                self.rank,
                stream,
            ),
        )
        _barrier(
            self.reduced_payload,
            self.reduced_payload_base,
            self.reduced_ready,
            stream,
        )
        _run_compiled(
            k.compile_stage2_tp_all_gather(),
            (
                fx.Int64(self.reduced_payload_base),
                fx.Int64(self.reduced_scale_base),
                ptr_arg(self.output),
                self.rank,
                stream,
            ),
        )
        return self.output


class _PersistentRunner:
    def __init__(self, tp_group) -> None:
        k = windowed_kernels
        self.rank = int(tp_group.rank_in_group)
        self.device = torch.device(tp_group.device)
        self.routes = tuple(
            torch.empty(
                (k.M, k.TOPK, k.WINDOW + k.WINDOW // 8),
                dtype=torch.uint8,
                device=self.device,
            )
            for _ in range(k.SLOTS)
        )
        self.state = _symmetric(self.device, (k.STATE_BYTES,))
        self.partials = _symmetric(
            self.device, (k.PHASES * persistent_kernels.PARTIAL_STRIDE,)
        )
        self.reduced_payloads = _symmetric(
            self.device,
            (k.PHASES * persistent_kernels.REDUCED_PAYLOAD_STRIDE,),
        )
        self.reduced_scales = _symmetric(
            self.device,
            (k.PHASES * persistent_kernels.REDUCED_SCALE_STRIDE,),
        )
        self.output = torch.empty(
            (k.M, k.H), dtype=torch.bfloat16, device=self.device
        )
        self.state.zero_()

        self.comm, self.windows, bases = _register(
            tp_group,
            self.rank,
            k.TP,
            (
                self.state,
                self.partials,
                self.reduced_payloads,
                self.reduced_scales,
            ),
        )
        (
            self.state_flat_base,
            self.partial_flat_base,
            self.reduced_payload_flat_base,
            self.reduced_scale_flat_base,
        ) = bases
        self.service = persistent_kernels.compile_stage2_service()
        self.service_stream = torch.cuda.Stream(device=self.device)
        self.start_event = torch.cuda.Event()
        self.done_event = torch.cuda.Event()

    def _local_args(self, phase, shared_partial):
        k = windowed_kernels
        begin = phase * persistent_kernels.PARTIAL_STRIDE
        partial = self.partials[
            begin : begin + persistent_kernels.PARTIAL_STRIDE
        ]
        shared = shared_partial[:, phase * k.WINDOW :]
        return ptr_arg(self.routes[phase % k.SLOTS]), ptr_arg(partial), ptr_arg(shared)

    def _launch_service(self):
        _run_compiled(
            self.service,
            (
                ptr_arg(self.state),
                fx.Int64(self.state_flat_base),
                fx.Int64(self.partial_flat_base),
                fx.Int64(self.reduced_payload_flat_base),
                fx.Int64(self.reduced_scale_flat_base),
                ptr_arg(self.output),
                ptr_arg(self.reduced_payloads),
                ptr_arg(self.reduced_scales),
                self.rank,
                self.service_stream,
            ),
        )

    def __call__(self, *, stage2_args, stage2_kwargs, shared_partial):
        k = windowed_kernels
        producer = torch.cuda.current_stream(self.device)
        common = _stage2_args(stage2_args, stage2_kwargs, k)
        _run_compiled(
            k.compile_stage2_compute(0),
            (ptr_arg(self.routes[0]), *common, producer),
        )
        for phase in range(k.PHASES - 1):
            _run_compiled(
                k.compile_persistent_cycle(phase),
                (
                    ptr_arg(self.routes[(phase + 1) % k.SLOTS]),
                    *common,
                    *self._local_args(phase, shared_partial),
                    ptr_arg(self.state),
                    producer,
                ),
            )
            if phase == 0:
                self.start_event.record(producer)
                self.service_stream.wait_event(self.start_event)
                self._launch_service()

        last = k.PHASES - 1
        _run_compiled(
            k.compile_persistent_drain(),
            (
                *self._local_args(last, shared_partial),
                ptr_arg(self.state),
                producer,
            ),
        )
        _run_compiled(
            k.compile_persistent_final_publish(),
            (ptr_arg(self.state), producer),
        )
        self.done_event.record(self.service_stream)
        producer.wait_event(self.done_event)
        return self.output


def create_flydsl_comm_fused_runners(*, tp_group, model_dim, inter_dim, experts, topk):
    shape = (model_dim, inter_dim, experts, topk, int(tp_group.world_size))
    kernels = full_width_kernels
    if shape != (kernels.H, kernels.I, kernels.E, kernels.TOPK, kernels.TP):
        raise KeyError(f"unsupported comm_fused shape {shape}")
    global _RUNNERS
    if _RUNNERS is None:
        _RUNNERS = {
            full_width_kernels.M: _FullWidthRunner(tp_group),
            windowed_kernels.M: _PersistentRunner(tp_group),
        }
    return _RUNNERS
