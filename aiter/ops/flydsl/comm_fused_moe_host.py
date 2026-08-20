# SPDX-License-Identifier: MIT
"""FlyDSL host runner for communication-fused MoE."""

import torch
import torch.distributed._symmetric_memory as symm_mem
import flydsl.expr as fx
from mori.cco import Communicator

from aiter.ops.flydsl.kernels.comm_fused_moe import full_width as full_width_kernels
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
        int(sorted_expert_ids.shape[0]) * 2,
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


class _WindowedRunner:
    def __init__(self, tp_group) -> None:
        kernels = windowed_kernels
        self.rank = int(tp_group.rank_in_group)
        self.device = torch.device(tp_group.device)
        self.routes = tuple(
            torch.empty(
                (kernels.M, kernels.TOPK, kernels.WINDOW + kernels.WINDOW // 8),
                dtype=torch.uint8,
                device=self.device,
            )
            for _ in range(kernels.SLOTS)
        )
        self.partial_ready = kernels.M * (
            kernels.WINDOW + kernels.WINDOW // 32
        )
        self.partials = tuple(
            _workspace(self.device, self.partial_ready)
            for _ in range(kernels.SLOTS)
        )
        self.reduced_ready = kernels.SHARD_ROWS * kernels.WINDOW
        self.reduced_payloads = tuple(
            _workspace(self.device, self.reduced_ready)
            for _ in range(kernels.SLOTS)
        )
        self.reduced_scales = tuple(
            _symmetric(self.device, (kernels.SHARD_ROWS, kernels.WINDOW // 32))
            for _ in range(kernels.SLOTS)
        )
        self.output = torch.empty(
            (kernels.M, kernels.H), dtype=torch.bfloat16, device=self.device
        )

        self.comm, self.windows, bases = _register(
            tp_group,
            self.rank,
            kernels.TP,
            (*self.partials, *self.reduced_payloads, *self.reduced_scales),
        )
        self.partial_bases = bases[: kernels.SLOTS]
        self.reduced_payload_bases = bases[kernels.SLOTS : 2 * kernels.SLOTS]
        self.reduced_scale_bases = bases[2 * kernels.SLOTS :]
        shard_begin = self.rank * kernels.SHARD_ROWS
        self.reduced_shards = tuple(
            self.output[
                shard_begin : shard_begin + kernels.SHARD_ROWS,
                window * kernels.WINDOW : (window + 1) * kernels.WINDOW,
            ]
            for window in range(kernels.H // kernels.WINDOW)
        )
        self.all_gather_outputs = tuple(
            self.output[
                :, window * kernels.WINDOW : (window + 1) * kernels.WINDOW
            ]
            for window in range(kernels.H // kernels.WINDOW)
        )

    def _local_args(self, window, shared_partial):
        slot = window % windowed_kernels.SLOTS
        shared = shared_partial[:, window * windowed_kernels.WINDOW :]
        return ptr_arg(self.routes[slot]), ptr_arg(self.partials[slot]), ptr_arg(shared)

    def _collective_args(self, reduce_scatter, all_gather):
        reduce_scatter_slot = reduce_scatter % windowed_kernels.SLOTS
        all_gather_slot = all_gather % windowed_kernels.SLOTS
        return (
            fx.Int64(self.partial_bases[reduce_scatter_slot]),
            ptr_arg(self.reduced_shards[reduce_scatter]),
            ptr_arg(self.reduced_payloads[reduce_scatter_slot]),
            ptr_arg(self.reduced_scales[reduce_scatter_slot]),
            fx.Int64(self.reduced_payload_bases[all_gather_slot]),
            fx.Int64(self.reduced_scale_bases[all_gather_slot]),
            ptr_arg(self.all_gather_outputs[all_gather]),
        )

    def _drain(
        self,
        local,
        reduce_scatter,
        all_gather,
        shared_partial,
        stream,
    ) -> None:
        kernels = windowed_kernels
        local_window = 0 if local is None else local
        reduce_window = 0 if reduce_scatter is None else reduce_scatter
        gather_window = 0 if all_gather is None else all_gather
        _run_compiled(
            kernels.compile_stage2_drain(
                local is not None,
                reduce_scatter is not None,
                all_gather is not None,
            ),
            (
                *self._local_args(local_window, shared_partial),
                *self._collective_args(reduce_window, gather_window),
                self.rank,
                stream,
            ),
        )

    def __call__(self, *, stage2_args, stage2_kwargs, shared_partial):
        k = windowed_kernels
        stream = torch.cuda.current_stream(self.device)
        common = _stage2_args(stage2_args, stage2_kwargs, k)
        _run_compiled(
            k.compile_stage2_compute(0), (ptr_arg(self.routes[0]), *common, stream)
        )
        for local in range(len(self.reduced_shards) - 1):
            has_reduce_scatter = local > 0
            has_all_gather = local > 1
            reduce_scatter = local - 1 if has_reduce_scatter else 0
            all_gather = local - 2 if has_all_gather else 0
            _run_compiled(
                k.compile_stage2_cycle(
                    local + 1,
                    has_reduce_scatter,
                    has_all_gather,
                ),
                (
                    ptr_arg(self.routes[(local + 1) % k.SLOTS]),
                    *common,
                    *self._local_args(local, shared_partial),
                    *self._collective_args(reduce_scatter, all_gather),
                    self.rank,
                    stream,
                ),
            )
            local_slot = local % k.SLOTS
            _barrier(
                self.partials[local_slot],
                self.partial_bases[local_slot],
                self.partial_ready,
                stream,
            )
            if has_reduce_scatter:
                reduce_slot = reduce_scatter % k.SLOTS
                _barrier(
                    self.reduced_payloads[reduce_slot],
                    self.reduced_payload_bases[reduce_slot],
                    self.reduced_ready,
                    stream,
                )

        last = len(self.reduced_shards) - 1
        self._drain(last, last - 1, last - 2, shared_partial, stream)
        last_slot = last % k.SLOTS
        reduce_slot = (last - 1) % k.SLOTS
        _barrier(
            self.partials[last_slot], self.partial_bases[last_slot], self.partial_ready, stream
        )
        _barrier(
            self.reduced_payloads[reduce_slot],
            self.reduced_payload_bases[reduce_slot],
            self.reduced_ready,
            stream,
        )
        self._drain(None, last, last - 1, shared_partial, stream)
        _barrier(
            self.reduced_payloads[last_slot],
            self.reduced_payload_bases[last_slot],
            self.reduced_ready,
            stream,
        )
        self._drain(None, None, last, shared_partial, stream)
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
            windowed_kernels.M: _WindowedRunner(tp_group),
        }
    return _RUNNERS
