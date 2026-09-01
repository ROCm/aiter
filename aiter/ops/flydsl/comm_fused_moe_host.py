# SPDX-License-Identifier: MIT
"""Production host runtime for communication-fused FlyDSL MoE."""

import csv
from dataclasses import MISSING, dataclass, fields
from functools import cache
from pathlib import Path

import flydsl.expr as fx
import torch
import torch.distributed._symmetric_memory as symm_mem
from mori.cco import Communicator

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.kernels.comm_fused_moe.gfx950.a8w4 import (
    gemm2_tp_atomic_pipeline,
    gemm2_tp_megakernel,
    gemm2_tp_window_pipeline,
)
from aiter.ops.flydsl.kernels.comm_fused_moe.gfx950.a8w4.shape import (
    Gemm2TPShape,
)
from aiter.ops.flydsl.kernels.comm_fused_moe.gfx950.a8w4.sync import (
    FLAT_VA_RANK_STRIDE,
    compile_epoch_barrier,
)
from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg
from aiter.ops.flydsl.moe_kernels import _run_compiled

_CONFIG_PATH = Path(__file__).parents[2] / "configs" / "comm_fused_moe.csv"
_PEER_VMM_ALLOCATION_ALIGNMENT = 2 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ShapeKey:
    gfx: str
    model_dim: int
    inter_dim: int
    experts: int
    topk: int
    tp: int

    def kernel_shape(self) -> Gemm2TPShape:
        return Gemm2TPShape(
            self.model_dim,
            self.inter_dim,
            self.experts,
            self.topk,
            self.tp,
        )


PipelineConfig = (
    gemm2_tp_atomic_pipeline.Gemm2TPAtomicPipelineConfig
    | gemm2_tp_megakernel.Gemm2TPMegakernelConfig
    | gemm2_tp_window_pipeline.Gemm2TPWindowPipelineConfig
)
_CONFIG_TYPES = {
    "gemm2_tp_atomic": gemm2_tp_atomic_pipeline.Gemm2TPAtomicPipelineConfig,
    "gemm2_tp_mega": gemm2_tp_megakernel.Gemm2TPMegakernelConfig,
    "gemm2_tp_window": gemm2_tp_window_pipeline.Gemm2TPWindowPipelineConfig,
}
_RUNNER_CACHE = {}


def _config(row, shape: Gemm2TPShape) -> PipelineConfig:
    config_type = _CONFIG_TYPES[row["family"]]
    values = {"shape": shape}
    for field in fields(config_type):
        if field.name == "shape":
            continue
        raw = row.get(field.name)
        if raw in (None, ""):
            if field.default is not MISSING:
                values[field.name] = field.default
                continue
            if field.default_factory is not MISSING:
                values[field.name] = field.default_factory()
                continue
            raise KeyError(field.name)
        if field.type is str:
            values[field.name] = raw
        elif field.type is bool:
            values[field.name] = bool(int(raw))
        else:
            values[field.name] = int(raw)
    return config_type(**values)


@cache
def _winner_table() -> dict[ShapeKey, dict[int, PipelineConfig]]:
    table = {}
    with _CONFIG_PATH.open(newline="") as file:
        for row in csv.DictReader(file):
            shape = ShapeKey(
                row["gfx"],
                int(row["model_dim"]),
                int(row["inter_dim"]),
                int(row["experts"]),
                int(row["topk"]),
                int(row["tp"]),
            )
            table.setdefault(shape, {})[int(row["m"])] = _config(
                row, shape.kernel_shape()
            )
    return table


def winners_for(shape: ShapeKey) -> dict[int, PipelineConfig]:
    table = _winner_table()
    try:
        return table[shape]
    except KeyError:
        raise KeyError(f"unsupported comm_fused shape {shape}") from None


def _symmetric(device, shape) -> torch.Tensor:
    requested_bytes = 1
    for extent in shape:
        requested_bytes *= int(extent)
    alignment = _PEER_VMM_ALLOCATION_ALIGNMENT
    allocated_bytes = max(
        alignment,
        (requested_bytes + alignment - 1) // alignment * alignment,
    )
    return symm_mem.empty((allocated_bytes,), dtype=torch.uint8, device=device)


def _packed_symmetric(
    device, sizes: tuple[int, ...]
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[int, ...]]:
    """Carve aligned views from one peer-VMM allocation and CCO window."""
    offsets = []
    total_bytes = 0
    for size in sizes:
        total_bytes = (total_bytes + 255) // 256 * 256
        offsets.append(total_bytes)
        total_bytes += int(size)
    workspace = _symmetric(device, (total_bytes,))
    tensors = tuple(
        workspace.narrow(0, offset, int(size)) for offset, size in zip(offsets, sizes)
    )
    return workspace, tensors, tuple(offsets)


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


def _barrier(tensor, flat_base, ready_offset, tp_size, stream) -> None:
    _run_compiled(
        compile_epoch_barrier(tp_size),
        (ptr_arg(tensor), fx.Int64(flat_base), fx.Int64(ready_offset), stream),
    )


def _stage2_args(args, kwargs, config):
    inter_states, w2 = args[0], args[2]
    sorted_token_ids, sorted_expert_ids, num_valid_ids = args[3:6]
    shape = config.shape
    return (
        ptr_arg(inter_states),
        ptr_arg(w2),
        ptr_arg(kwargs["a2_scale"].view(-1)),
        ptr_arg(kwargs["w2_scale"].view(-1)),
        ptr_arg(sorted_token_ids),
        ptr_arg(sorted_expert_ids),
        ptr_arg(kwargs["sorted_weights"]),
        ptr_arg(num_valid_ids),
        ptr_arg(inter_states),
        config.m,
        shape.model_dim,
        shape.inter_dim,
        int(sorted_expert_ids.shape[0]) * config.sort_block_m // config.tile_m,
    )


class _Gemm2TPAtomicPipelineRunner:
    def __init__(
        self,
        tp_group,
        config: gemm2_tp_atomic_pipeline.Gemm2TPAtomicPipelineConfig,
    ) -> None:
        self.config = config
        self.rank = int(tp_group.rank_in_group)
        self.device = torch.device(tp_group.device)
        shape = config.shape
        self.partial_ready = config.m * (shape.model_dim + shape.model_dim // 32)
        self.reduced_ready = config.shard_rows * shape.model_dim
        sizes = (
            (self.partial_ready + 8 + 255) // 256 * 256,
            (self.reduced_ready + 8 + 255) // 256 * 256,
            config.shard_rows * (shape.model_dim // 32),
        )
        self.workspace, tensors, offsets = _packed_symmetric(self.device, sizes)
        self.partial, self.reduced_payload, self.reduced_scale = tensors
        self.partial[self.partial_ready : self.partial_ready + 8].zero_()
        self.reduced_payload[self.reduced_ready : self.reduced_ready + 8].zero_()
        self.output = torch.empty(
            (config.m, shape.model_dim),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.comm, self.windows, (workspace_base,) = _register(
            tp_group, self.rank, shape.tp_size, (self.workspace,)
        )
        (
            self.partial_flat_base,
            self.reduced_payload_base,
            self.reduced_scale_base,
        ) = tuple(workspace_base + offset for offset in offsets)
        shard_begin = self.rank * config.shard_rows
        self.reduced_shard = self.output[shard_begin : shard_begin + config.shard_rows]

    def __call__(
        self,
        *,
        stage2_args: tuple,
        stage2_kwargs: dict,
        shared_partial,
        ordinary_stage2,
    ):
        k = gemm2_tp_atomic_pipeline
        config = self.config
        ordinary_stage2(
            *stage2_args[:6],
            shared_partial,
            *stage2_args[7:],
            **stage2_kwargs,
        )
        stream = torch.cuda.current_stream(self.device)
        _run_compiled(
            k.compile_stage2_quantize(config),
            (ptr_arg(shared_partial), ptr_arg(self.partial), stream),
        )
        _barrier(
            self.partial,
            self.partial_flat_base,
            self.partial_ready,
            config.shape.tp_size,
            stream,
        )
        _run_compiled(
            k.compile_stage2_tp_reduce_scatter(config),
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
            config.shape.tp_size,
            stream,
        )
        _run_compiled(
            k.compile_stage2_tp_all_gather(config),
            (
                fx.Int64(self.reduced_payload_base),
                fx.Int64(self.reduced_scale_base),
                ptr_arg(self.output),
                self.rank,
                stream,
            ),
        )
        return self.output


class _Gemm2TPMegakernelRunner:
    """Single-launch GEMM2 with per-N-tile TP collective services."""

    def __init__(
        self,
        tp_group,
        config: gemm2_tp_megakernel.Gemm2TPMegakernelConfig,
    ) -> None:
        shape = config.shape
        self.config = config
        self.rank = int(tp_group.rank_in_group)
        self.device = torch.device(tp_group.device)
        self.workspace = _symmetric(self.device, (config.workspace_bytes,))
        self.workspace.zero_()
        self.output = (
            self.workspace.narrow(0, config.output_offset, config.payload_bytes)
            .view(torch.bfloat16)
            .view(config.m, shape.model_dim)
        )
        self.comm, self.windows, bases = _register(
            tp_group,
            self.rank,
            shape.tp_size,
            (self.workspace,),
        )
        # Register all peer windows before launching a peer-dereferencing kernel.
        tp_group.barrier()
        self.shared_partial_window = None
        self.shared_partial_ptr = None
        self.shared_partial_flat_base = 0
        (self.workspace_flat_base,) = bases
        self.workspace.narrow(0, config.flat_base_offset, 8).view(torch.int64).fill_(
            self.workspace_flat_base
        )

    def prepare_shared_partial(self, shared_partial: torch.Tensor) -> torch.Tensor:
        """Stage a normal shared contribution in the registered output window."""

        if not self.config.shared_bf16_partials:
            return shared_partial
        if self.config.collective != "rs_broadcast":
            raise RuntimeError(
                "workspace-backed shared BF16 partials require "
                "collective='rs_broadcast'"
            )
        if shared_partial.data_ptr() != self.output.data_ptr():
            self.output.copy_(shared_partial)
        return self.output

    def __call__(
        self,
        *,
        stage2_args: tuple,
        stage2_kwargs: dict,
        shared_partial,
        ordinary_stage2,
    ):
        del ordinary_stage2
        k = gemm2_tp_megakernel
        stream = torch.cuda.current_stream(self.device)
        if self.config.shared_bf16_partials:
            shared_partial_ptr = shared_partial.data_ptr()
            if shared_partial_ptr == self.output.data_ptr():
                if self.shared_partial_ptr not in (None, shared_partial_ptr):
                    raise RuntimeError(
                        "GEMM2 TP megakernel shared_partial storage changed after "
                        "symmetric registration"
                    )
                self.shared_partial_ptr = shared_partial_ptr
                self.shared_partial_flat_base = (
                    self.workspace_flat_base + self.config.output_offset
                )
            elif self.shared_partial_window is None:
                self.shared_partial_window = self.comm.register_external_window(
                    shared_partial_ptr,
                    shared_partial.nbytes,
                )
                self.shared_partial_ptr = shared_partial_ptr
                self.shared_partial_flat_base = (
                    self.shared_partial_window.local_ptr
                    - self.rank * FLAT_VA_RANK_STRIDE
                )
            elif shared_partial_ptr != self.shared_partial_ptr:
                raise RuntimeError(
                    "GEMM2 TP megakernel shared_partial storage changed after "
                    "symmetric registration"
                )
        common = list(_stage2_args(stage2_args, stage2_kwargs, self.config))
        _run_compiled(
            k.compile_gemm2_tp_megakernel(self.config, self.rank),
            (
                ptr_arg(self.workspace),
                ptr_arg(shared_partial),
                fx.Int64(self.shared_partial_flat_base),
                *common[:8],
                *common[9:],
                stream,
            ),
        )
        return self.output


class _Gemm2TPWindowPipelineRunner:
    def __init__(
        self,
        tp_group,
        config: gemm2_tp_window_pipeline.Gemm2TPWindowPipelineConfig,
    ) -> None:
        k = gemm2_tp_window_pipeline
        shape = config.shape
        self.config = config
        self.rank = int(tp_group.rank_in_group)
        self.device = torch.device(tp_group.device)
        self.routes = tuple(
            torch.empty(
                (
                    config.m,
                    shape.topk,
                    config.window + config.window // 8,
                ),
                dtype=torch.uint8,
                device=self.device,
            )
            for _ in range(k.SLOTS)
        )
        self.partial_ready = config.m * (config.window + config.window // 32)
        self.reduced_ready = config.shard_rows * config.window
        sizes = (
            ((self.partial_ready + 8 + 255) // 256 * 256,) * k.SLOTS
            + ((self.reduced_ready + 8 + 255) // 256 * 256,) * k.SLOTS
            + (config.shard_rows * (config.window // 32),) * k.SLOTS
        )
        self.workspace, tensors, offsets = _packed_symmetric(self.device, sizes)
        self.partials = tensors[: k.SLOTS]
        self.reduced_payloads = tensors[k.SLOTS : 2 * k.SLOTS]
        self.reduced_scales = tensors[2 * k.SLOTS :]
        for partial in self.partials:
            partial[self.partial_ready : self.partial_ready + 8].zero_()
        for payload in self.reduced_payloads:
            payload[self.reduced_ready : self.reduced_ready + 8].zero_()
        self.output = torch.empty(
            (config.m, shape.model_dim),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.comm, self.windows, (workspace_base,) = _register(
            tp_group, self.rank, shape.tp_size, (self.workspace,)
        )
        bases = tuple(workspace_base + offset for offset in offsets)
        self.partial_bases = bases[: k.SLOTS]
        self.reduced_payload_bases = bases[k.SLOTS : 2 * k.SLOTS]
        self.reduced_scale_bases = bases[2 * k.SLOTS :]
        shard_begin = self.rank * config.shard_rows
        self.reduced_shards = tuple(
            self.output[
                shard_begin : shard_begin + config.shard_rows,
                phase * config.window : (phase + 1) * config.window,
            ]
            for phase in range(shape.model_dim // config.window)
        )
        self.gathered_outputs = tuple(
            self.output[:, phase * config.window : (phase + 1) * config.window]
            for phase in range(shape.model_dim // config.window)
        )

    def _local_args(self, phase, shared_partial):
        slot = phase % gemm2_tp_window_pipeline.SLOTS
        shared = shared_partial[:, phase * self.config.window :]
        return (
            ptr_arg(self.routes[slot]),
            ptr_arg(self.partials[slot]),
            ptr_arg(shared),
        )

    def _collective_args(self, reduce_scatter, all_gather):
        reduce_slot = reduce_scatter % gemm2_tp_window_pipeline.SLOTS
        gather_slot = all_gather % gemm2_tp_window_pipeline.SLOTS
        return (
            fx.Int64(self.partial_bases[reduce_slot]),
            ptr_arg(self.reduced_shards[reduce_scatter]),
            ptr_arg(self.reduced_payloads[reduce_slot]),
            ptr_arg(self.reduced_scales[reduce_slot]),
            fx.Int64(self.reduced_payload_bases[gather_slot]),
            fx.Int64(self.reduced_scale_bases[gather_slot]),
            ptr_arg(self.gathered_outputs[all_gather]),
        )

    def _drain(self, local, reduce_scatter, all_gather, shared_partial, stream):
        _run_compiled(
            gemm2_tp_window_pipeline.compile_stage2_drain(
                self.config,
                local is not None,
                reduce_scatter is not None,
                all_gather is not None,
            ),
            (
                *self._local_args(0 if local is None else local, shared_partial),
                *self._collective_args(
                    0 if reduce_scatter is None else reduce_scatter,
                    0 if all_gather is None else all_gather,
                ),
                self.rank,
                stream,
            ),
        )

    def __call__(
        self,
        *,
        stage2_args: tuple,
        stage2_kwargs: dict,
        shared_partial,
        ordinary_stage2,
    ):
        k = gemm2_tp_window_pipeline
        config = self.config
        stream = torch.cuda.current_stream(self.device)
        common = _stage2_args(stage2_args, stage2_kwargs, config)
        _run_compiled(
            k.compile_stage2_compute(config, 0),
            (ptr_arg(self.routes[0]), *common, stream),
        )
        for local in range(len(self.reduced_shards) - 1):
            reduce_scatter = local - 1
            all_gather = local - 2
            _run_compiled(
                k.compile_stage2_cycle(
                    config,
                    local + 1,
                    reduce_scatter >= 0,
                    all_gather >= 0,
                ),
                (
                    ptr_arg(self.routes[(local + 1) % k.SLOTS]),
                    *common,
                    *self._local_args(local, shared_partial),
                    *self._collective_args(max(reduce_scatter, 0), max(all_gather, 0)),
                    self.rank,
                    stream,
                ),
            )
            local_slot = local % k.SLOTS
            _barrier(
                self.partials[local_slot],
                self.partial_bases[local_slot],
                self.partial_ready,
                config.shape.tp_size,
                stream,
            )
            if reduce_scatter >= 0:
                reduce_slot = reduce_scatter % k.SLOTS
                _barrier(
                    self.reduced_payloads[reduce_slot],
                    self.reduced_payload_bases[reduce_slot],
                    self.reduced_ready,
                    config.shape.tp_size,
                    stream,
                )

        last = len(self.reduced_shards) - 1
        self._drain(last, last - 1, last - 2, shared_partial, stream)
        last_slot = last % k.SLOTS
        reduce_slot = (last - 1) % k.SLOTS
        _barrier(
            self.partials[last_slot],
            self.partial_bases[last_slot],
            self.partial_ready,
            config.shape.tp_size,
            stream,
        )
        _barrier(
            self.reduced_payloads[reduce_slot],
            self.reduced_payload_bases[reduce_slot],
            self.reduced_ready,
            config.shape.tp_size,
            stream,
        )
        self._drain(None, last, last - 1, shared_partial, stream)
        _barrier(
            self.reduced_payloads[last_slot],
            self.reduced_payload_bases[last_slot],
            self.reduced_ready,
            config.shape.tp_size,
            stream,
        )
        self._drain(None, None, last, shared_partial, stream)
        return self.output


_RUNNER_TYPES = {
    gemm2_tp_atomic_pipeline.Gemm2TPAtomicPipelineConfig: (
        _Gemm2TPAtomicPipelineRunner
    ),
    gemm2_tp_megakernel.Gemm2TPMegakernelConfig: _Gemm2TPMegakernelRunner,
    gemm2_tp_window_pipeline.Gemm2TPWindowPipelineConfig: (
        _Gemm2TPWindowPipelineRunner
    ),
}


def create_runner(tp_group, config: PipelineConfig):
    return _RUNNER_TYPES[type(config)](tp_group, config)


class _LazyRunners:
    def __init__(self, tp_group, configs: dict[int, PipelineConfig]) -> None:
        self.tp_group = tp_group
        self.configs = configs
        self.instances = {}

    def __contains__(self, tokens: int) -> bool:
        return tokens in self.configs

    def __getitem__(self, tokens: int):
        if tokens not in self.instances:
            self.instances[tokens] = create_runner(self.tp_group, self.configs[tokens])
        return self.instances[tokens]


def create_flydsl_comm_fused_runners(*, tp_group, model_dim, inter_dim, experts, topk):
    shape = ShapeKey(
        get_gfx_runtime(),
        model_dim,
        inter_dim,
        experts,
        topk,
        int(tp_group.world_size),
    )
    key = (id(tp_group), shape)
    if key not in _RUNNER_CACHE:
        _RUNNER_CACHE[key] = _LazyRunners(tp_group, winners_for(shape))
    return _RUNNER_CACHE[key]
