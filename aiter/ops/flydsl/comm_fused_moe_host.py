# SPDX-License-Identifier: MIT
"""Production host runtime for communication-fused FlyDSL MoE."""

import csv
from dataclasses import dataclass, fields
from functools import cache
from pathlib import Path

import flydsl.expr as fx
import torch
import torch.distributed._symmetric_memory as symm_mem
from mori.cco import Communicator

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.kernels.comm_fused_moe import atomic_compressed
from aiter.ops.flydsl.kernels.comm_fused_moe import full_width
from aiter.ops.flydsl.kernels.comm_fused_moe import persistent_window
from aiter.ops.flydsl.kernels.comm_fused_moe import small_m_allreduce
from aiter.ops.flydsl.kernels.comm_fused_moe import windowed
from aiter.ops.flydsl.kernels.comm_fused_moe.sync import (
    FLAT_VA_RANK_STRIDE,
    compile_epoch_barrier,
)
from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg
from aiter.ops.flydsl.moe_kernels import _run_compiled


_CONFIG_PATH = Path(__file__).parents[2] / "configs" / "comm_fused_moe.csv"


@dataclass(frozen=True, slots=True)
class ShapeKey:
    gfx: str
    model_dim: int
    inter_dim: int
    experts: int
    topk: int
    tp: int


PipelineConfig = (
    small_m_allreduce.Config
    | atomic_compressed.Config
    | full_width.Config
    | windowed.Config
    | persistent_window.Config
)
_CONFIG_TYPES = {
    "small": small_m_allreduce.Config,
    "atomic": atomic_compressed.Config,
    "full": full_width.Config,
    "window": windowed.Config,
    "persistent": persistent_window.Config,
}
_RUNNER_CACHE = {}


def _config(row) -> PipelineConfig:
    config_type = _CONFIG_TYPES[row["family"]]
    return config_type(
        **{field.name: int(row[field.name]) for field in fields(config_type)}
    )


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
            table.setdefault(shape, {})[int(row["m"])] = _config(row)
    return table


def winners_for(shape: ShapeKey) -> dict[int, PipelineConfig]:
    table = _winner_table()
    try:
        return table[shape]
    except KeyError:
        raise KeyError(f"unsupported comm_fused shape {shape}") from None


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


def _barrier(tensor, flat_base, ready_offset, stream) -> None:
    _run_compiled(
        compile_epoch_barrier(),
        (ptr_arg(tensor), fx.Int64(flat_base), fx.Int64(ready_offset), stream),
    )


def _stage2_args(args, kwargs, kernels, config):
    inter_states, w2 = args[0], args[2]
    sorted_token_ids, sorted_expert_ids, num_valid_ids = args[3:6]
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
        kernels.H,
        kernels.I,
        int(sorted_expert_ids.shape[0]) * config.sort_block_m // config.tile_m,
    )


class _SmallMRunner:
    """Exact small-M single-launch GEMM2 + TP8 AllReduce runner."""

    def __init__(self, tp_group, config: small_m_allreduce.Config) -> None:
        k = small_m_allreduce
        config.validate()
        if get_gfx_runtime() != "gfx950" or int(tp_group.world_size) != k.TP:
            raise ValueError(
                "small-M megakernel requires gfx950 TP8, got "
                f"gfx={get_gfx_runtime()} tp={int(tp_group.world_size)}"
            )
        self.config = config
        self.rank = int(tp_group.rank_in_group)
        self.device = torch.device(tp_group.device)
        self.routes = symm_mem.empty(
            (config.m * k.M1_TOPK, k.DEFAULT_HIDDEN),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.partial = symm_mem.empty(
            (k.M1_PARTIAL_BUFFERS, config.m, k.DEFAULT_HIDDEN),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.state = _symmetric(self.device, (k.m1_state_layout(config)[-1],))
        self.output = torch.empty(
            (config.m, k.DEFAULT_HIDDEN),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.state.zero_()
        self.comm, self.windows, bases = _register(
            tp_group, self.rank, k.TP, (self.routes, self.partial, self.state)
        )
        _routes_flat_base, self.partial_flat_base, self.state_flat_base = bases
        self.kernel = k.compile_m1_gemm2_allreduce(config, self.rank)

    def __call__(
        self,
        *,
        stage2_args: tuple,
        stage2_kwargs: dict,
        shared_partial,
        ordinary_stage2,
    ):
        del ordinary_stage2
        k = small_m_allreduce
        if tuple(shared_partial.shape) != (self.config.m, k.DEFAULT_HIDDEN):
            raise ValueError(
                "small-M megakernel requires shared_partial shape "
                f"({self.config.m}, {k.DEFAULT_HIDDEN}), got "
                f"{tuple(shared_partial.shape)}"
            )
        if int(stage2_args[7]) != k.M1_TOPK:
            raise ValueError(
                f"small-M megakernel requires topk={k.M1_TOPK}, "
                f"got {int(stage2_args[7])}"
            )
        inter_states, w2 = stage2_args[0], stage2_args[2]
        sorted_token_ids, sorted_expert_ids, num_valid_ids = stage2_args[3:6]
        sorted_weights = stage2_kwargs["sorted_weights"]
        if sorted_weights is None:
            raise ValueError("small-M megakernel requires Stage2 routing weights")
        if (
            stage2_kwargs.get("a2_scale") is None
            or stage2_kwargs.get("w2_scale") is None
        ):
            raise ValueError(
                "small-M megakernel requires FP8 activation and FP4 weight scales"
            )
        # The sorting buffers reserve entries for every expert.  Launching that
        # full capacity would create hundreds of empty GEMM CTAs at tiny M.
        # Each token can contribute at most one row to each of its TOPK experts,
        # so M*TOPK is a safe host-side upper bound on active sort blocks.
        sort_blocks = min(
            int(sorted_expert_ids.shape[0]), self.config.m * k.M1_TOPK
        )
        size_expert_ids = (
            sort_blocks * self.config.sort_block_m // self.config.tile_m
        )
        _run_compiled(
            self.kernel,
            (
                ptr_arg(self.routes),
                ptr_arg(inter_states),
                ptr_arg(w2),
                ptr_arg(stage2_kwargs["a2_scale"].view(-1)),
                ptr_arg(stage2_kwargs["w2_scale"].view(-1)),
                ptr_arg(sorted_token_ids),
                ptr_arg(sorted_expert_ids),
                ptr_arg(sorted_weights),
                ptr_arg(num_valid_ids),
                ptr_arg(shared_partial),
                self.config.m,
                k.DEFAULT_HIDDEN,
                k.M1_INTER_DIM,
                size_expert_ids,
                ptr_arg(self.partial),
                fx.Int64(self.partial_flat_base),
                ptr_arg(self.state),
                fx.Int64(self.state_flat_base),
                ptr_arg(self.output),
                self.rank,
                torch.cuda.current_stream(self.device),
            ),
        )
        return self.output


class _AtomicCompressedRunner:
    def __init__(self, tp_group, config: atomic_compressed.Config) -> None:
        k = atomic_compressed
        self.config = config
        self.rank = int(tp_group.rank_in_group)
        self.device = torch.device(tp_group.device)
        self.partial_ready = config.m * (k.H + k.H // 32)
        self.partial = _workspace(self.device, self.partial_ready)
        self.reduced_ready = config.shard_rows * k.H
        self.reduced_payload = _workspace(self.device, self.reduced_ready)
        self.reduced_scale = _symmetric(
            self.device, (config.shard_rows, k.H // 32)
        )
        self.output = torch.empty(
            (config.m, k.H), dtype=torch.bfloat16, device=self.device
        )
        self.comm, self.windows, bases = _register(
            tp_group,
            self.rank,
            k.TP,
            (self.partial, self.reduced_payload, self.reduced_scale),
        )
        (
            self.partial_flat_base,
            self.reduced_payload_base,
            self.reduced_scale_base,
        ) = bases
        shard_begin = self.rank * config.shard_rows
        self.reduced_shard = self.output[
            shard_begin : shard_begin + config.shard_rows
        ]

    def __call__(
        self,
        *,
        stage2_args: tuple,
        stage2_kwargs: dict,
        shared_partial,
        ordinary_stage2,
    ):
        k = atomic_compressed
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
        _barrier(self.partial, self.partial_flat_base, self.partial_ready, stream)
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


class _FullWidthRunner:
    def __init__(self, tp_group, config: full_width.Config) -> None:
        k = full_width
        self.config = config
        self.rank = int(tp_group.rank_in_group)
        self.device = torch.device(tp_group.device)
        self.route = torch.empty(
            (config.m, k.TOPK, k.H + k.H // 8),
            dtype=torch.uint8,
            device=self.device,
        )
        self.partial_ready = config.m * (k.H + k.H // 32)
        self.partial = _workspace(self.device, self.partial_ready)
        self.reduced_ready = config.shard_rows * k.H
        self.reduced_payload = _workspace(self.device, self.reduced_ready)
        self.reduced_scale = _symmetric(
            self.device, (config.shard_rows, k.H // 32)
        )
        self.output = torch.empty(
            (config.m, k.H), dtype=torch.bfloat16, device=self.device
        )
        self.comm, self.windows, bases = _register(
            tp_group,
            self.rank,
            k.TP,
            (self.partial, self.reduced_payload, self.reduced_scale),
        )
        (
            self.partial_flat_base,
            self.reduced_payload_base,
            self.reduced_scale_base,
        ) = bases
        shard_begin = self.rank * config.shard_rows
        self.reduced_shard = self.output[
            shard_begin : shard_begin + config.shard_rows
        ]

    def __call__(
        self,
        *,
        stage2_args: tuple,
        stage2_kwargs: dict,
        shared_partial,
        ordinary_stage2,
    ):
        k = full_width
        config = self.config
        stream = torch.cuda.current_stream(self.device)
        common = _stage2_args(stage2_args, stage2_kwargs, k, config)
        _run_compiled(
            k.compile_stage2_compute(config),
            (ptr_arg(self.route), *common, stream),
        )
        _run_compiled(
            k.compile_stage2_local_reduce(config),
            (
                ptr_arg(self.route),
                ptr_arg(self.partial),
                ptr_arg(shared_partial),
                stream,
            ),
        )
        _barrier(self.partial, self.partial_flat_base, self.partial_ready, stream)
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


class _WindowedRunner:
    def __init__(self, tp_group, config: windowed.Config) -> None:
        k = windowed
        self.config = config
        self.rank = int(tp_group.rank_in_group)
        self.device = torch.device(tp_group.device)
        self.routes = tuple(
            torch.empty(
                (config.m, k.TOPK, config.window + config.window // 8),
                dtype=torch.uint8,
                device=self.device,
            )
            for _ in range(k.SLOTS)
        )
        self.partial_ready = config.m * (config.window + config.window // 32)
        self.partials = tuple(
            _workspace(self.device, self.partial_ready) for _ in range(k.SLOTS)
        )
        self.reduced_ready = config.shard_rows * config.window
        self.reduced_payloads = tuple(
            _workspace(self.device, self.reduced_ready) for _ in range(k.SLOTS)
        )
        self.reduced_scales = tuple(
            _symmetric(self.device, (config.shard_rows, config.window // 32))
            for _ in range(k.SLOTS)
        )
        self.output = torch.empty(
            (config.m, k.H), dtype=torch.bfloat16, device=self.device
        )
        self.comm, self.windows, bases = _register(
            tp_group,
            self.rank,
            k.TP,
            (*self.partials, *self.reduced_payloads, *self.reduced_scales),
        )
        self.partial_bases = bases[: k.SLOTS]
        self.reduced_payload_bases = bases[k.SLOTS : 2 * k.SLOTS]
        self.reduced_scale_bases = bases[2 * k.SLOTS :]
        shard_begin = self.rank * config.shard_rows
        self.reduced_shards = tuple(
            self.output[
                shard_begin : shard_begin + config.shard_rows,
                phase * config.window : (phase + 1) * config.window,
            ]
            for phase in range(k.H // config.window)
        )
        self.gathered_outputs = tuple(
            self.output[:, phase * config.window : (phase + 1) * config.window]
            for phase in range(k.H // config.window)
        )

    def _local_args(self, phase, shared_partial):
        slot = phase % windowed.SLOTS
        shared = shared_partial[:, phase * self.config.window :]
        return (
            ptr_arg(self.routes[slot]),
            ptr_arg(self.partials[slot]),
            ptr_arg(shared),
        )

    def _collective_args(self, reduce_scatter, all_gather):
        reduce_slot = reduce_scatter % windowed.SLOTS
        gather_slot = all_gather % windowed.SLOTS
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
            windowed.compile_stage2_drain(
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
        k = windowed
        config = self.config
        stream = torch.cuda.current_stream(self.device)
        common = _stage2_args(stage2_args, stage2_kwargs, k, config)
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
                stream,
            )
            if reduce_scatter >= 0:
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
            self.partials[last_slot],
            self.partial_bases[last_slot],
            self.partial_ready,
            stream,
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


class _PersistentWindowRunner:
    def __init__(self, tp_group, config: persistent_window.Config) -> None:
        k = persistent_window
        self.config = config
        self.rank = int(tp_group.rank_in_group)
        self.device = torch.device(tp_group.device)
        self.routes = tuple(
            torch.empty(
                (config.m, k.TOPK, config.window + config.window // 8),
                dtype=torch.uint8,
                device=self.device,
            )
            for _ in range(k.SLOTS)
        )
        self.state = _symmetric(self.device, (config.state_bytes,))
        self.partials = _symmetric(
            self.device, (config.phases * config.partial_stride,)
        )
        self.reduced_payloads = _symmetric(
            self.device, (config.phases * config.reduced_payload_stride,)
        )
        self.reduced_scales = _symmetric(
            self.device, (config.phases * config.reduced_scale_stride,)
        )
        self.output = torch.empty(
            (config.m, k.H), dtype=torch.bfloat16, device=self.device
        )
        self.state.zero_()
        self.comm, self.windows, bases = _register(
            tp_group,
            self.rank,
            k.TP,
            (self.state, self.partials, self.reduced_payloads, self.reduced_scales),
        )
        (
            self.state_flat_base,
            self.partial_flat_base,
            self.reduced_payload_flat_base,
            self.reduced_scale_flat_base,
        ) = bases
        self.service = k.compile_stage2_service(config)
        self.phase0_compute = (
            k.compile_stage2_compute(config, 0)
            if config.producer_workers_per_n_tile == 0
            else k.compile_stage2_compute_bounded(
                config,
                0,
                config.producer_workers_per_n_tile,
                config.producer_ctas_per_cu_limit,
            )
        )
        self.service_stream = torch.cuda.Stream(
            device=self.device,
            priority=-1,
        )
        self.done_event = torch.cuda.Event()

    def _local_args(self, phase, shared_partial):
        config = self.config
        begin = phase * config.partial_stride
        partial = self.partials[begin : begin + config.partial_stride]
        shared = shared_partial[:, phase * config.window :]
        return (
            ptr_arg(self.routes[phase % persistent_window.SLOTS]),
            ptr_arg(partial),
            ptr_arg(shared),
        )

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

    def __call__(
        self,
        *,
        stage2_args: tuple,
        stage2_kwargs: dict,
        shared_partial,
        ordinary_stage2,
    ):
        k = persistent_window
        config = self.config
        producer = torch.cuda.current_stream(self.device)
        common = _stage2_args(stage2_args, stage2_kwargs, k, config)
        self._launch_service()
        _run_compiled(
            self.phase0_compute,
            (ptr_arg(self.routes[0]), *common, producer),
        )
        for phase in range(config.phases - 1):
            _run_compiled(
                k.compile_persistent_cycle(config, phase),
                (
                    ptr_arg(self.routes[(phase + 1) % k.SLOTS]),
                    *common,
                    *self._local_args(phase, shared_partial),
                    ptr_arg(self.state),
                    producer,
                ),
            )

        last = config.phases - 1
        _run_compiled(
            k.compile_persistent_drain(config),
            (*self._local_args(last, shared_partial), ptr_arg(self.state), producer),
        )
        _run_compiled(
            k.compile_persistent_final_publish(config),
            (ptr_arg(self.state), producer),
        )
        self.done_event.record(self.service_stream)
        producer.wait_event(self.done_event)
        return self.output


_RUNNER_TYPES = {
    small_m_allreduce.Config: _SmallMRunner,
    atomic_compressed.Config: _AtomicCompressedRunner,
    full_width.Config: _FullWidthRunner,
    windowed.Config: _WindowedRunner,
    persistent_window.Config: _PersistentWindowRunner,
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


def create_flydsl_comm_fused_runners(
    *, tp_group, model_dim, inter_dim, experts, topk
):
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
