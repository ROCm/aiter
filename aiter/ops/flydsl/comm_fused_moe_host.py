# SPDX-License-Identifier: MIT
"""Host runtime for communication-fused FlyDSL MoE (atomic family only).

Trimmed port of ``yifehuan/comm_fused_moe`` covering the ``atomic`` token
buckets.  The atomic pipeline keeps the ordinary Stage2 GEMM and replaces only
the post-Stage2 TP collective:

    ordinary Stage2 (accumulates into ``shared_partial``)
    -> MXFP8 quantize
    -> direct-pull ReduceScatter (writes this rank's BF16 shard)
    -> AllGather + BF16 decode                      [skipped when all_gather=0]

Data-parallel callers only ever consume their own shard, so they stop after the
ReduceScatter and read ``runner.reduced_shard``.
"""

import csv
from dataclasses import dataclass, fields
from functools import cache
from math import prod
from pathlib import Path

import flydsl.expr as fx
import torch
import torch.distributed._symmetric_memory as symm_mem
from mori.cco import Communicator

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.kernels.comm_fused_moe import atomic_compressed
from aiter.ops.flydsl.kernels.comm_fused_moe.sync import (
    FLAT_VA_RANK_STRIDE,
    compile_epoch_barrier,
)
from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg
from aiter.ops.flydsl.moe_kernels import _run_compiled

_CONFIG_PATH = Path(__file__).parents[2] / "configs" / "comm_fused_moe.csv"
_ARENA_ALIGNMENT = 64 * 1024


@dataclass(frozen=True, slots=True)
class ShapeKey:
    gfx: str
    model_dim: int
    inter_dim: int
    experts: int
    topk: int
    tp: int


PipelineConfig = atomic_compressed.Config
_CONFIG_TYPES = {"atomic": atomic_compressed.Config}
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
            if row["family"] not in _CONFIG_TYPES:
                continue
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


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


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


def _allocate_registered(
    tp_group,
    rank: int,
    tp: int,
    device,
    specs: tuple[tuple[tuple[int, ...], torch.dtype], ...],
):
    """Allocate typed views in one symmetric arena and register it once.

    The MORI CCO build in use accepts only a single external window per
    communicator - a second ``register_external_window`` fails with
    ``ccoWindowRegister (external) failed: -1`` - so every workspace shares one
    64 KiB-aligned arena and each view's flat peer base is derived by offset.
    """
    layouts = []
    arena_bytes = 0
    for shape, dtype in specs:
        arena_bytes = _align_up(arena_bytes, _ARENA_ALIGNMENT)
        element_size = torch.empty((), dtype=dtype).element_size()
        nbytes = prod(shape) * element_size
        layouts.append((arena_bytes, nbytes, shape, dtype))
        arena_bytes += nbytes
    arena_bytes = _align_up(arena_bytes, _ARENA_ALIGNMENT)
    if arena_bytes > FLAT_VA_RANK_STRIDE:
        raise ValueError(
            f"comm-fused symmetric arena requires {arena_bytes} bytes, "
            f"exceeding per-rank VMM stride {FLAT_VA_RANK_STRIDE}"
        )

    arena = _symmetric(device, (arena_bytes,))
    views = tuple(
        arena.narrow(0, offset, nbytes).view(dtype).view(shape)
        for offset, nbytes, shape, dtype in layouts
    )
    comm, windows, (arena_flat_base,) = _register(tp_group, rank, tp, (arena,))
    arena_address = arena.data_ptr()
    bases = tuple(arena_flat_base + view.data_ptr() - arena_address for view in views)
    return arena, views, comm, windows, bases


def _barrier(tensor, flat_base, ready_offset, stream) -> None:
    _run_compiled(
        compile_epoch_barrier(),
        (ptr_arg(tensor), fx.Int64(flat_base), fx.Int64(ready_offset), stream),
    )


class _AtomicCompressedRunner:
    def __init__(self, tp_group, config: atomic_compressed.Config) -> None:
        k = atomic_compressed
        self.config = config
        self.rank = int(tp_group.rank_in_group)
        self.device = torch.device(tp_group.device)
        self.partial_ready = config.m * (k.H + k.H // 32)
        self.reduced_ready = config.shard_rows * k.H
        # Each barrier-visible workspace carries its 8-byte epoch counter right
        # after its payload; `_barrier` addresses it by `ready_offset`.
        self.arena, views, self.comm, self.windows, bases = _allocate_registered(
            tp_group,
            self.rank,
            k.TP,
            self.device,
            (
                ((self.partial_ready + 8,), torch.uint8),
                ((self.reduced_ready + 8,), torch.uint8),
                ((config.shard_rows, k.H // 32), torch.uint8),
            ),
        )
        self.partial, self.reduced_payload, self.reduced_scale = views
        self.partial[self.partial_ready :].zero_()
        self.reduced_payload[self.reduced_ready :].zero_()
        self.output = torch.empty(
            (config.m, k.H), dtype=torch.bfloat16, device=self.device
        )
        (
            self.partial_flat_base,
            self.reduced_payload_base,
            self.reduced_scale_base,
        ) = bases
        shard_begin = self.rank * config.shard_rows
        self.reduced_shard = self.output[shard_begin : shard_begin + config.shard_rows]

    def __call__(
        self,
        *,
        stage2_args: tuple,
        stage2_kwargs: dict,
        shared_partial,
        ordinary_stage2,
        all_gather: bool = True,
    ):
        """Run Stage2 plus the TP collective.

        Returns the replicated ``[m, H]`` output when ``all_gather`` is set, and
        this rank's ``[m // TP, H]`` ReduceScatter shard otherwise.
        """
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
        if not all_gather:
            return self.reduced_shard
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


_RUNNER_TYPES = {
    atomic_compressed.Config: _AtomicCompressedRunner,
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
