# SPDX-License-Identifier: MIT
"""Production host runtime for communication-fused FlyDSL MoE."""

import csv
import math
import re
from dataclasses import MISSING, dataclass, fields
from functools import cache
from pathlib import Path

import flydsl.expr as fx
import torch
import torch.distributed._symmetric_memory as symm_mem
from mori.cco import Communicator

from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
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

_PEER_VMM_ALLOCATION_ALIGNMENT = 2 * 1024 * 1024
_MAX_ABS_ERROR = 1.0
_MAX_REL_L2_ERROR = 0.05
_ACT_TYPE = "ActivationType.Silu"
_DTYPE = "torch.bfloat16"
_Q_DTYPE_A = "torch.float8_e4m3fn"
_Q_DTYPE_W = "torch.float4_e2m1fn_x2"
_Q_TYPE = "QuantType.per_1x32"


@dataclass(frozen=True, slots=True)
class ShapeKey:
    gfx: str
    model_dim: int
    inter_dim: int
    experts: int
    topk: int
    tp: int
    cu_num: int | None = None
    act_type: str = _ACT_TYPE
    dtype: str = _DTYPE
    q_dtype_a: str = _Q_DTYPE_A
    q_dtype_w: str = _Q_DTYPE_W
    q_type: str = _Q_TYPE
    use_g1u1: int = 1
    doweight_stage1: int = 0

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
_CONFIG_NAME_PREFIX = "flydsl_comm_moe2_afp8_wfp4_bf16_"
_RUNNER_CACHE = {}


def _int_value(raw, name: str) -> int:
    numeric = float(raw)
    integer = int(numeric)
    if numeric != integer:
        raise ValueError(f"{name} must be an integer, got {raw!r}")
    return integer


def _mega_defaults() -> dict:
    defaults = {}
    for field in fields(gemm2_tp_megakernel.Gemm2TPMegakernelConfig):
        if field.name in ("shape", "m"):
            continue
        if field.default is MISSING:
            raise TypeError(f"missing megakernel default for {field.name}")
        defaults[field.name] = field.default
    return defaults


def config_name(config: PipelineConfig) -> str:
    prefix = _CONFIG_NAME_PREFIX
    if isinstance(config, gemm2_tp_atomic_pipeline.Gemm2TPAtomicPipelineConfig):
        return (
            f"{prefix}atomic_rs{config.reduce_scatter_grid}"
            f"_ag{config.all_gather_grid}"
        )
    if isinstance(config, gemm2_tp_window_pipeline.Gemm2TPWindowPipelineConfig):
        return (
            f"{prefix}window_t{config.tile_m}x{config.tile_n}x{config.tile_k}"
            f"_sbm{config.sort_block_m}_win{config.window}"
            f"_lw{config.local_workers}_rs{config.reduce_scatter_grid}"
            f"_ag{config.all_gather_grid}"
        )
    if not isinstance(config, gemm2_tp_megakernel.Gemm2TPMegakernelConfig):
        raise TypeError(f"unsupported comm_fused config {type(config)!r}")

    defaults = _mega_defaults()
    parts = [
        f"{prefix}t{config.tile_m}x{config.tile_n}x{config.tile_k}",
    ]
    numeric_tags = (
        ("sort_block_m", "sbm"),
        ("compute_groups", "cg"),
        ("block_threads", "bt"),
        ("vector_width", "v"),
        ("waves_per_eu", "w"),
        ("b_cache_modifier", "bnt"),
        ("local_load_cache_modifier", "ll"),
        ("remote_load_cache_modifier", "rl"),
        ("gather_load_cache_modifier", "gl"),
        ("remote_store_cache_modifier", "rs"),
        ("n_tile_cohort", "ntc"),
    )
    for field_name, tag in numeric_tags:
        value = getattr(config, field_name)
        if value != defaults[field_name]:
            parts.append(f"{tag}{value}")
    if config.route_store_scope != defaults["route_store_scope"]:
        parts.append(f"rts{config.route_store_scope}")
    if config.collective != defaults["collective"]:
        parts.append(
            {
                "rs_broadcast": "rsbcast",
                "rsag": "rsag",
            }[config.collective]
        )
    if config.service_groups != defaults["service_groups"]:
        parts.append(f"sg{config.service_groups}")
    if config.service_tile_group != defaults["service_tile_group"]:
        parts.append(f"stg{config.service_tile_group}")
    if config.producer_mode != defaults["producer_mode"]:
        parts.append(
            {
                "atomic_shared": "patomic",
                "routes_fp8_fixed": "pfp8",
            }[config.producer_mode]
        )
    if config.flat_producer_grid:
        parts.append("flat")
    return "_".join(parts)


def _parse_megakernel_name(name: str, shape: Gemm2TPShape, m: int):
    prefix = _CONFIG_NAME_PREFIX
    if not name.startswith(prefix):
        return None
    parts = name[len(prefix) :].split("_")
    tile = re.fullmatch(r"t(\d+)x(\d+)x(\d+)", parts.pop(0))
    if tile is None:
        raise ValueError(f"invalid megakernel tile in {name!r}")
    values = _mega_defaults()
    values.update(
        tile_m=int(tile.group(1)),
        tile_n=int(tile.group(2)),
        tile_k=int(tile.group(3)),
    )
    numeric_tags = {
        "sbm": "sort_block_m",
        "cg": "compute_groups",
        "bt": "block_threads",
        "v": "vector_width",
        "w": "waves_per_eu",
        "bnt": "b_cache_modifier",
        "ll": "local_load_cache_modifier",
        "rl": "remote_load_cache_modifier",
        "gl": "gather_load_cache_modifier",
        "rs": "remote_store_cache_modifier",
        "ntc": "n_tile_cohort",
        "sg": "service_groups",
        "stg": "service_tile_group",
    }
    collective = values["collective"]
    for part in parts:
        if part in ("direct", "rsbcast", "rsag"):
            if collective != values["collective"]:
                raise ValueError(f"duplicate collective in {name!r}")
            collective = {
                "direct": "direct",
                "rsbcast": "rs_broadcast",
                "rsag": "rsag",
            }[part]
        elif part == "patomic":
            values["producer_mode"] = "atomic_shared"
        elif part == "pfp8":
            values["producer_mode"] = "routes_fp8_fixed"
        elif part == "flat":
            values["flat_producer_grid"] = True
        elif part.startswith("rts"):
            values["route_store_scope"] = part[3:]
        else:
            for tag, field_name in sorted(
                numeric_tags.items(), key=lambda item: -len(item[0])
            ):
                if part.startswith(tag):
                    values[field_name] = _int_value(part[len(tag) :], field_name)
                    break
            else:
                raise ValueError(f"unknown megakernel option {part!r} in {name!r}")
    values["collective"] = collective
    return gemm2_tp_megakernel.Gemm2TPMegakernelConfig(shape=shape, m=m, **values)


def _config(row, shape: Gemm2TPShape) -> PipelineConfig:
    name = row["kernelName"]
    m = _int_value(row["token"], "token")
    atomic = re.fullmatch(
        rf"{re.escape(_CONFIG_NAME_PREFIX)}atomic_rs(\d+)_ag(\d+)", name
    )
    if atomic is not None:
        return gemm2_tp_atomic_pipeline.Gemm2TPAtomicPipelineConfig(
            shape, m, int(atomic.group(1)), int(atomic.group(2))
        )
    window = re.fullmatch(
        rf"{re.escape(_CONFIG_NAME_PREFIX)}window_"
        r"t(\d+)x(\d+)x(\d+)_sbm(\d+)_win(\d+)_lw(\d+)_rs(\d+)_ag(\d+)",
        name,
    )
    if window is not None:
        config = gemm2_tp_window_pipeline.Gemm2TPWindowPipelineConfig(
            shape, m, *(int(value) for value in window.groups())
        )
    else:
        config = _parse_megakernel_name(name, shape, m)
        if config is None:
            raise ValueError(f"unknown comm_fused kernelName {name!r}")
    block_m = _int_value(row["block_m"], "block_m")
    if config.sort_block_m != block_m:
        raise ValueError(
            f"kernelName sort_block_m={config.sort_block_m} does not match "
            f"CSV block_m={block_m}"
        )
    return config


def _optional_float(row, name: str) -> float | None:
    raw = row.get(name)
    if raw in (None, ""):
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def _row_is_accurate(row) -> bool:
    max_abs = _optional_float(row, "max_abs")
    rel_l2 = _optional_float(row, "rel_l2")
    return (max_abs is None or max_abs <= _MAX_ABS_ERROR) and (
        rel_l2 is None or rel_l2 <= _MAX_REL_L2_ERROR
    )


def _select_row(key, rows):
    accurate = [row for row in rows if _row_is_accurate(row)]
    if not accurate:
        raise ValueError(f"all comm_fused configs fail accuracy for {key}")
    if len(accurate) == 1:
        return accurate[0]
    measured = [
        (latency, row)
        for row in accurate
        if (latency := _optional_float(row, "us")) is not None
    ]
    if len(measured) != len(accurate):
        raise ValueError(
            f"duplicate comm_fused configs require measured 'us' for {key}"
        )
    return min(measured, key=lambda item: item[0])[1]


@cache
def _winner_table() -> dict[ShapeKey, dict[int, PipelineConfig]]:
    candidates = {}
    config_path = Path(AITER_CONFIGS.AITER_CONFIG_COMM_FUSED_MOE_FILE)
    with config_path.open(newline="") as file:
        for row in csv.DictReader(file):
            # Skip ordinary Stage2 + TP AllReduce fallback rows.
            if row["kernelName"].strip().lower() == "fallback":
                continue
            shape = ShapeKey(
                row["gfx"],
                int(row["model_dim"]),
                int(row["inter_dim"]),
                int(row["expert"]),
                int(row["topk"]),
                int(row["tp"]),
                int(row["cu_num"]),
                row["act_type"],
                row["dtype"],
                row["q_dtype_a"],
                row["q_dtype_w"],
                row["q_type"],
                _int_value(row["use_g1u1"], "use_g1u1"),
                _int_value(row["doweight_stage1"], "doweight_stage1"),
            )
            key = (shape, int(row["token"]))
            candidates.setdefault(key, []).append(row)
    table = {}
    for (shape, m), rows in candidates.items():
        row = _select_row((shape, m), rows)
        table.setdefault(shape, {})[m] = _config(row, shape.kernel_shape())
    return table


def winners_for(shape: ShapeKey) -> dict[int, PipelineConfig]:
    if shape.cu_num is None:
        shape = ShapeKey(
            shape.gfx,
            shape.model_dim,
            shape.inter_dim,
            shape.experts,
            shape.topk,
            shape.tp,
            get_cu_num(),
            shape.act_type,
            shape.dtype,
            shape.q_dtype_a,
            shape.q_dtype_w,
            shape.q_type,
            shape.use_g1u1,
            shape.doweight_stage1,
        )
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
    runtime_sort_block_m = int(kwargs["block_m"])
    if runtime_sort_block_m != config.sort_block_m:
        raise RuntimeError(
            "comm_fused sort_block_m does not match ordinary fused MoE: "
            f"config={config.sort_block_m}, runtime={runtime_sort_block_m}, "
            f"M={config.m}, shape={shape.tag}"
        )
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
        get_cu_num(),
    )
    key = (id(tp_group), shape)
    if key not in _RUNNER_CACHE:
        _RUNNER_CACHE[key] = _LazyRunners(tp_group, winners_for(shape))
    return _RUNNER_CACHE[key]
