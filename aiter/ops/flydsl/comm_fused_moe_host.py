# SPDX-License-Identifier: MIT
"""Production host runtime for communication-fused FlyDSL MoE."""

import csv
import math
import re
from dataclasses import MISSING, dataclass, fields, replace
from functools import cache
from pathlib import Path

import flydsl.expr as fx
import torch
import torch.distributed._symmetric_memory as symm_mem

from aiter.dist.device_communicators.rocm_version import get_rocm_version
from aiter.fused_moe import stage2_uses_route_reduce
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.flydsl.kernels.comm_fused_moe.gfx950.a8w4 import (
    atomic,
    megakernel,
    window,
)
from aiter.ops.flydsl.kernels.comm_fused_moe.gfx950.a8w4.collectives import (
    FLAT_VA_RANK_STRIDE,
    compile_epoch_barrier,
)
from aiter.ops.flydsl.kernels.comm_fused_moe.gfx950.a8w4.config import (
    SLOTS,
    AtomicConfig,
    MegakernelConfig,
    PipelineConfig,
    Shape,
    WindowConfig,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_arg

_PEER_VMM_ALLOCATION_ALIGNMENT = 2 * 1024 * 1024
_MIN_ROCM_VERSION = (7, 2)
_DEFAULT_ACT_TYPE = "ActivationType.Silu"
_DTYPE = "torch.bfloat16"
_Q_DTYPE_A = "torch.float8_e4m3fn"
_Q_DTYPE_W = "torch.float4_e2m1fn_x2"
_Q_TYPE = "QuantType.per_1x32"
_ADD_SHARED = True


@dataclass(frozen=True, slots=True)
class ShapeKey:
    gfx: str
    model_dim: int
    inter_dim: int
    experts: int
    topk: int
    tp: int
    cu_num: int | None = None
    act_type: str = _DEFAULT_ACT_TYPE
    dtype: str = _DTYPE
    q_dtype_a: str = _Q_DTYPE_A
    q_dtype_w: str = _Q_DTYPE_W
    q_type: str = _Q_TYPE
    use_g1u1: int = 1
    doweight_stage1: int = 0
    add_shared: bool | None = _ADD_SHARED
    comm: str = "ar"

    def kernel_shape(self) -> Shape:
        if self.add_shared is None:
            raise ValueError("kernel shape requires an explicit add_shared value")
        return Shape(
            self.model_dim,
            self.inter_dim,
            self.experts,
            self.topk,
            self.tp,
            self.add_shared,
        )


_CONFIG_NAME_PREFIX = "flydsl_comm_moe2_afp8_wfp4_bf16_"
_RUNNER_CACHE = {}


@cache
def _mori_communicator():
    from mori.cco import Communicator

    return Communicator


@cache
def is_flydsl_comm_fused_moe_available() -> bool:
    rocm_version = get_rocm_version()
    if rocm_version is None or rocm_version < _MIN_ROCM_VERSION:
        return False
    try:
        communicator = _mori_communicator()
    except (ImportError, OSError):
        return False
    return all(
        hasattr(communicator, name)
        for name in ("get_unique_id", "init", "register_external_window")
    )


def _int_value(raw, name: str) -> int:
    numeric = float(raw)
    integer = int(numeric)
    if numeric != integer:
        raise ValueError(f"{name} must be an integer, got {raw!r}")
    return integer


def _mega_defaults() -> dict:
    defaults = {}
    for field in fields(MegakernelConfig):
        if field.name in ("shape", "m"):
            continue
        if field.default is MISSING:
            raise TypeError(f"missing megakernel default for {field.name}")
        defaults[field.name] = field.default
    return defaults


def config_name(config: PipelineConfig) -> str:
    prefix = _CONFIG_NAME_PREFIX
    if isinstance(config, AtomicConfig):
        return (
            f"{prefix}atomic_rs{config.reduce_scatter_grid}"
            f"_ag{config.all_gather_grid}"
        )
    if isinstance(config, WindowConfig):
        return (
            f"{prefix}window_t{config.tile_m}x{config.tile_n}x{config.tile_k}"
            f"_sbm{config.sort_block_m}_win{config.window}"
            f"_lw{config.local_workers}_rs{config.reduce_scatter_grid}"
            f"_ag{config.all_gather_grid}"
        )
    if not isinstance(config, MegakernelConfig):
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
    if config.collective != defaults["collective"]:
        parts.append(
            {
                "rs_broadcast": "rsbcast",
                "rsag": "rsag",
                "rs": "rs",
            }[config.collective]
        )
    if config.service_groups != defaults["service_groups"]:
        parts.append(f"sg{config.service_groups}")
    if config.service_tile_group != defaults["service_tile_group"]:
        parts.append(f"stg{config.service_tile_group}")
    if config.producer_mode != defaults["producer_mode"]:
        parts.append("patomic")
    if config.flat_producer_grid:
        parts.append("flat")
    return "_".join(parts)


def _parse_megakernel_name(name: str, shape: Shape, m: int):
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
        if part in ("direct", "rsbcast", "rsag", "rs"):
            if collective != values["collective"]:
                raise ValueError(f"duplicate collective in {name!r}")
            collective = {
                "direct": "direct",
                "rsbcast": "rs_broadcast",
                "rsag": "rsag",
                "rs": "rs",
            }[part]
        elif part == "patomic":
            values["producer_mode"] = "atomic_shared"
        elif part == "flat":
            values["flat_producer_grid"] = True
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
    return MegakernelConfig(shape=shape, m=m, **values)


def _config(row, shape: Shape) -> PipelineConfig:
    name = row["comm_kernel_name"]
    m = _int_value(row["token"], "token")
    atomic = re.fullmatch(
        rf"{re.escape(_CONFIG_NAME_PREFIX)}atomic_rs(\d+)_ag(\d+)", name
    )
    if atomic is not None:
        return AtomicConfig(shape, m, int(atomic.group(1)), int(atomic.group(2)))
    window = re.fullmatch(
        rf"{re.escape(_CONFIG_NAME_PREFIX)}window_"
        r"t(\d+)x(\d+)x(\d+)_sbm(\d+)_win(\d+)_lw(\d+)_rs(\d+)_ag(\d+)",
        name,
    )
    if window is not None:
        config = WindowConfig(shape, m, *(int(value) for value in window.groups()))
    else:
        config = _parse_megakernel_name(name, shape, m)
        if config is None:
            raise ValueError(f"unknown comm_fused kernel name {name!r}")
    block_m = _int_value(row["block_m"], "block_m")
    if config.sort_block_m != block_m:
        raise ValueError(
            f"comm_kernel_name sort_block_m={config.sort_block_m} does not match "
            f"CSV block_m={block_m}"
        )
    return config


def _optional_float(row, name: str) -> float | None:
    raw = row.get(name)
    if raw in (None, ""):
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def _optional_bool(row, name: str, default: bool) -> bool:
    value = row.get(name)
    return default if value in (None, "", "nan") else bool(_int_value(value, name))


def _select_row(key, rows):
    if len(rows) == 1:
        return rows[0]
    measured = [
        (latency, row)
        for row in rows
        if (latency := _optional_float(row, "comm_stage2_tp_us")) is not None
    ]
    if len(measured) != len(rows):
        raise ValueError(
            "duplicate comm_fused configs require measured "
            f"'comm_stage2_tp_us' for {key}"
        )
    return min(measured, key=lambda item: item[0])[1]


@cache
def _winner_table() -> dict[ShapeKey, dict[int, PipelineConfig]]:
    candidates = {}
    config_path = Path(AITER_CONFIGS.AITER_CONFIG_FMOE_FILE)
    with config_path.open(newline="") as file:
        for row in csv.DictReader(file):
            name = (row.get("comm_kernel_name") or "").strip()
            if not name:
                continue
            shape = ShapeKey(
                row["gfx"],
                _int_value(row["model_dim"], "model_dim"),
                _int_value(row["inter_dim"], "inter_dim"),
                _int_value(row["expert"], "expert"),
                _int_value(row["topk"], "topk"),
                _int_value(row["comm_tp"], "comm_tp"),
                _int_value(row["cu_num"], "cu_num"),
                row["act_type"],
                row["dtype"],
                row["q_dtype_a"],
                row["q_dtype_w"],
                row["q_type"],
                _int_value(row["use_g1u1"], "use_g1u1"),
                _int_value(row["doweight_stage1"], "doweight_stage1"),
                _optional_bool(row, "comm_add_shared", _ADD_SHARED),
                (row.get("comm_mode") or "ar").strip() or "ar",
            )
            key = (shape, _int_value(row["token"], "token"))
            candidates.setdefault(key, []).append(row)
    table = {}
    for (shape, m), rows in candidates.items():
        row = _select_row((shape, m), rows)
        table.setdefault(shape, {})[m] = _config(row, shape.kernel_shape())
    return table


def winners_for(shape: ShapeKey) -> dict[int, PipelineConfig]:
    if not is_flydsl_comm_fused_moe_available():
        return {}
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
            shape.add_shared,
            shape.comm,
        )
    table = _winner_table()
    if shape.add_shared is None:
        matches = [
            configs
            for add_shared in (True, False)
            if (configs := table.get(replace(shape, add_shared=add_shared)))
        ]
        if len(matches) != 1:
            raise KeyError(
                "comm_fused layout must match exactly one shared-add contract, "
                f"got {len(matches)} for H={shape.model_dim}, "
                f"I={shape.inter_dim}, E={shape.experts}, topk={shape.topk}, "
                f"TP={shape.tp}"
            )
        return matches[0]
    try:
        return table[shape]
    except KeyError:
        raise KeyError(f"unsupported comm_fused shape {shape}") from None


def _symmetric(device, size: int) -> torch.Tensor:
    requested_bytes = int(size)
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
    workspace = _symmetric(device, total_bytes)
    tensors = tuple(
        workspace.narrow(0, offset, int(size)) for offset, size in zip(offsets, sizes)
    )
    return workspace, tensors, tuple(offsets)


def _register(tp_group, rank: int, tp: int, tensors):
    Communicator = _mori_communicator()
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
        ptr_arg(tensor),
        fx.Int64(flat_base),
        fx.Int64(ready_offset),
        stream,
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


class _RuntimeLayoutRunner:
    """Common model-runtime contract independent of collective type."""

    def _init_runtime_layout(self) -> None:
        self._runtime_shared_stage = None

    def output_rows_for(self, input_rows: int) -> int:
        output_rows, remainder = divmod(
            input_rows * int(self.output.shape[0]), self.config.m
        )
        if remainder:
            raise ValueError(
                f"runner layout maps bucket M={self.config.m} to "
                f"{self.output.shape[0]} output rows and cannot represent "
                f"input M={input_rows}"
            )
        return output_rows

    def prepare_padded_shared_partial(
        self, shared_partial: torch.Tensor, input_rows: int
    ) -> torch.Tensor:
        """Pad a full-layout shared contribution for this runner's bucket."""

        if self.output.shape[0] == self.config.m:
            padded = self.output
        else:
            padded = self._runtime_shared_stage
            if padded is None:
                padded = shared_partial.new_empty(
                    (self.config.m, *shared_partial.shape[1:])
                )
                self._runtime_shared_stage = padded
        padded[:input_rows].copy_(shared_partial)
        padded[input_rows:].zero_()
        return padded

    def prepare_shared_partial(self, shared_partial: torch.Tensor) -> torch.Tensor:
        return shared_partial


class _AtomicRunner(_RuntimeLayoutRunner):
    def __init__(
        self,
        tp_group,
        config: AtomicConfig,
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
        self._init_runtime_layout()
        self.comm, self.windows, (workspace_base,) = _register(
            tp_group, self.rank, shape.tp_size, (self.workspace,)
        )
        tp_group.barrier()
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
        config = self.config
        stage2_reduces_routes = stage2_uses_route_reduce(ordinary_stage2)
        add_shared = config.shape.add_shared and stage2_reduces_routes
        if not stage2_reduces_routes:
            if config.shape.add_shared:
                self.output.copy_(shared_partial)
            else:
                self.output.zero_()
        ordinary_stage2(
            *stage2_args[:6],
            self.output,
            *stage2_args[7:],
            **stage2_kwargs,
        )
        stream = torch.cuda.current_stream(self.device)
        _run_compiled(
            atomic.compile_quantize(config, add_shared),
            ptr_arg(self.output),
            ptr_arg(shared_partial if shared_partial is not None else self.output),
            ptr_arg(self.partial),
            stream,
        )
        _barrier(
            self.partial,
            self.partial_flat_base,
            self.partial_ready,
            config.shape.tp_size,
            stream,
        )
        _run_compiled(
            atomic.compile_reduce_scatter(config),
            fx.Int64(self.partial_flat_base),
            ptr_arg(self.reduced_shard),
            ptr_arg(self.reduced_payload),
            ptr_arg(self.reduced_scale),
            self.rank,
            stream,
        )
        _barrier(
            self.reduced_payload,
            self.reduced_payload_base,
            self.reduced_ready,
            config.shape.tp_size,
            stream,
        )
        _run_compiled(
            atomic.compile_all_gather(config),
            fx.Int64(self.reduced_payload_base),
            fx.Int64(self.reduced_scale_base),
            ptr_arg(self.output),
            self.rank,
            stream,
        )
        return self.output


class _MegakernelRunner(_RuntimeLayoutRunner):
    """Single-launch GEMM2 with per-N-tile TP collective services."""

    def __init__(
        self,
        tp_group,
        config: MegakernelConfig,
    ) -> None:
        shape = config.shape
        self.config = config
        self.rank = int(tp_group.rank_in_group)
        self.device = torch.device(tp_group.device)
        self.workspace = _symmetric(self.device, config.workspace_bytes)
        self.workspace.zero_()
        if config.collective == "rs":
            output_rows = config.m // shape.tp_size
        else:
            output_rows = config.m
        self.output = (
            self.workspace.narrow(0, config.output_offset, config.output_region_bytes)
            .view(torch.bfloat16)
            .view(output_rows, shape.model_dim)
        )
        self._init_runtime_layout()
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

        if not self.config.shape.add_shared:
            return self.output
        if not self.config.shared_bf16_partials:
            return shared_partial
        if self.config.producer_mode == "atomic_shared":
            return shared_partial
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
        stream = torch.cuda.current_stream(self.device)
        if not self.config.shape.add_shared:
            shared_partial = self.output
            if self.config.shared_bf16_partials:
                self.shared_partial_ptr = self.output.data_ptr()
                self.shared_partial_flat_base = (
                    self.workspace_flat_base + self.config.output_offset
                )
        if (
            self.config.shape.add_shared
            and self.config.shared_bf16_partials
            and self.config.producer_mode != "atomic_shared"
        ):
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
        common = _stage2_args(stage2_args, stage2_kwargs, self.config)
        _run_compiled(
            megakernel.compile_megakernel(self.config, self.rank),
            ptr_arg(self.workspace),
            ptr_arg(shared_partial),
            fx.Int64(self.shared_partial_flat_base),
            *common[:8],
            *common[9:],
            stream,
        )
        return self.output


class _WindowRunner(_RuntimeLayoutRunner):
    def __init__(
        self,
        tp_group,
        config: WindowConfig,
    ) -> None:
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
            for _ in range(SLOTS)
        )
        self.partial_ready = config.m * (config.window + config.window // 32)
        self.reduced_ready = config.shard_rows * config.window
        sizes = (
            ((self.partial_ready + 8 + 255) // 256 * 256,) * SLOTS
            + ((self.reduced_ready + 8 + 255) // 256 * 256,) * SLOTS
            + (config.shard_rows * (config.window // 32),) * SLOTS
        )
        self.workspace, tensors, offsets = _packed_symmetric(self.device, sizes)
        self.partials = tensors[:SLOTS]
        self.reduced_payloads = tensors[SLOTS : 2 * SLOTS]
        self.reduced_scales = tensors[2 * SLOTS :]
        for partial in self.partials:
            partial[self.partial_ready : self.partial_ready + 8].zero_()
        for payload in self.reduced_payloads:
            payload[self.reduced_ready : self.reduced_ready + 8].zero_()
        self.output = torch.empty(
            (config.m, shape.model_dim),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self._init_runtime_layout()
        self.comm, self.windows, (workspace_base,) = _register(
            tp_group, self.rank, shape.tp_size, (self.workspace,)
        )
        tp_group.barrier()
        bases = tuple(workspace_base + offset for offset in offsets)
        self.partial_bases = bases[:SLOTS]
        self.reduced_payload_bases = bases[SLOTS : 2 * SLOTS]
        self.reduced_scale_bases = bases[2 * SLOTS :]
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
        slot = phase % SLOTS
        shared = (
            shared_partial[:, phase * self.config.window :]
            if shared_partial is not None
            else self.output[:, phase * self.config.window :]
        )
        return (
            ptr_arg(self.routes[slot]),
            ptr_arg(self.partials[slot]),
            ptr_arg(shared),
        )

    def _collective_args(self, reduce_scatter, all_gather):
        reduce_slot = reduce_scatter % SLOTS
        gather_slot = all_gather % SLOTS
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
            window.compile_drain(
                self.config,
                local is not None,
                reduce_scatter is not None,
                all_gather is not None,
            ),
            *self._local_args(0 if local is None else local, shared_partial),
            *self._collective_args(
                0 if reduce_scatter is None else reduce_scatter,
                0 if all_gather is None else all_gather,
            ),
            self.rank,
            stream,
        )

    def __call__(
        self,
        *,
        stage2_args: tuple,
        stage2_kwargs: dict,
        shared_partial,
        ordinary_stage2,
    ):
        k = window
        config = self.config
        stream = torch.cuda.current_stream(self.device)
        common = _stage2_args(stage2_args, stage2_kwargs, config)
        _run_compiled(
            k.compile_compute(config, 0),
            ptr_arg(self.routes[0]),
            *common,
            stream,
        )
        for local in range(len(self.reduced_shards) - 1):
            reduce_scatter = local - 1
            all_gather = local - 2
            _run_compiled(
                k.compile_cycle(
                    config,
                    local + 1,
                ),
                ptr_arg(self.routes[(local + 1) % SLOTS]),
                *common,
                *self._local_args(local, shared_partial),
                *self._collective_args(max(reduce_scatter, 0), max(all_gather, 0)),
                self.rank,
                stream,
            )
            local_slot = local % SLOTS
            _barrier(
                self.partials[local_slot],
                self.partial_bases[local_slot],
                self.partial_ready,
                config.shape.tp_size,
                stream,
            )
            if reduce_scatter >= 0:
                reduce_slot = reduce_scatter % SLOTS
                _barrier(
                    self.reduced_payloads[reduce_slot],
                    self.reduced_payload_bases[reduce_slot],
                    self.reduced_ready,
                    config.shape.tp_size,
                    stream,
                )

        last = len(self.reduced_shards) - 1
        self._drain(last, last - 1, last - 2, shared_partial, stream)
        last_slot = last % SLOTS
        reduce_slot = (last - 1) % SLOTS
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
    AtomicConfig: _AtomicRunner,
    MegakernelConfig: _MegakernelRunner,
    WindowConfig: _WindowRunner,
}


def create_runner(tp_group, config: PipelineConfig):
    return _RUNNER_TYPES[type(config)](tp_group, config)


class _LazyRunners:
    def __init__(self, tp_group, configs: dict[int, PipelineConfig]) -> None:
        self.tp_group = tp_group
        self.configs = configs
        self.add_shared = next(iter(configs.values())).shape.add_shared
        self.instances = {}

    def __contains__(self, tokens: int) -> bool:
        return tokens in self.configs

    def __getitem__(self, tokens: int):
        if tokens not in self.instances:
            self.instances[tokens] = create_runner(self.tp_group, self.configs[tokens])
        return self.instances[tokens]


def create_flydsl_comm_fused_runners(
    *,
    tp_group,
    model_dim,
    inter_dim,
    experts,
    topk,
    act_type: str = _DEFAULT_ACT_TYPE,
    comm: str = "ar",
):
    if comm not in ("ar", "rs"):
        raise ValueError(f"comm must be 'ar' or 'rs', got {comm!r}")
    shape = ShapeKey(
        get_gfx_runtime(),
        model_dim,
        inter_dim,
        experts,
        topk,
        int(tp_group.world_size),
        act_type=act_type,
        add_shared=None,
        comm=comm,
    )
    configs = winners_for(shape)
    if not configs:
        raise KeyError(f"unsupported comm_fused shape {shape}")
    key = (id(tp_group), shape)
    if key not in _RUNNER_CACHE:
        _RUNNER_CACHE[key] = _LazyRunners(tp_group, configs)
    return _RUNNER_CACHE[key]
