# SPDX-License-Identifier: MIT
"""Production host runtime for communication-fused FlyDSL MoE.

Set ``AITER_COMM_FUSED_WINDOW_MXFP8_FALLBACK=1`` before runner creation to
temporarily replay Window AR with dynamic MXFP8 partial and reduced payloads.
"""

import csv
import json
import logging
import math
import os
from dataclasses import MISSING, dataclass, fields, replace
from functools import cache
from pathlib import Path

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import torch.distributed._symmetric_memory as symm_mem

from aiter.dist.device_communicators.rocm_version import get_rocm_version
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.flydsl.kernels.comm_fused_moe.gfx950.a8w4 import (
    direct,
    megakernel,
    window,
)
from aiter.ops.flydsl.kernels.comm_fused_moe.gfx950.a8w4.collectives import (
    FLAT_VA_RANK_STRIDE,
    compile_epoch_barrier,
    compile_epoch_barrier_pair,
)
from aiter.ops.flydsl.kernels.comm_fused_moe.gfx950.a8w4.config import (
    DIRECT_GRID_CAP,
    SLOTS,
    DirectConfig,
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
logger = logging.getLogger("aiter")
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
    act_type: str = _DEFAULT_ACT_TYPE
    dtype: str = _DTYPE
    q_dtype_a: str = _Q_DTYPE_A
    q_dtype_w: str = _Q_DTYPE_W
    q_type: str = _Q_TYPE
    use_g1u1: int = 1
    doweight_stage1: int = 0
    add_shared: bool = True
    comm: str = "ar"

    def kernel_shape(self) -> Shape:
        return Shape(
            self.model_dim,
            self.inter_dim,
            self.experts,
            self.topk,
            self.tp,
            self.add_shared,
        )


_CONFIG_NAME_PREFIX = "flydsl_comm_moe2_afp8_wfp4_bf16_"
_COMM_MODES = ("ar", "rs")
_RUNNER_CACHE = {}
_CONFIG_FAMILIES = {
    "direct": DirectConfig,
    "mega": MegakernelConfig,
    "window": WindowConfig,
}
_CONFIG_FAMILY_NAMES = {value: key for key, value in _CONFIG_FAMILIES.items()}
_DERIVED_CONFIG_FIELDS = {"shape", "m", "gather_output"}


def _window_mxfp8_fallback_enabled() -> bool:
    """Temporarily roll the complete Window AR payload path back to MXFP8."""

    value = os.environ.get("AITER_COMM_FUSED_WINDOW_MXFP8_FALLBACK", "0")
    if value not in ("0", "1"):
        raise ValueError(
            f"AITER_COMM_FUSED_WINDOW_MXFP8_FALLBACK must be 0 or 1, got {value!r}"
        )
    return value == "1"


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
    """Return a human-readable label; ``config_record`` is the config identity."""

    prefix = _CONFIG_NAME_PREFIX
    if isinstance(config, DirectConfig):
        name = f"{prefix}direct_sbm{config.sort_block_m}"
        if config.grid_cap != DIRECT_GRID_CAP:
            name += f"_g{config.grid_cap}"
        if config.vector_width != 8:
            name += f"_v{config.vector_width}"
        return name
    if isinstance(config, WindowConfig):
        name = (
            f"{prefix}window_t{config.tile_m}x{config.tile_n}x{config.tile_k}"
            f"_sbm{config.sort_block_m}_win{config.window}"
            f"_lw{config.local_workers}_rs{config.reduce_scatter_grid}"
            f"_ag{config.all_gather_grid}"
        )
        if not config.gather_output:
            name += "_sharded"
        if config.sorted_input:
            name += "_sorted"
        if config.drain_reduce_scatter_grid:
            name += f"_drs{config.drain_reduce_scatter_grid}"
        if config.drain_all_gather_grid:
            name += f"_dag{config.drain_all_gather_grid}"
        if config.local_load_cache_modifier != 2:
            name += f"_lrc{config.local_load_cache_modifier}"
        if config.local_rows_per_cta != 1:
            name += f"_lr{config.local_rows_per_cta}"
        if config.producer_spatial_partition:
            name += (
                f"_spart{config.producer_spatial_partition // 100}"
                f"x{config.producer_spatial_partition % 100}"
            )
        if config.producer_spatial_alt_partition:
            name += (
                f"to{config.producer_spatial_alt_partition // 100}"
                f"x{config.producer_spatial_alt_partition % 100}"
                f"gt{config.producer_spatial_switch_rows}"
            )
        if config.partial_payload_bits != 8:
            name += f"_ppb{config.partial_payload_bits}"
        if config.reduced_payload_bits != 8:
            name += f"_rpb{config.reduced_payload_bits}"
        if config.collective_order == "alternating":
            name += "_stagger"
        elif config.collective_order == "gather_first":
            name += "_gather_first"
        if config.reduce_scatter_load_cache_modifier != 2:
            name += f"_rsc{config.reduce_scatter_load_cache_modifier}"
        if config.all_gather_load_cache_modifier != 2:
            name += f"_agc{config.all_gather_load_cache_modifier}"
        return name
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
            }[config.collective]
        )
    if config.service_groups != defaults["service_groups"]:
        parts.append(f"sg{config.service_groups}")
    if config.service_tile_group != defaults["service_tile_group"]:
        parts.append(f"stg{config.service_tile_group}")
    if config.flat_producer_grid:
        parts.append("flat")
    if config.sorted_input:
        parts.append("sorted")
    if config.wide_partial_scales:
        parts.append("wideps")
    if config.coherent_direct_handoff:
        parts.append("coherent")
    return "_".join(parts)


def config_record(config: PipelineConfig) -> dict:
    """Return the structured, model-independent representation stored in CSV."""

    family = _CONFIG_FAMILY_NAMES.get(type(config))
    if family is None:
        raise TypeError(f"unsupported comm_fused config {type(config)!r}")
    params = {}
    for field in fields(config):
        if field.name in _DERIVED_CONFIG_FIELDS:
            continue
        value = getattr(config, field.name)
        if field.default is MISSING or value != field.default:
            params[field.name] = value
    return {"family": family, "params": params}


def _config(entry: dict, shape: Shape, m: int, mode: str) -> PipelineConfig:
    family = entry.get("family")
    config_type = _CONFIG_FAMILIES.get(family)
    if config_type is None:
        raise ValueError(f"unsupported comm_fused family {family!r}")
    params = entry.get("params")
    if not isinstance(params, dict):
        raise TypeError("comm_fused config params must be a JSON object")
    allowed = {field.name for field in fields(config_type)}.difference(
        _DERIVED_CONFIG_FIELDS
    )
    unknown = set(params).difference(allowed)
    if unknown:
        raise ValueError(f"unsupported {family} config parameters: {sorted(unknown)}")
    kwargs = dict(params)
    if config_type is WindowConfig:
        kwargs["gather_output"] = mode == "ar"
    return config_type(shape=shape, m=m, **kwargs)


def _check_block_m(config: PipelineConfig, row) -> None:
    block_m = _int_value(row["block_m"], "block_m")
    if config.sort_block_m != block_m:
        raise ValueError(
            f"comm_fused sort_block_m={config.sort_block_m} does not match "
            f"CSV block_m={block_m}"
        )


def _optional_float(row, name: str) -> float | None:
    raw = row.get(name)
    if raw in (None, ""):
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def _select_candidate(key, candidates):
    if len(candidates) == 1:
        return candidates[0]
    measured = [
        (latency, candidate)
        for candidate in candidates
        if (latency := _optional_float(candidate[1], "us")) is not None
    ]
    if len(measured) != len(candidates):
        raise ValueError(
            f"duplicate comm_fused configs require measured 'us' for {key}"
        )
    return min(measured, key=lambda item: item[0])[1]


def _comm_configs(row):
    raw = row.get("comm_fused_configs")
    if raw in (None, "", "nan", "0"):
        return {}
    configs = json.loads(raw)
    if not isinstance(configs, dict):
        raise TypeError("comm_fused_configs must be a JSON object")
    unknown = set(configs).difference(_COMM_MODES)
    if unknown:
        raise ValueError(f"unsupported comm_fused modes: {sorted(unknown)}")
    return configs


@cache
def _winner_table() -> dict[ShapeKey, dict[int, PipelineConfig]]:
    candidates = {}
    config_path = Path(AITER_CONFIGS.AITER_CONFIG_FMOE_FILE)
    with config_path.open(newline="") as file:
        for row in csv.DictReader(file):
            for mode, entry in _comm_configs(row).items():
                if not isinstance(entry, dict):
                    raise TypeError(
                        f"comm_fused_configs[{mode!r}] must be a JSON object"
                    )
                unknown = set(entry).difference(
                    {"family", "params", "tp", "add_shared", "us"}
                )
                if unknown:
                    raise ValueError(
                        f"unsupported comm_fused_configs[{mode!r}] fields: "
                        f"{sorted(unknown)}"
                    )
                tp = _int_value(entry["tp"], f"comm_fused_configs.{mode}.tp")
                add_shared = entry.get("add_shared")
                if not isinstance(add_shared, bool):
                    raise TypeError(
                        f"comm_fused_configs[{mode!r}].add_shared must be boolean"
                    )
                shape = ShapeKey(
                    row["gfx"],
                    _int_value(row["model_dim"], "model_dim"),
                    _int_value(row["inter_dim"], "inter_dim"),
                    _int_value(row["expert"], "expert"),
                    _int_value(row["topk"], "topk"),
                    tp,
                    _int_value(row["cu_num"], "cu_num"),
                    row["act_type"],
                    row["dtype"],
                    row["q_dtype_a"],
                    row["q_dtype_w"],
                    row["q_type"],
                    _int_value(row["use_g1u1"], "use_g1u1"),
                    _int_value(row["doweight_stage1"], "doweight_stage1"),
                    add_shared,
                    mode,
                )
                key = (shape, _int_value(row["token"], "token"))
                candidates.setdefault(key, []).append((row, entry))
    table = {}
    for (shape, m), entries in candidates.items():
        row, entry = _select_candidate((shape, m), entries)
        config = _config(entry, shape.kernel_shape(), m, shape.comm)
        _check_block_m(config, row)
        table.setdefault(shape, {})[m] = config
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
    try:
        return _winner_table()[shape]
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


@dataclass(slots=True)
class _WindowWorkspaceLayout:
    """Typed views and peer-flat addresses for one Window runner workspace."""

    storage: torch.Tensor
    phase_count: int
    partials: tuple[torch.Tensor, ...]
    reduced_payloads: tuple[torch.Tensor, ...]
    reduced_scales: tuple[torch.Tensor, ...]
    partial_ready: int
    reduced_ready: int
    partial_offsets: tuple[int, ...]
    reduced_payload_offsets: tuple[int, ...]
    reduced_scale_offsets: tuple[int, ...]
    partial_bases: tuple[int, ...] = ()
    reduced_payload_bases: tuple[int, ...] = ()
    reduced_scale_bases: tuple[int, ...] = ()

    @classmethod
    def allocate(cls, device, config: WindowConfig):
        phase_count = config.phase_count
        partial_ready = config.partial_epoch_offset
        reduced_ready = config.reduced_epoch_offset
        sizes = (config.partial_buffer_bytes,) * phase_count
        if config.gather_output:
            sizes += (config.reduced_buffer_bytes,) * phase_count
            sizes += (config.reduced_scale_bytes,) * phase_count

        storage, tensors, offsets = _packed_symmetric(device, sizes)
        partial_end = phase_count
        reduced_end = 2 * phase_count if config.gather_output else partial_end
        return cls(
            storage=storage,
            phase_count=phase_count,
            partials=tensors[:partial_end],
            reduced_payloads=tensors[partial_end:reduced_end],
            reduced_scales=tensors[reduced_end:],
            partial_ready=partial_ready,
            reduced_ready=reduced_ready,
            partial_offsets=offsets[:partial_end],
            reduced_payload_offsets=offsets[partial_end:reduced_end],
            reduced_scale_offsets=offsets[reduced_end:],
        )

    def bind_flat_base(self, flat_base: int) -> None:
        self.partial_bases = tuple(
            flat_base + offset for offset in self.partial_offsets
        )
        self.reduced_payload_bases = tuple(
            flat_base + offset for offset in self.reduced_payload_offsets
        )
        self.reduced_scale_bases = tuple(
            flat_base + offset for offset in self.reduced_scale_offsets
        )

    def clear_epochs(self) -> None:
        for partial in self.partials:
            partial[self.partial_ready : self.partial_ready + 8].zero_()
        for payload in self.reduced_payloads:
            payload[self.reduced_ready : self.reduced_ready + 8].zero_()


def _optional_ptr(tensor: torch.Tensor | None):
    if tensor is None:
        return flyc.from_c_void_p(fx.Uint8, 0)
    return ptr_arg(tensor)


@dataclass(frozen=True, slots=True)
class _WindowPhaseView:
    """Arguments owned by the local, reduce, and gather phases of one launch."""

    local_route: torch.Tensor | None
    local_partial: torch.Tensor | None
    shared: torch.Tensor | None
    reduce_partial_base: int | None
    reduced_shard: torch.Tensor | None
    reduced_payload: torch.Tensor | None
    reduced_scale: torch.Tensor | None
    gather_payload_base: int | None
    gather_scale_base: int | None
    gathered_output: torch.Tensor | None

    def launch_args(self) -> tuple:
        return (
            _optional_ptr(self.local_route),
            _optional_ptr(self.local_partial),
            _optional_ptr(self.shared),
            fx.Int64(self.reduce_partial_base or 0),
            _optional_ptr(self.reduced_shard),
            _optional_ptr(self.reduced_payload),
            _optional_ptr(self.reduced_scale),
            fx.Int64(self.gather_payload_base or 0),
            fx.Int64(self.gather_scale_base or 0),
            _optional_ptr(self.gathered_output),
        )


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


def _barrier_pair(
    first_tensor,
    first_flat_base,
    first_ready_offset,
    second_tensor,
    second_flat_base,
    second_ready_offset,
    tp_size,
    stream,
) -> None:
    _run_compiled(
        compile_epoch_barrier_pair(tp_size),
        ptr_arg(first_tensor),
        fx.Int64(first_flat_base),
        fx.Int64(first_ready_offset),
        ptr_arg(second_tensor),
        fx.Int64(second_flat_base),
        fx.Int64(second_ready_offset),
        stream,
    )


def _bind_inter_layout(runner, ordinary_stage2) -> None:
    keywords = getattr(ordinary_stage2, "keywords", None) or {}
    name = keywords.get("kernelName") or keywords.get("kernelName2") or ""
    sorted_input = "_moe2_layout_" in str(name)
    if runner.config.sorted_input != sorted_input:
        runner.config = replace(runner.config, sorted_input=sorted_input)


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

    @property
    def stage2_destination(self):
        return None

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
        self.output = (
            self.workspace.narrow(0, config.output_offset, config.payload_bytes)
            .view(torch.bfloat16)
            .view(config.output_rows, shape.model_dim)
        )
        self._runtime_shared_stage = None
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
        before_shared_add=None,
    ):
        _bind_inter_layout(self, ordinary_stage2)
        stream = torch.cuda.current_stream(self.device)
        if not self.config.shape.add_shared:
            shared_partial = self.output
            if self.config.shared_bf16_partials:
                self.shared_partial_ptr = self.output.data_ptr()
                self.shared_partial_flat_base = (
                    self.workspace_flat_base + self.config.output_offset
                )
        if self.config.shape.add_shared and self.config.shared_bf16_partials:
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
        if before_shared_add is not None:
            before_shared_add()
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
            torch.zeros(
                (
                    config.m,
                    shape.topk,
                    config.route_row_bytes,
                ),
                dtype=torch.uint8,
                device=self.device,
            )
            for _ in range(SLOTS)
        )
        self.workspace = _WindowWorkspaceLayout.allocate(self.device, config)
        self.workspace.clear_epochs()
        # The symmetric workspace comes from torch.empty(), so peer-visible epoch
        # words may contain stale values.  The host TP barrier below does not wait
        # for these asynchronous device clears; complete them before any rank can
        # register and observe the workspace.
        torch.cuda.synchronize(self.device)
        self.output = torch.empty(
            (config.output_rows, shape.model_dim),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self._runtime_shared_stage = None
        self.comm, self.windows, (workspace_base,) = _register(
            tp_group, self.rank, shape.tp_size, (self.workspace.storage,)
        )
        tp_group.barrier()
        self.workspace.bind_flat_base(workspace_base)
        shard_begin = self.rank * config.shard_rows if config.gather_output else 0
        self.reduced_output = self.output[shard_begin : shard_begin + config.shard_rows]

    def _shared_view(self, shared_partial):
        if self.config.gather_output:
            return shared_partial
        elif shared_partial is None:
            return None
        elif shared_partial.shape[0] == self.config.m:
            shard_begin = self.rank * self.config.shard_rows
            return shared_partial[shard_begin : shard_begin + self.config.shard_rows]
        elif shared_partial.shape[0] == self.config.shard_rows:
            return shared_partial
        else:
            raise ValueError(
                "window shared partial must use either the full or local-shard "
                f"row layout, got {tuple(shared_partial.shape)}"
            )

    def _phase_view(
        self,
        *,
        local: int | None,
        reduce_scatter: int | None,
        all_gather: int | None,
        shared_partial,
    ) -> _WindowPhaseView:
        workspace = self.workspace
        gather_output = self.config.gather_output
        return _WindowPhaseView(
            local_route=None if local is None else self.routes[local % SLOTS],
            local_partial=None if local is None else workspace.partials[local],
            shared=self._shared_view(shared_partial),
            reduce_partial_base=(
                None
                if reduce_scatter is None
                else workspace.partial_bases[reduce_scatter]
            ),
            reduced_shard=self.reduced_output if reduce_scatter is not None else None,
            reduced_payload=(
                workspace.reduced_payloads[reduce_scatter]
                if gather_output and reduce_scatter is not None
                else None
            ),
            reduced_scale=(
                workspace.reduced_scales[reduce_scatter]
                if gather_output and reduce_scatter is not None
                else None
            ),
            gather_payload_base=(
                workspace.reduced_payload_bases[all_gather]
                if gather_output and all_gather is not None
                else None
            ),
            gather_scale_base=(
                workspace.reduced_scale_bases[all_gather]
                if gather_output and all_gather is not None
                else None
            ),
            gathered_output=(
                self.output if gather_output and all_gather is not None else None
            ),
        )

    def _drain(self, local, reduce_scatter, all_gather, shared_partial, stream):
        phase = self._phase_view(
            local=local,
            reduce_scatter=reduce_scatter,
            all_gather=all_gather,
            shared_partial=shared_partial,
        )
        _run_compiled(
            window.compile_drain(
                self.config,
                local,
                reduce_scatter,
                all_gather,
            ),
            *phase.launch_args(),
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
        before_shared_add=None,
    ):
        k = window
        _bind_inter_layout(self, ordinary_stage2)
        config = self.config
        workspace = self.workspace
        phase_count = workspace.phase_count
        stream = torch.cuda.current_stream(self.device)
        common = _stage2_args(stage2_args, stage2_kwargs, config)
        _run_compiled(
            k.compile_compute(config, 0),
            ptr_arg(self.routes[0]),
            *common,
            stream,
        )
        shared_waited = before_shared_add is None or not config.shape.add_shared

        def wait_for_shared_if_used(*, local, reduce_scatter):
            nonlocal shared_waited
            uses_shared = (config.gather_output and local is not None) or (
                not config.gather_output and reduce_scatter is not None
            )
            if uses_shared and not shared_waited:
                before_shared_add()
                shared_waited = True

        for local in range(phase_count - 1):
            reduce_scatter = local - 1
            all_gather = local - 2 if config.gather_output else None
            wait_for_shared_if_used(
                local=local,
                reduce_scatter=reduce_scatter if reduce_scatter >= 0 else None,
            )
            phase = self._phase_view(
                local=local,
                reduce_scatter=reduce_scatter if reduce_scatter >= 0 else None,
                all_gather=(
                    all_gather if all_gather is not None and all_gather >= 0 else None
                ),
                shared_partial=shared_partial,
            )
            _run_compiled(
                k.compile_cycle(
                    config,
                    local + 1,
                ),
                ptr_arg(self.routes[(local + 1) % SLOTS]),
                *common,
                *phase.launch_args(),
                self.rank,
                stream,
            )
            if config.gather_output and reduce_scatter >= 0:
                _barrier_pair(
                    workspace.partials[local],
                    workspace.partial_bases[local],
                    workspace.partial_ready,
                    workspace.reduced_payloads[reduce_scatter],
                    workspace.reduced_payload_bases[reduce_scatter],
                    workspace.reduced_ready,
                    config.shape.tp_size,
                    stream,
                )
            else:
                _barrier(
                    workspace.partials[local],
                    workspace.partial_bases[local],
                    workspace.partial_ready,
                    config.shape.tp_size,
                    stream,
                )

        last = phase_count - 1
        wait_for_shared_if_used(local=last, reduce_scatter=last - 1)
        self._drain(
            last,
            last - 1,
            last - 2 if config.gather_output else None,
            shared_partial,
            stream,
        )
        if config.gather_output:
            _barrier_pair(
                workspace.partials[last],
                workspace.partial_bases[last],
                workspace.partial_ready,
                workspace.reduced_payloads[last - 1],
                workspace.reduced_payload_bases[last - 1],
                workspace.reduced_ready,
                config.shape.tp_size,
                stream,
            )
        else:
            _barrier(
                workspace.partials[last],
                workspace.partial_bases[last],
                workspace.partial_ready,
                config.shape.tp_size,
                stream,
            )
        self._drain(
            None,
            last,
            last - 1 if config.gather_output else None,
            shared_partial,
            stream,
        )
        if config.gather_output:
            _barrier(
                workspace.reduced_payloads[last],
                workspace.reduced_payload_bases[last],
                workspace.reduced_ready,
                config.shape.tp_size,
                stream,
            )
            self._drain(None, None, last, shared_partial, stream)
        # The workspace is private to this cached runner. Before either the
        # final AR payload or the final RS partial can be overwritten, the next
        # call reaches an earlier all-rank partial barrier. Stream ordering
        # proves that every rank has completed the preceding consumer by then,
        # so a trailing completion-only barrier is unnecessary.
        return self.output


class _DirectRunner(_RuntimeLayoutRunner):
    """Run ordinary Stage2 followed by direct collective postprocessing."""

    def __init__(
        self,
        tp_group,
        config: DirectConfig,
    ) -> None:
        self.config = config
        self.rank = int(tp_group.rank_in_group)
        self.device = torch.device(tp_group.device)
        self.workspace = _symmetric(self.device, config.workspace_bytes)
        self.workspace.zero_()
        self.partial = (
            self.workspace.narrow(0, 0, config.partial_bytes)
            .view(torch.bfloat16)
            .view(config.m, config.shape.model_dim)
        )
        self.stage2_output = self.partial
        self.output = torch.empty(
            (config.output_rows, config.shape.model_dim),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self._runtime_shared_stage = None
        self.comm, self.windows, (self.workspace_base,) = _register(
            tp_group,
            self.rank,
            config.shape.tp_size,
            (self.workspace,),
        )
        tp_group.barrier()
        self.reduce_scatter_add = direct.compile_reduce_scatter_add(config)

    @property
    def stage2_destination(self):
        return self.stage2_output

    def prepare_shared_partial(self, shared_partial: torch.Tensor) -> torch.Tensor:
        expected = (self.config.output_rows, self.config.shape.model_dim)
        if (
            tuple(shared_partial.shape) != expected
            or shared_partial.dtype != torch.bfloat16
            or shared_partial.device != self.device
            or not shared_partial.is_contiguous()
        ):
            raise ValueError(
                f"shared output must be contiguous BF16 {expected} on {self.device}"
            )
        return shared_partial

    def __call__(
        self,
        *,
        stage2_args: tuple,
        stage2_kwargs: dict,
        shared_partial: torch.Tensor | None,
        ordinary_stage2,
        before_shared_add=None,
    ) -> torch.Tensor:
        from aiter.fused_moe import stage2_uses_route_reduce

        if stage2_uses_route_reduce(ordinary_stage2):
            raise RuntimeError("direct collective requires an accumulating Stage2")
        if int(stage2_args[6].shape[0]) != self.config.m:
            raise RuntimeError(f"direct collective requires exact M={self.config.m}")
        if int(stage2_kwargs["block_m"]) != self.config.sort_block_m:
            raise RuntimeError("Stage2 sort block does not match direct collective")
        if shared_partial is None:
            raise RuntimeError("direct collective requires shared output")

        ordinary_stage2(
            *stage2_args[:6],
            self.partial,
            *stage2_args[7:],
            **stage2_kwargs,
        )
        if before_shared_add is not None:
            before_shared_add()
        _run_compiled(
            self.reduce_scatter_add,
            ptr_arg(self.workspace),
            fx.Int64(self.workspace_base),
            ptr_arg(self.output),
            ptr_arg(shared_partial),
            self.rank,
            torch.cuda.current_stream(self.device),
        )
        return self.output


_RUNNER_TYPES = {
    MegakernelConfig: _MegakernelRunner,
    WindowConfig: _WindowRunner,
    DirectConfig: _DirectRunner,
}


def create_runner(tp_group, config: PipelineConfig):
    runner_type = _RUNNER_TYPES[type(config)]
    return runner_type(tp_group, config)


class _LazyRunners:
    def __init__(
        self,
        tp_group,
        shape: ShapeKey,
        configs: dict[int, PipelineConfig],
    ) -> None:
        self.tp_group = tp_group
        self.shape = shape
        self.configs = configs
        self.instances = {}
        self.activated = set()

        # A Window runner creates a peer-VMM communicator and registers its
        # symmetric workspace.  Doing that lazily from the first DPA request is
        # unsafe: ranks may reach the first MoE layer at different times and
        # enter a different collective while one rank is still broadcasting
        # the communicator id.  Materialize only Window runners here, while
        # model initialization is ordered identically on every rank.  Kernel
        # compilation remains lazy.
        for tokens, config in sorted(configs.items()):
            if isinstance(config, WindowConfig):
                self.instances[tokens] = create_runner(self.tp_group, config)

    def __contains__(self, tokens: int) -> bool:
        return tokens in self.configs

    def __getitem__(self, tokens: int):
        config = self.configs[tokens]
        if tokens not in self.activated:
            if int(self.tp_group.rank_in_group) == 0:
                lookup_key = (
                    self.shape.gfx,
                    self.shape.cu_num,
                    tokens,
                    self.shape.model_dim,
                    self.shape.inter_dim,
                    self.shape.experts,
                    self.shape.topk,
                    self.shape.act_type,
                    self.shape.dtype,
                    self.shape.q_dtype_a,
                    self.shape.q_dtype_w,
                    self.shape.q_type,
                    self.shape.use_g1u1,
                    self.shape.doweight_stage1,
                )
                logger.info(
                    "[comm-fused-moe] activate kernel=%s for %s",
                    config_name(config),
                    lookup_key,
                )
            self.activated.add(tokens)
        if tokens not in self.instances:
            self.instances[tokens] = create_runner(self.tp_group, config)
        return self.instances[tokens]

    def output_rows_for(self, bucket: int, input_rows: int) -> int:
        config = self.configs[bucket]
        if getattr(config, "requires_exact_m", False) and input_rows != config.m:
            raise ValueError(f"{type(config).__name__} requires exact M={config.m}")
        output_rows, remainder = divmod(input_rows * config.output_rows, config.m)
        if remainder:
            raise ValueError(
                f"runner layout maps bucket M={config.m} to "
                f"{config.output_rows} output rows and cannot represent "
                f"input M={input_rows}"
            )
        return output_rows


def create_flydsl_comm_fused_runners(
    *,
    tp_group,
    model_dim,
    inter_dim,
    experts,
    topk,
    comm: str = "ar",
    add_shared: bool = True,
):
    if comm not in _COMM_MODES:
        raise ValueError(f"comm must be one of {_COMM_MODES}, got {comm!r}")
    shape = ShapeKey(
        get_gfx_runtime(),
        model_dim,
        inter_dim,
        experts,
        topk,
        int(tp_group.world_size),
        add_shared=add_shared,
        comm=comm,
    )
    configs = winners_for(shape)
    if not configs:
        raise KeyError(f"unsupported comm_fused shape {shape}")
    mxfp8_fallback = _window_mxfp8_fallback_enabled()
    rewrite_window_payloads = any(
        isinstance(config, WindowConfig)
        and config.gather_output
        and mxfp8_fallback
        and (config.partial_payload_bits != 8 or config.reduced_payload_bits != 8)
        for config in configs.values()
    )
    if rewrite_window_payloads:
        configs = {
            tokens: (
                replace(
                    config,
                    partial_payload_bits=8,
                    reduced_payload_bits=8,
                )
                if isinstance(config, WindowConfig) and config.gather_output
                else config
            )
            for tokens, config in configs.items()
        }
    key = (
        id(tp_group),
        shape,
        mxfp8_fallback,
    )
    if key not in _RUNNER_CACHE:
        _RUNNER_CACHE[key] = _LazyRunners(tp_group, shape, configs)
    return _RUNNER_CACHE[key]
