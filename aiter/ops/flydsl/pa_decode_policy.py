# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 FlyDSL Project Contributors

"""Host-only partition selection for paged decode, without autotuning.

Selection uses tensor geometry and device properties. It never benchmarks a
candidate, reads device context lengths, or loads an autotuning configuration.
The conservative gfx1250 bounds below are cost-policy constants, not hardware
limits. Other architectures retain the original planner defaults.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from .kernels.pa_decode_plan import PADecodePlan

_KV_TILE = 256
_QUERY_TILE = 16
_CLUSTER_LIMIT = 16
_MAX_SERIAL_TILES = 4
_MAX_TASK_WORK_BYTES = 1024 * 1024
_FP8_PARALLEL_SERIAL_TILES = 8


@dataclass(frozen=True)
class PADecodePartitionConfig:
    max_partitions: int
    workgroup_budget: int
    reason: str


def select_pa_decode_partition_config(
    *,
    arch: str,
    num_compute_units: int,
    num_seqs: int,
    num_kv_heads: int,
    query_group_size: int,
    head_dim: int,
    query_length: int,
    kv_dtype: str,
    max_context_length: int,
    max_partitions: int | None = None,
    sliding_window: int = 0,
    has_sinks: bool = False,
    output_layout_matches: bool = True,
) -> PADecodePartitionConfig:
    """Choose a cap and budget from a host-known scheduling bound.

    ``max_partitions`` remains the caller's upper limit. A smaller selected cap
    enables the existing cluster merger only when enough compute tasks remain
    and each task's serial KV work is bounded. Query rows within one CTA do not
    count as additional independent tasks.

    The estimate assumes nonempty rows of the scheduling-bound length. Mixed
    lengths still use the original GPU planner's proportional apportionment;
    the bound must cover every future length used with this plan.
    """
    for name, value in (
        ("num_compute_units", num_compute_units),
        ("num_seqs", num_seqs),
        ("num_kv_heads", num_kv_heads),
        ("query_group_size", query_group_size),
        ("head_dim", head_dim),
        ("query_length", query_length),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(max_context_length) is not int or max_context_length < 0:
        raise ValueError("max_context_length must be a nonnegative integer")
    if type(sliding_window) is not int:
        raise ValueError("sliding_window must be an integer")
    if kv_dtype not in ("fp8", "bf16"):
        raise ValueError("kv_dtype must be fp8 or bf16")
    limit = num_compute_units if max_partitions is None else max_partitions
    if type(limit) is not int or not 1 <= limit <= num_compute_units:
        raise ValueError("max_partitions must be in [1, num_compute_units]")
    budget = 2 * num_compute_units

    def original(reason):
        return PADecodePartitionConfig(limit, budget, reason)

    if arch.split(":", 1)[0] != "gfx1250":
        return original("original architecture policy")
    if (
        has_sinks
        or not output_layout_matches
        or sliding_window > 0
        or head_dim > 512
        or (head_dim != 64 and head_dim % 128)
        or limit == 1
    ):
        return original("original merge or window policy")

    rows = num_seqs * num_kv_heads
    if rows > num_compute_units * 15 // 16:
        return original("cluster residency limit")
    tiles = max(1, (max_context_length + _KV_TILE - 1) // _KV_TILE)
    capacity = min(
        num_seqs * limit,
        max(num_seqs, (budget + num_kv_heads - 1) // num_kv_heads),
    )
    bound = min(limit, capacity, tiles)
    if 2 <= bound <= _CLUSTER_LIMIT and num_seqs * bound <= max(
        capacity, (budget + num_kv_heads - 1) // num_kv_heads
    ):
        return original("existing cluster merge")

    # Retain at least one independent compute CTA per CU at the host bound.
    minimum = (num_compute_units + rows - 1) // rows
    minimum = 1 << (minimum - 1).bit_length()
    if minimum > min(limit, _CLUSTER_LIMIT):
        return original("insufficient compute parallelism for a low cap")

    query_tiles = (query_length * query_group_size + _QUERY_TILE - 1) // _QUERY_TILE

    def serial_fallback(reason):
        # Long FP8 single-query-tile tasks can use more independent CTAs to
        # overlap their KV loads. Keep the reducer and caller cap: widening a
        # cluster would change the compute/merge tradeoff being estimated.
        # This is an experimental geometry rule, not a bandwidth guarantee.
        default_parts = min(limit, tiles, max(1, capacity // num_seqs))
        default_serial = (tiles + default_parts - 1) // default_parts
        parallel_budget = 4 * num_compute_units
        parallel_capacity = min(
            num_seqs * limit,
            max(num_seqs, -(-parallel_budget // num_kv_heads)),
        )
        if (
            kv_dtype == "fp8"
            and query_tiles == 1
            and default_serial >= _FP8_PARALLEL_SERIAL_TILES
            and bound > _CLUSTER_LIMIT
            and min(limit, parallel_capacity, tiles) > _CLUSTER_LIMIT
            and parallel_capacity > capacity
        ):
            return PADecodePartitionConfig(
                limit,
                parallel_budget,
                "more independent reducer tasks for long FP8 single-tile work",
            )
        return original(reason)

    # One FP8 query tile has cheap partials; a wider cluster can expose more KV
    # parallelism. More query tiles and BF16 favor fewer partials to merge.
    preferred = 16 if kv_dtype == "fp8" and query_tiles == 1 else 8
    partitions = max(minimum, preferred)
    while partitions > limit or rows * partitions > 4 * num_compute_units:
        partitions //= 2
    if partitions < minimum:
        return original("cluster grid exceeds the work budget")

    serial_tiles = (tiles + partitions - 1) // partitions
    if serial_tiles > _MAX_SERIAL_TILES:
        return serial_fallback("too much serial KV work per low-cap task")
    # K and V bytes, weighted by the number of query matrix tiles that consume
    # them. FP8 scales are omitted from this compute-work proxy, not from the
    # benchmark's separately reported logical bandwidth.
    element_bytes = 1 if kv_dtype == "fp8" else 2
    task_work = serial_tiles * _KV_TILE * head_dim * 2 * element_bytes * query_tiles
    if task_work > _MAX_TASK_WORK_BYTES:
        return serial_fallback("too much KV/query work per low-cap task")
    return PADecodePartitionConfig(
        partitions,
        max(budget, rows * partitions),
        "bounded cluster merge with sufficient compute parallelism",
    )


def prepare_pa_decode_plan_with_policy(
    query: "torch.Tensor",
    key_cache: "torch.Tensor",
    value_cache: "torch.Tensor",
    context_lengths: "torch.Tensor",
    block_tables: "torch.Tensor",
    query_length: int,
    *,
    output: "torch.Tensor | None" = None,
    sinks: "torch.Tensor | None" = None,
    sliding_window: int = 0,
    max_partitions: int | None = None,
    max_batch_size: int | None = None,
    max_context_length: int | None = None,
) -> "PADecodePlan":
    """Build a reusable plan using geometry only, before graph capture.

    Use this instead of the tuning preparation API. Neither tuning environment
    variables nor saved tuning files affect this function. Allocate scratch
    for the returned ``plan.capacity`` and pass the plan to ``pa_decode``.

    Pass the real output when it has a different layout from query; omitting it
    assumes equal strides. ``max_context_length`` is page-rounded, as in decode;
    when omitted the table capacity supplies the conservative scheduling bound.
    Refresh with ``plan_pa_decode(..., plan=plan)`` when lengths change. If the
    scheduling bound grows, prepare a new plan and recapture its decode graph.
    """
    import torch

    from aiter.jit.utils.chip_info import get_gfx_runtime

    from .kernels.pa_decode_plan import plan_pa_decode

    if query.ndim != 3 or context_lengths.ndim != 1 or block_tables.ndim != 2:
        raise ValueError(
            "query, context_lengths and block_tables must be 3D, 1D and 2D"
        )
    if key_cache.ndim != 5:
        raise NotImplementedError("key_cache must use the packed 5D layout")
    if value_cache.ndim not in (4, 5):
        raise ValueError("value_cache must be 4D or 5D")
    if type(query_length) is not int or query_length < 1:
        raise ValueError("query_length must be a positive integer")
    batch = context_lengths.numel()
    heads, block_size = key_cache.shape[1], key_cache.shape[3]
    if heads < 1 or query.shape[1] % heads:
        raise ValueError("query heads must be divisible by positive KV heads")
    if query.shape[0] != batch * query_length or block_tables.shape[0] != batch:
        raise ValueError(
            "query/table rows do not match context_lengths and query_length"
        )
    if query.device.type != "cuda":
        raise ValueError("query must be on a GPU")
    for name, tensor in (
        ("key_cache", key_cache),
        ("value_cache", value_cache),
        ("context_lengths", context_lengths),
        ("block_tables", block_tables),
        ("output", output),
        ("sinks", sinks),
    ):
        if tensor is not None and tensor.device != query.device:
            raise ValueError(f"{name} must be on the query device")
    if output is not None and (
        output.shape != query.shape or output.dtype != query.dtype
    ):
        raise ValueError("output must match the query shape and dtype")
    context_bound = block_tables.shape[1] * block_size
    if max_context_length is not None:
        if (
            type(max_context_length) is not int
            or not 0 <= max_context_length <= context_bound
        ):
            raise ValueError("max_context_length must be within block-table capacity")
        context_bound = (max_context_length + block_size - 1) // block_size * block_size
    kv_dtype = "bf16" if key_cache.dtype == torch.bfloat16 else "fp8"
    if key_cache.dtype not in (
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    ):
        raise ValueError("key_cache must be BF16 or FP8")
    with torch.cuda.device(query.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "prepare_pa_decode_plan_with_policy must precede capture"
            )
        properties = torch.cuda.get_device_properties(query.device)
        config = select_pa_decode_partition_config(
            arch=get_gfx_runtime(),
            num_compute_units=properties.multi_processor_count,
            num_seqs=batch,
            num_kv_heads=heads,
            query_group_size=query.shape[1] // heads,
            head_dim=query.shape[2],
            query_length=query_length,
            kv_dtype=kv_dtype,
            max_context_length=context_bound,
            max_partitions=max_partitions,
            sliding_window=sliding_window,
            has_sinks=sinks is not None,
            output_layout_matches=output is None or output.stride() == query.stride(),
        )
        return plan_pa_decode(
            context_lengths,
            heads,
            max_partitions=config.max_partitions,
            workgroup_budget=config.workgroup_budget,
            max_batch_size=max_batch_size,
            sliding_window=sliding_window,
            query_length=query_length,
        )
