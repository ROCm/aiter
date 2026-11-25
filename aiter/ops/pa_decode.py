# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Prepared paged decode with a fixed ASM or FlyDSL backend.

The dispatcher uses host geometry and optional caller-owned CPU lengths. It
never reads GPU lengths back. Prepare outside capture, then refresh metadata
once per scheduling step and decode any number of layers sharing that step.
The caller orders accesses across streams; concurrent calls need separate plans.

The backend and reducer choice are fixed at preparation/capture. Updating GPU
lengths does not change a captured graph's backend. Select another prepared
plan/graph outside replay when a host policy chooses another backend.

Automatic selection is calibrated on MI355X with 256 compute units for BF16
Q4/GQA16/D128, page16, per-token FP8 KV. Measurements include one metadata
refresh and three independent KV layers per step; table-only ASM also includes
its GPU CSR conversion. Caller-supplied CPU lengths select the backend once at
preparation, within the bounded shape and length domain below.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cache
from numbers import Integral, Real
from typing import Any


@dataclass(frozen=True)
class PAContextStats:
    batch: int
    minimum: int
    maximum: int
    total: int
    used_pages: int

    @classmethod
    def from_lengths(cls, lengths: Sequence[int], page_size: int):
        # Do not implicitly coerce tensor scalars: int(cuda_tensor) synchronizes.
        if hasattr(lengths, "device"):
            raise TypeError(
                "context_lengths_cpu must be a host integer sequence, not a tensor"
            )
        values = tuple(lengths)
        if not values or any(
            isinstance(n, bool) or not isinstance(n, Integral) for n in values
        ):
            raise TypeError("context_lengths_cpu must contain host integers")
        if any(n < 0 for n in values) or page_size < 1:
            raise ValueError(
                "context lengths must be nonnegative and page_size positive"
            )
        return cls(
            len(values),
            int(min(values)),
            int(max(values)),
            int(sum(values)),
            int(sum((n + page_size - 1) // page_size for n in values)),
        )


@dataclass(frozen=True)
class PADecodeShape:
    arch: str
    batch: int
    query_length: int
    query_heads: int
    kv_heads: int
    head_dim: int
    page_size: int
    query_dtype: str
    kv_dtype: str
    asm_layout: bool
    per_token_scales: bool
    sliding_window: int = 0
    has_query_scale: bool = False
    has_alibi: bool = False
    has_sinks: bool = False
    mask_unowned_kv: bool = True
    softmax_scale: float | None = None
    compute_units: int | None = None


def _default_asm_scale(scale, head_dim: int) -> bool:
    # asm_pa.cu currently launches with log2(e)/sqrt(D), ignoring its scale
    # argument. Permit the FP32 spelling of this default, never another scale.
    return scale is None or (
        isinstance(scale, Real)
        and math.isclose(scale, head_dim**-0.5, rel_tol=1e-7, abs_tol=0.0)
    )


def _target_q4_shape(shape: PADecodeShape) -> bool:
    return (
        shape.arch == "gfx950"
        and shape.compute_units == 256
        and 1 <= shape.batch <= 32
        and shape.query_length == 4
        and shape.query_heads == 16
        and shape.kv_heads == 1
        and shape.head_dim == 128
        and shape.page_size == 16
        and shape.query_dtype == "bfloat16"
        and shape.kv_dtype == "float8_e4m3fn"
        and shape.asm_layout
        and shape.per_token_scales
        and shape.sliding_window <= 0
        and not shape.has_query_scale
        and not shape.has_alibi
        and not shape.has_sinks
        and _default_asm_scale(shape.softmax_scale, shape.head_dim)
    )


def _asm_supported(shape: PADecodeShape) -> bool:
    # ASM's existing contract requires finite unowned cache bytes/scales.
    return _target_q4_shape(shape) and not shape.mask_unowned_kv


def _compute_matches(shape, compute_type):
    # The pure selector consumes a CPU dtype name, never a tensor or its value.
    return compute_type is None or (
        isinstance(compute_type, str)
        and compute_type.removeprefix("torch.") == shape.kv_dtype
    )


def _within_calibration_domain(stats):
    return (
        stats is not None
        and stats.minimum >= 4
        and stats.maximum <= 768 * 1024
        and stats.total <= 6 * 1024 * 1024
    )


def select_pa_decode_backend(
    shape: PADecodeShape,
    stats: PAContextStats | None = None,
    *,
    backend: str = "auto",
    has_packed_csr: bool = False,
    max_partitions: int | None = None,
    workgroup_budget: int | None = None,
    compute_type: str | None = None,
) -> str:
    """Select using CPU metadata only; forced unsupported ASM calls raise.

    MI355X 256-CU Q4 thresholds were checked on real, uniform, and held-out
    lengths using complete three-layer steps. Existing packed CSR avoids
    conversion; a table-only call includes both CSR kernels in refresh.
    compute_type is a CPU dtype name for this pure helper. This decision is
    fixed in the prepared plan and a captured graph does not reselect it.
    A missing host length hint never triggers a GPU readback or automatic ASM.
    """
    if backend not in ("auto", "asm", "flydsl"):
        raise ValueError("backend must be 'auto', 'asm', or 'flydsl'")
    if stats is not None and stats.batch != shape.batch:
        raise ValueError("CPU context length count must match the query batch")
    if type(has_packed_csr) is not bool:
        raise TypeError("has_packed_csr must be a host bool")
    overrides = (
        max_partitions is not None
        or workgroup_budget is not None
        or not _compute_matches(shape, compute_type)
    )
    if backend == "asm" and overrides:
        raise ValueError(
            "ASM cannot honor FlyDSL max_partitions/workgroup_budget or a different compute_type"
        )
    supported = _asm_supported(shape) and (
        stats is None or stats.minimum >= shape.query_length
    )
    if backend == "asm" and not supported:
        raise ValueError(
            "ASM requires 256-CU gfx950 page16 BF16 Q4/GQA16/D128, per-token FP8, "
            "dense causal attention, default 1/sqrt(D) scale, "
            "contiguous packed caches and finite unowned KV"
        )
    if backend == "asm":
        return "asm"
    if (
        backend == "flydsl"
        or overrides
        or not supported
        or not _within_calibration_domain(stats)
    ):
        return "flydsl"
    if has_packed_csr and shape.batch in (24, 32) and stats.total >= 3 * 1024 * 1024:
        return "asm"
    if not has_packed_csr and shape.batch == 32 and stats.total >= 4 * 1024 * 1024:
        return "asm"
    return "flydsl"


def _fly_workgroup_budget(
    shape, stats, *, max_partitions=None, workgroup_budget=None, compute_type=None
):
    """Honor explicit FlyDSL choices; tune only the measured full-device scope."""
    if workgroup_budget is not None:
        return workgroup_budget
    if (
        max_partitions is not None
        or not _compute_matches(shape, compute_type)
        or not _target_q4_shape(shape)
        or not _within_calibration_domain(stats)
        or stats.batch != shape.batch
        or shape.batch not in (4, 8, 16, 24, 32)
        or stats.total < shape.batch * 32768
        or (shape.batch == 4 and (stats.maximum < 65536 or stats.total < 262144))
    ):
        return None
    return 512


def _import_fly_planner():
    from aiter.ops.flydsl.pa_decode import plan_pa_decode

    return plan_pa_decode


def _resolve_fly_planner(shape, stats, requested_backend, *, asm_fallback_allowed):
    """Return the Fly planner, or None for an eligible missing-package fallback.

    Only absence of the optional top-level flydsl package is recoverable here.
    A broken compiler install, another missing module, and planner/launch errors
    must remain visible; no execution error is used to choose another backend.
    """
    try:
        return _import_fly_planner()
    except ModuleNotFoundError as error:
        if error.name != "flydsl":
            raise
        if requested_backend == "auto" and asm_fallback_allowed and stats is not None:
            try:
                select_pa_decode_backend(shape, stats, backend="asm")
            except ValueError:
                pass
            else:
                return None
        raise ModuleNotFoundError(
            "FlyDSL is required for this PA configuration; install the AITER "
            "requirements.txt FlyDSL dependency (flydsl==0.5.0 at this revision)",
            name="flydsl",
        ) from error


@dataclass
class PADecodePlan:
    backend: str
    shape: PADecodeShape
    context_lengths: Any
    block_tables: Any
    context_stats: PAContextStats | None
    tensor_contract: tuple
    compute_type: Any = None
    max_context_length: int | None = None
    fly_workgroup_budget: int | None = None
    fly_plan: Any = None
    scratch: tuple = ()
    qo_indptr: Any = None
    kv_indptr: Any = None
    kv_indices: Any = None
    owns_csr: bool = False
    metadata: tuple = ()
    merge_waves: int = 8
    refreshed: bool = False
    _metadata_addresses: tuple = field(default_factory=tuple, repr=False)
    _step_addresses: tuple = field(default_factory=tuple, repr=False)
    _step_contract: tuple = field(default_factory=tuple, repr=False)


def _contract(tensor):
    return (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device)


def _check_step_tensors(context_lengths, block_tables, device):
    import torch

    if (
        context_lengths.device != device
        or block_tables.device != device
        or context_lengths.dtype != torch.int32
        or block_tables.dtype != torch.int32
        or context_lengths.ndim != 1
        or block_tables.ndim != 2
        or block_tables.shape[0] != context_lengths.numel()
        or not context_lengths.is_contiguous()
        or not block_tables.is_contiguous()
    ):
        raise ValueError(
            "lengths and block_tables must be contiguous int32 [B] and [B,C] on the query device"
        )


def _check_csr(kv_indptr, kv_indices, batch, device, stats):
    import torch

    if (kv_indptr is None) != (kv_indices is None):
        raise ValueError("provide both kv_indptr and kv_indices, or neither")
    if kv_indptr is None:
        return
    if (
        kv_indptr.shape != (batch + 1,)
        or kv_indices.ndim != 1
        or any(
            t.device != device or t.dtype != torch.int32 or not t.is_contiguous()
            for t in (kv_indptr, kv_indices)
        )
    ):
        raise ValueError(
            "packed CSR requires contiguous int32 [B+1] indptr and [capacity] indices"
        )
    if stats is not None and kv_indices.numel() < stats.used_pages:
        raise ValueError("kv_indices capacity is smaller than the host used-page count")


def _per_token_scales(key_scale, value_scale, key_cache):
    import torch

    shape = (key_cache.shape[0], key_cache.shape[1], key_cache.shape[3])
    return all(
        t is not None
        and t.dtype == torch.float32
        and t.device == key_cache.device
        and tuple(t.shape) in (shape, shape + (1,))
        and t.is_contiguous()
        for t in (key_scale, value_scale)
    )


def prepare_pa_decode_plan(
    query,
    key_cache,
    value_cache,
    context_lengths,
    block_tables,
    query_length: int,
    *,
    backend: str = "auto",
    context_lengths_cpu: Sequence[int] | None = None,
    kv_indptr=None,
    kv_indices=None,
    key_scale=None,
    value_scale=None,
    compute_type=None,
    query_scale=None,
    alibi_slopes=None,
    sinks=None,
    sliding_window: int = 0,
    mask_unowned_kv: bool = True,
    softmax_scale: float | None = None,
    max_context_length: int | None = None,
    max_partitions: int | None = None,
    workgroup_budget: int | None = None,
) -> PADecodePlan:
    """Allocate and refresh a fixed-backend plan outside graph capture.

    CPU lengths, when supplied, must describe the same scheduling step as the
    GPU vector. This association and valid used page IDs are caller contracts;
    checking them must not synchronize the device. The optional packed CSR must
    encode exactly ceil(context_lengths/16) used pages per row in logical order.
    Capacity padding is permitted and is never used as a live page count.
    ASM callers must keep every context at least query_length tokens long.
    max_context_length, when supplied, must bound every later refresh/replay.
    Pass softmax_scale to prepare when it is not 1/sqrt(D), so auto retains
    FlyDSL. ASM's native launcher only supports the default scale; decode also
    rejects another scale even when the caller omitted this preparation hint.
    If the optional flydsl package is absent, auto uses ASM only for its full
    supported contract and host length hints. Explicit FlyDSL requests and unsupported inputs report
    the missing dependency; compiler or execution errors never trigger fallback.

    Without supplied CSR, ASM refresh packs block_tables on the GPU before its
    native planner. Include both conversion kernels in complete-step timing.
    A plan may move between streams with caller-provided wait/event ordering,
    including default-stream preparation followed by side-stream capture.
    Do not concurrently reuse its metadata or scratch across streams/graphs.
    """
    import torch

    from aiter.jit.utils.chip_info import get_gfx_runtime

    if type(query_length) is not int or query_length < 1:
        raise ValueError("query_length must be a positive host integer")
    if softmax_scale is not None and not isinstance(softmax_scale, Real):
        raise TypeError("softmax_scale must be a host real scalar")
    if compute_type is not None and not isinstance(compute_type, torch.dtype):
        raise TypeError("compute_type must be a torch dtype")
    if query.device.type != "cuda" or query.ndim != 3 or key_cache.ndim != 5:
        raise ValueError("query must be CUDA [B*Q,H,D], key_cache must be packed 5D")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("prepare_pa_decode_plan must run outside graph capture")
    batch, remainder = divmod(query.shape[0], query_length)
    if remainder or batch < 1 or context_lengths.numel() != batch:
        raise ValueError(
            "query rows must equal context_lengths count times query_length"
        )
    _check_step_tensors(context_lengths, block_tables, query.device)
    if key_cache.device != query.device or value_cache.device != query.device:
        raise ValueError("Q/K/V must be on the same device")
    page_size, kv_heads = key_cache.shape[3], key_cache.shape[1]
    if kv_heads < 1 or query.shape[1] % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    stats = (
        None
        if context_lengths_cpu is None
        else PAContextStats.from_lengths(context_lengths_cpu, page_size)
    )
    if stats is not None and stats.maximum > block_tables.shape[1] * page_size:
        raise ValueError("CPU context lengths exceed block-table capacity")
    if max_context_length is not None and (
        type(max_context_length) is not int
        or not 0 <= max_context_length <= block_tables.shape[1] * page_size
    ):
        raise ValueError(
            "max_context_length must be a host integer within block-table capacity"
        )
    if (
        stats is not None
        and max_context_length is not None
        and stats.maximum > max_context_length
    ):
        raise ValueError("CPU context lengths exceed max_context_length")
    properties = torch.cuda.get_device_properties(query.device)
    compute_units = getattr(properties, "multi_processor_count", None)
    compute_name = (
        None if compute_type is None else str(compute_type).removeprefix("torch.")
    )
    _check_csr(kv_indptr, kv_indices, batch, query.device, stats)
    shape = PADecodeShape(
        get_gfx_runtime(),
        batch,
        query_length,
        query.shape[1],
        kv_heads,
        query.shape[2],
        page_size,
        str(query.dtype).removeprefix("torch."),
        str(key_cache.dtype).removeprefix("torch."),
        query.is_contiguous()
        and key_cache.is_contiguous()
        and value_cache.is_contiguous()
        and key_cache.shape[1:] == (1, 8, 16, 16)
        and value_cache.shape == (key_cache.shape[0], 1, 1, 128, 16)
        and value_cache.dtype == key_cache.dtype,
        _per_token_scales(key_scale, value_scale, key_cache),
        sliding_window,
        query_scale is not None,
        alibi_slopes is not None,
        sinks is not None,
        mask_unowned_kv,
        softmax_scale,
        compute_units,
    )
    selected = select_pa_decode_backend(
        shape,
        stats,
        backend=backend,
        has_packed_csr=kv_indptr is not None,
        max_partitions=max_partitions,
        workgroup_budget=workgroup_budget,
        compute_type=compute_name,
    )
    fly_planner = None
    if selected == "flydsl":
        fly_planner = _resolve_fly_planner(
            shape,
            stats,
            backend,
            asm_fallback_allowed=(
                _compute_matches(shape, compute_name)
                and max_partitions is None
                and workgroup_budget is None
            ),
        )
        if fly_planner is None:
            selected = "asm"
    if selected == "asm" and compute_type not in (None, key_cache.dtype):
        raise ValueError("ASM compute_type must match the FP8 KV cache")
    plan = PADecodePlan(
        selected,
        shape,
        context_lengths,
        block_tables,
        stats,
        (_contract(query), _contract(key_cache), _contract(value_cache)),
        compute_type,
        max_context_length,
    )
    plan._step_addresses = (context_lengths.data_ptr(), block_tables.data_ptr())
    plan._step_contract = (_contract(context_lengths), _contract(block_tables))
    if selected == "flydsl":
        plan.fly_workgroup_budget = _fly_workgroup_budget(
            shape,
            stats,
            max_partitions=max_partitions,
            workgroup_budget=workgroup_budget,
            compute_type=compute_name,
        )
        plan.fly_plan = fly_planner(
            context_lengths,
            kv_heads,
            max_partitions=max_partitions,
            workgroup_budget=plan.fly_workgroup_budget,
            sliding_window=sliding_window,
            query_length=query_length,
        )
        rows = query_length * query.shape[1] // kv_heads
        scratch_shape = (kv_heads, plan.fly_plan.capacity, rows)
        plan.scratch = (
            torch.empty(scratch_shape, dtype=torch.float32, device=query.device),
            torch.empty(scratch_shape, dtype=torch.float32, device=query.device),
            torch.empty(
                scratch_shape + (query.shape[2],),
                dtype=query.dtype,
                device=query.device,
            ),
        )
        plan.refreshed = True
    else:
        from aiter.ops import attention

        if max_partitions is not None or workgroup_budget is not None:
            raise ValueError(
                "ASM uses the native full-CU planner; FlyDSL budget overrides are unsupported"
            )
        plan.owns_csr = kv_indptr is None
        plan.qo_indptr = (
            torch.arange(batch + 1, dtype=torch.int32, device=query.device)
            * query_length
        )
        plan.kv_indptr = (
            torch.empty(batch + 1, dtype=torch.int32, device=query.device)
            if plan.owns_csr
            else kv_indptr
        )
        plan.kv_indices = (
            torch.empty(block_tables.numel(), dtype=torch.int32, device=query.device)
            if plan.owns_csr
            else kv_indices
        )
        plan.metadata = tuple(
            torch.empty(size, dtype=dtype, device=query.device)
            for size, dtype in attention.get_pa_metadata_info_v1(batch, kv_heads)
        )
        partial_rows = plan.metadata[5].numel() * query_length
        plan.scratch = (
            torch.empty(
                (partial_rows, 1, query.shape[1], query.shape[2]),
                dtype=torch.bfloat16,
                device=query.device,
            ),
            torch.empty(
                (partial_rows, 1, query.shape[1], 1),
                dtype=torch.float32,
                device=query.device,
            ),
        )
        if stats is not None:
            plan.merge_waves = attention._pa_ps_reduce_waves(
                batch, stats.used_pages, compute_units, compact_partials=True
            )
        plan._metadata_addresses = tuple(
            t.data_ptr() for t in (plan.kv_indptr, plan.kv_indices)
        )
        refresh_pa_decode_plan(plan)
    return plan


@cache
def _csr_kernels():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def indptr_kernel(Lengths, Indptr, B: tl.constexpr, BLOCK: tl.constexpr):
        rows = tl.arange(0, BLOCK)
        lengths = tl.load(Lengths + rows, rows < B, other=0)
        counts = (lengths + 15) // 16
        ends = tl.cumsum(counts, 0)
        tl.store(Indptr, 0)
        tl.store(Indptr + rows + 1, ends, rows < B)

    @triton.jit
    def indices_kernel(
        Table,
        Indptr,
        Indices,
        STRIDE: tl.constexpr,
        COLS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        start = tl.load(Indptr + row)
        count = tl.load(Indptr + row + 1) - start
        valid = (cols < count) & (cols < COLS)
        pages = tl.load(Table + row.to(tl.int64) * STRIDE + cols, valid, other=0)
        tl.store(Indices + start.to(tl.int64) + cols, pages, valid)

    return indptr_kernel, indices_kernel


def refresh_pa_decode_plan(plan: PADecodePlan) -> None:
    """Refresh the plan's bound GPU lengths/tables in place on the current stream.

    Caller-supplied CSR must already be refreshed and ordered on this stream.
    Captured replay must retain the same tensor addresses and update their
    contents in place. This function does not reselect the backend/reducer.
    The caller must order this stream after prior users of the plan and before
    its next decode. This function neither synchronizes nor inserts events.
    """
    if (
        plan.context_lengths.data_ptr(),
        plan.block_tables.data_ptr(),
    ) != plan._step_addresses or (
        _contract(plan.context_lengths),
        _contract(plan.block_tables),
    ) != plan._step_contract:
        raise ValueError("step tensor geometry/addresses changed; prepare a new plan")
    plan.refreshed = False
    if plan.backend == "flydsl":
        from aiter.ops.flydsl.pa_decode import plan_pa_decode

        plan_pa_decode(
            plan.context_lengths,
            plan.shape.kv_heads,
            query_length=plan.shape.query_length,
            sliding_window=plan.shape.sliding_window,
            plan=plan.fly_plan,
        )
    else:
        from aiter.ops import attention

        if (
            tuple(t.data_ptr() for t in (plan.kv_indptr, plan.kv_indices))
            != plan._metadata_addresses
        ):
            raise ValueError("plan CSR addresses changed; prepare a new plan")
        if plan.owns_csr:
            indptr_kernel, indices_kernel = _csr_kernels()
            batch, cols = plan.block_tables.shape
            indptr_kernel[(1,)](
                plan.context_lengths,
                plan.kv_indptr,
                B=batch,
                BLOCK=1 << (batch - 1).bit_length(),
            )
            if cols:
                indices_kernel[(batch, (cols + 255) // 256)](
                    plan.block_tables,
                    plan.kv_indptr,
                    plan.kv_indices,
                    STRIDE=plan.block_tables.stride(0),
                    COLS=cols,
                    BLOCK=256,
                )
        attention.get_pa_metadata_v1(
            plan.qo_indptr,
            plan.kv_indptr,
            plan.context_lengths,
            16,
            1,
            True,
            *plan.metadata,
            kv_granularity=16,
            block_size=16,
            max_seqlen_qo=4,
            uni_seqlen_qo=4,
            fast_mode=True,
        )
    plan.refreshed = True


def pa_decode(
    output,
    query,
    key_cache,
    value_cache,
    context_lengths,
    block_tables,
    softmax_scale: float,
    query_length: int,
    *,
    plan: PADecodePlan,
    key_scale=None,
    value_scale=None,
    query_scale=None,
    alibi_slopes=None,
    sinks=None,
):
    """Write output using a prepared/refreshed plan, without host synchronization.

    Return the same output tensor. The ASM plan owns BF16 split outputs and
    FP32 partial LSE. Its reducer does not materialize a final LSE.
    This call never silently switches backend after validation or launch errors.
    """
    if not isinstance(plan, PADecodePlan) or not plan.refreshed:
        raise ValueError("a prepared and refreshed PADecodePlan is required")
    if plan.backend == "asm" and (
        softmax_scale is None
        or not _default_asm_scale(softmax_scale, plan.shape.head_dim)
    ):
        raise ValueError(
            "ASM only supports softmax_scale=1/sqrt(head_dim); prepare a FlyDSL plan for another scale"
        )
    if (
        _contract(query),
        _contract(key_cache),
        _contract(value_cache),
    ) != plan.tensor_contract:
        raise ValueError("query/cache geometry changed; prepare a new plan")
    if (
        (context_lengths.data_ptr(), block_tables.data_ptr()) != plan._step_addresses
        or (_contract(context_lengths), _contract(block_tables)) != plan._step_contract
        or query_length != plan.shape.query_length
    ):
        raise ValueError(
            "decode must use the plan's bound step metadata and query length"
        )
    if (
        output.shape != query.shape
        or output.dtype != query.dtype
        or output.device != query.device
    ):
        raise ValueError("output must match query shape, dtype and device")
    flags = (query_scale is not None, alibi_slopes is not None, sinks is not None)
    if flags != (
        plan.shape.has_query_scale,
        plan.shape.has_alibi,
        plan.shape.has_sinks,
    ):
        raise ValueError("decode features changed; prepare a new plan")
    if plan.backend == "flydsl":
        from aiter.ops.flydsl.pa_decode import pa_decode as flydsl_decode

        flydsl_decode(
            output,
            query,
            key_cache,
            value_cache,
            context_lengths,
            block_tables,
            softmax_scale,
            query_length,
            compute_type=plan.compute_type,
            query_scale=query_scale,
            key_scale=key_scale,
            value_scale=value_scale,
            exp_sums=plan.scratch[0],
            max_logits=plan.scratch[1],
            temporary_output=plan.scratch[2],
            alibi_slopes=alibi_slopes,
            sinks=sinks,
            sliding_window=plan.shape.sliding_window,
            work_plan=plan.fly_plan,
            max_context_length=plan.max_context_length,
            mask_unowned_kv=plan.shape.mask_unowned_kv,
        )
    else:
        from aiter.ops import attention
        from aiter.ops.enum import QuantType

        if not output.is_contiguous() or not _per_token_scales(
            key_scale, value_scale, key_cache
        ):
            raise ValueError(
                "ASM output/scales must retain the contiguous per-token contract"
            )
        output_start = output.data_ptr()
        output_end = output_start + output.numel() * output.element_size()
        for tensor in (query, key_cache, value_cache, key_scale, value_scale):
            start = tensor.data_ptr()
            end = start + tensor.numel() * tensor.element_size()
            if max(output_start, start) < min(output_end, end):
                raise ValueError("ASM output must not overlap Q/K/V or their scales")
        _, work_indptr, work_info, reduce_indptr, final_map, partial_map = plan.metadata
        partial, partial_lse = plan.scratch
        attention.pa_ps_fwd_asm(
            query,
            key_cache,
            value_cache,
            plan.kv_indptr,
            plan.kv_indices,
            context_lengths,
            softmax_scale,
            query_length,
            key_scale,
            value_scale,
            output,
            plan.qo_indptr,
            work_indptr,
            work_info,
            partial,
            partial_lse,
            mask=1,
            high_precision=1,
            quant_type=QuantType.per_Token,
            # Retain the calibrated kernel when local CSV adds Q4 variants.
            kernelName="gfx950_ZN5aiter44pa_bf16_pertokenFp8_gqa16_1tg_4w_mtp_msk1_psE",
        )
        attention._attention_ps_reduce_asm(
            partial,
            partial_lse,
            reduce_indptr,
            final_map,
            partial_map,
            query_length,
            output,
            None,
            plan.merge_waves,
        )
    return output
