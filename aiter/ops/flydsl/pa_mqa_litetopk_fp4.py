# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from itertools import pairwise
from typing import NamedTuple

import torch

from .kernels.mqa_logits.pa_mqa_litetopk_fp4_common import (
    FP4_LITETOPK_SUPPORTED_TOPKS,
)

DEFAULT_TOPK = 512
DEFAULT_SAMPLE_LEN = 8192
DEFAULT_NUM_BUCKETS = 256
DEFAULT_REFRESH_EVERY = 64
DEFAULT_MERGE_CAP = 196_608
_SCAN_BLOCK_K = 512
_SCAN_NUM_WARPS = 8

STATUS_OK = 0
STATUS_CANDIDATE_OVERFLOW = 1 << 0
STATUS_UNDERFILLED = 1 << 1
STATUS_NONFINITE = 1 << 2
STATUS_INVALID_INDEX = 1 << 3
STATUS_BAD_CERTIFICATE = 1 << 4
STATUS_SELECTOR_FAILURE = 1 << 5


class FP4LiteTopKWorkspace(NamedTuple):
    sample_logits: torch.Tensor
    sample_ends: torch.Tensor
    scan_cta_info: torch.Tensor
    origin: torch.Tensor
    inv_delta: torch.Tensor
    threshold: torch.Tensor
    histogram: torch.Tensor
    candidate_values: torch.Tensor
    candidate_indices: torch.Tensor
    candidate_counts: torch.Tensor
    page_errors: torch.Tensor
    score_errors: torch.Tensor
    select_starts: torch.Tensor
    select_ends: torch.Tensor
    selected_slots: torch.Tensor
    selected_values: torch.Tensor
    output_counts: torch.Tensor
    status: torch.Tensor
    status_ok: torch.Tensor
    selector_workspace: torch.Tensor
    row_capacity: int
    topk: int
    sample_len: int
    num_buckets: int
    merge_cap: int
    stream_id: int


class FP4LiteTopKResult(NamedTuple):
    values: torch.Tensor
    raw_indices: torch.Tensor
    physical_indices: torch.Tensor
    counts: torch.Tensor
    candidate_counts: torch.Tensor
    status: torch.Tensor


def fp4_litetopk_workspace_nbytes(workspace: FP4LiteTopKWorkspace) -> int:
    """Return the bytes owned by a LiteTopK workspace."""
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in workspace
        if isinstance(tensor, torch.Tensor)
    )


def fp4_litetopk_workspace_size(
    rows: int,
    *,
    topk: int = DEFAULT_TOPK,
    sample_len: int = DEFAULT_SAMPLE_LEN,
    num_buckets: int = DEFAULT_NUM_BUCKETS,
    merge_cap: int = DEFAULT_MERGE_CAP,
) -> int:
    """Return required workspace bytes without allocating device memory."""
    if (
        rows <= 0
        or topk <= 0
        or sample_len < topk
        or num_buckets <= 1
        or merge_cap < topk
    ):
        raise ValueError("invalid LiteTopK workspace dimensions")
    from aiter.ops.topk import topk_ob_workspace_size

    row_bytes = (
        sample_len * 4
        + 6 * 4
        + num_buckets * 4
        + merge_cap * (4 + 4)
        + topk * (4 + 4)
        + 11 * 4
    )
    return (
        row_bytes * rows
        + 4
        + max(1, int(topk_ob_workspace_size(rows, merge_cap, topk, False)))
    )


def _tensor_byte_range(tensor: torch.Tensor) -> tuple[int, int]:
    start = tensor.data_ptr()
    return start, start + tensor.numel() * tensor.element_size()


def _byte_ranges_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _same_byte_range(left: torch.Tensor, right: torch.Tensor) -> bool:
    return _tensor_byte_range(left) == _tensor_byte_range(right)


def _workspace_tensors(workspace: FP4LiteTopKWorkspace) -> tuple[torch.Tensor, ...]:
    return tuple(item for item in workspace if isinstance(item, torch.Tensor))


def _validate_pairwise_disjoint(
    tensors: tuple[torch.Tensor, ...], *, message: str
) -> None:
    ranges = sorted(_tensor_byte_range(tensor) for tensor in tensors)
    if any(_byte_ranges_overlap(left, right) for left, right in pairwise(ranges)):
        raise ValueError(message)


def _validate_workspace(
    workspace: FP4LiteTopKWorkspace,
    *,
    rows: int,
    device: torch.device,
    stream: torch.cuda.Stream,
    topk: int,
    sample_len: int,
    num_buckets: int,
    merge_cap: int,
) -> tuple[tuple[torch.Tensor, tuple[int, int]], ...]:
    if (
        workspace.row_capacity < rows
        or workspace.topk != topk
        or workspace.sample_len != sample_len
        or workspace.num_buckets != num_buckets
        or workspace.merge_cap != merge_cap
        or workspace.stream_id != stream.cuda_stream
    ):
        raise ValueError("workspace is incompatible with this LiteTopK call")
    capacity = workspace.row_capacity
    expected = (
        (workspace.sample_logits, torch.float32, (capacity, sample_len)),
        (workspace.sample_ends, torch.int32, (capacity,)),
        (workspace.scan_cta_info, torch.int32, (capacity, 6)),
        (workspace.origin, torch.float32, (capacity,)),
        (workspace.inv_delta, torch.float32, (capacity,)),
        (workspace.threshold, torch.int32, (capacity,)),
        (workspace.histogram, torch.int32, (capacity, num_buckets)),
        (workspace.candidate_values, torch.float32, (capacity, merge_cap)),
        (workspace.candidate_indices, torch.int32, (capacity, merge_cap)),
        (workspace.candidate_counts, torch.int32, (capacity,)),
        (workspace.page_errors, torch.int32, (capacity,)),
        (workspace.score_errors, torch.int32, (capacity,)),
        (workspace.select_starts, torch.int32, (capacity,)),
        (workspace.select_ends, torch.int32, (capacity,)),
        (workspace.selected_slots, torch.int32, (capacity, topk)),
        (workspace.selected_values, torch.float32, (capacity, topk)),
        (workspace.output_counts, torch.int32, (capacity,)),
        (workspace.status, torch.int32, (capacity,)),
        (workspace.status_ok, torch.int32, (1,)),
    )
    if any(
        tensor.dtype != dtype
        or tuple(tensor.shape) != shape
        or tensor.device != device
        or not tensor.is_contiguous()
        for tensor, dtype, shape in expected
    ):
        raise ValueError("workspace tensors have incompatible metadata")
    from aiter.ops.topk import topk_ob_workspace_size

    required_selector_bytes = topk_ob_workspace_size(rows, merge_cap, topk, False)
    if (
        workspace.selector_workspace.dtype != torch.uint8
        or workspace.selector_workspace.device != device
        or not workspace.selector_workspace.is_contiguous()
        or workspace.selector_workspace.numel() < required_selector_bytes
    ):
        raise ValueError("selector_workspace is too small or has incompatible metadata")
    workspace_tensors = _workspace_tensors(workspace)
    workspace_ranges = tuple(
        (tensor, _tensor_byte_range(tensor)) for tensor in workspace_tensors
    )
    sorted_ranges = sorted(byte_range for _, byte_range in workspace_ranges)
    if any(
        _byte_ranges_overlap(left, right) for left, right in pairwise(sorted_ranges)
    ):
        raise ValueError("LiteTopK workspace tensors must not overlap")
    return workspace_ranges


def _validate_result(
    out: FP4LiteTopKResult,
    workspace: FP4LiteTopKWorkspace,
    *,
    rows: int,
    topk: int,
    device: torch.device,
    inputs: tuple[torch.Tensor, ...] = (),
    workspace_ranges: tuple[tuple[torch.Tensor, tuple[int, int]], ...] | None = None,
) -> None:
    expected = (
        (out.values, torch.float32, (rows, topk)),
        (out.raw_indices, torch.int32, (rows, topk)),
        (out.physical_indices, torch.int32, (rows, topk)),
        (out.counts, torch.int32, (rows,)),
        (out.candidate_counts, torch.int32, (rows,)),
        (out.status, torch.int32, (rows,)),
    )
    if any(
        tensor.dtype != dtype
        or tensor.shape != shape
        or tensor.device != device
        or not tensor.is_contiguous()
        for tensor, dtype, shape in expected
    ):
        raise ValueError("LiteTopK output tensors have incompatible metadata")
    if (
        out.counts.data_ptr() != workspace.output_counts[:rows].data_ptr()
        or out.candidate_counts.data_ptr()
        != workspace.candidate_counts[:rows].data_ptr()
        or out.status.data_ptr() != workspace.status[:rows].data_ptr()
    ):
        raise ValueError("candidate_counts and status outputs must alias the workspace")
    output_tensors = tuple(out)
    output_ranges = tuple(
        (output, _tensor_byte_range(output)) for output in output_tensors
    )
    sorted_output_ranges = sorted(byte_range for _, byte_range in output_ranges)
    if any(
        _byte_ranges_overlap(left, right)
        for left, right in pairwise(sorted_output_ranges)
    ):
        raise ValueError("LiteTopK output tensors must not overlap")

    allowed_workspace_aliases = (
        (out.values, workspace.selected_values, workspace.selected_values[:rows]),
        (out.counts, workspace.output_counts, workspace.output_counts[:rows]),
        (
            out.candidate_counts,
            workspace.candidate_counts,
            workspace.candidate_counts[:rows],
        ),
        (out.status, workspace.status, workspace.status[:rows]),
    )
    if workspace_ranges is None:
        workspace_ranges = tuple(
            (scratch, _tensor_byte_range(scratch))
            for scratch in _workspace_tensors(workspace)
        )
    allowed_workspace_ranges = tuple(
        (candidate, owner, _tensor_byte_range(view))
        for candidate, owner, view in allowed_workspace_aliases
    )
    input_ranges = tuple(_tensor_byte_range(input_tensor) for input_tensor in inputs)
    for output, output_range in output_ranges:
        allowed_owner = next(
            (
                owner
                for candidate, owner, view_range in allowed_workspace_ranges
                if candidate is output and output_range == view_range
            ),
            None,
        )
        for scratch, scratch_range in workspace_ranges:
            if scratch is allowed_owner:
                continue
            if _byte_ranges_overlap(output_range, scratch_range):
                raise ValueError("LiteTopK outputs must not overlap workspace scratch")
        if any(
            _byte_ranges_overlap(output_range, input_range)
            for input_range in input_ranges
        ):
            raise ValueError("LiteTopK outputs must not overlap read-only inputs")
    if any(
        _byte_ranges_overlap(scratch_range, input_range)
        for _, scratch_range in workspace_ranges
        for input_range in input_ranges
    ):
        raise ValueError("LiteTopK workspace must not overlap read-only inputs")


def allocate_fp4_litetopk_workspace(
    rows: int,
    device: torch.device | str,
    *,
    topk: int = DEFAULT_TOPK,
    sample_len: int = DEFAULT_SAMPLE_LEN,
    num_buckets: int = DEFAULT_NUM_BUCKETS,
    merge_cap: int = DEFAULT_MERGE_CAP,
    stream: torch.cuda.Stream | None = None,
) -> FP4LiteTopKWorkspace:
    """Allocate caller-owned scratch for the FP4 LiteTopK prefill path."""
    if rows <= 0:
        raise ValueError(f"rows must be positive, got {rows}")
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if sample_len < topk:
        raise ValueError(f"sample_len must be at least topk, got {sample_len} < {topk}")
    if num_buckets <= 1:
        raise ValueError(f"num_buckets must be greater than one, got {num_buckets}")
    if merge_cap < topk:
        raise ValueError(f"merge_cap must be at least topk, got {merge_cap} < {topk}")

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"LiteTopK workspace requires a GPU device, got {device}")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if stream is None:
        stream = torch.cuda.current_stream(device)
    try:
        stream_device = torch.device(stream.device)
    except (AttributeError, TypeError) as exc:
        raise TypeError("stream must be a torch.cuda.Stream") from exc
    if stream_device != device:
        raise ValueError(
            f"stream device {stream_device} does not match workspace device {device}"
        )

    from aiter.ops.topk import topk_ob_workspace_size

    with torch.cuda.stream(stream):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("FP4 LiteTopK workspace allocation is not capture-safe")
        selector_bytes = topk_ob_workspace_size(rows, merge_cap, topk, False)
        return FP4LiteTopKWorkspace(
            sample_logits=torch.empty(
                (rows, sample_len), dtype=torch.float32, device=device
            ),
            sample_ends=torch.empty(rows, dtype=torch.int32, device=device),
            scan_cta_info=torch.empty((rows, 6), dtype=torch.int32, device=device),
            origin=torch.empty(rows, dtype=torch.float32, device=device),
            inv_delta=torch.empty(rows, dtype=torch.float32, device=device),
            threshold=torch.empty(rows, dtype=torch.int32, device=device),
            histogram=torch.empty(
                (rows, num_buckets), dtype=torch.int32, device=device
            ),
            candidate_values=torch.empty(
                (rows, merge_cap), dtype=torch.float32, device=device
            ),
            candidate_indices=torch.empty(
                (rows, merge_cap), dtype=torch.int32, device=device
            ),
            candidate_counts=torch.empty(rows, dtype=torch.int32, device=device),
            page_errors=torch.empty(rows, dtype=torch.int32, device=device),
            score_errors=torch.empty(rows, dtype=torch.int32, device=device),
            select_starts=torch.zeros(rows, dtype=torch.int32, device=device),
            select_ends=torch.empty(rows, dtype=torch.int32, device=device),
            selected_slots=torch.empty((rows, topk), dtype=torch.int32, device=device),
            selected_values=torch.empty(
                (rows, topk), dtype=torch.float32, device=device
            ),
            output_counts=torch.empty(rows, dtype=torch.int32, device=device),
            status=torch.empty(rows, dtype=torch.int32, device=device),
            status_ok=torch.empty(1, dtype=torch.int32, device=device),
            selector_workspace=torch.empty(
                max(1, int(selector_bytes)), dtype=torch.uint8, device=device
            ),
            row_capacity=rows,
            topk=topk,
            sample_len=sample_len,
            num_buckets=num_buckets,
            merge_cap=merge_cap,
            stream_id=stream.cuda_stream,
        )


def prepare_fp4_litetopk_seed(
    sample_logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    workspace: FP4LiteTopKWorkspace,
    *,
    max_seq_len: int,
    emit_candidates: bool = True,
    reset_status: bool = True,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Build the conservative gate and emit the exact-once calibration prefix."""
    rows = sample_logits.shape[0]
    if rows > workspace.row_capacity:
        raise ValueError(
            f"workspace holds {workspace.row_capacity} rows, but input has {rows}"
        )
    device = sample_logits.device
    if stream is None:
        stream = torch.cuda.current_stream(device)
    workspace_ranges = _validate_workspace(
        workspace,
        rows=rows,
        device=device,
        stream=stream,
        topk=workspace.topk,
        sample_len=workspace.sample_len,
        num_buckets=workspace.num_buckets,
        merge_cap=workspace.merge_cap,
    )
    seed_inputs = (sample_logits, row_starts, row_ends)
    _validate_pairwise_disjoint(
        seed_inputs,
        message="LiteTopK seed inputs must not overlap",
    )
    sample_owner = (
        workspace.sample_logits
        if _same_byte_range(sample_logits, workspace.sample_logits[:rows])
        else None
    )
    for seed_input in seed_inputs:
        seed_range = _tensor_byte_range(seed_input)
        for scratch, scratch_range in workspace_ranges:
            if seed_input is sample_logits and scratch is sample_owner:
                continue
            if _byte_ranges_overlap(seed_range, scratch_range):
                raise ValueError("LiteTopK seed inputs must not overlap workspace")
    with torch.cuda.device(device), torch.cuda.stream(stream):
        if reset_status:
            workspace.status[:rows].zero_()
        if not emit_candidates:
            workspace.candidate_counts[:rows].zero_()
            workspace.select_starts[:rows].zero_()
            workspace.select_ends[:rows].zero_()
    _launch_fp4_litetopk_seed(
        sample_logits,
        row_starts,
        row_ends,
        workspace,
        max_seq_len=max_seq_len,
        emit_candidates=emit_candidates,
        stream=stream,
    )


def _launch_fp4_litetopk_seed(
    sample_logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    workspace: FP4LiteTopKWorkspace,
    *,
    max_seq_len: int,
    emit_candidates: bool,
    stream: torch.cuda.Stream,
) -> None:
    from aiter.ops.triton.attention.pa_mqa_litetopk_seed import (
        prepare_pa_mqa_litetopk_seed,
    )

    rows = sample_logits.shape[0]
    prepare_pa_mqa_litetopk_seed(
        sample_logits,
        row_starts,
        row_ends,
        origin=workspace.origin[:rows],
        inv_delta=workspace.inv_delta[:rows],
        threshold=workspace.threshold[:rows],
        histogram=workspace.histogram[:rows],
        candidate_values=workspace.candidate_values[:rows],
        candidate_indices=workspace.candidate_indices[:rows],
        candidate_counts=workspace.candidate_counts[:rows],
        select_starts=workspace.select_starts[:rows],
        select_ends=workspace.select_ends[:rows],
        status=workspace.status[:rows],
        max_seq_len=max_seq_len,
        topk=workspace.topk,
        sample_len=workspace.sample_len,
        num_buckets=workspace.num_buckets,
        merge_cap=workspace.merge_cap,
        status_candidate_overflow=STATUS_CANDIDATE_OVERFLOW,
        status_underfilled=STATUS_UNDERFILLED,
        status_nonfinite=STATUS_NONFINITE,
        status_invalid_index=STATUS_INVALID_INDEX,
        emit_candidates=emit_candidates,
        stream=stream,
    )


def flydsl_pa_mqa_litetopk_fp4_prefill(
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    weights: torch.Tensor,
    row_to_batch: torch.Tensor,
    local_starts: torch.Tensor,
    local_ends: torch.Tensor,
    max_seq_len: int,
    *,
    topk: int = DEFAULT_TOPK,
    sample_len: int = DEFAULT_SAMPLE_LEN,
    num_buckets: int = DEFAULT_NUM_BUCKETS,
    refresh_every: int = DEFAULT_REFRESH_EVERY,
    merge_cap: int = DEFAULT_MERGE_CAP,
    weight_scale: float = 1.0,
    workspace: FP4LiteTopKWorkspace | None = None,
    out: FP4LiteTopKResult | None = None,
    stream: torch.cuda.Stream | None = None,
    enforce_status: bool = True,
) -> FP4LiteTopKResult:
    """Run bounded-prefix FP4 paged-MQA scoring and LiteTopK selection."""
    if topk not in FP4_LITETOPK_SUPPORTED_TOPKS:
        raise ValueError(
            "FP4 LiteTopK requires topk in "
            f"{FP4_LITETOPK_SUPPORTED_TOPKS}, got {topk}"
        )
    if sample_len != DEFAULT_SAMPLE_LEN:
        raise ValueError(
            f"the initial LiteTopK path requires sample_len={DEFAULT_SAMPLE_LEN}"
        )
    if num_buckets != DEFAULT_NUM_BUCKETS:
        raise ValueError(
            f"the initial LiteTopK path requires num_buckets={DEFAULT_NUM_BUCKETS}"
        )
    if refresh_every != DEFAULT_REFRESH_EVERY:
        raise ValueError(
            "the production LiteTopK path requires "
            f"refresh_every={DEFAULT_REFRESH_EVERY}"
        )
    if merge_cap != DEFAULT_MERGE_CAP:
        raise ValueError(
            f"the production LiteTopK path requires merge_cap={DEFAULT_MERGE_CAP}"
        )
    if q_fp4.ndim != 3 or tuple(q_fp4.shape[1:]) != (64, 64):
        raise ValueError("q_fp4 must have shape [rows, 64, 64]")
    rows = q_fp4.shape[0]
    if rows <= 0:
        raise ValueError("LiteTopK requires at least one query row")
    if q_fp4.dtype != torch.uint8 or q_scale.dtype != torch.uint8:
        raise ValueError("q_fp4 and q_scale must use the packed uint8 FP4 ABI")
    if q_scale.shape != (rows, 1, 4, 16, 4):
        raise ValueError("q_scale must have shape [rows, 1, 4, 16, 4]")
    if kv_cache.dtype != torch.uint8 or tuple(kv_cache.shape[1:]) != (
        1,
        4,
        64,
        16,
    ):
        raise ValueError("kv_cache must be uint8 [pages, 1, 4, 64, 16]")
    if kv_scale.dtype != torch.uint8 or tuple(kv_scale.shape) != (
        kv_cache.shape[0],
        1,
        4,
        64,
    ):
        raise ValueError("kv_scale must be uint8 [pages, 1, 4, 64]")
    if weights.shape != (rows, 64) or weights.dtype != torch.bfloat16:
        raise ValueError("weights must be bfloat16 [rows, 64]")
    if block_tables.dtype != torch.int32 or block_tables.ndim != 2:
        raise ValueError("block_tables must be a 2D int32 tensor")
    if block_tables.shape[0] == 0 or block_tables.shape[1] == 0:
        raise ValueError("block_tables must have nonzero row and column capacity")
    if kv_cache.shape[0] == 0:
        raise ValueError("kv_cache must contain at least one physical page")
    metadata = (row_to_batch, local_starts, local_ends)
    if any(t.dtype != torch.int32 or t.shape != (rows,) for t in metadata):
        raise ValueError("row_to_batch/local_starts/local_ends must be int32 [rows]")
    if max_seq_len <= 0 or max_seq_len > merge_cap:
        raise ValueError(
            f"max_seq_len must be in [1, merge_cap={merge_cap}], got {max_seq_len}"
        )
    if max_seq_len > block_tables.shape[1] * 64:
        raise ValueError("max_seq_len exceeds block_tables capacity")
    device = q_fp4.device
    if device.type != "cuda":
        raise ValueError("q_fp4 must be on a GPU device")
    arch = str(torch.cuda.get_device_properties(device).gcnArchName).split(":")[0]
    if arch != "gfx950":
        raise ValueError(f"FP4 LiteTopK requires gfx950, got {arch}")
    inputs = (
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        *metadata,
    )
    if any(t.device != device for t in inputs):
        raise ValueError("all FP4 LiteTopK inputs must be on one device")
    if any(not t.is_contiguous() for t in inputs):
        raise ValueError("all FP4 LiteTopK inputs must be contiguous")
    if stream is None:
        stream = torch.cuda.current_stream(device)
    if torch.device(stream.device) != device:
        raise ValueError("stream and FP4 LiteTopK inputs must be on one device")

    with torch.cuda.device(device), torch.cuda.stream(stream):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("FP4 LiteTopK does not support graph capture")
        if workspace is None:
            workspace = allocate_fp4_litetopk_workspace(
                rows,
                device,
                topk=topk,
                sample_len=sample_len,
                num_buckets=num_buckets,
                merge_cap=merge_cap,
                stream=stream,
            )
        workspace_ranges = _validate_workspace(
            workspace,
            rows=rows,
            device=device,
            stream=stream,
            topk=topk,
            sample_len=sample_len,
            num_buckets=num_buckets,
            merge_cap=merge_cap,
        )
        if out is None:
            out = FP4LiteTopKResult(
                values=torch.empty((rows, topk), dtype=torch.float32, device=device),
                raw_indices=torch.empty((rows, topk), dtype=torch.int32, device=device),
                physical_indices=torch.empty(
                    (rows, topk), dtype=torch.int32, device=device
                ),
                counts=workspace.output_counts[:rows],
                candidate_counts=workspace.candidate_counts[:rows],
                status=workspace.status[:rows],
            )
        _validate_result(
            out,
            workspace,
            rows=rows,
            topk=topk,
            device=device,
            inputs=inputs,
            workspace_ranges=workspace_ranges,
        )

        suffix_cta_info = workspace.selected_slots.view(-1)[: rows * 6].view(rows, 6)

    from aiter.ops.triton.attention.pa_mqa_litetopk_seed import (
        build_pa_mqa_litetopk_scan_schedule,
    )

    from .kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
        flydsl_pa_mqa_litetopk_fp4_prefill_scan,
        flydsl_pa_mqa_litetopk_fp4_seed,
    )

    build_pa_mqa_litetopk_scan_schedule(
        row_to_batch,
        local_starts,
        local_ends,
        workspace.scan_cta_info[:rows],
        segment_start=0,
        segment_end=max_seq_len,
        block_k=_SCAN_BLOCK_K,
        stream=stream,
        page_errors=workspace.page_errors[:rows],
        max_seq_len=max_seq_len,
        block_table_rows=block_tables.shape[0],
        suffix_cta_info=suffix_cta_info,
        sample_len=sample_len,
        score_errors=workspace.score_errors[:rows],
        status=workspace.status[:rows],
        sample_ends=workspace.sample_ends[:rows],
        status_ok=workspace.status_ok,
    )

    flydsl_pa_mqa_litetopk_fp4_seed(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        workspace.scan_cta_info[:rows],
        workspace.origin[:rows],
        workspace.inv_delta[:rows],
        workspace.threshold[:rows],
        workspace.histogram[:rows],
        workspace.candidate_values[:rows],
        workspace.candidate_indices[:rows],
        workspace.candidate_counts[:rows],
        workspace.select_starts[:rows],
        workspace.select_ends[:rows],
        workspace.status[:rows],
        weight_scale=weight_scale,
        n_ctas=rows,
        merge_cap=merge_cap,
        topk=topk,
        status_candidate_overflow=STATUS_CANDIDATE_OVERFLOW,
        status_underfilled=STATUS_UNDERFILLED,
        stream=stream,
        status_nonfinite=STATUS_NONFINITE,
        page_errors=workspace.page_errors[:rows],
        block_k=_SCAN_BLOCK_K,
        num_warps=_SCAN_NUM_WARPS,
    )
    flydsl_pa_mqa_litetopk_fp4_prefill_scan(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        suffix_cta_info,
        workspace.origin[:rows],
        workspace.inv_delta[:rows],
        workspace.threshold[:rows],
        workspace.histogram[:rows],
        workspace.candidate_values[:rows],
        workspace.candidate_indices[:rows],
        workspace.candidate_counts[:rows],
        workspace.page_errors[:rows],
        workspace.score_errors[:rows],
        n_ctas=rows,
        merge_cap=merge_cap,
        topk=topk,
        refresh_every=refresh_every * 256 // _SCAN_BLOCK_K,
        weight_scale=weight_scale,
        block_k=_SCAN_BLOCK_K,
        num_warps=_SCAN_NUM_WARPS,
        stream=stream,
    )

    from aiter.ops.topk import _top_k_per_row_prefill
    from aiter.ops.triton.attention.pa_mqa_litetopk_finalize import (
        finalize_pa_mqa_litetopk,
        prepare_pa_mqa_litetopk_select,
    )

    prepare_pa_mqa_litetopk_select(
        workspace.candidate_counts[:rows],
        workspace.page_errors[:rows],
        workspace.score_errors[:rows],
        workspace.histogram[:rows],
        local_starts,
        local_ends,
        workspace.threshold[:rows],
        workspace.select_ends[:rows],
        workspace.status[:rows],
        topk=topk,
        merge_cap=merge_cap,
        num_buckets=num_buckets,
        status_candidate_overflow=STATUS_CANDIDATE_OVERFLOW,
        status_underfilled=STATUS_UNDERFILLED,
        status_nonfinite=STATUS_NONFINITE,
        status_invalid_index=STATUS_INVALID_INDEX,
        status_bad_certificate=STATUS_BAD_CERTIFICATE,
        stream=stream,
    )
    with torch.cuda.device(device), torch.cuda.stream(stream):
        _top_k_per_row_prefill(
            workspace.candidate_values[:rows],
            workspace.select_starts[:rows],
            workspace.select_ends[:rows],
            workspace.selected_slots[:rows],
            workspace.selected_values[:rows],
            rows,
            workspace.candidate_values.stride(0),
            workspace.candidate_values.stride(1),
            topk,
            workspace.selector_workspace,
            True,
        )
    finalize_pa_mqa_litetopk(
        workspace.candidate_indices[:rows],
        workspace.select_ends[:rows],
        workspace.selected_slots[:rows],
        workspace.selected_values[:rows],
        row_to_batch,
        local_starts,
        local_ends,
        block_tables,
        workspace.status[:rows],
        out.values,
        out.raw_indices,
        out.physical_indices,
        out.counts,
        topk=topk,
        page_size=64,
        physical_page_capacity=kv_cache.shape[0],
        status_nonfinite=STATUS_NONFINITE,
        status_invalid_index=STATUS_INVALID_INDEX,
        status_selector_failure=STATUS_SELECTOR_FAILURE,
        stream=stream,
        status_ok=workspace.status_ok,
    )
    if enforce_status:
        with torch.cuda.device(device), torch.cuda.stream(stream):
            assert_async = getattr(torch, "_assert_async", None)
            if assert_async is not None:
                assert_async(
                    workspace.status_ok, "FP4 LiteTopK produced an invalid row"
                )
            elif not bool(workspace.status_ok):
                raise RuntimeError("FP4 LiteTopK produced an invalid row")
    return out
