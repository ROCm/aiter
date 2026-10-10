from dataclasses import dataclass

import torch
import triton

from aiter.ops.attention import get_pa_metadata_info_v1
from aiter.ops.triton._triton_kernels.attention.pa_ps_metadata import (
    _pa_ps_chunk_scan,
    _pa_ps_schedule,
    _pa_ps_sequence_scan,
    _pa_ps_tile_scan,
    _pa_ps_write_metadata,
)


@dataclass(frozen=True)
class PaPsMetadataPlan:
    """Caller-owned PA_PS tensors, planning workspace, and launch configuration.

    The six metadata tensors use get_pa_metadata_info_v1 shapes and dtypes.
    Workspace tensors are contiguous int64: tile_prefix[batch],
    tile_totals[chunks], sequence_info[batch, 5], and chunk_prefix[chunks, 3],
    where chunks = ceil(batch / scan_block_size). Allocate all tensors on
    the input GPU before capture; their lifetime and reuse belong to the caller.
    work_info, reduce_final_map, and reduce_partial_map may have extra leading
    capacity. Other tensors must use the exact shapes above.
    """

    work_metadata_ptrs: torch.Tensor
    work_indptr: torch.Tensor
    work_info: torch.Tensor
    reduce_indptr: torch.Tensor
    reduce_final_map: torch.Tensor
    reduce_partial_map: torch.Tensor
    tile_prefix: torch.Tensor
    tile_totals: torch.Tensor
    sequence_info: torch.Tensor
    chunk_prefix: torch.Tensor
    num_heads_per_head_k: int
    num_heads_k: int
    max_qlen: int
    block_size: int
    max_partitions: int
    work_overhead: int
    scan_block_size: int
    num_warps: int


def plan_pa_ps_metadata(
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    context_lengths: torch.Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
    *,
    max_qlen: int,
    block_size: int = 16,
    max_partitions: int = 256,
    work_overhead: int | None = None,
    plan: PaPsMetadataPlan,
) -> PaPsMetadataPlan:
    """Fill caller-provided metadata for unchanged PA_PS ASM kernels in place.

    Inputs are contiguous GPU int32 vectors with packed Q/page indptrs. Each
    request has 1..max_qlen Q rows and valid context/page lengths for the ASM
    backend. Supports GQA8/16, max_qlen 1..4 and page16; causal masking stays
    in attention. max_partitions caps each request's splits. work_overhead is
    the positive per-task cost in 256-token tile units (config default: 1).
    Single-chunk batches split queries into independent work when all expanded
    tasks fit the persistent TG budget. KV partitions and reduction maps stay
    unchanged, including the max_qlen-spaced partial layout.

    Returns the supplied plan. The integration layer allocates its buffers,
    loads the launch configuration, and owns reuse and graph capture. Every
    call fills the plan on the input device's current stream without allocation
    or device-to-host reads. Tensor layouts and capacities are checked before
    any kernel launch. Refresh after changing lengths or indptrs.
    No-split records retain partial_qo_loc=-1. This is an opt-in scheduling
    policy, not a replacement for get_pa_metadata_v1.
    """
    if not isinstance(plan, PaPsMetadataPlan):
        raise TypeError("plan must be a caller-allocated PaPsMetadataPlan")
    device = context_lengths.device
    batch = context_lengths.numel()
    for name, tensor, size in (
        ("context_lengths", context_lengths, batch),
        ("qo_indptr", qo_indptr, batch + 1),
        ("kv_indptr", kv_indptr, batch + 1),
    ):
        if (
            tensor.device != device
            or device.type != "cuda"
            or tensor.dtype != torch.int32
            or tensor.shape != (size,)
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be a contiguous int32 vector on the input GPU"
            )
    if not 1 <= batch <= 65536:
        raise ValueError("batch must be in [1, 65536]")
    if num_heads_per_head_k not in (8, 16) or not 1 <= max_qlen <= 4:
        raise ValueError("planner supports GQA8/16 and max_qlen in [1, 4]")
    if block_size != 16:
        raise ValueError(
            "planner requires page16 to match the existing PA_PS ASM kernels"
        )
    if not 1 <= max_partitions <= 256:
        raise ValueError("max_partitions must be in [1, 256]")
    if num_heads_k < 1 or num_heads_k * num_heads_per_head_k > 65535:
        raise ValueError("invalid KV/query head count")
    if work_overhead is not None and (
        not isinstance(work_overhead, int) or not 1 <= work_overhead <= 1024
    ):
        raise ValueError("work_overhead must be an integer in [1, 1024]")
    if not isinstance(plan.work_overhead, int) or not 1 <= plan.work_overhead <= 1024:
        raise ValueError("plan.work_overhead must be an integer in [1, 1024]")
    for name in ("scan_block_size", "num_warps"):
        value = getattr(plan, name)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            or value & (value - 1)
        ):
            raise ValueError(f"plan.{name} must be a positive power-of-two integer")

    with torch.cuda.device(device):
        if (
            plan.num_heads_per_head_k != num_heads_per_head_k
            or plan.num_heads_k != num_heads_k
            or plan.max_qlen != max_qlen
            or plan.block_size != block_size
            or plan.max_partitions != max_partitions
            or (work_overhead is not None and plan.work_overhead != work_overhead)
        ):
            raise ValueError("reused plan does not match the input geometry or policy")
        metadata_specs = get_pa_metadata_info_v1(batch, num_heads_k)
        num_cu = metadata_specs[1][0] - 1
        if num_cu % num_heads_k:
            raise ValueError("KV head count must divide the persistent TG count")
        chunks = triton.cdiv(batch, plan.scan_block_size)
        buffer_specs = tuple(
            zip(
                (
                    "work_metadata_ptrs",
                    "work_indptr",
                    "work_info",
                    "reduce_indptr",
                    "reduce_final_map",
                    "reduce_partial_map",
                ),
                metadata_specs,
            )
        ) + (
            ("tile_prefix", ((batch,), torch.int64)),
            ("tile_totals", ((chunks,), torch.int64)),
            ("sequence_info", ((batch, 5), torch.int64)),
            ("chunk_prefix", ((chunks, 3), torch.int64)),
        )
        for name, (shape, dtype) in buffer_specs:
            tensor = getattr(plan, name)
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"plan.{name} must be a torch.Tensor")
            if (
                tensor.device != device
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"plan.{name} must be contiguous {dtype} on the input GPU"
                )
            expected_shape = (shape,) if isinstance(shape, int) else shape
            allow_extra = name in (
                "work_info",
                "reduce_final_map",
                "reduce_partial_map",
            )
            if (
                tensor.ndim != len(expected_shape)
                or tensor.shape[1:] != expected_shape[1:]
                or tensor.shape[0] < expected_shape[0]
                or (not allow_extra and tensor.shape[0] != expected_shape[0])
            ):
                capacity = "at least " if allow_extra else ""
                raise ValueError(
                    f"plan.{name} requires shape {capacity}{expected_shape}, "
                    f"got {tuple(tensor.shape)}"
                )
        groups = num_cu // num_heads_k
        max_parts = min(max_partitions, groups)
        block_chunks = triton.next_power_of_2(chunks)
        if chunks > 1:
            _pa_ps_tile_scan[(chunks,)](
                context_lengths,
                plan.tile_prefix,
                plan.tile_totals,
                batch,
                plan.scan_block_size,
                num_warps=plan.num_warps,
            )
        _pa_ps_sequence_scan[(chunks,)](
            context_lengths,
            qo_indptr,
            plan.tile_prefix,
            plan.tile_totals,
            plan.sequence_info,
            plan.chunk_prefix,
            batch,
            groups,
            max_parts,
            max_qlen,
            plan.work_overhead,
            chunks,
            block_chunks,
            plan.scan_block_size,
            num_warps=plan.num_warps,
        )
        if chunks > 1:
            _pa_ps_chunk_scan[(1,)](
                plan.chunk_prefix,
                chunks,
                block_chunks,
                num_warps=plan.num_warps,
            )
        _pa_ps_write_metadata[(batch, num_heads_k)](
            context_lengths,
            qo_indptr,
            kv_indptr,
            plan.sequence_info,
            plan.chunk_prefix,
            plan.work_info,
            plan.reduce_indptr,
            plan.reduce_final_map,
            plan.reduce_partial_map,
            chunks,
            num_heads_k,
            num_heads_per_head_k,
            max_qlen,
            block_size,
            plan.scan_block_size,
            triton.next_power_of_2(max_parts),
            num_warps=plan.num_warps,
        )
        _pa_ps_schedule[(1,)](
            context_lengths,
            plan.sequence_info,
            plan.chunk_prefix,
            plan.work_metadata_ptrs,
            plan.work_indptr,
            plan.work_info,
            batch,
            num_cu,
            groups,
            chunks,
            max_qlen > 1 and chunks == 1,
            plan.work_overhead,
            (batch - 1).bit_length(),
            max_parts.bit_length(),
            plan.scan_block_size,
            triton.next_power_of_2(num_cu + 1),
            num_warps=plan.num_warps,
        )
    return plan
