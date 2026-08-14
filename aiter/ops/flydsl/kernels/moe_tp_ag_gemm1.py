# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pure-TP Stage1 multi-source scheduler prerequisites.

The adapter dispatches source-local sorted blocks through one expert-major
GEMM1 launch. It also owns a device-side descriptor producer and a default-off
same-GPU generation/source-ready substrate. Peer buffers, transport, chunked
AllGather, and collective integration remain outside this module for now.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import arith, gpu, ptrtoint, range_constexpr
from flydsl.expr.typing import T

from aiter.ops.flydsl.moe_common import GateMode

from . import buffer_ops, communication_ops_utils as comm_ops
from .mixed_moe_gemm_2stage_common import compile_mixed_moe_gemm1_common
from .tensor_shim import _run_compiled, ptr_arg, ptr_rsrc

_EXPERT_BITS = 10
_SOURCE_BITS = 6
_SOURCE_SHIFT = _EXPERT_BITS
_BLOCK_SHIFT = _EXPERT_BITS + _SOURCE_BITS
_EXPERT_MASK = (1 << _EXPERT_BITS) - 1
_SOURCE_MASK = (1 << _SOURCE_BITS) - 1
_MAX_LOCAL_BLOCK = (1 << (31 - _BLOCK_SHIFT)) - 1


@dataclass(frozen=True)
class TPStage1AllReadyMetadata:
    """Packed source-local metadata consumed by the single-launch kernel."""

    sorted_token_ids: torch.Tensor
    sorted_scales: torch.Tensor
    work_descriptors: torch.Tensor
    num_valid_ids: torch.Tensor
    source_count: int
    tokens_per_source: int
    source_sorted_stride: int
    source_scale_stride: int


@dataclass(frozen=True)
class TPStage1DeviceMetadata(TPStage1AllReadyMetadata):
    """Graph-safe descriptor workspace and local source-ready protocol state."""

    source_expert_ids: torch.Tensor
    source_num_valid_ids: torch.Tensor
    expert_source_starts: torch.Tensor
    expert_source_counts: torch.Tensor
    expert_offsets: torch.Tensor
    source_expert_stride: int
    source_ready: torch.Tensor
    source_payload_epoch: torch.Tensor
    source_observed_epoch: torch.Tensor
    source_ready_entry: torch.Tensor
    source_current_epoch: torch.Tensor
    source_ready_errors: torch.Tensor


def _encode_work(expert: int, source: int, local_block: int) -> int:
    if expert < 0 or expert > _EXPERT_MASK:
        raise ValueError(f"expert id {expert} exceeds {_EXPERT_BITS}-bit descriptor")
    if source < 0 or source > _SOURCE_MASK:
        raise ValueError(f"source id {source} exceeds {_SOURCE_BITS}-bit descriptor")
    if local_block < 0 or local_block > _MAX_LOCAL_BLOCK:
        raise ValueError(
            f"local block {local_block} exceeds descriptor limit {_MAX_LOCAL_BLOCK}"
        )
    return expert | (source << _SOURCE_SHIFT) | (local_block << _BLOCK_SHIFT)


def prepare_tp_stage1_all_ready_metadata(
    source_metadata,
    *,
    tokens_per_source: int,
    experts: int,
    topk: int,
    tile_m: int,
) -> TPStage1AllReadyMetadata:
    """Pack per-source sorting results into an expert-major work descriptor.

    Descriptor construction is intentionally outside the compute-only timing.
    A later segmented route-prep kernel will produce the same layout directly.
    """
    source_count = len(source_metadata)
    if source_count <= 1 or source_count > _SOURCE_MASK + 1:
        raise ValueError("source_metadata must contain between 2 and 64 sources")
    if tokens_per_source <= 0 or experts <= 0 or experts > _EXPERT_MASK + 1:
        raise ValueError("invalid TP Stage1 token/expert dimensions")
    if topk <= 0 or tile_m != 32:
        raise ValueError("the first TP Stage1 gate requires topk > 0 and tile_m=32")

    expected_sorted_stride = tokens_per_source * topk + experts * tile_m - topk
    device = source_metadata[0][0].device
    sorted_ids_parts = []
    scale_parts = []
    work_by_expert = [[] for _ in range(experts)]
    source_scale_stride = None

    for source, metadata in enumerate(source_metadata):
        sorted_ids, _sorted_weights, expert_ids, num_valid_ids, sorted_scale = metadata
        if sorted_ids.device != device or expert_ids.device != device:
            raise ValueError("all source metadata must reside on one device")
        if sorted_ids.dtype != torch.int32 or expert_ids.dtype != torch.int32:
            raise TypeError("sorted token/expert ids must be int32")
        if sorted_ids.numel() != expected_sorted_stride:
            raise ValueError(
                f"source {source} sorted stride is {sorted_ids.numel()}, "
                f"expected {expected_sorted_stride}"
            )
        scale_bytes = sorted_scale.view(torch.uint8).contiguous().view(-1)
        if source_scale_stride is None:
            source_scale_stride = scale_bytes.numel()
        elif scale_bytes.numel() != source_scale_stride:
            raise ValueError("all source scale buffers must have the same stride")

        valid_rows = int(num_valid_ids.view(-1)[0].item())
        if valid_rows < 0 or valid_rows % tile_m:
            raise ValueError(
                f"source {source} num_valid_ids={valid_rows} is not BM-aligned"
            )
        valid_blocks = valid_rows // tile_m
        if valid_blocks > expert_ids.numel():
            raise ValueError("num_valid_ids exceeds sorted_expert_ids capacity")
        local_experts = expert_ids[:valid_blocks].cpu().tolist()
        for local_block, expert in enumerate(local_experts):
            if expert < 0 or expert >= experts:
                raise ValueError(
                    f"source {source} block {local_block} has invalid expert {expert}"
                )
            work_by_expert[expert].append(
                _encode_work(expert, source, local_block)
            )

        sorted_ids_parts.append(sorted_ids.contiguous().view(-1))
        scale_parts.append(scale_bytes)

    work = [encoded for expert_work in work_by_expert for encoded in expert_work]
    if not work:
        raise ValueError("TP Stage1 all-ready descriptor contains no work")
    work_descriptors = torch.tensor(work, dtype=torch.int32, device=device)
    num_valid_ids = torch.tensor(
        [len(work) * tile_m, 0], dtype=torch.int32, device=device
    )
    return TPStage1AllReadyMetadata(
        sorted_token_ids=torch.cat(sorted_ids_parts).contiguous(),
        sorted_scales=torch.cat(scale_parts).contiguous(),
        work_descriptors=work_descriptors.contiguous(),
        num_valid_ids=num_valid_ids,
        source_count=source_count,
        tokens_per_source=tokens_per_source,
        source_sorted_stride=expected_sorted_stride,
        source_scale_stride=int(source_scale_stride),
    )


@functools.cache
def compile_tp_stage1_descriptor_pack(
    *,
    source_count: int,
    experts: int,
    source_expert_stride: int,
    tile_m: int = 32,
    wait_source_ready: bool = False,
):
    """Compile a deterministic device-side expert-major descriptor packer.

    The input contains one sorted expert-id list per source.  Five lightweight
    kernels clear the range table, discover ``[expert, source]`` sorted ranges
    in one pass over the source blocks, finalize counts in parallel, prefix the
    expert totals, then fill the compact descriptor prefix in
    ``expert -> source -> local block`` order.
    The output buffer keeps a static maximum capacity for graph safety;
    ``num_valid_ids[0]`` carries the exact device-produced row count.

    The default-off ``wait_source_ready`` variant adds a device-resident
    generation ticket and a same-GPU release/acquire protocol. Its publisher
    and consumer intentionally run on one stream in this first substrate
    proof; no peer transport or communication/compute overlap is implied.
    """
    if source_count <= 1 or source_count > _SOURCE_MASK + 1:
        raise ValueError(f"source_count must be in [2, 64], got {source_count}")
    if experts <= 0 or experts > _EXPERT_MASK + 1:
        raise ValueError(f"experts must be in [1, 1024], got {experts}")
    if source_expert_stride <= 0 or source_expert_stride > _MAX_LOCAL_BLOCK + 1:
        raise ValueError(
            "source_expert_stride exceeds descriptor block field: "
            f"{source_expert_stride}"
        )
    if tile_m != 32:
        raise ValueError(f"the first descriptor packer requires tile_m=32, got {tile_m}")

    block = 64
    variant = "ready" if wait_source_ready else "nowait"
    clear_name = (
        f"moe_tp_gemm1_desc_clear_ranges_s{source_count}_e{experts}"
    )
    range_name = (
        f"moe_tp_gemm1_desc_sorted_range_s{source_count}_e{experts}"
        f"_b{source_expert_stride}_{variant}"
    )
    finalize_name = (
        f"moe_tp_gemm1_desc_finalize_ranges_s{source_count}_e{experts}"
    )
    prefix_name = (
        f"moe_tp_gemm1_desc_prefix_ranges_s{source_count}_e{experts}"
    )
    fill_name = (
        f"moe_tp_gemm1_desc_fill_ranges_s{source_count}_e{experts}"
        f"_b{source_expert_stride}"
    )

    if wait_source_ready:

        @flyc.kernel(
            name=f"moe_tp_gemm1_source_epoch_begin_s{source_count}",
            known_block_size=[1, 1, 1],
        )
        def source_epoch_begin_kernel(
            source_ready_entry: fx.Pointer,  # [1] i64
            source_current_epoch: fx.Pointer,  # [1] i64
        ):
            entry_addr = fx.Int64(ptrtoint(source_ready_entry))
            current_addr = fx.Int64(ptrtoint(source_current_epoch))
            generation = fx.Int64(
                comm_ops.atomic_add_agent(entry_addr, fx.Int64(1))
            )
            comm_ops.store_i64_global_agent(
                current_addr, generation + fx.Int64(1)
            )

        @flyc.kernel(
            name=f"moe_tp_gemm1_source_publish_s{source_count}",
            known_block_size=[block, 1, 1],
        )
        def source_publish_kernel(
            source_ready: fx.Pointer,  # [source_count] i64
            source_payload_epoch: fx.Pointer,  # [source_count] i64
            source_current_epoch: fx.Pointer,  # [1] i64
        ):
            source = fx.Uint32(gpu.thread_idx.x)
            if source < fx.Uint32(source_count):
                ready_addr = fx.Int64(ptrtoint(source_ready))
                payload_addr = fx.Int64(ptrtoint(source_payload_epoch))
                current_addr = fx.Int64(ptrtoint(source_current_epoch))
                epoch = fx.Int64(comm_ops.load_i64_agent_acquire(current_addr))
                source_byte_offset = fx.Int64(source) * fx.Int64(8)
                comm_ops.store_i64_global_agent(
                    payload_addr + source_byte_offset, epoch
                )
                comm_ops.store_i64_global_agent(
                    ready_addr + source_byte_offset, epoch
                )

    @flyc.kernel(name=clear_name, known_block_size=[block, 1, 1])
    def descriptor_clear_kernel(
        expert_source_starts: fx.Pointer,  # [experts, source_count] blocks, i32
        expert_source_counts: fx.Pointer,  # [experts, source_count] blocks, i32
    ):
        entry = fx.Uint32(gpu.block_idx.x) * fx.Uint32(block) + fx.Uint32(
            gpu.thread_idx.x
        )
        entry_count = fx.Uint32(experts * source_count)
        if entry < entry_count:
            starts_rsrc = ptr_rsrc(expert_source_starts)
            counts_rsrc = ptr_rsrc(expert_source_counts)
            buffer_ops.buffer_store(fx.Int32(0), starts_rsrc, entry)
            buffer_ops.buffer_store(fx.Int32(0), counts_rsrc, entry)

    @flyc.kernel(name=range_name, known_block_size=[block, 1, 1])
    def descriptor_range_kernel(
        source_expert_ids: fx.Pointer,  # [source_count, source_expert_stride] i32
        source_num_valid_ids: fx.Pointer,  # [source_count] padded rows, i32
        expert_source_starts: fx.Pointer,  # [experts, source_count] blocks, i32
        expert_source_counts: fx.Pointer,  # [experts, source_count] blocks, i32
        source_ready: fx.Pointer,  # [source_count] i64
        source_payload_epoch: fx.Pointer,  # [source_count] i64
        source_observed_epoch: fx.Pointer,  # [source_count] i64
        source_current_epoch: fx.Pointer,  # [1] i64
        source_ready_errors: fx.Pointer,  # [1] i32
    ):
        source = fx.Uint32(gpu.block_idx.x)
        lane = fx.Uint32(gpu.thread_idx.x)
        if source < fx.Uint32(source_count):
            if wait_source_ready:
                if lane == fx.Uint32(0):
                    ready_addr = fx.Int64(ptrtoint(source_ready))
                    payload_addr = fx.Int64(ptrtoint(source_payload_epoch))
                    observed_addr = fx.Int64(ptrtoint(source_observed_epoch))
                    current_addr = fx.Int64(ptrtoint(source_current_epoch))
                    error_addr = fx.Int64(ptrtoint(source_ready_errors))
                    epoch = fx.Int64(
                        comm_ops.load_i64_agent_acquire(current_addr)
                    )
                    source_byte_offset = fx.Int64(source) * fx.Int64(8)
                    comm_ops.wait_i64_agent_until_equals(
                        ready_addr + source_byte_offset, epoch
                    )
                    payload_epoch = fx.Int64(
                        comm_ops.load_i64_agent_acquire(
                            payload_addr + source_byte_offset
                        )
                    )
                    if payload_epoch != epoch:
                        comm_ops.atomic_add_agent(error_addr, fx.Int32(1))
                    comm_ops.store_i64_global_agent(
                        observed_addr + source_byte_offset, payload_epoch
                    )
                gpu.barrier()
            source_e_rsrc = ptr_rsrc(source_expert_ids)
            source_nvalid_rsrc = ptr_rsrc(source_num_valid_ids)
            starts_rsrc = ptr_rsrc(expert_source_starts)
            counts_rsrc = ptr_rsrc(expert_source_counts)
            valid_rows = fx.Uint32(
                buffer_ops.buffer_load(
                    source_nvalid_rsrc, source, vec_width=1, dtype=T.i32
                )
            )
            valid_blocks = valid_rows // fx.Uint32(tile_m)
            source_base = source * fx.Uint32(source_expert_stride)
            for local_block in range(lane, valid_blocks, fx.Uint32(block)):
                local_expert = fx.Uint32(
                    buffer_ops.buffer_load(
                        source_e_rsrc,
                        source_base + local_block,
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
                if local_expert < fx.Uint32(experts):
                    range_offset = (
                        local_expert * fx.Uint32(source_count) + source
                    )
                    is_start = local_block == fx.Uint32(0)
                    if local_block > fx.Uint32(0):
                        previous_expert = fx.Uint32(
                            buffer_ops.buffer_load(
                                source_e_rsrc,
                                source_base + local_block - fx.Uint32(1),
                                vec_width=1,
                                dtype=T.i32,
                            )
                        )
                        is_start = previous_expert != local_expert
                    if is_start:
                        buffer_ops.buffer_store(
                            fx.Int32(local_block), starts_rsrc, range_offset
                        )
                    next_block = local_block + fx.Uint32(1)
                    is_end = next_block == valid_blocks
                    if next_block < valid_blocks:
                        next_expert = fx.Uint32(
                            buffer_ops.buffer_load(
                                source_e_rsrc,
                                source_base + next_block,
                                vec_width=1,
                                dtype=T.i32,
                            )
                        )
                        is_end = next_expert != local_expert
                    if is_end:
                        # Keep the exclusive range end in the count table;
                        # the parallel finalize kernel converts it in-place.
                        buffer_ops.buffer_store(
                            fx.Int32(next_block), counts_rsrc, range_offset
                        )

    @flyc.kernel(name=finalize_name, known_block_size=[block, 1, 1])
    def descriptor_finalize_kernel(
        expert_source_starts: fx.Pointer,  # [experts, source_count] starts, i32
        expert_source_counts: fx.Pointer,  # range ends on input, counts on output
    ):
        entry = fx.Uint32(gpu.block_idx.x) * fx.Uint32(block) + fx.Uint32(
            gpu.thread_idx.x
        )
        entry_count = fx.Uint32(experts * source_count)
        if entry < entry_count:
            starts_rsrc = ptr_rsrc(expert_source_starts)
            counts_rsrc = ptr_rsrc(expert_source_counts)
            range_start = fx.Uint32(
                buffer_ops.buffer_load(
                    starts_rsrc, entry, vec_width=1, dtype=T.i32
                )
            )
            range_end = fx.Uint32(
                buffer_ops.buffer_load(
                    counts_rsrc, entry, vec_width=1, dtype=T.i32
                )
            )
            buffer_ops.buffer_store(
                fx.Int32(range_end - range_start), counts_rsrc, entry
            )

    @flyc.kernel(name=prefix_name, known_block_size=[1, 1, 1])
    def descriptor_prefix_kernel(
        expert_source_counts: fx.Pointer,  # [experts, source_count] i32
        expert_offsets: fx.Pointer,  # [experts] exclusive block offsets, i32
        num_valid_ids: fx.Pointer,  # [2] padded rows + compatibility slot, i32
    ):
        counts_rsrc = ptr_rsrc(expert_source_counts)
        offsets_rsrc = ptr_rsrc(expert_offsets)
        nvalid_rsrc = ptr_rsrc(num_valid_ids)
        total_blocks = fx.Uint32(0)
        for expert in range_constexpr(experts):
            buffer_ops.buffer_store(
                fx.Int32(total_blocks), offsets_rsrc, fx.Uint32(expert)
            )
            expert_base = fx.Uint32(expert * source_count)
            for source in range_constexpr(source_count):
                count = fx.Uint32(
                    buffer_ops.buffer_load(
                        counts_rsrc,
                        expert_base + fx.Uint32(source),
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
                total_blocks = total_blocks + count
        buffer_ops.buffer_store(
            fx.Int32(total_blocks * fx.Uint32(tile_m)),
            nvalid_rsrc,
            fx.Uint32(0),
        )
        buffer_ops.buffer_store(fx.Int32(0), nvalid_rsrc, fx.Uint32(1))

    @flyc.kernel(name=fill_name, known_block_size=[block, 1, 1])
    def descriptor_fill_kernel(
        expert_source_starts: fx.Pointer,  # [experts, source_count] blocks, i32
        expert_source_counts: fx.Pointer,  # [experts, source_count] blocks, i32
        expert_offsets: fx.Pointer,  # [experts] exclusive block offsets, i32
        work_descriptors: fx.Pointer,  # compact valid prefix + unused capacity, i32
    ):
        expert = fx.Uint32(gpu.block_idx.x)
        source = fx.Uint32(gpu.thread_idx.x)
        if source < fx.Uint32(source_count):
            starts_rsrc = ptr_rsrc(expert_source_starts)
            counts_rsrc = ptr_rsrc(expert_source_counts)
            offsets_rsrc = ptr_rsrc(expert_offsets)
            work_rsrc = ptr_rsrc(work_descriptors)

            output_offset = fx.Uint32(
                buffer_ops.buffer_load(
                    offsets_rsrc, expert, vec_width=1, dtype=T.i32
                )
            )
            expert_count_base = expert * fx.Uint32(source_count)
            for previous_source in range_constexpr(source_count):
                if previous_source < source:
                    output_offset = output_offset + fx.Uint32(
                        buffer_ops.buffer_load(
                            counts_rsrc,
                            expert_count_base + fx.Uint32(previous_source),
                            vec_width=1,
                            dtype=T.i32,
                        )
                    )

            source_begin = fx.Uint32(
                buffer_ops.buffer_load(
                    starts_rsrc,
                    expert_count_base + source,
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            source_count_blocks = fx.Uint32(
                buffer_ops.buffer_load(
                    counts_rsrc,
                    expert_count_base + source,
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            for source_write in range(
                fx.Uint32(0), source_count_blocks, fx.Uint32(1)
            ):
                local_block = source_begin + source_write
                encoded = (
                    expert
                    | (source << fx.Uint32(_SOURCE_SHIFT))
                    | (local_block << fx.Uint32(_BLOCK_SHIFT))
                )
                buffer_ops.buffer_store(
                    fx.Int32(encoded), work_rsrc, output_offset + source_write
                )

    @flyc.jit
    def launch_descriptor_pack(
        source_expert_ids: fx.Pointer,
        source_num_valid_ids: fx.Pointer,
        expert_source_starts: fx.Pointer,
        expert_source_counts: fx.Pointer,
        expert_offsets: fx.Pointer,
        work_descriptors: fx.Pointer,
        num_valid_ids: fx.Pointer,
        source_ready: fx.Pointer,
        source_payload_epoch: fx.Pointer,
        source_observed_epoch: fx.Pointer,
        source_ready_entry: fx.Pointer,
        source_current_epoch: fx.Pointer,
        source_ready_errors: fx.Pointer,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        if wait_source_ready:
            source_epoch_begin_kernel(
                source_ready_entry, source_current_epoch
            ).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)
            source_publish_kernel(
                source_ready, source_payload_epoch, source_current_epoch
            ).launch(grid=(1, 1, 1), block=(block, 1, 1), stream=stream)
        descriptor_clear_kernel(
            expert_source_starts,
            expert_source_counts,
        ).launch(
            grid=((experts * source_count + block - 1) // block, 1, 1),
            block=(block, 1, 1),
            stream=stream,
        )
        descriptor_range_kernel(
            source_expert_ids,
            source_num_valid_ids,
            expert_source_starts,
            expert_source_counts,
            source_ready,
            source_payload_epoch,
            source_observed_epoch,
            source_current_epoch,
            source_ready_errors,
        ).launch(grid=(source_count, 1, 1), block=(block, 1, 1), stream=stream)
        descriptor_finalize_kernel(
            expert_source_starts,
            expert_source_counts,
        ).launch(
            grid=((experts * source_count + block - 1) // block, 1, 1),
            block=(block, 1, 1),
            stream=stream,
        )
        descriptor_prefix_kernel(
            expert_source_counts,
            expert_offsets,
            num_valid_ids,
        ).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)
        descriptor_fill_kernel(
            expert_source_starts,
            expert_source_counts,
            expert_offsets,
            work_descriptors,
        ).launch(grid=(experts, 1, 1), block=(block, 1, 1), stream=stream)

    return launch_descriptor_pack


def launch_tp_stage1_descriptor_pack(
    metadata: TPStage1DeviceMetadata, *, wait_source_ready: bool = False
) -> None:
    """Regenerate descriptors, optionally gated by local source-ready epochs."""
    launcher = compile_tp_stage1_descriptor_pack(
        source_count=metadata.source_count,
        experts=metadata.expert_offsets.numel(),
        source_expert_stride=metadata.source_expert_stride,
        tile_m=32,
        wait_source_ready=wait_source_ready,
    )
    _run_compiled(
        launcher,
        ptr_arg(metadata.source_expert_ids),
        ptr_arg(metadata.source_num_valid_ids),
        ptr_arg(metadata.expert_source_starts),
        ptr_arg(metadata.expert_source_counts),
        ptr_arg(metadata.expert_offsets),
        ptr_arg(metadata.work_descriptors),
        ptr_arg(metadata.num_valid_ids),
        ptr_arg(metadata.source_ready),
        ptr_arg(metadata.source_payload_epoch),
        ptr_arg(metadata.source_observed_epoch),
        ptr_arg(metadata.source_ready_entry),
        ptr_arg(metadata.source_current_epoch),
        ptr_arg(metadata.source_ready_errors),
        fx.Stream(torch.cuda.current_stream()),
    )


def prepare_tp_stage1_device_metadata(
    source_metadata,
    *,
    tokens_per_source: int,
    experts: int,
    topk: int,
    tile_m: int,
) -> TPStage1DeviceMetadata:
    """Pack source tensors and launch the graph-safe GPU descriptor producer.

    Unlike :func:`prepare_tp_stage1_all_ready_metadata`, this path never reads
    a device scalar on the host.  The ``torch.cat``/``torch.stack`` operations
    only stage existing source-local outputs into the future fixed workspace;
    sorting kernels can write those slices directly in the next integration.
    """
    source_count = len(source_metadata)
    if source_count <= 1 or source_count > _SOURCE_MASK + 1:
        raise ValueError("source_metadata must contain between 2 and 64 sources")
    if tokens_per_source <= 0 or experts <= 0 or experts > _EXPERT_MASK + 1:
        raise ValueError("invalid TP Stage1 token/expert dimensions")
    if topk <= 0 or tile_m != 32:
        raise ValueError("the first TP Stage1 gate requires topk > 0 and tile_m=32")

    expected_sorted_stride = tokens_per_source * topk + experts * tile_m - topk
    device = source_metadata[0][0].device
    source_expert_stride = source_metadata[0][2].numel()
    source_scale_stride = source_metadata[0][4].view(torch.uint8).numel()
    sorted_ids_parts = []
    scale_parts = []
    expert_parts = []
    nvalid_parts = []
    for source, metadata in enumerate(source_metadata):
        sorted_ids, _sorted_weights, expert_ids, num_valid_ids, sorted_scale = metadata
        if any(
            tensor.device != device
            for tensor in (sorted_ids, expert_ids, num_valid_ids, sorted_scale)
        ):
            raise ValueError("all source metadata must reside on one device")
        if sorted_ids.dtype != torch.int32 or expert_ids.dtype != torch.int32:
            raise TypeError("sorted token/expert ids must be int32")
        if num_valid_ids.dtype != torch.int32:
            raise TypeError("source num_valid_ids must be int32")
        if sorted_ids.numel() != expected_sorted_stride:
            raise ValueError(
                f"source {source} sorted stride is {sorted_ids.numel()}, "
                f"expected {expected_sorted_stride}"
            )
        if expert_ids.numel() != source_expert_stride:
            raise ValueError("all source expert-id buffers must have one stride")
        scale_bytes = sorted_scale.view(torch.uint8).contiguous().view(-1)
        if scale_bytes.numel() != source_scale_stride:
            raise ValueError("all source scale buffers must have the same stride")
        sorted_ids_parts.append(sorted_ids.contiguous().view(-1))
        scale_parts.append(scale_bytes)
        expert_parts.append(expert_ids.contiguous().view(-1))
        nvalid_parts.append(num_valid_ids.contiguous().view(-1)[:1])

    metadata = TPStage1DeviceMetadata(
        sorted_token_ids=torch.cat(sorted_ids_parts).contiguous(),
        sorted_scales=torch.cat(scale_parts).contiguous(),
        work_descriptors=torch.empty(
            source_count * source_expert_stride,
            dtype=torch.int32,
            device=device,
        ),
        num_valid_ids=torch.empty(2, dtype=torch.int32, device=device),
        source_count=source_count,
        tokens_per_source=tokens_per_source,
        source_sorted_stride=expected_sorted_stride,
        source_scale_stride=source_scale_stride,
        source_expert_ids=torch.stack(expert_parts).contiguous(),
        source_num_valid_ids=torch.cat(nvalid_parts).contiguous(),
        expert_source_starts=torch.empty(
            (experts, source_count), dtype=torch.int32, device=device
        ),
        expert_source_counts=torch.empty(
            (experts, source_count), dtype=torch.int32, device=device
        ),
        expert_offsets=torch.empty(experts, dtype=torch.int32, device=device),
        source_expert_stride=source_expert_stride,
        source_ready=torch.zeros(source_count, dtype=torch.int64, device=device),
        source_payload_epoch=torch.zeros(
            source_count, dtype=torch.int64, device=device
        ),
        source_observed_epoch=torch.zeros(
            source_count, dtype=torch.int64, device=device
        ),
        source_ready_entry=torch.zeros(1, dtype=torch.int64, device=device),
        source_current_epoch=torch.zeros(1, dtype=torch.int64, device=device),
        source_ready_errors=torch.zeros(1, dtype=torch.int32, device=device),
    )
    launch_tp_stage1_descriptor_pack(metadata)
    return metadata


@dataclass(frozen=True)
class TPStage1PeerSourceLayout:
    """Byte offsets for the standalone peer source/chunk freshness gate."""

    payload_offset: int
    entry_offset: int
    current_epoch_offset: int
    payload_epoch_offset: int
    ready_offset: int
    observed_epoch_offset: int
    skew_scratch_offset: int
    errors_offset: int
    total_bytes: int


def tp_stage1_peer_source_workspace_layout(
    *, source_count: int, chunks_per_source: int, payload_elements: int
) -> TPStage1PeerSourceLayout:
    """Return the registered workspace layout for the TP2 peer protocol gate."""
    if source_count <= 1 or source_count > _SOURCE_MASK + 1:
        raise ValueError(f"source_count must be in [2, 64], got {source_count}")
    if chunks_per_source <= 0 or payload_elements <= 0:
        raise ValueError("chunks_per_source and payload_elements must be positive")

    records = source_count * chunks_per_source
    payload_offset = 0
    payload_bytes = records * payload_elements * 4
    entry_offset = (payload_bytes + 7) // 8 * 8
    current_epoch_offset = entry_offset + 8
    payload_epoch_offset = current_epoch_offset + 8
    ready_offset = payload_epoch_offset + records * 8
    observed_epoch_offset = ready_offset + records * 8
    skew_scratch_offset = observed_epoch_offset + records * 8
    errors_offset = skew_scratch_offset + 8
    total_bytes = (errors_offset + 4 + 7) // 8 * 8
    return TPStage1PeerSourceLayout(
        payload_offset=payload_offset,
        entry_offset=entry_offset,
        current_epoch_offset=current_epoch_offset,
        payload_epoch_offset=payload_epoch_offset,
        ready_offset=ready_offset,
        observed_epoch_offset=observed_epoch_offset,
        skew_scratch_offset=skew_scratch_offset,
        errors_offset=errors_offset,
        total_bytes=total_bytes,
    )


@functools.cache
def compile_tp_stage1_peer_source_exchange(
    *, source_count: int, chunks_per_source: int, payload_elements: int
):
    """Compile a real peer source/chunk publication freshness gate.

    Each rank is one source. It writes its local chunk payload into every
    IPC-registered destination workspace, then publishes full 64-bit epochs
    with system-scope release stores. The destination waits with system-scope
    acquire loads and copies the observed payload in deterministic
    ``source -> chunk -> element`` order.

    This is deliberately independent from descriptor/GEMM1 scheduling. It
    proves peer mapping, generation, memory order, skew tolerance, and graph
    freshness before peer transport is allowed into the compute path.
    """
    layout = tp_stage1_peer_source_workspace_layout(
        source_count=source_count,
        chunks_per_source=chunks_per_source,
        payload_elements=payload_elements,
    )
    records = source_count * chunks_per_source

    @flyc.kernel(
        name=(
            f"moe_tp_gemm1_peer_source_begin_s{source_count}"
            f"_c{chunks_per_source}_n{payload_elements}"
        ),
        known_block_size=[1, 1, 1],
    )
    def peer_source_begin_kernel(workspace: fx.Pointer):
        local_base = fx.Int64(ptrtoint(workspace))
        entry_addr = local_base + fx.Int64(layout.entry_offset)
        current_addr = local_base + fx.Int64(layout.current_epoch_offset)
        epoch = fx.Int64(
            comm_ops.atomic_add_agent(entry_addr, fx.Int64(1))
        ) + fx.Int64(1)
        comm_ops.store_i64_global_agent(current_addr, epoch)

    @flyc.kernel(
        name=(
            f"moe_tp_gemm1_peer_source_publish_s{source_count}"
            f"_c{chunks_per_source}_n{payload_elements}"
        ),
        known_block_size=[1, 1, 1],
    )
    def peer_source_publish_kernel(
        workspace: fx.Pointer,
        peer_rank_data: fx.Int64,
        local_payload: fx.Pointer,
        rank: fx.Int32,
        skew_iterations: fx.Int32,
    ):
        local_base = fx.Int64(ptrtoint(workspace))
        current_addr = local_base + fx.Int64(layout.current_epoch_offset)
        epoch = fx.Int64(comm_ops.load_i64_agent_acquire(current_addr))

        if rank == fx.Int32(source_count - 1):
            iteration = fx.Int32(0)
            skew_accumulator = fx.Int64(0)
            while iteration < skew_iterations:
                skew_accumulator = skew_accumulator + fx.Int64(
                    comm_ops.load_i64_agent_acquire(current_addr)
                )
                iteration = iteration + fx.Int32(1)
            comm_ops.store_i64_global_agent(
                local_base + fx.Int64(layout.skew_scratch_offset),
                skew_accumulator,
            )

        peer_table = buffer_ops.create_buffer_resource_from_addr(
            peer_rank_data, num_records_bytes=source_count * 8
        )
        local_payload_addr = fx.Int64(ptrtoint(local_payload))
        for destination in range_constexpr(source_count):
            peer_base = fx.Int64(
                buffer_ops.buffer_load(
                    peer_table,
                    fx.Int32(destination),
                    vec_width=1,
                    dtype=fx.Int64,
                )
            )
            for chunk in range_constexpr(chunks_per_source):
                record = rank * fx.Int32(chunks_per_source) + fx.Int32(chunk)
                peer_payload_addr = (
                    peer_base
                    + fx.Int64(layout.payload_offset)
                    + fx.Int64(record) * fx.Int64(payload_elements * 4)
                )
                local_chunk_addr = local_payload_addr + fx.Int64(
                    chunk * payload_elements * 4
                )
                element = fx.Int32(0)
                while element < fx.Int32(payload_elements):
                    value = fx.Int32(
                        comm_ops.load_i32_system_acquire(
                            local_chunk_addr + fx.Int64(element) * fx.Int64(4)
                        )
                    )
                    comm_ops.store_i32_system(peer_payload_addr, element, value)
                    element = element + fx.Int32(1)

                record_byte_offset = fx.Int64(record) * fx.Int64(8)
                comm_ops.store_i64_global_system(
                    peer_base
                    + fx.Int64(layout.payload_epoch_offset)
                    + record_byte_offset,
                    epoch,
                )
                comm_ops.store_i64_global_system(
                    peer_base
                    + fx.Int64(layout.ready_offset)
                    + record_byte_offset,
                    epoch,
                )

    @flyc.kernel(
        name=(
            f"moe_tp_gemm1_peer_source_consume_s{source_count}"
            f"_c{chunks_per_source}_n{payload_elements}"
        ),
        known_block_size=[1, 1, 1],
    )
    def peer_source_consume_kernel(
        workspace: fx.Pointer,
        observed_payload: fx.Pointer,
    ):
        local_base = fx.Int64(ptrtoint(workspace))
        observed_addr = fx.Int64(ptrtoint(observed_payload))
        current_addr = local_base + fx.Int64(layout.current_epoch_offset)
        epoch = fx.Int64(comm_ops.load_i64_agent_acquire(current_addr))
        error_addr = local_base + fx.Int64(layout.errors_offset)

        for record_index in range_constexpr(records):
            record_byte_offset = fx.Int64(record_index * 8)
            comm_ops.wait_i64_system_until_equals(
                local_base
                + fx.Int64(layout.ready_offset)
                + record_byte_offset,
                epoch,
            )
            payload_epoch = fx.Int64(
                comm_ops.load_i64_system_acquire(
                    local_base
                    + fx.Int64(layout.payload_epoch_offset)
                    + record_byte_offset
                )
            )
            if payload_epoch != epoch:
                comm_ops.atomic_add_agent(error_addr, fx.Int32(1))

            record_payload_addr = (
                local_base
                + fx.Int64(layout.payload_offset)
                + fx.Int64(record_index * payload_elements * 4)
            )
            observed_record_addr = observed_addr + fx.Int64(
                record_index * payload_elements * 4
            )
            element = fx.Int32(0)
            while element < fx.Int32(payload_elements):
                value = fx.Int32(
                    comm_ops.load_i32_system_acquire(
                        record_payload_addr + fx.Int64(element) * fx.Int64(4)
                    )
                )
                comm_ops.store_i32_system(observed_record_addr, element, value)
                element = element + fx.Int32(1)

            comm_ops.store_i64_global_agent(
                local_base
                + fx.Int64(layout.observed_epoch_offset)
                + record_byte_offset,
                payload_epoch,
            )

    @flyc.jit
    def launch_peer_source_exchange(
        workspace: fx.Pointer,
        peer_rank_data: fx.Int64,
        local_payload: fx.Pointer,
        observed_payload: fx.Pointer,
        rank: fx.Int32,
        skew_iterations: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        peer_source_begin_kernel(workspace).launch(
            grid=(1, 1, 1), block=(1, 1, 1), stream=stream
        )
        peer_source_publish_kernel(
            workspace,
            peer_rank_data,
            local_payload,
            rank,
            skew_iterations,
        ).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)
        peer_source_consume_kernel(workspace, observed_payload).launch(
            grid=(1, 1, 1), block=(1, 1, 1), stream=stream
        )

    return launch_peer_source_exchange


def launch_tp_stage1_peer_source_exchange(
    *,
    workspace: torch.Tensor,
    peer_rank_data: int,
    local_payload: torch.Tensor,
    observed_payload: torch.Tensor,
    rank: int,
    source_count: int,
    chunks_per_source: int,
    payload_elements: int,
    skew_iterations: int = 0,
    stream: torch.cuda.Stream | None = None,
) -> TPStage1PeerSourceLayout:
    """Launch one device-generated peer source/chunk publication epoch."""
    layout = tp_stage1_peer_source_workspace_layout(
        source_count=source_count,
        chunks_per_source=chunks_per_source,
        payload_elements=payload_elements,
    )
    expected_local_shape = (chunks_per_source, payload_elements)
    expected_observed_shape = (
        source_count,
        chunks_per_source,
        payload_elements,
    )
    if workspace.dtype != torch.uint8 or not workspace.is_contiguous():
        raise ValueError("peer source workspace must be contiguous uint8")
    if workspace.numel() < layout.total_bytes:
        raise ValueError(
            f"peer source workspace needs {layout.total_bytes} bytes, "
            f"got {workspace.numel()}"
        )
    if local_payload.dtype != torch.int32 or tuple(local_payload.shape) != (
        expected_local_shape
    ):
        raise ValueError(
            f"local payload must be int32{expected_local_shape}, "
            f"got {local_payload.dtype}{tuple(local_payload.shape)}"
        )
    if observed_payload.dtype != torch.int32 or tuple(observed_payload.shape) != (
        expected_observed_shape
    ):
        raise ValueError(
            f"observed payload must be int32{expected_observed_shape}, "
            f"got {observed_payload.dtype}{tuple(observed_payload.shape)}"
        )
    if not 0 <= rank < source_count:
        raise ValueError(f"rank={rank} is outside source_count={source_count}")
    if peer_rank_data <= 0 or skew_iterations < 0:
        raise ValueError("peer_rank_data must be valid and skew_iterations non-negative")
    if any(
        tensor.device != workspace.device
        for tensor in (local_payload, observed_payload)
    ):
        raise ValueError("workspace and peer payload tensors must share one device")

    launcher = compile_tp_stage1_peer_source_exchange(
        source_count=source_count,
        chunks_per_source=chunks_per_source,
        payload_elements=payload_elements,
    )
    if stream is None:
        stream = torch.cuda.current_stream(workspace.device)
    _run_compiled(
        launcher,
        ptr_arg(workspace),
        fx.Int64(peer_rank_data),
        ptr_arg(local_payload),
        ptr_arg(observed_payload),
        rank,
        skew_iterations,
        fx.Stream(stream),
    )
    return layout


@dataclass(frozen=True)
class TPStage1PeerPayloadLayout:
    """Registered workspace layout for real quantized Stage1 AG payloads."""

    values_offset: int
    scales_offset: int
    topk_ids_offset: int
    topk_weights_offset: int
    entry_offset: int
    current_epoch_offset: int
    payload_epoch_offset: int
    ready_offset: int
    observed_epoch_offset: int
    skew_scratch_offset: int
    errors_offset: int
    total_bytes: int
    source_values_bytes: int
    source_scales_bytes: int
    source_topk_ids_bytes: int
    source_topk_weights_bytes: int
    chunks_per_source: int


def tp_stage1_peer_payload_workspace_layout(
    *,
    source_count: int,
    tokens_per_source: int,
    model_dim: int,
    topk: int,
    tokens_per_chunk: int,
) -> TPStage1PeerPayloadLayout:
    """Return the workspace layout for MXFP8/scales/routes peer AG.

    The four payload regions are independently source-major so the values
    region can be viewed directly as ``[source_count * tokens, model_dim]`` by
    GEMM1.  Every source publishes one full 64-bit epoch per token chunk only
    after all four fields for that chunk are globally visible.
    """
    if source_count <= 1 or source_count > _SOURCE_MASK + 1:
        raise ValueError(f"source_count must be in [2, 64], got {source_count}")
    if tokens_per_source <= 0 or model_dim <= 0 or model_dim % 128:
        raise ValueError(
            "tokens_per_source must be positive and model_dim divisible by 128"
        )
    if topk <= 0 or tokens_per_chunk <= 0:
        raise ValueError("topk and tokens_per_chunk must be positive")

    chunks_per_source = (
        tokens_per_source + tokens_per_chunk - 1
    ) // tokens_per_chunk
    source_values_bytes = tokens_per_source * model_dim
    source_scales_bytes = tokens_per_source * (model_dim // 32)
    source_topk_ids_bytes = tokens_per_source * topk * 4
    source_topk_weights_bytes = tokens_per_source * topk * 4
    for name, size in (
        ("values", source_values_bytes),
        ("scales", source_scales_bytes),
        ("topk_ids", source_topk_ids_bytes),
        ("topk_weights", source_topk_weights_bytes),
    ):
        if size % 4:
            raise ValueError(f"source {name} bytes must be dword aligned, got {size}")

    def align8(value: int) -> int:
        return (value + 7) // 8 * 8

    values_offset = 0
    scales_offset = align8(values_offset + source_count * source_values_bytes)
    topk_ids_offset = align8(
        scales_offset + source_count * source_scales_bytes
    )
    topk_weights_offset = align8(
        topk_ids_offset + source_count * source_topk_ids_bytes
    )
    entry_offset = align8(
        topk_weights_offset + source_count * source_topk_weights_bytes
    )
    current_epoch_offset = entry_offset + 8
    records = source_count * chunks_per_source
    payload_epoch_offset = current_epoch_offset + 8
    ready_offset = payload_epoch_offset + records * 8
    observed_epoch_offset = ready_offset + records * 8
    skew_scratch_offset = observed_epoch_offset + records * 8
    errors_offset = skew_scratch_offset + 8
    total_bytes = align8(errors_offset + 4)
    return TPStage1PeerPayloadLayout(
        values_offset=values_offset,
        scales_offset=scales_offset,
        topk_ids_offset=topk_ids_offset,
        topk_weights_offset=topk_weights_offset,
        entry_offset=entry_offset,
        current_epoch_offset=current_epoch_offset,
        payload_epoch_offset=payload_epoch_offset,
        ready_offset=ready_offset,
        observed_epoch_offset=observed_epoch_offset,
        skew_scratch_offset=skew_scratch_offset,
        errors_offset=errors_offset,
        total_bytes=total_bytes,
        source_values_bytes=source_values_bytes,
        source_scales_bytes=source_scales_bytes,
        source_topk_ids_bytes=source_topk_ids_bytes,
        source_topk_weights_bytes=source_topk_weights_bytes,
        chunks_per_source=chunks_per_source,
    )


@functools.cache
def compile_tp_stage1_peer_payload_exchange(
    *,
    source_count: int,
    tokens_per_source: int,
    model_dim: int,
    topk: int,
    tokens_per_chunk: int,
    blocks_per_destination: int = 40,
):
    """Compile chunk-ready peer AG for the real quantized Stage1 payload.

    Payload data uses 16-byte vector buffer operations.  Only the generation
    publication is system-scope release: applying release ordering to every
    dword is correct but destroys target-shape bandwidth.
    """
    if blocks_per_destination <= 0:
        raise ValueError("blocks_per_destination must be positive")
    layout = tp_stage1_peer_payload_workspace_layout(
        source_count=source_count,
        tokens_per_source=tokens_per_source,
        model_dim=model_dim,
        topk=topk,
        tokens_per_chunk=tokens_per_chunk,
    )
    chunks_per_source = layout.chunks_per_source
    block = 256
    values_words_per_token = model_dim // 4
    scales_words_per_token = (model_dim // 32) // 4
    ids_words_per_token = topk
    weights_words_per_token = topk
    records = source_count * chunks_per_source
    name_suffix = (
        f"s{source_count}_m{tokens_per_source}_h{model_dim}_k{topk}"
        f"_c{tokens_per_chunk}_b{blocks_per_destination}_pack4"
    )
    # The launchers close over ``layout`` through their kernels.  FlyDSL's JIT
    # cache does not recursively fingerprint dataclass fields, so without an
    # explicit scalar tag a launcher compiled for one shape can silently reuse
    # another shape's control offsets.  Keep every geometry input here even
    # when some values can currently be derived from the others.
    cache_tag = (
        source_count,
        tokens_per_source,
        model_dim,
        topk,
        tokens_per_chunk,
        blocks_per_destination,
        layout.entry_offset,
        layout.current_epoch_offset,
        layout.payload_epoch_offset,
        layout.ready_offset,
        layout.observed_epoch_offset,
        layout.skew_scratch_offset,
        layout.errors_offset,
        layout.total_bytes,
    )

    @flyc.kernel(
        name=f"moe_tp_gemm1_peer_payload_begin_{name_suffix}",
        known_block_size=[64, 1, 1],
    )
    def peer_payload_begin_kernel(workspace: fx.Pointer):
        if fx.Int32(gpu.thread_idx.x) == fx.Int32(0):
            local_base = fx.Int64(ptrtoint(workspace))
            entry_addr = local_base + fx.Int64(layout.entry_offset)
            current_addr = local_base + fx.Int64(layout.current_epoch_offset)
            epoch = fx.Int64(
                comm_ops.atomic_add_agent(entry_addr, fx.Int64(1))
            ) + fx.Int64(1)
            comm_ops.store_i64_global_agent(current_addr, epoch)

    @flyc.kernel(
        name=f"moe_tp_gemm1_peer_payload_skew_{name_suffix}",
        known_block_size=[64, 1, 1],
    )
    def peer_payload_skew_kernel(
        workspace: fx.Pointer,
        rank: fx.Int32,
        skew_iterations: fx.Int32,
    ):
        if (
            fx.Int32(gpu.thread_idx.x) == fx.Int32(0)
            and rank == fx.Int32(source_count - 1)
        ):
            local_base = fx.Int64(ptrtoint(workspace))
            current_addr = local_base + fx.Int64(layout.current_epoch_offset)
            iteration = fx.Int32(0)
            accumulator = fx.Int64(0)
            while iteration < skew_iterations:
                accumulator = accumulator + fx.Int64(
                    comm_ops.load_i64_agent_acquire(current_addr)
                )
                iteration = iteration + fx.Int32(1)
            comm_ops.store_i64_global_agent(
                local_base + fx.Int64(layout.skew_scratch_offset), accumulator
            )

    @flyc.kernel(
        name=f"moe_tp_gemm1_peer_payload_copy_{name_suffix}",
        known_block_size=[block, 1, 1],
    )
    def peer_payload_copy_kernel(
        peer_rank_data: fx.Int64,
        local_values: fx.Pointer,
        local_scales: fx.Pointer,
        local_topk_ids: fx.Pointer,
        local_topk_weights: fx.Pointer,
        rank: fx.Int32,
        chunk: fx.Int32,
    ):
        flat_block = fx.Int32(gpu.block_idx.x)
        destination = flat_block // fx.Int32(blocks_per_destination)
        destination_block = flat_block % fx.Int32(blocks_per_destination)
        if destination < fx.Int32(source_count):
            peer_table = buffer_ops.create_buffer_resource_from_addr(
                peer_rank_data, num_records_bytes=source_count * 8
            )
            peer_base = fx.Int64(
                buffer_ops.buffer_load(
                    peer_table,
                    destination,
                    vec_width=1,
                    dtype=fx.Int64,
                )
            )
            token_start = chunk * fx.Int32(tokens_per_chunk)
            remaining = fx.Int32(tokens_per_source) - token_start
            token_count = (remaining < fx.Int32(tokens_per_chunk)).select(
                remaining, fx.Int32(tokens_per_chunk)
            )
            flat_thread = (
                destination_block * fx.Int32(block)
                + fx.Int32(gpu.thread_idx.x)
            )
            thread_stride = fx.Int32(blocks_per_destination * block)
            local_values_resource = buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(ptrtoint(local_values)),
                num_records_bytes=layout.source_values_bytes,
            )
            local_scales_resource = buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(ptrtoint(local_scales)),
                num_records_bytes=layout.source_scales_bytes,
            )
            local_ids_resource = buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(ptrtoint(local_topk_ids)),
                num_records_bytes=layout.source_topk_ids_bytes,
            )
            local_weights_resource = buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(ptrtoint(local_topk_weights)),
                num_records_bytes=layout.source_topk_weights_bytes,
            )
            peer_values_resource = buffer_ops.create_buffer_resource_from_addr(
                peer_base + fx.Int64(layout.values_offset),
                num_records_bytes=source_count * layout.source_values_bytes,
            )
            peer_scales_resource = buffer_ops.create_buffer_resource_from_addr(
                peer_base + fx.Int64(layout.scales_offset),
                num_records_bytes=source_count * layout.source_scales_bytes,
            )
            peer_ids_resource = buffer_ops.create_buffer_resource_from_addr(
                peer_base + fx.Int64(layout.topk_ids_offset),
                num_records_bytes=source_count * layout.source_topk_ids_bytes,
            )
            peer_weights_resource = buffer_ops.create_buffer_resource_from_addr(
                peer_base + fx.Int64(layout.topk_weights_offset),
                num_records_bytes=source_count * layout.source_topk_weights_bytes,
            )

            # Each field is copied independently so source-major field regions
            # stay directly consumable.  A scalar tail preserves small/odd test
            # shapes without penalizing the aligned target shape.
            values_words = token_count * fx.Int32(values_words_per_token)
            values_packs = values_words // fx.Int32(4)
            pack = flat_thread
            while pack < values_packs:
                field_word = pack * fx.Int32(4)
                source_word = (
                    token_start * fx.Int32(values_words_per_token) + field_word
                )
                destination_word = (
                    (rank * fx.Int32(tokens_per_source) + token_start)
                    * fx.Int32(values_words_per_token)
                    + field_word
                )
                value = buffer_ops.buffer_load(
                    local_values_resource,
                    source_word,
                    vec_width=4,
                    dtype=fx.Int32,
                )
                buffer_ops.buffer_store(
                    value, peer_values_resource, destination_word
                )
                pack = pack + thread_stride
            word = values_packs * fx.Int32(4) + flat_thread
            while word < values_words:
                source_word = (
                    token_start * fx.Int32(values_words_per_token) + word
                )
                destination_word = (
                    (rank * fx.Int32(tokens_per_source) + token_start)
                    * fx.Int32(values_words_per_token)
                    + word
                )
                value = buffer_ops.buffer_load(
                    local_values_resource,
                    source_word,
                    vec_width=1,
                    dtype=fx.Int32,
                )
                buffer_ops.buffer_store(
                    value, peer_values_resource, destination_word
                )
                word = word + thread_stride

            scales_words = token_count * fx.Int32(scales_words_per_token)
            scales_packs = scales_words // fx.Int32(4)
            pack = flat_thread
            while pack < scales_packs:
                field_word = pack * fx.Int32(4)
                source_word = (
                    token_start * fx.Int32(scales_words_per_token) + field_word
                )
                destination_word = (
                    (rank * fx.Int32(tokens_per_source) + token_start)
                    * fx.Int32(scales_words_per_token)
                    + field_word
                )
                value = buffer_ops.buffer_load(
                    local_scales_resource,
                    source_word,
                    vec_width=4,
                    dtype=fx.Int32,
                )
                buffer_ops.buffer_store(
                    value, peer_scales_resource, destination_word
                )
                pack = pack + thread_stride
            word = scales_packs * fx.Int32(4) + flat_thread
            while word < scales_words:
                source_word = (
                    token_start * fx.Int32(scales_words_per_token) + word
                )
                destination_word = (
                    (rank * fx.Int32(tokens_per_source) + token_start)
                    * fx.Int32(scales_words_per_token)
                    + word
                )
                value = buffer_ops.buffer_load(
                    local_scales_resource,
                    source_word,
                    vec_width=1,
                    dtype=fx.Int32,
                )
                buffer_ops.buffer_store(
                    value, peer_scales_resource, destination_word
                )
                word = word + thread_stride

            ids_words = token_count * fx.Int32(ids_words_per_token)
            ids_packs = ids_words // fx.Int32(4)
            pack = flat_thread
            while pack < ids_packs:
                field_word = pack * fx.Int32(4)
                source_word = (
                    token_start * fx.Int32(ids_words_per_token) + field_word
                )
                destination_word = (
                    (rank * fx.Int32(tokens_per_source) + token_start)
                    * fx.Int32(ids_words_per_token)
                    + field_word
                )
                value = buffer_ops.buffer_load(
                    local_ids_resource,
                    source_word,
                    vec_width=4,
                    dtype=fx.Int32,
                )
                buffer_ops.buffer_store(value, peer_ids_resource, destination_word)
                pack = pack + thread_stride
            word = ids_packs * fx.Int32(4) + flat_thread
            while word < ids_words:
                source_word = (
                    token_start * fx.Int32(ids_words_per_token) + word
                )
                destination_word = (
                    (rank * fx.Int32(tokens_per_source) + token_start)
                    * fx.Int32(ids_words_per_token)
                    + word
                )
                value = buffer_ops.buffer_load(
                    local_ids_resource,
                    source_word,
                    vec_width=1,
                    dtype=fx.Int32,
                )
                buffer_ops.buffer_store(value, peer_ids_resource, destination_word)
                word = word + thread_stride

            weights_words = token_count * fx.Int32(weights_words_per_token)
            weights_packs = weights_words // fx.Int32(4)
            pack = flat_thread
            while pack < weights_packs:
                field_word = pack * fx.Int32(4)
                source_word = (
                    token_start * fx.Int32(weights_words_per_token) + field_word
                )
                destination_word = (
                    (rank * fx.Int32(tokens_per_source) + token_start)
                    * fx.Int32(weights_words_per_token)
                    + field_word
                )
                value = buffer_ops.buffer_load(
                    local_weights_resource,
                    source_word,
                    vec_width=4,
                    dtype=fx.Int32,
                )
                buffer_ops.buffer_store(
                    value, peer_weights_resource, destination_word
                )
                pack = pack + thread_stride
            word = weights_packs * fx.Int32(4) + flat_thread
            while word < weights_words:
                source_word = (
                    token_start * fx.Int32(weights_words_per_token) + word
                )
                destination_word = (
                    (rank * fx.Int32(tokens_per_source) + token_start)
                    * fx.Int32(weights_words_per_token)
                    + word
                )
                value = buffer_ops.buffer_load(
                    local_weights_resource,
                    source_word,
                    vec_width=1,
                    dtype=fx.Int32,
                )
                buffer_ops.buffer_store(
                    value, peer_weights_resource, destination_word
                )
                word = word + thread_stride

    @flyc.kernel(
        name=f"moe_tp_gemm1_peer_payload_publish_{name_suffix}",
        known_block_size=[64, 1, 1],
    )
    def peer_payload_publish_kernel(
        workspace: fx.Pointer,
        peer_rank_data: fx.Int64,
        rank: fx.Int32,
        chunk: fx.Int32,
    ):
        destination = fx.Int32(gpu.thread_idx.x)
        if destination < fx.Int32(source_count):
            comm_ops.fence_system_release()
            local_base = fx.Int64(ptrtoint(workspace))
            epoch = fx.Int64(
                comm_ops.load_i64_agent_acquire(
                    local_base + fx.Int64(layout.current_epoch_offset)
                )
            )
            peer_table = buffer_ops.create_buffer_resource_from_addr(
                peer_rank_data, num_records_bytes=source_count * 8
            )
            peer_base = fx.Int64(
                buffer_ops.buffer_load(
                    peer_table,
                    destination,
                    vec_width=1,
                    dtype=fx.Int64,
                )
            )
            record = rank * fx.Int32(chunks_per_source) + chunk
            record_byte_offset = fx.Int64(record) * fx.Int64(8)
            comm_ops.store_i64_global_system(
                peer_base
                + fx.Int64(layout.payload_epoch_offset)
                + record_byte_offset,
                epoch,
            )
            comm_ops.store_i64_global_system(
                peer_base + fx.Int64(layout.ready_offset) + record_byte_offset,
                epoch,
            )

    @flyc.kernel(
        name=f"moe_tp_gemm1_peer_payload_wait_{name_suffix}",
        known_block_size=[64, 1, 1],
    )
    def peer_payload_wait_kernel(workspace: fx.Pointer):
        local_base = fx.Int64(ptrtoint(workspace))
        epoch = fx.Int64(
            comm_ops.load_i64_agent_acquire(
                local_base + fx.Int64(layout.current_epoch_offset)
            )
        )
        error_addr = local_base + fx.Int64(layout.errors_offset)
        record = fx.Int32(gpu.thread_idx.x)
        while record < fx.Int32(records):
            record_byte_offset = fx.Int64(record) * fx.Int64(8)
            comm_ops.wait_i64_system_until_equals(
                local_base
                + fx.Int64(layout.ready_offset)
                + record_byte_offset,
                epoch,
            )
            payload_epoch = fx.Int64(
                comm_ops.load_i64_system_acquire(
                    local_base
                    + fx.Int64(layout.payload_epoch_offset)
                    + record_byte_offset
                )
            )
            if payload_epoch != epoch:
                comm_ops.atomic_add_agent(error_addr, fx.Int32(1))
            comm_ops.store_i64_global_agent(
                local_base
                + fx.Int64(layout.observed_epoch_offset)
                + record_byte_offset,
                payload_epoch,
            )
            record = record + fx.Int32(64)

    @flyc.jit
    def launch_peer_payload_begin(
        workspace: fx.Pointer,
        rank: fx.Int32,
        skew_iterations: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = cache_tag
        peer_payload_begin_kernel(workspace).launch(
            grid=(1, 1, 1), block=(64, 1, 1), stream=stream
        )
        peer_payload_skew_kernel(
            workspace, rank, skew_iterations
        ).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    @flyc.jit
    def launch_peer_payload_chunk(
        workspace: fx.Pointer,
        peer_rank_data: fx.Int64,
        local_values: fx.Pointer,
        local_scales: fx.Pointer,
        local_topk_ids: fx.Pointer,
        local_topk_weights: fx.Pointer,
        rank: fx.Int32,
        chunk: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = cache_tag
        peer_payload_copy_kernel(
            peer_rank_data,
            local_values,
            local_scales,
            local_topk_ids,
            local_topk_weights,
            rank,
            chunk,
        ).launch(
            grid=(source_count * blocks_per_destination, 1, 1),
            block=(block, 1, 1),
            stream=stream,
        )
        peer_payload_publish_kernel(
            workspace,
            peer_rank_data,
            rank,
            chunk,
        ).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    @flyc.jit
    def launch_peer_payload_wait(
        workspace: fx.Pointer,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = cache_tag
        peer_payload_wait_kernel(workspace).launch(
            grid=(1, 1, 1), block=(64, 1, 1), stream=stream
        )

    return (
        launch_peer_payload_begin,
        launch_peer_payload_chunk,
        launch_peer_payload_wait,
    )


def launch_tp_stage1_peer_payload_exchange(
    *,
    workspace: torch.Tensor,
    peer_rank_data: int,
    local_values: torch.Tensor,
    local_scales: torch.Tensor,
    local_topk_ids: torch.Tensor,
    local_topk_weights: torch.Tensor,
    rank: int,
    source_count: int,
    tokens_per_chunk: int,
    skew_iterations: int = 0,
    blocks_per_destination: int = 40,
    stream: torch.cuda.Stream | None = None,
) -> TPStage1PeerPayloadLayout:
    """Publish and gather one real quantized Stage1 payload generation."""
    if local_values.ndim != 2 or local_values.element_size() != 1:
        raise ValueError("local_values must be a contiguous 2-D byte-wide tensor")
    tokens_per_source, model_dim = map(int, local_values.shape)
    if local_topk_ids.ndim != 2:
        raise ValueError("local_topk_ids must be 2-D")
    topk = int(local_topk_ids.shape[1])
    layout = tp_stage1_peer_payload_workspace_layout(
        source_count=source_count,
        tokens_per_source=tokens_per_source,
        model_dim=model_dim,
        topk=topk,
        tokens_per_chunk=tokens_per_chunk,
    )
    expected_scales_shape = (tokens_per_source, model_dim // 32)
    expected_routes_shape = (tokens_per_source, topk)
    if (
        tuple(local_scales.shape) != expected_scales_shape
        or local_scales.element_size() != 1
    ):
        raise ValueError(
            f"local_scales must be byte-wide {expected_scales_shape}, got "
            f"{local_scales.dtype}{tuple(local_scales.shape)}"
        )
    if (
        local_topk_ids.dtype != torch.int32
        or tuple(local_topk_ids.shape) != expected_routes_shape
    ):
        raise ValueError(
            f"local_topk_ids must be int32{expected_routes_shape}, got "
            f"{local_topk_ids.dtype}{tuple(local_topk_ids.shape)}"
        )
    if (
        local_topk_weights.dtype != torch.float32
        or tuple(local_topk_weights.shape) != expected_routes_shape
    ):
        raise ValueError(
            f"local_topk_weights must be float32{expected_routes_shape}, got "
            f"{local_topk_weights.dtype}{tuple(local_topk_weights.shape)}"
        )
    tensors = (
        workspace,
        local_values,
        local_scales,
        local_topk_ids,
        local_topk_weights,
    )
    if workspace.dtype != torch.uint8 or not workspace.is_contiguous():
        raise ValueError("peer payload workspace must be contiguous uint8")
    if workspace.numel() < layout.total_bytes:
        raise ValueError(
            f"peer payload workspace needs {layout.total_bytes} bytes, "
            f"got {workspace.numel()}"
        )
    if any(not tensor.is_contiguous() for tensor in tensors[1:]):
        raise ValueError("all local Stage1 payload tensors must be contiguous")
    if any(tensor.device != workspace.device for tensor in tensors[1:]):
        raise ValueError("workspace and local Stage1 payloads must share one device")
    if not 0 <= rank < source_count:
        raise ValueError(f"rank={rank} is outside source_count={source_count}")
    if peer_rank_data <= 0 or skew_iterations < 0:
        raise ValueError("peer_rank_data must be valid and skew_iterations non-negative")

    begin_launcher, chunk_launcher, wait_launcher = (
        compile_tp_stage1_peer_payload_exchange(
            source_count=source_count,
            tokens_per_source=tokens_per_source,
            model_dim=model_dim,
            topk=topk,
            tokens_per_chunk=tokens_per_chunk,
            blocks_per_destination=blocks_per_destination,
        )
    )
    if stream is None:
        stream = torch.cuda.current_stream(workspace.device)
    _run_compiled(
        begin_launcher,
        ptr_arg(workspace),
        rank,
        skew_iterations,
        fx.Stream(stream),
    )
    for chunk in range(layout.chunks_per_source):
        _run_compiled(
            chunk_launcher,
            ptr_arg(workspace),
            fx.Int64(peer_rank_data),
            ptr_arg(local_values),
            ptr_arg(local_scales),
            ptr_arg(local_topk_ids),
            ptr_arg(local_topk_weights),
            rank,
            chunk,
            fx.Stream(stream),
        )
    _run_compiled(
        wait_launcher,
        ptr_arg(workspace),
        fx.Stream(stream),
    )
    return layout


@functools.cache
def compile_moe_tp_gemm1_all_ready(
    *,
    source_count: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage1: bool,
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    act: str,
    situ_beta: float,
    situ_linear_beta: float,
    persist_m: int,
    use_async_copy: bool,
    k_batch: int,
    waves_per_eu: int,
    b_nt: int,
    gate_mode: str,
    model_dim_pad: int,
    inter_dim_pad: int,
    enable_bias: bool,
    a_scale_one: bool,
    xcd_swizzle: int,
    k_wave: int,
    v2_output_layout: bool = False,
):
    """Compile the default-off all-ready TP Stage1 compute prerequisite."""
    return compile_mixed_moe_gemm1_common(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=doweight_stage1,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        out_dtype=out_dtype,
        act=act,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        persist_m=persist_m,
        use_async_copy=use_async_copy,
        waves_per_eu=waves_per_eu,
        k_batch=k_batch,
        b_nt=b_nt,
        gate_mode=GateMode(gate_mode),
        model_dim_pad=model_dim_pad,
        inter_dim_pad=inter_dim_pad,
        enable_bias=enable_bias,
        a_scale_one=a_scale_one,
        xcd_swizzle=xcd_swizzle,
        k_wave=k_wave,
        v2_output_layout=v2_output_layout,
        tp_source_count=source_count,
    )
