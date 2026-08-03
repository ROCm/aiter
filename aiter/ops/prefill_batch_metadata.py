# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Operator-specific metadata built from shared prefill batch primitives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import torch

__all__ = [
    "CausalConvPrefillMetadata",
    "ChunkGrid",
    "GatedDeltaRuleChunkSchedule",
    "GatedDeltaRulePrefillMetadata",
    "PrefillBatchLayout",
    "build_causal_conv_prefill_metadata",
    "build_gated_delta_rule_prefill_metadata",
]


def _tensor_version(tensor: torch.Tensor) -> int | None:
    try:
        return tensor._version
    except (AttributeError, RuntimeError):
        return None


def _normalize_seq_lens(
    seq_lens_cpu: Sequence[int], cu_seqlens: torch.Tensor
) -> tuple[int, ...]:
    seq_lens = tuple(int(length) for length in seq_lens_cpu)
    if not seq_lens:
        raise ValueError("`seq_lens_cpu` must contain at least one sequence.")
    if any(length < 0 for length in seq_lens):
        raise ValueError("`seq_lens_cpu` must contain non-negative lengths.")
    if cu_seqlens.dim() != 1:
        raise ValueError(
            "The cumulative sequence-length tensor must be one-dimensional."
        )
    if cu_seqlens.numel() != len(seq_lens) + 1:
        raise ValueError(
            f"The cumulative sequence-length tensor describes "
            f"{cu_seqlens.numel() - 1} sequences, but `seq_lens_cpu` has "
            f"{len(seq_lens)}."
        )
    return seq_lens


def _host_cu_seqlens(seq_lens_cpu: Sequence[int]) -> list[int]:
    cumulative = [0]
    for length in seq_lens_cpu:
        cumulative.append(cumulative[-1] + length)
    return cumulative


def _assert_layout_matches(
    cu_seqlens: torch.Tensor, kernel_cu_seqlens: torch.Tensor
) -> None:
    expected = kernel_cu_seqlens.to(cu_seqlens.dtype)
    matches = torch.all(cu_seqlens == expected)
    if cu_seqlens.device.type == "cuda":
        torch._assert_async(
            matches,
            "`seq_lens_cpu` does not match the cumulative sequence lengths.",
        )
    elif not bool(matches):
        raise ValueError(
            "`seq_lens_cpu` does not match the cumulative sequence lengths."
        )


def _normalize_layout(
    seq_lens_cpu: Sequence[int],
    cu_seqlens: torch.Tensor,
    *,
    kernel_cu_seqlens: torch.Tensor | None = None,
) -> PrefillBatchLayout:
    seq_lens = _normalize_seq_lens(seq_lens_cpu, cu_seqlens)
    if kernel_cu_seqlens is None:
        kernel_cu_seqlens = torch.tensor(
            _host_cu_seqlens(seq_lens),
            dtype=torch.int32,
            device=cu_seqlens.device,
        )
    _assert_layout_matches(cu_seqlens, kernel_cu_seqlens)
    return PrefillBatchLayout(
        seq_lens_cpu=seq_lens,
        cu_seqlens=cu_seqlens,
        cu_seqlens_version=_tensor_version(cu_seqlens),
        kernel_cu_seqlens=kernel_cu_seqlens,
    )


@dataclass(frozen=True)
class PrefillBatchLayout:
    """One sequence view used by an individual prefill operator.

    Typed metadata is layout-specific and cannot be built or used while
    HIP/CUDA graph capture is active.
    """

    seq_lens_cpu: tuple[int, ...]
    cu_seqlens: torch.Tensor
    cu_seqlens_version: int | None
    kernel_cu_seqlens: torch.Tensor

    def validate(
        self,
        cu_seqlens: torch.Tensor,
        *,
        total_tokens: int | None = None,
        num_sequences: int | None = None,
    ) -> None:
        if (
            cu_seqlens.device.type == "cuda"
            and torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError(
                "Typed prefill metadata cannot be used during HIP/CUDA graph capture."
            )
        if self.cu_seqlens is not cu_seqlens:
            raise ValueError(
                "Prefill metadata is bound to the exact sequence-offset tensor "
                "used to build it; reuse that tensor or rebuild the metadata."
            )
        if (
            self.cu_seqlens_version is not None
            and _tensor_version(cu_seqlens) != self.cu_seqlens_version
        ):
            raise ValueError(
                "The sequence-offset tensor was modified after its metadata was "
                "built; rebuild the metadata."
            )
        if num_sequences is not None and len(self.seq_lens_cpu) != num_sequences:
            raise ValueError(
                f"The metadata describes {len(self.seq_lens_cpu)} sequences, "
                f"expected {num_sequences}."
            )
        if total_tokens is not None and sum(self.seq_lens_cpu) != total_tokens:
            raise ValueError(
                f"The metadata describes {sum(self.seq_lens_cpu)} tokens, "
                f"expected {total_tokens}."
            )


@dataclass(frozen=True)
class ChunkGrid:
    """Generic flattened ``(sequence_id, chunk_id)`` launch grid."""

    block_size: int
    total_chunks: int
    max_seq_chunks: int
    chunk_counts: tuple[int, ...]
    sequence_ids: torch.Tensor
    chunk_ids: torch.Tensor


@dataclass(frozen=True)
class CausalConvPrefillMetadata:
    """Eager causal-conv schedules for one ``query_start_loc`` layout."""

    layout: PrefillBatchLayout
    chunk_grids: Mapping[int, ChunkGrid]

    def get_chunk_grid(self, block_size: int) -> ChunkGrid:
        try:
            return self.chunk_grids[block_size]
        except KeyError as error:
            raise ValueError(
                f"No causal-conv schedule was prebuilt for block size {block_size}. "
                f"Available block sizes: {sorted(self.chunk_grids)}."
            ) from error

    def validate(
        self,
        query_start_loc: torch.Tensor,
        *,
        total_tokens: int,
        num_sequences: int,
    ) -> None:
        self.layout.validate(
            query_start_loc,
            total_tokens=total_tokens,
            num_sequences=num_sequences,
        )


@dataclass(frozen=True)
class GatedDeltaRuleChunkSchedule:
    """GDR launch grid plus hidden-state sequence offsets."""

    chunk_size: int
    num_decodes: int
    num_decode_tokens: int
    n_prefill: int
    total_prefill_tokens: int
    grid: ChunkGrid
    chunk_offsets: torch.Tensor
    kernel_cu_seqlens: torch.Tensor

    @property
    def total_chunks(self) -> int:
        return self.grid.total_chunks

    @property
    def max_seq_chunks(self) -> int:
        return self.grid.max_seq_chunks

    @property
    def sequence_ids(self) -> torch.Tensor:
        return self.grid.sequence_ids

    @property
    def chunk_ids(self) -> torch.Tensor:
        return self.grid.chunk_ids


@dataclass(frozen=True)
class GatedDeltaRulePrefillMetadata:
    """Reusable, eagerly built metadata shared by GDR K1--K6."""

    layout: PrefillBatchLayout
    schedule: GatedDeltaRuleChunkSchedule

    def get_chunk_schedule(
        self,
        chunk_size: int,
        *,
        num_decodes: int = 0,
        num_decode_tokens: int = 0,
    ) -> GatedDeltaRuleChunkSchedule:
        schedule = self.schedule
        if (
            chunk_size != schedule.chunk_size
            or num_decodes != schedule.num_decodes
            or num_decode_tokens != schedule.num_decode_tokens
        ):
            raise ValueError(
                "The GDR metadata was built with "
                f"chunk_size={schedule.chunk_size}, "
                f"num_decodes={schedule.num_decodes}, and "
                f"num_decode_tokens={schedule.num_decode_tokens}, but the call "
                f"requested chunk_size={chunk_size}, num_decodes={num_decodes}, "
                f"and num_decode_tokens={num_decode_tokens}; rebuild the metadata "
                "for the requested schedule."
            )
        return schedule

    def validate(
        self,
        *,
        cu_seqlens: torch.Tensor,
        chunk_size: int,
        num_decodes: int,
        num_decode_tokens: int,
        total_prefill_tokens: int | None = None,
        num_sequences: int | None = None,
    ) -> None:
        self.layout.validate(cu_seqlens, num_sequences=num_sequences)
        schedule = self.get_chunk_schedule(
            chunk_size,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
        )
        if (
            total_prefill_tokens is not None
            and schedule.total_prefill_tokens != total_prefill_tokens
        ):
            raise ValueError(
                f"The GDR metadata describes {schedule.total_prefill_tokens} "
                f"prefill tokens, expected {total_prefill_tokens}."
            )


def _chunk_counts(
    seq_lens_cpu: Sequence[int], block_size: int
) -> tuple[tuple[int, ...], int, int]:
    if block_size <= 0:
        raise ValueError(f"`block_size` must be positive, got {block_size}.")
    counts = tuple((length + block_size - 1) // block_size for length in seq_lens_cpu)
    return counts, sum(counts), max(counts)


def _build_chunk_grid(
    seq_lens_cpu: Sequence[int],
    *,
    block_size: int,
    device: torch.device,
) -> ChunkGrid:
    counts, total_chunks, max_seq_chunks = _chunk_counts(seq_lens_cpu, block_size)
    sequence_ids_cpu: list[int] = []
    chunk_ids_cpu: list[int] = []
    for sequence_id, num_chunks in enumerate(counts):
        sequence_ids_cpu.extend([sequence_id] * num_chunks)
        chunk_ids_cpu.extend(range(num_chunks))

    packed = torch.tensor(
        sequence_ids_cpu + chunk_ids_cpu,
        dtype=torch.int32,
        device=device,
    )
    return ChunkGrid(
        block_size=block_size,
        total_chunks=total_chunks,
        max_seq_chunks=max_seq_chunks,
        chunk_counts=counts,
        sequence_ids=packed[:total_chunks],
        chunk_ids=packed[total_chunks:],
    )


def build_causal_conv_prefill_metadata(
    seq_lens_cpu: Sequence[int],
    *,
    query_start_loc: torch.Tensor,
    block_sizes: Sequence[int] = (8, 16, 32, 64),
    shared_chunk_grids: Mapping[int, ChunkGrid] | None = None,
) -> CausalConvPrefillMetadata:
    """Eagerly build causal-conv schedules outside the forward hot path."""
    if (
        query_start_loc.device.type == "cuda"
        and torch.cuda.is_current_stream_capturing()
    ):
        raise RuntimeError(
            "Causal-conv metadata cannot be built during HIP/CUDA graph capture."
        )
    layout = _normalize_layout(seq_lens_cpu, query_start_loc)
    grids = dict(shared_chunk_grids or {})
    for block_size, existing in grids.items():
        counts, total_chunks, max_seq_chunks = _chunk_counts(
            layout.seq_lens_cpu, block_size
        )
        mismatches = []
        if existing.block_size != block_size:
            mismatches.append(
                f"block_size={existing.block_size}, expected {block_size}"
            )
        if existing.chunk_counts != counts:
            mismatches.append("chunk_counts differ")
        if existing.total_chunks != total_chunks:
            mismatches.append(
                f"total_chunks={existing.total_chunks}, expected {total_chunks}"
            )
        if existing.max_seq_chunks != max_seq_chunks:
            mismatches.append(
                f"max_seq_chunks={existing.max_seq_chunks}, "
                f"expected {max_seq_chunks}"
            )
        if existing.sequence_ids.device != query_start_loc.device:
            mismatches.append(
                f"sequence_ids.device={existing.sequence_ids.device}, "
                f"expected {query_start_loc.device}"
            )
        if existing.chunk_ids.device != query_start_loc.device:
            mismatches.append(
                f"chunk_ids.device={existing.chunk_ids.device}, "
                f"expected {query_start_loc.device}"
            )
        if existing.sequence_ids.dtype != torch.int32:
            mismatches.append(
                f"sequence_ids.dtype={existing.sequence_ids.dtype}, "
                "expected torch.int32"
            )
        if existing.chunk_ids.dtype != torch.int32:
            mismatches.append(
                f"chunk_ids.dtype={existing.chunk_ids.dtype}, expected torch.int32"
            )
        if existing.sequence_ids.numel() != total_chunks:
            mismatches.append(
                f"sequence_ids.numel()={existing.sequence_ids.numel()}, "
                f"expected {total_chunks}"
            )
        if existing.chunk_ids.numel() != total_chunks:
            mismatches.append(
                f"chunk_ids.numel()={existing.chunk_ids.numel()}, "
                f"expected {total_chunks}"
            )
        if mismatches:
            raise ValueError(
                f"The shared chunk grid for block size {block_size} does not "
                f"match the causal-conv sequence layout: {'; '.join(mismatches)}. "
                "Omit it or rebuild it for this layout and device."
            )
    for block_size in dict.fromkeys(int(size) for size in block_sizes):
        existing = grids.get(block_size)
        if existing is not None:
            continue
        grids[block_size] = _build_chunk_grid(
            layout.seq_lens_cpu,
            block_size=block_size,
            device=query_start_loc.device,
        )
    return CausalConvPrefillMetadata(
        layout=layout,
        chunk_grids=MappingProxyType(grids),
    )


def build_gated_delta_rule_prefill_metadata(
    seq_lens_cpu: Sequence[int],
    *,
    cu_seqlens: torch.Tensor,
    chunk_size: int = 64,
    num_decodes: int = 0,
    num_decode_tokens: int = 0,
) -> GatedDeltaRulePrefillMetadata:
    """Build reusable GDR prefill metadata on the ``cu_seqlens`` device."""
    if cu_seqlens.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "GDR metadata cannot be built during HIP/CUDA graph capture."
        )
    normalized_seq_lens = _normalize_seq_lens(seq_lens_cpu, cu_seqlens)
    if not 0 <= num_decodes < len(normalized_seq_lens):
        raise ValueError(
            f"`num_decodes` must be in [0, {len(normalized_seq_lens) - 1}], "
            f"got {num_decodes}."
        )
    expected_decode_tokens = sum(normalized_seq_lens[:num_decodes])
    if expected_decode_tokens != num_decode_tokens:
        raise ValueError(
            "`num_decode_tokens` must equal the sum of the leading decode "
            f"sequence lengths: expected {expected_decode_tokens}, "
            f"got {num_decode_tokens}."
        )

    prefill_lens = normalized_seq_lens[num_decodes:]
    counts, total_chunks, max_seq_chunks = _chunk_counts(prefill_lens, chunk_size)
    sequence_ids_cpu: list[int] = []
    chunk_ids_cpu: list[int] = []
    chunk_offsets_cpu = [0]
    kernel_cu_seqlens_cpu = [0]
    for sequence_id, (length, num_chunks) in enumerate(
        zip(prefill_lens, counts, strict=True)
    ):
        sequence_ids_cpu.extend([sequence_id] * num_chunks)
        chunk_ids_cpu.extend(range(num_chunks))
        chunk_offsets_cpu.append(chunk_offsets_cpu[-1] + num_chunks)
        kernel_cu_seqlens_cpu.append(kernel_cu_seqlens_cpu[-1] + length)

    packed = torch.tensor(
        sequence_ids_cpu
        + chunk_ids_cpu
        + chunk_offsets_cpu
        + _host_cu_seqlens(normalized_seq_lens)
        + kernel_cu_seqlens_cpu,
        dtype=torch.int32,
        device=cu_seqlens.device,
    )
    chunk_ids_end = 2 * total_chunks
    offsets_end = chunk_ids_end + len(chunk_offsets_cpu)
    source_cu_end = offsets_end + len(normalized_seq_lens) + 1
    grid = ChunkGrid(
        block_size=chunk_size,
        total_chunks=total_chunks,
        max_seq_chunks=max_seq_chunks,
        chunk_counts=counts,
        sequence_ids=packed[:total_chunks],
        chunk_ids=packed[total_chunks:chunk_ids_end],
    )
    layout = _normalize_layout(
        normalized_seq_lens,
        cu_seqlens,
        kernel_cu_seqlens=packed[offsets_end:source_cu_end],
    )
    schedule = GatedDeltaRuleChunkSchedule(
        chunk_size=chunk_size,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        n_prefill=len(prefill_lens),
        total_prefill_tokens=sum(prefill_lens),
        grid=grid,
        chunk_offsets=packed[chunk_ids_end:offsets_end],
        kernel_cu_seqlens=packed[source_cu_end:],
    )
    return GatedDeltaRulePrefillMetadata(layout=layout, schedule=schedule)
