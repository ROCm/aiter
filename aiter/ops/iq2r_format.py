# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Versioned storage contract for AITER's native-basis IQ2R weights.

This module is intentionally CPU-only and has no dependency on a compiled
extension.  Checkpoint tooling can therefore validate IQ2R metadata and byte
buffers before allocating GPU memory or launching a kernel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

import torch
from torch import Tensor

IQ2R_SCHEME = "iq2r"
IQ2R_FORMAT_NAME = "iq2r-512-fullsign-e8m0-native-v1"
IQ2R_FORMAT_VERSION = 4
IQ2R_PACKED_LAYOUT = "iq2r-cdna4-triplet6-v1"
IQ2R_ACTIVATION_BASIS = "native"
IQ2R_ARCHITECTURE = "gfx950"
IQ2R_ACTIVATION_DTYPE = "mxfp8-e4m3-e8m0"
IQ2R_CODEBOOK_DTYPE = "float8_e4m3fn"
IQ2R_SCALE_DTYPE = "e8m0"

IQ2R_CODEBOOK_ENTRIES = 512
IQ2R_VECTOR_SIZE = 8
IQ2R_SCALE_BLOCK = 32
IQ2R_TILE_N = 16
IQ2R_TILE_K = 128
IQ2R_LANE_RECORD_BYTES = 8
IQ2R_LANES_PER_TILE = 64
IQ2R_N_BLOCKS_PER_GROUP = 6
IQ2R_ATOMS_PER_TRIPLET = 3
IQ2R_TRIPLETS_PER_GROUP = IQ2R_N_BLOCKS_PER_GROUP // IQ2R_ATOMS_PER_TRIPLET
IQ2R_ATOM_PAIR_RECORDS_BYTES = IQ2R_LANES_PER_TILE * 2 * IQ2R_LANE_RECORD_BYTES
IQ2R_ATOM_TWO_RECORD_BYTES = 12
IQ2R_ATOM_TWO_RECORDS_OFFSET = IQ2R_ATOM_PAIR_RECORDS_BYTES
IQ2R_ATOM_TWO_METADATA_OFFSET = IQ2R_LANE_RECORD_BYTES
IQ2R_TRIPLET_BYTES = IQ2R_ATOM_PAIR_RECORDS_BYTES + (
    IQ2R_LANES_PER_TILE * IQ2R_ATOM_TWO_RECORD_BYTES
)
IQ2R_GROUP_BYTES = IQ2R_TRIPLETS_PER_GROUP * IQ2R_TRIPLET_BYTES
IQ2R_CODEBOOK_BYTES = IQ2R_CODEBOOK_ENTRIES * IQ2R_VECTOR_SIZE
IQ2R_BASE_PADDING_BYTES = 3
IQ2R_RESERVED_ZERO_CODEWORD = 0

IQ2R_GPT_OSS_EXPERTS = 128
IQ2R_GPT_OSS_K = 2880
IQ2R_GPT_OSS_GATE_UP_N = 5760
IQ2R_GPT_OSS_DOWN_N = 2880
IQ2R_GPT_OSS_TOP_K = 4


def _require_int(name: str, value: int, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def iq2r_k_tiles(k: int) -> int:
    """Number of stored K=128 tiles for a logical contraction width."""

    _require_int("k", k, positive=True)
    if k % IQ2R_SCALE_BLOCK:
        raise ValueError(f"IQ2R requires K divisible by {IQ2R_SCALE_BLOCK}, got {k}")
    return (k + IQ2R_TILE_K - 1) // IQ2R_TILE_K


def iq2r_n_blocks(n: int) -> int:
    """Number of logical N=16 blocks."""

    _require_int("n", n, positive=True)
    if n % IQ2R_TILE_N:
        raise ValueError(f"IQ2R requires N divisible by {IQ2R_TILE_N}, got {n}")
    return n // IQ2R_TILE_N


def iq2r_physical_n_blocks(n: int) -> int:
    """N-block count rounded to the six-block physical storage group."""

    logical = iq2r_n_blocks(n)
    return (
        (logical + IQ2R_N_BLOCKS_PER_GROUP - 1) // IQ2R_N_BLOCKS_PER_GROUP
    ) * IQ2R_N_BLOCKS_PER_GROUP


def iq2r_padded_k(k: int) -> int:
    return iq2r_k_tiles(k) * IQ2R_TILE_K


def iq2r_data_bytes(n: int, k: int) -> int:
    physical_n_blocks = iq2r_physical_n_blocks(n)
    return (
        physical_n_blocks
        // IQ2R_N_BLOCKS_PER_GROUP
        * iq2r_k_tiles(k)
        * IQ2R_GROUP_BYTES
    )


def iq2r_aux_bytes(n: int) -> int:
    return IQ2R_CODEBOOK_BYTES + iq2r_physical_n_blocks(n) + IQ2R_BASE_PADDING_BYTES


def iq2r_packed_sizes(n: int, k: int) -> tuple[int, int]:
    """Return ``(data_bytes, auxiliary_bytes)`` for one expert matrix."""

    return iq2r_data_bytes(n, k), iq2r_aux_bytes(n)


def iq2r_storage_bits_per_weight(n: int, k: int) -> float:
    data_bytes, auxiliary_bytes = iq2r_packed_sizes(n, k)
    return 8.0 * (data_bytes + auxiliary_bytes) / (n * k)


@dataclass(frozen=True, slots=True)
class IQ2RMetadata:
    """Immutable description of one IQ2R expert projection."""

    logical_n: int
    logical_k: int
    scheme: str = IQ2R_SCHEME
    format_name: str = IQ2R_FORMAT_NAME
    format_version: int = IQ2R_FORMAT_VERSION
    packed_layout: str = IQ2R_PACKED_LAYOUT
    activation_basis: str = IQ2R_ACTIVATION_BASIS
    architecture: str = IQ2R_ARCHITECTURE
    activation_dtype: str = IQ2R_ACTIVATION_DTYPE
    codebook_dtype: str = IQ2R_CODEBOOK_DTYPE
    scale_dtype: str = IQ2R_SCALE_DTYPE
    codebook_entries: int = IQ2R_CODEBOOK_ENTRIES
    vector_size: int = IQ2R_VECTOR_SIZE
    scale_block: int = IQ2R_SCALE_BLOCK
    tile_n: int = IQ2R_TILE_N
    tile_k: int = IQ2R_TILE_K
    lane_record_bytes: int = IQ2R_LANE_RECORD_BYTES
    n_blocks_per_group: int = IQ2R_N_BLOCKS_PER_GROUP
    atoms_per_triplet: int = IQ2R_ATOMS_PER_TRIPLET
    triplet_bytes: int = IQ2R_TRIPLET_BYTES
    group_bytes: int = IQ2R_GROUP_BYTES
    codebook_bytes: int = IQ2R_CODEBOOK_BYTES
    base_padding_bytes: int = IQ2R_BASE_PADDING_BYTES
    reserved_zero_codeword: int = IQ2R_RESERVED_ZERO_CODEWORD
    source_model_fingerprint: str | None = None
    calibration_fingerprint: str | None = None
    calibration_scheme: str | None = None
    calibration_corpus_hash: str | None = None
    calibration_coverage: str | None = None
    calibration_imputation_policy: str | None = None

    def __post_init__(self) -> None:
        self.validate_layout()

    @property
    def n_blocks(self) -> int:
        return iq2r_n_blocks(self.logical_n)

    @property
    def physical_n_blocks(self) -> int:
        return iq2r_physical_n_blocks(self.logical_n)

    @property
    def k_tiles(self) -> int:
        return iq2r_k_tiles(self.logical_k)

    @property
    def padded_k(self) -> int:
        return iq2r_padded_k(self.logical_k)

    @property
    def data_bytes(self) -> int:
        return iq2r_data_bytes(self.logical_n, self.logical_k)

    @property
    def auxiliary_bytes(self) -> int:
        return iq2r_aux_bytes(self.logical_n)

    @property
    def storage_bits_per_weight(self) -> float:
        return iq2r_storage_bits_per_weight(self.logical_n, self.logical_k)

    def validate_layout(self) -> None:
        _require_int("logical_n", self.logical_n, positive=True)
        _require_int("logical_k", self.logical_k, positive=True)
        expected = {
            "scheme": IQ2R_SCHEME,
            "format_name": IQ2R_FORMAT_NAME,
            "format_version": IQ2R_FORMAT_VERSION,
            "packed_layout": IQ2R_PACKED_LAYOUT,
            "activation_basis": IQ2R_ACTIVATION_BASIS,
            "architecture": IQ2R_ARCHITECTURE,
            "activation_dtype": IQ2R_ACTIVATION_DTYPE,
            "codebook_dtype": IQ2R_CODEBOOK_DTYPE,
            "scale_dtype": IQ2R_SCALE_DTYPE,
            "codebook_entries": IQ2R_CODEBOOK_ENTRIES,
            "vector_size": IQ2R_VECTOR_SIZE,
            "scale_block": IQ2R_SCALE_BLOCK,
            "tile_n": IQ2R_TILE_N,
            "tile_k": IQ2R_TILE_K,
            "lane_record_bytes": IQ2R_LANE_RECORD_BYTES,
            "n_blocks_per_group": IQ2R_N_BLOCKS_PER_GROUP,
            "atoms_per_triplet": IQ2R_ATOMS_PER_TRIPLET,
            "triplet_bytes": IQ2R_TRIPLET_BYTES,
            "group_bytes": IQ2R_GROUP_BYTES,
            "codebook_bytes": IQ2R_CODEBOOK_BYTES,
            "base_padding_bytes": IQ2R_BASE_PADDING_BYTES,
            "reserved_zero_codeword": IQ2R_RESERVED_ZERO_CODEWORD,
        }
        for name, required in expected.items():
            actual = getattr(self, name)
            if actual != required:
                raise ValueError(
                    f"unsupported IQ2R {name}: {actual!r}; expected {required!r}"
                )
        # Also applies divisibility checks.
        iq2r_packed_sizes(self.logical_n, self.logical_k)

    def validate_gpt_oss(self) -> None:
        self.validate_layout()
        if self.logical_k != IQ2R_GPT_OSS_K or self.logical_n not in (
            IQ2R_GPT_OSS_GATE_UP_N,
            IQ2R_GPT_OSS_DOWN_N,
        ):
            raise ValueError(
                "IQ2R GPT-OSS supports matrices [5760,2880] and [2880,2880], "
                f"got [{self.logical_n},{self.logical_k}]"
            )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "padded_k": self.padded_k,
                "physical_n_blocks": self.physical_n_blocks,
                "data_bytes": self.data_bytes,
                "auxiliary_bytes": self.auxiliary_bytes,
            }
        )
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "IQ2RMetadata":
        serialized = dict(value)
        required = (
            "logical_n",
            "logical_k",
            "scheme",
            "format_name",
            "format_version",
            "packed_layout",
            "activation_basis",
            "architecture",
            "activation_dtype",
            "codebook_dtype",
            "scale_dtype",
            "reserved_zero_codeword",
        )
        missing = [name for name in required if name not in serialized]
        if missing:
            raise ValueError(
                "serialized IQ2R metadata is missing required fields: "
                + ", ".join(missing)
            )
        init_fields = {field.name for field in fields(cls) if field.init}
        return cls(
            **{name: item for name, item in serialized.items() if name in init_fields}
        )


def iq2r_gpt_oss_metadata(projection: str, **provenance: Any) -> IQ2RMetadata:
    """Construct strict metadata for a GPT-OSS gate/up or down projection."""

    if projection == "gate_up":
        n = IQ2R_GPT_OSS_GATE_UP_N
    elif projection == "down":
        n = IQ2R_GPT_OSS_DOWN_N
    else:
        raise ValueError(f"projection must be 'gate_up' or 'down', got {projection!r}")
    metadata = IQ2RMetadata(logical_n=n, logical_k=IQ2R_GPT_OSS_K, **provenance)
    metadata.validate_gpt_oss()
    return metadata


def _iq2r_record_view(group: Tensor, triplet: int, atom: int) -> Tensor:
    begin = triplet * IQ2R_TRIPLET_BYTES
    triplet_data = group[..., begin : begin + IQ2R_TRIPLET_BYTES]
    if atom < 2:
        paired = triplet_data[..., :IQ2R_ATOM_PAIR_RECORDS_BYTES].view(
            *triplet_data.shape[:-1], IQ2R_LANES_PER_TILE, 16
        )
        return paired[
            ..., atom * IQ2R_LANE_RECORD_BYTES : (atom + 1) * IQ2R_LANE_RECORD_BYTES
        ]
    third = triplet_data[..., IQ2R_ATOM_TWO_RECORDS_OFFSET:].view(
        *triplet_data.shape[:-1],
        IQ2R_LANES_PER_TILE,
        IQ2R_ATOM_TWO_RECORD_BYTES,
    )
    return third[..., :IQ2R_LANE_RECORD_BYTES]


def _iq2r_metadata_view(group: Tensor, triplet: int, atom: int) -> Tensor:
    begin = triplet * IQ2R_TRIPLET_BYTES
    triplet_data = group[..., begin : begin + IQ2R_TRIPLET_BYTES]
    third = triplet_data[..., IQ2R_ATOM_TWO_RECORDS_OFFSET:].view(
        *triplet_data.shape[:-1],
        IQ2R_LANES_PER_TILE,
        IQ2R_ATOM_TWO_RECORD_BYTES,
    )
    return third[..., IQ2R_ATOM_TWO_METADATA_OFFSET + atom]


def _validate_output_slice(metadata: IQ2RMetadata, start: int, length: int) -> None:
    _require_int("start", start)
    _require_int("length", length, positive=True)
    if (
        start < 0
        or start % IQ2R_TILE_N
        or length % IQ2R_TILE_N
        or start + length > metadata.logical_n
    ):
        raise ValueError(
            f"IQ2R output slice must be in range and aligned to {IQ2R_TILE_N} columns"
        )


def iq2r_slice_output_data(
    data: Tensor,
    metadata: IQ2RMetadata,
    start: int,
    length: int,
) -> Tensor:
    """Repack an exact contiguous output-column slice of stacked IQ2R data."""

    _validate_byte_matrix("data", data, metadata.data_bytes)
    _validate_output_slice(metadata, start, length)
    target = IQ2RMetadata(logical_n=length, logical_k=metadata.logical_k)
    experts = data.shape[0]
    source_groups = metadata.physical_n_blocks // IQ2R_N_BLOCKS_PER_GROUP
    target_groups = target.physical_n_blocks // IQ2R_N_BLOCKS_PER_GROUP
    source = data.view(experts, source_groups, metadata.k_tiles, IQ2R_GROUP_BYTES)
    output = torch.zeros(
        (experts, target_groups, target.k_tiles, IQ2R_GROUP_BYTES),
        dtype=torch.uint8,
        device=data.device,
    )
    source_block_start = start // IQ2R_TILE_N
    for target_block in range(target.n_blocks):
        source_block = source_block_start + target_block
        source_group = source[:, source_block // IQ2R_N_BLOCKS_PER_GROUP]
        target_group = output[:, target_block // IQ2R_N_BLOCKS_PER_GROUP]
        source_triplet, source_atom = divmod(
            source_block % IQ2R_N_BLOCKS_PER_GROUP, IQ2R_ATOMS_PER_TRIPLET
        )
        target_triplet, target_atom = divmod(
            target_block % IQ2R_N_BLOCKS_PER_GROUP, IQ2R_ATOMS_PER_TRIPLET
        )
        _iq2r_record_view(target_group, target_triplet, target_atom).copy_(
            _iq2r_record_view(source_group, source_triplet, source_atom)
        )
        _iq2r_metadata_view(target_group, target_triplet, target_atom).copy_(
            _iq2r_metadata_view(source_group, source_triplet, source_atom)
        )
    return output.reshape(experts, target.data_bytes)


def iq2r_slice_output_auxiliary(
    auxiliary: Tensor,
    metadata: IQ2RMetadata,
    start: int,
    length: int,
) -> Tensor:
    """Slice codebooks/base exponents for an IQ2R output-column shard."""

    _validate_byte_matrix("auxiliary", auxiliary, metadata.auxiliary_bytes)
    _validate_output_slice(metadata, start, length)
    target = IQ2RMetadata(logical_n=length, logical_k=metadata.logical_k)
    output = torch.full(
        (auxiliary.shape[0], target.auxiliary_bytes),
        127,
        dtype=torch.uint8,
        device=auxiliary.device,
    )
    output[:, :IQ2R_CODEBOOK_BYTES].copy_(auxiliary[:, :IQ2R_CODEBOOK_BYTES])
    source_block_start = start // IQ2R_TILE_N
    output[:, IQ2R_CODEBOOK_BYTES : IQ2R_CODEBOOK_BYTES + target.n_blocks].copy_(
        auxiliary[
            :,
            IQ2R_CODEBOOK_BYTES
            + source_block_start : IQ2R_CODEBOOK_BYTES
            + source_block_start
            + target.n_blocks,
        ]
    )
    return output


def iq2r_slice_input_data(
    data: Tensor,
    metadata: IQ2RMetadata,
    start: int,
    length: int,
) -> Tensor:
    """Extract an exact contiguous K-tile slice of stacked IQ2R data."""

    _validate_byte_matrix("data", data, metadata.data_bytes)
    _require_int("start", start)
    _require_int("length", length, positive=True)
    if (
        start < 0
        or start % IQ2R_TILE_K
        or length % IQ2R_TILE_K
        or start + length > metadata.logical_k
    ):
        raise ValueError(
            f"IQ2R input slice must be in range and aligned to {IQ2R_TILE_K} columns"
        )
    target = IQ2RMetadata(logical_n=metadata.logical_n, logical_k=length)
    experts = data.shape[0]
    groups = metadata.physical_n_blocks // IQ2R_N_BLOCKS_PER_GROUP
    source = data.view(experts, groups, metadata.k_tiles, IQ2R_GROUP_BYTES)
    tile_start = start // IQ2R_TILE_K
    # ``contiguous()`` may return the original narrow view when its strides are
    # already dense, retaining a non-zero storage offset.  IQ2R consumers treat
    # byte zero as the first packed tile, so force a fresh zero-offset storage.
    return (
        source[:, :, tile_start : tile_start + target.k_tiles]
        .clone(memory_format=torch.contiguous_format)
        .view(experts, target.data_bytes)
    )


def _validate_byte_matrix(name: str, value: Tensor, expected_bytes: int) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.uint8:
        raise TypeError(f"{name} must have dtype uint8, got {value.dtype}")
    if value.ndim != 2:
        raise ValueError(
            f"{name} must have shape [experts,{expected_bytes}], got {tuple(value.shape)}"
        )
    if value.shape[1] != expected_bytes:
        raise ValueError(
            f"{name} has {value.shape[1]} bytes per expert, expected {expected_bytes}"
        )
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def iq2r_validate_expert_weights(
    data: Tensor,
    auxiliary: Tensor,
    metadata: IQ2RMetadata,
    *,
    expert_count: int | None = None,
    verify_reserved_zero: bool = True,
) -> None:
    """Validate stacked public buffers ``data`` and ``auxiliary``."""

    metadata.validate_layout()
    _validate_byte_matrix("data", data, metadata.data_bytes)
    _validate_byte_matrix("auxiliary", auxiliary, metadata.auxiliary_bytes)
    if data.shape[0] != auxiliary.shape[0]:
        raise ValueError(
            "data and auxiliary expert counts differ: "
            f"{data.shape[0]} != {auxiliary.shape[0]}"
        )
    if data.device != auxiliary.device:
        raise ValueError("data and auxiliary must be on the same device")
    if expert_count is not None:
        _require_int("expert_count", expert_count, positive=True)
        if data.shape[0] != expert_count:
            raise ValueError(
                f"IQ2R buffers have {data.shape[0]} experts, expected {expert_count}"
            )
    if verify_reserved_zero:
        zero_vectors = auxiliary[:, :IQ2R_VECTOR_SIZE]
        if torch.count_nonzero(zero_vectors).item() != 0:
            raise ValueError("IQ2R codebook entry zero must be the all-zero vector")


__all__ = [name for name in globals() if name.startswith("IQ2R_")] + [
    "IQ2RMetadata",
    "iq2r_aux_bytes",
    "iq2r_data_bytes",
    "iq2r_gpt_oss_metadata",
    "iq2r_k_tiles",
    "iq2r_n_blocks",
    "iq2r_packed_sizes",
    "iq2r_padded_k",
    "iq2r_physical_n_blocks",
    "iq2r_slice_input_data",
    "iq2r_slice_output_auxiliary",
    "iq2r_slice_output_data",
    "iq2r_storage_bits_per_weight",
    "iq2r_validate_expert_weights",
]
