# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from aiter.ops.flydsl.mxfp4_kname import (
    _parse_mxfp4_g1_kname,
    parse_flydsl_v2_gemm2_kernel,
)

DEFAULT_SCREEN_TOPK = 2


class AccumulationMode(str, Enum):
    ATOMIC = "atomic"
    NON_ATOMIC = "non-atomic"


@dataclass(frozen=True)
class GEMM1ScreeningKey:
    block_m: int


@dataclass(frozen=True)
class PipelineCompatibilityKey:
    block_m: int
    accumulation_mode: AccumulationMode


@dataclass(frozen=True)
class GEMM2ScreeningKey:
    block_m: int
    accumulation_mode: AccumulationMode


@dataclass(frozen=True)
class GEMM1CandidateIdentity:
    kernel_name: str


@dataclass(frozen=True)
class GEMM2ExecutionIdentity:
    kernel_name: str


@dataclass(frozen=True)
class PipelinePairKey:
    gemm1: GEMM1CandidateIdentity
    gemm2: GEMM2ExecutionIdentity


@dataclass(frozen=True)
class GEMM1StageCandidate:
    identity: GEMM1CandidateIdentity
    original_order: int


@dataclass(frozen=True)
class GEMM2StageCandidate:
    identity: GEMM2ExecutionIdentity
    original_order: int


@dataclass(frozen=True)
class PlannedPipelinePair:
    key: PipelinePairKey
    candidate_row: Mapping[str, Any]
    original_order: int


@dataclass(frozen=True)
class GEMM1ScreeningGroup:
    key: GEMM1ScreeningKey
    candidates: tuple[GEMM1StageCandidate, ...]


@dataclass(frozen=True)
class GEMM2ScreeningGroup:
    key: GEMM2ScreeningKey
    candidates: tuple[GEMM2StageCandidate, ...]


@dataclass(frozen=True)
class PipelineCompatibilityGroup:
    key: PipelineCompatibilityKey
    gemm1_screening_key: GEMM1ScreeningKey
    gemm2_screening_key: GEMM2ScreeningKey
    gemm1_candidates: tuple[GEMM1StageCandidate, ...]
    gemm2_candidates: tuple[GEMM2StageCandidate, ...]
    pairs: tuple[PlannedPipelinePair, ...]


@dataclass(frozen=True)
class StagedSearchPlan:
    pairs: tuple[PlannedPipelinePair, ...]
    gemm1_groups: tuple[GEMM1ScreeningGroup, ...]
    gemm2_groups: tuple[GEMM2ScreeningGroup, ...]
    pipeline_groups: tuple[PipelineCompatibilityGroup, ...]


@dataclass(frozen=True)
class _PairContract:
    block_m: int
    accumulation_mode: AccumulationMode

    @property
    def gemm1_screening_key(self) -> GEMM1ScreeningKey:
        return GEMM1ScreeningKey(self.block_m)

    @property
    def pipeline_key(self) -> PipelineCompatibilityKey:
        return PipelineCompatibilityKey(self.block_m, self.accumulation_mode)

    @property
    def gemm2_screening_key(self) -> GEMM2ScreeningKey:
        return GEMM2ScreeningKey(self.block_m, self.accumulation_mode)


@dataclass(frozen=True)
class _SourcePair:
    row: Mapping[str, Any]
    key: PipelinePairKey
    contract: _PairContract
    original_order: int


def build_staged_candidate_plan(
    shape_row: Mapping[str, Any],
    pruned_candidate_rows: Iterable[Mapping[str, Any]],
) -> StagedSearchPlan:
    """Project the shared M_est-pruned pair enumeration without executing pairs."""
    source_pairs = [
        _source_pair(shape_row, candidate_row, order)
        for order, candidate_row in enumerate(pruned_candidate_rows)
    ]
    unique = {}
    for pair in source_pairs:
        unique.setdefault(pair.key, pair)
    pairs = tuple(
        PlannedPipelinePair(pair.key, dict(pair.row), pair.original_order)
        for pair in unique.values()
    )
    return _group_candidate_plan(
        pairs, {key: pair.contract for key, pair in unique.items()}
    )


def _source_pair(
    shape_row: Mapping[str, Any],
    candidate_row: Mapping[str, Any],
    order: int,
) -> _SourcePair:
    row = dict(candidate_row)
    gemm1 = GEMM1CandidateIdentity(str(row["kernelName1"]))
    gemm2 = GEMM2ExecutionIdentity(str(row["kernelName2"]))
    parsed_gemm1 = _parse_mxfp4_g1_kname(gemm1.kernel_name)
    parsed_gemm2 = parse_flydsl_v2_gemm2_kernel(gemm2.kernel_name)
    if parsed_gemm2 is None:
        raise ValueError("GEMM2 must be layout-v2 for staged MXFP4 search")
    block_m = int(row["block_m"])
    if parsed_gemm1["BM"] != block_m or parsed_gemm2["tile_m"] != block_m:
        raise ValueError(
            "pipeline pair block_m does not match its GEMM1 and GEMM2 identities"
        )
    _validate_gemm1_contract(shape_row, parsed_gemm1)
    _validate_gemm2_contract(gemm2.kernel_name, block_m)
    expected_inline_quant = block_m == 16
    if bool(parsed_gemm1["inline_quant"]) != expected_inline_quant:
        raise ValueError(
            f"GEMM1 BM{block_m} violates the derived input quantization contract"
        )
    accumulation_mode = (
        AccumulationMode.ATOMIC
        if parsed_gemm2["epilog"] == "atomic"
        else AccumulationMode.NON_ATOMIC
    )
    return _SourcePair(
        row=row,
        key=PipelinePairKey(gemm1, gemm2),
        contract=_PairContract(
            block_m=block_m,
            accumulation_mode=accumulation_mode,
        ),
        original_order=order,
    )


def _validate_gemm1_contract(
    shape_row: Mapping[str, Any], parsed_gemm1: Mapping[str, Any]
) -> None:
    if parsed_gemm1["a_dtype"] != "fp4":
        raise ValueError("GEMM1 a_dtype must be fp4 for staged MXFP4 search")
    if parsed_gemm1["out_dtype"] != "fp4":
        raise ValueError("GEMM1 out_dtype must be fp4 for staged MXFP4 search")
    expected_activation = _shape_activation(shape_row)
    if parsed_gemm1["act"] != expected_activation:
        raise ValueError(
            "GEMM1 activation does not match the shape activation: "
            f"{parsed_gemm1['act']!r} != {expected_activation!r}"
        )
    if parsed_gemm1["enable_bias"]:
        raise ValueError("GEMM1 bias is unsupported by staged MXFP4 search")


def _shape_activation(shape_row: Mapping[str, Any]) -> str:
    activation = str(shape_row.get("act_type", "ActivationType.Silu"))
    for name in ("situv2", "swiglu", "silu"):
        if activation.lower().endswith(name):
            return name
    raise ValueError(f"unsupported MXFP4 activation: {activation!r}")


def _validate_gemm2_contract(kernel_name: str, block_m: int) -> None:
    parsed_v2 = parse_flydsl_v2_gemm2_kernel(kernel_name)
    if parsed_v2 is None:
        raise ValueError("GEMM2 must be layout-v2 for staged MXFP4 search")
    for field, expected in (
        ("a_dtype", "fp4"),
        ("b_dtype", "fp4"),
        ("out_dtype", "bf16"),
    ):
        if parsed_v2[field] != expected:
            raise ValueError(
                f"GEMM2 {field} must be {expected} for staged MXFP4 search"
            )
    if parsed_v2["sort_block_m"] != block_m:
        raise ValueError("GEMM2 sort block_m must match the pipeline block_m")


def _group_candidate_plan(
    pairs: tuple[PlannedPipelinePair, ...],
    contracts_by_pair: Mapping[PipelinePairKey, _PairContract],
) -> StagedSearchPlan:
    pipeline_pairs: dict[PipelineCompatibilityKey, list[PlannedPipelinePair]] = {}
    gemm1_candidates: dict[GEMM1ScreeningKey, dict[GEMM1CandidateIdentity, int]] = {}
    gemm2_candidates: dict[GEMM2ScreeningKey, dict[GEMM2ExecutionIdentity, int]] = {}

    for pair in pairs:
        contract = contracts_by_pair[pair.key]
        pipeline_pairs.setdefault(contract.pipeline_key, []).append(pair)
        gemm1_candidates.setdefault(contract.gemm1_screening_key, {}).setdefault(
            pair.key.gemm1, pair.original_order
        )
        gemm2_candidates.setdefault(contract.gemm2_screening_key, {}).setdefault(
            pair.key.gemm2, pair.original_order
        )

    gemm1_groups = tuple(
        GEMM1ScreeningGroup(
            key=key,
            candidates=tuple(
                GEMM1StageCandidate(identity, original_order)
                for identity, original_order in candidates.items()
            ),
        )
        for key, candidates in gemm1_candidates.items()
    )
    gemm2_groups = tuple(
        GEMM2ScreeningGroup(
            key=key,
            candidates=tuple(
                GEMM2StageCandidate(identity, original_order)
                for identity, original_order in candidates.items()
            ),
        )
        for key, candidates in gemm2_candidates.items()
    )
    pipeline_groups = tuple(
        _pipeline_group(key, tuple(group_pairs))
        for key, group_pairs in pipeline_pairs.items()
    )
    return StagedSearchPlan(
        pairs=pairs,
        gemm1_groups=gemm1_groups,
        gemm2_groups=gemm2_groups,
        pipeline_groups=pipeline_groups,
    )


def _pipeline_group(
    key: PipelineCompatibilityKey,
    pairs: tuple[PlannedPipelinePair, ...],
) -> PipelineCompatibilityGroup:
    gemm1_by_identity: dict[GEMM1CandidateIdentity, int] = {}
    gemm2_by_identity: dict[GEMM2ExecutionIdentity, int] = {}
    for pair in pairs:
        gemm1_by_identity.setdefault(pair.key.gemm1, pair.original_order)
        gemm2_by_identity.setdefault(pair.key.gemm2, pair.original_order)
    return PipelineCompatibilityGroup(
        key=key,
        gemm1_screening_key=GEMM1ScreeningKey(key.block_m),
        gemm2_screening_key=GEMM2ScreeningKey(key.block_m, key.accumulation_mode),
        gemm1_candidates=tuple(
            GEMM1StageCandidate(identity, original_order)
            for identity, original_order in gemm1_by_identity.items()
        ),
        gemm2_candidates=tuple(
            GEMM2StageCandidate(identity, original_order)
            for identity, original_order in gemm2_by_identity.items()
        ),
        pairs=pairs,
    )
