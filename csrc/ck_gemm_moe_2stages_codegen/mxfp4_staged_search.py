# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from aiter.ops.flydsl.mxfp4_kname import (
    _parse_mxfp4_g1_kname,
    mxfp4_intermediate_is_eligible,
    parse_flydsl_v2_gemm2_kernel,
    parse_g2_kname_any,
)

SearchMode = Literal["full", "staged"]
DEFAULT_SCREEN_TOPK = 2


@dataclass(frozen=True)
class Mxfp4SearchConfig:
    mode: SearchMode
    screen_topk: int


class AccumulationMode(str, Enum):
    ATOMIC = "atomic"
    NON_ATOMIC = "non-atomic"


@dataclass(frozen=True)
class GEMM1ScreeningKey:
    block_m: int


@dataclass(frozen=True)
class PipelineEquivalenceKey:
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
class PairKey:
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
    key: PairKey
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
    key: PipelineEquivalenceKey
    gemm1_screening_key: GEMM1ScreeningKey
    gemm2_screening_key: GEMM2ScreeningKey
    gemm1_candidates: tuple[GEMM1StageCandidate, ...]
    gemm2_candidates: tuple[GEMM2StageCandidate, ...]
    pairs: tuple[PlannedPipelinePair, ...]


@dataclass(frozen=True)
class StagedCandidatePlan:
    pairs: tuple[PlannedPipelinePair, ...]
    gemm1_groups: tuple[GEMM1ScreeningGroup, ...]
    gemm2_groups: tuple[GEMM2ScreeningGroup, ...]
    pipeline_groups: tuple[PipelineCompatibilityGroup, ...]


@dataclass(frozen=True)
class _PairContract:
    block_m: int
    accumulation_mode: AccumulationMode
    native_gemm2: bool

    @property
    def gemm1_screening_key(self) -> GEMM1ScreeningKey:
        return GEMM1ScreeningKey(self.block_m)

    @property
    def pipeline_key(self) -> PipelineEquivalenceKey:
        return PipelineEquivalenceKey(self.block_m, self.accumulation_mode)

    @property
    def gemm2_screening_key(self) -> GEMM2ScreeningKey:
        return GEMM2ScreeningKey(self.block_m, self.accumulation_mode)


@dataclass(frozen=True)
class _SourcePair:
    row: Mapping[str, Any]
    key: PairKey
    contract: _PairContract
    original_order: int


def resolve_search_config(
    *,
    search_mode: str,
    explicit_screen_topk: int | None,
    gfx: str,
) -> Mxfp4SearchConfig:
    """Validate MXFP4 search options before any shape is tuned."""
    if search_mode not in ("full", "staged"):
        raise ValueError(f"unsupported MXFP4 search mode: {search_mode!r}")
    if explicit_screen_topk is not None:
        if explicit_screen_topk <= 0:
            raise ValueError("--mxfp4-screen-topk must be a positive integer")
        if search_mode != "staged":
            raise ValueError("--mxfp4-screen-topk requires --mxfp4-search staged")
    if gfx != "gfx950":
        raise ValueError("--mxfp4-flydsl is only supported on gfx950")
    return Mxfp4SearchConfig(
        mode=search_mode,
        screen_topk=(
            DEFAULT_SCREEN_TOPK
            if explicit_screen_topk is None
            else explicit_screen_topk
        ),
    )


def build_staged_candidate_plan(
    shape_row: Mapping[str, Any],
    full_candidate_rows: Iterable[Mapping[str, Any]],
    *,
    mxfp4_intermediate: bool,
) -> StagedCandidatePlan:
    """Project staged candidates from the exhaustive pipeline-pair enumeration."""
    source_pairs = [
        _source_pair(shape_row, candidate_row, order)
        for order, candidate_row in enumerate(full_candidate_rows)
    ]
    native_dimensions_supported = (
        int(shape_row["model_dim"]) % 256 == 0
        and int(shape_row["inter_dim"]) % 256 == 0
    )
    filtered_pairs = [
        pair
        for pair in source_pairs
        if not pair.contract.native_gemm2 or native_dimensions_supported
    ]
    available_pairs: dict[PairKey, _SourcePair] = {}
    for pair in filtered_pairs:
        available_pairs.setdefault(pair.key, pair)
    canonical_pairs_by_key: dict[PairKey, _SourcePair] = {}
    for pair in filtered_pairs:
        effective_mxfp4_intermediate = (
            mxfp4_intermediate
            and mxfp4_intermediate_is_eligible(
                BM=pair.contract.block_m,
                D_HIDDEN=int(shape_row["model_dim"]),
                D_INTER=int(shape_row["inter_dim"]),
                NE=int(shape_row["expert"]),
            )
        )
        canonical_gemm2 = _canonical_gemm2_identity(
            pair.key.gemm2,
            pair.contract,
            effective_mxfp4_intermediate,
        )
        canonical_key = PairKey(pair.key.gemm1, canonical_gemm2)
        representative = available_pairs.get(canonical_key)
        if representative is None:
            raise ValueError(
                "canonical GEMM2 representative is absent from full search: "
                f"{canonical_gemm2.kernel_name!r}"
            )
        canonical_pairs_by_key.setdefault(canonical_key, representative)

    planned_pairs = tuple(
        PlannedPipelinePair(
            key=pair.key,
            candidate_row=dict(pair.row),
            original_order=pair.original_order,
        )
        for pair in sorted(
            canonical_pairs_by_key.values(), key=lambda pair: pair.original_order
        )
    )
    contracts_by_pair = {
        key: pair.contract for key, pair in canonical_pairs_by_key.items()
    }
    return _group_candidate_plan(planned_pairs, contracts_by_pair)


def _source_pair(
    shape_row: Mapping[str, Any],
    candidate_row: Mapping[str, Any],
    order: int,
) -> _SourcePair:
    row = dict(candidate_row)
    gemm1 = GEMM1CandidateIdentity(str(row["kernelName1"]))
    gemm2 = GEMM2ExecutionIdentity(str(row["kernelName2"]))
    parsed_gemm1 = _parse_mxfp4_g1_kname(gemm1.kernel_name)
    parsed_gemm2 = parse_g2_kname_any(gemm2.kernel_name)
    block_m = int(row["block_m"])
    if parsed_gemm1["BM"] != block_m or parsed_gemm2["BM"] != block_m:
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
        if parsed_gemm2["atomic"]
        else AccumulationMode.NON_ATOMIC
    )
    return _SourcePair(
        row=row,
        key=PairKey(gemm1, gemm2),
        contract=_PairContract(
            block_m=block_m,
            accumulation_mode=accumulation_mode,
            native_gemm2=not parsed_gemm2["v2"],
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
    if parsed_gemm1["interleave"]:
        raise ValueError("GEMM1 gate layout must be separated")
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
        return
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


def _canonical_gemm2_identity(
    identity: GEMM2ExecutionIdentity,
    contract: _PairContract,
    effective_mxfp4_intermediate: bool,
) -> GEMM2ExecutionIdentity:
    if (
        not contract.native_gemm2
        or contract.accumulation_mode is AccumulationMode.ATOMIC
    ):
        return identity
    parsed = parse_g2_kname_any(identity.kernel_name)
    if effective_mxfp4_intermediate:
        if parsed["mxfp4out"]:
            return identity
        if parsed["cshuffle"]:
            return GEMM2ExecutionIdentity(
                identity.kernel_name.replace("_cshuffle", "_f4out")
            )
        return GEMM2ExecutionIdentity(identity.kernel_name + "_f4out")
    if parsed["mxfp4out"]:
        return GEMM2ExecutionIdentity(identity.kernel_name.replace("_f4out", ""))
    return identity


def _group_candidate_plan(
    pairs: tuple[PlannedPipelinePair, ...],
    contracts_by_pair: Mapping[PairKey, _PairContract],
) -> StagedCandidatePlan:
    pipeline_pairs: dict[PipelineEquivalenceKey, list[PlannedPipelinePair]] = {}
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
    return StagedCandidatePlan(
        pairs=pairs,
        gemm1_groups=gemm1_groups,
        gemm2_groups=gemm2_groups,
        pipeline_groups=pipeline_groups,
    )


def _pipeline_group(
    key: PipelineEquivalenceKey,
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
