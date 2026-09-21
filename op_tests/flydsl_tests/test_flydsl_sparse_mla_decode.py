# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Focused correctness and graph-replay coverage for FlyDSL sparse MLA decode."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import pytest
import torch

H = 16
DV = 512
DT = 64
DIM = DV + DT
BLOCK_I = 64
LOG2E = math.log2(math.e)


@dataclass(frozen=True)
class Case:
    name: str
    seq: int
    width: int
    valid: int
    context: int
    shift: int
    seed: int

    @property
    def ng(self) -> int:
        """Split count the workspace must be sized for; ask the library."""
        from aiter.ops.flydsl import sparse_mla_decode_workspace_shape

        return sparse_mla_decode_workspace_shape(self.seq, self.width)[0][1]


CASES = (
    Case("seq1", seq=1, width=2048, valid=2048, context=62000, shift=0, seed=101),
    Case("primary", seq=6, width=2048, valid=2048, context=62000, shift=342, seed=102),
    Case("seq30", seq=30, width=2048, valid=2048, context=62000, shift=137, seed=108),
    Case("seq96", seq=96, width=2048, valid=2048, context=62000, shift=73, seed=109),
    Case("ctx64", seq=6, width=2048, valid=64, context=64, shift=0, seed=103),
    Case("ctx2000", seq=6, width=2048, valid=2000, context=2000, shift=88, seed=104),
    Case("ctx1", seq=6, width=2048, valid=1, context=1, shift=0, seed=105),
    Case("degen_k1", seq=6, width=64, valid=1, context=64, shift=0, seed=106),
    Case("k2049", seq=6, width=2112, valid=2049, context=62208, shift=341, seed=107),
)

GRAPH_CASES = ("seq1", "k2049")

if not torch.cuda.is_available():
    pytest.skip(
        "FlyDSL sparse MLA decode requires ROCm device access", allow_module_level=True
    )

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import flydsl_sparse_mla_decode
from aiter.ops.flydsl.mla_reduce_kernels import _use_fine_decode_combine
from aiter.ops.flydsl.sparse_mla_decode_kernels import (
    _pick_inner_iter,
    _use_split_major,
)

pytestmark = pytest.mark.skipif(
    get_gfx() != "gfx950",
    reason="FlyDSL sparse MLA decode requires gfx950 with FlyDSL installed",
)


@pytest.mark.parametrize(
    ("seq", "ng", "expected"),
    ((11, 32, 1), (12, 32, 2), (47, 32, 2), (48, 32, 4), (96, 32, 4), (96, 33, 1)),
)
def test_sparse_mla_decode_inner_iter_policy(seq: int, ng: int, expected: int):
    assert _pick_inner_iter(seq, ng) == expected


@pytest.mark.parametrize(
    ("seq", "groups", "expected"),
    ((48, 8, False), (64, 8, True), (96, 8, True)),
)
def test_sparse_mla_decode_split_major_policy(seq: int, groups: int, expected: bool):
    assert _use_split_major(seq, groups, num_cu=256) is expected


def test_sparse_mla_decode_reducer_policy():
    """Fine mode may switch off once as `seq` grows, and never back on.

    The coarse reducer's grid grows monotonically with `seq`, so the choice
    between it and the Dv-sliced one has exactly one crossing. Asserting the
    shape rather than a table of values is what catches the failure mode this
    replaced: a policy expressed as a *band* left both the short sequences and
    the long ones on the under-occupied path, and a table of sampled points
    inside the band could not see it.
    """
    num_cu = 256
    picks = [_use_fine_decode_combine(s, num_cu) for s in range(1, 129)]
    assert picks[0] is True, "a 1-row decode must not run the coarse reducer"
    assert picks[-1] is False, "fine must give way once coarse saturates"
    flips = sum(a is not b for a, b in itertools.pairwise(picks))
    assert flips == 1, f"policy is not monotone in seq: {flips} transitions"


def _snr_db(actual: torch.Tensor, ref: torch.Tensor) -> float:
    actual64 = actual.to(torch.float64)
    ref64 = ref.to(torch.float64)
    noise = torch.linalg.vector_norm((actual64 - ref64).reshape(-1))
    signal = torch.linalg.vector_norm(ref64.reshape(-1))
    if noise.item() == 0.0:
        return float("inf")
    if signal.item() == 0.0:
        return float("-inf")
    return 20.0 * math.log10((signal / noise).item())


def _build_case(case: Case):
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(case.seed)
    q = (torch.randn((case.seq, H, DIM), device=device, generator=g) * 0.25).clamp_(
        -2.0, 2.0
    )
    kv = (torch.randn((case.context, DIM), device=device, generator=g) * 0.25).clamp_(
        -2.0, 2.0
    )
    q_fp8 = q.to(torch.float8_e4m3fn).contiguous()
    kv_fp8 = kv.to(torch.float8_e4m3fn).contiguous()
    indices = torch.full((case.seq, case.width), -1, dtype=torch.int32, device=device)
    base = torch.arange(case.valid, dtype=torch.int32, device=device)
    for row in range(case.seq):
        if case.context > 1 and case.valid > 1:
            indices[row, : case.valid] = (base + row * case.shift) % case.context
        else:
            indices[row, 0] = 0
    return q_fp8, kv_fp8, indices.contiguous()


def _reference_partials(
    q_fp8: torch.Tensor,
    kv_fp8: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq, width = map(int, indices.shape)
    # A split owns tiles g, g + ng, g + 2 * ng, ... -- strided, not the adjacent
    # run that width // ng would imply.
    from aiter.ops.flydsl import sparse_mla_decode_workspace_shape

    ng = sparse_mla_decode_workspace_shape(seq, width)[0][1]
    inner = width // (ng * BLOCK_I)
    group = inner * BLOCK_I
    q = q_fp8.to(torch.float32)
    kv = kv_fp8.to(torch.float32)
    safe_indices = indices.clamp_min(0).to(torch.int64)
    gathered = kv.index_select(0, safe_indices.reshape(-1)).reshape(seq, width, DIM)
    valid = indices >= 0
    scores = torch.einsum("shd,skd->shk", q, gathered) * float(sm_scale)
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))

    split_scores = (
        scores.view(seq, H, inner, ng, BLOCK_I)
        .permute(0, 3, 1, 2, 4)
        .reshape(seq, ng, H, group)
    )
    values = (
        gathered[:, :, :DV]
        .view(seq, inner, ng, BLOCK_I, DV)
        .permute(0, 2, 1, 3, 4)
        .reshape(seq, ng, group, DV)
    )
    valid_split = (
        valid.view(seq, inner, ng, BLOCK_I).permute(0, 2, 1, 3).reshape(seq, ng, group)
    )
    split_max = split_scores.max(dim=-1).values
    live = valid_split.any(dim=-1).unsqueeze(-1)
    max_safe = torch.where(live, split_max, torch.zeros_like(split_max))
    weights = torch.exp(split_scores - max_safe.unsqueeze(-1)) * valid_split.unsqueeze(
        2
    )
    denom = weights.sum(dim=-1)
    inv = torch.where(denom > 0, denom.reciprocal(), torch.zeros_like(denom))
    partial = torch.einsum("sghk,sgkd->sghd", weights * inv.unsqueeze(-1), values)
    lse_nat = torch.where(
        denom > 0,
        torch.log(denom) + max_safe,
        torch.zeros_like(denom),
    )
    partial_lse = torch.where(
        denom > 0,
        lse_nat * LOG2E,
        torch.full_like(denom, -(2**30)),
    )

    scores_by_head = scores
    full_max = scores_by_head.max(dim=-1).values
    full_live = valid.any(dim=-1).unsqueeze(-1)
    full_max_safe = torch.where(full_live, full_max, torch.zeros_like(full_max))
    full_weights = torch.exp(
        scores_by_head - full_max_safe.unsqueeze(-1)
    ) * valid.unsqueeze(1)
    full_denom = full_weights.sum(dim=-1)
    final = torch.einsum("shk,skd->shd", full_weights, gathered[:, :, :DV])
    final = torch.where(
        full_denom.unsqueeze(-1) > 0,
        final / full_denom.unsqueeze(-1),
        torch.zeros_like(final),
    )
    return partial.to(torch.bfloat16), partial_lse, final.to(torch.bfloat16)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_sparse_mla_decode_exact_contract(case: Case):
    q, kv, indices = _build_case(case)
    partial_output = torch.empty(
        (case.seq, case.ng, H, DV), device=q.device, dtype=torch.bfloat16
    )
    partial_lse = torch.empty(
        (case.seq, case.ng, H), device=q.device, dtype=torch.float32
    )
    out = torch.empty((case.seq, H, DV), device=q.device, dtype=torch.bfloat16)
    sm_scale = 1.0 / math.sqrt(DIM)

    ref_partial, ref_lse, ref_out = _reference_partials(q, kv, indices, sm_scale)

    flydsl_sparse_mla_decode(
        q,
        kv,
        indices,
        out,
        sm_scale,
        partial_output=partial_output,
        partial_lse=partial_lse,
    )
    partial_output.fill_(float("nan"))
    partial_lse.fill_(float("nan"))
    out_repeat = torch.empty_like(out)
    flydsl_sparse_mla_decode(
        q,
        kv,
        indices,
        out_repeat,
        sm_scale,
        partial_output=partial_output,
        partial_lse=partial_lse,
    )
    torch.cuda.synchronize()

    sentinel_mask = ref_lse == -(2**30)
    assert torch.equal(partial_lse == -(2**30), sentinel_mask)
    live_mask = ~sentinel_mask
    if live_mask.any():
        assert _snr_db(partial_lse[live_mask], ref_lse[live_mask]) >= 80.0
    assert _snr_db(partial_output, ref_partial) >= 30.0
    assert _snr_db(out, ref_out) >= 30.0
    assert torch.isfinite(out.to(torch.float32)).all()
    assert torch.equal(out, out_repeat)


def test_sparse_mla_decode_rejects_ambiguous_kv_rank3_shape():
    case = next(item for item in CASES if item.name == "degen_k1")
    q, kv, indices = _build_case(case)
    out = torch.empty((case.seq, H, DV), device=q.device, dtype=torch.bfloat16)
    bad_kv = kv[:, None, :].expand(-1, 2, -1).contiguous()
    with pytest.raises(ValueError, match=r"kv must have shape"):
        flydsl_sparse_mla_decode(q, bad_kv, indices, out, 1.0 / math.sqrt(DIM))


def test_sparse_mla_decode_rejects_an_empty_kv_pool():
    """A zero-row pool reaches the kernel as a null pointer and faults it."""
    case = next(item for item in CASES if item.name == "degen_k1")
    q, kv, indices = _build_case(case)
    out = torch.empty((case.seq, H, DV), device=q.device, dtype=torch.bfloat16)
    empty = kv[:0].contiguous()
    with pytest.raises(ValueError, match="at least one row"):
        flydsl_sparse_mla_decode(q, empty, indices, out, 1.0 / math.sqrt(DIM))


def test_sparse_mla_decode_accepts_singleton_kv_rank3_shape():
    case = next(item for item in CASES if item.name == "degen_k1")
    q, kv, indices = _build_case(case)
    out_rank2 = torch.empty((case.seq, H, DV), device=q.device, dtype=torch.bfloat16)
    out_rank3 = torch.empty_like(out_rank2)
    sm_scale = 1.0 / math.sqrt(DIM)

    flydsl_sparse_mla_decode(q, kv, indices, out_rank2, sm_scale)
    flydsl_sparse_mla_decode(q, kv[:, None, :], indices, out_rank3, sm_scale)
    torch.cuda.synchronize()

    assert torch.equal(out_rank3, out_rank2)


@pytest.mark.parametrize(
    "case_name",
    GRAPH_CASES,
)
def test_sparse_mla_decode_graph_replay(case_name: str):
    case = next(item for item in CASES if item.name == case_name)
    q_a, kv_a, indices_a = _build_case(case)
    q_b, kv_b, indices_b = _build_case(
        Case(
            case.name,
            seq=case.seq,
            width=case.width,
            valid=case.valid,
            context=case.context,
            shift=case.shift,
            seed=case.seed + 1000,
        )
    )
    sm_scale = 1.0 / math.sqrt(DIM)

    out_a = torch.empty((case.seq, H, DV), device="cuda", dtype=torch.bfloat16)
    po_a = torch.empty((case.seq, case.ng, H, DV), device="cuda", dtype=torch.bfloat16)
    pl_a = torch.empty((case.seq, case.ng, H), device="cuda", dtype=torch.float32)
    flydsl_sparse_mla_decode(
        q_a,
        kv_a,
        indices_a,
        out_a,
        sm_scale,
        partial_output=po_a,
        partial_lse=pl_a,
    )
    expected_a = out_a.clone()

    out_b = torch.empty((case.seq, H, DV), device="cuda", dtype=torch.bfloat16)
    po_b = torch.empty((case.seq, case.ng, H, DV), device="cuda", dtype=torch.bfloat16)
    pl_b = torch.empty((case.seq, case.ng, H), device="cuda", dtype=torch.float32)
    flydsl_sparse_mla_decode(
        q_b,
        kv_b,
        indices_b,
        out_b,
        sm_scale,
        partial_output=po_b,
        partial_lse=pl_b,
    )
    expected_b = out_b.clone()

    q_live = q_a.clone()
    kv_live = kv_a.clone()
    indices_live = indices_a.clone()
    out_live = torch.empty((case.seq, H, DV), device="cuda", dtype=torch.bfloat16)
    po_live = torch.empty(
        (case.seq, case.ng, H, DV), device="cuda", dtype=torch.bfloat16
    )
    pl_live = torch.empty((case.seq, case.ng, H), device="cuda", dtype=torch.float32)
    stream = torch.cuda.Stream()

    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        flydsl_sparse_mla_decode(
            q_live,
            kv_live,
            indices_live,
            out_live,
            sm_scale,
            partial_output=po_live,
            partial_lse=pl_live,
        )
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        flydsl_sparse_mla_decode(
            q_live,
            kv_live,
            indices_live,
            out_live,
            sm_scale,
            partial_output=po_live,
            partial_lse=pl_live,
        )

    expected_by_tag = {"a": expected_a, "b": expected_b}
    source_by_tag = {
        "a": (q_a, kv_a, indices_a),
        "b": (q_b, kv_b, indices_b),
    }
    for tag in ("a", "b", "a", "b"):
        q_src, kv_src, indices_src = source_by_tag[tag]
        q_live.copy_(q_src)
        kv_live.copy_(kv_src)
        indices_live.copy_(indices_src)
        out_live.zero_()
        po_live.fill_(float("nan"))
        pl_live.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out_live, expected_by_tag[tag])
        assert torch.isfinite(out_live.to(torch.float32)).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
