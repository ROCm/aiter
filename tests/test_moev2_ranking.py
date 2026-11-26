# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The default ranking table and its checks (aiter/moev2/dispatch/ranking.py).

The checks over the whole dispatch snapshot key grid run in a subprocess
(tests/moe_dispatch/ranking_checks.py); the rest are unit tests on the CPU.

NOTE: part of the aiter.moev2 migration (under active development, not graduated).
Please coordinate changes with the moev2 owners (@qli88). See aiter/moev2/README.md.

Golden snapshots: after an intended change to legacy MoE dispatch, run
`python tests/moe_dispatch/snapshot.py --update` and paste the changes it prints into
the PR.
"""

import pathlib
import subprocess
import sys

import pytest

from aiter import ActivationType, QuantType, dtypes
from aiter.moev2.dispatch import ranking
from aiter.moev2.dispatch.ops import Op
from aiter.moev2.dispatch.ranking import Entry, Region, match, when
from aiter.moev2.problem import MoeProblem
from aiter.moev2.segments.base import Segment

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

CK = ranking.CK_2STAGE
ASM = "sort | quant.a1 | asm.stage1 | quant.inter | ck.gemm2"
PER_TENSOR = match(quant_type=QuantType.per_Tensor)


@pytest.fixture(autouse=True)
def _gfx950():
    from moe_dispatch import arch_sim

    with arch_sim.simulated_arch("gfx950"):
        yield


def _segments():
    from aiter.moev2.migration import a8w8_pt

    return a8w8_pt.SEGMENTS


def _problem(token=64, quant=QuantType.per_Tensor, inter_dim=256):
    fp8 = dtypes.fp8
    args = (token, 7168, inter_dim, 256, 8, dtypes.bf16, fp8, fp8, quant, True)
    args += (ActivationType.Silu, False, 0, 0)
    return MoeProblem.from_legacy_key(*args, arch="gfx950", cu_num=256)


def test_the_shipped_table_passes_the_static_checks():
    assert ranking.check_table(ranking.RANKING, _segments()) == []


def test_the_shipped_table_passes_over_the_key_grid():
    proc = subprocess.run(
        [sys.executable, str(_HERE / "moe_dispatch" / "ranking_checks.py"), "--check"],
        cwd=_HERE.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-6000:] + proc.stderr[-3000:]


def test_per_tensor_fp8_without_a_tuned_row_ranks_ck_2stage():
    from aiter.moev2.dispatch.resolver import feasible_plans

    plans = ranking.rankable(feasible_plans(_problem(), _segments())[0])
    label, plan = ranking.best(_problem(), plans)
    assert label == "fp8_per_tensor[otherwise]"
    assert " | ".join(s.name for s in plan) == CK
    assert ranking.best(_problem(quant=QuantType.per_Token), plans) is None


def test_when_thresholds_are_integer_intervals():
    w = when(rows_per_expert__le=8, model_dim__gt=2048)
    assert str(w) == "rows_per_expert<=8, model_dim>2048"
    assert w.holds(_problem(token=256))  # 256 * 8 // 256 = 8
    assert not w.holds(_problem(token=512))
    assert when(m__lt=64).disjoint(when(m__ge=64))
    assert not when(m__le=64).disjoint(when(m__ge=64))
    assert when(topk__in={6, 8}).disjoint(when(topk__in={4}))
    assert when(topk__in={6, 8}).disjoint(when(topk__gt=8))
    assert when(m__gt=10, m__le=10).empty
    for bad in ({"m": 3}, {"m__eq": 3}, {"tokens__le": 3}, {"m__le": 2.5}):
        with pytest.raises((ValueError, TypeError)):
            when(**bad)
    with pytest.raises(ValueError):
        match(q_type=QuantType.per_Tensor)


def test_overlapping_entries_fail():
    table = (
        Entry("a", PER_TENSOR, otherwise=[CK]),
        Entry("b", match(w1_dtype=dtypes.fp8), otherwise=[CK]),
    )
    (error,) = ranking.check_table(table, _segments())
    assert "entries a and b overlap" in error
    disjoint = (
        Entry("a", PER_TENSOR, otherwise=[CK]),
        Entry("b", match(quant_type=QuantType.per_Token), otherwise=[CK]),
    )
    assert ranking.check_table(disjoint, _segments()) == []


def test_overlapping_and_dead_regions_fail():
    table = (
        Entry(
            "a",
            PER_TENSOR,
            regions=[
                Region(when(m__le=64), prefer=[ASM, CK]),
                Region(when(m__ge=64), prefer=[CK]),
                Region(when(m__gt=100, m__lt=50), prefer=[CK]),
            ],
            otherwise=[CK],
        ),
    )
    errors = ranking.check_table(table, _segments())
    assert any("regions [m<=64] and [m>=64] overlap" in e for e in errors)
    assert any("[m>100, m<50]: dead region" in e for e in errors)


def test_bad_structures_fail():
    table = (
        Entry(
            "a",
            PER_TENSOR,
            regions=[Region(when(m__le=8), prefer=["sort | ck.gemm1 | ck.gemm2"])],
            otherwise=["sort | quant.a1 | ck.gemm9 | quant.inter | ck.gemm2"],
        ),
    )
    errors = ranking.check_table(table, _segments())
    assert any("sort and ck.gemm1 are not adjacent" in e for e in errors)
    assert any("unknown segments ['ck.gemm9']" in e for e in errors)
    with pytest.raises(TypeError):
        Entry("a", PER_TENSOR, otherwise=CK)


def test_a_last_structure_with_shape_conditions_fails():
    class SmallMOnly(Segment):
        name = "small.gemm1"
        covers = (Op.GEMM1, Op.ACT)

        def reject_reason(self, p):
            return None if p.shape.m <= 64 else "M > 64"

    segments = (*_segments(), SmallMOnly())
    small = "sort | quant.a1 | small.gemm1 | quant.inter | ck.gemm2"
    ok = (Entry("a", PER_TENSOR, otherwise=[small, CK]),)
    assert ranking.check_table(ok, segments) == []
    bad = (Entry("a", PER_TENSOR, otherwise=[CK, small]),)
    (error,) = ranking.check_table(bad, segments)
    assert "not always feasible: small.gemm1" in error


def test_grid_checks_find_unranked_keys_and_dead_regions():
    problems = [("m64", _problem(token=64)), ("m4096", _problem(token=4096))]
    sets = [_segments()]
    errors, hits = ranking.check_problems(problems, sets)
    assert errors == [] and hits == {"fp8_per_tensor[otherwise]": 2}

    table = (
        Entry(
            "a",
            PER_TENSOR,
            regions=[
                Region(when(m__le=64), prefer=[CK]),
                Region(when(m__gt=100_000), prefer=[CK]),
            ],
        ),
    )
    errors, hits = ranking.check_problems(problems, sets, table)
    assert errors == ["m4096: no region of a and no otherwise"]
    assert ranking.dead_lists(hits, table) == [
        "a[m>100000]: dead, no supported problem lands here"
    ]
    asm_last = (Entry("a", PER_TENSOR, otherwise=[CK, ASM]),)
    errors, _ = ranking.check_problems(problems[:1], sets, asm_last)
    assert errors == []
    assert ranking.check_problems(problems, sets, ())[0] == [
        "m64: lands in 0 entries []",
        "m4096: lands in 0 entries []",
    ]
