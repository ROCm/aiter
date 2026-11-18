# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""QSA sparse prefill warmup: the query tile a layout gets, the layouts that cover every
kernel-variant class, and the synthetic selection they run on.

CPU only: the runtime architecture, the platform and the wide-tile thresholds are injected
(gfx950 at 256 CUs and gfx942 at 80 CUs), so no test reads the host's GPU, and no kernel
runs. Expected values are derived by hand.
"""

from types import SimpleNamespace

import pytest
from conftest import requires_built_aiter

requires_built_aiter()
pytest.importorskip("flydsl")

import torch

import aiter.ops.flydsl.qsa_sparse_prefill as op
from aiter.ops import qsa
from aiter.ops.flydsl.kernels.qsa import sparse_prefill_prepare as prepare

# runtime architecture, union_min_rows, union_min_rows_per_head, first rows of the BQ21 and
# BQ42 tiles (gfx950: BQ21 from 2048 rows and BQ42 from 4096 at 256 CUs; gfx942 at 80 CUs).
GFX950 = ("gfx950", 1536, 0, {21: 2048, 42: 4096})
GFX942 = ("gfx942", 384, 64, {21: 1680, 42: 3360})
# Wide tiles that start below the union gate, so the first call with a union plan already
# takes them: gfx950 at 32 CUs (a partition; 8/21 tiles per CU gives 21 x 32 x 8/21 = 256
# rows for BQ21 and 512 for BQ42), and a union gate of 3000 rows over the 256-CU tiles.
GFX950_32_CUS = ("gfx950", 1536, 0, {21: 256, 42: 512})
UNION_ABOVE_WIDE = ("gfx950", 3000, 0, {21: 2048, 42: 4096})
ARCHS = {
    "gfx950": GFX950,
    "gfx942": GFX942,
    "gfx950_32_cus": GFX950_32_CUS,
    "union_gate_above_the_wide_tiles": UNION_ABOVE_WIDE,
}


@pytest.fixture
def platform(monkeypatch):
    def inject(arch):
        runtime, union_min_rows, per_head, wide = arch
        config = SimpleNamespace(
            union_min_rows=union_min_rows, union_min_rows_per_head=per_head
        )
        monkeypatch.setattr(op, "get_gfx_runtime", lambda: runtime)
        monkeypatch.setattr(op, "platform", lambda: config)
        monkeypatch.setattr(prepare, "wide_tile_rows", lambda tile: wide[tile])

    inject(GFX950)
    return inject


@pytest.mark.parametrize(
    ("group", "hk", "query_lens", "prefix_lens", "asked", "tile"),
    [
        # BQ21 for a full prefill of group 6 from 2048 rows.
        pytest.param(6, 1, (2048,), (0,), 32, 21, id="g6_wide_at_the_threshold"),
        pytest.param(6, 1, (2047,), (0,), 32, 16, id="g6_below_the_threshold"),
        pytest.param(6, 1, (4096,), (1,), 32, 16, id="g6_prefix"),
        pytest.param(6, 1, (2048, 2048), (0, 0), 32, 16, id="g6_two_requests"),
        pytest.param(6, 2, (4096,), (0,), 32, 16, id="g6_two_kv_heads"),
        pytest.param(6, 1, (4096,), (0,), 16, 16, id="g6_narrow_tile_asked"),
        # BQ42 for group 3 from 4096 rows, only when the base tile 32 is asked.
        pytest.param(3, 1, (4096,), (0,), 32, 42, id="g3_wide_at_the_threshold"),
        pytest.param(3, 1, (4095,), (0,), 32, 32, id="g3_below_the_threshold"),
        pytest.param(3, 1, (8192,), (0,), 16, 16, id="g3_narrow_tile_asked"),
        # No wide tile: 128 // group, rounded down to a power of two except at group 12.
        pytest.param(12, 1, (8192,), (0,), 32, 10, id="g12"),
        pytest.param(16, 1, (8192,), (0,), 32, 8, id="g16"),
        pytest.param(8, 1, (8192,), (0,), 32, 16, id="g8"),
        pytest.param(4, 1, (8192,), (0,), 32, 32, id="g4"),
        pytest.param(1, 1, (8192,), (0,), 32, 32, id="g1"),
        # No request: nothing indexes the first one.
        pytest.param(6, 1, (), (), 32, 16, id="no_requests"),
    ],
)
def test_the_query_tile_of_a_layout(
    platform, group, hk, query_lens, prefix_lens, asked, tile
):
    assert prepare.plan_query_tile(group, hk, query_lens, prefix_lens, asked) == tile


@pytest.mark.parametrize(
    ("arch", "group", "hk", "layouts"),
    [
        pytest.param(GFX950, 12, 1, [(1535,), (768, 768)], id="gfx950_g12"),
        pytest.param(GFX950, 6, 1, [(1535,), (768, 768), (2048,)], id="gfx950_g6"),
        pytest.param(GFX950, 3, 1, [(1535,), (768, 768), (4096,)], id="gfx950_g3"),
        pytest.param(GFX950, 6, 2, [(1535,), (768, 768)], id="gfx950_g6_hk2"),
        # The wide tiles start below the union gate: the first call with a union plan
        # takes them, so they start from the gate's rows.
        pytest.param(
            GFX950_32_CUS, 6, 1, [(1535,), (768, 768), (1536,)], id="gfx950_32_cus_g6"
        ),
        pytest.param(
            GFX950_32_CUS, 3, 1, [(1535,), (768, 768), (1536,)], id="gfx950_32_cus_g3"
        ),
        pytest.param(
            UNION_ABOVE_WIDE,
            6,
            1,
            [(2999,), (1500, 1500), (3000,)],
            id="union_gate_above_the_wide_tile_g6",
        ),
        # gfx942: union from max(384, 64 x group) rows.
        pytest.param(GFX942, 12, 1, [(767,), (384, 384)], id="gfx942_g12"),
        pytest.param(GFX942, 6, 1, [(383,), (192, 192), (1680,)], id="gfx942_g6"),
        pytest.param(GFX942, 3, 1, [(383,), (192, 192), (3360,)], id="gfx942_g3"),
    ],
)
def test_the_warmup_layouts_of_each_architecture(platform, arch, group, hk, layouts):
    platform(arch)
    assert list(op._warmup_layouts(group, hk)) == layouts


def variant(group, hk, query_lens, prefix_lens):
    """What a call's kernels depend on: no union plan (None), or the plan's query tile."""
    if sum(query_lens) < op._union_min_rows(group):
        return None
    return prepare.plan_query_tile(group, hk, query_lens, prefix_lens, 32)


ROWS = [1, 255, 256, 383, 384, 511, 512, 767, 768, 1535, 1536, 1680, 2047, 2048]
ROWS += [2999, 3000, 3001, 3359, 3360, 4095, 4096, 8192, 20000]


@pytest.mark.parametrize("arch", ARCHS.values(), ids=ARCHS)
@pytest.mark.parametrize("hk", [1, 2])
@pytest.mark.parametrize("group", [1, 2, 3, 4, 6, 8, 12, 16])
def test_every_layout_has_a_kernel_variant_the_warmup_runs(platform, arch, group, hk):
    platform(arch)
    warmed = {
        variant(group, hk, lens, (0,) * len(lens))
        for lens in op._warmup_layouts(group, hk)
    }
    for rows in ROWS:
        for lens, prefix in (
            ((rows,), (0,)),
            ((rows,), (7,)),
            ((rows // 2, rows - rows // 2), (0, 0)),
            ((rows // 3, rows // 3, rows - 2 * (rows // 3)), (5, 0, 9)),
        ):
            assert variant(group, hk, lens, prefix) in warmed, (rows, lens, prefix)


def test_the_selection_of_a_long_request_keeps_its_last_blocks_and_its_tail():
    # Position 1029 sees 1030 tokens: 257 blocks of 4 and 2 tail tokens. At most 256
    # blocks are kept, the last ones: tokens 4 .. 1027, then the tail 1028, 1029.
    selection = op._prefill_selection((1030,), 256, "cpu")
    row = selection[1029].tolist()
    assert row[:1026] == list(range(4, 1030))
    assert row[1026:] == [-1]


@pytest.mark.parametrize("lens", [(1030,), (5, 1, 9), (1100, 4)])
def test_the_selection_has_the_layout_the_op_documents(lens):
    topk = 256
    selection = op._prefill_selection(lens, topk, "cpu")
    assert selection.shape == (sum(lens), 4 * topk + 3)
    row = 0
    for n in lens:
        for position in range(n):
            visible = position + 1
            values = selection[row].tolist()
            blocks = min(visible // 4, topk)
            body = values[: 4 * blocks]
            tail = values[4 * blocks :][: visible % 4]
            # Distinct complete blocks j < visible // 4, each as tokens 4j .. 4j + 3.
            starts = body[::4]
            assert len(set(starts)) == blocks
            assert all(s % 4 == 0 and s // 4 < visible // 4 for s in starts)
            assert all(
                body[i : i + 4] == list(range(body[i], body[i] + 4))
                for i in range(0, 4 * blocks, 4)
            )
            # Then the tail tokens 4 (visible // 4) .. visible - 1, then -1.
            assert tail == list(range(4 * (visible // 4), visible))
            assert values[4 * blocks + len(tail) :] == [-1] * (
                4 * topk + 3 - 4 * blocks - len(tail)
            )
            row += 1


def test_the_warmup_runs_every_layout_on_synthetic_operands(platform, monkeypatch):
    synced = []
    monkeypatch.setattr(torch.cuda, "synchronize", synced.append)
    calls = []

    def run(q, indices, query_lens, prefix_lens):
        calls.append(
            (
                q.shape,
                q.dtype,
                q.abs().sum().item(),
                indices.shape,
                indices.dtype,
                query_lens,
                prefix_lens,
            )
        )
        return query_lens != (768, 768)

    warmed = qsa.qsa_sparse_prefill_warmup(
        run, num_q_heads=6, num_kv_heads=1, selection_width=4 * 512 + 3, device="cpu"
    )
    assert [c[5] for c in calls] == [(1535,), (768, 768), (2048,)]
    assert [c[6] for c in calls] == [(0,), (0, 0), (0,)]
    assert {c[0] for c in calls} == {
        torch.Size([1535, 6, 256]),
        torch.Size([1536, 6, 256]),
        torch.Size([2048, 6, 256]),
    }
    assert all(c[1:3] == (torch.bfloat16, 0.0) for c in calls)
    assert all(c[3][1] == 4 * 512 + 3 and c[4] == torch.int32 for c in calls)
    # Only the layouts `run` took are reported.
    assert warmed == ((1535,), (2048,))
    # The compiles are done, not queued, when the warmup returns.
    assert synced == ["cpu"]


def test_the_warmup_takes_its_group_from_the_query_and_kv_heads(platform, monkeypatch):
    # gfx942: union from max(384, 64 x group) rows, so 12 query heads over 2 KV heads
    # (group 6) plan union from 384 rows, over 1 KV head (group 12) from 768.
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    platform(GFX942)
    ran = []

    def run(q, indices, query_lens, prefix_lens):
        ran.append(query_lens)
        return True

    for kv_heads, layouts in ((2, ((383,), (192, 192))), (1, ((767,), (384, 384)))):
        ran.clear()
        qsa.qsa_sparse_prefill_warmup(
            run,
            num_q_heads=12,
            num_kv_heads=kv_heads,
            selection_width=4 * 512 + 3,
            device="cpu",
        )
        assert tuple(ran[:2]) == layouts


@pytest.mark.parametrize(
    "width", [4 * 128 + 3, 4 * 300 + 3, 4 * 1280 + 3, 4 * 256 + 4, 2]
)
def test_a_selection_width_the_op_does_not_take_warms_nothing(platform, width):
    def run(*args):
        raise AssertionError("no call may be made")

    assert (
        qsa.qsa_sparse_prefill_warmup(
            run, num_q_heads=6, num_kv_heads=1, selection_width=width, device="cpu"
        )
        == ()
    )


def _no_call(*args):
    raise AssertionError("no call may be made")


def _warmup_nothing_is_run():
    return qsa.qsa_sparse_prefill_warmup(
        _no_call,
        num_q_heads=6,
        num_kv_heads=1,
        selection_width=4 * 512 + 3,
        device="cpu",
    )


def test_without_flydsl_the_warmup_runs_nothing(platform, monkeypatch):
    monkeypatch.setattr(qsa, "_HAS_FLYDSL_PREFILL", False)
    assert _warmup_nothing_is_run() == ()


def test_on_an_architecture_without_a_tile_policy_the_warmup_runs_nothing(
    platform, monkeypatch
):
    monkeypatch.setattr(op, "get_gfx_runtime", lambda: "gfx90a")
    assert _warmup_nothing_is_run() == ()
