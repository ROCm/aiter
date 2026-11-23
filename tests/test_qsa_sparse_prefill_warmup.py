# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""QSA sparse prefill warmup: the query tile a layout gets, the layouts that cover every
kernel-variant class, and the synthetic selection they run on.

CPU only: the runtime architecture, the CU count, the platform and the wide-tile thresholds
are injected (gfx950 at 256 CUs and gfx942 at 80 CUs), so no test reads the host's GPU, and
no kernel runs. Expected values are derived by hand. The coverage test runs the op's own
plan and launch-argument builders on CPU tensors and records what they compile instead of
compiling it.
"""

import contextlib
from types import SimpleNamespace

import pytest
from conftest import requires_built_aiter

requires_built_aiter()
pytest.importorskip("flydsl")

import flydsl.compiler as flyc
import numpy as np
import torch

import aiter.ops.flydsl.qsa_sparse_prefill as op
from aiter.ops import qsa
from aiter.ops.flydsl.kernels.qsa import sparse_prefill_common as common
from aiter.ops.flydsl.kernels.qsa import sparse_prefill_direct as direct
from aiter.ops.flydsl.kernels.qsa import sparse_prefill_direct_fp8 as direct_fp8
from aiter.ops.flydsl.kernels.qsa import sparse_prefill_prepare as prepare
from aiter.ops.flydsl.kernels.qsa import sparse_prefill_union as union
from aiter.ops.flydsl.kernels.qsa import sparse_prefill_union_fp8 as union_fp8

# runtime architecture, CUs, union_min_rows, union_min_rows_per_head, first rows of the
# BQ21 and BQ42 tiles (gfx950: BQ21 from 2048 rows and BQ42 from 4096 at 256 CUs; gfx942
# at 80 CUs).
GFX950 = ("gfx950", 256, 1536, 0, {21: 2048, 42: 4096})
GFX942 = ("gfx942", 80, 384, 64, {21: 1680, 42: 3360})
# Wide tiles that start below the union gate, so the first call with a union plan already
# takes them: gfx950 at 32 CUs (a partition; 8/21 tiles per CU gives 21 x 32 x 8/21 = 256
# rows for BQ21 and 512 for BQ42), and a union gate of 3000 rows over the 256-CU tiles.
GFX950_32_CUS = ("gfx950", 32, 1536, 0, {21: 256, 42: 512})
UNION_ABOVE_WIDE = ("gfx950", 256, 3000, 0, {21: 2048, 42: 4096})


@pytest.fixture
def platform(monkeypatch):
    def inject(arch):
        runtime, cus, union_min_rows, per_head, wide = arch
        config = SimpleNamespace(
            union_min_rows=union_min_rows, union_min_rows_per_head=per_head
        )
        monkeypatch.setattr(op, "get_gfx_runtime", lambda: runtime)
        monkeypatch.setattr(op, "platform", lambda fp8=None: config)
        monkeypatch.setattr(prepare, "get_cu_num", lambda: cus)
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


# Prefixes of the warmup layouts: none (direct only, or a wide tile's full prefill), or 4
# tokens (the base tile: a prefix keeps a request off the wide tiles in both batch modes).
NO_PREFIX, PREFIX = (0,), (4,)


@pytest.mark.parametrize(
    ("arch", "group", "hk", "layouts"),
    [
        pytest.param(
            GFX950, 12, 1, [((1535,), NO_PREFIX), ((1536,), PREFIX)], id="gfx950_g12"
        ),
        pytest.param(
            GFX950,
            6,
            1,
            [((1535,), NO_PREFIX), ((1536,), PREFIX), ((2048,), NO_PREFIX)],
            id="gfx950_g6",
        ),
        pytest.param(
            GFX950,
            3,
            1,
            [((1535,), NO_PREFIX), ((1536,), PREFIX), ((4096,), NO_PREFIX)],
            id="gfx950_g3",
        ),
        pytest.param(
            GFX950, 6, 2, [((1535,), NO_PREFIX), ((1536,), PREFIX)], id="gfx950_g6_hk2"
        ),
        # The wide tiles start below the union gate: the first call with a union plan
        # takes them, so they start from the gate's rows.
        pytest.param(
            GFX950_32_CUS,
            6,
            1,
            [((1535,), NO_PREFIX), ((1536,), PREFIX), ((1536,), NO_PREFIX)],
            id="gfx950_32_cus_g6",
        ),
        pytest.param(
            GFX950_32_CUS,
            3,
            1,
            [((1535,), NO_PREFIX), ((1536,), PREFIX), ((1536,), NO_PREFIX)],
            id="gfx950_32_cus_g3",
        ),
        pytest.param(
            UNION_ABOVE_WIDE,
            6,
            1,
            [((2999,), NO_PREFIX), ((3000,), PREFIX), ((3000,), NO_PREFIX)],
            id="union_gate_above_the_wide_tile_g6",
        ),
        # gfx942: union from max(384, 64 x group) rows; then FP8 K/V's whole-task rows of
        # the base tile at 80 CUs, tile x ceil(80 / (hk x slices)), slices = ceil(tile x
        # group / 128): BQ10 at group 12 (800), BQ32 at 6 and 3 (1280, 2560), BQ16 at 6
        # over 2 KV heads (640). A wide tile is whole at its first rows (21 x 80, 42 x 80).
        pytest.param(
            GFX942,
            12,
            1,
            [((767,), NO_PREFIX), ((768,), PREFIX), ((800,), PREFIX)],
            id="gfx942_g12",
        ),
        pytest.param(
            GFX942,
            6,
            1,
            [
                ((383,), NO_PREFIX),
                ((384,), PREFIX),
                ((1680,), NO_PREFIX),
                ((1280,), PREFIX),
            ],
            id="gfx942_g6",
        ),
        pytest.param(
            GFX942,
            3,
            1,
            [
                ((383,), NO_PREFIX),
                ((384,), PREFIX),
                ((3360,), NO_PREFIX),
                ((2560,), PREFIX),
            ],
            id="gfx942_g3",
        ),
        pytest.param(
            GFX942,
            6,
            2,
            [((383,), NO_PREFIX), ((384,), PREFIX), ((640,), PREFIX)],
            id="gfx942_g6_hk2",
        ),
    ],
)
def test_the_warmup_layouts_of_each_architecture(platform, arch, group, hk, layouts):
    platform(arch)
    assert list(op._warmup_layouts(group, hk)) == layouts


# Compile caches of the prefill modules: the coverage test records each call's keys there.
CACHES = (
    (prepare, "_COMPILED"),
    (union, "_COMPILED"),
    (union_fp8, "_COMPILED"),
    (direct, "_COMPILED"),
    (direct_fp8, "_PACK"),
    (direct_fp8, "_DIRECT"),
)


class _Compile:
    """flyc.compile, with or without hints: compiles nothing, its kernels do nothing."""

    def __getitem__(self, hints):
        return self

    def __call__(self, launch, *args):
        return lambda *args: None


def _upload(array, device):
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.int32))


@pytest.fixture
def cpu_launches(monkeypatch):
    """The op's plan and launch-argument builders on CPU tensors: compiles stood in at
    flyc.compile, no streams or pinned copies, an injected architecture and CU count."""
    monkeypatch.setattr(flyc, "compile", _Compile())
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=0)
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    for module in (prepare, direct, direct_fp8):
        monkeypatch.setattr(module, "upload", _upload)

    def inject(gfx, cus):
        for module in (op, common, direct, union):
            monkeypatch.setattr(module, "get_gfx_runtime", lambda: gfx)
        for module in (common, prepare):
            monkeypatch.setattr(module, "get_cu_num", lambda: cus)
        common._calibrated.cache_clear()
        common.platform.cache_clear()

    yield inject
    common._calibrated.cache_clear()
    common.platform.cache_clear()


def compiled_keys(monkeypatch, group, hk, layouts, fp8, invariant, pool, topk=512):
    """The compile keys of the kernels the layouts' first calls take, from empty caches:
    each call builds the op's workspace and its launches (_Workspace, _Hot). pool: None
    for packed K/V, else a slot pool holding the context from its fifth row."""
    caches = {}
    for module, name in CACHES:
        caches[module.__name__, name] = cache = {}
        monkeypatch.setattr(module, name, cache)
    for query_lens, prefix_lens in layouts:
        rows, tokens = sum(query_lens), sum(query_lens) + sum(prefix_lens)
        q = torch.empty(rows, group * hk, 256, dtype=torch.bfloat16)
        dtype = torch.bfloat16 if fp8 is None else torch.float8_e4m3fnuz
        kv_rows = tokens if pool is None else tokens + 4
        k = torch.empty(kv_rows, hk, 256, dtype=dtype)
        v = torch.empty(kv_rows, hk, 256, dtype=dtype)
        slots = None if pool is None else torch.arange(4, kv_rows, dtype=torch.int32)
        indices = torch.empty(rows, 4 * topk + 3, dtype=torch.int32)
        scale = (kv_rows, hk) if fp8 == "token" else (1,)
        k_scale, v_scale = (None, None) if fp8 is None else (torch.ones(scale),) * 2
        workspace = op._Workspace(
            q, k, v, indices, query_lens, prefix_lens, hk * 256, slots, invariant, fp8
        )
        out = torch.empty_like(q)
        op._Hot(workspace, q, k, v, indices, out, slots, 1.0, k_scale, v_scale)
    return {(where, key) for where, cache in caches.items() for key in cache}


ROWS = [1, 255, 383, 384, 767, 768, 800, 1280, 1535, 1536, 1680, 2047, 2048, 2560]
ROWS += [2999, 3000, 3359, 3360, 4095, 4096, 8192, 20000]
# One request, with and without a prefix; two and three requests; and a context past the
# direct kernels' pack budget (they read the raw K/V there).
SWEEP = [
    layout
    for rows in ROWS
    for layout in (
        ((rows,), (0,)),
        ((rows,), (7,)),
        ((rows // 2, rows - rows // 2), (0, 0)),
        ((rows // 3, rows // 3, rows - 2 * (rows // 3)), (5, 0, 9)),
    )
] + [((64,), (131072,))]


@pytest.mark.parametrize("pool", [None, "slots", "wide"])
@pytest.mark.parametrize("invariant", [False, True], ids=["default", "invariant"])
@pytest.mark.parametrize(("group", "hk"), [(12, 1), (6, 1), (3, 1), (16, 1), (6, 2)])
@pytest.mark.parametrize(
    ("gfx", "cus", "fp8"),
    [
        ("gfx942", 80, None),
        ("gfx942", 80, "tensor"),
        ("gfx942", 80, "token"),
        ("gfx950", 256, None),
        ("gfx950", 32, None),
    ],
)
def test_the_warmup_compiles_every_kernel_a_layout_takes(
    monkeypatch, cpu_launches, gfx, cus, fp8, group, hk, invariant, pool
):
    # The caller's run serves one batch mode, K/V dtype and pool: every warmup layout
    # runs in it. With _WIDE_SPAN at 0 every slot pool is wide (past the span raw direct
    # reaches with 32-bit offsets).
    cpu_launches(gfx, cus)
    if pool == "wide":
        monkeypatch.setattr(op, "_WIDE_SPAN", 0)
    layouts = op._warmup_layouts(group, hk)
    warm = compiled_keys(monkeypatch, group, hk, layouts, fp8, invariant, pool)
    swept = compiled_keys(monkeypatch, group, hk, SWEEP, fp8, invariant, pool)
    assert swept <= warm, sorted(map(str, swept - warm))
    # The sweep compiles through every kernel module of its K/V dtype.
    names = {"_COMPILED"} if fp8 is None else {"_COMPILED", "_PACK", "_DIRECT"}
    modules = (
        (prepare, union, direct) if fp8 is None else (prepare, union_fp8, direct_fp8)
    )
    assert {where for where, _ in swept} == {
        (module.__name__, name)
        for module, name in CACHES
        if module in modules and name in names
    }


def _order_kernels(monkeypatch, tiles):
    """The constexpr values of the K3 (sparse_prefill_order_masks) variants a group-1
    plan of BQ32 tiles over 128 KV heads (tiles x 128 tasks of one slice) compiles on a
    cold cache, and of the variant it launches."""
    compiled = []

    def compile_once(launch, args, grid, constants):
        if launch is prepare._order_masks_launch:
            compiled.append(constants)

    monkeypatch.setattr(prepare, "_compiled", compile_once)
    one = torch.zeros(1, dtype=torch.int32)
    inputs = SimpleNamespace(
        indices=one,
        query_positions=one,
        block_indices=one,
        q=one,
        k=torch.zeros(1, 128, 1),
        kv_slots=None,
        topk=512,
    )
    tasks = tiles * 128
    grid = min(tasks, 256)
    plan = SimpleNamespace(
        num_tiles=tiles,
        union_tiles=tiles,
        query_tile=32,
        group_padded=1,
        block_capacity=8224,
        max_blocks=8224,
        grid=grid,
        task_order=torch.zeros(-(-tasks // grid) * grid, dtype=torch.int32),
        metadata=one,
        blocks=one,
        membership=one,
        counts=one,
        costs=one,
        residual=one,
        active=one,
        score_masks=one,
        query_tiles=one,
        direct_flag=one,
        union_ratio=1.0,
        core_ratio=0.0,
        split=1,
        route_costs=(0.0, 0.0, 0.0),
        routing=True,
        route_model=(0.0, 0.0, 0.0, 0.0, False),
        slotted=False,
        direct_groups=0,
    )
    list(prepare._launches(inputs, plan, None))
    return compiled, compiled[-1]


def test_the_task_count_compiles_no_new_order_kernel(monkeypatch):
    # The warmup compiles K3 on small layouts only, so a K3 variant only a larger plan
    # takes would compile while serving. 16385 one-row requests over 32895-token
    # prefixes at 128 KV heads hold 2,097,280 tasks (block capacity 8224), more than one
    # order CTA sorts (4096): K3 runs its CHUNKED variant there, which one tile's first
    # compile covers too.
    compiled, _ = _order_kernels(monkeypatch, 1)
    _, launched = _order_kernels(monkeypatch, 16385)
    assert launched in compiled


@pytest.mark.parametrize(
    ("query_lens", "prefix_lens", "row"),
    [((1030,), (0,), 1029), ((1026,), (4,), 1025)],
    ids=["no_prefix", "prefix"],
)
def test_the_selection_of_a_long_request_keeps_its_last_blocks_and_its_tail(
    query_lens, prefix_lens, row
):
    # The row sees 1030 tokens: 257 blocks of 4 and 2 tail tokens. At most 256 blocks are
    # kept, the last ones: tokens 4 .. 1027, then the tail 1028, 1029.
    selection = op._prefill_selection(query_lens, prefix_lens, 256, "cpu")
    row = selection[row].tolist()
    assert row[:1026] == list(range(4, 1030))
    assert row[1026:] == [-1]


@pytest.mark.parametrize(
    ("lens", "prefixes"),
    [
        ((1030,), (0,)),
        ((5, 1, 9), (0, 0, 0)),
        ((1100, 4), (0, 0)),
        ((5, 1, 9), (4, 1030, 3)),
    ],
)
def test_the_selection_has_the_layout_the_op_documents(lens, prefixes):
    topk = 256
    selection = op._prefill_selection(lens, prefixes, topk, "cpu")
    assert selection.shape == (sum(lens), 4 * topk + 3)
    row = 0
    for n, prefix in zip(lens, prefixes):
        for position in range(n):
            visible = prefix + position + 1
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
                indices[0, :6].tolist(),
                query_lens,
                prefix_lens,
            )
        )
        return prefix_lens != PREFIX

    warmed = qsa.qsa_sparse_prefill_warmup(
        run, num_q_heads=6, num_kv_heads=1, selection_width=4 * 512 + 3, device="cpu"
    )
    assert [c[6:] for c in calls] == [
        ((1535,), NO_PREFIX),
        ((1536,), PREFIX),
        ((2048,), NO_PREFIX),
    ]
    # Row 0 sees its request's prefix and itself: 1 token (the tail token 0), or 5 (block
    # 0, then the tail token 4).
    assert [c[5] for c in calls] == [
        [0, -1, -1, -1, -1, -1],
        [0, 1, 2, 3, 4, -1],
        [0, -1, -1, -1, -1, -1],
    ]
    assert [c[0] for c in calls] == [
        torch.Size([1535, 6, 256]),
        torch.Size([1536, 6, 256]),
        torch.Size([2048, 6, 256]),
    ]
    assert all(c[1:3] == (torch.bfloat16, 0.0) for c in calls)
    assert all(c[3][1] == 4 * 512 + 3 and c[4] == torch.int32 for c in calls)
    # Only the layouts `run` took are reported.
    assert warmed == (((1535,), NO_PREFIX), ((2048,), NO_PREFIX))
    # The compiles are done, not queued, when the warmup returns.
    assert synced == ["cpu"]


def test_the_warmup_takes_its_group_from_the_query_and_kv_heads(platform, monkeypatch):
    # gfx942: union from max(384, 64 x group) rows, so 12 query heads over 2 KV heads
    # (group 6) plan union from 384 rows, over 1 KV head (group 12) from 768.
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    platform(GFX942)
    ran = []

    def run(q, indices, query_lens, prefix_lens):
        ran.append((query_lens, prefix_lens))
        return True

    for kv_heads, layouts in (
        (2, (((383,), NO_PREFIX), ((384,), PREFIX))),
        (1, (((767,), NO_PREFIX), ((768,), PREFIX))),
    ):
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
