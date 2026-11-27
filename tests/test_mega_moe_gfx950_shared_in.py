# SPDX-License-Identifier: MIT
"""AITER_MEGA_SHARED_IN (default off): the shared expert inside the MegaMoE small-input kernel.

Stub tests, CPU only, no kernel compiled: the provider's call order and dense fallbacks, MegaMoEV2's route
decision, the small-input runtime's launch arguments (0 rows included), which launch configs fit it, and a model
of the kernel's counters (the done epoch is raised exactly once on every path that has rows).
"""

import contextlib
import random
import sys
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")
pytest.importorskip("flydsl")

D, INTER = 5120, 2304


def _weight(n, k):
    w = torch.zeros(n, k, dtype=torch.float8_e4m3fn)
    w.is_shuffled = True
    return w


class _Shared:
    """A stand-in for ATOM's shared ``Expert``: two native group-32 MXFP8 linears and the held FFN input pair."""

    def __init__(self, calls, rows):
        self.calls = calls
        self.gate_up_proj = SimpleNamespace(
            native_a8_group_rows=32,
            bias=None,
            weight=_weight(64, 32),
            weight_scale=torch.zeros(2, 1, dtype=torch.uint8),
            gemm_seam="v41.moe.shared_gate_up",
        )
        self.w2 = SimpleNamespace(
            native_a8_group_rows=32,
            bias=None,
            weight=_weight(32, 32),
            weight_scale=torch.zeros(1, 1, dtype=torch.uint8),
            gemm_seam="v41.moe.shared_down",
        )
        self.swiglu_limit = 10.0
        self.use_fused_clamp_act_mul = False
        self._act_seam = self._act
        self.out = torch.ones(rows.shape[0], D, dtype=torch.bfloat16)
        self.seg_fuse_input = (
            rows,
            torch.zeros(rows.shape[0], D, dtype=torch.float8_e4m3fn),
            torch.zeros(rows.shape[0], D // 32, dtype=torch.uint8),
        )

    def _act(self, combined, weights):
        raise AssertionError("not called")

    def __call__(self, rows):
        self.calls.order.append("dense")
        self.seg_fuse_input = None  # the dense expert reads the pair
        return self.out


def _cap(module):
    return SimpleNamespace(flags=MappingProxyType({"layer_id": 3, "module": module}))


def _rows(n=6):
    return torch.zeros(n, D, dtype=torch.bfloat16)


def _unsupported(shared, module):
    yield "group rows", lambda: setattr(
        shared.gate_up_proj, "native_a8_group_rows", 128
    )
    yield "bias", lambda: setattr(shared.w2, "bias", torch.zeros(1))
    yield "padded", lambda: setattr(shared.gate_up_proj, "is_output_padded", True)
    yield "unshuffled", lambda: setattr(
        shared.w2, "weight", torch.zeros(32, 32, dtype=torch.float8_e4m3fn)
    )
    yield "bf16 weight", lambda: setattr(
        shared.gate_up_proj, "weight", torch.zeros(64, 32, dtype=torch.bfloat16)
    )
    yield "no clamp", lambda: setattr(shared, "swiglu_limit", 0.0)
    yield "another clamp", lambda: setattr(module.experts, "swiglu_limit", 7.0)
    yield "triton fusion", lambda: setattr(shared, "use_fused_clamp_act_mul", True)
    yield "act seam owner", lambda: setattr(shared, "_act_seam", lambda *a: None)
    yield "gemm seam owner", lambda: sys.modules["atom"].plugin_host._owned.add(
        "v41.moe.shared_down"
    )


# ---------------------------------------------------------------------------------------------------------
# MegaMoEV2.forward: the route decides
def _v2(monkeypatch, route, ready):
    from aiter.ops.flydsl.kernels.mega_moe import mega_moe_v2 as v2

    calls = []
    small = SimpleNamespace(
        shared_in_ready=lambda rows, shared: ready,
        prepared=[],
        forward=lambda x, w, ids, **kw: calls.append(("small", kw))
        or torch.full_like(x, 7.0),
    )
    small.shared_in_prepare = small.prepared.append
    moe = object.__new__(v2.MegaMoEV2)
    moe.mtpr, moe.model_dim, moe.topk, moe.dev = 8192, D, 6, torch.device("cpu")
    moe._bundle_plan = SimpleNamespace(small_m_mtprs=(256,))
    moe._small_m_runtimes = [small]
    moe._route = lambda rows, max_tokens, **kw: (
        route,
        256 if route == v2.ROUTE_SMALL_M else 8192,
    )
    moe._forward_route = lambda route, x, w, ids, **kw: calls.append(
        ("route", route)
    ) or torch.full_like(x, 2.0)
    monkeypatch.setattr(v2, "check_forward_inputs", lambda *a, **k: None)
    monkeypatch.setattr(v2, "torch_stream", lambda stream: None)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())
    return v2, moe, calls


def _input(calls, n=6):
    from aiter.ops.flydsl.kernels.mega_moe.weights import SharedExpertInput

    dense = torch.ones(n, D, dtype=torch.bfloat16)
    return (
        SharedExpertInput(
            torch.zeros(n, D, dtype=torch.uint8),
            torch.zeros(n, D // 32, dtype=torch.uint8),
            torch.zeros(2 * INTER, D, dtype=torch.uint8),
            torch.zeros(2 * INTER // 32, D // 32, dtype=torch.uint8),
            torch.zeros(D, INTER, dtype=torch.uint8),
            torch.zeros(D // 32, INTER // 32, dtype=torch.uint8),
            dense=lambda: calls.append(("dense",)) or dense,
        ),
        dense,
    )


def test_v2_small_route_hands_shared_in_to_the_kernel(monkeypatch):
    v2, moe, calls = _v2(monkeypatch, "small_m", ready=True)
    moe._route = lambda rows, max_tokens, **kw: (v2.ROUTE_SMALL_M, 256)
    shared, _ = _input(calls)
    out = moe.forward(
        _rows(),
        torch.zeros(6, 6),
        torch.zeros(6, 6, dtype=torch.int32),
        max_tokens=6,
        shared_in=shared,
    )
    assert [c[0] for c in calls] == ["small"] and calls[0][1]["shared_in"] is shared
    assert not shared.used_dense and torch.equal(
        out, torch.full((6, D), 7.0, dtype=torch.bfloat16)
    )


def test_v2_small_route_that_cannot_fuse_runs_dense_then_shared_add(monkeypatch):
    v2, moe, calls = _v2(monkeypatch, "small_m", ready=False)
    moe._route = lambda rows, max_tokens, **kw: (v2.ROUTE_SMALL_M, 256)
    shared, dense = _input(calls)
    moe.forward(
        _rows(),
        torch.zeros(6, 6),
        torch.zeros(6, 6, dtype=torch.int32),
        max_tokens=6,
        shared_in=shared,
    )
    assert [c[0] for c in calls] == ["dense", "small"]
    assert (
        calls[1][1]["shared_add"] is dense
        and "shared_in" not in calls[1][1]
        and shared.used_dense
    )


def test_v2_other_routes_run_dense_and_add(monkeypatch):
    v2, moe, calls = _v2(monkeypatch, "other", ready=True)
    moe._route = lambda rows, max_tokens, **kw: (v2.ROUTE_STAGED, 8192)
    shared, _ = _input(calls)
    out = moe.forward(
        _rows(),
        torch.zeros(6, 6),
        torch.zeros(6, 6, dtype=torch.int32),
        shared_in=shared,
    )
    assert [c[0] for c in calls] == ["dense", "route"]
    assert torch.equal(out, torch.full((6, D), 3.0, dtype=torch.bfloat16))
    # Whatever the route: the small-input runtime is handed the weights, so a later capture can fuse.
    assert moe._small_m_runtimes[0].prepared == [shared]
    with pytest.raises(ValueError):
        moe.forward(
            _rows(),
            torch.zeros(6, 6),
            torch.zeros(6, 6, dtype=torch.int32),
            shared_in=shared,
            shared_add=_rows(),
        )


# ---------------------------------------------------------------------------------------------------------
# SmallMRuntime: launch arguments, 0 rows, which configs fit
def _small(monkeypatch, mtpr=256, world=8, epr=48, topk=6):
    from aiter.ops.flydsl.kernels.mega_moe import mega_moe_small_m as sm
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_config import (
        FUSED_TOKEN_BUCKETS,
        select_fused_config,
    )

    rt = object.__new__(sm.SmallMRuntime)
    rt.mtpr, rt.model_dim, rt.inter_dim, rt.topk, rt.world_size, rt.epr = (
        mtpr,
        D,
        INTER,
        topk,
        world,
        epr,
    )
    rt.dev, rt.swiglu_limit, rt._s1_num_cu = torch.device("cpu"), 10.0, 256
    rt._configs = {
        b: select_fused_config(
            b,
            mtpr,
            world_size=world,
            experts_per_rank=epr,
            model_dim=D,
            inter_dim=INTER,
        )
        for b in FUSED_TOKEN_BUCKETS
        if b <= mtpr
    }
    rt._idle_inputs = (
        torch.zeros(1, D, dtype=torch.bfloat16),
        torch.zeros(1, 6),
        torch.zeros(1, 6, dtype=torch.int32),
    )
    rt.w2, rt.w2_scale = torch.zeros(8, dtype=torch.uint8), torch.zeros(
        8, dtype=torch.uint8
    )
    rt._s1_disp_template = [0] * sm.DISPATCH_TABLE_SIZE
    rt.w2_dispatch_table = lambda w2, w2_scale: torch.zeros(
        sm.DISPATCH_TABLE_SIZE, dtype=torch.int64
    )
    launches = []
    rt._launch = lambda config, tokens, x, ids, wts, stream, **kw: launches.append(
        (tokens, x, kw)
    )
    capturing = SimpleNamespace(now=False)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: capturing.now
    )
    monkeypatch.setattr(sm, "check_forward_inputs", lambda x, *a, **k: int(x.shape[0]))
    return sm, rt, launches, capturing


def test_small_runtime_launches_the_shared_in_kernel_with_the_pair_views_and_table(
    monkeypatch,
):
    sm, rt, launches, capturing = _small(monkeypatch)
    for rows, ksplit in ((48, 2), (64, 2), (96, 1), (126, 1)):
        shared, _ = _input([], rows)
        assert rt.shared_in_ready(rows, shared)
        x = _rows(rows)
        out = rt.forward(
            x,
            torch.zeros(rows, 6),
            torch.zeros(rows, 6, dtype=torch.int32),
            stream=object(),
            shared_in=shared,
        )
        tokens, x_arg, kw = launches[-1]
        split, x_view, scale_view, table = kw["shared_in"]
        assert (tokens, split) == (rows, ksplit) and x_arg is x
        assert (
            out.shape == (rows, D)
            and out.dtype == torch.bfloat16
            and kw["shared_add"] is out
        )
        assert (
            x_view.data_ptr() == shared.x_fp8.data_ptr()
            and scale_view.data_ptr() == shared.x_scale.data_ptr()
        )
        assert (
            tuple(x_view.shape) == tuple(scale_view.shape) == (16,)
            and x_view.dtype == torch.uint8
        )
        slot = sm.DispatchSlot
        assert (
            table[slot.SH_GU_W] == shared.gate_up.data_ptr()
            and table[slot.SH_DN_S] == shared.down_scale.data_ptr()
        )
        assert table[slot.SH_ACT] == rt._sh_workspace["act"].data_ptr()
        assert (
            table[slot.SH_SYNC] == rt._sh_workspace["sync"].data_ptr()
            and table[slot.G2_BQ] == rt.w2.data_ptr()
        )
        assert rt.shared_dispatch_table(shared) is table  # built once per weight set
        # Under a graph capture only what an eager call built before may be used.
        capturing.now = True
        other, _ = _input([], rows)
        assert rt.shared_in_ready(rows, shared) and not rt.shared_in_ready(rows, other)
        capturing.now = False
    assert (
        rt._sh_workspace["act"].numel() == 256 * INTER
        and rt._sh_workspace["act_scale"].numel() >= 8 * INTER
    )


def test_small_runtime_with_no_rows_launches_the_base_kernel(monkeypatch):
    _sm, rt, launches, _ = _small(monkeypatch)
    rt.comb_op = None
    shared, _ = _input([], 6)
    x = torch.zeros(0, D, dtype=torch.bfloat16)
    out = rt.forward(
        x,
        torch.zeros(0, 6),
        torch.zeros(0, 6, dtype=torch.int32),
        stream=object(),
        shared_in=shared,
    )
    tokens, x_arg, kw = launches[-1]
    assert (
        tokens == 0 and x_arg is rt._idle_inputs[0]
    )  # the idle stand-in rows, as without shared_in
    assert (
        kw["shared_in"] is None and kw["shared_add"] is None
    )  # the kernel without the shared expert
    assert out.shape == (0, D) and not shared.used_dense
    assert not rt.shared_in_supported(0)


def test_small_runtime_refuses_what_it_cannot_fuse(monkeypatch):
    _sm, rt, _, _ = _small(monkeypatch)
    shared, _ = _input([], 12)
    assert not rt.shared_in_ready(13, shared)  # another row count than the pair's
    with pytest.raises(ValueError):
        rt.forward(
            _rows(13),
            torch.zeros(13, 6),
            torch.zeros(13, 6, dtype=torch.int32),
            stream=object(),
            shared_in=shared,
        )
    with pytest.raises(ValueError):
        rt.forward(
            _rows(12),
            torch.zeros(12, 6),
            torch.zeros(12, 6, dtype=torch.int32),
            stream=object(),
            shared_in=shared,
            shared_add=_rows(12),
        )


def test_shared_in_fits_the_served_verify_buckets_and_not_a_narrow_grid():
    from dataclasses import replace

    from aiter.ops.flydsl.kernels.mega_moe import mega_moe_small_m as sm
    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_config import select_fused_config

    for bucket, rows in (
        (64, 60),
        (64, 95),
        (128, 96),
        (128, 126),
    ):  # the serving verify rows of the decode capacity
        config = select_fused_config(
            bucket, 256, world_size=8, experts_per_rank=48, model_dim=D, inter_dim=INTER
        )
        consumers = (
            256 * min(config.stage1.grid_mult, 2) - 1 - config.stage1.num_dispatch_cu
        )
        assert sm.shared_in_fits(config, rows, 256, D, INTER), (bucket, rows)
        assert not sm.shared_in_fits(config, 0, 256, D, INTER)
        # One consumer workgroup per unit: 20 GEMM2 units an m-tile.
        assert sm.shared_in_fits(config, 32 * (consumers // 20), 256, D, INTER)
        assert not sm.shared_in_fits(config, 32 * (consumers // 20) + 1, 256, D, INTER)
        for change in (
            {"fused_stage1": replace(config.fused_stage1, plan_local=False)},
            {"fused_stage1": replace(config.fused_stage1, tok_done=True)},
            {
                "stage1": replace(config.stage1, grid_mult=1)
            },  # 31 consumers at 224 producers
        ):
            assert not sm.shared_in_fits(
                replace(config, **change), rows, 256, D, INTER
            ), change


def test_dispatch_slots_and_sync_offsets_do_not_collide():
    from aiter.ops.flydsl.kernels.mega_moe import mega_moe_small_m as sm

    values = [int(s) for s in sm.DispatchSlot]
    assert len(set(values)) == len(values) and sm.DISPATCH_TABLE_SIZE == max(values) + 1
    assert (
        sm._SH_READY >= 16 * 4 and sm._SH_G2_COUNT >= sm._SH_READY + 16 * 4
    )  # 16 m-tiles of counts and epochs
    assert sm._SH_DONE == sm._SH_G2_COUNT + 4 and sm._SH_SYNC_WORDS * 4 > sm._SH_DONE


# ---------------------------------------------------------------------------------------------------------
# the kernel's counters, as a model: consumer c runs GEMM1 unit c, then GEMM2 unit c (compile_mega_moe_stage1)
class _SyncModel:
    N1, N2 = 2 * INTER // 256, D // 256

    def __init__(self):
        from aiter.ops.flydsl.kernels.mega_moe import mega_moe_small_m as sm

        self.sm = sm
        self.words = [0] * sm._SH_SYNC_WORDS
        self.done_raised = 0

    def launch(self, rows, consumers, epoch, rng):
        """One launch on one rank: every consumer's pre-phase steps in a random interleaving. Returns whether the
        exit wait (rows > 0) can pass."""
        sm, w = self.sm, self.words
        tiles = (rows + 31) // 32
        assert (
            tiles * max(self.N1, self.N2) <= consumers
        ), "shared_in_fits: a consumer per unit"
        pending = {c: ["g1", "g2"] for c in range(consumers)}
        while pending:
            c = rng.choice(list(pending))
            step = pending[c][0]
            if step == "g1":
                if c < tiles * self.N1:
                    m = c // self.N1
                    w[m] += 1
                    if w[m] == self.N1:  # the m-tile's last GEMM1 unit
                        w[sm._SH_READY // 4 + m] = epoch
                        w[m] -= self.N1
                pending[c].pop(0)
            else:
                if c < tiles * self.N2:
                    m = c // self.N2
                    if w[sm._SH_READY // 4 + m] != epoch:
                        continue  # still waiting for its m-tile's GEMM1 units; someone else runs
                    w[sm._SH_G2_COUNT // 4] += 1
                    if w[sm._SH_G2_COUNT // 4] == tiles * self.N2:
                        w[sm._SH_DONE // 4] = epoch
                        self.done_raised += 1
                        w[sm._SH_G2_COUNT // 4] -= tiles * self.N2
                pending[c].pop(0)
            if not pending[c]:
                del pending[c]
        return rows == 0 or w[sm._SH_DONE // 4] == epoch


def test_done_epoch_is_raised_once_on_every_path_and_the_counters_rewind():
    model, rng = _SyncModel(), random.Random(31)
    launches_with_rows = 0
    for epoch, rows in enumerate(
        [96, 1, 0, 256, 33, 64, 0, 0, 126, 32, 255, 6] * 3, start=1
    ):
        consumers = rng.choice(
            (287, 383, 160)
        )  # dcu224 / dcu128 grids, and the smallest that fits 256 rows
        assert model.launch(rows, consumers, epoch, rng), (epoch, rows)
        launches_with_rows += rows > 0
        assert (
            model.done_raised == launches_with_rows
        )  # exactly once per launch that has rows, never without
        counts = model.words[:16] + [model.words[model.sm._SH_G2_COUNT // 4]]
        assert counts == [0] * 17, (
            epoch,
            rows,
        )  # every count is back at zero for the next launch


def test_tile_scale_layout_is_what_the_staging_reads():
    pytest.importorskip("aiter.ops.flydsl.kernels.silu_mul_quant_epilogue")
    from aiter.ops.flydsl.kernels.mega_moe import shared_expert as se

    seen = set()
    for m in range(3):
        for n in range(INTER // 128):
            for row in range(32):
                first = se.tile_scale_byte(m, n, row, 0, INTER)
                # AScaleTileLoader: LDS dword row * 18 + n of the tile <- stored dword m * 576 + n * 32 + row.
                assert first == 4 * (m * 32 * (INTER // 128) + n * 32 + row)
                assert (
                    first // 64
                    == se.tile_scale_byte(m, n, 0, 0, INTER) // 64 + row // 16
                )  # 2 lines per (m, n)
                seen.update(se.tile_scale_byte(m, n, row, g, INTER) for g in range(4))
    assert seen == set(range(3 * INTER))
    assert [
        se.dense_gate_up_ksplit(r) for r in (1, 32, 33, 48, 64, 65, 96, 128, 256)
    ] == [1, 1, 2, 2, 2, 1, 1, 1, 1]


# ---------------------------------------------------------------------------------------------------------
# EP4: the served small-input OVERRIDE, the K split from the live table, eager preparation, draft layers
_EP4_SERVED_OVERRIDE = (
    "32+64+128:s1.use_tile_resource=1,fs1.epi_evec=8,fs1.epi_lean=3,fs1.head_uniform=1,fs1.g2_epi_lean=2,"
    "fs1.expert_cached_w=2;64:s1.tile_n=256;32:s1.tile_n=256,s2.block_n=256;"
    "16+32+64+128:fs1.disp_order=31,fs1.sync_lean=1"
)


@contextlib.contextmanager
def _override(value):
    """select_fused_config under AITER_MEGA_SMALL_M_OVERRIDE=value (read at import: swapped the way the bench does)."""
    from aiter.ops.flydsl.kernels.mega_moe import mega_moe_config as mc

    saved = mc._SMALL_M_OVERRIDE
    try:
        mc._SMALL_M_OVERRIDE = value
        mc._small_m_override.cache_clear()
        yield mc
    finally:
        mc._SMALL_M_OVERRIDE = saved
        mc._small_m_override.cache_clear()


@pytest.mark.parametrize("override", ["", _EP4_SERVED_OVERRIDE])
def test_shared_in_fits_the_ep4_verify_buckets(override):
    from aiter.ops.flydsl.kernels.mega_moe import mega_moe_small_m as sm

    with _override(override) as mc:
        fits = {}
        for bucket in (b for b in mc.FUSED_TOKEN_BUCKETS if b <= 256):
            config = mc.select_fused_config(
                bucket,
                256,
                world_size=4,
                experts_per_rank=96,
                model_dim=D,
                inter_dim=INTER,
            )
            rows = [
                t
                for t in range(1, 257)
                if mc.nearest_token_bucket(t, mc.FUSED_TOKEN_BUCKETS) == bucket
            ]
            fits[bucket] = [
                t for t in rows if sm.shared_in_fits(config, t, 256, D, INTER)
            ] == rows
            if fits[bucket]:
                # A consumer workgroup for each of the largest call's units; at the decode graph batch sizes
                # (<= 66 rows, 3 m-tiles, 60 units) the pre-phase takes less than a quarter of the consumers.
                consumers = (
                    256 * min(config.stage1.grid_mult, 2)
                    - 1
                    - config.stage1.num_dispatch_cu
                )
                assert -(-rows[-1] // 32) * 20 <= consumers, (bucket, consumers)
                assert 4 * (-(-min(rows[-1], 66) // 32) * 20) <= consumers, (
                    bucket,
                    consumers,
                )
            else:
                assert not any(
                    sm.shared_in_fits(config, t, 256, D, INTER) for t in rows
                ), bucket
    # Graph bs 1 (6 rows) .. 31 (186 rows) fuse; the token-wise combine buckets (1-5 rows) and the 64-row tiles of
    # 192+ rows keep the dense shared expert.
    assert fits == {
        1: False,
        4: False,
        8: True,
        16: True,
        32: True,
        64: True,
        128: True,
        256: False,
    }


def test_ep4_runtime_picks_the_split_of_each_graph_batch_and_skips_what_it_cannot_match(
    monkeypatch,
):
    from aiter.ops.flydsl.kernels.mega_moe import shared_expert as se

    with _override(_EP4_SERVED_OVERRIDE):
        _sm, rt, _launches, _ = _small(monkeypatch, world=4, epr=96)
    monkeypatch.setattr(
        se, "dense_kernel_ksplit", lambda rows, n, k: None
    )  # the table of d29c94315
    # graph bs -> rows = 6 bs; 36-60 rows take the split-K 2 variant, as the dense gate_up of those rows.
    want = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 11: 1, 12: 1}
    assert {bs: rt._shared_ksplit(6 * bs) for bs in want} == want
    assert all(
        rt.shared_in_supported(6 * bs) for bs in want
    ) and not rt.shared_in_supported(5)
    assert rt._shared_in_ksplits(8) == [1] and rt._shared_in_ksplits(16) == [1]
    assert rt._shared_in_ksplits(32) == [1, 2] and rt._shared_in_ksplits(64) == [1, 2]
    assert (
        rt._shared_in_ksplits(1) == []
        and rt._shared_in_ksplits(4) == []
        and rt._shared_in_ksplits(256) == []
    )

    # A re-tuned table: gate_up split-K 4 at 33..48 rows, unsplit at 49..64, a split down GEMM at 13..16 rows and
    # no tuned row from 100 rows on. The kernel follows the table or leaves those rows to the dense launches.
    def table(rows, n, k):
        if n == 2 * INTER:
            return 4 if 33 <= rows <= 48 else (0 if rows >= 100 else 1)
        return 2 if 13 <= rows <= 16 else 1

    monkeypatch.setattr(se, "dense_kernel_ksplit", table)
    assert [
        rt._shared_ksplit(r) for r in (12, 14, 30, 36, 48, 54, 60, 66, 99, 100)
    ] == [1, 0, 1, 0, 0, 1, 1, 1, 1, 0]
    assert (
        rt.shared_in_supported(54)
        and not rt.shared_in_supported(36)
        and not rt.shared_in_supported(14)
    )
    assert rt._shared_in_ksplits(32) == [1] and rt._shared_in_ksplits(64) == [1]
    shared, _ = _input([], 36)
    assert not rt.shared_in_ready(36, shared)
    with pytest.raises(ValueError):
        rt.forward(
            _rows(36),
            torch.zeros(36, 6),
            torch.zeros(36, 6, dtype=torch.int32),
            stream=object(),
            shared_in=shared,
        )


def test_dense_kernel_ksplit_reads_the_tuned_rows_kernel_name(monkeypatch):
    import aiter.ops.gemm_op_a8w8 as gemm
    from aiter.ops.flydsl.kernels.mega_moe import shared_expert as se

    rows = {
        (48, 2 * INTER, D): {
            "kernelName": "flydsl_bmm_mxfp8_mfma_t48x64x512_w1x4_nb4_sk2_bd2_xcd",
            "splitK": 2,
        },
        (96, 2 * INTER, D): {
            "kernelName": "flydsl_mxpsh_32x64x512_F8_F8_B16_w2_x4_sk1_pf",
            "splitK": 1,
        },
        (96, D, INTER): {
            "kernelName": "flydsl_mxpsh_32x64x128_F8_F8_B16_w2_x4_sk1_pf",
            "splitK": 1,
        },
        (48, D, INTER): {
            "kernelName": "flydsl_mxpsh_64x32x256_F8_F8_B16_w2_x0_sk1_pf_nt",
            "splitK": 1,
        },
        (7, 2 * INTER, D): {
            "kernelName": "flydsl_mxfp8_8w_256x256x128_F8_F8_B16_2x4",
            "splitK": 1,
        },
    }
    seen = []

    def lookup(m, n, k, block):
        seen.append(block)
        return rows.get((m, n, k))

    monkeypatch.setattr(gemm, "get_mxscale_bpreshuffle_config", lookup)
    assert (
        se.dense_kernel_ksplit(48, 2 * INTER, D) == 2
        and se.dense_kernel_ksplit(96, D, INTER) == 1
    )
    assert se.dense_kernel_ksplit(7, 2 * INTER, D) == 0  # a kernel name without a split
    assert (
        se.dense_kernel_ksplit(20, 2 * INTER, D) == 0
    )  # untuned: the dense launch runs a heuristic kernel
    assert set(seen) == {"32x32"}
    assert se.dense_gate_up_ksplit(48) == 2 and se.dense_gate_up_ksplit(96) == 1
    assert (
        se.dense_gate_up_ksplit(7) == 0
        and se.dense_gate_up_ksplit(20) == 0
        and se.dense_gate_up_ksplit(0) == 0
    )
    monkeypatch.setattr(
        gemm,
        "get_mxscale_bpreshuffle_config",
        lambda *a: (_ for _ in ()).throw(RuntimeError("no gfx")),
    )
    assert [se.dense_gate_up_ksplit(r) for r in (32, 33, 64, 65)] == [
        1,
        2,
        2,
        1,
    ]  # no lookup: the pinned table


def test_live_table_split_is_the_pinned_one_for_the_ep4_decode_rows():
    """With AITER's own tuned table (the CPU replay environment keys it as gfx950 / 256 CUs): what the kernels of
    the precompiled cache were built for. Skips where the process has no device description.
    """
    from aiter.ops.flydsl.kernels.mega_moe import shared_expert as se

    if se.dense_kernel_ksplit(48, 2 * INTER, D) is None:
        pytest.skip("no AITER tuned-table lookup in this process")
    live = {rows: se.dense_gate_up_ksplit(rows) for rows in range(1, 193)}
    assert live == {rows: se._static_gate_up_ksplit(rows) for rows in range(1, 193)}


def test_small_runtime_prepares_at_an_eager_call_and_not_under_a_capture(monkeypatch):
    with _override(_EP4_SERVED_OVERRIDE):
        _sm, rt, launches, capturing = _small(monkeypatch, world=4, epr=96)
    shared, _ = _input(
        [], 4096
    )  # an eager prefill call's rows: another route, the same weights
    capturing.now = True
    assert (
        not rt.shared_in_prepare(shared) and getattr(rt, "_sh_workspace", None) is None
    )
    small, _ = _input([], 48)
    small.gate_up, small.gate_up_scale, small.down, small.down_scale = (
        shared.gate_up,
        shared.gate_up_scale,
        shared.down,
        shared.down_scale,
    )
    assert not rt.shared_in_ready(
        48, small
    )  # nothing was built before this capture: the dense launches
    capturing.now = False
    assert rt.shared_in_prepare(shared) and rt.shared_in_prepare(shared)
    assert len(rt._sh_disp_by_key) == 1 and not launches
    capturing.now = True
    assert rt.shared_in_ready(
        48, small
    )  # the captured decode call of the same layer fuses
    capturing.now = False
    bad, _ = _input([], 6)
    bad.down = torch.zeros(D, INTER + 32, dtype=torch.uint8)
    assert not rt.shared_in_prepare(bad) and len(rt._sh_disp_by_key) == 1
    rt.swiglu_limit = 0.0
    assert not rt.shared_in_prepare(shared)


# ---------------------------------------------------------------------------------------------------------
# AITER_MEGA_SHARED_IN_OPT (default 0): the K halves of a split GEMM1 unit on two workgroups (bit 1), the
# pre-phase before the gate (bit 2)
_EP4_D64_OVERRIDE = (
    "32+64+128:s1.use_tile_resource=1,fs1.epi_evec=8,fs1.epi_lean=3,fs1.head_uniform=1,fs1.g2_epi_lean=2,"
    "fs1.expert_cached_w=2;64:s1.tile_n=256;32:s1.tile_n=256,s2.block_n=256;16+32+64+128:fs1.disp_order=31;"
    "16+32:fs1.sync_lean=1"
)


def _launch_opt(rt, launches, rows):
    shared, _ = _input([], rows)
    rt.forward(
        _rows(rows),
        torch.zeros(rows, 6),
        torch.zeros(rows, 6, dtype=torch.int32),
        stream=object(),
        shared_in=shared,
    )
    tokens, _, kw = launches[-1]
    assert tokens == rows and kw["shared_in"][0] == rt._shared_ksplit(rows)
    return kw.get("shared_in_opt", "absent")


def test_shared_in_opt_is_per_call_and_leaves_the_other_row_counts_their_kernels(
    monkeypatch,
):
    from aiter.ops.flydsl.kernels.mega_moe import shared_expert as se

    monkeypatch.delenv("AITER_MEGA_SHARED_IN_OPT", raising=False)
    monkeypatch.setattr(
        se, "dense_kernel_ksplit", lambda rows, n, k: None
    )  # the table of d29c94315
    with _override(_EP4_D64_OVERRIDE):
        sm, rt, launches, _ = _small(monkeypatch, world=4, epr=96)
    # graph bs 4, 6, 8, 10, 11 = 24 (unsplit), 36, 48, 60 (split-K 2, 2 m-tiles), 66 rows (unsplit, 3 m-tiles)
    rows = (24, 36, 48, 60, 66)
    assert [_launch_opt(rt, launches, r) for r in rows] == [
        "absent"
    ] * 5  # unset: the launches they were
    for env, want in (
        ("0", ["absent"] * 5),
        ("1", ["absent", 1, 1, 1, "absent"]),
        ("2", [2] * 5),
        ("3", [2, 3, 3, 3, 2]),
    ):
        monkeypatch.setenv("AITER_MEGA_SHARED_IN_OPT", env)
        assert [_launch_opt(rt, launches, r) for r in rows] == want, env
    rt.shared_in_opt = 1  # a benchmark's per-call switch wins over the environment
    assert [_launch_opt(rt, launches, r) for r in rows] == ["absent", 1, 1, 1, "absent"]
    rt.shared_in_opt = 0
    assert [_launch_opt(rt, launches, r) for r in rows] == ["absent"] * 5
    rt.shared_in_opt = None
    monkeypatch.setenv("AITER_MEGA_SHARED_IN_OPT", "32")
    with pytest.raises(ValueError):
        _launch_opt(rt, launches, 48)
    # the workspace every option shares: the partials of 2 m-tiles' GEMM1 units, an epoch word a unit; every table
    # (one per weight set, whatever the option) names it
    assert (
        rt._sh_workspace["part"].numel() == 36 * se.PARTIAL_UNIT_BYTES
        and se.PARTIAL_UNIT_BYTES == 256 * 8 * 16
    )
    assert rt._sh_workspace["sync"].numel() * 4 >= sm._SH_PART_READY + 4 * 36
    assert all(
        table[sm.DispatchSlot.SH_PART] == rt._sh_workspace["part"].data_ptr()
        for table in rt._sh_disp_by_key.values()
    )


def test_shared_in_opt_split_needs_a_workgroup_per_half():
    from dataclasses import replace

    from aiter.ops.flydsl.kernels.mega_moe import mega_moe_small_m as sm

    with _override(_EP4_D64_OVERRIDE) as mc:
        for bucket, rows in ((32, 36), (32, 47), (64, 48), (64, 64)):
            config = mc.select_fused_config(
                bucket,
                256,
                world_size=4,
                experts_per_rank=96,
                model_dim=D,
                inter_dim=INTER,
            )
            assert sm.shared_in_fits_split(config, rows, 256, D, INTER), (
                bucket,
                rows,
            )  # 72 of 287 consumers
            narrow = replace(
                config, stage1=replace(config.stage1, num_dispatch_cu=448)
            )  # 63 consumers: 40 units fit
            assert sm.shared_in_fits(
                narrow, rows, 256, D, INTER
            ) and not sm.shared_in_fits_split(narrow, rows, 256, D, INTER)
    assert (
        sm._SH_PART_READY >= sm._SH_DONE + 4
        and sm._SH_PART_READY + 4 * sm._SH_PART_UNITS <= 4 * sm._SH_SYNC_WORDS
    )
    assert sm._SH_PART_UNITS >= 2 * (2 * INTER // 256)


def test_partial_slots_give_every_thread_two_cache_lines_of_its_own():
    pytest.importorskip("aiter.ops.flydsl.kernels.silu_mul_quant_epilogue")
    from aiter.ops.flydsl.kernels.mega_moe import shared_expert as se

    seen = set()
    for unit in (0, 1, 35):
        for thread in (0, 1, 63, 64, 255):
            slots = [se.partial_slot(unit, thread, index) for index in range(8)]
            assert (
                slots == list(range(slots[0], slots[0] + 8))
                and slots[0] * 16 % 128 == 0
            )
            assert (
                unit * se.PARTIAL_UNIT_BYTES <= slots[0] * 16
                and slots[-1] * 16 + 16 <= (unit + 1) * se.PARTIAL_UNIT_BYTES
            )
            seen.update(slots)
    assert len(seen) == 3 * 5 * 8


class _SplitSyncModel(_SyncModel):
    """_SyncModel with shared_in_opt bit 1: consumer c runs K half c & 1 of GEMM1 unit c >> 1 (the second half
    waits for the first half's epoch before it counts), then GEMM2 unit c."""

    def launch(self, rows, consumers, epoch, rng, pre_gate=False):
        sm, w = self.sm, self.words
        tiles = (rows + 31) // 32
        assert 2 * tiles * self.N1 <= consumers and tiles * self.N1 <= sm._SH_PART_UNITS
        gate_open = not pre_gate
        pending = {c: ["g1", "g2", "gate"] for c in range(consumers)}
        steps = 0
        while pending:
            steps += 1
            if pre_gate and not gate_open and steps > 50:
                gate_open = True  # the owner opens the gate some time after the consumers started
            c = rng.choice(list(pending))
            step = pending[c][0]
            if step == "g1":
                if c < 2 * tiles * self.N1:
                    unit = c >> 1
                    if c & 1 == 0:
                        w[sm._SH_PART_READY // 4 + unit] = epoch
                    else:
                        if w[sm._SH_PART_READY // 4 + unit] != epoch:
                            continue  # its K steps are done; it waits for the first half's partials
                        m = unit // self.N1
                        w[m] += 1
                        if w[m] == self.N1:
                            w[sm._SH_READY // 4 + m] = epoch
                            w[m] -= self.N1
                pending[c].pop(0)
            elif step == "g2":
                if c < tiles * self.N2:
                    m = c // self.N2
                    if w[sm._SH_READY // 4 + m] != epoch:
                        continue
                    w[sm._SH_G2_COUNT // 4] += 1
                    if w[sm._SH_G2_COUNT // 4] == tiles * self.N2:
                        w[sm._SH_DONE // 4] = epoch
                        self.done_raised += 1
                        w[sm._SH_G2_COUNT // 4] -= tiles * self.N2
                pending[c].pop(0)
            else:
                if not gate_open:
                    continue
                pending[c].pop(0)
            if not pending[c]:
                del pending[c]
        return w[sm._SH_DONE // 4] == epoch


def test_split_halves_raise_the_done_epoch_once_and_never_wait_on_a_later_step():
    model, rng = _SplitSyncModel(), random.Random(7)
    for epoch, rows in enumerate([48, 33, 64, 60, 36, 54] * 4, start=1):
        assert model.launch(
            rows, rng.choice((287, 319, 72)), epoch, rng, pre_gate=bool(epoch & 1)
        ), (epoch, rows)
        assert model.done_raised == epoch
        assert model.words[:16] + [model.words[model.sm._SH_G2_COUNT // 4]] == [0] * 17
        # the unit epochs of this launch are all in; a later launch's waits cannot pass on them
        assert all(
            model.words[model.sm._SH_PART_READY // 4 + u] == epoch
            for u in range(2 * model.N1)
        )


# ---------------------------------------------------------------------------------------------------------
# AITER_MEGA_SHARED_IN_OPT bits 4 and 8: the units of one weight slab on one XCD, the lean routed unit loop
def test_pair_map_gives_every_unit_one_consumer_and_a_slab_one_xcd():
    from aiter.ops.flydsl.kernels.mega_moe import mega_moe_small_m as sm

    for tiles in range(1, 9):
        for n_tiles in (18, 20):
            need = sm.shared_pair_consumers(tiles, n_tiles)
            units = {}
            for c in range(need + 40):
                unit = sm.shared_pair_unit(c, tiles, n_tiles)
                if unit is not None:
                    assert (
                        unit not in units
                        and 0 <= unit[0] < tiles
                        and 0 <= unit[1] < n_tiles
                    )
                    units[unit] = c
            assert (
                len(units) == tiles * n_tiles and max(units.values()) == need - 1
            )  # every unit once, none past
            for n in range(n_tiles):
                # the m-tiles' units of a column tile: 8 consumers apart = one XCD under static roles
                assert [units[(m, n)] - units[(0, n)] for m in range(tiles)] == [
                    8 * m for m in range(tiles)
                ]
            # both GEMMs of a consumer are the same (m-tile, column tile), where it has both
            assert all(
                sm.shared_pair_unit(c, tiles, 18)
                in (None, sm.shared_pair_unit(c, tiles, 20))
                for c in range(need)
            )
    assert [sm.shared_pair_unit(c, 1, 20) for c in range(21)] == [
        (0, c) for c in range(20)
    ] + [
        None
    ]  # 1 m-tile: m-major
    assert (
        sm.shared_pair_unit(-1, 2, 18) is None and sm.shared_pair_unit(3, 0, 18) is None
    )
    assert (
        sm.shared_pair_consumers(0, 20) == 0
        and sm.shared_pair_consumers(2, 20) == 44
        and sm.shared_pair_consumers(3, 20) == 68
    )


def test_shared_in_opt_bits_4_and_8_are_per_call_and_keep_the_other_kernels(
    monkeypatch,
):
    from dataclasses import replace

    from aiter.ops.flydsl.kernels.mega_moe import shared_expert as se

    monkeypatch.delenv("AITER_MEGA_SHARED_IN_OPT", raising=False)
    monkeypatch.setattr(
        se, "dense_kernel_ksplit", lambda rows, n, k: None
    )  # the table of d29c94315
    with _override(_EP4_D64_OVERRIDE):
        sm, rt, launches, _ = _small(monkeypatch, world=4, epr=96)
    # graph bs 2, 4, 6, 8, 10, 11, 16 = 12, 24 (one m-tile), 36, 48, 60 (split-K 2, 2 m-tiles), 66, 96 rows (3 m-tiles)
    rows = (12, 24, 36, 48, 60, 66, 96)
    for env, want in (
        (
            "4",
            ["absent", "absent", 4, 4, 4, 4, 4],
        ),  # bit 4: calls of two or more m-tiles only
        ("8", [8] * 7),
        ("12", [8, 8, 12, 12, 12, 12, 12]),
        (
            "5",
            ["absent", "absent", 1, 1, 1, 4, 4],
        ),  # bit 1's half-unit workgroups keep their own map
        ("14", [10, 10, 14, 14, 14, 14, 14]),
        ("16", ["absent"] * 7),  # the stagger alone staggers nothing
        ("20", ["absent", "absent", 20, 20, 20, 20, 20]),
        ("28", [8, 8, 28, 28, 28, 28, 28]),
        ("21", ["absent", "absent", 1, 1, 1, 20, 20]),
    ):
        monkeypatch.setenv("AITER_MEGA_SHARED_IN_OPT", env)
        assert [_launch_opt(rt, launches, r) for r in rows] == want, env
    rt.shared_in_opt = 0  # a benchmark's per-call switch wins over the environment
    assert [_launch_opt(rt, launches, r) for r in rows] == ["absent"] * 7
    rt.shared_in_opt = 12
    monkeypatch.delenv("AITER_MEGA_SHARED_IN_OPT")
    assert [_launch_opt(rt, launches, r) for r in rows] == [8, 8, 12, 12, 12, 12, 12]
    # the bundle preloads every (K split, option) a bucket's row counts take: bucket 32 holds one- and two-tile calls
    assert rt._shared_in_variants(32) == [(1, 8), (2, 12)] and rt._shared_in_variants(
        64
    ) == [(1, 12), (2, 12)]
    assert rt._shared_in_variants(16) == [(1, 8)] and rt._shared_in_variants(128) == [
        (1, 12)
    ]
    rt.shared_in_opt = 8
    assert rt._shared_in_variants(32) == [(1, 8), (2, 8)]
    rt.shared_in_opt = 28
    assert rt._shared_in_variants(32) == [(1, 8), (2, 28)] and rt._shared_in_variants(
        64
    ) == [(1, 28), (2, 28)]
    rt.shared_in_opt = None  # unset: one kernel per K split, as before the bits existed
    assert rt._shared_in_variants(32) == [(1, 0), (2, 0)] and rt._shared_in_variants(
        64
    ) == [(1, 0), (2, 0)]
    # bit 4 needs a consumer at the map's highest index, bit 8 a kernel without an early GEMM2 start
    config = rt._configs[64]
    assert sm.shared_in_fits_pair(config, 64, 256, D, INTER) and sm.shared_in_fits_pair(
        config, 95, 256, D, INTER
    )
    narrow = replace(
        config, stage1=replace(config.stage1, num_dispatch_cu=448)
    )  # 63 consumers: 40 m-major units fit
    assert sm.shared_in_fits(narrow, 64, 256, D, INTER) and sm.shared_in_fits_pair(
        narrow, 64, 256, D, INTER
    )  # 44
    assert sm.shared_in_fits(narrow, 96, 256, D, INTER) and not sm.shared_in_fits_pair(
        narrow, 96, 256, D, INTER
    )  # 68
    assert sm.shared_in_lean_fits(config)
    assert not sm.shared_in_lean_fits(
        replace(config, fused_stage2=replace(config.fused_stage2, early_k=2))
    )
    assert not sm.shared_in_lean_fits(
        replace(config, fused_stage1=replace(config.fused_stage1, plan_local=False))
    )
    rt._configs[64] = replace(
        config, fused_stage2=replace(config.fused_stage2, early_k=2)
    )
    rt.shared_in_opt = 12
    assert _launch_opt(rt, launches, 48) == 4


class _PairSyncModel(_SyncModel):
    """_SyncModel with shared_in_opt bit 4: consumer c runs the GEMM1 unit, then the GEMM2 unit, of
    shared_pair_unit(c, ...) -- the same counters and epochs, another unit per consumer.
    """

    def launch(self, rows, consumers, epoch, rng):
        sm, w = self.sm, self.words
        tiles = (rows + 31) // 32
        assert (
            sm.shared_pair_consumers(tiles, max(self.N1, self.N2)) <= consumers
        ), "shared_in_fits_pair"
        pending = {c: ["g1", "g2"] for c in range(consumers)}
        ran = {"g1": [], "g2": []}
        while pending:
            c = rng.choice(list(pending))
            step = pending[c][0]
            unit = sm.shared_pair_unit(c, tiles, self.N1 if step == "g1" else self.N2)
            if unit is not None:
                m = unit[0]
                if step == "g1":
                    w[m] += 1
                    if w[m] == self.N1:
                        w[sm._SH_READY // 4 + m] = epoch
                        w[m] -= self.N1
                else:
                    if w[sm._SH_READY // 4 + m] != epoch:
                        continue  # still waiting for its m-tile's GEMM1 units; someone else runs
                    w[sm._SH_G2_COUNT // 4] += 1
                    if w[sm._SH_G2_COUNT // 4] == tiles * self.N2:
                        w[sm._SH_DONE // 4] = epoch
                        self.done_raised += 1
                        w[sm._SH_G2_COUNT // 4] -= tiles * self.N2
                ran[step].append(unit)
            pending[c].pop(0)
            if not pending[c]:
                del pending[c]
        for step, n_tiles in (("g1", self.N1), ("g2", self.N2)):
            assert sorted(ran[step]) == [
                (m, n) for m in range(tiles) for n in range(n_tiles)
            ], (rows, step)
        return rows == 0 or w[sm._SH_DONE // 4] == epoch


def test_pair_map_raises_the_done_epoch_once_and_runs_every_unit_once():
    model, rng = _PairSyncModel(), random.Random(45)
    launches_with_rows = 0
    for epoch, rows in enumerate(
        [48, 33, 64, 0, 60, 66, 95, 1, 32, 96, 255, 36] * 3, start=1
    ):
        consumers = rng.choice(
            (287, 383, 192)
        )  # dcu224 / dcu128 grids, and the fewest that fit 256 rows
        assert model.launch(rows, consumers, epoch, rng), (epoch, rows)
        launches_with_rows += rows > 0
        assert model.done_raised == launches_with_rows
        assert (
            model.words[:16] + [model.words[model.sm._SH_G2_COUNT // 4]] == [0] * 17
        ), (epoch, rows)


def test_lean_loop_words_do_not_overlap():
    from aiter.ops.flydsl.kernels.mega_moe import mega_moe_small_m as sm

    words = [
        sm._LEAN_PLAN_SEEN,
        sm._LEAN_UNIT_READY,
        sm._LEAN_SHARED_DONE,
        sm._LEAN_EPOCH,
        sm._LEAN_BASE,
        sm._LEAN_UNITS,
        sm._LEAN_LIMIT,
    ]
    quads = [sm._LEAN_PLAN_ADDR, sm._LEAN_READY_ADDR, sm._LEAN_DONE_ADDR]
    taken = [b for w in words for b in range(w, w + 4)] + [
        b for q in quads for b in range(q, q + 8)
    ]
    assert (
        len(set(taken)) == len(taken)
        and max(taken) < sm._LEAN_BYTES
        and sm._LEAN_BYTES % 16 == 0
    )
    assert (
        all(w % 4 == 0 for w in words)
        and all(q % 8 == 0 for q in quads)
        and 1 <= sm._LEAN_EPOCHS <= 16
    )
    assert 1 <= sm._PAIR_STAGGER_UNITS <= 127  # one s_sleep
