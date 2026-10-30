# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for the two-kernel FlashKDA path against the default chunk_delta_attn
pipeline.

Both compute the same forward, so the default pipeline is the reference. They
are not bit-identical: the fused path keeps the recurrent state in registers
across the whole sequence and inverts (I - L) with bf16 multiplicative doubling,
where the default path materializes the state per chunk and uses an exact
forward substitution. The gap is bf16-level, so the checks are on relative
error rather than allclose.
"""

import contextlib
import importlib
import inspect
import math

import pytest
import torch
import triton

from aiter.ops.triton._triton_kernels.kimi_delta_attn import chunk_delta_attn_fwd
from aiter.ops.triton._triton_kernels.kimi_delta_attn import chunk_fwd as _chunk_fwd
from aiter.ops.triton._triton_kernels.kimi_delta_attn import fast_launch as _fast_launch
from aiter.ops.triton._triton_kernels.kimi_delta_attn import flash_kda as _flash_kda
from aiter.ops.triton._triton_kernels.kimi_delta_attn.chunk_delta_attn_utils import (
    input_guard,
)
from aiter.ops.triton._triton_kernels.kimi_delta_attn.flash_kda import (
    FLASH_KDA_CHUNK,
    flash_kda_fwd,
    flash_kda_supported,
)
from aiter.ops.triton.kimi_delta_attn import chunk_kimi_delta_attn
from op_tests.triton_tests.utils.kda_ref import chunk_kda_ref

device = "cuda"
dtype = torch.bfloat16

LOWER_BOUND = -5.0
K_DIM = 128


def make_inputs(B, T, H, seed=42):
    torch.manual_seed(seed)
    K = V = K_DIM
    scale = 1.0 / math.sqrt(K)
    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
    beta = torch.randn(B, T, H, device=device, dtype=torch.float32)
    A_log = torch.randn(H, device=device, dtype=torch.float32).abs() * 0.5
    dt_bias = torch.randn(H * K, device=device, dtype=torch.float32) * 0.1
    return q, k, v, g, beta, A_log, dt_bias, scale


@contextlib.contextmanager
def _force_default_pipeline():
    """Pin the reference to the five-kernel path.

    ``chunk_delta_attn_fwd`` dispatches to ``flash_kda_fwd`` whenever
    AITER_FDA_ENABLE is set. Without this, running the suite with that
    variable exported makes every comparison here flash_kda against itself,
    which passes unconditionally.
    """
    saved = _chunk_fwd.AITER_FDA_ENABLE
    _chunk_fwd.AITER_FDA_ENABLE = False
    try:
        yield
    finally:
        _chunk_fwd.AITER_FDA_ENABLE = saved


@contextlib.contextmanager
def _route(k1: bool | None = None, k2: bool | None = None):
    """Pin the named kernels to one of their two implementations.

    AITER_FDA_USE_GLUON is read once at import and resolved into these two
    flags, so a test that wants the other route sets the flags rather than the
    variable. Both default to on wherever the tile shape and the arch allow it,
    which leaves the Triton kernels unreached by every test that does not come
    through here. ``k2=True`` also pins pass C to Gluon on small shapes.
    """
    saved = (
        _flash_kda.AITER_FDA_USE_GLUON_K1,
        _flash_kda.AITER_FDA_USE_GLUON_K2,
        _flash_kda._K2C_ROUTE_BY_SIZE,
    )
    if k1 is not None:
        _flash_kda.AITER_FDA_USE_GLUON_K1 = k1
    if k2 is not None:
        _flash_kda.AITER_FDA_USE_GLUON_K2 = k2
        _flash_kda._K2C_ROUTE_BY_SIZE = not k2
    try:
        yield
    finally:
        (
            _flash_kda.AITER_FDA_USE_GLUON_K1,
            _flash_kda.AITER_FDA_USE_GLUON_K2,
            _flash_kda._K2C_ROUTE_BY_SIZE,
        ) = saved


@contextlib.contextmanager
def _count_launches(module, names):
    """Count the launches of ``module``'s fast launchers ``names``."""
    calls = dict.fromkeys(names, 0)
    saved = {name: getattr(module, name) for name in names}

    class _Counting:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            inner = saved[self.name][grid]

            def launch(**kw):
                calls[self.name] += 1
                return inner(**kw)

            return launch

    for name in names:
        setattr(module, name, _Counting(name))
    try:
        yield calls
    finally:
        for name in names:
            setattr(module, name, saved[name])


def _gluon_k2_module():
    return importlib.import_module(
        "aiter.ops.triton._gluon_kernels."
        f"{_flash_kda.DEVICE_ARCH}.kimi_delta_attn.flash_kda_k2"
    )


@contextlib.contextmanager
def _k2_tuner_search_space():
    """Hand K2's autotuner the candidates a tuning build gives it.

    The launch path pins one config, and Triton consults its autotune cache
    only when there is more than one to choose between, so a test about the
    key has to supply the space the key indexes.
    """
    if len(_flash_kda._K2_CONFIGS) < 2:
        pytest.skip("no published K2 candidates on this device to key between")
    kern = _flash_kda._flash_kda_segment_kernel
    saved = kern.configs
    kern.configs = list(_flash_kda._K2_CONFIGS)
    try:
        yield kern
    finally:
        kern.configs = saved
        kern.cache.clear()


def run_reference(q, k, v, g, beta, A_log, dt_bias, scale, **kw):
    """Default five-kernel pipeline."""
    with _force_default_pipeline():
        o, final_state, *_ = chunk_delta_attn_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            A_log=A_log,
            dt_bias=dt_bias,
            chunk_size=FLASH_KDA_CHUNK,
            safe_gate=True,
            lower_bound=kw.get("lower_bound", LOWER_BOUND),
            use_gate_in_kernel=True,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            disable_recompute=False,
            initial_state=kw.get("initial_state"),
            output_final_state=kw.get("output_final_state", False),
            state_v_first=kw.get("state_v_first", False),
            cu_seqlens=kw.get("cu_seqlens"),
        )
    return o, final_state


def run_flash(q, k, v, g, beta, A_log, dt_bias, scale, **kw):
    return flash_kda_fwd(
        chunks_per_seg=kw.get("chunks_per_seg"),
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        lower_bound=kw.get("lower_bound", LOWER_BOUND),
        initial_state=kw.get("initial_state"),
        output_final_state=kw.get("output_final_state", False),
        state_v_first=kw.get("state_v_first", False),
        cu_seqlens=kw.get("cu_seqlens"),
    )


def fp32_gold(q, k, v, g, beta, A_log, dt_bias, scale, **kw):
    """Token-at-a-time fp32 recurrence, driven by the same kw dict as the others.

    bf16 inputs upcast exactly, so this is the same call in fp32. A bf16 gold
    would round away the gap the ratio assertions below measure.
    """
    return chunk_kda_ref(
        q=q.float(),
        k=k.float(),
        v=v.float(),
        g=g.float(),
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=kw.get("lower_bound", LOWER_BOUND),
        initial_state=kw.get("initial_state"),
        output_final_state=kw.get("output_final_state", False),
        state_v_first=kw.get("state_v_first", False),
        cu_seqlens=kw.get("cu_seqlens"),
    )


def rel_err(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / (b.norm() + 1e-6)).item()


@pytest.mark.parametrize("B,T,H", [(1, 256, 4), (2, 512, 8), (1, 1024, 2)])
def test_batched_matches_reference(B, T, H):
    args = make_inputs(B, T, H)
    o_ref, _ = run_reference(*args)
    o_fkda, _ = run_flash(*args)
    assert rel_err(o_fkda, o_ref) < 2e-2


@pytest.mark.parametrize("T", [192, 200, 65, 320])
def test_tail_chunk(T):
    """T not a multiple of 64 exercises the partial trailing chunk."""
    args = make_inputs(1, T, 4)
    o_ref, _ = run_reference(*args)
    o_fkda, _ = run_flash(*args)
    assert rel_err(o_fkda, o_ref) < 2e-2


@pytest.mark.parametrize(
    "lens",
    [
        [128, 64],
        [100, 156, 33],  # all three end mid-chunk
        [64, 1, 191],  # single-token sequence next to a long one
        [512],
    ],
)
def test_varlen_matches_reference(lens):
    total = sum(lens)
    cu_seqlens = torch.tensor(
        [0] + list(torch.tensor(lens, device=device).cumsum(0)),
        device=device,
        dtype=torch.long,
    )
    args = make_inputs(1, total, 4)
    o_ref, _ = run_reference(*args, cu_seqlens=cu_seqlens)
    o_fkda, _ = run_flash(*args, cu_seqlens=cu_seqlens)
    assert rel_err(o_fkda, o_ref) < 2e-2


@pytest.mark.parametrize("state_v_first", [False, True])
def test_final_state(state_v_first):
    B, T, H = 2, 256, 4
    args = make_inputs(B, T, H)
    kw = {"output_final_state": True, "state_v_first": state_v_first}
    o_ref, ht_ref = run_reference(*args, **kw)
    o_fkda, ht_fkda = run_flash(*args, **kw)
    assert ht_fkda is not None and ht_fkda.shape == ht_ref.shape
    assert rel_err(o_fkda, o_ref) < 2e-2
    assert rel_err(ht_fkda, ht_ref) < 2e-2


@pytest.mark.parametrize("state_v_first", [False, True])
def test_initial_state_roundtrip(state_v_first):
    """Carrying a state in and back out, as chunked prefill does."""
    B, T, H = 2, 192, 4
    args = make_inputs(B, T, H)
    shape = (B, H, K_DIM, K_DIM)
    h0 = torch.randn(*shape, device=device, dtype=torch.float32) * 0.1
    kw = {
        "initial_state": h0,
        "output_final_state": True,
        "state_v_first": state_v_first,
    }
    o_ref, ht_ref = run_reference(*args, **kw)
    o_fkda, ht_fkda = run_flash(*args, **kw)
    assert rel_err(o_fkda, o_ref) < 2e-2
    assert rel_err(ht_fkda, ht_ref) < 2e-2


@pytest.mark.parametrize("state_v_first", [False, True])
def test_varlen_initial_state(state_v_first):
    """Packed sequences threading a state, in both state layouts.

    varlen and a V-first state are covered apart above but only together in the
    serving path, where prefill carries a state per sequence and hands it to
    decode in that layout.
    """
    lens = [128, 100, 64]
    cu_seqlens = torch.tensor(
        [0] + list(torch.tensor(lens, device=device).cumsum(0)),
        device=device,
        dtype=torch.long,
    )
    args = make_inputs(1, sum(lens), 4)
    h0 = (
        torch.randn(len(lens), 4, K_DIM, K_DIM, device=device, dtype=torch.float32)
        * 0.1
    )
    kw = {
        "cu_seqlens": cu_seqlens,
        "initial_state": h0,
        "output_final_state": True,
        "state_v_first": state_v_first,
    }
    o_ref, ht_ref = run_reference(*args, **kw)
    o_fkda, ht_fkda = run_flash(*args, **kw)
    assert rel_err(o_fkda, o_ref) < 2e-2
    assert rel_err(ht_fkda, ht_ref) < 2e-2


def test_chunked_prefill_equivalence():
    """Splitting a sequence and threading the state must match one shot."""
    B, T, H = 1, 256, 4
    q, k, v, g, beta, A_log, dt_bias, scale = make_inputs(B, T, H)
    o_full, ht_full = run_flash(
        q, k, v, g, beta, A_log, dt_bias, scale, output_final_state=True
    )

    split = 128
    o1, ht1 = run_flash(
        q[:, :split],
        k[:, :split],
        v[:, :split],
        g[:, :split],
        beta[:, :split],
        A_log,
        dt_bias,
        scale,
        output_final_state=True,
    )
    o2, ht2 = run_flash(
        q[:, split:],
        k[:, split:],
        v[:, split:],
        g[:, split:],
        beta[:, split:],
        A_log,
        dt_bias,
        scale,
        initial_state=ht1,
        output_final_state=True,
    )
    o_split = torch.cat([o1, o2], dim=1)
    assert rel_err(o_split, o_full) < 2e-2
    assert rel_err(ht2, ht_full) < 2e-2


# The shapes above are far too short for the heuristic to segment, so the
# segmented path is pinned explicitly here. chunks_per_seg=1 is the extreme
# case: every chunk becomes its own segment, so the scan carries the entire
# recurrence and any error in the affine decomposition shows up immediately.
@pytest.mark.parametrize("chunks_per_seg", [1, 2, 3, 8])
def test_segmented_matches_reference(chunks_per_seg):
    args = make_inputs(2, 512, 4)
    o_ref, _ = run_reference(*args)
    o_seg, _ = run_flash(*args, chunks_per_seg=chunks_per_seg)
    assert rel_err(o_seg, o_ref) < 2e-2


@pytest.mark.parametrize("chunks_per_seg", [1, 3])
def test_segmented_states(chunks_per_seg):
    """Segmenting must not disturb either end of the state threading."""
    B, T, H = 2, 384, 4
    args = make_inputs(B, T, H)
    h0 = torch.randn(B, H, K_DIM, K_DIM, device=device, dtype=torch.float32) * 0.1
    kw = {"initial_state": h0, "output_final_state": True}
    o_ref, ht_ref = run_reference(*args, **kw)
    o_seg, ht_seg = run_flash(*args, **kw, chunks_per_seg=chunks_per_seg)
    assert rel_err(o_seg, o_ref) < 2e-2
    assert rel_err(ht_seg, ht_ref) < 2e-2


def test_segmented_varlen_uneven():
    """Sequences whose last segment is short, and one shorter than a segment."""
    lens = [500, 33, 257, 64]
    cu_seqlens = torch.tensor(
        [0] + list(torch.tensor(lens, device=device).cumsum(0)),
        device=device,
        dtype=torch.long,
    )
    args = make_inputs(1, sum(lens), 4)
    h0 = torch.randn(len(lens), 4, K_DIM, K_DIM, device=device, dtype=torch.float32)
    kw = {
        "cu_seqlens": cu_seqlens,
        "initial_state": h0 * 0.1,
        "output_final_state": True,
    }
    o_ref, ht_ref = run_reference(*args, **kw)
    o_seg, ht_seg = run_flash(*args, **kw, chunks_per_seg=3)
    assert rel_err(o_seg, o_ref) < 2e-2
    assert rel_err(ht_seg, ht_ref) < 2e-2


def test_segmented_agrees_with_unsegmented():
    """The segment count is a scheduling choice and must not change the result."""
    args = make_inputs(1, 1024, 4)
    o_plain, ht_plain = run_flash(*args, output_final_state=True, chunks_per_seg=0)
    for m in (2, 7, 16):
        o_seg, ht_seg = run_flash(*args, output_final_state=True, chunks_per_seg=m)
        assert rel_err(o_seg, o_plain) < 5e-3, m
        assert rel_err(ht_seg, ht_plain) < 5e-3, m


def test_tuner_keeps_the_two_schedules_apart():
    """A segmented run must not hand its config to a later unsegmented one.

    K2's grid is ``cdiv(W, BW) * num_segs * H``, so the wide BW that suits a
    segmented sweep leaves the unsegmented scan a quarter of the blocks: at
    H=12 that pick costs 2.3x. The two collide unless the segment count reaches
    the autotune key, and `cache_results` then persists whichever won. Only a
    tuning build chooses, so the tuner is given its config space here.
    """
    args = make_inputs(1, 1024, 4)
    # An incoming state is what makes the two output passes otherwise identical:
    # without it the unsegmented one passes h_in=None and the key picks up the
    # difference through the dtypes it appends, hiding the collision.
    kw = {
        "initial_state": torch.zeros(
            1, 4, K_DIM, K_DIM, device=device, dtype=torch.float32
        )
    }
    # This is about the Triton kernel's autotuner, which never runs -- and whose
    # cache therefore stays empty -- when K2 is routed to Gluon.
    with _k2_tuner_search_space() as kern, _route(k2=False), _fast_launch.bypassed():
        run_flash(*args, chunks_per_seg=4, **kw)
        segmented_keys = set(kern.cache)
        run_flash(*args, chunks_per_seg=0, **kw)
        assert set(kern.cache) - segmented_keys, "unsegmented reused a segmented config"


def test_published_k2_schedules_can_split_their_tile():
    """Every published (BW, num_warps) must let the warps divide the state.

    Above BW // 16 the extra warps recompute columns their neighbours already
    hold -- still correct, which is why nothing downstream catches it, and the
    pairs are now editable from a config file rather than derived.

    An arch that publishes neither schedule has nothing here to check. Both
    lookups then return _K2_GLUON_FALLBACK, whose MIN_BLOCKS_PER_CU of 0 makes
    the occupancy test vacuous, so the wide branch is taken for every shape and
    exactly one pair is reachable by construction rather than by choice.
    """
    if (
        _flash_kda.chunk_delta_attn_tuned_config(
            "k2_ab_fused_gluon_wide", _flash_kda._K2_GLUON_FALLBACK, backend="gluon"
        )
        is _flash_kda._K2_GLUON_FALLBACK
    ):
        pytest.skip("this arch publishes no Gluon K2 schedules")
    kernels = ["k2_ab_fused_gluon"]
    if _flash_kda.DEVICE_ARCH == "gfx1250":
        kernels.append("k2_c_gluon")
    for kernel in kernels:
        reached = set()
        for W in (64, 128, 256):
            for num_segs in (1, 2, 8, 64, 512):
                for H in (1, 4, 12, 64):
                    sched = _flash_kda._k2_gluon_schedule(W, num_segs, H, kernel)
                    reached.add((sched.kwargs["BW"], sched.num_warps))
        assert len(reached) > 1, f"{kernel}: only one schedule reachable: {reached}"
        for bw, nw in sorted(reached):
            assert bw % 16 == 0, f"{kernel}: BW={bw} is not a whole number of tiles"
            assert nw <= bw // 16, f"{kernel}: BW={bw} cannot be split {nw} ways"
    # The gfx950 pass C publishes one schedule per segmentation instead.
    if _flash_kda.DEVICE_ARCH == "gfx950":
        for segmented in (False, True):
            sched = _flash_kda._k2c_gluon_schedule(segmented)
            if sched is None:
                continue
            bw, nw = sched.kwargs["BW"], sched.num_warps
            assert bw % 16 == 0, f"k2_c_gluon: BW={bw} is not a whole number of tiles"
            assert nw <= bw // 16, f"k2_c_gluon: BW={bw} cannot be split {nw} ways"


def _patch_gluon_schedule(monkeypatch, replace):
    """Make ``_published_gluon_schedule`` return ``replace[name]`` for those names."""
    published = _flash_kda._published_gluon_schedule
    monkeypatch.setattr(
        _flash_kda,
        "_published_gluon_schedule",
        lambda name: replace[name] if name in replace else published(name),
    )


def test_gluon_k1_rejects_a_wrong_warp_count(monkeypatch):
    """K1's layouts fix its warp count, so a config with another one must fail."""
    if _flash_kda.DEVICE_ARCH != "gfx950" or not _flash_kda._gluon_k1_usable(
        FLASH_KDA_CHUNK, K_DIM
    ):
        pytest.skip("only the gfx950 Gluon K1 validates its published schedule")
    good = _flash_kda._published_gluon_schedule("k1_prepare_gluon")
    wrong = triton.Config(dict(good.kwargs), num_warps=2 * good.num_warps)
    _patch_gluon_schedule(monkeypatch, {"k1_prepare_gluon": wrong})
    with _route(k1=True), pytest.raises(ValueError, match="num_warps"):
        run_flash(*make_inputs(1, 64, 2))


def test_unpublished_gluon_schedules_fall_back_to_triton(monkeypatch):
    """Without a published schedule, K1 and pass C take their Triton kernels."""
    if _flash_kda.DEVICE_ARCH != "gfx950":
        pytest.skip("only the gfx950 K1 and pass C fail closed")
    unpublished = ("k1_prepare_gluon", "k2_c_gluon", "k2_c_gluon_segmented")
    _patch_gluon_schedule(monkeypatch, dict.fromkeys(unpublished))
    args = make_inputs(1, 1024, 4)
    o_ref, _ = run_reference(*args)
    with (
        _route(k1=True, k2=True),
        _count_launches(_flash_kda, ["_prepare_fast", "_segment_fast"]) as calls,
    ):
        o, _ = run_flash(*args, chunks_per_seg=4)
    assert calls == {"_prepare_fast": 1, "_segment_fast": 1}
    assert rel_err(o, o_ref) < 2e-2


def _varlen(lens):
    return torch.tensor(
        [0] + list(torch.tensor(lens, device=device).cumsum(0)),
        device=device,
        dtype=torch.long,
    )


# One configuration per branch the two kernels take, so the route below is
# covered where it can differ rather than on a single smoke shape. Built on call
# because two of them carry tensors.
_CASES = {
    "batched": lambda: (make_inputs(2, 512, 8), {}),
    "tail chunk": lambda: (make_inputs(1, 200, 4), {}),
    "final state": lambda: (make_inputs(2, 256, 4), {"output_final_state": True}),
    "initial state": lambda: (
        make_inputs(2, 192, 4),
        {
            "initial_state": torch.randn(
                2, 4, K_DIM, K_DIM, device=device, dtype=torch.float32
            )
            * 0.1,
            "output_final_state": True,
        },
    ),
    "varlen v-first": lambda: (
        make_inputs(1, 292, 4),
        {
            "cu_seqlens": _varlen([128, 100, 64]),
            "output_final_state": True,
            "state_v_first": True,
        },
    ),
    "weak gate": lambda: (make_inputs(1, 512, 4), {"lower_bound": -0.01}),
    # Only a segmented shape has a pass A, the only Gluon K2 pass on gfx950.
    "segmented": lambda: (
        make_inputs(1, 1024, 4),
        {"output_final_state": True, "chunks_per_seg": 4},
    ),
    "segmented varlen": lambda: (
        make_inputs(1, 292, 4),
        {
            "cu_seqlens": _varlen([128, 100, 64]),
            "output_final_state": True,
            "state_v_first": True,
            "chunks_per_seg": 2,
        },
    ),
    "segmented initial state": lambda: (
        make_inputs(1, 1024, 4),
        {
            "initial_state": torch.randn(
                1, 4, K_DIM, K_DIM, device=device, dtype=torch.float32
            )
            * 0.1,
            "output_final_state": True,
            "chunks_per_seg": 3,
        },
    ),
    "segmented weak gate": lambda: (
        make_inputs(1, 512, 4),
        {"lower_bound": -0.01, "chunks_per_seg": 3},
    ),
}


@pytest.mark.parametrize("case", list(_CASES))
def test_triton_route_matches_reference(case):
    """The Triton kernels against the default pipeline, on the same bound.

    Every other comparison in this file runs whatever AITER_FDA_USE_GLUON
    resolved to, which is Gluon for both kernels on the arch this suite runs on.
    Without this the Triton K1 and K2 ship untested.
    """
    args, kw = _CASES[case]()
    o_ref, ht_ref = run_reference(*args, **kw)
    with _route(k1=False, k2=False):
        o, ht = run_flash(*args, **kw)
    assert rel_err(o, o_ref) < 2e-2
    if kw.get("output_final_state"):
        assert rel_err(ht, ht_ref) < 2e-2


@pytest.mark.parametrize("case", list(_CASES))
def test_routes_agree(case):
    """Which implementation ran must not be visible in the answer.

    Both sides write the same ABI and nothing downstream is told which one ran,
    so a divergence here is a bug in whichever kernel moved rather than a
    tolerance to widen.
    """
    args, kw = _CASES[case]()
    with _route(k1=False, k2=False):
        o_triton, ht_triton = run_flash(*args, **kw)
    for k1, k2 in ((True, False), (False, True), (True, True)):
        with _route(k1=k1, k2=k2):
            o, ht = run_flash(*args, **kw)
        assert rel_err(o, o_triton) < 2e-3, f"K1={k1} K2={k2}"
        if kw.get("output_final_state"):
            assert rel_err(ht, ht_triton) < 1e-4, f"K1={k1} K2={k2}"


def test_cases_reach_the_gluon_k2():
    """The cases above have to exercise the route they are comparing."""
    if not _flash_kda._gluon_k2_usable(FLASH_KDA_CHUNK, K_DIM, K_DIM):
        pytest.skip("this arch or tile shape never routes K2 to Gluon")
    launchers = ["k2_ab_fused_fast"]
    if _flash_kda._GLUON_K2C_ARCH_OK:
        launchers.append("k2_c_fast")
    reached = {name: set() for name in launchers}
    with _count_launches(_gluon_k2_module(), launchers) as calls:
        for current, make in _CASES.items():
            args, kw = make()
            before = dict(calls)
            with _route(k1=True, k2=True):
                run_flash(*args, **kw)
            for name in launchers:
                if calls[name] > before[name]:
                    reached[name].add(current)

    want = {name for name in _CASES if name.startswith("segmented")}
    missed = sorted(want - reached["k2_ab_fused_fast"])
    assert not missed, f"never reached the Gluon pass A: {missed}"
    if "k2_c_fast" in reached:
        missed = sorted(set(_CASES) - reached["k2_c_fast"])
        assert not missed, f"never reached the Gluon pass C: {missed}"


@pytest.mark.parametrize(
    "B, H, chunks_per_seg, gluon",
    [(1, 4, None, False), (4, 12, None, True), (1, 4, 4, True)],
    ids=["small", "large", "segmented"],
)
def test_pass_c_route_follows_size(monkeypatch, B, H, chunks_per_seg, gluon):
    """The published floor sends few-pair unsegmented shapes to the Triton pass C."""
    if not (
        _flash_kda._GLUON_K2C_ARCH_OK
        and _flash_kda._gluon_k2_usable(FLASH_KDA_CHUNK, K_DIM, K_DIM)
    ):
        pytest.skip("this arch or tile shape never routes pass C to Gluon")
    sched = _flash_kda._k2c_gluon_schedule(False)
    if sched is None or not sched.kwargs.get("MIN_BLOCKS_PER_CU"):
        pytest.skip("this arch publishes no pass C floor")
    if chunks_per_seg is None:
        assert _flash_kda._k2c_gluon_usable(False, K_DIM, B, H) == gluon
    monkeypatch.setattr(_flash_kda, "AITER_FDA_USE_GLUON_K2", True)
    monkeypatch.setattr(_flash_kda, "_K2C_ROUTE_BY_SIZE", True)
    args = make_inputs(B, 1024 if chunks_per_seg else 64, H)
    with (
        _count_launches(_gluon_k2_module(), ["k2_c_fast"]) as gluon_calls,
        _count_launches(_flash_kda, ["_segment_fast"]) as triton_calls,
    ):
        run_flash(*args, chunks_per_seg=chunks_per_seg)
    assert gluon_calls["k2_c_fast"] == int(gluon)
    assert triton_calls["_segment_fast"] == int(not gluon)


def test_segment_length_cap(monkeypatch):
    """A hard cap bounds every segment; the fallback's soft cap yields to the
    segment-count cap on long sequences."""
    monkeypatch.delenv("CHUNK_DELTA_ATTN_FLASH_KDA_SEG", raising=False)
    choose = _flash_kda._choose_chunks_per_seg
    long_seq = 64 * _flash_kda._SEG_MAX_SEGMENTS * 32
    monkeypatch.setattr(_flash_kda, "_seg_limits", lambda: (64, True))
    for n_chunks in (64, 512, 4096, long_seq):
        for H in (1, 4, 12):
            assert choose(n_chunks, 1, H, K_DIM) <= 64, (n_chunks, H)
    monkeypatch.setattr(_flash_kda, "_seg_limits", lambda: (32, False))
    assert choose(long_seq, 1, 1, K_DIM) == long_seq // _flash_kda._SEG_MAX_SEGMENTS


# A weak gate is the only setting that exposes the intra-chunk inverse. At the
# Kimi default lower_bound=-5 the decay damps L to almost nothing, so even an
# outright wrong inverse stays inside bf16 noise; these three catch it.
@pytest.mark.parametrize("lower_bound", [-1.0, -0.1, -0.01])
@pytest.mark.parametrize("chunks_per_seg", [0, 3])
def test_weak_gate(lower_bound, chunks_per_seg):
    args = make_inputs(1, 512, 4)
    o_ref, _ = run_reference(*args, lower_bound=lower_bound)
    o_fkda, _ = run_flash(*args, lower_bound=lower_bound, chunks_per_seg=chunks_per_seg)
    assert rel_err(o_fkda, o_ref) < 2e-2


# Gaussian keys in 128 dimensions are near-orthogonal, so L's off-diagonal
# entries sit at ~0.02 and every inverse of it looks accurate -- which is the
# regime every other case here feeds. Repeated tokens in real text correlate the
# keys instead, and with beta near 1 and a gate that barely decays L's entries
# approach 1. The powers the inverse is built from then run to 5e7 across a
# 32-wide chunk before cancelling back to a result bounded by 1, which no
# storage precision survives: this case reached the model as 0.2% on gsm8k while
# the rest of this file passed.
@pytest.mark.parametrize("rho", [0.9, 1.0])
def test_correlated_keys_weak_gate(rho):
    q, k, v, g, beta, A_log, dt_bias, scale = make_inputs(1, 512, 4)
    # rho=1 is a token repeated verbatim for the whole sequence.
    k = (rho * k[:, :1] + (1.0 - rho) * k).to(dtype)
    beta = torch.full_like(beta, 4.0)  # sigmoid(4) = 0.98, so L is barely shrunk
    args = (q, k, v, g, beta, A_log, dt_bias, scale)
    kw = {"lower_bound": -0.01}
    gold, _ = fp32_gold(*args, **kw)
    e_fkda = rel_err(run_flash(*args, **kw)[0], gold)
    e_ref = rel_err(run_reference(*args, **kw)[0], gold)
    assert e_fkda < 5e-2, f"flash_kda {e_fkda:.2e} against the fp32 recurrence"
    assert e_fkda < 1.3 * e_ref, f"flash_kda {e_fkda:.2e} vs default {e_ref:.2e}"


@pytest.mark.parametrize("lower_bound", [-5.0, -1.0, -0.01])
def test_matches_fp32_recurrence(lower_bound):
    """Against the token-at-a-time fp32 recurrence both paths blockify.

    The default pipeline is the reference everywhere else here, which cannot
    show the two paths drifting the same way; the absolute bound catches that.
    The second assertion keeps a merely noisier fused path from passing it, and
    covers a gap in the default path's own reference check, which pins
    chunk_size=64 -- a size flash_kda rejects.
    """
    args = make_inputs(1, 512, 4)
    gold, _ = fp32_gold(*args, lower_bound=lower_bound)
    e_fkda = rel_err(run_flash(*args, lower_bound=lower_bound)[0], gold)
    e_ref = rel_err(run_reference(*args, lower_bound=lower_bound)[0], gold)
    assert e_fkda < 2e-2, f"flash_kda {e_fkda:.2e} against the fp32 recurrence"
    assert e_fkda < 1.3 * e_ref, f"flash_kda {e_fkda:.2e} vs default {e_ref:.2e}"


@pytest.mark.parametrize("lower_bound", [-5.0, -1.0, -0.01])
def test_final_state_matches_fp32_recurrence(lower_bound):
    """The same absolute bound as above, on the state rather than the output.

    Decode continues the recurrence from this tensor, so a state that drifts the
    same way in both chunked paths passes every default-pipeline comparison here
    and still ruins generation. varlen with a V-first state is what the serving
    path asks for, and it is the combination the checks above split apart.
    """
    lens = [200, 100, 156]
    cu_seqlens = torch.tensor(
        [0] + list(torch.tensor(lens, device=device).cumsum(0)),
        device=device,
        dtype=torch.long,
    )
    args = make_inputs(1, sum(lens), 4)
    kw = {
        "cu_seqlens": cu_seqlens,
        "output_final_state": True,
        "state_v_first": True,
        "lower_bound": lower_bound,
    }
    _, gold = fp32_gold(*args, **kw)
    e_fkda = rel_err(run_flash(*args, **kw)[1], gold)
    e_ref = rel_err(run_reference(*args, **kw)[1], gold)
    assert e_fkda < 2e-2, f"flash_kda state {e_fkda:.2e} against the fp32 recurrence"
    assert e_fkda < 1.3 * e_ref, f"flash_kda state {e_fkda:.2e} vs default {e_ref:.2e}"


def test_supported_rejects_unsupported():
    q = torch.randn(1, 64, 4, 64, device=device, dtype=dtype)
    v = torch.randn(1, 64, 4, 64, device=device, dtype=dtype)
    A_log = torch.randn(4, device=device, dtype=torch.float32)
    base = {
        "chunk_size": FLASH_KDA_CHUNK,
        "safe_gate": True,
        "use_gate_in_kernel": True,
        "use_qk_l2norm_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True,
        "lower_bound": LOWER_BOUND,
        "A_log": A_log,
    }
    # K = V = 64 is not supported.
    assert not flash_kda_supported(q=q, v=v, **base)

    q128 = torch.randn(1, 64, 4, 128, device=device, dtype=dtype)
    v128 = torch.randn(1, 64, 4, 128, device=device, dtype=dtype)
    assert flash_kda_supported(q=q128, v=v128, **base)
    assert not flash_kda_supported(q=q128, v=v128, **{**base, "chunk_size": 64})
    assert not flash_kda_supported(q=q128, v=v128, **{**base, "lower_bound": None})
    assert not flash_kda_supported(q=q128, v=v128, **{**base, "safe_gate": False})
    assert not flash_kda_supported(
        q=q128, v=v128, **{**base, "use_gate_in_kernel": False}
    )
    # GVA (HV != H) is not supported.
    v_gva = torch.randn(1, 64, 8, 128, device=device, dtype=dtype)
    assert not flash_kda_supported(q=q128, v=v_gva, **base)


def test_public_wrapper_routes_to_flash_kda(monkeypatch):
    """``chunk_kimi_delta_attn`` must actually reach the fast path.

    Every other test here drives the internal functions, so nothing would catch
    the public wrapper pinning an argument that disqualifies the dispatch: the
    fast path would silently stop being used and the suite would stay green.
    """
    q, k, v, g, beta, A_log, dt_bias, scale = make_inputs(1, 512, 4)
    kwargs = {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "scale": scale,
        "chunk_size": FLASH_KDA_CHUNK,
        "safe_gate": True,
        "lower_bound": LOWER_BOUND,
        "use_gate_in_kernel": True,
        "use_qk_l2norm_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True,
        "output_final_state": True,
    }

    calls = []
    real = _chunk_fwd.flash_kda_fwd

    def counting(**kw):
        calls.append(1)
        return real(**kw)

    monkeypatch.setattr(_chunk_fwd, "flash_kda_fwd", counting)

    monkeypatch.setattr(_chunk_fwd, "AITER_FDA_ENABLE", False)
    o_ref, ht_ref = chunk_kimi_delta_attn(**kwargs)
    assert not calls, "the flag is off, the default pipeline should have served this"

    monkeypatch.setattr(_chunk_fwd, "AITER_FDA_ENABLE", True)
    o_fkda, ht_fkda = chunk_kimi_delta_attn(**kwargs)
    assert len(calls) == 1, "the wrapper never reached flash_kda_fwd"

    chunk_kimi_delta_attn(**{**kwargs, "safe_gate": False})
    assert len(calls) == 1, "safe_gate=False must stay on the default intra kernel"

    assert rel_err(o_fkda, o_ref) < 2e-2
    assert rel_err(ht_fkda, ht_ref) < 2e-2


def test_unset_chunk_size_follows_the_dispatch(monkeypatch):
    """``chunk_size=None`` has to resolve the way the dispatch goes.

    Picking 32 without reaching flash_kda would leave the call on the default
    pipeline at 32, which is the slowest of the three configurations rather
    than the fastest, and nothing about the result would reveal it.
    """
    q, k, v, g, beta, A_log, dt_bias, scale = make_inputs(1, 512, 4)
    kwargs = {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "scale": scale,
        "safe_gate": True,
        "lower_bound": LOWER_BOUND,
        "use_gate_in_kernel": True,
        "use_qk_l2norm_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True,
    }

    calls = []
    real = _chunk_fwd.flash_kda_fwd

    def counting(**kw):
        calls.append(1)
        return real(**kw)

    monkeypatch.setattr(_chunk_fwd, "flash_kda_fwd", counting)
    monkeypatch.setattr(_chunk_fwd, "AITER_FDA_ENABLE", True)

    chunk_kimi_delta_attn(chunk_size=None, **kwargs)
    assert len(calls) == 1, "an eligible call should have resolved to FLASH_KDA_CHUNK"

    # Ineligible, so it must fall back to 64 rather than to FLASH_KDA_CHUNK.
    # Same pipeline and same chunk size means the two agree bit for bit.
    o_auto, _ = chunk_kimi_delta_attn(chunk_size=None, **{**kwargs, "safe_gate": False})
    o_64, _ = chunk_kimi_delta_attn(chunk_size=64, **{**kwargs, "safe_gate": False})
    assert len(calls) == 1
    assert torch.equal(o_auto, o_64), "an ineligible call should have resolved to 64"


def _paged_pool(n, H, initial, pad_floats=0):
    """V-first paged cache. ``pad_floats`` extra between slots."""
    V = K = K_DIM
    inner = H * V * K
    slot_stride = inner + pad_floats
    storage = torch.zeros((n + 2) * slot_stride, device=device, dtype=torch.float32)
    cache = torch.as_strided(
        storage,
        size=(n + 2, H, V, K),
        stride=(slot_stride, V * K, K, 1),
    )
    indices = torch.arange(n, 0, -1, device=device, dtype=torch.int32)
    cache[indices] = initial
    return cache, indices, storage


def _kimi_kwargs(q, k, v, g, beta, A_log, dt_bias, scale, **kw):
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "scale": scale,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True,
        "safe_gate": True,
        "lower_bound": LOWER_BOUND,
        "state_v_first": True,
        **kw,
    }


@pytest.mark.parametrize("gluon_k2", [False, True])
@pytest.mark.parametrize("T,chunks_per_seg", [(256, 0), (1024, 4)])
def test_paged_cache_matches_dense(T, chunks_per_seg, gluon_k2):
    """In-kernel paged I/O matches gather into a dense V-first state, on both
    K2 routes."""
    if gluon_k2 and not _flash_kda._gluon_k2_usable(FLASH_KDA_CHUNK, K_DIM, K_DIM):
        pytest.skip("this arch or tile shape never routes K2 to Gluon")
    H = 4
    q, k, v, g, beta, A_log, dt_bias, scale = make_inputs(1, T, H)
    h0 = torch.randn(1, H, K_DIM, K_DIM, device=device, dtype=torch.float32) * 0.1
    common = {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "scale": scale,
        "lower_bound": LOWER_BOUND,
        "state_v_first": True,
        "chunks_per_seg": chunks_per_seg,
    }
    cache, indices, _storage = _paged_pool(1, H, h0)
    ptr_before = cache.data_ptr()
    out = torch.empty_like(v)
    paged_kw = {
        **common,
        "initial_state": None,
        "output_final_state": False,
        "out": out,
        "state_cache": cache,
        "state_indices": indices,
        "has_initial_state": torch.ones(1, device=device, dtype=torch.bool),
    }
    with _route(k2=gluon_k2):
        o_dense, ht = flash_kda_fwd(**common, initial_state=h0, output_final_state=True)
        # First paged launch autotunes a new PAGED_CACHE specialization; compare
        # after that, not against a trial config.
        flash_kda_fwd(**paged_kw)
        cache[indices] = h0
        o_paged, ht_paged = flash_kda_fwd(**paged_kw)
    assert ht_paged is None
    assert o_paged.data_ptr() == out.data_ptr()
    assert cache.data_ptr() == ptr_before, "paged cache must not be packed/copied"
    assert torch.equal(o_paged, o_dense)
    assert torch.equal(cache[indices], ht)
    assert torch.count_nonzero(cache[[0, -1]]).item() == 0


@pytest.mark.parametrize("seg", ["0", "4"])
def test_paged_cache_has_initial_state_false(seg, monkeypatch):
    """A zero-start slot is never read: its NaN contents must not leak.

    ``seg="4"`` cuts the 8-chunk sequence into two segments, so the load under
    test is the segment kernel's incoming-state load as well as the scan's.
    """
    monkeypatch.setenv("CHUNK_DELTA_ATTN_FLASH_KDA_SEG", seg)
    H = 4
    args = make_inputs(1, 256, H)
    v = args[2]
    dirty = torch.full(
        (1, H, K_DIM, K_DIM), float("nan"), device=device, dtype=torch.float32
    )
    o_zero, ht_zero = chunk_kimi_delta_attn(
        **_kimi_kwargs(
            *args,
            initial_state=torch.zeros_like(dirty),
            output_final_state=True,
        )
    )
    cache, indices, _storage = _paged_pool(1, H, dirty)
    out = torch.empty_like(v)
    paged_kw = _kimi_kwargs(
        *args,
        out=out,
        state_cache=cache,
        state_indices=indices,
        has_initial_state=torch.zeros(1, device=device, dtype=torch.bool),
    )
    # Warm the paged specialization's autotune, then re-dirty the slot.
    chunk_kimi_delta_attn(**paged_kw)
    cache[indices] = dirty
    o_paged, _ = chunk_kimi_delta_attn(**paged_kw)
    assert torch.equal(o_paged, o_zero)
    assert torch.equal(cache[indices], ht_zero)


def test_paged_cache_padded_stride_writes_live_pool():
    """stride(0) padding must not trigger a packed copy."""
    H = 4
    args = make_inputs(1, 256, H)
    v = args[2]
    h0 = torch.randn(1, H, K_DIM, K_DIM, device=device, dtype=torch.float32) * 0.1
    _, ht = chunk_kimi_delta_attn(
        **_kimi_kwargs(*args, initial_state=h0, output_final_state=True)
    )
    cache, indices, storage = _paged_pool(1, H, h0, pad_floats=128)
    assert not cache.is_contiguous()
    ptr = storage.data_ptr()
    out = torch.empty_like(v)
    chunk_kimi_delta_attn(
        **_kimi_kwargs(
            *args,
            out=out,
            state_cache=cache,
            state_indices=indices,
            has_initial_state=torch.ones(1, device=device, dtype=torch.bool),
        )
    )
    assert storage.data_ptr() == ptr
    assert torch.equal(cache[indices], ht)
    # Padding between slots and the unused first/last rows stay zero.
    inner = H * K_DIM * K_DIM
    slot_stride = inner + 128
    for slot in (0, int(indices.item()) + 1):
        row = storage[slot * slot_stride : (slot + 1) * slot_stride]
        assert torch.count_nonzero(row).item() == 0


@pytest.mark.parametrize("outer", ["expanded", "overlapping"])
def test_paged_cache_rejects_overlapping_slots(outer):
    """stride(0) below one [H, V, K] plane would alias neighbouring slots."""
    H = 4
    args = make_inputs(1, 128, H)
    v = args[2]
    plane = H * K_DIM * K_DIM
    if outer == "expanded":
        cache = torch.zeros(
            1, H, K_DIM, K_DIM, device=device, dtype=torch.float32
        ).expand(3, -1, -1, -1)
    else:
        storage = torch.zeros(3 * plane, device=device, dtype=torch.float32)
        cache = torch.as_strided(
            storage,
            size=(3, H, K_DIM, K_DIM),
            stride=(plane // 2, K_DIM * K_DIM, K_DIM, 1),
        )
    assert cache.stride(0) < plane
    with pytest.raises(ValueError, match="not dense"):
        chunk_kimi_delta_attn(
            **_kimi_kwargs(
                *args,
                out=torch.empty_like(v),
                state_cache=cache,
                state_indices=torch.tensor([1], device=device, dtype=torch.int32),
                has_initial_state=torch.ones(1, device=device, dtype=torch.bool),
            )
        )


def test_out_rejected_on_default_pipeline():
    """The default pipeline cannot honour `out`, so it must not drop it."""
    args = make_inputs(1, 128, 4)
    v = args[2]
    with (
        _force_default_pipeline(),
        pytest.raises(ValueError, match="out is only implemented"),
    ):
        chunk_kimi_delta_attn(**_kimi_kwargs(*args, out=torch.empty_like(v)))


def test_out_must_be_contiguous():
    """A strided `out` is rejected, not silently packed into a clone."""
    args = make_inputs(1, 128, 4)
    v = args[2]
    B, T, H, V = v.shape
    strided = torch.empty(B, T, H, 2 * V, device=device, dtype=v.dtype)[..., :V]
    assert not strided.is_contiguous()
    with pytest.raises(ValueError, match="out must be contiguous"):
        chunk_kimi_delta_attn(**_kimi_kwargs(*args, out=strided))


def test_out_is_written_in_place():
    """A contiguous `out` is the buffer the kernel writes, not a packed copy."""
    args = make_inputs(1, 128, 4)
    ref, _ = chunk_kimi_delta_attn(**_kimi_kwargs(*args))
    out = torch.full_like(args[2], float("nan"))
    o, _ = chunk_kimi_delta_attn(**_kimi_kwargs(*args, out=out))
    assert o.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_paged_cache_rejected_on_default_pipeline():
    args = make_inputs(1, 128, 4)
    v = args[2]
    h0 = torch.zeros(1, 4, K_DIM, K_DIM, device=device, dtype=torch.float32)
    cache, indices, _ = _paged_pool(1, 4, h0)
    with (
        _force_default_pipeline(),
        pytest.raises(ValueError, match="paged state_cache is only implemented"),
    ):
        chunk_kimi_delta_attn(
            **_kimi_kwargs(
                *args,
                out=torch.empty_like(v),
                state_cache=cache,
                state_indices=indices,
                has_initial_state=torch.ones(1, device=device, dtype=torch.bool),
            )
        )


def _bands(*xs, skew=0):
    """``xs`` as the bands of one fused tensor, as a fused input projection
    gives them. ``skew`` leading pad elements break the 16-byte alignment."""
    shape = xs[0].shape[-2:]
    pad = [xs[0].new_zeros(*xs[0].shape[:2], skew)] if skew else []
    fused = torch.cat(pad + [x.flatten(-2) for x in xs], dim=-1)[..., skew:]
    return tuple(b.unflatten(-1, shape) for b in fused.split(shape.numel(), dim=-1))


def _head_major(*xs):
    """``xs`` stored head by head, so no token's heads are adjacent."""
    return tuple(x.transpose(1, 2).contiguous().transpose(1, 2) for x in xs)


def _record_tdm_operands(monkeypatch):
    """Have the gfx1250 Gluon launchers note the storage they are handed for
    each operand they read by TDM, as ``{(launcher, argument): {data_ptr}}``."""
    g1, g2 = (
        importlib.import_module(
            f"aiter.ops.triton._gluon_kernels.gfx1250.kimi_delta_attn.{name}"
        )
        for name in ("flash_kda_k1", "flash_kda_k2")
    )
    seen = {}

    class _Recording:
        def __init__(self, real, name, operands):
            self.real, self.name, self.operands = real, name, operands

        def __getitem__(self, grid):
            launch = self.real[grid]

            def record(**kw):
                for op in self.operands:
                    seen.setdefault((self.name, op), set()).add(kw[op].data_ptr())
                return launch(**kw)

            return record

    for module, name, operands in (
        (g1, "_k1_fast", ("q", "k", "g_raw")),
        (g2, "k2_ab_fused_fast", ("v_input",)),
        (g2, "k2_c_fast", ("v_input",)),
    ):
        real = getattr(module, name)
        monkeypatch.setattr(module, name, _Recording(real, name, operands))
    return seen


_LAYOUTS = {
    "bands": _bands,
    "skewed bands": lambda *xs: _bands(*xs, skew=1),
    "batched bands": _bands,
    "head-major": _head_major,
}


@pytest.mark.parametrize("chunks_per_seg", [0, 4])
@pytest.mark.parametrize("layout", list(_LAYOUTS))
def test_strided_operands_match_packed(layout, chunks_per_seg, monkeypatch):
    """Strided q, k, v and g give the packed answer, to the bit."""
    if layout == "batched bands":
        args, kw = make_inputs(2, 512, 4), {}
    else:
        args, kw = make_inputs(1, 1024, 4), {"cu_seqlens": _varlen([300, 200, 524])}
    kw.update(
        output_final_state=True, state_v_first=True, chunks_per_seg=chunks_per_seg
    )
    views = _LAYOUTS[layout](*args[:4])
    assert not any(x.is_contiguous() for x in views)

    on_tdm = _flash_kda.DEVICE_ARCH == "gfx1250"
    with _route(k1=True, k2=True):
        o_ref, ht_ref = run_flash(*args, **kw)
        seen = _record_tdm_operands(monkeypatch) if on_tdm else None
        o, ht = run_flash(*views, *args[4:], **kw)
    assert torch.equal(o, o_ref)
    assert torch.equal(ht, ht_ref)
    if not on_tdm:
        return

    q, k, v, g = views
    handed = {
        ("_k1_fast", "q"): q,
        ("_k1_fast", "k"): k,
        ("_k1_fast", "g_raw"): g,
        ("k2_c_fast", "v_input"): v,
    }
    if chunks_per_seg:
        handed[("k2_ab_fused_fast", "v_input")] = v
    assert seen.keys() == handed.keys()
    for key, view in handed.items():
        in_place = seen[key] == {view.data_ptr()}
        assert in_place == (layout in ("bands", "skewed bands")), key


@pytest.mark.parametrize("pipeline", ["flash_kda", "default"])
def test_public_wrapper_passes_projection_bands(pipeline, monkeypatch):
    """``chunk_kimi_delta_attn`` passes q, k, v and g on unpacked, and both
    pipelines give the packed answer from bands."""
    args = make_inputs(1, 1024, 4)
    bands = _bands(*args[:4])
    extra = {"cu_seqlens": _varlen([300, 200, 524]), "output_final_state": True}
    monkeypatch.setattr(_chunk_fwd, "AITER_FDA_ENABLE", pipeline == "flash_kda")
    o_ref, ht_ref = chunk_kimi_delta_attn(**_kimi_kwargs(*args, **extra))

    received = []
    real = _chunk_fwd.flash_kda_fwd

    def recording(**kw):
        received.append([kw[name].data_ptr() for name in "qkvg"])
        return real(**kw)

    monkeypatch.setattr(_chunk_fwd, "flash_kda_fwd", recording)
    o, ht = chunk_kimi_delta_attn(**_kimi_kwargs(*bands, *args[4:], **extra))
    assert torch.equal(o, o_ref)
    assert torch.equal(ht, ht_ref)
    want = [[b.data_ptr() for b in bands]] if pipeline == "flash_kda" else []
    assert received == want


def test_input_guard_skips_by_position_too():
    """A name in ``skip`` keeps its storage whether it is passed by keyword or
    by position, and every other tensor is packed either way."""

    @input_guard(skip=("kept",))
    def f(packed, kept):
        return packed, kept

    base = torch.randn(4, 8, device=device)
    packed, kept = base[:, ::2], base[:, 1::2]
    for got_packed, got_kept in (f(packed, kept), f(packed=packed, kept=kept)):
        assert got_packed.is_contiguous()
        assert got_kept.data_ptr() == kept.data_ptr()
        assert not got_kept.is_contiguous()


def test_positional_call_keeps_skipped_storage(monkeypatch):
    """Passed by position, a padded ``state_cache`` still gets the final state
    in place, and on gfx1250 the bands are still read in place."""
    H = 4
    q, k, v, g, beta, A_log, dt_bias, scale = make_inputs(1, 256, H)
    h0 = torch.randn(1, H, K_DIM, K_DIM, device=device, dtype=torch.float32) * 0.1
    o_ref, ht_ref = run_flash(
        q, k, v, g, beta, A_log, dt_bias, scale,
        initial_state=h0, output_final_state=True, state_v_first=True,
        chunks_per_seg=0,
    )  # fmt: skip
    cache, indices, storage = _paged_pool(1, H, h0, pad_floats=128)
    ptr = storage.data_ptr()
    bands = _bands(q, k, v, g)
    out = torch.empty_like(v)
    by_name = {
        **dict(zip("qkvg", bands)),
        "beta": beta,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "scale": scale,
        "lower_bound": LOWER_BOUND,
        "initial_state": None,
        "output_final_state": False,
        "state_v_first": True,
        "cu_seqlens": None,
        "chunk_indices": None,
        "chunks_per_seg": 0,
        "out": out,
        "state_cache": cache,
        "state_indices": indices,
        "has_initial_state": torch.ones(1, device=device, dtype=torch.bool),
    }
    params = inspect.signature(flash_kda_fwd).parameters
    assert by_name.keys() == params.keys()

    on_tdm = _flash_kda.DEVICE_ARCH == "gfx1250"
    seen = _record_tdm_operands(monkeypatch) if on_tdm else None
    o, ht = flash_kda_fwd(*(by_name[name] for name in params))
    assert ht is None
    assert o.data_ptr() == out.data_ptr()
    assert torch.equal(o, o_ref)
    assert storage.data_ptr() == ptr
    assert torch.equal(cache[indices], ht_ref), "final state went to a packed clone"
    if on_tdm:
        q, k, v, g = bands
        handed = {
            ("_k1_fast", "q"): q,
            ("_k1_fast", "k"): k,
            ("_k1_fast", "g_raw"): g,
            ("k2_c_fast", "v_input"): v,
        }
        assert seen.keys() == handed.keys()
        for key, band in handed.items():
            assert seen[key] == {band.data_ptr()}, key
