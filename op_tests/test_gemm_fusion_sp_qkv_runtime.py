# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tests for the native runtime of the neutral ``sp_qkv_fused`` op.

No GPU, no torch tensors, no MORI: the payload is built from lightweight fake
tensors and the environment seams of ``aiter.ops.gemm_fusion.sp_qkv.runtime``
(resources, profile, plan matching, capture predicate, capability detection) are
replaced with fakes.  What is covered here is the runtime policy a GPU cannot
easily test on a CPU box:

* payload validation (structure vs tensor contract);
* coverage-limited native errors (no plan / one-stage band / legacy pad plan)
  and the static ``sp_qkv_covers`` gate (no resource/session/allocation);
* session cache keying and ``open_session``/``close_session`` lifecycle;
* prepare/bind outside graph capture, replay afterwards;
* the per-process call-id counter (rank-uniform sequence, 18-bit mask);
* profile selection through ``AITER_GEMM_FUSION_PROFILE``;
* the ERR-page readback -> typed ``GemmFusionDeviceError``;
* the optional post-prologue native-stage callback;
* lazy imports (importing the module must not touch MORI/the framework).
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

from aiter.ops.gemm_fusion.sp_qkv import runtime as rt

SP_QKV_DIR = Path(__file__).resolve().parents[1] / "aiter" / "ops" / "gemm_fusion"
REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeContractError(RuntimeError):
    """Stand-in for ``aiter.gemm_fusion.match.ContractError``."""


class FakeTensor:
    """Duck-typed tensor: the runtime only reads dtype/shape/stride/pointers."""

    _next_ptr = 1000

    def __init__(self, shape, dtype, device="cuda:0", ptr=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self._ptr = ptr if ptr is not None else FakeTensor.alloc_ptr()
        self._strides = _contiguous_strides(self.shape)

    @staticmethod
    def alloc_ptr():
        FakeTensor._next_ptr += 16
        return FakeTensor._next_ptr

    def data_ptr(self):
        return self._ptr

    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    def stride(self, dim=None):
        if dim is None:
            return self._strides
        return self._strides[dim]

    def is_contiguous(self):
        return True


def _contiguous_strides(shape):
    strides, acc = [], 1
    for d in reversed(shape):
        strides.append(acc)
        acc *= d
    return tuple(reversed(strides))


class FakeResources:
    def __init__(self):
        self.finalized = 0
        self.released = 0
        self.enqueued = []
        self.words = {}
        self.bound = self

    def finalize(self):
        self.finalized += 1

    def release(self):
        self.released += 1

    def readback_enqueue(self, step_idx, stream=None):
        self.enqueued.append(step_idx)

    def readback_consume(self, step_idx):
        return self.words.get(step_idx)


class FakePlan:
    abi = 3
    pad_m = False
    vid = "sp_qkv.small.v1"
    n_stages = 2

    def __init__(self):
        self.prepares = []
        self.runs = []
        self.owner = "testowner"

    def prepare(self, **kwargs):
        self.prepares.append(kwargs)

    def run(self, **kwargs):
        self.runs.append(kwargs)


def _payload(mode="dense", m=8, group=None, n_pages=2):
    t = 4 * m
    p = {name: None for name in rt.PAYLOAD_KEYS}
    p.update(
        q=FakeTensor((t, 16, 128), "bfloat16"),
        x=FakeTensor((m, rt._HIDDEN_K), "float8_e4m3fn"),
        x_scale=FakeTensor((m, 1), "float32"),
        w=FakeTensor(
            (9216 if mode == "dense" else 9856, rt._HIDDEN_K), "float8_e4m3fn"
        ),
        w_scale=FakeTensor((9216 if mode == "dense" else 9856,), "float32"),
        q_norm_w=FakeTensor((128,), "float32"),
        k_norm_w=FakeTensor((128,), "float32"),
        cos_cache=FakeTensor((4096, 32), "bfloat16"),
        sin_cache=FakeTensor((4096, 32), "bfloat16"),
        positions=FakeTensor((t,), "int64"),
        slot_mapping=FakeTensor((t,), "int64"),
        k_cache=FakeTensor((n_pages, 1, 8, rt._KV_PAGE[mode], 16), "float8_e4m3fn"),
        v_cache=FakeTensor((n_pages, 1, rt._KV_PAGE[mode], 128, 16), "float8_e4m3fn"),
        k_scale=FakeTensor((n_pages, 1, rt._KV_PAGE[mode]), "float32"),
        v_scale=FakeTensor((n_pages, 1, rt._KV_PAGE[mode]), "float32"),
        mode=mode,
        eps=1e-6,
        group=group,
        stream=object(),
    )
    if mode == "indexer":
        p.update(
            index_q=FakeTensor((t, 1, 128), "bfloat16"),
            index_q_norm_w=FakeTensor((128,), "float32"),
            index_k_norm_w=FakeTensor((128,), "float32"),
            index_cache=FakeTensor((16, 128, 128), "float8_e4m3fn"),
        )
    else:
        p.update(
            k=FakeTensor((t, 1, 128), "bfloat16"),
            v=FakeTensor((t, 1, 128), "bfloat16"),
        )
    return p


class Env:
    """Monkeypatched runtime seams for one test."""

    def __init__(self, monkeypatch):
        self.monkeypatch = monkeypatch
        self.capturing = False
        self.profile_names = []
        self.resources = []
        self.plan = FakePlan()
        self.n_stages = 2
        self.copies = []
        self.bucket = None  # None => bucket == payload M
        self.plan_factory = lambda spec, ctx, profile: (
            self.plan,
            self.n_stages,
            ctx["m"] if self.bucket is None else self.bucket,
        )
        self.gf = types.SimpleNamespace(
            MatchCtx=lambda **kwargs: dict(kwargs),
            ContractError=FakeContractError,
        )
        monkeypatch.setattr(rt, "_gf", lambda: self.gf)
        monkeypatch.setattr(rt, "_contract_error", lambda: FakeContractError)
        monkeypatch.setattr(rt, "_world_rank", lambda group: (0, 4))
        monkeypatch.setattr(rt, "_detect_arch", lambda: "gfx950")
        monkeypatch.setattr(rt, "_detect_cu", lambda: 256)
        monkeypatch.setattr(
            rt, "_detect_capabilities", lambda: frozenset({"mori_cco", "sdma"})
        )
        monkeypatch.setattr(rt, "_load_profile", self._load_profile)
        monkeypatch.setattr(rt, "_match_padded", self._match_padded)
        monkeypatch.setattr(rt, "_create_resources", self._create_resources)
        monkeypatch.setattr(rt, "_is_capturing", lambda: self.capturing)
        monkeypatch.setattr(rt, "_default_device", lambda: 0)
        monkeypatch.setattr(rt, "_new_tensor", self._new_tensor)
        monkeypatch.setattr(rt, "_copy_rows", self._copy_rows)
        rt._reset_for_testing()

    def _load_profile(self, name):
        self.profile_names.append(name)
        return types.SimpleNamespace(name=name)

    def _match_padded(self, spec, ctx, profile):
        return self.plan_factory(spec, ctx, profile)

    def _create_resources(self, rank, world, group, device, caps, profile):
        res = FakeResources()
        self.resources.append(res)
        return res

    def _new_tensor(self, shape, dtype_name, device):
        return FakeTensor(shape, dtype_name, device=device)

    def _copy_rows(self, dst, src, rows):
        self.copies.append((dst, src, rows))

    def session(self):
        return next(iter(rt._SESSIONS.values()))


@pytest.fixture
def env(monkeypatch):
    e = Env(monkeypatch)
    yield e
    rt._reset_for_testing()


class CoversEnv:
    """Real planner (profile + match_padded) but no resource/session/allocation."""

    def __init__(self, monkeypatch):
        self.created = []
        self.allocated = []
        self.cache_touched = []
        monkeypatch.setattr(rt, "_world_rank", lambda group: (0, 4))
        monkeypatch.setattr(rt, "_detect_arch", lambda: "gfx950")
        monkeypatch.setattr(rt, "_detect_cu", lambda: 256)
        monkeypatch.setattr(
            rt,
            "_detect_capabilities",
            lambda: frozenset({"mori_cco", "sdma", "gfx950_sc_bypass"}),
        )
        monkeypatch.setattr(
            rt, "_create_resources", lambda *args, **kwargs: self.created.append(args)
        )
        monkeypatch.setattr(rt, "_new_tensor", self._new_tensor)
        monkeypatch.setattr(
            rt, "_get_session", lambda *args, **kwargs: self.cache_touched.append(args)
        )
        monkeypatch.delenv(rt.PROFILE_ENV, raising=False)
        rt._reset_for_testing()

    def _new_tensor(self, shape, dtype_name, device=None):
        self.allocated.append((shape, dtype_name))
        return FakeTensor(shape, dtype_name, device=device)


@pytest.fixture
def covers_env(monkeypatch):
    e = CoversEnv(monkeypatch)
    yield e
    rt._reset_for_testing()


# ---------------------------------------------------------------------------
# payload validation
# ---------------------------------------------------------------------------
def test_payload_validation_structural_errors(env):
    payload = _payload()
    payload["unknown"] = 1
    with pytest.raises(ValueError, match="unknown keys"):
        rt.sp_qkv_native(payload)
    del payload["unknown"]
    payload["mode"] = "bogus"
    with pytest.raises(ValueError, match="mode"):
        rt.sp_qkv_native(payload)
    del payload["q"]
    with pytest.raises(ValueError, match="missing"):
        rt.sp_qkv_native(payload)


def test_payload_validation_tensor_contract_errors(env):
    payload = _payload()
    payload["x"] = FakeTensor((8, 4096), "float8_e4m3fn")
    with pytest.raises(FakeContractError, match="K="):
        rt.sp_qkv_native(payload)
    payload = _payload()
    payload["q"] = FakeTensor((4 * 8, 16, 64), "bfloat16")
    with pytest.raises(FakeContractError, match="q"):
        rt.sp_qkv_native(payload)
    payload = _payload()
    payload["positions"] = FakeTensor((0,), "int64")
    with pytest.raises(FakeContractError, match="no entries"):
        rt.sp_qkv_native(payload)
    payload = _payload("dense")
    payload["k_cache"] = FakeTensor((2, 1, 8, 16, 16), "float8_e4m3fn")
    with pytest.raises(FakeContractError, match="page"):
        rt.sp_qkv_native(payload)


def test_indexer_payload_requires_index_tensors(env):
    payload = _payload("indexer")
    payload["index_cache"] = None
    with pytest.raises(FakeContractError, match="index_cache"):
        rt.sp_qkv_native(payload)


# ---------------------------------------------------------------------------
# coverage-limited native path
# ---------------------------------------------------------------------------
def test_coverage_error_when_no_plan(env):
    env.n_stages = 0
    env.plan_factory = lambda spec, ctx, profile: (None, 0, None)
    with pytest.raises(FakeContractError) as exc:
        rt.sp_qkv_native(_payload(m=600))
    msg = str(exc.value)
    assert "coverage-limited" in msg
    assert "no fused prologue plan" in msg
    assert "mode='dense'" in msg
    assert "local M=600" in msg
    assert "no unfused fallback" in msg


def test_coverage_error_for_one_stage_plan(env):
    env.n_stages = 1
    env.plan = FakePlan()
    env.plan.vid = "sp_qkv.large.v1"
    env.plan.n_stages = 1
    with pytest.raises(FakeContractError, match="Phase-2"):
        rt.sp_qkv_native(_payload())


# ---------------------------------------------------------------------------
# static coverage query (sp_qkv_covers)
# ---------------------------------------------------------------------------
def test_sp_qkv_covers_prologue_cells(covers_env):
    for mode in rt.MODES:
        for m in (1, 2, 16, 128, 256):
            assert rt.sp_qkv_covers(_payload(mode, m)) is True, (mode, m)
    # a static query: no session/resource/allocation, and no session-cache touch
    assert covers_env.created == []
    assert covers_env.allocated == []
    assert covers_env.cache_touched == []
    assert rt._SESSIONS == {}


def test_sp_qkv_covers_one_stage_band_is_uncovered(covers_env):
    for m in (512, 600, 1024, 8192, 9216):
        assert rt.sp_qkv_covers(_payload("dense", m)) is False, m
    assert covers_env.created == []
    assert covers_env.allocated == []
    assert covers_env.cache_touched == []
    assert rt._SESSIONS == {}


def test_sp_qkv_covers_never_raises(covers_env, monkeypatch):
    assert rt.sp_qkv_covers({}) is False
    assert rt.sp_qkv_covers(None) is False  # type: ignore[arg-type]
    bad = _payload()
    bad["x"] = FakeTensor((8, 4096), "float8_e4m3fn")
    assert rt.sp_qkv_covers(bad) is False

    def _no_profile(name):
        raise RuntimeError("no profile in this process")

    monkeypatch.setattr(rt, "_load_profile", _no_profile)
    assert rt.sp_qkv_covers(_payload()) is False
    assert covers_env.created == []
    assert covers_env.allocated == []
    assert rt._SESSIONS == {}


BOUND_TENSORS = (
    "w_scale",
    "k_cache",
    "v_cache",
    "k_scale",
    "v_scale",
    "q_norm_w",
    "k_norm_w",
    "cos_cache",
    "sin_cache",
)
"""Payload entries the plan binds by pointer (caller outputs are NOT bound)."""


def _relink_bound(dst, src):
    """Reuse the bound tensors of ``src`` in ``dst`` (only per-call tensors differ)."""

    for name in BOUND_TENSORS:
        dst[name] = src[name]
    return dst


def test_staging_outputs_and_copy_back_for_pad_bucket(env):
    env.bucket = 16  # bucket M=16, real local m=8 => T_real=32, T_bucket=64
    payload = _payload(m=8)
    rt.sp_qkv_native(payload)
    prepared = env.plan.prepares[0]["outs"]
    assert prepared["q"].shape[0] == 64  # the bucket's T
    assert prepared["q"] is not payload["q"]
    assert len(env.copies) == 3  # q, k, v
    assert all(call[2] == 32 for call in env.copies)
    assert env.copies[0][0] is payload["q"]
    env.copies.clear()
    rt.sp_qkv_native(payload)  # replay: same binding, real rows copied again
    assert len(env.copies) == 3
    assert len(env.plan.prepares) == 1


def test_exact_bucket_also_uses_runtime_staging(env):
    payload = _payload(m=8)
    rt.sp_qkv_native(payload)
    prepared = env.plan.prepares[0]["outs"]
    assert prepared["q"].shape[0] == 32  # bucket T == real T here
    assert prepared["q"] is not payload["q"]
    assert all(call[1] is not payload["q"] for call in env.copies)  # src is staging
    assert [call[2] for call in env.copies] == [32, 32, 32]
    assert env.copies[0][0] is payload["q"]  # ...dst is the caller's tensor


def test_shared_output_buffer_across_layers(env):
    env.bucket = 16
    g = object()
    p1 = _payload(m=8, group=g)
    rt.sp_qkv_native(p1)
    p2 = _relink_bound(_payload(m=8, group=g), p1)  # different per-layer output tensors
    env.copies.clear()
    rt.sp_qkv_native(p2)  # same session/binding: caller outputs are not bound
    assert env.plan.prepares == [env.plan.prepares[0]]  # prepared once
    assert len(env.copies) == 3
    assert env.copies[0][0] is p2["q"]


def test_same_bucket_different_real_m_copies_per_call(env):
    env.bucket = 16
    p8 = _payload(m=8)
    rt.sp_qkv_native(p8)
    assert [call[2] for call in env.copies] == [32, 32, 32]
    env.copies.clear()
    p12 = _relink_bound(_payload(m=12), p8)  # same bucket, a different real m
    rt.sp_qkv_native(p12)
    assert len(env.plan.prepares) == 1  # one binding serves the bucket
    assert [call[2] for call in env.copies] == [48, 48, 48]
    assert env.copies[0][0] is p12["q"]


# ---------------------------------------------------------------------------
# session lifecycle and cache keying
# ---------------------------------------------------------------------------
def test_session_cache_keying(env):
    g1, g2 = object(), object()
    p1 = _payload(group=g1)  # a bound session needs address-stable tensors
    rt.sp_qkv_native(p1)
    rt.sp_qkv_native(p1)
    assert len(env.resources) == 1
    assert len(rt._SESSIONS) == 1
    rt.sp_qkv_native(_payload(group=g2))
    assert len(env.resources) == 2
    assert len(rt._SESSIONS) == 2


def test_open_and_close_session_lifecycle(env):
    g = object()
    a = rt.open_session(group=g)
    b = rt.open_session(group=g)
    assert a is b
    assert len(env.resources) == 1
    rt.close_session(group=g)
    assert env.resources[0].released == 1
    assert not rt._SESSIONS
    rt.close_session(group=g)  # idempotent


def test_close_session_refused_during_capture(env):
    g = object()
    rt.open_session(group=g)
    env.capturing = True
    with pytest.raises(RuntimeError, match="capture"):
        rt.close_session(group=g)


# ---------------------------------------------------------------------------
# graph-capture protocol
# ---------------------------------------------------------------------------
def test_prepare_before_capture_refusal_then_replay(env):
    payload = _payload()
    env.capturing = True
    with pytest.raises(RuntimeError, match="capture"):
        rt.sp_qkv_native(payload)
    env.capturing = False
    rt.sp_qkv_native(payload)
    assert len(env.plan.prepares) == 1
    assert len(env.plan.runs) == 1
    env.capturing = True
    rt.sp_qkv_native(payload)
    assert len(env.plan.prepares) == 1  # prepared once, replay only runs
    assert len(env.plan.runs) == 2


def test_moved_bound_pointer_rejected(env):
    payload = _payload()
    rt.sp_qkv_native(payload)
    payload["k_cache"] = FakeTensor((2, 1, 8, 128, 16), "float8_e4m3fn")
    with pytest.raises(FakeContractError, match="moved"):
        rt.sp_qkv_native(payload)


# ---------------------------------------------------------------------------
# call ids
# ---------------------------------------------------------------------------
def test_call_id_counter_rank_uniform_sequence(env):
    payload = _payload()
    rt.sp_qkv_native(payload)
    rt.sp_qkv_native(payload)
    rt.sp_qkv_native(payload)
    ids = [run["call_id"] for run in env.plan.runs]
    assert len(set(ids)) == 3
    assert all(1 <= i <= rt._TAG_CALLID_MASK for i in ids)
    assert ids == sorted(ids)
    rt._reset_for_testing()
    first = rt._next_call_id()
    rt._reset_for_testing()
    assert rt._next_call_id() == first  # the same sequence on every rank/process


# ---------------------------------------------------------------------------
# profile selection
# ---------------------------------------------------------------------------
def test_profile_env_selection(env, monkeypatch):
    monkeypatch.setenv(rt.PROFILE_ENV, "minimax_m3_sp4_gfx950_v1_r19")
    assert rt.profile_name() == "minimax_m3_sp4_gfx950_v1_r19"
    rt.sp_qkv_native(_payload())
    assert env.profile_names == ["minimax_m3_sp4_gfx950_v1_r19"]
    monkeypatch.delenv(rt.PROFILE_ENV)
    assert rt.profile_name() == rt.DEFAULT_PROFILE


def test_invalid_profile_is_reported(env, monkeypatch):
    def _raise(name):
        raise rt._gf().ContractError(f"cannot read profile {name}")

    monkeypatch.setattr(rt, "_load_profile", _raise)
    with pytest.raises(FakeContractError, match="cannot read profile"):
        rt.sp_qkv_native(_payload())
    monkeypatch.setattr(rt, "_load_profile", _raise)
    assert rt.sp_qkv_available() is False


# ---------------------------------------------------------------------------
# ERR-page readback
# ---------------------------------------------------------------------------
def test_err_readback_raises_typed_error(env):
    g = object()
    payload = _payload(group=g)
    rt.sp_qkv_native(payload)
    session = env.session()
    res = session.res
    assert res.enqueued == [0]
    res.words[0] = 0x21
    with pytest.raises(rt.GemmFusionDeviceError, match="0x00000021"):
        rt.sp_qkv_native(payload)


def test_check_errors_helper_and_close_consume(env):
    g = object()
    payload = _payload(group=g)
    rt.sp_qkv_native(payload)
    session = env.session()
    session.res.words[0] = 0x32
    with pytest.raises(rt.GemmFusionDeviceError):
        rt.check_errors(group=g)
    assert session.readback_done == 1
    rt.check_errors(group=g)  # already consumed: no repeated raise
    rt.close_session(group=g)


# ---------------------------------------------------------------------------
# optional post-prologue native stage
# ---------------------------------------------------------------------------
def test_native_stage_callback_invoked(env):
    calls = []

    def stage(**kwargs):
        calls.append(kwargs)

    g = object()
    payload = _payload(group=g)
    rt.open_session(group=g, native_stage=stage)
    rt.sp_qkv_native(payload)
    assert len(calls) == 1
    assert set(calls[0]["outs"]) == {"q", "k", "v"}
    # the continuation consumes the runtime staging (real rows first), not the
    # caller's public output tensors
    assert calls[0]["outs"]["q"] is not payload["q"]
    assert calls[0]["outs"]["q"].shape[0] == 32
    assert calls[0]["meta"]["positions"] is not None
    rt.sp_qkv_native(_payload())  # default session: no callback
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# availability / lazy imports
# ---------------------------------------------------------------------------
def test_sp_qkv_available_true_with_loadable_profile(env):
    assert rt.sp_qkv_available() is True


def test_import_is_lazy():
    """Importing the runtime must not import the fusion framework or MORI."""

    code = (
        "import sys;"
        "import aiter.ops.gemm_fusion.sp_qkv.runtime as m;"
        "bad=[n for n in ('aiter.gemm_fusion','mori','mori.cco') if n in sys.modules];"
        "print(','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", out.stdout
