# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only unit tests for aiter.gemm_fusion framework (S2-2a). Run: python -m pytest op_tests/test_gemm_fusion_framework.py"""

import json
import logging
import os
import sys
import shutil
import tempfile
import uuid
import types
from pathlib import Path

import pytest

try:
    import aiter.gemm_fusion as gf  # noqa: F401
except Exception:  # no torch / GPU: load the pure-Python subpackage without aiter/__init__
    for _k in [k for k in sys.modules if k == "aiter" or k.startswith("aiter.")]:
        del sys.modules[_k]
    _root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "aiter")
    _pkg = types.ModuleType("aiter")
    _pkg.__path__ = [os.path.abspath(_root)]
    sys.modules["aiter"] = _pkg
    import aiter.gemm_fusion as gf  # noqa: F401

from aiter.gemm_fusion import (
    Contract, ContractError, Field, FusionSpec, GemmSpec, HeadNormRopeCache, MatchCtx, Plan,
    ProfileError, SPExchange, WindowReq, match_prefix, parse_profile, qkv_fields, register_variant,
)
from aiter.gemm_fusion import catalog, profile as profmod


def mkspec(eps=1e-6, drop=()):
    f = qkv_fields(q=8192, kv=512, world=4, kv_shards=4)
    return FusionSpec(
        GemmSpec(n=9216, k=6144),
        (SPExchange(world=4, fields=f, drop=drop), HeadNormRopeCache(eps=eps)),
    )


def test_signature_stable_and_golden():
    s = mkspec()
    assert s.signature() == mkspec().signature()
    assert len(s.signature()) == 16
    # canonical float: 1e-6 spelled differently is the same signature
    assert mkspec(eps=0.000001).signature() == mkspec(eps=float("1e-06")).signature()
    assert '"f:1e-06"' in s.canonical_json()


def test_signature_sensitivity():
    base = mkspec().signature()
    assert mkspec(eps=1e-5).signature() != base
    assert mkspec(drop=("iq", "ik")).signature() != base
    assert mkspec().prefix(1).signature() != base
    # field order is semantic
    f = qkv_fields(q=8192, kv=512, world=4, kv_shards=4)
    s2 = FusionSpec(GemmSpec(n=9216, k=6144), (SPExchange(world=4, fields=tuple(reversed(f))), HeadNormRopeCache()))
    assert s2.signature() != base
    # int 1 vs bool True never alias
    with pytest.raises(ValueError):
        mkspec(eps=float("nan")).signature()


@pytest.fixture
def tmp_path():
    # plain mkdir: mkdtemp's restrictive ACL breaks on some sandboxed Windows hosts
    d = Path(tempfile.gettempdir()) / ('gf_' + uuid.uuid4().hex)
    d.mkdir()
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def reg(monkeypatch):
    monkeypatch.setattr(catalog, "_CATALOG", {})
    made = {}

    def add(vid, kinds, requires=()):
        @register_variant(
            vid=vid, family="t", stage_kinds=kinds, source_sha="x",
            contract=Contract(("gfx950",), 4, "t", "g", requires=requires, m_multiple=16),
        )
        class P(Plan):
            def __init__(self, spec, params, ctx):
                self.params = params

        made[vid] = P
        return P

    return add


def write_prof(tmp_path, rows, name="p.json"):
    p = tmp_path / name
    p.write_text(json.dumps({"profile": "t", "rows": rows}))
    profmod.clear_profile_cache()
    return str(p)


def row(sig, m, vid, **params):
    return {"sig": sig, "arch": "gfx950", "cu": 256, "world": 4, "m": m, "vid": vid, "params": params}


CTX = lambda m, avail=("mori_cco",): MatchCtx("gfx950", 256, 4, m, frozenset(avail))  # noqa: E731


def test_exact_match_and_dispatch_log(tmp_path, reg, caplog):
    reg("v.full", ("sp_exchange", "head_norm_rope_cache"))
    s = mkspec()
    p = write_prof(tmp_path, [row(s.signature(), 16, "v.full", sk=3)])
    # The real aiter logger sets propagate=False (aiter/__init__.py), so caplog (root handler) never sees child records:
    # capture "aiter.gemm_fusion" explicitly with our own handler, independent of the aiter logging convention/level.
    records = []

    class _H(logging.Handler):
        def emit(self, record):
            records.append(record)

    lg = logging.getLogger("aiter.gemm_fusion")
    h, old_level = _H(level=logging.INFO), lg.level
    lg.addHandler(h)
    lg.setLevel(logging.INFO)
    try:
        plan, n = match_prefix(s, CTX(16), p)
    finally:
        lg.removeHandler(h)
        lg.setLevel(old_level)
    assert n == 2 and plan.vid == "v.full" and plan.params == {"sk": 3}
    assert any("event=dispatch" in r.getMessage() and "vid=v.full" in r.getMessage() for r in records)
    # exact rows only: other M -> native
    assert match_prefix(s, CTX(32), p) == (None, 0)


def test_prefix_fallback(tmp_path, reg):
    reg("v.ex", ("sp_exchange",))
    s = mkspec()
    p = write_prof(tmp_path, [row(s.prefix(1).signature(), 2048, "v.ex")])
    plan, n = match_prefix(s, CTX(2048), p)
    assert n == 1 and plan.vid == "v.ex"


def test_longest_prefix_wins(tmp_path, reg):
    reg("v.ex", ("sp_exchange",))
    reg("v.full", ("sp_exchange", "head_norm_rope_cache"))
    s = mkspec()
    p = write_prof(tmp_path, [row(s.prefix(1).signature(), 16, "v.ex"), row(s.signature(), 16, "v.full")])
    assert match_prefix(s, CTX(16), p)[1] == 2


def test_contract_error(tmp_path, reg):
    reg("v.full", ("sp_exchange", "head_norm_rope_cache"), requires=("mori_cco",))
    s = mkspec()
    p = write_prof(tmp_path, [row(s.signature(), 16, "v.full")])
    with pytest.raises(ContractError, match="missing"):
        match_prefix(s, CTX(16, avail=()), p)
    p2 = write_prof(tmp_path, [row(s.signature(), 20, "v.full")], "q.json")
    with pytest.raises(ContractError, match="m 20"):
        match_prefix(s, CTX(20), p2)
    p3 = write_prof(tmp_path, [row(s.signature(), 16, "nope")], "r.json")
    with pytest.raises(ContractError, match="unregistered"):
        match_prefix(s, CTX(16), p3)


def test_duplicate_rows_and_validation():
    r = row("abc", 16, "v")
    with pytest.raises(ProfileError, match="duplicate"):
        parse_profile({"rows": [r, dict(r)]})
    with pytest.raises(ProfileError, match="int"):
        parse_profile({"rows": [dict(r, m=[17, 32])]})
    with pytest.raises(ProfileError, match="missing"):
        parse_profile({"rows": [{"sig": "x"}]})


def test_profile_env_and_immutable(tmp_path, monkeypatch):
    p = write_prof(tmp_path, [row("abc", 16, "v", a=1)])
    monkeypatch.setenv("AITER_GEMM_FUSION_PROFILE", p)
    prof = profmod.load_profile()
    rw = prof.lookup("abc", "gfx950", 256, 4, 16)
    with pytest.raises(TypeError):
        rw.params["a"] = 2
    monkeypatch.delenv("AITER_GEMM_FUSION_PROFILE")
    with pytest.raises(ProfileError):
        profmod.load_profile()


def test_duplicate_variant(reg):
    reg("dup", ("sp_exchange",))
    with pytest.raises(ValueError):
        reg("dup", ("sp_exchange",))


def test_capabilities_never_raise():
    from aiter.gemm_fusion import detect_capabilities

    assert isinstance(detect_capabilities(arch="gfx950"), frozenset)
    assert "gfx950_sc_bypass" in detect_capabilities(arch="gfx950")
    assert "gfx950_sc_bypass" not in detect_capabilities(arch="gfx942")
