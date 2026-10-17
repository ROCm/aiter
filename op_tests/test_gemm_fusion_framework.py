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


# ---- capability probe (CCO shared library via ctypes path; the `mori` Python package is never imported) ----
def _fake_elf(symbols):
    """Minimal little-endian ELF64 with one .dynsym (defined FUNC symbols) + .dynstr + .shstrtab (not dlopen-able)."""
    import struct as st

    dynstr = b"\0" + b"".join(s.encode() + b"\0" for s in symbols)
    offs, o = [], 1
    for s in symbols:
        offs.append(o)
        o += len(s) + 1
    dynsym = b"\0" * 24 + b"".join(st.pack("<IBBHQQ", off, 0x12, 0, 1, 0, 0) for off in offs)
    shstr = b"\0.dynsym\0.dynstr\0.shstrtab\0"
    body = dynsym + dynstr + shstr
    base = 64
    o_dynsym, o_dynstr, o_shstr = base, base + len(dynsym), base + len(dynsym) + len(dynstr)
    shoff = base + len(body)

    def sh(name, typ, off, size, link=0, entsize=0):
        return st.pack("<IIQQQQIIQQ", name, typ, 0, 0, off, size, link, 0, 8, entsize)

    shdrs = b"\0" * 64 + sh(1, 11, o_dynsym, len(dynsym), link=2, entsize=24) + sh(9, 3, o_dynstr, len(dynstr)) + sh(17, 3, o_shstr, len(shstr))
    ehdr = b"\x7fELF" + bytes([2, 1, 1, 0]) + b"\0" * 8 + st.pack("<HHIQQQIHHHHHH", 3, 0xE0, 1, 0, 0, shoff, 0, 64, 0, 0, 64, 4, 3)
    return ehdr + body + shdrs


def _mangle(f):  # mori::cco::<f>(...) -> _ZN4mori3cco<len><name>E...
    return f"_ZN4mori3cco{len(f)}{f}EPv"


SDMA_HDR = "#ifndef BUILD_CCO_SDMA\n#define BUILD_CCO_SDMA 0\n#endif\nstruct ccoSdmaContext {};\nint sdmaQueueCount;\n"


@pytest.fixture
def mori_tree(tmp_path):
    root = tmp_path / "mori"
    (root / "include" / "mori" / "cco").mkdir(parents=True)
    (root / "include" / "mori" / "cco" / "cco.hpp").write_text(SDMA_HDR)
    (root / "build" / "src" / "cco").mkdir(parents=True)
    return root


def _write_lib(path, extra=(), drop=()):
    from aiter.gemm_fusion.capabilities import CCO_REQUIRED_SYMBOLS

    syms = [_mangle(f) for f in CCO_REQUIRED_SYMBOLS if f not in drop] + list(extra)
    path.write_bytes(_fake_elf(syms))


def test_capabilities_probe_cco_library(mori_tree):
    from aiter.gemm_fusion import capabilities as cap

    so = mori_tree / "build" / "src" / "cco" / "libmori_cco.so"
    _write_lib(so)  # NO sdma/anvil symbol names (like the real libmori_cco.so 87b7322f): sdma must still be detected via the header
    env = {"MORI_ROOT": str(mori_tree)}
    # symbols present, library not dlopen-able (fake ELF) -> dlopen check fails; with the check off it passes
    assert cap.probe_mori_cco(env, check_dlopen=True)[0] is False
    ok, sdma, why = cap.probe_mori_cco(env, check_dlopen=False)
    assert ok and sdma, why
    assert cap.detect_capabilities(arch="gfx950", env=env, check_dlopen=False) == frozenset({"gfx950_sc_bypass", "mori_cco", "sdma"})
    # header without the SDMA device API -> mori_cco only
    hdr = mori_tree / "include" / "mori" / "cco" / "cco.hpp"
    hdr.write_text("// no sdma here\nint ccoDevCommCreate();\n")
    assert cap.detect_capabilities(arch="gfx950", env=env, check_dlopen=False) == frozenset({"gfx950_sc_bypass", "mori_cco"})
    hdr.write_text("#define BUILD_CCO_SDMA 0\nstruct ccoSdmaContext;\n")  # a token missing (sdmaQueueCount) -> not enough
    assert "sdma" not in cap.detect_capabilities(arch="gfx950", env=env, check_dlopen=False)
    hdr.write_text(SDMA_HDR)
    # a missing required symbol -> no capability at all
    _write_lib(so, drop=("ccoWindowRegister",))
    cap._SYM_CACHE.clear()
    ok, sdma, why = cap.probe_mori_cco(env, check_dlopen=False)
    assert not ok and not sdma and "ccoWindowRegister" in why
    # explicit MORI_CCO_SO wins over MORI_ROOT and a bad explicit path does not fall through
    good = mori_tree / "alt.so"
    _write_lib(good)
    cap._SYM_CACHE.clear()
    env2 = {"MORI_ROOT": str(mori_tree), "MORI_CCO_SO": str(good)}
    assert cap.resolve_cco_library(env2) == str(good) and cap.probe_mori_cco(env2, check_dlopen=False)[:2] == (True, True)
    assert cap.resolve_cco_library({"MORI_ROOT": str(mori_tree), "MORI_CCO_SO": str(mori_tree / "nope.so")}) is None
    # MORI_ROOT/lib is preferred over build/src/cco
    (mori_tree / "lib").mkdir()
    libp = mori_tree / "lib" / "libmori_cco.so"
    _write_lib(libp)
    assert cap.resolve_cco_library({"MORI_ROOT": str(mori_tree)}) == str(libp)


def test_capabilities_probe_failures_never_raise(mori_tree, tmp_path):
    from aiter.gemm_fusion import capabilities as cap

    so = mori_tree / "build" / "src" / "cco" / "libmori_cco.so"
    so.write_bytes(b"not an elf")
    env = {"MORI_ROOT": str(mori_tree)}
    ok, sdma, why = cap.probe_mori_cco(env)
    assert not ok and not sdma and "ELF" in why
    so.write_bytes(b"\x7fELF" + b"\x02\x01" + b"\0" * 10)  # truncated
    cap._SYM_CACHE.clear()
    assert cap.probe_mori_cco(env)[0] is False
    # header missing -> no capability even with a good library
    _write_lib(so)
    cap._SYM_CACHE.clear()
    (mori_tree / "include" / "mori" / "cco" / "cco.hpp").unlink()
    ok, _s, why = cap.probe_mori_cco(env, check_dlopen=False)
    assert not ok and "header" in why
    # nothing at all
    assert cap.probe_mori_cco({"MORI_ROOT": str(tmp_path / "empty")}, check_dlopen=False)[0] is False
    assert "mori_cco" not in cap.detect_capabilities(arch="gfx942", env={"MORI_ROOT": str(tmp_path / "empty")})


def test_capabilities_never_imports_mori_python_package(mori_tree):
    import sys

    from aiter.gemm_fusion import capabilities as cap

    before = "mori" in sys.modules
    cap.detect_capabilities(arch="gfx950", env={"MORI_ROOT": str(mori_tree)}, check_dlopen=False)
    assert ("mori" in sys.modules) == before
