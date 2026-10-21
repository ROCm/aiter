# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tests for the profile RANGE RULES (P1): the mechanism (aiter.gemm_fusion.profile / match), the shipped sp_qkv profile's large rules and the prebuild
enumeration they imply. Run: python -m pytest op_tests/test_gemm_fusion_profile_rules.py
"""

import dataclasses
import itertools
import json
import logging
import os
import shutil
import sys
import tempfile
import types
import uuid
from pathlib import Path

import pytest

try:
    import aiter.gemm_fusion as gf
except Exception:  # noqa: BLE001
    # no torch / GPU: load the pure-Python subpackage without aiter/__init__
    for _k in [k for k in sys.modules if k == "aiter" or k.startswith("aiter.")]:
        del sys.modules[_k]
    _root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "aiter")
    _pkg = types.ModuleType("aiter")
    _pkg.__path__ = [os.path.abspath(_root)]
    sys.modules["aiter"] = _pkg
    import aiter.gemm_fusion as gf  # noqa: F401

from aiter.gemm_fusion import (
    Contract,
    ContractError,
    FusionSpec,
    GemmSpec,
    HeadNormRopeCache,
    MatchCtx,
    Plan,
    ProfileError,
    Rule,
    SPExchange,
    catalog,
    match_prefix,
    parse_profile,
    qkv_fields,
    register_variant,
)
from aiter.gemm_fusion import prebuild as engine
from aiter.gemm_fusion import profile as profmod
from aiter.ops.gemm_fusion.sp_qkv import gen_profile, jit, p1_sweep, specs
from aiter.ops.gemm_fusion.sp_qkv import prebuild as pb
from aiter.ops.gemm_fusion.sp_qkv.test_utils import clear_profile_cache

PROFILE = "minimax_m3_sp4_gfx950_v1"
TOOL = {
    "hipcc": "/rocm/bin/hipcc",
    "clangxx": "/rocm/llvm/bin/clang++",
    "bundler": "/rocm/llvm/bin/clang-offload-bundler",
    "rocm": "/rocm",
    "mori_root": "/mori",
    "mori_so": "/mori/build/src/cco/libmori_cco.so",
}


# ----------------------------------------------------------------------------------------------------------- helpers
@pytest.fixture
def tmp_path():  # the suite runs with -p no:tmpdir; mkdtemp's ACL breaks on sandboxed Windows
    d = Path(tempfile.gettempdir()) / ("gfr_" + uuid.uuid4().hex)
    d.mkdir()
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def mkspec():
    f = qkv_fields(q=8192, kv=512, world=4, kv_shards=4)
    return FusionSpec(
        GemmSpec(n=9216, k=6144),
        (SPExchange(world=4, fields=f), HeadNormRopeCache()),
    )


@pytest.fixture
def reg(monkeypatch):
    """Scratch variants whose contract is m_multiple=16 (so rules can be inside or outside what the variant accepts)."""
    monkeypatch.setattr(catalog, "_CATALOG", {})
    made = {}

    def add(vid, kinds, requires=(), m_multiple=16):
        @register_variant(
            vid=vid,
            family="t",
            stage_kinds=kinds,
            source_sha="x",
            contract=Contract(
                ("gfx950",), 4, "t", "g", requires=requires, m_multiple=m_multiple
            ),
        )
        class P(Plan):
            def __init__(self, spec, params, ctx):
                self.params = params
                self.m = ctx.m

        made[vid] = P
        return P

    return add


def row(sig, m, vid, **params):
    return {
        "sig": sig,
        "arch": "gfx950",
        "cu": 256,
        "world": 4,
        "m": m,
        "vid": vid,
        "params": params,
    }


def rule(sig, m_min, m_max, step, vid, name="", cu=256, **params):
    r = {
        "sig": sig,
        "arch": "gfx950",
        "cu": cu,
        "world": 4,
        "m_min": m_min,
        "m_max": m_max,
        "m_multiple": step,
        "vid": vid,
        "params": params,
    }
    if name:
        r["name"] = name
    return r


def write_doc(tmp_path, rows=(), rules=(), name="p.json", raw=None):
    doc = (
        raw
        if raw is not None
        else {"profile": "t", "rows": list(rows), "rules": list(rules)}
    )
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    clear_profile_cache()
    return str(p)


def CTX(m, avail=("mori_cco",), cu=256):
    return MatchCtx("gfx950", cu, 4, m, frozenset(avail))


class capture:
    """Collect aiter.gemm_fusion log messages (the real aiter logger has propagate=False)."""

    def __enter__(self):
        self.msgs = []

        class H(logging.Handler):
            def emit(_s, rec):
                self.msgs.append(rec.getMessage())

        self.lg, self.h, self.old = (
            logging.getLogger("aiter.gemm_fusion"),
            H(level=logging.INFO),
            logging.getLogger("aiter.gemm_fusion").level,
        )
        self.lg.addHandler(self.h)
        self.lg.setLevel(logging.INFO)
        return self.msgs

    def __exit__(self, *a):
        self.lg.removeHandler(self.h)
        self.lg.setLevel(self.old)


# ----------------------------------------------------------------------------------------------------------- parse / validation / immutability
def test_rules_parse_are_immutable_and_kept_out_of_meta_and_len():
    r = rule("abc", 16, 64, 16, "v", name="x", n=2)
    prof = parse_profile(
        {
            "profile": "t",
            "window_budget": {"window_bytes": 4 << 20},
            "rows": [row("abc", 32, "v")],
            "rules": [r],
        }
    )
    assert (
        len(prof) == 1 and len(prof.rules()) == 1 and len(prof.entries()) == 2
    )  # len() counts exact rows only
    assert (
        "rules" not in prof.meta
        and "rows" not in prof.meta
        and prof.meta["window_budget"] == {"window_bytes": 4 << 20}
    )
    ru = prof.rules()[0]
    assert (
        isinstance(ru, Rule) and isinstance(prof.rules(), tuple) and ru.label() == "x"
    )
    with pytest.raises(dataclasses.FrozenInstanceError):  # frozen dataclass
        ru.m_min = 1
    with pytest.raises(TypeError):  # params is a read-only mapping
        ru.params["n"] = 4
    r["params"][
        "n"
    ] = 99  # mutating the source document afterwards does not reach the parsed profile
    assert dict(ru.params) == {"n": 2}
    assert (
        parse_profile({"rows": []}).rules() == ()
    )  # the field is optional: old profiles still parse
    assert (
        Rule("s", "a", 1, 1, 16, 64, 16, "v", types.MappingProxyType({})).label()
        == "m16-64/16"
    )


@pytest.mark.parametrize(
    "bad,match",
    [
        ("notalist", r"'rules' must be a list"),
        (["x"], r"rule 0 must be an object"),
        ([{"sig": "a"}], r"rule 0 missing"),
        ([dict(rule("a", 16, 64, 16, "v"), m_min=True)], r"m_min must be an int"),
        (
            [dict(rule("a", 16, 64, 16, "v"), m_multiple="16")],
            r"m_multiple must be an int",
        ),
        ([rule("a", 64, 16, 16, "v")], r"1 <= m_min <= m_max"),
        ([rule("a", 0, 16, 16, "v")], r"1 <= m_min <= m_max"),
        ([rule("a", 16, 64, 0, "v")], r"m_multiple >= 1"),
        ([rule("a", 17, 20, 16, "v")], r"covers no m"),
        ([dict(rule("a", 16, 64, 16, "v"), params=[1])], r"params must be an object"),
    ],
)
def test_rule_validation_errors(bad, match):
    with pytest.raises(ProfileError, match=match):
        parse_profile({"rows": [], "rules": bad})


def test_equal_specificity_overlap_is_rejected_different_specificity_and_disjoint_are_fine():
    a = rule("s", 16, 64, 16, "v")
    with pytest.raises(
        ProfileError, match="equally specific.*m_multiple 16.*cover m=32"
    ):
        parse_profile({"rows": [], "rules": [a, rule("s", 32, 96, 16, "v")]})
    with pytest.raises(ProfileError, match="equally specific"):
        parse_profile({"rows": [], "rules": [a, dict(a)]})
    # same multiple, disjoint ranges; same range with different keys: fine
    parse_profile(
        {
            "rows": [],
            "rules": [
                a,
                rule("s", 80, 96, 16, "v"),
                rule("t", 16, 64, 16, "v"),
                rule("s", 16, 64, 16, "v", cu=304),
            ],
        }
    )
    # same range, different m_multiple: the narrower (larger m_multiple) is more specific and wins, no error
    prof = parse_profile({"rows": [], "rules": [a, rule("s", 16, 64, 32, "v2")]})
    assert (
        prof.lookup_rule("s", "gfx950", 256, 4, 32).vid == "v2"
        and prof.lookup_rule("s", "gfx950", 256, 4, 48).vid == "v"
    )
    # same multiple but the overlap has no common m (ranges touch between grid points): fine
    parse_profile(
        {"rows": [], "rules": [rule("s", 16, 40, 32, "v"), rule("s", 48, 90, 32, "v")]}
    )


def test_most_specific_rule_wins_regardless_of_file_order():
    rules = [
        rule("s", 16, 128, 16, "base"),
        rule("s", 16, 128, 32, "mid"),
        rule("s", 16, 128, 64, "top"),
    ]
    expect = {}
    for m in range(16, 129, 16):
        expect[m] = "top" if m % 64 == 0 else "mid" if m % 32 == 0 else "base"
    for perm in itertools.permutations(rules):
        prof = parse_profile({"rows": [], "rules": list(perm)})
        got = {m: prof.lookup_rule("s", "gfx950", 256, 4, m).vid for m in expect}
        assert (
            got == expect
        )  # identical for every file order: no order-dependent tie-break exists
    assert (
        prof.lookup_rule("s", "gfx950", 256, 4, 20) is None
    )  # off the m_multiple grid
    assert prof.lookup_rule("s", "gfx950", 256, 4, 144) is None  # beyond m_max
    assert prof.lookup_rule("s", "gfx950", 256, 4, 0) is None


def test_resolve_prefers_the_exact_row_and_reports_the_source():
    prof = parse_profile(
        {"rows": [row("s", 32, "rowv")], "rules": [rule("s", 16, 64, 16, "rulev")]}
    )
    e, src = prof.resolve("s", "gfx950", 256, 4, 32)
    assert (e.vid, src) == ("rowv", "row")  # the exact row wins inside the rule's range
    e, src = prof.resolve("s", "gfx950", 256, 4, 48)
    assert (e.vid, src) == ("rulev", "rule")
    assert prof.resolve("s", "gfx950", 256, 4, 20) == (None, "none") and prof.resolve(
        "s", "gfx950", 256, 4, 80
    ) == (None, "none")
    # other keys never match: another sig, arch, cu or world
    assert (
        prof.resolve("t", "gfx950", 256, 4, 48)[1] == "none"
        and prof.resolve("s", "gfx942", 256, 4, 48)[1] == "none"
    )
    assert (
        prof.resolve("s", "gfx950", 304, 4, 48)[1] == "none"
        and prof.resolve("s", "gfx950", 256, 8, 48)[1] == "none"
    )
    # lookup() is still exact-only (rules are consulted by resolve)
    assert prof.lookup("s", "gfx950", 256, 4, 48) is None


# ----------------------------------------------------------------------------------------------------------- matching
def test_rule_matches_where_no_row_exists_logs_the_source_and_passes_the_rule_params(
    tmp_path, reg
):
    reg("v.full", ("sp_exchange", "head_norm_rope_cache"))
    s = mkspec()
    p = write_doc(
        tmp_path,
        rows=[row(s.signature(), 32, "v.full", sk=1)],
        rules=[rule(s.signature(), 16, 64, 16, "v.full", name="mid", sk=7)],
    )
    with capture() as msgs:
        exact, n1 = match_prefix(s, CTX(32), p)
        ruled, n2 = match_prefix(s, CTX(48), p)
        native = match_prefix(s, CTX(80), p)
        off_grid = match_prefix(s, CTX(24), p)
    assert (n1, exact.params, exact.profile_source) == (
        2,
        {"sk": 1},
        "row",
    )  # the exact row wins at m=32 although the rule spans it
    assert (n2, ruled.params, ruled.profile_source, ruled.m) == (
        2,
        {"sk": 7},
        "rule",
        48,
    )
    assert native == (None, 0) and off_grid == (None, 0)
    disp = [m for m in msgs if "event=dispatch" in m]
    assert "m=32" in disp[0] and "source=row" in disp[0] and "rule=" not in disp[0]
    assert (
        "m=48" in disp[1]
        and "source=rule rule=mid" in disp[1]
        and "vid=v.full" in disp[1]
        and "params={'sk': 7}" in disp[1]
    )
    assert (
        "vid=native" in disp[2]
        and "source=none" in disp[2]
        and "source=none" in disp[3]
    )
    # the plan got its own copy of the params: mutating it cannot reach the immutable profile
    ruled.params["sk"] = 99
    assert dict(match_prefix(s, CTX(48), p)[0].params) == {"sk": 7}


def test_longest_prefix_wins_across_rows_and_rules(tmp_path, reg):
    reg("v.ex", ("sp_exchange",))
    reg("v.full", ("sp_exchange", "head_norm_rope_cache"))
    s = mkspec()
    # a RULE at the longer prefix beats an exact ROW at the shorter prefix; an exact row at the longer prefix beats a rule at the shorter one
    p = write_doc(
        tmp_path,
        rows=[row(s.prefix(1).signature(), 32, "v.ex")],
        rules=[rule(s.signature(), 16, 64, 16, "v.full")],
    )
    plan, n = match_prefix(s, CTX(32), p)
    assert (n, plan.vid, plan.profile_source) == (2, "v.full", "rule")
    p2 = write_doc(
        tmp_path,
        rows=[row(s.signature(), 32, "v.full")],
        rules=[rule(s.prefix(1).signature(), 16, 64, 16, "v.ex")],
        name="q.json",
    )
    plan, n = match_prefix(s, CTX(32), p2)
    assert (n, plan.vid, plan.profile_source) == (2, "v.full", "row")
    plan, n = match_prefix(
        s, CTX(48), p2
    )  # no 2-stage entry at 48: the 1-stage rule applies
    assert (n, plan.vid, plan.profile_source) == (1, "v.ex", "rule")


def test_the_variant_contract_is_still_enforced_for_rule_matches(tmp_path, reg):
    reg("v.full", ("sp_exchange", "head_norm_rope_cache"), requires=("mori_cco",))
    s = mkspec()
    p = write_doc(tmp_path, rules=[rule(s.signature(), 16, 64, 16, "v.full")])
    with pytest.raises(ContractError, match="missing"):
        match_prefix(s, CTX(32, avail=()), p)  # capability missing
    # a context of another world / arch / cu never reaches the contract: the rule is keyed by them, so it resolves to native
    assert match_prefix(
        s, MatchCtx("gfx950", 256, 8, 32, frozenset({"mori_cco"})), p
    ) == (None, 0)
    # a rule whose m_multiple is not a multiple of the variant's (16) is a profile error, reported loudly at match time (never silently narrowed)
    p2 = write_doc(
        tmp_path,
        rules=[rule(s.signature(), 8, 64, 8, "v.full", name="loose")],
        name="q.json",
    )
    with pytest.raises(
        ContractError,
        match=r"range rule loose m_multiple 8 is not a multiple of the variant's 16",
    ):
        match_prefix(s, CTX(16), p2)
    p3 = write_doc(
        tmp_path, rules=[rule(s.signature(), 16, 64, 16, "nope")], name="r.json"
    )
    with pytest.raises(ContractError, match="unregistered"):
        match_prefix(s, CTX(32), p3)


def test_matching_is_deterministic_and_identical_on_every_rank(
    tmp_path, reg, monkeypatch
):
    reg("v.full", ("sp_exchange", "head_norm_rope_cache"))
    s = mkspec()
    rules = [
        rule(s.signature(), 16, 256, 16, "v.full", sk=1),
        rule(s.signature(), 16, 256, 32, "v.full", sk=2),
        rule(s.signature(), 64, 256, 64, "v.full", sk=3),
    ]
    rows = [row(s.signature(), 128, "v.full", sk=9)]
    results = []
    for rank in range(
        4
    ):  # each "rank" loads its own copy of the profile, in a different rule order, under different environments
        monkeypatch.setenv("RANK", str(rank))
        monkeypatch.setenv("PYTHONHASHSEED", str(rank))
        ordered = rules[rank % 3 :] + rules[: rank % 3]
        p = write_doc(tmp_path, rows=rows, rules=ordered, name=f"r{rank}.json")
        out = []
        for m in range(16, 289, 16):
            plan, n = match_prefix(s, CTX(m), p)
            out.append(
                (
                    m,
                    n,
                    (
                        None
                        if plan is None
                        else (plan.vid, dict(plan.params), plan.profile_source)
                    ),
                )
            )
        results.append(out)
    assert results[0] == results[1] == results[2] == results[3]
    by_m = {m: r for m, _n, r in results[0]}
    assert (
        by_m[128] == ("v.full", {"sk": 9}, "row")
        and by_m[64][1] == {"sk": 3}
        and by_m[32][1] == {"sk": 2}
        and by_m[48][1] == {"sk": 1}
        and by_m[272] is None
    )


# ----------------------------------------------------------------------------------------------------------- the shipped sp_qkv profile
_VARIANT_MODS = ("variants", "variants_large")
_SP_PKG = "aiter.ops.gemm_fusion.sp_qkv"


def _forget_variant_modules():
    """Drop the variant modules so the next match re-imports (and re-registers) them: sys.modules AND the package attributes (`from . import x` checks those first)."""
    pkg = sys.modules.get(_SP_PKG)
    for name in _VARIANT_MODS:
        sys.modules.pop(f"{_SP_PKG}.{name}", None)
        if pkg is not None and hasattr(pkg, name):
            delattr(pkg, name)


@pytest.fixture
def sp_qkv_catalog(monkeypatch):
    """A scratch catalog + freshly imported variant modules for one test; afterwards the ORIGINAL modules are put back exactly (so later tests that import the
    variants again do not register a second time into the restored catalog)."""
    pkg = sys.modules.get(_SP_PKG)
    saved_mods = {n: sys.modules.get(f"{_SP_PKG}.{n}") for n in _VARIANT_MODS}
    saved_attrs = {n: getattr(pkg, n, None) for n in _VARIANT_MODS}
    monkeypatch.setattr(catalog, "_CATALOG", {})
    _forget_variant_modules()
    clear_profile_cache()
    yield
    _forget_variant_modules()
    for n in _VARIANT_MODS:
        if saved_mods[n] is not None:
            sys.modules[f"{_SP_PKG}.{n}"] = saved_mods[n]
        if saved_attrs[n] is not None:
            setattr(sys.modules[_SP_PKG], n, saved_attrs[n])


LARGE_ROW_M = (2048, 4096, 8192)
GRID = range(1024, 8193, 256)
CAPS = frozenset({"mori_cco", "sdma"})


def test_shipped_profile_rules_are_signature_checked_and_match_the_generator():
    clear_profile_cache()
    prof = profmod.load_profile(PROFILE)
    assert len(prof) == 27 + 87 and len(prof.rules()) == 21
    large_rules = [r for r in prof.rules() if r.vid == "sp_qkv.large.v1"]
    small_rules = [r for r in prof.rules() if r.vid == "sp_qkv.small.v1"]
    assert (
        len(large_rules) == 9 and len(small_rules) == 12
    )  # 3 chunks-mod-4 layers x 3 modes; per mode 4 small ranges: r19 1..16, r19 17..32, r20 (MP64) 33..64, r21 (MP128) 65..128
    sigs = {
        specs.sp_qkv_spec(m).prefix(1).signature(): m for m in specs.MODES
    }  # large plans cover GEMM + SPExchange only
    for r in large_rules:
        assert r.sig in sigs and (r.arch, r.cu, r.world) == (
            "gfx950",
            256,
            4,
        ), r.label()
        assert (
            r.m_multiple in (256, 512, 1024)
            and 1024 <= r.m_min <= r.m_max <= 8192
            and "chunks_mod4" in r.name
        )
        assert (
            pb.row_mode(r) == sigs[r.sig]
        )  # the enumeration recovers the mode of a rule exactly as for a row
    sigs2 = {
        specs.sp_qkv_spec(m).signature(): m for m in specs.MODES
    }  # the small plans cover the whole 2-stage spec
    for r in small_rules:
        assert r.sig in sigs2 and (r.arch, r.cu, r.world) == ("gfx950", 256, 4)
        assert (r.m_min, r.m_max) in (
            (1, 16),
            (17, 32),
            (33, 64),
            (65, 128),
        ) and r.m_multiple == 1
        assert pb.row_mode(r) == sigs2[r.sig]
        assert ("mp64" in r.name) == (r.m_min == 33) and ("mp128" in r.name) == (
            r.m_min == 65
        )
        assert dict(r.params)["src"] == {33: "r20", 65: "r21"}.get(r.m_min, "r19")
        if r.m_min == 33:
            assert dict(r.params) == specs.small_r20_params(sigs2[r.sig])
        if r.m_min == 65:
            assert dict(r.params) == specs.small_r21_params(sigs2[r.sig])
    assert {r.sig for r in small_rules} == set(sigs2) and all(
        sum(1 for r in small_rules if r.sig == sg) == 4 for sg in sigs2
    )  # four disjoint ranges per mode; a large rule never shares a signature with a small row/rule
    small_sigs = {r.sig for r in prof.rows() if r.vid == "sp_qkv.small.v1"}
    assert small_sigs.isdisjoint({r.sig for r in large_rules})
    # the file is exactly what the generator writes (rules included): regenerate with op_tests/gen_gemm_fusion_profile.py
    with open(profmod._resolve(PROFILE), encoding="utf-8") as f:
        assert json.load(f) == json.loads(json.dumps(gen_profile.build_profile()))


def test_every_large_m_on_the_grid_is_an_exact_row_with_the_sweeps_best_params(
    sp_qkv_catalog,
):
    seen = {"row": 0, "rule": 0}
    for mode in specs.MODES:
        spec = specs.sp_qkv_spec(
            mode
        )  # the full 2-stage spec: the large rows sit at the 1-stage prefix
        for m in GRID:
            plan, n = match_prefix(spec, MatchCtx("gfx950", 256, 4, m, CAPS), PROFILE)
            assert (
                plan is not None
                and n == 1
                and plan.vid == "sp_qkv.large.v1"
                and plan.M == m
            ), (mode, m)
            assert (plan.n, plan.half_first) == p1_sweep.P1_BEST[mode][m], (mode, m)
            seen[plan.profile_source] += 1
            assert (
                (m // 256) % plan.n == 0 if plan.half_first != 2 else m == 2048
            )  # the plan constructor accepted the params at this M
    assert seen == {
        "row": 87,
        "rule": 0,
    }  # 29 grid points x 3 modes, every one exact: the rules are shadowed
    d2048, _ = match_prefix(
        specs.sp_qkv_spec("dense"), MatchCtx("gfx950", 256, 4, 2048, CAPS), PROFILE
    )
    assert (d2048.n, d2048.half_first, d2048.profile_source) == (
        3,
        2,
        "row",
    )  # quarter-first only here


def _rules_only_profile():
    """The shipped rules without any row: what a deployment would run for an m the sweep did not cover (the documented default)."""
    with open(profmod._resolve(PROFILE), encoding="utf-8") as f:
        shipped = json.load(f)
    return profmod.parse_profile(
        {"profile": "rules_only", "rows": [], "rules": shipped["rules"]}
    )


def test_fallback_rules_are_the_chunks_mod4_classes_and_valid_on_the_whole_grid(
    sp_qkv_catalog,
):
    only = _rules_only_profile()

    def at(mode, m):
        plan, _ = match_prefix(
            specs.sp_qkv_spec(mode), MatchCtx("gfx950", 256, 4, m, CAPS), only
        )
        return plan.n, plan.half_first, plan.profile_source

    for mode in specs.MODES:
        for m in GRID:
            chunks = m // 256
            want = (4 if chunks % 4 == 0 else 2 if chunks % 2 == 0 else 1, 0, "rule")
            assert at(mode, m) == want, (
                mode,
                m,
            )  # odd chunks -> n=1, chunks % 4 == 2 -> n=2, chunks % 4 == 0 -> n=4; half_first 0; valid for the real plan at all 87 points
    # a rules-only profile has no exact rows: off-grid / out-of-range m stay native
    for m in (512, 1100, 1281, 8448):
        assert match_prefix(
            specs.sp_qkv_spec("dense"), MatchCtx("gfx950", 256, 4, m, CAPS), only
        ) == (None, 0)


def test_off_grid_and_out_of_range_m_stay_native_and_small_m_is_unaffected(
    sp_qkv_catalog,
):
    for m in (256, 512, 768, 1000, 1100, 1281, 8448, 9216, 16384):
        for mode in specs.MODES:
            assert match_prefix(
                specs.sp_qkv_spec(mode), MatchCtx("gfx950", 256, 4, m, CAPS), PROFILE
            ) == (None, 0), (mode, m)
    for m in (16, 32):  # the small rows are exact only: unchanged
        plan, n = match_prefix(
            specs.sp_qkv_spec("dense"),
            MatchCtx("gfx950", 256, 4, m, CAPS | {"gfx950_sc_bypass"}),
            PROFILE,
        )
        assert n == 2 and plan.vid == "sp_qkv.small.v1" and plan.profile_source == "row"
    p24, n24 = match_prefix(
        specs.sp_qkv_spec("dense"),
        MatchCtx("gfx950", 256, 4, 24, CAPS | {"gfx950_sc_bypass"}),
        PROFILE,
    )
    assert (
        n24 == 2 and p24.profile_source == "rule" and p24.src == "r19"
    )  # no exact row at m=24: the r19 rule over 17..32 fuses it
    # another cu / world: no rule
    assert match_prefix(
        specs.sp_qkv_spec("dense"), MatchCtx("gfx950", 304, 4, 1024, CAPS), PROFILE
    ) == (None, 0)


def test_the_dispatch_log_names_row_or_rule_source(sp_qkv_catalog):
    only = _rules_only_profile()
    with capture() as msgs:
        match_prefix(
            specs.sp_qkv_spec("dense"), MatchCtx("gfx950", 256, 4, 1024, CAPS), only
        )
        match_prefix(
            specs.sp_qkv_spec("dense"), MatchCtx("gfx950", 256, 4, 2048, CAPS), PROFILE
        )
        match_prefix(
            specs.sp_qkv_spec("dense"), MatchCtx("gfx950", 256, 4, 1100, CAPS), PROFILE
        )
    disp = [m for m in msgs if "event=dispatch" in m and "n_stages=" in m]
    assert (
        "source=rule rule=dense_chunks_mod4_x1024" in disp[0]
        and "vid=sp_qkv.large.v1" in disp[0]
        and "m=1024" in disp[0]
    )
    assert "source=row" in disp[1] and "rule=" not in disp[1] and "m=2048" in disp[1]
    assert "vid=native" in disp[2] and "source=none" in disp[2]


# ----------------------------------------------------------------------------------------------------------- prebuild enumeration
def _units(profile, **kw):
    return engine.plan_profile(profile, recipe="aot1", tool=TOOL, **kw)


@pytest.fixture
def jit_root(tmp_path, monkeypatch):
    monkeypatch.setenv("AITER_JIT_DIR", str(tmp_path / "jit"))
    return jit.jit_root()


def test_the_shipped_rules_imply_no_extra_builds(jit_root):
    clear_profile_cache()
    prof = profmod.load_profile(PROFILE)
    units, skipped = _units(prof)
    rows_only = parse_profile(
        {"profile": "rows", "rows": [_row_doc(r) for r in prof.rows()]}
    )
    units_rows, _ = _units(rows_only)
    assert skipped == [] and len(units) == 15 == len(
        units_rows
    )  # builds are keyed by mode + build params (no_gridread), never by m, n or half_first
    assert {u.dirname for u in units} == {u.dirname for u in units_rows}


def _row_doc(r):
    return {
        "sig": r.sig,
        "arch": r.arch,
        "cu": r.cu,
        "world": r.world,
        "m": r.m,
        "vid": r.vid,
        "params": dict(r.params),
    }


def test_a_rule_with_a_new_build_param_adds_exactly_the_units_it_implies(jit_root):
    clear_profile_cache()
    prof = profmod.load_profile(PROFILE)
    sig = specs.sp_qkv_spec("dense").prefix(1).signature()
    extra = rule(
        sig,
        3328,
        3328,
        256,
        "sp_qkv.large.v1",
        name="new_build",
        n=1,
        half_first=0,
        no_gridread=0,
    )  # a different -DLG_NO_GRIDREAD: a different build
    doc = {"profile": "x", "rows": [_row_doc(r) for r in prof.rows()], "rules": [extra]}
    units, skipped = _units(parse_profile(doc))
    assert skipped == [] and len(units) == 16
    lg = [u for u in units if u.kind == "sp_qkv.large.v1"]
    assert len(lg) == 3 and {u.label.split()[0] for u in lg} == {"dense", "indexer"}
    # a profile made of rules ONLY still enumerates its builds (and the comm library)
    shipped = json.loads(Path(profmod._resolve(PROFILE)).read_text(encoding="utf-8"))
    only_rules = parse_profile({"profile": "r", "rows": [], "rules": shipped["rules"]})
    u2, _ = _units(only_rules)
    assert (
        sorted(u.kind for u in u2)
        == ["gf_comm"] + ["sp_qkv.large.v1"] * 2 + ["sp_qkv.small.v1"] * 12
    )  # dense + indexer large builds (reuse shares dense) + the 12 small builds the small rules imply (6 r19 + 3 r20 + 3 r21), no small rows
    # filters apply to rules like to rows
    assert (
        _units(only_rules, cu=304)[0] == []
        and _units(only_rules, arch="gfx942")[0] == []
        and len(_units(only_rules, world=4)[0]) == 15
    )


def test_unknown_signature_and_arch_rules_are_skipped_with_a_rule_label(jit_root):
    bad = rule("0" * 16, 1024, 2048, 256, "sp_qkv.large.v1", name="bad")
    other = dict(
        rule(
            specs.sp_qkv_spec("dense").prefix(1).signature(),
            1024,
            2048,
            256,
            "sp_qkv.large.v1",
        ),
        arch="gfx942",
    )
    units, skipped = _units(
        parse_profile({"profile": "x", "rows": [], "rules": [bad, other]})
    )
    assert units == []
    assert any(
        "rule m=1024..2048 step 256" in s and "not a known sp_qkv spec" in s
        for s in skipped
    ) and any("gfx950 only" in s for s in skipped)


def test_87_exact_large_rows_still_collapse_into_the_same_few_build_units():
    """Builds are keyed by mode + build params (no_gridread, src), never by m, n or half_first: the 87 exact P1 rows add no unit beyond the pre-sweep profile's 9
    (the small builds + the 2 large builds), and the rules (fallback layers) add none either.
    """
    units, skipped = _units(PROFILE)
    assert skipped == [] and len(units) == 15
    large = [u for u in units if u.kind == "sp_qkv.large.v1"]
    assert len(large) == 2 and {u.meta["src"] for u in large} == {
        "r12"
    }  # the dense and the indexer build (reuse shares the dense build)
    assert len({u.key for u in large}) == len(large)
    # the same enumeration without ANY large row (rules only) yields the same large builds: rules and rows imply identical units
    with open(profmod._resolve(PROFILE), encoding="utf-8") as f:
        doc = json.load(f)
    only_rules = profmod.parse_profile(
        {
            "profile": "r",
            "rows": [r for r in doc["rows"] if r["vid"] != "sp_qkv.large.v1"],
            "rules": doc["rules"],
        }
    )
    u2, s2 = _units(only_rules)
    assert s2 == [] and {(u.kind, u.key) for u in u2} == {
        (u.kind, u.key) for u in units
    }


# ----------------------------------------------------------------------------------------------------------- small coverage: P2 rows, MP64 (r20), native ranges
SMALL_CAPS = CAPS | {"gfx950_sc_bypass"}
DEFAULT_CAPTURE_M = (
    1,
    2,
    4,
    8,
    12,
    16,
    32,
    64,
    128,
)  # ATOM's default SP4 capture list pads to T = 4, 8, 16, 32, 48, 64, 128, 256, 512 (local m = T / 4)


def _small_at(mode, m, prof=PROFILE):
    return match_prefix(
        specs.sp_qkv_spec(mode), MatchCtx("gfx950", 256, 4, m, SMALL_CAPS), prof
    )


def test_every_default_capture_bucket_resolves_as_designed(sp_qkv_catalog):
    """The buckets ATOM captures by default: P2 (m 1, 2, 4, 8, 12) and m16 / m32 on r19, m64 (T=256) on r20, m128 (T=512) on r21 (MP128); all EXACT rows."""
    for mode in specs.MODES:
        for m in DEFAULT_CAPTURE_M:
            plan, n = _small_at(mode, m)
            assert plan is not None and n == 2 and plan.vid == "sp_qkv.small.v1", (
                mode,
                m,
            )
            assert plan.profile_source == "row", (
                mode,
                m,
            )  # every captured bucket is an EXACT row
            src, mp = (
                ("r21", 128) if m == 128 else ("r20", 64) if m == 64 else ("r19", 32)
            )
            assert (plan.src, plan.mp) == (src, mp), (mode, m)
        assert _small_at(mode, 129) == (
            None,
            0,
        )  # T=516: above MP128, below the large path: native


def test_p2_rows_copy_the_mode_m16_params_and_r20_r21_rows_use_the_mp_params(
    sp_qkv_catalog,
):
    clear_profile_cache()
    prof = profmod.load_profile(PROFILE)
    for mode in specs.MODES:
        sig = specs.sp_qkv_spec(mode).signature()
        p16 = dict(prof.lookup(sig, "gfx950", 256, 4, 16).params)
        assert p16["src"] == "r19"
        for m in specs.SMALL_P2_M:
            assert dict(prof.lookup(sig, "gfx950", 256, 4, m).params) == p16, (
                mode,
                m,
            )  # R13 P2: a copy of the m=16 params
        assert (
            dict(prof.lookup(sig, "gfx950", 256, 4, 64).params)
            == specs.small_r20_params(mode)
            == {"src": "r20", "sk": 3, "nh": 1, "no_credit": 1, "rx_bypass": 1}
        )
        assert (
            dict(prof.lookup(sig, "gfx950", 256, 4, 128).params)
            == specs.small_r21_params(mode)
            == {
                "src": "r21",
                "sk": 3,
                "nh": 1,
                "no_credit": 1,
                "rx_bypass": 1,
            }  # B_HEADS=1 (432 / 480 B WGs); nh=2 is the A/B, selected by params
        )
        # nothing but the P2 / m16 / m32 / m64 / m128 rows exists for the small spec
        assert sorted(r.m for r in prof.rows() if r.sig == sig) == [
            1,
            2,
            4,
            8,
            12,
            16,
            32,
            64,
            128,
        ]


def test_mp64_rule_covers_33_to_64_on_r20_and_the_mp128_rule_65_to_128_on_r21(
    sp_qkv_catalog,
):
    for mode in specs.MODES:
        for lo, hi, src, mp in ((33, 64, "r20", 64), (65, 128, "r21", 128)):
            for m in range(lo, hi + 1):
                plan, n = _small_at(mode, m)
                assert (
                    plan is not None
                    and n == 2
                    and plan.src == src
                    and plan.mp == mp
                    and plan.ctx_m == m
                ), (mode, m)
                assert plan.profile_source == ("row" if m == hi else "rule"), (
                    mode,
                    m,
                )  # the exact row sits at the top of each range
    with capture() as msgs:
        _small_at("dense", 48)
        _small_at("dense", 64)
        _small_at("indexer", 100)
        _small_at("indexer", 128)
    disp = [m for m in msgs if "event=dispatch" in m and "n_stages=2" in m]
    assert (
        "source=rule rule=dense_mp64_m33-64" in disp[0]
        and "m=48" in disp[0]
        and "'src': 'r20'" in disp[0]
    )
    assert "source=row" in disp[1] and "rule=" not in disp[1]
    assert (
        "source=rule rule=indexer_mp128_m65-128" in disp[2]
        and "m=100" in disp[2]
        and "'src': 'r21'" in disp[2]
    )
    assert "source=row" in disp[3] and "rule=" not in disp[3] and "m=128" in disp[3]


def test_everything_else_is_explicitly_native(sp_qkv_catalog):
    """m 129..255 (T 516..1020) and 256..1023 have no fused path (every m <= 128 fuses: see the test below)."""
    clear_profile_cache()
    prof = profmod.load_profile(PROFILE)
    ranges = prof.meta["native_m_ranges"]
    assert [(r["m_min"], r["m_max"]) for r in ranges] == [
        (129, 255)
    ] and "T 516..1020" in ranges[0]["note"]
    for mode in specs.MODES:
        for m in list(range(129, 256)) + [256, 512, 768, 1000, 1023, 1025, 8193]:
            assert _small_at(mode, m) == (None, 0), (
                mode,
                m,
            )  # the full spec: neither the 2-stage nor the 1-stage (large) prefix has an entry
    # the documented native range is also free of rows AND rules
    assert not any(129 <= r.m <= 255 for r in prof.rows())
    assert all(not (r.m_min <= 255 and r.m_max >= 129) for r in prof.rules())


def test_every_local_m_up_to_128_fuses_and_the_shape_switches_follow_their_m_limits(
    sp_qkv_catalog,
):
    """Exact rows at m 1, 2, 4, 8, 12, 16, 32, 64, 128; r19 rules fill 1..16 (copy of the m16 params) and 17..32 (copy of the m32 params), r20 fills 33..64, r21 65..128: so eager
    decode and custom capture lists at ANY m <= 128 fuse. push8 (an M <= 16 shape) must never be applied above 16, b_rowsplit / push_halves (M > 16 shapes) never at or below 16.
    """
    clear_profile_cache()
    prof = profmod.load_profile(PROFILE)
    exact = {1, 2, 4, 8, 12, 16, 32, 64, 128}
    for mode in specs.MODES:
        sig = specs.sp_qkv_spec(mode).signature()
        p16 = dict(prof.lookup(sig, "gfx950", 256, 4, 16).params)
        p32 = dict(prof.lookup(sig, "gfx950", 256, 4, 32).params)
        sources = {"row": 0, "rule": 0}
        for m in range(1, 129):
            plan, n = _small_at(mode, m)
            assert plan is not None and n == 2 and plan.vid == "sp_qkv.small.v1", (
                mode,
                m,
            )
            assert plan.profile_source == ("row" if m in exact else "rule"), (mode, m)
            sources[plan.profile_source] += 1
            want = (
                p16
                if m <= 16
                else (
                    p32
                    if m <= 32
                    else (
                        specs.small_r20_params(mode)
                        if m <= 64
                        else specs.small_r21_params(mode)
                    )
                )
            )
            assert dict(plan.params) == want, (mode, m)
            src, mp = (
                ("r19", 32) if m <= 32 else ("r20", 64) if m <= 64 else ("r21", 128)
            )
            assert (plan.src, plan.mp) == (src, mp), (mode, m)
            if plan.params.get("push8"):
                assert m <= 16, (mode, m, "push8 applied above 16")
            if plan.params.get("b_rowsplit") or plan.params.get("push_halves"):
                assert 16 < m <= 32, (
                    mode,
                    m,
                    "an M > 16 shape applied outside its r19 range",
                )
        assert sources == {
            "row": 9,
            "rule": 119,
        }  # 128 values: 9 exact rows, every other m through a rule
        # the shape limits are enforced by the plan itself (not only by the profile): the copy params at the wrong M are refused
        from aiter.gemm_fusion import ContractError

        v = sys.modules["aiter.ops.gemm_fusion.sp_qkv.variants"]
        if p16.get("push8"):
            with pytest.raises(ContractError, match="push8 build supports M<=16"):
                v.SpQkvSmall(
                    specs.sp_qkv_spec(mode),
                    p16,
                    MatchCtx("gfx950", 256, 4, 17, SMALL_CAPS),
                )
        if p32.get("b_rowsplit"):
            with pytest.raises(ContractError, match="b_rowsplit build supports M>16"):
                v.SpQkvSmall(
                    specs.sp_qkv_spec(mode),
                    p32,
                    MatchCtx("gfx950", 256, 4, 16, SMALL_CAPS),
                )
        # a src r19 plan ends at M=32, r20 at 64, r21 at 128
        with pytest.raises(
            ContractError, match=r"\(src r19\) supports local M 1\.\.32, got 33"
        ):
            v.SpQkvSmall(
                specs.sp_qkv_spec(mode), p32, MatchCtx("gfx950", 256, 4, 33, SMALL_CAPS)
            )
        with pytest.raises(
            ContractError, match=r"\(src r20\) supports local M 1\.\.64, got 65"
        ):
            v.SpQkvSmall(
                specs.sp_qkv_spec(mode),
                specs.small_r20_params(mode),
                MatchCtx("gfx950", 256, 4, 65, SMALL_CAPS),
            )
        with pytest.raises(
            ContractError, match=r"\(src r21\) supports local M 1\.\.128, got 129"
        ):
            v.SpQkvSmall(
                specs.sp_qkv_spec(mode),
                specs.small_r21_params(mode),
                MatchCtx("gfx950", 256, 4, 129, SMALL_CAPS),
            )


def test_p2_rows_add_no_unit_and_the_r20_r21_rows_and_rules_imply_six_small_units(
    jit_root,
):
    clear_profile_cache()
    prof = profmod.load_profile(PROFILE)
    units, skipped = _units(prof)
    smalls = [u for u in units if u.kind == "sp_qkv.small.v1"]
    assert (
        skipped == []
        and len(smalls) == 12
        and sorted(u.meta["src"] for u in smalls)
        == ["r19"] * 6 + ["r20"] * 3 + ["r21"] * 3
    )
    shipped = json.loads(Path(profmod._resolve(PROFILE)).read_text(encoding="utf-8"))
    p2 = set(specs.SMALL_P2_M)
    no_p2 = parse_profile(
        {
            "profile": "x",
            "rows": [
                _row_doc(r)
                for r in prof.rows()
                if not (r.vid == "sp_qkv.small.v1" and r.m in p2)
            ],
            "rules": shipped["rules"],
        }
    )
    u2, _ = _units(no_p2)
    assert {(u.kind, u.key) for u in u2} == {
        (u.kind, u.key) for u in units
    }  # the P2 rows (copies of m16) add no unit
    # the r20 / r21 units compile their own source with the bypass loads on
    for src in ("r20", "r21"):
        group = [u for u in smalls if u.meta["src"] == src]
        assert len(group) == 3
        for u in group:
            assert (
                "-DSM_RX_BYPASS=1" in u.meta["dev"]
                and "-DSM_T1_B_HEADS=1" in u.meta["dev"]
            )
            cmd = [
                a.replace("\\", "/")
                for a in jit.device_cmd_one(src, u.meta["dev"], TOOL, "/out", "sm_ran")
            ]
            assert any(a.endswith(f"small/{src}/device/ran_kernel.cu") for a in cmd)
    # without the r20 / r21 rows and rules the units disappear: they are implied by BOTH the exact rows and the rules
    for drop, left in (
        ({"r20", "r21"}, ["r19"] * 6),
        ({"r21"}, ["r19"] * 6 + ["r20"] * 3),
    ):
        only = parse_profile(
            {
                "profile": "x",
                "rows": [
                    _row_doc(r)
                    for r in prof.rows()
                    if dict(r.params).get("src") not in drop
                ],
                "rules": (
                    []
                    if "r20" in drop
                    else [
                        r_
                        for r_ in shipped["rules"]
                        if r_["params"].get("src") != "r21"
                    ]
                ),
            }
        )
        assert (
            sorted(
                u.meta["src"] for u in _units(only)[0] if u.kind == "sp_qkv.small.v1"
            )
            == left
        )
    # the B_HEADS=2 A/B is a different BUILD (a different -D), hence one more unit per mode that selects it, and its own owner
    row128 = next(
        _row_doc(r)
        for r in prof.rows()
        if r.vid == "sp_qkv.small.v1"
        and r.m == 128
        and r.sig == specs.sp_qkv_spec("dense").signature()
    )
    ab = dict(row128, m=127, params=dict(row128["params"], nh=2))
    u3, s3 = _units(
        parse_profile(
            {
                "profile": "x",
                "rows": [_row_doc(r) for r in prof.rows()] + [ab],
                "rules": shipped["rules"],
            }
        )
    )
    assert s3 == [] and len(u3) == len(units) + 1
    ab_unit = next(u for u in u3 if "-DSM_T1_B_HEADS=2" in u.meta.get("dev", []))
    assert ab_unit.meta["src"] == "r21" and ab_unit.key not in {u.key for u in units}
