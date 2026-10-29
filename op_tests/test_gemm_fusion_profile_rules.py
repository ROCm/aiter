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
    assert len(prof) == 30 + 93 and len(prof.rules()) == 24
    large_rules = [r for r in prof.rules() if r.vid == "sp_qkv.large.v1"]
    small_rules = [r for r in prof.rules() if r.vid == "sp_qkv.small.v1"]
    assert (
        len(large_rules) == 9 and len(small_rules) == 15
    )  # 3 chunks-mod-4 layers x 3 modes; per mode 5 small ranges: r19 1..16, r19 17..32, r20 (MP64) 33..64, r22 (MP128) 65..128, r23 (MP256) 129..255
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
            and 512 <= r.m_min <= r.m_max <= 8192
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
            (129, 255),
        ) and r.m_multiple == 1
        assert pb.row_mode(r) == sigs2[r.sig]
        assert (
            ("mp64" in r.name) == (r.m_min == 33)
            and ("mp128" in r.name) == (r.m_min == 65)
            and ("mp256" in r.name) == (r.m_min == 129)
        )
        assert dict(r.params)["src"] == {33: "r20", 65: "r22", 129: "r23"}.get(
            r.m_min, "r19"
        )
        if r.m_min == 33:
            assert dict(r.params) == specs.small_r20_params(sigs2[r.sig])
        if r.m_min == 65:
            assert dict(r.params) == specs.small_r22_params(
                sigs2[r.sig], specs.SMALL_R22_PROFILE_NH
            )
        if r.m_min == 129:
            assert dict(r.params) == specs.small_r23_params(
                sigs2[r.sig], specs.SMALL_R23_PROFILE_NH
            )
    assert {r.sig for r in small_rules} == set(sigs2) and all(
        sum(1 for r in small_rules if r.sig == sg) == 5 for sg in sigs2
    )  # five disjoint ranges per mode; a large rule never shares a signature with a small row/rule
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
    # the rules also cover the 512 and 768 r18 pad buckets (m_min 512: at 512 the chunks%4==2 layer applies n=2, at 768 3 chunks -> n=1; the exact rows say n=1
    # as well: shadowed by the rows)
    for mode in specs.MODES:
        assert at(mode, 768) == (1, 0, "rule"), (mode, 768)
        assert at(mode, 512) == (2, 0, "rule"), (mode, 512)
    # a rules-only profile has no exact rows: off-grid / out-of-range m stay native
    for m in (1100, 1281, 8448):
        assert match_prefix(
            specs.sp_qkv_spec("dense"), MatchCtx("gfx950", 256, 4, m, CAPS), only
        ) == (None, 0)


def test_off_grid_and_out_of_range_m_stay_native_and_small_m_is_unaffected(
    sp_qkv_catalog,
):
    for m in (
        257,
        1000,
        1100,
        1281,
        8448,
        9216,
        16384,
    ):  # match_prefix: exact rows / rules only (512 and 768 are exact rows now; bucket_for_m pads the rest)
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


def test_the_shipped_rules_imply_no_build_beyond_the_rows(jit_root):
    clear_profile_cache()
    prof = profmod.load_profile(PROFILE)
    units, skipped = _units(prof)
    rows_only = parse_profile(
        {"profile": "rows", "rows": [_row_doc(r) for r in prof.rows()]}
    )
    units_rows, _ = _units(rows_only)
    # builds are keyed by mode + build params (no_gridread, src), never by m, n or half_first: the rules imply no build beyond the rows' (the r23 owners have their bucket rows
    # at m=256; the r18 bucket rows are the src-r18 builds and the rules never mention r18)
    assert skipped == [] and len(units) == 20 == len(units_rows)
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
        **({"pad_min": r.pad_min} if r.pad_min is not None else {}),
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
    assert skipped == [] and len(units) == 21
    lg = [u for u in units if u.kind == "sp_qkv.large.v1"]
    assert len(lg) == 5 and {u.label.split()[0] for u in lg} == {"dense", "indexer"}
    # a profile made of rules ONLY still enumerates its builds (and the comm library)
    shipped = json.loads(Path(profmod._resolve(PROFILE)).read_text(encoding="utf-8"))
    only_rules = parse_profile({"profile": "r", "rows": [], "rules": shipped["rules"]})
    u2, _ = _units(only_rules)
    assert (
        sorted(u.kind for u in u2)
        == ["gf_comm"] + ["sp_qkv.large.v1"] * 2 + ["sp_qkv.small.v1"] * 15
    )  # dense + indexer large builds (reuse shares dense) + the 15 small builds the small rules imply (6 r19 + 3 r20 + 3 r22 + 3 r23), no small rows
    # filters apply to rules like to rows
    assert (
        _units(only_rules, cu=304)[0] == []
        and _units(only_rules, arch="gfx942")[0] == []
        and len(_units(only_rules, world=4)[0]) == 18
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
    """Builds are keyed by mode + build params (no_gridread, src), never by m, n or half_first: the 87 exact P1 rows + the 3 r13 grid-bucket rows add no unit beyond the
    pre-sweep profile's 2 r13 builds, and the r18 bucket rows add the 2 route-B builds (dense + indexer) the shipped profile needs.
    """
    units, skipped = _units(PROFILE)
    assert skipped == [] and len(units) == 20
    large = [u for u in units if u.kind == "sp_qkv.large.v1"]
    assert len(large) == 4 and {u.meta["src"] for u in large} == {
        "r13",
        "r18",
    }  # the dense and the indexer build of each src (reuse shares the dense build)
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
    rule_keys = {(u.kind, u.key) for u in u2}
    full_keys = {(u.kind, u.key) for u in units}
    assert s2 == [] and len(u2) == 18  # 15 small + the 2 r13 rule builds + the comm
    assert rule_keys < full_keys and full_keys - rule_keys == {
        (u.kind, u.key) for u in units if u.meta.get("src") == "r18"
    }  # the r18 builds come from the bucket ROWS only


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
    """The buckets ATOM captures by default: P2 (m 1, 2, 4, 8, 12) and m16 / m32 on r19, m64 (T=256) on r20, m128 (T=512) on r22 (MP128); all EXACT rows."""
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
                ("r22", 128) if m == 128 else ("r20", 64) if m == 64 else ("r19", 32)
            )
            assert (plan.src, plan.mp) == (src, mp), (mode, m)
        plan256, _n = _small_at(mode, 256)
        assert (plan256.src, plan256.mp, plan256.profile_source) == (
            "r23",
            256,
            "row",
        )  # the r23 bucket row (T=1024); m 129..255 is the r23 rule
        assert _small_at(mode, 257) == (
            None,
            0,
        )  # above MP256, below the large path: native


def test_p2_rows_copy_the_mode_m16_params_and_r20_r22_rows_use_the_mp_params(
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
            == specs.small_r22_params(mode, specs.SMALL_R22_PROFILE_NH)
            == {
                "src": "r22",
                "sk": 3,
                "nh": 1,
                "no_credit": 1,
                "rx_bypass": 1,
                "b_quad": 1,
                "gemm_bn128": 0,
                "rx3": 1,
                "p_nt": 2,
                "push_passtrip": 1,
            }  # r22 {nh 1, B-quad, BN64, 3-slot Rx, P sc0 sc1 nt, passtrip}: all GPU gates passed, T512 B/C 1.28-1.33x of native vs r21 nh=2 at 1.09-1.16x; r21 / BN128 / p_nt 0 / passtrip 0 are A/Bs by params
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
            256,
        ]


def test_mp64_rule_covers_33_to_64_on_r20_and_the_mp128_rule_65_to_128_on_r22(
    sp_qkv_catalog,
):
    for mode in specs.MODES:
        for lo, hi, src, mp in ((33, 64, "r20", 64), (65, 128, "r22", 128)):
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
        and "'src': 'r22'" in disp[2]
    )
    assert "source=row" in disp[3] and "rule=" not in disp[3] and "m=128" in disp[3]


def test_everything_else_is_explicitly_native(sp_qkv_catalog):
    """Every local m 1..255 fuses (r19 / r20 / r22 / r23: see the tests around); local m 256..1023 has no fused path (the large path starts at m=1024)."""
    clear_profile_cache()
    prof = profmod.load_profile(PROFILE)
    assert (
        prof.meta["native_m_ranges"] == []
    )  # nothing inside the small envelope is native
    for mode in specs.MODES:
        for m in range(1, 257):
            assert _small_at(mode, m)[0] is not None, (mode, m)
        for m in [
            257,
            300,
            1000,
            1023,
            1025,
            8193,
        ]:  # match_prefix: exact rows / rules only
            assert _small_at(mode, m) == (None, 0), (
                mode,
                m,
            )  # the full spec: neither the 2-stage nor the 1-stage (large) prefix has an entry
    # the small rows end at m=256 (the r23 bucket row), the small rules at m=255 (the large rules start at 1024)
    assert all(r.m <= 256 for r in prof.rows() if r.vid == "sp_qkv.small.v1")
    assert all(r.m_max <= 255 for r in prof.rules() if r.vid == "sp_qkv.small.v1")
    assert all(r.m_min >= 512 for r in prof.rules() if r.vid == "sp_qkv.large.v1")


def test_every_local_m_up_to_128_fuses_and_the_shape_switches_follow_their_m_limits(
    sp_qkv_catalog,
):
    """Exact rows at m 1, 2, 4, 8, 12, 16, 32, 64, 128; r19 rules fill 1..16 (copy of the m16 params) and 17..32 (copy of the m32 params), r20 fills 33..64, r22 65..128: so eager
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
                        else specs.small_r22_params(mode, specs.SMALL_R22_PROFILE_NH)
                    )
                )
            )
            assert dict(plan.params) == want, (mode, m)
            src, mp = (
                ("r19", 32) if m <= 32 else ("r20", 64) if m <= 64 else ("r22", 128)
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
        # a src r19 plan ends at M=32, r20 at 64, r21 / r22 at 128
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
            ContractError, match=r"\(src r22\) supports local M 1\.\.128, got 129"
        ):
            v.SpQkvSmall(
                specs.sp_qkv_spec(mode),
                specs.small_r22_params(mode, specs.SMALL_R22_PROFILE_NH),
                MatchCtx("gfx950", 256, 4, 129, SMALL_CAPS),
            )


def test_p2_rows_add_no_unit_and_the_r20_r22_rows_and_rules_imply_six_small_units(
    jit_root,
):
    clear_profile_cache()
    prof = profmod.load_profile(PROFILE)
    units, skipped = _units(prof)
    smalls = [u for u in units if u.kind == "sp_qkv.small.v1"]
    assert (
        skipped == []
        and len(smalls) == 15
        and sorted(u.meta["src"] for u in smalls)
        == ["r19"] * 6 + ["r20"] * 3 + ["r22"] * 3 + ["r23"] * 3
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
    # the r20 / r22 / r23 units compile their own source with the bypass loads on
    for src in ("r20", "r22", "r23"):
        group = [u for u in smalls if u.meta["src"] == src]
        assert len(group) == 3
        for u in group:
            assert (
                "-DSM_RX_BYPASS=1" in u.meta["dev"]
                and "-DSM_T1_B_HEADS=1"
                in u.meta["dev"]  # r20 has only NH1; the profile's r22 nh is 1 (B-quad)
            )
            if src == "r22":  # the profile arm: B-quad on, BN64, 3-slot Rx
                assert {
                    "-DSM_B_QUAD=1",
                    "-DSM_GEMM_BN128=0",
                    "-DSM_RX3=1",
                    "-DSM_P_NT=2",
                    "-DSM_PUSH_PASSTRIP=1",
                    "-DSM_SK_CFG=3",
                } <= set(u.meta["dev"])
            if src == "r23":  # the profile arm: BN64 PFK4, p_nt 2, all-passtrip
                assert {
                    "-DSM_B_QUAD=1",
                    "-DSM_GEMM_BN128=0",
                    "-DSM_GEMM_PFK=4",
                    "-DSM_RX3=1",
                    "-DSM_P_NT=2",
                    "-DSM_PUSH_PASSTRIP=1",
                } <= set(u.meta["dev"])
            cmd = [
                a.replace("\\", "/")
                for a in jit.device_cmd_one(src, u.meta["dev"], TOOL, "/out", "sm_ran")
            ]
            assert any(a.endswith(f"small/{src}/device/ran_kernel.cu") for a in cmd)
    # without the r20 / r22 / r23 rows and rules the units disappear: they are implied by BOTH the exact rows and the rules
    for drop, left in (
        ({"r20", "r22", "r23"}, ["r19"] * 6),
        ({"r22", "r23"}, ["r19"] * 6 + ["r20"] * 3),
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
                        if r_["params"].get("src") not in drop
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
    # the B_HEADS=2 A/B (the profile uses nh=1) is a different BUILD (a different -D), hence one more unit per mode that selects it, and its own owner
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
    ab_unit = next(
        u
        for u in u3
        if u.key not in {x.key for x in units} and u.kind == "sp_qkv.small.v1"
    )
    assert ab_unit.meta["src"] == "r22" and "-DSM_T1_B_HEADS=2" in ab_unit.meta["dev"]


def test_with_both_mp128_flags_off_the_profile_has_no_mp128_source_and_m65_to_255_is_native(
    monkeypatch, jit_root, sp_qkv_catalog
):
    """The no-MP128 layout (rollback / A/B): SMALL_R21_IN_PROFILE and SMALL_R22_IN_PROFILE both False reproduces the profile of the time no MP128 source was selected: 111 rows,
    18 rules, 9 small owners, a 408 MiB arena, 12 prebuild units, m 65..255 native."""
    from collections import Counter

    monkeypatch.setattr(
        specs, "LARGE_R18_IN_PROFILE", False
    )  # these tests pin the pre-r18 large band (the shipped r18 route-B buckets are the OUTER layout)
    monkeypatch.setattr(
        specs, "SMALL_R23_IN_PROFILE", False
    )  # the layouts below are the ones WITHOUT the MP256 slot
    monkeypatch.setattr(specs, "SMALL_R21_IN_PROFILE", False)
    monkeypatch.setattr(specs, "SMALL_R22_IN_PROFILE", False)
    doc = json.loads(json.dumps(gen_profile.build_profile()))
    prof = parse_profile(doc)
    assert len(prof) == 24 + 90 and len(prof.rules()) == 18 and len(doc["rows"]) == 114
    assert not {"r21", "r22"} & {dict(e.params).get("src") for e in prof.entries()}
    wb = doc["window_budget"]
    assert (
        wb["small_owners"] == 9
        and wb["max_windows"] == 46
        and Counter(wb["small_window_sizes"]) == {4 << 20: 18}
    )
    assert sum(wb["shared"].values()) + sum(wb["small_window_sizes"]) == 408 << 20
    assert [(r["m_min"], r["m_max"]) for r in doc["native_m_ranges"]] == [(65, 255)]
    for mode in specs.MODES:
        for m in (65, 100, 128, 255):
            assert _small_at(mode, m, prof) == (None, 0), (mode, m)
        assert _small_at(mode, 64, prof)[0].src == "r20"
    units, skipped = _units(prof)
    assert skipped == [] and len(units) == 12


def test_with_only_the_r21_flag_on_r21_nh2_takes_the_mp128_slot_with_8mib_views(
    monkeypatch, jit_root, sp_qkv_catalog
):
    """r21 (nh=2) stays in the code, selectable: the layout that shipped before r22 (114 rows, 21 rules, 12 owners, 21 x 4 MiB + 3 x 8 MiB windows, a 444 MiB arena, 15 units)."""
    from collections import Counter

    monkeypatch.setattr(
        specs, "LARGE_R18_IN_PROFILE", False
    )  # pre-r18 large band, so the old numbers apply
    monkeypatch.setattr(
        specs, "SMALL_R23_IN_PROFILE", False
    )  # the layouts below are the ones WITHOUT the MP256 slot
    monkeypatch.setattr(specs, "SMALL_R22_IN_PROFILE", False)
    monkeypatch.setattr(specs, "SMALL_R21_IN_PROFILE", True)
    doc = json.loads(json.dumps(gen_profile.build_profile()))
    prof = parse_profile(doc)
    srcs = {dict(e.params).get("src") for e in prof.entries()}
    assert "r21" in srcs and "r22" not in srcs
    assert len(prof) == 27 + 90 and len(prof.rules()) == 21
    wb = doc["window_budget"]
    assert wb["small_owners"] == 12 and wb["max_windows"] == 52
    assert Counter(wb["small_window_sizes"]) == {4 << 20: 21, 8 << 20: 3}
    assert sum(wb["shared"].values()) + sum(wb["small_window_sizes"]) == 444 << 20
    assert [(r["m_min"], r["m_max"]) for r in doc["native_m_ranges"]] == [(129, 255)]
    assert "r21 MP128" in doc["native_m_ranges"][0]["note"]
    for mode in specs.MODES:
        for m, source in ((65, "rule"), (100, "rule"), (128, "row")):
            plan, n = _small_at(mode, m, prof)
            assert (
                plan is not None
                and n == 2
                and (plan.src, plan.mp, plan.rx_slots, plan.profile_source)
                == ("r21", 128, 4, source)
            ), (mode, m)
            assert (
                dict(plan.params)
                == specs.small_r21_params(mode, specs.SMALL_R21_PROFILE_NH)
                and plan.params["nh"] == 2
            )
            assert plan.resources()[0].bytes == 8 << 20
    units, skipped = _units(prof)
    smalls = [u for u in units if u.kind == "sp_qkv.small.v1"]
    assert (
        skipped == []
        and len(units) == 15
        and sorted(u.meta["src"] for u in smalls)
        == ["r19"] * 6 + ["r20"] * 3 + ["r21"] * 3
    )
    assert all(
        "-DSM_T1_B_HEADS=2" in u.meta["dev"] for u in smalls if u.meta["src"] == "r21"
    )


# ----------------------------------------------------------------------------------------------------------- r22 (tuned MP128): selectable, NOT in the shipped profile
def test_the_shipped_profile_has_r22_at_the_mp128_slot_and_no_r21(sp_qkv_catalog):
    clear_profile_cache()
    prof = profmod.load_profile(PROFILE)
    assert specs.SMALL_R22_IN_PROFILE is True and specs.SMALL_R21_IN_PROFILE is False
    srcs = [dict(e.params).get("src") for e in prof.entries()]
    assert (
        srcs.count("r22") == 3 + 3 and "r21" not in srcs
    )  # 3 exact rows (m=128) + 3 rules (65..128), one per mode
    for mode in specs.MODES:
        plan, _n = _small_at(mode, 128)
        assert (plan.src, plan.profile_source, plan.rx_slots) == ("r22", "row", 3)
        assert dict(plan.params) == specs.small_r22_params(mode, 1)
        assert (
            plan.resources()[0].bytes == 4 << 20
        )  # ONE 4 MiB Rx view: the r21 8 MiB view is gone from the shipped budget
    wb = prof.meta["window_budget"]
    assert set(wb["small_window_sizes"]) == {
        4 << 20,
        8 << 20,
    }  # the 8 MiB views are the r23 (MP256) owners; the r22 Rx is one 4 MiB view


def test_with_the_r22_flag_on_both_flags_r22_wins_with_a_4mib_view_budget_and_units(
    monkeypatch, jit_root, sp_qkv_catalog
):
    from collections import Counter

    monkeypatch.setattr(
        specs, "LARGE_R18_IN_PROFILE", False
    )  # pre-r18 large band, so the old numbers apply
    monkeypatch.setattr(
        specs, "SMALL_R23_IN_PROFILE", False
    )  # the layouts below are the ones WITHOUT the MP256 slot
    monkeypatch.setattr(specs, "SMALL_R22_IN_PROFILE", True)
    doc = json.loads(json.dumps(gen_profile.build_profile()))
    prof = parse_profile(doc)
    srcs = {dict(e.params).get("src") for e in prof.entries()}
    assert "r22" in srcs and "r21" not in srcs  # r22 REPLACES r21 in the MP128 slot
    assert (
        len(prof) == 27 + 90
        and len(prof.rules()) == 21
        and [(r["m_min"], r["m_max"]) for r in doc["native_m_ranges"]] == [(129, 255)]
    )
    wb = doc["window_budget"]
    assert wb["small_owners"] == 12 and wb["max_windows"] == 52
    assert Counter(wb["small_window_sizes"]) == {
        4 << 20: 24
    }  # no 8 MiB Rx view any more: every small window is one 4 MiB granule
    assert sum(wb["shared"].values()) + sum(wb["small_window_sizes"]) == 432 << 20
    for mode in specs.MODES:
        for m, source in ((65, "rule"), (100, "rule"), (128, "row")):
            plan, n = _small_at(mode, m, prof)
            assert (
                plan is not None
                and n == 2
                and (plan.src, plan.mp, plan.rx_slots, plan.profile_source)
                == ("r22", 128, 3, source)
            ), (mode, m)
            assert dict(plan.params) == specs.small_r22_params(
                mode, specs.SMALL_R22_PROFILE_NH
            )
        assert (
            _small_at(mode, 129, prof) == (None, 0)
            and _small_at(mode, 64, prof)[0].src == "r20"
        )
    units, skipped = _units(prof)
    smalls = [u for u in units if u.kind == "sp_qkv.small.v1"]
    assert (
        skipped == []
        and len(units) == 15
        and sorted(u.meta["src"] for u in smalls)
        == ["r19"] * 6 + ["r20"] * 3 + ["r22"] * 3
    )
    for u in (u for u in smalls if u.meta["src"] == "r22"):
        assert {
            "-DSM_B_QUAD=1",
            "-DSM_GEMM_BN128=0",
            "-DSM_RX3=1",
            "-DSM_T1_B_HEADS=1",
            "-DSM_RX_BYPASS=1",
        } <= set(u.meta["dev"])
        cmd = [
            a.replace("\\", "/")
            for a in jit.device_cmd_one("r22", u.meta["dev"], TOOL, "/out", "sm_ran")
        ]
        assert any(a.endswith("small/r22/device/ran_kernel.cu") for a in cmd)
    # nh=2 (B-quad with two heads per B WG) is another build
    monkeypatch.setattr(specs, "SMALL_R22_PROFILE_NH", 2)
    doc2 = json.loads(json.dumps(gen_profile.build_profile()))
    u2, _ = _units(parse_profile(doc2))
    assert all(
        "-DSM_T1_B_HEADS=2" in u.meta["dev"]
        for u in u2
        if u.kind == "sp_qkv.small.v1" and u.meta["src"] == "r22"
    )
    # both flags on: r22 wins, r21 is not generated
    monkeypatch.setattr(specs, "SMALL_R21_IN_PROFILE", True)
    assert "r21" not in {
        r["params"]["src"] for r in gen_profile.build_profile()["rows"]
    }


def test_a_row_can_select_r22_directly_and_the_ab_knobs_make_other_owners_and_units(
    monkeypatch, jit_root, sp_qkv_catalog
):
    clear_profile_cache()
    prof = profmod.load_profile(PROFILE)
    shipped = json.loads(Path(profmod._resolve(PROFILE)).read_text(encoding="utf-8"))
    sig = specs.sp_qkv_spec("dense").signature()
    row128 = next(
        _row_doc(r)
        for r in prof.rows()
        if r.vid == "sp_qkv.small.v1" and r.m == 128 and r.sig == sig
    )
    base_units, _ = _units(prof)
    extra = [
        dict(
            row128, m=127, params={**specs.small_r22_params("dense"), "gemm_bn128": 1}
        ),  # the BN128 arm (the source default): another build than the profile's BN64
        dict(
            row128,
            m=126,
            params={
                **specs.small_r22_params("dense"),
                "b_quad": 0,
                "gemm_bn128": 0,
                "rx3": 0,
            },
        ),  # the r21-like A/B build
    ]
    p2 = parse_profile(
        {
            "profile": "x",
            "rows": [_row_doc(r) for r in prof.rows()] + extra,
            "rules": shipped["rules"],
        }
    )
    units, skipped = _units(p2)
    assert skipped == [] and len(units) == len(base_units) + 2
    base_keys = {u.key for u in base_units}
    r22 = [u for u in units if u.meta.get("src") == "r22" and u.key not in base_keys]
    assert len(r22) == 2 and {"-DSM_RX3=0" in u.meta["dev"] for u in r22} == {
        True,
        False,
    }
    plan127, _n = match_prefix(
        specs.sp_qkv_spec("dense"), MatchCtx("gfx950", 256, 4, 127, SMALL_CAPS), p2
    )
    assert (plan127.src, plan127.rx_slots, plan127.resources()[0].bytes) == (
        "r22",
        3,
        4 << 20,
    )
    plan126, _n = match_prefix(
        specs.sp_qkv_spec("dense"), MatchCtx("gfx950", 256, 4, 126, SMALL_CAPS), p2
    )
    assert (plan126.src, plan126.rx_slots, plan126.resources()[0].bytes) == (
        "r22",
        4,
        8 << 20,
    )  # rx3=0: the 8 MiB view again
    assert plan127.owner != plan126.owner


# ----------------------------------------------------------------------------------------------------------- r23 (MP256): selectable, NOT in the shipped profile
R23_KEY_SITES = [f"model.layers.{i}.self_attn" for i in range(62)] + [
    f"mtp.layers.{i}.self_attn" for i in range(3)
]


def test_the_shipped_profile_has_the_r23_rule_over_m129_to_255_and_no_native_range(
    sp_qkv_catalog,
):
    clear_profile_cache()
    prof = profmod.load_profile(PROFILE)
    assert specs.SMALL_R23_IN_PROFILE is True and specs.SMALL_R22_IN_PROFILE is True
    r23 = [e for e in prof.entries() if dict(e.params).get("src") == "r23"]
    rules = [e for e in r23 if hasattr(e, "m_min")]
    rows = [e for e in r23 if not hasattr(e, "m_min")]
    assert len(rules) == 3 and all(e.m_min == 129 and e.m_max == 255 for e in rules)
    assert len(rows) == 3 and all(
        e.m == 256 for e in rows
    )  # one RULE per mode over 129..255 + one BUCKET ROW per mode at 256
    for mode in specs.MODES:
        for m in (129, 200, 255):
            plan, _n = _small_at(mode, m)
            assert (plan.src, plan.mp, plan.rx_slots, plan.profile_source) == (
                "r23",
                256,
                3,
                "rule",
            ), (mode, m)
            assert dict(plan.params) == specs.small_r23_params(
                mode, specs.SMALL_R23_PROFILE_NH
            )
            assert plan.resources()[0].bytes == 8 << 20  # the 8 MiB Rx view
        assert _small_at(mode, 256)[0].profile_source == "row"  # the r23 bucket row
        assert _small_at(mode, 257) == (
            None,
            0,
        )  # m 257..1023 stays native; the large path starts at 1024
    assert prof.meta["native_m_ranges"] == []


def test_with_the_r23_flag_on_a_rule_covers_129_to_255_with_8mib_views_units_and_unique_keys(
    monkeypatch, jit_root, sp_qkv_catalog
):
    from collections import Counter

    from aiter.gemm_fusion import callid

    monkeypatch.setattr(
        specs, "LARGE_R18_IN_PROFILE", False
    )  # pre-r18 large band, so the old numbers apply
    monkeypatch.setattr(specs, "SMALL_R23_IN_PROFILE", True)
    doc = json.loads(json.dumps(gen_profile.build_profile()))
    prof = parse_profile(doc)
    assert (
        len(prof) == 30 + 90 and len(prof.rules()) == 21 + 3
    )  # one more rule per mode + one bucket row (m=256) per mode
    r23_rules = [r for r in doc["rules"] if r["params"].get("src") == "r23"]
    mode_of = {specs.sp_qkv_spec(m).signature(): m for m in specs.MODES}
    assert sorted(mode_of[r["sig"]] for r in r23_rules) == sorted(specs.MODES)
    assert all(
        (r["m_min"], r["m_max"], r["m_multiple"], r["name"])
        == (129, 255, 1, f"{mode_of[r['sig']]}_mp256_m129-255")
        for r in r23_rules
    )
    assert all(
        r["params"]
        == specs.small_r23_params(mode_of[r["sig"]], specs.SMALL_R23_PROFILE_NH)
        for r in r23_rules
    )
    assert (
        doc["native_m_ranges"] == []
    )  # with an MP128 source (r22) and r23 every local m 1..255 fuses
    wb = doc["window_budget"]
    assert wb["small_owners"] == 15 and wb["max_windows"] == 2 * 15 + 3 * 9 + 1
    assert Counter(wb["small_window_sizes"]) == {
        4 << 20: 27,
        8 << 20: 3,
    }  # r22 owners: 4 MiB + 4 MiB; r23 owners: an 8 MiB Rx view + a 4 MiB control window
    assert sum(wb["shared"].values()) + sum(wb["small_window_sizes"]) == 468 << 20
    for mode in specs.MODES:
        for m in (129, 200, 255):
            plan, n = _small_at(mode, m, prof)
            assert (
                plan is not None
                and n == 2
                and (plan.src, plan.mp, plan.rx_slots, plan.profile_source)
                == ("r23", 256, 3, "rule")
            ), (mode, m)
            assert plan.resources()[0].bytes == 8 << 20
        assert (
            _small_at(mode, 257, prof) == (None, 0)
            and _small_at(mode, 256, prof)[0].src == "r23"
            and _small_at(mode, 128, prof)[0].src == "r22"
        )  # the MP128 slot is untouched; m 256 is the r23 bucket row, m 257 is not covered
    units, skipped = _units(prof)
    smalls = [u for u in units if u.kind == "sp_qkv.small.v1"]
    assert (
        skipped == []
        and len(units) == 18
        and sorted(u.meta["src"] for u in smalls)
        == ["r19"] * 6 + ["r20"] * 3 + ["r22"] * 3 + ["r23"] * 3
    )
    for u in (u for u in smalls if u.meta["src"] == "r23"):
        assert {
            "-DSM_B_QUAD=1",
            "-DSM_GEMM_BN128=0",
            "-DSM_GEMM_PFK=4",
            "-DSM_P_NT=2",
            "-DSM_PUSH_PASSTRIP=1",
            "-DSM_PUSH_WS_LATE=0",
            "-DSM_RX3=1",
            "-DSM_T1_B_HEADS=1",
            "-DSM_RX_BYPASS=1",
        } <= set(u.meta["dev"])
        cmd = [
            a.replace("\\", "/")
            for a in jit.device_cmd_one("r23", u.meta["dev"], TOOL, "/out", "sm_ran")
        ]
        assert any(a.endswith("small/r23/device/ran_kernel.cu") for a in cmd)
    # the KEY gate: 286 token counts = 30 large (the 1024..8192 grid + the 768 bucket) + 128 (local m 1..128) + 128 (local m 129..256) x 65 sites, all distinct non-zero 31-bit ids
    rep = callid.check_profile_call_ids(prof, R23_KEY_SITES, world=4)
    assert (
        rep.unresolved == []
        and rep.keys == (30 + 128 + 128) * len(R23_KEY_SITES)
        and rep.ok
    )
    # nh=2 is another build; r23 without an MP128 source leaves m 65..128 native
    monkeypatch.setattr(specs, "SMALL_R23_PROFILE_NH", 2)
    u2, _ = _units(parse_profile(json.loads(json.dumps(gen_profile.build_profile()))))
    assert all(
        "-DSM_T1_B_HEADS=2" in u.meta["dev"]
        for u in u2
        if u.kind == "sp_qkv.small.v1" and u.meta["src"] == "r23"
    )
    monkeypatch.setattr(specs, "SMALL_R21_IN_PROFILE", False)
    monkeypatch.setattr(specs, "SMALL_R22_IN_PROFILE", False)
    doc3 = json.loads(json.dumps(gen_profile.build_profile()))
    assert [(r["m_min"], r["m_max"]) for r in doc3["native_m_ranges"]] == [(65, 128)]
    prof3 = parse_profile(doc3)
    for mode in specs.MODES:
        assert (
            _small_at(mode, 100, prof3) == (None, 0)
            and _small_at(mode, 129, prof3)[0].src == "r23"
        )


def test_a_row_can_select_r23_directly(monkeypatch, jit_root, sp_qkv_catalog):
    clear_profile_cache()
    prof = profmod.load_profile(PROFILE)
    shipped = json.loads(Path(profmod._resolve(PROFILE)).read_text(encoding="utf-8"))
    sig = specs.sp_qkv_spec("dense").signature()
    row128 = next(
        _row_doc(r)
        for r in prof.rows()
        if r.vid == "sp_qkv.small.v1" and r.m == 128 and r.sig == sig
    )
    base_units, _ = _units(prof)
    # the shipped rule already builds the profile arm: a row that selects the BN128 A/B is another build
    row = dict(
        row128,
        m=200,
        params={**specs.small_r23_params("dense"), "gemm_bn128": 1, "pfk": 3},
    )
    p2 = parse_profile(
        {
            "profile": "x",
            "rows": [_row_doc(r) for r in prof.rows()] + [row],
            "rules": shipped["rules"],
        }
    )
    units, skipped = _units(p2)
    assert skipped == [] and len(units) == len(base_units) + 1
    plan, _n = match_prefix(
        specs.sp_qkv_spec("dense"), MatchCtx("gfx950", 256, 4, 200, SMALL_CAPS), p2
    )
    assert (plan.src, plan.mp, plan.rx_slots, plan.profile_source) == (
        "r23",
        256,
        3,
        "row",
    )
    # r23 with a knob at 0 is refused by the plan constructor path (the generator / match never builds it)
    bad = dict(row128, m=201, params={**specs.small_r23_params("dense"), "rx3": 0})
    with pytest.raises(Exception, match="requires rx3=1"):
        _units(
            parse_profile(
                {
                    "profile": "x",
                    "rows": [_row_doc(r) for r in prof.rows()] + [bad],
                    "rules": shipped["rules"],
                }
            )
        )


# ======================================================================= R16 pad-M: bucket selection (match.bucket_for_m / match_padded)
SMALL_BUCKETS = (
    1,
    2,
    4,
    8,
    12,
    16,
    32,
    64,
    128,
    256,
)  # the exact small rows of the shipped profile (r19 / r20 / r22 / r23)


def _bucket(mode, m, prof=PROFILE, caps=CAPS, **kw):
    from aiter.gemm_fusion import bucket_for_m

    return bucket_for_m(
        specs.sp_qkv_spec(mode), MatchCtx("gfx950", 256, 4, m, caps, **kw), prof
    )


def test_bucket_for_m_small_pads_into_the_smallest_exact_row_within_its_band(
    sp_qkv_catalog,
):
    clear_profile_cache()
    prof = profmod.load_profile(PROFILE)
    for mode in specs.MODES:
        for m in range(1, 257):
            want = next(b for b in SMALL_BUCKETS if b >= m)
            assert _bucket(mode, m, prof) == want, (mode, m)
            if m in SMALL_BUCKETS:
                assert _bucket(mode, m, prof) == m  # exact-row behaviour is kept
        # every bucket's plan accepts the real m it was chosen for (the same pure predicate the selection used)
        from aiter.gemm_fusion import catalog

        sig = specs.sp_qkv_spec(mode).signature()
        for m in (3, 5, 9, 13, 20, 31, 33, 63, 65, 100, 129, 200, 255):
            row = prof.lookup(sig, "gfx950", 256, 4, _bucket(mode, m, prof))
            cls = catalog.get_variant(row.vid).factory
            assert cls.accepts_launch_m(dict(row.params), row.m, m), (mode, m)
        # the bands: r19 <= 32, r20 33..64, r22 65..128, r23 129..256
        for m, src in (
            (7, "r19"),
            (32, "r19"),
            (33, "r20"),
            (64, "r20"),
            (65, "r22"),
            (128, "r22"),
            (129, "r23"),
            (256, "r23"),
        ):
            row = prof.lookup(sig, "gfx950", 256, 4, _bucket(mode, m, prof))
            assert dict(row.params)["src"] == src, (mode, m)
        # above the small envelope: the shipped large rows are r18 (route B, pad-M): m 257..385 is native; the 512 bucket pads from 386 (all modes), the
        # 768 bucket from 513 (all modes); the 1024 bucket from 769, then ceil 256 up to 8192; 8193+ native
        for m in (257, 300, 385):
            assert _bucket(mode, m, prof) is None, (mode, m)
        for m in (386, 450, 511, 512):
            assert _bucket(mode, m, prof) == 512, (mode, m)
        for m in (513, 600, 700, 767, 768):
            assert _bucket(mode, m, prof) == 768, (mode, m)
        for m in (769, 896, 1000, 1023, 1024):
            assert _bucket(mode, m, prof) == 1024, (mode, m)
        for m in (1025, 1100, 1280):
            assert _bucket(mode, m, prof) == 1280, (mode, m)
        for m in (4096, 8191, 8192):
            assert _bucket(mode, m, prof) == (m if m != 8191 else 8192), (mode, m)
        for m in (8193, 20000):
            assert _bucket(mode, m, prof) is None, (mode, m)


def test_bucket_for_m_is_a_pure_function_of_the_real_m(sp_qkv_catalog):
    for mode in specs.MODES:
        for m in (5, 100, 200, 1100, 1280):
            ref = _bucket(mode, m)
            assert (
                _bucket(mode, m, caps=SMALL_CAPS, graph_mode="capture")
                == ref
                == _bucket(mode, m, caps=frozenset())
            )  # capabilities / graph mode never change the choice


def test_bucket_for_m_large_pads_to_ceil_256_with_r18_and_honours_the_floors(
    jit_root, sp_qkv_catalog
):
    doc = json.loads(json.dumps(gen_profile.build_profile()))
    assert doc["m_pad_min_large"] == specs.LARGE_R18_M_PAD_MIN_LARGE == 386
    assert specs.m_pad_min_large() == 386
    assert {
        r["params"]["src"]
        for r in doc["rows"] + doc["rules"]
        if r["vid"] == "sp_qkv.large.v1"
    } == {
        "r13",
        "r18",
    }  # the bucket rows are r18 (route B), the 1024.. grid rows stay r13
    # the r18 bucket rows carry the shipping floors explicitly (512: 386, 768: 513 in every mode; Evidence's route-B timing); the 1024 rows keep r13's 769
    floors = {
        (r["cell"].split("_")[0], r["m"]): r["pad_min"]
        for r in doc["rows"]
        if "pad_min" in r
    }
    assert floors == {
        ("dense", 512): 386,
        ("indexer", 512): 386,
        ("reuse", 512): 386,
        ("dense", 768): 513,
        ("indexer", 768): 513,
        ("reuse", 768): 513,
        ("dense", 1024): 769,
        ("indexer", 1024): 769,
        ("reuse", 1024): 769,
    }
    prof = parse_profile(doc)
    for mode in specs.MODES:
        for m, want in (
            (257, None),
            (385, None),
            (386, 512),
            (450, 512),
            (512, 512),
            (513, 768),
            (700, 768),
            (768, 768),
            (769, 1024),
            (896, 1024),
            (1000, 1024),
            (1023, 1024),
            (1024, 1024),
            (1025, 1280),
            (1100, 1280),
            (1279, 1280),
            (1280, 1280),
            (2049, 2304),
            (4097, 4352),
            (8191, 8192),
            (8192, 8192),
            (8193, None),
        ):
            assert _bucket(mode, m, prof) == want, (mode, m)
        assert (
            _bucket(mode, 129, prof) == 256 and _bucket(mode, 256, prof) == 256
        )  # the small band is unchanged
    # the 512 and the 768 bucket floors are the same in every mode (m 385 / 512 are the native stops below the rows)
    for m, want in ((385, None), (386, 512), (512, 512), (513, 768), (768, 768)):
        for mode in specs.MODES:
            assert _bucket(mode, m, prof) == want, (mode, m)

    # tightening ONE bucket's range by editing its row (the 896 study at the 1024 bucket): 769..895 becomes native for that mode, 768 is untouched
    d = json.loads(json.dumps(doc))
    for r in d["rows"]:
        if (
            r["vid"] == "sp_qkv.large.v1"
            and r["m"] == 1024
            and r["cell"].startswith("dense")
        ):
            r["pad_min"] = 896
    p = parse_profile(d)
    assert (
        _bucket("dense", 895, p) is None
        and _bucket("dense", 896, p) == 1024
        and _bucket("indexer", 895, p) == 1024
    )
    # "no larger bucket" rule: removing the 768 row's floor falls back to the profile field (386): the 512 row (386 floor) still serves 386..512 and the
    # 768 row pads from 386 on, so nothing below 386 falls through and 513..768 still lands on the 768 row
    d = json.loads(json.dumps(doc))
    for r in d["rows"]:
        if r["vid"] == "sp_qkv.large.v1" and r["m"] == 768:
            del r["pad_min"]
    p = parse_profile(d)
    assert (
        _bucket("reuse", 385, p) is None
        and _bucket("reuse", 386, p) == 512
        and _bucket("reuse", 700, p) == 768
        and _bucket("reuse", 768, p) == 768
        and _bucket("reuse", 769, p) == 1024
    )
    # a row without pad_min uses the profile field m_pad_min_large: the 1280 row (no row floor) with the field raised to 1400 pads only from 1400 on (1536 bucket)
    d = json.loads(json.dumps(doc))
    d["m_pad_min_large"] = 1400
    p = parse_profile(d)
    assert (
        _bucket("dense", 1300, p) is None
        and _bucket("dense", 1400, p) == 1536
        and _bucket("dense", 1536, p) == 1536
    )
    assert (
        _bucket("dense", 800, p) == 1024
    )  # rows WITH pad_min ignore the profile field
    # an exact row is never padded away: m == a row is that row whatever the floors
    d["m_pad_min_large"] = 4096
    assert _bucket("dense", 1280, parse_profile(d)) == 1280
    # without the field the class default (769) applies
    nofield = {k: v for k, v in doc.items() if k != "m_pad_min_large"}
    assert _bucket("dense", 1300, parse_profile(nofield)) == 1536
    for bad in (0, -1, "1024", True, 1.5):
        with pytest.raises(
            ProfileError, match="m_pad_min_large must be a positive int"
        ):
            parse_profile({**doc, "m_pad_min_large": bad})
    # row.pad_min is validated: an int in 1..m
    for bad in (0, -1, "300", True, 1.5, 769):
        d = json.loads(json.dumps(doc))
        next(r for r in d["rows"] if r["vid"] == "sp_qkv.large.v1" and r["m"] == 768)[
            "pad_min"
        ] = bad
        with pytest.raises(
            ProfileError, match=r"pad_min must be an int in 1\.\.m \(768\)"
        ):
            parse_profile(d)
    # the Row carries it (and _row_doc round-trips it)
    row = next(
        r
        for r in prof.rows()
        if r.m == 768 and r.vid == "sp_qkv.large.v1" and r.pad_min == 513
    )
    assert _row_doc(row)["pad_min"] == 513
    assert all(r.pad_min is None for r in prof.rows() if r.vid == "sp_qkv.small.v1")


def test_the_r12_layout_has_no_pad_m_for_the_large_band_only_exact_rows(
    monkeypatch, jit_root, sp_qkv_catalog
):
    monkeypatch.setattr(
        specs, "LARGE_R18_IN_PROFILE", False
    )  # the outer r18 buckets are off
    monkeypatch.setattr(
        specs, "LARGE_R13_IN_PROFILE", False
    )  # the rollback / A/B layout
    prof = parse_profile(json.loads(json.dumps(gen_profile.build_profile())))
    for mode in specs.MODES:
        for m in (257, 300, 513, 769, 896, 1000, 1023, 1025, 1100, 8191):
            assert _bucket(mode, m, prof) is None, (
                mode,
                m,
            )  # r12 launches its bound M only
        for m in (768, 1024, 1280, 8192):
            assert _bucket(mode, m, prof) == m, (mode, m)
        assert (
            _bucket(mode, 512, prof) is None
        )  # no 512 row in the r12 / pre-r18 layouts
        assert _bucket(mode, 129, prof) == 256  # the small band keeps padding


def test_match_padded_returns_the_bucket_plan_and_native_none(
    monkeypatch, jit_root, sp_qkv_catalog
):
    from aiter.gemm_fusion import match_padded

    caps = SMALL_CAPS
    plan, n, bucket = match_padded(
        specs.sp_qkv_spec("dense"), MatchCtx("gfx950", 256, 4, 100, caps), PROFILE
    )
    assert (n, bucket, plan.src, plan.bucket_m, plan.pad_m, plan.profile_source) == (
        2,
        128,
        "r22",
        128,
        True,
        "row",
    )
    plan, n, bucket = match_padded(
        specs.sp_qkv_spec("indexer"), MatchCtx("gfx950", 256, 4, 200, caps), PROFILE
    )
    assert (n, bucket, plan.src, plan.mp, plan.bucket_m) == (2, 256, "r23", 256, 256)
    plan, n, bucket = match_padded(
        specs.sp_qkv_spec("reuse"), MatchCtx("gfx950", 256, 4, 16, caps), PROFILE
    )
    assert (bucket, plan.bucket_m) == (16, 16)  # m == an exact row
    assert match_padded(
        specs.sp_qkv_spec("dense"), MatchCtx("gfx950", 256, 4, 8200, caps), PROFILE
    ) == (
        None,
        0,
        None,
    )  # m > 8192: native
    for (
        mode
    ) in (
        specs.MODES
    ):  # the shipped r18 512 bucket serves every real m from its floor 386 on
        p500, n500, b500 = match_padded(
            specs.sp_qkv_spec(mode), MatchCtx("gfx950", 256, 4, 500, CAPS), PROFILE
        )
        assert (n500, b500, p500.src, p500.M, p500.pad_m) == (
            1,
            512,
            "r18",
            512,
            True,
        )
    p600, _n, b600 = match_padded(
        specs.sp_qkv_spec("dense"), MatchCtx("gfx950", 256, 4, 600, CAPS), PROFILE
    )
    assert (b600, p600.M, p600.n, p600.half_first, p600.src) == (
        768,
        768,
        1,
        0,
        "r18",
    )  # the 768 r18 bucket pads from 513 in every mode (dense used to be native below 700)
    # the large plan (the shipped r13 layout): built for the BUCKET M, run with the real rows
    prof = parse_profile(json.loads(json.dumps(gen_profile.build_profile())))
    plan900, n900, bucket900 = match_padded(
        specs.sp_qkv_spec("dense"), MatchCtx("gfx950", 256, 4, 900, CAPS), prof
    )
    assert (n900, bucket900, plan900.src, plan900.M) == (
        1,
        1024,
        "r13",
        1024,
    )  # m_pad_min_large = 896
    plan, n, bucket = match_padded(
        specs.sp_qkv_spec("dense"), MatchCtx("gfx950", 256, 4, 1100, CAPS), prof
    )
    assert (n, bucket, plan.src, plan.M, plan.pad_m, plan.abi) == (
        1,
        1280,
        "r13",
        1280,
        True,
        4,
    )
    # a rank asking with another real m of the same bucket gets the same plan parameters (the choice depends on m alone)
    plan2, _n2, bucket2 = match_padded(
        specs.sp_qkv_spec("dense"), MatchCtx("gfx950", 256, 4, 1279, CAPS), prof
    )
    assert (
        bucket2 == 1280
        and dict(plan2.params) == dict(plan.params)
        and plan2.owner == plan.owner
    )


def test_the_r12_layout_keeps_the_budget_and_swaps_only_the_large_source(
    monkeypatch, jit_root, sp_qkv_catalog
):
    monkeypatch.setattr(
        specs, "LARGE_R18_IN_PROFILE", False
    )  # the outer r18 buckets are off
    r13 = json.loads(json.dumps(gen_profile.build_profile()))  # the r13 layout
    monkeypatch.setattr(specs, "LARGE_R13_IN_PROFILE", False)
    shipped = json.loads(json.dumps(gen_profile.build_profile()))  # the r12 layout
    assert (
        shipped["window_budget"] == r13["window_budget"]
        and len(shipped["rows"]) == len(r13["rows"])
        and len(shipped["rules"]) == len(r13["rules"])
    )
    for a, b in zip(shipped["rows"] + shipped["rules"], r13["rows"] + r13["rules"]):
        if a["vid"] == "sp_qkv.large.v1":
            assert a["params"]["src"] == "r12" and b["params"] == {
                **a["params"],
                "src": "r13",
            }
        else:
            assert a == b
    monkeypatch.setattr(specs, "LARGE_R13_IN_PROFILE", True)
    units, skipped = _units(parse_profile(r13))
    large = [u for u in units if u.kind == "sp_qkv.large.v1"]
    assert (
        skipped == []
        and len(units) == 18
        and {u.meta["src"] for u in large} == {"r13"}
        and len(large) == 2
    )


# ======================================================================= the 512 / 768 pad buckets: plan constraints and the r13 static asserts
def test_the_512_and_768_buckets_use_the_only_params_legal_for_2_and_3_chunks(
    monkeypatch, jit_root, sp_qkv_catalog
):
    from aiter.gemm_fusion import ContractError
    from aiter.ops.gemm_fusion.sp_qkv import variants_large

    # the capability (both buckets, n=1 / half_first 0) and what the profile ships: 768 only (Evidence: 512 never pays for dense / indexer, ~1.005 for reuse)
    assert specs.LARGE_PAD_BUCKETS == {512: (1, 0), 768: (1, 0)}
    assert specs.LARGE_PAD_BUCKETS_IN_PROFILE == (768,)
    assert specs.LARGE_BUCKET_PAD_MIN == {
        "dense": {512: 512, 768: 700, 1024: 769},
        "indexer": {512: 512, 768: 600, 1024: 769},
        "reuse": {512: 450, 768: 513, 1024: 769},
    }

    def plan(mode, m, **params):
        base = {"n": 1, "half_first": 0, "no_gridread": 1, "src": "r13"}
        return variants_large.SpQkvLarge(
            specs.sp_qkv_spec(mode).prefix(1),
            {**base, **params},
            MatchCtx("gfx950", 256, 4, m, CAPS),
        )

    for mode in specs.MODES:
        assert (
            plan(mode, 512).n == 1 and plan(mode, 768).n == 1
        )  # the plans exist for both
        # 2 chunks: puts per peer n in {1, 2} (n must divide the chunk count; n=2 may be half-first); 4 does not divide 2
        assert (
            plan(mode, 512, n=2).n == 2
            and plan(mode, 512, n=2, half_first=1).half_first == 1
        )
        with pytest.raises(ContractError, match=r"n=4 invalid for 2 chunks"):
            plan(mode, 512, n=4)
        # 3 chunks: only n=1 divides 3; quarter-first (n=3) exists only at 8 chunks (dense build), half-first needs n >= 2
        for bad, msg in (
            ({"n": 2}, r"n=2 invalid for 3 chunks"),
            ({"n": 4}, r"n=4 invalid for 3 chunks"),
            ({"n": 3, "half_first": 2}, "quarter-first"),
            ({"n": 1, "half_first": 1}, "half-first needs n >= 2"),
        ):
            with pytest.raises(ContractError, match=msg):
                plan(mode, 768, **bad)
    # the shipped profile: the r18 route-B bucket rows at 512 and 768 (n=1, floors 386 / 513 all modes, "not swept" note)
    clear_profile_cache()
    prof = profmod.load_profile(PROFILE)
    doc = json.loads(Path(profmod._resolve(PROFILE)).read_text(encoding="utf-8"))
    notes = {(r["sig"], r["m"]): r.get("note", "") for r in doc["rows"]}
    for mode in specs.MODES:
        sig = specs.sp_qkv_spec(mode).prefix(1).signature()
        for m, floor in ((512, 386), (768, 513)):
            row = prof.lookup(sig, "gfx950", 256, 4, m)
            assert (
                dict(row.params)
                == {
                    "n": 1,
                    "half_first": 0,
                    "no_gridread": 1,
                    "src": "r18",
                }
                and row.pad_min == floor
                and row.vid == "sp_qkv.large.v1"
            )
            assert "not swept" in notes[(sig, m)] and "r18 route B" in notes[(sig, m)]
    # the r13 CAPABILITY: with the r18 layout off, switching the 512 bucket on adds its rows with the per-mode floors (dense / indexer: exact m only, reuse from 450)
    monkeypatch.setattr(specs, "LARGE_R18_IN_PROFILE", False)
    monkeypatch.setattr(specs, "LARGE_PAD_BUCKETS_IN_PROFILE", (512, 768))
    p2 = parse_profile(json.loads(json.dumps(gen_profile.build_profile())))
    assert len(p2) == 120 + 3 == 123
    assert _bucket("dense", 500, p2) is None and _bucket("dense", 512, p2) == 512
    assert _bucket("indexer", 511, p2) is None and _bucket("indexer", 512, p2) == 512
    assert (
        _bucket("reuse", 449, p2) is None
        and _bucket("reuse", 450, p2) == 512
        and _bucket("reuse", 512, p2) == 512
    )
    assert (
        _bucket("reuse", 513, p2) == 768 and _bucket("dense", 513, p2) is None
    )  # 768 keeps its own floors
    # r13's static proof (lg_r6x.hpp r13_live_ok) lists the chunk counts it instantiates: n1 {1,2,4,8,16,32}, n2 {2..32}: both buckets are inside it or n=1 (a single group copying every live row: contiguous by construction)
    import re

    hpp = Path(
        os.path.join(jit.CSRC, "large", "r13", "device", "lg_r6x.hpp")
    ).read_text(encoding="utf-8")
    n1 = {int(x) for x in re.findall(r"r13_live_ok<1, 0>\((\d+)\)", hpp)}
    n2 = {int(x) for x in re.findall(r"r13_live_ok<2, 0>\((\d+)\)", hpp)}
    assert {1, 2, 4, 8, 16, 32} == n1 and 2 in n2
    assert (
        512 // 256 in n1 and 512 // 256 in n2
    )  # 2 chunks: n=1 and n=2 are both asserted
    assert (
        768 // 256 not in n1
    )  # 3 chunks: n=1 is the single-group plan (and the odd-chunk n=1 rows 1280 / 1792 / ... already run: Evidence G-M1)


def _r18_doc(monkeypatch, buckets=(512, 768)):
    monkeypatch.setattr(specs, "LARGE_R18_IN_PROFILE", True)
    monkeypatch.setattr(specs, "LARGE_R18_BUCKETS", buckets)
    return json.loads(json.dumps(gen_profile.build_profile()))


def test_the_r18_flag_is_on_and_the_shipped_profile_ships_both_buckets():
    assert (
        specs.LARGE_R18_IN_PROFILE is True
        and specs.LARGE_R18_BUCKETS == (512, 768)
        and specs.LARGE_R18_BUCKET_PAD_MIN == {512: 386, 768: 513}
    )
    doc = json.loads(Path(profmod._resolve(PROFILE)).read_text(encoding="utf-8"))
    assert {r["params"].get("src") for r in doc["rows"]} >= {"r13", "r18"}
    assert specs.large_pad_buckets_in_profile() == (512, 768)
    assert doc["m_pad_min_large"] == specs.m_pad_min_large() == 386


def test_the_r18_profile_ships_a_512_and_a_768_row_per_mode_with_the_measured_floors(
    monkeypatch, jit_root, sp_qkv_catalog
):
    from aiter.gemm_fusion import callid, match_padded

    shipped = json.loads(json.dumps(gen_profile.build_profile()))
    doc = _r18_doc(monkeypatch)
    big = {
        (r["cell"].split("_")[0], r["m"]): r
        for r in doc["rows"]
        if r["vid"] == "sp_qkv.large.v1" and r["m"] < 1024
    }
    assert (
        sorted(big) == [(mode, m) for mode in specs.MODES for m in (512, 768)]
        and len(doc["rows"]) == 123
    )
    for (mode, m), r in big.items():
        assert (
            r["params"] == {"n": 1, "half_first": 0, "no_gridread": 1, "src": "r18"}
            and "tile" not in r["params"]
        )  # the wave rule picks the tile per launch M
        assert (
            r["pad_min"] == {512: 386, 768: 513}[m]
            and "r18 route B" in r["note"]
            and "PROVISIONAL" not in r["note"]
        )
    assert {r["m_min"] for r in doc["rules"] if r["vid"] == "sp_qkv.large.v1"} == {512}
    assert (
        doc["window_budget"] == shipped["window_budget"]
    )  # the windows are r13's (S / R / C), no arena change
    assert (
        doc["m_pad_min_large"],
        shipped["m_pad_min_large"],
    ) == (
        386,
        386,
    )  # shipped == the r18 layout
    prof = parse_profile(doc)
    for mode in specs.MODES:
        for m, want in (
            (256, 256),
            (257, None),
            (300, None),
            (385, None),
            (386, 512),
            (450, 512),
            (512, 512),
            (513, 768),
            (600, 768),
            (700, 768),
            (768, 768),
            (769, 1024),
        ):
            assert _bucket(mode, m, prof) == want, (mode, m)
        for m, M in ((386, 512), (450, 512), (700, 768)):
            plan, n, bucket = match_padded(
                specs.sp_qkv_spec(mode), MatchCtx("gfx950", 256, 4, m, SMALL_CAPS), prof
            )
            assert (
                n,
                bucket,
                plan.vid,
                plan.src,
                plan.r18,
                plan.M,
                plan.pad_m,
                plan.tile,
            ) == (1, M, "sp_qkv.large.v1", "r18", True, M, True, None)
    units, skipped = _units(prof)
    assert (
        skipped == [] and len(units) == 18 + 2
    )  # two NEW large builds (src r18: dense-build and indexer-build), the r13 ones stay
    sites = [f"model.layers.{i}.self_attn" for i in range(62)] + [
        f"mtp.layers.{i}.self_attn" for i in range(3)
    ]
    keys, unresolved = callid.profile_call_id_keys(prof, sites, world=4, pad_m=True)
    assert unresolved == [] and {k[1] for k in keys} >= {
        4 * m for m in (386, 450, 512, 600, 700)
    }
    ids = callid.tag_unique_ids(keys)
    assert callid.check_call_ids(
        keys,
        lambda *k: ids[k],
        tag_bits=19,
        tag_vids=("sp_qkv.large.v1",),
        per_tokens=True,
    ).ok


def test_the_r18_single_bucket_profiles_and_the_flag_combinations(
    monkeypatch, jit_root, sp_qkv_catalog
):
    d512 = _r18_doc(monkeypatch, (512,))
    assert (
        sorted(
            {
                r["m"]
                for r in d512["rows"]
                if r["vid"] == "sp_qkv.large.v1" and r["m"] < 1024
            }
        )
        == [512]
        and len(d512["rows"]) == 120
    )
    d768 = _r18_doc(monkeypatch, (768,))
    assert sorted(
        {
            (r["m"], r["pad_min"])
            for r in d768["rows"]
            if r["vid"] == "sp_qkv.large.v1" and r["m"] < 1024
        }
    ) == [(768, 513)]


# ======================================================================= ROUTE A alternate profile (opt-in path)
def test_r19_alternate_profile_generates_the_measured_layout_and_the_default_stays_unchanged(
    tmp_path, monkeypatch, sp_qkv_catalog
):
    import hashlib

    shipped = (
        Path(__file__).resolve().parents[1]
        / "aiter"
        / "configs"
        / "gemm_fusion"
        / "minimax_m3_sp4_gfx950_v1.json"
    )
    alt = (
        Path(__file__).resolve().parents[1]
        / "aiter"
        / "configs"
        / "gemm_fusion"
        / "minimax_m3_sp4_gfx950_v1_r19.json"
    )
    assert (
        hashlib.sha256(shipped.read_bytes()).hexdigest()
        == "0f90d1cd84fb252a3392454439e03c49907114f7e288e25b53d288a3725514a9"
    )
    assert (
        hashlib.sha256(alt.read_bytes()).hexdigest()
        == "80dc9887e0ce07c6d63ac39cface07fbd29648875f901a1101a514bc3378cbfd"
    )
    assert specs.LARGE_R19_IN_PROFILE is False
    default_out = tmp_path / "default.json"
    gen_profile.main([str(default_out)])
    assert (
        hashlib.sha256(default_out.read_bytes()).hexdigest()
        == "0f90d1cd84fb252a3392454439e03c49907114f7e288e25b53d288a3725514a9"
    )
    alt_out = tmp_path / "alt.json"
    gen_profile.main([str(alt_out), "--large-r19"])
    assert (
        hashlib.sha256(alt_out.read_bytes()).hexdigest()
        == "80dc9887e0ce07c6d63ac39cface07fbd29648875f901a1101a514bc3378cbfd"
    )
    # the temporary enable is restored: the default flag and the default generator are unchanged afterwards
    assert specs.LARGE_R19_IN_PROFILE is False
    assert len(json.loads(alt_out.read_text(encoding="utf-8"))["rows"]) == 147
    assert json.loads(alt_out.read_text(encoding="utf-8"))["extended_large"][
        "measured"
    ].startswith("Gate-3")
