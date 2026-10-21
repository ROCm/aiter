# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tests for sp_qkv small-M integration (S2-2b-ii): profile, flags, JIT command plan, FusionResources."""

import contextlib
import json
import logging
import os
import sys
import types
from itertools import pairwise
from typing import ClassVar

import pytest

try:
    import aiter.gemm_fusion  # noqa: F401
except Exception:  # noqa: BLE001  (no torch / GPU: load the pure-Python subpackage)
    for _k in [k for k in sys.modules if k == "aiter" or k.startswith("aiter.")]:
        del sys.modules[_k]
    _pkg = types.ModuleType("aiter")
    _pkg.__path__ = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "aiter"))
    ]
    sys.modules["aiter"] = _pkg

from aiter.gemm_fusion import ContractError, MatchCtx, match_prefix
from aiter.gemm_fusion import profile as profmod
from aiter.gemm_fusion.plan import WindowReq
from aiter.gemm_fusion.resources import ResourceError, round_up
from aiter.ops.gemm_fusion.sp_qkv import gen_profile, jit, specs
from aiter.ops.gemm_fusion.sp_qkv.test_utils import (
    ArenaBackend,
    FakeBackend,
    FakeDist,
    FirstFitBackend,
    RealisticFirstFit,
    clear_profile_cache,
    make_res,
)
from aiter.ops.gemm_fusion.sp_qkv.test_utils import (
    log_capture as _log_capture,
)

PROFILE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "aiter",
    "configs",
    "gemm_fusion",
    "minimax_m3_sp4_gfx950_v1.json",
)


def _slurp(path, mode="r", **kw):
    with open(path, mode, **kw) as f:
        return f.read()


def _spit(path, data, mode="w", **kw):
    with open(path, mode, **kw) as f:
        f.write(data)


def test_profile_file_matches_generator():
    with open(PROFILE, encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk == json.loads(
        json.dumps(gen_profile.build_profile())
    ), "re-run op_tests/gen_gemm_fusion_profile.py"


def test_profile_rows_exact_and_distinct():
    clear_profile_cache()
    prof = profmod.load_profile(PROFILE)
    small = [r for r in prof.rows() if r.vid == "sp_qkv.small.v1"]
    assert (
        len(small) == 24 == len(specs.small_exact_cells())
    )  # 6 tuned + 15 P2 copies (m 1, 2, 4, 8, 12) + 3 r20 rows (m 64)
    for (mode, T), (vid, params) in specs.SMALL_CELLS.items():
        r = prof.lookup(specs.sp_qkv_spec(mode).signature(), "gfx950", 256, 4, T // 4)
        # the profile runs every small cell on r19 (R12); SMALL_CELLS keeps the per-cell r17/r18 sources of the pre-R12 build for the A/B
        assert (
            r is not None
            and dict(r.params) == specs.small_cell_params(mode, T)
            and dict(r.params) == {**params, "src": "r19"}
            and r.vid == vid
        )
    # three modes -> three different signatures
    assert len({specs.sp_qkv_spec(m).signature() for m in specs.MODES}) == 3
    assert (
        specs.sp_qkv_spec("dense").gemm.n == 9216
        and specs.sp_qkv_spec("reuse").gemm.n == 9856
    )
    assert specs.mode_of(specs.sp_qkv_spec("reuse")) == "reuse"


def test_field_map_matches_golden():
    f = {
        x.name: (x.offset, x.width, x.shards)
        for x in specs.sp_qkv_spec("indexer").stages[0].fields
    }
    assert f == {
        "q": (0, 8192, 4),
        "k": (8192, 512, 4),
        "v": (8704, 512, 4),
        "iq": (9216, 512, 4),
        "ik": (9728, 128, 1),
    }
    assert sum(w // s for _, w, s in f.values()) == 2560  # F indexer
    d = specs.sp_qkv_spec("dense").stages[0]
    assert sum(x.width // x.shards for x in d.fields) == 2304


def test_small_defs_per_cell():
    d, h = jit.small_defs(*[specs.SMALL_CELLS[("dense", 128)][1], "dense"])
    assert (
        "-DSM_B_ROWSPLIT=1" in h
        and "-DSM_RX_BYPASS=1" in h
        and "-DSM_DENSE=1" in h
        and "-DSM_NT_STORES=0" in d
    )
    d, h = jit.small_defs(specs.SMALL_CELLS[("indexer", 64)][1], "indexer")
    assert "-DSM_PUSH8=1" in h and "-DSM_R5X_INDEXER=1" in h and "-DSM_NO_CREDIT=1" in d
    d, h = jit.small_defs(specs.SMALL_CELLS[("reuse", 128)][1], "reuse")
    assert "-DSM_PUSH_HALVES=1" in h and not any(
        "ROWSPLIT" in x for x in h
    )  # r17 has no r18 macros
    d, h = jit.small_defs(specs.SMALL_CELLS[("reuse", 64)][1], "reuse")
    assert "-DSM_REUSE_NO_IDX=1" in h and "-DSM_RX_BYPASS=0" in h
    with pytest.raises(ValueError):
        jit.small_defs({"src": "r17", "rx_bypass": 1}, "dense")
    with pytest.raises(ValueError):
        jit.small_defs({"nt": 1}, "dense")
    with pytest.raises(ValueError):
        jit.small_defs({"bogus": 1}, "dense")


def test_jit_command_plan_and_cache_key(tmp_path_factory=None):
    tool = {
        "hipcc": "/rocm/bin/hipcc",
        "clangxx": "/rocm/llvm/bin/clang++",
        "bundler": "b",
        "rocm": "/rocm",
        "mori_root": "/m",
        "mori_so": "/m/libmori_cco.so",
    }
    dev, host = jit.small_defs(specs.SMALL_CELLS[("dense", 64)][1], "dense")
    cmds = jit.device_cmds("r17", dev, tool, "/out")
    assert len(cmds) == 2 and all(
        "--offload-arch=gfx950" in c
        and "-fno-fast-math" in c
        and "-D__HIPCC_RTC__" in c
        for c in cmds
    )
    assert any("third_party" in a and "opus_phase1" in a for a in cmds[0])
    assert cmds[0][-1].endswith("sm_gemm.so") and cmds[1][-3].endswith("ran_kernel.cu")
    hc = jit.host_cmds("r17", host, tool, "/out")[0]
    assert "-x" in hc and "/m/libmori_cco.so" in hc and "-D__HIPCC__" not in hc
    k1 = jit.cache_key("r17", dev, host, tool)
    assert k1 == jit.cache_key("r17", dev, host, tool)
    d2, h2 = jit.small_defs(specs.SMALL_CELLS[("indexer", 128)][1], "indexer")
    assert k1 != jit.cache_key("r17", d2, h2, tool)
    assert k1 != jit.cache_key("r17", dev, host, dict(tool, mori_root="/other"))


def test_vendored_sources_match_manifest():
    import hashlib

    for src in jit.SRCS:
        base = os.path.join(jit.CSRC, "small", src)
        if not os.path.isdir(
            base
        ):  # r19 (R12) is delivered by the implementor after G0-eq; its manifest is checked once it exists
            assert src == "r19", src
            continue
        n = 0
        for line in _slurp(os.path.join(base, "SOURCE_SHA256.txt")).splitlines(
            keepends=True
        ):
            h, f = line.split(None, 1)
            f = f.strip()
            if f.startswith("tests/") or f == "host/bridge_sm.py":
                continue
            assert (
                hashlib.sha256(_slurp(os.path.join(base, f), "rb")).hexdigest() == h
            ), f
            n += 1
        assert n >= 11


def test_resources_lifecycle_and_sizes():
    be, d = FakeBackend(), FakeDist()
    r = make_res(dist=d, backend=be)
    reqs = [WindowReq("rx", 1_300_000, "p2p_window"), WindowReq("ctl", 5000, "control")]
    w = r.ensure("plan/a", reqs)
    assert (
        w["rx"].bytes == w["ctl"].bytes == round_up(1_300_000) == 4 << 20
    )  # equal, 4 MiB multiple
    assert r.ensure("plan/a", reqs) is w  # idempotent per owner
    assert sum(1 for x in be.log if x[0] == "win") == 2
    r.finalize()
    assert ("dev", 1) in be.log and r.finalized
    with pytest.raises(ResourceError):
        r.ensure("plan/b", reqs)  # ensure after finalize
    be.log.clear()
    d.calls.clear()
    r.quiesce_and_reset()
    kinds = [x[0] for x in be.log]
    assert kinds == [
        "sync",
        "zero",
        "zero",
        "sync",
    ]  # device sync -> barrier -> reset -> sync -> barrier
    assert d.calls == [("barrier",), ("barrier",)]
    r.release()
    assert be.log[-1] == ("commdestroy",)
    r.release()  # idempotent


def ok_ctx_for(m):
    return MatchCtx("gfx950", 256, 4, m, frozenset({"mori_cco", "gfx950_sc_bypass"}))


def test_small_variant_registers_and_contract(monkeypatch):
    from aiter.gemm_fusion import catalog

    monkeypatch.setattr(catalog, "_CATALOG", {})
    sys.modules.pop("aiter.ops.gemm_fusion.sp_qkv.variants", None)
    import aiter.ops.gemm_fusion.sp_qkv.variants  # noqa: F401  (fresh import registers into the patched catalog)

    clear_profile_cache()
    spec = specs.sp_qkv_spec("dense")
    ok = MatchCtx("gfx950", 256, 4, 32, frozenset({"mori_cco", "gfx950_sc_bypass"}))
    plan, n = match_prefix(spec, ok, PROFILE)
    assert (
        n == 2
        and plan.vid == "sp_qkv.small.v1"
        and plan.mode == "dense"
        and plan.params["rx_bypass"] == 1
    )
    reqs = plan.resources()
    assert [r.kind for r in reqs] == ["p2p_window", "control", "control"] and all(
        r.bytes % (4 << 20) == 0 for r in reqs
    )  # rx, ctl and the R12 per-comm ERR/STEP page request (r19 = abi 3)
    assert plan.abi == 3 and plan.src == "r19" and reqs[2].share == "gf.global"
    with pytest.raises(ContractError, match="missing"):
        match_prefix(spec, MatchCtx("gfx950", 256, 4, 32, frozenset()), PROFILE)
    with pytest.raises(ContractError, match="gfx950_sc_bypass"):
        match_prefix(
            spec, MatchCtx("gfx950", 256, 4, 32, frozenset({"mori_cco"})), PROFILE
        )
    # reuse T64 (BYPASS 0) must not need the bypass capability
    plan, n = match_prefix(
        specs.sp_qkv_spec("reuse"),
        MatchCtx("gfx950", 256, 4, 16, frozenset({"mori_cco"})),
        PROFILE,
    )
    assert n == 2 and plan.params["reuse_no_idx"] == 1
    # not a validated cell -> native (m 1..64 all fuse: rows + the r19 / r20 rules; m 65..255 is native by design, incl. the default bucket m=128)
    for m_native in (65, 100, 128, 255):
        assert match_prefix(
            spec, MatchCtx("gfx950", 256, 4, m_native, frozenset({"mori_cco"})), PROFILE
        ) == (None, 0), m_native
    # the MP64 rule: m=48 resolves to the r20 plan, which needs the bypass capability (R2 is gfx950-only)
    with pytest.raises(ContractError, match="gfx950_sc_bypass"):
        match_prefix(
            spec, MatchCtx("gfx950", 256, 4, 48, frozenset({"mori_cco"})), PROFILE
        )
    p48, n48 = match_prefix(spec, ok_ctx_for(48), PROFILE)
    assert (
        n48 == 2
        and p48.src == "r20"
        and p48.mp == 64
        and p48.profile_source == "rule"
        and p48.abi == 3
    )
    # a spec with a different eps has a different signature -> native
    assert match_prefix(specs.sp_qkv_spec("dense", eps=1e-5), ok, PROFILE) == (None, 0)


# ======================================================================= large-M (S2-2b-iii)
def test_profile_has_111_rows_and_large_prefix_signatures():
    """24 small rows (see test_small_profile_rows...) + 87 EXACT large rows (one per mode x local m = 1024..8192 step 256, each cell's best (n, half_first) from the P1 sweep, src r12)."""
    from aiter.ops.gemm_fusion.sp_qkv import p1_sweep

    clear_profile_cache()
    prof = profmod.load_profile(PROFILE)
    assert len(prof) == 111  # 24 small + 87 large
    assert (
        len(p1_sweep.P1_GRID) == 29
        and sum(len(v) for v in p1_sweep.P1_BEST.values()) == 87
    )
    for mode in specs.MODES:
        sig = specs.sp_qkv_spec(mode).prefix(1).signature()
        for m in p1_sweep.P1_GRID:
            n, hf = p1_sweep.P1_BEST[mode][m]
            r = prof.lookup(sig, "gfx950", 256, 4, m)
            # the large rows select the R12 kernels explicitly (src r12); r11 stays selectable per row for the A/B; F0 only (no_gridread)
            assert (
                r is not None
                and r.vid == "sp_qkv.large.v1"
                and dict(r.params)
                == {"n": n, "half_first": hf, "no_gridread": 1, "src": "r12"}
            ), (mode, m)
    # the r11-era tuned rows (specs.LARGE_CELLS): kept where the sweep agrees, replaced (and noted in the file) where it differs
    with open(profmod._resolve(PROFILE), encoding="utf-8") as f:
        doc = json.load(f)
    notes = {
        (r["cell"].split("_T")[0], r["m"]): r.get("note", "")
        for r in doc["rows"]
        if r["vid"] == "sp_qkv.large.v1"
    }
    differ = set()
    for (mode, T), (_vid, old) in specs.LARGE_CELLS.items():
        new = p1_sweep.P1_BEST[mode][T // 4]
        if (old["n"], old["half_first"]) == new:
            assert notes[(mode, T // 4)] == "", (mode, T)
        else:
            differ.add((mode, T // 4))
            assert (
                f"replaces the r11-era tuned (n={old['n']}, half_first={old['half_first']})"
                in notes[(mode, T // 4)]
            ), (mode, T)
    assert differ == {
        ("dense", 4096),
        ("indexer", 4096),
        ("indexer", 8192),
        ("reuse", 4096),
    }
    assert sum(1 for v in notes.values() if v) == 4
    # reuse and dense large rows have different signatures (reuse drops iq/ik in the exchange)
    assert (
        specs.sp_qkv_spec("dense").prefix(1).signature()
        != specs.sp_qkv_spec("reuse").prefix(1).signature()
    )
    # 2-stage and 1-stage signatures differ
    assert (
        specs.sp_qkv_spec("dense").signature()
        != specs.sp_qkv_spec("dense").prefix(1).signature()
    )


def test_large_defs_and_commands():
    d = jit.large_defs({"n": 2, "half_first": 1, "no_gridread": 1}, "indexer")
    assert (
        "-DLG_X_INDEXER=1" in d
        and "-DLG_NO_GRIDREAD=1" in d
        and "-DBUILD_CCO_SDMA=1" in d
        and "-DLG_R8_FOLD=0" in d
    )
    assert "-DLG_X_INDEXER=0" in jit.large_defs(
        {}, "reuse"
    ) and "-DLG_X_INDEXER=0" in jit.large_defs({}, "dense")
    assert (
        "-DLG_NEGCTRL_NOWAIT=0" in d
        and len([x for x in d if x.startswith("-DLG_NEGCTRL_NOWAIT")]) == 1
    )
    with pytest.raises(ValueError):
        jit.large_defs({"bogus": 1}, "dense")
    with pytest.raises(ValueError):
        jit.large_defs({"no_gridread": 3}, "dense")
    tool = {
        "hipcc": "/rocm/bin/hipcc",
        "clangxx": "/rocm/llvm/bin/clang++",
        "bundler": "b",
        "rocm": "/rocm",
        "mori_root": "/m",
        "mori_so": "/m/libmori_cco.so",
    }
    dc = jit.large_device_cmd(d, tool, "/out")
    assert (
        "--offload-arch=gfx950" in dc
        and "-D__HIPCC_RTC__" in dc
        and dc[-3].endswith("gemm_quad_fp8_comm.cu")
    )
    hc = jit.large_host_cmd(d, tool, "/out")
    assert (
        "-DLG_X_INDEXER=1" in hc
        and "-DLG_NO_GRIDREAD=1" not in hc
        and "/m/libmori_cco.so" in hc
    )
    assert jit.large_cache_key(d, tool) != jit.large_cache_key(
        jit.large_defs({}, "dense"), tool
    )


def test_large_vendored_manifest():
    import hashlib

    base = os.path.join(jit.CSRC, "large", "r11")
    n = 0
    for line in _slurp(os.path.join(base, "SOURCE_SHA256.txt")).splitlines(
        keepends=True
    ):
        h, f = line.split(None, 1)
        f = f.strip()
        if f.startswith("tests/") or f == "host/bridge_comm.py":
            continue
        assert hashlib.sha256(_slurp(os.path.join(base, f), "rb")).hexdigest() == h, f
        n += 1
    assert n >= 10


def test_large_variant_contract_and_sizes(monkeypatch):
    from aiter.gemm_fusion import catalog

    monkeypatch.setattr(catalog, "_CATALOG", {})
    sys.modules.pop("aiter.ops.gemm_fusion.sp_qkv.variants", None)
    sys.modules.pop("aiter.ops.gemm_fusion.sp_qkv.variants_large", None)
    import aiter.ops.gemm_fusion.sp_qkv.variants_large as vl

    clear_profile_cache()
    caps = frozenset({"mori_cco", "sdma"})
    spec = specs.sp_qkv_spec("dense")
    plan, n = match_prefix(spec, MatchCtx("gfx950", 256, 4, 2048, caps), PROFILE)
    assert (
        n == 1
        and plan.vid == "sp_qkv.large.v1"
        and plan.half_first == 2
        and plan.n == 3
    )
    reqs = plan.resources()
    assert [r.kind for r in reqs] == [
        "p2p_window",
        "p2p_window",
        "control",
        "sdma_queue",
        "control",  # R12 (src r12): the per-comm ERR/STEP/DIAG page request (shared tag gf.global)
    ]
    assert reqs[4].share == "gf.global" and reqs[4].bytes == 4 << 20
    assert (
        all(r.bytes % (4 << 20) == 0 for r in reqs[:3])
        and reqs[0].bytes == reqs[1].bytes == reqs[2].bytes
    )
    # M=2048 dense: P = 8*2048*2304 = 37748736 -> + pad -> 40 MiB multiple of 4 MiB
    assert vl.large_window_bytes("dense", 2048) == 40 << 20
    assert vl.large_window_bytes("indexer", 8192) == round_up(8 * 8192 * 2560 + 4096)
    with pytest.raises(ContractError, match="sdma"):
        match_prefix(
            spec, MatchCtx("gfx950", 256, 4, 2048, frozenset({"mori_cco"})), PROFILE
        )
    # 2-stage prefix has no row at large M -> falls to the 1-stage row (n == 1) ; at M=16 the small variant would apply.
    # m=1024 is an exact P1 row (n=1: the sweep's best there); outside 1024..8192 or off the 256 grid it is still native
    p1024, n1024 = match_prefix(spec, MatchCtx("gfx950", 256, 4, 1024, caps), PROFILE)
    assert (
        n1024 == 1
        and p1024.profile_source == "row"
        and p1024.M == 1024
        and (p1024.n, p1024.half_first) == (1, 0)
    )
    for m_native in (512, 768, 1100, 8448):
        assert match_prefix(
            spec, MatchCtx("gfx950", 256, 4, m_native, caps), PROFILE
        ) == (
            None,
            0,
        )
    # invalid plan params are contract errors, not silent
    sp = specs.sp_qkv_spec("indexer").prefix(1)
    with pytest.raises(ContractError):
        vl.SpQkvLarge(
            sp, {"n": 3, "half_first": 2}, MatchCtx("gfx950", 256, 4, 2048, caps)
        )
    with pytest.raises(ContractError):
        vl.SpQkvLarge(
            specs.sp_qkv_spec("dense").prefix(1),
            {"n": 1, "half_first": 1},
            MatchCtx("gfx950", 256, 4, 2048, caps),
        )


def test_reset_runs_init_hooks():
    be, d = FakeBackend(), FakeDist()
    r = make_res(dist=d, backend=be)
    r.ensure("p", [WindowReq("a", 100, "p2p_window")])
    calls = []
    r.set_init("p", lambda: calls.append(1))
    assert calls == [1]
    r.finalize()
    # default zero mode h2d_deferred: windows are zeroed in finalize() (after ALL registrations + the devcomm) and the control image is rewritten
    # right after, so the registration-time init (calls[0]) is not lost; then the reset runs it once more
    assert calls == [1, 1]
    r.quiesce_and_reset()
    assert calls == [1, 1, 1]


def test_reset_order_with_drain_hook():
    """A2-H for large-M: device sync -> drain (all plans) + sync -> barrier -> zero+init -> sync -> barrier."""
    be, d = FakeBackend(), FakeDist()
    ev = []
    be_orig_sync, be_orig_zero = be.device_sync, be.zero
    be.device_sync = lambda dev: (ev.append("sync"), be_orig_sync(dev))[1]
    be.zero = lambda dev, loc, n: (ev.append("zero"), be_orig_zero(dev, loc, n))[1]
    d.barrier = lambda group=None: ev.append("barrier")
    r = make_res(dist=d, backend=be)
    r.ensure("big", [WindowReq("a", 100, "p2p_window")])
    r.ensure("small", [WindowReq("b", 100, "p2p_window")])
    r.set_drain("big", lambda: ev.append("drain"))
    r.set_init("big", lambda: ev.append("init"))
    r.finalize()
    ev.clear()
    r.quiesce_and_reset()
    # every registered window is zeroed once, THEN the init hooks run (so an init image is never wiped by a later zero), then sync + barrier
    assert ev == [
        "sync",
        "drain",
        "sync",
        "barrier",
        "zero",
        "zero",
        "init",
        "sync",
        "barrier",
    ]
    assert ev.index("init") > max(i for i, x in enumerate(ev) if x == "zero")
    # no drain hooks -> no extra sync (small-M only)
    r2 = make_res(dist=d, backend=FakeBackend())
    r2.ensure("s", [WindowReq("b", 100, "p2p_window")])
    ev2 = []
    r2.backend.device_sync = lambda dev: ev2.append("sync")
    r2.backend.zero = lambda dev, loc, n: ev2.append("zero")
    d.barrier = lambda group=None: ev2.append("barrier")
    r2.quiesce_and_reset()
    assert ev2 == ["sync", "barrier", "zero", "sync", "barrier"]


def test_release_runs_close_hooks_before_comm_teardown_and_poison_or():
    be, d = FakeBackend(), FakeDist()
    r = make_res(dist=d, backend=be)
    r.ensure("p", [WindowReq("a", 100, "p2p_window")])
    r.finalize()
    seen = []
    r.set_close("p", lambda: seen.append(("close", len(be.log))))
    r.set_poison("p", lambda: 0)
    r.set_poison("q", lambda: 6)
    assert r.poison_local() == 6
    n_before = len(be.log)
    r.release()
    assert seen and seen[0][0] == "close"
    tail = [x[0] for x in be.log[n_before:]]
    assert (
        tail.index("sync")
        < tail.index("devdestroy")
        < tail.index("dereg")
        < tail.index("commdestroy")
    )
    assert (
        seen[0][1] == n_before + 1
    )  # exactly after the leading device sync, before devcomm/window/comm teardown


def test_logger_capture_independent_of_propagate():
    """The dispatch log must be observable when aiter's logger has propagate=False (real aiter) -- same mechanism as the framework test."""
    parent = logging.getLogger("aiter")
    old = parent.propagate
    parent.propagate = False
    try:
        got = []

        class _H(logging.Handler):
            def emit(self, rec):
                got.append(rec.getMessage())

        lg = logging.getLogger("aiter.gemm_fusion")
        h = _H(level=logging.INFO)
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
        lg.info("event=dispatch probe")
        lg.removeHandler(h)
        assert got == ["event=dispatch probe"]
    finally:
        parent.propagate = old


def test_cpp_h1_h3_l5_present():
    base = os.path.join(jit.CSRC, "host")
    sm = _slurp(os.path.join(base, "sm_fusion_driver.cpp"), encoding="utf-8")
    lg = _slurp(os.path.join(base, "lg_fusion_driver.cpp"), encoding="utf-8")
    assert (
        "own_ctx" in sm and "g_swap_mu" in sm and "single-thread contract" in sm
    )  # H-1
    assert "sm_free_v1" in sm and "sm_poison_fast_v1" in sm  # L-5 / poison
    assert "sm_image_matches(hsaco_path, expected_hsaco_sha256_hex)" in lg  # H-3
    assert lg.index("sm_image_matches") < lg.index(
        "hipModuleLoad"
    )  # gate before any module action
    assert "lg_free_v1" in lg and "lg_poison_v1" in lg
    assert os.path.exists(os.path.join(base, "sha256_image.hpp"))


# ---- exact golden CaseSpec cross-check (case_config.py is importable without torch/GPU) ----
def _golden_case_config():
    import importlib.util

    cands = [os.environ.get("SP4_GOLDEN_DIR", "")] + [
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                *([".."] * n),
                "sp4-attention-fusion-cases-20261009",
            )
        )
        for n in (2, 3)
    ]
    for d in cands:
        p = os.path.join(d, "case_config.py") if d else ""
        if p and os.path.isfile(p):
            spec = importlib.util.spec_from_file_location("golden_case_config", p)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m
    return None


def test_specs_match_actual_golden_casespec_for_every_cell():
    """Page size / N / F / local M come from the REAL golden CaseSpec (dense page128, indexer+reuse page16); a fake
    cache_page_size must not be able to mask the dense row (Evidence V0 finding)."""
    cc = _golden_case_config()
    if cc is None:
        pytest.skip("golden case_config.py not found (set SP4_GOLDEN_DIR)")
    mode_map = {
        "qkv_dense": "dense",
        "qkv_indexer": "indexer",
        "qkv_indexer_reuse": "reuse",
    }
    clear_profile_cache()
    prof = profmod.load_profile(PROFILE)
    seen = 0
    for gmode, mode in mode_map.items():
        for T in (64, 128, 8192, 16384, 32768):
            case = cc.CaseSpec(gmode, T)
            sp = specs.sp_qkv_spec(mode)
            hn = sp.stages[1]
            assert hn.kv_cache == f"shuffle_page{case.cache_page_size}_fp8_ptok", (
                gmode,
                T,
            )
            assert sp.gemm.n == case.gemm_nk[0] and sp.gemm.k == case.gemm_nk[1], (
                gmode,
                T,
            )
            f_exch = sum(
                x.width // x.shards
                for x in sp.stages[0].fields
                if x.name not in sp.stages[0].drop
            )
            golden_f = case.describe()["exchange_shape"][
                1
            ]  # 2304 dense, 2560 indexer AND reuse (the receiver sees a [T,2560] tensor)
            assert golden_f == (2304 if mode == "dense" else 2560), (gmode, T)
            # approved reuse transfer is COMPACT (q/k/v only, F=2304) and is presented to the receiver as a [T,2560] view with row stride 2304
            assert f_exch == (2560 if mode == "indexer" else 2304), (gmode, T)
            small = T in (64, 128)
            full = (
                sp if small else sp.prefix(1)
            )  # small: GEMM+exchange+norm/rope/cache; large: GEMM+exchange only
            row = prof.lookup(full.signature(), "gfx950", 256, 4, case.local_tokens)
            assert (
                row is not None
            ), f"no profile row for {gmode} T={T} (page {case.cache_page_size})"
            assert row.vid == ("sp_qkv.small.v1" if small else "sp_qkv.large.v1")
            seen += 1
    assert seen == 15
    # the BlockSize the runtime derives for the binding equals the golden page
    import re

    for gmode, mode in mode_map.items():
        case = cc.CaseSpec(gmode, 64)
        assert (
            int(
                re.search(
                    r"page(\d+)", specs.sp_qkv_spec(mode).stages[1].kv_cache
                ).group(1)
            )
            == case.cache_page_size
        )


def test_dense_page128_signature_differs_from_page16():
    d = specs.sp_qkv_spec("dense")
    assert d.stages[1].kv_cache == "shuffle_page128_fp8_ptok"
    assert (
        specs.sp_qkv_spec("indexer").stages[1].kv_cache
        == specs.sp_qkv_spec("reuse").stages[1].kv_cache
        == "shuffle_page16_fp8_ptok"
    )
    import dataclasses

    wrong = type(d)(
        d.gemm,
        (
            d.stages[0],
            dataclasses.replace(d.stages[1], kv_cache="shuffle_page16_fp8_ptok"),
        ),
    )
    assert (
        wrong.signature() != d.signature()
    )  # a page16 dense spec must NOT hit the shipped dense row


def test_cache_page_contract_check(monkeypatch):
    from types import SimpleNamespace as NS

    from aiter.gemm_fusion import catalog

    monkeypatch.setattr(catalog, "_CATALOG", {})
    sys.modules.pop("aiter.ops.gemm_fusion.sp_qkv.variants", None)
    sys.modules.pop("aiter.ops.gemm_fusion.sp_qkv.variants_large", None)
    from aiter.ops.gemm_fusion.sp_qkv.variants import check_cache_page

    def caches(page, blocks=4):
        return NS(shape=(blocks, 1, 8, page, 16)), NS(
            shape=(blocks, 1, page // 16, 128, 16)
        )

    for page in (128, 16):
        check_cache_page(*caches(page), page)  # matches
    with pytest.raises(ContractError, match="page mismatch"):
        check_cache_page(*caches(16), 128)  # dense spec (page128) given page16 caches
    with pytest.raises(ContractError, match="page mismatch"):
        check_cache_page(*caches(128), 16)
    with pytest.raises(ContractError):
        check_cache_page(
            NS(shape=(4, 1, 128, 16)), NS(shape=(4, 1, 8, 128, 16)), 128
        )  # not the SHUFFLE rank


def test_plan_block_comes_from_spec_page(monkeypatch):
    """The binding block_size equals the page in the matched spec: dense 128, indexer/reuse 16 (real golden pages)."""
    from aiter.gemm_fusion import catalog

    monkeypatch.setattr(catalog, "_CATALOG", {})
    sys.modules.pop("aiter.ops.gemm_fusion.sp_qkv.variants", None)
    sys.modules.pop("aiter.ops.gemm_fusion.sp_qkv.variants_large", None)
    import aiter.ops.gemm_fusion.sp_qkv.variants as v

    caps = frozenset({"mori_cco", "gfx950_sc_bypass"})
    for mode, T, page in (
        ("dense", 64, 128),
        ("dense", 128, 128),
        ("indexer", 64, 16),
        ("reuse", 128, 16),
    ):
        plan = v.SpQkvSmall(
            specs.sp_qkv_spec(mode),
            specs.SMALL_CELLS[(mode, T)][1],
            MatchCtx("gfx950", 256, 4, T // 4, caps),
        )
        assert plan.block == page, (mode, T)


def test_binding_rejects_wrong_cache_page_before_any_driver_call(monkeypatch):
    """_binding must raise the page ContractError BEFORE it touches the driver (a dense spec with page16 caches must not bind)."""
    import types as _t
    from types import SimpleNamespace as NS

    from aiter.gemm_fusion import catalog

    monkeypatch.setattr(catalog, "_CATALOG", {})
    sys.modules.pop("aiter.ops.gemm_fusion.sp_qkv.variants", None)
    sys.modules.pop("aiter.ops.gemm_fusion.sp_qkv.variants_large", None)
    import aiter.ops.gemm_fusion.sp_qkv.variants as v

    class T:  # minimal tensor stand-in
        def __init__(self, shape, stride=None, dtype="f32"):
            self.shape, self._stride, self.dtype, self.device = (
                tuple(shape),
                stride,
                dtype,
                "cuda:0",
            )
            self.ptr = id(self) * 16

        def data_ptr(self):
            return self.ptr

        def stride(self, i=None):
            s = self._stride or tuple(1 for _ in self.shape)
            return s if i is None else s[i]

        def element_size(self):
            return 1

    fake_torch = _t.ModuleType("torch")
    fake_torch.zeros = lambda *a, **k: T((128,))
    fake_torch.float32 = "f32"
    fake_torch.finfo = lambda dt: NS(max=448.0)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    plan = v.SpQkvSmall(
        specs.sp_qkv_spec("dense"),
        specs.SMALL_CELLS[("dense", 64)][1],
        MatchCtx("gfx950", 256, 4, 16, frozenset({"mori_cco"})),
    )
    assert plan.block == 128
    res = NS(__dict__={})
    res = type("R", (), {})()  # plain object with a __dict__
    calls = []
    plan._context = lambda r: calls.append("context") or NS(
        bind=lambda sb: calls.append("bind")
    )
    x, xs, ws = T((16, 6144), (6144, 1)), T((16, 1)), T((9216,))
    meta = {n: T((128,)) for n in ("q_norm_w", "k_norm_w")}
    meta.update(
        cos_cache=T((1024, 32), (32, 1)),
        sin_cache=T((1024, 32), (32, 1)),
        positions=T((64,)),
        slot_mapping=T((64,)),
        k_scale=T((4, 1, 128), (128, 128, 1)),
        v_scale=T((4, 1, 128), (128, 128, 1)),
    )
    outs = {"q": T((64, 16, 128)), "k": T((64, 1, 128)), "v": T((64, 1, 128))}
    meta.update(
        k_cache=T((4, 1, 8, 16, 16)), v_cache=T((4, 1, 1, 128, 16))
    )  # page16 caches vs a page128 dense spec
    with pytest.raises(ContractError, match="page mismatch"):
        plan._binding(x, xs, ws, outs, meta, res)
    assert "bind" not in calls
    meta.update(
        k_cache=T((4, 1, 8, 128, 16), (16384, 16384, 2048, 16, 1)),
        v_cache=T((4, 1, 8, 128, 16), (16384, 16384, 2048, 16, 1)),
    )
    plan._binding(x, xs, ws, outs, meta, res)  # correct page -> reaches the driver bind
    assert "bind" in calls


def test_package_exports_everything_atom_uses():
    """ATOM's wrapper calls gf.FusionResources / gf.detect_* / gf.MatchCtx / ... on the PACKAGE (Evidence GPU gate AttributeError).
    Scan the ATOM worktree (if present) for every gf.<name> and require it to be exported; plus a pinned minimum list.
    """
    import re

    import aiter.gemm_fusion as g

    required = {
        "FusionResources",
        "ResourceError",
        "CcoBackend",
        "FusionSpec",
        "GemmSpec",
        "Field",
        "SPExchange",
        "HeadNormRopeCache",
        "MatchCtx",
        "ContractError",
        "match_prefix",
        "load_profile",
        "ProfileError",
        "detect_capabilities",
        "detect_arch",
        "detect_cu",
        "WindowReq",
        "Plan",
        "register_variant",
        "Contract",
        "qkv_fields",
    }
    assert not [n for n in required if not hasattr(g, n)]
    atom = os.environ.get("ATOM_WORKTREE") or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "atom-gemm-fusion")
    )
    used = set()
    root = os.path.join(atom, "atom", "model_ops", "gemm_fusion")
    if os.path.isdir(root):
        for fn in os.listdir(root):
            if fn.endswith(".py"):
                used |= set(
                    re.findall(
                        r"\bgf\.([A-Za-z_]\w*)",
                        _slurp(os.path.join(root, fn), encoding="utf-8"),
                    )
                )
        # `gf.profile` in hooks.py is the ATOM config's profile string, not a package attribute: only names that look like aiter API
        missing = sorted(n for n in used if not hasattr(g, n) and n != "profile")
        assert (
            not missing
        ), f"ATOM uses gf.{missing} which aiter.gemm_fusion does not export"


def test_fusion_resources_constructible_via_package_with_atom_kwargs():
    """Exact call shape of atom FusionResourcesWrapper: gf.FusionResources(rank=, world=, group=, device=, caps=)."""
    import aiter.gemm_fusion as g

    res = g.FusionResources(
        rank=0,
        world=4,
        group=None,
        device=0,
        caps=frozenset({"mori_cco"}),
        dist=FakeDist(),
        backend=FakeBackend(),
    )
    assert (
        res.caps == frozenset({"mori_cco"}) and not res.finalized and res.windows == {}
    )
    for attr in (
        "ensure",
        "finalize",
        "quiesce_and_reset",
        "release",
        "poison_local",
        "set_init",
        "set_drain",
        "set_close",
        "set_poison",
    ):
        assert callable(getattr(res, attr))


def _plan_for(vid_mode, T, large):
    from aiter.gemm_fusion import catalog

    catalog._CATALOG.clear()  # callers monkeypatch _CATALOG to a scratch dict; re-register fresh each call
    sys.modules.pop("aiter.ops.gemm_fusion.sp_qkv.variants", None)
    sys.modules.pop("aiter.ops.gemm_fusion.sp_qkv.variants_large", None)
    import aiter.ops.gemm_fusion.sp_qkv.variants as v

    caps = frozenset({"mori_cco", "sdma", "gfx950_sc_bypass"})
    sp = specs.sp_qkv_spec(vid_mode)
    if large:
        sp = sp.prefix(1)
        return v.variants_large.SpQkvLarge(
            sp,
            specs.large_cell_params(vid_mode, T),
            MatchCtx("gfx950", 256, 4, T // 4, caps),
        )
    cells = {(md, m): params for md, m, _vid, params in specs.small_exact_cells()}
    return v.SpQkvSmall(  # the PROFILE's params of this cell (src r19 / r20), not the pre-R12 SMALL_CELLS ones
        sp,
        cells[(vid_mode, T // 4)],
        MatchCtx("gfx950", 256, 4, T // 4, caps),
    )


def _small_owner_cells():
    """One (mode, T) per distinct small OWNER of the profile (mode + params): the P2 rows share their m=16 owner, r20 adds one owner per mode."""
    seen, out = set(), []
    for mode, m, _vid, params in specs.small_exact_cells():
        key = (mode, tuple(sorted(params.items())))
        if key not in seen:
            seen.add(key)
            out.append((mode, 4 * m))
    return out


def test_atom_cross_repo_api_audit(monkeypatch):
    """Everything ATOM (atom/model_ops/gemm_fusion) touches on aiter objects, mirrored from its call sites (audit list in the notes):
    resources.py: plan_owner(plan) = plan.build_key or plan.vid; inner.windows / inner.finalized / ensure(owner, plan.resources()) /
                  finalize / quiesce_and_reset / release / poison_local; plan.resources().
    site.py     : Plan.run(x=,x_scale=,w=,w_scale=,outs=,meta=,res=,stream=); plan.mode in ('reuse') -> reuse_view(res), else recv_view(res);
                  plan.prepare(res=) for 1-stage; plan.prepare(x=,x_scale=,w_scale=,outs=,meta=,res=) for 2-stage; r.plan.vid / n_stages.
    """
    import inspect

    from aiter.gemm_fusion import catalog

    monkeypatch.setattr(catalog, "_CATALOG", {})
    small = _plan_for("dense", 64, False)
    large = _plan_for("reuse", 8192, True)
    for plan in (small, large):
        # ATOM's plan_owner(): getattr(plan, "build_key", None) or getattr(plan, "vid", "plan") -> must be the window owner key
        owner = getattr(plan, "build_key", None) or getattr(plan, "vid", "plan")
        assert owner == plan.owner and owner not in ("", "plan")
        assert all(r.name.startswith(owner + "/") for r in plan.resources())
        assert hasattr(plan, "mode") and callable(plan.run) and callable(plan.prepare)
    assert small.build_key != large.build_key
    # two plans of the same vid but different build must NOT share an owner (ATOM would otherwise collide their windows)
    assert (
        _plan_for("dense", 64, False).build_key
        != _plan_for("indexer", 64, False).build_key
    )
    assert (
        _plan_for("dense", 64, False).build_key
        != _plan_for("dense", 128, False).build_key
    )
    # signatures match ATOM's call shapes
    run_p = set(inspect.signature(small.run).parameters)
    assert {"x", "x_scale", "w", "w_scale", "outs", "meta", "res", "stream"} <= run_p
    assert {"x", "x_scale", "w", "w_scale", "outs", "meta", "res", "stream"} <= set(
        inspect.signature(large.run).parameters
    )
    assert {"x", "x_scale", "w_scale", "outs", "meta", "res"} <= set(
        inspect.signature(small.prepare).parameters
    )
    prep_l = inspect.signature(large.prepare)
    assert "res" in prep_l.parameters and any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in prep_l.parameters.values()
    )  # tolerates extra kwargs
    # reuse_view / recv_view exist exactly where ATOM looks for them
    assert (
        large.mode == "reuse"
        and callable(large.reuse_view)
        and callable(large.recv_view)
    )
    assert not hasattr(small, "recv_view")  # ATOM only calls them for n_fused < 2
    assert small.n_stages == 2 and large.n_stages == 1
    # FusionResources surface used through wrapper.inner / bound
    import aiter.gemm_fusion as g

    r = g.FusionResources(
        rank=0,
        world=4,
        group=None,
        device=0,
        caps=frozenset(),
        dist=FakeDist(),
        backend=FakeBackend(),
    )
    assert r.windows == {} and r.finalized is False
    r.ensure(small.build_key, small.resources())
    assert (
        small.build_key in r.windows
    )  # wrapper.can_ensure() after finalize relies on `owner in inner.windows`
    r.finalize()
    assert r.finalized and small.build_key in r.windows
    assert r.poison_local() == 0
    r.err_reader = (
        lambda: 0
    )  # an R12 plan has the global page: reset() verifies ERR (a device read in production)
    r.quiesce_and_reset()
    r.release()


def _all_plan_requests():
    """Every (owner, [WindowReq]) the profile's owners register (9 small + 9 large), via the real plans' resources()."""

    out = {}
    for mode, T in _small_owner_cells():
        p = _plan_for(mode, T, False)
        out[p.build_key] = p.resources()
    for mode, T in specs.LARGE_CELLS:
        p = _plan_for(mode, T, True)
        out[p.build_key] = p.resources()
    return out


def test_uniform_policy_reproduces_phase1_equal_layout(monkeypatch):
    from aiter.gemm_fusion import catalog
    from aiter.gemm_fusion.resources import heap_bytes_for

    monkeypatch.setattr(catalog, "_CATALOG", {})
    reqs = _all_plan_requests()
    prof = profmod.load_profile(PROFILE)
    wb = prof.meta["window_budget"]
    n_windows = (
        sum(
            len(
                [
                    r
                    for r in v
                    if r.kind in ("p2p_window", "control") and r.share != "gf.global"
                ]
            )
            for v in reqs.values()
        )
        + 1  # every R12 plan requests the SAME per-comm global page: one window
    )
    assert wb["max_windows"] == n_windows == 46
    assert wb["window_bytes"] % (4 << 20) == 0 and wb["window_bytes"] == max(
        r.bytes for v in reqs.values() for r in v
    )
    be, d = FirstFitBackend(0), FakeDist()
    monkeypatch.delenv(
        "AITER_GEMM_FUSION_PROFILE", raising=False
    )  # the budget must come from the EXPLICIT profile=, never from the env
    clear_profile_cache()
    r = make_res(
        dist=d, backend=be, profile=PROFILE, policy="uniform"
    )  # A-3: arena is the default; uniform stays selectable
    assert r.window_bytes == wb["window_bytes"] and r.max_windows == 46
    assert (
        be.heap == heap_bytes_for(wb["window_bytes"], 46)
        and be.heap % (4 << 30) == 0
        and be.heap >= 46 * wb["window_bytes"] + (256 << 20)
    )
    for (
        owner,
        rq,
    ) in reqs.items():  # register ALL 15 plans (the gate02 sequence), in any order
        r.ensure(owner, rq)
    assert len(be.offsets) == 46 and set(be.sizes) == {
        wb["window_bytes"]
    }  # every window equal, 4 MiB multiple
    assert all(
        o % wb["window_bytes"] == 0 for o in be.offsets
    )  # offsets are multiples of the window size, as in Phase-1 (0, W, 2W, ...)
    assert be.next_off <= be.heap - (256 << 20)  # devcomm slack remains


def test_uniform_overflow_and_first_request_sizing(monkeypatch):
    from aiter.gemm_fusion.resources import ResourceError

    monkeypatch.delenv("AITER_GEMM_FUSION_PROFILE", raising=False)
    clear_profile_cache()
    r = make_res(
        dist=FakeDist(), backend=FirstFitBackend(0)
    )  # no profile: size from the first request
    assert r.window_bytes is None
    r.ensure(
        "a", [WindowReq("x", 40 << 20, "p2p_window"), WindowReq("c", 100, "control")]
    )
    assert r.window_bytes == 40 << 20 and r.backend.sizes == [40 << 20, 40 << 20]
    with pytest.raises(ResourceError, match="exceeds the uniform window size"):
        r.ensure("b", [WindowReq("x", 41 << 20, "p2p_window")])
    r.ensure(
        "c", [WindowReq("x", 4 << 20, "p2p_window")]
    )  # smaller plans pad up to the uniform size
    assert r.backend.sizes[-1] == 40 << 20
    r2 = make_res(
        dist=FakeDist(),
        backend=FirstFitBackend(0),
        window_bytes=164 << 20,
        max_windows=3,
    )  # a budget (the profile's window_budget) sizes the window up front
    assert r2.window_bytes == 164 << 20


def test_profile_window_budget_is_regenerated_consistently():
    prof = profmod.load_profile(PROFILE)
    assert dict(prof.meta["window_budget"]) == gen_profile.window_budget()


# ======================================================================= devcomm queue count (gate03_uniform lg_attach_v1 failure)
def test_devcomm_sdma_queues_is_max_not_number_of_plans(monkeypatch):
    """Every large plan declares ONE sdma_queue WindowReq. The shared devcomm must get sdmaQueueCount = 1 (the r11 kernels and
    lg_attach_v1 require exactly 1), not the number of plans (9 large plans used to request 9).
    """
    from aiter.gemm_fusion import catalog

    monkeypatch.setattr(catalog, "_CATALOG", {})
    monkeypatch.delenv("AITER_GEMM_FUSION_PROFILE", raising=False)
    clear_profile_cache()
    be = FakeBackend()
    r = make_res(
        dist=FakeDist(),
        backend=be,
        policy="uniform",
        window_bytes=164 << 20,
        max_windows=27,
    )
    for mode, T in specs.LARGE_CELLS:  # all 9 large plans
        p = _plan_for(mode, T, True)
        r.ensure(p.build_key, p.resources())
    assert sum(1 for q in (x for pl in r.windows for x in [pl]) if q) == 9
    r.finalize()
    assert ("dev", 1) in be.log and not any(x[0] == "dev" and x[1] != 1 for x in be.log)
    assert r.sdma_queues == 1
    # an explicit request for more queues per peer is honoured as the MAX, still not summed
    r2 = make_res(dist=FakeDist(), backend=FakeBackend())
    r2.ensure("a", [WindowReq("q", 2, "sdma_queue")])
    r2.ensure("b", [WindowReq("q", 0, "sdma_queue")])
    r2.ensure("c", [WindowReq("q", 1, "sdma_queue")])
    r2.finalize()
    assert r2.sdma_queues == 2 and ("dev", 2) in r2.backend.log


def test_large_attach_refuses_a_devcomm_with_wrong_queue_count_before_the_driver():
    from types import SimpleNamespace as NS

    from aiter.gemm_fusion import catalog

    catalog._CATALOG.clear()
    plan = _plan_for("dense", 8192, True)
    res = NS(finalized=True, sdma_queues=9, __dict__={})
    res = type("R", (), {"finalized": True, "sdma_queues": 9})()
    with pytest.raises(RuntimeError, match="sdma_queues=9.*exactly 1"):
        plan._context(res)


def test_lg_attach_every_failure_branch_reports_on_stderr():
    src = _slurp(
        os.path.join(jit.CSRC, "host", "lg_fusion_driver.cpp"), encoding="utf-8"
    )
    body = src[src.index("void* lg_attach_v1") : src.index("int32_t lg_free_v1")]
    for stage in (
        "stage=args",
        "stage=hipSetDevice",
        "stage=image_hash",
        "stage=topology",
        "stage=sdma_queues",
        "stage=hipModuleLoad",
        "stage=entry",
    ):
        assert stage in body, stage
    # no silent `return nullptr` left: every one is preceded by a diagnostic fprintf in the same branch
    for i, line in enumerate(body.splitlines()):
        if "return nullptr" in line and "LG_ATTACH" not in line:
            ctx = "\n".join(body.splitlines()[max(0, i - 4) : i + 1])
            assert "LG_ATTACH" in ctx, f"silent failure branch: {line.strip()}"


# ======================================================================= structured attach diagnostics + preflight
def test_decode_attach_diag_covers_every_stage():
    from aiter.ops.gemm_fusion.sp_qkv import large_rt as lrt

    M64 = 1 << 64
    assert "NULL args ['devcomm', 'Sva']" in lrt.decode_attach_diag(
        [1, 0b0000100010, 2, 2048, 1, 0, 0, 0]
    )
    assert "rank=7 M=2048 Q=1" in lrt.decode_attach_diag([1, 0, 7, 2048, 1, 0, 0, 0])
    assert "stage=image_hash" in lrt.decode_attach_diag([2, 0, 0, 0, 0, 0, 0, 0])
    assert "hipError=101" in lrt.decode_attach_diag([3, 101, 0, 0, 0, 0, 0, 0])
    assert "world=4 lsaSize=4 lsaRank=2 myNodeStart=-1" in lrt.decode_attach_diag(
        [4, 4, 4, 2, M64 - 1, 0, 0, 0]
    )
    assert "sdmaNumQueue=9 but the r11 kernels need 1" in lrt.decode_attach_diag(
        [5, 9, 1, 0, 0, 0, 0, 0]
    )
    assert "hipError=700" in lrt.decode_attach_diag([6, 700, 0, 0, 0, 0, 0, 0])
    assert "entry #5 (n3q)" in lrt.decode_attach_diag([7, 500, 5, 0, 0, 0, 0, 0])
    assert "unknown(42)" in lrt.decode_attach_diag([42] + [0] * 7)


def _large_plan_and_wins(res_rank=0, bytes_override=None):
    from aiter.gemm_fusion.resources import Window

    plan = _plan_for("dense", 8192, True)
    b = bytes_override or plan.resources()[0].bytes
    wins = {
        plan.owner + k: Window(plan.owner, plan.owner + k, "p2p_window", b, 11, 4096)
        for k in ("/S", "/R", "/C")
    }
    return plan, wins


def test_preflight_attach_names_each_failing_precondition(monkeypatch):
    import hashlib
    import shutil
    import tempfile
    import uuid
    from pathlib import Path

    tmp_path = Path(tempfile.gettempdir()) / ("pf_" + uuid.uuid4().hex)
    tmp_path.mkdir()
    from types import SimpleNamespace as NS

    from aiter.gemm_fusion import catalog
    from aiter.gemm_fusion.resources import Window

    monkeypatch.setattr(catalog, "_CATALOG", {})
    catalog._CATALOG.clear()
    sys.modules.pop("aiter.ops.gemm_fusion.sp_qkv.variants", None)
    sys.modules.pop("aiter.ops.gemm_fusion.sp_qkv.variants_large", None)
    import aiter.ops.gemm_fusion.sp_qkv.variants_large as vl

    plan, wins = _large_plan_and_wins()
    h = tmp_path / "img.hsaco"
    h.write_bytes(b"code-object")
    sha = hashlib.sha256(b"code-object").hexdigest()
    built = NS(hsaco=str(h), hsaco_sha256=sha)
    good = {
        "lsaSize": 4,
        "lsaRank": 1,
        "sdmaNumQueue": 1,
        "worldSize": 4,
        "myNodeStart": 0,
    }
    res = NS(rank=1, sdma_queues=1, devcomm_info=dict(good))
    vl.preflight_attach(
        res, plan.owner, 8192 // 4, wins, built
    )  # all good -> no raise (M = local rows)

    def fails(msg, **kw):
        r = NS(**{"rank": 1, "sdma_queues": 1, "devcomm_info": dict(good), **kw})
        with pytest.raises(RuntimeError, match=msg):
            vl.preflight_attach(
                r,
                plan.owner,
                kw.get("M", 2048),
                kw.get("wins", wins),
                kw.get("built", built),
            )

    fails("rank 5 outside", rank=5)
    fails("topology", devcomm_info={**good, "lsaRank": 3})
    fails(
        "sdmaNumQueue=9 but the r11 kernels need 1",
        devcomm_info={**good, "sdmaNumQueue": 9},
        sdma_queues=9,
    )
    w2 = dict(wins)
    w2[plan.owner + "/R"] = Window(
        plan.owner, plan.owner + "/R", "p2p_window", 4 << 20, 0, 0
    )
    fails("missing/NULL", wins=w2)
    w3 = dict(wins)
    w3[plan.owner + "/C"] = Window(
        plan.owner, plan.owner + "/C", "control", 4 << 20, 12, 4096
    )
    fails("equal size", wins=w3)
    small = {k: Window(plan.owner, k, "p2p_window", 4 << 20, 11, 4096) for k in wins}
    fails("< required", wins=small)
    fails(
        "hsaco missing", built=NS(hsaco=str(tmp_path / "nope.hsaco"), hsaco_sha256=sha)
    )
    fails("hsaco sha256", built=NS(hsaco=str(h), hsaco_sha256="0" * 64))
    with pytest.raises(RuntimeError, match="M=300 must be a multiple of 256"):
        vl.preflight_attach(res, plan.owner, 300, wins, built)
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_finalize_reads_back_devcomm_info_and_logs_it():

    class B(FakeBackend):
        def devcomm_info(self, comm, dev):
            return {
                "lsaSize": 4,
                "lsaRank": 0,
                "sdmaNumQueue": 1,
                "worldSize": 4,
                "myNodeStart": 0,
            }

    r = make_res(dist=FakeDist(), backend=B())
    r.ensure("a", [WindowReq("q", 0, "sdma_queue")])
    r.finalize()
    assert r.devcomm_info["sdmaNumQueue"] == 1 and r.sdma_queues == 1
    # a backend without the accessor (older gf_comm.so) must not break finalize
    r2 = make_res(dist=FakeDist(), backend=FakeBackend())
    r2.finalize()
    assert r2.devcomm_info is None


def test_lg_attach_writes_structured_diag_on_every_branch():
    src = _slurp(
        os.path.join(jit.CSRC, "host", "lg_fusion_driver.cpp"), encoding="utf-8"
    )
    body = src[src.index("void* lg_attach_v1") : src.index("int32_t lg_free_v1")]
    for code in (
        "diag[0] = 1;",
        "diag[0] = 2;",
        "diag[0] = 3;",
        "diag[0] = 4;",
        "diag[0] = 5;",
        "diag[0] = 6;",
        "diag[0] = 7;",
    ):
        assert code in body, code
    assert "uint64_t* diag" in body and "diag = scratch" in body  # NULL diag tolerated
    assert "gf_devcomm_info_v1" in _slurp(
        os.path.join(jit.CSRC, "host", "gf_comm.cpp"), encoding="utf-8"
    )


# ======================================================================= binding key / capture triage (gate04 'new binding required during capture')
def _sq(text):
    """Source text without whitespace and trailing commas: source-pinning assertions must not depend on how black wraps a line."""
    import re as _re

    return _re.sub(r"\s+", "", text).replace(",)", ")")


class _T:
    """Minimal tensor stand-in with a stable address, shape and strides."""

    _next = 0x10000

    def __init__(self, shape, stride=None, dtype="f32"):
        _T._next += 0x1000
        self.shape, self._stride, self.dtype, self.device, self.ptr = (
            tuple(shape),
            stride,
            dtype,
            "cuda:0",
            _T._next,
        )

    def dim(self):
        return len(self.shape)

    def numel(self):
        n = 1
        for s in self.shape:
            n *= s
        return n

    def is_contiguous(self):
        if self._stride is None:
            return True
        expect, ok = 1, True
        for size, st in zip(reversed(self.shape), reversed(self._stride)):
            if size != 1 and st != expect:
                ok = False
            expect *= size
        return ok

    def data_ptr(self):
        return self.ptr

    def stride(self, i=None):
        s = self._stride or tuple(1 for _ in self.shape)
        return s if i is None else s[i]

    def element_size(self):
        return 1


def _w_ok(rows=9216, k=6144, shuffled=True):
    """A well-formed preshuffled fp8 weight stand-in."""
    w = _T((rows, k), (k, 1), "fp8")
    w.is_shuffled = shuffled
    return w


def _binding_rig(monkeypatch, mode="dense", T=64, src=None, extra=None):
    """A dense small plan whose _context is a recording stub and whose torch is a counting fake."""
    import types as _t
    from types import SimpleNamespace as NS

    from aiter.gemm_fusion import catalog

    monkeypatch.setattr(catalog, "_CATALOG", {})
    sys.modules.pop("aiter.ops.gemm_fusion.sp_qkv.variants", None)
    sys.modules.pop("aiter.ops.gemm_fusion.sp_qkv.variants_large", None)
    import aiter.ops.gemm_fusion.sp_qkv.variants as v

    zeros_calls = []
    fake_torch = _t.ModuleType("torch")
    fake_torch.zeros = lambda *a, **k: (zeros_calls.append(1), _T((128,)))[1]
    fake_torch.float32, fake_torch.float8_e4m3fn, fake_torch.int64 = "f32", "fp8", "i64"
    fake_torch.finfo = lambda dt: NS(max=448.0)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    cell_params = dict(
        specs.SMALL_CELLS[(mode, T if (mode, T) in specs.SMALL_CELLS else 64)][1]
    )
    if src:  # e.g. "r19": the R12 (abi 3) build of the same cell
        cell_params["src"] = src
    cell_params.update(extra or {})  # e.g. rx_bypass=1 for r20
    plan = v.SpQkvSmall(
        specs.sp_qkv_spec(mode),
        cell_params,
        MatchCtx("gfx950", 256, 4, T // 4, frozenset({"mori_cco", "gfx950_sc_bypass"})),
    )
    binds = []
    plan._context = lambda r: NS(bind=lambda sb: binds.append(sb) or object())
    M = T // 4
    page = plan.block
    n_rows = 9856 if mode in ("indexer", "reuse") else 9216
    st = {
        "x": _T((M, 6144), (6144, 1), "fp8"),
        "xs": _T((M, 1), (1, 1)),
        "ws": _T((n_rows,), (1,)),
        "w": _w_ok(n_rows),
    }
    meta = {n: _T((128,)) for n in ("q_norm_w", "k_norm_w")}
    meta.update(
        cos_cache=_T((1024, 32), (32, 1)),
        sin_cache=_T((1024, 32), (32, 1)),
        positions=_T((T,), (1,), "i64"),
        slot_mapping=_T((T,), (1,), "i64"),
        k_cache=_T((4, 1, 8, page, 16), (page * 128, page * 128, page * 16, 16, 1)),
        v_cache=_T((4, 1, page // 16, 128, 16), (page * 128, page * 128, 2048, 16, 1)),
        k_scale=_T((4, 1, page), (page, page, 1)),
        v_scale=_T((4, 1, page), (page, page, 1)),
    )
    outs = {"q": _T((T, 16, 128)), "k": _T((T, 1, 128)), "v": _T((T, 1, 128))}
    res = type("R", (), {})()
    return plan, st, meta, outs, res, binds, zeros_calls


def test_binding_reuse_when_all_components_unchanged_and_no_per_call_allocation(
    monkeypatch,
):
    plan, st, meta, outs, res, binds, zeros_calls = _binding_rig(monkeypatch)
    b1 = plan._binding(st["x"], st["xs"], st["ws"], outs, meta, res)
    for _ in range(5):  # the eager run path calls _binding again with the same tensors
        assert (
            plan._binding(st["x"], st["xs"], st["ws"], outs, meta, res, in_capture=True)
            is b1
        )  # even in capture: a hit, no raise
    assert len(binds) == 1
    assert (
        len(zeros_calls) == 1
    )  # the 128-float placeholder is allocated ONCE (was once per call: setdefault evaluated torch.zeros eagerly)


def test_capture_miss_names_the_component_that_moved(monkeypatch):
    plan, st, meta, outs, res, binds, _z = _binding_rig(monkeypatch)
    plan._binding(st["x"], st["xs"], st["ws"], outs, meta, res)  # prepare-time binding
    meta2 = dict(meta)
    meta2["q_norm_w"] = _T(
        (128,)
    )  # e.g. ATOM re-created the float32 copy of the norm weight
    with pytest.raises(
        RuntimeError, match=r"graph capture.*q_norm_w: \d+ -> \d+"
    ) as ei:
        plan._binding(st["x"], st["xs"], st["ws"], outs, meta2, res, in_capture=True)
    assert (
        "x_scale" not in str(ei.value).split("differs from the closest")[1]
    )  # ONLY the moved component is listed
    meta3 = dict(meta)
    meta3["k_cache"] = _T(meta["k_cache"].shape, meta["k_cache"]._stride)
    with pytest.raises(RuntimeError, match=r"k_cache: \d+ -> \d+"):
        plan._binding(st["x"], st["xs"], st["ws"], outs, meta3, res, in_capture=True)
    assert len(binds) == 1  # nothing was bound by the failed attempts


def test_eager_second_binding_is_logged_with_the_diff(monkeypatch):
    plan, st, meta, outs, res, binds, _z = _binding_rig(monkeypatch)
    plan._binding(st["x"], st["xs"], st["ws"], outs, meta, res)
    msgs = []

    class H(logging.Handler):
        def emit(self, rec):
            msgs.append(rec.getMessage())

    lg = logging.getLogger("aiter.gemm_fusion")
    h, old = H(level=logging.INFO), lg.level
    lg.addHandler(h)
    lg.setLevel(logging.INFO)
    try:
        xs2 = _T(
            (16, 1)
        )  # the eager call used a DIFFERENT x_scale tensor than prepare did
        plan._binding(st["x"], xs2, st["ws"], outs, meta, res)
    finally:
        lg.removeHandler(h)
        lg.setLevel(old)
    assert len(binds) == 2
    assert any(
        "event=binding new" in m and "x_scale:" in m and "binding #2" in m for m in msgs
    ), msgs  # the hidden eager rebinding is now visible


def test_stride_changes_make_a_new_binding_not_stale_reuse(monkeypatch):
    plan, st, meta, outs, res, _binds, _z = _binding_rig(monkeypatch)
    plan._binding(st["x"], st["xs"], st["ws"], outs, meta, res)
    meta2 = dict(meta)
    kc = meta["k_cache"]
    meta2["k_cache"] = type(kc)(
        kc.shape, (kc._stride[0] * 2,) + tuple(kc._stride[1:])
    )  # same shape, different strides (and a new object)
    meta2["k_cache"].ptr = (
        kc.ptr
    )  # SAME address, different layout: only the stride part of the key can catch it
    with pytest.raises(RuntimeError, match=r"k_cache\.strides"):
        plan._binding(st["x"], st["xs"], st["ws"], outs, meta2, res, in_capture=True)


def test_strideA_is_a_per_launch_argument_not_baked_into_the_binding():
    import inspect

    from aiter.ops.gemm_fusion.sp_qkv import small_rt as rt

    assert "strideA" in inspect.signature(rt.SmallContext.launch).parameters
    src = _slurp(
        os.path.join(jit.CSRC, "host", "sm_fusion_driver.cpp"), encoding="utf-8"
    )
    assert (
        "const void* A, const void* W, uint32_t strideA, void* stream" in src
        and "c->b.strideA = strideA" in src
    )
    pysrc = _slurp(
        os.path.join(os.path.dirname(rt.__file__), "variants.py"), encoding="utf-8"
    )
    assert _sq(
        "sc.launch(blk, x.data_ptr(), w.data_ptr(), x.stride(0) * x.element_size()"
    ) in _sq(pysrc)


def test_run_passes_the_per_launch_x_layout(monkeypatch):
    from types import SimpleNamespace as NS

    plan, st, meta, outs, res, _binds, _z = _binding_rig(monkeypatch)
    calls = []
    plan._context = lambda r: NS(
        bind=lambda sb: object(), launch=lambda *a: calls.append(a)
    )

    sys.modules["torch"].cuda = NS(is_current_stream_capturing=lambda: False)
    monkeypatch.setattr(
        "aiter.ops.gemm_fusion.sp_qkv.small_rt.stream_handle", lambda s: "S"
    )
    w = _w_ok()
    x_padded = _T(
        (16, 6144), (8192, 1), "fp8"
    )  # a row-padded activation view: stride(0) = 8192 bytes
    plan.run(
        x=x_padded,
        x_scale=st["xs"],
        w=w,
        w_scale=st["ws"],
        outs=outs,
        meta=meta,
        res=res,
    )
    _blk, a_ptr, w_ptr, stride_a, stream = calls[0]
    assert (a_ptr, w_ptr, stride_a, stream) == (x_padded.ptr, w.ptr, 8192, "S")


# ======================================================================= process-global HIP state hygiene (E2: 'operation not supported' after FusionResources init)
def _host_src(name):
    return _slurp(os.path.join(jit.CSRC, "host", name), encoding="utf-8")


def test_no_bare_hipSetDevice_in_gemm_fusion_host_code():
    """hipSetDevice may appear ONLY inside gf_device_scope.hpp (the scope guard) and in the sm driver's macro that reroutes the vendored Phase-1 calls."""
    import re

    for name in ("gf_comm.cpp", "lg_fusion_driver.cpp", "sm_fusion_driver.cpp"):
        code = re.sub(r"//[^\n]*", "", _host_src(name))  # drop comments
        code = re.sub(
            r'"(?:\\.|[^"\\])*"', '""', code
        )  # drop string literals (log text mentions hipSetDevice)
        define_ok = "#define hipSetDevice(x) gf::set_device(x)" in code
        calls = re.findall(
            r"\bhipSetDevice\s*\(",
            code.replace("#define hipSetDevice(x) gf::set_device(x)", ""),
        )
        if name == "sm_fusion_driver.cpp":
            assert (
                calls == []
            ), "the sm driver may only mention hipSetDevice in the #define/#undef + comments"
            assert define_ok and "#undef hipSetDevice" in code
            raw = _host_src(
                name
            )  # (string literals are blanked in `code`, so the include line is only visible in the raw text)
            assert (
                raw.index("#define hipSetDevice")
                < raw.index('#include "driver_sm.cpp"')
                < raw.index("#undef hipSetDevice")
            )
        else:
            assert calls == [], f"{name}: bare hipSetDevice call"
    hdr = _host_src("gf_device_scope.hpp")
    assert (
        hdr.count("hipSetDevice(") >= 2
        and "hipSetDevice(prev_)" in hdr
        and "hipGetLastError()" in hdr
    )


def test_every_entry_point_runs_in_a_device_scope_and_comm_device_not_rank():
    gf = _host_src("gf_comm.cpp")
    assert (
        "int32_t gf_comm_create_v1(uint32_t rank, uint32_t world, int32_t device," in gf
    )  # explicit device, not derived from rank
    assert "comm->hipDev != device" in gf and "comm_dev(comm)" in gf
    for fn in (
        "gf_win_register_v1",
        "gf_win_view_v1",
        "gf_win_info_v1",
        "gf_win_free_v1",
        "gf_mem_info_v1",
        "gf_devcomm_create_v1",
        "gf_devcomm_destroy_v1",
        "gf_win_deregister_v1",
        "gf_comm_destroy_v1",
        "gf_zero_v1",
        "gf_device_sync_v1",
    ):
        body = gf[gf.index("int32_t " + fn) :]
        body = body[: body.index("\n}\n")]
        assert "gf::DeviceScope" in body, fn
    for fn in ("gf_comm_create_v1", "gf_win_register_v1", "gf_devcomm_create_v1"):
        body = gf[gf.index("int32_t " + fn) :]
        assert (
            "flush_sticky" in body[: body.index("\n}\n")]
        ), fn  # sticky error from peer-access enabling must not leak into torch
    sm = _host_src("sm_fusion_driver.cpp")
    for fn in (
        "sm_attach_v1",
        "sm_bind_new_v1",
        "sm_launch_blk_v1",
        "sm_blk_free_v1",
        "sm_free_v1",
        "sm_poison_fast_v1",
    ):
        body = sm[sm.index("int32_t " + fn) :]
        assert "gf::DeviceScope" in body[: body.index("\n}\n")], fn
    lg = _host_src("lg_fusion_driver.cpp")
    assert (
        "static_cast<mori::cco::ccoComm*>(comm)->hipDev" in lg
    )  # attach uses the comm's device
    for fn in (
        "lg_free_v1",
        "lg_poison_v1",
        "lg_init_control_v1",
        "lg_launch_v1",
        "lg_report_v1",
        "lg_drain_v1",
    ):
        body = lg[lg.index("int32_t " + fn) :]
        assert "gf::DeviceScope" in body[: body.index("\n}\n")], fn


def test_python_calls_only_the_scoped_large_entry_points():
    import ast  # noqa: F401

    src = _slurp(
        os.path.join(os.path.dirname(specs.__file__), "large_rt.py"), encoding="utf-8"
    )
    for bare in (
        "lg_comm_init_control",
        "lg_comm_launch",
        "lg_comm_read_report",
        "lg_comm_drain_all",
    ):
        assert (
            f"lib.{bare}" not in src
        ), f"{bare} must go through its device-scoped wrapper (no ctypes attribute access to the bare function)"
    for wrapped in (
        "lg_init_control_v1",
        "lg_launch_v1",
        "lg_report_v1",
        "lg_drain_v1",
    ):
        assert wrapped in src


def test_backend_requires_explicit_device_and_resources_defaults_to_torch_current_device(
    monkeypatch,
):
    import types as _t

    from aiter.gemm_fusion import resources as rmod
    from aiter.gemm_fusion.resources import ResourceError

    seen = []

    class B(FakeBackend):
        def comm_create(self, rank, world, uid, vmm, device=None):
            seen.append((rank, device))
            return 1

    # torch present: the device is torch's current device (3), NOT the SP rank (1)
    fake_torch = _t.ModuleType("torch")
    fake_torch.cuda = _t.SimpleNamespace(
        is_available=lambda: True, current_device=lambda: 3
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    r = make_res(rank=1, dist=FakeDist(), backend=B())
    assert r.device == 3 and seen == [(1, 3)]
    # an explicit device wins
    seen.clear()
    make_res(rank=1, device=5, dist=FakeDist(), backend=B())
    assert seen == [(1, 5)]
    # no torch/cuda: falls back to the rank (single-process tools)
    fake_torch.cuda = _t.SimpleNamespace(
        is_available=lambda: False, current_device=lambda: 9
    )
    seen.clear()
    make_res(rank=2, dist=FakeDist(), backend=B())
    assert seen == [(2, 2)]
    # the ctypes backend refuses a missing device instead of guessing
    be = object.__new__(rmod.CcoBackend)
    with pytest.raises(ResourceError, match="explicit HIP device"):
        rmod.CcoBackend.comm_create(be, 0, 4, b"u" * 128, 1 << 32)


def test_fusion_resources_budget_from_explicit_profile_with_env_unset(monkeypatch):
    from aiter.gemm_fusion.resources import heap_bytes_for

    monkeypatch.delenv("AITER_GEMM_FUSION_PROFILE", raising=False)
    monkeypatch.delenv("AITER_GEMM_FUSION_WINDOW_MIB", raising=False)
    monkeypatch.delenv("AITER_GEMM_FUSION_WINDOW_POLICY", raising=False)
    clear_profile_cache()
    for prof in (PROFILE, profmod.load_profile(PROFILE)):  # a path AND a Profile object
        d, be = FakeDist(), FirstFitBackend(0)
        r = make_res(dist=d, backend=be, profile=prof, policy="uniform")
        assert (
            r.window_bytes == 171966464 == 164 << 20 and r.max_windows == 46
        )  # the shipped profile's own numbers
        assert r.profile is not None and r.profile.name == "minimax_m3_sp4_gfx950_v1"
        assert be.heap == heap_bytes_for(
            171966464, 40
        ) and be.heap >= 40 * 171966464 + (256 << 20)
        # the failing ATOM sequence: a 4 MiB small plan first, then a 40 MiB large plan: both fit the 164 MiB uniform window
        r.ensure(
            "small",
            [WindowReq("rx", 1 << 20, "p2p_window"), WindowReq("ctl", 100, "control")],
        )
        r.ensure(
            "large",
            [
                WindowReq("S", 40 << 20, "p2p_window"),
                WindowReq("R", 40 << 20, "p2p_window"),
                WindowReq("C", 40 << 20, "control"),
            ],
        )
        assert set(be.sizes) == {171966464}


def test_profile_name_resolves_without_env(monkeypatch):

    monkeypatch.delenv("AITER_GEMM_FUSION_PROFILE", raising=False)
    clear_profile_cache()
    r = make_res(
        dist=FakeDist(), backend=FirstFitBackend(0), profile="minimax_m3_sp4_gfx950_v1"
    )  # a bare name -> configs/gemm_fusion/<name>.json
    assert r.window_bytes == 171966464 and r.max_windows == 46


def test_env_profile_is_never_read_by_fusion_resources(monkeypatch):
    """Even with AITER_GEMM_FUSION_PROFILE pointing at a valid profile, FusionResources(profile=None) must NOT use it silently."""

    monkeypatch.setenv("AITER_GEMM_FUSION_PROFILE", PROFILE)
    clear_profile_cache()
    r = make_res(dist=FakeDist(), backend=FirstFitBackend(0))
    assert (
        r.profile is None and r.window_bytes is None and r.max_windows == 0
    )  # unknown budget -> first-request sizing, with a warning


def test_profile_without_window_budget_warns_and_falls_back_to_first_request(
    monkeypatch,
):

    monkeypatch.delenv("AITER_GEMM_FUSION_WINDOW_MIB", raising=False)
    msgs = []

    class H(logging.Handler):
        def emit(self, rec):
            msgs.append(rec.getMessage())

    lg = logging.getLogger("aiter.gemm_fusion")
    h, old = H(level=logging.WARNING), lg.level
    lg.addHandler(h)
    lg.setLevel(logging.WARNING)
    try:
        bare = profmod.parse_profile({"profile": "nobudget", "rows": []})
        r = make_res(
            dist=FakeDist(), backend=FirstFitBackend(0), profile=bare
        )  # must NOT raise
        make_res(dist=FakeDist(), backend=FirstFitBackend(0))
    finally:
        lg.removeHandler(h)
        lg.setLevel(old)
    assert r.window_bytes is None and r.max_windows == 0 and r.profile is bare
    assert any(
        "event=resources no_budget" in m and "profile has no window_budget" in m
        for m in msgs
    )
    assert any(
        "event=resources no_budget" in m and "built without profile=" in m for m in msgs
    )


def test_bad_profile_argument_is_an_error():

    with pytest.raises(profmod.ProfileError):
        make_res(dist=FakeDist(), backend=FirstFitBackend(0), profile=123)
    with pytest.raises(profmod.ProfileError):
        make_res(
            dist=FakeDist(), backend=FirstFitBackend(0), profile="/no/such/profile.json"
        )


def test_no_implicit_env_profile_lookup_in_the_gemm_fusion_runtime():
    """Audit pin: the only reader of AITER_GEMM_FUSION_PROFILE is profile.load_profile(None) (documented top-level convenience, used by
    match_prefix(profile=None)); resources.py and every variant take the profile explicitly.
    """
    import re

    root = os.path.join(os.path.dirname(profmod.__file__), "..")
    hits = []
    for base in (
        os.path.join(root, "gemm_fusion"),
        os.path.join(root, "ops", "gemm_fusion"),
    ):
        for dp, _dn, fns in os.walk(base):
            for fn in fns:
                if (
                    fn.endswith(".py") and fn != "test_utils.py"
                ):  # test_utils is test code, not runtime
                    src = _slurp(os.path.join(dp, fn), encoding="utf-8")
                    code = re.sub(r"#[^\n]*", "", src)
                    for m in re.finditer(
                        r"ENV_PROFILE|AITER_GEMM_FUSION_PROFILE|load_profile\(\s*\)",
                        code,
                    ):
                        hits.append((fn, m.group(0)))
    files = {f for f, _ in hits}
    assert files <= {
        "profile.py"
    }, hits  # resources.py / variants*.py / match.py must not read the env profile
    pr = _slurp(os.path.join(root, "gemm_fusion", "resources.py"), encoding="utf-8")
    assert "load_profile()" not in pr and "ENV_PROFILE" not in pr


# ======================================================================= D14: REG_BARRIER (E2 seam), shared_large (D-A), late registration (G-late)
def _all_plans(monkeypatch):
    from aiter.gemm_fusion import catalog

    monkeypatch.setattr(catalog, "_CATALOG", {})
    out = []
    for mode, T in _small_owner_cells():
        p = _plan_for(mode, T, False)
        out.append((p.build_key, p.resources()))
    for mode, T in specs.LARGE_CELLS:
        p = _plan_for(mode, T, True)
        out.append((p.build_key, p.resources()))
    return out


def test_shared_set_hooks_are_registered_once_and_later_contexts_do_not_reinit(
    monkeypatch,
):

    r = make_res(
        dist=FakeDist(), backend=FakeBackend(), window_bytes=4 << 20, max_windows=1
    )
    calls = []
    assert (
        r.set_shared_hooks(
            init=lambda: calls.append("init1"),
            drain=lambda: calls.append("drain1"),
            poison=lambda: 0,
        )
        is True
    )
    assert calls == ["init1"]  # the first context initialises the live-to-be block
    assert (
        r.set_shared_hooks(
            init=lambda: calls.append("init2"),
            drain=lambda: calls.append("drain2"),
            poison=lambda: 5,
        )
        is False
    )
    assert calls == [
        "init1"
    ]  # the second attach did NOT run init on the (possibly live) block
    r.finalize()
    r.quiesce_and_reset()
    assert calls == [
        "init1",
        "drain1",
        "init1",
    ]  # one drain + one init per reset, from the FIRST context only
    assert (
        r.poison_local() == 0
    )  # the first context's poison read is the one registered


def test_preflight_accepts_a_shared_set_and_still_rejects_unequal_per_owner(
    monkeypatch,
):
    import hashlib
    from types import SimpleNamespace as NS

    from aiter.gemm_fusion import catalog
    from aiter.gemm_fusion.resources import Window

    monkeypatch.setattr(catalog, "_CATALOG", {})
    catalog._CATALOG.clear()
    sys.modules.pop("aiter.ops.gemm_fusion.sp_qkv.variants", None)
    sys.modules.pop("aiter.ops.gemm_fusion.sp_qkv.variants_large", None)
    import aiter.ops.gemm_fusion.sp_qkv.variants_large as vl

    plan = _plan_for("dense", 8192, True)
    b = plan.resources()[0].bytes
    import shutil
    import tempfile
    import uuid
    from pathlib import Path

    tmp = Path(tempfile.gettempdir()) / ("pf2_" + uuid.uuid4().hex)
    tmp.mkdir()
    try:
        h = tmp / "i.hsaco"
        h.write_bytes(b"x")
        built = NS(hsaco=str(h), hsaco_sha256=hashlib.sha256(b"x").hexdigest())
        res = NS(
            rank=0,
            sdma_queues=1,
            devcomm_info={
                "lsaSize": 4,
                "lsaRank": 0,
                "sdmaNumQueue": 1,
                "worldSize": 4,
                "myNodeStart": 0,
            },
        )
        mk = lambda n, k, by, sh: Window(
            "<shared>" if sh else plan.owner, plan.owner + n, k, by, 11, 4096, shared=sh
        )
        shared = {
            plan.owner + "/S": mk("/S", "p2p_window", b, True),
            plan.owner + "/R": mk("/R", "p2p_window", b, True),
            plan.owner + "/C": mk("/C", "control", 4 << 20, True),
        }
        vl.preflight_attach(
            res, plan.owner, 2048, shared, built
        )  # S=R=164 MiB, C=4 MiB: fine when SHARED
        unshared = {
            k: Window(
                plan.owner,
                k,
                "control" if k.endswith("C") else "p2p_window",
                by.bytes,
                11,
                4096,
            )
            for k, by in shared.items()
        }
        with pytest.raises(RuntimeError, match="windows must be equal size"):
            vl.preflight_attach(
                res, plan.owner, 2048, unshared, built
            )  # the per-owner convention is still enforced
        bad = dict(shared)
        bad[plan.owner + "/C"] = mk("/C", "control", 4096, True)
        with pytest.raises(RuntimeError, match="shared control window"):
            vl.preflight_attach(res, plan.owner, 2048, bad, built)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---- A1 audit (CPU): the vendored r11 large kernel keeps NO plan-specific epoch/target state in kernargs; the control protocol is plan-independent
def _r11(*parts):
    return _slurp(os.path.join(jit.CSRC, "large", "r11", *parts), encoding="utf-8")


def test_A1_audit_remote_targets_and_epoch_come_from_the_device_not_kernargs():
    import re

    abi = _r11("interface", "lg_comm_abi_v2.h")
    struct = abi[
        abi.index("struct LgCommArgsV2") : abi.index(
            "};", abi.index("struct LgCommArgsV2")
        )
    ]
    members = re.findall(
        r"^\s*(?:const\s+)?[\w:<>\*]+\s+\**(\w+)(?:\[\d+\])?\s*;", struct, re.MULTILINE
    )
    assert not [
        m
        for m in members
        if re.search(r"epoch|credit|arrival|^ep$|state", m, re.IGNORECASE)
    ], members  # no epoch/credit/arrival kernarg
    assert {"Swin", "Rwin", "Cwin", "M", "F", "chunks", "rank"} <= set(members)
    dev = _r11("device", "lg_comm_device.hpp") + _r11("device", "lg_r6x.hpp")
    assert (
        "ld_u64(cw(a, LGC_EP_OFF))" in dev
    )  # epoch is READ from the control window at launch
    assert (
        dev.count("cw_peer(a, ") >= 3 and "ccoGetLsaPeerPtr(a.Cwin, peer, off)" in dev
    )  # remote control targets: from the window handle + peer rank
    assert (
        "a.Rwin" in dev and "a.Swin" in dev
    )  # remote data targets: from the window handles
    assert re.search(
        r"slot\s*=\s*2ull\s*\*\s*a\.M\s*\*\s*a\.F", dev
    )  # data layout derives from per-launch M and F, not from the window
    # the control layout constants do not depend on the build (dense vs indexer) or on M: sharing one block across builds is layout-compatible
    offs = re.findall(r"#define\s+LGC_\w+_OFF\s+([^\n/]+)", abi)
    assert offs and not [
        o for o in offs if "LG_X_INDEXER" in o or "LG_COMM_F" in o or "a." in o
    ]
    # per-epoch device reset covers every plan: 4 x LG_MAX_CHUNKS words, and LG_MAX_CHUNKS covers the largest plan (M=8192 -> 32 chunks)
    assert (
        re.search(r"#define\s+LG_MAX_CHUNKS\s+32u", abi)
        and "4 * static_cast<int>(LG_MAX_CHUNKS)" in dev
    )
    assert max(T // 4 // 256 for (_m, T) in specs.LARGE_CELLS) == 32


def test_A1_audit_host_launch_args_are_per_plan_but_target_windows_are_handles():
    """What IS per plan in the captured kernarg block (M, N, F, chunks, plan code, grid) is consistent within ONE launch; the window arguments
    are the shared handles (c->S / c->R / c->C); nothing epoch- or address-like is baked.
    """
    import re

    drv = _r11("host", "lg_comm_driver.cpp")
    launch = drv[drv.index("int32_t lg_comm_launch") :]
    launch = launch[: launch.index("// Control/report readback")]
    assert (
        "a.Swin = c->S; a.Rwin = c->R; a.Cwin = c->C;" in launch
    )  # windows enter the kernel as the context's handles
    assert (
        "a.M = c->M" in launch
        and "a.F = LG_COMM_F" in launch
        and "a.chunks = c->M / LG_CHUNK_ROWS" in launch
    )  # per-plan layout constants
    assert (
        "a.reserved[0] = LG_R6_PLAN_CODE_N(npp, hg)" in launch
    )  # plan code: build/host agreement check only
    assert not re.search(
        r"a\.(epoch|ep|credit|arrival)\b", launch
    )  # no epoch/credit/arrival is passed by value
    # the context itself carries no epoch either
    ctx = drv[
        drv.index("struct LgCommCtx") : drv.index("};", drv.index("struct LgCommCtx"))
    ]
    assert not re.search(r"epoch|\bep\b", ctx)


# ======================================================================= zeroing (D14 addendum r1a: H-Z) + reviewer conditions C-S3/C-S4/C-S6
class _ZeroRecBackend(RealisticFirstFit):
    """Records the ORDER of registrations, zeroing, devcomm creation and device syncs."""

    def __init__(self):
        super().__init__()
        self.ev = []
        self.mode = None

    def set_zero_mode(self, mode):
        self.mode = mode

    def win_register(self, comm, nbytes):
        self.ev.append(("reg", nbytes))
        return super().win_register(comm, nbytes)

    def zero(self, dev, loc, n):
        self.ev.append(("zero", loc - self.base))

    def devcomm_create(self, comm, q):
        self.ev.append(("devcomm", q))
        return super().devcomm_create(comm, q)


def _ensure_pairs(r, n=3):
    for i in range(n):
        r.ensure(
            f"o{i}", [WindowReq("a", 100, "p2p_window"), WindowReq("b", 100, "control")]
        )


def test_default_zero_mode_defers_zeroing_until_after_all_registrations_and_the_devcomm(
    monkeypatch,
):

    for var in ("AITER_GEMM_FUSION_WINDOW_POLICY", "AITER_GEMM_FUSION_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    be = _ZeroRecBackend()
    r = make_res(dist=FakeDist(), backend=be, window_bytes=4 << 20, max_windows=6)
    _ensure_pairs(r)
    kinds = [e[0] for e in be.ev]
    assert (
        kinds == ["reg"] * 6 and r.pending_zero == 6
    )  # NOTHING zeroed between registrations (the H-Z suspect: zero after windows 2 and 4)
    r.finalize()
    kinds = [e[0] for e in be.ev]
    assert (
        kinds == ["reg"] * 6 + ["devcomm"] + ["zero"] * 6 and r.pending_zero == 0
    )  # all registrations, then the devcomm, THEN each window zeroed once
    assert len({e[1] for e in be.ev if e[0] == "zero"}) == 6


def test_launch_refuses_windows_that_are_not_zeroed_yet(monkeypatch):
    from types import SimpleNamespace as NS

    plan, st, meta, outs, res, _binds, _z = _binding_rig(monkeypatch)
    res.pending_zero = 3
    sys.modules["torch"].cuda = NS(is_current_stream_capturing=lambda: False)
    with pytest.raises(RuntimeError, match="3 CCO window.*not yet zeroed.*finalize"):
        plan.run(
            x=st["x"],
            x_scale=st["xs"],
            w=st["w"],
            w_scale=st["ws"],
            outs=outs,
            meta=meta,
            res=res,
        )
    res.pending_zero = 0
    plan._context = lambda r: NS(bind=lambda sb: object(), launch=lambda *a: None)
    monkeypatch.setattr(
        "aiter.ops.gemm_fusion.sp_qkv.small_rt.stream_handle", lambda s: "S"
    )
    plan.run(
        x=st["x"],
        x_scale=st["xs"],
        w=st["w"],
        w_scale=st["ws"],
        outs=outs,
        meta=meta,
        res=res,
    )  # zeroed: runs


def test_C_S3_documented_in_code_and_policy_requires_the_same_large_plan_sequence():
    """C-S3: the stricter contract and the host-side detection must be stated where the policy is defined."""
    from aiter.gemm_fusion import resources as rmod

    doc = rmod.__doc__
    assert (
        "same SEQUENCE of large plans" in doc
        and "digest" in doc
        and "not by the device" in doc
    )
    vl = _slurp(
        os.path.join(os.path.dirname(specs.__file__), "variants_large.py"),
        encoding="utf-8",
    )
    assert "same SEQUENCE" in vl


def test_C_S6_budget_log_has_registered_bytes_heap_and_hbm_delta(monkeypatch):

    monkeypatch.delenv("AITER_GEMM_FUSION_PROFILE", raising=False)
    msgs = []

    class H(logging.Handler):
        def emit(self, rec):
            msgs.append(rec.getMessage())

    class MemBackend(RealisticFirstFit):
        def __init__(self):
            super().__init__()
            self.free = 100 << 30

        def mem_info(self, device):
            return (self.free, 288 << 30)

        def win_register(self, comm, n):
            self.free -= n
            return super().win_register(comm, n)

    lg = logging.getLogger("aiter.gemm_fusion")
    h, old = H(level=logging.INFO), lg.level
    lg.addHandler(h)
    lg.setLevel(logging.INFO)
    try:
        r = make_res(
            dist=FakeDist(),
            backend=MemBackend(),
            window_bytes=4 << 20,
            max_windows=6,
            policy="uniform",
        )
        _ensure_pairs(r)
        r.finalize()
    finally:
        lg.removeHandler(h)
        lg.setLevel(old)
    bud = [m for m in msgs if "event=resources budget" in m]
    hbm = [m for m in msgs if m.startswith("event=resources hbm")]
    assert (
        bud
        and "registered_bytes=25165824" in bud[0]
        and "heap_bytes=" in bud[0]
        and "peer_mapped_va_bytes=" in bud[0]
    )
    assert (
        hbm
        and "delta_bytes=25165824" in hbm[0]
        and "registered_bytes=25165824" in hbm[0]
        and "delta_over_registered=1.000" in hbm[0]
    )
    assert "registered_bytes counts the fusion windows only" in bud[
        0
    ] and "NOT included and are unattributed" in bud[0].replace(
        "\n", " "
    )  # A-7: never claim MORI's overhead as fusion-free


# ======================================================================= D14 r1b/r1c: abs_aligned policy, shared_large absolute alignment, VA logging, probe sequences
MiB = 1 << 20


def _arena_res(monkeypatch, policy="arena"):

    for var in (
        "AITER_GEMM_FUSION_PROFILE",
        "AITER_GEMM_FUSION_ARENA_MIB",
        "AITER_GEMM_FUSION_ZERO_MODE",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AITER_GEMM_FUSION_WINDOW_POLICY", policy)
    clear_profile_cache()
    be = ArenaBackend()
    return make_res(dist=FakeDist(), backend=be, profile=PROFILE), be


def test_arena_registers_exactly_one_window_and_every_plan_window_is_a_view(
    monkeypatch,
):
    r, be = _arena_res(monkeypatch, "arena")
    plans = _all_plans(monkeypatch)
    import random

    random.Random(3).shuffle(plans)
    for owner, rq in plans:
        r.ensure(owner, rq)
    assert (
        len(be.regs) == 1 and len(r._regs) == 1
    )  # ONE registration: the only size the comm ever sees
    A = r._arena
    assert (
        A.bytes == 408 * MiB == r._arena_bytes and r.policy == "arena"
    )  # 336 MiB shared S/R/C + page, 72 MiB of small windows (9 owners x 2 x 4 MiB: 6 r19/r18-family + 3 r20 MP64)
    # views: inside the arena, 4 MiB aligned, pairwise disjoint (shared set once), local = arena.local + offset
    seen = {}
    for w in r._views:
        assert (
            w.arena_offset % (4 * MiB) == 0
            and w.bytes % (4 * MiB) == 0
            and w.local == A.local + w.arena_offset
        )
        assert 0 <= w.arena_offset and w.arena_offset + w.bytes <= A.bytes
        seen[w.arena_offset] = w.bytes
    spans = sorted(seen.items())
    assert all(a + n <= b for (a, n), (b, _m) in pairwise(spans))  # disjoint
    assert (
        sum(seen.values()) == 408 * MiB
    )  # 12 small + shared S/R/C + the global page fill the arena exactly: the budget is exact
    # every large owner got the SAME three views; the view struct reproduces arena + offset for local and peer addressing (stride/lsaRank unchanged)
    large = [o for o, _ in plans if o.startswith("sp_qkv.large/")]
    S0 = r.windows[large[0]][large[0] + "/S"]
    assert all(
        r.windows[o][o + "/S"] is S0 and r.windows[o][o + "/S"].shared for o in large
    )
    base, stride, lsa = be.win_info(r.comm, S0.handle)
    assert (
        base == A.local + S0.arena_offset
        and (stride, lsa) == be.win_info(r.comm, A.handle)[1:]
    )
    # the arena is zeroed once, in finalize(); the launch guard sees it pending before
    assert r.pending_zero == 1
    r.finalize()
    assert r.pending_zero == 0
    # release: views freed (never deregistered), the arena deregistered exactly once
    n_views = len(r._views)
    be.log.clear()
    r.release()
    assert n_views == 22 and len(be.freed) == 23 == len(
        set(be.freed)
    )  # 22 plan views (18 small + shared S/R/C + the global page) + the 1 view-self-test view, each freed exactly once
    assert (
        len([x for x in be.log if x[0] == "dereg"]) == 1 and r._views == []
    )  # the arena is the ONLY deregistered window


def test_native_view_struct_reads_only_what_the_kernels_read():
    """(1) of the lead's question, as a CPU audit of the sources: the view is a plain device copy of ccoWindowDevice with winBase shifted; the vendored kernels
    reach a window handle only through ccoGetLocalPtr / ccoGetLsaPeerPtr (winBase, stride4G, lsaRank); the small kernel resolves peer pointers on the host.
    """
    import re

    src = _host_src("gf_comm.cpp")
    body = src[
        src.index("int32_t gf_win_view_v1") : src.index("int32_t gf_win_view_free_v1")
    ]
    assert (
        "hipMemcpyDeviceToHost" in body
        and "h.winBase += offset" in body
        and "hipMalloc" in body
        and "hipMemcpyHostToDevice" in body
    )
    assert "ccoWindowDeregister" not in body  # a view is never registered with MORI
    dev = ""
    for root, _d, fns in os.walk(os.path.join(jit.CSRC, "large", "r11", "device")):
        dev += "".join(
            _slurp(os.path.join(root, f), encoding="utf-8")
            for f in fns
            if f.endswith((".hpp", ".cu", ".h"))
        )
    dev += _slurp(
        os.path.join(jit.CSRC, "small", "r18", "device", "sk_gemm_fp8.cu"),
        encoding="utf-8",
    )
    assert not re.search(
        r"->\s*(winBase|stride4G|lsaRank|ibgdaWin)|\.ibgdaWin|findWindow|windowTable|resourceWindow",
        dev,
    )  # no direct field access, no table lookups
    users = set(
        re.findall(
            r"(ccoGetLocalPtr|ccoGetLsaPeerPtr)\(\s*(?:a|args)\.(Swin|Rwin|Cwin)", dev
        )
    )
    assert {"Swin", "Rwin", "Cwin"} <= {
        w for _f, w in users
    }  # the handles are used ONLY as the first argument of those two accessors
    # no kernel source differs from the vendored accepted file (no kernel change): the vendored sources are unchanged since the vendoring commit


def test_release_frees_each_window_local_allocation_after_deregister_before_comm_destroy(
    monkeypatch,
):

    for var in ("AITER_GEMM_FUSION_PROFILE", "AITER_GEMM_FUSION_WINDOW_POLICY"):
        monkeypatch.delenv(var, raising=False)

    def build():
        ev = []

        class B(RealisticFirstFit):
            def win_deregister(self, comm, w):
                ev.append(("dereg", w))

            def win_free(self, comm, loc):
                ev.append(("free", loc))

            def comm_destroy(self, comm):
                ev.append(("comm_destroy",))

        r = make_res(
            dist=FakeDist(),
            backend=B(),
            policy="uniform",
            window_bytes=4 << 20,
            max_windows=4,
        )
        r.ensure(
            "o", [WindowReq("a", 100, "p2p_window"), WindowReq("b", 100, "p2p_window")]
        )
        r.finalize()
        ev.clear()
        r.release()
        return r, ev

    _r, ev = build()
    kinds = [e[0] for e in ev]
    assert (
        kinds[-1] == "comm_destroy" and kinds.count("dereg") == kinds.count("free") >= 2
    )
    for i, e in enumerate(ev):
        if e[0] == "dereg":
            assert (
                ev[i + 1][0] == "free"
            )  # Deregister then MemFree(local), per window, always (no switch)


def test_wait_budget_resolution_precedence_default_validation_and_profile_value(
    monkeypatch,
):
    from aiter.gemm_fusion.resources import (
        DEFAULT_WAIT_BUDGET_US,
        ResourceError,
        resolve_wait_budget_us,
    )

    monkeypatch.delenv("AITER_GEMM_FUSION_WAIT_BUDGET_US", raising=False)
    monkeypatch.delenv("AITER_GEMM_FUSION_PROFILE", raising=False)
    prof = profmod.load_profile(PROFILE)
    assert (
        prof.meta["wait_budget_us"] == 5_000_000 == DEFAULT_WAIT_BUDGET_US
    )  # the regenerated shipped profile carries it
    assert resolve_wait_budget_us(prof) == (5_000_000, "profile")
    assert resolve_wait_budget_us(None) == (
        5_000_000,
        "default",
    )  # no profile: the 5 s default (not the kernel's 50 ms)
    bare = profmod.parse_profile({"profile": "nobudget", "rows": []})
    assert resolve_wait_budget_us(bare) == (
        5_000_000,
        "default",
    )  # a profile without the field: default
    custom = profmod.parse_profile(
        {"profile": "c", "wait_budget_us": 2_000_000, "rows": []}
    )
    assert resolve_wait_budget_us(custom) == (2_000_000, "profile")
    monkeypatch.setenv(
        "AITER_GEMM_FUSION_WAIT_BUDGET_US", "123456"
    )  # env overrides (tuning only)
    assert resolve_wait_budget_us(custom) == (123456, "env") and resolve_wait_budget_us(
        None
    ) == (123456, "env")
    for bad in ("0", "-5", "abc", "1.5"):
        monkeypatch.setenv("AITER_GEMM_FUSION_WAIT_BUDGET_US", bad)
        with pytest.raises(ResourceError, match="AITER_GEMM_FUSION_WAIT_BUDGET_US"):
            resolve_wait_budget_us(None)
    monkeypatch.delenv("AITER_GEMM_FUSION_WAIT_BUDGET_US")
    for badv in (0, -1, 1.5, "5", True):
        with pytest.raises(ResourceError, match="wait_budget_us"):
            resolve_wait_budget_us(
                profmod.parse_profile(
                    {"profile": "b", "wait_budget_us": badv, "rows": []}
                )
            )
    # FusionResources resolves it ONCE from its explicit profile and logs it with the comm line
    msgs, undo = _log_capture()
    try:
        r = make_res(
            dist=FakeDist(),
            backend=RealisticFirstFit(),
            profile=PROFILE,
            window_bytes=4 << 20,
            max_windows=2,
        )
    finally:
        undo()
    assert (r.wait_budget_us, r.wait_budget_source) == (5_000_000, "profile")
    assert any(
        "event=resources comm" in m and "wait_budget_us=5000000 (profile)" in m
        for m in msgs
    )


def test_large_context_passes_the_budget_to_every_launch_and_defaults_to_the_kernel_when_unset():
    from types import SimpleNamespace as NS

    import aiter.ops.gemm_fusion.sp_qkv.large_rt as lrt

    seen = []
    lib = NS(
        lg_launch_v1=lambda h, A, W, As, Ws, sA, poll, budget, n, hf, stream: (
            seen.append(budget),
            0,
        )[1]
    )
    ctx = lrt.LargeContext(lib, 1, 2048, 2304, 0)
    assert ctx.budget_us == 0  # unset: the kernel's own default (50 ms)
    ctx.launch(1, 2, 3, 4, 5, 2, 0, None)
    ctx.budget_us = 5_000_000
    ctx.launch(1, 2, 3, 4, 5, 2, 0, None)
    ctx.launch(
        1, 2, 3, 4, 5, 2, 0, None, budget_us=777
    )  # an explicit per-call value still wins
    assert seen == [0, 5_000_000, 777]


def test_large_plan_context_gets_the_resources_budget_and_logs_it(monkeypatch):
    """variants_large._context copies FusionResources.wait_budget_us onto the LargeContext (so every launch, incl. the one captured in a graph, carries it)."""
    import inspect

    import aiter.ops.gemm_fusion.sp_qkv.variants_large as vl

    src = inspect.getsource(vl.SpQkvLarge._context)
    assert (
        _sq('got.budget_us = int(getattr(res, "wait_budget_us", 0) or 0)') in _sq(src)
        and "event=binding large_wait_budget" in src
        and "baked at graph capture" in src
    )
    run_src = inspect.getsource(vl.SpQkvLarge.run)
    assert (
        "c.launch(" in run_src and "budget_us=" not in run_src
    )  # the budget lives on the context: the launch call itself is unchanged


class _Rep:
    def __init__(self, poison=0, report=None):
        self.poison, self.rep = poison, report or {}

    def read(self):
        return self.poison

    def report(self):
        return self.rep


def test_poison_reason_is_logged_before_quiesce_clears_it_large_and_small(monkeypatch):
    from aiter.ops.gemm_fusion.sp_qkv.variants import decode_small_poison
    from aiter.ops.gemm_fusion.sp_qkv.variants_large import decode_large_poison

    for var in ("AITER_GEMM_FUSION_PROFILE", "AITER_GEMM_FUSION_WAIT_BUDGET_US"):
        monkeypatch.delenv(var, raising=False)
    be = RealisticFirstFit()
    ev = []
    be.device_sync = lambda dev: ev.append("sync")
    be.zero = lambda dev, loc, n: ev.append("zero")
    r = make_res(
        dist=FakeDist(),
        backend=be,
        window_bytes=4 << 20,
        max_windows=2,
        policy="uniform",
        profile=None,
    )
    r.ensure("L", [WindowReq("a", 100, "p2p_window")])
    r.ensure("S", [WindowReq("a", 100, "p2p_window")])
    large = _Rep(
        2,
        {"epoch": 7, "poison": 2, "poison_reason": 2, "credit_wait_ticks_100MHz": 123},
    )
    small = _Rep(0x105)

    def large_read():
        ev.append("read_large")
        return large.read()

    r.set_poison("L", large_read, large.report, decode_large_poison)
    r.set_poison(
        "S",
        lambda: (ev.append("read_small"), small.read())[1],
        None,
        decode_small_poison,
    )
    r.finalize()
    ev.clear()
    msgs, undo = _log_capture(logging.WARNING)
    try:
        r.quiesce_and_reset()
    finally:
        undo()
    pm = [m for m in msgs if "event=resources poison" in m]
    assert len(pm) == 2 and all(
        "where=quiesce_and_reset" in m and "wait_budget_us=5000000 (default)" in m
        for m in pm
    )
    L = next(m for m in pm if "owner=L" in m)
    assert (
        "W2_TIMEOUT(arrival wait" in L
        and "word=2" in L
        and "poison_reason=2" in L
        and "epoch=7" in L
        and "credit_wait_ticks_100MHz=123" in L
    )
    S = next(m for m in pm if "owner=S" in m)
    assert "OVERSHOOT" in S and "why=0x5" in S and "word=0x105" in S
    assert ev.index("zero") > max(
        i for i, e in enumerate(ev) if e in ("read_large", "read_small")
    )  # logged BEFORE any window is zeroed/cleared
    # not poisoned: nothing logged; the report is not even read
    large.poison, small.poison = 0, 0
    msgs, undo = _log_capture(logging.WARNING)
    try:
        r.quiesce_and_reset()
        assert r.log_poison("x") == 0
    finally:
        undo()
    assert not [m for m in msgs if "event=resources poison" in m]
    # poison_local() semantics are unchanged (OR of the words)
    large.poison = 1
    assert r.poison_local() == 1


def test_decoders_name_the_failing_wait():
    from aiter.ops.gemm_fusion.sp_qkv.variants import decode_small_poison
    from aiter.ops.gemm_fusion.sp_qkv.variants_large import decode_large_poison

    assert "W1_TIMEOUT(credit wait" in decode_large_poison(
        1
    ) and "W2_TIMEOUT(arrival wait" in decode_large_poison(2)
    assert (
        "POSTER_WAIT" in decode_large_poison(6)
        and "EPOCH_MISMATCH" in decode_large_poison(3)
        and "UNKNOWN" in decode_large_poison(99)
    )
    assert "OVERSHOOT" in decode_small_poison(
        0x142
    ) and "why=0x42" in decode_small_poison(0x142)
    assert "TIMEOUT-or-other" in decode_small_poison(
        0x40
    )  # the kernel's T != 4*M contract poison is reported as the raw code, not mislabelled


def test_wait_budget_documents_the_capture_bake_in_and_costs():
    from aiter.gemm_fusion import resources as rmod

    assert (
        "BAKED into a captured graph" in rmod.resolve_wait_budget_us.__doc__
        and "before capture" in rmod.resolve_wait_budget_us.__doc__
    )
    g = _slurp(
        os.path.join(os.path.dirname(specs.__file__), "gen_profile.py"),
        encoding="utf-8",
    )
    assert "BAKED into a captured graph" in g and "failure-detection latency" in g


# ======================================================================= arena review conditions A-1 (ABI pin + view read-back + fallback), A-3 (default), A-5 (budget doc), A-7
def _make_pinned_mori(tmp, lib_bytes=b"LIB", hdr_bytes=b"HDR"):
    import hashlib
    import tempfile
    import uuid
    from pathlib import Path

    root = Path(tempfile.gettempdir()) / ("mori_" + uuid.uuid4().hex)
    (root / "include" / "mori" / "cco").mkdir(parents=True)
    (root / "lib").mkdir()
    (root / "include" / "mori" / "cco" / "cco.hpp").write_bytes(hdr_bytes)
    (root / "lib" / "libmori_cco.so").write_bytes(lib_bytes)
    return (
        root,
        hashlib.sha256(lib_bytes).hexdigest(),
        hashlib.sha256(hdr_bytes).hexdigest(),
    )


def test_A1_abi_pin_check_compares_library_and_header_hashes(monkeypatch):
    import shutil

    from aiter.gemm_fusion import capabilities as caps

    assert (
        caps.PINNED_CCO_SO_SHA256.startswith("87b7322f")
        and len(caps.PINNED_CCO_SO_SHA256) == 64
    )
    assert (
        caps.PINNED_CCO_HEADER_SHA256.startswith("cce155961f59")
        and len(caps.PINNED_CCO_HEADER_SHA256) == 64
    )
    assert caps.PINNED_WINDOW_DEVICE_ABI == (32, 0, 8, 12, 16)
    root, sl, sh = _make_pinned_mori(None)
    try:
        env = {
            "MORI_ROOT": str(root),
            "MORI_CCO_SO": str(root / "lib" / "libmori_cco.so"),
        }
        ok, why = caps.check_mori_abi_pin(env)
        assert (
            not ok and "libmori_cco.so sha256" in why and "pinned 87b7322f" in why
        )  # the fake lib is not the pinned one
        monkeypatch.setattr(caps, "PINNED_CCO_SO_SHA256", sl)
        ok, why = caps.check_mori_abi_pin(env)
        assert not ok and "cco.hpp sha256" in why  # lib matches, header does not
        monkeypatch.setattr(caps, "PINNED_CCO_HEADER_SHA256", sh)
        assert caps.check_mori_abi_pin(env) == (True, "ok")
        (root / "include" / "mori" / "cco" / "cco.hpp").write_bytes(b"CHANGED")
        assert not caps.check_mori_abi_pin(env)[
            0
        ]  # a header edit is noticed (the hash cache keys on mtime/size)
        env2 = dict(env, AITER_GEMM_FUSION_MORI_ABI_PIN="off")
        assert caps.check_mori_abi_pin(env2) == (
            True,
            "pin check disabled by AITER_GEMM_FUSION_MORI_ABI_PIN",
        )
        assert not caps.check_mori_abi_pin(
            {"MORI_ROOT": str(root / "missing"), "MORI_CCO_SO": str(root / "nolib.so")}
        )[
            0
        ]  # never raises
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_A1_native_layout_static_asserts_and_runtime_abi_echo():
    src = _host_src("gf_comm.cpp")
    for frag in (
        "sizeof(mori::cco::ccoWindowDevice) == 32",
        "offsetof(mori::cco::ccoWindowDevice, winBase) == 0",
        "offsetof(mori::cco::ccoWindowDevice, stride4G) == 8",
        "offsetof(mori::cco::ccoWindowDevice, lsaRank) == 12",
        "offsetof(mori::cco::ccoWindowDevice, ibgdaWin) == 16",
        "#include <cstddef>",
    ):
        assert frag in src, frag
    assert (
        src.count("static_assert(") >= 6
        and "int32_t gf_abi_info_v1(uint64_t* out5)" in src
    )
    # the static_asserts sit BEFORE the view code that depends on them
    assert src.index(
        "static_assert(sizeof(mori::cco::ccoWindowDevice) == 32"
    ) < src.index("int32_t gf_win_view_v1")
    from aiter.gemm_fusion import capabilities as caps
    from aiter.gemm_fusion.resources import CcoBackend

    be = object.__new__(CcoBackend)

    class Lib:
        def __init__(self, vals):
            self.vals = vals

        def gf_abi_info_v1(self, b):
            for i, v in enumerate(self.vals):
                b[i] = v
            return 0

    import aiter.gemm_fusion.capabilities as c2

    real = c2.check_mori_abi_pin
    try:
        c2.check_mori_abi_pin = lambda env=None: (True, "ok")
        be.lib = Lib(caps.PINNED_WINDOW_DEVICE_ABI)
        assert be.abi_check() == (True, "ok")
        be.lib = Lib((40, 0, 8, 12, 24))  # a different compiled layout is refused
        ok, why = be.abi_check()
        assert not ok and "compiled ccoWindowDevice layout" in why
        c2.check_mori_abi_pin = lambda env=None: (False, "lib hash differs")
        assert be.abi_check() == (False, "lib hash differs")
    finally:
        c2.check_mori_abi_pin = real


def test_A3_arena_is_the_default_and_only_uniform_stays_selectable(monkeypatch):
    from aiter.gemm_fusion.resources import (
        DEFAULT_POLICY,
        POLICIES,
        ResourceError,
    )

    assert POLICIES == (
        "arena",
        "uniform",
    )  # per_owner / aligned / shared_large / abs_aligned / arena_per_owner were removed

    for var in (
        "AITER_GEMM_FUSION_WINDOW_POLICY",
        "AITER_GEMM_FUSION_PROFILE",
        "AITER_GEMM_FUSION_ARENA_MIB",
    ):
        monkeypatch.delenv(var, raising=False)
    clear_profile_cache()
    assert DEFAULT_POLICY == "arena"
    r = make_res(dist=FakeDist(), backend=ArenaBackend(), profile=PROFILE)
    assert (
        r.policy == "arena"
        and r.policy_source == "default"
        and r.policy_fallback_reason is None
        and r._vmm == 4 << 30
    )
    # selectable: the env var and the argument
    monkeypatch.setenv("AITER_GEMM_FUSION_WINDOW_POLICY", "uniform")
    ru = make_res(dist=FakeDist(), backend=RealisticFirstFit(), profile=PROFILE)
    assert (
        ru.policy == "uniform"
        and ru.policy_source == "explicit"
        and ru._vmm == 8 << 30
        and ru.policy_fallback_reason is None
    )  # 6.4 GiB: documented cost
    monkeypatch.delenv("AITER_GEMM_FUSION_WINDOW_POLICY")
    for gone in (
        "per_owner",
        "aligned",
        "shared_large",
        "abs_aligned",
        "arena_per_owner",
    ):
        with pytest.raises(ResourceError, match="must be one of"):
            make_res(
                dist=FakeDist(),
                backend=RealisticFirstFit(),
                profile=PROFILE,
                policy=gone,
            )
    # default + no budget: refused with a logged reason, falls back to uniform (the previous no-profile behaviour)
    msgs, undo = _log_capture(logging.WARNING)
    try:
        rn = make_res(
            dist=FakeDist(),
            backend=RealisticFirstFit(),
            window_bytes=4 << 20,
            max_windows=2,
        )
    finally:
        undo()
    assert rn.policy == "uniform" and "no arena size" in rn.policy_fallback_reason
    assert any(
        "event=resources arena_refused" in m and "reason=no arena size" in m
        for m in msgs
    )
    # an EXPLICIT arena without a budget still fails loudly at ensure() (configuration error, not a silent fallback)
    ra = make_res(
        dist=FakeDist(),
        backend=ArenaBackend(),
        window_bytes=4 << 20,
        max_windows=2,
        policy="arena",
    )
    assert ra.policy == "arena"
    with pytest.raises(ResourceError, match="needs the arena size"):
        ra.ensure("o", [WindowReq("a", 100, "p2p_window")])


def test_A1_abi_pin_mismatch_refuses_arena_at_construction_and_falls_back_to_uniform(
    monkeypatch,
):

    for var in ("AITER_GEMM_FUSION_WINDOW_POLICY", "AITER_GEMM_FUSION_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    clear_profile_cache()

    class Pinned(ArenaBackend):
        def __init__(self, ok, why):
            super().__init__()
            self.res = (ok, why)

        def abi_check(self):
            return self.res

    be = Pinned(True, "ok")
    assert make_res(dist=FakeDist(), backend=be, profile=PROFILE).policy == "arena"
    msgs, undo = _log_capture(logging.WARNING)
    try:
        bad = make_res(
            dist=FakeDist(),
            backend=Pinned(
                False, "libmori_cco.so sha256 deadbeef... != pinned 87b7322f..."
            ),
            profile=PROFILE,
        )
        explicit = make_res(
            dist=FakeDist(),
            backend=Pinned(False, "hdr differs"),
            profile=PROFILE,
            policy="arena",
        )  # a safety property: refused even if asked for
    finally:
        undo()
    for r in (bad, explicit):
        assert (
            r.policy == "uniform" and r._vmm == 8 << 30 and r.window_bytes == 171966464
        )  # uniform heap/budget from the profile
    assert (
        "MORI ABI pin mismatch" in bad.policy_fallback_reason
        and "deadbeef" in bad.policy_fallback_reason
    )
    assert sum("event=resources arena_refused" in m for m in msgs) == 2
    assert (
        bad.run_bindings()["window_policy"] == "uniform"
        and "MORI ABI pin mismatch"
        in bad.run_bindings()["window_policy_fallback_reason"]
    )


class _BadViewBackend(ArenaBackend):
    """win_info of a view disagrees with the arena (a MORI build whose struct layout differs): the view property does not hold."""

    def __init__(self, mode):
        super().__init__()
        self.mode, self.creates = mode, 0

    def comm_create(self, rank, world, uid, heap, device):
        self.creates += 1
        return (
            super().comm_create(rank, world, uid, heap, device)
            if hasattr(super(), "comm_create")
            else object()
        )

    def win_info(self, comm, w):
        base, stride, lsa = super().win_info(comm, w)
        if w in self.views:
            return {
                "base": (base + 4096, stride, lsa),
                "stride": (base, stride + 1, lsa),
                "lsa": (base, stride, lsa + 1),
            }[self.mode]
        return base, stride, lsa


def test_A1_view_readback_mismatch_refuses_arena_and_rebuilds_the_comm_as_uniform(
    monkeypatch,
):

    for var in ("AITER_GEMM_FUSION_WINDOW_POLICY", "AITER_GEMM_FUSION_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    clear_profile_cache()
    plans = _all_plans(monkeypatch)
    for mode in ("base", "stride", "lsa"):
        be = _BadViewBackend(mode)
        msgs, undo = _log_capture(logging.WARNING)
        try:
            r = make_res(dist=FakeDist(), backend=be, profile=PROFILE)
            assert r.policy == "arena"
            for owner, rq in plans:
                r.ensure(
                    owner, rq
                )  # the first ensure() runs the self-test, refuses, rebuilds the comm for uniform and re-dispatches
        finally:
            undo()
        assert (
            r.policy == "uniform"
            and "view read-back mismatch" in r.policy_fallback_reason
            and r.window_bytes == 171966464
        )
        assert (
            be.creates == 2
        )  # the comm was destroyed and recreated: NO second window size on the arena's comm
        assert (
            r._arena is None and r._views == [] and len(r._regs) == 46
        )  # uniform: 40 equal windows (15 plans -> 39 windows + the global page)
        assert all(w.bytes == 171966464 for w in r._regs)
        assert any(
            "event=resources arena_refused" in m and "view read-back mismatch" in m
            for m in msgs
        )
        assert r.run_bindings()["window_policy"] == "uniform"
    # (a) a rank that disagrees on the REGISTRATION itself is a fatal abort (the pre-existing collective agreement check), not a fallback
    from aiter.gemm_fusion.resources import ResourceError

    r2 = make_res(
        dist=FakeDist(diverge_rank=3), backend=ArenaBackend(), profile=PROFILE
    )
    with pytest.raises(ResourceError, match="registration disagreement"):
        r2.ensure(*plans[0])

    # (b) ONE rank failing the view self-test makes EVERY rank fall back (peers address the arena at identical offsets): the self-test verdicts are all-gathered
    class SplitDist(FakeDist):
        def all_gather_object(self, out, obj, group=None):
            self.calls.append(("allgather",))
            for i in range(len(out)):
                out[i] = obj
            if (
                isinstance(obj, tuple) and len(obj) == 2 and obj[0] is True
            ):  # the view self-test verdict: rank 2 reports a failure
                out[2] = (False, "rank2 view mismatch")

    be3 = ArenaBackend()  # this rank's own read-back is fine
    r3 = make_res(dist=SplitDist(), backend=be3, profile=PROFILE)
    for owner, rq in plans:
        r3.ensure(owner, rq)
    assert (
        r3.policy == "uniform"
        and "rank2 view mismatch" in r3.policy_fallback_reason
        and r3._arena is None
    )


def test_A1_each_later_view_is_read_back_and_a_mismatch_aborts(monkeypatch):
    from aiter.gemm_fusion.resources import ResourceError

    monkeypatch.delenv("AITER_GEMM_FUSION_WINDOW_POLICY", raising=False)
    clear_profile_cache()
    be = ArenaBackend()
    r = make_res(dist=FakeDist(), backend=be, profile=PROFILE)
    plans = _all_plans(monkeypatch)
    r.ensure(*plans[0])  # self-test passes, real views pass their read-back
    orig = be.win_info

    def shifted(comm, w):  # from now on views read back shifted
        t = orig(comm, w)
        return (t[0] + 8, t[1], t[2]) if w in be.views else t

    be.win_info = shifted
    with pytest.raises(
        ResourceError, match="view read-back mismatch.*fusion must be aborted"
    ):
        r.ensure(*plans[1])


def test_A5_wait_budget_rationale_and_pg_timeout_guidance_are_documented():
    from aiter.gemm_fusion import resources as rmod

    doc = rmod.resolve_wait_budget_us.__doc__
    assert (
        "213 ms" in doc
        and "PG timeout" in doc
        and "5 s" in doc
        and "detection latency" in doc
        and "run_bindings" in doc
    )
    g = _slurp(
        os.path.join(os.path.dirname(specs.__file__), "gen_profile.py"),
        encoding="utf-8",
    )
    assert "213 ms" in g and "PG timeout" in g


def test_A5_run_bindings_record_policy_budget_source_and_overrides(monkeypatch):

    for var in ("AITER_GEMM_FUSION_WINDOW_POLICY", "AITER_GEMM_FUSION_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AITER_GEMM_FUSION_WAIT_BUDGET_US", "9000000")
    clear_profile_cache()
    b = make_res(
        dist=FakeDist(), backend=ArenaBackend(), profile=PROFILE
    ).run_bindings()
    assert (
        b["wait_budget_us"] == 9000000 and b["wait_budget_source"] == "env"
    )  # an override is visible in the bindings
    assert (
        b["window_policy"] == "arena"
        and b["window_policy_source"] == "default"
        and b["zero_mode"] == "h2d_deferred"
    )


# ======================================================================= cleanup: replacements for the coverage of removed features + guards
def test_runtime_never_imports_test_utils_and_the_removed_names_are_gone():
    import re

    root = os.path.join(os.path.dirname(profmod.__file__), "..")
    runtime = []
    for base in (
        os.path.join(root, "gemm_fusion"),
        os.path.join(root, "ops", "gemm_fusion"),
    ):
        for dp, _dn, fns in os.walk(base):
            runtime += [
                os.path.join(dp, f)
                for f in fns
                if f.endswith(".py") and f != "test_utils.py"
            ]
    removed_env = (
        "REG_BARRIER",
        "REG_SYNC",
        "LATE_REG",
        "ZERO_MODE",
        "FREE_BEFORE_DESTROY",
        "PROBE_FREE",
        "ALIGN_CAP_MIB",
        "SPACER_GRAN_MIB",
        "ABS_ALIGN",
        "WINDOW_MIB",
        "DEBUG_ALIAS",
    )
    for f in runtime:
        code = _slurp(f, encoding="utf-8")
        assert "test_utils" not in re.sub(r"#[^\n]*", "", code).replace(
            '"""', ""
        ), f"{os.path.basename(f)} imports/mentions test_utils"
        for name in removed_env:
            assert (
                f"AITER_GEMM_FUSION_{name}" not in code
            ), f"{os.path.basename(f)} still reads AITER_GEMM_FUSION_{name}"
    assert not os.path.exists(
        os.path.join(root, "ops", "gemm_fusion", "sp_qkv", "window_probe.py")
    )
    import aiter.gemm_fusion as g
    from aiter.gemm_fusion import resources as rmod

    for n in ("clear_profile_cache", "list_variants"):
        assert not hasattr(g, n), n
    for n in (
        "assert_no_window_alias",
        "find_window_alias",
        "window_ranges",
        "local_m",
        "peers",
        "log_poison",
        "run_bindings",
        "poison_local",
    ):
        assert hasattr(rmod.FusionResources, n) == (
            n in ("log_poison", "run_bindings", "poison_local")
        ), n
    for n in (
        "tensor_extent",
        "iter_tensors",
        "WindowAliasError",
        "AbsLayout",
        "aligned_plan",
        "abs_layout_simulate",
        "policy_memory_table",
        "start_alignment",
    ):
        assert not hasattr(rmod, n), n
    for n in ("va_info", "set_zero_mode"):
        assert not hasattr(rmod.CcoBackend, n), n
    assert hasattr(
        rmod.CcoBackend, "peer_ptr"
    )  # restored for R12: the peer-ERR table of the global page (ccoGetPeerPtr)
    import inspect

    params = inspect.signature(rmod.FusionResources.__init__).parameters
    assert list(params) == [
        "self",
        "rank",
        "world",
        "group",
        "dist",
        "backend",
        "device",
        "caps",
        "policy",
        "profile",
    ]  # the ATOM-facing constructor, nothing else


def test_native_comm_has_no_memset_peer_ptr_or_va_info_entry_points():
    gf = _host_src("gf_comm.cpp")
    assert (
        "hipMemset(" not in gf
        and "gf_comm_va_info_v1" not in gf
        and "int32_t gf_peer_ptr_v1(" in gf  # restored for R12 (peer-ERR table)
    )
    assert (
        "int32_t gf_zero_v1(int32_t device, void* local, uint64_t bytes)" in gf
        and "zero_h2d(" in gf
    )  # MORI's H2D zeroing is the only method
    for kept in (
        "gf_win_info_v1",
        "gf_abi_info_v1",
        "gf_devcomm_info_v1",
        "gf_mem_info_v1",
        "gf_win_view_v1",
        "gf_win_free_v1",
        "gf_peer_ptr_v1",
    ):
        assert f"int32_t {kept}(" in gf, kept


def test_late_ensure_of_a_new_owner_is_refused_for_both_policies(monkeypatch):
    from aiter.gemm_fusion.resources import ResourceError

    monkeypatch.delenv("AITER_GEMM_FUSION_PROFILE", raising=False)
    clear_profile_cache()
    plans = _all_plans(monkeypatch)
    ra = make_res(dist=FakeDist(), backend=ArenaBackend(), profile=PROFILE)
    ra.ensure(*plans[0])
    ra.finalize()
    with pytest.raises(ResourceError, match="after finalize"):
        ra.ensure(*plans[1])
    assert (
        ra.ensure(*plans[0]) is ra.windows[plans[0][0]]
    )  # an owner that was resolved before finalize is still idempotent
    ru = make_res(
        dist=FakeDist(),
        backend=RealisticFirstFit(),
        policy="uniform",
        window_bytes=4 << 20,
        max_windows=4,
    )
    ru.ensure("a", [WindowReq("x", 100, "p2p_window")])
    ru.finalize()
    with pytest.raises(ResourceError, match="after finalize"):
        ru.ensure("b", [WindowReq("x", 100, "p2p_window")])


def test_arena_overflow_names_the_knob_and_a_diverging_rank_aborts_every_rank(
    monkeypatch,
):
    from aiter.gemm_fusion.resources import ResourceError

    for var in ("AITER_GEMM_FUSION_PROFILE", "AITER_GEMM_FUSION_WINDOW_POLICY"):
        monkeypatch.delenv(var, raising=False)
    clear_profile_cache()
    plans = _all_plans(monkeypatch)
    monkeypatch.setenv("AITER_GEMM_FUSION_ARENA_MIB", "8")
    r2 = make_res(dist=FakeDist(), backend=ArenaBackend(), profile=PROFILE)
    with pytest.raises(
        ResourceError, match="arena overflow.*AITER_GEMM_FUSION_ARENA_MIB"
    ):
        for owner, rq in plans:
            r2.ensure(owner, rq)
    monkeypatch.delenv("AITER_GEMM_FUSION_ARENA_MIB")
    r3 = make_res(
        dist=FakeDist(diverge_rank=1), backend=ArenaBackend(), profile=PROFILE
    )  # the registration agreement fails first
    with pytest.raises(ResourceError, match="registration disagreement"):
        r3.ensure(*plans[0])


def test_init_images_written_before_finalize_survive_the_deferred_zeroing(monkeypatch):
    for var in ("AITER_GEMM_FUSION_WINDOW_POLICY", "AITER_GEMM_FUSION_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    be = _ZeroRecBackend()
    r = make_res(dist=FakeDist(), backend=be, window_bytes=4 << 20, max_windows=6)
    ev = []
    r.ensure("o0", [WindowReq("a", 100, "p2p_window")])
    r.set_init(
        "o0", lambda: ev.append("init")
    )  # a control image written at attach time (before finalize)
    assert ev == ["init"]
    r.finalize()
    assert ev == [
        "init",
        "init",
    ]  # finalize() zeroed the window AFTER the first init, so the image is rewritten: not lost
    r.quiesce_and_reset()
    assert ev == ["init", "init", "init"]


def test_policy_memory_log_lists_both_policies_and_the_arena_is_the_408_mib_budget(
    monkeypatch,
):
    for var in (
        "AITER_GEMM_FUSION_PROFILE",
        "AITER_GEMM_FUSION_WINDOW_POLICY",
        "AITER_GEMM_FUSION_ARENA_MIB",
    ):
        monkeypatch.delenv(var, raising=False)
    clear_profile_cache()
    msgs, undo = _log_capture()
    try:
        r = make_res(dist=FakeDist(), backend=ArenaBackend(), profile=PROFILE)
    finally:
        undo()
    pm = [m for m in msgs if m.startswith("event=resources policy_memory")]
    assert (
        pm
        and "selected=arena" in pm[0]
        and "uniform=7544MiB" in pm[0]
        and "arena=408MiB" in pm[0]
    )
    assert r._arena_size_from_budget() == 408 * MiB and r._vmm == 4 << 30


# ---- test_utils: the aliasing assertion (C-S4) as free functions over a FusionResources
class _Tn:
    def __init__(self, ptr, shape, stride, es=2):
        self.p, self.shape, self._s, self.es = ptr, shape, stride, es

    def data_ptr(self):
        return self.p

    def stride(self):
        return self._s

    def element_size(self):
        return self.es


def test_test_utils_alias_checks_over_the_arena_strided_extents_nested_and_env(
    monkeypatch,
):
    from aiter.ops.gemm_fusion.sp_qkv import test_utils as tu

    for var in (
        "AITER_GEMM_FUSION_PROFILE",
        "AITER_GEMM_FUSION_WINDOW_POLICY",
        "AITER_GEMM_FUSION_DEBUG_ALIAS",
    ):
        monkeypatch.delenv(var, raising=False)
    clear_profile_cache()
    r = make_res(dist=FakeDist(), backend=ArenaBackend(), profile=PROFILE)
    r.ensure(*_all_plans(monkeypatch)[0])
    A = r._arena
    assert [lab for lab, _a, _b in tu.window_ranges(r)] == [
        "<arena>/arena [shared]"
    ]  # the one registered window covers every view
    base = A.local
    inside, past_end = _Tn(base + 1024, (16, 2304), (2304, 1)), _Tn(
        A.local + A.bytes - 100, (4, 2560), (2304, 1)
    )
    outside, adjacent = _Tn(base + A.bytes + (64 << 20), (16, 16), (16, 1)), _Tn(
        base + A.bytes, (4, 4), (4, 1)
    )
    assert (
        tu.find_window_alias(r, outside) == []
        and tu.find_window_alias(r, adjacent) == []
    )  # one past the end is not an overlap
    assert tu.find_window_alias(r, inside) and tu.find_window_alias(r, past_end)
    ext = tu.tensor_extent(
        past_end
    )  # the reuse view [T,2560] with row stride 2304 extends further than numel*elem: strides are honoured
    assert ext == (
        A.local + A.bytes - 100,
        A.local + A.bytes - 100 + ((3 * 2304 + 2559) + 1) * 2,
    )
    hits = tu.find_window_alias(
        r, {"out": [outside, (inside,)], "meta": {"x": outside}}
    )
    assert len(hits) == 1 and "['out'][1][0]" in hits[0]
    assert tu.tensor_extent(_Tn(0x1000, (0, 4), (4, 1))) is None
    tu.assert_no_window_alias(r, inside)  # default: no-op
    with pytest.raises(
        tu.WindowAliasError, match="points into a CCO window.*copy it within the op"
    ):
        tu.assert_no_window_alias(r, inside, force=True)
    tu.assert_no_window_alias(r, outside, force=True)
    monkeypatch.setenv("AITER_GEMM_FUSION_DEBUG_ALIAS", "1")
    with pytest.raises(tu.WindowAliasError):
        tu.assert_no_window_alias(
            r, {"impl._index_q": inside}, what="attn.impl._index_q"
        )


def test_test_utils_make_res_make_profile_and_clear_profile_cache(monkeypatch):
    from aiter.ops.gemm_fusion.sp_qkv import test_utils as tu

    prof = tu.make_profile(window_bytes=8 << 20, max_windows=3, wait_budget_us=777)
    assert (
        prof.meta["window_budget"] == {"window_bytes": 8 << 20, "max_windows": 3}
        and prof.meta["wait_budget_us"] == 777
    )
    r = tu.make_res(window_bytes=8 << 20, max_windows=3)
    assert (r.window_bytes, r.max_windows, r.policy) == (
        8 << 20,
        3,
        "uniform",
    )  # no arena size -> uniform fallback
    merged = tu.make_res(
        profile=PROFILE, window_bytes=8 << 20, max_windows=5, policy="uniform"
    )  # a synthetic budget merged over a real profile
    assert (merged.window_bytes, merged.max_windows) == (8 << 20, 5)
    clear_profile_cache()
    assert profmod._CACHE == {}


def test_test_utils_arena_probe_registers_once_checks_views_and_uses_rank_as_device(
    capsys,
):
    from aiter.ops.gemm_fusion.sp_qkv import test_utils as tu

    class B(ArenaBackend):
        devs: ClassVar[list] = []

        def comm_create(self, rank, world, uid, heap, device):
            B.devs.append((rank, device))
            return 1

        def comm_destroy(self, comm):
            pass

        def win_deregister(self, comm, w):
            pass

        def win_free(self, comm, loc):
            pass

    for rank in range(4):
        be = B()
        be.heap = 8 << 30
        assert tu.run_arena(4 << 30, rank, 4, FakeDist(), be, f"arena#r{rank}")
    assert B.devs == [(0, 0), (1, 1), (2, 2), (3, 3)]  # HIP device = rank, NOT 0
    out = capsys.readouterr().out
    assert (
        out.count("registrations=1 views=22") == 4
        and "result=FAIL" not in out
        and out.count("view_winBase_ok=1") == 88
    )
    B.devs.clear()
    tu.run_arena(4 << 30, 2, 4, FakeDist(), B(), "x", device=5)
    assert B.devs == [(2, 5)]

    # a view whose read-back disagrees is reported
    class Bad(B):
        def win_info(self, comm, w):
            b, s, l = super().win_info(comm, w)
            return (b + 8, s, l) if w in self.views else (b, s, l)

    bb = Bad()
    assert not tu.run_arena(4 << 30, 0, 4, FakeDist(), bb, "bad")
    capsys.readouterr()
    # the plain sequences
    assert (
        tu.sequence("gate03_prefix") == [4 * MiB] * 4 + [40 * MiB]
        and tu.sequence("phase1_equal") == [40 * MiB] * 3
    )
    be = B()
    be.heap = 4 << 30
    assert tu.run(tu.sequence("gate03_prefix"), 4 << 30, 3, 4, FakeDist(), be, "g")
    assert B.devs[-1] == (3, 3) and capsys.readouterr().out.count("result=OK") == 5


# ======================================================================= ctypes bindings vs the native extern "C" exports (a fake backend must not hide a missing symbol)
import ast as _ast
import re as _re

_SYM = _re.compile(r"^(gf|sm|lg)_[a-z0-9_]+_v\d+$")
_RUNTIME_PY = (
    "aiter/gemm_fusion/resources.py",
    "aiter/ops/gemm_fusion/sp_qkv/small_rt.py",
    "aiter/ops/gemm_fusion/sp_qkv/large_rt.py",
)
_HOST_CPP = {
    "gf": "gf_comm.cpp",
    "sm": "sm_fusion_driver.cpp",
    "lg": "lg_fusion_driver.cpp",
}


def _repo(*parts):
    return os.path.join(os.path.dirname(__file__), "..", *parts)


def _cpp_exports_of(path, _seen=None):
    """Names of the C functions DEFINED in a host .cpp (return-type + name at line start), following `#include "...cpp"` (the vendored drivers are included unmodified)."""
    _seen = _seen if _seen is not None else set()
    path = os.path.normpath(path)
    _seen.add(path)
    text = _slurp(path, encoding="utf-8")
    # a DEFINITION: return type + name + (params) + '{' (parameters may span lines; a declaration ends in ';' and is not an export)
    names = set(
        _re.findall(
            r"^[ \t]*(?:extern\s+\"C\"\s+)?(?:int32_t|int|void\*|void|uint32_t|uint64_t|const char\*)\s+((?:gf|sm|lg)_[a-z0-9_]+_v\d+)\s*\([^;{]*?\)\s*\{",
            text,
            _re.MULTILINE,
        )
    )
    for inc in _re.findall(r'#include\s+"([^"]+\.cpp)"', text):
        p = os.path.normpath(os.path.join(os.path.dirname(path), inc))
        if os.path.exists(p) and p not in _seen:
            names |= _cpp_exports_of(p, _seen)
    return names


def _py_bound_and_called(source):
    """(bound, called): the native symbol names a Python source binds (string constants in argtypes tables, `lib.<name>` attributes) and the ones it CALLS."""
    tree = _ast.parse(source)
    bound, called = set(), set()
    docs = {
        id(n.value)
        for n in _ast.walk(tree)
        if isinstance(n, _ast.Expr) and isinstance(n.value, _ast.Constant)
    }  # bare string statements (docstrings) are not bindings
    for n in _ast.walk(tree):
        if (
            isinstance(n, _ast.Constant)
            and id(n) not in docs
            and isinstance(n.value, str)
            and _SYM.match(n.value)
        ):
            bound.add(n.value)
        elif isinstance(n, _ast.Attribute) and _SYM.match(n.attr):
            bound.add(n.attr)
        if (
            isinstance(n, _ast.Call)
            and isinstance(n.func, _ast.Attribute)
            and _SYM.match(n.func.attr)
        ):
            called.add(n.func.attr)
    return bound, called


def _check_bindings(py_sources: dict, exports: dict):
    """-> (missing {file: [names not exported by their library]}, dead [bound but called nowhere in the runtime])."""
    missing, all_bound, all_called = {}, set(), set()
    for f, src in py_sources.items():
        b, c = _py_bound_and_called(src)
        all_called |= c
    for f, src in py_sources.items():
        b, _c = _py_bound_and_called(src)
        all_bound |= b
        bad = sorted(n for n in b if n not in exports[n.split("_")[0]])
        if bad:
            missing[f] = bad
    return missing, sorted(all_bound - all_called)


def _runtime_sources():
    out = {}
    for rel in _RUNTIME_PY:
        out[rel] = _slurp(_repo(rel), encoding="utf-8")
    called_elsewhere = ""
    base = _repo("aiter", "ops", "gemm_fusion", "sp_qkv")
    for fn in ("variants.py", "variants_large.py"):
        called_elsewhere += _slurp(os.path.join(base, fn), encoding="utf-8") + "\n"
    return out, called_elsewhere


def test_every_ctypes_binding_exists_in_the_native_exports_and_none_is_dead():
    exports = {
        k: _cpp_exports_of(_repo("csrc", "gemm_fusion", "sp_qkv", "host", v))
        for k, v in _HOST_CPP.items()
    }
    assert (
        len(exports["gf"]) >= 15 and len(exports["sm"]) >= 6 and len(exports["lg"]) >= 8
    ), {
        k: len(v) for k, v in exports.items()
    }  # the extraction really found them
    py, other = _runtime_sources()
    missing, dead = _check_bindings(py, exports)
    missing = {f: names for f, names in missing.items() if names}
    assert (
        not missing
    ), f"ctypes bindings with no native export (a fake backend would hide this: AttributeError undefined symbol at load): {missing}"
    # a bound name counts as used when ANY runtime module calls it (the attach/window_bytes entries are called from variants*.py)
    _b, called_elsewhere = _py_bound_and_called(other)
    dead = [n for n in dead if n not in called_elsewhere]
    assert (
        not dead
    ), f"bound but never called anywhere in the runtime (remove the binding and the native): {dead}"
    # every native export that Python binds is a REAL definition: spot-check the ones the A-1 / teardown / poison paths depend on
    for must in (
        "gf_abi_info_v1",
        "gf_win_info_v1",
        "gf_win_view_v1",
        "gf_win_free_v1",
        "gf_zero_v1",
        "gf_mem_info_v1",
        "gf_devcomm_info_v1",
    ):
        assert must in exports["gf"], must
    assert "gf_comm_va_info_v1" not in exports["gf"] and not any(
        "gf_comm_va_info_v1" in s for s in py.values()
    )  # the removed native is not bound any more


def test_binding_checker_flags_a_missing_native_and_a_dead_binding_negative_controls():
    exports = {"gf": {"gf_a_v1", "gf_b_v1"}, "sm": set(), "lg": set()}
    good = 'lib.gf_a_v1.restype = 1\nx = lib.gf_a_v1(2)\nfor n, a in [("gf_b_v1", [])]:\n    f = getattr(lib, n)\ny = lib.gf_b_v1(3)\n'
    assert _check_bindings({"p.py": good}, exports) == ({}, [])
    # the real incident: a name left in the argtypes table after its native was removed
    stale = good + 'for n, a in [("gf_comm_va_info_v1", [])]:\n    pass\n'
    missing, _dead = _check_bindings({"p.py": stale}, exports)
    assert missing == {"p.py": ["gf_comm_va_info_v1"]}
    # a binding that nothing calls is reported as dead
    dead_src = 'for n, a in [("gf_a_v1", []), ("gf_b_v1", [])]:\n    f = getattr(lib, n)\nx = lib.gf_a_v1(1)\n'
    assert _check_bindings({"p.py": dead_src}, exports) == ({}, ["gf_b_v1"])
    # comments and docstrings are not bindings
    assert _py_bound_and_called('# lib.gf_zzz_v1(1)\n"""gf_yyy_v1"""\n') == (
        set(),
        set(),
    )
    # the C++ extraction: definitions yes, declarations/calls no, included drivers followed
    import tempfile
    import uuid
    from pathlib import Path

    d = Path(tempfile.gettempdir()) / ("exp_" + uuid.uuid4().hex)
    d.mkdir()
    try:
        (d / "inc.cpp").write_text(
            'extern "C" {\nint32_t sm_inner_v1(void* c) {\n  return 0;\n}\n}\n'
        )
        (d / "top.cpp").write_text(
            '#include "inc.cpp"\nextern "C" {\nint32_t gf_outer_v1(int x) { return gf_other_v1(x); }\nvoid* lg_ptr_v1(void* h) { return h; }\n'
            "int32_t gf_decl_only_v1(int);\n}\n"
        )
        assert _cpp_exports_of(str(d / "top.cpp")) == {
            "sm_inner_v1",
            "gf_outer_v1",
            "lg_ptr_v1",
        }
    finally:
        import shutil

        shutil.rmtree(d, ignore_errors=True)


# ======================================================================= task-264: plan.prepare/run input contract (inputs.py)
def _small_ready(monkeypatch, mode="dense", T=64, src=None, extra=None):
    """The binding rig plus a recording launch stub: returns (plan, st, meta, outs, res, launches, binds)."""
    from types import SimpleNamespace as NS

    plan, st, meta, outs, res, binds, _z = _binding_rig(
        monkeypatch, mode, T, src, extra
    )
    launches = []
    plan._context = lambda r: NS(
        bind=lambda sb: binds.append(sb) or object(),
        launch=lambda *a: launches.append(a),
        launch_v3=lambda *a: launches.append(("v3",) + a),
    )
    sys.modules["torch"].cuda = NS(is_current_stream_capturing=lambda: False)
    monkeypatch.setattr(
        "aiter.ops.gemm_fusion.sp_qkv.small_rt.stream_handle", lambda s: "S"
    )
    return plan, st, meta, outs, res, launches, binds


def _run_small(plan, st, meta, outs, res, **over):
    kw = {
        "x": st["x"],
        "x_scale": st["xs"],
        "w": st["w"],
        "w_scale": st["ws"],
        "outs": outs,
        "meta": meta,
        "res": res,
    }
    kw.update(over)
    return plan.run(**kw)


def test_inputs_small_accepts_every_deliberately_allowed_form(monkeypatch):
    plan, st, meta, outs, res, launches, _binds = _small_ready(monkeypatch)
    M, T = 16, 64
    _run_small(plan, st, meta, outs, res)  # the baseline
    _run_small(
        plan, st, meta, outs, res, x=_T((M, 6144), (8192, 1), "fp8")
    )  # a row-strided view (stride(0) > K): strideA carries it
    _run_small(
        plan, st, meta, outs, res, w_scale=_T((9216, 1), (1, 1))
    )  # [N,1] (what ATOM's weight_scale is), not only [N]
    _run_small(
        plan, st, meta, outs, res, w_scale=_T((9344,), (1,)), w=_w_ok(9344)
    )  # N padded to 128 (ATOM): a prefix-compatible weight / scale with MORE rows
    _run_small(
        plan, st, meta, outs, res, x_scale=_T((M,), (1,))
    )  # [M] as well as [M,1]
    longer = {
        "positions": _T((T + 64,), (1,), "i64"),
        "slot_mapping": _T((T + 64,), (1,), "i64"),
    }  # a longer persistent buffer is safe: the kernel reads t < T
    _run_small(plan, st, {**meta, **longer}, outs, res)
    assert len(launches) == 6
    # a decode pad row has slot -1: only the LENGTH is checked (values are never inspected), so exact-length arrays are fine whatever they hold
    _run_small(plan, st, meta, outs, res)
    # reuse and indexer plans take the 9856-row weight; the dense plan accepts it too (it reads the first 9216 rows)
    _run_small(plan, st, meta, outs, res, w=_w_ok(9856), w_scale=_T((9856, 1), (1, 1)))


@pytest.mark.parametrize(
    "what,over,match",
    [
        (
            "x K-padded (shape[1]=6400)",
            {"x": _T((16, 6400), (6400, 1), "fp8")},
            r"x K=6400 != 6144.*K-padded activation",
        ),
        (
            "x K stride != 1",
            {"x": _T((16, 6144), (6144, 2), "fp8")},
            r"x must be K-contiguous",
        ),
        (
            "x row stride < K",
            {"x": _T((16, 6144), (4096, 1), "fp8")},
            r"row stride >= 6144",
        ),
        (
            "x not fp8",
            {"x": _T((16, 6144), (6144, 1), "bf16")},
            r"x dtype bf16 is not float8_e4m3fn",
        ),
        ("x 1-D", {"x": _T((16,), (1,), "fp8")}, r"x must be 2-D"),
        (
            "x_scale not fp32",
            {"x_scale": _T((16, 1), (1, 1), "bf16")},
            r"x_scale dtype bf16 is not float32",
        ),
        (
            "x_scale wrong length",
            {"x_scale": _T((15, 1), (1, 1))},
            r"x_scale must be contiguous with exactly 16 elements",
        ),
        (
            "x_scale non-contiguous",
            {"x_scale": _T((16, 2), (2, 1))},
            r"x_scale must be contiguous with exactly 16",
        ),
        (
            "w_scale not fp32",
            {"w_scale": _T((9216,), (1,), "bf16")},
            r"w_scale dtype bf16 is not float32",
        ),
        (
            "w_scale short",
            {"w_scale": _T((9215,), (1,))},
            r"w_scale must be contiguous \[N\] or \[N,1\] with N >= 9216",
        ),
        (
            "w_scale [N,2]",
            {"w_scale": _T((9216, 2), (2, 1))},
            r"w_scale must be contiguous \[N\] or \[N,1\]",
        ),
        (
            "w_scale non-contiguous",
            {"w_scale": _T((9216,), (2,))},
            r"w_scale must be contiguous",
        ),
        (
            "w K-padded (shape[1]=6400)",
            {"w": _w_ok(9216, 6400)},
            r"w K=6400 != 6144.*K-padded weight",
        ),
        (
            "w shuffled with K padding (attrs)",
            {
                "w": type(
                    "W", (_T,), {"aiter_padded_k": 6400, "aiter_original_k": 6144}
                )((9216, 6144), (6144, 1), "fp8")
            },
            r"shuffled with K padding",
        ),
        (
            "w original K != 6144 (attrs)",
            {
                "w": type("W", (_T,), {"aiter_original_k": 6000})(
                    (9216, 6144), (6144, 1), "fp8"
                )
            },
            r"shuffled with K padding",
        ),
        ("w unshuffled", {"w": _w_ok(9216, shuffled=False)}, r"is_shuffled=False"),
        ("w too few rows", {"w": _w_ok(9200)}, r"w has 9200 rows < N=9216"),
        (
            "w not fp8",
            {"w": _T((9216, 6144), (6144, 1), "bf16")},
            r"w dtype bf16 is not float8_e4m3fn",
        ),
        (
            "w non-contiguous",
            {"w": _T((9216, 6144), (6144 * 2, 1), "fp8")},
            r"w must be contiguous",
        ),
        ("w 1-D", {"w": _T((9216,), (1,), "fp8")}, r"w must be 2-D"),
    ],
)
def test_inputs_small_rejects_each_violation_before_any_binding_or_launch(
    monkeypatch, what, over, match
):
    from aiter.gemm_fusion import ContractError

    plan, st, meta, outs, res, launches, binds = _small_ready(monkeypatch)
    with pytest.raises(ContractError, match=match):
        _run_small(plan, st, meta, outs, res, **over)
    assert (
        launches == [] and binds == []
    ), what  # rejected BEFORE a binding was created or anything launched


@pytest.mark.parametrize(
    "what,meta_over,outs_over,match",
    [
        (
            "positions shorter than T (the pad_rows case: real tokens < padded)",
            {"positions": _T((61,), (1,), "i64")},
            {},
            r"positions has 61 entries but the kernel reads T = 4 x local M = 64: stage a T-long array",
        ),
        (
            "slot_mapping shorter than T",
            {"slot_mapping": _T((63,), (1,), "i64")},
            {},
            r"slot_mapping has 63 entries but the kernel reads T = 4 x local M = 64",
        ),
        (
            "positions int32",
            {"positions": _T((64,), (1,), "i32")},
            {},
            r"positions dtype i32 is not int64",
        ),
        (
            "slot_mapping int32",
            {"slot_mapping": _T((64,), (1,), "i32")},
            {},
            r"slot_mapping dtype i32 is not int64",
        ),
        (
            "q output too short",
            {},
            {"q": _T((60, 16, 128))},
            r"output 'q' has 60 rows but the kernel writes 64",
        ),
        ("k output too short", {}, {"k": _T((32, 1, 128))}, r"output 'k' has 32 rows"),
    ],
)
def test_inputs_small_rejects_short_or_mistyped_token_arrays(
    monkeypatch, what, meta_over, outs_over, match
):
    from aiter.gemm_fusion import ContractError

    plan, st, meta, outs, res, launches, binds = _small_ready(monkeypatch)
    with pytest.raises(ContractError, match=match):
        _run_small(plan, st, {**meta, **meta_over}, {**outs, **outs_over}, res)
    assert launches == [] and binds == [], what


def test_inputs_small_rejects_rows_outside_1_to_32_and_checks_prepare_without_a_weight(
    monkeypatch,
):
    from aiter.gemm_fusion import ContractError

    plan, st, meta, outs, res, _launches, binds = _small_ready(monkeypatch)
    with pytest.raises(
        ContractError,
        match=r"x has 33 rows, a src r\d+ small plan takes local M 1\.\.32",
    ):
        _run_small(plan, st, meta, outs, res, x=_T((33, 6144), (6144, 1), "fp8"))
    # prepare(): no weight is passed, everything else is checked ONCE, outside capture, before the binding exists
    plan.prepare(
        x=st["x"], x_scale=st["xs"], w_scale=st["ws"], outs=outs, meta=meta, res=res
    )
    assert len(binds) == 1
    bad = {
        **meta,
        "positions": _T((61,), (1,), "i64"),
    }  # real tokens (61) < padded rows x world (64)
    plan2, st2, meta2, outs2, res2, _l2, b2 = _small_ready(monkeypatch)
    with pytest.raises(ContractError, match="stage a T-long array.*slot_mapping -1"):
        plan2.prepare(
            x=st2["x"],
            x_scale=st2["xs"],
            w_scale=st2["ws"],
            outs=outs2,
            meta={**meta2, **bad},
            res=res2,
        )
    with pytest.raises(ContractError, match="x K=6400"):
        plan2.prepare(
            x=_T((16, 6400), (6400, 1), "fp8"),
            x_scale=st2["xs"],
            w_scale=st2["ws"],
            outs=outs2,
            meta=meta2,
            res=res2,
        )
    assert b2 == []


def test_inputs_small_indexer_and_reuse_need_the_9856_row_weight_dense_does_not(
    monkeypatch,
):
    from aiter.gemm_fusion import ContractError

    for mode in ("indexer", "reuse"):
        plan, st, meta, outs, res, _launches, _binds = _small_ready(
            monkeypatch, mode=mode
        )
        # the rig gives these modes their own k/v/index_q outputs only where the plan needs them; the weight contract is what is under test
        meta = {
            **meta,
            "index_cache": _T((4, 1, 16, 128), (2048, 2048, 128, 1)),
            "index_q_norm_w": _T((128,)),
            "index_k_norm_w": _T((128,)),
        }
        outs = (
            {"q": outs["q"], "index_q": _T((64, 1, 128))} if mode == "indexer" else outs
        )
        with pytest.raises(ContractError, match=r"w has 9216 rows < N=9856"):
            _run_small(
                plan, st, meta, outs, res, w=_w_ok(9216), w_scale=_T((9856, 1), (1, 1))
            )
        with pytest.raises(
            ContractError,
            match=r"w_scale must be contiguous \[N\] or \[N,1\] with N >= 9856",
        ):
            _run_small(plan, st, meta, outs, res, w_scale=_T((9216,), (1,)))


# ---- large plans
def _large_ready(monkeypatch, mode="dense", T=8192):
    import types as _t
    from types import SimpleNamespace as NS

    monkeypatch.setitem(sys.modules, "torch", _t.ModuleType("torch"))
    tm = sys.modules["torch"]
    tm.float32, tm.float8_e4m3fn, tm.int64 = "f32", "fp8", "i64"
    from aiter.gemm_fusion import catalog

    monkeypatch.setattr(catalog, "_CATALOG", {})
    plan = _plan_for(mode, T, True)
    M = T // 4
    launches = []
    plan._context = lambda r: NS(
        launch=lambda *a, **k: launches.append(a),
        launch_v3=lambda *a, **k: launches.append(a),
    )
    monkeypatch.setattr(
        "aiter.ops.gemm_fusion.sp_qkv.small_rt.stream_handle", lambda s: "S"
    )
    n_rows = 9856 if mode == "indexer" else 9216
    st = {
        "x": _T((M, 6144), (6144, 1), "fp8"),
        "xs": _T((M, 1), (1, 1)),
        "ws": _T((n_rows, 1), (1, 1)),
        "w": _w_ok(n_rows),
    }
    return plan, st, type("R", (), {})(), launches


def _run_large(plan, st, res, **over):
    kw = {
        "x": st["x"],
        "x_scale": st["xs"],
        "w": st["w"],
        "w_scale": st["ws"],
        "res": res,
        "call_id": 7,  # the profile's large plans are R12 (abi 3): call_id is required
    }
    kw.update(over)
    return plan.run(**kw)


def test_inputs_large_accepts_the_allowed_forms_and_rejects_each_violation(monkeypatch):
    from aiter.gemm_fusion import ContractError

    plan, st, res, launches = _large_ready(monkeypatch)
    M = 2048
    _run_large(
        plan, st, res
    )  # [N,1] w_scale and a caller-owned x_scale, as ATOM passes them
    _run_large(plan, st, res, x=_T((M, 6144), (8192, 1), "fp8"))  # row-strided view
    _run_large(
        plan, st, res, w=_w_ok(9344), w_scale=_T((9344, 1), (1, 1))
    )  # N-padded weight: prefix compatible
    _run_large(
        plan, st, res, w=_w_ok(9856), w_scale=_T((9856,), (1,))
    )  # the dense build reads the first 9216 rows of the indexer weight
    assert len(launches) == 4
    cases = [
        ({"x": _T((M, 6400), (6400, 1), "fp8")}, r"x K=6400 != 6144"),
        ({"x": _T((M, 6144), (6144, 2), "fp8")}, r"x must be K-contiguous"),
        (
            {"x": _T((M, 6144), (6144, 1), "bf16")},
            r"x dtype bf16 is not float8_e4m3fn",
        ),
        ({"x_scale": _T((M, 1), (1, 1), "bf16")}, r"x_scale dtype bf16"),
        ({"x_scale": _T((M - 1, 1), (1, 1))}, rf"exactly {M} elements"),
        ({"w_scale": _T((9215, 1), (1, 1))}, r"w_scale must be contiguous.*N >= 9216"),
        ({"w_scale": _T((9216,), (1,), "bf16")}, r"w_scale dtype bf16"),
        ({"w": _w_ok(9216, 6400)}, r"w K=6400 != 6144.*K-padded weight"),
        ({"w": _w_ok(9216, shuffled=False)}, r"is_shuffled=False"),
        ({"w": _w_ok(9215)}, r"w has 9215 rows < N=9216"),
        ({"w": _T((9216, 6144), (6144, 1), "bf16")}, r"w dtype bf16"),
        ({"w": _T((9216, 6144), (12288, 1), "fp8")}, r"w must be contiguous"),
    ]
    for over, match in cases:
        n = len(launches)
        with pytest.raises(ContractError, match=match):
            _run_large(plan, st, res, **over)
        assert len(launches) == n  # nothing launched
    # the pre-existing row-count error is unchanged
    with pytest.raises(ValueError, match="x rows 1024 != plan local M 2048"):
        _run_large(plan, st, res, x=_T((1024, 6144), (6144, 1), "fp8"))


def test_inputs_large_indexer_needs_9856_rows_and_prepare_checks_when_tensors_are_passed(
    monkeypatch,
):
    from aiter.gemm_fusion import ContractError

    plan, st, res, _launches = _large_ready(monkeypatch, mode="indexer", T=16384)
    with pytest.raises(ContractError, match=r"w has 9216 rows < N=9856"):
        _run_large(plan, st, res, w=_w_ok(9216))
    _run_large(plan, st, res)
    # prepare(): the old call shape (res only) still works; with the activation tensors (as ATOM passes them) they are checked once, outside capture
    plan.prepare(res=res)
    plan.prepare(res=res, x=st["x"], x_scale=st["xs"], w_scale=st["ws"])
    with pytest.raises(ContractError, match=r"x K=6400"):
        plan.prepare(
            res=res,
            x=_T((4096, 6400), (6400, 1), "fp8"),
            x_scale=st["xs"],
            w_scale=st["ws"],
        )
    with pytest.raises(ContractError, match=r"w_scale must be contiguous.*N >= 9856"):
        plan.prepare(
            res=res, x=st["x"], x_scale=st["xs"], w_scale=_T((9216, 1), (1, 1))
        )


def test_inputs_checks_are_pure_python_and_only_run_where_python_runs():
    """The contract: prepare() once, run() at capture once and in eager every call; a graph replay executes no Python. Pin the wiring (read from the sources: importing
    the variant modules again would re-register them in the catalog)."""
    import ast as _a

    base = os.path.dirname(specs.__file__)

    def method_src(fname, cls, meth):
        text = _slurp(os.path.join(base, fname), encoding="utf-8")
        for n in _a.parse(text).body:
            if isinstance(n, _a.ClassDef) and n.name == cls:
                for m in n.body:
                    if isinstance(m, _a.FunctionDef) and m.name == meth:
                        return _a.get_source_segment(text, m)
        raise AssertionError(f"{cls}.{meth} not found")

    small_prep, small_run = method_src(
        "variants.py", "SpQkvSmall", "prepare"
    ), method_src("variants.py", "SpQkvSmall", "run")
    assert _sq("self._check_inputs(x, x_scale, w_scale, outs, meta)") in _sq(small_prep)
    assert _sq("self._check_inputs(x, x_scale, w_scale, outs, meta, w)") in _sq(
        small_run
    )
    assert small_run.index("self._check_inputs") < small_run.index(
        "self._binding("
    )  # rejected before any binding
    assert _sq("self._check_inputs(x, x_scale, w_scale, w)") in _sq(
        method_src("variants_large.py", "SpQkvLarge", "run")
    )
    assert _sq("self._check_inputs(x, x_scale, w_scale)") in _sq(
        method_src("variants_large.py", "SpQkvLarge", "prepare")
    )
    src = _slurp(os.path.join(base, "inputs.py"), encoding="utf-8")
    code = "\n".join(l for l in src.split("\n") if not l.lstrip().startswith("#"))
    assert (
        ".item(" not in code
        and "synchronize" not in code
        and ".cuda" not in code
        and "hipMemcpy" not in code
    )  # attribute compares only; tensor VALUES are never read (slot -1 stays legal)


# ---- task-264 (B): small shapes run fused after padding: ATOM stages T-long arrays whose pad rows are (position 0, slot -1)
class _NoRead(_T):
    """A tensor whose VALUES must never be read by the contract (slot -1 / position 0 pad rows are legal and only the kernel interprets them)."""

    def _boom(self, *a, **k):
        raise AssertionError("the input contract must not read tensor values")

    item = tolist = cpu = numpy = __getitem__ = __iter__ = __bool__ = _boom


def test_inputs_small_accepts_pad_rows_staged_as_T_long_arrays_and_never_reads_values(
    monkeypatch,
):
    for mode, T in (("dense", 64), ("dense", 128), ("indexer", 64), ("reuse", 128)):
        plan, st, meta, outs, res, launches, _binds = _small_ready(
            monkeypatch, mode=mode, T=T
        )
        M = T // 4
        # 5 real tokens + (T - 5) pad rows: positions 0 / slot -1 for the pad rows; the arrays are T long, the activation has all M rows, outputs have T rows
        pos, slot = _NoRead((T,), (1,), "i64"), _NoRead((T,), (1,), "i64")
        meta = {**meta, "positions": pos, "slot_mapping": slot}
        if mode == "indexer":
            meta = {
                **meta,
                "index_cache": _T((4, 1, 16, 128), (2048, 2048, 128, 1)),
                "index_q_norm_w": _T((128,)),
                "index_k_norm_w": _T((128,)),
            }
            outs = {"q": outs["q"], "index_q": _T((T, 1, 128))}
        elif mode == "reuse":
            outs = dict(outs)
        _run_small(plan, st, meta, outs, res)
        plan.prepare(
            x=st["x"], x_scale=st["xs"], w_scale=st["ws"], outs=outs, meta=meta, res=res
        )
        assert len(launches) == 1 and st["x"].shape[0] == M
    # the unit-level pin of the kernel facts this relies on (ran_kernel.cu, r17/r18): the index cache and the KV cache writes are guarded by slot >= 0
    for src in ("r17", "r18"):
        text = _slurp(
            os.path.join(
                os.path.dirname(specs.__file__),
                "..",
                "..",
                "..",
                "..",
                "csrc",
                "gemm_fusion",
                "sp_qkv",
                "small",
                src,
                "device",
                "ran_kernel.cu",
            ),
            encoding="utf-8",
        )
        assert (
            text.count("if (slot >= 0) {") >= 2 and "if (is >= 0) {" in text
        ), src  # KV (k and v) writes skip slot -1; the index_cache write skips islot -1
        assert (
            "a.index_slot_mapping[t]" in text and "(live ? a.positions[t] : 0)" in text
        )


# ======================================================================= A1: R12 plumbing (jit/ABI tables, runtimes, abi 3 plans, profile, prebuild)
def test_jit_abi_tables_sources_and_missing_source_error():
    assert jit.SRCS == ("r17", "r18", "r19", "r20", "r21") and jit.LARGE_SRCS == (
        "r11",
        "r12",
    )
    assert [jit.src_abi("small", s) for s in jit.SRCS] == [2, 2, 3, 3, 3] and [
        jit.src_abi("large", s) for s in jit.LARGE_SRCS
    ] == [2, 3]
    with pytest.raises(ValueError, match="not in"):
        jit.src_abi("small", "r22")
    # r19 is derived from r18: it accepts the r18-only flags; r17 still rejects them
    _d, h = jit.small_defs({"src": "r19", "b_rowsplit": 1, "rx_bypass": 1}, "dense")
    assert "-DSM_B_ROWSPLIT=1" in h and "-DSM_RX_BYPASS=1" in h
    with pytest.raises(ValueError, match="require src=r18 or r19"):
        jit.small_defs({"src": "r17", "b_rowsplit": 1}, "dense")
    with pytest.raises(ValueError, match="src"):
        jit.small_defs({"src": "r22"}, "dense")
    # large: src is a parameter (default r11 for old profiles) and is validated
    assert jit.large_defs({"src": "r12", "n": 2}, "dense") == jit.large_defs(
        {"src": "r11", "n": 2}, "dense"
    ) + [
        "-DLG_TAG_EARLY=1",
        "-DLG_POLL_BATCH=1",
        "-DLG_QR_WGS=64",
    ]  # r12 = r11's flags + the r12b / r12c knobs (passed explicitly, defaults 1 / 1 / 64)
    with pytest.raises(ValueError, match="large src 'r13'"):
        jit.large_defs({"src": "r13"}, "dense")
    tool = dict(TOOL_SIM)
    # the r19 sources are delivered by the implementor later: a clear error, not a FileNotFoundError
    dev, host = jit.small_defs({"src": "r19"}, "dense")
    if not os.path.isdir(os.path.join(jit.CSRC, "small", "r19")):
        with pytest.raises(jit.BuildError, match="small source 'r19' is not present"):
            jit.cache_key("r19", dev, host, tool)
    # large r12 IS delivered: its key differs from r11's and its commands point into large/r12
    defs = jit.large_defs({"src": "r12"}, "dense")
    k11, k12 = jit.large_cache_key(defs, tool, "r11"), jit.large_cache_key(
        defs, tool, "r12"
    )
    assert k11 != k12 and k11 == jit.large_cache_key(
        defs, tool
    )  # the default stays r11
    fwd = lambda cmd: [a.replace("\\", "/") for a in cmd]
    out = fwd(jit.large_device_cmd(defs, tool, "/out", "r12"))
    assert any("/large/r12/device" in a for a in out) and any(
        a.endswith("large/r12/device/gemm_quad_fp8_comm.cu") for a in out
    )
    assert any(
        "/large/r11/device" in a for a in fwd(jit.large_device_cmd(defs, tool, "/out"))
    )
    assert any(
        "/large/r12/host" in a
        for a in fwd(jit.large_host_cmd(defs, tool, "/out", "r12"))
    )
    assert any(
        "/large/r12/device" in a
        for a in fwd(jit.large_device_only_cmd(defs, tool, "/out", "r12"))
    )


TOOL_SIM = {
    "hipcc": "/rocm/bin/hipcc",
    "clangxx": "/rocm/llvm/bin/clang++",
    "bundler": "/rocm/llvm/bin/clang-offload-bundler",
    "rocm": "/rocm",
    "mori_root": "/mori",
    "mori_so": "/mori/libmori_cco.so",
}


def test_small_cache_key_of_an_r19_source_follows_its_manifest(monkeypatch):
    import shutil
    import tempfile
    import uuid

    root = os.path.join(tempfile.gettempdir(), "csrc_" + uuid.uuid4().hex)
    os.makedirs(os.path.join(root, "small", "r19"))
    os.makedirs(os.path.join(root, "host"))
    for f in ("sm_fusion_driver.cpp", "gf_comm.cpp"):
        _spit(os.path.join(root, "host", f), "x")
    _spit(
        os.path.join(root, "small", "r19", "SOURCE_SHA256.txt"),
        "aa  device/ran_kernel.cu\n",
    )
    try:
        monkeypatch.setattr(jit, "CSRC", root)
        dev, host = jit.small_defs({"src": "r19"}, "dense")
        k1 = jit.cache_key("r19", dev, host, TOOL_SIM)
        _spit(
            os.path.join(root, "small", "r19", "SOURCE_SHA256.txt"),
            "bb  device/ran_kernel.cu\n",
        )
        k2 = jit.cache_key("r19", dev, host, TOOL_SIM)
        assert len(k1) == 20 and k1 != k2  # a new r19 manifest is a new build
        assert jit.recipe_key(k1, "aot1") != k1
    finally:
        shutil.rmtree(root, ignore_errors=True)


class _FakeCDLL:
    """A ctypes library stand-in: any attribute is a function object that takes restype / argtypes and records its calls."""

    def __init__(self, exports=None):
        self.exports = exports
        self.calls = []
        self._fn = {}
        self.returns = {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if self.exports is not None and name not in self.exports:
            raise AttributeError(name)
        f = self._fn.get(name)
        if f is None:
            f = self._fn[name] = types.SimpleNamespace(restype=None, argtypes=None)
            real = f

            class _Callable(types.SimpleNamespace):
                pass

            def call(*a, _n=name):
                self.calls.append((_n, a))
                return self.returns.get(_n, 0)

            self._fn[name] = _CallableFn(real, call)
        return self._fn[name]


class _CallableFn:
    def __init__(self, ns, call):
        self.__dict__["_ns"], self.__dict__["_call"] = ns, call

    def __setattr__(self, k, v):
        setattr(self._ns, k, v)

    def __getattr__(self, k):
        return getattr(self._ns, k)

    def __call__(self, *a):
        return self._call(*a)


def test_small_rt_v3_struct_layout_loader_and_launch_argument_order(monkeypatch):
    import ctypes as C

    from aiter.ops.gemm_fusion.sp_qkv import small_rt as srt

    S = srt.SmRanArgsV3
    assert C.sizeof(S) == srt.SM_RAN_ARGS_V3_BYTES == 64
    off = {n: getattr(S, n).offset for n, _t in S._fields_}
    assert off == {
        "blk": 0,
        "Ascale": 8,
        "positions": 16,
        "slot_mapping": 24,
        "err": 32,
        "step": 40,
        "M": 48,
        "rank": 52,
        "t_valid": 56,
        "call_id": 60,
    }
    fake3 = _FakeCDLL()
    monkeypatch.setattr(srt.C, "CDLL", lambda path: fake3)
    srt.load_driver("x.so", abi=3)
    bound = {n for n, f in fake3._fn.items() if f.argtypes is not None}
    assert {
        "sm_attach_v3",
        "sm_launch_blk_v3",
        "sm_bind_new_v1",
        "sm_blk_free_v1",
        "sm_free_v1",
        "sm_info_v1",
    } == bound  # no per-plan poison entry
    assert (
        len(fake3.sm_attach_v3.argtypes) == 11
        and len(fake3.sm_launch_blk_v3.argtypes) == 11
    )
    fake2 = _FakeCDLL()
    monkeypatch.setattr(srt.C, "CDLL", lambda path: fake2)
    srt.load_driver("x.so")
    assert (
        "sm_attach_v1" in fake2._fn
        and "sm_poison_fast_v1" in fake2._fn
        and "sm_attach_v3" not in fake2._fn
    )  # abi 2 unchanged
    ctx = srt.SmallContext(fake3, 0xC7)
    ctx.launch_v3("blk", 0xA, 0xB, 6144, 0xE, 0xF, 0x10, 61, 777, "stream")
    name, a = fake3.calls[-1]
    assert (
        name == "sm_launch_blk_v3"
        and a[0] == 0xC7
        and a[1] == "blk"
        and [x.value for x in a[2:9]] == [0xA, 0xB, 6144, 0xE, 0xF, 0x10, 61]
        and a[9].value == 777
        and a[10] == "stream"
    )
    fake3.returns = {"sm_launch_blk_v3": -5}
    with pytest.raises(RuntimeError, match="sm_launch_blk_v3 rc=-5"):
        ctx.launch_v3("blk", 1, 2, 3, 4, 5, 6, 7, 8, "s")


def test_large_rt_r12_loader_abi_check_and_launch(monkeypatch):
    import ctypes as C

    from aiter.ops.gemm_fusion.sp_qkv import large_rt as lrt

    assert (
        lrt.LG_COMM_ARGS_V3_BYTES,
        lrt.LG_COMM_ABI_VERSION_V3,
        lrt.LG_GF_PEERS_BYTES,
    ) == (352, 4, 32)
    fake = _FakeCDLL()
    monkeypatch.setattr(lrt.C, "CDLL", lambda path: fake)
    lrt.load_driver("x.so", abi=3)
    bound = {n for n, f in fake._fn.items() if f.argtypes is not None}
    assert {
        "lg_gpage_attach_v1",
        "lg_launch_v2",
        "lg_abi_info_v2",
        "lg_attach_v1",
        "lg_report_v1",
        "lg_drain_v1",
        "lg_init_control_v1",
    } <= bound
    assert not ({"lg_launch_v1", "lg_poison_v1"} & bound)  # compiled out for ABI >= 4
    assert (
        len(fake.lg_launch_v2.argtypes) == 12
        and len(fake.lg_gpage_attach_v1.argtypes) == 4
    )
    # abi_info + the static check
    good = (4, 352, 0, 8, 64, 3072, 3104, 32, 64)

    def fill(b, vals):
        for i, v in enumerate(vals):
            b[i] = v
        return 0

    class Lib:
        def __init__(self, vals):
            self.vals = vals

        def lg_abi_info_v2(self, b):
            return fill(b, self.vals)

    got = lrt.abi_info(Lib(good))
    assert got == {
        "abi": 4,
        "args_bytes": 352,
        "err_off": 0,
        "step_off": 8,
        "diag_off": 64,
        "ctag_off": 3072,
        "atag_off": 3104,
        "peers_bytes": 32,
        "qr_wgs": 64,
    }
    lrt.check_abi_info(got, 64)
    for field, bad in (
        ("abi", 3),
        ("args_bytes", 344),
        ("err_off", 4),
        ("diag_off", 128),
        ("peers_bytes", 16),
    ):
        with pytest.raises(RuntimeError, match=rf"ABI mismatch.*'{field}': \({bad}"):
            lrt.check_abi_info({**got, field: bad}, 64)
    # r12d: the host library reports its LG_QR_WGS (out[8]); the check refuses a value that differs from the one the device image was built with, naming both
    lrt.check_abi_info({**got, "qr_wgs": 1}, 1)
    for host, image in ((64, 1), (1, 64), (63, 64)):
        with pytest.raises(
            RuntimeError,
            match=rf"qr_wgs mismatch: the host library starts {host} quiet_ready workgroups but the device image was built with LG_QR_WGS={image}",
        ):
            lrt.check_abi_info({**got, "qr_wgs": host}, image)
    # launch_v3 / gpage_attach
    ctx = lrt.LargeContext(fake, C.c_void_p(0x11), 2048, 2304, 0)
    ctx.budget_us = 5_000_000
    ctx.launch_v3(0xA, 0xB, 0xC, 0xD, 6144, 3, 2, 12345, "stream")
    name, a = fake.calls[-1]
    assert (
        name == "lg_launch_v2"
        and a[-1] == "stream"
        and a[-2] == 12345
        and a[-3] == 2
        and a[-4] == 3
        and a[7] == 5_000_000
    )
    for bad in (-1, 1 << 31):
        with pytest.raises(ValueError, match="call_id"):
            ctx.launch_v3(1, 2, 3, 4, 5, 1, 0, bad, "s")
    ctx.gpage_attach(0x7000, 0x7008, 0x9000)
    n2, a2 = fake.calls[-1]
    assert n2 == "lg_gpage_attach_v1" and [x.value for x in a2[1:]] == [
        0x7000,
        0x7008,
        0x9000,
    ]
    ctx.gpage_attach(
        0x7000, 0x7008, None
    )  # no peers table: no propagation, still attaches
    assert fake.calls[-1][1][3].value in (0, None)
    fake.returns = {"lg_launch_v2": -104}
    with pytest.raises(RuntimeError, match="lg_launch_v2 rc=-104"):
        ctx.launch_v3(1, 2, 3, 4, 5, 1, 0, 5, "s")
    assert not hasattr(
        ctx, "nan_fill"
    )  # no separate fill node exists any more (r12c folded it into quiet_ready): the runtime has nothing to launch


def test_inputs_v3_token_arrays_call_id_and_short_arrays_are_legal(monkeypatch):
    from aiter.gemm_fusion import ContractError

    _plan, _st, _meta, outs, _res, _launches, _binds = _small_ready(
        monkeypatch, src="r19"
    )
    from aiter.ops.gemm_fusion.sp_qkv import inputs

    T = 64
    ok = _T((T,), (1,), "i64")
    assert (
        inputs.check_token_arrays_v3("o", T=T, positions=ok, slot_mapping=ok, outs=outs)
        == T
    )
    # SHORT arrays are legal for R12: t_valid is the shortest of the two (and at most T)
    assert (
        inputs.check_token_arrays_v3(
            "o",
            T=T,
            positions=_T((61,), (1,), "i64"),
            slot_mapping=_T((64,), (1,), "i64"),
            outs=outs,
        )
        == 61
    )
    assert (
        inputs.check_token_arrays_v3(
            "o",
            T=T,
            positions=_T((64,), (1,), "i64"),
            slot_mapping=_T((10,), (1,), "i64"),
            outs=outs,
        )
        == 10
    )
    assert (
        inputs.check_token_arrays_v3(
            "o",
            T=T,
            positions=_T((200,), (1,), "i64"),
            slot_mapping=_T((200,), (1,), "i64"),
            outs=outs,
        )
        == T
    )  # a longer buffer: only T are read
    for bad, match in (
        (_T((64,), (1,), "i32"), r"dtype i32 is not int64"),
        (_T((64,), (2,), "i64"), r"must be contiguous"),
        (_T((0,), (1,), "i64"), r"is empty"),
    ):
        with pytest.raises(ContractError, match=match):
            inputs.check_token_arrays_v3(
                "o", T=T, positions=bad, slot_mapping=ok, outs=outs
            )
    with pytest.raises(ContractError, match="output 'q' has 60 rows"):
        inputs.check_token_arrays_v3(
            "o", T=T, positions=ok, slot_mapping=ok, outs={"q": _T((60, 16, 128))}
        )
    assert (
        inputs.check_call_id("o", 1) == 1
        and inputs.check_call_id("o", (1 << 31) - 1) == (1 << 31) - 1
    )
    for bad in (0, -3, 1 << 31, True, 1.5, "7", None):
        with pytest.raises(
            ContractError, match=r"call_id must be an int in \[1, 2\^31\)"
        ):
            inputs.check_call_id("o", bad)


def test_small_abi3_plan_binding_key_run_kernargs_and_pad_rows(monkeypatch):
    from aiter.gemm_fusion import ContractError

    plan, st, meta, outs, res, launches, binds = _small_ready(monkeypatch, src="r19")
    T = 64
    assert plan.abi == 3 and plan.src == "r19"
    reqs = plan.resources()
    assert (
        [r.name.rsplit("/", 1)[-1] for r in reqs] == ["rx", "ctl", "gpage"]
        and reqs[2].share == "gf.global"
        and reqs[2].bytes == 4 << 20
    )
    # (a) run() needs call_id; it is validated before anything is bound or launched
    for bad in (None, 0, 1 << 31, True):
        with pytest.raises(ContractError, match="call_id"):
            (
                _run_small(plan, st, meta, outs, res, call_id=bad)
                if bad is not None
                else _run_small(plan, st, meta, outs, res)
            )
    assert launches == [] and binds == []
    # (b) pad rows: T-long arrays whose values are never read (position 0 / slot -1 live inside the kernel), kernargs carry the pointers + t_valid + call_id
    pos, slot = _NoRead((T,), (1,), "i64"), _NoRead((T,), (1,), "i64")
    m2 = {**meta, "positions": pos, "slot_mapping": slot}
    _run_small(plan, st, m2, outs, res, call_id=4242)
    (
        tag,
        _blk,
        a_ptr,
        w_ptr,
        stride_a,
        xs_ptr,
        pos_ptr,
        slot_ptr,
        t_valid,
        call_id,
        stream,
    ) = launches[-1]
    assert (
        tag == "v3"
        and (a_ptr, w_ptr, stride_a) == (st["x"].ptr, st["w"].ptr, 6144)
        and (xs_ptr, pos_ptr, slot_ptr) == (st["xs"].ptr, pos.ptr, slot.ptr)
    )
    assert (t_valid, call_id, stream) == (T, 4242, "S")
    # (c) short arrays (positions carries the real token count): legal, t_valid says how many rows are real
    _run_small(
        plan, st, {**meta, "positions": _T((61,), (1,), "i64")}, outs, res, call_id=4242
    )
    assert launches[-1][8] == 61
    # (d) the binding no longer contains x_scale / positions / slot_mapping: fresh tensors every call never create a binding (abi 2 does)
    assert len(binds) >= 1
    n = len(binds)
    for _ in range(3):
        _run_small(
            plan,
            {**st, "xs": _T((16, 1), (1, 1))},
            {
                **meta,
                "positions": _T((T,), (1,), "i64"),
                "slot_mapping": _T((T,), (1,), "i64"),
            },
            outs,
            res,
            call_id=9,
        )
    assert len(binds) == n  # no new binding
    sb = binds[0]
    assert (
        not sb.Ascale
        and not sb.positions
        and not sb.slot_mapping
        and not sb.index_slot_mapping
        and sb.Wscale == st["ws"].ptr
    )  # w_scale stays baked; the rest are kernargs
    plan2, st2, meta2, outs2, res2, launches2, binds2 = _small_ready(
        monkeypatch, src=None
    )  # control: the shipped abi-2 plan keeps the pointers in the binding
    assert plan2.abi == 2
    _run_small(plan2, st2, meta2, outs2, res2)
    _run_small(plan2, {**st2, "xs": _T((16, 1), (1, 1))}, meta2, outs2, res2)
    assert len(binds2) == 2 and launches2[-1][0] != "v3"
    # (e) no per-plan poison for R12 plans
    with pytest.raises(RuntimeError, match="no per-plan poison word"):
        plan.poison(res)
    assert not hasattr(plan2, "never") and plan2.poison  # abi 2 keeps its method


class _FakeRes:
    """Just enough of FusionResources for a plan's _context()."""

    def __init__(self, wins):
        self.rank, self.comm, self.devcomm, self.device = 1, 0xC0, 0xD0, 3
        self.finalized, self.sdma_queues, self.windows = True, 1, {}
        self.wins, self.calls = wins, []
        self.wait_budget_us, self.wait_budget_source = 5_000_000, "default"

    def ensure(self, owner, reqs):
        self.calls.append(("ensure", owner, [r.name for r in reqs]))
        self.windows[owner] = self.wins
        return self.wins

    def global_page(self):
        return types.SimpleNamespace(local=0x7000, err_addr=0x7000, step_addr=0x7008)

    def global_peers(self):
        return types.SimpleNamespace(data_ptr=lambda: 0x9000)

    def __getattr__(self, name):
        if name.startswith("set_"):
            return lambda *a, **k: (self.calls.append((name, a, k)), True)[1]
        raise AttributeError(name)


def test_small_abi3_context_attaches_with_the_page_and_registers_no_poison_hook(
    monkeypatch,
):
    plan, *_ = _binding_rig(monkeypatch, src="r19")
    del plan._context  # the rig stubbed it: use the real one against fakes
    v = sys.modules["aiter.ops.gemm_fusion.sp_qkv.variants"]
    from aiter.ops.gemm_fusion.sp_qkv import small_rt as srt

    exp = v.small_window_bytes(plan.mode)
    lib = _FakeCDLL()
    monkeypatch.setattr(
        v,
        "build_small",
        lambda params, mode: types.SimpleNamespace(
            driver_so="d.so", gemm_hsaco="g.hsaco", ran_hsaco="r.hsaco"
        ),
    )
    seen = {}
    monkeypatch.setattr(
        srt, "load_driver", lambda path, abi=2: (seen.update(abi=abi), lib)[1]
    )
    monkeypatch.setattr(
        srt, "info", lambda lib_: {"rx_bytes": exp["rx"], "ctl_bytes": exp["ctl"]}
    )
    wins = {
        plan.owner + "/rx": types.SimpleNamespace(handle=1, local=0x100),
        plan.owner + "/ctl": types.SimpleNamespace(handle=2, local=0x200),
    }
    res = _FakeRes(wins)
    plan._context(res)
    assert seen["abi"] == 3
    name, a = lib.calls[-1]
    assert name == "sm_attach_v3"
    assert (
        a[0] == 1 and a[1] == 0xC0 and a[2:6] == (1, 0x100, 2, 0x200)
    )  # rank, comm, rx handle/local, ctl handle/local
    assert (
        a[6].value == 0x7000 and a[7].value == 0x9000
    )  # the global page's local address, the device-resident peers table
    assert [c[0] for c in res.calls if c[0].startswith("set_")] == [
        "set_close"
    ]  # NO set_poison: the error is the per-comm ERR word
    # control: the shipped abi-2 plan still attaches with v1 and registers its poison hook
    plan2, *_ = _binding_rig(
        monkeypatch, src=None
    )  # re-imports the variant module: patch THIS one
    del plan2._context
    v2 = sys.modules["aiter.ops.gemm_fusion.sp_qkv.variants"]
    monkeypatch.setattr(
        v2,
        "build_small",
        lambda params, mode: types.SimpleNamespace(
            driver_so="d.so", gemm_hsaco="g.hsaco", ran_hsaco="r.hsaco"
        ),
    )
    lib2 = _FakeCDLL()
    seen2 = {}
    monkeypatch.setattr(
        srt, "load_driver", lambda path, abi=2: (seen2.update(abi=abi), lib2)[1]
    )
    res2 = _FakeRes(
        {
            plan2.owner + "/rx": types.SimpleNamespace(handle=1, local=0x100),
            plan2.owner + "/ctl": types.SimpleNamespace(handle=2, local=0x200),
        }
    )
    plan2._context(res2)
    assert (
        seen2["abi"] == 2
        and lib2.calls[-1][0] == "sm_attach_v1"
        and [c[0] for c in res2.calls if c[0].startswith("set_")]
        == ["set_close", "set_poison"]
    )


def _forget_sp_modules():
    """Drop the variant modules (sys.modules AND the package attributes: `from . import x` checks those first) so the next import re-registers them."""
    pkg = sys.modules.get("aiter.ops.gemm_fusion.sp_qkv")
    for n in ("variants", "variants_large"):
        sys.modules.pop(f"aiter.ops.gemm_fusion.sp_qkv.{n}", None)
        if pkg is not None and hasattr(pkg, n):
            delattr(pkg, n)


def _large_ctx_fixture(monkeypatch, src, shared=True, extra=None, host_qr_wgs=64):
    """A large plan whose real _context runs against fakes (driver library, build, windows, resources)."""
    from aiter.gemm_fusion import catalog
    from aiter.gemm_fusion.match import MatchCtx as MC

    monkeypatch.setattr(catalog, "_CATALOG", {})
    _forget_sp_modules()
    import aiter.ops.gemm_fusion.sp_qkv.variants_large as vl
    from aiter.ops.gemm_fusion.sp_qkv import large_rt as lrt

    params = {**specs.LARGE_CELLS[("dense", 16384)][1], "src": src, **(extra or {})}
    plan = vl.SpQkvLarge(
        specs.sp_qkv_spec("dense").prefix(1),
        params,
        MC("gfx950", 256, 4, 4096, frozenset({"mori_cco", "sdma"})),
    )

    class Lib(_FakeCDLL):
        def __getattr__(self, name):
            f = super().__getattr__(name)
            return f

    lib = Lib()
    expect = {
        "lg_window_bytes_v1": vl.large_window_bytes("dense", 4096),
        "lg_attach_v1": 0x55,
        "lg_comm_R_ptr": 0x1000,
    }

    def info_call(b):
        for i, v in enumerate((plan.F, plan.n, 0, 0, 0, 0)[:6]):
            b[i] = v
        return 0

    def abi_call(b):
        for i, v in enumerate((4, 352, 0, 8, 64, 3072, 3104, 32, host_qr_wgs)):
            b[i] = v
        return 0

    real_getattr = Lib.__getattr__

    def getattr_(self, name):
        f = real_getattr(self, name)
        if name == "lg_info_v1":
            return _CallableFn(
                types.SimpleNamespace(),
                lambda b: (self.calls.append((name, ())), info_call(b))[1],
            )
        if name == "lg_abi_info_v2":
            return _CallableFn(
                types.SimpleNamespace(),
                lambda b: (self.calls.append((name, ())), abi_call(b))[1],
            )
        if name in expect:
            return _CallableFn(
                types.SimpleNamespace(),
                lambda *a, _n=name: (self.calls.append((_n, a)), expect[_n])[1],
            )
        return f

    Lib.__getattr__ = getattr_
    monkeypatch.setattr(
        vl,
        "build_large",
        lambda p, m: types.SimpleNamespace(
            hsaco="h.hsaco", hsaco_sha256="s" * 64, driver_so="d.so"
        ),
    )
    monkeypatch.setattr(vl, "_sha", lambda path: "s" * 64)
    monkeypatch.setattr(vl, "preflight_attach", lambda *a, **k: None)
    seen = {}
    monkeypatch.setattr(
        lrt, "load_driver", lambda path, abi=2: (seen.update(abi=abi), lib)[1]
    )
    W = lambda h, loc, **kw: types.SimpleNamespace(handle=h, local=loc, **kw)
    wins = {
        plan.owner + "/S": W(1, 0x10),
        plan.owner + "/R": W(2, 0x20),
        plan.owner + "/C": W(3, 0x30, shared=shared),
    }
    return plan, vl, lib, _FakeRes(wins), seen


def test_large_r12_context_attaches_the_page_checks_the_abi_and_registers_no_poison(
    monkeypatch,
):
    plan, _vl, lib, res, seen = _large_ctx_fixture(monkeypatch, "r12")
    assert plan.abi == 3 and [r.name.rsplit("/", 1)[-1] for r in plan.resources()] == [
        "S",
        "R",
        "C",
        "q",
        "gpage",
    ]
    plan._context(res)
    assert seen["abi"] == 3
    names = [c[0] for c in lib.calls]
    assert (
        names.index("lg_abi_info_v2")
        < names.index("lg_attach_v1")
        < names.index("lg_gpage_attach_v1")
    )  # ABI check, attach, then the page
    gp = next(c for c in lib.calls if c[0] == "lg_gpage_attach_v1")
    assert [x.value for x in gp[1][1:]] == [
        0x7000,
        0x7008,
        0x9000,
    ]  # err, step, the peers table
    hooks = [c[0] for c in res.calls if c[0].startswith("set_")]
    assert (
        "set_poison" not in hooks
        and "set_report_hook" in hooks
        and "set_drain" in hooks
        or "set_shared_hooks" in hooks
    )
    sh = next(c for c in res.calls if c[0] == "set_shared_hooks")
    assert sh[2]["poison"] is None  # the shared control block registers init/drain only
    # r11 selectable for the A/B: no page, the poison reader is registered
    plan11, _vl11, lib11, res11, seen11 = _large_ctx_fixture(monkeypatch, "r11")
    assert plan11.abi == 2 and [
        r.name.rsplit("/", 1)[-1] for r in plan11.resources()
    ] == ["S", "R", "C", "q"]
    plan11._context(res11)
    assert seen11["abi"] == 2 and not any(
        c[0] in ("lg_abi_info_v2", "lg_gpage_attach_v1") for c in lib11.calls
    )
    sh11 = next(c for c in res11.calls if c[0] == "set_shared_hooks")
    assert sh11[2]["poison"] is not None


def test_large_r12_abi_mismatch_is_reported_before_attach(monkeypatch):
    plan, _vl, lib, res, _seen = _large_ctx_fixture(monkeypatch, "r12")
    from aiter.ops.gemm_fusion.sp_qkv import large_rt as lrt

    monkeypatch.setattr(
        lrt,
        "abi_info",
        lambda lib_: {
            "abi": 3,
            "args_bytes": 352,
            "err_off": 0,
            "step_off": 8,
            "diag_off": 64,
            "ctag_off": 3072,
            "atag_off": 3104,
            "peers_bytes": 32,
            "qr_wgs": 64,
        },
    )
    with pytest.raises(
        RuntimeError, match=r"large driver ABI mismatch.*'abi': \(3, 4\)"
    ):
        plan._context(res)
    assert not any(c[0] in ("lg_attach_v1", "lg_gpage_attach_v1") for c in lib.calls)


def test_large_r12_qr_wgs_mismatch_between_host_library_and_device_image_is_refused_at_attach(
    monkeypatch,
):
    # the device image is built with the plan's qr_wgs (default 64); the host library reports its own through lg_abi_info_v2 out[8]
    for extra, host, image in (
        (None, 64, 64),
        ({"qr_wgs": 1}, 1, 1),
        (None, 1, 64),
        ({"qr_wgs": 1}, 64, 1),
    ):
        plan, _vl, lib, res, _seen = _large_ctx_fixture(
            monkeypatch, "r12", extra=extra, host_qr_wgs=host
        )
        assert plan.qr_wgs == image
        if host == image:
            plan._context(res)  # attaches
            assert any(c[0] == "lg_gpage_attach_v1" for c in lib.calls)
        else:
            with pytest.raises(
                RuntimeError,
                match=rf"starts {host} quiet_ready workgroups but the device image was built with LG_QR_WGS={image}",
            ):
                plan._context(res)
            # refused BEFORE any attach: nothing to quarantine, nothing launched
            assert not any(
                c[0] in ("lg_attach_v1", "lg_gpage_attach_v1") for c in lib.calls
            )
    # r11 has no knob and no ABI-3 check
    plan11, *_ = _large_ctx_fixture(monkeypatch, "r11")
    assert plan11.qr_wgs is None


def test_large_r12_run_needs_call_id_and_calls_launch_v3_r11_ignores_it(monkeypatch):
    from aiter.gemm_fusion import ContractError

    plan, st, res, launches = _large_ready(
        monkeypatch
    )  # the profile's large cells: src r12
    assert plan.abi == 3
    with pytest.raises(ContractError, match="needs run\\(call_id=...\\)"):
        plan.run(x=st["x"], x_scale=st["xs"], w=st["w"], w_scale=st["ws"], res=res)
    assert launches == []
    _run_large(plan, st, res, call_id=31337)
    assert (
        launches[-1][-2] == 31337 and len(launches) == 1
    )  # (…, half_first, call_id, stream)
    with pytest.raises(RuntimeError, match="no per-plan poison word"):
        plan.poison(res)


def test_profile_selects_r12_for_large_rows_and_rules_and_budgets_the_page():
    clear_profile_cache()
    prof = profmod.load_profile(PROFILE)
    large = [e for e in prof.entries() if e.vid == "sp_qkv.large.v1"]
    assert len(large) == 87 + 9 and {dict(e.params)["src"] for e in large} == {
        "r12"
    }  # 87 exact rows + the 9 fallback rules
    small = [e for e in prof.rows() if e.vid == "sp_qkv.small.v1"]
    assert len(small) == 24 and {dict(e.params)["src"] for e in small} == {
        "r19",
        "r20",
    }  # r19 for local m <= 32, r20 (MP64) at m=64; r17/r18 stay selectable per row
    wb = prof.meta["window_budget"]
    assert wb["shared"]["gf.global"] == 4 << 20 and wb["max_windows"] == 46
    assert (
        specs.large_cell_params("dense", 8192)
        == {**specs.LARGE_CELLS[("dense", 8192)][1], "src": "r12"}
        and specs.LARGE_PROFILE_SRC == "r12"
    )
    # the arena is the sum of the shared set and the small windows: 384 MiB (336 shared + 72 small)
    assert sum(wb["shared"].values()) + sum(wb["small_window_sizes"]) == 408 << 20


def test_prebuild_large_units_follow_the_profile_source(monkeypatch):
    import shutil
    import tempfile
    import uuid

    from aiter.ops.gemm_fusion.sp_qkv import prebuild as pb

    root = os.path.join(tempfile.gettempdir(), "jitroot_" + uuid.uuid4().hex)
    os.makedirs(root)
    monkeypatch.setenv("AITER_JIT_DIR", root)
    try:
        u12 = pb.large_unit(
            {"n": 2, "half_first": 0, "src": "r12"}, "dense", TOOL_SIM, "aot1"
        )
        u11 = pb.large_unit(
            {"n": 2, "half_first": 0, "src": "r11"}, "dense", TOOL_SIM, "aot1"
        )
        udef = pb.large_unit(
            {"n": 2, "half_first": 0}, "dense", TOOL_SIM, "aot1"
        )  # an old profile row without src: r11
        assert u12.meta["src"] == "r12" and u12.key != u11.key and u11.key == udef.key
        sink = []
        for t in u12.tasks:
            if t.name == "lg_gemm":
                os.makedirs(os.path.join(root, "t", "img"), exist_ok=True)
                with contextlib.suppress(
                    Exception
                ):  # the fake runner writes no files, so the unbundle step fails: the compile command is what we look at
                    t.run(
                        os.path.join(root, "t"),
                        lambda argv: sink.append(list(argv)) or "",
                    )
        assert sink and any(
            "/large/r12/device" in a.replace("\\", "/") for a in sink[0]
        )
        from aiter.gemm_fusion import catalog as _cat
        from aiter.gemm_fusion import prebuild as engine

        monkeypatch.setattr(_cat, "_CATALOG", {})
        _forget_sp_modules()  # a fresh import registers the variants (and their prebuild hooks) into this scratch catalog
        units, skipped = engine.plan_profile(PROFILE, recipe="aot1", tool=TOOL_SIM)
        assert (
            skipped == [] and len(units) == 12
        )  # 6 small (r19) + 3 small (r20) + 2 large (r12) + the comm
        assert {u.meta["src"] for u in units if u.kind == "sp_qkv.large.v1"} == {"r12"}
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ======================================================================= small/r19 + large/r12 bound to the delivered host APIs
def _c_signature(path, fn):
    """Argument KINDS of the C definition `int32_t|void* <fn>(...) {` in a host source: 'u32' / 'u64' / 'i32' / 'ptr' (any pointer)."""
    import re

    text = _slurp(path, encoding="utf-8")
    m = re.search(
        rf"(?:int32_t|uint64_t|void\*)\s+{fn}\s*\(", text
    )  # the DEFINITION, not a mention in a comment
    assert m, f"{fn} not defined in {path}"
    depth, i = 1, m.end()
    while depth:
        depth += {"(": 1, ")": -1}.get(text[i], 0)
        i += 1
    params = [
        p.strip()
        for p in text[m.end() : i - 1].replace("\n", " ").split(",")
        if p.strip()
    ]
    kinds = []
    for p in params:
        if "*" in p:
            kinds.append("ptr")
        elif "uint32_t" in p:
            kinds.append("u32")
        elif "uint64_t" in p:
            kinds.append("u64")
        elif "int32_t" in p:
            kinds.append("i32")
        else:
            raise AssertionError(f"unrecognised C parameter {p!r} of {fn}")
    return kinds


def _py_kinds(argtypes):
    import ctypes as C

    return [
        {C.c_uint32: "u32", C.c_uint64: "u64", C.c_int32: "i32"}.get(t, "ptr")
        for t in argtypes
    ]


def test_every_r12_binding_matches_the_c_declaration_it_binds(monkeypatch):
    """The stub-vs-header drift guard: the argument kinds of each R12 entry point in small_rt / large_rt equal the C definition in the delivered host sources."""
    from aiter.ops.gemm_fusion.sp_qkv import large_rt as lrt
    from aiter.ops.gemm_fusion.sp_qkv import small_rt as srt

    host = os.path.join(jit.CSRC, "host")
    sm_src, lg_src = os.path.join(host, "sm_fusion_driver.cpp"), os.path.join(
        host, "lg_fusion_driver.cpp"
    )
    fake_s, fake_l = _FakeCDLL(), _FakeCDLL()
    monkeypatch.setattr(srt.C, "CDLL", lambda path: fake_s)
    srt.load_driver("x", abi=3)
    monkeypatch.setattr(lrt.C, "CDLL", lambda path: fake_l)
    lrt.load_driver("x", abi=3)
    for fn in (
        "sm_attach_v3",
        "sm_launch_blk_v3",
        "sm_bind_new_v1",
        "sm_blk_free_v1",
        "sm_free_v1",
        "sm_info_v1",
    ):
        assert _py_kinds(fake_s._fn[fn].argtypes) == _c_signature(sm_src, fn), fn
    for fn in (
        "lg_attach_v1",
        "lg_gpage_attach_v1",
        "lg_launch_v2",
        "lg_abi_info_v2",
        "lg_report_v1",
        "lg_drain_v1",
        "lg_init_control_v1",
        "lg_free_v1",
        "lg_info_v1",
        "lg_window_bytes_v1",
    ):
        assert _py_kinds(fake_l._fn[fn].argtypes) == _c_signature(lg_src, fn), fn
    # and the 64 B kernarg block the r19 header declares is the one small_rt mirrors
    h = _slurp(
        os.path.join(jit.CSRC, "small", "r19", "interface", "sm_abi_v3.h"),
        encoding="utf-8",
    )
    import re

    block = re.search(r"struct SmRanArgsV3\s*\{(.*?)\};", h, re.DOTALL).group(1)
    names = re.findall(r"(\w+)\s*;", re.sub(r"//.*", "", block).replace(",", ";"))
    from aiter.ops.gemm_fusion.sp_qkv import small_rt

    assert [n for n, _t in small_rt.SmRanArgsV3._fields_] == [
        n for n in names if n != "pad"
    ], (names, small_rt.SmRanArgsV3._fields_)


def test_src_sha_constants_are_the_manifest_hashes():
    import hashlib

    mod = sys.modules["aiter.ops.gemm_fusion.sp_qkv.variants"]
    for src in jit.SRCS:
        path = os.path.join(jit.CSRC, "small", src, "SOURCE_SHA256.txt")
        assert hashlib.sha256(_slurp(path, "rb")).hexdigest() == mod.SRC_SHA[src], src


def test_profile_small_rows_run_r19_with_the_r18_family_flags_and_one_unit_each():
    from aiter.gemm_fusion import prebuild as engine

    clear_profile_cache()
    prof = profmod.load_profile(PROFILE)
    small = [r for r in prof.rows() if r.vid == "sp_qkv.small.v1"]
    assert len(small) == 24
    for (mode, T), (_vid, legacy) in specs.SMALL_CELLS.items():
        row_params = dict(
            prof.lookup(
                specs.sp_qkv_spec(mode).signature(), "gfx950", 256, 4, T // 4
            ).params
        )
        d19, _h19 = jit.small_defs(row_params, mode)
        d_old, _h_old = jit.small_defs(legacy, mode)
        # r19 = r18 + R12: the r17 cells keep every flag they had; the only additions are the r18-family switches at their off values
        extra = set(d19) - set(d_old)
        assert set(d_old) <= set(d19) and extra <= {
            "-DSM_B_ROWSPLIT=0",
            "-DSM_RX_BYPASS=0",
            "-DSM_REUSE_NO_IDX=0",
        }, (mode, T, extra)
        if legacy.get("src") == "r18":
            assert (
                extra == set() and d19 == d_old
            )  # the r18 cells are byte-for-byte the same flag set
    tool = dict(TOOL_SIM)
    units, skipped = engine.plan_profile(PROFILE, recipe="aot1", tool=tool)
    smalls = [u for u in units if u.kind == "sp_qkv.small.v1"]
    assert (
        skipped == []
        and len(smalls)
        == 9  # 6 r19 owners (the P2 rows add none) + 3 r20 (MP64) owners
        and {u.meta["src"] for u in smalls} == {"r19", "r20"}
        and len({u.key for u in smalls}) == 9
    )
    u = next(x for x in smalls if x.meta["src"] == "r19")
    gemm_cmd = jit.device_cmd_one("r19", u.meta["dev"], tool, "/out", "sm_gemm")
    fwd = [a.replace("\\", "/") for a in gemm_cmd]
    assert any(a.endswith("small/r19/device/sk_gemm_fp8.cu") for a in fwd) and any(
        "-I" in a and "small/r19/device" in a for a in fwd
    )
    host = [
        a.replace("\\", "/")
        for a in jit.host_compile_cmd("r19", u.meta["host"], tool, "/out")
    ]
    assert any(
        a.endswith(("-Ismall/r19/host", "small/r19/host")) for a in host
    ) and any(a.endswith("host/sm_fusion_driver.cpp") for a in host)


# ======================================================================= small/r20 (R13 MP64): src r20 owner / variant, geometry, profile coverage
R20_PARAMS = {"src": "r20", "sk": 3, "nh": 1, "no_credit": 1, "rx_bypass": 1}


def test_r20_flag_validation_mp_table_and_the_removed_shapes():
    assert jit.SMALL_MP == {"r17": 32, "r18": 32, "r19": 32, "r20": 64, "r21": 128}
    assert [jit.src_mp(s) for s in jit.SRCS] == [32, 32, 32, 64, 128]
    with pytest.raises(ValueError, match="not in"):
        jit.src_mp("r22")
    # r20 builds with the profile's params: the bypass loads are REQUIRED, B_HEADS=1, no credit slow path
    dev, host = jit.small_defs(dict(R20_PARAMS), "dense")
    assert (
        "-DSM_RX_BYPASS=1" in dev
        and "-DSM_T1_B_HEADS=1" in dev
        and "-DSM_NO_CREDIT=1" in dev
        and "-DSM_PUSH8=0" in dev
    )
    assert (
        "-DSM_B_ROWSPLIT=0" in host and "-DSM_REUSE_NO_IDX=0" in host
    )  # the removed shapes are passed as 0: the header only #errors when they are non-zero
    for bad, match in (
        ({"push8": 1}, r"r20 has no \['push8'\]"),
        ({"push_halves": 1}, r"r20 has no \['push_halves'\]"),
        ({"b_rowsplit": 1}, r"r20 has no \['b_rowsplit'\]"),
        ({"rx_bypass": 0}, "requires rx_bypass=1"),
        ({"nh": 2}, r"r20 requires nh in \(1,\)"),
        ({"no_credit": 0}, "SM_NO_CREDIT must be 1"),
    ):
        with pytest.raises(ValueError, match=match):
            jit.small_defs({**R20_PARAMS, **bad}, "dense")
    no_bypass = {k: v for k, v in R20_PARAMS.items() if k != "rx_bypass"}
    with pytest.raises(ValueError, match="requires rx_bypass=1"):
        jit.small_defs(no_bypass, "reuse")  # every mode, not only dense
    with pytest.raises(ValueError, match="SM_NO_CREDIT must be 1"):
        jit.small_defs(
            {"src": "r19", "no_credit": 0}, "dense"
        )  # r19 refuses the credit slow path too
    # reuse_no_idx stays legal for reuse (an r20 switch), the profile does not set it (untuned)
    jit.small_defs({**R20_PARAMS, "reuse_no_idx": 1}, "reuse")


def test_r20_window_and_p_geometry_match_the_r20_notes():
    from aiter.ops.gemm_fusion.sp_qkv import variants as v

    assert v.small_window_bytes("dense", 64)["rx"] == 2_359_296
    assert (
        v.small_window_bytes("indexer", 64)["rx"]
        == v.small_window_bytes("reuse", 64)["rx"]
        == 2_621_440
    )
    assert all(
        v.small_window_bytes(m, 64)["rx"] <= 4 << 20 for m in specs.MODES
    )  # the 4 MiB arena view
    assert (
        v.small_p_bytes("dense", 64) == 7_077_888
        and v.small_p_bytes("indexer", 64) == v.small_p_bytes("reuse", 64) == 7_569_408
    )
    # MP32 (r17-r19) is what it was: Rx [2][4][32][F] bf16, P [3][32][N] fp32
    assert (
        v.small_window_bytes("dense")["rx"]
        == v.small_window_bytes("dense", 32)["rx"]
        == 1_179_648
    )
    assert v.small_p_bytes("dense", 32) == 3 * 32 * 9216 * 4
    assert (
        v.small_window_bytes("dense", 64)["ctl"]
        == v.small_window_bytes("dense", 32)["ctl"]
        == 256 + 8 * 2 * 4 * 72
    )  # the control window does not grow with MP


def test_r20_plan_m_range_owner_window_request_and_run_path(monkeypatch):
    from aiter.gemm_fusion import ContractError

    plan, st, meta, outs, res, launches, _binds = _small_ready(
        monkeypatch, "dense", 256, "r20", {"rx_bypass": 1}
    )
    assert (plan.src, plan.abi, plan.mp, plan.M if hasattr(plan, "M") else 64) == (
        "r20",
        3,
        64,
        64,
    )
    reqs = plan.resources()
    assert [r.name.rsplit("/", 1)[-1] for r in reqs] == ["rx", "ctl", "gpage"] and reqs[
        0
    ].bytes == reqs[
        1
    ].bytes == 4 << 20  # 2.36 MiB Rx rounds up to ONE 4 MiB view
    # M = 64 (T = 256, the default capture bucket) runs: the per-launch kernargs carry t_valid = T and the call_id
    pos, slot = _NoRead((256,), (1,), "i64"), _NoRead((256,), (1,), "i64")
    _run_small(
        plan,
        st,
        {**meta, "positions": pos, "slot_mapping": slot},
        outs,
        res,
        call_id=99,
    )
    tag, _blk, _a, _w, stride_a, _xs, _pos, _slot, t_valid, call_id, _stream = launches[
        -1
    ]
    assert (tag, t_valid, call_id, stride_a) == ("v3", 256, 99, 6144)
    # a short positions array is legal at M=64 as well (the kernel pads rows >= t_valid, dead passes included)
    _run_small(
        plan, st, {**meta, "positions": _T((140,), (1,), "i64")}, outs, res, call_id=99
    )
    assert launches[-1][8] == 140
    # 65 rows: refused by the plan (a src r20 plan ends at M = 64); the constructor refuses it too
    with pytest.raises(
        ContractError,
        match=r"x has 65 rows, a src r20 small plan takes local M 1\.\.64",
    ):
        _run_small(
            plan, st, meta, outs, res, x=_T((65, 6144), (6144, 1), "fp8"), call_id=1
        )
    v = sys.modules["aiter.ops.gemm_fusion.sp_qkv.variants"]
    mk = lambda src, m, extra=None: v.SpQkvSmall(
        specs.sp_qkv_spec("dense"),
        {**specs.SMALL_CELLS[("dense", 64)][1], "src": src, **(extra or {})},
        MatchCtx("gfx950", 256, 4, m, frozenset({"mori_cco", "gfx950_sc_bypass"})),
    )
    assert (
        mk("r20", 33, {"rx_bypass": 1}).mp == 64
        and mk("r20", 64, {"rx_bypass": 1}).mp == 64
    )
    with pytest.raises(
        ContractError, match=r"\(src r20\) supports local M 1\.\.64, got 65"
    ):
        mk("r20", 65, {"rx_bypass": 1})
    with pytest.raises(
        ContractError, match=r"\(src r19\) supports local M 1\.\.32, got 33"
    ):
        mk("r19", 33)
    # r20 without the gfx950 bypass capability is refused (every mode needs it)
    with pytest.raises(ContractError, match="gfx950_sc_bypass"):
        v.SpQkvSmall(
            specs.sp_qkv_spec("dense"),
            dict(R20_PARAMS),
            MatchCtx("gfx950", 256, 4, 40, frozenset({"mori_cco"})),
        )


def test_r20_owner_is_one_per_mode_and_the_p2_rows_share_the_m16_owner(monkeypatch):
    from aiter.gemm_fusion import catalog

    monkeypatch.setattr(catalog, "_CATALOG", {})
    owners = {}
    for mode, m, _vid, params in specs.small_exact_cells():
        p = _plan_for(mode, 4 * m, False)
        owners.setdefault(mode, {})[m] = p.build_key
        assert (
            p.src == params["src"]
            and p.mp == (64 if params["src"] == "r20" else 32)
            and p.abi == 3
        )
    for mode in specs.MODES:
        o = owners[mode]
        assert {o[m] for m in (1, 2, 4, 8, 12, 16)} == {
            o[16]
        }  # P2 = copies of the m16 params: ONE owner, the windows / context / epoch domain of m16
        assert o[32] != o[16] and o[64] not in (
            o[16],
            o[32],
        )  # the tuned m32 cell and the r20 owner are their own
    assert (
        len({k for o in owners.values() for k in o.values()}) == 9
    )  # 3 modes x (m16 group, m32, r20)


def test_window_budget_counts_the_nine_small_owners_and_the_mp64_geometry():
    clear_profile_cache()
    wb = profmod.load_profile(PROFILE).meta["window_budget"]
    assert wb["small_owners"] == 9 and wb["large_owners"] == 9
    assert (
        wb["small_window_sizes"] == [4 << 20] * 18
    )  # 9 owners x (rx + ctl), each a 4 MiB view (r20's 2.36 / 2.62 MB Rx fits)
    assert wb["max_windows"] == 2 * 9 + 3 * 9 + 1 == 46
    assert sum(wb["shared"].values()) + sum(wb["small_window_sizes"]) == 408 << 20
    with open(profmod._resolve(PROFILE), encoding="utf-8") as f:
        doc = json.load(f)
    assert (
        doc["native_m_ranges"] == [dict(r) for r in specs.NATIVE_M_RANGES]
        and doc["native_m_ranges"][0]["m_min"] == 65
        and doc["native_m_ranges"][0]["m_max"] == 255
    )


# ======================================================================= small/r21 (R13 MP128): src r21 owner / variant, 8 MiB Rx view, B_HEADS A/B
R21_PARAMS = {"src": "r21", "sk": 3, "nh": 1, "no_credit": 1, "rx_bypass": 1}


def test_r21_flag_validation_and_the_b_heads_ab():
    assert jit.src_mp("r21") == 128 and jit.src_abi("small", "r21") == 3
    dev, _host = jit.small_defs(dict(R21_PARAMS), "indexer")
    assert (
        "-DSM_RX_BYPASS=1" in dev
        and "-DSM_T1_B_HEADS=1" in dev
        and "-DSM_NO_CREDIT=1" in dev
    )
    # the A/B for the occupancy caveat: B_HEADS=2 is selectable for r21 (216 / 240 B WGs), not for r20
    dev2, _host2 = jit.small_defs({**R21_PARAMS, "nh": 2}, "dense")
    assert "-DSM_T1_B_HEADS=2" in dev2 and "-DSM_T1_B_HEADS=1" not in dev2
    for bad, match in (
        ({"nh": 3}, r"r21 requires nh in \(1, 2\) \(B_HEADS\), got 3"),
        ({"nh": 4}, r"r21 requires nh in \(1, 2\)"),
        ({"push8": 1}, r"r21 has no \['push8'\]"),
        ({"push_halves": 1}, r"r21 has no \['push_halves'\]"),
        ({"b_rowsplit": 1}, r"r21 has no \['b_rowsplit'\]"),
        ({"rx_bypass": 0}, "r21 requires rx_bypass=1"),
        ({"no_credit": 0}, "SM_NO_CREDIT must be 1"),
    ):
        with pytest.raises(ValueError, match=match):
            jit.small_defs({**R21_PARAMS, **bad}, "dense")
    with pytest.raises(ValueError, match=r"r20 requires nh in \(1,\)"):
        jit.small_defs({**R20_PARAMS, "nh": 2}, "dense")  # r20 keeps NH1 only
    # the profile's params for r21 are the NH1 default; nh=2 builds are a DIFFERENT unit key (an A/B is a different owner too)
    assert (
        specs.small_r21_params("dense") == R21_PARAMS
        and specs.small_r21_params("dense", 2)["nh"] == 2
    )
    with pytest.raises(ValueError, match="nh in"):
        specs.small_r21_params("dense", 3)
    assert jit.small_defs(
        specs.small_r21_params("reuse", 2), "reuse"
    ) != jit.small_defs(specs.small_r21_params("reuse", 1), "reuse")


def test_r21_window_and_p_geometry_and_the_8mib_rx_view():
    from aiter.ops.gemm_fusion.sp_qkv import variants as v

    assert v.small_window_bytes("dense", 128)["rx"] == 4_718_592
    assert (
        v.small_window_bytes("indexer", 128)["rx"]
        == v.small_window_bytes("reuse", 128)["rx"]
        == 5_242_880
    )
    assert (
        v.small_p_bytes("dense", 128) == 14_155_776
        and v.small_p_bytes("indexer", 128)
        == v.small_p_bytes("reuse", 128)
        == 15_138_816
    )
    for mode in specs.MODES:
        rx, ctl = v.small_window_caps(mode, 128)
        assert (rx, ctl) == (
            8 << 20,
            4 << 20,
        )  # the Rx view is 8 MiB (4 MiB < Rx <= 8 MiB); the control window stays ONE granule (not 8 MiB)
        assert v.small_window_caps(mode, 64) == (
            4 << 20,
            4 << 20,
        ) and v.small_window_caps(mode, 32) == (
            4 << 20,
            4 << 20,
        )  # MP <= 64: unchanged
        assert (
            v.small_window_bytes(mode, 128)["rx"] > 4 << 20
            and v.small_window_bytes(mode, 128)["rx"] <= 8 << 20
        )


def test_r21_plan_m_range_window_request_and_run_path(monkeypatch):
    from aiter.gemm_fusion import ContractError

    plan, st, meta, outs, res, launches, _binds = _small_ready(
        monkeypatch, "dense", 512, "r21", {"rx_bypass": 1}
    )
    assert (plan.src, plan.abi, plan.mp) == ("r21", 3, 128)
    reqs = plan.resources()
    assert [r.name.rsplit("/", 1)[-1] for r in reqs] == ["rx", "ctl", "gpage"]
    assert (
        reqs[0].bytes == 8 << 20 and reqs[1].bytes == 4 << 20
    )  # Rx: the 8 MiB view; control: one granule
    pos, slot = _NoRead((512,), (1,), "i64"), _NoRead((512,), (1,), "i64")
    _run_small(
        plan, st, {**meta, "positions": pos, "slot_mapping": slot}, outs, res, call_id=7
    )
    assert (
        launches[-1][0] == "v3" and launches[-1][8] == 512 and launches[-1][9] == 7
    )  # t_valid = T = 512 at M = 128
    with pytest.raises(
        ContractError,
        match=r"x has 129 rows, a src r21 small plan takes local M 1\.\.128",
    ):
        _run_small(
            plan, st, meta, outs, res, x=_T((129, 6144), (6144, 1), "fp8"), call_id=1
        )
    v = sys.modules["aiter.ops.gemm_fusion.sp_qkv.variants"]
    caps = frozenset({"mori_cco", "gfx950_sc_bypass"})
    mk = lambda m, extra=None: v.SpQkvSmall(
        specs.sp_qkv_spec("dense"),
        {**R21_PARAMS, **(extra or {})},
        MatchCtx("gfx950", 256, 4, m, caps),
    )
    assert (
        mk(65).mp == 128 and mk(128).mp == 128 and mk(128, {"nh": 2}).params["nh"] == 2
    )
    with pytest.raises(
        ContractError, match=r"\(src r21\) supports local M 1\.\.128, got 129"
    ):
        mk(129)
    assert (
        plan.owner != mk(65, {"nh": 2}).owner
    )  # the B_HEADS A/B is its own owner (windows / context): never mixed with NH1


# ======================================================================= r12b / r19-r20 "overhead round" knobs: defaults, explicit passing, manifests
def _header_defaults(text):
    """`#ifndef NAME` / `#define NAME <int>` default pairs of a C header or source."""
    import re

    return {
        n: int(v) for n, v in re.findall(r"#ifndef (\w+)\s*\n\s*#define \1 (\d+)", text)
    }


def test_large_r12_knobs_are_explicit_validated_and_equal_the_header_and_build_device_defaults():
    import re

    tool = dict(TOOL_SIM)
    d12, d11 = jit.large_defs({"src": "r12"}, "dense"), jit.large_defs(
        {"src": "r11"}, "dense"
    )
    assert d12 == d11 + [
        "-DLG_TAG_EARLY=1",
        "-DLG_POLL_BATCH=1",
        "-DLG_QR_WGS=64",
    ] and not any(
        "TAG_EARLY" in d or "POLL_BATCH" in d or "QR_WGS" in d for d in d11
    )  # r11 has no such knobs
    # the A/B: another value is another BUILD (a different key), and bad values / r11 knobs are refused before a compile
    ab = jit.large_defs({"src": "r12", "tag_early": 0, "poll_batch": 0}, "dense")
    assert "-DLG_TAG_EARLY=0" in ab and "-DLG_POLL_BATCH=0" in ab
    assert jit.large_cache_key(ab, tool, "r12") != jit.large_cache_key(d12, tool, "r12")
    assert "-DLG_POLL_BATCH=2" in jit.large_defs(
        {"src": "r12", "poll_batch": 2}, "indexer"
    )
    # qr_wgs (r12c: the NaN fill folded into quiet_ready; 64 = default, 1 = the single-WG A/B): validated, a different build key, and the HOST build gets it too
    one = jit.large_defs({"src": "r12", "qr_wgs": 1}, "dense")
    assert "-DLG_QR_WGS=1" in one and "-DLG_QR_WGS=64" not in one
    assert jit.large_cache_key(one, tool, "r12") != jit.large_cache_key(
        d12, tool, "r12"
    )
    host = lambda defs: [
        a
        for a in jit.large_host_cmd(defs, tool, "/out", "r12")
        if a.startswith("-DLG_")
    ]
    assert "-DLG_QR_WGS=64" in host(d12) and "-DLG_QR_WGS=1" in host(
        one
    )  # the host launches quiet_ready with LG_QR_WGS WGs: device == host
    assert "-DLG_QR_WGS=1" in [
        a for a in jit.large_device_cmd(one, tool, "/out", "r12") if "QR_WGS" in a
    ]
    for bad in (0, 2, 63, 128):
        with pytest.raises(ValueError, match="qr_wgs must be 1 or 64"):
            jit.large_defs({"src": "r12", "qr_wgs": bad}, "dense")
    for bad, match in (
        ({"tag_early": 2}, "tag_early must be 0 or 1"),
        ({"poll_batch": 3}, "poll_batch must be 0, 1 or 2"),
    ):
        with pytest.raises(ValueError, match=match):
            jit.large_defs({"src": "r12", **bad}, "dense")
    for src in ("r11", None):
        params = {"tag_early": 0} if src is None else {"src": src, "poll_batch": 0}
        with pytest.raises(ValueError, match="exist only in large/r12"):
            jit.large_defs({**params, "qr_wgs": 1}, "dense")
        with pytest.raises(ValueError, match="exist only in large/r12"):
            jit.large_defs(
                params, "dense"
            )  # no src = r11 (the default of old profiles)
    # THE CHECK: what Python passes == the header default (the effective value when a knob is NOT passed) == the build_device.sh default
    root = os.path.join(jit.CSRC, "large", "r12")
    header = _header_defaults(
        _slurp(os.path.join(root, "interface", "lg_comm_abi_v3.h"), encoding="utf-8")
    )
    script = dict(
        re.findall(
            r'(LG_\w+)="\$\{\1:-(\d+)\}"',
            _slurp(os.path.join(root, "build_device.sh"), encoding="utf-8"),
        )
    )
    assert (
        header["LG_TAG_EARLY"] == int(script["LG_TAG_EARLY"]) == 1
        and header["LG_POLL_BATCH"] == int(script["LG_POLL_BATCH"]) == 1
        and header["LG_QR_WGS"] == int(script["LG_QR_WGS"]) == 64
    )
    passed = dict(f[2:].split("=") for f in d12)
    for name, val in passed.items():
        if (
            name in script and name != "LG_X_INDEXER"
        ):  # LG_X_INDEXER is the MODE (the script default is the indexer build)
            assert val == script[name], (name, val, script[name])
    for name, val in script.items():
        if (
            name not in passed and name in header
        ):  # a flag the script passes and Python does not: the header default applies and must equal the script's
            assert header[name] == int(val), (name, header[name], val)
    # every tag-test seam is OFF by default on both sides (Python never passes them)
    seams = [n for n in script if n.startswith("LG_X_TEST_")]
    assert seams and all(
        int(script[n]) == 0 and header[n] == 0 and n not in passed for n in seams
    )


def test_small_r19_r20_overhead_knobs_default_to_one_and_the_python_defs_do_not_pass_them():
    for src in ("r19", "r20"):
        text = _slurp(
            os.path.join(jit.CSRC, "small", src, "device", "ran_kernel.cu"),
            encoding="utf-8",
        )
        d = _header_defaults(text)
        assert (
            d["SM_TAG_EARLY"] == 1 and d["SM_POLL_BATCH"] == 1
        ), src  # the source default applies when the flag is not passed
        params = {"src": src, "sk": 3, "nh": 1, "no_credit": 1, "rx_bypass": 1}
        dev, host = jit.small_defs(params, "dense")
        assert not any("TAG_EARLY" in x or "POLL_BATCH" in x for x in dev + host), src
        # the knobs are not profile params: an unknown param is refused (no silent A/B through a row)
        with pytest.raises(ValueError, match="unknown small-M params"):
            jit.small_defs({**params, "tag_early": 0}, "dense")


def test_large_src_sha_constants_are_the_manifest_hashes():
    """LARGE_SRC_SHA is read from the SOURCE (ast), not by importing variants_large: a second import would register the variant into the live catalog twice."""
    import ast
    import hashlib

    path = os.path.join(os.path.dirname(jit.__file__), "variants_large.py")
    tree = ast.parse(_slurp(path, encoding="utf-8"))
    consts = [
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", "") == "LARGE_SRC_SHA" for t in n.targets)
    ]
    assert len(consts) == 1
    sha = consts[0]
    assert set(sha) == set(jit.LARGE_SRCS)
    for src in jit.LARGE_SRCS:
        manifest = os.path.join(jit.CSRC, "large", src, "SOURCE_SHA256.txt")
        assert hashlib.sha256(_slurp(manifest, "rb")).hexdigest() == sha[src], src


# ======================================================================= launch gate: Plan.run refuses to launch while reset() is in progress
def test_plan_run_raises_the_typed_gate_error_while_reset_is_in_progress_small_and_large(
    monkeypatch,
):
    from aiter.gemm_fusion import LaunchGateError, ResourceError

    plan, st, meta, outs, res, launches, binds = _small_ready(monkeypatch, src="r19")
    pos = _NoRead((64,), (1,), "i64")
    m2 = {**meta, "positions": pos, "slot_mapping": pos}
    res.launch_gate = True
    with pytest.raises(
        LaunchGateError,
        match=r"sp_qkv\.small/.*launch was attempted while reset\(\) is in progress",
    ):
        _run_small(plan, st, m2, outs, res, call_id=7)
    assert (
        launches == [] and binds == []
    )  # refused BEFORE anything was bound, validated or launched
    assert issubclass(LaunchGateError, ResourceError)
    res.launch_gate = False
    _run_small(plan, st, m2, outs, res, call_id=7)
    assert len(launches) == 1  # the gate released: the same call launches
    # a resources object without the attribute (an older FusionResources) is not gated
    del res.launch_gate
    _run_small(plan, st, m2, outs, res, call_id=7)
    assert len(launches) == 2
    # the abi-2 (r17/r18) small plans are gated too
    plan2, st2, meta2, outs2, res2, launches2, _b = _small_ready(monkeypatch)
    res2.launch_gate = True
    with pytest.raises(LaunchGateError):
        _run_small(plan2, st2, meta2, outs2, res2)
    assert launches2 == []
    # large
    lplan, lst, lres, llaunches = _large_ready(monkeypatch)
    lres.launch_gate = True
    with pytest.raises(
        LaunchGateError, match=r"sp_qkv\.large/.*while reset\(\) is in progress"
    ):
        _run_large(lplan, lst, lres)
    assert llaunches == []
    lres.launch_gate = False
    _run_large(lplan, lst, lres)
    assert len(llaunches) == 1
