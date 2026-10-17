# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tests for sp_qkv small-M integration (S2-2b-ii): profile, flags, JIT command plan, FusionResources."""

import json
import logging
import os
import sys
import types

import pytest

try:
    import aiter.gemm_fusion  # noqa: F401
except Exception:
    for _k in [k for k in sys.modules if k == "aiter" or k.startswith("aiter.")]:
        del sys.modules[_k]
    _pkg = types.ModuleType("aiter")
    _pkg.__path__ = [os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "aiter"))]
    sys.modules["aiter"] = _pkg

from aiter.gemm_fusion import ContractError, MatchCtx, match_prefix, profile as profmod
from aiter.gemm_fusion.plan import WindowReq
from aiter.gemm_fusion.resources import FusionResources, ResourceError, round_up
from aiter.ops.gemm_fusion.sp_qkv import gen_profile, jit, specs

PROFILE = os.path.join(os.path.dirname(__file__), "..", "aiter", "configs", "gemm_fusion", "minimax_m3_sp4_gfx950_v1.json")


def test_profile_file_matches_generator():
    with open(PROFILE, encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk == json.loads(json.dumps(gen_profile.build_profile())), "re-run op_tests/gen_gemm_fusion_profile.py"


def test_profile_rows_exact_and_distinct():
    profmod.clear_profile_cache()
    prof = profmod.load_profile(PROFILE)
    small = [r for r in prof.rows() if r.vid == "sp_qkv.small.v1"]
    assert len(small) == 6
    for (mode, T), (vid, params) in specs.SMALL_CELLS.items():
        r = prof.lookup(specs.sp_qkv_spec(mode).signature(), "gfx950", 256, 4, T // 4)
        assert r is not None and dict(r.params) == params and r.vid == vid
    # three modes -> three different signatures
    assert len({specs.sp_qkv_spec(m).signature() for m in specs.MODES}) == 3
    assert specs.sp_qkv_spec("dense").gemm.n == 9216 and specs.sp_qkv_spec("reuse").gemm.n == 9856
    assert specs.mode_of(specs.sp_qkv_spec("reuse")) == "reuse"


def test_field_map_matches_golden():
    f = {x.name: (x.offset, x.width, x.shards) for x in specs.sp_qkv_spec("indexer").stages[0].fields}
    assert f == {"q": (0, 8192, 4), "k": (8192, 512, 4), "v": (8704, 512, 4), "iq": (9216, 512, 4), "ik": (9728, 128, 1)}
    assert sum(w // s for _, w, s in f.values()) == 2560  # F indexer
    d = specs.sp_qkv_spec("dense").stages[0]
    assert sum(x.width // x.shards for x in d.fields) == 2304


def test_small_defs_per_cell():
    d, h = jit.small_defs(*[specs.SMALL_CELLS[("dense", 128)][1], "dense"])
    assert "-DSM_B_ROWSPLIT=1" in h and "-DSM_RX_BYPASS=1" in h and "-DSM_DENSE=1" in h and "-DSM_NT_STORES=0" in d
    d, h = jit.small_defs(specs.SMALL_CELLS[("indexer", 64)][1], "indexer")
    assert "-DSM_PUSH8=1" in h and "-DSM_R5X_INDEXER=1" in h and "-DSM_NO_CREDIT=1" in d
    d, h = jit.small_defs(specs.SMALL_CELLS[("reuse", 128)][1], "reuse")
    assert "-DSM_PUSH_HALVES=1" in h and not any("ROWSPLIT" in x for x in h)  # r17 has no r18 macros
    d, h = jit.small_defs(specs.SMALL_CELLS[("reuse", 64)][1], "reuse")
    assert "-DSM_REUSE_NO_IDX=1" in h and "-DSM_RX_BYPASS=0" in h
    with pytest.raises(ValueError):
        jit.small_defs({"src": "r17", "rx_bypass": 1}, "dense")
    with pytest.raises(ValueError):
        jit.small_defs({"nt": 1}, "dense")
    with pytest.raises(ValueError):
        jit.small_defs({"bogus": 1}, "dense")


def test_jit_command_plan_and_cache_key(tmp_path_factory=None):
    tool = {"hipcc": "/rocm/bin/hipcc", "clangxx": "/rocm/llvm/bin/clang++", "bundler": "b", "rocm": "/rocm",
            "mori_root": "/m", "mori_so": "/m/libmori_cco.so"}
    dev, host = jit.small_defs(specs.SMALL_CELLS[("dense", 64)][1], "dense")
    cmds = jit.device_cmds("r17", dev, tool, "/out")
    assert len(cmds) == 2 and all("--offload-arch=gfx950" in c and "-fno-fast-math" in c and "-D__HIPCC_RTC__" in c for c in cmds)
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
        n = 0
        for line in open(os.path.join(base, "SOURCE_SHA256.txt")):
            h, f = line.split(None, 1)
            f = f.strip()
            if f.startswith("tests/") or f == "host/bridge_sm.py":
                continue
            assert hashlib.sha256(open(os.path.join(base, f), "rb").read()).hexdigest() == h, f
            n += 1
        assert n >= 11


# ---- FusionResources with fakes -------------------------------------------------------------------------
class FakeDist:
    def __init__(self):
        self.calls = []

    def broadcast_object_list(self, obj, src=0, group=None):
        self.calls.append(("bcast", src))

    def barrier(self, group=None):
        self.calls.append(("barrier",))


class FakeBackend:
    def __init__(self):
        self.log = []
        self.n = 0

    def uid(self, rank):
        return b"u" * 128

    def comm_create(self, rank, world, uid, vmm):
        self.log.append(("comm", vmm))
        return 1

    def win_register(self, comm, nbytes):
        self.n += 1
        self.log.append(("win", nbytes))
        return 100 + self.n, 4096 * self.n

    def peer_ptr(self, comm, local, peer):
        return local + peer

    def devcomm_create(self, comm, q):
        self.log.append(("dev", q))
        return 7

    def zero(self, dev, local, n):
        self.log.append(("zero", local, n))

    def device_sync(self, dev):
        self.log.append(("sync",))

    def devcomm_destroy(self, comm, dev):
        self.log.append(("devdestroy",))

    def win_deregister(self, comm, w):
        self.log.append(("dereg", w))

    def comm_destroy(self, comm):
        self.log.append(("commdestroy",))


def test_resources_lifecycle_and_sizes():
    be, d = FakeBackend(), FakeDist()
    r = FusionResources(0, 4, dist=d, backend=be)
    reqs = [WindowReq("rx", 1_300_000, "p2p_window"), WindowReq("ctl", 5000, "control")]
    w = r.ensure("plan/a", reqs)
    assert w["rx"].bytes == w["ctl"].bytes == round_up(1_300_000) == 4 << 20  # equal, 4 MiB multiple
    assert r.ensure("plan/a", reqs) is w  # idempotent per owner
    assert sum(1 for x in be.log if x[0] == "win") == 2
    r.finalize()
    assert ("dev", 1) in be.log and r.finalized
    with pytest.raises(ResourceError):
        r.ensure("plan/b", reqs)  # ensure after finalize
    be.log.clear(); d.calls.clear()
    r.quiesce_and_reset()
    kinds = [x[0] for x in be.log]
    assert kinds == ["sync", "zero", "zero", "sync"]  # device sync -> barrier -> reset -> sync -> barrier
    assert d.calls == [("barrier",), ("barrier",)]
    assert r.peers(w["rx"]) == [w["rx"].local + p for p in range(4)]
    r.release()
    assert be.log[-1] == ("commdestroy",)
    r.release()  # idempotent


def test_small_variant_registers_and_contract(monkeypatch):
    from aiter.gemm_fusion import catalog

    monkeypatch.setattr(catalog, "_CATALOG", {})
    sys.modules.pop("aiter.ops.gemm_fusion.sp_qkv.variants", None)
    import aiter.ops.gemm_fusion.sp_qkv.variants  # noqa: F401  (fresh import registers into the patched catalog)

    profmod.clear_profile_cache()
    spec = specs.sp_qkv_spec("dense")
    ok = MatchCtx("gfx950", 256, 4, 32, frozenset({"mori_cco", "gfx950_sc_bypass"}))
    plan, n = match_prefix(spec, ok, PROFILE)
    assert n == 2 and plan.vid == "sp_qkv.small.v1" and plan.mode == "dense" and plan.params["rx_bypass"] == 1
    reqs = plan.resources()
    assert [r.kind for r in reqs] == ["p2p_window", "control"] and all(r.bytes % (4 << 20) == 0 for r in reqs)
    with pytest.raises(ContractError, match="missing"):
        match_prefix(spec, MatchCtx("gfx950", 256, 4, 32, frozenset()), PROFILE)
    with pytest.raises(ContractError, match="gfx950_sc_bypass"):
        match_prefix(spec, MatchCtx("gfx950", 256, 4, 32, frozenset({"mori_cco"})), PROFILE)
    # reuse T64 (BYPASS 0) must not need the bypass capability
    plan, n = match_prefix(specs.sp_qkv_spec("reuse"), MatchCtx("gfx950", 256, 4, 16, frozenset({"mori_cco"})), PROFILE)
    assert n == 2 and plan.params["reuse_no_idx"] == 1
    # not a validated cell -> native
    assert match_prefix(spec, MatchCtx("gfx950", 256, 4, 48, frozenset({"mori_cco"})), PROFILE) == (None, 0)
    # a spec with a different eps has a different signature -> native
    assert match_prefix(specs.sp_qkv_spec("dense", eps=1e-5), ok, PROFILE) == (None, 0)


# ======================================================================= large-M (S2-2b-iii)
def test_profile_has_15_rows_and_large_prefix_signatures():
    profmod.clear_profile_cache()
    prof = profmod.load_profile(PROFILE)
    assert len(prof) == 15
    for (mode, T), (vid, params) in specs.LARGE_CELLS.items():
        sig = specs.sp_qkv_spec(mode).prefix(1).signature()
        r = prof.lookup(sig, "gfx950", 256, 4, T // 4)
        assert r is not None and r.vid == "sp_qkv.large.v1" and dict(r.params) == params
    # reuse and dense large rows have different signatures (reuse drops iq/ik in the exchange)
    assert specs.sp_qkv_spec("dense").prefix(1).signature() != specs.sp_qkv_spec("reuse").prefix(1).signature()
    # 2-stage and 1-stage signatures differ
    assert specs.sp_qkv_spec("dense").signature() != specs.sp_qkv_spec("dense").prefix(1).signature()


def test_large_defs_and_commands():
    d = jit.large_defs({"n": 2, "half_first": 1, "no_gridread": 1}, "indexer")
    assert "-DLG_X_INDEXER=1" in d and "-DLG_NO_GRIDREAD=1" in d and "-DBUILD_CCO_SDMA=1" in d and "-DLG_R8_FOLD=0" in d
    assert "-DLG_X_INDEXER=0" in jit.large_defs({}, "reuse") and "-DLG_X_INDEXER=0" in jit.large_defs({}, "dense")
    assert "-DLG_NEGCTRL_NOWAIT=0" in d and len([x for x in d if x.startswith("-DLG_NEGCTRL_NOWAIT")]) == 1
    with pytest.raises(ValueError):
        jit.large_defs({"bogus": 1}, "dense")
    with pytest.raises(ValueError):
        jit.large_defs({"no_gridread": 3}, "dense")
    tool = {"hipcc": "/rocm/bin/hipcc", "clangxx": "/rocm/llvm/bin/clang++", "bundler": "b", "rocm": "/rocm", "mori_root": "/m", "mori_so": "/m/libmori_cco.so"}
    dc = jit.large_device_cmd(d, tool, "/out")
    assert "--offload-arch=gfx950" in dc and "-D__HIPCC_RTC__" in dc and dc[-3].endswith("gemm_quad_fp8_comm.cu")
    hc = jit.large_host_cmd(d, tool, "/out")
    assert "-DLG_X_INDEXER=1" in hc and "-DLG_NO_GRIDREAD=1" not in hc and "/m/libmori_cco.so" in hc
    assert jit.large_cache_key(d, tool) != jit.large_cache_key(jit.large_defs({}, "dense"), tool)


def test_large_vendored_manifest():
    import hashlib

    base = os.path.join(jit.CSRC, "large", "r11")
    n = 0
    for line in open(os.path.join(base, "SOURCE_SHA256.txt")):
        h, f = line.split(None, 1)
        f = f.strip()
        if f.startswith("tests/") or f == "host/bridge_comm.py":
            continue
        assert hashlib.sha256(open(os.path.join(base, f), "rb").read()).hexdigest() == h, f
        n += 1
    assert n >= 10


def test_large_variant_contract_and_sizes(monkeypatch):
    from aiter.gemm_fusion import catalog

    monkeypatch.setattr(catalog, "_CATALOG", {})
    sys.modules.pop("aiter.ops.gemm_fusion.sp_qkv.variants", None)
    sys.modules.pop("aiter.ops.gemm_fusion.sp_qkv.variants_large", None)
    import aiter.ops.gemm_fusion.sp_qkv.variants_large as vl  # noqa: F401

    profmod.clear_profile_cache()
    caps = frozenset({"mori_cco", "sdma"})
    spec = specs.sp_qkv_spec("dense")
    plan, n = match_prefix(spec, MatchCtx("gfx950", 256, 4, 2048, caps), PROFILE)
    assert n == 1 and plan.vid == "sp_qkv.large.v1" and plan.half_first == 2 and plan.n == 3
    reqs = plan.resources()
    assert [r.kind for r in reqs] == ["p2p_window", "p2p_window", "control", "sdma_queue"]
    assert all(r.bytes % (4 << 20) == 0 for r in reqs[:3]) and reqs[0].bytes == reqs[1].bytes == reqs[2].bytes
    # M=2048 dense: P = 8*2048*2304 = 37748736 -> + pad -> 40 MiB multiple of 4 MiB
    assert vl.large_window_bytes("dense", 2048) == 40 << 20
    assert vl.large_window_bytes("indexer", 8192) == round_up(8 * 8192 * 2560 + 4096)
    with pytest.raises(ContractError, match="sdma"):
        match_prefix(spec, MatchCtx("gfx950", 256, 4, 2048, frozenset({"mori_cco"})), PROFILE)
    # 2-stage prefix has no row at large M -> falls to the 1-stage row (n == 1) ; at M=16 the small variant would apply
    assert match_prefix(spec, MatchCtx("gfx950", 256, 4, 1024, caps), PROFILE) == (None, 0)
    # invalid plan params are contract errors, not silent
    sp = specs.sp_qkv_spec("indexer").prefix(1)
    with pytest.raises(ContractError):
        vl.SpQkvLarge(sp, {"n": 3, "half_first": 2}, MatchCtx("gfx950", 256, 4, 2048, caps))
    with pytest.raises(ContractError):
        vl.SpQkvLarge(specs.sp_qkv_spec("dense").prefix(1), {"n": 1, "half_first": 1}, MatchCtx("gfx950", 256, 4, 2048, caps))


def test_reset_runs_init_hooks():
    be, d = FakeBackend(), FakeDist()
    r = FusionResources(0, 4, dist=d, backend=be)
    r.ensure("p", [WindowReq("a", 100, "p2p_window")])
    calls = []
    r.set_init("p", lambda: calls.append(1))
    assert calls == [1]
    r.finalize()
    r.quiesce_and_reset()
    assert calls == [1, 1]


def test_reset_order_with_drain_hook():
    """A2-H for large-M: device sync -> drain (all plans) + sync -> barrier -> zero+init -> sync -> barrier."""
    be, d = FakeBackend(), FakeDist()
    ev = []
    be_orig_sync, be_orig_zero = be.device_sync, be.zero
    be.device_sync = lambda dev: (ev.append("sync"), be_orig_sync(dev))[1]
    be.zero = lambda dev, loc, n: (ev.append("zero"), be_orig_zero(dev, loc, n))[1]
    d.barrier = lambda group=None: ev.append("barrier")
    r = FusionResources(0, 4, dist=d, backend=be)
    r.ensure("big", [WindowReq("a", 100, "p2p_window")])
    r.ensure("small", [WindowReq("b", 100, "p2p_window")])
    r.set_drain("big", lambda: ev.append("drain"))
    r.set_init("big", lambda: ev.append("init"))
    r.finalize()
    ev.clear()
    r.quiesce_and_reset()
    assert ev == ["sync", "drain", "sync", "barrier", "zero", "init", "zero", "sync", "barrier"]
    # no drain hooks -> no extra sync (small-M only)
    r2 = FusionResources(0, 4, dist=d, backend=FakeBackend())
    r2.ensure("s", [WindowReq("b", 100, "p2p_window")])
    ev2 = []
    r2.backend.device_sync = lambda dev: ev2.append("sync")
    r2.backend.zero = lambda dev, loc, n: ev2.append("zero")
    d.barrier = lambda group=None: ev2.append("barrier")
    r2.quiesce_and_reset()
    assert ev2 == ["sync", "barrier", "zero", "sync", "barrier"]


def test_release_runs_close_hooks_before_comm_teardown_and_poison_or():
    be, d = FakeBackend(), FakeDist()
    r = FusionResources(0, 4, dist=d, backend=be)
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
    assert tail.index("sync") < tail.index("devdestroy") < tail.index("dereg") < tail.index("commdestroy")
    assert seen[0][1] == n_before + 1  # exactly after the leading device sync, before devcomm/window/comm teardown


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
    sm = open(os.path.join(base, "sm_fusion_driver.cpp"), encoding="utf-8").read()
    lg = open(os.path.join(base, "lg_fusion_driver.cpp"), encoding="utf-8").read()
    assert "own_ctx" in sm and "g_swap_mu" in sm and "single-thread contract" in sm  # H-1
    assert "sm_free_v1" in sm and "sm_poison_fast_v1" in sm                            # L-5 / poison
    assert "sm_image_matches(hsaco_path, expected_hsaco_sha256_hex)" in lg             # H-3
    assert lg.index("sm_image_matches") < lg.index("hipModuleLoad")                    # gate before any module action
    assert "lg_free_v1" in lg and "lg_poison_v1" in lg
    assert os.path.exists(os.path.join(base, "sha256_image.hpp"))


# ---- exact golden CaseSpec cross-check (case_config.py is importable without torch/GPU) ----
def _golden_case_config():
    import importlib.util

    cands = [os.environ.get("SP4_GOLDEN_DIR", "")] + [
        os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * n), "sp4-attention-fusion-cases-20261009")) for n in (2, 3)
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
    mode_map = {"qkv_dense": "dense", "qkv_indexer": "indexer", "qkv_indexer_reuse": "reuse"}
    profmod.clear_profile_cache()
    prof = profmod.load_profile(PROFILE)
    seen = 0
    for gmode, mode in mode_map.items():
        for T in (64, 128, 8192, 16384, 32768):
            case = cc.CaseSpec(gmode, T)
            sp = specs.sp_qkv_spec(mode)
            hn = sp.stages[1]
            assert hn.kv_cache == f"shuffle_page{case.cache_page_size}_fp8_ptok", (gmode, T)
            assert sp.gemm.n == case.gemm_nk[0] and sp.gemm.k == case.gemm_nk[1], (gmode, T)
            f_exch = sum(x.width // x.shards for x in sp.stages[0].fields if x.name not in sp.stages[0].drop)
            golden_f = case.describe()["exchange_shape"][1]  # 2304 dense, 2560 indexer AND reuse (the receiver sees a [T,2560] tensor)
            assert golden_f == (2304 if mode == "dense" else 2560), (gmode, T)
            # approved reuse transfer is COMPACT (q/k/v only, F=2304) and is presented to the receiver as a [T,2560] view with row stride 2304
            assert f_exch == (2560 if mode == "indexer" else 2304), (gmode, T)
            small = T in (64, 128)
            full = sp if small else sp.prefix(1)  # small: GEMM+exchange+norm/rope/cache; large: GEMM+exchange only
            row = prof.lookup(full.signature(), "gfx950", 256, 4, case.local_tokens)
            assert row is not None, f"no profile row for {gmode} T={T} (page {case.cache_page_size})"
            assert row.vid == ("sp_qkv.small.v1" if small else "sp_qkv.large.v1")
            seen += 1
    assert seen == 15
    # the BlockSize the runtime derives for the binding equals the golden page
    import re

    for gmode, mode in mode_map.items():
        case = cc.CaseSpec(gmode, 64)
        assert int(re.search(r"page(\d+)", specs.sp_qkv_spec(mode).stages[1].kv_cache).group(1)) == case.cache_page_size


def test_dense_page128_signature_differs_from_page16():
    d = specs.sp_qkv_spec("dense")
    assert d.stages[1].kv_cache == "shuffle_page128_fp8_ptok"
    assert specs.sp_qkv_spec("indexer").stages[1].kv_cache == specs.sp_qkv_spec("reuse").stages[1].kv_cache == "shuffle_page16_fp8_ptok"
    import dataclasses

    wrong = type(d)(d.gemm, (d.stages[0], dataclasses.replace(d.stages[1], kv_cache="shuffle_page16_fp8_ptok")))
    assert wrong.signature() != d.signature()  # a page16 dense spec must NOT hit the shipped dense row

def test_cache_page_contract_check():
    from types import SimpleNamespace as NS

    from aiter.ops.gemm_fusion.sp_qkv.variants import check_cache_page

    def caches(page, blocks=4):
        return NS(shape=(blocks, 1, 8, page, 16)), NS(shape=(blocks, 1, page // 16, 128, 16))

    for page in (128, 16):
        check_cache_page(*caches(page), page)  # matches
    with pytest.raises(ContractError, match="page mismatch"):
        check_cache_page(*caches(16), 128)  # dense spec (page128) given page16 caches
    with pytest.raises(ContractError, match="page mismatch"):
        check_cache_page(*caches(128), 16)
    with pytest.raises(ContractError):
        check_cache_page(NS(shape=(4, 1, 128, 16)), NS(shape=(4, 1, 8, 128, 16)), 128)  # not the SHUFFLE rank


def test_plan_block_comes_from_spec_page(monkeypatch):
    """The binding block_size equals the page in the matched spec: dense 128, indexer/reuse 16 (real golden pages)."""
    from aiter.gemm_fusion import catalog

    monkeypatch.setattr(catalog, "_CATALOG", {})
    sys.modules.pop("aiter.ops.gemm_fusion.sp_qkv.variants", None)
    sys.modules.pop("aiter.ops.gemm_fusion.sp_qkv.variants_large", None)
    import aiter.ops.gemm_fusion.sp_qkv.variants as v

    caps = frozenset({"mori_cco", "gfx950_sc_bypass"})
    for mode, T, page in (("dense", 64, 128), ("dense", 128, 128), ("indexer", 64, 16), ("reuse", 128, 16)):
        plan = v.SpQkvSmall(specs.sp_qkv_spec(mode), specs.SMALL_CELLS[(mode, T)][1], MatchCtx("gfx950", 256, 4, T // 4, caps))
        assert plan.block == page, (mode, T)

def test_binding_rejects_wrong_cache_page_before_any_driver_call(monkeypatch):
    """_binding must raise the page ContractError BEFORE it touches the driver (a dense spec with page16 caches must not bind)."""
    import types as _t

    from aiter.gemm_fusion import catalog
    from types import SimpleNamespace as NS

    monkeypatch.setattr(catalog, "_CATALOG", {})
    sys.modules.pop("aiter.ops.gemm_fusion.sp_qkv.variants", None)
    sys.modules.pop("aiter.ops.gemm_fusion.sp_qkv.variants_large", None)
    import aiter.ops.gemm_fusion.sp_qkv.variants as v

    class T:  # minimal tensor stand-in
        def __init__(self, shape, stride=None, dtype="f32"):
            self.shape, self._stride, self.dtype, self.device = tuple(shape), stride, dtype, "cuda:0"
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

    plan = v.SpQkvSmall(specs.sp_qkv_spec("dense"), specs.SMALL_CELLS[("dense", 64)][1],
                        MatchCtx("gfx950", 256, 4, 16, frozenset({"mori_cco"})))
    assert plan.block == 128
    res = NS(__dict__={})
    res = type("R", (), {})()  # plain object with a __dict__
    calls = []
    plan._context = lambda r: calls.append("context") or NS(bind=lambda sb: calls.append("bind"))
    x, xs, ws = T((16, 6144), (6144, 1)), T((16, 1)), T((9216,))
    meta = {n: T((128,)) for n in ("q_norm_w", "k_norm_w")}
    meta.update(cos_cache=T((1024, 32), (32, 1)), sin_cache=T((1024, 32), (32, 1)), positions=T((64,)), slot_mapping=T((64,)),
                k_scale=T((4, 1, 128), (128, 128, 1)), v_scale=T((4, 1, 128), (128, 128, 1)))
    outs = {"q": T((64, 16, 128)), "k": T((64, 1, 128)), "v": T((64, 1, 128))}
    meta.update(k_cache=T((4, 1, 8, 16, 16)), v_cache=T((4, 1, 1, 128, 16)))  # page16 caches vs a page128 dense spec
    with pytest.raises(ContractError, match="page mismatch"):
        plan._binding(x, xs, ws, outs, meta, res)
    assert "bind" not in calls
    meta.update(k_cache=T((4, 1, 8, 128, 16), (16384, 16384, 2048, 16, 1)), v_cache=T((4, 1, 8, 128, 16), (16384, 16384, 2048, 16, 1)))
    blk = plan._binding(x, xs, ws, outs, meta, res)  # correct page -> reaches the driver bind
    assert "bind" in calls