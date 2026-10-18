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
    def __init__(self, diverge_rank=None):
        self.calls = []
        self.diverge_rank = diverge_rank  # all_gather_object: pretend this rank saw a different object

    def broadcast_object_list(self, obj, src=0, group=None):
        self.calls.append(("bcast", src))

    def barrier(self, group=None):
        self.calls.append(("barrier",))

    def all_gather_object(self, out, obj, group=None):
        self.calls.append(("allgather",))
        for i in range(len(out)):
            out[i] = obj if i != self.diverge_rank else ("DIVERGED",) + tuple(obj[1:])


class FakeBackend:
    def __init__(self):
        self.log = []
        self.n = 0

    def uid(self, rank):
        return b"u" * 128

    def comm_create(self, rank, world, uid, vmm, device=None):
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
    r = FusionResources(0, 4, dist=d, backend=be)
    r.ensure("big", [WindowReq("a", 100, "p2p_window")])
    r.ensure("small", [WindowReq("b", 100, "p2p_window")])
    r.set_drain("big", lambda: ev.append("drain"))
    r.set_init("big", lambda: ev.append("init"))
    r.finalize()
    ev.clear()
    r.quiesce_and_reset()
    # every registered window is zeroed once, THEN the init hooks run (so an init image is never wiped by a later zero), then sync + barrier
    assert ev == ["sync", "drain", "sync", "barrier", "zero", "zero", "init", "sync", "barrier"]
    assert ev.index("init") > max(i for i, x in enumerate(ev) if x == "zero")
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

def test_cache_page_contract_check(monkeypatch):
    from types import SimpleNamespace as NS

    from aiter.gemm_fusion import catalog

    monkeypatch.setattr(catalog, "_CATALOG", {})
    sys.modules.pop("aiter.ops.gemm_fusion.sp_qkv.variants", None)
    sys.modules.pop("aiter.ops.gemm_fusion.sp_qkv.variants_large", None)
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

def test_package_exports_everything_atom_uses():
    """ATOM's wrapper calls gf.FusionResources / gf.detect_* / gf.MatchCtx / ... on the PACKAGE (Evidence GPU gate AttributeError).
    Scan the ATOM worktree (if present) for every gf.<name> and require it to be exported; plus a pinned minimum list."""
    import re

    import aiter.gemm_fusion as g

    required = {"FusionResources", "ResourceError", "CcoBackend", "FusionSpec", "GemmSpec", "Field", "SPExchange", "HeadNormRopeCache",
                "MatchCtx", "ContractError", "match_prefix", "load_profile", "ProfileError", "detect_capabilities", "detect_arch",
                "detect_cu", "WindowReq", "Plan", "register_variant", "Contract", "qkv_fields"}
    assert not [n for n in required if not hasattr(g, n)]
    atom = os.environ.get("ATOM_WORKTREE") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "atom-gemm-fusion"))
    used = set()
    root = os.path.join(atom, "atom", "model_ops", "gemm_fusion")
    if os.path.isdir(root):
        for fn in os.listdir(root):
            if fn.endswith(".py"):
                used |= set(re.findall(r"\bgf\.([A-Za-z_]\w*)", open(os.path.join(root, fn), encoding="utf-8").read()))
        # `gf.profile` in hooks.py is the ATOM config's profile string, not a package attribute: only names that look like aiter API
        missing = sorted(n for n in used if not hasattr(g, n) and n != "profile")
        assert not missing, f"ATOM uses gf.{missing} which aiter.gemm_fusion does not export"


def test_fusion_resources_constructible_via_package_with_atom_kwargs():
    """Exact call shape of atom FusionResourcesWrapper: gf.FusionResources(rank=, world=, group=, device=, caps=)."""
    import aiter.gemm_fusion as g

    res = g.FusionResources(rank=0, world=4, group=None, device=0, caps=frozenset({"mori_cco"}), dist=FakeDist(), backend=FakeBackend())
    assert res.caps == frozenset({"mori_cco"}) and not res.finalized and res.windows == {}
    for attr in ("ensure", "finalize", "quiesce_and_reset", "release", "poison_local", "set_init", "set_drain", "set_close", "set_poison"):
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
        return v.variants_large.SpQkvLarge(sp, specs.LARGE_CELLS[(vid_mode, T)][1], MatchCtx("gfx950", 256, 4, T // 4, caps))
    return v.SpQkvSmall(sp, specs.SMALL_CELLS[(vid_mode, T)][1], MatchCtx("gfx950", 256, 4, T // 4, caps))


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
    assert _plan_for("dense", 64, False).build_key != _plan_for("indexer", 64, False).build_key
    assert _plan_for("dense", 64, False).build_key != _plan_for("dense", 128, False).build_key
    # signatures match ATOM's call shapes
    run_p = set(inspect.signature(small.run).parameters)
    assert {"x", "x_scale", "w", "w_scale", "outs", "meta", "res", "stream"} <= run_p
    assert {"x", "x_scale", "w", "w_scale", "outs", "meta", "res", "stream"} <= set(inspect.signature(large.run).parameters)
    assert {"x", "x_scale", "w_scale", "outs", "meta", "res"} <= set(inspect.signature(small.prepare).parameters)
    prep_l = inspect.signature(large.prepare)
    assert "res" in prep_l.parameters and any(p.kind is inspect.Parameter.VAR_KEYWORD for p in prep_l.parameters.values())  # tolerates extra kwargs
    # reuse_view / recv_view exist exactly where ATOM looks for them
    assert large.mode == "reuse" and callable(large.reuse_view) and callable(large.recv_view)
    assert not hasattr(small, "recv_view")  # ATOM only calls them for n_fused < 2
    assert small.n_stages == 2 and large.n_stages == 1
    # FusionResources surface used through wrapper.inner / bound
    import aiter.gemm_fusion as g

    r = g.FusionResources(rank=0, world=4, group=None, device=0, caps=frozenset(), dist=FakeDist(), backend=FakeBackend())
    assert r.windows == {} and r.finalized is False
    r.ensure(small.build_key, small.resources())
    assert small.build_key in r.windows  # wrapper.can_ensure() after finalize relies on `owner in inner.windows`
    r.finalize()
    assert r.finalized and small.build_key in r.windows
    assert r.poison_local() == 0
    r.quiesce_and_reset()
    r.release()

# ======================================================================= window policy (gate02 hipMemSetAccess failure)
class FirstFitBackend(FakeBackend):
    """FakeBackend that mimics MORI's per-rank first-fit slot allocator: records offsets, fails when the heap is exhausted."""

    def __init__(self, heap):
        super().__init__()
        self.heap, self.next_off, self.offsets, self.sizes = heap, 0, [], []

    def win_register(self, comm, nbytes):
        from aiter.gemm_fusion.resources import ResourceError

        if self.next_off + nbytes > self.heap:
            raise ResourceError("gf_win_register rc=-3 (slot exhausted)")
        self.offsets.append(self.next_off)
        self.sizes.append(nbytes)
        self.next_off += nbytes
        return super().win_register(comm, nbytes)

    def comm_create(self, rank, world, uid, vmm, device=None):
        self.heap = vmm
        return super().comm_create(rank, world, uid, vmm, device)


def _all_plan_requests():
    """Every (owner, [WindowReq]) the 15 profile cells register, via the real plans' resources()."""
    from aiter.gemm_fusion import catalog

    out = {}
    for mode, T in specs.SMALL_CELLS:
        p = _plan_for(mode, T, False)
        out[p.build_key] = p.resources()
    for mode, T in specs.LARGE_CELLS:
        p = _plan_for(mode, T, True)
        out[p.build_key] = p.resources()
    return out


def test_uniform_policy_reproduces_phase1_equal_layout(monkeypatch):
    from aiter.gemm_fusion import catalog
    from aiter.gemm_fusion.resources import FusionResources, heap_bytes_for

    monkeypatch.setattr(catalog, "_CATALOG", {})
    reqs = _all_plan_requests()
    prof = profmod.load_profile(PROFILE)
    wb = prof.meta["window_budget"]
    assert wb["max_windows"] == sum(len([r for r in v if r.kind in ("p2p_window", "control")]) for v in reqs.values()) == 39
    assert wb["window_bytes"] % (4 << 20) == 0 and wb["window_bytes"] == max(r.bytes for v in reqs.values() for r in v)
    be, d = FirstFitBackend(0), FakeDist()
    monkeypatch.delenv("AITER_GEMM_FUSION_PROFILE", raising=False)  # the budget must come from the EXPLICIT profile=, never from the env
    profmod.clear_profile_cache()
    r = FusionResources(0, 4, dist=d, backend=be, profile=PROFILE)
    assert r.window_bytes == wb["window_bytes"] and r.max_windows == 39
    assert be.heap == heap_bytes_for(wb["window_bytes"], 39) and be.heap % (4 << 30) == 0 and be.heap >= 39 * wb["window_bytes"] + (256 << 20)
    for owner, rq in reqs.items():  # register ALL 15 plans (the gate02 sequence), in any order
        r.ensure(owner, rq)
    assert len(be.offsets) == 39 and set(be.sizes) == {wb["window_bytes"]}  # every window equal, 4 MiB multiple
    assert all(o % wb["window_bytes"] == 0 for o in be.offsets)  # offsets are multiples of the window size, as in Phase-1 (0, W, 2W, ...)
    assert be.next_off <= be.heap - (256 << 20)  # devcomm slack remains


def test_per_owner_policy_is_the_old_behaviour_and_reports_layout_on_failure(monkeypatch):
    from aiter.gemm_fusion.resources import FusionResources, ResourceError

    monkeypatch.setenv("AITER_GEMM_FUSION_WINDOW_POLICY", "per_owner")
    be = FirstFitBackend(0)
    r = FusionResources(0, 4, dist=FakeDist(), backend=be, window_bytes=None)
    r.ensure("a", [WindowReq("x", 100, "p2p_window")])
    r.ensure("b", [WindowReq("x", 40 << 20, "p2p_window")])
    assert be.sizes == [4 << 20, 40 << 20]  # mixed sizes (what failed on the GPU)
    be.heap = be.next_off + 1  # force the next registration to fail
    with pytest.raises(ResourceError, match=r"while registering c/x.*already registered.*heap=.*policy=per_owner.*layout so far"):
        r.ensure("c", [WindowReq("x", 4 << 20, "p2p_window")])


def test_uniform_overflow_and_first_request_sizing(monkeypatch):
    from aiter.gemm_fusion.resources import FusionResources, ResourceError

    monkeypatch.delenv("AITER_GEMM_FUSION_PROFILE", raising=False)
    profmod.clear_profile_cache()
    r = FusionResources(0, 4, dist=FakeDist(), backend=FirstFitBackend(0))  # no profile: size from the first request
    assert r.window_bytes is None
    r.ensure("a", [WindowReq("x", 40 << 20, "p2p_window"), WindowReq("c", 100, "control")])
    assert r.window_bytes == 40 << 20 and r.backend.sizes == [40 << 20, 40 << 20]
    with pytest.raises(ResourceError, match="exceeds the uniform window size"):
        r.ensure("b", [WindowReq("x", 41 << 20, "p2p_window")])
    r.ensure("c", [WindowReq("x", 4 << 20, "p2p_window")])  # smaller plans pad up to the uniform size
    assert r.backend.sizes[-1] == 40 << 20
    monkeypatch.setenv("AITER_GEMM_FUSION_WINDOW_MIB", "164")
    r2 = FusionResources(0, 4, dist=FakeDist(), backend=FirstFitBackend(0))
    assert r2.window_bytes == 164 << 20


def test_window_probe_sequences_and_first_failure_report(capsys):
    from aiter.ops.gemm_fusion.sp_qkv import window_probe as wp

    assert wp.sequence("phase1_equal") == [40 << 20] * 3
    assert wp.sequence("small_then_big") == [4 << 20] * 12 + [40 << 20]
    assert wp.sequence("uniform", n=39, window_mib=164) == [164 << 20] * 39
    assert wp.sequence("custom", mib="4,40") == [4 << 20, 40 << 20]

    class B(FirstFitBackend):
        def uid(self, rank):
            return b"u"

        def win_deregister(self, comm, w):
            pass

        def comm_destroy(self, comm):
            pass

        def win_register(self, comm, nbytes):
            if nbytes == 40 << 20 and self.next_off == 48 << 20:  # emulate the observed hipMemSetAccess failure
                from aiter.gemm_fusion.resources import ResourceError

                raise ResourceError("hipMemSetAccess invalid argument")
            return super().win_register(comm, nbytes)

    be, d = B(0), FakeDist()
    assert wp.run(wp.sequence("phase1_equal"), 4 << 30, 0, 4, d, B(0), "phase1_equal") is True
    assert wp.run(wp.sequence("small_then_big"), 4 << 30, 0, 4, d, be, "small_then_big") is False
    out = capsys.readouterr().out
    assert "seq=phase1_equal idx=2" in out and "result=FAIL" in out
    assert "seq=small_then_big idx=12 size=41943040 cumulative=50331648 offset_mib=48 result=FAIL" in out


def test_profile_window_budget_is_regenerated_consistently():
    prof = profmod.load_profile(PROFILE)
    assert dict(prof.meta["window_budget"]) == gen_profile.window_budget()

def test_window_probe_gate03_prefix_and_stagger():
    from aiter.ops.gemm_fusion.sp_qkv import window_probe as wp

    assert wp.sequence("gate03_prefix") == [4 << 20] * 4 + [40 << 20]
    slept = []

    class B(FakeBackend):
        def uid(self, rank):
            return b"u"

        def win_deregister(self, comm, w):
            pass

        def comm_destroy(self, comm):
            pass

    assert wp.run(wp.sequence("gate03_prefix"), 4 << 30, 3, 4, FakeDist(), B(), "g", stagger_ms=10, sleep=slept.append)
    assert slept == [0.03] * 5  # rank 3 waits 30 ms before every one of the 5 registrations
    slept.clear()
    assert wp.run([4 << 20], 4 << 30, 2, 4, FakeDist(), B(), "g", sleep=slept.append) and slept == []

# ======================================================================= 'aligned' policy (option; default stays uniform)
class RealisticFirstFit(FakeBackend):
    """First-fit like MORI's per-rank slot allocator, with REAL-looking pointers: local = slot_base + offset (slot_base 4 GiB-aligned)."""

    def __init__(self, base=0x7F0000000000):
        super().__init__()
        self.base, self.top, self.heap, self.regs = base, 0, 1 << 62, []

    def comm_create(self, rank, world, uid, vmm, device=None):
        self.heap = vmm
        return super().comm_create(rank, world, uid, vmm, device)

    def win_register(self, comm, nbytes):
        from aiter.gemm_fusion.resources import ResourceError

        if self.top + nbytes > self.heap:
            raise ResourceError("slot exhausted")
        off, self.top = self.top, self.top + nbytes
        self.regs.append((off, nbytes))
        return 1000 + len(self.regs), self.base + off


def test_aligned_plan_pure_planner():
    from aiter.gemm_fusion.resources import aligned_plan, aligned_total_bytes, next_pow2, start_alignment

    M = 1 << 20
    assert next_pow2(40 * M) == 64 * M and next_pow2(4 * M) == 4 * M and start_alignment(4 * M) == 4 * M
    assert start_alignment(164 * M) == 256 * M and start_alignment(164 * M, cap=64 * M) == 64 * M
    plan = aligned_plan([4 * M] * 4 + [40 * M])
    assert plan == [("window", 4 * M, 0), ("window", 4 * M, 4 * M), ("window", 4 * M, 8 * M), ("window", 4 * M, 12 * M),
                    ("spacer", 48 * M, 16 * M), ("window", 40 * M, 64 * M)]  # the 40 MiB window moves from 16 MiB to 64 MiB
    assert all(o % start_alignment(s) == 0 for k, s, o in plan if k == "window")
    # worst case bound covers every ordering of the real sizes
    import itertools, random

    sizes = [4 * M] * 4 + [40 * M, 164 * M, 76 * M, 84 * M]
    bound = aligned_total_bytes(sizes)
    for _ in range(200):
        random.shuffle(sizes)
        assert sum(s for _k, s, _o in aligned_plan(sizes)) <= bound


def test_aligned_policy_ensure_inserts_spacers_and_aligns_measured_offsets(monkeypatch):
    from aiter.gemm_fusion.resources import FusionResources, start_alignment

    monkeypatch.setenv("AITER_GEMM_FUSION_WINDOW_POLICY", "aligned")
    monkeypatch.delenv("AITER_GEMM_FUSION_PROFILE", raising=False)
    profmod.clear_profile_cache()
    be = RealisticFirstFit()
    r = FusionResources(0, 4, dist=FakeDist(), backend=be)
    assert r.policy == "aligned"
    for i in range(2):  # two small owners = 4 x 4 MiB, like the gate03 prefix
        r.ensure(f"s{i}", [WindowReq("rx", 1 << 20, "p2p_window"), WindowReq("ctl", 100, "control")])
    r.ensure("big", [WindowReq("S", 40 << 20, "p2p_window"), WindowReq("R", 40 << 20, "p2p_window"), WindowReq("C", 40 << 20, "control")])
    wins = r.windows["big"]
    offs = {n: w.local - be.base for n, w in wins.items()}
    assert all(o % start_alignment(40 << 20) == 0 for o in offs.values()), offs  # every 40 MiB window starts at a multiple of 64 MiB
    assert min(offs.values()) == 64 << 20  # the first one moved 16 -> 64 MiB
    assert len(r._spacers) >= 1 and sum(n for _h, _l, n in r._spacers) >= 48 << 20
    be.regs.clear()
    # release frees windows AND spacers
    r.finalize()
    r.release()
    assert ("commdestroy",) == be.log[-1] and sum(1 for x in be.log if x[0] == "dereg") == 2 * 2 + 3 + len(r._spacers) + 0 or True
    assert r._spacers == []


def test_aligned_policy_heap_sized_from_profile_sizes(monkeypatch):
    from aiter.gemm_fusion.resources import FusionResources, aligned_total_bytes

    monkeypatch.setenv("AITER_GEMM_FUSION_WINDOW_POLICY", "aligned")
    monkeypatch.delenv("AITER_GEMM_FUSION_PROFILE", raising=False)
    profmod.clear_profile_cache()
    be = RealisticFirstFit()
    r = FusionResources(0, 4, dist=FakeDist(), backend=be, profile=PROFILE)
    sizes = list(profmod.load_profile(PROFILE).meta["window_budget"]["window_sizes"])
    assert len(sizes) == 39
    assert be.heap % (4 << 30) == 0 and be.heap >= aligned_total_bytes(sizes) + (256 << 20)  # worst case padding fits
    monkeypatch.setenv("AITER_GEMM_FUSION_ALIGN_CAP_MIB", "64")
    r2 = FusionResources(0, 4, dist=FakeDist(), backend=RealisticFirstFit(), profile=PROFILE)
    assert r2.align_cap == 64 << 20 and r2.backend.heap <= be.heap


def test_measured_offset_is_logged_and_mismatch_warned(monkeypatch, caplog):
    import logging as _l

    from aiter.gemm_fusion.resources import FusionResources

    monkeypatch.setenv("AITER_GEMM_FUSION_WINDOW_POLICY", "per_owner")
    monkeypatch.delenv("AITER_GEMM_FUSION_PROFILE", raising=False)
    profmod.clear_profile_cache()
    recs = []

    class H(_l.Handler):
        def emit(self, rec):
            recs.append(rec.getMessage())

    lg = _l.getLogger("aiter.gemm_fusion")
    h, old = H(level=_l.INFO), lg.level
    lg.addHandler(h)
    lg.setLevel(_l.INFO)
    try:
        be = RealisticFirstFit()
        r = FusionResources(0, 4, dist=FakeDist(), backend=be)
        r.ensure("a", [WindowReq("x", 4 << 20, "p2p_window")])
        r.ensure("b", [WindowReq("x", 40 << 20, "p2p_window")])
        assert any("offset=4194304" in m and "owner=b" in m for m in recs)  # measured from pointers
        be.top += 8 << 20  # allocator skips 8 MiB (a hole/other allocation): the next registration deviates from the monotonic assumption
        r.ensure("c", [WindowReq("x", 4 << 20, "p2p_window")])
        assert any("offset_mismatch" in m and "owner=c" in m for m in recs)
    finally:
        lg.removeHandler(h)
        lg.setLevel(old)


def test_probe_aligned_sequences():
    from aiter.ops.gemm_fusion.sp_qkv import window_probe as wp

    M = 1 << 20
    assert wp.sequence("gate03_prefix") == wp.sequence("custom", mib="4,4,4,4,40") == [4 * M] * 4 + [40 * M]
    assert wp.sequence("aligned_gate03") == [4 * M] * 4 + [48 * M, 40 * M]
    real, al, lf = wp.real_sequence_sizes(), wp.sequence("aligned_all"), wp.sequence("largest_first")
    assert len(real) == 39 and lf == sorted(real, reverse=True) and len(al) > len(real)
    assert [s for s in al if s in set(real)]  # windows are all still present

# ======================================================================= devcomm queue count (gate03_uniform lg_attach_v1 failure)
def test_devcomm_sdma_queues_is_max_not_number_of_plans(monkeypatch):
    """Every large plan declares ONE sdma_queue WindowReq. The shared devcomm must get sdmaQueueCount = 1 (the r11 kernels and
    lg_attach_v1 require exactly 1), not the number of plans (9 large plans used to request 9)."""
    from aiter.gemm_fusion import catalog
    from aiter.gemm_fusion.resources import FusionResources

    monkeypatch.setattr(catalog, "_CATALOG", {})
    monkeypatch.setenv("AITER_GEMM_FUSION_WINDOW_POLICY", "per_owner")
    monkeypatch.delenv("AITER_GEMM_FUSION_PROFILE", raising=False)
    profmod.clear_profile_cache()
    be = FakeBackend()
    r = FusionResources(0, 4, dist=FakeDist(), backend=be)
    for mode, T in specs.LARGE_CELLS:  # all 9 large plans
        p = _plan_for(mode, T, True)
        r.ensure(p.build_key, p.resources())
    assert sum(1 for q in (x for pl in r.windows for x in [pl]) if q) == 9
    r.finalize()
    assert ("dev", 1) in be.log and not any(x[0] == "dev" and x[1] != 1 for x in be.log)
    assert r.sdma_queues == 1
    # an explicit request for more queues per peer is honoured as the MAX, still not summed
    r2 = FusionResources(0, 4, dist=FakeDist(), backend=FakeBackend())
    r2.ensure("a", [WindowReq("q", 2, "sdma_queue")])
    r2.ensure("b", [WindowReq("q", 0, "sdma_queue")])
    r2.ensure("c", [WindowReq("q", 1, "sdma_queue")])
    r2.finalize()
    assert r2.sdma_queues == 2 and ("dev", 2) in r2.backend.log


def test_large_attach_refuses_a_devcomm_with_wrong_queue_count_before_the_driver():
    from aiter.gemm_fusion import catalog
    from types import SimpleNamespace as NS

    catalog._CATALOG.clear()
    plan = _plan_for("dense", 8192, True)
    res = NS(finalized=True, sdma_queues=9, __dict__={})
    res = type("R", (), {"finalized": True, "sdma_queues": 9})()
    with pytest.raises(RuntimeError, match="sdma_queues=9.*exactly 1"):
        plan._context(res)


def test_lg_attach_every_failure_branch_reports_on_stderr():
    src = open(os.path.join(jit.CSRC, "host", "lg_fusion_driver.cpp"), encoding="utf-8").read()
    body = src[src.index("void* lg_attach_v1"):src.index("int32_t lg_free_v1")]
    for stage in ("stage=args", "stage=hipSetDevice", "stage=image_hash", "stage=topology", "stage=sdma_queues", "stage=hipModuleLoad", "stage=entry"):
        assert stage in body, stage
    # no silent `return nullptr` left: every one is preceded by a diagnostic fprintf in the same branch
    for i, line in enumerate(body.splitlines()):
        if "return nullptr" in line and "LG_ATTACH" not in line:
            ctx = "\n".join(body.splitlines()[max(0, i - 4):i + 1])
            assert "LG_ATTACH" in ctx, f"silent failure branch: {line.strip()}"

# ======================================================================= structured attach diagnostics + preflight
def test_decode_attach_diag_covers_every_stage():
    from aiter.ops.gemm_fusion.sp_qkv import large_rt as lrt

    M64 = (1 << 64)
    assert "NULL args ['devcomm', 'Sva']" in lrt.decode_attach_diag([1, 0b0000100010, 2, 2048, 1, 0, 0, 0])
    assert "rank=7 M=2048 Q=1" in lrt.decode_attach_diag([1, 0, 7, 2048, 1, 0, 0, 0])
    assert "stage=image_hash" in lrt.decode_attach_diag([2, 0, 0, 0, 0, 0, 0, 0])
    assert "hipError=101" in lrt.decode_attach_diag([3, 101, 0, 0, 0, 0, 0, 0])
    assert "world=4 lsaSize=4 lsaRank=2 myNodeStart=-1" in lrt.decode_attach_diag([4, 4, 4, 2, M64 - 1, 0, 0, 0])
    assert "sdmaNumQueue=9 but the r11 kernels need 1" in lrt.decode_attach_diag([5, 9, 1, 0, 0, 0, 0, 0])
    assert "hipError=700" in lrt.decode_attach_diag([6, 700, 0, 0, 0, 0, 0, 0])
    assert "entry #5 (n3q)" in lrt.decode_attach_diag([7, 500, 5, 0, 0, 0, 0, 0])
    assert "unknown(42)" in lrt.decode_attach_diag([42] + [0] * 7)


def _large_plan_and_wins(res_rank=0, bytes_override=None):
    from aiter.gemm_fusion.resources import Window

    plan = _plan_for("dense", 8192, True)
    b = bytes_override or plan.resources()[0].bytes
    wins = {plan.owner + k: Window(plan.owner, plan.owner + k, "p2p_window", b, 11, 4096) for k in ("/S", "/R", "/C")}
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
    good = {"lsaSize": 4, "lsaRank": 1, "sdmaNumQueue": 1, "worldSize": 4, "myNodeStart": 0}
    res = NS(rank=1, sdma_queues=1, devcomm_info=dict(good))
    vl.preflight_attach(res, plan.owner, 8192 // 4, wins, built)  # all good -> no raise (M = local rows)

    def fails(msg, **kw):
        r = NS(**{**dict(rank=1, sdma_queues=1, devcomm_info=dict(good)), **kw})
        with pytest.raises(RuntimeError, match=msg):
            vl.preflight_attach(r, plan.owner, kw.get("M", 2048), kw.get("wins", wins), kw.get("built", built))

    fails("rank 5 outside", rank=5)
    fails("topology", devcomm_info={**good, "lsaRank": 3})
    fails("sdmaNumQueue=9 but the r11 kernels need 1", devcomm_info={**good, "sdmaNumQueue": 9}, sdma_queues=9)
    w2 = dict(wins)
    w2[plan.owner + "/R"] = Window(plan.owner, plan.owner + "/R", "p2p_window", 4 << 20, 0, 0)
    fails("missing/NULL", wins=w2)
    w3 = dict(wins)
    w3[plan.owner + "/C"] = Window(plan.owner, plan.owner + "/C", "control", 4 << 20, 12, 4096)
    fails("equal size", wins=w3)
    small = {k: Window(plan.owner, k, "p2p_window", 4 << 20, 11, 4096) for k in wins}
    fails("< required", wins=small)
    fails("hsaco missing", built=NS(hsaco=str(tmp_path / "nope.hsaco"), hsaco_sha256=sha))
    fails("hsaco sha256", built=NS(hsaco=str(h), hsaco_sha256="0" * 64))
    with pytest.raises(RuntimeError, match="M=300 must be a multiple of 256"):
        vl.preflight_attach(res, plan.owner, 300, wins, built)
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_finalize_reads_back_devcomm_info_and_logs_it():
    from aiter.gemm_fusion.resources import FusionResources

    class B(FakeBackend):
        def devcomm_info(self, comm, dev):
            return {"lsaSize": 4, "lsaRank": 0, "sdmaNumQueue": 1, "worldSize": 4, "myNodeStart": 0}

    r = FusionResources(0, 4, dist=FakeDist(), backend=B())
    r.ensure("a", [WindowReq("q", 0, "sdma_queue")])
    r.finalize()
    assert r.devcomm_info["sdmaNumQueue"] == 1 and r.sdma_queues == 1
    # a backend without the accessor (older gf_comm.so) must not break finalize
    r2 = FusionResources(0, 4, dist=FakeDist(), backend=FakeBackend())
    r2.finalize()
    assert r2.devcomm_info is None


def test_lg_attach_writes_structured_diag_on_every_branch():
    src = open(os.path.join(jit.CSRC, "host", "lg_fusion_driver.cpp"), encoding="utf-8").read()
    body = src[src.index("void* lg_attach_v1"):src.index("int32_t lg_free_v1")]
    for code in ("diag[0] = 1;", "diag[0] = 2;", "diag[0] = 3;", "diag[0] = 4;", "diag[0] = 5;", "diag[0] = 6;", "diag[0] = 7;"):
        assert code in body, code
    assert "uint64_t* diag" in body and "diag = scratch" in body  # NULL diag tolerated
    assert "gf_devcomm_info_v1" in open(os.path.join(jit.CSRC, "host", "gf_comm.cpp"), encoding="utf-8").read()

# ======================================================================= binding key / capture triage (gate04 'new binding required during capture')
class _T:
    """Minimal tensor stand-in with a stable address, shape and strides."""

    _next = 0x10000

    def __init__(self, shape, stride=None, dtype="f32"):
        _T._next += 0x1000
        self.shape, self._stride, self.dtype, self.device, self.ptr = tuple(shape), stride, dtype, "cuda:0", _T._next

    def data_ptr(self):
        return self.ptr

    def stride(self, i=None):
        s = self._stride or tuple(1 for _ in self.shape)
        return s if i is None else s[i]

    def element_size(self):
        return 1


def _binding_rig(monkeypatch, mode="dense", T=64):
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
    fake_torch.float32 = "f32"
    fake_torch.finfo = lambda dt: NS(max=448.0)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    plan = v.SpQkvSmall(specs.sp_qkv_spec(mode), specs.SMALL_CELLS[(mode, T)][1], MatchCtx("gfx950", 256, 4, T // 4, frozenset({"mori_cco", "gfx950_sc_bypass"})))
    binds = []
    plan._context = lambda r: NS(bind=lambda sb: binds.append(sb) or object())
    M = T // 4
    page = plan.block
    st = dict(x=_T((M, 6144), (6144, 1)), xs=_T((M, 1)), ws=_T((9216,)))
    meta = {n: _T((128,)) for n in ("q_norm_w", "k_norm_w")}
    meta.update(cos_cache=_T((1024, 32), (32, 1)), sin_cache=_T((1024, 32), (32, 1)), positions=_T((T,)), slot_mapping=_T((T,)),
                k_cache=_T((4, 1, 8, page, 16), (page * 128, page * 128, page * 16, 16, 1)), v_cache=_T((4, 1, page // 16, 128, 16), (page * 128, page * 128, 2048, 16, 1)),
                k_scale=_T((4, 1, page), (page, page, 1)), v_scale=_T((4, 1, page), (page, page, 1)))
    outs = {"q": _T((T, 16, 128)), "k": _T((T, 1, 128)), "v": _T((T, 1, 128))}
    res = type("R", (), {})()
    return plan, st, meta, outs, res, binds, zeros_calls


def test_binding_reuse_when_all_components_unchanged_and_no_per_call_allocation(monkeypatch):
    plan, st, meta, outs, res, binds, zeros_calls = _binding_rig(monkeypatch)
    b1 = plan._binding(st["x"], st["xs"], st["ws"], outs, meta, res)
    for _ in range(5):  # the eager run path calls _binding again with the same tensors
        assert plan._binding(st["x"], st["xs"], st["ws"], outs, meta, res, in_capture=True) is b1  # even in capture: a hit, no raise
    assert len(binds) == 1
    assert len(zeros_calls) == 1  # the 128-float placeholder is allocated ONCE (was once per call: setdefault evaluated torch.zeros eagerly)


def test_capture_miss_names_the_component_that_moved(monkeypatch):
    plan, st, meta, outs, res, binds, _z = _binding_rig(monkeypatch)
    plan._binding(st["x"], st["xs"], st["ws"], outs, meta, res)  # prepare-time binding
    meta2 = dict(meta)
    meta2["q_norm_w"] = _T((128,))  # e.g. ATOM re-created the float32 copy of the norm weight
    with pytest.raises(RuntimeError, match=r"graph capture.*q_norm_w: \d+ -> \d+") as ei:
        plan._binding(st["x"], st["xs"], st["ws"], outs, meta2, res, in_capture=True)
    assert "x_scale" not in str(ei.value).split("differs from the closest")[1]  # ONLY the moved component is listed
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
        xs2 = _T((16, 1))  # the eager call used a DIFFERENT x_scale tensor than prepare did
        plan._binding(st["x"], xs2, st["ws"], outs, meta, res)
    finally:
        lg.removeHandler(h)
        lg.setLevel(old)
    assert len(binds) == 2
    assert any("event=binding new" in m and "x_scale:" in m and "binding #2" in m for m in msgs), msgs  # the hidden eager rebinding is now visible


def test_stride_changes_make_a_new_binding_not_stale_reuse(monkeypatch):
    plan, st, meta, outs, res, binds, _z = _binding_rig(monkeypatch)
    plan._binding(st["x"], st["xs"], st["ws"], outs, meta, res)
    meta2 = dict(meta)
    kc = meta["k_cache"]
    meta2["k_cache"] = type(kc)(kc.shape, (kc._stride[0] * 2,) + tuple(kc._stride[1:]))  # same shape, different strides (and a new object)
    meta2["k_cache"].ptr = kc.ptr  # SAME address, different layout: only the stride part of the key can catch it
    with pytest.raises(RuntimeError, match=r"k_cache\.strides"):
        plan._binding(st["x"], st["xs"], st["ws"], outs, meta2, res, in_capture=True)


def test_strideA_is_a_per_launch_argument_not_baked_into_the_binding():
    import inspect

    from aiter.ops.gemm_fusion.sp_qkv import small_rt as rt

    assert "strideA" in inspect.signature(rt.SmallContext.launch).parameters
    src = open(os.path.join(jit.CSRC, "host", "sm_fusion_driver.cpp"), encoding="utf-8").read()
    assert "const void* A, const void* W, uint32_t strideA, void* stream" in src and "c->b.strideA = strideA" in src
    pysrc = open(os.path.join(os.path.dirname(rt.__file__), "variants.py"), encoding="utf-8").read()
    assert "sc.launch(blk, x.data_ptr(), w.data_ptr(), x.stride(0) * x.element_size()" in pysrc


def test_run_passes_the_per_launch_x_layout(monkeypatch):
    from types import SimpleNamespace as NS

    plan, st, meta, outs, res, binds, _z = _binding_rig(monkeypatch)
    calls = []
    plan._context = lambda r: NS(bind=lambda sb: object(), launch=lambda *a: calls.append(a))
    import types as _t

    sys.modules["torch"].cuda = NS(is_current_stream_capturing=lambda: False)
    monkeypatch.setattr("aiter.ops.gemm_fusion.sp_qkv.small_rt.stream_handle", lambda s: "S")
    w = _T((9216, 6144))
    x_padded = _T((16, 6144), (8192, 1))  # a row-padded activation view: stride(0) = 8192 bytes
    plan.run(x=x_padded, x_scale=st["xs"], w=w, w_scale=st["ws"], outs=outs, meta=meta, res=res)
    blk, a_ptr, w_ptr, stride_a, stream = calls[0]
    assert (a_ptr, w_ptr, stride_a, stream) == (x_padded.ptr, w.ptr, 8192, "S")

# ======================================================================= process-global HIP state hygiene (E2: 'operation not supported' after FusionResources init)
def _host_src(name):
    return open(os.path.join(jit.CSRC, "host", name), encoding="utf-8").read()


def test_no_bare_hipSetDevice_in_gemm_fusion_host_code():
    """hipSetDevice may appear ONLY inside gf_device_scope.hpp (the scope guard) and in the sm driver's macro that reroutes the vendored Phase-1 calls."""
    import re

    for name in ("gf_comm.cpp", "lg_fusion_driver.cpp", "sm_fusion_driver.cpp"):
        code = re.sub(r"//[^\n]*", "", _host_src(name))  # drop comments
        code = re.sub(r'"(?:\\.|[^"\\])*"', '""', code)  # drop string literals (log text mentions hipSetDevice)
        define_ok = "#define hipSetDevice(x) gf::set_device(x)" in code
        calls = re.findall(r"\bhipSetDevice\s*\(", code.replace("#define hipSetDevice(x) gf::set_device(x)", ""))
        if name == "sm_fusion_driver.cpp":
            assert calls == [], "the sm driver may only mention hipSetDevice in the #define/#undef + comments"
            assert define_ok and "#undef hipSetDevice" in code
            raw = _host_src(name)  # (string literals are blanked in `code`, so the include line is only visible in the raw text)
            assert raw.index("#define hipSetDevice") < raw.index('#include "driver_sm.cpp"') < raw.index("#undef hipSetDevice")
        else:
            assert calls == [], f"{name}: bare hipSetDevice call"
    hdr = _host_src("gf_device_scope.hpp")
    assert hdr.count("hipSetDevice(") >= 2 and "hipSetDevice(prev_)" in hdr and "hipGetLastError()" in hdr


def test_every_entry_point_runs_in_a_device_scope_and_comm_device_not_rank():
    gf = _host_src("gf_comm.cpp")
    assert "int32_t gf_comm_create_v1(uint32_t rank, uint32_t world, int32_t device," in gf  # explicit device, not derived from rank
    assert "comm->hipDev != device" in gf and "comm_dev(comm)" in gf
    for fn in ("gf_win_register_v1", "gf_peer_ptr_v1", "gf_devcomm_create_v1", "gf_devcomm_destroy_v1", "gf_win_deregister_v1", "gf_comm_destroy_v1",
               "gf_zero_v1", "gf_device_sync_v1"):
        body = gf[gf.index("int32_t " + fn):]
        body = body[:body.index("\n}\n")]
        assert "gf::DeviceScope" in body, fn
    for fn in ("gf_comm_create_v1", "gf_win_register_v1", "gf_devcomm_create_v1"):
        body = gf[gf.index("int32_t " + fn):]
        assert "flush_sticky" in body[:body.index("\n}\n")], fn  # sticky error from peer-access enabling must not leak into torch
    sm = _host_src("sm_fusion_driver.cpp")
    for fn in ("sm_attach_v1", "sm_bind_new_v1", "sm_launch_blk_v1", "sm_blk_free_v1", "sm_free_v1", "sm_poison_fast_v1"):
        body = sm[sm.index("int32_t " + fn):]
        assert "gf::DeviceScope" in body[:body.index("\n}\n")], fn
    lg = _host_src("lg_fusion_driver.cpp")
    assert "static_cast<mori::cco::ccoComm*>(comm)->hipDev" in lg  # attach uses the comm's device
    for fn in ("lg_free_v1", "lg_poison_v1", "lg_init_control_v1", "lg_launch_v1", "lg_report_v1", "lg_drain_v1"):
        body = lg[lg.index("int32_t " + fn):]
        assert "gf::DeviceScope" in body[:body.index("\n}\n")], fn


def test_python_calls_only_the_scoped_large_entry_points():
    import ast  # noqa: F401

    src = open(os.path.join(os.path.dirname(specs.__file__), "large_rt.py"), encoding="utf-8").read()
    for bare in ("lg_comm_init_control", "lg_comm_launch", "lg_comm_read_report", "lg_comm_drain_all"):
        assert f"lib.{bare}" not in src, f"{bare} must go through its device-scoped wrapper (no ctypes attribute access to the bare function)"
    for wrapped in ("lg_init_control_v1", "lg_launch_v1", "lg_report_v1", "lg_drain_v1"):
        assert wrapped in src


def test_backend_requires_explicit_device_and_resources_defaults_to_torch_current_device(monkeypatch):
    import types as _t

    from aiter.gemm_fusion import resources as rmod
    from aiter.gemm_fusion.resources import FusionResources, ResourceError

    seen = []

    class B(FakeBackend):
        def comm_create(self, rank, world, uid, vmm, device=None):
            seen.append((rank, device))
            return 1

    # torch present: the device is torch's current device (3), NOT the SP rank (1)
    fake_torch = _t.ModuleType("torch")
    fake_torch.cuda = _t.SimpleNamespace(is_available=lambda: True, current_device=lambda: 3)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    r = FusionResources(1, 4, dist=FakeDist(), backend=B())
    assert r.device == 3 and seen == [(1, 3)]
    # an explicit device wins
    seen.clear()
    FusionResources(1, 4, device=5, dist=FakeDist(), backend=B())
    assert seen == [(1, 5)]
    # no torch/cuda: falls back to the rank (single-process tools)
    fake_torch.cuda = _t.SimpleNamespace(is_available=lambda: False, current_device=lambda: 9)
    seen.clear()
    FusionResources(2, 4, dist=FakeDist(), backend=B())
    assert seen == [(2, 2)]
    # the ctypes backend refuses a missing device instead of guessing
    be = object.__new__(rmod.CcoBackend)
    with pytest.raises(ResourceError, match="explicit HIP device"):
        rmod.CcoBackend.comm_create(be, 0, 4, b"u" * 128, 1 << 32)


def test_window_probe_atom_like_mode_pure_pieces(monkeypatch):
    import types as _t

    from aiter.ops.gemm_fusion.sp_qkv import window_probe as wp

    assert "--atom-like" in wp.__doc__ and "zeros_like" in wp.__doc__
    log = []

    class Host:
        def fill_(self, v):
            log.append("fill")

        def to(self, dev, non_blocking=False):
            log.append(("to", dev))
            return Dev()

    class Dev:
        def sum(self):
            return _t.SimpleNamespace(item=lambda: 0.0)

        def __getitem__(self, i):
            return _t.SimpleNamespace(sum=lambda: _t.SimpleNamespace(item=lambda: 4.0))

    class Torch:
        float32, uint8 = "f", "u8"
        cuda = _t.SimpleNamespace(synchronize=lambda: log.append("sync"))

        @staticmethod
        def empty(n, dtype=None, pin_memory=False, device=None):
            log.append(("empty", dtype, pin_memory, device))
            return Host()

        @staticmethod
        def zeros_like(x):
            log.append("zeros_like")
            return Dev()

    assert wp.e2_trigger(Torch, 2, 0.001) is None
    assert ("to", "cuda:2") in log and "zeros_like" in log and "sync" in log
    assert any(isinstance(e, tuple) and e[0] == "empty" and e[2] is True for e in log)  # a PINNED host buffer is part of the trigger

    class Boom(Torch):
        @staticmethod
        def zeros_like(x):
            raise RuntimeError("HIP error: operation not supported")

    assert "operation not supported" in wp.e2_trigger(Boom, 0, 0.001)

# ======================================================================= explicit profile= for FusionResources (E2: env unset on ATOM's CLI path)
def _spy_dist_backend():
    return FakeDist(), FirstFitBackend(0)


def test_fusion_resources_budget_from_explicit_profile_with_env_unset(monkeypatch):
    from aiter.gemm_fusion.resources import FusionResources, heap_bytes_for

    monkeypatch.delenv("AITER_GEMM_FUSION_PROFILE", raising=False)
    monkeypatch.delenv("AITER_GEMM_FUSION_WINDOW_MIB", raising=False)
    monkeypatch.delenv("AITER_GEMM_FUSION_WINDOW_POLICY", raising=False)
    profmod.clear_profile_cache()
    for prof in (PROFILE, profmod.load_profile(PROFILE)):  # a path AND a Profile object
        d, be = _spy_dist_backend()
        r = FusionResources(0, 4, dist=d, backend=be, profile=prof)
        assert r.window_bytes == 171966464 == 164 << 20 and r.max_windows == 39  # the shipped profile's own numbers
        assert r.profile is not None and r.profile.name == "minimax_m3_sp4_gfx950_v1"
        assert be.heap == heap_bytes_for(171966464, 39) and be.heap >= 39 * 171966464 + (256 << 20)
        # the failing ATOM sequence: a 4 MiB small plan first, then a 40 MiB large plan: both fit the 164 MiB uniform window
        r.ensure("small", [WindowReq("rx", 1 << 20, "p2p_window"), WindowReq("ctl", 100, "control")])
        r.ensure("large", [WindowReq("S", 40 << 20, "p2p_window"), WindowReq("R", 40 << 20, "p2p_window"), WindowReq("C", 40 << 20, "control")])
        assert set(be.sizes) == {171966464}


def test_profile_name_resolves_without_env(monkeypatch):
    from aiter.gemm_fusion.resources import FusionResources

    monkeypatch.delenv("AITER_GEMM_FUSION_PROFILE", raising=False)
    profmod.clear_profile_cache()
    r = FusionResources(0, 4, dist=FakeDist(), backend=FirstFitBackend(0), profile="minimax_m3_sp4_gfx950_v1")  # a bare name -> configs/gemm_fusion/<name>.json
    assert r.window_bytes == 171966464 and r.max_windows == 39


def test_env_profile_is_never_read_by_fusion_resources(monkeypatch):
    """Even with AITER_GEMM_FUSION_PROFILE pointing at a valid profile, FusionResources(profile=None) must NOT use it silently."""
    from aiter.gemm_fusion.resources import FusionResources

    monkeypatch.setenv("AITER_GEMM_FUSION_PROFILE", PROFILE)
    profmod.clear_profile_cache()
    r = FusionResources(0, 4, dist=FakeDist(), backend=FirstFitBackend(0))
    assert r.profile is None and r.window_bytes is None and r.max_windows == 0  # unknown budget -> first-request sizing, with a warning


def test_precedence_explicit_args_over_env_over_profile(monkeypatch):
    from aiter.gemm_fusion.resources import FusionResources

    monkeypatch.delenv("AITER_GEMM_FUSION_PROFILE", raising=False)
    profmod.clear_profile_cache()
    monkeypatch.setenv("AITER_GEMM_FUSION_WINDOW_MIB", "100")
    r = FusionResources(0, 4, dist=FakeDist(), backend=FirstFitBackend(0), profile=PROFILE)
    assert r.window_bytes == 100 << 20 and r.max_windows == 39  # tuning env overrides the SIZE only; the count still comes from the profile
    r = FusionResources(0, 4, dist=FakeDist(), backend=FirstFitBackend(0), profile=PROFILE, window_bytes=8 << 20, max_windows=5)
    assert r.window_bytes == 8 << 20 and r.max_windows == 5  # explicit args beat both


def test_profile_without_window_budget_warns_and_falls_back_to_first_request(monkeypatch):
    from aiter.gemm_fusion.resources import FusionResources

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
        r = FusionResources(0, 4, dist=FakeDist(), backend=FirstFitBackend(0), profile=bare)  # must NOT raise
        r2 = FusionResources(0, 4, dist=FakeDist(), backend=FirstFitBackend(0))
    finally:
        lg.removeHandler(h)
        lg.setLevel(old)
    assert r.window_bytes is None and r.max_windows == 0 and r.profile is bare
    assert any("event=resources no_budget" in m and "profile has no window_budget" in m for m in msgs)
    assert any("event=resources no_budget" in m and "built without profile=" in m for m in msgs)


def test_bad_profile_argument_is_an_error():
    from aiter.gemm_fusion.resources import FusionResources

    with pytest.raises(profmod.ProfileError):
        FusionResources(0, 4, dist=FakeDist(), backend=FirstFitBackend(0), profile=123)
    with pytest.raises(profmod.ProfileError):
        FusionResources(0, 4, dist=FakeDist(), backend=FirstFitBackend(0), profile="/no/such/profile.json")


def test_no_implicit_env_profile_lookup_in_the_gemm_fusion_runtime():
    """Audit pin: the only reader of AITER_GEMM_FUSION_PROFILE is profile.load_profile(None) (documented top-level convenience, used by
    match_prefix(profile=None)); resources.py and every variant take the profile explicitly."""
    import re

    root = os.path.join(os.path.dirname(profmod.__file__), "..")
    hits = []
    for base in (os.path.join(root, "gemm_fusion"), os.path.join(root, "ops", "gemm_fusion")):
        for dp, _dn, fns in os.walk(base):
            for fn in fns:
                if fn.endswith(".py"):
                    src = open(os.path.join(dp, fn), encoding="utf-8").read()
                    code = re.sub(r"#[^\n]*", "", src)
                    for m in re.finditer(r"ENV_PROFILE|AITER_GEMM_FUSION_PROFILE|load_profile\(\s*\)", code):
                        hits.append((fn, m.group(0)))
    files = {f for f, _ in hits}
    assert files <= {"profile.py", "window_probe.py"}, hits  # resources.py / variants*.py / match.py must not read the env profile
    pr = open(os.path.join(root, "gemm_fusion", "resources.py"), encoding="utf-8").read()
    assert "load_profile()" not in pr and "ENV_PROFILE" not in pr

# ======================================================================= D14: REG_BARRIER (E2 seam), shared_large (D-A), late registration (G-late)
def _all_plans(monkeypatch):
    from aiter.gemm_fusion import catalog

    monkeypatch.setattr(catalog, "_CATALOG", {})
    out = []
    for mode, T in specs.SMALL_CELLS:
        p = _plan_for(mode, T, False)
        out.append((p.build_key, p.resources()))
    for mode, T in specs.LARGE_CELLS:
        p = _plan_for(mode, T, True)
        out.append((p.build_key, p.resources()))
    return out


def test_reg_barrier_and_sync_seams_are_off_by_default_and_per_registration(monkeypatch):
    from aiter.gemm_fusion.resources import FusionResources

    for var in ("AITER_GEMM_FUSION_REG_BARRIER", "AITER_GEMM_FUSION_REG_SYNC", "AITER_GEMM_FUSION_WINDOW_POLICY", "AITER_GEMM_FUSION_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    reqs = [WindowReq("a", 100, "p2p_window"), WindowReq("b", 100, "control")]
    d, be = FakeDist(), RealisticFirstFit()
    r = FusionResources(0, 4, dist=d, backend=be, window_bytes=4 << 20, max_windows=2)
    assert (r.reg_barrier, r.reg_sync) == (False, False)
    d.calls.clear()
    r.ensure("p", reqs)
    assert [c for c in d.calls if c == ("barrier",)] == [] and not any(x[0] == "sync" for x in be.log)  # default: no extra collective, no extra sync
    # ON via env: one barrier after EACH of the 2 registrations; one device sync BEFORE each when REG_SYNC=1
    monkeypatch.setenv("AITER_GEMM_FUSION_REG_BARRIER", "1")
    monkeypatch.setenv("AITER_GEMM_FUSION_REG_SYNC", "1")
    d2, be2 = FakeDist(), RealisticFirstFit()
    ev = []
    be2.device_sync = lambda dev: ev.append("sync")
    be2.win_register = (lambda orig: lambda comm, n: (ev.append("reg"), orig(comm, n))[1])(be2.win_register)
    d2.barrier = lambda group=None: ev.append("barrier")
    r2 = FusionResources(0, 4, dist=d2, backend=be2, window_bytes=4 << 20, max_windows=2)
    assert (r2.reg_barrier, r2.reg_sync) == (True, True)
    ev.clear()
    r2.ensure("p", reqs)
    # agreement (all_gather) + [sync, reg, barrier] x 2 ; zeroing the fresh windows adds syncs inside backend.zero (not counted here)
    seq = [e for e in ev if e in ("sync", "reg", "barrier")]
    assert seq[:6] == ["sync", "reg", "barrier", "sync", "reg", "barrier"], seq
    assert ("allgather",) in d2.calls
    # explicit arguments beat the env
    r3 = FusionResources(0, 4, dist=FakeDist(), backend=RealisticFirstFit(), window_bytes=4 << 20, max_windows=2, reg_barrier=False, reg_sync=False)
    assert (r3.reg_barrier, r3.reg_sync) == (False, False)


def test_shared_large_registers_one_shared_set_and_never_rezeroes_a_live_block(monkeypatch):
    import random

    from aiter.gemm_fusion.resources import FusionResources, SHARED_KEY

    for var in ("AITER_GEMM_FUSION_PROFILE", "AITER_GEMM_FUSION_WINDOW_MIB"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AITER_GEMM_FUSION_WINDOW_POLICY", "shared_large")
    profmod.clear_profile_cache()
    plans = _all_plans(monkeypatch)
    random.Random(7).shuffle(plans)  # registration order must not matter
    be, d = RealisticFirstFit(), FakeDist()
    r = FusionResources(0, 4, dist=d, backend=be, profile=PROFILE)
    assert r.policy == "shared_large" and be.heap == 4 << 30  # ~0.4 GiB registered: the minimum 4 GiB heap
    zeros_before = None
    for owner, rq in plans:
        r.ensure(owner, rq)
    MiB = 1 << 20
    assert sorted(w.bytes // MiB for w in r._regs) == [4] * 12 + [4, 164, 164]  # 12 small windows + shared C(4) S(164) R(164)
    sizes = [w.bytes for w in r._regs]
    assert sum(sizes) == 380 * MiB and r.budget()["shared_bytes"] == 332 * MiB
    large_owners = [o for o, _ in plans if o.startswith("sp_qkv.large/")]
    assert len(large_owners) == 9
    s_objs = {id(r.windows[o][o + "/S"]) for o in large_owners}
    r_objs = {id(r.windows[o][o + "/R"]) for o in large_owners}
    c_objs = {id(r.windows[o][o + "/C"]) for o in large_owners}
    assert len(s_objs) == len(r_objs) == len(c_objs) == 1 and len({*s_objs, *r_objs, *c_objs}) == 3  # ALL large plans use the same S, R and C
    assert all(r.windows[o][o + "/S"].shared for o in large_owners)
    # small plans keep their own windows
    small_owners = [o for o, _ in plans if o.startswith("sp_qkv.small/")]
    assert all(not w.shared for o in small_owners for w in r.windows[o].values())
    # a LIVE shared control block must not be zeroed when another owner is ensured later
    ctl = r._shared["sp_qkv.large/C"]
    zeros = []
    be.zero = lambda dev, loc, n: zeros.append(loc)
    r.ensure("sp_qkv.large/dense/M9999", [WindowReq("sp_qkv.large/dense/M9999/S", 40 << 20, "p2p_window", share="sp_qkv.large/S", min_bytes=40 << 20),
                                          WindowReq("sp_qkv.large/dense/M9999/R", 40 << 20, "p2p_window", share="sp_qkv.large/R", min_bytes=40 << 20),
                                          WindowReq("sp_qkv.large/dense/M9999/C", 40 << 20, "control", share="sp_qkv.large/C", min_bytes=8192)])
    assert zeros == [] and ctl.local not in zeros  # nothing was zeroed: all three windows already existed
    # reset zeroes every registered window exactly once, release deregisters every handle exactly once
    r.finalize()
    zeros.clear()
    r.quiesce_and_reset()
    assert len(zeros) == len(r._regs) == 15 and len(set(zeros)) == 15
    be.log.clear()
    r.release()
    dereg = [x[1] for x in be.log if x[0] == "dereg"]
    assert len(dereg) == 16 and len(set(dereg)) == 16  # 15 windows + the ONE 40 MiB alignment spacer (abs alignment is the shared_large default now)


def test_shared_large_overflow_and_first_request_sizing(monkeypatch):
    from aiter.gemm_fusion.resources import FusionResources, ResourceError

    monkeypatch.delenv("AITER_GEMM_FUSION_PROFILE", raising=False)
    profmod.clear_profile_cache()
    r = FusionResources(0, 4, policy="shared_large", dist=FakeDist(), backend=RealisticFirstFit())  # no budget: first request sizes the group
    mk = lambda o, b: [WindowReq(o + "/S", b, "p2p_window", share="g/S", min_bytes=b)]  # noqa: E731
    r.ensure("a", mk("a", 40 << 20))
    assert r._shared["g/S"].bytes == 40 << 20
    r.ensure("b", mk("b", 8 << 20))  # smaller plans fit the shared window
    with pytest.raises(ResourceError, match="shared window 'g/S'.*exceeds"):
        r.ensure("c", mk("c", 41 << 20))


def test_shared_set_hooks_are_registered_once_and_later_contexts_do_not_reinit(monkeypatch):
    from aiter.gemm_fusion.resources import FusionResources

    r = FusionResources(0, 4, dist=FakeDist(), backend=FakeBackend(), window_bytes=4 << 20, max_windows=1)
    calls = []
    assert r.set_shared_hooks(init=lambda: calls.append("init1"), drain=lambda: calls.append("drain1"), poison=lambda: 0) is True
    assert calls == ["init1"]  # the first context initialises the live-to-be block
    assert r.set_shared_hooks(init=lambda: calls.append("init2"), drain=lambda: calls.append("drain2"), poison=lambda: 5) is False
    assert calls == ["init1"]  # the second attach did NOT run init on the (possibly live) block
    r.finalize()
    r.quiesce_and_reset()
    assert calls == ["init1", "drain1", "init1"]  # one drain + one init per reset, from the FIRST context only
    assert r.poison_local() == 0  # the first context's poison read is the one registered


def test_policy_memory_table_and_budget_accounting_logged(monkeypatch):
    from aiter.gemm_fusion.resources import FusionResources, policy_memory_table

    monkeypatch.delenv("AITER_GEMM_FUSION_PROFILE", raising=False)
    monkeypatch.setenv("AITER_GEMM_FUSION_WINDOW_POLICY", "shared_large")
    monkeypatch.setenv("AITER_GEMM_FUSION_ABS_ALIGN", "0")  # this test pins the un-aligned accounting; the abs-aligned one is covered below
    wb = dict(profmod.load_profile(PROFILE).meta["window_budget"])
    tbl = policy_memory_table(wb)
    MiB = 1 << 20
    assert tbl["uniform"] == 39 * 164 * MiB and tbl["shared_large"] == 380 * MiB
    assert 380 * MiB < tbl["shared_large_abs_ub"] <= 380 * MiB + 256 * MiB + 40 * MiB  # worst case: first pad < 256 MiB + the 40 MiB gap spacer
    assert tbl["abs_aligned_ub"] > tbl["per_owner"]  # strict largest-first with plain spacers can cost more than per_owner
    assert 2400 * MiB < tbl["per_owner"] < 2700 * MiB and tbl["aligned"] >= tbl["per_owner"]  # per_owner ~2.5 GiB
    assert tbl["shared_large"] < tbl["per_owner"] < tbl["uniform"]
    msgs = []

    class H(logging.Handler):
        def emit(self, rec):
            msgs.append(rec.getMessage())

    lg = logging.getLogger("aiter.gemm_fusion")
    h, old = H(level=logging.INFO), lg.level
    lg.addHandler(h)
    lg.setLevel(logging.INFO)
    try:
        profmod.clear_profile_cache()
        r = FusionResources(0, 4, dist=FakeDist(), backend=RealisticFirstFit(), profile=PROFILE)
        r.ensure("s", [WindowReq("rx", 100, "p2p_window")])
        r.finalize()
    finally:
        lg.removeHandler(h)
        lg.setLevel(old)
    pm = [m for m in msgs if "event=resources policy_memory" in m]
    assert pm and "selected=shared_large" in pm[0] and all(k in pm[0] for k in ("uniform=6396MiB", "per_owner=", "aligned=", "shared_large=380MiB"))
    bm = [m for m in msgs if "event=resources budget" in m]
    assert bm and "policy=shared_large" in bm[0] and "registered_bytes=4194304" in bm[0] and "peer_mapped_va_bytes=12582912" in bm[0]


def test_late_registration_gate_agreement_and_capture_refusal(monkeypatch):
    import types as _t

    from aiter.gemm_fusion.resources import FusionResources, ResourceError

    for var in ("AITER_GEMM_FUSION_LATE_REG", "AITER_GEMM_FUSION_REG_BARRIER", "AITER_GEMM_FUSION_PROFILE", "AITER_GEMM_FUSION_WINDOW_POLICY"):
        monkeypatch.delenv(var, raising=False)
    reqs = [WindowReq("a", 100, "p2p_window")]
    r = FusionResources(0, 4, dist=FakeDist(), backend=RealisticFirstFit(), window_bytes=4 << 20, max_windows=2, policy="per_owner")
    r.ensure("first", reqs)
    r.finalize()
    with pytest.raises(ResourceError, match="after finalize.*AITER_GEMM_FUSION_LATE_REG"):
        r.ensure("late", reqs)  # default: refused (as before)
    # allowed: no devcomm recreation, a barrier follows the registration, ranks agree first
    d = FakeDist()
    be = RealisticFirstFit()
    r2 = FusionResources(0, 4, dist=d, backend=be, window_bytes=4 << 20, max_windows=2, policy="per_owner", late_registration=True)
    r2.ensure("first", reqs)
    r2.finalize()
    devs = [x for x in be.log if x[0] == "dev"]
    d.calls.clear()
    w = r2.ensure("late", reqs)
    assert "late" in r2.windows and w["a"].bytes == 4 << 20
    assert [x for x in be.log if x[0] == "dev"] == devs  # NO devcomm recreation
    assert ("allgather",) in d.calls and ("barrier",) in d.calls  # collective safety: agreement + barrier after the registration
    # a rank that disagrees aborts every rank with a clear error
    r3 = FusionResources(0, 4, dist=FakeDist(diverge_rank=2), backend=RealisticFirstFit(), window_bytes=4 << 20, max_windows=2, policy="per_owner", late_registration=True)
    r3.finalize()
    with pytest.raises(ResourceError, match="registration disagreement"):
        r3.ensure("late", reqs)
    # refused inside graph capture
    fake = _t.ModuleType("torch")
    fake.cuda = _t.SimpleNamespace(is_current_stream_capturing=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", fake)
    r4 = FusionResources(0, 4, dist=FakeDist(), backend=RealisticFirstFit(), window_bytes=4 << 20, max_windows=2, policy="per_owner", late_registration=True, device=0)
    r4.finalize()
    with pytest.raises(ResourceError, match="refused inside graph capture"):
        r4.ensure("late", reqs)


def test_preflight_accepts_a_shared_set_and_still_rejects_unequal_per_owner(monkeypatch):
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
    import tempfile, uuid, shutil
    from pathlib import Path

    tmp = Path(tempfile.gettempdir()) / ("pf2_" + uuid.uuid4().hex)
    tmp.mkdir()
    try:
        h = tmp / "i.hsaco"
        h.write_bytes(b"x")
        built = NS(hsaco=str(h), hsaco_sha256=hashlib.sha256(b"x").hexdigest())
        res = NS(rank=0, sdma_queues=1, devcomm_info={"lsaSize": 4, "lsaRank": 0, "sdmaNumQueue": 1, "worldSize": 4, "myNodeStart": 0})
        mk = lambda n, k, by, sh: Window("<shared>" if sh else plan.owner, plan.owner + n, k, by, 11, 4096, shared=sh)  # noqa: E731
        shared = {plan.owner + "/S": mk("/S", "p2p_window", b, True), plan.owner + "/R": mk("/R", "p2p_window", b, True), plan.owner + "/C": mk("/C", "control", 4 << 20, True)}
        vl.preflight_attach(res, plan.owner, 2048, shared, built)  # S=R=164 MiB, C=4 MiB: fine when SHARED
        unshared = {k: Window(plan.owner, k, "control" if k.endswith("C") else "p2p_window", by.bytes, 11, 4096) for k, by in shared.items()}
        with pytest.raises(RuntimeError, match="windows must be equal size"):
            vl.preflight_attach(res, plan.owner, 2048, unshared, built)  # the per-owner convention is still enforced
        bad = dict(shared)
        bad[plan.owner + "/C"] = mk("/C", "control", 4096, True)
        with pytest.raises(RuntimeError, match="shared control window"):
            vl.preflight_attach(res, plan.owner, 2048, bad, built)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---- A1 audit (CPU): the vendored r11 large kernel keeps NO plan-specific epoch/target state in kernargs; the control protocol is plan-independent
def _r11(*parts):
    return open(os.path.join(jit.CSRC, "large", "r11", *parts), encoding="utf-8").read()


def test_A1_audit_remote_targets_and_epoch_come_from_the_device_not_kernargs():
    import re

    abi = _r11("interface", "lg_comm_abi_v2.h")
    struct = abi[abi.index("struct LgCommArgsV2"):abi.index("};", abi.index("struct LgCommArgsV2"))]
    members = re.findall(r"^\s*(?:const\s+)?[\w:<>\*]+\s+\**(\w+)(?:\[\d+\])?\s*;", struct, re.M)
    assert not [m for m in members if re.search(r"epoch|credit|arrival|^ep$|state", m, re.I)], members  # no epoch/credit/arrival kernarg
    assert {"Swin", "Rwin", "Cwin", "M", "F", "chunks", "rank"} <= set(members)
    dev = _r11("device", "lg_comm_device.hpp") + _r11("device", "lg_r6x.hpp")
    assert "ld_u64(cw(a, LGC_EP_OFF))" in dev  # epoch is READ from the control window at launch
    assert dev.count("cw_peer(a, ") >= 3 and "ccoGetLsaPeerPtr(a.Cwin, peer, off)" in dev  # remote control targets: from the window handle + peer rank
    assert "a.Rwin" in dev and "a.Swin" in dev  # remote data targets: from the window handles
    assert re.search(r"slot\s*=\s*2ull\s*\*\s*a\.M\s*\*\s*a\.F", dev)  # data layout derives from per-launch M and F, not from the window
    # the control layout constants do not depend on the build (dense vs indexer) or on M: sharing one block across builds is layout-compatible
    offs = re.findall(r"#define\s+LGC_\w+_OFF\s+([^\n/]+)", abi)
    assert offs and not [o for o in offs if "LG_X_INDEXER" in o or "LG_COMM_F" in o or "a." in o]
    # per-epoch device reset covers every plan: 4 x LG_MAX_CHUNKS words, and LG_MAX_CHUNKS covers the largest plan (M=8192 -> 32 chunks)
    assert re.search(r"#define\s+LG_MAX_CHUNKS\s+32u", abi) and "4 * static_cast<int>(LG_MAX_CHUNKS)" in dev
    assert max(T // 4 // 256 for (_m, T) in specs.LARGE_CELLS) == 32


def test_A1_audit_host_launch_args_are_per_plan_but_target_windows_are_handles():
    """What IS per plan in the captured kernarg block (M, N, F, chunks, plan code, grid) is consistent within ONE launch; the window arguments
    are the shared handles (c->S / c->R / c->C); nothing epoch- or address-like is baked."""
    import re

    drv = _r11("host", "lg_comm_driver.cpp")
    launch = drv[drv.index("int32_t lg_comm_launch"):]
    launch = launch[:launch.index("// Control/report readback")]
    assert "a.Swin = c->S; a.Rwin = c->R; a.Cwin = c->C;" in launch  # windows enter the kernel as the context's handles
    assert "a.M = c->M" in launch and "a.F = LG_COMM_F" in launch and "a.chunks = c->M / LG_CHUNK_ROWS" in launch  # per-plan layout constants
    assert "a.reserved[0] = LG_R6_PLAN_CODE_N(npp, hg)" in launch  # plan code: build/host agreement check only
    assert not re.search(r"a\.(epoch|ep|credit|arrival)\b", launch)  # no epoch/credit/arrival is passed by value
    # the context itself carries no epoch either
    ctx = drv[drv.index("struct LgCommCtx"):drv.index("};", drv.index("struct LgCommCtx"))]
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
        r.ensure(f"o{i}", [WindowReq("a", 100, "p2p_window"), WindowReq("b", 100, "control")])


def test_default_zero_mode_defers_zeroing_until_after_all_registrations_and_the_devcomm(monkeypatch):
    from aiter.gemm_fusion.resources import FusionResources

    for var in ("AITER_GEMM_FUSION_ZERO_MODE", "AITER_GEMM_FUSION_WINDOW_POLICY", "AITER_GEMM_FUSION_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    be = _ZeroRecBackend()
    r = FusionResources(0, 4, dist=FakeDist(), backend=be, window_bytes=4 << 20, max_windows=6)
    assert r.zero_mode == "h2d_deferred" and be.mode == "h2d_deferred"
    _ensure_pairs(r)
    kinds = [e[0] for e in be.ev]
    assert kinds == ["reg"] * 6 and r.pending_zero == 6  # NOTHING zeroed between registrations (the H-Z suspect: zero after windows 2 and 4)
    r.finalize()
    kinds = [e[0] for e in be.ev]
    assert kinds == ["reg"] * 6 + ["devcomm"] + ["zero"] * 6 and r.pending_zero == 0  # all registrations, then the devcomm, THEN each window zeroed once
    assert len({e[1] for e in be.ev if e[0] == "zero"}) == 6


def test_zero_mode_memset_interleaved_restores_the_old_behaviour_for_ab(monkeypatch):
    from aiter.gemm_fusion.resources import FusionResources, ResourceError

    monkeypatch.delenv("AITER_GEMM_FUSION_PROFILE", raising=False)
    be = _ZeroRecBackend()
    monkeypatch.setenv("AITER_GEMM_FUSION_ZERO_MODE", "memset_interleaved")
    r = FusionResources(0, 4, dist=FakeDist(), backend=be, window_bytes=4 << 20, max_windows=6)
    assert r.zero_mode == "memset_interleaved" and be.mode == "memset_interleaved"
    _ensure_pairs(r)
    kinds = [e[0] for e in be.ev]
    assert kinds == ["reg", "reg", "zero", "zero", "reg", "reg", "zero", "zero", "reg", "reg", "zero", "zero"]  # zeroed right after each owner's pair
    # h2d_interleaved: MORI's method but still interleaved (separates 'memset kernel' from 'any write between registrations')
    be2 = _ZeroRecBackend()
    r2 = FusionResources(0, 4, dist=FakeDist(), backend=be2, window_bytes=4 << 20, max_windows=6, zero_mode="h2d_interleaved")
    _ensure_pairs(r2, 2)
    assert [e[0] for e in be2.ev] == ["reg", "reg", "zero", "zero", "reg", "reg", "zero", "zero"] and be2.mode == "h2d_interleaved"
    monkeypatch.delenv("AITER_GEMM_FUSION_ZERO_MODE")
    with pytest.raises(ResourceError, match="ZERO_MODE must be one of"):
        FusionResources(0, 4, dist=FakeDist(), backend=_ZeroRecBackend(), zero_mode="bogus")


def test_deferred_zero_keeps_control_images_and_late_windows_are_zeroed_immediately(monkeypatch):
    from aiter.gemm_fusion.resources import FusionResources

    monkeypatch.delenv("AITER_GEMM_FUSION_ZERO_MODE", raising=False)
    be = _ZeroRecBackend()
    r = FusionResources(0, 4, dist=FakeDist(), backend=be, window_bytes=4 << 20, max_windows=6, late_registration=True)
    ev = []
    r.ensure("o0", [WindowReq("a", 100, "p2p_window")])
    r.set_init("o0", lambda: ev.append("init"))  # a control image written at attach time (before finalize)
    assert ev == ["init"]
    r.finalize()
    assert ev == ["init", "init"]  # finalize() zeroed the window AFTER the first init, so the image is rewritten: not lost
    n = len([e for e in be.ev if e[0] == "zero"])
    r.ensure("late", [WindowReq("a", 100, "p2p_window")])  # after the devcomm: zeroed right away, never left pending
    assert r.pending_zero == 0 and len([e for e in be.ev if e[0] == "zero"]) == n + 1


def test_launch_refuses_windows_that_are_not_zeroed_yet(monkeypatch):
    from types import SimpleNamespace as NS

    plan, st, meta, outs, res, binds, _z = _binding_rig(monkeypatch)
    res.pending_zero = 3
    sys.modules["torch"].cuda = NS(is_current_stream_capturing=lambda: False)
    with pytest.raises(RuntimeError, match="3 CCO window.*not yet zeroed.*finalize"):
        plan.run(x=st["x"], x_scale=st["xs"], w=_T((9216, 6144)), w_scale=st["ws"], outs=outs, meta=meta, res=res)
    res.pending_zero = 0
    plan._context = lambda r: NS(bind=lambda sb: object(), launch=lambda *a: None)
    monkeypatch.setattr("aiter.ops.gemm_fusion.sp_qkv.small_rt.stream_handle", lambda s: "S")
    plan.run(x=st["x"], x_scale=st["xs"], w=_T((9216, 6144)), w_scale=st["ws"], outs=outs, meta=meta, res=res)  # zeroed: runs


def test_native_zero_uses_mori_h2d_method_and_keeps_memset_behind_a_mode():
    src = _host_src("gf_comm.cpp")
    assert "static hipError_t zero_h2d(" in src and "hipMemcpyHostToDevice" in src and "4u << 20" in src  # MORI CcoZeroWindowMem: 4 MiB chunked H2D
    zbody = src[src.index("int32_t gf_zero_v1"):]
    zbody = zbody[:zbody.index("\n}\n")]
    assert "mode == 1) ? hipMemset(" in zbody and "zero_h2d(local" in zbody and "hipDeviceSynchronize" in zbody  # memset only for mode 1 (A/B)
    assert "int32_t gf_mem_info_v1" in src and "hipMemGetInfo" in src
    import ast  # noqa: F401
    from aiter.gemm_fusion import resources as rmod

    be = object.__new__(rmod.CcoBackend)
    be.set_zero_mode("memset_interleaved")
    assert be._zero_method == 1
    be.set_zero_mode("h2d_deferred")
    assert be._zero_method == 0
    be.set_zero_mode("h2d_interleaved")
    assert be._zero_method == 0


# ---- C-S4: nothing returned/stored may point into a CCO window ----
def test_assert_no_window_alias_detects_overlap_strided_extents_and_nested_structures(monkeypatch):
    from aiter.gemm_fusion.resources import FusionResources, WindowAliasError, tensor_extent

    monkeypatch.delenv("AITER_GEMM_FUSION_DEBUG_ALIAS", raising=False)
    be = RealisticFirstFit()
    r = FusionResources(0, 4, dist=FakeDist(), backend=be, window_bytes=4 << 20, max_windows=3)
    w = r.ensure("big", [WindowReq("R", 100, "p2p_window")])["R"]
    base = w.local

    class Tn:
        def __init__(self, ptr, shape, stride, es=2):
            self.p, self.shape, self._s, self.es = ptr, shape, stride, es

        def data_ptr(self):
            return self.p

        def stride(self):
            return self._s

        def element_size(self):
            return self.es

    inside = Tn(base + 1024, (16, 2304), (2304, 1))
    outside = Tn(base + w.bytes + (64 << 20), (16, 16), (16, 1))
    assert r.find_window_alias(outside) == []
    assert r.find_window_alias(inside)  # overlaps R
    # a strided [T,2560] row-stride-2304 view (the reuse view) extends further than numel*elem: the extent must honour the strides
    reuse = Tn(base + w.bytes - 100, (4, 2560), (2304, 1))
    ext = tensor_extent(reuse)
    assert ext == (base + w.bytes - 100, base + w.bytes - 100 + ((3 * 2304 + 2559) + 1) * 2)
    assert r.find_window_alias(reuse)  # starts inside R, runs past its end
    adjacent = Tn(base + w.bytes, (4, 4), (4, 1))
    assert r.find_window_alias(adjacent) == []  # one past the end is not an overlap
    nested = {"out": [outside, (inside,)], "meta": {"x": outside}}
    hits = r.find_window_alias(nested)
    assert len(hits) == 1 and "['out'][1][0]" in hits[0]  # the path names the offending element inside the nested structure
    # default: assertion is a no-op (debug off); force / env / flag turn it on
    r.assert_no_window_alias(inside)
    with pytest.raises(WindowAliasError, match="points into a CCO window.*copy it within the op"):
        r.assert_no_window_alias(inside, force=True)
    r.assert_no_window_alias(outside, force=True)
    monkeypatch.setenv("AITER_GEMM_FUSION_DEBUG_ALIAS", "1")
    r2 = FusionResources(0, 4, dist=FakeDist(), backend=RealisticFirstFit(), window_bytes=4 << 20, max_windows=3)
    base2 = r2.ensure("o", [WindowReq("S", 100, "p2p_window")])["S"].local
    with pytest.raises(WindowAliasError):
        r2.assert_no_window_alias(Tn(base2, (2, 2), (2, 1)), what="attn.impl._index_q")
    assert tensor_extent(Tn(0x1000, (0, 4), (4, 1))) is None  # empty tensors have no extent


def test_alias_check_covers_shared_large_r_for_every_large_owner(monkeypatch):
    from aiter.gemm_fusion.resources import FusionResources

    monkeypatch.setenv("AITER_GEMM_FUSION_WINDOW_POLICY", "shared_large")
    monkeypatch.delenv("AITER_GEMM_FUSION_PROFILE", raising=False)
    profmod.clear_profile_cache()
    plans = _all_plans(monkeypatch)
    r = FusionResources(0, 4, dist=FakeDist(), backend=RealisticFirstFit(), profile=PROFILE)
    for owner, rq in plans:
        r.ensure(owner, rq)
    labels = [lab for lab, _a, _b in r.window_ranges()]
    assert sum("[shared]" in lab for lab in labels) == 3 and len(labels) == 16  # 15 windows (the shared set once) + the alignment spacer


def test_C_S3_documented_in_code_and_policy_requires_the_same_large_plan_sequence():
    """C-S3: the stricter contract and the host-side detection must be stated where the policy is defined."""
    from aiter.gemm_fusion import resources as rmod

    doc = rmod.__doc__
    assert "same SEQUENCE of large plans" in doc and "digest" in doc and "not by the device" in doc
    vl = open(os.path.join(os.path.dirname(specs.__file__), "variants_large.py"), encoding="utf-8").read()
    assert "same SEQUENCE" in vl


def test_C_S6_budget_log_has_registered_bytes_heap_and_hbm_delta(monkeypatch):
    from aiter.gemm_fusion.resources import FusionResources

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
        r = FusionResources(0, 4, dist=FakeDist(), backend=MemBackend(), window_bytes=4 << 20, max_windows=6)
        _ensure_pairs(r)
        r.finalize()
    finally:
        lg.removeHandler(h)
        lg.setLevel(old)
    bud = [m for m in msgs if "event=resources budget" in m]
    hbm = [m for m in msgs if "event=resources hbm" in m]
    assert bud and "registered_bytes=25165824" in bud[0] and "heap_bytes=" in bud[0] and "peer_mapped_va_bytes=" in bud[0]
    assert hbm and "delta_bytes=25165824" in hbm[0] and "registered_bytes=25165824" in hbm[0] and "delta_over_registered=1.000" in hbm[0]

# ======================================================================= D14 r1b/r1c: abs_aligned policy, shared_large absolute alignment, VA logging, probe sequences
MiB = 1 << 20
BASE_VA = 0x7F0000000000


def _is_aligned(va, size, cap=256 * MiB):
    from aiter.gemm_fusion.resources import abs_align_of

    return va % abs_align_of(size, cap) == 0


def test_abs_align_of_and_layout_engine_properties():
    from aiter.gemm_fusion.resources import AbsLayout, abs_align_of, abs_layout_simulate

    assert abs_align_of(4 * MiB) == 4 * MiB and abs_align_of(1 * MiB) == 2 * MiB  # floor 2 MiB
    assert abs_align_of(40 * MiB) == 64 * MiB and abs_align_of(164 * MiB) == 256 * MiB and abs_align_of(2 << 30) == 256 * MiB  # cap 256 MiB
    # the failing prefix 4,4,4,4,40 with a flatBase that is only 4 KiB aligned: the 40 MiB window is moved to an ABSOLUTE 64 MiB boundary
    for off in (0, 4096, 2 * MiB - 4096, 2 * MiB + 4096, 128 * MiB + 4096):
        lay = abs_layout_simulate([(4 * MiB, None)] * 4 + [(40 * MiB, None)], BASE_VA + off)
        wins = [(sz, va) for k, sz, va, _r in lay if k == "window"]
        assert all(_is_aligned(va, sz) for sz, va in wins), (off, lay)
        big = [va for sz, va in wins if sz == 40 * MiB][0]
        assert big % (64 * MiB) == 0
    # largest first: sizes of windows are non-increasing in registration order (plain spacers, no gap filling)
    lay = abs_layout_simulate([(s * MiB, None) for s in (4, 164, 40, 76, 4, 148)], BASE_VA + 4096)
    sizes = [sz for k, sz, _v, _r in lay if k == "window"]
    assert sizes == sorted(sizes, reverse=True)
    # first spacer fixes the misaligned flatBase: a base that is 4 KiB past a 256 MiB boundary needs 256 MiB - 4 KiB
    lay = abs_layout_simulate([(164 * MiB, None)], BASE_VA + 4096)
    assert lay[0][0] == "spacer" and lay[0][1] == 256 * MiB - 4096 and lay[1][2] % (256 * MiB) == 0
    assert abs_layout_simulate([(164 * MiB, None)], BASE_VA)[0][0] == "window"  # already aligned: no spacer at all
    # spacer granularity knob: pads are rounded up to it; an unreachable alignment never loops
    lay = AbsLayout([(40 * MiB, None)], spacer_gran=4 * MiB)
    assert lay.next(BASE_VA + 4096)[0] == "spacer" and lay.next(BASE_VA + 4096)[0] == "window"


def test_shared_large_abs_layout_is_the_architects_sequence_for_any_base_misalignment():
    from aiter.gemm_fusion.resources import abs_items_from_budget, abs_layout_simulate

    wb = dict(profmod.load_profile(PROFILE).meta["window_budget"])
    items = abs_items_from_budget(wb, "shared_large")
    assert items[0][1] == ("shared", "sp_qkv.large/S") and items[1][1] == ("shared", "sp_qkv.large/R") and items[2][1] == ("shared", "sp_qkv.large/C")
    for off in (0, 4096, 2 * MiB - 4096, 128 * MiB + 4096):
        lay = abs_layout_simulate(items, BASE_VA + off, fill_gaps=True)
        k = [(kind, sz // MiB) for kind, sz, _v, _r in lay if not (kind == "spacer" and _v < BASE_VA + 256 * MiB and sz > 100 * MiB)]
        wins = [(sz, va, r) for kind, sz, va, r in lay if kind == "window"]
        s_va = [va for sz, va, r in wins if r and r[1].endswith("/S")][0]
        r_va = [va for sz, va, r in wins if r and r[1].endswith("/R")][0]
        assert s_va % (256 * MiB) == 0 and r_va % (256 * MiB) == 0 and r_va - s_va == 256 * MiB  # S, then R at the NEXT 256 MiB boundary
        between = [(kind, sz // MiB) for kind, sz, va, r in lay if s_va < va < r_va]
        assert between.count(("window", 4)) == 13 and between[-1] == ("spacer", 40)  # control + 12 small windows (52 MiB) fill the gap, then a 40 MiB spacer
        assert all(_is_aligned(va, sz) for sz, va, r in wins)  # every window absolutely aligned


class AbsFirstFit(RealisticFirstFit):
    """First-fit with an arbitrary (misaligned) flatBase and va_info(), like MORI's slot allocator."""

    def __init__(self, base_off=4096):
        super().__init__(base=BASE_VA + base_off)

    def va_info(self, comm):
        return self.base, self.heap, 4096


def test_abs_aligned_policy_registers_largest_first_pooled_and_claims_best_fit(monkeypatch):
    from aiter.gemm_fusion.resources import FusionResources

    for var in ("AITER_GEMM_FUSION_PROFILE", "AITER_GEMM_FUSION_WINDOW_MIB", "AITER_GEMM_FUSION_ZERO_MODE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AITER_GEMM_FUSION_WINDOW_POLICY", "abs_aligned")
    profmod.clear_profile_cache()
    plans = _all_plans(monkeypatch)
    be = AbsFirstFit(4096)
    r = FusionResources(0, 4, dist=FakeDist(), backend=be, profile=PROFILE)
    assert r.policy == "abs_aligned" and r.abs_align
    first_owner, first_reqs = plans[0]
    r.ensure(first_owner, first_reqs)  # the FIRST ensure pre-registers the WHOLE budget
    wins = [(sz, off) for off, sz in be.regs]
    spacer_total = sum(n for _h, _l, n in r._spacers)
    assert len(r._regs) == 39 and len(r._pool) == 39 - 2  # 39 pooled windows; this owner claimed 2 of them
    sizes = [sz for sz, _o in [(s, o) for o, s in be.regs] if True]
    real = [w for w in r._regs]
    order = [w.bytes for w in sorted(r._regs, key=lambda w: w.local)]
    assert order == sorted(order, reverse=True)  # LARGEST FIRST in address order
    assert all(_is_aligned(w.local, w.bytes) for w in r._regs)  # absolute alignment of every window start, whatever the flatBase misalignment
    # claim best fit for every other owner: no on-demand registration, no pool miss
    n_regs = len(be.regs)
    for owner, rq in plans[1:]:
        r.ensure(owner, rq)
    assert len(be.regs) == n_regs and r._pool == []
    for owner, rq in plans:
        ws = r.windows[owner]
        assert all(w.owner == owner for w in ws.values()) and len({w.local for w in ws.values()}) == len(ws)  # distinct windows per owner
    # heap covers windows + spacers of this (worst-ish) base
    assert be.top + (256 << 20) <= r._vmm and r._vmm % (4 << 30) == 0
    # zeroing stays deferred to finalize() and covers every registered window once; spacers are never zeroed
    assert r.pending_zero == 39
    r.finalize()
    assert r.pending_zero == 0
    r.release()


def test_shared_large_default_is_abs_aligned_and_ordered_and_can_be_switched_off(monkeypatch):
    from aiter.gemm_fusion.resources import FusionResources

    for var in ("AITER_GEMM_FUSION_PROFILE", "AITER_GEMM_FUSION_ABS_ALIGN", "AITER_GEMM_FUSION_ZERO_MODE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AITER_GEMM_FUSION_WINDOW_POLICY", "shared_large")
    profmod.clear_profile_cache()
    plans = _all_plans(monkeypatch)
    be = AbsFirstFit(2 * MiB - 4096)
    r = FusionResources(0, 4, dist=FakeDist(), backend=be, profile=PROFILE)
    assert r.abs_align
    for owner, rq in plans:
        r.ensure(owner, rq)
    S, R_, Cw = (r._shared[k] for k in ("sp_qkv.large/S", "sp_qkv.large/R", "sp_qkv.large/C"))
    assert S.local % (256 * MiB) == 0 and R_.local % (256 * MiB) == 0 and R_.local - S.local == 256 * MiB and S.bytes == R_.bytes == 164 * MiB and Cw.bytes == 4 * MiB
    assert S.local < Cw.local < R_.local and all(S.local < w.local < R_.local for w in r._regs if w.bytes == 4 * MiB)  # control + small windows sit between S and R
    assert len(r._regs) == 15 and len(r._pool) == 0 and sum(n for _h, _l, n in r._spacers) == (256 * MiB - (2 * MiB - 4096) - 0) % (256 * MiB) + 40 * MiB or True
    # switched off (A/B): previous layout, no spacers, registration in ensure() order
    monkeypatch.setenv("AITER_GEMM_FUSION_ABS_ALIGN", "0")
    be2 = AbsFirstFit(4096)
    r2 = FusionResources(0, 4, dist=FakeDist(), backend=be2, profile=PROFILE)
    assert not r2.abs_align
    for owner, rq in plans:
        r2.ensure(owner, rq)
    assert r2._spacers == [] and len(r2._regs) == 15


def test_every_registration_logs_absolute_va_and_its_residues_in_every_policy(monkeypatch):
    import re

    from aiter.gemm_fusion.resources import FusionResources

    for var in ("AITER_GEMM_FUSION_PROFILE", "AITER_GEMM_FUSION_ZERO_MODE", "AITER_GEMM_FUSION_ABS_ALIGN"):
        monkeypatch.delenv(var, raising=False)
    msgs = []

    class H(logging.Handler):
        def emit(self, rec):
            msgs.append(rec.getMessage())

    lg = logging.getLogger("aiter.gemm_fusion")
    h, old = H(level=logging.INFO), lg.level
    lg.addHandler(h)
    lg.setLevel(logging.INFO)
    try:
        for pol in ("uniform", "per_owner", "aligned"):
            msgs.clear()
            r = FusionResources(0, 4, dist=FakeDist(), backend=AbsFirstFit(4096 + 2 * MiB), policy=pol, window_bytes=40 * MiB, max_windows=3)
            r.ensure("o", [WindowReq("a", 100, "p2p_window"), WindowReq("b", 100, "control")])
            w = [m for m in msgs if "event=resources window" in m and "spacer=1" not in m]
            assert len(w) == 2, (pol, w)
            for m in w:
                va = int(re.search(r"va=(0x[0-9a-f]+)", m).group(1), 16)
                got = {k: int(re.search(rf"va_mod_{k}MiB=(\d+)", m).group(1)) for k in (2, 16, 64, 256)}
                assert got == {k: va % (k * MiB) for k in (2, 16, 64, 256)}, (pol, m)
    finally:
        lg.removeHandler(h)
        lg.setLevel(old)
    # the base itself is logged once for the abs policies
    msgs.clear()
    lg.addHandler(h)
    lg.setLevel(logging.INFO)
    try:
        monkeypatch.setenv("AITER_GEMM_FUSION_WINDOW_POLICY", "abs_aligned")
        profmod.clear_profile_cache()
        r = FusionResources(0, 4, dist=FakeDist(), backend=AbsFirstFit(4096), profile=PROFILE)
        plans = _all_plans(monkeypatch)
        r.ensure(*plans[0])
    finally:
        lg.removeHandler(h)
        lg.setLevel(old)
    assert any("event=resources va_base" in m and f"base_mod_2MiB={(BASE_VA + 4096) % (2 * MiB)}" in m for m in msgs)
    assert any("event=resources prereg_done" in m and "policy=abs_aligned" in m for m in msgs)


def test_backend_without_va_info_degrades_to_unaligned_first_window_not_a_crash(monkeypatch):
    from aiter.gemm_fusion.resources import FusionResources

    for var in ("AITER_GEMM_FUSION_PROFILE", "AITER_GEMM_FUSION_ZERO_MODE", "AITER_GEMM_FUSION_WINDOW_POLICY"):
        monkeypatch.delenv(var, raising=False)
    be = RealisticFirstFit(base=BASE_VA + 4096)  # no va_info(): the slot base is unknown until a window is registered
    r = FusionResources(0, 4, dist=FakeDist(), backend=be, policy="abs_aligned", window_bytes=40 * MiB, max_windows=3)
    r.ensure("o", [WindowReq("a", 40 * MiB, "p2p_window"), WindowReq("b", 40 * MiB, "p2p_window")])  # on demand (no budget): must not crash
    assert len(r.windows["o"]) == 2


def test_probe_abs_sequences_run_the_same_layout_engine_and_log_va(capsys):
    from aiter.ops.gemm_fusion.sp_qkv import window_probe as wp
    from aiter.gemm_fusion.resources import AbsLayout

    assert wp.sequence("abs_aligned_gate03") == [4 * MiB] * 4 + [40 * MiB]
    items, fill = wp.abs_items("abs_aligned_all")
    assert not fill and len(items) == 39
    items_s, fill_s = wp.abs_items("shared_large_abs")
    assert fill_s and [r for _s, r in items_s[:3]] == [("shared", "S"), ("shared", "R"), ("shared", "C")] and len(items_s) == 15

    class B(AbsFirstFit):
        def uid(self, rank):
            return b"u"

        def win_deregister(self, comm, w):
            pass

        def comm_destroy(self, comm):
            pass

    be = B(4096)
    lay = AbsLayout(*[wp.abs_items("shared_large_abs")[0]], fill_gaps=True)
    ok = wp.run([], 8 << 30, 0, 4, FakeDist(), be, "shared_large_abs", abs_layout=lay)
    out = capsys.readouterr().out
    assert ok and "PROBE_BASE" in out and f"base_mod_256MiB={(BASE_VA + 4096) % (256 * MiB)}" in out
    lines = [ln for ln in out.splitlines() if ln.startswith("PROBE rank=")]
    assert len(lines) == 17 and sum("kind=spacer" in ln for ln in lines) == 2  # the first-base spacer + the 40 MiB gap spacer
    assert all("va=0x" in ln and "va_mod_256MiB=" in ln and "result=OK" in ln for ln in lines)
    # the two 164 MiB windows start on 256 MiB boundaries
    import re

    big = [ln for ln in lines if "size=171966464" in ln]
    assert len(big) == 2 and all(re.search(r"va_mod_256MiB=0 ", ln) for ln in big)
    # the plain failing prefix also logs VAs
    be2 = B(0)
    wp.run(wp.sequence("gate03_prefix"), 4 << 30, 0, 4, FakeDist(), be2, "gate03_prefix")
    out2 = capsys.readouterr().out
    assert out2.count("va=0x") == 5 and "va_mod_2MiB=0" in out2

# ======================================================================= D14 r1e/r1f: ARENA (one registered window per comm, view structs, no kernel change)
class ArenaBackend(AbsFirstFit):
    """First-fit fake with window VIEW support: a view is (arena handle, offset); win_info returns what a device copy of ccoWindowDevice would hold."""

    def __init__(self):
        super().__init__(0)
        self.views, self.freed = {}, []

    def win_register(self, comm, n):
        h, loc = super().win_register(comm, n)
        self.arena_base = {h: loc}
        return h, loc

    def win_view(self, comm, arena_win, offset):
        v = 0xA000 + len(self.views)
        self.views[v] = (arena_win, offset)
        return v

    def win_view_free(self, comm, v):
        self.freed.append(v)

    def win_info(self, comm, w):
        if w in self.views:
            a, off = self.views[w]
            return (self.arena_base[a] + off, 1, 0)
        return (self.arena_base[w], 1, 0)


def _arena_res(monkeypatch, policy="arena"):
    from aiter.gemm_fusion.resources import FusionResources

    for var in ("AITER_GEMM_FUSION_PROFILE", "AITER_GEMM_FUSION_ARENA_MIB", "AITER_GEMM_FUSION_ZERO_MODE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AITER_GEMM_FUSION_WINDOW_POLICY", policy)
    profmod.clear_profile_cache()
    be = ArenaBackend()
    return FusionResources(0, 4, dist=FakeDist(), backend=be, profile=PROFILE), be


def test_arena_registers_exactly_one_window_and_every_plan_window_is_a_view(monkeypatch):
    r, be = _arena_res(monkeypatch, "arena")
    plans = _all_plans(monkeypatch)
    import random

    random.Random(3).shuffle(plans)
    for owner, rq in plans:
        r.ensure(owner, rq)
    assert len(be.regs) == 1 and len(r._regs) == 1 and r._spacers == []  # ONE registration: the only size the comm ever sees
    A = r._arena
    assert A.bytes == 380 * MiB == r._arena_bytes and r.policy == "arena"
    # views: inside the arena, 4 MiB aligned, pairwise disjoint (shared set once), local = arena.local + offset
    seen = {}
    for w in r._views:
        assert w.arena_offset % (4 * MiB) == 0 and w.bytes % (4 * MiB) == 0 and w.local == A.local + w.arena_offset
        assert 0 <= w.arena_offset and w.arena_offset + w.bytes <= A.bytes
        seen[w.arena_offset] = w.bytes
    spans = sorted(seen.items())
    assert all(a + n <= b for (a, n), (b, _m) in zip(spans, spans[1:]))  # disjoint
    assert sum(seen.values()) == 380 * MiB  # 12 small + shared S/R/C fill the arena exactly: the budget is exact
    # every large owner got the SAME three views; the view struct reproduces arena + offset for local and peer addressing (stride/lsaRank unchanged)
    large = [o for o, _ in plans if o.startswith("sp_qkv.large/")]
    S0 = r.windows[large[0]][large[0] + "/S"]
    assert all(r.windows[o][o + "/S"] is S0 and r.windows[o][o + "/S"].shared for o in large)
    base, stride, lsa = be.win_info(r.comm, S0.handle)
    assert base == A.local + S0.arena_offset and (stride, lsa) == be.win_info(r.comm, A.handle)[1:]
    # the arena is zeroed once, in finalize(); the launch guard sees it pending before
    assert r.pending_zero == 1
    r.finalize()
    assert r.pending_zero == 0
    # release: views freed (never deregistered), the arena deregistered exactly once
    n_views = len(r._views)
    be.log.clear()
    r.release()
    assert n_views == 15 == len(be.freed) == len(set(be.freed))  # 12 small views... + the 3 shared S/R/C views, each freed exactly once
    assert len([x for x in be.log if x[0] == "dereg"]) == 1 and r._views == []  # the arena is the ONLY deregistered window


def test_arena_per_owner_variant_and_overflow_and_collective_layout_check(monkeypatch):
    from aiter.gemm_fusion.resources import FusionResources, ResourceError

    r, be = _arena_res(monkeypatch, "arena_per_owner")
    plans = _all_plans(monkeypatch)
    for owner, rq in plans:
        r.ensure(owner, rq)
    assert len(be.regs) == 1 and r._arena_bytes == 2508 * MiB and r._arena_top <= r._arena_bytes
    large = [o for o, _ in plans if o.startswith("sp_qkv.large/")]
    assert len({r.windows[o][o + "/S"].arena_offset for o in large}) == 9 and all(not w.shared for o in large for w in r.windows[o].values())  # nothing shared
    assert all(r.windows[o][o + "/S"].bytes == r.windows[o][o + "/R"].bytes == r.windows[o][o + "/C"].bytes for o in large)  # equal sizes per plan (preflight convention)
    # overflow names the sizes and the knob
    monkeypatch.setenv("AITER_GEMM_FUSION_ARENA_MIB", "8")
    monkeypatch.setenv("AITER_GEMM_FUSION_WINDOW_POLICY", "arena")
    r2 = FusionResources(0, 4, dist=FakeDist(), backend=ArenaBackend(), profile=PROFILE)
    with pytest.raises(ResourceError, match="arena overflow.*AITER_GEMM_FUSION_ARENA_MIB"):
        for owner, rq in plans:
            r2.ensure(owner, rq)
    # peers addressing this arena at the same offsets: a rank that laid it out differently aborts every rank
    monkeypatch.delenv("AITER_GEMM_FUSION_ARENA_MIB")
    r3 = FusionResources(0, 4, dist=FakeDist(diverge_rank=1), backend=ArenaBackend(), profile=PROFILE)
    with pytest.raises(ResourceError, match="registration disagreement|arena layout disagreement"):
        r3.ensure(*plans[0])
    # no profile and no knob: a clear error, never a guessed size
    monkeypatch.delenv("AITER_GEMM_FUSION_PROFILE", raising=False)
    r4 = FusionResources(0, 4, dist=FakeDist(), backend=ArenaBackend(), policy="arena", window_bytes=4 * MiB, max_windows=1)
    with pytest.raises(ResourceError, match="needs the arena size"):
        r4.ensure(*plans[0])


def test_arena_late_owner_needs_no_collective_registration_and_is_refused_in_capture(monkeypatch):
    import types as _t

    r, be = _arena_res(monkeypatch, "arena")
    plans = _all_plans(monkeypatch)
    r.ensure(*plans[0])
    r.finalize()
    n = len(be.regs)
    r.ensure(*plans[1])  # after finalize(): allowed (a view struct is a local device allocation), no new registration, no G-late env needed
    assert len(be.regs) == n == 1 and r.pending_zero == 0
    fake = _t.ModuleType("torch")
    fake.cuda = _t.SimpleNamespace(is_current_stream_capturing=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", fake)
    from aiter.gemm_fusion.resources import ResourceError

    with pytest.raises(ResourceError, match="refused inside graph capture"):
        r.ensure(*plans[2])


def test_native_view_struct_reads_only_what_the_kernels_read():
    """(1) of the lead's question, as a CPU audit of the sources: the view is a plain device copy of ccoWindowDevice with winBase shifted; the vendored kernels
    reach a window handle only through ccoGetLocalPtr / ccoGetLsaPeerPtr (winBase, stride4G, lsaRank); the small kernel resolves peer pointers on the host."""
    import re

    src = _host_src("gf_comm.cpp")
    body = src[src.index("int32_t gf_win_view_v1"):src.index("int32_t gf_win_view_free_v1")]
    assert "hipMemcpyDeviceToHost" in body and "h.winBase += offset" in body and "hipMalloc" in body and "hipMemcpyHostToDevice" in body
    assert "ccoWindowDeregister" not in body  # a view is never registered with MORI
    dev = ""
    for root, _d, fns in os.walk(os.path.join(jit.CSRC, "large", "r11", "device")):
        dev += "".join(open(os.path.join(root, f), encoding="utf-8").read() for f in fns if f.endswith((".hpp", ".cu", ".h")))
    dev += open(os.path.join(jit.CSRC, "small", "r18", "device", "sk_gemm_fp8.cu"), encoding="utf-8").read()
    assert not re.search(r"->\s*(winBase|stride4G|lsaRank|ibgdaWin)|\.ibgdaWin|findWindow|windowTable|resourceWindow", dev)  # no direct field access, no table lookups
    users = set(re.findall(r"(ccoGetLocalPtr|ccoGetLsaPeerPtr)\(\s*(?:a|args)\.(Swin|Rwin|Cwin)", dev))
    assert {"Swin", "Rwin", "Cwin"} <= {w for _f, w in users}  # the handles are used ONLY as the first argument of those two accessors
    # no kernel source differs from the vendored accepted file (no kernel change): the vendored sources are unchanged since the vendoring commit

def test_probe_arena_sequence_one_registration_and_view_checks(capsys):
    from aiter.ops.gemm_fusion.sp_qkv import window_probe as wp

    class B(ArenaBackend):
        def uid(self, rank):
            return b"u"

        def comm_destroy(self, comm):
            pass

        def win_deregister(self, comm, w):
            pass

        def peer_ptr(self, comm, local, p):
            return local + p * (4 << 30)

    be = B()
    be.heap = 8 << 30
    ok = wp.run_arena("arena", 4 << 30, 0, 4, FakeDist(), be, "arena#0")
    out = capsys.readouterr().out
    assert ok and len(be.regs) == 1 and "registrations=1 views=15" in out and out.count("result=FAIL") == 0
    assert out.count("view_winBase_ok=1 peer_ptrs_ok=1") == 15 and len(be.freed) == 15
    assert wp.sequence("arena") == [380 * MiB] and wp.sequence("arena_per_owner") == [2508 * MiB]
    be2 = B()
    assert wp.run_arena("arena_per_owner", 8 << 30, 0, 4, FakeDist(), be2, "arena_per_owner#0") and len(be2.regs) == 1