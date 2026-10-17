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
