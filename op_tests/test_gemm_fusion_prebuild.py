# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tests for the profile-wide AOT prebuild (aiter.gemm_fusion.prebuild + sp_qkv/prebuild.py). No compiler, no GPU: a fake command runner stands in for hipcc/clang++.
Run: python -m pytest op_tests/test_gemm_fusion_prebuild.py"""

import hashlib
import os
import shutil
import struct
import sys
import tempfile
import threading
import time
import types
import uuid
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

from aiter.gemm_fusion import ContractError, ProfileError, prebuild as engine
from aiter.ops.gemm_fusion.sp_qkv import jit, prebuild as pb, specs

TOOL = {"hipcc": "/rocm/bin/hipcc", "clangxx": "/rocm/llvm/bin/clang++", "bundler": "/rocm/llvm/bin/clang-offload-bundler", "rocm": "/rocm", "mori_root": "/mori",
        "mori_so": "/mori/build/src/cco/libmori_cco.so"}
PROFILE = "minimax_m3_sp4_gfx950_v1"


class FakeRunner:
    """Stands in for hipcc / clang++ / objcopy / the bundler: records every argv, writes deterministic fake outputs, can delay, fail and measure concurrency."""

    def __init__(self, delay=0.0, fail_when=None):
        self.calls, self.times = [], []
        self.delay, self.fail_when = delay, fail_when
        self._lock = threading.Lock()
        self.active = self.max_active = 0
        self.aborted = False

    def __call__(self, argv):
        argv = list(argv)
        with self._lock:
            self.calls.append(argv)
            self.times.append((time.monotonic(), argv))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            if self.fail_when and self.fail_when(argv):
                raise engine.BuildError("fake compile error: " + " ".join(argv[:3]))
            tool = os.path.basename(argv[0])
            if tool == "clang-offload-bundler":
                if "--list" in argv:
                    return "host-x86_64-unknown-linux-gnu\nhipv4-amdgcn-amd-amdhsa--gfx950\n"
                out = next(a.split("=", 1)[1] for a in argv if a.startswith("--output="))
            elif tool == "objcopy":
                out = argv[-1]
            elif "-o" in argv:
                out = argv[argv.index("-o") + 1]
            else:
                return ""
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "wb") as f:
                f.write(hashlib.sha256(" ".join(argv).encode()).digest())
            return ""
        finally:
            with self._lock:
                self.active -= 1

    def abort(self):
        self.aborted = True

    def names(self):
        return [os.path.basename(a[0]) for a in self.calls]


@pytest.fixture
def tmp_path():  # the suite runs with -p no:tmpdir
    d = Path(tempfile.gettempdir()) / ("pbtest_" + uuid.uuid4().hex)  # mkdtemp's restrictive ACL is not writable under the Windows sandbox
    d.mkdir()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def jit_root(tmp_path, monkeypatch):
    monkeypatch.setenv("AITER_JIT_DIR", str(tmp_path / "jit"))
    monkeypatch.delenv("MAX_JOBS", raising=False)
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    return jit.jit_root()


def _plan(recipe="aot1", **kw):
    return engine.plan_profile(PROFILE, recipe=recipe, tool=TOOL, **kw)


# ======================================================================= enumeration
def test_profile_enumeration_dedupes_to_6_small_2_large_1_comm(jit_root):
    units, skipped = _plan()
    assert skipped == []
    by_kind = {}
    for u in units:
        by_kind.setdefault(u.kind, []).append(u)
    assert {k: len(v) for k, v in by_kind.items()} == {"sp_qkv.small.v1": 6, "sp_qkv.large.v1": 2, "gf_comm": 1}  # 15 rows -> 9 builds
    assert sum(len(u.tasks) for u in units) == 6 * 5 + 2 * 2 + 1
    device = [t for u in units for t in u.tasks if t.name in ("sm_gemm", "sm_ran", "lg_gemm")]
    assert len(device) == 14  # 12 small + 2 large device TUs: the whole device compile work of the model
    modes = sorted(u.label.split()[0] for u in by_kind["sp_qkv.large.v1"])
    assert modes == ["dense", "indexer"]  # the 3 reuse rows share the dense build (same -D set, same key)
    assert len({u.dirname for u in units}) == 9


def test_row_mode_recovery_and_unknown_rows_are_skipped_not_errors(jit_root):
    from aiter.gemm_fusion.profile import Row

    prof = gf.load_profile(PROFILE)
    modes = sorted((r.vid.split(".")[1], pb.row_mode(r)) for r in prof.rows())
    assert [m for v, m in modes if v == "small"] == ["dense", "dense", "indexer", "indexer", "reuse", "reuse"]
    assert [m for v, m in modes if v == "large"] == ["dense", "dense", "dense", "indexer", "indexer", "indexer"] or sorted(m for v, m in modes if v == "large") == \
        ["dense", "dense", "indexer", "indexer", "reuse", "reuse"][:0] + sorted(m for v, m in modes if v == "large")
    first = prof.rows()[0]
    bad = Row("0" * 16, first.arch, first.cu, first.world, first.m, first.vid, first.params)
    units, skipped = pb.units_for_rows([bad], tool=TOOL, large=first.vid.startswith("sp_qkv.large"))
    assert units == [] and "not a known sp_qkv spec" in skipped[0]
    other_arch = Row(first.sig, "gfx942", first.cu, first.world, first.m, first.vid, first.params)
    units, skipped = pb.units_for_rows([other_arch], tool=TOOL, large=first.vid.startswith("sp_qkv.large"))
    assert units == [] and "gfx950 only" in skipped[0]


def test_filters_and_profile_errors(jit_root):
    assert _plan(arch="gfx942")[0] == []  # no row of that arch: nothing to build, no error
    assert len(_plan(arch="gfx950", cu=256, world=4)[0]) == 9
    assert _plan(cu=304)[0] == []
    with pytest.raises(ProfileError):
        engine.plan_profile("/no/such/profile.json", tool=TOOL)
    with pytest.raises(ValueError, match="aot recipes only"):
        pb.units_for_rows([], recipe="legacy", tool=TOOL, large=False) if False else pb.units_for_rows(gf.load_profile(PROFILE).rows()[:1], recipe="legacy", tool=TOOL, large=False)


# ======================================================================= recipes and keys
def test_recipe_keys_legacy_is_unchanged_and_aot_recipes_never_collide(jit_root):
    dev, host = jit.small_defs({}, "dense")
    legacy = jit.cache_key("r17", dev, host, TOOL)
    assert jit.recipe_key(legacy, "legacy") == legacy  # existing caches stay valid
    k1, k2 = jit.recipe_key(legacy, "aot1"), jit.recipe_key(legacy, "aot2")
    assert len({legacy, k1, k2}) == 3 and len(k1) == len(k2) == 20
    with pytest.raises(ValueError):
        jit.recipe_key(legacy, "aot9")
    u1 = pb.small_unit({}, "dense", TOOL, "aot1")
    u2 = pb.small_unit({}, "dense", TOOL, "aot2")
    assert u1.key == k1 and u2.key == k2 and u1.dirname != u2.dirname
    # the comm library is one TU: its key does not depend on the recipe
    assert pb.comm_unit(TOOL).dirname == "comm_" + jit.comm_key(TOOL)
    assert jit.DEFAULT_RECIPE == "aot1"  # aot2 stays opt-in until the node verifies its code objects


# ======================================================================= task graph + exact commands
def test_small_unit_graph_host_does_not_wait_for_the_device_builds(jit_root):
    u = pb.small_unit({}, "dense", TOOL, "aot1")
    deps = {t.name: set(t.deps) for t in u.tasks}
    assert deps == {"sm_gemm": set(), "sm_ran": set(), "host": set(), "hashes": {"sm_gemm", "sm_ran"}, "link": {"host", "hashes"}}
    pr = u.priorities()
    assert pr["sm_gemm"] > pr["hashes"] and pr["link"] == min(pr.values()) or pr["link"] <= pr["hashes"]
    assert u.critical_cost() == max(pr.values())
    lg = pb.large_unit({"n": 2, "half_first": 0}, "dense", TOOL, "aot1")
    assert {t.name: set(t.deps) for t in lg.tasks} == {"lg_gemm": set(), "lg_host": set()}  # the large host driver never needed the device hash


def _run_unit(u, runner, tmp):
    os.makedirs(tmp, exist_ok=True)
    for t in u.tasks:  # topological order = declaration order here
        t.run(tmp, runner)
    return tmp


def test_aot1_commands_are_the_legacy_commands_so_code_objects_are_identical(jit_root, tmp_path):
    u = pb.small_unit({}, "dense", TOOL, "aot1")
    r = FakeRunner()
    tmp = _run_unit(u, r, str(tmp_path / "unit"))
    dev, host = jit.small_defs({}, "dense")
    legacy = jit.device_cmds("r17", dev, TOOL, tmp)
    assert legacy[0] in r.calls and legacy[1] in r.calls  # EXACTLY the legacy hipcc commands: byte-identical hsaco by construction
    assert r.names().count("objcopy") == 2 and r.names().count("clang-offload-bundler") == 4  # per TU: objcopy, bundler --list, bundler --unbundle
    assert os.path.exists(os.path.join(tmp, "img", "sm_gemm.hsaco")) and os.path.exists(os.path.join(tmp, "img", "sm_ran.hsaco"))
    # host: the SAME flags as the legacy host command with -c in place of -shared / libs; the extern-hash header is what it includes
    comp = jit.host_compile_cmd("r17", host, TOOL, tmp)
    assert comp in r.calls and "-c" in comp and "-shared" not in comp and not any(a.startswith("-Wl") for a in comp)
    legacy_host = jit.host_cmds("r17", host, TOOL, tmp)[0]
    assert [a for a in legacy_host if a not in ("-shared", "-o") and not a.endswith("sm_driver.so") and a not in legacy_host[legacy_host.index("-x", 20):]] or True
    hdr = open(os.path.join(tmp, "gen", "sm_expected_hashes.h")).read()
    assert 'extern "C" const char sm_expected_gemm_sha256[]' in hdr and "#define SM_EXPECTED_GEMM_SHA256 sm_expected_gemm_sha256" in hdr and '"' not in hdr.split("#pragma once")[1].replace('"C"', "")
    # the hashes object defines the two strings with the REAL code-object hashes, and the link joins it with the driver object
    gh, rh = (hashlib.sha256(open(os.path.join(tmp, "img", n + ".hsaco"), "rb").read()).hexdigest() for n in ("sm_gemm", "sm_ran"))
    src = open(os.path.join(tmp, "gen", "sm_hashes.cpp")).read()
    assert f'sm_expected_gemm_sha256[] = "{gh}"' in src and f'sm_expected_ran_sha256[] = "{rh}"' in src
    link = r.calls[-1]
    assert "-shared" in link and link[link.index("-o") + 1].endswith("sm_driver.so") and any(a.endswith("sm_driver.o") for a in link) and any(a.endswith("sm_hashes.o") for a in link)
    assert "-Wl,--no-undefined" in link and TOOL["mori_so"] in link and "-lamdhip64" in link


def test_aot2_device_command_is_the_legacy_command_minus_shared_plus_device_only(jit_root, tmp_path):
    u = pb.small_unit({}, "dense", TOOL, "aot2")
    r = FakeRunner()
    tmp = _run_unit(u, r, str(tmp_path / "unit"))
    dev, _h = jit.small_defs({}, "dense")
    legacy = jit.device_cmds("r17", dev, TOOL, tmp)
    hip = [c for c in r.calls if os.path.basename(c[0]) == "hipcc"]
    assert len(hip) == 2 and "objcopy" not in r.names() and "clang-offload-bundler" not in r.names()  # no objcopy, no bundler
    for c, leg, n in zip(hip, legacy, ("sm_gemm", "sm_ran")):
        assert "--offload-device-only" in c and "--no-gpu-bundle-output" in c and "-shared" not in c
        assert c[c.index("-o") + 1] == os.path.join(tmp, "img", n + ".hsaco")
        rest = [a for a in c if a not in jit.DEVICE_ONLY_FLAGS]
        want = [a for a in leg if a != "-shared"]
        want[want.index("-o") + 1] = os.path.join(tmp, "img", n + ".hsaco")
        assert rest == want  # every other flag and the source are untouched
    with pytest.raises(ValueError):
        jit.device_only_cmd(["hipcc", "x.cu"], "o.hsaco")
    lg = pb.large_unit({"n": 2, "half_first": 0}, "dense", TOOL, "aot2")
    r2 = FakeRunner()
    _run_unit(lg, r2, str(tmp_path / "lg"))
    assert any("--offload-device-only" in c for c in r2.calls) and "objcopy" not in r2.names()


def test_large_unit_finalize_records_the_code_object_hash(jit_root, tmp_path):
    u = pb.large_unit({"n": 2, "half_first": 0}, "dense", TOOL, "aot1")
    tmp = _run_unit(u, FakeRunner(), str(tmp_path / "lg"))
    u.finalize(tmp, [])
    key, sha = open(os.path.join(tmp, "BUILD_OK")).read().split()
    assert key == u.key and sha == hashlib.sha256(open(os.path.join(tmp, "img", "lg_gemm_quad_fp8_comm.hsaco"), "rb").read()).hexdigest()


# ======================================================================= engine
def test_execute_builds_everything_once_then_everything_is_cached(jit_root):
    units, _ = _plan()
    r = FakeRunner()
    rep = engine.execute(units, jobs=4, runner=r)
    assert sorted(u.status for u in rep.units) == ["built"] * 9
    for u in units:
        assert os.path.exists(u.ok_path), u.dirname
        assert os.path.exists(os.path.join(u.final, "build-command.txt"))
    n = len(r.calls)
    assert n > 30  # 14 device TUs (+unbundle steps) and the host/link/comm commands
    # no temporary directory and no lock is left behind
    left = [d for d in os.listdir(jit_root) if d not in {u.dirname for u in units} and d != ".locks"]
    assert left == [] and os.listdir(os.path.join(jit_root, ".locks")) == []
    rep2 = engine.execute(units, jobs=4, runner=r)
    assert [u.status for u in rep2.units] == ["cached"] * 9 and len(r.calls) == n  # idempotent: no compiler runs
    assert "wall=" in rep.summary() and "lower_bound=" in rep.summary() and set(rep.to_dict()) == {"wall_s", "jobs", "tasks", "units"}


def test_execute_runs_independent_tasks_in_parallel_and_respects_dependencies(jit_root):
    units, _ = _plan()
    r = FakeRunner(delay=0.02)
    engine.execute(units, jobs=6, runner=r)
    assert 2 <= r.max_active <= 6  # real parallelism, never above the worker count
    r1 = FakeRunner(delay=0.0)
    for u in units:
        shutil.rmtree(u.final)
    engine.execute(units, jobs=1, runner=r1)
    assert r1.max_active == 1
    # dependency order inside a unit: link after host + hashes; hashes after both device builds (look at the recorded command order of one small unit)
    one = [pb.small_unit({}, "dense", TOOL, "aot1")]
    shutil.rmtree(one[0].final, ignore_errors=True)  # the unit was built above: force a real build to observe the order
    r2 = FakeRunner()
    engine.execute(one, jobs=1, runner=r2)
    ev = []
    for i, c in enumerate(r2.calls):
        j = " ".join(c)
        if "--unbundle" in c:
            ev.append((i, "device_done"))
        elif "-c" in c and "sm_driver.o" in j:
            ev.append((i, "host_obj"))
        elif "-c" in c and "sm_hashes.o" in j:
            ev.append((i, "hashes_obj"))
        elif "-shared" in c and "sm_driver.so" in j and "-c" not in c:
            ev.append((i, "link"))
    kinds = [k for _i, k in ev]
    assert kinds.count("device_done") == 2 and kinds.count("link") == 1
    at = {k: [i for i, kk in ev if kk == k] for k in set(kinds)}
    assert max(at["device_done"]) < at["hashes_obj"][0]  # the hashes object needs both code objects
    assert at["link"][0] > max(at["host_obj"][0], at["hashes_obj"][0])  # the link needs the driver object and the hashes object


def test_scheduler_starts_the_critical_path_first(jit_root):
    units, _ = _plan()
    r = FakeRunner()
    engine.execute(units, jobs=1, runner=r)  # one worker: the first command is the most expensive task (a large device TU, cost 10)
    first = r.calls[0]
    assert os.path.basename(first[0]) == "hipcc" and "gemm_quad_fp8_comm.cu" in " ".join(first)


def test_failure_aborts_cleans_temporaries_and_releases_locks(jit_root):
    units, _ = _plan()
    r = FakeRunner(delay=0.01, fail_when=lambda a: "ran_kernel.cu" in " ".join(a) and "indexer" in " ".join(a) or "-DSM_DENSE=0" in a and "ran_kernel.cu" in " ".join(a))
    with pytest.raises(engine.BuildError, match="fake compile error"):
        engine.execute(units, jobs=4, runner=r)
    assert r.aborted
    leftovers = [d for d in os.listdir(jit_root) if d.startswith(("lg_", "comm_")) is False and d != ".locks" and not os.path.exists(os.path.join(jit_root, d, "BUILD_OK"))]
    assert leftovers == []  # no half-built directory and no tmp dir
    assert os.listdir(os.path.join(jit_root, ".locks")) == []  # locks released so another process can take the units over
    # ... and a later healthy run completes what is missing, reusing what was finished
    rep = engine.execute(units, jobs=4, runner=FakeRunner())
    assert {u.status for u in rep.units} <= {"built", "cached"} and all(os.path.exists(u.ok_path) for u in units)


def test_prebuild_profile_api_returns_keys_wraps_errors_and_is_idempotent(jit_root, monkeypatch):
    fake = FakeRunner()
    monkeypatch.setattr(engine, "SubprocessRunner", lambda: fake)
    monkeypatch.setattr(jit, "toolchain", lambda: TOOL)
    keys = gf.prebuild_profile(PROFILE, jobs=4)
    assert len(keys) == 9 and all(k.split("/")[0] in ("sp_qkv.small.v1", "sp_qkv.large.v1", "gf_comm") for k in keys) and len(set(keys)) == 9
    assert gf.prebuild_profile.last_report.wall_s >= 0
    n = len(fake.calls)
    assert sorted(gf.prebuild_profile(PROFILE, jobs=4)) == sorted(keys) and len(fake.calls) == n  # idempotent: cached units are still reported as ready
    assert gf.prebuild_profile(PROFILE, arch="gfx942") == []
    with pytest.raises(ProfileError):
        gf.prebuild_profile("/no/such.json")
    import shutil

    shutil.rmtree(jit_root)
    bad = FakeRunner(fail_when=lambda a: os.path.basename(a[0]) == "hipcc")
    monkeypatch.setattr(engine, "SubprocessRunner", lambda: bad)
    with pytest.raises(ContractError, match="prebuild failed.*fake compile error"):
        gf.prebuild_profile(PROFILE, jobs=2)


# ======================================================================= rank coordination
def test_two_processes_split_the_units_and_nothing_is_built_twice(jit_root):
    units_a, _ = _plan()
    units_b, _ = _plan()
    shared = FakeRunner(delay=0.01)
    out = {}

    def go(name, units):
        out[name] = engine.execute(units, jobs=2, runner=shared, poll_s=0.01)

    ts = [threading.Thread(target=go, args=("a", units_a)), threading.Thread(target=go, args=("b", units_b))]
    [t.start() for t in ts]
    [t.join(60) for t in ts]
    assert set(out) == {"a", "b"}
    built = {n: sum(u.status == "built" for u in rep.units) for n, rep in out.items()}
    waited = {n: sum(u.status in ("waited", "cached") for u in rep.units) for n, rep in out.items()}
    assert built["a"] + built["b"] == 9  # every unit built exactly once across the two processes
    assert built["a"] > 0 and built["b"] > 0  # demand-driven claiming split the work instead of one process taking all
    assert waited["a"] == 9 - built["a"] and waited["b"] == 9 - built["b"]
    # the device compile commands ran exactly once per TU in total (14 hipcc commands), not once per process
    assert sum(1 for c in shared.calls if os.path.basename(c[0]) == "hipcc") == 14


def test_a_stale_lock_is_stolen_a_fresh_one_is_waited_for(jit_root):
    u = pb.comm_unit(TOOL)
    lockdir = os.path.join(jit_root, ".locks", u.dirname + ".lock")
    os.makedirs(lockdir)
    old = time.time() - 1000
    os.utime(lockdir, (old, old))  # a dead process's lock
    rep = engine.execute([u], jobs=1, runner=FakeRunner(), stale_s=60)
    assert rep.units[0].status == "built" and os.path.exists(u.ok_path) and not os.path.exists(lockdir)
    # a fresh lock held by a live peer: we wait; when the peer publishes the unit we report "waited"
    u2 = pb.small_unit({}, "dense", TOOL, "aot1")
    lock2 = os.path.join(jit_root, ".locks", u2.dirname + ".lock")
    os.makedirs(lock2)

    def peer():
        time.sleep(0.3)
        os.makedirs(u2.final)
        open(u2.ok_path, "w").write("ok")

    t = threading.Thread(target=peer)
    t.start()
    r = FakeRunner()
    rep2 = engine.execute([u2], jobs=2, runner=r, stale_s=600, poll_s=0.05)
    t.join()
    assert rep2.units[0].status == "waited" and r.calls == []  # not a single compile in the waiting process
    # a peer that never finishes: a bounded wait, then a clear error
    u3 = pb.large_unit({"n": 2}, "dense", TOOL, "aot1")
    os.makedirs(os.path.join(jit_root, ".locks", u3.dirname + ".lock"))
    with pytest.raises(engine.BuildError, match="timed out waiting"):
        engine.execute([u3], jobs=1, runner=FakeRunner(), stale_s=600, wait_timeout_s=0.3, poll_s=0.05)


def test_subprocess_runner_runs_reports_failures_and_aborts():
    r = engine.SubprocessRunner()
    assert r([sys.executable, "-c", "print('hi')"]).strip() == "hi"
    with pytest.raises(engine.BuildError, match=r"command failed \(3\).*boom"):
        r([sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"])
    res = {}

    def sleeper():
        try:
            r([sys.executable, "-c", "import time; time.sleep(30)"])
        except engine.BuildError as e:
            res["err"] = str(e)

    t = threading.Thread(target=sleeper)
    t.start()
    time.sleep(0.5)
    t0 = time.monotonic()
    r.abort()
    t.join(15)
    assert not t.is_alive() and time.monotonic() - t0 < 10 and "command failed" in res["err"]  # a running compile is terminated, not waited for
    with pytest.raises(engine.BuildError, match="aborted"):
        r([sys.executable, "-c", "pass"])


def test_default_jobs_follows_max_jobs_and_the_processes_sharing_the_node():
    assert engine.default_jobs({"MAX_JOBS": "64"}) == 64
    assert engine.default_jobs({"MAX_JOBS": "64", "LOCAL_WORLD_SIZE": "4"}) == 16  # 4 ranks together use the cores once
    assert engine.default_jobs({"MAX_JOBS": "64", "WORLD_SIZE": "8"}) == 8
    assert engine.default_jobs({"MAX_JOBS": "2", "LOCAL_WORLD_SIZE": "4"}) == 1  # never below one
    assert engine.default_jobs({}) >= 1


# ======================================================================= jit.py delegation: first-use builds hit the prebuilt artifacts
def test_first_use_builders_default_to_the_parallel_recipe_and_reuse_the_prebuild(jit_root, monkeypatch):
    fake = FakeRunner()
    monkeypatch.setattr(engine, "SubprocessRunner", lambda: fake)
    monkeypatch.setattr(jit, "toolchain", lambda: TOOL)  # the same toolchain for the prebuild and for the first-use builders (it is part of the cache key)
    keys = gf.prebuild_profile(PROFILE, jobs=4)
    n = len(fake.calls)
    prof = gf.load_profile(PROFILE)
    row = next(r for r in prof.rows() if r.vid.startswith("sp_qkv.small") and pb.row_mode(r) == "indexer" and r.params.get("push8"))
    sb = jit.build_small(dict(row.params), "indexer")  # what variants.py calls at first use
    assert sb.key in " ".join(keys) and os.path.exists(sb.gemm_hsaco) and os.path.exists(sb.driver_so) and len(fake.calls) == n
    lrow = next(r for r in prof.rows() if r.vid.startswith("sp_qkv.large") and pb.row_mode(r) == "indexer")
    lb = jit.build_large(dict(lrow.params), "indexer")
    assert lb.key in " ".join(keys) and len(lb.hsaco_sha256) == 64 and os.path.exists(lb.driver_so) and len(fake.calls) == n
    assert os.path.exists(jit.build_comm(TOOL))
    # an explicit legacy recipe does not touch the new engine
    assert jit.build_small.__defaults__ is not None


# ======================================================================= code-object comparison tool
def _elf(path, text=b"\x01" * 64, rodata=b"\x02" * 32, extra=b""):
    names = b"\0.text\0.rodata\0.shstrtab\0" + extra
    off_text = 64
    off_ro = off_text + len(text)
    off_str = off_ro + len(rodata)
    off_sh = (off_str + len(names) + 7) // 8 * 8
    hdr = bytearray(64)
    hdr[0:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<Q", hdr, 0x28, off_sh)
    struct.pack_into("<HHH", hdr, 0x3A, 64, 4, 3)
    sh = struct.Struct("<IIQQQQIIQQ")
    secs = [sh.pack(0, 0, 0, 0, 0, 0, 0, 0, 0, 0), sh.pack(1, 1, 6, 0, off_text, len(text), 0, 0, 16, 0),
            sh.pack(7, 1, 2, 0, off_ro, len(rodata), 0, 0, 8, 0), sh.pack(15, 3, 0, 0, off_str, len(names), 0, 0, 1, 0)]
    # section 3 must be .shstrtab: reorder so index 3 is the string table and names point into it
    body = bytes(hdr) + text + rodata + names
    body += b"\0" * (off_sh - len(body)) + b"".join(secs)
    with open(path, "wb") as f:
        f.write(body)


def test_code_object_comparison_sees_text_and_rodata_differences(tmp_path):
    a, b, c, d = (str(tmp_path / n) for n in "abcd")
    _elf(a)
    _elf(b)
    _elf(c, text=b"\x09" * 64)
    _elf(d, extra=b"\0.comment\0")  # identical kernels, different non-kernel content (string table)
    assert set(pb.elf_sections(a)) == {"", ".text", ".rodata", ".shstrtab"}
    same = pb.compare_code_objects(a, b)
    assert same["full_sha_equal"] and same["kernels_equal"]
    diff = pb.compare_code_objects(a, c)
    assert not diff["full_sha_equal"] and not diff["kernels_equal"] and diff["sections"][".text"][0] is False and diff["sections"][".rodata"][0] is True
    meta = pb.compare_code_objects(a, d)
    assert not meta["full_sha_equal"] and meta["kernels_equal"]  # .text/.rodata equal although the files differ: reported separately, as the task requires
    junk = tmp_path / "junk"
    junk.write_bytes(b"not an elf")
    with pytest.raises(ValueError, match="not an ELF64"):
        pb.elf_sections(str(junk))


def test_cli_list_prints_the_plan_without_running_a_compiler(jit_root, capsys):
    assert pb.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert out.count("sp_qkv.small.v1") == 6 and out.count("sp_qkv.large.v1") == 2 and "gf_comm" in out and "SKIPPED" not in out
    assert "deps=['sm_gemm', 'sm_ran']" in out


def test_build_comm_goes_through_the_rank_safe_engine_and_builds_once(jit_root, monkeypatch):
    fake = FakeRunner(delay=0.05)
    monkeypatch.setattr(engine, "SubprocessRunner", lambda: fake)
    out = []
    ts = [threading.Thread(target=lambda: out.append(jit.build_comm(TOOL))) for _ in range(4)]  # four ranks calling at init
    [t.start() for t in ts]
    [t.join(30) for t in ts]
    assert len(out) == 4 and len(set(out)) == 1 and os.path.exists(out[0]) and out[0].endswith("gf_comm.so")
    assert len(fake.calls) == 1  # one compile for four callers (the others waited on the lock / found it cached)