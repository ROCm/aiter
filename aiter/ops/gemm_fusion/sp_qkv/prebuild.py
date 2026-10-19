# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""sp_qkv planner for aiter.gemm_fusion.prebuild: profile rows -> build units -> compile tasks (torch-free; the commands come from jit.py, byte-for-byte the legacy ones
except where a recipe says otherwise).

Units of the shipped profile (aiter 15 rows): 6 small builds (one per (mode, params): dense/indexer/reuse x r17/r18 flags), 2 large builds (dense and indexer; the reuse rows share
the dense build: same -D set, same key) and 1 gf_comm.so. Task graph per unit (recipes aot1 / aot2):

  small:  sm_gemm (device TU)  --+
          sm_ran  (device TU)  --+--> hashes (sm_hashes.o: the two code-object sha256 as extern strings) --+--> link (sm_driver.so)
          host    (sm_driver.o: the host driver compiled WITHOUT waiting for the device builds) ----------+
  large:  lg_gemm (device TU, one code object with every plan entry)       lg_host (liblg_comm.so)          (independent)
  comm:   gf_comm (one TU)

  aot1: device TU = the legacy hipcc command + objcopy + clang-offload-bundler (byte-identical code objects to legacy).
  aot2: device TU = the same command with --offload-device-only --no-gpu-bundle-output straight to <name>.hsaco (no host pass, no objcopy, no bundler). Opt-in until
        `python -m aiter.ops.gemm_fusion.sp_qkv.prebuild --verify-device-only` shows .text/.rodata equal on the node.

CLI:  python -m aiter.ops.gemm_fusion.sp_qkv.prebuild [--profile NAME] [--jobs N] [--recipe aot1|aot2] [--list] [--report-json F] [--verify-device-only] [--compare A B]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

from aiter.gemm_fusion import prebuild as engine
from aiter.gemm_fusion.prebuild import BuildError, Task, Unit, default_jobs, execute

from . import jit, specs

# wall hints in seconds, used ONLY to order the scheduler (critical path first). Measured by Evidence on the node (32 quota cores, frozen 22a6a3b9, legacy per-key walls):
# small gemm 0.92-0.94, ran 0.47-0.54, host 0.54; large dense device 3.62 / indexer 4.48, host 0.47; gf_comm 0.43. The aot1 hash object and the final link are tiny (assumed).
# Recalibrate from `Report.tasks` (prebuild --report-json) if the sources change; a wrong hint only changes the order, never the result.
COST_HINTS = {"sm_gemm": 0.93, "sm_ran": 0.5, "sm_host": 0.54, "sm_hashes": 0.1, "sm_link": 0.2, "lg_gemm": 4.0, "lg_host": 0.47, "gf_comm": 0.43}


# ----------------------------------------------------------------------------------------------------------------------------- rows -> modes
def _mode_tables():
    full = {specs.sp_qkv_spec(m).signature(): m for m in ("dense", "indexer", "reuse")}
    pre = {specs.sp_qkv_spec(m).prefix(1).signature(): m for m in ("dense", "indexer", "reuse")}  # large plans cover GEMM + SPExchange only
    return full, pre


def row_mode(row) -> Optional[str]:
    """dense | indexer | reuse of a profile row, recovered from its signature (the row does not store the mode); None when the signature is not an sp_qkv spec."""
    full, pre = _mode_tables()
    return (pre if row.vid.startswith("sp_qkv.large") else full).get(row.sig)


# ----------------------------------------------------------------------------------------------------------------------------- task bodies
def _unbundle(run_cmd, tool, so: str, fatbin: str, hsaco: str):
    run_cmd(["objcopy", "-O", "binary", "--only-section=.hip_fatbin", so, fatbin])
    listing = run_cmd([tool["bundler"], "--type=o", f"--input={fatbin}", "--list"])
    tgt = next((ln.strip() for ln in listing.splitlines() if "gfx950" in ln), None)
    if not tgt:
        raise BuildError(f"no gfx950 bundle entry in {fatbin}")
    run_cmd([tool["bundler"], "--type=o", "--unbundle", f"--targets={tgt}", f"--input={fatbin}", f"--output={hsaco}"])


def _device_task_body(legacy_cmd_fn, name: str, hsaco_rel: str, device_only: bool, tool):
    """Compile one device TU to <tmp>/<hsaco_rel>: legacy command + unbundle, or the derived device-only command."""

    def run(tmp, run_cmd):
        os.makedirs(os.path.join(tmp, "img"), exist_ok=True)
        cmd = legacy_cmd_fn(tmp)
        hsaco = os.path.join(tmp, hsaco_rel)
        if device_only:
            run_cmd(jit.device_only_cmd(cmd, hsaco))
        else:
            so = cmd[cmd.index("-o") + 1]
            run_cmd(cmd)
            _unbundle(run_cmd, tool, so, os.path.join(tmp, "img", name + ".fatbin"), hsaco)

    return run


def small_unit(params: dict, mode: str, tool: Dict[str, str], recipe: str) -> Unit:
    src = params.get("src", "r17")
    dev, host = jit.small_defs(params, mode)
    key = jit.recipe_key(jit.cache_key(src, dev, host, tool), recipe)
    root = jit.jit_root()
    dev_only = recipe == "aot2"
    tasks: List[Task] = []
    for name, _cu in jit.SMALL_DEVICE_TUS:
        tasks.append(Task(name, _device_task_body(lambda tmp, n=name: jit.device_cmd_one(src, dev, tool, tmp, n), name, os.path.join("img", name + ".hsaco"), dev_only, tool),
                          cost=COST_HINTS[name]))

    def host_body(tmp, run_cmd):
        os.makedirs(os.path.join(tmp, "gen"), exist_ok=True)
        with open(os.path.join(tmp, "gen", "sm_expected_hashes.h"), "w") as f:
            f.write(jit.hashes_header_extern())
        run_cmd(jit.host_compile_cmd(src, host, tool, tmp))

    def hashes_body(tmp, run_cmd):
        os.makedirs(os.path.join(tmp, "gen"), exist_ok=True)
        gh, rh = jit._sha(os.path.join(tmp, "img", "sm_gemm.hsaco")), jit._sha(os.path.join(tmp, "img", "sm_ran.hsaco"))
        with open(os.path.join(tmp, "gen", "sm_hashes.cpp"), "w") as f:
            f.write(jit.hashes_source(gh, rh))
        run_cmd(jit.hashes_compile_cmd(tool, tmp))

    def link_body(tmp, run_cmd):
        run_cmd(jit.host_link_cmd(tool, tmp, ["sm_driver.o", "sm_hashes.o"], os.path.join(tmp, "sm_driver.so")))

    tasks += [Task("host", host_body, cost=COST_HINTS["sm_host"]),
              Task("hashes", hashes_body, deps=("sm_gemm", "sm_ran"), cost=COST_HINTS["sm_hashes"]),
              Task("link", link_body, deps=("host", "hashes"), cost=COST_HINTS["sm_link"])]

    def finalize(tmp, _log):
        with open(os.path.join(tmp, "BUILD_OK"), "w") as f:
            f.write(key + "\n")

    final = os.path.join(root, key)
    u = Unit("sp_qkv.small.v1", f"{mode} {src} " + ",".join(f"{k}={v}" for k, v in sorted(params.items())), key, key, root, tasks, finalize=finalize)
    u.result = jit.SmallBuild(key, final, os.path.join(final, "img", "sm_gemm.hsaco"), os.path.join(final, "img", "sm_ran.hsaco"), os.path.join(final, "sm_driver.so"))
    u.meta = {"src": src, "dev": dev, "host": host, "mode": mode}
    return u


def large_unit(params: dict, mode: str, tool: Dict[str, str], recipe: str) -> Unit:
    defs = jit.large_defs(params, mode)
    key = jit.recipe_key(jit.large_cache_key(defs, tool), recipe)
    root = jit.jit_root()
    dev_only = recipe == "aot2"
    name = "lg_gemm_quad_fp8_comm"
    tasks = [
        Task("lg_gemm", _device_task_body(lambda tmp: jit.large_device_cmd(defs, tool, tmp), name, os.path.join("img", name + ".hsaco"), dev_only, tool), cost=COST_HINTS["lg_gemm"]),
        Task("lg_host", lambda tmp, run_cmd: run_cmd(jit.large_host_cmd(defs, tool, tmp)), cost=COST_HINTS["lg_host"]),
    ]

    def finalize(tmp, _log):
        sha = jit._sha(os.path.join(tmp, "img", name + ".hsaco"))
        with open(os.path.join(tmp, "BUILD_OK"), "w") as f:
            f.write(key + " " + sha + "\n")

    dirname = "lg_" + key
    u = Unit("sp_qkv.large.v1", f"{'indexer' if mode == 'indexer' else 'dense'} " + ",".join(d for d in defs[:2]), key, dirname, root, tasks, finalize=finalize)
    u.meta = {"defs": defs, "mode": mode}
    return u


def comm_unit(tool: Dict[str, str]) -> Unit:
    h = jit.comm_key(tool)
    root = jit.jit_root()
    tasks = [Task("gf_comm", lambda tmp, run_cmd: run_cmd(jit.comm_cmd(tool, tmp)), cost=COST_HINTS["gf_comm"])]
    return Unit("gf_comm", "gf_comm.so", h, "comm_" + h, root, tasks, marker="gf_comm.so")


# ----------------------------------------------------------------------------------------------------------------------------- hooks for the generic prebuild
def units_for_rows(rows, *, recipe: Optional[str] = None, tool: Optional[Dict[str, str]] = None, large: bool) -> Tuple[List[Unit], List[str]]:
    recipe = recipe or jit.DEFAULT_RECIPE
    if recipe == "legacy":
        raise ValueError("prebuild plans the aot recipes only (legacy builds serially per call: jit.build_small(recipe='legacy'))")
    tool = tool or jit.toolchain()
    units: Dict[str, Unit] = {}
    skipped: List[str] = []
    for r in rows:
        mode = row_mode(r)
        if mode is None:
            skipped.append(f"{r.vid}: row m={r.m} cu={r.cu}: signature is not a known sp_qkv spec")
            continue
        if r.arch != "gfx950":
            skipped.append(f"{r.vid}: row arch {r.arch!r}: the sp_qkv kernels build for gfx950 only")
            continue
        u = (large_unit if large else small_unit)(dict(r.params), mode, tool, recipe)
        units.setdefault(u.dirname, u)
    if units:
        c = comm_unit(tool)
        units[c.dirname] = c
    return list(units.values()), skipped


# ----------------------------------------------------------------------------------------------------------------------------- jit.py delegation (first-use build, same artifacts)
def build_small(params: dict, mode: str, tool=None, recipe=None):
    tool = tool or jit.toolchain()
    u = small_unit(params, mode, tool, recipe or jit.DEFAULT_RECIPE)
    execute([u], default_jobs())
    return u.result


def build_comm(tool=None) -> str:
    tool = tool or jit.toolchain()
    u = comm_unit(tool)
    execute([u], default_jobs())
    return os.path.join(u.final, "gf_comm.so")


def build_large(params: dict, mode: str, tool=None, recipe=None):
    tool = tool or jit.toolchain()
    u = large_unit(params, mode, tool, recipe or jit.DEFAULT_RECIPE)
    execute([u], default_jobs())
    final = u.final
    sha = open(os.path.join(final, "BUILD_OK")).read().split()[-1]
    return jit.LargeBuild(u.key, final, os.path.join(final, "img", "lg_gemm_quad_fp8_comm.hsaco"), sha, os.path.join(final, "liblg_comm.so"))


# ----------------------------------------------------------------------------------------------------------------------------- code-object comparison (build-structure changes must keep the kernels)
_SH = struct.Struct("<IIQQQQIIQQ")  # ELF64 section header


def elf_sections(path: str) -> Dict[str, Tuple[int, str]]:
    """{section name: (size, sha256 of its bytes)} of an ELF64 little-endian file (an AMDGPU code object): pure Python, no llvm tools needed."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise ValueError(f"{path}: not an ELF64 little-endian file")
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    hdrs = [_SH.unpack_from(data, shoff + i * shentsize) for i in range(shnum)]
    stroff = hdrs[shstrndx][4]

    def name_of(off):
        end = data.index(b"\0", stroff + off)
        return data[stroff + off:end].decode()

    out = {}
    for h in hdrs:
        name, typ, _fl, _addr, off, size = h[0], h[1], h[2], h[3], h[4], h[5]
        nm = name_of(name)
        body = b"" if typ == 8 else data[off:off + size]  # SHT_NOBITS has no file bytes
        out[nm] = (size, hashlib.sha256(body).hexdigest())
    return out


def compare_code_objects(a: str, b: str, sections: Sequence[str] = (".text", ".rodata")) -> dict:
    """Do two code objects carry the same kernels? Compares the whole-file sha256 and the named sections (default .text = the machine code, .rodata = the kernel descriptors)."""
    sa, sb = elf_sections(a), elf_sections(b)
    res = {"full_sha_equal": jit._sha(a) == jit._sha(b), "sections": {}}
    for s in sections:
        res["sections"][s] = (s in sa and s in sb and sa[s] == sb[s], sa.get(s, (None,))[0], sb.get(s, (None,))[0])
    res["kernels_equal"] = all(v[0] for v in res["sections"].values())
    res["only_in_a"], res["only_in_b"] = sorted(set(sa) - set(sb)), sorted(set(sb) - set(sa))
    return res


def verify_device_only(tool=None, jobs=None) -> dict:
    """Compile every distinct device TU of the shipped profile twice (aot1 = legacy + unbundle, aot2 = device-only) in a scratch directory, in parallel, and compare the
    code objects. Prints/returns per-TU {kernels_equal, full_sha_equal, section sizes} plus the wall of each way (the saving of aot2 per TU)."""
    import time

    tool = tool or jit.toolchain()
    run = engine.SubprocessRunner()
    todo = []
    for (mode, _T), (_vid, p) in specs.SMALL_CELLS.items():
        todo.append((f"small/{mode}/{p.get('src', 'r17')}", small_unit(dict(p), mode, tool, "aot1")))
    for mode in ("dense", "indexer"):
        _vid, p = next(v for (m, _T), v in specs.LARGE_CELLS.items() if m == mode)
        todo.append((f"large/{mode}", large_unit(dict(p), mode, tool, "aot1")))
    seen, work = set(), []
    for label, u in todo:
        if u.dirname in seen:
            continue
        seen.add(u.dirname)
        for t in u.tasks:
            if t.name in ("sm_gemm", "sm_ran", "lg_gemm"):
                work.append((f"{label}/{t.name}", u, t))
    out: Dict[str, dict] = {}

    def one(item):
        label, u, t = item
        res = {}
        with tempfile.TemporaryDirectory(prefix="verify_", dir=jit.jit_root() if os.path.isdir(jit.jit_root()) else None) as scratch:
            walls = {}
            for way, dev_only in (("aot1", False), ("aot2", True)):
                tmp = os.path.join(scratch, way)
                os.makedirs(os.path.join(tmp, "img"))
                name = t.name if t.name != "lg_gemm" else "lg_gemm_quad_fp8_comm"
                if t.name == "lg_gemm":
                    cmd_fn = lambda tmp_: jit.large_device_cmd(u.meta["defs"], tool, tmp_)  # noqa: E731
                else:
                    cmd_fn = lambda tmp_, n=name: jit.device_cmd_one(u.meta["src"], u.meta["dev"], tool, tmp_, n)  # noqa: E731
                body = _device_task_body(cmd_fn, name, os.path.join("img", name + ".hsaco"), dev_only, tool)
                t0 = time.monotonic()
                body(tmp, run)
                walls[way] = round(time.monotonic() - t0, 2)
            a, b = (os.path.join(scratch, w, "img", (t.name if t.name != "lg_gemm" else "lg_gemm_quad_fp8_comm") + ".hsaco") for w in ("aot1", "aot2"))
            res = compare_code_objects(a, b)
            res["wall_s"] = walls
        return label, res

    with ThreadPoolExecutor(max_workers=jobs or default_jobs()) as pool:
        for label, res in pool.map(one, work):
            out[label] = res
    return out


# ----------------------------------------------------------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="AOT-build every sp_qkv kernel a gemm_fusion profile needs (parallel, rank-safe)")
    ap.add_argument("--profile", default="minimax_m3_sp4_gfx950_v1")
    ap.add_argument("--arch")
    ap.add_argument("--cu", type=int)
    ap.add_argument("--world", type=int)
    ap.add_argument("--jobs", type=int)
    ap.add_argument("--recipe", default=None, choices=[r for r in jit.RECIPES if r != "legacy"])
    ap.add_argument("--list", action="store_true", help="print the unit/task plan and exit (no compiler runs)")
    ap.add_argument("--report-json")
    ap.add_argument("--verify-device-only", action="store_true", help="aot1 vs aot2 code objects of every distinct device TU: .text/.rodata equality + per-way wall")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    a = ap.parse_args(argv)
    if a.compare:
        print(json.dumps(compare_code_objects(*a.compare), indent=1))
        return 0
    if a.verify_device_only:
        res = verify_device_only(jobs=a.jobs)
        print(json.dumps(res, indent=1))
        return 0 if all(v["kernels_equal"] for v in res.values()) else 3
    from aiter.gemm_fusion.prebuild import plan_profile, prebuild_profile

    if a.list:
        units, skipped = plan_profile(a.profile, arch=a.arch, cu=a.cu, world=a.world, recipe=a.recipe)
        for u in units:
            print(f"{u.kind:18s} {u.dirname}  {u.label}")
            for t in u.tasks:
                print(f"    {t.name:10s} deps={list(t.deps)} cost={t.cost}")
        for s in skipped:
            print("SKIPPED", s)
        return 0
    keys = prebuild_profile(a.profile, arch=a.arch, cu=a.cu, world=a.world, jobs=a.jobs, recipe=a.recipe)
    rep = prebuild_profile.last_report
    print(rep.summary(top=20))
    print("keys:", keys)
    if a.report_json:
        with open(a.report_json, "w") as f:
            json.dump(rep.to_dict(), f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
