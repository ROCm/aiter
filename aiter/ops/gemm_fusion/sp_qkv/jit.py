# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""JIT build of the sp_qkv small-M kernels (Phase-1 recipe, byte-identical sources).

Mirrors candidates/SM-T1-r{17,18}/build_device.sh + host/build_host.sh with explicit -D flags derived from the profile
row params. Device: hipcc -> .so -> .hip_fatbin -> clang-offload-bundler -> sm_gemm.hsaco / sm_ran.hsaco; the expected
code-object hashes are generated into gen/sm_expected_hashes.h (as the Phase-1 recipe does) and the host driver is built
against it. One build directory per cache key (sha256 of flags + source manifest + wrapper sources + toolchain).

Build RECIPES (cache keys are unchanged for "legacy"; the others fold the recipe name into the key because their artifacts differ):
  legacy  one build per call, serial: device .so x2 -> unbundle -> hash header -> host driver (the Phase-1 recipe, kept for A/B and as the reference).
  aot1    DEFAULT. Same device compile commands (so the code objects are byte-identical to legacy) but run by prebuild.py as ONE parallel task graph over every unit a profile
          needs; the host drivers no longer wait for the device builds (the expected hashes are linked in as a tiny extern-string object, sm_hashes.o).
  aot2    aot1 + device TUs compiled straight to a code object (--offload-device-only --no-gpu-bundle-output): no host pass, no objcopy, no bundler.
          Opt-in until `prebuild --verify-device-only` shows the .text/.rodata of every kernel equal to aot1 on the node.
See prebuild.py (profile enumeration, parallel scheduler, rank coordination, per-task wall report).

Environment (MORI is consumed read-only):
  MORI_ROOT      include root containing mori/cco/cco.hpp      (default /reference-mori)
  MORI_CCO_SO    libmori_cco.so                                (default $MORI_ROOT/build/src/cco/libmori_cco.so)
  ROCM_PATH      default /opt/rocm
  AITER_JIT_DIR  cache root (default ~/.aiter/jit); builds go to <root>/gemm_fusion_sp_qkv/<key>/
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..", "csrc", "gemm_fusion", "sp_qkv"))
SRCS = ("r17", "r18")
RECIPES = ("legacy", "aot1", "aot2")
DEFAULT_RECIPE = "aot1"


def recipe_key(key: str, recipe: str) -> str:
    """Cache key of a recipe: the legacy key itself, otherwise a hash of (legacy key, recipe) so that artifacts of different recipes never mix."""
    if recipe not in RECIPES:
        raise ValueError(f"recipe {recipe!r} not in {RECIPES}")
    return key if recipe == "legacy" else hashlib.sha256(f"{key}|{recipe}".encode()).hexdigest()[:20]


from aiter.gemm_fusion.prebuild import BuildError  # noqa: E402  (one error type for the legacy builds and the parallel engine)


@dataclass(frozen=True)
class SmallBuild:
    key: str
    dir: str
    gemm_hsaco: str
    ran_hsaco: str
    driver_so: str


# ---------------------------------------------------------------- flags
def small_defs(params: dict, mode: str) -> Tuple[List[str], List[str]]:
    """(device_defs, host_defs) for one profile row. mode in {dense, indexer, reuse}.
    params (all optional, ints): src ("r17"|"r18"), sk, nh, no_credit, push8, push_halves, b_rowsplit, rx_bypass,
    reuse_no_idx. NT stores are never enabled (rejected in ledger r160)."""
    if mode not in ("dense", "indexer", "reuse"):
        raise ValueError(f"mode {mode!r}")
    src = params.get("src", "r17")
    if src not in SRCS:
        raise ValueError(f"src {src!r}")
    allowed = {"src", "sk", "nh", "no_credit", "push8", "push_halves", "b_rowsplit", "rx_bypass", "reuse_no_idx", "nt"}
    extra = set(params) - allowed
    if extra:
        raise ValueError(f"unknown small-M params {sorted(extra)}")
    if params.get("nt", 0):
        raise ValueError("NT stores are rejected (ledger r160)")
    idx = 1 if mode == "indexer" else 0
    dense = 1 if mode == "dense" else 0
    nh = int(params.get("nh", 1))
    abi = [
        f"-DSM_R5X_INDEXER={idx}", f"-DSM_DENSE={dense}", f"-DSM_T1_B_HEADS={nh}",
        f"-DSM_SK_CFG={int(params.get('sk', 3))}", f"-DSM_PUSH8={int(params.get('push8', 0))}",
        f"-DSM_PUSH_HALVES={int(params.get('push_halves', 0))}", "-DSM_NT_STORES=0",
    ]
    if src == "r18":
        abi += [
            f"-DSM_B_ROWSPLIT={int(params.get('b_rowsplit', 0))}", f"-DSM_RX_BYPASS={int(params.get('rx_bypass', 0))}",
            f"-DSM_REUSE_NO_IDX={int(params.get('reuse_no_idx', 0))}",
        ]
    elif any(params.get(k) for k in ("b_rowsplit", "rx_bypass", "reuse_no_idx")):
        raise ValueError("b_rowsplit/rx_bypass/reuse_no_idx require src=r18")
    # device-only (Phase-1 build_device.sh defaults P0=1 P1=1 P2=0 STAMPS=0 R1=1 R2=0 R3=1 R4=1 U2=1)
    dev = abi + [
        "-DSM_U2_HOST_PTRS=1", "-DSM_P0=1", "-DSM_P1=1", "-DSM_P2=0", "-DSM_STAMPS=0", "-DSM_R1_RELAXED_SPIN=1",
        "-DSM_R2_LIGHT_RELEASE=0", "-DSM_R3_SRC_CREDIT=1", "-DSM_R4_LAST_WG_FENCE=1",
        f"-DSM_NO_CREDIT={int(params.get('no_credit', 1))}",
    ]
    # host gets every ABI-affecting switch (grid / thread count / ntiles are compile-time in sm_abi_v2.h)
    return dev, abi


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def cache_key(src: str, dev: List[str], host: List[str], tool: Dict[str, str]) -> str:
    h = hashlib.sha256()
    h.update(_sha(os.path.join(CSRC, "small", src, "SOURCE_SHA256.txt")).encode())
    for rel in ("host/sm_fusion_driver.cpp", "host/gf_comm.cpp"):
        h.update(_sha(os.path.join(CSRC, rel)).encode())
    h.update(repr((sorted(dev), sorted(host), sorted(tool.items()))).encode())
    return h.hexdigest()[:20]


def toolchain() -> Dict[str, str]:
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    mori_root = os.environ.get("MORI_ROOT", "/reference-mori")
    so = os.environ.get("MORI_CCO_SO", os.path.join(mori_root, "build", "src", "cco", "libmori_cco.so"))
    return {
        "hipcc": os.path.join(rocm, "bin", "hipcc"),
        "clangxx": os.path.join(rocm, "llvm", "bin", "clang++"),
        "bundler": os.path.join(rocm, "llvm", "bin", "clang-offload-bundler"),
        "rocm": rocm, "mori_root": mori_root, "mori_so": so,
    }


def jit_root() -> str:
    root = os.environ.get("AITER_JIT_DIR") or os.path.join(os.path.expanduser("~"), ".aiter", "jit")
    return os.path.join(root, "gemm_fusion_sp_qkv")


# ---------------------------------------------------------------- command plan (pure; unit-tested)
SMALL_DEVICE_TUS = (("sm_gemm", "sk_gemm_fp8.cu"), ("sm_ran", "ran_kernel.cu"))


def device_cmd_one(src: str, dev_defs: List[str], tool: Dict[str, str], out: str, name: str) -> List[str]:
    """The legacy compile command of ONE small device TU (name in sm_gemm | sm_ran): hipcc -> <out>/img/<name>.so (device fatbin embedded)."""
    sdir = os.path.join(CSRC, "small", src)
    flags = dev_defs + [
        f"-I{os.path.join(sdir, 'device')}", f"-I{os.path.join(sdir, 'interface')}",
        f"-I{os.path.join(CSRC, 'third_party', 'opus_phase1')}",  # Phase-1 opus headers (aiter's own opus.hpp differs)
        f"-I{tool['mori_root']}/include", "-D__HIPCC_RTC__", "-DBUILD_CCO_SDMA=1", "-DOPUS_TILE_CONTAINER=0",
        "-std=c++20", "-O3", "-fno-fast-math", "--offload-arch=gfx950",
    ]
    cu = dict(SMALL_DEVICE_TUS)[name]
    return [tool["hipcc"], *flags, "-fPIC", "-shared", os.path.join(sdir, "device", cu), "-o", os.path.join(out, "img", name + ".so")]


def device_cmds(src: str, dev_defs: List[str], tool: Dict[str, str], out: str) -> List[List[str]]:
    return [device_cmd_one(src, dev_defs, tool, out, n) for n, _cu in SMALL_DEVICE_TUS]


DEVICE_ONLY_FLAGS = ("--offload-device-only", "--no-gpu-bundle-output")


def device_only_cmd(legacy_cmd: List[str], hsaco: str) -> List[str]:
    """Derive the aot2 command from a legacy device command: the SAME flags and source, minus the host-side `-shared` link, with the device-only switches and the
    code object as the output (no host pass, no objcopy, no clang-offload-bundler). Derived mechanically so nothing else can drift."""
    if "-shared" not in legacy_cmd or "-o" not in legacy_cmd:
        raise ValueError("not a legacy device command")
    cmd = [a for a in legacy_cmd if a != "-shared"]
    i = cmd.index("-o")
    cmd[i + 1] = hsaco
    return [cmd[0], *DEVICE_ONLY_FLAGS, *cmd[1:]]


def host_cmds(src: str, host_defs: List[str], tool: Dict[str, str], out: str) -> List[List[str]]:
    sdir = os.path.join(CSRC, "small", src)
    flags = host_defs + [
        "-x", "c++", "-D__HIP_PLATFORM_AMD__=1", f"-I{tool['rocm']}/include", f"-I{os.path.join(out, 'gen')}",
        f"-I{os.path.join(sdir, 'interface')}", f"-I{os.path.join(sdir, 'host')}", f"-I{tool['mori_root']}/include",
        "-std=c++20", "-O3", "-fno-fast-math", "-fPIC", "-pthread",
    ]
    libs = ["-x", "none", tool["mori_so"], f"-L{tool['rocm']}/lib", "-lamdhip64", "-ldl", "-Wl,--no-undefined"]
    drv = [tool["clangxx"], *flags, "-shared", os.path.join(CSRC, "host", "sm_fusion_driver.cpp"),
           "-o", os.path.join(out, "sm_driver.so"), *libs]
    return [drv]


def host_compile_cmd(src: str, host_defs: List[str], tool: Dict[str, str], out: str) -> List[str]:
    """The host driver compiled to an object (same flags as host_cmds, `-c` instead of `-shared`, no link libs)."""
    sdir = os.path.join(CSRC, "small", src)
    flags = host_defs + [
        "-x", "c++", "-D__HIP_PLATFORM_AMD__=1", f"-I{tool['rocm']}/include", f"-I{os.path.join(out, 'gen')}",
        f"-I{os.path.join(sdir, 'interface')}", f"-I{os.path.join(sdir, 'host')}", f"-I{tool['mori_root']}/include",
        "-std=c++20", "-O3", "-fno-fast-math", "-fPIC", "-pthread",
    ]
    return [tool["clangxx"], *flags, "-c", os.path.join(CSRC, "host", "sm_fusion_driver.cpp"), "-o", os.path.join(out, "sm_driver.o")]


def hashes_compile_cmd(tool: Dict[str, str], out: str) -> List[str]:
    return [tool["clangxx"], "-x", "c++", "-std=c++20", "-O1", "-fPIC", "-c", os.path.join(out, "gen", "sm_hashes.cpp"), "-o", os.path.join(out, "sm_hashes.o")]


def host_link_cmd(tool: Dict[str, str], out: str, objs: List[str], so: str) -> List[str]:
    libs = ["-x", "none", tool["mori_so"], f"-L{tool['rocm']}/lib", "-lamdhip64", "-ldl", "-Wl,--no-undefined"]
    return [tool["clangxx"], "-shared", "-pthread", "-o", so, *[os.path.join(out, o) for o in objs], *libs]


def hashes_header_extern() -> str:
    """gen/sm_expected_hashes.h of the decoupled recipes: the vendored driver only passes SM_EXPECTED_*_SHA256 to sm_image_matches(const char*, const char*), so the macros can
    name extern strings that sm_hashes.o (generated after the device builds) defines: the host compile no longer depends on the device builds."""
    return ('#pragma once\nextern "C" const char sm_expected_gemm_sha256[];\nextern "C" const char sm_expected_ran_sha256[];\n'
            '#define SM_EXPECTED_GEMM_SHA256 sm_expected_gemm_sha256\n#define SM_EXPECTED_RAN_SHA256 sm_expected_ran_sha256\n')


def hashes_source(gemm_sha: str, ran_sha: str) -> str:
    return ('extern "C" {\nextern const char sm_expected_gemm_sha256[];\nextern const char sm_expected_ran_sha256[];\n'
            f'const char sm_expected_gemm_sha256[] = "{gemm_sha}";\nconst char sm_expected_ran_sha256[] = "{ran_sha}";\n}}\n')


def comm_cmd(tool: Dict[str, str], out: str) -> List[str]:
    flags = ["-x", "c++", "-D__HIP_PLATFORM_AMD__=1", f"-I{tool['rocm']}/include", f"-I{tool['mori_root']}/include",
             "-std=c++20", "-O3", "-fPIC", "-pthread"]
    libs = ["-x", "none", tool["mori_so"], f"-L{tool['rocm']}/lib", "-lamdhip64", "-ldl", "-Wl,--no-undefined"]
    return [tool["clangxx"], *flags, "-shared", os.path.join(CSRC, "host", "gf_comm.cpp"), "-o", os.path.join(out, "gf_comm.so"), *libs]


def _run(cmd: List[str], log: List[str]):
    log.append(" ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise BuildError(f"command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout[-4000:]}")
    return p.stdout


def _unbundle(tool, so, fatbin, hsaco, log):
    _run(["objcopy", "-O", "binary", "--only-section=.hip_fatbin", so, fatbin], log)
    listing = _run([tool["bundler"], "--type=o", f"--input={fatbin}", "--list"], log)
    tgt = next((ln.strip() for ln in listing.splitlines() if "gfx950" in ln), None)
    if not tgt:
        raise BuildError(f"no gfx950 bundle entry in {fatbin}")
    _run([tool["bundler"], "--type=o", "--unbundle", f"--targets={tgt}", f"--input={fatbin}", f"--output={hsaco}"], log)


def build_small(params: dict, mode: str, tool: Dict[str, str] = None, recipe: str = None) -> SmallBuild:
    """Build (or reuse) the 2 HSACOs + host driver for one parameter set. Safe under concurrent ranks (atomic rename). recipe: see the module doc; the default
    (aot1) runs the unit's TUs in parallel and hits the artifacts prebuild_profile() produced."""
    recipe = recipe or DEFAULT_RECIPE
    if recipe != "legacy":
        from . import prebuild

        return prebuild.build_small(params, mode, tool, recipe)
    tool = tool or toolchain()
    src = params.get("src", "r17")
    dev, host = small_defs(params, mode)
    key = cache_key(src, dev, host, tool)
    final = os.path.join(jit_root(), key)
    sb = SmallBuild(key, final, os.path.join(final, "img", "sm_gemm.hsaco"), os.path.join(final, "img", "sm_ran.hsaco"), os.path.join(final, "sm_driver.so"))
    if os.path.exists(os.path.join(final, "BUILD_OK")):
        return sb
    os.makedirs(jit_root(), exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=key + ".", dir=jit_root())
    log: List[str] = []
    try:
        os.makedirs(os.path.join(tmp, "img")); os.makedirs(os.path.join(tmp, "gen"))
        for cmd, n in zip(device_cmds(src, dev, tool, tmp), ("sm_gemm", "sm_ran")):
            _run(cmd, log)
            so = os.path.join(tmp, "img", n + ".so")
            _unbundle(tool, so, os.path.join(tmp, "img", n + ".fatbin"), os.path.join(tmp, "img", n + ".hsaco"), log)
        gh, rh = _sha(os.path.join(tmp, "img", "sm_gemm.hsaco")), _sha(os.path.join(tmp, "img", "sm_ran.hsaco"))
        with open(os.path.join(tmp, "gen", "sm_expected_hashes.h"), "w") as f:
            f.write(f'#pragma once\n#define SM_EXPECTED_GEMM_SHA256 "{gh}"\n#define SM_EXPECTED_RAN_SHA256 "{rh}"\n')
        for cmd in host_cmds(src, host, tool, tmp):
            _run(cmd, log)
        with open(os.path.join(tmp, "build-command.txt"), "w") as f:
            f.write("\n".join(log) + "\n")
        with open(os.path.join(tmp, "BUILD_OK"), "w") as f:
            f.write(key + "\n")
        try:
            os.rename(tmp, final)
        except OSError:  # another rank finished first
            if not os.path.exists(os.path.join(final, "BUILD_OK")):
                raise
            shutil.rmtree(tmp, ignore_errors=True)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return sb


def comm_key(tool: Dict[str, str]) -> str:
    return hashlib.sha256(_sha(os.path.join(CSRC, "host", "gf_comm.cpp")).encode() + repr(sorted(tool.items())).encode()).hexdigest()[:20]


def build_comm(tool: Dict[str, str] = None, recipe: str = None) -> str:
    """Build (or reuse) gf_comm.so; returns its path (one TU: the recipe does not change the artifact or its key). The default recipes go through the rank-safe engine
    (one process compiles, the others wait on the unit lock); `legacy` is the old atomic-rename-only path."""
    recipe = recipe or DEFAULT_RECIPE
    if recipe != "legacy":
        from . import prebuild

        return prebuild.build_comm(tool)
    tool = tool or toolchain()
    h = hashlib.sha256(_sha(os.path.join(CSRC, "host", "gf_comm.cpp")).encode() + repr(sorted(tool.items())).encode()).hexdigest()[:20]
    final = os.path.join(jit_root(), "comm_" + h)
    so = os.path.join(final, "gf_comm.so")
    if os.path.exists(so):
        return so
    os.makedirs(jit_root(), exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="comm_" + h + ".", dir=jit_root())
    try:
        _run(comm_cmd(tool, tmp), [])
        try:
            os.rename(tmp, final)
        except OSError:
            if not os.path.exists(so):
                raise
            shutil.rmtree(tmp, ignore_errors=True)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return so


# ======================================================================= large-M (LG-QSDMA-r11)
LARGE_SRC = "r11"


@dataclass(frozen=True)
class LargeBuild:
    key: str
    dir: str
    hsaco: str
    hsaco_sha256: str
    driver_so: str


def large_defs(params: dict, mode: str) -> List[str]:
    """-D flags (identical on device and host where the host reads them). mode in {dense, indexer, reuse}: reuse uses the DENSE build
    (LG_X_INDEXER=0, compact F=2304). params: no_gridread (0|1|2, default 1 = r11/r10 default). Everything else is the r11 recipe default."""
    if mode not in ("dense", "indexer", "reuse"):
        raise ValueError(f"mode {mode!r}")
    allowed = {"plan", "n", "half_first", "quarter_first", "no_gridread"}
    extra = set(params) - allowed
    if extra:
        raise ValueError(f"unknown large-M params {sorted(extra)}")
    ng = int(params.get("no_gridread", 1))
    if ng not in (0, 1, 2):
        raise ValueError("no_gridread")
    return [
        f"-DLG_X_INDEXER={1 if mode == 'indexer' else 0}", f"-DLG_NO_GRIDREAD={ng}", "-DBUILD_CCO_SDMA=1", "-DLG_REMOTE_HALF=0",
        "-DLG_R6_POSTER=1", "-DLG_R6_ARRIVAL_ACQ=0", "-DLG_X_FORCE_TIMEOUT_G1=0", "-DLG_X_FORCE_TIMEOUT_G0=0", "-DLG_R8_FOLD=0",
        "-DLG_X_NEG_DOUBLE_PUT=0", "-DLG_X_POSTER_LAST=0", "-DLG_DEBUG_POSTCOUNT=0", "-DLG_NEGCTRL_NOCAS=0", "-DLG_L1C_SWIZZLE=0",
        "-DLG_L2A_BYPASS=0", "-DLG_L2B_ONE_RELEASE=1", "-DLG_STAMPS=0", "-DLG_NEGCTRL_NOWB=0", "-DLG_NEGCTRL_NOWAIT=0", "-DLG_NEGCTRL_ISO=0", "-DLG_ISO_DELAY_ITERS=0",
    ]


def large_cache_key(defs: List[str], tool: Dict[str, str]) -> str:
    h = hashlib.sha256()
    h.update(_sha(os.path.join(CSRC, "large", LARGE_SRC, "SOURCE_SHA256.txt")).encode())
    h.update(_sha(os.path.join(CSRC, "host", "lg_fusion_driver.cpp")).encode())
    h.update(repr((sorted(defs), sorted(tool.items()))).encode())
    return h.hexdigest()[:20]


def large_device_cmd(defs: List[str], tool: Dict[str, str], out: str) -> List[str]:
    ldir = os.path.join(CSRC, "large", LARGE_SRC)
    flags = defs + [
        f"-I{os.path.join(ldir, 'device')}", f"-I{os.path.join(ldir, 'interface')}",
        f"-I{os.path.join(CSRC, 'third_party', 'opus_phase1')}", f"-I{tool['mori_root']}/include",
        "-D__HIPCC_RTC__", "-DOPUS_TILE_CONTAINER=0", "-std=c++20", "-O3", "-fno-fast-math", "--offload-arch=gfx950",
    ]
    return [tool["hipcc"], *flags, "-fPIC", "-shared", os.path.join(ldir, "device", "gemm_quad_fp8_comm.cu"),
            "-o", os.path.join(out, "img", "lg_gemm_quad_fp8_comm.so")]


def large_host_cmd(defs: List[str], tool: Dict[str, str], out: str) -> List[str]:
    ldir = os.path.join(CSRC, "large", LARGE_SRC)
    hostdefs = [d for d in defs if d.split("=")[0] in ("-DLG_REMOTE_HALF", "-DLG_R6_POSTER", "-DLG_X_INDEXER", "-DLG_R8_FOLD", "-DLG_DEBUG_POSTCOUNT")]
    flags = hostdefs + ["-x", "c++", "-D__HIP_PLATFORM_AMD__=1", f"-I{tool['rocm']}/include", f"-I{os.path.join(ldir, 'interface')}",
                        f"-I{os.path.join(ldir, 'host')}", f"-I{tool['mori_root']}/include", "-std=c++20", "-O2", "-fno-fast-math", "-fPIC", "-pthread"]
    libs = ["-x", "none", tool["mori_so"], f"-L{tool['rocm']}/lib", "-lamdhip64", "-ldl", "-Wl,--no-undefined"]
    return [tool["clangxx"], *flags, "-shared", os.path.join(CSRC, "host", "lg_fusion_driver.cpp"), "-o", os.path.join(out, "liblg_comm.so"), *libs]


def large_device_only_cmd(defs: List[str], tool: Dict[str, str], out: str) -> List[str]:
    return device_only_cmd(large_device_cmd(defs, tool, out), os.path.join(out, "img", "lg_gemm_quad_fp8_comm.hsaco"))


def build_large(params: dict, mode: str, tool: Dict[str, str] = None, recipe: str = None) -> LargeBuild:
    """Build (or reuse) the large-M HSACO (all plan entries n1/n2/n4/n2h/n4h/n3q + quiet_ready + drain) and host driver for one mode."""
    recipe = recipe or DEFAULT_RECIPE
    if recipe != "legacy":
        from . import prebuild

        return prebuild.build_large(params, mode, tool, recipe)
    tool = tool or toolchain()
    defs = large_defs(params, mode)
    key = large_cache_key(defs, tool)
    final = os.path.join(jit_root(), "lg_" + key)
    mk = lambda d, sha="": LargeBuild(key, d, os.path.join(d, "img", "lg_gemm_quad_fp8_comm.hsaco"), sha, os.path.join(d, "liblg_comm.so"))  # noqa: E731
    ok = os.path.join(final, "BUILD_OK")
    if os.path.exists(ok):
        return mk(final, open(ok).read().split()[-1])
    os.makedirs(jit_root(), exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="lg_" + key + ".", dir=jit_root())
    log: List[str] = []
    try:
        os.makedirs(os.path.join(tmp, "img"))
        _run(large_device_cmd(defs, tool, tmp), log)
        so = os.path.join(tmp, "img", "lg_gemm_quad_fp8_comm.so")
        _unbundle(tool, so, os.path.join(tmp, "img", "lg.fatbin"), os.path.join(tmp, "img", "lg_gemm_quad_fp8_comm.hsaco"), log)
        sha = _sha(os.path.join(tmp, "img", "lg_gemm_quad_fp8_comm.hsaco"))
        _run(large_host_cmd(defs, tool, tmp), log)
        with open(os.path.join(tmp, "build-command.txt"), "w") as f:
            f.write("\n".join(log) + "\n")
        with open(os.path.join(tmp, "BUILD_OK"), "w") as f:
            f.write(key + " " + sha + "\n")
        try:
            os.rename(tmp, final)
        except OSError:
            if not os.path.exists(ok):
                raise
            shutil.rmtree(tmp, ignore_errors=True)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return mk(final, open(ok).read().split()[-1])
