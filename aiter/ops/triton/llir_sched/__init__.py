# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""llirSched: the gfx950 LLVM-IR MFMA <-> memory scheduler, as a Triton pass plugin.

The plugin (``LlirSchedPlugin.cpp``, from ROCm/gfx950-gluon-tutorials) reorders the
MFMA, LDS and global-load instructions of a Gluon GEMM hot loop and pins the order
with ``llvm.amdgcn.sched.barrier``. It is built here with ``LLIR_SCHED_REQUIRE_OPT_IN``:
Triton loads it for every kernel in the process, but it only schedules kernels that
carry the ``aiter-llir-sched`` function attribute.

Kernels opt in by launching with ``**llir_sched.compile_options()``, which returns
``{"llvm_fn_attrs": (("aiter-llir-sched", <plugin id>),)}`` when the plugin is
available and ``{}`` otherwise, so a kernel always compiles, scheduled or not. The
plugin id is part of the attribute, so Triton's cache recompiles when the plugin
changes.

The plugin must be built against the exact LLVM of the installed Triton. It is
compiled on first use (a few tens of seconds) with the LLVM package Triton was built
against (``~/.triton/llvm/llvm-<hash>-*``, matched by the hash embedded in
``libtriton.so``) and linked against ``libtriton.so``, so no ``RTLD_GLOBAL`` import
trick is needed. Triton must be built with default symbol visibility
(``TRITON_EXT_ENABLED=1``; PyPI release wheels are).

Environment:
  AITER_LLIR_SCHED=0            disable the plugin (kernels compile unscheduled)
  AITER_LLIR_SCHED_LLVM_DIR     LLVM install to build against (default: auto-detect)
  AITER_LLIR_SCHED_CACHE        build cache directory (default: ~/.aiter/llir_sched)
"""

import functools
import glob
import hashlib
import mmap
import os
import re
import subprocess

import triton

from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

OPT_IN_ATTR = "aiter-llir-sched"
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LlirSchedPlugin.cpp")


def _libtriton_dir():
    return os.path.join(os.path.dirname(triton.__file__), "_C")


def _find_llvm_dir():
    env = os.environ.get("AITER_LLIR_SCHED_LLVM_DIR")
    if env:
        return env
    libtriton = os.path.join(_libtriton_dir(), "libtriton.so")
    candidates = []
    with open(libtriton, "rb") as f, mmap.mmap(
        f.fileno(), 0, access=mmap.ACCESS_READ
    ) as m:
        for d in sorted(glob.glob(os.path.expanduser("~/.triton/llvm/llvm-*"))):
            match = re.match(r"llvm-([0-9a-f]{8})-", os.path.basename(d))
            if (
                match
                and os.path.exists(os.path.join(d, "bin", "llvm-config"))
                and m.find(match.group(1).encode()) != -1
            ):
                candidates.append(d)
    if len(candidates) != 1:
        raise RuntimeError(
            "llirSched: cannot pick the LLVM that libtriton.so was built with "
            f"(candidates: {candidates or 'none under ~/.triton/llvm'}); "
            "set AITER_LLIR_SCHED_LLVM_DIR"
        )
    return candidates[0]


@functools.cache
def plugin_path():
    """Build (once) and return the plugin .so, or None if disabled or unavailable."""
    if os.environ.get("AITER_LLIR_SCHED", "1") == "0":
        return None
    try:
        llvm_dir = _find_llvm_dir()
        lib_dir = _libtriton_dir()
        libtriton = os.path.join(lib_dir, "libtriton.so")
        with open(_SRC, "rb") as f:
            src = f.read()
        key = hashlib.sha256(
            src
            + llvm_dir.encode()
            + libtriton.encode()
            + str(os.path.getmtime(libtriton)).encode()
        ).hexdigest()[:16]
        out_dir = os.path.join(
            os.environ.get(
                "AITER_LLIR_SCHED_CACHE", os.path.expanduser("~/.aiter/llir_sched")
            ),
            key,
        )
        so = os.path.join(out_dir, "libLlirSched.so")
        if not os.path.exists(so):
            os.makedirs(out_dir, exist_ok=True)
            cxxflags = subprocess.check_output(
                [os.path.join(llvm_dir, "bin", "llvm-config"), "--cxxflags"], text=True
            ).split()
            tmp = f"{so}.tmp{os.getpid()}"
            cmd = (
                ["g++", "-shared", "-fPIC", "-fvisibility=default"]
                + cxxflags
                + ["-DLLIR_SCHED_REQUIRE_OPT_IN", "-o", tmp, _SRC]
                + [f"-L{lib_dir}", "-l:libtriton.so", f"-Wl,-rpath,{lib_dir}"]
            )
            _LOGGER.info("llirSched: building %s against %s", so, llvm_dir)
            subprocess.run(cmd, check=True, capture_output=True)
            os.replace(tmp, so)
        return so
    except (OSError, RuntimeError, subprocess.CalledProcessError) as e:
        # Never block a kernel launch on the plugin.
        _LOGGER.warning(
            "llirSched: plugin unavailable, kernels run unscheduled (%s)", e
        )
        return None


def compile_options():
    """Launch kwargs that opt a kernel into llirSched; {} when the plugin is unavailable."""
    so = plugin_path()
    if so is None:
        return {}
    current = os.environ.get("LLVM_PASS_PLUGIN_PATH")
    if current and current != so:
        _LOGGER.warning(
            "llirSched: LLVM_PASS_PLUGIN_PATH already set to %s; not loading %s",
            current,
            so,
        )
        return {}
    os.environ["LLVM_PASS_PLUGIN_PATH"] = so
    return {"llvm_fn_attrs": ((OPT_IN_ATTR, os.path.basename(os.path.dirname(so))),)}
