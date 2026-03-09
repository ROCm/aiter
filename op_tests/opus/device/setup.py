# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Build opus device-test kernels into a shared library via hipcc.
No torch/pybind11 headers -- the .so exports only extern "C" functions,
loaded at runtime via ctypes from test_opus_device.py.

Usage:
    python setup.py          # build opus_device_test.so
    python setup.py --clean  # remove built artifacts
"""

import os
import subprocess
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_CSRC = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "..", "..", "csrc", "include")
)
_SO_NAME = "opus_device_test.so"

_CU_SOURCES = [
    "test_mfma_f16.cu",
    "test_mfma_f32.cu",
    "test_mfma_f8.cu",
    "test_mxfp.cu",
    "test_load_store_if.cu",
    "test_vector_add.cu",
    "test_async_load.cu",
    "test_dtype_convert.cu",
    "test_mdiv.cu",
    "test_numeric_limits.cu",
    "test_workgroup_barrier.cu",
]


def _detect_arch():
    try:
        out = subprocess.check_output(
            ["rocminfo"], stderr=subprocess.DEVNULL, text=True
        )
        for line in out.splitlines():
            if "gfx" in line and "Name:" in line:
                name = line.split()[-1].strip()
                if name.startswith("gfx"):
                    return name
    except Exception:
        pass
    return "native"


def _find_hipcc():
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    candidate = os.path.join(rocm, "bin", "hipcc")
    if os.path.isfile(candidate):
        return candidate
    try:
        return subprocess.check_output(
            ["which", "hipcc"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        pass
    return "hipcc"


def _compile_one(args):
    """Compile a single .cu -> .o.  Used as a worker function for parallel builds."""
    src, obj, hipcc, arch, verbose = args
    cmd = [
        hipcc,
        f"--offload-arch={arch}",
        "-fPIC",
        "-O3",
        "-D__HIPCC_RTC__",
        f"-I{_REPO_CSRC}",
        f"-I{_THIS_DIR}",
        "-c",
        src,
        "-o",
        obj,
    ]
    if verbose:
        print(f"[setup]   {os.path.basename(src)}")
    t0 = time.monotonic()
    subprocess.check_call(cmd)
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    return os.path.basename(src), elapsed_ms


def build(verbose=False, jobs=None):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    hipcc = _find_hipcc()
    arch = _detect_arch()
    so_path = os.path.join(_THIS_DIR, _SO_NAME)

    if jobs is None:
        jobs = min(len(_CU_SOURCES), os.cpu_count() or 4)

    if verbose:
        print(f"[setup] arch={arch}, jobs={jobs}")

    t0 = time.monotonic()

    # Parallel compile: each .cu -> .o
    tasks = []
    for s in _CU_SOURCES:
        src = os.path.join(_THIS_DIR, s)
        obj = os.path.join(_THIS_DIR, s.replace(".cu", ".o"))
        tasks.append((src, obj, hipcc, arch, verbose))

    objs = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_compile_one, t): t for t in tasks}
        for fut in as_completed(futures):
            name, ms = fut.result()
            objs.append(futures[fut][1])  # .o path
            if verbose:
                print(f"[setup]   {name} done in {ms}ms")

    compile_ms = int((time.monotonic() - t0) * 1000)

    # Link: all .o -> .so
    t_link = time.monotonic()
    link_cmd = [
        hipcc,
        f"--offload-arch={arch}",
        "-shared",
        "-fPIC",
        *objs,
        "-o",
        so_path,
    ]
    subprocess.check_call(link_cmd)
    link_ms = int((time.monotonic() - t_link) * 1000)

    total_ms = int((time.monotonic() - t0) * 1000)
    print(
        f"[setup] built {_SO_NAME} in {total_ms}ms "
        f"(compile {compile_ms}ms, link {link_ms}ms, jobs={jobs})"
    )

    # Clean up .o files
    for o in objs:
        try:
            os.remove(o)
        except OSError:
            pass

    return so_path


def clean():
    import glob as g

    removed = []
    for pat in [
        _SO_NAME,
        "build/",
        "*.egg-info/",
        "opus_device_test*.so",
        "opus_device_test*.pyd",
        "*.o",
    ]:
        for p in g.glob(os.path.join(_THIS_DIR, pat)):
            try:
                if os.path.isdir(p):
                    import shutil

                    shutil.rmtree(p)
                else:
                    os.remove(p)
                removed.append(os.path.basename(p))
            except OSError:
                pass
    if removed:
        print("Cleaned:", " ".join(removed))
    else:
        print("Nothing to clean.")


if __name__ == "__main__":
    if "--clean" in sys.argv:
        clean()
    else:
        verbose = "-v" in sys.argv or "--verbose" in sys.argv
        jobs = None
        for arg in sys.argv[1:]:
            if arg.startswith("-j"):
                try:
                    jobs = int(arg[2:])
                except ValueError:
                    pass
        build(verbose=verbose, jobs=jobs)
