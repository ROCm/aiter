# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Cold-JIT build-wall benchmark for the opus rmsnorm module (aiter issue #4055).
# Deletes the cached .so + build dir and times the first call that triggers a
# rebuild. opus is a single torch-free ctypes TU (~1s cold) that replaced the
# legacy CK module_rmsnorm (~1360 TUs / 4972 instantiations, ~225s cold).
#
#   python op_tests/bench_rmsnorm_compile.py

import os
import shutil
import time

import torch

from aiter.jit.core import get_user_jit_dir


def _purge(md_name):
    jit = get_user_jit_dir()
    for p in (
        os.path.join(jit, f"{md_name}.so"),
        os.path.join(jit, "build", md_name),
    ):
        if os.path.isfile(p):
            os.remove(p)
        elif os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)


def bench(md_name, trigger):
    _purge(md_name)
    x = torch.randn((16, 8192), dtype=torch.bfloat16, device="cuda")
    w = torch.ones(8192, dtype=torch.bfloat16, device="cuda")
    t0 = time.perf_counter()
    trigger(x, w)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"{md_name:22s} cold build + first call: {dt:7.1f}s")
    return dt


def trigger_opus(x, w):
    import aiter

    aiter.rms_norm(x, w, 1e-6)  # -> module_rmsnorm (1 torch-free TU)


if __name__ == "__main__":
    bench("module_rmsnorm", trigger_opus)
