# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Cold-JIT build-wall benchmark: CK module_rmsnorm vs opus module_rmsnorm_opus
# (aiter issue #4055). Deletes each module's cached .so + build dir and times the
# first call that triggers a rebuild.
#
#   python op_tests/bench_rmsnorm_compile.py            # both
#   python op_tests/bench_rmsnorm_compile.py --only opus

import argparse
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


def trigger_ck(x, w):
    os.environ["AITER_RMSNORM_BACKEND"] = "ck"
    import aiter

    aiter.rms_norm(x, w, 1e-6)  # -> module_rmsnorm (CK, ~1360 TUs)


def trigger_opus(x, w):
    os.environ["AITER_RMSNORM_BACKEND"] = "opus"
    import aiter

    aiter.rms_norm(x, w, 1e-6)  # -> module_rmsnorm_opus (1 torch-free TU)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["ck", "opus"], default=None)
    args = ap.parse_args()

    results = {}
    if args.only in (None, "opus"):
        results["opus"] = bench("module_rmsnorm_opus", trigger_opus)
    if args.only in (None, "ck"):
        results["ck"] = bench("module_rmsnorm", trigger_ck)

    if "ck" in results and "opus" in results and results["opus"]:
        print(f"\nspeedup: {results['ck'] / results['opus']:.0f}x faster cold build")
