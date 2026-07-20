# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Subprocess wrapper for gemm_a16w16_tune.py.

Runs the tuner in a child process so GPU-level crashes (SIGABRT, SIGSEGV)
are caught and retried automatically.  Same pattern as
``gradlib/gradlib/gemm_tuner.py``.

Usage:
    python3 csrc/gemm_a16w16/gemm_tuner.py [all gemm_a16w16_tune.py args]
"""

import gc
import multiprocessing as mp
import time

import torch


def run_tuner():
    from gemm_a16w16_tune import GemmA16W16Tuner

    key = [
        "gfx",
        "cu_num",
        "M",
        "N",
        "K",
        "bias",
        "dtype",
        "outdtype",
        "scaleAB",
        "bpreshuffle",
    ]
    resultList = [
        "libtype",
        "solidx",
        "splitK",
        "us",
        "kernelName",
        "err_ratio",
        "tflops",
        "bw",
    ]
    description = "Multi-backend bf16 GEMM tuner (csrc)"
    tuner = GemmA16W16Tuner("GemmA16W16Tuner", key, resultList, description=description)
    args = tuner.parse_args()
    tuner.run(args, False)


def clean():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.cuda, "memory_allocated"):
        torch.cuda.synchronize()
    try:
        if hasattr(mp, "resource_tracker"):
            mp.resource_tracker.ensure_running()
            if hasattr(mp.resource_tracker, "_CLEANUP_FUNCS"):
                for name in list(mp.resource_tracker._CLEANUP_FUNCS.keys()):
                    try:
                        mp.resource_tracker._CLEANUP_FUNCS.pop(name)()
                    except Exception:
                        pass
    except Exception as e:
        print(f"Resource cleanup warning: {e}")


if __name__ == "__main__":
    retries = 0
    MAX_TRY = 30
    mp.set_start_method("spawn", force=True)
    process = None
    while retries <= MAX_TRY:
        try:
            process = mp.Process(target=run_tuner, args=(), daemon=False)
            process.start()
            process.join()
            if process.exitcode < 0:
                time.sleep(0.5 * retries)
                print(
                    f"!Process killed by signal {-process.exitcode}, "
                    f"retrying ({retries}/{MAX_TRY})"
                )
                clean()
                retries += 1
            elif process.exitcode > 0:
                print(
                    f"!Process exited with code {process.exitcode} "
                    f"(tuning finished with warnings)"
                )
                break
            else:
                break
        except Exception as e:
            print(f"Process creation failed: {e}")
            retries += 1
            clean()
            time.sleep(1)
        finally:
            if process and process.is_alive():
                process.terminate()
                process.join(timeout=5)

    clean()
    print(f"retried num is {retries}")
