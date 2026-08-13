#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Run a FlyDSL MoE harness with rotating input buffers.

aiter's run_perftest only rotates tensors passed as *args to it; the FlyDSL
harnesses hand it a zero-arg closure, so every iteration re-reads the same
buffers. This runner patches the grouped a8w4 GEMM launcher (imported at call
time in grouped_moe_gfx1250._grouped_a8w4_tdm_moe, so a module-attribute patch
takes effect) and cycles each call through N identical clones of its inputs --
what run_perftest's deepcopy rotation would have done. Both the TP
(test_flydsl_grouped_gemm_gfx1250.py) and EP (test_moe_ep.py) harnesses reach
it, in both --scenario bench and per-kernel modes.

Rotated by default: the activation payload + its scale, and the weights +
their scales -- i.e. every input is cache-cold across iterations. NOT rotated:
output buffers, since the caller reads its results back from the tensor it
passed in.

Clones are exact copies, so results and accuracy gates are unchanged; only the
addresses differ, which is the point. They are snapshots taken at the first
call: valid because these harnesses feed identical data every iteration. A
harness that varied its input per iteration would silently measure stale data.

usage: ubench_moe_rotate.py <harness.py> [harness args...]
  env: ROTATE=n    activation buffers; 0 = auto (enough to exceed 2x L2),
                   1 = activations not rotated
       ROTATE_W=n  weight buffers, default 4 (cache-cold); 0/1 = static.
                   Costs n x the weight bytes in VRAM (~3GB per copy at TP4).
       ROTATE_BUDGET_GB=n  cap on total clone bytes (default 32); pools shrink
                   rather than OOM a long sweep.
       ROTATE_DEBUG=1      print each pool as it is allocated.
"""

import itertools
import os
import runpy
import sys

import torch

import aiter.ops.flydsl.batched_gemm_mxfp4 as bg

ROTATE = int(os.environ.get("ROTATE", "0"))
ROTATE_W = int(os.environ.get("ROTATE_W", "4"))
BUDGET = float(os.environ.get("ROTATE_BUDGET_GB", "32")) * 2**30
DEBUG = bool(os.environ.get("ROTATE_DEBUG"))
L2 = torch.cuda.get_device_properties(0).L2_cache_size

_orig = bg.flydsl_grouped_gemm_a8w4_masked
_pools: dict = {}
_ctr = itertools.count()
_bytes = 0


def _auto_n(t):
    """Enough clones to exceed 2x L2. A tensor already larger than that cannot
    be resident, so it gets no clones."""
    return min(16, max(1, (2 * L2) // max(t.nbytes, 1) + 1))


def _pool(t, n):
    """N identical clones of t, cached by storage identity. Returns None when
    no rotation applies, so the caller keeps the original tensor."""
    global _bytes
    if t is None or not isinstance(t, torch.Tensor) or n <= 1:
        return None
    # Keyed by shape/dtype, NOT data_ptr: run_perftest rotates its own *args
    # (the EP e2e path), so a data_ptr key would clone afresh for each of its
    # copies -- inside the timed loop, corrupting the measurement. Those copies
    # are deepcopies of one another, so one pool per logical tensor is correct
    # here and gets built during warmup. (Assumes, as this whole tool does,
    # that same-shape inputs carry the same data.)
    key = (tuple(t.shape), str(t.dtype), t.device)
    pool = _pools.get(key)
    if pool is None:
        # Keep the clone footprint under budget; a long sweep allocates fresh
        # buffers per shape, and pools pin them (pool[0] is the original).
        while n > 1 and _bytes + (n - 1) * t.nbytes > BUDGET:
            n -= 1
        if n <= 1:
            return None
        pool = [t] + [t.clone() for _ in range(n - 1)]
        _bytes += (n - 1) * t.nbytes
        _pools[key] = pool
        if DEBUG:
            print(
                f"[rotate] pool x{n} for {tuple(t.shape)} {t.dtype} "
                f"{t.nbytes / 2**20:.1f}MB -> {(n - 1) * t.nbytes / 2**30:.2f}GB "
                f"clones (total {_bytes / 2**30:.2f}GB)",
                flush=True,
            )
    return pool


def _pick(t, n, i):
    pool = _pool(t, n)
    return t if pool is None else pool[i % len(pool)]


def rotating_gemm(out, a, w, a_scale, w_scale, psum, *args, **kwargs):
    i = next(_ctr)
    # a and its scale share one count so the pair rotates together.
    n_a = ROTATE or _auto_n(a)
    a, a_scale = _pick(a, n_a, i), _pick(a_scale, n_a, i)
    if ROTATE_W:
        w, w_scale = _pick(w, ROTATE_W, i), _pick(w_scale, ROTATE_W, i)
    return _orig(out, a, w, a_scale, w_scale, psum, *args, **kwargs)


if ROTATE != 1 or ROTATE_W:
    bg.flydsl_grouped_gemm_a8w4_masked = rotating_gemm
    print(
        f"[rotate] activations={'static' if ROTATE == 1 else (ROTATE or 'auto')} "
        f"weights={'static' if ROTATE_W <= 1 else ROTATE_W} "
        f"budget={BUDGET / 2**30:.0f}GB",
        flush=True,
    )
else:
    print("[rotate] pass-through: nothing rotated", flush=True)

harness = sys.argv[1]
sys.argv = [harness] + sys.argv[2:]
runpy.run_path(harness, run_name="__main__")
