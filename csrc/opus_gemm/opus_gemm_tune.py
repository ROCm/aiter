# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""[DEBUG-ONLY] Single-shape / single-kid opus a16w16 tuner.

This script used to be the production opus tuning entry point and wrote
directly into a private CSV under aiter/ops/opus/configs/. Production
tuning has moved to gradlib:

    python3 gradlib/gemm_tuner.py --libtype opus
    # or as part of a multi-backend tune:
    python3 gradlib/gemm_tuner.py --libtype all

gradlib writes to aiter/configs/bf16_tuned_gemm.csv (or whatever the
user passes via --tuned_file / GTUNE_TUNED), stamping every opus row
with `libtype=='opus'`. The opus runtime dispatch
(aiter/ops/opus/common.py) reads those rows from the global CSV.

This file is retained for two reasons only:

  1. **Single-(M,N,K) smoke / debug**: hand-running a specific kid against
     a specific shape (-m M -n N -k K --kid K --splitK S) to compare
     against the gradlib winner or to investigate a bug.
  2. **Source of truth for tune-time helpers**: candidate_kids_for_shape,
     candidate_splitK, kid_rejects_shape / kid_rejects_bias, and
     _ensure_kids_compiled live here. They are imported by:
       - gradlib's GemmTuner (gradlib/gradlib/GemmTuner.py) for the
         production `--libtype opus` path,
       - this script's own `tune()` for the single-shape debug path.
     csrc/opus_gemm/opus_gemm_common.py only owns the data constants
     (SPLITK_KIDS / NON_SPLITK_KIDS / BIAS_AWARE_KIDS /
     HEURISTIC_DEFAULT_KIDS) plus _opus_sidecar_path(); it does NOT
     re-export any tune-time helper.

The default output path is /tmp/opus_debug_tuned.csv so this script can
never accidentally pollute the global aiter/configs/ tree.
"""

import json
import logging
import os
import sys

import pandas as pd
import torch
from aiter import dtypes, logger
from aiter.utility.base_tuner import GemmCommonTuner, INVALID_TIME
from aiter.utility.mp_tuner import mp_tuner
from aiter.ops.opus.gemm_op_a16w16 import (
    opus_gemm_a16w16_tune as _opus_gemm_a16w16_tune,
)

# opus_gemm_common is a sibling file in csrc/opus_gemm/.
from opus_gemm_common import (
    a16w16_kernels_list,
    a16w16_kernels_list_nooob,
    a16w16_kernels_list_cpol,
    a16w16_kernels_list_cpol_nooob,
    a16w16_flatmm_kernels_list,
    a16w16_flatmm_splitk_kernels_list,
    a16w16_flatmm_splitk_kernels_list_nooob,
    a16w16_persistent_kernels_list,
    a16w16_persistent_kernels_list_cpol,
    a16w16_persistent_kernels_list_nooob,
    a16w16_persistent_kernels_list_cpol_nooob,
    gfx942_nosplit_kernels_list,
    gfx942_splitk_kernels_list,
    gfx1250_kernels_list,
    SPLITK_KIDS,
    NON_SPLITK_KIDS,
    BIAS_AWARE_KIDS,
    HEURISTIC_DEFAULT_KIDS,
    _opus_sidecar_path,
)

# Tune-time host helpers (defined here, not in opus_gemm_common.py).

# Occupancy threshold tile (used by candidate_kids_for_shape).
OCCUPANCY_TILE_BM = 128
OCCUPANCY_TILE_BN = 128

BF16WS_EXACT_REDUCE_SHAPES = (
    (64, 8),
    (128, 4),
    (256, 2),
    (512, 1),
    (1024, 4),
    (1024, 2),
    (1024, 1),
    (2048, 1),
)

EVEN_LOOP_SPLITK_TAGS = frozenset(
    (
        "a16w16_kbuf2v_sk",
        "a16w16_kbuf2v_bk128_sk",
        "a16w16_quad_mfma32_kbuf1_sk",
    )
)


def _ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def _kid_uses_bf16_workspace(k_inst):
    return getattr(k_inst, "splitk_workspace_dtype", "fp32_t") == "bf16_t"


def _kid_rejects_outdtype(k_inst, out_dtype):
    return _kid_uses_bf16_workspace(k_inst) and out_dtype is not dtypes.bf16


def _flatmm_splitk_pfk(k) -> int:
    """Host-side computation of Traits::prefetch_k_iter for a splitk instance.

    Mirrors opus_flatmm_splitk_traits_gfx950's formula so the host can
    pre-compute the per-split iter budget without a device call. Hardcodes
    LDS=163840 (gfx950), same convention as the traits struct.
    """
    sizeof_da = 2  # bf16
    LOAD_GROUP_M = 64 if k.W_M >= 32 else 32
    LOAD_GROUP_N = 64 if k.W_N >= 32 else 32
    LOAD_GROUP_K = k.W_K * 2
    num_m = k.B_M // LOAD_GROUP_M
    num_n = k.B_N // LOAD_GROUP_N
    num_k = k.B_K // LOAD_GROUP_K
    smem_linear = 64 * 16 // sizeof_da  # WARP_SIZE=64
    smem_sub = smem_linear // LOAD_GROUP_K
    slots = LOAD_GROUP_M // smem_sub
    padding = 16 // sizeof_da if k.W_M >= 32 else 2 * 16 // sizeof_da
    per_glsz = slots * (smem_linear + padding) * sizeof_da
    per_iter = (num_m + num_n) * num_k * per_glsz
    lds_total = 163840
    return max(1, (lds_total // max(k.WG_PER_CU, 1)) // max(per_iter, 1))


def candidate_splitK(M: int, N: int, K: int, batch: int, cu_num: int, k_inst):
    """Pick literal KBatch values in {0, 1..16} worth probing for this (shape, kid).

    Output:
      - Always contains 0 (no-split fallback).
      - Probes every integer in [1, min(16, max_split_k)] so the tuner can
        find the optimal split factor without gaps.
      - Storage is literal KBatch, matching gptoss_bf16_tuned_gemm.csv convention.

    Workspace size cap (added with the >4 GiB reduce-BR fix): each
    candidate splitK value must keep
        split_k * batch * padded_M * padded_N * 4 <= UINT32_MAX
    so the splitk_reduce_kernel's buffer-resource num_records stays in
    range. We compute the same per-slice budget the host reject uses and
    silently drop any split_k that would push workspace past 4 GiB.
    """
    B_K = k_inst.B_K
    total_iters = _ceil_div(K, B_K)
    # gfx1250 cluster/TDM split-K triple-buffers but tolerates any k_steps>=1
    # (the launcher clamps split_k so no split is empty); pfk=1. gfx950 flatmm
    # split-K needs prefetch_k_iter iters per split.
    if k_inst.kernel_tag == "a16w16_cluster_tdm_splitk_ws":
        pfk = 1
    else:
        pfk = _flatmm_splitk_pfk(k_inst)

    candidates = {0}

    max_split_k = min(16, max(1, total_iters // max(pfk, 1)))
    if max_split_k < 1:
        return [0]

    # Workspace 4 GiB cap.
    padded_M = _ceil_div(M, k_inst.B_M) * k_inst.B_M
    padded_N = _ceil_div(N, k_inst.B_N) * k_inst.B_N
    per_slice_bytes = batch * padded_M * padded_N * 4
    UINT32_MAX_BYTES = (1 << 32) - 1
    if per_slice_bytes > 0:
        ws_cap = UINT32_MAX_BYTES // per_slice_bytes
        max_split_k = min(max_split_k, ws_cap)

    if max_split_k < 1:
        return [0]

    # 2-WG/CU co-residency cap: when wg_per_cu == 2 the grid (total workgroups,
    # = ceil(M/B_M) * ceil(N/B_N) * split_k) MUST stay strictly < 512 (= 2 *
    # 256-CU target) so the device holds it in a single 2-WG/CU wave and the
    # per-CU TDM direct-copy request budget (2 * per-operand req < 256) is never
    # oversubscribed across waves. Cap split_k accordingly; if even split_k==1
    # already reaches 512 tiles, this wg==2 kid is not usable for this shape.
    if getattr(k_inst, "wg_per_cu", 1) == 2:
        num_tiles = _ceil_div(M, k_inst.B_M) * _ceil_div(N, k_inst.B_N)
        grid_cap = (512 - 1) // num_tiles if num_tiles > 0 else 0
        if grid_cap < 1:
            return []
        max_split_k = min(max_split_k, grid_cap)

    for kb in range(1, max_split_k + 1):
        candidates.add(kb)

    # kbuf2v_sk and quad_mfma32 splitK families require both iters_full and
    # last_loops even AND >=2.
    if k_inst.kernel_tag in EVEN_LOOP_SPLITK_TAGS:
        total_iters = (K + k_inst.B_K - 1) // k_inst.B_K

        def _ok_sk(sk):
            iters_full = (total_iters + sk - 1) // sk
            last = total_iters - (sk - 1) * iters_full
            return last >= 2 and iters_full % 2 == 0 and last % 2 == 0

        any_valid = any(_ok_sk(s) for s in range(1, max_split_k + 1))
        candidates = {
            sk
            for sk in candidates
            if (sk == 0 and any_valid) or (sk >= 1 and _ok_sk(sk))
        }

    return sorted(candidates)


def kid_rejects_shape(k_inst, M, N, K):
    """Host-side prediction of whether a kid will produce wrong output or
    fail its runtime TORCH_CHECK for this (M, N, K).

    Returns True if the candidate must be rejected BEFORE submission to
    mp_tuner. Submitting rejected kids is not a hard error (the worker
    catches any resulting RuntimeError / max_delta violation and marks it
    invalid), but it clutters stderr with `run gpu func warning: ...` and
    `!!!! us = 0` retry spam, wastes perf iterations, and makes it hard to
    tell "this kid can't handle this shape" from "the profiler happened to
    drop a trace this time".

    Rejection categories:
      (a) Launcher TORCH_CHECK will throw:
          - a16w16 split-barrier: loops >= 2 and loops even.
          - a16w16_flatmm:        loops >= pfk.
          - a16w16_flatmm_splitk: loops >= pfk (splitK auto-clamped).
      (b) Silent-correctness bugs discovered during N=513 diagnosis:
          - a16w16 split-barrier: N % 16 != 0 -> vector store straddles row
            boundary in C (store_if pred only checks vector start).
          - a16w16_flatmm:        N % 16 != 0 -> same root cause.
          - a16w16_flatmm:        K % B_K != 0 -> no K-tail mask, garbage
            accumulates at K tail iter.
          - a16w16_flatmm_splitk: NONE. The reduce-kernel tail path + the
            splitk main kernel's mask_va_tail cover both edge cases, so
            splitk is safe for any (M, N, K).
    """
    # 4 GiB buffer-resource filter. Legacy a16w16 kids build a single
    # AMDGPU buffer-resource per tensor (A/B/C), whose `num_records` field
    # is 32-bit -- any of the three exceeding UINT32_MAX bytes wraps and
    # the kernel OOB-writes. 4g_safe kids carry per-WG-tight BR sizing so
    # they remain valid for arbitrarily large M/N/K; only legacy kids must
    # be filtered. Replaces the prior shape-wide reject at tune/runtime
    # entry (commit 739771b5 reverted).
    if not getattr(k_inst, "is_4g_safe", False):
        _UINT32_MAX_BYTES = (1 << 32) - 1
        # B is bf16 in a16w16 family; C dtype is decided per-call but is
        # at most 4B (fp32 accum). Use 4B upper bound for C so this filter
        # stays dtype-agnostic.
        if (
            M * K * 2 > _UINT32_MAX_BYTES
            or N * K * 2 > _UINT32_MAX_BYTES
            or M * N * 4 > _UINT32_MAX_BYTES
        ):
            return True

    B_K = k_inst.B_K
    loops = _ceil_div(K, B_K)

    if _kid_uses_bf16_workspace(k_inst):
        padded_N = _ceil_div(N, k_inst.B_N) * k_inst.B_N
        if loops < 2 or K % B_K != 0 or padded_N != N:
            return True
        return not any(N == n_exact for n_exact, _ in BF16WS_EXACT_REDUCE_SHAPES)

    if k_inst.kernel_tag in (
        "a16w16",
        "a16w16_kbuf1_large_tile",
        "a16w16_quad_mfma32_kbuf1",
        "a16w16_kbuf2v",
        "a16w16_kbuf2v_bk128",
        "a16w16_kbuf1",
    ):
        # Non-splitK a16w16 family all share the same K-dbuf parity +
        # MFMA layout constraints as a16w16 (10000).
        if loops < 2 or (loops % 2 != 0):
            return True
        if N % 16 != 0:
            return True
        if not k_inst.has_oob:
            if M % k_inst.B_M != 0 or N % k_inst.B_N != 0 or K % k_inst.B_K != 0:
                return True
        return False

    if k_inst.kernel_tag == "a16w16_flatmm":
        if loops < _flatmm_splitk_pfk(k_inst):
            return True
        if N % 16 != 0:
            return True
        if K % B_K != 0:
            return True
        return False

    if k_inst.kernel_tag == "a16w16_flatmm_splitk":
        if loops < _flatmm_splitk_pfk(k_inst):
            return True
        if not k_inst.has_oob:
            if M % k_inst.B_M != 0 or N % k_inst.B_N != 0 or K % k_inst.B_K != 0:
                return True
        # Workspace + reduce buffer-resource size limit.
        padded_M = _ceil_div(M, k_inst.B_M) * k_inst.B_M
        padded_N = _ceil_div(N, k_inst.B_N) * k_inst.B_N
        UINT32_MAX_BYTES = (1 << 32) - 1
        per_slice_bytes = 1 * padded_M * padded_N * 4  # batch=1 in tune path
        if per_slice_bytes > UINT32_MAX_BYTES:
            return True
        return False

    if k_inst.kernel_tag == "a16w16_cluster_tdm_splitk_ws":
        # gfx1250 WMMA kernel: ragged M/N ARE supported -- the main kernel
        # TDM-clamps OOB global reads to the real (M, N) extents (tensor_dim1 =
        # m - tile_row / n - tile_col), padded partials land in the padded fp32
        # workspace, and the reduce kernel only touches m in [0, M) / n in
        # [0, N). So M=49 runs as a padded M=64 tile. Ragged K is handled via
        # the TDM k_extent clamp. K must be even (a16w16 family). Workspace
        # store BR is 32-bit (per-slice cap).
        if K % 2 != 0:
            return True
        padded_M = _ceil_div(M, k_inst.B_M) * k_inst.B_M
        padded_N = _ceil_div(N, k_inst.B_N) * k_inst.B_N
        UINT32_MAX_BYTES = (1 << 32) - 1
        if 1 * padded_M * padded_N * 4 > UINT32_MAX_BYTES:  # batch=1 in tune path
            return True
        return False

    # kbuf2v_sk and quad_mfma32 splitK families require loops_per_split
    # (both full and last) even AND >=2.
    if k_inst.kernel_tag in EVEN_LOOP_SPLITK_TAGS:
        total_iters = _ceil_div(K, B_K)
        max_sk = max(1, min(16, total_iters // max(_flatmm_splitk_pfk(k_inst), 1)))
        for sk in range(1, max_sk + 1):
            iters_full = _ceil_div(total_iters, sk)
            last = total_iters - (sk - 1) * iters_full
            if last >= 2 and iters_full % 2 == 0 and last % 2 == 0:
                return False
        return True

    if k_inst.kernel_tag == "a16w16_persistent":
        if loops < 2 or (loops % 2 != 0):
            return True
        if N % 16 != 0:
            return True
        # In the >4 GiB regime, the persistent pipeline's M-outer + XCD
        # swizzle distribution only pays off when M >= N (work is dealt
        # along the long dimension). For very wide shapes (N >> M, e.g.
        # M=32k N=129k) split-barrier beats persistent by 5-8x. Restrict
        # 4g_safe persistent candidates to M > N to avoid tuning a dead
        # branch.
        if getattr(k_inst, "is_4g_safe", False):
            _U32 = (1 << 32) - 1
            over_4g = (M * K * 2 > _U32) or (N * K * 2 > _U32) or (M * N * 2 > _U32)
            if over_4g and M <= N:
                return True
        num_tiles_m = _ceil_div(M, k_inst.B_M)
        num_tiles_n = _ceil_div(N, k_inst.B_N)
        NUM_CU = 256
        split_m = max(1, _ceil_div(NUM_CU, num_tiles_n))
        while split_m < num_tiles_m and num_tiles_m % split_m != 0:
            split_m += 1
        if split_m > num_tiles_m:
            return True
        if num_tiles_m % split_m != 0:
            return True
        if not k_inst.has_oob:
            if M % k_inst.B_M != 0 or N % k_inst.B_N != 0 or K % k_inst.B_K != 0:
                return True
        return False

    if k_inst.kernel_tag == "a16w16_mono_tile":
        # mono_tile is divisible-only (has_oob=False is intrinsic; the
        # launcher asserts M%B_M==N%B_N==K%B_K==0) and has no splitk /
        # K-tail mask path. Reject any non-aligned shape up front.
        if M % k_inst.B_M != 0 or N % k_inst.B_N != 0 or K % k_inst.B_K != 0:
            return True
        # N%16 vector-store guard (same root cause as a16w16 family);
        # already implied by N%B_N==0 with B_N % 64 == 0, but keep the
        # explicit check for symmetry.
        if N % 16 != 0:
            return True
        return False

    return False


def kid_rejects_bias(k_inst, bias):
    """Reject candidates that cannot consume a non-empty bias.

    Bias support (mirrors opus_kid_supports_bias in opus_gemm.cu):
      * gfx950 a16w16 split-barrier (kernel_tag=="a16w16", kid 4..9 + mirrors)
      * gfx950 a16w16_flatmm_splitk (kid 200..299 + nooob mirror)
    Rejected:
      * persistent / flatmm-non-splitk (HAS_BIAS=false hardcoded in pipelines)
      * gfx942 SB kid 10000 (HAS_BIAS path removed 2026-06-02; was dead code)
      * gfx942 splitk family (kid 10200+): silently ignored bias
      * gfx942 non-splitK siblings (10002/03/11): no bias path
    """
    if not bias:
        return False
    # gfx1250 cluster_tdm_splitk_ws folds bias in the reduce kernel (bias-aware).
    if k_inst.kernel_tag == "a16w16_cluster_tdm_splitk_ws":
        return False
    if k_inst.kernel_tag not in ("a16w16", "a16w16_flatmm_splitk"):
        return True
    # arch_prefix distinguishes gfx942 (no bias) from gfx950 (has bias)
    # for the shared "a16w16" / "a16w16_flatmm_splitk" tags.
    return getattr(k_inst, "arch_prefix", "") == "gfx942"


def candidate_kids_for_shape(M, N, K, bias, cu_num):
    """Decide which kids should be tuned for shape (M, N, K, bias).

    Rules (all evaluated on a single instance B_M=B_N=128 occupancy proxy):

    1) Structural fallback: if K is not friendly to non-splitk launchers
       (need ``K % 64 == 0`` and ``(K // 64) % 2 == 0`` -- the
       split-barrier prefetch double-buffer + persistent ceil-K
       constraint), then non-splitk kids would all reject this shape at
       runtime. Return SPLITK_KIDS to guarantee a viable candidate.

    2) Otherwise compare ``ceil(M/128) * ceil(N/128)`` against ``2 *
       cu_num``. If smaller, the problem can't fill the device twice
       even with the smallest non-splitk tile; only splitk (which slices
       K to widen the work-distribution dimension) is meaningful.

    3) Otherwise both splitk and non-splitk kids are viable; tune both.

    4) Bias post-filter: if bias is True, drop kids that are not in
       BIAS_AWARE_KIDS. If that leaves the candidate set empty (corner
       case: SPLITK_KIDS minus bias-aware ? SPLITK_KIDS is fully
       bias-aware, so this should not fire in practice), fall back to
       SPLITK_KIDS again.

    Parameters
    ----------
    M, N, K : int
    bias : bool
    cu_num : int
        Number of compute units on the device (e.g. 256 for MI350).

    Returns
    -------
    frozenset[int]
        Subset of ``kernels_list.keys()``; never empty.
    """
    M = int(M)
    N = int(N)
    K = int(K)
    bias = bool(bias)
    cu_num = int(cu_num)

    # Step 1: structural tile-align fallback for non-splitk pipelines.
    # K % 64 == 0 and (K // 64) % 2 == 0 required for non-splitk pipeline's
    # prefetch double-buffer + persistent ceil-K constraint.
    non_splitk_tile_align_ok = (K % 64 == 0) and ((K // 64) % 2 == 0)
    if not non_splitk_tile_align_ok:
        cands = SPLITK_KIDS
    else:
        # Step 2/3: tune both splitk and non-splitk kids.
        cands = SPLITK_KIDS | NON_SPLITK_KIDS

    # Step 4: bias post-filter.
    if bias:
        narrowed = cands & BIAS_AWARE_KIDS
        if not narrowed:
            # SPLITK_KIDS ? BIAS_AWARE_KIDS, so this is a safety net.
            cands = SPLITK_KIDS
        else:
            cands = narrowed

    # Step 5: arch post-filter.
    try:
        from aiter.jit.utils.chip_info import get_gfx_runtime
        from opus_gemm_common import kernels_list as _klist

        _run_arch = get_gfx_runtime().lower()
        cands = frozenset(
            kid
            for kid in cands
            if (getattr(_klist.get(kid), "arch_prefix", "") or "gfx950").lower()
            == _run_arch
        )
    except Exception:
        pass  # unknown arch -> keep legacy multi-arch behaviour

    # Step 6: drop known-bad kids permanently.
    cands = cands - _OPUS_PERMA_BAD_KIDS
    return cands


# Kids we never want tuner to probe.
_OPUS_PERMA_BAD_KIDS = frozenset()


def _ensure_kids_compiled(candidate_kids):
    """Make sure every kid in ``candidate_kids`` is compiled into the
    current module_deepgemm_opus.so.

    Reads the subset-compile sidecar at ``_opus_sidecar_path()`` (lives in
    ``$JIT_BUILD/`` so it survives clear_build). If any kid in
    ``candidate_kids`` (or in ``HEURISTIC_DEFAULT_KIDS``) is missing from
    the sidecar, this function:

    1. Atomically expands the sidecar to the union of the existing
       contents, the new candidates, and the heuristic defaults.
    2. Clears the aiter.jit.core in-process module caches and removes
       the on-disk .so so the next ``@compile_ops("module_deepgemm_opus")``
       call rebuilds from scratch (the codegen step re-reads the sidecar).
    3. **Synchronously triggers the rebuild here** by calling
       build_module() directly so subsequent ``mp_tuner`` spawn-ed
       children inherit a .so on disk that already contains every
       required kid. Without this synchronous step, children would race
       against the parent's lazy build and the first to dlopen() would
       get the stale subset .so and fail with
       ``AITER_CHECK: Kernel id X not found in a16w16 tune lookup table``.

    Concurrency model
    -----------------
    Two race vectors are explicitly defended against:

    A. **Concurrent GemmTuner / parent processes** (multi-GPU multi-script):
       sidecar read + expand + write + build is wrapped in a ``FileBaton``
       (`$JIT_BUILD/lock_ensure_kids_opus`). One parent runs the full
       expand+build; the rest spin on the baton, then re-read the sidecar
       (it may already contain what they need, in which case they skip).

    B. **mp_tuner spawn-ed children inheriting `AITER_REBUILD=1`**:
       if the user launched the tuner with `AITER_REBUILD=1` in env
       (the usual workflow after editing source), every spawn child
       re-reads the env at import time and would `clear_build()` +
       rebuild on its first `compile_ops("module_deepgemm_opus")` call.
       The first child wins the FileBaton, deletes the .so we just
       baked, and the rest then race against an in-flight build,
       producing intermittent `FileNotFoundError` / partial ELF errors.
       To shut this off we **also clear `os.environ["AITER_REBUILD"]`**
       once our synchronous build succeeds, so every spawn child
       inherits a clean env (read: `AITER_REBUILD=0`) and goes straight
       to `dlopen()` of the .so we just produced. The original env is
       restored on the parent process only after this call returns;
       the parent itself does not need the rebuild flag past this
       point because we already added ``module_deepgemm_opus`` to
       ``rebuilded_list``.

    Returns
    -------
    bool
        True if a rebuild was triggered (sidecar grew), False if every
        required kid was already compiled.
    """
    from aiter.jit import core as _jit_core
    from aiter.jit.utils.file_baton import FileBaton
    from opus_gemm_common import heuristic_kids_for_arch

    candidate_kids = frozenset(int(k) for k in candidate_kids)
    # Restrict the heuristic-default kid set to the running GPU's arch.
    try:
        from aiter.jit.utils.chip_info import get_gfx_runtime

        _run_arch = get_gfx_runtime().lower()
        _heuristic = heuristic_kids_for_arch({_run_arch})
    except Exception:
        _heuristic = HEURISTIC_DEFAULT_KIDS  # unknown -> multi-arch fallback
    required = candidate_kids | _heuristic

    def _read_sidecar(path):
        if not os.path.exists(path):
            return set()
        try:
            with open(path) as f:
                return set(json.load(f))
        except (OSError, ValueError):
            return set()

    sidecar = _opus_sidecar_path()

    # Fast path: no lock needed if we are already a strict subset of whatever sidecar happens to be
    # on disk.
    if required <= _read_sidecar(sidecar):
        return False

    os.makedirs(_jit_core.bd_dir, exist_ok=True)
    lock_path = f"{_jit_core.bd_dir}/lock_ensure_kids_opus"
    baton = FileBaton(lock_path)

    if not baton.try_acquire():
        # A peer parent (multi-GPU / multi-process tune harness) is already extending the sidecar +
        # rebuilding.
        baton.wait()
        if required <= _read_sidecar(sidecar):
            return False
        # Peer's expand didn't cover us (rare: peer's `required` set was
        # disjoint from ours). Re-enter to extend further.
        return _ensure_kids_compiled(candidate_kids)

    try:
        compiled = _read_sidecar(sidecar)
        missing = required - compiled
        if not missing:
            # Another writer beat us inside the critical section.
            return False

        sys.stderr.write(
            f"[opus _ensure_kids_compiled] need to add {len(missing)} kids "
            f"to sidecar (existing={len(compiled)}, target={len(compiled | required)})\n"
        )

        # Persist the expanded set.
        new_set = sorted(compiled | required)
        os.makedirs(os.path.dirname(sidecar), exist_ok=True)
        with open(sidecar, "w") as f:
            json.dump(new_set, f)
        sys.stderr.write(
            f"[opus _ensure_kids_compiled] wrote sidecar at {sidecar} with "
            f"{len(new_set)} kids; triggering build...\n"
        )

        # Force a JIT rebuild scoped to JUST this build call.
        _prev_rebuild = _jit_core.AITER_REBUILD
        _prev_rebuild_env = os.environ.get("AITER_REBUILD")
        _jit_core.AITER_REBUILD = 1
        # Set env to "1" only for the synchronous build below; we reset
        # to "0" once it succeeds so spawn-ed children don't redo it.
        os.environ["AITER_REBUILD"] = "1"
        _jit_core.get_module.cache_clear()
        # __mds is name-mangled inside the module; access via private name.
        _mds = getattr(_jit_core, "_core__mds", None) or getattr(
            _jit_core, "__mds", None
        )
        if _mds is None:
            _mds = _jit_core.__dict__.get("__mds")
        if isinstance(_mds, dict):
            _mds.clear()
        # Reset rebuilded_list (used by compile_ops to track "we already rebuilt this once in this process").
        _jit_core.rebuilded_list = ["module_aiter_enum"]
        # Reset the in-process torch JIT extension versioner so the second synchronous rebuild here
        # (sidecar grew between shape N and N+1...
        try:
            import sys as _sys

            for _modname in ("cpp_extension", "aiter.jit.utils.cpp_extension"):
                _mod = _sys.modules.get(_modname)
                if _mod is None:
                    continue
                _jev = getattr(_mod, "JIT_EXTENSION_VERSIONER", None)
                if _jev is None:
                    continue
                _entries = getattr(_jev, "entries", None)
                if isinstance(_entries, dict):
                    _entries.pop("module_deepgemm_opus", None)
        except Exception:
            pass

        # Synchronously drive the rebuild in this (parent) process so that mp_tuner's spawn-ed children
        # see a fully-baked .so on disk and...
        _build_exc = None
        try:
            d_args = _jit_core.get_args_of_build("module_deepgemm_opus")
            _jit_core.build_module(
                md_name="module_deepgemm_opus",
                srcs=d_args["srcs"],
                flags_extra_cc=d_args["flags_extra_cc"],
                flags_extra_hip=d_args["flags_extra_hip"],
                blob_gen_cmd=d_args["blob_gen_cmd"],
                extra_include=d_args["extra_include"],
                extra_ldflags=d_args["extra_ldflags"],
                verbose=d_args.get("verbose", False),
                is_python_module=d_args.get("is_python_module", True),
                is_standalone=d_args.get("is_standalone", False),
                torch_exclude=d_args.get("torch_exclude", False),
                third_party=d_args.get("third_party", []),
                hipify=d_args.get("hipify", False),
                flags_extra_hip_per_source=d_args.get("flags_extra_hip_per_source", {}),
            )
            if "module_deepgemm_opus" not in _jit_core.rebuilded_list:
                _jit_core.rebuilded_list.append("module_deepgemm_opus")
        except Exception as exc:
            _build_exc = exc
            import traceback

            sys.stderr.write(
                f"[opus _ensure_kids_compiled] synchronous build_module FAILED:\n"
                f"  {type(_build_exc).__name__}: {_build_exc}\n"
            )
            traceback.print_exc()
            # Scrub the half-baked .so so workers don't dlopen a stale
            # subset .so and silently mis-dispatch kids.
            try:
                _so = os.path.join(
                    _jit_core.get_user_jit_dir(), "module_deepgemm_opus.so"
                )
                if os.path.exists(_so):
                    os.remove(_so)
            except Exception:
                pass
        finally:
            # Restore in-process flag for the parent (mp_tuner children
            # spawn from os.environ, not from this in-process value).
            _jit_core.AITER_REBUILD = _prev_rebuild
            # For children: if build succeeded, force AITER_REBUILD=0 in env so spawned workers go straight
            # to dlopen(); if build failed, res...
            if _build_exc is None:
                os.environ["AITER_REBUILD"] = "0"
            else:
                if _prev_rebuild_env is None:
                    os.environ.pop("AITER_REBUILD", None)
                else:
                    os.environ["AITER_REBUILD"] = _prev_rebuild_env

        if _build_exc is not None:
            raise RuntimeError(
                "opus_gemm subset-compile rebuild failed; see hipcc / "
                "codegen error in stderr above. The expanded sidecar "
                "has already been written (rerun will pick up where "
                "this left off)."
            ) from _build_exc

        return True
    finally:
        baton.release()


# Backwards-compat aliases for the old leading-underscore names.
_kid_rejects_shape = kid_rejects_shape
_kid_rejects_bias = kid_rejects_bias


# Debug-only tuner output: /tmp by default so this script can never accidentally pollute the
# global tuned CSV tree.
OPUS_DEBUG_TUNED_CSV = os.getenv(
    "AITER_OPUS_DEBUG_TUNED_CSV",
    "/tmp/opus_debug_tuned.csv",
)


# Silence per-task `avg: X us/iter with hipgraph` and `no valida data after post process!` log
# lines from aiter.test_common by de...
_AITER_VERBOSE = bool(int(os.environ.get("AITER_VERBOSE", "0")))


# Merge every a16w16-family kid into one tuner search space: * split-barrier a16w16: 4..9 legacy
# cpol = (0, 17) (traits default) ...
a16w16_all_kernels = {
    **a16w16_kernels_list,
    **a16w16_kernels_list_nooob,
    **a16w16_kernels_list_cpol,
    **a16w16_kernels_list_cpol_nooob,
    **a16w16_flatmm_kernels_list,
    **a16w16_flatmm_splitk_kernels_list,
    **a16w16_flatmm_splitk_kernels_list_nooob,
    **a16w16_persistent_kernels_list,
    **a16w16_persistent_kernels_list_cpol,
    **a16w16_persistent_kernels_list_nooob,
    **a16w16_persistent_kernels_list_cpol_nooob,
    **gfx942_nosplit_kernels_list,
    **gfx942_splitk_kernels_list,
    **gfx1250_kernels_list,
}

# Arch-filter the kid enumeration so the tuner only dispatches kids whose pipeline body has a
# non-empty implementation on the run...
try:
    from aiter.jit.utils.chip_info import get_gfx_runtime

    _run_arch = get_gfx_runtime().lower()
    a16w16_kernel_ids = sorted(
        kid
        for kid, k in a16w16_all_kernels.items()
        if (getattr(k, "arch_prefix", "") or "gfx950").lower() == _run_arch
    )
except Exception:
    # rocminfo unavailable -> fall back to enumerating everything so the
    # legacy multi-arch behaviour is preserved on build-only hosts.
    a16w16_kernel_ids = sorted(a16w16_all_kernels.keys())


# -- dtype handling ---------------------------------------------------------- CSV convention
# (matches gptoss_bf16_*_gemm.csv sch...
_DTYPE_SHORT_TO_TORCH = {
    "bf16": dtypes.bf16,
    "fp32": dtypes.fp32,
}

_DTYPE_TORCH_TO_CSV_STR = {
    dtypes.bf16: "torch.bfloat16",
    dtypes.fp32: "torch.float32",
}

_DTYPE_CSV_STR_TO_TORCH = {v: k for k, v in _DTYPE_TORCH_TO_CSV_STR.items()}

_DTYPE_TORCH_TO_BPE = {
    dtypes.bf16: 2,
    dtypes.fp32: 4,
}


def _dtype_csv_str_to_torch(s):
    """Map a CSV cell (e.g. 'torch.bfloat16') to a torch.dtype.

    Accepts both the canonical CSV form and the CLI short form; falls back
    to bf16 with a warning for anything unrecognized so a single weird row
    can't take down a whole tuning run.
    """
    if s is None:
        return None
    if isinstance(s, torch.dtype):
        return s
    s = str(s).strip()
    if s in _DTYPE_CSV_STR_TO_TORCH:
        return _DTYPE_CSV_STR_TO_TORCH[s]
    if s in _DTYPE_SHORT_TO_TORCH:
        return _DTYPE_SHORT_TO_TORCH[s]
    logger.warning(
        f"OpusGemmA16W16Tuner: unrecognized dtype string {s!r}, falling back to bf16"
    )
    return dtypes.bf16


def _dtype_torch_to_csv_str(t):
    return _DTYPE_TORCH_TO_CSV_STR.get(t, str(t))


def generate_data(
    batch,
    m,
    n,
    k,
    seed,
    dtype=dtypes.bf16,
    outdtype=dtypes.bf16,
    bias=False,
    device="cuda",
):
    # gen_data is the earliest subprocess-side hook mp_tuner.work_group invokes
    # (mp_tuner.py:139-143), before worker() is ever called.
    _install_opus_perftest_once()

    # mp_tuner pickles task tuples with `gen_args` containing primitive types only; torch.dtype
    # objects survive pickling fine, but if...
    if isinstance(dtype, str):
        dtype = _dtype_csv_str_to_torch(dtype)
    if isinstance(outdtype, str):
        outdtype = _dtype_csv_str_to_torch(outdtype)

    torch.manual_seed(seed)
    # Use the exact (M, N, K) from the tune request.
    XQ = torch.randn((batch, m, k), dtype=dtype, device=device)
    WQ = torch.randn((batch, n, k), dtype=dtype, device=device)
    Y = torch.empty((batch, m, n), dtype=outdtype, device=device)
    # bias dtype matches Y (match_d_out convention). Opus bias is per-output-
    # feature (per-N): the splitk reduce folds a [N] / [batch, N] vector
    # (see splitk_reduce_*.cuh "per-output-feature bias"), NOT per-row (M).
    if bias:
        bias_t = torch.randn((batch, n), dtype=outdtype, device=device)
    else:
        bias_t = None
    return XQ, WQ, Y, bias_t


MAX_DELTA_SCALE = 0.1


def opus_gemm_ref(XQ, WQ, bias=None, out_dtype=None):
    """Reference matmul (+ optional per-output-feature bias).

    out_dtype must match the tuner's Y.dtype so the post-run checkAllclose
    in mp_tuner.worker compares same-dtype tensors (mismatched dtype raises
    'BFloat16 did not match Float' inside checkAllclose). When omitted (e.g.
    legacy callers) we fall back to XQ.dtype, preserving the original bf16
    -> bf16 -> bf16 path.

    bias is summed in fp32 to match the kernel-side fp32 acc + cast order;
    accepted shapes are [N] (broadcast across batch) or [batch, N]. Opus folds
    a per-output-feature (per-N) bias, matching splitk_reduce_*.cuh.
    """
    if out_dtype is None:
        out_dtype = XQ.dtype
    acc = torch.bmm(XQ.float(), WQ.float().transpose(-1, -2))
    if bias is not None:
        # Per-N broadcast: unsqueeze the M axis so bias [..., N] -> [..., 1, N]
        # and add across M (acc is [batch, M, N]).
        acc = acc + bias.float().unsqueeze(-2)
    return acc.to(out_dtype)


def run_opus_gemm(XQ, WQ, Y, bias, kernelId, splitK):
    """Eager-path tuner func: runs the kernel AND an on-the-fly max_delta check.

    The check raises RuntimeError when the output is numerically off; mp_tuner's
    worker catches it and marks the candidate invalid. Used when --no-graph is
    passed (i.e. graph mode disabled) so the per-iter check is safe (no CUDA
    graph capture).
    """
    _quiet_aiter_logger_once()
    _opus_gemm_a16w16_tune(XQ, WQ, Y, bias, kernelId, splitK)
    ref = opus_gemm_ref(XQ, WQ, bias, Y.dtype)
    max_delta = (Y.float() - ref.float()).abs().max().item()
    max_ref = ref.float().abs().max().item()
    bound = max(max_ref * MAX_DELTA_SCALE, 1.0)
    if max_delta > bound:
        raise RuntimeError(
            f"maxDelta {max_delta:.1f} exceeds bound {bound:.1f} "
            f"(max|ref|={max_ref:.1f}, scale={MAX_DELTA_SCALE})"
        )
    return Y


_subproc_quiet_initialized = False


def _quiet_aiter_logger_once():
    """Silence per-task log spam in the mp_tuner subprocess.

    Each subprocess hits this on its first bench call; spawn workers don't
    inherit the parent logger level, so we set it per-process. Effects:
      * Suppresses `[aiter] avg: X us/iter with hipgraph` (test_common.py:116)
      * Suppresses `[aiter] no valida data after post process!` (test_common.py:383)
    Re-enable with `-v` / AITER_VERBOSE=1.
    """
    global _subproc_quiet_initialized
    if _subproc_quiet_initialized:
        return
    _subproc_quiet_initialized = True
    if _AITER_VERBOSE:
        return
    try:
        logger.setLevel(logging.WARNING)
    except Exception:
        pass


_bench_max_delta_checked = set()  # module-level per-subprocess cache


def run_opus_gemm_bench(XQ, WQ, Y, bias, kernelId, splitK):
    """Tuner bench func with capture-safe stream sync + per-task max_delta
    safety check.

    Stream sync rationale
    ---------------------
    When our custom run_perftest replacement (_opus_run_perftest below) uses
    torch.cuda.Event to time the graph replay, the end-event record needs
    the kernel in flight. The sync inside the bench func itself is for the
    WARMUP phase (outside capture), so that max_delta check and torch.bmm
    reference see a stable Y before validating.

    The sync is gated on is_current_stream_capturing() because
    cudaStreamSynchronize during CUDA graph capture invalidates the graph
    (HIP returns hipErrorStreamCaptureInvalidated).

    Correctness gate
    ----------------
    mp_tuner.worker's post-run checkAllclose(ref, Y, rtol, atol) gates on
    *fraction* of cells above tolerance, not the max single-cell absolute
    delta. We add a per-task max_delta check:
      * Runs once per (XQ, WQ, Y, kid, splitK) tuple per subprocess.
      * Skipped inside CUDA graph capture (.item() forbidden there).
      * Raises RuntimeError on violation; mp_tuner.worker marks the
        candidate us=-1, err_ratio=1.0.
    """
    _quiet_aiter_logger_once()
    _opus_gemm_a16w16_tune(XQ, WQ, Y, bias, kernelId, splitK)

    capturing = torch.cuda.is_current_stream_capturing()

    if not capturing:
        # Correctness gate (per-task, outside-capture only).
        task_key = (
            id(XQ),
            id(WQ),
            id(Y),
            id(bias) if bias is not None else 0,
            int(kernelId),
            int(splitK),
        )
        if task_key not in _bench_max_delta_checked:
            _bench_max_delta_checked.add(task_key)
            ref = opus_gemm_ref(XQ, WQ, bias, Y.dtype)
            max_delta = (Y.float() - ref.float()).abs().max().item()
            max_ref = ref.float().abs().max().item()
            bound = max(max_ref * MAX_DELTA_SCALE, 1.0)
            if max_delta > bound:
                raise RuntimeError(
                    f"maxDelta {max_delta:.1f} > bound {bound:.1f} "
                    f"(max|ref|={max_ref:.1f}, scale={MAX_DELTA_SCALE}) "
                    f"for kid={kernelId} splitK={splitK} bias={bias is not None}"
                )

        # Capture-safe sync: guarantees the warmup kernel has completed before we read Y for the
        # max_delta check (above) or before the ou...
        torch.cuda.current_stream().synchronize()
    return Y


# Custom run_perftest for opus a16w16 tuner: CUDA graph + cuda.Event timing.


def _opus_run_perftest(
    func,
    *args,
    num_iters=101,
    num_warmup=2,
    testGraph=False,
    num_rotate_args=0,
    needTrace=False,
    **kwargs,
):
    """Drop-in replacement for aiter.test_common.run_perftest with a CUDA-
    graph + cuda.Event timing path that works on ROCm / gfx950 where
    torch.profiler returns empty traces for multi-kernel graph replays.

    Return value matches the stock run_perftest: `(data, avg_us_per_iter)`.

    * Warmup: run `func` num_warmup+1 times in eager mode on the current
      stream. Last warmup doubles as a correctness witness (we want the
      stock mp_tuner.worker's post-run checkAllclose to see a valid Y).
    * Capture: wrap `num_iters` invocations of `func` into a CUDAGraph on
      a dedicated side stream (torch's recommended pattern).
    * Measure: record a start event, replay the graph once, record an end
      event, synchronize, compute `elapsed_time(start, end) / num_iters`.
    * Convert ms -> us (cuda.Event.elapsed_time returns ms).
    """
    # Inside the subprocess the bench func is `func` directly (kwargs may be empty).

    # Warmup on the main stream. Keeps the first measurement stable and
    # triggers JIT / kernel loads before capture.
    for _ in range(max(1, num_warmup)):
        data = func(*args, **kwargs)
    torch.cuda.synchronize()

    # Capture the timing loop.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(side):
        # A second warmup is recommended on the side stream to "prime" the allocator and any per-stream
        # state so capture records a stable...
        data = func(*args, **kwargs)
        side.synchronize()
        with torch.cuda.graph(graph, stream=side):
            for _ in range(num_iters):
                data = func(*args, **kwargs)
    torch.cuda.current_stream().wait_stream(side)

    # Measure replay with cuda.Event. hipEventElapsedTime returns ms.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    elapsed_ms = start.elapsed_time(end)
    avg_us = (elapsed_ms * 1000.0) / max(num_iters, 1)

    return data, avg_us


_opus_perftest_patch_installed = False


def _install_opus_perftest_once():
    """Monkey-patch aiter.test_common.run_perftest with our CUDA-graph +
    cuda.Event timing replacement. Runs exactly once per subprocess, the
    first time run_opus_gemm_bench is invoked.

    This is NOT a modification of aiter.test_common or mp_tuner source
    code -- it's a per-subprocess runtime override. The parent process
    and other tuners in the same conda env are unaffected.
    """
    global _opus_perftest_patch_installed
    if _opus_perftest_patch_installed:
        return
    _opus_perftest_patch_installed = True
    try:
        import aiter.test_common as _tc

        _tc.run_perftest = _opus_run_perftest
    except Exception as e:
        logger.warning(
            f"OpusGemmA16W16Tuner: failed to install custom run_perftest "
            f"({e}); falling back to stock version (expect empty-trace us=0 "
            f"on splitk kids)."
        )


class OpusGemmA16W16Tuner(GemmCommonTuner):
    ARG_DEFAULTS = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": OPUS_DEBUG_TUNED_CSV,
        "untune_file": "aiter/configs/model_configs/gptoss_bf16_untuned_gemm.csv",
        # Tighter than GemmCommonTuner default (0.05).
        "errRatio": 1.0,
        "batch": 100,
        "profile_file": "",
    }

    # 17-column schema matching aiter/configs/model_configs/gptoss_bf16_tuned_gemm.csv exactly.
    OUT_COLUMNS = [
        "cu_num",
        "M",
        "N",
        "K",
        "bias",
        "dtype",
        "outdtype",
        "scaleAB",
        "bpreshuffle",
        "libtype",
        "solidx",
        "splitK",
        "us",
        "kernelName",
        "err_ratio",
        "tflops",
        "bw",
    ]

    def getKernelName(self, kernelId):
        k = a16w16_all_kernels.get(kernelId)
        return k.name if k else None

    def _setup_specific_arguments(self):
        # Release `-k` short-form from base_tuner's `-k/--splitK` so we can repurpose it for the K dim
        # below, giving a symmetric `-m/-n/-...
        def _drop_action(pred):
            for action in list(self.parser._actions):
                if pred(action):
                    self.parser._actions.remove(action)
                    for s in action.option_strings:
                        self.parser._option_string_actions.pop(s, None)
                    for g in self.parser._action_groups:
                        if action in g._group_actions:
                            g._group_actions.remove(action)
                    return action
            return None

        _drop_action(
            lambda a: "-k" in a.option_strings or "--splitK" in a.option_strings
        )
        self.parser.add_argument(
            "-s",
            "--splitK",
            action="store_true",
            required=False,
            dest="splitK",
            help="Enable splitK search path (opus tuner always searches splitK "
            "for splitk kids regardless of this flag; kept for compatibility).",
        )

        # Retained for backwards compat.
        self.parser.add_argument(
            "--no-graph",
            dest="graph",
            action="store_false",
            default=True,
            help="(Deprecated, no-op) CUDA graph mode is always off for the "
            "opus tuner. Retained for argparse compat with upstream GemmCommonTuner.",
        )

        # Force re-tune & in-place overwrite of every shape listed in -i, regardless of whether the
        # tuned CSV already has a winner for th...
        self.parser.add_argument(
            "--retune",
            action="store_true",
            default=False,
            dest="retune",
            help="Re-tune every shape in -i untune_file even when a tuned "
            "winner already exists, and overwrite the old row in-place "
            "ONLY when the new measurement is strictly faster (us-wise) "
            "than the cached one. Ties or regressions keep the old row. "
            "Mutually exclusive with --all (which uses append+resort).",
        )

        # Direct M/N/K entry -- skip the untuned csv input.
        self.parser.add_argument(
            "-m",
            "--M",
            dest="shape_m",
            type=int,
            default=None,
            help="Tune a single M dim (pairs with -n/-k). Bypasses -i csv.",
        )
        self.parser.add_argument(
            "-n",
            "--N",
            dest="shape_n",
            type=int,
            default=None,
            help="Tune a single N dim (pairs with -m/-k). Bypasses -i csv.",
        )
        self.parser.add_argument(
            "-k",
            "--K",
            dest="shape_k",
            type=int,
            default=None,
            help="Tune a single K dim (pairs with -m/-n). Bypasses -i csv.",
        )
        self.parser.add_argument(
            "-b",
            "--BATCH",
            dest="shape_batch",
            type=int,
            default=1,
            help="Batch dim for single-shape CLI tuning (default 1).",
        )

        # Default dtype / outdtype (CSV columns override per-row when present).
        self.parser.add_argument(
            "--dtype",
            choices=sorted(_DTYPE_SHORT_TO_TORCH.keys()),
            default="bf16",
            help="Default input A/B dtype (bf16 or fp32). Per-row 'dtype' "
            "column in the untuned CSV overrides this.",
        )
        self.parser.add_argument(
            "--outdtype",
            choices=sorted(_DTYPE_SHORT_TO_TORCH.keys()),
            default="bf16",
            help="Default output Y dtype (bf16 or fp32). Per-row 'outdtype' "
            "column in the untuned CSV overrides this.",
        )

        # --bias is the per-shape bias toggle.
        self.parser.add_argument(
            "--bias",
            action="store_true",
            default=False,
            dest="bias",
            help="Default 'bias' value when single-shape tuning (-m/-n/-k) "
            "or when the untuned CSV lacks a 'bias' column. Tuner emits "
            "bias=True / bias=False as a separate dedup key so the same "
            "(M,N,K) can be tuned with and without bias independently.",
        )

        # Debug knob: restrict the kid sweep to a specific set.
        self.parser.add_argument(
            "--kid",
            dest="kid",
            type=str,
            default=None,
            help="Comma-separated kid id(s) to restrict the sweep to "
            "(e.g. --kid 10201 or --kid 10200,10201). Kids must still pass "
            "the host-side shape / bias / outdtype filters; rejected kids "
            "are dropped with a log line. Compile sidecar is auto-expanded "
            "to cover the requested kids if needed.",
        )

    def pre_process(self, args):
        """Override to:

        * Support CLI-based single-shape tuning (-m / -n / -k).
        * Treat (bias, dtype, outdtype) as part of the dedup / update key.
          self.keys is now 7-wide; same (M,N,K,outdtype) tuned with and
          without bias generates two distinct rows in the tuned CSV. We
          still need to populate missing columns from the CLI defaults
          (legacy CSVs without dtype/bias columns).
        * Honor --retune: skip the "already tuned" dedup mask so every
          shape in untunedf is re-measured, then rely on the existing
          update_tunedf path (concat=False) to overwrite the old row
          in-place when result_to_csv writes back. The actual replace-
          if-better gate lives in our update_tunedf override below.
        """
        if args.retune and args.all:
            self.parser.error(
                "--retune and --all are mutually exclusive. Use --retune "
                "for in-place overwrite of -i shapes (one row per shape "
                "stays in tuned CSV); use --all for append+resort retune."
            )
        # Stash for update_tunedf to consult: when set, our override only replaces an existing row when
        # the freshly tuned `us` is strictl...
        self._retune_only_if_better = bool(args.retune)
        # update_tunedf only receives (df_old, df_updates); cache verbose here so the override can
        # decide whether to dump the per-row rej...
        self._args_verbose_cached = bool(getattr(args, "verbose", False))
        cli_dtype_str = _dtype_torch_to_csv_str(_DTYPE_SHORT_TO_TORCH[args.dtype])
        cli_outdtype_str = _dtype_torch_to_csv_str(_DTYPE_SHORT_TO_TORCH[args.outdtype])
        cli_bias = bool(args.bias)

        if (
            args.shape_m is not None
            and args.shape_n is not None
            and args.shape_k is not None
        ):
            # Build a one-row untunedf directly from CLI args.
            row = {
                "cu_num": self.get_cu_num(),
                "M": int(args.shape_m),
                "N": int(args.shape_n),
                "K": int(args.shape_k),
                "bias": cli_bias,
                "dtype": cli_dtype_str,
                "outdtype": cli_outdtype_str,
            }
            self.untunedf = pd.DataFrame([row])
            self.tunedf = self.get_tuned_gemm_list(args.tune_file)
            # Match the original code path: only assign 'batch' if the column already exists.
            if "batch" in self.untunedf.columns:
                self.untunedf["batch"] = int(args.shape_batch)
            logger.info(
                f"OpusGemmA16W16Tuner: single-shape CLI tune "
                f"(M={args.shape_m}, N={args.shape_n}, K={args.shape_k}, "
                f"batch={args.shape_batch}, bias={cli_bias}, "
                f"dtype={args.dtype}, outdtype={args.outdtype})"
            )
            return

        # CSV-driven path.
        if args.all:
            self.get_retune_gemm_list(args)
        else:
            self.untunedf = self.get_untuned_gemm_list(args.untune_file)
            self.untunedf["cu_num"] = self.get_cu_num()
            # Fill missing bias / dtype / outdtype columns with the CLI defaults so legacy CSVs (4-col key
            # without these) still tune.
            if "bias" not in self.untunedf.columns:
                self.untunedf["bias"] = cli_bias
            else:
                # CSV cells are typically the strings "True" / "False"; the raw read-back can leave them as bool
                # already.
                self.untunedf["bias"] = (
                    self.untunedf["bias"]
                    .fillna(cli_bias)
                    .map(
                        lambda v: (
                            bool(v)
                            if isinstance(v, bool)
                            else str(v).strip().lower() == "true"
                        )
                    )
                )
            if "dtype" not in self.untunedf.columns:
                self.untunedf["dtype"] = cli_dtype_str
            else:
                self.untunedf["dtype"] = self.untunedf["dtype"].fillna(cli_dtype_str)
            if "outdtype" not in self.untunedf.columns:
                self.untunedf["outdtype"] = cli_outdtype_str
            else:
                self.untunedf["outdtype"] = self.untunedf["outdtype"].fillna(
                    cli_outdtype_str
                )

            self.untunedf = self.untunedf[list(self.keys)]
            self.tunedf = self.get_tuned_gemm_list(args.tune_file)

            if len(self.tunedf) != 0:
                # Dedup against the full 7-key.
                key_cols = list(self.keys)
                mask = (
                    self.untunedf[key_cols]
                    .apply(tuple, axis=1)
                    .isin(self.tunedf[key_cols].apply(tuple, axis=1))
                )
                if args.retune:
                    # --retune: keep all -i shapes (including those already tuned).
                    overwrite_count = int(mask.sum())
                    if overwrite_count > 0:
                        logger.info(
                            f"OpusGemmA16W16Tuner: --retune will overwrite "
                            f"{overwrite_count} existing tuned row(s) in "
                            f"{args.tune_file} (in-place; one row per "
                            f"shape stays)."
                        )
                        if args.verbose:
                            logger.info("rows to be overwritten:")
                            print(self.untunedf[mask])
                else:
                    if args.verbose:
                        logger.info("skiped tuned shapes:")
                        print(self.untunedf[mask])
                    self.untunedf = self.untunedf[~mask]

    @staticmethod
    def _bpes_from_dtype_strs(dtype_str, outdtype_str):
        """Compute (lhs_bpe, rhs_bpe, out_bpe) from CSV-form dtype strings."""
        in_bpe = _DTYPE_TORCH_TO_BPE.get(_dtype_csv_str_to_torch(dtype_str), 2)
        out_bpe = _DTYPE_TORCH_TO_BPE.get(_dtype_csv_str_to_torch(outdtype_str), 2)
        return (in_bpe, in_bpe, out_bpe)

    def calculate(self, results, bpes=None):
        """Compute (TFLOPs, GB/s) for a tuner result row.

        Per-shape (bias, dtype, outdtype) live in info[0] now. The 7-key
        layout puts them at slots 4 (bias bool), 5 (dtype), 6 (outdtype).
        Callers outside tune() that pre-compute bpes (e.g. update_tflops_bw
        on a legacy 4/6-key CSV) can pass bpes=... explicitly to bypass.
        """
        if bpes is None:
            info, _time, _err = results
            row_key = info[0]
            # Backwards compat: legacy 4-key / 6-key tuples have no bias
            # slot; route to dtype/outdtype slots that match each layout.
            if len(row_key) >= 7:
                dtype_str, outdtype_str = str(row_key[5]), str(row_key[6])
            elif len(row_key) >= 6:
                dtype_str, outdtype_str = str(row_key[4]), str(row_key[5])
            else:
                dtype_str, outdtype_str = "torch.bfloat16", "torch.bfloat16"
            bpes = self._bpes_from_dtype_strs(dtype_str, outdtype_str)
        return super().calculate(results, bpes=bpes)

    def result_to_df(self, results):
        rows = []
        for el in results:
            info, time, err_ratio = el
            keys, kernelId, splitK, kernelName = info
            # 7-key tuple now; legacy 4/6-key tuples still work.
            if len(keys) >= 7:
                cu_num, M, N, K, bias_v, dtype_str, outdtype_str = keys[:7]
            elif len(keys) >= 6:
                cu_num, M, N, K, dtype_str, outdtype_str = keys[:6]
                bias_v = False
            else:
                cu_num, M, N, K = keys[:4]
                bias_v, dtype_str, outdtype_str = (
                    False,
                    "torch.bfloat16",
                    "torch.bfloat16",
                )
            kernelName = (
                "None"
                if time == self.INVALID_TIME or time == self.INF_TIME
                else (self.getKernelName(kernelId) if kernelName == "" else kernelName)
            )
            tflops, bw = self.calculate(el)
            rows.append(
                {
                    "cu_num": cu_num,
                    "M": M,
                    "N": N,
                    "K": K,
                    "bias": bool(bias_v),
                    "dtype": str(dtype_str),
                    "outdtype": str(outdtype_str),
                    "scaleAB": False,
                    "bpreshuffle": False,
                    "libtype": "opus",
                    "solidx": kernelId,
                    "splitK": splitK,
                    "us": time,
                    "kernelName": kernelName,
                    "err_ratio": err_ratio,
                    "tflops": tflops,
                    "bw": bw,
                }
            )
        return pd.DataFrame(rows, columns=self.OUT_COLUMNS)

    def update_tflops_bw(self, file):
        df = self.get_tuned_gemm_list(file)
        for i in range(len(df)):
            cu_num = df.loc[i, "cu_num"]
            M = df.loc[i, "M"]
            N = df.loc[i, "N"]
            K = df.loc[i, "K"]
            us = df.loc[i, "us"]
            kid_col = "solidx" if "solidx" in df.columns else "kernelId"
            # If the on-disk CSV carries bias / dtype / outdtype columns, use them to keep the keys_tuple
            # aligned with the new 7-wide layout.
            bias_v = bool(df.loc[i, "bias"]) if "bias" in df.columns else False
            dtype_str = (
                str(df.loc[i, "dtype"]) if "dtype" in df.columns else "torch.bfloat16"
            )
            outdtype_str = (
                str(df.loc[i, "outdtype"])
                if "outdtype" in df.columns
                else "torch.bfloat16"
            )
            keys_tuple = (cu_num, M, N, K, bias_v, dtype_str, outdtype_str)
            info = ((keys_tuple, df.loc[i, kid_col], df.loc[i, "splitK"], ""), us, 0)
            tflops, bw = self.calculate(info)
            df.loc[i, "tflops"] = tflops
            df.loc[i, "bw"] = bw
        df.to_csv(file, index=False, na_rep="Null")

    def result_to_csv(self, resultdf, file, concat=False):
        """Replace-if-better gate for --retune.

        Default mode (no --retune): pre_process already strips already-tuned
        shapes from untunedf, so resultdf only contains brand-new keys; the
        parent's append-then-sortResults flow handles them correctly. This
        override is identity-equivalent in that flow.

        --retune mode: pre_process keeps already-tuned shapes in untunedf,
        so resultdf contains rows whose key may match one already on disk.
        Per the user's requirement we only let the new row reach the
        parent (which will append it; sortResults later dedups by keep=last)
        when its `us` is strictly smaller than the cached one. Otherwise
        the new row is dropped here so the cached row remains the winner.

        Why we hook here rather than in update_tunedf: GemmCommonTuner's
        result_to_csv calls update_tunedf only when concat=False (i.e.
        when args.all is True). With --retune we keep args.all=False, so
        the parent flow uses concat=True (append) + sortResults
        drop_duplicates(keep="last"). update_tunedf is therefore never
        invoked on the --retune path; the gate has to live at the
        result_to_csv layer.
        """
        if (
            getattr(self, "_retune_only_if_better", False)
            and not resultdf.empty
            and "us" in resultdf.columns
        ):
            old_df = self.get_tuned_gemm_list(file)
            if not old_df.empty and "us" in old_df.columns:
                key_columns = list(self.keys)
                old_keyed = old_df.assign(
                    _key=old_df[key_columns].apply(tuple, axis=1)
                )[["_key", "us"]].rename(columns={"us": "_old_us"})
                new_keyed = resultdf.assign(
                    _key=resultdf[key_columns].apply(tuple, axis=1)
                )

                merged = new_keyed.merge(old_keyed, on="_key", how="left")
                # NaN _old_us = brand-new key -> always keep (first-time tune).
                keep_mask = merged["_old_us"].isna() | (
                    merged["us"] < merged["_old_us"]
                )

                kept = int(keep_mask.sum())
                rejected = int((~keep_mask).sum())
                if rejected > 0:
                    losers = merged[~keep_mask]
                    logger.info(
                        f"OpusGemmA16W16Tuner: --retune kept "
                        f"{kept}/{kept + rejected} row(s) as improvements; "
                        f"{rejected} row(s) rejected (new us >= old us)."
                    )
                    if getattr(self, "_args_verbose_cached", False):
                        show = losers[key_columns + ["us", "_old_us"]].rename(
                            columns={"us": "new_us"}
                        )
                        logger.info(f"rejected (no-improvement) rows:\n{show}")

                resultdf = resultdf.loc[keep_mask.values].reset_index(drop=True)

        return super().result_to_csv(resultdf, file, concat)

    def tune(self, untunedf, tunedf, args):
        mp_num = args.mp
        shape_grouped = False
        errRatio = args.errRatio
        cu_num = self.get_cu_num()

        # --kid filter (debug): parse the CSV list once.
        forced_kids: set[int] | None = None
        if getattr(args, "kid", None):
            try:
                forced_kids = {int(s) for s in str(args.kid).split(",") if s.strip()}
            except ValueError:
                self.parser.error(
                    f"--kid expects comma-separated ints, got '{args.kid}'"
                )
            if not forced_kids:
                self.parser.error("--kid given but parsed empty kid set")
            unknown = forced_kids - set(a16w16_all_kernels.keys())
            if unknown:
                self.parser.error(
                    f"--kid: unknown kid id(s) {sorted(unknown)} "
                    f"(known a16w16 kids: {sorted(a16w16_all_kernels.keys())})"
                )
            logger.info(
                f"OpusGemmA16W16Tuner: --kid filter active, restricting "
                f"sweep to {sorted(forced_kids)}"
            )

        # Subset-compile safety: collect every kid this debug tuner could touch (full a16w16 family
        # minus host-side rejects per shape) an...
        opus_candidate_kids: set[int] = set()
        for i in range(len(untunedf)):
            M = int(untunedf.loc[i, "M"])
            N = int(untunedf.loc[i, "N"])
            K = int(untunedf.loc[i, "K"])
            bias_v = (
                bool(untunedf.loc[i, "bias"])
                if "bias" in untunedf.columns
                else bool(args.bias)
            )
            shape_cands = candidate_kids_for_shape(M, N, K, bias_v, cu_num)
            if forced_kids is not None:
                # Make sure the requested kids are baked in even if the
                # shape-driven candidate set would have excluded them.
                shape_cands = shape_cands | forced_kids
            opus_candidate_kids |= shape_cands
        if opus_candidate_kids:
            if _ensure_kids_compiled(opus_candidate_kids):
                logger.info(
                    f"opus_gemm_tune: expanded subset-compile sidecar to cover "
                    f"{len(opus_candidate_kids)} candidate kids; "
                    f"module_deepgemm_opus will rebuild on next call."
                )

        # mp_tuner.worker calls `run_perftest(func, *args, **kwargs)` with the func/kwargs we provide here.
        bench_func = run_opus_gemm_bench
        perf_kwargs = {"num_warmup": args.warmup, "num_iters": args.iters}

        logger.info(
            "OpusGemmA16W16Tuner: CUDA graph timing via cuda.Event "
            "(custom run_perftest replacement installed per-subprocess; "
            "bypasses torch.profiler's empty-trace issue on hipgraph replay)"
        )

        task = []
        tasks_data = []
        # generate_data returns a 4-tuple now: (XQ, WQ, Y, bias_or_None).
        opus_data_idx = [0, 1, 2, 3]
        ref_data_idx = [0, 1, 3]
        seed = 0

        for i in range(len(untunedf)):
            M = int(untunedf.loc[i, "M"])
            N = int(untunedf.loc[i, "N"])
            K = int(untunedf.loc[i, "K"])
            batch = int(untunedf.loc[i, "batch"]) if "batch" in untunedf.columns else 1

            # Per-row bias / dtype / outdtype: bias is part of self.keys now, so the column is guaranteed to
            # exist by pre_process.
            bias_v = (
                bool(untunedf.loc[i, "bias"])
                if "bias" in untunedf.columns
                else bool(args.bias)
            )
            dtype_str = (
                str(untunedf.loc[i, "dtype"])
                if "dtype" in untunedf.columns
                else _dtype_torch_to_csv_str(_DTYPE_SHORT_TO_TORCH[args.dtype])
            )
            outdtype_str = (
                str(untunedf.loc[i, "outdtype"])
                if "outdtype" in untunedf.columns
                else _dtype_torch_to_csv_str(_DTYPE_SHORT_TO_TORCH[args.outdtype])
            )
            in_dtype = _dtype_csv_str_to_torch(dtype_str)
            out_dtype = _dtype_csv_str_to_torch(outdtype_str)

            # Gate opus candidates by the same tolerance the other a16w16
            # solutions use (5e-2 for bf16, 1e-2 otherwise) instead of the
            # looser hardcoded 2e-2/1.0.
            check_rtol = 5e-2 if out_dtype == dtypes.bf16 else 1e-2
            check_atol = check_rtol

            # Sanity check: a16w16-family kernels lock the input to bf16 today (the launcher TORCH_CHECKs
            # XQ.dtype()==BFloat16).
            if in_dtype is not dtypes.bf16:
                logger.warning(
                    f"OpusGemmA16W16Tuner: skipping row M={M} N={N} K={K} "
                    f"with dtype={dtype_str} (a16w16 kernels currently only "
                    f"accept bf16 input)"
                )
                tasks_data.append((0, ()))
                continue

            seed = seed + 1

            total_kernel_nums = 0
            # 7-tuple matches self.keys; result_to_df / calculate read
            # bias / dtype / outdtype from slots 4 / 5 / 6.
            info_keys = (cu_num, M, N, K, bias_v, dtype_str, outdtype_str)

            for kid in a16w16_kernel_ids:
                # --kid filter (debug): skip everything outside the explicit set.
                if forced_kids is not None and kid not in forced_kids:
                    continue
                k_inst = a16w16_all_kernels[kid]

                # Pre-filter kids that can't produce correct output for this shape.
                if _kid_rejects_shape(k_inst, M, N, K):
                    if forced_kids is not None and kid in forced_kids:
                        logger.warning(
                            f"OpusGemmA16W16Tuner: --kid {kid} rejected "
                            f"by shape filter for (M={M},N={N},K={K}); skipping"
                        )
                    continue
                # bias=True is only consumed by split-barrier (a16w16) and splitk (a16w16_flatmm_splitk) kids;
                # flatmm kids reject any non-empty b...
                if _kid_rejects_bias(k_inst, bias_v):
                    continue
                # outdtype filter: the per-arch tune_lookup tables are keyed by (kid, D_C).
                _OUT_TORCH_TO_CTYPE = {
                    dtypes.bf16: "bf16_t",
                    dtypes.fp32: "fp32_t",
                }
                _is_splitk_tag = kid in SPLITK_KIDS
                _need = _OUT_TORCH_TO_CTYPE.get(out_dtype)
                if _kid_rejects_outdtype(k_inst, out_dtype):
                    continue
                if (
                    not _is_splitk_tag
                    and _need is not None
                    and _need not in getattr(k_inst, "output_dtypes", [])
                ):
                    continue

                # SplitK candidate set per shape+kid.
                if kid in SPLITK_KIDS:
                    splitK_range = candidate_splitK(M, N, K, batch, cu_num, k_inst)
                else:
                    splitK_range = [0]

                for splitK in splitK_range:
                    info = (info_keys, kid, splitK, "")
                    # Pass bias / dtype / outdtype down to generate_data via the gen_args tuple.
                    gen_args = (batch, M, N, K, seed, in_dtype, out_dtype, bias_v)
                    # opus_gemm_ref(XQ, WQ, bias, out_dtype): pass the
                    # data-indexed args plus out_dtype as a positional.
                    ref_args = (ref_data_idx, out_dtype)
                    task.append(
                        (
                            info,
                            generate_data,
                            gen_args,
                            bench_func,
                            (opus_data_idx, kid, splitK),
                            perf_kwargs,
                            opus_gemm_ref,
                            ref_args,
                            {},
                            None,
                            check_rtol,
                            check_atol,
                        )
                    )
                    total_kernel_nums += 1

            tasks_data.append((total_kernel_nums, ()))

        ret = []
        if task:
            # Silence per-task `avg: X us/iter with hipgraph` and `no valida data after post process!` spam
            # from aiter.test_common during the...
            prev_level = logger.level
            if not (_AITER_VERBOSE or getattr(args, "verbose", False)):
                logger.setLevel(logging.WARNING)
            try:
                ret = mp_tuner(
                    task,
                    tasks_data,
                    mp_num,
                    False,
                    shape_grouped,
                    errRatio,
                    timeout=args.timeout,
                    verbose=args.verbose,
                )
            finally:
                logger.setLevel(prev_level)

            # Post-filter: perftest under CUDA graph capture+replay occasionally yields us=0.0 exactly when
            # the torch.profiler trace comes ba...
            bad = 0
            fixed = []
            for item in ret:
                info, us, err = item
                if us is not None and us == 0.0:
                    fixed.append((info, INVALID_TIME, 1.0))
                    bad += 1
                else:
                    fixed.append(item)
            if bad:
                logger.warning(
                    f"OpusGemmA16W16Tuner: {bad}/{len(ret)} measurements "
                    f"yielded us==0 (empty profiler trace under CUDA graph "
                    f"capture). Marked invalid."
                )
            ret = fixed
        return ret


if __name__ == "__main__":
    import sys

    sys.stderr.write(
        "\n"
        "==============================================================\n"
        "  opus_gemm_tune.py is DEBUG-ONLY.\n"
        "  For production tuning use:\n"
        "      python3 gradlib/gemm_tuner.py --libtype opus\n"
        "  (or --libtype all to tune all backends in one pass).\n"
        "  gradlib writes to aiter/configs/bf16_tuned_gemm.csv (or the\n"
        "  path passed via --tuned_file / GTUNE_TUNED) and stamps every\n"
        "  opus row with libtype='opus' so the opus runtime dispatch\n"
        "  picks it up via aiter.ops.opus.common.lookup_tuned().\n"
        f"  This script writes to {OPUS_DEBUG_TUNED_CSV} by default and\n"
        "  will not pollute the global aiter/configs/ tree.\n"
        "==============================================================\n"
        "\n"
    )
    # 17-column schema matching aiter/configs/model_configs/gptoss_bf16_tuned_gemm.csv.
    key = ["cu_num", "M", "N", "K", "bias", "dtype", "outdtype"]
    resultList = [
        "scaleAB",
        "bpreshuffle",
        "libtype",
        "solidx",
        "splitK",
        "us",
        "kernelName",
        "err_ratio",
        "tflops",
        "bw",
    ]
    tuner = OpusGemmA16W16Tuner(
        "OpusGemmA16W16Tuner",
        key=key,
        resultList=resultList,
        description="Tune opus GEMM a16w16 / a16w16_flatmm / a16w16_flatmm_splitk kernels",
    )

    args = tuner.parse_args()
    tuner.run(args, False)
