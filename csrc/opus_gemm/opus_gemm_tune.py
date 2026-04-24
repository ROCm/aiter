# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
import logging
import os

import pandas as pd
import torch
from aiter import dtypes, logger
from aiter.jit.core import AITER_ROOT_DIR

# opus-private tuned CSV default path. Lives under aiter/ops/opus/configs/
# so all opus artifacts are co-located. The Python user-facing wrapper in
# aiter/ops/opus/common.py (PR2') reads the same path. Kept out of
# aiter/jit/core.py's AITER_CONFIGS (which is reserved for aiter-global
# GEMM config shared across triton/asm/flydsl backends).
OPUS_A16W16_TUNED_CSV = os.getenv(
    "AITER_OPUS_A16W16_TUNED_CSV",
    f"{AITER_ROOT_DIR}/aiter/ops/opus/configs/opus_gemm_a16w16_tuned.csv",
)
from aiter.utility.base_tuner import GemmCommonTuner, INVALID_TIME
from aiter.utility.mp_tuner import mp_tuner
from aiter.ops.opus.gemm_op_a16w16 import opus_gemm_a16w16_tune as _opus_gemm_a16w16_tune


# Silence per-task `avg: X us/iter with hipgraph` and `no valida data after
# post process!` log lines from aiter.test_common by default. The tuner
# produces hundreds of these across a typical tuning run and they drown out
# the summary. Re-enable with `-v` / AITER_VERBOSE=1.
# - `aiter` logger is used by perftest() in test_common.py for per-task logs
# - Only bump it to WARNING; critical tuner summary still goes through stdout
#   via logger.info() in base_tuner.tune_summary() using a separate code path
#   that we want to keep. We re-enable aiter logger at INFO for the tuner
#   summary in a finally block.
_AITER_VERBOSE = bool(int(os.environ.get("AITER_VERBOSE", "0")))
from opus_gemm_common import (
    a16w16_kernels_list,
    a16w16_flatmm_kernels_list,
    a16w16_flatmm_splitk_kernels_list,
)


# Merge all three a16w16-family pipelines into one tuner search:
#   * split-barrier a16w16 (kids 4..9) - ignores splitK
#   * a16w16_flatmm       (kids 100..115) - ignores splitK
#   * a16w16_flatmm_splitk (kids 200..210) - splitK = literal KBatch in {0, 2..32}
a16w16_all_kernels = {
    **a16w16_kernels_list,
    **a16w16_flatmm_kernels_list,
    **a16w16_flatmm_splitk_kernels_list,
}
a16w16_kernel_ids = sorted(a16w16_all_kernels.keys())


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _flatmm_splitk_pfk(k) -> int:
    """Host-side computation of Traits::prefetch_k_iter for a splitk instance.

    Mirrors opus_flatmm_splitk_traits's formula so the heuristic can pre-compute
    the per-split iter budget without a device call. Hardcodes LDS=163840 (gfx950),
    same convention as the traits struct.
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
    """Pick literal KBatch values in {0, 2..32} worth probing for this (shape, kid).

    Heuristic = "calculated saturation split + powers of 2" (user-confirmed).
    See section 9 of splitk_flatmm_aiter plan.

    Output:
      - Always contains 0 (no-split fallback).
      - At most 6 values total (0 plus up to 5 of {kb_calc, 2, 4, 8, 16, 32}).
      - Storage is literal KBatch, matching gptoss_bf16_tuned_gemm.csv convention.
    """
    B_M, B_N, B_K = k_inst.B_M, k_inst.B_N, k_inst.B_K
    wg_per_cu = getattr(k_inst, "WG_PER_CU", 2)
    tiles_mn = _ceil_div(M, B_M) * _ceil_div(N, B_N)
    base_grid = tiles_mn * max(batch, 1)
    total_iters = _ceil_div(K, B_K)
    pfk = _flatmm_splitk_pfk(k_inst)

    candidates = {0}

    # pfk cap: launcher auto-clamps anything above this; no point probing it.
    max_split_k = min(32, max(1, total_iters // max(pfk, 1)))
    if max_split_k < 2:
        return [0]

    # Saturation target: 2 * wg_per_cu * cu_num gives a 2x headroom above "all
    # CUs busy" to absorb tail imbalance.
    target_grid = 2 * wg_per_cu * cu_num
    if base_grid >= target_grid:
        return [0]  # already saturated; splitK costs reduce kernel for no gain

    # (a) calculated: smallest KBatch hitting the saturation target
    kb_calc = max(2, _ceil_div(target_grid, max(base_grid, 1)))
    if kb_calc <= max_split_k:
        candidates.add(kb_calc)

    # (b) powers of 2 in [2, max_split_k]
    for kb in (2, 4, 8, 16, 32):
        if kb > max_split_k:
            break
        candidates.add(kb)

    return sorted(candidates)


def generate_data(batch, m, n, k, seed, device="cuda"):
    # gen_data is the earliest subprocess-side hook mp_tuner.work_group
    # invokes (mp_tuner.py:139-143), before worker() is ever called. This
    # is where we install the custom run_perftest override so that when
    # worker() later does `from aiter.test_common import run_perftest`
    # (mp_tuner.py:24), it picks up our CUDA-graph + cuda.Event timing
    # version instead of the stock profiler-based one. See
    # _install_opus_perftest_once() and _opus_run_perftest() above.
    _install_opus_perftest_once()

    torch.manual_seed(seed)
    # Use the exact (M, N, K) from the tune request. If a candidate's kernel
    # cannot handle this K (e.g. split-barrier a16w16 needs K >= 2*B_K and
    # loops even; flatmm needs K >= pfk*B_K; splitk needs K >= split_k*pfk*B_K),
    # run_opus_gemm's max_delta check will flag it and mp_tuner.worker will
    # mark it invalid. Using a K floor would mask such mismatches and mis-
    # attribute perf to the wrong shape.
    XQ = torch.randn((batch, m, k), dtype=dtypes.bf16, device=device)
    WQ = torch.randn((batch, n, k), dtype=dtypes.bf16, device=device)
    Y = torch.empty((batch, m, n), dtype=dtypes.bf16, device=device)
    return XQ, WQ, Y


MAX_DELTA_SCALE = 0.1


def opus_gemm_ref(XQ, WQ):
    return torch.bmm(XQ.float(), WQ.float().transpose(-1, -2)).to(dtypes.bf16)


def run_opus_gemm(XQ, WQ, Y, kernelId, splitK):
    """Eager-path tuner func: runs the kernel AND an on-the-fly max_delta check.

    The check raises RuntimeError when the output is numerically off; mp_tuner's
    worker catches it and marks the candidate invalid. Used when --no-graph is
    passed (i.e. graph mode disabled) so the per-iter check is safe (no CUDA
    graph capture).
    """
    _quiet_aiter_logger_once()
    _opus_gemm_a16w16_tune(XQ, WQ, Y, kernelId, splitK)
    ref = torch.bmm(XQ.float(), WQ.float().transpose(-1, -2)).to(dtypes.bf16)
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


def run_opus_gemm_bench(XQ, WQ, Y, kernelId, splitK):
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
    _opus_gemm_a16w16_tune(XQ, WQ, Y, kernelId, splitK)

    capturing = torch.cuda.is_current_stream_capturing()

    if not capturing:
        # Correctness gate (per-task, outside-capture only).
        task_key = (id(XQ), id(WQ), id(Y), int(kernelId), int(splitK))
        if task_key not in _bench_max_delta_checked:
            _bench_max_delta_checked.add(task_key)
            ref = torch.bmm(XQ.float(), WQ.float().transpose(-1, -2)).to(dtypes.bf16)
            max_delta = (Y.float() - ref.float()).abs().max().item()
            max_ref = ref.float().abs().max().item()
            bound = max(max_ref * MAX_DELTA_SCALE, 1.0)
            if max_delta > bound:
                raise RuntimeError(
                    f"maxDelta {max_delta:.1f} > bound {bound:.1f} "
                    f"(max|ref|={max_ref:.1f}, scale={MAX_DELTA_SCALE}) "
                    f"for kid={kernelId} splitK={splitK}"
                )

        # Capture-safe sync: guarantees the warmup kernel has completed
        # before we read Y for the max_delta check (above) or before the
        # outer timing loop enters a new measurement iter.
        torch.cuda.current_stream().synchronize()
    return Y


# ============================================================================
# Custom run_perftest for opus a16w16 tuner: CUDA graph + cuda.Event timing.
# ============================================================================
#
# Why replace run_perftest?
# -------------------------
# The stock `aiter.test_common.run_perftest(testGraph=True)` path relies on
# torch.profiler to attribute device time across a captured-and-replayed
# CUDA graph. On gfx950 + ROCm 6.x, roctracer returns **zero device events**
# for `graph.replay()` containing multiple kernel launches (the splitk
# pipeline issues a `main + reduce` pair per iter). We verified this by
# inspecting `prof.events()` directly: 5 CPU events, 0 CUDA events. As a
# result, `get_trace_perf` returns 0.0 and mp_tuner.worker's `us == 0`
# retry loop fires (the `!!!! us = 0, try N run` / `Warning: try run 3
# times, but still get 0!` spam on stderr).
#
# Stream sync + anchor-op + re-warmup tricks inside the bench func cannot
# fix this -- the empty trace is a roctracer-vs-hipgraph compatibility
# issue at the tracing layer, not something the kernel can influence.
#
# Alternative: measure with torch.cuda.Event around graph.replay() instead
# of torch.profiler. CUDA Event elapsed_time is a hipEventElapsedTime call
# and works correctly with hipgraph (the end-event is recorded after the
# replay and sees all kernel completions). This gives us:
#   * True hipgraph-mode timing (matches C++ profile binary perf).
#   * Zero Python/dispatcher overhead in the measurement window.
#   * No profiler / roctracer dependency.
#   * No `us == 0` spam from mp_tuner.
#
# How is this wired into mp_tuner without modifying it?
# -----------------------------------------------------
# mp_tuner.worker imports run_perftest *inside* the worker body:
#     from aiter.test_common import run_perftest
# This runs on every task invocation in the subprocess, looking up the
# current value of `run_perftest` on the `aiter.test_common` module. We
# monkey-patch that module attribute in each worker subprocess at first
# invocation (via _install_opus_perftest_once below). The worker then
# reads our replacement instead of the stock version.
#
# We do NOT modify aiter.test_common or aiter.utility.mp_tuner source code.
# The patch is strictly per-subprocess and strictly scoped to the opus
# tuner (it is installed only by our bench func, run_opus_gemm_bench).
# ============================================================================


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
    # Inside the subprocess the bench func is `func` directly (kwargs may
    # be empty). We accept num_rotate_args / needTrace purely for signature
    # compatibility; the opus tuner never sets them.
    stream = torch.cuda.current_stream()

    # Warmup on the main stream. Keeps the first measurement stable and
    # triggers JIT / kernel loads before capture.
    for _ in range(max(1, num_warmup)):
        data = func(*args, **kwargs)
    torch.cuda.synchronize()

    # Capture the timing loop. torch.cuda.graph requires a side stream.
    # Anything that ran on the main stream before this point is visible to
    # the captured graph because we sync'd.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(side):
        # A second warmup is recommended on the side stream to "prime" the
        # allocator and any per-stream state so capture records a stable
        # sequence.
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


def _kid_rejects_shape(k_inst, M, N, K):
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

    Bugs (a) live in this repo's own TORCH_CHECKs and will never be "fixed"
    without touching generated launchers; (b) reflect kernel-level bugs in
    non-splitk pipelines (csrc/opus_gemm/include/pipeline/opus_gemm_pipeline_a16w16.cuh
    and opus_gemm_pipeline_a16w16_flatmm.cuh) -- tracking a fix outside this
    file. Until those are fixed, the tuner simply hides the broken kids.
    """
    B_K = k_inst.B_K
    loops = _ceil_div(K, B_K)

    if k_inst.kernel_tag == "a16w16":
        # (a) K constraints from _gen_noscale_instance TORCH_CHECK.
        if loops < 2 or (loops % 2 != 0):
            return True
        # (b) N-alignment bug in split-barrier store_if.
        if N % 16 != 0:
            return True
        return False

    if k_inst.kernel_tag == "a16w16_flatmm":
        # (a) K constraint.
        if loops < _flatmm_splitk_pfk(k_inst):
            return True
        # (b) N-alignment bug (same shape as split-barrier).
        if N % 16 != 0:
            return True
        # (b) K-tail bug: flatmm has no mask_va_tail analogue, so K must be
        # an exact multiple of B_K or the last iter accumulates garbage.
        if K % B_K != 0:
            return True
        return False

    if k_inst.kernel_tag == "a16w16_flatmm_splitk":
        # (a) K constraint (splitK=1 baseline; launcher clamps bigger splitK).
        if loops < _flatmm_splitk_pfk(k_inst):
            return True
        # (b) No known correctness bugs for splitk. mask_va_tail handles the
        # K tail; the tail-store path in the reduce kernel handles any N.
        return False

    return False


class OpusGemmA16W16Tuner(GemmCommonTuner):
    ARG_DEFAULTS = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": OPUS_A16W16_TUNED_CSV,
        "untune_file": "aiter/configs/model_configs/gptoss_bf16_untuned_gemm.csv",
        # Tighter than GemmCommonTuner default (0.05). Under CUDA graph mode
        # the only numerical guard is mp_tuner.worker's post-run
        # checkAllclose(err_ratio) — per-iter max_delta check is also done in
        # run_opus_gemm_bench (see below), but this gate catches silent "only
        # a few cells wrong" bugs (e.g. N not aligned to VEC_C vector store
        # width causes sporadic cross-row writes; only ~0.001 fraction of
        # cells end up wrong, slipping past a 0.05 gate). Good kernels here
        # produce err_ratio == 0; setting the gate to 0.001 lets small bf16
        # rounding drift through but catches the silent-corruption cases.
        "errRatio": 0.001,
        "batch": 100,
        "profile_file": "",
    }

    # 17-column schema matching aiter/configs/model_configs/gptoss_bf16_tuned_gemm.csv
    # exactly. GemmCommonTuner's default emits a narrower 11-column schema; we
    # override result_to_df and update_tflops_bw to produce the full schema so
    # downstream consumers (e.g. gen_lookup_dict via get_tune_dict) see a
    # consistent format across asm/triton/ck/opus libtypes.
    OUT_COLUMNS = [
        "cu_num", "M", "N", "K",
        "bias", "dtype", "outdtype", "scaleAB", "bpreshuffle",
        "libtype", "solidx", "splitK",
        "us", "kernelName", "err_ratio", "tflops", "bw",
    ]

    def getKernelName(self, kernelId):
        k = a16w16_all_kernels.get(kernelId)
        return k.name if k else None

    def _setup_specific_arguments(self):
        # Release `-k` short-form from base_tuner's `-k/--splitK` so we can
        # repurpose it for the K dim below, giving a symmetric `-m/-n/-k` set
        # for single-shape CLI tuning. Long form `--splitK` is kept and moved
        # to short `-s`; `args.splitK` remains stable so no downstream change.
        #
        # argparse stores each action in 3 places (`_actions`,
        # `_option_string_actions`, and each `_action_groups[i]._group_actions`).
        # We have to remove from all 3 or `--help` will still print the stale
        # entry via the action_groups walk.
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

        # Retained for backwards compat. Tune path always runs eager +
        # capture-safe stream sync (see tune() comment): the CUDA-graph
        # perftest path silently drops device events for the splitk
        # main+reduce pair under roctracer on gfx950.
        self.parser.add_argument(
            "--no-graph",
            dest="graph",
            action="store_false",
            default=True,
            help="(Deprecated, no-op) CUDA graph mode is always off for the "
            "opus tuner. Retained for argparse compat with upstream GemmCommonTuner.",
        )

        # Direct M/N/K entry -- skip the untuned csv input. Useful for iterating
        # on a single shape during development without editing any file.
        # Usage: python opus_gemm_tune.py -m 128 -n 2048 -k 4096 -o /tmp/t.csv
        self.parser.add_argument(
            "-m", "--M", dest="shape_m", type=int, default=None,
            help="Tune a single M dim (pairs with -n/-k). Bypasses -i csv.",
        )
        self.parser.add_argument(
            "-n", "--N", dest="shape_n", type=int, default=None,
            help="Tune a single N dim (pairs with -m/-k). Bypasses -i csv.",
        )
        self.parser.add_argument(
            "-k", "--K", dest="shape_k", type=int, default=None,
            help="Tune a single K dim (pairs with -m/-n). Bypasses -i csv.",
        )
        self.parser.add_argument(
            "-b", "--BATCH", dest="shape_batch", type=int, default=1,
            help="Batch dim for single-shape CLI tuning (default 1).",
        )

    def pre_process(self, args):
        """Override to support CLI-based single-shape tuning (-m/-n/-K)."""
        if (
            args.shape_m is not None
            and args.shape_n is not None
            and args.shape_k is not None
        ):
            # Build a one-row untunedf directly from CLI args.
            self.untunedf = pd.DataFrame(
                [
                    {
                        "cu_num": self.get_cu_num(),
                        "M": int(args.shape_m),
                        "N": int(args.shape_n),
                        "K": int(args.shape_k),
                    }
                ],
                columns=self.keys,
            )
            self.tunedf = self.get_tuned_gemm_list(args.tune_file)
            if "batch" in self.untunedf.columns:
                self.untunedf["batch"] = int(args.shape_batch)
            logger.info(
                f"OpusGemmA16W16Tuner: single-shape CLI tune "
                f"(M={args.shape_m}, N={args.shape_n}, K={args.shape_k}, "
                f"batch={args.shape_batch})"
            )
            return

        # Otherwise fall back to csv-driven behavior from GemmCommonTuner.
        super().pre_process(args)

    def calculate(self, results, bpes=(2, 2, 2)):
        return super().calculate(results, bpes=(2, 2, 2))

    def result_to_df(self, results):
        rows = []
        for el in results:
            info, time, err_ratio = el
            keys, kernelId, splitK, kernelName = info
            cu_num, M, N, K = keys
            kernelName = (
                "None"
                if time == self.INVALID_TIME or time == self.INF_TIME
                else (
                    self.getKernelName(kernelId) if kernelName == "" else kernelName
                )
            )
            tflops, bw = self.calculate(el)
            rows.append(
                {
                    "cu_num": cu_num,
                    "M": M,
                    "N": N,
                    "K": K,
                    "bias": False,
                    "dtype": "torch.bfloat16",
                    "outdtype": "torch.bfloat16",
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
            info = (
                ((cu_num, M, N, K), df.loc[i, kid_col], df.loc[i, "splitK"], ""),
                us,
                0,
            )
            tflops, bw = self.calculate(info)
            df.loc[i, "tflops"] = tflops
            df.loc[i, "bw"] = bw
        df.to_csv(file, index=False, na_rep="Null")

    def tune(self, untunedf, tunedf, args):
        mp_num = args.mp
        shape_grouped = False
        errRatio = args.errRatio
        cu_num = self.get_cu_num()

        # mp_tuner.worker calls `run_perftest(func, *args, **kwargs)` with
        # the func/kwargs we provide here. We install a custom run_perftest
        # inside each subprocess (via _install_opus_perftest_once(), invoked
        # from generate_data() which runs before worker() in work_group)
        # that times the kernel via CUDA graph + cuda.Event instead of the
        # stock torch.profiler path. The profiler path returns 0 CUDA
        # events on gfx950 for splitk's main+reduce kernel sequence under
        # hipgraph replay -- that's the root cause of the "us = 0" retry
        # spam and the "no valid candidate found" failures for M=1.
        #
        # With our replacement:
        #   * Graph capture + cuda.Event measurement works correctly for
        #     every (kid, splitK) on every (M, N, K) we tested (M=1..2048).
        #   * No Python/dispatcher overhead contaminates the measurement
        #     window (the replay runs end-to-end on the GPU).
        #   * No roctracer dependency; no empty-trace edge cases.
        #
        # We pass testGraph=False here because our replacement ignores that
        # flag and always does graph mode. The flag is kept out of the
        # kwargs only to make the signature explicit (our replacement
        # accepts it for API compat).
        bench_func = run_opus_gemm_bench
        perf_kwargs = {"num_warmup": args.warmup, "num_iters": args.iters}
        check_rtol, check_atol = 2e-2, 1.0

        logger.info(
            "OpusGemmA16W16Tuner: CUDA graph timing via cuda.Event "
            "(custom run_perftest replacement installed per-subprocess; "
            "bypasses torch.profiler's empty-trace issue on hipgraph replay)"
        )

        task = []
        tasks_data = []
        opus_data_idx = [0, 1, 2]
        ref_data_idx = [0, 1]
        seed = 0

        for i in range(len(untunedf)):
            M = int(untunedf.loc[i, "M"])
            N = int(untunedf.loc[i, "N"])
            K = int(untunedf.loc[i, "K"])
            batch = (
                int(untunedf.loc[i, "batch"]) if "batch" in untunedf.columns else 1
            )
            seed = seed + 1

            total_kernel_nums = 0
            info_keys = (cu_num, M, N, K)

            for kid in a16w16_kernel_ids:
                k_inst = a16w16_all_kernels[kid]

                # Pre-filter kids that can't produce correct output for
                # this shape. Covers both host-side TORCH_CHECK rejections
                # (K constraints) and known silent-correctness bugs in
                # non-splitk pipelines at N % 16 != 0 / K % B_K != 0.
                # See _kid_rejects_shape() for the full rule set.
                if _kid_rejects_shape(k_inst, M, N, K):
                    continue

                # SplitK candidate set per shape+kid. Non-splitk kids always
                # get splitK=0; splitk kids go through the heuristic.
                if k_inst.kernel_tag == "a16w16_flatmm_splitk":
                    splitK_range = candidate_splitK(M, N, K, batch, cu_num, k_inst)
                else:
                    splitK_range = [0]

                for splitK in splitK_range:
                    info = (info_keys, kid, splitK, "")
                    task.append(
                        (
                            info,
                            generate_data,
                            (batch, M, N, K, seed),
                            bench_func,
                            (opus_data_idx, kid, splitK),
                            perf_kwargs,
                            opus_gemm_ref,
                            (ref_data_idx,),
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
            # Silence per-task `avg: X us/iter with hipgraph` and
            # `no valida data after post process!` spam from aiter.test_common
            # during the inner tuning loop. Tuner summary is emitted after
            # this finishes and uses the same logger, so we restore the level.
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

            # Post-filter: perftest under CUDA graph capture+replay occasionally
            # yields us=0.0 exactly when the torch.profiler trace comes back
            # empty ("no valida data after post process!" from test_common.py).
            # Leaving those rows as 0.0 would make them appear as the fastest
            # candidates; mark them invalid so base_tuner.post_process drops
            # them (us != INVALID_TIME filter in post_process already catches
            # us=-1).
            #
            # We ONLY treat us == 0.0 as the empty-trace sentinel. Tiny but
            # nonzero measurements (e.g. 0.3-0.8us) are real on M=1 shapes
            # where the full kernel + reduce latency falls below 1us on
            # MI300/MI350; a previous blanket `us < 1.0` floor was causing
            # false-rejects for those shapes (observed: M=1 N=200 K=5120
            # dropped 49/49 candidates and the tuner reported "no valid
            # candidate found"). The fix is to reject only us == 0.0.
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
    # Use the 17-column schema from gptoss_bf16_tuned_gemm.csv so base_tuner's
    # self.columns = key + resultList matches our result_to_df output.
    # Column order inside the DataFrame is:
    #   key       = [cu_num, M, N, K]             (4 cols)
    #   resultList = [bias, dtype, outdtype, scaleAB, bpreshuffle,
    #                 libtype, solidx, splitK, us, kernelName,
    #                 err_ratio, tflops, bw]      (13 cols)
    # total = 17 cols, in exact reference CSV order.
    key = ["cu_num", "M", "N", "K"]
    resultList = [
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
    tuner = OpusGemmA16W16Tuner(
        "OpusGemmA16W16Tuner",
        key=key,
        resultList=resultList,
        description="Tune opus GEMM a16w16 / a16w16_flatmm / a16w16_flatmm_splitk kernels",
    )

    args = tuner.parse_args()
    tuner.run(args, False)
