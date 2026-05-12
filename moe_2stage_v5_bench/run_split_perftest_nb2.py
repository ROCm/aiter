"""Stage-split perftest runner for /app/aiter/op_tests/test_moe_2stage.py.

For each (config, M) pair this:
  1. Uses ``test_fmoe``'s setup logic to prepare quantised inputs + shuffled
     weights.
  2. Monkey-patches ``_gfx1250_moe_stage1`` / ``_gfx1250_moe_stage2`` to
     capture the last call's positional + kwarg args (so we have ready-to-
     replay closures that hit the real FlyDSL kernels with the exact same
     tensors).
  3. Runs ``fused_moe`` once via the test (this also reports fused time and
     prints the AITER_GFX1250_PROBE tile-info log).
  4. Uses ``aiter.test_common.run_perftest`` to time stage1 and stage2
     **separately**, each with their own warmup + iters.

Notes
-----
* ``run_perftest`` returns ``(data, avg_us)`` where ``avg`` is computed by
  ``aiter`` using ``torch.profiler`` (and is the actual on-GPU kernel time
  median, see ``aiter/test_common.perftest``).
* The MoE stage kernels mutate the ``out`` tensor in-place; ``run_perftest``
  with ``num_rotate_args=1`` (we pass it that way) avoids deep-copying large
  weight buffers and reuses the same ``out`` tensor each iteration -- which
  is exactly the same semantics as the production caller and matches
  ``fused_moe``'s timing window.
* All output is also dumped to stdout so the wrapper shell script can
  redirect into the per-config log files alongside the markdown summary.
"""

from __future__ import annotations
import argparse
import json
import os
import statistics
import sys
import time

# ----------------------- env setup (must precede imports) -------------------
os.environ.setdefault("PYTHONPATH", "")
for _p in ("/app/FlyDSL/build-fly/python_packages", "/app/FlyDSL"):
    if _p not in os.environ["PYTHONPATH"]:
        os.environ["PYTHONPATH"] = _p + ":" + os.environ["PYTHONPATH"]
sys.path.insert(0, "/app/FlyDSL/build-fly/python_packages")
sys.path.insert(0, "/app/FlyDSL")
sys.path.insert(0, "/app/aiter")

os.environ.setdefault("ENABLE_CK", "0")
os.environ.setdefault("AITER_USE_OPUS_MOE_SORTING", "1")
os.environ.setdefault("AITER_LOG_MORE", "1")
# NOTE: AITER_GFX1250_PROBE=1 emits a few hundred us of stdout per
# fused_moe() iter (see fused_moe.py:1011, :1195). We capture the
# real tile_n / tile_k / block_m / in_dtype / activation via the
# monkey-patched stage1/stage2 kwargs instead -- those are the
# *actual* values handed to the FlyDSL kernel. Leaving probe off keeps
# the fused-time number comparable to the baseline doc in
# /app/aiter/moe_2stage_bench_results.md.
os.environ.setdefault("AITER_GFX1250_PROBE", "0")
os.environ.setdefault("AITER_MOE_WARMUP", "5")
os.environ.setdefault("AITER_MOE_ITERS", "20")
os.environ.setdefault("AITER_MOE_L2_FLUSH", "1")

import torch
import aiter
from aiter import dtypes
import aiter.fused_moe as fm
from aiter.test_common import run_perftest


# ----------------------- inject num_buffers=2 into FlyDSL compile calls -----
# fused_moe._gfx1250_moe_stage1/_stage2 call ``compile_moe_gemm1`` /
# ``compile_moe_gemm2`` without specifying ``num_buffers``, which defaults to
# 1 (no K-pipelining). To compare the pipelined variant we wrap both compile
# entry points to force-set ``num_buffers=2``. The wrapped functions live in
# the FlyDSL mxscale module; we patch them there so the lru_cache key
# includes the new value and yields a freshly compiled kernel.
import aiter.ops.flydsl.kernels.moe_gemm_2stage_mxscale_gfx1250 as _mxscale_mod

_NUM_BUFFERS_OVERRIDE = int(os.environ.get("AITER_GFX1250_NUM_BUFFERS", "2"))

_orig_compile_moe_gemm1 = _mxscale_mod.compile_moe_gemm1
_orig_compile_moe_gemm2 = _mxscale_mod.compile_moe_gemm2


def _compile_moe_gemm1_nb_override(*args, **kwargs):
    kwargs.setdefault("num_buffers", _NUM_BUFFERS_OVERRIDE)
    return _orig_compile_moe_gemm1(*args, **kwargs)


def _compile_moe_gemm2_nb_override(*args, **kwargs):
    kwargs.setdefault("num_buffers", _NUM_BUFFERS_OVERRIDE)
    return _orig_compile_moe_gemm2(*args, **kwargs)


_mxscale_mod.compile_moe_gemm1 = _compile_moe_gemm1_nb_override
_mxscale_mod.compile_moe_gemm2 = _compile_moe_gemm2_nb_override
print(
    f"[NB_OVERRIDE] compile_moe_gemm1 / compile_moe_gemm2 wrapped to inject "
    f"num_buffers={_NUM_BUFFERS_OVERRIDE}",
    flush=True,
)


# ----------------------- monkey-patch stage1 / stage2 -----------------------
_last_s1_args: list = []
_last_s2_args: list = []
_s1_calls = [0]
_s2_calls = [0]


def _make_capture(orig_func, store):
    def wrapped(*args, **kwargs):
        store.clear()
        store.append((args, dict(kwargs)))
        return orig_func(*args, **kwargs)
    return wrapped


_orig_s1 = fm._gfx1250_moe_stage1
_orig_s2 = fm._gfx1250_moe_stage2
fm._gfx1250_moe_stage1 = _make_capture(_orig_s1, _last_s1_args)
fm._gfx1250_moe_stage2 = _make_capture(_orig_s2, _last_s2_args)
try:
    fm.get_2stage_cfgs.cache_clear()
except Exception:
    pass


# ---- bring in test_fmoe ----
import importlib

_saved_argv = sys.argv[:]
sys.argv = [sys.argv[0], "--no-flydsl-csv", "--no-legacy"]
try:
    test_mod = importlib.import_module("op_tests.test_moe_2stage")
finally:
    sys.argv = _saved_argv


def _stats(xs):
    xs = sorted(xs)
    if not xs:
        return {"median": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "median": xs[len(xs) // 2],
        "mean": sum(xs) / len(xs),
        "min": xs[0],
        "max": xs[-1],
    }


def run_case(tag, token, model_dim, inter_dim, expert, topk, aq, wq, act):
    """Run a single (config, M) point; returns dict with all metrics."""
    print(
        f"\n========== {tag}  M={token} "
        f"dim={model_dim}/{inter_dim} E={expert} topk={topk} "
        f"AQ={aq} WQ={wq} act={act} ==========",
        flush=True,
    )

    _last_s1_args.clear()
    _last_s2_args.clear()

    # 1. Run test_fmoe (warmup + 20 cuda.Event timed iters; reports `us`).
    t0 = time.perf_counter()
    ret = test_mod.test_fmoe(
        dtype=dtypes.bf16,
        token=token,
        model_dim=model_dim,
        inter_dim=inter_dim,
        E=expert,
        topk=topk,
        actType=act,
        qType=aiter.QuantType.per_1x32,
        AQDType=aq,
        WQDType=wq,
        use_g1u1=True,
        doweight_stage1=False,
        strict_accuracy=False,
        hidden_pad=0,
        intermediate_pad=0,
        preshuffle=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"[CASE] test_fmoe wall={elapsed:.1f}s ret={ret}", flush=True)

    if not _last_s1_args or not _last_s2_args:
        print(
            f"[CASE][SKIP] {tag} M={token}: stage1/stage2 capture empty"
            f"  s1_calls={len(_last_s1_args)} s2_calls={len(_last_s2_args)}",
            flush=True,
        )
        return None

    # 2. Read the captured args -- these are the *exact* tensors stage1
    #    / stage2 were called with on the final fused_moe iteration.
    s1_args, s1_kwargs = _last_s1_args[0]
    s2_args, s2_kwargs = _last_s2_args[0]

    # 3. Print the tile / block info the patched stage1/stage2 received,
    #    so the md table can carry per-config tile metadata.
    def _shape(t):
        if isinstance(t, torch.Tensor):
            return f"{tuple(t.shape)}/{t.dtype}"
        return repr(t)

    s1_meta = {
        "in_dtype": s1_kwargs.get("in_dtype"),
        "out_dtype_str": s1_kwargs.get("out_dtype_str"),
        "tile_n": s1_kwargs.get("tile_n"),
        "tile_k": s1_kwargs.get("tile_k"),
        "block_m": s1_kwargs.get("block_m"),
        "activation": str(s1_kwargs.get("activation")).split(".")[-1],
        "num_buffers": _NUM_BUFFERS_OVERRIDE,
    }
    s2_meta = {
        "in_dtype": s2_kwargs.get("in_dtype"),
        "out_dtype_str": s2_kwargs.get("out_dtype_str"),
        "tile_n": s2_kwargs.get("tile_n"),
        "tile_k": s2_kwargs.get("tile_k"),
        "block_m": s2_kwargs.get("block_m"),
        "num_buffers": _NUM_BUFFERS_OVERRIDE,
    }

    # block_m / activation may live in positional args. Fall back to the
    # _gfx1250_moe_stage1 signature defaults if missing.
    if s1_meta["block_m"] is None and len(s1_args) >= 9:
        s1_meta["block_m"] = s1_kwargs.get("block_m", None)
    print(f"[TILE] stage1 meta={s1_meta}", flush=True)
    print(f"[TILE] stage2 meta={s2_meta}", flush=True)
    print(
        f"[TILE] stage1 a={_shape(s1_args[0])} w1={_shape(s1_args[1])} "
        f"w2={_shape(s1_args[2])} out={_shape(s1_args[6])}",
        flush=True,
    )
    print(
        f"[TILE] stage2 a={_shape(s2_args[0])} w1={_shape(s2_args[1])} "
        f"w2={_shape(s2_args[2])} out={_shape(s2_args[6])}",
        flush=True,
    )

    # 4. Run stage1 alone under run_perftest.
    # The patched module-level functions still capture into the store --
    # we want the raw kernel, so call _orig_s1 directly.
    n_warm = int(os.environ.get("AITER_MOE_WARMUP", "5"))
    n_iter = int(os.environ.get("AITER_MOE_ITERS", "20"))
    s1_data, s1_avg_us = run_perftest(
        _orig_s1,
        *s1_args, **s1_kwargs,
        num_iters=n_iter,
        num_warmup=n_warm,
        testGraph=False,
        num_rotate_args=1,
    )
    s2_data, s2_avg_us = run_perftest(
        _orig_s2,
        *s2_args, **s2_kwargs,
        num_iters=n_iter,
        num_warmup=n_warm,
        testGraph=False,
        num_rotate_args=1,
    )

    # 5. Direct cuda.Event timing for stage1 / stage2.  ``run_perftest`` uses
    # ``torch.profiler`` to derive its return value, which on AMD HIP adds
    # per-kernel measurement overhead (typically 3-5x slowdown for short
    # kernels) and also folds in ``out.zero_()`` / metadata kernels into
    # ``avg us/iter``.  Direct cuda.Event records around the kernel give
    # the on-GPU kernel-only wall time -- this is the number that lines up
    # with the fused ``cuda.Event`` measurement and with the legacy
    # /tmp/aiter_bench/stage_bd_*.json results.
    def _cudaevent_time(fn, args, kwargs, warmup, iters):
        for _ in range(warmup):
            fn(*args, **kwargs)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            fn(*args, **kwargs)
            ends[i].record()
        torch.cuda.synchronize()
        lats = sorted(starts[i].elapsed_time(ends[i]) * 1000.0 for i in range(iters))
        return {
            "median": lats[len(lats) // 2],
            "mean":   sum(lats) / len(lats),
            "min":    lats[0],
            "max":    lats[-1],
        }

    s1_cuda = _cudaevent_time(_orig_s1, s1_args, s1_kwargs, n_warm, n_iter)
    s2_cuda = _cudaevent_time(_orig_s2, s2_args, s2_kwargs, n_warm, n_iter)

    out = {
        "tag": tag,
        "token": token,
        "model_dim": model_dim,
        "inter_dim": inter_dim,
        "expert": expert,
        "topk": topk,
        "fused_us": ret.get("us"),
        # run_perftest's profiler-based "avg" (HIP profiler overhead inflates
        # this 3-5x for short kernels; kept for completeness).
        "stage1_us_perftest": s1_avg_us,
        "stage2_us_perftest": s2_avg_us,
        # Pure cuda.Event timing of the same args (this is what lines up
        # with the fused cuda.Event measurement).
        "stage1_cuda": s1_cuda,
        "stage2_cuda": s2_cuda,
        "stage1_meta": s1_meta,
        "stage2_meta": s2_meta,
    }
    print(
        f"[PERFTEST] {tag} M={token}: "
        f"fused={out['fused_us']:.2f}us | "
        f"stage1(run_perftest)={s1_avg_us:.2f}us  "
        f"stage2(run_perftest)={s2_avg_us:.2f}us | "
        f"stage1(cuda.Event median)={s1_cuda['median']:.2f}us  "
        f"stage2(cuda.Event median)={s2_cuda['median']:.2f}us | "
        f"sum_cuda={s1_cuda['median'] + s2_cuda['median']:.2f}us  "
        f"non-GEMM={out['fused_us'] - (s1_cuda['median'] + s2_cuda['median']):.2f}us",
        flush=True,
    )
    return out


CASES = {
    "DSV3_TP1_a4w4": (7168, 2048, 256, 8, dtypes.fp4x2, dtypes.fp4x2,
                       aiter.ActivationType.Silu),
    "DSV3_TP4_a4w4": (7168,  512, 256, 8, dtypes.fp4x2, dtypes.fp4x2,
                       aiter.ActivationType.Silu),
    "DSV3_TP8_a4w4": (7168,  256, 256, 8, dtypes.fp4x2, dtypes.fp4x2,
                       aiter.ActivationType.Silu),
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=",".join(CASES.keys()))
    ap.add_argument("--tokens", default="1,64")
    ap.add_argument("--out", required=True,
                    help="path for the per-run JSON dump")
    args = ap.parse_args()

    selected = [s.strip() for s in args.cases.split(",") if s.strip()]
    tokens = [int(x) for x in args.tokens.split(",")]

    results = []
    for tag in selected:
        if tag not in CASES:
            print(f"[CASE][SKIP] unknown tag {tag!r}", flush=True)
            continue
        model_dim, inter_dim, expert, topk, aq, wq, act = CASES[tag]
        for tk in tokens:
            try:
                r = run_case(tag, tk, model_dim, inter_dim, expert, topk,
                             aq, wq, act)
                if r is not None:
                    results.append(r)
            except Exception as ex:
                import traceback
                print(f"[CASE][ERROR] {tag} M={tk}: {ex}", flush=True)
                traceback.print_exc()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[DONE] wrote {args.out} ({len(results)} rows)", flush=True)
