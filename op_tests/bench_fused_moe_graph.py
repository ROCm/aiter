"""True end-to-end fused_moe latency via CUDA graph replay + cuda.Event.

The stock bench times the EAGER fused_moe with per-iter torch.cuda.empty_cache()
(host dispatch + alloc churn -> ~975us), or sums per-kernel profiler device-time
(~470us, excludes launch gaps). Neither is the deployed latency.

This captures ONE fused_moe call into a CUDA graph and times N replays with a
single cuda.Event pair (no empty_cache, no per-iter sync) -> the real graphed
end-to-end. Drives the same config via test's run_moe by monkeypatching
run_perftest to capture the timed callable.

Run:
  python op_tests/bench_fused_moe_graph.py            # random weights
  python op_tests/bench_fused_moe_graph.py --half-init
"""

import argparse
import os

os.environ.setdefault("AITER_USE_GROUPED_GEMM", "1")
os.environ.setdefault("AITER_FORCE_GFX1250", "1")
os.environ.setdefault("WEIGHT_SCALE_OP_SEL", "1")
os.environ.setdefault("AITER_GROUPED_GEMM_NAIVE", "0")
os.environ.setdefault("AITER_MOE_EXPERT_BALANCE", "true")
os.environ.setdefault("AITER_GROUPED_DEBUG", "0")
os.environ.setdefault("FLYDSL_DUMP_IR", "0")
os.environ.setdefault("AITER_GROUPED_GEMM_AS_PROLOGUE", "1")
os.environ.setdefault("AITER_GROUPED_GEMM_WAVE_SPECIALIZED", "1")
# Avoid the stock cuda.Event+empty_cache path firing inside perftest.
os.environ["AITER_LOG_MORE"] = "0"

import sys

import torch

import aiter.test_common as tc

_ITERS = 200
_WARMUP = 20


def _graph_time(func):
    # Warmup (also lets the allocator settle) on a side stream for capture.
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(_WARMUP):
            func()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = func()

    for _ in range(_WARMUP):
        g.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(_ITERS):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    us = start.elapsed_time(end) * 1e3 / _ITERS
    return out, us


def _patched_run_perftest(func, *a, num_warmup=2, num_iters=101, **kw):
    # Ignore stock timing; graph-capture the callable and time replays.
    return _graph_time(func)


tc.run_perftest = _patched_run_perftest

sys.path.insert(0, os.path.dirname(__file__))
import test_flydsl_grouped_gemm_gfx1250 as T  # noqa: E402
from aiter import ActivationType  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--model-dim", type=int, default=4096)
    ap.add_argument("--inter-dim", type=int, default=2048)
    ap.add_argument("--half-init", action="store_true")
    args = ap.parse_args()

    m = T.run_moe(
        "a4w4",
        experts=args.experts,
        tokens=args.tokens,
        topk=args.topk,
        model_dim=args.model_dim,
        inter_dim=args.inter_dim,
        layout="gugu",
        activation=ActivationType.Silu,
        bench=True,
        raise_on_fail=False,
        warmup=_WARMUP,
        iters=_ITERS,
        half_init=args.half_init,
    )
    init = "const(0.5)" if args.half_init else "random"
    print(
        f"[fused_moe GRAPH e2e] init={init} "
        f"e{args.experts} tk{args.topk} tok{args.tokens} "
        f"md{args.model_dim} id{args.inter_dim}: "
        f"{m['us']:.2f} us/iter  (cuda graph replay, {_ITERS} iters)",
        flush=True,
    )


if __name__ == "__main__":
    main()
