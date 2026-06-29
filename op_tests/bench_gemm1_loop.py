"""Isolated loop microbenchmark for the grouped MoE **gemm1** kernel (gfx1250).

Instead of timing the whole fused_moe chain (scatter / quant / gemm1 / gemm2 /
gather), this captures ONLY the compiled stage1 (gemm1) launch + its already
built arguments from a single setup call, then replays just that one kernel in
a CUDA-graph loop timed with device events.

Config defaults to the gugu / silu / a4w4 case used in test_rand_vs_const.sh:
  experts=256, topk=6, tokens=64, model_dim=4096, inter_dim=2048.

Run:
  python op_tests/bench_gemm1_loop.py                 # random weights
  python op_tests/bench_gemm1_loop.py --half-init     # const (0.5) weights
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
# Match --wst (wave-specialized TDM) of the chained bench.
os.environ.setdefault("AITER_GROUPED_GEMM_WAVE_SPECIALIZED", "1")
# Skip the cuda.Event+empty_cache path in perftest; use profiler trace timing.
os.environ["AITER_LOG_MORE"] = "0"

import sys

import torch

from aiter import ActivationType
import aiter.ops.flydsl.kernels.moe_grouped_gemm_mxscale_gfx1250 as _mod

_cap = {}


def _make_patch(stage, orig):
    def _capturing_compile(**kw):
        launch = orig(**kw)

        def _recording_launch(*a, **k):
            _cap[stage] = {"launch": launch, "args": a, "kwargs": k}
            return launch(*a, **k)

        return _recording_launch

    return _capturing_compile


_mod.compile_moe_grouped_gemm1_mxfp4_masked = _make_patch(
    1, _mod.compile_moe_grouped_gemm1_mxfp4_masked
)
_mod.compile_moe_grouped_gemm2_mxfp4_masked = _make_patch(
    2, _mod.compile_moe_grouped_gemm2_mxfp4_masked
)

# Capture the low-level dynamic_per_group_scaled_quant custom op (pre-allocated
# buffers, pure kernel launch -> capturable, no per-iter alloc/host overhead).
# The LAST call is the stage2 (a2) quant after gemm1.
import aiter.ops.quant as _q

_orig_dpgsq = _q.dynamic_per_group_scaled_quant


def _rec_dpgsq(*a, **k):
    out = _orig_dpgsq(*a, **k)
    _cap["a2quant"] = (_orig_dpgsq, a, k)
    return out


_q.dynamic_per_group_scaled_quant = _rec_dpgsq

sys.path.insert(0, os.path.dirname(__file__))
import test_flydsl_grouped_gemm_gfx1250 as T  # noqa: E402


def _stage_bytes(stage, experts, model_dim, inter_dim):
    # fp4 weight packed 2/byte + e8m0 scale (1 byte / 32 elems).
    if stage == 1:  # gate+up: [2*inter, model] per expert
        rows, k = 2 * inter_dim, model_dim
    else:  # down: [model, inter] per expert
        rows, k = model_dim, inter_dim
    w = experts * rows * k // 2
    s = experts * rows * (k // 32)
    return w + s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--model-dim", type=int, default=4096)
    ap.add_argument("--inter-dim", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--half-init", action="store_true")
    ap.add_argument("--stage", type=int, default=1, choices=(1, 2))
    ap.add_argument(
        "--launch-overhead",
        action="store_true",
        help="compare per-launch HOST time: @flyc.jit dispatch vs flyc.compile",
    )
    ap.add_argument(
        "--chain",
        action="store_true",
        help="loop gemm1 + a2-quant + gemm2 together (no scatter/gather/sort)",
    )
    args = ap.parse_args()

    _cap.clear()
    # One setup call builds all grouped inputs and runs fused_moe once; our
    # wrapper captures the gemm1 launch + concrete args during that call.
    T.run_moe(
        "a4w4",
        experts=args.experts,
        tokens=args.tokens,
        topk=args.topk,
        model_dim=args.model_dim,
        inter_dim=args.inter_dim,
        layout="gugu",
        activation=ActivationType.Silu,
        bench=False,
        raise_on_fail=False,
        half_init=args.half_init,
    )
    if args.chain:
        for need in (1, 2):
            if need not in _cap:
                raise RuntimeError(f"gemm{need} launch not captured")
        if "a2quant" not in _cap:
            raise RuntimeError("a2 quant call not captured")
        g1 = _cap[1]
        g2 = _cap[2]
        qfn, qa, qk = _cap["a2quant"]
        s = torch.cuda.current_stream()
        g1k = dict(g1["kwargs"]); g1k["stream"] = s
        g2k = dict(g2["kwargs"]); g2k["stream"] = s

        _do_quant = os.environ.get("BENCH_NO_QUANT") != "1"

        def step():
            g1["launch"](*g1["args"], **g1k)
            if _do_quant:
                qfn(*qa, **qk)
            g2["launch"](*g2["args"], **g2k)

        # run_perftest in cuda.event mode (now times the whole loop -> pipelined
        # throughput). Low-level quant op keeps host overhead tiny.
        from aiter.test_common import run_perftest

        _, us = run_perftest(
            step,
            num_warmup=args.warmup,
            num_iters=args.iters,
            use_cuda_event=True,
        )
        b1 = _stage_bytes(1, args.experts, args.model_dim, args.inter_dim)
        b2 = _stage_bytes(2, args.experts, args.model_dim, args.inter_dim)
        bw = (b1 + b2) / (us * 1e-6) / 1e12
        init = "const(0.5)" if args.half_init else "random"
        print(
            f"[chain gemm1+quant+gemm2 gugu silu] init={init} "
            f"e{args.experts} tk{args.topk} tok{args.tokens} "
            f"md{args.model_dim} id{args.inter_dim}: "
            f"{us:.2f} us/iter  weight+scale={(b1+b2)/1e9:.3f} GB  bw={bw:.2f} TB/s "
            f"(loop, {args.iters} iters, no scatter/gather)",
            flush=True,
        )
        return

    if args.stage not in _cap:
        raise RuntimeError(
            f"gemm{args.stage} launch was not captured (grouped path not taken?)"
        )

    launch = _cap[args.stage]["launch"]
    a = _cap[args.stage]["args"]
    k = dict(_cap[args.stage]["kwargs"])
    k["stream"] = torch.cuda.current_stream()

    if args.launch_overhead:
        import time as _time

        # Pure-CPU dispatch time: how long the Python launch call takes to
        # ENQUEUE (no GPU wait). Measured by submitting many launches without
        # syncing and dividing the host wall by count; valid as long as the GPU
        # queue does not throttle host submission for this many calls.
        N = args.iters
        for _ in range(args.warmup):
            launch(*a, **k)
        torch.cuda.synchronize()
        t0 = _time.perf_counter()
        for _ in range(N):
            launch(*a, **k)
        host_total = (_time.perf_counter() - t0) * 1e6  # us for N submits
        torch.cuda.synchronize()
        init = "const(0.5)" if args.half_init else "random"
        print(
            f"[launch-overhead gemm{args.stage}] init={init}: "
            f"@flyc.jit host dispatch ~= {host_total / N:.2f} us/launch "
            f"({N} submits). NOTE: grouped path calls this @flyc.jit launcher "
            f"directly (no flyc.compile / no graph).",
            flush=True,
        )
        return

    # Back-to-back launches of ONLY gemm1, timed with device events. The launches
    # queue up so GPU stays saturated and host launch overhead overlaps.
    for _ in range(args.warmup):
        launch(*a, **k)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        launch(*a, **k)
    end.record()
    torch.cuda.synchronize()

    us = start.elapsed_time(end) * 1e3 / args.iters
    nbytes = _stage_bytes(args.stage, args.experts, args.model_dim, args.inter_dim)
    bw = nbytes / (us * 1e-6) / 1e12  # TB/s
    init = "const(0.5)" if args.half_init else "random"
    print(
        f"[gemm{args.stage}-loop gugu silu] init={init} "
        f"e{args.experts} tk{args.topk} tok{args.tokens} "
        f"md{args.model_dim} id{args.inter_dim}: "
        f"{us:.2f} us/iter  weight+scale={nbytes/1e9:.3f} GB  bw={bw:.2f} TB/s "
        f"(launch loop, {args.iters} iters)",
        flush=True,
    )


if __name__ == "__main__":
    main()
