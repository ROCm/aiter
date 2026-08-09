#!/usr/bin/env python3
"""Median-of-N perf probe for a migrated FlyDSL kernel.

Run this in two separate processes (old vs new source, via git worktree) and
compare medians. Interleaved A/B in one process is unreliable on a shared GPU.

Usage: python perf_ab.py <kernel> [iters]
"""
import statistics
import sys
import time

import torch


def _bench(fn, iters, reps=7):
    """Median of `reps` batched timings, each averaging `iters` launches.

    Timed with CUDA events around a batch so per-iteration host overhead and
    sync latency stay out of the measurement -- and everything host-side must
    already be hoisted out of `fn`, or you measure Python, not the kernel.
    """
    for _ in range(20):  # warmup, JIT, and clock ramp
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        beg, end = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize()
        beg.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        out.append(beg.elapsed_time(end) * 1e3 / iters)  # us/launch
    return out


def causal_conv1d(iters):
    sys.path.insert(0, "op_tests")
    from test_causal_conv1d_prefill_split_qkv import (  # noqa: E402
        K_DIM,
        V_DIM,
        _call_backend,
        make_inputs,
        qsl_to,
    )

    x, w, b, cs, ci, hi, qsl = make_inputs((1, 512, 1024, 2048), with_initial_state=True)
    # Hoist ALL host-side work: the clone is a device copy and .tolist() forces a
    # GPU->CPU sync, so leaving them in the loop measures neither kernel nor backend.
    qsl_d = qsl_to(qsl)
    seq_lens = qsl_d.diff().tolist()
    cs_work = cs.clone()
    kw = dict(
        x=x, weight=w, bias=b, conv_states=cs_work, query_start_loc=qsl_d,
        cache_indices=ci, has_initial_state=hi, k_dim=K_DIM, v_dim=V_DIM,
        seq_lens_cpu=seq_lens, activation="silu",
    )

    def go():
        _call_backend("flydsl", **kw)

    return _bench(go, iters)


def moe_sorting(iters):
    from aiter.ops.flydsl.moe_sorting import moe_sorting_flydsl  # noqa: E402

    tokens, E, topk, model_dim = 4096, 32, 5, 4096
    ids = torch.randint(0, E, (tokens, topk), dtype=torch.int32, device="cuda")
    wts = torch.rand((tokens, topk), dtype=torch.float32, device="cuda")

    def go():
        moe_sorting_flydsl(ids, wts, E, model_dim, torch.bfloat16)

    return _bench(go, iters)


KERNELS = {"causal_conv1d": causal_conv1d, "moe_sorting": moe_sorting}

if __name__ == "__main__":
    name = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    ts = KERNELS[name](iters)
    print(
        f"{name}: median={statistics.median(ts):.2f}us "
        f"min={min(ts):.2f} max={max(ts):.2f} reps={len(ts)}"
    )
