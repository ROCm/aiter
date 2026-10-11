# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL two-launch delayed mHC seam with RMSNorm (gfx942) vs fp32 torch.

Four tables:
  test_mhc_seam          flydsl (policy config) and the Triton seam against the fp32
                         reference, plus a CUDA-graph replay check of the flydsl launches.
                         "graph us" is the replayed-graph latency per call (both launches
                         and the gap between them); "kernel us" the profiler's summed
                         kernel time of eager calls.
  test_mhc_seam_config   one row per legal knob set (the tuning sweep).
  test_mhc_seam_streams  two seams on two streams at once.
  test_mhc_seam_errors   invalid inputs raise ValueError / TypeError.

The reference, inputs and output checks are the Mega-mHC test's
(``op_tests/test_flydsl_mega_mhc.py``): both ops implement the same bf16 seam contract.
"""

import argparse
import itertools

import pandas as pd
import test_flydsl_mega_mhc as mega_ref
import torch
from test_flydsl_mega_mhc import _call_kwargs, check_outputs

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942"]


def _traffic(T, H, mode):
    return mega_ref._traffic(T, H, mode, "bf16")


def _graph_err(fn, ref, mode):
    """Capture the two launches in a CUDA graph and replay it twice."""
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn()  # compile outside the capture
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        out = fn()
    errs = []
    for i in range(2):
        g.replay()
        torch.cuda.synchronize()
        errs.append(check_outputs(f"flydsl graph replay {i}", out, ref, "bf16", mode))
    return max(errs)


# Input sets the graph timing rotates through: enough that one pass over them overflows the
# 256 MB MALL, so a replay reads its inputs from HBM as an inference step would.
_ROTATE_BYTES = 512 * 2**20


def _clone_inputs(args, kw):
    def clone(v):
        return v.clone() if isinstance(v, torch.Tensor) else v

    return tuple(clone(a) for a in args), {k: clone(v) for k, v in kw.items()}


def _graph_us(op, args, kw, T, H, mode, replays=10):
    """Per-call latency as an inference graph sees it: `calls` seams captured into one CUDA
    graph, replayed and timed with events. Unlike the profiler's summed kernel time this
    includes the idle gap between the two launches; host time is excluded either way."""
    n_sets = max(1, min(16, -(-_ROTATE_BYTES // _traffic(T, H, mode))))
    calls = 20 if T <= 4096 else 4
    sets = [(args, kw)] + [_clone_inputs(args, kw) for _ in range(n_sets - 1)]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for a, k in sets:
            op(*a, **k)  # compile and warm up outside the capture
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for i in range(calls):
            a, k = sets[i % n_sets]
            op(*a, **k)
    g.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        g.replay()
    end.record()
    end.synchronize()
    us = start.elapsed_time(end) * 1000 / (replays * calls)
    del g, sets  # back to the caching allocator for the next shape
    return us


@benchmark()
def test_mhc_seam(T, H, mode):
    from aiter.ops.flydsl import flydsl_mhc_seam
    from aiter.ops.triton.fusions.mhc_fused_post_pre_delayed_rmsnorm import (
        mhc_fused_post_pre_delayed_rmsnorm,
    )

    args, kw, ref = _call_kwargs(mode, T, H)
    ops = {
        "flydsl": flydsl_mhc_seam,
        "triton_seam": mhc_fused_post_pre_delayed_rmsnorm,
    }
    ret = {"gfx": get_gfx()}
    for name, op in ops.items():
        if not T:  # empty-batch edge case: the call must return, nothing to time
            out = op(*args, **kw)
            ret[f"{name} err"] = 0.0 if out[3].shape == (0, H) else 1.0
            continue
        # graph us: replayed CUDA graph, both launches and the gap between them;
        # kernel us: the profiler's summed kernel time of eager calls
        ret[f"{name} graph us"] = _graph_us(op, args, kw, T, H, mode)
        out, us = run_perftest(op, *args, **kw)
        ret[f"{name} kernel us"] = us
        ret[f"{name} err"] = check_outputs(name, out, ref, "bf16", mode)
    if T:
        ret["graph speedup"] = ret["triton_seam graph us"] / ret["flydsl graph us"]
        ret["flydsl graph err"] = _graph_err(
            lambda: flydsl_mhc_seam(*args, **kw), ref, mode
        )
    return ret


@benchmark()
def test_mhc_seam_config(T, H, block_m, chunk_c, splits):
    from aiter.jit.utils.chip_info import get_cu_num
    from aiter.ops.flydsl import flydsl_mhc_seam
    from aiter.ops.flydsl.mhc_seam_kernels import get_mhc_seam_config

    cfg = {"BLOCK_M": block_m, "CHUNK_C": chunk_c, "SPLITS": splits}
    args, kw, ref = _call_kwargs("post", T, H)
    out, us = run_perftest(lambda: flydsl_mhc_seam(*args, config=cfg, **kw))
    policy = get_mhc_seam_config(T, H, get_gfx(), get_cu_num())
    return {
        "gfx": get_gfx(),
        "us": us,
        "TB/s": _traffic(T, H, "post") / us / 1e6,
        "policy": all(policy[k] == v for k, v in cfg.items()),
        "err": check_outputs(f"config {cfg}", out, ref, "bf16", "post"),
    }


def test_mhc_seam_streams(T, H, splits):
    """Two seams on two streams at once; each must match its own reference."""
    from aiter.ops.flydsl import flydsl_mhc_seam

    cfg = {"SPLITS": splits}
    cases = [_call_kwargs("post", T, H, seed=s) for s in (1, 2)]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    torch.cuda.synchronize()
    outs = [None, None]
    for _ in range(20):
        for i, (args, kw, _ref) in enumerate(cases):
            with torch.cuda.stream(streams[i]):
                outs[i] = flydsl_mhc_seam(*args, config=cfg, **kw)
    torch.cuda.synchronize()
    errs = [
        check_outputs(
            f"stream {i} splits={splits}", outs[i], cases[i][2], "bf16", "post"
        )
        for i in range(2)
    ]
    return {"gfx": get_gfx(), "T": T, "splits": splits, "err": max(errs)}


def test_mhc_seam_errors(H):
    """Each invalid call must raise the documented exception type."""
    from aiter.ops.flydsl import flydsl_mhc_seam

    T = 4
    args, kw, _ref = _call_kwargs("post", T, H)
    residual, fn = args[:2]

    def with_kw(**over):
        return lambda: flydsl_mhc_seam(*args, **dict(kw, **over))

    def with_args(i, value):
        a = list(args)
        a[i] = value
        return lambda: flydsl_mhc_seam(*a, **kw)

    cases = {
        "fp32 residual": (with_args(0, residual.float()), TypeError),
        "hc_mult 2": (with_args(0, residual[:, :2].contiguous()), ValueError),
        "fn shape": (with_args(1, fn[:, :-32].contiguous()), ValueError),
        "fn bf16": (with_args(1, fn.to(torch.bfloat16)), TypeError),
        "no norm_weight": (with_kw(norm_weight=None), ValueError),
        "post w/o comb": (with_kw(comb_res_mix=None), ValueError),
        "unknown knob": (with_kw(config={"TILE_K": 64}), ValueError),
        "illegal chunk": (with_kw(config={"CHUNK_C": 96}), ValueError),
        "splits not dividing": (with_kw(config={"SPLITS": 7}), ValueError),
        "chunk lanes not 2^k": (with_kw(config={"CHUNK_C": 320}), ValueError),
        "block_m above block": (with_kw(config={"BLOCK_M": 512}), ValueError),
    }
    bad = []
    for name, (call, exc) in cases.items():
        try:
            call()
        except exc:
            continue
        except Exception as e:  # noqa: BLE001 - reported as a failure below
            bad.append(f"{name}: {type(e).__name__}")
            continue
        bad.append(f"{name}: no error")
    if bad:
        mega_ref._FAILURES.append(f"errors: {bad}")
    return {"gfx": get_gfx(), "cases": len(cases), "failed": "; ".join(bad) or "-"}


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("flydsl mhc seam unsupported on %s; skipping", get_gfx())
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL two-launch mHC seam: correctness + perf",
    )
    parser.add_argument(
        "-t",
        "--tokens",
        type=int,
        nargs="*",
        default=[0, 1, 32, 128, 384, 1024, 4096, 8192, 16384],
    )
    parser.add_argument("--hidden", type=int, nargs="*", default=[5120])
    parser.add_argument(
        "--mode", nargs="*", default=["post", "no_post", "identity_pre"]
    )
    # knob sweep axes (test_mhc_seam_config); empty --block_m skips the sweep
    parser.add_argument("--block_m", type=int, nargs="*", default=[32])
    parser.add_argument("--chunk_c", type=int, nargs="*", default=[64])
    parser.add_argument("--splits", type=int, nargs="*", default=[16, 40])
    parser.add_argument("--stream_splits", type=int, nargs="*", default=[1, 10])
    args = parser.parse_args()

    rows = [
        test_mhc_seam(T, H, mode)
        for T, H, mode in itertools.product(args.tokens, args.hidden, args.mode)
    ]
    aiter.logger.info(
        "mhc seam summary (markdown):\n%s", pd.DataFrame(rows).to_markdown(index=False)
    )

    if args.block_m:
        from aiter.jit.utils.chip_info import get_lds_capacity_bytes
        from aiter.ops.flydsl.kernels.mhc_seam import check_config

        lds = get_lds_capacity_bytes(get_gfx())
        rows = []
        for T, H, bm, cc, sp in itertools.product(
            [t for t in args.tokens if t > 0],
            args.hidden,
            args.block_m,
            args.chunk_c,
            args.splits,
        ):
            try:
                check_config(T, H, 4, bm, cc, sp, lds_bytes=lds)
            except ValueError:
                continue
            rows.append(test_mhc_seam_config(T, H, bm, cc, sp))
        aiter.logger.info(
            "mhc seam config sweep (markdown):\n%s",
            pd.DataFrame(rows).to_markdown(index=False),
        )

    rows = [
        test_mhc_seam_streams(T, H, sp)
        for T, H, sp in itertools.product(
            [t for t in args.tokens if 0 < t <= 4096][-2:],
            args.hidden,
            args.stream_splits,
        )
    ]
    aiter.logger.info(
        "mhc seam two-stream check (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )

    rows = [test_mhc_seam_errors(H) for H in args.hidden]
    aiter.logger.info(
        "mhc seam invalid-input check (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )

    if mega_ref._FAILURES:
        raise AssertionError(
            f"{len(mega_ref._FAILURES)} mhc seam checks failed: {mega_ref._FAILURES[:8]}"
        )


if __name__ == "__main__":
    main()
