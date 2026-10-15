# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL two-launch delayed mHC seam with RMSNorm (gfx942) vs fp32 torch.

Tables:
  test_mhc_seam          flydsl (policy config) and the Triton seam against the fp32
                         reference, plus a CUDA-graph replay check of the flydsl launches.
                         "graph us" is the replayed-graph latency per call (both launches
                         and the gap between them); "kernel us" the profiler's summed
                         kernel time of eager calls.
  test_mhc_seam_config   the tuning sweep, post mode: one row per legal knob set of either
                         NORM path, with the nt hints, timed as graph replay; then the
                         fastest set per (T, NORM).
  test_mhc_seam_norm     the policy config with each NORM path forced, post mode (the one
                         the separate path serves): where the RMSNorm runs faster, in the
                         finish pass or as aiter.rms_norm after it.
  test_mhc_seam_streams  two seams on two streams at once, sharing one weight, either path.
  test_mhc_seam_knobs    builds the policy does not reach at these sizes, forced.
  test_mhc_seam_weight_update  a captured graph after an in-place update of fn, per NORM.
  test_mhc_seam_errors   invalid inputs raise ValueError / TypeError.

The reference, inputs and output checks are the Mega-mHC test's
(``op_tests/test_flydsl_mega_mhc.py``): both ops implement the same bf16 seam contract.
"""

import argparse
import functools
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


def _splits_error(T, H, splits):
    """The validator's complaint about this SPLITS, or None if it takes it.

    Asked of check_config rather than re-derived here, so the rule stays in one place.
    A caller that passes a config gets BLOCK_M and CHUNK_C from MHC_SEAM_DEFAULTS,
    which is what both callers below do.
    """
    from aiter.jit.utils.chip_info import get_lds_capacity_bytes
    from aiter.ops.flydsl.kernels import mhc_seam
    from aiter.ops.flydsl.mhc_seam_kernels import MHC_SEAM_DEFAULTS

    try:
        mhc_seam.check_config(
            T,
            H,
            4,
            MHC_SEAM_DEFAULTS["BLOCK_M"],
            MHC_SEAM_DEFAULTS["CHUNK_C"],
            splits,
            lds_bytes=get_lds_capacity_bytes(get_gfx()),
        )
    except ValueError as e:
        return str(e)
    return None


def _smallest_illegal_splits(T, H):
    return next(sp for sp in range(2, H + 1) if _splits_error(T, H, sp))


def _with_fn(op, fn):
    """``op`` with the weight bound outside its arguments. run_perftest's rotation deep-copies
    every argument, and a copied ``fn`` makes the seam repack it on every call, which
    a model's one weight never does."""

    def call(residual, *rest, **kw):
        return op(residual, fn, *rest, **kw)

    return call


def _clone_inputs(args, kw):
    def clone(v):
        return v.clone() if isinstance(v, torch.Tensor) else v

    return tuple(clone(a) for a in args), {k: clone(v) for k, v in kw.items()}


def _graph_us(op, args, kw, T, replays=10):
    """Per-call latency as an inference graph sees it: `calls` seams captured into one CUDA
    graph, replayed and timed with events. Unlike the profiler's summed kernel time this
    includes the idle gap between the two launches; host time is excluded either way."""
    from aiter.ops.flydsl.kernels.kernels_common import ceildiv

    per_set = sum(
        v.nbytes for v in (*args, *kw.values()) if isinstance(v, torch.Tensor)
    )
    n_sets = ceildiv(_ROTATE_BYTES, per_set)
    # Whole passes over the sets, so the wrap from one replay to the next revisits none early.
    calls = n_sets * ceildiv(20 if T <= 4096 else 4, n_sets)
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


@functools.lru_cache(maxsize=None)
def _warn_triton_skip(span, H, T):
    aiter.logger.warning(
        "triton seam needs H %% %d == 0; skipping it at H=%d, T=%d", span, H, T
    )


def _seam_ops(T, H):
    """The seams to time against each other at this shape.

    The triton seam's tuned config fixes NUM_KSPLIT * TILE_K and its k-loop has no column
    mask, so it only launches where H divides that product: every T at H = 5120, not the
    small-T configs at 4096 or 7168. Skip it there rather than fail the sweep.
    """
    from aiter.ops.flydsl import flydsl_mhc_seam
    from aiter.ops.triton.fusions.mhc_fused_post_pre_delayed_rmsnorm import (
        get_mhc_fused_post_pre_delayed_rmsnorm_config,
        mhc_fused_post_pre_delayed_rmsnorm,
    )

    ops = {"flydsl": flydsl_mhc_seam}
    if not T:  # the triton seam returns before it reads a config
        ops["triton_seam"] = mhc_fused_post_pre_delayed_rmsnorm
        return ops

    cfg = get_mhc_fused_post_pre_delayed_rmsnorm_config(T)
    span = cfg["NUM_KSPLIT"] * cfg["TILE_K"]
    if H % span:
        _warn_triton_skip(span, H, T)  # once per shape, not once per mode
        return ops

    ops["triton_seam"] = mhc_fused_post_pre_delayed_rmsnorm
    return ops


@benchmark()
def test_mhc_seam(T, H, mode):
    from aiter.ops.flydsl import flydsl_mhc_seam

    args, kw, ref = _call_kwargs(mode, T, H)
    ops = _seam_ops(T, H)
    ret = {"gfx": get_gfx()}
    for name, op in ops.items():
        if not T:  # empty-batch edge case: the call must return, nothing to time
            out = op(*args, **kw)
            shapes = [tuple(o.shape) for o in out]
            want = [(0, 4, H), (0, 4, 1), (0, 4, 4), (0, H), (0, 4)]
            ret[f"{name} err"] = 0.0 if shapes == want else 1.0
            if shapes != want:
                mega_ref._FAILURES.append(f"{name} T=0 {mode}: shapes {shapes}")
            continue

        # graph us: replayed CUDA graph, both launches and the gap between them;
        # kernel us: the profiler's summed kernel time of eager calls
        ret[f"{name} graph us"] = _graph_us(op, args, kw, T)
        out, us = run_perftest(_with_fn(op, args[1]), args[0], *args[2:], **kw)
        ret[f"{name} kernel us"] = us
        ret[f"{name} err"] = check_outputs(name, out, ref, "bf16", mode)
    if T:
        ret["flydsl graph err"] = _graph_err(
            lambda: flydsl_mhc_seam(*args, **kw), ref, mode
        )
    if "triton_seam graph us" in ret and T:
        ret["graph speedup"] = ret["triton_seam graph us"] / ret["flydsl graph us"]
    return ret


@benchmark()
def test_mhc_seam_config(T, H, norm, block_m, chunk_c, splits, cache_nt, cache_nt_x):
    from aiter.jit.utils.chip_info import get_cu_num
    from aiter.ops.flydsl import flydsl_mhc_seam
    from aiter.ops.flydsl.mhc_seam_kernels import get_mhc_seam_config

    cfg = {"NORM": norm, "BLOCK_M": block_m, "CHUNK_C": chunk_c, "SPLITS": splits}
    cfg.update(CACHE_NT=cache_nt, CACHE_NT_X=cache_nt_x)
    cfg = {k: v for k, v in cfg.items() if v is not None}
    args, kw, ref = _call_kwargs("post", T, H)
    op = functools.partial(flydsl_mhc_seam, config=cfg)
    us = _graph_us(op, args, kw, T)
    out = op(*args, **kw)
    policy = get_mhc_seam_config(T, H, get_gfx(), get_cu_num())
    return {
        "gfx": get_gfx(),
        "graph us": us,
        "TB/s": _traffic(T, H, "post") / us / 1e6,
        "policy": all(policy[k] == v for k, v in cfg.items()),
        "err": check_outputs(f"config {cfg}", out, ref, "bf16", "post"),
    }


@benchmark()
def test_mhc_seam_norm(T, H, mode, norms):
    from aiter.jit.utils.chip_info import get_cu_num
    from aiter.ops.flydsl import flydsl_mhc_seam
    from aiter.ops.flydsl.mhc_seam_kernels import get_mhc_seam_config

    args, kw, ref = _call_kwargs(mode, T, H)
    policy = get_mhc_seam_config(T, H, get_gfx(), get_cu_num())
    ret = {"gfx": get_gfx(), "policy": policy["NORM"]}
    for norm in norms:
        # each path at its own policy config; fused below the separate threshold
        cfg = get_mhc_seam_config(T, H, get_gfx(), get_cu_num(), norm=norm)
        op = functools.partial(flydsl_mhc_seam, config=dict(cfg, NORM=norm))
        ret[f"{norm} graph us"] = _graph_us(op, args, kw, T)
        out = op(*args, **kw)
        ret[f"{norm} err"] = check_outputs(f"NORM={norm}", out, ref, "bf16", mode)
        ret[f"{norm} graph err"] = _graph_err(
            functools.partial(op, *args, **kw), ref, mode
        )
    if {"fused", "separate"} <= set(norms):
        ret["separate speedup"] = ret["fused graph us"] / ret["separate graph us"]
    return ret


def _shared_fn_cases(T, H):
    """Two post-mode seams with their own activations and gates and one weight, as two
    streams of one model have: the first call packs it for both."""
    first = _call_kwargs("post", T, H, seed=1)
    fn = first[0][1]
    residual, y, _, hc_scale, hc_base, post, comb, pre, w = mega_ref.make_inputs(
        T, H, 2
    )
    kw = {"norm_weight": w, "norm_eps": mega_ref.NORM_EPS}
    kw.update(sublayer_out=y, post_layer_mix=post, comb_res_mix=comb)
    ref = mega_ref.run_torch(
        residual, fn, hc_scale, hc_base, *mega_ref.ARGS, pre_mix=pre, **kw
    )
    kw.update(residual_out=torch.empty_like(residual), pre_mix=pre)
    return [first, ((residual, fn, hc_scale, hc_base, *mega_ref.ARGS), kw, ref)]


def test_mhc_seam_streams(T, H, splits, norm):
    """Two seams on two streams at once; each must match its own reference."""
    from aiter.ops.flydsl import flydsl_mhc_seam

    cfg = {"SPLITS": splits, "NORM": norm}
    cases = _shared_fn_cases(T, H)
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
            f"stream {i} {norm} splits={splits}",
            outs[i],
            cases[i][2],
            "bf16",
            "post",
        )
        for i in range(2)
    ]
    return {"gfx": get_gfx(), "T": T, "norm": norm, "splits": splits, "err": max(errs)}


def test_mhc_seam_knobs(H):
    """Builds the policy never reaches at the test's sizes, forced; each must match the
    reference."""
    from aiter.ops.flydsl import flydsl_mhc_seam

    rows = []
    for mode, cfg in (
        ("no_post", {"STAGE_NT": True}),
        ("post", {"NORM": "fused", "STAGE_NT": True, "GATE_STEP": False}),
    ):
        args, kw, ref = _call_kwargs(mode, 128, H)
        out = flydsl_mhc_seam(*args, config=cfg, **kw)
        err = check_outputs(f"forced {mode} {cfg}", out, ref, "bf16", mode)
        rows.append({"gfx": get_gfx(), "mode": mode, "config": cfg, "err": err})
    return rows


def test_mhc_seam_weight_update(H, norm):
    """A captured graph follows an in-place update of fn after one eager call, and a write
    through fn.data once the caller repacks fn_packed in place."""
    from aiter.ops.flydsl import flydsl_mhc_seam, flydsl_mhc_seam_pack_fn

    T = 256
    args, kw, _ = _call_kwargs("post", T, H)
    cfg = {"NORM": norm}
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        flydsl_mhc_seam(*args, config=cfg, **kw)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        out = flydsl_mhc_seam(*args, config=cfg, **kw)
    args[1].mul_(2)
    flydsl_mhc_seam(*args, config=cfg, **kw)  # the eager call that repacks
    g.replay()
    torch.cuda.synchronize()

    # Against the reference, not an eager call: a stale pack would also make the eager
    # call stale, and the two would agree on the old fn.
    ref_kw = {k: v for k, v in kw.items() if k not in ("residual_out", "pre_mix")}
    ref = mega_ref.run_torch(*args, pre_mix=kw["pre_mix"], **ref_kw)
    mul_err = check_outputs(
        f"weight update in place ({norm})", out, ref, "bf16", "post"
    )

    fn = args[1]
    packed = flydsl_mhc_seam_pack_fn(fn)
    with torch.cuda.stream(s):
        flydsl_mhc_seam(*args, config=cfg, fn_packed=packed, **kw)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        out = flydsl_mhc_seam(*args, config=cfg, fn_packed=packed, **kw)
    fn.data.copy_(fn * 0.5)
    packed.copy_(flydsl_mhc_seam_pack_fn(fn))
    g.replay()
    torch.cuda.synchronize()
    ref = mega_ref.run_torch(*args, pre_mix=kw["pre_mix"], **ref_kw)
    err = check_outputs(f"weight update via fn.data ({norm})", out, ref, "bf16", "post")
    return {
        "gfx": get_gfx(),
        "T": T,
        "norm": norm,
        "in-place replay err": mul_err,
        "fn.data replay err": err,
    }


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
        "unknown norm": (with_kw(config={"NORM": "post"}), ValueError),
        "illegal chunk": (with_kw(config={"CHUNK_C": 96}), ValueError),
        # from the validator, not re-derived: a fixed 7 is legal at H = 7168.
        "splits not dividing": (
            with_kw(config={"SPLITS": _smallest_illegal_splits(T, H)}),
            ValueError,
        ),
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


def _stream_rows(tokens, hidden, stream_splits):
    """Two-stream rows for each (T, H, SPLITS) the policy's chunk count can take.

    A legal SPLITS divides the chunk count, which follows H: the default 10 fits
    H = 5120 (20 chunks) but not H = 4096 (16), so a split this shape cannot take is
    dropped rather than raised on.
    """
    rows = []
    for T, H, sp in itertools.product(
        [t for t in tokens if 0 < t <= 4096][-2:], hidden, stream_splits
    ):
        why = _splits_error(T, H, sp)
        if why:
            aiter.logger.warning("stream splits %d at H=%d, T=%d: %s", sp, H, T, why)
            continue

        rows += [
            test_mhc_seam_streams(T, H, sp, norm) for norm in ("fused", "separate")
        ]
    return rows


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
    # knob sweep axes (test_mhc_seam_config, post mode); empty --block_m skips the sweep
    parser.add_argument("--block_m", type=int, nargs="*", default=[32])
    parser.add_argument("--chunk_c", type=int, nargs="*", default=[64])
    parser.add_argument("--splits", type=int, nargs="*", default=[16, 40])
    parser.add_argument(
        "--sweep_norm", nargs="*", default=["fused"], choices=["fused", "separate"]
    )
    # 0 / 1; the default sweeps neither and leaves the hint to the wrapper's default
    parser.add_argument("--cache_nt", type=int, nargs="*", default=[])
    parser.add_argument("--cache_nt_x", type=int, nargs="*", default=[])
    parser.add_argument("--stream_splits", type=int, nargs="*", default=[1, 10])
    # NORM paths timed side by side (test_mhc_seam_norm); empty skips the table
    parser.add_argument("--norm", nargs="*", default=[], choices=["fused", "separate"])
    args = parser.parse_args()

    rows = [
        test_mhc_seam(T, H, mode)
        for T, H, mode in itertools.product(args.tokens, args.hidden, args.mode)
    ]
    aiter.logger.info(
        "mhc seam summary (markdown):\n%s", pd.DataFrame(rows).to_markdown(index=False)
    )

    if args.norm:
        rows = [
            test_mhc_seam_norm(T, H, "post", args.norm)
            for T, H in itertools.product(
                [t for t in args.tokens if t > 0], args.hidden
            )
        ]
        aiter.logger.info(
            "mhc seam NORM paths (markdown):\n%s",
            pd.DataFrame(rows).to_markdown(index=False),
        )

    if args.block_m:
        from aiter.jit.utils.chip_info import get_lds_capacity_bytes
        from aiter.ops.flydsl.kernels import mhc_seam

        lds = get_lds_capacity_bytes(get_gfx())
        flags = lambda v: [bool(x) for x in v] or [None]
        rows = []
        for T, H, norm, bm, cc, sp, nt, ntx in itertools.product(
            [t for t in args.tokens if t > 0],
            args.hidden,
            args.sweep_norm,
            args.block_m,
            args.chunk_c,
            args.splits,
            flags(args.cache_nt),
            flags(args.cache_nt_x),
        ):
            try:
                mhc_seam.check_config(T, H, 4, bm, cc, sp, lds_bytes=lds)
            except ValueError:
                continue
            rows.append(test_mhc_seam_config(T, H, norm, bm, cc, sp, nt, ntx))
        sweep = pd.DataFrame(rows)
        aiter.logger.info(
            "mhc seam config sweep (markdown):\n%s", sweep.to_markdown(index=False)
        )
        # fastest config per (T, H, NORM): the rows a policy table is made from
        if sweep.empty:
            best = sweep
        else:
            best = sweep.loc[sweep.groupby(["T", "H", "norm"])["graph us"].idxmin()]
        aiter.logger.info(
            "mhc seam config sweep, best per T (markdown):\n%s",
            best.to_markdown(index=False),
        )

    rows = _stream_rows(args.tokens, args.hidden, args.stream_splits)
    if rows:
        aiter.logger.info(
            "mhc seam two-stream check (markdown):\n%s",
            pd.DataFrame(rows).to_markdown(index=False),
        )
    else:
        aiter.logger.warning(
            "no stream splits legal at these hidden sizes; two-stream check skipped"
        )

    rows = [r for H in args.hidden for r in test_mhc_seam_knobs(H)]
    aiter.logger.info(
        "mhc seam forced-knob check (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )

    rows = [
        test_mhc_seam_weight_update(H, norm)
        for H in args.hidden
        for norm in ("fused", "separate")
    ]
    aiter.logger.info(
        "mhc seam weight-update check (markdown):\n%s",
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
