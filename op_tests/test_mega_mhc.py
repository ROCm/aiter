# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mega-MHC correctness checks and HIP / FlyDSL / Triton performance table.

Run directly for correctness and performance, or use --check-only for the
full reference, split-K reuse, graph and stream regression checks.
"""

import argparse
import itertools
import math
import os
from contextlib import contextmanager
from functools import cache

import pandas as pd
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.mega_mhc import _counter_state_ok, mega_mhc, prepare_mega_mhc
from aiter.test_common import benchmark, run_perftest

# rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult, sinkhorn_repeat
ARGS = (1e-6, 1e-6, 1e-6, 2.0, 20)
MODES = ("post", "no_post", "identity_pre", "no_post_identity")
BACKENDS = ("hip", "flydsl", "triton")


def ref_mhc_fused_post_pre_delayed_rmsnorm(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult,
    sinkhorn_repeat,
    pre_mix,
    sublayer_out=None,
    post_layer_mix=None,
    comb_res_mix=None,
    norm_weight=None,
    norm_eps=1e-6,
):
    """FP32 post-mix, delayed pre-mix and RMSNorm, with BF16 seam rounding."""
    t, n, _ = residual.shape
    if sublayer_out is not None:
        mixed = torch.einsum(
            "tij,tih->tjh", comb_res_mix.float().view(t, n, n), residual.float()
        )
        r = (
            mixed
            + post_layer_mix.float().view(t, n, 1) * sublayer_out.float().unsqueeze(1)
        ).to(residual.dtype)
    else:
        r = residual
    x = r.flatten(1).float()
    mixes = (x @ fn.t()) * torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_eps)
    pre = torch.sigmoid(mixes[:, :n] * hc_scale[0] + hc_base[:n]) + hc_pre_eps
    post = (
        torch.sigmoid(mixes[:, n : 2 * n] * hc_scale[1] + hc_base[n : 2 * n])
        * hc_post_mult
    )
    comb = mixes[:, 2 * n :].view(-1, n, n) * hc_scale[2] + hc_base[2 * n :].view(
        1, n, n
    )
    comb = torch.softmax(comb, dim=-1) + hc_sinkhorn_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    li = (pre_mix.float().view(t, n, 1) * r.float()).sum(dim=1).to(residual.dtype)
    lf = li.float()
    li = (
        lf
        * torch.rsqrt(lf.square().mean(-1, keepdim=True) + norm_eps)
        * norm_weight.float()
    ).to(residual.dtype)
    return r, post.unsqueeze(-1), comb, li, pre


def make_inputs(t, h, device="cuda", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    residual = (torch.randn(t, 4, h, device=device, generator=g) * 1.5).to(
        torch.bfloat16
    )
    y = torch.randn(t, h, device=device, generator=g).to(torch.bfloat16)
    fn = torch.randn(24, 4 * h, device=device, generator=g) * 0.01
    scale = torch.tensor([0.7, 1.3, 0.9], device=device)
    base = torch.randn(24, device=device, generator=g) * 0.5
    post = torch.rand(t, 4, 1, device=device, generator=g) * 2
    comb = torch.softmax(torch.randn(t, 4, 4, device=device, generator=g), -1)
    pre = torch.sigmoid(torch.randn(t, 4, device=device, generator=g)) + 1e-6
    weight = (1 + 0.2 * torch.randn(h, device=device, generator=g)).to(torch.bfloat16)
    return residual, y, fn, scale, base, post, comb, pre, weight


def reference_for(args, kw):
    reference_kw = dict(kw)
    if kw["pre_mix"] is None:
        identity = torch.zeros(
            (args[0].shape[0], 4), device=args[0].device, dtype=torch.float32
        )
        identity[:, 0] = 1
        reference_kw["pre_mix"] = identity
    return ref_mhc_fused_post_pre_delayed_rmsnorm(*args, **reference_kw)


def case(t, h, mode, seed=0):
    r, y, fn, scale, base, post, comb, pre, w = make_inputs(t, h, seed=seed)
    args = (r, fn, scale, base, *ARGS)
    kw = {"pre_mix": pre, "norm_weight": w}
    if "no_post" not in mode:
        kw.update(sublayer_out=y, post_layer_mix=post, comb_res_mix=comb)
    if "identity" in mode:
        kw["pre_mix"] = None
    return args, kw, reference_for(args, kw)


def check(out, reference):
    torch.testing.assert_close(out[0], reference[0], rtol=1.6e-2, atol=1e-5)
    torch.testing.assert_close(out[3], reference[3], rtol=1e-2, atol=2e-2)
    for i in (1, 2, 4):
        torch.testing.assert_close(out[i], reference[i], rtol=1e-3, atol=5e-4)


@cache
def backend_function(name):
    if name == "hip":
        return mega_mhc
    if name == "flydsl":
        from aiter.ops.flydsl.mega_mhc_kernels import flydsl_mega_mhc

        return flydsl_mega_mhc
    if name == "triton":
        from aiter.ops.triton.fusions.mhc_fused_post_pre_delayed_rmsnorm import (
            mhc_fused_post_pre_delayed_rmsnorm,
        )

        return mhc_fused_post_pre_delayed_rmsnorm
    raise ValueError(f"unknown backend: {name}")


def benchmark_bytes(m, hidden_size, mode):
    """Dominant tensor traffic, matching the FlyDSL Mega-MHC benchmark.

    Count BF16 residual input and norm output, plus sublayer input and residual
    output with post-mix. Read the FP32 projection weight once (equivalently its
    BF16 hi/lo pack). Exclude scratch, repeated loads and small gate/norm tensors;
    this is a common effective-bandwidth model, not measured HBM traffic.
    """
    if not m:
        return 0
    per_token = hidden_size * (10 if "no_post" in mode else 20)
    fn_bytes = 24 * 4 * hidden_size * 4
    return m * per_token + fn_bytes


class _BenchmarkBackend:
    """Prepare each rotated weight/workspace before run_perftest starts timing."""

    def __init__(self, name, tokens, scalar_args):
        self.name = name
        self.backend = backend_function(name)
        self.tokens = tokens
        self.scalar_args = scalar_args
        self.workspaces = {}
        self.prepared_addresses = set()
        self.used_addresses = set()

    def __call__(
        self, r, fn, scale, base, pre, sublayer, post, comb, weight, residual_out
    ):
        self.used_addresses.add(r.data_ptr())
        kw = {
            "pre_mix": pre,
            "sublayer_out": sublayer,
            "post_layer_mix": post,
            "comb_res_mix": comb,
            "norm_weight": weight,
            "residual_out": residual_out,
        }
        if self.name == "hip":
            kw["workspace"] = self.workspaces[id(fn)]
        return self.backend(r, fn, scale, base, *self.scalar_args, **kw)

    def prepare(self, *rotated):
        # Called before profiling for the original tensors and every copy.
        # HIP owns a workspace per fn; FlyDSL warms its per-weight cache here.
        self.prepared_addresses.add(rotated[0].data_ptr())
        if self.name == "hip":
            self.workspaces[id(rotated[1])] = prepare_mega_mhc(rotated[1], self.tokens)
        self(*rotated)
        self.used_addresses.discard(rotated[0].data_ptr())


@benchmark()
def benchmark_mega_mhc(m, hidden_size, mode, backends=BACKENDS, seed=0, num_iters=101):
    """Like test_mhc.py: check outputs, time each backend, return a table row."""
    args, kw, reference = case(m, hidden_size, mode, seed)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    result = {"status": "PASS"}
    num_bytes = benchmark_bytes(m, hidden_size, mode)
    with torch.cuda.stream(stream):
        # Pass actual tensors to run_perftest so its default memory-sizing and
        # deepcopy rotation can see all inputs and the residual output buffer.
        tensors = (
            *args[:4],
            kw["pre_mix"],
            kw.get("sublayer_out"),
            kw.get("post_layer_mix"),
            kw.get("comb_res_mix"),
            kw["norm_weight"],
            torch.empty_like(args[0]) if "sublayer_out" in kw else None,
        )
        for name in backends:
            caller = _BenchmarkBackend(name, m, args[4:])
            caller.prepare(*tensors)
            out = caller(*tensors)
            check(out, reference)
            if not m:
                result[f"{name}_us"] = float("nan")
                result[f"{name}_TB/s"] = float("nan")
                continue

            out, us = run_perftest(
                caller, *tensors, num_iters=num_iters, prepare_func=caller.prepare
            )
            if not math.isfinite(us) or us <= 0:
                raise RuntimeError(f"{name} returned an invalid GPU timing: {us}")
            check(out, reference)
            assert caller.used_addresses == caller.prepared_addresses
            print(
                f"  {name}: {len(caller.prepared_addresses)} auto-rotated buffer sets"
            )
            result[f"{name}_us"] = us
            result[f"{name}_TB/s"] = num_bytes / us / 1e6
            for workspace in caller.workspaces.values():
                assert _counter_state_ok(workspace)
            # Keep graph correctness coverage separate from rotating timings.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                graph_out = caller(*tensors)
            saved = args[0].clone()
            args[0].mul_(0.91)
            graph.replay()
            check(graph_out, reference_for(args, kw))
            args[0].copy_(saved)
            graph.replay()
            check(graph_out, reference)
            if name == "hip":
                assert _counter_state_ok(caller.workspaces[id(args[1])])
            del graph, graph_out, out, saved
            caller.workspaces.clear()
    stream.synchronize()
    return result


def check_reference(t, h, mode):
    args, kw, reference = case(t, h, mode)
    workspace = prepare_mega_mhc(args[1], t)
    out = mega_mhc(*args, workspace=workspace, **kw)
    check(out, reference)
    if "no_post" in mode:
        assert out[0] is args[0]
    assert _counter_state_ok(workspace)
    # At the production H, also compare both implementations with the same FP32 reference.
    if t and h == 5120:
        triton_out = backend_function("triton")(*args, **kw)
        check(triton_out, reference)


def check_split_reuse(split_k):
    args, kw, reference = case(17, 5120, "post", seed=3)
    workspace = prepare_mega_mhc(args[1], 17, split_k=split_k)
    # Eager calls reuse counters and overwrite partials. New residual data must
    # not observe a previous call's partials or already normalized collapse.
    for step in range(4):
        args[0].mul_(0.9)
        reference = ref_mhc_fused_post_pre_delayed_rmsnorm(*args, **kw)
        out = mega_mhc(*args, workspace=workspace, **kw)
        check(out, reference)
        assert _counter_state_ok(workspace)


def check_graphs_and_streams(token_counts):
    cases = [
        case(token_counts[0], 5120, "post", seed=7),
        case(token_counts[1], 5120, "no_post", seed=9),
    ]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    torch.cuda.synchronize()
    graphs, workspaces, outputs = [], [], []
    for (args, kw, _), stream in zip(cases, streams):
        with torch.cuda.stream(stream):
            workspace = prepare_mega_mhc(args[1], args[0].shape[0])
            mega_mhc(*args, workspace=workspace, **kw)  # JIT warmup outside capture
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = mega_mhc(*args, workspace=workspace, **kw)
        graphs.append(graph)
        workspaces.append(workspace)
        outputs.append(out)
    for _ in range(30):
        for graph, stream in zip(graphs, streams):
            with torch.cuda.stream(stream):
                graph.replay()
    torch.cuda.synchronize()
    for out, (_, _, reference), workspace in zip(outputs, cases, workspaces):
        check(out, reference)
        assert _counter_state_ok(workspace)


def check_workspace_contract():
    args, kw, _ = case(4, 5120, "post")
    workspace = prepare_mega_mhc(args[1], 4)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), expect_error(ValueError, message="current stream"):
        mega_mhc(*args, workspace=workspace, **kw)
    args[1].add_(0.01)
    with expect_error(ValueError, message="version"):
        mega_mhc(*args, workspace=workspace, **kw)
    with expect_error(ValueError, message="share storage"):
        mega_mhc(*args, residual_out=args[0], **kw)
    with expect_error(ValueError, message="split_k"):
        prepare_mega_mhc(args[1], 4, split_k=3)
    with (
        expect_error(RuntimeError, message="before graph capture"),
        torch.cuda.graph(torch.cuda.CUDAGraph()),
    ):
        prepare_mega_mhc(args[1], 4)


def check_inference_weights():
    with torch.inference_mode():
        args, kw, reference = case(4, 5120, "identity_pre")
        workspace = prepare_mega_mhc(args[1], 4)
        check(mega_mhc(*args, workspace=workspace, **kw), reference)


def check_oversubscribed_splits():
    # 260 token blocks * 5 splits exceeds the CU count: no residency assumption.
    args, kw, reference = case(4160, 5120, "post", seed=21)
    workspace = prepare_mega_mhc(args[1], 4160, split_k=5)
    for _ in range(3):
        check(mega_mhc(*args, workspace=workspace, **kw), reference)
        assert _counter_state_ok(workspace)


def check_dispatch_boundaries(tokens, mode):
    args, kw, reference = case(tokens, 5120, mode, seed=41)
    ws = prepare_mega_mhc(args[1], tokens)
    for _ in range(2):
        check(mega_mhc(*args, workspace=ws, **kw), reference)
        assert _counter_state_ok(ws)


def check_large_tiles_changed_graph_input(tokens, mode):
    """Exercise register/LDS staging and normalization, including padded rows."""
    args, kw, reference = case(tokens, 5120, mode, seed=53)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        ws = prepare_mega_mhc(args[1], tokens)
        check(mega_mhc(*args, workspace=ws, **kw), reference)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = mega_mhc(*args, workspace=ws, **kw)
        for _ in range(2):
            args[0].mul_(0.91)
            graph.replay()
            check(out, reference_for(args, kw))
            assert _counter_state_ok(ws)


def check_switch_sinkhorn_repeat_on_same_workspace(tokens):
    args, kw, _ = case(tokens, 5120, "post", seed=17)
    ws = prepare_mega_mhc(args[1], tokens)
    for repeat in (20, 3, 20, 1):
        changed_args = (*args[:-1], repeat)
        reference = ref_mhc_fused_post_pre_delayed_rmsnorm(*changed_args, **kw)
        check(mega_mhc(*changed_args, workspace=ws, **kw), reference)
        assert _counter_state_ok(ws)


def check_switch_post_on_same_workspace(tokens):
    """Post and no-post may use different splits in the same scratch allocation."""
    args, kw, _ = case(tokens, 5120, "post", seed=67)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        ws = prepare_mega_mhc(args[1], tokens)
        no_post_kw = {key: kw[key] for key in ("pre_mix", "norm_weight")}
        identity_kw = dict(kw, pre_mix=None)
        variants = (kw, no_post_kw, identity_kw, kw)
        for options in variants:
            check(
                mega_mhc(*args, workspace=ws, **options), reference_for(args, options)
            )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = [mega_mhc(*args, workspace=ws, **options) for options in variants]
        for _ in range(15):
            graph.replay()
        args[0].mul_(0.91)
        graph.replay()
        for output, options in zip(outputs, variants):
            check(output, reference_for(args, options))
        assert _counter_state_ok(ws)


def check_generation_wrap_and_changed_graph_input(tokens):
    args, kw, _ = case(tokens, 5120, "post", seed=49)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        ws = prepare_mega_mhc(args[1], tokens)
        ctr = ws.counters.view(-1, 32)
        ctr[:, 0].fill_(-256)
        ctr[:, 3].fill_(0xFFFFFF)
        mega_mhc(*args, workspace=ws, **kw)
        assert torch.all(ctr[:, 0] == 0)
        assert _counter_state_ok(ws)
        # Warm both dispatches before capture. Each replay alternates between
        # generation-tagged and zero-based arrivals sharing the same partials.
        generic_args = (*args[:-1], 3)
        mega_mhc(*generic_args, workspace=ws, **kw)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            tuned_out = mega_mhc(*args, workspace=ws, **kw)
            generic_out = mega_mhc(*generic_args, workspace=ws, **kw)
        for _ in range(30):
            graph.replay()
        args[0].mul_(0.91)
        graph.replay()
        check(tuned_out, ref_mhc_fused_post_pre_delayed_rmsnorm(*args, **kw))
        check(generic_out, ref_mhc_fused_post_pre_delayed_rmsnorm(*generic_args, **kw))
        assert _counter_state_ok(ws)


def check_masked_dispatch_uses_generic():
    args, kw, reference = case(4, 5120, "post", seed=19)
    original_mask = os.environ.get("ROC_GLOBAL_CU_MASK")
    os.environ["ROC_GLOBAL_CU_MASK"] = "0:0-127"
    try:
        ws = prepare_mega_mhc(args[1], 4)
        assert ws.variant == "generic"
        assert ws.split_k == 1
        check(mega_mhc(*args, workspace=ws, **kw), reference)
        assert _counter_state_ok(ws)
    finally:
        if original_mask is None:
            os.environ.pop("ROC_GLOBAL_CU_MASK", None)
        else:
            os.environ["ROC_GLOBAL_CU_MASK"] = original_mask


@contextmanager
def expect_error(error_type, message):
    try:
        yield
    except error_type as error:
        assert message in str(error), f"expected {message!r} in {error!r}"
    else:
        raise AssertionError(f"expected {error_type.__name__}: {message}")


def run_correctness_checks():
    """Run the complete HIP regression matrix without timing or a test runner."""
    groups = [
        (
            check_reference,
            [
                (t, h, mode)
                for t, h in [
                    (0, 5120),
                    (1, 32),
                    (3, 64),
                    (17, 256),
                    (4, 5120),
                    (17, 5120),
                    (64, 5120),
                    (384, 5120),
                    (4096, 5120),
                    (4160, 5120),
                    (16384, 5120),
                    (17, 7168),
                    (17, 65536),
                ]
                for mode in MODES
            ],
        ),
        (check_split_reuse, [(split_k,) for split_k in [1, 2, 5, 10, 16, 20, 32]]),
        (
            check_graphs_and_streams,
            [
                (token_counts,)
                for token_counts in [(17, 384), (2048, 2048), (4160, 4160)]
            ],
        ),
        (check_workspace_contract, [()]),
        (check_inference_weights, [()]),
        (check_oversubscribed_splits, [()]),
    ]
    if get_gfx_runtime() == "gfx950":
        groups += [
            (
                check_dispatch_boundaries,
                [
                    (tokens, mode)
                    for tokens in [
                        1,
                        5,
                        8,
                        16,
                        128,
                        129,
                        256,
                        257,
                        385,
                        512,
                        513,
                        1024,
                        1025,
                        1536,
                        2047,
                        2048,
                        2049,
                        4097,
                        4161,
                        7920,
                        7921,
                        8191,
                        8192,
                        8193,
                        15840,
                        15841,
                        16385,
                    ]
                    for mode in MODES
                ],
            ),
            (
                check_large_tiles_changed_graph_input,
                [
                    (tokens, mode)
                    for mode in MODES
                    for tokens in [1025, 2048, 3072, 4096, 4160, 8192, 8193, 15841]
                ],
            ),
            (
                check_switch_sinkhorn_repeat_on_same_workspace,
                [(tokens,) for tokens in [4, 17, 128, 1024, 4160]],
            ),
            (
                check_switch_post_on_same_workspace,
                [(tokens,) for tokens in [16, 17, 128, 2048]],
            ),
            (
                check_generation_wrap_and_changed_graph_input,
                [(tokens,) for tokens in [4, 17, 128, 1024, 2048]],
            ),
            (check_masked_dispatch_uses_generic, [()]),
        ]
    count = 0
    for function, cases in groups:
        label = function.__name__.removeprefix("check_")
        print(f"Checking {label}: {len(cases)} cases", flush=True)
        for values in cases:
            try:
                function(*values)
            except Exception as error:
                raise AssertionError(f"{label}{values} failed") from error
            count += 1
    torch.cuda.synchronize()
    print(f"Mega-MHC correctness: {count} passed", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Mega-MHC BF16 correctness and HIP / FlyDSL / Triton performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "-t",
        "--tokens",
        type=int,
        nargs="+",
        default=[
            1,
            2,
            3,
            4,
            5,
            8,
            16,
            17,
            32,
            64,
            128,
            256,
            384,
            512,
            1024,
            2048,
            4096,
            4160,
            8192,
            16384,
        ],
    )
    parser.add_argument(
        "-n", "--hidden", "--hidden-size", type=int, nargs="+", default=[5120]
    )
    parser.add_argument("--mode", "--modes", choices=MODES, nargs="+", default=MODES)
    parser.add_argument("--backends", choices=BACKENDS, nargs="+", default=BACKENDS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--iters",
        type=int,
        default=101,
        help="timed calls with automatic buffer rotation",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="run the full HIP regression matrix instead of the selected benchmarks",
    )
    args = parser.parse_args()
    if any(m < 0 for m in args.tokens):
        parser.error("tokens must be nonnegative")
    if any(h <= 0 or h % 32 for h in args.hidden):
        parser.error("hidden size must be a positive multiple of 32")
    if args.iters < 2:
        parser.error("iters must be at least 2 (the first sample is warmup)")
    if not torch.cuda.is_available() or get_gfx_runtime() not in ("gfx950", "gfx942"):
        parser.error("Mega-MHC requires gfx950 or gfx942")
    if args.check_only:
        run_correctness_checks()
        return
    if "flydsl" in args.backends and get_gfx_runtime() != "gfx950":
        parser.error("FlyDSL Mega-MHC requires gfx950; use --backends hip triton")
    backends = tuple(dict.fromkeys(args.backends))
    for name in backends:
        try:
            backend_function(name)
        except ImportError as error:
            parser.error(
                f"{name} is unavailable: {error}; install the backend or omit it from --backends"
            )
    print(
        f"Mega-MHC BF16 | {get_gfx_runtime()} | {torch.cuda.get_device_name()} | "
        f"GPU time, {args.iters} calls per case with automatic buffer rotation",
        flush=True,
    )
    print(
        "Effective bandwidth: dominant tensor bytes / GPU time (TB/s); "
        "counts projection weights once, excludes scratch and repeated loads.",
        flush=True,
    )
    rows = []
    for h, mode, m in itertools.product(args.hidden, args.mode, args.tokens):
        print(f"Checking T={m}, H={h}, mode={mode}", flush=True)
        rows.append(benchmark_mega_mhc(m, h, mode, backends, args.seed, args.iters))
    columns = ["m", "hidden_size", "mode", "seed", "status"]
    columns += [f"{name}_{unit}" for name in backends for unit in ("us", "TB/s")]
    print("\nMega-MHC performance (us, TB/s):", flush=True)
    print(
        pd.DataFrame(rows)[columns].to_string(
            index=False, float_format="{:.3f}".format, na_rep="-"
        )
    )


if __name__ == "__main__":
    main()
