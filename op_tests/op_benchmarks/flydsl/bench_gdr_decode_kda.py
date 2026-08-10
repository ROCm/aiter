#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tuning sweep and A/B benchmark for the KDA (per-channel) FlyDSL GDR decode.

Kimi-K3 decodes at a 1:1 head ratio, which matches no row in
``gdr_decode_tuned.csv`` -- all of them are GQA at ratio 2 or 4 -- so an untuned
call falls back to (1, 4, 8) and any timing taken before the sweep measures the
fallback. ``--sweep`` produces the missing rows; ``--bench`` compares against the
deployed Triton kernel. Rows go out as ``gate_mode=kda``, which the table keys on,
so they never reach the scalar gate.

``--bench`` reports two numbers because they differ by ~2x at decode batch sizes
and quoting the wrong one is the easy mistake: ``call_us`` (host cost included,
what a decode loop runs at) and ``kernel_us`` (excluded, what the sweep ranks on).

Usage:
    # Tuning sweep, emits rows in gdr_decode_tuned.csv format
    python bench_gdr_decode_kda.py --sweep -o rows.csv

    # A/B against vLLM's fused_recurrent_kda_packed_decode
    python bench_gdr_decode_kda.py --bench

    # Both, at the ticket's batch sizes
    python bench_gdr_decode_kda.py --sweep --bench

The comparator lives on vLLM's kimi-k3 branch: point --vllm at a checkout. Without
it --bench drops the Triton column rather than failing, since --sweep needs no vLLM.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import triton
from flydsl.runtime.device import get_rocm_arch

from aiter.ops.flydsl.kernels.gdr_decode import create_vk_gdr_decode_kernel
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, get_dtype_str

# The KDA oracle is test-only, so it ships with op_tests rather than with aiter
# and is not importable from an installed wheel.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from op_tests.kda_ref import kda_gate, l2norm, naive_recurrent_kda

# K3 @ TP8: 12 heads 1:1, 128-wide, bf16 activations over an f32 paged state.
H, K, V = 12, 128, 128
DTYPE = torch.bfloat16
STATE_DTYPE = torch.float32
G_MIN = -5.0
BATCHES = (1, 4, 64, 256)

# Wider than the original space (NUM_BLOCKS_PER_V_DIM<=8, NUM_WARPS<=4, sized for
# GQA's 32-64 value heads). K3 has 12, so only splitting V harder fills the grid.
NUM_BLOCKS_PER_V_DIM_CHOICES = (1, 2, 4, 8, 16, 32, 64)
NUM_WARPS_CHOICES = (1, 2, 4, 8, 16)
WARP_THREADS_K_CHOICES = (1, 2, 4, 8, 16, 32)

# What get_default_kwargs falls back to when no row matches the shape.
FALLBACK_CONFIG = (1, 4, 8)

# How many of the sweep's leaders get re-timed, and over how many trials.
TOP_N = 5
TRIALS = 5

# Calls per call_us measurement: enough to bury the trailing sync (the last
# call's device time, the only part not overlapped), few enough that B=256 is 30 ms.
LOOP_ITERS = 500

CSV_HEADER = (
    "arch,dtype,state_dtype,b,sq,num_k_heads,num_v_heads,head_k_dim,head_v_dim,"
    "NUM_BLOCKS_PER_V_DIM,NUM_WARPS,WARP_THREADS_K,duration,gate_mode"
)


def valid_configs():
    """Mirror the kernel's shape asserts, so a real compile error stays
    distinguishable from a geometry that was never legal."""
    values_per_thread_k = 4 if STATE_DTYPE is torch.float32 else 8
    out = []
    for nbpv, nw, wtk in itertools.product(
        NUM_BLOCKS_PER_V_DIM_CHOICES, NUM_WARPS_CHOICES, WARP_THREADS_K_CHOICES
    ):
        warp_threads_v = 64 // wtk
        if warp_threads_v * wtk != 64:
            continue
        warp_tile_k = wtk * values_per_thread_k
        if K % warp_tile_k or K // warp_tile_k < 1:
            continue
        if V % nbpv:
            continue
        tile_v = V // nbpv
        warp_group_tile_v = nw * warp_threads_v
        if tile_v % warp_group_tile_v or tile_v // warp_group_tile_v < 1:
            continue
        out.append((nbpv, nw, wtk))
    return out


def make_inputs(B, device="cuda", seed=0):
    """One KDA decode case, in both APIs' layouts over identical values.

    Triton reads q/k/v packed as ``[B, 2*H*K + H*V]``, FlyDSL takes three
    tensors, so "same inputs" means same numbers -- FlyDSL's are zero-copy views
    at Sq = 1. Slots start at 1: Triton returns zeros for ``state_idx <= 0``, so
    slot 0 would time an early return.
    """
    torch.manual_seed(seed)
    mixed_qkv = torch.randn(B, 2 * H * K + H * V, dtype=DTYPE, device=device)
    q = mixed_qkv[:, : H * K].view(B, 1, H, K)
    k = mixed_qkv[:, H * K : 2 * H * K].view(B, 1, H, K)
    v = mixed_qkv[:, 2 * H * K :].view(B, 1, H, V)

    raw_g = torch.randn(1, B, H, K, dtype=DTYPE, device=device)
    raw_beta = torch.randn(1, B, H, dtype=DTYPE, device=device)
    A_log = (torch.randn(H, dtype=torch.float32, device=device) * 0.5).contiguous()
    dt_bias = (torch.randn(H, K, dtype=torch.float32, device=device) * 0.1).contiguous()

    n_slots = B + 1
    pool = torch.randn(n_slots, H, V, K, dtype=STATE_DTYPE, device=device)
    indices = torch.arange(1, 1 + B, dtype=torch.int32, device=device)

    return {
        "mixed_qkv": mixed_qkv,
        "q": q,
        "k": k,
        "v": v,
        "a": raw_g[0].unsqueeze(1),  # (B, 1, H, K), a view of raw_g
        "b": raw_beta[0].unsqueeze(1),  # (B, 1, H)
        "raw_g": raw_g,
        "raw_beta": raw_beta,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "pool": pool,
        "indices": indices,
        "B": B,
    }


def flydsl_runner(inp, config):
    """Bind one explicit config, bypassing the wrapper's CSV lookup.

    ``flydsl_gdr_decode`` resolves its config from ``gdr_decode_tuned.csv``, so a
    sweep must build the kernel directly. need_shuffle_state=False, K3's layout.
    """
    nbpv, nw, wtk = config
    state = inp["pool"]
    out = torch.zeros(inp["B"], 1, H, V, dtype=DTYPE, device=state.device)
    q, k, v = inp["q"].contiguous(), inp["k"].contiguous(), inp["v"].contiguous()
    exe = create_vk_gdr_decode_kernel(
        get_dtype_str(DTYPE),
        get_dtype_str(inp["A_log"].dtype),
        get_dtype_str(inp["dt_bias"].dtype),
        get_dtype_str(state.dtype),
        1,
        H,
        H,
        K,
        V,
        q.stride(),
        k.stride(),
        v.stride(),
        state.stride(),
        inp["a"].stride(),
        inp["b"].stride(),
        True,
        "kda",
        NUM_BLOCKS_PER_V_DIM=nbpv,
        NUM_WARPS=nw,
        WARP_THREADS_K=wtk,
    )
    a, b = inp["a"], inp["b"]
    A_log, dt_bias = inp["A_log"], inp["dt_bias"]
    indices = inp["indices"]
    stream = torch.cuda.current_stream()

    def run():
        _run_compiled(
            exe,
            q,
            k,
            v,
            a,
            b,
            dt_bias,
            A_log,
            indices,  # read slots
            indices,  # write slots -- decode updates the slot it read
            state,
            out,
            inp["B"],
            stream,
        )

    return run, out


def torch_reference(inp, pool_before):
    """``pool_before`` must be the state from *before* the kernel ran: both
    kernels decay the pool in place, so ``inp["pool"]`` would feed the reference
    already-decayed state and compare two different problems."""
    initial_state = pool_before[inp["indices"].long()].clone().transpose(-1, -2)
    return naive_recurrent_kda(
        l2norm(inp["q"]),
        l2norm(inp["k"]),
        inp["v"],
        kda_gate(inp["a"], inp["A_log"], inp["dt_bias"], g_min=G_MIN),
        inp["b"].float().sigmoid(),
        scale=K**-0.5,
        initial_state=initial_state,
        output_final_state=True,
    )


def rmse_ratio(ref, got):
    """vLLM's bar (test_kda.py:92): RMSE-relative, absolute-error escape hatch."""
    ref, got = ref.detach().float(), got.detach().float()
    abs_err = (ref - got).abs().max().item()
    rmse = (ref - got).square().mean().sqrt().item()
    base = ref.square().mean().sqrt().item()
    return abs_err, rmse / (base + 1e-8)


def check(inp, out, state_after, pool_before):
    ref_out, ref_state = torch_reference(inp, pool_before)
    got_state = state_after[inp["indices"].long()].transpose(-1, -2)
    o_abs, o_rel = rmse_ratio(ref_out, out)
    s_abs, s_rel = rmse_ratio(ref_state, got_state)
    ok = (o_abs <= 1e-3 or o_rel < 1e-3) and (s_abs <= 1e-3 or s_rel < 1e-3)
    return ok, max(o_abs, s_abs)


def kernel_us(fn):
    """Time the *kernel*: device time for one call, CUDA events, L2 flushed.
    The events sit on the device timeline, so host cost is not in the number."""
    return triton.testing.do_bench(fn, warmup=25, rep=100) * 1e3


def call_us(fn):
    """Time the *call*: wall clock per call, calls issued back to back.

    Host cost included, so this exceeds ``kernel_us`` whenever the host cannot
    keep the GPU fed -- here, every batch below 64. A decode loop is thousands of
    these, and vLLM marks this op ``@eager_break_during_capture``, so the host
    cost is paid per step in production rather than captured away.

    One sync, at the end: syncing per call would serialise host and device.
    """
    for _ in range(25):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(LOOP_ITERS):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / LOOP_ITERS * 1e6


def confirm(B, candidates):
    """Re-time the sweep's leaders over independent trials, rank on median.

    Also times the untuned fallback, so the sweep reports what it bought and not
    just what it picked. Candidates come from this GPU's sweep -- the winning set
    differs between gfx942 and gfx950.
    """
    print(f"    re-timing the top {len(candidates)} over {TRIALS} trials:")
    rows = []
    for cfg in list(candidates) + [FALLBACK_CONFIG]:
        times, err = [], math.nan
        for t in range(TRIALS):
            inp = make_inputs(B, seed=t)
            pool0 = inp["pool"].clone()
            run, out = flydsl_runner(inp, cfg)
            run()
            torch.cuda.synchronize()
            if t == 0:
                _, err = check(inp, out, inp["pool"], pool0)
            inp["pool"].copy_(pool0)
            times.append(kernel_us(run))
        rows.append((statistics.median(times), min(times), max(times), cfg, err))

    fallback = next(r[0] for r in rows if r[3] == FALLBACK_CONFIG)
    rows.sort()
    for med, lo, hi, cfg, _ in rows:
        tag = "   <-- untuned fallback" if cfg == FALLBACK_CONFIG else ""
        print(
            f"      {cfg!s:<12} median {med:8.2f}  min {lo:8.2f}  max {hi:8.2f}"
            f"  spread {(hi - lo) / med * 100:4.1f}%"
            f"  vs fallback {fallback / med:.2f}x{tag}"
        )
    best_med, _, _, best_cfg, best_err = rows[0]
    print(
        f"    winner {best_cfg} at {best_med:.2f} us"
        f" ({fallback / best_med:.2f}x the fallback, err {best_err:.1e})\n"
    )
    return best_med, best_cfg, best_err


def sweep(args):
    arch = get_rocm_arch()
    configs = valid_configs()
    print(f"arch {arch} · {len(configs)} valid configs · batches {list(BATCHES)}\n")

    rows = []
    for B in BATCHES:
        results = []
        for cfg in configs:
            inp = make_inputs(B)
            baseline_pool = inp["pool"].clone()
            try:
                run, out = flydsl_runner(inp, cfg)
                run()
                torch.cuda.synchronize()
            except Exception as exc:  # noqa: BLE001 - any failure is a lost config
                print(f"  B={B:<4} {cfg}  COMPILE/RUN FAIL: {type(exc).__name__}")
                continue

            ok, err = check(inp, out, inp["pool"], baseline_pool)
            if not ok:
                # Parity at every config: fast-because-wrong must not win.
                print(f"  B={B:<4} {cfg}  PARITY FAIL err={err:.2e}")
                continue

            inp["pool"].copy_(baseline_pool)
            run, _ = flydsl_runner(inp, cfg)
            us = kernel_us(run)
            results.append((us, cfg, err))

        results.sort()
        worst = results[-1][0]
        print(f"B={B}: {len(results)} configs passed parity")
        print(
            f"    spread: slowest valid config is {worst / results[0][0]:.1f}x the best"
        )

        # The single-shot argmin is not trustworthy: the leaders land within a
        # few percent, the same order as run-to-run drift. Rank on the median.
        best_us, best_cfg, _ = confirm(B, [c for _, c, _ in results[:TOP_N]])

        nbpv, nw, wtk = best_cfg
        rows.append(
            f"{arch},{DTYPE},{STATE_DTYPE},{B},1,{H},{H},{K},{V},"
            f"{nbpv},{nw},{wtk},{best_us},kda"
        )

    print(CSV_HEADER)
    for r in rows:
        print(r)
    if args.output:
        Path(args.output).write_text(
            CSV_HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {args.output}")
    return rows


def load_triton(vllm_path):
    """Load the comparator from a vLLM checkout without importing vLLM itself.

    ``import vllm`` drags in pydantic and the config stack, so the file is loaded
    directly with its three vLLM imports stubbed. The stubs are faithful: at the
    default ``FLA_USE_FAST_OPS=0`` vLLM's ``exp``/``log`` *are* ``tl.exp``/
    ``tl.log``, and the rest is plain triton and integer math. Guarded below.
    """
    import importlib.util
    import types

    if os.environ.get("FLA_USE_FAST_OPS", "0") == "1":
        # Otherwise vLLM binds fast_expf/fast_logf and the stub changes the math.
        raise RuntimeError("unset FLA_USE_FAST_OPS to compare like for like")

    path = Path(vllm_path or "/workspace/vllm")
    target = path / "vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py"
    if not target.exists():
        print(f"note: comparator not found at {target}; skipping the Triton column")
        return None

    import triton as _triton
    import triton.language as _tl

    def _mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    _mod("vllm")
    _mod("vllm.third_party")
    _mod("vllm.third_party.flash_linear_attention")
    _mod("vllm.third_party.flash_linear_attention.ops")
    _mod("vllm.third_party.flash_linear_attention.ops.op", exp=_tl.exp, log=_tl.log)
    _mod("vllm.triton_utils", tl=_tl, triton=_triton)
    _mod("vllm.utils")
    _mod(
        "vllm.utils.math_utils",
        cdiv=lambda a, b: -(a // -b),
        next_power_of_2=lambda n: 1 if n < 1 else 1 << (n - 1).bit_length(),
    )

    spec = importlib.util.spec_from_file_location("_kda_fused_recurrent", target)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001 - degrade to the FlyDSL column alone
        print(f"note: vLLM comparator unavailable ({type(exc).__name__}: {exc})")
        return None
    return mod.fused_recurrent_kda_packed_decode


def bench_one(B, flydsl_gdr_decode, triton_fn):
    """One row of both A/B tables: every measurement at one batch size.

    Its own function so the timed closures bind parameters, not a loop variable.
    Both kernels decay the pool in place, hence the ``copy_(pool0)`` before each
    measurement.
    """
    inp = make_inputs(B)
    pool0 = inp["pool"].clone()
    out = torch.zeros(B, 1, H, V, dtype=DTYPE, device="cuda")

    # Hoisted deliberately: q/k/v are packed-buffer views and the wrapper calls
    # .contiguous(), so timing this would charge FlyDSL for unpacking per call.
    # Packed reads are a separate ticket, so the unpack gets its own column.
    q, k, v = inp["q"].contiguous(), inp["k"].contiguous(), inp["v"].contiguous()

    def fly():
        flydsl_gdr_decode(
            q,
            k,
            v,
            inp["a"],
            inp["b"],
            inp["dt_bias"],
            inp["A_log"],
            inp["indices"],
            inp["pool"],
            out,
            use_qk_l2norm=True,
            need_shuffle_state=False,
        )

    fly()
    torch.cuda.synchronize()
    fly_ok, fly_err = check(inp, out, inp["pool"], pool0)
    inp["pool"].copy_(pool0)
    fly_kernel = kernel_us(fly)
    inp["pool"].copy_(pool0)
    fly_call = call_us(fly)

    def tri():
        return triton_fn(
            inp["mixed_qkv"],
            inp["raw_g"],
            inp["raw_beta"],
            inp["A_log"],
            inp["dt_bias"].view(-1),
            G_MIN,
            inp["pool"],
            inp["indices"],
        )

    tri_kernel, tri_call, tri_ok, tri_err = math.nan, math.nan, None, math.nan
    if triton_fn is not None:
        inp["pool"].copy_(pool0)
        tri_out, _ = tri()
        torch.cuda.synchronize()
        tri_ok, tri_err = check(inp, tri_out[0].unsqueeze(1), inp["pool"], pool0)
        inp["pool"].copy_(pool0)
        tri_kernel = kernel_us(tri)
        inp["pool"].copy_(pool0)
        tri_call = call_us(tri)

    # What a caller got before the sweep: with no 12x12 row the wrapper falls
    # back to (1, 4, 8). Bound directly, since the wrapper now finds a row.
    inp["pool"].copy_(pool0)
    untuned_run, _ = flydsl_runner(inp, FALLBACK_CONFIG)
    untuned_run()
    torch.cuda.synchronize()
    inp["pool"].copy_(pool0)
    untuned_kernel = kernel_us(untuned_run)

    unpack_kernel = kernel_us(
        lambda: (
            inp["q"].contiguous(),
            inp["k"].contiguous(),
            inp["v"].contiguous(),
        )
    )

    def ratio(tri, fly):
        return math.nan if math.isnan(tri) else tri / fly

    return {
        "B": B,
        # How fast a decode loop runs. Host cost included; the ticket's number.
        "tri_call": tri_call,
        "fly_call": fly_call,
        "call_speedup": ratio(tri_call, fly_call),
        # How fast the kernels are. Host cost excluded; what the sweep ranks on.
        "tri_kernel": tri_kernel,
        "fly_kernel": fly_kernel,
        "kernel_speedup": ratio(tri_kernel, fly_kernel),
        # Kernel-axis asides: what tuning bought, what a packed caller would pay.
        "untuned_kernel": untuned_kernel,
        "unpack_kernel": unpack_kernel,
        "fly_ok": fly_ok,
        "fly_err": fly_err,
        "tri_ok": tri_ok,
        "tri_err": tri_err,
    }


def print_table(caption, columns, table):
    """``columns`` is (heading, width, key) each, plus 'x' on any *_speedup key."""
    print(f"\n{caption}\n")
    header = " ".join(f"{head:>{w}}" for head, w, _ in columns)
    print(header)
    print("-" * len(header))
    for r in table:
        cells = []
        for _, w, key in columns:
            if key == "B":
                cells.append(f"{r['B']:>{w}}")
            elif key.endswith("speedup"):
                cells.append(f"{r[key]:>{w - 1}.2f}x")
            else:
                cells.append(f"{r[key]:>{w}.2f}")
        print(" ".join(cells))


def bench(args):
    """A/B at the ticket's batch sizes, both kernels on identical values.

    Two tables, because "how fast is it" has two answers that disagree by ~2x at
    B=1 and are both true. The call table is the ticket's number: below B=64 both
    kernels are host-bound, so the kernel is not the constraint. The kernel table
    is what the sweep ranks on. FlyDSL goes through the public wrapper either
    way, so the numbers are what a consumer gets, tuning table included.
    """
    from aiter.ops.flydsl import flydsl_gdr_decode

    arch = get_rocm_arch()
    triton_fn = load_triton(args.vllm)
    print(f"\narch {arch} · H={H} K=V={K} bf16 · f32 state · all times us")

    table = [bench_one(B, flydsl_gdr_decode, triton_fn) for B in BATCHES]

    print_table(
        "how fast a decode loop runs -- wall clock per call, calls back to back",
        [
            ("B", 5, "B"),
            ("Triton", 10, "tri_call"),
            ("FlyDSL", 10, "fly_call"),
            ("speedup", 9, "call_speedup"),
        ],
        table,
    )
    print_table(
        "how fast the kernels are -- device time per call, host cost excluded",
        [
            ("B", 5, "B"),
            ("Triton", 10, "tri_kernel"),
            ("FlyDSL", 10, "fly_kernel"),
            ("speedup", 9, "kernel_speedup"),
            ("FlyDSL untuned", 15, "untuned_kernel"),
            ("QKV unpack", 12, "unpack_kernel"),
        ],
        table,
    )

    print("\nparity vs the torch reference (both must hold, or the timing is noise):")
    for r in table:
        tri_ok, tri_err = r["tri_ok"], r["tri_err"]
        t = "n/a" if tri_ok is None else f"{'ok' if tri_ok else 'FAIL'} {tri_err:.1e}"
        fly = f"{'ok' if r['fly_ok'] else 'FAIL'} {r['fly_err']:.1e}"
        print(f"  B={r['B']:<4} FlyDSL {fly}   Triton {t}")
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", action="store_true", help="tune for the 1:1 head ratio")
    p.add_argument("--bench", action="store_true", help="A/B against Triton")
    p.add_argument("-o", "--output", help="write the winning CSV rows here")
    p.add_argument("--vllm", help="path to a vLLM kimi-k3 checkout")
    args = p.parse_args()
    if not args.sweep and not args.bench:
        args.sweep = args.bench = True
    if args.sweep:
        sweep(args)
    if args.bench:
        bench(args)


if __name__ == "__main__":
    main()
