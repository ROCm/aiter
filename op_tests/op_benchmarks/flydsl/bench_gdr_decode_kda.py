#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tuning sweep and A/B benchmark for the KDA (per-channel) FlyDSL GDR decode.

Kimi-K3 decodes at a 1:1 head ratio. ``--sweep`` prints the winning
``(NUM_BLOCKS_PER_V_DIM, NUM_WARPS, WARP_THREADS_K)`` triples to paste into
``_KDA_DECODE_BY_ARCH`` in ``linear_attention_kernels.py``. ``--bench`` compares
those rows with both main's shape-based tiling policy and vLLM's standalone
Triton packed-decode fallback. Scalar GDR never consults this table.

``--bench`` reports device time (``kernel_us``): vLLM launches this kernel in a
graph with other kernels, so host-side wall clock is not the serving number.
Times are mean ± std over ``TRIALS`` trials on fresh inputs. The kernel-axis win
is a few percent, which is the size of the drift, so a number without its spread
cannot be quoted as a win.

Usage:
    # Tuning sweep, prints triples to paste into `_KDA_DECODE_BY_ARCH`
    python bench_gdr_decode_kda.py --sweep

    # A/B against vLLM's fused_recurrent_kda_packed_decode
    python bench_gdr_decode_kda.py --bench

    # Both, at the ticket's batch sizes
    python bench_gdr_decode_kda.py --sweep --bench

The comparator lives in vLLM's tree. ``--bench`` looks under ``--vllm``, then
``/workspace/vllm_vllm-project``, ``/workspace/vllm``, then those names as
siblings of this aiter checkout. ``--bench`` fails if it cannot load the
comparator; a table with an empty baseline is not a successful A/B.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import statistics
import sys
from pathlib import Path

import torch
import triton
from flydsl.runtime.device import get_rocm_arch

from aiter.ops.flydsl.kernels.gdr_decode import create_vk_gdr_decode_kernel
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, get_dtype_str
from aiter.ops.flydsl.linear_attention_kernels import _decode_tiling

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

# How many of the sweep's leaders get re-timed, and how many independent trials
# every timing is repeated over -- in the sweep and in the A/B alike. One
# measurement cannot separate a real difference from run-to-run drift when the
# two kernels land within a few percent of each other, which they do here.
TOP_N = 5
TRIALS = 5

# What `_KDA_DECODE_BY_ARCH` holds, so the sweep emits a pasteable entry.
TABLE_NAME = "_KDA_DECODE_BY_ARCH"


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


def main_tiling(B):
    """The config current main chooses when there is no KDA-tuned row."""
    config = _decode_tiling(B, H, K, V, str(STATE_DTYPE))
    return (
        config["NUM_BLOCKS_PER_V_DIM"],
        config["NUM_WARPS"],
        config["WARP_THREADS_K"],
    )


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
    """Bind one explicit config, bypassing the wrapper's KDA tiling table.

    ``flydsl_gdr_decode`` resolves its config from ``_KDA_DECODE_BY_ARCH``, so a
    sweep must build the kernel directly. need_shuffle_state=False, K3's layout.

    q/k/v stay views into ``mixed_qkv``, the same buffer the Triton comparator
    reads, and the kernel is compiled against those strides. One layout for the
    sweep and the A/B both: a config tuned against dense inputs would be tuned
    for a kernel no consumer builds.
    """
    nbpv, nw, wtk = config
    state = inp["pool"]
    out = torch.zeros(inp["B"], 1, H, V, dtype=DTYPE, device=state.device)
    q, k, v = inp["q"], inp["k"], inp["v"]
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


def confirm(B, candidates):
    """Re-time the sweep's leaders over independent trials, rank on median.

    Also times current main's shape-based tiling, so the sweep reports what it
    bought and not just what it picked. Candidates come from this GPU's sweep --
    the winning set differs between gfx942 and gfx950.
    """
    print(f"    re-timing the top {len(candidates)} over {TRIALS} trials:")
    baseline_config = main_tiling(B)
    rows = []
    for cfg in dict.fromkeys((*candidates, baseline_config)):
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

    baseline = next(r[0] for r in rows if r[3] == baseline_config)
    rows.sort()
    for med, lo, hi, cfg, _ in rows:
        tag = "   <-- main tiling" if cfg == baseline_config else ""
        print(
            f"      {cfg!s:<12} median {med:8.2f}  min {lo:8.2f}  max {hi:8.2f}"
            f"  spread {(hi - lo) / med * 100:4.1f}%"
            f"  vs main {baseline / med:.2f}x{tag}"
        )
    best_med, _, _, best_cfg, best_err = rows[0]
    print(
        f"    winner {best_cfg} at {best_med:.2f} us"
        f" ({baseline / best_med:.2f}x main's tiling, err {best_err:.1e})\n"
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
        _, best_cfg, _ = confirm(B, [c for _, c, _ in results[:TOP_N]])

        nbpv, nw, wtk = best_cfg
        rows.append(f"{B}: ({nbpv}, {nw}, {wtk})")

    # One line, shaped like the table it goes into; the durations above are the
    # evidence for it and are not stored.
    entry = f'    "{arch}": {{{", ".join(rows)}}},'
    print(f"\npaste into {TABLE_NAME}, replacing this arch's row:\n")
    print(entry)
    if args.output:
        Path(args.output).write_text(entry + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")
    return entry


_COMPARATOR_REL = Path("vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py")


def _comparator_candidates(vllm_path):
    """Places a vLLM checkout of ``fused_recurrent.py`` is expected to live."""
    roots = []
    if vllm_path:
        roots.append(Path(vllm_path))
    parent = _REPO_ROOT.parent
    for name in ("vllm_vllm-project", "vllm"):
        roots.append(Path("/workspace") / name)
        roots.append(parent / name)
    seen = set()
    out = []
    for root in roots:
        target = (root / _COMPARATOR_REL).resolve()
        if target in seen:
            continue
        seen.add(target)
        out.append(target)
    return out


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

    tried = _comparator_candidates(vllm_path)
    target = next((p for p in tried if p.is_file()), None)
    if target is None:
        locations = "\n".join(f"  - {p}" for p in tried)
        raise FileNotFoundError(
            "vLLM Triton comparator not found; looked at:\n"
            f"{locations}\n"
            "clone it with:\n"
            "  git clone https://github.com/vllm-project/vllm.git "
            "vllm_vllm-project"
        )

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
    except Exception as exc:
        raise RuntimeError(
            f"could not load vLLM Triton comparator at {target}"
        ) from exc
    print(f"comparator: {target}")
    return mod.fused_recurrent_kda_packed_decode


def bench_one(B, flydsl_gdr_decode, triton_fn, seed=0):
    """One trial of both A/B tables: every measurement at one batch size.

    Its own function so the timed closures bind parameters, not a loop variable.
    Both kernels decay the pool in place, hence the ``copy_(pool0)`` before each
    measurement.
    """
    inp = make_inputs(B, seed=seed)
    pool0 = inp["pool"].clone()
    out = torch.zeros(B, 1, H, V, dtype=DTYPE, device="cuda")

    # K3 hands q/k/v over as views into one packed buffer, and since #4573 the
    # wrapper compiles against the strides it is given instead of copying to
    # dense. So the strided reads are what a consumer pays and they belong inside
    # FlyDSL's own number. Flattening first is not the cheaper option and is not
    # measured: it costs 12.8-13.7 us against a strided penalty of at most 0.55
    # us (B=64; free at B=1, faster at B=4). Measured on gfx950, both layouts.
    q, k, v = inp["q"], inp["k"], inp["v"]

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

    tri_kernel, tri_ok, tri_err = math.nan, None, math.nan
    if triton_fn is not None:
        inp["pool"].copy_(pool0)
        tri_out, _ = tri()
        torch.cuda.synchronize()
        tri_ok, tri_err = check(inp, tri_out[0].unsqueeze(1), inp["pool"], pool0)
        inp["pool"].copy_(pool0)
        tri_kernel = kernel_us(tri)

    # What current main chooses with no KDA row. Bound directly because this
    # branch's wrapper finds the tuned row. The inputs differ only by config.
    inp["pool"].copy_(pool0)
    untuned_run, _ = flydsl_runner(inp, main_tiling(B))
    untuned_run()
    torch.cuda.synchronize()
    inp["pool"].copy_(pool0)
    untuned_kernel = kernel_us(untuned_run)

    def ratio(tri, fly):
        return math.nan if math.isnan(tri) else tri / fly

    return {
        "B": B,
        "tri_kernel": tri_kernel,
        "fly_kernel": fly_kernel,
        "kernel_speedup": ratio(tri_kernel, fly_kernel),
        # Kernel-axis aside: what KDA tuning buys over current main's policy.
        "untuned_kernel": untuned_kernel,
        "fly_ok": fly_ok,
        "fly_err": fly_err,
        "tri_ok": tri_ok,
        "tri_err": tri_err,
    }


_TIMED_KEYS = (
    "tri_kernel",
    "fly_kernel",
    "kernel_speedup",
    "untuned_kernel",
)


def mean_std(xs):
    """Mean and *sample* std, so one trial reports nan instead of a confident 0."""
    if any(math.isnan(x) for x in xs):
        return math.nan, math.nan
    return statistics.fmean(xs), statistics.stdev(xs) if len(xs) > 1 else math.nan


def bench_trials(B, flydsl_gdr_decode, triton_fn):
    """Repeat every measurement at one batch size over ``TRIALS`` trials.

    A fresh seed per trial, so the spread covers the input values and their
    allocation too, not only clock drift. Speedups average the per-trial ratio
    rather than dividing the two averages: the kernels are timed back to back
    within a trial, so the ratio is the paired quantity and its std is the one
    that says whether the win survives the noise.
    """
    runs = [bench_one(B, flydsl_gdr_decode, triton_fn, seed=t) for t in range(TRIALS)]

    row = {"B": B}
    for key in _TIMED_KEYS:
        row[key] = mean_std([r[key] for r in runs])

    # Parity has to hold in every trial, and the reported error is the worst one.
    row["fly_ok"] = all(r["fly_ok"] for r in runs)
    row["fly_err"] = max(r["fly_err"] for r in runs)
    row["tri_ok"] = (
        None if runs[0]["tri_ok"] is None else all(r["tri_ok"] for r in runs)
    )
    row["tri_err"] = max(r["tri_err"] for r in runs)
    return row


def print_table(caption, columns, table):
    """``columns`` is (heading, width, key) each, plus 'x' on any *_speedup key.

    Every cell but ``B`` is mean ± std over ``TRIALS`` trials.
    """
    print(f"\n{caption}\n{TRIALS} trials, mean ± std\n")
    header = " ".join(f"{head:>{w}}" for head, w, _ in columns)
    print(header)
    print("-" * len(header))
    for r in table:
        cells = []
        for _, w, key in columns:
            if key == "B":
                cells.append(f"{r['B']:>{w}}")
            else:
                mean, std = r[key]
                unit = "x" if key.endswith("speedup") else ""
                cells.append(f"{f'{mean:.2f}{unit} ± {std:.2f}':>{w}}")
        print(" ".join(cells))


def bench(args):
    """A/B at the ticket's batch sizes, both kernels on identical values.

    Device time only: vLLM graphs this kernel with others, so wall-clock per
    eager call is not the serving number. FlyDSL goes through the public wrapper,
    so the numbers are what a consumer gets, tuning table included.

    Every cell is repeated over ``TRIALS`` trials. The kernels sit within a few
    percent of each other, so a single shot cannot tell the gap from the drift.

    Both sides read K3's packed q/k/v, which is the ticket's "same inputs", and
    the sweep tunes against those same strides -- so the row the wrapper looks up
    describes the kernel being measured. The Triton column is its recurrence-only
    packed-decode fallback, not vLLM's HIP conv+recurrence+norm fusion.
    """
    from aiter.ops.flydsl import flydsl_gdr_decode

    arch = get_rocm_arch()
    triton_fn = load_triton(args.vllm)
    print(f"\narch {arch} · H={H} K=V={K} bf16 · f32 state · all times us")
    print("baseline: vLLM Triton packed decode (not HIP fused KDA decode)")

    table = [bench_trials(B, flydsl_gdr_decode, triton_fn) for B in BATCHES]

    print_table(
        "how fast the kernels are -- device time per call, host cost excluded",
        [
            ("B", 5, "B"),
            ("Triton", 14, "tri_kernel"),
            ("FlyDSL", 14, "fly_kernel"),
            ("speedup", 14, "kernel_speedup"),
            ("FlyDSL main tiling", 19, "untuned_kernel"),
        ],
        table,
    )

    print(
        f"\nparity vs the torch reference, worst of {TRIALS} trials"
        " (both must hold, or the timing is noise):"
    )
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
    p.add_argument("-o", "--output", help="write the winning table entry here")
    p.add_argument(
        "--vllm",
        help="path to a vLLM checkout (default: vllm_vllm-project/, then vllm/)",
    )
    args = p.parse_args()
    if not args.sweep and not args.bench:
        args.sweep = args.bench = True
    if args.sweep:
        sweep(args)
    if args.bench:
        bench(args)


if __name__ == "__main__":
    main()
