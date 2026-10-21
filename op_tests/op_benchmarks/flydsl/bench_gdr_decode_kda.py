#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tune the KDA FlyDSL GDR decode tiling and print a `_KDA_DECODE_BY_PART` row.

q/k/v stay views into one packed buffer. A config timed on dense inputs would
not match the kernel a consumer compiles.
"""

from __future__ import annotations

import argparse
import itertools
import math
import statistics
import sys
from pathlib import Path

import torch
import triton
from flydsl.runtime.device import get_rocm_arch

from aiter.ops.flydsl.kernels.gdr_decode import create_vk_gdr_decode_kernel
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, get_dtype_str
from aiter.ops.flydsl.linear_attention_kernels import _decode_tiling, get_num_sms

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

# Leaders land within a few percent, the same order as run-to-run drift, so
# the top TOP_N are re-timed over TRIALS and ranked on the median.
TOP_N = 5
TRIALS = 5

# What `_KDA_DECODE_BY_PART` holds, so the sweep emits a pasteable entry.
TABLE_NAME = "_KDA_DECODE_BY_PART"


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
    """One KDA decode case. q/k/v are views into one packed buffer.

    Indices start at 1, leaving slot 0 unused. A negative index is a
    graph-padding row and returns without writing.
    """
    torch.manual_seed(seed)
    packed = torch.randn(B, 2 * H * K + H * V, dtype=DTYPE, device=device)
    q = packed[:, : H * K].view(B, 1, H, K)
    k = packed[:, H * K : 2 * H * K].view(B, 1, H, K)
    v = packed[:, 2 * H * K :].view(B, 1, H, V)

    n_slots = B + 1
    pool = torch.randn(n_slots, H, V, K, dtype=STATE_DTYPE, device=device)
    indices = torch.arange(1, 1 + B, dtype=torch.int32, device=device)

    return {
        "q": q,
        "k": k,
        "v": v,
        "a": torch.randn(B, 1, H, K, dtype=DTYPE, device=device),
        "b": torch.randn(B, 1, H, dtype=DTYPE, device=device),
        "A_log": (
            torch.randn(H, dtype=torch.float32, device=device) * 0.5
        ).contiguous(),
        "dt_bias": (
            torch.randn(H, K, dtype=torch.float32, device=device) * 0.1
        ).contiguous(),
        "pool": pool,
        "indices": indices,
        "B": B,
    }


def flydsl_runner(inp, config):
    """Bind one explicit config, bypassing the wrapper's KDA tiling table.

    ``flydsl_gdr_decode`` resolves its config from ``_KDA_DECODE_BY_PART``, so a
    sweep must build the kernel directly. The kernel is compiled against the
    packed q/k/v strides.
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
    """RMSE-relative error, with an absolute-error escape hatch."""
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

    Also times the untuned config, the one `_decode_tiling()` picks with no
    tuned row, so the sweep reports what it bought and not just what it picked.
    Candidates come from this GPU's sweep -- the winning set differs between
    gfx942 and gfx950.
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
        tag = "   <-- untuned" if cfg == baseline_config else ""
        print(
            f"      {cfg!s:<12} median {med:8.2f}  min {lo:8.2f}  max {hi:8.2f}"
            f"  spread {(hi - lo) / med * 100:4.1f}%"
            f"  vs untuned {baseline / med:.2f}x{tag}"
        )
    best_med, _, _, best_cfg, best_err = rows[0]
    print(
        f"    winner {best_cfg} at {best_med:.2f} us"
        f" ({baseline / best_med:.2f}x the untuned config, err {best_err:.1e})\n"
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
    # evidence for it and are not stored. Keyed by CU count as well as arch,
    # because the split is chosen against this part's grid.
    entry = f'    ("{arch}", {get_num_sms()}): {{{", ".join(rows)}}},'
    print(f"\npaste into {TABLE_NAME}, replacing this part's row:\n")
    print(entry)
    if args.output:
        Path(args.output).write_text(entry + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")
    return entry


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-o", "--output", help="write the winning table entry here")
    sweep(p.parse_args())


if __name__ == "__main__":
    main()
