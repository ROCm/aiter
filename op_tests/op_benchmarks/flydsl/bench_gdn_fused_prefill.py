# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the single-kernel FlyDSL GDN prefill against the multi-kernel path.

Per shape, one GDN prefill layer on the same GPU and operands:

  fused    ``gdn_fused_prefill_fwd`` (one launch)
  opt_vk   ``chunk_gated_delta_rule_opt_vk`` with the FlyDSL prepare stage and
           each hidden-state backend in ``--backends``; the table reports the
           fastest one

Both run the serving layout: q/k/v as strided views of one packed
projection, l2norm in-op, and a bf16 state pool updated in place through
``cache_indices``. Sequences are equal length (T / N) and packed varlen.

Shapes default to the (H, Hg) x (T, N) grid of the Qwen3.5 tables in
``aiter/configs/model_configs/qwen3_5_*_chunk_gdn_h_opt_tuned.csv`` plus
``EXTRA_SHAPES``. Columns:

  us       summed device time of every kernel in the layer (torch profiler)
  wall us  per-call CUDA-event time, which also counts launch gaps
  rel_l2   worst sequence's o and final-state error against an fp32 chunked
           recurrence (checked against the token-by-token reference of
           op_tests/test_gdn_fused_prefill.py at startup)
  gate     ``gdn_fused_prefill_supported(..., max_seqlen=T / N)``

Usage, from the repo root (the ``op_tests`` import needs it on sys.path):

    python -m op_tests.op_benchmarks.flydsl.bench_gdn_fused_prefill
    python -m op_tests.op_benchmarks.flydsl.bench_gdn_fused_prefill \\
        --shape 8:2:16384:1 64:16:32768:16 --backends triton flydsl -o out.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import run_perftest
from op_tests.test_gdn_fused_prefill import (
    _opt_vk_candidate,
    _rel_l2,
    _seq_errors,
    run_torch_reference,
)

BT = 64
K = V = 128
TUNED_TABLES = "aiter/configs/model_configs/qwen3_5_*_chunk_gdn_h_opt_tuned.csv"
# (H, Hg, T, N) points worth covering beyond the tuned tables.
EXTRA_SHAPES = [(8, 2, 32768, 2)]


def tracked_shapes() -> list[tuple[int, int, int, int]]:
    """(H, Hg, T, N) from the Qwen3.5 tuned tables, plus ``EXTRA_SHAPES``."""
    shapes = set(EXTRA_SHAPES)
    for path in sorted(glob.glob(TUNED_TABLES)):
        with open(path, encoding="utf-8", newline="") as f:
            rows = csv.DictReader(line for line in f if not line.startswith("#"))
            for row in rows:
                shapes.add(
                    (int(row["H"]), int(row["Hg"]), int(row["T_flat"]), int(row["N"]))
                )
    return sorted(shapes)


def make_inputs(h, hg, t, n, seed=2026):
    """Packed-projection inputs; call under ``torch.device("cuda")``."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    mixed = torch.randn(t, 2 * hg * K + h * V, dtype=torch.bfloat16, generator=gen)
    q, k, v = torch.split(mixed, [hg * K, hg * K, h * V], dim=-1)
    g = (
        -torch.nn.functional.softplus(
            torch.randn(1, t, h, dtype=torch.float32, generator=gen)
        )
        * 0.5
    )
    beta = torch.randn(1, t, h, dtype=torch.float32, generator=gen).sigmoid()
    seq_lens = [t // n] * n
    cu = torch.arange(0, t + 1, t // n, dtype=torch.int32)
    pool = 0.05 * torch.randn(n + 1, h, V, K, dtype=torch.bfloat16, generator=gen)
    idx = torch.arange(1, n + 1, dtype=torch.int32)
    return (
        q.view(1, t, hg, K),
        k.view(1, t, hg, K),
        v.view(1, t, h, V),
        g,
        beta,
        cu,
        seq_lens,
        pool,
        idx,
    )


@torch.no_grad()
def chunked_reference(q, k, v, g, beta, cu, pool, idx):
    """fp32 chunked form of the gated delta rule. Returns o, final states."""
    h, rep = v.shape[2], v.shape[2] // q.shape[2]
    scale = K**-0.5

    def l2n(x):
        x = x.float()
        return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)

    # [H, T, D] per operand, [H, T] per gate.
    qf = l2n(q[0]).repeat_interleave(rep, dim=1).transpose(0, 1)
    kf = l2n(k[0]).repeat_interleave(rep, dim=1).transpose(0, 1)
    vf = v[0].float().transpose(0, 1)
    gf, bf = g[0].float().T, beta[0].float().T
    out = torch.empty(h, q.shape[1], V, dtype=torch.float32)
    finals = []
    for i in range(cu.numel() - 1):
        state = pool[int(idx[i])].float().clone()  # [H, V, K]
        for c0 in range(int(cu[i]), int(cu[i + 1]), BT):
            sl = slice(c0, min(c0 + BT, int(cu[i + 1])))
            qc, kc, vc, bc = qf[:, sl], kf[:, sl], vf[:, sl], bf[:, sl]
            gc = gf[:, sl].cumsum(-1)
            c = gc.shape[-1]
            causal = torch.ones(c, c, dtype=torch.bool).tril()
            diff = (gc[:, :, None] - gc[:, None, :]).masked_fill(~causal, -torch.inf)
            decay = diff.exp()  # exp(gc_t - gc_j) for j <= t, else 0
            a = (bc[:, :, None] * (kc @ kc.transpose(-1, -2)) * decay).tril(-1)
            rhs = bc[..., None] * (
                vc - gc.exp()[..., None] * (kc @ state.transpose(-1, -2))
            )
            eye = torch.eye(c).expand_as(a)
            vnew = torch.linalg.solve_triangular(eye + a, rhs, upper=False)
            out[:, sl] = scale * (
                gc.exp()[..., None] * (qc @ state.transpose(-1, -2))
                + ((qc @ kc.transpose(-1, -2)) * decay) @ vnew
            )
            g_last = gc[:, -1:]
            state = (
                g_last.exp()[..., None] * state
                + (vnew * (g_last - gc).exp()[..., None]).transpose(-1, -2) @ kc
            )
        finals.append(state)
    return out.transpose(0, 1).unsqueeze(0), torch.stack(finals)


def check_reference():
    """The chunked reference must match the token-by-token recurrence."""
    inputs = make_inputs(8, 2, 3 * 300, 3)
    q, k, v, g, beta, cu, _, pool, idx = inputs
    o_c, s_c = chunked_reference(q, k, v, g, beta, cu, pool, idx)
    track = torch.full((3,), -1, dtype=torch.int32)
    o_t, s_t, _ = run_torch_reference(q, k, v, g, beta, cu, pool, idx, track)
    err = max(_rel_l2(o_c, o_t), _rel_l2(s_c, s_t))
    assert err < 1e-4, f"chunked reference drifts from the recurrence: {err:.2e}"


def bench_shape(h, hg, t, n, backends):
    from aiter.ops.flydsl.gdn_fused_prefill import (
        _pick_bv,
        gdn_fused_prefill_fwd,
        gdn_fused_prefill_supported,
    )

    with torch.device("cuda"):
        q, k, v, g, beta, cu, seq_lens, pool0, idx = make_inputs(h, hg, t, n)
        o_ref, final_ref = chunked_reference(q, k, v, g, beta, cu, pool0, idx)

    def fused(pool):
        return gdn_fused_prefill_fwd(
            q, k, v, g, beta, ssm_states=pool, cache_indices=idx, cu_seqlens=cu
        )

    candidates = {"fused": fused}
    for backend in backends:
        candidates[backend] = _opt_vk_candidate(
            q, k, v, g, beta, cu, seq_lens, idx, backend
        )

    res = {}
    for name, fn in candidates.items():
        pool = pool0.clone()
        o_errs, s_errs, _ = _seq_errors(
            fn(pool), o_ref, pool, final_ref, cu, idx, range(n)
        )
        _, us = run_perftest(fn, pool0.clone(), num_iters=20, num_warmup=3)
        _, wall = run_perftest(
            fn, pool0.clone(), num_iters=20, num_warmup=3, use_cuda_event=True
        )
        res[name] = (us, wall, max(o_errs.values()), max(s_errs.values()))

    best = min(backends, key=lambda b: res[b][0])
    fu, bu = res["fused"], res[best]
    row = {
        "H": h,
        "Hg": hg,
        "N": n,
        "T": t,
        "fused WGs": (V // _pick_bv(n, h, q.device)) * n * h,
        "gate": gdn_fused_prefill_supported(q, k, v, g, beta, pool0, max_seqlen=t // n),
        "fused us": fu[0],
        "opt_vk us": bu[0],
        "opt_vk h": best,
        "speedup": bu[0] / fu[0],
        "fused wall us": fu[1],
        "opt_vk wall us": bu[1],
        "wall speedup": bu[1] / fu[1],
        "fused o rel_l2": fu[2],
        "fused state rel_l2": fu[3],
        "opt_vk o rel_l2": bu[2],
        "opt_vk state rel_l2": bu[3],
    }
    for backend in backends:
        row[f"{backend} us"] = res[backend][0]
    return row


def main():
    if get_gfx() != "gfx950":
        aiter.logger.warning("FlyDSL fused GDN prefill needs gfx950; skipping")
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Fused vs multi-kernel GDN prefill, same GPU",
    )
    parser.add_argument(
        "--shape",
        nargs="*",
        default=None,
        help="H:Hg:T:N points; default: the Qwen3.5 tuned-table grid",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["triton", "flydsl", "hip"],
        default=["triton", "flydsl", "hip"],
        help="opt_vk hidden-state backends; the fastest is reported",
    )
    parser.add_argument("-o", "--output", help="also write the table to this CSV")
    args = parser.parse_args()

    # The test module sets a default device on import, which routes every torch
    # call through a Python hook and inflates host time (hence wall us) by up
    # to ~100 us per layer. Inputs and references set it locally instead.
    torch.set_default_device(None)
    with torch.device("cuda"):
        check_reference()
    if args.shape:
        shapes = [tuple(int(x) for x in s.split(":")) for s in args.shape]
    else:
        shapes = tracked_shapes()
    rows = []
    for h, hg, t, n in shapes:
        rows.append(bench_shape(h, hg, t, n, args.backends))
        aiter.logger.info("done %s", rows[-1])
    df = pd.DataFrame(rows)
    if args.output:
        df.to_csv(os.path.abspath(args.output), index=False)
    aiter.logger.info(
        "gdn fused prefill vs opt_vk:\n%s", df.to_markdown(index=False, floatfmt=".3g")
    )


if __name__ == "__main__":
    main()
