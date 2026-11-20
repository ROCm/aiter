# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CUDA-graph latency of the hidden-5120 mHC seam and step-end tail vs existing paths.

Each candidate is captured as one graph of ``--graph_calls`` calls on distinct inputs
(so nothing stays cache-hot between calls) and replayed ``--replays`` times; the
candidates of a row are measured in alternating rounds (A B ... B A) and the table
reports the best and the median round, in us per call.

Seam (post-mix seam with carried pre gate, DeepSeek-V4.1 decode):
  flydsl            flydsl_mhc_post_pre_rmsnorm
  flydsl_mega_mhc   aiter.ops.flydsl.flydsl_mega_mhc
  hip_mega_mhc      aiter.ops.mega_mhc.mega_mhc (pre-packed fn)
  ``mxfp8+aux`` adds the auxiliary hidden state (mean of the four new streams): the
  existing paths need an ``aiter.mhc_post`` and a ``mean`` launch for it.

Tail (post-mix + collapse + final RMSNorm). ``post`` writes the layer input only,
``post+keep`` also the new residual R' (the existing paths always write it), and
``post+keep+aux`` also the mean of the four streams:
  flydsl            flydsl_mhc_post_collapse_rmsnorm
  chain             aiter.mhc_post + collapse + aiter.rms_norm, the collapse being
                    vLLM's ``hc_collapse_triton`` when vLLM is importable (else torch)
  triton_fused      vLLM's ``mhc_post_collapse_rms_norm_triton`` (if importable)

Run: python3 op_tests/op_benchmarks/flydsl/bench_mhc_post_rmsnorm.py [-t 16 64 ...]
"""

import argparse
import statistics

import pandas as pd
import torch

import aiter
from aiter.ops.flydsl import (
    flydsl_mega_mhc,
    flydsl_mhc_pack_fn,
    flydsl_mhc_post_collapse_rmsnorm,
    flydsl_mhc_post_pre_rmsnorm,
)
from aiter.ops.mega_mhc import mega_mhc, mega_mhc_shuffle_fn

H = 5120
EPS = 1e-6
POST_MULT = 2.0


def _graph(fn, n, stream):
    with torch.cuda.stream(stream):
        fn(0)
        fn(1 % n)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        for i in range(n):
            fn(i)
    return g


def _replay_us(g, n, replays):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(replays):
        g.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1000 / (replays * n)


def measure(cands, n, replays, rounds, stream):
    graphs = {k: _graph(f, n, stream) for k, f in cands.items()}
    times = {k: [] for k in graphs}
    order = list(graphs) + list(graphs)[::-1]
    for _ in range(rounds):
        for k in order:
            times[k].append(_replay_us(graphs[k], n, replays))
    del graphs
    return {k: (min(v), statistics.median(v)) for k, v in times.items()}


def _collapse():
    try:
        from vllm.model_executor.kernels.mhc.triton import (
            hc_collapse_triton,
            mhc_post_collapse_rms_norm_triton,
        )

        return hc_collapse_triton, mhc_post_collapse_rms_norm_triton
    except Exception:  # noqa: BLE001 - vLLM is optional here

        def collapse(x, pre):
            return torch.einsum("tj,tjh->th", pre, x.float()).bfloat16()

        return collapse, None


def _inputs(T, n, dev):
    return {
        "R": [torch.randn(T, 4, H, device=dev).bfloat16() for _ in range(n)],
        "y": [torch.randn(T, H, device=dev).bfloat16() for _ in range(n)],
        "RO": [
            torch.empty(T, 4, H, dtype=torch.bfloat16, device=dev) for _ in range(n)
        ],
        "pm": torch.rand(T, 4, 1, device=dev),
        "pre": torch.rand(T, 4, device=dev),
        "cm": torch.softmax(torch.randn(T, 4, 4, device=dev), -1),
    }


def _row(op, mode, T, res):
    row = {"op": op, "mode": mode, "T": T}
    for k, (best, med) in res.items():
        row[f"{k} best us"] = round(best, 2)
        row[f"{k} median us"] = round(med, 2)
    print(row, flush=True)
    return row


def bench_seam(T, mode, x, wts, cfg):
    """One seam row; ``mode`` is bf16, mxfp8 or mxfp8+aux."""
    R, y, RO, pm, pre, cm = (x[k] for k in ("R", "y", "RO", "pm", "pre", "cm"))
    fn, fn_fly, fn_hip, sc, base, w = wts
    gates = (EPS, EPS, EPS, POST_MULT, 20)
    od = "bf16" if mode == "bf16" else "mxfp8"
    aux = mode.endswith("aux")

    def served_aux(i):
        # the existing aux path: a separate post-mix and mean of the four streams
        r = torch.empty_like(R[i])
        aiter.mhc_post(r, y[i], R[i], pm, cm)
        r.mean(dim=1)

    def fly(i):
        return flydsl_mhc_post_pre_rmsnorm(
            R[i], fn_fly, sc, base, *gates, pre, y[i], pm, cm, w, EPS,
            residual_out=RO[i], out_dtype=od, aux_mean=aux,
        )  # fmt: skip

    def fly_mega(i):
        out = flydsl_mega_mhc(
            R[i], fn, sc, base, *gates, pre, y[i], pm, cm, w, EPS,
            residual_out=RO[i], out_dtype=od,
        )  # fmt: skip
        if aux:
            served_aux(i)
        return out

    def hip_mega(i):
        out = mega_mhc(
            R[i], fn_hip, sc, base, *gates, pre, y[i], pm, cm, w, EPS,
            residual_out=RO[i], w_preshuffle_bf16=True, out_dtype=od,
            scale_row_major=od == "mxfp8",
        )  # fmt: skip
        if aux:
            served_aux(i)
        return out

    cands = {"flydsl": fly, "flydsl_mega_mhc": fly_mega, "hip_mega_mhc": hip_mega}
    return _row("seam", mode, T, measure(cands, *cfg))


def bench_tail(T, mode, x, w, cfg):
    """One tail row; ``mode`` is post, post+keep or post+keep+aux."""
    R, y, RO, pm, pre, cm = (x[k] for k in ("R", "y", "RO", "pm", "pre", "cm"))
    collapse, triton_fused = _collapse()
    aux = mode.endswith("aux")
    keep = "keep" in mode

    def fly(i):
        return flydsl_mhc_post_collapse_rmsnorm(
            R[i], pre, w, EPS, y[i], pm, cm,
            residual_out=RO[i] if keep else None, aux_mean=aux,
        )  # fmt: skip

    def chain(i):
        aiter.mhc_post(RO[i], y[i], R[i], pm, cm)
        if aux:
            RO[i].mean(dim=1)
        return aiter.rms_norm(collapse(RO[i], pre), w, EPS)

    cands = {"flydsl": fly, "chain": chain}
    if triton_fused is not None and not aux:

        def fused(i):
            return triton_fused(R[i], y[i], pm, cm, pre, w, EPS)

        cands["triton_fused"] = fused
    return _row("tail", mode, T, measure(cands, *cfg))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "-t",
        "--tokens",
        type=int,
        nargs="*",
        default=[16, 32, 64, 128, 192, 256, 384, 512, 768],
    )
    p.add_argument("--ops", nargs="*", default=["seam", "tail"])
    p.add_argument("--graph_calls", type=int, default=50)
    p.add_argument("--replays", type=int, default=5)
    p.add_argument("--rounds", type=int, default=4)
    args = p.parse_args()

    torch.manual_seed(0)
    dev = torch.device("cuda")
    fn = torch.randn(24, 4 * H, device=dev) * 0.02
    w = torch.randn(H, device=dev).bfloat16()
    wts = (
        fn,
        flydsl_mhc_pack_fn(fn),
        mega_mhc_shuffle_fn(fn),
        torch.tensor([0.5, 0.7, 0.9], device=dev),
        torch.randn(24, device=dev) * 0.3,
        w,
    )
    cfg = (args.graph_calls, args.replays, args.rounds, torch.cuda.Stream())
    rows = []
    for T in args.tokens:
        x = _inputs(T, args.graph_calls, dev)
        if "seam" in args.ops:
            for mode in ("bf16", "mxfp8", "mxfp8+aux"):
                rows.append(bench_seam(T, mode, x, wts, cfg))
        if "tail" in args.ops:
            for mode in ("post", "post+keep", "post+keep+aux"):
                rows.append(bench_tail(T, mode, x, w, cfg))
        del x
    aiter.logger.info(
        "mhc seam / tail graph latency (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
