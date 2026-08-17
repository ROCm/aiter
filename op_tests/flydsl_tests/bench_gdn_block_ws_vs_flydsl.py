# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Block-level GDN prefill bench: the opus families vs the FlyDSL/Triton pipeline.

``test_flydsl_linear_attention_prefill.py::TestPerformance`` times K5 alone, on
inputs (``k``/``w``/``u``/``g_cumsum``) that already went through K1..K4.  This
script instead drives the whole block from ``q``/``k``/``v``/``g``/``beta`` so
the K1..K4 front end and the K6 output kernel are part of the measurement, and
splits the result per kernel via ``torch.profiler``.

The default shape is the ``varlen-64k-qwen-ptpc-ali`` group: full_prompt_len
8192, Hk 16 / Hv 64 at TP 8, i.e. Hg=2 key heads and H=8 value heads (GQA ratio
4), K=V=128, bf16, packed varlen with state I/O.  ``--n-seqs`` sweeps the
scheduler budget the same way that group's ``max_num_batched_tokens`` list does
(N sequences of full_prompt_len tokens each).

``--Hk`` / ``--Hv`` / ``--tp`` move the head counts.  Only the per-rank counts
Hg = Hk/TP and H = Hv/TP reach the kernels, so halving TP doubles both of them
and, with N fixed, doubles the N x H (sequence, head) chains the state scan has
to run -- which is what separates the fused and split families.  The GQA ratio
is Hv/Hk and the split has to be exact in both directions.

Usage:

    # every backend, then the comparison tables
    HIP_VISIBLE_DEVICES=7 python op_tests/flydsl_tests/bench_gdn_block_ws_vs_flydsl.py

    # one backend at a time, then report from the saved JSON
    python <this> --backend ws
    python <this> --backend cs
    python <this> --report --compare

    # the same case at another TP, kept in its own directory
    python <this> --tp 4 --outdir /tmp/gdn_block_bench_tp4

    # plain MHA instead of GQA 4 (Hk == Hv)
    python <this> --Hk 64 --Hv 64

Backends:
    ws       opus_gdn_wu_prefill_fwd, k2_mode=0, which packed varlen resolves to
             WS: one HIP kernel for K1..K4, then the split scan and output.
    wf       the same front end with k2_mode=1, the fused W/U K2 that carries
             the state scan and the output in one kernel.
    cf       opus_gdn_c_prefill_fwd, c_mode=1: the C-input front end (chunk
             inverse instead of W/U) and one fused scan/output kernel.
    cs       c_mode=2: the same front end, then the split C scan and the K6 that
             it shares with WS.
    flydsl   chunk_gated_delta_rule_opt_vk(use_chunk_flydsl=True): Triton
             prepare pair for K1..K4, FlyDSL K5, Triton K6.
    prepare  the same, plus use_prepare_flydsl=True: the Triton prepare pair is
             replaced by the fused FlyDSL K1..K4 kernel, so only K6 is Triton.
    triton   chunk_gated_delta_rule_opt_vk(use_chunk_flydsl=False), all Triton.

The C-input families need gfx942 and reject a sequence length that BT does not
divide, so a full-backend sweep skips them elsewhere; the default 8192-token
segments are aligned.  Only ``ws`` is what ``path="auto"`` would pick for a
packed batch, the other three opus columns are explicit-request territory.

Every ``chunk_gated_delta_rule_opt_vk`` backend gets a prebuilt
``prefill_metadata``, which is how these paths are driven in production and is a
precondition of the fused prepare kernel on a varlen batch (it sizes its
rectangular grid from the host-resident schedule).  opus takes ``cu_seqlens``
only and derives its own schedule on the host.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

# -- shape: varlen-64k-qwen-ptpc-ali ------------------------------------

D = 128  # K == V
# Model-level head counts and the tensor-parallel split. `--Hk` / `--Hv` / `--tp`
# override these; every kernel sees only the per-rank counts below.
HK, HV, TP = 16, 64, 8
HG = HK // TP  # 2 key heads per rank
H = HV // TP  # 8 value heads per rank
FULL_PROMPT_LEN = 8192

NUM_WARMUP = 5
NUM_ITERS = 50
PROF_ITERS = 20

BACKENDS = ("ws", "wf", "cf", "cs", "flydsl", "prepare", "triton")
LABEL = {
    "ws": "opus WS",
    "wf": "opus WF",
    "cf": "opus CF",
    "cs": "opus CS",
    "flydsl": "tri K14+fly K5",
    "prepare": "fly K14+fly K5",
    "triton": "triton only",
}
SHORT = {
    "ws": "WS",
    "wf": "WF",
    "cf": "CF",
    "cs": "CS",
    "flydsl": "fly",
    "prepare": "prep",
    "triton": "tri",
}

# Ordered (substring, stage) attribution rules; first match wins.  opus and the
# FlyDSL prepare kernel each fuse the front end into one kernel, so it is a
# single row for them.
RULES = (
    ("gdn_prepare", "K1-K4 fused"),
    ("gdn_k1_neumann_c_kernel", "K1-K3+C build"),
    ("gdn_k1_neumann_kernel", "K1-K4 fused"),
    ("gdn_k1_", "K1-K4 fused"),
    ("cumsum_scaled_dot_kkt", "K1+K2 cumsum/KKT"),
    ("merge_16x16_to_64x64_inverse", "K3 solve_tril"),
    ("solve_tril", "K3 solve_tril"),
    ("recompute_w_u", "K4 W/U"),
    ("chunk_gated_delta_rule_fwd_h_hip_kernel", "K5 state scan"),
    ("chunk_gdn_fwd_h_flydsl", "K5 state scan"),
    ("chunk_gated_delta_rule_fwd_kernel_h", "K5 state scan"),
    ("gdn_k2_scan", "K5 state scan"),
    ("gdn_k2_out_kernel", "K6 output"),
    ("chunk_fwd_kernel_o", "K6 output"),
)
# CF and CS launch the same gdn_k2_c_kernel symbol for different work, so the
# stage of a kernel is not a property of its name alone.  These win over RULES.
BACKEND_RULES = {
    "wf": (("gdn_k2_kernel", "K5+K6 fused"),),
    "cf": (("gdn_k2_c_kernel", "K5+K6 fused"),),
    "cs": (("gdn_k2_c_kernel", "K5 state scan"),),
}
STAGES = (
    "K1-K4 fused",
    "K1-K3+C build",
    "K1+K2 cumsum/KKT",
    "K3 solve_tril",
    "K4 W/U",
    "K5 state scan",
    "K5+K6 fused",
    "K6 output",
    "other",
)
# Everything ahead of the state scan.  The W/U families spend it on W and U, the
# C ones on the chunk inverse, so only the total is comparable across them.
FRONT_END = (
    "K1-K4 fused",
    "K1-K3+C build",
    "K1+K2 cumsum/KKT",
    "K3 solve_tril",
    "K4 W/U",
)


def stage_of(name: str, backend: str | None = None) -> str:
    for needle, stage in BACKEND_RULES.get(backend, ()):
        if needle in name:
            return stage
    for needle, stage in RULES:
        if needle in name:
            return stage
    return "other"


# -- shape plumbing -----------------------------------------------------


def resolve_heads(hk: int, hv: int, tp: int) -> tuple[int, int]:
    """Per-rank (key, value) head counts, with the divisibility the kernels need."""
    if hk <= 0 or hv <= 0 or tp <= 0:
        raise SystemExit(f"--Hk, --Hv and --tp must be positive, got {(hk, hv, tp)}")
    if hk % tp or hv % tp:
        raise SystemExit(
            f"--Hk and --Hv must both be divisible by --tp, got Hk={hk} Hv={hv} tp={tp}"
        )
    hg, h = hk // tp, hv // tp
    # GQA shares one key head across H/Hg value heads, so the split has to be exact.
    if h % hg:
        raise SystemExit(
            f"value heads per rank ({h}) must be a multiple of key heads per rank "
            f"({hg}); got Hk={hk} Hv={hv} tp={tp}"
        )
    return hg, h


def shape_record() -> dict:
    """The shape a saved JSON was measured at, so a report never guesses it."""
    return dict(
        Hk=HK,
        Hv=HV,
        tp=TP,
        Hg=HG,
        H=H,
        K=D,
        V=D,
        full_prompt_len=FULL_PROMPT_LEN,
    )


# -- inputs -------------------------------------------------------------


def build_inputs(n_seqs: int) -> dict:
    """Packed varlen inputs: n_seqs segments of FULL_PROMPT_LEN tokens.

    Mirrors ``_build_context_lens(full_prompt_len, max_num_batched_tokens)``,
    which slices the scheduler budget into equal full_prompt_len segments.  q/k
    are l2-normalised and g is a log-sigmoid decay, matching what the
    GatedDeltaNet block feeds the kernel.  The seed depends only on n_seqs, so
    separate backend processes see bit-identical inputs.
    """
    from aiter.ops.prefill_batch_metadata import (
        build_gated_delta_rule_prefill_metadata,
    )

    torch.manual_seed(20260811 + n_seqs)
    lens = [FULL_PROMPT_LEN] * n_seqs
    total = sum(lens)
    cu = torch.tensor(
        [0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32, device="cuda"
    )
    # Built once per shape, as a serving layer would, and reused by every
    # timed iteration.
    meta = build_gated_delta_rule_prefill_metadata(lens, cu_seqlens=cu, chunk_size=64)
    q = F.normalize(torch.randn(1, total, HG, D, device="cuda"), dim=-1).to(
        torch.bfloat16
    )
    k = F.normalize(torch.randn(1, total, HG, D, device="cuda"), dim=-1).to(
        torch.bfloat16
    )
    v = (torch.randn(1, total, H, D, device="cuda") * 0.1).to(torch.bfloat16)
    g = F.logsigmoid(torch.randn(1, total, H, device="cuda", dtype=torch.float32))
    beta = torch.sigmoid(torch.randn_like(g)).to(torch.bfloat16)
    h0 = torch.randn(n_seqs, H, D, D, device="cuda", dtype=torch.float32) * 0.01
    return dict(q=q, k=k, v=v, g=g, beta=beta, h0=h0, cu=cu, meta=meta, total=total)


def unsupported_reason(backend: str) -> str:
    """Why this device cannot run the backend, or "" when it can."""
    if backend in ("cf", "cs"):
        gfx = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
        if gfx != "gfx942":
            return f"the C-input families require gfx942, this is {gfx}"
    if backend in ("cf", "cs") and FULL_PROMPT_LEN % 64:
        return f"the C-input families require 64 | full_prompt_len={FULL_PROMPT_LEN}"
    return ""


def make_callable(backend: str, t: dict):
    if backend in ("ws", "wf"):
        from aiter.ops.opus_gdn_wu_prefill import (
            OPUS_GDN_K2_SPLIT,
            OPUS_GDN_K2_WU_FUSED,
            opus_gdn_wu_prefill_fwd,
        )

        # Ask for the family by name: a packed batch resolves k2_mode=0 to WS, so
        # leaving it on auto would make the wf column a copy of the ws one.
        k2_mode = OPUS_GDN_K2_SPLIT if backend == "ws" else OPUS_GDN_K2_WU_FUSED

        def run():
            return opus_gdn_wu_prefill_fwd(
                t["q"],
                t["k"],
                t["v"],
                t["g"],
                t["beta"],
                initial_state=t["h0"],
                output_final_state=True,
                k2_mode=k2_mode,
                use_env_overrides=False,
                cu_seqlens=t["cu"],
            )

        return run

    if backend in ("cf", "cs"):
        from aiter.ops.opus_gdn_c_prefill import (
            OPUS_GDN_C_FUSED,
            OPUS_GDN_C_SPLIT,
            opus_gdn_c_prefill_fwd,
        )

        c_mode = OPUS_GDN_C_FUSED if backend == "cf" else OPUS_GDN_C_SPLIT

        def run():
            return opus_gdn_c_prefill_fwd(
                t["q"],
                t["k"],
                t["v"],
                t["g"],
                t["beta"],
                initial_state=t["h0"],
                output_final_state=True,
                c_mode=c_mode,
                use_env_overrides=False,
                cu_seqlens=t["cu"],
            )

        return run

    from aiter.ops.triton.gated_delta_net import chunk_gated_delta_rule_opt_vk

    use_chunk_flydsl = backend in ("flydsl", "prepare")
    use_prepare_flydsl = backend == "prepare"

    def run():
        return chunk_gated_delta_rule_opt_vk(
            q=t["q"],
            k=t["k"],
            v=t["v"],
            g=t["g"],
            beta=t["beta"],
            initial_state=t["h0"],
            output_final_state=True,
            cu_seqlens=t["cu"],
            use_chunk_flydsl=use_chunk_flydsl,
            use_prepare_flydsl=use_prepare_flydsl,
            prefill_metadata=t["meta"],
        )

    return run


# -- measurement --------------------------------------------------------


def bench_wall_us(run) -> float:
    """Median per-call wall time.  Median, not mean: the first iterations of a
    sweep can still pick up autotune/cache effects."""
    for _ in range(NUM_WARMUP):
        run()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_ITERS)]
    for i in range(NUM_ITERS):
        starts[i].record()
        run()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))
    return times[len(times) // 2]


def profile_kernels(run) -> dict[str, float]:
    """Per-kernel device time (us per call), keyed by profiler symbol."""
    for _ in range(NUM_WARMUP):
        run()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(PROF_ITERS):
            run()
        torch.cuda.synchronize()
    out: dict[str, float] = {}
    for evt in prof.key_averages():
        if evt.device_type is None or "cuda" not in str(evt.device_type).lower():
            continue
        us = evt.self_device_time_total / PROF_ITERS
        if us > 0.0:
            out[evt.key] = out.get(evt.key, 0.0) + us
    return out


def run_backend(backend: str, n_list: list[int], outdir: str) -> str:
    import aiter

    os.makedirs(outdir, exist_ok=True)
    props = torch.cuda.get_device_properties(0)
    rows = []
    for n in n_list:
        t = build_inputs(n)
        run = make_callable(backend, t)
        o, final_state = run()
        if n == n_list[0]:
            # Kept so --compare can diff the backends' outputs across processes.
            torch.save(
                {"o": o.detach().clone(), "final_state": final_state.detach().clone()},
                os.path.join(outdir, f"out_{backend}_n{n}.pt"),
            )
        wall = bench_wall_us(run)
        kernels = profile_kernels(run)
        if backend == "prepare" and not any("gdn_prepare" in name for name in kernels):
            # opt_vk falls back to the Triton prepare pair rather than failing
            # when the request is outside the fused kernel's slice, which would
            # silently make this column a copy of the flydsl one.
            raise RuntimeError(
                f"n={n}: the fused FlyDSL prepare kernel did not run; "
                f"opt_vk fell back to Triton.  kernels: {sorted(kernels)}"
            )
        if backend in ("cf", "cs"):
            # CF and CS launch the same K2-C symbol, and the stage table tells
            # them apart by whether the shared K6 follows it.  Check that here
            # rather than trusting the requested c_mode.
            has_k6 = any("gdn_k2_out_kernel" in name for name in kernels)
            if (backend == "cs") != has_k6:
                raise RuntimeError(
                    f"n={n}: {backend} {'did not run' if backend == 'cs' else 'ran'}"
                    f" the shared K6.  kernels: {sorted(kernels)}"
                )
        rows.append(
            dict(
                backend=backend,
                n_seqs=n,
                total_tokens=t["total"],
                wall_us=wall,
                kernel_sum_us=sum(kernels.values()),
                kernels=kernels,
            )
        )
        print(
            f"[{backend}] n={n} T={t['total']:6d} wall={wall:9.1f}us "
            f"kernels={sum(kernels.values()):9.1f}us ({len(kernels)} distinct)",
            flush=True,
        )
    path = os.path.join(outdir, f"bench_{backend}.json")
    with open(path, "w") as fh:
        json.dump(
            dict(
                backend=backend,
                aiter_path=aiter.__file__,
                gfx=props.gcnArchName,
                cus=props.multi_processor_count,
                shape=shape_record(),
                rows=rows,
            ),
            fh,
            indent=1,
        )
    print(f"wrote {path}")
    return path


# -- reporting ----------------------------------------------------------


def buckets(row: dict, backend: str) -> dict[str, float]:
    out = {s: 0.0 for s in STAGES}
    for name, us in row["kernels"].items():
        out[stage_of(name, backend)] += us
    return out


def per_case_table(data: dict, present: list[str], i: int, n: int) -> None:
    """One table per case: a row per backend, a column per pipeline stage.

    ``front`` is the pre-scan total -- K1..K4 for the W/U and Triton backends,
    the chunk-inverse front end for the C ones -- so it is comparable across
    backends whether they split those stages or fuse them; the
    ``K1+K2``/``K3``/``K4`` columns then break it down for the ones that split.
    WF and CF fuse the state scan with the output, so they report ``K5+K6``
    where the split families report ``K5`` and ``K6``.  ``total`` is the sum of
    the profiler's per-kernel device time and ``wall`` the median end to end, so
    their difference is the launch gap.
    """
    rows = {b: data[b]["rows"][i] for b in present}
    bk = {b: buckets(rows[b], b) for b in present}
    cols = (
        ("K1+K2", 8, lambda b: bk[b]["K1+K2 cumsum/KKT"]),
        ("K3", 8, lambda b: bk[b]["K3 solve_tril"]),
        ("K4", 8, lambda b: bk[b]["K4 W/U"]),
        ("front", 9, lambda b: sum(bk[b][s] for s in FRONT_END)),
        ("K5", 9, lambda b: bk[b]["K5 state scan"]),
        ("K6", 9, lambda b: bk[b]["K6 output"]),
        ("K5+K6", 9, lambda b: bk[b]["K5+K6 fused"]),
        ("other", 7, lambda b: bk[b]["other"]),
        ("total", 9, lambda b: rows[b]["kernel_sum_us"]),
        ("wall", 9, lambda b: rows[b]["wall_us"]),
    )
    print(f"\n== N={n}, T={rows[present[0]]['total_tokens']} (us) ==")
    hdr = f"{'scheme':<15}" + "".join(f"{name:>{w}}" for name, w, _ in cols)
    if "ws" in rows:
        hdr += f"{'vs WS':>8}"
    print(hdr)
    print("-" * len(hdr))
    for b in present:
        line = f"{LABEL[b]:<15}"
        for _, w, get in cols:
            us = get(b)
            line += f"{us:{w}.1f}" if us else f"{'-':>{w}}"
        if "ws" in rows:
            ratio = rows[b]["wall_us"] / rows["ws"]["wall_us"]
            line += f"{ratio:7.2f}x"
        print(line)


def report(outdir: str) -> None:
    data = {}
    for b in BACKENDS:
        path = os.path.join(outdir, f"bench_{b}.json")
        if os.path.exists(path):
            data[b] = json.load(open(path))
    if not data:
        print(f"no bench_*.json under {outdir}")
        return
    present = [b for b in BACKENDS if b in data]
    ref = data[present[0]]
    n_list = [r["n_seqs"] for r in ref["rows"]]

    # One directory holds one shape. Now that --Hk / --Hv / --tp can move it, a
    # leftover JSON from an earlier shape would otherwise be tabulated as if it
    # were the same measurement.
    stale = [b for b in present if data[b]["shape"] != ref["shape"]]
    if stale:
        print(
            f"dropping {', '.join(stale)}: measured at a different shape than "
            f"{SHORT[present[0]]}.  Use one --outdir per shape."
        )
        for b in stale:
            del data[b]
        present = [b for b in BACKENDS if b in data]

    shape = ref["shape"]
    print(f"\ngfx={ref['gfx']}  CUs={ref['cus']}  shape={shape}")
    print(
        f"full_prompt_len={shape['full_prompt_len']}, "
        f"Hk={shape.get('Hk', '?')} Hv={shape.get('Hv', '?')} at TP "
        f"{shape.get('tp', '?')} => Hg={shape['Hg']} H={shape['H']}, "
        f"K=V={shape['K']} bf16, packed varlen, state I/O on"
    )
    for b in present:
        print(f"  {SHORT[b]:>4}  {LABEL[b]:<14} <- {data[b]['aiter_path']}")
    print(
        f"\nwall = median of {NUM_ITERS} iters; per-kernel = {PROF_ITERS}-iter "
        "torch.profiler device time"
    )
    print(
        "opt_vk backends run with a prebuilt prefill_metadata; opus derives its "
        "own schedule from cu_seqlens\n"
    )

    # Short names here: seven backends fit on a line, and the legend above
    # already spells each one out.
    print("== end to end (us) ==")
    hdr = f"{'N':>2} {'T':>6} " + " ".join(f"{SHORT[b]:>10}" for b in present)
    if "ws" in data:
        hdr += "".join(f"{SHORT[b] + '/WS':>10}" for b in present if b != "ws")
    print(hdr)
    print("-" * len(hdr))
    for i, n in enumerate(n_list):
        walls = {b: data[b]["rows"][i]["wall_us"] for b in present}
        line = f"{n:2d} {ref['rows'][i]['total_tokens']:6d} " + " ".join(
            f"{walls[b]:10.1f}" for b in present
        )
        if "ws" in data:
            line += "".join(
                f"{walls[b] / walls['ws']:9.2f}x" for b in present if b != "ws"
            )
        print(line)

    for i, n in enumerate(n_list):
        per_case_table(data, present, i, n)

    print(f"\n== raw kernel names, N={n_list[0]} ==")
    for b in present:
        print(f"-- {LABEL[b]}")
        for name, us in sorted(
            data[b]["rows"][0]["kernels"].items(), key=lambda x: -x[1]
        ):
            print(f"   {us:9.1f}  [{stage_of(name, b)}]  {name[:110]}")


def compare(outdir: str, n: int) -> None:
    """Max/mean abs diff between the backends' saved outputs."""
    saved = {}
    for b in BACKENDS:
        path = os.path.join(outdir, f"out_{b}_n{n}.pt")
        if os.path.exists(path):
            saved[b] = torch.load(path)
    if len(saved) < 2:
        print(f"need at least two out_*_n{n}.pt under {outdir}")
        return
    base = next(iter(saved))
    print(f"\n== numeric agreement vs {LABEL[base]}, N={n} ==")
    print(f"   |o| mean = {saved[base]['o'].float().abs().mean():.6f}")
    for b, d in saved.items():
        if b == base:
            continue
        do = (saved[base]["o"].float() - d["o"].float()).abs()
        df = (saved[base]["final_state"].float() - d["final_state"].float()).abs()
        print(
            f"   {LABEL[b]:<14} o max={do.max():.6f} mean={do.mean():.8f} | "
            f"final_state max={df.max():.6f} mean={df.mean():.8f}"
        )


# -- entry point --------------------------------------------------------


def main() -> int:
    global HK, HV, TP, HG, H

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--backend",
        choices=BACKENDS,
        help="bench a single backend instead of all of them",
    )
    ap.add_argument(
        "--report", action="store_true", help="report from existing JSON only"
    )
    ap.add_argument("--compare", action="store_true", help="diff the saved outputs")
    ap.add_argument("--n-seqs", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument(
        "--tp", type=int, default=TP, help=f"tensor-parallel split (default {TP})"
    )
    ap.add_argument(
        "--Hk", type=int, default=HK, help=f"model key head count (default {HK})"
    )
    ap.add_argument(
        "--Hv", type=int, default=HV, help=f"model value head count (default {HV})"
    )
    ap.add_argument(
        "--outdir",
        default="/tmp/gdn_block_bench",
        help="one directory per shape; a report drops JSON from another shape",
    )
    args = ap.parse_args()

    HK, HV, TP = args.Hk, args.Hv, args.tp
    HG, H = resolve_heads(HK, HV, TP)

    if args.report or args.compare:
        if args.report:
            report(args.outdir)
        if args.compare:
            compare(args.outdir, args.n_seqs[0])
        return 0

    if args.backend:
        # An explicit request runs even when it is expected to fail, so the
        # error is the wrapper's own rather than a silent skip.
        run_backend(args.backend, args.n_seqs, args.outdir)
    else:
        for backend in BACKENDS:
            reason = unsupported_reason(backend)
            if reason:
                print(f"[{backend}] skipped: {reason}", flush=True)
                continue
            run_backend(backend, args.n_seqs, args.outdir)
    report(args.outdir)
    compare(args.outdir, args.n_seqs[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
