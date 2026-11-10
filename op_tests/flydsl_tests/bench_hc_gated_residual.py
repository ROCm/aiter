# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Latency benchmark for the two-stage HC Gated-Residual operator.

Times fused two-stage ``combine_and_mix`` with a pre-folded weight across a token
sweep and validates each shape against the float32 oracle. The seam graph mode
uses prepared or ordinary weights; --deferred compares the partial-only
deferred K1 with baseline seam dispatches for kernel-trace attribution.

    HIP_VISIBLE_DEVICES=2 python op_tests/flydsl_tests/bench_hc_gated_residual.py
    HIP_VISIBLE_DEVICES=2 python op_tests/flydsl_tests/bench_hc_gated_residual.py --tokens 512 4096 8192
"""

import argparse
import ctypes
import ctypes.util
import statistics
import sys

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

from aiter.jit.utils.chip_info import get_gfx, get_gfx_runtime
from aiter.ops.flydsl.hc_seam import (
    hc_gated_combine,
    hc_seam,
    hc_seam_prepare_weights,
    hc_seam_prepared,
)
from aiter.ops.flydsl.kernels.hyper_connection_gated_residual import (
    flydsl_gr_two_stage_combine_and_mix,
    fold_norm_weight,
    gr_combine_and_mix,
    merge_gr_two_stage_weight,
)
from aiter.ops.flydsl.kernels.hyper_connection_gated_residual.k1 import (
    _build_down_deferred_partial_pipe,
)
from aiter.ops.flydsl.kernels.hyper_connection_gated_residual.reference import (
    seam_deferred_partials_ref,
)
from aiter.ops.flydsl.kernels.hyper_connection_gated_residual.tuned import seam_config

HC, HS, LOWRANK = 4, 2560, 320
HIDDEN = HC * HS
EPS = 1e-6


def make_inputs(tokens, seed=0):
    torch.manual_seed(seed)

    def _rand(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda") * scale).bfloat16()

    inp = {
        "residual": _rand(tokens, HIDDEN),
        "block_output": _rand(tokens, HS),
        "injection": _rand(tokens, HC),
        "norm_weight": _rand(HS, scale=0.1),
        "w_down": _rand(LOWRANK, HIDDEN, scale=HIDDEN**-0.5),
        "w_up": _rand(HIDDEN, LOWRANK, scale=LOWRANK**-0.5),
        "w_inject": _rand(HC, HIDDEN, scale=HIDDEN**-0.5),
    }
    merged = merge_gr_two_stage_weight(inp["w_down"], inp["w_inject"], HC)
    inp["_w_folded"] = fold_norm_weight(merged, inp["norm_weight"], HC)
    return inp


def _run(inp):
    return flydsl_gr_two_stage_combine_and_mix(
        inp["residual"],
        inp["block_output"],
        inp["injection"],
        inp["norm_weight"],
        inp["w_down"],
        inp["w_up"],
        inp["w_inject"],
        HC,
        EPS,
        w_down_merged=inp["_w_folded"],
        fold_w=True,
    )


def _validate(inp, tokens):
    _r2, x, _inj = _run(inp)
    torch.cuda.synchronize()
    _r2o, xo, _injo = gr_combine_and_mix(
        inp["residual"],
        inp["block_output"],
        inp["injection"],
        inp["norm_weight"],
        inp["w_down"],
        inp["w_up"],
        inp["w_inject"],
        HC,
        EPS,
        round_bf16=True,
    )
    ex = (x.float() - xo).abs().max().item()
    if x.isnan().any() or ex > 0.2:
        print(
            f"  [WARN] M={tokens}: x max-abs-err {ex:.3f} (nan={x.isnan().any().item()})"
        )


def _time_us(inp, iters, warmup_s):
    import time

    torch.cuda.synchronize()
    deadline = time.perf_counter() + warmup_s
    while time.perf_counter() < deadline:
        _run(inp)
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        s.record()
        _run(inp)
        e.record()
        e.synchronize()
        samples.append(s.elapsed_time(e) * 1.0e3)
    return statistics.median(samples)


# Full-call graph timing includes runtime weight preparation.
BENCH_MODES = (
    "seam",
    "prepared",
    "legacy",
    "separate",
    "combine-only",
    "mix-only",
    "all",
)


def seam_make_inputs(rows, *, combine, next_gate, seed, w_scale):
    torch.manual_seed(seed)

    def _rand(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda") * scale).bfloat16()

    gate = 2.0 * torch.sigmoid(torch.randn(rows, HC, device="cuda"))
    return {
        "y": _rand(rows, HS) if combine else None,
        "residual": _rand(rows, HIDDEN, scale=3.0),
        "gate_prev": gate.contiguous() if combine else None,
        "norm_weight": _rand(HIDDEN, scale=0.1 * w_scale),
        "w_down": _rand(LOWRANK, HIDDEN, scale=0.04 * w_scale),
        "w_up": _rand(HIDDEN, LOWRANK, scale=0.1 * w_scale),
        "w_inject": _rand(HC, HIDDEN, scale=0.04 * w_scale) if next_gate else None,
    }


def seam_arm_kernel(inp):
    return hc_seam(
        inp["y"],
        inp["residual"],
        inp["gate_prev"],
        inp["norm_weight"],
        inp["w_down"],
        inp["w_up"],
        inp["w_inject"],
        EPS,
    )


def seam_arm_combine_only(inp):
    if inp["y"] is None:
        raise NotImplementedError("combine-only requires a pending block output")
    return hc_gated_combine(inp["y"], inp["residual"], inp["gate_prev"])


def seam_arm_mix_only(inp):
    return seam_arm_kernel({**inp, "y": None, "gate_prev": None})


def seam_arm_separate(inp):
    residual = seam_arm_combine_only(inp) if inp["y"] is not None else inp["residual"]
    return seam_arm_mix_only({**inp, "residual": residual})


def seam_arm_legacy(inp):
    if inp["y"] is None:
        raise NotImplementedError("legacy fused arm requires combine")
    # Raw BF16 logits are prepared outside the graph, not reinterpreted as gates.
    return flydsl_gr_two_stage_combine_and_mix(
        inp["residual"],
        inp["y"],
        inp["_raw_injection"],
        inp["norm_weight"],
        inp["w_down"],
        inp["w_up"],
        inp["w_inject"],
        HC,
        EPS,
    )


def seam_arm_prepared(inp):
    # Weights were packed outside the captured region (inp["_prepared"]).
    return hc_seam_prepared(
        inp["y"],
        inp["residual"],
        inp["gate_prev"],
        inp["_prepared"],
        EPS,
        **inp.get("_geometry", {}),
    )


SEAM_ARMS = {
    "legacy": seam_arm_legacy,
    "separate": seam_arm_separate,
    "seam": seam_arm_kernel,
    "prepared": seam_arm_prepared,
    "combine-only": seam_arm_combine_only,
    "mix-only": seam_arm_mix_only,
}


def _graph_node_counts(graph):
    """(kernel nodes, memcpy+memset nodes) in a captured graph."""
    hip = ctypes.CDLL(ctypes.util.find_library("amdhip64"))
    get_nodes = hip.hipGraphGetNodes
    get_nodes.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    get_type = hip.hipGraphNodeGetType
    get_type.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    count = ctypes.c_size_t()
    handle = ctypes.c_void_p(graph.raw_cuda_graph())
    if get_nodes(handle, None, ctypes.byref(count)):
        raise RuntimeError("hipGraphGetNodes failed")
    nodes = (ctypes.c_void_p * count.value)()
    if get_nodes(handle, nodes, ctypes.byref(count)):
        raise RuntimeError("hipGraphGetNodes failed")
    kernels = copies = 0
    for node in nodes:
        kind = ctypes.c_int()
        if get_type(node, ctypes.byref(kind)):
            raise RuntimeError("hipGraphNodeGetType failed")
        kernels += kind.value == 0  # hipGraphNodeTypeKernel
        copies += kind.value in (1, 2)  # hipGraphNodeTypeMemcpy, ...Memset
    return kernels, copies


def seam_main(args):
    """Unprofiled graph median per boundary, with distinct layer weights.

    A single-boundary run is cache-resident sanity timing, not HBM ranking.
    """
    if get_gfx_runtime() != "gfx950":
        raise NotImplementedError("finished-gate seam benchmark requires gfx950")
    if args.boundaries < 1 or args.replays < 1:
        raise ValueError("boundaries and replays must be positive")
    modes = (
        ("legacy", "seam", "prepared", "separate")
        if args.bench_mode == "all"
        else (args.bench_mode,)
    )
    for rows in args.rows:
        layers = [
            seam_make_inputs(
                rows,
                combine=args.combine,
                next_gate=args.next_gate,
                seed=args.seed + i,
                w_scale=args.weight_scale,
            )
            for i in range(args.boundaries)
        ]
        print(
            f"rows={rows} mode={args.bench_mode} combine={int(args.combine)} "
            f"next_gate={int(args.next_gate)} seed={args.seed} "
            f"w_scale={args.weight_scale} boundaries={args.boundaries} "
            f"replays={args.replays}"
        )
        for inp in layers:
            if inp["gate_prev"] is not None:
                inp["_raw_injection"] = (
                    HC * torch.logit(inp["gate_prev"] * 0.5)
                ).bfloat16()
        for inp in layers:
            # Packed once per layer, outside every captured region.
            inp["_prepared"] = hc_seam_prepare_weights(
                inp["norm_weight"], inp["w_down"], inp["w_up"], inp["w_inject"]
            )
        for inp in layers:
            inp["_geometry"] = {
                "row_tile": args.row_tile,
                "k1_row_tile": args.k1_row_tile,
                "k2_row_tile": args.k2_row_tile,
                "fallback": args.fallback,
                "deferred": args.seam_deferred,
                "k2_branch_mode": args.k2_branch_mode,
                "k2_slp": None if args.k2_slp is None else bool(args.k2_slp),
                "k2_n_tile": args.k2_n_tile,
                "k2_m_waves": args.k2_m_waves,
                "k2_n_waves": args.k2_n_waves,
            }
        from aiter.ops.flydsl.kernels.hyper_connection_gated_residual.tuned import (
            seam_config,
            seam_deferred_default,
            seam_deferred_k1_config,
            seam_deferred_k2_config,
            seam_k2_branch_config,
            seam_k2_default,
        )

        config = seam_config(
            "gfx950",
            rows,
            args.combine,
            args.next_gate,
            layers[0]["_prepared"].merged.shape[0],
        )
        if not (args.row_tile or args.k1_row_tile or args.k2_row_tile or args.fallback):
            if args.k2_branch_mode is None:
                config["k2"].update(seam_k2_default(rows)[1])
            else:
                config["k2"].update(seam_k2_branch_config(rows, args.k2_branch_mode))
        effective_deferred = args.seam_deferred or (
            seam_deferred_default(rows)
            and not any(
                (
                    args.row_tile,
                    args.k1_row_tile,
                    args.k2_row_tile,
                    args.fallback,
                    args.k2_branch_mode,
                    args.k2_n_tile,
                    args.k2_m_waves,
                    args.k2_n_waves,
                )
            )
        )
        if effective_deferred:
            config["k1"] = seam_deferred_k1_config(rows)
            if args.k2_branch_mode is None:
                config["k2"] = seam_deferred_k2_config(rows)[1]
        tile = max(config["k1"]["block_m"], config["k2"]["block_m"])
        if args.row_tile or args.k1_row_tile or args.k2_row_tile:
            tile = (
                max(
                    args.k1_row_tile or args.row_tile or 0,
                    args.k2_row_tile or args.row_tile or 0,
                )
                or tile
            )
        print(
            f"geometry row_tile={tile} padded={((rows + tile - 1) // tile) * tile} "
            f"forced={args.row_tile is not None} fallback={args.fallback} "
            f"k2_branch_mode={args.k2_branch_mode} deferred={effective_deferred} config={config}"
        )
        torch.cuda.synchronize()
        for mode in modes:
            arm = SEAM_ARMS[mode]
            label = (
                "full-call (includes packing)"
                if mode in ("legacy", "seam", "separate")
                else mode
            )
            for inp in layers:
                arm(inp)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph):
                results = [arm(inp) for inp in layers]
            kernel_nodes, copy_nodes = _graph_node_counts(graph)
            graph.instantiate()
            if kernel_nodes == 0:
                raise RuntimeError("timing graph has no kernel nodes")
            for _ in range(10):
                graph.replay()
            torch.cuda.synchronize()
            # Batched replays, one terminal sync: per-replay events would add
            # host round-trips that the SGLang protocol does not pay.
            samples = []
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(args.trials):
                start.record()
                for _ in range(args.replays):
                    graph.replay()
                end.record()
                end.synchronize()
                samples.append(
                    start.elapsed_time(end) * 1000 / args.replays / args.boundaries
                )
            print(
                f"M={rows} arm={mode} [{label}] us_per_boundary="
                f"{statistics.median(samples):.3f} "
                f"kernel_nodes={kernel_nodes / args.boundaries:g} "
                f"memcpy_memset_nodes={copy_nodes / args.boundaries:g}"
            )
            del results, graph


def deferred_main(args):
    """Deferred K1 partials versus default combine+down in a same-process trace.

    The baseline graph contains the full seam so rocprof can attribute its
    combine and down dispatches. The deferred graph contains only one partial
    dispatch; these graph totals are deliberately not a full-call comparison.
    Single-weight graphs are cache-resident diagnostics, not HBM admission.
    """
    if get_gfx_runtime() != "gfx950":
        raise NotImplementedError("deferred K1 requires gfx950")
    for rows in args.rows:
        inp = seam_make_inputs(
            rows,
            combine=args.combine,
            next_gate=args.next_gate,
            seed=args.seed,
            w_scale=args.weight_scale,
        )
        inp["_prepared"] = hc_seam_prepare_weights(
            inp["norm_weight"], inp["w_down"], inp["w_up"], inp["w_inject"]
        )
        chunk = args.chunk_width
        padded = ((rows + 63) // 64) * 64
        splits = HIDDEN // chunk
        residual_out = torch.empty_like(inp["residual"])
        partial = torch.empty(
            (splits * padded, LOWRANK), device="cuda", dtype=torch.float32
        )
        stats = torch.empty((splits * rows,), device="cuda", dtype=torch.float32)
        builder = _build_down_deferred_partial_pipe(
            chunk_width=chunk, slp=bool(args.slp), has_combine=args.combine
        )
        launch_args = (
            inp["residual"],
            inp["y"] if args.combine else inp["residual"],
            inp["gate_prev"] if args.combine else inp["residual"],
            inp["norm_weight"],
            inp["w_down"],
            residual_out,
            partial,
            stats,
        )
        compiled = flyc.compile(
            builder, *launch_args, fx.Stream(torch.cuda.current_stream())
        )
        r_ref, d_ref, ss_ref = seam_deferred_partials_ref(
            inp["y"],
            inp["residual"],
            inp["gate_prev"],
            inp["norm_weight"],
            inp["w_down"],
            chunk_width=chunk,
        )
        d = partial.view(HC, HS // chunk, padded, LOWRANK)[:, :, :rows]
        assert (d.double() - d_ref).norm() / d_ref.double().norm() < 1e-4
        assert ((stats.view_as(ss_ref).double() - ss_ref).abs() / ss_ref).max() < 1e-5
        if args.combine:
            assert torch.equal(residual_out, r_ref)

        def deferred(compiled=compiled, launch_args=launch_args):
            compiled(*launch_args, fx.Stream(torch.cuda.current_stream()))

        def baseline(inp=inp):
            seam_arm_prepared(inp)

        for _ in range(10):
            baseline()
            deferred()
        torch.cuda.synchronize()
        graphs = []
        for fn in (baseline, deferred):
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph):
                for _ in range(args.boundaries):
                    fn()
            graphs.append(graph)
        samples = [[], []]
        for trial in range(args.trials + 10):
            for i in (trial % 2, 1 - trial % 2):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.replays):
                    graphs[i].replay()
                end.record()
                end.synchronize()
                if trial >= 10:
                    samples[i].append(
                        start.elapsed_time(end) * 1000 / args.boundaries / args.replays
                    )
        print(
            f"deferred M={rows} chunk={chunk} slp={args.slp} "
            f"combine={int(args.combine)} baseline_geometry="
            f"{seam_config('gfx950', rows, args.combine, args.next_gate, 384 if args.next_gate else 320)}"
        )
        print(
            f"graph-us baseline-full-seam={statistics.median(samples[0]):.6f} "
            f"deferred-partial-only={statistics.median(samples[1]):.6f}; "
            f"nodes={[_graph_node_counts(g) for g in graphs]}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--deferred", action="store_true", help="private deferred-partial A/B trace"
    )
    ap.add_argument(
        "--chunk-width", type=int, choices=(256, 320, 512, 640, 1280, 2560), default=256
    )
    ap.add_argument("--slp", type=int, choices=(0, 1), default=0)
    ap.add_argument(
        "--seam", action="store_true", help="run full-call seam graph timing"
    )
    ap.add_argument("--rows", type=int, nargs="+", default=[16, 64, 128, 256])
    ap.add_argument("--combine", type=int, choices=(0, 1), default=1)
    ap.add_argument(
        "--next-gate", dest="next_gate", type=int, choices=(0, 1), default=1
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weight-scale", type=float, default=1.0)
    ap.add_argument("--bench-mode", choices=BENCH_MODES, default="seam")
    ap.add_argument(
        "--seam-deferred", action="store_true", help="three-launch prepared seam"
    )
    ap.add_argument(
        "--k2-branch-mode",
        choices=("branchwise", "combined", "separate"),
        default=None,
        help="force a K2 schedule; default lets the library pick per M",
    )
    ap.add_argument("--k2-slp", type=int, choices=(0, 1), default=None)
    ap.add_argument("--k2-n-tile", type=int, default=None)
    ap.add_argument("--k2-m-waves", type=int, default=None)
    ap.add_argument("--k2-n-waves", type=int, default=None)
    ap.add_argument("--row-tile", type=int, choices=(16, 32, 64), default=None)
    ap.add_argument("--k1-row-tile", type=int, choices=(16, 32, 64), default=None)
    ap.add_argument("--k2-row-tile", type=int, choices=(16, 32, 64), default=None)
    ap.add_argument(
        "--fallback", choices=("splitk", "decouple", "skinny"), default=None
    )
    ap.add_argument("--boundaries", type=int, default=24)
    ap.add_argument(
        "--replays", type=int, default=20, help="graph replays per timed batch"
    )
    ap.add_argument("--trials", type=int, default=15, help="timed batches")
    ap.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[1, 8, 32, 128, 512, 2048, 4096, 8192, 16384, 32768],
    )
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--warmup-s", type=float, default=1.0)
    args = ap.parse_args()
    args.combine, args.next_gate = bool(args.combine), bool(args.next_gate)

    arch = get_gfx()
    if arch not in ("gfx950", "gfx942"):
        print(f"[skip] requires gfx950/gfx942 FlyDSL, got {arch}")
        sys.exit(0)

    if args.deferred:
        deferred_main(args)
        return

    if args.seam:
        seam_main(args)
        return

    print(
        f"device={torch.cuda.get_device_name(0)} ({arch})  "
        f"HIDDEN={HIDDEN} LOWRANK={LOWRANK} HC={HC}  "
        f"iters={args.iters} warmup={args.warmup_s}s  fused two-stage (fold_w=True)"
    )
    print(f"  {'tokens':>7} {'fk2+fold':>10}")
    print("  " + "-" * 20)
    for tokens in args.tokens:
        inp = make_inputs(tokens)
        _validate(inp, tokens)
        us = _time_us(inp, args.iters, args.warmup_s)
        print(f"  {tokens:7d} {us:9.1f}u")


if __name__ == "__main__":
    main()
