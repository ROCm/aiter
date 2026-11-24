# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Time the RDNA attention kernels on the shapes of shapes.csv against their
roofline, as markdown.

Decode specs run under CUDA graphs, prefill specs without, as a server runs
them (harness.py). `path` is what ran: `kernel` (decode), `prefill` or
`unserved`; %roof is against utils.roofline. --batch takes batch specs or
SPECS names. Forcing a knob, --ablate or --jit builds the variants with
jit.py. --non-causal times encoder attention on segments of L = S tokens.

    python op_tests/op_benchmarks/rdna_attn/matrix.py
"""

import argparse
import csv
import functools
import statistics
import sys
import time
from pathlib import Path
from typing import NamedTuple

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

import utils

_SHAPES = _HERE.parent / "shapes.csv"
# Named batches for --batch.
SPECS = {
    "decode": ["2q1s4k", "8q1s4k", "8q4s4k", "32q1s1k", "64q1s1k", "32q4s1k"],
    "prefill-batch": ["8q256", "16q64s1k", "4q1k", "2q2k_2q512s4k", "4q128s16k"],
    "mixed": [
        "4q1s8k_q512",
        "16q1s4k_2q1k",
        "16q1s2k_q64s2k",
        "32q1s1k_q2k",
        "q1s32k_q512s4k_q100",
        "64q1s1k_q2k",
    ],
}


class Shape(NamedTuple):
    hq: int
    hkv: int
    d: int
    window: int


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--shapes", default=str(_SHAPES))
    p.add_argument("--hq", type=int, nargs="+", help="keep only these Hq")
    p.add_argument("--hkv", type=int, nargs="+", help="keep only these Hkv")
    p.add_argument("--head-dim", type=int, nargs="+", help="keep only these D")
    p.add_argument("--gqa", type=int, nargs="+", help="keep only these Hq/Hkv")
    p.add_argument(
        "--windowed",
        action="store_true",
        help="the sliding-window rows instead of the full-attention ones",
    )


def load(args: argparse.Namespace) -> tuple[list[Shape], int]:
    """The shapes left after the filters in `args`.

    Returns:
        (shapes, number of sliding-window rows skipped).

    """
    shapes: list[Shape] = []
    windowed = 0
    # Decoder shapes run causal, encoder shapes non-causal (--non-causal).
    mode = "encoder" if getattr(args, "non_causal", False) else "decoder"
    with open(args.shapes) as fh:
        for r in csv.DictReader(fh):
            if not int(r.get(mode, mode == "decoder")):
                continue
            s = Shape(int(r["Hq"]), int(r["Hkv"]), int(r["D"]), int(r["window"]))
            if args.hq and s.hq not in args.hq:
                continue
            if args.hkv and s.hkv not in args.hkv:
                continue
            if args.head_dim and s.d not in args.head_dim:
                continue
            if args.gqa and (s.hq % s.hkv or s.hq // s.hkv not in args.gqa):
                continue
            if bool(s.window) != getattr(args, "windowed", False):
                windowed += bool(s.window)
                continue
            shapes.append(s)
    return shapes, windowed


def describe(args: argparse.Namespace) -> str:
    """The active filters, as text."""
    bits = []
    for name, val in (
        ("Hq", args.hq),
        ("Hkv", args.hkv),
        ("D", args.head_dim),
        ("Hq/Hkv", args.gqa),
    ):
        if val:
            bits.append(f"{name}={','.join(str(v) for v in val)}")
    return " ".join(bits)


def our_variants(reqs, hq, hkv, d, window, block_size, dtype, forced, spec=0):
    """The builds harness.Attention picks for a batch."""
    import jit
    import numpy as np

    from aiter.ops import rdna_attn as R

    m = reqs[0].q_len
    out = []
    with jit.override(forced):
        if all(r.q_len == m for r in reqs) and m <= R.MAX_M:
            return [
                jit.variant_for(
                    hq, hkv, d, m, block_size, 1, window, dtype, len(reqs) > 1
                )
            ]
        # Leading decodes of one length, up to 1 + spec tokens.
        lead = reqs[0].q_len if reqs[0].q_len <= 1 + spec else 0
        decodes = next((i for i, r in enumerate(reqs) if r.q_len != lead), len(reqs))
        if decodes:
            out.append(
                jit.variant_for(
                    hq, hkv, d, lead, block_size, 1, window, dtype, decodes > 1
                )
            )
        rest = reqs[decodes:]
        if len(rest) == 1:
            v = jit.prefill_variant_for(
                hq, hkv, d, block_size, 1, window, dtype, rest[0].q_len
            )
        else:
            v, _ = jit.prefill_batch_variant_for(
                hq,
                hkv,
                d,
                block_size,
                1,
                window,
                dtype,
                np.cumsum([0] + [r.q_len for r in rest]),
                np.array([r.kv_len for r in rest]),
            )
    return out if v is None else [*out, v]


def _segments(spec: str) -> list[int]:
    """Segments of a spec: 577x5 is five of 577, 1024,257 those two."""
    if "x" in spec:
        n, count = spec.split("x")
        return [int(n)] * int(count)
    return [int(x) for x in spec.split(",")]


def encoder_layers(inputs, segs, scale) -> None:
    """Encoder attention of each layer's (Q, K, V) over the segments."""
    from aiter.ops import rdna_attn as R

    for q, k, v in inputs:
        R.encoder_forward(q, k, v, segs, scale)


def non_causal(args: argparse.Namespace, shapes: list[Shape]) -> None:
    """--non-causal: per shape and segment list, encoder attention."""
    import torch

    from aiter.ops import rdna_attn as R

    dtype = utils.torch_dtype(args.dtype)
    specs = args.segments or [str(m) for m in args.m]
    layers = 10

    def timed(run):
        """Median us per layer."""
        for _ in range(3):
            run()
        ts = []
        for _ in range(20 * args.reps):
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            run()
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b) * 1e3 / layers)
        return statistics.median(ts)

    print(f"<!-- non-causal, {args.dtype}; {layers} layers -->")
    columns = ["segments", "Hq", "Hkv", "D", "bound", "roofline", "ours", "%roof"]
    print("| " + " | ".join(columns) + " |")
    print("| --- " * len(columns) + "|")
    for spec in specs:
        segs = _segments(spec)
        n = sum(segs)
        for hq, hkv, d, window in shapes:
            if window:
                continue
            roof, bound = utils.roofline_noncausal(
                hq, hkv, d, segs, torch.finfo(dtype).bits // 8
            )
            # Each layer its own Q, K and V.
            inputs = [
                [
                    (torch.randn(n, h, d, device="cuda") * 0.5).to(dtype)
                    for h in (hq, hkv, hkv)
                ]
                for _ in range(layers)
            ]
            head = [spec, hq, hkv, d, bound, f"{roof:.2f}"]
            why = R.encoder_unsupported(inputs[0][0], hkv)
            if why:
                print(f"| {' | '.join(map(str, head))} | {why} |", flush=True)
                continue

            us = timed(functools.partial(encoder_layers, inputs, segs, d**-0.5))
            cells = [*head, f"{us:.2f}", f"{roof / us * 100:.1f} %"]
            print("| " + " | ".join(map(str, cells)) + " |", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    add_arguments(p)
    utils.add_dtype_argument(p)
    p.add_argument(
        "--m",
        type=int,
        nargs="+",
        help="query tokens per sequence (default 1 4; with --prefill "
        "128 256 512 1024 2048 4096 8192)",
    )
    p.add_argument(
        "--prefill",
        action="store_true",
        help="one sequence of --m new tokens after each of --prefixes cached ones",
    )
    p.add_argument("--prefixes", type=int, nargs="+", default=[0, 4096, 16384])
    p.add_argument(
        "--non-causal",
        action="store_true",
        help="encoder attention: segments of --m tokens, or --segments",
    )
    p.add_argument(
        "--segments",
        nargs="+",
        help="with --non-causal, one call's segments: 577x5 or 1024,257,9",
    )
    p.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        default=[128, 512, 1024, 4096, 8192, 16384, 32768],
    )
    p.add_argument(
        "--batch",
        nargs="+",
        help="batch specs instead of --m x --contexts; "
        f"{', '.join(SPECS)} stand for the lists in SPECS",
    )
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument(
        "--spec-tokens",
        type=int,
        default=0,
        help="speculative tokens per decode: requests of up to 1 + this many "
        "tokens are decodes, as when serving with speculative decoding",
    )
    p.add_argument(
        "--reps",
        type=int,
        default=1,
        help="repetitions per cell",
    )
    for knob in dict.fromkeys(utils.KNOBS + utils.PREFILL_KNOBS):
        p.add_argument(
            f"--{knob.replace('_', '-')}",
            type=int,
            help=f"force {knob.upper()} on every configuration (built by jit.py)",
        )
    p.add_argument(
        "--ablate",
        type=int,
        default=0,
        help="profiling only, wrong results; see ABLATE in the kernel",
    )
    p.add_argument(
        "--jit",
        action="store_true",
        help="run jit.py builds of the tables' entries rather than the modules'",
    )
    args = p.parse_args()
    dtype = utils.torch_dtype(args.dtype)

    import harness
    import jit
    from batch_spec import parse_batch_spec

    from aiter.ops import rdna_attn as R

    if args.m is None:
        args.m = (
            [128, 256, 512, 1024, 2048, 4096, 8192]
            if args.prefill or args.non_causal
            else [1, 4]
        )
    if args.non_causal:
        # Encoder attention runs the tables' builds through encoder_forward.
        knobs = dict.fromkeys(utils.KNOBS + utils.PREFILL_KNOBS)
        forced = [k for k in knobs if getattr(args, k) is not None]
        if forced or args.ablate or args.jit:
            p.error(
                "--non-causal times the tables' builds: no forced knobs, "
                "--ablate or --jit"
            )
        shapes, _ = load(args)
        non_causal(args, shapes)
        return
    if args.batch:
        specs = [s for b in args.batch for s in SPECS.get(b, [b])]
    elif args.prefill:
        specs = [f"q{m}s{m + p}" for p in args.prefixes for m in args.m]
    else:
        specs = [f"q{m}s{s}" for s in args.contexts for m in args.m]
    parsed = {spec: parse_batch_spec(spec) for spec in specs}
    # One max_model_len for the run.
    max_len = max(r.kv_len for reqs in parsed.values() for r in reqs)

    shapes, windowed = load(args)
    if not shapes:
        print(f"<!-- no shapes match {describe(args) or 'the filters'} -->")
        return

    names = utils.PREFILL_KNOBS if args.prefill else utils.KNOBS
    forced = {k: getattr(args, k) for k in names if getattr(args, k) is not None}
    kernels = R
    if forced or args.ablate or args.jit:
        kernels = jit
        jit.install(ablate=args.ablate)
        wanted = [
            v
            for hq, hkv, d, window in shapes
            for reqs in parsed.values()
            for v in our_variants(
                reqs,
                hq,
                hkv,
                d,
                window,
                args.block_size,
                dtype,
                forced,
                args.spec_tokens,
            )
        ]
        jit.precompile(wanted)
        jit.seal()
        time.sleep(5)  # let clocks settle after the build

    def timeit(spec, hq, hkv, d, window):
        """(us per layer, spread %, path) of a spec."""
        rs = [
            harness.run(
                spec,
                hq,
                hkv,
                d,
                dtype=dtype,
                block_size=args.block_size,
                window=window,
                use_cuda_graphs=all(r.q_len <= 8 for r in parsed[spec]),
                max_model_len=max_len,
                spec_tokens=args.spec_tokens,
                kernels=kernels,
            )
            for _ in range(args.reps)
        ]
        unserved = next((r for r in rs if r.path == "unserved"), None)
        if unserved is not None:
            raise RuntimeError(f"unserved: {unserved.rejected}")
        return (
            statistics.median(r.median_us for r in rs),
            max(r.spread_pct for r in rs),
            rs[0].path,
        )

    active = describe(args)
    knobs = " ".join(f"{k}={v}" for k, v in forced.items())
    print(
        f"<!-- {len(shapes)} configurations; "
        f"{windowed} sliding-window rows skipped; {args.dtype}, block size "
        f"{args.block_size}"
        f"{'; filters: ' + active if active else ''}"
        f"{'; forced: ' + knobs if knobs else ''}"
        f"{f'; ABLATE={args.ablate}, wrong numbers' if args.ablate else ''} -->"
    )
    columns = [
        "spec",
        "Hq",
        "Hkv",
        "D",
        "window",
        "path",
        "AI",
        "bound",
        "roofline",
        "ours",
        "%roof",
        "spread",
    ]
    print("| " + " | ".join(columns) + " |")
    print("| --- " * len(columns) + "|")
    # Spec-major, contexts upward.
    for spec in specs:
        for hq, hkv, d, window in shapes:
            roof, intensity, bound = utils.spec_roofline(
                spec,
                hq,
                hkv,
                d,
                block_size=args.block_size,
                window=window,
                realistic=args.prefill,
            )
            head = [spec, hq, hkv, d, window or "-"]
            try:
                with jit.override(forced):
                    ours, spread, path = timeit(spec, hq, hkv, d, window)
            except Exception as exc:  # noqa: BLE001 - a forced knob may not build
                print(
                    f"| {' | '.join(map(str, head))} | {type(exc).__name__}: "
                    f"{str(exc).splitlines()[0]} |",
                    flush=True,
                )
                continue
            cells = [
                *head,
                path,
                f"{intensity:.0f}",
                bound,
                f"{roof:.2f}",
                f"{ours:.2f}",
                f"{roof / ours * 100:.1f} %",
                f"{spread:.1f} %",
            ]
            print("| " + " | ".join(map(str, cells)) + " |", flush=True)


if __name__ == "__main__":
    main()
