# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MHA harness for the interleaved randomized-block measurement in
``aiter.utility.block_race``.

The measurement and the statistics live in the shared core, so other families
can race their own candidates. What remains here is the part that knows about
MHA: the problem shape, how to build its inputs, how to launch one candidate,
and which candidates exist.

Three modes. ``race`` runs the delta-based elimination the tuner uses.
``sweep`` is the block-size calibration that chose the default block, kept
because the answer depends on the machine. ``null`` races one configuration
against copies of itself, which is the experiment that turns delta from a
judgement call into a measurement: the truth is known to be a tie, so any
separation reported is measurement error.

Run with PYTHONPATH set to the repository root and HIP_VISIBLE_DEVICES
pinned, e.g.
    HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD python -m \
        op_tests.tuners.prototype_interleaved_bench
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time

import torch

from aiter.ops.mha_fwd_policy import enumerate_mha_fwd_candidates
from aiter.ops.triton.attention.mha import flash_attn_varlen_func
from aiter.utility.block_race import (
    JsonlBlockJournal,
    RaceEntrant,
    check_t_implementation,
    cuda_event_timer,
    indistinguishable_set,
    measure_blocks,
    position_effect,
    race,
    rank,
    wilcoxon_floor,
)

# The Kimi hd192/hdv128 varlen shape the storage comparison has been using.
SHAPE = {
    "total_q": 4096,
    "total_k": 42700,
    "max_seqlen_q": 4096,
    "max_seqlen_k": 42700,
    "nhead": 12,
    "hdim_q": 192,
    "hdim_v": 128,
}


def make_entrant(backend: str, config: dict, origin: str = "catalogue", tag: str = ""):
    """Wrap one MHA configuration as something the shared race can measure.

    The incumbent is marked protected: it is measured in every block like
    everything else, but it is never eliminated, because a run that stops
    measuring the configuration already in use cannot tell an improvement from
    a regression.
    """
    base = f"{backend}:{json.dumps(config, sort_keys=True)}"
    return RaceEntrant(
        label=f"{base}#{tag}" if tag else base,
        payload={"backend": backend, "config": dict(config), "origin": origin},
        protected="incumbent" in origin,
    )


def build_inputs(device: str = "cuda"):
    dtype = torch.bfloat16
    q = torch.randn(
        SHAPE["total_q"], SHAPE["nhead"], SHAPE["hdim_q"], device=device, dtype=dtype
    )
    k = torch.randn(
        SHAPE["total_k"], SHAPE["nhead"], SHAPE["hdim_q"], device=device, dtype=dtype
    )
    v = torch.randn(
        SHAPE["total_k"], SHAPE["nhead"], SHAPE["hdim_v"], device=device, dtype=dtype
    )
    cu_q = torch.tensor([0, SHAPE["total_q"]], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, SHAPE["total_k"]], device=device, dtype=torch.int32)
    return q, k, v, cu_q, cu_k


def make_invoker(inputs):
    q, k, v, cu_q, cu_k = inputs
    scale = SHAPE["hdim_q"] ** -0.5

    def invoke(entrant: RaceEntrant):
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            SHAPE["max_seqlen_q"],
            SHAPE["max_seqlen_k"],
            dropout_p=0.0,
            softmax_scale=scale,
            causal=False,
            window_size=(-1, -1),
            return_lse=False,
            config=dict(entrant.payload["config"]),
            backend=entrant.payload["backend"],
        )

    return invoke


def collect_candidates(
    gfx: str,
    strategy: str = "smoke",
    limit: int | None = None,
    seed: int = 0,
) -> list[RaceEntrant]:
    """The restricted catalogue plus whatever each kernel resolves today.

    The incumbents are part of the field by construction rather than a second
    phase, and adding them here costs one more visit per block.

    ``limit`` takes a seeded sample of the catalogue, which is how the cost of
    a race is measured as the field grows without running the full catalogue
    first. The sample is drawn before the incumbents are added, so they are
    present at every size and the sizes stay comparable.
    """
    catalogue = list(enumerate_mha_fwd_candidates(gfx, strategy, ["triton", "gluon"]))
    if limit is not None and limit < len(catalogue):
        catalogue = random.Random(seed).sample(catalogue, limit)
    entrants = [make_entrant(c.backend, dict(c.backend_config)) for c in catalogue]
    seen = {entrant.label for entrant in entrants}

    for backend in ("gluon", "triton"):
        try:
            if backend == "gluon":
                from aiter.ops.triton._gluon_kernels.gfx950.attention.mha import (
                    _get_config as resolve,
                )

                config = resolve(is_fp8=False, has_pe=False)
            else:
                from aiter.ops.triton._triton_kernels.attention.mha import (
                    _get_config as resolve,
                )

                config = resolve(
                    False,
                    torch.bfloat16,
                    has_pe=False,
                    head_dim_v=SHAPE["hdim_v"],
                )
        except Exception as error:  # noqa: BLE001 - a missing default is not fatal
            print(f"  no incumbent for {backend}: {error}")
            continue
        incumbent = make_entrant(backend, dict(config), "incumbent")
        if incumbent.label in seen:
            # Already in the catalogue; relabel so the report can point at it.
            entrants = [
                make_entrant(
                    e.payload["backend"], e.payload["config"], "catalogue+incumbent"
                )
                if e.label == incumbent.label
                else e
                for e in entrants
            ]
        else:
            entrants.append(incumbent)
    return entrants


def screen(entrants: list[RaceEntrant], invoke, warmup: int) -> list[RaceEntrant]:
    """Drop candidates that cannot run this problem, and warm the rest.

    Done once, outside the blocks, so a compile or load cost is never charged
    to a timed call.
    """
    survivors = []
    for entrant in entrants:
        try:
            for _ in range(warmup):
                invoke(entrant)
            torch.cuda.synchronize()
        except Exception as error:  # noqa: BLE001 - unsupported is an outcome
            print(f"  dropped {entrant.label}: {type(error).__name__}")
            continue
        survivors.append(entrant)
    return survivors


def measure_contiguous(entrants, time_calls, calls):
    """The current harness's shape: every call for a candidate back to back.

    The reference the interleaved estimates are checked against. If they
    disagree, interleaving is biased and block size is not merely a power
    versus cost trade.
    """
    return measure_blocks(entrants, time_calls, calls, 1, seed=0)


def origin_of(entrants, label: str) -> str:
    return next(
        (e.payload["origin"] for e in entrants if e.label == label), "catalogue"
    )


def run_race(args, entrants, time_calls) -> None:
    print(
        f"\nracing {len(entrants)} candidates, delta={args.delta:.1%}, "
        f"block={args.block_calls} calls, at most {args.max_blocks} blocks"
    )
    journal = (
        JsonlBlockJournal(args.journal, resume=args.resume) if args.journal else None
    )
    started = time.perf_counter()
    result = race(
        entrants,
        time_calls,
        delta=args.delta,
        alpha=args.alpha,
        block_calls=args.block_calls,
        min_blocks=args.min_blocks,
        max_blocks=args.max_blocks,
        seed=args.seed,
        journal=journal,
        resume=args.resume,
    )
    wall = time.perf_counter() - started

    samples = result.samples
    kernel_seconds = sum(sum(sum(b) for b in s.blocks) for s in samples.values()) / 1e6
    exhaustive = (
        args.max_blocks
        * args.block_calls
        * sum(s.estimate for s in samples.values() if math.isfinite(s.estimate))
        / 1e6
    )
    print(
        f"\nspent {result.calls_spent} calls / {kernel_seconds:.1f} s of kernel "
        f"time in {wall:.1f} s wall; measuring every candidate for all "
        f"{args.max_blocks} blocks would have cost {exhaustive:.1f} s "
        f"({exhaustive / max(kernel_seconds, 1e-9):.1f}x)"
    )
    print(
        f"converged after {result.blocks_run} blocks: the winner is within "
        f"{args.delta:.0%} of the best candidate in the catalogue"
        if result.certified
        else f"budget exhausted at {result.blocks_run} blocks without certifying "
        f"the winner; treat the result as a ranking, not a guarantee"
    )

    survivors = [v for v in result.verdicts if v.state in ("leader", "within_delta")]
    print(f"\nwithin delta of the best: {len(survivors)} of {len(entrants)}")
    for verdict in sorted(survivors, key=lambda v: v.estimate):
        mark = (
            "  <- incumbent" if "incumbent" in origin_of(entrants, verdict.label) else ""
        )
        won = "  *winner*" if verdict.label == result.winner else ""
        print(
            f"  {verdict.estimate:9.1f} us  {verdict.relative_gap:+6.2%}  "
            f"spread {verdict.relative_spread:.2%}  {verdict.label}{mark}{won}"
        )
    print(f"\nselected by {result.tie_break}: {result.winner}")

    # Each backend contributes its own incumbent, so report them one at a time
    # rather than collapsing them into a single verdict about "the" incumbent.
    for verdict in result.verdicts:
        if "incumbent" not in origin_of(entrants, verdict.label):
            continue
        backend = verdict.label.split(":")[0]
        if verdict.state == "protected_behind":
            print(
                f"  the {backend} incumbent was kept in the field but measured "
                f"{verdict.relative_gap:+.2%} behind the leader"
            )
        elif verdict.label == result.winner:
            print(
                f"  the {backend} incumbent is inside the indifference zone, so "
                f"keeping it is the outcome that changes nothing"
            )
        else:
            print(
                f"  the {backend} incumbent is within delta at "
                f"{verdict.relative_gap:+.2%} but did not win the tie-break"
            )

    dropped = sorted(
        (v for v in result.verdicts if v.state == "eliminated"),
        key=lambda v: v.blocks_used,
    )
    print(f"\neliminated {len(dropped)}, and how early:")
    for verdict in dropped[:20]:
        print(
            f"  after {verdict.blocks_used:>3} blocks  {verdict.relative_gap:+8.1%}  "
            f"{verdict.label.split(':')[0]:>7}  {verdict.estimate:9.1f} us"
        )
    if len(dropped) > 20:
        print(f"  ... and {len(dropped) - 20} more")

    if args.json_out:
        emit_json(args, entrants, result, wall, kernel_seconds)


def emit_json(args, entrants, result, wall, kernel_seconds) -> None:
    """Machine-readable record, so a number quoted in a document has a source."""
    payload = {
        "shape": SHAPE,
        "gfx": args.gfx,
        "strategy": args.strategy,
        "candidates_requested": args.candidates,
        "candidates_raced": len(entrants),
        "delta": args.delta,
        "alpha": args.alpha,
        "block_calls": args.block_calls,
        "min_blocks": args.min_blocks,
        "max_blocks": args.max_blocks,
        "seed": args.seed,
        "screen_seconds": getattr(args, "_screen_seconds", None),
        "screen_peak_bytes": getattr(args, "_screen_peak_bytes", None),
        "blocks_run": result.blocks_run,
        "blocks_replayed": result.blocks_replayed,
        "calls_spent": result.calls_spent,
        "kernel_seconds": kernel_seconds,
        "wall_seconds": wall,
        "certified": result.certified,
        "winner": result.winner,
        "tie_break": result.tie_break,
        "survivors": result.survivors,
        "history": [
            {
                "block": record.block,
                "active": record.active,
                "calls_spent": record.calls_spent,
                "wall_seconds": record.wall_seconds,
                "eliminated": record.eliminated,
            }
            for record in result.history
        ],
        "verdicts": [
            {
                "label": v.label,
                "state": v.state,
                "estimate_us": v.estimate,
                "relative_gap": v.relative_gap,
                "relative_spread": v.relative_spread,
                "blocks_used": v.blocks_used,
            }
            for v in result.verdicts
        ],
    }
    directory = os.path.dirname(args.json_out)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(args.json_out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {args.json_out}")


def run_null(args, entrants, time_calls) -> None:
    """Race one configuration against copies of itself.

    Every other experiment here can only compare procedures against each
    other, because the true ranking of two different configurations is never
    known. Replicas make it known: the truth is that all of them are equally
    fast, so any separation the race reports is measurement error and any
    elimination is a false positive with a known correct answer.

    That turns delta from a judgement call into a measurement. A delta below
    the spread seen here is asking the machine for a distinction it cannot
    make twice, and no amount of extra blocks will fix it, because the
    variation is between sessions and not within them.
    """
    seed = next(
        (e for e in entrants if "incumbent" in e.payload["origin"]), entrants[0]
    )
    replicas = [
        make_entrant(seed.payload["backend"], seed.payload["config"], "replica", f"r{i}")
        for i in range(args.replicas)
    ]
    print(
        f"\nnull experiment: {args.replicas} copies of one configuration\n"
        f"  {seed.label}\n"
        f"  ground truth: every gap is zero, so every elimination is an error"
    )

    result = race(
        replicas,
        time_calls,
        delta=args.delta,
        alpha=args.alpha,
        block_calls=args.block_calls,
        min_blocks=args.min_blocks,
        max_blocks=args.max_blocks,
        seed=args.seed,
    )

    estimates = sorted(s.estimate for s in result.samples.values())
    spread = estimates[-1] / estimates[0] - 1.0
    false_positives = [v for v in result.verdicts if v.state == "eliminated"]
    print(
        f"\nspent {result.calls_spent} calls over {result.blocks_run} blocks\n"
        f"  fastest replica  {estimates[0]:9.1f} us\n"
        f"  slowest replica  {estimates[-1]:9.1f} us\n"
        f"  spread across identical configs: {spread:.2%}\n"
        f"  false eliminations: {len(false_positives)} of {args.replicas - 1}"
    )
    for verdict in false_positives:
        print(f"    wrongly dropped {verdict.label} at {verdict.relative_gap:+.2%}")
    print(
        f"\ndelta={args.delta:.1%} is "
        + (
            "above the observed spread, so the race is not being asked to "
            "resolve differences this machine cannot reproduce"
            if args.delta > spread
            else "BELOW the observed spread: the race is resolving differences "
            "smaller than the noise between identical configurations"
        )
    )
    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(
                {
                    "mode": "null",
                    "replicas": args.replicas,
                    "delta": args.delta,
                    "estimates_us": estimates,
                    "spread": spread,
                    "false_eliminations": len(false_positives),
                    "blocks_run": result.blocks_run,
                },
                handle,
                indent=2,
            )
        print(f"\nwrote {args.json_out}")


def run_sweep(args, entrants, time_calls) -> None:
    print(f"\nreference pass: {args.calls_per_candidate} contiguous calls each")
    reference = measure_contiguous(entrants, time_calls, args.calls_per_candidate)
    reference_estimate = {label: s.estimate for label, s in reference.items()}

    block_sizes = [int(b) for b in args.block_sizes.split(",") if b.strip()]
    print(
        f"\nsweeping block size at a fixed budget of "
        f"{args.calls_per_candidate} calls per candidate"
    )
    print(
        f"{'block':>6} {'blocks':>7} {'wall s':>8} {'kernel s':>9} "
        f"{'median bias':>12} {'max |bias|':>11} {'test floor':>11} {'winner':>8}"
    )

    results = {}
    for block_calls in block_sizes:
        blocks = max(2, args.calls_per_candidate // block_calls)
        started = time.perf_counter()
        samples = measure_blocks(
            entrants, time_calls, block_calls, blocks, seed=args.seed
        )
        wall = time.perf_counter() - started

        # Signed and unsigned are different questions. A systematic switching
        # cost shifts every candidate the same way and shows in the median;
        # plain noise from short blocks shows only in the maximum.
        signed = [
            (sample.estimate - reference_estimate[label]) / reference_estimate[label]
            for label, sample in samples.items()
            if math.isfinite(reference_estimate.get(label, float("inf")))
        ]
        kernel_seconds = (
            sum(
                sum(sum(block) for block in sample.blocks)
                for sample in samples.values()
            )
            / 1e6
        )
        winner = rank(samples)[0][0].split(":")[0]
        print(
            f"{block_calls:>6} {blocks:>7} {wall:>8.1f} {kernel_seconds:>9.1f} "
            f"{statistics.median(signed) if signed else float('nan'):>11.2%} "
            f"{max(abs(b) for b in signed) if signed else float('nan'):>10.2%} "
            f"{wilcoxon_floor(blocks):>11.4f} {winner:>8}"
        )
        results[block_calls] = samples

    print("\nlatency by position within a block (median, normalized per candidate)")
    print("a switching cost inside the timed calls shows up as slow early positions")
    for block_calls in block_sizes:
        if block_calls < 5:
            continue
        effect = position_effect(results[block_calls])
        head = "  ".join(f"p{p}={v:.3f}" for p, v, _ in effect[:6])
        print(f"  block={block_calls:>3}: {head}")

    # Analyse at the largest block whose block count still leaves the test able
    # to reject anything. Reporting a survivor set from a powerless test would
    # read as "everything ties" when it means "we learned nothing".
    # The bar a comparison has to clear is alpha divided by the number of
    # comparisons, not alpha, so multiplicity belongs in the capability check.
    comparisons = max(1, len(entrants) - 1)
    strictest = 0.05 / comparisons
    capable = [
        b
        for b in block_sizes
        if wilcoxon_floor(max(2, args.calls_per_candidate // b)) <= strictest
    ]
    analysis_block = max(capable) if capable else min(block_sizes)
    analysis_blocks = max(2, args.calls_per_candidate // analysis_block)
    print(f"\nranking at block={analysis_block} ({analysis_blocks} blocks)")
    print(
        f"  {comparisons} comparisons -> Holm's strictest threshold is "
        f"{strictest:.5f}; a rank test needs "
        f"{math.ceil(math.log2(1 / strictest))} blocks to reach it"
    )
    for label, estimate in rank(results[analysis_block])[:6]:
        mark = "  <- incumbent" if "incumbent" in origin_of(entrants, label) else ""
        print(f"  {estimate:9.1f} us  {label}{mark}")

    survivors = indistinguishable_set(results[analysis_block])
    print(
        f"\ncandidates indistinguishable from the fastest: "
        f"{len(survivors)} of {len(entrants)}"
    )
    for label in survivors:
        mark = "  <- incumbent" if "incumbent" in origin_of(entrants, label) else ""
        print(f"  {results[analysis_block][label].estimate:9.1f} us  {label}{mark}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gfx", default="gfx950")
    parser.add_argument(
        "--mode",
        choices=("race", "sweep", "null"),
        default="race",
        help="race runs delta-based elimination; sweep is the block-size "
        "calibration that chose the default block; null races a configuration "
        "against copies of itself to measure the noise floor that bounds delta "
        "from below",
    )
    parser.add_argument(
        "--replicas",
        type=int,
        default=8,
        help="null mode only: how many copies of the one configuration to race",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.02,
        help="indifference zone. Differences smaller than this are treated as "
        "settled rather than as a harder question. Default 2%% is just above "
        "the measured session-to-session drift of an unchanged configuration",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--block-calls",
        type=int,
        default=10,
        help="calls per visit. 10 was chosen by --mode sweep: the switching "
        "cost sits in the first call of a block, so a block of 10 carries "
        "0.07%% bias against a contiguous measurement while a block of 1 "
        "carries 0.82%%",
    )
    parser.add_argument("--min-blocks", type=int, default=3)
    parser.add_argument("--max-blocks", type=int, default=30)
    parser.add_argument(
        "--calls-per-candidate",
        type=int,
        default=200,
        help="sweep mode only: held constant across block sizes so the sweep "
        "isolates block size from measurement budget",
    )
    parser.add_argument(
        "--block-sizes",
        default="1,2,5,10,25,50",
        help="sweep mode only: comma-separated calls per visit",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20240917)
    parser.add_argument(
        "--strategy",
        choices=("smoke", "exhaustive"),
        default="smoke",
        help="which catalogue to draw from before --candidates samples it",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=None,
        help="race a seeded sample of this many catalogue entries, for "
        "measuring how the cost of a race grows with the size of the field",
    )
    parser.add_argument(
        "--journal",
        default=None,
        help="append one record per completed block here, so an interrupted "
        "race can be resumed without re-measuring what it already did",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="replay the journal before measuring anything new",
    )
    parser.add_argument("--json-out", default=None, help="machine-readable summary")
    args = parser.parse_args()

    check_t_implementation()
    torch.cuda.init()
    inputs = build_inputs()
    invoke = make_invoker(inputs)
    time_calls = cuda_event_timer(invoke)

    print(f"building catalogue for {args.gfx}")
    entrants = collect_candidates(args.gfx, args.strategy, args.candidates, args.seed)
    print(f"  {len(entrants)} candidates before screening")

    # Warm-up is where a large field first hurts: every candidate is compiled
    # and resident in one process before any block runs. Timed and reported so
    # the cost of growing the catalogue is visible rather than inferred.
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    entrants = screen(entrants, invoke, args.warmup)
    args._screen_seconds = time.perf_counter() - started
    args._screen_peak_bytes = torch.cuda.max_memory_allocated()
    print(
        f"  {len(entrants)} survive, warmed in {args._screen_seconds:.1f} s, "
        f"peak {args._screen_peak_bytes / 2**30:.2f} GiB"
    )
    for entrant in entrants:
        if entrant.payload["origin"] != "catalogue":
            print(f"  incumbent: {entrant.label}")

    if args.mode == "race":
        run_race(args, entrants, time_calls)
    elif args.mode == "null":
        run_null(args, entrants, time_calls)
    else:
        run_sweep(args, entrants, time_calls)


if __name__ == "__main__":
    main()
