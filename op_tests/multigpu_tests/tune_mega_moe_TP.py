# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tune the pinned single-family TP MoE path, one config per (shape, token).

``mega_moe_tp`` can be told to bypass the stock tuned lookup and serve every
a4w4 shape from exactly two kernel families -- ``flydsl_mxmoe_g1_a4w4_*`` for
GEMM1 and ``flydsl_moe2_layout_afp4_wfp4_*`` for GEMM2 -- because those are the
only two that expose a ``_composition`` hook and can therefore be hosted inside
one megakernel.  Staying on one *family* is the goal; staying on one *tile* is
not.  A fixed block_m ladder picked the best of {32, 64, 128} in only 18 of 28
measured TP8 cells, and the misses cost up to 2.55x.

This script walks the legal tile space per (shape, token) and writes the winner
to ``aiter/configs/flydsl_fuse_kernel_tuned_fmoe.csv``, which
``MegaMoeTPEngine._pinned_row`` reads back.

Search strategy is coordinate descent, not exhaustive: the full product is
96-192 candidates per shape and every distinct tile is a separate JIT build.
Pass 1 sweeps the GEMM1 axes (block_m, BN, BK) against a fixed GEMM2; pass 2
sweeps the GEMM2 axes (tile_n, tile_k, epilogue, nt) against the pass-1 winner.
``--passes`` repeats the pair, which lets a pass-2 result pull pass 1 to a
different block_m.

Usage (8 GPUs, all four models, full token sweep)::

    export AITER_USE_SYSTEM_TRITON=1 AITER_SITUV2_A4W4=1 AITER_FLYDSL_STAGE2_FP8=1
    torchrun --nproc_per_node=8 op_tests/multigpu_tests/tune_mega_moe_TP.py \
        --models kimi3 dsv3 dsv4 glm5 --out /tmp/pin_tuned.csv

Every rank walks the same candidate list in the same order and reduces each
timing with ``max`` across ranks, so all ranks independently agree on the
winner without a broadcast.  Rank 0 alone writes the CSV, rewriting it after
every cell so an interrupted run still leaves a usable file.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time

import torch

# Reuse the benchmark's fixtures rather than reimplementing weights/routing.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_mega_moe_TP import (  # noqa: E402
    MODELS,
    MegaMoeTP,
    SplitTpMoe,
    build_sharded_weights,
    barrier,
    cleanup_dist,
    make_inputs,
    rank_max,
    rel_l2,
    setup_dist,
    time_us,
)

from aiter.fused_moe import get_padded_M  # noqa: E402
from aiter.jit.utils.chip_info import get_cu_num, get_gfx  # noqa: E402
from aiter.ops.flydsl.mega_moe_tp import (  # noqa: E402
    mega_moe_tp_supported,
    set_mega_override,
    pinned_candidates,
    pinned_default_choice,
    pinned_kernel_names,
    set_pin_override,
    set_stage12_override,
)

logger = logging.getLogger("tune_mega_moe_tp")

#: Columns of the emitted CSV.  The first block mirrors ``tuned_fmoe.csv`` so
#: the two files are comparable by eye; ``tp`` is extra because a TP MoE tile
#: depends on the shard, and ``inter_dim`` is the **per-rank** shard for the
#: same reason.
CSV_COLUMNS = [
    "gfx",
    "cu_num",
    "tp",
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
    "gate_mode",
    "block_m",
    "ksplit",
    "kernelName1",
    "kernelName2",
    "us",
    "us_default",
    "speedup_vs_default",
    "candidates_timed",
    "mega",
    "mega_us",
    "nomega_us",
    # Not derivable from either kernel name: a compile hint on the merged
    # kernel. 0 means "let the register allocator decide".
    "waves_per_eu",
    "_tag",
]

#: A candidate whose output drifts further than this from the reference is
#: treated as not existing. Same bound the benchmark's --rtol uses.
_TUNE_RTOL = 0.06

#: Axes swept in each coordinate-descent pass.
_G1_AXES = ("block_m_nt", "g1_bn", "g1_bk")
_G2_AXES = ("g2_tn", "g2_tk", "g2_epilog", "g2_nt")
#: Walked as its own pass: it is a compile hint on the merged kernel rather
#: than a tile, so it interacts with every other axis and none of them in
#: particular, and giving it a pass of its own keeps the descent cheap.
_KERNEL_AXES = ("waves_per_eu",)


def _axis_value(choice: dict, axis: str):
    """Axis value of one candidate.

    ``block_m`` and ``g1_nt`` move together: BM128 has no ``nt`` build, so
    treating them as independent axes would generate candidates that do not
    exist.  They are swept as the single compound axis ``block_m_nt``.
    """
    if axis == "block_m_nt":
        return (choice["block_m"], choice["g1_nt"])
    return choice[axis]


def _neighbours(candidates: list[dict], current: dict, axes: tuple[str, ...]):
    """Candidates differing from ``current`` only along ``axes``."""
    fixed = [a for a in (*_G1_AXES, *_G2_AXES, *_KERNEL_AXES) if a not in axes]
    out = []
    for cand in candidates:
        if all(_axis_value(cand, a) == _axis_value(current, a) for a in fixed):
            out.append(cand)
    return out


def _describe(choice: dict) -> str:
    return (
        f"bm{choice['block_m']}{'nt' if choice['g1_nt'] else ''}"
        f"_g1n{choice['g1_bn']}k{choice['g1_bk']}"
        f"_g2n{choice['g2_tn']}k{choice['g2_tk']}"
        f"_{choice['g2_epilog']}{'_nt' if choice['g2_nt'] else ''}"
        f"{'_wpe%d' % choice['waves_per_eu'] if choice.get('waves_per_eu') else ''}"
    )


def _time_choice(moe, inputs, choice, args, ctx, reference=None) -> float:
    """Microseconds for one candidate, or ``inf`` if it will not run.

    A candidate that fails to build or raises at run time is a hole in the
    enumerated space, not a fatal error -- the point of the sweep is to find
    out which ones work.  The verdict is reduced with ``max`` so one rank's
    failure removes the candidate on every rank and the ranks stay in lockstep.

    A candidate that runs but computes the *wrong answer* is also a hole, and
    timing alone cannot see it. Without this check the sweep will happily crown
    a config that is merely fast: a BM16 GEMM1 paired with a ``reduce`` GEMM2
    was picked that way and then failed the benchmark at rel_l2 0.43.
    """
    set_pin_override(choice)
    try:
        out = moe(inputs)
        torch.cuda.synchronize()
        failed = 0.0
        if reference is not None:
            err = rel_l2(out, reference)
            if not (err == err) or err > _TUNE_RTOL:
                logger.debug(
                    "candidate %s wrong: rel_l2=%s", _describe(choice), err
                )
                failed = 1.0
    except Exception as exc:  # noqa: BLE001 - probing an enumerated space
        logger.debug("candidate %s failed: %s", _describe(choice), exc)
        failed = 1.0
    # Agree across ranks before timing: a collective kernel that died on one
    # rank would otherwise hang the others in the timed region.
    if rank_max(failed, ctx.device) > 0.0:
        set_pin_override(None)
        return float("inf")
    try:
        _, us_max = time_us(
            lambda: moe(inputs),
            iters=args.iters,
            warmup=args.warmup,
            device=ctx.device,
            rounds=args.rounds,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("candidate %s failed while timed: %s", _describe(choice), exc)
        us_max = float("inf")
    finally:
        set_pin_override(None)
    return us_max


def _tune_cell(moe, inputs, candidates, args, ctx, bucket, log_prefix: str,
               reference=None):
    """Coordinate descent over the candidate space for one (shape, token).

    Returns ``(best_choice, best_us, us_default, n_timed)``.  ``us_default`` is
    the heuristic-ladder config's time, i.e. what this cell costs today, so the
    CSV records what the tuning actually bought.
    """
    timed: dict[str, float] = {}

    def timed_us(choice):
        key = _describe(choice)
        if key not in timed:
            timed[key] = _time_choice(moe, inputs, choice, args, ctx, reference)
        return timed[key]

    # Baseline: the heuristic ladder, forced explicitly. Clearing the override
    # instead would read back the CSV this run is in the middle of writing.
    default = pinned_default_choice(moe.config, bucket)
    us_default = timed_us(default)

    # Seed the descent at the baseline so pass 1 starts from a known-good point
    # rather than the first entry of an arbitrarily ordered product.
    current = default
    best_us = us_default
    for pass_no in range(args.passes):
        improved = False
        for axes in (_G1_AXES, _G2_AXES, _KERNEL_AXES):
            for cand in _neighbours(candidates, current, axes):
                us = timed_us(cand)
                if us < best_us:
                    best_us, current, improved = us, cand, True
        logger.info(
            "%s pass %d/%d -> %s %.1fus (%d timed)",
            log_prefix,
            pass_no + 1,
            args.passes,
            _describe(current),
            best_us,
            len(timed),
        )
        if not improved:
            break
    return current, best_us, us_default, len(timed)


def _time_mega_ab(moe, inputs, best, args, ctx):
    """Time the winning tile with the merged kernel on and off.

    Fusing everything into one launch trades GPU time for host time -- the two
    GEMMs alone measured 296 us apart and 391 us merged -- so it wins exactly
    where the host side dominates. That boundary is not derivable from the
    dimensions and it moves with the tile, so it is measured per shape here
    rather than decided at runtime.

    Returns ``(mega_wins, us_mega, us_nomega)``.
    """
    set_pin_override(best)
    out = {}
    for label, flag in (("mega", True), ("nomega", False)):
        set_stage12_override(flag)
        set_mega_override(flag)
        try:
            moe(inputs)
            torch.cuda.synchronize()
            ok = 0.0
        except Exception:  # noqa: BLE001
            ok = 1.0
        if rank_max(ok, ctx.device) > 0.0:
            out[label] = float("inf")
            continue
        try:
            _, out[label] = time_us(
                lambda: moe(inputs),
                iters=args.iters,
                warmup=args.warmup,
                device=ctx.device,
                rounds=args.rounds,
            )
        except Exception:  # noqa: BLE001
            out[label] = float("inf")
    set_stage12_override(None)
    set_mega_override(None)
    set_pin_override(None)
    return out["mega"] <= out["nomega"], out["mega"], out["nomega"]


def _write_csv(path: str, rows: list[dict]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    p.add_argument(
        "--tokens",
        type=int,
        nargs="*",
        default=[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
        help="GLOBAL token counts to tune, each divisible by TP.",
    )
    p.add_argument("--tp", type=int, default=0, help="TP size; 0 = WORLD_SIZE.")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--route", choices=["balanced", "random"], default="balanced")
    p.add_argument(
        "--passes",
        type=int,
        default=2,
        help="Coordinate-descent passes over (GEMM1 axes, GEMM2 axes). A second "
        "pass lets a GEMM2 win pull block_m somewhere the first pass rejected.",
    )
    p.add_argument(
        "--atomic-only",
        action="store_true",
        help="Restrict the search to atomic-epilogue configs but leave the "
        "merged kernel alone. Isolates what the epilogue itself costs from "
        "what hosting it in one kernel costs -- --mega-only conflates the two.",
    )
    p.add_argument(
        "--mega-only",
        action="store_true",
        help="Search only configs the merged kernel can host (atomic epilogue, "
        "tile_m == block_m, no spart) with it forced on. The default search "
        "optimises the layer and often lands on a 'reduce' row, which the "
        "merged kernel cannot use at all -- so the two questions need two "
        "sweeps: what is fastest, and what is fastest *as one kernel*.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="CSV to write. Defaults to the path the engine reads, i.e. "
        "aiter/configs/flydsl_fuse_kernel_tuned_fmoe.csv.",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ctx = setup_dist()
    tp = args.tp or ctx.world
    if args.out is None:
        from aiter.ops.flydsl.mega_moe_tp import pin_tuned_csv_path

        args.out = pin_tuned_csv_path()

    if not mega_moe_tp_supported():
        if ctx.rank == 0:
            logger.error("fused TP MoE needs gfx950, found %s", get_gfx())
        cleanup_dist()
        return 1

    gfx, cu_num = get_gfx(), int(get_cu_num())
    rows: list[dict] = []
    t0 = time.time()
    for model_name in args.models:
        shape = MODELS[model_name]
        tokens = [t for t in args.tokens if t % tp == 0]
        if not tokens:
            continue
        max_local = max(tokens) // tp
        weights = build_sharded_weights(shape, ctx, tp, args.seed)
        moe = MegaMoeTP(weights, ctx, max_local_tokens=max_local, ag_wire_quant="auto")
        split = SplitTpMoe(weights, ctx, max_local)
        candidates = pinned_candidates(moe.config)
        if args.atomic_only:
            candidates = [c for c in candidates if c["g2_epilog"] == "atomic"]
        if args.mega_only:
            from aiter.ops.flydsl.kernels.mega_moe_tp.stage12_rs import (
                stage12_supported,
            )
            from aiter.ops.flydsl.mxfp4_kname import (
                _parse_mxfp4_g1_kname,
                parse_flydsl_v2_gemm2_kernel,
            )

            def _hostable(choice):
                k1, k2 = pinned_kernel_names(moe.config, choice)
                try:
                    return stage12_supported(
                        _parse_mxfp4_g1_kname(k1),
                        parse_flydsl_v2_gemm2_kernel(k2),
                        shape.model_dim,
                    )
                except Exception:  # noqa: BLE001
                    return False

            candidates = [c for c in candidates if _hostable(c)]
            # Both, not just the second: ``_mega_ag`` short-circuits on stage12
            # being off, so forcing only the AllGather tunes the three-kernel
            # path while reporting mega numbers.
            set_stage12_override(True)
            set_mega_override(True)
        if ctx.rank == 0:
            logger.info(
                "%s: %d candidates, %d token counts",
                shape.tag(tp),
                len(candidates),
                len(tokens),
            )
        for global_tokens in tokens:
            inputs = make_inputs(
                shape, ctx, tp, global_tokens, args.seed, args.route
            )
            prefix = f"[{model_name} M={global_tokens}]"
            bucket = int(get_padded_M(global_tokens))
            # The split chain is the reference every candidate is checked
            # against. Timing alone cannot tell a fast config from a wrong one.
            reference = split(inputs).clone()
            torch.cuda.synchronize()
            best, best_us, us_default, n_timed = _tune_cell(
                moe, inputs, candidates, args, ctx, bucket, prefix, reference
            )
            del reference
            kernel1, kernel2 = pinned_kernel_names(moe.config, best)
            if args.mega_only:
                # Already forced on and the search space was restricted to what
                # it can host, so there is nothing to A/B.
                mega_wins, us_mega, us_nomega = True, best_us, float("nan")
            else:
                mega_wins, us_mega, us_nomega = _time_mega_ab(
                    moe, inputs, best, args, ctx
                )
            if ctx.rank == 0:
                rows.append(
                    {
                        "gfx": gfx,
                        "cu_num": cu_num,
                        "tp": tp,
                        "token": global_tokens,
                        "model_dim": shape.model_dim,
                        "inter_dim": weights.local_inter_dim,
                        "expert": shape.experts,
                        "topk": shape.topk,
                        "act_type": str(shape.act_type),
                        "dtype": "torch.bfloat16",
                        "q_dtype_a": "torch.float4_e2m1fn_x2",
                        "q_dtype_w": "torch.float4_e2m1fn_x2",
                        "q_type": "QuantType.per_1x32",
                        "use_g1u1": 1,
                        "doweight_stage1": 0,
                        "gate_mode": "SEPARATED",
                        "block_m": best["block_m"],
                        "waves_per_eu": best.get("waves_per_eu", 0),
                        "ksplit": 0,
                        "kernelName1": kernel1,
                        "kernelName2": kernel2,
                        "us": round(best_us, 4),
                        "us_default": round(us_default, 4),
                        "speedup_vs_default": (
                            round(us_default / best_us, 4)
                            if best_us not in (0.0, float("inf"))
                            and us_default != float("inf")
                            else ""
                        ),
                        "candidates_timed": n_timed,
                        "mega": int(mega_wins),
                        "mega_us": round(us_mega, 4),
                        "nomega_us": round(us_nomega, 4),
                        "_tag": "mega_moe_tp_pinned",
                    }
                )
                _write_csv(args.out, rows)
                logger.info(
                    "%s best %s %.1fus (default %.1fus, %.2fx) [%.0fs elapsed]",
                    prefix,
                    _describe(best),
                    best_us,
                    us_default,
                    (us_default / best_us) if best_us else float("nan"),
                    time.time() - t0,
                )
            del inputs
            torch.cuda.empty_cache()
        del moe, weights
        torch.cuda.empty_cache()
        barrier()

    if ctx.rank == 0:
        logger.info("wrote %d rows to %s", len(rows), args.out)
    cleanup_dist()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
