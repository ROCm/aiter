# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compare the two fused K5+K6 implementations: opus WF vs FlyDSL VK (PR #4884).

Both fuse the same boundary -- the inter-chunk state scan and the output
projection in one dispatch -- so the honest comparison is kernel to kernel:
``gdn_k2_kernel`` for opus, ``chunk_gdn_fwd_h_o_flydsl_vk_*`` for FlyDSL.  Wall
time is recorded too, but it also carries each pipeline's own front end (opus
fuses K1..K4 into one HIP kernel, FlyDSL into ``gdn_prepare_kernel``), so the
two numbers answer different questions.

Two things about this machine are worth stating up front, because they bound
what the numbers mean:

* It is an MI308X with 80 CUs.  PR #4884 gates its VK path on
  ``_device_cu_count() >= 304``, so ``fusion=AUTO`` would never fuse here.  The
  FlyDSL column is forced on with ``fusion=ALWAYS``, and the BV-variant
  heuristic it then runs was tuned against 304 CUs.
* Both columns run ``output_final_state=True`` with a real ``initial_state``.
  The PR's own benchmark was flagged in review for comparing a baseline that
  saved the final state against a fused candidate that did not.

The grid is the one the block bench uses: GQA 4 (Hk=16 / Hv=64), TP 1/2/4/8 so
H = 64/32/16/8, packed varlen, and a token budget T cut into B = T/seqlen equal
segments.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_gdn_block_ws_vs_flydsl as B

HK_MODEL, HV_MODEL = 16, 64  # GQA ratio 4
TPS = (1, 2, 4, 8)
SEQLENS = (1024, 2048, 4096, 8192)
TOTALS = (8192, 16384, 32768, 65536)

# Which profiler symbols make up the fused stage, and which the front end.
K5K6 = {
    "wf": "gdn_k2_kernel",
    "fly": "fwd_h_o_flydsl",
    "flyfix": "fwd_h_o_flydsl",
}
FRONT = {"wf": "gdn_k1_", "fly": "gdn_prepare", "flyfix": "gdn_prepare"}
BACKENDS = ("wf", "fly", "flyfix")
VARIANT_RE = re.compile(r"flydsl_vk_(\w+)")


def cu_scaled_variant(bh: int, cus: int, V: int = 128) -> str:
    """The BV variant whose grid is closest to one CTA wave on ``cus`` CUs.

    ``_hn_variant`` in the PR keys off absolute ``H*N`` thresholds (32 / 80)
    with no CU term, which puts one wave of CTAs on a 304-CU part and up to
    3.2 waves here.  Smallest BV whose grid still fits, largest otherwise.
    """
    for tag, bv in (("bv16", 16), ("bv32", 32), ("bv64w8", 64)):
        if -(-V // bv) * bh <= cus:
            return tag
    return "bv64w8"


def fly_fused_callable(t: dict):
    """FlyDSL K1..K4 + the fused K5+K6, forced past the 304-CU gate."""
    from aiter.ops.triton.gated_delta_net import chunk_gated_delta_rule_opt_vk

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
            prefill_metadata=t["meta"],
            use_chunk_flydsl=True,
            use_prepare_flydsl=True,
            fusion="always",
        )

    return run


def fly_fixed_callable(t: dict, variant: str):
    """The fused K5+K6 with an explicit BV instance.

    ``chunk_gated_delta_rule_opt_vk`` does not forward ``variant``, so this
    drives the wrapper directly.  K1..K4 runs once outside the timed region --
    it is variant-independent, and the ``fly`` column already measures it.
    """
    from aiter.ops.flydsl.linear_attention_prefill_kernels import (
        chunk_gated_delta_rule_fwd_h_o_flydsl,
        gdn_prepare_fwd_flydsl,
    )

    w, u, g_cumsum = gdn_prepare_fwd_flydsl(
        k=t["k"],
        v=t["v"],
        g=t["g"],
        beta=t["beta"],
        cu_seqlens=t["cu"],
        use_exp2=True,
        prefill_metadata=t["meta"],
    )
    o = t["v"].new_empty(t["v"].shape)

    def run():
        return chunk_gated_delta_rule_fwd_h_o_flydsl(
            q=t["q"],
            k=t["k"],
            w=w,
            u=u,
            g=g_cumsum,
            initial_state=t["h0"],
            output_final_state=True,
            cu_seqlens=t["cu"],
            prefill_metadata=t["meta"],
            variant=variant,
            o=o,
        )

    return run


def measure_one(backend: str, t: dict, variant: str | None = None) -> dict:
    if backend == "wf":
        run = B.make_callable("wf", t)
    elif backend == "fly":
        run = fly_fused_callable(t)
    else:
        run = fly_fixed_callable(t, variant)
    run()
    torch.cuda.synchronize()
    wall = B.bench_wall_us(run)
    kernels = B.profile_kernels(run)

    k5k6 = sum(us for n, us in kernels.items() if K5K6[backend] in n)
    front = sum(us for n, us in kernels.items() if FRONT[backend] in n)
    if k5k6 <= 0.0:
        raise RuntimeError(
            f"{backend}: no {K5K6[backend]} kernel ran -- the path fell back. "
            f"kernels: {sorted(kernels)}"
        )
    out = {
        "wall_us": wall,
        "k5k6_us": k5k6,
        "front_us": front,
        "kernel_sum_us": sum(kernels.values()),
    }
    for n in kernels:
        m = VARIANT_RE.search(n)
        if m:
            out["variant"] = m.group(1)
            break
    return out


def measure(tp: int, seqlen: int, total: int, cus: int) -> dict:
    hg, h = HK_MODEL // tp, HV_MODEL // tp
    n_seqs = total // seqlen
    B.HK, B.HV, B.TP = HK_MODEL, HV_MODEL, tp
    B.HG, B.H = hg, h
    B.FULL_PROMPT_LEN = seqlen

    cell: dict = {
        "tp": tp,
        "H": h,
        "Hg": hg,
        "seqlen": seqlen,
        "total_tokens": total,
        "n_seqs": n_seqs,
        "bh": h * n_seqs,
        "backends": {},
        "errors": {},
    }
    try:
        t = B.build_inputs(n_seqs)
    except Exception as exc:  # noqa: BLE001
        cell["skipped"] = f"inputs: {exc}"
        return cell

    want = cu_scaled_variant(cell["bh"], cus)
    cell["cu_scaled_variant"] = want
    for backend in BACKENDS:
        try:
            cell["backends"][backend] = measure_one(
                backend, t, want if backend == "flyfix" else None
            )
        except Exception as exc:  # noqa: BLE001
            cell["errors"][backend] = f"{type(exc).__name__}: {exc}"
        finally:
            torch.cuda.synchronize()

    del t
    torch.cuda.empty_cache()
    return cell


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tps", type=int, nargs="+", default=list(TPS))
    ap.add_argument("--seqlens", type=int, nargs="+", default=list(SEQLENS))
    ap.add_argument("--totals", type=int, nargs="+", default=list(TOTALS))
    ap.add_argument("--out", default=str(Path(__file__).with_name("k5k6_compare.json")))
    args = ap.parse_args()

    import aiter
    from aiter.ops.flydsl.linear_attention_prefill_kernels import (
        _device_cu_count,
        should_use_fused_gfx942,
    )

    props = torch.cuda.get_device_properties(0)
    out = {
        "gfx": props.gcnArchName,
        "cus": props.multi_processor_count,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "aiter_path": aiter.__file__,
        "flydsl_cu_count": _device_cu_count(),
        "auto_would_fuse_here": should_use_fused_gfx942(H=8, N=1, V=128),
        "Hk_model": HK_MODEL,
        "Hv_model": HV_MODEL,
        "gqa_ratio": HV_MODEL // HK_MODEL,
        "tps": list(args.tps),
        "seqlens": list(args.seqlens),
        "totals": list(args.totals),
        "num_iters": B.NUM_ITERS,
        "prof_iters": B.PROF_ITERS,
        "cells": [],
    }

    total_cells = len(args.tps) * len(args.seqlens) * len(args.totals)
    done, t0 = 0, time.time()
    for tp in args.tps:
        for seqlen in args.seqlens:
            for tok in args.totals:
                if tok % seqlen:
                    done += 1
                    continue
                cell = measure(tp, seqlen, tok, out["cus"])
                out["cells"].append(cell)
                done += 1
                bk = cell["backends"]
                head = (
                    f"[{done:3d}/{total_cells}] TP={tp} H={cell['H']:2d} "
                    f"s={seqlen:5d} B={cell['n_seqs']:2d} T={tok:6d} "
                    f"B*H={cell['bh']:5d}"
                )
                if all(b in bk for b in BACKENDS):
                    wf, fly, fix = (bk[b]["k5k6_us"] for b in BACKENDS)

                    def verdict(us: float, ref: float = wf) -> str:
                        r = ref / us
                        return f"fly {r:.2f}x" if r > 1 else f"opus {1 / r:.2f}x"

                    print(
                        f"{head}  K5+K6 wf={wf:9.1f} | "
                        f"fly={fly:9.1f} ({bk['fly'].get('variant', '?'):7s}) "
                        f"{verdict(fly):10s} | "
                        f"fix={fix:9.1f} ({cell['cu_scaled_variant']:7s}) "
                        f"{verdict(fix):10s}",
                        flush=True,
                    )
                else:
                    print(f"{head}  errors={cell['errors']}", flush=True)
                with open(args.out, "w") as fh:
                    json.dump(out, fh, indent=1)

    print(f"\nwrote {args.out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
