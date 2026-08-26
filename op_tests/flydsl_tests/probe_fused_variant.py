# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Is the FlyDSL fused K5+K6 slow on 80 CUs, or is its variant rule mis-tuned?

``_hn_variant`` picks BV from absolute ``H*N`` thresholds (``<=32 -> bv16``,
``<=80 -> bv32``, else ``bv64w8``) with no CU term.  On 304 CUs those land near
one wave of CTAs; on this 80-CU MI308X the same thresholds ask for up to 3.2x
the CTAs the device can hold.  This forces each variant explicitly and reports
the best, so the grid's numbers can be split into "the kernel is slower here"
and "the host picked the wrong instance here".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_gdn_block_ws_vs_flydsl as B

VARIANTS = ("bv16", "bv32", "bv64", "bv64w8")
CU = 80


def fused_callable(t, variant, hg, h):
    from aiter.ops.flydsl.linear_attention_prefill_kernels import (
        chunk_gated_delta_rule_fwd_h_o_flydsl,
        gdn_prepare_fwd_flydsl,
    )

    # K1..K4 once, outside the timed region: this probe is about the K5+K6
    # instance choice, and the prepare stage is identical across variants.
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


def main() -> int:
    from aiter.ops.flydsl.kernels.chunk_gated_delta_h_gfx942 import (
        select_fused_variant,
    )

    rows = []
    # seqlen fixed: B*H is the only variable that moved the grid results, and
    # holding the chain length constant keeps the columns comparable.
    seqlen = 8192
    for tp, n_seqs in ((8, 1), (8, 2), (8, 4), (8, 8), (4, 8), (2, 8), (1, 8)):
        hg, h = 16 // tp, 64 // tp
        B.HK, B.HV, B.TP, B.HG, B.H = 16, 64, tp, hg, h
        B.FULL_PROMPT_LEN = seqlen
        t = B.build_inputs(n_seqs)
        bh = h * n_seqs
        auto = select_fused_variant(H=h, N=n_seqs, V=128)

        row = {
            "tp": tp,
            "H": h,
            "n_seqs": n_seqs,
            "seqlen": seqlen,
            "bh": bh,
            "auto_variant": auto,
            "times": {},
            "ctas": {},
        }
        for v in VARIANTS:
            bv = int(v.replace("w8", "")[2:])
            row["ctas"][v] = -(-128 // bv) * bh
            try:
                run = fused_callable(t, v, hg, h)
                run()
                torch.cuda.synchronize()
                row["times"][v] = B.bench_wall_us(run)
            except Exception as exc:  # noqa: BLE001
                row["times"][v] = None
                row.setdefault("errors", {})[v] = f"{type(exc).__name__}: {exc}"
        ok = {k: v for k, v in row["times"].items() if v}
        best = min(ok, key=ok.get)
        row["best_variant"] = best
        row["auto_penalty"] = ok[auto] / ok[best] if auto in ok and best in ok else None
        rows.append(row)

        cells = "  ".join(
            f"{v}={ok[v]:8.1f}" if v in ok else f"{v}={'n/a':>8s}" for v in VARIANTS
        )
        print(
            f"B*H={bh:4d} TP={tp} H={h:2d} B={n_seqs}  auto={auto:7s} "
            f"best={best:7s} penalty={row['auto_penalty']:.2f}x  {cells}",
            flush=True,
        )
        del t
        torch.cuda.empty_cache()

    out = Path(__file__).with_name("fused_variant_probe.json")
    with open(out, "w") as fh:
        json.dump({"cus": CU, "seqlen": seqlen, "rows": rows}, fh, indent=1)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
