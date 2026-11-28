# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Probe 3: does fused_moe break when the weight tensor declares unrouted experts?

``moonep_dispatch_combine_op.py:310`` claims, as measured fact:

    aiter's quantised MoE gives a **zero** result when the weight tensor
    declares experts that no row routes to -- any spare slot breaks it, while
    every expert count from 1 to 48 is exact when all of them are used.

That, not VA contiguity, is why the MoonEP experts step splits into two
``fused_moe`` calls.  It also decides whether a row-contiguous ``[E + B]``
weight range (probe 1 + 2) buys anything: on that layout every rank declares
all E experts but routes to only the handful the planner gave it, which is
exactly the failing shape -- if the claim holds unconditionally.

Four configurations per quant type, all compared against a reference that runs
the *compacted* weight tensor (only routed experts, ids remapped), i.e. the
"all declared experts are used" case the claim says is exact:

    A  full weights, global ids, no mask          <- naive single call
    B  full weights, global ids, mask = all ones  <- does the EP accumulate
                                                     path tolerate spares?
    C  compact weights, global ids, mask = routed <- aiter's official EP path

Single GPU; nothing here depends on VMM or on multiple ranks.

Run:
    python3 op_tests/probe_spare_expert_slots.py
"""

from __future__ import annotations

import argparse
import sys

import torch
from einops import rearrange

from aiter import ActivationType, QuantType, dtypes, pertoken_quant
from aiter.fused_moe import fused_moe

BLK = 128


def make_weights(E, H, I, dev, seed=7):
    g = torch.Generator(device=dev).manual_seed(seed)
    w1 = (torch.randn(E, 2 * I, H, generator=g, device=dev) * 0.05).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I, generator=g, device=dev) * 0.05).to(torch.bfloat16)
    return w1, w2


def block_quant(w, blk_n=BLK, blk_k=BLK):
    """fp8 per-(128x128) block quant, mirroring op_tests/test_moe_blockscale.py."""
    E, N, K = w.shape
    tmp = rearrange(
        w.view(-1, N // blk_n, blk_n, K // blk_k, blk_k),
        "e bn n bk k -> e bn bk (n k)",
    ).contiguous()
    q, s = pertoken_quant(tmp, quant_dtype=dtypes.fp8)
    q = rearrange(
        q.view(-1, N // blk_n, K // blk_k, blk_n, blk_k),
        "e bn bk n k -> e (bn n) (bk k)",
    ).contiguous()
    return q, s.view(E, -1)


def run(x, w1, w2, ids, weights, quant, s1=None, s2=None, mask=None):
    return fused_moe(
        x, w1, w2, weights, ids, mask,
        ActivationType.Silu, quant_type=quant, w1_scale=s1, w2_scale=s2,
    )


def report(name, ref, got):
    if got is None:
        print(f"    {name:<34s} ERROR")
        return
    zero = bool((got == 0).all())
    d = (ref.float() - got.float()).abs()
    md = d.max().item()
    rel = md / max(ref.float().abs().max().item(), 1e-9)
    verdict = "ZERO" if zero else ("OK" if rel < 2e-2 else "WRONG")
    print(f"    {name:<34s} {verdict:<6s} max|d|={md:.4e} rel={rel:.3e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--inter", type=int, default=512)
    ap.add_argument("--tokens", type=int, default=512)
    args = ap.parse_args()

    dev = "cuda:0"
    torch.cuda.set_device(0)
    H, I, n = args.hidden, args.inter, args.tokens

    cases = [
        ("all 16 routed",        16, list(range(16))),
        ("18 decl, 16 routed",   18, list(range(16))),   # the MoonEP E+B shape
        ("16 decl, 8 trailing",  16, list(range(8))),
        ("16 decl, 8 even",      16, list(range(0, 16, 2))),
        ("48 decl, 6 scattered", 48, [0, 7, 13, 22, 35, 44]),
    ]
    quants = [("No (bf16)", QuantType.No), ("per_128x128 (fp8)", QuantType.per_128x128)]

    g = torch.Generator(device=dev).manual_seed(4242)
    x = (torch.randn(n, H, generator=g, device=dev) * 0.1).to(torch.bfloat16)

    for qname, quant in quants:
        print(f"\n{'=' * 78}\nquant = {qname}\n{'=' * 78}")
        for cname, E_decl, routed in cases:
            k = len(routed)
            ids_global = torch.tensor(
                [routed[i % k] for i in range(n)], dtype=torch.int32, device=dev
            ).view(n, 1)
            remap = {g_: i for i, g_ in enumerate(routed)}
            ids_local = torch.tensor(
                [remap[int(v)] for v in ids_global.flatten()],
                dtype=torch.int32, device=dev,
            ).view(n, 1)
            tw = torch.ones(n, 1, dtype=torch.float32, device=dev)

            w1, w2 = make_weights(E_decl, H, I, dev)
            cw1, cw2 = w1[routed].contiguous(), w2[routed].contiguous()
            if quant == QuantType.No:
                f1, f2, fs1, fs2 = w1, w2, None, None
                c1, c2, cs1, cs2 = cw1, cw2, None, None
            else:
                f1, fs1 = block_quant(w1); f2, fs2 = block_quant(w2)
                c1, cs1 = block_quant(cw1); c2, cs2 = block_quant(cw2)

            print(f"\n  [{cname}]  E_decl={E_decl} routed={k} spare={E_decl - k}")
            try:
                ref = run(x, c1, c2, ids_local, tw, quant, cs1, cs2)
            except Exception as e:  # noqa: BLE001
                print(f"    reference (compact, no spare) ERROR: {type(e).__name__}: {e}")
                continue
            if bool((ref == 0).all()):
                print("    !! reference itself is all-zero -- case is not usable")
                continue

            ones = torch.ones(E_decl, dtype=torch.int32, device=dev)
            m_routed = torch.zeros(E_decl, dtype=torch.int32, device=dev)
            m_routed[torch.tensor(routed, device=dev)] = 1

            for name, fn in (
                ("A full/global/no-mask",
                 lambda: run(x, f1, f2, ids_global, tw, quant, fs1, fs2)),
                ("B full/global/mask=ones",
                 lambda: run(x, f1, f2, ids_global, tw, quant, fs1, fs2, ones)),
                ("C compact/global/mask=routed",
                 lambda: run(x, c1, c2, ids_global, tw, quant, cs1, cs2, m_routed)),
            ):
                try:
                    got = fn()
                    torch.cuda.synchronize()
                except Exception as e:  # noqa: BLE001
                    print(f"    {name:<34s} ERROR {type(e).__name__}: {e}")
                    continue
                report(name, ref, got)

    print(
        "\nlegend: OK = matches the compact reference; ZERO = all-zero output; "
        "WRONG = ran but wrong numbers"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
