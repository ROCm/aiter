# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host side of the decode-shape (8-128 row) MoE expert GEMM pair built from
``decode_expert_tiles``: per-row-count tiling, launcher selection and the two launches
(GEMM1 + SwiGLU + MXFP4 quant, GEMM2 + top-k weighting) with the shared expert folded into
both. GEMM1 builds its 16-row expert blocks from ``topk_ids`` (no sort launch); routing
(router GEMM / top-k / activation quant) and the slot + shared reduction stay with the caller.

Tiling follows amd-kernels' ``experts_a4w4_small.py`` id-rows host and the per-row-count retune
of the shared tiles on top of it (rows per shared gate/up tile, columns per shared down tile,
prefetch depth, GEMM1 prologue issue priority); every choice keeps the K order, so outputs are
bit-identical across tilings.
"""

import torch

from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from .decode_expert_tiles import compile_gemm1_shared_first, compile_gemm2_shared_first
from .decode_tile_primitives import (
    id_pairs_for,
    max_sorted_rows,
    shared_fold_down_rows_for,
    shared_fold_tile_for,
    shared_g2_columns_for,
    small_config_for,
)

H, I, E, TOPK = 6144, 768, 128, 4

ROWS = (8, 128)
# rows where the retuned GEMM2 prefetch / GEMM1 prologue priority win
ASM_ROWS = (32, 128)


def supported(n: int) -> bool:
    return ROWS[0] <= n <= ROWS[1]


def tiles_for(n: int, asm: bool = True) -> dict:
    """Per-row-count tiling: 8 rows shared gate/up prefetch 3; 16 rows routed GEMM1 prefetch 6 +
    128-column GEMM2 tiles; to 64 rows shared down tiles 128 columns + 32-row shared gate/up
    tiles; to 128 rows 64-row shared gate/up tiles with prefetch 1. ``asm`` (32-128 rows):
    routed GEMM2 prefetch 3 and GEMM1 prologue at issue priority 3 (-0.3..-1.2 us per layer).
    """
    if n <= 16:
        cfg = {"sh_pf": 3} if n == 8 else {"g1_pf": 6, "g2_tn": 128}
    elif n <= 64:
        cfg = {"sh_tn": 128, "sh_bm": 32}
    else:
        cfg = {"sh_tn": 128, "sh_bm": 64, "sh_pf": 1}
    if asm and ASM_ROWS[0] <= n <= ASM_ROWS[1]:
        cfg = dict(cfg, g2_pf=3, g1_prio=(3, 0))
    return cfg


class DecodeExpertPlan:
    """Compiled launchers + grid sizes for ``n`` rows."""

    def __init__(self, n: int, asm: bool = True, tiles: dict | None = None):
        assert supported(n), n
        cfg = tiles_for(n, asm) if tiles is None else tiles
        kw, bm = small_config_for(n)
        assert kw == 4 and bm == 16
        self.n, self.bm = n, bm
        self.rows = max_sorted_rows(n, E, TOPK, bm)
        sh_bm, sh_kw, sh_pf = shared_fold_tile_for(n)
        sh_bm, sh_pf = cfg.get("sh_bm", sh_bm), cfg.get("sh_pf", sh_pf)
        self.nsh1 = (n + sh_bm - 1) // sh_bm * (24 // (4 // sh_kw))
        sh_tn = cfg.get("sh_tn", shared_g2_columns_for(n))
        g2_bm = shared_fold_down_rows_for(sh_bm, bm)
        self.nsh2 = (n + g2_bm - 1) // g2_bm * (H // sh_tn)
        g1_kw = {
            "sh_bm": sh_bm,
            "sh_kw": sh_kw,
            "sh_pf": sh_pf,
            "pairs": id_pairs_for(n),
        }
        if "g1_prio" in cfg:
            g1_kw["prio"] = cfg["g1_prio"]
        self.k1 = compile_gemm1_shared_first(
            kw=kw, pf=cfg.get("g1_pf", 3), bm=bm, **g1_kw
        )
        self.k2 = compile_gemm2_shared_first(
            tn=cfg.get("g2_tn", 256),
            bm=bm,
            pf=cfg.get("g2_pf", 2),
            sh_tn=sh_tn,
            sh_bm=g2_bm,
        )


_plans: dict = {}


def get_plan(n: int, asm: bool = True) -> DecodeExpertPlan:
    key = (n, asm)
    p = _plans.get(key)
    if p is None:
        p = _plans[key] = DecodeExpertPlan(n, asm)
    return p


def decode_expert_gemms(plan, xq, xs, q8, s8, w, shared, topk_ids, topk_w):
    """Routed + shared expert GEMM1 and GEMM2 for ``plan.n`` rows.

    Args:
        xq, xs: MXFP4 activation codes (n, H/2) and e8m0 scales (n, H/32).
        q8, s8: MXFP8 codes (n, H) and scales (n, H/32) of the shared expert's input.
        w: the routed experts' ``(w13, s13, w2, s2)`` (preshuffled MXFP4 + e8m0 scales).
        shared: the shared expert's ``(w13, s13, w2, s2)`` (MXFP8 + e8m0).
        topk_ids, topk_w: (n, 4) int32 ids / float32 weights.

    Returns:
        ``(slots, shared_out)``: (n, 4, H) BF16 weighted routed slot rows and (n, H) BF16
        shared-expert output, for the caller's slot + shared reduction.
    """
    n, dev = plan.n, xq.device
    w13, s13, w2, s2 = w
    sw13, ss13, sw2, ss2 = shared
    rows = plan.rows
    # GEMM1 reads topk_ids and publishes the block -> expert map (+ block count) and the rows
    eids = torch.empty(rows // 16 + 1, device=dev, dtype=torch.int32)
    nv = torch.empty(rows, device=dev, dtype=torch.int32)
    q = torch.empty((rows, 384), device=dev, dtype=torch.uint8)
    s = torch.empty((rows, 24), device=dev, dtype=torch.uint8)
    out = torch.empty((n, 4, H), device=dev, dtype=torch.bfloat16)
    hq = torch.empty((n, 768), device=dev, dtype=torch.uint8)
    hs = torch.empty((n, 24), device=dev, dtype=torch.uint8)
    sh_out = torch.empty((n, H), device=dev, dtype=torch.bfloat16)
    stream = torch.cuda.current_stream(dev)
    ids = topk_ids.view(-1)
    _run_compiled(
        plan.k1,
        xq.view(-1),
        xs.view(torch.uint8).view(-1),
        w13.view(-1),
        s13.view(-1),
        q.view(-1),
        s.view(-1),
        ids,
        eids,
        nv,
        q8.view(-1),
        s8.view(-1),
        sw13.view(-1),
        ss13.view(-1),
        hq.view(-1),
        hs.view(-1),
        n,
        rows,
        plan.nsh1,
        stream,
    )
    _run_compiled(
        plan.k2,
        q.view(-1),
        s.view(-1),
        w2.view(-1),
        s2.view(-1),
        out.view(-1),
        nv,
        topk_w.view(-1),
        eids,
        nv,
        hq.view(-1),
        hs.view(-1),
        sw2.view(-1),
        ss2.view(-1),
        sh_out.view(-1),
        n,
        rows,
        plan.nsh2,
        stream,
    )
    return out, sh_out
