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

from .decode_expert_tiles import (
    E,
    H,
    I,
    compile_gemm1_shared_first,
    compile_gemm2_shared_first,
)
from .decode_id_routing import TOPK
from .decode_tile_primitives import (
    id_pairs_for,
    max_sorted_rows,
    shared_fold_down_rows_for,
    shared_fold_tile_for,
    shared_g2_columns_for,
    small_config_for,
)

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

    def __init__(
        self,
        n: int,
        asm: bool = True,
        tiles: dict | None = None,
        w13_split: bool = False,
    ):
        assert supported(n), n
        cfg = tiles_for(n, asm) if tiles is None else tiles
        kw, bm = small_config_for(n)
        assert kw == 4 and bm == 16
        self.n, self.bm = n, bm
        self.rows = max_sorted_rows(n, E, TOPK, bm)
        sh_bm, sh_kw, sh_pf = shared_fold_tile_for(n)
        sh_bm, sh_kw, sh_pf = (
            cfg.get("sh_bm", sh_bm),
            cfg.get("sh_kw", sh_kw),
            cfg.get("sh_pf", sh_pf),
        )
        self.nsh1 = (n + sh_bm - 1) // sh_bm * (24 // (4 // sh_kw))
        sh_tn = cfg.get("sh_tn", shared_g2_columns_for(n))
        g2_bm = shared_fold_down_rows_for(sh_bm, bm)
        self.nsh2 = (n + g2_bm - 1) // g2_bm * (H // sh_tn)
        g1_kw = {
            "sh_bm": sh_bm,
            "sh_kw": sh_kw,
            "sh_pf": sh_pf,
            "pairs": id_pairs_for(n),
            "w13_split": w13_split,
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


def get_plan(
    n: int, asm: bool = True, w13_split: bool = False, tiles: dict | None = None
) -> DecodeExpertPlan:
    """``tiles``: a tuning config (``decode_moe_cfg.ExpertCfg.to_tiles()``) replacing ``tiles_for(n, asm)``; every
    field a tile knob of ``tiles_for`` plus ``sh_kw`` (shared gate/up K-waves), all bit-identical to the default
    tiling. ``w13_split``: the routed ``w13`` operand is in the preshuffled ``[E, 2, I/16, block]`` layout (gate
    blocks, then up blocks, as ``shuffle_weight`` leaves it) instead of the gate / up interleaved
    ``[E, I/16, 2, block]``; same outputs. The scales ``s13`` stay in the interleaved layout.
    """
    key = (n, asm, w13_split, None if tiles is None else tuple(sorted(tiles.items())))
    p = _plans.get(key)
    if p is None:
        p = _plans[key] = DecodeExpertPlan(n, asm, tiles=tiles, w13_split=w13_split)
    return p


def _bytes(t, nbytes, name, dev):
    """Metadata-only check of one raw-buffer operand (no sync, graph-safe) and its flat uint8
    view. The tiles bind their buffer resources from ``plan.n``, not from the tensors, so a
    short, strided or foreign-device operand would read out of bounds or mis-address."""
    assert t.device == dev, f"{name} is on {t.device}, expected {dev}"
    assert t.is_contiguous(), f"{name} must be contiguous"
    assert t.element_size() == 1, f"{name} must be a one-byte dtype, got {t.dtype}"
    assert t.numel() == nbytes, f"{name} has {t.numel()} bytes, expected {nbytes}"
    return t.view(torch.uint8).view(-1)


def _check_operands(n, xq, xs, q8, s8, w, shared, topk_ids, topk_w):
    dev = xq.device
    assert dev.type == "cuda", dev
    w13, s13, w2, s2 = w
    sw13, ss13, sw2, ss2 = shared
    ops = (
        _bytes(xq, n * (H // 2), "xq", dev),
        _bytes(xs, n * (H // 32), "xs", dev),
        _bytes(w13, E * I * H, "w13", dev),
        _bytes(s13, E * I * (H // 16), "s13", dev),
        _bytes(w2, E * H * I // 2, "w2", dev),
        _bytes(s2, E * H * (I // 32), "s2", dev),
        _bytes(q8, n * H, "q8", dev),
        _bytes(s8, n * (H // 32), "s8", dev),
        _bytes(sw13, 2 * I * H, "shared w13", dev),
        _bytes(ss13, I * (H // 16), "shared s13", dev),
        _bytes(sw2, H * I, "shared w2", dev),
        _bytes(ss2, H * (I // 32), "shared s2", dev),
    )
    for t, dt, name in (
        (topk_ids, torch.int32, "topk_ids"),
        (topk_w, torch.float32, "topk_w"),
    ):
        assert (
            t.device == dev and t.is_contiguous()
        ), f"{name} must be contiguous on {dev}"
        assert t.dtype == dt and t.shape == (
            n,
            TOPK,
        ), f"{name}: {t.dtype} {tuple(t.shape)}"
    return ops


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
    xq, xs, w13, s13, w2, s2, q8, s8, sw13, ss13, sw2, ss2 = _check_operands(
        n, xq, xs, q8, s8, w, shared, topk_ids, topk_w
    )
    rows = plan.rows
    # the compiled handle is bound to the device it was first launched on: key it per device
    key = ("dev", dev.index)
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
        xq,
        xs,
        w13,
        s13,
        q.view(-1),
        s.view(-1),
        ids,
        eids,
        nv,
        q8,
        s8,
        sw13,
        ss13,
        hq.view(-1),
        hs.view(-1),
        n,
        rows,
        plan.nsh1,
        stream,
        specialization_key=key,
    )
    _run_compiled(
        plan.k2,
        q.view(-1),
        s.view(-1),
        w2,
        s2,
        out.view(-1),
        nv,
        topk_w.view(-1),
        eids,
        nv,
        hq.view(-1),
        hs.view(-1),
        sw2,
        ss2,
        sh_out.view(-1),
        n,
        rows,
        plan.nsh2,
        stream,
        specialization_key=key,
    )
    return out, sh_out
