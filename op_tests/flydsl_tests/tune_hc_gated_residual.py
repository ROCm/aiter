# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Joint (cartesian) tuner + exporter for the fused-K1 config of the two-stage
HC Gated-Residual op (SILOTIGER-1042).

Coordinate-descent misses cross-dimension interactions (the 2x2-wave win was
invisible to it, §6.7d), so this sweeps the K1 GEMM dimensions jointly:
  split-K:   split_k x sk_block_m x dn_block_n x (dn_m_waves, dn_n_waves)
  decouple:  block_k x dn_block_m x dn_block_n x (dn_m_waves, dn_n_waves)
For each token count it validates every candidate against the trusted heuristic
default, times the K1 call (combine+norm+down, fold_w=True -- the shipped config),
and keeps the fastest. ``--export`` writes the winners into the package's
``tuned_configs.json`` "k1" table (which flydsl_k1_combine_norm_down consults:
explicit arg > tuned plan > heuristic).

    HIP_VISIBLE_DEVICES=2 python tune_hc_gated_residual.py --tokens 4096 8192 --export
"""

import argparse
import json
import os

import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.hyper_connection_gated_residual import (
    fold_norm_weight,
    merge_gr_two_stage_weight,
)
from aiter.ops.flydsl.kernels.hyper_connection_gated_residual.k1 import (
    flydsl_k1_combine_norm_down,
)

HC, HS, LOWRANK = 4, 2560, 320
HIDDEN = HC * HS
EPS = 1e-6
N_PAD = 384
MMA_N = MMA_M = 16  # gfx950 bf16 mma tile
K_TILES = HIDDEN // 64  # 160

TABLE = os.path.join(
    os.path.dirname(__file__), "..", "..", "aiter", "ops", "flydsl", "kernels",
    "hyper_connection_gated_residual", "tuned_configs.json",
)
TABLE = os.path.normpath(TABLE)


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
        "w_inject": _rand(HC, HIDDEN, scale=HIDDEN**-0.5),
    }
    merged = merge_gr_two_stage_weight(inp["w_down"], inp["w_inject"], HC)
    inp["_w_folded"] = fold_norm_weight(merged, inp["norm_weight"], HC)
    return inp


def timeit(fn, iters=60, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True)
    e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0


def k1(inp, **cfg):
    # Tune the SHIPPED configuration: folded weights + fold_w=True, so the table
    # reflects the metric _k1_then_k2 actually runs.
    cfg = {k: v for k, v in cfg.items() if k != "method"}
    return flydsl_k1_combine_norm_down(
        inp["residual"], inp["block_output"], inp["injection"], inp["norm_weight"],
        inp["_w_folded"], LOWRANK, HC, EPS, fold_w=True, use_tuned=False, **cfg,
    )


def decouple_candidates(M):
    cfgs = []
    for bk in (64, 128):
        for bm in (32, 64, 128):
            if M % bm or HIDDEN % bk:
                continue
            for bn in (64, 128, 192, 384):
                if N_PAD % bn:
                    continue
                if bk == 128 and bn > 128:  # LDS / down-pipe limit
                    continue
                for mw, nw in ((1, 4), (2, 2), (2, 4), (4, 2), (1, 2)):
                    threads = mw * nw * 64
                    if bm % (mw * MMA_M) or bn % (nw * MMA_N):
                        continue
                    if bm * bk < threads * 8:  # async-LDS coverage
                        continue
                    cfgs.append(dict(method="decouple", split_k=1, block_k=bk,
                                     dn_block_m=bm, dn_block_n=bn,
                                     dn_m_waves=mw, dn_n_waves=nw))
    return cfgs


def splitk_candidates(M):
    cfgs = []
    for sk in (4, 8, 16):
        if K_TILES % sk:
            continue
        for skbm in (32, 64):
            if M % skbm:
                continue
            for bn in (128, 384):
                if N_PAD % bn:
                    continue
                for mw, nw in ((1, 4), (2, 2), (1, 2)):
                    threads = mw * nw * 64
                    if skbm % (mw * MMA_M) or bn % (nw * MMA_N):
                        continue
                    if skbm * 64 < threads * 8:  # async-LDS coverage (block_k=64)
                        continue
                    cfgs.append(dict(method="splitk", split_k=sk, sk_block_m=skbm,
                                     block_k=64, dn_block_n=bn,
                                     dn_m_waves=mw, dn_n_waves=nw))
    return cfgs


def candidates(M):
    if M <= 2048:
        return splitk_candidates(M) + decouple_candidates(M)
    return decouple_candidates(M)


def _entry_from_cfg(cfg, us):
    e = {"method": cfg["method"], "_us": round(us, 3)}
    if cfg["method"] == "decouple":
        e.update(block_k=cfg["block_k"], dn_block_m=cfg["dn_block_m"],
                 dn_block_n=cfg["dn_block_n"], dn_m_waves=cfg["dn_m_waves"],
                 dn_n_waves=cfg["dn_n_waves"])
    else:
        e.update(split_k=cfg["split_k"], sk_block_m=cfg["sk_block_m"],
                 block_k=cfg["block_k"], dn_block_n=cfg["dn_block_n"],
                 dn_m_waves=cfg["dn_m_waves"], dn_n_waves=cfg["dn_n_waves"])
    return e


def validate(inp, cfg, ref_r2, ref_packed):
    r2, packed = k1(inp, **cfg)
    torch.cuda.synchronize()
    e_r2 = (r2.float() - ref_r2.float()).abs().max().item()
    e_pk = (packed.float() - ref_packed.float()).abs().max().item()
    return max(e_r2, e_pk)


ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=int, nargs="+",
                default=[64, 128, 256, 512, 1024, 2048, 4096, 8192, 12288,
                         16384, 24576, 32768, 49152, 65536])
ap.add_argument("--export", action="store_true")
ap.add_argument("--tol", type=float, default=0.3)
args = ap.parse_args()

arch = get_gfx()
results = {}
for M in args.tokens:
    inp = make_inputs(M)
    ref_r2, ref_packed = k1(inp)  # trusted heuristic default (use_tuned=False)
    torch.cuda.synchronize()
    t_def = timeit(lambda: k1(inp))
    print(f"\nM={M}  heuristic-default={t_def:.1f}u")
    best = (t_def, None)
    for cfg in candidates(M):
        try:
            err = validate(inp, cfg, ref_r2, ref_packed)
            if err > args.tol:
                continue
            t = timeit(lambda: k1(inp, **cfg))
            if cfg["method"] == "decouple":
                tag = f"dec bk{cfg['block_k']} bm{cfg['dn_block_m']} bn{cfg['dn_block_n']} {cfg['dn_m_waves']}x{cfg['dn_n_waves']}"
            else:
                tag = f"sk{cfg['split_k']} skbm{cfg['sk_block_m']} bn{cfg['dn_block_n']} {cfg['dn_m_waves']}x{cfg['dn_n_waves']}"
            mark = ""
            if t < best[0]:
                best = (t, cfg)
                mark = " *"
            print(f"   {tag:<34} {t:8.1f}u  err={err:.3f}{mark}")
        except Exception as ex:  # noqa: BLE001 - report and skip bad candidates
            print(f"   {cfg}  ERR {type(ex).__name__}: {str(ex)[:36]}")
    bt, bc = best
    if bc is None:
        print(f"   -> M={M}: heuristic default already best ({bt:.1f}u)")
    else:
        print(f"   -> M={M}: best {bt:.1f}u ({t_def / bt:.2f}x vs default) {bc}")
        results[str(M)] = _entry_from_cfg(bc, bt)

if args.export and results:
    with open(TABLE) as f:
        tbl = json.load(f)
    tbl.setdefault(arch, {}).setdefault("k1", {}).update(results)
    with open(TABLE, "w") as f:
        json.dump(tbl, f, indent=2)
        f.write("\n")
    print(f"\nexported {len(results)} entries to {TABLE} under {arch}/k1")
elif args.export:
    print("\nnothing to export (heuristic default won everywhere)")
