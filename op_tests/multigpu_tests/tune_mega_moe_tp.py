# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tune the fused TP MegaMoE launch config per model / TP / comm mode / tokens.

Per cell: sweep the :class:`LaunchCfg` variants (row tile, dynamic vs static
schedule, E4M3 vs bf16 route rows, LL packets), time each (CUDA graph, slowest
rank, best of --rounds), gate on the split path's output (rel L2 < --rtol, no
watchdog, replay == eager) and merge the fastest into the tuned CSV::

    python op_tests/multigpu_tests/tune_mega_moe_tp.py --models glm5 m3 \\
        --tokens 8 16 32 64 96 128 256 512 1024 2048 --comm-modes ag_rs ar_ar

One process drives all --tp GPUs (like ``test_mega_moe_TP.py --single-process``);
each cell runs in its own subprocess. HIP_VISIBLE_DEVICES pins the GPUs.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_mega_moe_TP as T
import torch

from aiter.ops.flydsl.kernels.mega_moe_tp.mega_moe_tp import (
    CSV_CFG,
    CSV_KEY,
    LDS_LIMIT,
    LaunchCfg,
    tuned_config_path,
)

CSV_COLS = [
    *CSV_KEY[:4],
    "token",
    *CSV_KEY[4:],
    *CSV_CFG,
    "us",
    "model",
]


def idle_gpus(n: int, wait_s: int = 1800) -> str:
    t0 = time.time()
    while True:
        busy, cards = set(), []
        for _ in range(5):
            smi = subprocess.run(
                ["rocm-smi", "--showuse", "--showmemuse", "--json"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout
            d = {k: v for k, v in json.loads(smi).items() if k.startswith("card")}
            cards = sorted(int(k[4:]) for k in d)
            busy |= {
                int(k[4:])
                for k, v in d.items()
                if float(v.get("GPU use (%)", 0)) >= 5
                or float(v.get("GPU Memory Allocated (VRAM%)", 0)) >= 2
            }
            time.sleep(0.4)
        free = sorted(
            (g for g in cards if g not in busy), key=lambda g: not 4 <= g <= 7
        )
        if len(free) >= n:
            return ",".join(map(str, sorted(free[:n])))
        if time.time() - t0 > wait_s:
            raise RuntimeError(f"no {n} idle GPUs after {wait_s}s")
        print(f"[tune] waiting for {n} idle GPUs (busy: {sorted(busy)})", flush=True)
        time.sleep(30)


def _valid(eng, cfg: LaunchCfg) -> bool:
    return 1 <= cfg.mt <= 6 and eng._lds(cfg.mt, cfg.dyn) <= LDS_LIMIT


def schedules(eng, m: int) -> list[LaunchCfg]:
    tot = m * eng.tp
    base = eng.default_config(m)
    rpe = (tot * eng.K + eng.E - 1) // eng.E
    out = []
    for dyn in (False, True):
        if dyn and not (
            tot <= eng.dyn_max and eng.I // 128 >= 2 and tot * eng.K <= 4 * eng.E
        ):
            continue
        mt = eng._fit_mt((rpe + 15) // 16, dyn)
        for llr in [True, False] if base.ll else [False]:
            out.append(
                LaunchCfg(mt=mt, dyn=dyn, route_fp8=not dyn, ll=base.ll, llr=llr)
            )
    return [c for c in out if _valid(eng, c)]


def refinements(best: LaunchCfg) -> list[LaunchCfg]:
    d = best.__dict__
    return [
        LaunchCfg(**{**d, "mt": best.mt - 1}),
        LaunchCfg(**{**d, "mt": best.mt + 1}),
        LaunchCfg(**{**d, "route_fp8": not best.route_fp8}),
        LaunchCfg(**{**d, "ll": not best.ll, "llr": False}),
    ]


def merge_csv(path: str, rows: list[dict]) -> None:
    old = []
    if os.path.exists(path):
        with open(path) as f:
            old = list(csv.DictReader(f))
    key = lambda r: tuple(str(r[k]) for k in (*CSV_KEY, "token"))
    new = {key(r) for r in rows}
    allr = [r for r in old if key(r) not in new] + rows
    allr.sort(key=lambda r: (r["model"], r["comm_mode"], int(r["tp"]), int(r["token"])))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, lineterminator="\n")
        w.writeheader()
        for r in allr:
            w.writerow({c: r.get(c, "") for c in CSV_COLS})


def tune_cell(a, name: str, mode: str, tok: int) -> dict | None:
    T._flydsl_multi_device()
    from p2p_collectives import P2PGroup

    from aiter.ops.flydsl.kernels.symmetric_arena import PeerArenaGroup

    tp, m = a.tp, tok // a.tp
    devices = [torch.device("cuda", i) for i in range(tp)]
    torch.cuda.set_device(devices[0])
    group = PeerArenaGroup(devices)
    ctxs = [T.DistCtx(rank=r, world=tp, device=d) for r, d in enumerate(devices)]
    targs = argparse.Namespace(sp_warmup=2, rounds=a.rounds, iters=a.iters)
    p2p = P2PGroup(devices, m)
    shape = T.MODELS[name]
    weights = []
    for c in ctxs:
        with torch.cuda.device(c.device):
            weights.append(T.build_sharded_weights(shape, c, tp, 0))
    inputs = [T.make_inputs(shape, c, tp, tok, 0, "balanced", mode) for c in ctxs]
    split_cls = T.SplitTpMoeAR if mode == "ar_ar" else T.SplitTpMoe
    split = []
    for r, c in enumerate(ctxs):
        with torch.cuda.device(c.device):
            split.append(split_cls(weights[r], c, m, comm=p2p.comm(r)))
    y = [t.clone() for t in T._sp_run(devices, lambda r: split[r](inputs[r]))]
    split_us = float("nan")
    if not a.no_split:
        split_us, _ = T._sp_time_graph(devices, lambda r: split[r](inputs[r]), targs)
    fused = []
    for r, c in enumerate(ctxs):
        with torch.cuda.device(c.device):
            fused.append(T.MegaMoeTP(weights[r], c, m, group=group, comm_mode=mode))
    engs = [f.engine.engine for f in fused]
    best, tried = None, set()
    tag = f"[tune] {name} {mode} M={tok}"

    def run(cfg) -> bool:
        nonlocal best
        if cfg in tried or not _valid(engs[0], cfg):
            return True
        tried.add(cfg)
        for e in engs:
            e._cfgs[m] = cfg
        try:
            yf = [t.clone() for t in T._sp_run(devices, lambda r: fused[r](inputs[r]))]
            errs = [e.poll_errors() for e in engs]
            err = T._sp_rel_l2(yf, y)
            if any(errs) or not err < a.rtol:
                print(f"{tag} {cfg}: rejected errs={errs} rel_l2={err:.4f}", flush=True)
                return not any(errs)
            us, rep = T._sp_time_graph(devices, lambda r: fused[r](inputs[r]), targs)
            rerr = T._sp_rel_l2(rep, yf) if rep is not None else float("nan")
            if not rerr < 1e-3:
                print(f"{tag} {cfg}: rejected replay rel_l2={rerr:.4f}", flush=True)
                return True
        except Exception as exc:  # noqa: BLE001 - a variant that fails is skipped
            print(f"{tag} {cfg}: failed {exc}", flush=True)
            return True
        print(f"{tag} {cfg}: {us:.1f} us (rel_l2 {err:.4f})", flush=True)
        if best is None or us < best[0] * 0.99:
            best = (us, cfg)
        return True

    if all(run(c) for c in schedules(engs[0], m)) and best is not None:
        for c in refinements(best[1]):
            if not run(c):
                break
    if best is None:
        print(f"{tag}: no valid config", flush=True)
        return None
    us, cfg = best
    e = engs[0]
    return {
        "gfx": e.gfx,
        "cu_num": e.n_cta,
        "tp": tp,
        "comm_mode": mode,
        "token": tok,
        "model_dim": e.H,
        "inter_dim": e.I,
        "expert": e.E,
        "topk": e.K,
        "act": e.activation,
        "block_m": cfg.block_m,
        "dyn": int(cfg.dyn),
        "route_fp8": int(cfg.route_fp8),
        "ll": int(cfg.ll),
        "llr": int(cfg.llr),
        "us": f"{us:.2f}",
        "split_us": f"{split_us:.2f}",
        "speedup": f"{split_us / us:.3f}" if not math.isnan(split_us) else "",
        "model": name,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--models", nargs="+", default=["glm5", "m3"])
    p.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 96, 128, 256, 512, 1024, 2048],
    )
    p.add_argument("--comm-modes", nargs="+", default=["ag_rs", "ar_ar"])
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument(
        "--rtol", type=float, default=0.06, help="fused vs split rel L2 gate"
    )
    p.add_argument("--out", default=tuned_config_path())
    p.add_argument("--no-split", action="store_true", help="skip timing the split path")
    p.add_argument(
        "--cell", nargs=3, metavar=("MODEL", "MODE", "TOKENS"), help=argparse.SUPPRESS
    )
    a = p.parse_args()
    if a.cell is not None:
        row = tune_cell(a, a.cell[0], a.cell[1], int(a.cell[2]))
        if row is not None:
            print(f"[tune] BEST {row}", flush=True)
            print("TUNE_ROW " + json.dumps(row), flush=True)
        return 0
    os.environ.setdefault("HIP_VISIBLE_DEVICES", idle_gpus(a.tp))
    print(f"[tune] GPUs {os.environ['HIP_VISIBLE_DEVICES']} -> {a.out}", flush=True)
    for name, mode, tok in itertools.product(
        a.models, a.comm_modes, sorted(set(a.tokens))
    ):
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--cell",
            name,
            mode,
            str(tok),
            "--tp",
            str(a.tp),
            "--iters",
            str(a.iters),
            "--rounds",
            str(a.rounds),
            "--rtol",
            str(a.rtol),
            "--out",
            a.out,
        ] + (["--no-split"] if a.no_split else [])
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600, check=False
        )
        rows = []
        for line in r.stdout.splitlines():
            if line.startswith("[tune]"):
                print(line, flush=True)
            elif line.startswith("TUNE_ROW "):
                rows.append(json.loads(line[len("TUNE_ROW ") :]))
        if r.returncode != 0:
            tail = "\n".join((r.stderr or "").splitlines()[-5:])
            print(
                f"[tune] {name} {mode} M={tok}: cell exited {r.returncode}\n{tail}",
                flush=True,
            )
        if rows:
            merge_csv(a.out, rows)
    print(f"[tune] done -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
