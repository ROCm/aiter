# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tune the fused TP MegaMoE launch config per model / TP / comm mode / tokens.

For every (model, comm mode, token count) cell this sweeps the launch variants
of :class:`aiter.ops.flydsl.kernels.mega_moe_tp.fused_tp_engine.LaunchCfg` --
row tile ``block_m`` (16 * MT), weight ring depth ``nsk``, dynamic vs static
schedule, E4M3 vs bf16 route rows, column split of leftover experts -- times
each one (CUDA graph, every rank replayed from its own thread, slowest rank,
best of --rounds), gates it on the split path's output (rel L2 < --rtol, no
watchdog), and writes the fastest into the tuned CSV the engine reads::

    aiter/configs/model_configs/mega_moe_tp_a4w4_tuned.csv

One process drives all TP GPUs (peer access), like
``test_mega_moe_TP.py --single-process``. It waits for --tp idle GPUs
(HIP_VISIBLE_DEVICES pins them)::

    python op_tests/multigpu_tests/tune_mega_moe_tp.py --models glm5 m3 \\
        --tokens 8 16 32 64 96 128 256 512 1024 2048 --comm-modes ag_rs ar_ar
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_mega_moe_TP as T  # noqa: E402  (sets the tuned-a4w4 env first)
import torch  # noqa: E402

from aiter.ops.flydsl.kernels.mega_moe_tp.fused_tp_engine import (  # noqa: E402
    CSV_CFG,
    CSV_KEY,
    LDS_LIMIT,
    LaunchCfg,
    tuned_config_path,
)

CSV_COLS = list(CSV_KEY[:4]) + ["token"] + list(CSV_KEY[4:]) + list(CSV_CFG) + [
    "us",
    "split_us",
    "speedup",
    "model",
]


def idle_gpus(n: int, wait_s: int = 1800) -> str:
    """n GPUs with < 5% use and < 2% VRAM over 5 samples (prefers 4-7)."""
    t0 = time.time()
    while True:
        busy, d = set(), {}
        for _ in range(5):
            out = subprocess.run(
                ["rocm-smi", "--showuse", "--showmemuse", "--json"],
                capture_output=True,
                text=True,
            ).stdout
            d = json.loads(out)
            for k, v in d.items():
                if k.startswith("card") and (
                    float(v.get("GPU use (%)", 0)) >= 5
                    or float(v.get("GPU Memory Allocated (VRAM%)", 0)) >= 2
                ):
                    busy.add(int(k[4:]))
            time.sleep(0.4)
        cards = sorted(int(k[4:]) for k in d if k.startswith("card"))
        free = [g for g in cards if g not in busy]
        free = [g for g in free if 4 <= g <= 7] + [g for g in free if not 4 <= g <= 7]
        if len(free) >= n:
            return ",".join(map(str, sorted(free[:n])))
        if time.time() - t0 > wait_s:
            raise RuntimeError(f"no {n} idle GPUs after {wait_s}s")
        print(f"[tune] waiting for {n} idle GPUs (busy: {sorted(busy)})", flush=True)
        time.sleep(30)


def _valid(eng, cfg: LaunchCfg) -> bool:
    return (
        1 <= cfg.mt <= 6
        and (eng.H // 128) % cfg.nsk == 0
        and eng._lds(cfg.mt, cfg.dyn, cfg.nsk) <= LDS_LIMIT
    )


def schedules(eng, m: int) -> list[LaunchCfg]:
    """Stage 1: one variant per schedule (static with / without column split,
    dynamic), at the heuristic tile / ring / route format."""
    tot = m * eng.tp
    base = eng.default_config(m)
    out = []
    for dyn in (False, True):
        if dyn and not (tot <= eng.dyn_max and eng.I // 128 >= 2 and tot * eng.K <= 4 * eng.E):
            continue
        rpe = (tot * eng.K + eng.E - 1) // eng.E
        mt = eng._fit_mt((rpe + 15) // 16, dyn, 4)
        xss = [True] if dyn or eng._scheds[True].xsplit == eng._scheds[False].xsplit else [True, False]
        for xs in xss:
            cfg = LaunchCfg(mt=mt, nsk=base.nsk if dyn == base.dyn else 4, dyn=dyn,
                            route_fp8=not dyn, xsplit=xs, ll=base.ll)
            if _valid(eng, cfg):
                out.append(cfg)
    return out


def refinements(eng, best: LaunchCfg) -> list[LaunchCfg]:
    """Stage 2: the best schedule with the neighbouring row tiles, the other
    route-row format and (dynamic) the other ring depths."""
    out = []
    for mt in (best.mt - 1, best.mt + 1):
        out.append(LaunchCfg(**{**best.__dict__, "mt": mt}))
    out.append(LaunchCfg(**{**best.__dict__, "route_fp8": not best.route_fp8}))
    out.append(LaunchCfg(**{**best.__dict__, "ll": not best.ll}))
    if best.dyn:
        for nsk in (4, 6, 8):
            if nsk != best.nsk:
                out.append(LaunchCfg(**{**best.__dict__, "nsk": nsk}))
    return [c for c in out if _valid(eng, c)]


def merge_csv(path: str, rows: list[dict]) -> None:
    old = []
    if os.path.exists(path):
        with open(path) as f:
            old = list(csv.DictReader(f))
    key = lambda r: tuple(str(r[k]) for k in CSV_KEY) + (str(r["token"]),)
    new_keys = {key(r) for r in rows}
    keep = [r for r in old if key(r) not in new_keys]
    allr = keep + rows
    allr.sort(key=lambda r: (r["model"], r["comm_mode"], int(r["tp"]), int(r["token"])))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for r in allr:
            w.writerow({c: r.get(c, "") for c in CSV_COLS})


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["glm5", "m3"])
    p.add_argument("--tokens", type=int, nargs="+",
                   default=[8, 16, 32, 64, 96, 128, 256, 512, 1024, 2048])
    p.add_argument("--comm-modes", nargs="+", default=["ag_rs", "ar_ar"])
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--rtol", type=float, default=0.06, help="fused vs split rel L2 gate")
    p.add_argument("--out", default=tuned_config_path())
    p.add_argument("--no-split", action="store_true", help="skip timing the split path")
    p.add_argument("--max-tokens", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--cell", nargs=3, metavar=("MODEL", "MODE", "TOKENS"),
                   help="(internal) tune one cell in this process, print its row as JSON")
    a = p.parse_args()
    if a.cell is None:
        return drive(a)

    if "HIP_VISIBLE_DEVICES" not in os.environ:
        os.environ["HIP_VISIBLE_DEVICES"] = idle_gpus(a.tp)
    a.models, a.comm_modes, a.tokens = [a.cell[0]], [a.cell[1]], [int(a.cell[2])]
    os.environ["AITER_MEGAMOE_TUNED"] = "0"  # candidates are set explicitly
    T._flydsl_multi_device()
    from p2p_collectives import P2PGroup

    from aiter.ops.flydsl.kernels.mega_moe_tp.symmetric_arena import PeerArenaGroup

    tp = a.tp
    devices = [torch.device("cuda", i) for i in range(tp)]
    torch.cuda.set_device(devices[0])
    group = PeerArenaGroup(devices)
    ctxs = [T.DistCtx(rank=r, world=tp, device=d) for r, d in enumerate(devices)]
    targs = argparse.Namespace(sp_warmup=2, rounds=a.rounds, iters=a.iters)
    tokens = sorted(set(a.tokens))
    maxtok = max(max(tokens), a.max_tokens)
    p2p = P2PGroup(devices, maxtok // tp)
    for name in a.models:
        shape = T.MODELS[name]
        weights = []
        for c in ctxs:
            with torch.cuda.device(c.device):
                weights.append(T.build_sharded_weights(shape, c, tp, 0))
        for mode, tok in itertools.product(a.comm_modes, tokens):
            m = tok // tp
            rows = []
            inputs = [T.make_inputs(shape, c, tp, tok, 0, "balanced", mode) for c in ctxs]
            split_cls = T.SplitTpMoeAR if mode == "ar_ar" else T.SplitTpMoe
            split = []
            for r, c in enumerate(ctxs):
                with torch.cuda.device(c.device):
                    split.append(split_cls(weights[r], c, maxtok // tp, comm=p2p.comm(r)))
            y = [t.clone() for t in T._sp_run(devices, lambda r: split[r](inputs[r]))]
            split_us = float("nan")
            if not a.no_split:
                split_us, _ = T._sp_time_graph(devices, lambda r: split[r](inputs[r]), targs)
            fused = []
            for r, c in enumerate(ctxs):
                with torch.cuda.device(c.device):
                    fused.append(T.MegaMoeTP(weights[r], c, m, group=group, comm_mode=mode))
            engs = [f.engine.engine for f in fused]
            best = None
            tried: set = set()

            def run(cfg):
                nonlocal best
                if cfg in tried:
                    return True
                tried.add(cfg)
                for e in engs:
                    e._cfgs[m] = cfg
                try:
                    yf = [t.clone() for t in T._sp_run(devices, lambda r: fused[r](inputs[r]))]
                    errs = [e.poll_errors() for e in engs]
                    err = T._sp_rel_l2(yf, y)
                    if any(errs) or not err < a.rtol:
                        print(f"[tune] {name} {mode} M={tok} {cfg}: rejected errs={errs} rel_l2={err:.4f}", flush=True)
                        # a fired watchdog leaves the arena out of step
                        return not any(errs)
                    us, rep = T._sp_time_graph(devices, lambda r: fused[r](inputs[r]), targs)
                    # graph replays must reproduce the eager output (a race
                    # would show up here first)
                    rerr = T._sp_rel_l2(rep, yf) if rep is not None else float("nan")
                    if not rerr < 1e-3:
                        print(f"[tune] {name} {mode} M={tok} {cfg}: rejected replay rel_l2={rerr:.4f}", flush=True)
                        return True
                except Exception as exc:  # noqa: BLE001 - a variant that fails is skipped
                    print(f"[tune] {name} {mode} M={tok} {cfg}: failed {exc}", flush=True)
                    return True
                print(f"[tune] {name} {mode} M={tok} block_m={cfg.block_m} nsk={cfg.nsk} dyn={int(cfg.dyn)} "
                      f"rf8={int(cfg.route_fp8)} xs={int(cfg.xsplit)} ll={int(cfg.ll)}: {us:.1f} us (rel_l2 {err:.4f})", flush=True)
                # earlier candidates win ties within 1%
                if best is None or us < best[0] * 0.99:
                    best = (us, cfg)
                return True

            ok = all(run(c) for c in schedules(engs[0], m))
            if ok and best is not None:
                for c in refinements(engs[0], best[1]):
                    if not run(c):
                        break
            if best is None:
                print(f"[tune] {name} {mode} M={tok}: no valid config", flush=True)
                continue
            us, cfg = best
            e = engs[0]
            row = dict(
                gfx=e.gfx, cu_num=e.n_cta, tp=tp, comm_mode=mode, token=tok,
                model_dim=e.H, inter_dim=e.I, expert=e.E, topk=e.K, act=e.activation,
                block_m=cfg.block_m, nsk=cfg.nsk, dyn=int(cfg.dyn),
                route_fp8=int(cfg.route_fp8), xsplit=int(cfg.xsplit), ll=int(cfg.ll),
                us=f"{us:.2f}", split_us=f"{split_us:.2f}",
                speedup=f"{split_us / us:.3f}" if split_us == split_us else "",
                model=name,
            )
            rows.append(row)
            print(f"[tune] BEST {name} {mode} M={tok}: {row}", flush=True)
            print("TUNE_ROW " + json.dumps(row), flush=True)
            del fused, engs, split
            for d in devices:
                with torch.cuda.device(d):
                    torch.cuda.empty_cache()
        del weights
    return 0


def drive(a) -> int:
    """Tune every cell in its own process (a variant that faults the GPU
    takes only its cell down), merging each best row into the CSV."""
    if "HIP_VISIBLE_DEVICES" not in os.environ:
        os.environ["HIP_VISIBLE_DEVICES"] = idle_gpus(a.tp)
    print(f"[tune] GPUs {os.environ['HIP_VISIBLE_DEVICES']} -> {a.out}", flush=True)
    for name, mode, tok in itertools.product(a.models, a.comm_modes, sorted(set(a.tokens))):
        cmd = [sys.executable, os.path.abspath(__file__), "--cell", name, mode, str(tok),
               "--tp", str(a.tp), "--iters", str(a.iters), "--rounds", str(a.rounds),
               "--rtol", str(a.rtol), "--out", a.out] + (["--no-split"] if a.no_split else [])
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        rows = []
        for line in r.stdout.splitlines():
            if line.startswith("[tune]"):
                print(line, flush=True)
            elif line.startswith("TUNE_ROW "):
                rows.append(json.loads(line[len("TUNE_ROW "):]))
        if r.returncode != 0:
            tail = "\n".join((r.stderr or "").splitlines()[-5:])
            print(f"[tune] {name} {mode} M={tok}: cell exited {r.returncode}\n{tail}", flush=True)
        if rows:
            merge_csv(a.out, rows)
    print(f"[tune] done -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
