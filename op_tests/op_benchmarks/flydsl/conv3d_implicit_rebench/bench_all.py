#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Re-benchmark every row of the two conv3d tuned CSVs on one idle GPU.

Per row:
  * checks that the production entry point (no explicit tile) resolves to the
    row's tuned tile via ``_lookup_tuned_tile``;
  * mismatch -- elements outside rtol=atol=2e-2 against bf16 F.conv3d;
  * torch.profiler windows that replay ``run_perftest``'s protocol exactly:
    min(101, 32 GiB / input bytes) deep copies of (x, w, bias), no warm round
    over the copies, 101 calls. Each window is preceded by one of two
    preconditions, because MI355X is power-limited and the high-power rows run
    10-18% slower once the part has been loaded for a second:
      - burst:     1 s idle, then the window (the regime the CSV was tuned in);
      - sustained: the same conv back to back for 1 s, then the window.
    From each window:
      - e2e_us    -- median over calls of the sum of every device kernel the
        call launched (weight repack on a cold copy, the C=3 channel pad, the
        conv): the quantity the CSV ``us`` column holds;
      - kernel_us -- median of the ``conv3d_implicit_kernel`` launch alone;
      - aux_us    -- mean non-conv device time per call.
    Taking both from one window matters: two separately-timed passes sit in
    different power states and can disagree by more than the aux kernels they
    are meant to separate.
  * REPS windows per precondition; the median rep is kept.

Also measures the achievable peaks on the same card (torch.mm 8192^3 bf16,
device-to-device copy).

Run from the aiter repo root (writes data/meta.json and data/results.jsonl
next to this script unless --out is given):
    HIP_VISIBLE_DEVICES=0 python3 op_tests/op_benchmarks/flydsl/conv3d_implicit_rebench/bench_all.py

HIP_VISIBLE_DEVICES indexes HIP's enumeration, which on this node is not
rocm-smi's: match them by PCI bus id (``rocm-smi --showbus`` against the "pci"
field this script prints) before trusting a rocm-smi idle check. The script
itself refuses to start on a device with more than 2 GiB already in use.
"""

import argparse
import copy
import functools
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "op_tests"))

import numpy as np
import pandas as pd
import test_flydsl_conv_implicit as T
import torch
import torch.nn.functional as F
import torch.profiler as tpf

from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.ops.flydsl import (
    conv_kernels,
    flydsl_conv_implicit,
)

CSVS = {
    "wan21": f"{REPO}/aiter/configs/model_configs/wan21_vae_bf16_tuned_conv3d.csv",
    "qwenimage": f"{REPO}/aiter/configs/model_configs/qwenimage_vae_bf16_tuned_conv3d.csv",
}
KEY_COLS = list(conv_kernels.TUNED_KEY_COLUMNS)
RTOL = ATOL = 2e-2
LAYOUT = "NDHWC"
REPS = 3
NUM_ITERS = 101  # run_perftest default
ROT_BYTES = 32 << 30  # run_perftest: L2_cache_size (4 MiB) * 64 * 128
PRECOND_S = 1.0


def precondition(mode, fn):
    torch.cuda.synchronize()
    if mode == "burst":
        time.sleep(PRECOND_S)
        return
    t0 = time.time()
    while time.time() - t0 < PRECOND_S:
        for _ in range(10):
            fn()
        torch.cuda.synchronize()


def _key(n, c, d, h, w, k, kt, kh, kw, s, p):
    s = s if isinstance(s, tuple) else (1, s, s) if kt == 1 else (s, s, s)
    p = p if isinstance(p, tuple) else (0, p, p) if kt == 1 else (p, p, p)
    return (n, c, d, h, w, k, kt, kh, kw, *s, *p, 1, 1, 1, 1, True)


def case_index():
    """20-col key -> list of (model, res, case, calls) from the op test's generators."""
    idx = {}

    def add(key, meta):
        idx.setdefault(key, []).append(meta)

    for hh, ww in ((480, 832), (368, 544)):
        res = f"{hh}x{ww}"
        for case, xs, ws, calls in T.wan_vae_conv3d(hh, ww, 81):
            add(_key(*xs, ws[0], *ws[2:], 1, 0), ("wan21", res, case, calls))
        for case, _b, xs, ws, st, pad, calls in T.wan_vae_aux(
            hh, ww, 81
        ) + T.wan_vae_decode(hh, ww, 81):
            if len(xs) == 4:
                n, c, h, w = xs
                key = _key(n, c, 1, h, w, ws[0], 1, *ws[2:], st, pad)
            else:
                st = st if isinstance(st, tuple) else (st,) * 3
                key = _key(*xs, ws[0], *ws[2:], st, pad)
            add(key, ("wan21", res, case, calls))
    for hh, ww in ((1024, 1024), (1328, 1328)):
        res = f"{hh}x{ww}"
        for case, xs, ws, st, pad, calls in T.qwen_vae_conv2d(hh, ww):
            n, c, h, w = xs
            add(
                _key(n, c, 1, h, w, ws[0], 1, *ws[2:], st, pad),
                ("qwenimage", res, case, calls),
            )
    return idx


def gemm_dims(kv):
    ext = lambda i, p, dl, k, s: (i + 2 * p - dl * (k - 1) - 1) // s + 1
    do = ext(kv["D"], kv["pad_d"], kv["dil_d"], kv["kT"], kv["stride_d"])
    ho = ext(kv["H"], kv["pad_h"], kv["dil_h"], kv["kH"], kv["stride_h"])
    wo = ext(kv["W"], kv["pad_w"], kv["dil_w"], kv["kW"], kv["stride_w"])
    g = kv["groups"]
    m = kv["N"] * do * ho * wo
    n = kv["K"] // g
    k = (kv["C"] // g) * kv["kT"] * kv["kH"] * kv["kW"]
    x_b = kv["N"] * kv["C"] * kv["D"] * kv["H"] * kv["W"] * 2
    w_b = kv["K"] * (kv["C"] // g) * kv["kT"] * kv["kH"] * kv["kW"] * 2
    y_b = kv["N"] * kv["K"] * do * ho * wo * 2
    return m, n, k, x_b + w_b + y_b


def make_data(kv, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(
        (kv["N"], kv["D"], kv["H"], kv["W"], kv["C"]),
        device="cuda",
        dtype=torch.bfloat16,
    )
    w = torch.randn(
        (kv["K"], kv["C"] // kv["groups"], kv["kT"], kv["kH"], kv["kW"]),
        device="cuda",
        dtype=torch.bfloat16,
    )
    b = torch.randn((kv["K"],), device="cuda", dtype=torch.float32)
    return x, w, b


def params(kv):
    return {
        "stride": (kv["stride_d"], kv["stride_h"], kv["stride_w"]),
        "padding": (kv["pad_d"], kv["pad_h"], kv["pad_w"]),
        "dilation": (kv["dil_d"], kv["dil_h"], kv["dil_w"]),
        "groups": kv["groups"],
    }


def call(x, w, b, p):
    return flydsl_conv_implicit(
        x, w, bias=b, input_layout=LAYOUT, output_layout=LAYOUT, **p
    )


def ramp_clocks(seconds=2.0):
    a = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    deadline = time.time() + seconds
    while time.time() < deadline:
        for _ in range(20):
            a = torch.mm(a, a).clamp_(-1.0, 1.0)
        torch.cuda.synchronize()
    del a
    torch.cuda.empty_cache()


def timed_window(x, w, b, p, warm_cache=False, precond=None, attempts=3):
    """``_timed_window`` retried: the profiler occasionally delivers no kernel events."""
    for i in range(attempts):
        try:
            return _timed_window(x, w, b, p, warm_cache, precond)
        except _NoEvents:
            print(
                f"  profiler returned no conv events, retry {i + 1}/{attempts}",
                flush=True,
            )
    raise RuntimeError("profiler returned no conv events")


class _NoEvents(Exception):
    pass


def _timed_window(x, w, b, p, warm_cache=False, precond=None):
    """One profiled window of NUM_ITERS calls -> (e2e median, conv median, aux mean, info).

    warm_cache=False replays run_perftest: only the original args are warmed,
    so every other copy repacks its weight right before its conv, and the conv
    then reads a packed weight that is still in L2/MALL.
    warm_cache=True runs one round over every copy first, so each call hits
    the weight cache and the conv reads its packed weight cold -- what a VAE
    forward does, where the weights never change.
    """
    in_bytes = x.nbytes + w.nbytes + b.nbytes
    nrot = int(min(NUM_ITERS, max(1, math.ceil(ROT_BYTES / in_bytes))))
    sets = [copy.deepcopy((x, w, b)) for _ in range(nrot - 1)] + [(x, w, b)]
    call(x, w, b, p)  # run_perftest's warmup runs on the original args only
    call(x, w, b, p)
    if warm_cache:
        for s in sets:
            call(*s, p)
    if precond is not None:
        precond()
    torch.cuda.synchronize()
    with tpf.profile(activities=[tpf.ProfilerActivity.CUDA]) as prof:
        for i in range(NUM_ITERS):
            xs, ws, bs = sets[i % nrot]
            call(xs, ws, bs, p)
        torch.cuda.synchronize()
    ev = sorted(
        (e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA),
        key=lambda e: e.time_range.start,
    )
    # the conv launch is the last kernel of every call (splitK=1 on every row)
    per_call, conv, cur, aux, aux_names = [], [], 0.0, 0.0, {}
    for e in ev:
        cur += e.device_time
        if "conv3d_implicit" in e.name:
            conv.append(e.device_time)
            per_call.append(cur)
            cur = 0.0
        else:
            aux += e.device_time
            short = e.name.split("<")[0][:60]
            aux_names[short] = aux_names.get(short, 0) + 1
    del sets
    torch.cuda.empty_cache()
    if len(conv) < NUM_ITERS // 2:
        raise _NoEvents()
    info = {
        "nrot": nrot,
        "conv_launches": len(conv),
        "aux_kernels": aux_names,
        "conv_p10": float(np.percentile(conv, 10)),
        "conv_p90": float(np.percentile(conv, 90)),
    }
    return float(np.median(per_call)), float(np.median(conv)), aux / NUM_ITERS, info


def _event_time(fn, n):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / n * 1e-3


def peaks():
    """Achievable matrix / copy throughput under both preconditions (median of 3)."""
    out = {}
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    bm = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    src = torch.empty(4 << 30, device="cuda", dtype=torch.uint8)
    dst = torch.empty_like(src)
    mm = functools.partial(torch.mm, a, bm)
    cp = functools.partial(dst.copy_, src)
    for _ in range(3):
        mm()
        cp()
    for mode in ("burst", "sustained"):
        sfx = "" if mode == "burst" else "_sus"
        ts = []
        for _ in range(3):
            precondition(mode, mm)
            ts.append(_event_time(mm, 20))
        out[f"mm_tflops{sfx}"] = 2 * 8192**3 / float(np.median(ts)) / 1e12
        ts = []
        for _ in range(3):
            precondition(mode, cp)
            ts.append(_event_time(cp, 10))
        out[f"copy_gbs{sfx}"] = 2 * src.nbytes / float(np.median(ts)) / 1e9
    del a, bm, src, dst
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "data"))
    ap.add_argument("--limit", type=int, default=0, help="debug: first N rows only")
    ap.add_argument("--only", nargs="*", default=[], help="case@HxW names to rerun")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dev = (get_gfx(), get_cu_num())
    prop = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    meta = {
        "gfx": dev[0],
        "cu_num": dev[1],
        "device": torch.cuda.get_device_name(),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        # rocm-smi's GPU index is not HIP's on this node; the bus id is the link
        "pci": f"{prop.pci_domain_id:04x}:{prop.pci_bus_id:02x}:{prop.pci_device_id:02x}",
        "used_gib_at_start": (total - free) / 2**30,
        "torch": torch.__version__,
    }
    print(json.dumps(meta), flush=True)
    if meta["used_gib_at_start"] > 2.0:
        raise SystemExit(
            f"device {meta['pci']} already has {meta['used_gib_at_start']:.1f} GiB in use"
        )
    meta.update(peaks())
    print(json.dumps(meta), flush=True)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    idx = case_index()
    rows = []
    for model, path in CSVS.items():
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        for _, r in df.iterrows():
            rows.append((model, r))
    if args.only:
        rows = [
            (m_, r)
            for m_, r in rows
            if any(
                f"{c[2]}@{c[1]}" in args.only
                for c in idx.get(
                    tuple(int(r[k]) if k != "bias" else True for k in KEY_COLS), []
                )
            )
        ]
    if args.limit:
        rows = rows[: args.limit]

    out_path = os.path.join(args.out, "results.jsonl")
    with open(out_path, "w") as fo:
        for i, (model, r) in enumerate(rows):
            kv = {
                c: (conv_kernels._parse_tuned_bool(r[c]) if c == "bias" else int(r[c]))
                for c in KEY_COLS
            }
            key = tuple(kv[c] for c in KEY_COLS)
            names = idx.get(key, [])
            m, n, k, moved = gemm_dims(kv)
            csv_tile = (int(r.tile_m), int(r.tile_n), int(r.wave_m), int(r.wave_n))
            hit = conv_kernels._lookup_tuned_tile(key, None)
            rec = {
                "model": model,
                "cases": [list(x) for x in names],
                "key": list(key),
                "M": m,
                "N": n,
                "K": k,
                "bytes": moved,
                "tile": "x".join(map(str, csv_tile[:2]))
                + "/"
                + "x".join(map(str, csv_tile[2:])),
                "wgm": int(r.wgm),
                "splitK": int(r.splitK),
                "csv_us": float(r.us),
                "lookup_ok": hit is not None
                and tuple(hit[0]) == csv_tile
                and int(hit[1]) == int(r.wgm),
            }
            try:
                x, w, b = make_data(kv)
                p = params(kv)
                y = call(x, w, b, p).float()
                ref = F.conv3d(
                    x.permute(0, 4, 1, 2, 3).contiguous(), w, bias=b.to(x.dtype), **p
                ).permute(0, 2, 3, 4, 1)
                bad = int((~torch.isclose(y, ref.float(), rtol=RTOL, atol=ATOL)).sum())
                del y, ref
                torch.cuda.empty_cache()
                rec.update(mismatch=bad, numel=int(m * n), status="ok")
                fn = functools.partial(call, x, w, b, p)
                # (label, weight cache warm, precondition):
                #   csv  -- the CSV's protocol, e2e comparable to `us`
                #   ""   -- kernel-only, cache warm, settled-idle clocks
                #   _sus -- kernel-only after 1 s of this conv
                plans = (
                    ("csv", False, "burst"),
                    ("", True, "burst"),
                    ("_sus", True, "sustained"),
                )
                for sfx, warm, mode in plans:
                    reps = []
                    for _ in range(REPS):
                        reps.append(
                            timed_window(
                                x,
                                w,
                                b,
                                p,
                                warm_cache=warm,
                                precond=functools.partial(precondition, mode, fn),
                            )
                        )
                    reps.sort(key=lambda t: t[1])
                    e2e, kus, aux, info = reps[len(reps) // 2]
                    if sfx == "csv":
                        rec.update(
                            e2e_us=e2e,
                            kernel_us_csvproto=kus,
                            aux_us_csvproto=aux,
                            rep_e2e_us=sorted(t[0] for t in reps),
                            nrot=info["nrot"],
                        )
                        continue
                    rec.update(
                        {
                            f"kernel_us{sfx}": kus,
                            f"aux_us{sfx}": aux,
                            f"rep_kernel_us{sfx}": [t[1] for t in reps],
                            f"conv_p10{sfx}": info["conv_p10"],
                            f"conv_p90{sfx}": info["conv_p90"],
                        }
                    )
                    if sfx == "":
                        rec.update(
                            conv_launches=info["conv_launches"],
                            aux_kernels=info["aux_kernels"],
                        )
                del x, w, b
                torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001
                rec.update(status=f"error: {exc!r}")
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            nm = ",".join(f"{a[2]}@{a[1]}" for a in names) or "?"
            print(
                f"[{i+1}/{len(rows)}] {nm:40s} csv={rec['csv_us']:.1f} "
                f"e2e={rec.get('e2e_us', -1):.1f} kern={rec.get('kernel_us', -1):.1f} "
                f"kern_sus={rec.get('kernel_us_sus', -1):.1f} "
                f"aux={rec.get('aux_us', -1):.1f} bad={rec.get('mismatch')} "
                f"lookup={rec['lookup_ok']} {rec['status'][:60]}",
                flush=True,
            )


if __name__ == "__main__":
    main()
