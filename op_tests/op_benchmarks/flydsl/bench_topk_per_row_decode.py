# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Decode top-k: every kernel `top_k_per_row_decode` can run, on seeded inputs.

Arms are the adaptive kernel (`adaptive`), the single-launch kernel (`single`),
the FlyDSL one-block kernel (`oneblock`) and the C++ one-block kernel (`hip`).
`main` is not timed: each record names the arm main's dispatch picks for the
cell, so its time is that arm's. Every arm is checked against `torch.topk`
after the eager call and after each graph replay.

Two timers, both on CUDA graphs:
  t1  16 calls per graph, 30 replays, GPU time per call, median of 3
  t2  1 call per graph, replay + synchronize, host wall time, median of 31
Each cell runs on every seed in --seeds; decide on the median over seeds.

Slices (rows 1..512, k 256..4096, stable 0/1 unless noted):
  A1  every row as wide as the buffer, 10 widths
  A2  buffer 128K/256K/1M, every row 4K, 16K, 48K, W/8, W/4 or W/2 long
  A3  buffer 128K/256K/1M, 25/50/80% of rows full, the rest 4K
  A4  buffer 128K/256K/1M, each row uniform in [4K, W]
  B   the cases #6329 published, k=2048

  python op_tests/op_benchmarks/flydsl/bench_topk_per_row_decode.py --list
  python op_tests/op_benchmarks/flydsl/bench_topk_per_row_decode.py \\
      --slice A1 --rows 4 --k 2048 --out a1.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import statistics
import sys
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

HARNESS_VERSION = 2
ROWS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
KS = (256, 512, 1024, 2048, 4096)
DIAG_WIDTHS = (4096, 8192, 16384, 20000, 32768, 65536, 131072, 262144, 524288, 1048576)
PADDED_WIDTHS = (131072, 262144, 1048576)
SHORT = 4096
# Past every live length, so a kernel that reads padding selects it and fails.
POISON = 1.0e30
ARMS = ("adaptive", "single", "oneblock", "hip")

_REPO = Path(__file__).resolve().parents[3]
_KERNELS = "aiter/ops/flydsl/kernels/topk/"
_HOST = ("aiter/ops/topk.py", "aiter/ops/flydsl/topk/topk_per_row.py")
_ARM_SOURCES = {
    "adaptive": (_KERNELS + "topk_per_row_decode_adaptive.py",),
    "single": (
        _KERNELS + "topk_per_row_decode.py",
        _KERNELS + "topk_per_row_decode_persistent.py",
        _KERNELS + "topk_common.py",
    ),
    "oneblock": (_KERNELS + "radix_topk_one_block.py", _KERNELS + "topk_common.py"),
    "hip": ("csrc/kernels/topk_per_row_kernels.cu",),
}


@dataclass(frozen=True)
class Cell:
    slice: str
    width: int
    rows: int
    k: int
    stable: int
    # "full", "u<len>", "m<pct>", "spread", or "b<long>x<len>+<short>x<len>"
    profile: str


def _mixed(rows: int, width: int, pct: int) -> str:
    n_long = (rows * pct + 50) // 100
    return f"b{n_long}x{width}+{rows - n_long}x{SHORT}"


_B_UNSTABLE = (
    (1, 1048576, "u512"),
    (1, 1048576, "u16384"),
    (2, 1048576, "u20480"),
    (16, 1048576, "u4096"),
    (32, 1048576, "u49152"),
    (20, 524288, "b16x524288+4x4096"),
    (16, 524288, "b8x524288+8x4096"),
    (16, 1048576, "b8x1048576+8x4096"),
    (16, 1048576, "b4x1048576+12x4096"),
    (20, 524288, "b4x524288+16x4096"),
    (1, 131072, "full"),
    (4, 131072, "full"),
    (32, 131072, "full"),
    (4, 262144, "full"),
    (16, 262144, "full"),
    (16, 524288, "full"),
    (32, 524288, "full"),
    (16, 1048576, "full"),
)


def _profiles(s: str, w: int, r: int) -> list[str]:
    if s == "A1":
        return ["full"]
    if s == "A2":
        ps = [f"u{n}" for n in (4096, 16384, 49152, w // 8, w // 4, w // 2)]
    elif s == "A3":
        ps = [_mixed(r, w, p) for p in (25, 50, 80)]
    elif s == "A4":
        return ["spread"]
    else:
        raise ValueError(f"unknown slice {s!r}")
    # W/8 can equal a fixed length, and few rows round two shares to one count.
    return list(dict.fromkeys(ps))


def cells(slices, rows=ROWS, ks=KS, stables=(0, 1), widths=None):
    """Every cell of `slices`, filtered by the other arguments."""
    for s in slices:
        if s == "B":
            for stable in stables:
                for r, w, p in _B_UNSTABLE:
                    if stable and p == "u512":
                        continue  # #6329's stable table drops it
                    if (widths is None or w in widths) and 2048 in ks:
                        yield Cell("B", w, r, 2048, stable, p)
            continue
        ws = DIAG_WIDTHS if s == "A1" else PADDED_WIDTHS
        for w in ws if widths is None else [w for w in ws if w in widths]:
            for r in rows:
                for p in _profiles(s, w, r):
                    for k in ks:
                        for stable in stables:
                            if k <= w:
                                yield Cell(s, w, r, k, stable, p)


def _seed(*key) -> int:
    return zlib.crc32(repr(key).encode())


def live_lengths(cell: Cell, seed: int) -> list[int]:
    p, w, r = cell.profile, cell.width, cell.rows
    if p == "full":
        return [w] * r
    if p.startswith("u"):
        return [int(p[1:])] * r
    if p == "spread":
        g = torch.Generator().manual_seed(_seed("spread", w, r, seed))
        return torch.randint(SHORT, w + 1, (r,), generator=g).tolist()
    if p.startswith("b"):
        lens = []
        for part in p[1:].split("+"):
            n, length = part.split("x")
            lens += [int(length)] * int(n)
        # Long rows at seeded positions, not all first.
        g = torch.Generator().manual_seed(_seed("order", w, r, p, seed))
        return [lens[i] for i in torch.randperm(r, generator=g).tolist()]
    raise ValueError(f"unknown profile {p!r}")


def make_inputs(cell: Cell, seed: int):
    lens = live_lengths(cell, seed)
    g = torch.Generator(device="cuda").manual_seed(
        _seed("x", cell.width, cell.rows, cell.profile, seed)
    )
    x = torch.empty(cell.rows, cell.width, device="cuda", dtype=torch.float32)
    x.normal_(generator=g)
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    cols = torch.arange(cell.width, device="cuda")
    x.masked_fill_(cols[None, :] >= seq_lens[:, None], POISON)
    return x, seq_lens


def check(x, seq_lens, out, k) -> bool:
    """Same value multiset as `torch.topk` over each row's live length, -1 past it."""
    lens = seq_lens.long()
    idx = out.long()
    valid = idx >= 0
    if not torch.equal(valid.sum(1), torch.clamp(lens, max=k)):
        return False
    if bool((idx >= lens[:, None]).any()) or bool((idx < -1).any()):
        return False
    s = idx.sort(1).values
    if bool(((s[:, 1:] == s[:, :-1]) & (s[:, 1:] >= 0)).any()):
        return False
    cols = torch.arange(x.shape[1], device=x.device)
    masked = x.masked_fill(cols[None, :] >= lens[:, None], float("-inf"))
    want = torch.topk(masked, k, dim=1).values
    got = x.gather(1, idx.clamp(min=0)).masked_fill(~valid, float("-inf"))
    return torch.equal(torch.sort(want, dim=1).values, torch.sort(got, dim=1).values)


def timer1(call, copies=16, iters=30, inner=3) -> list[float]:
    runs = []
    for _ in range(inner):
        for _ in range(20):
            call()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(copies):
                call()
        for _ in range(10):
            g.replay()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            g.replay()
        e.record()
        torch.cuda.synchronize()
        runs.append(s.elapsed_time(e) * 1000.0 / (iters * copies))
        del g
    return runs


def timer2(call, reps=31, warmup=3) -> list[float]:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        call()
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    runs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        runs.append((time.perf_counter() - t0) * 1e6)
    del g
    return runs


class Arms:
    """One callable per arm, plus the arm main's dispatch would pick."""

    def __init__(self):
        from aiter.jit.utils.chip_info import get_gfx_runtime
        from aiter.ops import topk
        from aiter.ops.flydsl.kernels.kernels_common import get_warp_size
        from aiter.ops.flydsl.topk import topk_per_row as host

        self.topk, self.host = topk, host
        params = inspect.signature(host._decode_with_backend).parameters
        self._takes_bound = "max_row_len" in params
        # The adaptive arm is forced past the gate, which never picks it on a
        # wave32 card; its scan is laid out for wave64. A card without bands
        # still runs it, since the bench is what fits them.
        self._wave64 = get_warp_size(get_gfx_runtime()) == 64

    def _backend(self, x, sl, out, cell, backend, bound=None):
        extra = (bound,) if self._takes_bound else ()
        self.host._decode_with_backend(
            x, 1, sl, out, cell.rows, x.stride(0), x.stride(1), cell.k,
            bool(cell.stable), None, backend, *extra,
        )  # fmt: skip

    def supported(self, arm, x, sl, out, cell) -> bool:
        a = (x, 1, sl, out, cell.rows, x.stride(0), x.stride(1), cell.k)
        if arm == "adaptive" and not self._wave64:
            return False
        if arm in ("adaptive", "single"):
            return self.host.is_flydsl_top_k_per_row_decode_supported(*a, None)
        if arm == "oneblock":
            # Whether the kernel launches here, not whether main routes to it:
            # main's allow-list leaves gfx942 out, and main_pick keeps it.
            sig = self.host._tensor_signature
            return self.host._is_flydsl_radix_topk_one_block_call_supported(
                sig(x), None, sig(sl), sig(out), 1, cell.rows, x.stride(0),
                x.stride(1), cell.k, None, True,
                self.host._FLYDSL_TOPK_ONE_BLOCK_ARCHES[0],
            )  # fmt: skip
        return True

    def call(self, arm, x, sl, out, cell):
        topk, s = self.topk, bool(cell.stable)
        if arm == "adaptive":
            return lambda: self._backend(
                x, sl, out, cell, topk.BACKEND_ADAPTIVE, cell.width
            )
        if arm == "single":
            return lambda: self._backend(x, sl, out, cell, topk.BACKEND_SINGLE)
        if arm == "oneblock":
            return lambda: topk.flydsl_radix_topk_one_block_decode(
                x, 1, sl, out, cell.rows, x.stride(0), x.stride(1), cell.k, s
            )
        return lambda: topk.top_k_per_row_decode(
            x, 1, sl, out, cell.rows, x.stride(0), x.stride(1), cell.k, s
        )

    def main_pick(self, x, sl, out, cell) -> str:
        topk = self.topk
        backend = topk.decode_backend_for_call(
            x, 1, sl, out, cell.rows, x.stride(0), x.stride(1), cell.k,
            bool(cell.stable), None,
        )  # fmt: skip
        if backend == topk.BACKEND_SINGLE:
            return "single"
        if backend != topk.BACKEND_DEFAULT:
            raise RuntimeError(f"main's dispatch picked {backend!r}")
        if self.host._is_flydsl_radix_topk_one_block_supported(
            x, None, sl, out, None, cell.rows, x.stride(0), x.stride(1),
            cell.k, is_decode=True, next_n=1,
        ):  # fmt: skip
            return "oneblock"
        return "hip"


def environment() -> dict:
    import flydsl

    props = torch.cuda.get_device_properties(0)
    src = {}
    for arm, files in _ARM_SOURCES.items():
        h = hashlib.sha256()
        for f in (*files, *_HOST):
            h.update((_REPO / f).read_bytes())
        src[arm] = h.hexdigest()[:12]
    return {
        "harness": HARNESS_VERSION,
        "device": props.name,
        "arch": props.gcnArchName.split(":")[0],
        "cu": props.multi_processor_count,
        "torch": torch.__version__,
        "rocm": torch.version.hip,
        "flydsl": flydsl.__version__,
        "src": src,
    }


def run_cell(arms: Arms, cell: Cell, arm_names, seed=0, rotate=0, timers=("t1", "t2")):
    """One record per arm, arms in an order rotated by `rotate`."""
    arms.host.clear_topk_per_row_decode_workspace_cache()
    x, sl = make_inputs(cell, seed)
    probe = torch.empty(cell.rows, cell.k, dtype=torch.int32, device="cuda")
    pick = arms.main_pick(x, sl, probe, cell)
    order = (
        list(arm_names)[rotate % len(arm_names) :]
        + list(arm_names)[: rotate % len(arm_names)]
    )
    recs = []
    for arm in order:
        rec = {**asdict(cell), "seed": seed, "arm": arm, "main_pick": pick}
        out = torch.full((cell.rows, cell.k), -2, dtype=torch.int32, device="cuda")
        if not arms.supported(arm, x, sl, out, cell):
            recs.append({**rec, "supported": False})
            continue
        call = arms.call(arm, x, sl, out, cell)
        saved = arms.topk._FLYDSL_TOPK_DECODE_DISABLED
        arms.topk._FLYDSL_TOPK_DECODE_DISABLED = arm == "hip"
        t0 = time.perf_counter()
        try:
            call()
            torch.cuda.synchronize()
            ok = check(x, sl, out, cell.k)
            if "t1" in timers:
                out.fill_(-2)
                runs = timer1(call)
                ok = ok and check(x, sl, out, cell.k)
                rec.update(
                    t1=round(statistics.median(runs), 3),
                    t1_runs=[round(v, 3) for v in runs],
                )
            if "t2" in timers:
                out.fill_(-2)
                runs = timer2(call)
                ok = ok and check(x, sl, out, cell.k)
                q = statistics.quantiles(runs, n=10)
                rec.update(
                    t2=round(statistics.median(runs), 3),
                    t2_p10=round(q[0], 3),
                    t2_p90=round(q[-1], 3),
                )
            rec.update(supported=True, correct=bool(ok))
        # Recorded, not raised, so one cell cannot end a sweep.
        except Exception as exc:  # noqa: BLE001
            rec.update(
                supported=True,
                correct=False,
                error=f"{type(exc).__name__}: {exc}"[:300],
            )
        finally:
            arms.topk._FLYDSL_TOPK_DECODE_DISABLED = saved
        rec["wall_s"] = round(time.perf_counter() - t0, 3)
        recs.append(rec)
    del x, sl
    return recs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--slice", nargs="+", default=["A1", "A2", "A3", "A4", "B"])
    ap.add_argument("--rows", type=int, nargs="+", default=list(ROWS))
    ap.add_argument("--k", type=int, nargs="+", default=list(KS))
    ap.add_argument("--stable", type=int, nargs="+", default=[0, 1], choices=(0, 1))
    ap.add_argument("--widths", type=int, nargs="+", default=None)
    ap.add_argument("--profiles", nargs="+", default=None, help="e.g. full u16384")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--timers", nargs="+", default=["t1", "t2"], choices=("t1", "t2"))
    ap.add_argument("--rounds", type=int, default=1)
    # main's stable kernels run up to 2x slower on some draws; with 3 seeds two slow
    # draws already move the median.
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default=None, help="jsonl; stdout if omitted")
    ap.add_argument("--list", action="store_true", help="count the cells and exit")
    args = ap.parse_args(argv)

    todo = [
        c
        for c in cells(args.slice, args.rows, args.k, args.stable, args.widths)
        if args.profiles is None or c.profile in args.profiles
    ]
    if args.list:
        by = {}
        for c in todo:
            by[c.slice] = by.get(c.slice, 0) + 1
        print(json.dumps({"cells": len(todo), **by}))
        return

    arms = Arms()
    bad = 0
    with (
        open(args.out, "a") if args.out else contextlib.nullcontext(sys.stdout)
    ) as sink:
        print(json.dumps({"env": environment()}), file=sink, flush=True)
        for rnd in range(args.rounds):
            for i, cell in enumerate(todo):
                for j, seed in enumerate(args.seeds):
                    for rec in run_cell(
                        arms, cell, args.arms, seed, i + j + rnd, args.timers
                    ):
                        rec["round"] = rnd
                        bad += rec["supported"] and not rec["correct"]
                        print(json.dumps(rec), file=sink, flush=True)
    if bad:
        print(f"{bad} wrong or failed results", file=sys.stderr)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
