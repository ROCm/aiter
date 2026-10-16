"""Tuner for the native gfx942 bf16 GEMM and its tuned-config table.

Two files:

* The tuned table (default: the file ``gemm_a16w16`` reads,
  ``aiter/configs/flydsl_gemm_a16w16_gfx942_tuned.csv``) holds one complete
  config per (gfx, cu_num, M, N, K, out_dtype). Shapes without a row fall back
  to the ``gfx942_auto_config`` heuristic.
* The measurement log (``--csv``, default ``gfx942_gemm_tuning.csv``) records
  every attempted config, including compile and correctness failures.
  ``--resume`` reuses its screen timings instead of re-measuring.

Tuning a shape:

1. Screen: time the current config, the heuristic and up to ``--limit``
   candidates once each. Candidates are streamed-operand pingpong configs,
   ranked by the fitted pingpong cost model so a small budget covers the
   likeliest winners first.
2. Refine: re-screen the best few with other ``group_m`` / ``direct_store``.
3. Promote: time the fastest ``--finalists`` plus the incumbent (the table row,
   else the heuristic) over ``--rounds`` interleaved rounds and write the
   winner only if its median beats the incumbent's by ``--margin``. Single
   runs vary by ~4%, so a screen result alone never replaces a row.

Examples:

  # Seed the table with the heuristic configs for a shape list.
  HIP_VISIBLE_DEVICES=6 PYTHONPATH=$PWD flydsl_venv/bin/python \\
    op_tests/flydsl_tests/tune_gfx942_csv.py --seed \\
    --shapes op_tests/flydsl_tests/gfx942_gemm_tuning_shapes.csv

  # Tune one shape (screen 80 candidates), updating the table if a config wins.
  HIP_VISIBLE_DEVICES=6 PYTHONPATH=$PWD flydsl_venv/bin/python \\
    op_tests/flydsl_tests/tune_gfx942_csv.py --M 1024 --N 336 --K 10240 \\
    --limit 80 --resume

  # Benchmark what gemm_a16w16 runs by default (table row, else heuristic).
  HIP_VISIBLE_DEVICES=6 PYTHONPATH=$PWD flydsl_venv/bin/python \\
    op_tests/flydsl_tests/tune_gfx942_csv.py --current-only \\
    --shapes op_tests/flydsl_tests/gfx942_gemm_tuning_shapes.csv --csv run.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

import torch

from aiter.ops.flydsl.kernels.gemm_a16w16_gfx942 import (
    GFX942_NUM_CUS,
    TUNED_BOOL_COLUMNS,
    TUNED_CONFIG_COLUMNS,
    gemm_a16w16,
    gfx942_auto_config,
    make_gemm_a16w16_gfx942_param,
    parse_tuned_config,
    pingpong_fill_cost_us,
    reload_tuned_table,
    tuned_table_path,
)

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "gfx942_gemm_tuning.csv"

CONFIG_COLUMNS = list(TUNED_CONFIG_COLUMNS)
BOOL_COLUMNS = TUNED_BOOL_COLUMNS
CSV_COLUMNS = [
    "timestamp", "host", "device", "gfx", "cu_num", "regime", "M", "N", "K",
    "source", *CONFIG_COLUMNS, "status", "error", "err_ratio", "fly_us",
    "hipb_us", "ratio", "tflops", "measurement",
]
TABLE_COLUMNS = [
    "gfx", "cu_num", "M", "N", "K", "out_dtype", *CONFIG_COLUMNS,
    "us", "hipb_us", "ratio", "source", "timestamp",
]

# Kernel defaults for columns a config leaves out.
CONFIG_DEFAULTS = {
    "block_m": 128, "block_n": 128, "block_k": 64, "stages": 2, "split_k": 1,
    "m_waves": 2, "n_waves": 2, "k_waves": 1, "group_m": 0, "async_load_bytes": 4,
    "buffered_load": False, "direct_store": False, "pingpong": False,
    "split_k_reduce": False, "prefetch_depth": 1, "stream_operands": False,
}
STREAM_PIPELINE = dict(stages=1, buffered_load=True, async_load_bytes=16,
                       pingpong=True, prefetch_depth=1, stream_operands=True)

SCREEN_TILES_M = (32, 64, 128, 256)
SCREEN_TILES_N = (32, 48, 64, 96, 112, 128, 160, 192, 256)
# Earlier layouts win model ties, so a limited screen tries them first.
SCREEN_WAVES = ((2, 2, 1), (2, 1, 2), (1, 2, 2), (1, 4, 1), (4, 1, 1), (1, 1, 4))


def normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(CONFIG_DEFAULTS)
    cfg.update({key: value for key, value in config.items() if key in CONFIG_DEFAULTS})
    if cfg["split_k"] == 1:
        cfg["split_k_reduce"] = False
    return {name: cfg[name] for name in CONFIG_COLUMNS}


def config_key(m: int, n: int, k: int, config: dict[str, Any]) -> tuple[Any, ...]:
    cfg = normalized_config(config)
    return (m, n, k, *(cfg[name] for name in CONFIG_COLUMNS))


def parse_config_row(row: dict[str, str]) -> dict[str, Any]:
    """Config of a measurement-log or table row (blank cells take defaults)."""
    return normalized_config(parse_tuned_config(row))


def describe(cfg: dict[str, Any]) -> str:
    c = normalized_config(cfg)
    pipe = "stream" if c["stream_operands"] else ("pp" if c["pingpong"] else f"st{c['stages']}")
    return (f"{c['block_m']}x{c['block_n']}x{c['block_k']} "
            f"w{c['m_waves']}{c['n_waves']}{c['k_waves']} sk{c['split_k']} "
            f"gm{c['group_m']} {pipe}{' r2g' if c['direct_store'] else ''}")


# ---------------------------------------------------------------- timing


def graph_time_us(fn, iters: int, warmup: int, batch: int) -> float:
    """GPU time per launch with Python submission gaps removed."""
    fn()  # Compile/JIT before capture.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(batch):
            fn()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / (iters * batch)


def rel_err(output: torch.Tensor, reference: torch.Tensor) -> float:
    diff = (output.float() - reference).abs().max().item()
    return diff / (reference.abs().max().item() + 1e-9)


# ---------------------------------------------------------------- files


def read_shapes(path: Path) -> list[tuple[str, int, int, int]]:
    with path.open(newline="") as handle:
        return [
            (row.get("regime", "unspecified"), int(row["M"]), int(row["N"]), int(row["K"]))
            for row in csv.DictReader(handle)
        ]


def migrate_header(path: Path, columns: list[str]) -> None:
    """Rewrite a file whose header predates the current columns."""
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames == columns:
            return
        rows = list(reader)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_row(path: Path, row: dict[str, Any]) -> None:
    write_header = not path.exists() or path.stat().st_size == 0
    if not write_header:
        migrate_header(path, CSV_COLUMNS)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in CSV_COLUMNS})
        handle.flush()
        os.fsync(handle.fileno())


def logged_screen_times(path: Path, m: int, n: int, k: int) -> dict[tuple, float]:
    """Best logged screen time per config key of one shape."""
    times: dict[tuple, float] = {}
    if not path.exists():
        return times
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (int(row["M"]), int(row["N"]), int(row["K"])) != (m, n, k):
                continue
            key = config_key(m, n, k, parse_config_row(row))
            if row["status"] == "ok" and row["fly_us"]:
                times[key] = min(times.get(key, float("inf")), float(row["fly_us"]))
            else:
                times.setdefault(key, float("inf"))
    return times


def table_key(gfx: str, cu_num: int, m: int, n: int, k: int, out_dtype: str) -> tuple:
    return (gfx, int(cu_num), int(m), int(n), int(k), out_dtype)


def read_table(path: Path) -> dict[tuple, dict[str, str]]:
    rows: dict[tuple, dict[str, str]] = {}
    if path.exists():
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                rows[table_key(row["gfx"], row["cu_num"], row["M"], row["N"], row["K"],
                               row["out_dtype"])] = row
    return rows


def write_table(path: Path, rows: dict[tuple, dict[str, Any]]) -> None:
    """Rewrite the table sorted by key (atomic replace), then drop cached copies."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow({name: rows[key].get(name, "") for name in TABLE_COLUMNS})
    os.replace(tmp, path)
    reload_tuned_table()


def table_row(key: tuple, cfg: dict[str, Any], fly_us: float, hipb_us: float,
              source: str) -> dict[str, Any]:
    gfx, cu_num, m, n, k, out_dtype = key
    return {
        "gfx": gfx, "cu_num": cu_num, "M": m, "N": n, "K": k, "out_dtype": out_dtype,
        **normalized_config(cfg),
        "us": f"{fly_us:.4f}", "hipb_us": f"{hipb_us:.4f}",
        "ratio": f"{fly_us / hipb_us:.4f}", "source": source,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def device_metadata() -> tuple[str, str, int]:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return props.name, props.gcnArchName.split(":")[0], props.multi_processor_count


# ---------------------------------------------------------------- candidates


def valid_config(config: dict[str, Any], m: int, n: int, k: int) -> tuple[bool, str]:
    cfg = normalized_config(config)
    if k % cfg["block_k"]:
        return False, "K not divisible by block_k"
    k_tiles = k // cfg["block_k"]
    split_tiles = -(-k_tiles // cfg["split_k"])
    last_split_tiles = k_tiles - (cfg["split_k"] - 1) * split_tiles
    if last_split_tiles < 1:
        return False, "split_k leaves an empty K partition"
    if cfg["direct_store"] and n % cfg["block_n"]:
        return False, "direct_store requires N to be a block_n multiple"
    if cfg["stream_operands"]:
        min_tiles = 2
    else:
        min_tiles = 4 if cfg["prefetch_depth"] == 2 else 3
    if cfg["pingpong"] and last_split_tiles < min_tiles:
        return False, f"pingpong needs at least {min_tiles} K tiles per split"
    try:
        make_gemm_a16w16_gfx942_param(**cfg)
    except Exception as error:  # validation failures are useful tuning evidence
        return False, str(error)
    return True, ""


def model_cost(m: int, n: int, k: int, cfg: dict[str, Any]) -> float:
    """Busiest-CU estimate: rounds of workgroups x the fitted per-round cost."""
    tiles = -(-m // cfg["block_m"]) * -(-n // cfg["block_n"])
    rounds = -(-(tiles * cfg["split_k"]) // GFX942_NUM_CUS)
    return rounds * pingpong_fill_cost_us(
        cfg["block_m"], cfg["block_n"], cfg["k_waves"], cfg["split_k"], max(k // 128, 1)
    )


def grid_configs(m: int, n: int, k: int) -> list[dict[str, Any]]:
    """Valid streamed-operand pingpong configs, cheapest modelled first.

    The model cannot tell block_k or wave layouts apart, so the list takes the
    best variant of every (tile, split_k) family before any second variant.
    """
    block_ks = (128, 64, 256) if k >= 2048 else (64, 128, 256)
    out = []
    for order, (bm, bn, waves, bk, sk) in enumerate(
        (bm, bn, w, bk, sk)
        for bm in SCREEN_TILES_M for bn in SCREEN_TILES_N for w in SCREEN_WAVES
        for bk in block_ks for sk in range(1, 9)
    ):
        mw, nw, kw = waves
        if bm % (mw * 16) or bn % (nw * 16) or k % bk or bk % (kw * 64):
            continue
        if bm > 2 * m or bn > 2 * n:
            continue
        tiles = -(-m // bm) * -(-n // bn)
        # Split-K only to fill one round; never past it.
        if sk > 1 and tiles * sk > GFX942_NUM_CUS:
            continue
        k_tiles = k // bk
        per_split = -(-k_tiles // sk)
        if k_tiles - (sk - 1) * per_split < 2:
            continue
        cfg = normalized_config(dict(
            block_m=bm, block_n=bn, block_k=bk, split_k=sk, m_waves=mw, n_waves=nw,
            k_waves=kw, group_m=8 if tiles > GFX942_NUM_CUS else 4,
            split_k_reduce=sk > 1, direct_store=False, **STREAM_PIPELINE,
        ))
        if valid_config(cfg, m, n, k)[0]:
            out.append((model_cost(m, n, k, cfg), order, cfg))
    out.sort(key=lambda item: item[:2])
    families: dict[tuple, list[dict[str, Any]]] = {}
    for _, _, cfg in out:
        families.setdefault((cfg["block_m"], cfg["block_n"], cfg["split_k"]), []).append(cfg)
    ranked = []
    for depth in range(max((len(f) for f in families.values()), default=0)):
        ranked += [f[depth] for f in families.values() if depth < len(f)]
    return ranked


def refine_configs(cfg: dict[str, Any], n: int) -> Iterable[dict[str, Any]]:
    for group_m in (0, 4, 8, 16):
        if group_m != cfg["group_m"]:
            yield {**cfg, "group_m": group_m}
    if n % cfg["block_n"] == 0 and (cfg["split_k"] == 1 or cfg["split_k_reduce"]):
        yield {**cfg, "direct_store": not cfg["direct_store"]}


# ---------------------------------------------------------------- measuring


class ShapeBench:
    """Inputs, oracle and hipBLASLt baseline of one shape."""

    def __init__(self, m: int, n: int, k: int, out_dtype: torch.dtype, args):
        from aiter import hipb_create_extension
        from aiter.ops.gradlib import _hipb_mm

        self.m, self.n, self.k, self.out_dtype, self.args = m, n, k, out_dtype, args
        torch.manual_seed(0)
        self.x = (torch.randn(m, k, device="cuda") * k**-0.5).to(torch.bfloat16)
        self.w = (torch.randn(n, k, device="cuda") * k**-0.5).to(torch.bfloat16)
        self.reference = self.x.float() @ self.w.float().t()
        self.out = torch.empty((m, n), dtype=out_dtype, device="cuda")
        hipb_create_extension()
        hipb_out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        self.hipb = lambda: _hipb_mm(self.x, self.w.t(), -1, hipb_out, None, None,
                                     None, None, False, False)
        self.hipb_us = self.time(self.hipb)

    def time(self, fn) -> float:
        return graph_time_us(fn, self.args.iters, self.args.warmup, self.args.batch)

    def runner(self, cfg: dict[str, Any]):
        return lambda: gemm_a16w16(self.x, self.w.t(), out=self.out, layout="nt",
                                   user_kwargs=cfg)

    def check(self, cfg: dict[str, Any]) -> tuple[str, float, str]:
        """(status, err_ratio, error) after one correctness run."""
        ok, why = valid_config(cfg, self.m, self.n, self.k)
        if not ok:
            return "invalid", 0.0, why
        try:
            self.runner(cfg)()
            torch.cuda.synchronize()
        except Exception as error:  # noqa: BLE001
            return "error", 0.0, f"{type(error).__name__}: {error}".replace("\n", " ")[:500]
        err = rel_err(self.out, self.reference)
        return ("ok" if err < 2e-2 else "incorrect"), err, ""

    def interleaved(self, cfgs: list[dict[str, Any]], rounds: int) -> tuple[list[float], float]:
        """Median time of each config and of hipBLASLt over interleaved rounds."""
        fns = [self.runner(cfg) for cfg in cfgs] + [self.hipb]
        times = [[] for _ in fns]
        for _ in range(rounds):
            for i, fn in enumerate(fns):
                times[i].append(self.time(fn))
        medians = [statistics.median(t) for t in times]
        return medians[:-1], medians[-1]


def log_measurement(args, regime, bench: ShapeBench, source, cfg, status, err, error,
                    fly_us=None) -> None:
    device, gfx, cu_num = device_metadata()
    m, n, k = bench.m, bench.n, bench.k
    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(), "device": device, "gfx": gfx, "cu_num": cu_num,
        "regime": regime, "M": m, "N": n, "K": k, "source": source,
        **normalized_config(cfg), "status": status, "error": error,
        "err_ratio": f"{err:.8g}" if status in ("ok", "incorrect") else "",
        "hipb_us": f"{bench.hipb_us:.4f}",
        "measurement": f"cuda_graph_batch{args.batch}",
    }
    if fly_us is not None:
        row.update(fly_us=f"{fly_us:.4f}", ratio=f"{fly_us / bench.hipb_us:.6f}",
                   tflops=f"{2 * m * n * k / fly_us / 1e6:.4f}")
    append_row(args.csv, row)


def screen(args, regime, bench: ShapeBench, source: str, cfg: dict[str, Any],
           logged: dict[tuple, float]) -> float:
    """Single screen time of a config (inf if it fails), logged to the measurement CSV."""
    key = config_key(bench.m, bench.n, bench.k, cfg)
    if args.resume and key in logged:
        return logged[key]
    status, err, error = bench.check(cfg)
    fly_us = bench.time(bench.runner(cfg)) if status == "ok" else None
    log_measurement(args, regime, bench, source, cfg, status, err, error, fly_us)
    logged[key] = fly_us if fly_us is not None else float("inf")
    if args.verbose:
        shown = f"{fly_us:8.2f}us" if fly_us is not None else f"{status:>10}"
        print(f"    {source:<10} {shown}  {describe(cfg)}", flush=True)
    return logged[key]


# ---------------------------------------------------------------- modes


def current_config(table: dict, key: tuple, m: int, n: int, k: int):
    """(config, source) gemm_a16w16 would run: the table row, else the heuristic."""
    row = table.get(key)
    if row is not None:
        return parse_config_row(row), "table"
    return normalized_config(gfx942_auto_config(m, n, k)), "heuristic"


def tune_shape(args, regime: str, m: int, n: int, k: int, table: dict) -> None:
    out_dtype = torch.float32 if args.out_dtype == "fp32" else torch.bfloat16
    _, gfx, cu_num = device_metadata()
    key = table_key(gfx, cu_num, m, n, k, args.out_dtype)
    bench = ShapeBench(m, n, k, out_dtype, args)
    incumbent, inc_source = current_config(table, key, m, n, k)
    heuristic = normalized_config(gfx942_auto_config(m, n, k))
    print(f"== {regime} {m}x{n}x{k} {args.out_dtype}: hipBLASLt {bench.hipb_us:.2f}us, "
          f"incumbent ({inc_source}) {describe(incumbent)}", flush=True)

    logged = logged_screen_times(args.csv, m, n, k)
    screened: dict[tuple, tuple[float, dict]] = {}

    def run(source, cfg):
        cfg = normalized_config(cfg)
        ck = config_key(m, n, k, cfg)
        if ck not in screened:
            screened[ck] = (screen(args, regime, bench, source, cfg, logged), cfg)

    run("incumbent", incumbent)
    run("heuristic", heuristic)
    grid = grid_configs(m, n, k)
    budget = args.limit or len(grid)
    for cfg in grid[:budget]:
        run("grid", cfg)
    ranked = sorted((t, i, c) for i, (t, c) in enumerate(screened.values()) if t < float("inf"))
    for _, _, cfg in ranked[: args.refine]:
        for variant in refine_configs(cfg, n):
            run("refine", variant)
    ranked = sorted((t, i, c) for i, (t, c) in enumerate(screened.values()) if t < float("inf"))

    finalists = [incumbent]
    for _, _, cfg in ranked:
        if len(finalists) > args.finalists:
            break
        if cfg not in finalists:
            finalists.append(cfg)
    if heuristic not in finalists:
        finalists.append(heuristic)
    medians, hipb_med = bench.interleaved(finalists, args.rounds)
    for cfg, med in sorted(zip(finalists, medians), key=lambda item: item[1]):
        tag = " (incumbent)" if cfg == incumbent else ""
        print(f"    final {med:8.2f}us ({med / hipb_med:.3f}x)  {describe(cfg)}{tag}", flush=True)

    best = min(range(len(finalists)), key=lambda i: medians[i])
    inc_med = medians[0]
    if best != 0 and medians[best] < inc_med * (1 - args.margin):
        winner, win_med, source = finalists[best], medians[best], "tuned"
        print(f"    PROMOTE {describe(winner)}: {inc_med:.2f} -> {win_med:.2f}us", flush=True)
    else:
        winner, win_med = incumbent, inc_med
        source = table[key]["source"] if key in table else "heuristic"
        print(f"    keep incumbent ({inc_med:.2f}us)", flush=True)
    if not args.dry_run:
        table[key] = table_row(key, winner, win_med, hipb_med, source)
        write_table(args.table, table)


def seed_shape(args, regime: str, m: int, n: int, k: int, table: dict) -> None:
    out_dtype = torch.float32 if args.out_dtype == "fp32" else torch.bfloat16
    _, gfx, cu_num = device_metadata()
    key = table_key(gfx, cu_num, m, n, k, args.out_dtype)
    if key in table and not args.overwrite:
        print(f"  {regime} {m}x{n}x{k}: already in table, skipped", flush=True)
        return
    cfg = normalized_config(gfx942_auto_config(m, n, k))
    bench = ShapeBench(m, n, k, out_dtype, args)
    status, err, error = bench.check(cfg)
    if status != "ok":
        print(f"  {regime} {m}x{n}x{k}: heuristic config {status}: {error}", flush=True)
        return
    (fly_us,), hipb_us = bench.interleaved([cfg], args.rounds)
    print(f"  {regime} {m}x{n}x{k}: {fly_us:.2f}us ({fly_us / hipb_us:.3f}x) "
          f"{describe(cfg)}", flush=True)
    if not args.dry_run:
        table[key] = table_row(key, cfg, fly_us, hipb_us, "heuristic")
        write_table(args.table, table)


def benchmark_shape(args, regime: str, m: int, n: int, k: int, table: dict) -> None:
    """Time what gemm_a16w16 runs by default and log it to the measurement CSV."""
    out_dtype = torch.float32 if args.out_dtype == "fp32" else torch.bfloat16
    _, gfx, cu_num = device_metadata()
    cfg, source = current_config(table, table_key(gfx, cu_num, m, n, k, args.out_dtype),
                                 m, n, k)
    bench = ShapeBench(m, n, k, out_dtype, args)
    status, err, error = bench.check(cfg)
    fly_us = bench.time(bench.runner(cfg)) if status == "ok" else None
    log_measurement(args, regime, bench, source, cfg, status, err, error, fly_us)
    shown = f"{fly_us:.2f}us ({fly_us / bench.hipb_us:.3f}x)" if fly_us else status
    print(f"  {regime:<12} {m:5d}x{n:5d}x{k:5d} {source:<9} {shown}  {describe(cfg)}",
          flush=True)


def print_table(path: Path) -> None:
    for key, row in sorted(read_table(path).items()):
        gfx, cu_num, m, n, k, out_dtype = key
        print(f"{gfx} cu{cu_num} {m}x{n}x{k} {out_dtype}: {row['us']}us "
              f"({row['ratio']}x) {row['source']:<9} {describe(parse_config_row(row))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--M", type=int)
    parser.add_argument("--N", type=int)
    parser.add_argument("--K", type=int)
    parser.add_argument("--regime", default="manual")
    parser.add_argument("--shapes", type=Path)
    parser.add_argument("--out-dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--table", type=Path, default=None,
                        help="tuned table to read/update (default: the file gemm_a16w16 reads)")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="measurement log")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--seed", action="store_true",
                      help="write the heuristic config of each shape into the table")
    mode.add_argument("--current-only", action="store_true",
                      help="benchmark the default config (table row, else heuristic)")
    mode.add_argument("--report-only", action="store_true", help="print the tuned table")
    parser.add_argument("--overwrite", action="store_true", help="--seed replaces existing rows")
    parser.add_argument("--resume", action="store_true",
                        help="reuse screen timings already in the measurement log")
    parser.add_argument("--limit", type=int, default=80,
                        help="grid candidates screened per shape (0 = all)")
    parser.add_argument("--refine", type=int, default=3,
                        help="best screened configs re-tried with other group_m/direct_store")
    parser.add_argument("--finalists", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=5, help="interleaved timing rounds")
    parser.add_argument("--margin", type=float, default=0.02,
                        help="relative gain a challenger needs to replace the incumbent")
    parser.add_argument("--dry-run", action="store_true", help="never write the table")
    parser.add_argument("--verbose", action="store_true", help="print every screened config")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--batch", type=int, default=20)
    args = parser.parse_args()
    args.table = args.table or Path(tuned_table_path())

    if args.report_only:
        print_table(args.table)
        return
    if args.shapes:
        shapes = read_shapes(args.shapes)
    elif None not in (args.M, args.N, args.K):
        shapes = [(args.regime, args.M, args.N, args.K)]
    else:
        parser.error("provide --M/--N/--K or --shapes")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    table = read_table(args.table)
    action = seed_shape if args.seed else benchmark_shape if args.current_only else tune_shape
    for regime, m, n, k in shapes:
        action(args, regime, m, n, k, table)


if __name__ == "__main__":
    main()
