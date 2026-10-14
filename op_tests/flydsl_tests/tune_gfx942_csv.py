"""Resumable CSV tuner for the native gfx942 bf16 GEMM.

The CSV is a measurement database: every attempted configuration is appended
immediately, including compile failures and correctness failures.  Re-running
with ``--resume`` skips completed shape/config keys. Batched CUDA Graph timing
removes Python submission gaps; finalists should still be confirmed with
rocprofv3.

Examples:

  HIP_VISIBLE_DEVICES=6 PYTHONPATH=$PWD \
    flydsl_venv/bin/python op_tests/flydsl_tests/tune_gfx942_csv.py \
      --M 4096 --N 4096 --K 4096 --limit 80 --resume

  HIP_VISIBLE_DEVICES=6 PYTHONPATH=$PWD \
    flydsl_venv/bin/python op_tests/flydsl_tests/tune_gfx942_csv.py \
      --shapes op_tests/flydsl_tests/gfx942_gemm_tuning_shapes.csv \
      --current-only --resume
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import socket
import time
from pathlib import Path
from typing import Any, Iterable

import torch

from aiter.ops.flydsl.kernels.gemm_a16w16_gfx942 import (
    gemm_a16w16,
    gfx942_auto_config,
    make_gemm_a16w16_gfx942_param,
)


HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "gfx942_gemm_tuning.csv"

CONFIG_COLUMNS = [
    "block_m",
    "block_n",
    "block_k",
    "stages",
    "split_k",
    "m_waves",
    "n_waves",
    "k_waves",
    "group_m",
    "async_load_bytes",
    "buffered_load",
    "direct_store",
]
CSV_COLUMNS = [
    "timestamp",
    "host",
    "device",
    "gfx",
    "cu_num",
    "regime",
    "M",
    "N",
    "K",
    "source",
    *CONFIG_COLUMNS,
    "status",
    "error",
    "err_ratio",
    "fly_us",
    "hipb_us",
    "ratio",
    "tflops",
    "measurement",
]


def normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "block_m": 128,
        "block_n": 128,
        "block_k": 64,
        "stages": 2,
        "split_k": 1,
        "m_waves": 2,
        "n_waves": 2,
        "k_waves": 1,
        "group_m": 0,
        "async_load_bytes": 4,
        "buffered_load": False,
        "direct_store": False,
    }
    defaults.update(config)
    return {name: defaults[name] for name in CONFIG_COLUMNS}


def config_key(m: int, n: int, k: int, config: dict[str, Any]) -> tuple[Any, ...]:
    cfg = normalized_config(config)
    return (m, n, k, *(cfg[name] for name in CONFIG_COLUMNS))


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


def read_shapes(path: Path) -> list[tuple[str, int, int, int]]:
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        return [
            (
                row.get("regime", "unspecified"),
                int(row["M"]),
                int(row["N"]),
                int(row["K"]),
            )
            for row in rows
        ]


def completed_keys(path: Path) -> set[tuple[Any, ...]]:
    if not path.exists():
        return set()
    keys = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            cfg = {
                name: (
                    row[name].lower() == "true"
                    if name in ("buffered_load", "direct_store")
                    else int(row[name])
                )
                for name in CONFIG_COLUMNS
            }
            keys.add(config_key(int(row["M"]), int(row["N"]), int(row["K"]), cfg))
    return keys


def append_row(path: Path, row: dict[str, Any]) -> None:
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in CSV_COLUMNS})
        handle.flush()
        os.fsync(handle.fileno())


def print_best_rows(
    path: Path, shape: tuple[int, int, int] | None = None
) -> None:
    """Print the best saved configuration for each shape."""
    if not path.exists():
        print(f"No tuning database at {path}")
        return
    best: dict[tuple[str, int, int, int], dict[str, str]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["status"] != "ok" or not row["ratio"]:
                continue
            dimensions = (int(row["M"]), int(row["N"]), int(row["K"]))
            if shape is not None and dimensions != shape:
                continue
            key = (row["regime"], *dimensions)
            if key not in best or float(row["ratio"]) < float(best[key]["ratio"]):
                best[key] = row
    for key in sorted(best):
        row = best[key]
        cfg = {
            name: (
                row[name].lower() == "true"
                if name in ("buffered_load", "direct_store")
                else int(row[name])
            )
            for name in CONFIG_COLUMNS
        }
        regime, m, n, k = key
        print(
            f"SAVED BEST {regime} {m}x{n}x{k}: {float(row['fly_us']):.3f}us "
            f"({float(row['ratio']):.3f}x hipBLASLt) "
            f"{json.dumps(cfg, sort_keys=True)}"
        )


def focused_configs(m: int, n: int, k: int) -> Iterable[tuple[str, dict[str, Any]]]:
    """Curated systematic neighborhood around the known gfx942 regimes."""
    auto = normalized_config(gfx942_auto_config(m, n, k))
    yield "auto", auto

    def preferred(first, values):
        return list(dict.fromkeys([first, *values]))

    tiles = preferred((auto["block_m"], auto["block_n"]), [
        (32, 32),
        (32, 64),
        (64, 64),
        (128, 64),
        (128, 96),
        (128, 128),
        (192, 96),
        (192, 128),
        (256, 128),
        (256, 224),
    ])
    waves = preferred((auto["m_waves"], auto["n_waves"]), [
        (2, 2),
        (4, 2),
        (2, 4),
        (4, 4),
        (3, 2),
        (6, 2),
    ])
    block_ks = [
        bk for bk in preferred(auto["block_k"], [64, 128, 256]) if k % bk == 0
    ]
    split_ks = [
        sk
        for sk in preferred(auto["split_k"], [1, 2, 4, 8])
        if k % sk == 0
    ]

    def neighbor(config: dict[str, Any]):
        cfg = normalized_config(config)
        yield "neighbor", cfg
        if (
            cfg["split_k"] == 1
            and m % cfg["block_m"] == 0
            and n % cfg["block_n"] == 0
        ):
            direct = dict(cfg)
            direct["direct_store"] = not cfg["direct_store"]
            yield "neighbor-direct", direct

    # Cover each axis once before entering the exhaustive Cartesian product.
    # This makes a small --limit useful while preserving deterministic resume.
    for bm, bn in tiles:
        yield from neighbor({**auto, "block_m": bm, "block_n": bn})
    for bk in block_ks:
        yield from neighbor({**auto, "block_k": bk})
    for mw, nw in waves:
        yield from neighbor({**auto, "m_waves": mw, "n_waves": nw})
    for split_k in split_ks:
        yield from neighbor({**auto, "split_k": split_k})
    for group_m in preferred(auto["group_m"], [0, 8] if n <= 1024 else [0]):
        yield from neighbor({**auto, "group_m": group_m})
    if auto["split_k"] == 1:
        for stages, buffered_load, dma in (
            (2, False, 4),
            (1, True, 16),
            (2, True, 16),
        ):
            yield from neighbor(
                {
                    **auto,
                    "stages": stages,
                    "buffered_load": buffered_load,
                    "async_load_bytes": dma,
                }
            )

    for (bm, bn), bk, (mw, nw), split_k in itertools.product(
        tiles, block_ks, waves, split_ks
    ):
        if bm % (mw * 16) or bn % (nw * 16):
            continue
        if k % (split_k * bk):
            continue

        group_ms = preferred(auto["group_m"], [0, 8] if n <= 1024 else [0])
        pipelines = preferred(
            (
                auto["stages"],
                auto["buffered_load"],
                auto["async_load_bytes"],
            ),
            [(2, False, 4)],
        )
        if split_k == 1:
            pipelines = preferred(
                pipelines[0],
                [*pipelines[1:], (1, True, 16), (2, True, 16)],
            )
        else:
            pipelines = [(2, False, 4), (3, False, 4), (4, False, 4)]

        for group_m, (stages, buffered_load, dma) in itertools.product(
            group_ms, pipelines
        ):
            base = normalized_config(
                {
                    "block_m": bm,
                    "block_n": bn,
                    "block_k": bk,
                    "stages": stages,
                    "split_k": split_k,
                    "m_waves": mw,
                    "n_waves": nw,
                    "group_m": group_m,
                    "async_load_bytes": dma,
                    "buffered_load": buffered_load,
                }
            )
            yield "focused", base
            if (
                split_k == 1
                and m % bm == 0
                and n % bn == 0
            ):
                direct = dict(base)
                direct["direct_store"] = True
                yield "focused-direct", direct


def valid_config(config: dict[str, Any], m: int, n: int, k: int) -> tuple[bool, str]:
    cfg = normalized_config(config)
    if k % cfg["block_k"]:
        return False, "K not divisible by block_k"
    if k % (cfg["split_k"] * cfg["block_k"]):
        return False, "K not divisible by split_k*block_k"
    if cfg["direct_store"] and (m % cfg["block_m"] or n % cfg["block_n"]):
        return False, "direct_store requires exact M/N tiles"
    try:
        make_gemm_a16w16_gfx942_param(**cfg)
    except Exception as error:  # validation failures are useful tuning evidence
        return False, str(error)
    return True, ""


def device_metadata() -> tuple[str, str, int]:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    name = props.name
    # This tuner is gfx942-specific; recording it explicitly keeps the CSV
    # compatible with AITER's existing tuned-config schema.
    return name, "gfx942", 304


def tune_shape(
    *,
    regime: str,
    m: int,
    n: int,
    k: int,
    output_csv: Path,
    done: set[tuple[Any, ...]],
    current_only: bool,
    limit: int,
    warmup: int,
    iters: int,
    batch: int,
) -> None:
    torch.manual_seed(0)
    x = (torch.randn(m, k, device="cuda") * k**-0.5).to(torch.bfloat16)
    w = (torch.randn(n, k, device="cuda") * k**-0.5).to(torch.bfloat16)
    reference = x.float() @ w.float().t()

    from aiter import hipb_create_extension
    from aiter.ops.gradlib import _hipb_mm

    hipb_create_extension()
    hipb_output = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    hipb_us = graph_time_us(
        lambda: _hipb_mm(
            x,
            w.t(),
            -1,
            hipb_output,
            None,
            None,
            None,
            None,
            False,
            False,
        ),
        iters,
        warmup,
        batch,
    )
    device, gfx, cu_num = device_metadata()
    host = socket.gethostname()

    candidates = focused_configs(m, n, k)
    seen: set[tuple[Any, ...]] = set()
    attempted = 0
    for source, config in candidates:
        key = config_key(m, n, k, config)
        if key in seen:
            continue
        seen.add(key)
        if current_only and source != "auto":
            continue
        if key in done:
            continue
        if limit and attempted >= limit:
            break
        attempted += 1

        cfg = normalized_config(config)
        ok, validation_error = valid_config(cfg, m, n, k)
        row: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": host,
            "device": device,
            "gfx": gfx,
            "cu_num": cu_num,
            "regime": regime,
            "M": m,
            "N": n,
            "K": k,
            "source": source,
            **cfg,
            "hipb_us": f"{hipb_us:.4f}",
            "measurement": f"cuda_graph_batch{batch}",
        }
        if not ok:
            row.update(status="invalid", error=validation_error)
            append_row(output_csv, row)
            done.add(key)
            continue

        try:
            output = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
            run = lambda: gemm_a16w16(  # noqa: E731
                x,
                w.t(),
                out=output,
                layout="nt",
                user_kwargs=cfg,
            )
            run()
            torch.cuda.synchronize()
            error = rel_err(output, reference)
            if error >= 2e-2:
                row.update(status="incorrect", err_ratio=f"{error:.8g}")
            else:
                fly_us = graph_time_us(
                    run,
                    iters,
                    warmup,
                    batch,
                )
                flop = 2 * m * n * k
                row.update(
                    status="ok",
                    err_ratio=f"{error:.8g}",
                    fly_us=f"{fly_us:.4f}",
                    ratio=f"{fly_us / hipb_us:.6f}",
                    tflops=f"{flop / fly_us / 1e6:.4f}",
                )
        except Exception as error:
            row.update(
                status="error",
                error=f"{type(error).__name__}: {error}".replace("\n", " ")[:500],
            )

        append_row(output_csv, row)
        done.add(key)
        status = row["status"]
        timing = f"{row.get('fly_us', '-')}us" if status == "ok" else "-"
        print(
            f"  {regime:<14} {m:5d}x{n:5d}x{k:5d} "
            f"{status:<9} {timing:>12} {json.dumps(cfg, sort_keys=True)}",
            flush=True,
        )

    print_best_rows(output_csv, shape=(m, n, k))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int)
    parser.add_argument("--N", type=int)
    parser.add_argument("--K", type=int)
    parser.add_argument("--regime", default="manual")
    parser.add_argument("--shapes", type=Path)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--current-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="max new configs per shape")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--batch", type=int, default=20)
    args = parser.parse_args()

    if args.report_only:
        print_best_rows(args.csv)
        return

    if args.shapes:
        shapes = read_shapes(args.shapes)
    elif None not in (args.M, args.N, args.K):
        shapes = [(args.regime, args.M, args.N, args.K)]
    else:
        parser.error("provide --M/--N/--K or --shapes")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    done = completed_keys(args.csv) if args.resume else set()
    for regime, m, n, k in shapes:
        tune_shape(
            regime=regime,
            m=m,
            n=n,
            k=k,
            output_csv=args.csv,
            done=done,
            current_only=args.current_only,
            limit=args.limit,
            warmup=args.warmup,
            iters=args.iters,
            batch=args.batch,
        )


if __name__ == "__main__":
    main()
