# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Runtime correctness for FlyDSL INT4 QuickReduce (``QRInt4``).

Pytest collects validity cases only (no timing). ``python3`` this file
runs an aiter-op-test ``@benchmark`` / markdown sweep. Each rank times
``fly.allreduce`` with ``run_perftest`` after ``compile()``. The oracle
is an untimed fp32 NCCL all-reduce of the same per-rank inputs. INT4 is
lossy, so validity uses SQNR, a calibrated mismatch ratio, and a
per-tile SQNR floor.

5120 is a measured calibration width, not an ABI requirement. The kernel
is gfx942/gfx950 TP∈{2,4,8}; other archs skip. Pytest skips a world size
when fewer GPUs are visible than TP.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest

pytest.importorskip("flydsl")

from aiter.ops.flydsl.kernels.qr_int4 import DEFAULT_GRID_CAP
from aiter.ops.flydsl.kernels.qr_int4_kernel import (
    SUPPORTED_WORLDS,
    TILE_BYTES,
    WORLD,
)

ARCH = get_gfx_runtime()
SUPPORTED_ARCHS = ("gfx942", "gfx950")
SQNR_MIN_DB = 18.0
# Dropped/stale 32 KiB tile is ~0 dB; INT4 codec tiles stay well above this.
TILE_SQNR_MIN_DB = 8.0
# Calibrated to the INT4 group-16 codec vs fp32 all-reduce, not bit identity.
CLOSE_RTOL = 1e-1
CLOSE_ATOL = 1e-1
CLOSE_ERR_RATIO = 0.5
SUPER_TILE = 8
TP = WORLD

pytestmark = pytest.mark.skipif(
    ARCH not in SUPPORTED_ARCHS,
    reason="QRInt4 unsupported arch (need gfx942 or gfx950)",
)

# Distinct correctness branches, not a tokens x hidden product.
# hidden=5120 is calibration; 4096 proves the map is not width-locked.
# (8, 1024) is a half-tile tail (TILE_BYTES = 32 KiB).
# TP2/4: ST=1 calib plus one ST=8 case (num_tiles > grid_cap) each.
# Pytest skips a world size when fewer GPUs are visible than TP.
_PYTEST_CASES = (
    (8, 8, 1024, "partial-tile"),
    (8, 512, 5120, "st1-auto-calib"),
    (8, 9216, 4096, "st8-alt-hidden"),
    (8, 32768, 5120, "st8-calib-prefill"),
    (4, 512, 5120, "tp4-st1-auto-calib"),
    (4, 9216, 4096, "tp4-st8-alt-hidden"),
    (2, 512, 5120, "tp2-st1-auto-calib"),
    (2, 9216, 4096, "tp2-st8-alt-hidden"),
)


def _num_tiles(tokens: int, hidden: int) -> int:
    nbytes = tokens * hidden * 2
    return max(1, (nbytes + TILE_BYTES - 1) // TILE_BYTES)


def _pick_st(
    tokens: int,
    hidden: int,
    requested: int = SUPER_TILE,
    *,
    grid_cap: int = DEFAULT_GRID_CAP,
) -> int:
    tiles = _num_tiles(tokens, hidden)
    if requested == 1 or tiles > grid_cap:
        return requested
    return 1


def _sqnr(ref_pow: torch.Tensor, mse: torch.Tensor) -> torch.Tensor:
    score = torch.where(
        (ref_pow <= 0) & (mse <= 0),
        torch.full_like(mse, float("inf")),
        10.0 * torch.log10(ref_pow / mse),
    )
    return torch.nan_to_num(score, nan=float("-inf"), neginf=float("-inf"))


def _sqnr_db(got: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        _sqnr((reference * reference).mean(), ((got - reference) ** 2).mean()).item()
    )


def _min_tile_sqnr_db(got: torch.Tensor, reference: torch.Tensor) -> float:
    """Worst 32 KiB-tile SQNR. A dropped or stale tile is ~0 dB."""
    tile_elems = TILE_BYTES // 2
    g = got.reshape(-1)
    r = reference.reshape(-1)
    n = int(g.numel())
    n_full = (n // tile_elems) * tile_elems
    vals = []
    if n_full:
        gt = g[:n_full].view(-1, tile_elems)
        rt = r[:n_full].view(-1, tile_elems)
        mse = ((gt - rt) ** 2).mean(dim=1)
        pow_ = (rt * rt).mean(dim=1)
        vals.append(float(_sqnr(pow_, mse).min().item()))
    if n > n_full:
        vals.append(_sqnr_db(g[n_full:], r[n_full:]))
    return min(vals) if vals else _sqnr_db(got, reference)


def _run_rank(args) -> None:
    import torch.distributed as dist

    from aiter.ops.flydsl import QRInt4

    rank = args.rank
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=args.init_method,
        world_size=args.tp,
        rank=rank,
    )
    gloo = dist.new_group(backend="gloo")
    group = dist.group.WORLD

    fly = QRInt4(
        group=gloo,
        device=device,
        rank=rank,
        world_size=args.tp,
        super_tile=args.super_tile,
        grid_cap=args.grid_cap,
    )
    compile_tokens = min(512, max(args.tokens))
    compile_hidden = max(args.hiddens)
    compile_inp = torch.empty(
        (compile_tokens, compile_hidden), device=device, dtype=torch.bfloat16
    )
    compile_out = torch.empty_like(compile_inp)
    dist.barrier()
    fly.compile(compile_inp, compile_out)
    dist.barrier()
    del compile_inp, compile_out

    rows = []
    for tokens, hidden in zip(args.tokens, args.hiddens, strict=True):
        gen = torch.Generator().manual_seed(1234 + rank)
        inp = (
            torch.randn(tokens, hidden, generator=gen, dtype=torch.float32) * 0.1
        ).to(device=device, dtype=torch.bfloat16)
        out = torch.empty_like(inp)
        ref = inp.to(torch.float32)
        dist.all_reduce(ref, group=group)

        dist.barrier()
        out.zero_()
        fly.allreduce(inp, out)
        torch.cuda.synchronize()
        got = out.to(torch.float32)
        dist.barrier()
        nbytes = int(inp.numel()) * int(inp.element_size())
        tiles = max(1, (nbytes + TILE_BYTES - 1) // TILE_BYTES)
        st_used = fly._pick_st(tiles)
        st1 = fly._by_st.get(1, fly._by_st[st_used])
        close_err = checkAllclose(
            ref,
            got,
            rtol=CLOSE_RTOL,
            atol=CLOSE_ATOL,
            tol_err_ratio=CLOSE_ERR_RATIO,
            printLog=False,
            msg=f"qr_int4 rank {rank}",
        )
        row = {
            "tokens": tokens,
            "hidden": hidden,
            "grid_cap": args.grid_cap,
            "st1_grid": st1.grid,
            "st_used": st_used,
            "grid": fly._by_st[st_used].grid,
            "sqnr_db": _sqnr_db(got, ref),
            "min_tile_sqnr_db": _min_tile_sqnr_db(got, ref),
            "err": close_err,
            "us": None,
        }
        if args.time_it:
            dist.barrier(group=group)
            torch.cuda.synchronize()

            def _allreduce(eng=fly, src=inp, dst=out):
                eng.allreduce(src, dst)
                return dst

            _, us = run_perftest(_allreduce)
            row["us"] = us
        rows.append(row)
        del inp, out, ref, got
        torch.cuda.empty_cache()

    gathered = [None] * args.tp
    dist.all_gather_object(gathered, rows, group=gloo)
    if rank == 0 and args.out:
        with open(args.out, "w") as fh:
            json.dump({"ranks": gathered}, fh)
    dist.barrier(group=group)
    fly.close()
    dist.destroy_process_group()


def _spawn(
    world_size: int,
    pairs: list[tuple[int, int]],
    *,
    time_it: bool,
    super_tile: int = SUPER_TILE,
    grid_cap: int = DEFAULT_GRID_CAP,
) -> list[list[dict]]:
    # HIP QR/fused-AR use multiprocessing.Pool from ``python3`` __main__.
    # This file is also collected by pytest, and FlyDSL JIT needs a fresh
    # interpreter per rank, so ranks are Popen of this file with --rank
    # (not Pool / torchrun). Init method matches the HIP QR helpers.
    if world_size not in SUPPORTED_WORLDS:
        raise ValueError(f"unsupported world_size={world_size}")
    n_gpu = torch.cuda.device_count()
    if n_gpu < world_size:
        pytest.skip(f"QRInt4 needs {world_size} GPUs, have {n_gpu}")
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    out_path = os.path.join(tempfile.mkdtemp(prefix="flydsl_qr_int4_"), "rank0.json")
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        f"{_REPO_ROOT}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else _REPO_ROOT
    )
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("FLYDSL_GPU_ARCH", ARCH)
    tokens = ",".join(str(t) for t, _ in pairs)
    hiddens = ",".join(str(h) for _, h in pairs)
    procs = []
    logs = []
    for rank in range(world_size):
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--rank",
            str(rank),
            "--init-method",
            init_method,
            "--tp",
            str(world_size),
            "--tokens",
            tokens,
            "--hiddens",
            hiddens,
            "--super-tile",
            str(super_tile),
            "--grid-cap",
            str(grid_cap),
        ]
        if time_it:
            cmd.append("--time-it")
        if rank == 0:
            cmd += ["--out", out_path]
        log = open(  # noqa: SIM115
            f"/tmp/flydsl_qr_int4_tp{world_size}_rank{rank}.log",
            "w",
        )
        procs.append(
            subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        )
        logs.append(log)
    rc = 0
    deadline = time.time() + float(os.environ.get("FLYDSL_QR_TIMEOUT", "3600"))
    for proc in procs:
        try:
            rc |= proc.wait(timeout=max(1.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            proc.kill()
            rc |= 1
    for log in logs:
        log.close()
    if rc != 0:
        tails = []
        for rank in range(world_size):
            path = f"/tmp/flydsl_qr_int4_tp{world_size}_rank{rank}.log"
            try:
                with open(path) as fh:
                    tails.append(f"===== rank {rank} =====\n{fh.read()[-4000:]}")
            except OSError:
                pass
        raise RuntimeError("QRInt4 ranks failed\n" + "\n".join(tails))
    with open(out_path) as fh:
        payload = json.load(fh)
    ranks = payload["ranks"]
    if len(ranks) != world_size:
        raise RuntimeError(f"QRInt4 gathered {len(ranks)} ranks, expected {world_size}")
    return ranks


def _assert_validity(
    ranks: list[list[dict]],
    *,
    tokens: int,
    hidden: int,
    world_size: int,
    label: str,
) -> dict:
    st1_grid = ranks[0][0]["st1_grid"]
    expected_st = _pick_st(tokens, hidden, grid_cap=st1_grid)
    if len(ranks) != world_size:
        raise AssertionError(
            f"{label}: gathered {len(ranks)} ranks, expected {world_size}"
        )
    fails = []
    for rank, rows in enumerate(ranks):
        if not rows:
            fails.append(f"rank {rank}: no rows")
            continue
        row = rows[0]
        if row["st_used"] != expected_st:
            fails.append(f"rank {rank}: ST={row['st_used']}, expected {expected_st}")
        if row["sqnr_db"] < SQNR_MIN_DB:
            fails.append(
                f"rank {rank}: SQNR {row['sqnr_db']:.2f} dB < {SQNR_MIN_DB}"
            )
        if row["min_tile_sqnr_db"] < TILE_SQNR_MIN_DB:
            fails.append(
                f"rank {rank}: min-tile SQNR {row['min_tile_sqnr_db']:.2f} dB "
                f"< {TILE_SQNR_MIN_DB}"
            )
        if row["err"] >= CLOSE_ERR_RATIO:
            fails.append(
                f"rank {rank}: checkAllclose err {row['err']:.3f} "
                f">= {CLOSE_ERR_RATIO}"
            )
    if fails:
        raise AssertionError(
            f"{label} tp={world_size} tokens={tokens} hidden={hidden}: "
            + "; ".join(fails)
        )
    return ranks[0][0]


@pytest.mark.parametrize("world_size,tokens,hidden,label", _PYTEST_CASES)
def test_qr_int4_sqnr_vs_fp32_allreduce(world_size, tokens, hidden, label):
    ranks = _spawn(world_size, [(tokens, hidden)], time_it=False)
    _assert_validity(
        ranks,
        tokens=tokens,
        hidden=hidden,
        world_size=world_size,
        label=label,
    )


@benchmark()
def test_qr_int4(tokens, hidden, dtype, tp, grid_cap=DEFAULT_GRID_CAP):
    ranks = _spawn(tp, [(tokens, hidden)], time_it=True, grid_cap=grid_cap)
    row = _assert_validity(
        ranks,
        tokens=tokens,
        hidden=hidden,
        world_size=tp,
        label="bench",
    )
    nbytes = tokens * hidden * 2
    # Reduce of tp ranks: (world-1) adds per element, plus INT4 codec ALU.
    flops = tokens * hidden * (tp - 1)
    rank_us = [r[0]["us"] for r in ranks]
    us = statistics.median(rank_us)
    return {
        "gfx": ARCH,
        "tp": tp,
        "grid_cap": row["grid_cap"],
        "st_used": row["st_used"],
        "flydsl us": us,
        "flydsl TFLOPS": (flops / us / 1e6) if us else 0.0,
        "flydsl TB/s": (nbytes / us / 1e6) if us else 0.0,
        "flydsl err": max(r[0]["err"] for r in ranks),
        "flydsl sqnr_db": min(r[0]["sqnr_db"] for r in ranks),
    }


test_qr_int4.__test__ = False


def main():
    if ARCH not in SUPPORTED_ARCHS:
        aiter.logger.warning("QRInt4 unsupported on %s; skipping", ARCH)
        return
    n_gpu = torch.cuda.device_count()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.d_dtypes["bf16"]],
        help="Payload dtype (bf16 only).\n    e.g.: -d bf16",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[1],
        help="Unused (not batch). Values other than 1 are skipped.",
    )
    parser.add_argument(
        "--tp",
        type=int,
        nargs="*",
        default=[TP],
        help="World sizes to sweep (2, 4, or 8). Default 8.\n    e.g.: --tp 8",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (512, 5120),
            (9216, 5120),
            (32768, 5120),
        ],
        help="(tokens, hidden) pairs. 5120 is calibration, not ABI.\n"
        "    e.g.: -s 512,5120 9216,5120",
    )
    parser.add_argument(
        "-o",
        "--out",
        default=None,
        help="Optional JSON output path (rank-0 bench payload).",
    )
    parser.add_argument(
        "--grid-cap",
        type=int,
        default=DEFAULT_GRID_CAP,
        help="Persistent launch/inbox block cap (default 304*4=1216, occupancy-clamped).",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        if dtype != dtypes.bf16:
            aiter.logger.warning("QRInt4 payload is bf16; skipping %s", dtype)
            continue
        df = []
        for tp, batch, mnk in itertools.product(args.tp, args.batch, args.mnk):
            if batch != 1:
                continue
            if tp not in SUPPORTED_WORLDS:
                aiter.logger.warning("QRInt4 unsupported world_size=%s; skipping", tp)
                continue
            if n_gpu < tp:
                aiter.logger.warning(
                    "QRInt4 needs %s GPUs, have %s; skipping tp=%s",
                    tp,
                    n_gpu,
                    tp,
                )
                continue
            if not isinstance(mnk, tuple) or len(mnk) < 2:
                raise ValueError(f"-s expects tokens,hidden; got {mnk!r}")
            tokens, hidden = int(mnk[0]), int(mnk[1])
            df.append(test_qr_int4(tokens, hidden, dtype, tp, grid_cap=args.grid_cap))
        if df:
            table = pd.DataFrame(df)
            aiter.logger.info(
                "flydsl QR INT4 summary (markdown):\n%s",
                table.to_markdown(index=False),
            )
            if args.out:
                out_dir = os.path.dirname(os.path.abspath(args.out))
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                with open(args.out, "w") as fh:
                    json.dump(
                        {
                            "meta": {
                                "gfx": ARCH,
                                "grid_cap": args.grid_cap,
                                "timer": "run_perftest",
                            },
                            "rows": df,
                        },
                        fh,
                        indent=2,
                        default=str,
                    )
                aiter.logger.info("wrote %s", args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--init-method", default=None)
    parser.add_argument("--tokens", default="")
    parser.add_argument("--hiddens", default="")
    parser.add_argument("--super-tile", type=int, default=SUPER_TILE)
    parser.add_argument("--grid-cap", type=int, default=DEFAULT_GRID_CAP)
    parser.add_argument("--time-it", action="store_true")
    parser.add_argument("--out", default=None)
    known, rest = parser.parse_known_args()
    if known.rank is not None:
        rank_parser = argparse.ArgumentParser()
        rank_parser.add_argument("--tp", type=int, default=TP)
        rank_args, _ = rank_parser.parse_known_args(rest)
        known.tp = rank_args.tp
        known.tokens = [int(t) for t in known.tokens.split(",") if t]
        known.hiddens = [int(h) for h in known.hiddens.split(",") if h]
        if len(known.tokens) != len(known.hiddens):
            raise SystemExit("tokens and hiddens lists must match")
        _run_rank(known)
    else:
        sys.argv = [sys.argv[0]] + rest
        main()
