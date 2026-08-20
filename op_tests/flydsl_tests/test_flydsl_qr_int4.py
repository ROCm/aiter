# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Runtime correctness for FlyDSL INT4 QuickReduce (``QRInt4``).

Pytest collects validity cases only (no timing). ``python3`` this file
runs an aiter-op-test ``@benchmark`` / markdown sweep. Each rank times
``fly.allreduce`` with default ``run_perftest`` after ``compile()``. The
oracle is an untimed fp32 NCCL all-reduce of the same per-rank inputs.
INT4 is lossy, so the gate is SQNR vs that oracle, not bit-identity or
tight ``checkAllclose``.

5120 is a measured calibration width, not an ABI requirement. The kernel
is gfx942 TP8 only; other archs skip.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
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
from aiter.test_common import benchmark, run_perftest

pytest.importorskip("flydsl")

from aiter.ops.flydsl.kernels.qr_int4_kernel import (
    GRID,
    TILE_BYTES,
    WORLD,
)

ARCH = get_gfx_runtime()
SUPPORTED_ARCHS = ("gfx942",)
SQNR_MIN_DB = 18.0
SUPER_TILE = 8
TP = WORLD

pytestmark = pytest.mark.skipif(
    ARCH not in SUPPORTED_ARCHS,
    reason="QRInt4 is gfx942-only",
)

# Distinct correctness branches, not a tokens x hidden product.
# hidden=5120 is calibration; 4096 proves the map is not width-locked.
# (8, 1024) is a half-tile tail (TILE_BYTES = 32 KiB).
_PYTEST_CASES = (
    (8, 1024, "partial-tile"),
    (512, 5120, "st1-auto-calib"),
    (9216, 4096, "st8-alt-hidden"),
    (32768, 5120, "st8-calib-prefill"),
)


def _num_tiles(tokens: int, hidden: int) -> int:
    nbytes = tokens * hidden * 2
    return max(1, (nbytes + TILE_BYTES - 1) // TILE_BYTES)


def _pick_st(tokens: int, hidden: int, requested: int = SUPER_TILE) -> int:
    tiles = _num_tiles(tokens, hidden)
    if requested == 1 or tiles > GRID:
        return requested
    return 1


def _sqnr_db(got: torch.Tensor, reference: torch.Tensor) -> float:
    mse = float(((got - reference) ** 2).mean().item())
    ref_pow = float((reference * reference).mean().item())
    if mse <= 0.0 or ref_pow <= 0.0:
        return float("inf")
    return 10.0 * math.log10(ref_pow / mse)


def _rel_mae(got: torch.Tensor, reference: torch.Tensor) -> float:
    scale = float(reference.abs().mean().item())
    err = float((got - reference).abs().mean().item())
    return err / scale if scale else 0.0


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
    try:
        gloo = dist.new_group(backend="gloo")
    except Exception:  # noqa: BLE001
        gloo = dist.group.WORLD
    group = dist.group.WORLD

    fly = QRInt4(
        group=gloo,
        device=device,
        rank=rank,
        world_size=args.tp,
        super_tile=args.super_tile,
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
        dist.barrier()
        got = out.to(torch.float32)
        nbytes = int(inp.numel()) * int(inp.element_size())
        tiles = max(1, (nbytes + TILE_BYTES - 1) // TILE_BYTES)
        row = {
            "tokens": tokens,
            "hidden": hidden,
            "st_used": fly._pick_st(tiles),
            "sqnr_db": _sqnr_db(got, ref),
            "rel_mae": _rel_mae(got, ref),
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
    dist.all_gather_object(gathered, rows, group=group)
    if rank == 0 and args.out:
        with open(args.out, "w") as fh:
            json.dump({"ranks": gathered}, fh)
    dist.barrier(group=group)
    fly.close()
    dist.destroy_process_group()


def _spawn_tp8(
    pairs: list[tuple[int, int]],
    *,
    time_it: bool,
    super_tile: int = SUPER_TILE,
) -> list[dict]:
    # HIP QR/fused-AR use multiprocessing.Pool from ``python3`` __main__.
    # This file is also collected by pytest, and FlyDSL JIT needs a fresh
    # interpreter per rank, so ranks are Popen of this file with --rank
    # (not Pool / torchrun). Init method matches the HIP QR helpers.
    if torch.cuda.device_count() < TP:
        pytest.skip(f"QRInt4 needs {TP} GPUs, have {torch.cuda.device_count()}")
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    out_path = os.path.join(tempfile.mkdtemp(prefix="flydsl_qr_int4_"), "rank0.json")
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        f"{_REPO_ROOT}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else _REPO_ROOT
    )
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("FLYDSL_GPU_ARCH", "gfx942")
    env.pop("HIP_VISIBLE_DEVICES", None)
    tokens = ",".join(str(t) for t, _ in pairs)
    hiddens = ",".join(str(h) for _, h in pairs)
    procs = []
    logs = []
    for rank in range(TP):
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--rank",
            str(rank),
            "--init-method",
            init_method,
            "--tokens",
            tokens,
            "--hiddens",
            hiddens,
            "--super-tile",
            str(super_tile),
        ]
        if time_it:
            cmd.append("--time-it")
        if rank == 0:
            cmd += ["--out", out_path]
        log = open(f"/tmp/flydsl_qr_int4_rank{rank}.log", "w")  # noqa: SIM115
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
        for rank in range(TP):
            path = f"/tmp/flydsl_qr_int4_rank{rank}.log"
            try:
                with open(path) as fh:
                    tails.append(f"===== rank {rank} =====\n{fh.read()[-4000:]}")
            except OSError:
                pass
        raise RuntimeError("QRInt4 ranks failed\n" + "\n".join(tails))
    with open(out_path) as fh:
        payload = json.load(fh)
    return payload["ranks"][0]


@pytest.mark.parametrize("tokens,hidden,label", _PYTEST_CASES)
def test_qr_int4_sqnr_vs_fp32_allreduce(tokens, hidden, label):
    rows = _spawn_tp8([(tokens, hidden)], time_it=False)
    row = rows[0]
    expected_st = _pick_st(tokens, hidden)
    assert (
        row["st_used"] == expected_st
    ), f"{label}: host picked ST={row['st_used']}, expected {expected_st}"
    assert row["sqnr_db"] >= SQNR_MIN_DB, (
        f"{label} tokens={tokens} hidden={hidden} ST={row['st_used']}: "
        f"SQNR {row['sqnr_db']:.2f} dB < {SQNR_MIN_DB} (rel MAE {row['rel_mae']:.3e})"
    )


@benchmark()
def test_qr_int4(tokens, hidden, dtype):
    rows = _spawn_tp8([(tokens, hidden)], time_it=True)
    row = rows[0]
    nbytes = tokens * hidden * 2
    # Reduce of 8 ranks: (world-1) adds per element, plus INT4 codec ALU.
    flops = tokens * hidden * (TP - 1)
    if row["sqnr_db"] < SQNR_MIN_DB:
        raise AssertionError(
            f"SQNR {row['sqnr_db']:.2f} dB < {SQNR_MIN_DB} "
            f"for {tokens}x{hidden} ST={row['st_used']}"
        )
    us = row["us"] or 0.0
    return {
        "gfx": ARCH,
        "st_used": row["st_used"],
        "flydsl us": us,
        "flydsl TFLOPS": (flops / us / 1e6) if us else 0.0,
        "flydsl TB/s": (nbytes / us / 1e6) if us else 0.0,
        "flydsl err": row["rel_mae"],
        "flydsl sqnr_db": row["sqnr_db"],
    }


test_qr_int4.__test__ = False


def main():
    if ARCH not in SUPPORTED_ARCHS:
        aiter.logger.warning("QRInt4 unsupported on %s; skipping", ARCH)
        return
    if torch.cuda.device_count() < TP:
        aiter.logger.warning(
            "QRInt4 needs %s GPUs, have %s; skipping",
            TP,
            torch.cuda.device_count(),
        )
        return

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
        help="Unused (world size is 8). Values other than 1 are skipped.",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (512, 4096),
            (512, 5120),
            (9216, 4096),
            (9216, 5120),
            (32768, 5120),
        ],
        help="(tokens, hidden) pairs. 5120 is calibration, not ABI.\n"
        "    e.g.: -s 512,4096 9216,4096",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        if dtype != dtypes.bf16:
            aiter.logger.warning("QRInt4 payload is bf16; skipping %s", dtype)
            continue
        df = []
        for batch, mnk in itertools.product(args.batch, args.mnk):
            if batch != 1:
                continue
            if not isinstance(mnk, tuple) or len(mnk) < 2:
                raise ValueError(f"-s expects tokens,hidden; got {mnk!r}")
            tokens, hidden = int(mnk[0]), int(mnk[1])
            df.append(test_qr_int4(tokens, hidden, dtype))
        if df:
            table = pd.DataFrame(df)
            aiter.logger.info(
                "flydsl QR INT4 summary (markdown):\n%s",
                table.to_markdown(index=False),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--init-method", default=None)
    parser.add_argument("--tp", type=int, default=TP)
    parser.add_argument("--tokens", default="")
    parser.add_argument("--hiddens", default="")
    parser.add_argument("--super-tile", type=int, default=SUPER_TILE)
    parser.add_argument("--time-it", action="store_true")
    parser.add_argument("--out", default=None)
    known, rest = parser.parse_known_args()
    if known.rank is not None:
        known.tokens = [int(t) for t in known.tokens.split(",") if t]
        known.hiddens = [int(h) for h in known.hiddens.split(",") if h]
        if len(known.tokens) != len(known.hiddens):
            raise SystemExit("tokens and hiddens lists must match")
        _run_rank(known)
    else:
        sys.argv = [sys.argv[0]] + rest
        main()
