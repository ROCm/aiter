# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Runtime correctness for FlyDSL INT4 quick all-reduce (``QuickAllReduceInt4``).

``python3`` this file first runs untimed codec/shape validity, then an
aiter-op-test ``@benchmark`` / markdown sweep. Every rank is a
``multiprocessing`` spawn worker that builds its own
``QuickAllReduceInt4`` engine, calls ``compile()``, and in the sweep
times ``fly.allreduce`` with ``run_perftest``. The oracle is an untimed
fp32 NCCL all-reduce of the same per-rank inputs. INT4 is lossy, so
validity uses SQNR, a calibrated mismatch ratio, and a per-tile SQNR
floor.

hidden=5120 is the width the kernel was tuned on, not a shape the kernel
requires. QuickAllReduceInt4 runs on gfx942/gfx950 at TP∈{2,4,8}; other
archs skip, and ``main()`` skips a world size when fewer GPUs are visible
than TP.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from multiprocessing import Pool, freeze_support, set_start_method

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest

set_start_method("spawn", force=True)

from aiter.ops.flydsl.kernels.quick_allreduce_int4 import (
    SUPPORTED_WORLDS,
    TILE_BYTES,
    WORLD,
)
from aiter.ops.flydsl.quick_allreduce_int4 import DEFAULT_GRID_CAP

try:
    ARCH = get_gfx_runtime()
except (KeyError, RuntimeError):
    ARCH = None
SUPPORTED_ARCHS = ("gfx942", "gfx950")
SQNR_MIN_DB = 18.0
# A 32 KiB tile the kernel never wrote scores ~0 dB; codec noise stays above 8.
TILE_SQNR_MIN_DB = 8.0
# Calibrated to the INT4 group-16 codec vs fp32 all-reduce, not bit identity.
CLOSE_RTOL = 1e-1
CLOSE_ATOL = 1e-1
CLOSE_ERR_RATIO = 0.5
SUPER_TILE = 8
TP = WORLD
_FILLS = (
    "normal",
    "pos_underflow",
    "neg_underflow",
    "overflow_512",
    "zeros",
)
_CODEC_FILL_CASES = (
    ("pos_underflow", "pos-underflow-2^-8"),
    ("neg_underflow", "neg-underflow-2^-8"),
    ("overflow_512", "overflow-512"),
    ("zeros", "true-zero-scale"),
)
# One spawn per world size so compile happens once. hidden=4096 is a width
# the ST pick was not fitted to; (8, 1024) is smaller than one 32 KiB tile.
_VALIDITY_SHAPES = {
    8: [(8, 1024), (512, 5120), (9216, 4096), (32768, 5120)],
    4: [(512, 5120), (9216, 4096)],
    2: [(512, 5120), (9216, 4096)],
}


def _make_inp(
    tokens: int, hidden: int, fill: str, *, rank: int, device: torch.device
) -> torch.Tensor:
    shape = (tokens, hidden)
    if fill == "normal":
        gen = torch.Generator().manual_seed(1234 + rank)
        return (torch.randn(shape, generator=gen, dtype=torch.float32) * 0.1).to(
            device=device, dtype=torch.bfloat16
        )
    if fill == "pos_underflow":
        val = 2.0**-8
    elif fill == "neg_underflow":
        val = -(2.0**-8)
    elif fill == "overflow_512":
        # Drives the E4M3 scale above its largest exponent, which the encoder
        # has to saturate. Only rank 0 carries the value: if every rank sent
        # 512 the reduced sum would also saturate the INT4 group codec, and
        # the case would fail on codec range rather than on scale encoding.
        val = 512.0 if rank == 0 else 0.0
    elif fill == "zeros":
        val = 0.0
    else:
        raise ValueError(f"unknown fill {fill!r}; expected one of {_FILLS}")
    return torch.full(shape, val, device=device, dtype=torch.bfloat16)


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
    """Worst 32 KiB-tile SQNR, so one unwritten tile cannot be averaged away."""
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


def _run_rank(
    rank: int,
    tp: int,
    init_method: str,
    tokens: list[int],
    hiddens: list[int],
    super_tile: int,
    grid_cap: int,
    fill: str,
    time_it: bool,
) -> list[dict]:
    import torch.distributed as dist

    from aiter.ops.flydsl import QuickAllReduceInt4

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        world_size=tp,
        rank=rank,
        device_id=device,
    )
    # QuickAllReduceInt4 exchanges IPC metadata over a non-NCCL group;
    # NCCL stays for the fp32 reference all-reduce.
    gloo = dist.new_group(backend="gloo")
    group = dist.group.WORLD

    fly = QuickAllReduceInt4(
        group=gloo,
        device=device,
        rank=rank,
        world_size=tp,
        super_tile=super_tile,
        grid_cap=grid_cap,
    )
    # compile() launches every ST binary on this shape and all ranks must pass
    # the same one, so keep the JIT buffer small at the widest hidden size.
    compile_tokens = min(512, max(tokens))
    compile_hidden = max(hiddens)
    compile_inp = torch.empty(
        (compile_tokens, compile_hidden), device=device, dtype=torch.bfloat16
    )
    compile_out = torch.empty_like(compile_inp)
    dist.barrier()
    fly.compile(compile_inp, compile_out)
    dist.barrier()
    del compile_inp, compile_out

    rows = []
    try:
        for ntok, hidden in zip(tokens, hiddens, strict=True):
            inp = _make_inp(ntok, hidden, fill, rank=rank, device=device)
            ref = inp.to(torch.float32)
            dist.all_reduce(ref, group=group)
            dist.barrier()

            out = torch.empty_like(inp)
            fly.allreduce(inp, out)
            got = out.to(torch.float32)
            dist.barrier()

            close_err = checkAllclose(
                ref,
                got,
                rtol=CLOSE_RTOL,
                atol=CLOSE_ATOL,
                tol_err_ratio=CLOSE_ERR_RATIO,
                printLog=False,
                msg=f"quick_allreduce_int4 rank {rank}",
            )
            row = {
                "tokens": ntok,
                "hidden": hidden,
                "grid_cap": grid_cap,
                "sqnr_db": _sqnr_db(got, ref),
                "min_tile_sqnr_db": _min_tile_sqnr_db(got, ref),
                "err": float(close_err),
                "us": None,
            }
            if time_it:
                dist.barrier(group=group)
                torch.cuda.synchronize()

                def _allreduce(eng=fly, src=inp, dst=out):
                    eng.allreduce(src, dst)
                    return dst

                # use_cuda_event is mandatory here: run_perftest's default
                # timer wraps the iterations in torch.profiler, which collects
                # no device rows inside a spawn worker and then fails reducing
                # its empty trace. cuda.Event timing is unaffected.
                _, us = run_perftest(_allreduce, use_cuda_event=True)
                row["us"] = float(us)
            rows.append(row)
            del inp, out, ref, got
            torch.cuda.empty_cache()
    finally:
        fly.close()
        dist.destroy_process_group()
    return rows


def _spawn(
    world_size: int,
    pairs: list[tuple[int, int]],
    *,
    time_it: bool,
    super_tile: int = SUPER_TILE,
    grid_cap: int = DEFAULT_GRID_CAP,
    fill: str = "normal",
) -> list[list[dict]]:
    if world_size not in SUPPORTED_WORLDS:
        raise ValueError(f"unsupported world_size={world_size}")
    n_gpu = torch.cuda.device_count()
    if n_gpu < world_size:
        raise RuntimeError(f"QuickAllReduceInt4 needs {world_size} GPUs, have {n_gpu}")
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    token_list = [t for t, _ in pairs]
    hidden_list = [h for _, h in pairs]
    timeout = float(os.environ.get("FLYDSL_QR_TIMEOUT", "3600"))
    pool = Pool(processes=world_size)
    try:
        results = [
            pool.apply_async(
                _run_rank,
                kwds={
                    "rank": rank,
                    "tp": world_size,
                    "init_method": init_method,
                    "tokens": token_list,
                    "hiddens": hidden_list,
                    "super_tile": super_tile,
                    "grid_cap": grid_cap,
                    "fill": fill,
                    "time_it": time_it,
                },
            )
            for rank in range(world_size)
        ]
        ranks = [fut.get(timeout=timeout) for fut in results]
    except Exception:
        pool.terminate()
        raise
    else:
        pool.close()
    finally:
        pool.join()
    if len(ranks) != world_size:
        raise RuntimeError(
            f"QuickAllReduceInt4 gathered {len(ranks)} ranks, expected {world_size}"
        )
    return ranks


def _assert_validity(
    ranks: list[list[dict]],
    *,
    pairs: list[tuple[int, int]],
    world_size: int,
    label: str,
) -> dict:
    if len(ranks) != world_size:
        raise AssertionError(
            f"{label}: gathered {len(ranks)} ranks, expected {world_size}"
        )
    fails = []
    for rank, rows in enumerate(ranks):
        if len(rows) != len(pairs):
            fails.append(f"rank {rank}: {len(rows)} rows, expected {len(pairs)}")
            continue
        for (tokens, hidden), row in zip(pairs, rows, strict=True):
            where = f"rank {rank} tokens={tokens} hidden={hidden}"
            if row["sqnr_db"] < SQNR_MIN_DB:
                fails.append(f"{where}: SQNR {row['sqnr_db']:.2f} dB < {SQNR_MIN_DB}")
            if row["min_tile_sqnr_db"] < TILE_SQNR_MIN_DB:
                fails.append(
                    f"{where}: min-tile SQNR {row['min_tile_sqnr_db']:.2f} dB "
                    f"< {TILE_SQNR_MIN_DB}"
                )
            if row["err"] >= CLOSE_ERR_RATIO:
                fails.append(
                    f"{where}: checkAllclose err {row['err']:.3f} >= {CLOSE_ERR_RATIO}"
                )
    if fails:
        raise AssertionError(f"{label} tp={world_size}: " + "; ".join(fails))
    return ranks[0][0]


@benchmark()
def test_quick_allreduce_int4(tokens, hidden, dtype, tp, grid_cap=DEFAULT_GRID_CAP):
    ranks = _spawn(tp, [(tokens, hidden)], time_it=True, grid_cap=grid_cap)
    row = _assert_validity(
        ranks,
        pairs=[(tokens, hidden)],
        world_size=tp,
        label="bench",
    )
    nbytes = tokens * hidden * 2
    # (tp - 1) adds per element; codec ALU work is not counted.
    flops = tokens * hidden * (tp - 1)
    rank_us = [r[0]["us"] for r in ranks]
    us = max(rank_us)
    return {
        "gfx": ARCH,
        "tp": tp,
        "grid_cap": row["grid_cap"],
        "flydsl us": us,
        "flydsl TFLOPS": (flops / us / 1e6) if us else 0.0,
        "flydsl TB/s": (nbytes / us / 1e6) if us else 0.0,
        "flydsl err": max(r[0]["err"] for r in ranks),
        "flydsl sqnr_db": min(r[0]["sqnr_db"] for r in ranks),
    }


test_quick_allreduce_int4.__test__ = False


def main():
    if ARCH not in SUPPORTED_ARCHS:
        aiter.logger.warning("QuickAllReduceInt4 unsupported on %s; skipping", ARCH)
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
        help="Not a QuickAllReduceInt4 dimension; only 1 runs, other values are skipped.",
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
        help="(tokens, hidden) pairs; hidden is free, 5120 is the tuned width.\n"
        "    e.g.: -s 512,5120 9216,5120",
    )
    parser.add_argument(
        "-o",
        "--out",
        default=None,
        help="Optional JSON output path for the sweep rows.",
    )
    parser.add_argument(
        "--grid-cap",
        type=int,
        default=DEFAULT_GRID_CAP,
        help="Persistent-launch block cap; the engine clamps it to the\n"
        "    measured resident workgroups per CU.",
    )
    args = parser.parse_args()

    if n_gpu < 2:
        aiter.logger.warning(
            "QuickAllReduceInt4 needs 2 GPUs for codec fills, have %s; skipping",
            n_gpu,
        )
    else:
        codec_pairs = [(16, 1024)]
        for fill, label in _CODEC_FILL_CASES:
            ranks = _spawn(
                2, codec_pairs, time_it=False, fill=fill, grid_cap=args.grid_cap
            )
            _assert_validity(ranks, pairs=codec_pairs, world_size=2, label=label)

    for tp, pairs in _VALIDITY_SHAPES.items():
        if n_gpu < tp:
            aiter.logger.warning(
                "QuickAllReduceInt4 needs %s GPUs, have %s; skipping validity tp=%s",
                tp,
                n_gpu,
                tp,
            )
            continue
        ranks = _spawn(tp, pairs, time_it=False, grid_cap=args.grid_cap)
        _assert_validity(ranks, pairs=pairs, world_size=tp, label="validity")

    for dtype in args.dtype:
        if dtype != dtypes.bf16:
            aiter.logger.warning(
                "QuickAllReduceInt4 payload is bf16; skipping %s", dtype
            )
            continue
        df = []
        for tp, batch, mnk in itertools.product(args.tp, args.batch, args.mnk):
            if batch != 1:
                continue
            if tp not in SUPPORTED_WORLDS:
                aiter.logger.warning(
                    "QuickAllReduceInt4 unsupported world_size=%s; skipping", tp
                )
                continue
            if n_gpu < tp:
                aiter.logger.warning(
                    "QuickAllReduceInt4 needs %s GPUs, have %s; skipping tp=%s",
                    tp,
                    n_gpu,
                    tp,
                )
                continue
            if not isinstance(mnk, tuple) or len(mnk) != 2:
                raise ValueError(f"-s expects tokens,hidden; got {mnk!r}")
            tokens, hidden = int(mnk[0]), int(mnk[1])
            df.append(
                test_quick_allreduce_int4(
                    tokens, hidden, dtype, tp, grid_cap=args.grid_cap
                )
            )
        if df:
            table = pd.DataFrame(df)
            aiter.logger.info(
                "flydsl quick allreduce INT4 summary (markdown):\n%s",
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
                                "timer": "run_perftest cuda_event",
                            },
                            "rows": df,
                        },
                        fh,
                        indent=2,
                        default=str,
                    )
                aiter.logger.info("wrote %s", args.out)


if __name__ == "__main__":
    freeze_support()
    main()
