# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness for the exact one-shot all-reduce.

Three things are checked, and the second and third matter more than the first:

1. The sum is right, against an fp32 reference, at the bf16 rounding floor.
2. The result is **bit-identical on every rank**. The kernel accumulates in a
   fixed rank order for exactly this reason, and an SQNR check cannot see an
   ordering bug -- both answers would be equally "accurate".
3. Repeated back-to-back calls stay correct under deliberate rank skew. The
   inbox is double-buffered by ``colour & 1`` and the safety argument depends on
   a straggler's read of call k finishing before anyone's push for call k+2; a
   quiescent test never exercises that.

Run directly (these are argparse scripts, not pytest):

    HIP_VISIBLE_DEVICES=0,1 python3 -m op_tests.flydsl_tests.test_qr_1stage -tp 2
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HIDDEN = 7168
# Shapes chosen to straddle the interesting boundaries: a single 4 KiB tile,
# a partial last tile, an exact multiple, and enough tiles to force several
# per block at a small grid cap.
SHAPES = [1, 2, 3, 5, 8, 11, 16]

# The single-block corner, which HIDDEN=7168 cannot reach: at atoms=1 the tile
# is 4096 B, so even m=1 there is 4 tiles and 4 blocks. This protocol has failed
# at exactly one block before ("5.3 seconds at 1 block, invisible at 448",
# qr_int4_kernel.py), so the release/invalidate pairing needs its own case.
# (rows, hidden) pairs sized in bf16 against a 4096 B tile: one exact tile, one
# partial tile, and one that is two tiles so the block loops twice.
NARROW_SHAPES = [(1, 2048), (1, 1024), (1, 3072), (2, 2048)]


def _sqnr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    ref = ref.double()
    err = ref - got.double()
    p = (ref * ref).mean().item()
    e = (err * err).mean().item()
    if e == 0:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(p / e)).item()


def _worker(
    rank: int,
    world_size: int,
    init_method: str,
    atoms: int,
    grid_cap: int,
    fanout: str,
):
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="gloo", init_method=init_method, rank=rank, world_size=world_size
        )
        from aiter.ops.flydsl.kernels.qr_1stage import OneShotAllReduce

        device = torch.device(f"cuda:{rank}")
        eng = OneShotAllReduce(
            group=dist.group.WORLD,
            device=device,
            rank=rank,
            world_size=world_size,
            atoms=atoms,
            grid_cap=grid_cap,
            fanout=fanout,
            # MAX_PAYLOAD_BYTES is a speed policy, not a correctness limit --
            # the kernel is exact at every size -- so it must not decide what
            # this test covers. Lifted so the shape list stays free to include
            # sizes production would route elsewhere.
            max_bytes=1 << 30,
        )

        warm = torch.zeros(1, HIDDEN, dtype=torch.bfloat16, device=device)
        eng.compile(warm, torch.empty_like(warm))

        def _check(m, hidden, tag):
            """One shape: accuracy against fp32, then bit-identity across ranks."""
            bad = []
            torch.manual_seed(1234 + m * 8191 + hidden)
            # Same seed on every rank, then a per-rank shift, so the reference
            # can be computed locally without another collective.
            parts = [
                torch.randn(m, hidden, dtype=torch.bfloat16, device=device) * (r + 1)
                for r in range(world_size)
            ]
            inp = parts[rank].contiguous()
            out = torch.empty_like(inp)
            eng.allreduce(inp, out)
            torch.cuda.synchronize()

            ref = torch.zeros(m, hidden, dtype=torch.float32, device=device)
            for p in parts:
                ref += p.float()

            db = _sqnr_db(ref, out)
            if db < 45.0:
                bad.append(f"{tag}: SQNR {db:.2f} dB below the 45 dB bf16 floor")

            # Bit-identity across ranks: gather the raw bits, compare exactly.
            # Widened to int32 because gloo rejects int16 ("Invalid scalar
            # type"); the widening is exact, so the comparison is still on bits.
            bits = out.view(torch.int16).to(torch.int32).cpu()
            gathered = [torch.empty_like(bits) for _ in range(world_size)]
            dist.all_gather(gathered, bits)
            for r, g in enumerate(gathered):
                if not torch.equal(g, gathered[0]):
                    n = int((g != gathered[0]).sum())
                    bad.append(
                        f"{tag}: rank {r} differs from rank 0 in {n} bf16 lanes "
                        "(accumulation order is not rank-stable)"
                    )
                    break
            return bad

        failures = []
        for m in SHAPES:
            failures += _check(m, HIDDEN, f"m={m}")
        # Single-block corner. Reported with the block count so a failure names
        # the regime rather than just the shape.
        for m, hidden in NARROW_SHAPES:
            nb = eng._grid_x(eng._num_tiles(m * hidden * 2))
            failures += _check(m, hidden, f"{m}x{hidden} ({nb} block(s))")

        # Run-ahead: many back-to-back calls with one rank deliberately late.
        m = 5
        torch.manual_seed(99)
        parts = [
            torch.randn(m, HIDDEN, dtype=torch.bfloat16, device=device) * (r + 1)
            for r in range(world_size)
        ]
        inp = parts[rank].contiguous()
        out = torch.empty_like(inp)
        ref = torch.zeros(m, HIDDEN, dtype=torch.float32, device=device)
        for p in parts:
            ref += p.float()
        drag = torch.randn(4096, 4096, device=device, dtype=torch.float32)
        bad = 0
        for it in range(200):
            # Rank 0 does unrelated work first, so it enters each call late and
            # the others get a chance to run ahead into the other parity slot.
            if rank == 0 and it % 3 == 0:
                for _ in range(3):
                    drag = drag @ drag.T * 1e-6
            eng.allreduce(inp, out)
            if it % 25 == 0:
                torch.cuda.synchronize()
                if _sqnr_db(ref, out) < 45.0:
                    bad += 1
        torch.cuda.synchronize()
        if _sqnr_db(ref, out) < 45.0 or bad:
            failures.append(f"run-ahead loop: {bad} bad checks of 8")

        eng.close()
        if failures:
            print(f"[rank {rank}] FAIL\n  " + "\n  ".join(failures), flush=True)
        else:
            print(f"[rank {rank}] PASS", flush=True)
        dist.barrier()
        dist.destroy_process_group()
        sys.exit(1 if failures else 0)
    except Exception:
        print(f"[rank {rank}] EXCEPTION\n{traceback.format_exc()}", flush=True)
        sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-tp", type=int, default=2)
    ap.add_argument("--atoms", type=int, default=1)
    ap.add_argument("--grid-cap", type=int, default=64)
    ap.add_argument("--fanout", default="peer", choices=("peer", "atom"))
    args = ap.parse_args()

    n = torch.cuda.device_count()
    if n < args.tp:
        raise SystemExit(f"need {args.tp} GPUs, saw {n}")

    port = int(os.environ.get("AR_TEST_PORT", "29571"))
    init_method = f"tcp://127.0.0.1:{port}"
    mp.set_start_method("spawn", force=True)
    procs = []
    for r in range(args.tp):
        p = mp.Process(
            target=_worker,
            args=(
                r,
                args.tp,
                init_method,
                args.atoms,
                args.grid_cap,
                args.fanout,
            ),
        )
        p.start()
        procs.append(p)
    codes = []
    for p in procs:
        p.join()
        codes.append(p.exitcode)
    print(f"exit codes: {codes}")
    raise SystemExit(0 if all(c == 0 for c in codes) else 1)


if __name__ == "__main__":
    main()
