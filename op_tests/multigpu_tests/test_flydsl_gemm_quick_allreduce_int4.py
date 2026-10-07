# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness of the serial FlyDSL HGEMM + INT4 QuickReduce.

Each rank multiplies its own bf16 shard, then the op all-reduces. The
reference is an fp32 NCCL all-reduce of that same FlyDSL partial, so the
score is the INT4 reduce and not a second GEMM. A standalone
``flydsl_hgemm`` plus ``QuickAllReduceInt4.allreduce`` must match
``apply`` on the same inputs.

``python3`` this file. It skips when fewer than 8 GPUs are visible or
the arch is not gfx950 (the HGEMM kernel's arch).
"""

from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool, set_start_method

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.distributed as dist

from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm
from aiter.ops.flydsl.gemm_quick_allreduce_int4 import (
    DEFAULT_HGEMM,
    GemmQuickAllReduceInt4,
)

set_start_method("spawn", force=True)

try:
    ARCH = get_gfx_runtime()
except (KeyError, RuntimeError):
    ARCH = None

TP = 8
SQNR_MIN_DB = 18.0
# (256, 512, 128) is a smoke shape. (16384, 7168, 1536) is o_proj at a
# full prefill chunk. dense down_proj is the same M and N with K=4224.
SHAPES = (
    (256, 512, 128),
    (16384, 7168, 1536),
)
WARMUP = 3
ITERS = 10


def _sqnr_db(got: torch.Tensor, reference: torch.Tensor) -> float:
    g = got.float()
    r = reference.float()
    mse = ((g - r) ** 2).mean()
    pow_ = (r * r).mean()
    if float(pow_) <= 0 and float(mse) <= 0:
        return float("inf")
    return float((10.0 * torch.log10(pow_ / mse)).item())


def _median_us(fn) -> float:
    stream = torch.cuda.current_stream()
    samples = []
    for i in range(WARMUP + ITERS):
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        start.record(stream)
        fn()
        end.record(stream)
        end.synchronize()
        if i >= WARMUP:
            samples.append(start.elapsed_time(end) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def _run_rank(rank: int, init_method: str) -> list[dict]:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        world_size=TP,
        rank=rank,
        device_id=device,
    )
    gloo = dist.new_group(backend="gloo")
    fused = GemmQuickAllReduceInt4.from_group(
        gloo, device, rank, world_size=TP, super_tile=8
    )
    rows = []
    try:
        for m, n, k in SHAPES:
            gen_a = torch.Generator(device="cpu").manual_seed(1000 + rank)
            gen_b = torch.Generator(device="cpu").manual_seed(2000 + rank)
            a = (torch.randn(m, k, generator=gen_a) * 0.1).to(device, torch.bfloat16)
            b = (torch.randn(n, k, generator=gen_b) * 0.1).to(device, torch.bfloat16)
            out = torch.empty(m, n, dtype=torch.bfloat16, device=device)
            fused.compile(a, b, out)

            partial = torch.empty_like(out)
            flydsl_hgemm(a, b, out=partial, **DEFAULT_HGEMM)
            ref = partial.float()
            dist.all_reduce(ref)
            standalone = torch.empty_like(out)
            fused.qr.allreduce(partial, standalone)
            fused.apply(a, b, out)
            torch.cuda.synchronize()

            same = torch.tensor(
                [int(torch.equal(out, standalone))],
                device=device,
                dtype=torch.int32,
            )
            dist.all_reduce(same, op=dist.ReduceOp.MIN)
            row = {
                "M": m,
                "N": n,
                "K": k,
                "sqnr_db": round(_sqnr_db(out, ref), 2),
                "matches_standalone": bool(same.item()),
            }
            if rank == 0:
                partial_s = torch.empty_like(out)
                alone_s = torch.empty_like(out)

                def _gemm(a=a, b=b, partial_s=partial_s):
                    flydsl_hgemm(a, b, out=partial_s, **DEFAULT_HGEMM)

                def _qr(partial_s=partial_s, alone_s=alone_s):
                    fused.qr.allreduce(partial_s, alone_s)

                def _serial():
                    _gemm()
                    _qr()

                def _fused(a=a, b=b, out=out):
                    fused.apply(a, b, out)

                row["gemm_us"] = round(_median_us(_gemm), 1)
                row["qr_us"] = round(_median_us(_qr), 1)
                row["serial_us"] = round(_median_us(_serial), 1)
                row["fused_us"] = round(_median_us(_fused), 1)
            rows.append(row)
            dist.barrier()
    finally:
        fused.close()
        dist.destroy_process_group()
    return rows


def main() -> None:
    if ARCH != "gfx950":
        print(f"GEMM_QR skip arch={ARCH}")
        return
    if torch.cuda.device_count() < TP:
        print(f"GEMM_QR skip gpus={torch.cuda.device_count()}")
        return
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    pool = Pool(processes=TP)
    try:
        futs = [
            pool.apply_async(_run_rank, kwds={"rank": rank, "init_method": init_method})
            for rank in range(TP)
        ]
        ranks = [fut.get(timeout=1800) for fut in futs]
    except Exception:
        pool.terminate()
        raise
    else:
        pool.close()
    finally:
        pool.join()

    failed = False
    for row in ranks[0]:
        key = (row["M"], row["N"], row["K"])
        sqnr = min(ranks[r][SHAPES.index(key)]["sqnr_db"] for r in range(TP))
        match = all(
            ranks[r][SHAPES.index(key)]["matches_standalone"] for r in range(TP)
        )
        payload = dict(row)
        payload["sqnr_db"] = sqnr
        payload["matches_standalone"] = match
        payload["ok"] = match and sqnr >= SQNR_MIN_DB
        failed |= not payload["ok"]
        print("GEMM_QR " + json.dumps(payload), flush=True)
    if failed:
        raise SystemExit("GemmQuickAllReduceInt4 failed SQNR or standalone match")


if __name__ == "__main__":
    main()
