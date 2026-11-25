#!/usr/bin/env python3
"""ar2s MXFP8 output vs per_1x32_mx_quant_hip of its bf16 output, with an all-zero row, on 4x
gfx1250 (TP4, hidden 6144): the E4M3 and E8M0 bytes must agree, the all-zero row's scale bytes
being 85 (the quantizers' amax floor, aiter::kAmaxFloor), replicated and sequence-parallel.

usage: python3 test_ar2s_mxfp8_zero_group.py
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

H, TP, EPS, ROWS = 6144, 4, 1e-6, 8


def worker(rank, port):
    from aiter.dist.parallel_state import (
        ensure_model_parallel_initialized,
        get_tp_group,
        init_distributed_environment,
        set_custom_all_reduce,
    )
    from aiter.ops.quant import per_1x32_mx_quant_hip
    from aiter.utility.dtypes import fp8, fp8_e8m0

    torch.cuda.set_device(rank)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=TP, rank=rank, distributed_init_method=f"tcp://127.0.0.1:{port}"
    )
    ensure_model_parallel_initialized(TP, 1)
    tp = get_tp_group()
    ca = tp.device_communicator.ca_comm
    assert ca is not None and ca._ar2s_ctx, "ar2s not set up"
    g = torch.Generator(device="cuda").manual_seed(1000 + rank)
    x = (
        torch.randn(ROWS, H, device="cuda", generator=g, dtype=torch.float32) * 0.5
    ).to(torch.bfloat16)
    gr = torch.Generator(device="cuda").manual_seed(99)
    res = torch.randn(ROWS, H, device="cuda", generator=gr, dtype=torch.float32).to(
        torch.bfloat16
    )
    w = (torch.randn(H, device="cuda", generator=gr, dtype=torch.float32) * 0.1).to(
        torch.bfloat16
    )
    x[0].zero_()
    res[0].zero_()
    ok = True
    for sp in (False, True):
        q, _, s = ca.ar2s_fused_ar_rms_ex(
            x, res, w, EPS, True, fp8_out=True, sp_residual=sp
        )
        o, _ = ca.ar2s_fused_ar_rms_ex(x, res, w, EPS, True, sp_residual=sp)
        q_ref, s_ref = per_1x32_mx_quant_hip(o, quant_dtype=fp8, scale_type=fp8_e8m0)
        qb, sb, qrb, srb = (
            t.contiguous().view(torch.uint8) for t in (q, s, q_ref, s_ref)
        )
        this = (
            torch.equal(qb, qrb) and torch.equal(sb, srb) and bool((sb[0] == 85).all())
        )
        if rank == 0:
            print(
                f"ar2s fp8_out sp_residual={sp}: bytes == per_1x32_mx_quant_hip: {this} "
                f"(e4m3 neq {(qb != qrb).sum().item()}, e8m0 neq {(sb != srb).sum().item()}, "
                f"all-zero row scales {sorted(set(sb[0].tolist()))})",
                flush=True,
            )
        ok = ok and this
    t = torch.tensor([1 if ok else 0], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=tp.device_group)
    if rank == 0:
        print(f"AR2S ZERO-GROUP {'PASS' if t.item() else 'FAIL'}", flush=True)
    dist.barrier(group=tp.device_group)
    if not t.item():
        raise SystemExit(1)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    port = 29500 + os.getpid() % 1000
    procs = [
        mp.get_context("spawn").Process(target=worker, args=(r, port))
        for r in range(TP)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    sys.exit(max(p.exitcode or 0 for p in procs))
