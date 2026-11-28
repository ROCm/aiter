# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused AllReduce + RMSNorm + MXFP8 (per-group mx_scale mode).

`CustomAllreduce.custom_fused_ar_rms_per_group_quant(..., group_size=32,
mx_scale=True)` writes E4M3 values with one E8M0 scale per 32 elements. On
every rank and for every kernel (1-stage, 2-stage, split), eager and replayed
from a CUDA graph, this checks:

- values and scales are, byte for byte, what aiter's own MX quantizer
  (`per_1x32_mx_quant_hip`, fp8 with e8m0 scales) makes of the kernel's bf16
  output (`emit_bf16`), and dequantizing them lands within half an E4M3 step
  of that output;
- the residual equals the unquantized fused AR + RMSNorm's;
- the bf16 output equals the unquantized fused AR + RMSNorm's: bit for bit
  where both take the 1-stage kernel; past that the unquantized path runs
  fusedAllReduceRMSNorm2Stage / the split kernel, whose sum of squares is in
  another order, so up to 1e-4 of the elements may differ, each by at most one
  bf16 ulp.

    python test_fused_ar_rms_mxfp8.py [-t 4]
"""

import argparse
import os
from multiprocessing import Pool, set_start_method

import torch
import torch.distributed as dist

set_start_method("spawn", force=True)

HIDDEN = 6144
EPS = 1e-6
# (rows, kernel the per-group path takes at TP4: 1 = 1-stage, 2 = 2-stage, s = split)
CASES = [(1, "1"), (16, "1"), (37, "1"), (38, "2"), (42, "2"), (64, "s"), (1355, "s")]
# Past the 1-stage kernel: share of bf16 elements allowed to differ from the
# unquantized path (measured at TP4: 1 of 38 x 6144, 52 of 1355 x 6144).
MAX_DIFF_FRACTION = 1e-4


def mxfp8_reference(x):
    """aiter's MXFP8 quant of x ([M, K] bf16): E4M3 bytes, E8M0 bytes."""
    from aiter import dtypes
    from aiter.ops.quant import per_1x32_mx_quant_hip

    q, s = per_1x32_mx_quant_hip(x, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0)
    return q.view(torch.uint8), s.view(torch.uint8)[:, : x.shape[1] // 32]


def dequant_ok(q, s, x):
    """Values times 2^(scale - 127) within half an E4M3 step (2^-4 relative to
    the scale's top binade) of x: catches a wrong scale convention that the
    byte comparison would share with the reference."""
    from aiter import dtypes

    m, k = x.shape
    scale = torch.exp2(s.view(torch.uint8).float() - 127.0).view(m, k // 32, 1)
    deq = q.view(dtypes.fp8).float().view(m, k // 32, 32) * scale
    fp8_max = torch.finfo(dtypes.fp8).max
    tol = scale * fp8_max * 2.0**-4
    return bool(((deq - x.float().view(m, k // 32, 32)).abs() <= tol).all())


def bf16_close(n, ref):
    """Elements differing by at most one bf16 ulp, on at most
    MAX_DIFF_FRACTION of them."""
    d = (n.float() - ref.float()).abs()
    one_ulp = ref.float().abs() * 2.0**-7 + torch.finfo(torch.float32).tiny
    ndiff = int((n != ref).sum())
    return bool((d <= one_ulp).all()) and ndiff <= MAX_DIFF_FRACTION * n.numel()


def run_case(ca, rank, rows, kernel):
    """One row count on this rank: (rows, kernel, eager ok, graph ok, number of
    bf16 elements differing from the unquantized fused AR + RMSNorm)."""
    from aiter.dist.communication_op import (
        tensor_model_parallel_fused_allreduce_rmsnorm,
    )
    from aiter.dist.parallel_state import graph_capture

    g = torch.Generator(device="cuda").manual_seed(1000 * rows + rank)
    x = (torch.randn(rows, HIDDEN, device="cuda", generator=g) * 3).bfloat16()
    res = torch.randn(rows, HIDDEN, device="cuda", generator=g).bfloat16()
    w_gen = torch.Generator("cuda").manual_seed(7)
    w = (torch.randn(HIDDEN, device="cuda", generator=w_gen) * 0.1).bfloat16()
    if rows > 3:
        x[3, :64] = 0
        res[3, :64] = 0  # all-zero groups: the amax floor decides their scale

    ref_norm, ref_res = tensor_model_parallel_fused_allreduce_rmsnorm(
        x, res, w, EPS, gemma_norm=True
    )
    use_1stage = kernel == "1"

    def fused():
        return ca.custom_fused_ar_rms_per_group_quant(
            x,
            res,
            w,
            EPS,
            32,
            use_1stage,
            emit_bf16=True,
            gemma_norm=True,
            mx_scale=True,
        )

    def check(out):
        q, r, s, n = out
        assert s.dtype == torch.float8_e8m0fnu and s.shape == (rows, HIDDEN // 32)
        rq, rs = mxfp8_reference(n)
        ok_quant = (
            torch.equal(q.view(torch.uint8), rq)
            and torch.equal(s.view(torch.uint8), rs)
            and dequant_ok(q, s, n)
        )
        ok_res = torch.equal(r, ref_res)
        ok_norm = torch.equal(n, ref_norm) if use_1stage else bf16_close(n, ref_norm)
        return ok_quant and ok_res and ok_norm, int((n != ref_norm).sum())

    eager, eager_ndiff = check(fused())
    graph_ok = True
    if rows <= 64:
        graph = torch.cuda.CUDAGraph()
        with graph_capture() as gc, torch.cuda.graph(graph, stream=gc.stream):
            out = fused()
        for t in out:
            t.zero_()
        graph.replay()
        torch.cuda.synchronize()
        graph_ok, _ = check(out)
    return rows, kernel, eager, graph_ok, eager_ndiff


def run_rank(rank, tp, init_method):
    from aiter.dist.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        get_tp_group,
        init_distributed_environment,
        set_custom_all_reduce,
    )

    torch.cuda.set_device(rank)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp, rank=rank, distributed_init_method=init_method
    )
    ensure_model_parallel_initialized(tp, 1)
    dist.all_reduce(torch.zeros(1).cuda(), group=get_tp_group().device_group)
    ca = get_tp_group().device_communicator.ca_comm
    try:
        rows_out = [run_case(ca, rank, rows, kernel) for rows, kernel in CASES]
    finally:
        torch.cuda.synchronize()
        get_tp_group().barrier()
        destroy_model_parallel()
        destroy_distributed_environment()
    return rows_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--tp", type=int, default=4)
    args = ap.parse_args()
    from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() == "gfx1250":
        # The per-group fused kernels are old-arch only
        # (CustomAllreduce._require_old_arch_fused_kernels).
        print("SKIP: per-group fused AR + RMSNorm quant is not built for gfx1250")
        return

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    init = get_distributed_init_method(get_ip(), get_open_port())
    with Pool(args.tp) as pool:
        per_rank = [
            r.get()
            for r in [
                pool.apply_async(run_rank, (i, args.tp, init)) for i in range(args.tp)
            ]
        ]
    fails = 0
    print(
        "rows kernel eager graph bf16_elems_differing_from_unquantized(max over ranks)"
    )
    for i, (rows, kernel, *_rest) in enumerate(per_rank[0]):
        eager = all(pr[i][2] for pr in per_rank)
        graph = all(pr[i][3] for pr in per_rank)
        ndiff = max(pr[i][4] for pr in per_rank)
        fails += not (eager and graph)
        print(f"{rows:5d} {kernel:6s} {eager!s:5} {graph!s:5} {ndiff}")
    print("FAILURES", fails)
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
