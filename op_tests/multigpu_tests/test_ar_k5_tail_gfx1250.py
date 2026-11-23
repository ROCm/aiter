#!/usr/bin/env python3
"""The routed MoE tail deferred into the k5 one-shot AR (aiter moe_defer_tail), 4x gfx1250, hidden 6144.

For each row count T and routing pattern, every rank builds the operands of the MoE's last launch
(GEMM2 rows, topids_to_rows, fp32 route weights, the shared expert's output as addend, routed
scale 2.0 or 1.0) and runs the two entry points the model calls on the MoE output
(tensor_model_parallel_fused_allreduce_rmsnorm_mxfp8_quant: the next layer's input; ..._rmsnorm:
the final norm):
  ref: flydsl_moe_gather_reduce (the production launch) writes the output, then the AR;
  new: the output is left unwritten and registered as a deferred tail; the AR computes it
       (k5 tail kernel where k5 takes the call, the gather-reduce first on any other path).
Compares out / residual_out / scale bytes on every rank, eager and CUDA graph (registration and
AR captured, replayed 3x), then times both (graph of 1 and 10 calls). Patterns: rand (distinct
rows), dup (rows shared between routes), pad (the last rows' routes dropped: row -1, weight 0),
hot (every route of a token on one GEMM2 row region), ties (equal weights), special values
(zeros, -0, bf16 denormals, large, a NaN row on a dropped route). Any k5 poll timeout stops.
"""

import argparse
import os
import statistics
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

H, TP, EPS, TOPK = 6144, 4, 1e-6, 4


def worker(rank, port, a):
    from aiter.dist.communication_op import (
        tensor_model_parallel_fused_allreduce_rmsnorm,
        tensor_model_parallel_fused_allreduce_rmsnorm_mxfp8_quant,
        tensor_model_parallel_sp_fused_allreduce_rmsnorm,
        tensor_model_parallel_sp_fused_allreduce_rmsnorm_mxfp8_quant,
    )
    from aiter.dist.parallel_state import (
        ensure_model_parallel_initialized,
        get_tp_group,
        graph_capture,
        init_distributed_environment,
        set_custom_all_reduce,
    )
    from aiter.ops.flydsl.grouped_moe_gfx1250 import flydsl_moe_gather_reduce
    from aiter.ops.flydsl.moe_deferred_tail import (
        MoeDeferredTail,
        has_deferred_moe_tail,
        moe_defer_tail,
        register_deferred_moe_tail,
    )

    torch.cuda.set_device(rank)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=TP, rank=rank, distributed_init_method=f"tcp://127.0.0.1:{port}"
    )
    ensure_model_parallel_initialized(TP, 1)
    tp = get_tp_group()
    group = tp.device_group
    ca = tp.device_communicator.ca_comm
    assert ca is not None and ca._k5_ctx, "k5 not set up"
    dist.all_reduce(torch.zeros(1, device="cuda"), group=group)
    # which path took each deferred tail: the k5 kernel (counted here) or the materialized launch
    taken = {"k5": 0}
    _k5_tail = ca.k5_fused_ar_rms_tail

    def _counting_k5_tail(*args, **kwargs):
        got = _k5_tail(*args, **kwargs)
        taken["k5"] += got is not None
        return got

    ca.k5_fused_ar_rms_tail = _counting_k5_tail

    def say(*s):
        if rank == 0:
            print(" ".join(str(v) for v in s), flush=True)

    def check_err(where):
        e = torch.tensor([ca.k5_ar_errors()], device="cuda")
        dist.all_reduce(e, group=group)
        if e.item():
            say(f"POLL TIMEOUT at {where}: err={e.item()}; stopping")
            raise SystemExit(3)

    gw = torch.Generator(device="cuda").manual_seed(7)
    nw = (torch.randn(H, device="cuda", generator=gw, dtype=torch.float32) * 0.1).to(
        torch.bfloat16
    )

    def operands(n, pattern):
        """GEMM2 rows (1, R, H), topids_to_rows [n, 4], weights [n, 4], addend [n, H], residual."""
        pid = ["rand", "dup", "pad", "hot", "ties", "special"].index(pattern)
        g = torch.Generator(device="cuda").manual_seed(
            1000 + rank * 7919 + n * 31 + pid * 100003
        )
        R = ((n * TOPK + 64 + 15) // 16) * 16
        rows_bf = (torch.randn(1, R, H, device="cuda", generator=g) * 0.25).to(
            torch.bfloat16
        )
        # row R - 1 is never routed (the pad pattern poisons it)
        perm = (
            torch.randperm(R - 1, device="cuda", generator=g)[: n * TOPK]
            .to(torch.int32)
            .view(n, TOPK)
        )
        w = torch.rand(n, TOPK, device="cuda", generator=g) * 0.5
        if pattern == "dup":
            perm[:, 1] = perm[:, 0]
            if n > 1:
                perm[1::2, 3] = perm[0::2, 2][: perm[1::2, 3].shape[0]]
        elif pattern == "pad":
            npad = max(1, n // 4)
            perm[n - npad :] = -1
            w[n - npad :] = 0
            rows_bf[0, R - 1].fill_(
                float("nan")
            )  # never read: dropped routes read nothing
        elif pattern == "hot":
            base = torch.arange(n, device="cuda", dtype=torch.int32) % 4
            perm = (
                base.view(n, 1) * TOPK
                + torch.arange(TOPK, device="cuda", dtype=torch.int32)
            ).contiguous()
        elif pattern == "ties":
            w[:] = 0.25
        elif pattern == "special":
            rows_bf[0, :, :8] = 0
            rows_bf[0, :, 8:16] = -0.0
            rows_bf[0, :, 16:24] = torch.tensor(1e-39, dtype=torch.float32).to(
                torch.bfloat16
            )  # denormal
            rows_bf[0, :, 24:32] = 3.0e4
            w[0] = torch.tensor(
                [1.0, 0.0, 1e-37, 0.5], device="cuda"
            )  # denormal products
        addend = (torch.randn(n, H, device="cuda", generator=g) * 0.5).to(
            torch.bfloat16
        )
        if pattern == "special":
            addend[:, 8:16] = -0.0
            addend[:, 16:24] = torch.tensor(-1e-39, dtype=torch.float32).to(
                torch.bfloat16
            )
        gr = torch.Generator(device="cuda").manual_seed(99 + n)
        res = torch.randn(n, H, device="cuda", generator=gr).to(torch.bfloat16)
        return rows_bf, perm.contiguous(), w.contiguous(), addend, res

    def bf16_op(x, res):
        o, r = tensor_model_parallel_fused_allreduce_rmsnorm(
            x, res, nw, EPS, gemma_norm=True
        )
        return o, r, None

    def mx_op(x, res):
        o, r, s = tensor_model_parallel_fused_allreduce_rmsnorm_mxfp8_quant(
            x, res, nw, EPS, gemma_norm=True
        )
        return o, r, s

    def sp_mx_op(x, res):
        o, r, sc = tensor_model_parallel_sp_fused_allreduce_rmsnorm_mxfp8_quant(
            x, res, nw, EPS, gemma_norm=True
        )
        return o, r, sc

    def sp_bf16_op(x, res):
        o, r = tensor_model_parallel_sp_fused_allreduce_rmsnorm(
            x, res, nw, EPS, gemma_norm=True
        )
        return o, r, None

    # the model's entry points: the plain ones, and the sequence-parallel ones (ATOM_M3_PREFILL_SP_AR), which a
    # decode-sized call leaves for the plain path
    ops = {"mxfp8": mx_op, "bf16": bf16_op, "sp_mxfp8": sp_mx_op, "sp_bf16": sp_bf16_op}

    def ref_path(op, rows_bf, perm, w, addend, scale, res):
        out = torch.empty((perm.shape[0], H), dtype=torch.bfloat16, device="cuda")
        flydsl_moe_gather_reduce(
            rows_bf, perm, w, out=out, addend=addend, addend_scale=scale
        )
        return op(out, res)

    def new_path(op, rows_bf, perm, w, addend, scale, res):
        out = torch.empty((perm.shape[0], H), dtype=torch.bfloat16, device="cuda")
        with moe_defer_tail() as scope:
            assert register_deferred_moe_tail(
                MoeDeferredTail(out, rows_bf, perm, w, addend, scale, None)
            )
        assert scope.deferred is not None
        got = op(out, res)
        assert not has_deferred_moe_tail(), "the AR did not take the tail"
        return got

    def bits(t):
        return None if t is None else t.contiguous().view(torch.uint8)

    def same(t1, t2):
        if t1 is None or t2 is None:
            return t1 is None and t2 is None
        return torch.equal(bits(t1), bits(t2))

    def all_ok(ok):
        t = torch.tensor([1 if ok else 0], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group)
        return bool(t.item())

    def neq(t1, t2):
        return 0 if t1 is None else int((bits(t1) != bits(t2)).sum().item())

    # 1. smoke: 1 row, each output, the tail kernel once (2 workgroups)
    rows_bf, perm, w, addend, res = operands(1, "rand")
    for name, op in ops.items():
        new_path(op, rows_bf, perm, w, addend, 2.0, res)
        torch.cuda.synchronize()
        check_err(f"smoke {name}")
    say(
        f"SMOKE ok: k5 tail at 1 row through {list(ops)} (k5 tail launches {taken['k5']} of {len(ops)})"
    )

    fails = 0
    if a.check:
        for n in [int(t) for t in a.check_tokens.split(",")]:
            for pattern in a.patterns.split(","):
                for scale in (2.0, 1.0) if pattern in ("rand", "special") else (2.0,):
                    opnds = operands(n, pattern)
                    rows_bf, perm, w, addend, res = opnds
                    for name, op in ops.items():
                        ref = ref_path(op, rows_bf, perm, w, addend, scale, res)
                        got = new_path(op, rows_bf, perm, w, addend, scale, res)
                        torch.cuda.synchronize()
                        check_err(f"check {name} {n} {pattern}")
                        ok = all_ok(
                            same(got[0], ref[0])
                            and same(got[1], ref[1])
                            and same(got[2], ref[2])
                        )
                        msg = (
                            f"CHECK eager T={n} {pattern} scale={scale} {name}: bitexact_all_ranks={ok}"
                            f" out_neq={neq(got[0], ref[0])} res_neq={neq(got[1], ref[1])}"
                        )
                        fails += 0 if ok else 1
                        if a.graph:
                            torch.cuda.synchronize()
                            gph = torch.cuda.CUDAGraph()
                            with graph_capture() as gc, torch.cuda.graph(
                                gph, stream=gc.stream
                            ):
                                gout = new_path(
                                    op, rows_bf, perm, w, addend, scale, res
                                )
                            torch.cuda.synchronize()
                            for _ in range(3):
                                gph.replay()
                            torch.cuda.synchronize()
                            check_err(f"check graph {name} {n} {pattern}")
                            gok = all_ok(
                                same(gout[0], ref[0])
                                and same(gout[1], ref[1])
                                and same(gout[2], ref[2])
                            )
                            msg += f" graph={gok}"
                            fails += 0 if gok else 1
                            del gph
                        if not ok or rank == 0 and a.verbose:
                            say(msg)
            say(
                f"CHECK T={n} done, fails so far {fails}, k5 tail launches so far {taken['k5']}"
            )
        say(f"CHECK SUMMARY fails={fails}")

    def time_fn(fn, iters, reps=7):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            dist.barrier(group=group)
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            s.record()
            for _ in range(iters):
                fn()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) * 1e3 / iters)
        t = torch.tensor([statistics.median(ts)], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
        return t.item()

    def graph_time(fn, gk, iters):
        torch.cuda.synchronize()
        gph = torch.cuda.CUDAGraph()
        with graph_capture() as gc, torch.cuda.graph(gph, stream=gc.stream):
            for _ in range(gk):
                fn()
        torch.cuda.synchronize()
        t = time_fn(gph.replay, max(1, iters // gk)) / gk
        del gph
        return t

    if a.bench:
        say(
            "BENCH (us per call, graph10 / graph1, rank max of medians): gather-reduce + AR vs AR with the tail"
        )
        for n in [int(t) for t in a.tokens.split(",")]:
            rows_bf, perm, w, addend, res = operands(n, "rand")
            row = {}
            for name, op in ops.items():
                for tag, path in (("ref", ref_path), ("tail", new_path)):
                    fn = lambda: path(
                        op, rows_bf, perm, w, addend, 2.0, res
                    )  # noqa: E731
                    row[f"{name}_{tag}"] = (
                        graph_time(fn, 10, 100),
                        graph_time(fn, 1, 40),
                    )
            check_err(f"bench {n}")
            say(
                f"BENCH T={n:4d} "
                + " ".join(f"{k}={v[0]:.2f}/{v[1]:.2f}" for k, v in row.items())
                + f"  delta mxfp8 {row['mxfp8_tail'][0] - row['mxfp8_ref'][0]:+.2f}/"
                + f"{row['mxfp8_tail'][1] - row['mxfp8_ref'][1]:+.2f}"
            )
    dist.barrier(group=group)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--patterns", default="rand,dup,pad,hot,ties,special")
    ap.add_argument(
        "--check-tokens",
        default="1,2,3,4,5,7,8,9,15,16,17,31,32,33,63,64,65,79,80,81,96,127,128,"
        "129,130,160,170,171,192,200,255,256,300",
    )
    ap.add_argument("--tokens", default="8,32,64,80,128,160")
    a = ap.parse_args()
    mp.set_start_method("spawn")
    port = 29500 + os.getpid() % 1000
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(r, port, a)) for r in range(TP)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    sys.exit(max(p.exitcode or 0 for p in procs))
