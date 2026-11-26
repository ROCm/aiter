#!/usr/bin/env python3
"""gfx1250 one-shot decode AR + RMSNorm (k5) check, 4x gfx1250, hidden 6144.

Calls the model entry points (tensor_model_parallel_fused_allreduce_rmsnorm and
..._mxfp8_quant) with k5 off (the production path: Gluon <= 128 rows, Lamport fused (+ quant)
above) and on, in the same process (k5 is switched by the CustomAllreduce context), and
  1. smoke: one call of each output at 1 row,
  2. compares the bits (out, residual_out, scale) on every rank, eager and CUDA graph, and
     checks the ranks agree,
  3. times both (graph of 10 calls and of 1 call, rank max of medians).
Any k5 poll timeout stops the run.
"""

import argparse
import os
import statistics
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

H, TP, EPS = 6144, 4, 1e-6


def worker(rank, port, a):
    from aiter.dist.communication_op import (
        tensor_model_parallel_fused_allreduce_rmsnorm,
        tensor_model_parallel_fused_allreduce_rmsnorm_mxfp8_quant,
    )
    from aiter.dist.parallel_state import (
        ensure_model_parallel_initialized,
        get_tp_group,
        graph_capture,
        init_distributed_environment,
        set_custom_all_reduce,
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
    assert tp.device_communicator.gluon_ar_plain is not None, "Gluon AR not set up"
    k5_ctx = ca._k5_ctx
    dist.all_reduce(torch.zeros(1, device="cuda"), group=group)

    def say(*s):
        if rank == 0:
            print(" ".join(str(v) for v in s), flush=True)

    def k5(on):
        ca._k5_ctx = k5_ctx if on else 0

    def check_err(where):
        e = torch.tensor([ca.k5_ar_errors()], device="cuda")
        dist.all_reduce(e, group=group)
        if e.item():
            say(f"POLL TIMEOUT at {where}: err={e.item()}; stopping")
            raise SystemExit(3)

    gw = torch.Generator(device="cuda").manual_seed(7)
    w = (torch.randn(H, device="cuda", generator=gw, dtype=torch.float32) * 0.1).to(
        torch.bfloat16
    )

    def inputs(n, special):
        g = torch.Generator(device="cuda").manual_seed(1000 + rank * 7 + n)
        x = (
            torch.randn(n, H, device="cuda", generator=g, dtype=torch.float32) * 0.5
        ).to(torch.bfloat16)
        gr = torch.Generator(device="cuda").manual_seed(99 + n)
        res = torch.randn(n, H, device="cuda", generator=gr, dtype=torch.float32).to(
            torch.bfloat16
        )
        if special:
            x[0].zero_()
            res[0].zero_()
            if n > 1:
                x[1] *= 64
            if n > 2:
                x[2] *= 1e-3
            if n > 3:
                res[3] *= 100
        return x, res

    def bf16_op(x, res):
        o, r = tensor_model_parallel_fused_allreduce_rmsnorm(
            x, res, w, EPS, gemma_norm=True
        )
        return o, r, None

    def mx_op(x, res):
        o, r, s = tensor_model_parallel_fused_allreduce_rmsnorm_mxfp8_quant(
            x, res, w, EPS, gemma_norm=True
        )
        return o, r, s

    ops = {"bf16": bf16_op, "mxfp8": mx_op}

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

    def rank_consistent(t):
        t0 = bits(t).clone()
        dist.broadcast(t0, 0, group=group)
        return all_ok(torch.equal(t0, bits(t)))

    # 1. smoke: 1 row, one call of each output (2 workgroups)
    x, res = inputs(1, False)
    k5(True)
    for name, op in ops.items():
        op(x, res)
        check_err(f"smoke {name}")
    say("SMOKE ok: bf16 and mxfp8 at 1 row")

    # 2. bits
    fails = 0
    if a.check:
        for n in [int(t) for t in a.check_tokens.split(",")]:
            for special in (False, True):
                x, res = inputs(n, special)
                for name, op in ops.items():
                    k5(False)
                    ref = op(x, res)
                    k5(True)
                    got = op(x, res)
                    check_err(f"check {name} {n}")
                    ok = all_ok(
                        same(got[0], ref[0])
                        and same(got[1], ref[1])
                        and same(got[2], ref[2])
                    )
                    rc = rank_consistent(got[0]) and rank_consistent(got[1])
                    neq = (bits(got[0]) != bits(ref[0])).sum().item()
                    say(
                        f"CHECK eager rows={n} special={int(special)} {name}: bitexact_all_ranks={ok} "
                        f"out_neq={neq} rank_consistent={rc}"
                    )
                    fails += 0 if ok and rc else 1
                    # graph: k5 captured once, replayed, compared with the reference
                    torch.cuda.synchronize()
                    gph = torch.cuda.CUDAGraph()
                    with graph_capture() as gc, torch.cuda.graph(gph, stream=gc.stream):
                        gout = op(x, res)
                    torch.cuda.synchronize()
                    for _ in range(3):
                        gph.replay()
                    torch.cuda.synchronize()
                    check_err(f"check graph {name} {n}")
                    gok = all_ok(
                        same(gout[0], ref[0])
                        and same(gout[1], ref[1])
                        and same(gout[2], ref[2])
                    )
                    say(
                        f"CHECK graph rows={n} special={int(special)} {name}: bitexact_all_ranks={gok}"
                    )
                    fails += 0 if gok else 1
                    del gph
        say(f"CHECK SUMMARY fails={fails}")

    # 2b. diagnostic: the fused Lamport-order MXFP8 epilogue vs Lamport + per_1x32_mx_quant_hip
    if a.diag:
        diag_fails = 0
        for n in [int(t) for t in a.diag_tokens.split(",")]:
            for special in (False, True):
                x, res = inputs(n, special)
                k5(False)
                ref_o, ref_r, ref_s = mx_op(x, res)
                ref_bf, _, _ = bf16_op(x, res)
                k5(True)
                got_o, got_r, got_s = ca.k5_fused_ar_rms(
                    x, res, w, EPS, True, lamport_order=True, fp8_out=True
                )
                check_err(f"diag {n}")
                so, sr = bits(ref_s), bits(got_s)
                bad = (so != sr).nonzero()
                nb = torch.tensor(
                    [
                        bad.shape[0],
                        (bits(ref_o) != bits(got_o)).sum().item(),
                        (bits(ref_r) != bits(got_r)).sum().item(),
                    ],
                    device="cuda",
                )
                allnb = [torch.zeros_like(nb) for _ in range(TP)]
                dist.all_gather(allnb, nb, group=group)
                say(
                    f"DIAG rows={n} special={int(special)} per rank [scale_neq, out_neq, res_neq]: "
                    + " ".join(str(t.tolist()) for t in allnb)
                )
                diag_fails += sum(int(t.sum().item() != 0) for t in allnb)
                if (
                    special
                ):  # the all-zero row: production's scale byte is 85 (amax floor 1e-10)
                    zero_ok = all_ok(
                        bool((sr[0] == 85).all()) and torch.equal(sr[0], so[0])
                    )
                    say(
                        f"ZERO-GROUP rows={n}: fused MXFP8 scale of the all-zero row == per_1x32_mx_quant_hip's (85): {zero_ok}"
                    )
                    diag_fails += 0 if zero_ok else 1
                if rank == 0 and bad.shape[0]:
                    for r_, g_ in bad[:12].tolist():
                        grp = ref_bf[r_, 32 * g_ : 32 * g_ + 32].float()
                        raw = (
                            ref_bf[r_, 32 * g_ : 32 * g_ + 4]
                            .contiguous()
                            .view(torch.int16)
                            .tolist()
                        )
                        print(
                            f"  row {r_} group {g_}: ref_scale={so[r_, g_].item()} k5_scale={sr[r_, g_].item()} "
                            f"group_amax={grp.abs().max().item():.6g} first4_bf16_bits={[hex(v & 0xffff) for v in raw]} "
                            f"ref_fp8_first4={bits(ref_o)[r_, 32 * g_:32 * g_ + 4].tolist()} "
                            f"k5_fp8_first4={bits(got_o)[r_, 32 * g_:32 * g_ + 4].tolist()}",
                            flush=True,
                        )

        say(f"DIAG SUMMARY fails={diag_fails}")

    # 3. timing
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
        rows_out = []
        for n in [int(t) for t in a.tokens.split(",")]:
            x, res = inputs(n, False)
            row = {"rows": n}
            for name, op in ops.items():
                for on in (False, True):
                    k5(on)
                    tg = graph_time(lambda: op(x, res), 10, 100)
                    t1 = graph_time(lambda: op(x, res), 1, 40)
                    row[f"{name}_{'k5' if on else 'prod'}"] = (tg, t1)
            check_err(f"bench {n}")
            rows_out.append(row)
            say("BENCH", row)
        say("BENCH SUMMARY (us per call, graph10 / graph1, rank max of medians)")
        for row in rows_out:
            say(
                f"rows={row['rows']:4d} "
                + " ".join(
                    f"{k}={v[0]:.2f}/{v[1]:.2f}" for k, v in row.items() if k != "rows"
                )
            )
    k5(True)
    dist.barrier(group=group)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--diag-tokens", default="129,192,256")
    ap.add_argument(
        "--check-tokens", default="1,7,16,64,80,96,120,128,129,160,192,200,255,256"
    )
    ap.add_argument("--tokens", default="16,64,80,96,128,160,192,256")
    a = ap.parse_args()
    os.environ.setdefault("AITER_GLUON_FUSED_AR", "1")
    mp.set_start_method("spawn")
    port = 29500 + os.getpid() % 1000
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(r, port, a)) for r in range(TP)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    sys.exit(max(p.exitcode or 0 for p in procs))
