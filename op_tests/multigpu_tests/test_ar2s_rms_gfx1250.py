#!/usr/bin/env python3
"""TP4 test + benchmark of the gfx1250 pushed two-shot AR + RMSNorm (ar2s_rms_ex), hidden 6144, Gemma norm.

Modes against the bf16 path (ar2s AR + residual add + RMSNorm, then per_1x32_mx_quant_hip for the
MXFP8 consumers):
  fp8      ar2s writes MXFP8 itself                       -> q, scale, residual_out bit-exact
  sp       sequence-parallel residual, bf16 out           -> out bit-exact on all rows, residual_out
                                                             bit-exact on own rows; non-own residual
                                                             rows are poisoned with NaN to prove they
                                                             are never read
  sp_fp8   both                                            -> q, scale bit-exact, own residual rows
  aux      all_reduce + residual from an SP residual      -> the whole residual_out bit-exact
  api      the public tensor_model_parallel_sp_* ops       -> as above at prefill sizes, the replicated
                                                             ops below the ar2s range
Timing: median over reps of back-to-back eager calls (rank max), and the same in a CUDA graph.

usage (4 GPUs): python3 test_ar2s_rms_gfx1250.py \
    --tokens 4,100,2752,6144,14336,16384,24576,32768 [--check] [--bench] [--graph]
--smoke: one call per mode at 4 rows (R=1 -> a single workgroup) and exit.
"""

import argparse
import os
import statistics

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

H = 6144
TP = 4
EPS = 1e-6


def worker(rank, port, args, q):
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
    group = tp.device_group
    ca = tp.device_communicator.ca_comm
    assert ca is not None and getattr(ca, "_ar2s_ctx", 0), "ar2s is not set up"
    dist.all_reduce(torch.zeros(1, device="cuda"), group=group)
    torch.manual_seed(1234)
    w = (torch.randn(H, device="cuda") * 0.1).to(torch.bfloat16)

    def base_bf16(x, res):
        return ca.ar2s_fused_ar_rms(x, res, w, EPS, True)

    def base_fp8(x, res):
        out, res_out = ca.ar2s_fused_ar_rms(x, res, w, EPS, True)
        qq, ss = per_1x32_mx_quant_hip(out, quant_dtype=fp8, scale_type=fp8_e8m0)
        return qq, res_out, ss

    from aiter.dist.communication_op import (
        tensor_model_parallel_all_reduce,
        tensor_model_parallel_fused_allreduce_rmsnorm,
        tensor_model_parallel_fused_allreduce_rmsnorm_mxfp8_quant,
    )

    modes = {
        # the model's calls through the (patched) dispatch: ar2s up to 288 MiB, RCCL + norm above
        "model_bf16": lambda x, res: tensor_model_parallel_fused_allreduce_rmsnorm(
            x, res, w, EPS, gemma_norm=True
        ),
        "model_fp8": lambda x, res: tensor_model_parallel_fused_allreduce_rmsnorm_mxfp8_quant(
            x, res, w, EPS, gemma_norm=True
        ),
        "base_bf16": base_bf16,
        "base_fp8": base_fp8,
        "fp8": lambda x, res: ca.ar2s_fused_ar_rms_ex(
            x, res, w, EPS, True, fp8_out=True
        ),
        "sp": lambda x, res: ca.ar2s_fused_ar_rms_ex(
            x, res, w, EPS, True, sp_residual=True
        ),
        "sp_fp8": lambda x, res: ca.ar2s_fused_ar_rms_ex(
            x, res, w, EPS, True, fp8_out=True, sp_residual=True
        ),
        "aux": lambda x, res: ca.ar2s_fused_ar_rms_ex(
            x, res, w, EPS, True, sp_residual=True, aux_residual=True
        ),
        "rccl_add": lambda x, res: tensor_model_parallel_all_reduce(x) + res,
    }

    def time_eager(fn, iters, reps=7):
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

    def time_graph(fn, iters, reps=7):
        fn()
        torch.cuda.synchronize()
        dist.barrier(group=group)
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.cuda.graph(g, stream=s):
                for _ in range(iters):
                    fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            dist.barrier(group=group)
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            a.record()
            g.replay()
            b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b) * 1e3 / iters)
        t = torch.tensor([statistics.median(ts)], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
        return t.item()

    names = args.modes.split(",")
    results = []
    for n in [int(t) for t in args.tokens.split(",")]:
        g = torch.Generator(device="cuda").manual_seed(1000 + rank * 7 + n)
        x = (torch.randn(n, H, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
        gr = torch.Generator(device="cuda").manual_seed(99 + n)
        res = torch.randn(n, H, device="cuda", generator=gr).to(torch.bfloat16)
        row = {"tokens": n, "launches": len(ca.ar2s_launch_rows(n, H))}
        if args.check or args.smoke:
            own = ca.ar2s_sp_own_rows(n, H)
            mask = torch.zeros(n, dtype=torch.bool, device="cuda")
            for a, b in own:
                mask[a:b] = True
            res_sp = res.clone()
            res_sp[~mask] = float("nan")  # never read under sp
            ref_out, ref_res = base_bf16(x, res)
            ref_q, _, ref_s = base_fp8(x, res)
            torch.cuda.synchronize()
            chk = {}
            q8, r8, s8 = modes["fp8"](x, res)
            chk["fp8"] = (
                torch.equal(q8.view(torch.uint8), ref_q.view(torch.uint8))
                and torch.equal(s8.view(torch.uint8), ref_s.view(torch.uint8))
                and torch.equal(r8, ref_res)
            )
            o_sp, r_sp = modes["sp"](x, res_sp)
            chk["sp"] = torch.equal(o_sp, ref_out) and torch.equal(
                r_sp[mask], ref_res[mask]
            )
            q_sp, r_sp8, s_sp = modes["sp_fp8"](x, res_sp)
            chk["sp_fp8"] = (
                torch.equal(q_sp.view(torch.uint8), ref_q.view(torch.uint8))
                and torch.equal(s_sp.view(torch.uint8), ref_s.view(torch.uint8))
                and torch.equal(r_sp8[mask], ref_res[mask])
            )
            # aux: the whole residual_out from a sequence-parallel residual (non-own rows poisoned)
            r_aux = ca.ar2s_fused_ar_rms_ex(
                x, res_sp, w, EPS, True, sp_residual=True, aux_residual=True
            )
            chk["aux"] = torch.equal(r_aux, ref_res)
            # the public ops (communication_op): sp at prefill sizes, the replicated ops below
            from aiter.dist.communication_op import (
                tensor_model_parallel_sp_allreduce_add_residual,
                tensor_model_parallel_sp_fused_allreduce_rmsnorm,
                tensor_model_parallel_sp_fused_allreduce_rmsnorm_mxfp8_quant,
            )

            eligible = n * H * 2 > (32 << 20)
            rin = res_sp if eligible else res
            o_api, r_api = tensor_model_parallel_sp_fused_allreduce_rmsnorm(
                x, rin, w, EPS, gemma_norm=True
            )
            q_api, r_api8, s_api = (
                tensor_model_parallel_sp_fused_allreduce_rmsnorm_mxfp8_quant(
                    x, rin, w, EPS, gemma_norm=True
                )
            )
            a_api = tensor_model_parallel_sp_allreduce_add_residual(x, rin, w)
            rows_ok = mask if eligible else torch.ones_like(mask)
            if eligible:
                chk["api"] = (
                    torch.equal(o_api, ref_out)
                    and torch.equal(r_api[rows_ok], ref_res[rows_ok])
                    and torch.equal(q_api.view(torch.uint8), ref_q.view(torch.uint8))
                    and torch.equal(s_api.view(torch.uint8), ref_s.view(torch.uint8))
                    and torch.equal(a_api, ref_res)
                )
            else:  # the replicated ops: whole residual; compare against themselves called directly
                from aiter.dist.communication_op import (
                    tensor_model_parallel_fused_allreduce_rmsnorm,
                    tensor_model_parallel_fused_allreduce_rmsnorm_mxfp8_quant,
                    tensor_model_parallel_all_reduce,
                )

                o2, r2 = tensor_model_parallel_fused_allreduce_rmsnorm(
                    x, res, w, EPS, gemma_norm=True
                )
                q2, r28, s2 = tensor_model_parallel_fused_allreduce_rmsnorm_mxfp8_quant(
                    x, res, w, EPS, gemma_norm=True
                )
                a2 = tensor_model_parallel_all_reduce(x) + res
                chk["api"] = (
                    torch.equal(o_api, o2)
                    and torch.equal(r_api, r2)
                    and torch.equal(q_api.view(torch.uint8), q2.view(torch.uint8))
                    and torch.equal(s_api.view(torch.uint8), s2.view(torch.uint8))
                    and torch.equal(a_api, a2)
                )
            # every rank must hold the same out
            o0 = q_sp.view(torch.uint8).clone()
            dist.broadcast(o0, 0, group=group)
            chk["rank_consistent"] = torch.equal(o0, q_sp.view(torch.uint8))
            torch.cuda.synchronize()
            flags = torch.tensor([int(v) for v in chk.values()], device="cuda")
            dist.all_reduce(flags, op=dist.ReduceOp.MIN, group=group)
            row["check"] = dict(zip(chk.keys(), [bool(v) for v in flags.tolist()]))
            row["own_rows"] = own
        if args.bench and not args.smoke:
            nbytes = n * H * 2
            iters = max(3, min(50, int(2e8 // max(nbytes, 1))))
            for name in names:
                row[name] = round(time_eager(lambda: modes[name](x, res), iters), 1)
                if args.graph:
                    row[name + "_g"] = round(
                        time_graph(lambda: modes[name](x, res), min(iters, 10)), 1
                    )
        results.append(row)
        if rank == 0:
            print(row, flush=True)
    if rank == 0:
        q.put(results)
    dist.barrier(group=group)
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="4,2752,6144,12288,14336,16384")
    ap.add_argument("--modes", default="base_bf16,base_fp8,fp8,sp,sp_fp8")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.tokens = "4"
    mp.set_start_method("spawn", force=True)
    q = mp.Queue()
    port = 29000 + os.getpid() % 1000
    ps = [mp.Process(target=worker, args=(r, port, a, q)) for r in range(TP)]
    for p in ps:
        p.start()
    res = q.get()
    for p in ps:
        p.join()
    print("SUMMARY")
    for r in res:
        print(r)


if __name__ == "__main__":
    main()
