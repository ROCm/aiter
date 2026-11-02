"""MegaMoEGfx1250 one-kernel combine (+ tail) vs the production two-kernel combine
followed by ATOM's gated add (sigmoid_mul) and aiter's Gemma RMSNorm.

Launch (one rank a GPU): torchrun --standalone --nproc_per_node=4 test_mega_moe_combine_tail.py [--rows 32,512] [--perf]
(run from this directory: it imports bench_mega_moe; Qwen3.5 geometry: E=256 top-8, H=2048, I=512, mori dispatch, fp4 wire).
"""

import argparse
import json
import os

os.environ.setdefault("AITER_FORCE_A8W4", "0")
os.environ.pop("AITER_MEGA_COMBINE_MERGED", None)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from aiter import dtypes  # noqa: E402
from aiter.ops.flydsl.kernels.mega_moe_gfx1250 import MegaMoEGfx1250  # noqa: E402
from aiter.ops.flydsl.kernels.mega_moe_gfx1250.mega_moe import CombineTail  # noqa: E402
from aiter.ops.fused_qk_rmsnorm_group_quant import fused_qk_rmsnorm_group_quant  # noqa: E402
from bench_mega_moe import (  # noqa: E402
    Dist,
    _import_mori_comm,
    make_shared_weights,
    raw_quant_weights,
    resolve_spec,
    shuffle_group,
)

from atom.model_ops.sigmoid_gate import shared_expert_gate_add  # noqa: E402

E, TOPK, H, I = 256, 8, 2048, 512
EPS = 1e-6
DT = torch.bfloat16


def ulps(a, b):
    return (a.view(torch.int16).int() - b.view(torch.int16).int()).abs().max().item()


def graph_us(fns, reps=30):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            for f in fns:
                f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns:
            f()
    dist.barrier()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    dist.barrier()
    best = None
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        dist.barrier()
        a.record()
        for _ in range(reps):
            g.replay()
        b.record()
        b.synchronize()
        t = a.elapsed_time(b) * 1e3 / (reps * len(fns))
        best = t if best is None else min(best, t)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="32,160,512,640")
    ap.add_argument("--perf", action="store_true")
    ap.add_argument("--layers", type=int, default=4)
    args = ap.parse_args()
    d = Dist()
    dev = torch.device("cuda", d.local_rank)
    r, ws = d.rank, d.world
    spec = resolve_spec("a4w4_mxfp4")
    EPR = E // ws
    w1_bf, w2_bf, _, _ = make_shared_weights(E, H, I, dtypes.bf16, dev)
    q1, gs1, q2, gs2 = raw_quant_weights(
        w1_bf[r * EPR : (r + 1) * EPR].contiguous(), w2_bf[r * EPR : (r + 1) * EPR].contiguous()
    )
    w1, w2, w1s, w2s = shuffle_group(q1, gs1, q2, gs2, spec, EPR)
    del w1_bf, w2_bf
    Communicator = _import_mori_comm()
    uid = Communicator.get_unique_id() if r == 0 else None
    comm = Communicator.init(ws, r, d.bcast_uid(uid), per_rank_vmm=8 * 1024**3)
    rows_list = [int(x) for x in args.rows.split(",")]
    mega = MegaMoEGfx1250(
        communicator=comm,
        rank=r,
        world_size=ws,
        model_dim=H,
        inter_dim=I,
        experts=E,
        topk=TOPK,
        max_tokens_per_rank=max(rows_list),
        activation=spec["activation"],
        gate_mode=spec["gate_mode"].value,
        quant_type=spec["aiter_qtype"],
        dispatch_wire="fp4",
        dispatch_backend="mori",
    )
    comm.barrier()
    results = []
    for m in rows_list:
        gen = torch.Generator(device=dev).manual_seed(1000 * r + m)

        def rn(*s, scale=1.0):
            return (torch.randn(*s, generator=gen, device=dev) * scale).to(DT)

        L = args.layers
        lay = []
        for _ in range(L):
            score = torch.rand(m, E, generator=gen, device=dev)
            ids = score.topk(TOPK, dim=-1)[1].to(torch.int32).contiguous()
            wts = torch.rand(m, TOPK, generator=gen, device=dev)
            wts = (wts / wts.sum(-1, keepdim=True)).contiguous()
            logits = rn(m, E + 1, scale=3.0)
            lay.append(
                dict(
                    x=rn(m, H, scale=0.5), ids=ids, wts=wts, shared=rn(m, H, scale=0.5),
                    gate=logits[:, E:], logits=logits, residual=rn(m, H, scale=4.0),
                    w=rn(H, scale=0.1),
                )
            )

        def moe(t, merged=False, tail=False):
            mega._combine_merged = merged
            kw = {}
            if tail:
                kw["combine_tail"] = CombineTail(t["shared"], t["gate"], t["residual"], t["w"], EPS)
            return mega(
                t["x"], t["wts"], t["ids"], w1=w1, w2=w2, w1_scale=w1s, w2_scale=w2s, **kw
            )

        def ref_tail(t, routed):
            g = shared_expert_gate_add(t["shared"], t["gate"], routed, True)
            out, res = torch.empty_like(g), torch.empty_like(g)
            fused_qk_rmsnorm_group_quant(
                q=g, q_weight=t["w"], q_epsilon=EPS, q_out_unquantized=out,
                q_res_out=res, q_residual=t["residual"], gemma_norm=True,
            )
            return out, res

        t = lay[0]

        def step(name):
            torch.cuda.synchronize()
            if os.environ.get("CT_VERBOSE"):
                print(f"[r{r} m{m}] {name}", flush=True)

        routed = moe(t).clone()
        step("two-kernel")
        out_ref, res_ref = ref_tail(t, routed)
        routed_m = moe(t, merged=True).clone()
        step("merged")
        out_f, res_f = moe(t, tail=True)
        step("tail")
        routed_again = moe(t).clone()
        step("two-kernel again")  # the two-kernel path after the one-kernel ones
        torch.cuda.synchronize()
        rec = {
            "rank": r, "ws": ws, "rows": m,
            "merged_routed_bitwise": bool(torch.equal(routed_m, routed)),
            "mixed_order_routed_bitwise": bool(torch.equal(routed_again, routed)),
            "tail_residual_bitwise": bool(torch.equal(res_f, res_ref)),
            "tail_out_max_ulp": ulps(out_f, out_ref),
            "tail_out_max_abs": (out_f.float() - out_ref.float()).abs().max().item(),
            "tail_out_mismatch": (out_f != out_ref).float().mean().item(),
        }
        # graph capture of L layers with the tail, replayed; equal to eager
        eager = [moe(tt, tail=True) for tt in lay]
        eager = [(a.clone(), b.clone()) for a, b in eager]
        holder = {}

        def chain():
            holder["o"] = [moe(tt, tail=True) for tt in lay]

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            chain()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            chain()
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
        rec["graph_tail_equal_eager"] = all(
            torch.equal(a, c) and torch.equal(b, dd) for (a, b), (c, dd) in zip(holder["o"], eager)
        )
        del g
        if args.perf:
            rec["us_two_kernel_plus_tail"] = graph_us(
                [lambda tt=tt: ref_tail(tt, moe(tt)) for tt in lay]
            )
            rec["us_merged_plus_tail"] = graph_us(
                [lambda tt=tt: ref_tail(tt, moe(tt, merged=True)) for tt in lay]
            )
            rec["us_one_kernel_tail"] = graph_us([lambda tt=tt: moe(tt, tail=True) for tt in lay])
        results.append(rec)
        print(json.dumps(rec), flush=True)
    comm.barrier()
    dist.barrier()
    d.shutdown()


if __name__ == "__main__":
    main()
