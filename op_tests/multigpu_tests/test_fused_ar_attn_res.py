# SPDX-License-Identifier: MIT
"""Correctness of AllReduceAttnRes (aiter/ops/flydsl/fused_ar_attn_res.py) on one node.

Both branches: the fused kernel (M <= FUSED_MAX_TOKENS) and the all-reduce + attn_res_gate above it. Every
call gets new inputs, so a kernel that reads a peer's data from an earlier call fails; the same holds for
CUDA-graph replays. The reference reduces in fp32 over the ranks, rounds the sum and the prefix to bf16 as
the op does, and runs the read in fp64. Fused calls: the prefix bit for bit, the read within 0.03 of the
reference. Calls above FUSED_MAX_TOKENS: bit for bit what custom_all_reduce + attn_res_gate give when called
directly; their own distance to the fp64 reference (up to about 0.04 here) is printed, not judged.

    python op_tests/multigpu_tests/test_fused_ar_attn_res.py --tp 8
"""

import argparse

import torch
import torch.distributed as dist
from torch.multiprocessing import Pool, set_start_method

H = 7168
NB = 9
TOL = 3e-2
CASES_M = (1, 3, 8, 28, 29, 32)
CASES_NVB = (1, 5, 8)
GRAPH_M = (8, 32)


def _reference(parts_sum, residual, bank, nvb, cw, ow, eps=1e-6):
    prefix = (residual.float() + parts_sum.bfloat16().float()).bfloat16()
    cand = torch.cat([bank[:, :nvb], prefix[:, None]], 1).double()
    dot = (cand * cw.double()).sum(-1)
    score = dot / torch.sqrt((cand * cand).sum(-1) / H + eps)
    w = torch.softmax(score, -1)
    mix = (w[..., None] * cand).sum(1)
    out = mix / torch.sqrt((mix * mix).sum(-1, keepdim=True) / H + eps) * ow.double()
    return out, prefix


def _worker(rank, world, init_method):
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "gloo", init_method=init_method, world_size=world, rank=rank
    )
    group = dist.group.WORLD
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
    from aiter.ops.flydsl.fused_ar_attn_res import FUSED_MAX_TOKENS, AllReduceAttnRes

    ca = CustomAllreduce(group, dev)
    op = AllReduceAttnRes(
        group=group,
        device=dev,
        rank=rank,
        world_size=world,
        hidden=H,
        all_reduce=ca.custom_all_reduce,
    )
    gen = torch.Generator().manual_seed(1000 + rank)

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen) * scale).to(dev)

    cw = rnd(H, scale=0.05).float()
    ow = (1.0 + 0.1 * rnd(H)).float()

    def fresh(m, inp, residual, bank):
        inp.copy_(rnd(m, H, scale=2.0))
        residual.copy_(rnd(m, H, scale=4.0))
        bank.copy_(rnd(m, NB, H, scale=4.0))
        parts = inp.float().cpu()
        dist.all_reduce(parts, op=dist.ReduceOp.SUM, group=group)
        return parts.to(dev)

    from aiter.ops.triton.fusions.attn_res import attn_res_gate

    def check(tag, got, parts_sum, inp, residual, bank, nvb):
        ref_out, ref_prefix = _reference(parts_sum, residual, bank, nvb, cw, ow)
        e_out = (got[0].double() - ref_out).abs().max().item()
        exact = torch.equal(got[1], ref_prefix)
        if op.fuses(inp.shape[0], nvb):
            return (tag, e_out, exact, e_out <= TOL and exact)
        direct = attn_res_gate(
            residual,
            bank[:, :nvb],
            cw,
            1e-6,
            add_hidden=ca.custom_all_reduce(inp),
            output_rms_weight=ow,
            output_rms_eps=1e-6,
        )
        same = torch.equal(got[0], direct[0]) and torch.equal(got[1], direct[1])
        tag += " (vs direct call: " + ("bit-exact" if same else "DIFFERS") + ")"
        return (tag, e_out, exact, same and exact)

    rows = []
    for m in CASES_M:
        for nvb in CASES_NVB:
            inp = torch.empty(m, H, dtype=torch.bfloat16, device=dev)
            residual = torch.empty_like(inp)
            bank = torch.empty(m, NB, H, dtype=torch.bfloat16, device=dev)
            branch = "fused" if op.fuses(m, nvb) else "ar+gate"
            for call in range(3):
                s = fresh(m, inp, residual, bank)
                got = op(inp, residual, bank, nvb, cw, ow)
                torch.cuda.synchronize()
                rows.append(
                    check(
                        f"M={m} nvb={nvb} {branch} eager call {call}",
                        got,
                        s,
                        inp,
                        residual,
                        bank,
                        nvb,
                    )
                )
    for m in GRAPH_M:
        nvb = 8
        inp = torch.empty(m, H, dtype=torch.bfloat16, device=dev)
        residual = torch.empty_like(inp)
        bank = torch.empty(m, NB, H, dtype=torch.bfloat16, device=dev)
        fresh(m, inp, residual, bank)
        op(inp, residual, bank, nvb, cw, ow)
        torch.cuda.synchronize()
        dist.barrier(group=group)
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with ca.capture(), torch.cuda.graph(graph, stream=stream):
            got = op(inp, residual, bank, nvb, cw, ow)
        torch.cuda.current_stream().wait_stream(stream)
        branch = "fused" if op.fuses(m, nvb) else "ar+gate"
        for replay in range(3):
            s = fresh(m, inp, residual, bank)
            got[0].fill_(float("nan"))
            torch.cuda.synchronize()
            dist.barrier(group=group)
            graph.replay()
            torch.cuda.synchronize()
            rows.append(
                check(
                    f"M={m} nvb={nvb} {branch} graph replay {replay}",
                    got,
                    s,
                    inp,
                    residual,
                    bank,
                    nvb,
                )
            )
    dist.barrier(group=group)
    op.close()
    return (rows, FUSED_MAX_TOKENS) if rank == 0 else None


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--tp", type=int, default=8)
    args = parser.parse_args()
    set_start_method("spawn", force=True)
    from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with Pool(processes=args.tp) as pool:
        results = pool.starmap(
            _worker, [(r, args.tp, init_method) for r in range(args.tp)]
        )
    rows, max_fused = next(r for r in results if r is not None)
    print(
        f"| case (FUSED_MAX_TOKENS = {max_fused}, tp {args.tp}) | out max_abs | prefix exact | pass |"
    )
    print("|---|---|---|---|")
    for tag, e_out, exact, ok in rows:
        print(f"| {tag} | {e_out:.4f} | {exact} | {ok} |")
    fails = [r for r in rows if not r[3]]
    print(f"{len(rows) - len(fails)} of {len(rows)} passed")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
