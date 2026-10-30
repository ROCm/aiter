"""Fused LL AR + residual + Gemma RMSNorm (+ per-token FP8) against the unfused path.

Reference: LL all-reduce, then rmsnorm2d_fwd_with_add (Gemma), then per-token FP8
quant, exactly what the communicator ran before. Checks eager and CUDA-graph calls.
"""

import sys
from multiprocessing import set_start_method

import torch
import torch.distributed as dist

set_start_method("spawn", force=True)
HIDDEN, EPS = 6144, 1e-6
MS = [1, 4, 12, 16, 32, 64, 128, 256, 341]
import os
DUMP = os.environ.get("LL_TEST_DUMP", "")


def worker(rank, world, init, result_q):
    import aiter
    from aiter import dtypes
    from aiter.dist.communication_op import (
        tensor_model_parallel_all_reduce,
        tensor_model_parallel_fused_allreduce_rmsnorm,
        tensor_model_parallel_fused_allreduce_rmsnorm_quant,
    )
    from aiter.dist.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        get_tp_group,
        graph_capture,
        init_distributed_environment,
        set_custom_all_reduce,
    )

    torch.cuda.set_device(rank)
    set_custom_all_reduce(True)
    init_distributed_environment(world_size=world, rank=rank, distributed_init_method=init)
    ensure_model_parallel_initialized(world, 1)
    dist.all_reduce(torch.zeros(1).cuda(), group=get_tp_group().device_group)
    quant = aiter.get_hip_quant(aiter.QuantType.per_Token)
    gen = torch.Generator(device="cuda").manual_seed(1234)
    w = torch.randn(HIDDEN, dtype=torch.bfloat16, device="cuda", generator=gen) * 0.1
    lines = []
    ok_all = True
    for m in MS:
        torch.manual_seed(100 + rank * 7 + m)
        x = torch.randn(m, HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.5
        torch.manual_seed(999 + m)  # residual is replicated across ranks, as in the model
        r = torch.randn(m, HIDDEN, dtype=torch.bfloat16, device="cuda")

        # reference (unfused)
        ar = tensor_model_parallel_all_reduce(x)
        ref_out = torch.empty_like(ar)
        ref_res = torch.empty_like(ar)
        aiter.rmsnorm2d_fwd_with_add(ref_out, ar, r, ref_res, w, EPS, True, 0)
        ref_q, ref_s = quant(ref_out, quant_dtype=dtypes.fp8)

        for mode in ("eager", "graph"):
            if mode == "eager":
                o, ro = tensor_model_parallel_fused_allreduce_rmsnorm(x, r, w, EPS, gemma_norm=True)
                q, rq, s = tensor_model_parallel_fused_allreduce_rmsnorm_quant(
                    x, r, w, EPS, quant_type="per_token", gemma_norm=True)
            else:
                g = torch.cuda.CUDAGraph()
                with graph_capture() as gc, torch.cuda.graph(g, stream=gc.stream):
                    o, ro = tensor_model_parallel_fused_allreduce_rmsnorm(x, r, w, EPS, gemma_norm=True)
                    q, rq, s = tensor_model_parallel_fused_allreduce_rmsnorm_quant(
                        x, r, w, EPS, quant_type="per_token", gemma_norm=True)
                for _ in range(3):
                    g.replay()
            torch.cuda.synchronize()
            res_eq = torch.equal(ro, ref_res) and torch.equal(rq, ref_res)
            out_diff = (o.float() - ref_out.float()).abs().max().item()
            out_eq = torch.equal(o, ref_out)
            ulp = torch.exp2(torch.floor(torch.log2(ref_out.float().abs().clamp_min(1e-30))) - 7)
            out_ulps = ((o.float() - ref_out.float()).abs() / ulp).max().item()
            out_neq = (o != ref_out).float().mean().item()
            q_mis = (q.view(torch.uint8) != ref_q.view(torch.uint8)).float().mean().item()
            s_diff = ((s.view(-1) - ref_s.view(-1)).abs() / ref_s.view(-1).abs()).max().item()
            deq = (q.float() * s.view(-1, 1) - ref_out.float()).abs().max().item()
            # The fused FP8 must be the standalone quant of the fused bf16 output, bit for bit.
            self_q, self_s = quant(o, quant_dtype=dtypes.fp8)
            self_eq = torch.equal(q.view(torch.uint8), self_q.view(torch.uint8)) and torch.equal(
                s.view(-1), self_s.view(-1))
            ok = res_eq and out_ulps <= 1.0 and q_mis < 0.01 and s_diff < 1e-2 and self_eq
            ok_all &= ok
            if mode == "eager" and DUMP:
                torch.save({"o": o.cpu(), "ro": ro.cpu(), "q": q.view(torch.uint8).cpu(), "s": s.cpu()},
                           f"{DUMP}/r{rank}_m{m}.pt")
            lines.append(
                f"M={m:4d} {mode:5s} residual_bitwise={res_eq} bf16_out_bitwise={out_eq} "
                f"bf16_max_ulp={out_ulps:.0f} bf16_neq={out_neq:.5f} fp8_mismatch={q_mis:.4f} scale_rel={s_diff:.1e} "
                f"dequant_max_abs={deq:.2e} fp8_eq_quant_of_own_bf16={self_eq} {'OK' if ok else 'FAIL'}")
    if rank == 0:
        result_q.put((ok_all, lines))
    torch.cuda.synchronize()
    get_tp_group().barrier()
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

    world = 4
    init = get_distributed_init_method(get_ip(), get_open_port())
    ctx = torch.multiprocessing.get_context("spawn")
    q = ctx.SimpleQueue()
    torch.multiprocessing.spawn(worker, args=(world, init, q), nprocs=world, join=True)
    ok, lines = q.get()
    print("\n".join(lines))
    print("ALL OK" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)
