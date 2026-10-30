"""Prefill-sized fused AR + residual + Gemma RMSNorm (+ per-token FP8), exact reference.

The expected all-reduce is exact: every rank's input all-gathered and summed in rank
order in fp32, then rounded to bf16, as the kernels sum. From it the reference runs
rmsnorm2d_fwd_with_add (Gemma) and the per-token FP8 quant.
"""

import sys
from multiprocessing import set_start_method

import torch
import torch.distributed as dist

set_start_method("spawn", force=True)
HIDDEN, EPS = 6144, 1e-6
MS = [342, 512, 2517, 5000, 8192, 10922]


def worker(rank, world, init, result_q):
    import aiter
    from aiter import dtypes
    from aiter.dist.communication_op import (
        tensor_model_parallel_fused_allreduce_rmsnorm,
        tensor_model_parallel_fused_allreduce_rmsnorm_quant,
    )
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
    init_distributed_environment(world_size=world, rank=rank, distributed_init_method=init)
    ensure_model_parallel_initialized(world, 1)
    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1).cuda(), group=group)
    quant = aiter.get_hip_quant(aiter.QuantType.per_Token)
    gen = torch.Generator(device="cuda").manual_seed(1234)
    w = torch.randn(HIDDEN, dtype=torch.bfloat16, device="cuda", generator=gen) * 0.1
    lines, ok_all = [], True
    for m in MS:
        torch.manual_seed(100 + rank * 7 + m)
        x = torch.randn(m, HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.5
        torch.manual_seed(999 + m)
        r = torch.randn(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
        parts = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(parts, x, group=group)
        acc = torch.zeros(m, HIDDEN, dtype=torch.float32, device="cuda")
        for p in parts:
            acc += p.float()
        ar = acc.to(torch.bfloat16)
        del parts, acc
        ref_out, ref_res = torch.empty_like(ar), torch.empty_like(ar)
        aiter.rmsnorm2d_fwd_with_add(ref_out, ar, r, ref_res, w, EPS, True, 0)

        for _ in range(2):  # the second call reuses the staging of the first
            o, ro = tensor_model_parallel_fused_allreduce_rmsnorm(x, r, w, EPS, gemma_norm=True)
            q, rq, s = tensor_model_parallel_fused_allreduce_rmsnorm_quant(
                x, r, w, EPS, quant_type="per_token", gemma_norm=True)
        torch.cuda.synchronize()
        res_eq = torch.equal(ro, ref_res) and torch.equal(rq, ref_res)
        ulp = torch.exp2(torch.floor(torch.log2(ref_out.float().abs().clamp_min(1e-30))) - 7)
        out_ulps = ((o.float() - ref_out.float()).abs() / ulp).max().item()
        self_q, self_s = quant(o, quant_dtype=dtypes.fp8)
        self_eq = torch.equal(q.view(torch.uint8), self_q.view(torch.uint8)) and torch.equal(
            s.view(-1), self_s.view(-1))
        # every rank must hold the same bits
        dig = torch.tensor([o.view(torch.int16).double().sum().item(),
                            q.view(torch.uint8).double().sum().item()], device="cuda")
        digs = [torch.empty_like(dig) for _ in range(world)]
        dist.all_gather(digs, dig, group=group)
        same = all(torch.equal(d, digs[0]) for d in digs)
        ok = res_eq and out_ulps <= 1.0 and self_eq and same
        ok_all &= ok
        lines.append(f"M={m:6d} residual_bitwise={res_eq} bf16_max_ulp={out_ulps:.0f} "
                     f"fp8_eq_quant_of_own_bf16={self_eq} ranks_identical={same} {'OK' if ok else 'FAIL'}")
        del x, r, ar, ref_out, ref_res, o, ro, q, rq, s
        torch.cuda.empty_cache()
    if rank == 0:
        result_q.put((ok_all, lines))
    torch.cuda.synchronize()
    get_tp_group().barrier()
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

    init = get_distributed_init_method(get_ip(), get_open_port())
    q = torch.multiprocessing.get_context("spawn").SimpleQueue()
    torch.multiprocessing.spawn(worker, args=(4, init, q), nprocs=4, join=True)
    ok, lines = q.get()
    print("\n".join(lines))
    print("ALL OK" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)
