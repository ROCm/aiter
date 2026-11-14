# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""The missing baseline: MoonEP transport vs the EP backends ATOM already has.

Before wiring MoonEP into ATOM as a third ``ATOM_EP_BACKEND`` it has to beat the
two that are already there.  Nothing in the MoonEP work so far compares against
them -- every number was against upstream MoonEP on B300, which answers a
different question.  If our dispatch/combine does not beat mori's and the
existing FlyDSL op's on the same box, the integration has no payoff and should
not be built.

What MoonEP could bring, and what it costs, stated up front so the table can be
read against it:

+ **dedup**: a token routed to K=8 experts usually lands on only ~5.3 distinct
  ranks, so one payload copy per distinct rank instead of per expert.
+ **push combine**: remote writes (448 GB/s measured) instead of remote reads
  (235 GB/s), at balanced-to-typical routing.
- **an extra collective plus planning on the critical path**: MoonEP needs the
  all-ranks tokens-per-expert histogram before it can plan, and ~113 us to plan.
  Neither of the other two needs anything.
- **BF16 only** (``moonep_dispatch.py:63``), while the FlyDSL op supports
  ``fp8_direct_cast``.  Everything here is therefore forced to BF16 /
  ``quant_type="none"`` so the transports are compared on equal footing -- this
  flatters MoonEP, since fp8 would halve the other backends' wire bytes.
- ~2.2 GB of symmetric buffers, plus 940 MB more for push.

Method is the same as everywhere else in this series: routing from MoonEP's own
``generate_topk_routing`` at its three regimes, ``time_gpu_op`` with warmup 5 /
iters 20 and a cross-rank **mean**, eager, logical bytes ``S*K*H*2`` for GB/s
(upstream's normalising constant -- not a link utilisation).

Run under torchrun with one process per GPU::

    torchrun --standalone --nproc-per-node=8 op_tests/bench_ep_backends.py
"""

from __future__ import annotations

import json
import os
import traceback

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori
import mori.shmem as ms
import torch
import torch.distributed as dist
from mori.shmem import mori_shmem_create_tensor, mori_shmem_free_tensor

from aiter.ops.flydsl.kernels.moonep_combine_fast import (
    make_moonep_combine_fast_jit,
)
from aiter.ops.flydsl.kernels.moonep_combine_prologue import (
    make_moonep_combine_prologue_jit,
)
from aiter.ops.flydsl.kernels.moonep_combine_push import (
    make_moonep_publish_src_slots_jit,
    make_moonep_push_rows_jit,
    make_moonep_reduce_local_jit,
)
from aiter.ops.flydsl.moonep import MoonEPGpuPlanner, MoonEPPlanConfig
from aiter.ops.flydsl.kernels.moonep_dispatch_op import (
    MoonEPPreplannedDispatchOp,
)
from op_tests.bench_comm_aligned import generate_topk_routing

S = int(os.environ.get("MOONEP_S", "8192"))
H = int(os.environ.get("MOONEP_H", "7168"))
K = int(os.environ.get("MOONEP_K", "8"))
E = int(os.environ.get("MOONEP_E", "384"))
WARMUP = int(os.environ.get("MOONEP_BENCH_WARMUP", "5"))
ITERS = int(os.environ.get("MOONEP_BENCH_ITERS", "20"))
RATIOS = [float(x) for x in os.environ.get("MOONEP_RATIOS", "0.1,1.0,5.0").split(",")]
BLOCK_NUM = int(os.environ.get("MOONEP_BLOCK_NUM", "1024"))
# mori/flydsl default launch geometry; -1 lets each op pick its own.
PEER_BLOCK_NUM = int(os.environ.get("MOONEP_PEER_BLOCK_NUM", "-1"))
BACKENDS = os.environ.get("MOONEP_BACKENDS", "mori,flydsl,moonep").split(",")


def time_gpu_op(launch_fn, group):
    for _ in range(WARMUP):
        launch_fn()
    torch.cuda.synchronize()
    dist.barrier(group=group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        launch_fn()
    end.record()
    end.synchronize()
    local_us = start.elapsed_time(end) / ITERS * 1e3
    world = dist.get_world_size(group=group)
    dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    t = torch.tensor([local_us], dtype=torch.float64, device=dev)
    outs = [torch.empty(1, dtype=torch.float64, device=dev) for _ in range(world)]
    dist.all_gather(outs, t, group=group)
    return torch.cat(outs).mean().item()


class Launcher:
    def __init__(self, jit, ptr_args, stream):
        self._raw = tuple(ptr_args) + (stream,)
        self._compiled = flyc.compile(
            jit, *(fx.Int64(p) for p in ptr_args), stream
        )

    def __call__(self):
        self._compiled(*self._raw)


def main() -> int:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    epn = E // world_size

    # Use the process-group init that ATOM's backends use, not the unique-id
    # path the MoonEP benches used, so all three backends share one runtime.
    cpu_group = dist.new_group(backend="gloo")
    torch._C._distributed_c10d._register_process_group("mori", cpu_group)
    mori.shmem.shmem_torch_process_group_init("mori")

    hidden = torch.randn(S, H, dtype=torch.bfloat16, device=device)
    weights = torch.rand(S, K, dtype=torch.float32, device=device)

    # Every op must be sized for the worst case any ratio produces, and the
    # size has to agree across ranks because the buffers are symmetric.
    worst_recv = 0
    routings = {}
    for ratio in RATIOS:
        topk, tpe = generate_topk_routing(
            S, K, E, world_size, ratio, device, 1234, rank=rank
        )
        gathered = [torch.empty_like(tpe) for _ in range(world_size)]
        dist.all_gather(gathered, tpe)
        tpe_all = torch.stack(gathered)
        # Rows this rank receives = entries anywhere naming an expert we own.
        owned = tpe_all[:, rank * epn : (rank + 1) * epn].sum()
        t = owned.to(torch.float64).clone()
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        worst_recv = max(worst_recv, int(t.item()))
        routings[ratio] = (topk, tpe, tpe_all)
    max_recv = int(worst_recv * 1.15) + 1024
    if rank == 0:
        print(
            f"[setup] worst-case recv tokens/rank across ratios = {worst_recv}, "
            f"sizing buffers for {max_recv} "
            f"({max_recv * H * 2 / 1e9:.2f} GB per recv buffer)",
            flush=True,
        )

    results = {}

    def record(name, ratio, key, value):
        results.setdefault(f"{name}/r{ratio}", {})[key] = value

    # ---------------- mori ------------------------------------------------
    def run_mori(ratio, topk, tpe, tpe_all):
        # aiter's own mori tests (op_tests/multigpu_tests/test_dispatch_combine.py)
        # only ever exercise the fp8 path, where ``scale`` is a real tensor;
        # ``scales=None`` with scale_dim=0 is an untested configuration and is
        # the prime suspect for the memory fault.  Feed a dummy scale column so
        # the call matches the shape of the known-good one.
        scale = torch.ones(S, 1, dtype=torch.float32, device=device)
        cfg = mori.ops.EpDispatchCombineConfig(
            data_type=torch.bfloat16,
            rank=rank,
            world_size=world_size,
            hidden_dim=H,
            scale_dim=1,
            scale_type_size=4,
            max_token_type_size=2,
            # Headroom: the recv cap derives from this, and S*K == world_size*S
            # here, so a cap at exactly S leaves none for routing imbalance.
            max_num_inp_token_per_rank=2 * S,
            num_experts_per_rank=epn,
            num_experts_per_token=K,
            # Mirror aiter's MoriAll2AllManager._make_all2all_kwargs exactly.
            # In particular max_total_recv_tokens is left at its default: the op
            # derives its internal buffer sizes from it, and overriding it here
            # under-sized them and faulted the kernel.
            warp_num_per_block=16,
            block_num=80,
            # Left at the default in ATOM, but ATOM sizes
            # max_num_inp_token_per_rank with headroom over the real batch.
            # Here S*K == world_size*S == 65536, i.e. the default cap sits
            # exactly at the *mean* recv count, so any imbalance overflows it.
            max_total_recv_tokens=max_recv,
            kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode,
        )
        op = mori.ops.EpDispatchCombineOp(cfg)
        idx = topk.to(torch.int32).contiguous()
        out = op.dispatch(hidden, weights, scale, idx)
        packed = out[0] if isinstance(out, (tuple, list)) else out
        torch.cuda.synchronize(device)
        dist.barrier()

        def d():
            op.dispatch(hidden, weights, scale, idx)

        def c():
            op.combine(packed, weights, idx)

        record("mori", ratio, "dispatch_us", time_gpu_op(d, dist.group.WORLD))
        torch.cuda.synchronize(device)
        dist.barrier()
        record("mori", ratio, "combine_us", time_gpu_op(c, dist.group.WORLD))
        torch.cuda.synchronize(device)
        dist.barrier()
        del op

    # ---------------- existing FlyDSL op ----------------------------------
    def run_flydsl(ratio, topk, tpe, tpe_all):
        from aiter.ops.flydsl.kernels.flydsl_dispatch_combine_intranode_op import (
            FlyDSLDispatchCombineConfig,
            FlyDSLDispatchCombineIntraNodeOp,
        )

        cfg = FlyDSLDispatchCombineConfig(
            rank=rank,
            world_size=world_size,
            hidden_dim=H,
            max_num_inp_token_per_rank=S,
            num_experts_per_rank=epn,
            num_experts_per_token=K,
            data_type=torch.bfloat16,
            max_token_type_size=2,
            quant_type="none",
        )
        op = FlyDSLDispatchCombineIntraNodeOp(cfg)
        idx = topk.to(torch.int32).contiguous()
        out = op.dispatch(hidden, weights, None, idx)
        packed = out[0] if isinstance(out, (tuple, list)) else out
        torch.cuda.synchronize(device)
        dist.barrier()

        def d():
            op.dispatch(hidden, weights, None, idx)

        def c():
            op.combine(packed, weights, idx)

        record("flydsl", ratio, "dispatch_us", time_gpu_op(d, dist.group.WORLD))
        torch.cuda.synchronize(device)
        dist.barrier()
        record("flydsl", ratio, "combine_us", time_gpu_op(c, dist.group.WORLD))
        torch.cuda.synchronize(device)
        dist.barrier()
        del op

    # ---------------- ours -------------------------------------------------
    def run_moonep(ratio, topk, tpe, tpe_all):
        config = MoonEPPlanConfig(
            rank=rank,
            world_size=world_size,
            num_tokens=S,
            top_k=K,
            num_experts=E,
            prefetch_slots=1,  # no weight migration in a transport comparison
            token_padding=128,
        )
        planner = MoonEPGpuPlanner(config, device)
        plan = planner.build(topk, tpe_all).clone()
        nvs = config.num_dispatch_rows
        op = MoonEPPreplannedDispatchOp(config, H, block_num=BLOCK_NUM)
        stream = torch.cuda.current_stream(device)
        op.dispatch(hidden, weights, plan)
        op.dispatch(hidden, weights, plan)
        torch.cuda.synchronize(device)
        dist.barrier()

        n_entries = S * K
        src_slot = mori_shmem_create_tensor((nvs,), torch.int32)
        staging = mori_shmem_create_tensor((n_entries, H), torch.bfloat16)
        src_slot.fill_(-1)
        torch.cuda.synchronize(device)
        ms.shmem_barrier_all()
        peer_slot = torch.tensor(
            [
                ms.shmem_ptr_p2p(src_slot.data_ptr(), rank, p)
                for p in range(world_size)
            ],
            dtype=torch.int64,
            device=device,
        )
        peer_stage = torch.tensor(
            [
                ms.shmem_ptr_p2p(staging.data_ptr(), rank, p)
                for p in range(world_size)
            ],
            dtype=torch.int64,
            device=device,
        )
        dup_count = torch.zeros(nvs, dtype=torch.int32, device=device)
        dup_list = torch.zeros(nvs * (K - 1), dtype=torch.int32, device=device)

        prologue = Launcher(
            make_moonep_combine_prologue_jit(
                hidden_dim=H, num_dispatch_rows=nvs, top_k=K, block_num=BLOCK_NUM
            ),
            (
                op.expert_output.data_ptr(),
                op.recv_duplicate_src.data_ptr(),
                dup_count.data_ptr(),
                dup_list.data_ptr(),
            ),
            stream,
        )
        dedup = Launcher(
            make_moonep_combine_fast_jit(
                num_tokens=S,
                hidden_dim=H,
                top_k=K,
                num_dispatch_rows=nvs,
                skip_duplicates=True,
            ),
            (
                plan.dst.data_ptr(),
                op.peer_expert_output_ptrs.data_ptr(),
                op.peer_weight_ptrs.data_ptr(),
                op.combine_output.data_ptr(),
                op.gathered_route_weights.data_ptr(),
            ),
            stream,
        )
        publish = Launcher(
            make_moonep_publish_src_slots_jit(
                num_tokens=S,
                top_k=K,
                num_dispatch_rows=nvs,
                rank=rank,
                block_num=min(BLOCK_NUM, 256),
            ),
            (plan.dst.data_ptr(), peer_slot.data_ptr()),
            stream,
        )
        push = Launcher(
            make_moonep_push_rows_jit(
                hidden_dim=H,
                num_dispatch_rows=nvs,
                num_tokens=S,
                top_k=K,
                block_num=BLOCK_NUM,
            ),
            (
                op.expert_output.data_ptr(),
                src_slot.data_ptr(),
                peer_stage.data_ptr(),
            ),
            stream,
        )
        reduce_local = Launcher(
            make_moonep_reduce_local_jit(
                num_tokens=S, hidden_dim=H, top_k=K, num_dispatch_rows=nvs
            ),
            (
                plan.dst.data_ptr(),
                staging.data_ptr(),
                op.peer_weight_ptrs.data_ptr(),
                op.combine_output.data_ptr(),
                op.gathered_route_weights.data_ptr(),
            ),
            stream,
        )
        publish()
        torch.cuda.synchronize(device)
        dist.barrier()

        def d():
            op.dispatch(hidden, weights, plan)

        def c_pull():
            prologue()
            ms.shmem_barrier_on_stream(stream)
            dedup()

        def c_push():
            prologue()
            push()
            ms.shmem_barrier_on_stream(stream)
            reduce_local()

        def planning():
            # The cost the other two backends do not pay: an all_gather of the
            # tokens-per-expert histogram, then the plan itself.
            g = [torch.empty_like(tpe) for _ in range(world_size)]
            dist.all_gather(g, tpe)
            planner.build(topk, torch.stack(g))

        record("moonep", ratio, "dispatch_us", time_gpu_op(d, dist.group.WORLD))
        torch.cuda.synchronize(device)
        dist.barrier()
        record("moonep", ratio, "combine_us", time_gpu_op(c_pull, dist.group.WORLD))
        torch.cuda.synchronize(device)
        dist.barrier()
        record(
            "moonep", ratio, "combine_push_us", time_gpu_op(c_push, dist.group.WORLD)
        )
        torch.cuda.synchronize(device)
        dist.barrier()
        record(
            "moonep", ratio, "plan_overhead_us", time_gpu_op(planning, dist.group.WORLD)
        )
        torch.cuda.synchronize(device)
        dist.barrier()

        mori_shmem_free_tensor(staging)
        mori_shmem_free_tensor(src_slot)
        op.close()
        del op

    runners = {"mori": run_mori, "flydsl": run_flydsl, "moonep": run_moonep}

    for ratio in RATIOS:
        topk, tpe, tpe_all = routings[ratio]
        for name in BACKENDS:
            try:
                runners[name](ratio, topk, tpe, tpe_all)
                if rank == 0:
                    print(f"[done] {name} ratio={ratio}", flush=True)
            except Exception as exc:  # one broken backend must not kill the run
                if rank == 0:
                    print(f"[FAIL] {name} ratio={ratio}: {type(exc).__name__}: {exc}")
                    traceback.print_exc()
                results[f"{name}/r{ratio}"] = {"error": f"{type(exc).__name__}: {exc}"}
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            dist.barrier()

    if rank == 0:
        logical = S * K * H * 2
        print()
        print(f"EP={world_size} S={S} H={H} K={K} E={E} bf16, quant=none")
        print(f"warmup={WARMUP} iters={ITERS}, cross-rank mean, eager")
        print(f"logical bytes = S*K*H*2 = {logical / 1e6:.1f} MB (upstream convention)")
        print()
        print(
            f"{'backend/ratio':<20}{'dispatch us':>12}{'combine us':>12}"
            f"{'comb push':>12}{'plan+ag':>10}{'d GB/s':>9}{'c GB/s':>9}"
        )
        for key in sorted(results):
            r = results[key]
            if "error" in r:
                print(f"{key:<20}  {r['error']}")
                continue
            d = r.get("dispatch_us", float("nan"))
            c = r.get("combine_us", float("nan"))
            cp = r.get("combine_push_us", float("nan"))
            pl = r.get("plan_overhead_us", float("nan"))
            print(
                f"{key:<20}{d:>12.1f}{c:>12.1f}{cp:>12.1f}{pl:>10.1f}"
                f"{logical / d / 1e3:>9.1f}{logical / c / 1e3:>9.1f}"
            )
        print("\nEP_BACKENDS_JSON " + json.dumps(results))

    dist.barrier()
    ms.shmem_barrier_all()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
