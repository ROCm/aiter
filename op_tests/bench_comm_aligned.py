# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Our MoonEP port, reported in upstream ``bench_comm.py``'s table format.

Produces the same row shape as MoonEP's own sweep, minus the backward columns
we have no counterpart for::

    | maxvio | planning | dispatch_f | epilogue_f | comb_prolog_f | combine_f | prefetch |

Everything that decides a number is taken from upstream rather than reinvented:

* **Routing** comes from ``MoonEP/tests/generate_topk_routing.py``, vendored
  below unchanged (it is pure torch, so it runs on ROCm as-is).  The sweep uses
  upstream's own three regimes, ``RATIOS = [0.1, 1.0, 5.0]``
  (``bench_comm.py:475``): near-balanced, typical dropless-MoE skew,
  pathological.
* **maxvio** is the global expert-load imbalance, ``max/mean - 1`` over the
  all-reduced tokens-per-expert histogram (``bench_comm.py:186-192``).
* **Timing** is a port of ``time_gpu_op``: warmup 5, iters 20, cross-rank
  **mean**.
* **Byte accounting** follows ``bench_comm.py:371-419`` where a GB/s column is
  printed.

Op mapping -- two of these are easy to get wrong, so they are spelled out:

``dispatch_f``      upstream times ``launch_dispatch`` **alone**; the epilogue is
                    its own column.  Ours is therefore the scatter kernel plus
                    ``shmem_barrier_on_stream`` (upstream's dispatch kernel
                    carries its own cross-rank barrier), and explicitly **not**
                    ``op.dispatch``, which also runs the epilogue.
``combine_f``       upstream's combine is **dedup-aware**: it skips duplicate
                    entries because ``combine_prologue`` already folded them
                    (``moonep/combine.py:320``).  The comparable kernel is our
                    ``skip_duplicates`` combine, not the reference one that
                    pulls all K rows.  The reference is reported separately as
                    ``combine_f_allK`` so the difference stays visible.
``planning``        our fused single-kernel GPU planner.  The CPU reference
                    planner is ~802 ms and is not a candidate.

Deviations, both forced and both stated in the output:

1. Eager, not CUDA graph -- ``torch.cuda.graph`` does not capture the FlyDSL
   launch path ("The CUDA Graph is empty").  On the B300 run the two were
   indistinguishable (<=0.1%).
2. Routing is *statistically* the same as upstream's, not bit-identical: the
   generator draws from a device RNG, and the ROCm stream differs from CUDA's.
   So maxvio lands near, not on, upstream's values -- the measured maxvio is
   reported per row rather than assumed.

Upstream runs at ``--num-sms 32``.  Our kernels are far from their best there,
so the table is emitted at every requested block count; 32 is the literally
aligned point and the rest shows where our kernels actually want to run.

Run under torchrun with one process per GPU::

    torchrun --standalone --nproc-per-node=8 op_tests/bench_comm_aligned.py
"""

from __future__ import annotations

import csv
import json
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.shmem as ms
import torch
import torch.distributed as dist
from mori.shmem import mori_shmem_create_tensor, mori_shmem_free_tensor

from aiter.ops.flydsl.kernels.moonep_combine import make_moonep_combine_jit
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
from aiter.ops.flydsl.kernels.moonep_dispatch_epilogue_fast import (
    make_moonep_dispatch_epilogue_fast_jit,
)
from aiter.ops.flydsl.kernels.moonep_weight_prefetch_fast import (
    make_moonep_weight_prefetch_fast_jit,
)
from aiter.ops.flydsl.moonep import (
    MoonEPGpuPlanner,
    MoonEPPlanConfig,
    MoonEPReferencePlan,
)
from aiter.ops.flydsl.moonep_ep import MoonEPBF16ReferenceEP
from op_tests.test_moonep_gfx950_real_shape import (
    B,
    E,
    H,
    I,
    K,
    R,
    S,
    TOKEN_PADDING,
    _identity_home_weights,
    _share_shmem_unique_id,
)

WARMUP = int(os.environ.get("MOONEP_BENCH_WARMUP", "5"))
ITERS = int(os.environ.get("MOONEP_BENCH_ITERS", "20"))
# bench_comm.py:475 RATIOS -- sigma of the lognormal expert-logit distribution.
RATIOS = [float(x) for x in os.environ.get("MOONEP_RATIOS", "0.1,1.0,5.0").split(",")]
# bench_comm's --num-sms default is 32; ours maps to block_num.
NUM_SMS = [int(x) for x in os.environ.get("MOONEP_NUM_SMS", "32,1024").split(",")]
OUT_CSV = os.environ.get("MOONEP_OUT_CSV", "")


def generate_topk_routing(S, K, E, R, bias_ratio, dev, seed, rank=0):
    """Verbatim from ``MoonEP/tests/generate_topk_routing.py``.

    Vendored rather than imported because MoonEP is a CUDA-only package (every
    ``moonep.*`` kernel module imports ``cutlass``/``cuda.bindings``), so it
    cannot be installed in the ROCm container.  This function itself is pure
    torch and runs unchanged.
    """

    g_shared = torch.Generator(device=dev).manual_seed(seed)
    g_local = torch.Generator(device=dev).manual_seed(rank)
    if bias_ratio == 0.0:
        epn = E // R
        toks = torch.arange(S, device=dev)
        ks = torch.arange(K, device=dev)
        target_rank = (toks[:, None] + ks[None, :]) % R
        target_local = ((toks[:, None] // R) + ks[None, :]) % epn
        perm = torch.randperm(epn, device=dev, generator=g_local)
        topk = (target_rank * epn + perm[target_local]).to(torch.int32)
    else:
        logits = torch.exp(
            torch.normal(
                mean=0.0, std=bias_ratio, size=(E,), device=dev, generator=g_shared
            )
        )
        probs = logits[None, :].expand(S, E)
        topk = torch.multinomial(
            probs, K, replacement=False, generator=g_local
        ).to(torch.int32)
    tpe = torch.bincount(topk.flatten(), minlength=E).to(torch.int32)
    return topk, tpe


def time_gpu_op(launch_fn, group):
    """Port of ``bench_comm.time_gpu_op`` (eager branch); returns the mean."""

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
    """compile-then-call wrapper whose first invocation actually executes."""

    def __init__(self, jit, ptr_args, stream):
        self._raw = tuple(ptr_args) + (stream,)
        self._compiled = flyc.compile(
            jit, *(fx.Int64(p) for p in ptr_args), stream
        )

    def __call__(self):
        self._compiled(*self._raw)


def dispatch_kernel_only(op, hidden, route_weights, plan, stream):
    """Just the scatter kernel -- upstream times ``launch_dispatch`` alone."""

    raw = (
        hidden.data_ptr(),
        route_weights.data_ptr(),
        plan.dst.data_ptr(),
        op.peer_hidden_ptrs.data_ptr(),
        op.peer_weight_ptrs.data_ptr(),
        op.peer_duplicate_src_ptrs.data_ptr(),
        op.config.num_tokens,
        stream,
    )
    compiled = flyc.compile(
        op._jit,
        fx.Int64(raw[0]),
        fx.Int64(raw[1]),
        fx.Int64(raw[2]),
        fx.Int64(raw[3]),
        fx.Int64(raw[4]),
        fx.Int64(raw[5]),
        raw[6],
        stream,
    )

    def run():
        compiled(*raw)

    return run


def main() -> int:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    assert (
        ms.shmem_init_attr(
            ms.MORI_SHMEM_INIT_WITH_UNIQUEID,
            rank,
            world_size,
            _share_shmem_unique_id(rank),
        )
        == 0
    )

    config = MoonEPPlanConfig(
        rank=rank,
        world_size=world_size,
        num_tokens=S,
        top_k=K,
        num_experts=E,
        prefetch_slots=B,
        token_padding=TOKEN_PADDING,
    )
    nvs = config.num_dispatch_rows
    n_entries = S * K
    experts_per_rank = E // R
    planner = MoonEPGpuPlanner(config, device)

    hidden = torch.randn(S, H, dtype=torch.bfloat16, device=device)
    route_weights = torch.rand(S, K, dtype=torch.float32, device=device)

    rows = []
    raw_json = {}

    for ratio in RATIOS:
        # bench_comm.py:182 -- shared seed 1234, per-rank draws seeded by rank.
        topk, tpe = generate_topk_routing(S, K, E, R, ratio, device, 1234, rank=rank)
        global_tpe = tpe.to(torch.int64).clone()
        dist.all_reduce(global_tpe)
        max_load = int(global_tpe.max().item())
        mean_load = float(global_tpe.sum().item()) / E
        maxvio = max_load / mean_load - 1.0 if mean_load else 0.0

        gathered = [torch.empty_like(tpe) for _ in range(R)]
        dist.all_gather(gathered, tpe)
        tpe_all = torch.stack(gathered)
        plan = planner.build(topk, tpe_all).clone()
        torch.cuda.synchronize(device)
        dist.barrier()

        raw_dst, is_primary = MoonEPReferencePlan.decode_dst(plan.dst)
        dedup_ratio = 1.0 - int(is_primary.sum().item()) / n_entries
        etc = plan.experts_to_copy
        valid = etc >= 0
        recv = int(valid[rank].sum().item())
        t = torch.tensor([float(recv)], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        max_recv = int(t.item())
        send_counts = torch.bincount(
            etc[valid].long() // experts_per_rank, minlength=R
        )
        max_send = int(send_counts.max().item())
        mx_experts = max(max_send, max_recv)

        for num_sms in NUM_SMS:
            ep = MoonEPBF16ReferenceEP(
                config, H, I, dispatch_block_num=num_sms, prefetch_block_num=num_sms
            )
            hg, hu, hd = _identity_home_weights(device)
            ep.load_home_weights(hg, hu, hd)
            del hg, hu, hd
            torch.cuda.empty_cache()
            op = ep.dispatch_op
            stream = torch.cuda.current_stream(device)

            # Populate recv_duplicate_src, which the prologue needs.
            op.dispatch(hidden, route_weights, plan)
            op.dispatch(hidden, route_weights, plan)
            torch.cuda.synchronize(device)
            dist.barrier()
            dup_rows = int((op.recv_duplicate_src >= 0).sum().item())

            src_slot = mori_shmem_create_tensor((nvs,), torch.int32)
            staging = mori_shmem_create_tensor(
                (n_entries, H), torch.bfloat16
            )
            src_slot.fill_(-1)
            torch.cuda.synchronize(device)
            ms.shmem_barrier_all()
            peer_slot_ptrs = torch.tensor(
                [ms.shmem_ptr_p2p(src_slot.data_ptr(), rank, p) for p in range(R)],
                dtype=torch.int64,
                device=device,
            )
            peer_staging_ptrs = torch.tensor(
                [ms.shmem_ptr_p2p(staging.data_ptr(), rank, p) for p in range(R)],
                dtype=torch.int64,
                device=device,
            )
            dup_count = torch.zeros(nvs, dtype=torch.int32, device=device)
            dup_list = torch.zeros(nvs * (K - 1), dtype=torch.int32, device=device)

            d_kernel = dispatch_kernel_only(op, hidden, route_weights, plan, stream)

            def dispatch_f():
                d_kernel()
                ms.shmem_barrier_on_stream(stream)

            fast_ep_jit = make_moonep_dispatch_epilogue_fast_jit(
                hidden_dim=H,
                num_dispatch_rows=nvs,
                num_groups=E + B,
                block_num=num_sms,
                warp_num_per_block=4,
            )
            op._epilogue_jit = fast_ep_jit
            op._epilogue_compiled = None

            def epilogue_f():
                op._run_epilogue(plan, stream)

            epilogue_f()  # first call only compiles

            prologue = Launcher(
                make_moonep_combine_prologue_jit(
                    hidden_dim=H,
                    num_dispatch_rows=nvs,
                    top_k=K,
                    block_num=min(num_sms, 1024),
                ),
                (
                    op.expert_output.data_ptr(),
                    op.recv_duplicate_src.data_ptr(),
                    dup_count.data_ptr(),
                    dup_list.data_ptr(),
                ),
                stream,
            )
            combine_dedup = Launcher(
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
            combine_allk = Launcher(
                make_moonep_combine_jit(
                    num_tokens=S, hidden_dim=H, top_k=K, num_dispatch_rows=nvs
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
                    block_num=min(num_sms, 256),
                ),
                (plan.dst.data_ptr(), peer_slot_ptrs.data_ptr()),
                stream,
            )
            push = Launcher(
                make_moonep_push_rows_jit(
                    hidden_dim=H,
                    num_dispatch_rows=nvs,
                    num_tokens=S,
                    top_k=K,
                    block_num=num_sms,
                ),
                (
                    op.expert_output.data_ptr(),
                    src_slot.data_ptr(),
                    peer_staging_ptrs.data_ptr(),
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

            gate = ep.gate_op
            local_sel = plan.experts_to_copy[rank].contiguous()
            prefetch = Launcher(
                make_moonep_weight_prefetch_fast_jit(
                    experts_per_rank=gate.experts_per_rank,
                    prefetch_slots=gate.prefetch_slots,
                    weight_numel=gate.weight_numel,
                    block_num=num_sms,
                    block_threads=gate.block_threads,
                ),
                (
                    local_sel.data_ptr(),
                    gate.peer_home_weight_ptrs.data_ptr(),
                    gate.prefetched_weights.data_ptr(),
                ),
                stream,
            )

            def combine_f():
                ms.shmem_barrier_on_stream(stream)
                combine_dedup()

            def combine_allk_f():
                ms.shmem_barrier_on_stream(stream)
                combine_allk()

            def combine_push():
                push()
                ms.shmem_barrier_on_stream(stream)
                reduce_local()

            def planning():
                planner.build(topk, tpe_all)

            timings = {}
            for name, fn in (
                ("planning", planning),
                ("dispatch_f", dispatch_f),
                ("epilogue_f", epilogue_f),
                ("comb_prolog_f", prologue),
                ("combine_f", combine_f),
                ("combine_f_allK", combine_allk_f),
                ("combine_f_push", combine_push),
                ("prefetch", prefetch),
            ):
                timings[name] = time_gpu_op(fn, dist.group.WORLD)
                torch.cuda.synchronize(device)
                dist.barrier()

            if rank == 0:
                row = {
                    "num_sms": num_sms,
                    "unb_r": ratio,
                    "maxvio": maxvio,
                    "dedup_pct": dedup_ratio * 100,
                    "mx_experts": mx_experts,
                    "dups": dup_rows,
                    **timings,
                }
                rows.append(row)
                raw_json[f"r{ratio}/sms{num_sms}"] = row
                print(
                    f"[done] unb_r={ratio} maxvio={maxvio:.2f} sms={num_sms} "
                    f"plan={timings['planning']:.1f} d_f={timings['dispatch_f']:.1f} "
                    f"c_f={timings['combine_f']:.1f}",
                    flush=True,
                )

            mori_shmem_free_tensor(staging)
            mori_shmem_free_tensor(src_slot)
            ep.close()
            del ep
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            dist.barrier()

    if rank == 0:
        print()
        print(
            f"MoonEP port on 8x MI355X -- upstream bench_comm format, forward only"
        )
        print(f"EP={R} S={S} H={H} K={K} E={E} Hp={I}  warmup={WARMUP} iters={ITERS}")
        print("eager (torch.cuda.graph does not capture FlyDSL launches)")
        print(
            "routing: MoonEP tests/generate_topk_routing.py, seed 1234, "
            "RATIOS=[0.1, 1.0, 5.0]"
        )
        print()
        print("upstream 8xB300, same shape, 32 SM (for reference):")
        print(
            "| maxvio | planning | dispatch_f | epilogue_f | comb_prolog_f "
            "| combine_f | prefetch |"
        )
        for mv, pl, df, ef, cp, cf, pf in (
            (0.32, 60.65, 958.0, 93.66, 137.1, 836.6, 161.5),
            (9.75, 60.95, 957.8, 92.80, 136.8, 850.5, 125.7),
            (47.0, 57.10, 1220.0, 61.27, 91.75, 1167.0, 130.9),
        ):
            print(
                f"| {mv:>6.2f} | {pl:>8.2f} | {df:>10.1f} | {ef:>10.2f} "
                f"| {cp:>13.1f} | {cf:>9.1f} | {pf:>8.1f} |"
            )
        for num_sms in NUM_SMS:
            print()
            print(f"ours, {num_sms} blocks:")
            print(
                "| maxvio | planning | dispatch_f | epilogue_f | comb_prolog_f "
                "| combine_f | prefetch |"
            )
            for r in rows:
                if r["num_sms"] != num_sms:
                    continue
                print(
                    f"| {r['maxvio']:>6.2f} | {r['planning']:>8.2f} "
                    f"| {r['dispatch_f']:>10.1f} | {r['epilogue_f']:>10.2f} "
                    f"| {r['comb_prolog_f']:>13.1f} | {r['combine_f']:>9.1f} "
                    f"| {r['prefetch']:>8.1f} |"
                )
            print("  combine variants (combine_f is the dedup-aware one, as upstream):")
            for r in rows:
                if r["num_sms"] != num_sms:
                    continue
                print(
                    f"    maxvio={r['maxvio']:>6.2f} dedup={r['dedup_pct']:>5.2f}% "
                    f"dups={r['dups']:>6} mx_experts={r['mx_experts']:>2} "
                    f"| dedup {r['combine_f']:>8.1f} "
                    f"| all-K {r['combine_f_allK']:>8.1f} "
                    f"| push {r['combine_f_push']:>8.1f}"
                )
        if OUT_CSV and rows:
            with open(OUT_CSV, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"\nwrote {OUT_CSV}")
        print("\nBENCH_COMM_ALIGNED_JSON " + json.dumps(raw_json))

    dist.barrier()
    ms.shmem_barrier_all()
    ms.shmem_finalize()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
