# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Apples-to-apples MoonEP dispatch timing against upstream ``bench_comm.py``.

``profile_moonep_gfx950_real_shape.py`` reports ``dispatch_zero_copy`` as the
per-rank **max** of an **eager** pipeline stage that bundles three things:
the FlyDSL scatter kernel, ``shmem_barrier_on_stream``, and the local
duplicate-expansion epilogue.  Upstream's ``dispatch_fwd`` is none of those:

  MoonEP/benchmarks/bench_comm.py
    * ``dispatch_fwd`` times ``launch_dispatch(...)`` alone; the epilogue is a
      separate ``epilogue_fwd`` row.  Upstream's dispatch kernel carries its own
      cross-rank barrier, so our comparable unit is kernel + shmem barrier.
    * ``time_gpu_op`` captures ``iters`` launches in a CUDA graph and times the
      replay, so host launch overhead is excluded (``cudagraph=False`` gives the
      eager number).
    * warmup=5, iters=20, ``num_sms=32``.
    * the reported number is the **cross-rank mean** (all_gather then mean),
      not the max.

This script mirrors that helper exactly and reports every sub-part separately,
so the difference between our number and upstream's is a property of the
kernels rather than of the harness.

Run under torchrun with one process per GPU::

    torchrun --standalone --nproc-per-node=8 \
        op_tests/benchmark_moonep_dispatch_aligned.py
"""

from __future__ import annotations

import json
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.shmem as ms
import torch
import torch.distributed as dist

from aiter.ops.flydsl.kernels.moonep_combine_fast import (
    make_moonep_combine_fast_jit,
)
from aiter.ops.flydsl.kernels.moonep_combine_prologue import (
    make_moonep_combine_prologue_jit,
)
from aiter.ops.flydsl.kernels.moonep_dispatch_epilogue_fast import (
    make_moonep_dispatch_epilogue_fast_jit,
)
from aiter.ops.flydsl.moonep import MoonEPPlanConfig, build_reference_plan
from aiter.ops.flydsl.moonep_ep import MoonEPBF16ReferenceEP
from op_tests.profile_moonep_gfx950_real_shape import _inputs
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
# Upstream sweeps the comm kernel at 32 SMs; ours defaults to 128 blocks.
BLOCK_NUMS = [int(x) for x in os.environ.get("MOONEP_BLOCK_NUMS", "32,64,128,256").split(",")]
# The profile harness routes with ``(token*K + k + rank*13) % B``, which only
# ever touches B=48 of the E=384 experts and gives each token 8 consecutive
# ids.  That skews the dedup ratio and the wire traffic, so "uniform" draws a
# realistic top-k (K distinct experts per token, uniform over E) instead.
ROUTING = os.environ.get("MOONEP_ROUTING", "uniform")
# token_padding isolates the epilogue's two kernels: the duplicate-expansion
# work is independent of it, while tp=1 removes the zero-fill work entirely,
# so epilogue(tp=128) - epilogue(tp=1) is the zero_padding kernel.
TP = int(os.environ.get("MOONEP_TOKEN_PADDING", str(TOKEN_PADDING)))


def time_gpu_op(launch_fn, group, *, cudagraph: bool):
    """Port of MoonEP ``bench_comm.time_gpu_op``.

    Returns ``(cross_rank_mean_us, cross_rank_max_us, local_us)``.  Upstream
    reports only the mean; the max is added here because our dispatch is the one
    stage whose per-rank spread is large.
    """

    for _ in range(WARMUP):
        launch_fn()
    torch.cuda.synchronize()
    dist.barrier(group=group)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if cudagraph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(ITERS):
                launch_fn()
        torch.cuda.synchronize()
        dist.barrier(group=group)
        start.record()
        graph.replay()
        end.record()
    else:
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
    allr = torch.cat(outs)
    return allr.mean().item(), allr.max().item(), local_us


def _kernel_only_fn(op, hidden, route_weights, plan):
    """Launch just the FlyDSL scatter kernel (no barrier, no epilogue).

    Mirrors ``MoonEPPreplannedDispatchOp.dispatch``'s launch block; kept here
    rather than added to the op so the validated PoC file is untouched.
    """

    stream = torch.cuda.current_stream(op.device)
    args = (
        fx.Int64(hidden.data_ptr()),
        fx.Int64(route_weights.data_ptr()),
        fx.Int64(plan.dst.data_ptr()),
        fx.Int64(op.peer_hidden_ptrs.data_ptr()),
        fx.Int64(op.peer_weight_ptrs.data_ptr()),
        fx.Int64(op.peer_duplicate_src_ptrs.data_ptr()),
        op.config.num_tokens,
        stream,
    )
    if op._compiled is None:
        op._compiled = flyc.compile(op._jit, *args)

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

    def run():
        op._compiled(*raw)

    return run


def main() -> int:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    status = ms.shmem_init_attr(
        ms.MORI_SHMEM_INIT_WITH_UNIQUEID,
        rank,
        world_size,
        _share_shmem_unique_id(rank),
    )
    assert status == 0

    config = MoonEPPlanConfig(
        rank=rank,
        world_size=world_size,
        num_tokens=S,
        top_k=K,
        num_experts=E,
        prefetch_slots=B,
        token_padding=TP,
    )
    hidden, route_weights, topk, local_tpe = _inputs(rank, device)
    if ROUTING == "uniform":
        gen = torch.Generator(device="cpu").manual_seed(20260810 + rank)
        topk = (
            torch.rand(S, E, generator=gen).topk(K, dim=1).indices.to(torch.int32)
        ).to(device)
        local_tpe = torch.bincount(
            topk.reshape(-1).to(torch.int64), minlength=E
        ).to(torch.int32)
    gathered = [torch.empty_like(local_tpe) for _ in range(R)]
    dist.all_gather(gathered, local_tpe)
    plan = build_reference_plan(config, topk, torch.stack(gathered))
    torch.cuda.synchronize(device)
    dist.barrier()

    group = dist.group.WORLD
    results = {}

    # Ceiling probe: the SAME dispatch kernel driven by an idealised plan --
    # entry i goes to peer i%R at row i//R, every entry a primary.  Same
    # mechanism (mori peer pointers, FlyDSL buffer stores, identical code),
    # only the addressing changes, so the delta against the real plan is
    # exactly the cost of the scattered access pattern.  If the real plan
    # already matches this, the bottleneck is the link, not our addressing.
    n_entries = S * K
    flat = torch.arange(n_entries, dtype=torch.int64, device=device)
    ideal_dst = (
        (flat % R) * config.num_dispatch_rows + flat // R
    ).to(torch.int32).view(S, K).contiguous()
    ideal_plan = plan.clone()
    object.__setattr__(ideal_plan, "dst", ideal_dst)
    ideal_wire_bytes = n_entries * (R - 1) // R * H * 2

    for block_num in BLOCK_NUMS:
        ep = MoonEPBF16ReferenceEP(
            config, H, I, dispatch_block_num=block_num, prefetch_block_num=block_num
        )
        home_gate, home_up, home_down = _identity_home_weights(device)
        ep.load_home_weights(home_gate, home_up, home_down)
        del home_gate, home_up, home_down
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        dist.barrier()

        op = ep.dispatch_op
        stream = torch.cuda.current_stream(device)
        kernel_only = _kernel_only_fn(op, hidden, route_weights, plan)
        kernel_ideal = _kernel_only_fn(op, hidden, route_weights, ideal_plan)

        def kernel_plus_barrier():
            kernel_only()
            ms.shmem_barrier_on_stream(stream)

        def epilogue_only():
            op._run_epilogue(plan, stream)

        # A/B the tuned epilogue by swapping the op's JIT in place; the
        # reference builder and the op itself are untouched.
        ref_jit, ref_compiled = op._epilogue_jit, op._epilogue_compiled
        fast_jit = make_moonep_dispatch_epilogue_fast_jit(
            hidden_dim=H,
            num_dispatch_rows=config.num_dispatch_rows,
            num_groups=E + B,
            block_num=block_num,
            warp_num_per_block=4,
        )

        def epilogue_fast():
            op._epilogue_jit, op._epilogue_compiled = fast_jit, fast_compiled[0]
            op._run_epilogue(plan, stream)
            fast_compiled[0] = op._epilogue_compiled
            op._epilogue_jit, op._epilogue_compiled = ref_jit, ref_compiled_box[0]

        fast_compiled = [None]
        ref_compiled_box = [ref_compiled]

        def full_dispatch():
            op.dispatch(hidden, route_weights, plan)

        # Upstream ``combine_fwd`` is launch_combine alone; its kernel carries
        # the cross-rank barrier, and ours puts shmem_barrier_on_stream in
        # front of the kernel, so op.combine is the comparable unit.
        def combine():
            op.combine(plan)

        # Same in-place JIT swap trick as the epilogue A/B.
        ref_combine_jit = op._combine_jit
        fast_combine_jit = make_moonep_combine_fast_jit(
            num_tokens=S,
            hidden_dim=H,
            top_k=K,
            num_dispatch_rows=config.num_dispatch_rows,
        )
        fast_combine_compiled = [None]
        ref_combine_compiled = [None]

        # Upstream-equivalent path: fold duplicates into their primary on the
        # destination rank, then let combine skip them (combine.py:320).
        dup_count = torch.zeros(
            config.num_dispatch_rows, dtype=torch.int32, device=device
        )
        dup_list = torch.zeros(
            config.num_dispatch_rows * (K - 1), dtype=torch.int32, device=device
        )
        prologue_jit = make_moonep_combine_prologue_jit(
            hidden_dim=H,
            num_dispatch_rows=config.num_dispatch_rows,
            top_k=K,
            block_num=min(block_num, 1024),
        )
        prologue_compiled = [None]
        dedup_combine_jit = make_moonep_combine_fast_jit(
            num_tokens=S,
            hidden_dim=H,
            top_k=K,
            num_dispatch_rows=config.num_dispatch_rows,
            skip_duplicates=True,
        )
        dedup_combine_compiled = [None]

        def prologue_only():
            args = (
                fx.Int64(op.recv_hidden.data_ptr()),
                fx.Int64(op.recv_duplicate_src.data_ptr()),
                fx.Int64(dup_count.data_ptr()),
                fx.Int64(dup_list.data_ptr()),
                stream,
            )
            if prologue_compiled[0] is None:
                prologue_compiled[0] = flyc.compile(prologue_jit, *args)
            else:
                prologue_compiled[0](
                    op.recv_hidden.data_ptr(),
                    op.recv_duplicate_src.data_ptr(),
                    dup_count.data_ptr(),
                    dup_list.data_ptr(),
                    stream,
                )

        def combine_dedup():
            prologue_only()
            op._combine_jit = dedup_combine_jit
            op._combine_compiled = dedup_combine_compiled[0]
            op.combine(plan)
            dedup_combine_compiled[0] = op._combine_compiled
            op._combine_jit = ref_combine_jit
            op._combine_compiled = ref_combine_compiled[0]

        def combine_ideal():
            # Same probe as dispatch_kernel_ideal_dst: perfectly sequential
            # remote rows, same kernel and mechanism.  Output is meaningless.
            op.combine(ideal_plan)

        def combine_fast():
            op._combine_jit = fast_combine_jit
            op._combine_compiled = fast_combine_compiled[0]
            op.combine(plan)
            fast_combine_compiled[0] = op._combine_compiled
            op._combine_jit = ref_combine_jit
            op._combine_compiled = ref_combine_compiled[0]

        # Upstream ``prefetch`` moves ONE [epn, H, Hp] weight matrix per call.
        # Our gate/up/down are three separate ops each moving one matrix, so a
        # single op -- not their sum -- is the comparable unit.
        def prefetch_one():
            ep.gate_op.prefetch(plan.experts_to_copy)

        # Prime every JIT and stage combine's input (untimed setup, as upstream
        # does with hidden_buf_local).
        kernel_only()
        ms.shmem_barrier_on_stream(stream)
        epilogue_only()
        ref_compiled_box[0] = op._epilogue_compiled
        epilogue_fast()

        # The tuned epilogue must be byte-identical to the reference one.  Both
        # run on the same freshly dispatched state, which the barrier publishes.
        def _fresh_dispatch():
            kernel_only()
            ms.shmem_barrier_on_stream(stream)
            torch.cuda.synchronize(device)

        _fresh_dispatch()
        epilogue_only()
        torch.cuda.synchronize(device)
        ref_hidden = op.recv_hidden.clone()
        ref_weights = op.recv_route_weights.clone()
        _fresh_dispatch()
        epilogue_fast()
        torch.cuda.synchronize(device)
        same_h = torch.equal(ref_hidden, op.recv_hidden)
        same_w = torch.equal(ref_weights, op.recv_route_weights)
        if not (same_h and same_w):
            bad = (ref_hidden != op.recv_hidden).nonzero()
            raise AssertionError(
                f"rank{rank} bn{block_num}: fast epilogue differs "
                f"(hidden_ok={same_h} weights_ok={same_w}, "
                f"{bad.shape[0]} mismatched hidden elements)"
            )
        if rank == 0:
            print(f"[check] bn{block_num}: fast epilogue == reference epilogue")
        _fresh_dispatch()
        dist.barrier()

        combine()
        ref_combine_compiled[0] = op._combine_compiled
        torch.cuda.synchronize(device)
        ref_out = op.combine_output.clone()
        ref_gw = op.gathered_route_weights.clone()
        dist.barrier()
        combine_fast()
        torch.cuda.synchronize(device)
        if not (
            torch.equal(ref_out, op.combine_output)
            and torch.equal(ref_gw, op.gathered_route_weights)
        ):
            bad = (ref_out != op.combine_output).nonzero()
            raise AssertionError(
                f"rank{rank} bn{block_num}: fast combine differs "
                f"({bad.shape[0]} mismatched output elements)"
            )
        if rank == 0:
            print(f"[check] bn{block_num}: fast combine == reference combine")

        # The dedup path folds duplicates in fp32 and rounds to bf16 once more
        # than summing all K in fp32 inside combine, so compare with a
        # tolerance rather than bit-for-bit.
        staged = op.recv_hidden.clone()
        combine_dedup()
        torch.cuda.synchronize(device)
        err = (
            (ref_out.float() - op.combine_output.float()).abs().max().item()
        )
        scale = ref_out.float().abs().max().item()
        if rank == 0:
            print(
                f"[check] bn{block_num}: dedup combine max|err|={err:.3e} "
                f"(max|ref|={scale:.3e}, rel={err / max(scale, 1e-30):.2e})"
            )
        if err > 5e-2 * max(scale, 1e-30):
            raise AssertionError(
                f"rank{rank}: dedup combine off by {err:.3e} vs {scale:.3e}"
            )
        op.recv_hidden.copy_(staged)
        torch.cuda.synchronize(device)
        dist.barrier()
        ref_combine_compiled[0] = op._combine_compiled
        combine_fast()
        prefetch_one()
        torch.cuda.synchronize(device)
        dist.barrier()

        # NOTE: torch.cuda.graph does not capture the FlyDSL launch path here
        # ("The CUDA Graph is empty"), so graph replay times an empty graph.
        # Eager only until that is fixed.
        for mode in ("eager",):
            use_graph = mode == "cudagraph"
            for name, fn in (
                ("dispatch_kernel", kernel_only),
                ("dispatch_kernel_ideal_dst", kernel_ideal),
                ("dispatch_kernel+barrier", kernel_plus_barrier),
                ("epilogue_reference", epilogue_only),
                ("epilogue_fast", epilogue_fast),
                ("full_dispatch_api", full_dispatch),
                ("combine_reference", combine),
                ("combine_fast", combine_fast),
                ("combine_ideal_dst", combine_ideal),
                ("combine_prologue", prologue_only),
                ("combine_dedup_total", combine_dedup),
                ("prefetch_one_matrix", prefetch_one),
            ):
                try:
                    mean_us, max_us, _ = time_gpu_op(fn, group, cudagraph=use_graph)
                except Exception as exc:  # graph capture may reject the barrier
                    if rank == 0:
                        print(f"[skip] {mode}/{name}/bn{block_num}: {type(exc).__name__}: {exc}")
                    torch.cuda.synchronize(device)
                    dist.barrier()
                    continue
                results[f"{mode}/{name}/bn{block_num}"] = (mean_us, max_us)
                torch.cuda.synchronize(device)
                dist.barrier()

        ep.close()
        del ep
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        dist.barrier()

    if rank == 0:
        # Upstream's GB/s column is a *logical* count: S*K*H*2 over the time,
        # including this rank's own 1/R share.  Reproduced here only so the two
        # numbers are directly comparable -- it is not a link utilisation.
        logical_bytes = S * K * H * 2
        # What actually crosses xGMI: dedup means one hidden row per (token,
        # distinct destination rank), minus this rank's own share.
        raw_dst, is_primary = type(plan).decode_dst(plan.dst)
        primaries = int(is_primary.sum().item())
        local_rows = int(
            ((raw_dst // config.num_dispatch_rows == rank) & is_primary).sum().item()
        )
        wire_bytes = (primaries - local_rows) * H * 2
        # Upstream byte accounting: combine_fwd_bytes = S*K*H*2 (logical).
        # Prefetch moves one [slots, H, I] bf16 matrix; slots is routing
        # dependent, so report it rather than comparing raw times.
        slots = int((plan.experts_to_copy[rank] >= 0).sum().item())
        dup_rows = int((~is_primary).sum().item())
        pad_rows = int(plan.zero_fill_ranges[:, 1].sum().item())
        prefetch_bytes = slots * H * I * 2

        print(f"shape S={S} H={H} K={K} E={E} R={R} B={B} tp={TP} routing={ROUTING} NvS={config.num_dispatch_rows}")
        print(f"warmup={WARMUP} iters={ITERS}")
        print(f"primary rows={primaries} (local {local_rows}) -> wire {wire_bytes/1e6:.1f} MB/rank")
        print(f"ideal-dst probe writes {ideal_wire_bytes/1e6:.1f} MB/rank")
        print(f"prefetch slots={slots}/{B} -> {prefetch_bytes/1e6:.1f} MB per weight matrix")
        print(f"epilogue: dup rows={dup_rows} (read+write {2*dup_rows*H*2/1e6:.1f} MB), "
              f"pad rows={pad_rows} (write {pad_rows*H*2/1e6:.1f} MB)")
        print(f"upstream reference (8xB300, same shape, 32 SM, 3-run median):")
        print("  planning 60.65  dispatch_f 958.0  epilogue_f 93.66")
        print("  comb_prolog_f 137.1  combine_f 836.6  prefetch 161.5")
        print()
        print(f"{'measurement':<38}{'mean us':>10}{'max us':>10}"
              f"{'logical GB/s':>14}{'wire GB/s':>12}")
        for key in sorted(results):
            mean_us, max_us = results[key]
            if "prefetch" in key:
                lo, wi = prefetch_bytes, prefetch_bytes
            elif "combine_ideal_dst" in key:
                lo, wi = logical_bytes, logical_bytes * (R - 1) // R
            elif "ideal_dst" in key:
                lo, wi = logical_bytes, ideal_wire_bytes
            else:
                lo, wi = logical_bytes, wire_bytes
            print(
                f"{key:<38}{mean_us:>10.1f}{max_us:>10.1f}"
                f"{lo / mean_us / 1e3:>14.1f}{wi / mean_us / 1e3:>12.1f}"
            )
        print()
        print("MOONEP_DISPATCH_ALIGNED_JSON " + json.dumps(
            {k: {"mean_us": v[0], "max_us": v[1]} for k, v in results.items()}
        ))

    dist.barrier()
    ms.shmem_barrier_all()
    ms.shmem_finalize()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
