# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Combine variants: pull reference vs deduplicated pull vs push, with a real
correctness check.

Correctness
-----------
``benchmark_moonep_dispatch_aligned`` compared two GPU kernels against each
other and printed ``max|ref|=0.000e+00``.  The cause is that
``MoonEPPreplannedDispatchOp.combine`` only *compiles* on its first call --

    if self._combine_compiled is None:
        self._combine_compiled = flyc.compile(...)   # compiles, does not run
    else:
        self._combine_compiled(...)

-- so the reference snapshot was taken from a ``combine_output`` that had never
been written.  Both combine checks there were comparing zeros to zeros.  (The
timings are unaffected: one compile lands inside the 5 warmup calls.)

This script fixes it three ways at once:

* every launcher is primed twice, so the timed and checked calls both execute;
* expert outputs are staged with deterministic non-zero data, and the reference
  output is asserted non-trivial before anything is compared to it;
* the ground truth is computed independently in torch from an all-gather of
  every rank's staged rows, so all three GPU paths are checked against
  arithmetic rather than against each other.

Variants
--------
``pull_reference``  today's combine: K remote rows per token, ~232 GB/s.
``pull_dedup``      prologue folds duplicates locally, combine skips them
                    (upstream ``combine.py:320``).  Same direction, 1.51x fewer
                    bytes.
``push``            ``moonep_combine_push``: the rank holding the expert output
                    writes it home (remote *writes*, measured at 448 GB/s vs 235
                    GB/s for remote reads), then the owning rank reduces out of
                    its own HBM.  Same wire bytes as ``pull_dedup``.

Run under torchrun with one process per GPU::

    torchrun --standalone --nproc-per-node=8 \
        op_tests/benchmark_moonep_combine_push.py
"""

from __future__ import annotations

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
from aiter.ops.flydsl.moonep import (
    MoonEPPlanConfig,
    MoonEPReferencePlan,
    build_reference_plan,
)
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
BLOCK_NUMS = [
    int(x) for x in os.environ.get("MOONEP_BLOCK_NUMS", "256,1024").split(",")
]
ROUTING = os.environ.get("MOONEP_ROUTING", "uniform")
# Tokens per chunk when building the torch ground truth; bounds the temporary
# [chunk, K, H] fp32 tensor.
CHECK_CHUNK = int(os.environ.get("MOONEP_CHECK_CHUNK", "256"))
CHECK_ONLY = os.environ.get("MOONEP_CHECK_ONLY", "0") == "1"
# Keep going past a failed check so one run reports every variant.
DIAG = os.environ.get("MOONEP_DIAG", "0") == "1"


def time_gpu_op(launch_fn, group):
    """Cross-rank mean, same methodology as benchmark_moonep_dispatch_aligned."""

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
    allr = torch.cat(outs)
    return allr.mean().item(), allr.max().item()


class Launcher:
    """flyc.compile-then-call wrapper whose first invocation actually runs.

    The op's own lazy pattern compiles without executing, which is what made the
    old correctness check vacuous.
    """

    def __init__(self, jit, ptr_args, stream):
        self._jit = jit
        self._stream = stream
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
        token_padding=TOKEN_PADDING,
    )
    nvs = config.num_dispatch_rows
    n_entries = S * K

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

    # Symmetric buffers the push path needs, allocated in the same order on
    # every PE.  staging is [S*K, H]; only `primaries` of those rows are ever
    # written, so a prefix sum over is_primary would compact it -- see the
    # module docstring of moonep_combine_push.
    src_slot = mori_shmem_create_tensor((nvs,), torch.int32)
    staging = mori_shmem_create_tensor((n_entries, H), torch.bfloat16)
    torch.cuda.synchronize(device)
    ms.shmem_barrier_all()
    peer_src_slot_ptrs = torch.tensor(
        [ms.shmem_ptr_p2p(src_slot.data_ptr(), rank, p) for p in range(R)],
        dtype=torch.int64,
        device=device,
    )
    peer_staging_ptrs = torch.tensor(
        [ms.shmem_ptr_p2p(staging.data_ptr(), rank, p) for p in range(R)],
        dtype=torch.int64,
        device=device,
    )

    # Deterministic non-zero expert outputs.  The old harness left these at
    # whatever dispatch happened to leave behind and then snapshotted a buffer
    # that had never been written.
    hgen = torch.Generator(device="cpu").manual_seed(770077 + rank)
    staged_cpu = (torch.randn(nvs, H, generator=hgen) * 0.5).to(torch.bfloat16)
    staged = staged_cpu.to(device)

    raw_dst, is_primary = MoonEPReferencePlan.decode_dst(plan.dst)
    peer_of = (raw_dst // nvs).to(torch.int64)
    row_of = (raw_dst % nvs).to(torch.int64)

    def torch_ground_truth(all_staged):
        """Expected combine output, computed from arithmetic rather than from
        another kernel.  ``all_staged`` is [R, NvS, H] bf16 on device."""

        out = torch.empty(S, H, dtype=torch.bfloat16, device=device)
        for lo in range(0, S, CHECK_CHUNK):
            hi = min(lo + CHECK_CHUNK, S)
            p = peer_of[lo:hi].reshape(-1)
            r = row_of[lo:hi].reshape(-1)
            rows = all_staged[p, r].reshape(hi - lo, K, H).float()
            out[lo:hi] = rows.sum(dim=1).to(torch.bfloat16)
        return out

    results = {}
    checks = {}

    for block_num in BLOCK_NUMS:
        ep = MoonEPBF16ReferenceEP(
            config, H, I, dispatch_block_num=block_num, prefetch_block_num=block_num
        )
        home_gate, home_up, home_down = _identity_home_weights(device)
        ep.load_home_weights(home_gate, home_up, home_down)
        del home_gate, home_up, home_down
        torch.cuda.empty_cache()
        op = ep.dispatch_op
        stream = torch.cuda.current_stream(device)

        dup_count = torch.zeros(nvs, dtype=torch.int32, device=device)
        dup_list = torch.zeros(nvs * (K - 1), dtype=torch.int32, device=device)
        prologue = Launcher(
            make_moonep_combine_prologue_jit(
                hidden_dim=H,
                num_dispatch_rows=nvs,
                top_k=K,
                block_num=min(block_num, 1024),
            ),
            (
                # expert_output, not recv_hidden: peer_expert_output_ptrs is
                # mapped to expert_output (moonep_dispatch_op.py:104), so that
                # is the buffer combine actually reads.  Staging into
                # recv_hidden instead is what made the old reference all zeros.
                op.expert_output.data_ptr(),
                op.recv_duplicate_src.data_ptr(),
                dup_count.data_ptr(),
                dup_list.data_ptr(),
            ),
            stream,
        )
        publish = Launcher(
            make_moonep_publish_src_slots_jit(
                num_tokens=S,
                top_k=K,
                num_dispatch_rows=nvs,
                rank=rank,
                block_num=min(block_num, 256),
            ),
            (plan.dst.data_ptr(), peer_src_slot_ptrs.data_ptr()),
            stream,
        )
        push = Launcher(
            make_moonep_push_rows_jit(
                hidden_dim=H,
                num_dispatch_rows=nvs,
                num_tokens=S,
                top_k=K,
                block_num=block_num,
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
                num_tokens=S,
                hidden_dim=H,
                top_k=K,
                num_dispatch_rows=nvs,
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
        # The reference combine goes through our own launcher rather than
        # op.combine: the op compiles-without-running on first call, which is
        # the bug that made the previous harness compare zeros to zeros.  Same
        # builder and same arguments the op uses (moonep_dispatch_op.py:137).
        ref_combine = Launcher(
            make_moonep_combine_jit(
                num_tokens=S,
                hidden_dim=H,
                top_k=K,
                num_dispatch_rows=nvs,
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
        dedup_combine = Launcher(
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

        # One real dispatch, purely for its metadata: recv_duplicate_src is what
        # tells the prologue which rows are duplicates, and it stays -1 until
        # dispatch writes it.  Without this the prologue would be a silent no-op
        # and the dedup/push checks would pass for the wrong reason.  Called
        # twice because the op's first call only compiles.
        op.dispatch(hidden, route_weights, plan)
        op.dispatch(hidden, route_weights, plan)
        torch.cuda.synchronize(device)
        dist.barrier()
        n_dup_rows = int((op.recv_duplicate_src >= 0).sum().item())
        if n_dup_rows == 0:
            raise AssertionError(
                "recv_duplicate_src has no duplicates; dispatch metadata never "
                "landed, so the dedup path would be untested"
            )
        if rank == 0:
            print(
                f"[setup] bn{block_num}: {n_dup_rows} duplicate rows staged "
                f"for the prologue",
                flush=True,
            )

        def restage():
            """Reset every rank's expert output to the pristine staged data.

            The prologue mutates expert_output in place (that is the point), and
            peers read it, so this has to be collective.  recv_duplicate_src and
            recv_route_weights are left alone -- they are dispatch metadata.
            """
            op.expert_output.copy_(staged)
            src_slot.fill_(-1)
            torch.cuda.synchronize(device)
            dist.barrier()
            ms.shmem_barrier_all()

        # ---- variants -------------------------------------------------
        def pull_reference():
            ms.shmem_barrier_on_stream(stream)
            ref_combine()

        def pull_dedup():
            prologue()
            ms.shmem_barrier_on_stream(stream)
            dedup_combine()

        def push_combine():
            prologue()
            push()
            ms.shmem_barrier_on_stream(stream)
            reduce_local()

        # Assert the buffer we stage into really is the one combine reads,
        # rather than trusting the mapping.  This is the exact confusion that
        # produced the old all-zero reference.
        if (
            op.peer_expert_output_ptrs[rank].item()
            != op.expert_output.data_ptr()
        ):
            raise AssertionError(
                "peer_expert_output_ptrs[self] does not alias expert_output; "
                "the combine payload buffer moved and staging would be silent"
            )
        if rank == 0:
            print(
                f"[setup] NvS={nvs} entries={n_entries} "
                f"expert_output={op.expert_output.data_ptr():#x}",
                flush=True,
            )

        # ---- correctness ----------------------------------------------
        restage()
        all_staged = torch.empty(R, nvs, H, dtype=torch.bfloat16, device=device)
        dist.all_gather_into_tensor(all_staged, staged)
        expected = torch_ground_truth(all_staged)
        del all_staged
        torch.cuda.empty_cache()
        exp_scale = expected.float().abs().max().item()
        if exp_scale <= 0:
            raise AssertionError("ground truth is all zeros; staging never landed")

        def check(name, fn, tol, stage_fn=None):
            (stage_fn or restage)()
            op.combine_output.zero_()
            torch.cuda.synchronize(device)
            dist.barrier()
            # Exactly one call: the prologue folds duplicates into their
            # primary in place, so a second call would fold them again.
            pre_recv = op.expert_output.float().abs().max().item()
            fn()
            torch.cuda.synchronize(device)
            dist.barrier()
            got = op.combine_output.float()
            if rank == 0:
                print(
                    f"[diag] {name}: |expert_out|={pre_recv:.3e} "
                    f"|staging|={staging.float().abs().max().item():.3e} "
                    f"|src_slot>=0|={int((src_slot >= 0).sum().item())} "
                    f"|out|={got.abs().max().item():.3e} "
                    f"|expected|={exp_scale:.3e} "
                    # gathered weights are written by the same kernel from a
                    # different code path: non-zero here with a zero payload
                    # means the kernel ran and the gather loop is at fault;
                    # zero here means it never launched.
                    f"|gathered_w|="
                    f"{op.gathered_route_weights.abs().max().item():.3e} "
                    f"|recv_w|={op.recv_route_weights.abs().max().item():.3e}",
                    flush=True,
                )
            if got.abs().max().item() <= 0 and not DIAG:
                raise AssertionError(f"{name}: output is all zeros, nothing ran")
            err = (expected.float() - got).abs().max().item()
            checks[f"{name}/bn{block_num}"] = {
                "max_abs_err": err,
                "rel": err / exp_scale,
                "max_abs_ref": exp_scale,
            }
            ok = err <= tol * exp_scale
            if not ok and DIAG:
                if rank == 0:
                    print(f"[check] bn{block_num} {name}: FAIL (diag mode, continuing)")
                return
            if rank == 0:
                print(
                    f"[check] bn{block_num} {name:<15} "
                    f"max|err|={err:.3e}  max|ref|={exp_scale:.3e}  "
                    f"rel={err / exp_scale:.2e}  {'OK' if ok else 'FAIL'}",
                    flush=True,
                )
            if not ok:
                raise AssertionError(
                    f"rank{rank} {name}: rel error {err / exp_scale:.3e} > {tol}"
                )

        # publish is per-plan, not per-combine; the push variants need it staged
        # before their first run.
        restage()
        publish()
        torch.cuda.synchronize(device)
        dist.barrier()
        published = src_slot.clone()

        def restage_pushed():
            restage()
            src_slot.copy_(published)
            torch.cuda.synchronize(device)
            dist.barrier()
            ms.shmem_barrier_all()

        # pull_reference sums all K rows in fp32 -> should match the ground
        # truth to bf16 rounding only.
        check("pull_reference", pull_reference, 2e-2)
        # dedup and push fold duplicates in the prologue, which rounds to bf16
        # once more than the reference does, so they get a looser bound.
        check("pull_dedup", pull_dedup, 5e-2)
        # push additionally needs src_slot restored, since restage clears it.
        check("push", push_combine, 5e-2, stage_fn=restage_pushed)

        if CHECK_ONLY:
            ep.close()
            del ep
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            dist.barrier()
            continue

        # ---- timing ----------------------------------------------------
        restage_pushed()
        for name, fn in (
            ("pull_reference", pull_reference),
            ("pull_dedup", pull_dedup),
            ("push_total", push_combine),
            ("push_publish_per_plan", publish),
            ("push_rows_only", push),
            ("push_reduce_only", reduce_local),
            ("prologue_only", prologue),
        ):
            mean_us, max_us = time_gpu_op(fn, dist.group.WORLD)
            results[f"{name}/bn{block_num}"] = (mean_us, max_us)
            torch.cuda.synchronize(device)
            dist.barrier()

        ep.close()
        del ep
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        dist.barrier()

    if rank == 0:
        primaries = int(is_primary.sum().item())
        local_rows = int(((peer_of == rank) & is_primary).sum().item())
        dedup_bytes = (primaries - local_rows) * H * 2
        full_bytes = (n_entries - n_entries // R) * H * 2
        local_reduce_bytes = primaries * H * 2 + S * H * 2

        print()
        print(
            f"shape S={S} H={H} K={K} E={E} R={R} routing={ROUTING} NvS={nvs}"
        )
        print(f"warmup={WARMUP} iters={ITERS} (eager, cross-rank mean)")
        print(
            f"wire bytes/rank: all-K {full_bytes / 1e6:.1f} MB, "
            f"dedup'd {dedup_bytes / 1e6:.1f} MB "
            f"(primaries {primaries}, local {local_rows})"
        )
        print(f"push staging = {n_entries * H * 2 / 1e6:.0f} MB symmetric")
        print()
        print("measured link ceilings on this machine (moonep_link_probe):")
        print("  remote read  235 GB/s (flat, depth 1..32, cache policy 0..19)")
        print("  remote write 448 GB/s (flat, depth 1..16)")
        print("upstream 8xB300 same shape: comb_prolog 137.1 + combine 836.6")
        print()
        print(f"{'measurement':<34}{'mean us':>10}{'max us':>10}{'wire GB/s':>12}")
        for key in sorted(results):
            mean_us, max_us = results[key]
            if key.startswith("pull_reference"):
                wire = full_bytes
            elif key.startswith(("pull_dedup", "push_total", "push_rows_only")):
                wire = dedup_bytes
            elif key.startswith("push_reduce_only"):
                wire = local_reduce_bytes
            else:
                wire = 0
            gb = wire / mean_us / 1e3 if wire else float("nan")
            print(f"{key:<34}{mean_us:>10.1f}{max_us:>10.1f}{gb:>12.1f}")
        print()
        print("MOONEP_COMBINE_PUSH_JSON " + json.dumps(
            {
                "timing": {
                    k: {"mean_us": v[0], "max_us": v[1]} for k, v in results.items()
                },
                "checks": checks,
            }
        ))

    dist.barrier()
    ms.shmem_barrier_all()
    mori_shmem_free_tensor(staging)
    mori_shmem_free_tensor(src_slot)
    ms.shmem_finalize()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
