# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Measure aiter distributed-init wall time (init_dist_env).

================================ WHY THIS EXISTS ================================
A customer wants faster service bring-up at peak hours, which put a hard
requirement on the startup speed of aiter's communication operators. The
suspected bottleneck is the distributed-init path:

    init_dist_env  (aiter/ops/communication.py)
      -> init_distributed_environment   (builds the world group)
      -> ensure_model_parallel_initialized
           -> initialize_model_parallel  (aiter/dist/parallel_state.py)
                builds one GroupCoordinator per parallel dimension:
                TP, DCP, PCP, PP, DP, EP

Working hypotheses about where the time goes (to be CONFIRMED by this script's
numbers before any code is changed):

  1. new_group storm. initialize_model_parallel unconditionally creates 6
     dimension groups, and each GroupCoordinator calls new_group TWICE per
     subgroup (once NCCL device group, once gloo CPU group). new_group is a
     *collective over the whole world* even for size==1 dims, so in an MoE
     deployment DCP/PCP/PP (all size 1) cost ~6 wasted global new_group calls
     (gloo ones are the slow part).
  2. Per-dim device communicators. With TP+DP/EP (MoE), TP, DP and EP each
     build a full CudaCommunicator = PyNccl init + CustomAllreduce (IPC handle
     exchange + init_custom_ar) + QuickAllReduce. Three sets, mostly necessary
     work, but heavy.
  3. in_the_same_node_as is called 6x (CustomAllreduce + QuickAllReduce, once
     each, across the 3 device-comm groups). Each call does a
     broadcast_object_list + shared-memory create/unlink + barrier + all_reduce.
     The result is identical for ranks on the same node, so 5 of the 6 are
     redundant. (This barrier inside in_the_same_node_as is almost certainly the
     "barrier at the end of init_env" that felt expensive -- there is no
     separate explicit barrier in init_dist_env.)

Customer topology is TP + DP/EP (MoE), single node, multi-GPU. aiter
communication ops are NEVER cross-machine -- always single-node multi-GPU.

This script drives the REAL entry point (init_dist_env) and prints a
hierarchical breakdown (via AITER_INIT_TIMING) so the hypotheses above can be
checked against real numbers. The timing harness itself lives in
aiter/dist/init_timing.py; instrumentation points are in parallel_state.py and
communication.py (search for `timed(`).

=================================== HOW TO RUN =================================
Launch (single node, N GPUs); pure TP is the most robust mode -- run it first
to get a clean baseline:

    AITER_INIT_TIMING=1 torchrun --nproc_per_node=8 \
        op_tests/multigpu_tests/test_init_timing.py

Topology is taken from env (defaults to pure TP = world_size):

    TP=8                      # tensor parallel size      (default: world_size)
    DP=1                      # data parallel size        (default: 1)
    PCP=1                     # prefill context parallel  (default: 1)
    DCP=1                     # decode context parallel   (default: 1)
    REPEATS=1                 # cold-init iterations       (default: 1)

For a TP4 x DP2 (MoE-like) run on 8 GPUs:

    AITER_INIT_TIMING=1 TP=4 DP=2 torchrun --nproc_per_node=8 \
        op_tests/multigpu_tests/test_init_timing.py

================================ HOW TO READ IT ================================
Two layers of output:
  * End-to-end: this script times init_dist_env per rank and prints
    min/max/avg + skew across ranks.
  * Breakdown: rank 0 prints a nested tree (only when AITER_INIT_TIMING is set)
    showing each region's wall time. Look for:
      - init_group:dcp / :pcp / :pp  -> should be ~0 if hypothesis 1 is right
        these are the size==1 wasted groups (candidate: skip by size condition).
      - in_the_same_node_as           -> appears ~6x; sum them to size the
        redundant-detection cost (candidate: cache / single-node short-circuit).
      - device_communicator:tp / :dp / :ep -> the three heavy comm builds.
      - new_group:*:cpu(gloo) vs :device -> how much the gloo groups cost.

============================== AGREED CONSTRAINTS ==============================
Decisions already made with the author, to respect when proposing fixes later:
  - Do NOT hard-code skipping of DCP/PCP/PP. Skip by a size==1 condition only,
    so PP>1 etc. still work in the future.
  - aiter comm is single-node multi-GPU only (never cross-machine) -- a
    single-node short-circuit for in_the_same_node_as is acceptable.
  - The DP group DOES need CustomAllreduce/QuickAllReduce -- do not drop its
    device communicator.

=================================== CAVEATS ===================================
- world_size must equal TP * DP * PCP (PP is forced to 1 by init_dist_env).
- Only the *cold* (first) init is representative of cluster startup; that is the
  number the customer cares about. REPEATS>1 is supported for variance checking
  but each repeat tears everything down first (destroy_dist_env) so every
  iteration measures a cold init.
- DP>1 binding: init_dist_env binds the signal tensor to device=rankID where
  rankID is the DP-local rank. On a single node this assumes each process sees
  its target GPU as cuda:dp_local_rank (e.g. via CUDA_VISIBLE_DEVICES per
  process). Pure-TP (DP=1) needs no special binding and is the most robust mode
  for this measurement.
- Uses init_method="env://", so it MUST be launched via torchrun (which sets
  RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT).
"""

import os
import sys
import time

import torch
import torch.distributed as dist


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size < 2:
        print("SKIP: need torchrun --nproc_per_node>=2")
        sys.exit(0)

    tp = _env_int("TP", world_size)
    dp = _env_int("DP", 1)
    pcp = _env_int("PCP", 1)
    dcp = _env_int("DCP", 1)
    repeats = _env_int("REPEATS", 1)

    expected_world = tp * dp * pcp
    if expected_world != world_size:
        if rank == 0:
            print(
                f"ERROR: TP({tp}) * DP({dp}) * PCP({pcp}) = {expected_world} "
                f"!= WORLD_SIZE({world_size}). Adjust TP/DP/PCP env or "
                f"--nproc_per_node."
            )
        sys.exit(1)

    if os.environ.get("AITER_INIT_TIMING", "0") in ("0", "", "false", "False"):
        if rank == 0:
            print(
                "WARNING: AITER_INIT_TIMING is not set; the per-region "
                "breakdown will be empty. Re-run with AITER_INIT_TIMING=1."
            )

    torch.cuda.set_device(local_rank)

    from aiter.ops.communication import init_dist_env, destroy_dist_env
    from aiter.dist.init_timing import reset_init_timing

    if rank == 0:
        print(
            f"\n[init-timing] world_size={world_size} "
            f"TP={tp} DP={dp} PCP={pcp} DCP={dcp} repeats={repeats}\n"
        )

    for it in range(repeats):
        # Make sure nothing is left over from a previous iteration. This fully
        # tears down the process group so every iteration measures a *cold*
        # init (the case the customer actually cares about at cluster startup).
        destroy_dist_env()
        reset_init_timing()

        # Wall-clock around the whole entry point, measured per rank. The
        # in-function report (AITER_INIT_TIMING) gives the breakdown; this
        # gives an independent end-to-end number that includes process-group
        # bring-up.
        # init_distributed_environment computes the global rank as
        #   data_parallel_rank * (tp*pcp) + rankID
        # so rankID must be the rank *within* the DP group, not the global rank.
        dp_local_rank = rank % (tp * pcp)
        data_parallel_rank = rank // (tp * pcp)

        t0 = time.perf_counter()
        init_dist_env(
            tensor_model_parallel_size=tp,
            rankID=dp_local_rank,
            local_rank=local_rank,
            data_parallel_size=dp,
            data_parallel_rank=data_parallel_rank,
            decode_context_parallel_size=dcp,
            prefill_context_model_parallel_size=pcp,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1e3

        # Gather every rank's end-to-end time so we can see skew.
        times = [None] * world_size
        dist.all_gather_object(times, elapsed_ms)
        if rank == 0:
            tmin, tmax = min(times), max(times)
            avg = sum(times) / len(times)
            print(
                f"[init-timing] iter {it}: end-to-end init_dist_env "
                f"min={tmin:.1f} max={tmax:.1f} avg={avg:.1f} ms "
                f"(skew={tmax - tmin:.1f} ms)"
            )

    destroy_dist_env()
    if rank == 0:
        print("\n[init-timing] done")


if __name__ == "__main__":
    main()
