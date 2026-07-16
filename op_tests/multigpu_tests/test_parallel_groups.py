# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Test script to verify TP/EP/DP/PP/PCP parallel group initialization and
communication operations in aiter.

Layout order: ExternalDP x DP x PP x PCP x TP
  - ep_size = dp * pcp * tp
  - world_size = external_dp * dp * pp * pcp * tp
  - PCP (Prefill Context Parallel) is an INDEPENDENT dimension that grows
    world_size (world = ... x pcp x tp) and sits just outside TP (mirrors
    vLLM). It is NOT decode context parallel (DCP, which reuses TP GPUs).

Examples:
    # tp=2, dp=2, pp=1, world_size=4 (external_dp=1), ep=4
    python test_parallel_groups.py --tp 2 --dp 2 --pp 1

    # tp=2, dp=2, pp=1, world_size=8 (external_dp=2), ep=4
    python test_parallel_groups.py --tp 2 --dp 2 --pp 1 -w 8

    # tp=2, dp=2, pp=2, world_size=8 (external_dp=1), ep=4
    python test_parallel_groups.py --tp 2 --dp 2 --pp 2

    # PCP: tp=2, pcp=2, world_size=4 (default for PCP testing)
    python test_parallel_groups.py --tp 2 --pcp 2 --dp 1

    # PCP: tp=4, pcp=2, world_size=8
    python test_parallel_groups.py --tp 4 --pcp 2 --dp 1

    # pcp=1 regression: 5D layout must match the old 4D TP/PP/DP/EP grouping
    python test_parallel_groups.py --tp 8 --pcp 1 --dp 1
"""

import argparse
import logging
import os
from multiprocessing import Pool, freeze_support, set_start_method

import torch
import torch.distributed as dist

from aiter import dtypes
from aiter.dist.communication_op import (
    tensor_model_parallel_all_reduce,
    expert_parallel_all_reduce,
    data_parallel_all_reduce,
)
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    get_ep_group,
    get_dp_group,
    get_pp_group,
    get_pcp_group,
    get_prefill_context_model_parallel_rank,
    get_prefill_context_model_parallel_world_size,
    init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.test_common import checkAllclose

logger = logging.getLogger("aiter")

set_start_method("spawn", force=True)


def _worker(
    world_size,
    tp_size,
    pp_size,
    dp_size,
    pcp_size,
    rank,
    all_inputs,
    shape,
    dtype,
    distributed_init_method,
):
    """
    Each worker:
      1. Initializes distributed env with the given parallel config.
      2. Verifies group topology (sizes and member ranks).
      3. Runs all-reduce on TP / EP / DP groups via communication_op wrappers.
      4. Returns results for the main process to verify correctness.
    """
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method=distributed_init_method,
    )
    ensure_model_parallel_initialized(
        tp_size,
        pp_size,
        data_parallel_size=dp_size,
        prefill_context_model_parallel_size=pcp_size,
    )

    tp_grp = get_tp_group()
    ep_grp = get_ep_group()
    dp_grp = get_dp_group()
    pp_grp = get_pp_group()
    pcp_grp = get_pcp_group()

    group_info = {
        "tp_ranks": tp_grp.ranks,
        "tp_size": tp_grp.world_size,
        "ep_ranks": ep_grp.ranks,
        "ep_size": ep_grp.world_size,
        "dp_ranks": dp_grp.ranks,
        "dp_size": dp_grp.world_size,
        "pp_ranks": pp_grp.ranks,
        "pp_size": pp_grp.world_size,
        "pcp_ranks": pcp_grp.ranks,
        "pcp_size": pcp_grp.world_size,
        # accessor helpers (used by ATOM to gate PCP logic)
        "pcp_ws_accessor": get_prefill_context_model_parallel_world_size(),
        "pcp_rank_accessor": get_prefill_context_model_parallel_rank(),
    }

    x = all_inputs[rank].to(device)

    # warmup / sync
    dist.all_reduce(torch.zeros(1, device=device), group=tp_grp.device_group)
    torch.cuda.synchronize()

    tp_out = tensor_model_parallel_all_reduce(x.clone()).cpu()
    ep_out = expert_parallel_all_reduce(x.clone()).cpu()
    dp_out = data_parallel_all_reduce(x.clone()).cpu()

    results = {
        "group_info": group_info,
        "tp_out": tp_out,
        "ep_out": ep_out,
        "dp_out": dp_out,
    }

    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()

    return results


def _compute_group_allreduce_ref(all_inputs, group_ranks):
    """Sum inputs for all ranks that belong to the same group."""
    ref = torch.zeros_like(all_inputs[0])
    for r in group_ranks:
        ref += all_inputs[r]
    return ref


def _ref_groups_5d(world, dp, pp, pcp, tp):
    """Expected group membership, mirroring initialize_model_parallel's 5D
    layout [ExtDP, DP, PP, PCP, TP]. Ground truth for PCP topology asserts.
    """
    all_ranks = torch.arange(world).reshape(-1, dp, pp, pcp, tp)
    return {
        "tp": [x.tolist() for x in all_ranks.view(-1, tp).unbind(0)],
        "pcp": [
            x.tolist() for x in all_ranks.transpose(3, 4).reshape(-1, pcp).unbind(0)
        ],
        "dp": [x.tolist() for x in all_ranks.transpose(1, 4).reshape(-1, dp).unbind(0)],
        "ep": [
            x.tolist()
            for x in all_ranks.transpose(1, 2).reshape(-1, dp * pcp * tp).unbind(0)
        ],
    }


def _ref_groups_4d_old(world, dp, pp, tp):
    """Old 4D layout [ExtDP, DP, PP, TP] (no PCP). Ground truth for the pcp=1
    regression check: with pcp=1 the 5D layout must reduce to exactly this.
    """
    all_ranks = torch.arange(world).reshape(-1, dp, pp, tp)
    return {
        "tp": [x.tolist() for x in all_ranks.view(-1, tp).unbind(0)],
        "pp": [x.tolist() for x in all_ranks.transpose(2, 3).reshape(-1, pp).unbind(0)],
        "dp": [x.tolist() for x in all_ranks.transpose(1, 3).reshape(-1, dp).unbind(0)],
        "ep": [
            x.tolist() for x in all_ranks.transpose(1, 2).reshape(-1, dp * tp).unbind(0)
        ],
    }


def _find_group(groups, rank):
    """Return the sorted member list of the group that contains `rank`."""
    for g in groups:
        if rank in g:
            return sorted(g)
    raise AssertionError(f"rank {rank} not found in any group: {groups}")


def test_parallel_groups(
    tp_size,
    pp_size,
    dp_size,
    pcp_size,
    shape,
    dtype,
    distributed_init_method,
    world_size=None,
):
    min_world_size = tp_size * pp_size * dp_size * pcp_size
    if world_size is None:
        world_size = min_world_size
    assert world_size >= min_world_size and world_size % min_world_size == 0, (
        f"world_size ({world_size}) must be a multiple of tp*pp*dp*pcp "
        f"({min_world_size})"
    )
    external_dp = world_size // min_world_size
    ep_size = dp_size * pcp_size * tp_size

    print(f"\n{'='*60}")
    print(
        f"  Config: tp={tp_size}, pp={pp_size}, dp={dp_size}, pcp={pcp_size} "
        f"=> ep={ep_size}"
    )
    print(f"  world_size={world_size}  external_dp={external_dp}")
    print(f"  shape={shape}  dtype={dtype}")
    print(f"{'='*60}")

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"

    all_inputs = [torch.randn(shape, dtype=dtype) for _ in range(world_size)]

    pool = Pool(processes=world_size)
    async_results = []
    for rank in range(world_size):
        async_results.append(
            pool.apply_async(
                _worker,
                args=(
                    world_size,
                    tp_size,
                    pp_size,
                    dp_size,
                    pcp_size,
                    rank,
                    all_inputs,
                    shape,
                    dtype,
                    distributed_init_method,
                ),
            )
        )
    pool.close()
    pool.join()

    results = [r.get() for r in async_results]
    ref5d = _ref_groups_5d(world_size, dp_size, pp_size, pcp_size, tp_size)

    # ---- Verify group topology ----
    print("\n[Group Topology]")
    for rank, res in enumerate(results):
        info = res["group_info"]
        assert (
            info["tp_size"] == tp_size
        ), f"Rank {rank}: expected tp_size={tp_size}, got {info['tp_size']}"
        assert (
            info["ep_size"] == ep_size
        ), f"Rank {rank}: expected ep_size={ep_size}, got {info['ep_size']}"
        assert (
            info["dp_size"] == dp_size
        ), f"Rank {rank}: expected dp_size={dp_size}, got {info['dp_size']}"
        assert (
            info["pp_size"] == pp_size
        ), f"Rank {rank}: expected pp_size={pp_size}, got {info['pp_size']}"
        assert (
            info["pcp_size"] == pcp_size
        ), f"Rank {rank}: expected pcp_size={pcp_size}, got {info['pcp_size']}"
        # PCP accessor helpers must agree with the group
        assert info["pcp_ws_accessor"] == pcp_size, (
            f"Rank {rank}: pcp_ws_accessor={info['pcp_ws_accessor']} "
            f"!= pcp_size={pcp_size}"
        )
        # PCP / TP / DP membership must match the 5D reference layout
        exp_pcp = _find_group(ref5d["pcp"], rank)
        exp_tp = _find_group(ref5d["tp"], rank)
        assert (
            sorted(info["pcp_ranks"]) == exp_pcp
        ), f"Rank {rank}: PCP ranks {sorted(info['pcp_ranks'])} != expected {exp_pcp}"
        assert (
            sorted(info["tp_ranks"]) == exp_tp
        ), f"Rank {rank}: TP ranks {sorted(info['tp_ranks'])} != expected {exp_tp}"
        # accessor rank == this rank's index inside its PCP group
        assert info["pcp_rank_accessor"] == exp_pcp.index(rank), (
            f"Rank {rank}: pcp_rank_accessor={info['pcp_rank_accessor']} "
            f"!= expected {exp_pcp.index(rank)}"
        )
        print(
            f"  Rank {rank}: TP{info['tp_ranks']}  EP{info['ep_ranks']}  "
            f"DP{info['dp_ranks']}  PP{info['pp_ranks']}  PCP{info['pcp_ranks']}"
        )
    print("  [PASS] Group sizes + PCP topology match expected values.\n")

    # ---- pcp=1 regression: 5D layout must equal the old 4D grouping ----
    if pcp_size == 1:
        print("[pcp=1 Regression vs old 4D layout]")
        ref_old = _ref_groups_4d_old(world_size, dp_size, pp_size, tp_size)
        for rank, res in enumerate(results):
            info = res["group_info"]
            for tag, key in [
                ("tp", "tp_ranks"),
                ("dp", "dp_ranks"),
                ("pp", "pp_ranks"),
                ("ep", "ep_ranks"),
            ]:
                got = sorted(info[key])
                exp = _find_group(ref_old[tag], rank)
                assert got == exp, (
                    f"Rank {rank} {tag.upper()}: {got} != old-4D {exp} "
                    f"(pcp=1 regression broken!)"
                )
        print("  [PASS] pcp=1 5D layout is identical to the old 4D TP/PP/DP/EP.\n")

    # ---- Verify all-reduce correctness ----
    all_pass = True
    for rank, res in enumerate(results):
        info = res["group_info"]

        tp_ref = _compute_group_allreduce_ref(all_inputs, info["tp_ranks"])
        ep_ref = _compute_group_allreduce_ref(all_inputs, info["ep_ranks"])
        dp_ref = _compute_group_allreduce_ref(all_inputs, info["dp_ranks"])

        for tag, ref, out in [
            ("TP", tp_ref, res["tp_out"]),
            ("EP", ep_ref, res["ep_out"]),
            ("DP", dp_ref, res["dp_out"]),
        ]:
            msg = f"Rank {rank} {tag} all-reduce  group={info[f'{tag.lower()}_ranks']}"
            err = checkAllclose(ref, out, msg=msg)
            if err != 0:
                all_pass = False

    if all_pass:
        print(f"\n  [ALL PASS] tp={tp_size}, dp={dp_size}, pp={pp_size}, ep={ep_size}")
    else:
        print("\n  [FAIL] Some checks failed!")
    print()


parser = argparse.ArgumentParser(description="Test TP/EP/DP/PP parallel groups")
parser.add_argument("--tp", type=int, default=2, help="tensor parallel size")
parser.add_argument("--dp", type=int, default=2, help="data parallel size")
parser.add_argument("--pp", type=int, default=1, help="pipeline parallel size")
parser.add_argument(
    "--pcp",
    type=int,
    default=1,
    help="prefill context parallel size (default 1 = no PCP). "
    "For PCP testing use e.g. --tp 2 --pcp 2 --dp 1.",
)
parser.add_argument(
    "-w",
    "--world-size",
    type=int,
    default=None,
    help="total world size (default: tp*dp*pp*pcp). "
    "Set larger to include external data parallelism.",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=["fp16", "bf16"],
    default="fp16",
    help="data type (default: fp16)",
)
parser.add_argument(
    "-s",
    "--shape",
    type=dtypes.str2tuple,
    default=(128, 4096),
    help="tensor shape, e.g. -s 128,4096",
)


if __name__ == "__main__":
    freeze_support()
    args = parser.parse_args()
    dtype = dtypes.d_dtypes[args.dtype]

    test_parallel_groups(
        tp_size=args.tp,
        pp_size=args.pp,
        dp_size=args.dp,
        pcp_size=args.pcp,
        shape=args.shape,
        dtype=dtype,
        distributed_init_method=get_distributed_init_method(get_ip(), get_open_port()),
        world_size=args.world_size,
    )
