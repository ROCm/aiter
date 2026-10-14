#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for the MegaMoE A8W4 deployment profiles.

Each job builds the runtime's own ``MegaMoEV2`` for one rank of a deployment
profile and calls its ``preload_aot_bundles()``. The runtime derives the bundle
plan, sizes its buffers and loads every production Stage1/prepare, Stage2 and
combine variant itself; inside :func:`aiter.aot.flydsl.dry_run.dry_run` those
kernels are compiled into the cache instead of launched. mori shmem, through
which the runtime allocates symmetric buffers and exchanges peer pointers, hands
out FakeTensors and fake addresses, so no GPU or process group is needed. This
module only owns the deployment profiles.

aiter fixes some arch-dependent values at import (e.g. ``dtypes.fp8``), so like
the MoE replay this one runs in a process of its own, with the GPUs hidden and
``GPU_ARCHS`` naming gfx950 before aiter is imported.

Usage:
    # Every profile the service ships (DeepSeek-V4-Pro and Kimi-K3)
    python -m aiter.aot.flydsl.mega_moe

    # One custom profile
    python -m aiter.aot.flydsl.mega_moe --experts-per-rank 48 --mtpr 8192
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from types import ModuleType
from unittest import mock

import torch

from aiter.aot.flydsl.common import replay_env, run_jobs_parallel

ARCH = "gfx950"
# The combine launch geometry is tuned per GPU model, read from the device name.
DEVICE_NAME = "AMD Instinct MI355X"
NUM_CU = 256
DEFAULT_MTPRS = (8192, 16384, 32768)
DEFAULT_QUANTS = ("a8w4",)
# DeepSeek-V4-Pro deployment profiles: r0, r32, and r64 redundant experts.
# Keep all three in the default AOT job set so the service never falls back to
# an online compile merely because EPLB changes the physical expert count.
DEFAULT_EXPERTS_PER_RANKS = (48, 52, 56)
# Kimi-K3 deployment profile (EP8 / epr112, topk16, d3584 / i3072).  Added to
# the default AOT job set so K3 serving never falls back to an online compile.
K3_EXPERTS_PER_RANKS = (112,)
K3_TOPK = 16
K3_MODEL_DIM = 3584
K3_INTER_DIM = 3072
# K3 uses SiTUv2 (hidden_act="situ"); tanh self-saturates so no hard clamp.
# beta=4.0 / linear_beta=25.0 are the K3 checkpoint constants the MegaMoE SiTU
# kernel bakes in (SGLang kimi_k3.py asserts exactly these).
K3_ACT = "situv2"
K3_SITU_BETA = 4.0
K3_SITU_LINEAR_BETA = 25.0
WORLD_SIZE = 8
TOPK = 6
MODEL_DIM = 7168
INTER_DIM = 3072
SWIGLU_LIMIT = 10.0


def default_jobs(
    mtprs=DEFAULT_MTPRS,
    experts_per_ranks=DEFAULT_EXPERTS_PER_RANKS,
    *,
    quants=DEFAULT_QUANTS,
    world_size=WORLD_SIZE,
    topk=TOPK,
    model_dim=MODEL_DIM,
    inter_dim=INTER_DIM,
    swiglu_limit=SWIGLU_LIMIT,
    act="silu",
    situ_beta=1.0,
    situ_linear_beta=1.0,
):
    """One job per rank of every (max tokens per rank, experts per rank) pair;
    rank is a compile-time parameter of the kernels."""
    shape_suffix = ""
    if (world_size, topk, model_dim, inter_dim) != (
        WORLD_SIZE,
        TOPK,
        MODEL_DIM,
        INTER_DIM,
    ):
        shape_suffix = f"_w{world_size}_k{topk}_d{model_dim}_i{inter_dim}"
    act_suffix = "" if act == "silu" else f"_{act}"
    return [
        {
            "kernel_name": (
                f"mega_moe_{quant}_mtpr{mtpr}_epr{experts_per_rank}"
                f"_rank{rank}{shape_suffix}{act_suffix}"
            ),
            "quant": quant,
            "mtpr": mtpr,
            "experts_per_rank": experts_per_rank,
            "rank": rank,
            "world_size": world_size,
            "topk": topk,
            "model_dim": model_dim,
            "inter_dim": inter_dim,
            "swiglu_limit": swiglu_limit,
            "act": act,
            "situ_beta": situ_beta,
            "situ_linear_beta": situ_linear_beta,
        }
        for mtpr in mtprs
        for experts_per_rank in experts_per_ranks
        for quant in quants
        for rank in range(world_size)
    ]


def production_jobs(
    mtprs=DEFAULT_MTPRS,
    *,
    quants=DEFAULT_QUANTS,
    world_size=WORLD_SIZE,
    swiglu_limit=SWIGLU_LIMIT,
):
    """Every geometry the service ships: V4-Pro (default shape) + Kimi-K3.

    Used by the default build so both profiles are AOT-covered and neither
    triggers an online JIT compile at serving time.
    """
    jobs = default_jobs(
        mtprs, quants=quants, world_size=world_size, swiglu_limit=swiglu_limit
    )
    jobs += default_jobs(
        mtprs,
        K3_EXPERTS_PER_RANKS,
        quants=quants,
        world_size=world_size,
        topk=K3_TOPK,
        model_dim=K3_MODEL_DIM,
        inter_dim=K3_INTER_DIM,
        swiglu_limit=0.0,
        act=K3_ACT,
        situ_beta=K3_SITU_BETA,
        situ_linear_beta=K3_SITU_LINEAR_BETA,
    )
    return jobs


def _mori_unavailable(*_args, **_kwargs):
    raise RuntimeError("mori is not installed")


@contextmanager
def _fake_shmem(device):
    """mori shmem that hands out FakeTensors and fake peer addresses."""
    if sys.modules.get("mori") is None and importlib.util.find_spec("mori") is None:
        # The replay never reaches mori itself, so a build host need not have
        # it. The stand-in then stays, since modules imported under it keep it.
        package, shmem = ModuleType("mori"), ModuleType("mori.shmem")
        package.shmem = shmem
        shmem.shmem_barrier_all = _mori_unavailable
        shmem.shmem_ptr_p2p = _mori_unavailable
        shmem.mori_shmem_create_tensor = _mori_unavailable
        sys.modules.update({"mori": package, "mori.shmem": shmem})
    import mori.shmem as ms

    from aiter.ops.flydsl.kernels import (
        flydsl_dispatch_combine_intranode_op as dispatch_combine,
    )

    def create_tensor(shape, dtype):
        return torch.empty(shape, dtype=dtype, device=device)

    with (
        mock.patch.object(ms, "shmem_barrier_all", lambda: None),
        mock.patch.object(ms, "shmem_ptr_p2p", lambda ptr, _rank, _pe: ptr),
        mock.patch.object(dispatch_combine, "mori_shmem_create_tensor", create_tensor),
    ):
        yield


def _weights(job: dict, device) -> dict:
    """Expert weights as the runtime takes them: preshuffled MXFP4 values with
    E8M0 scales."""
    from aiter import dtypes
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    experts = job["experts_per_rank"]
    model_dim, inter_dim = job["model_dim"], job["inter_dim"]

    def empty(*shape, dtype):
        return torch.empty(shape, dtype=dtype, device=device)

    w1 = empty(experts, 2 * inter_dim, model_dim // 2, dtype=dtypes.fp4x2)
    w1_scale = empty(experts * 2 * inter_dim, model_dim // 32, dtype=dtypes.fp8_e8m0)
    w2 = empty(experts, model_dim, inter_dim // 2, dtype=dtypes.fp4x2)
    w2_scale = empty(experts * model_dim, inter_dim // 32, dtype=dtypes.fp8_e8m0)
    return {
        "w1": shuffle_weight_a16w4(w1, 16, True),
        "w1_scale": shuffle_scale_a16w4(w1_scale, experts, True),
        "w2": shuffle_weight_a16w4(w2, 16, False),
        "w2_scale": shuffle_scale_a16w4(w2_scale, experts, False),
    }


def compile_one_config(**job):
    """Build one rank of a profile and leave its production kernels in the cache."""
    from aiter.aot.flydsl.dry_run import dry_run

    result = {**job, "compile_time": None}
    started = time.time()
    try:
        device = torch.device("cuda", job["rank"])
        with dry_run(ARCH, NUM_CU, name=DEVICE_NAME), _fake_shmem(device):
            # Imported only now: without mori, its stand-in must be in place.
            from aiter.ops.flydsl.kernels.mega_moe.mega_moe_v2 import MegaMoEV2

            moe = MegaMoEV2(
                rank=job["rank"],
                world_size=job["world_size"],
                model_dim=job["model_dim"],
                inter_dim=job["inter_dim"],
                experts=job["experts_per_rank"] * job["world_size"],
                topk=job["topk"],
                quant=job["quant"],
                max_tok_per_rank=job["mtpr"],
                swiglu_limit=job["swiglu_limit"],
                act=job["act"],
                situ_beta=job["situ_beta"],
                situ_linear_beta=job["situ_linear_beta"],
                **_weights(job, device),
            )
            moe.preload_aot_bundles()
            # preload_aot_bundles() leaves out quantize(), which fixed-slot
            # forward and the per-stage benchmarks call.
            moe.quantize(
                torch.empty((2, job["model_dim"]), dtype=torch.bfloat16, device=device)
            )
        result["compile_time"] = time.time() - started
    except Exception as error:  # noqa: BLE001
        print(f"  [FAIL] {job['kernel_name']}: {type(error).__name__}: {error}")
    return result


def run(argv: list[str] | None = None) -> int:
    """Replay in a process of its own; returns its exit status."""
    command = [sys.executable, "-m", "aiter.aot.flydsl.mega_moe", *(argv or [])]
    return subprocess.run(command, env=replay_env(ARCH), check=False).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mtpr", type=int, nargs="+", default=list(DEFAULT_MTPRS))
    parser.add_argument(
        "--experts-per-rank",
        type=int,
        nargs="+",
        default=list(DEFAULT_EXPERTS_PER_RANKS),
        help="deployment profiles to compile (for example 48 52 56 for r0/r32/r64)",
    )
    parser.add_argument("--world-size", type=int, default=WORLD_SIZE)
    parser.add_argument("--topk", type=int, default=TOPK)
    parser.add_argument("--model-dim", type=int, default=MODEL_DIM)
    parser.add_argument("--inter-dim", type=int, default=INTER_DIM)
    parser.add_argument("--swiglu-limit", type=float, default=SWIGLU_LIMIT)
    parser.add_argument("--act", default="silu", choices=("silu", "situv2"))
    parser.add_argument("--situ-beta", type=float, default=1.0)
    parser.add_argument("--situ-linear-beta", type=float, default=1.0)
    parser.add_argument(
        "--quant",
        nargs="+",
        choices=DEFAULT_QUANTS,
        default=list(DEFAULT_QUANTS),
    )
    args = parser.parse_args()
    if os.environ.get("GPU_ARCHS") != ARCH:
        raise SystemExit(run(sys.argv[1:]))
    default_shape = (
        list(args.experts_per_rank) == list(DEFAULT_EXPERTS_PER_RANKS)
        and args.topk == TOPK
        and args.model_dim == MODEL_DIM
        and args.inter_dim == INTER_DIM
    )
    if default_shape:
        # Default build ships both V4-Pro and Kimi-K3 so neither JITs online.
        jobs = production_jobs(
            tuple(args.mtpr),
            quants=tuple(args.quant),
            world_size=args.world_size,
            swiglu_limit=args.swiglu_limit,
        )
    else:
        jobs = default_jobs(
            tuple(args.mtpr),
            tuple(args.experts_per_rank),
            quants=tuple(args.quant),
            world_size=args.world_size,
            topk=args.topk,
            model_dim=args.model_dim,
            inter_dim=args.inter_dim,
            swiglu_limit=args.swiglu_limit,
            act=args.act,
            situ_beta=args.situ_beta,
            situ_linear_beta=args.situ_linear_beta,
        )
    results = run_jobs_parallel(compile_one_config, jobs)
    failed = sum(result["compile_time"] is None for result in results)
    print(
        f"[aiter] FlyDSL MegaMoE replay: {len(results) - failed} ok, {failed} failed",
        flush=True,
    )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
