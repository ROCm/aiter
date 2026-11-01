#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for the MegaMoE A8W4 and A4W4 deployments.

Each job builds the runtime's own ``MegaMoEV2`` for one rank of a ``mega_moe``
deployment of ``moe_deployments.json`` (see :mod:`aiter.aot.flydsl.deployments`)
and calls its ``preload_aot_bundles()``. The runtime derives the bundle plan,
sizes its buffers and loads every production Stage1/prepare, Stage2 and
combine variant itself; inside :func:`aiter.aot.flydsl.dry_run.dry_run` those
kernels are compiled into the cache instead of launched. mori shmem, through
which the runtime allocates symmetric buffers and exchanges peer pointers, hands
out FakeTensors and fake addresses, so no GPU or process group is needed.

aiter fixes some arch-dependent values at import (e.g. ``dtypes.fp8``), so like
the MoE replay this one runs in a process of its own, started with
``GPU_ARCHS`` naming gfx950 and ``AITER_AOT_REPLAY=1``. It is skipped when
``GPU_ARCHS`` names targets without gfx950.

Usage:
    # Every MegaMoE deployment the service ships (DeepSeek-V4-Pro and Kimi-K3)
    python -m aiter.aot.flydsl.mega_moe

    # Only some of them
    python -m aiter.aot.flydsl.mega_moe --experts-per-rank 48 --mtpr 8192

    # Deployments of your own
    python -m aiter.aot.flydsl.mega_moe --deployments /path/to/deployments.json
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

from aiter.aot.flydsl.common import (
    aot_target_archs,
    check_replay_process,
    replay_env,
    run_jobs_parallel,
)
from aiter.aot.flydsl.deployments import DEFAULT_DEPLOYMENTS, QUANTS
from aiter.aot.flydsl.deployments import load as load_deployments
from aiter_worker_limits import adopt_legacy_max_jobs

ARCH = "gfx950"
# The combine launch geometry is tuned per GPU model, read from the device name.
DEVICE_NAME = "AMD Instinct MI355X"
NUM_CU = 256


def deployment_jobs(
    paths=DEFAULT_DEPLOYMENTS,
    *,
    mtprs=None,
    quants=None,
    experts_per_ranks=None,
) -> list[dict]:
    """One job per rank of every ``mega_moe`` deployment of the files
    ``paths``, at each of its max tokens per rank; rank is a compile-time
    parameter of the kernels. ``mtprs``, ``quants`` and ``experts_per_ranks``
    narrow the jobs down to the values they list."""
    jobs = []
    for profile in load_deployments(paths).mega_moe:
        if quants and profile.quant not in quants:
            continue
        if experts_per_ranks and profile.experts_per_rank not in experts_per_ranks:
            continue
        model = profile.model
        for mtpr in profile.max_tokens_per_rank:
            if mtprs and mtpr not in mtprs:
                continue
            jobs += [
                {
                    "kernel_name": f"{profile.id}/mtpr{mtpr}/rank{rank}",
                    "quant": profile.quant,
                    "mtpr": mtpr,
                    "experts_per_rank": profile.experts_per_rank,
                    "rank": rank,
                    "world_size": profile.ep,
                    "topk": model.top_k,
                    "model_dim": model.model_dim,
                    "inter_dim": model.inter_dim,
                    "swiglu_limit": model.swiglu_limit,
                    "act": model.activation.lower(),
                    "situ_beta": model.situ_beta,
                    "situ_linear_beta": model.situ_linear_beta,
                }
                for rank in range(profile.ep)
            ]
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
    targets = aot_target_archs()
    if targets and ARCH not in targets:
        print(
            f"[aiter] FlyDSL MegaMoE replay: skipped, GPU_ARCHS omits {ARCH}",
            flush=True,
        )
        return 0
    command = [sys.executable, "-m", "aiter.aot.flydsl.mega_moe", *(argv or [])]
    return subprocess.run(command, env=replay_env(ARCH), check=False).returncode


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--deployments",
        nargs="+",
        default=DEFAULT_DEPLOYMENTS,
        help="MoE deployment file(s) whose mega_moe deployments to compile "
        "(default: moe_deployments.json)",
    )
    parser.add_argument(
        "--mtpr", type=int, nargs="+", help="only these max tokens per rank"
    )
    parser.add_argument(
        "--experts-per-rank",
        type=int,
        nargs="+",
        help="only deployments with these experts per rank "
        "(for example 48 52 56 for DeepSeek-V4-Pro r0/r32/r64)",
    )
    parser.add_argument("--quant", nargs="+", choices=QUANTS, help="only these quants")
    args = parser.parse_args()
    if os.environ.get("AITER_AOT_REPLAY") != "1":
        raise SystemExit(run(sys.argv[1:]))
    check_replay_process(ARCH)
    jobs = deployment_jobs(
        args.deployments,
        mtprs=args.mtpr,
        quants=args.quant,
        experts_per_ranks=args.experts_per_rank,
    )
    if not jobs:
        print("[aiter] FlyDSL MegaMoE replay: no deployment matches", flush=True)
        raise SystemExit(1)
    results = run_jobs_parallel(compile_one_config, jobs)
    failed = sum(result["compile_time"] is None for result in results)
    print(
        f"[aiter] FlyDSL MegaMoE replay: {len(results) - failed} ok, {failed} failed",
        flush=True,
    )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    adopt_legacy_max_jobs()
    main()
