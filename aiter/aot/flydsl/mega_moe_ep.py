#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for the fused EP MegaMoE (``MegaMoeEP``) deployment profiles.

Each job builds the runtime's own ``MegaMoeEP`` for one (profile, max tokens per
rank, quant) and calls its ``prepare()``: the collective serving runs before
graph capture to load every launch variant a token count in [0, mtpr] selects.
One ``forward()`` follows, for the activation quant it launches ahead of the
kernel. Inside :func:`aiter.aot.flydsl.dry_run.dry_run` those kernels are compiled into
the cache instead of launched. The hipIpc symmetric arena, through which the
engine maps its peers' buffers, hands out a FakeTensor and fake peer addresses,
so no GPU or process group is needed. This module only owns the deployment
profiles.

Rank, world size, the token count and the peer addresses are launch arguments
of the EP kernel, so one job per profile serves every rank.

The replay runs in a process of its own, started with
``GPU_ARCHS`` naming gfx950 and ``AITER_AOT_REPLAY=1``. It is skipped when
``GPU_ARCHS`` names targets without gfx950.

Usage:
    # Every profile the service ships (DeepSeek-V4-Pro and Kimi-K2.6)
    python -m aiter.aot.flydsl.mega_moe_ep

    # One custom profile
    python -m aiter.aot.flydsl.mega_moe_ep --experts-per-rank 48 --mtpr 8192

    # Check that a cache holds every kernel of the same jobs (compiles nothing)
    python -m aiter.aot.flydsl.mega_moe_ep --check
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from unittest import mock

import torch

from aiter.aot.flydsl.common import (
    aot_target_archs,
    check_replay_process,
    replay_env,
    run_jobs_parallel,
)
from aiter.aot.flydsl.mega_moe import ARCH, DEVICE_NAME, NUM_CU
from aiter_worker_limits import adopt_legacy_max_jobs

# Serving builds a decode-capacity instance (512 tokens per rank, the small-batch
# kernel) next to the full one.
DEFAULT_MTPRS = (512, 8192)
DEFAULT_QUANTS = ("a8w4", "a4w4")
WORLD_SIZE = 8
# DeepSeek-V4-Pro: r0, r32 and r64 redundant experts.
DEFAULT_EXPERTS_PER_RANKS = (48, 52, 56)
TOPK = 6
MODEL_DIM = 7168
INTER_DIM = 3072
SWIGLU_LIMIT = 10.0
# Kimi-K2.6 (EP8 / epr48, topk8, d7168 / i2048, unclamped SiLU), a8w4 only.
K26_EXPERTS_PER_RANKS = (48,)
K26_TOPK = 8
K26_INTER_DIM = 2048
K26_QUANTS = ("a8w4",)


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
):
    """One job per (max tokens per rank, experts per rank, quant)."""
    shape_suffix = ""
    if (world_size, topk, model_dim, inter_dim) != (
        WORLD_SIZE,
        TOPK,
        MODEL_DIM,
        INTER_DIM,
    ):
        shape_suffix = f"_w{world_size}_k{topk}_d{model_dim}_i{inter_dim}"
    return [
        {
            "kernel_name": (
                f"mega_moe_ep_{quant}_mtpr{mtpr}_epr{experts_per_rank}{shape_suffix}"
            ),
            "quant": quant,
            "mtpr": mtpr,
            "experts_per_rank": experts_per_rank,
            "world_size": world_size,
            "topk": topk,
            "model_dim": model_dim,
            "inter_dim": inter_dim,
            "swiglu_limit": swiglu_limit,
        }
        for mtpr in mtprs
        for experts_per_rank in experts_per_ranks
        for quant in quants
    ]


def production_jobs(mtprs=DEFAULT_MTPRS, *, quants=DEFAULT_QUANTS):
    """Every geometry the service ships: V4-Pro (default shape) + Kimi-K2.6."""
    jobs = default_jobs(mtprs, quants=quants)
    k26_quants = tuple(q for q in quants if q in K26_QUANTS)
    jobs += default_jobs(
        mtprs,
        K26_EXPERTS_PER_RANKS,
        quants=k26_quants,
        topk=K26_TOPK,
        inter_dim=K26_INTER_DIM,
        swiglu_limit=0.0,
    )
    return jobs


@contextmanager
def _fake_arena(rank, world_size):
    """The engine's hipIpc ``SymmetricArena`` as one rank's FakeTensor, whose
    address every peer base takes."""
    from aiter.ops.flydsl.kernels.mega_moe_ep import mega_moe_ep as engine
    from aiter.ops.flydsl.kernels.mega_moe_tp.mega_moe_tp import (
        _ALIGN,
        SymmetricArena,
    )

    class FakeArena(SymmetricArena):
        def __init__(self, *, group=None, device=None):
            self.group, self.device = group, device
            self.rank, self.world_size = rank, world_size
            self._slices, self._cursor = {}, 0
            self._storage, self._base_ptrs, self._opened = None, (), []

        def commit(self):
            total = -(-self._cursor // _ALIGN) * _ALIGN
            self._storage = torch.zeros(total, dtype=torch.uint8, device=self.device)
            for s in self._slices.values():
                s.local = self._storage[s.offset : s.offset + s.nbytes].view(s.dtype)
                s.local = s.local.view(s.shape)
            self._base_ptrs = (int(self._storage.data_ptr()),) * self.world_size
            return self

        def barrier(self):
            pass

    with mock.patch.object(engine, "SymmetricArena", FakeArena):
        yield


def _weights(job: dict, device) -> dict:
    """Expert weights as serving binds them: MXFP4 values with E8M0 scales,
    already preshuffled (the layout does not change a kernel)."""
    from aiter import dtypes

    experts = job["experts_per_rank"]
    model_dim, inter_dim = job["model_dim"], job["inter_dim"]

    def empty(*shape, dtype):
        return torch.empty(shape, dtype=dtype, device=device)

    return {
        "w1": empty(experts, 2 * inter_dim, model_dim // 2, dtype=dtypes.fp4x2),
        "w1_scale": empty(
            experts * 2 * inter_dim, model_dim // 32, dtype=dtypes.fp8_e8m0
        ),
        "w2": empty(experts, model_dim, inter_dim // 2, dtype=dtypes.fp4x2),
        "w2_scale": empty(experts * model_dim, inter_dim // 32, dtype=dtypes.fp8_e8m0),
    }


def compile_one_config(**job):
    """Build one profile and leave every launch variant of it in the cache."""
    from aiter.aot.flydsl.dry_run import dry_run

    result = {**job, "compile_time": None}
    started = time.time()
    try:
        rank = 0
        device = torch.device("cuda", rank)
        with (
            dry_run(ARCH, NUM_CU, name=DEVICE_NAME),
            _fake_arena(rank, job["world_size"]),
        ):
            from aiter.ops.flydsl.mega_moe_ep import MegaMoeEP, MegaMoeEPConfig

            moe = MegaMoeEP(
                MegaMoeEPConfig(
                    rank=rank,
                    world_size=job["world_size"],
                    model_dim=job["model_dim"],
                    inter_dim=job["inter_dim"],
                    experts=job["experts_per_rank"] * job["world_size"],
                    topk=job["topk"],
                    max_tok_per_rank=job["mtpr"],
                    quant=job["quant"],
                    swiglu_limit=job["swiglu_limit"],
                ),
                device=device,
                weights_preshuffled=True,
                **_weights(job, device),
            )
            moe.prepare()
            # prepare() leaves out the activation quant that forward() launches.
            h, k = job["model_dim"], job["topk"]
            moe.forward(
                torch.empty((1, h), dtype=torch.bfloat16, device=device),
                torch.empty((1, k), dtype=torch.float32, device=device),
                torch.empty((1, k), dtype=torch.int32, device=device),
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
            f"[aiter] FlyDSL MegaMoE EP replay: skipped, GPU_ARCHS omits {ARCH}",
            flush=True,
        )
        return 0
    command = [sys.executable, "-m", "aiter.aot.flydsl.mega_moe_ep", *(argv or [])]
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
    parser.add_argument(
        "--quant",
        nargs="+",
        choices=DEFAULT_QUANTS,
        default=list(DEFAULT_QUANTS),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compile nothing; fail on any kernel the replay needs but the cache lacks",
    )
    args = parser.parse_args()
    if args.check:
        # Every launch the replay reaches must load from the cache instead of compiling.
        os.environ["FLYDSL_RUNTIME_RUN_ONLY"] = "1"
    if os.environ.get("AITER_AOT_REPLAY") != "1":
        raise SystemExit(run(sys.argv[1:]))
    check_replay_process(ARCH)
    default_shape = (
        list(args.experts_per_rank) == list(DEFAULT_EXPERTS_PER_RANKS)
        and args.world_size == WORLD_SIZE
        and args.topk == TOPK
        and args.model_dim == MODEL_DIM
        and args.inter_dim == INTER_DIM
        and args.swiglu_limit == SWIGLU_LIMIT
    )
    if default_shape:
        jobs = production_jobs(tuple(args.mtpr), quants=tuple(args.quant))
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
        )
    results = run_jobs_parallel(compile_one_config, jobs)
    failed = sum(result["compile_time"] is None for result in results)
    print(
        f"[aiter] FlyDSL MegaMoE EP {'cache check' if args.check else 'replay'}: "
        f"{len(results) - failed} ok, {failed} failed",
        flush=True,
    )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    adopt_legacy_max_jobs()
    main()
