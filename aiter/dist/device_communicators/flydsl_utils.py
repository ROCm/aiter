# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Init helpers shared by the FlyDSL backends of ``CustomAllreduce`` and
``QuickAllReduce``."""

import logging

import torch
import torch.distributed as dist

logger = logging.getLogger("aiter")


def all_ranks_agree(ok: bool, group) -> bool:
    """Whether *every* rank in *group* reports success."""

    flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def all_ranks_agree_on_plan(plan, group, *, owner: str) -> bool:
    """Whether every rank in the group holds the same FlyDSL plan.

    plan is this rank's dict of every input to its engine build, or an str
    naming why this rank declines. It is computed locally, from the environment
    and the host. Hence, ranks can disagree. One ``all_gather_object`` on the
    CPU group settles it before any engine is built.

    Every rank that reaches the FlyDSL init must call this, whatever its own
    verdict, including when FlyDSL is off or not importable.

    True only if every rank holds the same dict.
    """

    plans = [None] * dist.get_world_size(group)
    dist.all_gather_object(plans, plan, group=group)
    if all(isinstance(p, str) for p in plans):
        return False
    if all(p == plans[0] for p in plans):
        return True
    logger.warning(
        "%s disabled: the ranks disagree on the FlyDSL setup: %s",
        owner,
        _plan_differences(plans),
    )
    return False


def _plan_differences(plans) -> str:
    """The ranks that decline and the plan fields that differ, per rank."""

    parts = [
        f"rank {r} declines ({p})" for r, p in enumerate(plans) if isinstance(p, str)
    ]
    built = {r: p for r, p in enumerate(plans) if not isinstance(p, str)}
    for key in sorted({k for p in built.values() for k in p}):
        values = {r: p.get(key) for r, p in built.items()}
        if any(v != values[min(values)] for v in values.values()):
            parts.append(
                f"{key}: " + ", ".join(f"rank {r}={v}" for r, v in values.items())
            )
    return "; ".join(parts)


def preload_fly_engines(engines) -> None:
    """Compile the binaries each engine serves and load them onto the device,
    without launching them.

    *engines* holds ``(engine, (lo, hi))`` pairs, ``lo..hi`` being the payload
    bytes (inclusive) the dispatcher routes to that engine. Binaries its ladder
    assigns only to payloads outside that range are left alone: they never run.

    Otherwise FlyDSL would compile a binary, JIT its host wrapper, and load its
    HIP module on its first launch. The kernels take the payload size at runtime,
    so one preload per binary covers every size.
    """

    for engine, payload_range in engines:
        engine.preload(payload_range=payload_range)
