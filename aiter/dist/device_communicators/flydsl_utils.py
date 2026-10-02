# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Init helpers shared by the FlyDSL backends of ``CustomAllreduce`` and
``QuickAllReduce``."""

import torch
import torch.distributed as dist


def all_ranks_agree(ok: bool, group) -> bool:
    """Whether *every* rank in *group* reports success."""

    flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def preload_fly_engines(engines) -> None:
    """Compile the binaries each engine serves, without launching them.

    *engines* holds ``(engine, (lo, hi))`` pairs, ``lo..hi`` being the payload
    bytes (inclusive) the dispatcher routes to that engine. Binaries its ladder
    assigns only to payloads outside that range are left alone: they never run.

    FlyDSL compiles a binary on its first launch, and if that happens inside an
    active CUDA graph capture, the capture pays seconds of JIT. The kernels take
    the payload size at runtime, so one preload per binary covers every size.

    The HIP module load is left to the first launch.
    """

    for engine, payload_range in engines:
        engine.preload(payload_range=payload_range)
