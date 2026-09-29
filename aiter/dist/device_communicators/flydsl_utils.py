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


def warm_fly_engines(engines, device) -> None:
    """Compile and load the binaries each engine serves, by launching each once.

    *engines* holds ``(engine, (lo, hi))`` pairs, ``lo..hi`` being the payload
    bytes (inclusive) the dispatcher routes to that engine. Binaries its ladder
    assigns only to payloads outside that range are left alone: they never run.

    FlyDSL compiles and loads a binary on its first launch. If that happens
    inside an active CUDA graph capture, the capture pays seconds of JIT and
    issues HIP module loads that capture is not guaranteed to allow. The kernels
    take the payload size at runtime, so one launch per binary covers every size.
    """

    # bf16 and a multiple of 16 B, as every FlyDSL all-reduce requires.
    warm = torch.zeros(2048, dtype=torch.bfloat16, device=device)
    for engine, payload_range in engines:
        engine.compile_and_launch(warm, payload_range=payload_range)
