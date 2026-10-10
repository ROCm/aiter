# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pure aggregation and input identity shared by acceptance replay scripts."""

import hashlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def fingerprint(tensor: "torch.Tensor") -> str:
    import torch

    return hashlib.sha256(
        tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def trace_kernel_metrics(
    events: Sequence[dict[str, Any]], iters: int
) -> tuple[list[dict[str, Any]], float, float]:
    kernels = [
        event
        for event in events
        if event.get("cat") == "kernel" and event.get("ph") == "X"
    ]
    assert kernels and all(
        event.get("cat") != "gpu_user_annotation" for event in kernels
    )
    kernel_sum = sum(event["dur"] for event in kernels) / iters
    span = (
        max(event["ts"] + event["dur"] for event in kernels)
        - min(event["ts"] for event in kernels)
    ) / iters
    return kernels, kernel_sum, span
