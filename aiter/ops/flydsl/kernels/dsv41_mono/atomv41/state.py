# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fixed-address buffers the mono kernels hand to the rest of the step."""

from dataclasses import dataclass

import torch

from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.runtime.consensus import MonoUnsupported


@dataclass(frozen=True)
class AuxTaps:
    """The DSpark drafter's aux hidden-state buffers.

    The drafter fills them from forward pre-hooks on target layers ``layer_ids``
    (each the mean over the hc copies of the settled residual entering the
    layer). A mono layer does not go through ``nn.Module.__call__``, so the hooks
    do not fire and the kernels write these buffers themselves.
    """

    layer_ids: tuple[int, ...]
    buffers: tuple[torch.Tensor, ...]


def bind_aux_taps(drafter, hidden_size: int) -> AuxTaps:
    # the layer ids the drafter armed its hooks from (DSparkProposer._aux_capture_spec)
    draft_config = drafter.speculative_config.draft_model_hf_config
    layer_ids = tuple(
        int(i) for i in getattr(draft_config, "dspark_target_layer_ids", ())
    )
    buffers = tuple(getattr(drafter, "_aux_buffers", ()))
    if not buffers:
        raise MonoUnsupported("the drafter captures no aux hidden states")
    if len(layer_ids) != len(buffers):
        raise MonoUnsupported(
            f"{len(buffers)} aux buffers for target layers {layer_ids}"
        )
    for buf in buffers:
        if buf.dim() != 2 or buf.shape[1] != hidden_size or not buf.is_contiguous():
            raise MonoUnsupported(f"aux buffer {tuple(buf.shape)} {buf.stride()}")
        if buf.dtype != torch.bfloat16:
            raise MonoUnsupported(f"aux buffer {buf.dtype}")
    return AuxTaps(layer_ids, buffers)
