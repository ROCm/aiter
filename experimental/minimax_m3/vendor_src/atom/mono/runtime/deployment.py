# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Whether a deployment can run a mono decode at all: the refusals every model
shares (one TP group on one node, decode steps in the model's own graph), which
a model's own list extends."""

from atom.mono.plan.shard import shard_refusal
from atom.plugin.prepare import is_sglang, is_vllm


def common_refusals(atom_config, dims, supported_tp) -> list[tuple[bool, str]]:
    """``(holds, why not)`` of the conditions every mono model needs: ``dims``
    (the model's ``Dims``) shards at this TP, which is one of ``supported_tp``;
    no expert, data, pipeline or context parallelism, no TBO, not a plugin, no
    piecewise cudagraph."""
    tp = atom_config.tensor_parallel_size
    shard = shard_refusal(dims, tp)
    return [
        (shard is None, shard),
        (tp in supported_tp, f"TP {tp} not in {supported_tp}"),
        (not atom_config.enable_expert_parallel, "expert parallel"),
        (atom_config.parallel_config.data_parallel_size == 1, "DP > 1"),
        (atom_config.pipeline_parallel_size == 1, "PP > 1"),
        (atom_config.decode_context_parallel_size == 1, "decode CP"),
        (atom_config.prefill_context_parallel_size == 1, "prefill CP"),
        (not (atom_config.enable_tbo or atom_config.enable_tbo_decode), "TBO"),
        (not (is_vllm() or is_sglang()), "plugin mode"),
        (
            not atom_config.compilation_config.cudagraph_mode.requires_piecewise_compilation(),
            "piecewise cudagraph",
        ),
    ]


def first_refusal(checks) -> str | None:
    """The reason of the first check that does not hold, or None."""
    return next((why for ok, why in checks if not ok), None)
