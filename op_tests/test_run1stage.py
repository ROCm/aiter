# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Regression test for fused_moe_2stages' metadata.run_1stage guard.

fused_moe_2stages composes metadata.stage1 / metadata.stage2 as a genuine
two-stage pipeline. When get_2stage_cfgs (or a caller's _metadata_transform)
resolves metadata.run_1stage=True, metadata.stage1 is instead a complete,
single-call kernel whose return value IS the final output -- not an
intermediate for stage2 to consume. _fused_moe_impl's own top-level dispatch
already special-cases this; fused_moe_2stages, reachable independently (e.g.
via a fallback/re-resolution path), did not -- see ROCm/aiter#7888.

This test forces metadata.run_1stage=True via the existing
_metadata_transform hook with a stub stage1/stage2, so it exercises the
dispatch control-flow deterministically without depending on which real
kernel a given shape/dtype combination happens to resolve to. It still needs
real GPU/chip detection (get_2stage_cfgs calls get_gfx()/get_cu_num()), so it
is not a pure-CPU unit test, but it does not depend on any specific kernel
actually being tuned/available for the probe shape used to build the initial
metadata -- only the override path is checked.
"""

import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import fused_moe_2stages


def _make_moe_tensors(device):
    token_num, model_dim, inter_dim, expert, topk = 4, 64, 128, 2, 1
    hidden_states = torch.randn(token_num, model_dim, dtype=torch.bfloat16, device=device)
    w1 = torch.randn(expert, inter_dim * 2, model_dim, dtype=torch.bfloat16, device=device)
    w2 = torch.randn(expert, model_dim, inter_dim, dtype=torch.bfloat16, device=device)
    sorted_ids = torch.arange(token_num, dtype=torch.int32, device=device)
    sorted_weights = torch.ones(token_num, dtype=torch.float32, device=device)
    sorted_expert_ids = torch.zeros(token_num, dtype=torch.int32, device=device)
    num_valid_ids = torch.tensor([token_num], dtype=torch.int32, device=device)
    moe_out = torch.empty(token_num, model_dim, dtype=torch.bfloat16, device=device)
    return {
        "hidden_states": hidden_states,
        "w1": w1,
        "w2": w2,
        "topk": topk,
        "sorted_ids": sorted_ids,
        "sorted_weights": sorted_weights,
        "sorted_expert_ids": sorted_expert_ids,
        "num_valid_ids": num_valid_ids,
        "moe_out": moe_out,
        "isG1U1": True,
        "block_size_M": 32,
    }


def test_fused_moe_2stages_respects_run_1stage():
    """metadata.run_1stage=True must early-return stage1's output, never
    call stage2 -- matching _fused_moe_impl's own dispatch."""
    device = "cuda"
    kwargs = _make_moe_tensors(device)
    token_num, model_dim = kwargs["hidden_states"].shape

    sentinel_output = torch.full(
        (token_num, model_dim), 7.0, dtype=torch.bfloat16, device=device
    )
    stage1_calls = []
    stage2_calls = []

    def stub_stage1(*args, **fkwargs):
        stage1_calls.append((args, fkwargs))
        return sentinel_output

    def stub_stage2(*args, **fkwargs):
        stage2_calls.append((args, fkwargs))
        raise AssertionError(
            "stage2 must never be called when metadata.run_1stage is True"
        )

    def force_run_1stage(metadata):
        metadata.run_1stage = True
        metadata.stage1 = stub_stage1
        metadata.stage2 = stub_stage2
        return metadata

    result = fused_moe_2stages(
        kwargs["hidden_states"],
        kwargs["w1"],
        kwargs["w2"],
        kwargs["topk"],
        kwargs["sorted_ids"],
        kwargs["sorted_weights"],
        kwargs["sorted_expert_ids"],
        kwargs["num_valid_ids"],
        kwargs["moe_out"],
        kwargs["isG1U1"],
        kwargs["block_size_M"],
        _metadata_transform=force_run_1stage,
    )

    assert len(stage1_calls) == 1, (
        f"expected stage1 to be called exactly once, got {len(stage1_calls)}"
    )
    assert len(stage2_calls) == 0, (
        f"stage2 was called {len(stage2_calls)} time(s); it must never run "
        "when metadata.run_1stage is True"
    )
    torch.testing.assert_close(result, sentinel_output)


if __name__ == "__main__":
    test_fused_moe_2stages_respects_run_1stage()
    print("PASS: fused_moe_2stages respects metadata.run_1stage")
