# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`rope_rotate_activation`'s e8m0 layouts must be what each consumer reads.

Every fp4 scale layout is the same size, so handing a kernel the wrong
permutation neither fails nor mis-sizes anything -- it reads plausible scales at
wrong offsets. End-to-end accuracy barely moves, so compare bytes directly.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

from aiter import dtypes
from aiter.ops.quant import rope_rotate_activation

_SPEC = importlib.util.spec_from_file_location(
    "_opus_optest", Path(__file__).with_name("test_pa_mqa_logits_opus.py")
)
T = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(T)

HEADS, HEAD_DIM, GROUP = T.HEADS, T.HEAD_DIM, 32
TOKENS = 7


def _run(layout):
    """Quantize one fixed input and return (packed_q, scale) for `layout`."""
    torch.manual_seed(0)
    dev = T.dev
    x = torch.randn(TOKENS, HEADS, HEAD_DIM, dtype=dtypes.bf16, device=dev)
    cos = torch.randn(4096, HEAD_DIM // 2, dtype=dtypes.bf16, device=dev)
    sin = torch.randn(4096, HEAD_DIM // 2, dtype=dtypes.bf16, device=dev)
    pos = torch.arange(TOKENS, dtype=torch.int64, device=dev)
    q = torch.empty(TOKENS, HEADS, HEAD_DIM // 2, dtype=torch.uint8, device=dev)
    # Byte count is layout-independent; only the permutation differs.
    scale = torch.zeros(
        TOKENS * HEADS * (HEAD_DIM // GROUP), dtype=torch.uint8, device=dev
    )
    rope_rotate_activation(
        q.view(dtypes.fp4x2), x, cos, sin, pos, HEAD_DIM // 2,
        out_scale=scale, group_size=GROUP, do_rotate_act=False,
        scale_layout=layout,
    )  # fmt: skip
    return q, scale


def test_opus32_matches_reference_permutation():
    q_nat, s_nat = _run("natural")
    q_opus, s_opus = _run("opus32")
    # Same quantization, so the data must be byte-identical: a diff would mean
    # the layout changed VALUES, not just positions.
    assert torch.equal(q_nat, q_opus)
    want = T.scale_to_opus(s_nat.view(TOKENS * HEADS, HEAD_DIM // GROUP), HEADS)
    assert torch.equal(s_opus.view(want.shape), want)


def test_fly16_matches_reference_permutation():
    _, s_nat = _run("natural")
    _, s_fly = _run("fly16")
    want = T.q_scale_flydsl(s_nat.view(TOKENS * HEADS, HEAD_DIM // GROUP), TOKENS)
    assert torch.equal(s_fly.view(want.shape), want)


def test_layouts_are_permutations_of_one_another():
    """Guards the failure mode: same bytes, same multiset, different order."""
    _, s_fly = _run("fly16")
    _, s_opus = _run("opus32")
    assert s_fly.numel() == s_opus.numel()
    assert torch.equal(s_fly.sort().values, s_opus.sort().values)
    assert not torch.equal(s_fly, s_opus)


def test_shuffle_scale_bool_still_means_fly16():
    _, s_fly = _run("fly16")
    torch.manual_seed(0)
    dev = T.dev
    x = torch.randn(TOKENS, HEADS, HEAD_DIM, dtype=dtypes.bf16, device=dev)
    cos = torch.randn(4096, HEAD_DIM // 2, dtype=dtypes.bf16, device=dev)
    sin = torch.randn(4096, HEAD_DIM // 2, dtype=dtypes.bf16, device=dev)
    pos = torch.arange(TOKENS, dtype=torch.int64, device=dev)
    q = torch.empty(TOKENS, HEADS, HEAD_DIM // 2, dtype=torch.uint8, device=dev)
    scale = torch.zeros(
        TOKENS * HEADS * (HEAD_DIM // GROUP), dtype=torch.uint8, device=dev
    )
    rope_rotate_activation(
        q.view(dtypes.fp4x2), x, cos, sin, pos, HEAD_DIM // 2,
        out_scale=scale, group_size=GROUP, do_rotate_act=False, shuffle_scale=True,
    )  # fmt: skip
    assert torch.equal(scale, s_fly)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
