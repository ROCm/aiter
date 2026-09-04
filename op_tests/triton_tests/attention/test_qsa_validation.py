# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.attention.qsa import (
    qsa_expand_block_indices,
    qsa_paged_mqa_logits,
    qsa_select_paged_tokens,
)


def _cpu_placeholder():
    return torch.empty(0)


@pytest.mark.parametrize("compress_ratio", (0, -1))
def test_qsa_select_rejects_nonpositive_compress_ratio(compress_ratio):
    tensors = [_cpu_placeholder() for _ in range(6)]

    with pytest.raises(ValueError, match="compress_ratio must be positive"):
        qsa_select_paged_tokens(
            *tensors,
            token_topk=8,
            compress_ratio=compress_ratio,
        )


@pytest.mark.parametrize("compress_ratio", (True, 1.5, "4"))
def test_qsa_select_rejects_noninteger_compress_ratio(compress_ratio):
    tensors = [_cpu_placeholder() for _ in range(6)]

    with pytest.raises(TypeError, match="compress_ratio must be an int"):
        qsa_select_paged_tokens(
            *tensors,
            token_topk=8,
            compress_ratio=compress_ratio,
        )


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("token_topk", 0),
        ("token_topk", 1.5),
        ("logits_workspace_bytes", 0),
        ("logits_workspace_bytes", 1.5),
    ),
)
def test_qsa_select_validates_positive_integer_options(name, value):
    tensors = [_cpu_placeholder() for _ in range(6)]
    arguments = {"token_topk": 8, "logits_workspace_bytes": 1024}
    arguments[name] = value
    error = TypeError if isinstance(value, float) else ValueError

    with pytest.raises(error, match=rf"{name} must"):
        qsa_select_paged_tokens(*tensors, **arguments)


@pytest.mark.parametrize("entry_point", ("paged_mqa_logits", "expand_block_indices"))
def test_qsa_entry_points_validate_compress_ratio_before_tensors(entry_point):
    tensors = [_cpu_placeholder() for _ in range(6)]

    with pytest.raises(ValueError, match="compress_ratio must be positive"):
        if entry_point == "paged_mqa_logits":
            qsa_paged_mqa_logits(*tensors, compress_ratio=0)
        else:
            qsa_expand_block_indices(
                *tensors[:4],
                compress_ratio=0,
                token_topk=8,
            )
