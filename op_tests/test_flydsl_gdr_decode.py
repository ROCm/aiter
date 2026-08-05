# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CI-visible tests for the FlyDSL GDR decode op: its export, and its numerics.

CI collects only ``op_tests/test_*.py`` at depth 1 (see
``.github/scripts/split_tests.sh``), so the kernel's in-package suite at
``aiter/ops/flydsl/test_flydsl_linear_attention.py`` never runs automatically.
This file puts both properties under CI.

The export checks matter on their own: reaching into ``linear_attention_kernels``
keeps working if the export drops back out of ``__init__.py``, so no other test
would notice that regression. The numeric checks reuse the in-package harness
rather than carrying a second copy of its Triton reference.

Keep it thin -- shapes belong here, the reference kernel belongs in-package.
"""

import pytest
import torch

import aiter.ops.flydsl as flydsl_ops
from aiter.ops.flydsl import is_flydsl_available

pytestmark = pytest.mark.skipif(
    not is_flydsl_available(), reason="flydsl is not installed"
)

# Skips itself at import when flydsl or a GPU is missing, which skips this module
# with it.
from aiter.ops.flydsl.test_flydsl_linear_attention import (
    Args,
    check_gdr_decode,
)


def test_flydsl_gdr_decode_is_importable_from_package_namespace():
    assert hasattr(flydsl_ops, "flydsl_gdr_decode"), (
        "flydsl_gdr_decode is missing from aiter.ops.flydsl -- check that its "
        "import is uncommented in aiter/ops/flydsl/__init__.py"
    )
    assert callable(flydsl_ops.flydsl_gdr_decode)


def test_flydsl_gdr_decode_is_advertised_in_all():
    assert "flydsl_gdr_decode" in flydsl_ops.__all__


def test_package_export_is_the_kernel_wrapper_itself():
    from aiter.ops.flydsl.linear_attention_kernels import flydsl_gdr_decode

    assert flydsl_ops.flydsl_gdr_decode is flydsl_gdr_decode


@pytest.mark.parametrize(
    "args",
    [
        # Smallest GQA shape, the pre-847 scalar path.
        Args(
            dtype=torch.bfloat16,
            b=1,
            sq=1,
            num_k_heads=2,
            num_v_heads=8,
            head_k_dim=128,
            head_v_dim=128,
        ),
        # Batch large enough to exercise the state-shuffle indices.
        Args(
            dtype=torch.bfloat16,
            b=128,
            sq=1,
            num_k_heads=2,
            num_v_heads=8,
            head_k_dim=128,
            head_v_dim=128,
        ),
        # f32 bias against a bf16 query, decoupled from the query dtype for KDA.
        Args(
            dtype=torch.bfloat16,
            b=1,
            sq=1,
            num_k_heads=2,
            num_v_heads=8,
            head_k_dim=128,
            head_v_dim=128,
            dt_bias_dtype=torch.float32,
        ),
    ],
    ids=["gqa_b1", "gqa_b128", "f32_dt_bias"],
)
def test_flydsl_gdr_decode_matches_reference(args):
    check_gdr_decode(args)
