# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest

from aiter.ops.flydsl.kernels.mixed_moe_gemm_2stage_common import (
    _check_moe_weight_addressing_limits,
)
from aiter.ops.flydsl.kernels.moe_2stage_a16wmix.utils import (
    _check_weight_addressing_limits,
)
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import (
    check_weight_addressing_limits as check_mxfp4_weight_addressing_limits,
)


def test_addressing_limit_guards():
    limit = 1 << 31
    common = {
        "stage": "test",
        "per_expert_w_bytes": 1,
        "shared_w_bytes": 1,
        "scale_w_bytes": 1,
        "shared_scale_w_bytes": 1,
    }
    _check_moe_weight_addressing_limits(**common)
    for field in (
        "per_expert_w_bytes",
        "shared_w_bytes",
        "scale_w_bytes",
        "shared_scale_w_bytes",
    ):
        with pytest.raises(ValueError, match="signed 32-bit buffer"):
            _check_moe_weight_addressing_limits(**(common | {field: limit}))

    with pytest.raises(ValueError, match="raw weight per expert"):
        _check_weight_addressing_limits(
            stage="test", w_dtype="bf16", n_out=1, k=1 << 30, experts=1
        )
    with pytest.raises(ValueError, match="global weight scale"):
        _check_weight_addressing_limits(
            stage="test", w_dtype="fp4", n_out=256, k=256, experts=1 << 20
        )
    check_mxfp4_weight_addressing_limits(
        stage="test", per_expert_w_bytes=limit - 1, scale_w_bytes=limit - 1
    )
    with pytest.raises(ValueError, match="raw weight per expert"):
        check_mxfp4_weight_addressing_limits(
            stage="test", per_expert_w_bytes=limit, scale_w_bytes=1
        )
    with pytest.raises(ValueError, match="global weight scale"):
        check_mxfp4_weight_addressing_limits(
            stage="test", per_expert_w_bytes=1, scale_w_bytes=limit
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
