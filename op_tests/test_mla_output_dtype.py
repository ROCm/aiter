# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter import dtypes
from aiter.mla import mla_decode_fwd


@pytest.mark.parametrize("output_dtype", [dtypes.fp8, dtypes.fp32])
def test_mla_decode_rejects_unsupported_output_dtype(output_dtype):
    q = torch.empty((1, 16, 576), dtype=dtypes.bf16)
    kv = torch.empty((1, 1, 1, 576), dtype=dtypes.bf16)
    output = torch.empty((1, 16, 512), dtype=output_dtype)

    with pytest.raises(ValueError, match="output must use a 16-bit floating dtype"):
        mla_decode_fwd(
            q,
            kv,
            output,
            None,
            None,
            None,
            None,
            1,
        )
