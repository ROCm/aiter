# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16


@pytest.mark.parametrize(
    "n,k", [(4096, 8192), (8192, 128), (32768, 512), (128, 4096), (288, 4096)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_batch_invariant_shape_configs(n, k, dtype):
    if not torch.cuda.is_available():
        pytest.skip("Requires a GPU")
    if get_gfx_runtime() != "gfx950" or get_cu_num() != 256:
        pytest.skip("Requires gfx950 with 256 CUs")
    generator = torch.Generator(device="cuda").manual_seed(42)
    x = torch.randn(8192, k, generator=generator, device="cuda", dtype=dtype)
    w = torch.randn(n, k, generator=generator, device="cuda", dtype=dtype)
    full = gemm_a16w16(x, w, dtype=dtype, backend="triton")
    # These JSON files also serve FP16 and direct Triton callers.
    for m in (
        1,
        8,
        16,
        32,
        64,
        65,
        128,
        129,
        256,
        257,
        512,
        513,
        1024,
        1025,
        2048,
        2049,
    ):
        assert torch.equal(
            gemm_a16w16(x[:m].clone(), w, dtype=dtype, backend="triton"), full[:m]
        ), f"Cross-M drift at M={m}"
    indices = torch.tensor(
        [0, 64, 128, 512, 1024, 2048, 4096, 8191], device="cuda", dtype=torch.int64
    )
    reference = (x[indices].float() @ w.float().T).to(dtype)
    torch.testing.assert_close(full[indices], reference, rtol=1e-2, atol=1e-2)
