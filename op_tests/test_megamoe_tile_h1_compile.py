# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest
import torch


def test_target_h1_fused_role_kernel_compiles_and_runs_comm_role():
    from aiter.ops.flydsl.kernels.megamoe_tile.kernels import compile_hier_stage1_a4w4_silu

    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    src = torch.arange(4096, dtype=torch.int32, device=dev).remainder(251).to(torch.uint8)
    dst = torch.zeros_like(src)
    signal = torch.zeros(1, dtype=torch.int64, device=dev)
    dummy_u8 = torch.zeros(16, dtype=torch.uint8, device=dev)
    cumsum = torch.zeros(2, dtype=torch.int32, device=dev)

    launch = compile_hier_stage1_a4w4_silu(
        D_HIDDEN=3584,
        D_INTER=384,
        NE=56,
        TOPK=16,
        COMM_BLOCKS=1,
        rejoin_compute=False,
    )
    # gemm_grid=0 executes only the low-ticket communication role, but the code
    # object contains the complete target GMM1+SiLU+A4 body.
    launch(
        src.data_ptr(),
        dst.data_ptr(),
        signal.data_ptr(),
        src.numel(),
        17,
        dummy_u8.data_ptr(),
        dummy_u8.data_ptr(),
        dummy_u8.data_ptr(),
        dummy_u8.data_ptr(),
        dummy_u8.data_ptr(),
        cumsum.data_ptr(),
        dummy_u8.data_ptr(),
        0,
        0,
        dummy_u8.data_ptr(),
        dummy_u8.data_ptr(),
        dummy_u8.data_ptr(),
        stream=torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()
    assert torch.equal(src, dst)
    assert signal.item() == 17
    assert launch.lds_bytes > 0


def test_h1_rejects_unsupported_tile_variants():
    from aiter.ops.flydsl.kernels.megamoe_tile.kernels import (
        compile_hier_stage1_a4w4,
        compile_hier_stage1_a4w4_persistent,
    )

    with pytest.raises(ValueError, match="BN=BK=256"):
        compile_hier_stage1_a4w4(
            D_HIDDEN=1024,
            D_INTER=256,
            NE=8,
            TOPK=2,
            BN=128,
            BK=128,
        )
    with pytest.raises(ValueError, match="unsupported H1 A4W4 variant"):
        compile_hier_stage1_a4w4(
            D_HIDDEN=1024,
            D_INTER=256,
            NE=8,
            TOPK=2,
            BM=64,
            use_nt=True,
        )
    with pytest.raises(ValueError, match="requires COMM_BLOCKS=1"):
        compile_hier_stage1_a4w4_persistent(
            D_HIDDEN=1024,
            D_INTER=256,
            NE=8,
            TOPK=2,
            COMM_BLOCKS=2,
        )
