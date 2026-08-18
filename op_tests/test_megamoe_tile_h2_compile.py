# SPDX-License-Identifier: MIT
from __future__ import annotations

import torch


def test_target_h2_fused_role_kernel_compiles_and_runs_comm_role():
    from aiter.ops.flydsl.kernels.megamoe_tile.kernels import compile_hier_stage2_a4w4

    dev = torch.device("cuda", 0)
    src = torch.arange(4096, dtype=torch.int32, device=dev).remainder(251).to(torch.uint8)
    dst = torch.zeros_like(src)
    signal = torch.zeros(1, dtype=torch.int64, device=dev)
    dummy = torch.zeros(16, dtype=torch.uint8, device=dev)
    cumsum = torch.zeros(2, dtype=torch.int32, device=dev)
    launch = compile_hier_stage2_a4w4(
        D_HIDDEN=3584, D_INTER=384, NE=56, TOPK=16
    )
    launch(
        src.data_ptr(),
        dst.data_ptr(),
        signal.data_ptr(),
        src.numel(),
        29,
        dummy.data_ptr(),
        dummy.data_ptr(),
        dummy.data_ptr(),
        dummy.data_ptr(),
        dummy.data_ptr(),
        cumsum.data_ptr(),
        dummy.data_ptr(),
        dummy.data_ptr(),
        0,
        0,
        0,
        dummy.data_ptr(),
        stream=torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()
    assert torch.equal(src, dst)
    assert signal.item() == 29
    assert launch.lds_bytes == 8192
