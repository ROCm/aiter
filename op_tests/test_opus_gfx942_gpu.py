# SPDX-License-Identifier: MIT
"""Representative gfx942 GPU regression for A8 blockscale bpreshuffle."""

from __future__ import annotations

import os

import pytest
import torch

from aiter import dtypes
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle
from aiter.ops.opus import opus_gemm
from aiter.ops.shuffle import shuffle_weight


pytestmark = pytest.mark.skipif(
    os.getenv("OPUS_GFX942_GPU", "0") != "1",
    reason="set OPUS_GFX942_GPU=1 for the gfx942 acceptance case",
)


def _require_gfx942() -> None:
    assert torch.cuda.is_available()
    props = torch.cuda.get_device_properties(0)
    arch = str(getattr(props, "gcnArchName", "")).split(":", 1)[0].lower()
    if arch != "gfx942":
        pytest.fail(f"gfx942 GPU validation requires gfx942 hardware, got {arch}")


def _problem():
    M, N, K = 128, 256, 256
    XQ = (
        torch.arange(M * K, device="cuda", dtype=torch.int32)
        .remainder(5)
        .sub(2)
        .reshape(M, K)
        .to(dtypes.fp8)
    )
    WQ = (
        torch.arange(N * K, device="cuda", dtype=torch.int32)
        .remainder(7)
        .sub(3)
        .reshape(N, K)
        .to(dtypes.fp8)
    )
    WQ_storage = shuffle_weight(WQ, layout=(16, 16))
    x_scale = torch.ones((M, K // 128), device="cuda", dtype=torch.float32)
    x_scale[1::2].mul_(0.5)
    x_scale[:, 1].mul_(2.0)
    x_scale_storage = x_scale.T.contiguous().view_as(x_scale)
    w_scale = torch.ones(
        (N // 128, K // 128), device="cuda", dtype=torch.float32
    )
    w_scale[1].mul_(2.0)
    w_scale[:, 1].mul_(0.25)

    golden = torch.zeros((M, N), device="cuda", dtype=torch.float32)
    for block_k in range(K // 128):
        partial = XQ[:, block_k * 128 : (block_k + 1) * 128].float() @ WQ[
            :, block_k * 128 : (block_k + 1) * 128
        ].float().T
        golden.add_(
            partial
            * x_scale[:, block_k].unsqueeze(1)
            * w_scale[:, block_k].repeat_interleave(128).unsqueeze(0)
        )
    return XQ, WQ_storage, x_scale_storage, w_scale, golden.to(torch.bfloat16)


def _assert_close(actual, golden):
    torch.cuda.synchronize()
    torch.testing.assert_close(actual.float(), golden.float(), rtol=0.03, atol=0.5)


def test_gfx942_kid11000_public_and_tuned_production_routes(tmp_path, monkeypatch):
    _require_gfx942()
    XQ, WQ, x_scale, w_scale, golden = _problem()
    Y = torch.empty_like(golden)

    assert opus_gemm(
        XQ,
        WQ,
        Y,
        kid=11000,
        layout="bpreshuffle",
        x_scale=x_scale,
        w_scale=w_scale,
    ) is Y
    _assert_close(Y, golden)

    props = torch.cuda.get_device_properties(0)
    csv_path = tmp_path / "gfx942_opus_bpreshuffle.csv"
    csv_path.write_text(
        "gfx,cu_num,M,N,K,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio\n"
        f"gfx942,{props.multi_processor_count},128,256,256,"
        "opus,11000,0,0.0,gfx942_acceptance,0.0,0.0,0.0\n"
    )
    monkeypatch.setenv(
        "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE", str(csv_path)
    )
    tuned_y = gemm_a8w8_blockscale_bpreshuffle(
        XQ, WQ, x_scale, w_scale, dtype=torch.bfloat16
    )
    _assert_close(tuned_y, golden)
