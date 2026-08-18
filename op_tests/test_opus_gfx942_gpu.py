# SPDX-License-Identifier: MIT
"""MI300/MI308-only GPU validation for gfx942 OPUS A8 kid 11000."""

from __future__ import annotations

import os

import pytest
import torch

from aiter import dtypes
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle
from aiter.ops.opus import gemm_op_a8w8 as a8
from aiter.ops.opus import opus_bmm, opus_gemm
from aiter.ops.shuffle import shuffle_weight
from csrc.opus_gemm.opus_gemm_common import get_kernel_instance


pytestmark = pytest.mark.skipif(
    os.getenv("OPUS_GFX942_EXHAUSTIVE", "0") != "1",
    reason="set OPUS_GFX942_EXHAUSTIVE=1 for gfx942 GPU acceptance",
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


def _assert_close(actual: torch.Tensor, golden: torch.Tensor) -> None:
    torch.cuda.synchronize()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), golden.float(), rtol=0.03, atol=0.5)


@pytest.mark.parametrize(
    ("kid", "workspace_dtype"),
    ((10200, torch.float32), (10210, torch.bfloat16)),
)
def test_gfx942_a16_workspace_graph_two_streams_and_caller_tensor(
    kid, workspace_dtype
):
    _require_gfx942()
    instance = get_kernel_instance("gfx942", "a16w16", kid)
    assert instance is not None
    M, N, K = int(instance.B_M), int(instance.B_N), 32 * int(instance.B_K)
    torch.manual_seed(0x942000 + kid)
    XQ = torch.randn((1, M, K), device="cuda", dtype=torch.bfloat16)
    WQ = torch.randn((1, N, K), device="cuda", dtype=torch.bfloat16)
    golden = torch.bmm(XQ.float(), WQ.float().transpose(1, 2))
    Y = torch.empty((1, M, N), device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(
        (2, 1, M, N), device="cuda", dtype=workspace_dtype
    )
    graph_stream = torch.cuda.Stream()
    graph_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(graph_stream):
        opus_bmm(
            XQ, WQ, Y, kid=kid, split_k=2, workspace=workspace
        )
    graph_stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=graph_stream):
        opus_bmm(
            XQ, WQ, Y, kid=kid, split_k=2, workspace=workspace
        )
    graph.replay()
    _assert_close(Y, golden)

    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    outputs = [
        torch.empty((1, M, N), device="cuda", dtype=torch.bfloat16)
        for _ in streams
    ]
    workspaces = [
        torch.empty((2, 1, M, N), device="cuda", dtype=workspace_dtype)
        for _ in streams
    ]
    for stream, output, caller_workspace in zip(
        streams, outputs, workspaces, strict=True
    ):
        with torch.cuda.stream(stream):
            opus_bmm(
                XQ,
                WQ,
                output,
                kid=kid,
                split_k=2,
                workspace=caller_workspace,
            )
    for stream in streams:
        stream.synchronize()
    for output in outputs:
        _assert_close(output, golden)


def test_gfx942_a8_11000_raw_2d_and_batch1_3d():
    _require_gfx942()
    XQ, WQ, x_scale, w_scale, golden = _problem()
    Y = torch.empty_like(golden)
    a8._opus_gemm_a8w8_blockscale_bpreshuffle_launch_raw(
        XQ, WQ, x_scale, w_scale, Y, 11000
    )
    _assert_close(Y, golden)

    Y3 = torch.empty((1, *golden.shape), device="cuda", dtype=torch.bfloat16)
    a8._opus_gemm_a8w8_blockscale_bpreshuffle_launch_raw(
        XQ.unsqueeze(0),
        WQ.unsqueeze(0),
        x_scale,
        w_scale,
        Y3,
        11000,
    )
    _assert_close(Y3[0], golden)


def test_gfx942_a8_11000_unified_public_and_real_tuned_csv(tmp_path, monkeypatch):
    _require_gfx942()
    XQ, WQ, x_scale, w_scale, golden = _problem()
    Y = torch.empty_like(golden)
    returned = opus_gemm(
        XQ,
        WQ,
        Y,
        kid=11000,
        layout="bpreshuffle",
        x_scale=x_scale,
        w_scale=w_scale,
    )
    assert returned is Y
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


def test_gfx942_a8_11000_graph_replay_and_two_streams():
    _require_gfx942()
    XQ, WQ, x_scale, w_scale, golden = _problem()
    Y = torch.empty_like(golden)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        opus_gemm(
            XQ,
            WQ,
            Y,
            kid=11000,
            layout="bpreshuffle",
            x_scale=x_scale,
            w_scale=w_scale,
        )
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        opus_gemm(
            XQ,
            WQ,
            Y,
            kid=11000,
            layout="bpreshuffle",
            x_scale=x_scale,
            w_scale=w_scale,
        )
    graph.replay()
    _assert_close(Y, golden)

    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    outputs = (torch.empty_like(golden), torch.empty_like(golden))
    for index, live_stream in enumerate(streams):
        with torch.cuda.stream(live_stream):
            opus_gemm(
                XQ,
                WQ,
                outputs[index],
                kid=11000,
                layout="bpreshuffle",
                x_scale=x_scale,
                w_scale=w_scale,
            )
    for live_stream in streams:
        live_stream.synchronize()
    for output in outputs:
        _assert_close(output, golden)


def test_gfx942_a8_11000_rejects_invalid_contracts():
    _require_gfx942()
    XQ, WQ, x_scale, w_scale, golden = _problem()
    Y = torch.empty_like(golden)

    bad_calls = (
        lambda: a8._opus_gemm_a8w8_blockscale_bpreshuffle_launch_raw(
            XQ.unsqueeze(0).expand(2, -1, -1),
            WQ.unsqueeze(0).expand(2, -1, -1),
            x_scale.unsqueeze(0).expand(2, -1, -1),
            w_scale.unsqueeze(0).expand(2, -1, -1),
            Y.unsqueeze(0).expand(2, -1, -1),
            11000,
        ),
        lambda: opus_gemm(
            XQ,
            WQ,
            Y,
            kid=11000,
            layout="bpreshuffle",
            x_scale=x_scale.to(torch.bfloat16),
            w_scale=w_scale,
        ),
        lambda: opus_gemm(
            XQ,
            WQ,
            Y,
            kid=11000,
            layout="bpreshuffle",
            x_scale=x_scale[:, :1],
            w_scale=w_scale,
        ),
        lambda: opus_gemm(
            XQ[:, ::2],
            WQ,
            Y,
            kid=11000,
            layout="bpreshuffle",
            x_scale=x_scale,
            w_scale=w_scale,
        ),
        lambda: opus_gemm(
            XQ,
            WQ,
            Y,
            kid=1,
            layout="bpreshuffle",
            x_scale=x_scale,
            w_scale=w_scale,
        ),
        lambda: opus_gemm(
            XQ,
            WQ,
            Y,
            kid=20000,
            layout="bpreshuffle",
            x_scale=x_scale,
            w_scale=w_scale,
        ),
    )
    for call in bad_calls:
        with pytest.raises((RuntimeError, ValueError, NotImplementedError)):
            call()
