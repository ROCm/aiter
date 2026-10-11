# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Per-N-row K32 E8M0 scales through quantization and exact/public BMM."""

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.batched_gemm_op_a8w8 import batched_gemm_a8w8_mxscale_bpreshuffle
from aiter.ops.opus import opus_bmm
from aiter.ops.opus.gemm_op_a8w8 import bmm_a8w8_mxscale_opus
from aiter.ops.opus.launch_plan import _build_a8w8_mxscale_bmm_plan
from aiter.ops.quant import per_1x32_f8_scale_f8_quant
from aiter.ops.shuffle import shuffle_weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950", reason="gfx950 required"
)


def inputs(b, m, n, k, dtype):
    torch.manual_seed(197)

    def quant(shape):
        # Every row and K32 block varies independently. Square grouping or a
        # wrong MFMA lane's scale changes the result by powers of two.
        blocks = torch.randn((*shape[:-1], k // 32, 32), device="cuda")
        exponents = (
            torch.arange(blocks.numel() // 32, device="cuda").reshape(
                *shape[:-1], k // 32, 1
            )
            % 9
            - 4
        )
        values = (blocks * torch.exp2(exponents.float())).reshape(shape).bfloat16()
        q, scales = per_1x32_f8_scale_f8_quant(values, scale_type=torch.uint8)
        scales = scales.view(*shape[:-1], k // 32)
        dequant = q.float() * torch.exp2(
            scales.view(torch.uint8).float() - 127
        ).repeat_interleave(32, dim=-1)
        return q, scales, dequant

    x, xs, xd = quant((m, b, k))
    w, ws, wd = quant((b, n, k))
    ref = torch.einsum("mbk,bnk->mbn", xd, wd).to(dtype)
    return x, shuffle_weight(w, (16, 16)), xs, ws, ref


@pytest.mark.parametrize("kid", [*range(9700, 9711), 9713])
@pytest.mark.parametrize(
    "shape", [(1, 1, 16, 128), (2, 17, 64, 512), (3, 65, 192, 1536)]
)
@pytest.mark.parametrize("split", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_exact_1x32(kid, shape, split, dtype):
    b, m, n, k = shape
    try:
        _build_a8w8_mxscale_bmm_plan(
            arch="gfx950",
            kid=kid,
            output_dtype=dtype,
            M=m,
            batch=b,
            N=n,
            K=k,
            split_k=split,
        )
    except ValueError as exc:
        pytest.skip(str(exc))
    x, w, xs, ws, ref = inputs(b, m, n, k, dtype)
    storage = torch.full((m * b * n + 16,), float("nan"), device="cuda", dtype=dtype)
    y = storage[:-16].view(m, b, n)
    opus_bmm(
        x.transpose(0, 1),
        w,
        y.transpose(0, 1),
        kid=kid,
        layout="mxscale_bmm",
        x_scale=xs.transpose(0, 1),
        w_scale=ws,
        split_k=split,
    )
    torch.testing.assert_close(y, ref, rtol=0.02, atol=0.03)
    assert storage[-16:].isnan().all()


@pytest.mark.parametrize("shape", [(1, 17, 16, 128), (2, 65, 64, 512)])
def test_inferred_route_and_graph(shape):
    x, w, xs, ws, ref = inputs(*shape, torch.bfloat16)
    torch.testing.assert_close(
        batched_gemm_a8w8_mxscale_bpreshuffle(x, w, xs, ws), ref, rtol=0.02, atol=0.03
    )
    y = torch.empty_like(ref)

    # A stale square-block tuned ID must fall back to the matching 1x32 ID.
    def run():
        return bmm_a8w8_mxscale_opus(
            x,
            w,
            xs,
            ws,
            out=y,
            kernelId=9179,
            b_preshuffled=True,
            group_n=1,
            group_size=32,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(2):
        y.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(y, ref, rtol=0.02, atol=0.03)


def test_rejects_square_scale_shape():
    x, w, xs, ws, ref = inputs(2, 17, 64, 512, torch.bfloat16)
    wrong_ws = ws[:, ::32, :].contiguous()
    for split in (1, 2):
        with pytest.raises((ValueError, RuntimeError), match="w_scale"):
            opus_bmm(
                x.transpose(0, 1),
                w,
                torch.empty_like(ref).transpose(0, 1),
                kid=9700,
                layout="mxscale_bmm",
                x_scale=xs.transpose(0, 1),
                w_scale=wrong_ws,
                split_k=split,
            )


@pytest.mark.parametrize("kid", range(8440, 8450))
@pytest.mark.parametrize("split", [1, 2])
def test_existing_wave1_group128(kid, split):
    from op_tests.test_opus_a8w8_bmm import _quant_block_e8m0, _quant_per_token_e8m0

    torch.manual_seed(89)
    x, xs, xf = _quant_per_token_e8m0(torch.randn(2, 17, 1024, device="cuda"))
    w, ws, wf = _quant_block_e8m0(torch.randn(2, 128, 1024, device="cuda"))
    xd = x.float() * xf.repeat_interleave(128, -1)
    wd = w.float() * wf.repeat_interleave(128, 1).repeat_interleave(128, -1)
    ref = torch.bmm(xd, wd.transpose(1, 2)).bfloat16()
    y = torch.empty_like(ref)
    opus_bmm(
        x,
        shuffle_weight(w, (16, 16)),
        y,
        kid=kid,
        layout="mxscale_bmm",
        x_scale=xs,
        w_scale=ws,
        split_k=split,
    )
    torch.testing.assert_close(y, ref, rtol=0.02, atol=0.03)


@pytest.mark.parametrize("kid", [*range(9700, 9711), 9713])
def test_tuner_data_and_bench(kid):
    from csrc.opus_gemm import opus_bmm_mxscale_tune as tune

    data = tune.gen_bmm_mxscale_data(2, 17, 64, 1024, kid, torch.float32, kid, 2)
    assert data[3].dtype == data[4].dtype == torch.uint8
    assert data[3].shape == (17, 2, 32)
    assert data[4].shape == (2, 64, 32)
    data[2].fill_(float("nan"))
    y = tune.run_bmm_mxscale_bench(*(data[i] for i in tune.BMM_BENCH_KEYS), kid, 2)
    torch.testing.assert_close(y, data[6], rtol=0.02, atol=0.03)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_wide_matches_decode_and_graph(dtype):
    x, w, xs, ws, ref = inputs(4, 256, 1024, 512, dtype)
    baseline = bmm_a8w8_mxscale_opus(
        x,
        w,
        xs,
        ws,
        dtype=dtype,
        kernelId=9700,
        b_preshuffled=True,
        group_n=1,
        group_size=32,
    )
    # On cancellation-heavy large matrices the existing scaled MFMA can differ
    # from a dequantized FP64 dot near zero. Check the independent reference in
    # aggregate and require the optimized tile to preserve every output bit.
    relative_error = (
        baseline.float() - ref.float()
    ).abs().mean() / ref.float().abs().mean()
    assert relative_error < 0.003
    actual = batched_gemm_a8w8_mxscale_bpreshuffle(x, w, xs, ws, dtype=dtype)
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = batched_gemm_a8w8_mxscale_bpreshuffle(x, w, xs, ws, dtype=dtype)
    for _ in range(2):
        actual.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
