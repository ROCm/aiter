# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.quant.fp8_group32 import quantize_fp8_group32
from aiter.ops.triton.quant.fused_mxfp8_quant import fused_dual_rmsnorm_mxfp8_quant
from aiter.ops.triton.utils._triton.arch_info import get_arch
from op_tests.triton_tests.utils import mono_native as native

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_arch() not in ("gfx1250", "gfx1250-strict"),
    reason="Standalone mono acceptance requires gfx1250",
)


def assert_bytes(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(
        actual.contiguous().view(torch.uint8),
        expected.contiguous().view(torch.uint8),
        rtol=0,
        atol=0,
    )


def guarded(shape, dtype, *, padded=False):
    if padded:
        raw = torch.full(
            (shape[0] + 2, shape[1] + 16), 0x5A, device="cuda", dtype=torch.uint8
        )
        if dtype == torch.bfloat16:
            raw = torch.full(
                (shape[0] + 2, (shape[1] + 16) * 2),
                0x5A,
                device="cuda",
                dtype=torch.uint8,
            )
        typed = raw.view(dtype)
        out = typed[1:-1, 8:-8]
        mask = torch.ones_like(raw, dtype=torch.bool)
        size = out.element_size()
        mask[1:-1, 8 * size : -8 * size] = False
    else:
        size = torch.empty((), dtype=dtype, device="cuda").element_size()
        count = 1
        for dim in shape:
            count *= dim
        raw = torch.full((count * size + 256,), 0x5A, device="cuda", dtype=torch.uint8)
        out = raw[128:-128].view(dtype).view(shape)
        mask = torch.ones_like(raw, dtype=torch.bool)
        mask[128:-128] = False
    return out, raw, mask


def assert_guards(outputs):
    for _, raw, mask in outputs:
        assert torch.all(raw[mask] == 0x5A)


@pytest.mark.parametrize("rows", [0, 1, 95, 96, 255, 256, 257, 513])
@pytest.mark.parametrize("widths", [(1280, 512), (96, 33)])
def test_dual_norm_native_persistent(rows, widths):
    torch.manual_seed(451)
    nq, nk = widths
    storage = torch.randn((rows, nq + nk), dtype=torch.bfloat16, device="cuda")
    q, k = storage[:, :nq], storage[:, nq:]
    weights = [torch.randn((n,), dtype=torch.bfloat16, device="cuda") for n in widths]
    if rows:
        storage[0].zero_()
    before = [t.clone() for t in (storage, *weights)]
    outputs = [
        guarded((rows, nq), torch.float8_e4m3fn, padded=True),
        guarded((rows, nq // 32), torch.uint8, padded=True),
        guarded((rows, nk), torch.bfloat16, padded=True),
    ]
    kwargs = {"eps_q": 1e-20, "eps_k": 1e-20, "amax_floor": 1e-10}
    expected = native.fused_dual_rmsnorm_mxfp8_quant(q, k, *weights, **kwargs)
    for (out, _, _), ref in zip(outputs, expected):
        out.view(torch.uint8).copy_(ref.view(torch.uint8) ^ 0xFF)
    actual = fused_dual_rmsnorm_mxfp8_quant(
        q,
        k,
        *weights,
        **kwargs,
        yq=outputs[0][0],
        sq=outputs[1][0],
        yk=outputs[2][0],
        persistent=True,
    )
    for a, b in zip(actual, expected):
        assert_bytes(a, b)
    for a, b in zip((storage, *weights), before):
        assert_bytes(a, b)
    assert_guards(outputs)
    if rows:
        kv64 = k.double() * torch.rsqrt(
            k.double().square().mean(-1, keepdim=True) + 1e-20
        )
        kv64 *= weights[1].double()
        torch.testing.assert_close(
            actual[2].float(), kv64.float(), rtol=1 / 128, atol=2**-12
        )


@pytest.mark.parametrize("floor", [0.0, 1e-10, 1e-4])
def test_dual_norm_separate_eps_and_floor(floor):
    torch.manual_seed(452)
    q = torch.randn((257, 1280), device="cuda", dtype=torch.bfloat16) * 1e-6
    k = torch.randn((257, 512), device="cuda", dtype=torch.bfloat16) * 1e-6
    wq = torch.full((1280,), 1e-6, device="cuda", dtype=torch.bfloat16)
    wk = torch.full((512,), 0.5, device="cuda", dtype=torch.bfloat16)
    args = (q, k, wq, wk, 1e-20, 1e-5)
    expected = native.fused_dual_rmsnorm_mxfp8_quant(*args, amax_floor=floor)
    actual = fused_dual_rmsnorm_mxfp8_quant(*args, amax_floor=floor, persistent=True)
    for a, b in zip(actual, expected):
        assert_bytes(a, b)


def fp8_oracle(x):
    groups = x.float().reshape(-1, 32)
    amax = groups.abs().amax(-1).clamp_min(1e-10)
    exponent = torch.ceil(torch.log2((amax * (1.0 / 448.0)).double()))
    scale = torch.exp2(exponent).float()
    q = (groups / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    codes = (exponent + 127).to(torch.uint8)
    return (
        q.reshape(x.shape),
        codes.reshape(*x.shape[:-1], x.shape[-1] // 32),
        (q.float() * scale[:, None]).to(x.dtype).reshape(x.shape),
    )


@pytest.mark.parametrize(
    "shape",
    [
        (0, 32),
        (1, 32),
        (3, 288),
        (17, 5120),
        (2, 3, 128),
        (96, 2048),
        (96, 5120),
        (96, 576),
        (128, 2048),
        (129, 2048),
        (256, 2048),
        (257, 2048),
        (512, 2048),
        (513, 2048),
        (513, 2080),
        (255, 1024),
        (256, 1024),
        (257, 1024),
        (513, 1024),
        (8193, 32),
    ],
)
def test_output_quant_native_persistent(shape):
    torch.manual_seed(312)
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    if x.numel():
        flat = x.view(-1)
        flat[:32] = 0
        if x.numel() >= 128:
            flat[32:64] *= 1e-12
            flat[64:96] *= 1e12
            flat[96:128] = -0.0
    before = x.clone()
    expected = native.quantize_fp8(x)
    oracle_q, oracle_s, oracle_qat = fp8_oracle(x)
    outputs = [
        guarded(shape, torch.float8_e4m3fn),
        guarded((*shape[:-1], shape[-1] // 32), torch.float8_e8m0fnu),
    ]
    qat_output = guarded(shape, torch.bfloat16)
    for persistent in (False, True):
        for (out, _, _), ref in zip(outputs, expected):
            out.view(torch.uint8).copy_(ref.view(torch.uint8) ^ 0xFF)
        actual = quantize_fp8_group32(
            x, persistent=persistent, out=outputs[0][0], scales=outputs[1][0]
        )
        for a, b in zip(actual, expected):
            assert_bytes(a, b)
        assert_bytes(actual[0], oracle_q)
        assert_bytes(actual[1].view(torch.uint8), oracle_s)
        expected_qat = native.quantize_fp8(x, dequantize=True)
        qat_output[0].view(torch.uint8).copy_(expected_qat.view(torch.uint8) ^ 0xFF)
        qat = quantize_fp8_group32(
            x, persistent=persistent, dequantize=True, out=qat_output[0]
        )
        assert_bytes(qat, expected_qat)
        assert_bytes(qat, oracle_qat)
        assert_guards(outputs)
        assert_guards([qat_output])
    assert_bytes(x, before)


def test_output_quant_scale_boundaries():
    values = torch.tensor(
        [0.0, -0.0, 1e-12, 224, 448, 450, 896, -448, -450],
        dtype=torch.bfloat16,
        device="cuda",
    )
    x = values[:, None].expand(-1, 32).contiguous()
    expected = native.quantize_fp8(x)
    actual = quantize_fp8_group32(x, persistent=True)
    oracle_q, oracle_s, _ = fp8_oracle(x)
    for a, b in zip(actual, expected):
        assert_bytes(a, b)
    assert_bytes(actual[0], oracle_q)
    assert_bytes(actual[1].view(torch.uint8), oracle_s)


def test_output_quant_scale_ownership():
    values = torch.tensor(
        [0.0, -0.0, 1e-12, 224, 448, 450, 896, -448, -450],
        dtype=torch.bfloat16,
        device="cuda",
    )
    groups = torch.arange(513 * 64, device="cuda") % values.numel()
    x = values[groups, None].expand(-1, 32).contiguous().view(513, 2048)
    before = x.clone()
    expected = native.quantize_fp8(x)
    oracle_q, oracle_s, _ = fp8_oracle(x)
    outputs = [
        guarded(x.shape, torch.float8_e4m3fn),
        guarded((513, 64), torch.float8_e8m0fnu),
    ]
    for persistent in (False, True):
        for (out, _, _), ref in zip(outputs, expected):
            out.view(torch.uint8).copy_(ref.view(torch.uint8) ^ 0xFF)
        actual = quantize_fp8_group32(
            x, persistent=persistent, out=outputs[0][0], scales=outputs[1][0]
        )
        for a, b in zip(actual, expected):
            assert_bytes(a, b)
        assert_bytes(actual[0], oracle_q)
        assert_bytes(actual[1].view(torch.uint8), oracle_s)
        assert_guards(outputs)
    assert_bytes(x, before)


def test_output_contract_rejects_bad_buffers():
    x = torch.zeros((96, 2048), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="contiguous BF16"):
        quantize_fp8_group32(x[:, ::2], persistent=True)
    bad = torch.empty((96, 2047), device="cuda", dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="out must have shape"):
        quantize_fp8_group32(x, persistent=True, out=bad)
    q, k = x[:, :1280], x[:, 1280:1792]
    wq = torch.ones((1280,), device="cuda", dtype=torch.bfloat16)
    wk = torch.ones((512,), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="yk must have shape"):
        fused_dual_rmsnorm_mxfp8_quant(q, k, wq, wk, 1e-20, persistent=True, yk=x)


@pytest.mark.parametrize("op", ["a03", "a09", "a09-five-iterations"])
def test_graph_replay_changes_inputs(op):
    torch.manual_seed(453)
    if op == "a03":
        x = torch.randn((513, 1792), device="cuda", dtype=torch.bfloat16)
        wq = torch.ones((1280,), device="cuda", dtype=torch.bfloat16)
        wk = torch.ones((512,), device="cuda", dtype=torch.bfloat16)
        args = (x[:, :1280], x[:, 1280:], wq, wk, 1e-20)
        outs = fused_dual_rmsnorm_mxfp8_quant(*args, amax_floor=1e-10, persistent=True)

        def run():
            return fused_dual_rmsnorm_mxfp8_quant(
                *args,
                amax_floor=1e-10,
                persistent=True,
                yq=outs[0],
                sq=outs[1],
                yk=outs[2],
            )

        def reference():
            return native.fused_dual_rmsnorm_mxfp8_quant(*args, amax_floor=1e-10)

    else:
        rows = 513 if op == "a09-five-iterations" else 257
        x = torch.randn((rows, 2048), device="cuda", dtype=torch.bfloat16)
        outs = quantize_fp8_group32(x, persistent=True)

        def run():
            return quantize_fp8_group32(x, persistent=True, out=outs[0], scales=outs[1])

        def reference():
            return native.quantize_fp8(x)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    previous = reference()
    for _ in range(20):
        x.normal_()
        graph.replay()
        expected = reference()
        assert any(
            not torch.equal(a.view(torch.uint8), b.view(torch.uint8))
            for a, b in zip(expected, previous)
        )
        for a, b in zip(outs, expected):
            assert_bytes(a, b)
        previous = expected
