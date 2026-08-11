# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter
from aiter.ops.triton.attention.mha import kimi_k3_fp8_prefill_gfx942
from aiter.ops.triton.quant.per_head import (
    dynamic_per_head_quant_fp8,
    static_per_head_quant_fp8,
)
from aiter.ops.triton.utils import types
from aiter.ops.triton.utils._triton import arch_info

pytestmark = pytest.mark.skipif(
    arch_info.get_arch() != "gfx942",
    reason="Kimi-K3 FP8 prefill is gfx942-only",
)


def _quantize_reference(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = torch.finfo(types.e4m3_dtype).max
    descale = (value.float().abs().amax(dim=(0, 2)) / maximum).clamp_min(1.0e-12)
    quantized = (value.float() / descale[None, :, None]).clamp(-maximum, maximum)
    return quantized.to(types.e4m3_dtype), descale[None, :].contiguous()


def _static_quantize_reference(
    value: torch.Tensor,
    descale: torch.Tensor,
) -> torch.Tensor:
    maximum = torch.finfo(types.e4m3_dtype).max
    quantized = (value.float() / descale[:, :, None]).clamp(-maximum, maximum)
    return quantized.to(types.e4m3_dtype)


def test_dynamic_per_head_quant_fp8() -> None:
    torch.manual_seed(7)
    value = torch.randn((257, 12, 192), dtype=torch.bfloat16, device="cuda")
    actual = dynamic_per_head_quant_fp8(value, types.e4m3_dtype)
    expected = _quantize_reference(value)
    torch.testing.assert_close(actual[1], expected[1], atol=1e-7, rtol=1e-6)
    mismatch_rate = (actual[0] != expected[0]).float().mean().item()
    assert mismatch_rate < 5e-3


def test_dynamic_per_head_quant_fp8_rejects_non_fp8_output() -> None:
    value = torch.empty((1, 1, 1), dtype=torch.float32)
    with pytest.raises(TypeError, match="unsupported FP8 output dtype"):
        dynamic_per_head_quant_fp8(value, torch.float16)


def test_static_per_head_quant_fp8_uses_supplied_descales_and_out() -> None:
    torch.manual_seed(17)
    value = torch.randn((257, 12, 192), dtype=torch.bfloat16, device="cuda")
    descale = torch.tensor(
        [
            0.0625,
            0.125,
            0.25,
            0.5,
            0.75,
            1.25,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            8.0,
        ],
        dtype=torch.float32,
        device="cuda",
    )[None, :]
    out = torch.empty_like(value, dtype=types.e4m3_dtype)
    out_ptr = out.data_ptr()

    actual = static_per_head_quant_fp8(
        value,
        descale,
        types.e4m3_dtype,
        out=out,
    )
    expected = _static_quantize_reference(value, descale)

    assert actual is out
    assert actual.data_ptr() == out_ptr
    mismatch_rate = (actual != expected).float().mean().item()
    assert mismatch_rate < 5e-3


def test_static_per_head_quant_fp8_hip_graph_replay() -> None:
    torch.manual_seed(23)
    value = torch.randn((65, 12, 192), dtype=torch.bfloat16, device="cuda")
    descale = torch.linspace(
        0.125,
        2.875,
        value.shape[1],
        dtype=torch.float32,
        device="cuda",
    )[None, :]
    out = torch.empty_like(value, dtype=types.e4m3_dtype)

    static_per_head_quant_fp8(value, descale, types.e4m3_dtype, out=out)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = static_per_head_quant_fp8(
            value,
            descale,
            types.e4m3_dtype,
            out=out,
        )

    captured_output = graph_out.clone()
    value.copy_(torch.randn_like(value))
    graph.replay()
    torch.cuda.synchronize()

    expected = _static_quantize_reference(value, descale)
    assert graph_out is out
    assert not torch.equal(graph_out, captured_output)
    mismatch_rate = (graph_out != expected).float().mean().item()
    assert mismatch_rate < 5e-3


def test_kimi_k3_fp8_prefill_reports_all_requirements(monkeypatch) -> None:
    monkeypatch.setattr(arch_info, "get_arch", lambda: "gfx950")
    with pytest.raises(
        NotImplementedError,
        match="requires gfx942 and torch.float8_e4m3fnuz support",
    ):
        kimi_k3_fp8_prefill_gfx942(
            q=None,
            k=None,
            v=None,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=1,
            max_seqlen_k=1,
            softmax_scale=1.0,
            causal=False,
            descale_q=None,
            descale_k=None,
            descale_v=None,
        )


def test_kimi_k3_fp8_prefill_rejects_invalid_out() -> None:
    q_len = 16
    k_len = 16
    q = torch.empty((q_len, 12, 192), dtype=types.e4m3_dtype, device="cuda")
    k = torch.empty((k_len, 12, 192), dtype=types.e4m3_dtype, device="cuda")
    v = torch.empty((k_len, 12, 128), dtype=types.e4m3_dtype, device="cuda")
    cu_q = torch.tensor([0, q_len], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, k_len], dtype=torch.int32, device="cuda")
    descale = torch.ones((1, 12), dtype=torch.float32, device="cuda")
    kwargs = {
        "q": q,
        "k": k,
        "v": v,
        "cu_seqlens_q": cu_q,
        "cu_seqlens_k": cu_k,
        "max_seqlen_q": q_len,
        "max_seqlen_k": k_len,
        "softmax_scale": 192**-0.5,
        "causal": False,
        "descale_q": descale,
        "descale_k": descale,
        "descale_v": descale,
    }

    with pytest.raises(ValueError, match="expected out shape"):
        kimi_k3_fp8_prefill_gfx942(
            **kwargs,
            out=torch.empty((q_len, 12, 127), dtype=torch.bfloat16, device="cuda"),
        )
    with pytest.raises(TypeError, match="expected out dtype"):
        kimi_k3_fp8_prefill_gfx942(
            **kwargs,
            out=torch.empty((q_len, 12, 128), dtype=torch.float16, device="cuda"),
        )
    with pytest.raises(ValueError, match="expected out on"):
        kimi_k3_fp8_prefill_gfx942(
            **kwargs,
            out=torch.empty((q_len, 12, 128), dtype=torch.bfloat16),
        )


@pytest.mark.parametrize(
    "q_len,k_len,causal",
    [
        (128, 65, False),
        (128, 4096, False),
        (4096, 4096, True),
    ],
)
def test_kimi_k3_fp8_prefill(
    q_len: int,
    k_len: int,
    causal: bool,
) -> None:
    torch.manual_seed(11)
    q_bf16 = torch.randn((q_len, 12, 192), dtype=torch.bfloat16, device="cuda")
    k_bf16 = torch.randn((k_len, 12, 192), dtype=torch.bfloat16, device="cuda")
    v_bf16 = torch.randn((k_len, 12, 128), dtype=torch.bfloat16, device="cuda")
    q, dq = dynamic_per_head_quant_fp8(q_bf16, types.e4m3_dtype)
    k, dk = dynamic_per_head_quant_fp8(k_bf16, types.e4m3_dtype)
    v, dv = dynamic_per_head_quant_fp8(v_bf16, types.e4m3_dtype)
    q_ref = (q.float() * dq[0][None, :, None]).to(torch.bfloat16)
    k_ref = (k.float() * dk[0][None, :, None]).to(torch.bfloat16)
    v_ref = (v.float() * dv[0][None, :, None]).to(torch.bfloat16)
    cu_q = torch.tensor([0, q_len], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, k_len], dtype=torch.int32, device="cuda")
    scale = 192**-0.5

    reference = aiter.flash_attn_varlen_func(
        q=q_ref,
        k=k_ref,
        v=v_ref,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=q_len,
        max_seqlen_k=k_len,
        softmax_scale=scale,
        causal=causal,
        return_lse=True,
        how_v3_bf16_cvt=1,
    )
    output, lse = kimi_k3_fp8_prefill_gfx942(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=q_len,
        max_seqlen_k=k_len,
        softmax_scale=scale,
        causal=causal,
        descale_q=dq,
        descale_k=dk,
        descale_v=dv,
    )
    torch.testing.assert_close(output, reference[0], atol=0.08, rtol=0.08)
    torch.testing.assert_close(lse, reference[1], atol=0.2, rtol=0.02)
