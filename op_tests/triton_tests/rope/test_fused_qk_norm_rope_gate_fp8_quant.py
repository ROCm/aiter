# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter
from aiter import flash_attn_varlen_func
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops import mha as mha_module
from aiter.ops.triton.rope import (
    fused_qk_norm_rope_gate_fp8_quant as fp8_quant_module,
)
from aiter.ops.triton.rope.fused_qk_norm_rope_gate_fp8_quant import (
    FP8_DTYPE,
    FP8_MAX,
    fused_qk_norm_rope_gate_fp8_quant,
)

NUM_QUERY_HEADS = 8
NUM_KV_HEADS = 1
HEAD_DIM = 256
ROTARY_DIM = 64
EPS = 1.0e-6
requires_gfx950 = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_runtime() != "gfx950",
    reason="requires gfx950",
)


def _make_cos_sin_cache(tokens: int, device: torch.device) -> torch.Tensor:
    half = ROTARY_DIM // 2
    angles = torch.arange(tokens, dtype=torch.float32, device=device)[
        :, None
    ] * torch.exp(
        -torch.arange(half, dtype=torch.float32, device=device)[None, :]
        * (torch.log(torch.tensor(10000.0, device=device)) / half)
    )
    return torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)


def _reference_qk_gate(
    q_gate: torch.Tensor,
    key: torch.Tensor,
    query_norm_weight: torch.Tensor,
    key_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = q_gate.shape[0]
    q_gate_view = q_gate.view(tokens, NUM_QUERY_HEADS, 2 * HEAD_DIM)
    query = q_gate_view[..., :HEAD_DIM]
    gate = q_gate_view[..., HEAD_DIM:]
    key_view = key.view(tokens, NUM_KV_HEADS, HEAD_DIM)

    def normalize(values: torch.Tensor, raw_weight: torch.Tensor) -> torch.Tensor:
        inv_rms = torch.rsqrt(values.float().square().mean(dim=-1, keepdim=True) + EPS)
        return (values.float() * inv_rms * (raw_weight.float() + 1.0)).to(values.dtype)

    query = normalize(query, query_norm_weight)
    key_view = normalize(key_view, key_norm_weight)
    half = ROTARY_DIM // 2
    cos = cos_sin_cache[positions, :half].float()
    sin = cos_sin_cache[positions, half:].float()

    def apply_rope(values: torch.Tensor) -> torch.Tensor:
        first = values[..., :half].float()
        second = values[..., half:ROTARY_DIM].float()
        rotated_first = (first * cos[:, None] - second * sin[:, None]).to(values.dtype)
        rotated_second = (second * cos[:, None] + first * sin[:, None]).to(values.dtype)
        return torch.cat(
            (
                rotated_first,
                rotated_second,
                values[..., ROTARY_DIM:],
            ),
            dim=-1,
        )

    return (
        apply_rope(query).reshape(tokens, -1),
        apply_rope(key_view).reshape(tokens, -1),
        gate.reshape(tokens, -1),
    )


def _expected_descales(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    quant_sequence_start: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = query.view(-1, NUM_QUERY_HEADS, HEAD_DIM).float()
    key = key.view(-1, NUM_KV_HEADS, HEAD_DIM).float()
    value = value.view(-1, NUM_KV_HEADS, HEAD_DIM).float()
    num_sequences = cu_seqlens.numel() - 1
    output = [
        torch.empty(
            (num_sequences, NUM_KV_HEADS),
            dtype=torch.float32,
            device=query.device,
        )
        for _ in range(3)
    ]
    for sequence in range(quant_sequence_start, num_sequences):
        start = int(cu_seqlens[sequence].item())
        end = int(cu_seqlens[sequence + 1].item())
        output[0][sequence, 0] = (query[start:end].abs().amax() / FP8_MAX).clamp_min(
            1.0e-12
        )
        output[1][sequence, 0] = (key[start:end].abs().amax() / FP8_MAX).clamp_min(
            1.0e-12
        )
        output[2][sequence, 0] = (value[start:end].abs().amax() / FP8_MAX).clamp_min(
            1.0e-12
        )
    return output[0], output[1], output[2]


def _make_inputs(lengths: list[int]):
    device = torch.device("cuda")
    total_tokens = sum(lengths)
    torch.manual_seed(1234 + total_tokens)
    q_gate = torch.randn(
        total_tokens,
        NUM_QUERY_HEADS * 2 * HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    key = torch.randn(
        total_tokens,
        NUM_KV_HEADS * HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    value = torch.randn_like(key)
    query_norm_weight = torch.linspace(
        -0.125,
        0.125,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    key_norm_weight = torch.linspace(
        0.125,
        -0.125,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    positions = torch.arange(total_tokens, dtype=torch.int64, device=device)
    cos_sin_cache = _make_cos_sin_cache(total_tokens, device)
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    return (
        q_gate,
        key,
        value,
        query_norm_weight,
        key_norm_weight,
        cos_sin_cache,
        positions,
        cu_seqlens,
    )


@pytest.mark.parametrize("lengths", [[128], [5, 17, 108], [8193]])
@pytest.mark.parametrize("preallocate_outputs", [False, True])
@requires_gfx950
def test_fused_qk_norm_rope_gate_fp8_quant(lengths, preallocate_outputs):
    inputs = _make_inputs(lengths)
    q_gate, key, value, q_weight, k_weight, cache, positions, cu_seqlens = inputs
    output_kwargs = {}
    expected_outputs = None
    if preallocate_outputs:
        total_tokens = sum(lengths)
        expected_outputs = (
            torch.empty(
                total_tokens,
                NUM_QUERY_HEADS * HEAD_DIM,
                dtype=q_gate.dtype,
                device=q_gate.device,
            ),
            torch.empty_like(key),
            torch.empty(
                total_tokens,
                NUM_QUERY_HEADS * HEAD_DIM,
                dtype=q_gate.dtype,
                device=q_gate.device,
            ),
            torch.empty(
                total_tokens,
                NUM_QUERY_HEADS,
                HEAD_DIM,
                dtype=aiter.dtypes.fp8,
                device=q_gate.device,
            ),
            torch.empty(
                total_tokens,
                NUM_KV_HEADS,
                HEAD_DIM,
                dtype=aiter.dtypes.fp8,
                device=q_gate.device,
            ),
            torch.empty(
                total_tokens,
                NUM_KV_HEADS,
                HEAD_DIM,
                dtype=aiter.dtypes.fp8,
                device=q_gate.device,
            ),
            torch.empty(
                256,
                NUM_KV_HEADS,
                dtype=torch.float32,
                device=q_gate.device,
            ),
            torch.empty(
                256,
                NUM_KV_HEADS,
                dtype=torch.float32,
                device=q_gate.device,
            ),
            torch.empty(
                256,
                NUM_KV_HEADS,
                dtype=torch.float32,
                device=q_gate.device,
            ),
        )
        output_kwargs = {
            "query_out": expected_outputs[0],
            "key_out": expected_outputs[1],
            "gate_out": expected_outputs[2],
            "query_fp8_out": expected_outputs[3],
            "key_fp8_out": expected_outputs[4],
            "value_fp8_out": expected_outputs[5],
            "query_descale_out": expected_outputs[6],
            "key_descale_out": expected_outputs[7],
            "value_descale_out": expected_outputs[8],
        }
    output = fused_qk_norm_rope_gate_fp8_quant(
        *inputs,
        num_actual_tokens=sum(lengths),
        num_query_heads=NUM_QUERY_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        eps=EPS,
        **output_kwargs,
    )
    torch.cuda.synchronize()
    if expected_outputs is not None:
        for actual, expected in zip(output, expected_outputs):
            assert actual is expected

    ref_query, ref_key, ref_gate = _reference_qk_gate(
        q_gate,
        key,
        q_weight,
        k_weight,
        cache,
        positions,
    )
    torch.testing.assert_close(output.query, ref_query, rtol=1.0e-2, atol=1.0e-2)
    torch.testing.assert_close(output.key, ref_key, rtol=1.0e-2, atol=1.0e-2)
    torch.testing.assert_close(output.gate, ref_gate, rtol=0, atol=0)

    expected_descales = _expected_descales(
        output.query,
        output.key,
        value,
        cu_seqlens,
        quant_sequence_start=0,
    )
    actual_descales = (
        output.query_descale,
        output.key_descale,
        output.value_descale,
    )
    for actual, expected in zip(actual_descales, expected_descales):
        torch.testing.assert_close(
            actual[: len(lengths)],
            expected,
            rtol=2.0e-6,
            atol=1.0e-8,
        )

    references = (
        output.query.view(-1, NUM_QUERY_HEADS, HEAD_DIM).float(),
        output.key.view(-1, NUM_KV_HEADS, HEAD_DIM).float(),
        value.view(-1, NUM_KV_HEADS, HEAD_DIM).float(),
    )
    quantized = (output.query_fp8, output.key_fp8, output.value_fp8)
    for sequence in range(len(lengths)):
        start = int(cu_seqlens[sequence].item())
        end = int(cu_seqlens[sequence + 1].item())
        for reference, fp8, descale in zip(references, quantized, actual_descales):
            reconstructed = fp8[start:end].float() * descale[sequence, 0]
            relative_error = (
                reconstructed - reference[start:end]
            ).abs().amax() / reference[start:end].abs().amax()
            assert relative_error < 0.04
            assert torch.isfinite(reconstructed).all()


@requires_gfx950
def test_fused_qk_norm_rope_gate_fp8_quant_mixed_decode_extend_suffix():
    lengths = [1, 1, 1, 16]
    inputs = _make_inputs(lengths)
    q_gate, key, value, q_weight, k_weight, cache, positions, cu_seqlens = inputs
    output = fused_qk_norm_rope_gate_fp8_quant(
        *inputs,
        num_actual_tokens=sum(lengths),
        quant_token_start=3,
        quant_sequence_start=3,
        num_query_heads=NUM_QUERY_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        eps=EPS,
    )
    torch.cuda.synchronize()

    ref_query, ref_key, ref_gate = _reference_qk_gate(
        q_gate,
        key,
        q_weight,
        k_weight,
        cache,
        positions,
    )
    torch.testing.assert_close(output.query, ref_query, rtol=0, atol=0)
    torch.testing.assert_close(output.key, ref_key, rtol=0, atol=0)
    torch.testing.assert_close(output.gate, ref_gate, rtol=0, atol=0)

    expected_descales = _expected_descales(
        output.query,
        output.key,
        value,
        cu_seqlens,
        quant_sequence_start=3,
    )
    for actual, expected in zip(
        (
            output.query_descale,
            output.key_descale,
            output.value_descale,
        ),
        expected_descales,
    ):
        torch.testing.assert_close(
            actual[3:4],
            expected[3:4],
            rtol=2.0e-6,
            atol=1.0e-8,
        )

    references = (
        output.query.view(-1, NUM_QUERY_HEADS, HEAD_DIM).float(),
        output.key.view(-1, NUM_KV_HEADS, HEAD_DIM).float(),
        value.view(-1, NUM_KV_HEADS, HEAD_DIM).float(),
    )
    quantized = (output.query_fp8, output.key_fp8, output.value_fp8)
    descales = (
        output.query_descale,
        output.key_descale,
        output.value_descale,
    )
    for reference, fp8, descale in zip(references, quantized, descales):
        reconstructed = fp8[3:].float() * descale[3, 0]
        relative_error = (reconstructed - reference[3:]).abs().amax() / reference[
            3:
        ].abs().amax()
        assert relative_error < 0.04


def test_fused_qk_norm_rope_gate_fp8_quant_rejects_cpu_inputs():
    q_gate = torch.empty(1, NUM_QUERY_HEADS * 2 * HEAD_DIM)
    key = torch.empty(1, NUM_KV_HEADS * HEAD_DIM)
    with pytest.raises(ValueError, match="requires a CUDA/HIP device"):
        fused_qk_norm_rope_gate_fp8_quant(
            q_gate,
            key,
            key,
            torch.empty(HEAD_DIM),
            torch.empty(HEAD_DIM),
            torch.empty(1, ROTARY_DIM),
            torch.zeros(1, dtype=torch.int64),
            torch.tensor([0, 1], dtype=torch.int32),
            num_actual_tokens=1,
            num_query_heads=NUM_QUERY_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            rotary_dim=ROTARY_DIM,
        )


def test_fused_qk_norm_rope_gate_fp8_quant_rejects_non_gfx950(monkeypatch):
    monkeypatch.setattr(fp8_quant_module, "get_gfx_runtime", lambda: "gfx942")
    with pytest.raises(RuntimeError, match="supported only on gfx950"):
        fp8_quant_module._validate_gfx950_fp8()


def test_fused_qk_norm_rope_gate_fp8_quant_rejects_non_fn_fp8(monkeypatch):
    monkeypatch.setattr(fp8_quant_module, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(aiter.dtypes, "fp8", torch.float8_e4m3fnuz)
    with pytest.raises(RuntimeError, match="requires float8_e4m3fn"):
        fp8_quant_module._validate_gfx950_fp8()


def test_fused_qk_norm_rope_gate_fp8_quant_output_dtype():
    assert FP8_DTYPE == torch.float8_e4m3fn


@requires_gfx950
def test_fused_qk_norm_rope_gate_fp8_quant_rejects_fp16_inputs():
    inputs = tuple(tensor.cuda() for tensor in _make_inputs([8]))
    inputs = tuple(
        tensor.to(torch.float16) if tensor.dtype == torch.bfloat16 else tensor
        for tensor in inputs
    )
    with pytest.raises(ValueError, match="q_gate must use bfloat16"):
        fused_qk_norm_rope_gate_fp8_quant(
            *inputs,
            num_actual_tokens=inputs[0].shape[0],
            num_query_heads=NUM_QUERY_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            rotary_dim=ROTARY_DIM,
            eps=EPS,
        )


@requires_gfx950
def test_fused_qk_norm_rope_gate_fp8_quant_rejects_noncontiguous_inner_dim():
    inputs = list(_make_inputs([8]))
    q_gate = torch.randn(
        inputs[0].shape[0],
        inputs[0].shape[1] * 2,
        dtype=torch.bfloat16,
        device="cuda",
    )[:, ::2]
    assert q_gate.stride(-1) != 1
    inputs[0] = q_gate
    with pytest.raises(ValueError, match="contiguous innermost dimension"):
        fused_qk_norm_rope_gate_fp8_quant(
            *inputs,
            num_actual_tokens=q_gate.shape[0],
            num_query_heads=NUM_QUERY_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            rotary_dim=ROTARY_DIM,
            eps=EPS,
        )


@requires_gfx950
def test_fused_qk_norm_rope_gate_fp8_quant_masks_misaligned_suffix():
    lengths = [1, 1, 1, 16]
    inputs = _make_inputs(lengths)
    output = fused_qk_norm_rope_gate_fp8_quant(
        *inputs,
        num_actual_tokens=sum(lengths),
        quant_token_start=4,
        quant_sequence_start=3,
        num_query_heads=NUM_QUERY_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        eps=EPS,
    )
    reconstructed = output.value_fp8[4:].float() * output.value_descale[3, 0]
    assert torch.isfinite(reconstructed).all()


@requires_gfx950
def test_fused_qk_norm_rope_gate_fp8_quant_bounds_padded_tokens():
    inputs = _make_inputs([8, 8])
    output = fused_qk_norm_rope_gate_fp8_quant(
        *inputs,
        num_actual_tokens=12,
        num_query_heads=NUM_QUERY_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        eps=EPS,
    )
    assert torch.isfinite(output.query_descale[:2]).all()
    assert torch.isfinite(output.key_descale[:2]).all()
    assert torch.isfinite(output.value_descale[:2]).all()


def test_flash_attn_varlen_rejects_descales_without_ck(monkeypatch):
    monkeypatch.setattr(mha_module, "ENABLE_CK", False)
    q = torch.empty((1, 1, HEAD_DIM), dtype=FP8_DTYPE)
    descale = torch.ones((1, 1), dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 1], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="requires ENABLE_CK=1"):
        mha_module.flash_attn_varlen_func(
            q,
            q,
            q,
            cu_seqlens,
            cu_seqlens,
            1,
            1,
            q_descale=descale,
            k_descale=descale,
            v_descale=descale,
        )


def test_flash_attn_varlen_rejects_partial_descales():
    q = torch.empty((1, 1, HEAD_DIM), dtype=FP8_DTYPE)
    descale = torch.ones((1, 1), dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 1], dtype=torch.int32)
    with pytest.raises(
        ValueError, match="requires q_descale, k_descale, and v_descale"
    ):
        mha_module.flash_attn_varlen_func(
            q,
            q,
            q,
            cu_seqlens,
            cu_seqlens,
            1,
            1,
            q_descale=descale,
        )


@requires_gfx950
def test_fused_qk_norm_rope_gate_fp8_quant_fmha_abi():
    lengths = [128]
    inputs = _make_inputs(lengths)
    output = fused_qk_norm_rope_gate_fp8_quant(
        *inputs,
        num_actual_tokens=128,
        num_query_heads=NUM_QUERY_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        eps=EPS,
    )
    cu_seqlens = inputs[-1]
    attention_kwargs = {
        "cu_seqlens_q": cu_seqlens,
        "cu_seqlens_k": cu_seqlens,
        "max_seqlen_q": 128,
        "max_seqlen_k": 128,
        "min_seqlen_q": 1,
        "dropout_p": 0.0,
        "softmax_scale": HEAD_DIM**-0.5,
        "causal": True,
        "window_size": (-1, -1),
    }
    reference = flash_attn_varlen_func(
        output.query.view(128, NUM_QUERY_HEADS, HEAD_DIM),
        output.key.view(128, NUM_KV_HEADS, HEAD_DIM),
        inputs[2].view(128, NUM_KV_HEADS, HEAD_DIM),
        **attention_kwargs,
    )
    actual = flash_attn_varlen_func(
        output.query_fp8,
        output.key_fp8,
        output.value_fp8,
        q_descale=output.query_descale[:1],
        k_descale=output.key_descale[:1],
        v_descale=output.value_descale[:1],
        **attention_kwargs,
    )
    torch.cuda.synchronize()

    relative_error = (
        actual.float() - reference.float()
    ).abs().amax() / reference.float().abs().amax()
    assert relative_error < 0.08
    assert torch.isfinite(actual).all()
