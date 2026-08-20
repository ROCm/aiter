# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter
from aiter.ops.triton.rope.qwen3_next_fp8_qkv import (
    FP8_MAX,
    qwen3_next_fp8_qkv_prep,
)

NUM_QUERY_HEADS = 8
NUM_KV_HEADS = 1
HEAD_DIM = 256
ROTARY_DIM = 64
EPS = 1.0e-6


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


@pytest.mark.parametrize("lengths", [[128], [5, 17, 108], [8192]])
def test_qwen3_next_fp8_qkv_prep(lengths):
    inputs = _make_inputs(lengths)
    q_gate, key, value, q_weight, k_weight, cache, positions, cu_seqlens = inputs
    output = qwen3_next_fp8_qkv_prep(
        *inputs,
        num_actual_tokens=sum(lengths),
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


def test_qwen3_next_fp8_qkv_prep_mixed_decode_extend_suffix():
    lengths = [1, 1, 1, 16]
    inputs = _make_inputs(lengths)
    q_gate, key, value, q_weight, k_weight, cache, positions, cu_seqlens = inputs
    output = qwen3_next_fp8_qkv_prep(
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


def test_qwen3_next_fp8_qkv_prep_rejects_cpu_inputs():
    q_gate = torch.empty(1, NUM_QUERY_HEADS * 2 * HEAD_DIM)
    key = torch.empty(1, NUM_KV_HEADS * HEAD_DIM)
    with pytest.raises(ValueError, match="requires a CUDA/HIP device"):
        qwen3_next_fp8_qkv_prep(
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


def test_qwen3_next_fp8_qkv_output_dtype():
    assert aiter.dtypes.fp8 in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
