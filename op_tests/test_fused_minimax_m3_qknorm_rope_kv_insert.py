# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2024-2026, The vLLM team.
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter
from aiter.utility import dtypes

HEAD_DIM = 128
ROTARY_DIM = 64


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fused MiniMax-M3 qknorm/rope/cache tests require CUDA/ROCm",
)


def make_cos_sin_cache(max_pos: int, rotary_dim: int, dtype: torch.dtype) -> torch.Tensor:
    base = 5_000_000.0
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cuda")
            / rotary_dim
        )
    )
    positions = torch.arange(max_pos, dtype=torch.float32, device="cuda")
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype)


def gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    return xf * torch.rsqrt(variance + eps) * (1.0 + weight.float())


def apply_rope_neox_partial(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    half = rotary_dim // 2
    cos_sin = cos_sin_cache[positions].float()
    cos = cos_sin[..., :half].unsqueeze(1)
    sin = cos_sin[..., half:].unsqueeze(1)

    rot = x[..., :rotary_dim]
    x1 = rot[..., :half]
    x2 = rot[..., half:]
    out = x.clone()
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:rotary_dim] = x2 * cos + x1 * sin
    return out


def norm_rope_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    normed = gemma_rmsnorm(x.float(), weight, eps)
    return apply_rope_neox_partial(normed, positions, cos_sin_cache, ROTARY_DIM).to(
        dtype
    )


def make_sparse_case(num_tokens: int = 17, block_size: int = 16):
    torch.manual_seed(123)
    dtype = torch.bfloat16
    eps = 1e-6
    max_pos = 4096
    num_heads, num_kv_heads, num_index_heads = 16, 4, 4

    q_w = torch.randn(HEAD_DIM, dtype=dtype, device="cuda") * 0.1
    k_w = torch.randn(HEAD_DIM, dtype=dtype, device="cuda") * 0.1
    iq_w = torch.randn(HEAD_DIM, dtype=dtype, device="cuda") * 0.1
    ik_w = torch.randn(HEAD_DIM, dtype=dtype, device="cuda") * 0.1
    cos_sin = make_cos_sin_cache(max_pos, ROTARY_DIM, dtype)
    positions = torch.randint(0, max_pos, (num_tokens,), dtype=torch.int64, device="cuda")

    q_size = num_heads * HEAD_DIM
    kv_size = num_kv_heads * HEAD_DIM
    iq_size = num_index_heads * HEAD_DIM
    ik_size = HEAD_DIM
    qkv = torch.randn(
        num_tokens,
        q_size + 2 * kv_size + iq_size + ik_size,
        dtype=dtype,
        device="cuda",
    )

    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    slot_mapping = torch.randperm(
        num_blocks * block_size, dtype=torch.int64, device="cuda"
    )[:num_tokens]
    index_slot_mapping = torch.randperm(
        num_blocks * block_size, dtype=torch.int64, device="cuda"
    )[:num_tokens]

    return {
        "qkv": qkv,
        "q_norm_weight": q_w,
        "k_norm_weight": k_w,
        "index_q_norm_weight": iq_w,
        "index_k_norm_weight": ik_w,
        "cos_sin_cache": cos_sin,
        "positions": positions,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "num_index_heads": num_index_heads,
        "block_size": block_size,
        "num_blocks": num_blocks,
        "slot_mapping": slot_mapping,
        "index_slot_mapping": index_slot_mapping,
        "sizes": (q_size, kv_size, kv_size, iq_size, ik_size),
        "eps": eps,
        "dtype": dtype,
    }


def make_refs(case: dict, qkv_orig: torch.Tensor):
    q_size, kv_size, _, iq_size, ik_size = case["sizes"]
    q_in, k_in, v_in, iq_in, ik_in = qkv_orig.split(
        [q_size, kv_size, kv_size, iq_size, ik_size], dim=-1
    )
    num_tokens = qkv_orig.size(0)

    q_ref = norm_rope_ref(
        q_in.view(num_tokens, case["num_heads"], HEAD_DIM),
        case["q_norm_weight"],
        case["positions"],
        case["cos_sin_cache"],
        case["eps"],
        case["dtype"],
    ).view(num_tokens, q_size)
    k_ref = norm_rope_ref(
        k_in.view(num_tokens, case["num_kv_heads"], HEAD_DIM),
        case["k_norm_weight"],
        case["positions"],
        case["cos_sin_cache"],
        case["eps"],
        case["dtype"],
    )
    iq_ref = norm_rope_ref(
        iq_in.view(num_tokens, case["num_index_heads"], HEAD_DIM),
        case["index_q_norm_weight"],
        case["positions"],
        case["cos_sin_cache"],
        case["eps"],
        case["dtype"],
    ).view(num_tokens, iq_size)
    ik_ref = norm_rope_ref(
        ik_in.view(num_tokens, 1, HEAD_DIM),
        case["index_k_norm_weight"],
        case["positions"],
        case["cos_sin_cache"],
        case["eps"],
        case["dtype"],
    ).view(num_tokens, HEAD_DIM)
    v_ref = v_in.view(num_tokens, case["num_kv_heads"], HEAD_DIM)
    return q_ref, k_ref, v_ref, iq_ref, ik_ref


def test_fused_minimax_m3_qknorm_rope_kv_insert_bf16():
    case = make_sparse_case()
    qkv_orig = case["qkv"].clone()
    q_size, _, _, iq_size, _ = case["sizes"]
    q_out = torch.empty(case["qkv"].size(0), q_size, dtype=case["dtype"], device="cuda")
    index_q_out = torch.empty(
        case["qkv"].size(0), iq_size, dtype=case["dtype"], device="cuda"
    )
    kv_cache = torch.zeros(
        case["num_blocks"],
        2,
        case["block_size"],
        case["num_kv_heads"],
        HEAD_DIM,
        dtype=case["dtype"],
        device="cuda",
    )
    index_cache = torch.zeros(
        case["num_blocks"], case["block_size"], HEAD_DIM, dtype=case["dtype"], device="cuda"
    )

    aiter.fused_minimax_m3_qknorm_rope_kv_insert(
        case["qkv"],
        case["q_norm_weight"],
        case["k_norm_weight"],
        case["cos_sin_cache"],
        case["positions"],
        case["num_heads"],
        case["num_kv_heads"],
        ROTARY_DIM,
        case["eps"],
        case["index_q_norm_weight"],
        case["index_k_norm_weight"],
        case["num_index_heads"],
        case["slot_mapping"],
        kv_cache,
        index_cache,
        case["block_size"],
        q_out,
        index_q_out,
        case["index_slot_mapping"],
    )

    q_ref, k_ref, v_ref, iq_ref, ik_ref = make_refs(case, qkv_orig)
    torch.testing.assert_close(q_out, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(index_q_out, iq_ref, rtol=1e-2, atol=1e-2)

    for token in range(case["qkv"].size(0)):
        slot = case["slot_mapping"][token].item()
        block, offset = divmod(slot, case["block_size"])
        torch.testing.assert_close(
            kv_cache[block, 0, offset], k_ref[token], rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(kv_cache[block, 1, offset], v_ref[token], rtol=0, atol=0)

        index_slot = case["index_slot_mapping"][token].item()
        torch.testing.assert_close(
            index_cache.view(-1, HEAD_DIM)[index_slot], ik_ref[token], rtol=1e-2, atol=1e-2
        )


def test_fused_minimax_m3_qknorm_rope_kv_insert_fp8_dispatch():
    if dtypes.fp8 is torch.uint8:
        pytest.skip("native torch FP8 dtype is unavailable on this platform")

    case = make_sparse_case()
    qkv_orig = case["qkv"].clone()
    q_size, _, _, iq_size, _ = case["sizes"]
    q_out = torch.empty(case["qkv"].size(0), q_size, dtype=case["dtype"], device="cuda")
    index_q_out = torch.empty(
        case["qkv"].size(0), iq_size, dtype=case["dtype"], device="cuda"
    )
    kv_cache = torch.zeros(
        case["num_blocks"],
        2,
        case["block_size"],
        case["num_kv_heads"],
        HEAD_DIM,
        dtype=dtypes.fp8,
        device="cuda",
    )
    index_cache = torch.zeros(
        case["num_blocks"], case["block_size"], HEAD_DIM, dtype=case["dtype"], device="cuda"
    )
    k_scale = torch.tensor(0.75, dtype=torch.float32, device="cuda")
    v_scale = torch.tensor(1.25, dtype=torch.float32, device="cuda")

    aiter.fused_minimax_m3_qknorm_rope_kv_insert(
        case["qkv"],
        case["q_norm_weight"],
        case["k_norm_weight"],
        case["cos_sin_cache"],
        case["positions"],
        case["num_heads"],
        case["num_kv_heads"],
        ROTARY_DIM,
        case["eps"],
        case["index_q_norm_weight"],
        case["index_k_norm_weight"],
        case["num_index_heads"],
        case["slot_mapping"],
        kv_cache,
        index_cache,
        case["block_size"],
        q_out,
        index_q_out,
        case["index_slot_mapping"],
        kv_cache_dtype="fp8_e4m3",
        k_scale=k_scale,
        v_scale=v_scale,
    )

    q_ref, k_ref, v_ref, iq_ref, ik_ref = make_refs(case, qkv_orig)
    torch.testing.assert_close(q_out, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(index_q_out, iq_ref, rtol=1e-2, atol=1e-2)

    for token in range(case["qkv"].size(0)):
        slot = case["slot_mapping"][token].item()
        block, offset = divmod(slot, case["block_size"])
        torch.testing.assert_close(
            kv_cache[block, 0, offset].float() * k_scale,
            k_ref[token].float(),
            rtol=2.5e-1,
            atol=2.5e-1,
        )
        torch.testing.assert_close(
            kv_cache[block, 1, offset].float() * v_scale,
            v_ref[token].float(),
            rtol=2.5e-1,
            atol=2.5e-1,
        )

        index_slot = case["index_slot_mapping"][token].item()
        torch.testing.assert_close(
            index_cache.view(-1, HEAD_DIM)[index_slot], ik_ref[token], rtol=1e-2, atol=1e-2
        )
