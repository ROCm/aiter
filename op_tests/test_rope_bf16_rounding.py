import pytest
import torch

import aiter
from aiter.ops.triton.fusions.fused_kv_cache import (
    fused_qk_rope_cat_and_cache_mla,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.types import e4m3_dtype

DEVICE_ARCH = arch_info.get_arch()
pytestmark = pytest.mark.skipif(
    DEVICE_ARCH not in ("gfx1250", "gfx1250-strict"),
    reason="gfx1250 BF16 RoPE rounding regression",
)


def _deterministic_bf16(shape, modulus, divisor):
    values = torch.arange(torch.tensor(shape).prod().item(), device="cuda")
    values = ((values % modulus) - modulus // 2).float() / divisor
    return values.reshape(shape).bfloat16()


def _rope_reference(x, cos, sin, positions, neox):
    freqs_cos = cos.view(cos.shape[0], -1)[positions.reshape(-1)]
    freqs_sin = sin.view(sin.shape[0], -1)[positions.reshape(-1)]
    if x.ndim == 4:
        freqs_cos = freqs_cos.view(x.shape[0], x.shape[1], 1, -1)
        freqs_sin = freqs_sin.view(x.shape[0], x.shape[1], 1, -1)
    else:
        freqs_cos = freqs_cos.view(x.shape[0], 1, -1)
        freqs_sin = freqs_sin.view(x.shape[0], 1, -1)

    if neox:
        x0, x1 = x.chunk(2, dim=-1)
    else:
        x0, x1 = x[..., 0::2], x[..., 1::2]

    y0 = (x0.float() * freqs_cos.float() - (x1 * freqs_sin).float()).bfloat16()
    y1 = (x1.float() * freqs_cos.float() + (x0 * freqs_sin).float()).bfloat16()
    if neox:
        return torch.cat((y0, y1), dim=-1)
    result = torch.empty_like(x)
    result[..., 0::2] = y0
    result[..., 1::2] = y1
    return result


@pytest.mark.parametrize("neox", [False, True])
def test_cached_positions_2c_bf16_rounding(neox):
    tokens, query_heads, head_size = 13, 128, 64
    query = _deterministic_bf16((1, tokens, query_heads, head_size), 509, 37.0)
    key = _deterministic_bf16((1, tokens, 1, head_size), 383, 29.0)
    positions = torch.tensor(
        [[0, 1, 2, 3, 7, 11, 17, 23, 31, 47, 63, 95, 127]],
        device="cuda",
        dtype=torch.int64,
    )
    cos = _deterministic_bf16((256, 1, 1, head_size // 2), 257, 128.0)
    sin = _deterministic_bf16((256, 1, 1, head_size // 2), 263, 131.0)
    query_ref = _rope_reference(query, cos, sin, positions, neox)
    key_ref = _rope_reference(key, cos, sin, positions, neox)

    aiter.rope_cached_positions_2c_fwd_inplace(
        query,
        key,
        cos,
        sin,
        positions,
        int(not neox),
        reuse_freqs_front_part=True,
        nope_first=False,
    )

    assert torch.equal(query, query_ref)
    assert torch.equal(key, key_ref)


def test_fused_decode_rope_fp8_qout_rounding():
    tokens, query_heads, lora_dim, rope_dim = 1, 128, 512, 64
    q_nope = _deterministic_bf16((tokens, query_heads, lora_dim), 509, 3700.0)
    q_pe = _deterministic_bf16((tokens, query_heads, rope_dim), 383, 29.0)
    k_nope = _deterministic_bf16((tokens, 1, lora_dim), 277, 3700.0)
    k_pe = _deterministic_bf16((tokens, 1, rope_dim), 269, 31.0)
    positions = torch.tensor([14], device="cuda", dtype=torch.int64)
    cos = _deterministic_bf16((256, 1, 1, rope_dim // 2), 257, 128.0)
    sin = _deterministic_bf16((256, 1, 1, rope_dim // 2), 263, 131.0)
    q_pe_ref = _rope_reference(
        q_pe,
        cos,
        sin,
        positions,
        neox=False,
    )
    q_out_ref = torch.cat((q_nope, q_pe_ref), dim=-1).to(e4m3_dtype)

    kv_cache = torch.empty(
        (1, 1, 64, lora_dim + rope_dim),
        dtype=e4m3_dtype,
        device="cuda",
    )
    q_out = torch.empty_like(q_out_ref)
    fused_qk_rope_cat_and_cache_mla(
        q_nope,
        q_pe,
        k_nope,
        k_pe,
        kv_cache,
        torch.tensor([14], device="cuda", dtype=torch.int64),
        positions,
        cos,
        sin,
        torch.tensor(1.0, device="cuda"),
        False,
        q_out=q_out,
        shuffled_kv_cache=True,
    )

    assert torch.equal(q_out, q_out_ref)
