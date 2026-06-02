"""Cross-layer FP8 GQA paged-attention decode tests for AITER.

The physical cross-layer layout stores all layers for a physical KV block next
to each other while preserving the per-layer K/V inner layouts consumed by the
project-internal FP8 GQA HIP decode path.
"""

import math
import random

import pytest
import torch

from aiter import dtypes
from aiter.ops.attention import _pa_fp8_gqa_eligible
from aiter.paged_attn import PagedAttention


FP8 = dtypes.fp8
NUM_Q_HEADS = 8
NUM_KV_HEADS = 1
HEAD_SIZE = 128
BLOCK_SIZE = 16
PARTITION_SIZE = 256


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/HIP device")


def check_decode_native_cross_layer_views(k_cache, v_cache):
    """Validate per-layer K/V views for the existing decode kernels."""
    assert k_cache.dim() == 5
    assert v_cache.dim() == 4

    num_blocks, num_kv_heads, head_div_x, block_size, x = k_cache.shape
    head_size = head_div_x * x
    assert x == 16
    assert v_cache.shape == (num_blocks, num_kv_heads, head_size, block_size)

    # Inner layout must remain the same as the decode-native contiguous layout.
    assert k_cache.stride(2) == block_size * x
    assert k_cache.stride(3) == x
    assert k_cache.stride(4) == 1
    assert v_cache.stride(2) == block_size
    assert v_cache.stride(3) == 1

    # The current FP8 launcher passes K's outer strides for both K and V.
    assert k_cache.stride(0) == v_cache.stride(0)
    assert k_cache.stride(1) == v_cache.stride(1)

    max_voffset = (
        (max(1, num_blocks) - 1) * k_cache.stride(0)
        + head_size * block_size
        - 1
    )
    assert max_voffset <= 0xFFFFFFFF


def select_cross_layer_kv_cache(k_cache_all_layers, v_cache_all_layers, layer_idx):
    """Return per-layer non-contiguous K/V views from physical cross-layer cache.

    Physical layout:
      K: [num_blocks, num_kv_heads, num_layers, head_size/16, block_size, 16]
      V: [num_blocks, num_kv_heads, num_layers, head_size,    block_size]
    """
    num_blocks, num_kv_heads, num_layers, head_div_x, block_size, x = (
        k_cache_all_layers.shape
    )
    head_size = head_div_x * x
    assert v_cache_all_layers.shape == (
        num_blocks, num_kv_heads, num_layers, head_size, block_size)
    assert 0 <= layer_idx < num_layers

    k_layer = k_cache_all_layers.permute(2, 0, 1, 3, 4, 5)[layer_idx]
    v_layer = v_cache_all_layers.permute(2, 0, 1, 3, 4)[layer_idx]
    check_decode_native_cross_layer_views(k_layer, v_layer)
    return k_layer, v_layer


def make_cross_layer_kv_cache(
    num_blocks, num_kv_heads, num_layers, head_size, block_size, device="cuda"
):
    """Allocate decode-native physical cross-layer K/V caches for tests."""
    assert head_size % 16 == 0
    k_cache_all_layers = torch.empty(
        (num_blocks, num_kv_heads, num_layers, head_size // 16, block_size, 16),
        dtype=FP8,
        device=device,
    )
    v_cache_all_layers = torch.empty(
        (num_blocks, num_kv_heads, num_layers, head_size, block_size),
        dtype=FP8,
        device=device,
    )
    return k_cache_all_layers, v_cache_all_layers


def make_case(num_seqs, ctx_len, mtp=1, seed=0):
    _require_cuda()
    random.seed(seed)
    torch.manual_seed(seed)

    blocks_per_seq = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = num_seqs * blocks_per_seq

    q_bf16 = torch.randn(
        num_seqs * mtp, NUM_Q_HEADS, HEAD_SIZE,
        dtype=dtypes.bf16, device="cuda")
    k_bf16 = torch.randn(
        num_blocks, NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE,
        dtype=dtypes.bf16, device="cuda")
    v_bf16 = torch.randn_like(k_bf16)

    fp8_max = torch.finfo(FP8).max
    q_scale = (q_bf16.float().abs().amax(dim=-1) / fp8_max).clamp_min(1e-6)
    k_scale = (k_bf16.float().abs().amax(dim=2) / fp8_max).clamp_min(1e-6)
    v_scale = (
        v_bf16.float()
        .abs()
        .permute(1, 0, 2, 3)
        .reshape(NUM_KV_HEADS, -1)
        .amax(dim=1)
        / fp8_max
    ).clamp_min(1e-6)

    q_fp8 = (q_bf16.float() / q_scale[:, :, None]).clamp(-fp8_max, fp8_max).to(FP8)
    k_fp8 = (k_bf16.float() / k_scale[:, :, None, :]).clamp(-fp8_max, fp8_max).to(FP8)
    v_fp8 = (v_bf16.float() / v_scale[None, :, None, None]).clamp(-fp8_max, fp8_max).to(FP8)
    k_fp8 = (
        k_fp8.view(num_blocks, NUM_KV_HEADS, HEAD_SIZE // 16, 16, BLOCK_SIZE)
        .permute(0, 1, 2, 4, 3)
        .contiguous()
    )

    block_table = torch.zeros(
        (num_seqs, blocks_per_seq), dtype=torch.int32, device="cuda")
    for s in range(num_seqs):
        block_table[s] = torch.tensor(
            [random.randint(0, num_blocks - 1) for _ in range(blocks_per_seq)],
            dtype=torch.int32,
            device="cuda",
        )
    context_lens = torch.full(
        (num_seqs,), ctx_len, dtype=torch.int32, device="cuda")

    return {
        "q": q_fp8,
        "k": k_fp8,
        "v": v_fp8,
        "qs": q_scale,
        "ks": k_scale,
        "vs": v_scale,
        "bt": block_table,
        "ctxl": context_lens,
        "ctx_len": ctx_len,
        "num_blocks": num_blocks,
    }


def make_cross_layer_cache(case, num_layers, layer_idx):
    k_all, v_all = make_cross_layer_kv_cache(
        case["num_blocks"],
        NUM_KV_HEADS,
        num_layers,
        HEAD_SIZE,
        BLOCK_SIZE,
        device="cuda",
    )
    k_all.zero_()
    v_all.zero_()
    k_all[:, :, layer_idx].copy_(case["k"])
    v_all[:, :, layer_idx].copy_(case["v"])
    return k_all, v_all


def run_aiter_fp8_gqa_decode(case, k_cache, v_cache, mtp):
    output = torch.empty_like(case["q"], dtype=dtypes.bf16)
    assert _pa_fp8_gqa_eligible(
        case["q"], k_cache, v_cache, output,
        NUM_KV_HEADS, BLOCK_SIZE, PARTITION_SIZE, mtp,
        None, None, case["qs"], case["ks"], case["vs"],
    )
    return PagedAttention.forward_decode(
        case["q"],
        k_cache,
        v_cache,
        case["bt"],
        case["ctxl"],
        case["ctx_len"],
        kv_cache_dtype="auto",
        num_kv_heads=NUM_KV_HEADS,
        scale=1.0 / math.sqrt(HEAD_SIZE),
        alibi_slopes=None,
        k_scale=case["ks"],
        v_scale=case["vs"],
        q_scale=case["qs"],
        mtp=mtp,
    )


def test_select_cross_layer_kv_cache_strides():
    c = make_case(num_seqs=2, ctx_len=256, mtp=1, seed=11)
    num_layers = 4
    layer_idx = 2
    k_all, v_all = make_cross_layer_cache(c, num_layers, layer_idx)

    k_view, v_view = select_cross_layer_kv_cache(k_all, v_all, layer_idx)

    assert not k_view.is_contiguous()
    assert not v_view.is_contiguous()
    assert k_view.shape == c["k"].shape
    assert v_view.shape == c["v"].shape
    assert k_view.stride(0) == c["k"].stride(0) * num_layers
    assert v_view.stride(0) == c["v"].stride(0) * num_layers
    assert k_view.stride(0) == v_view.stride(0)
    assert k_view.stride(1) == v_view.stride(1)
    assert k_view.stride()[2:] == c["k"].stride()[2:]
    assert v_view.stride()[2:] == c["v"].stride()[2:]


@pytest.mark.parametrize("mtp,num_seqs", [(1, 16), (2, 17)])
@pytest.mark.parametrize("layer_idx", [0, 2])
def test_decode_cross_layer_matches_contiguous_aiter(mtp, num_seqs, layer_idx):
    c = make_case(num_seqs=num_seqs, ctx_len=1024, mtp=mtp, seed=1234 + layer_idx)
    num_layers = 4
    k_all, v_all = make_cross_layer_cache(c, num_layers, layer_idx)

    out_ref = run_aiter_fp8_gqa_decode(c, c["k"], c["v"], mtp)
    k_view, v_view = select_cross_layer_kv_cache(k_all, v_all, layer_idx)
    out_xlayer = run_aiter_fp8_gqa_decode(c, k_view, v_view, mtp)

    torch.testing.assert_close(out_xlayer, out_ref, rtol=5e-1, atol=5e-2)
