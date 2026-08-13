# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Per-token fp8 KV dequant scales in unified_attention (gfx1250 3D gluon).

``k_descale``/``v_descale`` are normally one scalar for the whole cache. A
``[num_blocks, num_kv_heads, block_size]`` fp32 tensor instead carries one scale
per cached token, which is what dynamic per-token fp8 KV quantization produces
(MiniMax-M3 sparse attention). The scalar path folds the descale into the
softmax scale and the output factor; the per-token path folds K's onto the score
columns and V's onto the softmax probabilities.
"""

import pytest
import torch

from aiter.ops.triton.attention.unified_attention import unified_attention
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.types import e4m3_dtype
from aiter.test_common import checkAllclose
from op_tests.triton_tests.attention.test_unified_attention import (
    ref_paged_attn,
    shuffle_kv_cache,
)

DEVICE_ARCH = arch_info.get_arch()


def quant_kv_cache_per_token(cache: torch.Tensor):
    """Dynamic per-token fp8 quantization of a [nb, block, nkv, hd] KV cache.

    Returns ``(cache_fp8, scales, cache_dequant)`` with ``scales`` shaped
    ``[nb, nkv, block]`` fp32 -- the layout the kernel indexes by block table
    entry and KV head.
    """
    fp8_max = torch.finfo(e4m3_dtype).max
    amax = cache.abs().amax(dim=-1, keepdim=True).to(torch.float32)
    scales = (amax / fp8_max).clamp(min=1e-12)
    cache_fp8 = (
        (cache.to(torch.float32) / scales).clamp(-fp8_max, fp8_max).to(e4m3_dtype)
    )
    cache_dequant = (cache_fp8.to(torch.float32) * scales).to(cache.dtype)
    # [nb, block, nkv, 1] -> [nb, nkv, block]
    scales = scales.squeeze(-1).permute(0, 2, 1).contiguous()
    return cache_fp8, scales, cache_dequant


@pytest.mark.parametrize(
    "kv_lens",
    [
        [37],
        [16, 128, 523],
        [1328, 47, 8192, 129],
    ],
)
@pytest.mark.parametrize("num_heads", [(8, 1), (64, 8)])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("block_size", [16, 128])
@torch.inference_mode()
def test_unified_attn_3d_per_token_kv_descale(
    kv_lens: list[int],
    num_heads: tuple[int, int],
    head_size: int,
    block_size: int,
) -> None:
    if DEVICE_ARCH != "gfx1250":
        pytest.skip(f"per-token KV descale is gfx1250-only, got {DEVICE_ARCH}")

    torch.manual_seed(0)
    device = "cuda"
    num_query_heads, num_kv_heads = num_heads
    num_seqs = len(kv_lens)
    scale = head_size**-0.5

    max_kv_len = max(kv_lens)
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    num_blocks = max(1024, num_seqs * max_num_blocks_per_seq)

    # Decode: one query token per sequence (max_seqlen_q == 1 -> 3D kernel).
    query = torch.randn(
        num_seqs, num_query_heads, head_size, dtype=torch.bfloat16, device=device
    )
    key_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=torch.bfloat16,
        device=device,
    )
    value_cache = torch.randn_like(key_cache)

    key_fp8, k_descale, key_dequant = quant_kv_cache_per_token(key_cache)
    value_fp8, v_descale, value_dequant = quant_kv_cache_per_token(value_cache)
    key_shuffled, value_shuffled = shuffle_kv_cache(key_fp8, value_fp8)

    block_tables = torch.randint(
        0,
        num_blocks,
        (num_seqs, max_num_blocks_per_seq),
        dtype=torch.int32,
        device=device,
    )
    cu_query_lens = torch.arange(num_seqs + 1, dtype=torch.int32, device=device)
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    output = torch.empty_like(query)

    unified_attention(
        q=query,
        k=key_shuffled,
        v=value_shuffled,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_t,
        max_seqlen_q=1,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0,
        q_descale=None,
        k_descale=k_descale,
        v_descale=v_descale,
        shuffled_kv_cache=True,
    )

    # Reference reads the same dequantized values, so this compares the kernel's
    # arithmetic only -- fp8 quantization error is present on both sides.
    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_dequant,
        value_cache=value_dequant,
        query_lens=[1] * num_seqs,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        out_dtype=torch.bfloat16,
    )

    tol_err_ratio = 0.01
    assert (
        checkAllclose(
            output.to(torch.bfloat16),
            ref_output.to(torch.bfloat16),
            atol=1.5e-2,
            rtol=1e-2,
            tol_err_ratio=tol_err_ratio,
            msg="per-token KV descale output",
        )
        <= tol_err_ratio
    )


@pytest.mark.parametrize("case", ["not_shuffled", "bf16_kv", "bad_shape", "k_only"])
@torch.inference_mode()
def test_unified_attn_per_token_kv_descale_rejects(case: str) -> None:
    """Unsupported per-token descale configs must raise, never silently drop it."""
    if DEVICE_ARCH != "gfx1250":
        pytest.skip(f"per-token KV descale is gfx1250-only, got {DEVICE_ARCH}")

    device = "cuda"
    num_seqs, num_query_heads, num_kv_heads, head_size = 2, 8, 1, 128
    block_size, num_blocks, kv_len = 16, 64, 32

    query = torch.randn(
        num_seqs, num_query_heads, head_size, dtype=torch.bfloat16, device=device
    )
    key_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=torch.bfloat16,
        device=device,
    )
    value_cache = torch.randn_like(key_cache)
    key_fp8, k_descale, _ = quant_kv_cache_per_token(key_cache)
    value_fp8, v_descale, _ = quant_kv_cache_per_token(value_cache)

    shuffled_kv_cache = True
    if case == "bf16_kv":
        k, v = shuffle_kv_cache(key_cache, value_cache)
    elif case == "not_shuffled":
        k, v = key_fp8, value_fp8
        shuffled_kv_cache = False
    else:
        k, v = shuffle_kv_cache(key_fp8, value_fp8)
    if case == "bad_shape":
        k_descale = k_descale.reshape(num_blocks, block_size, num_kv_heads)
    if case == "k_only":
        v_descale = torch.tensor([1.0], dtype=torch.float32, device=device)

    with pytest.raises((NotImplementedError, ValueError)):
        unified_attention(
            q=query,
            k=k,
            v=v,
            out=torch.empty_like(query),
            cu_seqlens_q=torch.arange(num_seqs + 1, dtype=torch.int32, device=device),
            seqused_k=torch.full((num_seqs,), kv_len, dtype=torch.int32, device=device),
            max_seqlen_q=1,
            max_seqlen_k=kv_len,
            softmax_scale=head_size**-0.5,
            causal=True,
            window_size=(-1, -1),
            block_table=torch.zeros(
                (num_seqs, kv_len // block_size), dtype=torch.int32, device=device
            ),
            softcap=0,
            q_descale=None,
            k_descale=k_descale,
            v_descale=v_descale,
            shuffled_kv_cache=shuffled_kv_cache,
        )


# ---------------------------------------------------------------------------
# 2D kernel (prefill): per-token descale on the Triton 2D path.
#
# MiniMax-M3's DENSE fp8 layers hit this: bf16 q + fp8 KV fails
# is_2d_gluon_available()'s q_dtype == kv_cache_dtype check, and a prefill is
# not ALL_DECODE, so use_2d_kernel() picks the Triton 2D kernel. Before the
# per-token path existed there, the caller silently passed a per-tensor scalar
# and K/V came out ~fp8_max/amax (~400x) too large.
# ---------------------------------------------------------------------------
# A PyTorch reference is used rather than a second unified_attention call: a
# bf16-q + bf16-KV call satisfies is_2d_gluon_available(), so it would run the
# GLUON 2D kernel while the fp8 case runs the TRITON 2D kernel, and the test
# would be comparing two different kernels instead of isolating the descale.
@pytest.mark.parametrize("block_size", [32, 128])
@pytest.mark.parametrize("num_heads", [(64, 4), (16, 1)])
@pytest.mark.parametrize("q_len", [64, 256])
def test_unified_attn_2d_per_token_kv_descale(block_size, num_heads, q_len):
    torch.manual_seed(0)
    hq, hkv = num_heads
    d = 128
    num_blocks_per_seq = 8
    batch = 2
    nb = num_blocks_per_seq * batch
    seq = num_blocks_per_seq * block_size
    if q_len > seq:
        pytest.skip("query longer than the cached sequence")

    k_log = torch.randn(nb, block_size, hkv, d, dtype=torch.bfloat16, device="cuda")
    v_log = torch.randn_like(k_log)

    fp8_max = torch.finfo(e4m3_dtype).max
    k_s = k_log.float().abs().amax(-1, keepdim=True).clamp(min=1e-12) / fp8_max
    v_s = v_log.float().abs().amax(-1, keepdim=True).clamp(min=1e-12) / fp8_max
    k_q = (k_log.float() / k_s).clamp(-fp8_max, fp8_max).to(e4m3_dtype)
    v_q = (v_log.float() / v_s).clamp(-fp8_max, fp8_max).to(e4m3_dtype)
    # The exact values the kernel should reconstruct from cache + scales.
    k_deq = (k_q.float() * k_s).to(torch.bfloat16)
    v_deq = (v_q.float() * v_s).to(torch.bfloat16)
    k_scale = k_s.squeeze(-1).permute(0, 2, 1).contiguous()
    v_scale = v_s.squeeze(-1).permute(0, 2, 1).contiguous()

    x = 16 // k_q.element_size()
    k_c8 = k_q.view(nb, block_size, hkv, d // x, x).permute(0, 2, 3, 1, 4).contiguous()
    v_c8 = v_q.view(nb, block_size // x, x, hkv, d).permute(0, 3, 1, 4, 2).contiguous()

    q = torch.randn(batch * q_len, hq, d, dtype=torch.bfloat16, device="cuda")
    cu_q = torch.arange(batch + 1, dtype=torch.int32, device="cuda") * q_len
    seqused = torch.full((batch,), seq, dtype=torch.int32, device="cuda")
    bt = torch.arange(nb, dtype=torch.int32, device="cuda").view(
        batch, num_blocks_per_seq
    )

    out = torch.empty_like(q)
    unified_attention(
        q,
        k_c8,
        v_c8,
        out,
        cu_seqlens_q=cu_q,
        max_seqlen_q=q_len,
        seqused_k=seqused,
        max_seqlen_k=seq,
        softmax_scale=d**-0.5,
        causal=True,
        alibi_slopes=None,
        window_size=(-1, -1),
        block_table=bt,
        softcap=0,
        q_descale=None,
        k_descale=k_scale,
        v_descale=v_scale,
        sinks=None,
        shuffled_kv_cache=True,
    )
    assert torch.isfinite(out).all()

    # ---- PyTorch reference over the dequantized cache ----
    ctx = seq - q_len
    g = hq // hkv
    ref = torch.empty_like(out)
    for b in range(batch):
        blocks = bt[b].tolist()
        kb = torch.cat([k_deq[i] for i in blocks], dim=0).float()  # [seq, hkv, d]
        vb = torch.cat([v_deq[i] for i in blocks], dim=0).float()
        qb = q[b * q_len : (b + 1) * q_len].float()  # [q_len, hq, d]
        # key j is visible to query i iff j <= ctx + i
        pos = torch.arange(q_len, device="cuda")[:, None] + ctx
        vis = torch.arange(seq, device="cuda")[None, :] <= pos
        for h in range(hq):
            kh, vh = kb[:, h // g], vb[:, h // g]
            s = (qb[:, h] @ kh.T) * (d**-0.5)
            s = s.masked_fill(~vis, float("-inf"))
            ref[b * q_len : (b + 1) * q_len, h] = (torch.softmax(s, dim=-1) @ vh).to(
                ref.dtype
            )

    rel = (out.float() - ref.float()).abs().max() / ref.float().abs().max()
    # A per-tensor descale here would be ~fp8_max/amax off, i.e. rel ~ O(100).
    assert rel < 5e-2, f"per-token descale mismatch on the 2D kernel: rel={rel}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
