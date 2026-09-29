# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Segmented-layout MLA cache writes, checked byte for byte against torch.

Both ops write the flat block layout [page x kv_lora (nope)][page x pe]:
concat_and_cache_mla_seg for K only, fused_qk_rope_concat_and_cache_mla_seg
for K plus the roped, quantized Q. Shapes are picked to reach every dispatch
the host chooses between -- one or several heads (or tokens) per block, the
gfx1250 tensor-engine path, token counts that leave a partial last block --
and every case carries one padded token (slot -1) that must write nothing.
"""

import pytest
import torch

from aiter import dtypes
from aiter.ops.cache import concat_and_cache_mla_seg, fused_qk_rope_concat_and_cache_mla_seg

KV_LORA, PE_DIM, PAGE = 512, 64, 64
MAX_POS = 65536
Q_OUT_DIM = 768  # 576 written, the tail must stay untouched


def _slots(t, num_blocks, scatter, gen):
    if scatter:
        slots = torch.randperm(num_blocks * PAGE, generator=gen)[:t].to(torch.int64)
    else:
        slots = torch.arange(t, dtype=torch.int64)
    if t >= 4:
        slots[3] = -1
    return slots.cuda()


def _quant(x, scale):
    fmax = torch.finfo(dtypes.fp8).max
    return (x.float() / scale).clamp(-fmax, fmax).to(dtypes.fp8)


def _rope(pe, positions, cos, sin, is_neox):
    x = pe.float()
    c = cos.index_select(0, positions).float()
    s = sin.index_select(0, positions).float()
    while c.dim() < x.dim():
        c, s = c.unsqueeze(-2), s.unsqueeze(-2)
    half = PE_DIM // 2
    if is_neox:
        a, b = x[..., :half], x[..., half:]
        return torch.cat([a * c - b * s, b * c + a * s], dim=-1)
    pair = x.view(*x.shape[:-1], half, 2)
    a, b = pair[..., 0], pair[..., 1]
    return torch.stack([a * c - b * s, b * c + a * s], dim=-1).view(*x.shape)


def _expected_cache(kv_cache, slots, nope, pe):
    """Scatter nope/pe (already in cache dtype) into a copy of kv_cache."""
    ref = kv_cache.clone()
    valid = slots >= 0
    blk, off = slots[valid] // PAGE, slots[valid] % PAGE
    nope_idx = off[:, None] * KV_LORA + torch.arange(KV_LORA, device="cuda")[None, :]
    pe_idx = PAGE * KV_LORA + off[:, None] * PE_DIM + torch.arange(PE_DIM, device="cuda")[None, :]
    ref[blk[:, None], nope_idx] = nope[valid]
    ref[blk[:, None], pe_idx] = pe[valid]
    return ref


def _bytes_differ(a, b):
    return int((a.view(torch.uint8) != b.view(torch.uint8)).sum().item())


@pytest.mark.parametrize("kv_dtype", ["fp8", "auto"])
@pytest.mark.parametrize("scatter", [True, False])
# 1 and 192 on the one-token-per-block path; 12289 / 16383 on four per block;
# 24577 / 65531 on eight -- all but 1 leave a partial last block.
@pytest.mark.parametrize("t", [1, 192, 12289, 16383, 24577, 65531])
def test_concat_and_cache_mla_seg(kv_dtype, scatter, t):
    gen = torch.Generator().manual_seed(t)
    kv_c = torch.randn((t, KV_LORA), generator=gen).to(dtypes.bf16).cuda()
    k_pe = torch.randn((t, PE_DIM), generator=gen).to(dtypes.bf16).cuda()
    num_blocks = max(4096, (t + PAGE - 1) // PAGE)
    cdt = dtypes.fp8 if kv_dtype == "fp8" else dtypes.bf16
    kv_cache = torch.zeros((num_blocks, PAGE * (KV_LORA + PE_DIM)), dtype=cdt, device="cuda")
    slots = _slots(t, num_blocks, scatter, gen)
    scale = torch.tensor([0.25], dtype=torch.float32, device="cuda")

    if kv_dtype == "fp8":
        nope, pe = _quant(kv_c, scale.item()), _quant(k_pe, scale.item())
    else:
        nope, pe = kv_c, k_pe
    ref = _expected_cache(kv_cache, slots, nope, pe)

    concat_and_cache_mla_seg(kv_c, k_pe, kv_cache, slots, kv_dtype, scale)
    torch.cuda.synchronize()
    assert _bytes_differ(kv_cache, ref) == 0


@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize("scatter", [True, False])
# (H, T): (1, *) and small T*H stay one head per block; (6, 1536) two heads,
# (12, 1536) four, (32, 1536) / (128, 1536) / (128, 192) eight.
@pytest.mark.parametrize(
    "h,t", [(1, 7), (1, 1536), (32, 192), (6, 1536), (12, 1536), (32, 1536), (128, 192), (128, 1536)]
)
def test_fused_qk_rope_concat_and_cache_mla_seg(is_neox, scatter, h, t):
    gen = torch.Generator().manual_seed(h * 100003 + t)
    q_nope = torch.randn((t, h, KV_LORA), generator=gen).to(dtypes.bf16).cuda()
    q_pe = torch.randn((t, h, PE_DIM), generator=gen).to(dtypes.bf16).cuda()
    kv_c = torch.randn((t, KV_LORA), generator=gen).to(dtypes.bf16).cuda()
    k_pe = torch.randn((t, PE_DIM), generator=gen).to(dtypes.bf16).cuda()
    num_blocks = 4096
    kv_cache = torch.zeros((num_blocks, PAGE * (KV_LORA + PE_DIM)), dtype=dtypes.fp8, device="cuda")
    q_out = torch.zeros((t, h, Q_OUT_DIM), dtype=dtypes.fp8, device="cuda")
    slots = _slots(t, num_blocks, scatter, gen)
    positions = torch.randint(0, MAX_POS, (t,), generator=gen).cuda()
    theta = torch.randn((MAX_POS, PE_DIM // 2), generator=gen).cuda()
    cos, sin = torch.cos(theta).to(dtypes.bf16), torch.sin(theta).to(dtypes.bf16)
    q_scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")
    k_scale = torch.tensor([0.25], dtype=torch.float32, device="cuda")

    valid = (slots >= 0)[:, None, None]
    ref_q = torch.zeros_like(q_out)
    ref_q[..., :KV_LORA] = torch.where(valid, _quant(q_nope, q_scale.item()).float(), 0).to(dtypes.fp8)
    ref_q[..., KV_LORA : KV_LORA + PE_DIM] = torch.where(
        valid, _quant(_rope(q_pe, positions, cos, sin, is_neox), q_scale.item()).float(), 0
    ).to(dtypes.fp8)
    ref_cache = _expected_cache(
        kv_cache,
        slots,
        _quant(kv_c, k_scale.item()),
        _quant(_rope(k_pe, positions, cos, sin, is_neox), k_scale.item()),
    )

    fused_qk_rope_concat_and_cache_mla_seg(
        q_nope, q_pe, kv_c, k_pe, kv_cache, q_out, slots,
        k_scale, q_scale, positions, cos, sin, is_neox, True,
    )
    torch.cuda.synchronize()
    assert _bytes_differ(q_out, ref_q) == 0
    assert _bytes_differ(kv_cache, ref_cache) == 0
