# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# The SWA + top-k two-loop on per-tensor fp8 (fp8_scalar) block caches, in the
# DeepSeek-V4 geometry: rope inside the 512-wide row, V the whole row.

import pytest
import torch

from aiter.ops.triton.attention.sparse_mla import sparse_mla_fwd
from aiter.ops.triton.utils._triton import arch_info

D = 512
FP8 = torch.float8_e4m3fn


def _skip_unless_gfx950():
    if not torch.cuda.is_available() or arch_info.get_arch() != "gfx950":
        pytest.skip("gfx950 only")


def paged_fp8_cache(nb, block, scale, pad, device, gen):
    """[nb, block, D] fp8 view of a pool whose page pitch is block * D + pad."""
    pitch = block * D + pad
    raw = torch.zeros(nb * pitch, dtype=torch.uint8, device=device)
    vals = torch.randn(nb, block, D, generator=gen, device=device) * 2.0
    codes = (vals / scale).to(FP8)
    view = raw.as_strided((nb, block, D), (pitch, D, 1)).view(FP8)
    view.copy_(codes)
    return view, codes.float() * scale


def ragged(C, rows, lens, device, gen, invalid_frac):
    indptr = torch.zeros(C + 1, dtype=torch.int32, device=device)
    indptr[1:] = torch.cumsum(lens, 0)
    idx = torch.randint(0, rows, (int(indptr[-1]),), generator=gen, device=device)
    if invalid_frac:
        drop = torch.rand(idx.shape, generator=gen, device=device) < invalid_frac
        idx[drop] = -1
    return idx.to(torch.int32), indptr


def reference(q, segs, sm_scale, sink):
    C, H, _ = q.shape
    out = torch.empty(C, H, D, dtype=torch.float32, device=q.device)
    qf = q.float()
    for c in range(C):
        rows = []
        for truth, idx, ptr in segs:
            sel = idx[ptr[c] : ptr[c + 1]].long()
            sel = sel[sel >= 0]
            rows.append(truth.reshape(-1, D)[sel])
        kv = torch.cat(rows)
        if kv.shape[0] == 0:
            out[c] = 0.0  # only the sink: no value rows to mix
            continue
        logits = qf[c] @ kv.T * sm_scale  # [H, n]
        m = torch.maximum(logits.max(-1).values, sink)
        p = torch.exp(logits - m[:, None])
        denom = p.sum(-1) + torch.exp(sink - m)
        out[c] = (p @ kv) / denom[:, None]
    return out


# C 96 / 192 / 300 at 32 / 64 heads take 32- and 64-head programs (_staged_head_block),
# in the per-tensor launch and in the bf16 two-loop it is checked against.
@pytest.mark.parametrize("C", [1, 7, 64, 96, 192, 300])
@pytest.mark.parametrize("H", [16, 32, 64])
@pytest.mark.parametrize("fp8_q", [False, True])
@pytest.mark.parametrize("pad", [0, 256])
@pytest.mark.parametrize("dots", ["bf16", "fp8"])
def test_fp8_scalar_two_loop(C, H, fp8_q, pad, dots):
    _skip_unless_gfx950()
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(C * 1000 + H + pad)
    swa_scale = torch.tensor([0.75], device=dev)
    cmp_scale = torch.tensor([1.25], device=dev)
    swa, swa_truth = paged_fp8_cache(48, 32, swa_scale, pad, dev, gen)
    cmp, cmp_truth = paged_fp8_cache(40, 32, cmp_scale, pad, dev, gen)

    swa_lens = torch.randint(0, 129, (C,), generator=gen, device=dev)
    cmp_lens = torch.randint(0, 513, (C,), generator=gen, device=dev)
    swa_idx, swa_ptr = ragged(C, 48 * 32, swa_lens, dev, gen, 0.05)
    cmp_idx, cmp_ptr = ragged(C, 40 * 32, cmp_lens, dev, gen, 0.05)

    q = (torch.randn(C, H, D, generator=gen, device=dev) * 0.5).to(torch.bfloat16)
    q_scale = None
    if fp8_q:
        q_scale = torch.tensor([1.0], device=dev)
        q = q.to(FP8)
    sink = torch.randn(H, generator=gen, device=dev)
    sm_scale = D**-0.5

    out = torch.zeros(C, H, D, dtype=torch.bfloat16, device=dev)
    sparse_mla_fwd(
        q,
        swa,
        swa_ptr,
        swa_idx,
        sm_scale,
        kv_scale=swa_scale,
        kv_lora_rank=D,
        qk_rope_head_dim=0,
        has_invalid=True,
        q_scale=q_scale,
        out=out,
        attn_sink=sink,
        extra_kv=cmp,
        extra_indptr=cmp_ptr,
        extra_indices=cmp_idx,
        extra_kv_scale=cmp_scale,
        dot_precision=dots,
    )
    q_ref = q.float() * (q_scale if fp8_q else 1.0)
    ref = reference(
        q_ref,
        [(swa_truth, swa_idx, swa_ptr), (cmp_truth, cmp_idx, cmp_ptr)],
        sm_scale,
        sink,
    )
    if dots == "fp8":
        # P is rounded to e4m3 (and a bf16 q too), ~2% relative L2 against the
        # fp32 reference, the level of the one-segment fp8 path. A V-scale
        # mistake shows up as a slope away from 1.
        o = out.float()
        rel = ((o - ref).norm() / ref.norm()).item()
        slope = ((o * ref).sum() / (ref * ref).sum()).item()
        assert rel < 0.035 and abs(slope - 1) < 0.005, f"relL2 {rel} slope {slope}"
        return
    err = (out.float() - ref).abs().max().item()
    assert err < 2e-2, f"max abs err {err}"

    # Same values through the production bf16 two-loop.
    out_bf16 = torch.zeros_like(out)
    sparse_mla_fwd(
        q.to(torch.bfloat16) if not fp8_q else q_ref.to(torch.bfloat16),
        swa_truth.to(torch.bfloat16),
        swa_ptr,
        swa_idx,
        sm_scale,
        kv_lora_rank=D,
        qk_rope_head_dim=0,
        has_invalid=True,
        out=out_bf16,
        attn_sink=sink,
        extra_kv=cmp_truth.to(torch.bfloat16),
        extra_indptr=cmp_ptr,
        extra_indices=cmp_idx,
    )
    # As close to the reference (the two are rounded to bf16 separately, so they
    # can sit 2e-2 apart on either side of it).
    err_bf16 = (out_bf16.float() - ref).abs().max().item()
    assert err_bf16 < 2e-2, f"bf16 two-loop max abs err {err_bf16}"


def test_swa_only_padded_pitch():
    """No extra segment (SWA-only layers), page pitch padded."""
    _skip_unless_gfx950()
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(7)
    scale = torch.tensor([1.0], device=dev)
    swa, truth = paged_fp8_cache(16, 32, scale, 512, dev, gen)
    C, H = 33, 16
    idx, ptr = ragged(
        C, 16 * 32, torch.randint(1, 129, (C,), generator=gen, device=dev), dev, gen, 0
    )
    q = (torch.randn(C, H, D, generator=gen, device=dev) * 0.5).to(torch.bfloat16)
    sink = torch.randn(H, generator=gen, device=dev)
    out, _ = sparse_mla_fwd(
        q,
        swa,
        ptr,
        idx,
        D**-0.5,
        kv_scale=scale,
        kv_lora_rank=D,
        qk_rope_head_dim=0,
        has_invalid=True,
        attn_sink=sink,
    )
    ref = reference(q.float(), [(truth, idx, ptr)], D**-0.5, sink)
    err = (out.float() - ref).abs().max().item()
    assert err < 2e-2, f"max abs err {err}"
