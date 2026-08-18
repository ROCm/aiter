# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness tests for minimax_m3_qknorm_rope_cache_shuffle_insert.

The reference reimplements Gemma QK-norm, partial NeoX RoPE and the page-16
SHUFFLE cache index math in torch, so a layout mistake shows up as a gross
mismatch rather than a rounding difference.
"""

import pytest
import torch

from aiter import minimax_m3_qknorm_rope_cache_shuffle_insert
from aiter.utility.dtypes import get_dtype_fp8

HEAD_DIM = 128
PAGE_SIZE = 16


def gemma_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """x * rsqrt(mean(x^2) + eps) * (1 + w), computed in fp32."""
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    return xf * torch.rsqrt(var + eps) * (1.0 + w.float())


def partial_neox_rope(
    x: torch.Tensor, cos_sin: torch.Tensor, rotary_dim: int
) -> torch.Tensor:
    """Rotate the leading rotary_dim dims; cos_sin is [cos(half) | sin(half)]."""
    half = rotary_dim // 2
    cos, sin = cos_sin[..., :half].float(), cos_sin[..., half:].float()
    out = x.clone()
    x1, x2 = x[..., :half], x[..., half:rotary_dim]
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:rotary_dim] = x2 * cos + x1 * sin
    return out


def make_inputs(
    num_tokens,
    nq,
    nkv,
    niq,
    rotary_dim,
    dtype,
    cache_dtype,
    idx_cache_dtype,
    num_pages,
    idx_num_pages,
    seed=0,
):
    torch.manual_seed(seed)
    dev = "cuda"
    row = (nq + 2 * nkv + niq + 1) * HEAD_DIM
    qkv = torch.randn(num_tokens, row, device=dev, dtype=dtype)

    weights = {
        name: torch.randn(HEAD_DIM, device=dev, dtype=dtype) * 0.1
        for name in ("q", "k", "iq", "ik")
    }
    max_pos = 4096
    cos_sin = torch.randn(max_pos, rotary_dim, device=dev, dtype=dtype)
    positions = torch.randint(0, max_pos, (num_tokens,), device=dev, dtype=torch.int64)

    x = 16 // torch.empty((), dtype=cache_dtype).element_size()
    k_cache = torch.zeros(
        num_pages, nkv, HEAD_DIM // x, PAGE_SIZE, x, device=dev, dtype=cache_dtype
    )
    v_cache = torch.zeros(
        num_pages, nkv, PAGE_SIZE // x, HEAD_DIM, x, device=dev, dtype=cache_dtype
    )
    index_cache = torch.zeros(
        idx_num_pages, PAGE_SIZE, HEAD_DIM, device=dev, dtype=idx_cache_dtype
    )

    # Distinct slots so no two tokens collide, then punch a few padded tokens.
    slot_mapping = torch.randperm(num_pages * PAGE_SIZE, device=dev)[:num_tokens].to(
        torch.int64
    )
    index_slot_mapping = torch.randperm(idx_num_pages * PAGE_SIZE, device=dev)[
        :num_tokens
    ].to(torch.int64)
    if num_tokens >= 4:
        slot_mapping[1] = -1
        index_slot_mapping[2] = -1

    q_out = torch.zeros(num_tokens, nq * HEAD_DIM, device=dev, dtype=dtype)
    index_q_out = torch.zeros(
        num_tokens, niq * HEAD_DIM, device=dev, dtype=idx_cache_dtype
    )
    return dict(
        qkv=qkv,
        weights=weights,
        cos_sin=cos_sin,
        positions=positions,
        k_cache=k_cache,
        v_cache=v_cache,
        index_cache=index_cache,
        slot_mapping=slot_mapping,
        index_slot_mapping=index_slot_mapping,
        q_out=q_out,
        index_q_out=index_q_out,
        x=x,
    )


def reference(t, nq, nkv, niq, rotary_dim, eps, process_index, k_scale, v_scale):
    """Returns (q_out, index_q_out, k_cache, v_cache, index_cache) references."""
    qkv, w = t["qkv"], t["weights"]
    num_tokens = qkv.shape[0]
    dtype = qkv.dtype
    cache_dtype = t["k_cache"].dtype
    idx_dtype = t["index_cache"].dtype
    x = t["x"]

    heads = qkv.view(num_tokens, -1, HEAD_DIM)
    cos_sin = t["cos_sin"][t["positions"]].unsqueeze(1)  # [N, 1, rotary_dim]

    def norm_rope(sub, weight):
        return partial_neox_rope(gemma_rmsnorm(sub, weight, eps), cos_sin, rotary_dim)

    q = norm_rope(heads[:, :nq], w["q"]).to(dtype)
    k = norm_rope(heads[:, nq : nq + nkv], w["k"]).to(dtype)
    v = heads[:, nq + nkv : nq + 2 * nkv]
    iq_begin = nq + 2 * nkv
    # index_q is quantized straight from fp32; index_k goes via the model dtype
    # (see the kernel comment: it mirrors the unfused bf16 -> fp8 insert).
    index_q_f32 = norm_rope(heads[:, iq_begin : iq_begin + niq], w["iq"])
    index_k = norm_rope(heads[:, iq_begin + niq : iq_begin + niq + 1], w["ik"]).to(dtype)

    q_ref = q.reshape(num_tokens, nq * HEAD_DIM)
    iq_src = index_q_f32 if idx_dtype != dtype else index_q_f32.to(dtype)
    iq_ref = iq_src.reshape(num_tokens, niq * HEAD_DIM).clamp(-448.0, 448.0).to(idx_dtype)

    k_ref = torch.zeros_like(t["k_cache"])
    v_ref = torch.zeros_like(t["v_cache"])
    idx_ref = torch.zeros_like(t["index_cache"])

    def quant(vals, scale):
        if cache_dtype == dtype:
            return vals
        f = vals.float()
        if scale is not None:
            f = f / scale.item()
        return f.to(cache_dtype)

    for tok in range(num_tokens):
        slot = int(t["slot_mapping"][tok])
        if slot >= 0:
            page, off = slot // PAGE_SIZE, slot % PAGE_SIZE
            k_ref[page, :, :, off, :] = quant(k[tok], k_scale).view(
                nkv, HEAD_DIM // x, x
            )
            v_ref[page, :, off // x, :, off % x] = quant(v[tok], v_scale)
        if not process_index:
            continue
        islot = int(t["index_slot_mapping"][tok])
        if islot >= 0:
            ipage, ioff = islot // PAGE_SIZE, islot % PAGE_SIZE
            idx_ref[ipage, ioff] = index_k[tok, 0].to(idx_dtype)
    return q_ref, iq_ref, k_ref, v_ref, idx_ref


def compare(name, got, ref):
    """bf16/fp8 tolerance; a layout bug misses by orders of magnitude."""
    g, r = got.float(), ref.float()
    torch.testing.assert_close(g, r, rtol=2e-2, atol=2e-2, msg=lambda m: f"{name}: {m}")
    exact = (g == r).float().mean().item()
    assert exact > 0.9, f"{name}: only {exact:.1%} of elements bit-exact"


def run_op(t, nq, nkv, niq, rotary_dim, eps, **kwargs):
    minimax_m3_qknorm_rope_cache_shuffle_insert(
        t["qkv"],
        t["weights"]["q"],
        t["weights"]["k"],
        t["cos_sin"],
        t["positions"],
        nq,
        nkv,
        niq,
        rotary_dim,
        eps,
        t["slot_mapping"],
        t["k_cache"],
        t["v_cache"],
        t["q_out"],
        **kwargs,
    )
    torch.cuda.synchronize()


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("fp8_index", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 7, 64])
def test_fused_insert(quantized, fp8_index, num_tokens):
    nq, nkv, niq, rotary_dim, eps = 16, 1, 1, 64, 1e-6
    dtype = torch.bfloat16
    fp8 = get_dtype_fp8()
    t = make_inputs(
        num_tokens,
        nq,
        nkv,
        niq,
        rotary_dim,
        dtype,
        fp8 if quantized else dtype,
        fp8 if fp8_index else dtype,
        num_pages=8,
        idx_num_pages=8,
    )
    k_scale = torch.tensor([0.7], device="cuda") if quantized else None
    v_scale = torch.tensor([1.3], device="cuda") if quantized else None

    run_op(
        t,
        nq,
        nkv,
        niq,
        rotary_dim,
        eps,
        index_q_norm_weight=t["weights"]["iq"],
        index_k_norm_weight=t["weights"]["ik"],
        index_slot_mapping=t["index_slot_mapping"],
        index_cache=t["index_cache"],
        index_q_out=t["index_q_out"],
        kv_cache_dtype="fp8" if quantized else "auto",
        k_scale=k_scale,
        v_scale=v_scale,
    )

    q_ref, iq_ref, k_ref, v_ref, idx_ref = reference(
        t, nq, nkv, niq, rotary_dim, eps, True, k_scale, v_scale
    )
    compare("q_out", t["q_out"], q_ref)
    compare("index_q_out", t["index_q_out"], iq_ref)
    compare("k_cache", t["k_cache"], k_ref)
    compare("v_cache", t["v_cache"], v_ref)
    compare("index_cache", t["index_cache"], idx_ref)


def test_skip_index_branch_leaves_index_outputs_untouched():
    nq, nkv, niq, rotary_dim, eps = 16, 1, 1, 64, 1e-6
    dtype = torch.bfloat16
    t = make_inputs(
        32, nq, nkv, niq, rotary_dim, dtype, dtype, dtype, num_pages=8, idx_num_pages=8
    )
    run_op(t, nq, nkv, niq, rotary_dim, eps, skip_index_branch=True)

    q_ref, _, k_ref, v_ref, _ = reference(
        t, nq, nkv, niq, rotary_dim, eps, False, None, None
    )
    compare("q_out", t["q_out"], q_ref)
    compare("k_cache", t["k_cache"], k_ref)
    compare("v_cache", t["v_cache"], v_ref)
    assert t["index_cache"].float().abs().sum() == 0
    assert t["index_q_out"].float().abs().sum() == 0


def test_multi_kv_head_and_full_rotary():
    nq, nkv, niq, rotary_dim, eps = 32, 4, 2, HEAD_DIM, 1e-6
    dtype = torch.bfloat16
    fp8 = get_dtype_fp8()
    t = make_inputs(
        48, nq, nkv, niq, rotary_dim, dtype, fp8, fp8, num_pages=16, idx_num_pages=8
    )
    k_scale = torch.tensor([1.0], device="cuda")
    v_scale = torch.tensor([1.0], device="cuda")
    run_op(
        t,
        nq,
        nkv,
        niq,
        rotary_dim,
        eps,
        index_q_norm_weight=t["weights"]["iq"],
        index_k_norm_weight=t["weights"]["ik"],
        index_slot_mapping=t["index_slot_mapping"],
        index_cache=t["index_cache"],
        index_q_out=t["index_q_out"],
        kv_cache_dtype="fp8",
        k_scale=k_scale,
        v_scale=v_scale,
    )
    q_ref, iq_ref, k_ref, v_ref, idx_ref = reference(
        t, nq, nkv, niq, rotary_dim, eps, True, k_scale, v_scale
    )
    compare("q_out", t["q_out"], q_ref)
    compare("index_q_out", t["index_q_out"], iq_ref)
    compare("k_cache", t["k_cache"], k_ref)
    compare("v_cache", t["v_cache"], v_ref)
    compare("index_cache", t["index_cache"], idx_ref)
