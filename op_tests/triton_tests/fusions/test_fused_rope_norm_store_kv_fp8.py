# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit test for ``rope_norm_store_kv_fp8``.

The test was audited for hidden parameter couplings — any "X is always set
together with Y" that could mask a kernel branch. Coverage axes:

- ``is_neox ∈ {True, False}``  — NeoX *and* GPT-J rotation paths.
- ``qk_norm_policy ∈ {0, 1, 2}`` × ``norm_weights ∈ {both, neither, q_only,
  k_only}``  — decouples weight presence from policy.
- Prefill scenarios chosen so each request's new tokens span **multiple**
  physical blocks (exercises ``block_logical = pos // BLOCK_SIZE`` changing
  mid-sequence).
- Block table is randomly permuted (not monotonic) so any wrong indirection
  produces a visibly wrong cache write.
- Mixed ``n_new`` across requests (chunked prefill).

Plus dedicated scenario tests for: negative norm weights, ``max_seqlens >
max(new_tokens_per_req)``, and caller-preallocated ``out_q``.
"""

from typing import Optional, Tuple

import pytest
import torch

from aiter.ops.triton.fusions.fused_rope_norm_store_kv_fp8 import (
    rope_norm_store_kv_fp8,
)
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype


# ─────────────────────────── PyTorch reference ─────────────────────────────


def _ref_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x.float() / rms * w.float()).to(x.dtype)


def _ref_rope_neox(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    cos = cos[..., : d // 2]
    sin = sin[..., : d // 2]
    o1 = x1.float() * cos - x2.float() * sin
    o2 = x2.float() * cos + x1.float() * sin
    return torch.cat([o1, o2], dim=-1).to(x.dtype)


def _ref_rope_gptj(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """GPT-J interleaved rotation. cos/sin still sized ``[..., D/2]``."""
    d = x.shape[-1]
    pair = x.float().reshape(*x.shape[:-1], d // 2, 2)
    x1 = pair[..., 0]
    x2 = pair[..., 1]
    cos = cos[..., : d // 2]
    sin = sin[..., : d // 2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.stack([o1, o2], dim=-1).reshape(x.shape).to(x.dtype)


def _ref_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, is_neox: bool
) -> torch.Tensor:
    return _ref_rope_neox(x, cos, sin) if is_neox else _ref_rope_gptj(x, cos, sin)


def _ref(
    qkv: torch.Tensor,
    cos_sin: torch.Tensor,
    num_seqlen_per_req: torch.Tensor,
    q_index: torch.Tensor,
    kvcache_indices: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    is_prefill: bool,
    max_seqlens: int,
    q_norm_weight: Optional[torch.Tensor],
    k_norm_weight: Optional[torch.Tensor],
    qk_norm_policy: int,
    rms_eps: float,
    fp8_dtype: torch.dtype,
    is_neox: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fp8_max = float(torch.finfo(fp8_dtype).max)
    num_rows = qkv.shape[0]
    block_size = key_cache.shape[1]
    num_kv_heads = key_cache.shape[2]
    head_dim = key_cache.shape[3]
    num_req = num_seqlen_per_req.shape[0]
    hidden = qkv.shape[1]
    num_q_heads = (hidden - 2 * num_kv_heads * head_dim) // head_dim

    q = qkv[:, : num_q_heads * head_dim].reshape(num_rows, num_q_heads, head_dim)
    k = qkv[
        :, num_q_heads * head_dim : (num_q_heads + num_kv_heads) * head_dim
    ].reshape(num_rows, num_kv_heads, head_dim)
    v = qkv[:, (num_q_heads + num_kv_heads) * head_dim :].reshape(
        num_rows, num_kv_heads, head_dim
    )

    out_q_bf = torch.empty_like(q)
    out_k_bf = torch.empty_like(k)

    for r in range(num_req):
        s, e = int(q_index[r].item()), int(q_index[r + 1].item())
        n_new = e - s
        total = int(num_seqlen_per_req[r].item())
        positions = torch.arange(total - n_new, total, device=qkv.device)
        cos = cos_sin[positions, : head_dim // 2]
        sin = cos_sin[positions, head_dim // 2 :]
        cos_q = cos.unsqueeze(1).expand(-1, num_q_heads, -1)
        sin_q = sin.unsqueeze(1).expand(-1, num_q_heads, -1)
        cos_k = cos.unsqueeze(1).expand(-1, num_kv_heads, -1)
        sin_k = sin.unsqueeze(1).expand(-1, num_kv_heads, -1)
        q_r = q[s:e]
        k_r = k[s:e]

        if qk_norm_policy == 1:
            q_o = _ref_rope(q_r, cos_q, sin_q, is_neox)
            k_o = _ref_rope(k_r, cos_k, sin_k, is_neox)
            if q_norm_weight is not None:
                q_o = _ref_rmsnorm(q_o, q_norm_weight, rms_eps)
            if k_norm_weight is not None:
                k_o = _ref_rmsnorm(k_o, k_norm_weight, rms_eps)
        elif qk_norm_policy == 2:
            qn = (
                _ref_rmsnorm(q_r, q_norm_weight, rms_eps)
                if q_norm_weight is not None
                else q_r
            )
            kn = (
                _ref_rmsnorm(k_r, k_norm_weight, rms_eps)
                if k_norm_weight is not None
                else k_r
            )
            q_o = _ref_rope(qn, cos_q, sin_q, is_neox)
            k_o = _ref_rope(kn, cos_k, sin_k, is_neox)
        else:
            q_o = _ref_rope(q_r, cos_q, sin_q, is_neox)
            k_o = _ref_rope(k_r, cos_k, sin_k, is_neox)

        out_q_bf[s:e] = q_o
        out_k_bf[s:e] = k_o

    def _dyn_q(x):
        amax = x.float().abs().amax(dim=-1).clamp(min=1e-12)
        scale = (amax / fp8_max).to(torch.float32)
        y = (x.float() / scale.unsqueeze(-1)).clamp(-fp8_max, fp8_max).to(fp8_dtype)
        return y, scale

    out_q_fp8, q_scale_flat = _dyn_q(out_q_bf)
    out_k_fp8, k_scale_flat = _dyn_q(out_k_bf)
    out_v_fp8 = (
        (v.float() / v_scale.view(1, -1, 1))
        .clamp(-fp8_max, fp8_max)
        .to(fp8_dtype)
    )

    # Scatter to paged cache and k-scale slab.
    for r in range(num_req):
        s, e = int(q_index[r].item()), int(q_index[r + 1].item())
        n_new = e - s
        total = int(num_seqlen_per_req[r].item())
        for t in range(n_new):
            kv_pos = total - n_new + t
            blk_log = kv_pos // block_size
            blk_off = kv_pos % block_size
            blk_phys = int(kvcache_indices[r, blk_log].item())
            key_cache[blk_phys, blk_off] = out_k_fp8[s + t]
            value_cache[blk_phys, blk_off] = out_v_fp8[s + t]
            sr = blk_off // 32
            sc = blk_off % 32
            k_scale[blk_phys, sr, :, sc] = k_scale_flat[s + t]

    if is_prefill:
        pad = ((max_seqlens + 127) // 128) * 128
        q_scale_out = torch.ones(
            num_req, num_q_heads, pad, dtype=torch.float32, device=qkv.device
        )
        for r in range(num_req):
            s, e = int(q_index[r].item()), int(q_index[r + 1].item())
            q_scale_out[r, :, : e - s] = q_scale_flat[s:e].t()
    else:
        q_scale_out = q_scale_flat

    split_k_flag = torch.zeros(
        num_req, num_kv_heads, dtype=torch.int32, device=qkv.device
    )
    return out_q_fp8, q_scale_out, split_k_flag


# ─────────────────────────── Test fixture helpers ──────────────────────────


def _build_inputs(
    num_req: int,
    new_tokens_per_req: list,
    seq_lens: list,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    fp8_dtype: torch.dtype,
    *,
    shuffle_block_table: bool = True,
    norm_weight_signed: bool = True,
    block_pool_factor: int = 2,  # >=1; pool of physical blocks larger than needed
    device: str = "cuda",
):
    """Generate consistent inputs for reference + kernel.

    By default we randomly permute the physical-block IDs (so a buggy
    indirection through the block table will produce a visible mismatch)
    and allow signed RMSNorm weights.
    """
    assert len(new_tokens_per_req) == num_req == len(seq_lens)
    num_rows = sum(new_tokens_per_req)
    max_seqlen = max(seq_lens)

    n_blk_per_req = [(s + block_size - 1) // block_size for s in seq_lens]
    max_blocks = max(n_blk_per_req)
    needed_blocks = sum(n_blk_per_req)
    total_blocks = max(block_pool_factor * needed_blocks, needed_blocks)

    hidden = num_q_heads * head_dim + 2 * num_kv_heads * head_dim
    qkv = torch.randn(num_rows, hidden, dtype=torch.bfloat16, device=device)

    cos_sin = torch.randn(max_seqlen, head_dim, dtype=torch.float32, device=device)

    num_seqlen_per_req = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    q_index = torch.zeros(num_req + 1, dtype=torch.int32, device=device)
    q_index[1:] = torch.tensor(
        new_tokens_per_req, dtype=torch.int32, device=device
    ).cumsum(0)

    if shuffle_block_table:
        pool = torch.randperm(total_blocks, device=device).to(torch.int32)
    else:
        pool = torch.arange(total_blocks, dtype=torch.int32, device=device)
    kvcache_indices = torch.zeros(
        num_req, max_blocks, dtype=torch.int32, device=device
    )
    cursor = 0
    for r in range(num_req):
        n_blk = n_blk_per_req[r]
        kvcache_indices[r, :n_blk] = pool[cursor : cursor + n_blk]
        cursor += n_blk

    key_cache = torch.zeros(
        total_blocks, block_size, num_kv_heads, head_dim,
        dtype=fp8_dtype, device=device,
    )
    value_cache = torch.zeros_like(key_cache)
    assert block_size % 32 == 0, "test uses SCALE_COLS=32; block_size must be a multiple of 32"
    k_scale = torch.zeros(
        total_blocks, block_size // 32, num_kv_heads, 32,
        dtype=torch.float32, device=device,
    )

    # v_scale: positive; nothing in the spec says it can be negative.
    v_scale = (torch.rand(num_kv_heads, dtype=torch.float32, device=device) + 0.5) * 0.05

    # norm weights: signed (default) — bugs that silently abs() the weight
    # would show up only when sign actually matters.
    if norm_weight_signed:
        q_norm_w = torch.randn(head_dim, dtype=torch.float32, device=device) * 0.7
        k_norm_w = torch.randn(head_dim, dtype=torch.float32, device=device) * 0.7
    else:
        q_norm_w = torch.randn(head_dim, dtype=torch.float32, device=device).abs() + 0.5
        k_norm_w = torch.randn(head_dim, dtype=torch.float32, device=device).abs() + 0.5

    return dict(
        qkv=qkv,
        cos_sin=cos_sin,
        num_seqlen_per_req=num_seqlen_per_req,
        q_index=q_index,
        kvcache_indices=kvcache_indices,
        key_cache=key_cache,
        value_cache=value_cache,
        k_scale=k_scale,
        v_scale=v_scale,
        q_norm_w=q_norm_w,
        k_norm_w=k_norm_w,
        num_rows=num_rows,
        max_seqlen=max_seqlen,
    )


def _fp8_close(a: torch.Tensor, b: torch.Tensor) -> bool:
    """fp8 e4m3 closeness, calibrated against fp32-reduction-order noise.

    Two complementary bounds catch real structural bugs without flagging
    benign reduction-order rounding:

    1. **Bulk check.** ≥99% of elements must be within 1 fp8 ULP
       (rel ≤ 0.14). A wrong index, wrong rotation, missing abs, or wrong
       block-table dereference would move many elements off by >1 ULP and
       trip this. Reduction-order noise from RMSNorm fp32 sums typically
       leaves <1% of elements in this band.

    2. **Outlier check.** No element may be off by more than 4 fp8 ULPs
       AT THE TENSOR'S MAXIMUM MAGNITUDE. Per-element relative error blows
       up to ~1 for small values where ref and kernel pick adjacent fp8
       representables on opposite sides of zero (a 0.05-vs.-0.5 sign flip
       near a rotation zero crossing is genuine fp32-noise behavior and is
       small in absolute terms). Bounding by the tensor amax (= tensor_max
       / 8 per ULP × 4 budget = tensor_max / 2) lets those through while
       still failing on a wildly wrong value.
    """
    af, bf = a.float(), b.float()
    denom = torch.maximum(af.abs(), bf.abs()).clamp(min=1.0)
    rel = (af - bf).abs() / denom
    bulk_ok = (rel <= 0.14).float().mean().item() >= 0.99

    tensor_amax = max(af.abs().max().item(), bf.abs().max().item(), 1e-3)
    max_abs_diff = (af - bf).abs().max().item()
    outlier_ok = max_abs_diff <= tensor_amax / 2.0  # 4 ULPs at amax

    return bool(bulk_ok and outlier_ok)


# ─────────────── Main parametrized matrix (close to full coverage) ────────


# Decode scenarios — mixed n_new across requests so the
# (token_to_req, within_req) maps see non-uniform stripes.
_DECODE_SCENARIO = dict(
    is_prefill=False,
    new_tokens_per_req=[1, 4, 16, 1],
    seq_lens=[64, 256, 512, 128],
)

# Prefill scenarios — each request's new tokens deliberately span >=2 physical
# blocks so the kernel's per-token block_logical change is exercised.
# block_size=64: r0 spans blocks 1,2,3 (pos 120..199); r1 spans 1,2 (pos 80..149).
# block_size=32: r0 spans blocks 0,1,2 (pos 30..79); r1 spans 0,1 (pos 15..49).
_PREFILL_SCENARIO_64 = dict(
    is_prefill=True,
    new_tokens_per_req=[80, 70],
    seq_lens=[200, 150],
)
_PREFILL_SCENARIO_32 = dict(
    is_prefill=True,
    new_tokens_per_req=[50, 35],
    seq_lens=[80, 50],
)


def _pick_prefill(block_size: int):
    return _PREFILL_SCENARIO_32 if block_size == 32 else _PREFILL_SCENARIO_64


@pytest.mark.parametrize("is_neox", [True, False], ids=["neox", "gptj"])
@pytest.mark.parametrize("qk_norm_policy", [0, 1, 2])
@pytest.mark.parametrize(
    "norm_weights",
    ["both", "neither", "q_only", "k_only"],
    ids=["w=both", "w=neither", "w=q_only", "w=k_only"],
)
@pytest.mark.parametrize(
    "scenario_kind", ["prefill", "decode"], ids=["prefill", "decode"]
)
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,head_dim,block_size",
    [(8, 1, 128, 64), (16, 4, 128, 64), (4, 4, 64, 32)],
)
def test_rope_norm_store_kv_fp8(
    is_neox: bool,
    qk_norm_policy: int,
    norm_weights: str,
    scenario_kind: str,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(0)

    fp8_dtype = get_fp8_e4m3_dtype()

    scenario = (
        _pick_prefill(block_size) if scenario_kind == "prefill" else _DECODE_SCENARIO
    )
    new_tokens_per_req = scenario["new_tokens_per_req"]
    seq_lens = scenario["seq_lens"]
    is_prefill = scenario["is_prefill"]
    num_req = len(seq_lens)

    inp = _build_inputs(
        num_req,
        new_tokens_per_req,
        seq_lens,
        num_q_heads,
        num_kv_heads,
        head_dim,
        block_size,
        fp8_dtype,
        shuffle_block_table=True,
        norm_weight_signed=True,
    )

    if qk_norm_policy == 0:
        q_norm_w = None
        k_norm_w = None
    elif norm_weights == "both":
        q_norm_w, k_norm_w = inp["q_norm_w"], inp["k_norm_w"]
    elif norm_weights == "neither":
        q_norm_w, k_norm_w = None, None
    elif norm_weights == "q_only":
        q_norm_w, k_norm_w = inp["q_norm_w"], None
    else:  # k_only
        q_norm_w, k_norm_w = None, inp["k_norm_w"]

    ref_key = inp["key_cache"].clone()
    ref_val = inp["value_cache"].clone()
    ref_kscale = inp["k_scale"].clone()
    ref_q_fp8, ref_qscale, ref_split = _ref(
        inp["qkv"], inp["cos_sin"], inp["num_seqlen_per_req"], inp["q_index"],
        inp["kvcache_indices"], ref_key, ref_val, ref_kscale, inp["v_scale"],
        is_prefill, max(new_tokens_per_req), q_norm_w, k_norm_w,
        qk_norm_policy, 1e-6, fp8_dtype, is_neox,
    )

    tri_q_fp8, tri_qscale, tri_split = rope_norm_store_kv_fp8(
        qkv=inp["qkv"], cos_sin=inp["cos_sin"],
        num_seqlen_per_req=inp["num_seqlen_per_req"],
        q_index=inp["q_index"], kvcache_indices=inp["kvcache_indices"],
        key_cache=inp["key_cache"], value_cache=inp["value_cache"],
        k_scale=inp["k_scale"], v_scale=inp["v_scale"],
        is_prefill=is_prefill, max_seqlens=max(new_tokens_per_req),
        q_norm_weight=q_norm_w, k_norm_weight=k_norm_w,
        qk_norm_policy=qk_norm_policy, is_neox=is_neox, rms_eps=1e-6,
    )

    assert _fp8_close(tri_q_fp8, ref_q_fp8), (
        f"Q FP8 mismatch: max abs diff "
        f"{(tri_q_fp8.float() - ref_q_fp8.float()).abs().max().item()}"
    )
    torch.testing.assert_close(tri_qscale, ref_qscale, rtol=1e-2, atol=1e-4)
    assert torch.equal(tri_split, ref_split)
    assert _fp8_close(inp["key_cache"], ref_key)
    assert _fp8_close(inp["value_cache"], ref_val)
    torch.testing.assert_close(inp["k_scale"], ref_kscale, rtol=1e-2, atol=1e-4)


# ──────────────── Targeted scenario tests for the remaining couplings ─────


def _run_one(
    inp,
    is_prefill,
    max_seqlens,
    q_norm_w,
    k_norm_w,
    qk_norm_policy,
    is_neox,
    fp8_dtype,
    out_q=None,
):
    ref_key = inp["key_cache"].clone()
    ref_val = inp["value_cache"].clone()
    ref_kscale = inp["k_scale"].clone()
    ref_q_fp8, ref_qscale, ref_split = _ref(
        inp["qkv"], inp["cos_sin"], inp["num_seqlen_per_req"], inp["q_index"],
        inp["kvcache_indices"], ref_key, ref_val, ref_kscale, inp["v_scale"],
        is_prefill, max_seqlens, q_norm_w, k_norm_w,
        qk_norm_policy, 1e-6, fp8_dtype, is_neox,
    )
    tri_q_fp8, tri_qscale, tri_split = rope_norm_store_kv_fp8(
        qkv=inp["qkv"], cos_sin=inp["cos_sin"],
        num_seqlen_per_req=inp["num_seqlen_per_req"],
        q_index=inp["q_index"], kvcache_indices=inp["kvcache_indices"],
        key_cache=inp["key_cache"], value_cache=inp["value_cache"],
        k_scale=inp["k_scale"], v_scale=inp["v_scale"],
        is_prefill=is_prefill, max_seqlens=max_seqlens,
        q_norm_weight=q_norm_w, k_norm_weight=k_norm_w,
        qk_norm_policy=qk_norm_policy, is_neox=is_neox, rms_eps=1e-6,
        out_q=out_q,
    )
    assert _fp8_close(tri_q_fp8, ref_q_fp8)
    torch.testing.assert_close(tri_qscale, ref_qscale, rtol=1e-2, atol=1e-4)
    assert torch.equal(tri_split, ref_split)
    assert _fp8_close(inp["key_cache"], ref_key)
    assert _fp8_close(inp["value_cache"], ref_val)
    torch.testing.assert_close(inp["k_scale"], ref_kscale, rtol=1e-2, atol=1e-4)
    return tri_q_fp8


def test_scenario_max_seqlens_larger_than_actual():
    """Caller passes max_seqlens > max(new_tokens_per_req): prefill q_scale
    must still be ones-padded out to ``ceil(max_seqlens/128)*128``."""
    torch.manual_seed(1)
    fp8_dtype = get_fp8_e4m3_dtype()
    inp = _build_inputs(
        2, [80, 70], [200, 150], 8, 1, 128, 64, fp8_dtype,
    )
    # max_seqlens supplied is larger than actual max new-token count (80).
    _run_one(
        inp, is_prefill=True, max_seqlens=200,  # bigger than 80
        q_norm_w=inp["q_norm_w"], k_norm_w=inp["k_norm_w"],
        qk_norm_policy=1, is_neox=True, fp8_dtype=fp8_dtype,
    )


def test_scenario_preallocated_out_q():
    """Caller-supplied out_q buffer is used and filled correctly."""
    torch.manual_seed(2)
    fp8_dtype = get_fp8_e4m3_dtype()
    inp = _build_inputs(
        3, [1, 4, 16], [64, 256, 512], 8, 1, 128, 64, fp8_dtype,
    )
    num_rows = inp["num_rows"]
    out_q = torch.empty(num_rows, 8, 128, dtype=fp8_dtype, device="cuda")
    # Pre-fill with a sentinel pattern so we can confirm the kernel wrote.
    out_q.fill_(torch.finfo(fp8_dtype).max)
    tri_q = _run_one(
        inp, is_prefill=False, max_seqlens=16,
        q_norm_w=inp["q_norm_w"], k_norm_w=inp["k_norm_w"],
        qk_norm_policy=2, is_neox=False, fp8_dtype=fp8_dtype,
        out_q=out_q,
    )
    # Same object returned (in-place).
    assert tri_q.data_ptr() == out_q.data_ptr()


def test_scenario_no_block_shuffle_sanity():
    """Sanity: with monotonic block table the same code path still passes
    (regression guard against any accidental reliance on permutation)."""
    torch.manual_seed(3)
    fp8_dtype = get_fp8_e4m3_dtype()
    inp = _build_inputs(
        2, [80, 70], [200, 150], 8, 1, 128, 64, fp8_dtype,
        shuffle_block_table=False,
    )
    _run_one(
        inp, is_prefill=True, max_seqlens=80,
        q_norm_w=inp["q_norm_w"], k_norm_w=inp["k_norm_w"],
        qk_norm_policy=1, is_neox=True, fp8_dtype=fp8_dtype,
    )


def test_scenario_negative_norm_weight_path():
    """Force negative norm weights — bugs that silently abs() weights would
    flip many output signs. Also forces policy 2 + GPT-J + cross-block."""
    torch.manual_seed(4)
    fp8_dtype = get_fp8_e4m3_dtype()
    inp = _build_inputs(
        2, [80, 70], [200, 150], 16, 4, 128, 64, fp8_dtype,
    )
    # Make q-norm explicitly negative (k stays signed-random from _build).
    inp["q_norm_w"] = -inp["q_norm_w"].abs() - 0.1
    _run_one(
        inp, is_prefill=True, max_seqlens=80,
        q_norm_w=inp["q_norm_w"], k_norm_w=inp["k_norm_w"],
        qk_norm_policy=2, is_neox=False, fp8_dtype=fp8_dtype,
    )
