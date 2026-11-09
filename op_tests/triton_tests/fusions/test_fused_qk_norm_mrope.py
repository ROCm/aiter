# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.fusions.fused_qk_norm_mrope import (
    fused_qk_norm_mrope,
    fused_qk_norm_mrope_cache,
)
from aiter.ops.triton.utils._triton import arch_info

pytestmark = pytest.mark.skipif(
    arch_info.get_arch() != "gfx1250", reason="gfx1250 Gluon kernel"
)

HEAD_DIM = 256
ROPE_HALF = 32
MAX_POS = 4096
SECTION = [11, 11, 10]
BLOCK = 16
X = 8  # bf16 values in 16 bytes


def make_inputs(num_tokens, q_heads, k_heads):
    torch.manual_seed(0)
    # Q and K are column slices of one [T, gate | q | k | v] projection output.
    qkv = torch.randn(
        (num_tokens, 2 * (q_heads + k_heads) * HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    q = qkv[:, q_heads * HEAD_DIM : 2 * q_heads * HEAD_DIM]
    k = qkv[:, 2 * q_heads * HEAD_DIM : (2 * q_heads + k_heads) * HEAD_DIM]
    pos_t = torch.randint(0, MAX_POS - 16, (num_tokens,), device="cuda")
    return {
        "q": q,
        "k": k,
        "q_weight": torch.randn(HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.2,
        "k_weight": torch.randn(HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.2,
        "eps": 1e-6,
        # H and W differ from T, as for an image token.
        "positions": torch.stack([pos_t, pos_t + 3, pos_t + 11]),
        "cos": torch.rand((MAX_POS, ROPE_HALF), dtype=torch.float32, device="cuda"),
        "sin": torch.rand((MAX_POS, ROPE_HALF), dtype=torch.float32, device="cuda"),
        "mrope_section": SECTION,
    }


def torch_ref(x, weight, heads, inputs):
    """Gemma RMSNorm in fp32, then Neox RoPE on values [0, 64) with interleaved MRoPE positions."""
    x = x.float().unflatten(-1, (heads, HEAD_DIM))
    normed = (
        x
        * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + inputs["eps"])
        * (weight.float() + 1)
    )
    i = torch.arange(ROPE_HALF, device="cuda")
    use_h = (i % 3 == 1) & (i < 3 * SECTION[1])
    use_w = (i % 3 == 2) & (i < 3 * SECTION[2])
    t, h, w = inputs["positions"]
    pos = torch.where(use_h, h[:, None], torch.where(use_w, w[:, None], t[:, None]))
    # Give cos and sin a head axis.
    cos, sin = inputs["cos"][pos, i][:, None], inputs["sin"][pos, i][:, None]
    x1, x2 = normed[..., :ROPE_HALF], normed[..., ROPE_HALF : 2 * ROPE_HALF]
    out = normed.clone()
    out[..., :ROPE_HALF] = x1 * cos - x2 * sin
    out[..., ROPE_HALF : 2 * ROPE_HALF] = x2 * cos + x1 * sin
    return out.flatten(-2)


@pytest.mark.parametrize("num_tokens", [1, 37, 300, 4001])
@pytest.mark.parametrize("q_heads, k_heads", [(4, 1), (16, 2)])
def test_fused_qk_norm_mrope(num_tokens, q_heads, k_heads):
    inputs = make_inputs(num_tokens, q_heads, k_heads)
    q_out, k_out = fused_qk_norm_mrope(**inputs)
    ref_q = torch_ref(inputs["q"], inputs["q_weight"], q_heads, inputs)
    ref_k = torch_ref(inputs["k"], inputs["k_weight"], k_heads, inputs)
    # One bf16 rounding of each output value.
    torch.testing.assert_close(q_out.float(), ref_q, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(k_out.float(), ref_k, atol=1e-2, rtol=1e-2)


def make_cache_inputs(num_tokens, q_heads, k_heads):
    inputs = make_inputs(num_tokens, q_heads, k_heads)
    num_blocks = num_tokens // BLOCK + 4
    # A V with strided rows, and a different slot for each token. Some tokens are pad tokens.
    inputs["v"] = torch.randn(
        (num_tokens, 2 * k_heads * HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )[:, : k_heads * HEAD_DIM]
    # The last tokens have no slot (graph padding).
    slots = torch.randperm(num_blocks * BLOCK, device="cuda")[
        : num_tokens - num_tokens // 10
    ]
    slots[2::7] = -1
    inputs["slot_mapping"] = slots
    inputs["k_cache"] = torch.zeros(
        (num_blocks, k_heads, HEAD_DIM // X, BLOCK, X),
        dtype=torch.bfloat16,
        device="cuda",
    )
    inputs["v_cache"] = torch.zeros(
        (num_blocks, k_heads, BLOCK // X, HEAD_DIM, X),
        dtype=torch.bfloat16,
        device="cuda",
    )
    return inputs


@pytest.mark.parametrize("num_tokens", [1, 37, 300, 7001])
@pytest.mark.parametrize("q_heads, k_heads", [(4, 1), (16, 2)])
def test_fused_qk_norm_mrope_cache(num_tokens, q_heads, k_heads):
    inputs = make_cache_inputs(num_tokens, q_heads, k_heads)
    q_out, k_out = fused_qk_norm_mrope_cache(**inputs)
    ref_q = torch_ref(inputs["q"], inputs["q_weight"], q_heads, inputs)
    ref_k = torch_ref(inputs["k"], inputs["k_weight"], k_heads, inputs)
    torch.testing.assert_close(q_out.float(), ref_q, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(k_out.float(), ref_k, atol=1e-2, rtol=1e-2)

    # The tokens with a slot fill their cache rows. All other rows stay zero.
    slots = inputs["slot_mapping"]
    written = torch.zeros(num_tokens, dtype=torch.bool, device="cuda")
    written[: len(slots)] = slots >= 0
    blk, off = (
        slots[written[: len(slots)]] // BLOCK,
        slots[written[: len(slots)]] % BLOCK,
    )
    ref_k_cache = torch.zeros_like(inputs["k_cache"], dtype=torch.float32)
    ref_k_cache[blk, :, :, off, :] = ref_k[written].unflatten(
        -1, (k_heads, HEAD_DIM // X, X)
    )
    ref_v_cache = torch.zeros_like(inputs["v_cache"])
    ref_v_cache[blk, :, off // X, :, off % X] = inputs["v"][written].unflatten(
        -1, (k_heads, HEAD_DIM)
    )
    torch.testing.assert_close(
        inputs["k_cache"].float(), ref_k_cache, atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(inputs["v_cache"], ref_v_cache, atol=0, rtol=0)


@pytest.mark.parametrize("num_tokens", [1, 37, 300])
@pytest.mark.parametrize("kv_scale", [1.0, 0.5])
def test_fused_qk_norm_mrope_cache_fp8(num_tokens, kv_scale):
    """An fp8 cache: K and V stored as value / kv_scale clamped to the fp8 range, Q
    returned in fp8 (clamped, unscaled); each value rounded to bf16 first."""
    import aiter

    fp8 = aiter.dtypes.fp8
    fp8_max = torch.finfo(fp8).max
    q_heads, k_heads, block, x = 16, 2, 64, 16  # 16 fp8 values in 16 bytes
    inputs = make_inputs(num_tokens, q_heads, k_heads)
    num_blocks = num_tokens // block + 4
    v = torch.randn(
        (num_tokens, 2 * k_heads * HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )[:, : k_heads * HEAD_DIM]
    v[:, :8] = 3000.0  # past the fp8 range: stored clamped
    inputs["v"] = v
    slots = torch.randperm(num_blocks * block, device="cuda")[:num_tokens]
    slots[2::7] = -1
    inputs["slot_mapping"] = slots
    inputs["k_cache"] = torch.zeros(
        (num_blocks, k_heads, HEAD_DIM // x, block, x), dtype=torch.uint8, device="cuda"
    ).view(fp8)
    inputs["v_cache"] = torch.zeros(
        (num_blocks, k_heads, block // x, HEAD_DIM, x), dtype=torch.uint8, device="cuda"
    ).view(fp8)
    q_out, k_out = fused_qk_norm_mrope_cache(**inputs, kv_scale=kv_scale)
    ref_q = torch_ref(inputs["q"], inputs["q_weight"], q_heads, inputs).bfloat16()
    ref_k = torch_ref(inputs["k"], inputs["k_weight"], k_heads, inputs).bfloat16()
    assert q_out.dtype == fp8 and k_out.dtype == torch.bfloat16
    to_fp8 = lambda t: t.float().clamp(-fp8_max, fp8_max).to(fp8).float()
    torch.testing.assert_close(q_out.float(), to_fp8(ref_q), atol=0.5, rtol=0.125)
    torch.testing.assert_close(k_out.float(), ref_k.float(), atol=1e-2, rtol=1e-2)

    written = slots >= 0
    blk, off = slots[written] // block, slots[written] % block
    ref_k_cache = torch.zeros(
        inputs["k_cache"].shape, dtype=torch.float32, device="cuda"
    )
    ref_k_cache[blk, :, :, off, :] = to_fp8(ref_k.float() / kv_scale)[written].unflatten(
        -1, (k_heads, HEAD_DIM // x, x)
    )
    ref_v_cache = torch.zeros(
        inputs["v_cache"].shape, dtype=torch.float32, device="cuda"
    )
    ref_v_cache[blk, :, off // x, :, off % x] = to_fp8(v.float() / kv_scale)[
        written
    ].unflatten(-1, (k_heads, HEAD_DIM))
    torch.testing.assert_close(
        inputs["k_cache"].float(), ref_k_cache, atol=0.5, rtol=0.125
    )
    # V takes no arithmetic before the cast but the scale: exact.
    assert not torch.isnan(inputs["v_cache"].float()).any()
    torch.testing.assert_close(inputs["v_cache"].float(), ref_v_cache, atol=0, rtol=0)


@pytest.mark.parametrize("num_tokens", [1, 37, 300, 1100])
@pytest.mark.parametrize("block", [64, 128])
@pytest.mark.parametrize("q_heads, k_heads", [(16, 2), (4, 1)])
@pytest.mark.parametrize("kv_scale", [1.0, 0.5])
def test_fused_qk_norm_mrope_cache_k4v8(num_tokens, block, q_heads, k_heads, kv_scale):
    """An A8K4V8 cache (NVFP4 K, fp8 V): byte-identical, over the whole cache, to the two-kernel path --
    fused_qk_norm_mrope, then reshape_and_cache_k4v8 of its bf16 K (k_scale 1) and of V / kv_scale --
    and Q in fp8 as the fp8 cache returns it."""
    import aiter
    from aiter.ops.triton.fusions.fused_kv_cache import reshape_and_cache_k4v8

    fp8 = aiter.dtypes.fp8
    fp8_max = torch.finfo(fp8).max
    inputs = make_inputs(num_tokens, q_heads, k_heads)
    num_blocks = num_tokens // block + 4
    v = torch.randn(
        (num_tokens, 2 * k_heads * HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )[:, : k_heads * HEAD_DIM]
    inputs["v"] = v
    slots = torch.randperm(num_blocks * block, device="cuda")[:num_tokens]
    slots[2::7] = -1  # pad tokens
    # the last tokens past the slot table are padding too
    slots = slots[: max(1, num_tokens - 2)] if num_tokens > 3 else slots
    inputs["slot_mapping"] = slots
    k_row = HEAD_DIM // 2 + HEAD_DIM // 16
    torch.manual_seed(5)  # random old content: rows not written must keep it
    k_cache0 = torch.randint(
        0, 256, (num_blocks, k_heads, block, k_row), dtype=torch.uint8, device="cuda"
    )
    v_cache0 = torch.randn(
        (num_blocks, k_heads, block // 16, HEAD_DIM, 16), device="cuda"
    ).to(fp8)
    inputs["k_cache"], inputs["v_cache"] = k_cache0.clone(), v_cache0.clone()

    q_out, k_out = fused_qk_norm_mrope_cache(**inputs, kv_scale=kv_scale)

    no_cache = {
        k: inputs[k]
        for k in ("q", "k", "q_weight", "k_weight", "eps", "positions", "cos", "sin", "mrope_section")
    }
    q_ref, k_ref = fused_qk_norm_mrope(**no_cache)
    k_cache_ref, v_cache_ref = k_cache0.clone(), v_cache0.clone()
    slots_full = torch.full((num_tokens,), -1, dtype=slots.dtype, device="cuda")
    slots_full[: len(slots)] = slots
    reshape_and_cache_k4v8(
        k_ref.view(num_tokens, k_heads, HEAD_DIM),
        v.view(num_tokens, k_heads, HEAD_DIM),
        k_cache_ref,
        v_cache_ref,
        slots_full,
        k_scale=None,
        v_scale=torch.tensor([kv_scale], dtype=torch.float32, device="cuda"),
    )
    assert q_out.dtype == fp8 and k_out.dtype == torch.bfloat16
    assert torch.equal(k_out, k_ref), "bf16 K output differs from fused_qk_norm_mrope"
    q_ref8 = q_ref.float().clamp(-fp8_max, fp8_max).to(fp8)
    assert torch.equal(q_out.view(torch.uint8), q_ref8.view(torch.uint8)), "fp8 Q differs"
    assert torch.equal(
        inputs["v_cache"].view(torch.uint8), v_cache_ref.view(torch.uint8)
    ), "fp8 V cache differs from reshape_and_cache_k4v8"
    assert torch.equal(inputs["k_cache"], k_cache_ref), "NVFP4 K cache differs from reshape_and_cache_k4v8"


def _k4v8_page_slots(scenario, block, num_blocks, gen):
    """Slot mappings of prefill-like batches: (slots [S], num_tokens T >= S)."""
    pages = torch.randperm(num_blocks, generator=gen).tolist()
    nxt = iter(pages)

    def run(start, n):  # n tokens of one sequence from position `start` on fresh pages
        first = [next(nxt) for _ in range((start + n + block - 1) // block)]
        pos = torch.arange(start, start + n)
        return [first[p // block] * block + p % block for p in pos.tolist()]

    if scenario == "full_pages":
        s = run(0, 4 * block)
    elif scenario == "mid_start":  # a chunk continuing a cached prefix: starts mid-page, ends mid-page
        s = run(13, 3 * block + 20)
    elif scenario == "multi":  # several sequences, pads, and tokens past the slot table
        s = run(0, 2 * block + 5) + [-1, -1] + run(block - 7, block + 30) + run(0, block) + [-1] + run(5, 40)
        s = s + run(0, block)  # a full page at the very end
        return torch.tensor(s[:-3]), len(s)  # the last 3 tokens have no slot
    elif scenario == "spec_decode":  # 4 contiguous tokens a sequence: never a full page
        s = []
        for _ in range(96):
            s += run(int(torch.randint(0, block - 4, (1,), generator=gen)), 4)
    elif scenario == "multi_mid":  # 8 sequences, each from its own mid-page offset, 1-5 pages, pads at the end
        s = []
        for i in range(8):
            s += run(int(torch.randint(1, block, (1,), generator=gen)), int(torch.randint(block // 2, 5 * block, (1,), generator=gen)))
        s += [-1] * 7
    elif scenario == "short":  # sequences shorter than a page, some page-aligned
        s = []
        for i in range(40):
            start = 0 if i % 3 == 0 else int(torch.randint(1, block // 2, (1,), generator=gen))
            s += run(start, int(torch.randint(1, block // 2, (1,), generator=gen)))
    elif scenario == "exact_pages_shuffled":  # full pages, each its own sequence, in random page order
        s = []
        for _ in range(5):
            s += run(0, block)
    return torch.tensor(s), len(s)


@pytest.mark.parametrize(
    "scenario",
    ["full_pages", "mid_start", "multi", "multi_mid", "short", "spec_decode", "exact_pages_shuffled"],
)
@pytest.mark.parametrize("block", [64, 128])
@pytest.mark.parametrize("q_heads, k_heads", [(16, 2), (4, 1)])
def test_fused_qk_norm_mrope_cache_k4v8_pages(scenario, block, q_heads, k_heads):
    """Prefill-like slot maps: the page path (full pages through LDS + TDM) writes the same bytes as the
    row path alone and as the two-kernel reference (fused_qk_norm_mrope + reshape_and_cache_k4v8)."""
    import aiter
    import aiter.ops.triton.fusions.fused_qk_norm_mrope as mod
    from aiter.ops.triton.fusions.fused_kv_cache import reshape_and_cache_k4v8

    fp8 = aiter.dtypes.fp8
    kv_scale = 0.5
    gen = torch.Generator().manual_seed(3)
    num_blocks = 200  # spec_decode takes a fresh page for each of its 96 sequences
    slots, num_tokens = _k4v8_page_slots(scenario, block, num_blocks, gen)
    slots = slots.to(torch.int64).cuda()
    inputs = make_inputs(num_tokens, q_heads, k_heads)
    v = torch.randn(
        (num_tokens, 2 * k_heads * HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )[:, : k_heads * HEAD_DIM]
    inputs["v"] = v
    inputs["slot_mapping"] = slots
    k_row = HEAD_DIM // 2 + HEAD_DIM // 16
    torch.manual_seed(5)
    k_cache0 = torch.randint(
        0, 256, (num_blocks, k_heads, block, k_row), dtype=torch.uint8, device="cuda"
    )
    v_cache0 = torch.randn((num_blocks, k_heads, block // 16, HEAD_DIM, 16), device="cuda").to(fp8)

    def run(min_tokens):
        old = mod._K4V8_PAGE_MIN_TOKENS
        mod._K4V8_PAGE_MIN_TOKENS = min_tokens
        try:
            inp = dict(inputs, k_cache=k_cache0.clone(), v_cache=v_cache0.clone())
            q_out, k_out = fused_qk_norm_mrope_cache(**inp, kv_scale=kv_scale)
            return q_out, k_out, inp["k_cache"], inp["v_cache"]
        finally:
            mod._K4V8_PAGE_MIN_TOKENS = old

    q_p, k_p, kc_p, vc_p = run(1)  # the page path on
    q_r, k_r, kc_r, vc_r = run(0)  # rows only
    assert torch.equal(q_p.view(torch.uint8), q_r.view(torch.uint8))
    assert torch.equal(k_p, k_r)
    assert torch.equal(vc_p.view(torch.uint8), vc_r.view(torch.uint8)), "V cache: page path != row path"
    assert torch.equal(kc_p, kc_r), "K cache: page path != row path"

    no_cache = {
        k: inputs[k]
        for k in ("q", "k", "q_weight", "k_weight", "eps", "positions", "cos", "sin", "mrope_section")
    }
    _, k_ref = fused_qk_norm_mrope(**no_cache)
    kc_ref, vc_ref = k_cache0.clone(), v_cache0.clone()
    slots_full = torch.full((num_tokens,), -1, dtype=torch.int64, device="cuda")
    slots_full[: len(slots)] = slots
    reshape_and_cache_k4v8(
        k_ref.view(num_tokens, k_heads, HEAD_DIM),
        v.view(num_tokens, k_heads, HEAD_DIM),
        kc_ref,
        vc_ref,
        slots_full,
        k_scale=None,
        v_scale=torch.tensor([kv_scale], dtype=torch.float32, device="cuda"),
    )
    assert torch.equal(vc_p.view(torch.uint8), vc_ref.view(torch.uint8)), "V cache != reference"
    assert torch.equal(kc_p, kc_ref), "K cache != reference"
