"""FlyDSL V4.1 fused compressor against a torch reference.

Only valid plan rows are compared. The latent (pooling and RMSNorm) and the
rotated echo (RoPE) are checked against an fp64 reference within BF16
rounding. The packed cache row must equal, byte for byte, the NVFP4 / MXFP8
quant of the kernel's own rotated echo, which is the value the quant reads.
"""

import pytest
import torch

from aiter.ops.flydsl.kernels.fused_compress_attn_v41 import fused_compress_attn_v41

_DIM = 512
_ROPE = 64
_FORMATS = {"nvfp4": 288, "mxfp8": 528}


def _stage(name, got, expected):
    if torch.equal(got, expected):
        return
    mismatch = int((got != expected).sum())
    raise AssertionError(f"{name}: {mismatch} elements differ")


def _strided(rows, cols, dtype):
    wide = torch.empty(rows, cols + 16, device="cuda", dtype=dtype)
    wide[:, cols:] = 7
    return wide[:, :cols]


def _plan(ratio):
    # ragged_id is the last token of the group. window_len steps come from
    # the state ring; position 0 with ratio 2 pads the first of the two steps.
    if ratio == 2:
        rows = [
            [1, 0, 1, 0],
            [3, 0, 3, 1],
            [5, 1, 0, 1],
            [9, 0, 9, 0],
            [0, 0, -1, 0],
            [4, -1, -1, 0],
            [7, 1, 2, 0],
        ]
    else:
        rows = [
            [0, 0, 0, 0],
            [2, 0, 2, 1],
            [4, 1, 5, 0],
            [1, -1, -1, 0],
        ]
    return torch.tensor(rows, dtype=torch.int32, device="cuda")


def _inputs(ratio):
    tokens = 10
    dtype = torch.float32 if ratio == 2 else torch.bfloat16
    kv = _strided(tokens, _DIM, dtype)
    score = _strided(tokens, _DIM, dtype)
    kv.copy_(torch.randn(tokens, _DIM, device="cuda"))
    score.copy_(torch.randn(tokens, _DIM, device="cuda"))
    kv[0, 0] = 10000
    kv[0, 1] = torch.tensor(float("-0"), device="cuda").to(dtype)
    state_wide = torch.empty(2, 8, _DIM + 16, device="cuda")
    score_wide = torch.empty(2, 8, _DIM + 16, device="cuda")
    state = state_wide[:, :, :_DIM]
    score_state = score_wide[:, :, :_DIM]
    state.copy_(torch.randn(2, 8, _DIM, device="cuda"))
    score_state.copy_(torch.randn(2, 8, _DIM, device="cuda"))
    ape = torch.randn(ratio, _DIM, device="cuda")
    weight = torch.randn(_DIM, device="cuda").abs() + 0.1
    cos = torch.randn(16, _ROPE // 2, device="cuda").to(torch.bfloat16)
    sin = torch.randn(16, _ROPE // 2, device="cuda").to(torch.bfloat16)
    slots = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    tables = torch.tensor([[1, 3, 0], [4, 2, 5]], dtype=torch.int32, device="cuda")
    return kv, score, state, score_state, ape, weight, cos, sin, slots, tables


def _cache(num_pages, rows_per_page, row_bytes, gap):
    page_stride = rows_per_page * row_bytes + gap
    storage = torch.full(
        (num_pages * page_stride,), 0xA5, dtype=torch.uint8, device="cuda"
    )
    cache = storage.as_strided(
        (num_pages, rows_per_page, row_bytes),
        (page_stride, row_bytes, 1),
    )
    return storage, cache


def _reference(
    kv, score, state, score_state, plan, slots, ape, weight, cos, sin, ratio
):
    latent = torch.zeros(plan.shape[0], _DIM, dtype=torch.float64, device="cuda")
    rotated = torch.zeros_like(latent)
    for i, (ragged, batch, position, window) in enumerate(plan.tolist()):
        if position < 0 or batch < 0:
            continue
        slot = int(slots[batch])
        kvs, scores = [], []
        for k in range(ratio):
            s = position - ratio + 1 + k
            if s < 0:
                kvs.append(torch.zeros_like(latent[i]))
                scores.append(torch.full_like(latent[i], float("-inf")))
            elif k >= window:
                row = ragged - (ratio - 1 - k)
                kvs.append(kv[row].double())
                scores.append(score[row].double() + ape[k].double())
            else:
                ring = s % state.shape[1]
                kvs.append(state[slot, ring].double())
                scores.append(score_state[slot, ring].double())
        weights = torch.softmax(torch.stack(scores), 0)
        comp = (weights * torch.stack(kvs)).sum(0)
        normed = comp * torch.rsqrt((comp * comp).mean() + 1e-6) * weight.double()
        latent[i] = normed
        entry = (position // ratio) * ratio
        c, sn = cos[entry].double(), sin[entry].double()
        even, odd = normed[-_ROPE::2], normed[-_ROPE + 1 :: 2]
        rotated[i] = normed
        rotated[i, -_ROPE:] = torch.stack(
            [even * c - odd * sn, odd * c + even * sn], -1
        ).flatten()
    return latent, rotated


def _nvfp4_rule(rotated):
    # Group 16, E4M3 scale max(amax, 6 * 2^-9) / 6, asymmetric midpoints.
    x = rotated.float().cpu().view(32, 16)
    amax = x.abs().amax(1).clamp_min(6.0 * 2.0**-9)
    scale = (amax * torch.tensor(1.0 / 6.0)).to(torch.float8_e4m3fn)
    mag = (x / scale.float()[:, None]).abs()
    code = torch.zeros_like(mag, dtype=torch.int32)
    for level, (bound, strict) in enumerate(
        [(0.25, 1), (0.75, 0), (1.25, 1), (1.75, 0), (2.5, 1), (3.5, 0), (5.0, 1)]
    ):
        code = torch.where(mag > bound if strict else mag >= bound, level + 1, code)
    nibble = code | (((x.view(torch.int32) >> 31) & 1) << 3)
    values = (nibble[:, 0::2] | (nibble[:, 1::2] << 4)).to(torch.uint8).flatten()
    return torch.cat([values, scale.view(torch.uint8)])


def _mx_rule(rotated):
    # aiter's MXFP8 rule: RoundUp scale over a NaN-skipping amax, capped at
    # 0xFE; values are divided by the scale without a clamp.
    x = rotated.float().cpu().view(16, 32)
    amax = torch.where(x.isnan(), 0.0, x.abs()).amax(1).clamp_min(1e-10)
    bits = (amax * torch.tensor(1.0 / 448)).view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    code = exponent + ((exponent < 0xFF) & ((bits & 0x7FFFFF) != 0)).int()
    code = code.clamp_max(254)
    q = (x.double() * torch.exp2(127.0 - code.double())[:, None]).float()
    values = q.to(torch.float8_e4m3fn).view(torch.uint8).flatten()
    return torch.cat([values, code.to(torch.uint8)])


def _expected_cache(rotated, plan, tables, cache, ratio, cache_format):
    rule = _nvfp4_rule if cache_format == "nvfp4" else _mx_rule
    out = cache.clone()
    rows_per_page = cache.shape[1]
    for i, (_, batch, position, _) in enumerate(plan.tolist()):
        if position < 0 or batch < 0:
            continue
        entry = position // ratio
        page = int(tables[batch, entry // rows_per_page])
        out[page, entry % rows_per_page].copy_(rule(rotated[i]))
    return out


def _live(plan):
    return (plan[:, 1] >= 0) & (plan[:, 2] >= 0)


@pytest.mark.parametrize("cache_format", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("padded", [False, True])
def test_fused_compress_attn_v41(cache_format, ratio, padded):
    torch.manual_seed(ratio * 19 + (0 if cache_format == "nvfp4" else 5))
    plan = _plan(ratio)
    if padded:
        # Enough idle rows that FlyDSL leaves its few-row split-row launch.
        idle = torch.tensor([[0, -1, -1, 0]], dtype=torch.int32, device="cuda")
        plan = torch.cat([plan, idle.expand(4200, 4)])
    kv, score, state, score_state, ape, weight, cos, sin, slots, tables = _inputs(ratio)
    storage, cache = _cache(6, 4, _FORMATS[cache_format], gap=64)
    latent = torch.zeros(plan.shape[0], _DIM, dtype=torch.bfloat16, device="cuda")
    rotated = torch.zeros_like(latent)
    fused_compress_attn_v41(
        kv_in=kv,
        score_in=score,
        kv_state=state,
        score_state=score_state,
        plan=plan,
        state_slot_mapping=slots,
        ape=ape,
        rms_weight=weight,
        rms_eps=1e-6,
        cos_cache=cos,
        sin_cache=sin,
        cache=cache,
        block_tables=tables,
        ratio=ratio,
        cache_format=cache_format,
        latent_out=latent,
        rotated_out=rotated,
    )
    live = _live(plan)
    ref_latent, ref_rotated = _reference(
        kv, score, state, score_state, plan, slots, ape, weight, cos, sin, ratio
    )
    torch.testing.assert_close(
        latent[live].double(), ref_latent[live], rtol=2**-7, atol=1e-6
    )
    torch.testing.assert_close(
        rotated[live].double(), ref_rotated[live], rtol=2**-7, atol=1e-6
    )
    expected = _expected_cache(rotated, plan, tables, cache, ratio, cache_format)
    _stage("quant or paged store (packed row differs)", cache, expected)
    page_stride = 4 * _FORMATS[cache_format] + 64
    for page in range(6):
        gap = storage[
            page * page_stride + 4 * _FORMATS[cache_format] : (page + 1) * page_stride
        ]
        assert torch.all(gap == 0xA5)
    inactive = ~live
    assert torch.all(latent[inactive] == 0)
    assert torch.all(rotated[inactive] == 0)


def test_negative_batch_does_not_write():
    torch.manual_seed(3)
    plan = torch.tensor([[2, -1, 4, 0]], dtype=torch.int32, device="cuda")
    kv, score, state, score_state, ape, weight, cos, sin, slots, tables = _inputs(2)
    _, cache = _cache(6, 4, 288, gap=64)
    before = cache.clone()
    latent = torch.zeros(1, _DIM, dtype=torch.bfloat16, device="cuda")
    rotated = torch.zeros_like(latent)
    fused_compress_attn_v41(
        kv_in=kv,
        score_in=score,
        kv_state=state,
        score_state=score_state,
        plan=plan,
        state_slot_mapping=slots,
        ape=ape,
        rms_weight=weight,
        rms_eps=1e-6,
        cos_cache=cos,
        sin_cache=sin,
        cache=cache,
        block_tables=tables,
        ratio=2,
        cache_format="nvfp4",
        latent_out=latent,
        rotated_out=rotated,
    )
    assert torch.equal(cache, before)
    assert torch.all(latent == 0)
    assert torch.all(rotated == 0)


def _same_fp8(got, expected):
    # Inf converts to a NaN whose sign differs by arch (0x7F or 0xFF).
    nan = lambda b: (b & 0x7F) == 0x7F
    return torch.equal(nan(got), nan(expected)) and torch.equal(
        got[~nan(got)], expected[~nan(expected)]
    )


def test_mxfp8_nonfinite_follows_aiter_mx_rule():
    torch.manual_seed(4)
    plan = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32, device="cuda")
    kv, score, state, score_state, ape, weight, cos, sin, slots, tables = _inputs(1)

    def run(weight):
        _, cache = _cache(6, 4, _FORMATS["mxfp8"], gap=64)
        rotated = torch.zeros(1, _DIM, dtype=torch.bfloat16, device="cuda")
        fused_compress_attn_v41(
            kv_in=kv,
            score_in=score,
            kv_state=state,
            score_state=score_state,
            plan=plan,
            state_slot_mapping=slots,
            ape=ape,
            rms_weight=weight,
            rms_eps=1e-6,
            cos_cache=cos,
            sin_cache=sin,
            cache=cache,
            block_tables=tables,
            ratio=1,
            cache_format="mxfp8",
            rotated_out=rotated,
        )
        return cache[int(tables[0, 0]), 0].cpu(), rotated[0]

    # The weight scales single elements after RMSNorm: group 0 holds an Inf
    # beside a finite 2^127 (element 0, the large one), group 1 a NaN, group
    # 2 is all NaN. The echo is BF16, so a few rounds pin element 0 to 2^127.
    for _ in range(3):
        _, rotated = run(weight)
        weight[0] *= 2.0**127 / float(rotated[0])
    weight[8] = float("inf")
    weight[40] = float("nan")
    weight[64:96] = float("nan")
    row, rotated = run(weight)
    assert float(rotated[0]) == 2.0**127
    expected = _mx_rule(rotated)
    assert row[_DIM] == 0xFE and row[0] == 0x38
    assert _same_fp8(row[:_DIM], expected[:_DIM])
    assert torch.equal(row[_DIM:], expected[_DIM:])
