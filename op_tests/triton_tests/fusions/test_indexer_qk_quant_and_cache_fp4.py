# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.cache import indexer_qk_rope_quant_and_cache
from aiter.ops.triton.fusions.indexer_qk_quant_and_cache_fp4 import (
    indexer_qk_quant_and_cache_fp4,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.utility.fp4_utils import e8m0_to_f32, f32_to_mx_e8m0_scale, f32_to_mxfp4
from aiter.utility.mx_types import MxScaleRoundModeInt

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_arch() != "gfx950", reason="requires gfx950"
)


def generate_inputs(rows, heads, rope_dim, *, strided=False):
    stride = 2 if strided else 1
    q = torch.randn((rows * stride, heads, 128), device="cuda", dtype=torch.bfloat16)[
        ::stride
    ]
    k = torch.randn((rows * stride, 128), device="cuda", dtype=torch.bfloat16)[::stride]
    weights = torch.randn((rows, heads * stride), device="cuda", dtype=torch.bfloat16)[
        :, :heads
    ]
    pages = (rows + 63) // 64 + 2
    return {
        "q": q,
        "k": k,
        "weights": weights,
        "weights_out": torch.full(
            (rows, heads), -99, device="cuda", dtype=torch.bfloat16
        ),
        "q_out": torch.full((rows, heads, 64), 0xA5, device="cuda", dtype=torch.uint8),
        "q_scale_out": torch.full(
            (rows, 2, 32, heads // 16), 0xA5, device="cuda", dtype=torch.uint8
        ),
        "kv_cache": torch.full(
            (pages, 4, 64, 16), 0xA5, device="cuda", dtype=torch.uint8
        ),
        "kv_cache_scale": torch.full(
            (pages, 2, 32, 4), 0xA5, device="cuda", dtype=torch.uint8
        ),
        "slot_mapping": torch.arange(rows, device="cuda", dtype=torch.int64) + 64,
        "norm_weight": torch.randn((128,), device="cuda", dtype=torch.float32),
        "norm_bias": torch.randn((128,), device="cuda", dtype=torch.float32),
        "positions": (
            torch.arange(rows, device="cuda", dtype=torch.int64) if rope_dim else None
        ),
        "cos_cache": (
            torch.randn((max(rows, 1), 32), device="cuda", dtype=torch.bfloat16)
            if rope_dim
            else None
        ),
        "sin_cache": (
            torch.randn((max(rows, 1), 32), device="cuda", dtype=torch.bfloat16)
            if rope_dim
            else None
        ),
        "epsilon": 1e-6,
        "rope_dim": rope_dim,
    }


def _rope_ref(x, args):
    x = x.float()
    if not args["rope_dim"]:
        return x
    pos = args["positions"].clamp(0, args["cos_cache"].shape[0] - 1)
    cos, sin = (args[name][pos].float() for name in ("cos_cache", "sin_cache"))
    if x.ndim == 3:
        cos, sin = cos[:, None], sin[:, None]
    if args.get("is_neox", True):
        a, b = x[..., :32].clone(), x[..., 32:64].clone()
        x[..., :32], x[..., 32:64] = a * cos - b * sin, b * cos + a * sin
    else:
        a, b = x[..., :64:2].clone(), x[..., 1:64:2].clone()
        x[..., :64:2], x[..., 1:64:2] = a * cos - b * sin, b * cos + a * sin
    return x.bfloat16().float()


def _quant_ref(x):
    groups = x.reshape(*x.shape[:-1], 4, 32)
    scale = f32_to_mx_e8m0_scale(
        groups.abs().amax(-1), mode=MxScaleRoundModeInt.RoundUp
    )
    scaled = groups / e8m0_to_f32(scale).unsqueeze(-1)
    return f32_to_mxfp4(scaled.flatten(-2)).view(torch.uint8), scale.view(torch.uint8)


def _assert_reference(args):
    q_ref, qs_ref = _quant_ref(_rope_ref(args["q"], args))
    k = args["k"].float()
    k = k - k.mean(-1, keepdim=True)
    k = k * torch.rsqrt(k.square().mean(-1, keepdim=True) + args["epsilon"])
    nw, nb = args["norm_weight"], args["norm_bias"]
    if args["rope_dim"]:
        k = (k.double() * nw.double() + nb.double()).float()
    else:
        k = k * nw.bfloat16().float() + nb.bfloat16().float()
    k_ref, ks_ref = _quant_ref(_rope_ref(k.bfloat16(), args))
    rows, heads, _ = args["q"].shape
    slots = args["slot_mapping"]
    live = torch.zeros(rows, device="cuda", dtype=torch.bool)
    live[: slots.numel()] = slots >= 0
    q_live = torch.ones_like(live) if args.get("compute_all_q_rope", False) else live
    torch.testing.assert_close(args["q_out"][q_live], q_ref[q_live], rtol=0, atol=0)
    torch.testing.assert_close(
        args["weights_out"][q_live], args["weights"][q_live], rtol=0, atol=0
    )
    assert (args["q_out"][~q_live] == 0xA5).all()
    assert (args["q_scale_out"][~q_live] == 0xA5).all()
    assert (args["weights_out"][~q_live] == -99).all()
    for head in range(heads):
        for group in range(4):
            got = args["q_scale_out"][
                q_live, group % 2, head % 32, group // 2 * (heads // 32) + head // 32
            ]
            torch.testing.assert_close(got, qs_ref[q_live, head, group], rtol=0, atol=0)
    valid_slots = slots[slots >= 0]
    page, row = valid_slots // 64, valid_slots % 64
    # Read each physical OPUS group directly; no production shuffle is used.
    for group in range(4):
        got = args["kv_cache"][page, group, row, :]
        torch.testing.assert_close(
            got, k_ref[live, group * 16 : (group + 1) * 16], rtol=0, atol=0
        )
        got = args["kv_cache_scale"][
            page, group % 2, row % 32, group // 2 * 2 + row // 32
        ]
        torch.testing.assert_close(got, ks_ref[live, group], rtol=0, atol=0)
    untouched = torch.ones(
        (args["kv_cache"].shape[0], 64), device="cuda", dtype=torch.bool
    )
    untouched[page, row] = False
    p, r = untouched.nonzero(as_tuple=True)
    assert (args["kv_cache"][p, :, r, :] == 0xA5).all()
    for group in range(4):
        assert (
            args["kv_cache_scale"][p, group % 2, r % 32, group // 2 * 2 + r // 32]
            == 0xA5
        ).all()


@pytest.mark.parametrize(
    "heads,rope_dim,is_neox",
    [(32, 0, True), (64, 0, True), (32, 64, True), (64, 64, False)],
)
@pytest.mark.parametrize("all_q", [False, True])
def test_opus_writer_bytes_and_graph(heads, rope_dim, is_neox, all_q):
    torch.manual_seed(17)
    args = generate_inputs(71, heads, rope_dim, strided=True)
    args.update(is_neox=is_neox, compute_all_q_rope=all_q)
    args["slot_mapping"] = args["slot_mapping"][:-3]
    args["slot_mapping"][::3] = -1
    indexer_qk_quant_and_cache_fp4(**args)
    _assert_reference(args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        indexer_qk_quant_and_cache_fp4(**args)
    args["q"].mul_(2)
    args["k"].neg_()
    graph.replay()
    _assert_reference(args)


def test_public_cache_entry_and_empty_input():
    for rows in (0, 6):
        args = generate_inputs(rows, 32, 0)
        indexer_qk_rope_quant_and_cache(
            **args,
            scale_layout="opus",
            quant_block_size=32,
            scale_fmt="ue8m0",
            weights_scale=0.1
        )
        if rows:
            _assert_reference(args)


@pytest.mark.parametrize(
    "field,error",
    [
        ("q_scale_out", "uint8 output"),
        ("kv_cache_scale", "uint8 output"),
        ("kv_cache", "K cache"),
    ],
)
def test_reject_invalid_cache_views(field, error):
    args = generate_inputs(1, 32, 0)
    args[field] = args[field].flatten() if field == "kv_cache" else None
    with pytest.raises(ValueError, match=error):
        indexer_qk_quant_and_cache_fp4(**args)


def test_cache_entry_keeps_legacy_dispatch(monkeypatch):
    from aiter.ops import cache

    args = generate_inputs(1, 32, 64)
    args.pop("rope_dim")
    seen = []
    monkeypatch.setattr(
        cache, "_indexer_qk_rope_quant_and_cache_hip", lambda *a: seen.append(a)
    )
    cache.indexer_qk_rope_quant_and_cache(
        **args, quant_block_size=128, scale_fmt="ue8m0", weights_scale=0.1
    )
    assert len(seen) == 1
    assert seen[0][0] is args["q"] and seen[0][-2] is args["q_scale_out"]
    with pytest.raises(ValueError, match="group32"):
        cache.indexer_qk_rope_quant_and_cache(
            **args,
            scale_layout="opus",
            quant_block_size=128,
            scale_fmt="ue8m0",
            weights_scale=0.1
        )


def test_opus_cache_page_beyond_four_gib():
    args = generate_inputs(1, 32, 0)
    page = (1 << 20) + 1
    args["kv_cache"] = torch.empty(
        (page + 2, 4, 64, 16), device="cuda", dtype=torch.uint8
    )
    args["kv_cache_scale"] = torch.empty(
        (page + 2, 2, 32, 4), device="cuda", dtype=torch.uint8
    )
    for p in (1, page - 1, page, page + 1):
        args["kv_cache"][p].fill_(0xA5)
        args["kv_cache_scale"][p].fill_(0xA5)
    args["slot_mapping"].fill_(page * 64 + 63)
    indexer_qk_quant_and_cache_fp4(**args)
    for p in (1, page - 1, page + 1):
        assert (args["kv_cache"][p] == 0xA5).all()
        assert (args["kv_cache_scale"][p] == 0xA5).all()
    assert (args["kv_cache"][page, :, :63] == 0xA5).all()
    assert (args["kv_cache"][page, :, 63] != 0xA5).any()
