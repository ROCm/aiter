# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Byte-exact tests for the fused scatter + e8m0-scale preshuffle kernel.

Reference = the old two-pass path that the kernel fuses:
    scatter a1_scale_token_u8 -> a1_scale_raw (row-major grouped layout)
    _grouped_a8w4_preshuffle_e8m0_scale(a1_scale_raw)
Fused = flydsl_moe_scatter_preshuffle_scale (single kernel).

The fused kernel writes EVERY output position exactly once (valid rows get the
gathered scale, padding lanes get 0). The zero-init reference is likewise fully
defined (its padding rows are 0, permuted to 0). So a full-tensor byte compare
is valid regardless of how the output buffer was initialised -- which the
`init` variants (zero / garbage / None) also assert.

The matrix sweeps the kernel's whole geometry space:
  * model_dim  -> Ws = model_dim//32 -> src_dwords (incl. odd 7168 -> Ws=224)
  * warp_tile_m -> wmma_rep in {1,2,4,8} (8 exercises the dwordx4-chunked store)
  * scale_k_per_tile in {4,8,16} (k_wmma_steps in {1,2,4})
  * routing patterns: randperm / single-expert-skew / subset-skew, and shapes
    from a single token up to many tiles, empty experts, full experts, E==1,
    topk==E, prime sizes.
"""

import torch
import pytest

from aiter.ops.flydsl.grouped_moe_gfx1250 import (
    _build_route_maps_naive,
    _grouped_a8w4_preshuffle_e8m0_scale,
)
from aiter.ops.flydsl.moe_kernels import (
    flydsl_moe_scatter_preshuffle_scale,
    flydsl_moe_preshuffle_scale,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _make_topk_ids(pattern, token_num, topk, E, seed):
    assert topk <= E, (topk, E)
    g = torch.Generator().manual_seed(seed)
    if pattern == "randperm":
        rows = [torch.randperm(E, generator=g)[:topk] for _ in range(token_num)]
    elif pattern == "single":
        # every token hits the same `topk` experts -> those experts are full,
        # the remaining E-topk experts are entirely padding.
        fixed = torch.arange(topk)
        rows = [fixed.clone() for _ in range(token_num)]
    elif pattern == "skew":
        # concentrate routes into a subset -> mixed full/partial/empty experts.
        sub = min(E, max(topk, E // 3 if E >= 3 else E))
        rows = [torch.randperm(sub, generator=g)[:topk] for _ in range(token_num)]
    else:
        raise ValueError(pattern)
    return torch.stack(rows).to(torch.int32)


def _padded_max_m(token_num, warp_tile_m):
    raw = token_num  # production static upper bound (<=1 row/expert/token)
    return max(warp_tile_m, ((raw + warp_tile_m - 1) // warp_tile_m) * warp_tile_m)


def _reference(a1_scale_token_u8, rows_to_tokens, E, max_m, warp_tile_m, scale_k_per_tile):
    device = a1_scale_token_u8.device
    Ws = a1_scale_token_u8.shape[1]
    a1_scale_raw = torch.zeros((E, max_m, Ws), dtype=torch.uint8, device=device)
    rows = torch.nonzero(rows_to_tokens >= 0, as_tuple=False).squeeze(1)
    toks = rows_to_tokens[rows].to(torch.long)
    a1_scale_raw.view(E * max_m, Ws)[rows] = a1_scale_token_u8[toks]
    return _grouped_a8w4_preshuffle_e8m0_scale(
        a1_scale_raw, warp_tile=warp_tile_m, scale_k_per_tile=scale_k_per_tile
    )


def _run_case(model_dim, warp_tile_m, scale_k_per_tile, token_num, topk, E, pattern,
              init="zero", seed=0):
    device = "cuda"
    Ws = model_dim // 32
    wmma_rep = warp_tile_m // 16
    max_m = _padded_max_m(token_num, warp_tile_m)

    topk_ids = _make_topk_ids(pattern, token_num, topk, E, seed).to(device)
    _, rows_to_tokens, _ = _build_route_maps_naive(topk_ids, E, max_m)

    gen = torch.Generator(device=device).manual_seed(seed + 1)
    a1_scale_token_u8 = torch.randint(
        0, 256, (token_num, Ws), dtype=torch.uint8, device=device, generator=gen
    )

    ref = _reference(
        a1_scale_token_u8, rows_to_tokens, E, max_m, warp_tile_m, scale_k_per_tile
    )

    out_shape = (E, max_m // wmma_rep, Ws * wmma_rep)
    if init == "none":
        out = None
    elif init == "garbage":
        out = torch.full(out_shape, 0xAA, dtype=torch.uint8, device=device)
    else:
        out = torch.zeros(out_shape, dtype=torch.uint8, device=device)

    got = flydsl_moe_scatter_preshuffle_scale(
        a1_scale_token_u8,
        rows_to_tokens,
        E,
        max_m,
        wmma_rep=wmma_rep,
        scale_k_per_tile=scale_k_per_tile,
        grouped_a1_scale=out,
    )

    assert got.shape == ref.shape, (got.shape, ref.shape)
    ndiff = (got != ref).sum().item()
    assert ndiff == 0, (
        f"mismatch {ndiff}/{ref.numel()} bytes for "
        f"model_dim={model_dim} warp_tile_m={warp_tile_m} skpt={scale_k_per_tile} "
        f"tok={token_num} topk={topk} E={E} pat={pattern} init={init}"
    )


# --------------------------------------------------------------------------- #
# generated matrix (>100 cases sweeping the full geometry + routing space)
# --------------------------------------------------------------------------- #
_MODEL_DIMS = [256, 512, 1024, 2048, 4096, 7168, 8192]
_WARP_TILES = [16, 32, 64, 128]           # wmma_rep = 1, 2, 4, 8
_SKPTS = [4, 8, 16]                        # k_wmma_steps = 1, 2, 4

# (token_num, topk, E): single token, empty experts, full experts, primes, many tiles.
_SHAPES = [
    (1, 1, 1),
    (1, 1, 8),
    (1, 4, 8),
    (2, 1, 1),
    (3, 2, 8),
    (7, 3, 5),
    (16, 2, 8),
    (31, 4, 16),
    (64, 1, 4),
    (64, 8, 32),
    (128, 6, 64),
    (200, 4, 32),
    (257, 8, 128),   # prime-ish token_num, many tiles
]
_PATTERNS = ["randperm", "single", "skew"]


def _gen_geometries():
    geos = []
    for md in _MODEL_DIMS:
        Ws = md // 32
        for wt in _WARP_TILES:
            for sk in _SKPTS:
                if Ws % sk != 0:        # scale_k_per_tile must divide Ws
                    continue
                geos.append((md, wt, sk))
    return geos


def _gen_configs():
    geos = _gen_geometries()
    cfgs = []
    # Two (shape, pattern) draws per geometry -> broad coverage, compile reuse.
    for gi, (md, wt, sk) in enumerate(geos):
        for off in (0, 6):
            shp = _SHAPES[(gi + off) % len(_SHAPES)]
            pat = _PATTERNS[(gi + off) % len(_PATTERNS)]
            cfgs.append((md, wt, sk, *shp, pat))
    # Plus every (shape x pattern) at least once on a fixed mid geometry.
    for shp in _SHAPES:
        for pat in _PATTERNS:
            cfgs.append((512, 64, 8, *shp, pat))
    return cfgs


_CONFIGS = _gen_configs()


def test_matrix_has_enough_cases():
    assert len(_CONFIGS) >= 100, f"only {len(_CONFIGS)} cases"


@pytest.mark.parametrize("cfg", _CONFIGS, ids=lambda c: "_".join(map(str, c)))
def test_scatter_preshuffle_scale_matrix(cfg):
    if not torch.cuda.is_available():
        pytest.skip("needs GPU")
    md, wt, sk, tok, topk, E, pat = cfg
    seed = (md + wt * 7 + sk * 13 + tok * 17 + topk * 19 + E * 23 + len(pat)) & 0xFFFF
    _run_case(md, wt, sk, tok, topk, E, pat, init="zero", seed=seed)


# --------------------------------------------------------------------------- #
# explicit corner cases
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("init", ["zero", "garbage", "none"])
@pytest.mark.parametrize(
    "cfg",
    [
        (256, 128, 8, 1, 1, 1, "single"),    # wmma_rep=8, single row, one expert
        (512, 64, 8, 1, 1, 8, "randperm"),   # one token, mostly-empty experts
        (7168, 64, 8, 200, 8, 64, "skew"),   # production-ish odd Ws=224
        (1024, 32, 16, 257, 4, 32, "randperm"),  # k_wmma_steps=4, many tiles
        (256, 16, 4, 128, 1, 4, "single"),   # wmma_rep=1 (scalar store), full expert
    ],
)
def test_init_variants(cfg, init):
    if not torch.cuda.is_available():
        pytest.skip("needs GPU")
    md, wt, sk, tok, topk, E, pat = cfg
    _run_case(md, wt, sk, tok, topk, E, pat, init=init, seed=7)


@pytest.mark.parametrize("cfg", _CONFIGS, ids=lambda c: "_".join(map(str, c)))
def test_preshuffle_only_matrix(cfg):
    """Gather-less preshuffle (stage2 path): source already grouped row-major.

    Compared byte-exact against the torch ``_grouped_a8w4_preshuffle_e8m0_scale``.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs GPU")
    md, wt, sk, tok, _topk, E, _pat = cfg
    device = "cuda"
    Ws = md // 32
    wmma_rep = wt // 16
    max_m = _padded_max_m(tok, wt)
    seed = (md + wt * 7 + sk * 13 + tok * 17 + E * 23) & 0xFFFF

    gen = torch.Generator(device=device).manual_seed(seed)
    scale_raw = torch.randint(
        0, 256, (E, max_m, Ws), dtype=torch.uint8, device=device, generator=gen
    )

    ref = _grouped_a8w4_preshuffle_e8m0_scale(
        scale_raw, warp_tile=wt, scale_k_per_tile=sk
    )
    got = flydsl_moe_preshuffle_scale(
        scale_raw, E, max_m, wmma_rep=wmma_rep, scale_k_per_tile=sk
    )
    ndiff = (got != ref).sum().item()
    assert ndiff == 0, f"preshuffle-only mismatch {ndiff}/{ref.numel()} for {cfg}"


def test_all_padding_block_is_zero():
    """A wholly-empty expert must yield an all-zero output slice."""
    if not torch.cuda.is_available():
        pytest.skip("needs GPU")
    # E=4 experts, every token -> expert 0 only: experts 1..3 are all padding.
    _run_case(512, 64, 8, 32, 1, 4, "single", init="garbage", seed=3)


if __name__ == "__main__":
    import sys

    print(f"generated {len(_CONFIGS)} matrix cases")
    sys.exit(pytest.main([__file__, "-q"]))
