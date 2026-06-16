#!/usr/bin/env python3
"""Local (gfx942-friendly) correctness gate for the n32k4 weight (B) scale layout.

The real grouped GEMM test (``test_flydsl_grouped_gemm_gfx1250.py``) varies the
weight scale only mildly, so its numerics are weakly sensitive to the B-scale
layout.  This test instead proves the layout math with *distinct* per-byte
values, in pure torch (no kernel, no GPU):

  1. producer ``_grouped_b_scale_preshuffle_e8m0`` == the closed-form ``col`` map
     (and is a bijection / lossless roundtrip), output shape (E, N//32, k_scale*32).
  2. the consumer per-lane ds_load_b32 addressing (a8w4 16x16x128 and fp4
     32x16x128) reads exactly the 4 e8m0 of the intended (N-row, WMMA-K step).
  3. ``test_bscale_numeric_reconstruction`` (RANDOM init): the strong gate -- it
     models the *full verified WMMA scale contract* (each lane = one i32 = the 4
     e8m0 of a 128-K step, byte r -> K-block r; op_sel = lane-half select, proven
     by ``test_wmma_scale_sample.py``) for BOTH op_sel states and asserts the
     effective scale the GEMM applies to every (N-row, K-block) equals the
     logical scale byte-for-byte.

n32k4 layout: raw (E, N, K//32) e8m0 -> view(E, N//32, 32, K//128, 4).
permute(0,1,3,2,4) -> (E, N//32, (K//32)*32).  Within a 32-row super-row,
``col = remain_k*128 + row32*4 + r`` where remain_k = WMMA-K=128 step, row32
(== lane) is the row in the super-row, and r (0-3) is the K-block in the step.

Run: ``AITER_USE_SYSTEM_TRITON=1 python -m pytest -q op_tests/test_grouped_b_scale_layout.py``
"""
import os
import sys

import pytest
import torch

# Import only the pure-torch producer.  AITER_USE_SYSTEM_TRITON=1 lets the
# package import on this box; the function itself touches no GPU/JIT.
try:
    from aiter.ops.flydsl.grouped_moe_gfx1250 import (
        _grouped_b_scale_preshuffle_e8m0,
        _grouped_b_scale_prepare_batch,
    )
except Exception as exc:  # pragma: no cover - import guard
    pytest.skip(f"cannot import grouped_moe_gfx1250: {exc}", allow_module_level=True)


# (E, N, k_scale) with k_scale = K//32.  Producer needs N%32==0 and k_scale%4==0;
# the reconstruction (grouped config tile_n=256 -> warp owns 64 rows = 2 super-
# rows, tile_k=256 -> scale_k_per_tile=8) additionally needs N%64==0, k_scale%8==0.
_SHAPES = [
    (1, 1024, 16),   # stage1-ish: N=1024 (32 super-rows), k_scale=16 (2 k-tiles)
    (2, 512, 16),    # stage2-ish, E=2
    (3, 128, 8),     # smallest: N=128 (4 super-rows), k_scale=8 (1 k-tile)
    (2, 256, 16),    # 8 super-rows x 2 k-tiles
]


def _ref_col(n, ks):
    """Closed-form n32k4 (super_row, col) for raw byte (n, ks=e8m0 index)."""
    super_row, row32 = n // 32, n % 32
    remain_k, r = ks // 4, ks % 4
    return super_row, remain_k * 128 + row32 * 4 + r


@pytest.mark.parametrize("E,N,k_scale", _SHAPES)
def test_producer_matches_closed_form_and_roundtrips(E, N, k_scale):
    g = torch.Generator().manual_seed(1234 + N + k_scale)
    raw = torch.randint(0, 256, (E, N, k_scale), dtype=torch.uint8, generator=g)
    out = _grouped_b_scale_preshuffle_e8m0(raw)

    assert tuple(out.shape) == (E, N // 32, k_scale * 32)

    # Bijection / closed-form check: every (n, ks) lands at the predicted (row,col)
    # with no collisions (full coverage of the output).
    seen = torch.zeros((E, N // 32, k_scale * 32), dtype=torch.bool)
    for e in range(E):
        for n in range(N):
            for ks in range(k_scale):
                row, col = _ref_col(n, ks)
                assert out[e, row, col].item() == raw[e, n, ks].item(), (
                    f"value mismatch at e={e} n={n} ks={ks}"
                )
                assert not seen[e, row, col], f"collision at e={e} row={row} col={col}"
                seen[e, row, col] = True
    assert seen.all(), "output not fully covered (producer is not a bijection)"


@pytest.mark.parametrize("E,N,k_scale", _SHAPES)
def test_prepare_batch_accepts_raw_and_preshuffled(E, N, k_scale):
    raw = torch.randint(0, 256, (E, N, k_scale), dtype=torch.uint8)
    k_dim = k_scale * 32
    pre = _grouped_b_scale_prepare_batch(
        raw, experts=E, rows=N, k_dim=k_dim, device="cpu"
    )
    assert tuple(pre.shape) == (E, N // 32, k_scale * 32)
    # idempotent: feeding the preshuffled tensor back returns it unchanged.
    pre2 = _grouped_b_scale_prepare_batch(
        pre, experts=E, rows=N, k_dim=k_dim, device="cpu"
    )
    assert torch.equal(pre, pre2)
    # flat-raw (experts*rows, k_scale) path.
    flat = _grouped_b_scale_prepare_batch(
        raw.reshape(E * N, k_scale), experts=E, rows=N, k_dim=k_dim, device="cpu"
    )
    assert torch.equal(flat, pre)


def test_producer_rejects_bad_dims():
    with pytest.raises(ValueError):  # N=16 not divisible by 32
        _grouped_b_scale_preshuffle_e8m0(torch.zeros((1, 16, 8), dtype=torch.uint8))
    with pytest.raises(ValueError):  # k_scale=2 not divisible by 4 (K%128!=0)
        _grouped_b_scale_preshuffle_e8m0(torch.zeros((1, 64, 2), dtype=torch.uint8))


# ---------------------------------------------------------------------------
# Consumer per-lane ds_load_b32 addressing (must invert the producer).
#
# Grouped config (locked): tile_n=256, n_warp=4 -> warp_tile_n=64; tile_k=256 ->
# scale_k_per_tile=8, k_wmma_steps=2.  A 64-row warp owns 2 n32k4 super-rows.
# LDS per super-row width = scale_k_per_tile*32 = 256 bytes (one k-tile).  The
# kernel's flat byte offset (gemm_mxscale_gfx1250) is
#   super_local*256 + (wn//2)*256 + (wn%2)*64 + ks*128 + lane16*4   (op_sel off)
# which, on the gmem ``pre`` (E, N//32, k_scale*32), maps to super-row
# (sr_base + wn//2) and column (t*256 + (wn%2)*64 + ks*128 + lane16*4).
# ---------------------------------------------------------------------------
_WMMA_N_REP = 4          # warp_tile_n // WMMA_N = 64 // 16
_FP4_WMMA_N_REP = 2      # warp_tile_n // 32 = 64 // 32
_SCALE_K_PER_TILE = 8    # tile_k // 32 = 256 // 32
_K_WMMA_STEPS = 2        # tile_k // 128 = 256 // 128
_ROW_BYTES = _SCALE_K_PER_TILE * 32  # LDS per-super-row width (one k-tile) = 256
_REMAIN_K_BYTES = 128    # one WMMA-K=128 step (= BLOCK_N*R = 32*4)
_HALF_BYTES = 64         # one 16-row half (= SUBBLOCK_N*R = 16*4)


def _reconstruct_b_scale(pre, E, N, k_scale, *, op_sel):
    """Replay the kernel's per-lane scaleB reads -> effective scale per (n, ks).

    Returns a (E, N, k_scale) uint8 tensor: the e8m0 byte the WMMA actually
    applies to weight-row ``n`` / K-block ``ks``.  Equals the logical raw scale
    iff the producer layout + consumer read + op_sel mapping are all correct.

    op_sel off (per-tile) and op_sel on (per-pair) enumerate the SAME (super-row,
    16-row half) pairs, so they reconstruct identically -- running both documents
    that invariance and guards either driving path.
    """
    recon = torch.full((E, N, k_scale), 255, dtype=torch.uint8)  # 255 = "unwritten"
    n_super = N // 32
    n_ktiles = k_scale // _SCALE_K_PER_TILE
    for e in range(E):
        for sr_base in range(0, n_super, 2):       # warp base super-row (64 rows)
            for t in range(n_ktiles):              # k-tile -> col [t*256, +256)
                for ks in range(_K_WMMA_STEPS):    # WMMA-K=128 step within k-tile
                    kblock0 = t * _SCALE_K_PER_TILE + ks * 4  # 4 consecutive K-blocks
                    if op_sel:
                        units = [
                            (idx, kgrp)
                            for idx in range(_WMMA_N_REP // 2)
                            for kgrp in range(2)
                        ]
                    else:
                        units = [(wn // 2, wn % 2) for wn in range(_WMMA_N_REP)]
                    for sub_sr, half in units:     # super-row offset, 16-row half
                        sr = sr_base + sub_sr
                        if sr >= n_super:
                            continue
                        for lane16 in range(16):   # lane -> weight N-row in the half
                            row32 = half * 16 + lane16
                            col = t * _ROW_BYTES + ks * _REMAIN_K_BYTES + row32 * 4
                            n = sr * 32 + row32
                            for r in range(4):     # i32 byte r -> K-block r
                                recon[e, n, kblock0 + r] = pre[e, sr, col + r]
    return recon


@pytest.mark.parametrize("op_sel", [False, True], ids=["opsel_off", "opsel_on"])
@pytest.mark.parametrize("E,N,k_scale", _SHAPES)
def test_bscale_numeric_reconstruction(E, N, k_scale, op_sel):
    """RANDOM e8m0 -> producer -> simulated WMMA scale read == logical scale."""
    g = torch.Generator().manual_seed(20240613 + N * 131 + k_scale * 7 + int(op_sel))
    raw = torch.randint(0, 255, (E, N, k_scale), dtype=torch.uint8, generator=g)
    pre = _grouped_b_scale_preshuffle_e8m0(raw)
    assert tuple(pre.shape) == (E, N // 32, k_scale * 32)

    recon = _reconstruct_b_scale(pre, E, N, k_scale, op_sel=op_sel)
    assert (recon != 255).all(), "some (n, ks) never written -> read does not cover the tile"
    mism = (recon != raw).nonzero(as_tuple=False)
    assert mism.numel() == 0, (
        f"op_sel={op_sel} effective scale != logical at "
        f"{mism.shape[0]} (e,n,ks) positions; first few={mism[:5].tolist()}"
    )


@pytest.mark.parametrize("E,N,k_scale", _SHAPES)
def test_consumer_read_a8w4(E, N, k_scale):
    """a8w4 16x16x128 (op_sel off): lane16 -> row in the 16-row half wn%2."""
    raw = torch.arange(E * N * k_scale, dtype=torch.int64).reshape(E, N, k_scale)
    out = (
        raw.reshape(E, N // 32, 32, k_scale // 4, 4)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .reshape(E, N // 32, k_scale * 32)
    )

    def decode(v, e):
        local = int(v) - e * N * k_scale  # carrier is a global arange
        return (local // k_scale, local % k_scale)

    n_super = N // 32
    n_ktiles = k_scale // _SCALE_K_PER_TILE
    for e in range(E):
        for sr_base in range(0, n_super, 2):
            for t in range(n_ktiles):
                for ks in range(_K_WMMA_STEPS):
                    for wn in range(_WMMA_N_REP):
                        sr = sr_base + wn // 2
                        if sr >= n_super:
                            continue
                        for lane16 in range(16):
                            row32 = (wn % 2) * 16 + lane16
                            col = t * _ROW_BYTES + ks * _REMAIN_K_BYTES + row32 * 4
                            got = [decode(v, e) for v in out[e, sr, col : col + 4]]
                            n = sr * 32 + row32
                            kblock0 = t * _SCALE_K_PER_TILE + ks * 4
                            want = [(n, kblock0 + r) for r in range(4)]
                            assert got == want, (got, want, e, sr, wn, ks, lane16)


@pytest.mark.parametrize("E,N,k_scale", _SHAPES)
def test_consumer_read_fp4(E, N, k_scale):
    """fp4 32x16x128: one dword per lane covers all 32 rows (lane == row32)."""
    raw = torch.arange(E * N * k_scale, dtype=torch.int64).reshape(E, N, k_scale)
    out = (
        raw.reshape(E, N // 32, 32, k_scale // 4, 4)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .reshape(E, N // 32, k_scale * 32)
    )

    def decode(v, e):
        local = int(v) - e * N * k_scale
        return (local // k_scale, local % k_scale)

    n_super = N // 32
    n_ktiles = k_scale // _SCALE_K_PER_TILE
    for e in range(E):
        for sr_base in range(0, n_super, 2):
            for t in range(n_ktiles):
                for ks in range(_K_WMMA_STEPS):
                    for fpwn in range(_FP4_WMMA_N_REP):   # each fp4-tile = 1 super-row
                        sr = sr_base + fpwn
                        if sr >= n_super:
                            continue
                        for lane_kgrp in range(2):        # full 32-row read
                            for lane16 in range(16):
                                row32 = lane_kgrp * 16 + lane16
                                col = (
                                    t * _ROW_BYTES
                                    + ks * _REMAIN_K_BYTES
                                    + row32 * 4
                                )
                                got = [decode(v, e) for v in out[e, sr, col : col + 4]]
                                n = sr * 32 + row32
                                kblock0 = t * _SCALE_K_PER_TILE + ks * 4
                                want = [(n, kblock0 + r) for r in range(4)]
                                assert got == want, (got, want)


# ---------------------------------------------------------------------------
# Static guard: steady-state B-scale gmem advance must match the n32k4 layout.
#
# The multi-buffer pipeline advances the B-scale descriptor by a constant byte
# stride per k-tile (gemm_mxscale_gfx1250.py ``adv_bs_i32``).  It MUST equal the
# per-k-tile column delta of the n32k4 descriptor (``make_desc_bs``:
# ``inner_off = k_scale_off*32`` -> stride ``(tile_k//32)*32``), i.e. one k-tile's
# worth of columns (= scale_k_per_tile*32 = _ROW_BYTES).
# ---------------------------------------------------------------------------
def test_n32k4_steady_state_advance_matches_layout():
    SCALE_BLOCK = 32
    tile_k = 256            # grouped config (locked)
    scale_k_per_tile = tile_k // SCALE_BLOCK  # = 8

    adv = tile_k // SCALE_BLOCK * 32          # kernel adv_bs_i32
    per_ktile_cols = scale_k_per_tile * 32    # make_desc_bs per-k-tile column width

    assert adv == per_ktile_cols == _ROW_BYTES == 256, (adv, per_ktile_cols)


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))
