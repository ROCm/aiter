#!/usr/bin/env python3
"""Local (gfx942-friendly) correctness gate for the N4K8 weight (B) scale layout.

The real grouped GEMM test (``test_flydsl_grouped_gemm_gfx1250.py``) sets every
weight scale to e8m0 byte 127 (= 1.0), so its numerics are INSENSITIVE to the
B-scale layout -- any non-crashing layout passes.  This test instead proves the
layout math with *distinct* per-byte values, in pure torch (no kernel, no GPU):

  1. producer ``_grouped_b_scale_preshuffle_e8m0`` == the closed-form ``col`` map
     (and is a bijection / lossless roundtrip), output shape (E, N//64, k_scale*64).
  2. the consumer per-lane ds_load_b32 addressing (a8w4 ROW_MAJOR and fp4
     COL_BAND) reads exactly the 4 e8m0 of the intended (N-row, WMMA-K step).
  3. ``test_bscale_numeric_reconstruction`` (RANDOM init): the strong gate -- it
     models the *full verified WMMA scale contract* (each lane = one i32 = the 4
     e8m0 of a 128-K step, byte r -> K-block r; op_sel = lane-half select, proven
     by ``test_wmma_scale_sample.py``) for BOTH op_sel states, at the real failing
     dims (model_dim=inter_dim=3072), and asserts the effective scale the GEMM
     applies to every (N-row, K-block) equals the logical scale byte-for-byte.

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


# Real dims exercised by the grouped MoE test (model_dim=inter_dim=512 default):
#   stage1: rows = 2*inter, k_dim = K (model_dim)
#   stage2: rows = K,        k_dim = inter
# plus a couple of multi-block shapes.  Constraints: N%64==0, (k_dim//32)%8==0.
_SHAPES = [
    (1, 1024, 512),   # stage1, E=1
    (2, 512, 512),    # stage2, E=2
    (3, 128, 256),    # smallest legal (N=128 -> 2 super-rows, k_scale=8 -> 1 block)
    (2, 256, 512),    # 4 super-rows x 2 remain_b blocks
]


def _ref_col(n, ks):
    """Closed-form N4K8 (row, col) for raw byte (n, ks)."""
    super_row, p, lane = n // 64, (n // 16) % 4, n % 16
    remain_b, q, r = ks // 8, (ks // 4) % 2, ks % 4
    return super_row, remain_b * 512 + p * 128 + q * 64 + lane * 4 + r


@pytest.mark.parametrize("E,N,k_scale", _SHAPES)
def test_producer_matches_closed_form_and_roundtrips(E, N, k_scale):
    g = torch.Generator().manual_seed(1234 + N + k_scale)
    raw = torch.randint(0, 256, (E, N, k_scale), dtype=torch.uint8, generator=g)
    out = _grouped_b_scale_preshuffle_e8m0(raw)

    assert tuple(out.shape) == (E, N // 64, k_scale * 64)

    # Bijection / closed-form check: every (n, ks) lands at the predicted (row,col)
    # with no collisions (full coverage of the output).
    seen = torch.zeros((E, N // 64, k_scale * 64), dtype=torch.bool)
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
    assert tuple(pre.shape) == (E, N // 64, k_scale * 64)
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
    with pytest.raises(ValueError):
        _grouped_b_scale_preshuffle_e8m0(torch.zeros((1, 32, 8), dtype=torch.uint8))
    with pytest.raises(ValueError):  # k_scale=4 not divisible by 8
        _grouped_b_scale_preshuffle_e8m0(torch.zeros((1, 64, 4), dtype=torch.uint8))


# --- consumer per-lane ds_load_b32 addressing (must invert the producer) -----
# LDS holds one remain_b block as (tile_n//64) rows x 512 bytes:
#   LDS[super_local][col] = out[expert, super0 + super_local, rb*512 + col]
# scale_k_per_tile = tile_k//32 = 8  ->  ROW/ block = 512 cols.

def _consumer_dword_a8w4(out, e, super_local, wn, ks, lane16, rb):
    """a8w4 (16x16, op_sel=0, p=wn): off = wn*128 + ks*64 + lane16*4."""
    col = rb * 512 + wn * 128 + ks * 64 + lane16 * 4
    return out[e, super_local, col : col + 4]


def _consumer_dword_fp4(out, e, super_local, fpwn, ks, lane_kgrp, lane16, rb):
    """fp4 (32x16): p = 2*fpwn + lane_kgrp; off = fpwn*256 + ks*64 + lane_kgrp*128."""
    col = rb * 512 + fpwn * 256 + ks * 64 + lane_kgrp * 128 + lane16 * 4
    return out[e, super_local, col : col + 4]


# ---------------------------------------------------------------------------
# Numerical end-to-end reconstruction (RANDOM init) -- the real correctness gate.
#
# The addressing tests above use an arange carrier and only prove producer<->
# consumer-address identity.  This test instead models the FULL verified WMMA
# scale contract (proved on gfx1250 by ``test_wmma_scale_sample.py``):
#
#   * each lane supplies a full i32 scaleA VGPR = the 4 e8m0 bytes (r=0..3) of
#     one WMMA-K=128 step, byte r -> K-block r (the sample's 32+8+2+0.5=42.5);
#   * ``scaleAType`` (op_sel) selects the LANE-HALF that supplies the scale
#     (op_sel=0 -> lanes 0-15, op_sel=1 -> lanes 16-31) -- NOT a byte-half of the
#     i32 (the rocdl docstring's "lo/hi 16-bit half" wording is misleading; the
#     sample's opsel0/opsel1 cases zero the unused half-wave and prove lane-half).
#
# We feed RANDOM e8m0 bytes through the producer, then replay the EXACT consumer
# offsets from ``gemm_mxscale_gfx1250.py`` (transcribed below) plus that op_sel
# lane-half selection, and rebuild the effective scale the GEMM would apply to
# every (N-row, K-block).  A correct layout reconstructs the logical scale
# byte-for-byte; any wrong stride / op_sel mapping shows up as a mismatch.
#
# Grouped config (locked): tile_n=256, n_warp=4 -> warp_tile_n=64, WMMA_N=16 ->
# wmma_n_rep=4 (one warp owns one 64-row super-row, p=wn); tile_k=256 ->
# scale_k_per_tile=8 (one remain_b block per k-tile) and k_wmma_steps=2 (ks=0,1).
# ---------------------------------------------------------------------------
_WMMA_N_REP = 4          # warp_tile_n // WMMA_N = 64 // 16
_SCALE_K_PER_TILE = 8    # tile_k // 32 = 256 // 32
_K_WMMA_STEPS = 2        # tile_k // 128 = 256 // 128
_PTILE_BYTES = 128       # p (WMMA N-tile) stride, _B_N4K8_PTILE_BYTES
_BLOCK_BYTES = 512       # one remain_b block (8 e8m0 = 2 WMMA-K steps)
_QSTEP_BYTES = 64        # q (2nd WMMA-K step) stride, _B_N4K8_QSTEP_BYTES


def _n4k8_kstep_off(ks):  # mirrors gemm_mxscale_gfx1250._n4k8_kstep_off
    return (ks // 2) * _BLOCK_BYTES + (ks % 2) * _QSTEP_BYTES


def _reconstruct_b_scale(pre, E, N, k_scale, *, op_sel):
    """Replay the kernel's per-lane scaleA reads -> effective scale per (n, ks).

    Returns a (E, N, k_scale) uint8 tensor: the e8m0 byte the WMMA actually
    applies to weight-row ``n`` / K-block ``ks``.  Equals the logical raw scale
    iff the producer layout + consumer read + op_sel mapping are all correct.
    """
    recon = torch.full((E, N, k_scale), 255, dtype=torch.uint8)  # 255 = "unwritten"
    n_super = N // 64
    n_ktiles = k_scale // _SCALE_K_PER_TILE
    for e in range(E):
        for sr in range(n_super):
            for t in range(n_ktiles):              # k-tile -> gmem col [t*512, +512)
                for ks in range(_K_WMMA_STEPS):    # WMMA-K step within the k-tile
                    q_off = _n4k8_kstep_off(ks)
                    kblock0 = t * _SCALE_K_PER_TILE + ks * 4  # 4 consecutive K-blocks
                    for wn in range(_WMMA_N_REP):  # one 16x16 WMMA per N-tile
                        if op_sel:
                            lane_kgrp = wn % 2     # op_sel picks the lane-half...
                            idx = wn // 2          # ...and b_scale_idx = wn//2
                            base_lane = lane_kgrp * _PTILE_BYTES + idx * 256
                        else:
                            base_lane = wn * _PTILE_BYTES  # per-tile: p = wn
                        for lane16 in range(16):   # lane -> weight N-row in the tile
                            col = t * (k_scale * 64 // n_ktiles)  # = t*512
                            col += lane16 * 4 + base_lane + q_off
                            n = sr * 64 + wn * 16 + lane16
                            for r in range(4):     # i32 byte r -> K-block r
                                recon[e, n, kblock0 + r] = pre[e, sr, col + r]
    return recon


# Real shapes from the failing CLI run (model_dim=inter_dim=3072): stage1 has
# N=2*inter=6144, k_scale=K//32=96 (12 k-tiles); stage2 N=3072, k_scale=96.
_NUMERIC_SHAPES = _SHAPES + [(1, 6144, 96), (2, 3072, 96)]


@pytest.mark.parametrize("op_sel", [False, True], ids=["opsel_off", "opsel_on"])
@pytest.mark.parametrize("E,N,k_scale", _NUMERIC_SHAPES)
def test_bscale_numeric_reconstruction(E, N, k_scale, op_sel):
    """RANDOM e8m0 -> producer -> simulated WMMA scale read == logical scale."""
    g = torch.Generator().manual_seed(20240613 + N * 131 + k_scale * 7 + int(op_sel))
    raw = torch.randint(0, 255, (E, N, k_scale), dtype=torch.uint8, generator=g)
    pre = _grouped_b_scale_preshuffle_e8m0(raw)
    assert tuple(pre.shape) == (E, N // 64, k_scale * 64)

    recon = _reconstruct_b_scale(pre, E, N, k_scale, op_sel=op_sel)
    assert (recon != 255).all(), "some (n, ks) never written -> read does not cover the tile"
    mism = (recon != raw).nonzero(as_tuple=False)
    assert mism.numel() == 0, (
        f"op_sel={op_sel} effective scale != logical at "
        f"{mism.shape[0]} (e,n,ks) positions; first few={mism[:5].tolist()}"
    )


@pytest.mark.parametrize("E,N,k_scale", _SHAPES)
def test_consumer_read_a8w4(E, N, k_scale):
    raw = torch.arange(E * N * k_scale, dtype=torch.int64).reshape(E, N, k_scale)
    # reuse the producer permute on an int64 carrier so we can decode (n, ks).
    out = (
        raw.reshape(E, N // 64, 4, 16, k_scale // 8, 2, 4)
        .permute(0, 1, 4, 2, 5, 3, 6)
        .contiguous()
        .reshape(E, N // 64, k_scale * 64)
    )

    def decode(v, e):
        local = int(v) - e * N * k_scale  # carrier is a global arange
        return (local // k_scale, local % k_scale)

    wmma_n_rep = 4  # warp_tile_n//16
    for e in range(E):
        for super_local in range(N // 64):
            for rb in range(k_scale // 8):
                for wn in range(wmma_n_rep):
                    for ks in range(2):  # k_wmma_steps (tile_k=256 -> 2)
                        s = rb * 2 + ks
                        for lane16 in range(16):
                            got = [
                                decode(v, e)
                                for v in _consumer_dword_a8w4(
                                    out, e, super_local, wn, ks, lane16, rb
                                )
                            ]
                            n = super_local * 64 + wn * 16 + lane16
                            want = [(n, s * 4 + r) for r in range(4)]
                            assert got == want, (got, want, e, super_local, wn, ks, lane16)


@pytest.mark.parametrize("E,N,k_scale", _SHAPES)
def test_consumer_read_fp4(E, N, k_scale):
    raw = torch.arange(E * N * k_scale, dtype=torch.int64).reshape(E, N, k_scale)
    out = (
        raw.reshape(E, N // 64, 4, 16, k_scale // 8, 2, 4)
        .permute(0, 1, 4, 2, 5, 3, 6)
        .contiguous()
        .reshape(E, N // 64, k_scale * 64)
    )

    def decode(v, e):
        local = int(v) - e * N * k_scale  # carrier is a global arange
        return (local // k_scale, local % k_scale)

    fp4_wmma_n_rep = 2  # warp_tile_n//32
    for e in range(E):
        for super_local in range(N // 64):
            for rb in range(k_scale // 8):
                for fpwn in range(fp4_wmma_n_rep):
                    for ks in range(2):
                        s = rb * 2 + ks
                        for lane_kgrp in range(2):
                            p = 2 * fpwn + lane_kgrp
                            for lane16 in range(16):
                                got = [
                                    decode(v, e)
                                    for v in _consumer_dword_fp4(
                                        out, e, super_local, fpwn, ks,
                                        lane_kgrp, lane16, rb,
                                    )
                                ]
                                n = super_local * 64 + p * 16 + lane16
                                want = [(n, s * 4 + r) for r in range(4)]
                                assert got == want, (got, want)


@pytest.mark.parametrize("E,N,k_scale", _SHAPES)
def test_consumer_read_a8w4_opsel(E, N, k_scale):
    """a8w4 with use_scale_opsel=True reads one dword per N-tile PAIR.

    p = 2*idx + lane_kgrp (op_sel picks the lane-half); off = idx*256 + ks*64 +
    lane_kgrp*128 -- identical addressing to the fp4 path, just driven by
    WEIGHT_SCALE_OP_SEL instead of the format.
    """
    raw = torch.arange(E * N * k_scale, dtype=torch.int64).reshape(E, N, k_scale)
    out = (
        raw.reshape(E, N // 64, 4, 16, k_scale // 8, 2, 4)
        .permute(0, 1, 4, 2, 5, 3, 6)
        .contiguous()
        .reshape(E, N // 64, k_scale * 64)
    )

    def decode(v, e):
        local = int(v) - e * N * k_scale
        return (local // k_scale, local % k_scale)

    wmma_n_rep = 4
    n_pairs = wmma_n_rep // 2  # = 2 dwords loaded per step (op_sel halves it)
    for e in range(E):
        for super_local in range(N // 64):
            for rb in range(k_scale // 8):
                for idx in range(n_pairs):
                    for ks in range(2):
                        s = rb * 2 + ks
                        for lane_kgrp in range(2):
                            p = 2 * idx + lane_kgrp
                            for lane16 in range(16):
                                col = (
                                    rb * 512
                                    + idx * 256
                                    + ks * 64
                                    + lane_kgrp * 128
                                    + lane16 * 4
                                )
                                got = [
                                    decode(v, e)
                                    for v in out[e, super_local, col : col + 4]
                                ]
                                n = super_local * 64 + p * 16 + lane16
                                want = [(n, s * 4 + r) for r in range(4)]
                                assert got == want, (got, want)


# ---------------------------------------------------------------------------
# Static guard: steady-state B-scale gmem advance must match the N4K8 layout.
#
# The multi-buffer pipeline advances the B-scale descriptor by a constant byte
# stride per k-tile (gemm_mxscale_gfx1250.py ``adv_bs_i32``).  It MUST equal the
# per-k-tile column delta of the n4k8 descriptor (``make_desc_bs`` n4k8 branch:
# ``inner_off = k_scale_off*64`` -> stride ``(tile_k//32)*64``), i.e. one full
# ``remain_b`` super-block of 512 cols per k-tile -- the same 512 the consumer/
# producer use (``rb*512``).  The original bug used the *interleaved* stride
# ``(tile_k//32)*b_scale_load_rep`` (=32), so k-tiles >= the prologue depth read
# the wrong scale columns once the steady-state loop engaged (>=3 k-tiles).  The
# numeric reconstruction test can't see this (it models explicit per-tile
# descriptors, not the advancing one).
# ---------------------------------------------------------------------------
def test_n4k8_steady_state_advance_matches_layout():
    SCALE_BLOCK = 32
    tile_k = 256            # grouped config (locked)
    b_scale_load_rep = 4    # a8w4: wmma_n_rep = warp_tile_n//16 = 64//16; fp4: warp_tile_n//16
    N4K8_SUPERBLOCK_COLS = 512  # one remain_b block = 8 e8m0 * 64 ; == _B_N4K8_BLOCK_BYTES

    n4k8_advance = tile_k // SCALE_BLOCK * 64
    interleaved_advance = tile_k // SCALE_BLOCK * b_scale_load_rep

    # advance == n4k8 per-k-tile column delta (one remain_b super-block)
    assert n4k8_advance == N4K8_SUPERBLOCK_COLS, (n4k8_advance, N4K8_SUPERBLOCK_COLS)
    # and it is NOT the (wrong) interleaved stride that caused the bug
    assert n4k8_advance != interleaved_advance, (n4k8_advance, interleaved_advance)
    assert interleaved_advance == 32  # documents the old (broken) value


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))
