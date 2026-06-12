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


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))
