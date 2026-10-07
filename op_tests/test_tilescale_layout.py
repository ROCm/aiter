# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tilescale layout (aiter.ops.tilescale): scale-map golden hashes, code and scale round trips, reference
quantize / dequantize. Runs on CPU; uses the GPU when one is present."""

import hashlib

import pytest
import torch

import aiter.ops.tilescale as TS

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# sha256 (first 16 hex digits) of the int64 scale map of 512 rows, frozen from the FlyDSL MXFP4/MXFP6 GEMMs'
# packed-scale layout (256-wide tile). Covers both parities of K/256 and both role-B interleaves.
GOLDEN = {
    (3072, False, 0): "c5b46b5238e74456",
    (3072, True, 0): "4f600c1968e8f7b4",
    (3072, True, 4): "75161d3c22b54d82",
    (9216, False, 0): "0a84d4276f8b6373",
    (9216, True, 0): "49d7fdc2e2ef4e0a",
    (9216, True, 4): "5b6c8558ebd03d27",
    (12288, False, 0): "d679dbf5472dc532",
    (12288, True, 0): "d5cab66698fe6ca8",
    (12288, True, 4): "187efed496f428f2",
}


@pytest.mark.parametrize("key", sorted(GOLDEN))
def test_scale_map_golden(key):
    K, is_b, ilv = key
    m = TS._scale_map(512, K, is_b, ilv, "cpu").to(torch.int64).numpy().tobytes()
    assert hashlib.sha256(m).hexdigest()[:16] == GOLDEN[key]


@pytest.mark.parametrize("K", [256, 768, 1024, 3072, 9216])
@pytest.mark.parametrize("is_b,ilv", [(False, 0), (True, 0), (True, 4)])
def test_scale_map_is_permutation(K, is_b, ilv):
    m = TS._scale_map(512, K, is_b, ilv, DEV).reshape(-1)
    assert torch.equal(torch.sort(m).values, torch.arange(m.numel(), device=DEV))


def test_ilv_role_a_rejected():
    with pytest.raises(ValueError):
        TS.ts_scale_byte(torch.tensor(0), torch.tensor(0), is_b=False, K=256, ilv=4)


@pytest.mark.parametrize("rows", [256, 300, 512])
@pytest.mark.parametrize("K", [256, 1536, 3072])
@pytest.mark.parametrize("is_b,ilv", [(False, 0), (True, 0), (True, 4)])
def test_scale_round_trip_and_padding(rows, K, is_b, ilv):
    g = torch.Generator(device=DEV).manual_seed(rows + K)
    s = torch.randint(0, 256, (rows, K // 32), dtype=torch.uint8, device=DEV, generator=g)
    slab = TS.pack_scales_ref(s, is_b=is_b, ilv=ilv)
    assert slab.numel() == TS.ts_scale_bytes(rows, K)
    assert torch.equal(TS.unpack_scales_ref(slab, rows, K, is_b=is_b, ilv=ilv), s)
    rp = -(-rows // 256) * 256
    if rp > rows:  # padded rows hold scale byte 127
        full = TS.unpack_scales_ref(slab, rp, K, is_b=is_b, ilv=ilv)
        assert bool((full[rows:] == 127).all())


@pytest.mark.parametrize("layout", ["row", "k128"])
@pytest.mark.parametrize("rows,K", [(256, 256), (300, 1024), (512, 3072)])
def test_fp4_codes_round_trip(layout, rows, K):
    g = torch.Generator(device=DEV).manual_seed(rows * K)
    c = torch.randint(0, 16, (rows, K), dtype=torch.uint8, device=DEV, generator=g)
    buf = TS.pack_fp4_codes_ref(c, layout)
    assert buf.numel() == TS.ts_code_bytes(rows, K, TS.FP4)
    assert torch.equal(TS.unpack_fp4_codes_ref(buf, rows, K, layout), c)


def test_fp4_k128_is_blocked_row():
    """k128 = the row layout with every 16-row x 128-K block made one contiguous KiB."""
    c = torch.randint(0, 16, (256, 512), dtype=torch.uint8, device=DEV)
    row = TS.pack_fp4_codes_ref(c, "row")
    k128 = TS.pack_fp4_codes_ref(c, "k128").reshape(-1)
    for rb, kb in ((0, 0), (3, 2), (15, 3)):
        assert torch.equal(k128[(rb * 4 + kb) * 1024 : (rb * 4 + kb + 1) * 1024].view(16, 64),
                           row[rb * 16 : rb * 16 + 16, kb * 64 : kb * 64 + 64])


@pytest.mark.parametrize("rows,K", [(256, 256), (300, 1024), (512, 3072)])
def test_fp6_codes_round_trip(rows, K):
    g = torch.Generator(device=DEV).manual_seed(rows + 7 * K)
    c = torch.randint(0, 64, (rows, K), dtype=torch.uint8, device=DEV, generator=g)
    buf = TS.pack_fp6_codes_ref(c)
    assert buf.numel() == TS.ts_code_bytes(rows, K, TS.FP6)
    assert torch.equal(TS.unpack_fp6_codes_ref(buf, rows, K), c)
    c0, c1 = TS.fp6_planes(buf, rows, K)
    assert c0.shape == (-(-rows // 256) * 256, K // 2) and c1.shape == (c0.shape[0], K // 4)


def test_fp6_group_bit_order():
    """Value i of a 32-group at bits 6i..6i+5: group bytes 0-15 open C0's first 16-B chunk, 16-23 open C1's."""
    c = torch.zeros(256, 256, dtype=torch.uint8, device=DEV)
    c[0, 0], c[0, 1], c[0, 31] = 0x3F, 0x01, 0x2A
    buf = TS.pack_fp6_codes_ref(c)
    c0, c1 = TS.fp6_planes(buf, 256, 256)
    assert c0[0, 0].item() == 0x7F and c0[0, 1].item() == 0x00  # 0x3F | 0x01 << 6
    assert c1[0, 7].item() == 0x2A << 2  # value 31 occupies bits 186..191 = byte 23 bits 2..7


@pytest.mark.parametrize("fmt", [TS.FP4, TS.FP6])
def test_quant_dequant_ref(fmt):
    g = torch.Generator(device=DEV).manual_seed(1)
    x = torch.randn(256, 1024, device=DEV, generator=g, dtype=torch.float64)
    codes, scales = TS.quant_mx_ref(x, fmt)
    y = TS.dequant_ref(codes, scales, fmt)
    cos = torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0).item()
    assert cos > (0.99 if fmt == TS.FP4 else 0.999)
    # representable values are fixed points
    codes2, scales2 = TS.quant_mx_ref(y, fmt)
    assert torch.equal(TS.dequant_ref(codes2, scales2, fmt), y)


def test_fp4_values_are_exact_fp6():
    """Every E2M1 value is an E2M3 value: the A6W4 product equals A6W6 on the FP6 re-encoding."""
    c4 = torch.arange(16, device=DEV, dtype=torch.uint8)
    c6 = ((c4 & 8) << 2) | ((c4 & 6) << 2) | ((c4 & 1) << 2)
    assert torch.equal(TS.code_values(c4, TS.FP4), TS.code_values(c6, TS.FP6))


@pytest.mark.parametrize("K", [3072, 2560])
@pytest.mark.parametrize("ilv", [0, 4])
def test_kouter_slab_is_k256_outer(K, ilv):
    """The K256-outer slab is the standard role-B slab [wi, J, 1 KiB] with its two outer axes swapped, and every
    256-aligned K range of it is contiguous."""
    rows = 768
    std = TS._scale_map(rows, K, True, ilv, "cpu").reshape(-1)
    ko = TS._scale_map(rows, K, True, ilv, "cpu", kouter=True).reshape(-1)
    nwi, kk = rows // 128, K // 256
    assert torch.equal(torch.sort(ko).values, torch.arange(ko.numel()))
    swap = (std // 1024 % kk) * nwi + std // 1024 // kk  # J * nwi + wi
    assert torch.equal(ko, swap * 1024 + std % 1024)
    codes = torch.randint(0, 16, (rows, K), dtype=torch.uint8)
    assert torch.equal(TS.unpack_fp4_codes_ref(TS.pack_fp4_codes_ref(codes, "kouter"), rows, K, "kouter"), codes)
