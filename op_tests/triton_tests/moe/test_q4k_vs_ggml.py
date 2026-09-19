# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sergey Subbotin <ssubbotin@gmail.com>
#
# Anchors q4k_pack_reference to llama.cpp's Q4_K_M layout.
#
# test_moe_q4k_streaming.py checks the Triton kernel against
# q4k_pack_reference. That leaves the reference itself unanchored: a layout
# drift in q4k_pack_reference would move kernel and reference together and
# every kernel test would still pass. This module pins the reference to bytes
# produced by llama.cpp's own dequantize_row_q4_K.
#
# data/q4k_ggml_golden.npz holds, for each block, the packed 144 bytes, the
# (dall, dmin, sc, m, q) they were built from, and the fp32 output of
# dequantize_row_q4_K in libggml-base.so. Regenerate it with
# op_tests/triton_tests/moe/gen_q4k_ggml_golden.py on a machine with
# llama.cpp built; the committed file is what CI uses, so no llama.cpp
# dependency is needed to run these tests.

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

from op_tests.triton_tests.moe.q4k_pack_reference import (
    BLOCK_BYTES,
    QK_K,
    dequant_block,
    pack_block,
    unpack_scale_min,
)

GOLDEN_PATH = Path(__file__).parent / "data" / "q4k_ggml_golden.npz"


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN_PATH.exists():
        pytest.fail(f"missing golden fixture: {GOLDEN_PATH}")
    return np.load(GOLDEN_PATH, allow_pickle=False)


def _block_bytes(golden, i):
    return golden["packed"][i * BLOCK_BYTES : (i + 1) * BLOCK_BYTES].tobytes()


def test_golden_fixture_shape(golden):
    n = len(golden["names"])
    assert n >= 20, "fixture should cover the format's corners, not a token sample"
    assert golden["packed"].size == n * BLOCK_BYTES
    assert golden["golden"].shape == (n, QK_K)
    assert str(golden["provenance"]).startswith("dequantize_row_q4_K")


def test_dequant_matches_ggml_bit_exactly(golden):
    """Our dequant must reproduce libggml's fp32 output bit for bit.

    Both compute d*sc*q - dmin*m in fp32 in the same order, so there is no
    reordering slack to absorb: any difference here is a layout bug, not
    rounding.
    """
    names = golden["names"]
    mismatched = []
    for i, name in enumerate(names):
        ours = dequant_block(_block_bytes(golden, i))
        if not np.array_equal(ours, golden["golden"][i]):
            absmax = float(np.max(np.abs(ours - golden["golden"][i])))
            mismatched.append(f"{name} (absmax {absmax:.6g})")
    assert not mismatched, "dequant drifted from llama.cpp: " + ", ".join(mismatched)


def test_pack_block_is_byte_exact(golden):
    """Re-packing the stored parameters must reproduce the stored bytes.

    Catches drift in the 6-bit scale/min encode and the nibble interleave,
    which test_dequant_matches_ggml_bit_exactly alone would miss if the pack
    and unpack sides drifted symmetrically.
    """
    names = golden["names"]
    mismatched = []
    for i, name in enumerate(names):
        again = pack_block(
            float(golden["dall"][i]),
            float(golden["dmin"][i]),
            golden["sc"][i],
            golden["m"][i],
            golden["q"][i],
        )
        if again != _block_bytes(golden, i):
            mismatched.append(str(name))
    assert not mismatched, "pack_block drifted: " + ", ".join(mismatched)


def test_scale_min_roundtrip(golden):
    """unpack_scale_min must invert the encoding for every fixture block."""
    for i, name in enumerate(golden["names"]):
        sc, m = unpack_scale_min(
            np.frombuffer(_block_bytes(golden, i)[4:16], dtype=np.uint8)
        )
        assert np.array_equal(sc, golden["sc"][i]), f"sc mismatch in {name}"
        assert np.array_equal(m, golden["m"][i]), f"m mismatch in {name}"


def test_fixture_covers_both_scale_paths(golden):
    """get_scale_min_k4 splits at sub-block 4; both halves need saturation.

    Guards against someone regenerating the fixture from blocks that only
    exercise the low-6-bit path, which would silently weaken this anchor.
    """
    sc, m = golden["sc"], golden["m"]
    assert (sc[:, :4] >= 48).any() and (sc[:, 4:] >= 48).any()
    assert (m[:, :4] >= 48).any() and (m[:, 4:] >= 48).any()
    assert (golden["q"] == 15).any() and (golden["q"] == 0).any()


@pytest.mark.skipif(
    not os.environ.get("AITER_GGML_LIB"),
    reason="set AITER_GGML_LIB=/path/to/libggml-base.so to re-derive from llama.cpp",
)
def test_golden_still_matches_live_ggml(golden):
    """Re-derive the fixture against a live libggml, when one is available.

    Opt-in: the committed fixture is the anchor CI uses. This is for
    refreshing it against a newer llama.cpp.
    """
    lib = ctypes.CDLL(os.environ["AITER_GGML_LIB"])
    fn = lib.dequantize_row_q4_K
    fn.restype = None
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]

    packed = np.ascontiguousarray(golden["packed"])
    n = len(golden["names"])
    out = np.zeros(n * QK_K, dtype=np.float32)
    fn(
        packed.ctypes.data_as(ctypes.c_void_p),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int64(n * QK_K),
    )
    np.testing.assert_array_equal(out.reshape(n, QK_K), golden["golden"])
