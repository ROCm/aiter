# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sergey Subbotin <ssubbotin@gmail.com>
#
# Regenerates data/q4k_ggml_golden.npz, the fixture that anchors
# q4k_pack_reference to llama.cpp's Q4_K_M layout. Not part of the test run:
# aiter CI consumes the committed .npz and needs no llama.cpp checkout.
#
# Usage:
#   python3 gen_q4k_ggml_golden.py --ggml-lib /path/to/build/bin/libggml-base.so \
#                                  [--llama-cpp /path/to/llama.cpp] \
#                                  [--out data/q4k_ggml_golden.npz]
#
# Run it when refreshing the anchor against a newer llama.cpp. The companion
# test_q4k_vs_ggml.py::test_golden_still_matches_live_ggml (opt-in via
# AITER_GGML_LIB) checks a committed fixture against a live library without
# rewriting it.

import argparse
import ctypes
import subprocess
from pathlib import Path

import numpy as np
from q4k_pack_reference import BLOCK_BYTES, QK_K, pack_block


def build_cases():
    """(name, dall, dmin, sc, m, q) tuples covering the format's corners.

    get_scale_min_k4 uses two bit layouts: sub-blocks 0-3 take the low 6 bits
    of scales[j] / scales[j+4]; sub-blocks 4-7 splice the low nibbles of
    scales[j+4] with the top 2 bits of scales[j-4] / scales[j]. Cases must
    drive both halves and hit 6-bit saturation, where the splice is easiest
    to get wrong.
    """
    rng = np.random.default_rng(20260917)
    cases = []

    def add(name, dall, dmin, sc, m, q):
        cases.append(
            (
                name,
                np.float32(dall),
                np.float32(dmin),
                np.asarray(sc, dtype=np.uint8),
                np.asarray(m, dtype=np.uint8),
                np.asarray(q, dtype=np.uint8),
            )
        )

    z8 = np.zeros(8, dtype=np.uint8)
    f8 = np.full(8, 63, dtype=np.uint8)

    add("zeros", 0.0, 0.0, z8, z8, np.zeros(QK_K, np.uint8))
    add("max_nibble", 0.25, 0.0, f8, z8, np.full(QK_K, 15, np.uint8))
    add("max_scale_max_min", 0.5, 0.05, f8, f8, np.full(QK_K, 15, np.uint8))
    add("scale_saturate_only", 0.125, 0.0, f8, z8, np.zeros(QK_K, np.uint8))
    add("min_saturate_only", 0.0, 0.0625, z8, f8, np.zeros(QK_K, np.uint8))
    add(
        "ramp_scales",
        0.03125,
        0.0078125,
        np.arange(0, 64, 8, dtype=np.uint8),
        np.arange(63, -1, -8, dtype=np.uint8),
        np.tile(np.arange(16, dtype=np.uint8), QK_K // 16),
    )
    add(
        "high_bits_scales",
        0.0625,
        0.0,
        np.full(8, 0x30, np.uint8),
        z8,
        np.full(QK_K, 1, np.uint8),
    )
    add(
        "low_bits_scales",
        0.0625,
        0.0,
        np.full(8, 0x0F, np.uint8),
        z8,
        np.full(QK_K, 1, np.uint8),
    )

    # Nibble interleave: low and high nibble of a qs byte must not swap.
    q_low = np.zeros(QK_K, np.uint8)
    q_low[:32] = 15
    add("nibble_low_only", 0.25, 0.0, np.full(8, 32, np.uint8), z8, q_low)
    q_high = np.zeros(QK_K, np.uint8)
    q_high[32:64] = 15
    add("nibble_high_only", 0.25, 0.0, np.full(8, 32, np.uint8), z8, q_high)

    # fp16 super-block scales: sign and both ends of the exponent range.
    add(
        "negative_d",
        -0.375,
        0.0,
        np.full(8, 20, np.uint8),
        z8,
        np.full(QK_K, 9, np.uint8),
    )
    add(
        "negative_dmin",
        0.25,
        -0.03125,
        np.full(8, 20, np.uint8),
        np.full(8, 20, np.uint8),
        np.full(QK_K, 9, np.uint8),
    )
    add("tiny_d", 6.103515625e-05, 0.0, f8, z8, np.full(QK_K, 15, np.uint8))
    add(
        "large_d",
        64.0,
        8.0,
        np.full(8, 40, np.uint8),
        np.full(8, 10, np.uint8),
        np.full(QK_K, 7, np.uint8),
    )

    for i in range(12):
        add(
            f"random_{i}",
            float(np.float16(rng.uniform(-0.5, 0.5))),
            float(np.float16(rng.uniform(-0.05, 0.05))),
            rng.integers(0, 64, size=8, dtype=np.uint8),
            rng.integers(0, 64, size=8, dtype=np.uint8),
            rng.integers(0, 16, size=QK_K, dtype=np.uint8),
        )
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ggml-lib",
        required=True,
        help="path to libggml-base.so exporting dequantize_row_q4_K",
    )
    ap.add_argument(
        "--llama-cpp",
        default=None,
        help="llama.cpp checkout, recorded as fixture provenance",
    )
    ap.add_argument(
        "--out", default=str(Path(__file__).parent / "data" / "q4k_ggml_golden.npz")
    )
    args = ap.parse_args()

    cases = build_cases()
    n = len(cases)
    names = [c[0] for c in cases]
    dall = np.array([c[1] for c in cases], dtype=np.float32)
    dmin = np.array([c[2] for c in cases], dtype=np.float32)
    sc = np.stack([c[3] for c in cases])
    m = np.stack([c[4] for c in cases])
    q = np.stack([c[5] for c in cases])

    packed = np.frombuffer(
        b"".join(
            pack_block(float(dall[i]), float(dmin[i]), sc[i], m[i], q[i])
            for i in range(n)
        ),
        dtype=np.uint8,
    ).copy()
    assert packed.size == n * BLOCK_BYTES

    lib = ctypes.CDLL(args.ggml_lib)
    fn = lib.dequantize_row_q4_K
    fn.restype = None
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]

    golden = np.zeros(n * QK_K, dtype=np.float32)
    fn(
        packed.ctypes.data_as(ctypes.c_void_p),
        golden.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int64(n * QK_K),
    )

    rev = "unknown"
    if args.llama_cpp:
        res = subprocess.run(
            ["git", "-C", args.llama_cpp, "log", "-1", "--format=%H %cI"],
            capture_output=True,
            check=False,
            text=True,
        )
        if res.returncode == 0:
            rev = res.stdout.strip()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        names=np.array(names),
        dall=dall,
        dmin=dmin,
        sc=sc,
        m=m,
        q=q,
        packed=packed,
        golden=golden.reshape(n, QK_K),
        provenance=np.array(
            f"dequantize_row_q4_K from {Path(args.ggml_lib).name} " f"@ llama.cpp {rev}"
        ),
    )
    print(f"wrote {args.out}: {n} blocks, llama.cpp {rev}")


if __name__ == "__main__":
    main()
