#!/usr/bin/env python3
"""
LDS filter for GEMM kernel tuning on AMD MI355X (gfx950).

Computes LDS-safe block size ranges so that the total LDS usage
does not exceed the 160 KB (163840 bytes) hardware limit.

Usage examples:
    python lds_filter.py --kernel a8w8 --num-stages 2 3 --print-cli
    python lds_filter.py --kernel a16w16 --num-stages 2 --print-cli
"""

import argparse
import sys
from itertools import product

# ── hardware constant ────────────────────────────────────────────────────────
LDS_LIMIT = 163840  # 160 KB

# ── default block-size ranges (matching screen.py) ───────────────────────────
DEFAULT_BM = [4, 8, 16, 32, 64, 128, 256, 512]
DEFAULT_BN = [16, 32, 64, 128, 256]
DEFAULT_BK = [128, 256, 512, 1024]

# ── per-kernel parameters ────────────────────────────────────────────────────
KERNEL_PARAMS = {
    "a16w16": {
        "a_dtype_size": 2,
        "w_dtype_size": 2,
        "scale_fn": lambda bm, bn, bk: 0,
        "bk_values": None,
        "preshuffle": None,
    },
    "a16w16_agnostic": {
        "a_dtype_size": 2,
        "w_dtype_size": 2,
        "scale_fn": lambda bm, bn, bk: 0,
        "bk_values": None,
        "preshuffle": None,
    },
    "a16w16_atomic": {
        "a_dtype_size": 2,
        "w_dtype_size": 2,
        "scale_fn": lambda bm, bn, bk: 0,
        "bk_values": None,
        "preshuffle": None,
    },
    "a16w16_gated": {
        "a_dtype_size": 2,
        "w_dtype_size": 2,
        "scale_fn": lambda bm, bn, bk: 0,
        "bk_values": None,
        "preshuffle": None,
    },
    "a16w8_blockscale": {
        "a_dtype_size": 2,
        "w_dtype_size": 1,
        "scale_fn": lambda bm, bn, bk: (bm * bk // 128 + bn * bk // 128),
        "bk_values": [128],
        "preshuffle": None,
    },
    "a16wfp4": {
        "a_dtype_size": 2,
        "w_dtype_size": 1,
        "scale_fn": lambda bm, bn, bk: bn * bk // 32,
        "bk_values": None,
        "preshuffle": None,
    },
    "a8w8": {
        "a_dtype_size": 1,
        "w_dtype_size": 1,
        "scale_fn": lambda bm, bn, bk: 0,
        "bk_values": None,
        "preshuffle": None,
    },
    "a8w8_blockscale": {
        "a_dtype_size": 1,
        "w_dtype_size": 1,
        "scale_fn": lambda bm, bn, bk: (bm * bk // 128 + bn * bk // 128),
        "bk_values": [128],
        "preshuffle": None,
    },
    "a8w8_per_token_scale": {
        "a_dtype_size": 1,
        "w_dtype_size": 1,
        "scale_fn": lambda bm, bn, bk: bm * 4,
        "bk_values": None,
        "preshuffle": None,
    },
    "a8wfp4": {
        "a_dtype_size": 1,
        "w_dtype_size": 1,
        "scale_fn": lambda bm, bn, bk: bm * bk // 32 + bn * bk // 32,
        "bk_values": None,
        "preshuffle": None,
    },
    "afp4wfp4": {
        "a_dtype_size": 1,
        "w_dtype_size": 1,
        "scale_fn": lambda bm, bn, bk: bm * bk // 32 + bn * bk // 32,
        "bk_values": None,
        "preshuffle": None,
    },
    "afp4wfp4_pre_quant_atomic": {
        "a_dtype_size": 1,
        "w_dtype_size": 1,
        "scale_fn": lambda bm, bn, bk: bn * bk // 32,
        "bk_values": None,
        "preshuffle": None,
    },
}


def _min_bk(kernel_name: str) -> int:
    """Return minimum BK value required by a kernel."""
    if "afp4wfp4" in kernel_name:
        return 256
    if "wfp4" in kernel_name:
        return 64
    return 0


def compute_lds(bm, bn, bk, a_dtype_size, w_dtype_size, scale_fn,
                num_stages, preshuffle):
    """Return total LDS bytes for the given tile / pipeline configuration."""
    lds_a = 0 if preshuffle == "A" else bm * bk * a_dtype_size
    lds_w = 0 if preshuffle == "W" else bn * bk * w_dtype_size
    lds_total = (lds_a + lds_w) * num_stages + scale_fn(bm, bn, bk)
    return lds_total


def filter_valid_combos(kernel_name, num_stages,
                        bm_range=None, bn_range=None, bk_range=None):
    """Return (valid_bms, valid_bns, valid_bks) that fit LDS for *num_stages*."""
    params = KERNEL_PARAMS[kernel_name]
    a_sz = params["a_dtype_size"]
    w_sz = params["w_dtype_size"]
    scale_fn = params["scale_fn"]
    bk_values = params["bk_values"]
    preshuffle = params["preshuffle"]
    min_bk = _min_bk(kernel_name)

    bm_range = bm_range or DEFAULT_BM
    bn_range = bn_range or DEFAULT_BN
    bk_range = bk_range or DEFAULT_BK

    valid_bms = set()
    valid_bns = set()
    valid_bks = set()
    n_total = 0
    n_valid = 0

    for bm, bn, bk in product(bm_range, bn_range, bk_range):
        # BK constraint: must be in bk_values list if specified
        if bk_values is not None and bk not in bk_values:
            continue
        # BK minimum constraint
        if bk < min_bk:
            continue

        n_total += 1
        lds = compute_lds(bm, bn, bk, a_sz, w_sz, scale_fn,
                          num_stages, preshuffle)
        if lds <= LDS_LIMIT:
            n_valid += 1
            valid_bms.add(bm)
            valid_bns.add(bn)
            valid_bks.add(bk)

    return (sorted(valid_bms), sorted(valid_bns), sorted(valid_bks),
            n_valid, n_total)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute LDS-safe block-size ranges for GEMM kernel tuning "
                    "on AMD MI355X (gfx950).")
    parser.add_argument(
        "--kernel", type=str, required=True,
        choices=sorted(KERNEL_PARAMS.keys()),
        help="Kernel name")
    parser.add_argument(
        "--num-stages", nargs="+", type=int, default=[2],
        help="Pipeline num_stages values to evaluate (default: 2)")
    parser.add_argument(
        "--block-size-m-range", nargs="+", type=int, default=None,
        help="Override default BM range")
    parser.add_argument(
        "--block-size-n-range", nargs="+", type=int, default=None,
        help="Override default BN range")
    parser.add_argument(
        "--block-size-k-range", nargs="+", type=int, default=None,
        help="Override default BK range")
    parser.add_argument(
        "--print-cli", action="store_true", default=False,
        help="Print screen.py-compatible CLI args instead of human-readable summary")
    return parser.parse_args()


def main():
    args = parse_args()

    for ns in args.num_stages:
        valid_bms, valid_bns, valid_bks, n_valid, n_total = filter_valid_combos(
            args.kernel, ns,
            bm_range=args.block_size_m_range,
            bn_range=args.block_size_n_range,
            bk_range=args.block_size_k_range,
        )

        if args.print_cli:
            bm_str = " ".join(str(v) for v in valid_bms)
            bn_str = " ".join(str(v) for v in valid_bns)
            bk_str = " ".join(str(v) for v in valid_bks)
            print(f"# num_stages={ns}")
            print(f"--block-size-m-range {bm_str} "
                  f"--block-size-n-range {bn_str} "
                  f"--block-size-k-range {bk_str} "
                  f"--num-stages-range {ns}")
        else:
            print(f"=== kernel={args.kernel}  num_stages={ns} ===")
            print(f"  LDS limit       : {LDS_LIMIT} bytes (160 KB)")
            print(f"  Valid combos     : {n_valid} / {n_total}")
            print(f"  BLOCK_SIZE_M     : {valid_bms}")
            print(f"  BLOCK_SIZE_N     : {valid_bns}")
            print(f"  BLOCK_SIZE_K     : {valid_bks}")
            print()


if __name__ == "__main__":
    sys.exit(main())
