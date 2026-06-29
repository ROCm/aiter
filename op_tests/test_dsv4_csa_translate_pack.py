# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Unit test for the DeepSeek-V4 csa_translate_pack kernel (aiter port of ATOM's
atom/model_ops/v4_kernels/csa_translate_pack.py)."""

import argparse
import sys

import torch

from aiter.ops.triton.dsv4.csa_translate_pack import (
    csa_translate_pack,
    csa_translate_pack_reference,
)
from aiter.test_common import checkAllclose

torch.set_default_device("cuda")
torch.manual_seed(0)


def run(batch, valid_ks, index_topk, csa_block_capacity, mnbps, swa_pages, window_size):
    """valid_ks: per-token CSA valid count. Build a consistent indptr/topk set."""
    T = len(valid_ks)
    batch_id = torch.arange(T, dtype=torch.int32) % batch
    positions = torch.arange(T, dtype=torch.int32) + window_size  # pos+1 > window
    # skip per token: inline = min(pos+1, window) = window here (pos>=window).
    skip = torch.full((T,), window_size, dtype=torch.int32)
    slice_len = skip + torch.tensor(valid_ks, dtype=torch.int32)
    indptr = torch.zeros(T + 1, dtype=torch.int32)
    indptr[1:] = slice_len.cumsum(0)
    total = int(indptr[-1].item())

    # topk_local: seq-local row ids in [0, mnbps*csa_block_capacity)
    max_row = mnbps * csa_block_capacity
    topk_local = torch.randint(0, max_row, (T, index_topk), dtype=torch.int32)
    block_tables = torch.randint(0, 1000, (batch, mnbps), dtype=torch.int32)

    out_ref = torch.full((total,), -7, dtype=torch.int32)
    out_ker = torch.full((total,), -7, dtype=torch.int32)
    csa_translate_pack_reference(
        topk_local,
        block_tables,
        positions,
        indptr,
        batch_id,
        skip,
        out_ref,
        swa_pages=swa_pages,
        csa_block_capacity=csa_block_capacity,
        window_size=0,
    )
    # kernel: use inline-skip path (window_size>0, skip=None) -> must match ref
    csa_translate_pack(
        topk_local,
        block_tables,
        positions,
        indptr,
        batch_id,
        None,
        out_ker,
        swa_pages=swa_pages,
        csa_block_capacity=csa_block_capacity,
        window_size=window_size,
    )
    return checkAllclose(
        out_ref.float(),
        out_ker.float(),
        atol=0,
        rtol=0,
        msg=f"csa_translate_pack b{batch} T{T} topk{index_topk}",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index-topk", type=int, default=1024)
    p.add_argument("--ratio", type=int, default=4)
    args = p.parse_args()
    cbc = 128 // args.ratio  # csa_block_capacity = 32
    errs = [
        run(
            2,
            [5, 0, 17, 1024],
            args.index_topk,
            cbc,
            mnbps=64,
            swa_pages=256,
            window_size=128,
        ),
        run(
            3,
            [1, 1, 1, 64, 200, 0],
            args.index_topk,
            cbc,
            mnbps=32,
            swa_pages=128,
            window_size=64,
        ),
    ]
    fail = sum(1 for e in errs if not (e == 0 or (isinstance(e, float) and e < 1e-9)))
    if fail:
        print(f"{fail} case(s) FAILED")
        sys.exit(1)
    print("All cases passed.")


if __name__ == "__main__":
    main()
