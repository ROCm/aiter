# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Unit test for DeepSeek-V4 state-write kernels (aiter port of ATOM's
atom/model_ops/v4_kernels/state_writes.py): swa_write + update_compressor_states."""

import argparse
import sys

import torch

from aiter import dtypes
from aiter.ops.triton.dsv4.state_writes import (
    swa_write,
    swa_write_reference,
    update_compressor_states,
    update_compressor_states_reference,
)
from aiter.test_common import checkAllclose

torch.set_default_device("cuda")
torch.manual_seed(0)


def test_swa_write(batch, seqlens, head_dim, cache_size):
    T = sum(seqlens)
    kv = (torch.randn(T, head_dim) * 0.3).to(dtypes.bf16)
    cu = torch.zeros(batch + 1, dtype=torch.int32)
    cu[1:] = torch.tensor(seqlens, dtype=torch.int32).cumsum(0)
    # absolute positions per seq = arange(seqlen)
    positions = torch.cat([torch.arange(s, dtype=torch.int64) for s in seqlens])
    state_slot = torch.arange(batch, dtype=torch.int32)
    write_per_batch = min(max(seqlens), cache_size)

    swa_ref = torch.zeros(batch, cache_size, head_dim, dtype=dtypes.bf16)
    swa_out = torch.zeros(batch, cache_size, head_dim, dtype=dtypes.bf16)
    swa_write_reference(
        kv, positions, cu, state_slot, swa_ref, cache_size, write_per_batch
    )
    swa_write(kv, positions, cu, state_slot, swa_out, cache_size, write_per_batch)
    return checkAllclose(swa_ref, swa_out, msg=f"swa_write b{batch} D{head_dim}")


def test_update_compressor_states(num_write, dim, ratio, overlap, num_slots):
    state_size = (2 if overlap else 1) * ratio
    kv = (torch.randn(num_write, dim) * 0.3).to(dtypes.bf16)
    score = (torch.randn(num_write, dim) * 0.3).to(dtypes.bf16)
    ape = (torch.randn(ratio, dim) * 0.3).to(dtypes.bf16)
    # write_plan rows: (ragged_id, batch_id, position, _). Real plans never
    # write the same (slot, position % state_size) twice; emit unique dests so
    # the parallel kernel has no races to disagree with the sequential ref.
    assert num_write <= num_slots * state_size
    plan = torch.zeros(num_write, 4, dtype=torch.int32)
    plan[:, 0] = torch.arange(num_write, dtype=torch.int32)
    plan[:, 1] = torch.arange(num_write, dtype=torch.int32) % num_slots
    plan[:, 2] = torch.arange(num_write, dtype=torch.int32) // num_slots
    state_slot_mapping = torch.arange(num_slots, dtype=torch.int32)

    kvs_ref = torch.zeros(num_slots, state_size, dim, dtype=dtypes.bf16)
    scs_ref = torch.zeros(num_slots, state_size, dim, dtype=dtypes.bf16)
    kvs_out = torch.zeros(num_slots, state_size, dim, dtype=dtypes.bf16)
    scs_out = torch.zeros(num_slots, state_size, dim, dtype=dtypes.bf16)
    update_compressor_states_reference(
        kv,
        score,
        ape,
        kvs_ref,
        scs_ref,
        write_plan=plan,
        state_slot_mapping=state_slot_mapping,
        ratio=ratio,
        overlap=overlap,
    )
    update_compressor_states(
        kv,
        score,
        ape,
        kvs_out,
        scs_out,
        write_plan=plan,
        num_write=num_write,
        state_slot_mapping=state_slot_mapping,
        ratio=ratio,
        overlap=overlap,
    )
    e1 = checkAllclose(kvs_ref, kvs_out, msg=f"compress kv_state r{ratio} ov{overlap}")
    e2 = checkAllclose(
        scs_ref, scs_out, msg=f"compress score_state r{ratio} ov{overlap}"
    )
    return max(e1 if isinstance(e1, float) else 0, e2 if isinstance(e2, float) else 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head-dim", type=int, default=512)
    p.add_argument("--cache-size", type=int, default=128)
    p.add_argument("--dim", type=int, default=512)
    args = p.parse_args()
    errs = []
    errs.append(test_swa_write(2, [64, 40], args.head_dim, args.cache_size))
    errs.append(test_swa_write(3, [200, 1, 130], args.head_dim, args.cache_size))
    errs.append(test_update_compressor_states(18, args.dim, 4, True, 3))
    errs.append(test_update_compressor_states(20, args.dim, 128, False, 2))
    fail = sum(1 for e in errs if not (e == 0 or (isinstance(e, float) and e < 0.02)))
    if fail:
        print(f"{fail} case(s) FAILED")
        sys.exit(1)
    print("All cases passed.")


if __name__ == "__main__":
    main()
