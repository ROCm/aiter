# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter.test_common import checkAllclose, run_perftest, benchmark
from aiter import dtypes
from aiter import (
    pertoken_quant,
    dtypes,
    indexer_k_quant_and_cache,
    cp_gather_indexer_k_quant_cache,
)
import argparse
import pandas as pd

MAX_TOKEN_SUPPORTED = 16384
TILE = 16  # MFMA 16x16 tile size used by the preshuffle layout
torch.set_default_device("cuda")


def _write_kv_cache_preshuffle(
    kv_cache,
    k_fp8_row,
    scale_bytes,
    slot,
    head_dim,
    block_size,
    quant_block_size,
):
    """Write one token's FP8 K + scale into kv_cache using the preshuffle
    (MFMA 16x16 tile) layout. Scale region layout is unchanged."""
    block_id = slot // block_size
    block_offset = slot % block_size
    token_tile_id = block_offset // TILE
    token_in_tile = block_offset % TILE
    cache_stride = head_dim + head_dim // quant_block_size * 4
    block_flat = kv_cache.view(-1, block_size * cache_stride)[block_id]
    for col_tile_id in range(head_dim // TILE):
        col_base = col_tile_id * TILE
        tile_base = (
            token_tile_id * TILE * head_dim
            + col_tile_id * TILE * TILE
            + token_in_tile * TILE
        )
        block_flat[tile_base : tile_base + TILE] = k_fp8_row[col_base : col_base + TILE]
    n_scale_bytes = head_dim // quant_block_size * 4
    scale_offset = block_size * head_dim + block_offset * n_scale_bytes
    block_flat[scale_offset : scale_offset + n_scale_bytes] = scale_bytes


def run_torch(k, kv_cache, slot_mapping, quant_block_size, scale_fmt, preshuffle=False):
    num_token, head_dim = k.shape
    block_size = kv_cache.shape[1]
    per_token_amax, _ = torch.max(
        input=torch.abs(k.view(-1, quant_block_size)), dim=-1, keepdim=True
    )
    scale = per_token_amax / torch.finfo(dtypes.fp8).max
    if scale_fmt == "ue8m0":
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    k_fp8, scale = pertoken_quant(
        k.view(-1, quant_block_size), quant_dtype=dtypes.fp8, scale=scale
    )
    k_fp8 = k_fp8.view(num_token, head_dim)
    for i in range(num_token):
        slot = slot_mapping[i].item()
        if slot < 0:
            continue
        if preshuffle:
            scale_bytes = scale[i].view(dtypes.fp8).reshape(-1)
            _write_kv_cache_preshuffle(
                kv_cache,
                k_fp8[i],
                scale_bytes,
                slot,
                head_dim,
                block_size,
                quant_block_size,
            )
        else:
            block_id = slot // block_size
            block_offset = slot % block_size
            kv_cache[block_id, block_offset, :head_dim] = k_fp8[i]
            kv_cache[block_id, block_offset, head_dim:] = scale[i].view(dtypes.fp8)


@benchmark()
def test_indexer_k_quant_and_cache(
    num_token, block_size, quant_block_size, head_dim=128, preshuffle=False
):
    assert (
        num_token <= MAX_TOKEN_SUPPORTED
    ), f"test only support max_token={MAX_TOKEN_SUPPORTED}"
    if preshuffle:
        assert block_size % TILE == 0 and head_dim % TILE == 0, (
            f"preshuffle requires block_size and head_dim multiples of {TILE}, "
            f"got block_size={block_size}, head_dim={head_dim}"
        )
    block_num = (num_token + block_size - 1) // block_size
    k = torch.randn((num_token, head_dim), dtype=dtypes.bf16)
    slot_mapping = torch.arange(0, num_token, 1, dtype=torch.int64)
    scale_fmt = "ue8m0"
    # For preshuffle we zero-init both buffers so full-tensor compare is meaningful
    # (paged blocks may contain padding slots that don't align to row-per-token view).
    # Non-preshuffle keeps the original `torch.empty` + sliced-compare behaviour.
    alloc = torch.zeros if preshuffle else torch.empty
    kv_cache = alloc((block_num, block_size, head_dim + 4), dtype=dtypes.fp8)
    run_torch(k, kv_cache, slot_mapping, quant_block_size, scale_fmt, preshuffle=preshuffle)
    kv_cache2 = alloc((block_num, block_size, head_dim + 4), dtype=dtypes.fp8)
    _, us = run_perftest(
        indexer_k_quant_and_cache,
        k,
        kv_cache2,
        slot_mapping,
        quant_block_size,
        scale_fmt,
        preshuffle,
    )
    if preshuffle:
        err = checkAllclose(kv_cache.to(torch.float), kv_cache2.to(torch.float))
    else:
        err = checkAllclose(
            kv_cache.view(-1, head_dim + 4)[:num_token].to(torch.float),
            kv_cache2.view(-1, head_dim + 4)[:num_token].to(torch.float),
        )
    # scale = kv_cache[:, :, head_dim:].view(torch.float)
    # scale2 = kv_cache2[:, :, head_dim:].view(torch.float)
    ret = {"aiter us": us, "aiter err": err}
    if not preshuffle:
        # vllm reference op does not support preshuffle mode.
        try:
            from vllm import _custom_ops as ops

            kv_cache3 = torch.empty((block_num, block_size, head_dim + 4), dtype=dtypes.fp8)
            _, us2 = run_perftest(
                ops.indexer_k_quant_and_cache,
                k,
                kv_cache3,
                slot_mapping,
                quant_block_size,
                scale_fmt,
            )
            err2 = checkAllclose(
                kv_cache.view(-1, head_dim + 4)[:num_token].to(torch.float),
                kv_cache3.view(-1, head_dim + 4)[:num_token].to(torch.float),
            )
            ret.update({"vllm us": us2, "vllm err": err2})
        except Exception:
            # Ignore all exceptions here because vllm._custom_ops is optional and may not be available.
            pass
    return ret


@benchmark()
def test_cp_gather_indexer_k_quant_cache(
    num_token, block_size, quant_block_size, head_dim=128, preshuffle=False
):
    """Round-trip: write with indexer_k_quant_and_cache(preshuffle=P),
    read back with cp_gather_indexer_k_quant_cache(preshuffle=P), and compare
    to the direct pertoken-quant reference. Verifies write+gather layouts are
    internally consistent and match the expected quantized values."""
    assert (
        num_token <= MAX_TOKEN_SUPPORTED
    ), f"test only support max_token={MAX_TOKEN_SUPPORTED}"
    if preshuffle:
        assert block_size % TILE == 0 and head_dim % TILE == 0, (
            f"preshuffle requires block_size and head_dim multiples of {TILE}, "
            f"got block_size={block_size}, head_dim={head_dim}"
        )
    block_num = (num_token + block_size - 1) // block_size
    k = torch.randn((num_token, head_dim), dtype=dtypes.bf16)
    slot_mapping = torch.arange(0, num_token, 1, dtype=torch.int64)
    scale_fmt = "ue8m0"

    # Reference quantized values (layout-agnostic).
    per_token_amax, _ = torch.max(
        input=torch.abs(k.view(-1, quant_block_size)), dim=-1, keepdim=True
    )
    ref_scale = per_token_amax / torch.finfo(dtypes.fp8).max
    if scale_fmt == "ue8m0":
        ref_scale = torch.pow(2.0, torch.ceil(torch.log2(ref_scale)))
    ref_k_fp8, ref_scale = pertoken_quant(
        k.view(-1, quant_block_size), quant_dtype=dtypes.fp8, scale=ref_scale
    )
    ref_k_fp8 = ref_k_fp8.view(num_token, head_dim)
    ref_scale = ref_scale.view(num_token, head_dim // quant_block_size)

    # Write phase.
    kv_cache = torch.zeros((block_num, block_size, head_dim + 4), dtype=dtypes.fp8)
    indexer_k_quant_and_cache(
        k, kv_cache, slot_mapping, quant_block_size, scale_fmt, preshuffle
    )

    # Gather phase: batch_size=1, linear block_table covering every slot in order.
    block_table = torch.arange(0, block_num, dtype=torch.int32).view(1, -1)
    cu_seq_lens = torch.tensor([0, num_token], dtype=torch.int32)
    dst_k = torch.empty((num_token, head_dim), dtype=dtypes.fp8)
    dst_scale = torch.empty(
        (num_token, head_dim // quant_block_size), dtype=torch.float32
    )
    _, us = run_perftest(
        cp_gather_indexer_k_quant_cache,
        kv_cache,
        dst_k,
        dst_scale,
        block_table,
        cu_seq_lens,
        preshuffle,
    )
    err_k = checkAllclose(dst_k.to(torch.float), ref_k_fp8.to(torch.float))
    err_s = checkAllclose(dst_scale, ref_scale)
    return {"aiter us": us, "k err": err_k, "scale err": err_s}


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Test indexer_k_quant_and_cache.",
)
parser.add_argument(
    "-m",
    type=int,
    nargs="*",
    default=[1, 64, 128, 257, 1028, 16384],
    help="""token num""",
)
parser.add_argument(
    "-b",
    "--block_size",
    type=int,
    nargs="*",
    default=[1],
    help="""block_size, default: 1""",
)
parser.add_argument(
    "-p",
    "--preshuffle",
    action="store_true",
    help="""Also run preshuffle=True. Requires block_size and head_dim to be multiples of 16; combos that don't meet this are silently skipped.""",
)
parser.add_argument(
    "-g",
    "--gather",
    action="store_true",
    help="""Also run cp_gather_indexer_k_quant_cache round-trip tests.""",
)

args = parser.parse_args()

preshuffle_modes = [False] + ([True] if args.preshuffle else [])

df = []
gather_df = []
for m in args.m:
    for block_size in args.block_size:
        for preshuffle in preshuffle_modes:
            if preshuffle and (block_size % TILE != 0):
                continue
            ret = test_indexer_k_quant_and_cache(m, block_size, 128, 128, preshuffle)
            df.append(ret)
            if args.gather:
                gret = test_cp_gather_indexer_k_quant_cache(
                    m, block_size, 128, 128, preshuffle
                )
                gather_df.append(gret)
df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("indexer_k_quant_and_cache summary (markdown):\n%s", df_md)
if args.gather:
    gather_df = pd.DataFrame(gather_df)
    aiter.logger.info(
        "cp_gather_indexer_k_quant_cache round-trip summary (markdown):\n%s",
        gather_df.to_markdown(index=False),
    )
