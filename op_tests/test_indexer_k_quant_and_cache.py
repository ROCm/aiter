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


def _split_k_scale(kv_cache, head_dim):
    """Split a kv_cache tensor into its K-data bytes and scale float32 regions.

    kv_cache shape: [block_num, block_size, head_dim + head_dim/quant_block_size * 4] (fp8).
    Both write and gather kernels treat each paged block as a block-major packed
    byte buffer: first `block_size*head_dim` bytes for K, then the rest for scales.
    """
    block_num, block_size, cache_stride = kv_cache.shape
    flat = kv_cache.view(block_num, block_size * cache_stride)
    k_bytes = flat[:, : block_size * head_dim].contiguous()
    scale_region = flat[:, block_size * head_dim :].contiguous()
    return k_bytes, scale_region.view(torch.float32)


def _write_block_preshuffle(block_flat, k_fp8_row, block_offset, head_dim):
    """Write one token's FP8 K values into a block using the MFMA 16x16 preshuffle layout."""
    token_tile_id = block_offset // TILE
    token_in_tile = block_offset % TILE
    for col_tile_id in range(head_dim // TILE):
        col_base = col_tile_id * TILE
        tile_base = (
            token_tile_id * TILE * head_dim
            + col_tile_id * TILE * TILE
            + token_in_tile * TILE
        )
        block_flat[tile_base : tile_base + TILE] = k_fp8_row[col_base : col_base + TILE]


def _compute_ref_scale(k_flat_quant_blocks, scale_fmt):
    """Replicate the kernel's fp32 scale computation exactly.

    The kernel works in fp32 throughout; doing the ue8m0 log2/ceil in bf16
    loses precision near power-of-two boundaries and can make the reference
    scale differ from the kernel's by a factor of 2. Cast to fp32 first.
    """
    per_token_amax, _ = torch.max(
        input=torch.abs(k_flat_quant_blocks.to(torch.float32)), dim=-1, keepdim=True
    )
    scale = per_token_amax / torch.finfo(dtypes.fp8).max
    if scale_fmt == "ue8m0":
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    return scale


def run_torch(k, kv_cache, slot_mapping, quant_block_size, scale_fmt, preshuffle=False):
    num_token, head_dim = k.shape
    block_num, block_size, cache_stride = kv_cache.shape
    scale = _compute_ref_scale(k.view(-1, quant_block_size), scale_fmt)
    k_fp8, scale = pertoken_quant(
        k.view(-1, quant_block_size), quant_dtype=dtypes.fp8, scale=scale
    )
    k_fp8 = k_fp8.view(num_token, head_dim)
    n_scale_bytes = head_dim // quant_block_size * 4
    kv_flat = kv_cache.view(block_num, block_size * cache_stride)
    for i in range(num_token):
        slot = slot_mapping[i].item()
        if slot < 0:
            continue
        block_id = slot // block_size
        block_offset = slot % block_size
        block_flat = kv_flat[block_id]
        if preshuffle:
            _write_block_preshuffle(block_flat, k_fp8[i], block_offset, head_dim)
        else:
            # Block-major packed layout to match the C++ kernel:
            #   [K slot 0 | K slot 1 | ... | K slot (B-1) | Scale slot 0 | ... | Scale slot (B-1)]
            k_offset = block_offset * head_dim
            block_flat[k_offset : k_offset + head_dim] = k_fp8[i]
        scale_offset = block_size * head_dim + block_offset * n_scale_bytes
        block_flat[scale_offset : scale_offset + n_scale_bytes] = (
            scale[i].view(dtypes.fp8).reshape(-1)
        )


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
    # Zero-init so unwritten padding slots (if any) match between ref and kernel.
    kv_cache = torch.zeros((block_num, block_size, head_dim + 4), dtype=dtypes.fp8)
    run_torch(k, kv_cache, slot_mapping, quant_block_size, scale_fmt, preshuffle=preshuffle)
    kv_cache2 = torch.zeros((block_num, block_size, head_dim + 4), dtype=dtypes.fp8)
    _, us = run_perftest(
        indexer_k_quant_and_cache,
        k,
        kv_cache2,
        slot_mapping,
        quant_block_size,
        scale_fmt,
        preshuffle,
    )
    # Compare K bytes (as FP8) and scale float32 regions separately to avoid the
    # FP8-bit-reinterpretation artifact when a float32 scale is viewed as 4 FP8 bytes.
    k_ref, s_ref = _split_k_scale(kv_cache, head_dim)
    k_got, s_got = _split_k_scale(kv_cache2, head_dim)
    err_k = checkAllclose(k_ref.to(torch.float), k_got.to(torch.float))
    err_s = checkAllclose(s_ref, s_got)
    ret = {"aiter us": us, "aiter k_err": err_k, "aiter s_err": err_s}
    if not preshuffle:
        # vllm reference op does not support preshuffle mode.
        try:
            from vllm import _custom_ops as ops

            kv_cache3 = torch.zeros(
                (block_num, block_size, head_dim + 4), dtype=dtypes.fp8
            )
            _, us2 = run_perftest(
                ops.indexer_k_quant_and_cache,
                k,
                kv_cache3,
                slot_mapping,
                quant_block_size,
                scale_fmt,
            )
            k_vllm, s_vllm = _split_k_scale(kv_cache3, head_dim)
            err2_k = checkAllclose(k_ref.to(torch.float), k_vllm.to(torch.float))
            err2_s = checkAllclose(s_ref, s_vllm)
            ret.update({"vllm us": us2, "vllm k_err": err2_k, "vllm s_err": err2_s})
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

    # Reference quantized values (layout-agnostic). Use the same fp32 scale
    # helper as run_torch so we match the kernel's fp32 precision exactly.
    ref_scale = _compute_ref_scale(k.view(-1, quant_block_size), scale_fmt)
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
