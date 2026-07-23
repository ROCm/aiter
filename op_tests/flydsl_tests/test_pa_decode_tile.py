# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and performance sweep for FlyDSL paged-attention Tile."""

import argparse
import itertools

import aiter
import pandas as pd
import torch
from aiter import dtypes, per_tensor_quant
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]
KV_COMPUTE_BLOCK = 256

# Pairwise coverage of the normal-accuracy axes in FlyDSL's PA regression test:
# batches {3, 81, 128}, Q/KV heads {(4,1), (8,1), (16,1)}, head dims
# {128, 256}, and contexts {1027, 8192}. Keep the original 257-token boundary
# case as well. Both supported block sizes are crossed with every case in main().
DEFAULT_BATCH_SIZES = [3, 81, 128]
DEFAULT_SHAPES = [
    (8, 1, 128, 257),
    (4, 1, 128, 1027),
    (8, 1, 128, 1027),
    (8, 1, 256, 1027),
    (16, 1, 128, 8192),
]

try:
    from aiter.ops.flydsl.pa_decode_tile import (
        get_recommended_splits,
        pa_decode_tile,
    )
except ImportError:
    get_recommended_splits = None
    pa_decode_tile = None


def _quant_dtype() -> torch.dtype:
    return torch.float8_e4m3fn if get_gfx() == "gfx950" else torch.float8_e4m3fnuz


def run_torch(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_query_heads, head_dim = query.shape
    block_size = key_cache.shape[2]
    num_kv_heads = key_cache.shape[1]
    query_group_size = num_query_heads // num_kv_heads
    softmax_scale = head_dim**-0.5
    output = torch.empty_like(query)

    for seq_idx in range(batch_size):
        context_length = int(context_lengths[seq_idx].item())
        token_ids = torch.arange(context_length, device=query.device)
        logical_pages = token_ids // block_size
        token_offsets = token_ids % block_size
        physical_pages = block_tables[seq_idx, logical_pages].long()

        keys = (
            key_cache[physical_pages, :, token_offsets, :].float() * key_scale.float()
        )
        values = (
            value_cache[physical_pages, :, :, token_offsets].float()
            * value_scale.float()
        )
        keys = keys.repeat_interleave(query_group_size, dim=1)
        values = values.repeat_interleave(query_group_size, dim=1)
        scores = (
            torch.einsum("hd,khd->hk", query[seq_idx].float(), keys) * softmax_scale
        )
        probs = torch.softmax(scores, dim=-1)
        output[seq_idx] = torch.einsum("hk,khd->hd", probs, values).to(query.dtype)
    return output


def _run_flydsl(
    output,
    query,
    key_cache,
    value_cache,
    block_tables,
    context_lengths,
    key_scale,
    value_scale,
    num_partitions,
    pmax,
    psum,
    pout,
):
    pa_decode_tile(
        output,
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lengths,
        key_scale,
        value_scale,
        num_partitions=num_partitions,
        pmax=pmax,
        psum=psum,
        pout=pout,
    )
    return output


@benchmark()
def test_pa_decode_tile(
    batch_size,
    num_query_heads,
    num_kv_heads,
    head_dim,
    context_length,
    block_size,
    dtype,
):
    if pa_decode_tile is None or get_recommended_splits is None:
        raise RuntimeError("FlyDSL is not available")
    if dtype not in (dtypes.fp16, dtypes.bf16):
        raise ValueError(f"pa_decode_tile only supports fp16/bf16, got {dtype}")
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")

    torch.manual_seed(0)
    blocks_per_sequence = (context_length + block_size - 1) // block_size
    num_blocks = batch_size * blocks_per_sequence

    query = torch.empty(
        batch_size,
        num_query_heads,
        head_dim,
        dtype=dtype,
    ).uniform_(-0.5, 0.5)
    key = torch.empty(
        num_blocks,
        num_kv_heads,
        block_size,
        head_dim,
        dtype=dtype,
    ).uniform_(-0.5, 0.5)
    value = torch.empty(
        num_blocks,
        num_kv_heads,
        head_dim,
        block_size,
        dtype=dtype,
    ).uniform_(-0.5, 0.5)

    quant_dtype = _quant_dtype()
    key_quant, key_scale = per_tensor_quant(key, quant_dtype=quant_dtype)
    value_quant, value_scale = per_tensor_quant(value, quant_dtype=quant_dtype)
    key_cache = (
        key_quant.view(
            num_blocks,
            num_kv_heads,
            block_size,
            head_dim // 16,
            16,
        )
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    value_cache = value_quant.contiguous()
    block_tables = torch.arange(num_blocks, dtype=torch.int32).reshape(
        batch_size, blocks_per_sequence
    )
    context_lengths = torch.full((batch_size,), context_length, dtype=torch.int32)
    output = torch.empty_like(query)

    reference = run_torch(
        query,
        key_quant,
        value_quant,
        block_tables,
        context_lengths,
        key_scale,
        value_scale,
    )

    query_group_size = num_query_heads // num_kv_heads
    num_partitions = get_recommended_splits(
        batch_size,
        num_kv_heads,
        split_kv_blocks=KV_COMPUTE_BLOCK // block_size,
    )
    partial_shape = (
        batch_size,
        num_kv_heads,
        num_partitions,
        query_group_size,
    )
    pmax = torch.empty(partial_shape, dtype=dtypes.fp32)
    psum = torch.empty_like(pmax)
    pout = torch.empty(*partial_shape, head_dim, dtype=dtype)

    candidates = {
        "flydsl": lambda: _run_flydsl(
            output,
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            key_scale,
            value_scale,
            num_partitions,
            pmax,
            psum,
            pout,
        )
    }

    # QK and PV each perform one multiply-add per query-head/context pair.
    flops = 4 * batch_size * num_query_heads * context_length * head_dim
    # Logical tensor traffic: Q + O + the referenced K/V tokens and metadata.
    nbytes = (
        2 * query.numel() * query.element_size()
        + 2
        * batch_size
        * num_kv_heads
        * context_length
        * head_dim
        * key_cache.element_size()
        + block_tables.numel() * block_tables.element_size()
        + context_lengths.numel() * context_lengths.element_size()
        + key_scale.numel() * key_scale.element_size()
        + value_scale.numel() * value_scale.element_size()
    )

    ret = {"gfx": get_gfx(), "partitions": num_partitions}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        err = checkAllclose(
            reference.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=3e-2,
            atol=3e-2,
            tol_err_ratio=0.0,
            msg=f"{name}: pa_decode_tile",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if not torch.cuda.is_available():
        aiter.logger.warning("ROCm is not available; skipping pa_decode_tile")
        return
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("pa_decode_tile unsupported on %s; skipping", get_gfx())
        return
    if pa_decode_tile is None:
        aiter.logger.warning("flydsl is unavailable; skipping pa_decode_tile")
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL pa_decode_tile correctness + perf sweep",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.bf16],
        help="""Query/output data type.
        e.g.: -d bf16 fp16""",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=DEFAULT_BATCH_SIZES,
        help="""Batch sizes.
        e.g.: -b 1 3 16""",
    )
    parser.add_argument(
        "-s",
        "--shapes",
        type=dtypes.str2tuple,
        nargs="*",
        default=DEFAULT_SHAPES,
        help="""(num_query_heads,num_kv_heads,head_dim,context_length).
        e.g.: -s 8,1,128,257""",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        nargs="*",
        choices=[16, 64],
        default=[16, 64],
        help="""KV-cache block sizes.""",
    )
    args = parser.parse_args()

    rows = []
    for dtype, batch_size, shape, block_size in itertools.product(
        args.dtype, args.batch, args.shapes, args.block_size
    ):
        num_query_heads, num_kv_heads, head_dim, context_length = shape
        rows.append(
            test_pa_decode_tile(
                batch_size,
                num_query_heads,
                num_kv_heads,
                head_dim,
                context_length,
                block_size,
                dtype,
            )
        )

    df = pd.DataFrame(rows)
    aiter.logger.info(
        "pa_decode_tile summary (markdown):\n%s", df.to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
