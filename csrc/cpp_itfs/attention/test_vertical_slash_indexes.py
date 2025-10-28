# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
from csrc.cpp_itfs.attention.vertical_slash_indexes import convert_vertical_slash_index
import triton.language as tl
import triton

@triton.jit
def sort_block_offset(
    block_offset_ptr,
    block_count_ptr,
    NNZ_S: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_VALUE: tl.constexpr,
):
    # BLOCK_SIZE: tl.constexpr = 256
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    row_idx = tl.program_id(2)
    num_heads = tl.num_programs(1)
    num_rows = tl.num_programs(2)
    index = batch_idx * num_heads * num_rows + head_idx * num_rows + row_idx
    block_count = tl.load(block_count_ptr + index)
    block_offset_ptr = block_offset_ptr + index * NNZ_S
    off = tl.arange(0, BLOCK_SIZE)
    mask = off < block_count
    block_offset = tl.load(block_offset_ptr+off, mask=mask, other=MAX_VALUE)
    block_offset = tl.sort(block_offset)
    tl.store(block_offset_ptr+off, block_offset, mask=mask)


class TestConvertVerticalSlashIndex:
    """Test cases for convert_vertical_slash_index function."""

    def test_basic_functionality(self):
        """Test basic functionality with simple inputs."""
        torch.random.manual_seed(0)
        batch_size = 4
        num_heads = 16
        nnz_vertical = 1000
        nnz_slash = 1000
        context_size = 7100
        block_size_M = 64
        block_size_N = 64

        # Create input tensors
        q_seqlens = torch.tensor(
            [ 7100, 5005, 6797, 6886], dtype=torch.int32, device="cuda"
        )
        kv_seqlens = torch.tensor(
            [22537, 15583, 13619, 10264], dtype=torch.int32, device="cuda"
        )

        # Create vertical indexes - should be sorted ascending
        vertical_indexes = (
            torch.ones(
                (batch_size, num_heads, nnz_vertical), dtype=torch.int32, device="cuda"
            )
            * 1000
        )
        vertical_indexes = torch.load("/mnt/raid0/sixifang/aiter/input_value/v_idx.pt")
        # Sort vertical indexes for each head
        # for b in range(batch_size):
        #     for h in range(num_heads):
        #         vertical_indexes[b, h] = torch.sort(vertical_indexes[b, h])[0]

        # Create slash indexes - should be sorted ascending
        slash_indexes = (
            torch.ones(
                (batch_size, num_heads, nnz_slash), dtype=torch.int32, device="cuda"
            )
            * 1000
        )
        slash_indexes = torch.load("/mnt/raid0/sixifang/aiter/input_value/s_idx.pt")
        # Sort slash indexes for each head
        # for b in range(batch_size):
        #     for h in range(num_heads):
        #         slash_indexes[b, h] = torch.sort(slash_indexes[b, h])[0]

        # Call the function
        with torch.profiler.profile() as prof:
            block_count, block_offset, column_count, column_index = (
                convert_vertical_slash_index(
                    q_seqlens,
                    kv_seqlens,
                    vertical_indexes,
                    slash_indexes,
                    context_size,
                    block_size_M,
                    block_size_N,
                )
            )
        
        # print(block_count, block_offset, column_count, column_index)
        # block_offset = torch.sort(block_offset, stable=False)
            num_rows = triton.cdiv(context_size, block_size_M)
            # block_offset = torch.sort(block_offset, stable=False).values
            sort_block_offset[(batch_size, num_heads, num_rows)](block_offset, block_count, nnz_slash, triton.next_power_of_2(nnz_slash), torch.iinfo(torch.int32).max)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        ref_block_count = torch.load("/mnt/raid0/sixifang/aiter/input_value/block_count.pt")
        ref_block_offset = torch.load("/mnt/raid0/sixifang/aiter/input_value/block_offset.pt")
        ref_column_count = torch.load("/mnt/raid0/sixifang/aiter/input_value/column_count.pt")
        ref_column_index = torch.load("/mnt/raid0/sixifang/aiter/input_value/column_index.pt")
        torch.testing.assert_close(block_count, ref_block_count)
        torch.testing.assert_close(block_offset, ref_block_offset)
        torch.testing.assert_close(column_count, ref_column_count)
        torch.testing.assert_close(column_index, ref_column_index)
        # print(block_count.sum(), torch.load("/mnt/raid0/sixifang/aiter/input_value/block_count.pt").sum())
        # print(block_offset.sum(), torch.load("/mnt/raid0/sixifang/aiter/input_value/block_offset.pt").sum())
        # print(column_count.sum(), torch.load("/mnt/raid0/sixifang/aiter/input_value/column_count.pt").sum())
        # print(column_index.sum(), torch.load("/mnt/raid0/sixifang/aiter/input_value/column_index.pt").sum())
        # Verify output shapes
        # num_rows = (context_size + block_size_M - 1) // block_size_M
        # assert block_count.shape == (batch_size, num_heads, num_rows)
        # assert block_offset.shape == (batch_size, num_heads, num_rows, nnz_slash)
        # assert column_count.shape == (batch_size, num_heads, num_rows)
        # assert column_index.shape == (batch_size, num_heads, num_rows, nnz_vertical)

        # # Verify output dtypes
        # assert block_count.dtype == torch.int32
        # assert block_offset.dtype == torch.int32
        # assert column_count.dtype == torch.int32
        # assert column_index.dtype == torch.int32

        # # Verify output devices
        # assert block_count.device.type == "cuda"
        # assert block_offset.device.type == "cuda"
        # assert column_count.device.type == "cuda"
        # assert column_index.device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-s"])