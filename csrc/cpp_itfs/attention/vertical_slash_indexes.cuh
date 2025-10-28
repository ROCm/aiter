// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#pragma once

#include <assert.h>

#include <hip/hip_runtime.h>


__global__ void convert_vertical_slash_index_kernel(
    const int* seqlens,          // [BATCH, ]
    const int* ctxlens,          // [BATCH, ]
    const int* vertical_indexes, // [BATCH, N_HEADS, NNZ_V]
    const int* slash_indexes,    // [BATCH, N_HEADS, NNZ_S]
    int* block_count,            // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    int* block_offset,           // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_S]
    int* column_count,           // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    int* column_index,           // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_V]
    const int N_HEADS,
    const int N_ROWS,
    const int BLOCK_SIZE_M,
    const int BLOCK_SIZE_N,
    const int NNZ_V,
    const int NNZ_S
)
{
    const int head_idx  = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int group_idx = blockIdx.z;

    const int thread_idx = threadIdx.x;

    const int seqlen = seqlens[batch_idx];
    const int ctxlen = ctxlens[batch_idx];
    // const int N_HEADS = gridDim.x;
    const int block_idx_m = group_idx * blockDim.x + thread_idx;
    int start_m = block_idx_m * BLOCK_SIZE_M;
    if(start_m >= seqlen)
    {
        return;
    }
    start_m += ctxlen;
    const int end_m = start_m + BLOCK_SIZE_M;

    const int* v_ptr = vertical_indexes + (batch_idx * N_HEADS + head_idx) * NNZ_V;
    const int* s_ptr = slash_indexes + (batch_idx * N_HEADS + head_idx) * NNZ_S;

    const int row_offset = (batch_idx * N_HEADS + head_idx) * N_ROWS + block_idx_m;
    block_count += row_offset;
    block_offset += row_offset * NNZ_S;
    column_count += row_offset;
    column_index += row_offset * NNZ_V;

    int tmp_col_cnt = 0, tmp_blk_cnt = 0;
    int s = 0, v = 0;
    int v_idx = v_ptr[v++];
    int s_idx = s_ptr[s++];
    while(s_idx >= end_m)
    {
        s_idx = s_ptr[s++];
    }
    
    s_idx = max(
        end_m - s_idx,
        BLOCK_SIZE_M); // since s_idx has been compute using (total_len - s_idx) in python script
    int s_idx2 = s_ptr[s];
    int range_start = s_idx - BLOCK_SIZE_M, range_end = s_idx;
    
    while(1)
    {
        if(v_idx < end_m && v_idx < range_end)
        {
            if(v_idx < range_start)
            {
                column_index[tmp_col_cnt++] = v_idx;
            }
            v_idx = v < NNZ_V ? v_ptr[v] : end_m + BLOCK_SIZE_M;
            v += v < NNZ_V ? 1 : 0;
        }
        else
        {
            if(s >= NNZ_S)
            {
                break;
            }
            s_idx = max(end_m - s_idx2, BLOCK_SIZE_M);
            s++;
            s_idx2 = s_ptr[s];

            const int old_range_end = range_end;
            if(s_idx > range_end){
                block_offset[tmp_blk_cnt++] = s_idx > range_end + BLOCK_SIZE_M ? range_start : range_end;
                range_end = s_idx > range_end + BLOCK_SIZE_M ? s_idx : range_end + BLOCK_SIZE_M;
            }
            range_start = s_idx > old_range_end + BLOCK_SIZE_M ? s_idx - BLOCK_SIZE_M : range_start;
        }
    }
    block_offset[tmp_blk_cnt++] = range_start;
    block_count[0]  = tmp_blk_cnt;
    column_count[0] = tmp_col_cnt;
}