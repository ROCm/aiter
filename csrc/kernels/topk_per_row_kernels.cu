// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/all.h>

#include "dispatch_utils.h"
#include <hipcub/hipcub.hpp>
#include <hipcub/util_type.hpp>

namespace aiter {

static inline __device__ uint16_t extractBinIdx(float x)
{
    union
    {
        __half h;
        uint16_t u16;
    } tmp;
    tmp.h   = __float2half_rn(x);
    tmp.u16 = (x < 0.f) ? (~tmp.u16 & 0xffff) : (tmp.u16 | 0x8000);
    return 511 - (tmp.u16 >> 7);
}

using fp32x1 = __attribute__((__ext_vector_type__(1))) float;
using fp32x2 = __attribute__((__ext_vector_type__(2))) float;
using fp32x4 = __attribute__((__ext_vector_type__(4))) float;

template <int vec>
struct to_vector;

template <>
struct to_vector<1>
{
    using type = fp32x1;
};

template <>
struct to_vector<2>
{
    using type = fp32x2;
};

template <>
struct to_vector<4>
{
    using type = fp32x4;
};

template <int kNumThreadsPerBlock = 512, int kNumBins = 512, int kTopK = 2048, int Vector = 4>
__device__ void topk_per_row_kernel(const float* logits,
                                    const int rowStart,
                                    const int rowEnd,
                                    const int rowIdx,
                                    int* outIndices,
                                    int stride0,
                                    int stride1)
{
    // The number of elements per thread for the final top-k sort.
    static constexpr int kNumTopKItemsPerThread = kTopK / kNumThreadsPerBlock;
    // The class to sort the elements during the final top-k sort.
    using TopKSort =
        hipcub::BlockRadixSort<float, kNumThreadsPerBlock, kNumTopKItemsPerThread, int>;

    // The number of slots for the final pass.
    static constexpr int kNumFinalItems = 3072;
    // The number of elements per thread for the final sort.
    static constexpr int kNumFinalItemsPerThread = kNumFinalItems / kNumThreadsPerBlock;
    // The class to sort the elements during the final pass.
    using FinalSort =
        hipcub::BlockRadixSort<float, kNumThreadsPerBlock, kNumFinalItemsPerThread, int>;

    // The class to compute the inclusive prefix-sum over the histogram.
    using Scan = hipcub::BlockScan<int, kNumThreadsPerBlock>;

    using VectorType = typename to_vector<Vector>::type;

    // Shared memory to compute the block scan.
    __shared__ typename Scan::TempStorage smemScan;

    // The structure to store the final items (for the final pass).
    struct FinalItems
    {
        // Shared memory to store the indices for the final pass.
        int indices[kNumFinalItems];
        // Shared memory to store the logits for the final pass.
        float logits[kNumFinalItems];
    };

    // Shared memory to compute the block sort.
    __shared__ union
    {
        FinalItems items;
        typename FinalSort::TempStorage finalSort;
        typename TopKSort::TempStorage topKSort;
    } smemFinal;

    // Shared memory to store the histogram.
    __shared__ int smemHistogram[kNumBins];
    // Shared memory to store the selected indices.
    __shared__ int smemIndices[kTopK];
    // Shared memory to store the threshold bin.
    __shared__ int smemThresholdBinIdx[1];
    // Shared memory counter to register the candidates for the final phase.
    __shared__ int smemFinalDstIdx[1];

    // The length of the row.
    int rowLen = rowEnd - rowStart;

    // Shortcut if the length of the row is smaller than Top-K. Indices are not
    // sorted by their corresponding logit.
    if(rowLen <= kTopK)
    {
        for(int rowIt = threadIdx.x; rowIt < rowLen; rowIt += kNumThreadsPerBlock)
        {
            int idx                            = rowStart + rowIt;
            outIndices[rowIdx * kTopK + rowIt] = idx - rowStart;
        }
        for(int rowIt = rowLen + threadIdx.x; rowIt < kTopK; rowIt += kNumThreadsPerBlock)
        {
            outIndices[rowIdx * kTopK + rowIt] = -1;
        }
        return;
    }

    // Clear the histogram.
    if(threadIdx.x < kNumBins)
    {
        smemHistogram[threadIdx.x] = 0;
    }

    // Make sure the histogram is ready.
    __syncthreads();

    // Fetch elements one-by-one.
    for(int rowIt = rowStart + threadIdx.x; rowIt < (rowEnd + Vector - 1) / Vector;
        rowIt += kNumThreadsPerBlock)
    {
        int64_t offset = ((int64_t)rowIdx) * (stride0 / Vector) + ((int64_t)rowIt) * stride1;
        auto v         = reinterpret_cast<const VectorType*>(logits)[offset];

#pragma unroll
        for(int j = 0; j < Vector; j++)
        {
            float logit  = (rowIt * Vector + j) < rowEnd ? v[j] : -INFINITY;
            uint16_t idx = extractBinIdx(logit);
            atomicAdd(&smemHistogram[idx], 1);
        }
    }

    // Make sure the histogram is ready.
    __syncthreads();

    // Read the values from SMEM.
    int binCount{0};
    if(threadIdx.x < kNumBins)
    {
        binCount = smemHistogram[threadIdx.x];
    }

    // Make sure each thread has read its value.
    __syncthreads();

    // Compute the prefix sum.
    int prefixSum{0}, totalSum{0};
    Scan(smemScan).ExclusiveSum(binCount, prefixSum, totalSum);

    // Update the histogram with the prefix sums.
    if(threadIdx.x < kNumBins)
    {
        smemHistogram[threadIdx.x] = prefixSum;
    }

    // Make sure the data is in shared memory.
    __syncthreads();

    // Find the last valid bin.
    if(threadIdx.x < kNumBins)
    {
        int nextPrefixSum = threadIdx.x == kNumBins - 1 ? totalSum : smemHistogram[threadIdx.x + 1];
        if(prefixSum < kTopK && nextPrefixSum >= kTopK)
        {
            smemThresholdBinIdx[0] = threadIdx.x;
        }
    }

    // Clear the counter to store the items for the final phase.
    if(threadIdx.x == 0)
    {
        smemFinalDstIdx[0] = 0;
    }

    // Make sure the data is in shared memory.
    __syncthreads();

    // The threshold bin.
    int thresholdBinIdx = smemThresholdBinIdx[0];

    // Fetch elements one-by-one and populate the shared memory buffers.
    for(int rowIt = rowStart + threadIdx.x; rowIt < (rowEnd + Vector - 1) / Vector;
        rowIt += kNumThreadsPerBlock)
    {
        int64_t offset = ((int64_t)rowIdx) * stride0 / Vector + ((int64_t)rowIt) * stride1;
        auto v         = reinterpret_cast<const VectorType*>(logits)[offset];

#pragma unroll
        for(auto j = 0; j < Vector; j++)
        {
            float logit = (rowIt * Vector + j) < rowEnd ? v[j] : -INFINITY;
            // float logit    = v[j];
            uint16_t idx = extractBinIdx(logit);
            if(idx < thresholdBinIdx)
            {
                int dstIdx          = atomicAdd(&smemHistogram[idx], 1);
                smemIndices[dstIdx] = Vector * rowIt + j;
            }
            else if(idx == thresholdBinIdx)
            {
                int dstIdx = atomicAdd(&smemFinalDstIdx[0], 1);
                if(dstIdx < kNumFinalItems)
                {
                    smemFinal.items.logits[dstIdx]  = logit;
                    smemFinal.items.indices[dstIdx] = Vector * rowIt + j;
                }
            }
        }
    }

    // Make sure the elements are in shared memory.
    // __syncthreads();

    // The logits of the elements to be sorted in the final pass.
    float finalLogits[kNumFinalItemsPerThread];
    // The indices of the elements to be sorted in the final pass.
    int finalIndices[kNumFinalItemsPerThread];

// Init.
#pragma unroll
    for(int ii = 0; ii < kNumFinalItemsPerThread; ++ii)
    {
        finalLogits[ii] = -FLT_MAX;
    }

    __syncthreads();

// Read the elements from SMEM.
#pragma unroll
    for(int ii = 0; ii < kNumFinalItemsPerThread; ++ii)
    {
        int srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
        if(srcIdx < smemFinalDstIdx[0])
        {
            finalLogits[ii]  = smemFinal.items.logits[srcIdx];
            finalIndices[ii] = smemFinal.items.indices[srcIdx];
        }
    }

    // Make sure the shared memory has been read.
    __syncthreads();

    // Sort the elements.
    FinalSort(smemFinal.finalSort).SortDescendingBlockedToStriped(finalLogits, finalIndices);

    // Copy the data back to the shared memory storage.
    int baseIdx = thresholdBinIdx > 0 ? smemHistogram[thresholdBinIdx - 1] : 0;
#pragma unroll
    for(int ii = 0; ii < kNumFinalItemsPerThread; ++ii)
    {
        int srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
        int dstIdx = baseIdx + srcIdx;
        if(dstIdx < kTopK)
        {
            smemIndices[dstIdx] = finalIndices[ii];
        }
    }

    // Make sure the data is in shared memory.
    __syncthreads();

// Store to global memory.
#pragma unroll
    for(int ii = 0; ii < kNumTopKItemsPerThread; ++ii)
    {
        int offset         = rowIdx * kTopK + ii * kNumThreadsPerBlock + threadIdx.x;
        outIndices[offset] = smemIndices[ii * kNumThreadsPerBlock + threadIdx.x] - rowStart;
    }
}

template <int kNumThreadsPerBlock = 512, int Vector = 4>
static __global__ void topk_per_row(const float* logits,
                                    const int* rowStarts,
                                    const int* rowEnds,
                                    int* outIndices,
                                    int stride0,
                                    int stride1)
{
    // The number of bins in the histogram.
    static constexpr int kNumBins = 512;

    // The top-k width.
    static constexpr int kTopK = 2048;

    // The row computed by this block.
    int rowIdx = blockIdx.x;

    // The range of logits within the row.
    int rowStart = rowStarts[rowIdx];
    int rowEnd   = rowEnds[rowIdx];

    topk_per_row_kernel<kNumThreadsPerBlock, kNumBins, kTopK, Vector>(
        logits, rowStart, rowEnd, rowIdx, outIndices, stride0, stride1);
}

template <int kNumThreadsPerBlock = 512, int Vector = 4>
static __global__ void topk_per_row_decode(
    const float* logits, const int* seqLens, int* outIndices, int stride0, int stride1, int next_n)
{
    // The number of bins in the histogram.
    static constexpr int kNumBins = kNumThreadsPerBlock;

    // The top-k width.
    static constexpr int kTopK = 2048;

    // The row computed by this block.
    int rowIdx = blockIdx.x;

    // The range of logits within the row.
    int rowStart = 0;
    int seq_len  = seqLens[rowIdx / next_n];
    int rowEnd   = seq_len - next_n + (rowIdx % next_n) + 1;

    topk_per_row_kernel<kNumThreadsPerBlock, kNumBins, kTopK, Vector>(
        logits, rowStart, rowEnd, rowIdx, outIndices, stride0, stride1);
}

} // namespace aiter

void topk_per_row(const torch::Tensor& logits,
                  const torch::Tensor& rowStarts,
                  const torch::Tensor& rowEnds,
                  torch::Tensor& indices,
                  int64_t numRows,
                  int64_t stride0,
                  int64_t stride1)
{
    // Compute the results on the device.
    constexpr int kNumThreadsPerBlock = 512;

    // The top-k width.
    static constexpr int kTopK = 2048;

    const hipStream_t stream = at::hip::getCurrentHIPStream();

    if(stride0 % 4 == 0)
        aiter::topk_per_row<kNumThreadsPerBlock, 4>
            <<<numRows, kNumThreadsPerBlock, 0, stream>>>(logits.data_ptr<float>(),
                                                          rowStarts.data_ptr<int>(),
                                                          rowEnds.data_ptr<int>(),
                                                          indices.data_ptr<int>(),
                                                          static_cast<int>(stride0),
                                                          static_cast<int>(stride1));
    else
        aiter::topk_per_row<kNumThreadsPerBlock, 1>
            <<<numRows, kNumThreadsPerBlock, 0, stream>>>(logits.data_ptr<float>(),
                                                          rowStarts.data_ptr<int>(),
                                                          rowEnds.data_ptr<int>(),
                                                          indices.data_ptr<int>(),
                                                          static_cast<int>(stride0),
                                                          static_cast<int>(stride1));
}

void topk_per_row_decode(const torch::Tensor& logits,
                         int64_t next_n,
                         const torch::Tensor& seqLens,
                         torch::Tensor& indices,
                         int64_t numRows,
                         int64_t stride0,
                         int64_t stride1)
{
    // Compute the results on the device.
    constexpr int kNumThreadsPerBlock = 1024;
    const hipStream_t stream          = at::hip::getCurrentHIPStream();

    if(stride0 % 4 == 0)
        aiter::topk_per_row_decode<kNumThreadsPerBlock, 4>
            <<<numRows, kNumThreadsPerBlock, 0, stream>>>(logits.data_ptr<float>(),
                                                          seqLens.data_ptr<int>(),
                                                          indices.data_ptr<int>(),
                                                          static_cast<int>(stride0),
                                                          static_cast<int>(stride1),
                                                          static_cast<int>(next_n));
    else
        aiter::topk_per_row_decode<kNumThreadsPerBlock, 1>
            <<<numRows, kNumThreadsPerBlock, 0, stream>>>(logits.data_ptr<float>(),
                                                          seqLens.data_ptr<int>(),
                                                          indices.data_ptr<int>(),
                                                          static_cast<int>(stride0),
                                                          static_cast<int>(stride1),
                                                          static_cast<int>(next_n));
}
