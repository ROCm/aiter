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

static inline __device__ uint32_t floatAsSortableUint(float x)
{
    uint32_t bits = __float_as_uint(x);
    bits          = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;
    return bits;
}

template <int step>
static inline __device__ uint32_t extractBinIdx(float x)
{
    uint32_t bits = floatAsSortableUint(x);

    if constexpr(step == 0)
    {
        return bits >> 21;
    }
    else if constexpr(step == 1)
    {
        return (bits >> 10) & 0x7ff;
    }
    else
    {
        return bits & 0x3ff;
    }
}

template <int shift>
static inline __device__ bool isPartialMatch(float x, uint32_t pattern)
{
    if constexpr(shift == 0)
    {
        return true;
    }
    uint32_t bits = floatAsSortableUint(x);
    return (bits ^ pattern) >> shift == 0;
}

template <int step,
          int kNumThreadsPerBlock,
          int kNumBins,
          int kTopK,
          int kNumFinalItems,
          int Vector,
          typename SmemFinalType>
__device__ bool processHistogramStep(const float* logits,
                                     int rowEnd,
                                     uint32_t& logitPattern,
                                     int& thresholdBinIdx,
                                     int* smemHistogram,
                                     int* smemIndices,
                                     int* smemThresholdBinIdx,
                                     int* smemFinalDstIdx,
                                     int* smemFinalBinSize,
                                     int* smemFoundTopKValues,
                                     SmemFinalType& smemFinal,
                                     int stride1,
                                     int rowStart)
{
    using VectorType = typename to_vector<Vector>::type;
    // Clear the histogram.
#pragma unroll
    for(int idx = threadIdx.x; idx < kNumBins; idx += kNumThreadsPerBlock)
    {
        smemHistogram[idx] = 0;
    }

    // Make sure the histogram is ready.
    __syncthreads();

    // Update pattern
    constexpr auto patternShift = step == 0 ? 0 : step == 1 ? 21 : 10;
    if constexpr(step == 1)
    {
        logitPattern = static_cast<uint32_t>(thresholdBinIdx & 0x7ff) << patternShift;
    }
    else if constexpr(step == 2)
    {
        logitPattern |= static_cast<uint32_t>(thresholdBinIdx & 0x7ff) << patternShift;
    }

    // Fetch elements one-by-one.
    for(int vecIdx = (rowStart / Vector) + threadIdx.x; vecIdx < (rowEnd + Vector - 1) / Vector;
        vecIdx += kNumThreadsPerBlock)
    {
        auto v         = reinterpret_cast<const VectorType*>(logits)[vecIdx];
#pragma unroll
        for(int j = 0; j < Vector; j++)
        {
            int vIdx = vecIdx * Vector + j;
            if(vIdx >= rowEnd)
                break;
            float logit = v[j];
            if(isPartialMatch<patternShift>(logit, logitPattern))
            {
                uint32_t binIdx = extractBinIdx<step>(logit);
                atomicAdd(&smemHistogram[binIdx], 1);
            }
        }
    }

    // Make sure the histogram is ready.
    __syncthreads();

    // Reads the value of the starting position in the smemIndices array
    int lastValue = smemFoundTopKValues[0];

    for(int round = 0; round < kNumBins / kNumThreadsPerBlock; round++)
    {
        // Read the values from SMEM.
        int idx = threadIdx.x + kNumThreadsPerBlock * round;
        int binCount{0};
        binCount = smemHistogram[idx];

        // Make sure each thread has read its value.
        __syncthreads();

        // Compute the prefix sum.
        int prefixSum{0}, totalSum{0};
        using Scan = hipcub::BlockScan<int, kNumThreadsPerBlock>;
        Scan(smemFinal.smemScan).ExclusiveSum(binCount, prefixSum, totalSum);

        // Update the histogram with the prefix sums.
        prefixSum += lastValue;
        totalSum += lastValue;
        smemHistogram[idx] = prefixSum;

        // Make sure the data is in shared memory.
        __syncthreads();

        // Find the last valid bin.
        bool foundThreshold = false;
        if(prefixSum < kTopK)
        {
            int nextPrefixSum =
                threadIdx.x == kNumThreadsPerBlock - 1 ? totalSum : smemHistogram[idx + 1];

            if(nextPrefixSum >= kTopK)
            {
                smemThresholdBinIdx[0] = idx;
                smemFinalBinSize[0]    = nextPrefixSum - prefixSum;
                smemFoundTopKValues[0] = prefixSum;
                foundThreshold         = true;
            }
        }

        // Early exit: if any thread found the threshold, we can skip remaining
        // rounds
        if(__syncthreads_or(foundThreshold))
        {
            break;
        }

        lastValue = totalSum;
    }

    // Make sure the data is in shared memory.
    __syncthreads();

    // The threshold bin.
    thresholdBinIdx = smemThresholdBinIdx[0];

    // Fetch elements one-by-one and populate the shared memory buffers.
    for(int vecIdx = (rowStart / Vector) + threadIdx.x; vecIdx < (rowEnd + Vector - 1) / Vector;
        vecIdx += kNumThreadsPerBlock)
    {
        // Compute the vector offset for coalesced VectorType load
        auto v = reinterpret_cast<const VectorType*>(logits)[vecIdx];
#pragma unroll
        for(int j = 0; j < Vector; j++)
        {
            int vIdx = vecIdx * Vector + j;
            if(vIdx >= rowEnd)
                break;
            float logit = v[j];

            // Check for pattern match
            if(!isPartialMatch<patternShift>(logit, logitPattern))
                continue;

            uint32_t binIdx = extractBinIdx<step>(logit);

            if(binIdx < thresholdBinIdx)
            {
                int dstIdx          = atomicAdd(&smemHistogram[binIdx], 1);
                smemIndices[dstIdx] = vIdx;
            }

            if constexpr(step < 2)
            {
                // Fill final items only if threshold bin fits
                if(binIdx == thresholdBinIdx && smemFinalBinSize[0] <= kNumFinalItems)
                {
                    int dstIdx                      = atomicAdd(&smemFinalDstIdx[0], 1);
                    smemFinal.items.logits[dstIdx]  = logit;
                    smemFinal.items.indices[dstIdx] = vIdx;
                }
            }
            else
            {
                if(binIdx == thresholdBinIdx)
                {
                    int dstIdx = atomicAdd(&smemHistogram[binIdx], 1);
                    if(dstIdx < kTopK)
                    {
                        smemIndices[dstIdx] = vIdx;
                    }
                }
            }
        }
    }

    // Make sure the elements are in shared memory.
    __syncthreads();

    // Check if we should continue to next step
    return smemFinalBinSize[0] > kNumFinalItems;
}

template <int kNumThreadsPerBlock        = 512,
          int kNumBins                   = 512,
          int kTopK                      = 2048,
          bool useRadixSort              = true,
          int Vector                     = 4,
          bool sortResultLogitDescending = false>
__device__ void topk_per_row_kernel(const float* logits,
                                    const int rowStart,
                                    const int rowEnd,
                                    int* outIndices,
                                    int stride1)
{
    // The number of slots for the final pass.
    static constexpr int kNumFinalItems = 2048;
    // The number of elements per thread for the final sort.
    static constexpr int kNumFinalItemsPerThread = kNumFinalItems / kNumThreadsPerBlock;
    // The class to sort the elements during the final pass.
    using FinalSort =
        hipcub::BlockRadixSort<float, kNumThreadsPerBlock, kNumFinalItemsPerThread, int>;

    // The class to compute the inclusive prefix-sum over the histogram.
    using Scan = hipcub::BlockScan<int, kNumThreadsPerBlock>;

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
        typename Scan::TempStorage smemScan;
    } smemFinal;

    // Shared memory to store the histogram.
    __shared__ int smemHistogram[kNumBins];
    // Shared memory to store the selected indices.
    __shared__ int smemIndices[kTopK];
    // Shared memory to store the threshold bin.
    __shared__ int smemThresholdBinIdx[1];
    // Shared memory counter to register the candidates for the final phase.
    __shared__ int smemFinalDstIdx[1];
    // Shared memory to determine if the threshold bin fits in the final items.
    __shared__ int smemFinalBinSize[1];
    // Shared memory to keep track of the top-k values found so far by the
    // previous iterations
    __shared__ int smemFoundTopKValues[1];

    // The length of the row.
    int rowLen = rowEnd - rowStart;

    // Shortcut if the length of the row is smaller than Top-K. Indices are not
    // sorted by their corresponding logit.
    if(rowLen <= kTopK)
    {
        for(int rowIt = threadIdx.x; rowIt < rowLen; rowIt += kNumThreadsPerBlock)
        {
            outIndices[rowIt] = rowIt - rowStart;
        }
        for(int rowIt = rowLen + threadIdx.x; rowIt < kTopK; rowIt += kNumThreadsPerBlock)
        {
            outIndices[rowIt] = -1;
        }
        return;
    }

    // Initialize values
    if(threadIdx.x == 0)
    {
        smemFinalDstIdx[0]     = 0;
        smemFoundTopKValues[0] = 0;
    }
    __syncthreads();
    int thresholdBinIdx   = -1;
    uint32_t logitPattern = 0;

    // Step 0: Process first 11 bits
    bool continueToNextStep =
        processHistogramStep<0, kNumThreadsPerBlock, kNumBins, kTopK, kNumFinalItems, Vector>(
            logits,
            rowEnd,
            logitPattern,
            thresholdBinIdx,
            smemHistogram,
            smemIndices,
            smemThresholdBinIdx,
            smemFinalDstIdx,
            smemFinalBinSize,
            smemFoundTopKValues,
            smemFinal,
            stride1,
            rowStart);

    if(continueToNextStep)
    {
        // Step 1: Process next 11 bits
        continueToNextStep =
            processHistogramStep<1, kNumThreadsPerBlock, kNumBins, kTopK, kNumFinalItems, Vector>(
                logits,
                rowEnd,
                logitPattern,
                thresholdBinIdx,
                smemHistogram,
                smemIndices,
                smemThresholdBinIdx,
                smemFinalDstIdx,
                smemFinalBinSize,
                smemFoundTopKValues,
                smemFinal,
                stride1,
                rowStart);

        if(continueToNextStep)
        {
            // Step 2: Process final 10 bits
            processHistogramStep<2, kNumThreadsPerBlock, kNumBins, kTopK, kNumFinalItems, Vector>(
                logits,
                rowEnd,
                logitPattern,
                thresholdBinIdx,
                smemHistogram,
                smemIndices,
                smemThresholdBinIdx,
                smemFinalDstIdx,
                smemFinalBinSize,
                smemFoundTopKValues,
                smemFinal,
                stride1,
                rowStart);
        }
    }

    if(!continueToNextStep)
    {
        // The histogram did not proceed to the final 10 bits, therefore we need to
        // sort the final items The logits of the elements to be sorted in the final
        // pass.
        if constexpr(useRadixSort)
        {
            // Sorting with radix sort
            float finalLogits[kNumFinalItemsPerThread];
            // The indices of the elements to be sorted in the final pass.
            int finalIndices[kNumFinalItemsPerThread];

#pragma unroll
            for(int ii = 0; ii < kNumFinalItemsPerThread; ++ii)
            {
                finalLogits[ii] = -FLT_MAX;
            }

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
            FinalSort(smemFinal.finalSort)
                .SortDescendingBlockedToStriped(finalLogits, finalIndices);

            // Copy the data back to the shared memory storage.
            int baseIdx = smemFoundTopKValues[0];

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
        }
        else
        {
            // Sorting with insertion sort
            auto baseIdx = smemFoundTopKValues[0];
            for(int i = threadIdx.x; i < smemFinalDstIdx[0]; i += kNumThreadsPerBlock)
            {
                int outIndex = 0;
                auto logit   = smemFinal.items.logits[i];
                for(int j = 0; j < smemFinalDstIdx[0]; j++)
                {
                    auto otherLogit = smemFinal.items.logits[j];
                    if(logit < otherLogit || (logit == otherLogit && i < j))
                    {
                        outIndex++;
                    }
                }
                // Store if outIndex is in bounds
                if(outIndex + baseIdx < kTopK)
                {
                    smemIndices[outIndex + baseIdx] = smemFinal.items.indices[i];
                }
            }
        }
        __syncthreads();
    }

    if constexpr(sortResultLogitDescending)
    {
        // Sorting with radix sort
        float finalLogits[kNumFinalItemsPerThread];
        // The indices of the elements to be sorted in the final pass.
        int finalIndices[kNumFinalItemsPerThread];

// Read the elements from SMEM.
#pragma unroll
        for(int ii = 0; ii < kNumFinalItemsPerThread; ++ii)
        {
            int srcIdx       = ii * kNumThreadsPerBlock + threadIdx.x;
            const auto index = smemIndices[srcIdx];
            const auto logit = logits[index * stride1];
            finalLogits[ii]  = logit;
            finalIndices[ii] = index;
        }

        // Make sure the shared memory has been read.
        __syncthreads();

        // Sort the elements.
        FinalSort(smemFinal.finalSort).SortDescendingBlockedToStriped(finalLogits, finalIndices);

        // Store to global memory
#pragma unroll
        for(int ii = 0; ii < kNumFinalItemsPerThread; ++ii)
        {
            int srcIdx         = ii * kNumThreadsPerBlock + threadIdx.x;
            outIndices[srcIdx] = finalIndices[ii] - rowStart;
        }
    }

    if constexpr(!sortResultLogitDescending)
    {
        // Store to global memory.
#pragma unroll
        for(int i = threadIdx.x; i < kTopK; i += kNumThreadsPerBlock)
        {
            outIndices[i] = smemIndices[i] - rowStart;
        }
    }
}

template <int kNumThreadsPerBlock = 512, bool useRadixSort = true, int Vector = 4>
static __global__ void topk_per_row(const float* logits,
                                    const int* rowStarts,
                                    const int* rowEnds,
                                    int* outIndices,
                                    int stride0,
                                    int stride1,
                                    int rowOffset)
{
    // The number of bins in the histogram.
    static constexpr int kNumBins = 2048;

    // The top-k width.
    static constexpr int kTopK = 2048;

    // The row computed by this block.
    int64_t rowIdx = static_cast<int64_t>(blockIdx.x) + rowOffset;

    // The range of logits within the row.
    int rowStart = rowStarts[rowIdx];
    int rowEnd   = rowEnds[rowIdx];

    // Local pointers to this block
    auto outIndicesLocal = outIndices + rowIdx * kTopK;
    auto logitsLocal     = logits + rowIdx * stride0;

    topk_per_row_kernel<kNumThreadsPerBlock, kNumBins, kTopK, useRadixSort, Vector>(
        logitsLocal, rowStart, rowEnd, outIndicesLocal, stride1);
}

template <int kNumThreadsPerBlock = 512, bool useRadixSort = true, int Vector = 4>
static __global__ void topk_per_row_decode(
    const float* logits, const int* seqLens, int* outIndices, int stride0, int stride1, int next_n)
{
    // The number of bins in the histogram.
    static constexpr int kNumBins = 2048;

    // The top-k width.
    static constexpr int kTopK = 2048;

    // The row computed by this block.
    int64_t rowIdx = static_cast<int64_t>(blockIdx.x);

    // The range of logits within the row.
    int rowStart = 0;
    int seq_len  = seqLens[rowIdx / next_n];
    int rowEnd   = seq_len - next_n + (rowIdx % next_n) + 1;

    // Local pointers to this block
    auto outIndicesLocal = outIndices + rowIdx * kTopK;
    auto logitsLocal     = logits + rowIdx * stride0;

    topk_per_row_kernel<kNumThreadsPerBlock, kNumBins, kTopK, useRadixSort, Vector>(
      logitsLocal, rowStart, rowEnd, outIndicesLocal, stride1);
}

} // namespace aiter

void top_k_per_row_prefill(const torch::Tensor& logits,
                  const torch::Tensor& rowStarts,
                  const torch::Tensor& rowEnds,
                  torch::Tensor& indices,
                  int64_t numRows,
                  int64_t stride0,
                  int64_t stride1)
{
    constexpr int kSortingAlgorithmThreshold = 12288;

    // Compute the results on the device.
    constexpr int kNumThreadsPerBlock = 512;

    // The top-k width.
    static constexpr int kTopK = 2048;

    const hipStream_t stream = at::hip::getCurrentHIPStream();

    int numInsertionBlocks = std::min(static_cast<int>(numRows), kSortingAlgorithmThreshold);

    if(stride0 % 4 == 0)
    {
        aiter::topk_per_row<kNumThreadsPerBlock, false, 4>
            <<<numInsertionBlocks, kNumThreadsPerBlock, 0, stream>>>(logits.data_ptr<float>(),
                                                                     rowStarts.data_ptr<int>(),
                                                                     rowEnds.data_ptr<int>(),
                                                                     indices.data_ptr<int>(),
                                                                     static_cast<int>(stride0),
                                                                     static_cast<int>(stride1),
                                                                     0);
    }
    else
    {
        aiter::topk_per_row<kNumThreadsPerBlock, false, 1>
            <<<numInsertionBlocks, kNumThreadsPerBlock, 0, stream>>>(logits.data_ptr<float>(),
                                                                     rowStarts.data_ptr<int>(),
                                                                     rowEnds.data_ptr<int>(),
                                                                     indices.data_ptr<int>(),
                                                                     static_cast<int>(stride0),
                                                                     static_cast<int>(stride1),
                                                                     0);
    }

    if(numRows > kSortingAlgorithmThreshold)
    {
        int numRadixBlocks = numRows - kSortingAlgorithmThreshold;
        if(stride0 % 4 == 0)
        {
            aiter::topk_per_row<kNumThreadsPerBlock, true, 4>
                <<<numRadixBlocks, kNumThreadsPerBlock, 0, stream>>>(logits.data_ptr<float>(),
                                                                     rowStarts.data_ptr<int>(),
                                                                     rowEnds.data_ptr<int>(),
                                                                     indices.data_ptr<int>(),
                                                                     static_cast<int>(stride0),
                                                                     static_cast<int>(stride1),
                                                                     kSortingAlgorithmThreshold);
        }
        else
        {
            aiter::topk_per_row<kNumThreadsPerBlock, true, 1>
                <<<numRadixBlocks, kNumThreadsPerBlock, 0, stream>>>(logits.data_ptr<float>(),
                                                                     rowStarts.data_ptr<int>(),
                                                                     rowEnds.data_ptr<int>(),
                                                                     indices.data_ptr<int>(),
                                                                     static_cast<int>(stride0),
                                                                     static_cast<int>(stride1),
                                                                     kSortingAlgorithmThreshold);
        }
    }
}

void top_k_per_row_decode(const torch::Tensor& logits,
                         int64_t next_n,
                         const torch::Tensor& seqLens,
                         torch::Tensor& indices,
                         int64_t numRows,
                         int64_t stride0,
                         int64_t stride1)
{
    constexpr int kSortingAlgorithmThreshold = 12288;
    // Compute the results on the device.
    constexpr int kNumThreadsPerBlock = 1024;
    const hipStream_t stream          = at::hip::getCurrentHIPStream();
    const auto numColumns = logits.size(1);

    if(numColumns < kSortingAlgorithmThreshold)
    {
        if(stride0 % 4 == 0)
        {
            aiter::topk_per_row_decode<kNumThreadsPerBlock, false, 4>
                <<<numRows, kNumThreadsPerBlock, 0, stream>>>(logits.data_ptr<float>(),
                                                              seqLens.data_ptr<int>(),
                                                              indices.data_ptr<int>(),
                                                              static_cast<int>(stride0),
                                                              static_cast<int>(stride1),
                                                              static_cast<int>(next_n));
        }
        else
        {
            aiter::topk_per_row_decode<kNumThreadsPerBlock, false, 1>
                <<<numRows, kNumThreadsPerBlock, 0, stream>>>(logits.data_ptr<float>(),
                                                              seqLens.data_ptr<int>(),
                                                              indices.data_ptr<int>(),
                                                              static_cast<int>(stride0),
                                                              static_cast<int>(stride1),
                                                              static_cast<int>(next_n));
        }
    }
    else
    {
        if (stride0 % 4 == 0)
        {
            aiter::topk_per_row_decode<kNumThreadsPerBlock, true, 4>
                <<<numRows, kNumThreadsPerBlock, 0, stream>>>(logits.data_ptr<float>(),
                                                              seqLens.data_ptr<int>(),
                                                              indices.data_ptr<int>(),
                                                              static_cast<int>(stride0),
                                                          static_cast<int>(stride1),
                                                          static_cast<int>(next_n));
    } else {
        aiter::topk_per_row_decode<kNumThreadsPerBlock, true, 1>
            <<<numRows, kNumThreadsPerBlock, 0, stream>>>(logits.data_ptr<float>(),
                                                          seqLens.data_ptr<int>(),
                                                          indices.data_ptr<int>(),
                                                          static_cast<int>(stride0),
                                                          static_cast<int>(stride1),
                                                          static_cast<int>(next_n));
    }
}
}
