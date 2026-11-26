// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include <hip/hip_runtime.h>

struct Record {
    int expert, first, rows, segments;
};

struct PlanArgs {
    const int* offsets;
    int* counts;
    Record* table;
    int experts;
};

#if defined(COMPACT_PLAN)
extern "C" __global__ void opus_gemm_gfx950_bf16_latent_moe_bwd_plan(PlanArgs a) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= a.experts) return;
    const int first = a.offsets[e], rows = a.offsets[e+1] - first;
    a.counts[e] = rows;
    a.table[e] = {e, first, rows, 1 << 16};
}
#elif defined(DESCRIPTOR_PLAN)
struct MArgs {
    const __bf16 *dy, *weight;
    __bf16* dx;
    const int *rows, *offsets;
    long long expert_stride;
    int m, n, k, lda, ldb, ldc, batch_a, batch_c, groups;
};
struct alignas(128) Descriptor { MArgs args; int rows; };
static_assert(sizeof(MArgs) == 88 && sizeof(Descriptor) == 128);
struct DescriptorArgs {
    const __bf16 *dy, *weight;
    __bf16* dx;
    const int* offsets;
    Record* table;
    Descriptor* descriptors;
    int experts, n, k, capacity;
};
extern "C" __global__ __launch_bounds__(1024)
void opus_gemm_gfx950_bf16_m_grouped_256x256x256x64_2x2_16x16x32_0x0x0_nn_descriptor_plan(DescriptorArgs a) {
    int e = threadIdx.x;
    int first = e < a.experts ? a.offsets[e] : 0;
    int rows = e < a.experts ? a.offsets[e+1] - first : 0;
    int chunks = (rows + 255) / 256;
    __shared__ int ends[1024];
    ends[e] = chunks;
    __syncthreads();
    for (int d = 1; d < 1024; d *= 2) {
        int left = e >= d ? ends[e-d] : 0;
        __syncthreads();
        ends[e] += left;
        __syncthreads();
    }
    if (e < a.experts) {
        a.table[e] = {e, first, rows, 1 << 16};
        for (int j = 0; j < chunks; ++j) {
            int row = first + j*256, count = min(rows-j*256, 256);
            auto& desc = a.descriptors[ends[e]-chunks+j];
            desc.rows = count;
            desc.args = {a.dy + (long long)row*a.n, a.weight + (long long)e*a.n*a.k,
                         a.dx + (long long)row*a.k, &desc.rows, nullptr, (long long)a.n*a.k,
                         count, a.k, a.n, a.n, a.k, a.k, 0, 0, 1};
        }
    }
    for (int i = ends[1023] + e; i < a.capacity; i += 1024) a.descriptors[i].rows = 0;
}
#elif defined(SORT_PLAN)
struct SortArgs {
    const int* offsets;
    int* counts;
    Record* table;
    int experts, descending;
};
extern "C" __global__ __launch_bounds__(1024)
void opus_gemm_gfx950_bf16_k_grouped_256x256x256x64_2x2_16x16x32_0x0x0_tn_f32_sort_plan(SortArgs a) {
    const int t = threadIdx.x;
    unsigned key = ~0u;
    if (t < a.experts) {
        int rows = a.offsets[t+1] - a.offsets[t];
        a.counts[t] = rows;
        key = (unsigned(a.descending ? 140000-rows : rows) << 10) | t;
    }
    __shared__ unsigned exchange[1024];
    int size = 1;
    while (size < a.experts) size *= 2;
    for (int k = 2; k <= size; k *= 2) {
        for (int d = k/2; d; d /= 2) {
            unsigned peer;
            if (d < 64) {
                peer = __shfl_xor(key, d, 64);
            } else {
                exchange[t] = key;
                __syncthreads();
                peer = exchange[t ^ d];
                __syncthreads();
            }
            key = ((t & d) == 0) == ((t & k) == 0) ? min(key, peer) : max(key, peer);
        }
    }
    if (t < a.experts) {
        int e = key & 1023, first = a.offsets[e];
        a.table[t] = {e, first, a.offsets[e+1]-first, 1 << 16};
    }
}
#endif
