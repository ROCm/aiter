// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx1250's MXFP4 MQA-logits instances and their registry entries.
//
// The ONLY translation unit built with -fno-honor-nans (optCompilerConfig.json's per-source
// flags): the kernels' relu is max_num with the NaN put back from the scales
// (MQA_LOGITS_MXFP4_NUM_RELU, mqa_logits_mxfp4_gfx1250_common.cuh), and the flag cannot be scoped
// to the relu. Rerun the op test's NaN / Inf rows after every compiler upgrade: the flag lets the
// compiler assume no NaN.
//
// Also built with `-mllvm -enable-post-misched=1`, overriding aiter's default of 0: without the
// post-RA scheduler the V4.1 prefill kernel ran 2.3% slower and the candidate decode kernel up to
// 0.5 us slower than the same source built standalone (cold A/B; V4 and V4.1 decode unaffected).
#include "mqa_logits_mxfp4.h"
#include "mqa_logits_mxfp4_gfx1250_traits.h"

// The device pass on gfx1250 gets the kernels; every other pass gets empty stubs so the launch
// symbols resolve. Both halves of the guard are load-bearing: the host pass reports warp size 64.
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1250__)
#include "mqa_logits_mxfp4_gfx1250_h64.cuh"
#include "mqa_logits_mxfp4_gfx1250_v41.cuh"
#else
namespace opus_logits::gfx1250 {
template<class T, int SCHED> __global__ void h64_kernel(mqa_logits_mxfp4_kargs) {}
template<class T, int SCHED> __global__ void v41_kernel(mqa_logits_mxfp4_kargs) {}
}  // namespace opus_logits::gfx1250
#endif

namespace opus_logits::gfx1250 {
namespace {

// The shipped instances (opus-ops opus_logits/common/docs/MQA_LOGITS_INTERFACE_DESIGN.md §3.2
// has the selection and the measurements behind it).
using h64_q4s2T  = h64_traits<4, 2, 1>;           // V4 prefill, MTP > 1 decode
using h64_q1s2T  = h64_traits<1, 2, 1>;           // V4 MTP-1 decode
using v41_r1w1s2L = v41_traits<1, 1, 2, 0, 0>;    // V4.1 full-layer MTP-1 decode (Free)
using v41_r3w2s2T = v41_traits<3, 2, 2, 1, 0>;    // V4.1 full-layer MTP > 1 decode (Free)
using v41_r4w2s2T = v41_traits<4, 2, 2, 1, 0>;    // V4.1 full-layer prefill (Prefill)
using v41g_s4L    = v41_traits<1, 1, 4, 0, 1>;    // V4.1 candidate decode (Prefill)
using v41g_s3L    = v41_traits<1, 1, 3, 0, 1>;    // V4.1 candidate prefill (Prefill)

template<class T>
auto h64_kernel_of(int mode) -> void (*)(mqa_logits_mxfp4_kargs) {
    switch (mode) {
    case SCHED_TABLE:   return h64_kernel<T, SCHED_TABLE>;
    case SCHED_FREE:    return h64_kernel<T, SCHED_FREE>;
    case SCHED_PREFILL: return h64_kernel<T, SCHED_PREFILL>;
    default:            return nullptr;
    }
}

// The kernel of `mode`, or null. Free is not compiled for the candidate kernel (mqa_assign says
// why).
template<class T>
auto v41_kernel_of(int mode) -> void (*)(mqa_logits_mxfp4_kargs) {
    switch (mode) {
    case SCHED_TABLE:   return v41_kernel<T, SCHED_TABLE>;
    case SCHED_PREFILL: return v41_kernel<T, SCHED_PREFILL>;
    case SCHED_FREE:
        if constexpr (T::GATHER) return nullptr;
        else return v41_kernel<T, SCHED_FREE>;
    default:            return nullptr;
    }
}

template<class T, auto KERNEL_OF>
hipError_t launch(int mode, const mqa_logits_mxfp4_kargs& k, int grid, hipStream_t stream) {
    const auto kernel = KERNEL_OF(mode);
    if (!kernel) return hipErrorInvalidValue;
    void* args[] = {const_cast<mqa_logits_mxfp4_kargs*>(&k)};
    return hipLaunchKernel(reinterpret_cast<const void*>(kernel), dim3(grid), dim3(T::BLOCK_SIZE),
                           args, 0, stream);
}

template<class T, auto KERNEL_OF>
hipError_t occupancy(int mode, int* ctas_per_cu) {
    const auto kernel = KERNEL_OF(mode);
    if (!kernel) return hipErrorInvalidValue;
    return hipOccupancyMaxActiveBlocksPerMultiprocessor(ctas_per_cu, kernel, T::BLOCK_SIZE, 0);
}

constexpr int MODES_ALL     = (1 << SCHED_TABLE) | (1 << SCHED_FREE) | (1 << SCHED_PREFILL);
constexpr int MODES_GATHER  = (1 << SCHED_TABLE) | (1 << SCHED_PREFILL);
constexpr int ENTRY_ANY_RUN = 1 << 30;   // any power-of-two multiple of the 64-key tile

template<class T>
constexpr mqa_logits_instance h64_entry() {
    // A table entry is one 64-key tile: 64 rows (page_size * pages_per_block).
    return {"gfx1250", T::N_HEADS, 1, T::NUM_WAVES, T::STAGES, T::SCALE_TDM, 0,
            T::KV_TILE_SIZE, T::KV_TILE_SIZE, T::KV_TILE_SIZE, MODES_ALL, 0,
            launch<T, h64_kernel_of<T>>, occupancy<T, h64_kernel_of<T>>};
}

template<class T>
constexpr mqa_logits_instance v41_entry() {
    // Full layers: an entry is a run of 2^tpb_shift tiles. Candidate layers: an 8-row page.
    return {"gfx1250", T::N_HEADS, T::ROWS_PER_WAVE, T::NUM_WAVES, T::STAGES, T::SCALE_TDM, T::GATHER,
            T::GATHER ? T::PAGE_ROWS : T::TILE_KEYS, T::GATHER ? T::PAGE_ROWS : ENTRY_ANY_RUN,
            T::TILE_KEYS, T::GATHER ? MODES_GATHER : MODES_ALL, 1,
            launch<T, v41_kernel_of<T>>, occupancy<T, v41_kernel_of<T>>};
}

}  // namespace
}  // namespace opus_logits::gfx1250

namespace opus_logits {

const mqa_logits_instance* mqa_logits_gfx1250_instances(int* count) {
    using namespace gfx1250;
    static const mqa_logits_instance table[] = {
        h64_entry<h64_q4s2T>(),
        h64_entry<h64_q1s2T>(),
        v41_entry<v41_r1w1s2L>(),
        v41_entry<v41_r3w2s2T>(),
        v41_entry<v41_r4w2s2T>(),
        v41_entry<v41g_s4L>(),
        v41_entry<v41g_s3L>(),
    };
    *count = static_cast<int>(sizeof(table) / sizeof(table[0]));
    return table;
}

}  // namespace opus_logits
