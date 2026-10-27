// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx950's MXFP4 MQA-logits instances and their registry entries: DeepSeek-V4's 64 index heads
// on 64-row pages (V4.1's 32 heads run FlyDSL's row-group kernel on gfx950).
//
// Built with `-mllvm -enable-post-misched=1` (optCompilerConfig.json's per-source flags),
// overriding aiter's default of 0: the one-row kernel's MFMA / VALU interleave depends on it.
#include "mqa_logits_mxfp4.h"
#include "mqa_logits_mxfp4_gfx950_traits.h"

// The device pass on gfx950 gets the kernels; every other pass gets empty stubs so the launch
// symbols resolve.
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
#include "mqa_logits_mxfp4_gfx950_kv.cuh"
#include "mqa_logits_mxfp4_gfx950_qshare.cuh"
#else
namespace opus_logits::gfx950 {
template<class T, int SCHED> __global__ void kv_kernel(mqa_logits_mxfp4_kargs) {}
template<class T, int SCHED> __global__ void qshare_kernel(mqa_logits_mxfp4_kargs) {}
}  // namespace opus_logits::gfx950
#endif

namespace opus_logits::gfx950 {
namespace {

// The shipped instances (opus-ops opus_logits/common/docs/MQA_LOGITS_INTERFACE_DESIGN.md has the
// selection and the measurements behind it). Six- and eight-row qshare CTAs (qshare_traits<6 / 8>)
// were 20-50% behind these at their own row counts: one such CTA fills a CU, so its waves stall
// at the per-phase barrier together, where two or four smaller CTAs a CU overlap theirs.
using kv64      = mqa_logits_traits<64, 1>;    // decode, 1 row a sequence; prefill
using kv256     = mqa_logits_traits<256, 4>;   // aiter's former qlen1_kv256 (no tier picks it)
using qshare_q2 = qshare_traits<2>;            // decode, 2 rows a sequence; 5-6 (three groups)
using qshare_q4 = qshare_traits<4>;            // decode, 3-4 rows; 7-8 (two groups)

template<class T>
auto kv_kernel_of(int mode) -> void (*)(mqa_logits_mxfp4_kargs) {
    switch (mode) {
    case SCHED_TABLE:   return kv_kernel<T, SCHED_TABLE>;
    case SCHED_FREE:    return kv_kernel<T, SCHED_FREE>;
    case SCHED_PREFILL: return kv_kernel<T, SCHED_PREFILL>;
    default:            return nullptr;
    }
}

template<class T>
auto qshare_kernel_of(int mode) -> void (*)(mqa_logits_mxfp4_kargs) {
    switch (mode) {
    case SCHED_TABLE:   return qshare_kernel<T, SCHED_TABLE>;
    case SCHED_FREE:    return qshare_kernel<T, SCHED_FREE>;
    case SCHED_PREFILL: return qshare_kernel<T, SCHED_PREFILL>;
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

constexpr int MODES_ALL = (1 << SCHED_TABLE) | (1 << SCHED_FREE) | (1 << SCHED_PREFILL);

// A table entry is one 64-row page (pages_per_block 1): the MFMA-permuted kv_cache / kv_scale are
// laid out a page at a time. A 256-key tile spans four entries.
template<class T>
constexpr mqa_logits_instance kv_entry() {
    // One row a CTA whatever its waves (they split the tile): rows_per_wave * waves == 1.
    return {"gfx950", T::N_HEADS, 1, 1, 2, 0, 0, T::PAGE_SIZE, T::PAGE_SIZE, T::KV_TILE_SIZE,
            MODES_ALL, 0, launch<T, kv_kernel_of<T>>, occupancy<T, kv_kernel_of<T>>};
}

template<class T>
constexpr mqa_logits_instance qshare_entry() {
    // scale_tdm 1: the key scales come in by an async copy with the tile, as gfx1250's TDM.
    return {"gfx950", T::N_HEADS, 1, T::NUM_WAVES, T::LDS_STAGES, 1, 0, T::PAGE_SIZE, T::PAGE_SIZE,
            T::KV_TILE_SIZE, MODES_ALL, 0,
            launch<T, qshare_kernel_of<T>>, occupancy<T, qshare_kernel_of<T>>};
}

}  // namespace
}  // namespace opus_logits::gfx950

namespace opus_logits {

const mqa_logits_instance* mqa_logits_gfx950_instances(int* count) {
    using namespace gfx950;
    static const mqa_logits_instance table[] = {
        kv_entry<kv64>(),
        kv_entry<kv256>(),
        qshare_entry<qshare_q2>(),
        qshare_entry<qshare_q4>(),
    };
    *count = static_cast<int>(sizeof(table) / sizeof(table[0]));
    return table;
}

}  // namespace opus_logits
