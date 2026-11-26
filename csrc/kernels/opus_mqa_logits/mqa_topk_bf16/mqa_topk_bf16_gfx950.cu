// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx950's mqa_topk_bf16 instances: their traits and registry entries. The scoring kernels are in
// mqa_topk_bf16_gfx950_scan.cuh, the fallback in mqa_topk_bf16_gfx950_fallback.cuh.
#include "mqa_topk_bf16.h"

namespace opus_mqa_topk::gfx950 {

// WAVES waves of 4 rows a workgroup. walk_lds (logits, emit): the waves share a ring of STAGES
// 64-key stages; walk (sample, fallback): each wave walks on its own.
template<int WAVES_, int STAGES_ = 4>
struct scan_traits {
    static constexpr int WAVES  = WAVES_;
    static constexpr int BLOCK  = WAVES * 64;
    static constexpr int STAGES = STAGES_;
};

}  // namespace opus_mqa_topk::gfx950

// The device pass on gfx950 gets the kernels; every other pass gets empty stubs so the launch
// symbols resolve.
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
#include "mqa_topk_bf16_gfx950_fallback.cuh"
#include "mqa_topk_bf16_gfx950_scan.cuh"
#else
namespace opus_mqa_topk::gfx950 {
template<class T, int STAGE> __global__ void scan_kernel(mqa_topk_bf16_kargs) {}
template<class T> __global__ void fallback_kernel(mqa_topk_bf16_kargs) {}
template<class T> __global__ void emit_kernel(mqa_topk_bf16_kargs) {}
}  // namespace opus_mqa_topk::gfx950
#endif

namespace opus_mqa_topk::gfx950 {
namespace {

using w8s4 = scan_traits<8, 4>;   // logits, emit: 32 rows a CU, 64 KB of stages
using w4   = scan_traits<4>;      // sample, fallback

template<class T, class W>
constexpr mqa_topk_instance entry() {
    using K = instance_kernels<T::BLOCK, W::BLOCK, scan_kernel<T, STAGE_LOGITS>,
                               scan_kernel<T, STAGE_SAMPLE>, emit_kernel<T>,
                               fallback_kernel<W>>;
    return {.arch          = "gfx950",
            .rows_per_wave = 4,
            .waves         = T::WAVES,
            .sample_waves  = T::WAVES,
            .tile_keys     = 16,
            .launch        = K::launch,
            .occupancy     = K::occupancy};
}

}  // namespace
}  // namespace opus_mqa_topk::gfx950

namespace opus_mqa_topk {

const mqa_topk_instance* mqa_topk_gfx950_instances(int* count) {
    static const mqa_topk_instance table[] = {
        gfx950::entry<gfx950::w8s4, gfx950::w4>(),
    };
    *count = static_cast<int>(sizeof(table) / sizeof(table[0]));
    return table;
}

}  // namespace opus_mqa_topk
