// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file test_tcopy_gfx1250.cu
 * @brief Device test for opus::tcopy_window with 2-slot LDS ping-pong on gfx1250.
 *
 * Kernel: 32x64xK WMMA GEMM driven by tcopy_window K-step.
 *   - 2-WG cluster (cluster_dims(2,1,1)); A multicast via SelectedWgs=<0,1>.
 *   - 4 producer waves load A/B tiles to LDS using tcopy_window::move() over K.
 *   - 4 consumer waves accumulate v_c via wmma_f16_16x16x32_f16.
 *   - LDS is double-buffered (2-deep ping-pong); s_wait_tensorcnt(1) keeps the
 *     prefetch in flight while the current tile is consumed.
 *
 * Named barriers / waveid / sync helpers come from opus/opus.hpp.
 */

#ifdef __HIP_DEVICE_COMPILE__
// ── Device pass ─────────────────────────────────────────────────────────────
#include "opus/opus.hpp"
#include "opus/hip_minimal.hpp"

#if defined(__gfx1250__)

// Fp16x16Packer: f16x16 <-> f16x8[2] view, for wmma_f16_16x16x32_f16 src/dst packing.
union Fp16x16Packer { opus::fp16x16_t vec16; opus::fp16x8_t vec8[2]; };

__global__ __launch_bounds__(256, 2) __cluster_dims__(2, 1, 1)
void tcopy_gfx1250_kernel(const void* __restrict__ ptr_a,
                          const void* __restrict__ ptr_b,
                          void* __restrict__       ptr_c,
                          int stride_a,
                          int stride_b,
                          int stride_c)
{
    using namespace opus;
    using opus::operator""_I;

    constexpr int     Block_K              = 128;
    constexpr int     Block_M              = 32;
    constexpr int     Block_N              = 32;
    constexpr int32_t consumerSubWarpNum   = 4;
    DECLARE_NAMED_BARRIERS();

    const int      wave_id                = static_cast<int>(waveid_in_workgroup());
    const int      sub_consumer_wave_id   = wave_id % 4;
    const uint32_t cluster_workgroup_id_x = __builtin_amdgcn_cluster_workgroup_id_x();
    const int32_t  c_cluster_offset_elems = static_cast<int32_t>(cluster_workgroup_id_x) * Block_N;

    const int K_STEPS = (stride_a + Block_K - 1) / Block_K;

    // LDS layout: 2 slots, each containing A (slot_bytes_A) + B (slot_bytes_A).
    constexpr int slot_bytes_A    = Block_M * (Block_K + 8) * static_cast<int>(sizeof(fp16_t));
    constexpr int slot_bytes_B    = slot_bytes_A;
    constexpr int slot_pair_bytes = slot_bytes_A + slot_bytes_B;
    constexpr int num_slots       = 2;

    __shared__ char Smem[num_slots * slot_pair_bytes];
    const uintptr_t smembase = reinterpret_cast<uintptr_t>(Smem);

    using NoSelectedWgs = seq<>;
    using WinB = tcopy_window<fp16_t, Block_K, 16, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 5, 3, NoSelectedWgs>;
    using SelectedWgs   = seq<0, 1>;
    using WinA = tcopy_window<fp16_t, Block_K, 16, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 1, 5, 3, SelectedWgs>;

    // dlds(ks): alternating ±slot_pair_bytes for ping-pong slot toggle.
    const auto dlds = [&](int ks) -> intptr_t {
        return (ks & 1) ? +static_cast<intptr_t>(slot_pair_bytes)
                        : -static_cast<intptr_t>(slot_pair_bytes);
    };

    // ────────────────────────────── producers ──────────────────────────────
    if (wave_id < 4) {
        if (wave_id < 2) {
            // Producer A (cluster multicast)
            sync_workgroup();

            WinA win_a;
            const uintptr_t wave_lds_off = static_cast<uintptr_t>(wave_id) * 16 * (Block_K + 8) * sizeof(fp16_t);
            win_a.make(smembase, ptr_a, wave_lds_off,
                       static_cast<uint32_t>(stride_a),
                       static_cast<uint32_t>(Block_M - wave_id * 16),
                       static_cast<uint64_t>(stride_a),
                       0, static_cast<uint32_t>(wave_id * 16));

            win_a.load_to_lds();
            if (K_STEPS > 1) {
                win_a.move(Block_K, 0_I, 0_I, 0_I, 0_I, /*lds=*/+slot_pair_bytes);
                win_a.load_to_lds();
            }

            for (int ks = 0; ks < K_STEPS; ++ks) {
                if (ks < K_STEPS - 1) __builtin_amdgcn_s_wait_tensorcnt(1);
                else                  __builtin_amdgcn_s_wait_tensorcnt(0);
                s_barrier_join_ptr(&__nbar_1);
                __builtin_amdgcn_s_barrier_signal(1);

                s_barrier_join_ptr(&__nbar_2);
                __builtin_amdgcn_s_barrier_wait(2);

                if (ks + 2 < K_STEPS) {
                    win_a.move(Block_K, 0_I, 0_I, 0_I, 0_I, /*lds=*/dlds(ks));
                    win_a.load_to_lds();
                }
            }
        } else {
            // Producer B (per-WG, no multicast)
            sync_workgroup();

            WinB win_b;
            const uintptr_t wave_lds_off = static_cast<uintptr_t>(wave_id - 2) * 16 * (Block_K + 8) * sizeof(fp16_t);
            const uint32_t  b_origin1    = static_cast<uint32_t>(cluster_workgroup_id_x * Block_N + (wave_id - 2) * 16);

            win_b.make(smembase + slot_bytes_A, ptr_b, wave_lds_off,
                       static_cast<uint32_t>(stride_b),
                       static_cast<uint32_t>(Block_N - (wave_id - 2) * 16),
                       static_cast<uint64_t>(stride_b),
                       0, b_origin1);

            win_b.load_to_lds();
            if (K_STEPS > 1) {
                win_b.move(Block_K, 0_I, 0_I, 0_I, 0_I, /*lds=*/+slot_pair_bytes);
                win_b.load_to_lds();
            }

            for (int ks = 0; ks < K_STEPS; ++ks) {
                if (ks < K_STEPS - 1) __builtin_amdgcn_s_wait_tensorcnt(1);
                else                  __builtin_amdgcn_s_wait_tensorcnt(0);
                s_barrier_join_ptr(&__nbar_1);
                __builtin_amdgcn_s_barrier_signal(1);

                s_barrier_join_ptr(&__nbar_2);
                __builtin_amdgcn_s_barrier_wait(2);

                if (ks + 2 < K_STEPS) {
                    win_b.move(Block_K, 0_I, 0_I, 0_I, 0_I, /*lds=*/dlds(ks));
                    win_b.load_to_lds();
                }
            }
        }
    }
    // ────────────────────────────── consumers ──────────────────────────────
    else {
        s_barrier_init_ptr(&__nbar_1, 4);
        s_barrier_init_ptr(&__nbar_2, 4);
        sync_workgroup();

        constexpr int32_t AKSldPack    = 16 / static_cast<int32_t>(sizeof(fp16_t));
        constexpr int32_t AKSldLane    = 16 / AKSldPack;
        constexpr int32_t AMSldLane    = get_warp_size() / AKSldLane;
        constexpr int32_t AMSldRepeat  = Block_M / (AMSldLane * consumerSubWarpNum / 2);
        constexpr int32_t AKSldRepeat  = Block_K / (AKSldPack * AKSldLane);
        static_assert(AKSldLane * AMSldLane == get_warp_size(), "A sld lane product");
        constexpr int32_t SMemKPitch   = Block_K + 8;

        auto block_sld_shape_a  = make_tuple(number<AMSldRepeat>{}, number<consumerSubWarpNum / 2>{}, number<AKSldRepeat>{}, number<AKSldLane>{}, number<AMSldLane>{}, number<AKSldPack>{});
        auto block_sld_stride_a = make_tuple(AMSldLane * SMemKPitch * consumerSubWarpNum / 2, AMSldLane * SMemKPitch, AKSldPack * AKSldLane, AKSldPack, SMemKPitch, 1_I);
        auto block_sld_win_a    = make_layout<0>(block_sld_shape_a, block_sld_stride_a);

        constexpr int32_t BSldKPack   = 16 / static_cast<int32_t>(sizeof(fp16_t));
        constexpr int32_t BSldKLane   = 16 / BSldKPack;
        constexpr int32_t BSldNLane   = get_warp_size() / BSldKLane;
        constexpr int32_t BSldNRepeat = Block_N / (BSldNLane * consumerSubWarpNum / 2);
        constexpr int32_t BSldKRepeat = Block_K / (BSldKPack * BSldKLane);
        static_assert(BSldKLane * BSldNLane == get_warp_size(), "B sld lane product");

        auto block_sld_shape_b  = make_tuple(number<BSldNRepeat>{}, number<consumerSubWarpNum / 2>{}, number<BSldKRepeat>{}, number<BSldKLane>{}, number<BSldNLane>{}, number<BSldKPack>{});
        auto block_sld_stride_b = make_tuple(BSldNLane * SMemKPitch * consumerSubWarpNum / 2, BSldNLane * SMemKPitch, BSldKPack * BSldKLane, BSldKPack, SMemKPitch, 1_I);
        auto block_sld_win_b    = make_layout<0>(block_sld_shape_b, block_sld_stride_b);

        const int32_t sub_consumer_wave_m = sub_consumer_wave_id / 2;
        const int32_t sub_consumer_wave_n = sub_consumer_wave_id % 2;
        const int32_t lid                 = static_cast<int32_t>(lane_id());
        const int32_t a_lane_m            = lid / AMSldLane;
        const int32_t a_lane_n            = lid % AMSldLane;
        const int32_t b_lane_m            = lid / BSldNLane;
        const int32_t b_lane_n            = lid % BSldNLane;

        fp16x8_t v_c = {.0f};

        constexpr int KtileElems  = 32;
        static_assert(Block_K % KtileElems == 0, "Block_K must be multiple of 32");
        constexpr int K_WmmaTiles = Block_K / KtileElems;

        for (int ks = 0; ks < K_STEPS; ++ks) {
            s_barrier_join_ptr(&__nbar_1);
            __builtin_amdgcn_s_barrier_wait(1);

            const int32_t slot_offset = (ks & 1) * slot_pair_bytes;

            #pragma unroll
            for (int kt = 0; kt < K_WmmaTiles; ++kt) {
                const int32_t kr0 = 2 * kt;
                const int32_t kr1 = kr0 + 1;

                const int32_t a_sld_os0 = slot_offset
                    + block_sld_win_a(0_I, sub_consumer_wave_m, kr0, a_lane_m, a_lane_n, 0_I) * static_cast<int32_t>(sizeof(fp16_t));
                const int32_t a_sld_os1 = slot_offset
                    + block_sld_win_a(0_I, sub_consumer_wave_m, kr1, a_lane_m, a_lane_n, 0_I) * static_cast<int32_t>(sizeof(fp16_t));
                const int32_t b_sld_os0 = slot_offset + slot_bytes_A
                    + block_sld_win_b(0_I, sub_consumer_wave_n, kr0, b_lane_m, b_lane_n, 0_I) * static_cast<int32_t>(sizeof(fp16_t));
                const int32_t b_sld_os1 = slot_offset + slot_bytes_A
                    + block_sld_win_b(0_I, sub_consumer_wave_n, kr1, b_lane_m, b_lane_n, 0_I) * static_cast<int32_t>(sizeof(fp16_t));

                fp16x8_t sld_a0, sld_a1, sld_b0, sld_b1;
                asm volatile(
                    "ds_read_b128 %[a0], %[a_os0]\n\t"
                    "ds_read_b128 %[a1], %[a_os1]\n\t"
                    "ds_read_b128 %[b0], %[b_os0]\n\t"
                    "ds_read_b128 %[b1], %[b_os1]\n\t"
                    : [a0]"=v"(sld_a0), [a1]"=v"(sld_a1),
                      [b0]"=v"(sld_b0), [b1]"=v"(sld_b1)
                    : [a_os0]"v"(a_sld_os0), [a_os1]"v"(a_sld_os1),
                      [b_os0]"v"(b_sld_os0), [b_os1]"v"(b_sld_os1)
                    : "memory");
                asm volatile("" : : "v"(a_sld_os0), "v"(a_sld_os1), "v"(b_sld_os0), "v"(b_sld_os1) : "memory");
                asm volatile("s_wait_dscnt(0)" ::: "memory");

                Fp16x16Packer convertA = __builtin_bit_cast(Fp16x16Packer, array<fp16x8_t, 2>{sld_a0, sld_a1});
                Fp16x16Packer convertB = __builtin_bit_cast(Fp16x16Packer, array<fp16x8_t, 2>{sld_b0, sld_b1});

                __builtin_amdgcn_sched_barrier(0);
                v_c = __builtin_amdgcn_wmma_f16_16x16x32_f16(0, convertB.vec16, 0, convertA.vec16, 0, v_c, false, false);
            }

            s_barrier_join_ptr(&__nbar_2);
            __builtin_amdgcn_s_barrier_signal(2);
        }

        // C store (gfx942-style, adapted for RDNA4 wave32 WMMA).
        constexpr int32_t CGstNPack = 8;
        constexpr int32_t CGstNLane = 2;
        constexpr int32_t CGstMLane = 16;

        auto block_gmem_gst_shape_c  = make_tuple(number<consumerSubWarpNum / 2>{}, number<consumerSubWarpNum / 2>{}, number<CGstNLane>{}, number<CGstMLane>{}, number<CGstNPack>{});
        auto block_gmem_gst_stride_c = make_tuple(CGstMLane * stride_c, CGstNPack * CGstNLane, CGstNPack, stride_c, 1_I);
        auto block_gmem_gst_win_c    = make_layout<0>(block_gmem_gst_shape_c, block_gmem_gst_stride_c);

        int32_t c_offset_elem = block_gmem_gst_win_c(sub_consumer_wave_m, sub_consumer_wave_n, lid / 16, lid % 16, 0_I) + c_cluster_offset_elems;
        *(reinterpret_cast<fp16x8_t*>(reinterpret_cast<fp16_t*>(ptr_c) + c_offset_elem)) = v_c;
    }
}

#else  // !__gfx1250__
__global__ void tcopy_gfx1250_kernel(const void*, const void*, void*, int, int, int) {}
#endif // __gfx1250__

#else
// ── Host pass ───────────────────────────────────────────────────────────────
#include "opus/hip_minimal.hpp"
#include <cstdio>

#define HIP_CALL(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %d at %s:%d\n", (int)err, __FILE__, __LINE__); \
        return; \
    } \
} while(0)

__global__ void tcopy_gfx1250_kernel(const void*, const void*, void*, int, int, int) {}

extern "C" void run_tcopy_gfx1250(const void* d_a, const void* d_b, void* d_c,
                                  int stride_a, int stride_b, int stride_c)
{
    // 2 WGs forming a cluster (cluster_dims(2,1,1)); each WG has 256 threads = 8 waves.
    hipLaunchKernelGGL(tcopy_gfx1250_kernel,
                       dim3(2, 1, 1), dim3(256), 0, 0,
                       d_a, d_b, d_c, stride_a, stride_b, stride_c);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}
#endif // __HIP_DEVICE_COMPILE__
