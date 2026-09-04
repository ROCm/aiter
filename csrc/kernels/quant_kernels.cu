// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

// This translation unit is torch-free: define AITER_NO_TORCH_TYPES before any
// aiter header so aiter_opus_plus.h does not pull in the c10 half/bfloat16
// headers. The kernels use aiter::hip2opus + the _rmTorch dispatch macros, never
// the t2opus<c10::*> specializations, so nothing here needs torch/ATen/c10.
#define AITER_NO_TORCH_TYPES
#include "aiter_hip_common.h"
#include "aiter_dispatch.h"
#include "aiter_opus_plus.h"
#include "aiter_stream.h"
#include "gemm_dispatch_utils.h"
#include "quant.h"
#include "mx_quant_utils.h"
#include "rocprim/rocprim.hpp"


const int32_t BlockSize           = 256;
const int32_t groupQuantBlockSize = 64;

namespace aiter {

// Host-side twin of the kernel's kTunedForThisArch. Cached: get_gpu_arch() queries the
// device properties, and this sits on the launch path of every quant call.
static inline bool dyn_gq_tuned_arch()
{
    static const bool tuned = (get_gpu_arch() == "gfx1250");
    return tuned;
}

#ifndef DYN_GQ_TDM_MIN_BLK_PER_SIMD
#define DYN_GQ_TDM_MIN_BLK_PER_SIMD 6
#endif

#ifndef DYN_GQ_TDM_KPT
#define DYN_GQ_TDM_KPT 2
#endif

// emit_e8m0_scale = false (default): legacy behaviour — fp4 outputs an e8m0
// byte scale, fp8 / i8 output a continuous fp32 per-group scale.
//
// emit_e8m0_scale = true (opt-in for fp8): compute a power-of-2 per-group
// scale via f32_to_e8m0_scale and write a single E8M0 byte per group,
// matching the fp4 byte layout. Used by the MXFP8 "split" path
// `per_1x32_mx_quant_hip(quant_dtype=fp8, scale_type=fp8_e8m0)` so the
// produced byte scale is directly consumable by `mxfp4_moe_sort_hip` /
// MXFP8 GEMM kernels without a post-hoc fp32 -> e8m0 conversion.
// Everything below that is a TUNING choice rather than a correctness one is gated on
// gfx1250, because gfx1250 is where it was measured. The group ordering, the 32B chunk
// split and the per-layout block size were each picked off a sweep on that part, and each
// leans on something arch-specific -- wave32, b128 as the widest per-lane access, that
// part's cache-line and dispatch behaviour. None of it was measured on gfx950 (wave64,
// different load widths), so that target keeps exactly the shape it had before.
template <typename DTYPE_I, typename DTYPE_O, int thread_data_size = 32, int32_t group_size = 128, bool shuffle_scale = true, int32_t block_size = 64, bool emit_e8m0_scale = false, bool enable_tdm = false>
__global__ void __launch_bounds__(block_size)
dynamic_per_group_scaled_quant_kernel(DTYPE_O* __restrict__ out,
                                      float* __restrict__ scale,
                                      DTYPE_I const* __restrict__ input,
                                      float const* __restrict__ scale_ub,
                                      int64_t ori_rows,
                                      int32_t ori_cols,
                                      int32_t ori_row_stride,
                                      int64_t oob_size,
                                      int32_t const* __restrict__ num_rows = nullptr,
                                      const int32_t num_cols_factor        = 1)
{
    static_assert(!emit_e8m0_scale
                      || std::is_same_v<DTYPE_O, opus::fp4_t>
                      || std::is_same_v<DTYPE_O, opus::fp8_t>,
                  "emit_e8m0_scale is only valid for fp4 / fp8 outputs");

    // fp4 always emits e8m0 byte scale (no fp32-scale variant exists today);
    // fp8 emits e8m0 byte scale iff caller opted in via emit_e8m0_scale.
    static constexpr bool use_e8m0_scale =
        std::is_same_v<DTYPE_O, opus::fp4_t> || emit_e8m0_scale;

    if(num_rows != nullptr)
    {
        ori_rows = static_cast<int64_t>(*num_rows) * num_cols_factor;
    }
    static constexpr int num_thread_per_group = group_size / thread_data_size;
    const int lane_in_group  = threadIdx.x % num_thread_per_group;
    // Reused at the end as the scale-store index, which the shuffled layouts rewrite.
    int64_t groupId          = static_cast<int64_t>(blockIdx.x) * block_size / num_thread_per_group
                               + threadIdx.x / num_thread_per_group;
    int32_t scaleN           = ori_cols / group_size;
    // Shuffle tiles e8m0 bytes 8-wide along scaleN for the MX hardware
    // scale-load layout (group_size == 32 only); group_size == 128 shuffle
    // is a plain transpose and needs no padding.
    int32_t scaleN_pad       = (use_e8m0_scale && shuffle_scale && group_size == 32)
                                   ? (((scaleN + 7) / 8) * 8)
                                   : scaleN;
    // Group ordering follows whichever stream is the scattered one.
    //
    // The scale store is one byte (or float) per group from a single lane, so its access
    // pattern is decided entirely by how consecutive groups map to (x, y). Row-major order
    // (consecutive groups walk y within a row) suits the plain `scale[groupId]` layout, but
    // the transposed layout writes `y * ori_rows + x`, which then puts consecutive lanes
    // `ori_rows` bytes apart -- one byte touched per cache line. Measured on gfx1250 at
    // [16384, 7168] that scattering cost 4.8 us, 15% of the kernel, to move 0.9 MB.
    //
    // Column-major order (consecutive groups walk x at a fixed y) makes the transposed
    // store contiguous instead. It strides the data reads across rows in exchange, but each
    // group still reads one contiguous 256B run, so DRAM absorbs it: the fp32-scale variant
    // measures unchanged either way. Net for the transposed path: 36.3 -> 32.0 us.
#if defined(__gfx1250__)
    static constexpr bool kTunedForThisArch = true;
#else
    static constexpr bool kTunedForThisArch = false;
#endif
    static constexpr bool kColumnMajorGroups =
        kTunedForThisArch && use_e8m0_scale && shuffle_scale && group_size == 128;

    // TDM staging parameters. Only the column-major path qualifies: it is the one whose
    // wave-sized run of groups forms a 2D tile (consecutive rows at one y). Row-major
    // groups walk y, which would make a wave's slice a strided set of partial rows.
    static constexpr int kGroupsPerWave     = WARP_SIZE / num_thread_per_group;
    static constexpr int kTdmWaves          = block_size / WARP_SIZE;
    // Tiles a wave stages before consuming any. Measured on gfx1250 at [16384, 7168],
    // e8m0 + transposed, kernel time:
    //     KPT   1(off)    2      3      4      8
    //     us     27.0   25.96  29.3   32.1   54.8
    // 2 wins and everything deeper loses badly: the whole per-group body is unrolled once
    // per tile, so the instruction count grows with KPT (344 -> 1795 at KPT 4) and buys
    // nothing once two loads are already in flight.
    static constexpr int kTdmKPT            = DYN_GQ_TDM_KPT;
    static constexpr int kTdmRing           = 3;
    static constexpr int kTdmGroupsPerStep  = kTdmWaves * kGroupsPerWave;
    static constexpr int kTdmGroupsPerBlock = kTdmGroupsPerStep * kTdmKPT;
    static constexpr int kTdmSlotElems      = kTdmGroupsPerStep * group_size;
    static constexpr int kTdmLdsWaves       = 1;   // one shared staging area per block
    // The host owns this decision, because the two paths need different grids: a staged
    // block covers kTdmGroupsPerBlock groups, an unstaged one covers kTdmGroupsPerStep.
    // Deciding it here instead would let the host size the grid for staging while the
    // kernel fell back, and half the groups would never be visited.
    static constexpr bool kUseTdmShape =
        enable_tdm && kColumnMajorGroups && kTdmKPT > 1 && kTdmWaves >= 1 &&
        (block_size % WARP_SIZE) == 0;

#if defined(__gfx1250__)
    // One descriptor per BLOCK: the tile is every wave's rows at once. tensorcnt retires
    // per wave, so the issuing wave is the only one that can wait on it and the others
    // have to be released by a barrier -- the trade is half the descriptors and a 2x
    // wider tile against two barriers per stage.
    using TdmWindow = opus::tdm<DTYPE_I, opus::seq<group_size, kTdmGroupsPerStep>>;
#endif

    // All four lanes of a group share a group id, hence the same (x, y), so the DPP reduce
    // below never reads across an active/inactive lane boundary.
    auto resolve = [&](int64_t gid, int64_t& gx, int32_t& gy) -> bool {
        if constexpr(kColumnMajorGroups)
        {
            gx = gid % ori_rows;
            gy = static_cast<int32_t>(gid / ori_rows);
        }
        else
        {
            gx = gid / scaleN_pad;
            gy = static_cast<int32_t>(gid % scaleN_pad);
        }
        if constexpr(use_e8m0_scale)
            return gx < ori_rows && gy < scaleN;
        else
            return gx < ori_rows;
    };
    int64_t x;
    int32_t y;

    using vec_i = opus::vector_t<DTYPE_I, thread_data_size>;
    static constexpr int32_t vec_size_o =
        std::is_same_v<DTYPE_O, opus::fp4_t> ? thread_data_size / 2 : thread_data_size;
    // The non-e8m0 (continuous fp32-scale) path uses the exact 1/DTYPE_MAX
    // divisor. The e8m0 path instead derives a power-of-2 scale via
    // fp_f32_to_e8m0_scale<> below (which folds in / max_pos), so it does not
    // use this divisor.
    const float inverted_DTYPE_MAX =
        (1. / static_cast<float>(opus::finfo<DTYPE_O>::max()));

    // How a group's elements are split across its threads.
    //
    // The obvious split -- thread t takes the contiguous run [t*64B, +64B) -- strides the
    // lanes of a load 64B apart, so one b128 pulls 16B out of each of 32 separate 64B
    // segments and nothing merges; the matching store covers 16B of every 32B. Handing each
    // thread two 32B chunks instead (chunks t and t+ntpg) halves both strides: a load
    // covers 16B of every 32B and the store becomes fully contiguous across a group.
    //
    // 32B is the sweet spot, not 16B. b128 is the widest per-lane access gfx1250 has (no
    // b256), so a 16B chunk would make the loads perfectly contiguous but shrink the fp8
    // store to 8B per lane -- 4x buffer_store_b64 instead of 2x b128. Measured, that trade
    // lost: the store side gave back more than the load side gained.
    //
    // The elements a thread holds are no longer contiguous within the group, but nothing
    // downstream cares: amax is order-independent, and store_vector's interleave mode lays
    // the output back down in exactly this pattern.
    static constexpr int kChunkElems = 32 / static_cast<int>(sizeof(DTYPE_I));
    static constexpr int kChunks     = kChunkElems ? thread_data_size / kChunkElems : 0;
    static constexpr bool kInterleavedChunks =
        kTunedForThisArch && !std::is_same_v<DTYPE_O, opus::fp4_t> && sizeof(DTYPE_I) == 2 &&
        kChunkElems > 0 && thread_data_size % kChunkElems == 0 && kChunks > 1;

    using chunk_t = opus::vector_t<DTYPE_I, kChunkElems>;

    // Gather this thread's slice of one group out of a contiguous base pointer, which is
    // either the group's address in global memory or its staging area in LDS.
    auto gather = [&](DTYPE_I const* gbase) -> vec_i {
        vec_i td;
        if constexpr(kInterleavedChunks)
        {
            auto const* chunks = reinterpret_cast<chunk_t const*>(gbase);
            opus::static_for<kChunks>([&](auto i) {
                chunk_t c = chunks[lane_in_group + i.value * num_thread_per_group];
                opus::static_for<kChunkElems>(
                    [&](auto j) { td[i.value * kChunkElems + j.value] = c[j.value]; });
            });
        }
        else
        {
            td = reinterpret_cast<vec_i const*>(gbase)[lane_in_group];
        }
        return td;
    };

    auto process = [&](vec_i thread_data, int64_t x, int32_t y, int64_t groupId) {
    float absMax      = 1e-10f;
    for(size_t j = 0; j < thread_data_size; j++)
    {
        absMax = max(absMax, abs(static_cast<float>(thread_data[j])));
    }
    absMax = multithread_reduce(absMax, aiter::Max(), num_thread_per_group);

    // MX e8m0 path: use the project-wide default round mode
    // (``kDefaultMxScaleRoundMode``, currently RoundUp = NV / DSv4 RCEIL).
    // The helper returns the dequant scale (e.g. ceil_pow2(amax/max_pos))
    // directly, so the (>>23)&0xFF extraction yields the e8m0 byte. fp4
    // always e8m0; fp8 only when emit_e8m0_scale (use_e8m0_scale gates this).
    // rmode is shared across fp4/fp8; only the dtype constant differs.
    float inverted_scale;
    if constexpr (use_e8m0_scale)
    {
        constexpr aiter::MxDtype kMxDtype =
            std::is_same_v<DTYPE_O, opus::fp4_t>
                ? aiter::MxDtype::FP4_E2M1
#if defined(__gfx942__)
                : aiter::MxDtype::FP8_E4M3_FNUZ;
#else
                : aiter::MxDtype::FP8_E4M3;
#endif
        inverted_scale =
            aiter::fp_f32_to_e8m0_scale<aiter::kDefaultMxScaleRoundMode, kMxDtype>(absMax);
    }
    else
    {
        inverted_scale = absMax * inverted_DTYPE_MAX;
    }
    // The output is always laid out row-major, whatever order the groups were walked in,
    // so index it by (x, y) rather than by the raw linear group id -- under
    // kColumnMajorGroups that id counts down a column and would scatter the data store.
    const int64_t out_group = kColumnMajorGroups ? (x * scaleN_pad + y) : groupId;
    // Interleaved layout: this thread's first chunk sits at lane * kChunkElems, and
    // store_vector's interleave mode strides the rest by num_thread_per_group chunks.
    static constexpr int kLaneStride = kInterleavedChunks ? kChunkElems : vec_size_o;
    const int64_t row_offset = std::is_same_v<DTYPE_O, opus::fp4_t>
                               ? out_group * group_size / 2 + lane_in_group * kLaneStride
                               : out_group * group_size + lane_in_group * kLaneStride;
    // The scale write is deferred to the very end of the kernel, but its ADDRESS and VALUE
    // are resolved here, before the conversion. Leaving that arithmetic in the tail put it
    // right after the data stores, where it reused their address and data VGPRs and forced
    // the compiler to guard them with `s_wait_xcnt 0x0` -- a wait for every outstanding
    // VMEM to read its operands. An ATT capture charged 18% of the kernel's total latency
    // to that single wait. Resolved up front, the tail is a bare store with nothing after
    // it to clobber, at a cost of three VGPRs held live across the conversion.
    //
    // A null `scale_dst` doubles as the "not the group's first lane" predicate.
    const float row_scale = inverted_scale;
    using scale_elem_t    = std::conditional_t<use_e8m0_scale, uint8_t, float>;
    scale_elem_t* scale_dst = nullptr;
    scale_elem_t  scale_val{};
    if(lane_in_group == 0)
    {
        int64_t scale_idx = groupId;
        if constexpr(shuffle_scale)
        {
            if constexpr(use_e8m0_scale && group_size == 32)
                scale_idx = aiter::mx_scale_shuffle_idx(scaleN_pad, static_cast<int>(x), y);
            else
                scale_idx = y * ori_rows + x;
        }
        if constexpr(use_e8m0_scale)
        {
            scale_dst = reinterpret_cast<uint8_t*>(scale) + scale_idx;
            scale_val = static_cast<uint8_t>(
                (__builtin_bit_cast(uint32_t, row_scale) >> 23) & 0b11111111);
        }
        else
        {
            scale_dst = scale + scale_idx;
            scale_val = row_scale;
        }
    }
    // Which form of the scale the store path consumes:
    //   - fp4 takes `row_scale` directly, via `cvt_scalef32_pk_fp4_f32`.
    //   - kStoreTakesDivisor: `scaled_cast_div` also takes `row_scale` directly, which lets
    //     gfx1250 lower bf16 -> fp8 to `v_cvt_scalef32_pk8_fp8_bf16` -- 8 elements per
    //     instruction, replacing the per-element fma_mix + med3 and per-pair cvt_pk chain
    //     (90 instructions -> 4 at thread_data_size 32) and the fp32 divide below with it.
    //   - everything else does software `input * inv_scale`, so it needs the reciprocal.
    //
    // use_e8m0_scale is a correctness precondition here, not a tuning gate: that hardware
    // convert reads its scale operand as an MX E8M0 factor, keeping the exponent bits and
    // discarding the mantissa, so it is exact iff the row scale is a power of two. The
    // continuous `absMax * inv_DTYPE_MAX` scale is not, and gets silently truncated to the
    // enclosing power of two -- measured 1.79x off, with the large codes saturating to NaN.
    //
    // Conversely the reciprocal must stay gated on the store form and not on
    // use_e8m0_scale: gating it that way once skipped the reciprocal for fp8 + e8m0 on the
    // software path and produced fp8 bytes ~2x off (`split_elem_err ≈ 100%`).
    static constexpr bool kStoreTakesDivisor =
        use_e8m0_scale && std::is_same_v<DTYPE_O, opus::fp8_t> &&
        std::is_same_v<DTYPE_I, opus::bf16_t> && (thread_data_size % 8 == 0);
    if constexpr(!std::is_same_v<DTYPE_O, opus::fp4_t> && !kStoreTakesDivisor)
    {
        inverted_scale = 1.0f / inverted_scale;
    }

    using DTYPE_STORE = std::conditional_t<std::is_same_v<DTYPE_O, opus::fp4_t>, uint8_t, DTYPE_O>;
    auto* out_ptr     = reinterpret_cast<DTYPE_STORE*>(out);
    auto buffer_o = opus::make_gmem<DTYPE_STORE>(out_ptr, oob_size);

    if constexpr(kInterleavedChunks)
    {
        store_vector<DTYPE_STORE, DTYPE_I, thread_data_size, RT, true, num_thread_per_group,
                     kChunks, DTYPE_O, kStoreTakesDivisor>(
            buffer_o, thread_data, row_offset, inverted_scale);
    }
    else
    {
        store_vector<DTYPE_STORE, DTYPE_I, thread_data_size, RT, false, WARP_SIZE, 1, DTYPE_O,
                     kStoreTakesDivisor>(buffer_o, thread_data, row_offset, inverted_scale);
    }

    // Scale write, deferred to last on purpose: address and value were resolved above, so
    // this is a bare store with no arithmetic after it competing for the data stores'
    // registers. See the note at `scale_dst`.
    if(scale_dst != nullptr)
    {
        *scale_dst = scale_val;
    }
    };   // process

#if defined(__gfx1250__)
    if constexpr(kUseTdmShape)
    {
        // Each wave stages its own groups through LDS with tensor_load_to_lds and keeps a
        // ring of them in flight. tensorcnt retires per wave, so no barrier is needed: a
        // wave only ever reads what it issued.
        //
        // kTdmRing is 3 because the hardware allows exactly 3 tensor ops in flight per wave
        // (opus.hpp, "3 tensor ops in flight per wave, 6 per SIMD"). The sequence below
        // sits at that cap and never above it: priming issues 3, and each iteration retires
        // one (the staged wait) before issuing one. Raising the ring past 3 would stall the
        // issue instead of deepening the pipeline.
        //
        // The tile is the whole reason this works: under column-major order a wave's
        // kGroupsPerWave groups are that many CONSECUTIVE rows at one y, i.e. a plain 2D
        // region -- dim0 = group_size contiguous elements, dim1 = the rows, dim0 stride =
        // ori_row_stride. One descriptor moves it, and the hardware picks the transaction
        // sizes instead of us paying the 50% per-instruction coverage of a b128 whose
        // lanes sit 32B apart.
        __shared__ DTYPE_I tdm_lds[kTdmLdsWaves * kTdmRing * kTdmSlotElems];

        const int wave_id  = threadIdx.x / WARP_SIZE;
        const int lane_grp = (threadIdx.x % WARP_SIZE) / num_thread_per_group;
        DTYPE_I* my_lds    = tdm_lds;
        const int slot_grp = wave_id * kGroupsPerWave + lane_grp;   // group within the tile
        const bool issuer  = (wave_id == 0);

        // Tile t of this wave starts at group base + t * kTdmGroupsPerStep.
        const int64_t block_g0 = static_cast<int64_t>(blockIdx.x) * kTdmGroupsPerBlock;
        const int64_t win_g0  = block_g0;                       // one window for the block
        const int64_t wave_g0 = block_g0 + wave_id * kGroupsPerWave;
        int64_t x0;
        int32_t y0;
        resolve(win_g0, x0, y0);

        auto w = opus::make_tdm<TdmWindow>(
            static_cast<u32_t>(reinterpret_cast<u64_t>(my_lds)), input,
            static_cast<u32_t>(ori_cols), static_cast<u32_t>(ori_rows),
            static_cast<u64_t>(ori_row_stride),
            static_cast<u32_t>(y0 * group_size), static_cast<u32_t>(x0));

        // Prime the ring, then for each tile wait only for ITS load. The count left
        // outstanding is what is still in flight behind it -- waiting on <0> instead
        // serialises the ring and gives back most of the gain.
        if(issuer)
        {
            opus::static_for<kTdmRing < kTdmKPT ? kTdmRing : kTdmKPT>([&](auto i) {
                if constexpr(i.value > 0) w.move(0, kTdmGroupsPerStep);
                w.async_load(static_cast<u32_t>((i.value % kTdmRing) * kTdmSlotElems));
            });
        }

        opus::static_for<kTdmKPT>([&](auto k) {
            constexpr int issued    = (k.value + kTdmRing) < kTdmKPT ? (k.value + kTdmRing)
                                                                     : kTdmKPT;
            constexpr int remaining = issued - (k.value + 1);
            if(issuer)
                opus::s_wait_tensorcnt<(remaining < 0 ? 0 : remaining)>();
            __builtin_amdgcn_s_barrier();   // publish the tile to the non-issuing waves

            const int64_t gid = wave_g0 + static_cast<int64_t>(k.value) * kTdmGroupsPerStep
                                + lane_grp;
            int64_t gx;
            int32_t gy;
            if(resolve(gid, gx, gy))
            {
                DTYPE_I const* slot = my_lds + (k.value % kTdmRing) * kTdmSlotElems
                                    + slot_grp * group_size;
                process(gather(slot), gx, gy, gid);
            }

            if constexpr(k.value + kTdmRing < kTdmKPT)
            {
                // The refill lands in the slot this iteration just read -- (k + ring) % ring
                // == k % ring -- and the DMA writes LDS asynchronously, which the compiler
                // does not model as aliasing the ds_reads above. Without this the refill can
                // overwrite the slot while those reads are still outstanding: a
                // write-after-read race that happens to survive today only because
                // `process` has a long tail of stores after its last LDS read.
                opus::s_wait_dscnt<0>();
                __builtin_amdgcn_s_barrier();   // all waves done reading before the refill
                if(issuer)
                { w.move(0, kTdmGroupsPerStep); w.async_load(
                    static_cast<u32_t>(((k.value + kTdmRing) % kTdmRing) * kTdmSlotElems)); }
            }
        });
        return;
    }
#endif

    if(!resolve(groupId, x, y))
        return;
    process(gather(input + x * ori_row_stride + y * group_size), x, y, groupId);
}

__global__ void initializeScale(float *d_data, int size, float value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        d_data[idx] = value;
    }
}

template <typename DTYPE_I, typename DTYPE_O, int thread_data_size = 16>
__device__ std::tuple<float, DTYPE_I*> data_to_per_row_scale(const DTYPE_I* __restrict__ input,
                                                             const int32_t cols)
{
    static constexpr int32_t vec_size_i =
        thread_data_size == 0 ? 16 / sizeof(DTYPE_O) : thread_data_size;
    static constexpr int32_t vec_size_o =
        std::is_same_v<DTYPE_O, opus::fp4_t> ? vec_size_i / 2 : vec_size_i;
    static constexpr int32_t load_chunk_bytes = sizeof(DTYPE_I) * vec_size_i % 16 == 0 ? 16 : (sizeof(DTYPE_I) * vec_size_i % 8 == 0 ? 8 : 4);
    using vec_i = opus::vector_t<DTYPE_I, vec_size_i>;
    const float inverted_DTYPE_MAX =
        (1. / static_cast<float>(opus::finfo<DTYPE_O>::max()));

    const int64_t row_offset        = blockIdx.x * cols;
    auto const* ptr_i               = reinterpret_cast<DTYPE_I const*>(input + row_offset);
    auto const* input_vecs          = reinterpret_cast<vec_i const*>(ptr_i);
    static constexpr int32_t ooba_i = 4 / sizeof(DTYPE_I);
    const int32_t oob_i             = (cols + ooba_i - 1) / ooba_i * ooba_i;
    auto buffer_i = opus::make_gmem<DTYPE_I>(ptr_i, oob_i * sizeof(DTYPE_I));

    // double load core loop start
    const int32_t num_elems_tail = cols % vec_size_i;
    const int32_t num_vecs       = (cols + vec_size_i - 1) / vec_size_i;

    vec_i vec_cur;
    size_t vec_idx    = threadIdx.x;
    size_t vec_stride = BlockSize;
    if(vec_idx < num_vecs)
    {
        vec_cur = load_vector_nbytes<DTYPE_I, vec_size_i, load_chunk_bytes>(buffer_i, vec_idx * vec_size_i);
    }

    float absMax = 0.f;
    if constexpr(thread_data_size == 0)
    {
        vec_i vec_nxt;
        for(vec_idx += vec_stride; vec_idx < num_vecs; vec_idx += vec_stride)
        {
            vec_nxt = load_vector_nbytes<DTYPE_I, vec_size_i, load_chunk_bytes>(buffer_i, vec_idx * vec_size_i);
            for(size_t j = 0; j < vec_size_i; j++)
            {
                absMax = max(absMax, abs(static_cast<float>(vec_cur[j])));
            }
            vec_cur = vec_nxt;
        }
        vec_idx -= vec_stride;
    }
    if(vec_idx < num_vecs)
    {
#pragma unroll
        for(size_t j = 0; j < vec_size_i; j++)
        {
            absMax = max(absMax, abs(static_cast<float>(vec_cur[j])));
        }
    }
    // double load core loop end

    absMax = block_reduce<float, aiter::Max, BlockSize, true>(absMax, aiter::Max());

    float row_scale = std::is_same_v<DTYPE_O, opus::fp4_t>
                          ? aiter::fp4_f32_to_e8m0_scale(absMax)
                          : absMax * inverted_DTYPE_MAX;
    return std::make_tuple(row_scale, reinterpret_cast<DTYPE_I*>(&vec_cur));
}

__device__ __forceinline__ float atomicMaxFloat(float *addr, float value)
  {
    float old;
    old = (value >= 0)
              ? __int_as_float(atomicMax((int *)addr, __float_as_int(value)))
              : __uint_as_float(
                    atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
  }

template <typename DTYPE_I, typename DTYPE_O>
__global__ void
data_to_scale_kernel(float* __restrict__ scale, const DTYPE_I* __restrict__ input, const int cols)
{
    auto res        = data_to_per_row_scale<DTYPE_I, DTYPE_O, 0>(input, cols);
    float row_scale = std::get<0>(res);
    if(threadIdx.x == 0)
    {
        atomicMaxFloat(scale, row_scale);
    }
}

template <typename DTYPE_I, typename DTYPE_O>
__device__ void scaled_quant_impl(DTYPE_O* __restrict__ out,
                                  const DTYPE_I* __restrict__ input,
                                  const float* __restrict__ scale,
                                  const int32_t cols)
{

    const float inverted_scale =
        std::is_same_v<DTYPE_O, opus::fp4_t> ? (*scale) : __builtin_amdgcn_rcpf(*scale);
    static constexpr int32_t vec_size_i = 16 / sizeof(DTYPE_O);
    static constexpr int32_t vec_size_o =
        std::is_same_v<DTYPE_O, opus::fp4_t> ? vec_size_i / 2 : vec_size_i;

    using vec_i       = opus::vector_t<DTYPE_I, vec_size_i>;
    using DTYPE_STORE = std::conditional_t<std::is_same_v<DTYPE_O, opus::fp4_t>, uint8_t, DTYPE_O>;

    const int64_t row_offset        = blockIdx.x * cols;
    auto const* ptr_i               = reinterpret_cast<DTYPE_I const*>(input + row_offset);
    auto const* input_vecs          = reinterpret_cast<vec_i const*>(ptr_i);
    auto* ptr_o                     = std::is_same_v<DTYPE_O, opus::fp4_t>
                                          ? reinterpret_cast<DTYPE_STORE*>(out + row_offset / 2)
                                          : reinterpret_cast<DTYPE_STORE*>(out + row_offset);
    static constexpr int32_t ooba_i = 4 / sizeof(DTYPE_I);
    static constexpr int32_t ooba_o = 4 / sizeof(DTYPE_O);
    const int32_t oob_i             = (cols + ooba_i - 1) / ooba_i * ooba_i;
    const int32_t oob_o             = (cols + ooba_o - 1) / ooba_o * ooba_o;

    auto buffer_i = opus::make_gmem<DTYPE_I>(ptr_i, oob_i * sizeof(DTYPE_I));
    auto buffer_o = opus::make_gmem<DTYPE_STORE>(ptr_o, oob_o * sizeof(DTYPE_STORE));

    // double load core loop start
    const int32_t num_elems_tail = cols % vec_size_i;
    const int32_t num_vecs       = (cols + vec_size_i - 1) / vec_size_i;
    const int32_t tail_thread    = num_vecs % BlockSize;
    vec_i vec_nxt;
    vec_i vec_cur;
    // size_t vec_idx = threadIdx.x * vec_size_i;
    // size_t vec_stride = BlockSize * vec_size_i;
    size_t vec_idx    = threadIdx.x;
    size_t vec_stride = BlockSize;
    if(vec_idx < num_vecs)
    {
        vec_cur = load_vector_nbytes<DTYPE_I, vec_size_i, 16>(buffer_i, vec_idx * vec_size_i);
    }

    for(vec_idx += vec_stride; vec_idx < num_vecs; vec_idx += vec_stride)
    {
        vec_nxt = load_vector_nbytes<DTYPE_I, vec_size_i, 16>(buffer_i, vec_idx * vec_size_i);
        store_vector<DTYPE_STORE, DTYPE_I, vec_size_i, RT, false, WARP_SIZE, 1, DTYPE_O>(buffer_o, vec_cur, (vec_idx - vec_stride) * vec_size_o, inverted_scale);
        vec_cur = vec_nxt;
    }

    if(vec_idx - vec_stride < num_vecs)
    {
        store_vector<DTYPE_STORE, DTYPE_I, vec_size_i, RT, false, WARP_SIZE, 1, DTYPE_O>(buffer_o, vec_cur, (vec_idx - vec_stride) * vec_size_o, inverted_scale);
    }
    // double load core loop end
}

template <typename DTYPE_I, typename DTYPE_O, int thread_data_size = 16>
__device__ void scaled_quant_vgpr_impl(DTYPE_O* __restrict__ out,
                                       DTYPE_I* __restrict__ input,
                                       const float* __restrict__ scale,
                                       const int cols,
                                       int64_t out_offset)
{

    const float inverted_scale =
        std::is_same_v<DTYPE_O, opus::fp4_t> ? (*scale) : __builtin_amdgcn_rcpf(*scale);
    static constexpr int32_t vec_size_i = thread_data_size;
    static constexpr int32_t vec_size_o =
        std::is_same_v<DTYPE_O, opus::fp4_t> ? vec_size_i / 2 : vec_size_i;

    using vec_i       = opus::vector_t<DTYPE_I, vec_size_i>;
    using DTYPE_STORE = std::conditional_t<std::is_same_v<DTYPE_O, opus::fp4_t>, uint8_t, DTYPE_O>;

    auto const* ptr_i               = reinterpret_cast<DTYPE_I const*>(input);
    auto const* input_vecs          = reinterpret_cast<vec_i const*>(ptr_i);
    auto* out_ptr                   = reinterpret_cast<DTYPE_O*>(out);
    auto* ptr_o                     = std::is_same_v<DTYPE_O, opus::fp4_t>
                                          ? reinterpret_cast<DTYPE_STORE*>(out + out_offset / 2)
                                          : reinterpret_cast<DTYPE_STORE*>(out + out_offset);
    static constexpr int32_t ooba_i = 4 / sizeof(DTYPE_I);
    static constexpr int32_t ooba_o = 4 / sizeof(DTYPE_O);
    const int32_t oob_i             = (cols + ooba_i - 1) / ooba_i * ooba_i;
    const int32_t oob_o             = (cols + ooba_o - 1) / ooba_o * ooba_o;

    auto buffer_o = opus::make_gmem<DTYPE_STORE>(ptr_o, oob_o * sizeof(DTYPE_STORE));
    const int32_t num_vecs = (cols + vec_size_i - 1) / vec_size_i;

    if(threadIdx.x < num_vecs)
    {
        store_vector<DTYPE_STORE, DTYPE_I, thread_data_size, RT, false, WARP_SIZE, 1, DTYPE_O>(buffer_o, *input_vecs, threadIdx.x * vec_size_o, inverted_scale);
    }
}

template <typename DTYPE_I, typename DTYPE_O>
__global__ void scaled_quant_kernel(DTYPE_O* __restrict__ out,
                                    const DTYPE_I* __restrict__ input,
                                    const float* __restrict__ scale,
                                    const int cols)
{
    scaled_quant_impl<DTYPE_I>(out, input, scale, cols);
}

template <typename DTYPE_I, typename DTYPE_O, int thread_data_size = 16>
__global__ void
dynamic_per_token_scaled_quant_kernel(DTYPE_O* __restrict__ out,
                                      float* __restrict__ scale,
                                      DTYPE_I* __restrict__ input,
                                      float const* __restrict__ scale_ub,
                                      const int32_t cols,
                                      int32_t const* __restrict__ num_rows = nullptr,
                                      const int32_t num_rows_factor        = 1)
{
    const int token_idx = blockIdx.x;
    if(num_rows != nullptr)
    {
        int32_t rows = *num_rows * num_rows_factor;
        if(token_idx >= rows)
            return;
    }
    auto res         = data_to_per_row_scale<DTYPE_I, DTYPE_O, thread_data_size>(input, cols);
    float row_scale  = std::get<0>(res);
    DTYPE_I* vec_ptr = std::get<1>(res);

    if(threadIdx.x == 0)
    {
        if constexpr(std::is_same_v<DTYPE_O, opus::fp4_t>)
        {
            auto* tmp        = reinterpret_cast<uint8_t*>(scale);
            uint8_t exponent = (__builtin_bit_cast(uint32_t, row_scale) >> 23) & 0b11111111;
            tmp[token_idx]   = exponent;
        }
        else
        {
            scale[token_idx] = row_scale;
        }
    }

    if constexpr(thread_data_size == 0)
    {
        scaled_quant_impl<DTYPE_I>(out, input, &row_scale, cols);
    }
    else
    {
        const int64_t row_offset = blockIdx.x * cols;
        scaled_quant_vgpr_impl<DTYPE_I, DTYPE_O, thread_data_size>(out, vec_ptr, &row_scale, cols, row_offset);
    }
}

template <typename DTYPE_I, typename DTYPE_O, int block_size, int thread_data_size = 16>
__device__ std::tuple<float, float*>
smooth_data_to_per_row_scale(const DTYPE_I* __restrict__ input,
                             const float* __restrict__ smooth_scale,
                             int32_t smscale_map_idx,
                             const int32_t cols)
{
    static constexpr int32_t vec_size_i =
        thread_data_size == 0 ? 16 / sizeof(DTYPE_O) : thread_data_size;
    static constexpr int32_t vec_size_o =
        std::is_same_v<DTYPE_O, opus::fp4_t> ? vec_size_i / 2 : vec_size_i;
    using vec_s = opus::vector_t<float, vec_size_i>;
    const float inverted_DTYPE_MAX =
        (1. / static_cast<float>(opus::finfo<DTYPE_O>::max()));

    auto const* ptr_smscale = reinterpret_cast<float const*>(smooth_scale + smscale_map_idx * cols);
    auto const* smscale_vecs = reinterpret_cast<vec_s const*>(ptr_smscale);
    auto buffer_s = opus::make_gmem<float>(ptr_smscale, cols * sizeof(float));

    vec_s smscale_cur;
    size_t vec_idx = threadIdx.x;
    float absMax   = 1e-10f;
    smscale_cur = load_vector_nbytes<float, thread_data_size, 16>(buffer_s, vec_idx * vec_size_i);
#pragma unroll
    for(size_t j = 0; j < vec_size_i; j++)
    {
        smscale_cur[j] = static_cast<float>(input[j]) * smscale_cur[j];
        absMax         = max(absMax, abs(smscale_cur[j]));
    }

    absMax = block_reduce<float, aiter::Max, block_size, true>(absMax, aiter::Max());

    float row_scale = std::is_same_v<DTYPE_O, opus::fp4_t>
                          ? aiter::fp4_f32_to_e8m0_scale(absMax)
                          : absMax * inverted_DTYPE_MAX;
    return std::make_tuple(row_scale, reinterpret_cast<float*>(&smscale_cur));
}

template <typename DTYPE_I, typename DTYPE_O, int block_size, int thread_data_size = 16, bool transpose_out_dim01 = false, bool has_smscale_map = false, bool has_smscale_hash = false, int max_smscale_map_hash_size = 1024>
__global__ void smooth_per_token_scaled_quant_kernel(DTYPE_O* __restrict__ out,
                                                     float* __restrict__ scale,
                                                     DTYPE_I* __restrict__ input,
                                                     float* __restrict__ smooth_scale,
                                                     int* __restrict__ smooth_scale_map,
                                                     int* __restrict__ smooth_scale_map_hash,
                                                     const int32_t num_tg,
                                                     const int32_t cols,
                                                     int32_t const* __restrict__ num_rows = nullptr,
                                                     const int32_t num_rows_factor        = 1,
                                                     const int32_t input_dim0             = 1,
                                                     const int32_t input_dim1             = 1,
                                                     const int32_t input_stride0_cols     = 1,
                                                     const int32_t input_stride1_cols     = 1,
                                                     const int32_t out_stride0_cols       = 1,
                                                     const int32_t out_stride1_cols       = 1,
                                                     const int32_t smooth_scale_map_hash_size = 256)
{
    __shared__ int32_t smooth_scale_map_hash_shared[1024];
    // const int num_tg = gridDim.x;
    int rows = num_rows == nullptr ? input_dim0 * input_dim1 : *num_rows * num_rows_factor;
    if constexpr(has_smscale_hash)
    {
        auto buffer_hash = opus::make_gmem<int>(smooth_scale_map_hash, smooth_scale_map_hash_size * sizeof(int));
        constexpr int32_t async_load_num = (max_smscale_map_hash_size + block_size - 1) / block_size;
        static_assert(max_smscale_map_hash_size <= 1024, "max_smscale_map_hash_size must be less than 1024");
        #pragma unroll
        for(int i = 0; i < async_load_num; i++)
        {
#if defined(__GFX9__)
            const int lds_ptr_sgpr = __builtin_amdgcn_readfirstlane((reinterpret_cast<uintptr_t>((smooth_scale_map_hash_shared + threadIdx.x / WARP_SIZE * WARP_SIZE + i * block_size))));
            uint32_t offset = threadIdx.x * sizeof(int) + i * block_size * sizeof(int);
            asm volatile( "s_mov_b32 m0 %0\n\t"
                "buffer_load_dword %1, %2, 0 offen offset:0 lds\n\t"
                ::"s"(lds_ptr_sgpr), "v"(offset), "s"(buffer_hash.cached_rsrc): "memory", "m0");
#else
            buffer_hash.async_load(smooth_scale_map_hash_shared + threadIdx.x + i * block_size, threadIdx.x + i * block_size);
#endif
        }
    }

    const int rows_per_tg = rows / num_tg;
    const int remainder   = rows - rows_per_tg * num_tg;
    const int chunk_start = blockIdx.x < remainder
                          ? blockIdx.x * (rows_per_tg + 1)
                          : remainder * (rows_per_tg + 1) + (blockIdx.x - remainder) * rows_per_tg;
    const int chunk_size  = rows_per_tg + (blockIdx.x < remainder ? 1 : 0);
    const int chunk_end   = chunk_start + chunk_size;
    const int lane_idx    = threadIdx.x % WARP_SIZE;

    int smscale_map_idx_list = 0;
    int pre_real_token_idx = -1;
    for(int i = 0; i < chunk_size; i++)
    {
        int i_rem = i & (WARP_SIZE - 1);
        if constexpr(has_smscale_map)
        {
            if (i_rem == 0)
            {
                auto buffer_map = opus::make_gmem<int>(smooth_scale_map + chunk_start, chunk_size * sizeof(int));
                smscale_map_idx_list = buffer_map.load(lane_idx + i)[0];
#if defined(__gfx1250__)
                opus::s_wait_loadcnt(opus::number<0>{});
#else
                opus::s_waitcnt_vmcnt(opus::number<0>{});
#endif
                if (i == 0)
                {
                    __syncthreads();
                }
                if constexpr(has_smscale_hash)
                {
                    smscale_map_idx_list = smooth_scale_map_hash_shared[smscale_map_idx_list];
                }
            }
            
        }
        int token_idx = chunk_start + i;
        int idx_input_dim0 = token_idx / input_dim1;
        int idx_input_dim1 = token_idx % input_dim1;
        int real_token_idx = idx_input_dim1 * input_stride1_cols +
                            idx_input_dim0 * input_stride0_cols;
        int32_t smscale_map_idx = __builtin_amdgcn_readlane(smscale_map_idx_list, i_rem);
       
        if (smscale_map_idx < 0)
        {
            continue;
        }
        static constexpr int32_t vec_size_i =
            thread_data_size == 0 ? 16 / sizeof(DTYPE_O) : thread_data_size;
        static constexpr int32_t load_chunk_bytes = sizeof(DTYPE_I) * vec_size_i % 16 == 0 ? 16 : (sizeof(DTYPE_I) * vec_size_i % 8 == 0 ? 8 : 4);
        // using vec_i = opus::vector_t<DTYPE_I, vec_size_i>;
        using vec_i = opus::vector_t<DTYPE_I, vec_size_i>;
        using vec_f = opus::vector_t<float, vec_size_i>;

        vec_f vec_input_f;
        float* input_f_ptr = reinterpret_cast<float*>(&vec_input_f);
        if (real_token_idx != pre_real_token_idx)
        {
            pre_real_token_idx = real_token_idx;
            auto buffer_input = opus::make_gmem<DTYPE_I>(input + (int64_t)real_token_idx * (int64_t)cols, cols * sizeof(DTYPE_I));
            vec_i vec_input = load_vector_nbytes<DTYPE_I, vec_size_i, load_chunk_bytes, RT>(buffer_input, threadIdx.x * vec_size_i);
            for(int i = 0; i < vec_size_i; i++)
            {
                vec_input_f[i] = static_cast<float>(vec_input[i]);
            }
        }
        auto res = smooth_data_to_per_row_scale<float, DTYPE_O, block_size, thread_data_size>(
            input_f_ptr, smooth_scale, smscale_map_idx, cols);
        float row_scale = std::get<0>(res);
        float* vec_ptr  = std::get<1>(res);

        int out_token_idx;
        if constexpr(transpose_out_dim01)
        {   
            int idx_out_dim0 = token_idx / input_dim0;
            int idx_out_dim1 = token_idx % input_dim0;
            out_token_idx = idx_out_dim1 * out_stride1_cols +
                            idx_out_dim0 * out_stride0_cols;
        }
        else
        {
            out_token_idx = idx_input_dim1 * out_stride1_cols +
                            idx_input_dim0 * out_stride0_cols;
        }
        if(threadIdx.x == 0)
        {
            if constexpr(std::is_same_v<DTYPE_O, opus::fp4_t>)
            {
                auto* tmp        = reinterpret_cast<uint8_t*>(scale);
                uint8_t exponent = (__builtin_bit_cast(uint32_t, row_scale) >> 23) & 0b11111111;
                tmp[out_token_idx]   = exponent;
            }
            else
            {
                scale[out_token_idx] = row_scale;
            }
        }

        int64_t out_offset = (int64_t)out_token_idx * (int64_t)cols;    
        scaled_quant_vgpr_impl<float, DTYPE_O, thread_data_size>(out, vec_ptr, &row_scale, cols, out_offset);
    }
}

void static_per_tensor_quant(aiter_tensor_t& out,         // [..., d]
                             const aiter_tensor_t& input, // [..., d]
                             const aiter_tensor_t& scale) // [1]
{
    const int cols = input.size(-1);
    int rows       = input.numel() / cols;
    dim3 grid(rows);
    dim3 block(BlockSize);
    HipDeviceGuard device_guard(input.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    if(out.dtype() == AITER_DTYPE_fp8)
    {
        AITER_DISPATCH_FLOATING16_TYPES_rmTorch(input.dtype(), "scaled_quant_kernel", [&] {
            using input_dtype = typename aiter::hip2opus<scalar_t>::type;
            aiter::scaled_quant_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<opus::fp8_t*>(out.data_ptr()),
                reinterpret_cast<input_dtype*>(input.data_ptr()),
                reinterpret_cast<float*>(scale.data_ptr()),
                cols);
        });
    }
    else if(out.dtype() == AITER_DTYPE_i8)
    {
        AITER_DISPATCH_FLOATING16_TYPES_rmTorch(input.dtype(), "scaled_quant_kernel", [&] {
            using input_dtype = typename aiter::hip2opus<scalar_t>::type;
            aiter::scaled_quant_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<opus::i8_t*>(out.data_ptr()),
                reinterpret_cast<input_dtype*>(input.data_ptr()),
                reinterpret_cast<float*>(scale.data_ptr()),
                cols);
        });
    }
    else
    {
        AITER_CHECK(false, __func__, " not support output type: ", AiterDtype_to_str(out.dtype()));
    }
}

#define DYNAMIC_PER_TOKEN_SCALED_QUANT_KERNEL_IMPL(quant_kernel, DTYPE_O, THREAD_DATA)      \
    AITER_DISPATCH_FLOATING16_TYPES_rmTorch(input.dtype(), "quant_kernel", [&] {             \
        using input_dtype = typename aiter::hip2opus<scalar_t>::type;                        \
        aiter::quant_kernel<input_dtype, DTYPE_O, THREAD_DATA><<<grid, block, 0, stream>>>( \
            reinterpret_cast<DTYPE_O*>(out.data_ptr()),                                     \
            reinterpret_cast<float*>(scales.data_ptr()),                                    \
            reinterpret_cast<input_dtype*>(input.data_ptr()),                               \
            scale_ub.has_value() ? reinterpret_cast<float*>(scale_ub->data_ptr()) : nullptr,            \
            cols,                                                                           \
            num_rows_ptr,                                                                   \
            num_rows_factor);                                                               \
    });

#define DYNAMIC_PER_TOKEN_SCALED_QUANT_KERNEL_DISPATCH(quant_kernel, DTYPE_O, cols) \
    if(cols <= 8 * BlockSize)                                                       \
    {                                                                               \
        DYNAMIC_PER_TOKEN_SCALED_QUANT_KERNEL_IMPL(quant_kernel, DTYPE_O, 8)        \
    }                                                                               \
    else if(cols <= 16 * BlockSize)                                                 \
    {                                                                               \
        DYNAMIC_PER_TOKEN_SCALED_QUANT_KERNEL_IMPL(quant_kernel, DTYPE_O, 16)       \
    }                                                                               \
    else if(cols <= 32 * BlockSize)                                                 \
    {                                                                               \
        DYNAMIC_PER_TOKEN_SCALED_QUANT_KERNEL_IMPL(quant_kernel, DTYPE_O, 32)       \
    }                                                                               \
    else                                                                            \
    {                                                                               \
        DYNAMIC_PER_TOKEN_SCALED_QUANT_KERNEL_IMPL(quant_kernel, DTYPE_O, 0)        \
    }

#define DISPATCH_GROUP_SIZE(gs, ...) \
    if((gs) == 32)        { constexpr int32_t _GS = 32;  __VA_ARGS__ } \
    else if((gs) == 64)   { constexpr int32_t _GS = 64;  __VA_ARGS__ } \
    else                  { constexpr int32_t _GS = 128; __VA_ARGS__ }

void dynamic_per_tensor_quant(aiter_tensor_t& out,         // [..., d]
                              const aiter_tensor_t& input,  // [..., d]
                              aiter_tensor_t& scale)        // [1]
{
    const int cols = input.size(-1);
    int rows       = input.numel() / cols;
    dim3 grid(rows);
    dim3 block(BlockSize);
    HipDeviceGuard device_guard(input.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    if(out.dtype() == AITER_DTYPE_fp8)
    {
        AITER_DISPATCH_FLOATING16_TYPES_rmTorch(input.dtype(), "scaled_quant_kernel", [&] {
            using input_dtype = typename aiter::hip2opus<scalar_t>::type;
            aiter::initializeScale<<<dim3(1), dim3(64), 0, stream>>>(
                reinterpret_cast<float*>(scale.data_ptr()), 1, 0.0f);
            aiter::data_to_scale_kernel<input_dtype, opus::fp8_t><<<grid, block, 0, stream>>>(
                reinterpret_cast<float*>(scale.data_ptr()), reinterpret_cast<input_dtype*>(input.data_ptr()), cols);
            aiter::scaled_quant_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<opus::fp8_t*>(out.data_ptr()),
                reinterpret_cast<input_dtype*>(input.data_ptr()),
                reinterpret_cast<float*>(scale.data_ptr()),
                cols);
        });
    }
    else if(out.dtype() == AITER_DTYPE_i8)
    {
        AITER_DISPATCH_FLOATING16_TYPES_rmTorch(input.dtype(), "scaled_quant_kernel", [&] {
            using input_dtype = typename aiter::hip2opus<scalar_t>::type;
            aiter::initializeScale<<<dim3(1), dim3(64), 0, stream>>>(
                reinterpret_cast<float*>(scale.data_ptr()), 1, 0.0f);
            aiter::data_to_scale_kernel<input_dtype, opus::i8_t><<<grid, block, 0, stream>>>(
                reinterpret_cast<float*>(scale.data_ptr()), reinterpret_cast<input_dtype*>(input.data_ptr()), cols);
            aiter::scaled_quant_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<opus::i8_t*>(out.data_ptr()),
                reinterpret_cast<input_dtype*>(input.data_ptr()),
                reinterpret_cast<float*>(scale.data_ptr()),
                cols);
        });
    }
    else
    {
        AITER_CHECK(false, __func__, " not support output type: ", AiterDtype_to_str(out.dtype()));
    }
}

void dynamic_per_token_scaled_quant(aiter_tensor_t& out,         // [..., d]
                                    const aiter_tensor_t& input, // [..., d]
                                    aiter_tensor_t& scales,
                                    std::optional<aiter_tensor_t> scale_ub,
                                    bool shuffle_scale,
                                    std::optional<aiter_tensor_t> num_rows,
                                    int num_rows_factor)
{
    AITER_CHECK(input.is_contiguous());
    AITER_CHECK(out.is_contiguous());

    int const cols        = input.size(-1);
    int const rows        = input.numel() / cols;
    int32_t* num_rows_ptr = num_rows.has_value() ? reinterpret_cast<int32_t*>(num_rows->data_ptr()) : nullptr;

    HipDeviceGuard device_guard(input.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    if(cols == 32 || cols == 64 || cols == 128)
    {
        DISPATCH_GROUP_SIZE(cols,
            static constexpr int thread_data_size     = 32;
            static constexpr int num_thread_per_group = _GS / thread_data_size;
            auto launch_group_quant = [&](auto out_type_tag, int ori_cols, int ori_rows, int num_group, auto shuffle_tag) {
                using out_t = decltype(out_type_tag);
                constexpr bool ss = decltype(shuffle_tag)::value;
            // Block size follows the group ordering, which is what decides whether a bigger
            // block helps or hurts. Row-major order walks y inside a row, so a wider block
            // just reads a longer contiguous run -- measured 28.7 -> 26.4 us going 64 ->
            // 256. Column-major order (the transposed-scale layout) walks x instead, so a
            // wider block spans more rows, each a separate 256B run ori_row_stride apart,
            // and the lost read locality outweighs the saved launch overhead: 27.0 -> 27.7.
                constexpr bool kColMajor =
                    std::is_same_v<out_t, opus::fp4_t> && ss && _GS == 128;
                // See the note at the other launch site: the 64/256 split is gfx1250-only.
                static constexpr int32_t kBlkTuned = kColMajor ? 64 : 256;
                const int32_t blk_rt = dyn_gq_tuned_arch() ? kBlkTuned : 64;
                const int num_group_per_tg = blk_rt / num_thread_per_group;
                static constexpr int32_t ooba = 4 / sizeof(out_t);
                const int64_t oob_elems =
                    (static_cast<int64_t>(ori_rows) * ori_cols + ooba - 1) / ooba * ooba;
                const int64_t oob_size = oob_elems * static_cast<int64_t>(sizeof(out_t));
                dim3 const grid((num_group + num_group_per_tg - 1) / num_group_per_tg);
                dim3 const block(blk_rt);
                AITER_DISPATCH_FLOATING16_TYPES_rmTorch(
                    input.dtype(), "dynamic_per_group_scaled_quant_kernel", [&] {
                        using input_dtype = typename aiter::hip2opus<scalar_t>::type;
                        auto launch_bs = [&](auto blk_tag) {
                        constexpr int32_t BS = decltype(blk_tag)::value;
                        aiter::dynamic_per_group_scaled_quant_kernel<input_dtype, out_t, thread_data_size, _GS, ss, BS>
                            <<<grid, block, 0, stream>>>(
                            reinterpret_cast<out_t*>(out.data_ptr()),
                            reinterpret_cast<float*>(scales.data_ptr()),
                            reinterpret_cast<input_dtype*>(input.data_ptr()),
                            scale_ub.has_value() ? reinterpret_cast<float*>(scale_ub->data_ptr()) : nullptr,
                            ori_rows,
                            ori_cols,
                            ori_cols,
                            oob_size,
                            num_rows_ptr,
                            num_rows_factor);
                        };
                        if(blk_rt == kBlkTuned)
                            launch_bs(std::integral_constant<int32_t, kBlkTuned>{});
                        else
                            launch_bs(std::integral_constant<int32_t, 64>{});
                    });
            };
            auto do_launch = [&](auto shuffle_tag) {
                if(out.dtype() == AITER_DTYPE_fp8)
                {
                    int ori_cols  = out.size(-1);
                    int ori_rows  = rows / (ori_cols / _GS);
                    launch_group_quant(opus::fp8_t{}, ori_cols, ori_rows, rows, shuffle_tag);
                }
                else if(out.dtype() == AITER_DTYPE_i8)
                {
                    int ori_cols  = _GS;
                    int ori_rows  = rows;
                    launch_group_quant(opus::i8_t{}, ori_cols, ori_rows, rows, shuffle_tag);
                }
#if defined(__Float4_e2m1fn_x2)
                else if(out.dtype() == AITER_DTYPE_fp4x2)
                {
                    int ori_cols  = out.size(-1) * 2;
                    int ori_rows  = rows / (ori_cols / _GS);
                    constexpr bool ss = decltype(shuffle_tag)::value;
                    int num_group = ss ? ori_rows * (((ori_cols / _GS) + 7) / 8 * 8) : rows;
                    launch_group_quant(opus::fp4_t{}, ori_cols, ori_rows, num_group, shuffle_tag);
                }
#endif
                else
                {
                    AITER_CHECK(false, __func__, " not support output type: ", AiterDtype_to_str(out.dtype()));
                }
            };
            if(shuffle_scale)
                do_launch(std::true_type{});
            else
                do_launch(std::false_type{});
        )
    }
    else
    {
        dim3 const grid(rows);
        dim3 const block(BlockSize);
        if(out.dtype() == AITER_DTYPE_fp8)
        {
            DYNAMIC_PER_TOKEN_SCALED_QUANT_KERNEL_DISPATCH(
                dynamic_per_token_scaled_quant_kernel, opus::fp8_t, cols);
        }
        else if(out.dtype() == AITER_DTYPE_i8)
        {
            DYNAMIC_PER_TOKEN_SCALED_QUANT_KERNEL_DISPATCH(
                dynamic_per_token_scaled_quant_kernel, opus::i8_t, cols);
        }
#if defined(__Float4_e2m1fn_x2)
        else if(out.dtype() == AITER_DTYPE_fp4x2)
        {
            DYNAMIC_PER_TOKEN_SCALED_QUANT_KERNEL_DISPATCH(
                dynamic_per_token_scaled_quant_kernel, opus::fp4_t, cols);
        }
#endif
        else
        {
            AITER_CHECK(false, __func__, " not support output type: ", AiterDtype_to_str(out.dtype()));
        }
    }
}

// Canonical dynamic per-group scaled quant. Accepts fp8 / i8 / fp4x2 output;
// the per-group scale layout is selected by ``scales.dtype()``:
//   * AITER_DTYPE_fp8_e8m0 / u8 -> e8m0 byte scale (one byte per group of
//     ``group_size`` elements). Required for MXFP4/MXFP8 GEMM consumers.
//   * AITER_DTYPE_fp32          -> continuous fp32 per-group scale.
// fp4 outputs always emit e8m0 (there is no fp32-scale fp4 path); fp8 picks
// the path by scale dtype; i8 only supports fp32. The legacy entry point
// `dynamic_per_group_scaled_quant_fp4` (kept as a forwarder below) hard-coded
// fp4 only, which made the MXFP8 1xG byte-scale path unreachable here.
void dynamic_per_group_scaled_quant(aiter_tensor_t& out,         // [..., d]
                                    const aiter_tensor_t& input, // [..., d]
                                    aiter_tensor_t& scales,
                                    int group_size,
                                    bool shuffle_scale,
                                    std::optional<aiter_tensor_t> num_rows,
                                    int num_rows_factor)
{
    AITER_CHECK(group_size == 32 || group_size == 64 || group_size == 128,
                __func__,
                " only support group_size [32, 64 , 128]");
    AITER_CHECK(out.is_contiguous());

    int const cols        = input.size(-1);
    int const rows        = input.numel() / cols;
    int const row_stride  = input.stride(-2);
    int32_t* num_rows_ptr = num_rows.has_value() ? reinterpret_cast<int32_t*>(num_rows->data_ptr()) : nullptr;

    AITER_CHECK(cols % group_size == 0, __func__, " cols is not divisible by group_size");

    // Decide e8m0 vs fp32 scale path from scales.dtype(). Note u8 alias is
    // accepted for callers that build the scale tensor as raw uint8.
    const bool use_e8m0_scale =
        scales.dtype() == AITER_DTYPE_fp8_e8m0 || scales.dtype() == AITER_DTYPE_u8;
    AITER_CHECK(use_e8m0_scale || scales.dtype() == AITER_DTYPE_fp32,
                __func__,
                " expects scales.dtype in {fp8_e8m0, u8, fp32}, got ",
                AiterDtype_to_str(scales.dtype()));

    HipDeviceGuard device_guard(input.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    DISPATCH_GROUP_SIZE(group_size,
        static constexpr int thread_data_size     = 32;
        static constexpr int num_thread_per_group = _GS / thread_data_size;
        int scaleN    = cols / _GS;

        auto launch = [&](auto out_type_tag, auto shuffle_tag, auto e8m0_tag) {
            using out_t = decltype(out_type_tag);
            constexpr bool ss = decltype(shuffle_tag)::value;
            constexpr bool ee = decltype(e8m0_tag)::value;
            // Block size follows the group ordering, which is what decides whether a bigger
            // block helps or hurts. Row-major order walks y inside a row, so a wider block
            // just reads a longer contiguous run -- measured 28.7 -> 26.4 us going 64 ->
            // 256. Column-major order (the transposed-scale layout) walks x instead, so a
            // wider block spans more rows, each a separate 256B run ori_row_stride apart,
            // and the lost read locality outweighs the saved launch overhead: 27.0 -> 27.7.
            constexpr bool kColMajor =
                (std::is_same_v<out_t, opus::fp4_t> || ee) && ss && _GS == 128;
            // The 64/256 split was swept on gfx1250 and leans on that part's wave width
            // and dispatch behaviour; nothing was measured on gfx950, so an untuned arch
            // keeps the 64 it always had. Block size is a template argument, so both are
            // instantiated and the arch picks between them at launch.
            static constexpr int32_t kBlkTuned = kColMajor ? 64 : 256;
            const int32_t blk_rt = dyn_gq_tuned_arch() ? kBlkTuned : 64;
            const int num_group_per_tg = blk_rt / num_thread_per_group;
            dim3 const block(blk_rt);
            // e8m0 + shuffle pads scaleN up to a multiple of 8 (tile width)
            // for the MX hw layout (_GS == 32 only); group_size == 128
            // shuffle is a plain transpose, no padding.
            int num_group;
            if constexpr(ee)
            {
                num_group = (ss && _GS == 32) ? rows * ((scaleN + 7) / 8 * 8) : rows * scaleN;
            }
            else
            {
                num_group = rows * scaleN;
            }
            static constexpr int32_t ooba = 4 / sizeof(out_t);
            const int64_t oob_elems =
                (static_cast<int64_t>(rows) * cols + ooba - 1) / ooba * ooba;
            const int64_t oob_size = oob_elems * static_cast<int64_t>(sizeof(out_t));

            // TDM staging, decided here because it changes the grid: a staged block owns
            // DYN_GQ_TDM_KPT steps' worth of groups instead of one.
            //   - It needs a wave's groups to stay inside one column and the block span to
            //     tile the matrix, hence rows % groups-per-staged-block.
            //   - It halves the block count, which only pays once there are blocks to
            //     spare. Measured at [T, 7168]: staging costs 15% at T=128 (224 blocks)
            //     and 4% at T=512, and wins from a few thousand blocks up. The gate is
            //     blocks per SIMD, not T, so it travels across shapes.
            const int kGroupsPerStagedBlock = num_group_per_tg * DYN_GQ_TDM_KPT;
            static constexpr int kTdmMinBlocksPerSimd = DYN_GQ_TDM_MIN_BLK_PER_SIMD;
            const int simds = static_cast<int>(get_num_cu_func()) * 4;   // 4 SIMDs per CU
            const bool use_tdm =
                dyn_gq_tuned_arch() &&
                kColMajor && DYN_GQ_TDM_KPT > 1 && (rows % kGroupsPerStagedBlock) == 0 &&
                (num_group / kGroupsPerStagedBlock) >= simds * kTdmMinBlocksPerSimd;

            // The grid is sized by the UNSTAGED block span even when staging is on, so a
            // staged block (which owns kGroupsPerStagedBlock) leaves the back half of the
            // grid with nothing in range; those blocks reject on the first resolve and
            // retire. That looks like waste and sizing the grid exactly instead is the
            // obvious fix -- but it measured 15-24% SLOWER across M = 4096/8192/16384
            // (8.05/14.38/25.24 us against 9.95/17.01/28.96), with device code identical
            // and only the launch dimension differing. The mechanism is not understood, so
            // this keeps the configuration that measures faster rather than the one that
            // reasons better. Revisit with a dispatch-level profile.
            dim3 const grid((num_group + num_group_per_tg - 1) / num_group_per_tg);
            AITER_DISPATCH_FLOATING16_TYPES_rmTorch(
                input.dtype(), "dynamic_per_group_scaled_quant_kernel", [&] {
                    using input_dtype = typename aiter::hip2opus<scalar_t>::type;
                    auto launch_one = [&](auto tdm_tag, auto blk_tag) {
                    constexpr bool tt = decltype(tdm_tag)::value;
                    constexpr int32_t BS = decltype(blk_tag)::value;
                    aiter::dynamic_per_group_scaled_quant_kernel<input_dtype, out_t, thread_data_size, _GS, ss, BS, ee, tt>
                        <<<grid, block, 0, stream>>>(
                        reinterpret_cast<out_t*>(out.data_ptr()),
                        reinterpret_cast<float*>(scales.data_ptr()),
                        reinterpret_cast<input_dtype*>(input.data_ptr()),
                        nullptr,
                        rows,
                        cols,
                        row_stride,
                        oob_size,
                        num_rows_ptr,
                        num_rows_factor);
                    };
                    using BlkTuned = std::integral_constant<int32_t, kBlkTuned>;
                    using Blk64    = std::integral_constant<int32_t, 64>;
                    if(blk_rt == kBlkTuned)
                    {
                        if(use_tdm) launch_one(std::true_type{},  BlkTuned{});
                        else        launch_one(std::false_type{}, BlkTuned{});
                    }
                    else
                    {
                        if(use_tdm) launch_one(std::true_type{},  Blk64{});
                        else        launch_one(std::false_type{}, Blk64{});
                    }
                });
        };

        auto do_launch = [&](auto shuffle_tag, auto e8m0_tag) {
            constexpr bool ee = decltype(e8m0_tag)::value;
            if(out.dtype() == AITER_DTYPE_fp8)
            {
                launch(opus::fp8_t{}, shuffle_tag, e8m0_tag);
            }
            else if(out.dtype() == AITER_DTYPE_i8)
            {
                static_assert(true, "i8 path does not support e8m0 scale");
                AITER_CHECK(!ee, __func__, " i8 output does not support e8m0 scale");
                launch(opus::i8_t{}, shuffle_tag, std::false_type{});
            }
#if defined(__Float4_e2m1fn_x2)
            else if(out.dtype() == AITER_DTYPE_fp4x2 || out.dtype() == AITER_DTYPE_u8)
            {
                // fp4 always uses e8m0 scale regardless of `use_e8m0_scale`.
                launch(opus::fp4_t{}, shuffle_tag, std::true_type{});
            }
#endif
            else
            {
                AITER_CHECK(false, __func__, " not support output type: ", AiterDtype_to_str(out.dtype()));
            }
        };

        auto with_e8m0 = [&](auto shuffle_tag) {
            if(use_e8m0_scale)
                do_launch(shuffle_tag, std::true_type{});
            else
                do_launch(shuffle_tag, std::false_type{});
        };
        if(shuffle_scale)
            with_e8m0(std::true_type{});
        else
            with_e8m0(std::false_type{});
    )
}

// Backward-compat thin forwarder. Asserts fp4x2/u8 output and delegates to
// the dtype-aware canonical entry. Existing callers (Python compile_ops
// binding `dynamic_per_group_scaled_quant_fp4`, downstream tests, etc.)
// continue to work unchanged.
void dynamic_per_group_scaled_quant_fp4(aiter_tensor_t& out,         // [..., d]
                                        const aiter_tensor_t& input, // [..., d]
                                        aiter_tensor_t& scales,
                                        int group_size,
                                        bool shuffle_scale,
                                        std::optional<aiter_tensor_t> num_rows,
                                        int num_rows_factor)
{
    AITER_CHECK(out.dtype() == AITER_DTYPE_fp4x2 || out.dtype() == AITER_DTYPE_u8,
                __func__,
                " expects fp4x2 / uint8 output; use dynamic_per_group_scaled_quant for fp8/i8");
    dynamic_per_group_scaled_quant(
        out, input, scales, group_size, shuffle_scale, num_rows, num_rows_factor);
}

#define SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_IMPL(quant_kernel, DTYPE_O, THREAD_DATA, BLOCK_SIZE, TRANSPOSE_OUT_DIM01, HAS_MAP, HAS_HASH) \
    AITER_DISPATCH_FLOATING16_TYPES_rmTorch(input.dtype(), "quant_kernel", [&] {                                       \
        using input_dtype = typename aiter::hip2opus<scalar_t>::type;                                                  \
        const int cu_num = get_num_cu_func();                                                                          \
        const int max_warp_per_simd = 8;                                                                               \
        const int warp_per_simd = BLOCK_SIZE / (opus::get_warp_size() * 4);                                            \
        int grid_size = enable_ps ? max_warp_per_simd / warp_per_simd * cu_num : rows;                                 \
        dim3 const grid(grid_size);                                                                                    \
        aiter::quant_kernel<input_dtype, DTYPE_O, BLOCK_SIZE, THREAD_DATA, TRANSPOSE_OUT_DIM01, HAS_MAP, HAS_HASH, MAX_EXPERT_SIZE> \
            <<<grid, dim3(BLOCK_SIZE), 0, stream>>>(                                                                   \
                reinterpret_cast<DTYPE_O*>(out.data_ptr()),                                                            \
                reinterpret_cast<float*>(scales.data_ptr()),                                                           \
                reinterpret_cast<input_dtype*>(input.data_ptr()),                                                      \
                reinterpret_cast<float*>(smooth_scale.data_ptr()),                                                     \
                smooth_scale_map_ptr,                                                                                  \
                smooth_scale_map_hash_ptr,                                                                             \
                grid_size,                                                                                             \
                cols,                                                                                                  \
                num_rows_ptr,                                                                                          \
                num_rows_factor,                                                                                       \
                input_dim0,                                                                                            \
                input_dim1,                                                                                            \
                input_stride0_cols,                                                                                    \
                input_stride1_cols,                                                                                    \
                out_stride0_cols,                                                                                      \
                out_stride1_cols,                                                                                      \
                smooth_scale_map_hash_size);                                                                           \
    });

#define SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_IMPL_(quant_kernel, DTYPE_O, THREAD_DATA, BLOCK_SIZE)                             \
    if(transpose_out_dim01)                                                                                                    \
    {                                                                                                                          \
        if(smooth_scale_map_ptr != nullptr && smooth_scale_map_hash_ptr != nullptr)                                            \
            SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_IMPL(quant_kernel, DTYPE_O, THREAD_DATA, BLOCK_SIZE, true, true, true)        \
        else if(smooth_scale_map_ptr != nullptr)                                                                               \
            SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_IMPL(quant_kernel, DTYPE_O, THREAD_DATA, BLOCK_SIZE, true, true, false)       \
        else                                                                                                                   \
            SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_IMPL(quant_kernel, DTYPE_O, THREAD_DATA, BLOCK_SIZE, true, false, false)      \
    }                                                                                                                          \
    else                                                                                                                       \
    {                                                                                                                          \
        if(smooth_scale_map_ptr != nullptr && smooth_scale_map_hash_ptr != nullptr)                                            \
            SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_IMPL(quant_kernel, DTYPE_O, THREAD_DATA, BLOCK_SIZE, false, true, true)       \
        else if(smooth_scale_map_ptr != nullptr)                                                                               \
            SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_IMPL(quant_kernel, DTYPE_O, THREAD_DATA, BLOCK_SIZE, false, true, false)      \
        else                                                                                                                   \
            SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_IMPL(quant_kernel, DTYPE_O, THREAD_DATA, BLOCK_SIZE, false, false, false)     \
    }

#define SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_DISPATCH(quant_kernel, DTYPE_O, cols)           \
    if(cols <= 8 * BlockSize)                                                                \
    {                                                                                        \
        SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_IMPL_(quant_kernel, DTYPE_O, 8, BlockSize)      \
    }                                                                                        \
    else if(cols <= 16 * BlockSize)                                                          \
    {                                                                                        \
        SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_IMPL_(quant_kernel, DTYPE_O, 16, BlockSize)     \
    }                                                                                        \
    else if(cols <= 16 * BlockSize * 2)                                                      \
    {                                                                                        \
        SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_IMPL_(quant_kernel, DTYPE_O, 16, BlockSize * 2) \
    }                                                                                        \
    else                                                                                     \
    {                                                                                        \
        AITER_CHECK(false, "input last dim has exceeded the maximum value ", 32 * BlockSize); \
    }

void smooth_per_token_scaled_quant(
    aiter_tensor_t& out,         // [..., d]
    const aiter_tensor_t& input, // [..., d]
    aiter_tensor_t& scales,
    const aiter_tensor_t& smooth_scale,
    std::optional<aiter_tensor_t> smooth_scale_map,
    bool shuffle_scale,
    std::optional<aiter_tensor_t> num_rows,
    int num_rows_factor,
    std::optional<aiter_tensor_t> smooth_scale_map_hash,
    bool enable_ps)
{

    int const cols        = input.size(-1);
    int const rows        = input.numel() / cols;
    int32_t* num_rows_ptr = num_rows.has_value() ? reinterpret_cast<int32_t*>(num_rows->data_ptr()) : nullptr;
    int32_t* smooth_scale_map_ptr =
        smooth_scale_map.has_value() ? reinterpret_cast<int32_t*>(smooth_scale_map->data_ptr()) : nullptr;
    int32_t* smooth_scale_map_hash_ptr =
        smooth_scale_map_hash.has_value() ? reinterpret_cast<int32_t*>(smooth_scale_map_hash->data_ptr()) : nullptr;
    AITER_CHECK(
        input.dim() < 4, __func__, " only support input dim <=3, but get dim: ", input.dim());
    int32_t input_dim0    = input.size(0);
    int32_t input_dim1    = input.dim() > 2 ? input.size(1) : 1;
    int32_t input_stride0 = input.stride(0);
    int32_t input_stride1 = input.dim() > 2 ? input.stride(1) : cols;
    int32_t out_dim0 = out.size(0);
    int32_t out_dim1 = out.dim() > 2 ? out.size(1) : 1;
    int32_t out_stride0 = out.stride(0);
    int32_t out_stride1 = out.dim() > 2 ? out.stride(1) : cols;
    int32_t input_stride0_cols = input_stride0 / cols;
    int32_t input_stride1_cols = input_stride1 / cols;
    int32_t out_stride0_cols = out_stride0 / cols;
    int32_t out_stride1_cols = out_stride1 / cols;
    constexpr int32_t MAX_EXPERT_SIZE = 1024;
    int32_t smooth_scale_map_hash_size =
        smooth_scale_map_hash.has_value() ? smooth_scale_map_hash->numel() : 0;
    AITER_CHECK(
        smooth_scale_map_hash_size <= MAX_EXPERT_SIZE, __func__, " smooth_scale_map_hash_size is too large, only support <= ", MAX_EXPERT_SIZE);
    AITER_CHECK((input_dim0 * input_dim1 == out_dim0 * out_dim1) && (input_dim0 == out_dim0 || input_dim0 == out_dim1),
        __func__, "This kernel view input as 3D (m,k,n) and output as 3D (m,k,n)/(k,m,n)");
    const bool transpose_out_dim01 = input_dim0 != out_dim0;

    HipDeviceGuard device_guard(input.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    if(out.dtype() == AITER_DTYPE_fp8)
    {
        SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_DISPATCH(
            smooth_per_token_scaled_quant_kernel, opus::fp8_t, cols);
    }
    else if(out.dtype() == AITER_DTYPE_i8)
    {
        SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_DISPATCH(
            smooth_per_token_scaled_quant_kernel, opus::i8_t, cols);
    }
#if defined(__Float4_e2m1fn_x2)
    else if(out.dtype() == AITER_DTYPE_fp4x2 || out.dtype() == AITER_DTYPE_u8)
    {
        SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_DISPATCH(
            smooth_per_token_scaled_quant_kernel, opus::fp4_t, cols);
    }
#endif
    else
    {
        AITER_CHECK(false, __func__, " not support output type: ", AiterDtype_to_str(out.dtype()));
    }
}

template <typename DTYPE, int BLOCK_SIZE = 256, int thread_data_size = 4, int MAX_ITERS = 10000>
__global__ void partial_transpose_kernel(DTYPE* __restrict__ out,
                                         DTYPE* __restrict__ input,
                                         const int* __restrict__ num_rows,
                                         const int cols)
{
    using vec_i                     = opus::vector_t<DTYPE, thread_data_size>;
    int GRID_SIZE                   = gridDim.x;
    int ori_rows                    = *num_rows;
    int thread_per_row              = (cols + thread_data_size - 1) / thread_data_size;
    auto const* ptr_i               = reinterpret_cast<DTYPE const*>(input);
    static constexpr int32_t ooba_i = 4 / sizeof(DTYPE);
    const int32_t oob_i             = (ori_rows * cols + ooba_i - 1) / ooba_i * ooba_i;
    static constexpr int32_t load_chunk_bytes = sizeof(DTYPE) * thread_data_size % 16 == 0 ? 16 : (sizeof(DTYPE) * thread_data_size % 8 == 0 ? 8 : 4);
    auto buffer_i = opus::make_gmem<DTYPE>(ptr_i, oob_i * sizeof(DTYPE));
    for(int i = 0; i < MAX_ITERS; i++)
    {
        int64_t y = i * GRID_SIZE * BLOCK_SIZE + blockIdx.x * BLOCK_SIZE + threadIdx.x;
        int x     = y % thread_per_row * thread_data_size;
        y         = y / thread_per_row;
        if(y >= ori_rows)
            return;
        vec_i input_vecs   = load_vector_nbytes<DTYPE, thread_data_size, load_chunk_bytes>(buffer_i, y * cols + x);
        int64_t out_offset = x * ori_rows + y;
        // printf("blockIdx: %d, threadIdx:%d, y: %d, x: %d, ori_rows: %d, cols: %d, val:%f\n",
        // blockIdx.x, threadIdx.x, y, x, ori_rows, cols,
        // static_cast<float>(input_vecs[0]));
        for(int j = 0; j < thread_data_size; j++)
        {
            if((x + j) < cols)
            {
                out[out_offset + j * ori_rows] = input_vecs[j];
            }
        }
    }
}

void partial_transpose(aiter_tensor_t& out,         // [rows, d]
                       const aiter_tensor_t& input, // [rows, d]
                       const aiter_tensor_t& num_rows)
{
    AITER_CHECK(out.is_contiguous());
    AITER_CHECK(input.is_contiguous());

    uint32_t num_cu       = get_num_cu_func();
    int const cols        = input.size(-1);
    int const rows        = input.numel() / cols;
    int32_t* num_rows_ptr = reinterpret_cast<int32_t*>(num_rows.data_ptr());

    HipDeviceGuard device_guard(input.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    if(cols <= 1024)
    {
        const int BlockSize        = 256;
        const int GridSize         = num_cu * 8; // Adjust as needed
        const int thread_data_size = 1024 / BlockSize;

        dim3 grid(GridSize);
        dim3 block(BlockSize);

        VLLM_DISPATCH_FLOATING_TYPES_rmTorch(input.dtype(), "partial_transpose_kernel", [&] {
            using input_dtype = typename aiter::hip2opus<scalar_t>::type;
            aiter::partial_transpose_kernel<input_dtype, BlockSize, thread_data_size>
                <<<grid, block, 0, stream>>>(reinterpret_cast<input_dtype*>(out.data_ptr()),
                                             reinterpret_cast<input_dtype*>(input.data_ptr()),
                                             num_rows_ptr,
                                             cols);
        });
    }
    else if(cols <= 2048)
    {
        const int BlockSize        = 256;
        const int GridSize         = num_cu * 4;
        const int thread_data_size = 2048 / BlockSize;

        dim3 grid(GridSize);
        dim3 block(BlockSize);

        VLLM_DISPATCH_FLOATING_TYPES_rmTorch(input.dtype(), "partial_transpose_kernel", [&] {
            using input_dtype = typename aiter::hip2opus<scalar_t>::type;
            aiter::partial_transpose_kernel<input_dtype, BlockSize, thread_data_size>
                <<<grid, block, 0, stream>>>(reinterpret_cast<input_dtype*>(out.data_ptr()),
                                             reinterpret_cast<input_dtype*>(input.data_ptr()),
                                             num_rows_ptr,
                                             cols);
        });
    }
    else if(cols <= 4096)
    {
        const int BlockSize        = 256;
        const int GridSize         = num_cu * 2;
        const int thread_data_size = 4096 / BlockSize;

        dim3 grid(GridSize);
        dim3 block(BlockSize);

        VLLM_DISPATCH_FLOATING_TYPES_rmTorch(input.dtype(), "partial_transpose_kernel", [&] {
            using input_dtype = typename aiter::hip2opus<scalar_t>::type;
            aiter::partial_transpose_kernel<input_dtype, BlockSize, thread_data_size>
                <<<grid, block, 0, stream>>>(reinterpret_cast<input_dtype*>(out.data_ptr()),
                                             reinterpret_cast<input_dtype*>(input.data_ptr()),
                                             num_rows_ptr,
                                             cols);
        });
    }
    else if(cols <= 8192)
    {
        const int BlockSize        = 512;
        const int GridSize         = num_cu;
        const int thread_data_size = 8192 / BlockSize;

        dim3 grid(GridSize);
        dim3 block(BlockSize);

        VLLM_DISPATCH_FLOATING_TYPES_rmTorch(input.dtype(), "partial_transpose_kernel", [&] {
            using input_dtype = typename aiter::hip2opus<scalar_t>::type;
            aiter::partial_transpose_kernel<input_dtype, BlockSize, thread_data_size>
                <<<grid, block, 0, stream>>>(reinterpret_cast<input_dtype*>(out.data_ptr()),
                                             reinterpret_cast<input_dtype*>(input.data_ptr()),
                                             num_rows_ptr,
                                             cols);
        });
    }
    else
    {
        AITER_CHECK(false, __func__, " cols is not supported: ", cols);
    }
}


template <typename DTYPE_I, typename DTYPE_O, int block_size, int thread_data_size = 16, bool transpose_out_dim01 = false, bool has_smscale_hash = false, int max_smscale_map_hash_size = 1024>
__global__ void moe_smooth_per_token_scaled_quant_kernel_v1(DTYPE_O* __restrict__ out,
                                                     float* __restrict__ scale,
                                                     DTYPE_I* __restrict__ input,
                                                     float* __restrict__ smooth_scale,
                                                     int* __restrict__ smooth_scale_map,
                                                     int* __restrict__ smooth_scale_map_hash,
                                                     const int32_t num_rows,
                                                     const int32_t m_repeat,
                                                     const int32_t cols,
                                                     const int32_t input_stride     = 1,
                                                     const int32_t smooth_scale_map_hash_size = 256)
{
    __shared__ int32_t smooth_scale_map_hash_shared[1024];
    int token_idx = blockIdx.x;
    int lane_idx = threadIdx.x % WARP_SIZE;
    static constexpr int32_t vec_size_i =
        thread_data_size == 0 ? 16 / sizeof(DTYPE_I) : thread_data_size;
    static constexpr int32_t load_chunk_bytes = 
        (sizeof(DTYPE_I) * vec_size_i % 16 == 0 ? 16 : (sizeof(DTYPE_I) * vec_size_i % 8 == 0 ? 8 : 4));
    if constexpr(has_smscale_hash)
    {
        auto buffer_hash = opus::make_gmem<int>(smooth_scale_map_hash, smooth_scale_map_hash_size * sizeof(int));
        constexpr int32_t async_load_num = (max_smscale_map_hash_size + block_size - 1) / block_size;
        static_assert(max_smscale_map_hash_size <= 1024, "max_smscale_map_hash_size must be less than 1024");
        #pragma unroll
        for(int i = 0; i < async_load_num; i++)
        {
#if defined(__GFX9__)
            const int lds_ptr_sgpr = __builtin_amdgcn_readfirstlane((reinterpret_cast<uintptr_t>((smooth_scale_map_hash_shared + threadIdx.x / WARP_SIZE * WARP_SIZE + i * block_size))));
            uint32_t offset = threadIdx.x * sizeof(int) + i * block_size * sizeof(int);
            asm volatile( "s_mov_b32 m0 %0\n\t"
                "buffer_load_dword %1, %2, 0 offen offset:0 lds\n\t"
                ::"s"(lds_ptr_sgpr), "v"(offset), "s"(buffer_hash.cached_rsrc): "memory", "m0");
#else
            buffer_hash.async_load(smooth_scale_map_hash_shared + threadIdx.x + i * block_size, threadIdx.x + i * block_size);
#endif
        }
    }
    int smscale_map_idx_list = 0;
    auto buffer_map = opus::make_gmem<int>(smooth_scale_map + token_idx * m_repeat, m_repeat * sizeof(int));
    smscale_map_idx_list = buffer_map.load(lane_idx)[0];
    using vec_i = opus::vector_t<DTYPE_I, vec_size_i>;
    using vec_f = opus::vector_t<float, vec_size_i>;
    vec_f vec_input_f;
    float* input_f_ptr = reinterpret_cast<float*>(&vec_input_f);
    auto buffer_input = opus::make_gmem<DTYPE_I>(input + (int64_t)token_idx * (int64_t)input_stride, cols * sizeof(DTYPE_I));
    vec_i vec_input = load_vector_nbytes<DTYPE_I, vec_size_i, load_chunk_bytes, RT>(buffer_input, threadIdx.x * vec_size_i);
#if defined(__gfx1250__)
    opus::s_wait_loadcnt(opus::number<vec_size_i * sizeof(DTYPE_I) / load_chunk_bytes>{});
#else
    opus::s_waitcnt_vmcnt(opus::number<vec_size_i * sizeof(DTYPE_I) / load_chunk_bytes>{});
#endif
    __syncthreads();
    if constexpr(has_smscale_hash)
    {
        if(lane_idx < m_repeat && smscale_map_idx_list >= 0 && smscale_map_idx_list < smooth_scale_map_hash_size)
        {
            smscale_map_idx_list = smooth_scale_map_hash_shared[smscale_map_idx_list];
        }
    }
    for(int i = 0; i < vec_size_i; i++)
    {
        vec_input_f[i] = static_cast<float>(vec_input[i]);
    }
    for(int i = 0; i < m_repeat; i++)
    {
        int32_t smscale_map_idx = __builtin_amdgcn_readlane(smscale_map_idx_list, i);
        if(smscale_map_idx < 0)
        {
            continue;
        }
        auto res = smooth_data_to_per_row_scale<float, DTYPE_O, block_size, thread_data_size>(
            input_f_ptr, smooth_scale, smscale_map_idx, cols);
        float row_scale = std::get<0>(res);
        float* vec_ptr  = std::get<1>(res);

        int out_token_idx;
        if constexpr(transpose_out_dim01)
        {   
            out_token_idx = i * num_rows + token_idx;
        }
        else
        {
            out_token_idx = token_idx * m_repeat + i;
        }
        if(threadIdx.x == 0)
        {
            if constexpr(std::is_same_v<DTYPE_O, opus::fp4_t>)
            {
                auto* tmp        = reinterpret_cast<uint8_t*>(scale);
                uint8_t exponent = (__builtin_bit_cast(uint32_t, row_scale) >> 23) & 0b11111111;
                tmp[out_token_idx]   = exponent;
            }
            else
            {
                scale[out_token_idx] = row_scale;
            }
        }

        int64_t out_offset = (int64_t)out_token_idx * (int64_t)cols;    
        scaled_quant_vgpr_impl<float, DTYPE_O, thread_data_size>(out, vec_ptr, &row_scale, cols, out_offset);
    }
}


#define MOE_SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_V1_IMPL(quant_kernel, DTYPE_O, THREAD_DATA, BLOCK_SIZE, TRANSPOSE_OUT_DIM01, HAS_HASH) \
    AITER_DISPATCH_FLOATING16_TYPES_rmTorch(input.dtype(), "quant_kernel", [&] {                                       \
        using input_dtype = typename aiter::hip2opus<scalar_t>::type;                                                  \
        int grid_size = rows;                                                                                          \
        dim3 const grid(grid_size);                                                                                    \
        aiter::quant_kernel<input_dtype, DTYPE_O, BLOCK_SIZE, THREAD_DATA, TRANSPOSE_OUT_DIM01, HAS_HASH, MAX_EXPERT_SIZE> \
            <<<grid, dim3(BLOCK_SIZE), 0, stream>>>(                                                                   \
                reinterpret_cast<DTYPE_O*>(out.data_ptr()),                                                            \
                reinterpret_cast<float*>(scales.data_ptr()),                                                           \
                reinterpret_cast<input_dtype*>(input.data_ptr()),                                                      \
                reinterpret_cast<float*>(smooth_scale.data_ptr()),                                                     \
                smooth_scale_map_ptr,                                                                                  \
                smooth_scale_map_hash_ptr,                                                                             \
                rows,                                                                                                  \
                m_repeat,                                                                                              \
                cols,                                                                                                  \
                input_stride,                                                                                          \
                smooth_scale_map_hash_size);                                                                           \
    });


#define MOE_SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_V1_IMPL_(quant_kernel, DTYPE_O, THREAD_DATA, BLOCK_SIZE)                             \
    if(transpose_out_dim01)                                                                                                    \
    {                                                                                                                          \
        if(smooth_scale_map_hash_ptr != nullptr)                                                                               \
            MOE_SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_V1_IMPL(quant_kernel, DTYPE_O, THREAD_DATA, BLOCK_SIZE, true, true)       \
        else                                                                                                                   \
            MOE_SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_V1_IMPL(quant_kernel, DTYPE_O, THREAD_DATA, BLOCK_SIZE, true, false)      \
    }                                                                                                                          \
    else                                                                                                                       \
    {                                                                                                                          \
        if(smooth_scale_map_hash_ptr != nullptr)                                                                               \
            MOE_SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_V1_IMPL(quant_kernel, DTYPE_O, THREAD_DATA, BLOCK_SIZE, false, true)      \
        else                                                                                                                   \
            MOE_SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_V1_IMPL(quant_kernel, DTYPE_O, THREAD_DATA, BLOCK_SIZE, false, false)     \
    }

#define MOE_SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_V1_DISPATCH(quant_kernel, DTYPE_O, cols)           \
    if(cols <= 4 * BlockSize)                                                                \
    {                                                                                        \
        MOE_SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_V1_IMPL_(quant_kernel, DTYPE_O, 8, BlockSize /2)      \
    }                                                                                        \
    else if(cols <= 8 * BlockSize)                                                                \
    {                                                                                        \
        MOE_SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_V1_IMPL_(quant_kernel, DTYPE_O, 8, BlockSize)      \
    }                                                                                        \
    else if(cols <= 16 * BlockSize)                                                          \
    {                                                                                        \
        MOE_SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_V1_IMPL_(quant_kernel, DTYPE_O, 16, BlockSize)     \
    }                                                                                        \
    else if(cols <= 16 * BlockSize * 2)                                                      \
    {                                                                                        \
        MOE_SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_V1_IMPL_(quant_kernel, DTYPE_O, 16, BlockSize * 2) \
    }                                                                                        \
    else                                                                                     \
    {                                                                                        \
        AITER_CHECK(false, "input last dim has exceeded the maximum value ", 32 * BlockSize); \
    }

void moe_smooth_per_token_scaled_quant_v1(
    aiter_tensor_t& out,         // [..., d]
    const aiter_tensor_t& input, // [..., d]
    aiter_tensor_t& scales,
    const aiter_tensor_t& smooth_scale,
    const aiter_tensor_t& smooth_scale_map, // topk_ids
    bool shuffle_scale,
    std::optional<aiter_tensor_t> smooth_scale_map_hash,
    bool transpose_out)
{
    int const cols        = input.size(-1);
    int const rows        = input.numel() / cols;
    int32_t* smooth_scale_map_ptr = reinterpret_cast<int32_t*>(smooth_scale_map.data_ptr());
    int32_t* smooth_scale_map_hash_ptr =
        smooth_scale_map_hash.has_value() ? reinterpret_cast<int32_t*>(smooth_scale_map_hash->data_ptr()) : nullptr;
    int m_repeat = out.numel() / (rows * cols);
    int32_t input_stride = input.stride(-2);
    constexpr int32_t MAX_EXPERT_SIZE = 1024;
    int32_t smooth_scale_map_hash_size =
        smooth_scale_map_hash.has_value() ? smooth_scale_map_hash->numel() : 0;
    AITER_CHECK(out.is_contiguous(), __func__, " out is not contiguous");
    AITER_CHECK(
        smooth_scale_map_hash_size <= MAX_EXPERT_SIZE, __func__, " smooth_scale_map_hash_size is too large, only support <= ", MAX_EXPERT_SIZE);
    const bool transpose_out_dim01 = transpose_out;

    HipDeviceGuard device_guard(input.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    if(out.dtype() == AITER_DTYPE_fp8)
    {
        MOE_SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_V1_DISPATCH(
            moe_smooth_per_token_scaled_quant_kernel_v1, opus::fp8_t, cols);
    }
    else if(out.dtype() == AITER_DTYPE_i8)
    {
        MOE_SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_V1_DISPATCH(
            moe_smooth_per_token_scaled_quant_kernel_v1, opus::i8_t, cols);
    }
#if defined(__Float4_e2m1fn_x2)
    else if(out.dtype() == AITER_DTYPE_fp4x2 || out.dtype() == AITER_DTYPE_u8)
    {
        MOE_SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_V1_DISPATCH(
            moe_smooth_per_token_scaled_quant_kernel_v1, opus::fp4_t, cols);
    }
#endif
    else
    {
        AITER_CHECK(false, __func__, " not support output type: ", AiterDtype_to_str(out.dtype()));
    }
}


template <typename DTYPE_I, typename DTYPE_O, int block_size, int thread_data_size = 16>
__global__ void moe_smooth_per_token_scaled_quant_kernel_v2(DTYPE_O* __restrict__ out,
                                                            float* __restrict__ scale,
                                                            DTYPE_I* __restrict__ input,
                                                            float* __restrict__ smooth_scale,
                                                            int* __restrict__ sorted_token_ids,
                                                            int* __restrict__ sorted_expert_ids,
                                                            int* __restrict__ num_valid_ids,
                                                            const int32_t num_experts,
                                                            const int32_t num_tokens,
                                                            const int32_t num_blocks,
                                                            const int32_t num_tg,
                                                            const int32_t cols,
                                                            const int32_t topk,
                                                            const int32_t block_m,
                                                            const int32_t block_m_log2split,
                                                            const int32_t input_stride0,
                                                            const int32_t input_stride1,
                                                            const bool shuffle_scale,
                                                            const bool transpose_out_dim01)
{
    int num_valid_ids_value = num_valid_ids[0];
    int block_idx = blockIdx.x;
    const int32_t sub_block_m = block_m >> block_m_log2split;
    for(; block_idx < num_blocks; block_idx += num_tg)
    {
        int sorted_ids_offset = block_idx * sub_block_m;
        if (sorted_ids_offset >= num_valid_ids_value)
        {
            return;
        }
        int lane_idx = threadIdx.x % WARP_SIZE;
        static constexpr int32_t vec_size_i =
            thread_data_size == 0 ? 16 / sizeof(DTYPE_I) : thread_data_size;
        static constexpr int32_t load_chunk_bytes =
            (sizeof(DTYPE_I) * vec_size_i % 16 == 0 ? 16 : (sizeof(DTYPE_I) * vec_size_i % 8 == 0 ? 8 : 4));
        auto buffer_token_ids = opus::make_gmem<int>(sorted_token_ids + sorted_ids_offset, sub_block_m * sizeof(int));
        int token_id_info_list = buffer_token_ids.load(lane_idx)[0];
        int expert_id = sorted_expert_ids[block_idx >> block_m_log2split];
        if (expert_id >= num_experts)
        {
            return;
        }
        using vec_i = opus::vector_t<DTYPE_I, vec_size_i>;
        using vec_f = opus::vector_t<float, vec_size_i>;
        const float inverted_DTYPE_MAX =
            (1. / static_cast<float>(opus::finfo<DTYPE_O>::max()));
        auto buffer_smscale = opus::make_gmem<float>(smooth_scale + expert_id * cols, cols * sizeof(float));
        vec_f smscale = load_vector_nbytes<float, thread_data_size, 16>(buffer_smscale, threadIdx.x * vec_size_i);
        int token_id_list = token_id_info_list & 0xFFFFFF;
        int topk_id_list = token_id_info_list >> 24;
        for(int i = 0; i < sub_block_m; i++)
        { 
            int token_idx = __builtin_amdgcn_readlane(token_id_list, i);
            int topk_id = __builtin_amdgcn_readlane(topk_id_list, i);
            if(token_idx >= num_tokens)
            {
                break;
            }
            int64_t input_offset = (int64_t)token_idx * (int64_t)input_stride0 + (int64_t)(topk_id * input_stride1);
            auto buffer_input = opus::make_gmem<DTYPE_I>(input + input_offset, cols * sizeof(DTYPE_I));
            vec_i vec_input = load_vector_nbytes<DTYPE_I, vec_size_i, load_chunk_bytes, RT>(buffer_input, threadIdx.x * vec_size_i);
            vec_f vec_input_f;
            float* input_f_ptr = reinterpret_cast<float*>(&vec_input_f);
            for(int i = 0; i < vec_size_i; i++)
            {
                vec_input_f[i] = static_cast<float>(vec_input[i]);
            }
            float absMax = 1e-10f;
            #pragma unroll
            for(int j = 0; j < vec_size_i; j++)
            {
                vec_input_f[j] = vec_input_f[j] * smscale[j];
                absMax         = max(absMax, abs(vec_input_f[j]));
            }
            absMax = block_reduce<float, aiter::Max, block_size, true>(absMax, aiter::Max());

            float row_scale = std::is_same_v<DTYPE_O, opus::fp4_t>
                                ? aiter::fp4_f32_to_e8m0_scale(absMax)
                                : absMax * inverted_DTYPE_MAX;
            
            int out_token_idx;
            if (transpose_out_dim01)
            {   
                out_token_idx = topk_id * num_tokens + token_idx;
            }
            else
            {
                out_token_idx = token_idx * topk + topk_id;
            }
            if(threadIdx.x == 0)
            {
                if constexpr(std::is_same_v<DTYPE_O, opus::fp4_t>)
                {
                    auto* tmp        = reinterpret_cast<uint8_t*>(scale);
                    uint8_t exponent = (__builtin_bit_cast(uint32_t, row_scale) >> 23) & 0b11111111;
                    tmp[out_token_idx]   = exponent;
                }
                else
                {
                    scale[out_token_idx] = row_scale;
                }
            }
            int64_t out_offset = (int64_t)out_token_idx * (int64_t)cols;    
            scaled_quant_vgpr_impl<float, DTYPE_O, thread_data_size>(out, input_f_ptr, &row_scale, cols, out_offset);
        }
    }
}


#define MOE_SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_V2_IMPL(quant_kernel, DTYPE_O, THREAD_DATA, BLOCK_SIZE)  \
    AITER_DISPATCH_FLOATING16_TYPES_rmTorch(input.dtype(), "quant_kernel", [&] {                           \
        using input_dtype = typename aiter::hip2opus<scalar_t>::type;                                      \
        int blocks_per_cu = 8 * 4 / (BLOCK_SIZE / WARP_SIZE);                                             \
        int num_tg = persistent_mode ? num_cu * blocks_per_cu : num_blocks;                               \
        dim3 const grid(num_tg);                                                                          \
        aiter::quant_kernel<input_dtype, DTYPE_O, BLOCK_SIZE, THREAD_DATA>                                \
            <<<grid, dim3(BLOCK_SIZE), 0, stream>>>(                                                      \
                reinterpret_cast<DTYPE_O*>(out.data_ptr()),                                               \
                reinterpret_cast<float*>(scales.data_ptr()),                                              \
                reinterpret_cast<input_dtype*>(input.data_ptr()),                                         \
                reinterpret_cast<float*>(smooth_scale.data_ptr()),                                        \
                reinterpret_cast<int*>(sorted_token_ids.data_ptr()),                                      \
                reinterpret_cast<int*>(sorted_expert_ids.data_ptr()),                                     \
                reinterpret_cast<int*>(num_valid_ids.data_ptr()),                                         \
                num_experts,                                                                              \
                num_tokens,                                                                               \
                num_blocks,                                                                               \
                num_tg,                                                                                   \
                cols,                                                                                     \
                topk,                                                                                     \
                block_m,                                                                                  \
                block_m_log2split,                                                                        \
                input_stride0,                                                                            \
                input_stride1,                                                                            \
                shuffle_scale,                                                                            \
                transpose_out);                                                                           \
    });


#define MOE_SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_V2_DISPATCH(quant_kernel, DTYPE_O, cols)           \
    if(cols <= 4 * BlockSize)                                                                \
    {                                                                                        \
        MOE_SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_V2_IMPL(quant_kernel, DTYPE_O, 8, BlockSize /2)      \
    }                                                                                        \
    else if(cols <= 8 * BlockSize)                                                                \
    {                                                                                        \
        MOE_SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_V2_IMPL(quant_kernel, DTYPE_O, 8, BlockSize)      \
    }                                                                                        \
    else if(cols <= 16 * BlockSize)                                                          \
    {                                                                                        \
        MOE_SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_V2_IMPL(quant_kernel, DTYPE_O, 16, BlockSize)     \
    }                                                                                        \
    else if(cols <= 16 * BlockSize * 2)                                                      \
    {                                                                                        \
        MOE_SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_V2_IMPL(quant_kernel, DTYPE_O, 16, BlockSize * 2) \
    }                                                                                        \
    else                                                                                     \
    {                                                                                        \
        AITER_CHECK(false, "input last dim has exceeded the maximum value ", 32 * BlockSize); \
    }


void moe_smooth_per_token_scaled_quant_v2(
    aiter_tensor_t& out,         // [..., d]
    const aiter_tensor_t& input, // [..., d]
    aiter_tensor_t& scales,
    const aiter_tensor_t& smooth_scale,
    const aiter_tensor_t& sorted_token_ids,
    const aiter_tensor_t& sorted_expert_ids,
    const aiter_tensor_t& num_valid_ids,
    int block_m,
    bool shuffle_scale,
    bool transpose_out)
{
    AITER_CHECK(out.is_contiguous());
    int cols = input.size(-1);
    int num_tokens = input.size(0);
    int num_experts = smooth_scale.size(0);
    int topk = out.numel() / (num_tokens * cols);
    int input_stride0= input.stride(0);
    int input_stride1= input.dim() == 2 ? 0 : input.stride(1);

    const int num_cu = get_num_cu_func();
    int block_split = 16;
    int block_m_log2split = log2(block_split);
    AITER_CHECK(block_m % block_split == 0, __func__, " block_m is not divisible by block_split");
    int sub_block_m = block_m >> block_m_log2split;
    int num_blocks = sorted_expert_ids.size(0) * block_split;
    const bool persistent_mode = true;

    HipDeviceGuard device_guard(input.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    if(out.dtype() == AITER_DTYPE_fp8)
    {
        MOE_SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_V2_DISPATCH(
            moe_smooth_per_token_scaled_quant_kernel_v2, opus::fp8_t, cols);
    }
    else if(out.dtype() == AITER_DTYPE_i8)
    {
        MOE_SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_V2_DISPATCH(
            moe_smooth_per_token_scaled_quant_kernel_v2, opus::i8_t, cols);
    }
#if defined(__Float4_e2m1fn_x2)
    else if(out.dtype() == AITER_DTYPE_fp4x2 || out.dtype() == AITER_DTYPE_u8)
    {
        MOE_SMOOTH_PER_TOKEN_SCALED_QUANT_KERNEL_V2_DISPATCH(
            moe_smooth_per_token_scaled_quant_kernel_v2, opus::fp4_t, cols);
    }
#endif
    else
    {
        AITER_CHECK(false, __func__, " not support output type: ", AiterDtype_to_str(out.dtype()));
    }
}


// Fused dynamic MX (fp4 / fp8) quantization + MoE-sort writeback.
// Template parameter DTYPE_O selects the element format (opus::fp4_t for MXFP4,
// opus::fp8_t for MXFP8); both paths emit the same E8M0 scale byte layout via
// `aiter::mx_scale_shuffle_idx`. The legacy kernel name
// `mxfp4_quant_moe_sort_kernel` was misleading because it implied fp4-only
// — the implementation has always been dtype-templated.
template <typename DTYPE_O, int thread_data_size>
__device__ void store_zero_mx_quant_moe_sort_row(
    DTYPE_O* __restrict__ out,
    uint8_t* __restrict__ scale,
    const int sorted_row,
    const int32_t scaleN_pad,
    const int32_t scaleN_valid,
    const int scale_k,
    const int num_thread_per_group,
    const int topk_id,
    const int topk,
    const int64_t offset_base,
    const int cols)
{
    if(threadIdx.x % num_thread_per_group == 0 && scale_k < scaleN_valid)
    {
        int addr    = aiter::mx_scale_shuffle_idx(scaleN_pad, sorted_row, scale_k);
        scale[addr] = 0;
    }
    if(topk_id < topk || topk == 1)
    {
        const int64_t row_bytes = std::is_same_v<DTYPE_O, opus::fp4_t> ? cols / 2 : cols;
        uint8_t* out_u8 = reinterpret_cast<uint8_t*>(out);
        const int64_t row_offset = offset_base * row_bytes;
        auto buffer_o = opus::make_gmem<uint8_t>(out_u8 + row_offset, row_bytes);

        static constexpr int32_t zero_vec_bytes = 16;
        opus::vector_t<uint8_t, zero_vec_bytes> zero_vec;
#pragma unroll
        for(int j = 0; j < zero_vec_bytes; j++)
        {
            zero_vec[j] = 0;
        }

        const int32_t num_vecs = (row_bytes + zero_vec_bytes - 1) / zero_vec_bytes;
        for(int32_t vec_idx = threadIdx.x; vec_idx < num_vecs; vec_idx += blockDim.x)
        {
            buffer_o.template store<zero_vec_bytes>(zero_vec, vec_idx * zero_vec_bytes);
        }
    }
}

template <typename DTYPE_I, typename DTYPE_O, int block_size, int thread_data_size = 16>
__global__ void fused_mx_quant_moe_sort_kernel(
    DTYPE_O* __restrict__ out,
    uint8_t* __restrict__ scale,
    DTYPE_I const* __restrict__ input,
    int32_t const* __restrict__ sorted_ids,
    int32_t const* __restrict__ num_valid_ids,
    float const* __restrict__ sorted_weights,
    const int32_t num_tokens,
    const int32_t cols,
    const int32_t group_size,
    const int32_t tgs_per_block_m,
    const int32_t sub_block_m,
    const int32_t num_blocks,
    const int32_t num_tg,
    const int32_t topk,
    const int32_t input_stride)
{
    int num_thread_per_group = group_size / thread_data_size;
    int num_valid_ids_value  = num_valid_ids[0];
    int block_idx            = blockIdx.x;
    int lane_idx             = threadIdx.x % WARP_SIZE;
    const int scale_k        = threadIdx.x / num_thread_per_group;
    static constexpr int32_t vec_size_i =
        thread_data_size == 0 ? 16 / sizeof(DTYPE_I) : thread_data_size;
    static constexpr int32_t load_chunk_bytes =
        (sizeof(DTYPE_I) * vec_size_i % 16 == 0 ? 16
                                                : (sizeof(DTYPE_I) * vec_size_i % 8 == 0 ? 8 : 4));
    using vec_i = opus::vector_t<DTYPE_I, vec_size_i>;
    using vec_f = opus::vector_t<float, vec_size_i>;
    // Continuous fp32 scale divisor for non-MX dtypes (e.g. int8). MX dtypes
    // (fp4 / fp8) take the e8m0 path via fp_f32_to_e8m0_scale<RoundUp, dtype>
    // below, which returns a pure pow-2 dequant scale directly.
    const float inverted_DTYPE_MAX =
        1.0f / static_cast<float>(opus::finfo<DTYPE_O>::max());

    // HW-native FP8 element dtype: gfx942 ships e4m3fnuz (max_pos=240),
    // gfx950+ ships OCP e4m3fn (max_pos=448). The legacy
    // ``fp_f32_to_e8m0_scale<RoundUp, FP4>(absMax) * 1/floor_pow2(MAX)`` formula here used
    // to over-scale the FP8 working value by ~2x (factor*amax > max_pos),
    // saturating the high tail; emit_mx_e8m0_scale<RoundUp, dtype> picks the
    // correct ``ceil_pow2(amax / max_pos)`` per arch instead.
    constexpr aiter::MxDtype kHwFp8Dtype =
#if defined(__gfx942__)
        aiter::MxDtype::FP8_E4M3_FNUZ;
#else
        aiter::MxDtype::FP8_E4M3;
#endif

    const int32_t scaleN_valid = (cols + group_size - 1) / group_size;
    const int32_t scaleN_pad   = ((scaleN_valid + 7) / 8) * 8;

    for(; block_idx < num_blocks; block_idx += num_tg)
    {
        int sub_idx         = block_idx % tgs_per_block_m;
        int block_m_start   = (block_idx - sub_idx) * sub_block_m;
        int sorted_ids_base = block_m_start + sub_idx;
        if(sorted_ids_base >= num_valid_ids_value)
        {
            return;
        }
        int token_id_info_list;
        if (lane_idx < sub_block_m)
        {
            int strided_idx = sorted_ids_base + lane_idx * tgs_per_block_m;
            token_id_info_list = (strided_idx < num_valid_ids_value)
                ? sorted_ids[strided_idx]
                : num_tokens;
        }
        int token_id_list = token_id_info_list & 0xFFFFFF;
        int topk_id_list  = token_id_info_list >> 24;
        for(int i = 0; i < sub_block_m; i++)
        {
            int token_idx = __builtin_amdgcn_readlane(token_id_list, i);
            int topk_id   = __builtin_amdgcn_readlane(topk_id_list, i);
            if(token_idx >= num_tokens)
            {
                break;
            }

            int64_t offset_base = topk == 1 ? (int64_t)(token_idx) : (int64_t)(token_idx * topk + topk_id);
            const int sorted_row = sorted_ids_base + i * tgs_per_block_m;
            if(sorted_weights != nullptr && sorted_weights[sorted_row] == 0.0f)
            {
                store_zero_mx_quant_moe_sort_row<DTYPE_O, thread_data_size>(
                    out,
                    scale,
                    sorted_row,
                    scaleN_pad,
                    scaleN_valid,
                    scale_k,
                    num_thread_per_group,
                    topk_id,
                    topk,
                    offset_base,
                    cols);
                continue;
            }
            auto buffer_input =
                opus::make_gmem<DTYPE_I>(input + offset_base * input_stride, cols * sizeof(DTYPE_I));
            vec_i vec_input = load_vector_nbytes<DTYPE_I, vec_size_i, load_chunk_bytes, RT>(
                buffer_input, threadIdx.x * vec_size_i);
            vec_f vec_input_f;
            float* input_f_ptr = reinterpret_cast<float*>(&vec_input_f);
            float absMax       = 1e-10f;
            #pragma unroll
            for(int j = 0; j < vec_size_i; j++)
            {
                vec_input_f[j] = static_cast<float>(vec_input[j]);
                absMax         = max(absMax, abs(vec_input_f[j]));
            }
            absMax = multithread_reduce(absMax, aiter::Max(), num_thread_per_group);

            // MXFP4 / MXFP8 use the project-wide default round mode
            // (kDefaultMxScaleRoundMode, currently NV ROUND_UP =
            // ceil_pow2(amax / max_pos)). The helper returns the dequant
            // scale as a pow-2 fp32, so the ``(>> 23) & 0xFF`` extraction
            // below yields the stored e8m0 byte directly. Other dtypes fall
            // back to a continuous fp32 scale.
            float row_scale;
            if constexpr (std::is_same_v<DTYPE_O, opus::fp4_t>)
            {
                row_scale = aiter::fp4_f32_to_e8m0_scale(absMax);
            }
            else if constexpr (std::is_same_v<DTYPE_O, opus::fp8_t>)
            {
                row_scale = aiter::fp_f32_to_e8m0_scale<aiter::kDefaultMxScaleRoundMode,
                                                       kHwFp8Dtype>(absMax);
            }
            else
            {
                row_scale = absMax * inverted_DTYPE_MAX;
            }

            if(threadIdx.x % num_thread_per_group == 0 && scale_k < scaleN_valid)
            {
                uint8_t bs_e8m0 = (__builtin_bit_cast(uint32_t, row_scale) >> 23) & 0xFF;
                int addr        = aiter::mx_scale_shuffle_idx(scaleN_pad, sorted_row, scale_k);
                scale[addr]     = bs_e8m0;
            }

            if(topk_id < topk || topk == 1)
            {
                scaled_quant_vgpr_impl<float, DTYPE_O, thread_data_size>(
                    out, input_f_ptr, &row_scale, cols, offset_base * cols);
            }
        }
    }
}


#define FUSED_MX_QUANT_MOE_SORT_KERNEL_IMPL(DTYPE_O, THREAD_DATA, BLOCK_SIZE)                       \
    AITER_DISPATCH_FLOATING16_TYPES_rmTorch(input.dtype(), "fused_mx_quant_moe_sort_kernel", [&] {  \
        using input_dtype = typename aiter::hip2opus<scalar_t>::type;                               \
        AITER_CHECK(group_size % THREAD_DATA == 0, __func__, " group_size is not divisible by THREAD_DATA"); \
        int blocks_per_cu = 8 * 4 / (BLOCK_SIZE / WARP_SIZE);                                       \
        int num_tg = persistent_mode ? num_cu * blocks_per_cu : num_blocks;                         \
        dim3 const grid(num_tg);                                                                    \
        fused_mx_quant_moe_sort_kernel<input_dtype, DTYPE_O, BLOCK_SIZE, THREAD_DATA>               \
            <<<grid, dim3(BLOCK_SIZE), 0, stream>>>(                                           \
                reinterpret_cast<DTYPE_O*>(output.data_ptr()),                                  \
                reinterpret_cast<uint8_t*>(scale.data_ptr()),                                   \
                reinterpret_cast<input_dtype const*>(input.data_ptr()),                         \
                reinterpret_cast<int32_t*>(sorted_ids.data_ptr()),                              \
                reinterpret_cast<int32_t*>(num_valid_ids.data_ptr()),                           \
                sorted_weights_ptr,                                                            \
                token_num,                                                                      \
                cols,                                                                           \
                group_size,                                                                     \
                tgs_per_block_m,                                                                \
                sub_block_m,                                                                     \
                num_blocks,                                                                     \
                num_tg,                                                                         \
                topk,                                                                           \
                input_stride);                                                                  \
    });


#define FUSED_MX_QUANT_MOE_SORT_KERNEL_DISPATCH(DTYPE_O, cols_)                                \
    if(cols_ <= 2 * BlockSize)                                                                 \
    {                                                                                          \
        FUSED_MX_QUANT_MOE_SORT_KERNEL_IMPL(DTYPE_O, 8, BlockSize / 4)                         \
    }                                                                                          \
    else if(cols_ <= 4 * BlockSize)                                                            \
    {                                                                                          \
        FUSED_MX_QUANT_MOE_SORT_KERNEL_IMPL(DTYPE_O, 8, BlockSize / 2)                         \
    }                                                                                          \
    else if(cols_ <= 8 * BlockSize)                                                            \
    {                                                                                          \
        FUSED_MX_QUANT_MOE_SORT_KERNEL_IMPL(DTYPE_O, 8, BlockSize)                             \
    }                                                                                          \
    else if(cols_ <= 16 * BlockSize)                                                           \
    {                                                                                          \
        FUSED_MX_QUANT_MOE_SORT_KERNEL_IMPL(DTYPE_O, 16, BlockSize)                            \
    }                                                                                          \
    else if(cols_ <= 16 * BlockSize * 2)                                                       \
    {                                                                                          \
        FUSED_MX_QUANT_MOE_SORT_KERNEL_IMPL(DTYPE_O, 32, BlockSize)                            \
    }                                                                                          \
    else                                                                                       \
    {                                                                                          \
        AITER_CHECK(false, "input last dim has exceeded the maximum value ", 32 * BlockSize);  \
    }

void fused_dynamic_mx_quant_moe_sort_hip(
    aiter_tensor_t& output,
    aiter_tensor_t& scale,
    const aiter_tensor_t& input,
    const aiter_tensor_t& sorted_ids,
    const aiter_tensor_t& num_valid_ids,
    int token_num,
    int block_m,
    int group_size,
    std::optional<aiter_tensor_t> sorted_weights
)
{
    int cols = input.size(-1);
    int topk = input.numel() / (cols * token_num);
    int num_experts = (sorted_ids.size(0) + topk - topk * token_num) / block_m;
    if(sorted_weights.has_value())
    {
        AITER_CHECK(sorted_weights->dtype() == AITER_DTYPE_fp32,
                    __func__,
                    " sorted_weights must be fp32 when provided");
        AITER_CHECK(sorted_weights->numel() >= sorted_ids.size(0),
                    __func__,
                    " sorted_weights must have at least sorted_ids.size(0) elements");
    }
    float const* sorted_weights_ptr =
        sorted_weights.has_value() ? reinterpret_cast<float const*>(sorted_weights->data_ptr()) : nullptr;

    const int num_cu = get_num_cu_func();
    int sub_block_m = (token_num * topk) > (num_cu * 8) || num_experts < 64 ? 2 : 4;
    AITER_CHECK(block_m % sub_block_m == 0, __func__, " block_m is not divisible by sub_block_m");
    int tgs_per_block_m = block_m / sub_block_m;
    int num_blocks = (sorted_ids.size(0) + sub_block_m - 1) / sub_block_m;
    const bool persistent_mode = false;
    const int input_stride     = input.stride(-2);

    HipDeviceGuard device_guard(input.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    if(output.dtype() == AITER_DTYPE_fp8)
    {
        FUSED_MX_QUANT_MOE_SORT_KERNEL_DISPATCH(opus::fp8_t, cols);
    }
#if defined(__Float4_e2m1fn_x2)
    else if(output.dtype() == AITER_DTYPE_fp4x2 || output.dtype() == AITER_DTYPE_u8)
    {
        FUSED_MX_QUANT_MOE_SORT_KERNEL_DISPATCH(opus::fp4_t, cols);
    }
#endif
    else
    {
        AITER_CHECK(false, __func__, ": not support output type: ", AiterDtype_to_str(output.dtype()));
    }
}

// Perf gate threshold for the coalesced LDS-staged store path in
// mxfp4_moe_sort_kernel (see below). The LDS staging + double __syncthreads
// cost scales with the LDS footprint per row (scaleN_pad bytes) times the
// number of grid tiles (num_blocks). On MI355X/gfx950 uncoalesced byte stores
// are already cheap, so the coalesced path is a net win for small/mid
// (scaleN_pad x num_blocks) but loses once that product is large (empirically
// the only regressor is dim=7168 @ M=16384: 224*4352=974,848; the largest
// winner is dim=7168 @ M=8192: 224*2304=516,096). The threshold sits between
// them with ~35% margin. This is a PURE PERFORMANCE knob: both the coalesced
// and the scatter paths are byte-exact, so the value only affects speed, never
// correctness.
constexpr long MXFP4_MOE_SORT_COALESCED_LDS_WORK_MAX = 700000;

template <int block_size, int num_rows, int thread_data_size = 16, int group_size = 32>
__global__ void mxfp4_moe_sort_kernel(
    uint8_t* __restrict__ out_scale,
    uint8_t* __restrict__ scale,
    int32_t const* __restrict__ sorted_ids,
    int32_t const* __restrict__ num_valid_ids,
    const int32_t num_tokens,
    const int32_t cols,
    const int32_t num_blocks,
    const int32_t num_tg,
    const int32_t topk)
{
    constexpr int threads_per_row = block_size / num_rows;
    int num_valid_ids_value  = num_valid_ids[0];
    int block_idx            = blockIdx.x;
    int row_i                = threadIdx.x / threads_per_row;
    int scale_k              = threadIdx.x % threads_per_row * thread_data_size;
    const int scale_per_row = (cols + group_size - 1) / group_size;
    static constexpr int32_t vec_size_i = thread_data_size;
    static constexpr int32_t load_chunk_bytes =
        (sizeof(uint8_t) * vec_size_i % 16 == 0 ? 16
                                                : (sizeof(uint8_t) * vec_size_i % 8 == 0 ? 8 
                                                : (sizeof(uint8_t) * vec_size_i % 4 == 0 ? 4 : 2)));
    using vec_i = opus::vector_t<uint8_t, vec_size_i>;
    const int32_t scaleN_valid = (cols + group_size - 1) / group_size;
    const int32_t scaleN_pad   = ((scaleN_valid + 7) / 8) * 8;
    auto buffer_scale =
                opus::make_gmem<uint8_t>(scale, scale_per_row * num_tokens * topk * sizeof(uint8_t));

    // Optimized store path: when one threadblock maps to exactly one 32-row
    // swizzle tile (num_rows == 32), we stage this block's scale bytes into
    // LDS in natural [row][col] order, then emit the swizzled out_scale layout
    // with fully-coalesced 4-byte (dword) stores instead of 32 strided 1-byte
    // scatters. The swizzle address for a fixed 32-row tile decomposes so that
    // each contiguous 256-byte region == (one tile) x (8 columns), and one
    // aligned dword at offset (y_blk*256 + c*64 + xl_lo*4) packs exactly:
    //   byte0 = data[xl_lo   ][col_a]   byte1 = data[xl_lo+16][col_a]
    //   byte2 = data[xl_lo   ][col_a+4] byte3 = data[xl_lo+16][col_a+4]
    // with col_a = y_blk*8 + c. So 64 consecutive dwords fill a 256B block.
    //
    // The coalesced path is gated by a pure perf heuristic computed here from
    // values already in scope (scaleN_pad = LDS footprint per row, num_blocks =
    // grid tiles); see MXFP4_MOE_SORT_COALESCED_LDS_WORK_MAX above. When the
    // gate is off we fall through to the original per-byte scatter path below.
    // Both paths are byte-exact; the gate only selects the faster one. The
    // (long) casts avoid int32 overflow of the product.
    if constexpr (num_rows == 32)
    {
      if(((long)scaleN_pad * (long)num_blocks) <= MXFP4_MOE_SORT_COALESCED_LDS_WORK_MAX)
      {
        // LDS holds 32 rows x scaleN_pad bytes (upper bound is compile-time:
        // threads_per_row * thread_data_size columns). Zero-staged so invalid
        // rows and padding columns store 0 (byte-exact with the reference,
        // whose buffer is zero-initialised).
        //
        // The LDS row stride is PADDED (LDS_STRIDE = lds_cols + LDS_PAD), kept
        // separate from the output stride scaleN_pad. Without padding, a
        // scaleN_pad that is a multiple of 32 dwords (e.g. dim=4096 ->
        // scaleN_pad=128 = 32 LDS banks) makes all 32 rows of a tile alias onto
        // the same LDS bank, causing a ~32-way bank conflict in the packed
        // read-out (s_scale[xl_lo*..] and s_scale[(xl_lo+16)*..]). Padding the
        // stride by 4 bytes (one dword) breaks that aliasing while preserving
        // dword alignment of the staged reads. This is invisible to the output:
        // out_scale addressing still uses scaleN_pad, so the result is
        // byte-identical; only the in-LDS layout changes.
        constexpr int lds_cols   = threads_per_row * thread_data_size;
        constexpr int LDS_PAD    = 4;
        constexpr int LDS_STRIDE = lds_cols + LDS_PAD;
        __shared__ uint8_t s_scale[num_rows * LDS_STRIDE];

        for(; block_idx < num_blocks; block_idx += num_tg)
        {
            // Skip tiles whose 32 rows are entirely in the padding region
            // (all rows >= num_valid_ids -> all invalid). The output buffer is
            // pre-zeroed, so leaving such tiles untouched is byte-exact with the
            // reference, and matches the original kernel's per-row guard. This
            // avoids streaming zeros to the large E*block_m padding region.
            // block_idx is uniform across the block, so no __syncthreads hazard.
            if(block_idx * num_rows >= num_valid_ids_value)
            {
                continue;
            }
            int sorted_row = block_idx * num_rows + row_i;
            int token_id_info = num_tokens;
            if (sorted_row < num_valid_ids_value)
            {
                token_id_info = sorted_ids[sorted_row];
            }
            int token_idx = token_id_info & 0xFFFFFF;
            int topk_id   = token_id_info >> 24;
            bool valid = (token_idx < num_tokens && (topk == 1 || topk_id < topk));

            vec_i vec_scale;
            if(valid)
            {
                int64_t scale_offset;
                if (topk == 1)
                {
                    scale_offset = (int64_t)(token_idx) * scale_per_row;
                }
                else
                {
                    scale_offset = (int64_t)(token_idx * topk + topk_id) * scale_per_row;
                }
                vec_scale = load_vector_nbytes<uint8_t, vec_size_i, load_chunk_bytes, RT>(
                    buffer_scale, scale_offset + scale_k);
            }

            __syncthreads();
            for(int j = 0; j < vec_size_i; j++)
            {
                int col = scale_k + j;
                if(col < scaleN_pad)
                {
                    s_scale[row_i * LDS_STRIDE + col] =
                        (valid && col < scaleN_valid) ? vec_scale[j] : (uint8_t)0;
                }
            }
            __syncthreads();

            const int ngroups   = scaleN_pad / 8;
            const int total_dw  = ngroups * 64;
            const int64_t tile_base = (int64_t)block_idx * scaleN_pad * 32;
            for(int d = threadIdx.x; d < total_dw; d += block_size)
            {
                int y_blk  = d >> 6;
                int t      = d & 63;
                int xl_lo  = t & 15;
                int c      = t >> 4;
                int col_a  = y_blk * 8 + c;
                int col_b  = col_a + 4;
                uint32_t b0 = s_scale[xl_lo * LDS_STRIDE + col_a];
                uint32_t b1 = s_scale[(xl_lo + 16) * LDS_STRIDE + col_a];
                uint32_t b2 = s_scale[xl_lo * LDS_STRIDE + col_b];
                uint32_t b3 = s_scale[(xl_lo + 16) * LDS_STRIDE + col_b];
                uint32_t packed = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
                int64_t addr = tile_base + (int64_t)y_blk * 256 + (int64_t)c * 64 + xl_lo * 4;
                *reinterpret_cast<uint32_t*>(out_scale + addr) = packed;
            }
        }
        return;
      }
    }

    for(; block_idx < num_blocks; block_idx += num_tg)
    {
        int sorted_row = block_idx * num_rows + row_i;
        int token_id_info = num_tokens;
        if (sorted_row < num_valid_ids_value)
        {
            token_id_info = sorted_ids[sorted_row];
        }
        int token_idx = token_id_info & 0xFFFFFF;
        int topk_id   = token_id_info >> 24;
        if(token_idx < num_tokens && (topk == 1 || topk_id < topk))
        {
            int64_t scale_offset;
            if (topk == 1)
            {
                scale_offset = (int64_t)(token_idx) * scale_per_row;
            }
            else
            {
                scale_offset = (int64_t)(token_idx * topk + topk_id) * scale_per_row;
            }
            vec_i vec_scale = load_vector_nbytes<uint8_t, vec_size_i, load_chunk_bytes, RT>(
                buffer_scale, scale_offset + scale_k);

            for(int j = 0; j < vec_size_i; j++)
            {
                if((scale_k + j) < scaleN_valid)
                {
                    int addr = aiter::mx_scale_shuffle_idx(scaleN_pad, sorted_row, scale_k + j);
                    out_scale[addr] = vec_scale[j];
                }
            }
        }
    }
}


#define MXFP4_MOE_SORT_KERNEL_IMPL(MAX_COL, THREAD_DATA, BLOCK_SIZE)                    \
    constexpr int GROUP_SIZE = 32;                                                      \
    constexpr int NUM_ROWS = BLOCK_SIZE / (MAX_COL /(GROUP_SIZE * THREAD_DATA));        \
    AITER_CHECK(BLOCK_SIZE % (MAX_COL /(GROUP_SIZE * THREAD_DATA)) == 0);               \
    int num_blocks = (sorted_ids.size(0) + NUM_ROWS - 1) / NUM_ROWS;                    \
    int blocks_per_cu = 8 * 4 / (BLOCK_SIZE / WARP_SIZE);                               \
    int num_tg = persistent_mode ? num_cu * blocks_per_cu : num_blocks;                 \
    dim3 const grid(num_tg);                                                            \
    /* The coalesced-store perf gate is computed inside the kernel from        */       \
    /* num_blocks + scaleN_pad (no extra launch arg / signature change).       */       \
    mxfp4_moe_sort_kernel<BLOCK_SIZE, NUM_ROWS, THREAD_DATA, GROUP_SIZE>                \
        <<<grid, dim3(BLOCK_SIZE), 0, stream>>>(                                        \
            reinterpret_cast<uint8_t*>(out_scale.data_ptr()),                           \
            reinterpret_cast<uint8_t*>(scale.data_ptr()),                               \
            reinterpret_cast<int32_t*>(sorted_ids.data_ptr()),                          \
            reinterpret_cast<int32_t*>(num_valid_ids.data_ptr()),                       \
            token_num, cols, num_blocks, num_tg, topk);


#define MXFP4_MOE_SORT_KERNEL_DISPATCH(cols_)                                                  \
    if(cols_ <= 256)                                                                           \
    {                                                                                          \
        MXFP4_MOE_SORT_KERNEL_IMPL(256, 4, 256)                                                \
    }                                                                                          \
    else if(cols_ <= 512)                                                                      \
    {                                                                                          \
        MXFP4_MOE_SORT_KERNEL_IMPL(512, 4, 256)                                                \
    }                                                                                          \
    else if(cols_ <= 1024)                                                                     \
    {                                                                                          \
        MXFP4_MOE_SORT_KERNEL_IMPL(1024, 4, 256)                                               \
    }                                                                                          \
    else if(cols_ <= 2048)                                                                     \
    {                                                                                          \
        MXFP4_MOE_SORT_KERNEL_IMPL(2048, 8, 256)                                               \
    }                                                                                          \
    else if(cols_ <= 4096)                                                                     \
    {                                                                                          \
        MXFP4_MOE_SORT_KERNEL_IMPL(4096, 16, 256)                                              \
    }                                                                                          \
    else if(cols_ <= 6144)                                                                     \
    {                                                                                          \
        MXFP4_MOE_SORT_KERNEL_IMPL(6144, 24, 256)                                              \
    }                                                                                          \
    else if(cols_ <= 8192)                                                                     \
    {                                                                                          \
        MXFP4_MOE_SORT_KERNEL_IMPL(8192, 32, 256)                                              \
    }                                                                                          \
    else if(cols_ <= 16384)                                                                    \
    {                                                                                          \
        MXFP4_MOE_SORT_KERNEL_IMPL(16384, 32, 256)                                             \
    }                                                                                          \
    else                                                                                       \
    {                                                                                          \
        AITER_CHECK(false, "input last dim has exceeded the maximum value ", 16384);            \
    }

void mxfp4_moe_sort_hip(
    aiter_tensor_t& out_scale,
    const aiter_tensor_t& scale,
    const aiter_tensor_t& sorted_ids,
    const aiter_tensor_t& num_valid_ids,
    int token_num,
    int cols
)
{
    const int num_cu = get_num_cu_func();
    const bool persistent_mode = false;
    int topk = scale.numel() / ((cols + 31) / 32 * token_num);

    HipDeviceGuard device_guard(scale.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    MXFP4_MOE_SORT_KERNEL_DISPATCH(cols);
}

} // namespace aiter