// SPDX-License-Identifier: MIT
// Experimental H16 four-wave geometry: QK partitions N64, PV partitions D512.
// Reuses PR #3459 HipKittens BF16 LDS readers, four-slot PV schedule, O managers.
#pragma once
#include "sparse_mla_bf16_common.cuh"

namespace sparse_mla_bf16::wave4 {
struct Traits
{
    static constexpr uint32_t kBlockN = 64, kBlockK = 32, kTileM = 16;
    static constexpr uint32_t kQkHeadDim = 512, kVoHeadDim = 512;
    static constexpr uint32_t kNumWarps = 4, kNumThreads = 256, kRoundMode = 0;
    static constexpr uint32_t kKvBytes = 65536, kPBytes = 2048;
    static constexpr uint32_t kPOffset      = kKvBytes;
    static constexpr uint32_t kReduceOffset = kPOffset + kPBytes;
    static constexpr uint32_t kIndexOffset  = kReduceOffset + 512;
    static constexpr uint32_t kLdsBytes     = kIndexOffset + 256;
    static_assert(kLdsBytes == 68352);
};
template <uint32_t Lo, uint32_t Hi>
using ranges = hkdart::split_many_t<hkdart::type_list<hkdart::range<Lo, Hi>>, 4>;
struct Regs
{
    using comp_t                        = float;
    static constexpr uint32_t kNumKvSub = 2;
    static constexpr uint32_t k_o_begin = 128, k_v0_begin = 52;
    static constexpr uint32_t k_k0_begin = 40, k_k1_begin = 44, k_k2_begin = 48;
    using all_ranges = ranges<40, 159>;
    using o_ranges   = ranges<128, 159>;
    using oaccu_t    = hk::art<float, 16, 128, hk::row_l, hk::rt_16x16_s, o_ranges>;
    using score_t    = hk::art<float, 16, 16, hk::col_l, hk::rt_16x16_s, ranges<56, 59>>;
    using p_mfma_a_t = hk::art<hk::bf16, 16, 32, hk::row_l, hk::rt_16x32_s, ranges<56, 59>>;
    using p_mfma_b_t = hk::art<hk::bf16, 16, 32, hk::row_l, hk::rt_16x32_s, ranges<60, 63>>;
    using pv_v_0_t   = hk::art<hk::bf16, 16, 32, hk::row_l, hk::rt_16x32_s, ranges<52, 55>>;
    using pv_v_1_t   = hk::art<hk::bf16, 16, 32, hk::row_l, hk::rt_16x32_s, ranges<40, 43>>;
    using pv_v_2_t   = hk::art<hk::bf16, 16, 32, hk::row_l, hk::rt_16x32_s, ranges<44, 47>>;
    using pv_v_3_t   = hk::art<hk::bf16, 16, 32, hk::row_l, hk::rt_16x32_s, ranges<48, 51>>;
};

template <bool kIsFirstIter, bool kDoRescale, typename T>
__device__ __forceinline__ void
pv_gemm(Bf16KvManager<T>& kv_manager, const uintptr_t p_lds_v, const float rescale)
{
    using R                       = Regs;
    using comp_t                  = typename R::comp_t;
    constexpr uint32_t k_o_begin  = R::k_o_begin;
    constexpr uint32_t k_v0_begin = R::k_v0_begin;
    constexpr uint32_t k_k0_begin = R::k_k0_begin;
    constexpr uint32_t k_k1_begin = R::k_k1_begin;
    constexpr uint32_t k_k2_begin = R::k_k2_begin;
    // kBlockN=64 (m16x8 double-tile) contracts BOTH 32-row sub-tiles in one call:
    // sub-tile A (rows 0:32) uses p_mfma_a, B (rows 32:64) uses p_mfma_b. kBlockN=32
    // has only A. Picked per-iter by row_base (see below), no caller arg.
    typename R::p_mfma_a_t p_mfma_a;
    typename R::p_mfma_b_t p_mfma_b;
    typename R::pv_v_0_t pv_v_0;
    typename R::pv_v_1_t pv_v_1;
    typename R::pv_v_2_t pv_v_2;
    typename R::pv_v_3_t pv_v_3;

    // D-tiling: each D-iter emits 2 oaccu 16x16 sub-tiles (32 D-cols); 512/32 = 16
    // D-iters per 32-row KV sub-tile. kBlockN=64 runs BOTH sub-tiles in one call, so
    // num_pv_iter = kNumKvSub * 16. iter [0,16) = sub-tile A, [16,32) = sub-tile B.
    constexpr uint32_t kDIters     = 128u / (2u * T::kTileM); // 16
    constexpr uint32_t num_pv_iter = R::kNumKvSub * kDIters;  // 16 or 32
    // Flat V base tiles across all sub-tiles: 32 per sub-tile (S_0..S_31 each).
    constexpr uint32_t kVTilesPerSub = 2u * kDIters;                 // 32
    constexpr uint32_t kNumVTiles    = R::kNumKvSub * kVTilesPerSub; // 32 or 64

    auto pk_mul_pair = [&](float r, auto base_c) {
        constexpr uint32_t base = decltype(base_c)::value;
        const float2 r2         = {r, r};
        asm volatile("v_pk_mul_f32 v[%0:%1], %2, v[%0:%1]" : : "n"(base), "n"(base + 1), "v"(r2));
    };
    auto mul_pair = [&](float r, auto base_c) {
        constexpr uint32_t base = decltype(base_c)::value;
        asm volatile("v_mul_f32_e32 v[%0], %1, v[%0]" : : "n"(base), "v"(r));
        asm volatile("v_mul_f32_e32 v[%0], %1, v[%0]" : : "n"(base + 1), "v"(r));
    };

    // Issue both ds_read_b64 for base tile S_jj (within its sub-tile) into a
    // round-robin slot (4 slots). kRowBase selects the KV sub-tile (0=A, 32=B),
    // folded into the V ds_read imm by load_transposed_v_to_gpr.
    auto load_S = [&]<uint32_t kRowBase, uint32_t jj, uint32_t slot>() {
        constexpr uint32_t base = (slot == 0u)   ? k_v0_begin
                                  : (slot == 1u) ? k_k0_begin
                                  : (slot == 2u) ? k_k1_begin
                                                 : k_k2_begin;
        kv_manager.template load_transposed_v_to_gpr<kRowBase + 0u, jj * 16u, base + 0>(p_lds_v);
        kv_manager.template load_transposed_v_to_gpr<kRowBase + 16u, jj * 16u, base + 2>(p_lds_v);
    };
    // mfma: oaccu_dst (+)= pv_v_{slot}^T @ p_mfma[sub-tile]. 3-arg init only on the
    // FIRST sub-tile of a first-iter call (kRowBase==0); sub-tile B always accumulates
    // onto A's oaccu (same oaccu tile, second N-half of P).
    auto do_mma = [&]<uint32_t kRowBase, uint32_t slot, typename OA>(OA& oaccu_dst) {
        auto run = [&](auto& v) {
            if constexpr(kIsFirstIter && kRowBase == 0u)
                hk::mma_ABt(oaccu_dst, v, p_mfma_a);
            else if constexpr(kRowBase == 0u)
                hk::mma_ABt(oaccu_dst, v, p_mfma_a, oaccu_dst);
            else
                hk::mma_ABt(oaccu_dst, v, p_mfma_b, oaccu_dst);
        };
        if constexpr(slot == 0u)
            run(pv_v_0);
        else if constexpr(slot == 1u)
            run(pv_v_1);
        else if constexpr(slot == 2u)
            run(pv_v_2);
        else
            run(pv_v_3);
    };

    if constexpr(kDoRescale)
    {
        opus::static_for<2>([&](auto s) {
            pk_mul_pair(rescale, opus::number<k_o_begin + s.value * 4u + 0u>{});
            pk_mul_pair(rescale, opus::number<k_o_begin + s.value * 4u + 2u>{});
        });
    }

    // Prologue: preload the first 3 flat V tiles (S_0,S_1,S_2 of sub-tile A) into
    // slots 0,1,2. ONE prologue for the whole call -- sub-tile B's S_0..S_2 are
    // prefetched by sub-tile A's tail iters via the flat prefetch index below.
    load_S.template operator()<0u, 0u, 0u>();
    load_S.template operator()<0u, 1u, 1u>();
    load_S.template operator()<0u, 2u, 2u>();

    opus::static_for<num_pv_iter>([&](auto i) {
        constexpr uint32_t iter     = i.value;
        constexpr uint32_t col_idx  = iter % kDIters;         // 0..15 D-tile within sub-tile
        constexpr uint32_t row_base = (iter / kDIters) * 32u; // 0 (A) or 32 (B)
        constexpr bool has_next     = (iter + 1u) < num_pv_iter;
        // Rescale only folds on the FIRST sub-tile pass (row_base==0) and only when a
        // NEXT D-tile in that pass exists (col_idx+1 < kDIters) -- else oaccu overflows.
        constexpr bool resc_next = kDoRescale && (row_base == 0u) && (col_idx + 1u < kDIters);
        constexpr uint32_t next_oaccu_base = k_o_begin + (col_idx + 1u) * 8u;

        // Flat V-tile index across all sub-tiles: prefetch runs on this so sub-tile A's
        // tail prefetches sub-tile B's S_0.. (ONE prologue). Consume uses col_idx.
        constexpr uint32_t jf_lo = 2u * iter;      // flat mfma idx (a)
        constexpr uint32_t jf_hi = 2u * iter + 1u; // flat mfma idx (b)
        // 4-slot round-robin (flat tile T -> slot T%4). Reading slot j%4 while refilling
        // slot (j+3)%4 keeps the refill ds_read's dst != the mfma's src reg.
        constexpr uint32_t slot_lo = jf_lo % 4u;
        constexpr uint32_t slot_hi = jf_hi % 4u;

        constexpr uint32_t oaccu_base = k_o_begin + col_idx * 8u;
        using oaccu_a_r =
            hkdart::split_many_t<hkdart::type_list<hkdart::range<oaccu_base + 0, oaccu_base + 3>>,
                                 4>;
        using oaccu_b_r =
            hkdart::split_many_t<hkdart::type_list<hkdart::range<oaccu_base + 4, oaccu_base + 7>>,
                                 4>;
        hk::art<comp_t, T::kTileM, T::kTileM, hk::col_l, hk::rt_16x16_s, oaccu_a_r> oaccu_a;
        hk::art<comp_t, T::kTileM, T::kTileM, hk::col_l, hk::rt_16x16_s, oaccu_b_r> oaccu_b;

        // Prefetch S at flat index jf+3 -> its own sub-tile row_base + local jj%32.
        auto prefetch = [&]<uint32_t jf>() {
            if constexpr(jf < kNumVTiles)
            {
                constexpr uint32_t pf_row = (jf / kVTilesPerSub) * 32u;
                constexpr uint32_t pf_jj  = jf % kVTilesPerSub;
                load_S.template operator()<pf_row, pf_jj, jf % 4u>();
            }
        };

        // ---- mfma_a (flat jf_lo, reads slot_lo) ----
        __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(has_next ? 4 : 2, -1));
        prefetch.template operator()<jf_lo + 3u>();
        do_mma.template operator()<row_base, slot_lo>(oaccu_a);
        if constexpr(resc_next)
        {
            mul_pair(rescale, opus::number<next_oaccu_base + 0 * 4 + 0>{});
            mul_pair(rescale, opus::number<next_oaccu_base + 1 * 4 + 0>{});
        }

        // ---- mfma_b (flat jf_hi, reads slot_hi) ----
        __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(has_next ? 4 : 0, -1));
        prefetch.template operator()<jf_hi + 3u>();
        do_mma.template operator()<row_base, slot_hi>(oaccu_b);
        if constexpr(resc_next)
        {
            mul_pair(rescale, opus::number<next_oaccu_base + 0 * 4 + 2>{});
            mul_pair(rescale, opus::number<next_oaccu_base + 1 * 4 + 2>{});
        }
    });
}

// All waves need the same H16 Q. Only one cooperative 16KiB global/LDS copy
// occurs; the four wave-private register copies are populated after the barrier.
__device__ __forceinline__ void load_q(const Params& p, uint16_t* tile)
{
    const uint32_t lane = threadIdx.x & 63u;
    const uint32_t wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    const bool aligned  = ((reinterpret_cast<uintptr_t>(p.q) | uint64_t(p.q_stride0) * 2u |
                           uint64_t(p.q_stride1) * 2u) &
                          15u) == 0;
    if(aligned)
    {
        const uint32_t row = lane >> 2, quad = lane & 3u;
        const uint64_t offset =
            uint64_t(blockIdx.x) * uint64_t(p.q_stride0) + uint64_t(row) * uint64_t(p.q_stride1);
        const uint32_t base = __builtin_amdgcn_readfirstlane(
            static_cast<uint32_t>(reinterpret_cast<uintptr_t>(tile)));
        for(uint32_t c = wave; c < 16u; c += 4u)
        {
            const uint32_t L           = c * 32u + ((quad * 8u) ^ (((row >> 2) & 1u) * 16u));
            const uint32_t col         = sb8_inv_perm_col_elems(L);
            const uint64_t address     = reinterpret_cast<uintptr_t>(p.q + offset + col);
            const uint32_t destination = base + c * 1024u;
            asm volatile("s_mov_b32 m0, %0\n\ts_nop 0\n\tglobal_load_lds_dwordx4 %1, off"
                         :
                         : "s"(destination), "v"(address)
                         : "m0", "memory");
        }
        drain();
        __syncthreads();
        const uint32_t r = lane & 15u, col = (lane >> 4) * 8u;
        const uint32_t address = base + r * 64u + ((col * 2u) ^ (((r >> 2) & 1u) * 32u));
        opus::static_for<16>(
            [&](auto c) { hkm::ds_read_b128<64u + c.value * 4u>(address, c.value * 1024u); });
    }
    else
    {
        const uint32_t head = lane & 15u;
        const uint64_t offset =
            uint64_t(blockIdx.x) * uint64_t(p.q_stride0) + uint64_t(head) * uint64_t(p.q_stride1);
        opus::static_for<16>([&](auto c) {
            const uint32_t col = sb8_inv_perm_col_elems(c.value * 32u + (lane >> 4) * 8u);
            opus::static_for<4>([&](auto j) {
                const uint32_t pair = uint32_t(p.q[offset + col + j.value * 2u]) |
                                      (uint32_t(p.q[offset + col + j.value * 2u + 1u]) << 16);
                write_u32<64u + c.value * 4u + j.value>(pair);
            });
            __builtin_amdgcn_sched_barrier(0);
        });
    }
    drain();
    __syncthreads(); // every replicated Q register read retires before KV overwrite
}

// Two PR 32-row subtiles, each preserving the exact sb8 + row-bank swizzle.
// This initial N64 geometry intentionally uses a single synchronous KV buffer.
__device__ __forceinline__ void
stage_kv(const Params& p, uint16_t* tile, int32_t* selected, int64_t start, int64_t end)
{
    const uint32_t tid = threadIdx.x, lane = tid & 63u;
    if(tid < 64u)
    {
        int32_t slot = -1;
        if(start + tid < end)
            slot = p.indices[start + tid];
        selected[tid] = slot;
    }
    __syncthreads();
    const bool aligned =
        ((reinterpret_cast<uintptr_t>(p.kv) | uint64_t(p.kv_stride0) * 2u) & 15u) == 0;
    if(aligned)
    {
        const uint32_t wave = __builtin_amdgcn_readfirstlane(tid >> 6);
        const uint32_t base = __builtin_amdgcn_readfirstlane(
            static_cast<uint32_t>(reinterpret_cast<uintptr_t>(tile)));
        for(uint32_t block = wave; block < 64u; block += 4u)
        {
            const uint32_t local = block & 31u, r = lane >> 2, quad = lane & 3u;
            const uint32_t row = (block >> 5) * 32u + (local & 1u) * 16u + r;
            const uint32_t L   = (local >> 1) * 32u + ((quad * 8u) ^ (((r >> 2) & 1u) * 16u));
            const uint32_t col = sb8_inv_perm_col_elems(L);
            const int32_t slot = selected[row];
            const uint32_t destination = base + block * 1024u;
            if(slot >= 0 && int64_t(slot) < p.slots)
            {
                const uint64_t offset  = uint64_t(slot) * uint64_t(p.kv_stride0) + uint64_t(col);
                const uint64_t address = reinterpret_cast<uintptr_t>(p.kv + offset);
                asm volatile("s_mov_b32 m0, %0\n\ts_nop 0\n\tglobal_load_lds_dwordx4 %1, off"
                             :
                             : "s"(destination), "v"(address)
                             : "m0", "memory");
            }
            else
            {
                const v4ui zero                      = {0, 0, 0, 0};
                *reinterpret_cast<v4ui*>(reinterpret_cast<uintptr_t>(tile) + block * 1024u +
                                         lane * 16u) = zero;
            }
        }
    }
    else
    {
        for(uint32_t linear = tid; linear < 64u * 512u; linear += 256u)
        {
            const uint32_t row = linear / 512u, col = linear % 512u;
            const int32_t slot = selected[row];
            uint16_t value     = 0;
            if(slot >= 0 && int64_t(slot) < p.slots)
                value = p.kv[uint64_t(slot) * uint64_t(p.kv_stride0) + uint64_t(col)];
            const uint32_t L = sb8_perm_col_elems(col), r = row & 15u;
            const uint32_t byte = (row / 32u) * 32768u +
                                  (L / 32u * 2u + (row % 32u) / 16u) * 1024u + r * 64u +
                                  ((2u * (L & 31u)) ^ (((r >> 2) & 1u) * 32u));
            tile[byte / 2u] = value;
        }
    }
    drain();
    __syncthreads();
}

__device__ __forceinline__ void qk(uintptr_t tile)
{
    // Wave w computes its 16 selected tokens against all 16 query heads.
    const uint32_t wave       = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    const uintptr_t wave_tile = tile + (wave / 2u) * 32768u + (wave % 2u) * 1024u;
    Regs::score_t score;
    auto issue = [&]<uint32_t S>() {
        constexpr uint32_t base = 40u + (S % 3u) * 4u;
        hk::art<hk::bf16, 16, 32, hk::row_l, hk::rt_16x32_s, ranges<base, base + 3u>> k;
        Bf16KvManager<Traits>::template load_k_to_gpr<0u, S * 32u>(k, wave_tile);
    };
    auto mma = [&]<uint32_t S>() {
        constexpr uint32_t base = 40u + (S % 3u) * 4u, qb = 64u + S * 4u;
        hk::art<hk::bf16, 16, 32, hk::row_l, hk::rt_16x32_s, ranges<base, base + 3u>> k;
        hk::art<hk::bf16, 16, 32, hk::row_l, hk::rt_16x32_s, ranges<qb, qb + 3u>> q;
        if constexpr(S == 0)
            hk::mma_ABt(score, k, q);
        else
            hk::mma_ABt(score, k, q, score);
    };
    opus::static_for<3>([&](auto s) { issue.template operator()<s.value>(); });
    opus::static_for<16>([&](auto s) {
        constexpr uint32_t remain = 15u - s.value;
        __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(remain < 2u ? remain : 2u, -1));
        mma.template operator()<s.value>();
        if constexpr(s.value + 3u < 16u)
            issue.template operator()<s.value + 3u>();
    });
    asm volatile("s_nop 7\n\ts_nop 7"); // opaque MFMA score dependency is not compiler-visible
    __builtin_amdgcn_sched_barrier(0);
}

__device__ __forceinline__ void softmax(const Params& p,
                                        const int32_t* selected,
                                        float* exchange,
                                        uint16_t* p_lds,
                                        float& maximum,
                                        float& denominator)
{
    const uint32_t lane = threadIdx.x & 63u, wave = threadIdx.x >> 6;
    const uint32_t head = lane & 15u, group = lane >> 4;
    const uint32_t first_key = wave * 16u + group * 4u;
    const uint32_t index_address =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(selected)) + first_key * sizeof(int32_t);
    hkm::ds_read_b128<40>(index_address, 0);
    __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(0, -1));
    float local_max = -INFINITY;
    opus::static_for<4>([&](auto i) {
        const int32_t slot = read_i32<40u + i.value>();
        float x            = (read_f32<56u + i.value>() * p.scale) * 1.4426950408889634074f;
        if(slot < 0 || int64_t(slot) >= p.slots)
            x = -INFINITY;
        write_f32<56u + i.value>(x);
        local_max = fmaxf(local_max, x);
    });
    local_max = fmaxf(local_max, __shfl_xor(local_max, 16, 64));
    local_max = fmaxf(local_max, __shfl_xor(local_max, 32, 64));
    if(lane < 16u)
        exchange[wave * 16u + head] = local_max;
    __syncthreads();
    float all_max = -INFINITY;
    opus::static_for<4>([&](auto w) { all_max = fmaxf(all_max, exchange[w.value * 16u + head]); });
    const float new_max = fmaxf(maximum, all_max);
    const float alpha   = denominator > 0.0f ? __builtin_amdgcn_exp2f(maximum - new_max) : 0.0f;
    maximum             = new_max;
    Regs::oaccu_t o;
    hk::mul_vgpr(o, o, alpha);
    float local_sum = 0.0f;
    opus::static_for<4>([&](auto i) {
        const float x      = read_f32<56u + i.value>();
        const float weight = x == -INFINITY ? 0.0f : __builtin_amdgcn_exp2f(x - maximum);
        write_f32<56u + i.value>(weight);
        local_sum += weight;
    });
    local_sum += __shfl_xor(local_sum, 16, 64);
    local_sum += __shfl_xor(local_sum, 32, 64);
    if(lane < 16u)
        exchange[64u + wave * 16u + head] = local_sum;
    // Each wave owns a disjoint N16 slab of BF16 P. Normal [head,key] LDS layout
    // makes the subsequent full-N64 operand copies explicit and bit-preserving.
    v2ui packed;
    const float p0 = read_f32<56>(), p1 = read_f32<57>();
    const float p2 = read_f32<58>(), p3 = read_f32<59>();
    asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(packed[0]) : "v"(p0), "v"(p1));
    asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(packed[1]) : "v"(p2), "v"(p3));
    const uintptr_t address = reinterpret_cast<uintptr_t>(p_lds) + (head * 64u + first_key) * 2u;
    hkm::ds_write_b64(packed, static_cast<uint32_t>(address), 0);
    __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(0, -1));
    __syncthreads(); // all P and four partial denominators are visible
    float all_sum = 0.0f;
    opus::static_for<4>([&](auto w) { all_sum += exchange[64u + w.value * 16u + head]; });
    denominator = denominator * alpha + all_sum;
    const uint32_t p_address =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p_lds)) + (head * 64u + group * 4u) * 2u;
    // PR's transposed-V operand orders the two N16 halves as group*4 and +16.
    hkm::ds_read_b64<56>(p_address, 0);
    hkm::ds_read_b64<58>(p_address, 32);
    hkm::ds_read_b64<60>(p_address, 64);
    hkm::ds_read_b64<62>(p_address, 96);
    drain();
    __syncthreads(); // denominator/max consumers retire before next iteration reuses exchange
}

template <bool Split>
__device__ __forceinline__ void run(Params p)
{
    hkdart::clobber<Regs::all_ranges>();
    extern __shared__ __align__(16) unsigned char storage[];
    auto* tile           = reinterpret_cast<uint16_t*>(storage);
    auto* p_lds          = reinterpret_cast<uint16_t*>(storage + Traits::kPOffset);
    auto* exchange       = reinterpret_cast<float*>(storage + Traits::kReduceOffset);
    auto* selected       = reinterpret_cast<int32_t*>(storage + Traits::kIndexOffset);
    const uint32_t query = blockIdx.x, partition = blockIdx.y;
    const int32_t first = p.indptr[query], last = p.indptr[query + 1u], length = last - first;
    const int32_t begin = first + int64_t(length) * partition / p.splits;
    const int32_t end   = first + int64_t(length) * (partition + 1u) / p.splits;
    load_q(p, tile);
    Regs::oaccu_t o;
    hk::zero(o);
    float maximum = -INFINITY, denominator = 0.0f;
    Bf16KvManager<Traits> manager;
    const uint32_t wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    for(int64_t pos = begin; pos < end; pos += 64)
    {
        stage_kv(p, tile, selected, pos, end);
        qk(reinterpret_cast<uintptr_t>(tile));
        softmax(p, selected, exchange, p_lds, maximum, denominator);
        // D128/wave is a multiple of the sb8 permutation's 64-element period.
        const uintptr_t v_tile = reinterpret_cast<uintptr_t>(tile) + wave * 8192u;
        sparse_mla_bf16::wave4::pv_gemm<false, false, Traits>(manager, v_tile, 1.0f);
        asm volatile("s_nop 7\n\ts_nop 7");
        drain();
        __syncthreads();
    }
    const float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    hk::mul_vgpr(o, o, inverse);
    const float lse            = denominator > 0.0f
                                     ? (maximum + __builtin_amdgcn_logf(denominator)) * 0.6931471805599453094f
                                     : -INFINITY;
    const uint32_t lane        = threadIdx.x & 63u;
    const uint64_t output_slot = Split ? uint64_t(query) * uint64_t(p.splits) + partition : query;
    if(wave == 0u && lane < 16u)
    {
        if constexpr(Split)
            p.partial_lse[output_slot * 16u + lane] = lse;
        else if(p.lse)
            p.lse[output_slot * 16u + lane] = lse;
    }
    // PR O manager preserves global head stride512. Offset its descriptor to
    // this wave's D128, set head-wave id to zero, and give each wave a private
    // dead-KV LDS bounce so its data cannot race another output wave.
    const uintptr_t bounce = reinterpret_cast<uintptr_t>(tile) + wave * 4352u;
    if constexpr(Split)
    {
        OManager32bitsV4Gen1Swizzle<Traits, float> output;
        float* dst = p.partial_o + output_slot * 16u * 512u + wave * 128u;
        opus::static_for<2>([&](auto i) {
            output.template output_to_vram_pair<128u + i.value * 16u, i.value * 64u, false>(
                dst, 0, 0, 0, bounce, 16);
            drain();
        });
    }
    else
    {
        OManager16bitsV4Gen1Swizzle<Traits, hk::bf16> output;
        auto* dst = reinterpret_cast<hk::bf16*>(p.out + output_slot * 16u * 512u + wave * 128u);
        opus::static_for<2>([&](auto i) {
            output.template output_to_vram_pair<128u + i.value * 16u, i.value * 64u, false>(
                dst, 0, 0, 0, bounce, 16);
            drain();
        });
    }
}
} // namespace sparse_mla_bf16::wave4

namespace sparse_mla_bf16 {
template <int Heads, bool Split>
__global__ __launch_bounds__(256) __attribute__((amdgpu_num_vgpr(40))) void sparse_mla_v2(Params p)
{
    if constexpr(Heads == 16)
        wave4::run<Split>(p);
    else
        sparse_mla_legacy<Heads, Split>(p);
}
} // namespace sparse_mla_bf16
