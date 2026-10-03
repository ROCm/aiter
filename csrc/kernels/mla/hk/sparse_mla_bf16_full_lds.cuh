// SPDX-License-Identifier: MIT
// Full-LDS H64 implementation: Q, two KV tiles and P stay in 139,520 B LDS.
// Included inside namespace sparse_mla_bf16.
struct FullLdsTraits : Traits<64>
{
    static constexpr uint32_t kNumWarps = 8, kNumThreads = 512;
    static constexpr uint32_t kIndexOffset = 139264, kLdsBytes = 139520;
};

struct FullLdsRegs
{
    using comp_t = float;
    template <uint32_t B, uint32_t N>
    using ranges = hkdart::split_many_t<hkdart::type_list<hkdart::range<B, B + N - 1u>>, 4>;
    template <uint32_t B>
    using bf16_fragment = hk::art<hk::bf16, 16, 32, hk::row_l, hk::rt_16x32_s, ranges<B, 4>>;
    // All compiler-managed instructions must remain below v40. QK operands
    // occupy40:55; scores60:63. Softmax reuses40:47 for eight scores and48:55
    // for masks. PV uses40:55 for four V slots and60:63 for packed full-N32 P.
    static constexpr uint32_t k_o_begin = 64, kNumKvSub = 1;
    static constexpr uint32_t k_v0_begin = 40, k_k0_begin = 44;
    static constexpr uint32_t k_k1_begin = 48, k_k2_begin = 52;
    using pinned_ranges = ranges<40, 88>;
    using oaccu_t       = hk::art<float, 16, 256, hk::row_l, hk::rt_16x16_s, ranges<64, 64>>;
    using score_t       = hk::art<float, 16, 16, hk::col_l, hk::rt_16x16_s, ranges<60, 4>>;
    using p_mfma_a_t    = bf16_fragment<60>;
    using p_mfma_b_t    = bf16_fragment<60>; // unused by the N32 PV schedule
    using pv_v_0_t      = bf16_fragment<40>;
    using pv_v_1_t      = bf16_fragment<44>;
    using pv_v_2_t      = bf16_fragment<48>;
    using pv_v_3_t      = bf16_fragment<52>;
};

#include "sparse_mla_bf16_full_lds_pv.cuh"

// LDS communications use explicit DS operations and a barrier that does not
// implicitly drain VMEM. Next-KV DMA must be allowed to overlap these phases.
__device__ __forceinline__ void full_lds_barrier()
{
    __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(0, -1));
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_barrier" ::: "memory");
}

// Same packet mapping/validity/zeroing as stage_kv_async; H64 row reuse only.
__device__ __forceinline__ void full_lds_stage_kv_async(
    const Params& p, uint16_t* tile, int32_t* selected, int64_t start, int64_t end)
{
    using T = FullLdsTraits;
    static_assert(T::kNumWarps == 8u);
    const uint32_t tid = threadIdx.x;
    if(tid < 32u)
    {
        int32_t slot = -1;
        if(start + tid < end)
            slot = p.indices[start + tid];
        selected[tid] = slot;
    }
    __syncthreads(); // index publication precedes every DMA; none is issued yet
    const uint32_t lane = tid & 63u;
    const uint32_t wave = __builtin_amdgcn_readfirstlane(tid >> 6);
    const uint32_t base =
        __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(tile)));
    // Four D32 packets share one selected row in this eight-wave layout.
    const uint32_t r = lane >> 2, quad = lane & 3u;
    const uint32_t row = (wave & 1u) * 16u + r;
    const int32_t slot = selected[row];
    const bool valid   = slot >= 0 && int64_t(slot) < p.slots;
    // Each lane's predicate is invariant across all four packets. Restore
    // EXEC for every lane before the common LDS drain below; no CTA barrier
    // is entered from either divergent branch.
    if(valid)
    {
        const uint64_t offset   = uint64_t(slot) * uint64_t(p.kv_stride0);
        const uint64_t row_base = reinterpret_cast<uintptr_t>(p.kv + offset);
        opus::static_for<4>([&](auto packet_) {
            const uint32_t block = wave + packet_.value * T::kNumWarps;
            const uint32_t L     = (block >> 1) * 32u + ((quad * 8u) ^ (((r >> 2) & 1u) * 16u));
            const uint32_t col   = sb8_inv_perm_col_elems(L);
            const uint32_t destination = base + block * 1024u;
            const uint64_t address     = row_base + uint64_t(col) * sizeof(uint16_t);
            asm volatile("s_mov_b32 m0, %0\n\ts_nop 0\n\tglobal_load_lds_dwordx4 %1, off"
                         :
                         : "s"(destination), "v"(address)
                         : "m0", "memory");
        });
    }
    else
    {
        opus::static_for<4>([&](auto packet_) {
            const uint32_t block = wave + packet_.value * T::kNumWarps;
            // Invalid lanes own the same private 16 B landing as valid lanes.
            // Zero through the dead QK operands to avoid hoisted compiler
            // constants crossing the pinned register boundary.
            asm volatile("v_mov_b32 v40, 0\n\tv_mov_b32 v41, 0\n\t"
                         "v_mov_b32 v42, 0\n\tv_mov_b32 v43, 0" ::
                             : "memory");
            hkm::ds_write_b128<40>(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(tile)) +
                                       block * 1024u + lane * 16u,
                                   0);
        });
    }
    __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(0, -1));
    __builtin_amdgcn_sched_barrier(0);
}

__device__ __forceinline__ void full_lds_load_q(const Params& p, uint16_t* q_lds)
{
    const uint32_t tid = threadIdx.x, lane = tid & 63u;
    const uint32_t wave = __builtin_amdgcn_readfirstlane(tid >> 6);
    const uint32_t base =
        __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(q_lds)));
    const bool aligned = ((reinterpret_cast<uintptr_t>(p.q) | (uint64_t(p.q_stride0) * 2u) |
                           (uint64_t(p.q_stride1) * 2u)) &
                          15u) == 0;
    if(aligned)
    {
        // 4 head groups * 16 D32 blocks. Eight waves jointly load Q exactly
        // once; paired compute waves later read the same16KiB head-group area.
        for(uint32_t block = wave; block < 64u; block += 8u)
        {
            const uint32_t group = block / 16u, c = block % 16u;
            const uint32_t row = lane >> 2, quad = lane & 3u;
            const uint32_t col =
                sb8_inv_perm_col_elems(c * 32u + ((quad * 8u) ^ (((row >> 2) & 1u) * 16u)));
            const uint64_t offset = uint64_t(blockIdx.x) * uint64_t(p.q_stride0) +
                                    uint64_t(group * 16u + row) * uint64_t(p.q_stride1) + col;
            const uint64_t address     = reinterpret_cast<uintptr_t>(p.q + offset);
            const uint32_t destination = base + block * 1024u;
            asm volatile("s_mov_b32 m0, %0\n\ts_nop 0\n\tglobal_load_lds_dwordx4 %1, off"
                         :
                         : "s"(destination), "v"(address)
                         : "m0", "memory");
        }
    }
    else
    {
        // Minimum contract alignment is2B, including odd BF16 row/head strides.
        for(uint32_t linear = tid; linear < 64u * 512u; linear += 512u)
        {
            uint32_t item = linear;
            asm volatile("" : "+v"(item));
            const uint32_t head = item / 512u, col = item % 512u;
            const uint64_t offset = uint64_t(blockIdx.x) * uint64_t(p.q_stride0) +
                                    uint64_t(head) * uint64_t(p.q_stride1) + col;
            const uint16_t value = p.q[offset];
            const uint32_t L = sb8_perm_col_elems(col), row = head & 15u;
            const uint32_t byte = (head / 16u) * 16384u + (L / 32u) * 1024u + row * 64u +
                                  ((2u * (L & 31u)) ^ (((row >> 2) & 1u) * 32u));
            q_lds[byte / 2u] = value;
        }
    }
    drain();
    __syncthreads();
}

template <uint32_t KeyHalf>
__device__ __forceinline__ void full_lds_qk(uintptr_t q_base, uintptr_t kv_base)
{
    using T             = FullLdsTraits;
    const uint32_t lane = threadIdx.x & 63u, row = lane & 15u;
    const uint32_t col       = (lane >> 4) * 8u;
    const uint32_t q_address = static_cast<uint32_t>(q_base) + (threadIdx.x >> 7) * 16384u +
                               row * 64u + ((col * 2u) ^ (((row >> 2) & 1u) * 32u));
    FullLdsRegs::score_t score;
    auto issue = [&]<uint32_t C>() {
        constexpr uint32_t qr = 40u + (C % 2u) * 4u, kr = 48u + (C % 2u) * 4u;
        typename FullLdsRegs::template bf16_fragment<kr> k;
        hkm::ds_read_b128<qr>(q_address, C * 1024u);
        Bf16KvManager<T>::template load_k_to_gpr<KeyHalf * 16u, C * 32u>(k, kv_base);
    };
    auto mma = [&]<uint32_t C>() {
        typename FullLdsRegs::template bf16_fragment<40u + (C % 2u) * 4u> q;
        typename FullLdsRegs::template bf16_fragment<48u + (C % 2u) * 4u> k;
        if constexpr(C == 0u)
            hk::mma_ABt(score, k, q);
        else
            hk::mma_ABt(score, k, q, score);
    };
    issue.template operator()<0u>();
    issue.template operator()<1u>();
    opus::static_for<16>([&](auto c_) {
        constexpr uint32_t c = c_.value;
        // Each future fragment consists of exactly Q+K LDS reads. Wait until
        // the oldest pair is ready, leaving the next pair in flight.
        __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(c < 15u ? 2 : 0, -1));
        mma.template operator()<c>();
        if constexpr(c + 2u < 16u)
            issue.template operator()<c + 2u>();
    });
    // Memory waits do not scoreboard the opaque numeric MFMA accumulator.
    asm volatile("s_nop 7\n\ts_nop 7");
    __builtin_amdgcn_sched_barrier(0);
}

__device__ __forceinline__ void full_lds_softmax(
    const Params& p, const int32_t* selected, uintptr_t scores, float& maximum, float& denominator)
{
    const uint32_t lane = threadIdx.x & 63u, group = lane >> 4;
    const uint32_t head = (threadIdx.x >> 7) * 16u + (lane & 15u);
    const uint32_t score_address =
        static_cast<uint32_t>(scores) + (head / 16u) * 2048u + (group * 16u + (head & 15u)) * 16u;
    const uint32_t mask_address =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(selected)) + group * 16u;
    // Both paired waves read the same N32 scores and perform the same softmax.
    // This deliberately duplicates its arithmetic in the first candidate but
    // needs no extra state slab, no BF16 score approximation and no broadcast.
    hkm::ds_read_b128<40>(score_address, 0);
    hkm::ds_read_b128<44>(score_address, 1024);
    hkm::ds_read_b128<48>(mask_address, 0);
    hkm::ds_read_b128<52>(mask_address, 64);
    __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(0, -1));
    __builtin_amdgcn_sched_barrier(0);
    constexpr float log2e = 1.4426950408889634074f;
    float local_max       = -INFINITY;
    opus::static_for<8>([&](auto i_) {
        constexpr uint32_t i = i_.value;
        const int32_t slot   = read_i32<48u + i>();
        float score          = (read_f32<40u + i>() * p.scale) * log2e;
        if(slot < 0 || int64_t(slot) >= p.slots)
            score = -INFINITY;
        write_f32<40u + i>(score);
        local_max = fmaxf(local_max, score);
    });
    local_max            = fmaxf(local_max, __shfl_xor(local_max, 16, 64));
    local_max            = fmaxf(local_max, __shfl_xor(local_max, 32, 64));
    const float next_max = fmaxf(maximum, local_max);
    const float alpha    = denominator > 0.0f ? __builtin_amdgcn_exp2f(maximum - next_max) : 0.0f;
    maximum              = next_max;
    FullLdsRegs::oaccu_t o;
    hk::mul_vgpr(o, o, alpha);
    float sum = 0.0f;
    opus::static_for<8>([&](auto i_) {
        constexpr uint32_t i = i_.value;
        const float score    = read_f32<40u + i>();
        const float weight   = score == -INFINITY ? 0.0f : __builtin_amdgcn_exp2f(score - maximum);
        write_f32<40u + i>(weight);
        sum += weight;
    });
    sum += __shfl_xor(sum, 16, 64);
    sum += __shfl_xor(sum, 32, 64);
    denominator = denominator * alpha + sum;
    pack_2f32_to_bf16_pair_pinned<60, 40>();
    pack_2f32_to_bf16_pair_pinned<61, 42>();
    pack_2f32_to_bf16_pair_pinned<62, 44>();
    pack_2f32_to_bf16_pair_pinned<63, 46>();
}

template <bool Split, uint32_t Half>
__device__ __forceinline__ void full_lds_output(const Params& p, uint64_t slot, uintptr_t bounce)
{
    using T                   = FullLdsTraits; // physical output pitch remains D512
    const uint32_t head_group = threadIdx.x >> 7;
    // Preserve independent LDS bounce regions for both waves of each pair.
    // Float output needs8*4352=34816B, which fits dead Q64KiB but not one KV32KiB.
    const uintptr_t half_bounce = bounce + Half * 4u * 4352u;
    if constexpr(Split)
    {
        OManager32bitsV4Gen1Swizzle<T, float> output;
        float* dst = p.partial_o + slot * 64u * 512u;
        opus::static_for<4>([&](auto i) {
            output.template output_to_vram_pair<64u + i.value * 16u,
                                                Half * 256u + i.value * 64u,
                                                false>(dst, head_group, 0, 0, half_bounce, 64);
            drain();
        });
    }
    else
    {
        OManager16bitsV4Gen1Swizzle<T, hk::bf16> output;
        auto* dst = reinterpret_cast<hk::bf16*>(p.out + slot * 64u * 512u);
        opus::static_for<4>([&](auto i) {
            output.template output_to_vram_pair<64u + i.value * 16u,
                                                Half * 256u + i.value * 64u,
                                                false>(dst, head_group, 0, 0, half_bounce, 64);
            drain();
        });
    }
}

template <bool Split>
__device__ __forceinline__ void full_lds_h64(Params p)
{
    using T = FullLdsTraits;
    static_assert(T::kNumWarps == 8u && T::kLdsBytes == 139520u);
    hkdart::clobber<FullLdsRegs::pinned_ranges>();
    extern __shared__ __align__(16) unsigned char storage[];
    auto* q_lds            = reinterpret_cast<uint16_t*>(storage);
    auto* tile             = reinterpret_cast<uint16_t*>(storage + 65536u);
    auto* next_tile        = reinterpret_cast<uint16_t*>(storage + 98304u);
    const uintptr_t scores = reinterpret_cast<uintptr_t>(storage + 131072u);
    auto* selected         = reinterpret_cast<int32_t*>(storage + 139264u);
    auto* next_selected    = selected + 32u;
    const uint32_t query = blockIdx.x, partition = blockIdx.y;
    const int32_t row_start = p.indptr[query], row_end = p.indptr[query + 1u];
    const int32_t length = row_end - row_start;
    const int64_t begin  = int64_t(row_start) + int64_t(length) * partition / p.splits;
    const int64_t end    = int64_t(row_start) + int64_t(length) * (partition + 1u) / p.splits;
    full_lds_load_q(p, q_lds);
    FullLdsRegs::oaccu_t o;
    hk::zero(o);
    float maximum = -INFINITY, denominator = 0.0f;
    const uint32_t lane = threadIdx.x & 63u, wave = threadIdx.x >> 6;
    const bool aligned =
        ((reinterpret_cast<uintptr_t>(p.kv) | (uint64_t(p.kv_stride0) * 2u)) & 15u) == 0;
    Bf16KvManager<T> manager;
    if(aligned && begin < end)
    {
        full_lds_stage_kv_async(p, tile, selected, begin, end);
        drain();
        __syncthreads();
    }
    for(int64_t pos = begin; pos < end; pos += 32)
    {
        if(aligned)
        {
            if(pos + 32 < end)
                full_lds_stage_kv_async(p, next_tile, next_selected, pos + 32, end);
        }
        else
        {
            // v44:59 staging payloads are dead Q/K slots in this map; live
            // O64:127 remains untouched. Retains minimum2B input alignment.
            stage_kv<T>(p, tile, selected, pos, end);
        }
        const uintptr_t kv_address = reinterpret_cast<uintptr_t>(tile);
        if(wave & 1u)
            full_lds_qk<1u>(reinterpret_cast<uintptr_t>(q_lds), kv_address);
        else
            full_lds_qk<0u>(reinterpret_cast<uintptr_t>(q_lds), kv_address);
        const uint32_t head = (wave / 2u) * 16u + (lane & 15u);
        const uint32_t key  = (wave & 1u) * 16u + (lane >> 4) * 4u;
        hkm::ds_write_b128<60>(static_cast<uint32_t>(scores) + (head / 16u) * 2048u +
                                   ((key / 4u) * 16u + (head & 15u)) * 16u,
                               0);
        full_lds_barrier(); // both N16 halves published; next KV stays pending
        full_lds_softmax(p, selected, scores, maximum, denominator);
        if(wave & 1u)
            full_lds_pv_gemm<256u, false, false, T>(manager, kv_address, 1.0f);
        else
            full_lds_pv_gemm<0u, false, false, T>(manager, kv_address, 1.0f);
        drain();
        __syncthreads(); // all score/current reads retired and next DMA complete
        if(aligned)
        {
            auto* old_tile     = tile;
            tile               = next_tile;
            next_tile          = old_tile;
            auto* old_selected = selected;
            selected           = next_selected;
            next_selected      = old_selected;
        }
    }
    const float inv_sum = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    hk::mul_vgpr(o, o, inv_sum);
    constexpr float ln2 = 0.6931471805599453094f;
    const float lse =
        denominator > 0.0f ? (maximum + __builtin_amdgcn_logf(denominator)) * ln2 : -INFINITY;
    const uint64_t slot = Split ? uint64_t(query) * uint64_t(p.splits) + partition : query;
    if((wave & 1u) == 0 && lane < 16u)
    {
        const uint64_t head = uint64_t(wave / 2u) * 16u + lane;
        if constexpr(Split)
            p.partial_lse[slot * 64u + head] = lse;
        else if(p.lse)
            p.lse[slot * 64u + head] = lse;
    }
    // No Q read survives the terminal loop barrier (or prologue barrier when
    // empty), so its64KiB region is now safe for the independent O bounce slots.
    if(wave & 1u)
        full_lds_output<Split, 1u>(p, slot, reinterpret_cast<uintptr_t>(storage));
    else
        full_lds_output<Split, 0u>(p, slot, reinterpret_cast<uintptr_t>(storage));
}
