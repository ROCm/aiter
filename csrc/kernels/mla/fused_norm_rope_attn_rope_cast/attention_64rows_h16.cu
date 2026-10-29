// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "attention_64rows.cuh"

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
#include "../opus/mla_v41_prefill_a8w8_16mx1_16nx4.hpp"
namespace fused_mla_h16 {
using namespace opus;
using namespace opus_mla_v41_prefill_a8w8_16mx1_16nx4;

template <class D>
__device__ __forceinline__ auto load16(const D* ptr)
{
    vector_t<D, 16> out;
    __builtin_memcpy(&out, __builtin_assume_aligned(ptr, 4), 16);
    return out;
}

template <class T, class L>
__device__ __forceinline__ auto load_values(const typename T::D_ATTN* ptr, const L& layout)
{
    constexpr int n    = layout_load_traits<L, 16>::r_elem.value;
    const auto offsets = layout_to_offsets<16>(layout);
    vector_t<typename T::D_ATTN, n * 16> out;
    static_for<n>([&](auto i) {
        set_slice(out,
                  load16(ptr + offsets[i.value]),
                  number<i.value * 16>{},
                  number<(i.value + 1) * 16>{});
    });
    return out;
}

template <class T, class V, class S>
__device__ __forceinline__ float row_max(const V& v, S& scratch, int warp, int lane)
{
    float m = -__builtin_inff();
    static_for<vector_traits<V>::size()>([&](auto i) { m = fmaxf(m, v[i.value]); });
    auto maximum = [](float a, float b) { return fmaxf(a, b); };
    m            = fused_norm_rope_attn_rope_cast_fwd::lane_swap_reduce<32>(m, maximum);
    m            = fused_norm_rope_attn_rope_cast_fwd::lane_swap_reduce<16>(m, maximum);
    if(lane < 16)
        store(scratch, m, lane * 4 + warp);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    const auto all = load<4>(scratch, (lane % 16) * 4);
    return fmaxf(fmaxf(all[0], all[1]), fmaxf(all[2], all[3]));
}

template <bool Cooperative, class V>
__device__ __forceinline__ int
prepare_q(const FusedNormRopeAttnRopeCastParams& a, V& q, int token, int lane, int head_start, int warp, char* smem)
{
    namespace fused = fused_norm_rope_attn_rope_cast_fwd;
    const int group = lane / 16;
    const auto* src = static_cast<const bf16_t*>(a.raw_q_ptr) +
                      (int64_t(token) * a.H + head_start + lane % 16) * 512;
    const float inv     = fused_q_norm::inv_rms<4, 16>(src, group, a.q_norm, a.norm_eps);
    const int64_t pos   = a.positions64 ? static_cast<const int64_t*>(a.positions_ptr)[token]
                                        : static_cast<const int*>(a.positions_ptr)[token];
    const auto* cs      = static_cast<const bf16_t*>(a.cos_ptr) + pos * 32;
    const auto* sn      = static_cast<const bf16_t*>(a.sin_ptr) + pos * 32;
    unsigned scale_word = 0;
    static_for<Cooperative ? 1 : 4>([&](auto ek) {
        const int chunk = Cooperative ? warp : ek.value;
        unsigned codes[2];
        static_for<2>([&](auto piece) {
            const int d = chunk * 128 + piece.value * 64 + group * 16;
            auto x      = fused_q_norm::load_bf16x16(src + d);
            if(a.q_norm)
                fused_q_norm::apply(x, inv);
            if(chunk == 3 && piece.value == 1)
                if(!a.q_is_rotated)
                    static_for<8>([&](auto i) {
                        const int ri = (d - 448) / 2 + i.value;
                        const auto rotated =
                            fused::rope_pair<false>(fused::bf32(x[i.value * 2]),
                                                    fused::bf32(x[i.value * 2 + 1]),
                                                    fused::bf32(cs[ri]),
                                                    fused::bf32(sn[ri]));
                        x[i.value * 2]     = static_cast<bf16_t>(rotated[0]);
                        x[i.value * 2 + 1] = static_cast<bf16_t>(rotated[1]);
                    });
            vector_t<float, 16> value;
            float maximum = 1e-30f;
            static_for<16>([&](auto i) {
                value[i.value] = fused::bf32(x[i.value]);
                maximum        = fmaxf(maximum, fabsf(value[i.value]));
            });
            maximum             = fmaxf(maximum,
                            std::bit_cast<float>(__builtin_amdgcn_ds_bpermute(
                                (lane ^ 16) * 4, std::bit_cast<unsigned>(maximum))));
            const unsigned bits = std::bit_cast<unsigned>(maximum / 448.0f);
            const int e         = int((bits - 1) >> 23) + 1 - 127;
            const auto quant =
                cast<opus::fp8_t>(value * std::bit_cast<float>(uint32_t(127 - e) << 23));
            if constexpr(Cooperative)
                *reinterpret_cast<vector_t<opus::fp8_t, 16>*>(
                    smem + ((warp * 2 + piece.value) * 64 + lane) * 16) = quant;
            else
                set_slice(q,
                          __builtin_bit_cast(vector_t<::fp8_t, 16>, quant),
                          number<ek.value * 32 + piece.value * 16>{},
                          number<ek.value * 32 + (piece.value + 1) * 16>{});
            codes[piece.value] = e + 127;
        });
        const unsigned pair = codes[0] | (codes[1] << 8);
        const int src_lane  = lane % 16 + (group % 2) * 32;
        const unsigned byte =
            (__builtin_amdgcn_ds_bpermute(src_lane * 4, pair) >> (group / 2 * 8)) & 255;
        if constexpr(Cooperative)
            reinterpret_cast<unsigned char*>(smem + 8192)[lane * 4 + warp] = byte;
        else
            scale_word |= byte << (ek.value * 8);
    });
    if constexpr(Cooperative)
    {
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        static_for<4>([&](auto ek) {
            static_for<2>([&](auto piece) {
                const auto value = *reinterpret_cast<const vector_t<::fp8_t, 16>*>(
                    smem + ((ek.value * 2 + piece.value) * 64 + lane) * 16);
                set_slice(q, value, number<ek.value * 32 + piece.value * 16>{},
                          number<ek.value * 32 + (piece.value + 1) * 16>{});
            });
        });
        scale_word = reinterpret_cast<const unsigned*>(smem + 8192)[lane];
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
    }
    return static_cast<int>(scale_word);
}

// Based on opus/mla_v41_prefill_a8w8_16mx1_16nx4.hpp; retain FP32 softmax sums and safe CSR loads.
template <class Traits, class VQ, class VO>
__device__ __attribute__((always_inline)) void accumulate(
    const FusedNormRopeAttnRopeCastParams& a,
    fused_norm_rope_attn_rope_cast_fwd::CsrRange prefix,
    fused_norm_rope_attn_rope_cast_fwd::CsrRange extend,
    char* smem_kv,
    char* smem_ml,
    char* smem_p,
    char* smem_sum,
    const VQ& v_q,
    int scale_q,
    VO& v_o,
    typename Traits::D_ACC& m_row,
    typename Traits::D_ACC& l_row,
    float& true_max,
    typename Traits::D_ACC temperature_scale)
{
    using namespace opus;
    using T      = opus::remove_cvref_t<Traits>;
    using D_ATTN = typename T::D_ATTN;
    using D_PV   = typename T::D_PV;
    using D_ACC  = typename T::D_ACC;

    // Round each source separately so its tail lanes stay masked.
    const int prefix_len   = prefix.end - prefix.begin;
    const int extend_len   = extend.end - extend.begin;
    const int prefix_tiles = (uint32_t(prefix_len) + T::KV_TILE_SIZE - 1) / T::KV_TILE_SIZE;
    const int extend_tiles = (uint32_t(extend_len) + T::KV_TILE_SIZE - 1) / T::KV_TILE_SIZE;
    const int num_kv_tiles = prefix_tiles + extend_tiles;
    if(num_kv_tiles <= 0)
        return;

    int lane_id = thread_id_x() % T::WARP_SIZE;
    asm volatile("" : "+v"(lane_id));
    int warp_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);

    auto s_sum = make_smem(reinterpret_cast<D_ACC*>(smem_sum));
    auto s_m   = make_smem(reinterpret_cast<D_ACC*>(smem_ml));
    auto s_p   = make_smem(reinterpret_cast<D_PV*>(smem_p));
    auto s_kv  = make_smem(reinterpret_cast<D_PV*>(smem_kv));

    auto mfma0 = make_mfma<D_ATTN, D_ATTN, D_ACC>(
        number<T::W_M>{}, number<T::W_N>{}, number<T::W_K_QK>{}, mfma_adaptor_swap_ab{});
    auto mma1 = make_tiled_mma<D_PV, D_PV, D_ACC>(seq<T::GEMM1_E_M, T::GEMM1_E_N, T::GEMM1_E_K>{},
                                                  seq<T::T_M, T::T_N, T::T_K>{},
                                                  seq<T::W_M, T::W_N, T::W_K_PV>{},
                                                  mfma_adaptor_swap_ab{});
    auto u_rk = make_layout_rk<T>(lane_id);
    auto u_sk = make_layout_sk<T>(warp_id, lane_id);
    auto u_rv = make_layout_rv<T>(warp_id, lane_id);

    typename decltype(mfma0)::vtype_c v_s;
    typename decltype(mma1)::vtype_a v_p;
    typename decltype(mma1)::vtype_b v_v;

    constexpr index_t s_len = vector_traits<typename decltype(mfma0)::vtype_c>::size();
    auto v_p_warps          = reinterpret_cast<vector_t<D_PV, s_len>*>(&v_p);

    struct KvPage
    {
        int id;
        bool extend;
    };
    const int key_in_tile = warp_id * T::W_N + lane_id % T::W_N;
    auto load_kv_page     = [&](int tile_idx) {
        const bool from_extend = extend_len > 0 && tile_idx >= prefix_tiles;
        const int length = from_extend ? extend_len : prefix_len;
        const int begin = from_extend ? extend.begin : prefix.begin;
        const auto* indices = from_extend ? a.kv_indices_extend : a.kv_indices_prefix;
        auto g_indices = make_gmem(indices + begin, length * sizeof(int));
        const int local_tile = tile_idx - (from_extend ? prefix_tiles : 0);
        const uint32_t offset = uint32_t(local_tile) * T::KV_TILE_SIZE + key_in_tile;
        const int id = load(g_indices, offset < uint32_t(length) ? offset : uint32_t(length - 1))[0];
        return KvPage{id, from_extend};
    };
    auto is_live = [&](KvPage page) {
        return uint32_t(page.id) < uint32_t(page.extend ? a.extend_rows : a.prefix_rows);
    };

    const int kblk        = lane_id / T::W_M;
    const u32_t scale_sel = kblk | ((4 + kblk) << 8) | 0x0c0c0000u;
    const u32_t scale_sh0 = 8 * (kblk / 2);
    const u32_t scale_sh1 = scale_sh0 + 16;

    auto data = [&](KvPage page) {
        const auto* ptr = reinterpret_cast<const D_ATTN*>(page.extend ? a.kv_ptr : a.unified_kv_ptr);
        const int stride = page.extend ? a.stride_kv_extend : a.stride_kv_prefix;
        return is_live(page) ? ptr + int64_t(page.id) * stride
                             : reinterpret_cast<const D_ATTN*>(
                                   fused_norm_rope_attn_rope_cast_fwd::safe_kv.data);
    };
    const bool prefix_packed =
        reinterpret_cast<uintptr_t>(a.unified_kv_scale_ptr) ==
            reinterpret_cast<uintptr_t>(a.unified_kv_ptr) + 512 &&
        a.stride_kv_scale_prefix == a.stride_kv_prefix;
    const bool extend_packed =
        reinterpret_cast<uintptr_t>(a.kv_scale_ptr) == reinterpret_cast<uintptr_t>(a.kv_ptr) + 512 &&
        a.stride_kv_scale_extend == a.stride_kv_extend;
    auto scales = [&](KvPage page) {
        const bool packed = page.extend ? extend_packed : prefix_packed;
        const auto* ptr =
            reinterpret_cast<const D_ATTN*>(page.extend ? a.kv_scale_ptr : a.unified_kv_scale_ptr);
        const int stride = page.extend ? a.stride_kv_scale_extend : a.stride_kv_scale_prefix;
        return packed ? data(page) + 512
                      : (is_live(page) ? ptr + int64_t(page.id) * stride
                                       : reinterpret_cast<const D_ATTN*>(
                                             fused_norm_rope_attn_rope_cast_fwd::safe_kv.scales));
    };
    auto kv_page = load_kv_page(0);
    int live     = is_live(kv_page);
    s_waitcnt_vmcnt(0_I);
    auto v_k   = load_values<T>(data(kv_page), u_rk);
    auto v_scl = load16(scales(kv_page));
    kv_page    = load_kv_page(1);

    for(int tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx)
    {
        const auto k_scl_w = std::bit_cast<vector_t<u32_t, T::D_SCALE_SIZE / 4>>(v_scl);
        const int scale_k  = static_cast<int>(
            __builtin_amdgcn_perm(__builtin_amdgcn_perm(k_scl_w[3], k_scl_w[2], scale_sel),
                                  __builtin_amdgcn_perm(k_scl_w[1], k_scl_w[0], scale_sel),
                                  0x05040100u));

        constexpr index_t k_step = vector_traits<typename decltype(mfma0)::vtype_a>::size();
        clear(v_s);
        static_for<T::GEMM0_E_K>([&](auto ek) {
            v_s = mfma0(slice(v_q, number<ek.value * k_step>{}, number<(ek.value + 1) * k_step>{}),
                        slice(v_k, number<ek.value * k_step>{}, number<(ek.value + 1) * k_step>{}),
                        v_s,
                        scale_q,
                        scale_k,
                        ek,
                        ek);
        });

        constexpr index_t k_len = vector_traits<decltype(v_k)>::size();
        vector_t<D_PV, k_len> v_k_bf16;
        auto& k_w       = reinterpret_cast<const vector_t<u32_t, k_len / 4>&>(v_k);
        auto* k_bf16_pk = reinterpret_cast<vector_t<D_PV, 2>*>(&v_k_bf16);
        static_for<k_len / 4>([&](auto d) {
            constexpr int ek  = d.value / 8;
            constexpr int h   = (d.value / 4) % 2;
            const u32_t e8m0  = __builtin_amdgcn_ubfe(k_scl_w[ek], h ? scale_sh1 : scale_sh0, 8);
            const float scale = std::bit_cast<float>(e8m0 << 23);
            k_bf16_pk[d.value * 2 + 0] =
                __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(k_w[d.value], scale, false);
            k_bf16_pk[d.value * 2 + 1] =
                __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(k_w[d.value], scale, true);
        });
        store_v_swizzled<T>(s_kv, v_k_bf16, u_sk, lane_id);

        const int next_live = is_live(kv_page);
        v_k                 = load_values<T>(data(kv_page), u_rk);
        v_scl               = load16(scales(kv_page));
        kv_page             = load_kv_page(tile_idx + 2);

        constexpr D_ACC RESCALE_THRESHOLD = D_ACC(8.0f);
        const bool from_extend = extend_len > 0 && tile_idx >= prefix_tiles;
        const int current_length = from_extend ? extend_len : prefix_len;
        const int current_tile = tile_idx - (from_extend ? prefix_tiles : 0);
        attn_mask_oob_score<T>(v_s, current_length, current_tile, warp_id, lane_id);
        if(__builtin_amdgcn_ballot_w64(!live))
            static_for<s_len>([&](auto i) {
                const int src = (lane_id / 16) * 4 + i.value;
                if(!__builtin_amdgcn_ds_bpermute(src * 4, live))
                    v_s[i.value] = -__builtin_inff();
            });
        const D_ACC tile_max = row_max<T>(v_s, s_m, warp_id, lane_id) * temperature_scale;
        true_max             = fmaxf(true_max, tile_max);
        D_ACC rescale_m      = D_ACC(1.0f);
        if(__builtin_expect(__builtin_amdgcn_ballot_w64(tile_max - m_row <= RESCALE_THRESHOLD) !=
                                __builtin_amdgcn_read_exec(),
                            0))
        {
            const D_ACC row_max = max(m_row, tile_max);
            rescale_m           = __builtin_amdgcn_exp2f(m_row - row_max);
            m_row               = row_max;
            scale_output_tile<T>(v_o, rescale_m);
        }
        attn_row_scale_sub<T>(v_s, temperature_scale, m_row);
        attn_exp2_slice<T, 0, s_len>(v_s);

        float tile_sum = 0.0f;
        static_for<s_len>([&](auto i) { tile_sum += v_s[i.value]; });
        auto add = [](float a, float b) { return a + b; };
        tile_sum = fused_norm_rope_attn_rope_cast_fwd::lane_swap_reduce<32>(tile_sum, add);
        tile_sum = fused_norm_rope_attn_rope_cast_fwd::lane_swap_reduce<16>(tile_sum, add);
        if(lane_id < 16)
            store(s_sum, tile_sum, lane_id * 4 + warp_id);
        auto v_p_seg = cast<D_PV>(v_s);
        store<s_len>(s_p, v_p_seg, warp_id * T::W_M * T::W_N + lane_id * s_len);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        static_for<T::NUM_WARPS>([&](auto i) {
            v_p_warps[i.value] = load<s_len>(s_p, i.value * T::W_M * T::W_N + lane_id * s_len);
        });
        const auto sums = load<4>(s_sum, (lane_id % 16) * 4);
        l_row           = l_row * rescale_m + ((sums[0] + sums[1]) + (sums[2] + sums[3]));

        v_v = tr_load<T::VEC_TR_V>(s_kv, u_rv);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_sched_barrier(0);
        v_o  = mma1(v_p, v_v, v_o);
        live = next_live;
        __builtin_amdgcn_s_barrier();
    }
}
} // namespace fused_mla_h16
#endif

template <int Heads, bool RawQ, bool Split>
__global__ __launch_bounds__(256, 2) void fused_norm_rope_attn_rope_cast_h16_kernel(
    FusedNormRopeAttnRopeCastParams a)
{
#if defined(__gfx950__)
    using namespace opus;
    using T = opus_mla_v41_prefill_a8w8_16mx1_16nx4_traits<16, 64, 4, ::fp8_t, bf16_t, bf16_t>;
    using E = FusedNormRopeAttnRopeCastConfig<0, RawQ, RawQ, 0, Heads, 32>;
    namespace fused = fused_norm_rope_attn_rope_cast_fwd;
    namespace small = fused_mla_h16;
    fused::specialize_heads<Heads>(a);
    const int token = blockIdx.x, lane = threadIdx.x % 64, warp = threadIdx.x / 64;
    const int head_start = Heads == 16 ? 0 : blockIdx.y * 16;
    __shared__ __align__(16) char smem[T::smem_bytes() + 256];
    char* kv_smem  = smem;
    char* max_smem = kv_smem + 64 * 528 * 2;
    char* p_smem   = max_smem + 256;
    char* sum_smem = p_smem + 2048;
    vector_t<::fp8_t, 128> q;
    int scale_q;
    if constexpr(RawQ)
        scale_q = small::prepare_q<Heads == 16>(a, q, token, lane, head_start, warp, kv_smem);
    else
    {
        auto gq = make_gmem(static_cast<const ::fp8_t*>(a.q_ptr) +
                                (int64_t(token) * Heads + head_start) * 512,
                            16 * 512);
        auto gs = make_gmem(static_cast<const ::fp8_t*>(a.q_scale_ptr) +
                                (int64_t(token) * Heads + head_start) * 16,
                            16 * 16);
        q       = load<16>(gq, small::make_layout_q<T>(lane, 512));
        scale_q = std::bit_cast<int>(load<1>(gs, small::make_layout_q_scale<T>(lane, 16)));
    }
    vector_t<float, 32> output{};
    float m = -__FLT_MAX__, l = 0.0f, true_max = -__builtin_inff();
    constexpr float LOG2_E = 1.44269504089f, LN_2 = 0.69314718056f;
    const auto p = fused::csr_range<E, Split>(a, false, token);
    const auto e = fused::csr_range<E, Split>(a, true, token);
    small::accumulate<T>(a,
                         p,
                         e,
                         kv_smem,
                         max_smem,
                         p_smem,
                         sum_smem,
                         q,
                         scale_q,
                         output,
                         m,
                         l,
                         true_max,
                         a.softmax_scale * LOG2_E);
    if constexpr(Split)
    {
        float* partial = a.partial_ptr + (int64_t(token) * a.num_splits + blockIdx.z) * Heads * 516;
        const int group = Heads == 16 ? 0 : (int(blockIdx.y) + 2 - warp % 2) % 2;
        static_for<8>([&](auto i) {
            *reinterpret_cast<vector_t<float, 4>*>(partial + (group * 8 + i.value) * 256 * 4 +
                                                   threadIdx.x * 4) =
                slice(output, number<i.value * 4>{}, number<i.value * 4 + 4>{});
        });
        if(warp == 0 && lane < 16)
            *reinterpret_cast<vector_t<float, 4>*>(partial + Heads * 512 +
                                                   (head_start + lane) * 4) =
                vector_t<float, 4>{m, l, true_max, 0.0f};
        return;
    }
    if(warp == 0 && lane < 16)
    {
        if(a.max_ptr)
            a.max_ptr[int64_t(token) * Heads + head_start + lane] = true_max * LN_2;
        if(a.lse_ptr)
            static_cast<float*>(a.lse_ptr)[int64_t(token) * Heads + head_start + lane] =
                l > 0.0f ? (m + __builtin_amdgcn_logf(l)) * LN_2 : __builtin_inff();
    }
    const float sink =
        a.attn_sink_ptr
            ? static_cast<const float*>(a.attn_sink_ptr)[head_start + lane % 16] * LOG2_E
            : -__builtin_inff();
    const float final_m     = fmaxf(m, sink);
    const float alpha       = __builtin_amdgcn_exp2f(m - final_m);
    const float denominator = l * alpha + __builtin_amdgcn_exp2f(sink - final_m);
    output *= denominator > 0.0f ? alpha / denominator : 0.0f;
    fused::fused_epilogue<E, decltype(output), true>(a, output, warp, lane, token, head_start);
#endif
}


__global__ __launch_bounds__(64) void fused_h16_tiled_split_reduce(
    FusedNormRopeAttnRopeCastParams a)
{
#if defined(__gfx950__)
    using namespace opus;
    const int lane = threadIdx.x, token = blockIdx.x;
    const int head = blockIdx.y * 4 + lane % 4, d = blockIdx.z * 64 + (lane / 4) * 4;
    constexpr int stride = 16 * 516;
    const float* base = a.partial_ptr + int64_t(token) * a.num_splits * stride;
    float sums[2], maxima[2], factors[2];
    float maximum = -__FLT_MAX__;
    static_for<2>([&](auto i) {
        const int split = lane / 4 + i.value * 16;
        const auto stats = *reinterpret_cast<const vector_t<float, 4>*>(
            base + (split < a.num_splits ? split : 0) * stride + 16 * 512 + head * 4);
        sums[i.value] = split < a.num_splits ? stats[1] : 0.0f;
        maxima[i.value] = stats[0];
        maximum = fmaxf(maximum, sums[i.value] > 0.0f ? stats[0] : -__FLT_MAX__);
    });
    for(int offset = 4; offset < 64; offset *= 2)
        maximum = fmaxf(maximum, __shfl_xor(maximum, offset));
    const float sink = a.attn_sink_ptr
                           ? static_cast<const float*>(a.attn_sink_ptr)[head] * 1.44269504089f
                           : -__builtin_inff();
    maximum = fmaxf(maximum, sink);
    float denominator = 0.0f;
    static_for<2>([&](auto i) {
        factors[i.value] = sums[i.value] > 0.0f
                               ? __builtin_amdgcn_exp2f(maxima[i.value] - maximum) : 0.0f;
        denominator += sums[i.value] * factors[i.value];
    });
    for(int offset = 4; offset < 64; offset *= 2)
        denominator += __shfl_xor(denominator, offset);
    if(a.attn_sink_ptr)
        denominator += __builtin_amdgcn_exp2f(sink - maximum);
    const float reciprocal = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    vector_t<float, 4> output{};
    const int warp = d / 128;
    const int source_lane = head + (d % 16) / 4 * 16;
    const int chunk = (d % 128) / 16;
    for(int split = 0; split < a.num_splits; split += 4)
    {
        vector_t<float, 4> values[4];
        float weights[4];
        static_for<4>([&](auto i) {
            const int index = split + i.value;
            const int safe_index = index < a.num_splits ? index : split;
            const float factor = safe_index < 16 ? factors[0] : factors[1];
            weights[i.value] = index < a.num_splits
                                   ? __shfl(factor, (safe_index % 16) * 4 + lane % 4) : 0.0f;
            values[i.value] = *reinterpret_cast<const vector_t<float, 4>*>(
                base + safe_index * stride + chunk * 1024 + (warp * 64 + source_lane) * 4);
        });
        static_for<4>([&](auto i) {
            static_for<4>([&](auto j) { output[j.value] += values[i.value][j.value] * weights[i.value]; });
        });
    }
    output = cast<float>(cast<dtypes::bf16>(output * reciprocal));
    if(d >= 448)
    {
        const int64_t position = a.positions64
                                     ? static_cast<const int64_t*>(a.positions_ptr)[token]
                                     : static_cast<const int*>(a.positions_ptr)[token];
        const auto* cosine = static_cast<const dtypes::bf16*>(a.cos_ptr) + position * 32 + (d - 448) / 2;
        const auto* sine = static_cast<const dtypes::bf16*>(a.sin_ptr) + position * 32 + (d - 448) / 2;
        static_for<2>([&](auto i) {
            constexpr int j = i.value * 2;
            const float c = cast<float>(cosine[i.value]), v = cast<float>(sine[i.value]);
            const float even = output[j], odd = output[j + 1];
            output[j] = even * c + odd * v;
            output[j + 1] = odd * c - even * v;
        });
    }
    float maximum_abs = 1e-8f;
    static_for<4>([&](auto i) { maximum_abs = fmaxf(maximum_abs, fabsf(output[i.value])); });
    for(int offset = 4; offset < 32; offset *= 2)
        maximum_abs = fmaxf(maximum_abs, __shfl_xor(maximum_abs, offset));
    const unsigned bits = std::bit_cast<unsigned>(maximum_abs / 448.0f);
    const int exponent = int((bits - 1) >> 23) + 1 - 127;
    const float multiplier = std::bit_cast<float>(unsigned(127 - exponent) << 23);
    const auto quantized = cast<opus::fp8_t>(output * multiplier);
    const int64_t row = int64_t(token) * 16 + head;
    *reinterpret_cast<vector_t<opus::fp8_t, 4>*>(a.output_q_ptr + row * 512 + d) = quantized;
    if(d % 32 == 0)
        a.output_s_ptr[row * 16 + d / 32] = exponent + 127;
#endif
}

hipError_t run_fused_norm_rope_attn_rope_cast_fwd_h16(const FusedNormRopeAttnRopeCastParams& args,
                                                      bool raw_q,
                                                      hipStream_t stream)
{
    if(args.H == 16)
    {
        const dim3 grid(args.N, 1, args.num_splits);
        if(args.num_splits > 1)
        {
            if(raw_q)
                fused_norm_rope_attn_rope_cast_h16_kernel<16, true, true>
                    <<<grid, 256, 0, stream>>>(args);
            else
                fused_norm_rope_attn_rope_cast_h16_kernel<16, false, true>
                    <<<grid, 256, 0, stream>>>(args);
            const auto status = hipGetLastError();
            if(status != hipSuccess)
                return status;
            if(!args.debug_ptr && !args.lse_ptr && !args.max_ptr && !args.out_ptr)
            {
                if(args.num_splits >= 4 && args.N <= 32)
                {
                    fused_h16_tiled_split_reduce<<<dim3(args.N, 4, 8), 64, 0, stream>>>(args);
                    return hipGetLastError();
                }
                return launch_small_head_output_reduce(args, stream);
            }
            fused_norm_rope_attn_rope_cast_decode_reduce_kernel<16, 0, false, 32>
                <<<dim3(args.N), 256, 0, stream>>>(args);
        }
        else if(raw_q)
            fused_norm_rope_attn_rope_cast_h16_kernel<16, true, false>
                <<<grid, 256, 0, stream>>>(args);
        else
            fused_norm_rope_attn_rope_cast_h16_kernel<16, false, false>
                <<<grid, 256, 0, stream>>>(args);
        return hipGetLastError();
    }
    return hipErrorInvalidValue;
}

hipError_t run_fused_norm_rope_attn_rope_cast_h32_tile16_attention(
    const FusedNormRopeAttnRopeCastParams& args, bool raw_q, hipStream_t stream)
{
    const dim3 grid(args.N, 2, args.num_splits);
    if(raw_q)
        fused_norm_rope_attn_rope_cast_h16_kernel<32, true, true><<<grid, 256, 0, stream>>>(args);
    else
        fused_norm_rope_attn_rope_cast_h16_kernel<32, false, true><<<grid, 256, 0, stream>>>(args);
    return hipGetLastError();
}
