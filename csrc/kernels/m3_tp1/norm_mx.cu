// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

// The norm is fused_qk_rmsnorm_group_quant's Gemma add+RMSNorm (BlockSize 512, 16 elements a
// thread), same helpers and module flags, so res_out and the bf16 output are its bytes.
#define AITER_NO_TORCH_TYPES
#include "aiter_hip_common.h"
#include "aiter_opus_plus.h"
#include "aiter_stream.h"
#include "m3_tp1_prefill.h"
#include "mx_quant_utils.h"
#include "quant_amax_floor.h"

namespace aiter {
namespace {

// ADD3: x = bf16(r * 2 + s). RED4: r = moe_reduction of the [m, 4, n] route rows (0 + X0 + .. + X3).
// HOUT: write that x to out_bf16 and stop. MX8 / MX4: quantize the bf16-rounded norm output.
template <int BlockSize,
          int thread_data_size,
          bool ADD3,
          bool OUT_BF16,
          bool MX8,
          bool MX4,
          bool RED4,
          bool HOUT>
__global__ void m3_tp1_norm_mx_kernel(const opus::bf16_t* __restrict__ x,
                                      const opus::bf16_t* __restrict__ s_add,
                                      const opus::bf16_t* __restrict__ res,
                                      const opus::bf16_t* __restrict__ w,
                                      opus::bf16_t* __restrict__ res_out,
                                      opus::bf16_t* __restrict__ out_bf16,
                                      uint8_t* __restrict__ q8,
                                      uint8_t* __restrict__ s8,
                                      uint8_t* __restrict__ q4,
                                      uint8_t* __restrict__ s4,
                                      float eps,
                                      int m,
                                      int n,
                                      int s8_w4)
{
#if defined(__gfx950__)
    using DTYPE_I = opus::bf16_t;
    static constexpr int32_t load_chunk_bytes =
        sizeof(DTYPE_I) * thread_data_size % 16 == 0 ? 16 : 8;
    static constexpr int32_t load_vec_size = load_chunk_bytes / sizeof(DTYPE_I);
    static constexpr int32_t num_load_inst = thread_data_size / load_vec_size;
    static constexpr int32_t load_aux      = (num_load_inst > 1) ? RT : GROUP_NT;
    static constexpr int32_t ooba_i        = 4 / sizeof(DTYPE_I);
    constexpr bool interleave              = false;
    constexpr int interleave_size          = WARP_SIZE;

    const int idx = blockIdx.x;
    if(idx >= m)
        return;
    const int tid = threadIdx.x;
    using vec_i   = opus::vector_t<DTYPE_I, thread_data_size>;
    using vec_f   = opus::vector_t<float, thread_data_size>;
    using vec2_f  = opus::vector_t<float, 2>;

    const int oob_i        = (n + ooba_i - 1) / ooba_i * ooba_i;
    const int row_offset   = tid * thread_data_size;
    const bool row_active  = row_offset < n;
    const int64_t roff     = idx * static_cast<int64_t>(n);

    auto in_buffer = opus::make_gmem<DTYPE_I>(x + roff, oob_i * sizeof(DTYPE_I));
    auto w_buffer  = opus::make_gmem<DTYPE_I>(w, oob_i * sizeof(DTYPE_I));
    vec_i thread_data_i{};
    vec_i thread_data_w{};
    vec_i thread_data_s{};
    vec_i thread_data_res{};
    if(row_active)
    {
        vec_i xk[4];
        if constexpr(RED4)
        {
            auto x4_buffer = opus::make_gmem<DTYPE_I>(x + roff * 4, 4 * oob_i * sizeof(DTYPE_I));
            for(int k = 0; k < 4; ++k)
                xk[k] = load_vector_nbytes<DTYPE_I, thread_data_size, load_chunk_bytes, load_aux, interleave, interleave_size>(
                    x4_buffer, k * n + row_offset);
        }
        else
            thread_data_i = load_vector_nbytes<DTYPE_I, thread_data_size, load_chunk_bytes, load_aux, interleave, interleave_size>(
                in_buffer, row_offset);
        if constexpr(!HOUT)
            thread_data_w = load_vector_nbytes<DTYPE_I, thread_data_size, load_chunk_bytes, RT, interleave, interleave_size>(
                w_buffer, row_offset);
        if constexpr(ADD3)
        {
            auto s_buffer = opus::make_gmem<DTYPE_I>(s_add + roff, oob_i * sizeof(DTYPE_I));
            thread_data_s = load_vector_nbytes<DTYPE_I, thread_data_size, load_chunk_bytes, load_aux, interleave, interleave_size>(
                s_buffer, row_offset);
        }
        if constexpr(!HOUT)
        {
            auto res_buffer = opus::make_gmem<DTYPE_I>(res + roff, oob_i * sizeof(DTYPE_I));
            thread_data_res = load_vector_nbytes<DTYPE_I, thread_data_size, load_chunk_bytes, load_aux, interleave, interleave_size>(
                res_buffer, row_offset);
        }
        if constexpr(RED4)
        {
            // inline asm: -ffast-math must neither reassociate the sum nor drop the 0 + X0 (signed zero)
            for(int i = 0; i < thread_data_size; ++i)
            {
                float acc      = 0.0f;
                const float v0 = static_cast<float>(xk[0][i]);
                asm volatile("v_add_f32 %0, %1, %2" : "=v"(acc) : "v"(acc), "v"(v0));
                for(int k = 1; k < 4; ++k)
                {
                    const float vk = static_cast<float>(xk[k][i]);
                    asm volatile("v_add_f32 %0, %1, %2" : "=v"(acc) : "v"(acc), "v"(vk));
                }
                thread_data_i[i] = opus::cast<DTYPE_I>(acc);
            }
        }
    }
    vec_f thread_data_f{};
    vec2_f rcp;
    if(row_active)
    {
        if constexpr(ADD3)
        {
            for(int i = 0; i < thread_data_size; ++i)
            {
                const float h = __fadd_rn(__fmul_rn(static_cast<float>(thread_data_i[i]), 2.0f),
                                          static_cast<float>(thread_data_s[i]));
                thread_data_i[i] = opus::cast<DTYPE_I>(h);
            }
        }
        if constexpr(HOUT)
        {
            for(int i = 0; i < thread_data_size; ++i)
                thread_data_f[i] = static_cast<float>(thread_data_i[i]);
            auto out_h_buffer = opus::make_gmem<DTYPE_I>(out_bf16 + roff, oob_i * sizeof(DTYPE_I));
            store_vector<DTYPE_I, float, thread_data_size, load_aux, interleave, interleave_size, num_load_inst, DTYPE_I>(
                out_h_buffer, thread_data_f, row_offset);
        }
        else
        {
            for(int i = 0; i < thread_data_size; ++i)
                thread_data_f[i] = static_cast<float>(thread_data_i[i]) + static_cast<float>(thread_data_res[i]);
            auto out_res_buffer = opus::make_gmem<DTYPE_I>(res_out + roff, oob_i * sizeof(DTYPE_I));
            store_vector<DTYPE_I, float, thread_data_size, load_aux, interleave, interleave_size, num_load_inst, DTYPE_I>(
                out_res_buffer, thread_data_f, row_offset);
        }
    }
    if constexpr(HOUT)
        return;

    float square_sum = 0.0f;
    for(int i = 0; i < thread_data_size; ++i)
        square_sum += thread_data_f[i] * thread_data_f[i];
    auto sum_f = [](float a, float b) { return a + b; };
    rcp[0]     = block_reduce<float, decltype(sum_f), BlockSize, true>(square_sum, sum_f);
    rcp[0]     = rsqrtf(rcp[0] / n + eps);
    rcp[1]     = rcp[0];

    vec2_f* thread_data_f2 = reinterpret_cast<vec2_f*>(&thread_data_f);
    if(row_active)
    {
        for(int i = 0; i < thread_data_size / 2; ++i)
        {
            asm volatile("v_pk_mul_f32 %0, %1, %2" : "=v"(thread_data_f2[i]) : "v"(thread_data_f2[i]), "v"(rcp));
        }
        for(int i = 0; i < thread_data_size / 2; ++i)
        {
            vec2_f& thread_data_w2 = rcp;
            thread_data_w2[0]      = static_cast<float>(thread_data_w[2 * i]);
            thread_data_w2[1]      = static_cast<float>(thread_data_w[2 * i + 1]);
            thread_data_w2[0] += 1.0f; // GEMMA_NORM
            thread_data_w2[1] += 1.0f;
            asm volatile("v_pk_mul_f32 %0, %1, %2"
                         : "=v"(thread_data_f2[i])
                         : "v"(thread_data_f2[i]), "v"(thread_data_w2));
        }
    }
    if constexpr(OUT_BF16)
    {
        if(row_active)
        {
            auto out_u_buffer = opus::make_gmem<DTYPE_I>(out_bf16 + roff, oob_i * sizeof(DTYPE_I));
            store_vector<DTYPE_I, float, thread_data_size, RT, interleave, interleave_size, num_load_inst, DTYPE_I>(
                out_u_buffer, thread_data_f, row_offset);
        }
    }
    if constexpr(MX8 || MX4)
    {
        // the bf16 the norm output holds; a 32-group is threads 2g and 2g + 1
        vec_i xb{};
        float amax_local = 0.0f;
        for(int i = 0; i < thread_data_size; ++i)
        {
            xb[i]      = opus::cast<DTYPE_I>(thread_data_f[i]);
            amax_local = fmaxf(amax_local, fabsf(static_cast<float>(xb[i])));
        }
        const float amax_pair = fmaxf(amax_local, __shfl_xor(amax_local, 1));
        const int g           = tid / 2;
        const bool writer     = (tid % 2 == 0) && row_active;
        if constexpr(MX8)
        {
            // code = ceil_pow2(max(amax, 1e-10) / 448), q = clamp(x * 2^(127 - code), +-448) -> e4m3 RNE
            const float amax    = fmaxf(amax_pair, kAmaxFloor);
            const uint32_t bits = __builtin_bit_cast(uint32_t, __fmul_rn(amax, 1.0f / 448.0f));
            const uint32_t code = (bits >> 23) + ((bits & 0x7FFFFFu) != 0u ? 1u : 0u);
            const int32_t rexp  = 254 - static_cast<int32_t>(code);
            const float recip   = __builtin_bit_cast(float, static_cast<uint32_t>(rexp > 0 ? rexp : 0) << 23);
            static_assert(thread_data_size == 16);
            uint32_t words[thread_data_size / 4];
            for(int k = 0; k < thread_data_size / 4; ++k)
            {
                float v[4];
                for(int t = 0; t < 4; ++t)
                {
                    const float y = __fmul_rn(static_cast<float>(xb[4 * k + t]), recip);
                    v[t]          = fminf(fmaxf(y, -448.0f), 448.0f);
                }
                int wd   = 0;
                wd       = __builtin_amdgcn_cvt_pk_fp8_f32(v[0], v[1], wd, false);
                wd       = __builtin_amdgcn_cvt_pk_fp8_f32(v[2], v[3], wd, true);
                words[k] = static_cast<uint32_t>(wd);
            }
            if(row_active)
                *reinterpret_cast<uint4*>(q8 + roff + row_offset) = make_uint4(words[0], words[1], words[2], words[3]);
            if(writer)
            {
                int64_t si;
                if(s8_w4)
                {
                    // shuffle_scale_w4: 256-byte block (row / 32, group / 8), word (g % 4) * 16 + row % 16, byte (g / 4) * 2 + row % 32 / 16
                    const int rr = idx & 31, gg = g & 7;
                    si = (static_cast<int64_t>(idx >> 5) * (n >> 8) + (g >> 3)) * 256 +
                         ((gg & 3) * 16 + (rr & 15)) * 4 + (gg >> 2) * 2 + (rr >> 4);
                }
                else
                    si = idx * static_cast<int64_t>(n / 32) + g;
                s8[si] = static_cast<uint8_t>(code);
            }
        }
        if constexpr(MX4)
        {
            // aiter's per-1x32 MXFP4 rule (dynamic_per_group_scaled_quant, e8m0 RoundUp)
            const float amax = fmaxf(amax_pair, kAmaxFloor);
            const float sc   = fp_f32_to_e8m0_scale<kDefaultMxScaleRoundMode, MxDtype::FP4_E2M1>(amax);
            if(row_active)
            {
                auto q4_buffer = opus::make_gmem<uint8_t>(q4 + idx * static_cast<int64_t>(n / 2), (n / 2) * sizeof(uint8_t));
                store_vector<uint8_t, DTYPE_I, thread_data_size, RT, false, WARP_SIZE, 1, opus::fp4_t, false>(
                    q4_buffer, xb, row_offset / 2, sc);
            }
            if(writer)
                s4[idx * static_cast<int64_t>(n / 32) + g] =
                    static_cast<uint8_t>((__builtin_bit_cast(uint32_t, sc) >> 23) & 0xFFu);
        }
    }
#else
    // gfx950-only (OCP e4m3, MXFP4 casts): compiled out for the other archs of a multi-arch
    // build; the host wrapper never launches it there.
    __builtin_trap();
#endif
}

constexpr int64_t kAdd3 = 1, kOutBf16 = 2, kMx8 = 4, kMx4 = 8, kRed4 = 16, kHout = 32;

// t's data if the flags use it: contiguous, `dtype` (bf16) or 1-byte (u8), at least `numel` elements
template <typename T>
T* checked(const std::optional<aiter_tensor_t>& t, bool used, const char* name, AiterDtype dtype, int64_t numel)
{
    if(!used)
        return nullptr;
    AITER_CHECK(t.has_value() && t->is_contiguous() && t->numel() >= static_cast<size_t>(numel) &&
                    (dtype == AITER_DTYPE_u8 ? t->element_size() == 1 : t->dtype() == dtype),
                "m3_tp1_norm_mx: ", name, " must be a contiguous ", dtype == AITER_DTYPE_u8 ? "1-byte" : "bf16",
                " tensor of at least ", numel, " elements for these flags");
    return static_cast<T*>(t->data_ptr());
}

} // namespace

void m3_tp1_norm_mx(const aiter_tensor_t& x,
                    int64_t flags,
                    std::optional<aiter_tensor_t> s_add,
                    std::optional<aiter_tensor_t> res,
                    std::optional<aiter_tensor_t> w,
                    std::optional<aiter_tensor_t> res_out,
                    std::optional<aiter_tensor_t> out_bf16,
                    std::optional<aiter_tensor_t> q8,
                    std::optional<aiter_tensor_t> s8,
                    std::optional<aiter_tensor_t> q4,
                    std::optional<aiter_tensor_t> s4,
                    double eps,
                    int64_t m,
                    int64_t n,
                    bool s8_w4)
{
    aiter_detail::AiterThrowGuard throw_guard;

    AITER_CHECK(flags == kMx8 || flags == (kAdd3 | kMx8) || flags == (kRed4 | kAdd3 | kMx8) ||
                    flags == (kOutBf16 | kMx8) || flags == (kOutBf16 | kMx8 | kMx4) ||
                    flags == (kAdd3 | kHout) || flags == (kRed4 | kAdd3 | kHout),
                "m3_tp1_norm_mx: unsupported flags ", flags);
    // the instantiation fused_qk_rmsnorm_group_quant runs for this width (BlockSize 512)
    AITER_CHECK(n > 4096 && n <= 8192 && n % 32 == 0, "m3_tp1_norm_mx: n must be in (4096, 8192] and a multiple of 32, got ", n);
    AITER_CHECK(!s8_w4 || ((flags & kMx8) && n % 256 == 0), "m3_tp1_norm_mx: s8_w4 needs MX8 and n % 256 == 0");
    AITER_CHECK(m >= 0 && m <= INT32_MAX, "m3_tp1_norm_mx: bad m ", m);
    const bool red4 = flags & kRed4, hout = flags & kHout;
    AITER_CHECK(x.dtype() == AITER_DTYPE_bf16 && x.is_contiguous() && x.dim() == (red4 ? 3 : 2) &&
                    x.size(0) == m && x.size(-1) == n && (!red4 || x.size(1) == 4),
                "m3_tp1_norm_mx: x must be a contiguous bf16 ", red4 ? "[m, 4, n]" : "[m, n]", " tensor");
    const int64_t mn = m * n, s8_rows = s8_w4 ? (m + 255) / 256 * 256 : m;
    const AiterDtype dt_bf16 = AITER_DTYPE_bf16, dt_u8 = AITER_DTYPE_u8;
    auto* s_p  = checked<const opus::bf16_t>(s_add, flags & kAdd3, "s_add", dt_bf16, mn);
    auto* r_p  = checked<const opus::bf16_t>(res, !hout, "res", dt_bf16, mn);
    auto* w_p  = checked<const opus::bf16_t>(w, !hout, "w", dt_bf16, n);
    auto* ro_p = checked<opus::bf16_t>(res_out, !hout, "res_out", dt_bf16, mn);
    auto* ob_p = checked<opus::bf16_t>(out_bf16, (flags & kOutBf16) || hout, "out_bf16", dt_bf16, mn);
    auto* q8_p = checked<uint8_t>(q8, flags & kMx8, "q8", dt_u8, mn);
    auto* s8_p = checked<uint8_t>(s8, flags & kMx8, "s8", dt_u8, s8_rows * (n / 32));
    auto* q4_p = checked<uint8_t>(q4, flags & kMx4, "q4", dt_u8, mn / 2);
    auto* s4_p = checked<uint8_t>(s4, flags & kMx4, "s4", dt_u8, mn / 32);
    AITER_CHECK(reinterpret_cast<uintptr_t>(q8_p) % 16 == 0, "m3_tp1_norm_mx: q8 must be 16-byte aligned");
    if(m == 0)
        return;

    const HipDeviceGuard guard(x.device_id);
    const hipStream_t stream = getCurrentHIPStream();
#define M3_TP1_NORM_LAUNCH(A3, OB, M8, M4, R4, HO)                                          \
    m3_tp1_norm_mx_kernel<512, 16, A3, OB, M8, M4, R4, HO><<<dim3(m), dim3(512), 0, stream>>>( \
        static_cast<const opus::bf16_t*>(x.data_ptr()), s_p, r_p, w_p, ro_p, ob_p, q8_p, s8_p,   \
        q4_p, s4_p, static_cast<float>(eps), static_cast<int>(m), static_cast<int>(n), s8_w4 ? 1 : 0)
    switch(flags)
    {
    case kMx8: M3_TP1_NORM_LAUNCH(false, false, true, false, false, false); break;
    case kAdd3 | kMx8: M3_TP1_NORM_LAUNCH(true, false, true, false, false, false); break;
    case kRed4 | kAdd3 | kMx8: M3_TP1_NORM_LAUNCH(true, false, true, false, true, false); break;
    case kOutBf16 | kMx8: M3_TP1_NORM_LAUNCH(false, true, true, false, false, false); break;
    case kOutBf16 | kMx8 | kMx4: M3_TP1_NORM_LAUNCH(false, true, true, true, false, false); break;
    case kAdd3 | kHout: M3_TP1_NORM_LAUNCH(true, false, false, false, false, true); break;
    case kRed4 | kAdd3 | kHout: M3_TP1_NORM_LAUNCH(true, false, false, false, true, true); break;
    }
#undef M3_TP1_NORM_LAUNCH
    HIP_CALL(hipGetLastError());
}

} // namespace aiter
