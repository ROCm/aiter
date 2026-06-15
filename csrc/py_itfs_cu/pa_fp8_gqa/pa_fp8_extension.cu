// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026 Page_Attetion_GQA_fp8 project
//
// PyTorch extension for the FP8 paged-attention decode kernel (v0).
//
// Public C++ entry points (exposed via pybind11):
//   pa_fp8_decode_v0 : main + reduce dispatch (Q/K per-token, V per-head)
//
// Layout matches `pa_decode_gluon` (see Python wrapper for full contract).

#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>

#include <cstdint>

#include "pa_fp8_main_kernel.cuh"
#include "pa_fp8_main_kernel_v2.cuh"
#include "pa_fp8_reduce_kernel.cuh"

using namespace pa_fp8_gqa;

namespace {

inline bool is_fp8_e4m3(const at::Tensor& t)
{
    return t.dtype() == at::kFloat8_e4m3fnuz;
}

inline bool is_bf16(const at::Tensor& t)
{
    return t.dtype() == at::kBFloat16;
}

inline bool is_fp16(const at::Tensor& t)
{
    return t.dtype() == at::kHalf;
}

// Tag helper used by the v1/v2 dispatcher to carry a compile-time QIn type
// through lambda hops (so we don't need verbose 4-level nested ifs).
template <typename T> struct qin_t { using type = T; };

inline bool has_pscale_tensors(const at::Tensor& p_scale,
                               const at::Tensor& p_scale_inv)
{
    // EMPTY tensor (numel == 0) is the convention for "no p-scale".  Both
    // tensors must agree on presence to avoid silent half-applied scaling.
    const bool a = p_scale.defined()     && p_scale.numel()     > 0;
    const bool b = p_scale_inv.defined() && p_scale_inv.numel() > 0;
    TORCH_CHECK(a == b,
        "p_scale and p_scale_inv must be either both empty or both populated");
    return a;
}

inline const float* maybe_data_ptr_f32(const at::Tensor& t)
{
    if (!t.defined() || t.numel() == 0) return nullptr;
    TORCH_CHECK(t.scalar_type() == at::kFloat,
                "auxiliary tensor must be float32");
    return t.data_ptr<float>();
}

template <typename output_t>
void launch_main_v0_impl(
    const at::Tensor& query,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const at::Tensor& block_tables,
    const at::Tensor& context_lens,
    const at::Tensor& q_scale_t,
    const at::Tensor& k_scale_t,
    const at::Tensor& v_scale_t,
    at::Tensor&       exp_sums,
    at::Tensor&       max_logits,
    at::Tensor&       tmp_out,
    int num_seqs, int num_kv_heads, int num_q_heads,
    int head_size, int block_size, int mtp,
    int max_num_partitions,
    double scale)
{
    TORCH_CHECK(num_q_heads == 8, "v0 requires num_q_heads=8, got ", num_q_heads);
    TORCH_CHECK(num_kv_heads == 1, "v0 requires num_kv_heads=1, got ", num_kv_heads);
    TORCH_CHECK(head_size == 128, "v0 requires head_size=128, got ", head_size);
    TORCH_CHECK(block_size == 16, "v0 requires block_size=16, got ", block_size);
    TORCH_CHECK(mtp == 1 || mtp == 2,
                "v0 requires mtp in {1, 2}, got ", mtp);

    const auto stream = at::hip::getCurrentHIPStream();

    // strides expected by the kernel
    const int q_stride        = query.stride(0);
    const int kv_block_stride = k_cache.stride(0);
    const int kv_head_stride  = k_cache.stride(1);
    const int max_num_blocks_per_seq = static_cast<int>(block_tables.size(1));

    dim3 grid(num_seqs, max_num_partitions, num_kv_heads);
    dim3 block(v0::kNumThreads);

    const __hip_fp8_e4m3_fnuz* q_ptr =
        reinterpret_cast<const __hip_fp8_e4m3_fnuz*>(query.data_ptr());
    const __hip_fp8_e4m3_fnuz* k_ptr =
        reinterpret_cast<const __hip_fp8_e4m3_fnuz*>(k_cache.data_ptr());
    const __hip_fp8_e4m3_fnuz* v_ptr =
        reinterpret_cast<const __hip_fp8_e4m3_fnuz*>(v_cache.data_ptr());

    auto launch_with_mtp = [&](auto mtp_const) {
        constexpr int kMtpC = decltype(mtp_const)::value;
        pa_fp8_main_kernel_v0<output_t, kMtpC>
            <<<grid, block, 0, stream>>>(
                q_ptr, k_ptr, v_ptr,
                static_cast<float>(scale),
                q_scale_t.data_ptr<float>(),
                k_scale_t.data_ptr<float>(),
                v_scale_t.data_ptr<float>(),
                block_tables.data_ptr<int>(),
                context_lens.data_ptr<int>(),
                max_num_blocks_per_seq,
                q_stride, kv_block_stride, kv_head_stride,
                exp_sums.data_ptr<float>(),
                max_logits.data_ptr<float>(),
                reinterpret_cast<output_t*>(tmp_out.data_ptr()));
    };

    if (mtp == 1) launch_with_mtp(std::integral_constant<int, 1>{});
    else          launch_with_mtp(std::integral_constant<int, 2>{});
}

template <typename output_t>
void launch_reduce_impl(
    at::Tensor&       out,
    const at::Tensor& exp_sums,
    const at::Tensor& max_logits,
    const at::Tensor& tmp_out,
    const at::Tensor& context_lens,
    int num_seqs, int num_heads, int /*head_size*/, int mtp,
    int max_num_partitions,
    int fixed_num_partitions = -1,
    int num_kblocks_per_fat_part = 0)
{
    const auto stream = at::hip::getCurrentHIPStream();
    dim3 grid(num_heads, num_seqs, mtp);
    dim3 block(v_red::kReduceNumThreads);

    if (max_num_partitions <= 64) {
        auto launch_v3 = [&](auto np_const) {
            constexpr int kNP = decltype(np_const)::value;
            pa_fp8_reduce_kernel_v3<output_t, kNP>
                <<<grid, block, 0, stream>>>(
                    reinterpret_cast<output_t*>(out.data_ptr()),
                    exp_sums.data_ptr<float>(),
                    max_logits.data_ptr<float>(),
                    reinterpret_cast<const output_t*>(tmp_out.data_ptr()),
                    context_lens.data_ptr<int>(),
                    max_num_partitions,
                    fixed_num_partitions,
                    num_kblocks_per_fat_part);
        };

        if      (max_num_partitions <= 1 ) launch_v3(std::integral_constant<int, 1 >{});
        else if (max_num_partitions <= 2 ) launch_v3(std::integral_constant<int, 2 >{});
        else if (max_num_partitions <= 4 ) launch_v3(std::integral_constant<int, 4 >{});
        else if (max_num_partitions <= 8 ) launch_v3(std::integral_constant<int, 8 >{});
        else if (max_num_partitions <= 16) launch_v3(std::integral_constant<int, 16>{});
        else if (max_num_partitions <= 32) launch_v3(std::integral_constant<int, 32>{});
        else                               launch_v3(std::integral_constant<int, 64>{});
        return;
    }

    const int npar_loops =
        (max_num_partitions + WARP_SIZE - 1) / WARP_SIZE;

    if (npar_loops <= 2) {
        auto launch_v2 = [&](auto nc_const) {
            constexpr int kNC = decltype(nc_const)::value;
            pa_fp8_reduce_kernel_v2<output_t, kNC>
                <<<grid, block, 0, stream>>>(
                    reinterpret_cast<output_t*>(out.data_ptr()),
                    exp_sums.data_ptr<float>(),
                    max_logits.data_ptr<float>(),
                    reinterpret_cast<const output_t*>(tmp_out.data_ptr()),
                    context_lens.data_ptr<int>(),
                    max_num_partitions,
                    fixed_num_partitions,
                    num_kblocks_per_fat_part);
        };

        if (npar_loops <= 1) launch_v2(std::integral_constant<int, 1>{});
        else                 launch_v2(std::integral_constant<int, 2>{});
        return;
    }

    auto launch = [&](auto nl_const) {
        constexpr int kNL = decltype(nl_const)::value;
        pa_fp8_reduce_kernel<output_t, kNL>
            <<<grid, block, 0, stream>>>(
                reinterpret_cast<output_t*>(out.data_ptr()),
                exp_sums.data_ptr<float>(),
                max_logits.data_ptr<float>(),
                reinterpret_cast<const output_t*>(tmp_out.data_ptr()),
                context_lens.data_ptr<int>(),
                max_num_partitions,
                fixed_num_partitions,
                num_kblocks_per_fat_part);
    };

    if      (npar_loops <= 4 ) launch(std::integral_constant<int, 4 >{});
    else if (npar_loops <= 8 ) launch(std::integral_constant<int, 8 >{});
    else if (npar_loops <= 16) launch(std::integral_constant<int, 16>{});
    else                       launch(std::integral_constant<int, 32>{});
}

// (launch_main_v1_impl removed — v1 path deprecated; decode_auto uses v2.)

// ---------------------------------------------------------------------------
// v2 main launcher — gluon-aligned inter-tile K prefetch.
// ---------------------------------------------------------------------------
template <typename output_t>
void launch_main_v2_impl(
    const at::Tensor& query,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const at::Tensor& block_tables,
    const at::Tensor& context_lens,
    const at::Tensor& q_scale_t,
    const at::Tensor& k_scale_t,
    const at::Tensor& v_scale_t,
    const at::Tensor& p_scale_t,
    const at::Tensor& p_scale_inv_t,
    double p_scale_value,
    double p_scale_inv_value,
    at::Tensor&       exp_sums,
    at::Tensor&       max_logits,
    at::Tensor&       tmp_out,
    int num_seqs, int num_kv_heads, int num_q_heads,
    int head_size, int block_size, int mtp,
    int num_fat_partitions,
    int num_kblocks_per_fat_part,
    double scale)
{
    TORCH_CHECK(num_q_heads == 8, "v2 requires num_q_heads=8, got ", num_q_heads);
    TORCH_CHECK(num_kv_heads == 1, "v2 requires num_kv_heads=1, got ", num_kv_heads);
    TORCH_CHECK(head_size == 128, "v2 requires head_size=128, got ", head_size);
    TORCH_CHECK(block_size == 16, "v2 requires block_size=16, got ", block_size);
    TORCH_CHECK(mtp == 1 || mtp == 2, "v2 requires mtp in {1, 2}, got ", mtp);
    TORCH_CHECK(num_kblocks_per_fat_part >= 1,
                "num_kblocks_per_fat_part must be >= 1");

    const bool query_is_fp8  = is_fp8_e4m3(query);
    const bool query_is_bf16 = is_bf16(query);
    TORCH_CHECK(query_is_fp8 || query_is_bf16,
        "v2 query must be float8_e4m3fnuz or bfloat16; got ", query.dtype());
    const bool has_pscale_tensor = has_pscale_tensors(p_scale_t, p_scale_inv_t);
    const bool has_pscale_scalar = (p_scale_value > 0.0 || p_scale_inv_value > 0.0);
    TORCH_CHECK(!has_pscale_scalar || (p_scale_value > 0.0 && p_scale_inv_value > 0.0),
        "scalar p_scale and p_scale_inv must either both be positive or both be zero");
    TORCH_CHECK(!(has_pscale_tensor && has_pscale_scalar),
        "p_scale/p_scale_inv must be provided as either tensors or scalar values, not both");
    const bool has_pscale = has_pscale_tensor || has_pscale_scalar;

    const auto stream = at::hip::getCurrentHIPStream();
    const int q_stride        = query.stride(0);
    const int kv_block_stride = k_cache.stride(0);
    const int kv_head_stride  = k_cache.stride(1);
    const int max_num_blocks_per_seq = static_cast<int>(block_tables.size(1));

    // k_scale strides (in fp32 elements), read from the tensor itself so any
    // layout — dense flat or strided/padded FlyDSL packed view — is consumed
    // zero-copy by the kernel (no host .contiguous() required):
    //   flat   [nb, nkv, block_size]       → stride(0)=nkv*bs, stride(-2)=bs
    //   packed [nb, 1,   nkv, head_dim/4]   → stride(0)=padded-block stride,
    //                                         stride(-2)=head_dim/4
    // The kernel indexes
    //   k_scale[kphys*ks_block_stride + kv_head*ks_head_stride + slot].
    const int ks_block_stride =
        static_cast<int>(k_scale_t.stride(0));
    const int ks_head_stride =
        static_cast<int>(k_scale_t.stride(k_scale_t.dim() - 2));

    dim3 grid(num_seqs, num_fat_partitions, num_kv_heads);
    dim3 block(v0::kNumThreads);

    const __hip_fp8_e4m3_fnuz* k_ptr =
        reinterpret_cast<const __hip_fp8_e4m3_fnuz*>(k_cache.data_ptr());
    const __hip_fp8_e4m3_fnuz* v_ptr =
        reinterpret_cast<const __hip_fp8_e4m3_fnuz*>(v_cache.data_ptr());

    const float* q_scale_ptr     = query_is_fp8 ? q_scale_t.data_ptr<float>()
                                                : nullptr;
    const float* p_scale_ptr     = maybe_data_ptr_f32(p_scale_t);
    const float* p_scale_inv_ptr = maybe_data_ptr_f32(p_scale_inv_t);

    auto launch_full = [&](auto mtp_c, auto qin_tag, auto hps_c, auto pf_c) {
        constexpr int  kMtp     = decltype(mtp_c)::value;
        using QIn               = typename decltype(qin_tag)::type;
        constexpr bool kHPS     = decltype(hps_c)::value;
        constexpr bool kPrefetch = decltype(pf_c)::value;

        const QIn* q_ptr = reinterpret_cast<const QIn*>(query.data_ptr());

        pa_fp8_main_kernel_v2<output_t, kMtp, QIn, kHPS, kPrefetch>
            <<<grid, block, 0, stream>>>(
                q_ptr, k_ptr, v_ptr,
                static_cast<float>(scale),
                q_scale_ptr,
                k_scale_t.data_ptr<float>(),
                v_scale_t.data_ptr<float>(),
                p_scale_ptr,
                p_scale_inv_ptr,
                static_cast<float>(p_scale_value),
                static_cast<float>(p_scale_inv_value),
                block_tables.data_ptr<int>(),
                context_lens.data_ptr<int>(),
                max_num_blocks_per_seq,
                q_stride, kv_block_stride, kv_head_stride,
                ks_block_stride, ks_head_stride,
                exp_sums.data_ptr<float>(),
                max_logits.data_ptr<float>(),
                reinterpret_cast<output_t*>(tmp_out.data_ptr()),
                num_kblocks_per_fat_part);
    };

    // Cross-kbi K/K-scale prefetch helps when each CTA executes ≥2 kbi
    // iters: the prefetch latency hides behind PV MFMA of the previous
    // iter.  When `num_kblocks_per_fat_part == 1` (1 iter per CTA), the
    // prologue prefetch + a wasted "next-kbi" prefetch on the only iter
    // are pure overhead → fall back to the no-prefetch baseline.
    //
    // CUDA-graph-capture safe: dispatch decision uses only
    // `num_kblocks_per_fat_part`, which is a function of (bs, mtp,
    // num_fat_partitions) — all known at launcher time independent of
    // actual context_len.
    const bool use_prefetch = (num_kblocks_per_fat_part >= 2);

    auto launch_with_pf = [&](auto mtp_c, auto qin_tag, auto hps_c) {
        if (use_prefetch) launch_full(mtp_c, qin_tag, hps_c, std::true_type{});
        else              launch_full(mtp_c, qin_tag, hps_c, std::false_type{});
    };

    auto launch_with_hps = [&](auto mtp_c, auto qin_tag) {
        if (has_pscale) launch_with_pf(mtp_c, qin_tag, std::true_type{});
        else            launch_with_pf(mtp_c, qin_tag, std::false_type{});
    };

    auto launch_with_qin = [&](auto mtp_c) {
        if (query_is_fp8)
            launch_with_hps(mtp_c, qin_t<__hip_fp8_e4m3_fnuz>{});
        else
            launch_with_hps(mtp_c, qin_t<__hip_bfloat16>{});
    };

    if (mtp == 1) launch_with_qin(std::integral_constant<int, 1>{});
    else          launch_with_qin(std::integral_constant<int, 2>{});
}

// (launch_main_v3_impl removed — v3 cross-iter prefetch dropped occupancy.)

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public entry: pa_fp8_decode_v0
//
// Signature mirrors v5 with extra per-tensor scale args:
//   output, tmp_out, exp_sums, max_logits, query, k_cache, v_cache,
//   block_tables, context_lens, num_seqs, num_kv_heads, num_q_heads,
//   head_size, block_size, mtp, max_num_partitions, scale,
//   q_scale, k_scale, v_scale.
// ---------------------------------------------------------------------------
void pa_fp8_decode_v0(
    at::Tensor&       output,
    at::Tensor&       tmp_out,
    at::Tensor&       exp_sums,
    at::Tensor&       max_logits,
    const at::Tensor& query,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const at::Tensor& block_tables,
    const at::Tensor& context_lens,
    const at::Tensor& q_scale,           // [1] fp32 (per-tensor)
    const at::Tensor& k_scale,           // [1] fp32
    const at::Tensor& v_scale,           // [1] fp32
    int64_t num_seqs, int64_t num_kv_heads, int64_t num_q_heads,
    int64_t head_size, int64_t block_size, int64_t mtp,
    int64_t max_num_partitions,
    double scale)
{
    const c10::hip::OptionalHIPGuardMasqueradingAsCUDA guard(query.device());

    TORCH_CHECK(is_fp8_e4m3(query),   "query must be float8_e4m3fnuz");
    TORCH_CHECK(is_fp8_e4m3(k_cache), "k_cache must be float8_e4m3fnuz");
    TORCH_CHECK(is_fp8_e4m3(v_cache), "v_cache must be float8_e4m3fnuz");
    TORCH_CHECK(q_scale.scalar_type() == at::kFloat
                && k_scale.scalar_type() == at::kFloat
                && v_scale.scalar_type() == at::kFloat,
                "scales must be float32 tensors");

    if (is_bf16(output))
    {
        launch_main_v0_impl<__hip_bfloat16>(
            query, k_cache, v_cache, block_tables, context_lens,
            q_scale, k_scale, v_scale,
            exp_sums, max_logits, tmp_out,
            (int)num_seqs, (int)num_kv_heads, (int)num_q_heads,
            (int)head_size, (int)block_size, (int)mtp,
            (int)max_num_partitions, scale);
        launch_reduce_impl<__hip_bfloat16>(
            output, exp_sums, max_logits, tmp_out, context_lens,
            (int)num_seqs, (int)num_q_heads, (int)head_size, (int)mtp,
            (int)max_num_partitions,
            /*fixed_num_partitions=*/-1,
            /*num_kblocks_per_fat_part=*/0);
    }
    else if (is_fp16(output))
    {
        launch_main_v0_impl<_Float16>(
            query, k_cache, v_cache, block_tables, context_lens,
            q_scale, k_scale, v_scale,
            exp_sums, max_logits, tmp_out,
            (int)num_seqs, (int)num_kv_heads, (int)num_q_heads,
            (int)head_size, (int)block_size, (int)mtp,
            (int)max_num_partitions, scale);
        launch_reduce_impl<_Float16>(
            output, exp_sums, max_logits, tmp_out, context_lens,
            (int)num_seqs, (int)num_q_heads, (int)head_size, (int)mtp,
            (int)max_num_partitions,
            /*fixed_num_partitions=*/-1,
            /*num_kblocks_per_fat_part=*/0);
    }
    else
    {
        TORCH_CHECK(false, "output must be bf16 or fp16");
    }
}

// (pa_fp8_decode_v1 removed — v1 path deprecated; decode_auto uses v2.)

// ---------------------------------------------------------------------------
// pa_fp8_decode_v2 — gluon-aligned fat-CTA variant, optimized for bs >= 32.
//
// Same signature as v1 except the kernel uses inter-tile K prefetch, lower
// occupancy (1 CTA/CU) and gluon-style sched_group_barrier patterns.  The
// (output, tmp_out, exp_sums, max_logits) tensor layouts and the reduce
// kernel that consumes them are identical to v1.
// ---------------------------------------------------------------------------
void pa_fp8_decode_v2(
    at::Tensor&       output,
    at::Tensor&       tmp_out,
    at::Tensor&       exp_sums,
    at::Tensor&       max_logits,
    const at::Tensor& query,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const at::Tensor& block_tables,
    const at::Tensor& context_lens,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale,
    const at::Tensor& p_scale,
    const at::Tensor& p_scale_inv,
    double p_scale_value,
    double p_scale_inv_value,
    int64_t num_seqs, int64_t num_kv_heads, int64_t num_q_heads,
    int64_t head_size, int64_t block_size, int64_t mtp,
    int64_t num_fat_partitions,
    int64_t num_kblocks_per_fat_part,
    double scale)
{
    const c10::hip::OptionalHIPGuardMasqueradingAsCUDA guard(query.device());

    TORCH_CHECK(is_fp8_e4m3(query) || is_bf16(query),
        "query must be float8_e4m3fnuz or bfloat16 (got ", query.dtype(), ")");
    TORCH_CHECK(is_fp8_e4m3(k_cache), "k_cache must be float8_e4m3fnuz");
    TORCH_CHECK(is_fp8_e4m3(v_cache), "v_cache must be float8_e4m3fnuz");
    TORCH_CHECK(k_scale.scalar_type() == at::kFloat
                && v_scale.scalar_type() == at::kFloat,
                "k_scale / v_scale must be float32 tensors");
    if (is_fp8_e4m3(query))
        TORCH_CHECK(q_scale.scalar_type() == at::kFloat,
            "q_scale must be float32 when query is fp8");

    if (is_bf16(output))
    {
        launch_main_v2_impl<__hip_bfloat16>(
            query, k_cache, v_cache, block_tables, context_lens,
            q_scale, k_scale, v_scale, p_scale, p_scale_inv,
            p_scale_value, p_scale_inv_value,
            exp_sums, max_logits, tmp_out,
            (int)num_seqs, (int)num_kv_heads, (int)num_q_heads,
            (int)head_size, (int)block_size, (int)mtp,
            (int)num_fat_partitions, (int)num_kblocks_per_fat_part, scale);
        // num_kblocks_per_fat_part=0 is the v2 RUNTIME-page-size sentinel:
        // the reduce derives active partitions from the actual context +
        // num_fat_partitions, matching the v2 main kernel's runtime
        // `cdiv(total_num_kblocks, gridDim.y)` slicing (FlyDSL-aligned).
        launch_reduce_impl<__hip_bfloat16>(
            output, exp_sums, max_logits, tmp_out, context_lens,
            (int)num_seqs, (int)num_q_heads, (int)head_size, (int)mtp,
            (int)num_fat_partitions,
            /*fixed_num_partitions=*/(int)num_fat_partitions,
            /*num_kblocks_per_fat_part=*/0);
    }
    else if (is_fp16(output))
    {
        launch_main_v2_impl<_Float16>(
            query, k_cache, v_cache, block_tables, context_lens,
            q_scale, k_scale, v_scale, p_scale, p_scale_inv,
            p_scale_value, p_scale_inv_value,
            exp_sums, max_logits, tmp_out,
            (int)num_seqs, (int)num_kv_heads, (int)num_q_heads,
            (int)head_size, (int)block_size, (int)mtp,
            (int)num_fat_partitions, (int)num_kblocks_per_fat_part, scale);
        // num_kblocks_per_fat_part=0: v2 RUNTIME-page-size sentinel (see bf16 branch).
        launch_reduce_impl<_Float16>(
            output, exp_sums, max_logits, tmp_out, context_lens,
            (int)num_seqs, (int)num_q_heads, (int)head_size, (int)mtp,
            (int)num_fat_partitions,
            /*fixed_num_partitions=*/(int)num_fat_partitions,
            /*num_kblocks_per_fat_part=*/0);
    }
    else
    {
        TORCH_CHECK(false, "output must be bf16 or fp16");
    }
}

// (pa_fp8_decode_v3 removed — v3 cross-iter prefetch dropped occupancy.)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("pa_fp8_decode_v0", &pa_fp8_decode_v0,
          "FP8 paged-attention decode v0 (main + reduce, Q/K per-token, V per-head scales)",
          pybind11::arg("output"),
          pybind11::arg("tmp_out"),
          pybind11::arg("exp_sums"),
          pybind11::arg("max_logits"),
          pybind11::arg("query"),
          pybind11::arg("k_cache"),
          pybind11::arg("v_cache"),
          pybind11::arg("block_tables"),
          pybind11::arg("context_lens"),
          pybind11::arg("q_scale"),
          pybind11::arg("k_scale"),
          pybind11::arg("v_scale"),
          pybind11::arg("num_seqs"),
          pybind11::arg("num_kv_heads"),
          pybind11::arg("num_q_heads"),
          pybind11::arg("head_size"),
          pybind11::arg("block_size"),
          pybind11::arg("mtp"),
          pybind11::arg("max_num_partitions"),
          pybind11::arg("scale"));

    m.def("pa_fp8_decode_v2", &pa_fp8_decode_v2,
          "FP8 paged-attention decode v2 (gluon-aligned fat-CTA with "
          "inter-tile K prefetch; tuned for bs >= 32, ctx >= 64k; "
          "FlyDSL-aligned: accepts bf16 OR fp8 query and optional p_scale)",
          pybind11::arg("output"),
          pybind11::arg("tmp_out"),
          pybind11::arg("exp_sums"),
          pybind11::arg("max_logits"),
          pybind11::arg("query"),
          pybind11::arg("k_cache"),
          pybind11::arg("v_cache"),
          pybind11::arg("block_tables"),
          pybind11::arg("context_lens"),
          pybind11::arg("q_scale"),
          pybind11::arg("k_scale"),
          pybind11::arg("v_scale"),
          pybind11::arg("p_scale"),
          pybind11::arg("p_scale_inv"),
          pybind11::arg("p_scale_value"),
          pybind11::arg("p_scale_inv_value"),
          pybind11::arg("num_seqs"),
          pybind11::arg("num_kv_heads"),
          pybind11::arg("num_q_heads"),
          pybind11::arg("head_size"),
          pybind11::arg("block_size"),
          pybind11::arg("mtp"),
          pybind11::arg("num_fat_partitions"),
          pybind11::arg("num_kblocks_per_fat_part"),
          pybind11::arg("scale"));
}
