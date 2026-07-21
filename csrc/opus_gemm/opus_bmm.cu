// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

// Host-side BMM frontends. These expose BMM/grouped-layout APIs while reusing
// the generated opus GEMM backend launcher symbols.
#include "gfx950/opus_gemm_pipeline_a8w8_scale_gfx950.cuh"
#include "gfx950/opus_gemm_pipeline_a8w8_mxscale_flatmm_splitk_gfx950.cuh"

using opus_bmm_a8w8_mxscale_splitk_traits_gfx950 =
    opus_gemm_a8w8_scale_traits_gfx950<512,
      opus::seq<128, 256, 128>,
      opus::tuple<fp8_t, fp8_t, fp32_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>>;

using opus_bmm_a8w8_mxscale_scale_m256n256k128_bf16_traits_gfx950 =
    opus_gemm_a8w8_scale_traits_gfx950<512,
      opus::seq<256, 256, 128>,
      opus::tuple<fp8_t, fp8_t, bf16_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>>;

using opus_bmm_a8w8_mxscale_scale_m256n256k128_fp32_traits_gfx950 =
    opus_gemm_a8w8_scale_traits_gfx950<512,
      opus::seq<256, 256, 128>,
      opus::tuple<fp8_t, fp8_t, fp32_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>>;

using opus_bmm_a8w8_mxscale_flatmm_splitk_traits_gfx950 =
    opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950<256,
      opus::seq<32, 128, 128>,
      opus::tuple<fp8_t, fp8_t, fp32_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>,
      2>;

using opus_bmm_a8w8_mxscale_flatmm64_splitk_traits_gfx950 =
    opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950<256,
      opus::seq<64, 128, 128>,
      opus::tuple<fp8_t, fp8_t, fp32_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>,
      2>;

using opus_bmm_a8w8_mxscale_flatmm_m64n128k256_wg1_splitk_traits_gfx950 =
    opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950<256,
      opus::seq<64, 128, 256>,
      opus::tuple<fp8_t, fp8_t, fp32_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>,
      1>;

using opus_bmm_a8w8_mxscale_flatmm_m128n64k256_wg1_splitk_traits_gfx950 =
    opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950<256,
      opus::seq<128, 64, 256>,
      opus::tuple<fp8_t, fp8_t, fp32_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>,
      1>;

using opus_bmm_a8w8_mxscale_flatmm256_splitk_traits_gfx950 =
    opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950<256,
      opus::seq<32, 256, 128>,
      opus::tuple<fp8_t, fp8_t, fp32_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>,
      1>;

using opus_bmm_a8w8_mxscale_flatmm_m64n32k256_splitk_traits_gfx950 =
    opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950<256,
      opus::seq<64, 32, 256>,
      opus::tuple<fp8_t, fp8_t, fp32_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>,
      2>;

using opus_bmm_a8w8_mxscale_flatmm_m32n64k256_splitk_traits_gfx950 =
    opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950<256,
      opus::seq<32, 64, 256>,
      opus::tuple<fp8_t, fp8_t, fp32_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>,
      2>;

using opus_bmm_a8w8_mxscale_flatmm_m64n64k128_splitk_traits_gfx950 =
    opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950<256,
      opus::seq<64, 64, 128>,
      opus::tuple<fp8_t, fp8_t, fp32_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>,
      2>;

using opus_bmm_a8w8_mxscale_flatmm_m64n64k128_wg1_splitk_traits_gfx950 =
    opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950<256,
      opus::seq<64, 64, 128>,
      opus::tuple<fp8_t, fp8_t, fp32_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>,
      1>;

using opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950 =
    opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950<256,
      opus::seq<128, 128, 128>,
      opus::tuple<fp8_t, fp8_t, fp32_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>,
      1>;

using opus_bmm_a8w8_mxscale_flatmm_m64n32k256_wg1_splitk_traits_gfx950 =
    opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950<256,
      opus::seq<64, 32, 256>,
      opus::tuple<fp8_t, fp8_t, fp32_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>,
      1>;

using opus_bmm_a8w8_mxscale_flatmm_m32n64k256_wg1_splitk_traits_gfx950 =
    opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950<256,
      opus::seq<32, 64, 256>,
      opus::tuple<fp8_t, fp8_t, fp32_t, fp32_t, unsigned char>,
      opus::seq<16, 16, 4>,
      opus::seq<1, 128, 128>,
      1>;

#ifdef __HIP_DEVICE_COMPILE__
template __global__ void
gemm_a8w8_scale_splitk_kernel<opus_bmm_a8w8_mxscale_splitk_traits_gfx950>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_scale_kernel<opus_bmm_a8w8_mxscale_scale_m256n256k128_bf16_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
template __global__ void
gemm_a8w8_scale_kernel<opus_bmm_a8w8_mxscale_scale_m256n256k128_fp32_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
template __global__ void
gemm_a8w8_scale_k1024_kernel<opus_bmm_a8w8_mxscale_scale_m256n256k128_bf16_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
template __global__ void
gemm_a8w8_scale_k1024_kernel<opus_bmm_a8w8_mxscale_scale_m256n256k128_fp32_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
template __global__ void
gemm_a8w8_scale_k1024_lb1_kernel<opus_bmm_a8w8_mxscale_scale_m256n256k128_bf16_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
template __global__ void
gemm_a8w8_scale_k1024_lb1_kernel<opus_bmm_a8w8_mxscale_scale_m256n256k128_fp32_traits_gfx950>(
    opus_gemm_scale_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_splitk_traits_gfx950>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_splitk_traits_gfx950, __bf16>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_splitk_traits_gfx950, float>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm64_splitk_traits_gfx950>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm64_splitk_traits_gfx950, __bf16>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm64_splitk_traits_gfx950, float>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m64n128k256_wg1_splitk_traits_gfx950, __bf16>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m64n128k256_wg1_splitk_traits_gfx950, float>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m128n64k256_wg1_splitk_traits_gfx950, __bf16>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m128n64k256_wg1_splitk_traits_gfx950, float>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm256_splitk_traits_gfx950>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m64n32k256_splitk_traits_gfx950>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m64n32k256_splitk_traits_gfx950, __bf16>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m64n32k256_splitk_traits_gfx950, float>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m32n64k256_splitk_traits_gfx950>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m32n64k256_splitk_traits_gfx950, __bf16>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m32n64k256_splitk_traits_gfx950, float>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m64n64k128_splitk_traits_gfx950>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m64n64k128_splitk_traits_gfx950, __bf16>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m64n64k128_splitk_traits_gfx950, float>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m64n64k128_splitk_traits_gfx950, __bf16, false, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m64n64k128_splitk_traits_gfx950, float, false, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, __bf16>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, float>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, __bf16, false, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, float, false, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_nphase_kernel<opus_bmm_a8w8_mxscale_flatmm64_splitk_traits_gfx950, __bf16, 2>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_nphase_kernel<opus_bmm_a8w8_mxscale_flatmm64_splitk_traits_gfx950, float, 2>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_mouter_kernel<opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, __bf16>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_mouter_kernel<opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, float>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_mouter_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, __bf16, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_mouter_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, float, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave8n2_kernel<opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, __bf16>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave8n2_kernel<opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, float>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_kernel<opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, __bf16>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_kernel<opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, float>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, __bf16, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, float, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, __bf16, false, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, float, false, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, __bf16, false, true, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, float, false, true, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, __bf16, false, true, false, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, float, false, true, false, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, __bf16, false, true, false, false, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, float, false, true, false, false, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4m2_selfload_kernel<opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, __bf16>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4m2_selfload_kernel<opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, float>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4m2_selfload_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, __bf16, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4m2_selfload_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, float, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4m2_selfload_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, __bf16, true, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_wave4m2_selfload_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, float, true, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m32n64k256_splitk_traits_gfx950, __bf16, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<
    opus_bmm_a8w8_mxscale_flatmm_m32n64k256_splitk_traits_gfx950, float, true>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m64n32k256_wg1_splitk_traits_gfx950, __bf16>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m64n32k256_wg1_splitk_traits_gfx950, float>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m32n64k256_wg1_splitk_traits_gfx950, __bf16>(
    opus_gemm_scale_splitk_kargs_gfx950);
template __global__ void
gemm_a8w8_mxscale_flatmm_splitk_kernel<opus_bmm_a8w8_mxscale_flatmm_m32n64k256_wg1_splitk_traits_gfx950, float>(
    opus_gemm_scale_splitk_kargs_gfx950);
#endif

template <typename D_OUT, int VEC = 16, int BLOCK = 64>
__global__ void opus_bmm_splitk_reduce_kernel(
    const opus_splitk_ws_handle* __restrict__ ws_handle,
    D_OUT* __restrict__ out,
    int split_k, int M, int N, int batch,
    int padded_M, int padded_N,
    int stride_c, int stride_c_batch)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
  using namespace opus;
  constexpr int STEP = 16 / sizeof(D_OUT);
  static_assert(VEC % STEP == 0);

  const int bm_id = int(block_id_y());
  const int n_base = (int(block_id_x()) * BLOCK + int(thread_id_x())) * VEC;
  if (bm_id >= batch * M || n_base >= N) return;
  const int b = bm_id / M;
  const int m = bm_id - b * M;

  const float* __restrict__ workspace =
      reinterpret_cast<const float*>(ws_handle->ptr);
  const long split_stride = (long)batch * padded_M * padded_N;
  const int base = b * padded_M * padded_N + m * padded_N + n_base;
  auto g_ws = make_gmem(workspace, (unsigned int)(split_stride * split_k * sizeof(float)));
  vector_t<float, VEC> acc;
  static_for<VEC>([&](auto i) { acc[i.value] = 0.0f; });
  for (int s = 0; s < split_k; ++s) {
    #pragma unroll
    for (int g = 0; g < VEC / 4; ++g) {
      auto v = g_ws.template load<4>((long)s * split_stride + base + g * 4);
      static_for<4>([&](auto j) { acc[g * 4 + j.value] += v[j.value]; });
    }
  }

  vector_t<D_OUT, VEC> out_v;
  static_for<VEC>([&](auto i) { out_v[i.value] = static_cast<D_OUT>(acc[i.value]); });
  const int c_idx = b * stride_c_batch + m * stride_c + n_base;
  const size_t c_records =
      (size_t)(M - 1) * (size_t)stride_c
    + (size_t)(batch - 1) * (size_t)stride_c_batch
    + (size_t)N;
  auto g_c = make_gmem(out, (unsigned int)(c_records * sizeof(D_OUT)));
  if (n_base + VEC <= N) {
    static_for<VEC / STEP>([&](auto group) {
      constexpr int off = group.value * STEP;
      g_c.template store<STEP>(slice(out_v, number<off>{}, number<off + STEP>{}), c_idx + off);
    });
  } else {
    #pragma unroll
    for (int i = 0; i < VEC; ++i) {
      if (n_base + i < N) g_c.template store<1>(out_v[i], c_idx + i);
    }
  }
#endif
#endif
}

#ifdef __HIP_DEVICE_COMPILE__
template __global__ void opus_bmm_splitk_reduce_kernel<__bf16>(
    const opus_splitk_ws_handle*, __bf16*,
    int, int, int, int, int, int, int, int);
template __global__ void opus_bmm_splitk_reduce_kernel<float>(
    const opus_splitk_ws_handle*, float*,
    int, int, int, int, int, int, int, int);
template __global__ void opus_bmm_splitk_reduce_kernel<__bf16, 8, 128>(
    const opus_splitk_ws_handle*, __bf16*,
    int, int, int, int, int, int, int, int);
template __global__ void opus_bmm_splitk_reduce_kernel<float, 8, 128>(
    const opus_splitk_ws_handle*, float*,
    int, int, int, int, int, int, int, int);
#endif

#ifndef __HIP_DEVICE_COMPILE__

#include "opus_bmm.h"
#include "opus_gemm_arch.cuh"
#include "opus_build_archs.h"
#include "opus_gemm_manifest.h"
#include "opus_gemm_utils.cuh"  // bf16_t / fp32_t
#include "aiter_stream.h"

#include <optional>

// ── opus_bmm_a8w8_scale_mmajor() — zero-copy fp8 block-scale BMM ────────────
// O/Y are [M, batch, *] (dim0=M, dim1=batch); x_scale is
// [M, batch, K/GROUP_K] (per-token M). Weight WQ + w_scale stay batch-major
// ([batch, N, K] / [batch, N/GROUP_N, K/GROUP_K]). No caller-side transpose --
// feeds the DSV4 wo_a activation o=[num_tokens, n_groups, K] directly.
#ifdef OPUS_BUILD_HAS_GFX950
template <typename D_C>
void opus_gemm_512x128x256x128_4x2_16x16x128_1x128x128_mmajor(
    aiter_tensor_t &, aiter_tensor_t &, aiter_tensor_t &,
    std::optional<aiter_tensor_t>, std::optional<aiter_tensor_t>);
template <typename D_C>
void opus_gemm_a8w8_mxscale_512x128x256x128_4x2_16x16x128_1x128x128_mmajor(
    aiter_tensor_t &, aiter_tensor_t &, aiter_tensor_t &,
    std::optional<aiter_tensor_t>, std::optional<aiter_tensor_t>);
template<typename Traits>
__global__ void gemm_a8w8_scale_splitk_kernel(opus_gemm_scale_splitk_kargs_gfx950 kargs);
template<typename Traits, typename D_OUT, bool DIRECT_ONLY>
__global__ void gemm_a8w8_mxscale_flatmm_splitk_kernel(opus_gemm_scale_splitk_kargs_gfx950 kargs);
template<typename Traits, typename D_OUT, int N_PHASES>
__global__ void gemm_a8w8_mxscale_flatmm_splitk_nphase_kernel(opus_gemm_scale_splitk_kargs_gfx950 kargs);
template<typename Traits, typename D_OUT>
__global__ void gemm_a8w8_mxscale_flatmm_splitk_mouter_kernel(opus_gemm_scale_splitk_kargs_gfx950 kargs);
template<typename Traits, typename D_OUT>
__global__ void gemm_a8w8_mxscale_flatmm_splitk_wave8n2_kernel(opus_gemm_scale_splitk_kargs_gfx950 kargs);
template<typename Traits, typename D_OUT>
__global__ void gemm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_kernel(opus_gemm_scale_splitk_kargs_gfx950 kargs);
template<typename Traits, typename D_OUT, bool SKIP_SCALE_WAIT,
         bool PACK_SCALE_ON_DEMAND>
__global__ void gemm_a8w8_mxscale_flatmm_splitk_wave4m2_selfload_kernel(opus_gemm_scale_splitk_kargs_gfx950 kargs);
#endif

static void opus_bmm_a8w8_common_checks(aiter_tensor_t &O, aiter_tensor_t &wo_a,
                                        aiter_tensor_t &Y, const char *who)
{
  aiter_detail::g_aiter_can_throw = true;
  AITER_CHECK(O.dim() == 3 && wo_a.dim() == 3 && Y.dim() == 3,
              who, ": O/wo_a/Y must be 3D "
              "([M,batch,K] / [batch,N,K] / [M,batch,N])");
  AITER_CHECK(O.dtype() == AITER_DTYPE_fp8 && wo_a.dtype() == AITER_DTYPE_fp8,
              who, ": O and wo_a must be fp8");
  AITER_CHECK(Y.dtype() == AITER_DTYPE_fp32 || Y.dtype() == AITER_DTYPE_bf16,
              who, ": Y must be fp32 or bf16");
}

void opus_bmm_a8w8_scale_mmajor(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale)
{
  opus_bmm_a8w8_common_checks(O, wo_a, Y, "opus_bmm_a8w8_scale_mmajor");
#ifdef OPUS_BUILD_HAS_GFX950
  const auto &arch_info = opus_get_arch_info();
  AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950,
              "opus_bmm_a8w8_scale_mmajor is gfx950-only; current device ",
              arch_info.dev, " has gcnArchName='", arch_info.name, "'");
  if (Y.dtype() == AITER_DTYPE_bf16) {
    opus_gemm_512x128x256x128_4x2_16x16x128_1x128x128_mmajor<bf16_t>(
        O, wo_a, Y, x_scale, w_scale);
  } else {
    opus_gemm_512x128x256x128_4x2_16x16x128_1x128x128_mmajor<fp32_t>(
        O, wo_a, Y, x_scale, w_scale);
  }
#else
  AITER_CHECK(false,
              "opus_bmm_a8w8_scale_mmajor requires OPUS_BUILD_HAS_GFX950");
#endif
}

void opus_bmm_a8w8_mxscale_mmajor(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    int kernelId)
{
  opus_bmm_a8w8_common_checks(O, wo_a, Y, "opus_bmm_a8w8_mxscale_mmajor");
#ifdef OPUS_BUILD_HAS_GFX950
  const auto &arch_info = opus_get_arch_info();
  AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950,
              "opus_bmm_a8w8_mxscale_mmajor is gfx950-only; current device ",
              arch_info.dev, " has gcnArchName='", arch_info.name, "'");
  (void)kernelId;
  if (Y.dtype() == AITER_DTYPE_bf16) {
    opus_gemm_a8w8_mxscale_512x128x256x128_4x2_16x16x128_1x128x128_mmajor<bf16_t>(
        O, wo_a, Y, x_scale, w_scale);
  } else {
    opus_gemm_a8w8_mxscale_512x128x256x128_4x2_16x16x128_1x128x128_mmajor<fp32_t>(
        O, wo_a, Y, x_scale, w_scale);
  }
#else
  AITER_CHECK(false,
              "opus_bmm_a8w8_mxscale_mmajor requires OPUS_BUILD_HAS_GFX950");
#endif
}

void opus_bmm_a8w8_mxscale_splitk_mmajor(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    int splitK)
{
  opus_bmm_a8w8_common_checks(O, wo_a, Y, "opus_bmm_a8w8_mxscale_splitk_mmajor");
#ifdef OPUS_BUILD_HAS_GFX950
  const auto &arch_info = opus_get_arch_info();
  AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950,
              "opus_bmm_a8w8_mxscale_splitk_mmajor is gfx950-only; current device ",
              arch_info.dev, " has gcnArchName='", arch_info.name, "'");
  AITER_CHECK(splitK > 1, "splitK must be > 1");

  using Traits = opus_bmm_a8w8_mxscale_splitk_traits_gfx950;

  const int M = O.size(0);
  const int batch = O.size(1);
  const int N = wo_a.size(1);
  const int K = O.size(2);
  const int split_k = splitK;
  const int total_iters = (K + Traits::B_K - 1) / Traits::B_K;
  const int iters_full = (total_iters + split_k - 1) / split_k;
  const int last_loops = total_iters - (split_k - 1) * iters_full;
  AITER_CHECK(last_loops >= 2,
              "opus_bmm_a8w8_mxscale_splitk_mmajor requires every split to "
              "have at least 2 K-tiles; K=", K, ", splitK=", split_k,
              ", last split loops=", last_loops);
  const int num_tiles_m = (M + 128 - 1) / 128;
  const int num_tiles_n = (N + 256 - 1) / 256;
  const int padded_M = num_tiles_m * 128;
  const int padded_N = num_tiles_n * 256;

  extern opus_splitk_ws_handle* opus_splitk_ws_get(hipStream_t, bool);
  auto stream = aiter::getCurrentHIPStream();
  hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
  HIP_CALL(hipStreamIsCapturing(stream, &capture_status));
  const bool capturing = (capture_status != hipStreamCaptureStatusNone);
  auto* ws_handle = opus_splitk_ws_get(stream, /*allow_create=*/!capturing);

  size_t ws_bytes = (size_t)split_k * (size_t)batch
                  * (size_t)padded_M * (size_t)padded_N * sizeof(float);
  if (ws_handle->ptr == nullptr || ws_bytes > ws_handle->bytes) {
    AITER_CHECK(!capturing,
                "splitk workspace grow inside HIP graph capture is not supported");
    void* new_ptr = nullptr;
    const size_t kGrowAlign = (size_t)4 * 1024 * 1024;
    size_t grow_bytes = ((ws_bytes + kGrowAlign - 1) / kGrowAlign) * kGrowAlign;
    HIP_CALL(hipMalloc(&new_ptr, grow_bytes));
    if (ws_handle->ptr != nullptr) {
      HIP_CALL(hipDeviceSynchronize());
      HIP_CALL(hipFree(ws_handle->ptr));
    }
    ws_handle->ptr = new_ptr;
    ws_handle->bytes = grow_bytes;
  }

  opus_gemm_scale_splitk_kargs_gfx950 kargs{};
  kargs.ptr_a = O.data_ptr();
  kargs.ptr_b = wo_a.data_ptr();
  kargs.ws_handle = ws_handle;
  kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = batch;
  kargs.split_k = split_k;
  kargs.stride_a = (int)O.stride(0);
  kargs.stride_b = (int)wo_a.stride(1);
  kargs.stride_ws = padded_N;
  kargs.stride_a_batch = (int)O.stride(1);
  kargs.stride_b_batch = (int)wo_a.stride(0);
  kargs.stride_ws_batch = padded_M * padded_N;
  kargs.ptr_sfa = x_scale.data_ptr();
  kargs.ptr_sfb = w_scale.data_ptr();
  kargs.stride_sfa = (int)x_scale.stride(0);
  kargs.stride_sfa_batch = (int)x_scale.stride(1);
  kargs.stride_sfb = (int)w_scale.stride(1);
  kargs.stride_sfb_batch = (int)w_scale.stride(0);

  dim3 grid_main(num_tiles_m * num_tiles_n * split_k, 1, batch);
  dim3 block_main(512);
  gemm_a8w8_scale_splitk_kernel<Traits><<<grid_main, block_main, 0, stream>>>(kargs);

  constexpr int REDUCE_VEC = 16;
  constexpr int REDUCE_BS = 64;
  dim3 grid_reduce((N + REDUCE_VEC * REDUCE_BS - 1) / (REDUCE_VEC * REDUCE_BS),
                   batch * M, 1);
  dim3 block_reduce(REDUCE_BS);
  const int y_stride_c = (int)Y.stride(0);
  const int y_stride_c_batch = (int)Y.stride(1);
  if (Y.dtype() == AITER_DTYPE_bf16) {
    opus_bmm_splitk_reduce_kernel<__bf16>
        <<<grid_reduce, block_reduce, 0, stream>>>(
            ws_handle, reinterpret_cast<__bf16*>(Y.data_ptr()),
            split_k, M, N, batch, padded_M, padded_N,
            y_stride_c, y_stride_c_batch);
  } else {
    opus_bmm_splitk_reduce_kernel<float>
        <<<grid_reduce, block_reduce, 0, stream>>>(
            ws_handle, reinterpret_cast<float*>(Y.data_ptr()),
            split_k, M, N, batch, padded_M, padded_N,
            y_stride_c, y_stride_c_batch);
  }
#else
  AITER_CHECK(false,
              "opus_bmm_a8w8_mxscale_splitk_mmajor requires OPUS_BUILD_HAS_GFX950");
#endif
}

template <typename Traits, bool DIRECT_ONLY = false, bool PREFETCH_SCALE = false>
static void opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor_impl(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    int splitK,
    bool fusedReduce,
    const char* who)
{
  opus_bmm_a8w8_common_checks(O, wo_a, Y, who);
#ifdef OPUS_BUILD_HAS_GFX950
  const auto &arch_info = opus_get_arch_info();
  AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950,
              who, " is gfx950-only; current device ",
              arch_info.dev, " has gcnArchName='", arch_info.name, "'");
  AITER_CHECK(splitK >= 1, "splitK must be >= 1");
  if constexpr (DIRECT_ONLY) {
    AITER_CHECK(splitK == 1, who, " consumer-self-load kernel requires splitK == 1");
  }
  if (fusedReduce) {
    AITER_CHECK(splitK == 2, who, " fused reduce currently supports splitK == 2 only");
    AITER_CHECK(Traits::B_M == 32 && Traits::B_N == 128 && Traits::B_K == 128,
                who, " fused reduce currently supports the flatmm 32x128x128 variant only");
  }

  const int M = O.size(0);
  const int batch = O.size(1);
  const int N = wo_a.size(1);
  const int K = O.size(2);
  AITER_CHECK(M % Traits::B_M == 0,
              "flatmm splitK v1 requires M % ", Traits::B_M, " == 0, got ", M);
  AITER_CHECK(N % Traits::B_N == 0,
              "flatmm splitK v1 requires N % ", Traits::B_N, " == 0, got ", N);
  AITER_CHECK(K % Traits::B_K == 0,
              "flatmm splitK v1 requires K % ", Traits::B_K, " == 0, got ", K);

  const int split_k = splitK;
  // Match the rest of opus GEMM: splitK is the number of K partitions.
  // splitK == 1 means no K split, so the main kernel writes Y directly.
  const bool no_split_k = (split_k == 1);
  const int total_iters = K / Traits::B_K;
  const int iters_full = (total_iters + split_k - 1) / split_k;
  const int last_loops = total_iters - (split_k - 1) * iters_full;
  AITER_CHECK(last_loops >= Traits::prefetch_k_iter,
              "flatmm splitK v1 requires every split to have at least ",
              Traits::prefetch_k_iter, " K-tiles; K=", K,
              " gives total_iters=", total_iters, ", splitK=", split_k,
              ", last split loops=", last_loops);

  const int num_tiles_m = (M + Traits::B_M - 1) / Traits::B_M;
  const int num_tiles_n = (N + Traits::B_N - 1) / Traits::B_N;
  const int padded_M = num_tiles_m * Traits::B_M;
  const int padded_N = num_tiles_n * Traits::B_N;
  const size_t partial_bytes = (size_t)split_k * (size_t)batch
                             * (size_t)padded_M * (size_t)padded_N * sizeof(float);
  const size_t counter_offset = (partial_bytes + 255) & ~((size_t)255);
  const size_t counter_bytes = (size_t)batch * (size_t)num_tiles_m
                             * (size_t)num_tiles_n * sizeof(int);

  auto stream = aiter::getCurrentHIPStream();

  opus_gemm_scale_splitk_kargs_gfx950 kargs{};
  kargs.ptr_a = O.data_ptr();
  kargs.ptr_b = wo_a.data_ptr();
  kargs.ws_handle = nullptr;
  kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = batch;
  kargs.split_k = split_k;
  kargs.stride_a = (int)O.stride(0);
  kargs.stride_b = (int)wo_a.stride(1);
  kargs.stride_ws = padded_N;
  kargs.stride_a_batch = (int)O.stride(1);
  kargs.stride_b_batch = (int)wo_a.stride(0);
  kargs.stride_ws_batch = padded_M * padded_N;
  kargs.ptr_sfa = x_scale.data_ptr();
  kargs.ptr_sfb = w_scale.data_ptr();
  kargs.stride_sfa = (int)x_scale.stride(0);
  kargs.stride_sfa_batch = (int)x_scale.stride(1);
  kargs.stride_sfb = (int)w_scale.stride(1);
  kargs.stride_sfb_batch = (int)w_scale.stride(0);

  dim3 grid_main(num_tiles_m * num_tiles_n * split_k, 1, batch);
  dim3 block_main(Traits::BLOCK_SIZE);
  if (no_split_k) {
    kargs.ptr_c = Y.data_ptr();
    kargs.stride_c = (int)Y.stride(0);
    kargs.stride_c_batch = (int)Y.stride(1);
    if (Y.dtype() == AITER_DTYPE_bf16) {
      gemm_a8w8_mxscale_flatmm_splitk_kernel<Traits, __bf16, DIRECT_ONLY, PREFETCH_SCALE>
          <<<grid_main, block_main, 0, stream>>>(kargs);
    } else {
      gemm_a8w8_mxscale_flatmm_splitk_kernel<Traits, float, DIRECT_ONLY, PREFETCH_SCALE>
          <<<grid_main, block_main, 0, stream>>>(kargs);
    }
    return;
  }

  extern opus_splitk_ws_handle* opus_splitk_ws_get(hipStream_t, bool);
  hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
  HIP_CALL(hipStreamIsCapturing(stream, &capture_status));
  const bool capturing = (capture_status != hipStreamCaptureStatusNone);
  auto* ws_handle = opus_splitk_ws_get(stream, /*allow_create=*/!capturing);

  const size_t ws_bytes = fusedReduce ? (counter_offset + counter_bytes) : partial_bytes;
  if (ws_handle->ptr == nullptr || ws_bytes > ws_handle->bytes) {
    AITER_CHECK(!capturing,
                "splitk workspace grow inside HIP graph capture is not supported");
    void* new_ptr = nullptr;
    const size_t kGrowAlign = (size_t)4 * 1024 * 1024;
    size_t grow_bytes = ((ws_bytes + kGrowAlign - 1) / kGrowAlign) * kGrowAlign;
    HIP_CALL(hipMalloc(&new_ptr, grow_bytes));
    if (ws_handle->ptr != nullptr) {
      HIP_CALL(hipDeviceSynchronize());
      HIP_CALL(hipFree(ws_handle->ptr));
    }
    ws_handle->ptr = new_ptr;
    ws_handle->bytes = grow_bytes;
  }
  kargs.ws_handle = ws_handle;

  if (fusedReduce) {
    kargs.ptr_c = Y.data_ptr();
    kargs.stride_c = (int)Y.stride(0);
    kargs.stride_c_batch = (int)Y.stride(1);
    kargs.counter_offset_bytes = counter_offset;
    HIP_CALL(hipMemsetAsync(static_cast<char*>(ws_handle->ptr) + counter_offset,
                            0, counter_bytes, stream));
    if (Y.dtype() == AITER_DTYPE_bf16) {
      gemm_a8w8_mxscale_flatmm_splitk_kernel<Traits, __bf16>
          <<<grid_main, block_main, 0, stream>>>(kargs);
    } else {
      gemm_a8w8_mxscale_flatmm_splitk_kernel<Traits, float>
          <<<grid_main, block_main, 0, stream>>>(kargs);
    }
    return;
  }

  gemm_a8w8_mxscale_flatmm_splitk_kernel<Traits>
      <<<grid_main, block_main, 0, stream>>>(kargs);

  constexpr int REDUCE_VEC = 8;
  constexpr int REDUCE_BS = 128;
  dim3 grid_reduce((N + REDUCE_VEC * REDUCE_BS - 1) / (REDUCE_VEC * REDUCE_BS),
                   batch * M, 1);
  dim3 block_reduce(REDUCE_BS);
  const int y_stride_c = (int)Y.stride(0);
  const int y_stride_c_batch = (int)Y.stride(1);
  if (Y.dtype() == AITER_DTYPE_bf16) {
    opus_bmm_splitk_reduce_kernel<__bf16, REDUCE_VEC, REDUCE_BS>
        <<<grid_reduce, block_reduce, 0, stream>>>(
            ws_handle, reinterpret_cast<__bf16*>(Y.data_ptr()),
            split_k, M, N, batch, padded_M, padded_N,
            y_stride_c, y_stride_c_batch);
  } else {
    opus_bmm_splitk_reduce_kernel<float, REDUCE_VEC, REDUCE_BS>
        <<<grid_reduce, block_reduce, 0, stream>>>(
            ws_handle, reinterpret_cast<float*>(Y.data_ptr()),
            split_k, M, N, batch, padded_M, padded_N,
            y_stride_c, y_stride_c_batch);
  }
#else
  AITER_CHECK(false, who, " requires OPUS_BUILD_HAS_GFX950");
#endif
}

template <typename Traits, int N_PHASES>
static void opus_bmm_a8w8_mxscale_flatmm_splitk_nphase_mmajor_impl(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    int splitK,
    const char* who)
{
  opus_bmm_a8w8_common_checks(O, wo_a, Y, who);
#ifdef OPUS_BUILD_HAS_GFX950
  const auto &arch_info = opus_get_arch_info();
  AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950,
              who, " is gfx950-only; current device ",
              arch_info.dev, " has gcnArchName='", arch_info.name, "'");
  AITER_CHECK(splitK == 1, who, " requires splitK == 1");

  const int M = O.size(0);
  const int batch = O.size(1);
  const int N = wo_a.size(1);
  const int K = O.size(2);
  constexpr int LOGICAL_B_N = Traits::B_N * N_PHASES;
  AITER_CHECK(M % Traits::B_M == 0,
              who, " requires M % ", Traits::B_M, " == 0, got ", M);
  AITER_CHECK(N % LOGICAL_B_N == 0,
              who, " requires N % ", LOGICAL_B_N, " == 0, got ", N);
  AITER_CHECK(K % Traits::B_K == 0,
              who, " requires K % ", Traits::B_K, " == 0, got ", K);
  const int total_iters = K / Traits::B_K;
  AITER_CHECK(total_iters >= Traits::prefetch_k_iter,
              who, " requires at least ", Traits::prefetch_k_iter,
              " K-tiles, got ", total_iters);

  auto stream = aiter::getCurrentHIPStream();

  opus_gemm_scale_splitk_kargs_gfx950 kargs{};
  kargs.ptr_a = O.data_ptr();
  kargs.ptr_b = wo_a.data_ptr();
  kargs.ws_handle = nullptr;
  kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = batch;
  kargs.split_k = 1;
  kargs.stride_a = (int)O.stride(0);
  kargs.stride_b = (int)wo_a.stride(1);
  kargs.stride_ws = N;
  kargs.stride_a_batch = (int)O.stride(1);
  kargs.stride_b_batch = (int)wo_a.stride(0);
  kargs.stride_ws_batch = M * N;
  kargs.ptr_sfa = x_scale.data_ptr();
  kargs.ptr_sfb = w_scale.data_ptr();
  kargs.stride_sfa = (int)x_scale.stride(0);
  kargs.stride_sfa_batch = (int)x_scale.stride(1);
  kargs.stride_sfb = (int)w_scale.stride(1);
  kargs.stride_sfb_batch = (int)w_scale.stride(0);
  kargs.ptr_c = Y.data_ptr();
  kargs.stride_c = (int)Y.stride(0);
  kargs.stride_c_batch = (int)Y.stride(1);

  const int num_tiles_m = M / Traits::B_M;
  const int num_tiles_n = N / LOGICAL_B_N;
  dim3 grid_main(num_tiles_m * num_tiles_n, 1, batch);
  dim3 block_main(Traits::BLOCK_SIZE);
  if (Y.dtype() == AITER_DTYPE_bf16) {
    gemm_a8w8_mxscale_flatmm_splitk_nphase_kernel<Traits, __bf16, N_PHASES>
        <<<grid_main, block_main, 0, stream>>>(kargs);
  } else {
    gemm_a8w8_mxscale_flatmm_splitk_nphase_kernel<Traits, float, N_PHASES>
        <<<grid_main, block_main, 0, stream>>>(kargs);
  }
#else
  AITER_CHECK(false, who, " requires OPUS_BUILD_HAS_GFX950");
#endif
}

template <typename Traits, bool SKIP_SCALE_WAIT = false>
static void opus_bmm_a8w8_mxscale_flatmm_splitk_mouter_mmajor_impl(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    int splitK,
    const char* who)
{
  opus_bmm_a8w8_common_checks(O, wo_a, Y, who);
#ifdef OPUS_BUILD_HAS_GFX950
  const auto &arch_info = opus_get_arch_info();
  AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950,
              who, " is gfx950-only; current device ",
              arch_info.dev, " has gcnArchName='", arch_info.name, "'");
  AITER_CHECK(splitK == 1, who, " requires splitK == 1");

  const int M = O.size(0);
  const int batch = O.size(1);
  const int N = wo_a.size(1);
  const int K = O.size(2);
  AITER_CHECK(M % Traits::B_M == 0,
              who, " requires M % ", Traits::B_M, " == 0, got ", M);
  AITER_CHECK(N % Traits::B_N == 0,
              who, " requires N % ", Traits::B_N, " == 0, got ", N);
  AITER_CHECK(K % Traits::B_K == 0,
              who, " requires K % ", Traits::B_K, " == 0, got ", K);
  const int total_iters = K / Traits::B_K;
  AITER_CHECK(total_iters >= Traits::prefetch_k_iter,
              who, " requires at least ", Traits::prefetch_k_iter,
              " K-tiles, got ", total_iters);

  auto stream = aiter::getCurrentHIPStream();

  opus_gemm_scale_splitk_kargs_gfx950 kargs{};
  kargs.ptr_a = O.data_ptr();
  kargs.ptr_b = wo_a.data_ptr();
  kargs.ws_handle = nullptr;
  kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = batch;
  const int num_tiles_m = M / Traits::B_M;
  const int num_tiles_n = N / Traits::B_N;
  const int m_per_wg = (num_tiles_m >= 16) ? 2 : 1;
  kargs.split_k = m_per_wg;
  kargs.stride_a = (int)O.stride(0);
  kargs.stride_b = (int)wo_a.stride(1);
  kargs.stride_ws = N;
  kargs.stride_a_batch = (int)O.stride(1);
  kargs.stride_b_batch = (int)wo_a.stride(0);
  kargs.stride_ws_batch = M * N;
  kargs.ptr_sfa = x_scale.data_ptr();
  kargs.ptr_sfb = w_scale.data_ptr();
  kargs.stride_sfa = (int)x_scale.stride(0);
  kargs.stride_sfa_batch = (int)x_scale.stride(1);
  kargs.stride_sfb = (int)w_scale.stride(1);
  kargs.stride_sfb_batch = (int)w_scale.stride(0);
  kargs.ptr_c = Y.data_ptr();
  kargs.stride_c = (int)Y.stride(0);
  kargs.stride_c_batch = (int)Y.stride(1);

  const int split_m = (num_tiles_m + m_per_wg - 1) / m_per_wg;
  constexpr int NUM_XCD = 8;
  const int m_grp_per_xcd = (split_m + NUM_XCD - 1) / NUM_XCD;
  kargs.stride_ws = split_m;
  kargs.stride_ws_batch = m_grp_per_xcd;
  dim3 grid_main(NUM_XCD * m_grp_per_xcd * num_tiles_n, 1, batch);
  dim3 block_main(Traits::BLOCK_SIZE);
  if (Y.dtype() == AITER_DTYPE_bf16) {
    gemm_a8w8_mxscale_flatmm_splitk_mouter_kernel<Traits, __bf16, SKIP_SCALE_WAIT>
        <<<grid_main, block_main, 0, stream>>>(kargs);
  } else {
    gemm_a8w8_mxscale_flatmm_splitk_mouter_kernel<Traits, float, SKIP_SCALE_WAIT>
        <<<grid_main, block_main, 0, stream>>>(kargs);
  }
#else
  AITER_CHECK(false, who, " requires OPUS_BUILD_HAS_GFX950");
#endif
}

template <typename Traits>
static void opus_bmm_a8w8_mxscale_flatmm_splitk_wave8n2_mmajor_impl(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    int splitK,
    const char* who)
{
  opus_bmm_a8w8_common_checks(O, wo_a, Y, who);
#ifdef OPUS_BUILD_HAS_GFX950
  const auto &arch_info = opus_get_arch_info();
  AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950,
              who, " is gfx950-only; current device ",
              arch_info.dev, " has gcnArchName='", arch_info.name, "'");
  AITER_CHECK(splitK == 1, who, " requires splitK == 1");

  const int M = O.size(0);
  const int batch = O.size(1);
  const int N = wo_a.size(1);
  const int K = O.size(2);
  constexpr int LOGICAL_B_N = Traits::B_N * 2;
  AITER_CHECK(M % Traits::B_M == 0,
              who, " requires M % ", Traits::B_M, " == 0, got ", M);
  AITER_CHECK(N % LOGICAL_B_N == 0,
              who, " requires N % ", LOGICAL_B_N, " == 0, got ", N);
  AITER_CHECK(K % Traits::B_K == 0,
              who, " requires K % ", Traits::B_K, " == 0, got ", K);

  auto stream = aiter::getCurrentHIPStream();

  opus_gemm_scale_splitk_kargs_gfx950 kargs{};
  kargs.ptr_a = O.data_ptr();
  kargs.ptr_b = wo_a.data_ptr();
  kargs.ws_handle = nullptr;
  kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = batch;
  kargs.split_k = 1;
  kargs.stride_a = (int)O.stride(0);
  kargs.stride_b = (int)wo_a.stride(1);
  kargs.stride_ws = N;
  kargs.stride_a_batch = (int)O.stride(1);
  kargs.stride_b_batch = (int)wo_a.stride(0);
  kargs.stride_ws_batch = M * N;
  kargs.ptr_sfa = x_scale.data_ptr();
  kargs.ptr_sfb = w_scale.data_ptr();
  kargs.stride_sfa = (int)x_scale.stride(0);
  kargs.stride_sfa_batch = (int)x_scale.stride(1);
  kargs.stride_sfb = (int)w_scale.stride(1);
  kargs.stride_sfb_batch = (int)w_scale.stride(0);
  kargs.ptr_c = Y.data_ptr();
  kargs.stride_c = (int)Y.stride(0);
  kargs.stride_c_batch = (int)Y.stride(1);

  const int num_tiles_m = M / Traits::B_M;
  const int num_tiles_n = N / LOGICAL_B_N;
  dim3 grid_main(num_tiles_m * num_tiles_n, 1, batch);
  dim3 block_main(512);
  if (Y.dtype() == AITER_DTYPE_bf16) {
    gemm_a8w8_mxscale_flatmm_splitk_wave8n2_kernel<Traits, __bf16>
        <<<grid_main, block_main, 0, stream>>>(kargs);
  } else {
    gemm_a8w8_mxscale_flatmm_splitk_wave8n2_kernel<Traits, float>
        <<<grid_main, block_main, 0, stream>>>(kargs);
  }
#else
  AITER_CHECK(false, who, " requires OPUS_BUILD_HAS_GFX950");
#endif
}

template <typename Traits, bool ISSUE_NEXT_BEFORE_SCALE = false,
          bool SKIP_SCALE_WAIT = false, bool SINGLE_LDS_SLOT = false,
          bool ISSUE_NEXT_AFTER_MMA = false, bool PACK_SCALE_ON_DEMAND = false>
static void opus_bmm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_mmajor_impl(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    int splitK,
    const char* who)
{
  opus_bmm_a8w8_common_checks(O, wo_a, Y, who);
#ifdef OPUS_BUILD_HAS_GFX950
  const auto &arch_info = opus_get_arch_info();
  AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950,
              who, " is gfx950-only; current device ",
              arch_info.dev, " has gcnArchName='", arch_info.name, "'");
  AITER_CHECK(splitK == 1, who, " requires splitK == 1");

  const int M = O.size(0);
  const int batch = O.size(1);
  const int N = wo_a.size(1);
  const int K = O.size(2);
  constexpr int LOGICAL_B_N = Traits::B_N * 2;
  AITER_CHECK(M % Traits::B_M == 0,
              who, " requires M % ", Traits::B_M, " == 0, got ", M);
  AITER_CHECK(N % LOGICAL_B_N == 0,
              who, " requires N % ", LOGICAL_B_N, " == 0, got ", N);
  AITER_CHECK(K % Traits::B_K == 0,
              who, " requires K % ", Traits::B_K, " == 0, got ", K);

  auto stream = aiter::getCurrentHIPStream();

  opus_gemm_scale_splitk_kargs_gfx950 kargs{};
  kargs.ptr_a = O.data_ptr();
  kargs.ptr_b = wo_a.data_ptr();
  kargs.ws_handle = nullptr;
  kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = batch;
  kargs.split_k = 1;
  kargs.stride_a = (int)O.stride(0);
  kargs.stride_b = (int)wo_a.stride(1);
  kargs.stride_ws = N;
  kargs.stride_a_batch = (int)O.stride(1);
  kargs.stride_b_batch = (int)wo_a.stride(0);
  kargs.stride_ws_batch = M * N;
  kargs.ptr_sfa = x_scale.data_ptr();
  kargs.ptr_sfb = w_scale.data_ptr();
  kargs.stride_sfa = (int)x_scale.stride(0);
  kargs.stride_sfa_batch = (int)x_scale.stride(1);
  kargs.stride_sfb = (int)w_scale.stride(1);
  kargs.stride_sfb_batch = (int)w_scale.stride(0);
  kargs.ptr_c = Y.data_ptr();
  kargs.stride_c = (int)Y.stride(0);
  kargs.stride_c_batch = (int)Y.stride(1);

  const int num_tiles_m = M / Traits::B_M;
  const int num_tiles_n = N / LOGICAL_B_N;
  dim3 grid_main(num_tiles_m * num_tiles_n, 1, batch);
  dim3 block_main(256);
  if (Y.dtype() == AITER_DTYPE_bf16) {
    gemm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_kernel<
        Traits, __bf16, ISSUE_NEXT_BEFORE_SCALE, SKIP_SCALE_WAIT, SINGLE_LDS_SLOT,
        ISSUE_NEXT_AFTER_MMA, PACK_SCALE_ON_DEMAND>
        <<<grid_main, block_main, 0, stream>>>(kargs);
  } else {
    gemm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_kernel<
        Traits, float, ISSUE_NEXT_BEFORE_SCALE, SKIP_SCALE_WAIT, SINGLE_LDS_SLOT,
        ISSUE_NEXT_AFTER_MMA, PACK_SCALE_ON_DEMAND>
        <<<grid_main, block_main, 0, stream>>>(kargs);
  }
#else
  AITER_CHECK(false, who, " requires OPUS_BUILD_HAS_GFX950");
#endif
}

template <typename Traits, bool SKIP_SCALE_WAIT = false,
          bool PACK_SCALE_ON_DEMAND = false>
static void opus_bmm_a8w8_mxscale_flatmm_splitk_wave4m2_selfload_mmajor_impl(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    int splitK,
    const char* who)
{
  opus_bmm_a8w8_common_checks(O, wo_a, Y, who);
#ifdef OPUS_BUILD_HAS_GFX950
  const auto &arch_info = opus_get_arch_info();
  AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950,
              who, " is gfx950-only; current device ",
              arch_info.dev, " has gcnArchName='", arch_info.name, "'");
  AITER_CHECK(splitK == 1, who, " requires splitK == 1");

  const int M = O.size(0);
  const int batch = O.size(1);
  const int N = wo_a.size(1);
  const int K = O.size(2);
  constexpr int LOGICAL_B_M = Traits::B_M * 2;
  AITER_CHECK(M % LOGICAL_B_M == 0,
              who, " requires M % ", LOGICAL_B_M, " == 0, got ", M);
  AITER_CHECK(N % Traits::B_N == 0,
              who, " requires N % ", Traits::B_N, " == 0, got ", N);
  AITER_CHECK(K % Traits::B_K == 0,
              who, " requires K % ", Traits::B_K, " == 0, got ", K);

  auto stream = aiter::getCurrentHIPStream();

  opus_gemm_scale_splitk_kargs_gfx950 kargs{};
  kargs.ptr_a = O.data_ptr();
  kargs.ptr_b = wo_a.data_ptr();
  kargs.ws_handle = nullptr;
  kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = batch;
  kargs.split_k = 1;
  kargs.stride_a = (int)O.stride(0);
  kargs.stride_b = (int)wo_a.stride(1);
  kargs.stride_ws = N;
  kargs.stride_a_batch = (int)O.stride(1);
  kargs.stride_b_batch = (int)wo_a.stride(0);
  kargs.stride_ws_batch = M * N;
  kargs.ptr_sfa = x_scale.data_ptr();
  kargs.ptr_sfb = w_scale.data_ptr();
  kargs.stride_sfa = (int)x_scale.stride(0);
  kargs.stride_sfa_batch = (int)x_scale.stride(1);
  kargs.stride_sfb = (int)w_scale.stride(1);
  kargs.stride_sfb_batch = (int)w_scale.stride(0);
  kargs.ptr_c = Y.data_ptr();
  kargs.stride_c = (int)Y.stride(0);
  kargs.stride_c_batch = (int)Y.stride(1);

  const int num_tiles_m = M / LOGICAL_B_M;
  const int num_tiles_n = N / Traits::B_N;
  dim3 grid_main(num_tiles_m * num_tiles_n, 1, batch);
  dim3 block_main(256);
  if (Y.dtype() == AITER_DTYPE_bf16) {
    gemm_a8w8_mxscale_flatmm_splitk_wave4m2_selfload_kernel<
        Traits, __bf16, SKIP_SCALE_WAIT, PACK_SCALE_ON_DEMAND>
        <<<grid_main, block_main, 0, stream>>>(kargs);
  } else {
    gemm_a8w8_mxscale_flatmm_splitk_wave4m2_selfload_kernel<
        Traits, float, SKIP_SCALE_WAIT, PACK_SCALE_ON_DEMAND>
        <<<grid_main, block_main, 0, stream>>>(kargs);
  }
#else
  AITER_CHECK(false, who, " requires OPUS_BUILD_HAS_GFX950");
#endif
}

template <typename Bf16Traits, typename Fp32Traits, bool K1024_ONLY = false, bool K1024_LB1 = false>
static void opus_bmm_a8w8_mxscale_scale_pipeline_mmajor_impl(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    int splitK,
    const char* who)
{
  opus_bmm_a8w8_common_checks(O, wo_a, Y, who);
#ifdef OPUS_BUILD_HAS_GFX950
  const auto &arch_info = opus_get_arch_info();
  AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950,
              who, " is gfx950-only; current device ",
              arch_info.dev, " has gcnArchName='", arch_info.name, "'");
  AITER_CHECK(splitK == 1, who, " requires splitK == 1");

  const int M = O.size(0);
  const int batch = O.size(1);
  const int N = wo_a.size(1);
  const int K = O.size(2);
  AITER_CHECK(M % Bf16Traits::B_M == 0,
              who, " requires M % ", Bf16Traits::B_M, " == 0, got ", M);
  AITER_CHECK(N % Bf16Traits::B_N == 0,
              who, " requires N % ", Bf16Traits::B_N, " == 0, got ", N);
  AITER_CHECK(K % Bf16Traits::B_K == 0,
              who, " requires K % ", Bf16Traits::B_K, " == 0, got ", K);
  if constexpr (K1024_ONLY || K1024_LB1) {
    AITER_CHECK(K == 1024, who, " requires K == 1024, got ", K);
  }

  opus_gemm_scale_kargs_gfx950 kargs{};
  kargs.ptr_a = O.data_ptr();
  kargs.ptr_b = wo_a.data_ptr();
  kargs.ptr_c = Y.data_ptr();
  kargs.m = M;
  kargs.n = N;
  kargs.k = K;
  kargs.batch = batch;
  kargs.stride_a = (int)O.stride(0);
  kargs.stride_b = (int)wo_a.stride(1);
  kargs.stride_c = (int)Y.stride(0);
  kargs.stride_a_batch = (int)O.stride(1);
  kargs.stride_b_batch = (int)wo_a.stride(0);
  kargs.stride_c_batch = (int)Y.stride(1);
  kargs.ptr_sfa = x_scale.data_ptr();
  kargs.ptr_sfb = w_scale.data_ptr();
  kargs.stride_sfa = (int)x_scale.stride(0);
  kargs.stride_sfa_batch = (int)x_scale.stride(1);
  kargs.stride_sfb = (int)w_scale.stride(1);
  kargs.stride_sfb_batch = (int)w_scale.stride(0);

  const int num_tiles_m = M / Bf16Traits::B_M;
  const int num_tiles_n = N / Bf16Traits::B_N;
  dim3 grid_main(num_tiles_m * num_tiles_n, 1, batch);
  dim3 block_main(Bf16Traits::BLOCK_SIZE);
  auto stream = aiter::getCurrentHIPStream();
  if (Y.dtype() == AITER_DTYPE_bf16) {
    if constexpr (K1024_LB1) {
      gemm_a8w8_scale_k1024_lb1_kernel<Bf16Traits><<<grid_main, block_main, 0, stream>>>(kargs);
    } else if constexpr (K1024_ONLY) {
      gemm_a8w8_scale_k1024_kernel<Bf16Traits><<<grid_main, block_main, 0, stream>>>(kargs);
    } else {
      gemm_a8w8_scale_kernel<Bf16Traits><<<grid_main, block_main, 0, stream>>>(kargs);
    }
  } else {
    if constexpr (K1024_LB1) {
      gemm_a8w8_scale_k1024_lb1_kernel<Fp32Traits><<<grid_main, block_main, 0, stream>>>(kargs);
    } else if constexpr (K1024_ONLY) {
      gemm_a8w8_scale_k1024_kernel<Fp32Traits><<<grid_main, block_main, 0, stream>>>(kargs);
    } else {
      gemm_a8w8_scale_kernel<Fp32Traits><<<grid_main, block_main, 0, stream>>>(kargs);
    }
  }
#else
  AITER_CHECK(false, who, " requires OPUS_BUILD_HAS_GFX950");
#endif
}

void opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    int splitK,
    int kernelId)
{
  switch (kernelId) {
    case 320:
      opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor_impl<
          opus_bmm_a8w8_mxscale_flatmm_m64n32k256_splitk_traits_gfx950>(
          O, wo_a, Y, x_scale, w_scale, splitK,
          false,
          "opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor(m64n32k256)");
      break;
    case 640:
      opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor_impl<
          opus_bmm_a8w8_mxscale_flatmm_m32n64k256_splitk_traits_gfx950>(
          O, wo_a, Y, x_scale, w_scale, splitK,
          false,
          "opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor(m32n64k256)");
      break;
    case 646:
      opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor_impl<
          opus_bmm_a8w8_mxscale_flatmm_m32n64k256_splitk_traits_gfx950, true>(
          O, wo_a, Y, x_scale, w_scale, splitK,
          false,
          "opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor(m32n64k256_selfload)");
      break;
    case 650:
      opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor_impl<
          opus_bmm_a8w8_mxscale_flatmm_m64n64k128_splitk_traits_gfx950>(
          O, wo_a, Y, x_scale, w_scale, splitK,
          false,
          "opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor(m64n64k128)");
      break;
    case 653:
      opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor_impl<
          opus_bmm_a8w8_mxscale_flatmm_m64n64k128_splitk_traits_gfx950, false, true>(
          O, wo_a, Y, x_scale, w_scale, splitK,
          false,
          "opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor(m64n64k128_scale_prefetch)");
      break;
    case 128:
      opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor_impl<
          opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950>(
          O, wo_a, Y, x_scale, w_scale, splitK,
          false,
          "opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor(m128n128k128_wg1)");
      break;
    case 138:
      opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor_impl<
          opus_bmm_a8w8_mxscale_flatmm_m64n128k256_wg1_splitk_traits_gfx950>(
          O, wo_a, Y, x_scale, w_scale, splitK,
          false,
          "opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor(m64n128k256_wg1)");
      break;
    case 139:
      opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor_impl<
          opus_bmm_a8w8_mxscale_flatmm_m128n64k256_wg1_splitk_traits_gfx950>(
          O, wo_a, Y, x_scale, w_scale, splitK,
          false,
          "opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor(m128n64k256_wg1)");
      break;
    case 137:
      opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor_impl<
          opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, false, true>(
          O, wo_a, Y, x_scale, w_scale, splitK,
          false,
          "opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor(m128n128k128_scale_prefetch)");
      break;
    case 129:
      opus_bmm_a8w8_mxscale_flatmm_splitk_nphase_mmajor_impl<
          opus_bmm_a8w8_mxscale_flatmm64_splitk_traits_gfx950, 2>(
          O, wo_a, Y, x_scale, w_scale, splitK,
          "opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor(m64n256k128_nphase)");
      break;
    case 131:
      opus_bmm_a8w8_mxscale_flatmm_splitk_mouter_mmajor_impl<
          opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950>(
          O, wo_a, Y, x_scale, w_scale, splitK,
          "opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor(m128n128k128_persistent_mouter_wg1)");
      break;
    case 144:
      opus_bmm_a8w8_mxscale_flatmm_splitk_mouter_mmajor_impl<
          opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, true>(
          O, wo_a, Y, x_scale, w_scale, splitK,
          "opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor(m128n128k128_persistent_mouter_wg1_skip_scale_wait)");
      break;
    case 132:
      opus_bmm_a8w8_mxscale_flatmm_splitk_wave8n2_mmajor_impl<
          opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950>(
          O, wo_a, Y, x_scale, w_scale, splitK,
          "opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor(m128n256k128_wave8n2)");
      break;
    case 133:
      opus_bmm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_mmajor_impl<
          opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950>(
          O, wo_a, Y, x_scale, w_scale, splitK,
          "opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor(m128n256k128_wave4n2_selfload)");
      break;
    case 140:
      opus_bmm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_mmajor_impl<
          opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, true>(
          O, wo_a, Y, x_scale, w_scale, splitK,
          "opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor(m128n256k128_wave4n2_selfload_issue_next)");
      break;
    case 141:
      opus_bmm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_mmajor_impl<
          opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, false, true>(
          O, wo_a, Y, x_scale, w_scale, splitK,
          "opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor(m128n256k128_wave4n2_selfload_skip_scale_wait)");
      break;
    case 145:
      opus_bmm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_mmajor_impl<
          opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, false, true, true>(
          O, wo_a, Y, x_scale, w_scale, splitK,
          "opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor(m128n256k128_wave4n2_selfload_single_lds)");
      break;
    case 146:
      opus_bmm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_mmajor_impl<
          opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, false, true, false, true>(
          O, wo_a, Y, x_scale, w_scale, splitK,
          "opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor(m128n256k128_wave4n2_selfload_issue_after_mma)");
      break;
    case 147:
      opus_bmm_a8w8_mxscale_flatmm_splitk_wave4n2_selfload_mmajor_impl<
          opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, false, true, false, false, true>(
          O, wo_a, Y, x_scale, w_scale, splitK,
          "opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor(m128n256k128_wave4n2_selfload_on_demand_scale_pack)");
      break;
    case 134:
      opus_bmm_a8w8_mxscale_flatmm_splitk_wave4m2_selfload_mmajor_impl<
          opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950>(
          O, wo_a, Y, x_scale, w_scale, splitK,
          "opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor(m256n128k128_wave4m2_selfload)");
      break;
    case 142:
      opus_bmm_a8w8_mxscale_flatmm_splitk_wave4m2_selfload_mmajor_impl<
          opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, true>(
          O, wo_a, Y, x_scale, w_scale, splitK,
          "opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor(m256n128k128_wave4m2_selfload_skip_scale_wait)");
      break;
    case 148:
      opus_bmm_a8w8_mxscale_flatmm_splitk_wave4m2_selfload_mmajor_impl<
          opus_bmm_a8w8_mxscale_flatmm_m128n128k128_wg1_splitk_traits_gfx950, true, true>(
          O, wo_a, Y, x_scale, w_scale, splitK,
          "opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor(m256n128k128_wave4m2_selfload_on_demand_scale_pack)");
      break;
    case 149:
      AITER_CHECK(splitK == 1,
                  "opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor(m512n256k256_scale_pipeline) requires splitK == 1");
      if (Y.dtype() == AITER_DTYPE_bf16) {
        opus_gemm_a8w8_mxscale_512x128x256x128_4x2_16x16x128_1x128x128_mmajor<bf16_t>(
            O, wo_a, Y, x_scale, w_scale);
      } else {
        opus_gemm_a8w8_mxscale_512x128x256x128_4x2_16x16x128_1x128x128_mmajor<fp32_t>(
            O, wo_a, Y, x_scale, w_scale);
      }
      break;
    case 150:
      opus_bmm_a8w8_mxscale_scale_pipeline_mmajor_impl<
          opus_bmm_a8w8_mxscale_scale_m256n256k128_bf16_traits_gfx950,
          opus_bmm_a8w8_mxscale_scale_m256n256k128_fp32_traits_gfx950>(
          O, wo_a, Y, x_scale, w_scale, splitK,
          "opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor(m256n256k128_scale_pipeline)");
      break;
    case 151:
      opus_bmm_a8w8_mxscale_scale_pipeline_mmajor_impl<
          opus_bmm_a8w8_mxscale_scale_m256n256k128_bf16_traits_gfx950,
          opus_bmm_a8w8_mxscale_scale_m256n256k128_fp32_traits_gfx950,
          true>(
          O, wo_a, Y, x_scale, w_scale, splitK,
          "opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor(m256n256k1024_scale_pipeline)");
      break;
    case 152:
      opus_bmm_a8w8_mxscale_scale_pipeline_mmajor_impl<
          opus_bmm_a8w8_mxscale_scale_m256n256k128_bf16_traits_gfx950,
          opus_bmm_a8w8_mxscale_scale_m256n256k128_fp32_traits_gfx950,
          false,
          true>(
          O, wo_a, Y, x_scale, w_scale, splitK,
          "opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor(m256n256k1024_scale_pipeline_lb1)");
      break;
    case 322:
      opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor_impl<
          opus_bmm_a8w8_mxscale_flatmm_m64n32k256_wg1_splitk_traits_gfx950>(
          O, wo_a, Y, x_scale, w_scale, splitK,
          false,
          "opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor(m64n32k256_wg1)");
      break;
    case 642:
      opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor_impl<
          opus_bmm_a8w8_mxscale_flatmm_m32n64k256_wg1_splitk_traits_gfx950>(
          O, wo_a, Y, x_scale, w_scale, splitK,
          false,
          "opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor(m32n64k256_wg1)");
      break;
    case 256:
      opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor_impl<
          opus_bmm_a8w8_mxscale_flatmm256_splitk_traits_gfx950>(
          O, wo_a, Y, x_scale, w_scale, splitK,
          false,
          "opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor(n256)");
      break;
    case 64:
      opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor_impl<
          opus_bmm_a8w8_mxscale_flatmm64_splitk_traits_gfx950>(
          O, wo_a, Y, x_scale, w_scale, splitK,
          false,
          "opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor(k64)");
      break;
    case 100:
      opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor_impl<
          opus_bmm_a8w8_mxscale_flatmm_splitk_traits_gfx950>(
          O, wo_a, Y, x_scale, w_scale, splitK,
          true,
          "opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor(fused)");
      break;
    case 0:
    case 32:
    default:
      opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor_impl<
          opus_bmm_a8w8_mxscale_flatmm_splitk_traits_gfx950>(
          O, wo_a, Y, x_scale, w_scale, splitK,
          false,
          "opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor(k32)");
      break;
  }
}

// ── opus_bmm_a8w8_uniform_scale(_mmajor)() — fp8 block-scale uniform BMM ──────────
// fp8 Route-B low-barrier variant (4-wave full-tile, direct store). Two layout
// surfaces share the same kernel/tiles:
//   * batch-major: O=[batch,M,K], wo_a=[batch,N,K], Y=[batch,M,N].
//   * mmajor: O=[M,batch,K], wo_a=[batch,N,K], Y=[M,batch,N].
// kernelId selects the tile (700=128x128, 701=256x128); Y dtype in {fp32,bf16}.
#ifdef OPUS_BUILD_HAS_GFX950
template <typename D_C>
void opus_gemm_a8w8_uniform_scale_256x128x128x128_2x2_16x16x128_1x128x128(
    aiter_tensor_t &, aiter_tensor_t &, aiter_tensor_t &,
    std::optional<aiter_tensor_t>, std::optional<aiter_tensor_t>);
template <typename D_C>
void opus_gemm_a8w8_uniform_scale_256x256x128x128_2x2_16x16x128_1x128x128(
    aiter_tensor_t &, aiter_tensor_t &, aiter_tensor_t &,
    std::optional<aiter_tensor_t>, std::optional<aiter_tensor_t>);
template <typename D_C>
void opus_gemm_a8w8_uniform_scale_256x128x128x128_2x2_16x16x128_1x128x128_mmajor(
    aiter_tensor_t &, aiter_tensor_t &, aiter_tensor_t &,
    std::optional<aiter_tensor_t>, std::optional<aiter_tensor_t>);
template <typename D_C>
void opus_gemm_a8w8_uniform_scale_256x256x128x128_2x2_16x16x128_1x128x128_mmajor(
    aiter_tensor_t &, aiter_tensor_t &, aiter_tensor_t &,
    std::optional<aiter_tensor_t>, std::optional<aiter_tensor_t>);

template <bool MMAJOR>
static void opus_uniform_scale_dispatch(int kernelId, aiter_tensor_t &O,
                                        aiter_tensor_t &wo_a, aiter_tensor_t &Y,
                                        aiter_tensor_t &x_scale,
                                        aiter_tensor_t &w_scale)
{
  const bool y_bf16 = (Y.dtype() == AITER_DTYPE_bf16);
  switch (kernelId)
  {
    case 701:
      if constexpr (MMAJOR) {
        if (y_bf16)
          opus_gemm_a8w8_uniform_scale_256x256x128x128_2x2_16x16x128_1x128x128_mmajor<bf16_t>(O, wo_a, Y, x_scale, w_scale);
        else
          opus_gemm_a8w8_uniform_scale_256x256x128x128_2x2_16x16x128_1x128x128_mmajor<fp32_t>(O, wo_a, Y, x_scale, w_scale);
      } else {
        if (y_bf16)
          opus_gemm_a8w8_uniform_scale_256x256x128x128_2x2_16x16x128_1x128x128<bf16_t>(O, wo_a, Y, x_scale, w_scale);
        else
          opus_gemm_a8w8_uniform_scale_256x256x128x128_2x2_16x16x128_1x128x128<fp32_t>(O, wo_a, Y, x_scale, w_scale);
      }
      break;
    case 700:
    default:
      if constexpr (MMAJOR) {
        if (y_bf16)
          opus_gemm_a8w8_uniform_scale_256x128x128x128_2x2_16x16x128_1x128x128_mmajor<bf16_t>(O, wo_a, Y, x_scale, w_scale);
        else
          opus_gemm_a8w8_uniform_scale_256x128x128x128_2x2_16x16x128_1x128x128_mmajor<fp32_t>(O, wo_a, Y, x_scale, w_scale);
      } else {
        if (y_bf16)
          opus_gemm_a8w8_uniform_scale_256x128x128x128_2x2_16x16x128_1x128x128<bf16_t>(O, wo_a, Y, x_scale, w_scale);
        else
          opus_gemm_a8w8_uniform_scale_256x128x128x128_2x2_16x16x128_1x128x128<fp32_t>(O, wo_a, Y, x_scale, w_scale);
      }
      break;
  }
}
#endif

static void opus_uniform_scale_common_checks(aiter_tensor_t &O, aiter_tensor_t &wo_a,
                                             aiter_tensor_t &Y, const char *who)
{
  aiter_detail::g_aiter_can_throw = true;
  AITER_CHECK(O.dim() == 3 && wo_a.dim() == 3 && Y.dim() == 3,
              who, ": O/wo_a/Y must be 3D");
  AITER_CHECK(O.dtype() == AITER_DTYPE_fp8 && wo_a.dtype() == AITER_DTYPE_fp8,
              who, ": O and wo_a must be fp8");
  AITER_CHECK(Y.dtype() == AITER_DTYPE_fp32 || Y.dtype() == AITER_DTYPE_bf16,
              who, ": Y must be fp32 or bf16");
}

void opus_bmm_a8w8_uniform_scale(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    int kernelId)
{
  opus_uniform_scale_common_checks(O, wo_a, Y, "opus_bmm_a8w8_uniform_scale");
#ifdef OPUS_BUILD_HAS_GFX950
  const auto &arch_info = opus_get_arch_info();
  AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950,
              "opus_bmm_a8w8_uniform_scale is gfx950-only; current device ",
              arch_info.dev, " has gcnArchName='", arch_info.name, "'");
  opus_uniform_scale_dispatch<false>(kernelId, O, wo_a, Y, x_scale, w_scale);
#else
  AITER_CHECK(false, "opus_bmm_a8w8_uniform_scale requires OPUS_BUILD_HAS_GFX950");
#endif
}

void opus_bmm_a8w8_uniform_scale_mmajor(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    int kernelId)
{
  opus_uniform_scale_common_checks(O, wo_a, Y, "opus_bmm_a8w8_uniform_scale_mmajor");
#ifdef OPUS_BUILD_HAS_GFX950
  const auto &arch_info = opus_get_arch_info();
  AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950,
              "opus_bmm_a8w8_uniform_scale_mmajor is gfx950-only; current device ",
              arch_info.dev, " has gcnArchName='", arch_info.name, "'");
  opus_uniform_scale_dispatch<true>(kernelId, O, wo_a, Y, x_scale, w_scale);
#else
  AITER_CHECK(false,
              "opus_bmm_a8w8_uniform_scale_mmajor requires OPUS_BUILD_HAS_GFX950");
#endif
}

#endif  // !__HIP_DEVICE_COMPILE__
