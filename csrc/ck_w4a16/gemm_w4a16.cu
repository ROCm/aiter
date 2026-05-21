// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// AIESW-32176: CK WMMA W4A16 b_scale GEMM dispatcher (gfx1151 / RDNA 3.5).
//
// Single torch op `gemm_w4a16` covers both the symmetric (uint4b8) and
// asymmetric (AWQ per-group zero-point) variants — pass `scaled_zp=None` for
// symmetric, pass `(zp - 8) * scale` precomputed at weight load for
// asymmetric. The asymmetric path requires the CK PR adding the
// per-group zero-point machinery to wmma_cshuffle_v3_b_scale (see
// ROCm/rocm-libraries `[CK] Add per-group zero-point support to
// wmma_cshuffle_v3_b_scale (asymmetric int4)`).
//
// Tuned for the four Qwen3-4B prefill linear columns. Same kernel handles all
// at runtime; M is dynamic. K must be a multiple of KPerBlock (32) and of the
// active ScaleBlockK (32 or 128 — selected at runtime from `group_size`).
// Output dtype dispatches between fp16 and bf16.

#include "gemm_w4a16_common.cuh"

namespace {

using namespace ck_w4a16;

// Hot-path build of the CK Argument struct + invoker call. Templated on T
// (fp16 or bf16), ScaleBlockK (32 or 128 — AWQ group_size),
// PreDequantToLDS (false = fused-dequant baseline, true = pre-dequant-to-LDS
// variant — see DeviceGemmInstance docs in gemm_w4a16_common.cuh), and
// TruncateBf16Round (false = IEEE round-to-nearest-even, true = bit-cast
// truncate). `p_b_zero_point` is nullptr for the symmetric path.
// Argument validation (dtype / shape / contiguity) is done by the
// dispatcher one level up before we get here.
template <typename T, ck::index_t ScaleBlockK, bool PreDequantToLDS,
          bool TruncateBf16Round>
inline void run_kernel_inner(const at::Tensor& in_a,
                             const at::Tensor& in_b,
                             const at::Tensor& in_s,
                             at::Tensor& out,
                             int64_t group_size,
                             const T* p_b_zero_point) {
  // TODO(AIESW-32282): drop this guard once the pre-dequant-to-LDS pipeline
  // is implemented. The template surface is wired end-to-end (4x dtypes/G
  // pairs x 2 PreDequantToLDS flavors x 2 TruncateBf16Round flavors = 16
  // instantiations), so the test + dispatcher can route here even though
  // the kernel body for PreDequantToLDS=true is currently the same as the
  // false specialization (see the DeviceGemmInstanceImpl<..., true> stub
  // in gemm_w4a16_common.cuh).
  if constexpr (PreDequantToLDS) {
    TORCH_CHECK(false,
                "CK W4A16 b_scale GEMM: PreDequantToLDS=true is not yet "
                "implemented (template hook surfaced, kernel body pending — "
                "see TODO(AIESW-32282) in gemm_w4a16_common.cuh).");
  }

  // AIESW-32282: TruncateBf16Round is now a true runtime template flag. Both
  // truncate / RTE flavors instantiate as separate template-mangled symbols
  // in the same .so. The fp16 path silently ignores the flag (fp16 has no
  // rounding chain to skip; i4_to_half4_scale is already the optimal
  // bit-trick path — DequantPack8WithZpTruncate's fp16 overload delegates
  // to DequantPack8WithZp).

  const ck::index_t M = static_cast<ck::index_t>(in_a.size(0));
  const ck::index_t K = static_cast<ck::index_t>(in_a.size(1));
  const ck::index_t N = static_cast<ck::index_t>(in_b.size(1));
  const ck::index_t StrideA = K;
  const ck::index_t StrideB = K;
  const ck::index_t StrideC = N;
  const ck::index_t Scale_Stride_BN =
      static_cast<ck::index_t>(K / group_size);
  const ck::index_t KBatch = 1;

#ifdef USE_ROCM
  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA guard(device_of(in_a));
#else
  const at::cuda::OptionalCUDAGuard guard(device_of(in_a));
#endif

  auto gemm =
      DeviceGemmInstance<T, ScaleBlockK, PreDequantToLDS, TruncateBf16Round>{};
  auto invoker = gemm.MakeInvoker();
  auto argument = gemm.MakeArgument(
      reinterpret_cast<const T*>(in_a.data_ptr()),
      reinterpret_cast<const BDataType*>(in_b.data_ptr()),
      reinterpret_cast<T*>(out.data_ptr()), M, N, K, StrideA, StrideB, StrideC,
      Scale_Stride_BN, reinterpret_cast<const T*>(in_s.data_ptr()), KBatch,
      PassThrough{}, PassThrough{}, PassThrough{}, p_b_zero_point);

  // IsSupportedArgument check is debug-only — measured ~840 us/call overhead
  // when run on every dispatch and the device-op rejects shapes only on
  // misuse (group_size != ScaleBlockK, K not divisible by KPerBlock, etc.)
  // which are already caught by the TORCH_CHECKs in the dispatcher above.
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      gemm.IsSupportedArgument(argument),
      "CK W4A16 b_scale device op rejected shape (M=", M, ", N=", N,
      ", K=", K, ", G=", group_size, ")");

#ifdef USE_ROCM
  invoker.Run(argument, StreamConfig{at::hip::getCurrentHIPStream()});
#else
  StreamConfig stream;
  stream.stream_id_ = at::cuda::getCurrentCUDAStream();
  invoker.Run(argument, stream);
#endif
}

// Runtime dispatch on group_size + PreDequantToLDS + TruncateBf16Round
// into the matching template instantiation. Only group_size in {32, 128}
// are wired today (the two AWQ group sizes shipped by the models we target
// on gfx1151); the dispatcher one level up gates `group_size` to those two
// before we ever get here. The two bool flags multiply the instantiation
// count to 8 per dtype (2 group sizes x 2 PreDequantToLDS x 2
// TruncateBf16Round); kernel build time scales accordingly. The
// PreDequantToLDS=true path currently TORCH_CHECKs at runtime (see
// TODO(AIESW-32282)). The TruncateBf16Round flag is asserted against the
// build-time mode (see run_kernel_inner; build-time switch because the CK
// threadwise transfer hardcodes DequantPack8 / DequantPack8WithZp).
template <typename T>
inline void run_kernel(const at::Tensor& in_a,
                       const at::Tensor& in_b,
                       const at::Tensor& in_s,
                       at::Tensor& out,
                       int64_t group_size,
                       bool pre_dequant_to_lds,
                       bool truncate_bf16_round,
                       const T* p_b_zero_point) {
  // 2x2x2 dispatch: group_size x pre_dequant_to_lds x truncate_bf16_round.
  auto dispatch_g = [&](auto g_const, auto pdl_const, auto trunc_const) {
    constexpr ck::index_t G = decltype(g_const)::value;
    constexpr bool PDL      = decltype(pdl_const)::value;
    constexpr bool TBT      = decltype(trunc_const)::value;
    run_kernel_inner<T, G, PDL, TBT>(in_a, in_b, in_s, out, group_size,
                                     p_b_zero_point);
  };

#define _CK_W4A16_DISPATCH_G(G_VAL)                                            \
  do {                                                                         \
    if (pre_dequant_to_lds && truncate_bf16_round) {                           \
      dispatch_g(std::integral_constant<ck::index_t, G_VAL>{},                 \
                 std::true_type{}, std::true_type{});                          \
    } else if (pre_dequant_to_lds) {                                           \
      dispatch_g(std::integral_constant<ck::index_t, G_VAL>{},                 \
                 std::true_type{}, std::false_type{});                         \
    } else if (truncate_bf16_round) {                                          \
      dispatch_g(std::integral_constant<ck::index_t, G_VAL>{},                 \
                 std::false_type{}, std::true_type{});                         \
    } else {                                                                   \
      dispatch_g(std::integral_constant<ck::index_t, G_VAL>{},                 \
                 std::false_type{}, std::false_type{});                        \
    }                                                                          \
  } while (0)

  if (group_size == 128) {
    _CK_W4A16_DISPATCH_G(128);
  } else if (group_size == 32) {
    _CK_W4A16_DISPATCH_G(32);
  } else {
    TORCH_CHECK(false,
                "CK W4A16 b_scale GEMM: unsupported group_size=", group_size,
                " (only 32 and 128 are wired)");
  }
#undef _CK_W4A16_DISPATCH_G
}

// Type-erase the optional zero-point pointer for the dispatcher.
template <typename T>
inline const T* zp_ptr(const std::optional<at::Tensor>& scaled_zp) {
  return scaled_zp.has_value()
             ? reinterpret_cast<const T*>(scaled_zp.value().data_ptr())
             : nullptr;
}

}  // namespace

// gemm_w4a16
//
// in_a:        [M, K]            fp16 or bf16, contiguous (row-major).
// in_b:        [K0, N, K1/2]     int8 in CK pk_i4_v3 b_scale layout
//                                (K0 = K/KPerBlock, K1 = KPerBlock).
// in_s:        [N, K/G]          activation-dtype scales, contiguous
//                                row-major. CK calls this `b1_k_n` shape
//                                [K/G, N] with stride (K/G, 1) — exactly a
//                                view over [N, K/G] row-major; no transpose.
// Y:           [M, N]            activation-dtype, caller-allocated output.
//                                Output dtype determines the kernel template
//                                instantiation (F16 vs B16).
// group_size:  AWQ per-group quantization granularity. Must be one of {32, 128}
//              — these select the matching ScaleBlockK template instantiation
//              of the CK kernel. K must be divisible by group_size.
// scaled_zp:   optional [N, K/G] activation-dtype. None for symmetric
//              (uint4b8); set to (zp - 8) * scale precomputed at weight load
//              for asymmetric (AWQ). The kernel uses the algebraic identity
//                  (nibble - zp) * scale
//                    == (nibble - 8) * scale - (zp - 8) * scale
//              so the asymmetric path costs one extra activation-dtype vector
//              subtract per dequant pack vs the symmetric path.
// pre_dequant_to_lds:
//              optional bool, defaults to false. False selects the existing
//              fused-dequant baseline (CK dequants int4 inside the WMMA
//              inner loop). True selects the pre-dequant-to-LDS variant
//              (dequant once per K-block into bf16/fp16 in LDS scratch,
//              WMMA reads activation-dtype B from LDS). The true path is
//              currently STUBBED and will TORCH_CHECK at runtime — see
//              TODO(AIESW-32282) in include/gemm_w4a16_common.cuh.
// truncate_bf16_round:
//              optional bool, defaults to false. False = IEEE round-to-
//              nearest-even (the existing path). True = bit-cast truncate
//              for the trailing fp32->bf16 step in the bf16 dequant; saves
//              ~3 RDNA3.5 VALU instructions per nibble (v_add3_u32 +0x7fff
//              round bias, v_cmp_o_f32 + v_cndmask_b16 0x7fc0 NaN-quietening
//              chain) at <0.5 ULP of bf16 error. Silently ignored on the
//              fp16 path (fp16 already uses the optimal bit-trick). True
//              runtime switch — both flavors share the .so as distinct
//              template-mangled symbols (AIESW-32282: BElementwiseOperation
//              now threads through the CK threadwise transfer; see
//              gemm_w4a16_common.cuh).
torch::Tensor gemm_w4a16(at::Tensor& in_a,
                         at::Tensor& in_b,
                         at::Tensor& in_s,
                         at::Tensor& Y,
                         int64_t group_size,
                         std::optional<at::Tensor> scaled_zp,
                         std::optional<bool> pre_dequant_to_lds,
                         std::optional<bool> truncate_bf16_round) {
  TORCH_CHECK(in_a.is_cuda() && in_b.is_cuda() && in_s.is_cuda() &&
                  Y.is_cuda(),
              "All tensors must be on GPU");
  TORCH_CHECK(in_a.dtype() == Y.dtype() && in_s.dtype() == Y.dtype(),
              "in_a / in_s / Y must share dtype (fp16 or bf16); got in_a=",
              in_a.dtype(), " in_s=", in_s.dtype(), " Y=", Y.dtype());
  TORCH_CHECK(Y.dtype() == at::kHalf || Y.dtype() == at::kBFloat16,
              "Y dtype must be float16 or bfloat16; got ", Y.dtype());
  TORCH_CHECK(in_a.dim() == 2, "in_a must be 2-D [M, K]");
  TORCH_CHECK(in_b.dim() == 3,
              "in_b must be 3-D [K0, N, K1/2] (CK pk_i4 layout)");
  TORCH_CHECK(in_s.dim() == 2, "in_s must be 2-D [N, K/G] row-major");
  TORCH_CHECK(group_size == 32 || group_size == 128,
              "CK W4A16 b_scale GEMM: only group_size 32 and 128 are wired; "
              "got group_size=", group_size);

  const int64_t M = in_a.size(0);
  const int64_t K = in_a.size(1);
  const int64_t K0 = in_b.size(0);
  const int64_t N = in_b.size(1);
  const int64_t K1_half = in_b.size(2);
  TORCH_CHECK(K0 * ck_w4a16::KPerBlock == K, "K0 * KPerBlock != K (", K0, "*",
              ck_w4a16::KPerBlock, "!=", K, ")");
  TORCH_CHECK(K1_half * 2 == ck_w4a16::KPerBlock,
              "in_b last dim must be KPerBlock/2 (", K1_half,
              "*2 !=", ck_w4a16::KPerBlock, ")");
  TORCH_CHECK(in_s.size(0) == N && in_s.size(1) * group_size == K,
              "in_s shape must be [N, K/G]; got [", in_s.size(0), ",",
              in_s.size(1), "] for K=", K, " N=", N, " G=", group_size);
  TORCH_CHECK(in_s.is_contiguous(),
              "in_s must be contiguous row-major [N, K/G]");
  TORCH_CHECK(Y.size(0) == M && Y.size(1) == N,
              "Y must be [M, N]; got [", Y.size(0), ",", Y.size(1),
              "] for M=", M, " N=", N);

  if (scaled_zp.has_value()) {
    const at::Tensor& zp = scaled_zp.value();
    TORCH_CHECK(zp.is_cuda(), "scaled_zp must be on GPU");
    TORCH_CHECK(zp.dtype() == Y.dtype(),
                "scaled_zp dtype must match Y; got zp=", zp.dtype(),
                " Y=", Y.dtype());
    TORCH_CHECK(zp.sizes() == in_s.sizes(),
                "scaled_zp must have the same shape as in_s [N, K/G]");
    TORCH_CHECK(zp.is_contiguous(),
                "scaled_zp must be contiguous row-major [N, K/G]");
  }

  const bool pdl = pre_dequant_to_lds.value_or(false);
  const bool tbt = truncate_bf16_round.value_or(false);
  if (Y.dtype() == at::kHalf) {
    run_kernel<F16>(in_a, in_b, in_s, Y, group_size, pdl, tbt,
                    zp_ptr<F16>(scaled_zp));
  } else {  // at::kBFloat16
    // The CK submodule now ships the bf16 DequantPack8 / DequantPack8WithZp
    // overloads (commit "[CK] DequantPack8 + DequantPack8WithZp bf16
    // overloads (asymmetric int4 / AWQ)"), so the bhalf8_t instantiation
    // links cleanly. Dispatch to the same templated kernel.
    run_kernel<B16>(in_a, in_b, in_s, Y, group_size, pdl, tbt,
                    zp_ptr<B16>(scaled_zp));
  }
  return Y;
}
