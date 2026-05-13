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
// at runtime; M is dynamic. K must be a multiple of KPerBlock (32) and
// Scale_Block_K (128). Output dtype dispatches between fp16 and bf16.

#include "gemm_w4a16_common.cuh"

namespace {

using namespace ck_w4a16;

// Hot-path build of the CK Argument struct + invoker call. Templated on T
// (fp16 or bf16). `p_b_zero_point` is nullptr for the symmetric path. Argument
// validation (dtype / shape / contiguity) is done by the dispatcher one level
// up before we get here.
template <typename T>
inline void run_kernel(const at::Tensor& in_a,
                       const at::Tensor& in_b,
                       const at::Tensor& in_s,
                       at::Tensor& out,
                       int64_t group_size,
                       const T* p_b_zero_point) {
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

  auto gemm = DeviceGemmInstance<T>{};
  auto invoker = gemm.MakeInvoker();
  auto argument = gemm.MakeArgument(
      reinterpret_cast<const T*>(in_a.data_ptr()),
      reinterpret_cast<const BDataType*>(in_b.data_ptr()),
      reinterpret_cast<T*>(out.data_ptr()), M, N, K, StrideA, StrideB, StrideC,
      Scale_Stride_BN, reinterpret_cast<const T*>(in_s.data_ptr()), KBatch,
      PassThrough{}, PassThrough{}, PassThrough{}, p_b_zero_point);

  // IsSupportedArgument check is debug-only — measured ~840 us/call overhead
  // when run on every dispatch and the device-op rejects shapes only on
  // misuse (group_size != Scale_Block_K, K not divisible by KPerBlock, etc.)
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
// group_size:  must equal Scale_Block_K (128).
// scaled_zp:   optional [N, K/G] activation-dtype. None for symmetric
//              (uint4b8); set to (zp - 8) * scale precomputed at weight load
//              for asymmetric (AWQ). The kernel uses the algebraic identity
//                  (nibble - zp) * scale
//                    == (nibble - 8) * scale - (zp - 8) * scale
//              so the asymmetric path costs one extra activation-dtype vector
//              subtract per dequant pack vs the symmetric path.
torch::Tensor gemm_w4a16(at::Tensor& in_a,
                         at::Tensor& in_b,
                         at::Tensor& in_s,
                         at::Tensor& Y,
                         int64_t group_size,
                         std::optional<at::Tensor> scaled_zp) {
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
  TORCH_CHECK(group_size == ck_w4a16::Scale_Block_K,
              "group_size must equal CK Scale_Block_K (",
              ck_w4a16::Scale_Block_K, ")");

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

  if (Y.dtype() == at::kHalf) {
    run_kernel<F16>(in_a, in_b, in_s, Y, group_size, zp_ptr<F16>(scaled_zp));
  } else {  // at::kBFloat16
    // The CK patch (AIESW-32176) currently only ships the fp16 overloads of
    // DequantPack8 / DequantPack8WithZp; the bf16 template instantiation
    // would link-fail. Bail out cleanly until upstream CK adds the bhalf8_t
    // overloads.
    TORCH_CHECK(false,
                "CK W4A16 b_scale GEMM: bf16 path not built. Rebuild with "
                "the bf16 DequantPack8(WithZp) overloads in unary_element_"
                "wise_operation.hpp, or pass a fp16 Y tensor.");
  }
  return Y;
}
