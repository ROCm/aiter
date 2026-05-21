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
// (fp16 or bf16), ScaleBlockK (32 or 128 — AWQ group_size), and
// PreDequantToLDS (false = fused-dequant baseline, true = pre-dequant-to-LDS
// variant — see DeviceGemmInstance docs in gemm_w4a16_common.cuh).
// `p_b_zero_point` is nullptr for the symmetric path. Argument validation
// (dtype / shape / contiguity) is done by the dispatcher one level up
// before we get here.
template <typename T, ck::index_t ScaleBlockK, bool PreDequantToLDS,
          ck_w4a16::TileConfigKind Tile>
inline void run_kernel_inner(const at::Tensor& in_a,
                             const at::Tensor& in_b,
                             const at::Tensor& in_s,
                             at::Tensor& out,
                             int64_t group_size,
                             const T* p_b_zero_point) {
  // TODO(AIESW-32282): drop this guard once the pre-dequant-to-LDS pipeline
  // is implemented. The template surface is wired end-to-end (4x dtypes/G
  // pairs x 2 PreDequantToLDS flavors = 8 instantiations), so the test +
  // dispatcher can route here even though the kernel body for
  // PreDequantToLDS=true is currently the same as the false specialization
  // (see the DeviceGemmInstanceImpl<..., true> stub in
  // gemm_w4a16_common.cuh).
  if constexpr (PreDequantToLDS) {
    TORCH_CHECK(false,
                "CK W4A16 b_scale GEMM: PreDequantToLDS=true is not yet "
                "implemented (template hook surfaced, kernel body pending — "
                "see TODO(AIESW-32282) in gemm_w4a16_common.cuh).");
  }

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
      DeviceGemmInstance<T, ScaleBlockK, PreDequantToLDS, Tile>{};
  auto invoker = gemm.MakeInvoker();
  // AIESW-32735 B'': the Baseline_PackedSb tile uses BScaleDataType=float
  // (per-group fp32 carrying both scale and bias_eff), so the in_s pointer
  // must be cast to const float* and p_b_zero_point must be nullptr (sym
  // branch of the v1 pipeline). All other tiles use BScaleDataType=T.
  auto launch_with_args = [&](auto&& argument) {
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
  };

  if constexpr (Tile == ck_w4a16::TileConfigKind::Baseline_PackedSb &&
                !PreDequantToLDS) {
    auto argument = gemm.MakeArgument(
        reinterpret_cast<const T*>(in_a.data_ptr()),
        reinterpret_cast<const BDataType*>(in_b.data_ptr()),
        reinterpret_cast<T*>(out.data_ptr()), M, N, K, StrideA, StrideB, StrideC,
        Scale_Stride_BN, reinterpret_cast<const float*>(in_s.data_ptr()), KBatch,
        PassThrough{}, PassThrough{}, PassThrough{},
        /*p_b_zero_point=*/static_cast<const float*>(nullptr));
    launch_with_args(argument);
  } else {
    auto argument = gemm.MakeArgument(
        reinterpret_cast<const T*>(in_a.data_ptr()),
        reinterpret_cast<const BDataType*>(in_b.data_ptr()),
        reinterpret_cast<T*>(out.data_ptr()), M, N, K, StrideA, StrideB, StrideC,
        Scale_Stride_BN, reinterpret_cast<const T*>(in_s.data_ptr()), KBatch,
        PassThrough{}, PassThrough{}, PassThrough{}, p_b_zero_point);
    launch_with_args(argument);
  }
}

// Runtime dispatch on group_size + PreDequantToLDS into the matching
// template instantiation. Only group_size in {32, 128} are wired today
// (the two AWQ group sizes shipped by the models we target on gfx1151);
// the dispatcher one level up gates `group_size` to those two before we
// ever get here. The PreDequantToLDS flag multiplies the instantiation
// count to 4 per dtype (2 group sizes x 2 PreDequantToLDS); kernel build
// time scales accordingly. The PreDequantToLDS=true path currently
// TORCH_CHECKs at runtime (see TODO(AIESW-32282)).
template <typename T>
inline void run_kernel(const at::Tensor& in_a,
                       const at::Tensor& in_b,
                       const at::Tensor& in_s,
                       at::Tensor& out,
                       int64_t group_size,
                       bool pre_dequant_to_lds,
                       int64_t tile_config_kind,
                       const T* p_b_zero_point) {
  // 2 x 2 x 4 dispatch: group_size x pre_dequant_to_lds x tile_config_kind.
  auto launch = [&](auto g_const, auto pdl_const, auto tile_const) {
    constexpr ck::index_t G                  = decltype(g_const)::value;
    constexpr bool PDL                       = decltype(pdl_const)::value;
    constexpr ck_w4a16::TileConfigKind Tile  = decltype(tile_const)::value;
    run_kernel_inner<T, G, PDL, Tile>(in_a, in_b, in_s, out, group_size,
                                      p_b_zero_point);
  };

  // AIESW-32735: only the AWQ-asym fp16 path needs experimental tile configs
  // today; we still need the macro to work for both group_size values and
  // both pre_dequant_to_lds branches because that's the existing surface.
  // The tile_config dimension is the new axis. The default (Baseline) tile
  // keeps the existing kernel selection bit-for-bit.
#define _CK_W4A16_DISPATCH_G_PDL_TILE(G_VAL, PDL_TY, TILE_KIND)                \
  launch(std::integral_constant<ck::index_t, G_VAL>{},                         \
         PDL_TY{},                                                             \
         std::integral_constant<ck_w4a16::TileConfigKind, TILE_KIND>{})

#define _CK_W4A16_DISPATCH_G_PDL(G_VAL, PDL_TY)                                \
  do {                                                                         \
    switch (static_cast<ck_w4a16::TileConfigKind>(tile_config_kind)) {         \
      case ck_w4a16::TileConfigKind::Baseline:                                 \
        _CK_W4A16_DISPATCH_G_PDL_TILE(G_VAL, PDL_TY,                           \
                                      ck_w4a16::TileConfigKind::Baseline);    \
        break;                                                                 \
      case ck_w4a16::TileConfigKind::WideM:                                    \
        _CK_W4A16_DISPATCH_G_PDL_TILE(G_VAL, PDL_TY,                           \
                                      ck_w4a16::TileConfigKind::WideM);       \
        break;                                                                 \
      case ck_w4a16::TileConfigKind::LargeK:                                   \
        _CK_W4A16_DISPATCH_G_PDL_TILE(G_VAL, PDL_TY,                           \
                                      ck_w4a16::TileConfigKind::LargeK);      \
        break;                                                                 \
      case ck_w4a16::TileConfigKind::WideM_LargeK:                             \
        _CK_W4A16_DISPATCH_G_PDL_TILE(G_VAL, PDL_TY,                           \
                                      ck_w4a16::TileConfigKind::WideM_LargeK);\
        break;                                                                 \
      case ck_w4a16::TileConfigKind::Tile64:                                   \
        _CK_W4A16_DISPATCH_G_PDL_TILE(G_VAL, PDL_TY,                           \
                                      ck_w4a16::TileConfigKind::Tile64);      \
        break;                                                                 \
      case ck_w4a16::TileConfigKind::Tile64_LargeK:                            \
        _CK_W4A16_DISPATCH_G_PDL_TILE(G_VAL, PDL_TY,                           \
                                      ck_w4a16::TileConfigKind::Tile64_LargeK);\
        break;                                                                 \
      case ck_w4a16::TileConfigKind::NarrowN:                                  \
        _CK_W4A16_DISPATCH_G_PDL_TILE(G_VAL, PDL_TY,                           \
                                      ck_w4a16::TileConfigKind::NarrowN);     \
        break;                                                                 \
      case ck_w4a16::TileConfigKind::Baseline_Sym_V1:                          \
        _CK_W4A16_DISPATCH_G_PDL_TILE(G_VAL, PDL_TY,                           \
                                      ck_w4a16::TileConfigKind::Baseline_Sym_V1); \
        break;                                                                 \
      case ck_w4a16::TileConfigKind::Baseline_Sym_V3:                          \
        _CK_W4A16_DISPATCH_G_PDL_TILE(G_VAL, PDL_TY,                           \
                                      ck_w4a16::TileConfigKind::Baseline_Sym_V3); \
        break;                                                                 \
      case ck_w4a16::TileConfigKind::Baseline_Bias:                            \
        _CK_W4A16_DISPATCH_G_PDL_TILE(G_VAL, PDL_TY,                           \
                                      ck_w4a16::TileConfigKind::Baseline_Bias); \
        break;                                                                 \
      case ck_w4a16::TileConfigKind::Baseline_PackedSb:                        \
        _CK_W4A16_DISPATCH_G_PDL_TILE(G_VAL, PDL_TY,                           \
                                      ck_w4a16::TileConfigKind::Baseline_PackedSb); \
        break;                                                                 \
      default:                                                                 \
        TORCH_CHECK(false,                                                     \
                    "CK W4A16 b_scale GEMM: unsupported tile_config_kind=",    \
                    tile_config_kind,                                          \
                    " (valid: 0=Baseline, 1=WideM, 2=LargeK, "                 \
                    "3=WideM_LargeK, 4=Tile64, 5=Tile64_LargeK, "              \
                    "6=NarrowN, 7=Baseline_Sym_V1, 8=Baseline_Sym_V3, "        \
                    "9=Baseline_Bias, 10=Baseline_PackedSb)");                 \
    }                                                                          \
  } while (0)

#define _CK_W4A16_DISPATCH_G(G_VAL)                                            \
  do {                                                                         \
    if (pre_dequant_to_lds) {                                                  \
      _CK_W4A16_DISPATCH_G_PDL(G_VAL, std::true_type);                         \
    } else {                                                                   \
      _CK_W4A16_DISPATCH_G_PDL(G_VAL, std::false_type);                        \
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
#undef _CK_W4A16_DISPATCH_G_PDL
#undef _CK_W4A16_DISPATCH_G_PDL_TILE
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
//
// AIESW-32282: bf16 rounding mode is no longer a runtime axis. The CK
// kernel ships truncate-to-bf16 only (i4_to_bhalf4_scale /
// i4_to_bhalf4_zp_scale in unary_element_wise_operation.hpp), verified
// statistically indistinguishable from Triton on lm_eval gsm8k 5-shot.
torch::Tensor gemm_w4a16(at::Tensor& in_a,
                         at::Tensor& in_b,
                         at::Tensor& in_s,
                         at::Tensor& Y,
                         int64_t group_size,
                         std::optional<at::Tensor> scaled_zp,
                         std::optional<bool> pre_dequant_to_lds,
                         std::optional<int64_t> tile_config) {
  TORCH_CHECK(in_a.is_cuda() && in_b.is_cuda() && in_s.is_cuda() &&
                  Y.is_cuda(),
              "All tensors must be on GPU");
  // AIESW-32735 B'': for tile_config=10 (Baseline_PackedSb) the in_s slot
  // carries a packed fp32 buffer (scale in low 16, bias_eff in high 16) so
  // the in_s dtype is float32 and the shape's K-dim stays the same. All
  // other tiles use the activation dtype for in_s.
  const int64_t _early_tile_kind = tile_config.value_or(0);
  const bool _is_packed_sb_tile = (_early_tile_kind == 10);
  TORCH_CHECK(in_a.dtype() == Y.dtype(),
              "in_a / Y must share dtype (fp16 or bf16); got in_a=",
              in_a.dtype(), " Y=", Y.dtype());
  if (_is_packed_sb_tile) {
    TORCH_CHECK(in_s.dtype() == at::kFloat,
                "tile_config=10 (Baseline_PackedSb) requires in_s dtype fp32 "
                "(packed scale+bias_eff); got in_s=", in_s.dtype());
    TORCH_CHECK(!scaled_zp.has_value(),
                "tile_config=10 (Baseline_PackedSb) must be called with "
                "scaled_zp=None — the sym branch of the v1 pipeline is used "
                "(packed buffer carries both scale and bias_eff).");
  } else {
    TORCH_CHECK(in_s.dtype() == Y.dtype(),
                "in_s / Y must share dtype (fp16 or bf16); got in_s=",
                in_s.dtype(), " Y=", Y.dtype());
  }
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
  const int64_t tile_kind = tile_config.value_or(0);
  if (Y.dtype() == at::kHalf) {
    run_kernel<F16>(in_a, in_b, in_s, Y, group_size, pdl, tile_kind,
                    zp_ptr<F16>(scaled_zp));
  } else {  // at::kBFloat16
    // The CK submodule now ships the bf16 DequantPack8 / DequantPack8WithZp
    // overloads (commit "[CK] DequantPack8 + DequantPack8WithZp bf16
    // overloads (asymmetric int4 / AWQ)"), so the bhalf8_t instantiation
    // links cleanly. Dispatch to the same templated kernel.
    run_kernel<B16>(in_a, in_b, in_s, Y, group_size, pdl, tile_kind,
                    zp_ptr<B16>(scaled_zp));
  }
  return Y;
}
