// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

// Host entries of the gfx950 bf16 grouped GEMMs (pre-built .co images); all shape / stride checks live here.
#ifndef __HIP_DEVICE_COMPILE__

#include "opus_gemm_arch.cuh"                      // opus_get_gfx_arch
#include "opus_build_archs.h"                      // OPUS_BUILD_HAS_GFX950
#include "opus_gemm_grouped.h"

#include <algorithm>
#include <climits>
#include <string>
#ifdef OPUS_BUILD_HAS_GFX950
#include "gfx950/opus_co_launch_gfx950.cuh"
#include "gfx950/opus_gemm_traits_a16w16_grouped_gfx950.cuh"
#include "opus_gemm_a16w16_grouped_kid_table.h"    // generated gfx950 grouped kid tables
#endif

namespace {

// Device, current device and gfx950 checks shared by both entries.
void check_devices(const char* entry, std::initializer_list<const aiter_tensor_t*> tensors)
{
  const aiter_tensor_t& first = **tensors.begin();
  for (const auto* t : tensors)
  {
    AITER_CHECK(t->is_gpu(), entry, ": all tensors must be GPU tensors");
    AITER_CHECK(t->device_id == first.device_id, entry, ": all tensors must be on device ",
                first.device_id);
  }
  int current_device = -1;
  HIP_CALL(hipGetDevice(&current_device));
  AITER_CHECK(current_device == first.device_id, entry, ": current HIP device ", current_device,
              " does not match tensor device ", first.device_id);
}

#ifdef OPUS_BUILD_HAS_GFX950
struct GroupedKid
{
  int kid, b_m, b_n;
  const char* name;
};

const GroupedKid& find_kid(const char* entry, const GroupedKid* kids, int kid)
{
  AITER_CHECK(opus_get_gfx_arch() == OpusGfxArch::Gfx950, entry, ": gfx950 only");
  const GroupedKid* k = kids;
  while (k->name != nullptr && k->kid != kid)
    ++k;
  AITER_CHECK(k->name != nullptr, entry, ": unknown kid ", kid, " (or its images are not built)");
  return *k;
}

constexpr GroupedKid m_grouped_kids[] = {OPUS_A16W16_M_GROUPED_KID_TABLE{-1, 0, 0, nullptr}};
constexpr GroupedKid k_grouped_kids[] = {OPUS_A16W16_K_GROUPED_KID_TABLE{-1, 0, 0, nullptr}};
#endif

} // namespace

// M-grouped GEMM (DeepGEMM m_grouped): group = grouped_layout [M] (contiguous, plan kernel -> workspace [2 G])
// or masked_m [G] (masked).
void opus_gemm_a16w16_grouped_launch(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    aiter_tensor_t &group,
    aiter_tensor_t &workspace,
    int kid,
    bool masked)
{
  aiter_detail::g_aiter_can_throw = true;
  const char* entry = "opus_gemm_a16w16_grouped_launch";
  check_devices(entry, {&XQ, &WQ, &Y, &group, &workspace});
#ifdef OPUS_BUILD_HAS_GFX950
  const GroupedKid& k = find_kid(entry, m_grouped_kids, kid);

  AITER_CHECK(XQ.dtype() == AITER_DTYPE_bf16 && WQ.dtype() == AITER_DTYPE_bf16 &&
                  Y.dtype() == AITER_DTYPE_bf16,
              entry, ": XQ, WQ and Y must be bf16");
  AITER_CHECK(group.dtype() == AITER_DTYPE_i32 && group.dim() == 1 && group.stride(0) == 1,
              entry, ": the group tensor must be a contiguous 1-D int32 tensor");
  const int rank = masked ? 3 : 2;
  AITER_CHECK(XQ.dim() == rank && Y.dim() == rank && WQ.dim() == 3, entry,
              masked ? ": masked needs XQ [G, max_m, K], WQ [G, N, K], Y [G, max_m, N]"
                     : ": contiguous needs XQ [M, K], WQ [G, N, K], Y [M, N]");
  const int64_t G = WQ.size(0), N = WQ.size(1), K = WQ.size(2);
  const int64_t M = XQ.size(-2);
  AITER_CHECK(XQ.size(-1) == K && Y.size(-2) == M && Y.size(-1) == N &&
                  (!masked || (XQ.size(0) == G && Y.size(0) == G)),
              entry, ": shapes of XQ / WQ / Y do not match");
  AITER_CHECK(group.size(0) == (masked ? G : M), entry, ": the group tensor must have ",
              masked ? "G" : "M", " entries");
  AITER_CHECK(K % 64 == 0 && K >= 128, entry, ": K=", K, " must be a multiple of 64 and >= 128");
  AITER_CHECK(masked || M % OPUS_A16W16_GROUP_ALIGN == 0, entry, ": contiguous M=", M,
              " must be a multiple of ", OPUS_A16W16_GROUP_ALIGN);
  AITER_CHECK(masked || (G <= OPUS_M_GROUPED_PLAN_MAX_GROUPS && workspace.dtype() == AITER_DTYPE_i32 &&
                         workspace.dim() == 1 && workspace.stride(0) == 1 && workspace.size(0) >= 2 * G),
              entry, ": contiguous needs G <= ", OPUS_M_GROUPED_PLAN_MAX_GROUPS,
              " and an int32 workspace of 2 G entries");
  if (M == 0 || N == 0 || G == 0)
    return;

  // B: K-contiguous [G, N, K] (NT) or N-contiguous [G, K, N] (NN).
  const bool b_t = !(WQ.stride(2) == 1 && WQ.stride(1) >= K);
  AITER_CHECK(!b_t || (WQ.stride(1) == 1 && WQ.stride(2) >= N), entry,
              ": WQ must be K-contiguous [G, N, K] or N-contiguous [G, K, N]");
  const int64_t ldb = b_t ? WQ.stride(2) : WQ.stride(1);
  const int64_t lda = XQ.stride(-2), ldc = Y.stride(-2);
  AITER_CHECK(XQ.stride(-1) == 1 && lda >= K && Y.stride(-1) == 1 && ldc >= N, entry,
              ": XQ and Y rows must be contiguous");
  const int64_t a_batch = masked ? XQ.stride(0) : 0, c_batch = masked ? Y.stride(0) : 0;
  AITER_CHECK(lda % 8 == 0 && ldb % 8 == 0 && WQ.stride(0) % 8 == 0 && a_batch % 8 == 0,
              entry, ": leading dimensions and batch strides must be multiples of 8 elements");
  for (const auto* t : {&XQ, &WQ, &Y})
    AITER_CHECK(reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0, entry,
                ": XQ / WQ / Y must be 16-byte aligned");
  // The kernel keeps per-group B offsets and the batch strides of A / C in 32-bit ints.
  AITER_CHECK((b_t ? K : N) * ldb <= INT_MAX && a_batch <= INT_MAX && c_batch <= INT_MAX &&
                  M <= INT_MAX && ldc * 256 <= INT_MAX,
              entry, ": a group of XQ / WQ / Y exceeds the kernel's 32-bit offsets");

  opus_gemm_a16w16_grouped_kargs_gfx950 kargs{};
  kargs.ptr_a = XQ.data_ptr();
  kargs.ptr_b = WQ.data_ptr();
  kargs.ptr_c = Y.data_ptr();
  int* const ws = static_cast<int*>(workspace.data_ptr());
  kargs.ptr_rows = masked ? static_cast<const int*>(group.data_ptr()) : ws;
  kargs.ptr_first = masked ? nullptr : ws + G;
  kargs.groups = static_cast<int>(G);
  kargs.stride_b_group = WQ.stride(0);
  kargs.m = static_cast<int>(M);
  kargs.n = static_cast<int>(N);
  kargs.k = static_cast<int>(K);
  kargs.stride_a = static_cast<int>(lda);
  kargs.stride_b = static_cast<int>(ldb);
  kargs.stride_c = static_cast<int>(ldc);
  kargs.stride_a_batch = static_cast<int>(a_batch);
  kargs.stride_c_batch = static_cast<int>(c_batch);

  // Grid: an upper bound of the tiles (<= ceil(rows / 256) per group) x N tiles; extra blocks exit.
  const int64_t tiles_n = (N + k.b_n - 1) / k.b_n;
  const int64_t chunks = (M + OPUS_A16W16_GROUP_ALIGN - 1) / OPUS_A16W16_GROUP_ALIGN;
  const int64_t slots = masked ? G * ((M + 255) / 256) : (chunks + G + 1) / 2;
  AITER_CHECK(slots * tiles_n <= INT_MAX, entry, ": grid too large");
  const std::string symbol = std::string(k.name) + (b_t ? "_nn" : "_nt");
  const std::string image = symbol + ".co";
  const hipStream_t stream = aiter::getCurrentHIPStream();
  if (!masked) {
    opus_gemm_a16w16_m_grouped_plan_kargs_gfx950 plan{};
    plan.layout = static_cast<const int*>(group.data_ptr());
    plan.rows = ws;
    plan.first = ws + G;
    plan.m = static_cast<int>(M);
    plan.groups = static_cast<int>(G);
    opus_co_launch_gfx950<sizeof(plan)>(
        opus_gfx950_co::co_kernel(symbol + "_plan", image, OPUS_A16W16_GROUPED_CO_DIR), dim3(1),
        dim3(OPUS_M_GROUPED_PLAN_THREADS), plan, stream);
  }
  opus_co_launch_gfx950<sizeof(kargs)>(
      opus_gfx950_co::co_kernel(symbol, image, OPUS_A16W16_GROUPED_CO_DIR),
      dim3(static_cast<unsigned>(slots * tiles_n), 1, 1), dim3(256), kargs, stream);
#else
  AITER_CHECK(false, entry, ": the module was built without gfx950");
#endif
}

// K-grouped GEMM (DeepGEMM k_grouped): Y[g] = (C[g] +) XQ[rows of g]^T WQ[rows of g], grouped_layout [G] rows
// per group; groups over seg_len rows split (fp32 Y), workspace = plan table + turn counters.
void opus_gemm_a16w16_k_grouped_launch(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    aiter_tensor_t &grouped_layout,
    std::optional<aiter_tensor_t> C,
    aiter_tensor_t &workspace,
    int kid,
    int seg_len)
{
  aiter_detail::g_aiter_can_throw = true;
  const char* entry = "opus_gemm_a16w16_k_grouped_launch";
  if (C)
    check_devices(entry, {&XQ, &WQ, &Y, &grouped_layout, &workspace, &*C});
  else
    check_devices(entry, {&XQ, &WQ, &Y, &grouped_layout, &workspace});
#ifdef OPUS_BUILD_HAS_GFX950
  const GroupedKid& k = find_kid(entry, k_grouped_kids, kid);

  const bool f32 = Y.dtype() == AITER_DTYPE_fp32;
  AITER_CHECK(XQ.dtype() == AITER_DTYPE_bf16 && WQ.dtype() == AITER_DTYPE_bf16 &&
                  (f32 || Y.dtype() == AITER_DTYPE_bf16),
              entry, ": XQ and WQ must be bf16, Y fp32 or bf16");
  AITER_CHECK(f32 || (!C && seg_len <= 0), entry,
              ": bf16 Y supports neither C nor a split (seg_len <= 0)");
  AITER_CHECK(grouped_layout.dtype() == AITER_DTYPE_i32 && grouped_layout.dim() == 1 &&
                  grouped_layout.stride(0) == 1,
              entry, ": grouped_layout must be a contiguous 1-D int32 tensor [G]");
  AITER_CHECK(workspace.dtype() == AITER_DTYPE_i32 && workspace.dim() == 1 &&
                  workspace.stride(0) == 1,
              entry, ": workspace must be a contiguous 1-D int32 tensor");
  AITER_CHECK(XQ.dim() == 2 && WQ.dim() == 2 && Y.dim() == 3, entry,
              ": needs XQ [R, M], WQ [R, N], Y [G, M, N]");
  const int64_t R = XQ.size(0), M = XQ.size(1), N = WQ.size(1), G = Y.size(0);
  AITER_CHECK(WQ.size(0) == R && Y.size(1) == M && Y.size(2) == N && grouped_layout.size(0) == G, entry,
              ": shapes of XQ / WQ / Y / grouped_layout do not match");
  if (C)
    AITER_CHECK(C->dtype() == Y.dtype() && C->dim() == 3 && C->size(0) == G &&
                    C->size(1) == M && C->size(2) == N && C->stride(0) == Y.stride(0) &&
                    C->stride(1) == Y.stride(1) && C->stride(2) == Y.stride(2),
                entry, ": C must have the dtype, shape and strides of Y");
  if (M == 0 || N == 0 || G == 0)
    return;

  const int64_t lda = XQ.stride(0), ldb = WQ.stride(0), ldc = Y.stride(1);
  AITER_CHECK(XQ.stride(1) == 1 && lda >= M && WQ.stride(1) == 1 && ldb >= N &&
                  Y.stride(2) == 1 && ldc >= N && Y.stride(0) >= M * ldc,
              entry, ": XQ, WQ and the matrices of Y must be row-major");
  AITER_CHECK(lda % 8 == 0 && ldb % 8 == 0 && ldc % 8 == 0 && N % 8 == 0 && Y.stride(0) % 8 == 0,
              entry, ": N, the row pitches and the group pitch of Y must be multiples of 8 elements");
  for (const auto* t : {&XQ, &WQ, &Y})
    AITER_CHECK(reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0, entry,
                ": XQ / WQ / Y must be 16-byte aligned");
  if (C)
    AITER_CHECK(reinterpret_cast<uintptr_t>(C->data_ptr()) % 16 == 0, entry,
                ": C must be 16-byte aligned");
  AITER_CHECK(seg_len <= 0 || seg_len % OPUS_A16W16_GROUP_ALIGN == 0, entry, ": seg_len=", seg_len,
              " must be a multiple of ", OPUS_A16W16_GROUP_ALIGN);
  constexpr int NO_SPLIT = INT_MAX / OPUS_A16W16_GROUP_ALIGN * OPUS_A16W16_GROUP_ALIGN;
  const int seg = seg_len > 0 ? seg_len : NO_SPLIT;
  // A segment is addressed with 32-bit byte offsets: at most max_rows rows (the plan kernel traps past it).
  const int64_t max_rows = INT_MAX / (2 * std::max(lda, ldb)) - 64;
  AITER_CHECK(R <= INT_MAX && max_rows >= OPUS_A16W16_GROUP_ALIGN && ldc * 256 * 4 <= INT_MAX &&
                  (seg_len <= 0 || seg_len <= max_rows),
              entry, ": seg_len=", seg_len, " exceeds the kernel's 32-bit offsets (at most ", max_rows, " rows)");

  const int64_t tiles_m = (M + k.b_m - 1) / k.b_m, tiles_n = (N + k.b_n - 1) / k.b_n;
  const int64_t tiles = tiles_m * tiles_n;
  const int64_t full_items = R / seg;
  const int64_t turn_count = full_items > 0 ? G * tiles : 0;
  const int64_t items = full_items + G;
  // full_items < 32767: a group's segment count (<= full_items + 1) is packed as nseg << 16 in an int.
  AITER_CHECK(items * tiles <= INT_MAX && G * tiles <= INT_MAX && full_items < 32767, entry,
              ": grid too large");
  AITER_CHECK(workspace.size(0) >= 4 * items + turn_count, entry, ": workspace needs ",
              4 * items + turn_count, " int32 entries, got ", workspace.size(0));

  int* const ws = static_cast<int*>(workspace.data_ptr());
  opus_gemm_a16w16_k_grouped_plan_kargs_gfx950 plan{};
  plan.grouped_layout = static_cast<const int*>(grouped_layout.data_ptr());
  plan.table = ws;
  plan.turn = ws + 4 * items;
  plan.groups = static_cast<int>(G);
  plan.seg_len = seg;
  plan.rows = static_cast<int>(R);
  plan.full_items = static_cast<int>(full_items);
  plan.turn_count = static_cast<int>(turn_count);
  plan.max_rows = static_cast<int>(std::min<int64_t>(max_rows, INT_MAX));

  opus_gemm_a16w16_k_grouped_kargs_gfx950 kargs{};
  kargs.ptr_a = XQ.data_ptr();
  kargs.ptr_b = WQ.data_ptr();
  kargs.ptr_c = Y.data_ptr();
  kargs.ptr_src = C ? C->data_ptr() : nullptr;
  kargs.table = ws;
  kargs.turn = plan.turn;
  kargs.stride_c_group = Y.stride(0);
  kargs.m = static_cast<int>(M);
  kargs.n = static_cast<int>(N);
  kargs.stride_a = static_cast<int>(lda);
  kargs.stride_b = static_cast<int>(ldb);
  kargs.stride_c = static_cast<int>(ldc);
  kargs.write_empty = !(C && C->data_ptr() == Y.data_ptr());

  const std::string symbol = std::string(k.name) + (f32 ? "_tn_f32" : "_tn");
  const std::string image = symbol + ".co";
  const hipStream_t stream = aiter::getCurrentHIPStream();
  opus_co_launch_gfx950<sizeof(plan)>(
      opus_gfx950_co::co_kernel(symbol + "_plan", image, OPUS_A16W16_GROUPED_CO_DIR), dim3(1),
      dim3(OPUS_K_GROUPED_PLAN_THREADS), plan, stream);
  opus_co_launch_gfx950<sizeof(kargs)>(
      opus_gfx950_co::co_kernel(symbol, image, OPUS_A16W16_GROUPED_CO_DIR),
      dim3(static_cast<unsigned>(items * tiles)), dim3(256), kargs, stream);
#else
  AITER_CHECK(false, entry, ": the module was built without gfx950");
#endif
}

#endif // !__HIP_DEVICE_COMPILE__
