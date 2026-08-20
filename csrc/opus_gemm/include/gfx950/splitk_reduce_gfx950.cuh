// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Split-K reduce kernel: tile-agnostic; sums an fp32 workspace across the
// split-K axis, casts fp32 -> D_OUT, and writes to C. Split out of
// opus_gemm_pipeline_a16w16_flatmm_splitk_gfx950.cuh so the reduce path can be
// shared by future split-K main pipelines (a8w8 etc.) without dragging in
// the full a16w16 flatmm pipeline header.
//
// Template parameters:
//   * VEC_       - elements per thread along N (fast-path tile width).
//   * BLOCK_     - threads per block.
//   * D_OUT      - output element type. Currently exercised with __bf16 and
//                  float; any other 16-bit / 32-bit type that opus::vector_t
//                  supports also works mechanically because the store path
//                  dispatches on sizeof(D_OUT).
//   * HAS_BIAS_  - when true, fold a per-output-feature bias (D_BIAS_ vector
//                  along N, F.linear convention) into acc before the cast to
//                  D_OUT. Defaults off so the no-bias template instantiation
//                  stays binary-identical to the pre-bias code path.
//   * D_BIAS_    - bias element type. Currently 2B (bf16) or 4B (fp32). Must
//                  match the on-disk bias buffer dtype; mirrors the user-
//                  facing "match D_OUT" convention but is template-distinct
//                  so future fp32-bias-on-bf16-out callers can specialize.
//
// All splitk launchers invoke this kernel with <VEC_=16, BLOCK_=64>; D_OUT
// defaults to __bf16 to keep call-sites that omit the type unchanged.
//
// Grid: (ceil(N, VEC * BLOCK), batch * M, 1).
// Each thread handles VEC fp32 lanes along N; the workspace load path is
// always fp32 (4x buffer_load_dwordx4 for VEC=16) regardless of D_OUT.
//
// Store path bytes-per-thread:
//   * D_OUT = __bf16 -> 16 elems x 2B = 32B   (2 x buffer_store_dwordx4)
//   * D_OUT = float  -> 16 elems x 4B = 64B   (4 x buffer_store_dwordx4)
// We pick STEP = 16 / sizeof(D_OUT) so each store covers exactly one
// dwordx4 (128-bit) chunk and the inner loop runs VEC / STEP times.
//
// Store-path 3-way split on the N tail (unchanged from the bf16-only
// implementation):
//   * (n_base + VEC <= N): fast path, VEC/STEP x buffer_store_dwordx4.
//   * (n_base < N):        tail path, VEC_valid scalar buffer_store_b16/b32
//                          per in-range element. Prevents a 128-bit vector
//                          store from straddling the row-N boundary and
//                          silently landing in the next row of the
//                          row-major C tensor (the buffer rsrc only checks
//                          linear byte offset, not per-row column bounds).
//
// Bias semantics (HAS_BIAS_=true):
//   * Bias is per-output-feature (per-N, F.linear convention), so each
//     thread loads VEC bias values at its own n_base offset. We use
//     buffer_load (VGPR vmem) rather than the old SGPR s_load_dword
//     pattern because the bias value differs per thread.
//   * Layout: bias_stride_batch = 0 means [N] (broadcast across batch);
//     bias_stride_batch = N means [batch, N]. The kernel reads
//     `b * stride_bias_batch + n_base` and is shape-agnostic.
//   * Issue order: bias buffer_load is fired BEFORE the split-K vmcnt
//     accumulation so it overlaps with the vmem reduction. vmcnt(0) at
//     the end of the split-K loop drains both workspace and bias loads.
//   * Math is done in fp32 on top of the existing acc[VEC] before the
//     cast, so precision matches the existing fp32 reduction.
#pragma once

#include "../opus_gemm_utils.cuh"
#include "opus_gemm_traits_a16w16_gfx950.cuh"  // opus_splitk_ws_handle
#include <cstdint>   // uint16_t / uint32_t used by the bias-fold and bf16 store paths

template<int VEC_ = 16, int BLOCK_ = 64, typename D_OUT = __bf16,
         bool HAS_BIAS_ = false, typename D_BIAS_ = D_OUT,
         bool HAS_OOB_ = true>
__global__ void splitk_reduce_kernel(
    const opus_splitk_ws_handle* __restrict__ ws_handle,
    D_OUT*       __restrict__ c_out,
    int split_k, int M, int N, int batch,
    int padded_M, int padded_N,
    const D_BIAS_* __restrict__ bias = nullptr,
    int bias_stride_batch = 0)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    // gfx950-only kernel body. See opus_gemm_pipeline_a16w16_gfx950.cuh for the
    // multi-arch wheel rationale.
    //
    // Deref the handle slot at entry; survives a post-capture grow.
    const float* __restrict__ workspace =
        reinterpret_cast<const float*>(ws_handle->ptr);
    constexpr int VEC   = VEC_;
    constexpr int BLOCK = BLOCK_;
    constexpr bool HAS_BIAS = HAS_BIAS_;
    constexpr bool HAS_OOB = HAS_OOB_;
    using D_BIAS = D_BIAS_;

    // STEP = elements per buffer_store_dwordx4. STEP * sizeof(D_OUT) == 16.
    constexpr int STEP = 16 / sizeof(D_OUT);
    static_assert(STEP * sizeof(D_OUT) == 16,
                  "D_OUT must divide a 128-bit store boundary cleanly "
                  "(supported sizes: 2B / 4B; e.g. __bf16, float)");
    static_assert(VEC % STEP == 0,
                  "VEC must be a multiple of STEP so the fast path tiles "
                  "into whole dwordx4 stores");
    static_assert(!HAS_BIAS || sizeof(D_BIAS) == 2 || sizeof(D_BIAS) == 4,
                  "splitk_reduce HAS_BIAS path supports only 2B or 4B D_BIAS "
                  "(bf16 / fp32). Other widths require half-extract changes.");

    const int bm_id  = int(opus::block_id_y());            // 0..batch*M-1
    const int nblk   = int(opus::block_id_x());
    const int tid    = int(opus::thread_id_x());
    const int n_base = (nblk * BLOCK + tid) * VEC;

    const int b = bm_id / M;
    const int m = bm_id - b * M;

    // -- Bias prefetch (per-N vector load) ----------------------------------
    // Bias is per-output-feature [N] (F.linear convention). Each thread
    // loads VEC bias values at its own n_base. Fired before the split-K
    // accumulation so the vmem loads overlap.
    opus::vector_t<float, VEC> bias_fp32;
    if constexpr (HAS_BIAS) {
        #pragma unroll
        for (int t = 0; t < VEC; ++t) bias_fp32[t] = 0.0f;
        const D_BIAS* bias_base_ptr = bias + b * bias_stride_batch;
        auto g_bias = opus::make_gmem(bias_base_ptr,
                        (unsigned int)((bias_stride_batch ? bias_stride_batch : N)
                                       * sizeof(D_BIAS)));
        // Load VEC bias elements as groups of 4 (buffer_load_dwordx4 for
        // fp32; buffer_load_b64 pairs for bf16 -- opus::load handles both).
        #pragma unroll
        for (int g = 0; g < VEC / 4; ++g) {
            auto bv4 = g_bias.template load<4>(n_base + g * 4);
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                bias_fp32[g * 4 + j] = static_cast<float>(bv4[j]);
        }
    }

    const int  ws_row_base  = b * padded_M * padded_N + m * padded_N + n_base;
    const long split_stride = (long)batch * padded_M * padded_N;

    auto g_ws = opus::make_gmem(workspace,
                                (unsigned int)(split_stride * split_k * sizeof(float)));

    opus::vector_t<float, VEC> acc;
    #pragma unroll
    for (int t = 0; t < VEC; ++t) acc[t] = 0.0f;

    for (int s = 0; s < split_k; ++s) {
        int ws_idx = ws_row_base + (int)(s * split_stride);
        #pragma unroll
        for (int g = 0; g < VEC / 4; ++g) {
            auto v4 = g_ws.template load<4>(ws_idx + g * 4);
            #pragma unroll
            for (int j = 0; j < 4; ++j) acc[g * 4 + j] += v4[j];
        }
    }

    if constexpr (HAS_BIAS) {
        // Bias vmem loads were issued before the split-K loop. By the time
        // the loop finishes (vmcnt(0) inside g_ws.load), all bias loads
        // have also drained (they share the same vmcnt counter). No
        // additional wait is needed.
        #pragma unroll
        for (int t = 0; t < VEC; ++t) acc[t] += bias_fp32[t];
    }

    opus::vector_t<D_OUT, VEC> out;
    #pragma unroll
    for (int t = 0; t < VEC; ++t) out[t] = static_cast<D_OUT>(acc[t]);

    auto g_c = opus::make_gmem(c_out,
                               (unsigned int)((size_t)batch * M * N * sizeof(D_OUT)));
    const int c_idx = b * M * N + m * N + n_base;  // in D_OUT elements

    // Store-path macro helpers (compile-time offsets -> buffer_store_dwordx4/x2/dword/short).
    using opus::slice;
    using opus::number;
#define OPUS_REDUCE_ST8(OFF) g_c.template store<8>(slice(out, number<OFF>{}, number<OFF+8>{}), c_idx + (OFF))
#define OPUS_REDUCE_ST4(OFF) g_c.template store<4>(slice(out, number<OFF>{}, number<OFF+4>{}), c_idx + (OFF))
#define OPUS_REDUCE_ST2(OFF) g_c.template store<2>(slice(out, number<OFF>{}, number<OFF+2>{}), c_idx + (OFF))
#define OPUS_REDUCE_ST1(OFF) g_c.template store<1>(out[OFF], c_idx + (OFF))

    if constexpr (!HAS_OOB) {
        // Non-OOB: N is tile-aligned so no partial-VEC tail, but still
        // need to skip threads whose n_base is past N (reduce grid is
        // over-provisioned: grid.x = ceil(N, VEC*BLOCK)).
        if (n_base + VEC <= N) {
            opus::static_for<VEC / STEP>([&](auto g_c_idx) {
                constexpr int g = decltype(g_c_idx)::value;
                g_c.template store<STEP>(
                    slice(out, number<g * STEP>{}, number<(g + 1) * STEP>{}),
                    c_idx + g * STEP);
            });
        }
    } else {
        if (n_base + VEC <= N) {
            // Fast path: entire VEC chunk is in-row.
            opus::static_for<VEC / STEP>([&](auto g_c_idx) {
                constexpr int g = decltype(g_c_idx)::value;
                g_c.template store<STEP>(
                    slice(out, number<g * STEP>{}, number<(g + 1) * STEP>{}),
                    c_idx + g * STEP);
            });
        } else if (n_base < N) {
            // Tail path: decompose valid ? [1, VEC-1] into descending
            // power-of-2 chunks so we emit dwordx4/dwordx2/dword/short
            // instead of VEC scalar stores.
            // Ref: demon_gcn/opus_gemm/mxfp8_e8m0/gemm_mxfp_a8w8_1d1d.hpp
            static_assert(VEC == 16, "reduce tail switch assumes VEC=16");
            const int valid = N - n_base;
            if constexpr (sizeof(D_OUT) == 2) {
                // bf16: STEP=8, store<8>=dwordx4, store<4>=dwordx2, store<2>=dword, store<1>=short
                switch (valid) {
                    case  1: OPUS_REDUCE_ST1( 0); break;
                    case  2: OPUS_REDUCE_ST2( 0); break;
                    case  3: OPUS_REDUCE_ST2( 0); OPUS_REDUCE_ST1( 2); break;
                    case  4: OPUS_REDUCE_ST4( 0); break;
                    case  5: OPUS_REDUCE_ST4( 0); OPUS_REDUCE_ST1( 4); break;
                    case  6: OPUS_REDUCE_ST4( 0); OPUS_REDUCE_ST2( 4); break;
                    case  7: OPUS_REDUCE_ST4( 0); OPUS_REDUCE_ST2( 4); OPUS_REDUCE_ST1( 6); break;
                    case  8: OPUS_REDUCE_ST8( 0); break;
                    case  9: OPUS_REDUCE_ST8( 0); OPUS_REDUCE_ST1( 8); break;
                    case 10: OPUS_REDUCE_ST8( 0); OPUS_REDUCE_ST2( 8); break;
                    case 11: OPUS_REDUCE_ST8( 0); OPUS_REDUCE_ST2( 8); OPUS_REDUCE_ST1(10); break;
                    case 12: OPUS_REDUCE_ST8( 0); OPUS_REDUCE_ST4( 8); break;
                    case 13: OPUS_REDUCE_ST8( 0); OPUS_REDUCE_ST4( 8); OPUS_REDUCE_ST1(12); break;
                    case 14: OPUS_REDUCE_ST8( 0); OPUS_REDUCE_ST4( 8); OPUS_REDUCE_ST2(12); break;
                    case 15: OPUS_REDUCE_ST8( 0); OPUS_REDUCE_ST4( 8); OPUS_REDUCE_ST2(12); OPUS_REDUCE_ST1(14); break;
                }
            } else {
                // fp32: STEP=4, store<4>=dwordx4, store<2>=dwordx2, store<1>=dword
                switch (valid) {
                    case  1: OPUS_REDUCE_ST1( 0); break;
                    case  2: OPUS_REDUCE_ST2( 0); break;
                    case  3: OPUS_REDUCE_ST2( 0); OPUS_REDUCE_ST1( 2); break;
                    case  4: OPUS_REDUCE_ST4( 0); break;
                    case  5: OPUS_REDUCE_ST4( 0); OPUS_REDUCE_ST1( 4); break;
                    case  6: OPUS_REDUCE_ST4( 0); OPUS_REDUCE_ST2( 4); break;
                    case  7: OPUS_REDUCE_ST4( 0); OPUS_REDUCE_ST2( 4); OPUS_REDUCE_ST1( 6); break;
                    case  8: OPUS_REDUCE_ST4( 0); OPUS_REDUCE_ST4( 4); break;
                    case  9: OPUS_REDUCE_ST4( 0); OPUS_REDUCE_ST4( 4); OPUS_REDUCE_ST1( 8); break;
                    case 10: OPUS_REDUCE_ST4( 0); OPUS_REDUCE_ST4( 4); OPUS_REDUCE_ST2( 8); break;
                    case 11: OPUS_REDUCE_ST4( 0); OPUS_REDUCE_ST4( 4); OPUS_REDUCE_ST2( 8); OPUS_REDUCE_ST1(10); break;
                    case 12: OPUS_REDUCE_ST4( 0); OPUS_REDUCE_ST4( 4); OPUS_REDUCE_ST4( 8); break;
                    case 13: OPUS_REDUCE_ST4( 0); OPUS_REDUCE_ST4( 4); OPUS_REDUCE_ST4( 8); OPUS_REDUCE_ST1(12); break;
                    case 14: OPUS_REDUCE_ST4( 0); OPUS_REDUCE_ST4( 4); OPUS_REDUCE_ST4( 8); OPUS_REDUCE_ST2(12); break;
                    case 15: OPUS_REDUCE_ST4( 0); OPUS_REDUCE_ST4( 4); OPUS_REDUCE_ST4( 8); OPUS_REDUCE_ST2(12); OPUS_REDUCE_ST1(14); break;
                }
            }
        }
        // else: whole VEC chunk is past N -> write nothing.
    }
#undef OPUS_REDUCE_ST8
#undef OPUS_REDUCE_ST4
#undef OPUS_REDUCE_ST2
#undef OPUS_REDUCE_ST1
#else
    // Non-gfx950 device pass: empty stub. See gfx950 branch above.
#endif  // __gfx950__
#endif  // __HIP_DEVICE_COMPILE__
}

// -- opus_bmm_splitk_reduce_kernel ------------------------------------------
// mmajor batched-matmul split-K reduce. Sums the fp32 split-K workspace along
// the split axis and casts to D_OUT, writing an mmajor C tensor whose row
// (M) and batch strides are passed explicitly (stride_c / stride_c_batch).
// Distinct from splitk_reduce_kernel above (no bias fold, mmajor C strides,
// VEC/BLOCK defaults tuned for the BMM launcher's 8x128 grid). Lives here so
// the codegen'd a8w8_mxscale BMM launcher (.device.cu / fused host TU) can
// reference it; the explicit instantiations are emitted alongside the baseline
// reduce into splitk_reduce_gfx950.device.cu (gen_instances_gfx950.py's
// splitk_reduce_extra hook).
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
