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
//   * HAS_BIAS_  - when true, fold a per-row bias (D_BIAS_ scalar) into acc
//                  before the cast to D_OUT. Defaults off so the no-bias
//                  template instantiation stays binary-identical to the
//                  pre-bias code path.
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
//   * Bias is per-row (per-m), so every thread in a block (which fixes one
//     (b, m) pair via block_id_y = bm_id) reads the SAME scalar. We use an
//     SGPR scalar load (s_load_dword) instead of a per-thread vmem buffer
//     load -- this matches the load_sfb pattern in a8w8_scale and avoids
//     redundant L1 traffic.
//   * bias_ptr is forwarded as the host-computed `ptr_bias`; the inline asm
//     "s" constraint lets the compiler readfirstlane the address into an
//     SGPR pair (no manual move needed).
//   * bf16 bias: s_load_dword is 4B-aligned, so we issue the load at byte
//     offset (m >> 1) * 4 (which fetches bf16[m&~1] and bf16[m|1]) and pick
//     the correct half via (m & 1). bf16 -> fp32 is `<<16`. Out-of-range
//     reads on the "other" half are safe because the kargs-side buffer is
//     contiguous M (or batch*M) elements wide; the unused half is dropped.
//   * fp32 bias: a single dword via s_load_dword + bit_cast to float.
//   * Layout: bias_stride_batch = 0 means [M] (broadcast across batch);
//     bias_stride_batch = M means [batch, M]. The kernel only reads
//     `b * stride_bias_batch + m` and is shape-agnostic.
//   * Issue order: s_load is fired BEFORE the split-K vmcnt accumulation
//     loop. lgkmcnt(0) is awaited just before the bias add, by which time
//     the vmem accumulation has already drained on its own (vmcnt(0) inside
//     g_ws.load loop). The two counters are independent so the bias load
//     overlaps the entire split-K loop.
//   * Math is done in fp32 on top of the existing acc[VEC] before the cast,
//     so precision matches the existing fp32 reduction (no bf16 round-trip).
#pragma once

#include "../opus_gemm_utils.cuh"
#include <cstdint>   // uint16_t / uint32_t used by the bias-fold and bf16 store paths

template<int VEC_ = 16, int BLOCK_ = 64, typename D_OUT = __bf16,
         bool HAS_BIAS_ = false, typename D_BIAS_ = D_OUT>
__global__ void splitk_reduce_kernel(
    const float* __restrict__ workspace,
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
    constexpr int VEC   = VEC_;
    constexpr int BLOCK = BLOCK_;
    constexpr bool HAS_BIAS = HAS_BIAS_;
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

    // ── Bias prefetch (SGPR scalar load) ──────────────────────────────────
    // Fired BEFORE the split-K vmcnt accumulation so it overlaps with the
    // vmem reduction. The single dword lands in an SGPR; we extract the bf16
    // half (or read the fp32 directly) into a fp32 scalar after the loop
    // once lgkmcnt has drained.
    float bias_fp32 = 0.0f;
    uint32_t bias_dword = 0;
    if constexpr (HAS_BIAS) {
        // Host-side ptr already has any allocator-base offset baked in. We
        // add the per-batch row offset here so the SGPR address is exactly
        // the row's bf16 / fp32 base; b * stride_bias_batch collapses to 0
        // when stride_bias_batch=0 (broadcast [M]).
        const D_BIAS* bias_row_ptr = bias + b * bias_stride_batch;
        // Byte offset to the dword that contains bias[m]. For bf16 this
        // covers bias[m & ~1] and bias[m | 1]; for fp32 it's bias[m] alone.
        const int byte_offset = (sizeof(D_BIAS) == 2)
            ? ((m >> 1) * 4)
            : (m * 4);
        asm volatile("s_load_dword %0, %1, %2\n\t"
                     : "=s"(bias_dword)
                     : "s"(bias_row_ptr), "s"(byte_offset)
                     : "memory");
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
        // Wait on the scalar load issued at the head of the kernel. The
        // split-K vmem accumulation above is on vmcnt and is unaffected.
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");

        if constexpr (sizeof(D_BIAS) == 4) {
            bias_fp32 = __builtin_bit_cast(float, bias_dword);
        } else {
            // bf16: pick high or low 16 bits depending on m parity. bf16 ->
            // fp32 is "shift left 16" because bf16 is the high half of fp32.
            const uint16_t half = (m & 1) ? static_cast<uint16_t>(bias_dword >> 16)
                                          : static_cast<uint16_t>(bias_dword & 0xffffu);
            const uint32_t bf32 = static_cast<uint32_t>(half) << 16;
            bias_fp32 = __builtin_bit_cast(float, bf32);
        }

        #pragma unroll
        for (int t = 0; t < VEC; ++t) acc[t] += bias_fp32;
    }

    opus::vector_t<D_OUT, VEC> out;
    #pragma unroll
    for (int t = 0; t < VEC; ++t) out[t] = static_cast<D_OUT>(acc[t]);

    auto g_c = opus::make_gmem(c_out,
                               (unsigned int)((size_t)batch * M * N * sizeof(D_OUT)));
    const int c_idx = b * M * N + m * N + n_base;  // in D_OUT elements

    if (n_base + VEC <= N) {
        // Fast path: entire VEC chunk is in-row -> VEC/STEP x buffer_store_dwordx4.
        // static_for promotes the loop index to constexpr so opus::slice's
        // compile-time bounds (opus::number<>) can be formed.
        opus::static_for<VEC / STEP>([&](auto g_c_idx) {
            constexpr int g = decltype(g_c_idx)::value;
            g_c.template store<STEP>(
                opus::slice(out,
                            opus::number<g * STEP>{},
                            opus::number<(g + 1) * STEP>{}),
                c_idx + g * STEP);
        });
    } else if (n_base < N) {
        // Tail path: only the first `valid` elements of the VEC chunk are in
        // row m. Scalar-store them one by one; the remaining (VEC - valid)
        // elements would otherwise spill into row m+1 via a 128-bit vector
        // store (buffer rsrc cannot catch this because the target byte offset
        // is still < rsrc.size in a row-major tensor).
        const int valid = N - n_base;
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            if (j < valid) g_c.template store<1>(out[j], c_idx + j);
        }
    }
    // else: whole VEC chunk is past N -> write nothing.
#else
    // Non-gfx950 device pass: empty stub. See gfx950 branch above.
#endif  // __gfx950__
#endif  // __HIP_DEVICE_COMPILE__
}
