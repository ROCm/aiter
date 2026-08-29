// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Host-side launcher for the gfx1250 a8w8 mxscale BMM with a preshuffled B.
//
// One hand-written tile, deliberately. The gfx950 side of this family has its
// per-kid launchers codegen'd into blob/impl/*.cuh, and that is where this
// should end up too -- but a codegen kid cannot be edited while its kernel body
// is still being written, so the scaffold keeps one tile in source and leaves
// the codegen registration as the follow-up documented at the bottom of this
// file. Host pass only.
//
// This header is INCLUDED BY opus_bmm.cu, which owns the public entry point
// opus_bmm_a8w8_mxscale_bpreshuffle() alongside every other BMM frontend -- the
// same shape as the gfx950 sibling next door. It only forward-declares the
// kernel; the instantiation (and hence the launch stub) lives in
// opus_bmm_a8w8_mxscale_bpreshuffle_gfx1250.cu, which exists for no other
// reason and disappears once the tile is codegen'd.
#ifndef __HIP_DEVICE_COMPILE__
#pragma once

#include "opus_bmm.h"
#include "opus_gemm_arch.cuh"
#include "opus_build_archs.h"
#include "opus_gemm_utils.cuh"      // bf16_t / fp32_t
#include "aiter_stream.h"

#include "opus_bmm_traits_a8w8_mxscale_bpreshuffle_gfx1250.cuh"

// The device kernel is defined in the device pass of
// opus_bmm_a8w8_mxscale_bpreshuffle_gfx1250.cu; declaring it here lets the host
// pass take its address without pulling the pipeline body into every host TU.
template <typename UserTraits>
__global__ void bmm_a8w8_mxscale_bpreshuffle_kernel_gfx1250(
    opus_bmm_a8w8_mxscale_kargs_gfx1250 kargs);

// Shape / dtype / layout validation. Same contract as the gfx950 BMM
// (opus_bmm_a8w8_common_checks) plus the two things the preshuffled B adds.
static inline void opus_bmm_a8w8_mxscale_bpreshuffle_checks_gfx1250(
    aiter_tensor_t& O, aiter_tensor_t& wo_a, aiter_tensor_t& Y,
    aiter_tensor_t& x_scale, aiter_tensor_t& w_scale, const char* who)
{
    aiter_detail::g_aiter_can_throw = true;
    // The AXIS ORDER of the sizes is fixed at (M, batch, K) -- not the memory
    // layout. A physically batch-major [batch,M,K] buffer IS supported and is in
    // fact the normal case: every caller in op_tests/test_opus_a8w8_bmm.py holds
    // [g,m,k] and passes `.transpose(0,1)`, which is a free view. The strides
    // then carry the real layout (stride_a = stride(0), stride_a_batch =
    // stride(1)), so nothing is copied and nothing is assumed contiguous except
    // K. The order cannot be inferred instead of fixed: [M,batch,K] and
    // [batch,M,K] are both rank 3, so guessing would silently transpose the
    // problem whenever M and batch are interchangeable. Same contract as the
    // gfx950 sibling (opus_bmm_a8w8_common_checks), on purpose -- the two share
    // a python dispatch.
    AITER_CHECK(O.dim() == 3 && wo_a.dim() == 3 && Y.dim() == 3, who,
                ": O/wo_a/Y must be 3D, sizes ordered "
                "([M,batch,K] / [batch,N,K] / [M,batch,N]); a batch-major "
                "[batch,M,K] tensor is passed as .transpose(0,1)");
    AITER_CHECK(O.dtype() == AITER_DTYPE_fp8 && wo_a.dtype() == AITER_DTYPE_fp8,
                who, ": O and wo_a must be fp8");
    AITER_CHECK(Y.dtype() == AITER_DTYPE_fp32 || Y.dtype() == AITER_DTYPE_bf16,
                who, ": Y must be fp32 or bf16");
    // A is indexed along K with unit stride (kargs carries no K stride), so K
    // must be innermost. The batch axis position is free -- stride_a_batch
    // describes it fully.
    AITER_CHECK(O.stride(2) == 1, who,
                ": O (x) must be K-contiguous (stride(2)==1); got ", (long)O.stride(2));
    // wo_a is the OUTPUT of shuffle_weight(w, layout=(16,16)), i.e. an opaque
    // contiguous blob whose logical [batch,N,K] shape is preserved but whose
    // interior order is not row-major. Checking stride(2)==1 on it would be
    // meaningless (and would pass for the WRONG buffer); what matters is that it
    // is contiguous and the right size.
    AITER_CHECK(wo_a.is_contiguous(), who,
                ": wo_a must be the contiguous shuffle_weight(w, (16,16)) buffer");
    const int n = (int)Y.size(2);
    const int k = (int)O.size(2);
    AITER_CHECK(n % 16 == 0, who,
                ": N must be a multiple of 16 (shuffle_weight block); got ", n);
    AITER_CHECK(k % 32 == 0, who,
                ": K must be a multiple of 32 (shuffle_weight block); got ", k);
    AITER_CHECK(x_scale.dtype() == AITER_DTYPE_fp8_e8m0 ||
                x_scale.element_size() == 1, who, ": x_scale must be e8m0 bytes");
    AITER_CHECK(w_scale.dtype() == AITER_DTYPE_fp8_e8m0 ||
                w_scale.element_size() == 1, who, ": w_scale must be e8m0 bytes");
}

// One tile, one launch. Traits is a full
// opus_bmm_a8w8_mxscale_bpreshuffle_traits_gfx1250<...> instantiation.
template <typename Traits>
static inline void opus_bmm_a8w8_mxscale_bpreshuffle_launch_gfx1250(
    aiter_tensor_t& O, aiter_tensor_t& wo_a, aiter_tensor_t& Y,
    aiter_tensor_t& x_scale, aiter_tensor_t& w_scale, int splitK)
{
    using T = Traits;
    opus_bmm_a8w8_mxscale_bpreshuffle_checks_gfx1250(
        O, wo_a, Y, x_scale, w_scale, "opus_bmm_a8w8_mxscale_bpreshuffle");

    const int m     = (int)O.size(0);
    const int batch = (int)O.size(1);
    const int k     = (int)O.size(2);
    const int n     = (int)Y.size(2);

    // Split-K is not wired in the scaffold (no fp32 workspace, no reduce kernel).
    // Accept only 1 rather than silently ignoring the request -- a silently
    // dropped splitK returns plausible-but-wrong-looking timings, not an error.
    AITER_CHECK(splitK <= 1,
                "opus_bmm_a8w8_mxscale_bpreshuffle: splitK>1 is not implemented "
                "yet (got ", splitK, "); add the ws+reduce path from "
                "opus_gemm_pipeline_a16w16_cluster_tdm_splitk_ws_gfx1250.cuh");

    // w_scale's SHAPE is the only thing that distinguishes a per-column scale
    // from a 128x128 block scale, and the tile's GROUP_N picks which one it will
    // read. Nothing else in the call signature reveals the difference: both are
    // contiguous e8m0 with dim 3, and indexing a per-column scale as if it were
    // blocked (or the reverse) stays in bounds and returns a real exponent for
    // the wrong columns. That is a silent wrong answer, so check it here.
    const int sfb_rows = (n + T::kGroupN - 1) / T::kGroupN;
    const int sfb_cols = k / T::kGroupK;
    AITER_CHECK(w_scale.dim() == 3, "opus_bmm_a8w8_mxscale_bpreshuffle",
                ": w_scale must be 3D [batch, N/GROUP_N, K/GROUP_K]; got dim ",
                w_scale.dim());
    AITER_CHECK((int)w_scale.size(1) == sfb_rows,
                "opus_bmm_a8w8_mxscale_bpreshuffle: this tile has GROUP_N=",
                T::kGroupN, ", so w_scale.size(1) must be ceil(N/GROUP_N)=",
                sfb_rows, " for N=", n, "; got ", (long)w_scale.size(1),
                ". A 128x128 block scale needs a GROUP_N=128 tile and a "
                "per-column scale needs a GROUP_N=1 tile.");
    AITER_CHECK((int)w_scale.size(2) == sfb_cols,
                "opus_bmm_a8w8_mxscale_bpreshuffle: w_scale.size(2) must be "
                "K/GROUP_K=", sfb_cols, " for K=", k, " GROUP_K=", T::kGroupK,
                "; got ", (long)w_scale.size(2));

    // A tile that stages A's scales in LDS allocates that panel at compile time
    // but fills it with the RUNTIME K/GROUP_K as its row pitch, so a K past the
    // bound does not truncate -- it runs the fill off the end of the panel and
    // corrupts whatever LDS follows. Silent, and not even confined to the scale
    // path. Hence a hard error rather than a fallback.
    if constexpr (T::kSfALds || T::kSfBLds) {
        // The TDM fill is the tighter bound of the two: its width is the D#'s
        // pad interval and so is fixed at compile time, which means a larger K
        // does not merely overrun the panel, it silently drops the K-groups past
        // the descriptor's tile.
        constexpr int kg_cap = T::kSfATdm ? T::kSfATdmKG : T::kSfAPanelKG;
        AITER_CHECK(k <= kg_cap * T::kGroupK,
                    "opus_bmm_a8w8_mxscale_bpreshuffle: this tile stages scales "
                    "in LDS, which caps K at ",
                    kg_cap * T::kGroupK, "; got K=", k,
                    ". Use a tile without SF_A_LDS/SF_B_LDS for larger K.");
    }

    opus_bmm_a8w8_mxscale_kargs_gfx1250 kargs{};
    kargs.ptr_a   = O.data_ptr();
    kargs.ptr_b   = wo_a.data_ptr();
    kargs.ptr_c   = Y.data_ptr();
    kargs.ptr_sfa = x_scale.data_ptr();
    kargs.ptr_sfb = w_scale.data_ptr();
    kargs.m       = m;
    kargs.n       = n;
    kargs.k       = k;
    kargs.batch   = batch;
    kargs.split_k = 1;
    kargs.stride_a = (int)O.stride(0);
    kargs.stride_b = k;                       // unshuffled row pitch; see traits
    kargs.stride_c = (int)Y.stride(0);
    kargs.stride_a_batch = (int)O.stride(1);
    kargs.stride_b_batch = (int)wo_a.stride(0);
    kargs.stride_c_batch = (int)Y.stride(1);
    // Scale strides follow the SAME axis convention as the operands, and the two
    // sides are NOT symmetric: x_scale rides with A as [M,batch,K/GROUP_K] (so
    // the per-row stride is stride(0) and the batch stride is stride(1)), while
    // w_scale rides with B as [batch,N,K/GROUP_K] (per-row stride(1), batch
    // stride(0)). Transcribed from gen_instances_gfx950.py's launcher body
    // rather than derived, because getting these two swapped reads a valid
    // in-bounds scale for the wrong row and produces plausible wrong numbers.
    kargs.stride_sfa       = (int)x_scale.stride(0);
    kargs.stride_sfa_batch = (int)x_scale.stride(1);
    kargs.stride_sfb       = (int)w_scale.stride(1);
    kargs.stride_sfb_batch = (int)w_scale.stride(0);

    dim3 grid((unsigned)((m + T::kBlockM - 1) / T::kBlockM),
              (unsigned)((n + T::kBlockN - 1) / T::kBlockN),
              (unsigned)batch);
    dim3 block((unsigned)T::BLOCK_SIZE);
    hipStream_t stream = aiter::getCurrentHIPStream();
    bmm_a8w8_mxscale_bpreshuffle_kernel_gfx1250<T><<<grid, block, 0, stream>>>(kargs);
}

// FOLLOW-UP (not part of the scaffold): move this launcher into codegen.
// csrc/opus_gemm/codegen/gen_instances_gfx1250.py already self-registers five
// arch maps keyed by kernel_tag (PIPELINE_HEADER_MAP / TRAITS_HEADER_MAP /
// KERNEL_FUNC_MAP / TRAITS_NAME_MAP / KARGS_NAME_MAP) plus register_emit(); a
// new "bmm_a8w8_mxscale_bpreshuffle" tag pointing at the two headers above,
// with an emit fn shaped like gen_cluster_tdm_splitk_ws_instance, replaces this
// file and gives the tile a kid id the tuner can reach. Do that once the kernel
// body is settled -- not before, because every regenerated kid has to be
// rebuilt and `rule cuda_compile` has no depfile.

#endif // !__HIP_DEVICE_COMPILE__
