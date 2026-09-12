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

// The non-specialized variant is a distinct symbol with an identical signature;
// see opus_bmm_pipeline_a8w8_mxscale_bpreshuffle_nospec_gfx1250.cuh.
template <typename UserTraits>
__global__ void bmm_a8w8_mxscale_bpreshuffle_nospec_kernel_gfx1250(
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

// Non-specialized launch. The checks, the kargs and the grid are the same as the
// launcher above -- only the kernel symbol differs. The two are kept as separate
// functions rather than one parameterised by symbol because the K cap the
// specialized launcher enforces is an SF_A_LDS/SF_B_LDS property, and no NO_SPEC_
// tile carries those flags today; folding them together would make that
// coupling implicit. If a NO_SPEC_ tile ever takes a scale panel, move the cap
// check here too.
template <typename T>
static inline void opus_bmm_a8w8_mxscale_bpreshuffle_nospec_launch_gfx1250(
    aiter_tensor_t& O, aiter_tensor_t& wo_a, aiter_tensor_t& Y,
    aiter_tensor_t& x_scale, aiter_tensor_t& w_scale, int splitK)
{
    static_assert(T::kNoSpec,
                  "the nospec launcher takes a NO_SPEC_ tile; a specialized "
                  "tile launched through it would run with no producer waves "
                  "and read uninitialised LDS");
    opus_bmm_a8w8_mxscale_bpreshuffle_checks_gfx1250(
        O, wo_a, Y, x_scale, w_scale,
        "opus_bmm_a8w8_mxscale_bpreshuffle_nospec");
    AITER_CHECK(splitK == 1,
                "opus_bmm_a8w8_mxscale_bpreshuffle_nospec: splitK must be 1");

    const int m = (int)O.size(0), batch = (int)O.size(1), k = (int)O.size(2);
    const int n = (int)wo_a.size(1);

    opus_bmm_a8w8_mxscale_kargs_gfx1250 kargs{};
    kargs.ptr_a   = O.data_ptr();
    kargs.ptr_b   = wo_a.data_ptr();
    kargs.ptr_c   = Y.data_ptr();
    kargs.ptr_sfa = x_scale.data_ptr();
    kargs.ptr_sfb = w_scale.data_ptr();
    kargs.m = m; kargs.n = n; kargs.k = k; kargs.batch = batch; kargs.split_k = 1;
    kargs.stride_a = (int)O.stride(0);
    kargs.stride_b = k;
    kargs.stride_c = (int)Y.stride(0);
    kargs.stride_a_batch = (int)O.stride(1);
    kargs.stride_b_batch = (int)wo_a.stride(0);
    kargs.stride_c_batch = (int)Y.stride(1);
    kargs.stride_sfa       = (int)x_scale.stride(0);
    kargs.stride_sfa_batch = (int)x_scale.stride(1);
    kargs.stride_sfb       = (int)w_scale.stride(1);
    kargs.stride_sfb_batch = (int)w_scale.stride(0);

    dim3 grid((unsigned)((m + T::kBlockM - 1) / T::kBlockM),
              (unsigned)((n + T::kBlockN - 1) / T::kBlockN),
              (unsigned)batch);
    dim3 block((unsigned)T::BLOCK_SIZE);
    hipStream_t stream = aiter::getCurrentHIPStream();
    bmm_a8w8_mxscale_bpreshuffle_nospec_kernel_gfx1250<T>
        <<<grid, block, 0, stream>>>(kargs);
}

// ===========================================================================
// CLUSTER-LAUNCH, FUSED SPLIT-K variant.
// ===========================================================================
// Kernel:  opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_clusterclaunch_gfx1250.cuh
// Kargs:   opus_gemm_cluster_claunch_kargs_gfx1250 (traits header)
//
// Same tile traits as the launcher above -- the split is entirely an epilogue
// and grid-shape change -- plus four compile-time knobs:
//
//   SplitK      k-slices per output tile. Each slice is one WORKGROUP, and all
//               SplitK of them are one CLUSTER (cluster.x), so they co-reside
//               and can synchronise with a cluster barrier instead of a
//               semaphore. SplitK-1 of them publish a partial; the last sums.
//   MClusterWg  adjacent M-TILES folded into the same cluster (cluster.y). Those
//               peers share one N-tile and one batch, so their B (weight) load
//               is TDM-multicast: one global fetch lands in all their LDS.
//               1 = no multicast.
//   DataWs      partial-workspace element type. bf16 halves the workspace
//               traffic; fp32 keeps full precision per partial. The reduce
//               accumulator is fp32 either way, so this trades ONLY the
//               per-partial storage rounding.
//   D_OUT       C element type (bf16_t or fp32_t), independent of the tile
//               alias's D_C: the epilogue holds fp32 to the end and casts once.
//
// SplitK * MClusterWg <= 16 (cluster WG budget), enforced in the kernel.

// Defined in the device pass, like the non-split kernel above.
template <typename UserTraits, int SplitK, typename DataWs, int MClusterWg, typename D_OUT>
__global__ void gemm_a8w8_mxscale_bpreshuffle_clusterclaunch_kernel_gfx1250(
    opus_gemm_cluster_claunch_kargs_gfx1250 kargs);

// Partial-workspace size, in ELEMENTS of DataWs. Call this to size the
// torch.empty the caller passes as `ws`; there is no internal allocation,
// deliberately, because the buffer is reused across a decode loop's launches
// and a per-launch hipMalloc would dominate the kernel it feeds.
//
// SplitK-1, not SplitK: the last split keeps its own partial in registers and
// never round-trips it. SplitK == 1 needs no workspace at all.
template <typename Traits, int SplitK>
static inline size_t opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch_ws_elems_gfx1250(
    int m, int n, int batch)
{
    if constexpr (SplitK <= 1) {
        (void)m; (void)n; (void)batch;
        return 0;
    } else {
        using T = Traits;
        const size_t ntm = (size_t)((m + T::kBlockM - 1) / T::kBlockM);
        const size_t ntn = (size_t)((n + T::kBlockN - 1) / T::kBlockN);
        return ntm * ntn * (size_t)batch * (size_t)(SplitK - 1)
               * (size_t)T::kBlockM * (size_t)T::kBlockN;
    }
}

// `ws` may be null iff SplitK == 1; `bias` may always be null.
template <typename Traits, int SplitK, typename DataWs, int MClusterWg, typename D_OUT>
static inline void opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch_launch_gfx1250(
    aiter_tensor_t& O, aiter_tensor_t& wo_a, aiter_tensor_t& Y,
    aiter_tensor_t& x_scale, aiter_tensor_t& w_scale,
    aiter_tensor_t* ws, aiter_tensor_t* bias, int splitK)
{
    using T = Traits;
    static_assert(SplitK >= 1, "SplitK must be >= 1");
    static_assert(MClusterWg >= 1, "MClusterWg must be >= 1");
    // Per-DIMENSION bound as well as the product: __cluster_dims__ encodes each
    // extent in 4 bits, so 16x1 is rejected by the compiler even though the
    // cluster budget is 16 workgroups. Use 8x2 or 4x4 for a full cluster.
    static_assert(SplitK <= 15 && MClusterWg <= 15,
                  "each cluster dimension must be <= 15 (4-bit field); a 16-WG "
                  "cluster must be 8x2 or 4x4");
    static_assert(SplitK * MClusterWg <= 16,
                  "cluster WG count (SplitK*MClusterWg) must be <= 16");

    opus_bmm_a8w8_mxscale_bpreshuffle_checks_gfx1250(
        O, wo_a, Y, x_scale, w_scale,
        "opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch");

    const int m     = (int)O.size(0);
    const int batch = (int)O.size(1);
    const int k     = (int)O.size(2);
    const int n     = (int)Y.size(2);

    // splitK is a RUNTIME argument on the shared BMM entry point but a
    // COMPILE-TIME cluster dimension here, so the two must agree exactly. A
    // mismatch cannot be honoured (the cluster geometry is baked into the
    // kernel object) and must not be silently ignored -- a dropped splitK
    // returns a correct answer at the wrong speed, which reads as a tuning
    // result rather than a bug.
    AITER_CHECK(splitK == SplitK,
                "opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch: this "
                "instantiation is compiled for splitK=", SplitK,
                " but was called with splitK=", splitK);

    // D_OUT is what the kernel actually stores; Y's dtype is what the caller
    // allocated. Disagreement here writes bf16 into an fp32 buffer (or the
    // reverse) with no fault and no wrong-looking output size.
    if constexpr (std::is_same<D_OUT, bf16_t>::value) {
        AITER_CHECK(Y.dtype() == AITER_DTYPE_bf16,
                    "opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch: this "
                    "instantiation writes bf16 C, but Y is not bf16");
    } else {
        AITER_CHECK(Y.dtype() == AITER_DTYPE_fp32,
                    "opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch: this "
                    "instantiation writes fp32 C, but Y is not fp32");
    }

    // EVERY split must own at least one B_K tile. The kernel splits the tiles
    // BALANCED (first k_rem splits take one extra), which guarantees that only
    // while SplitK <= ceil(K/B_K); past that some split gets zero tiles, and a
    // zero-tile split still has to publish a partial its peers will sum. Rather
    // than teach the kernel to publish zeros, refuse the launch: a SplitK
    // greater than the K-tile count has no parallelism to win anyway.
    const int k_steps_tot = (k + T::kBlockK - 1) / T::kBlockK;
    AITER_CHECK(SplitK <= k_steps_tot,
                "opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch: splitK=",
                SplitK, " exceeds the K-tile count ceil(K/B_K)=", k_steps_tot,
                " for K=", k, " B_K=", T::kBlockK,
                "; every split must own at least one K tile");

    const int num_tiles_m = (m + T::kBlockM - 1) / T::kBlockM;
    const int num_tiles_n = (n + T::kBlockN - 1) / T::kBlockN;

    // Same w_scale shape / LDS-panel-K checks as the non-split launcher; the
    // split changes neither the scale layout nor the panel, because both panels
    // still cover the tile's WHOLE K range on every split.
    const int sfb_rows = (n + T::kGroupN - 1) / T::kGroupN;
    const int sfb_cols = k / T::kGroupK;
    AITER_CHECK(w_scale.dim() == 3,
                "opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch",
                ": w_scale must be 3D [batch, N/GROUP_N, K/GROUP_K]; got dim ",
                w_scale.dim());
    AITER_CHECK((int)w_scale.size(1) == sfb_rows,
                "opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch: this tile "
                "has GROUP_N=", T::kGroupN, ", so w_scale.size(1) must be "
                "ceil(N/GROUP_N)=", sfb_rows, " for N=", n, "; got ",
                (long)w_scale.size(1));
    AITER_CHECK((int)w_scale.size(2) == sfb_cols,
                "opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch: "
                "w_scale.size(2) must be K/GROUP_K=", sfb_cols, " for K=", k,
                " GROUP_K=", T::kGroupK, "; got ", (long)w_scale.size(2));
    if constexpr (T::kSfALds || T::kSfBLds) {
        constexpr int kg_cap = T::kSfATdm ? T::kSfATdmKG : T::kSfAPanelKG;
        AITER_CHECK(k <= kg_cap * T::kGroupK,
                    "opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch: this "
                    "tile stages scales in LDS, which caps K at ",
                    kg_cap * T::kGroupK, "; got K=", k);
    }

    // Workspace. Sized in DataWs elements but validated in BYTES, because the
    // caller is free to hand over any dtype's buffer (a byte/uint8 scratch is
    // the common case) as long as it is big enough and 16B-aligned for the
    // dwordx4 traffic.
    void* ptr_ws = nullptr;
    if constexpr (SplitK > 1) {
        const size_t need_elems =
            opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch_ws_elems_gfx1250<T, SplitK>(
                m, n, batch);
        const size_t need_bytes = need_elems * sizeof(DataWs);
        AITER_CHECK(ws != nullptr && ws->data_ptr() != nullptr,
                    "opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch: splitK=",
                    SplitK, " needs a partial workspace of ", (long long)need_bytes,
                    " bytes; none was passed");
        const size_t have_bytes = ws->numel() * ws->element_size();
        AITER_CHECK(have_bytes >= need_bytes,
                    "opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch: workspace "
                    "is ", (long long)have_bytes, " bytes but this launch needs ",
                    (long long)need_bytes,
                    " (num_tiles_m*num_tiles_n*batch*(splitK-1)*B_M*B_N elements "
                    "of the DataWs dtype)");
        AITER_CHECK((reinterpret_cast<uintptr_t>(ws->data_ptr()) & 15u) == 0,
                    "opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch: workspace "
                    "must be 16B-aligned (the partial store/reduce is dwordx4)");
        ptr_ws = ws->data_ptr();
    } else {
        (void)ws;
    }

    // Bias is optional and, when present, bf16 [N] -- the kernel reads it as
    // bf16 unconditionally, so an fp32 bias would be silently misread.
    const void* ptr_bias   = nullptr;
    int stride_bias_batch  = 0;
    if (bias != nullptr && bias->data_ptr() != nullptr) {
        AITER_CHECK(bias->dtype() == AITER_DTYPE_bf16,
                    "opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch: bias must "
                    "be bf16");
        AITER_CHECK(bias->dim() == 1 || bias->dim() == 2,
                    "opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch: bias must "
                    "be [N] or [batch, N]");
        if (bias->dim() == 1) {
            AITER_CHECK((int)bias->size(0) == n,
                        "opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch: bias "
                        "[N] must have N=", n, "; got ", (long)bias->size(0));
            stride_bias_batch = 0;          // one bias broadcast over batch
        } else {
            AITER_CHECK((int)bias->size(0) == batch && (int)bias->size(1) == n,
                        "opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch: bias "
                        "[batch, N] must be [", batch, ", ", n, "]");
            stride_bias_batch = (int)bias->stride(0);
        }
        ptr_bias = bias->data_ptr();
    }

    opus_gemm_cluster_claunch_kargs_gfx1250 kargs{};
    kargs.ptr_a    = O.data_ptr();
    kargs.ptr_b    = wo_a.data_ptr();
    kargs.ptr_c    = Y.data_ptr();
    kargs.ptr_sfa  = x_scale.data_ptr();
    kargs.ptr_sfb  = w_scale.data_ptr();
    kargs.ptr_ws   = ptr_ws;
    kargs.ptr_bias = ptr_bias;
    kargs.m        = m;
    kargs.n        = n;
    kargs.k        = k;
    kargs.batch    = batch;
    kargs.split_k  = SplitK;
    kargs.stride_a = (int)O.stride(0);
    kargs.stride_b = k;                       // unshuffled row pitch; see traits
    kargs.stride_c = (int)Y.stride(0);
    kargs.stride_a_batch = (int)O.stride(1);
    kargs.stride_b_batch = (int)wo_a.stride(0);
    kargs.stride_c_batch = (int)Y.stride(1);
    // Asymmetric on purpose, same as the non-split launcher: x_scale rides with
    // A as [M, batch, K/GROUP_K] (row stride(0), batch stride(1)); w_scale rides
    // with B as [batch, N, K/GROUP_K] (row stride(1), batch stride(0)).
    kargs.stride_sfa       = (int)x_scale.stride(0);
    kargs.stride_sfa_batch = (int)x_scale.stride(1);
    kargs.stride_sfb       = (int)w_scale.stride(1);
    kargs.stride_sfb_batch = (int)w_scale.stride(0);
    kargs.stride_bias_batch = stride_bias_batch;
    kargs.num_tiles_m      = num_tiles_m;
    kargs.num_tiles_n      = num_tiles_n;

    // GRID, in workgroups, matching the kernel's decomposition exactly:
    //   x = SplitK                                   (cluster.x, the k-slices)
    //   y = num_tiles_m rounded UP to MClusterWg     (cluster.y, the M-peers)
    //   z = num_tiles_n * batch                      (gn fastest, then batch)
    //
    // The round-up in y is what makes the launch legal for any num_tiles_m: a
    // cluster is an indivisible allocation unit, so grid.y must be a multiple of
    // MClusterWg. The surplus peers are real workgroups that run the whole
    // pipeline against an out-of-range window and drop their C store -- the
    // kernel's m_oob path -- rather than exiting, because a peer that exits
    // takes its cluster's barrier quorum with it.
    const unsigned grid_y =
        (unsigned)(((num_tiles_m + MClusterWg - 1) / MClusterWg) * MClusterWg);
    dim3 grid((unsigned)SplitK, grid_y, (unsigned)((size_t)num_tiles_n * (size_t)batch));
    dim3 block((unsigned)T::BLOCK_SIZE);
    hipStream_t stream = aiter::getCurrentHIPStream();
    // Plain <<<>>>: the cluster geometry is a COMPILE-TIME __cluster_dims__
    // attribute on the kernel, not a launch-time argument, so there is no
    // hipExtLaunch here and grid stays a workgroup count.
    gemm_a8w8_mxscale_bpreshuffle_clusterclaunch_kernel_gfx1250<T, SplitK, DataWs,
                                                                MClusterWg, D_OUT>
        <<<grid, block, 0, stream>>>(kargs);
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
