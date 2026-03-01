// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Self-contained MOE sorting kernel for opus path.
// Contains: type definitions, problem definitions, kernel code, and API dispatch.
// No external kernel library dependency.

#pragma once
#include <torch/extension.h>

void moe_sorting_opus_fwd(torch::Tensor &topk_ids,
                          torch::Tensor &topk_weights,
                          torch::Tensor &sorted_token_ids,
                          torch::Tensor &sorted_weights,
                          torch::Tensor &sorted_expert_ids,
                          torch::Tensor &num_valid_ids,
                          torch::Tensor &moe_buf,
                          int num_experts,
                          int unit_size,
                          std::optional<torch::Tensor> local_expert_mask = std::nullopt,
                          std::optional<torch::Tensor> num_local_tokens = std::nullopt,
                          int dispatch_policy = 0);

#ifdef MOE_SORTING_OPUS_IMPL
// ============================================================================
// Implementation section - only compiled in the .cu translation unit
// ============================================================================

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <string>
#include <stdexcept>

#include "opus/opus.hpp"

#ifndef OPUS_MOE_SORTING_MOCK_ID
#define OPUS_MOE_SORTING_MOCK_ID 1
#endif

#ifndef OPUS_HAS_ROW_NEWBCAST
#if defined(__HIP_DEVICE_COMPILE__) && defined(__HIP_PLATFORM_AMD__)
#if defined(__gfx908__) || defined(__gfx906__) || defined(__gfx900__)
#define OPUS_HAS_ROW_NEWBCAST 0
#else
#define OPUS_HAS_ROW_NEWBCAST 1
#endif
#else
#define OPUS_HAS_ROW_NEWBCAST 0
#endif
#endif

#ifndef OPUS_WA_ISSUE_2028
#define OPUS_WA_ISSUE_2028 0
#endif

#ifndef MOE_SORTING_FUSE_MP_01
#define MOE_SORTING_FUSE_MP_01 1
#endif

namespace aiter {

// ---------------------------------------------------------------------------
// Math utilities
template <typename X, typename Y>
OPUS_H_D constexpr auto integer_divide_ceil(X a, Y b)
{
    return (a + b - 1) / b;
}

template <typename X, typename Y>
OPUS_H_D constexpr auto integer_least_multiple(X x, Y y)
{
    return y * integer_divide_ceil(x, y);
}

// ---------------------------------------------------------------------------
// HIP error checking
#define OPUS_HIP_CHECK_ERROR(expr)                                                       \
    do {                                                                             \
        hipError_t __e = (expr);                                                    \
        if (__e != hipSuccess)                                                      \
            throw std::runtime_error(std::string("HIP error: ") +                  \
                                     hipGetErrorString(__e));                       \
    } while (0)

// ---------------------------------------------------------------------------
// stream_config: for kernel launch
struct stream_config
{
    hipStream_t stream_id_ = nullptr;
    bool time_kernel_      = false;
    int log_level_         = 0;
    int cold_niters_       = 0;
    int nrepeat_           = 1;
};

// ---------------------------------------------------------------------------
// Kernel launch infrastructure
template <typename Kernel, typename Kargs>
__global__ void __launch_bounds__(1024) opus_moe_sorting_entry(Kargs kargs)
{
    Kernel{}.operator()(kargs);
}

template <typename Kernel>
struct packaged_kernel
{
    using Kargs = typename Kernel::Kargs;
    dim3     grids;
    dim3     blocks;
    uint32_t lds_bytes;
    Kargs    kargs;

    void operator()(const stream_config& s) const
    {
        auto* func = &opus_moe_sorting_entry<Kernel, Kargs>;
        if (lds_bytes > 0)
            (void)hipFuncSetAttribute(reinterpret_cast<const void*>(func),
                                      hipFuncAttributeMaxDynamicSharedMemorySize, lds_bytes);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(opus_moe_sorting_entry<Kernel, Kargs>),
                           grids, blocks, lds_bytes, s.stream_id_, kargs);
    }
};

template <opus::index_t BlockSize = 0, typename Kernel>
auto make_kernel(Kernel, dim3 grids, dim3 blocks, uint32_t lds, typename Kernel::Kargs kargs)
{
    (void)opus::number<BlockSize>{};
    return packaged_kernel<Kernel>{grids, blocks, lds, kargs};
}

inline float launch_kernel(const stream_config& s)
{
    (void)s;
    return 0.0f;
}

template <typename K0, typename... Ks>
float launch_kernel(const stream_config& s, K0&& k0, Ks&&... ks)
{
    if constexpr (std::is_invocable_v<K0, const stream_config&>)
        k0(s);
    launch_kernel(s, std::forward<Ks>(ks)...);
    return 0.0f;
}

} // namespace aiter


// --- Problem definitions ---
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT


#include <string>
#include <type_traits>

namespace aiter {

template <typename IndexType_,
          typename WeightType_,
          opus::index_t InternalLoadUnroll_,
          opus::index_t ExpertTile_ = 0>
struct MoeSortingProblem
{
    // TODO: this kernel only support warp per row
    using WeightType = opus::remove_cvref_t<WeightType_>;
    using IndexType  = opus::remove_cvref_t<IndexType_>;

    static constexpr opus::index_t WarpsPerBlock = 1;
    static constexpr opus::index_t InternalLoadUnroll =
        InternalLoadUnroll_;                           // TODO: need better design(like tile size)
    static constexpr opus::index_t ExpertTile = ExpertTile_; // TODO: only used in store out
};

template <typename IndexType_,
          typename WeightType_,
          opus::index_t SubTokenTile_,    // 1,2,4,8, or 0 in the future
          bool SubTokenOneShot_,    // if we only loop over once or not
          bool LocalExpertMasking_, // used in EP case
          bool LocalToken_,         // used in EP case
          bool SkipExpertsWithZeroTokens_ = true,
          opus::index_t ExpertTile_             = 0>
struct MoeSortingProblemEx
{
    // TODO: this kernel only support warp per row
    using WeightType = opus::remove_cvref_t<WeightType_>;
    using IndexType  = opus::remove_cvref_t<IndexType_>;

    static constexpr opus::index_t WarpsPerBlock          = 1;
    static constexpr opus::index_t SubTokenTile           = SubTokenTile_;
    static constexpr bool SubTokenOneShot           = SubTokenOneShot_;
    static constexpr bool LocalExpertMasking        = LocalExpertMasking_;
    static constexpr bool LocalToken                = LocalToken_;
    static constexpr bool SkipExpertsWithZeroTokens = SkipExpertsWithZeroTokens_;
    static_assert(SubTokenTile == 1 || SubTokenTile == 2 || SubTokenTile == 4 || SubTokenTile == 8);
    static constexpr opus::index_t ExpertTile = ExpertTile_; // TODO: only used in store out
};

template <typename IndexType_,
          typename WeightType_, // used for expert mesh in ws
          typename MeshType_,
          opus::index_t SubTokenTile_,    // 1,2,4,8
          bool LocalExpertMasking_, // used in EP case
          bool LocalToken_,         // used in EP case
          bool SkipExpertsWithZeroTokens_ = true>
struct MoeSortingProblemMp
{
    // TODO: this kernel only support warp per row
    using WeightType = opus::remove_cvref_t<WeightType_>;
    using MeshType   = opus::remove_cvref_t<MeshType_>;
    using IndexType  = opus::remove_cvref_t<IndexType_>;

    static constexpr opus::index_t SubTokenTile           = SubTokenTile_;
    static constexpr bool LocalExpertMasking        = LocalExpertMasking_;
    static constexpr bool LocalToken                = LocalToken_;
    static constexpr bool SkipExpertsWithZeroTokens = SkipExpertsWithZeroTokens_;
    static_assert(SubTokenTile == 1 || SubTokenTile == 2 || SubTokenTile == 4 ||
                  SubTokenTile == 8 || SubTokenTile == 16);
};

template <bool LocalToken_, opus::index_t BlockSize_ = 1024, opus::index_t Occu_ = 1>
struct MoeSortingClearWorkspaceProblem
{
    static constexpr bool LocalToken   = LocalToken_;
    static constexpr opus::index_t BlockSize = BlockSize_;
    static constexpr opus::index_t Occu      = Occu_;
};

} // namespace aiter


// --- Kernel implementation ---
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT


#include <string>
#include <type_traits>

#if !defined(OPUS_HAS_ROW_NEWBCAST)
// row_newbcast (DPP modifier 0x157) support by architecture:
// - Not supported: gfx908 (MI100) and older
// - Supported: gfx90a (MI200), gfx94x (MI300), and all RDNA architectures

#if defined(__HIP_DEVICE_COMPILE__) && defined(__HIP_PLATFORM_AMD__)
#if defined(__gfx908__) || defined(__gfx906__) || defined(__gfx900__)
// Explicitly disable for known unsupported architectures
#define OPUS_HAS_ROW_NEWBCAST 0
#else
// Assume support for gfx90a and newer (including all gfx94x and RDNA)
// This is safer as new architectures typically maintain backward compatibility
#define OPUS_HAS_ROW_NEWBCAST 1
#endif
#else
// Conservative default for non-AMD or host compilation
#define OPUS_HAS_ROW_NEWBCAST 0
#endif
#endif

namespace aiter {

#define MOE_SORTING_MOCK_ID(token_id_, topk_id_) \
    static_cast<uint32_t>(((token_id_) & 0x00ffffff) | (((topk_id_) & 0xff) << 24))

#ifndef MOE_SORTING_FUSE_MP_01
#define MOE_SORTING_FUSE_MP_01 1
#endif

// clang-format off
// [indexing implementation-1]
// using M_a as constexpr block_size to partition all tokens into different slices
// each slice map to one expert, and one expert can have multiple slices
// e.g. num_experts = 6, topk=3, M_a = 4, input_tokens = 5
// before sort, topk_ids is : [[0, 3, 5], [2, 3, 5], [1, 3, 5], [1, 2, 3], [1, 3, 5]]
//                            tok-0      tok-1      tok-2      tok-3      tok-4
//           topk_weight is : [[a, b, c], [d, e, f], [g, h, i], [j, k, l], [m, n, o]] (some float number)
//
// token_id_per_expert is : [[0], [2, 3, 4], [1, 3], [0, 1, 2, 3, 4], [], [0, 1, 2, 5]]
//  (only for reference)    exp-0  exp-1     exp-2   exp-3          exp-4  exp-5
// weight_id_per_expert is: [[a], [g, j, m], [d, k], [b, e, h, l, n], [], [c, f, i, o]]
//
// max_num_tokens_padded : topk * input_tokens + num_experts * M_a - topk (updated)
// * this could be larger than actual, since actual tokens are on GPU
//
// sorted_token_ids_ptr   : [0, 6, 6, 6, 2, 3, 4, 6, 1, 3, 6, 6, 0, 1, 2, 3, 4, 6, 6, 6, 6, 6, 6, 6, 0, 1, 2, 5]
//                          |-  exp-0  -|-  exp-1  -|-  exp-2  -|-      exp-3          -|-  exp-4 -|-  exp-5  -|
// sorted_weight_ptr      : [a, *, *, *, g, j, m, *, d, k, *, *, b, e, h, l, n, *, *, *, *, *, *, *, c, f, i, o]
//
// * length is max_num_tokens_padded, actual size is num_tokens_post_padded_ptr
//
// * Note on token_id_per_expert/sorted_token_ids_ptr data:
// currently we do not have topk information from the data of token_id_per_expert/sorted_token_ids_ptr.
// In some cases(like smooth-quant), we need topk information to indexing into tokens quant from
// different expert smooth quant. So we modify the number stored inside token_id_per_expert/sorted_token_ids_ptr
//
//       32bit    0........23 24.....31 bit
//      (data) -> (token_id | topk_id)
// low 24 bit is for token id, top 8 bit is for topk id
//
// the input after smooth-quant is [topk, token, hidden_dim], originally it is [token, hidden_dim]
// the input scale for token is [topk, token, 1], the smooth-quant scale for first gemm is [expert, interm_dim]
//
// sorted_expert_ids_ptr  : [0, 1, 2, 3, 3, 4, 5]
// * length is (max_num_tokens_padded + block_size - 1) / block_size
//
// num_tokens_post_padded_ptr : [28]
// num_sorted_tiles_ptr : [7]
//
// skip_experts_with_zero_tokens(SkipExpertsWithZeroTokens)
// if enabled, the expert with no tokens will be skipped, in stead of padding to at least 1 unit_size(M_a)
//
//                                            (pack below tensor, skip element marked with `-`)
//                           Y  Y  Y  Y  Y  Y  Y  Y  Y  Y  Y  Y  Y  Y  Y  Y  Y  Y  Y  Y  -  -  -  -  Y  Y  Y  Y
// sorted_token_ids_ptr   : [0, 6, 6, 6, 2, 3, 4, 6, 1, 3, 6, 6, 0, 1, 2, 3, 4, 6, 6, 6, 6, 6, 6, 6, 0, 1, 2, 5]
//                          |-  exp-0  -|-  exp-1  -|-  exp-2  -|-      exp-3          -|-  exp-4 -|-  exp-5  -|
// sorted_weight_ptr      : [a, *, *, *, g, j, m, *, d, k, *, *, b, e, h, l, n, *, *, *, *, *, *, *, c, f, i, o]
//
//
// sorted_expert_ids_ptr  : [0, 1, 2, 3, 3, 5]
// num_tokens_post_padded_ptr : [24]
//
// * local_expert_mask : indicate local expert mask used on current GPU (used for EP case)
//   and modify the output expert-ID, because we will only have enbaled expert on specific GPU.
//   we call expert input to this kernel as "global expert id", output as "local expert id"
//
// * local_expert_mask : [1, 0, 1, 1, 0, 1] (mask out expert-id=1, 4)
//
//                                            (pack below tensor, skip element marked with `-`)
//                         Y  Y  Y  Y  -  -  -  -  Y  Y  Y  Y  Y  Y  Y  Y  Y  Y  Y  Y  -  -  -  -  Y  Y  Y  Y
// sorted_token_ids_ptr : [0, 6, 6, 6, 2, 3, 4, 6, 1, 3, 6, 6, 0, 1, 2, 3, 4, 6, 6, 6, 6, 6, 6, 6, 0, 1, 2, 5]
//                        |-  exp-0  -|-  exp-1  -|-  exp-2  -|-      exp-3          -|-  exp-4 -|-  exp-5  -|
// sorted_weight_ptr    : [a, *, *, *, g, j, m, *, d, k, *, *, b, e, h, l, n, *, *, *, *, *, *, *, c, f, i, o]
//
// sorted_expert_ids_ptr  : [0, 1, 2, 2, 3] (note original it was exper-id= 0, 2, 3, 5, but we produce "local expert id")
// num_tokens_post_padded_ptr : [20]
//
// * different from vLLM
//   1) token_id stored in sorted_token_ids_ptr is actual token_id, not token_id*top_K expanded id
//   2)need sorted_weight_ptr
//   3) use num_sorted_tiles_ptr, already divided by M_a
//
// * below used for indexing
//  1) sorted_token_ids_ptr [max_num_tokens_padded]
//  2) sorted_weight_ptr
//  3) sorted_expert_ids_ptr
//  4)num_tokens_post_padded_ptr/num_sorted_tiles_ptr (select one)
//
//   max_num_tokens_padded: opk_ids.numel() + num_experts * (block_size - 1)


OPUS_H constexpr auto moe_sorting_get_smem_row_col(int tokens_, int num_experts_)
{
    /*               num_experts + 1
    *   +--------------------------------------+
    *   |                                      |
    *   |                                      |
    *   |                                      |    * -> sub-tokens
    *   |                                      |
    *   |                                      |
    *   +--------------------------------------+
    *   |                                      |    2 -> cumsum buffer
    *   +--------------------------------------+
    *
    */
    int smem_cols = num_experts_ + 1;  // usually experts is power of 2. padding here
    int smem_rows = [&](){
        opus::index_t target_occupancy_ = 2;
        constexpr opus::index_t total_ = opus::get_smem_size() / sizeof(opus::index_t);
        constexpr opus::index_t sub_unroll = 8;
        constexpr opus::index_t cumsum_bufs = 2;  // 1 for cumsum, 1 for cnt
        // at lease 2 lines, one for sub_token unroll, one for cumsum
        // should be enough

        int r = total_ / target_occupancy_ / smem_cols;

        // Note: at lease allocate cumsum_bufs + sub_unroll as num-row. Otherwise, fallback to mp kernel
        if(r < (cumsum_bufs + sub_unroll))
            return cumsum_bufs;

        // round to sub_unroll multipl
        int r_for_sub_token = r - cumsum_bufs;
        r_for_sub_token = r_for_sub_token / sub_unroll * sub_unroll;
        int r_token_min = (tokens_ + sub_unroll - 1) / sub_unroll * sub_unroll;
        r_for_sub_token = min(r_for_sub_token, r_token_min);

        // final check, but usually should not happen
        if( ((r_for_sub_token + cumsum_bufs) * smem_cols *  target_occupancy_ ) > total_ ) {
            throw std::runtime_error("can't run this kernel, request LDS over size");
        }

        return r_for_sub_token + cumsum_bufs;
    }();

    return opus::make_tuple(smem_rows, smem_cols);
}

// if return 0 or negative, means LDS is not enough
OPUS_H opus::index_t moe_sorting_get_sub_token(int tokens_, int num_experts_)
{
    auto rc_                 = moe_sorting_get_smem_row_col(tokens_, num_experts_);
    auto sub_token_          = opus::get<0>(rc_) - 2;
    return sub_token_;
}

struct MoeSortingHostArgs
{
    const void* p_topk_ids;     // [token, topk]
    const void* p_weights;      // [token, topk]

    const void* p_local_expert_mask; // [experts]
    const void* p_local_tokens;  // [1] if not nullptr, tokens read from here

    void* p_sorted_token_ids;
    void* p_sorted_weights;
    void* p_sorted_expert_ids;
    void* p_total_tokens_post_pad; // [2], [0]:outputed tokens_post_padded, [1]:actual tokens on current rank (local_tokens or tokens)
    // we fused the setzero of output of fused-moe buffer
    // set this pointer to nullptr will skip this operation
    void* p_moe_buf;
    void* p_ws;             // size is moe_sorting_get_workspace_size()
                            // if return zero, then could be nullptr
                            // must be cleard before use
    opus::index_t tokens;         // if p_local_tokens is not nullptr, this indicate the max possible tokens used for ws/LDS calculation
    opus::index_t unit_size;      // this is the M_a of fused-moe kernel
    opus::index_t num_experts;
    opus::index_t topk;
    // NOTE:
    // moe_buf_* is a 2d ws buffer used for the following fmoe kernel
    // arranged as row*col, where row=tokens(or local_token), col=interm_dim
    // we fuse this clearing inside sorting kernel
    // Besides, we require inter_dim to be multiple of 16 byte(make sure when alloc ws for fmoe)
    opus::index_t moe_buf_interm_dim; // p_moe_buf interm_dim
    opus::index_t moe_buf_elem_bytes; // p_moe_buf byte size(8bit, 16bit, 32bit, etc.)

};

template <typename Problem_>
struct MoeSortingKernel
{
    using Problem = opus::remove_cvref_t<Problem_>;

    using IndexType  = typename Problem::IndexType;
    using WeightType = typename Problem::WeightType;

    typedef MoeSortingHostArgs MoeSortingKargs;

    using Hargs = MoeSortingHostArgs;

    static constexpr opus::index_t kBlockSize = 256;
    static constexpr opus::index_t OCCUPANCY  = 2; // hard coded

    struct Kargs
    {
        const void* p_topk_ids;
        const void* p_weights;
        const void* p_local_expert_mask;
        const void* p_local_tokens;  // [1] if not nullptr, tokens read from here
        void* p_sorted_token_ids;
        void* p_sorted_weights;
        void* p_sorted_expert_ids;
        void* p_total_tokens_post_pad;
        void* p_moe_buf;
        opus::index_t tokens;
        opus::index_t num_experts;
        opus::index_t moe_buf_interm_dim; // p_moe_buf interm_dim
        opus::index_t moe_buf_elem_bytes; // p_moe_buf byte size(8bit, 16bit, 32bit, etc.)
        opus::index_t tokens_per_thread;
        opus::index_t smem_rows;
        opus::mdiv unit_size_mdiv;
        opus::mdiv topk_mdiv;
        opus::mdiv expert_mdiv;
        // opus::mdiv sub_tokens_mdiv;
    };

    OPUS_H static constexpr auto get_num_cu()
    {
        opus::index_t num_cu = [&]() {
            hipDeviceProp_t dev_prop;
            hipDevice_t dev;
            OPUS_HIP_CHECK_ERROR(hipGetDevice(&dev));
            OPUS_HIP_CHECK_ERROR(hipGetDeviceProperties(&dev_prop, dev));
            return dev_prop.multiProcessorCount;
        }();
        return num_cu;
    }

    OPUS_H static constexpr auto GridSize(const Hargs& h)
    {
        (void)h;
        return get_num_cu() * OCCUPANCY;
    }

    OPUS_H static constexpr auto BlockSize(const Hargs& h)
    {
        (void)h;
        return dim3(256);
    }

    // in byte
    OPUS_H static constexpr auto GetSmemSize(const Hargs& h)
    {
        auto rc = moe_sorting_get_smem_row_col(h.tokens, h.num_experts);
        return opus::get<0>(rc) * opus::get<1>(rc) * sizeof(opus::index_t);
    }

    OPUS_H static constexpr auto MakeKargs(const Hargs& h)
    {
        Kargs k;
        k.p_topk_ids              = h.p_topk_ids;
        k.p_weights               = h.p_weights;
        k.p_local_expert_mask     = h.p_local_expert_mask;
        k.p_local_tokens          = h.p_local_tokens;
        k.p_sorted_token_ids      = h.p_sorted_token_ids;
        k.p_sorted_weights        = h.p_sorted_weights;
        k.p_sorted_expert_ids     = h.p_sorted_expert_ids;
        k.p_moe_buf               = h.p_moe_buf;
        k.p_total_tokens_post_pad = h.p_total_tokens_post_pad;
        k.tokens                  = h.tokens;
        k.num_experts             = h.num_experts;
        k.moe_buf_interm_dim      = h.moe_buf_interm_dim;
        k.moe_buf_elem_bytes      = h.moe_buf_elem_bytes;

        const auto blocks   = BlockSize(h);
        k.tokens_per_thread = integer_divide_ceil(h.tokens * h.topk, blocks.x);
        k.unit_size_mdiv    = opus::mdiv{static_cast<uint32_t>(h.unit_size)};
        k.topk_mdiv         = opus::mdiv{static_cast<uint32_t>(h.topk)};
        // NOTE: tokens could from p_local_tokens, so here the LDS will be bigger than expected (but works)
        k.smem_rows         = opus::get<0>(moe_sorting_get_smem_row_col(h.tokens, h.num_experts));
        k.expert_mdiv      = opus::mdiv{static_cast<uint32_t>(h.num_experts)};
        // k.sub_tokens_mdiv  = opus::mdiv{static_cast<uint32_t>(k.smem_rows - 1)};
        return k;
    }

    // [a, b, c, d....] -> [a, a+b, a+b+c, a+b+c+d, ....]
    // NOTE: wave_size need at least be 16!! dpp 16 is one row
    template <typename data_t, int wave_size>
    __device__ inline void wave_cumsum(data_t& thread_data) const
    {
        // wave_size must be power of 2
        constexpr int row_mask    = 0xf;
        constexpr int bank_mask   = 0xf;
        constexpr bool bound_ctrl = true;   // ! out-of-bound is zero !
        auto reduce_op = [&](auto x_, auto y_) { return x_ + y_; };

        if constexpr(wave_size > 1)
        {
            thread_data = reduce_op(
                thread_data,
                __builtin_bit_cast(data_t, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, thread_data),
                                                            0x111,
                                                            row_mask,
                                                            bank_mask,
                                                            bound_ctrl))); // row_shr:1
        }

        if constexpr(wave_size > 2)
        {
            thread_data = reduce_op(
                thread_data,
                __builtin_bit_cast(data_t, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, thread_data),
                                                            0x112,
                                                            row_mask,
                                                            bank_mask,
                                                            bound_ctrl))); // row_shr:2
        }
        if constexpr(wave_size > 4)
        {
            thread_data =
                reduce_op(thread_data,
                        __builtin_bit_cast(data_t, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, thread_data),
                                                                        0x114,
                                                                        row_mask,
                                                                        bank_mask,
                                                                        bound_ctrl))); // row_shr:4
        }
        if constexpr(wave_size == 8) {

            // wave-size=8 need one extra shift
            thread_data =
                reduce_op(thread_data,
                        __builtin_bit_cast(data_t, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, thread_data),
                                                                        0x118,
                                                                        row_mask,
                                                                        bank_mask,
                                                                        bound_ctrl))); // row_shr:8
#if OPUS_HAS_ROW_NEWBCAST
            data_t xxx =__builtin_bit_cast(data_t,
                            __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, thread_data),
                                                        0x157,
                                                        row_mask,
                                                        bank_mask,
                                                        bound_ctrl)); // row_newbcast:7

            data_t yyy = (__lane_id() / 8) % 2 == 0 ? 0 : xxx;
            thread_data = thread_data - yyy;
#else
            // portable fallback for gfx908 and older: emulate row_newbcast:7 via ds_bpermute
            // For wave_size == 8 context, we need to broadcast from lane 7 of the 16-lane group
            int broadcast_src_lane = (__lane_id() & ~15) + 7;  // Lane 7 of the 16-lane group
            int broadcast_addr = broadcast_src_lane << 2;      // Convert to byte address
            int bcast7 = __builtin_amdgcn_ds_bpermute(broadcast_addr, __builtin_bit_cast(int, thread_data));

            // Apply subtraction only to odd 8-lane groups (lanes 8-15 of each 16-lane unit)
            if ((__lane_id() / 8) % 2 != 0) {  // Note: != 0, not == 0
                thread_data = thread_data - __builtin_bit_cast(data_t, bcast7);
            }
#endif

        }
        if constexpr(wave_size > 8)
        {
            thread_data =
                reduce_op(thread_data,
                        __builtin_bit_cast(data_t, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, thread_data),
                                                                        0x118,
                                                                        row_mask,
                                                                        bank_mask,
                                                                        bound_ctrl))); // row_shr:8
        }

        if constexpr(wave_size > 16)
        {
            // now row-0, row-0+row-1, row-1+row-2, row-2+row-3
            int v_remote_tmp = __builtin_amdgcn_ds_bpermute(((__lane_id() & 0x30) - 1) << 2, __builtin_bit_cast(int, thread_data));
            v_remote_tmp = __lane_id() >= 16 ? v_remote_tmp : 0;
            thread_data = reduce_op(thread_data, __builtin_bit_cast(data_t, v_remote_tmp));
        }

        if constexpr(wave_size > 32)
        {
            // lane-id 48...63->31
            int v_remote_tmp = __builtin_amdgcn_ds_bpermute(((__lane_id() & 0x30) - 17) << 2, __builtin_bit_cast(int, thread_data));
            v_remote_tmp = __lane_id() >= 32 ? v_remote_tmp : 0;
            thread_data = reduce_op(thread_data, __builtin_bit_cast(data_t, v_remote_tmp));
        }
    }

    // reduce single pixel within a wave
    template <typename T, typename F, opus::index_t wave_size_ = opus::get_warp_size()>
    __device__ static constexpr T wave_reduce(T local, F reduce_f, opus::number<wave_size_> = {})
    {
        // constexpr int wave_size = 64;
        // constexpr int reduce_stage = 6; // 1<<6=64
        // clang-format off
        constexpr int reduce_stage = [](){
            if constexpr(wave_size_ == 2) return 1;
            else if constexpr(wave_size_ == 4) return 2;
            else if constexpr(wave_size_ == 8) return 3;
            else if constexpr(wave_size_ == 16) return 4;
            else if constexpr(wave_size_ == 32) return 5;
            else if constexpr(wave_size_ == 64) return 6;
            else return 0;
        }();
        // clang-format on
        T v_local = local;
#pragma unroll reduce_stage
        for(int i_stage = 0; i_stage < reduce_stage; i_stage++)
        {
            int src_lane = __lane_id() ^ (1 << i_stage);
            int32_t v_remote_tmp =
                __builtin_amdgcn_ds_bpermute(src_lane << 2, __builtin_bit_cast(int32_t, v_local));
            T v_remote = __builtin_bit_cast(T, v_remote_tmp);
            v_local    = reduce_f(v_local, v_remote);
        }
        return v_local;
    }

    OPUS_D opus::index_t calc_index(opus::index_t total_col, opus::index_t row, opus::index_t col) const
    {
        return row * total_col + col;
    }

    OPUS_D void
    moe_buf_set_zero_kernel_2d(void* buf, opus::index_t row, opus::index_t col, opus::index_t elem_bytes) const
    {
        const opus::long_index_t total_pixels = static_cast<opus::long_index_t>(row) * col;
        const opus::long_index_t total_bytes  = total_pixels * elem_bytes;
        const opus::long_index_t total_elems  = total_bytes / 16; // always use dwordx4

        using vector_type  = opus::vector_t<opus::index_t, 4>;
        vector_type* p_buf = reinterpret_cast<vector_type*>(buf);
        auto zero_         = vector_type{0};

        for(opus::long_index_t i = (blockIdx.x - 1) * kBlockSize + threadIdx.x; i < total_elems;
            i += (gridDim.x - 1) * kBlockSize)
        {
            p_buf[i] = zero_;
        }
    }

    // only support opus::index_t, and single pixel access
    struct simple_smem_indexer
    {
        opus::index_t* smem;
        opus::index_t row_stride;

        // this is 2D
        OPUS_D simple_smem_indexer(opus::index_t* smem_, opus::index_t row_stride_)
            : smem(smem_), row_stride(row_stride_)
        {
        }
        OPUS_D const opus::index_t& operator()(opus::index_t i_row, opus::index_t i_col) const
        {
            return smem[i_row * row_stride + i_col];
        }
        OPUS_D opus::index_t& operator()(opus::index_t i_row, opus::index_t i_col)
        {
            return smem[i_row * row_stride + i_col];
        }

        // this is 1D or linear
        OPUS_D simple_smem_indexer(opus::index_t* smem_) : smem(smem_), row_stride(0) {}
        OPUS_D const opus::index_t& operator()(opus::index_t idx) const { return smem[idx]; }
        OPUS_D opus::index_t& operator()(opus::index_t idx) { return smem[idx]; }
    };

    OPUS_D void
    moe_align_block_size_kernel_ex(const IndexType* __restrict__ topk_id,
                                   const WeightType* __restrict__ weights,
                                   const IndexType* __restrict__ local_expert_mask,
                                   opus::index_t* p_sorted_token_ids,
                                   WeightType* p_sorted_weights,
                                   opus::index_t* p_sorted_expert_ids,
                                   opus::index_t* p_total_tokens_post_pad,
                                   const opus::index_t num_experts,
                                   const opus::index_t tokens,
                                   const opus::mdiv unit_size_mdiv,
                                   const opus::mdiv topk_mdiv,
                                   const opus::mdiv expert_mdiv,
                                   const opus::index_t smem_rows,
                                   void* smem) const
    {
        const opus::index_t tid            = static_cast<opus::index_t>(threadIdx.x);
        const opus::index_t wid            = __builtin_amdgcn_readfirstlane(tid / opus::get_warp_size());
        const opus::index_t lid            = __lane_id();
        constexpr opus::index_t block_size = 256;           // blockDim.x;
        const opus::index_t sub_tokens     = smem_rows - 2; // sub_tokens_mdiv.divisor;
        const opus::index_t topk           = topk_mdiv.divisor;
        auto f_sum                   = [](auto x_, auto y_) { return x_ + y_; };

        const opus::index_t smem_cols = num_experts + 1;

        simple_smem_indexer smem_cumsum{reinterpret_cast<opus::index_t*>(smem) + 0};
        simple_smem_indexer smem_cumdup{reinterpret_cast<opus::index_t*>(smem) + smem_cols};
        simple_smem_indexer smem_tokens{reinterpret_cast<opus::index_t*>(smem) + 2 * smem_cols,
                                        smem_cols};

        // #pragma unroll 8
        for(int i = tid; i < (sub_tokens * num_experts); i += block_size)
        {
            uint32_t curr_token_id, curr_expert_id;
            expert_mdiv.divmod(i, curr_token_id, curr_expert_id);
            smem_tokens(curr_token_id, curr_expert_id) = 0;
        }
        __syncthreads();

        for(int i_token = 0; i_token < tokens; i_token += sub_tokens)
        {
            // NOTE: below for loop can't have barrier inside!!
            for(int i = tid; i < (sub_tokens * topk); i += block_size)
            {
                uint32_t curr_token_id, curr_topk_id;
                topk_mdiv.divmod(i, curr_token_id, curr_topk_id);
                int i_t = i_token + curr_token_id;

                if(i_t < tokens)
                {
                    int eid = topk_id[i_t * topk + curr_topk_id];

                    if constexpr(Problem::SubTokenOneShot)
                        smem_tokens(curr_token_id, eid) = curr_topk_id + 1;
                    else
                        smem_tokens(curr_token_id, eid)++;
                }
                opus::s_waitcnt_lgkmcnt(opus::number<0>{});
            }
            __syncthreads(); // make sure different i_token iteration not overlap by different wave
        }

        // counting
        if(tid == 0)
        {
            smem_cumsum(0) = 0;
            // smem_cumdup(0) = 0;
        }

        {
            constexpr int lane_group_sz = 8;
            int lane_group_id           = tid / lane_group_sz;
            int lane_group_os           = tid % lane_group_sz;
            constexpr int lane_group_nm = block_size / lane_group_sz;

            for(int i_e = lane_group_id; i_e < num_experts; i_e += lane_group_nm)
            {
                opus::index_t local_c[Problem::SubTokenTile];
                opus::index_t cnt = 0;

                for(int i = 0; i < sub_tokens; i += 8 * Problem::SubTokenTile)
                {
#pragma unroll Problem::SubTokenTile
                    for(int j = 0; j < Problem::SubTokenTile; j++)
                    {
                        local_c[j] = smem_tokens(i + j * 8 + lane_group_os, i_e);
                        if constexpr(Problem::SubTokenOneShot)
                        {
                            local_c[j] = local_c[j] != 0 ? 1 : 0;
                        }
                    }

#pragma unroll Problem::SubTokenTile
                    for(int j = 0; j < Problem::SubTokenTile; j++)
                    {
                        cnt += wave_reduce(local_c[j], f_sum, opus::number<8>{});
                    }
                }
                if(lane_group_os == 0)
                    smem_cumsum(i_e + 1) = cnt;
            }
        }

        if constexpr(Problem::LocalExpertMasking)
        {
            smem_cumdup(0) = 0;
            for(int i_e = tid; i_e < num_experts; i_e += block_size)
            {
                // reuse this buffer
                smem_cumdup(i_e + 1) = local_expert_mask[i_e];
            }
        }

        __syncthreads();

        {
            if(wid == 0)
            {
                // NOTE: under this block can never use __syncthreads!
                int i_e_          = 0;
                int local_cumsum_ = 0;
                for(; i_e_ < num_experts; i_e_ += opus::get_warp_size())
                {
                    int pre_cumsum_ = smem_cumsum(lid == 0 ? i_e_ : 0);
                    int local_cnt   = smem_cumsum(i_e_ + lid + 1);
                    int blocks_pers_expert =
                        unit_size_mdiv.div(local_cnt + unit_size_mdiv.divisor - 1);

                    int pre_cumsum_masking = [&]() {
                        if constexpr(Problem::LocalExpertMasking)
                            return smem_cumdup(lid == 0 ? i_e_ : 0);
                        else
                            return 0; // not used
                    }();
                    int local_masking = [&]() {
                        if constexpr(Problem::LocalExpertMasking)
                            return smem_cumdup(i_e_ + lid + 1);
                        else
                            return 0; // not used
                    }();
                    int padded_tokens_per_expert = [&]() {
                        int x_ = [&]() {
                            if constexpr(Problem::SkipExpertsWithZeroTokens)
                            {
                                // if local_cnt is zero, blocks_pers_expert will be zero
                                // this is what we want to achieve
                                return blocks_pers_expert * unit_size_mdiv.divisor;
                            }
                            else
                            {
                                return max(blocks_pers_expert, 1) * unit_size_mdiv.divisor;
                            }
                        }();
                        if constexpr(Problem::LocalExpertMasking)
                        {
                            return local_masking ? x_ : 0;
                        }
                        else
                            return x_;
                    }();

                    local_cumsum_ = padded_tokens_per_expert;
                    local_cumsum_ += pre_cumsum_; // note pre_cumsum must be added after local
                                                  // cumsum padded in case local cumsum is zero, but
                                                  // pre_sumsum has value, which will result int
                                                  // zero local cumsum(but we want at least padded)
                    wave_cumsum<int, opus::get_warp_size()>(local_cumsum_);

                    if((i_e_ + lid) < num_experts)
                        smem_cumsum(i_e_ + lid + 1) = local_cumsum_;

                    if constexpr(Problem::LocalExpertMasking)
                    {
                        local_masking += pre_cumsum_masking;
                        wave_cumsum<int, opus::get_warp_size()>(local_masking);
                        if((i_e_ + lid) < num_experts)
                            smem_cumdup(i_e_ + lid + 1) = local_masking;
                    }

                    // NOTE: this waitcnt is a must, compiler will not generate waitcnt lgkmcnt()
                    // for above write however __syncthreads will cause barrier with waves other
                    // than 0(which is not we want)
                    opus::s_waitcnt_lgkmcnt(opus::number<0>{});
                }
                if((lid + i_e_ - opus::get_warp_size()) == (num_experts - 1))
                {
                    *p_total_tokens_post_pad   = local_cumsum_;
                    p_total_tokens_post_pad[1] = tokens;
                }
            }
            __syncthreads();
        }

        for(int i_e = tid; i_e < num_experts; i_e += block_size)
        {
            int e_start = smem_cumsum(i_e);
            int e_end   = smem_cumsum(i_e + 1);

            int expert_id = [&]() {
                if constexpr(Problem::LocalExpertMasking)
                {
                    // local expert id from cumsum
                    return smem_cumdup(i_e);
                }
                else
                    return i_e;
            }();

            smem_cumdup(i_e) = e_start; // duplicate cumsum for later use
            if constexpr(Problem::SkipExpertsWithZeroTokens)
            {
                if(e_start == e_end) // skip zero token expert
                    continue;
            }

            if constexpr(Problem::LocalExpertMasking)
            {
                if(local_expert_mask[i_e] == 0)
                    continue;
            }

            for(int i = e_start; i < e_end; i += unit_size_mdiv.divisor)
            {
                p_sorted_expert_ids[unit_size_mdiv.div(i)] = expert_id;
            }
        }
        __syncthreads();

        smem_cumdup(num_experts) = smem_cumsum(num_experts);

        // fill the p_sorted_token_ids/p_sorted_weights
        for(int i_token = 0; i_token < tokens; i_token += sub_tokens)
        {
            if constexpr(!Problem::SubTokenOneShot)
            {
                // clear every time
                for(int i = tid; i < (sub_tokens * num_experts); i += block_size)
                {
                    uint32_t curr_token_id, curr_expert_id;
                    expert_mdiv.divmod(i, curr_token_id, curr_expert_id);
                    smem_tokens(curr_token_id, curr_expert_id) = 0;
                }
                __syncthreads();

                // load again
                for(int i = tid; i < (sub_tokens * topk); i += block_size)
                {
                    uint32_t curr_token_id_, curr_topk_id_;
                    topk_mdiv.divmod(i, curr_token_id_, curr_topk_id_);
                    int curr_token_id = static_cast<int>(curr_token_id_);
                    int curr_topk_id  = static_cast<int>(curr_topk_id_);
                    int i_t           = i_token + curr_token_id;
                    if(i_t < tokens)
                    {
                        int eid                         = topk_id[i_t * topk + curr_topk_id];
                        smem_tokens(curr_token_id, eid) = curr_topk_id + 1; // at least 1
                    }
                }
                __syncthreads();
            }

            {
                constexpr int lane_group_sz = 8;
                int lane_group_id           = tid / lane_group_sz;
                int lane_group_os           = tid % lane_group_sz;
                constexpr int lane_group_nm = block_size / lane_group_sz;
                for(int eid = lane_group_id; eid < num_experts; eid += lane_group_nm)
                {
                    if constexpr(Problem::LocalExpertMasking)
                    {
                        if(local_expert_mask[eid] == 0)
                            continue;
                    }
                    int position = smem_cumsum(eid);
                    for(int i_sub_token = lane_group_os; i_sub_token < sub_tokens;
                        i_sub_token += lane_group_sz)
                    {
                        auto x = smem_tokens(i_sub_token, eid);

                        int local_cnt_cache = x != 0 ? 1 : 0;
                        int local_cnt       = local_cnt_cache;
                        wave_cumsum<int, lane_group_sz>(local_cnt);
                        if(x != 0)
                        {
                            // now x is topk value
#if OPUS_MOE_SORTING_MOCK_ID
                            p_sorted_token_ids[position + local_cnt - 1] =
                                MOE_SORTING_MOCK_ID(i_token + i_sub_token, x - 1);
#else
                            p_sorted_token_ids[position + local_cnt - 1] = i_token + i_sub_token;
#endif
                            p_sorted_weights[position + local_cnt - 1] =
                                weights[(i_token + i_sub_token) * topk + x - 1];
                        }

                        int remote_cnt = __builtin_amdgcn_ds_bpermute(
                            (lane_group_sz * (lane_group_id + 1) - 1) << 2, local_cnt);

                        position += remote_cnt;
                    }
                    smem_cumsum(eid) = position;
                }
            }
            __syncthreads();
        }

        // add the skip number
        for(int eid = tid; eid < num_experts; eid += block_size)
        {
            int e_start = smem_cumsum(eid);
            int e_end   = smem_cumdup(eid + 1);
            if constexpr(Problem::SkipExpertsWithZeroTokens)
            {
                if(e_start == e_end) // skip zero token expert
                    continue;
            }
            while(e_start < e_end)
            {
#if OPUS_MOE_SORTING_MOCK_ID
                p_sorted_token_ids[e_start] = MOE_SORTING_MOCK_ID(tokens, topk);
#else
                p_sorted_token_ids[e_start] = tokens;
#endif
                p_sorted_weights[e_start] = static_cast<WeightType>(0.0);
                e_start++;
            }
        }
    }

    OPUS_D void operator()(Kargs kargs) const
    {
        opus::index_t tokens_ = [&]() {
            if constexpr(Problem::LocalToken)
            {
                return reinterpret_cast<const opus::index_t*>(kargs.p_local_tokens)[0];
            }
            else
            {
                return kargs.tokens;
            }
        }();

        if(blockIdx.x > 0)
        {
            if(kargs.p_moe_buf)
            {
                moe_buf_set_zero_kernel_2d(
                    kargs.p_moe_buf, tokens_, kargs.moe_buf_interm_dim, kargs.moe_buf_elem_bytes);
            }
            return;
        }

        extern __shared__ char smem[];

        return moe_align_block_size_kernel_ex(
            static_cast<const IndexType*>(kargs.p_topk_ids),
            static_cast<const WeightType*>(kargs.p_weights),
            static_cast<const IndexType*>(kargs.p_local_expert_mask),
            static_cast<IndexType*>(kargs.p_sorted_token_ids),
            static_cast<WeightType*>(kargs.p_sorted_weights),
            static_cast<IndexType*>(kargs.p_sorted_expert_ids),
            static_cast<IndexType*>(kargs.p_total_tokens_post_pad),
            kargs.num_experts,
            tokens_,
            kargs.unit_size_mdiv,
            kargs.topk_mdiv,
            kargs.expert_mdiv,
            kargs.smem_rows,
            smem);
    }
};

namespace impl {

// [expert, padded_tokens]
OPUS_H_D opus::index_t moe_sorting_mp_mesh_stride(opus::index_t tokens)
{
    // Pad to multiply of 32. This can make sure even if the mesh is in 8bit,
    // we can still use dwordx4 load/store
    constexpr opus::index_t chunk = 32;
    return (tokens + chunk - 1) / chunk * chunk;
};

// 4-i32 mesh, 2-i16 mseh, 1-i8 mesh
OPUS_H opus::index_t moe_sorting_mesh_byte_size(opus::index_t tokens_,
                                                opus::index_t /*num_experts_*/,
                                                opus::index_t topk_)
{
    // small token case, let's run mesh with dword score board
    if(tokens_ < 512)
        return 4;
    else
    {
        if(topk_ >= 255)
            return 2; // 16bit mesh
        else
            return 1; // 8bit mesh if small enough
    }
}

OPUS_H_D opus::index_t moe_sorting_mp_mesh_smem_size(opus::index_t tokens,
                                                          opus::index_t num_experts,
                                                          opus::index_t topk)
{
    opus::index_t row_size = moe_sorting_mp_mesh_stride(tokens);
    opus::index_t elem     = num_experts * row_size;
    return elem * moe_sorting_mesh_byte_size(tokens, num_experts, topk);
};

OPUS_H_D opus::index_t moe_sorting_mp_cumsum_smem_size(opus::index_t num_experts)
{
    constexpr opus::index_t chunk = 32;
    opus::index_t row_size        = num_experts + 1;
    return (row_size + chunk - 1) / chunk * chunk * sizeof(opus::index_t);
};

OPUS_H_D opus::index_t moe_sorting_mp_sem_smem_size()
{
    constexpr opus::index_t chunk = 32;
    return chunk * sizeof(opus::index_t);
};

template <typename T, typename F, opus::index_t wave_size_ = opus::get_warp_size()>
OPUS_D constexpr T moe_sorting_wave_reduce(T local, F reduce_f, opus::number<wave_size_> = {})
{
    // constexpr int wave_size = 64;
    // constexpr int reduce_stage = 6; // 1<<6=64
    // clang-format off
    constexpr int reduce_stage = [](){
        if constexpr(wave_size_ == 2) return 1;
        else if constexpr(wave_size_ == 4) return 2;
        else if constexpr(wave_size_ == 8) return 3;
        else if constexpr(wave_size_ == 16) return 4;
        else if constexpr(wave_size_ == 32) return 5;
        else if constexpr(wave_size_ == 64) return 6;
        else return 0;
    }();
    // clang-format on
    T v_local = local;
#pragma unroll reduce_stage
    for(int i_stage = 0; i_stage < reduce_stage; i_stage++)
    {
        int src_lane = __lane_id() ^ (1 << i_stage);
        int32_t v_remote_tmp =
            __builtin_amdgcn_ds_bpermute(src_lane << 2, __builtin_bit_cast(int32_t, v_local));
        T v_remote = __builtin_bit_cast(T, v_remote_tmp);
        v_local    = reduce_f(v_local, v_remote);
    }
    return v_local;
}

// [a, b, c, d....] -> [a, a+b, a+b+c, a+b+c+d, ....]
// NOTE: wave_size need at least be 16!! dpp 16 is one row
template <typename data_t, int wave_size>
OPUS_D void moe_sorting_wave_cumsum(data_t& thread_data)
{
    // wave_size must be power of 2
    constexpr int row_mask    = 0xf;
    constexpr int bank_mask   = 0xf;
    constexpr bool bound_ctrl = true; // ! out-of-bound is zero !
    auto reduce_op            = [&](auto x_, auto y_) { return x_ + y_; };

    if constexpr(wave_size > 1)
    {
        thread_data = reduce_op(
            thread_data,
            __builtin_bit_cast(data_t,
                               __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, thread_data),
                                                        0x111,
                                                        row_mask,
                                                        bank_mask,
                                                        bound_ctrl))); // row_shr:1
    }

    if constexpr(wave_size > 2)
    {
        thread_data = reduce_op(
            thread_data,
            __builtin_bit_cast(data_t,
                               __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, thread_data),
                                                        0x112,
                                                        row_mask,
                                                        bank_mask,
                                                        bound_ctrl))); // row_shr:2
    }
    if constexpr(wave_size > 4)
    {
        thread_data = reduce_op(
            thread_data,
            __builtin_bit_cast(data_t,
                               __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, thread_data),
                                                        0x114,
                                                        row_mask,
                                                        bank_mask,
                                                        bound_ctrl))); // row_shr:4
    }
    if constexpr(wave_size == 8)
    {

        // wave-size=8 need one extra shift
        thread_data = reduce_op(
            thread_data,
            __builtin_bit_cast(data_t,
                               __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, thread_data),
                                                        0x118,
                                                        row_mask,
                                                        bank_mask,
                                                        bound_ctrl))); // row_shr:8
#if OPUS_HAS_ROW_NEWBCAST
        data_t xxx =
            __builtin_bit_cast(data_t,
                               __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, thread_data),
                                                        0x157,
                                                        row_mask,
                                                        bank_mask,
                                                        bound_ctrl)); // row_newbcast:7

        data_t yyy  = (__lane_id() / 8) % 2 == 0 ? 0 : xxx;
        thread_data = thread_data - yyy;
#else
        // portable fallback for gfx908 and older: emulate row_newbcast:7 via ds_bpermute
        // For wave_size == 8 context, we need to broadcast from lane 7 of the 16-lane group
        int broadcast_src_lane = (__lane_id() & ~15) + 7; // Lane 7 of the 16-lane group
        int broadcast_addr     = broadcast_src_lane << 2; // Convert to byte address
        int bcast7 =
            __builtin_amdgcn_ds_bpermute(broadcast_addr, __builtin_bit_cast(int, thread_data));

        // Apply subtraction only to odd 8-lane groups (lanes 8-15 of each 16-lane unit)
        if((__lane_id() / 8) % 2 != 0)
        { // Note: != 0, not == 0
            thread_data = thread_data - __builtin_bit_cast(data_t, bcast7);
        }
#endif
    }
    if constexpr(wave_size > 8)
    {
        thread_data = reduce_op(
            thread_data,
            __builtin_bit_cast(data_t,
                               __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, thread_data),
                                                        0x118,
                                                        row_mask,
                                                        bank_mask,
                                                        bound_ctrl))); // row_shr:8
    }

    if constexpr(wave_size > 16)
    {
        // now row-0, row-0+row-1, row-1+row-2, row-2+row-3
        int v_remote_tmp = __builtin_amdgcn_ds_bpermute(((__lane_id() & 0x30) - 1) << 2,
                                                        __builtin_bit_cast(int, thread_data));
        v_remote_tmp     = __lane_id() >= 16 ? v_remote_tmp : 0;
        thread_data      = reduce_op(thread_data, __builtin_bit_cast(data_t, v_remote_tmp));
    }

    if constexpr(wave_size > 32)
    {
        // lane-id 48...63->31
        int v_remote_tmp = __builtin_amdgcn_ds_bpermute(((__lane_id() & 0x30) - 17) << 2,
                                                        __builtin_bit_cast(int, thread_data));
        v_remote_tmp     = __lane_id() >= 32 ? v_remote_tmp : 0;
        thread_data      = reduce_op(thread_data, __builtin_bit_cast(data_t, v_remote_tmp));
    }
}

template <opus::index_t kBlockSize = 256>
OPUS_D void moe_buf_set_zero_kernel_2d(
    void* buf, opus::index_t row, opus::index_t col, opus::index_t elem_bytes, opus::index_t gid, opus::index_t blocks)
{
    const opus::long_index_t total_pixels = static_cast<opus::long_index_t>(row) * col;
    const opus::long_index_t total_bytes  = total_pixels * elem_bytes;
    const opus::long_index_t total_elems  = total_bytes / 16; // always use dwordx4

    using vector_type  = opus::vector_t<opus::index_t, 4>;
    vector_type* p_buf = reinterpret_cast<vector_type*>(buf);
    auto zero_         = vector_type{0};

    for(opus::long_index_t i = gid * kBlockSize + threadIdx.x; i < total_elems; i += blocks * kBlockSize)
    {
        p_buf[i] = zero_;
    }
}

} // namespace impl

// TODO: tokens could be from
// prefer to run mp kernel if is not oneshot
OPUS_H bool moe_sorting_is_oneshot(int tokens_, int num_experts_)
{
#if OPUS_WA_ISSUE_2028
    if(tokens_ >= 65536 * 2)
    {
        return true;
    }
#endif
    auto sub_token_          = moe_sorting_get_sub_token(tokens_, num_experts_);
    bool is_sub_token_onshot = tokens_ <= sub_token_;
    return is_sub_token_onshot;
}

// return size in byte
OPUS_H opus::index_t moe_sorting_mp_get_workspace_size(int tokens_, int num_experts_, int topk_)
{
    opus::index_t s_ = impl::moe_sorting_mp_mesh_smem_size(tokens_, num_experts_, topk_) +
                 impl::moe_sorting_mp_cumsum_smem_size(num_experts_)
#if MOE_SORTING_FUSE_MP_01
                 + impl::moe_sorting_mp_sem_smem_size();
#else
        ;
#endif
    return s_;
}

// return size in byte
// dispatch_policy: 0-automatically pick up kerel. 1-always use single kernel, 2-always use mp
// kernel
OPUS_H opus::index_t moe_sorting_get_workspace_size(int tokens_,
                                                    int num_experts_,
                                                    int topk_,
                                                    int dispatch_policy_)
{
#if 1
    // return 0;
    if(dispatch_policy_ == 0)
    {
        if(moe_sorting_is_oneshot(tokens_, num_experts_))
        {
            return 0;
        }
        else
        {
            return moe_sorting_mp_get_workspace_size(tokens_, num_experts_, topk_);
        }
    }
    else if(dispatch_policy_ == 1)
    {
        return 0; // always use single kernel
    }
    else
    {
        return moe_sorting_mp_get_workspace_size(tokens_, num_experts_, topk_);
    }
#else
    return moe_sorting_mp_get_workspace_size(tokens_, num_experts_, topk_);
#endif
}

template <typename Problem_>
struct MoeSortingClearWorkspaceKernel
{
    using Problem                       = opus::remove_cvref_t<Problem_>;
    static constexpr opus::index_t kBlockSize = Problem::BlockSize;
    static constexpr opus::index_t OCCUPANCY  = Problem::Occu;

    using Hargs = MoeSortingHostArgs;

    struct Kargs
    {
        const void* p_local_tokens; // [1], if not nullptr, use this as actual tokens
        void* p_expert_mesh;        // [expert, tokens]
        opus::index_t tokens; // if p_local_tokens is not nullptr, this indicate the max possible tokens
                        // used for ws/LDS calculation
        opus::index_t num_experts;
        opus::index_t mesh_stride; // mesh_stride for p_expert_mesh
        opus::index_t mesh_byte_size;
    };

    OPUS_H static constexpr auto get_num_cu()
    {
        opus::index_t num_cu = [&]() {
            hipDeviceProp_t dev_prop;
            hipDevice_t dev;
            OPUS_HIP_CHECK_ERROR(hipGetDevice(&dev));
            OPUS_HIP_CHECK_ERROR(hipGetDeviceProperties(&dev_prop, dev));
            return dev_prop.multiProcessorCount;
        }();
        return num_cu;
    }

    OPUS_H static constexpr auto MakeKargs(const Hargs& h)
    {
        Kargs k;
        k.p_local_tokens = h.p_local_tokens;
        k.p_expert_mesh  = h.p_ws;
        k.tokens         = h.tokens;
        k.num_experts    = h.num_experts;
        k.mesh_stride    = impl::moe_sorting_mp_mesh_stride(h.tokens);
        k.mesh_byte_size = impl::moe_sorting_mesh_byte_size(h.tokens, h.num_experts, h.topk);
        return k;
    }

    OPUS_H static constexpr auto GridSize(const Hargs&) { return get_num_cu() * OCCUPANCY; }

    OPUS_H static constexpr auto BlockSize(const Hargs&) { return dim3(kBlockSize); }

    // in byte
    OPUS_H static constexpr auto GetSmemSize() { return 0; }

    OPUS_D void operator()(Kargs kargs) const
    {
        opus::index_t tokens = [&]() {
            if constexpr(Problem::LocalToken)
            {
                return reinterpret_cast<const opus::index_t*>(kargs.p_local_tokens)[0];
            }
            else
            {
                return kargs.tokens;
            }
        }();

        opus::index_t mesh_stride = [&]() {
            if constexpr(Problem::LocalToken)
            {
                return impl::moe_sorting_mp_mesh_stride(tokens);
            }
            else
            {
                return kargs.mesh_stride;
            }
        }();

        opus::index_t row_size    = mesh_stride; // impl::moe_sorting_mp_mesh_stride(tokens);
        opus::index_t pixels      = kargs.num_experts * row_size;
        opus::index_t total_bytes = pixels * kargs.mesh_byte_size;
        opus::index_t total_elems = total_bytes / 16; // always use dwordx4

        using vector_type          = opus::vector_t<opus::index_t, 4>;
        vector_type* p_expert_mesh = reinterpret_cast<vector_type*>(kargs.p_expert_mesh);
        auto zero_                 = vector_type{0};

        for(opus::index_t i = blockIdx.x * kBlockSize + threadIdx.x; i < total_elems;
            i += gridDim.x * kBlockSize)
        {
            p_expert_mesh[i] = zero_;
        }
    }
};

// below kernel is multi-phase implementation for large token and/or expert case

// write into a buffer to record the token cnt
// e.g. num_experts = 6, topk=3, M_a = 4, input_tokens = 5
// before sort, topk_ids is : [[0, 3, 5], [2, 3, 5], [1, 3, 5], [1, 2, 3], [1, 3, 5]]
//                            tok-0      tok-1      tok-2      tok-3      tok-4
//           topk_weight is : [[a, b, c], [d, e, f], [g, h, i], [j, k, l], [m, n, o]] (some float
//           number)
//
// token_id_per_expert is : [[0], [2, 3, 4], [1, 3], [0, 1, 2, 3, 4], [], [0, 1, 2, 5]]
//  (only for reference)    exp-0  exp-1     exp-2   exp-3          exp-4  exp-5
// weight_id_per_expert is: [[a], [g, j, m], [d, k], [b, e, h, l, n], [], [c, f, i, o]]
/*

p_expert_mesh:
     t0 t1 t2 t3 t4 r5
    +--+--+--+--+--+--+
e0  | 1|  |  |  |  |  |
e1  |  |  | 1| 1| 1|  |
e2  |  | 1|  | 1|  |  |
e3  | 1| 1| 1| 1| 1|  |
e4  |  |  |  |  |  |  |
e5  | 1| 1| 1|  |  | 1|


p_expert_cumsum:
    | 1| 3| 2| 5| 0| 4|
     e0 e1 e2 e3 e4 e5

p_expert_cumsum(with M_a pad, and skip zero tokens):
    | 4| 4| 4| 8| 0| 4|
     e0 e1 e2 e3 e4 e5

p_expert_cumsum
    | 0| 4| 8|12|20|20|24|

local_expert_mask : [1, 0, 1, 1, 0, 1] (mask out expert-id=1, 4)

p_m_cumsum
    | 0| 1| 1| 2| 3| 3| 4|

*/

// count topk_id into mesh
template <typename Problem_>
struct MoeSortingMultiPhaseKernel_P0_v1
{
    using Problem = opus::remove_cvref_t<Problem_>;

    using IndexType  = typename Problem::IndexType;
    using WeightType = typename Problem::WeightType;
    using MeshType   = typename Problem::MeshType;

    static constexpr opus::index_t kBlockSize = 256;
    static constexpr opus::index_t OCCUPANCY  = 2; // hard coded

    typedef MoeSortingHostArgs MoeSortingKargs;

    using Hargs = MoeSortingHostArgs;

    struct Kargs
    {
        const void* p_topk_ids;     // [tokens, topk]
        const void* p_local_tokens; // [1], if not nullptr, use this as actual tokens
        void* p_expert_mesh;        // [expert, tokens]
        opus::index_t tokens; // if p_local_tokens is not nullptr, this indicate the max possible tokens
                        // used for ws/LDS calculation
        opus::index_t num_experts;
        opus::index_t mesh_stride; // mesh_stride for p_expert_mesh
        opus::mdiv topk_mdiv;
    };

    OPUS_H static constexpr auto get_num_cu()
    {
        opus::index_t num_cu = [&]() {
            hipDeviceProp_t dev_prop;
            hipDevice_t dev;
            OPUS_HIP_CHECK_ERROR(hipGetDevice(&dev));
            OPUS_HIP_CHECK_ERROR(hipGetDeviceProperties(&dev_prop, dev));
            return dev_prop.multiProcessorCount;
        }();
        return num_cu;
    }

    OPUS_H static constexpr auto MakeKargs(const Hargs& h)
    {
        Kargs k;
        k.p_topk_ids     = h.p_topk_ids;
        k.p_local_tokens = h.p_local_tokens;
        k.p_expert_mesh  = h.p_ws;
        k.tokens         = h.tokens;
        k.num_experts    = h.num_experts;
        k.mesh_stride    = impl::moe_sorting_mp_mesh_stride(h.tokens);
        k.topk_mdiv      = opus::mdiv{static_cast<uint32_t>(h.topk)};
        return k;
    }

    OPUS_H static constexpr auto GridSize(const Hargs&) { return get_num_cu() * OCCUPANCY; }

    OPUS_H static constexpr auto BlockSize(const Hargs&) { return dim3(kBlockSize); }

    // in byte
    OPUS_H static constexpr auto GetSmemSize() { return 0; }

    OPUS_D void operator()(Kargs kargs) const
    {
        using topk_id_t = opus::vector_t<IndexType, Problem::SubTokenTile>;

        const topk_id_t* p_topk_ids = reinterpret_cast<const topk_id_t*>(kargs.p_topk_ids);
        MeshType* p_expert_mesh     = reinterpret_cast<MeshType*>(kargs.p_expert_mesh);
        opus::index_t tokens              = [&]() {
            if constexpr(Problem::LocalToken)
            {
                return reinterpret_cast<const opus::index_t*>(kargs.p_local_tokens)[0];
            }
            else
            {
                return kargs.tokens;
            }
        }();
        opus::index_t rounded_tokens = [&]() {
            if constexpr(Problem::LocalToken)
            {
                return (tokens + Problem::SubTokenTile - 1) / Problem::SubTokenTile *
                       Problem::SubTokenTile;
            }
            else
                return tokens;
        }();
        opus::index_t mesh_stride = [&]() {
            if constexpr(Problem::LocalToken)
            {
                return impl::moe_sorting_mp_mesh_stride(tokens);
            }
            else
            {
                return kargs.mesh_stride;
            }
        }();
        opus::index_t total_elem = rounded_tokens * kargs.topk_mdiv.divisor / Problem::SubTokenTile;

#pragma unroll Problem::SubTokenTile
        for(opus::index_t i = blockIdx.x * kBlockSize + threadIdx.x; i < total_elem;
            i += gridDim.x * kBlockSize)
        {
            auto x = p_topk_ids[i];
            opus::static_for<Problem::SubTokenTile>([&](auto j) {
                IndexType eid = x[j.value]; // ext_vector_type must use int to []
                uint32_t curr_token_id, curr_topk_id;
                kargs.topk_mdiv.divmod(i * Problem::SubTokenTile + j, curr_token_id, curr_topk_id);
                if(eid < kargs.num_experts)
                {
                    if constexpr(Problem::LocalToken)
                    {
                        if(static_cast<opus::index_t>(curr_token_id) < tokens)
                            p_expert_mesh[eid * mesh_stride + curr_token_id] =
                                (curr_topk_id + 1) & 0xffff;
                    }
                    else
                        p_expert_mesh[eid * mesh_stride + curr_token_id] =
                            (curr_topk_id + 1) & 0xffff;
                }
            });
        }
    }
};
template <typename Problem_>
struct MoeSortingMultiPhaseKernel_P0_v2
{
    using Problem = opus::remove_cvref_t<Problem_>;

    using IndexType  = typename Problem::IndexType;
    using WeightType = typename Problem::WeightType;
    using MeshType   = typename Problem::MeshType;

    static constexpr opus::index_t kBlockSize = 512;

    typedef MoeSortingHostArgs MoeSortingKargs;

    using Hargs = MoeSortingHostArgs;

    struct Kargs
    {
        const void* p_topk_ids;     // [tokens, topk]
        const void* p_local_tokens; // [1], if not nullptr, use this as actual tokens
        void* p_expert_mesh;        // [expert, tokens]
        opus::index_t tokens; // if p_local_tokens is not nullptr, this indicate the max possible tokens
                        // used for ws/LDS calculation
        opus::index_t mesh_stride; // mesh_stride for p_expert_mesh
        opus::mdiv topk_mdiv;

        const void* p_local_expert_mask; // [expert]
        void* p_expert_cumsum;           // [expert]
        opus::index_t num_experts;
    };

    OPUS_H static constexpr auto get_num_cu()
    {
        opus::index_t num_cu = [&]() {
            hipDeviceProp_t dev_prop;
            hipDevice_t dev;
            OPUS_HIP_CHECK_ERROR(hipGetDevice(&dev));
            OPUS_HIP_CHECK_ERROR(hipGetDeviceProperties(&dev_prop, dev));
            return dev_prop.multiProcessorCount;
        }();
        return num_cu;
    }

    OPUS_H static constexpr auto MakeKargs(const Hargs& h)
    {
        Kargs k;
        k.p_topk_ids      = h.p_topk_ids;
        k.p_local_tokens  = h.p_local_tokens;
        k.p_expert_mesh   = h.p_ws;
        k.p_expert_cumsum = reinterpret_cast<void*>(
            reinterpret_cast<char*>(h.p_ws) +
            impl::moe_sorting_mp_mesh_smem_size(h.tokens, h.num_experts, h.topk));
        k.tokens              = h.tokens;
        k.mesh_stride         = impl::moe_sorting_mp_mesh_stride(h.tokens);
        k.topk_mdiv           = opus::mdiv{static_cast<uint32_t>(h.topk)};
        k.p_local_expert_mask = h.p_local_expert_mask;
        k.num_experts         = h.num_experts;
        return k;
    }

    OPUS_H static constexpr auto GridSize(const Hargs& h) { return h.num_experts; }

    OPUS_H static constexpr auto BlockSize(const Hargs&) { return dim3(kBlockSize); }

    // in byte
    OPUS_H_D static constexpr auto GetSmemSize()
    {
        return kBlockSize / opus::get_warp_size() * sizeof(IndexType);
    }

    OPUS_D void operator()(Kargs kargs) const
    {
        constexpr opus::index_t index_pack = Problem::SubTokenTile; // always packed
        __shared__ char smem[GetSmemSize()];
        using topk_id_t             = opus::vector_t<IndexType, index_pack>;
        const int eid               = blockIdx.x;
        const topk_id_t* p_topk_ids = reinterpret_cast<const topk_id_t*>(kargs.p_topk_ids);
        const IndexType* p_local_expert_mask =
            static_cast<const IndexType*>(kargs.p_local_expert_mask);
        IndexType* p_expert_cumsum = reinterpret_cast<IndexType*>(kargs.p_expert_cumsum);
        opus::index_t lane_id            = threadIdx.x % opus::get_warp_size();
        opus::index_t wave_id            = threadIdx.x / opus::get_warp_size();
        const opus::index_t tokens       = [&]() {
            if constexpr(Problem::LocalToken)
            {
                return reinterpret_cast<const opus::index_t*>(kargs.p_local_tokens)[0];
            }
            else
            {
                return kargs.tokens;
            }
        }();
        opus::index_t rounded_tokens = [&]() {
            if constexpr(Problem::LocalToken)
            {
                return (tokens + index_pack - 1) / index_pack * index_pack;
            }
            else
                return tokens;
        }();
        opus::index_t mesh_stride = [&]() {
            if constexpr(Problem::LocalToken)
            {
                return impl::moe_sorting_mp_mesh_stride(tokens);
            }
            else
            {
                return kargs.mesh_stride;
            }
        }();

        IndexType mask = 1;
        if constexpr(Problem::LocalExpertMasking)
        {
            mask = p_local_expert_mask[eid];
        }
        MeshType* p_expert_mesh =
            reinterpret_cast<MeshType*>(kargs.p_expert_mesh) + eid * mesh_stride;
        for(opus::index_t i = threadIdx.x; i < mesh_stride; i += kBlockSize)
        {
            p_expert_mesh[i] = 0;
        }
        __syncthreads();

        opus::index_t total_elem = rounded_tokens * kargs.topk_mdiv.divisor / index_pack;

#pragma unroll index_pack
        for(opus::index_t i = threadIdx.x; i < total_elem; i += kBlockSize)
        {
            auto x = p_topk_ids[i];
            opus::static_for<index_pack>([&](auto j) {
                IndexType eid_x = x[j.value]; // ext_vector_type must use int to []
                if(eid_x == eid)
                {
                    uint32_t curr_token_id, curr_topk_id;
                    kargs.topk_mdiv.divmod(i * index_pack + j, curr_token_id, curr_topk_id);
                    if constexpr(Problem::LocalToken)
                    {
                        if(static_cast<opus::index_t>(curr_token_id) < tokens)
                            p_expert_mesh[curr_token_id] = (curr_topk_id + 1) & 0xffff;
                    }
                    else
                        p_expert_mesh[curr_token_id] = (curr_topk_id + 1) & 0xffff;
                }
            });
        }
        __syncthreads();

        {

            using r_t                  = opus::vector_t<MeshType, index_pack>; // always use int32x4
            auto f_sum                 = [](auto x_, auto y_) { return x_ + y_; };
            const r_t* p_expert_mesh_r = reinterpret_cast<r_t*>(p_expert_mesh);

            int loops = (mesh_stride / index_pack + kBlockSize - 1) / kBlockSize;

            if(Problem::LocalToken && mask == 0)
                return;      // skip
            opus::index_t cnt = 0; // per-wave cnt
            for(int i = 0; i < loops; i++)
            {
                int position = i * kBlockSize + threadIdx.x;
                r_t v{0};
                if(position < (mesh_stride / index_pack))
                    v = p_expert_mesh_r[position];
                opus::index_t local_sum = 0;
                opus::static_for<index_pack>(
                    [&](auto i_vec) { local_sum += v[i_vec.value] != 0 ? 1 : 0; });
                cnt += impl::moe_sorting_wave_reduce(local_sum, f_sum);
            }

            // reduce cross wave
            IndexType* s = reinterpret_cast<IndexType*>(smem);
            if(lane_id == 0)
            {
                s[wave_id] = cnt;
            }
            __syncthreads();

            if(threadIdx.x == 0)
            {
                opus::index_t c = 0;
                for(auto i = 0; i < (kBlockSize / opus::get_warp_size()); i++)
                {
                    c += s[i];
                }
                p_expert_cumsum[eid] = c;
            }
        }
    }
};

// cnt total tokens for a expert
template <typename Problem_>
struct MoeSortingMultiPhaseKernel_P1
{
    using Problem = opus::remove_cvref_t<Problem_>;

    using IndexType  = typename Problem::IndexType;
    using WeightType = typename Problem::WeightType;
    using MeshType   = typename Problem::MeshType;

    static constexpr opus::index_t kBlockSize = 256;
    static constexpr opus::index_t OCCUPANCY  = 2; // hard coded

    typedef MoeSortingHostArgs MoeSortingKargs;

    using Hargs = MoeSortingHostArgs;
    struct Kargs
    {
        const void* p_local_expert_mask; // [expert]
        const void* p_local_tokens;      // [1], if not nullptr, use this as actual tokens
        void* p_expert_mesh;             // [expert, tokens]
        void* p_expert_cumsum;
        opus::index_t mesh_stride; // mesh_stride for p_expert_mesh
    };

    OPUS_H static constexpr auto MakeKargs(const Hargs& h)
    {
        Kargs k;
        k.p_local_expert_mask = h.p_local_expert_mask;
        k.p_local_tokens      = h.p_local_tokens;
        k.p_expert_mesh       = h.p_ws;
        k.p_expert_cumsum     = reinterpret_cast<void*>(
            reinterpret_cast<char*>(h.p_ws) +
            impl::moe_sorting_mp_mesh_smem_size(h.tokens, h.num_experts, h.topk));
        k.mesh_stride = impl::moe_sorting_mp_mesh_stride(h.tokens);

        return k;
    }

    OPUS_H static constexpr auto GridSize(const Hargs& h) { return dim3(h.num_experts); }

    OPUS_H static constexpr auto BlockSize(const Hargs&) { return dim3(kBlockSize); }

    // in byte
    OPUS_H_D static constexpr auto GetSmemSize()
    {
        return kBlockSize / opus::get_warp_size() * sizeof(IndexType);
    }

    OPUS_D void operator()(Kargs kargs) const
    {
        __shared__ char smem[GetSmemSize()];

        int eid                      = blockIdx.x;
        constexpr opus::index_t index_pack = Problem::SubTokenTile;              // always packed
        using r_t                    = opus::vector_t<MeshType, index_pack>; // always use int32x4

        const IndexType* p_local_expert_mask =
            static_cast<const IndexType*>(kargs.p_local_expert_mask);
        IndexType* p_expert_cumsum = reinterpret_cast<IndexType*>(kargs.p_expert_cumsum);

        auto f_sum = [](auto x_, auto y_) { return x_ + y_; };

        opus::index_t tokens = [&]() {
            if constexpr(Problem::LocalToken)
            {
                return reinterpret_cast<const opus::index_t*>(kargs.p_local_tokens)[0];
            }
            else
            {
                return 0; // will not use if not LocalToken
            }
        }();

        opus::index_t mesh_stride = [&]() {
            if constexpr(Problem::LocalToken)
            {
                return impl::moe_sorting_mp_mesh_stride(tokens);
            }
            else
            {
                return kargs.mesh_stride;
            }
        }();

        r_t* p_expert_mesh = reinterpret_cast<r_t*>(
            reinterpret_cast<MeshType*>(kargs.p_expert_mesh) + eid * mesh_stride);

        int loops = (mesh_stride / index_pack + kBlockSize - 1) / kBlockSize;

        if constexpr(Problem::LocalExpertMasking)
        {
            IndexType mask = p_local_expert_mask[eid];
            if(mask == 0)
                return; // skip
        }

        opus::index_t cnt = 0; // per-wave cnt
        for(int i = 0; i < loops; i++)
        {
            int position = i * kBlockSize + threadIdx.x;
            r_t v{0};
            if(position < (mesh_stride / index_pack))
                v = p_expert_mesh[position];
            opus::index_t local_sum = 0;
            opus::static_for<index_pack>(
                [&](auto i_vec) { local_sum += v[i_vec.value] != 0 ? 1 : 0; });
            cnt += impl::moe_sorting_wave_reduce(local_sum, f_sum);
        }

        opus::index_t lane_id = threadIdx.x % opus::get_warp_size();
        opus::index_t wave_id = threadIdx.x / opus::get_warp_size();

        // reduce cross wave
        IndexType* s = reinterpret_cast<IndexType*>(smem);
        if(lane_id == 0)
        {
            s[wave_id] = cnt;
        }
        __syncthreads();

        if(threadIdx.x == 0)
        {
            opus::index_t c = 0;
            for(auto i = 0; i < (kBlockSize / opus::get_warp_size()); i++)
            {
                c += s[i];
            }
            p_expert_cumsum[eid] = c;
        }
    }
};

#if MOE_SORTING_FUSE_MP_01
template <typename Problem_>
struct MoeSortingMultiPhaseKernel_P01
{
    using Problem = opus::remove_cvref_t<Problem_>;

    using IndexType  = typename Problem::IndexType;
    using WeightType = typename Problem::WeightType;
    using MeshType   = typename Problem::MeshType;

    static constexpr opus::index_t kBlockSize = 256;
    static constexpr opus::index_t OCCUPANCY  = 2; // hard coded

    typedef MoeSortingHostArgs MoeSortingKargs;

    using Hargs = MoeSortingHostArgs;

    struct Kargs
    {
        const void* p_topk_ids;          // [tokens, topk]
        const void* p_local_expert_mask; // [expert]
        const void* p_local_tokens;      // [1]
        void* p_expert_mesh;             // [expert, tokens]
        void* p_expert_cumsum;           // [expert + 1]
        void* p_expert_sem;              // [1]
        opus::index_t tokens;
        opus::index_t num_experts;
        opus::index_t mesh_stride; // mesh_stride for p_expert_mesh
        opus::index_t wg_count;    // used for semaphore
        opus::mdiv topk_mdiv;
    };

    OPUS_H static constexpr auto get_num_cu()
    {
        opus::index_t num_cu = [&]() {
            hipDeviceProp_t dev_prop;
            hipDevice_t dev;
            OPUS_HIP_CHECK_ERROR(hipGetDevice(&dev));
            OPUS_HIP_CHECK_ERROR(hipGetDeviceProperties(&dev_prop, dev));
            return dev_prop.multiProcessorCount;
        }();
        return num_cu;
    }

    OPUS_H static constexpr auto MakeKargs(const Hargs& h)
    {
        Kargs k;
        k.p_topk_ids          = h.p_topk_ids;
        k.p_local_expert_mask = h.p_local_expert_mask;
        k.p_local_tokens      = h.p_local_tokens;
        k.p_expert_mesh       = h.p_ws;
        k.p_expert_cumsum     = reinterpret_cast<void*>(
            reinterpret_cast<char*>(h.p_ws) +
            impl::moe_sorting_mp_mesh_smem_size(h.tokens, h.num_experts, h.topk));
        k.p_expert_sem = reinterpret_cast<void*>(
            reinterpret_cast<char*>(h.p_ws) +
            impl::moe_sorting_mp_mesh_smem_size(h.tokens, h.num_experts, h.topk) +
            impl::moe_sorting_mp_cumsum_smem_size(h.num_experts));
        k.tokens      = h.tokens;
        k.num_experts = h.num_experts;
        k.mesh_stride = impl::moe_sorting_mp_mesh_stride(h.tokens);
        k.wg_count    = [&]() {
            if constexpr(Problem::LocalToken)
            {
                return GridSize(h);
            }
            else
            {
                return WGCounts(h);
            }
        }();
        k.topk_mdiv = opus::mdiv{static_cast<uint32_t>(h.topk)};
        return k;
    }

    OPUS_H static constexpr auto GridSize(const Hargs&) { return get_num_cu() * OCCUPANCY; }

    OPUS_H static constexpr auto BlockSize(const Hargs&) { return dim3(kBlockSize); }

    OPUS_H static constexpr auto WGCounts(const Hargs& h)
    {
        opus::index_t total_elem = h.tokens * h.topk / Problem::SubTokenTile;
        opus::index_t elem_cnt   = (total_elem + kBlockSize - 1) / kBlockSize;

        // no more than grid_size
        return min(elem_cnt, GridSize(h));
    }

    // in byte
    OPUS_H_D static constexpr auto GetSmemSize()
    {
        return kBlockSize / opus::get_warp_size() * sizeof(IndexType);
    }

    OPUS_D void operator()(Kargs kargs) const
    {
        opus::workgroup_barrier wb{reinterpret_cast<uint32_t*>(kargs.p_expert_sem)};
        opus::index_t tokens = [&]() {
            if constexpr(Problem::LocalToken)
            {
                return reinterpret_cast<const opus::index_t*>(kargs.p_local_tokens)[0];
            }
            else
            {
                return kargs.tokens;
            }
        }();
        opus::index_t rounded_tokens = [&]() {
            if constexpr(Problem::LocalToken)
            {
                return (tokens + Problem::SubTokenTile - 1) / Problem::SubTokenTile *
                       Problem::SubTokenTile;
            }
            else
                return tokens;
        }();
        opus::index_t wg_count = [&]() {
            if constexpr(Problem::LocalToken)
            {
                opus::index_t total_elem = rounded_tokens * kargs.topk / Problem::SubTokenTile;
                opus::index_t elem_cnt   = (total_elem + kBlockSize - 1) / kBlockSize;

                // no more than grid_size
                return min(elem_cnt, kargs.wg_count);
            }
            else
            {
                return kargs.wg_count;
            }
        }();

        {
            using topk_id_t = opus::vector_t<IndexType, Problem::SubTokenTile>;

            const topk_id_t* p_topk_ids = reinterpret_cast<const topk_id_t*>(kargs.p_topk_ids);
            IndexType* p_expert_mesh    = reinterpret_cast<IndexType*>(kargs.p_expert_mesh);
            opus::index_t total_elem = rounded_tokens * kargs.topk_mdiv.divisor / Problem::SubTokenTile;

#pragma unroll Problem::SubTokenTile
            for(opus::index_t i = blockIdx.x * kBlockSize + threadIdx.x; i < total_elem;
                i += kBlockSize * gridDim.x)
            {
                auto x = p_topk_ids[i];
                opus::static_for<Problem::SubTokenTile>([&](auto j) {
                    IndexType eid = x[j.value]; // ext_vector_type must use int to []
                    uint32_t curr_token_id, curr_topk_id;
                    kargs.topk_mdiv.divmod(
                        i * Problem::SubTokenTile + j, curr_token_id, curr_topk_id);
                    // p_expert_mesh[eid * kargs.mesh_stride + curr_token_id] = curr_topk_id + 1;
                    if constexpr(Problem::LocalToken)
                    {
                        if(static_cast<opus::index_t>(curr_token_id) < tokens)
                            p_expert_mesh[eid * kargs.mesh_stride + curr_token_id] =
                                (curr_topk_id + 1) & 0xffff;
                    }
                    else
                        p_expert_mesh[eid * kargs.mesh_stride + curr_token_id] =
                            (curr_topk_id + 1) & 0xffff;
                });
            }
            if(static_cast<opus::index_t>(blockIdx.x) < wg_count)
            {
                wb.inc();
            }
        }

        {
            __shared__ char smem[GetSmemSize()];
            int eid = blockIdx.x;

            // early exist in case of extra atomic wait
            if(eid >= kargs.num_experts)
                return;

            wb.wait_lt(wg_count);

            for(; eid < kargs.num_experts; eid += gridDim.x)
            {
                // if(threadIdx.x == 0)
                //     printf("!!! bid:%d, eid:%d (%d, %d)\n",
                //            static_cast<int>(blockIdx.x),
                //            eid,
                //            kargs.num_experts,
                //            static_cast<int>(blockDim.x));
                constexpr opus::index_t index_pack = 4;                         // always packed
                using r_t          = opus::vector_t<IndexType, index_pack>; // always use int32x4
                r_t* p_expert_mesh = reinterpret_cast<r_t*>(
                    reinterpret_cast<opus::index_t*>(kargs.p_expert_mesh) + eid * kargs.mesh_stride);

                const IndexType* p_local_expert_mask =
                    static_cast<const IndexType*>(kargs.p_local_expert_mask);
                IndexType* p_expert_cumsum = reinterpret_cast<IndexType*>(kargs.p_expert_cumsum);

                auto f_sum = [](auto x_, auto y_) { return x_ + y_; };

                int loops = (kargs.mesh_stride / index_pack + kBlockSize - 1) / kBlockSize;

                if constexpr(Problem::LocalExpertMasking)
                {
                    IndexType mask = p_local_expert_mask[eid];
                    if(mask == 0)
                        continue; // skip
                }

                opus::index_t cnt = 0; // per-wave cnt
                for(int i = 0; i < loops; i++)
                {
                    int position = i * kBlockSize + threadIdx.x;
                    r_t v{0};
                    if(position < (kargs.mesh_stride / index_pack))
                        v = p_expert_mesh[position];
                    opus::index_t local_sum = 0;
                    opus::static_for<index_pack>(
                        [&](auto i_vec) { local_sum += v[i_vec.value] != 0 ? 1 : 0; });
                    cnt += impl::moe_sorting_wave_reduce(local_sum, f_sum);
                }

                opus::index_t lane_id = threadIdx.x % opus::get_warp_size();
                opus::index_t wave_id = threadIdx.x / opus::get_warp_size();

                // reduce cross wave
                IndexType* s = reinterpret_cast<IndexType*>(smem);
                __syncthreads();
                if(lane_id == 0)
                {
                    s[wave_id] = cnt;
                }
                __syncthreads();

                if(threadIdx.x == 0)
                {
                    opus::index_t c = 0;
                    for(auto i = 0; i < (kBlockSize / opus::get_warp_size()); i++)
                    {
                        c += s[i];
                    }
                    p_expert_cumsum[eid] = c;
                }
            }
        }
    }
};
#endif

// token count cumsum
template <typename Problem_>
struct MoeSortingMultiPhaseKernel_P2
{
    using Problem = opus::remove_cvref_t<Problem_>;

    using IndexType  = typename Problem::IndexType;
    using WeightType = typename Problem::WeightType;
    using MeshType   = typename Problem::MeshType;

    static constexpr opus::index_t kBlockSize = 256;
    static constexpr opus::index_t OCCUPANCY  = 2; // hard coded

    typedef MoeSortingHostArgs MoeSortingKargs;

    using Hargs = MoeSortingHostArgs;
    struct Kargs
    {
        const void* p_local_expert_mask; // [expert]
        const void* p_local_tokens;      // [1]
        void* p_expert_mesh;             // [expert, tokens]
        void* p_expert_cumsum;           // [expert + 1]
        void* p_total_tokens_post_pad;   // [2]
        void* p_sorted_expert_ids;
        void* p_moe_buf;
        opus::index_t tokens;
        opus::index_t num_experts;
        opus::index_t mesh_stride; // mesh_stride for p_expert_mesh
        opus::mdiv unit_size_mdiv;
        opus::index_t moe_buf_interm_dim;
        opus::index_t moe_buf_elem_bytes;
    };

    OPUS_H static constexpr auto MakeKargs(const Hargs& h)
    {
        Kargs k;
        k.p_local_expert_mask = h.p_local_expert_mask;
        k.p_local_tokens      = h.p_local_tokens;
        k.p_expert_cumsum     = reinterpret_cast<void*>(
            reinterpret_cast<char*>(h.p_ws) +
            impl::moe_sorting_mp_mesh_smem_size(h.tokens, h.num_experts, h.topk));
        k.p_total_tokens_post_pad = h.p_total_tokens_post_pad;
        k.p_sorted_expert_ids     = h.p_sorted_expert_ids;

        k.p_moe_buf = h.p_moe_buf;

        k.tokens         = h.tokens;
        k.num_experts    = h.num_experts;
        k.mesh_stride    = impl::moe_sorting_mp_mesh_stride(h.tokens);
        k.unit_size_mdiv = opus::mdiv{static_cast<uint32_t>(h.unit_size)};

        k.moe_buf_interm_dim = h.moe_buf_interm_dim;
        k.moe_buf_elem_bytes = h.moe_buf_elem_bytes;

        return k;
    }

    OPUS_H static constexpr auto get_num_cu()
    {
        opus::index_t num_cu = [&]() {
            hipDeviceProp_t dev_prop;
            hipDevice_t dev;
            OPUS_HIP_CHECK_ERROR(hipGetDevice(&dev));
            OPUS_HIP_CHECK_ERROR(hipGetDeviceProperties(&dev_prop, dev));
            return dev_prop.multiProcessorCount;
        }();
        return num_cu;
    }

    OPUS_H static constexpr auto GridSize(const Hargs& h)
    {
        return dim3(h.num_experts + get_num_cu() * OCCUPANCY);
    }

    OPUS_H static constexpr auto BlockSize(const Hargs&) { return dim3(kBlockSize); }

    // in byte
    OPUS_H_D static constexpr auto GetSmemSize()
    {
        return (4 + 2 * kBlockSize / opus::get_warp_size()) * sizeof(IndexType);
    }

    // reduce single pixel within a wave
    OPUS_D void operator()(Kargs kargs) const
    {
        if(blockIdx.x > 0)
        {
            impl::moe_buf_set_zero_kernel_2d<kBlockSize>(kargs.p_moe_buf,
                                                         kargs.tokens,
                                                         kargs.moe_buf_interm_dim,
                                                         kargs.moe_buf_elem_bytes,
                                                         blockIdx.x - 1,
                                                         gridDim.x - 1);
            return;
        }
        __shared__ char smem[GetSmemSize()];
        IndexType* s = reinterpret_cast<IndexType*>(smem);

        const IndexType* p_local_expert_mask =
            static_cast<const IndexType*>(kargs.p_local_expert_mask);
        IndexType* p_expert_cumsum = reinterpret_cast<IndexType*>(kargs.p_expert_cumsum);
        IndexType* p_total_tokens_post_pad =
            reinterpret_cast<IndexType*>(kargs.p_total_tokens_post_pad);
        IndexType* p_sorted_expert_ids = reinterpret_cast<IndexType*>(kargs.p_sorted_expert_ids);

        const opus::index_t loops = (kargs.num_experts + kBlockSize - 1) / kBlockSize;
        opus::index_t wave_id     = threadIdx.x / opus::get_warp_size();
        opus::index_t lane_id     = threadIdx.x % opus::get_warp_size();

        IndexType prev_cumsum_a = 0;
        IndexType prev_cumsum_b = 0;

        for(opus::index_t i = 0; i < loops; i++)
        {
            opus::index_t position = i * kBlockSize + threadIdx.x;
            IndexType a_     = 0; // token count for a expert
            IndexType b_     = 0; // mask for a expert
            if(position < kargs.num_experts)
            {
                a_ = p_expert_cumsum[position];
                if constexpr(Problem::LocalExpertMasking)
                    b_ = p_local_expert_mask[position];
            }

            int blocks_pers_expert =
                kargs.unit_size_mdiv.div(a_ + kargs.unit_size_mdiv.divisor - 1);
            // pad token
            int padded_blocks_per_expert = [&]() {
                int x_ = [&]() {
                    if constexpr(Problem::SkipExpertsWithZeroTokens)
                    {
                        // if local_cnt is zero, blocks_pers_expert will be zero
                        // this is what we want to achieve
                        return blocks_pers_expert; //  * kargs.unit_size_mdiv.divisor;
                    }
                    else
                    {
                        return max(blocks_pers_expert, 1);
                    }
                }();
                if constexpr(Problem::LocalExpertMasking)
                {
                    return b_ ? x_ : 0;
                }
                else
                    return x_;
            }();

            IndexType cumsum_a = padded_blocks_per_expert;
            IndexType cumsum_b = b_;

            // Note: we first cumsum local round, then add previous cumsum
            impl::moe_sorting_wave_cumsum<IndexType, opus::get_warp_size()>(cumsum_a);
            impl::moe_sorting_wave_cumsum<IndexType, opus::get_warp_size()>(cumsum_b);

            __syncthreads();
            if(lane_id == opus::get_warp_size() - 1)
            {
                s[4 + wave_id]                                = cumsum_a;
                s[4 + wave_id + kBlockSize / opus::get_warp_size()] = cumsum_b;
            }

            __syncthreads();

            // reduce cross wave
            opus::static_for<kBlockSize / opus::get_warp_size() - 1>([&](auto i_w) {
                IndexType prev_a = s[4 + i_w];
                IndexType prev_b = s[4 + i_w + kBlockSize / opus::get_warp_size()];
                prev_a           = wave_id > i_w ? prev_a : 0; // mask out
                prev_b           = wave_id > i_w ? prev_b : 0; // mask out
                cumsum_a += prev_a;
                cumsum_b += prev_b;
            });

            // Now let's add previous cumsum
            cumsum_a += prev_cumsum_a;
            cumsum_b += prev_cumsum_b;

            if(threadIdx.x == kBlockSize - 1)
            {
                s[2] = cumsum_a; // store the last cumsum
                s[3] = cumsum_b;
            }

            IndexType out_0 = cumsum_a - padded_blocks_per_expert; // exclusive cumsum tok cnt
            IndexType out_1 = cumsum_b - b_;                       // exclusive cumsum mask cnt

            __syncthreads();
            prev_cumsum_a = s[2];
            prev_cumsum_b = s[3];

            if(position < kargs.num_experts)
            {
                p_expert_cumsum[position] = out_0 * kargs.unit_size_mdiv.divisor;
            }

            {
                if constexpr(Problem::LocalExpertMasking)
                {
                    if(b_)
                    {
                        for(int j = 0; j < blocks_pers_expert; j++)
                        {
                            p_sorted_expert_ids[out_0 + j] = out_1;
                        }
                    }
                }
                else
                {
                    for(int j = 0; j < blocks_pers_expert; j++)
                    {
                        p_sorted_expert_ids[out_0 + j] = position;
                    }
                }
            }
        }

        if(threadIdx.x == 0)
        {
            auto total_tokens_post_pad         = prev_cumsum_a * kargs.unit_size_mdiv.divisor;
            p_total_tokens_post_pad[0]         = total_tokens_post_pad;
            p_expert_cumsum[kargs.num_experts] = total_tokens_post_pad;
        }
    }
};

template <typename Problem_>
struct MoeSortingMultiPhaseKernel_P3
{
    using Problem = opus::remove_cvref_t<Problem_>;

    using IndexType  = typename Problem::IndexType;
    using WeightType = typename Problem::WeightType;
    using MeshType   = typename Problem::MeshType;

    static constexpr opus::index_t kBlockSize = 256;
    static constexpr opus::index_t OCCUPANCY  = 2; // hard coded

    typedef MoeSortingHostArgs MoeSortingKargs;

    using Hargs = MoeSortingHostArgs;

    struct Kargs
    {
        const void* p_weights;
        const void* p_local_expert_mask;
        const void* p_local_tokens;
        void* p_sorted_token_ids;
        void* p_sorted_weights;
        void* p_expert_mesh; // [token, expert]
        void* p_expert_cumsum;

        opus::index_t tokens;
        opus::index_t num_experts;
        opus::index_t mesh_stride; // mesh_stride for p_expert_mesh
        opus::mdiv topk_mdiv;
    };

    OPUS_H static constexpr auto MakeKargs(const Hargs& h)
    {
        Kargs k;
        k.p_weights           = h.p_weights;
        k.p_local_expert_mask = h.p_local_expert_mask;
        k.p_local_tokens      = h.p_local_tokens;
        k.p_sorted_token_ids  = h.p_sorted_token_ids;
        k.p_sorted_weights    = h.p_sorted_weights;
        k.p_expert_mesh       = h.p_ws;
        k.p_expert_cumsum     = reinterpret_cast<void*>(
            reinterpret_cast<char*>(h.p_ws) +
            impl::moe_sorting_mp_mesh_smem_size(h.tokens, h.num_experts, h.topk));
        k.tokens      = h.tokens;
        k.num_experts = h.num_experts;
        k.topk_mdiv   = opus::mdiv{static_cast<uint32_t>(h.topk)};
        k.mesh_stride = impl::moe_sorting_mp_mesh_stride(h.tokens);
        return k;
    }

    OPUS_H static constexpr auto GridSize(const Hargs& h) { return dim3(h.num_experts); }

    OPUS_H static constexpr auto BlockSize(const Hargs&) { return dim3(kBlockSize); }

    // in byte
    OPUS_H_D static constexpr auto GetSmemSize()
    {
        return (4 + kBlockSize / opus::get_warp_size()) * sizeof(IndexType);
    }

    OPUS_D void operator()(Kargs kargs) const
    {
        __shared__ char smem[GetSmemSize()];

        const IndexType* p_local_expert_mask =
            static_cast<const IndexType*>(kargs.p_local_expert_mask);
        IndexType* s                  = reinterpret_cast<IndexType*>(smem);
        IndexType* p_expert_mesh      = reinterpret_cast<IndexType*>(kargs.p_expert_mesh);
        IndexType* p_sorted_token_ids = reinterpret_cast<IndexType*>(kargs.p_sorted_token_ids);
        IndexType* p_expert_cumsum    = reinterpret_cast<IndexType*>(kargs.p_expert_cumsum);
        const WeightType* p_weights   = static_cast<const WeightType*>(kargs.p_weights);
        WeightType* p_sorted_weights  = reinterpret_cast<WeightType*>(kargs.p_sorted_weights);

        opus::index_t tokens = [&]() {
            if constexpr(Problem::LocalToken)
            {
                return reinterpret_cast<const opus::index_t*>(kargs.p_local_tokens)[0];
            }
            else
            {
                return kargs.tokens;
            }
        }();
        int eid     = blockIdx.x;
        int wave_id = threadIdx.x / opus::get_warp_size();
        int lane_id = threadIdx.x % opus::get_warp_size();
        int e_start = p_expert_cumsum[eid];
        int e_end   = p_expert_cumsum[eid + 1];
        if constexpr(Problem::SkipExpertsWithZeroTokens)
        {
            if(e_start == e_end)
                return;
        }

        if constexpr(Problem::LocalExpertMasking)
        {
            int e_mask = p_local_expert_mask[eid];
            if(e_mask == 0)
                return; // skip empty expert
        }

        // cumsum one by one
        int loops       = (kargs.mesh_stride + kBlockSize - 1) / kBlockSize;
        int prev_cumsum = 0;
        for(int i = 0; i < loops; i++)
        {
            int i_token = i * kBlockSize + threadIdx.x;
            IndexType x = 0;
            if(i_token < tokens)
            {
                x = p_expert_mesh[eid * kargs.mesh_stride + i_token];
            }
            int i_topk = x - 1;          // topk of this token
            int i_show = x != 0 ? 1 : 0; // has this token or not
            int cumsum = i_show;
            impl::moe_sorting_wave_cumsum<int, opus::get_warp_size()>(cumsum);

            __syncthreads();
            if(lane_id == opus::get_warp_size() - 1)
            {
                s[4 + wave_id] = cumsum;
            }
            __syncthreads();

            // reduce cross wave
            opus::static_for<kBlockSize / opus::get_warp_size() - 1>([&](auto i_w) {
                IndexType prev = s[4 + i_w];
                prev           = wave_id > i_w ? prev : 0; // mask out
                cumsum += prev;
            });
            cumsum += prev_cumsum; // add previous round cumsum
            if(threadIdx.x == kBlockSize - 1)
            {
                s[0] = cumsum;
            }
            __syncthreads();

            int position = cumsum - i_show;
            prev_cumsum  = s[0]; // update the last cumsum

            if(i_show)
            {
#if OPUS_MOE_SORTING_MOCK_ID
                p_sorted_token_ids[e_start + position] = MOE_SORTING_MOCK_ID(i_token, i_topk);
#else
                p_sorted_token_ids[e_start + position] = i_token;
#endif
                p_sorted_weights[e_start + position] =
                    p_weights[i_token * kargs.topk_mdiv.divisor + i_topk];
            }
        }

        for(opus::index_t i = e_start + prev_cumsum + threadIdx.x; i < e_end; i += kBlockSize)
        {
#if OPUS_MOE_SORTING_MOCK_ID
            p_sorted_token_ids[i] = MOE_SORTING_MOCK_ID(tokens, kargs.topk_mdiv.divisor);
#else
            p_sorted_token_ids[i] = tokens;
#endif
            p_sorted_weights[i] = static_cast<WeightType>(0.0);
        }
    }
};

namespace impl {
// we use dynamic LDS size here
OPUS_H constexpr auto moe_sorting_get_smem_size_p23(int num_experts_)
{
    constexpr opus::index_t kBlockSize     = 256; // hardcoded 256
    const opus::index_t expert_cumsum_elem = num_experts_ + 1;
    return (4 + 2 * kBlockSize / opus::get_warp_size() + expert_cumsum_elem) * sizeof(int);
}
} // namespace impl

// token count cumsum
template <typename Problem_>
struct MoeSortingMultiPhaseKernel_P23
{
    using Problem = opus::remove_cvref_t<Problem_>;

    using IndexType  = typename Problem::IndexType;
    using WeightType = typename Problem::WeightType;
    using MeshType   = typename Problem::MeshType;

    static constexpr opus::index_t kBlockSize = 256;
    static constexpr opus::index_t OCCUPANCY  = 2; // hard coded

    typedef MoeSortingHostArgs MoeSortingKargs;

    using Hargs = MoeSortingHostArgs;
    struct Kargs
    {
        const void* p_weights;
        const void* p_local_expert_mask; // [expert]
        const void* p_local_tokens;      // [1]
        void* p_expert_mesh;             // [expert, tokens]
        void* p_expert_cumsum;           // [expert + 1]
        void* p_total_tokens_post_pad;   // [2]
        void* p_sorted_expert_ids;

        void* p_sorted_token_ids;
        void* p_sorted_weights;
        void* p_moe_buf;

        opus::index_t tokens;
        opus::index_t num_experts;
        opus::index_t mesh_stride; // mesh_stride for p_expert_mesh
        opus::mdiv unit_size_mdiv;
        opus::mdiv topk_mdiv;
        // NOTE:
        // moe_buf_* is a 2d ws buffer used for the following fmoe kernel
        // arranged as row*col, where row=tokens(or local_token), col=interm_dim
        // we fuse this clearing inside sorting kernel
        // Besides, we require inter_dim to be multiple of 16 byte(make sure when alloc ws for fmoe)
        opus::index_t moe_buf_interm_dim; // p_moe_buf interm_dim
        opus::index_t moe_buf_elem_bytes; // p_moe_buf byte size(8bit, 16bit, 32bit, etc.)
    };

    OPUS_H static constexpr auto MakeKargs(const Hargs& h)
    {
        Kargs k;
        k.p_weights           = h.p_weights;
        k.p_local_expert_mask = h.p_local_expert_mask;
        k.p_local_tokens      = h.p_local_tokens;
        k.p_expert_mesh       = h.p_ws;
        k.p_expert_cumsum     = reinterpret_cast<void*>(
            reinterpret_cast<char*>(h.p_ws) +
            impl::moe_sorting_mp_mesh_smem_size(h.tokens, h.num_experts, h.topk));
        k.p_total_tokens_post_pad = h.p_total_tokens_post_pad;
        k.p_sorted_expert_ids     = h.p_sorted_expert_ids;

        k.p_sorted_token_ids = h.p_sorted_token_ids;
        k.p_sorted_weights   = h.p_sorted_weights;

        k.p_moe_buf = h.p_moe_buf;

        k.tokens         = h.tokens;
        k.num_experts    = h.num_experts;
        k.mesh_stride    = impl::moe_sorting_mp_mesh_stride(h.tokens);
        k.unit_size_mdiv = opus::mdiv{static_cast<uint32_t>(h.unit_size)};
        k.topk_mdiv      = opus::mdiv{static_cast<uint32_t>(h.topk)};

        k.moe_buf_interm_dim = h.moe_buf_interm_dim;
        k.moe_buf_elem_bytes = h.moe_buf_elem_bytes;

        return k;
    }

    OPUS_H static constexpr auto get_num_cu()
    {
        opus::index_t num_cu = [&]() {
            hipDeviceProp_t dev_prop;
            hipDevice_t dev;
            OPUS_HIP_CHECK_ERROR(hipGetDevice(&dev));
            OPUS_HIP_CHECK_ERROR(hipGetDeviceProperties(&dev_prop, dev));
            return dev_prop.multiProcessorCount;
        }();
        return num_cu;
    }

    OPUS_H static constexpr auto GridSize(const Hargs& h)
    {
        return dim3(h.num_experts + get_num_cu() * OCCUPANCY);
    }

    OPUS_H static constexpr auto BlockSize(const Hargs&) { return dim3(kBlockSize); }

    // only use this at host !
    OPUS_H static constexpr auto GetSmemSize(const Hargs& h)
    {
        const auto smem_23 = impl::moe_sorting_get_smem_size_p23(h.num_experts);
        const auto smem_sf = kBlockSize * 4 * sizeof(IndexType);
        return max(smem_23, smem_sf);
    }

    // reduce single pixel within a wave
    OPUS_D void operator()(Kargs kargs) const
    {
        opus::index_t tokens = [&]() {
            if constexpr(Problem::LocalToken)
            {
                return reinterpret_cast<const opus::index_t*>(kargs.p_local_tokens)[0];
            }
            else
            {
                return kargs.tokens;
            }
        }();

        if(static_cast<opus::index_t>(blockIdx.x) >= kargs.num_experts)
        {
            impl::moe_buf_set_zero_kernel_2d<kBlockSize>(kargs.p_moe_buf,
                                                         tokens,
                                                         kargs.moe_buf_interm_dim,
                                                         kargs.moe_buf_elem_bytes,
                                                         blockIdx.x - kargs.num_experts,
                                                         gridDim.x - kargs.num_experts);
            return;
        }

        extern __shared__ char smem[];
        {
            IndexType* s = reinterpret_cast<IndexType*>(smem);

            const IndexType* p_local_expert_mask =
                static_cast<const IndexType*>(kargs.p_local_expert_mask);
            IndexType* p_expert_cumsum      = reinterpret_cast<IndexType*>(kargs.p_expert_cumsum);
            IndexType* p_expert_cumsum_smem = s + 4 + 2 * kBlockSize / opus::get_warp_size();
            IndexType* p_total_tokens_post_pad =
                reinterpret_cast<IndexType*>(kargs.p_total_tokens_post_pad);
            IndexType* p_sorted_expert_ids =
                reinterpret_cast<IndexType*>(kargs.p_sorted_expert_ids);

            const opus::index_t loops = (kargs.num_experts + kBlockSize - 1) / kBlockSize;
            opus::index_t wave_id     = threadIdx.x / opus::get_warp_size();
            opus::index_t lane_id     = threadIdx.x % opus::get_warp_size();

            IndexType prev_cumsum_a = 0;
            IndexType prev_cumsum_b = 0;

            for(opus::index_t i = 0; i < loops; i++)
            {
                opus::index_t position = i * kBlockSize + threadIdx.x;
                IndexType a_     = 0; // token count for a expert
                IndexType b_     = 0; // mask for a expert
                if(position < kargs.num_experts)
                {
                    a_ = p_expert_cumsum[position];
                    if constexpr(Problem::LocalExpertMasking)
                        b_ = p_local_expert_mask[position];
                }

                int blocks_pers_expert =
                    kargs.unit_size_mdiv.div(a_ + kargs.unit_size_mdiv.divisor - 1);
                // pad token
                int padded_blocks_per_expert = [&]() {
                    int x_ = [&]() {
                        if constexpr(Problem::SkipExpertsWithZeroTokens)
                        {
                            // if local_cnt is zero, blocks_pers_expert will be zero
                            // this is what we want to achieve
                            return blocks_pers_expert; //  * kargs.unit_size_mdiv.divisor;
                        }
                        else
                        {
                            return max(blocks_pers_expert, 1);
                        }
                    }();
                    if constexpr(Problem::LocalExpertMasking)
                    {
                        return b_ ? x_ : 0;
                    }
                    else
                        return x_;
                }();

                IndexType cumsum_a = padded_blocks_per_expert;
                IndexType cumsum_b = b_;

                // Note: we first cumsum local round, then add previous cumsum
                impl::moe_sorting_wave_cumsum<IndexType, opus::get_warp_size()>(cumsum_a);
                impl::moe_sorting_wave_cumsum<IndexType, opus::get_warp_size()>(cumsum_b);

                __syncthreads();
                if(lane_id == opus::get_warp_size() - 1)
                {
                    s[4 + wave_id]                                = cumsum_a;
                    s[4 + wave_id + kBlockSize / opus::get_warp_size()] = cumsum_b;
                }

                __syncthreads();

                // reduce cross wave
                opus::static_for<kBlockSize / opus::get_warp_size() - 1>([&](auto i_w) {
                    IndexType prev_a = s[4 + i_w];
                    IndexType prev_b = s[4 + i_w + kBlockSize / opus::get_warp_size()];
                    prev_a           = wave_id > i_w ? prev_a : 0; // mask out
                    prev_b           = wave_id > i_w ? prev_b : 0; // mask out
                    cumsum_a += prev_a;
                    cumsum_b += prev_b;
                });

                // Now let's add previous cumsum
                cumsum_a += prev_cumsum_a;
                cumsum_b += prev_cumsum_b;

                if(threadIdx.x == kBlockSize - 1)
                {
                    s[2] = cumsum_a; // store the last cumsum
                    s[3] = cumsum_b;
                }

                IndexType out_0 = cumsum_a - padded_blocks_per_expert; // exclusive cumsum tok cnt
                IndexType out_1 = cumsum_b - b_;                       // exclusive cumsum mask cnt

                __syncthreads();
                prev_cumsum_a = s[2];
                prev_cumsum_b = s[3];

                if(position < kargs.num_experts)
                {
                    p_expert_cumsum_smem[position] = out_0 * kargs.unit_size_mdiv.divisor;
                }

                {
                    if(blockIdx.x == 0)
                    {
                        if constexpr(Problem::LocalExpertMasking)
                        {
                            if(b_)
                            {
                                for(int j = 0; j < blocks_pers_expert; j++)
                                {
                                    p_sorted_expert_ids[out_0 + j] = out_1;
                                }
                            }
                        }
                        else
                        {
                            for(int j = 0; j < blocks_pers_expert; j++)
                            {
                                p_sorted_expert_ids[out_0 + j] = position;
                            }
                        }
                    }
                }
            }

            if(threadIdx.x == 0)
            {
                auto total_tokens_post_pad = prev_cumsum_a * kargs.unit_size_mdiv.divisor;
                if(blockIdx.x == 0)
                {
                    p_total_tokens_post_pad[0] = total_tokens_post_pad;
                    p_total_tokens_post_pad[1] = tokens;
                }
                p_expert_cumsum_smem[kargs.num_experts] = total_tokens_post_pad;
            }
        }

        __syncthreads();
        {
            const IndexType* p_local_expert_mask =
                static_cast<const IndexType*>(kargs.p_local_expert_mask);
            IndexType* s                  = reinterpret_cast<IndexType*>(smem);
            MeshType* p_expert_mesh       = reinterpret_cast<MeshType*>(kargs.p_expert_mesh);
            IndexType* p_sorted_token_ids = reinterpret_cast<IndexType*>(kargs.p_sorted_token_ids);
            IndexType* p_expert_cumsum_smem = s + 4 + 2 * kBlockSize / opus::get_warp_size();
            const WeightType* p_weights     = static_cast<const WeightType*>(kargs.p_weights);
            WeightType* p_sorted_weights    = reinterpret_cast<WeightType*>(kargs.p_sorted_weights);

            int eid     = blockIdx.x;
            int wave_id = threadIdx.x / opus::get_warp_size();
            int lane_id = threadIdx.x % opus::get_warp_size();
            int e_start = p_expert_cumsum_smem[eid];
            int e_end   = p_expert_cumsum_smem[eid + 1];
            if constexpr(Problem::SkipExpertsWithZeroTokens)
            {
                if(e_start == e_end)
                    return;
            }

            if constexpr(Problem::LocalExpertMasking)
            {
                int e_mask = p_local_expert_mask[eid];
                if(e_mask == 0)
                    return; // skip empty expert
            }

            opus::index_t mesh_stride = [&]() {
                if constexpr(Problem::LocalToken)
                {
                    return impl::moe_sorting_mp_mesh_stride(tokens);
                }
                else
                {
                    return kargs.mesh_stride;
                }
            }();

            // cumsum one by one
            constexpr opus::index_t index_pack = Problem::SubTokenTile;              // always packed
            using r_t                    = opus::vector_t<MeshType, index_pack>; // always use int32x4
            using d_t                    = opus::vector_t<opus::index_t, index_pack>;
            int loops                    = (mesh_stride / index_pack + kBlockSize - 1) / kBlockSize;

            int prev_cumsum = 0;

            for(int i = 0; i < loops; i++)
            {
                int i_token_pack = i * kBlockSize + threadIdx.x;
                r_t x_v          = 0;
                if(i_token_pack < (tokens + index_pack - 1) / index_pack)
                {
                    x_v = reinterpret_cast<r_t*>(p_expert_mesh + eid * mesh_stride)[i_token_pack];
                }

                r_t x_r;
#if 0
                if constexpr(index_pack != 1)
                {
                    // shuffle, we must have contiguout thread holds contiguout token
                    __syncthreads();
                    reinterpret_cast<r_t*>(s)[threadIdx.x] = x_v;
                    __syncthreads();

                    opus::static_for<index_pack>([&](auto j_) {
                        constexpr auto j = j_.value;
                        x_r[j]           = reinterpret_cast<MeshType*>(s)[threadIdx.x + j * kBlockSize];
                    });
                }
#else
                x_r = x_v;
#endif
                {
#if 0
#pragma unroll
                    for(int j = 0; j < index_pack / 2; j++)
                    {
                        int i_token = i * kBlockSize * index_pack + threadIdx.x + j * kBlockSize;
                        opus::index_t x   = x_d[j];
                        int i_topk  = x - 1;          // topk of this token
                        int i_show  = x != 0 ? 1 : 0; // has this token or not
                        int cumsum  = i_show;
                        impl::moe_sorting_wave_cumsum<int, opus::get_warp_size()>(cumsum);

                        __syncthreads();
                        if(lane_id == opus::get_warp_size() - 1)
                        {
                            s[4 + wave_id] = cumsum;
                        }
                        __syncthreads();

                        // reduce cross wave
                        opus::static_for<kBlockSize / opus::get_warp_size() - 1>([&](auto i_w) {
                            IndexType prev = s[4 + i_w];
                            prev           = wave_id > i_w ? prev : 0; // mask out
                            cumsum += prev;
                        });
                        cumsum += prev_cumsum; // add previous round cumsum
                        if(threadIdx.x == kBlockSize - 1)
                        {
                            s[0] = cumsum;
                        }
                        __syncthreads();

                        int position = cumsum - i_show;
                        prev_cumsum  = s[0]; // update the last cumsum

                        if(i_show)
                        {
#if OPUS_MOE_SORTING_MOCK_ID
                            p_sorted_token_ids[e_start + position] =
                                MOE_SORTING_MOCK_ID(i_token, i_topk);
#else
                            p_sorted_token_ids[e_start + position] = i_token;
#endif
                            p_sorted_weights[e_start + position] =
                                p_weights[i_token * kargs.topk_mdiv.divisor + i_topk];
                        }
                    }
#endif
                    {
                        d_t i_topk;
                        d_t i_show;
                        // = 0;
                        int cumsum_store = 0;

                        opus::static_for<index_pack>([&](auto j_) {
                            constexpr auto j = j_.value;
                            i_topk[j]        = static_cast<opus::index_t>(x_r[j] - 1);
                            i_show[j]        = static_cast<opus::index_t>(x_r[j] != 0 ? 1 : 0);
                            cumsum_store += i_show[j];
                        });
                        int cumsum = cumsum_store;
                        impl::moe_sorting_wave_cumsum<int, opus::get_warp_size()>(cumsum);

                        __syncthreads();
                        if(lane_id == opus::get_warp_size() - 1)
                        {
                            s[4 + wave_id] = cumsum;
                        }
                        __syncthreads();

                        // reduce cross wave
                        opus::static_for<kBlockSize / opus::get_warp_size() - 1>([&](auto i_w) {
                            IndexType prev = s[4 + i_w];
                            prev           = wave_id > i_w ? prev : 0; // mask out
                            cumsum += prev;
                        });
                        cumsum += prev_cumsum; // add previous round cumsum
                        if(threadIdx.x == kBlockSize - 1)
                        {
                            s[0] = cumsum;
                        }
                        __syncthreads();
                        prev_cumsum = s[0]; // update the last cumsum

                        int position = cumsum - cumsum_store;
                        opus::static_for<index_pack>([&](auto j_) {
                            constexpr auto j = j_.value;
                            // int i_token = i * kBlockSize * index_pack + threadIdx.x + j *
                            // kBlockSize;
                            int i_token =
                                i * kBlockSize * index_pack + threadIdx.x * index_pack + j;

                            if(i_show[j])
                            {
#if OPUS_MOE_SORTING_MOCK_ID
                                p_sorted_token_ids[e_start + position] =
                                    MOE_SORTING_MOCK_ID(i_token, i_topk[j]);
#else
                                p_sorted_token_ids[e_start + position] = i_token;
#endif
                                p_sorted_weights[e_start + position] =
                                    p_weights[i_token * kargs.topk_mdiv.divisor + i_topk[j]];
                            }
                            position += i_show[j];
                        });

#if 0
                        int i_token = i * kBlockSize * index_pack + threadIdx.x * 2 + j * kBlockSize * 2;
                        opus::index_t x   = x_d[j];
                        opus::index_t x0  = static_cast<opus::index_t>(x & 0xffff);
                        opus::index_t x1  = static_cast<opus::index_t>(x >> 16);
                        int i_topk_0  = x0 - 1;          // topk of this token
                        int i_show_0  = x0 != 0 ? 1 : 0; // has this token or not
                        int i_topk_1  = x1 - 1;          // topk of this token
                        int i_show_1  = x1 != 0 ? 1 : 0; // has this token or not
                        int cumsum  = i_show_0 + i_show_1;
                        impl::moe_sorting_wave_cumsum<int, opus::get_warp_size()>(cumsum);

                        __syncthreads();
                        if(lane_id == opus::get_warp_size() - 1)
                        {
                            s[4 + wave_id] = cumsum;
                        }
                        __syncthreads();

                        // reduce cross wave
                        opus::static_for<kBlockSize / opus::get_warp_size() - 1>([&](auto i_w) {
                            IndexType prev = s[4 + i_w];
                            prev           = wave_id > i_w ? prev : 0; // mask out
                            cumsum += prev;
                        });
                        cumsum += prev_cumsum; // add previous round cumsum
                        if(threadIdx.x == kBlockSize - 1)
                        {
                            s[0] = cumsum;
                        }
                        __syncthreads();

                        int position_0 = cumsum - i_show_0 - i_show_1;
                        prev_cumsum  = s[0]; // update the last cumsum

                        if(i_show_0)
                        {
#if OPUS_MOE_SORTING_MOCK_ID
                            p_sorted_token_ids[e_start + position_0] =
                                MOE_SORTING_MOCK_ID(i_token, i_topk_0);
#else
                            p_sorted_token_ids[e_start + position_0] = i_token;
#endif
                            p_sorted_weights[e_start + position_0] =
                                p_weights[i_token * kargs.topk_mdiv.divisor + i_topk_0];
                        }

                        int position_1 = cumsum - i_show_1;

                        if(i_show_1)
                        {
#if OPUS_MOE_SORTING_MOCK_ID
                            p_sorted_token_ids[e_start + position_1] =
                                MOE_SORTING_MOCK_ID(i_token + 1, i_topk_1);
#else
                            p_sorted_token_ids[e_start + position_1] = i_token + 1;
#endif
                            p_sorted_weights[e_start + position_1] =
                                p_weights[(i_token + 1) * kargs.topk_mdiv.divisor + i_topk_1];
                        }
#endif
                    }
                }
            }

            for(opus::index_t i = e_start + prev_cumsum + threadIdx.x; i < e_end; i += kBlockSize)
            {
#if OPUS_MOE_SORTING_MOCK_ID
                p_sorted_token_ids[i] = MOE_SORTING_MOCK_ID(tokens, kargs.topk_mdiv.divisor);
#else
                p_sorted_token_ids[i] = tokens;
#endif
                p_sorted_weights[i] = static_cast<WeightType>(0.0);
            }
        }
    }
};

#undef MOE_SORTING_MOCK_ID

} // namespace aiter


// --- API dispatch ---
// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Opus-based MOE sorting API dispatch layer.



#include <string>

struct moe_sorting_opus_trait
{
    std::string index_type;
    std::string weight_type;
    bool local_expert_masking;
    bool clear_workspace_inside_api;
    int dispatch_policy;
};

struct moe_sorting_opus_args : public aiter::MoeSortingHostArgs {};

inline int moe_sorting_opus_get_workspace_size(int tokens, int num_experts, int topk, int dispatch_policy)
{
    return aiter::moe_sorting_get_workspace_size(tokens, num_experts, topk, dispatch_policy);
}

// Forward declaration
inline float moe_sorting_opus_mp(moe_sorting_opus_trait t, moe_sorting_opus_args a, aiter::stream_config s);

// ---------------------------------------------------------------------------
// Dispatch macros

#define OPUS_MOE_SORTING_DISPATCH_(                                                                    \
    sub_token_tile_, sub_token_onshot_, local_expert_masking_, local_token_)                            \
    constexpr opus::index_t sub_token_tile = sub_token_tile_;                                        \
    constexpr bool sub_token_onshot           = sub_token_onshot_;                                      \
    constexpr bool local_expert_masking       = local_expert_masking_;                                  \
    constexpr bool local_token                = local_token_;                                           \
    using ms_problem                          = aiter::MoeSortingProblemEx<opus::index_t,                   \
                                                                             ms_weight_type,            \
                                                                             sub_token_tile,            \
                                                                             sub_token_onshot,          \
                                                                             local_expert_masking,      \
                                                                             local_token>;              \
    using kernel                              = aiter::MoeSortingKernel<ms_problem>;                  \
    auto kargs                                = kernel::MakeKargs(a);                                   \
    const dim3 grids                          = kernel::GridSize(a);                                    \
    const dim3 blocks                         = kernel::BlockSize(a);                                   \
    const auto lds_bytes                      = kernel::GetSmemSize(a);                                 \
    float ave_time                            = aiter::launch_kernel(                                 \
        s, aiter::make_kernel(kernel{}, grids, blocks, lds_bytes, kargs)); \
    return ave_time;

#define OPUS_MOE_SORTING_DISPATCH_SUB_TOKEN_(                                                  \
    row_, sub_token_onshot_, local_expert_masking_, local_token_)                         \
    if(row_ % 8 == 0)                                                                     \
    {                                                                                     \
        OPUS_MOE_SORTING_DISPATCH_(8, sub_token_onshot_, local_expert_masking_, local_token_); \
    }                                                                                     \
    else if(row_ % 4 == 0)                                                                \
    {                                                                                     \
        OPUS_MOE_SORTING_DISPATCH_(4, sub_token_onshot_, local_expert_masking_, local_token_); \
    }                                                                                     \
    else if(row_ % 2 == 0)                                                                \
    {                                                                                     \
        OPUS_MOE_SORTING_DISPATCH_(2, sub_token_onshot_, local_expert_masking_, local_token_); \
    }                                                                                     \
    else                                                                                  \
    {                                                                                     \
        OPUS_MOE_SORTING_DISPATCH_(1, sub_token_onshot_, local_expert_masking_, local_token_); \
    }

#define OPUS_MOE_SORTING_DISPATCH_DYNAMIC_TOKEN_(row_, sub_token_onshot_, local_expert_masking_)    \
    if(is_local_token)                                                                         \
    {                                                                                          \
        OPUS_MOE_SORTING_DISPATCH_SUB_TOKEN_(row_, sub_token_onshot_, local_expert_masking_, true)  \
    }                                                                                          \
    else                                                                                       \
    {                                                                                          \
        OPUS_MOE_SORTING_DISPATCH_SUB_TOKEN_(row_, sub_token_onshot_, local_expert_masking_, false) \
    }

#define OPUS_MOE_SORTING_DISPATCH_SUBTO_(row_, local_expert_masking_)                \
    if(is_sub_token_onshot)                                                     \
    {                                                                           \
        OPUS_MOE_SORTING_DISPATCH_DYNAMIC_TOKEN_(row_, true, local_expert_masking_)  \
    }                                                                           \
    else                                                                        \
    {                                                                           \
        OPUS_MOE_SORTING_DISPATCH_DYNAMIC_TOKEN_(row_, false, local_expert_masking_) \
    }

#define OPUS_MOE_SORTING_DISPATCH_EMASK_(row_)        \
    if(is_local_expert_masking)                  \
    {                                            \
        OPUS_MOE_SORTING_DISPATCH_SUBTO_(row_, true)  \
    }                                            \
    else                                         \
    {                                            \
        OPUS_MOE_SORTING_DISPATCH_SUBTO_(row_, false) \
    }

// ---------------------------------------------------------------------------
// Multi-phase dispatch macros

#define OPUS_MOE_SORTING_MP_0_V1(mesh_type_, unroll_num_, expert_masking_, local_token_)          \
    [&]() {                                                                                  \
        constexpr opus::index_t unroll_num = unroll_num_;                                 \
        constexpr bool expert_masking         = expert_masking_;                             \
        constexpr bool local_token            = local_token_;                                \
        using ms_problem                      = aiter::MoeSortingProblemMp<ms_index_t,     \
                                                                             ms_weight_type, \
                                                                             mesh_type_,     \
                                                                             unroll_num,     \
                                                                             expert_masking, \
                                                                             local_token>;   \
        using kernel      = aiter::MoeSortingMultiPhaseKernel_P0_v1<ms_problem>;           \
        auto kargs        = kernel::MakeKargs(a);                                            \
        const dim3 grids  = kernel::GridSize(a);                                             \
        const dim3 blocks = kernel::BlockSize(a);                                            \
        return aiter::make_kernel<kernel::kBlockSize>(kernel{}, grids, blocks, 0, kargs);  \
    }()

#define OPUS_MOE_SORTING_MP_0_V2(mesh_type_, unroll_num_, expert_masking_, local_token_)          \
    [&]() {                                                                                  \
        constexpr opus::index_t unroll_num = unroll_num_;                                 \
        constexpr bool expert_masking         = expert_masking_;                             \
        constexpr bool local_token            = local_token_;                                \
        using ms_problem                      = aiter::MoeSortingProblemMp<ms_index_t,     \
                                                                             ms_weight_type, \
                                                                             mesh_type_,     \
                                                                             unroll_num,     \
                                                                             expert_masking, \
                                                                             local_token>;   \
        using kernel      = aiter::MoeSortingMultiPhaseKernel_P0_v2<ms_problem>;           \
        auto kargs        = kernel::MakeKargs(a);                                            \
        const dim3 grids  = kernel::GridSize(a);                                             \
        const dim3 blocks = kernel::BlockSize(a);                                            \
        return aiter::make_kernel(kernel{}, grids, blocks, 0, kargs);                      \
    }()

#define OPUS_MOE_SORTING_MP_1(mesh_type_, unroll_num_, expert_masking_, local_token_)                    \
    [&]() {                                                                                         \
        constexpr opus::index_t unroll_num = unroll_num_;                                        \
        constexpr bool expert_masking         = expert_masking_;                                    \
        constexpr bool local_token            = local_token_;                                       \
        using ms_problem                      = aiter::MoeSortingProblemMp<ms_index_t,            \
                                                                             ms_weight_type,        \
                                                                             mesh_type_,            \
                                                                             unroll_num,            \
                                                                             expert_masking,        \
                                                                             local_token>;          \
        using kernel                          = aiter::MoeSortingMultiPhaseKernel_P1<ms_problem>; \
        auto kargs                            = kernel::MakeKargs(a);                               \
        const dim3 grids                      = kernel::GridSize(a);                                \
        const dim3 blocks                     = kernel::BlockSize(a);                               \
        return aiter::make_kernel(kernel{}, grids, blocks, 0, kargs);                             \
    }()

#define OPUS_MOE_SORTING_MP_23(mesh_type_, unroll_num_, expert_masking_, local_token_)                    \
    [&]() {                                                                                          \
        constexpr opus::index_t unroll_num = unroll_num_;                                         \
        constexpr bool expert_masking         = expert_masking_;                                     \
        constexpr bool local_token            = local_token_;                                        \
        using ms_problem                      = aiter::MoeSortingProblemMp<ms_index_t,             \
                                                                             ms_weight_type,         \
                                                                             mesh_type_,             \
                                                                             unroll_num,             \
                                                                             expert_masking,         \
                                                                             local_token>;           \
        using kernel                          = aiter::MoeSortingMultiPhaseKernel_P23<ms_problem>; \
        auto kargs                            = kernel::MakeKargs(a);                                \
        const dim3 grids                      = kernel::GridSize(a);                                 \
        const dim3 blocks                     = kernel::BlockSize(a);                                \
        const auto lds_size                   = kernel::GetSmemSize(a);                              \
        return aiter::make_kernel(kernel{}, grids, blocks, lds_size, kargs);                       \
    }()

#define OPUS_MOR_SORTING_MP_DISPATCH_SMALL_(mesh_type_, token_vec_0_, token_vec_1_, token_vec_23_)      \
    if(t.local_expert_masking)                                                                     \
    {                                                                                              \
        if(is_local_token)                                                                         \
        {                                                                                          \
            float ave_time =                                                                       \
                aiter::launch_kernel(s,                                                          \
                                       OPUS_MOE_SORTING_MP_0_V2(mesh_type_, token_vec_0_, true, true),  \
                                       OPUS_MOE_SORTING_MP_23(mesh_type_, token_vec_23_, true, true));  \
            return ave_time;                                                                       \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            float ave_time =                                                                       \
                aiter::launch_kernel(s,                                                          \
                                       OPUS_MOE_SORTING_MP_0_V2(mesh_type_, token_vec_0_, true, false), \
                                       OPUS_MOE_SORTING_MP_23(mesh_type_, token_vec_23_, true, false)); \
            return ave_time;                                                                       \
        }                                                                                          \
    }                                                                                              \
    else                                                                                           \
    {                                                                                              \
        if(is_local_token)                                                                         \
        {                                                                                          \
            float ave_time =                                                                       \
                aiter::launch_kernel(s,                                                          \
                                       OPUS_MOE_SORTING_MP_0_V2(mesh_type_, token_vec_0_, false, true), \
                                       OPUS_MOE_SORTING_MP_23(mesh_type_, token_vec_23_, false, true)); \
            return ave_time;                                                                       \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            float ave_time = aiter::launch_kernel(                                               \
                s,                                                                                 \
                OPUS_MOE_SORTING_MP_0_V2(mesh_type_, token_vec_0_, false, false),                       \
                OPUS_MOE_SORTING_MP_23(mesh_type_, token_vec_23_, false, false));                       \
            return ave_time;                                                                       \
        }                                                                                          \
    }

#define OPUS_MOR_SORTING_MP_DISPATCH_(mesh_type_, token_vec_0_, token_vec_1_, token_vec_23_)            \
    if(t.local_expert_masking)                                                                     \
    {                                                                                              \
        if(is_local_token)                                                                         \
        {                                                                                          \
            float ave_time =                                                                       \
                aiter::launch_kernel(s,                                                          \
                                       maybe_clear_workspace,                                      \
                                       OPUS_MOE_SORTING_MP_0_V1(mesh_type_, token_vec_0_, true, true),  \
                                       OPUS_MOE_SORTING_MP_1(mesh_type_, token_vec_1_, true, true),     \
                                       OPUS_MOE_SORTING_MP_23(mesh_type_, token_vec_23_, true, true));  \
            return ave_time;                                                                       \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            float ave_time =                                                                       \
                aiter::launch_kernel(s,                                                          \
                                       maybe_clear_workspace,                                      \
                                       OPUS_MOE_SORTING_MP_0_V1(mesh_type_, token_vec_0_, true, false), \
                                       OPUS_MOE_SORTING_MP_1(mesh_type_, token_vec_1_, true, false),    \
                                       OPUS_MOE_SORTING_MP_23(mesh_type_, token_vec_23_, true, false)); \
            return ave_time;                                                                       \
        }                                                                                          \
    }                                                                                              \
    else                                                                                           \
    {                                                                                              \
        if(is_local_token)                                                                         \
        {                                                                                          \
            float ave_time =                                                                       \
                aiter::launch_kernel(s,                                                          \
                                       maybe_clear_workspace,                                      \
                                       OPUS_MOE_SORTING_MP_0_V1(mesh_type_, token_vec_0_, false, true), \
                                       OPUS_MOE_SORTING_MP_1(mesh_type_, token_vec_1_, false, true),    \
                                       OPUS_MOE_SORTING_MP_23(mesh_type_, token_vec_23_, false, true)); \
            return ave_time;                                                                       \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            float ave_time = aiter::launch_kernel(                                               \
                s,                                                                                 \
                maybe_clear_workspace,                                                             \
                OPUS_MOE_SORTING_MP_0_V1(mesh_type_, token_vec_0_, false, false),                       \
                OPUS_MOE_SORTING_MP_1(mesh_type_, token_vec_1_, false, false),                          \
                OPUS_MOE_SORTING_MP_23(mesh_type_, token_vec_23_, false, false));                       \
            return ave_time;                                                                       \
        }                                                                                          \
    }

#define OPUS_MOR_SORTING_CLEAR_WS_DISPATCH_(is_local_token_, block_size_, occu_)                \
    [&]() {                                                                                \
        using problem_ =                                                                   \
            aiter::MoeSortingClearWorkspaceProblem<is_local_token_, block_size_, occu_>; \
        using kernel      = aiter::MoeSortingClearWorkspaceKernel<problem_>;             \
        auto kargs        = kernel::MakeKargs(a);                                          \
        const dim3 grids  = kernel::GridSize(a);                                           \
        const dim3 blocks = kernel::BlockSize(a);                                          \
        return aiter::make_kernel(kernel{}, grids, blocks, 0, kargs);                    \
    }()

// ---------------------------------------------------------------------------
// Main API functions

inline float moe_sorting_opus(moe_sorting_opus_trait t, moe_sorting_opus_args a, aiter::stream_config s)
{
    if(t.weight_type == "fp32" && t.index_type == "int32")
    {
        if(moe_sorting_opus_get_workspace_size(a.tokens, a.num_experts, a.topk, t.dispatch_policy) != 0)
        {
            return moe_sorting_opus_mp(t, a, s);
        }
        using ms_weight_type         = float;
        auto sub_token_              = aiter::moe_sorting_get_sub_token(a.tokens, a.num_experts);
        auto row_                    = sub_token_ / 8;
        bool is_sub_token_onshot     = a.tokens <= sub_token_;
        bool is_local_expert_masking = t.local_expert_masking;
        bool is_local_token          = a.p_local_tokens != nullptr;

        OPUS_MOE_SORTING_DISPATCH_EMASK_(row_);
    }
    return -1;
}

inline float moe_sorting_opus_mp(moe_sorting_opus_trait t, moe_sorting_opus_args a, aiter::stream_config s)
{
    bool is_local_token = a.p_local_tokens != nullptr;
    if(t.weight_type == "fp32" && t.index_type == "int32")
    {
        using ms_index_t     = opus::index_t;
        using ms_weight_type = float;

        auto maybe_clear_workspace = [=](const aiter::stream_config& s_) {
            if(t.clear_workspace_inside_api)
            {
                if(is_local_token)
                {
                    auto k = OPUS_MOR_SORTING_CLEAR_WS_DISPATCH_(true, 1024, 1);
                    k(s_);
                }
                else
                {
                    auto k = OPUS_MOR_SORTING_CLEAR_WS_DISPATCH_(false, 1024, 1);
                    k(s_);
                }
            }
        };

        if(a.tokens < 2048)
        {
            if(aiter::impl::moe_sorting_get_smem_size_p23(a.num_experts) >
               opus::get_smem_size())
            {
                printf("opus moe_sorting: do not support large expert %d\n", a.num_experts);
                return -1;
            }
            else
            {
                opus::index_t mesh_byte_size =
                    aiter::impl::moe_sorting_mesh_byte_size(a.tokens, a.num_experts, a.topk);
                if(mesh_byte_size == 1)
                {
                    if(a.tokens * a.topk % 4 == 0)
                    {
                        OPUS_MOR_SORTING_MP_DISPATCH_SMALL_(uint8_t, 4, 16, 16)
                    }
                    else
                    {
                        OPUS_MOR_SORTING_MP_DISPATCH_SMALL_(uint8_t, 1, 16, 16)
                    }
                }
                else if(mesh_byte_size == 2)
                {
                    printf("opus moe_sorting: do not support large topk %d\n", a.topk);
                    return -1;
                }
                else
                {
                    OPUS_MOR_SORTING_MP_DISPATCH_SMALL_(opus::index_t, 1, 1, 1)
                }
            }
        }
        else
        {
            if(aiter::impl::moe_sorting_get_smem_size_p23(a.num_experts) >
               opus::get_smem_size())
            {
                printf("opus moe_sorting: do not support large expert %d\n", a.num_experts);
                return -1;
            }
            else
            {
                opus::index_t mesh_byte_size =
                    aiter::impl::moe_sorting_mesh_byte_size(a.tokens, a.num_experts, a.topk);
                if(mesh_byte_size == 1)
                {
                    if(a.tokens * a.topk % 4 == 0)
                    {
                        OPUS_MOR_SORTING_MP_DISPATCH_(uint8_t, 4, 16, 16)
                    }
                    else
                    {
                        OPUS_MOR_SORTING_MP_DISPATCH_(uint8_t, 1, 16, 16)
                    }
                }
                else if(mesh_byte_size == 2)
                {
                    printf("opus moe_sorting: do not support large topk %d\n", a.topk);
                    return -1;
                }
                else
                {
                    OPUS_MOR_SORTING_MP_DISPATCH_(opus::index_t, 1, 1, 1)
                }
            }
        }
    }
    return -1;
}


#endif // MOE_SORTING_OPUS_IMPL
