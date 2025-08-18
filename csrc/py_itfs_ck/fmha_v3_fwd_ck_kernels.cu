#include "mha_common.h"
#include "py_itfs_common.h"
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>

#include "ck_tile/core.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/kernel/fmha_fwd_v3_kernel.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline.hpp"

#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "fmha_fwd.hpp"
#include "mask.hpp"

#define DEBUG_DTYPE_FP16 0
#define DEBUG_DTYPE_BF16 1
#define DEBUG_MASK_NONE 0
#define DEBUG_MASK_CAUSAL 1

#define DEBUG_SINGLE_INST 0
#define DEBUG_SINGLE_INST_DTYPE DEBUG_DTYPE_BF16
#define DEBUG_SINGLE_INST_MASK DEBUG_MASK_NONE

namespace aiter {
namespace torch_itfs {
namespace {
struct host_args
{
    ck_tile::index_t batch;
    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t hdim_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;

    float scale_s;

    ck_tile::index_t window_size_left;
    ck_tile::index_t window_size_right;
    ck_tile::index_t mask_type;

    const void* q_ptr;
    ck_tile::index_t stride_q;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t batch_stride_q;

    const void* k_ptr;
    ck_tile::index_t stride_k;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t batch_stride_k;

    const void* v_ptr;
    ck_tile::index_t stride_v;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t batch_stride_v;

    void* o_ptr;
    ck_tile::index_t stride_o;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t batch_stride_o;
};

//////////////////////////////////////////////////////////////////////////////////////
template <typename DataType, bool PadSeqlen, bool IsMasking>
struct get_kernel
{
    using fmha_dtype = DataType;
    //                                        M0   N0  K0   N1   K1
    using fmha_block_tile = ck_tile::sequence<256, 32, 128, 128, 32, 128>;

    using fmha_warp_gemm_shape = ck_tile::sequence<32, 32, 16>;

    using fmha_block_warps = ck_tile::sequence<8, 1, 1>;

    using fmha_shape = ck_tile::TileFmhaShape<fmha_block_tile,
                                              fmha_block_warps,
                                              fmha_warp_gemm_shape,
                                              fmha_block_warps,
                                              fmha_warp_gemm_shape,
                                              true // IsVLayoutRowMajor
                                              >;

    using fmha_traits = ck_tile::TileFmhaTraits<PadSeqlen, // kPadSeqLenQ
                                                PadSeqlen, // kPadSeqLenK
                                                false,     // kPadHeadDimQ
                                                false,     // kPadHeadDimV
                                                false,     // kHasLogitsSoftCap
                                                ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                                                false, // kHasBiasGrad
                                                false, // kStoreLSE
                                                false, // kHasDropout
                                                false, // kDoFp8StaticQuant
                                                -1,    // kBlockPerCu
                                                false  // kSkipMinSeqlenQ
                                                >;

    using fmha_variant =
        ck_tile::ComposedAttention<false * ck_tile::LOGITS_SOFT_CAP, // VARIANT_CODE
                                   CK_TILE_FMHA_FWD_FAST_EXP2        // UseExp2
                                   >; // placeholder type, we are not using this

    using fmha_mask = ck_tile::SimplifiedGenericAttentionMask<IsMasking>;

    using fmha_problem = ck_tile::BlockFmhaPipelineProblem<
        typename FmhaFwdTypeConfig<fmha_dtype>::QDataType,
        typename FmhaFwdTypeConfig<fmha_dtype>::KDataType,
        typename FmhaFwdTypeConfig<fmha_dtype>::VDataType,
        typename FmhaFwdTypeConfig<fmha_dtype>::SaccDataType,
        typename FmhaFwdTypeConfig<fmha_dtype>::SMPLComputeDataType,
        typename FmhaFwdTypeConfig<fmha_dtype>::BiasDataType,
        typename FmhaFwdTypeConfig<fmha_dtype>::RandValOutputDataType,
        typename FmhaFwdTypeConfig<fmha_dtype>::LSEDataType,
        typename FmhaFwdTypeConfig<fmha_dtype>::PDataType,
        typename FmhaFwdTypeConfig<fmha_dtype>::OaccDataType,
        typename FmhaFwdTypeConfig<fmha_dtype>::ODataType,
        fmha_shape,
        false, // kIsGroupMode
        fmha_variant,
        fmha_mask,
        fmha_traits>;

    using fmha_pipeline = ck_tile::BlockFmhaFwdV3Pipeline<fmha_problem>;

    using fmha_epilogue = ck_tile::Default2DEpilogue<
        ck_tile::Default2DEpilogueProblem<typename FmhaFwdTypeConfig<fmha_dtype>::OaccDataType,
                                          typename FmhaFwdTypeConfig<fmha_dtype>::ODataType,
                                          true, // kPadM
                                          true  // kPadM
                                          >>;

    using type = ck_tile::FmhaFwdV3Kernel<fmha_pipeline, fmha_epilogue>;
};

template <typename DataType, bool PadSeqlen, bool IsMasking>
using get_kernel_t = typename get_kernel<DataType, PadSeqlen, IsMasking>::type;

template <typename Kernel>
void launch(const host_args& args)
{
    auto kargs = Kernel::MakeKargs(args.q_ptr,
                                   args.k_ptr,
                                   args.v_ptr,
                                   nullptr, // lse_ptr
                                   args.o_ptr,
                                   args.seqlen_q,
                                   args.seqlen_k,
                                   args.hdim_q,
                                   args.hdim_v,
                                   args.nhead_q,
                                   args.nhead_q / args.nhead_k,
                                   args.scale_s,
                                   args.stride_q,
                                   args.stride_k,
                                   args.stride_v,
                                   args.stride_o,
                                   args.nhead_stride_q,
                                   args.nhead_stride_k,
                                   args.nhead_stride_v,
                                   0, // nhead_stride_lse
                                   args.nhead_stride_o,
                                   args.batch_stride_q,
                                   args.batch_stride_k,
                                   args.batch_stride_v,
                                   0, // batch_stride_lse
                                   args.batch_stride_o,
                                   args.window_size_left,
                                   args.window_size_right,
                                   args.mask_type);

    dim3 grids            = Kernel::GridSize(args.batch, args.nhead_q, args.seqlen_q, args.hdim_v);
    constexpr dim3 blocks = Kernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = Kernel::kBlockPerCu;

    auto stream = at::cuda::getCurrentHIPStream().stream();
    ck_tile::stream_config stream_config{stream};

    [[maybe_unused]] const float time = ck_tile::launch_kernel(
        stream_config,
        ck_tile::make_kernel<blocks.x, kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
}
} // namespace
//////////////////////////////////////////////////////////////////////////////////////

std::vector<at::Tensor> fmha_v3_fwd_ck(const at::Tensor& q, // [b, sq, hq, d]
                                       const at::Tensor& k, // [b, sk, hk, d]
                                       const at::Tensor& v, // [b, sk, hk, d_v]
                                       float softmax_scale,
                                       bool is_causal,
                                       int window_size_left,
                                       int window_size_right,
                                       bool return_softmax_lse)
{
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == at::ScalarType::Half || q_dtype == at::ScalarType::BFloat16,
                "FlashAttention only support fp16 and bf16 data type");

    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q);
    CHECK_DEVICE(k);
    CHECK_DEVICE(v);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const auto sizes = q.sizes();

    const int batch_size  = sizes[0];
    int seqlen_q          = sizes[1];
    int num_heads         = sizes[2];
    const int head_size_q = sizes[3];
    const int head_size_v = v.sizes()[3];
    const int seqlen_k    = k.size(1);
    const int num_heads_k = k.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_q <= 256, "CK only supports head dimension at most 256");
    TORCH_CHECK(head_size_v <= 256, "CK only supports head dimension at most 256");
    TORCH_CHECK(head_size_q % 8 == 0,
                "query, key, value, and out_ must have a head_size_q that is a multiple of 8");
    TORCH_CHECK(head_size_v % 8 == 0,
                "query, key, value, and out_ must have a head_size_q that is a multiple of 8");
    TORCH_CHECK(
        num_heads % num_heads_k == 0,
        "ck_tile::number of heads in key/value must divide ck_tile::number of heads in query");

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_q);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size_q);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_v);

    if(window_size_left >= seqlen_k)
    {
        window_size_left = -1;
    }
    if(window_size_right >= seqlen_k)
    {
        window_size_right = -1;
    }

    mask_info mask;
    if(is_causal)
    {
        // Causal is the special case where window_size_right == 0 and window_size_left < 0.
        window_size_right         = 0;
        std::string mask_identify = "b:" + std::to_string(window_size_left) + "," + "0";
        mask                      = mask_info::decode(mask_identify, seqlen_q, seqlen_k); // casual
    }
    else if(window_size_left == -1 && window_size_right == -1)
    {
        mask = mask_info::decode("0", seqlen_q, seqlen_k); // no mask
    }
    else
    {
        // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
        std::string mask_identify =
            "b:" + std::to_string(window_size_left) + "," + std::to_string(window_size_right);
        mask = mask_info::decode(mask_identify, seqlen_q, seqlen_k); // local
    }

    host_args args;

    args.batch    = batch_size;
    args.seqlen_q = seqlen_q;
    args.seqlen_k = seqlen_k;
    args.hdim_q   = head_size_q;
    args.hdim_v   = head_size_v;
    args.nhead_q  = num_heads;
    args.nhead_k  = num_heads_k;

    args.scale_s = softmax_scale;

    args.window_size_left  = mask.left;
    args.window_size_right = mask.right;
    args.mask_type         = static_cast<ck_tile::index_t>(mask.type);

    args.q_ptr          = q.data_ptr();
    args.batch_stride_q = q.stride(0);
    args.stride_q       = q.stride(1);
    args.nhead_stride_q = q.stride(2);

    args.k_ptr          = k.data_ptr();
    args.batch_stride_k = k.stride(0);
    args.stride_k       = k.stride(1);
    args.nhead_stride_k = k.stride(2);

    args.v_ptr          = v.data_ptr();
    args.batch_stride_v = v.stride(0);
    args.stride_v       = v.stride(1);
    args.nhead_stride_v = v.stride(2);

    auto opts = q.options();
    at::Tensor out =
        torch::empty({batch_size, seqlen_q, num_heads, head_size_v}, opts.dtype(q_dtype));

    args.o_ptr          = out.data_ptr();
    args.batch_stride_o = out.stride(0);
    args.stride_o       = out.stride(1);
    args.nhead_stride_o = out.stride(2);

    // TODO: compile fp16/bf16, masking=true/false kernels separately
    if(q_dtype == at::ScalarType::Half)
    {
        if(mask.type == mask_enum::no_mask)
        {
#if !DEBUG_SINGLE_INST || \
    (DEBUG_SINGLE_INST_DTYPE == DEBUG_DTYPE_FP16 && DEBUG_SINGLE_INST_MASK == DEBUG_MASK_NONE)
#if 0
            if(seqlen_q % 256 == 0 && seqlen_k % 32 == 0)
            {
                launch<get_kernel_t<FmhaFwdFp16, false, false>>(args);
            }
            else
            {
                launch<get_kernel_t<FmhaFwdFp16, true, false>>(args);
            }
#endif
            launch<get_kernel_t<FmhaFwdFp16, true, false>>(args);
#endif
        }
        else
        {
#if !DEBUG_SINGLE_INST || \
    (DEBUG_SINGLE_INST_DTYPE == DEBUG_DTYPE_FP16 && DEBUG_SINGLE_INST_MASK == DEBUG_MASK_CAUSAL)
#if 0
            if(seqlen_q % 256 == 0 && seqlen_k % 32 == 0)
            {
                launch<get_kernel_t<FmhaFwdFp16, false, true>>(args);
            }
            else
            {
                launch<get_kernel_t<FmhaFwdFp16, true, true>>(args);
            }
#endif
            launch<get_kernel_t<FmhaFwdFp16, true, true>>(args);
#endif
        }
    }
    else if(q_dtype == at::ScalarType::BFloat16)
    {
        if(mask.type == mask_enum::no_mask)
        {
#if !DEBUG_SINGLE_INST || \
    (DEBUG_SINGLE_INST_DTYPE == DEBUG_DTYPE_BF16 && DEBUG_SINGLE_INST_MASK == DEBUG_MASK_NONE)
#if 0
            if(seqlen_q % 256 == 0 && seqlen_k % 32 == 0)
            {
                launch<get_kernel_t<FmhaFwdBf16, false, false>>(args);
            }
            else
            {
                launch<get_kernel_t<FmhaFwdBf16, true, false>>(args);
            }
#endif
            launch<get_kernel_t<FmhaFwdBf16, true, false>>(args);
#endif
        }
        else
        {
#if !DEBUG_SINGLE_INST || \
    (DEBUG_SINGLE_INST_DTYPE == DEBUG_DTYPE_BF16 && DEBUG_SINGLE_INST_MASK == DEBUG_MASK_CAUSAL)
#if 0
            if(seqlen_q % 256 == 0 && seqlen_k % 32 == 0)
            {
                launch<get_kernel_t<FmhaFwdBf16, false, true>>(args);
            }
            else
            {
                launch<get_kernel_t<FmhaFwdBf16, true, true>>(args);
            }
#endif
            launch<get_kernel_t<FmhaFwdBf16, true, true>>(args);
#endif
        }
    }

    return {out};
}

} // namespace torch_itfs
} // namespace aiter
