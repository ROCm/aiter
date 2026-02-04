#include "rope/rope_common.h"

using namespace at;

template <typename T>
struct KernelElementType
{
    using type = T;
};

template <>
struct KernelElementType<c10::Half>
{
    using type = __half;
};

template <>
struct KernelElementType<c10::BFloat16>
{
    using type = hip_bfloat16;
};

void fused_qk_norm_mrope_3d_cache_pts_quant_shuffle(Tensor& qkv,
                                                    Tensor& qw,
                                                    Tensor& kw,
                                                    Tensor& cos_sin,
                                                    Tensor& positions,
                                                    int64_t num_tokens,
                                                    int64_t num_heads_q,
                                                    int64_t num_heads_k,
                                                    int64_t num_heads_v,
                                                    int64_t head_size,
                                                    bool is_neox_style,
                                                    std::vector<int64_t> mrope_section_,
                                                    bool is_interleaved,
                                                    double eps,
                                                    Tensor& q_out,
                                                    Tensor& k_cache,
                                                    Tensor& v_cache,
                                                    Tensor& slot_mapping,
                                                    Tensor& per_tensor_k_scale,
                                                    Tensor& per_tensor_v_scale,
                                                    std::optional<Tensor> k_out,
                                                    std::optional<Tensor> v_out,
                                                    bool return_kv,
                                                    bool use_shuffle_layout,
                                                    int64_t block_size,
                                                    int64_t x)
{
    TORCH_CHECK(mrope_section_.size() == 3);
    TORCH_CHECK(qkv.is_contiguous() && qw.is_contiguous() && kw.is_contiguous() &&
                cos_sin.is_contiguous());
    TORCH_CHECK(k_cache.is_contiguous() && v_cache.is_contiguous() && slot_mapping.is_contiguous());
    std::array<int64_t, 3> mrope_section;
    mrope_section[0] = mrope_section_[0];
    mrope_section[1] = mrope_section_[1];
    mrope_section[2] = mrope_section_[2];
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(qkv));
    auto stream         = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
    auto pos_strides    = positions.strides();
    auto kv_cache_dtype = k_cache.scalar_type();
    auto qkv_dtype      = qkv.scalar_type();
    TORCH_CHECK(pos_strides.size() == 2);
    float per_tensor_k_scale_ = per_tensor_k_scale.item<float>();
    float per_tensor_v_scale_ = per_tensor_v_scale.item<float>();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kBFloat16, kHalf, qkv_dtype, "fused_qk_norm_mrope_3d_cache_pts_quant_shuffle", [&] {
            using T = KernelElementType<scalar_t>::type;

            if(kv_cache_dtype == qkv_dtype)
            {
                T* k_out_ptr = (return_kv && k_out.has_value())
                                   ? (T*)k_out.value().data_ptr<scalar_t>()
                                   : nullptr;
                T* v_out_ptr = (return_kv && v_out.has_value())
                                   ? (T*)v_out.value().data_ptr<scalar_t>()
                                   : nullptr;
                mrope_utils::fused_mrope_rms_set_kv<T, 3, T>((T*)qkv.data_ptr<scalar_t>(),
                                                             (T*)qw.data_ptr<scalar_t>(),
                                                             (T*)kw.data_ptr<scalar_t>(),
                                                             (T*)cos_sin.data_ptr<scalar_t>(),
                                                             positions.data_ptr<int64_t>(),
                                                             pos_strides[0],
                                                             pos_strides[1],
                                                             num_tokens,
                                                             num_heads_q,
                                                             num_heads_k,
                                                             num_heads_v,
                                                             head_size,
                                                             is_neox_style,
                                                             eps,
                                                             mrope_section,
                                                             is_interleaved,
                                                             (T*)q_out.data_ptr<scalar_t>(),
                                                             (T*)k_cache.data_ptr<scalar_t>(),
                                                             (T*)v_cache.data_ptr<scalar_t>(),
                                                             slot_mapping.data_ptr<int64_t>(),
                                                             stream,
                                                             per_tensor_k_scale_,
                                                             per_tensor_v_scale_,
                                                             k_out_ptr,
                                                             v_out_ptr,
                                                             use_shuffle_layout,
                                                             block_size,
                                                             x);
            }
            else
            {
                // Check if kv_cache_dtype is fp8e4m3fnuz or fp8e4m3fn
                if(kv_cache_dtype == at::ScalarType::Float8_e4m3fnuz)
                {
                    mrope_utils::fp8e4m3fnuz* k_out_fp8_ptr =
                        (return_kv && k_out.has_value())
                            ? (mrope_utils::fp8e4m3fnuz*)k_out.value().data_ptr()
                            : nullptr;
                    mrope_utils::fp8e4m3fnuz* v_out_fp8_ptr =
                        (return_kv && v_out.has_value())
                            ? (mrope_utils::fp8e4m3fnuz*)v_out.value().data_ptr()
                            : nullptr;
                    mrope_utils::fused_mrope_rms_set_kv<T, 3, mrope_utils::fp8e4m3fnuz>(
                        (T*)qkv.data_ptr<scalar_t>(),
                        (T*)qw.data_ptr<scalar_t>(),
                        (T*)kw.data_ptr<scalar_t>(),
                        (T*)cos_sin.data_ptr<scalar_t>(),
                        positions.data_ptr<int64_t>(),
                        pos_strides[0],
                        pos_strides[1],
                        num_tokens,
                        num_heads_q,
                        num_heads_k,
                        num_heads_v,
                        head_size,
                        is_neox_style,
                        eps,
                        mrope_section,
                        is_interleaved,
                        (T*)q_out.data_ptr<scalar_t>(),
                        (mrope_utils::fp8e4m3fnuz*)k_cache.data_ptr(),
                        (mrope_utils::fp8e4m3fnuz*)v_cache.data_ptr(),
                        slot_mapping.data_ptr<int64_t>(),
                        stream,
                        per_tensor_k_scale_,
                        per_tensor_v_scale_,
                        k_out_fp8_ptr,
                        v_out_fp8_ptr,
                        use_shuffle_layout,
                        block_size,
                        x);
                }
                else if(kv_cache_dtype == at::ScalarType::Float8_e4m3fn)
                {
                    mrope_utils::fp8e4m3fn* k_out_fp8_ptr =
                        (return_kv && k_out.has_value())
                            ? (mrope_utils::fp8e4m3fn*)k_out.value().data_ptr()
                            : nullptr;
                    mrope_utils::fp8e4m3fn* v_out_fp8_ptr =
                        (return_kv && v_out.has_value())
                            ? (mrope_utils::fp8e4m3fn*)v_out.value().data_ptr()
                            : nullptr;
                    mrope_utils::fused_mrope_rms_set_kv<T, 3, mrope_utils::fp8e4m3fn>(
                        (T*)qkv.data_ptr<scalar_t>(),
                        (T*)qw.data_ptr<scalar_t>(),
                        (T*)kw.data_ptr<scalar_t>(),
                        (T*)cos_sin.data_ptr<scalar_t>(),
                        positions.data_ptr<int64_t>(),
                        pos_strides[0],
                        pos_strides[1],
                        num_tokens,
                        num_heads_q,
                        num_heads_k,
                        num_heads_v,
                        head_size,
                        is_neox_style,
                        eps,
                        mrope_section,
                        is_interleaved,
                        (T*)q_out.data_ptr<scalar_t>(),
                        (mrope_utils::fp8e4m3fn*)k_cache.data_ptr(),
                        (mrope_utils::fp8e4m3fn*)v_cache.data_ptr(),
                        slot_mapping.data_ptr<int64_t>(),
                        stream,
                        per_tensor_k_scale_,
                        per_tensor_v_scale_,
                        k_out_fp8_ptr,
                        v_out_fp8_ptr,
                        use_shuffle_layout,
                        block_size,
                        x);
                }
                else
                {
                    TORCH_CHECK(false, "Unsupported KV cache dtype: ", kv_cache_dtype);
                }
            }
        });
}
