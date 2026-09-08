#include "aiter_dispatch.h"
#include "aiter_stream.h"
#include "fused_qk_norm_mrope_cache_quant.h"
#include "rope/rope_common.h"

// The innermost dim must be dense: the kernel stores whole vectors per head, so a
// non-unit element stride cannot be expressed by the block/token/head strides below.
static inline bool has_unit_element_stride(const aiter_tensor_t& t)
{
    return t.numel() == 0 || t.stride(t.dim() - 1) == 1;
}

void fused_qk_norm_mrope_3d_cache_pts_quant_shuffle(aiter_tensor_t& qkv,
                                                    aiter_tensor_t& qw,
                                                    aiter_tensor_t& kw,
                                                    aiter_tensor_t& cos_sin,
                                                    aiter_tensor_t& positions,
                                                    int64_t num_tokens,
                                                    int64_t num_heads_q,
                                                    int64_t num_heads_k,
                                                    int64_t num_heads_v,
                                                    int64_t head_size,
                                                    bool is_neox_style,
                                                    std::vector<int64_t> mrope_section_,
                                                    bool is_interleaved,
                                                    double eps,
                                                    aiter_tensor_t& q_out,
                                                    aiter_tensor_t& k_cache,
                                                    aiter_tensor_t& v_cache,
                                                    aiter_tensor_t& slot_mapping,
                                                    aiter_tensor_t& per_tensor_k_scale,
                                                    aiter_tensor_t& per_tensor_v_scale,
                                                    std::optional<aiter_tensor_t> k_out,
                                                    std::optional<aiter_tensor_t> v_out,
                                                    bool return_kv,
                                                    bool use_shuffle_layout,
                                                    int64_t block_size,
                                                    int64_t x,
                                                    int64_t rotary_dim,
                                                    bool gemma_norm)
{
    AITER_CHECK(mrope_section_.size() == 3);
    AITER_CHECK(qkv.is_contiguous() && qw.is_contiguous() && kw.is_contiguous() &&
                cos_sin.is_contiguous());
    // k_cache/v_cache may be a strided view of a larger KV allocation -- e.g. blocks-first
    // (num_blocks, 2, block, heads, hs)[:, 0], or a packed (blocks, heads, block, 2*hs) buffer
    // transposed and split. Both are non-contiguous but well-formed. Every stride the kernel
    // needs is read off the tensors below, so only slot_mapping must be contiguous here.
    AITER_CHECK(slot_mapping.is_contiguous());
    AITER_CHECK(has_unit_element_stride(k_cache) && has_unit_element_stride(v_cache),
                "k_cache/v_cache must have unit element stride (innermost dim)");
    // Block/token/head strides taken from the tensors rather than assumed: a transposed or
    // interleaved view diverges from the contiguous (num_heads*hs, hs) layout on dims 1 and 2,
    // not just dim 0. Missing dims pass 0, which the kernel treats as "assume contiguous".
    int64_t k_block_stride_ = k_cache.stride(0);
    int64_t v_block_stride_ = v_cache.stride(0);
    int64_t k_token_stride_ = k_cache.dim() >= 2 ? k_cache.stride(1) : 0;
    int64_t v_token_stride_ = v_cache.dim() >= 2 ? v_cache.stride(1) : 0;
    int64_t k_head_stride_  = k_cache.dim() >= 3 ? k_cache.stride(2) : 0;
    int64_t v_head_stride_  = v_cache.dim() >= 3 ? v_cache.stride(2) : 0;
    std::array<int64_t, 3> mrope_section;
    mrope_section[0] = mrope_section_[0];
    mrope_section[1] = mrope_section_[1];
    mrope_section[2] = mrope_section_[2];
    HipDeviceGuard device_guard(qkv.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    auto kv_cache_dtype      = k_cache.dtype();
    auto qkv_dtype           = qkv.dtype();
    AITER_CHECK(positions.dim() == 2);
    int64_t positions_stride_0 = positions.stride(0);
    int64_t positions_stride_1 = positions.stride(1);
    float per_tensor_k_scale_  = *reinterpret_cast<float*>(per_tensor_k_scale.data_ptr());
    float per_tensor_v_scale_  = *reinterpret_cast<float*>(per_tensor_v_scale.data_ptr());
    AITER_DISPATCH_FLOATING16_TYPES_rmTorch(
        qkv_dtype, "fused_qk_norm_mrope_3d_cache_pts_quant_shuffle", [&] {
            using T = scalar_t;

            if(kv_cache_dtype == qkv_dtype)
            {
                T* k_out_ptr = (return_kv && k_out.has_value())
                                   ? reinterpret_cast<T*>(k_out.value().data_ptr())
                                   : nullptr;
                T* v_out_ptr = (return_kv && v_out.has_value())
                                   ? reinterpret_cast<T*>(v_out.value().data_ptr())
                                   : nullptr;
                mrope_utils::fused_mrope_rms_set_kv<T, 3, T>(
                    reinterpret_cast<T*>(qkv.data_ptr()),
                    reinterpret_cast<T*>(qw.data_ptr()),
                    reinterpret_cast<T*>(kw.data_ptr()),
                    reinterpret_cast<T*>(cos_sin.data_ptr()),
                    reinterpret_cast<int64_t*>(positions.data_ptr()),
                    positions_stride_0,
                    positions_stride_1,
                    num_tokens,
                    num_heads_q,
                    num_heads_k,
                    num_heads_v,
                    head_size,
                    is_neox_style,
                    eps,
                    mrope_section,
                    is_interleaved,
                    reinterpret_cast<T*>(q_out.data_ptr()),
                    reinterpret_cast<T*>(k_cache.data_ptr()),
                    reinterpret_cast<T*>(v_cache.data_ptr()),
                    reinterpret_cast<int64_t*>(slot_mapping.data_ptr()),
                    stream,
                    per_tensor_k_scale_,
                    per_tensor_v_scale_,
                    k_out_ptr,
                    v_out_ptr,
                    use_shuffle_layout,
                    block_size,
                    x,
                    rotary_dim,
                    k_block_stride_,
                    v_block_stride_,
                    gemma_norm,
                    k_token_stride_,
                    k_head_stride_,
                    v_token_stride_,
                    v_head_stride_);
            }
            else
            {
                if(kv_cache_dtype == AITER_DTYPE_fp8)
                {
                    if(is_fp8_ocp_arch())
                    {
                        mrope_utils::fp8e4m3fn* k_out_fp8_ptr =
                            (return_kv && k_out.has_value())
                                ? reinterpret_cast<mrope_utils::fp8e4m3fn*>(k_out.value().data_ptr())
                                : nullptr;
                        mrope_utils::fp8e4m3fn* v_out_fp8_ptr =
                            (return_kv && v_out.has_value())
                                ? reinterpret_cast<mrope_utils::fp8e4m3fn*>(v_out.value().data_ptr())
                                : nullptr;
                        mrope_utils::fused_mrope_rms_set_kv<T, 3, mrope_utils::fp8e4m3fn>(
                            reinterpret_cast<T*>(qkv.data_ptr()),
                            reinterpret_cast<T*>(qw.data_ptr()),
                            reinterpret_cast<T*>(kw.data_ptr()),
                            reinterpret_cast<T*>(cos_sin.data_ptr()),
                            reinterpret_cast<int64_t*>(positions.data_ptr()),
                            positions_stride_0,
                            positions_stride_1,
                            num_tokens,
                            num_heads_q,
                            num_heads_k,
                            num_heads_v,
                            head_size,
                            is_neox_style,
                            eps,
                            mrope_section,
                            is_interleaved,
                            reinterpret_cast<T*>(q_out.data_ptr()),
                            reinterpret_cast<mrope_utils::fp8e4m3fn*>(k_cache.data_ptr()),
                            reinterpret_cast<mrope_utils::fp8e4m3fn*>(v_cache.data_ptr()),
                            reinterpret_cast<int64_t*>(slot_mapping.data_ptr()),
                            stream,
                            per_tensor_k_scale_,
                            per_tensor_v_scale_,
                            k_out_fp8_ptr,
                            v_out_fp8_ptr,
                            use_shuffle_layout,
                            block_size,
                            x,
                            rotary_dim,
                            k_block_stride_,
                            v_block_stride_,
                            gemma_norm,
                            k_token_stride_,
                            k_head_stride_,
                            v_token_stride_,
                            v_head_stride_);
                    }
                    else
                    {
                        mrope_utils::fp8e4m3fnuz* k_out_fp8_ptr =
                            (return_kv && k_out.has_value())
                                ? reinterpret_cast<mrope_utils::fp8e4m3fnuz*>(
                                      k_out.value().data_ptr())
                                : nullptr;
                        mrope_utils::fp8e4m3fnuz* v_out_fp8_ptr =
                            (return_kv && v_out.has_value())
                                ? reinterpret_cast<mrope_utils::fp8e4m3fnuz*>(
                                      v_out.value().data_ptr())
                                : nullptr;
                        mrope_utils::fused_mrope_rms_set_kv<T, 3, mrope_utils::fp8e4m3fnuz>(
                            reinterpret_cast<T*>(qkv.data_ptr()),
                            reinterpret_cast<T*>(qw.data_ptr()),
                            reinterpret_cast<T*>(kw.data_ptr()),
                            reinterpret_cast<T*>(cos_sin.data_ptr()),
                            reinterpret_cast<int64_t*>(positions.data_ptr()),
                            positions_stride_0,
                            positions_stride_1,
                            num_tokens,
                            num_heads_q,
                            num_heads_k,
                            num_heads_v,
                            head_size,
                            is_neox_style,
                            eps,
                            mrope_section,
                            is_interleaved,
                            reinterpret_cast<T*>(q_out.data_ptr()),
                            reinterpret_cast<mrope_utils::fp8e4m3fnuz*>(k_cache.data_ptr()),
                            reinterpret_cast<mrope_utils::fp8e4m3fnuz*>(v_cache.data_ptr()),
                            reinterpret_cast<int64_t*>(slot_mapping.data_ptr()),
                            stream,
                            per_tensor_k_scale_,
                            per_tensor_v_scale_,
                            k_out_fp8_ptr,
                            v_out_fp8_ptr,
                            use_shuffle_layout,
                            block_size,
                            x,
                            rotary_dim,
                            k_block_stride_,
                            v_block_stride_,
                            gemma_norm,
                            k_token_stride_,
                            k_head_stride_,
                            v_token_stride_,
                            v_head_stride_);
                    }
                }
                else
                {
                    AITER_CHECK(false,
                                "Unsupported KV cache dtype: ",
                                AiterDtype_to_str(kv_cache_dtype));
                }
            }
        });
}
