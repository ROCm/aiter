#include "aiter_hip_common.h"
#include "aiter_tensor.h"
#include "aiter_stream.h"
#include "aiter_dispatch.h"
#include "rope/rope_common.h"
#include <array>
#include <optional>
#include <vector>

void fused_qk_norm_mrope_3d_cache_pts_quant_shuffle(const aiter_tensor_t& qkv,
                                                    const aiter_tensor_t& qw,
                                                    const aiter_tensor_t& kw,
                                                    const aiter_tensor_t& cos_sin,
                                                    const aiter_tensor_t& positions,
                                                    int64_t num_tokens,
                                                    int64_t num_heads_q,
                                                    int64_t num_heads_k,
                                                    int64_t num_heads_v,
                                                    int64_t head_size,
                                                    bool is_neox_style,
                                                    std::vector<int64_t> mrope_section_,
                                                    bool is_interleaved,
                                                    double eps,
                                                    const aiter_tensor_t& q_out,
                                                    const aiter_tensor_t& k_cache,
                                                    const aiter_tensor_t& v_cache,
                                                    const aiter_tensor_t& slot_mapping,
                                                    float per_tensor_k_scale,
                                                    float per_tensor_v_scale,
                                                    std::optional<aiter_tensor_t> k_out,
                                                    std::optional<aiter_tensor_t> v_out,
                                                    bool return_kv,
                                                    bool use_shuffle_layout,
                                                    int64_t block_size,
                                                    int64_t x,
                                                    int64_t rotary_dim)
{
    AITER_CHECK(mrope_section_.size() == 3);
    std::array<int64_t, 3> mrope_section;
    mrope_section[0] = mrope_section_[0];
    mrope_section[1] = mrope_section_[1];
    mrope_section[2] = mrope_section_[2];
    HipDeviceGuard device_guard(qkv.device_id);
    auto stream         = aiter::getCurrentHIPStream();
    AITER_CHECK(positions.dim() == 2, "positions must be 2D");
    int64_t position_stride_0 = positions.stride(0);
    int64_t position_stride_1 = positions.stride(1);
    auto kv_cache_dtype = k_cache.dtype();
    auto qkv_dtype      = qkv.dtype();
    AITER_DISPATCH_FLOATING(qkv_dtype, "fused_qk_norm_mrope_3d_cache_pts_quant_shuffle", [&] {
            using T = scalar_t;

            if(kv_cache_dtype == qkv_dtype)
            {
                T* k_out_ptr = (return_kv && k_out.has_value())
                                   ? reinterpret_cast<T*>(k_out.value().data_ptr())
                                   : nullptr;
                T* v_out_ptr = (return_kv && v_out.has_value())
                                   ? reinterpret_cast<T*>(v_out.value().data_ptr())
                                   : nullptr;
                mrope_utils::fused_mrope_rms_set_kv<T, 3, T>(reinterpret_cast<T*>(qkv.data_ptr()),
                                                             reinterpret_cast<T*>(qw.data_ptr()),
                                                             reinterpret_cast<T*>(kw.data_ptr()),
                                                             reinterpret_cast<T*>(cos_sin.data_ptr()),
                                                             reinterpret_cast<int64_t*>(positions.data_ptr()),
                                                             position_stride_0,
                                                             position_stride_1,
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
                                                             per_tensor_k_scale,
                                                             per_tensor_v_scale,
                                                             k_out_ptr,
                                                             v_out_ptr,
                                                             use_shuffle_layout,
                                                             block_size,
                                                             x,
                                                             rotary_dim);
            }
            else if(kv_cache_dtype == AITER_DTYPE_fp8)
            {
                    mrope_utils::fp8e4m3fnuz* k_out_fp8_ptr =
                        (return_kv && k_out.has_value())
                            ? reinterpret_cast<mrope_utils::fp8e4m3fnuz*>(k_out.value().data_ptr())
                            : nullptr;
                    mrope_utils::fp8e4m3fnuz* v_out_fp8_ptr =
                        (return_kv && v_out.has_value())
                            ? reinterpret_cast<mrope_utils::fp8e4m3fnuz*>(v_out.value().data_ptr())
                            : nullptr;
                    mrope_utils::fused_mrope_rms_set_kv<T, 3, mrope_utils::fp8e4m3fnuz>(
                        reinterpret_cast<T*>(qkv.data_ptr()),
                        reinterpret_cast<T*>(qw.data_ptr()),
                        reinterpret_cast<T*>(kw.data_ptr()),
                        reinterpret_cast<T*>(cos_sin.data_ptr()),
                        reinterpret_cast<int64_t*>(positions.data_ptr()),
                        position_stride_0,
                        position_stride_1,
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
                        per_tensor_k_scale,
                        per_tensor_v_scale,
                        k_out_fp8_ptr,
                        v_out_fp8_ptr,
                        use_shuffle_layout,
                        block_size,
                        x,
                        rotary_dim);
            }
            else
            {
                AITER_CHECK(false, "Unsupported KV cache dtype: ", AiterDtype_to_str(kv_cache_dtype));
            }
        });
}
