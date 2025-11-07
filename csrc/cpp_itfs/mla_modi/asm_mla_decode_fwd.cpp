#include <fmt/core.h>
#include "asm_mla_decode_fwd.h"
#include "../utils.h"

namespace aiter {

#define MD_NAME "asm_mla_decode_fwd"

void asm_mla_decode_fwd(std::optional<std::string> folder,
                        void* q,                 //   [num_seqs, num_heads, head_size]
                        void* kv_buffer,         //   [num_page, page_size, num_kv_heads, head_size]
                        void* qo_indptr,         //   [batch_size+1]
                        void* kv_indptr,         //   [batch_size+1]
                        void* kv_page_indices,   //   [num_page_used]
                        void* kv_last_page_lens, //   [batch_size]
                        int max_seqlen_q,
                        float softmax_scale,
                        // following are output
                        void* logits,   //[batch_size, num_kv_splits, num_heads, v_head_dim]
                        void* attn_lse, //[batch_size, num_kv_splits, num_heads,  1]
                        void* output,
                        int num_seqs,
                        int num_heads,
                        int num_kv_heads,
                        int q_stride_0,
                        int kv_buffer_stride_0,
                        int attn_lse_stride_0,
                        int attn_lse_stride_1,
                        int attn_lse_stride_2,
                        int output_stride_0,
                        int output_stride_1,
                        const int page_size,
                        const std::string q_dtype,
                        const std::string kv_dtype,
                        const int num_kv_splits,
                        const int v_head_dim,
                        const hipStream_t stream)
{
    int gqa_ratio = num_heads / num_kv_heads;
    if(num_kv_heads != 1)
    {
        throw std::invalid_argument("only support num_kv_heads==1 for now");
    }
    if(gqa_ratio == 16 && max_seqlen_q != 1)
    {
        throw std::invalid_argument("only support max_seqlen_q==1 when gqa_ratio==16");
    }
    std::list<std::string> args{std::to_string(gqa_ratio),
                                  std::to_string(page_size),
                                  q_dtype,
                                  kv_dtype,
                                  std::to_string(num_kv_splits),
                                  std::to_string(v_head_dim)};
    std::string func_name = get_default_func_name(MD_NAME, args);
    if(!folder)
    {
        folder = func_name;
    }
    if(not_built(folder.value()))
    {
        std::string cmd = fmt::format(
            R"(python3 -m csrc.cpp_itfs.mla.asm_mla_decode_fwd --gqa_ratio={gqa_ratio} \
                                    --page_size={page_size} \
                                    --q_dtype={q_dtype} \
                                    --kv_dtype={kv_dtype} \
                                    --num_kv_splits={num_kv_splits} \
                                    --v_head_dim={v_head_dim})",
            fmt::arg("gqa_ratio", gqa_ratio),
            fmt::arg("page_size", page_size),
            fmt::arg("q_dtype", q_dtype),
            fmt::arg("kv_dtype", kv_dtype),
            fmt::arg("num_kv_splits", num_kv_splits),
            fmt::arg("v_head_dim", v_head_dim));
        execute_cmd(cmd);
    }
    run_lib(func_name,
            folder.value(),
            q,
            kv_buffer,
            qo_indptr,
            kv_indptr,
            kv_page_indices,
            kv_last_page_lens,
            max_seqlen_q,
            softmax_scale,
            logits,
            attn_lse,
            output,
            num_seqs,
            num_heads,
            num_kv_heads,
            q_stride_0,
            kv_buffer_stride_0,
            attn_lse_stride_0,
            attn_lse_stride_1,
            attn_lse_stride_2,
            output_stride_0,
            output_stride_1,
            reinterpret_cast<const void*>(stream));
}
#undef MD_NAME
} // namespace aiter
