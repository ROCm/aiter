#include "utils.h"
#include <fmt/core.h>
#include "pa.h"

#define MD_NAME "pa_ragged"

#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

#define warpSize 64

void paged_attention_ragged(
    torch::Tensor& out, // [num_seqs, num_heads, head_size]
    torch::Tensor& workspace_buffer,
    torch::Tensor& query,       // [num_seqs, num_heads, head_size]
    torch::Tensor& key_cache,   // [num_blocks, num_heads, block_size, head_size] or
                                // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor& value_cache, // [num_blocks, num_heads, block_size, head_size] or
                                // [num_blocks, block_size, num_heads, head_size]
    double scale,
    torch::Tensor& kv_indptr,         // [num_seqs + 1]
    torch::Tensor& kv_page_indices,   // [max_num_blocks]
    std::optional<torch::Tensor>& kv_last_page_lens, // [num_seqs]
    int64_t block_size,
    int64_t max_num_partitions,
    const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype,
    const std::string& kv_cache_layout,
    float logits_soft_cap,
    torch::Tensor& k_scale,
    torch::Tensor& v_scale,
    const std::optional<torch::Tensor>& fp8_out_scale){
    const void* stream = reinterpret_cast<void*>(at::hip::getCurrentHIPStream().stream());
    std::string dtype;
    std::string kv_dtype;
    if(kv_cache_dtype == "auto")
    {
        if(query.dtype() == at::ScalarType::Half)
        {
            dtype = "_Float16";
            kv_dtype = "_Float16";
        }
        else if(query.dtype() == at::ScalarType::BFloat16)
        {
            dtype = "__hip_bfloat16";
            kv_dtype = "__hip_bfloat16";
        }
        else
        {
            TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
        }
    }
    else if(kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3")
    {
        if(query.dtype() == at::ScalarType::Half)
        {
            dtype = "_Float16";
            kv_dtype = "uint8_t";
        }
        else if(query.dtype() == at::ScalarType::BFloat16)
        {
            dtype = "__hip_bfloat16";
            kv_dtype = "uint8_t";
        }
        else
        {
            TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
        }
    }

    const int num_kv_heads = kv_cache_layout=="HND" ? key_cache.size(1) : key_cache.size(2);
    int num_seqs        = query.size(0);
    int num_heads       = query.size(1);
    int head_size       = query.size(2);
    int q_stride        = query.stride(0);
    int kv_block_stride = key_cache.stride(0);
    const int gqa_ratio = num_heads / num_kv_heads;
    int kv_head_stride  = kv_cache_layout == "HND" ? key_cache.stride(1) : key_cache.stride(2);
    int kv_seq_stride   = kv_cache_layout == "HND" ? key_cache.stride(2) : key_cache.stride(1);
    int npar_loops = DIVIDE_ROUND_UP(max_num_partitions, warpSize);
    init_root_dir();
    std::string folder = fmt::format("{md_name}_{gqa_ratio}_{head_size}_{npar_loops}_{dtype}_{kv_dtype}_{fp8_kv_dtype}_{out_dtype}_{block_size}_{alibi_enabled}",
                                    fmt::arg("md_name", MD_NAME),
                                    fmt::arg("gqa_ratio", gqa_ratio),
                                    fmt::arg("head_size", head_size),
                                    fmt::arg("npar_loops", npar_loops),
                                    fmt::arg("dtype", dtype),
                                    fmt::arg("kv_dtype", kv_dtype),
                                    fmt::arg("fp8_kv_dtype", kv_cache_dtype), 
                                    fmt::arg("out_dtype", dtype),
                                    fmt::arg("block_size", block_size),
                                    fmt::arg("alibi_enabled", alibi_slopes ? "true" : "false"));
    if(!std::filesystem::exists(get_root_dir()/"build"/folder/"lib.so")){
        std::string cmd = fmt::format(R"(python3 pa.py --gqa_ratio={gqa_ratio} \
                                    --head_size={head_size} \
                                    --npar_loops={npar_loops} \
                                    --dtype={dtype} \
                                    --kv_dtype={kv_dtype} \
                                    --fp8_kv_dtype={fp8_kv_dtype} \
                                    --out_dtype={out_dtype} \
                                    --block_size={block_size} \
                                    --alibi_enabled={alibi_enabled})",
                                    fmt::arg("gqa_ratio", gqa_ratio),
                                    fmt::arg("head_size", head_size),
                                    fmt::arg("npar_loops", npar_loops),
                                    fmt::arg("dtype", dtype),
                                    fmt::arg("kv_dtype", kv_dtype),
                                    fmt::arg("fp8_kv_dtype", kv_cache_dtype), 
                                    fmt::arg("out_dtype", dtype),
                                    fmt::arg("block_size", block_size),
                                    fmt::arg("alibi_enabled", alibi_slopes ? "true" : "false"));
        executeCmd(cmd);
    }

    void* query_ptr = query.data_ptr();
    void* key_cache_ptr = key_cache.data_ptr();
    void* value_cache_ptr = value_cache.data_ptr();
    void* workspace_buffer_ptr = workspace_buffer.data_ptr();
    int* kv_indptr_ptr         = kv_indptr.data_ptr<int>();
    int* kv_page_indices_ptr   = kv_page_indices.data_ptr<int>();
    int* kv_last_page_lens_ptr = kv_last_page_lens ? kv_last_page_lens.value().data_ptr<int>() : nullptr;
    const float* k_scale_ptr = k_scale.data_ptr<float>();
    const float* v_scale_ptr = v_scale.data_ptr<float>();
    const float* fp8_out_scale_ptr =
        fp8_out_scale ? fp8_out_scale.value().data_ptr<float>() : nullptr;
    void* out_ptr = out.data_ptr();
    const float* alibi_slopes_ptr = alibi_slopes ? alibi_slopes.value().data_ptr<float>() : nullptr;

    run_lib(folder, out_ptr, workspace_buffer_ptr, query_ptr, key_cache_ptr, value_cache_ptr, scale, num_seqs, num_kv_heads, num_heads, max_num_partitions, q_stride, kv_block_stride, kv_head_stride, kv_seq_stride, kv_indptr_ptr, kv_page_indices_ptr, kv_last_page_lens_ptr, alibi_slopes_ptr, logits_soft_cap, k_scale_ptr, v_scale_ptr, fp8_out_scale_ptr, stream);
}