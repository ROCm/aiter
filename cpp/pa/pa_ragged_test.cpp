#include <gtest/gtest.h>
#include <torch/torch.h>
#include "pa_ragged_torch.h"


TEST(PagedAttentionTest, BasicTest) {
    // Test parameters
    const int num_seqs = 2;
    const int num_heads = 8;
    const int head_size = 128;
    const int block_size = 1;
    const int num_blocks = 4;
    const int max_num_partitions = 2;
    const double scale = 1.0 / std::sqrt(head_size);
    
    // Create test tensors
    auto options = torch::TensorOptions()
        .dtype(torch::kBFloat16)
        .device(torch::kCUDA);

    // Create query tensor [num_seqs, num_heads, head_size]
    auto query = torch::randn({num_seqs, num_heads, head_size}, options);
    
    // Create key/value cache tensors [num_blocks, num_heads, head_size, block_size]
    auto key_cache = torch::randn({num_blocks, 1, head_size, block_size}, options);
    auto value_cache = torch::randn({num_blocks, 1, head_size, block_size}, options);
    
    // Create output tensor
    auto out = torch::zeros({num_seqs, num_heads, head_size}, options);
    
    // Create workspace buffer
    auto workspace_buffer = torch::zeros({num_seqs * num_heads * max_num_partitions * 2}, 
                                       torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Create kv_indptr [num_seqs + 1]
    auto kv_indptr = torch::tensor({0, 2, 4}, 
                                 torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    
    // Create kv_page_indices [max_num_blocks]
    auto kv_page_indices = torch::tensor({0, 1, 2, 3}, 
                                       torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    
    // Create kv_last_page_lens [num_seqs]
    std::optional<torch::Tensor> kv_last_page_lens = std::nullopt;
    
    // Create scaling factors for FP8
    auto k_scale = torch::ones({1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto v_scale = torch::ones({1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Optional parameters
    std::optional<torch::Tensor> alibi_slopes = std::nullopt;
    std::optional<torch::Tensor> fp8_out_scale = std::nullopt;

    // Call the function
    paged_attention_ragged_torch(
        out,
        workspace_buffer,
        query,
        key_cache,
        value_cache,
        scale,
        kv_indptr,
        kv_page_indices,
        kv_last_page_lens,
        block_size,
        max_num_partitions,
        alibi_slopes,
        "auto",  // kv_cache_dtype
        "HND",   // kv_cache_layout
        0.0f,    // logits_soft_cap
        k_scale,
        v_scale,
        fp8_out_scale
    );

    // Basic checks
    EXPECT_EQ(out.sizes(), std::vector<int64_t>({num_seqs, num_heads, head_size}));
    EXPECT_FALSE(torch::any(torch::isnan(out)).item<bool>());
    EXPECT_FALSE(torch::any(torch::isinf(out)).item<bool>());
    
    // You might want to add more specific checks based on expected output values
    // This could include comparing against a reference implementation or 
    // checking specific properties of the attention output
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}