#include <gtest/gtest.h>
#include <torch/torch.h>
#include "asm_mla_decode_fwd_torch.h"
#include "../utils.h"

class MLADecodeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up common test dimensions
        batch_size = 1;
        num_heads = 16;
        // head_dim = 128;
        seq_len = 21;
        qk_head_dim=576;
        v_head_dim=512;
        num_kv_splits=16;
        
        // Create input tensors
        q = torch::randn({batch_size, num_heads, qk_head_dim}, 
            torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        
        kv_buffer = torch::randn({65536, 1, 1, qk_head_dim}, 
            torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        
        output = torch::ones({batch_size, num_heads, v_head_dim}, 
            torch::dtype(torch::kBFloat16).device(torch::kCUDA)) * (-1);
        
        // Create index tensors
        kv_indptr = torch::tensor({0, seq_len}, 
            torch::dtype(torch::kInt32).device(torch::kCUDA));
        
        kv_indices = torch::randint(0, 65536, {seq_len+1}, 
            torch::dtype(torch::kInt32).device(torch::kCUDA));
        
        kv_last_page_lens = torch::ones({batch_size}, 
            torch::dtype(torch::kInt32).device(torch::kCUDA));

        logits = torch::zeros({batch_size, num_kv_splits, num_heads, v_head_dim}, 
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
        
        attn_lse = torch::zeros({batch_size, num_kv_splits, num_heads, 1}, 
            torch::dtype(torch::kFloat32).device(torch::kCUDA));
    }

    // Test dimensions
    int batch_size, num_heads, qk_head_dim, v_head_dim, seq_len, num_kv_splits;
    
    // Input tensors
    torch::Tensor q, kv_buffer, output, kv_indptr, kv_indices;
    torch::Tensor kv_last_page_lens, logits, attn_lse;
    
};

TEST_F(MLADecodeTest, BasicFunctionality) {
    // Test with default optional parameters
    aiter::mla_decode_fwd(
        q, kv_buffer, output, kv_indptr, kv_indices,
        kv_last_page_lens, std::nullopt, 0.0, num_kv_splits, logits, attn_lse
    );
    
    // Verify output has expected shape
    EXPECT_EQ(output.sizes(), std::vector<int64_t>({batch_size, num_heads, v_head_dim}));
}

TEST_F(MLADecodeTest, CustomScaleAndSplits) {
    float custom_scale = 0.1f;
    // int custom_splits = 4;
    
    aiter::mla_decode_fwd(
        q, kv_buffer, output, kv_indptr, kv_indices,
        kv_last_page_lens, custom_scale
    );
}

TEST_F(MLADecodeTest, InvalidLogitCap) {
    ASSERT_THROW(aiter::mla_decode_fwd(
        q, kv_buffer, output, kv_indptr, kv_indices,
        kv_last_page_lens, std::nullopt, 1.0
    ), std::invalid_argument);
}