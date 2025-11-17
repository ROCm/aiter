#include "pa_ragged.h"
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <random>
#include <vector>

using namespace aiter;

class PagedAttentionTest : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        // Common test parameters
        num_seqs           = 2;
        num_heads          = 8;
        num_kv_heads       = 1; // GQA ratio of 8
        head_size          = 128;
        block_size         = 16;
        num_blocks         = 4;
        max_num_partitions = 2;
        scale              = 1.0f / std::sqrt(head_size);

        // Calculate strides
        q_stride        = num_heads * head_size;
        kv_block_stride = num_kv_heads * head_size * block_size;
        kv_head_stride  = head_size * block_size;
        kv_seq_stride   = head_size;

        // Initialize HIP
        hipSetDevice(0);
        // ASSERT_EQ(err, hipSuccess) << "Failed to set HIP device";
    }

    void TearDown() override
    {
        // Clean up any allocated memory
        if(query_ptr)
            hipFree(query_ptr);
        if(key_cache_ptr)
            hipFree(key_cache_ptr);
        if(value_cache_ptr)
            hipFree(value_cache_ptr);
        if(out_ptr)
            hipFree(out_ptr);
        if(workspace_buffer_ptr)
            hipFree(workspace_buffer_ptr);
        if(kv_indptr_ptr)
            hipFree(kv_indptr_ptr);
        if(kv_page_indices_ptr)
            hipFree(kv_page_indices_ptr);
        if(kv_last_page_lens_ptr)
            hipFree(kv_last_page_lens_ptr);
        if(k_scale_ptr)
            hipFree(k_scale_ptr);
        if(v_scale_ptr)
            hipFree(v_scale_ptr);
    }

    // Helper function to allocate and initialize GPU memory
    template <typename T>
    T* allocateAndInitGPU(size_t size, const std::vector<T>& data)
    {
        T* ptr;
        hipMalloc(&ptr, size * sizeof(T));
        // ASSERT_EQ(err, hipSuccess) << "Failed to allocate GPU memory";

        if(!data.empty())
        {
            hipMemcpy(ptr, data.data(), size * sizeof(T), hipMemcpyHostToDevice);
            // ASSERT_EQ(err, hipSuccess) << "Failed to copy data to GPU";
        }
        return ptr;
    }

    // Test parameters
    int num_seqs;
    int num_heads;
    int num_kv_heads;
    int head_size;
    int block_size;
    int num_blocks;
    int max_num_partitions;
    float scale;

    // Strides
    int q_stride;
    int kv_block_stride;
    int kv_head_stride;
    int kv_seq_stride;

    // GPU pointers
    void* query_ptr            = nullptr;
    void* key_cache_ptr        = nullptr;
    void* value_cache_ptr      = nullptr;
    void* out_ptr              = nullptr;
    void* workspace_buffer_ptr = nullptr;
    int* kv_indptr_ptr         = nullptr;
    int* kv_page_indices_ptr   = nullptr;
    int* kv_last_page_lens_ptr = nullptr;
    float* k_scale_ptr         = nullptr;
    float* v_scale_ptr         = nullptr;
};

TEST_F(PagedAttentionTest, BasicTest)
{
    // Create random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Prepare input data
    std::vector<float> query_data(num_seqs * num_heads * head_size);
    std::vector<float> key_cache_data(num_blocks * num_kv_heads * head_size * block_size);
    std::vector<float> value_cache_data(num_blocks * num_kv_heads * head_size * block_size);
    std::vector<int> kv_indptr_data       = {0, 2, 4};
    std::vector<int> kv_page_indices_data = {0, 1, 2, 3};
    std::vector<float> k_scale_data       = {1.0f};
    std::vector<float> v_scale_data       = {1.0f};

    // Fill with random data
    for(auto& val : query_data)
        val = dist(gen);
    for(auto& val : key_cache_data)
        val = dist(gen);
    for(auto& val : value_cache_data)
        val = dist(gen);

    // Allocate GPU memory
    query_ptr       = allocateAndInitGPU(query_data.size(), query_data);
    key_cache_ptr   = allocateAndInitGPU(key_cache_data.size(), key_cache_data);
    value_cache_ptr = allocateAndInitGPU(value_cache_data.size(), value_cache_data);
    out_ptr         = allocateAndInitGPU(num_seqs * num_heads * head_size, std::vector<float>());
    workspace_buffer_ptr =
        allocateAndInitGPU(num_seqs * num_heads * max_num_partitions * 2, std::vector<float>());
    kv_indptr_ptr       = allocateAndInitGPU(kv_indptr_data.size(), kv_indptr_data);
    kv_page_indices_ptr = allocateAndInitGPU(kv_page_indices_data.size(), kv_page_indices_data);
    k_scale_ptr         = allocateAndInitGPU(k_scale_data.size(), k_scale_data);
    v_scale_ptr         = allocateAndInitGPU(v_scale_data.size(), v_scale_data);

    // Create HIP stream
    hipStream_t stream;
    hipStreamCreate(&stream);
    // ASSERT_EQ(err, hipSuccess) << "Failed to create HIP stream";

    // Call paged_attention_ragged
    paged_attention_ragged(std::nullopt, // folder
                           query_ptr,
                           key_cache_ptr,
                           value_cache_ptr,
                           workspace_buffer_ptr,
                           kv_indptr_ptr,
                           kv_page_indices_ptr,
                           kv_last_page_lens_ptr,
                           k_scale_ptr,
                           v_scale_ptr,
                           nullptr, // fp8_out_scale_ptr
                           out_ptr,
                           nullptr, // alibi_slopes_ptr
                           scale,
                           num_seqs,
                           num_kv_heads,
                           num_heads,
                           max_num_partitions,
                           head_size,
                           block_size,
                           0.0f, // logits_soft_cap
                           q_stride,
                           kv_block_stride,
                           kv_head_stride,
                           kv_seq_stride,
                           "_Float16", // dtype
                           "_Float16", // kv_dtype
                           "auto",     // kv_cache_dtype
                           "_Float16", // out_dtype
                           stream);

    // Wait for completion
    hipStreamSynchronize(stream);
    // ASSERT_EQ(err, hipSuccess) << "Failed to synchronize stream";

    // Clean up stream
    hipStreamDestroy(stream);
    // ASSERT_EQ(err, hipSuccess) << "Failed to destroy stream";
}
