#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include "asm_mla_decode_fwd.h"
#include "../utils.h"
#include <random>

// Helper macro for HIP error checking
#define HIP_CHECK(cmd)                                \
    do                                                \
    {                                                 \
        hipError_t error = (cmd);                     \
        if(error != hipSuccess)                       \
        {                                             \
            fprintf(stderr,                           \
                    "HIP error: '%s'(%d) at %s:%d\n", \
                    hipGetErrorString(error),         \
                    error,                            \
                    __FILE__,                         \
                    __LINE__);                        \
            exit(EXIT_FAILURE);                       \
        }                                             \
    } while(0)

class AsmMLADecodeFwdTest : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        // Set up common test dimensions
        batch_size    = 1;
        num_heads     = 16;
        num_kv_heads  = 1;
        seq_len       = 21;
        head_size     = 576;
        v_head_dim    = 512;
        num_kv_splits = 16;
        page_size     = 256;
        num_pages     = 32;

        // Calculate sizes for memory allocation
        size_t q_size  = batch_size * num_heads * head_size * sizeof(__hip_bfloat16);
        size_t kv_size = num_pages * page_size * num_kv_heads * head_size * sizeof(__hip_bfloat16);
        size_t output_size          = batch_size * num_heads * v_head_dim * sizeof(__hip_bfloat16);
        size_t kv_indptr_size       = (batch_size + 1) * sizeof(int32_t);
        size_t kv_page_indices_size = seq_len * sizeof(int32_t);
        size_t kv_last_page_lens_size = batch_size * sizeof(int32_t);
        size_t logits_size    = batch_size * num_kv_splits * num_heads * v_head_dim * sizeof(float);
        size_t attn_lse_size  = batch_size * num_kv_splits * num_heads * sizeof(float);
        size_t qo_indptr_size = (batch_size + 1) * sizeof(int32_t);
        // Allocate device memory
        HIP_CHECK(hipMalloc(&q_ptr, q_size));
        HIP_CHECK(hipMalloc(&kv_buffer_ptr, kv_size));
        HIP_CHECK(hipMalloc(&output_ptr, output_size));
        HIP_CHECK(hipMalloc(&qo_indptr_ptr, qo_indptr_size));
        HIP_CHECK(hipMalloc(&kv_indptr_ptr, kv_indptr_size));
        HIP_CHECK(hipMalloc(&kv_page_indices_ptr, kv_page_indices_size));
        HIP_CHECK(hipMalloc(&kv_last_page_lens_ptr, kv_last_page_lens_size));
        HIP_CHECK(hipMalloc(&logits_ptr, logits_size));
        HIP_CHECK(hipMalloc(&attn_lse_ptr, attn_lse_size));

        // Initialize host data
        std::vector<__hip_bfloat16> h_q(batch_size * num_heads * head_size);
        std::vector<__hip_bfloat16> h_kv(num_pages * page_size * num_kv_heads * head_size);
        std::vector<__hip_bfloat16> h_output(batch_size * num_heads * v_head_dim,
                                             __hip_bfloat16(-1.0f));
        std::vector<int32_t> h_qo_indptr{0, seq_len};
        std::vector<int32_t> h_kv_indptr{0, seq_len};
        std::vector<int32_t> h_kv_page_indices(seq_len);
        std::vector<int32_t> h_kv_last_page_lens(batch_size, 1);
        std::vector<float> h_logits(batch_size * num_kv_splits * num_heads * v_head_dim, 0.0f);
        std::vector<float> h_attn_lse(batch_size * num_kv_splits * num_heads, 0.0f);

        // Initialize random data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> normal_dist(0.0f, 1.0f);
        std::uniform_int_distribution<int32_t> uniform_dist(0, num_pages - 1);

        // Fill random data
        for(size_t i = 0; i < h_q.size(); ++i)
        {
            h_q[i] = __hip_bfloat16(normal_dist(gen));
        }
        for(size_t i = 0; i < h_kv.size(); ++i)
        {
            h_kv[i] = __hip_bfloat16(normal_dist(gen));
        }
        for(size_t i = 0; i < h_kv_page_indices.size(); ++i)
        {
            h_kv_page_indices[i] = uniform_dist(gen);
        }

        // Copy data to device
        HIP_CHECK(hipMemcpy(q_ptr, h_q.data(), q_size, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(kv_buffer_ptr, h_kv.data(), kv_size, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(output_ptr, h_output.data(), output_size, hipMemcpyHostToDevice));
        HIP_CHECK(
            hipMemcpy(kv_indptr_ptr, h_kv_indptr.data(), kv_indptr_size, hipMemcpyHostToDevice));
        HIP_CHECK(
            hipMemcpy(qo_indptr_ptr, h_qo_indptr.data(), qo_indptr_size, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(kv_page_indices_ptr,
                            h_kv_page_indices.data(),
                            kv_page_indices_size,
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(kv_last_page_lens_ptr,
                            h_kv_last_page_lens.data(),
                            kv_last_page_lens_size,
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(logits_ptr, h_logits.data(), logits_size, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(attn_lse_ptr, h_attn_lse.data(), attn_lse_size, hipMemcpyHostToDevice));

        // Calculate strides
        q_stride_0         = num_heads * head_size;
        kv_buffer_stride_0 = page_size * num_kv_heads * head_size;
        output_stride_0    = num_heads * v_head_dim;
        output_stride_1    = v_head_dim;
        attn_lse_stride_0  = num_kv_splits * num_heads;
        attn_lse_stride_1  = num_heads;
        attn_lse_stride_2  = 1;
    }

    void TearDown() override
    {
        // Free device memory
        HIP_CHECK(hipFree(q_ptr));
        HIP_CHECK(hipFree(kv_buffer_ptr));
        HIP_CHECK(hipFree(output_ptr));
        HIP_CHECK(hipFree(qo_indptr_ptr));
        HIP_CHECK(hipFree(kv_indptr_ptr));
        HIP_CHECK(hipFree(kv_page_indices_ptr));
        HIP_CHECK(hipFree(kv_last_page_lens_ptr));
        HIP_CHECK(hipFree(logits_ptr));
        HIP_CHECK(hipFree(attn_lse_ptr));
    }

    // Test dimensions
    int batch_size, num_heads, num_kv_heads, head_size;
    int v_head_dim, seq_len, num_kv_splits, page_size;
    int num_pages;

    // Strides
    int q_stride_0, kv_buffer_stride_0;
    int output_stride_0, output_stride_1;
    int attn_lse_stride_0, attn_lse_stride_1, attn_lse_stride_2;

    // Device pointers
    void *q_ptr, *kv_buffer_ptr, *output_ptr;
    void *qo_indptr_ptr, *kv_indptr_ptr, *kv_page_indices_ptr, *kv_last_page_lens_ptr;
    void *logits_ptr, *attn_lse_ptr;
};

TEST_F(AsmMLADecodeFwdTest, BasicFunctionality)
{
    float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Call the function with default parameters
    aiter::asm_mla_decode_fwd(std::nullopt, // folder
                              q_ptr,
                              kv_buffer_ptr,
                              qo_indptr_ptr,
                              kv_indptr_ptr,
                              kv_page_indices_ptr,
                              kv_last_page_lens_ptr,
                              1,
                              softmax_scale,
                              logits_ptr,
                              attn_lse_ptr,
                              output_ptr,
                              batch_size,
                              num_heads,
                              num_kv_heads,
                              q_stride_0,
                              kv_buffer_stride_0,
                              attn_lse_stride_0,
                              attn_lse_stride_1,
                              attn_lse_stride_2,
                              output_stride_0,
                              output_stride_1,
                              page_size,
                              "__hip_bfloat16", // q_dtype
                              "__hip_bfloat16", // kv_dtype
                              num_kv_splits,
                              v_head_dim,
                              stream);

    // Synchronize stream
    HIP_CHECK(hipStreamSynchronize(stream));

    // Verify output is not all -1
    std::vector<__hip_bfloat16> h_output(batch_size * num_heads * v_head_dim);
    HIP_CHECK(hipMemcpy(h_output.data(),
                        output_ptr,
                        h_output.size() * sizeof(__hip_bfloat16),
                        hipMemcpyDeviceToHost));

    bool all_negative_one = true;
    for(const auto& val : h_output)
    {
        if(float(val) != -1.0f)
        {
            all_negative_one = false;
            break;
        }
    }
    EXPECT_FALSE(all_negative_one);

    HIP_CHECK(hipStreamDestroy(stream));
}

// TEST_F(AsmMLADecodeFwdTest, CustomSoftmaxScale) {
//     float custom_scale = 0.1f;
//     hipStream_t stream;
//     HIP_CHECK(hipStreamCreate(&stream));

//     aiter::asm_mla_decode_fwd(
//         std::nullopt,
//         q_ptr,
//         kv_buffer_ptr,
//         kv_indptr_ptr,
//         kv_page_indices_ptr,
//         kv_last_page_lens_ptr,
//         custom_scale,
//         logits_ptr,
//         attn_lse_ptr,
//         output_ptr,
//         batch_size,
//         num_heads,
//         num_kv_heads,
//         q_stride_0,
//         kv_buffer_stride_0,
//         attn_lse_stride_0,
//         attn_lse_stride_1,
//         attn_lse_stride_2,
//         output_stride_0,
//         output_stride_1,
//         page_size,
//         "bf16",
//         "bf16",
//         num_kv_splits,
//         v_head_dim,
//         stream
//     );

//     HIP_CHECK(hipStreamSynchronize(stream));

//     // Verify output is not all -1
//     std::vector<__hip_bfloat16> h_output(batch_size * num_heads * v_head_dim);
//     HIP_CHECK(hipMemcpy(h_output.data(), output_ptr,
//                        h_output.size() * sizeof(__hip_bfloat16),
//                        hipMemcpyDeviceToHost));

//     bool all_negative_one = true;
//     for (const auto& val : h_output) {
//         if (float(val) != -1.0f) {
//             all_negative_one = false;
//             break;
//         }
//     }
//     EXPECT_FALSE(all_negative_one);

//     HIP_CHECK(hipStreamDestroy(stream));
// }
