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

// ---------------------------------------------------------------------------
// Numerical correctness against a CPU reference.
//
// The pre-existing test above only checks that the call does not crash: it has
// no assertions on the output, so a dispatcher that forwards its arguments in
// the wrong order still "passes". This test compares the kernel output with a
// straightforward CPU implementation, and exercises the paged gather itself —
// several pages per sequence and a partially filled last page — so the
// kv_indptr / kv_page_indices / kv_last_page_lens arguments have to arrive
// intact for it to pass.
// ---------------------------------------------------------------------------
class PagedAttentionNumericsTest : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        num_seqs           = 2;
        num_heads          = 8;
        num_kv_heads       = 1; // GQA ratio of 8
        head_size          = 128;
        block_size         = 16;
        max_num_partitions = 2;
        scale              = 1.0f / std::sqrt(static_cast<float>(head_size));

        // Uneven context lengths: 2 pages (last one full) and 3 pages (last one
        // partially filled), so last_page_lens is actually load-bearing.
        ctx_lens = {32, 40};

        q_stride        = num_heads * head_size;
        kv_block_stride = num_kv_heads * head_size * block_size;
        kv_head_stride  = head_size * block_size;
        kv_seq_stride   = head_size;

        hipSetDevice(0);
    }

    int num_seqs, num_heads, num_kv_heads, head_size, block_size, max_num_partitions;
    int q_stride, kv_block_stride, kv_head_stride, kv_seq_stride;
    float scale;
    std::vector<int> ctx_lens;
};

TEST_F(PagedAttentionNumericsTest, MatchesCpuReference)
{
    using half_t = _Float16;

    std::vector<int> kv_indptr{0}, kv_page_indices, kv_last_page_lens;
    int next_page = 0;
    for(int s = 0; s < num_seqs; ++s)
    {
        const int npages = (ctx_lens[s] + block_size - 1) / block_size;
        for(int p = 0; p < npages; ++p)
            kv_page_indices.push_back(next_page++);
        kv_indptr.push_back(static_cast<int>(kv_page_indices.size()));
        const int last = ctx_lens[s] % block_size;
        kv_last_page_lens.push_back(last == 0 ? block_size : last);
    }
    const int num_pages = next_page;

    const size_t q_elems  = static_cast<size_t>(num_seqs) * num_heads * head_size;
    const size_t kv_elems = static_cast<size_t>(num_pages) * kv_block_stride;

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<half_t> h_q(q_elems), h_k(kv_elems), h_v(kv_elems);
    for(auto& x : h_q)
        x = static_cast<half_t>(dist(rng));
    for(auto& x : h_k)
        x = static_cast<half_t>(dist(rng));
    for(auto& x : h_v)
        x = static_cast<half_t>(dist(rng));

    void *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_out = nullptr, *d_ws = nullptr;
    int *d_indptr = nullptr, *d_pages = nullptr, *d_lastlen = nullptr;
    float *d_kscale = nullptr, *d_vscale = nullptr;

    ASSERT_EQ(hipMalloc(&d_q, q_elems * sizeof(half_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_k, kv_elems * sizeof(half_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_v, kv_elems * sizeof(half_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_out, q_elems * sizeof(half_t)), hipSuccess);
    const size_t ws_bytes = static_cast<size_t>(num_seqs) * num_heads * max_num_partitions *
                            (head_size + 2) * sizeof(float);
    ASSERT_EQ(hipMalloc(&d_ws, ws_bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_indptr, kv_indptr.size() * sizeof(int)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_pages, kv_page_indices.size() * sizeof(int)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_lastlen, kv_last_page_lens.size() * sizeof(int)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_kscale, sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_vscale, sizeof(float)), hipSuccess);

    const float one = 1.0f;
    hipMemcpy(d_q, h_q.data(), q_elems * sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(d_k, h_k.data(), kv_elems * sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(d_v, h_v.data(), kv_elems * sizeof(half_t), hipMemcpyHostToDevice);
    hipMemcpy(d_indptr, kv_indptr.data(), kv_indptr.size() * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(
        d_pages, kv_page_indices.data(), kv_page_indices.size() * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_lastlen,
              kv_last_page_lens.data(),
              kv_last_page_lens.size() * sizeof(int),
              hipMemcpyHostToDevice);
    hipMemcpy(d_kscale, &one, sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_vscale, &one, sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_out, 0, q_elems * sizeof(half_t));
    hipMemset(d_ws, 0, ws_bytes);

    hipStream_t stream;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    paged_attention_ragged(std::nullopt,
                           d_q,
                           d_k,
                           d_v,
                           d_ws,
                           d_indptr,
                           d_pages,
                           d_lastlen,
                           d_kscale,
                           d_vscale,
                           nullptr, // fp8_out_scale_ptr
                           d_out,
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
                           "_Float16",
                           "_Float16",
                           "auto",
                           "_Float16",
                           stream);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    std::vector<half_t> h_out(q_elems);
    ASSERT_EQ(hipMemcpy(h_out.data(), d_out, q_elems * sizeof(half_t), hipMemcpyDeviceToHost),
              hipSuccess);

    // CPU reference: softmax(scale * q.k) @ v over each sequence's cached
    // tokens, gathered through the page table.
    double max_abs_err = 0.0, ref_absmax = 0.0;
    for(int s = 0; s < num_seqs; ++s)
    {
        const int ctx        = ctx_lens[s];
        const int page_begin = kv_indptr[s];
        for(int h = 0; h < num_heads; ++h)
        {
            const int kvh = h / (num_heads / num_kv_heads);
            std::vector<float> logits(ctx);
            float m = -INFINITY;
            for(int t = 0; t < ctx; ++t)
            {
                const int page     = kv_page_indices[page_begin + t / block_size];
                const int slot     = t % block_size;
                const size_t kbase = static_cast<size_t>(page) * kv_block_stride +
                                     static_cast<size_t>(kvh) * kv_head_stride +
                                     static_cast<size_t>(slot) * kv_seq_stride;
                float dot = 0.f;
                for(int d = 0; d < head_size; ++d)
                    dot += static_cast<float>(
                               h_q[(static_cast<size_t>(s) * num_heads + h) * head_size + d]) *
                           static_cast<float>(h_k[kbase + d]);
                logits[t] = dot * scale;
                m         = std::max(m, logits[t]);
            }
            float denom = 0.f;
            for(int t = 0; t < ctx; ++t)
            {
                logits[t] = std::exp(logits[t] - m);
                denom += logits[t];
            }
            for(int d = 0; d < head_size; ++d)
            {
                float acc = 0.f;
                for(int t = 0; t < ctx; ++t)
                {
                    const int page     = kv_page_indices[page_begin + t / block_size];
                    const int slot     = t % block_size;
                    const size_t vbase = static_cast<size_t>(page) * kv_block_stride +
                                         static_cast<size_t>(kvh) * kv_head_stride +
                                         static_cast<size_t>(slot) * kv_seq_stride;
                    acc += logits[t] * static_cast<float>(h_v[vbase + d]);
                }
                const float ref = acc / denom;
                const float got = static_cast<float>(
                    h_out[(static_cast<size_t>(s) * num_heads + h) * head_size + d]);
                max_abs_err = std::max(max_abs_err, std::fabs(static_cast<double>(ref) - got));
                ref_absmax  = std::max(ref_absmax, static_cast<double>(std::fabs(ref)));
            }
        }
    }

    // Normalize by the output magnitude: with random inputs the individual
    // attention outputs average near zero, so a per-element relative error is
    // not meaningful.
    ASSERT_GT(ref_absmax, 1e-3) << "reference output is degenerate";
    const double norm_err = max_abs_err / ref_absmax;
    EXPECT_LT(norm_err, 2e-2) << "max_abs_err=" << max_abs_err << " ref_absmax=" << ref_absmax;

    hipStreamDestroy(stream);
    hipFree(d_q);
    hipFree(d_k);
    hipFree(d_v);
    hipFree(d_out);
    hipFree(d_ws);
    hipFree(d_indptr);
    hipFree(d_pages);
    hipFree(d_lastlen);
    hipFree(d_kscale);
    hipFree(d_vscale);
}
