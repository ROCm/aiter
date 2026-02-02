#pragma once

#include <cstdio>
#include <cmath>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "mhc_types.h"

namespace mhc {

template<int BLOCK_SIZE>
__global__ void float_to_bf16_kernel(__hip_bfloat16* __restrict__ out, const float* __restrict__ inp,
                                     int size) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < size) {
        out[idx] = (__hip_bfloat16)inp[idx];
    }
}

template<int BLOCK_SIZE>
__global__ void bf16_to_float_kernel(float* __restrict__ out, const __hip_bfloat16* __restrict__ inp,
                                     int size) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < size) {
        out[idx] = (float)inp[idx];
    }
}

inline void float_to_bf16(__hip_bfloat16* out, const float* inp, int size, hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float_to_bf16_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(out, inp, size);
}

inline void bf16_to_float(float* out, const __hip_bfloat16* inp, int size, hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bf16_to_float_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(out, inp, size);
}

__device__ __forceinline__ float fast_exp(float x) {
    return __expf(x);
}

__device__ __forceinline__ float fast_sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

__device__ __forceinline__ __hip_bfloat162 mhc_floats2bfloat162(float x, float y) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
    __hip_bfloat162 out;
    out.x = __float2bfloat16(x);
    out.y = __float2bfloat16(y);
    return out;
#else
    return mhc_floats2bfloat162(x, y);
#endif
}

template<int BLOCK_SIZE>
__global__ void fused_h_activations_kernel(
    float* __restrict__ H_pre_out, float* __restrict__ H_post_out, float* __restrict__ H_res_out,
    const float* __restrict__ H_proj_concat, const float* __restrict__ rms, float alpha_pre,
    float alpha_post, float alpha_res, const float* __restrict__ b_pre,
    const float* __restrict__ b_post, const float* __restrict__ b_res, int B, int n) {
    int n_sq = n * n;
    int total_pre = B * n;
    int total_post = B * n;
    int total_res = B * n_sq;
    int stride = n + n + n_sq;

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (idx < total_pre) {
        int b = idx / n;
        int j = idx % n;
        float r_inv = 1.0f / rms[b];
        float val = H_proj_concat[b * stride + j];
        val = alpha_pre * val * r_inv + b_pre[j];
        H_pre_out[idx] = fast_sigmoid(val);
    }

    int idx2 = idx;
    if (idx2 < total_post) {
        int b = idx2 / n;
        int j = idx2 % n;
        float r_inv = 1.0f / rms[b];
        float val = H_proj_concat[b * stride + n + j];
        val = alpha_post * val * r_inv + b_post[j];
        H_post_out[idx2] = 2.0f * fast_sigmoid(val);
    }

    int idx3 = idx;
    if (idx3 < total_res) {
        int b = idx3 / n_sq;
        int local = idx3 % n_sq;
        int i = local / n;
        int j = local % n;
        float r_inv = 1.0f / rms[b];
        float val = H_proj_concat[b * stride + n + n + local];
        val = alpha_res * val * r_inv + b_res[i * n + j];
        H_res_out[idx3] = fast_exp(val);
    }
}

inline void fused_h_activations(float* H_pre_out, float* H_post_out, float* H_res_out,
                                const float* H_proj_concat, const float* rms, float alpha_pre,
                                float alpha_post, float alpha_res, const float* b_pre,
                                const float* b_post, const float* b_res, int B, int n,
                                hipStream_t stream = nullptr) {
    constexpr int BLOCK = 256;
    int n_sq = n * n;
    int max_total = B * n_sq;
    int blocks = (max_total + BLOCK - 1) / BLOCK;

#if defined(MHC_ENABLE_PDL) && !defined(__HIP_PLATFORM_AMD__)
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t config = {};
    config.numAttrs = 1;
    config.attrs = attrs;
    config.blockDim = {BLOCK, 1, 1};
    config.gridDim = {(unsigned int)blocks, 1, 1};
    config.dynamicSmemBytes = 0;
    config.stream = stream;

    cudaLaunchKernelEx(&config, fused_h_activations_kernel<BLOCK>, H_pre_out, H_post_out, H_res_out,
                       H_proj_concat, rms, alpha_pre, alpha_post, alpha_res, b_pre, b_post, b_res,
                       B, n);
#else
    fused_h_activations_kernel<BLOCK><<<blocks, BLOCK, 0, stream>>>(
        H_pre_out, H_post_out, H_res_out, H_proj_concat, rms, alpha_pre, alpha_post, alpha_res,
        b_pre, b_post, b_res, B, n);
#endif
}

__global__ void flush_l2_kernel(float* buf, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        buf[idx] = buf[idx] + 1.0f;
    }
}

struct L2Flusher {
    static constexpr int L2_SIZE_BYTES = 50 * 1024 * 1024;
    static constexpr int FLUSH_SIZE = L2_SIZE_BYTES / sizeof(float) * 2;
    float* buf;

    L2Flusher() : buf(nullptr) {
        HIP_CALL(hipMalloc(&buf, FLUSH_SIZE * sizeof(float)));
        HIP_CALL(hipMemset(buf, 0, FLUSH_SIZE * sizeof(float)));
    }

    ~L2Flusher() {
        if (buf)
            HIP_CALL(hipFree(buf));
    }

    void flush() {
        int block_size = 256;
        int num_blocks = (FLUSH_SIZE + block_size - 1) / block_size;
        flush_l2_kernel<<<num_blocks, block_size>>>(buf, FLUSH_SIZE);
        HIP_CALL(hipDeviceSynchronize());
    }
};

inline float max_abs_diff(const float* a, const float* b, int n) {
    float max_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff)
            max_diff = diff;
    }
    return max_diff;
}

inline bool check_test(float max_diff, float tolerance, const char* test_name = nullptr) {
    if (test_name) {
        printf("%s: ", test_name);
    }
    printf("max diff = %e, ", max_diff);
    if (max_diff < tolerance) {
        printf("PASSED (tol: %e)\n", tolerance);
        return true;
    } else {
        printf("FAILED (tol: %e)\n", tolerance);
        return false;
    }
}

struct BenchTimer {
    hipEvent_t start, stop;

    BenchTimer() {
        HIP_CALL(hipEventCreate(&start));
        HIP_CALL(hipEventCreate(&stop));
    }

    ~BenchTimer() {
        HIP_CALL(hipEventDestroy(start));
        HIP_CALL(hipEventDestroy(stop));
    }

    void record_start() { HIP_CALL(hipEventRecord(start)); }

    void record_stop() { HIP_CALL(hipEventRecord(stop)); }

    float elapsed_ms() {
        HIP_CALL(hipEventSynchronize(stop));
        float ms = 0.0f;
        HIP_CALL(hipEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

enum ProfilerTag {
    TagSetup = 0,
    TagLoad,
    TagCompute,
    TagReduce,
    TagStore,
    TagSync,
    TagOther,
    TagCount
};

inline const char* profiler_tag_name(ProfilerTag tag) {
    switch (tag) {
    case TagSetup:
        return "Setup";
    case TagLoad:
        return "Load";
    case TagCompute:
        return "Compute";
    case TagReduce:
        return "Reduce";
    case TagStore:
        return "Store";
    case TagSync:
        return "Sync";
    case TagOther:
        return "Other";
    default:
        return "Unknown";
    }
}

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
__device__ __forceinline__ int64_t globaltimer() { return (int64_t)clock64(); }
__device__ __forceinline__ int get_smid() { return 0; }
#else
__device__ __forceinline__ int64_t globaltimer() {
    int64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)::"memory");
    return t;
}

__device__ __forceinline__ int get_smid() {
    int sm_id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));
    return sm_id;
}
#endif

struct DeviceProfiler {
    int64_t* data_ptr;
    int sm_id;
    int cnt;
    int max_entries;

    __device__ void init(int num_entries, int64_t* buffer, int block_id) {
        max_entries = num_entries;
        data_ptr = buffer + block_id * (1 + num_entries * 4);
        sm_id = get_smid();
        cnt = 0;
    }

    __device__ void start(ProfilerTag tag) {
        if (cnt >= max_entries)
            return;
        data_ptr[1 + cnt * 4 + 0] = sm_id;
        data_ptr[1 + cnt * 4 + 1] = tag;
        data_ptr[1 + cnt * 4 + 2] = globaltimer();
    }

    __device__ void stop() {
        if (cnt >= max_entries)
            return;
        data_ptr[1 + cnt * 4 + 3] = globaltimer() - data_ptr[1 + cnt * 4 + 2];
        cnt++;
    }

    __device__ void flush() { data_ptr[0] = cnt; }
};

struct ProfilerEntry {
    int sm_id;
    ProfilerTag tag;
    int64_t start_time;
    int64_t duration_ns;
};

struct HostProfiler {
    int64_t* d_buffer;
    int64_t* h_buffer;
    int num_blocks;
    int max_entries_per_block;
    size_t buffer_size;

    HostProfiler(int num_blocks_, int max_entries_per_block_)
        : num_blocks(num_blocks_), max_entries_per_block(max_entries_per_block_) {
        buffer_size = num_blocks * (1 + max_entries_per_block * 4) * sizeof(int64_t);
        HIP_CALL(hipMalloc(&d_buffer, buffer_size));
        HIP_CALL(hipMemset(d_buffer, 0, buffer_size));
        h_buffer = (int64_t*)malloc(buffer_size);
    }

    ~HostProfiler() {
        if (d_buffer)
            HIP_CALL(hipFree(d_buffer));
        if (h_buffer)
            free(h_buffer);
    }

    int64_t* device_ptr() { return d_buffer; }

    void download() { HIP_CALL(hipMemcpy(h_buffer, d_buffer, buffer_size, hipMemcpyDeviceToHost)); }

    void print_summary() {
        download();

        int64_t tag_totals[TagCount] = {0};
        int tag_counts[TagCount] = {0};
        int64_t min_start = INT64_MAX;
        int64_t max_end = 0;

        int entry_stride = 1 + max_entries_per_block * 4;

        for (int b = 0; b < num_blocks; b++) {
            int64_t* block_data = h_buffer + b * entry_stride;
            int num_entries = (int)block_data[0];

            for (int e = 0; e < num_entries; e++) {
                int tag = (int)block_data[1 + e * 4 + 1];
                int64_t start = block_data[1 + e * 4 + 2];
                int64_t duration = block_data[1 + e * 4 + 3];

                if (tag < TagCount) {
                    tag_totals[tag] += duration;
                    tag_counts[tag]++;
                }

                if (start < min_start)
                    min_start = start;
                if (start + duration > max_end)
                    max_end = start + duration;
            }
        }

        int64_t wall_time = max_end - min_start;
        printf("\nProfiler Summary (%d blocks, %d max entries/block)\n", num_blocks,
               max_entries_per_block);
        printf("==================================================\n");
        printf("Total wall time: %.2f us\n\n", wall_time / 1000.0f);
        printf("%-10s %10s %10s %10s\n", "Phase", "Total(us)", "Count", "Avg(us)");
        printf("------------------------------------------\n");

        for (int t = 0; t < TagCount; t++) {
            if (tag_counts[t] > 0) {
                float total_us = tag_totals[t] / 1000.0f;
                float avg_us = total_us / tag_counts[t];
                printf("%-10s %10.2f %10d %10.2f\n", profiler_tag_name((ProfilerTag)t), total_us,
                       tag_counts[t], avg_us);
            }
        }
    }

    void print_timeline(int max_blocks = 4) {
        download();

        int entry_stride = 1 + max_entries_per_block * 4;
        int blocks_to_print = (max_blocks < num_blocks) ? max_blocks : num_blocks;

        printf("\nTimeline (first %d blocks)\n", blocks_to_print);
        printf("==========================\n");

        for (int b = 0; b < blocks_to_print; b++) {
            int64_t* block_data = h_buffer + b * entry_stride;
            int num_entries = (int)block_data[0];

            printf("\nBlock %d (%d events):\n", b, num_entries);

            for (int e = 0; e < num_entries; e++) {
                int sm_id = (int)block_data[1 + e * 4 + 0];
                int tag = (int)block_data[1 + e * 4 + 1];
                int64_t duration = block_data[1 + e * 4 + 3];

                printf("  SM%02d: %-10s %8.2f us\n", sm_id, profiler_tag_name((ProfilerTag)tag),
                       duration / 1000.0f);
            }
        }
    }
};

} // namespace mhc
