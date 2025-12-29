// Direct HIP kernel test for causal_conv1d_update
// Test configuration: batch=1, dim=2048, seqlen=1, width=4, state_len=3, silu=true, dtype=float32

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <iomanip>

// Kernel parameters structure (same as in the original kernel)
struct ConvParamsBaseUpdate {
    using index_t = uint32_t;
    
    int batch, dim, seqlen, width;
    bool silu_activation;
    
    index_t x_batch_stride;
    index_t x_c_stride;
    index_t x_l_stride;
    
    index_t weight_c_stride;
    index_t weight_width_stride;
    
    index_t out_batch_stride;
    index_t out_c_stride;
    index_t out_l_stride;
    
    int conv_state_len;
    index_t conv_state_batch_stride;
    index_t conv_state_c_stride;
    index_t conv_state_l_stride;
    
    void *__restrict__ x_ptr;
    void *__restrict__ weight_ptr;
    void *__restrict__ bias_ptr;
    void *__restrict__ out_ptr;
    void *__restrict__ conv_state_ptr;
    int32_t *__restrict__ cache_seqlens;
    int32_t *__restrict__ conv_state_indices_ptr;
    int pad_slot_id;
};

// Kernel traits
template<int kNThreads_, int kWidth_, typename input_t_, typename weight_t_>
struct Causal_conv1d_update_kernel_traits {
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kWidth = kWidth_;
    static constexpr int kNBytes = sizeof(input_t);
};

// The actual kernel (non-circular buffer version)
template<typename Ktraits, bool kIsCircularBuffer>
__global__ __launch_bounds__(Ktraits::kNThreads)
void causal_conv1d_update_kernel(ConvParamsBaseUpdate params) {
    constexpr int kWidth = Ktraits::kWidth;
    constexpr int kNThreads = Ktraits::kNThreads;
    using input_t = typename Ktraits::input_t;
    using weight_t = typename Ktraits::weight_t;

    const int tidx = threadIdx.x;
    const int batch_id = blockIdx.x;
    const int channel_id = blockIdx.y * kNThreads + tidx;
    
    if (channel_id >= params.dim) return;

    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride
        + channel_id * params.x_c_stride;

    const int conv_state_batch_coord = params.conv_state_indices_ptr == nullptr
        ? batch_id
        : params.conv_state_indices_ptr[batch_id];
    
    if (conv_state_batch_coord == params.pad_slot_id){
        return;
    }
    
    input_t *conv_state = reinterpret_cast<input_t *>(params.conv_state_ptr)
        + conv_state_batch_coord * params.conv_state_batch_stride
        + channel_id * params.conv_state_c_stride;
    
    weight_t *weight = reinterpret_cast<weight_t *>(params.weight_ptr) + channel_id * params.weight_c_stride;
    input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
        + channel_id * params.out_c_stride;
    float bias_val = params.bias_ptr == nullptr ? 0.f : float(reinterpret_cast<weight_t *>(params.bias_ptr)[channel_id]);

    int state_len = params.conv_state_len;
    int advance_len = params.seqlen;
    int cache_seqlen = kIsCircularBuffer ? params.cache_seqlens[batch_id] % state_len : 0;
    int update_idx = cache_seqlen - (kWidth - 1);
    update_idx = update_idx < 0 ? update_idx + state_len : update_idx;

    float weight_vals[kWidth] = {0};
    #pragma unroll
    for (int i = 0; i < kWidth; ++i) { weight_vals[i] = float(weight[i * params.weight_width_stride]); }

    float x_vals[kWidth] = {0};
    
    if constexpr (!kIsCircularBuffer) {
        #pragma unroll 2
        for (int i = 0; i < state_len - advance_len - (kWidth - 1); ++i) {
            conv_state[i * params.conv_state_l_stride] = conv_state[(i + advance_len) * params.conv_state_l_stride];
        }
        
        #pragma unroll
        for (int i = 0; i < kWidth - 1; ++i) {
            input_t state_val = conv_state[(state_len - (kWidth - 1) + i) * params.conv_state_l_stride];
            if (i < advance_len + (kWidth - 1) && state_len - advance_len - (kWidth - 1) + i >= 0) {
                conv_state[(state_len - advance_len - (kWidth - 1) + i) * params.conv_state_l_stride] = state_val;
            }
            x_vals[i] = float(state_val);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < kWidth - 1; ++i, update_idx = update_idx + 1 >= state_len ? update_idx + 1 - state_len : update_idx + 1) {
            input_t state_val = conv_state[update_idx * params.conv_state_l_stride];
            x_vals[i] = float(state_val);
        }
    }
    
    #pragma unroll 2
    for (int i = 0; i < params.seqlen; ++i) {
        input_t x_val = x[i * params.x_l_stride];
        
        if constexpr (!kIsCircularBuffer) {
            if (i < advance_len && state_len - advance_len + i >= 0) {
                conv_state[(state_len - advance_len + i) * params.conv_state_l_stride] = x_val;
            }
        } else {
            conv_state[update_idx * params.conv_state_l_stride] = x_val;
            ++update_idx;
            update_idx = update_idx >= state_len ? update_idx - state_len : update_idx;
        }
        
        x_vals[kWidth - 1] = float(x_val);
        
        float out_val = bias_val;
        #pragma unroll
        for (int j = 0; j < kWidth; ++j) { out_val += weight_vals[j] * x_vals[j]; }
        
        if (params.silu_activation) { out_val = out_val / (1 + expf(-out_val)); }
        
        out[i * params.out_l_stride] = input_t(out_val);
        
        #pragma unroll
        for (int i = 0; i < kWidth - 1; ++i) { x_vals[i] = x_vals[i + 1]; }
    }
}

// CPU reference implementation - matches GPU kernel logic exactly
void cpu_causal_conv1d_update(
    const float* x,           // [batch, dim, seqlen]
    float* conv_state,        // [batch, dim, state_len]
    const float* weight,      // [dim, width]
    const float* bias,        // [dim]
    float* out,               // [batch, dim, seqlen]
    int batch, int dim, int seqlen, int width, int state_len, bool silu) {
    
    for (int b = 0; b < batch; ++b) {
        for (int d = 0; d < dim; ++d) {
            // Get pointers for this batch and channel
            const float* x_ptr = x + b * dim * seqlen + d * seqlen;
            float* state_ptr = conv_state + b * dim * state_len + d * state_len;
            const float* weight_ptr = weight + d * width;
            float* out_ptr = out + b * dim * seqlen + d * seqlen;
            float bias_val = bias ? bias[d] : 0.0f;
            
            // Load weight values
            float weight_vals[4] = {0};
            for (int i = 0; i < width; ++i) {
                weight_vals[i] = weight_ptr[i];
            }
            
            // Sliding window buffer for input values (matches GPU kernel)
            float x_vals[4] = {0};
            int advance_len = seqlen;
            
            // Step 1: Shift old state data (non-circular buffer mode)
            // Loop from i=0 to i < state_len - advance_len - (width - 1)
            int shift_count = state_len - advance_len - (width - 1);
            for (int i = 0; i < shift_count; ++i) {
                state_ptr[i] = state_ptr[i + advance_len];
            }
            
            // Step 2: Load the most recent (width-1) historical states into x_vals
            // and copy them to earlier positions in state buffer
            for (int i = 0; i < width - 1; ++i) {
                float state_val = state_ptr[state_len - (width - 1) + i];
                // Copy condition from GPU kernel
                if (i < advance_len + (width - 1) && state_len - advance_len - (width - 1) + i >= 0) {
                    state_ptr[state_len - advance_len - (width - 1) + i] = state_val;
                }
                x_vals[i] = state_val;
            }
            
            // Step 3: Main convolution loop - Process each new input token
            for (int i = 0; i < seqlen; ++i) {
                // Read new input
                float x_val = x_ptr[i];
                
                // Update conv_state with new input (non-circular mode)
                if (i < advance_len && state_len - advance_len + i >= 0) {
                    state_ptr[state_len - advance_len + i] = x_val;
                }
                
                // Add new input to the sliding window
                x_vals[width - 1] = x_val;
                
                // Compute convolution output
                float out_val = bias_val;
                for (int j = 0; j < width; ++j) {
                    out_val += weight_vals[j] * x_vals[j];
                }
                
                // Apply SiLU activation
                if (silu) {
                    out_val = out_val / (1.0f + expf(-out_val));
                }
                
                // Write output
                out_ptr[i] = out_val;
                
                // Shift the sliding window left by 1 position
                for (int j = 0; j < width - 1; ++j) {
                    x_vals[j] = x_vals[j + 1];
                }
            }
        }
    }
}

// Helper function to check HIP errors
#define HIP_CHECK(cmd) \
    do { \
        hipError_t error = cmd; \
        if (error != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

int main() {
    // Test configuration
    const int batch = 1;
    const int dim = 2048;
    const int seqlen = 1;
    const int width = 4;
    const int state_len = 3;
    const bool silu = true;
    const int warmup_iters = 10;
    const int bench_iters = 100;
    
    std::cout << "=== Causal Conv1D Update Kernel Test ===" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Batch: " << batch << std::endl;
    std::cout << "  Dim: " << dim << std::endl;
    std::cout << "  Seqlen: " << seqlen << std::endl;
    std::cout << "  Width: " << width << std::endl;
    std::cout << "  State Length: " << state_len << std::endl;
    std::cout << "  SiLU: " << (silu ? "Yes" : "No") << std::endl;
    std::cout << "  Dtype: Float32" << std::endl;
    std::cout << "  Warmup iterations: " << warmup_iters << std::endl;
    std::cout << "  Benchmark iterations: " << bench_iters << std::endl;
    std::cout << std::endl;
    
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    // Allocate host memory
    std::vector<float> h_x(batch * dim * seqlen);
    std::vector<float> h_conv_state(batch * dim * state_len);
    std::vector<float> h_conv_state_cpu(batch * dim * state_len);
    std::vector<float> h_weight(dim * width);
    std::vector<float> h_bias(dim);
    std::vector<float> h_out(batch * dim * seqlen);
    std::vector<float> h_out_cpu(batch * dim * seqlen);
    
    // Initialize with random data
    for (auto& v : h_x) v = dis(gen);
    for (auto& v : h_conv_state) v = dis(gen);
    for (auto& v : h_weight) v = dis(gen);
    for (auto& v : h_bias) v = dis(gen);
    
    // Copy state for CPU reference
    h_conv_state_cpu = h_conv_state;
    
    // Allocate device memory
    float *d_x, *d_conv_state, *d_weight, *d_bias, *d_out;
    HIP_CHECK(hipMalloc(&d_x, batch * dim * seqlen * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_conv_state, batch * dim * state_len * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_weight, dim * width * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_bias, dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, batch * dim * seqlen * sizeof(float)));
    
    // Copy data to device
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), batch * dim * seqlen * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_conv_state, h_conv_state.data(), batch * dim * state_len * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weight, h_weight.data(), dim * width * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bias, h_bias.data(), dim * sizeof(float), hipMemcpyHostToDevice));
    
    // Setup kernel parameters
    ConvParamsBaseUpdate params;
    params.batch = batch;
    params.dim = dim;
    params.seqlen = seqlen;
    params.width = width;
    params.silu_activation = silu;
    params.conv_state_len = state_len;
    
    // Strides (contiguous layout)
    params.x_batch_stride = dim * seqlen;
    params.x_c_stride = seqlen;
    params.x_l_stride = 1;
    
    params.weight_c_stride = width;
    params.weight_width_stride = 1;
    
    params.out_batch_stride = dim * seqlen;
    params.out_c_stride = seqlen;
    params.out_l_stride = 1;
    
    params.conv_state_batch_stride = dim * state_len;
    params.conv_state_c_stride = state_len;
    params.conv_state_l_stride = 1;
    
    // Pointers
    params.x_ptr = d_x;
    params.weight_ptr = d_weight;
    params.bias_ptr = d_bias;
    params.out_ptr = d_out;
    params.conv_state_ptr = d_conv_state;
    params.cache_seqlens = nullptr;  // Non-circular buffer mode
    params.conv_state_indices_ptr = nullptr;
    params.pad_slot_id = -1;
    
    // Kernel configuration
    constexpr int kNThreads = 64;
    using Ktraits = Causal_conv1d_update_kernel_traits<kNThreads, 4, float, float>;
    dim3 grid(batch, (dim + kNThreads - 1) / kNThreads);
    dim3 block(kNThreads);
    
    std::cout << "Grid: (" << grid.x << ", " << grid.y << ", " << grid.z << ")" << std::endl;
    std::cout << "Block: (" << block.x << ", " << block.y << ", " << block.z << ")" << std::endl;
    std::cout << std::endl;
    
    // Warmup
    std::cout << "Warming up..." << std::endl;
    for (int i = 0; i < warmup_iters; ++i) {
        // Reset state before each warmup
        HIP_CHECK(hipMemcpy(d_conv_state, h_conv_state.data(), batch * dim * state_len * sizeof(float), hipMemcpyHostToDevice));
        hipLaunchKernelGGL(
            (causal_conv1d_update_kernel<Ktraits, false>),
            grid, block, 0, 0, params);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    // Reset state before benchmarking
    HIP_CHECK(hipMemcpy(d_conv_state, h_conv_state.data(), batch * dim * state_len * sizeof(float), hipMemcpyHostToDevice));
    
    // Benchmark
    std::cout << "Benchmarking..." << std::endl;
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    HIP_CHECK(hipEventRecord(start, 0));
    for (int i = 0; i < bench_iters; ++i) {
        hipLaunchKernelGGL(
            (causal_conv1d_update_kernel<Ktraits, false>),
            grid, block, 0, 0, params);
    }
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float elapsed_ms;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    
    // Reset state and run once for accuracy check
    HIP_CHECK(hipMemcpy(d_conv_state, h_conv_state.data(), batch * dim * state_len * sizeof(float), hipMemcpyHostToDevice));
    hipLaunchKernelGGL(
        (causal_conv1d_update_kernel<Ktraits, false>),
        grid, block, 0, 0, params);
    HIP_CHECK(hipDeviceSynchronize());
    
    // Copy result back
    HIP_CHECK(hipMemcpy(h_out.data(), d_out, batch * dim * seqlen * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_conv_state.data(), d_conv_state, batch * dim * state_len * sizeof(float), hipMemcpyDeviceToHost));
    
    // Run CPU reference
    std::cout << "Running CPU reference..." << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_causal_conv1d_update(
        h_x.data(), h_conv_state_cpu.data(), h_weight.data(), h_bias.data(),
        h_out_cpu.data(), batch, dim, seqlen, width, state_len, silu);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();
    
    // Compare results
    std::cout << std::endl;
    std::cout << "=== Accuracy Test ===" << std::endl;
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    int count = 0;
    
    for (int i = 0; i < batch * dim * seqlen; ++i) {
        float diff = std::abs(h_out[i] - h_out_cpu[i]);
        max_diff = std::max(max_diff, diff);
        sum_diff += diff;
        count++;
    }
    
    float avg_diff = sum_diff / count;
    std::cout << "Output comparison:" << std::endl;
    std::cout << "  Max absolute error: " << std::scientific << max_diff << std::endl;
    std::cout << "  Avg absolute error: " << std::scientific << avg_diff << std::endl;
    
    // Compare conv_state
    float max_state_diff = 0.0f;
    float sum_state_diff = 0.0f;
    for (int i = 0; i < batch * dim * state_len; ++i) {
        float diff = std::abs(h_conv_state[i] - h_conv_state_cpu[i]);
        max_state_diff = std::max(max_state_diff, diff);
        sum_state_diff += diff;
    }
    float avg_state_diff = sum_state_diff / (batch * dim * state_len);
    std::cout << "State comparison:" << std::endl;
    std::cout << "  Max absolute error: " << std::scientific << max_state_diff << std::endl;
    std::cout << "  Avg absolute error: " << std::scientific << avg_state_diff << std::endl;
    
    // Check if test passed
    bool accuracy_passed = (max_diff < 1e-5f) && (max_state_diff < 1e-5f);
    std::cout << std::endl;
    std::cout << "Accuracy test: " << (accuracy_passed ? "PASSED ✓" : "FAILED ✗") << std::endl;
    
    // Performance results
    std::cout << std::endl;
    std::cout << "=== Performance Test ===" << std::endl;
    float avg_time_us = (elapsed_ms * 1000.0f) / bench_iters;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "GPU kernel:" << std::endl;
    std::cout << "  Average time: " << avg_time_us << " us" << std::endl;
    std::cout << "  Throughput: " << (1000000.0f / avg_time_us) << " iterations/sec" << std::endl;
    std::cout << "CPU reference:" << std::endl;
    std::cout << "  Time: " << cpu_time << " us" << std::endl;
    std::cout << "Speedup: " << (float)cpu_time / avg_time_us << "x" << std::endl;
    
    // Cleanup
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_conv_state));
    HIP_CHECK(hipFree(d_weight));
    HIP_CHECK(hipFree(d_bias));
    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    
    std::cout << std::endl;
    std::cout << "Test completed successfully!" << std::endl;
    
    return 0;
}

