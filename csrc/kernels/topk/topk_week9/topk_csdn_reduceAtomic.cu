/**
 * Top-K 完整前 K 个输出 - 指令级并行优化版（基于 profiling 分析）
 * 输入: N=50000 FP32, K=2048
 * Phase1: radix 前缀得到 kth_value_bits → Phase2: last_filter 稳定输出前 K
 *
 * 与 calus-version 保持不同算法：
 * - 不做每轮 compaction / one-pass low-bit shortcut
 * - 仍然坚持"全量 radix 锁阈值 + 最后 full scan 输出"的两阶段方案
 *
 * 相对于 topk_hip_opt1.cu 的改动（基于 ATT profiling CSV 的指令依赖分析）：
 *
 * 核心优化：在 Barrier 1 和 Barrier 2 之间的"死区"为 gt/eq 线程提前发射 in[i] load。
 *
 * 原理：profiling 显示 Barrier 2 (s_barrier) stall = 15256 cycles，
 * 非 thread-0 线程在此期间完全空闲。而 in[i] load 延迟只有 ~3000 cycles。
 * 在 Barrier 1 之后立即为 gt/eq 线程（仅 ~4%）发射 in[i] load，
 * load 在 Barrier 2 等待期间完成，store 阶段直接用寄存器值，零 stall。
 *
 * 不为所有线程预取（之前的错误做法），只为 gt|eq 线程发射，不浪费带宽。
 */

 #include <hip/hip_runtime.h>

 #include <algorithm>
 #include <cstdint>
 #include <cstdlib>
 #include <iomanip>
 #include <iostream>
 #include <vector>
 
 #define N 50000
 #define K 2048
 #define BLOCK_SIZE 256
 #define VECTOR_SIZE 4
 #define WAVE_SIZE 64
 #define WAVES_PER_BLOCK (BLOCK_SIZE / WAVE_SIZE)
 
 #define NumPasses 8
 #define NumBuckets 16
 #define BitsPerPass 4
 
 using namespace std;
 using u32x4 = __attribute__((__ext_vector_type__(4))) uint32_t;
 
 static inline int read_positive_env_or_default(const char* name, int default_value) {
   const char* value = std::getenv(name);
   if (value == nullptr || *value == '\0') return default_value;
 
   char* end = nullptr;
   long parsed = std::strtol(value, &end, 10);
   if (end == value || *end != '\0' || parsed <= 0) return default_value;
   return static_cast<int>(parsed);
 }
 
 struct Counter {
   uint32_t kth_value_bits;
   int num_of_kth_needed;
   unsigned int out_cnt;
   unsigned int out_back_cnt;
 };
 
 __device__ __forceinline__ uint32_t twiddle_float(float key) {
   uint32_t x = __float_as_uint(key);
   uint32_t mask = (x & 0x80000000u) ? 0xffffffffu : 0x80000000u;
   return (key == key) ? (x ^ mask) : 0xffffffffu;
 }
 
 __device__ __forceinline__ int get_start_bit(int pass) {
   return 32 - (pass + 1) * BitsPerPass;
 }
 
 __device__ __forceinline__ int calc_bucket(uint32_t bits, int pass) {
   return (bits >> get_start_bit(pass)) & (NumBuckets - 1);
 }
 
 __device__ __forceinline__ unsigned int lane_id_in_wave() {
   return threadIdx.x & (WAVE_SIZE - 1);
 }
 
 __device__ __forceinline__ unsigned int wave_id_in_block() {
   return threadIdx.x / WAVE_SIZE;
 }
 
 __device__ __forceinline__ unsigned long long lane_mask_lt(unsigned int lane) {
   return lane == 0 ? 0ull : ((1ull << lane) - 1ull);
 }
 
 __global__ void preprocess_bits_kernel(const float* __restrict__ data,
                                        uint32_t* __restrict__ bits,
                                        int n) {
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   int stride = gridDim.x * blockDim.x;
 
   for (int i = tid; i < n; i += stride * 2) {
     int i1 = i + stride;
     bits[i] = twiddle_float(data[i]);
     if (i1 < n) bits[i1] = twiddle_float(data[i1]);
   }
 }
 
 __global__ void reset_kernel(Counter* counter, unsigned int* global_hist) {
   if (threadIdx.x == 0) {
     counter->kth_value_bits = 0u;
     counter->num_of_kth_needed = 0;
     counter->out_cnt = 0u;
     counter->out_back_cnt = 0u;
   }
   if (threadIdx.x < NumBuckets) {
     global_hist[threadIdx.x] = 0u;
   }
 }
 
 __global__ __launch_bounds__(BLOCK_SIZE) void radix_hist_kernel(
     const uint32_t* __restrict__ bits_data,
     int n,
     int pass,
     const Counter* __restrict__ counter,
     unsigned int* __restrict__ global_hist) {
   __shared__ unsigned int hist[NumBuckets];
 
   const int prev_start_bit = (pass == 0) ? 32 : get_start_bit(pass - 1);
   const uint32_t prev_mask = (pass == 0) ? 0u : (0xffffffffu << prev_start_bit);
   const uint32_t kth_prefix = counter->kth_value_bits & prev_mask;
 
   for (int b = threadIdx.x; b < NumBuckets; b += blockDim.x) hist[b] = 0u;
   __syncthreads();
 
   const int num_vecs = n / VECTOR_SIZE;
   const u32x4* in_vec = reinterpret_cast<const u32x4*>(bits_data);
   const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
   const int global_stride = gridDim.x * blockDim.x;
 
   for (int vec_idx = global_tid; vec_idx < num_vecs; vec_idx += global_stride) {
     const u32x4 v = in_vec[vec_idx];
 #pragma unroll
     for (int j = 0; j < VECTOR_SIZE; ++j) {
       const uint32_t bits = v[j];
       if (pass == 0 || ((bits & prev_mask) == kth_prefix)) {
         atomicAdd(hist + calc_bucket(bits, pass), 1u);
       }
     }
   }
 
   for (int i = num_vecs * VECTOR_SIZE + global_tid; i < n; i += global_stride) {
     const uint32_t bits = bits_data[i];
     if (pass == 0 || ((bits & prev_mask) == kth_prefix)) {
       atomicAdd(hist + calc_bucket(bits, pass), 1u);
     }
   }
   __syncthreads();
 
   for (int b = threadIdx.x; b < NumBuckets; b += blockDim.x) {
     const unsigned int block_sum = hist[b];
     if (block_sum != 0u) atomicAdd(global_hist + b, block_sum);
   }
 }
 
 __global__ void radix_scan_choose_kernel(
     int pass,
     int k,
     Counter* __restrict__ counter,
     unsigned int* __restrict__ global_hist) {
   __shared__ unsigned int hist[NumBuckets];
   const int start_bit = get_start_bit(pass);
 
   for (int i = threadIdx.x; i < NumBuckets; i += blockDim.x) hist[i] = global_hist[i];
   __syncthreads();
 
   if (threadIdx.x == 0) {
     for (int i = 1; i < NumBuckets; ++i) hist[i] += hist[i - 1];
 
     uint32_t kth_bits = (pass == 0) ? 0u : counter->kth_value_bits;
     int k_remaining = (pass == 0) ? k : counter->num_of_kth_needed;
     const unsigned int total = hist[NumBuckets - 1];
 
     for (int b = 0; b < NumBuckets; ++b) {
       if (static_cast<int>(total) - k_remaining < static_cast<int>(hist[b])) {
         kth_bits |= static_cast<uint32_t>(b) << start_bit;
         k_remaining -= static_cast<int>(total - hist[b]);
         break;
       }
     }
 
     counter->kth_value_bits = kth_bits;
     counter->num_of_kth_needed = k_remaining;
     if (pass == NumPasses - 1) {
       counter->out_cnt = 0u;
       counter->out_back_cnt = 0u;
     }
   }
   __syncthreads();
 
   if (pass < NumPasses - 1) {
     for (int i = threadIdx.x; i < NumBuckets; i += blockDim.x) global_hist[i] = 0u;
   }
 }
 
 __global__ __launch_bounds__(BLOCK_SIZE) void last_filter_kernel(
     const float* __restrict__ in,
     const uint32_t* __restrict__ bits_data,
     float* __restrict__ out,
     unsigned int* __restrict__ out_idx,
     int n,
     int k,
     Counter* __restrict__ counter) {
   const uint32_t kth_value_bits = counter->kth_value_bits;
   const unsigned int num_of_kth_needed = static_cast<unsigned int>(counter->num_of_kth_needed);
   unsigned int* p_out_cnt = &counter->out_cnt;
   unsigned int* p_out_back = &counter->out_back_cnt;
 
   __shared__ unsigned int gt_wave_counts[WAVES_PER_BLOCK];
   __shared__ unsigned int eq_wave_counts[WAVES_PER_BLOCK];
   __shared__ unsigned int gt_wave_offsets[WAVES_PER_BLOCK];
   __shared__ unsigned int eq_wave_offsets[WAVES_PER_BLOCK];
   __shared__ unsigned int gt_block_base;
   __shared__ unsigned int eq_block_base;
 
   const unsigned int lane = lane_id_in_wave();
   const unsigned int wave = wave_id_in_block();
 
   for (int base = blockIdx.x * blockDim.x; base < n; base += gridDim.x * blockDim.x) {
     const int i = base + threadIdx.x;
     const bool valid = i < n;
     const uint32_t bits = valid ? bits_data[i] : 0u;
     const bool gt = valid && (bits > kth_value_bits);
     const bool eq = valid && (bits == kth_value_bits);
 
     const unsigned long long gt_mask = __ballot(gt);
     const unsigned long long eq_mask = __ballot(eq);
     const unsigned int gt_rank = static_cast<unsigned int>(__popcll(gt_mask & lane_mask_lt(lane)));
     const unsigned int eq_rank = static_cast<unsigned int>(__popcll(eq_mask & lane_mask_lt(lane)));
     const unsigned int gt_count = static_cast<unsigned int>(__popcll(gt_mask));
     const unsigned int eq_count = static_cast<unsigned int>(__popcll(eq_mask));
 
     if (lane == 0) {
       gt_wave_counts[wave] = gt_count;
       eq_wave_counts[wave] = eq_count;
     }
     __syncthreads();  // Barrier 1: wave counts 写入 LDS 完毕
 
     // 在 Barrier 1 和 Barrier 2 之间，非 thread-0 线程原本完全空闲（等 barrier 15256 cycles）。
     // 利用这个"死区"提前为 gt/eq 线程发射 in[i] 的 global load：
     // - 只有 gt||eq 的线程发射 load（~4% 线程），不浪费带宽
     // - load latency (~3000 cycles) 被 thread-0 的工作 + Barrier 2 等待完全吸收
     // - 到 Barrier 2 结束时，in[i] 已在寄存器中，store 阶段无需再等
     float value = 0.0f;
     if (gt | eq) {
       value = in[i];
     }
 
     if (threadIdx.x == 0) {
       unsigned int gt_running = 0u;
       unsigned int eq_running = 0u;
       for (int w = 0; w < WAVES_PER_BLOCK; ++w) {
         gt_wave_offsets[w] = gt_running;
         eq_wave_offsets[w] = eq_running;
         gt_running += gt_wave_counts[w];
         eq_running += eq_wave_counts[w];
       }
       gt_block_base = gt_running ? atomicAdd(p_out_cnt, gt_running) : 0u;
       eq_block_base = eq_running ? atomicAdd(p_out_back, eq_running) : 0u;
     }
     __syncthreads();  // Barrier 2: in[i] load 在此期间完成
 
     if (gt) {
       const unsigned int pos = gt_block_base + gt_wave_offsets[wave] + gt_rank;
       if (pos < static_cast<unsigned int>(k)) {
         out[pos] = value;
         out_idx[pos] = static_cast<unsigned int>(i);
       }
     } else if (eq) {
       const unsigned int back_pos = eq_block_base + eq_wave_offsets[wave] + eq_rank;
       if (back_pos < num_of_kth_needed) {
         const unsigned int pos = static_cast<unsigned int>(k) - 1u - back_pos;
         out[pos] = value;
         out_idx[pos] = static_cast<unsigned int>(i);
       }
     }
     __syncthreads();
   }
 }
 
 int main() {
   float* data = new float[N];
   float* out = new float[K];
   unsigned int* out_idx = new unsigned int[K];
 
   for (int i = 0; i < N; ++i) {
     data[i] = static_cast<float>(rand() % 100000) / 1000.0f - 50.0f;
   }
 
   float* data_dev = nullptr;
   float* out_dev = nullptr;
   uint32_t* bits_dev = nullptr;
   unsigned int* out_idx_dev = nullptr;
   Counter* counter_dev = nullptr;
   unsigned int* global_hist_dev = nullptr;
 
   hipMalloc(reinterpret_cast<void**>(&data_dev), N * sizeof(float));
   hipMalloc(reinterpret_cast<void**>(&bits_dev), N * sizeof(uint32_t));
   hipMalloc(reinterpret_cast<void**>(&out_dev), K * sizeof(float));
   hipMalloc(reinterpret_cast<void**>(&out_idx_dev), K * sizeof(unsigned int));
   hipMalloc(reinterpret_cast<void**>(&counter_dev), sizeof(Counter));
   hipMalloc(reinterpret_cast<void**>(&global_hist_dev), NumBuckets * sizeof(unsigned int));
 
   const int preprocess_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
   const int hist_blocks = read_positive_env_or_default("TOPK_HIST_BLOCKS", 256);
   const int filter_blocks = read_positive_env_or_default("TOPK_FILTER_BLOCKS", 192);
 
   auto run_operator = [&]() {
     hipLaunchKernelGGL(preprocess_bits_kernel, dim3(preprocess_blocks), dim3(BLOCK_SIZE), 0, 0,
                        data_dev, bits_dev, N);
     hipLaunchKernelGGL(reset_kernel, dim3(1), dim3(NumBuckets), 0, 0, counter_dev, global_hist_dev);
 
     for (int pass = 0; pass < NumPasses; ++pass) {
       hipLaunchKernelGGL(radix_hist_kernel, dim3(hist_blocks), dim3(BLOCK_SIZE), 0, 0,
                          bits_dev, N, pass, counter_dev, global_hist_dev);
       hipLaunchKernelGGL(radix_scan_choose_kernel, dim3(1), dim3(BLOCK_SIZE), 0, 0,
                          pass, K, counter_dev, global_hist_dev);
     }
 
     hipLaunchKernelGGL(last_filter_kernel, dim3(filter_blocks), dim3(BLOCK_SIZE), 0, 0,
                        data_dev, bits_dev, out_dev, out_idx_dev, N, K, counter_dev);
   };
 
   hipMemcpy(data_dev, data, N * sizeof(float), hipMemcpyHostToDevice);
   run_operator();
   hipDeviceSynchronize();
   hipMemcpy(out, out_dev, K * sizeof(float), hipMemcpyDeviceToHost);
   hipMemcpy(out_idx, out_idx_dev, K * sizeof(unsigned int), hipMemcpyDeviceToHost);
 
   hipEvent_t start, stop;
   hipEventCreate(&start);
   hipEventCreate(&stop);
   const int iterations = 1;
 
   hipEventRecord(start);
   for (int iter = 0; iter < iterations; ++iter) run_operator();
   hipEventRecord(stop);
   hipEventSynchronize(stop);
 
   float ms = 0.0f;
   hipEventElapsedTime(&ms, start, stop);
   const float avg_us = (ms * 1000.0f) / iterations;
 
   std::vector<float> ref_sorted(data, data + N);
   std::sort(ref_sorted.begin(), ref_sorted.end(), [](float a, float b) { return a > b; });
 
   int correct = 0;
   std::vector<float> out_sorted(out, out + K);
   std::sort(out_sorted.begin(), out_sorted.end(), [](float a, float b) { return a > b; });
   for (int i = 0; i < K; ++i) {
     if (out_sorted[i] == ref_sorted[i]) ++correct;
   }
   const float accuracy = static_cast<float>(correct) / K * 100.0f;
 
   cout << "========================================" << endl;
   cout << " TopK Full K (ILP optimized last_filter)" << endl;
   cout << "========================================" << endl;
   cout << "N = " << N << ", K = " << K << " (FP32)" << endl;
   cout << "NumPasses = " << NumPasses << ", NumBuckets = " << NumBuckets << endl;
   cout << "HistBlocks = " << hist_blocks << ", FilterBlocks = " << filter_blocks << endl;
   cout << "Iterations: " << iterations << endl;
   cout << "----------------------------------------" << endl;
   cout << fixed << setprecision(2);
   cout << "Accuracy: " << accuracy << "% (" << correct << "/" << K << " correct)" << endl;
   cout << "Operator latency (preprocess + top-k):  " << avg_us << " us" << endl;
   cout << "========================================" << endl;
 
   delete[] data;
   delete[] out;
   delete[] out_idx;
   hipFree(data_dev);
   hipFree(bits_dev);
   hipFree(out_dev);
   hipFree(out_idx_dev);
   hipFree(counter_dev);
   hipFree(global_hist_dev);
   hipEventDestroy(start);
   hipEventDestroy(stop);
   return 0;
 }
 