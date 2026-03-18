
/**
 * Top-K 完整前 K 个输出 - 优化版（vector 2x unroll / in-flight loads）
 * 输入: N=50000 FP32, K=2048
 * Phase1: radix 前缀得到 kth_value_bits → Phase2: last_filter 稳定输出前 K
 */
 
 #include <hip/hip_runtime.h>
 #include <iostream>
 #include <algorithm>
 #include <cstdlib>
 #include <iomanip>
 #include <cstdint>
 #include <vector>

 #define N 50000
 #define K 2048
 #define BLOCK_SIZE 256
 #define VECTOR_SIZE 4

 #define NumPasses 8
 #define NumBuckets 16
 #define BitsPerPass 4

 using namespace std;

 using fp32x4 = __attribute__((__ext_vector_type__(4))) float;
using u32x4  = __attribute__((__ext_vector_type__(4))) uint32_t;

 __device__ __forceinline__ uint32_t twiddle_float(float key) {
   uint32_t x = __float_as_uint(key);
   uint32_t mask = (x & 0x80000000u) ? 0xffffffffu : 0x80000000u;
   return (key == key) ? (x ^ mask) : 0xffffffffu;
 }

 __device__ __forceinline__ int get_start_bit(int pass) {
   return 32 - (pass + 1) * BitsPerPass;  // 28,24,20,16,12,8,4,0
 }

 __device__ __forceinline__ int calc_bucket(uint32_t bits, int pass) {
   int start = get_start_bit(pass);
   return (bits >> start) & (NumBuckets - 1);
 }

__global__ void preprocess_bits_kernel(const float* __restrict__ data,
                                       uint32_t* __restrict__ bits,
                                       int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  // 2x unroll: let each thread keep two independent global loads in flight.
  for (int i = tid; i < n; i += stride * 2) {
    int i1 = i + stride;

    float x0 = data[i];
    uint32_t b0 = twiddle_float(x0);

    if (i1 < n) {
      float x1 = data[i1];
      uint32_t b1 = twiddle_float(x1);
      bits[i1] = b1;
    }

    bits[i] = b0;
  }
}

struct Counter {
  uint32_t kth_value_bits;
  int num_of_kth_needed;
  unsigned int out_cnt;
  unsigned int out_back_cnt;
};

// 用 GPU kernel 清零 counter + global_hist，避免 host 侧 hipMemset
__global__ void reset_kernel(Counter* counter, unsigned int* global_hist) {
  if (threadIdx.x == 0) {
    counter->kth_value_bits = 0;
    counter->num_of_kth_needed = 0;
    counter->out_cnt = 0;
    counter->out_back_cnt = 0;
  }
  if (threadIdx.x < NumBuckets) {
    global_hist[threadIdx.x] = 0;
  }
}

__global__ __launch_bounds__(BLOCK_SIZE) void radix_hist_kernel(
    const uint32_t* __restrict__ bits_data,
     int n,
     int pass,
     const Counter* __restrict__ counter,
     unsigned int* __restrict__ global_hist) {
   __shared__ unsigned int hist[NumBuckets];

   int prev_start_bit = (pass == 0) ? 32 : get_start_bit(pass - 1);
   uint32_t kth_bits = counter->kth_value_bits;
   uint32_t prev_mask = (pass == 0) ? 0u : (0xffffffffu << prev_start_bit);

   for (int i = threadIdx.x; i < NumBuckets; i += blockDim.x) hist[i] = 0;
   __syncthreads();

   const int num_vecs = n / VECTOR_SIZE;
  const u32x4* in_vec = reinterpret_cast<const u32x4*>(bits_data);

  for (int vec_idx = blockIdx.x * blockDim.x + threadIdx.x; vec_idx < num_vecs;
       vec_idx += gridDim.x * blockDim.x) {
    u32x4 v = in_vec[vec_idx];
#pragma unroll
    for (int j = 0; j < VECTOR_SIZE; j++) {
      uint32_t bits = v[j];
      bool count_me = (pass == 0) || (((bits >> prev_start_bit) << prev_start_bit) == (kth_bits & prev_mask));
      if (count_me) {
        int b = calc_bucket(bits, pass);
        atomicAdd(hist + b, 1u);
      }
    }
  }
   int tail_start = num_vecs * VECTOR_SIZE;
   for (int i = tail_start + blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    uint32_t bits = bits_data[i];
     bool count_me = (pass == 0) || (((bits >> prev_start_bit) << prev_start_bit) == (kth_bits & prev_mask));
     if (count_me) {
       int b = calc_bucket(bits, pass);
       atomicAdd(hist + b, 1u);
     }
   }
   __syncthreads();

   for (int i = threadIdx.x; i < NumBuckets; i += blockDim.x)
     atomicAdd(global_hist + i, hist[i]);
 }

 __global__ void radix_scan_choose_kernel(
     int pass,
     int k,
     Counter* __restrict__ counter,
     unsigned int* __restrict__ global_hist) {
   __shared__ unsigned int hist[NumBuckets];

   int start_bit = get_start_bit(pass);

   for (int i = threadIdx.x; i < NumBuckets; i += blockDim.x)
     hist[i] = global_hist[i];
   __syncthreads();

   if (threadIdx.x == 0) {
     for (int i = 1; i < NumBuckets; i++) hist[i] += hist[i - 1];
   }
   __syncthreads();

   if (threadIdx.x == 0) {
     uint32_t kth_bits = (pass == 0) ? 0u : counter->kth_value_bits;
     int k_remaining = (pass == 0) ? k : counter->num_of_kth_needed;
     unsigned int total = hist[NumBuckets - 1];
     for (int b = 0; b < NumBuckets; b++) {
       if ((int)total - k_remaining < (int)hist[b]) {
         kth_bits |= (uint32_t)b << start_bit;
         k_remaining = k_remaining - (int)(total - hist[b]);
         break;
       }
     }
     counter->kth_value_bits = kth_bits;
     counter->num_of_kth_needed = k_remaining;
     if (pass == NumPasses - 1) {
       counter->out_cnt = 0;
       counter->out_back_cnt = 0;
     }
   }
   __syncthreads();
   if (pass < NumPasses - 1) {
     for (int i = threadIdx.x; i < NumBuckets; i += blockDim.x)
       global_hist[i] = 0;
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
  // 先把 phase1 选出来的阈值信息读到寄存器。
  // 后续每个线程都会反复用到这几个值，所以提前缓存，避免循环里重复访存。
   uint32_t kth_value_bits    = counter->kth_value_bits;
   int num_of_kth_needed      = counter->num_of_kth_needed;
   unsigned int* p_out_cnt    = &counter->out_cnt;
   unsigned int* p_out_back   = &counter->out_back_cnt;

  // 这个 kernel 做最后一次全量扫描：
  // 1. bits > kth_value_bits 的元素一定属于 top-k，放到输出前半段；
  // 2. bits == kth_value_bits 的元素只需要取前 num_of_kth_needed 个，放到输出尾部；
  // 3. 两类元素都通过全局原子加抢占输出位置。
  //
  // 数据流上，这里基本是：
  // in/global + bits/global -> VGPR 比较判断 -> atomicAdd 申请位置 -> out/global 写回
  // 不使用 LDS，也没有 block 内聚合。
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    // 读入当前元素的原值和其 twiddled bits。
    // value 用于最终输出，bits 用于和 kth_value_bits 做比较。
    float value = in[i];
    uint32_t bits = bits_data[i];
    if (bits > kth_value_bits) {
      // 严格大于阈值的元素一定要保留。
      // 每命中一个元素就对全局计数器 out_cnt 做一次 atomicAdd(1)，
      // 返回值 pos 是这个线程在输出前半段中的唯一写入位置。
      unsigned int pos = atomicAdd(p_out_cnt, 1u);
      if (pos < (unsigned int)k) {
        // 理论上最终有效位置数不会超过 k，这里保留边界保护。
        out[pos] = value;
        out_idx[pos] = (unsigned int)i;
      }
    } else if (bits == kth_value_bits) {
      // 等于阈值的元素可能有很多个，但我们只需要其中一部分来把 top-k 凑满。
      // 这里仍然是“每命中一个元素做一次 atomicAdd(1)”。
      // back_pos 表示它是第几个等于 kth 的元素。
      unsigned int back_pos = atomicAdd(p_out_back, 1u);
      if (back_pos < (unsigned int)num_of_kth_needed) {
        // 等于 kth 的元素从输出尾部反向回填：
        // 假设还需要 m 个，那么第 0 个写 k-1，第 1 个写 k-2，依次类推。
        // 这样能和前面 bits > kth 的元素拼成完整 top-k。
        unsigned int pos = (unsigned int)k - 1 - back_pos;
        out[pos] = value;
        out_idx[pos] = (unsigned int)i;
      }
    }
  }
 }

 int main() {
   float* data = new float[N];
   float* out = new float[K];
   unsigned int* out_idx = new unsigned int[K];

   for (int i = 0; i < N; i++)
     data[i] = (float)(rand() % 100000) / 1000.f - 50.f;

  float *data_dev, *out_dev;
  uint32_t* bits_dev;
   unsigned int* out_idx_dev;
   Counter* counter_dev;
   unsigned int* global_hist_dev;

   hipMalloc((void**)&data_dev, N * sizeof(float));
  hipMalloc((void**)&bits_dev, N * sizeof(uint32_t));
   hipMalloc((void**)&out_dev, K * sizeof(float));
   hipMalloc((void**)&out_idx_dev, K * sizeof(unsigned int));
   hipMalloc((void**)&counter_dev, sizeof(Counter));
   hipMalloc((void**)&global_hist_dev, NumBuckets * sizeof(unsigned int));

   const int hist_blocks = (N / VECTOR_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

  auto run_all = [&]() {
    // 用 GPU kernel 清零，不走 host 侧 hipMemset
    hipLaunchKernelGGL(reset_kernel, 1, NumBuckets, 0, 0, counter_dev, global_hist_dev);
    for (int pass = 0; pass < NumPasses; pass++) {
      hipLaunchKernelGGL(radix_hist_kernel, hist_blocks, BLOCK_SIZE, 0, 0,
                         bits_dev, N, pass, counter_dev, global_hist_dev);
      hipLaunchKernelGGL(radix_scan_choose_kernel, 1, BLOCK_SIZE, 0, 0,
                         pass, K, counter_dev, global_hist_dev);
    }
    hipLaunchKernelGGL(last_filter_kernel, 256, 256, 0, 0,
                       data_dev, bits_dev, out_dev, out_idx_dev, N, K, counter_dev);
  };

  auto run_operator = [&]() {
    hipLaunchKernelGGL(preprocess_bits_kernel, (N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, 0,
                       data_dev, bits_dev, N);
    run_all();
  };

  hipMemcpy(data_dev, data, N * sizeof(float), hipMemcpyHostToDevice);

  // 先跑一次验证正确性
  run_operator();
  hipDeviceSynchronize();
  hipMemcpy(out, out_dev, K * sizeof(float), hipMemcpyDeviceToHost);
  hipMemcpy(out_idx, out_idx_dev, K * sizeof(unsigned int), hipMemcpyDeviceToHost);

  // Profiling build: skip warmup so rocprofv3 sees a single clean run.
  for (int i = 0; i < 0; i++) {
    run_operator();
  }
  hipDeviceSynchronize();

  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);
  int iterations = 1;

  // 计时：算子本身（preprocess + TopK，不含 H2D/D2H）
  hipEventRecord(start);
  for (int iter = 0; iter < iterations; iter++) {
    run_operator();
  }
  hipEventRecord(stop);
  hipEventSynchronize(stop);
  float ms = 0;
  hipEventElapsedTime(&ms, start, stop);
  float avg_us = (ms * 1000.0f) / iterations;

   std::vector<float> ref_sorted(data, data + N);
   std::sort(ref_sorted.begin(), ref_sorted.end(), [](float a, float b) { return a > b; });

   int correct = 0;
   std::vector<float> out_sorted(out, out + K);
   std::sort(out_sorted.begin(), out_sorted.end(), [](float a, float b) { return a > b; });
   for (int i = 0; i < K; i++) {
     if (out_sorted[i] == ref_sorted[i]) correct++;
   }
   float accuracy = (float)correct / K * 100.0f;

   cout << "========================================" << endl;
   cout << "   TopK Full K (best: 8 pass x 4 bit)   " << endl;
   cout << "========================================" << endl;
   cout << "N = " << N << ", K = " << K << " (FP32)" << endl;
   cout << "NumPasses = " << NumPasses << ", NumBuckets = " << NumBuckets << endl;
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
