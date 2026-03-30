/**
 * TopK + Compaction（opt2）：单次 compact 的 Radix Top-K
 *
 * 相对 opt1：在 pass 0 的 scan_choose 之后增加一次 compact_kernel，后续 pass 只遍历
 * compact 缓冲区，避免每轮都扫全量 N。
 *
 * 流程概要
 *   - opt1：每一 pass 的直方图都扫满 N 个元素（8 个 pass × N，量级约 8N）。
 *   - opt2：pass 0 的 scan_choose 之后调用 compact_kernel：
 *       · bits 的高位前缀已严格大于第 k 大 twiddle 值 → 直接写入 out_idx（已确定在 Top-K）
 *       · 高位前缀与第 k 大相等 → 仍可能进 Top-K，写入 compact buf，供后续 pass 细分
 *       · 高位前缀更小 → 不可能进 Top-K，丢弃
 *     compact buf 规模约为 N / 16（每 pass 区分 16 个桶，首 pass 后候选约剩 1/16）。
 *   - pass 1～7：仅对 compact buf 建直方图并 scan_choose，不再读全量数组。
 *   - last_filter：仍在 compact buf 上，用完整 32 位 twiddle 与 kth 比较，补齐 out_idx。
 *
 * 访存量（量级，与下方宏 N 一致）
 *   hist(pass0) + compact(pass0) + hist(pass1～7) + last_filter
 *   ≈ N + N + (N/16)×8 ≈ 2.5N（opt1 约 8N + N，此处为示意对比）。
 *
 * 核函数个数：预处理 1 + pass0(hist+scan+compact) + pass1～7 各(hist+scan) + last_filter
 *   ≈ 1 + 3 + 14 + 1，比 opt1 多 1 个 compact kernel。
 *
 * 性能：后续 pass 的直方图由「全 N」缩为「约 N/16」，通常可抵消 compact 的额外开销。
 */

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#define N 60000
#define K 2048
#define BLOCK_SIZE 256
#define VECTOR_SIZE 4
#define WAVE_SIZE 64
#define WAVES_PER_BLOCK (BLOCK_SIZE / WAVE_SIZE)

#define NumPasses 8
#define NumBuckets 16
#define BitsPerPass 4

// compact 缓冲区最大容量：pass 0 之后候选数约为 N/16，取 N 作为上界以防极端分布
#define BUF_LEN N

using namespace std;
using u32x4 = __attribute__((__ext_vector_type__(4))) uint32_t;

static inline int read_positive_env_or_default(const char* name, int default_value) {
  const char* value = std::getenv(name);
  if (!value || !*value) return default_value;
  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed <= 0) return default_value;
  return static_cast<int>(parsed);
}

struct Counter {
  uint32_t kth_value_bits;
  int num_of_kth_needed;
  unsigned int out_cnt;       // 已写入 out_idx「前半段」的、确定大于第 k 大的个数
  unsigned int out_back_cnt;  // last_filter 中等于第 k 大、写入 out_idx「尾部」的计数
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

// 阶段一：float → 可排序的 twiddle 位模式，并清零 Counter / 直方图 / compact 计数
// 与重置融合在同一核内，少一次启动开销
__global__ void preprocess_bits_kernel(const float* __restrict__ data,
                                       uint32_t* __restrict__ bits, int n,
                                       Counter* __restrict__ counter,
                                       uint32_t* __restrict__ global_hist,
                                       unsigned int* __restrict__ compact_n) {
  // 仅 block 0 单线程重置全局状态
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    counter->kth_value_bits = 0u;
    counter->num_of_kth_needed = 0;
    counter->out_cnt = 0u;
    counter->out_back_cnt = 0u;
    *compact_n = 0u;
  }
  if (blockIdx.x == 0 && threadIdx.x < NumBuckets) global_hist[threadIdx.x] = 0u;

  // 按步长 2×stride 展开，与原始实现一致
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (int i = tid; i < n; i += stride * 2) {
    bits[i] = twiddle_float(data[i]);
    int i1 = i + stride;
    if (i1 < n) bits[i1] = twiddle_float(data[i1]);
  }
}

// radix_hist_kernel：按当前 pass 对 twiddle 位分桶计数（含前缀过滤，与 opt1 一致）
//   pass 0：遍历全长 bits（n = n_full），不做前缀过滤
//   pass 1：遍历 compact buf（n = *compact_n），元素已保证与 kth 的首个 4 位段一致
//   pass 2～7：仍遍历 compact buf，仅统计与 kth 已确定的高位前缀一致的元素
// kth 的已确定前缀来自 counter->kth_value_bits，保证直方图只统计尚未被淘汰的候选
__global__ __launch_bounds__(BLOCK_SIZE) void radix_hist_kernel(
    const uint32_t* __restrict__ bits_cur,
    int pass,
    int n_full,
    const unsigned int* __restrict__ compact_n,
    const Counter* __restrict__ counter,        // 读取 kth_value_bits 做前缀匹配
    uint32_t* __restrict__ global_hist) {
  __shared__ unsigned int hist[NumBuckets];

  const int n = (compact_n != nullptr) ? (int)(*compact_n) : n_full;

  // 前缀过滤：仅当元素与 kth 在更高有效位上已一致时才入桶（pass 0 时不过滤）
  const int prev_start_bit = (pass == 0) ? 32 : get_start_bit(pass - 1);
  const uint32_t prev_mask = (pass == 0) ? 0u : (0xffffffffu << prev_start_bit);
  const uint32_t kth_prefix = counter->kth_value_bits & prev_mask;

  for (int b = threadIdx.x; b < NumBuckets; b += blockDim.x) hist[b] = 0u;
  __syncthreads();

  const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_stride = gridDim.x * blockDim.x;
  const int num_vecs = n / VECTOR_SIZE;
  const u32x4* in_vec = reinterpret_cast<const u32x4*>(bits_cur);

  for (int vi = global_tid; vi < num_vecs; vi += global_stride) {
    u32x4 v = in_vec[vi];
#pragma unroll
    for (int j = 0; j < VECTOR_SIZE; ++j) {
      const uint32_t bits = v[j];
      if (pass == 0 || ((bits & prev_mask) == kth_prefix))
        atomicAdd(hist + calc_bucket(bits, pass), 1u);
    }
  }
  for (int i = num_vecs * VECTOR_SIZE + global_tid; i < n; i += global_stride) {
    const uint32_t bits = bits_cur[i];
    if (pass == 0 || ((bits & prev_mask) == kth_prefix))
      atomicAdd(hist + calc_bucket(bits, pass), 1u);
  }
  __syncthreads();

  for (int b = threadIdx.x; b < NumBuckets; b += blockDim.x)
    if (hist[b]) atomicAdd(global_hist + b, hist[b]);
}

// radix_scan_choose：由前缀和确定当前 pass 的 kth 桶，并更新 kth 位模式；out_cnt 跨 pass 累加、不在此清零
__global__ void radix_scan_choose_kernel(int pass, int k, Counter* __restrict__ counter,
                                         uint32_t* __restrict__ global_hist) {
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
      if ((int)total - k_remaining < (int)hist[b]) {
        kth_bits |= (uint32_t)b << start_bit;
        k_remaining -= (int)(total - hist[b]);
        break;
      }
    }
    counter->kth_value_bits = kth_bits;
    counter->num_of_kth_needed = k_remaining;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < NumBuckets; i += blockDim.x) global_hist[i] = 0u;
}

// compact_kernel：仅在 pass 0 的 scan_choose 之后调用一次
//   当前 pass 掩码后的 bits > kth → 已确定属于 Top-K → 写入 out_idx
//   掩码后 == kth → 仍可能等于最终第 k 大 → 写入 compact buf（bits_out / idx_out）
//   掩码后 < kth → 不可能进 Top-K → 丢弃
// pass 0 时 bits_in 为全长 twiddle 数组，orig_idx 即全局下标 i
// 用 wave ballot + block 内前缀和聚合，减少全局 atomicAdd
__global__ __launch_bounds__(BLOCK_SIZE) void compact_kernel(
    const uint32_t* __restrict__ bits_in,
    uint32_t* __restrict__ bits_out,
    uint32_t* __restrict__ idx_out,
    uint32_t* __restrict__ out_idx,
    int pass,
    int n,      // 本核处理的元素个数（pass 0 时为 N）
    int k,
    unsigned int* __restrict__ compact_n,  // 原子变量：写入 compact buf 的元素个数
    Counter* __restrict__ counter) {
  const uint32_t kth_value_bits = counter->kth_value_bits;
  const int start_bit = get_start_bit(pass);
  // cur_mask：pass 0 只比较当前 4 位段（由高到低第一段）
  const uint32_t cur_mask = 0xffffffffu << start_bit;
  unsigned int* p_out_cnt = &counter->out_cnt;

  const unsigned int lane = lane_id_in_wave();
  const unsigned int wave = wave_id_in_block();

  __shared__ unsigned int gt_wave_counts[WAVES_PER_BLOCK];
  __shared__ unsigned int eq_wave_counts[WAVES_PER_BLOCK];
  __shared__ unsigned int gt_wave_offsets[WAVES_PER_BLOCK];
  __shared__ unsigned int eq_wave_offsets[WAVES_PER_BLOCK];
  __shared__ unsigned int gt_block_base;
  __shared__ unsigned int eq_block_base;

  for (int base = blockIdx.x * blockDim.x; base < n; base += gridDim.x * blockDim.x) {
    const int local_i = base + threadIdx.x;
    const bool valid = local_i < n;

    uint32_t bits = 0u;
    uint32_t orig_idx = 0u;
    if (valid) {
      bits = bits_in[local_i];
      orig_idx = (uint32_t)local_i;  // pass 0：下标即原数组索引
    }

    const uint32_t bits_masked = bits & cur_mask;
    const bool gt = valid && (bits_masked > kth_value_bits);
    const bool eq = valid && (bits_masked == kth_value_bits);

    const unsigned long long gt_ballot = __ballot(gt);
    const unsigned long long eq_ballot = __ballot(eq);
    const unsigned int gt_rank = (unsigned int)__popcll(gt_ballot & lane_mask_lt(lane));
    const unsigned int eq_rank = (unsigned int)__popcll(eq_ballot & lane_mask_lt(lane));
    const unsigned int gt_count = (unsigned int)__popcll(gt_ballot);
    const unsigned int eq_count = (unsigned int)__popcll(eq_ballot);

    if (lane == 0) {
      gt_wave_counts[wave] = gt_count;
      eq_wave_counts[wave] = eq_count;
    }
    __syncthreads();  // 同步：wave 计数写回 shared

    if (threadIdx.x == 0) {
      unsigned int gt_run = 0u, eq_run = 0u;
      for (int w = 0; w < WAVES_PER_BLOCK; ++w) {
        gt_wave_offsets[w] = gt_run;
        eq_wave_offsets[w] = eq_run;
        gt_run += gt_wave_counts[w];
        eq_run += eq_wave_counts[w];
      }
      gt_block_base = gt_run ? atomicAdd(p_out_cnt, gt_run) : 0u;
      eq_block_base = eq_run ? atomicAdd(compact_n, eq_run) : 0u;
    }
    __syncthreads();  // 同步：block 基址写入后再 scatter

    if (gt) {
      const unsigned int pos = gt_block_base + gt_wave_offsets[wave] + gt_rank;
      if (pos < (unsigned int)k) {
        out_idx[pos] = orig_idx;
      }
    } else if (eq) {
      const unsigned int pos = eq_block_base + eq_wave_offsets[wave] + eq_rank;
      bits_out[pos] = bits;
      idx_out[pos] = orig_idx;
    }
    __syncthreads();
  }
}

// last_filter：在 compact buf（bits_in / idx_in，长度 *compact_n_ptr）上做最终划分
// 候选在 pass 0 已与 kth 的首段 4 位对齐；此处用完整 32 位 twiddle 比较：
//   bits > kth_value_bits → 确定大于第 k 大 → 写入 out_idx 前部（与 out_cnt 配合）
//   bits == kth_value_bits → 按需取 num_of_kth_needed 个 → 写入 out_idx 后部
// 同样用 ballot + block 内聚合降低 atomic 竞争
__global__ __launch_bounds__(BLOCK_SIZE) void last_filter_kernel(
    const uint32_t* __restrict__ bits_in,
    const uint32_t* __restrict__ idx_in,
    uint32_t* __restrict__ out_idx,
    int k,
    const unsigned int* __restrict__ compact_n_ptr,
    Counter* __restrict__ counter) {
  const int n = (int)(*compact_n_ptr);
  const uint32_t kth_value_bits = counter->kth_value_bits;
  const unsigned int num_of_kth_needed = (unsigned int)counter->num_of_kth_needed;
  unsigned int* p_out_cnt = &counter->out_cnt;
  unsigned int* p_out_back = &counter->out_back_cnt;

  const unsigned int lane = lane_id_in_wave();
  const unsigned int wave = wave_id_in_block();

  __shared__ unsigned int gt_wave_counts[WAVES_PER_BLOCK];
  __shared__ unsigned int eq_wave_counts[WAVES_PER_BLOCK];
  __shared__ unsigned int gt_wave_offsets[WAVES_PER_BLOCK];
  __shared__ unsigned int eq_wave_offsets[WAVES_PER_BLOCK];
  __shared__ unsigned int gt_block_base;
  __shared__ unsigned int eq_block_base;

  for (int base = blockIdx.x * blockDim.x; base < n; base += gridDim.x * blockDim.x) {
    const int i = base + threadIdx.x;
    const bool valid = i < n;
    const uint32_t bits = valid ? bits_in[i] : 0u;
    const bool gt = valid && (bits > kth_value_bits);
    const bool eq = valid && (bits == kth_value_bits);

    const unsigned long long gt_ballot = __ballot(gt);
    const unsigned long long eq_ballot = __ballot(eq);
    const unsigned int gt_rank = (unsigned int)__popcll(gt_ballot & lane_mask_lt(lane));
    const unsigned int eq_rank = (unsigned int)__popcll(eq_ballot & lane_mask_lt(lane));
    const unsigned int gt_count = (unsigned int)__popcll(gt_ballot);
    const unsigned int eq_count = (unsigned int)__popcll(eq_ballot);

    if (lane == 0) {
      gt_wave_counts[wave] = gt_count;
      eq_wave_counts[wave] = eq_count;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      unsigned int gt_run = 0u, eq_run = 0u;
      for (int w = 0; w < WAVES_PER_BLOCK; ++w) {
        gt_wave_offsets[w] = gt_run;
        eq_wave_offsets[w] = eq_run;
        gt_run += gt_wave_counts[w];
        eq_run += eq_wave_counts[w];
      }
      gt_block_base = gt_run ? atomicAdd(p_out_cnt, gt_run) : 0u;
      eq_block_base = eq_run ? atomicAdd(p_out_back, eq_run) : 0u;
    }
    __syncthreads();

    if (gt) {
      const unsigned int pos = gt_block_base + gt_wave_offsets[wave] + gt_rank;
      if (pos < (unsigned int)k) {
        out_idx[pos] = idx_in[i];
      }
    } else if (eq) {
      const unsigned int back_pos = eq_block_base + eq_wave_offsets[wave] + eq_rank;
      if (back_pos < num_of_kth_needed) {
        const unsigned int pos = (unsigned int)k - 1u - back_pos;
        out_idx[pos] = idx_in[i];
      }
    }
    __syncthreads();
  }
}


int main() {
  float* data = new float[N];
  unsigned int* out_idx = new unsigned int[K];

  {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1e6f, 1e6f);
    for (int i = 0; i < N; ++i) data[i] = dist(rng);
  }

  float* data_dev = nullptr;
  uint32_t* bits_dev = nullptr;
  unsigned int* out_idx_dev = nullptr;
  Counter* counter_dev = nullptr;
  uint32_t* global_hist_dev = nullptr;

  // compact 缓冲区：pass 0 的 compact_kernel 写入；pass 1～7 与 last_filter 只读此处
  uint32_t* bits_buf = nullptr;   // 候选元素的 twiddle 位
  uint32_t* idx_buf = nullptr;    // 候选在原输入 data 中的下标
  unsigned int* compact_n_dev = nullptr;  // 当前 compact 区有效长度（device 上原子更新）

  hipMalloc((void**)&data_dev, N * sizeof(float));
  hipMalloc((void**)&bits_dev, N * sizeof(uint32_t));
  hipMalloc((void**)&out_idx_dev, K * sizeof(unsigned int));
  hipMalloc((void**)&counter_dev, sizeof(Counter));
  hipMalloc((void**)&global_hist_dev, NumBuckets * sizeof(uint32_t));
  hipMalloc((void**)&bits_buf, BUF_LEN * sizeof(uint32_t));
  hipMalloc((void**)&idx_buf, BUF_LEN * sizeof(uint32_t));
  hipMalloc((void**)&compact_n_dev, sizeof(unsigned int));

  const int preprocess_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int hist_blocks = read_positive_env_or_default("TOPK_HIST_BLOCKS", 256);
  const int compact_blocks = read_positive_env_or_default("TOPK_COMPACT_BLOCKS", 256);
  const int filter_blocks = read_positive_env_or_default("TOPK_FILTER_BLOCKS", 256);

  hipMemcpy(data_dev, data, N * sizeof(float), hipMemcpyHostToDevice);

  hipStream_t stream;
  hipStreamCreate(&stream);

  auto run_operator = [&]() {
    // 阶段一：预处理 twiddle + 重置计数（融合核）
    hipLaunchKernelGGL(preprocess_bits_kernel, dim3(preprocess_blocks), dim3(BLOCK_SIZE), 0, stream,
                       data_dev, bits_dev, N, counter_dev, global_hist_dev, compact_n_dev);

    // 阶段二：pass 0 — 全量直方图 + scan_choose + 一次 compact（生成 compact 区）
    hipLaunchKernelGGL(radix_hist_kernel, dim3(hist_blocks), dim3(BLOCK_SIZE), 0, stream,
                       bits_dev, 0, N, (const unsigned int*)nullptr, counter_dev, global_hist_dev);
    hipLaunchKernelGGL(radix_scan_choose_kernel, dim3(1), dim3(BLOCK_SIZE), 0, stream,
                       0, K, counter_dev, global_hist_dev);
    // compact_kernel 仅调用一次：
    //   当前段掩码后 > kth → 写入 out_idx；== kth → 写入 bits_buf/idx_buf（约 N/16 个候选）
    hipLaunchKernelGGL(compact_kernel, dim3(compact_blocks), dim3(BLOCK_SIZE), 0, stream,
                       bits_dev,
                       bits_buf, idx_buf,
                       out_idx_dev,
                       0, N, K,
                       compact_n_dev,
                       counter_dev);

    // 阶段三：pass 1～7 — 仅对 compact 区（bits_buf，长度 *compact_n_dev）建直方图
    for (int pass = 1; pass < NumPasses; ++pass) {
      hipLaunchKernelGGL(radix_hist_kernel, dim3(hist_blocks), dim3(BLOCK_SIZE), 0, stream,
                         bits_buf, pass, N, compact_n_dev, counter_dev, global_hist_dev);
      hipLaunchKernelGGL(radix_scan_choose_kernel, dim3(1), dim3(BLOCK_SIZE), 0, stream,
                         pass, K, counter_dev, global_hist_dev);
    }

    // 阶段四：last_filter — 在 compact 区上按完整 kth 位模式写出最终 Top-K 下标
    hipLaunchKernelGGL(last_filter_kernel, dim3(filter_blocks), dim3(BLOCK_SIZE), 0, stream,
                       bits_buf, idx_buf, out_idx_dev, K,
                       compact_n_dev, counter_dev);
  };

  // 预热，稳定计时
  for (int i = 0; i < 50; ++i) run_operator();
  hipStreamSynchronize(stream);

  // 正式计时
  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);
  const int iterations = 200;

  hipEventRecord(start, stream);
  for (int iter = 0; iter < iterations; ++iter) run_operator();
  hipEventRecord(stop, stream);
  hipEventSynchronize(stop);

  float ms = 0.0f;
  hipEventElapsedTime(&ms, start, stop);
  const float avg_us = (ms * 1000.0f) / iterations;

  // 正确性：预热后再跑一轮，将结果拷回 CPU
  run_operator();
  hipStreamSynchronize(stream);
  hipMemcpy(out_idx, out_idx_dev, K * sizeof(unsigned int), hipMemcpyDeviceToHost);

  // 校验：按下标取回 float，与 CPU partial_sort 得到的 Top-K 多重集合比较
  std::vector<float> gpu_vals(K), cpu_vals(K);
  std::vector<int> sorted_idx(N);
  std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
  std::partial_sort(sorted_idx.begin(), sorted_idx.begin() + K, sorted_idx.end(),
                    [&](int a, int b) { return data[a] > data[b]; });
  for (int i = 0; i < K; ++i) {
    gpu_vals[i] = (out_idx[i] < (unsigned int)N) ? data[out_idx[i]] : -1e30f;
    cpu_vals[i] = data[sorted_idx[i]];
  }
  std::sort(gpu_vals.begin(), gpu_vals.end(), std::greater<float>());
  std::sort(cpu_vals.begin(), cpu_vals.end(), std::greater<float>());
  bool match = (gpu_vals == cpu_vals);

  cout << "========================================" << endl;
  cout << " TopK Compaction (opt2)                 " << endl;
  cout << "========================================" << endl;
  cout << "N = " << N << ", K = " << K << " (FP32)" << endl;
  cout << "HistBlocks=" << hist_blocks << " CompactBlocks=" << compact_blocks
       << " FilterBlocks=" << filter_blocks << endl;
  cout << "Iterations: " << iterations << endl;
  cout << "----------------------------------------" << endl;
  cout << fixed << setprecision(2);
  cout << "Top-K value set match: " << (match ? "PASS" : "FAIL") << endl;
  cout << "Operator latency (preprocess + top-k):  " << avg_us << " us" << endl;
  cout << "========================================" << endl;

  delete[] data;
  delete[] out_idx;
  hipFree(data_dev);
  hipFree(bits_dev);
  hipFree(out_idx_dev);
  hipFree(counter_dev);
  hipFree(global_hist_dev);
  hipFree(bits_buf);
  hipFree(idx_buf);
  hipFree(compact_n_dev);
  hipEventDestroy(start);
  hipEventDestroy(stop);
  hipStreamDestroy(stream);
  return 0;
}