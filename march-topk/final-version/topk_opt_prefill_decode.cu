// SPDX-License-Identifier: MIT
// ============================================================================
// topk_opt5.cu — 基于 radix-select 的 GPU Top-K 优化版本
// ============================================================================
//
// 算法概述：
//   使用基数排序(radix sort)的思想进行 top-K 选择。将 32 位浮点数转换为无符号整数
//   (twiddle)后，从最高位开始，每次取 BitsPerPass 位构建直方图，通过前缀和 + 分桶
//   确定第 K 大元素所在的桶，逐步缩小搜索范围。
//   - BitsPerPass=11 → 2048 个桶，只需 3 个 pass 覆盖 32 位
//   - 使用单 block (1024 threads) 执行全部逻辑，避免多 kernel launch 开销
//
// 相对 baseline (topk_baseline_oneblock.cu) 的主要优化：
//   [OPT3] calc_bucket: 使用 __builtin_amdgcn_ubfe (v_bfe_u32 单指令) 替代 shift+mask
//   [OPT3] vectorized_process: 移除 batching (acc/prev_bin_idx)，直接 atomicAdd
//   [OPT3/4] vectorized_process: 4x 展开宽加载，load-compute 交织提高 VMEM 利用率
//   [OPT5-A] filter_and_histogram_for_one_block: pass>0 的 !out_buf 路径优化
//            - baseline 使用 for+stride 标量循环；opt5 改用 vectorized_process 宽加载
//            - 同时避免重复调用 twiddle_in，先算一次 bits 再分别提取 prefix 和 bucket
//   [OPT5-B] last_filter: !in_idx_buf 路径改用 vectorized_process (dwordx4 宽加载)
//            - baseline 使用 for(i += blockDim.x) 标量循环
//   [OPT5-C] radix_topk_one_block_kernel: 禁用 compact，所有 pass 读原始输入
//            - baseline 使用 set_buf_pointers + compact buffer 乒乓；
//              opt5 所有 pass 传 out_buf=nullptr，走更快的 vectorized_process 路径
//
// 性能 (AMD MI300X, gfx942, N=60000, K=2048):
//   baseline: ~51.6us → opt5: ~47.45us
//
// 编译:
//   hipcc -O3 -std=c++17 topk_opt5.cu -o topk_opt5 --offload-arch=gfx942
// ============================================================================

#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include <vector>

// HIP 错误检查宏
#ifndef HIP_CALL
#define HIP_CALL(call)                                                       \
    do                                                                       \
    {                                                                        \
        hipError_t err = call;                                               \
        if(err != hipSuccess)                                                \
        {                                                                    \
            printf("\n[TOPK] %s:%d fail to call %s ---> [HIP error](%s)\n",   \
                   __FILE__,                                                 \
                   __LINE__,                                                 \
                   #call,                                                    \
                   hipGetErrorString(err));                                  \
            std::exit(0);                                                    \
        }                                                                    \
    } while(0)
#endif

// 获取 GPU 的 Compute Unit 数量（用于多 block 路径的 grid 尺寸计算）
static inline uint32_t get_num_cu_func()
{
    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
    return static_cast<uint32_t>(dev_prop.multiProcessorCount);
}

namespace aiter {

// ============================================================================
// 向量类型定义
// fp32x4 = 4个float打包，对应128位 (global_load_dwordx4 指令)
// ============================================================================
using fp32x1 = __attribute__((__ext_vector_type__(1))) float;
using fp32x2 = __attribute__((__ext_vector_type__(2))) float;
using fp32x4 = __attribute__((__ext_vector_type__(4))) float;
using fp32x8 = __attribute__((__ext_vector_type__(8))) float;

// 编译期从 vec 数量映射到向量类型
template <int vec>
struct to_vector;

template <>
struct to_vector<1>
{
    using type = fp32x1;
};
template <>
struct to_vector<2>
{
    using type = fp32x2;
};
template <>
struct to_vector<4>
{
    using type = fp32x4;
};
template <>
struct to_vector<8>
{
    using type = fp32x8;
};

// WideT = fp32x4，每次宽加载读取 4 个 float = 16 字节 = 128 位
using WideT                        = fp32x4;
constexpr int VECTORIZED_READ_SIZE = 16;   // 16 字节 = sizeof(fp32x4)
constexpr int WARP_SIZE            = 64;   // AMD GPU wavefront 大小 = 64

// 执行阶段：Prefill（预填充/prompt阶段）或 Decode（解码/生成阶段）
enum class Phase
{
    Prefill,
    Decode,
};

// ============================================================================
// 基础工具函数（与 baseline 相同，无改动）
// ============================================================================

// 计算桶数：2^BitsPerPass。BitsPerPass=11 → 2048 个桶
template <int BitsPerPass>
__host__ __device__ constexpr int calc_num_buckets()
{
    return 1 << BitsPerPass;
}

// 向上取整除法
template <typename IntType>
constexpr __host__ __device__ IntType ceildiv(IntType a, IntType b)
{
    return (a + b - 1) / b;
}

// 对齐到 b 的倍数
template <typename IntType>
constexpr __host__ __device__ IntType alignTo(IntType a, IntType b)
{
    return ceildiv(a, b) * b;
}

// 计算需要多少个 pass：sizeof(T)*8 / BitsPerPass 向上取整
// float32 + 11-bit → ceil(32/11) = 3 个 pass
template <typename T, int BitsPerPass>
__host__ __device__ constexpr int calc_num_passes()
{
    return ceildiv<int>(sizeof(T) * 8, BitsPerPass);
}

// 计算第 pass 轮从第几个 bit 开始提取
// pass=0: bit 21 (最高11位: bit31..21)
// pass=1: bit 10 (中间11位: bit20..10)
// pass=2: bit  0 (最低10位: bit9..0，最后一个pass可能不足11位)
template <typename T, int BitsPerPass>
__device__ constexpr int calc_start_bit(int pass)
{
    int start_bit = static_cast<int>(sizeof(T) * 8) - (pass + 1) * BitsPerPass;
    int r         = start_bit < 0 ? 0 : start_bit;
    return r;
}

// 计算第 pass 轮的掩码，用于提取对应位段
// 正常情况下 mask = (1<<11)-1 = 0x7FF；最后一个 pass 可能只有10位
template <typename T, int BitsPerPass>
__device__ constexpr unsigned calc_mask(int pass)
{
    static_assert(BitsPerPass <= 31);
    int num_bits = calc_start_bit<T, BitsPerPass>(pass - 1) - calc_start_bit<T, BitsPerPass>(pass);
    return (1 << num_bits) - 1;
}

// ============================================================================
// twiddle_in: 将浮点数转换为无符号整数，使得大小顺序保持一致
// ============================================================================
// 浮点数的 IEEE 754 编码不能直接按 uint32 比较大小（符号位、指数的问题）。
// twiddle_in 做如下变换：
//   正数: bits ^ 0x7FFFFFFF（翻转除符号位外的所有位）
//   负数: bits ^ 0x00000000（不翻转）
// 变换后 uint32 的大小顺序 = 原始 float 从大到小的顺序（适合 select largest）。
// 与 baseline 相同，无改动。
template <typename T>
__device__ typename hipcub::Traits<T>::UnsignedBits twiddle_in(T key, bool select_min)
{
    auto bits = reinterpret_cast<typename hipcub::Traits<T>::UnsignedBits&>(key);
    if constexpr(std::is_same_v<T, float>)
    {
        uint32_t mask = (key < 0) ? 0 : 0x7fffffff;
        return bits ^ mask;
    }
    else
    {
        bits = hipcub::Traits<T>::TwiddleIn(bits);
        if(!select_min)
        {
            bits = ~bits;
        }
        return bits;
    }
}

// ============================================================================
// calc_bucket: 计算元素属于哪个桶
// ============================================================================
// [OPT3 改动] baseline 使用 (twiddle_in(x) >> start_bit) & mask，编译为 2 条指令:
//   v_lshrrev_b32 + v_and_b32
// opt3 改为 __builtin_amdgcn_ubfe，编译为单条指令:
//   v_bfe_u32 (bit field extract)
// 每个元素节省 1 条 ALU 指令。
template <typename T, int BitsPerPass>
__device__ int calc_bucket(T x, int start_bit, unsigned mask, bool select_min)
{
    static_assert(BitsPerPass <= sizeof(int) * 8 - 1,
                  "BitsPerPass is too large that the result type could not be int");
    unsigned bits = twiddle_in(x, select_min);
    return __builtin_amdgcn_ubfe(bits, static_cast<unsigned>(start_bit), static_cast<unsigned>(BitsPerPass));
}

// 判断是否为2的幂（与 baseline 相同）
template <typename I>
constexpr inline std::enable_if_t<std::is_integral<I>::value, bool>
is_a_power_of_two(I val) noexcept
{
    return ((val - 1) & val) == 0;
}

// 计算 compact buffer 的长度（与 baseline 相同）
// 对于 float/int, ratio=4, 所以 buf_len = len/(4*8) = len/32
// N=60000 → buf_len = 1856 (对齐到64)
// 注意：在 opt5 的单 block 路径中 compact 被禁用，此函数仅用于计算 workspace 大小
template <typename T, typename IdxT, typename RATIO_T = float>
__host__ __device__ IdxT calc_buf_len(IdxT len)
{
    constexpr RATIO_T ratio = 2 + sizeof(IdxT) * 2 / sizeof(T);
    IdxT buf_len            = len / (ratio * 8);
    static_assert(is_a_power_of_two(sizeof(T)));
    static_assert(is_a_power_of_two(sizeof(IdxT)));
    constexpr IdxT aligned = 256 / std::min(sizeof(T), sizeof(IdxT));
    buf_len                = buf_len & (~(aligned - 1));
    return buf_len;
}

// ============================================================================
// vectorized_process: 向量化数据遍历 + 回调处理
// ============================================================================
// 这是 opt5 中最核心的数据加载函数，被 filter_and_histogram 和 last_filter 共用。
//
// [OPT3 改动] baseline 的 lambda 签名是 f(value, idx, acc, prev_bin_idx, is_last)
//   其中 acc/prev_bin_idx 用于"连续相同桶批量累加"优化。但对于 11-bit radix (2048桶)，
//   随机数据中连续元素落入同一桶的概率仅 1/2048，批量累加几乎不触发，
//   反而每个元素额外付出 ~10 条分支指令。opt3 移除了 acc/prev_bin_idx 参数，
//   lambda 简化为 f(value, idx)，每个元素直接 atomicAdd。
//
// [OPT3/4 改动] baseline 的宽加载循环只有单层展开:
//   for(i += num_threads) { wide.scalar = in_cast[i]; process 4 elements; }
//   opt3/4 改为 4x 展开 + load-compute 交织:
//   - 先发射 wide0, wide1 两个 global_load_dwordx4
//   - 处理 wide0 的 4 个元素（此时 wide1 的加载可以并行进行）
//   - 再发射 wide2, wide3 两个 global_load_dwordx4
//   - 处理 wide1, wide2, wide3 的元素
//   这样编译器可以用 s_waitcnt vmcnt(1) 让多个宽加载流水线化，
//   而不是 vmcnt(0) 等每个加载完成后才处理。
//
// 参数:
//   thread_rank: 当前线程在 block 中的 ID (threadIdx.x)
//   num_threads: block 中的线程总数 (blockDim.x)
//   in:          输入数据指针
//   len:         输入数据长度
//   f:           对每个元素调用的回调函数 f(value, index)
template <typename T, typename IdxT, typename Func>
__device__ void
vectorized_process(size_t thread_rank, size_t num_threads, T const* in, IdxT len, Func f)
{
    // 当 T 的大小 >= WideT (fp32x4=16字节) 时，无法做宽加载，退化为标量循环
    if constexpr(sizeof(T) >= sizeof(WideT))
    {
        for(IdxT i = thread_rank; i < len; i += num_threads)
        {
            f(in[i], i);
        }
    }
    else
    {
        static_assert(sizeof(WideT) % sizeof(T) == 0);
        // items_per_scalar: 一个 WideT 包含多少个 T。fp32x4 / float = 4
        constexpr int items_per_scalar = sizeof(WideT) / sizeof(T);

        // union 用于在 WideT 和 T[4] 之间转换，避免额外的内存操作
        union
        {
            WideT scalar;
            T array[items_per_scalar];
        } wide0, wide1, wide2, wide3;

        // 处理指针对齐：in 可能不是 16 字节对齐的，
        // 需要先用标量加载处理开头未对齐的元素
        int skip_cnt =
            (reinterpret_cast<size_t>(in) % sizeof(WideT))
                ? ((sizeof(WideT) - reinterpret_cast<size_t>(in) % sizeof(WideT)) / sizeof(T))
                : 0;
        if(skip_cnt > len)
        {
            skip_cnt = len;
        }
        // in_cast: 对齐后的 WideT 指针，可以用 global_load_dwordx4 加载
        WideT const* in_cast = reinterpret_cast<decltype(in_cast)>(in + skip_cnt);
        // len_cast: 对齐部分可以做多少次宽加载
        const IdxT len_cast  = (len - skip_cnt) / items_per_scalar;

        // ---- 主循环：4x 展开的宽加载 ----
        // 每次迭代处理 4 * items_per_scalar = 16 个元素
        // strideW = num_threads * 4：所有线程的步长
        IdxT i = thread_rank;
        const IdxT strideW = num_threads * 4;
        for(; i + num_threads * 3 < len_cast; i += strideW)
        {
            // 发射两个 global_load_dwordx4（wide0, wide1）
            // GPU 的 VMEM 流水线可以同时处理多个加载请求
            wide0.scalar = in_cast[i + num_threads * 0];
            wide1.scalar = in_cast[i + num_threads * 1];

            // 处理 wide0（此时 wide1 的加载在后台进行，编译器插入 s_waitcnt vmcnt(1)）
            {
                const IdxT real_i = skip_cnt + (i + num_threads * 0) * items_per_scalar;
#pragma unroll
                for(int j = 0; j < items_per_scalar; ++j)
                {
                    f(wide0.array[j], real_i + j);
                }
            }

            // 再发射两个 global_load_dwordx4（wide2, wide3）
            wide2.scalar = in_cast[i + num_threads * 2];
            wide3.scalar = in_cast[i + num_threads * 3];

            // 依次处理 wide1, wide2, wide3
            {
                const IdxT real_i = skip_cnt + (i + num_threads * 1) * items_per_scalar;
#pragma unroll
                for(int j = 0; j < items_per_scalar; ++j)
                {
                    f(wide1.array[j], real_i + j);
                }
            }
            {
                const IdxT real_i = skip_cnt + (i + num_threads * 2) * items_per_scalar;
#pragma unroll
                for(int j = 0; j < items_per_scalar; ++j)
                {
                    f(wide2.array[j], real_i + j);
                }
            }
            {
                const IdxT real_i = skip_cnt + (i + num_threads * 3) * items_per_scalar;
#pragma unroll
                for(int j = 0; j < items_per_scalar; ++j)
                {
                    f(wide3.array[j], real_i + j);
                }
            }
        }
        // ---- 尾部循环：处理不足 4 个 wide 的剩余部分 ----
        for(; i < len_cast; i += num_threads)
        {
            wide0.scalar = in_cast[i];
            const IdxT real_i = skip_cnt + i * items_per_scalar;
#pragma unroll
            for(int j = 0; j < items_per_scalar; ++j)
            {
                f(wide0.array[j], real_i + j);
            }
        }

        // ---- 处理开头未对齐的元素（标量加载）----
        static_assert(WARP_SIZE >= items_per_scalar);
        if(thread_rank < skip_cnt)
        {
            f(in[thread_rank], thread_rank);
        }
        // ---- 处理结尾不足一个 WideT 的元素（标量加载）----
        const IdxT remain_i = skip_cnt + len_cast * items_per_scalar + thread_rank;
        if(remain_i < len)
        {
            f(in[remain_i], remain_i);
        }
    }
}

// ============================================================================
// Counter: 跨 pass 传递状态的共享结构（与 baseline 相同，无改动）
// ============================================================================
// k:              当前还需要找的元素数（每个 pass 后更新：k = k - 前面桶的元素总数）
// len:            当前 kth 桶中的元素数
// previous_len:   上一个 pass 的 len（用于 compact 路径决定读取源）
// kth_value_bits: 逐步拼接的第 K 大元素的 twiddle 后比特值（每 pass 填入新的位段）
// filter_cnt:     compact 写入计数器
// finished_block_cnt: 多 block 路径的同步计数器
// out_cnt:        输出中"确定大于 kth"的元素写入位置
// out_back_cnt:   输出中"等于 kth"的元素从末尾反向写入位置
// 各字段 alignas(128) 避免 false sharing
template <typename T, typename IdxT>
struct alignas(128) Counter
{
    IdxT k;
    IdxT len;
    IdxT previous_len;
    typename hipcub::Traits<T>::UnsignedBits kth_value_bits;
    alignas(128) IdxT filter_cnt;
    alignas(128) unsigned int finished_block_cnt;
    alignas(128) IdxT out_cnt;
    alignas(128) IdxT out_back_cnt;
};

// ============================================================================
// filter_and_histogram: 多 block 路径的直方图构建 + 过滤
// ============================================================================
// 多 block 路径中，每个 block 先在 shared memory 中建局部直方图，
// 然后 flush 到 global memory 的 histogram。
// pass=0: 只建直方图
// pass>0: 同时过滤元素（compact 到 out_buf 或直接写入 out）
// 与 baseline 类似，但 lambda 签名经过 OPT3 简化。
// 注意：opt5 的单 block 路径（main 中使用的路径）不调用此函数，
//       此函数仅供 standalone_stable_radix_topk_ 的多 block 路径使用。
template <typename T, typename IdxT, int BitsPerPass, bool WRITE_TOPK_VALUES>
__device__ void filter_and_histogram(T const* in_buf,
                                     IdxT const* in_idx_buf,
                                     T* out_buf,
                                     IdxT* out_idx_buf,
                                     T* out,
                                     IdxT* out_idx,
                                     IdxT previous_len,
                                     Counter<T, IdxT>* counter,
                                     IdxT* histogram,
                                     bool select_min,
                                     int pass,
                                     bool early_stop,
                                     IdxT k)
{
    constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
    __shared__ IdxT histogram_smem[num_buckets];
    for(IdxT i = threadIdx.x; i < num_buckets; i += blockDim.x)
    {
        histogram_smem[i] = 0;
    }
    __syncthreads();

    int const start_bit = calc_start_bit<T, BitsPerPass>(pass);
    unsigned const mask = calc_mask<T, BitsPerPass>(pass);

    if(pass == 0)
    {
        auto f = [select_min, start_bit, mask](T value, IdxT, int&, int&, bool) {
            int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, select_min);
            atomicAdd(histogram_smem + bucket, static_cast<IdxT>(1));
        };
        vectorized_process(static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x,
                           static_cast<size_t>(blockDim.x) * gridDim.x,
                           in_buf,
                           previous_len,
                           f);
    }
    else
    {
        IdxT* p_filter_cnt           = &counter->filter_cnt;
        IdxT* p_out_cnt              = &counter->out_cnt;
        auto const kth_value_bits    = counter->kth_value_bits;
        int const previous_start_bit = calc_start_bit<T, BitsPerPass>(pass - 1);

        auto f = [in_idx_buf,
                  out_buf,
                  out_idx_buf,
                  out,
                  out_idx,
                  select_min,
                  start_bit,
                  mask,
                  previous_start_bit,
                  kth_value_bits,
                  p_filter_cnt,
                  p_out_cnt,
                  early_stop](T value, IdxT i, int&, int&, bool) {
            const auto previous_bits = (twiddle_in(value, select_min) >> previous_start_bit)
                                       << previous_start_bit;
            if(previous_bits == kth_value_bits)
            {
                if(early_stop)
                {
                    IdxT pos = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
                    if(WRITE_TOPK_VALUES)
                    {
                        out[pos] = value;
                    }
                    out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
                }
                else
                {
                    if(out_buf)
                    {
                        IdxT pos         = atomicAdd(p_filter_cnt, static_cast<IdxT>(1));
                        out_buf[pos]     = value;
                        out_idx_buf[pos] = in_idx_buf ? in_idx_buf[i] : i;
                    }
                    int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, select_min);
                    atomicAdd(histogram_smem + bucket, static_cast<IdxT>(1));
                }
            }
            else if((out_buf || early_stop) && previous_bits < kth_value_bits)
            {
                IdxT pos = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
                if(WRITE_TOPK_VALUES)
                {
                    out[pos] = value;
                }
                out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
            }
        };
        vectorized_process(static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x,
                           static_cast<size_t>(blockDim.x) * gridDim.x,
                           in_buf,
                           previous_len,
                           f);
    }
    if(early_stop)
    {
        return;
    }
    __syncthreads();

    for(int i = threadIdx.x; i < num_buckets; i += blockDim.x)
    {
        if(histogram_smem[i] != 0)
        {
            atomicAdd(histogram + i, histogram_smem[i]);
        }
    }
}

// ============================================================================
// scan: 对直方图做前缀和（inclusive sum）
// ============================================================================
// 前缀和后，histogram[i] = 桶 0..i 的元素总数。
// 用于 choose_bucket 中确定第 K 大元素落在哪个桶。
//
// 使用 hipcub::BlockScan 实现高效的 warp-shuffle 前缀和。
// 当 num_buckets >= BlockSize 时（如 2048 >= 1024），
// 每个线程处理 items_per_thread 个桶，使用 BlockLoad/BlockStore 转置布局。
// 与 baseline 相同，无改动。
template <typename IdxT, int BitsPerPass, int BlockSize>
__device__ void scan(IdxT volatile* histogram)
{
    constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
    if constexpr(num_buckets >= BlockSize)
    {
        static_assert(num_buckets % BlockSize == 0);
        constexpr int items_per_thread = num_buckets / BlockSize;
        typedef hipcub::BlockLoad<IdxT, BlockSize, items_per_thread, hipcub::BLOCK_LOAD_TRANSPOSE>
            BlockLoad;
        typedef hipcub::BlockStore<IdxT, BlockSize, items_per_thread, hipcub::BLOCK_STORE_TRANSPOSE>
            BlockStore;
        typedef hipcub::BlockScan<IdxT, BlockSize> BlockScan;

        __shared__ union
        {
            typename BlockLoad::TempStorage load;
            typename BlockScan::TempStorage scan;
            typename BlockStore::TempStorage store;
        } temp_storage;

        IdxT thread_data[items_per_thread];

        BlockLoad(temp_storage.load).Load(histogram, thread_data);
        __syncthreads();

        BlockScan(temp_storage.scan).InclusiveSum(thread_data, thread_data);
        __syncthreads();

        BlockStore(temp_storage.store).Store(histogram, thread_data);
    }
    else
    {
        typedef hipcub::BlockScan<IdxT, BlockSize> BlockScan;
        __shared__ typename BlockScan::TempStorage temp_storage;

        IdxT thread_data = 0;
        if(threadIdx.x < num_buckets)
        {
            thread_data = histogram[threadIdx.x];
        }

        BlockScan(temp_storage).InclusiveSum(thread_data, thread_data);
        __syncthreads();

        if(threadIdx.x < num_buckets)
        {
            histogram[threadIdx.x] = thread_data;
        }
    }
}

// ============================================================================
// choose_bucket: 根据前缀和确定第 K 大元素所在的桶
// ============================================================================
// 在前缀和后的直方图中，找到满足 histogram[i-1] < k <= histogram[i] 的桶 i。
// 该桶包含第 K 大元素。更新 counter:
//   counter->k   = 还需要从该桶中选取的元素数
//   counter->len = 该桶的元素总数
//   counter->kth_value_bits |= bucket << start_bit  (拼接该 pass 确定的位段)
// 与 baseline 相同，无改动。
template <typename T, typename IdxT, int BitsPerPass>
__device__ void
choose_bucket(Counter<T, IdxT>* counter, IdxT const* histogram, const IdxT k, int const pass)
{
    constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
    for(int i = threadIdx.x; i < num_buckets; i += blockDim.x)
    {
        IdxT prev = (i == 0) ? 0 : histogram[i - 1];
        IdxT cur  = histogram[i];
        if(prev < k && cur >= k)
        {
            counter->k   = k - prev;
            counter->len = cur - prev;
            typename hipcub::Traits<T>::UnsignedBits bucket = i;
            int start_bit                                   = calc_start_bit<T, BitsPerPass>(pass);
            counter->kth_value_bits |= bucket << start_bit;
        }
    }
}

// ============================================================================
// last_filter: 最后一个 pass 完成后，将结果写入输出数组
// ============================================================================
// 经过所有 pass 之后，kth_value_bits 包含了第 K 大元素的完整 twiddle 值。
// 遍历输入数据：
//   bits < kth_value_bits  → 确定是 top-K 元素，写入 out[out_cnt++]（从前往后）
//   bits == kth_value_bits → 可能是第 K 大，写入 out[k-1-back_cnt]（从后往前填充）
//   bits > kth_value_bits  → 不是 top-K，跳过
//
// [OPT5-B 改动] baseline 的 else 分支 (!in_idx_buf) 使用简单的标量循环:
//   for(IdxT i = threadIdx.x; i < current_len; i += blockDim.x)
//   每次只加载 1 个 float (global_load_dword, 32位)。
//   opt5 改为调用 vectorized_process，使用 global_load_dwordx4 (128位) 宽加载，
//   每次加载 4 个 float，带宽利用率提升 4 倍。
//
// in_idx_buf 路径保持 8x 标量展开（compact buffer 场景需要同时读 in_buf 和 in_idx_buf
// 两个数组，无法用同一个 vectorized_process 同时宽加载两个数组）。
template <typename T,
          typename IdxT,
          int BitsPerPass,
          bool WRITE_TOPK_VALUES,
          bool prioritize_smaller_indice = false>
__device__ void last_filter(T const* in_buf,
                            IdxT const* in_idx_buf,
                            T* out,
                            IdxT* out_idx,
                            IdxT current_len,
                            IdxT k,
                            Counter<T, IdxT>* counter,
                            bool const select_min,
                            int const pass,
                            bool const use_one_pass = false)
{
    auto const kth_value_bits = counter->kth_value_bits;
    int const start_bit       = calc_start_bit<T, BitsPerPass>(pass);
    const IdxT num_of_kth_needed = counter->k;
    IdxT* p_out_cnt              = &counter->out_cnt;
    IdxT* p_out_back_cnt         = &counter->out_back_cnt;

    // 单元素处理逻辑（内联 lambda，被下面的两个路径共用）
    auto process_one = [&](T value, IdxT idx) {
        // 计算当前元素截至当前 pass 的累积 bits
        auto const bits = use_one_pass
                              ? twiddle_in(value, select_min) & ((1 << BitsPerPass) - 1)
                              : (twiddle_in(value, select_min) >> start_bit) << start_bit;
        if(bits < kth_value_bits)
        {
            // 严格大于第 K 大，直接写入输出前部
            IdxT pos = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
            if(WRITE_TOPK_VALUES) { out[pos] = value; }
            out_idx[pos] = idx;
        }
        else if(bits == kth_value_bits)
        {
            // 等于第 K 大，从输出末尾往前填充（最多填 num_of_kth_needed 个）
            IdxT back_pos = atomicAdd(p_out_back_cnt, static_cast<IdxT>(1));
            if(back_pos < num_of_kth_needed)
            {
                IdxT pos = k - 1 - back_pos;
                if(WRITE_TOPK_VALUES) { out[pos] = value; }
                if constexpr(!prioritize_smaller_indice) { out_idx[pos] = idx; }
            }
        }
    };

    if(in_idx_buf)
    {
        // 有 in_idx_buf 的路径（compact 场景）：8x 标量展开
        // 需要同时读取 in_buf[i] 和 in_idx_buf[i]，无法用 vectorized_process
        const IdxT stride = blockDim.x;
        const IdxT strideN = stride * 8;
        IdxT i = threadIdx.x;
        for(; i + stride * 7 < current_len; i += strideN)
        {
            T v0 = in_buf[i + stride * 0];
            T v1 = in_buf[i + stride * 1];
            T v2 = in_buf[i + stride * 2];
            T v3 = in_buf[i + stride * 3];
            T v4 = in_buf[i + stride * 4];
            T v5 = in_buf[i + stride * 5];
            T v6 = in_buf[i + stride * 6];
            T v7 = in_buf[i + stride * 7];
            IdxT idx0 = in_idx_buf[i + stride * 0];
            IdxT idx1 = in_idx_buf[i + stride * 1];
            IdxT idx2 = in_idx_buf[i + stride * 2];
            IdxT idx3 = in_idx_buf[i + stride * 3];
            IdxT idx4 = in_idx_buf[i + stride * 4];
            IdxT idx5 = in_idx_buf[i + stride * 5];
            IdxT idx6 = in_idx_buf[i + stride * 6];
            IdxT idx7 = in_idx_buf[i + stride * 7];
            process_one(v0, idx0);
            process_one(v1, idx1);
            process_one(v2, idx2);
            process_one(v3, idx3);
            process_one(v4, idx4);
            process_one(v5, idx5);
            process_one(v6, idx6);
            process_one(v7, idx7);
        }
        for(; i < current_len; i += stride)
        {
            process_one(in_buf[i], in_idx_buf[i]);
        }
    }
    else
    {
        // [OPT5-B 改动] 无 in_idx_buf 的路径：使用 vectorized_process (dwordx4 宽加载)
        // baseline: for(i += blockDim.x) { process_one(in_buf[i], i); }  — 标量加载
        // opt5:     vectorized_process → global_load_dwordx4 (128位) + 4x展开
        vectorized_process(threadIdx.x, blockDim.x, in_buf, current_len,
            [&](T value, IdxT i) { process_one(value, i); });
    }
}

// ============================================================================
// last_filter_kernel: 多 block 路径的最终过滤 kernel
// ============================================================================
// 作为单独的 kernel launch，在所有 radix pass 完成后调用。
// 注意：opt5 的单 block 路径不使用此 kernel（last_filter 在同一个 kernel 内调用）。
template <typename T,
          typename IdxT,
          int BitsPerPass,
          bool WRITE_TOPK_VALUES,
          Phase phase,
          bool prioritize_smaller_indice = false>
__global__ void last_filter_kernel(T const* in,
                                   IdxT const* in_idx,
                                   T const* in_buf,
                                   IdxT const* in_idx_buf,
                                   T* out,
                                   IdxT* out_idx,
                                   IdxT len,
                                   const IdxT* rowStarts,
                                   const IdxT* rowEnds,
                                   IdxT k,
                                   IdxT next_n,
                                   Counter<T, IdxT>* counters,
                                   bool const select_min)
{
    const int64_t batch_id = blockIdx.y;
    const IdxT row_len     = phase == Phase::Prefill
                                 ? rowEnds[batch_id] - rowStarts[batch_id]
                                 : rowEnds[batch_id / next_n] - next_n + (batch_id % next_n) + 1;

    Counter<T, IdxT>* counter = counters + batch_id;
    IdxT previous_len         = counter->previous_len;
    if(previous_len == 0)
    {
        return;
    }
    const IdxT buf_len = calc_buf_len<T>(len);
    if(previous_len > buf_len || in_buf == in)
    {
        in_buf       = in + batch_id * len;
        in_idx_buf   = in_idx ? (in_idx + batch_id * len) : nullptr;
        previous_len = row_len;
    }
    else
    {
        in_buf += batch_id * buf_len;
        in_idx_buf += batch_id * buf_len;
    }
    out += batch_id * k;
    out_idx += batch_id * k;

    constexpr int pass      = calc_num_passes<T, BitsPerPass>() - 1;
    constexpr int start_bit = calc_start_bit<T, BitsPerPass>(pass);

    auto const kth_value_bits    = counter->kth_value_bits;
    const IdxT num_of_kth_needed = counter->k;
    IdxT* p_out_cnt              = &counter->out_cnt;
    IdxT* p_out_back_cnt         = &counter->out_back_cnt;
    IdxT* p_equal                = out_idx + k - num_of_kth_needed;
    (void)p_equal;

    auto f = [k,
              select_min,
              kth_value_bits,
              num_of_kth_needed,
              p_out_cnt,
              p_out_back_cnt,
              in_idx_buf,
              out,
              out_idx](T value, IdxT i, int&, int&, bool) {
        const auto bits = (twiddle_in(value, select_min) >> start_bit) << start_bit;
        if(bits < kth_value_bits)
        {
            IdxT pos = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
            if(WRITE_TOPK_VALUES)
            {
                out[pos] = value;
            }
            out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
        }
        else if(bits == kth_value_bits)
        {
            IdxT new_idx  = in_idx_buf ? in_idx_buf[i] : i;
            IdxT back_pos = atomicAdd(p_out_back_cnt, static_cast<IdxT>(1));
            if(back_pos < num_of_kth_needed)
            {
                IdxT pos = k - 1 - back_pos;
                if(WRITE_TOPK_VALUES)
                {
                    out[pos] = value;
                }
                if constexpr(!prioritize_smaller_indice)
                {
                    out_idx[pos] = new_idx;
                }
            }
        }
    };

    vectorized_process(static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x,
                       static_cast<size_t>(blockDim.x) * gridDim.x,
                       in_buf,
                       previous_len,
                       f);
}

// ============================================================================
// radix_kernel: 多 block 路径的 radix pass kernel
// ============================================================================
// 多个 block 并行构建直方图（先写 shared，再 flush 到 global）。
// 最后一个完成的 block 通过 atomicInc 选举，执行 scan + choose_bucket。
// 注意：opt5 的单 block 路径不使用此 kernel。
template <typename T,
          typename IdxT,
          int BitsPerPass,
          int BlockSize,
          bool fused_last_filter,
          bool WRITE_TOPK_VALUES,
          bool prioritize_smaller_indice = false,
          Phase phase                    = Phase::Prefill>
__global__ void radix_kernel(T const* in,
                             IdxT const* in_idx,
                             T const* in_buf,
                             IdxT const* in_idx_buf,
                             T* out_buf,
                             IdxT* out_idx_buf,
                             T* out,
                             IdxT* out_idx,
                             Counter<T, IdxT>* counters,
                             IdxT* histograms,
                             const IdxT len,
                             const IdxT* rowStarts,
                             const IdxT* rowEnds,
                             const IdxT k,
                             const IdxT next_n,
                             bool const select_min,
                             int const pass)
{
    const int64_t batch_id = blockIdx.y;

    IdxT row_len = len;
    if(phase == Phase::Prefill)
    {
        if(rowStarts && rowEnds)
        {
            row_len = rowEnds[batch_id] - rowStarts[batch_id];
        }
    }
    else
    {
        row_len = rowEnds[batch_id / next_n] - next_n + (batch_id % next_n) + 1;
    }

    auto counter = counters + batch_id;
    IdxT current_k;
    IdxT previous_len;
    IdxT current_len;
    if(pass == 0)
    {
        current_k    = k;
        previous_len = row_len;
        current_len  = row_len;
    }
    else
    {
        current_k    = counter->k;
        current_len  = counter->len;
        previous_len = counter->previous_len;
    }
    if(current_len == 0)
    {
        return;
    }

    bool const early_stop = (current_len == current_k);
    const IdxT buf_len    = calc_buf_len<T>(len);

    if(pass == 0 || pass == 1 || previous_len > buf_len)
    {
        in_buf       = in + batch_id * len;
        in_idx_buf   = in_idx ? (in_idx + batch_id * len) : nullptr;
        previous_len = row_len;
    }
    else
    {
        in_buf += batch_id * buf_len;
        in_idx_buf += batch_id * buf_len;
    }
    if(pass == 0 || current_len > buf_len)
    {
        out_buf     = nullptr;
        out_idx_buf = nullptr;
    }
    else
    {
        out_buf += batch_id * buf_len;
        out_idx_buf += batch_id * buf_len;
    }
    out += batch_id * k;
    out_idx += batch_id * k;

    constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
    auto histogram            = histograms + batch_id * num_buckets;

    filter_and_histogram<T, IdxT, BitsPerPass, WRITE_TOPK_VALUES>(in_buf,
                                                                  in_idx_buf,
                                                                  out_buf,
                                                                  out_idx_buf,
                                                                  out,
                                                                  out_idx,
                                                                  previous_len,
                                                                  counter,
                                                                  histogram,
                                                                  select_min,
                                                                  pass,
                                                                  early_stop,
                                                                  k);
    __threadfence();

    bool isLastBlock = false;
    if(threadIdx.x == 0)
    {
        unsigned int finished = atomicInc(&counter->finished_block_cnt, gridDim.x - 1);
        isLastBlock           = (finished == (gridDim.x - 1));
    }

    if(__syncthreads_or(isLastBlock))
    {
        if(early_stop)
        {
            if(threadIdx.x == 0)
            {
                counter->previous_len = 0;
                counter->len          = 0;
            }
            return;
        }

        scan<IdxT, BitsPerPass, BlockSize>(histogram);
        __syncthreads();
        choose_bucket<T, IdxT, BitsPerPass>(counter, histogram, current_k, pass);
        __syncthreads();

        constexpr int num_passes = calc_num_passes<T, BitsPerPass>();
        if(pass != num_passes - 1)
        {
            for(int i = threadIdx.x; i < num_buckets; i += blockDim.x)
            {
                histogram[i] = 0;
            }
        }
        if(threadIdx.x == 0)
        {
            counter->previous_len = current_len;
            counter->filter_cnt   = 0;
        }

        if(pass == num_passes - 1)
        {
            const volatile IdxT num_of_kth_needed = counter->k;
            for(IdxT i = threadIdx.x; i < num_of_kth_needed; i += blockDim.x)
            {
                out_idx[k - num_of_kth_needed + i] = std::numeric_limits<IdxT>::max();
            }
            __syncthreads();
            if constexpr(fused_last_filter)
            {
                last_filter<T, IdxT, BitsPerPass, WRITE_TOPK_VALUES, prioritize_smaller_indice>(
                    out_buf ? out_buf : in_buf,
                    out_idx_buf ? out_idx_buf : in_idx_buf,
                    out,
                    out_idx,
                    out_buf ? current_len : row_len,
                    k,
                    counter,
                    select_min,
                    pass);
            }
        }
    }
}

// 计算多 block 路径的 grid 维度（最小化 tail wave penalty）
// 与 baseline 相同，无改动。单 block 路径不使用此函数。
template <typename T,
          typename IdxT,
          int BitsPerPass,
          int BlockSize,
          bool WRITE_TOPK_VALUES,
          Phase phase = Phase::Prefill>
unsigned calc_grid_dim(int batch_size, IdxT len, int sm_cnt)
{
    static_assert(VECTORIZED_READ_SIZE / sizeof(T) >= 1);

    int active_blocks;
    HIP_CALL(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks,
        radix_kernel<T, IdxT, BitsPerPass, BlockSize, false, WRITE_TOPK_VALUES, false, phase>,
        BlockSize,
        0));
    active_blocks *= sm_cnt;

    IdxT best_num_blocks         = 0;
    float best_tail_wave_penalty = 1.0f;
    const IdxT max_num_blocks    = ceildiv<IdxT>(len, VECTORIZED_READ_SIZE / sizeof(T) * BlockSize);
    for(int num_waves = 1;; ++num_waves)
    {
        IdxT num_blocks = std::min(
            max_num_blocks, static_cast<IdxT>(std::max(num_waves * active_blocks / batch_size, 1)));
        IdxT items_per_thread  = ceildiv<IdxT>(len, num_blocks * BlockSize);
        items_per_thread       = alignTo<IdxT>(items_per_thread, VECTORIZED_READ_SIZE / sizeof(T));
        num_blocks             = ceildiv<IdxT>(len, items_per_thread * BlockSize);
        float actual_num_waves = static_cast<float>(num_blocks) * batch_size / active_blocks;
        float tail_wave_penalty =
            (ceilf(actual_num_waves) - actual_num_waves) / ceilf(actual_num_waves);

        if(tail_wave_penalty < 0.15)
        {
            best_num_blocks = num_blocks;
            break;
        }
        else if(tail_wave_penalty < best_tail_wave_penalty)
        {
            best_num_blocks        = num_blocks;
            best_tail_wave_penalty = tail_wave_penalty;
        }

        if(num_blocks == max_num_blocks)
        {
            break;
        }
    }
    return best_num_blocks;
}

// ============================================================================
// set_buf_pointers: 设置乒乓 buffer 指针（compact 路径使用）
// ============================================================================
// 多 block 路径版本：使用两组独立的 buffer (buf1/buf2) 交替读写。
// pass 0: 从原始输入读，不写 compact buffer
// pass 1: 从原始输入读，写入 buf1
// pass 2: 从 buf1 读，写入 buf2
// pass 3: 从 buf2 读，写入 buf1
// ...依次交替
// 与 baseline 相同，无改动。
template <typename T, typename IdxT>
__host__ __device__ void set_buf_pointers(T const* in,
                                          IdxT const* in_idx,
                                          T* buf1,
                                          IdxT* idx_buf1,
                                          T* buf2,
                                          IdxT* idx_buf2,
                                          int pass,
                                          T const*& in_buf,
                                          IdxT const*& in_idx_buf,
                                          T*& out_buf,
                                          IdxT*& out_idx_buf)
{
    if(pass == 0)
    {
        in_buf      = in;
        in_idx_buf  = nullptr;
        out_buf     = nullptr;
        out_idx_buf = nullptr;
    }
    else if(pass == 1)
    {
        in_buf      = in;
        in_idx_buf  = in_idx;
        out_buf     = buf1;
        out_idx_buf = idx_buf1;
    }
    else if(pass % 2 == 0)
    {
        in_buf      = buf1;
        in_idx_buf  = idx_buf1;
        out_buf     = buf2;
        out_idx_buf = idx_buf2;
    }
    else
    {
        in_buf      = buf2;
        in_idx_buf  = idx_buf2;
        out_buf     = buf1;
        out_idx_buf = idx_buf1;
    }
}

// 单 block 路径版本的 set_buf_pointers：使用 char* bufs 指向连续内存块，
// 内部划分为两组 buffer。逻辑同上。
// 注意：在 opt5 中，单 block 路径禁用了 compact，此函数不被调用，
//       仅保留供代码完整性和多 block 路径兼容。
// 与 baseline 相同，无改动。
template <typename T, typename IdxT>
__device__ void set_buf_pointers(T const* in,
                                 IdxT const* in_idx,
                                 char* bufs,
                                 IdxT buf_len,
                                 int pass,
                                 T const*& in_buf,
                                 IdxT const*& in_idx_buf,
                                 T*& out_buf,
                                 IdxT*& out_idx_buf)
{
    if(pass == 0)
    {
        in_buf      = in;
        in_idx_buf  = nullptr;
        out_buf     = nullptr;
        out_idx_buf = nullptr;
    }
    else if(pass == 1)
    {
        in_buf      = in;
        in_idx_buf  = in_idx;
        out_buf     = reinterpret_cast<T*>(bufs);
        out_idx_buf = reinterpret_cast<IdxT*>(bufs + sizeof(T) * 2 * buf_len);
    }
    else if(pass % 2 == 0)
    {
        in_buf      = reinterpret_cast<T*>(bufs);
        in_idx_buf  = reinterpret_cast<IdxT*>(bufs + sizeof(T) * 2 * buf_len);
        out_buf     = const_cast<T*>(in_buf + buf_len);
        out_idx_buf = const_cast<IdxT*>(in_idx_buf + buf_len);
    }
    else
    {
        out_buf     = reinterpret_cast<T*>(bufs);
        out_idx_buf = reinterpret_cast<IdxT*>(bufs + sizeof(T) * 2 * buf_len);
        in_buf      = out_buf + buf_len;
        in_idx_buf  = out_idx_buf + buf_len;
    }
}

// ============================================================================
// filter_and_histogram_for_one_block: 单 block 路径的核心函数
// ============================================================================
// 对输入数据构建直方图，同时（可选地）将匹配元素 compact 到 out_buf。
// 这是每个 pass 中计算量最大的部分，也是优化的重点。
//
// 三个分支：
//   pass==0:   首次遍历，只构建直方图
//   !out_buf:  后续 pass 但 compact 被禁用（opt5 单 block 路径走此分支）
//   out_buf:   后续 pass 且启用 compact（opt5 中不使用）
//
// ======================== 与 baseline 的主要差异 ========================
//
// baseline 的 pass==0:
//   lambda 签名: f(value, idx, acc, prev_bin_idx, is_last)
//   使用 batching 优化：连续相同桶的元素累加后一次性 atomicAdd。
//   还额外计算 histogram[num_buckets + bucket_low] 和 local_min/max 用于 one_pass 检测。
// [OPT3 改动] opt5 的 pass==0:
//   lambda 签名简化为: f(value, idx)
//   移除了 batching（对 2048 桶，连续命中概率仅 1/2048，batch 收益忽略不计）。
//   移除了 one_pass 检测（对 float32 数据范围 [-1e9, 1e9]，one_pass 不可能触发）。
//   移除了 histogram 的 *2 分配（baseline 分配 num_buckets*2 给 one_pass 的第二组直方图）。
//   结果：pass==0 的 histogram 只需 num_buckets 个 slot。
//
// baseline 的 pass>0 且 !out_buf:
//   使用标量 for 循环: for(i += blockDim.x) { if(prefix==kth) histogram[bucket]++ }
//   每个元素调用两次 twiddle_in：一次算 prefix bits，一次在 calc_bucket 中算 bucket。
//   if 分支导致编译器生成 s_and_saveexec / s_or_b64 (exec mask save/restore)。
// [OPT5-A 改动] opt5 的 pass>0 且 !out_buf:
//   1. 改用 vectorized_process 进行宽加载 (dwordx4)
//   2. 只调用一次 twiddle_in，得到 bits 后分别提取 prefix 和 bucket：
//      pb = (bits >> prev_start_bit) << prev_start_bit  (prefix)
//      bucket = __builtin_amdgcn_ubfe(bits, start_bit, BitsPerPass)  (v_bfe_u32)
//   3. 仍保留 if(pb == kth_value_bits) 条件分支（尝试过 dummy bucket 方案，
//      但编译器已经很好地处理了这个分支，dummy bucket 的额外 LDS 写入反而更慢）
//
// baseline 的 pass>0 且 out_buf:
//   使用标量 for 循环读取。
// [OPT5 改动] opt5 的 pass>0 且 out_buf:
//   改用 vectorized_process 宽加载，且使用 __builtin_amdgcn_ubfe 提取 bucket。
//   （但在 opt5 单 block 路径中 out_buf 始终为 nullptr，此分支实际不执行。）
//
template <typename T, typename IdxT, int BitsPerPass, bool WRITE_TOPK_VALUES, int BlockSize>
__device__ bool filter_and_histogram_for_one_block(T const* in_buf,
                                                   IdxT const* in_idx_buf,
                                                   T* out_buf,
                                                   IdxT* out_idx_buf,
                                                   T* out,
                                                   IdxT* out_idx,
                                                   const IdxT previous_len,
                                                   Counter<T, IdxT>* counter,
                                                   IdxT* histogram,
                                                   bool select_min,
                                                   int pass,
                                                   IdxT k)
{
    constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
    // [OPT3 改动] baseline 初始化 num_buckets*2 (给 one_pass 的第二组直方图)
    // opt5 只初始化 num_buckets 个 slot（one_pass 功能被移除）
    for(int i = threadIdx.x; i < num_buckets; i += blockDim.x)
    {
        histogram[i] = 0;
    }
    IdxT* p_filter_cnt = &counter->filter_cnt;
    if(threadIdx.x == 0)
    {
        *p_filter_cnt = 0;
    }
    __syncthreads();

    int const start_bit = calc_start_bit<T, BitsPerPass>(pass);
    unsigned const mask = calc_mask<T, BitsPerPass>(pass);

    if(pass == 0)
    {
        // ---- Pass 0: 仅构建直方图 ----
        // [OPT3 改动] baseline 的 lambda 有 5 个参数 (value, idx, acc, prev_bin_idx, is_last)
        //   用于 batch 累加优化。opt5 简化为 2 个参数 (value, idx)，直接 atomicAdd。
        // [OPT3 改动] baseline 还在 pass 0 中计算 one_pass 检测（local_min/max + BlockReduce）
        //   opt5 移除了 one_pass，所以返回 false 而不检测。
        auto f = [histogram, select_min, start_bit, mask](T value, IdxT) {
            int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, select_min);
            atomicAdd(histogram + bucket, static_cast<IdxT>(1));
        };
        vectorized_process(threadIdx.x, blockDim.x, in_buf, previous_len, f);

        return false;  // baseline 此处可能返回 use_one_pass=true
    }
    else if(!out_buf)
    {
        // ---- Pass > 0, compact 禁用（opt5 单 block 路径走这里）----
        // [OPT5-A 改动] baseline 使用标量 for 循环:
        //   for(IdxT i = threadIdx.x; i < previous_len; i += blockDim.x) {
        //       T value = in_buf[i];
        //       auto pb = (twiddle_in(value, select_min) >> prev_start_bit) << prev_start_bit;
        //       if(pb == kth_value_bits) {
        //           int bucket = calc_bucket<T, BitsPerPass>(value, ...);  // 第二次 twiddle_in
        //           atomicAdd(histogram + bucket, 1);
        //       }
        //   }
        //   问题1: 标量循环用 global_load_dword (32位)，带宽利用率低
        //   问题2: calc_bucket 内部再次调用 twiddle_in，重复计算
        //
        //   opt5 改为:
        //   1. 用 vectorized_process 做 dwordx4 宽加载 (128位)
        //   2. 只调用一次 twiddle_in(value)，得到 bits
        //   3. 从 bits 分别计算 pb (prefix) 和 bucket (v_bfe_u32)
        auto const kth_value_bits    = counter->kth_value_bits;
        int const previous_start_bit = calc_start_bit<T, BitsPerPass>(pass - 1);

        auto hist_vec = [histogram, select_min, start_bit, mask,
                         kth_value_bits, previous_start_bit](T value, IdxT) {
            // 只做一次 twiddle_in（baseline 做了两次）
            auto const bits = twiddle_in(value, select_min);
            // 提取上一个 pass 累积的前缀 bits
            auto const pb = (bits >> previous_start_bit) << previous_start_bit;
            // 用 v_bfe_u32 单指令提取当前 pass 的桶号（baseline 用 shift+mask 两条指令）
            int bucket = __builtin_amdgcn_ubfe(bits, static_cast<unsigned>(start_bit), static_cast<unsigned>(BitsPerPass));
            // 只有前缀匹配 kth 的元素才写入直方图
            if(pb == kth_value_bits)
            {
                atomicAdd(histogram + bucket, static_cast<IdxT>(1));
            }
        };
        vectorized_process(threadIdx.x, blockDim.x, in_buf, previous_len, hist_vec);
    }
    else
    {
        // ---- Pass > 0, compact 启用（opt5 单 block 路径不走这里，保留供兼容）----
        // [OPT5 改动] baseline 使用标量循环；opt5 改用 vectorized_process + ubfe
        IdxT* p_out_cnt              = &counter->out_cnt;
        auto const kth_value_bits    = counter->kth_value_bits;
        int const previous_start_bit = calc_start_bit<T, BitsPerPass>(pass - 1);

        auto process_hist = [histogram, out_buf, out_idx_buf, out, out_idx,
                             in_idx_buf, select_min, start_bit, mask,
                             kth_value_bits, previous_start_bit,
                             p_filter_cnt, p_out_cnt](T value, IdxT idx) {
            auto const bits = twiddle_in(value, select_min);
            auto const pb = (bits >> previous_start_bit) << previous_start_bit;
            if(pb == kth_value_bits)
            {
                // 前缀匹配 kth：compact 到 out_buf 并计入直方图
                IdxT pos         = atomicAdd(p_filter_cnt, static_cast<IdxT>(1));
                out_buf[pos]     = value;
                out_idx_buf[pos] = in_idx_buf ? in_idx_buf[idx] : idx;
                int bucket = __builtin_amdgcn_ubfe(bits, static_cast<unsigned>(start_bit), static_cast<unsigned>(BitsPerPass));
                atomicAdd(histogram + bucket, static_cast<IdxT>(1));
            }
            else if(pb < kth_value_bits)
            {
                // 前缀严格小于 kth：已确认是 top-K，直接写入输出
                IdxT pos = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
                if(WRITE_TOPK_VALUES) { out[pos] = value; }
                out_idx[pos] = in_idx_buf ? in_idx_buf[idx] : idx;
            }
            // 前缀严格大于 kth：不是 top-K，丢弃
        };
        vectorized_process(threadIdx.x, blockDim.x, in_buf, previous_len, process_hist);
    }

    return false;
}

// ============================================================================
// radix_topk_one_block_kernel: 单 block 路径的主 kernel
// ============================================================================
// 这是 opt5 中实际使用的 kernel（在 main 中通过
// standalone_stable_radix_topk_one_block_ 调用）。
//
// 执行流程 (以 BitsPerPass=11, float32 为例, 3 个 pass):
//   Pass 0: 取 bit31..21 → 2048 桶直方图 → 前缀和 → 选择 kth 桶
//   Pass 1: 取 bit20..10 → 2048 桶直方图（仅统计 pass0 匹配的元素）→ 选择 kth 桶
//   Pass 2: 取 bit9..0   → 2048 桶直方图（仅统计 pass0+1 匹配的元素）→ last_filter
//
// ======================== 与 baseline 的主要差异 ========================
//
// [OPT3 改动] histogram 大小: baseline 分配 num_buckets*2 (支持 one_pass);
//   opt5 只分配 num_buckets (one_pass 功能被移除)。
//
// [OPT5-C 改动] Compact 禁用:
//   baseline 的循环中调用 set_buf_pointers 设置乒乓 buffer，
//   pass>0 时将匹配元素 compact 到 buf，后续 pass 从 buf 读取（数据量小）。
//   但对于 11-bit radix + 3 pass 场景:
//     - pass 0 后，kth 桶约有 60000/2048 ≈ 29 个元素
//     - compact 写入 29 个元素的开销（atomicAdd + 写 buf + 写 idx_buf）
//       反而超过了 pass 1-2 直接从原始 60000 个元素中过滤的开销
//       （因为 vectorized_process 的 dwordx4 宽加载效率很高）
//   因此 opt5 禁用 compact: 所有 pass 都传 out_buf=nullptr，
//   每个 pass 都从原始 in 读取全部 row_len 个元素。
//   好处: 始终走 filter_and_histogram_for_one_block 的 !out_buf 路径，
//         该路径使用 vectorized_process (dwordx4 宽加载) + 无 compact 写入开销。
//
//   baseline 对应代码（被替换的部分）:
//     set_buf_pointers(in, in_idx, bufs, buf_len, pass, in_buf, in_idx_buf, out_buf, out_idx_buf);
//     IdxT previous_len = counter.previous_len;
//     if(previous_len > buf_len) { in_buf = in; in_idx_buf = in_idx; previous_len = row_len; }
//     if(current_len > buf_len)  { out_buf = nullptr; out_idx_buf = nullptr; }
//     filter_and_histogram_for_one_block(..., in_buf, in_idx_buf, out_buf, out_idx_buf, previous_len, ...);
//   opt5 简化为:
//     filter_and_histogram_for_one_block(..., in, in_idx, nullptr, nullptr, row_len, ...);
//
template <typename T,
          typename IdxT,
          int BitsPerPass,
          int BlockSize,
          bool WRITE_TOPK_VALUES,
          bool prioritize_smaller_indice = false,
          Phase phase>
__global__ void radix_topk_one_block_kernel(T const* in,
                                            IdxT const* in_idx,
                                            const int64_t len,
                                            const IdxT* rowStarts,
                                            const IdxT* rowEnds,
                                            const IdxT k,
                                            T* out,
                                            IdxT* out_idx,
                                            bool const select_min,
                                            char* bufs,
                                            const int next_n)
{
    constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
    __shared__ Counter<T, IdxT> counter;
    // [OPT3 改动] baseline: __shared__ IdxT histogram[num_buckets * 2]
    // opt5: 只需 num_buckets（移除了 one_pass 检测，不需要第二组直方图）
    __shared__ IdxT histogram[num_buckets];

    // 每个 block 处理一个 batch（一行数据）
    const int64_t batch_id = blockIdx.x;

    // 确定当前行的起止范围
    IdxT rowStart = 0;
    IdxT rowEnd   = len;
    if(phase == Phase::Prefill)
    {
        if(rowStarts && rowEnds)
        {
            rowStart = rowStarts[batch_id];
            rowEnd   = rowEnds[batch_id];
        }
    }
    else
    {
        rowEnd   = rowEnds[batch_id / next_n] - next_n + (batch_id % next_n) + 1;
        rowStart = 0;
    }

    const IdxT row_len = rowEnd - rowStart;

    // 初始化 counter（线程 0 负责）
    if(threadIdx.x == 0)
    {
        counter.k              = k;          // 需要找的 top-K 数量
        counter.len            = row_len;    // kth 桶中的元素数（初始为总长度）
        counter.previous_len   = row_len;    // 上一个 pass 的 len
        counter.kth_value_bits = 0;          // 逐 pass 拼接的 kth 值
        counter.out_cnt        = 0;          // 输出中已写入的"大于 kth"元素数
        counter.out_back_cnt   = 0;          // 输出中已写入的"等于 kth"元素数
    }
    __syncthreads();

    // 偏移到当前 batch 的数据
    in += batch_id * len;
    out += batch_id * k;
    out_idx += batch_id * k;
    if(in_idx)
    {
        in_idx += batch_id * len;
    }

    // 特殊情况：如果数据量 <= k，直接全部输出
    if(row_len <= k)
    {
        for(int rowIt = threadIdx.x; rowIt < k; rowIt += BlockSize)
        {
            out_idx[rowIt] = rowIt < row_len ? rowIt + rowStart : -1;
            if(WRITE_TOPK_VALUES)
            {
                out[rowIt] = rowIt < row_len ? in[rowIt + rowStart] : 0;
            }
        }
        return;
    }

    // compact buffer 相关（虽然 opt5 禁用了 compact，但 workspace 仍然需要分配）
    const IdxT buf_len = calc_buf_len<T, IdxT, unsigned>(len);
    bufs += batch_id * buf_len * 2 * (sizeof(T) + sizeof(IdxT));

    // ======================== 主循环：3 个 radix pass ========================
    // [OPT5-C 改动] baseline 使用 set_buf_pointers + compact 乒乓 buffer:
    //   T const* in_buf; IdxT const* in_idx_buf; T* out_buf; IdxT* out_idx_buf;
    //   set_buf_pointers(in, in_idx, bufs, buf_len, pass, ...);
    //   if(previous_len > buf_len) { in_buf = in; ... }  // 数据太多放不进 buf，回退到原始输入
    //   if(current_len > buf_len)  { out_buf = nullptr; }  // 同上
    //
    // opt5 直接传 in (原始输入) + nullptr (禁用 compact):
    //   filter_and_histogram_for_one_block(in, in_idx, nullptr, nullptr, ..., row_len, ...);
    //   这样每个 pass 都遍历完整的 60000 个元素，但走 vectorized_process 宽加载路径。
    constexpr int num_passes = calc_num_passes<T, BitsPerPass>();
#pragma unroll
    for(int pass = 0; pass < num_passes; ++pass)
    {
        const IdxT current_k = (pass == 0) ? k : counter.k;

        // 构建直方图（+ 可选的 compact）
        filter_and_histogram_for_one_block<T, IdxT, BitsPerPass, WRITE_TOPK_VALUES, BlockSize>(
                in,             // [OPT5-C] baseline 传 in_buf (可能是 compact buffer)
                in_idx,         // [OPT5-C] baseline 传 in_idx_buf
                nullptr,        // [OPT5-C] out_buf 禁用 (baseline 传 out_buf)
                nullptr,        // [OPT5-C] out_idx_buf 禁用 (baseline 传 out_idx_buf)
                out,
                out_idx,
                row_len,        // [OPT5-C] 始终读完整输入 (baseline 传 previous_len，可能是 compact 后的长度)
                &counter,
                histogram,
                select_min,
                pass,
                k);
        __syncthreads();

        // 对直方图做前缀和
        // [OPT3 改动] baseline: scan(histogram + use_one_pass * num_buckets)
        // opt5: 移除 one_pass，直接 scan(histogram)
        scan<IdxT, BitsPerPass, BlockSize>(histogram);
        __syncthreads();

        // 根据前缀和确定 kth 桶，更新 counter
        // [OPT3 改动] baseline: choose_bucket(..., current_k, pass + use_one_pass * num_passes)
        // opt5: 移除 one_pass 偏移
        choose_bucket<T, IdxT, BitsPerPass>(&counter,
                                            histogram,
                                            current_k,
                                            pass);
        if(threadIdx.x == 0)
        {
            counter.previous_len = counter.len;
        }
        __syncthreads();

        // 最后一个 pass 或提前结束：执行 last_filter 写入最终结果
        if(pass == num_passes - 1)
        {
            // [OPT5-C] baseline: last_filter(out_buf ? out_buf : in, ..., out_buf ? current_len : row_len, ...)
            // opt5: compact 禁用，始终从 in 读取 row_len 个元素
            last_filter<T, IdxT, BitsPerPass, WRITE_TOPK_VALUES, prioritize_smaller_indice>(
                in, in_idx, out, out_idx, row_len, k, &counter, select_min, pass, false);
            break;
        }
        else if(counter.len == counter.k)
        {
            // 提前终止：kth 桶中的元素数恰好等于还需要的数量，全部写入输出
            last_filter<T, IdxT, BitsPerPass, WRITE_TOPK_VALUES, false>(
                in, in_idx, out, out_idx, row_len, k, &counter, select_min, pass);
            break;
        }
    }
}

// ============================================================================
// Host 端辅助函数（与 baseline 相同，无改动）
// ============================================================================

// 计算多组数据的总对齐大小（256字节对齐）
inline size_t calc_aligned_size(std::vector<size_t> const& sizes)
{
    const size_t ALIGN_BYTES = 256;
    const size_t ALIGN_MASK  = ~(ALIGN_BYTES - 1);
    size_t total             = 0;
    for(auto sz : sizes)
    {
        total += (sz + ALIGN_BYTES - 1) & ALIGN_MASK;
    }
    return total + ALIGN_BYTES - 1;
}

// 根据基地址和各段大小，计算每段的对齐后指针
inline std::vector<void*> calc_aligned_pointers(void const* p, std::vector<size_t> const& sizes)
{
    const size_t ALIGN_BYTES = 256;
    const size_t ALIGN_MASK  = ~(ALIGN_BYTES - 1);

    char* ptr =
        reinterpret_cast<char*>((reinterpret_cast<size_t>(p) + ALIGN_BYTES - 1) & ALIGN_MASK);

    std::vector<void*> aligned_pointers;
    aligned_pointers.reserve(sizes.size());
    for(auto sz : sizes)
    {
        aligned_pointers.push_back(ptr);
        ptr += (sz + ALIGN_BYTES - 1) & ALIGN_MASK;
    }

    return aligned_pointers;
}

// ============================================================================
// standalone_stable_radix_topk_: 多 block 路径的 Host 入口
// ============================================================================
// 分配 workspace (counters, histograms, 两组 compact buffer)，
// 然后循环调用 radix_kernel（每个 pass 一次 kernel launch）。
// 注意：opt5 的 main 函数不使用此路径，仅使用 standalone_stable_radix_topk_one_block_。
// 与 baseline 类似，无本质改动。
template <typename T,
          typename IdxT,
          int BitsPerPass,
          int BlockSize,
          bool WRITE_TOPK_VALUES,
          Phase phase = Phase::Prefill>
void standalone_stable_radix_topk_(void* buf,
                                   size_t& buf_size,
                                   T const* in,
                                   IdxT const* in_idx,
                                   int batch_size,
                                   int64_t len,
                                   IdxT* rowStarts,
                                   IdxT* rowEnds,
                                   IdxT k,
                                   T* out,
                                   IdxT* out_idx,
                                   bool select_min,
                                   bool fused_last_filter,
                                   unsigned grid_dim,
                                   hipStream_t stream,
                                   bool sorted = false,
                                   int next_n  = 0)
{
    (void)sorted;
    static_assert(calc_num_passes<T, BitsPerPass>() > 1);
    constexpr int num_buckets = calc_num_buckets<BitsPerPass>();

    Counter<T, IdxT>* counters = nullptr;
    IdxT* histograms           = nullptr;
    T* buf1                    = nullptr;
    IdxT* idx_buf1             = nullptr;
    T* buf2                    = nullptr;
    IdxT* idx_buf2             = nullptr;

    {
        IdxT len_candidates       = calc_buf_len<T, IdxT>(len);
        std::vector<size_t> sizes = {sizeof(*counters) * batch_size,
                                     sizeof(*histograms) * num_buckets * batch_size,
                                     sizeof(*buf1) * len_candidates * batch_size,
                                     sizeof(*idx_buf1) * len_candidates * batch_size,
                                     sizeof(*buf2) * len_candidates * batch_size,
                                     sizeof(*idx_buf2) * len_candidates * batch_size};

        size_t total_size = calc_aligned_size(sizes);
        if(!buf)
        {
            buf_size = total_size;
            return;
        }

        std::vector<void*> aligned_pointers = calc_aligned_pointers(buf, sizes);
        counters                            = static_cast<decltype(counters)>(aligned_pointers[0]);
        histograms = static_cast<decltype(histograms)>(aligned_pointers[1]);
        buf1       = static_cast<decltype(buf1)>(aligned_pointers[2]);
        idx_buf1   = static_cast<decltype(idx_buf1)>(aligned_pointers[3]);
        buf2       = static_cast<decltype(buf2)>(aligned_pointers[4]);
        idx_buf2   = static_cast<decltype(idx_buf2)>(aligned_pointers[5]);

        HIP_CALL(hipMemsetAsync(aligned_pointers[0],
                                0,
                                static_cast<char*>(aligned_pointers[2]) -
                                    static_cast<char*>(aligned_pointers[0]),
                                stream));
    }

    T const* in_buf        = nullptr;
    IdxT const* in_idx_buf = nullptr;
    T* out_buf             = nullptr;
    IdxT* out_idx_buf      = nullptr;

    dim3 blocks(grid_dim, batch_size);

    constexpr int num_passes = calc_num_passes<T, BitsPerPass>();

    auto kernel =
        radix_kernel<T, IdxT, BitsPerPass, BlockSize, false, WRITE_TOPK_VALUES, false, phase>;

    for(int pass = 0; pass < num_passes; ++pass)
    {
        set_buf_pointers(in,
                         in_idx,
                         buf1,
                         idx_buf1,
                         buf2,
                         idx_buf2,
                         pass,
                         in_buf,
                         in_idx_buf,
                         out_buf,
                         out_idx_buf);

        if(fused_last_filter && pass == num_passes - 1)
        {
            kernel = radix_kernel<T,
                                  IdxT,
                                  BitsPerPass,
                                  BlockSize,
                                  true,
                                  WRITE_TOPK_VALUES,
                                  false,
                                  phase>;
        }

        kernel<<<blocks, BlockSize, 0, stream>>>(in,
                                                 in_idx,
                                                 in_buf,
                                                 in_idx_buf,
                                                 out_buf,
                                                 out_idx_buf,
                                                 out,
                                                 out_idx,
                                                 counters,
                                                 histograms,
                                                 len,
                                                 rowStarts,
                                                 rowEnds,
                                                 k,
                                                 next_n,
                                                 select_min,
                                                 pass);
    }

    if(!fused_last_filter)
    {
        last_filter_kernel<T, IdxT, BitsPerPass, WRITE_TOPK_VALUES, phase, false>
            <<<blocks, BlockSize, 0, stream>>>(in,
                                               in_idx,
                                               out_buf,
                                               out_idx_buf,
                                               out,
                                               out_idx,
                                               len,
                                               rowStarts,
                                               rowEnds,
                                               k,
                                               next_n,
                                               counters,
                                               select_min);
    }
}

// ============================================================================
// standalone_stable_radix_topk_one_block_: 单 block 路径的 Host 入口
// ============================================================================
// 这是 opt5 中 main 函数实际调用的路径。
// 1. 第一次调用 buf=nullptr: 计算所需 workspace 大小并返回
// 2. 第二次调用 buf=已分配空间: launch kernel
//
// 模板参数（在 main 中的实例化）:
//   T=float, IdxT=int, BitsPerPass=11, BlockSize=1024, WRITE_TOPK_VALUES=false
//
// launch 配置:
//   grid = batch_size (每行一个 block)
//   block = BlockSize = 1024 线程
//
// 与 baseline 的 standalone_stable_radix_topk_one_block_ 结构相同，
// 但调用的是优化后的 radix_topk_one_block_kernel。
template <typename T,
          typename IdxT,
          int BitsPerPass,
          int BlockSize,
          bool WRITE_TOPK_VALUES,
          Phase phase = Phase::Prefill>
void standalone_stable_radix_topk_one_block_(void* buf,
                                             size_t& buf_size,
                                             T const* in,
                                             IdxT const* in_idx,
                                             int batch_size,
                                             int64_t len,
                                             IdxT* rowStarts,
                                             IdxT* rowEnds,
                                             IdxT k,
                                             T* out,
                                             IdxT* out_idx,
                                             bool select_min,
                                             hipStream_t stream,
                                             bool sorted = false,
                                             int next_n  = 0)
{
    (void)sorted;
    static_assert(calc_num_passes<T, BitsPerPass>() > 1);

    char* bufs         = nullptr;
    const IdxT buf_len = calc_buf_len<T, IdxT, unsigned>(len);

    {
        // workspace 大小 = compact buffer 的空间（虽然 opt5 不使用 compact，
        // 但 kernel 参数要求传入 bufs 指针，且 workspace 很小不影响性能）
        std::vector<size_t> sizes = {buf_len * 2 * (sizeof(T) + sizeof(IdxT)) * batch_size};
        size_t total_size         = calc_aligned_size(sizes);
        if(!buf)
        {
            buf_size = total_size;
            return;
        }

        std::vector<void*> aligned_pointers = calc_aligned_pointers(buf, sizes);
        bufs                                = static_cast<decltype(bufs)>(aligned_pointers[0]);
    }

    // Launch: 1 block × 1024 线程（单 block 路径，整个算法在一次 kernel launch 中完成）
    radix_topk_one_block_kernel<T, IdxT, BitsPerPass, BlockSize, WRITE_TOPK_VALUES, false, phase>
        <<<batch_size, BlockSize, 0, stream>>>(
            in, in_idx, len, rowStarts, rowEnds, k, out, out_idx, select_min, bufs, next_n);
}

} // namespace aiter

// ============================================================================
// 测试与 benchmark 代码
// ============================================================================

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <unordered_set>
#include <vector>

namespace {

// 从环境变量读取正整数，不存在则返回默认值
int read_positive_env_or_default(const char* name, int default_value)
{
    const char* value = std::getenv(name);
    if(value == nullptr || *value == '\0')
    {
        return default_value;
    }
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if(end == value || *end != '\0' || parsed <= 0)
    {
        return default_value;
    }
    return static_cast<int>(parsed);
}

// CPU 参考实现：partial_sort 获取 top-K 索引（用于正确性验证）
std::vector<int> cpu_topk_indices(const std::vector<float>& data, int rows, int cols, int k)
{
    std::vector<int> out(rows * k);
    for(int r = 0; r < rows; ++r)
    {
        std::vector<std::pair<float, int>> v(cols);
        for(int c = 0; c < cols; ++c)
            v[c] = {data[r * cols + c], c};
        std::partial_sort(v.begin(), v.begin() + k, v.end(),
            [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                if(a.first != b.first) return a.first > b.first;
                return a.second < b.second;
            });
        for(int i = 0; i < k; ++i)
            out[r * k + i] = v[i].second;
    }
    return out;
}

// 计算 GPU 结果与 CPU 参考的 top-K 准确率
// 比较方式：将 GPU 和 CPU 各自的 K 个值排序后逐一比较
float accuracy_topk(const std::vector<int>& gpu, const std::vector<int>& cpu,
                   int rows, int k, int cols, const std::vector<float>& data)
{
    int total = rows * k, hit = 0;
    for(int r = 0; r < rows; ++r)
    {
        std::vector<float> cpu_vals(k), gpu_vals(k);
        for(int i = 0; i < k; ++i)
        {
            cpu_vals[i] = data[r * cols + cpu[r * k + i]];
            int gidx = gpu[r * k + i];
            gpu_vals[i] = (gidx >= 0 && gidx < cols) ? data[r * cols + gidx] : -1e30f;
        }
        std::sort(cpu_vals.begin(), cpu_vals.end());
        std::sort(gpu_vals.begin(), gpu_vals.end());
        for(int i = 0; i < k; ++i)
        {
            if(cpu_vals[i] == gpu_vals[i]) hit++;
        }
    }
    return total > 0 ? static_cast<float>(hit) / static_cast<float>(total) : 0.0f;
}

} // namespace

int main()
{
    // ---- 测试参数 ----
    const int rows = 1;                    // batch size
    constexpr int cols = 60000;            // 每行元素数 (N)
    constexpr int k = 2048;                // top-K
    const int profile_repeats = read_positive_env_or_default("TOPK_PROFILE_REPEATS", 1000);

    std::cout << "[W4S8 OPT5 dummy-bucket+vectorized-lastfilter - random FP32] rows=" << rows << " cols=" << cols << " k=" << k << "\n";

    // ---- 初始化 HIP 运行时 + 生成随机测试数据 ----
    HIP_CALL(hipFree(0));  // 触发 HIP 运行时初始化
    std::vector<float> h_logits(rows * cols);
    std::mt19937 gen(42);  // 固定种子保证可复现
    std::uniform_real_distribution<float> dist(-1e9f, 1e9f);
    for(size_t i = 0; i < h_logits.size(); ++i) h_logits[i] = dist(gen);

    // ---- 分配 GPU 内存 ----
    float* d_logits = nullptr;
    int* d_indices = nullptr;
    int* d_rowStarts = nullptr;
    int* d_rowEnds = nullptr;
    int* d_seqLens = nullptr;
    HIP_CALL(hipMalloc(&d_logits, h_logits.size() * sizeof(float)));
    HIP_CALL(hipMalloc(&d_indices, rows * k * sizeof(int)));
    HIP_CALL(hipMalloc(&d_rowStarts, rows * sizeof(int)));
    HIP_CALL(hipMalloc(&d_rowEnds, rows * sizeof(int)));
    HIP_CALL(hipMalloc(&d_seqLens, rows * sizeof(int)));

    HIP_CALL(hipMemcpy(d_logits, h_logits.data(), h_logits.size() * sizeof(float), hipMemcpyHostToDevice));

    // Prefill: rowStarts=[0], rowEnds=[cols]
    std::vector<int> h_rowStarts(rows, 0);
    std::vector<int> h_rowEnds(rows, cols);
    HIP_CALL(hipMemcpy(d_rowStarts, h_rowStarts.data(), rows * sizeof(int), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_rowEnds, h_rowEnds.data(), rows * sizeof(int), hipMemcpyHostToDevice));

    // Decode: seqLens=[cols]（模拟已生成 cols 个 token 的场景）
    constexpr int NEXT_N = 1;
    std::vector<int> h_seqLens(rows, cols);
    HIP_CALL(hipMemcpy(d_seqLens, h_seqLens.data(), rows * sizeof(int), hipMemcpyHostToDevice));

    // ---- 查询 workspace 大小并分配（取 Prefill 和 Decode 的最大值）----
    size_t ws_prefill = 0, ws_decode = 0;
    aiter::standalone_stable_radix_topk_one_block_<float, int, 11, 1024, false, aiter::Phase::Prefill>(
        nullptr, ws_prefill, d_logits, static_cast<int*>(nullptr), rows, cols,
        static_cast<int*>(nullptr), static_cast<int*>(nullptr), k, nullptr, d_indices, false, 0, true);
    aiter::standalone_stable_radix_topk_one_block_<float, int, 11, 1024, false, aiter::Phase::Decode>(
        nullptr, ws_decode, d_logits, static_cast<int*>(nullptr), rows, cols,
        static_cast<int*>(nullptr), static_cast<int*>(nullptr), k, nullptr, d_indices, false, 0, true, NEXT_N);

    size_t workspace_size = std::max(ws_prefill, ws_decode);
    void* d_workspace = nullptr;
    HIP_CALL(hipMalloc(&d_workspace, workspace_size));

    const hipStream_t stream = 0;
    hipEvent_t start, stop;
    HIP_CALL(hipEventCreate(&start));
    HIP_CALL(hipEventCreate(&stop));

    const int warmup_iters = 200, iters = 200, rounds = 10;

    // ---- 通用 benchmark + 输出函数 ----
    auto run_benchmark = [&](const char* label, auto kernel_fn) {
        std::vector<float> round_us;
        round_us.reserve(rounds);

        for(int r = 0; r < rounds; ++r)
        {
            for(int i = 0; i < warmup_iters; ++i) kernel_fn();
            HIP_CALL(hipDeviceSynchronize());
            HIP_CALL(hipEventRecord(start, stream));
            for(int i = 0; i < iters; ++i) kernel_fn();
            HIP_CALL(hipEventRecord(stop, stream));
            HIP_CALL(hipEventSynchronize(stop));
            float ms = 0.0f;
            HIP_CALL(hipEventElapsedTime(&ms, start, stop));
            round_us.push_back((ms * 1000.0f) / static_cast<float>(iters));
        }

        // profiler 采样
        for(int i = 0; i < profile_repeats; ++i) kernel_fn();
        HIP_CALL(hipDeviceSynchronize());

        // 正确性验证
        std::vector<int> h_indices(rows * k);
        HIP_CALL(hipMemcpy(h_indices.data(), d_indices, rows * k * sizeof(int), hipMemcpyDeviceToHost));
        auto cpu_indices = cpu_topk_indices(h_logits, rows, cols, k);
        float acc = accuracy_topk(h_indices, cpu_indices, rows, k, cols, h_logits);

        // 统计延迟
        float sum = 0.0f, min_us = std::numeric_limits<float>::max(), max_us = 0.0f;
        for(float v : round_us) { sum += v; min_us = std::min(min_us, v); max_us = std::max(max_us, v); }
        float mean = sum / static_cast<float>(round_us.size());
        float var = 0.0f;
        for(float v : round_us) { float d = v - mean; var += d * d; }
        float std_us = std::sqrt(var / static_cast<float>(round_us.size()));
        std::vector<float> sorted_us = round_us;
        std::sort(sorted_us.begin(), sorted_us.end());
        float median_us = sorted_us[sorted_us.size() / 2];
        size_t trim = sorted_us.size() / 5, lo = trim, hi = sorted_us.size() - trim;
        float trimmed_sum = 0.0f;
        for(size_t i = lo; i < hi; ++i) trimmed_sum += sorted_us[i];
        float trimmed_mean = trimmed_sum / static_cast<float>(hi - lo);

        std::cout << "\n[" << label << "]\n";
        std::cout << "  accuracy=" << acc << "\n";
        std::cout << "  avg_latency_us=" << mean << "\n";
        std::cout << "  median_latency_us=" << median_us << "\n";
        std::cout << "  trimmed_avg_latency_us=" << trimmed_mean << "\n";
        std::cout << "  min_latency_us=" << min_us << "\n";
        std::cout << "  max_latency_us=" << max_us << "\n";
        std::cout << "  std_latency_us=" << std_us << "\n";
    };

    // =====================================================================
    // Benchmark: Prefill
    // =====================================================================
    HIP_CALL(hipMemset(d_indices, 0, rows * k * sizeof(int)));
    run_benchmark("Prefill", [&]() {
        size_t dummy = 0;
        aiter::standalone_stable_radix_topk_one_block_<float, int, 11, 1024, false, aiter::Phase::Prefill>(
            d_workspace, dummy, d_logits, static_cast<int*>(nullptr), rows, cols,
            d_rowStarts, d_rowEnds, k, nullptr, d_indices, false, stream, true);
    });

    // =====================================================================
    // Benchmark: Decode
    // =====================================================================
    HIP_CALL(hipMemset(d_indices, 0, rows * k * sizeof(int)));
    run_benchmark("Decode", [&]() {
        size_t dummy = 0;
        aiter::standalone_stable_radix_topk_one_block_<float, int, 11, 1024, false, aiter::Phase::Decode>(
            d_workspace, dummy, d_logits, static_cast<int*>(nullptr), rows, cols,
            nullptr, d_seqLens, k, nullptr, d_indices, false, stream, true, NEXT_N);
    });

    HIP_CALL(hipEventDestroy(start));
    HIP_CALL(hipEventDestroy(stop));
    HIP_CALL(hipFree(d_workspace));
    HIP_CALL(hipFree(d_indices));
    HIP_CALL(hipFree(d_logits));
    HIP_CALL(hipFree(d_rowStarts));
    HIP_CALL(hipFree(d_rowEnds));
    HIP_CALL(hipFree(d_seqLens));
    return 0;
}
