// topk_baseline_multiblocks.cu
// Standalone benchmark for AIR radix-sort TopK kernel (multi-block path)
// Extracted from aiter/csrc/kernels/topk_per_row_kernels.cu
// Test: 60K random FP32 elements, top-2048, measure latency for prefill & decode
//
// Build (inside ROCm docker):
//   hipcc -O3 -std=c++17 topk_baseline_multiblocks.cu -o topk_baseline_multiblocks
// Run:
//   ./topk_baseline_multiblocks

#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <hipcub/util_type.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <cassert>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <limits>

// =============================================================================
// HIP utilities
// =============================================================================

#define HIP_CALL(call)                                                         \
    do {                                                                        \
        hipError_t err = (call);                                                \
        if (err != hipSuccess) {                                                \
            fprintf(stderr, "HIP Error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    hipGetErrorString(err));                                    \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

static int get_num_cu_func()
{
    int device;
    HIP_CALL(hipGetDevice(&device));
    hipDeviceProp_t props;
    HIP_CALL(hipGetDeviceProperties(&props, device));
    return props.multiProcessorCount;
}

// =============================================================================
// Kernel code (radix-sort MULTI-BLOCK path from topk_per_row_kernels.cu)
// =============================================================================

namespace aiter {

using fp32x4 = __attribute__((__ext_vector_type__(4))) float;

using WideT                        = fp32x4;
constexpr int VECTORIZED_READ_SIZE = 16;
constexpr int WARP_SIZE            = 64;

enum class Phase
{
    Prefill,
    Decode,
};

template <int BitsPerPass>
__host__ __device__ constexpr int calc_num_buckets()
{
    return 1 << BitsPerPass;
}

template <typename IntType>
constexpr __host__ __device__ IntType ceildiv(IntType a, IntType b)
{
    return (a + b - 1) / b;
}

template <typename IntType>
constexpr __host__ __device__ IntType alignTo(IntType a, IntType b)
{
    return ceildiv(a, b) * b;
}

template <typename T, int BitsPerPass>
__host__ __device__ constexpr int calc_num_passes()
{
    return ceildiv<int>(sizeof(T) * 8, BitsPerPass);
}

template <typename T, int BitsPerPass>
__device__ constexpr int calc_start_bit(int pass)
{
    int start_bit = static_cast<int>(sizeof(T) * 8) - (pass + 1) * BitsPerPass;
    return start_bit < 0 ? 0 : start_bit;
}

template <typename T, int BitsPerPass>
__device__ constexpr unsigned calc_mask(int pass)
{
    static_assert(BitsPerPass <= 31);
    int num_bits = calc_start_bit<T, BitsPerPass>(pass - 1) - calc_start_bit<T, BitsPerPass>(pass);
    return (1 << num_bits) - 1;
}

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
        if(!select_min) { bits = ~bits; }
        return bits;
    }
}

template <typename T>
__device__ T twiddle_out(typename hipcub::Traits<T>::UnsignedBits bits, bool select_min)
{
    if(!select_min) { bits = ~bits; }
    bits = hipcub::Traits<T>::TwiddleOut(bits);
    return reinterpret_cast<T&>(bits);
}

template <typename T, int BitsPerPass>
__device__ int calc_bucket(T x, int start_bit, unsigned mask, bool select_min)
{
    static_assert(BitsPerPass <= sizeof(int) * 8 - 1);
    return (twiddle_in(x, select_min) >> start_bit) & mask;
}

template <typename I>
constexpr inline std::enable_if_t<std::is_integral<I>::value, bool>
is_a_power_of_two(I val) noexcept
{
    return ((val - 1) & val) == 0;
}

template <typename T, typename IdxT, typename RATIO_T = float>
__host__ __device__ IdxT calc_buf_len(IdxT len)
{
    constexpr RATIO_T ratio = 2 + sizeof(IdxT) * 2 / sizeof(T);
    IdxT buf_len = len / (ratio * 8);
    static_assert(is_a_power_of_two(sizeof(T)));
    static_assert(is_a_power_of_two(sizeof(IdxT)));
    constexpr IdxT aligned = 256 / std::min(sizeof(T), sizeof(IdxT));
    buf_len                = buf_len & (~(aligned - 1));
    return buf_len;
}

// ---------------------------------------------------------------------------
// vectorized_process (version used by multi-block filter_and_histogram)
// Uses global thread_rank / num_threads across all blocks
// ---------------------------------------------------------------------------
template <typename T, typename IdxT, typename Func>
__device__ void
vectorized_process(size_t thread_rank, size_t num_threads, T const* in, IdxT len, Func f)
{
    T val;
    int acc          = 0;
    int prev_bin_idx = -1;

    if constexpr(sizeof(T) >= sizeof(WideT))
    {
        for(IdxT i = thread_rank; i < len; i += num_threads)
        {
            val = in[i];
            f(in[i], i, acc, prev_bin_idx, false);
        }
    }
    else
    {
        static_assert(sizeof(WideT) % sizeof(T) == 0);
        constexpr int items_per_scalar = sizeof(WideT) / sizeof(T);

        union
        {
            WideT scalar;
            T array[items_per_scalar];
        } wide;

        int skip_cnt =
            (reinterpret_cast<size_t>(in) % sizeof(WideT))
                ? ((sizeof(WideT) - reinterpret_cast<size_t>(in) % sizeof(WideT)) / sizeof(T))
                : 0;
        if(skip_cnt > len) { skip_cnt = len; }
        WideT const* in_cast = reinterpret_cast<decltype(in_cast)>(in + skip_cnt);
        const IdxT len_cast  = (len - skip_cnt) / items_per_scalar;

        for(IdxT i = thread_rank; i < len_cast; i += num_threads)
        {
            wide.scalar       = in_cast[i];
            const IdxT real_i = skip_cnt + i * items_per_scalar;
#pragma unroll
            for(int j = 0; j < items_per_scalar; ++j)
            {
                val = wide.array[j];
                f(wide.array[j], real_i + j, acc, prev_bin_idx, false);
            }
        }

        static_assert(WARP_SIZE >= items_per_scalar);
        if(thread_rank < skip_cnt)
        {
            val = in[thread_rank];
            f(in[thread_rank], thread_rank, acc, prev_bin_idx, false);
        }
        const IdxT remain_i = skip_cnt + len_cast * items_per_scalar + thread_rank;
        if(remain_i < len)
        {
            val = in[remain_i];
            f(in[remain_i], remain_i, acc, prev_bin_idx, false);
        }
    }

    if(acc > 0) { f(-val, 0, acc, prev_bin_idx, true); }
}

// ---------------------------------------------------------------------------
// Counter
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// filter_and_histogram (MULTI-BLOCK version)
// Each block computes a local histogram in smem, then merges via atomicAdd
// to global histogram.
// ---------------------------------------------------------------------------
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
                    if(WRITE_TOPK_VALUES) { out[pos] = value; }
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
                if(WRITE_TOPK_VALUES) { out[pos] = value; }
                out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
            }
        };
        vectorized_process(static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x,
                           static_cast<size_t>(blockDim.x) * gridDim.x,
                           in_buf,
                           previous_len,
                           f);
    }
    if(early_stop) { return; }
    __syncthreads();

    // merge per-block histograms into global histogram
    for(int i = threadIdx.x; i < num_buckets; i += blockDim.x)
    {
        if(histogram_smem[i] != 0)
        {
            atomicAdd(histogram + i, histogram_smem[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// scan: prefix-sum over histogram
// ---------------------------------------------------------------------------
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
        if(threadIdx.x < num_buckets) { thread_data = histogram[threadIdx.x]; }
        BlockScan(temp_storage).InclusiveSum(thread_data, thread_data);
        __syncthreads();
        if(threadIdx.x < num_buckets) { histogram[threadIdx.x] = thread_data; }
    }
}

// ---------------------------------------------------------------------------
// choose_bucket
// ---------------------------------------------------------------------------
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
            int start_bit = calc_start_bit<T, BitsPerPass>(pass);
            counter->kth_value_bits |= bucket << start_bit;
        }
    }
}

// ---------------------------------------------------------------------------
// last_filter (device function, used by fused path inside radix_kernel)
// ---------------------------------------------------------------------------
template <typename T, typename IdxT, int BitsPerPass, bool WRITE_TOPK_VALUES,
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
    auto const kth_value_bits    = counter->kth_value_bits;
    int const start_bit          = calc_start_bit<T, BitsPerPass>(pass);
    const IdxT num_of_kth_needed = counter->k;
    IdxT* p_out_cnt              = &counter->out_cnt;
    IdxT* p_out_back_cnt         = &counter->out_back_cnt;

    if(in_idx_buf)
    {
        for(IdxT i = threadIdx.x; i < current_len; i += blockDim.x)
        {
            const T value   = in_buf[i];
            auto const bits = use_one_pass
                                  ? twiddle_in(value, select_min) & ((1 << BitsPerPass) - 1)
                                  : (twiddle_in(value, select_min) >> start_bit) << start_bit;
            if(bits < kth_value_bits)
            {
                IdxT pos = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
                if(WRITE_TOPK_VALUES) { out[pos] = value; }
                out_idx[pos] = in_idx_buf[i];
            }
            else if(bits == kth_value_bits)
            {
                IdxT new_idx  = in_idx_buf[i];
                IdxT back_pos = atomicAdd(p_out_back_cnt, static_cast<IdxT>(1));
                if(back_pos < num_of_kth_needed)
                {
                    IdxT pos = k - 1 - back_pos;
                    if(WRITE_TOPK_VALUES) { out[pos] = value; }
                    if constexpr(!prioritize_smaller_indice) { out_idx[pos] = new_idx; }
                }
            }
        }
    }
    else
    {
        for(IdxT i = threadIdx.x; i < current_len; i += blockDim.x)
        {
            const T value   = in_buf[i];
            auto const bits = use_one_pass
                                  ? twiddle_in(value, select_min) & ((1 << BitsPerPass) - 1)
                                  : (twiddle_in(value, select_min) >> start_bit) << start_bit;
            if(bits < kth_value_bits)
            {
                IdxT pos = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
                if(WRITE_TOPK_VALUES) { out[pos] = value; }
                out_idx[pos] = i;
            }
            else if(bits == kth_value_bits)
            {
                IdxT new_idx  = i;
                IdxT back_pos = atomicAdd(p_out_back_cnt, static_cast<IdxT>(1));
                if(back_pos < num_of_kth_needed)
                {
                    IdxT pos = k - 1 - back_pos;
                    if(WRITE_TOPK_VALUES) { out[pos] = value; }
                    if constexpr(!prioritize_smaller_indice) { out_idx[pos] = new_idx; }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// last_filter_kernel (separate kernel for non-fused multi-block path)
// ---------------------------------------------------------------------------
template <typename T, typename IdxT, int BitsPerPass, bool WRITE_TOPK_VALUES,
          Phase phase, bool prioritize_smaller_indice = false>
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
    if(previous_len == 0) { return; }

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

    auto f = [k, select_min, kth_value_bits, num_of_kth_needed,
              p_out_cnt, p_out_back_cnt, in_idx_buf,
              out, out_idx](T value, IdxT i, int&, int&, bool) {
        const auto bits = (twiddle_in(value, select_min) >> start_bit) << start_bit;
        if(bits < kth_value_bits)
        {
            IdxT pos = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
            if(WRITE_TOPK_VALUES) { out[pos] = value; }
            out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
        }
        else if(bits == kth_value_bits)
        {
            IdxT new_idx  = in_idx_buf ? in_idx_buf[i] : i;
            IdxT back_pos = atomicAdd(p_out_back_cnt, static_cast<IdxT>(1));
            if(back_pos < num_of_kth_needed)
            {
                IdxT pos = k - 1 - back_pos;
                if(WRITE_TOPK_VALUES) { out[pos] = value; }
                if constexpr(!prioritize_smaller_indice) { out_idx[pos] = new_idx; }
            }
        }
    };

    vectorized_process(static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x,
                       static_cast<size_t>(blockDim.x) * gridDim.x,
                       in_buf,
                       previous_len,
                       f);
}

// ---------------------------------------------------------------------------
// radix_kernel (MULTI-BLOCK kernel)
// Multiple blocks collaborate on one row; last block does scan+choose_bucket.
// ---------------------------------------------------------------------------
template <typename T, typename IdxT, int BitsPerPass, int BlockSize,
          bool fused_last_filter, bool WRITE_TOPK_VALUES,
          bool prioritize_smaller_indice = false, Phase phase = Phase::Prefill>
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
    if(current_len == 0) { return; }

    bool const early_stop = (current_len == current_k);
    const IdxT buf_len    = calc_buf_len<T>(len);

    // "previous_len > buf_len" means previous pass skipped writing buffer
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
    // "current_len > buf_len" means current pass will skip writing buffer
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

    filter_and_histogram<T, IdxT, BitsPerPass, WRITE_TOPK_VALUES>(
        in_buf, in_idx_buf, out_buf, out_idx_buf, out, out_idx,
        previous_len, counter, histogram, select_min, pass, early_stop, k);
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
        // reset histogram for next pass
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
                    out, out_idx,
                    out_buf ? current_len : row_len,
                    k, counter, select_min, pass);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// calc_grid_dim: occupancy-based grid sizing for multi-block
// ---------------------------------------------------------------------------
template <typename T, typename IdxT, int BitsPerPass, int BlockSize,
          bool WRITE_TOPK_VALUES, Phase phase>
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

        if(num_blocks == max_num_blocks) { break; }
    }
    return best_num_blocks;
}

// ---------------------------------------------------------------------------
// set_buf_pointers (HOST version with separate buf1/buf2 for multi-block)
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------
inline size_t calc_aligned_size(std::vector<size_t> const& sizes)
{
    const size_t ALIGN_BYTES = 256;
    const size_t ALIGN_MASK  = ~(ALIGN_BYTES - 1);
    size_t total             = 0;
    for(auto sz : sizes) { total += (sz + ALIGN_BYTES - 1) & ALIGN_MASK; }
    return total + ALIGN_BYTES - 1;
}

inline std::vector<void*> calc_aligned_pointers(void const* p, std::vector<size_t> const& sizes)
{
    const size_t ALIGN_BYTES = 256;
    const size_t ALIGN_MASK  = ~(ALIGN_BYTES - 1);
    char* ptr = reinterpret_cast<char*>(
        (reinterpret_cast<size_t>(p) + ALIGN_BYTES - 1) & ALIGN_MASK);

    std::vector<void*> aligned_pointers;
    aligned_pointers.reserve(sizes.size());
    for(auto sz : sizes)
    {
        aligned_pointers.push_back(ptr);
        ptr += (sz + ALIGN_BYTES - 1) & ALIGN_MASK;
    }
    return aligned_pointers;
}

// ---------------------------------------------------------------------------
// standalone_stable_radix_topk_ (MULTI-BLOCK host launcher)
// Launches radix_kernel for each pass, then last_filter_kernel if not fused.
// ---------------------------------------------------------------------------
template <typename T, typename IdxT, int BitsPerPass, int BlockSize,
          bool WRITE_TOPK_VALUES, Phase phase = Phase::Prefill>
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
        counters   = static_cast<decltype(counters)>(aligned_pointers[0]);
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
        set_buf_pointers(in, in_idx, buf1, idx_buf1, buf2, idx_buf2,
                         pass, in_buf, in_idx_buf, out_buf, out_idx_buf);

        if(fused_last_filter && pass == num_passes - 1)
        {
            kernel = radix_kernel<T, IdxT, BitsPerPass, BlockSize,
                                  true, WRITE_TOPK_VALUES, false, phase>;
        }

        kernel<<<blocks, BlockSize, 0, stream>>>(
            in, in_idx, in_buf, in_idx_buf, out_buf, out_idx_buf,
            out, out_idx, counters, histograms,
            len, rowStarts, rowEnds, k, next_n, select_min, pass);
    }

    if(!fused_last_filter)
    {
        last_filter_kernel<T, IdxT, BitsPerPass, WRITE_TOPK_VALUES, phase, false>
            <<<blocks, BlockSize, 0, stream>>>(
                in, in_idx, out_buf, out_idx_buf, out, out_idx,
                len, rowStarts, rowEnds, k, next_n, counters, select_min);
    }
}

// ---------------------------------------------------------------------------
// standalone_stable_radix_11bits (multi-block entry point)
// ---------------------------------------------------------------------------
template <typename T, typename IdxT, bool WRITE_TOPK_VALUES,
          bool sorted = false, Phase phase = Phase::Prefill>
void standalone_stable_radix_11bits(void* buf,
                                    size_t& buf_size,
                                    T const* in,
                                    int batch_size,
                                    int64_t len,
                                    IdxT* rowStarts,
                                    IdxT* rowEnds,
                                    IdxT k,
                                    T* out,
                                    IdxT* out_idx,
                                    bool greater,
                                    hipStream_t stream,
                                    int next_n = 0)
{
    constexpr int block_dim          = 1024;
    constexpr bool fused_last_filter = false;

    int sm_cnt = get_num_cu_func();
    unsigned grid_dim =
        calc_grid_dim<T, IdxT, 11, block_dim, WRITE_TOPK_VALUES, phase>(batch_size, len, sm_cnt);

    standalone_stable_radix_topk_<T, IdxT, 11, block_dim, WRITE_TOPK_VALUES, phase>(
        buf, buf_size, in, static_cast<IdxT*>(nullptr),
        batch_size, len, rowStarts, rowEnds, k, out, out_idx,
        !greater, fused_last_filter, grid_dim, stream, sorted, next_n);
}

} // namespace aiter

// =============================================================================
// Benchmark main
// =============================================================================

int main(int argc, char** argv)
{
    // -------------------------------------------------------------------------
    // Parameters
    // -------------------------------------------------------------------------
    constexpr int    LEN       = 60000;
    constexpr int    K         = 2048;
    constexpr int    BATCH     = 1;
    constexpr int    WARMUP    = 50;
    constexpr int    REPEAT    = 200;
    constexpr bool   IS_LARGEST = true;

    printf("=== AIR Radix TopK Benchmark (MULTI-BLOCK) ===\n");
    printf("LEN=%d  K=%d  BATCH=%d  WARMUP=%d  REPEAT=%d\n\n", LEN, K, BATCH, WARMUP, REPEAT);

    // -------------------------------------------------------------------------
    // Host data: random FP32 (full range)
    // -------------------------------------------------------------------------
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1e6f, 1e6f);

    std::vector<float> h_in(BATCH * LEN);
    for(auto& v : h_in) { v = dist(rng); }

    std::vector<int> h_rowStarts(BATCH, 0);
    std::vector<int> h_rowEnds(BATCH, LEN);
    std::vector<int> h_seqLens(BATCH, LEN);
    constexpr int NEXT_N = 1;

    // -------------------------------------------------------------------------
    // Device allocations
    // -------------------------------------------------------------------------
    float* d_in        = nullptr;
    int*   d_out_idx   = nullptr;
    int*   d_rowStarts = nullptr;
    int*   d_rowEnds   = nullptr;
    int*   d_seqLens   = nullptr;

    HIP_CALL(hipMalloc(&d_in,        sizeof(float) * BATCH * LEN));
    HIP_CALL(hipMalloc(&d_out_idx,   sizeof(int) * BATCH * K));
    HIP_CALL(hipMalloc(&d_rowStarts, sizeof(int) * BATCH));
    HIP_CALL(hipMalloc(&d_rowEnds,   sizeof(int) * BATCH));
    HIP_CALL(hipMalloc(&d_seqLens,   sizeof(int) * BATCH));

    HIP_CALL(hipMemcpy(d_in,        h_in.data(),        sizeof(float) * BATCH * LEN, hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_rowStarts, h_rowStarts.data(),  sizeof(int) * BATCH,         hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_rowEnds,   h_rowEnds.data(),    sizeof(int) * BATCH,         hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_seqLens,   h_seqLens.data(),    sizeof(int) * BATCH,         hipMemcpyHostToDevice));

    // -------------------------------------------------------------------------
    // Workspace (query size then allocate)
    // -------------------------------------------------------------------------
    size_t ws_size_prefill = 0;
    size_t ws_size_decode  = 0;

    aiter::standalone_stable_radix_11bits<float, int, false, true, aiter::Phase::Prefill>(
        nullptr, ws_size_prefill, nullptr, BATCH, LEN,
        nullptr, nullptr, K, nullptr, nullptr, IS_LARGEST, 0);

    aiter::standalone_stable_radix_11bits<float, int, false, true, aiter::Phase::Decode>(
        nullptr, ws_size_decode, nullptr, BATCH, LEN,
        nullptr, nullptr, K, nullptr, nullptr, IS_LARGEST, 0, NEXT_N);

    size_t ws_size = std::max(ws_size_prefill, ws_size_decode);
    void* d_workspace = nullptr;
    HIP_CALL(hipMalloc(&d_workspace, ws_size));
    printf("Workspace size: %zu bytes (%.2f KB)\n\n", ws_size, ws_size / 1024.0);

    // Print grid_dim info
    {
        int sm_cnt = get_num_cu_func();
        unsigned gd = aiter::calc_grid_dim<float, int, 11, 1024, false, aiter::Phase::Prefill>(
            BATCH, LEN, sm_cnt);
        printf("CU count: %d,  grid_dim (blocks per row): %u\n\n", sm_cnt, gd);
    }

    // -------------------------------------------------------------------------
    // HIP events for timing
    // -------------------------------------------------------------------------
    hipEvent_t start, stop;
    HIP_CALL(hipEventCreate(&start));
    HIP_CALL(hipEventCreate(&stop));

    hipStream_t stream;
    HIP_CALL(hipStreamCreate(&stream));

    // =====================================================================
    // Benchmark: PREFILL (multi-block)
    // =====================================================================
    {
        size_t dummy = 0;

        // Warmup
        for(int i = 0; i < WARMUP; ++i)
        {
            aiter::standalone_stable_radix_11bits<float, int, false, true, aiter::Phase::Prefill>(
                d_workspace, dummy, d_in, BATCH, LEN,
                d_rowStarts, d_rowEnds, K, nullptr, d_out_idx,
                IS_LARGEST, stream);
        }
        HIP_CALL(hipStreamSynchronize(stream));

        // Timed runs
        HIP_CALL(hipEventRecord(start, stream));
        for(int i = 0; i < REPEAT; ++i)
        {
            aiter::standalone_stable_radix_11bits<float, int, false, true, aiter::Phase::Prefill>(
                d_workspace, dummy, d_in, BATCH, LEN,
                d_rowStarts, d_rowEnds, K, nullptr, d_out_idx,
                IS_LARGEST, stream);
        }
        HIP_CALL(hipEventRecord(stop, stream));
        HIP_CALL(hipEventSynchronize(stop));

        float total_ms = 0;
        HIP_CALL(hipEventElapsedTime(&total_ms, start, stop));
        float avg_us = total_ms / REPEAT * 1000.0f;

        printf("[Prefill] avg latency: %.2f us  (%.4f ms)  over %d runs\n",
               avg_us, total_ms / REPEAT, REPEAT);
    }

    // =====================================================================
    // Benchmark: DECODE (multi-block)
    // =====================================================================
    {
        size_t dummy = 0;

        // Warmup
        for(int i = 0; i < WARMUP; ++i)
        {
            aiter::standalone_stable_radix_11bits<float, int, false, true, aiter::Phase::Decode>(
                d_workspace, dummy, d_in, BATCH, LEN,
                nullptr, d_seqLens, K, nullptr, d_out_idx,
                IS_LARGEST, stream, NEXT_N);
        }
        HIP_CALL(hipStreamSynchronize(stream));

        // Timed runs
        HIP_CALL(hipEventRecord(start, stream));
        for(int i = 0; i < REPEAT; ++i)
        {
            aiter::standalone_stable_radix_11bits<float, int, false, true, aiter::Phase::Decode>(
                d_workspace, dummy, d_in, BATCH, LEN,
                nullptr, d_seqLens, K, nullptr, d_out_idx,
                IS_LARGEST, stream, NEXT_N);
        }
        HIP_CALL(hipEventRecord(stop, stream));
        HIP_CALL(hipEventSynchronize(stop));

        float total_ms = 0;
        HIP_CALL(hipEventElapsedTime(&total_ms, start, stop));
        float avg_us = total_ms / REPEAT * 1000.0f;

        printf("[Decode]  avg latency: %.2f us  (%.4f ms)  over %d runs\n",
               avg_us, total_ms / REPEAT, REPEAT);
    }

    // =====================================================================
    // Correctness verification (prefill path)
    // =====================================================================
    {
        size_t dummy = 0;
        HIP_CALL(hipMemset(d_out_idx, 0, sizeof(int) * BATCH * K));
        aiter::standalone_stable_radix_11bits<float, int, false, true, aiter::Phase::Prefill>(
            d_workspace, dummy, d_in, BATCH, LEN,
            d_rowStarts, d_rowEnds, K, nullptr, d_out_idx,
            IS_LARGEST, stream);
        HIP_CALL(hipStreamSynchronize(stream));

        std::vector<int> h_out_idx(BATCH * K);
        HIP_CALL(hipMemcpy(h_out_idx.data(), d_out_idx, sizeof(int) * BATCH * K, hipMemcpyDeviceToHost));

        printf("\n--- Correctness check (prefill, batch 0) ---\n");

        std::vector<int> sorted_idx(LEN);
        std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
        std::partial_sort(sorted_idx.begin(), sorted_idx.begin() + K, sorted_idx.end(),
                          [&](int a, int b) { return h_in[a] > h_in[b]; });

        float kth_value = h_in[sorted_idx[K - 1]];

        int errors = 0;
        int out_of_range = 0;
        for(int i = 0; i < K; ++i)
        {
            int idx = h_out_idx[i];
            if(idx < 0 || idx >= LEN)
            {
                out_of_range++;
                continue;
            }
            if(h_in[idx] < kth_value) { errors++; }
        }

        std::vector<float> gpu_vals(K), cpu_vals(K);
        for(int i = 0; i < K; ++i)
        {
            int gpu_idx = h_out_idx[i];
            gpu_vals[i] = (gpu_idx >= 0 && gpu_idx < LEN) ? h_in[gpu_idx] : -FLT_MAX;
            cpu_vals[i] = h_in[sorted_idx[i]];
        }
        std::sort(gpu_vals.begin(), gpu_vals.end(), std::greater<float>());
        std::sort(cpu_vals.begin(), cpu_vals.end(), std::greater<float>());

        bool match = (gpu_vals == cpu_vals);

        printf("  K-th largest value (CPU ref): %f\n", kth_value);
        printf("  Out-of-range indices: %d\n", out_of_range);
        printf("  Values below k-th:    %d\n", errors);
        printf("  Top-K value set match: %s\n", match ? "PASS" : "FAIL");
    }

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------
    HIP_CALL(hipEventDestroy(start));
    HIP_CALL(hipEventDestroy(stop));
    HIP_CALL(hipStreamDestroy(stream));
    HIP_CALL(hipFree(d_in));
    HIP_CALL(hipFree(d_out_idx));
    HIP_CALL(hipFree(d_rowStarts));
    HIP_CALL(hipFree(d_rowEnds));
    HIP_CALL(hipFree(d_seqLens));
    HIP_CALL(hipFree(d_workspace));

    printf("\nDone.\n");
    return 0;
}
