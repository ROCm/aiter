#include "rope/rope_common.h"
#include "hip_float8.h"

using namespace at;

namespace rope_rms {

static constexpr int kBytesPerAccess = 16;

namespace block_utils {

template <typename T>
__inline__ __device__ T warp_shfl_xor_sync(T val, int offset) {
    return __shfl_xor(val, offset, 32);
}

template <typename T>
__inline__ __device__ T warp_reduce_sum(T val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += warp_shfl_xor_sync(val, offset);
    return val;
}

template <typename T>
__inline__ __device__ T warp_shfl_sync(T val, int src_id) {
    return __shfl(val, src_id, 32);
}

} // namespace block_utils

template <typename T, int vec_size>
struct alignas(sizeof(T) * vec_size) vec_t {
    T data[vec_size];
    __device__ __forceinline__ T &operator[](int i) {
        return data[i];
    }
    __device__ __forceinline__ T const &operator[](int i) const {
        return data[i];
    }
    __device__ __forceinline__ void load(const T *ptr) {
        *this = *reinterpret_cast<vec_t<T, vec_size> *>(const_cast<T *>(ptr));
    }
    __device__ __forceinline__ void loop_load(const T *ptr) {
#pragma unroll
        for (int i = 0; i < vec_size; ++i) {
            data[i] = ptr[i];
        }
    }
    __device__ __forceinline__ void store(T *ptr) {
        *reinterpret_cast<vec_t<T, vec_size> *>(ptr) = *this;
    }
    __device__ __forceinline__ void loop_store(T *ptr) {
#pragma unroll
        for (int i = 0; i < vec_size; ++i) {
            ptr[i] = data[i];
        }
    }
    __device__ __forceinline__ void nontemporal_load(const T *ptr) {
        constexpr int ITERS = vec_size * sizeof(T) / sizeof(uint32_t);
#pragma unroll
        for (int i = 0; i < ITERS; ++i) {
            reinterpret_cast<uint32_t *>(&data)[i] = __builtin_nontemporal_load((uint32_t *)ptr + i);
        }
    }
    __device__ __forceinline__ void nontemporal_store(T *ptr) {
        constexpr int ITERS = vec_size * sizeof(T) / sizeof(uint32_t);
#pragma unroll
        for (int i = 0; i < ITERS; ++i) {
            __builtin_nontemporal_store(reinterpret_cast<uint32_t *>(&data)[i], (uint32_t *)ptr + i);
        }
    }
    __device__ __forceinline__ void fill(T val) {
#pragma unroll
        for (int i = 0; i < vec_size; ++i) {
            data[i] = val;
        }
    }
};

template <typename T, int vec_size>
__inline__ __device__ vec_t<T, vec_size> warp_shfl_sync_vec(vec_t<T, vec_size> &val, int offset) {
    constexpr int ITERS = vec_size * sizeof(T) / sizeof(uint32_t);
    vec_t<T, vec_size> out;
#pragma unroll
    for (int i = 0; i < ITERS; ++i) {
        uint32_t val_ = reinterpret_cast<uint32_t *>(&val)[i];
        reinterpret_cast<uint32_t *>(&out)[i] = block_utils::warp_shfl_sync<uint32_t>(val_, offset);
    }
    return out;
}

template <typename T, int VEC_SIZE>
__device__ __forceinline__ void warp_rms_norm_(
    vec_t<T, VEC_SIZE> &input,
    vec_t<T, VEC_SIZE> &gamma,
    float rms_dim,
    float rms_eps) {
    vec_t<T, VEC_SIZE> norm_out;
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        float v = (float)input[i];
        acc += v * v;
    }
    int warp_id = threadIdx.x / 32;
    int warp_t_id = threadIdx.x % 32;
    acc = block_utils::warp_reduce_sum<float>(acc);
    acc = block_utils::warp_shfl_sync<float>(acc, 0);
    auto s_val = rsqrtf(acc / rms_dim + rms_eps);
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        input[i] = static_cast<T>((float)input[i] * s_val * (float)gamma[i]);
    }
}

template <typename T, int VEC_SIZE, int HEAD_SIZE, bool IS_INTERLEAVED, int M>
__device__ __forceinline__ void mrope_load_cos_sin_vec(vec_t<T, VEC_SIZE> &out,
                                                       const T *cos_sin, const int64_t *positions, int64_t ps0, int64_t ps1,
                                                       int64_t token_id, int64_t num_tokens,
                                                       int access_id_in_head, std::array<int64_t, M> &mrope_section) {
    constexpr int HALF_HEAD_SIZE = HEAD_SIZE / 2;
    if constexpr (IS_INTERLEAVED) {
        for (int i = 0; i < VEC_SIZE; ++i) {
            auto id = access_id_in_head + i;
            auto id_ = (access_id_in_head < HALF_HEAD_SIZE) ? id : id - HALF_HEAD_SIZE;
            auto mid_ = id_ % M;
            if (mid_ >= 1 && id_ < mrope_section[mid_] * M) {
                auto p = positions[mid_ * ps0 + token_id * ps1];
                out[i] = cos_sin[p * HEAD_SIZE + id];
            } else {
                out[i] = cos_sin[positions[token_id * ps1] * HEAD_SIZE + id];
            }
        }
    } else {
        for (int i = 0; i < VEC_SIZE; ++i) {
            auto id = access_id_in_head + i;
            auto id_ = (access_id_in_head < HALF_HEAD_SIZE) ? id : id - HALF_HEAD_SIZE;
            int mid;
            int end = 0;
            for (mid = 0; mid < M; ++mid) {
                end += mrope_section[mid];
                if (id_ < end)
                    break;
            }
            auto p = positions[mid * ps0 + token_id * ps1];
            out[i] = cos_sin[p * HEAD_SIZE + id];
        }
    }
}

template <typename T, int HEAD_SIZE, bool IS_MROPE, bool IS_INTERLEAVED, int M>
__global__ void fused_mrope_rms_neox_kernel(
    T *qkv, const T *q_w, const T *k_w, const T *cos_sin, const int64_t *positions, int64_t ps0, int64_t ps1,
    int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v, double eps,
    std::array<int64_t, M> mrope_section, int64_t num_tokens, int64_t total_warps) {
    constexpr int VEC_SIZE = HEAD_SIZE / 32;
    constexpr int HALF_HEAD_SIZE = HEAD_SIZE / 2;
    const auto warp_id = threadIdx.x / 32;
    const auto num_warps_per_block = blockDim.x / 32;
    const auto global_warp_id = blockIdx.x * num_warps_per_block + warp_id;
    if (global_warp_id >= total_warps) {
        return;
    }
    auto token_id = global_warp_id / (num_heads_q + num_heads_k);
    auto head_id_in_token = global_warp_id % (num_heads_q + num_heads_k);
    bool is_q = head_id_in_token < num_heads_q;
    auto access_id_in_head = (threadIdx.x % 32) * VEC_SIZE;
    auto neighbor_offset = access_id_in_head < HALF_HEAD_SIZE ? HALF_HEAD_SIZE / VEC_SIZE : -HALF_HEAD_SIZE / VEC_SIZE;
    auto qkv_ = qkv + token_id * (num_heads_q + num_heads_k + num_heads_v) * HEAD_SIZE + head_id_in_token * HEAD_SIZE;

    vec_t<T, VEC_SIZE> w_vec;

    if (is_q) {
        w_vec.load(q_w + access_id_in_head);
    } else {
        w_vec.load(k_w + access_id_in_head);
    }

    vec_t<T, VEC_SIZE> x_vec, cos_sin_vec;
    x_vec.load(qkv_ + access_id_in_head);
    if constexpr (IS_MROPE) {
        mrope_load_cos_sin_vec<T, VEC_SIZE, HEAD_SIZE, IS_INTERLEAVED, M>(
            cos_sin_vec, cos_sin, positions, ps0, ps1, token_id, num_tokens, access_id_in_head, mrope_section);
    } else {
        auto position_ = positions[token_id * ps1];
        cos_sin_vec.load(&cos_sin[position_ * HEAD_SIZE + access_id_in_head]);
    }

    warp_rms_norm_<T, VEC_SIZE>(x_vec, w_vec, HEAD_SIZE, eps);
    auto nb_cos_sin_vec = warp_shfl_sync_vec<T, VEC_SIZE>(cos_sin_vec, threadIdx.x + neighbor_offset);
    auto nb_x_vec = warp_shfl_sync_vec<T, VEC_SIZE>(x_vec, threadIdx.x + neighbor_offset);
    vec_t<T, VEC_SIZE> out_vec;
    if (neighbor_offset > 0) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            out_vec[i] = (float)x_vec[i] * (float)cos_sin_vec[i] - (float)nb_x_vec[i] * (float)nb_cos_sin_vec[i]; // x0 * cos - x1 * sin
        }
    } else {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            out_vec[i] = (float)x_vec[i] * (float)nb_cos_sin_vec[i] + (float)nb_x_vec[i] * (float)cos_sin_vec[i]; // x1 * cos + x0 * sin
        }
    }
    out_vec.store(qkv_ + access_id_in_head);
}

template <typename T, int HEAD_SIZE, bool IS_MROPE, bool IS_INTERLEAVED, int M>
__global__ void fused_mrope_rms_noneox_kernel(
    T *qkv, const T *q_w, const T *k_w, const T *cos_sin, const int64_t *positions, int64_t ps0, int64_t ps1,
    int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v, double eps,
    std::array<int64_t, M> mrope_section, int64_t num_tokens, int64_t total_warps) {
    constexpr int VEC_SIZE = HEAD_SIZE / 32;
    constexpr int HALF_HEAD_SIZE = HEAD_SIZE / 2;
    const auto warp_id = threadIdx.x / 32;
    const auto num_warps_per_block = blockDim.x / 32;
    const auto global_warp_id = blockIdx.x * num_warps_per_block + warp_id;
    if (global_warp_id >= total_warps) {
        return;
    }
    auto token_id = global_warp_id / (num_heads_q + num_heads_k);
    auto head_id_in_token = global_warp_id % (num_heads_q + num_heads_k);
    bool is_q = head_id_in_token < num_heads_q;
    auto access_id_in_head = (threadIdx.x % 32) * VEC_SIZE;
    auto qkv_ = qkv + token_id * (num_heads_q + num_heads_k + num_heads_v) * HEAD_SIZE + head_id_in_token * HEAD_SIZE;

    vec_t<T, VEC_SIZE> w_vec;

    if (is_q) {
        w_vec.load(q_w + access_id_in_head);
    } else {
        w_vec.load(k_w + access_id_in_head);
    }

    vec_t<T, VEC_SIZE> x_vec, cos_vec, sin_vec;
    x_vec.load(qkv_ + access_id_in_head);
    if constexpr (IS_MROPE) {
        mrope_load_cos_sin_vec<T, VEC_SIZE, HEAD_SIZE, IS_INTERLEAVED, M>(
            cos_vec, cos_sin, positions, ps0, ps1, token_id, num_tokens, access_id_in_head / 2, mrope_section);
        mrope_load_cos_sin_vec<T, VEC_SIZE, HEAD_SIZE, IS_INTERLEAVED, M>(
            sin_vec, cos_sin, positions, ps0, ps1, token_id, num_tokens, access_id_in_head / 2 + HALF_HEAD_SIZE, mrope_section);
    } else {
        auto position_ = positions[token_id * ps1];
        cos_vec.load(&cos_sin[position_ * HEAD_SIZE + access_id_in_head / 2]);
        sin_vec.load(&cos_sin[position_ * HEAD_SIZE + access_id_in_head / 2 + HALF_HEAD_SIZE]);
    }

    warp_rms_norm_<T, VEC_SIZE>(x_vec, w_vec, HEAD_SIZE, eps);

    vec_t<T, VEC_SIZE> out_vec;
#pragma unroll
    for (int i = 0; i < VEC_SIZE / 2; ++i) {
        out_vec[2 * i + 0] = (float)x_vec[2 * i + 0] * (float)cos_vec[i] - (float)x_vec[2 * i + 1] * (float)sin_vec[i];
        out_vec[2 * i + 1] = (float)x_vec[2 * i + 1] * (float)cos_vec[i] + (float)x_vec[2 * i + 0] * (float)sin_vec[i];
    }

    out_vec.store(qkv_ + access_id_in_head);
}

template <typename T>
void fused_rope_rms(
    T *qkv, const T *q_w, const T *k_w, const T *cos_sin, const int64_t *positions, int64_t ps0, int64_t ps1,
    int64_t num_tokens, int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v, int64_t head_size,
    bool is_neox_style, double eps, hipStream_t stream) {
    TORCH_CHECK(head_size == 64 || head_size == 128 || head_size == 256);
    constexpr int block_size = 256;
    auto total_warps = num_tokens * (num_heads_q + num_heads_k);
    auto num_warps_per_block = block_size / 32;
    dim3 threadsPerBlock(block_size);
    dim3 numBlocks((total_warps + num_warps_per_block - 1) / num_warps_per_block);
    std::array<int64_t, 1> mrope_section = {0};

#define DISPATCH_NEOX(HEAD_SIZE)                                                                                                              \
    if (is_neox_style) {                                                                                                                      \
        fused_mrope_rms_neox_kernel<T, HEAD_SIZE, false, false, 1><<<numBlocks, threadsPerBlock, 0, stream>>>(                                \
            qkv, q_w, k_w, cos_sin, positions, ps0, ps1, num_heads_q, num_heads_k, num_heads_v, eps, mrope_section, num_tokens, total_warps); \
    } else {                                                                                                                                  \
        fused_mrope_rms_noneox_kernel<T, HEAD_SIZE, false, false, 1><<<numBlocks, threadsPerBlock, 0, stream>>>(                              \
            qkv, q_w, k_w, cos_sin, positions, ps0, ps1, num_heads_q, num_heads_k, num_heads_v, eps, mrope_section, num_tokens, total_warps); \
    }

    switch (head_size) {
    case 64:
        DISPATCH_NEOX(64)
        break;
    case 128:
        DISPATCH_NEOX(128)
        break;
    case 256:
        DISPATCH_NEOX(256)
        break;
    }

#undef DISPATCH_NEOX
}

template <typename T, int M>
void fused_mrope_rms(
    T *qkv, const T *q_w, const T *k_w, const T *cos_sin, const int64_t *positions, int64_t ps0, int64_t ps1,
    int64_t num_tokens, int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v, int64_t head_size,
    bool is_neox_style, double eps, std::array<int64_t, M> mrope_section, bool is_interleaved, hipStream_t stream) {
    TORCH_CHECK(head_size == 64 || head_size == 128 || head_size == 256);
    auto dim = std::accumulate(mrope_section.begin(), mrope_section.end(), 0);
    TORCH_CHECK(dim == head_size / 2);
    constexpr int block_size = 256;
    auto total_warps = num_tokens * (num_heads_q + num_heads_k);
    auto num_warps_per_block = block_size / 32;
    dim3 threadsPerBlock(block_size);
    dim3 numBlocks((total_warps + num_warps_per_block - 1) / num_warps_per_block);

#define DISPATCH_NEOX(HEAD_SIZE, IS_INTERLEAVED)                                                                                    \
    if (is_neox_style) {                                                                                                            \
        fused_mrope_rms_neox_kernel<T, HEAD_SIZE, true, IS_INTERLEAVED, M><<<numBlocks, threadsPerBlock, 0, stream>>>(              \
            qkv, q_w, k_w, cos_sin, positions, ps0, ps1, num_heads_q, num_heads_k, num_heads_v, eps, mrope_section, num_tokens, total_warps); \
    } else {                                                                                                                        \
        fused_mrope_rms_noneox_kernel<T, HEAD_SIZE, true, IS_INTERLEAVED, M><<<numBlocks, threadsPerBlock, 0, stream>>>(            \
            qkv, q_w, k_w, cos_sin, positions, ps0, ps1, num_heads_q, num_heads_k, num_heads_v, eps, mrope_section, num_tokens, total_warps); \
    }

    if (is_interleaved) {
        switch (head_size) {
        case 64:
            DISPATCH_NEOX(64, true)
            break;
        case 128:
            DISPATCH_NEOX(128, true)
            break;
        case 256:
            DISPATCH_NEOX(256, true)
            break;
        }
    } else {
        switch (head_size) {
        case 64:
            DISPATCH_NEOX(64, false)
            break;
        case 128:
            DISPATCH_NEOX(128, false)
            break;
        case 256:
            DISPATCH_NEOX(256, false)
            break;
        }
    }

#undef DISPATCH_NEOX
}

static constexpr int WARP_SIZE = 32;

struct alignas(1) fp8e4m3fn {
    struct from_bits_t {
    };
    __host__ __device__ static constexpr from_bits_t from_bits() {
        return from_bits_t();
    }
    uint8_t data;

    fp8e4m3fn() = default;
    __host__ __device__ constexpr fp8e4m3fn(const fp8e4m3fn &) = default;
    __host__ __device__ constexpr fp8e4m3fn(uint8_t v) = delete;
    explicit __host__ __device__ constexpr fp8e4m3fn(uint8_t v, from_bits_t) :
        data(v) {
    }

    explicit __host__ __device__ fp8e4m3fn(float v) {
        data = hip_fp8_impl::to_float8<4, 3, float, false /*negative_zero_nan*/, true /*clip*/>(v);
    }

    explicit __host__ __device__ fp8e4m3fn(double v) :
        fp8e4m3fn(static_cast<float>(v)) {
    }

    explicit inline __host__ __device__ operator float() const {
        return hip_fp8_impl::from_float8<4, 3, float, false /*negative_zero_nan*/>(data);
    }
};

struct alignas(1) fp8e4m3fnuz {
    struct from_bits_t {
    };
    __host__ __device__ static constexpr from_bits_t from_bits() {
        return from_bits_t();
    }
    uint8_t data;

    fp8e4m3fnuz() = default;
    __host__ __device__ constexpr fp8e4m3fnuz(const fp8e4m3fnuz&) = default;
    __host__ __device__ constexpr fp8e4m3fnuz(uint8_t v) = delete;
    explicit __host__ __device__ constexpr fp8e4m3fnuz(uint8_t v, from_bits_t) : data(v) {}

    explicit __host__ __device__ fp8e4m3fnuz(float v) {
        data = hip_fp8_impl::to_float8<4, 3, float, true /*negative_zero_nan*/, true /*clip*/>(v);
    }

    explicit __host__ __device__ fp8e4m3fnuz(double v) : fp8e4m3fnuz(static_cast<float>(v)) {}

    explicit inline __host__ __device__ operator float() const {
        return hip_fp8_impl::from_float8<4, 3, float, true /*negative_zero_nan*/>(data);
    }
};

template <typename T, int VEC_SIZE, typename OutT>
__device__ __forceinline__ vec_t<OutT, VEC_SIZE> convert_to(vec_t<T, VEC_SIZE>& in_vec, float scale) {
    vec_t<OutT, VEC_SIZE> out_vec;
#pragma unroll
    for(int i = 0; i < VEC_SIZE; ++i)
    {
        float out  = ck_tile::type_convert<float>(in_vec[i]) / scale;
        out_vec[i] = ck_tile::type_convert<OutT>(out);
    }
    return out_vec;
}

template <typename T, int HEAD_SIZE, bool IS_MROPE, bool IS_INTERLEAVED, int M, typename KVT = T>
__global__ void fused_mrope_rms_neox_kv_kernel(T* qkv,
                                               const T* q_w,
                                               const T* k_w,
                                               const T* cos_sin,
                                               const int64_t* positions,
                                               int64_t ps0,
                                               int64_t ps1,
                                               int num_heads_q,
                                               int num_heads_k,
                                               int num_heads_v,
                                               double eps,
                                               std::array<int64_t, M> mrope_section,
                                               int num_tokens,
                                               int total_warps,
                                               T* q                    = nullptr,
                                               KVT* k_cache            = nullptr,
                                               KVT* v_cache            = nullptr,
                                               int64_t* kv_loc         = nullptr,
                                               float k_scale           = 1.0,
                                               float v_scale           = 1.0,
                                               KVT* k_out              = nullptr,
                                               KVT* v_out              = nullptr,
                                               bool return_kv          = false,
                                               bool use_shuffle_layout = false,
                                               int block_size          = 0,
                                               int x                   = 0)
{
    constexpr int VEC_SIZE        = HEAD_SIZE / WARP_SIZE;
    constexpr int HALF_HEAD_SIZE  = HEAD_SIZE / 2;
    const int warp_id             = threadIdx.x / WARP_SIZE;
    const int num_warps_per_block = blockDim.x / WARP_SIZE;
    const int global_warp_id      = blockIdx.x * num_warps_per_block + warp_id;
    if(global_warp_id >= total_warps)
    {
        return;
    }

    // Warp allocation: all Q first, then all K, then all V
    const int num_heads_qk   = num_heads_q + num_heads_k;
    const int num_heads      = num_heads_q + num_heads_k + num_heads_v;
    const int total_q_warps  = num_tokens * num_heads_q;
    const int total_k_warps  = num_tokens * num_heads_k;
    const int total_qk_warps = total_q_warps + total_k_warps;

    // Determine if current warp processes Q, K, or V
    const bool is_q = global_warp_id < total_q_warps;
    const bool is_k = !is_q && global_warp_id < total_qk_warps;
    const bool is_v = global_warp_id >= total_qk_warps;

    int token_id, head_id_in_token;

    if(is_q)
    {
        // Q warps: global_warp_id in range [0, total_q_warps)
        token_id         = global_warp_id / num_heads_q;
        head_id_in_token = global_warp_id % num_heads_q;
    }
    else if(is_k)
    {
        // K warps: global_warp_id in range [total_q_warps, total_qk_warps)
        const int k_warp_id = global_warp_id - total_q_warps;
        token_id            = k_warp_id / num_heads_k;
        head_id_in_token    = num_heads_q + (k_warp_id % num_heads_k);
    }
    else
    {
        // V warps: global_warp_id in range [total_qk_warps, total_warps)
        const int v_warp_id = global_warp_id - total_qk_warps;
        token_id            = v_warp_id / num_heads_v;
        head_id_in_token    = num_heads_qk + (v_warp_id % num_heads_v);
    }

    const int access_id_in_head = (threadIdx.x % WARP_SIZE) * VEC_SIZE;
    const int neighbor_offset =
        access_id_in_head < HALF_HEAD_SIZE ? HALF_HEAD_SIZE / VEC_SIZE : -HALF_HEAD_SIZE / VEC_SIZE;
    T* qkv_ = qkv + (static_cast<int64_t>(token_id) * num_heads + head_id_in_token) * HEAD_SIZE;

    if(!is_v)
    {
        vec_t<T, VEC_SIZE> w_vec;
        if(is_q)
        {
            w_vec.load(q_w + access_id_in_head);
        }
        else
        {
            w_vec.load(k_w + access_id_in_head);
        }
        vec_t<T, VEC_SIZE> x_vec, cos_sin_vec;
        x_vec.load(qkv_ + access_id_in_head);
        if constexpr(IS_MROPE)
        {
            mrope_load_cos_sin_vec<T, VEC_SIZE, HEAD_SIZE, IS_INTERLEAVED, M>(cos_sin_vec,
                                                                              cos_sin,
                                                                              positions,
                                                                              ps0,
                                                                              ps1,
                                                                              token_id,
                                                                              num_tokens,
                                                                              access_id_in_head,
                                                                              mrope_section);
        }
        else
        {
            auto position_ = positions[token_id * ps1];
            cos_sin_vec.load(&cos_sin[position_ * HEAD_SIZE + access_id_in_head]);
        }
        warp_rms_norm_<T, VEC_SIZE>(x_vec, w_vec, HEAD_SIZE, eps);
        auto nb_cos_sin_vec =
            warp_shfl_sync_vec<T, VEC_SIZE>(cos_sin_vec, threadIdx.x + neighbor_offset);
        auto nb_x_vec = warp_shfl_sync_vec<T, VEC_SIZE>(x_vec, threadIdx.x + neighbor_offset);
        vec_t<T, VEC_SIZE> out_vec;
        if(neighbor_offset > 0)
        {
#pragma unroll
            for(int i = 0; i < VEC_SIZE; ++i)
            {
                out_vec[i] = (float)x_vec[i] * (float)cos_sin_vec[i] -
                             (float)nb_x_vec[i] * (float)nb_cos_sin_vec[i]; // x0 * cos - x1 * sin
            }
        }
        else
        {
#pragma unroll
            for(int i = 0; i < VEC_SIZE; ++i)
            {
                out_vec[i] = (float)x_vec[i] * (float)nb_cos_sin_vec[i] +
                             (float)nb_x_vec[i] * (float)cos_sin_vec[i]; // x1 * cos + x0 * sin
            }
        }
        if(is_q)
        {
            T* q_ =
                q + (static_cast<int64_t>(token_id) * num_heads_q + head_id_in_token) * HEAD_SIZE;
            out_vec.store(q_ + access_id_in_head);
        }
        else
        {
            const int64_t slot_id = kv_loc[token_id];
            if(slot_id < 0)
            {
                return;
            }
            const int head_id_kv = head_id_in_token - num_heads_q;
            if(use_shuffle_layout)
            {
                // Shuffle layout: [num_blocks, num_kv_heads, head_size // x, block_size, x]
                const int block_id      = static_cast<int>(slot_id / block_size);
                const int block_offset  = static_cast<int>(slot_id % block_size);
                const int k_head_stride = HEAD_SIZE * block_size;
                const int64_t dst_base =
                    static_cast<int64_t>(block_id) * num_heads_k * k_head_stride +
                    head_id_kv * k_head_stride;

                // Pre-compute K base offset: since VEC_SIZE <= x, all elements are in the same
                // chunk
                const int chunk_id     = access_id_in_head / x;
                const int block_size_x = block_size * x;
                const int64_t k_base =
                    dst_base + chunk_id * block_size_x + block_offset * x + (access_id_in_head % x);

                // For key: all elements are contiguous
                if constexpr(std::is_same_v<T, KVT>)
                {
#pragma unroll
                    for(int i = 0; i < VEC_SIZE; ++i)
                    {
                        k_cache[k_base + i] = out_vec[i];
                    }
                    if(return_kv && k_out != nullptr)
                    {
                        const int64_t k_out_offset =
                            (static_cast<int64_t>(token_id) * num_heads_k + head_id_kv) *
                                HEAD_SIZE +
                            access_id_in_head;
                        out_vec.store(k_out + k_out_offset);
                    }
                }
                else
                {
                    auto out_vec_fp8 = convert_to<T, VEC_SIZE, KVT>(out_vec, k_scale);
#pragma unroll
                    for(int i = 0; i < VEC_SIZE; ++i)
                    {
                        k_cache[k_base + i] = out_vec_fp8[i];
                    }
                    if(return_kv && k_out != nullptr)
                    {
                        const int64_t k_out_offset =
                            (static_cast<int64_t>(token_id) * num_heads_k + head_id_kv) *
                                HEAD_SIZE +
                            access_id_in_head;
                        out_vec_fp8.store(k_out + k_out_offset);
                    }
                }
            }
            else
            {
                // Normal layout: [num_slots, num_kv_heads, head_size]
                const int64_t offset =
                    (slot_id * num_heads_k + head_id_kv) * HEAD_SIZE + access_id_in_head;
                if constexpr(std::is_same_v<T, KVT>)
                {
                    out_vec.store(k_cache + offset);
                    if(return_kv && k_out != nullptr)
                    {
                        const int64_t k_out_offset =
                            (static_cast<int64_t>(token_id) * num_heads_k + head_id_kv) *
                                HEAD_SIZE +
                            access_id_in_head;
                        out_vec.store(k_out + k_out_offset);
                    }
                }
                else
                {
                    auto out_vec_fp8 = convert_to<T, VEC_SIZE, KVT>(out_vec, k_scale);
                    out_vec_fp8.store(k_cache + offset);
                    if(return_kv && k_out != nullptr)
                    {
                        const int64_t k_out_offset =
                            (static_cast<int64_t>(token_id) * num_heads_k + head_id_kv) *
                                HEAD_SIZE +
                            access_id_in_head;
                        out_vec_fp8.store(k_out + k_out_offset);
                    }
                }
            }
        }
    }
    else
    {
        vec_t<T, VEC_SIZE> v_vec;
        v_vec.load(qkv_ + access_id_in_head);
        const int64_t slot_id = kv_loc[token_id];
        if(slot_id < 0)
        {
            return;
        }
        const int head_id_kv = head_id_in_token - num_heads_qk;
        if(use_shuffle_layout)
        {
            // Shuffle layout: [num_blocks, num_kv_heads, block_size // x, head_size, x]
            const int block_id      = static_cast<int>(slot_id / block_size);
            const int block_offset  = static_cast<int>(slot_id % block_size);
            const int v_head_stride = (block_size / x) * HEAD_SIZE * x;
            const int64_t dst_base  = static_cast<int64_t>(block_id) * num_heads_v * v_head_stride +
                                     head_id_kv * v_head_stride;
            // Pre-compute V base offset (fixed for this token)
            const int v_slot_chunk    = block_offset / x;
            const int v_slot_in_chunk = block_offset % x;
            const int64_t v_base      = dst_base + v_slot_chunk * HEAD_SIZE * x + v_slot_in_chunk;
            // For value: shuffle within each block_size chunk
            if constexpr(std::is_same_v<T, KVT>)
            {
#pragma unroll
                for(int i = 0; i < VEC_SIZE; ++i)
                {
                    const int offset_in_head             = access_id_in_head + i;
                    v_cache[v_base + offset_in_head * x] = v_vec[i];
                }
                if(return_kv && v_out != nullptr)
                {
                    const int64_t v_out_offset =
                        (static_cast<int64_t>(token_id) * num_heads_v + head_id_kv) * HEAD_SIZE +
                        access_id_in_head;
                    v_vec.store(v_out + v_out_offset);
                }
            }
            else
            {
                auto v_vec_fp8 = convert_to<T, VEC_SIZE, KVT>(v_vec, v_scale);
#pragma unroll
                for(int i = 0; i < VEC_SIZE; ++i)
                {
                    const int offset_in_head             = access_id_in_head + i;
                    v_cache[v_base + offset_in_head * x] = v_vec_fp8[i];
                }
                if(return_kv && v_out != nullptr)
                {
                    const int64_t v_out_offset =
                        (static_cast<int64_t>(token_id) * num_heads_v + head_id_kv) * HEAD_SIZE +
                        access_id_in_head;
                    v_vec_fp8.store(v_out + v_out_offset);
                }
            }
        }
        else
        {
            // Normal layout: [num_slots, num_kv_heads, head_size]
            const int64_t offset =
                (slot_id * num_heads_v + head_id_kv) * HEAD_SIZE + access_id_in_head;
            if constexpr(std::is_same_v<T, KVT>)
            {
                v_vec.store(v_cache + offset);
                if(return_kv && v_out != nullptr)
                {
                    const int64_t v_out_offset =
                        (static_cast<int64_t>(token_id) * num_heads_v + head_id_kv) * HEAD_SIZE +
                        access_id_in_head;
                    v_vec.store(v_out + v_out_offset);
                }
            }
            else
            {
                auto v_vec_fp8 = convert_to<T, VEC_SIZE, KVT>(v_vec, v_scale);
                v_vec_fp8.store(v_cache + offset);
                if(return_kv && v_out != nullptr)
                {
                    const int64_t v_out_offset =
                        (static_cast<int64_t>(token_id) * num_heads_v + head_id_kv) * HEAD_SIZE +
                        access_id_in_head;
                    v_vec_fp8.store(v_out + v_out_offset);
                }
            }
        }
    }
}

template <typename T, int HEAD_SIZE, bool IS_MROPE, bool IS_INTERLEAVED, int M, typename KVT = T>
__global__ void fused_mrope_rms_noneox_kv_kernel(T* qkv,
                                                 const T* q_w,
                                                 const T* k_w,
                                                 const T* cos_sin,
                                                 const int64_t* positions,
                                                 int64_t ps0,
                                                 int64_t ps1,
                                                 int num_heads_q,
                                                 int num_heads_k,
                                                 int num_heads_v,
                                                 double eps,
                                                 std::array<int64_t, M> mrope_section,
                                                 int num_tokens,
                                                 int total_warps,
                                                 T* q                    = nullptr,
                                                 KVT* k_cache            = nullptr,
                                                 KVT* v_cache            = nullptr,
                                                 int64_t* kv_loc         = nullptr,
                                                 float k_scale           = 1.0,
                                                 float v_scale           = 1.0,
                                                 KVT* k_out              = nullptr,
                                                 KVT* v_out              = nullptr,
                                                 bool return_kv          = false,
                                                 bool use_shuffle_layout = false,
                                                 int block_size          = 0,
                                                 int x                   = 0)
{
    constexpr int VEC_SIZE        = HEAD_SIZE / WARP_SIZE;
    constexpr int HALF_HEAD_SIZE  = HEAD_SIZE / 2;
    const int warp_id             = threadIdx.x / WARP_SIZE;
    const int num_warps_per_block = blockDim.x / WARP_SIZE;
    const int global_warp_id      = blockIdx.x * num_warps_per_block + warp_id;
    if(global_warp_id >= total_warps)
    {
        return;
    }
    // Warp allocation: all Q first, then all K, then all V
    const int num_heads_qk   = num_heads_q + num_heads_k;
    const int num_heads      = num_heads_q + num_heads_k + num_heads_v;
    const int total_q_warps  = num_tokens * num_heads_q;
    const int total_k_warps  = num_tokens * num_heads_k;
    const int total_qk_warps = total_q_warps + total_k_warps;

    // Determine if current warp processes Q, K, or V
    const bool is_q = global_warp_id < total_q_warps;
    const bool is_k = !is_q && global_warp_id < total_qk_warps;
    const bool is_v = global_warp_id >= total_qk_warps;

    int token_id, head_id_in_token;

    if(is_q)
    {
        // Q warps: global_warp_id in range [0, total_q_warps)
        token_id        = global_warp_id / num_heads_q;
        head_id_in_token = global_warp_id % num_heads_q;
    }
    else if(is_k)
    {
        // K warps: global_warp_id in range [total_q_warps, total_qk_warps)
        const int k_warp_id = global_warp_id - total_q_warps;
        token_id            = k_warp_id / num_heads_k;
        head_id_in_token    = num_heads_q + (k_warp_id % num_heads_k);
    }
    else
    {
        // V warps: global_warp_id in range [total_qk_warps, total_warps)
        const int v_warp_id = global_warp_id - total_qk_warps;
        token_id            = v_warp_id / num_heads_v;
        head_id_in_token    = num_heads_qk + (v_warp_id % num_heads_v);
    }

    const int access_id_in_head = (threadIdx.x % WARP_SIZE) * VEC_SIZE;
    T* qkv_ = qkv + (static_cast<int64_t>(token_id) * num_heads + head_id_in_token) * HEAD_SIZE;

    if(!is_v)
    {
        vec_t<T, VEC_SIZE> w_vec;
        if(is_q)
        {
            w_vec.load(q_w + access_id_in_head);
        }
        else
        {
            w_vec.load(k_w + access_id_in_head);
        }
        vec_t<T, VEC_SIZE> x_vec, cos_vec, sin_vec;
        x_vec.load(qkv_ + access_id_in_head);
        if constexpr(IS_MROPE)
        {
            mrope_load_cos_sin_vec<T, VEC_SIZE, HEAD_SIZE, IS_INTERLEAVED, M>(cos_vec,
                                                                              cos_sin,
                                                                              positions,
                                                                              ps0,
                                                                              ps1,
                                                                              token_id,
                                                                              num_tokens,
                                                                              access_id_in_head / 2,
                                                                              mrope_section);
            mrope_load_cos_sin_vec<T, VEC_SIZE, HEAD_SIZE, IS_INTERLEAVED, M>(
                sin_vec,
                cos_sin,
                positions,
                ps0,
                ps1,
                token_id,
                num_tokens,
                access_id_in_head / 2 + HALF_HEAD_SIZE,
                mrope_section);
        }
        else
        {
            auto position_ = positions[token_id * ps1];
            cos_vec.load(&cos_sin[position_ * HEAD_SIZE + access_id_in_head / 2]);
            sin_vec.load(&cos_sin[position_ * HEAD_SIZE + access_id_in_head / 2 + HALF_HEAD_SIZE]);
        }
        warp_rms_norm_<T, VEC_SIZE>(x_vec, w_vec, HEAD_SIZE, eps);
        vec_t<T, VEC_SIZE> out_vec;
#pragma unroll
        for(int i = 0; i < VEC_SIZE / 2; ++i)
        {
            out_vec[2 * i + 0] = (float)x_vec[2 * i + 0] * (float)cos_vec[i] -
                                 (float)x_vec[2 * i + 1] * (float)sin_vec[i];
            out_vec[2 * i + 1] = (float)x_vec[2 * i + 1] * (float)cos_vec[i] +
                                 (float)x_vec[2 * i + 0] * (float)sin_vec[i];
        }
        if(is_q)
        {
            T* q_ =
                q + (static_cast<int64_t>(token_id) * num_heads_q + head_id_in_token) * HEAD_SIZE;
            out_vec.store(q_ + access_id_in_head);
        }
        else
        {
            const int64_t slot_id = kv_loc[token_id];
            if(slot_id < 0)
            {
                return;
            }
            const int head_id_kv = head_id_in_token - num_heads_q;
            if(use_shuffle_layout)
            {
                // Shuffle layout: [num_blocks, num_kv_heads, head_size // x, block_size, x]
                const int block_id      = static_cast<int>(slot_id / block_size);
                const int block_offset  = static_cast<int>(slot_id % block_size);
                const int k_head_stride = HEAD_SIZE * block_size;
                const int64_t dst_base =
                    static_cast<int64_t>(block_id) * num_heads_k * k_head_stride +
                    head_id_kv * k_head_stride;

                // Pre-compute K base offset: since VEC_SIZE <= x, all elements are in the same
                // chunk
                const int chunk_id     = access_id_in_head / x;
                const int block_size_x = block_size * x;
                const int64_t k_base =
                    dst_base + chunk_id * block_size_x + block_offset * x + (access_id_in_head % x);

                // For key: all elements are contiguous
                if constexpr(std::is_same_v<T, KVT>)
                {
#pragma unroll
                    for(int i = 0; i < VEC_SIZE; ++i)
                    {
                        k_cache[k_base + i] = out_vec[i];
                    }
                    if(return_kv && k_out != nullptr)
                    {
                        const int64_t k_out_offset =
                            (static_cast<int64_t>(token_id) * num_heads_k + head_id_kv) *
                                HEAD_SIZE +
                            access_id_in_head;
                        out_vec.store(k_out + k_out_offset);
                    }
                }
                else
                {
                    auto out_vec_fp8 = convert_to<T, VEC_SIZE, KVT>(out_vec, k_scale);
#pragma unroll
                    for(int i = 0; i < VEC_SIZE; ++i)
                    {
                        k_cache[k_base + i] = out_vec_fp8[i];
                    }
                    if(return_kv && k_out != nullptr)
                    {
                        const int64_t k_out_offset =
                            (static_cast<int64_t>(token_id) * num_heads_k + head_id_kv) *
                                HEAD_SIZE +
                            access_id_in_head;
                        out_vec_fp8.store(k_out + k_out_offset);
                    }
                }
            }
            else
            {
                // Normal layout: [num_slots, num_kv_heads, head_size]
                const int64_t offset =
                    (slot_id * num_heads_k + head_id_kv) * HEAD_SIZE + access_id_in_head;
                if constexpr(std::is_same_v<T, KVT>)
                {
                    out_vec.store(k_cache + offset);
                    if(return_kv && k_out != nullptr)
                    {
                        const int64_t k_out_offset =
                            (static_cast<int64_t>(token_id) * num_heads_k + head_id_kv) *
                                HEAD_SIZE +
                            access_id_in_head;
                        out_vec.store(k_out + k_out_offset);
                    }
                }
                else
                {
                    auto out_vec_fp8 = convert_to<T, VEC_SIZE, KVT>(out_vec, k_scale);
                    out_vec_fp8.store(k_cache + offset);
                    if(return_kv && k_out != nullptr)
                    {
                        const int64_t k_out_offset =
                            (static_cast<int64_t>(token_id) * num_heads_k + head_id_kv) *
                                HEAD_SIZE +
                            access_id_in_head;
                        out_vec_fp8.store(k_out + k_out_offset);
                    }
                }
            }
        }
    }
    else
    {
        vec_t<T, VEC_SIZE> v_vec;
        v_vec.load(qkv_ + access_id_in_head);
        const int64_t slot_id = kv_loc[token_id];
        if(slot_id < 0)
        {
            return;
        }
        const int head_id_kv = head_id_in_token - num_heads_qk;
        if(use_shuffle_layout)
        {
            // Shuffle layout: [num_blocks, num_kv_heads, block_size // x, head_size, x]
            const int block_id      = static_cast<int>(slot_id / block_size);
            const int block_offset  = static_cast<int>(slot_id % block_size);
            const int v_head_stride = (block_size / x) * HEAD_SIZE * x;
            const int64_t dst_base  = static_cast<int64_t>(block_id) * num_heads_v * v_head_stride +
                                     head_id_kv * v_head_stride;
            // Pre-compute V base offset (fixed for this token)
            const int v_slot_chunk    = block_offset / x;
            const int v_slot_in_chunk = block_offset % x;
            const int64_t v_base      = dst_base + v_slot_chunk * HEAD_SIZE * x + v_slot_in_chunk;
            // For value: shuffle within each block_size chunk
            if constexpr(std::is_same_v<T, KVT>)
            {
#pragma unroll
                for(int i = 0; i < VEC_SIZE; ++i)
                {
                    const int offset_in_head             = access_id_in_head + i;
                    v_cache[v_base + offset_in_head * x] = v_vec[i];
                }
                if(return_kv && v_out != nullptr)
                {
                    const int64_t v_out_offset =
                        (static_cast<int64_t>(token_id) * num_heads_v + head_id_kv) * HEAD_SIZE +
                        access_id_in_head;
                    v_vec.store(v_out + v_out_offset);
                }
            }
            else
            {
                auto v_vec_fp8 = convert_to<T, VEC_SIZE, KVT>(v_vec, v_scale);
#pragma unroll
                for(int i = 0; i < VEC_SIZE; ++i)
                {
                    const int offset_in_head             = access_id_in_head + i;
                    v_cache[v_base + offset_in_head * x] = v_vec_fp8[i];
                }
                if(return_kv && v_out != nullptr)
                {
                    const int64_t v_out_offset =
                        (static_cast<int64_t>(token_id) * num_heads_v + head_id_kv) * HEAD_SIZE +
                        access_id_in_head;
                    v_vec_fp8.store(v_out + v_out_offset);
                }
            }
        }
        else
        {
            // Normal layout: [num_slots, num_kv_heads, head_size]
            const int64_t offset =
                (slot_id * num_heads_v + head_id_kv) * HEAD_SIZE + access_id_in_head;
            if constexpr(std::is_same_v<T, KVT>)
            {
                v_vec.store(v_cache + offset);
                if(return_kv && v_out != nullptr)
                {
                    const int64_t v_out_offset =
                        (static_cast<int64_t>(token_id) * num_heads_v + head_id_kv) * HEAD_SIZE +
                        access_id_in_head;
                    v_vec.store(v_out + v_out_offset);
                }
            }
            else
            {
                auto v_vec_fp8 = convert_to<T, VEC_SIZE, KVT>(v_vec, v_scale);
                v_vec_fp8.store(v_cache + offset);
                if(return_kv && v_out != nullptr)
                {
                    const int64_t v_out_offset =
                        (static_cast<int64_t>(token_id) * num_heads_v + head_id_kv) * HEAD_SIZE +
                        access_id_in_head;
                    v_vec_fp8.store(v_out + v_out_offset);
                }
            }
        }
    }
}

template <typename T, int M, typename KVT>
void fused_mrope_rms_set_kv(T* qkv,
                            const T* q_w,
                            const T* k_w,
                            const T* cos_sin,
                            const int64_t* positions,
                            int64_t ps0,
                            int64_t ps1,
                            int64_t num_tokens,
                            int64_t num_heads_q,
                            int64_t num_heads_k,
                            int64_t num_heads_v,
                            int64_t head_size,
                            bool is_neox_style,
                            double eps,
                            std::array<int64_t, M> mrope_section,
                            bool is_interleaved,
                            T* q,
                            KVT* k_cache,
                            KVT* v_cache,
                            int64_t* kv_loc,
                            float k_scale,
                            float v_scale,
                            hipStream_t stream,
                            KVT* k_out              = nullptr,
                            KVT* v_out              = nullptr,
                            bool return_kv          = false,
                            bool use_shuffle_layout = false,
                            int64_t block_size      = 0,
                            int64_t x               = 0)
{
    TORCH_CHECK(head_size == 64 || head_size == 128 || head_size == 256);
    auto dim = std::accumulate(mrope_section.begin(), mrope_section.end(), 0);
    TORCH_CHECK(dim == head_size / 2);
    constexpr int THREAD_BLOCK_SIZE = 256;
    auto total_warps                = num_tokens * (num_heads_q + num_heads_k + num_heads_v);
    auto num_warps_per_block        = THREAD_BLOCK_SIZE / WARP_SIZE;
    dim3 threadsPerBlock(THREAD_BLOCK_SIZE);
    dim3 numBlocks((total_warps + num_warps_per_block - 1) / num_warps_per_block);

#define DISPATCH_NEOX(HEAD_SIZE, IS_INTERLEAVED)                                     \
    if(is_neox_style)                                                                \
    {                                                                                \
        fused_mrope_rms_neox_kv_kernel<T, HEAD_SIZE, true, IS_INTERLEAVED, M, KVT>   \
            <<<numBlocks, threadsPerBlock, 0, stream>>>(qkv,                         \
                                                        q_w,                         \
                                                        k_w,                         \
                                                        cos_sin,                     \
                                                        positions,                   \
                                                        ps0,                         \
                                                        ps1,                         \
                                                        num_heads_q,                 \
                                                        num_heads_k,                 \
                                                        num_heads_v,                 \
                                                        eps,                         \
                                                        mrope_section,               \
                                                        num_tokens,                  \
                                                        total_warps,                 \
                                                        q,                           \
                                                        k_cache,                     \
                                                        v_cache,                     \
                                                        kv_loc,                      \
                                                        k_scale,                     \
                                                        v_scale,                     \
                                                        k_out,                       \
                                                        v_out,                       \
                                                        return_kv,                   \
                                                        use_shuffle_layout,          \
                                                        block_size,                  \
                                                        x);                          \
    }                                                                                \
    else                                                                             \
    {                                                                                \
        fused_mrope_rms_noneox_kv_kernel<T, HEAD_SIZE, true, IS_INTERLEAVED, M, KVT> \
            <<<numBlocks, threadsPerBlock, 0, stream>>>(qkv,                         \
                                                        q_w,                         \
                                                        k_w,                         \
                                                        cos_sin,                     \
                                                        positions,                   \
                                                        ps0,                         \
                                                        ps1,                         \
                                                        num_heads_q,                 \
                                                        num_heads_k,                 \
                                                        num_heads_v,                 \
                                                        eps,                         \
                                                        mrope_section,               \
                                                        num_tokens,                  \
                                                        total_warps,                 \
                                                        q,                           \
                                                        k_cache,                     \
                                                        v_cache,                     \
                                                        kv_loc,                      \
                                                        k_scale,                     \
                                                        v_scale,                     \
                                                        k_out,                       \
                                                        v_out,                       \
                                                        return_kv,                   \
                                                        use_shuffle_layout,          \
                                                        block_size,                  \
                                                        x);                          \
    }

    if(is_interleaved)
    {
        switch(head_size)
        {
        case 64: DISPATCH_NEOX(64, true) break;
        case 128: DISPATCH_NEOX(128, true) break;
        case 256: DISPATCH_NEOX(256, true) break;
        }
    }
    else
    {
        switch(head_size)
        {
        case 64: DISPATCH_NEOX(64, false) break;
        case 128: DISPATCH_NEOX(128, false) break;
        case 256: DISPATCH_NEOX(256, false) break;
        }
    }

#undef DISPATCH_NEOX
}

template <typename T, typename KVT>
void fused_rope_rms_set_kv(T* qkv,
                           const T* q_w,
                           const T* k_w,
                           const T* cos_sin,
                           const int64_t* positions,
                           int64_t ps0,
                           int64_t ps1,
                           int64_t num_tokens,
                           int64_t num_heads_q,
                           int64_t num_heads_k,
                           int64_t num_heads_v,
                           int64_t head_size,
                           bool is_neox_style,
                           double eps,
                           T* q,
                           KVT* k_cache,
                           KVT* v_cache,
                           int64_t* kv_loc,
                           float k_scale,
                           float v_scale,
                           hipStream_t stream,
                           KVT* k_out              = nullptr,
                           KVT* v_out              = nullptr,
                           bool return_kv          = false,
                           bool use_shuffle_layout = false,
                           int64_t block_size      = 0,
                           int64_t x               = 0)
{
    TORCH_CHECK(head_size == 64 || head_size == 128 || head_size == 256);
    constexpr int THREAD_BLOCK_SIZE = 256;
    auto total_warps                = num_tokens * (num_heads_q + num_heads_k + num_heads_v);
    auto num_warps_per_block        = THREAD_BLOCK_SIZE / WARP_SIZE;
    dim3 threadsPerBlock(THREAD_BLOCK_SIZE);
    dim3 numBlocks((total_warps + num_warps_per_block - 1) / num_warps_per_block);
    std::array<int64_t, 1> mrope_section = {0};

#define DISPATCH_NEOX(HEAD_SIZE)                                             \
    if(is_neox_style)                                                        \
    {                                                                        \
        fused_mrope_rms_neox_kv_kernel<T, HEAD_SIZE, false, false, 1, KVT>   \
            <<<numBlocks, threadsPerBlock, 0, stream>>>(qkv,                 \
                                                        q_w,                 \
                                                        k_w,                 \
                                                        cos_sin,             \
                                                        positions,           \
                                                        ps0,                 \
                                                        ps1,                 \
                                                        num_heads_q,         \
                                                        num_heads_k,         \
                                                        num_heads_v,         \
                                                        eps,                 \
                                                        mrope_section,       \
                                                        num_tokens,          \
                                                        total_warps,         \
                                                        q,                   \
                                                        k_cache,             \
                                                        v_cache,             \
                                                        kv_loc,              \
                                                        k_scale,             \
                                                        v_scale,             \
                                                        k_out,               \
                                                        v_out,               \
                                                        return_kv,           \
                                                        use_shuffle_layout,  \
                                                        block_size,          \
                                                        x);                  \
    }                                                                        \
    else                                                                     \
    {                                                                        \
        fused_mrope_rms_noneox_kv_kernel<T, HEAD_SIZE, false, false, 1, KVT> \
            <<<numBlocks, threadsPerBlock, 0, stream>>>(qkv,                 \
                                                        q_w,                 \
                                                        k_w,                 \
                                                        cos_sin,             \
                                                        positions,           \
                                                        ps0,                 \
                                                        ps1,                 \
                                                        num_heads_q,         \
                                                        num_heads_k,         \
                                                        num_heads_v,         \
                                                        eps,                 \
                                                        mrope_section,       \
                                                        num_tokens,          \
                                                        total_warps,         \
                                                        q,                   \
                                                        k_cache,             \
                                                        v_cache,             \
                                                        kv_loc,              \
                                                        k_scale,             \
                                                        v_scale,             \
                                                        k_out,               \
                                                        v_out,               \
                                                        return_kv,           \
                                                        use_shuffle_layout,  \
                                                        block_size,          \
                                                        x);                  \
    }

    switch(head_size)
    {
    case 64: DISPATCH_NEOX(64) break;
    case 128: DISPATCH_NEOX(128) break;
    case 256: DISPATCH_NEOX(256) break;
    }

#undef DISPATCH_NEOX
}

} // namespace rope_rms

template <typename T>
struct KernelElementType {
    using type = T;
};

template <>
struct KernelElementType<c10::Half> {
    using type = __half;
};

template <>
struct KernelElementType<c10::BFloat16> {
    using type = hip_bfloat16;
};

void fused_qk_norm_mrope_3d_cache_quant_shuffle(Tensor& qkv,
                               Tensor& qw,
                               Tensor& kw,
                               Tensor& cos_sin,
                               Tensor& positions,
                               int64_t num_tokens,
                               int64_t num_heads_q,
                               int64_t num_heads_k,
                               int64_t num_heads_v,
                               int64_t head_size,
                               bool is_neox_style,
                               std::vector<int64_t> mrope_section_,
                               bool is_interleaved,
                               double eps,
                               Tensor& q,
                               Tensor& k_cache,
                               Tensor& v_cache,
                               Tensor& kv_loc,
                               Tensor& k_scale,
                               Tensor& v_scale,
                               std::optional<Tensor> k_out,
                               std::optional<Tensor> v_out,
                               bool return_kv,
                               bool use_shuffle_layout,
                               int64_t block_size,
                               int64_t x)
{
    TORCH_CHECK(mrope_section_.size() == 3);
    TORCH_CHECK(qkv.is_contiguous() && qw.is_contiguous() && kw.is_contiguous() &&
                cos_sin.is_contiguous());
    TORCH_CHECK(k_cache.is_contiguous() && v_cache.is_contiguous() && kv_loc.is_contiguous());
    std::array<int64_t, 3> mrope_section;
    mrope_section[0] = mrope_section_[0];
    mrope_section[1] = mrope_section_[1];
    mrope_section[2] = mrope_section_[2];
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(qkv));
    auto stream         = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
    auto pos_strides    = positions.strides();
    auto kv_cache_dtype = k_cache.scalar_type();
    auto qkv_dtype      = qkv.scalar_type();
    TORCH_CHECK(pos_strides.size() == 2);
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, qkv_dtype, "fused_mrope_3d_rms_set_kv", [&] {
        using T = KernelElementType<scalar_t>::type;

        if(kv_cache_dtype == qkv_dtype)
        {
            T* k_out_ptr =
                (return_kv && k_out.has_value()) ? (T*)k_out.value().data_ptr<scalar_t>() : nullptr;
            T* v_out_ptr =
                (return_kv && v_out.has_value()) ? (T*)v_out.value().data_ptr<scalar_t>() : nullptr;
            rope_rms::fused_mrope_rms_set_kv<T, 3, T>((T*)qkv.data_ptr<scalar_t>(),
                                                      (T*)qw.data_ptr<scalar_t>(),
                                                      (T*)kw.data_ptr<scalar_t>(),
                                                      (T*)cos_sin.data_ptr<scalar_t>(),
                                                      positions.data_ptr<int64_t>(),
                                                      pos_strides[0],
                                                      pos_strides[1],
                                                      num_tokens,
                                                      num_heads_q,
                                                      num_heads_k,
                                                      num_heads_v,
                                                      head_size,
                                                      is_neox_style,
                                                      eps,
                                                      mrope_section,
                                                      is_interleaved,
                                                      (T*)q.data_ptr<scalar_t>(),
                                                      (T*)k_cache.data_ptr<scalar_t>(),
                                                      (T*)v_cache.data_ptr<scalar_t>(),
                                                      kv_loc.data_ptr<int64_t>(),
                                                      k_scale.item<float>(),
                                                      v_scale.item<float>(),
                                                      stream,
                                                      k_out_ptr,
                                                      v_out_ptr,
                                                      return_kv,
                                                      use_shuffle_layout,
                                                      block_size,
                                                      x);
        }
        else
        {
            // Check if kv_cache_dtype is fp8e4m3fnuz or fp8e4m3fn
            if(kv_cache_dtype == at::ScalarType::Float8_e4m3fnuz)
            {
                rope_rms::fp8e4m3fnuz* k_out_fp8_ptr =
                    (return_kv && k_out.has_value())
                        ? (rope_rms::fp8e4m3fnuz*)k_out.value().data_ptr()
                        : nullptr;
                rope_rms::fp8e4m3fnuz* v_out_fp8_ptr =
                    (return_kv && v_out.has_value())
                        ? (rope_rms::fp8e4m3fnuz*)v_out.value().data_ptr()
                        : nullptr;
                rope_rms::fused_mrope_rms_set_kv<T, 3, rope_rms::fp8e4m3fnuz>(
                    (T*)qkv.data_ptr<scalar_t>(),
                    (T*)qw.data_ptr<scalar_t>(),
                    (T*)kw.data_ptr<scalar_t>(),
                    (T*)cos_sin.data_ptr<scalar_t>(),
                    positions.data_ptr<int64_t>(),
                    pos_strides[0],
                    pos_strides[1],
                    num_tokens,
                    num_heads_q,
                    num_heads_k,
                    num_heads_v,
                    head_size,
                    is_neox_style,
                    eps,
                    mrope_section,
                    is_interleaved,
                    (T*)q.data_ptr<scalar_t>(),
                    (rope_rms::fp8e4m3fnuz*)k_cache.data_ptr(),
                    (rope_rms::fp8e4m3fnuz*)v_cache.data_ptr(),
                    kv_loc.data_ptr<int64_t>(),
                    k_scale.item<float>(),
                    v_scale.item<float>(),
                    stream,
                    k_out_fp8_ptr,
                    v_out_fp8_ptr,
                    return_kv,
                    use_shuffle_layout,
                    block_size,
                    x);
            }
            else if(kv_cache_dtype == at::ScalarType::Float8_e4m3fn)
            {
                rope_rms::fp8e4m3fn* k_out_fp8_ptr =
                    (return_kv && k_out.has_value())
                        ? (rope_rms::fp8e4m3fn*)k_out.value().data_ptr()
                        : nullptr;
                rope_rms::fp8e4m3fn* v_out_fp8_ptr =
                    (return_kv && v_out.has_value())
                        ? (rope_rms::fp8e4m3fn*)v_out.value().data_ptr()
                        : nullptr;
                rope_rms::fused_mrope_rms_set_kv<T, 3, rope_rms::fp8e4m3fn>(
                    (T*)qkv.data_ptr<scalar_t>(),
                    (T*)qw.data_ptr<scalar_t>(),
                    (T*)kw.data_ptr<scalar_t>(),
                    (T*)cos_sin.data_ptr<scalar_t>(),
                    positions.data_ptr<int64_t>(),
                    pos_strides[0],
                    pos_strides[1],
                    num_tokens,
                    num_heads_q,
                    num_heads_k,
                    num_heads_v,
                    head_size,
                    is_neox_style,
                    eps,
                    mrope_section,
                    is_interleaved,
                    (T*)q.data_ptr<scalar_t>(),
                    (rope_rms::fp8e4m3fn*)k_cache.data_ptr(),
                    (rope_rms::fp8e4m3fn*)v_cache.data_ptr(),
                    kv_loc.data_ptr<int64_t>(),
                    k_scale.item<float>(),
                    v_scale.item<float>(),
                    stream,
                    k_out_fp8_ptr,
                    v_out_fp8_ptr,
                    return_kv,
                    use_shuffle_layout,
                    block_size,
                    x);
            }
            else
            {
                TORCH_CHECK(false, "Unsupported KV cache dtype: ", kv_cache_dtype);
            }
        }
    });
}
