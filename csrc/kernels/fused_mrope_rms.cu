#include "rope/rope_common.h"

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

void fused_mrope_3d_rms(Tensor &qkv, Tensor &qw, Tensor &kw, Tensor &cos_sin, Tensor &positions,
                    int64_t num_tokens, int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v, int64_t head_size,
                    bool is_neox_style, std::vector<int64_t> mrope_section_, bool is_interleaved, double eps) {
    TORCH_CHECK(mrope_section_.size() == 3);
    TORCH_CHECK(qkv.is_contiguous() && qw.is_contiguous() && kw.is_contiguous() && cos_sin.is_contiguous());
    std::array<int64_t, 3> mrope_section;
    mrope_section[0] = mrope_section_[0];
    mrope_section[1] = mrope_section_[1];
    mrope_section[2] = mrope_section_[2];
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(qkv));
    auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
    auto pos_strides = positions.strides();
    TORCH_CHECK(pos_strides.size() == 2);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kBFloat16,
        kHalf,
        qkv.scalar_type(),
        "fused_mrope_3d_rms", [&] {
            using T = KernelElementType<scalar_t>::type;
            rope_rms::fused_mrope_rms<T, 3>(
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
                stream);
        });
}

void fused_rope_rms(Tensor &qkv, Tensor &qw, Tensor &kw, Tensor &cos_sin, Tensor &positions,
                    int64_t num_tokens, int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v, int64_t head_size,
                    bool is_neox_style, double eps) {
    TORCH_CHECK(qkv.is_contiguous() && qw.is_contiguous() && kw.is_contiguous() && cos_sin.is_contiguous());
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(qkv));
    auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
    auto pos_strides = positions.strides();
    TORCH_CHECK(pos_strides.size() == 1);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kBFloat16,
        kHalf,
        qkv.scalar_type(),
        "fused_rope_rms", [&] {
            using T = KernelElementType<scalar_t>::type;
            rope_rms::fused_rope_rms<T>(
                (T*)qkv.data_ptr<scalar_t>(),
                (T*)qw.data_ptr<scalar_t>(),
                (T*)kw.data_ptr<scalar_t>(),
                (T*)cos_sin.data_ptr<scalar_t>(),
                positions.data_ptr<int64_t>(),
                0,
                pos_strides[0],
                num_tokens,
                num_heads_q,
                num_heads_k,
                num_heads_v,
                head_size,
                is_neox_style,
                eps,
                stream);
        });
}
