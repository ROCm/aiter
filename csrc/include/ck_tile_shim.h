#pragma once
// ============================================================================
// ck_tile_shim.h — CK-free compatibility layer for AITER kernels
//
// When DISABLE_CK=1, this header replaces ck_tile/core.hpp.  It provides all
// ck_tile:: types, functions, and macros that the kernel sources and the local
// ck_tile/vec_convert.h header depend on.  Wherever an opus:: equivalent
// exists, we forward to it rather than re-implementing.
//
// Include chain:
//   kernel.cu  ->  #include "vec_convert.h"
//                    ->  #include "aiter_hip_common.h"
//                          ->  #include "ck_tile_shim.h"   (THIS FILE)
//                    ck_tile/vec_convert.h then defines:
//                      ck_tile::vec_t = ck_tile::thread_buffer
//                      ck_tile::fp4x2_t, ck_tile::vec_convert specialisations
//
// Therefore this file must NOT define vec_t, fp4x2_t, or vec_convert;
// those come from ck_tile/vec_convert.h after this header is included.
// ============================================================================

#include "opus/opus.hpp"
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cmath>
#include <type_traits>

// ============================================================================
// CK compatibility macros (used by ck_tile/vec_convert.h)
// ============================================================================
#ifndef CK_TILE_HOST_DEVICE
#define CK_TILE_HOST_DEVICE __host__ __device__
#endif
#ifndef CK_TILE_DEVICE
#define CK_TILE_DEVICE __device__
#endif
#ifndef CK_TILE_HOST
#define CK_TILE_HOST __host__
#endif

namespace ck_tile {

// ============================================================================
// 0. Type traits aliases
// ============================================================================
template <typename T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

// ============================================================================
// 1. Fundamental scalar types — forwarded from opus
// ============================================================================
using index_t      = opus::index_t;
using long_index_t = int64_t;

using fp8_t  = opus::fp8_t;
using bf8_t  = opus::bf8_t;
using fp16_t = opus::fp16_t;
using bf16_t = opus::bf16_t;
using fp32_t = opus::fp32_t;
using int8_t = ::int8_t;

// CK alternative names
using half_t     = fp16_t;
using bfloat16_t = bf16_t;

// ext_vector_type aliases used by ck_tile/vec_convert.h assembly helpers
using fp32x2_t  = opus::fp32x2_t;   // float __attribute__((ext_vector_type(2)))
using int16x2_t = short __attribute__((ext_vector_type(2)));
using int8x2_t  = signed char __attribute__((ext_vector_type(2)));

// ============================================================================
// 2. thread_buffer_view<U, M> — POD helper returned by thread_buffer::get_as<U>()
//    Supports operator()(i) and [] for element access.  Same byte layout as
//    U data_[M], so it can be memcpy'd into buffer_view::set().
// ============================================================================
template <typename U, int M>
struct thread_buffer_view {
    U data_[M];

    CK_TILE_HOST_DEVICE U& operator()(index_t i) { return data_[i]; }
    CK_TILE_HOST_DEVICE const U& operator()(index_t i) const { return data_[i]; }
    CK_TILE_HOST_DEVICE U& operator[](index_t i) { return data_[i]; }
    CK_TILE_HOST_DEVICE const U& operator[](index_t i) const { return data_[i]; }

    // get_as<V>() — reinterpret as different element type
    template <typename V>
    CK_TILE_HOST_DEVICE thread_buffer_view<V, (sizeof(U) * M / sizeof(V))>& get_as() {
        return *reinterpret_cast<thread_buffer_view<V, (sizeof(U) * M / sizeof(V))>*>(this);
    }
    template <typename V>
    CK_TILE_HOST_DEVICE const thread_buffer_view<V, (sizeof(U) * M / sizeof(V))>& get_as() const {
        return *reinterpret_cast<const thread_buffer_view<V, (sizeof(U) * M / sizeof(V))>*>(this);
    }
};

// ============================================================================
// 3. thread_buffer<T,N> — the array-based vector class
//    ck_tile/vec_convert.h aliases:  template<T,N> using vec_t = thread_buffer<T,N>;
// ============================================================================
template <typename T, int N>
struct thread_buffer {
    T data_[N];

    CK_TILE_HOST_DEVICE constexpr thread_buffer() : data_{} {}

    CK_TILE_HOST_DEVICE constexpr T& operator[](index_t i) { return data_[i]; }
    CK_TILE_HOST_DEVICE constexpr const T& operator[](index_t i) const { return data_[i]; }
    CK_TILE_HOST_DEVICE constexpr T& operator()(index_t i) { return data_[i]; }
    CK_TILE_HOST_DEVICE constexpr const T& operator()(index_t i) const { return data_[i]; }

    // get_as<U>() — reinterpret underlying bytes as elements of type U.
    // Returns thread_buffer_view<U, M>& (aliasing into this object's storage).
    template <typename U>
    CK_TILE_HOST_DEVICE thread_buffer_view<U, (sizeof(T) * N / sizeof(U))>& get_as() {
        static_assert(sizeof(T) * N % sizeof(U) == 0,
            "buffer size must be evenly divisible by target element size");
        return *reinterpret_cast<thread_buffer_view<U, (sizeof(T) * N / sizeof(U))>*>(this);
    }
    template <typename U>
    CK_TILE_HOST_DEVICE const thread_buffer_view<U, (sizeof(T) * N / sizeof(U))>& get_as() const {
        static_assert(sizeof(T) * N % sizeof(U) == 0,
            "buffer size must be evenly divisible by target element size");
        return *reinterpret_cast<const thread_buffer_view<U, (sizeof(T) * N / sizeof(U))>*>(this);
    }

    // Cross-type assignment
    template <typename U, int M>
    CK_TILE_HOST_DEVICE thread_buffer& operator=(const thread_buffer<U, M>& other) {
        constexpr int bytes = sizeof(thread_buffer) < sizeof(other) ? sizeof(thread_buffer) : sizeof(other);
        __builtin_memcpy(data_, &other, bytes);
        return *this;
    }
    template <typename U, int M>
    CK_TILE_HOST_DEVICE thread_buffer& operator=(const thread_buffer_view<U, M>& other) {
        constexpr int bytes = sizeof(thread_buffer) < sizeof(other) ? sizeof(thread_buffer) : sizeof(other);
        __builtin_memcpy(data_, &other, bytes);
        return *this;
    }
};

// ============================================================================
// 4. vector_traits<T> — base template (vec_convert.h specialises for fp4x2_t)
// ============================================================================
template <typename T>
struct vector_traits {
    using scalar_type = T;
    static constexpr index_t vector_size = 1;
};
template <typename T, int N>
struct vector_traits<thread_buffer<T, N>> {
    using scalar_type = T;
    static constexpr index_t vector_size = N;
};
// ext_vector_type specialization — required by topk_softmax_kernels_group.cu
template <typename T, int N>
struct vector_traits<T __attribute__((ext_vector_type(N)))> {
    using scalar_type = T;
    static constexpr index_t vector_size = N;
};

// ============================================================================
// 5. numeric<T> — forwarded to opus::numeric_limits<T>
// ============================================================================
template <typename T>
struct numeric {
    CK_TILE_HOST_DEVICE static constexpr T max()    { return opus::numeric_limits<T>::max(); }
    CK_TILE_HOST_DEVICE static constexpr T min()    { return opus::numeric_limits<T>::min(); }
    CK_TILE_HOST_DEVICE static constexpr T lowest() { return opus::numeric_limits<T>::lowest(); }
};

// ============================================================================
// 6. fp8 interpretation metadata (used by ck_tile/vec_convert.h)
// ============================================================================
// MUST match CK's enum ordering: OCP=0,1  FNUZ=2,3
enum class fp8_interpretation {
    E4M3_OCP  = 0,
    E5M2_OCP  = 1,
    E4M3_FNUZ = 2,
    E5M2_FNUZ = 3,
};

template <typename T> struct numeric_traits;

template <>
struct numeric_traits<fp8_t> {
#if defined(__gfx950__)
    // gfx950 (MI355) uses OCP E4M3 format (same as CK)
    static constexpr fp8_interpretation f8_interpret = fp8_interpretation::E4M3_OCP;
#else
    static constexpr fp8_interpretation f8_interpret = fp8_interpretation::E4M3_FNUZ;
#endif
};
template <>
struct numeric_traits<bf8_t> {
#if defined(__gfx950__)
    static constexpr fp8_interpretation f8_interpret = fp8_interpretation::E5M2_OCP;
#else
    static constexpr fp8_interpretation f8_interpret = fp8_interpretation::E5M2_FNUZ;
#endif
};

// ============================================================================
// 7. type_convert<To>(from) — scalar conversion via opus::cast
// ============================================================================
template <typename To, typename From>
CK_TILE_HOST_DEVICE constexpr inline To type_convert(From val) {
    if constexpr (std::is_same_v<To, From>) {
        return val;
    } else if constexpr (std::is_same_v<To, float>) {
        if constexpr (std::is_same_v<From, fp16_t> || std::is_same_v<From, bf16_t> ||
                      std::is_same_v<From, fp8_t> || std::is_same_v<From, bf8_t>) {
            return opus::cast<opus::fp32_t, From>(val);
        } else {
            return static_cast<To>(val);
        }
    } else if constexpr (std::is_same_v<From, float>) {
        if constexpr (std::is_same_v<To, fp8_t>) {
            // Saturate to fp8 range before conversion.
            // opus::cast (hardware cvt_pk_fp8) maps overflow to NaN (0x7F/0x80),
            // but CK saturates to fp8_max. Match CK behavior.
            // FNUZ E4M3: max=240, OCP E4M3: max=448
            constexpr bool is_fnuz = (numeric_traits<fp8_t>::f8_interpret == fp8_interpretation::E4M3_FNUZ);
            constexpr float fp8_max = is_fnuz ? 240.0f : 448.0f;
            float clamped = val;
            if (clamped > fp8_max) clamped = fp8_max;
            if (clamped < -fp8_max) clamped = -fp8_max;
            // NaN passthrough: NaN comparisons are false, so clamped stays NaN
            return opus::cast<To, opus::fp32_t>(clamped);
        } else if constexpr (std::is_same_v<To, fp16_t> || std::is_same_v<To, bf16_t>) {
            return opus::cast<To, opus::fp32_t>(val);
        } else {
            return static_cast<To>(val);
        }
    } else {
        return type_convert<To>(type_convert<float>(val));
    }
}

// ============================================================================
// 8. bit_cast<To>(from)
// ============================================================================
template <typename To, typename From>
CK_TILE_HOST_DEVICE inline To bit_cast(From val) {
    return __builtin_bit_cast(To, val);
}

// ============================================================================
// 9. buffer_view — pointer-based global memory load/store
//
//    Kernel API patterns:
//      auto buf = make_buffer_view<address_space_enum::global>(ptr, size);
//      buf.init_raw();
//      vec_t<T,N> v = buf.template get<vec_t<T,N>>(off, 0, true);
//      buf.template set<vec_t<T,N>>(off, 0, true, val);
//      buf.set(off, 0, true, val);                         // no .template
//
//    CK's buffer_view::set is a template<typename V> member function.
//    In dependent contexts, callers use `.template set(...)` or
//    `.template set<VecType>(...)`.  Both forms work because set is a
//    template and C++17 allows .template before a template-id or a
//    deduced call in a dependent expression.
//
//    Clang 20 (ROCm 7.x) issues -Wmissing-template-arg-list-after-template-kw
//    for `.template set(...)` without explicit args.  This is a warning, not
//    an error, unless -Werror is set.  Our set is a template, so both forms
//    are syntactically valid.
// ============================================================================
enum class address_space_enum { global, lds, sgpr, vgpr };

template <typename T>
struct buffer_view {
    T* ptr_;
    int size_bytes_;

    CK_TILE_DEVICE buffer_view(T* p, int size_elems)
        : ptr_(p), size_bytes_(size_elems * static_cast<int>(sizeof(T))) {}

    CK_TILE_DEVICE void init_raw() {}

    // get<VecType>(element_offset, soffset=0, pred=true)
    template <typename VecType>
    CK_TILE_DEVICE VecType get(index_t elem_offset, index_t /*soffset*/ = 0, bool pred = true) const {
        VecType result{};
        if (pred) {
            __builtin_memcpy(&result, ptr_ + elem_offset, sizeof(VecType));
        }
        return result;
    }

    // set<VecType=void>(element_offset, soffset, pred, value)
    template <typename VecType = void, typename V = VecType>
    CK_TILE_DEVICE void set(index_t elem_offset, index_t /*soffset*/, bool pred, const V& val) {
        if (pred) {
            __builtin_memcpy(ptr_ + elem_offset, &val, sizeof(V));
        }
    }
};

template <address_space_enum, typename T>
CK_TILE_DEVICE inline buffer_view<T> make_buffer_view(T* p, int size_elems) {
    return buffer_view<T>(p, size_elems);
}

// ============================================================================
// 10. get_warp_size
// ============================================================================
CK_TILE_HOST_DEVICE constexpr int get_warp_size() { return 64; }

// ============================================================================
// 12. constant / number / bool_constant — matches CK's integral_constant.hpp
//     Must have constexpr arithmetic operators for warp_sort.h compile-time math
// ============================================================================
template <auto v>
struct constant {
    using value_type = decltype(v);
    using type = constant;
    static constexpr value_type value = v;
    CK_TILE_HOST_DEVICE constexpr operator value_type() const noexcept { return value; }
    CK_TILE_HOST_DEVICE constexpr value_type operator()() const noexcept { return value; }
    CK_TILE_HOST_DEVICE static constexpr bool is_static() { return true; }
};

// Constexpr unary operators
template <auto x>
CK_TILE_HOST_DEVICE constexpr auto operator+(constant<x>) { return constant<(+x)>{}; }
template <auto x>
CK_TILE_HOST_DEVICE constexpr auto operator-(constant<x>) { return constant<(-x)>{}; }
template <auto x>
CK_TILE_HOST_DEVICE constexpr auto operator~(constant<x>) { return constant<(~x)>{}; }
template <auto x>
CK_TILE_HOST_DEVICE constexpr auto operator!(constant<x>) { return constant<(!x)>{}; }

// Constexpr binary operators (required by warp_sort.h)
#define CK_TILE_SHIM_BINARY_OP(OP) \
    template <auto x, auto y> \
    CK_TILE_HOST_DEVICE constexpr auto operator OP(constant<x>, constant<y>) { \
        return constant<(x OP y)>{}; \
    }
CK_TILE_SHIM_BINARY_OP(+)
CK_TILE_SHIM_BINARY_OP(-)
CK_TILE_SHIM_BINARY_OP(*)
CK_TILE_SHIM_BINARY_OP(/)
CK_TILE_SHIM_BINARY_OP(%)
CK_TILE_SHIM_BINARY_OP(&)
CK_TILE_SHIM_BINARY_OP(|)
CK_TILE_SHIM_BINARY_OP(^)
CK_TILE_SHIM_BINARY_OP(<<)
CK_TILE_SHIM_BINARY_OP(>>)
CK_TILE_SHIM_BINARY_OP(&&)
CK_TILE_SHIM_BINARY_OP(||)
CK_TILE_SHIM_BINARY_OP(==)
CK_TILE_SHIM_BINARY_OP(!=)
CK_TILE_SHIM_BINARY_OP(>)
CK_TILE_SHIM_BINARY_OP(<)
CK_TILE_SHIM_BINARY_OP(>=)
CK_TILE_SHIM_BINARY_OP(<=)
#undef CK_TILE_SHIM_BINARY_OP

template <index_t v>
using number = constant<v>;
template <bool b>
using bool_constant = constant<b>;

// ============================================================================
// 13. static_for, make_tuple, ext_vector_t
// ============================================================================
template <index_t Begin, index_t End, index_t Step = 1>
struct static_for {
    template <typename F>
    CK_TILE_HOST_DEVICE constexpr void operator()(F&& f) const {
        _impl(std::forward<F>(f), number<Begin>{});
    }
private:
    template <typename F, index_t I>
    CK_TILE_HOST_DEVICE constexpr void _impl(F&& f, number<I>) const {
        if constexpr (I < End) {
            f(number<I>{});
            _impl(std::forward<F>(f), number<I + Step>{});
        }
    }
};

using opus::make_tuple;

template <typename T, index_t N>
using ext_vector_t = T __attribute__((ext_vector_type(N)));

// ============================================================================
// 14a. Warp intrinsics
// ============================================================================
CK_TILE_DEVICE inline index_t get_lane_id() {
    return __lane_id();
}
template <bool ReturnSgpr = true>
CK_TILE_DEVICE inline index_t get_warp_id(bool_constant<ReturnSgpr> = {}) {
    const index_t warp_id = threadIdx.x / get_warp_size();
    if constexpr (ReturnSgpr) {
        return __builtin_amdgcn_readfirstlane(warp_id);
    } else {
        return warp_id;
    }
}
template <typename T>
CK_TILE_DEVICE inline T warp_shuffle_up(T val, int delta) {
    return __shfl_up(val, delta);
}
template <typename T>
CK_TILE_DEVICE inline T warp_shuffle_down(T val, int delta) {
    return __shfl_down(val, delta);
}
template <typename T>
CK_TILE_DEVICE inline T warp_shuffle(T val, int src_lane) {
    return __shfl(val, src_lane);
}

// ============================================================================
// 14b. Math utilities
// ============================================================================
template <typename T>
CK_TILE_HOST_DEVICE constexpr T min(T a) { return a; }
template <typename T, typename U>
CK_TILE_HOST_DEVICE constexpr std::common_type_t<T, U> min(T a, U b) { return a < b ? a : b; }
template <typename T>
CK_TILE_HOST_DEVICE constexpr T max(T a) { return a; }
template <typename T, typename U>
CK_TILE_HOST_DEVICE constexpr std::common_type_t<T, U> max(T a, U b) { return a > b ? a : b; }
CK_TILE_HOST_DEVICE constexpr index_t integer_least_multiple(index_t x, index_t y) {
    return ((x + y - 1) / y) * y;
}
CK_TILE_HOST_DEVICE constexpr index_t next_power_of_two(index_t x) {
    x--;
    x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16;
    return x + 1;
}
CK_TILE_HOST_DEVICE constexpr index_t integer_divide_ceil(index_t a, index_t b) {
    return (a + b - 1) / b;
}
CK_TILE_DEVICE inline float exp(float x) { return __expf(x); }

// ============================================================================
// 14b. log2e_v
// ============================================================================
template <typename T>
constexpr T log2e_v = static_cast<T>(1.4426950408889634);

// ============================================================================
// 14. stream_config / launch_kernel stubs
// ============================================================================
struct stream_config {
    hipStream_t stream_id_ = nullptr;
    bool time_kernel_ = false;
};
template <typename... C>
float launch_kernel(const stream_config& s, C&&... c) {
    (c(s), ...);
    return 0;
}

// ============================================================================
// 17. Tile Distribution Engine — CK-free replacement for reduce.cu
//     Provides: sequence, tuple (type-level), tile_distribution_encoding,
//     make_static_tile_distribution, naive_tensor_view, tile_window,
//     distributed_tile_window, distributed_tensor, load_tile, store_tile,
//     sweep_tile, set_tile, tile_elementwise_in, cast_tile, block_sync_lds
// ============================================================================

// --- sequence<Is...> (compile-time integer sequence) ---
template <index_t... Is>
struct sequence {};

// --- tuple (type-level holder for tile_distribution_encoding args) ---
// Note: this is distinct from opus::tuple (runtime). Only used as template arg.
template <typename... Ts>
struct tuple {};

// --- tile_distribution_encoding: purely type-level metadata ---
template <typename, typename, typename, typename, typename, typename>
struct tile_distribution_encoding {};

// --- distribution_tag: carries kElemsPerThread for decltype chain ---
template <int32_t N>
struct distribution_tag {
    static constexpr int32_t kElemsPerThread = N;
};

// --- distributed_tensor: register-level storage ---
template <typename T, int32_t kElemsPerThread>
struct distributed_tensor {
    T data_[kElemsPerThread];

    CK_TILE_DEVICE T& operator()(index_t i) { return data_[i]; }
    CK_TILE_DEVICE const T& operator()(index_t i) const { return data_[i]; }

    static CK_TILE_HOST_DEVICE constexpr auto get_tile_distribution() {
        return distribution_tag<kElemsPerThread>{};
    }
};

// --- make_static_distributed_tensor ---
template <typename T, int32_t N>
CK_TILE_DEVICE auto make_static_distributed_tensor(distribution_tag<N>) {
    return distributed_tensor<T, N>{};
}

// --- tile_distribution: holds compile-time distribution constants ---
template <int32_t kNumRepeat_, int32_t kNumWarpN_, int32_t kThrPerWarpN_, int32_t kVectorN_>
struct tile_distribution {
    static constexpr int32_t kNumRepeat   = kNumRepeat_;
    static constexpr int32_t kNumWarpN    = kNumWarpN_;
    static constexpr int32_t kThrPerWarpN = kThrPerWarpN_;
    static constexpr int32_t kVectorN     = kVectorN_;
    static constexpr int32_t kElemsPerThread = kNumRepeat * kVectorN;
};

// --- make_static_tile_distribution: extract constants from encoding type ---
template <typename SeqR, typename S0, index_t R, index_t WN, index_t TW, index_t VN,
          typename T2, typename T3, typename SA, typename SB>
CK_TILE_HOST_DEVICE constexpr auto make_static_tile_distribution(
    tile_distribution_encoding<SeqR, tuple<S0, sequence<R, WN, TW, VN>>, T2, T3, SA, SB>)
{
    return tile_distribution<R, WN, TW, VN>{};
}

// --- naive_tensor_view ---
template <typename T>
struct naive_tensor_view {
    T* ptr_;
};

template <address_space_enum, typename T, typename A, typename B, typename C, typename D>
CK_TILE_DEVICE auto make_naive_tensor_view(T* ptr, A, B, C, D) {
    return naive_tensor_view<T>{ptr};
}

// --- tile_window ---
template <typename T, int32_t kSizeDV>
struct tile_window {
    T* ptr_;

    CK_TILE_DEVICE void set_bottom_tensor_view_data_ptr(const std::remove_const_t<T>* p) {
        ptr_ = const_cast<T*>(p);
    }
};

// make_tile_window (3-arg): view + window_size + origin
// Origin is {0,0} brace-init — accept any type
template <typename T, typename WinA, typename WinB, typename Origin>
CK_TILE_DEVICE auto make_tile_window(
    naive_tensor_view<T> view,
    opus::tuple<WinA, WinB>,
    Origin)
{
    constexpr int32_t kDV = remove_cvref_t<WinB>::value;
    return tile_window<T, kDV>{view.ptr_};
}

// Overload for brace-init-list {0,0} which can't deduce Origin
template <typename T, typename WinA, typename WinB>
CK_TILE_DEVICE auto make_tile_window(
    naive_tensor_view<T> view,
    opus::tuple<WinA, WinB> win,
    std::initializer_list<int>)
{
    constexpr int32_t kDV = remove_cvref_t<WinB>::value;
    return tile_window<T, kDV>{view.ptr_};
}

// --- distributed_tile_window: window + distribution for load_tile/store_tile ---
template <typename T, int32_t kSizeDV, int32_t kElemsPerThread, int32_t kVectorN>
struct distributed_tile_window {
    T* ptr_;

    CK_TILE_DEVICE void set_bottom_tensor_view_data_ptr(const std::remove_const_t<T>* p) {
        ptr_ = const_cast<T*>(p);
    }
};

// make_tile_window (2-arg): tile_window + distribution
template <typename T, int32_t kSizeDV,
          int32_t kNumRepeat, int32_t kNumWarpN, int32_t kThrPerWarpN, int32_t kVectorN>
CK_TILE_DEVICE auto make_tile_window(
    tile_window<T, kSizeDV> win,
    tile_distribution<kNumRepeat, kNumWarpN, kThrPerWarpN, kVectorN>)
{
    constexpr int32_t kElems = kNumRepeat * kVectorN;
    return distributed_tile_window<T, kSizeDV, kElems, kVectorN>{win.ptr_};
}

// --- load_tile: vectorized global load respecting distribution ---
template <typename T, int32_t kSizeDV, int32_t kElemsPerThread, int32_t kVectorN>
CK_TILE_DEVICE auto load_tile(
    distributed_tile_window<T, kSizeDV, kElemsPerThread, kVectorN>& win)
{
    using scalar_t = std::remove_const_t<T>;
    distributed_tensor<float, kElemsPerThread> result;
    const int32_t tid     = threadIdx.x;
    const int32_t warp_id = tid / 64;
    const int32_t lane_id = tid % 64;
    constexpr int32_t kNumRepeat = kElemsPerThread / kVectorN;
    constexpr int32_t NumWarpN   = kSizeDV / (kNumRepeat * 64 * kVectorN);

    int32_t elem_idx = 0;
    #pragma unroll
    for (int32_t rep = 0; rep < kNumRepeat; ++rep) {
        const int32_t base = rep * (NumWarpN * 64 * kVectorN)
                           + warp_id * (64 * kVectorN)
                           + lane_id * kVectorN;
        if constexpr (kVectorN == 1) {
            result.data_[elem_idx++] = static_cast<float>(win.ptr_[base]);
        } else {
            using load_vec_t = scalar_t __attribute__((ext_vector_type(kVectorN)));
            load_vec_t tmp = *reinterpret_cast<const load_vec_t*>(win.ptr_ + base);
            #pragma unroll
            for (int32_t v = 0; v < kVectorN; ++v) {
                result.data_[elem_idx + v] = static_cast<float>(tmp[v]);
            }
            elem_idx += kVectorN;
        }
    }
    return result;
}

// --- store_tile: vectorized global store ---
template <typename T, int32_t kSizeDV, int32_t kElemsPerThread>
CK_TILE_DEVICE void store_tile(
    tile_window<T, kSizeDV>& win,
    const distributed_tensor<T, kElemsPerThread>& tensor)
{
    const int32_t tid     = threadIdx.x;
    const int32_t warp_id = tid / 64;
    const int32_t lane_id = tid % 64;
    constexpr int32_t MaxVec = 16 / sizeof(T);
    constexpr int32_t kVectorN = (kElemsPerThread <= MaxVec) ? kElemsPerThread : MaxVec;
    constexpr int32_t kNumRepeat = kElemsPerThread / kVectorN;
    constexpr int32_t NumWarpN   = kSizeDV / (kNumRepeat * 64 * kVectorN);

    int32_t elem_idx = 0;
    #pragma unroll
    for (int32_t rep = 0; rep < kNumRepeat; ++rep) {
        const int32_t base = rep * (NumWarpN * 64 * kVectorN)
                           + warp_id * (64 * kVectorN)
                           + lane_id * kVectorN;
        if constexpr (kVectorN == 1) {
            win.ptr_[base] = tensor.data_[elem_idx++];
        } else {
            using store_vec_t = T __attribute__((ext_vector_type(kVectorN)));
            store_vec_t tmp;
            #pragma unroll
            for (int32_t v = 0; v < kVectorN; ++v) {
                tmp[v] = tensor.data_[elem_idx + v];
            }
            *reinterpret_cast<store_vec_t*>(win.ptr_ + base) = tmp;
            elem_idx += kVectorN;
        }
    }
}

// --- sweep_tile: iterate over thread-local elements ---
template <typename T, int32_t N, typename F>
CK_TILE_DEVICE void sweep_tile(distributed_tensor<T, N>& tensor, F&& f) {
    #pragma unroll
    for (int32_t i = 0; i < N; ++i) { f(i); }
}
template <typename T, int32_t N, typename F>
CK_TILE_DEVICE void sweep_tile(const distributed_tensor<T, N>& tensor, F&& f) {
    #pragma unroll
    for (int32_t i = 0; i < N; ++i) { f(i); }
}

// --- set_tile ---
template <typename T, int32_t N>
CK_TILE_DEVICE void set_tile(distributed_tensor<T, N>& tensor, T value) {
    #pragma unroll
    for (int32_t i = 0; i < N; ++i) { tensor.data_[i] = value; }
}

// --- tile_elementwise_in ---
template <typename F, typename T, int32_t N>
CK_TILE_DEVICE auto tile_elementwise_in(F&& f, const distributed_tensor<T, N>& tensor) {
    distributed_tensor<T, N> result;
    #pragma unroll
    for (int32_t i = 0; i < N; ++i) { result.data_[i] = f(tensor.data_[i]); }
    return result;
}

// --- cast_tile ---
template <typename OutT, typename T, int32_t N>
CK_TILE_DEVICE auto cast_tile(const distributed_tensor<T, N>& tensor) {
    distributed_tensor<OutT, N> result;
    #pragma unroll
    for (int32_t i = 0; i < N; ++i) { result.data_[i] = type_convert<OutT>(tensor.data_[i]); }
    return result;
}

// --- block_sync_lds ---
CK_TILE_DEVICE void block_sync_lds() {
    __syncthreads();
}

} // namespace ck_tile
