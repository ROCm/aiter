#pragma once

#include <opus/opus.hpp>
#include "defs.h"

template<class D_ACC>
__device__ inline D_ACC mla_rescale(D_ACC lse, D_ACC ref) {
    return (lse > opus::numeric_limits<D_ACC>::lowest())
             ? __builtin_exp2f((lse - ref) * D_ACC(MLA_DECODE_LOG2_E))
             : D_ACC(0.0f);
}

template<class T>
__device__ inline void mla_write_lse(const opus_mla_decode_reduce_kargs& kargs, int row, int head,
                                     int lane, typename T::D_ACC m, typename T::D_ACC denom) {
    using D_ACC = typename T::D_ACC;
    if (lane != 0 || kargs.lse_ptr == nullptr) return;
    D_ACC* lse = reinterpret_cast<D_ACC*>(kargs.lse_ptr);
    lse[(size_t)row * kargs.H + head] = (denom > D_ACC(0.0f))
        ? (m + __builtin_log2f(denom) * MLA_DECODE_LN_2)
        : opus::numeric_limits<D_ACC>::infinity();
}

template<class T, class Row>
__device__ void mla_reduce_online(const opus_mla_decode_reduce_kargs& kargs, int out_row, int head,
                                  int lane, int ns, Row partial_row) {
    using D_OUT = typename T::D_OUT;
    using D_ACC = typename T::D_ACC;
    using D_ACCx4 = opus::vector_t<D_ACC, 4>;
    constexpr int WARP = T::WARP_SIZE;
    constexpr int D    = T::D_VO_SIZE;
    constexpr int VEC  = 4;
    constexpr int NVEC = D / (WARP * VEC);

    const int H = kargs.H;
    const D_ACC* lse_accum = reinterpret_cast<const D_ACC*>(kargs.lse_accum);
    const D_ACC* o_accum   = reinterpret_cast<const D_ACC*>(kargs.o_accum);

    auto load_split = [&](int i, D_ACCx4* dst) {
        const D_ACC* p = o_accum + ((size_t)partial_row(i) * H + head) * D + lane * VEC;
        #pragma unroll
        for (int v = 0; v < NVEC; ++v)
            dst[v] = *reinterpret_cast<const D_ACCx4*>(p + v * WARP * VEC);
    };
    auto load_lse = [&](int i) { return lse_accum[(size_t)partial_row(i) * H + head]; };

    D_ACCx4 acc[NVEC];
    #pragma unroll
    for (int v = 0; v < NVEC; ++v) acc[v] = D_ACCx4{0.0f, 0.0f, 0.0f, 0.0f};
    D_ACC m = opus::numeric_limits<D_ACC>::lowest();
    D_ACC denom = 0.0f;

    D_ACCx4 cur[NVEC];
    load_split(0, cur);
    D_ACC lse_cur = load_lse(0);
    for (int i = 0; i < ns; ++i) {
        D_ACCx4 nxt[NVEC];
        D_ACC lse_nxt = 0.0f;
        if (i + 1 < ns) {
            load_split(i + 1, nxt);
            lse_nxt = load_lse(i + 1);
        }
        const D_ACC new_m   = opus::max(m, lse_cur);
        const D_ACC old_scl = __builtin_exp2f((m - new_m) * D_ACC(MLA_DECODE_LOG2_E));
        const D_ACC cur_scl = mla_rescale(lse_cur, new_m);
        #pragma unroll
        for (int v = 0; v < NVEC; ++v) acc[v] = old_scl * acc[v] + cur_scl * cur[v];
        denom = denom * old_scl + cur_scl;
        m = new_m;
        #pragma unroll
        for (int v = 0; v < NVEC; ++v) cur[v] = nxt[v];
        lse_cur = lse_nxt;
    }

    mla_write_lse<T>(kargs, out_row, head, lane, m, denom);

    const D_ACC inv_denom = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
    #pragma unroll
    for (int v = 0; v < NVEC; ++v) acc[v] *= inv_denom;

    D_OUT* out = reinterpret_cast<D_OUT*>(kargs.out_ptr)
               + (size_t)out_row * kargs.stride_o_b + head * kargs.stride_o_h;
    #pragma unroll
    for (int v = 0; v < NVEC; ++v) {
        opus::vector_t<D_OUT, 4> ov;
        #pragma unroll
        for (int e = 0; e < VEC; ++e) ov[e] = static_cast<D_OUT>(acc[v][e]);
        *reinterpret_cast<opus::vector_t<D_OUT, 4>*>(out + lane * VEC + v * WARP * VEC) = ov;
    }
}

template<class T, class Row>
__device__ void mla_reduce_two_pass(const opus_mla_decode_reduce_kargs& kargs, int out_row, int head,
                                    int lane, int ns, Row partial_row) {
    using D_OUT = typename T::D_OUT;
    using D_ACC = typename T::D_ACC;
    using D_ACCx4 = opus::vector_t<D_ACC, 4>;
    constexpr int WARP = T::WARP_SIZE;
    constexpr int D    = T::D_VO_SIZE;
    constexpr int VEC  = 4;
    constexpr int NVEC = D / (WARP * VEC);

    const int H = kargs.H;
    const D_ACC* lse_accum = reinterpret_cast<const D_ACC*>(kargs.lse_accum);
    const D_ACC* o_accum   = reinterpret_cast<const D_ACC*>(kargs.o_accum);

    auto load_lse = [&](int i) { return lse_accum[(size_t)partial_row(i) * H + head]; };

    D_ACC local_max = opus::numeric_limits<D_ACC>::lowest();
    for (int s = lane; s < ns; s += WARP)
        local_max = opus::max(local_max, load_lse(s));
    #pragma unroll
    for (int off = WARP / 2; off >= 1; off >>= 1)
        local_max = opus::max(local_max, opus::shfl(local_max, lane ^ off));
    const D_ACC m = local_max;

    D_ACC local_sum = 0.0f;
    for (int s = lane; s < ns; s += WARP)
        local_sum += mla_rescale(load_lse(s), m);
    #pragma unroll
    for (int off = WARP / 2; off >= 1; off >>= 1)
        local_sum += opus::shfl(local_sum, lane ^ off);

    mla_write_lse<T>(kargs, out_row, head, lane, m, local_sum);

    const D_ACC inv_denom = (local_sum > 0.0f) ? (1.0f / local_sum) : 0.0f;

    D_ACCx4 acc[NVEC];
    #pragma unroll
    for (int v = 0; v < NVEC; ++v) acc[v] = D_ACCx4{0.0f, 0.0f, 0.0f, 0.0f};

    auto load_split = [&](int i, D_ACCx4* dst) {
        const D_ACC* p = o_accum + ((size_t)partial_row(i) * H + head) * D + lane * VEC;
        #pragma unroll
        for (int v = 0; v < NVEC; ++v)
            dst[v] = *reinterpret_cast<const D_ACCx4*>(p + v * WARP * VEC);
    };

    for (int i = 0; i < ns; ++i) {
        D_ACCx4 cur[NVEC];
        load_split(i, cur);
        const D_ACC w = mla_rescale(load_lse(i), m);
        #pragma unroll
        for (int v = 0; v < NVEC; ++v) acc[v] += w * cur[v];
    }

    D_OUT* out = reinterpret_cast<D_OUT*>(kargs.out_ptr)
               + (size_t)out_row * kargs.stride_o_b + head * kargs.stride_o_h;
    #pragma unroll
    for (int v = 0; v < NVEC; ++v) {
        opus::vector_t<D_OUT, 4> ov;
        #pragma unroll
        for (int e = 0; e < VEC; ++e) ov[e] = static_cast<D_OUT>(acc[v][e] * inv_denom);
        *reinterpret_cast<opus::vector_t<D_OUT, 4>*>(out + lane * VEC + v * WARP * VEC) = ov;
    }
}

template<class Traits, int HEADS_PER_BLOCK = 8>
__global__ void mla_reduce_kernel(opus_mla_decode_reduce_kargs kargs) {
    using T = opus::remove_cvref_t<Traits>;
    constexpr int WARP = T::WARP_SIZE;
    constexpr int ONLINE_MAX_NS = 4;

    const int warp = opus::thread_id_x() / WARP;
    const int lane = opus::thread_id_x() % WARP;
    const int tile = opus::block_id_x();
    const int head = opus::block_id_y() * HEADS_PER_BLOCK + warp;
    if (head >= kargs.H) return;

    const int start = kargs.reduce_indptr[tile];
    const int ns    = kargs.reduce_indptr[tile + 1] - start;
    if (ns <= 0 || kargs.reduce_partial_map[start] < 0) return;

    const int q_start = kargs.reduce_final_map[tile * 2];
    const int q_end   = kargs.reduce_final_map[tile * 2 + 1];

    for (int s = 0; s < q_end - q_start; ++s) {
        auto partial_row = [&](int i) { return kargs.reduce_partial_map[start + i] + s; };
        if (ns <= ONLINE_MAX_NS)
            mla_reduce_online<T>(kargs, q_start + s, head, lane, ns, partial_row);
        else
            mla_reduce_two_pass<T>(kargs, q_start + s, head, lane, ns, partial_row);
    }
}
