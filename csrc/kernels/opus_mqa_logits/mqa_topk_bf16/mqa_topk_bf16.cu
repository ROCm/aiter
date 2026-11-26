// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// BF16 MQA logits + top-k (OPUS): the host side of mqa_topk_bf16.h -- the arch-neutral stages (tau,
// select), the registry query and the launches. The scoring kernels live in their arch's TU
// (mqa_topk_bf16_<arch>.cu) and are reached only through their registry entries.
#include "mqa_topk_bf16.h"

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "mqa_topk_bf16_select.cuh"
#include <string>
#include <vector>

namespace opus_mqa_topk {

// -- tau: a wave a row ----------------------------------------------------------------------------
// Also resets the row's pool counter and (workgroup 0) the fallback list.
// A row's samples are staged into LDS, at most TAU_MAX_SAMPLES order keys a wave: a row with more
// takes every stride-th, still spread over its window.
constexpr int TAU_MAX_SAMPLES = 2304;

__global__ __launch_bounds__(256) void tau_kernel(mqa_topk_bf16_kargs k) {
    __shared__ unsigned hist[4][256];
    __shared__ unsigned skeys[4][TAU_MAX_SAMPLES];
    const int tid = threadIdx.x, lane = tid % 64, wave = tid / 64;
    const mqa_topk_ws ws = ws_layout(k.num_rows, k.num_keys, k.cap_row, k.fb_slots);
    char* base = static_cast<char*>(k.ptr_workspace);
    if (blockIdx.x == 0 && tid < 4) reinterpret_cast<int*>(base + ws.fb_ctrl)[tid] = 0;
    const int row = blockIdx.x * 4 + wave;
    if (row >= k.num_rows) return;
    const window wd = clamp_window(k.ptr_row_starts[row], k.ptr_row_ends[row], k.num_keys);
    const int n     = wd.hi - wd.lo;
    float tau;
    if (n == 0) {
        tau = __builtin_inff();
    } else if (n <= k.cap_row) {
        tau = -__builtin_inff();
    } else {
        const int jl     = sample_at_or_after(wd.lo, k.num_keys);
        const int ns     = sample_at_or_after(wd.hi, k.num_keys) - jl;
        const int stride = (ns + TAU_MAX_SAMPLES - 1) / TAU_MAX_SAMPLES;
        const int nk     = (ns + stride - 1) / stride;
        // e of the samples are expected in the top W; their rank e + 5 sqrt(e) + 8 leaves the pool
        // about n / nk (e + 5 sqrt(e) + 8) keys, five deviations clear of W on either side.
        const float e = (float)k.width * (float)nk / (float)n;
        const int m   = (int)__builtin_ceilf(e + 5.0f * __builtin_sqrtf(e) + 8.0f);
        if (m > nk) {
            tau = -__builtin_inff();
        } else {
            const float* s = reinterpret_cast<const float*>(base + ws.samples) +
                             (size_t)row * num_samples(k.num_keys) + jl;
            constexpr int BATCH = 12;   // loads in flight a lane
            for (int j0 = lane; j0 < nk; j0 += BATCH * 64) {
                float v[BATCH];
#pragma unroll
                for (int x = 0; x < BATCH; ++x)
                    v[x] = j0 + x * 64 < nk ? s[(size_t)(j0 + x * 64) * stride] : 0.f;
#pragma unroll
                for (int x = 0; x < BATCH; ++x)
                    if (j0 + x * 64 < nk) skeys[wave][j0 + x * 64] = order_key(v[x]);
            }
            __builtin_amdgcn_wave_barrier();
            tau = order_value(wave_kth_largest(skeys[wave], nk, m, hist[wave], lane));
        }
    }
    if (lane == 0) {
        reinterpret_cast<float*>(base + ws.tau)[row] = tau;
        reinterpret_cast<int*>(base + ws.counts)[row] = 0;
    }
}

// -- select: a workgroup a row --------------------------------------------------------------------
// The pool in registers, CPT candidates a thread (all loads in flight at once: the pool is cold);
// rows it cannot finish go to the fallback list.
template<int CPT>
__global__ __launch_bounds__(SEL_BLOCK) void select_kernel(mqa_topk_bf16_kargs k) {
    __shared__ select_smem sm;
    const int row = blockIdx.x, tid = threadIdx.x;
    const mqa_topk_ws ws = ws_layout(k.num_rows, k.num_keys, k.cap_row, k.fb_slots);
    char* base      = static_cast<char*>(k.ptr_workspace);
    const window wd = clamp_window(k.ptr_row_starts[row], k.ptr_row_ends[row], k.num_keys);
    const int n = wd.hi - wd.lo, W = k.width;
    const int C = reinterpret_cast<const int*>(base + ws.counts)[row];
    int* out_i   = k.ptr_indices + (size_t)row * W;
    float* out_v = (k.flags & MQA_TOPK_VALUES) ? k.ptr_values + (size_t)row * W : nullptr;
    if (n == 0) {
        for (int p = tid; p < W; p += SEL_BLOCK) {
            out_i[p] = -1;
            if (out_v) out_v[p] = -__builtin_inff();
        }
        return;
    }
    if (C > k.cap_row || C < (n < W ? n : W)) {
        if (tid == 0) {
            int* ctrl = reinterpret_cast<int*>(base + ws.fb_ctrl);
            reinterpret_cast<int*>(base + ws.fb_list)[atomicAdd(ctrl, 1)] = row;
        }
        return;
    }
    const int2* pool = reinterpret_cast<const int2*>(base + ws.cands) + (size_t)row * k.cap_row;
    reg_source<CPT> src;
    int2 c[CPT];
#pragma unroll
    for (int x = 0; x < CPT; ++x) {
        const int i = tid + x * SEL_BLOCK;
        c[x]        = i < C ? pool[i] : make_int2(-1, 0);
    }
#pragma unroll
    for (int x = 0; x < CPT; ++x) {
        src.key[x] = order_key(__builtin_bit_cast(float, c[x].y));
        src.idx[x] = c[x].x;
    }
    select_row(src, C, W, W, wd, out_i, out_v, sm, tid);
}

}  // namespace opus_mqa_topk

// -- the registry query and the launches ----------------------------------------------------------

namespace {

namespace om = opus_mqa_topk;

using registry_fn = const om::mqa_topk_instance* (*)(int*);
constexpr registry_fn REGISTRIES[] = {om::mqa_topk_gfx950_instances};

const om::mqa_topk_instance& instance_at(int id) {
    AITER_CHECK(id >= 0, "no mqa_topk_bf16 instance ", id);
    int base = 0;
    for (registry_fn fn : REGISTRIES) {
        int n = 0;
        const om::mqa_topk_instance* table = fn(&n);
        if (id < base + n) return table[id - base];
        base += n;
    }
    AITER_CHECK(false, "no mqa_topk_bf16 instance ", id, "; there are ", base);
    return *REGISTRIES[0](&base);   // unreachable
}

std::string device_arch(int device_id) {
    hipDeviceProp_t prop;
    HIP_CALL(hipGetDeviceProperties(&prop, device_id));
    std::string arch = prop.gcnArchName;
    return arch.substr(0, arch.find(':'));
}

int ceil_div(int64_t a, int64_t b) { return static_cast<int>((a + b - 1) / b); }

void check_bounds(int rows, int keys, int chunk_keys) {
    AITER_CHECK(rows >= 0 && keys >= 1 && keys < om::MAX_KEYS, "need rows >= 0 and 1 <= keys < ",
                om::MAX_KEYS, "; got ", rows, ", ", keys);
    AITER_CHECK(chunk_keys >= 16 && chunk_keys % 16 == 0,
                "chunk_keys must be a positive multiple of 16, got ", chunk_keys);
}

mqa_topk_bf16_kargs make_kargs(int64_t q, int64_t k, int64_t weights, int64_t row_starts,
                               int64_t row_ends, int rows, int keys, int chunk_keys, int relu) {
    mqa_topk_bf16_kargs a{};
    a.ptr_q          = reinterpret_cast<const void*>(q);
    a.ptr_k          = reinterpret_cast<const void*>(k);
    a.ptr_weights    = reinterpret_cast<const float*>(weights);
    a.ptr_row_starts = reinterpret_cast<const int*>(row_starts);
    a.ptr_row_ends   = reinterpret_cast<const int*>(row_ends);
    a.num_rows       = rows;
    a.num_keys       = keys;
    a.chunk_keys     = chunk_keys;
    a.sample_chunk   = om::SAMPLE_CHUNK;
    a.flags          = relu ? om::MQA_TOPK_RELU : 0;
    return a;
}

}  // namespace

#define MQA_TOPK_INSTANCE_FIELDS                                                                \
    X(id) X(rows_per_wave) X(waves) X(sample_waves) X(tile_keys) X(resident_logits)             \
    X(resident_sample) X(resident_emit) X(resident_fallback)

int mqa_topk_bf16_instances(int device_id, aiter_tensor_t& out) {
    aiter_detail::AiterThrowGuard throw_guard;
    AITER_CHECK(out.is_cpu() && out.dtype() == AITER_DTYPE_i32 && out.is_contiguous(),
                "out must be a contiguous host int32 tensor");
    HipDeviceGuard guard(device_id);
    const std::string arch = device_arch(device_id);
    int cus = 0;
    HIP_CALL(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, device_id));
#define X(f) +1
    constexpr int fields = 0 MQA_TOPK_INSTANCE_FIELDS;
#undef X
    std::vector<int> recs{fields};
    int id = 0;
    for (registry_fn fn : REGISTRIES) {
        int n = 0;
        const om::mqa_topk_instance* table = fn(&n);
        for (int i = 0; i < n; ++i, ++id) {
            const om::mqa_topk_instance& in = table[i];
            if (arch != in.arch) continue;
            int resident[om::SCAN_STAGES] = {};
            for (int s = 0; s < om::SCAN_STAGES; ++s) {
                int per_cu = 0;
                HIP_CALL(in.occupancy(s, &per_cu));
                resident[s] = per_cu * cus;
            }
            const int row[] = {id,          in.rows_per_wave, in.waves,    in.sample_waves,
                               in.tile_keys, resident[0],     resident[1], resident[2],
                               resident[3]};
            static_assert(sizeof(row) / sizeof(row[0]) == fields);
            recs.insert(recs.end(), row, row + fields);
        }
    }
    const int n = static_cast<int>(recs.size());
    if (static_cast<int64_t>(out.numel()) >= n)
        std::copy(recs.begin(), recs.end(), reinterpret_cast<int*>(out.data_ptr()));
    return n;
}

int64_t mqa_topk_bf16_workspace_bytes(int rows, int keys, int cap_row, int fb_slots) {
    aiter_detail::AiterThrowGuard throw_guard;
    AITER_CHECK(rows >= 0 && keys >= 1 && cap_row >= 1 && fb_slots >= 1,
                "bad workspace bounds");
    return om::ws_layout(rows, keys, cap_row, fb_slots).total;
}

void mqa_topk_bf16_launch(int64_t q,
                          int64_t k,
                          int64_t weights,
                          int64_t row_starts,
                          int64_t row_ends,
                          int64_t indices,
                          int64_t values,
                          int64_t workspace,
                          int64_t workspace_bytes,
                          int rows,
                          int keys,
                          int topk,
                          int cap_row,
                          int chunk_keys,
                          int fb_slots,
                          int relu,
                          int instance,
                          int device,
                          int64_t stream) {
    aiter_detail::AiterThrowGuard throw_guard;
    const om::mqa_topk_instance& in = instance_at(instance);
    check_bounds(rows, keys, chunk_keys);
    const int width = topk < keys ? topk : keys;
    AITER_CHECK(topk >= 1 && cap_row >= width && cap_row <= 48 * om::SEL_BLOCK && fb_slots >= 1,
                "need topk >= 1, min(topk, keys) <= cap_row <= ", 48 * om::SEL_BLOCK,
                ", fb_slots >= 1");
    const int64_t need = om::ws_layout(rows, keys, cap_row, fb_slots).total;
    AITER_CHECK(workspace_bytes >= need, "the workspace holds ", workspace_bytes,
                " bytes; this call needs ", need);
    if (rows == 0) return;

    mqa_topk_bf16_kargs a = make_kargs(q, k, weights, row_starts, row_ends, rows, keys, chunk_keys,
                                       relu);
    a.ptr_indices   = reinterpret_cast<int*>(indices);
    a.ptr_values    = reinterpret_cast<float*>(values);
    a.ptr_workspace = reinterpret_cast<void*>(workspace);
    a.topk          = topk;
    a.width         = width;
    a.cap_row       = cap_row;
    a.fb_slots      = fb_slots;
    a.flags |= values ? om::MQA_TOPK_VALUES : 0;

    HipDeviceGuard guard(device);
    const hipStream_t s = reinterpret_cast<hipStream_t>(stream);
    HIP_CALL_LAUNCH(in.launch(om::STAGE_SAMPLE, a,
                              ceil_div(rows, in.rows_per_wave * in.sample_waves) *
                                  ceil_div(om::num_samples(keys), om::SAMPLE_CHUNK),
                              s));
    om::tau_kernel<<<ceil_div(rows, 4), 256, 0, s>>>(a);
    HIP_CALL_LAUNCH(hipGetLastError());
    HIP_CALL_LAUNCH(in.launch(om::STAGE_EMIT, a,
                              ceil_div(rows, in.rows_per_wave * in.waves) *
                                  ceil_div(keys, chunk_keys),
                              s));
    if (cap_row <= 24 * om::SEL_BLOCK)
        om::select_kernel<24><<<rows, om::SEL_BLOCK, 0, s>>>(a);
    else
        om::select_kernel<48><<<rows, om::SEL_BLOCK, 0, s>>>(a);
    HIP_CALL_LAUNCH(hipGetLastError());
    HIP_CALL_LAUNCH(in.launch(om::STAGE_FALLBACK, a, fb_slots, s));
}

void mqa_logits_bf16_launch(int64_t q,
                            int64_t k,
                            int64_t weights,
                            int64_t row_starts,
                            int64_t row_ends,
                            int64_t out,
                            int rows,
                            int keys,
                            int chunk_keys,
                            int relu,
                            int instance,
                            int device,
                            int64_t stream) {
    aiter_detail::AiterThrowGuard throw_guard;
    const om::mqa_topk_instance& in = instance_at(instance);
    check_bounds(rows, keys, chunk_keys);
    if (rows == 0) return;
    mqa_topk_bf16_kargs a = make_kargs(q, k, weights, row_starts, row_ends, rows, keys, chunk_keys,
                                       relu);
    a.ptr_values = reinterpret_cast<float*>(out);
    HipDeviceGuard guard(device);
    const int row_blocks = ceil_div(rows, in.rows_per_wave * in.waves);
    HIP_CALL_LAUNCH(in.launch(om::STAGE_LOGITS, a, row_blocks * ceil_div(keys, chunk_keys),
                              reinterpret_cast<hipStream_t>(stream)));
}
