// SPDX-License-Identifier: MIT
// gfx950-private deterministic grid-stride replay for the strict packed
// hybrid G64 persistent experiment.
//
// This is a new, deliberately narrow symbol.  It copies the established
// packed Replay/NW4/CACHED/U-forward/V-forward/P0 arithmetic while replacing
// only its conservative host-upper block mapping.  The legacy recurrence
// template remains untouched and is the fallback for every other feature
// combination.
#pragma once

#include "k2_kda_context_parallel_kernel.hpp"

namespace flashkda_hip::gfx950 {

// Logical task = (exact filtered context, head, V64 group).  Four waves own
// four V16 slices, so two logical V groups cover D=128.  The host caps the
// physical 1-D grid (normally to one device wave); each CTA deterministically
// walks blockIdx.x, blockIdx.x+gridDim.x, ... without an atomic counter.
template <bool HO = false, bool SFP32 = false>
__global__ void __launch_bounds__(4 * 64)
k2_kda_context_replay_hybrid_g64_grid_stride_nw4_kernel(
        const __bf16* __restrict__ v_g,       // packed [T_total,H,D]
        const float* __restrict__ beta_g,     // activated [n_ht,C]
        __bf16* __restrict__ out_g,
        const __bf16* __restrict__ ws_kd,
        const __bf16* __restrict__ ws_qd,
        const __bf16* __restrict__ ws_kr,
        const float* __restrict__ ws_gt,      // activated decay [n_ht,D]
        const __bf16* __restrict__ ws_inv,
        const __bf16* __restrict__ ws_mqk,
        const float* __restrict__ affine_b,   // scanned h_in [G,H,K,V]
        void* __restrict__ final_state,       // [N,H,V,K], HO only
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ tile_prefix,
        const int* __restrict__ context_prefix,
        int N,
        int total_tiles,
        int H) {
    constexpr int GROUP_CHUNKS = 64;
    constexpr int NW = 4;
    constexpr int C = 16;
    constexpr int D = 128;
    constexpr int BV = 16;
    constexpr int SD = D + 4;
    constexpr int NKB = D / C;
    constexpr int NTHREADS = NW * 64;
    constexpr int ROW_VECS = (C * D) / 8;
    constexpr int RW = ROW_VECS / NTHREADS;
    constexpr int VR = (C * BV) / 64;
    constexpr int V_GROUPS = D / (NW * BV);
    static_assert(RW == 1 && VR == 4 && V_GROUPS == 2,
                  "persistent replay mapping changed");

    const int tid = int(threadIdx.x);
    const int wave = tid >> 6;
    const int lane = tid & 63;

    // Exact context count is published by the same-stream compact prefix
    // builder.  It is allowed to be zero for an all-short hybrid batch.
    const int exact_contexts = N > 0 ? context_prefix[N] : 0;
    const int64_t task_count =
        int64_t(exact_contexts) * H * V_GROUPS;
    if (N <= 0 || H <= 0 || task_count <= 0)
        return;

    // Exact P0 cached/U/V-forward LDS footprint.  No vmat/umat or second
    // arena is declared in this specialization.
    __shared__ __bf16 kd[C * SD];
    __shared__ __bf16 qd[C * SD];
    __shared__ __bf16 kr[C * D];
    __shared__ __bf16 inv[C * C];
    __shared__ __bf16 mqk[C * C];
    __shared__ float decay[D];
    __shared__ float beta[C];

    for (int64_t task64 = int64_t(blockIdx.x);
         task64 < task_count;
         task64 += int64_t(gridDim.x)) {
        const int64_t context_head_v = int64_t(H) * V_GROUPS;
        const int global_context = int(task64 / context_head_v);
        const int task_rem = int(task64 -
                                 int64_t(global_context) * context_head_v);
        const int h = task_rem / V_GROUPS;
        const int v_group = task_rem - h * V_GROUPS;
        const int v0 = (v_group * NW + wave) * BV;

        int lo = 0;
        int hi = N;
        while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (context_prefix[mid] <= global_context)
                lo = mid;
            else
                hi = mid;
        }
        const int seq = lo;
        const int local_group = global_context - context_prefix[seq];
        const int64_t bos = cu_seqlens[seq];
        const int seq_len = int(cu_seqlens[seq + 1] - bos);
        const int seq_chunks = (seq_len + C - 1) / C;
        const int first_chunk = local_group * GROUP_CHUNKS;
        const int group_chunks =
            min(GROUP_CHUNKS, seq_chunks - first_chunk);
        if (group_chunks <= 0)
            continue;

        const bool is_last_group =
            first_chunk + group_chunks == seq_chunks;
        const int ht_base =
            h * total_tiles + tile_prefix[seq] + first_chunk;
        const int t0_base = int(bos) + first_chunk * C;
        const int64_t context_slab =
            (int64_t(global_context) * H + h) * D * D;

        // Scanned affine_b holds the true incoming state in the recurrence's
        // established register-B lane layout.
        float sreg[NKB][4];
        #pragma unroll
        for (int ktile = 0; ktile < NKB; ++ktile) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int vv = v0 + (lane & 15);
                const int kk = ktile * C + (lane >> 4) * 4 + i;
                sreg[ktile][i] = affine_b[
                    context_slab + int64_t(kk) * D + vv];
            }
        }

        bf16x8 kd_r[RW];
        bf16x8 qd_r[RW];
        bf16x8 kr_r[RW];
        bf16x8 inv_r;
        bf16x8 mqk_r;
        f32x4 gt_r;
        __bf16 v_r[VR];
        bf16x4 v_fragment;
        float beta_r;

        auto stage = [&](int ht, int t0, int alen) {
            #pragma unroll
            for (int j = 0; j < RW; ++j) {
                const int vi = tid + j * NTHREADS;
                kd_r[j] = reinterpret_cast<const bf16x8*>(
                    ws_kd + int64_t(ht) * C * D)[vi];
                qd_r[j] = reinterpret_cast<const bf16x8*>(
                    ws_qd + int64_t(ht) * C * D)[vi];
                kr_r[j] = reinterpret_cast<const bf16x8*>(
                    ws_kr + int64_t(ht) * C * D)[vi];
            }
            if (tid < (C * C) / 8) {
                inv_r = reinterpret_cast<const bf16x8*>(
                    ws_inv + int64_t(ht) * C * C)[tid];
                mqk_r = reinterpret_cast<const bf16x8*>(
                    ws_mqk + int64_t(ht) * C * C)[tid];
            }
            if (tid < D / 4) {
                gt_r = reinterpret_cast<const f32x4*>(
                    ws_gt + int64_t(ht) * D)[tid];
            }
            #pragma unroll
            for (int j = 0; j < VR; ++j) {
                const int m = (lane >> 4) * 4 + j;
                const int vv = lane & 15;
                v_r[j] = m < alen
                    ? v_g[(int64_t(t0 + m) * H + h) * D + v0 + vv]
                    : (__bf16)0.0f;
            }
            if (tid < C)
                beta_r = beta_g[int64_t(ht) * C + tid];
        };

        auto commit = [&]() {
            #pragma unroll
            for (int j = 0; j < RW; ++j) {
                const int vi = tid + j * NTHREADS;
                const int row = vi >> 4;
                const int col8 = vi & 15;
                reinterpret_cast<bf16x8*>(
                    kd + row * SD)[col8] = kd_r[j];
                reinterpret_cast<bf16x8*>(
                    qd + row * SD)[col8] = qd_r[j];

                const int source_element = vi * 8;
                const int c = source_element / D;
                const int k = source_element - c * D;
                const int ktile = k / C;
                const int ki = k - ktile * C;
                __bf16* const kr_dst =
                    kr + ktile * C * C + c * C + ki;
                *reinterpret_cast<bf16x8*>(kr_dst) = kr_r[j];
            }
            if (tid < (C * C) / 8) {
                reinterpret_cast<bf16x8*>(inv)[tid] = inv_r;
                reinterpret_cast<bf16x8*>(mqk)[tid] = mqk_r;
            }
            if (tid < D / 4)
                reinterpret_cast<f32x4*>(decay)[tid] = gt_r;
            #pragma unroll
            for (int j = 0; j < VR; ++j)
                v_fragment[j] = v_r[j];
            if (tid < C)
                beta[tid] = beta_r;
        };

        int t0_cur = t0_base;
        int alen_cur = min(C, seq_len - first_chunk * C);
        stage(ht_base, t0_cur, alen_cur);
        commit();
        __syncthreads();

        for (int chunk = 0; chunk < group_chunks; ++chunk) {
            const int t0 = t0_cur;
            const int alen = alen_cur;
            const bool has_next = chunk + 1 < group_chunks;
            if (has_next) {
                const int next_global_chunk = first_chunk + chunk + 1;
                const int ht_next = ht_base + chunk + 1;
                const int t0_next = int(bos) + next_global_chunk * C;
                const int alen_next =
                    min(C, seq_len - next_global_chunk * C);
                stage(ht_next, t0_next, alen_next);
                t0_cur = t0_next;
                alen_cur = alen_next;
            }

            const f32x4 residual = RegBX32::template run<SD, NKB>(
                kd, sreg, lane);
            bf16x4 vnew_bf;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int m = (lane >> 4) * 4 + i;
                const float source = bf16_to_f32(v_fragment[i]);
                const float value =
                    (source - residual[i]) * beta[m];
                vnew_bf[i] = f32_to_bf16(value);
            }

            const f32x4 u = context_mfma_row_major_a_reg_b(
                inv, vnew_bf, lane);
            bf16x4 u_bf;
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                u_bf[i] = f32_to_bf16(u[i]);

            const f32x4 from_state = RegBX32::template run<SD, NKB>(
                qd, sreg, lane);
            const f32x4 from_local = context_mfma_row_major_a_reg_b(
                mqk, u_bf, lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int m = (lane >> 4) * 4 + i;
                const int vv = lane & 15;
                if (m < alen) {
                    const __bf16 a = f32_to_bf16(from_state[i]);
                    const __bf16 b = f32_to_bf16(from_local[i]);
                    out_g[(int64_t(t0 + m) * H + h) * D + v0 + vv] =
                        f32_to_bf16(bf16_to_f32(a) + bf16_to_f32(b));
                }
            }

            #pragma unroll
            for (int ktile = 0; ktile < NKB; ++ktile) {
                const f32x4 carry = context_mfma_tiled_kr_reg_b(
                    kr, u_bf, ktile, lane);
                const int kbase = ktile * C + (lane >> 4) * 4;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sreg[ktile][i] =
                        sreg[ktile][i] * decay[kbase + i] + carry[i];
                }
            }
            __syncthreads();
            if (has_next) {
                commit();
                __syncthreads();
            }
        }

        if constexpr (HO) {
            if (is_last_group) {
                const int64_t state_slab =
                    (int64_t(seq) * H + h) * D * D;
                #pragma unroll
                for (int ktile = 0; ktile < NKB; ++ktile) {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int vv = v0 + (lane & 15);
                        const int kk =
                            ktile * C + (lane >> 4) * 4 + i;
                        const int64_t idx =
                            state_slab + int64_t(vv) * D + kk;
                        if constexpr (SFP32) {
                            reinterpret_cast<float*>(final_state)[idx] =
                                sreg[ktile][i];
                        } else {
                            reinterpret_cast<__bf16*>(final_state)[idx] =
                                f32_to_bf16(sreg[ktile][i]);
                        }
                    }
                }
            }
        }
    }
}

}  // namespace flashkda_hip::gfx950
