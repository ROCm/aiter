// gfx950-private deterministic grid-stride fused affine-map producer.
//
// This is a packed-hybrid G64-only counterpart to the established strict-opt-in
// fused affine B/A kernel.  It preserves that kernel's P0/NW4 activated-beta,
// cached-decay, U/V-forward arithmetic while replacing its conservative 2-D
// host upper-bound grid with a capped 1-D physical grid.  Logical work is read
// from the exact device prefix after the same-stream prefix builder completes:
//
//   logical task = (global context group, head, V32)
//   physical CTA = blockIdx.x, blockIdx.x + gridDim.x, ...
//
// No atomic counter or reset is required.  All-short batches publish
// context_prefix[N] == 0, so every capped CTA returns immediately; mixed and
// long batches retain every independent affine group.  The host chooses a
// nonzero physical grid cap (one or two CTAs per CU are the initial probes).
// The established fused symbol remains untouched as the rollback control.
#pragma once

#include "k2_kda_context_affine_ab_fused_kernel.hpp"

namespace flashkda_hip::gfx950 {

__global__ void __launch_bounds__(4 * 64)
k2_kda_context_affine_ab_fused_persistent_g64_nw4_kernel(
        const __bf16* __restrict__ v_g,          // packed [T_total,H,D]
        const float* __restrict__ beta_g,        // activated [n_ht,C]
        const __bf16* __restrict__ ws_kd,        // [n_ht,C,D]
        const __bf16* __restrict__ ws_kr,        // [n_ht,C,D]
        const float* __restrict__ ws_gt,         // decay [n_ht,D]
        const __bf16* __restrict__ ws_inv,       // [n_ht,C,C]
        float* __restrict__ affine_b,             // [G,H,K,V]
        __bf16* __restrict__ affine_a,            // [G,H,K,K]
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
    constexpr int V_GROUPS = D / (NW * BV);
    constexpr int SD = D + 4;
    constexpr int NKB = D / C;
    constexpr int NTHREADS = NW * 64;
    constexpr int ROW_VECS = (C * D) / 8;
    constexpr int RW = ROW_VECS / NTHREADS;
    constexpr int VR = (C * BV) / 64;
    static_assert(V_GROUPS == 2 && RW == 1 && VR == 4,
                  "persistent fused affine AB requires G64/NW4/V32 tasks");

    const int tid = int(threadIdx.x);
    const int wave = tid >> 6;
    const int lane = tid & 63;

    // Exactly the established U/V-forward P0 arena: 4,224 + 4,096 + 512 +
    // 512 + 64 bytes = 9,408 bytes.  It is reused only after the unconditional
    // final-chunk CTA barrier in each logical task.
    __shared__ __bf16 kd[C * SD];
    __shared__ __bf16 kr[C * D];
    __shared__ __bf16 inv[C * C];
    __shared__ float decay[D];
    __shared__ float beta[C];

    if (N <= 0 || H <= 0)
        return;
    const int context_count = context_prefix[N];
    const int64_t logical_count =
        int64_t(context_count) * H * V_GROUPS;
    if (context_count <= 0 || logical_count <= 0)
        return;

    for (int64_t task64 = int64_t(blockIdx.x);
         task64 < logical_count;
         task64 += int64_t(gridDim.x)) {
        const int64_t context_head64 = task64 / V_GROUPS;
        const int v_group = int(task64 - context_head64 * V_GROUPS);
        const int global_context = int(context_head64 / H);
        const int h = int(context_head64 - int64_t(global_context) * H);
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
        // A coherent prefix makes this positive.  Keep the guard CTA-uniform
        // so malformed metadata cannot strand any barrier below.
        if (group_chunks <= 0)
            continue;

        const int ht_base =
            h * total_tiles + tile_prefix[seq] + first_chunk;
        const int t0_base = int(bos) + first_chunk * C;
        const int64_t context_slab =
            (int64_t(global_context) * H + h) * D * D;

        float breg[NKB][4];
        float areg[NKB][4];
        #pragma unroll
        for (int ktile = 0; ktile < NKB; ++ktile) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int vv = v0 + (lane & 15);
                const int kk = ktile * C + (lane >> 4) * 4 + i;
                breg[ktile][i] = 0.0f;
                areg[ktile][i] = kk == vv ? 1.0f : 0.0f;
            }
        }

        // Keep the established next-chunk register stage and the exact ordering
        // between the B and A recurrences.
        bf16x8 kd_r[RW];
        bf16x8 kr_r[RW];
        bf16x8 inv_r;
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
                kr_r[j] = reinterpret_cast<const bf16x8*>(
                    ws_kr + int64_t(ht) * C * D)[vi];
            }
            if (tid < (C * C) / 8) {
                inv_r = reinterpret_cast<const bf16x8*>(
                    ws_inv + int64_t(ht) * C * C)[tid];
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
                reinterpret_cast<bf16x8*>(kd + row * SD)[col8] = kd_r[j];

                const int source_element = vi * 8;
                const int c = source_element / D;
                const int k = source_element - c * D;
                const int ktile = k / C;
                const int ki = k - ktile * C;
                __bf16* const kr_dst =
                    kr + ktile * C * C + c * C + ki;
                *reinterpret_cast<bf16x8*>(kr_dst) = kr_r[j];
            }
            if (tid < (C * C) / 8)
                reinterpret_cast<bf16x8*>(inv)[tid] = inv_r;
            if (tid < D / 4)
                reinterpret_cast<f32x4*>(decay)[tid] = gt_r;
            #pragma unroll
            for (int i = 0; i < VR; ++i)
                v_fragment[i] = v_r[i];
            if (tid < C)
                beta[tid] = beta_r;
        };

        auto advance = [&]<bool ZERO_SOURCE>(float (&state)[NKB][4]) {
            const f32x4 residual =
                RegBX32::template run<SD, NKB>(kd, state, lane);
            bf16x4 vnew_bf;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int m = (lane >> 4) * 4 + i;
                const float source = ZERO_SOURCE
                    ? bf16_to_f32((__bf16)0.0f)
                    : bf16_to_f32(v_fragment[i]);
                const float value = (source - residual[i]) * beta[m];
                vnew_bf[i] = f32_to_bf16(value);
            }

            const f32x4 u =
                context_mfma_row_major_a_reg_b(inv, vnew_bf, lane);
            bf16x4 u_bf;
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                u_bf[i] = f32_to_bf16(u[i]);

            #pragma unroll
            for (int ktile = 0; ktile < NKB; ++ktile) {
                const f32x4 carry = context_mfma_tiled_kr_reg_b(
                    kr, u_bf, ktile, lane);
                const int kbase = ktile * C + (lane >> 4) * 4;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    state[ktile][i] =
                        state[ktile][i] * decay[kbase + i] + carry[i];
                }
            }
        };

        int t0_cur = t0_base;
        int alen_cur = min(C, seq_len - first_chunk * C);
        stage(ht_base, t0_cur, alen_cur);
        commit();
        __syncthreads();

        for (int chunk = 0; chunk < group_chunks; ++chunk) {
            const bool has_next = chunk + 1 < group_chunks;

            // Preserve the standalone AffineB instruction and rounding order.
            advance.template operator()<false>(breg);
            #pragma unroll
            for (int ktile = 0; ktile < NKB; ++ktile) {
                asm volatile(
                    ""
                    : "+v"(breg[ktile][0]), "+v"(breg[ktile][1]),
                      "+v"(breg[ktile][2]), "+v"(breg[ktile][3])
                    :
                    : "memory");
            }

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

            // AffineA consumes the same current LDS publication and retains the
            // exact zero-source recurrence of the established fused kernel.
            advance.template operator()<true>(areg);

            // This unconditional final-chunk barrier also makes reuse of the P0
            // arena by the next grid-stride task safe.
            __syncthreads();
            if (has_next) {
                commit();
                __syncthreads();
            }
        }

        #pragma unroll
        for (int ktile = 0; ktile < NKB; ++ktile) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int vv = v0 + (lane & 15);
                const int kk = ktile * C + (lane >> 4) * 4 + i;
                const int64_t idx =
                    context_slab + int64_t(kk) * D + vv;
                affine_b[idx] = breg[ktile][i];
                affine_a[idx] = f32_to_bf16(areg[ktile][i]);
            }
        }
    }
}

}  // namespace flashkda_hip::gfx950
