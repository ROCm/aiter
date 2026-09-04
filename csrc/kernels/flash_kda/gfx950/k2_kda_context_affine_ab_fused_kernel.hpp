// gfx950-private strict-opt-in fused affine-map producer.
//
// This header provides one narrowly-scoped, strict-opt-in candidate for packed
// context groups: NW4, activated/cached beta+decay operands,
// register-forwarded U/V, and the P0 single-LDS arena.  Every other dispatch
// retains the standalone B/A producers.  One CTA advances two independent
// recurrences,
//
//   b: h=0, real v   -> FP32 affine_b[K,V]
//   A: h=I, zero v   -> BF16 affine_a[K,K],
//
// while publishing the common kd/kr/inv/decay/beta operands only once.  The
// established standalone B/A kernels remain untouched and are the fallback
// until a caller explicitly dispatches this candidate. G8/G16 are compiled
// only to support host-guarded dense-N1 diagnostics; the launcher remains
// responsible for excluding packed and scratch-unsafe G8/G16 shapes.
#pragma once

#include "k2_kda_context_parallel_kernel.hpp"

namespace flashkda_hip::gfx950 {

template <int GROUP_CHUNKS>
__global__ void __launch_bounds__(4 * 64)
k2_kda_context_affine_ab_fused_nw4_kernel(
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
    static_assert(
        GROUP_CHUNKS == 8 || GROUP_CHUNKS == 16 || GROUP_CHUNKS == 32 ||
            GROUP_CHUNKS == 64 || GROUP_CHUNKS == 128,
        "fused affine AB supports only G8/G16/G32/G64/G128");
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
    static_assert(RW == 1 && VR == 4,
                  "fused affine AB is specialized for NW4/V16");

    const int tid = int(threadIdx.x);
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int global_context = int(blockIdx.x) / H;
    const int h = int(blockIdx.x) - global_context * H;
    const int v0 = (int(blockIdx.y) * NW + wave) * BV;

    if (N <= 0 || global_context >= context_prefix[N])
        return;
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
        return;

    const int ht_base = h * total_tiles + tile_prefix[seq] + first_chunk;
    const int t0_base = int(bos) + first_chunk * C;
    const int64_t context_slab =
        (int64_t(global_context) * H + h) * D * D;

    // Exactly the U/V-forward P0 arena: 4,224 + 4,096 + 512 + 512 + 64
    // bytes = 9,408 bytes.  No qd/mqk/vmat/umat storage is needed.
    __shared__ __bf16 kd[C * SD];
    __shared__ __bf16 kr[C * D];
    __shared__ __bf16 inv[C * C];
    __shared__ float decay[D];
    __shared__ float beta[C];

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

    // The next chunk resides in registers while A consumes the current LDS
    // publication.  Placing stage between B and A leaves 13 independent A
    // MFMAs available to cover VMEM latency without keeping the prefetched
    // vectors live across both recurrences.
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

        // Preserve the standalone AffineB instruction/rounding order.
        advance.template operator()<false>(breg);

        // Tie the compiler boundary to every updated B-state fragment.  This
        // emits no ISA instruction but prevents the following next-chunk VMEM
        // from being hoisted above an unfinished B MFMA.
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
            const int alen_next = min(C, seq_len - next_global_chunk * C);
            stage(ht_next, t0_next, alen_next);
            t0_cur = t0_next;
            alen_cur = alen_next;
        }

        // AffineA uses the same current LDS operands and the exact zero-source
        // arithmetic of its standalone recurrence.  Its state and rounding
        // sequence are independent of B.
        advance.template operator()<true>(areg);

        // Retain the established P0 barrier protocol.  This intentionally
        // does not fold in the separate final-barrier-removal experiment.
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
            const int64_t idx = context_slab + int64_t(kk) * D + vv;
            affine_b[idx] = breg[ktile][i];
            affine_a[idx] = f32_to_bf16(areg[ktile][i]);
        }
    }
}

// Dense B=1 counterpart to the packed producer above.  Keep this as a
// separate kernel instead of sharing a helper with the frozen packed body:
// public/raw single-sequence packed calls are normalized to the byte-identical
// dense layout by the adapter, while preserving the packed G32/G64/G128
// machine code remains a required rollback invariant.  Dense N>1 never
// dispatches this specialization.
template <int GROUP_CHUNKS>
__global__ void __launch_bounds__(4 * 64)
k2_kda_context_affine_ab_fused_dense_nw4_kernel(
        const __bf16* __restrict__ v_g,          // dense B=1 [T,H,D]
        const float* __restrict__ beta_g,        // activated [H*NT,C]
        const __bf16* __restrict__ ws_kd,        // [H*NT,C,D]
        const __bf16* __restrict__ ws_kr,        // [H*NT,C,D]
        const float* __restrict__ ws_gt,         // decay [H*NT,D]
        const __bf16* __restrict__ ws_inv,       // [H*NT,C,C]
        float* __restrict__ affine_b,             // [G,H,K,V]
        __bf16* __restrict__ affine_a,            // [G,H,K,K]
        int T_seq,
        int H,
        int NT) {
    static_assert(
        GROUP_CHUNKS == 8 || GROUP_CHUNKS == 16 || GROUP_CHUNKS == 32 ||
            GROUP_CHUNKS == 64 || GROUP_CHUNKS == 128,
        "dense fused affine AB supports only G8/G16/G32/G64/G128");
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
    static_assert(RW == 1 && VR == 4,
                  "dense fused affine AB is specialized for NW4/V16");

    const int tid = int(threadIdx.x);
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int global_context = int(blockIdx.x) / H;
    const int h = int(blockIdx.x) - global_context * H;
    const int v0 = (int(blockIdx.y) * NW + wave) * BV;

    // Mirror the established recurrence's !VL mapping exactly.  With N=1 the
    // global context is the local group, the token base is zero, and the K1
    // workspace for head h starts at h*NT.
    if (NT <= 0)
        return;
    const int groups_per_sequence =
        (NT + GROUP_CHUNKS - 1) / GROUP_CHUNKS;
    if (global_context >= groups_per_sequence)
        return;
    const int first_chunk = global_context * GROUP_CHUNKS;
    const int group_chunks = min(GROUP_CHUNKS, NT - first_chunk);
    if (group_chunks <= 0)
        return;

    const int ht_base = h * NT + first_chunk;
    const int t0_base = first_chunk * C;
    const int64_t context_slab =
        (int64_t(global_context) * H + h) * D * D;

    // Exactly the U/V-forward P0 arena used by the packed specialization.
    __shared__ __bf16 kd[C * SD];
    __shared__ __bf16 kr[C * D];
    __shared__ __bf16 inv[C * C];
    __shared__ float decay[D];
    __shared__ float beta[C];

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
    int alen_cur = min(C, T_seq - first_chunk * C);
    stage(ht_base, t0_cur, alen_cur);
    commit();
    __syncthreads();

    for (int chunk = 0; chunk < group_chunks; ++chunk) {
        const bool has_next = chunk + 1 < group_chunks;

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
            const int t0_next = next_global_chunk * C;
            const int alen_next = min(C, T_seq - next_global_chunk * C);
            stage(ht_next, t0_next, alen_next);
            t0_cur = t0_next;
            alen_cur = alen_next;
        }

        advance.template operator()<true>(areg);
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
            const int64_t idx = context_slab + int64_t(kk) * D + vv;
            affine_b[idx] = breg[ktile][i];
            affine_a[idx] = f32_to_bf16(areg[ktile][i]);
        }
    }
}

// Strict-opt-in next-stage-early experiment.  Keep both established kernels
// above textually untouched: this independent body extends the lifetime of the
// staged kd/kr/inv/decay/beta/V registers across the complete AffineB
// recurrence, which may expose more VMEM latency but can also increase VGPR
// pressure or spill.  The host policy therefore admits only measured G16/G64
// fused layouts behind an exact-"1" switch, and compiled metadata must be
// audited before this candidate can graduate.
//
// The current chunk was already committed to LDS and v_fragment before the
// loop.  Loading the next chunk into the disjoint *_r prefetch registers is
// therefore independent of both current recurrences.  Only that load moves:
// AffineB still precedes AffineA, and the established barrier, commit, second
// barrier, and final writeback order remain unchanged.
template <int GROUP_CHUNKS, bool DENSE, bool EQUAL_DENSE_N4 = false,
          bool STAGE_EARLY = true>
__device__ __forceinline__ void
k2_kda_context_affine_ab_fused_stage_early_body(
        const __bf16* __restrict__ v_g,
        const float* __restrict__ beta_g,
        const __bf16* __restrict__ ws_kd,
        const __bf16* __restrict__ ws_kr,
        const float* __restrict__ ws_gt,
        const __bf16* __restrict__ ws_inv,
        float* __restrict__ affine_b,
        __bf16* __restrict__ affine_a,
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ tile_prefix,
        const int* __restrict__ context_prefix,
        int N,
        int total_tiles,
        int T_seq,
        int H,
        int NT,
        __bf16* __restrict__ kd,
        __bf16* __restrict__ kr,
        __bf16* __restrict__ inv,
        float* __restrict__ decay,
        float* __restrict__ beta) {
    static_assert(
        (!DENSE && GROUP_CHUNKS == 64) ||
            (DENSE && (GROUP_CHUNKS == 16 || GROUP_CHUNKS == 64)),
        "stage-early affine AB supports packed G64 or dense G16/G64");
    static_assert(!EQUAL_DENSE_N4 || (DENSE && GROUP_CHUNKS == 64),
                  "equal dense specialization is N4/G64-only");
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
    static_assert(RW == 1 && VR == 4,
                  "stage-early affine AB is specialized for NW4/V16");

    const int tid = int(threadIdx.x);
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int global_context = int(blockIdx.x) / H;
    const int h = int(blockIdx.x) - global_context * H;
    const int v0 = (int(blockIdx.y) * NW + wave) * BV;

    int first_chunk;
    int group_chunks;
    int ht_base;
    int token_base;
    int seq_len;
    if constexpr (DENSE) {
        (void)cu_seqlens;
        (void)tile_prefix;
        (void)context_prefix;
        (void)N;
        (void)total_tiles;
        if (NT <= 0)
            return;
        const int groups_per_sequence =
            (NT + GROUP_CHUNKS - 1) / GROUP_CHUNKS;
        if constexpr (EQUAL_DENSE_N4) {
            constexpr int kSequences = 4;
            constexpr int kGroupsPerSequence = 4;
            static_assert(GROUP_CHUNKS == 64,
                          "equal dense mapping requires G64");
            if (N != kSequences || T_seq != 4096 || NT != 256 ||
                groups_per_sequence != kGroupsPerSequence ||
                global_context >= kSequences * kGroupsPerSequence)
                return;
            const int seq = global_context / kGroupsPerSequence;
            const int local_group =
                global_context - seq * kGroupsPerSequence;
            first_chunk = local_group * GROUP_CHUNKS;
            group_chunks = GROUP_CHUNKS;
            ht_base = (seq * H + h) * NT + first_chunk;
            token_base = seq * T_seq;
        } else {
            if (global_context >= groups_per_sequence)
                return;
            first_chunk = global_context * GROUP_CHUNKS;
            group_chunks = min(GROUP_CHUNKS, NT - first_chunk);
            if (group_chunks <= 0)
                return;
            ht_base = h * NT + first_chunk;
            token_base = 0;
        }
        seq_len = T_seq;
    } else {
        (void)T_seq;
        (void)NT;
        if (N <= 0 || global_context >= context_prefix[N])
            return;
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
        token_base = int(cu_seqlens[seq]);
        seq_len = int(cu_seqlens[seq + 1] - cu_seqlens[seq]);
        const int seq_chunks = (seq_len + C - 1) / C;
        first_chunk = local_group * GROUP_CHUNKS;
        group_chunks = min(GROUP_CHUNKS, seq_chunks - first_chunk);
        if (group_chunks <= 0)
            return;
        ht_base = h * total_tiles + tile_prefix[seq] + first_chunk;
    }

    const int t0_base = token_base + first_chunk * C;
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
    int alen_cur;
    if constexpr (EQUAL_DENSE_N4)
        alen_cur = C;
    else
        alen_cur = min(C, seq_len - first_chunk * C);
    stage(ht_base, t0_cur, alen_cur);
    commit();
    __syncthreads();

    for (int chunk = 0; chunk < group_chunks; ++chunk) {
        const bool has_next = chunk + 1 < group_chunks;

        // The existing stage-early symbols issue all independent next-chunk
        // VMEM before beginning AffineB.  The exact-N4 established sibling
        // instantiates the same body with STAGE_EARLY=false and retains the
        // frozen B -> stage(next) -> A schedule.
        if constexpr (STAGE_EARLY) {
            if (has_next) {
                const int next_global_chunk = first_chunk + chunk + 1;
                const int ht_next = ht_base + chunk + 1;
                const int t0_next = token_base + next_global_chunk * C;
                const int alen_next = EQUAL_DENSE_N4
                    ? C : min(C, seq_len - next_global_chunk * C);
                stage(ht_next, t0_next, alen_next);
                t0_cur = t0_next;
                alen_cur = alen_next;
            }
        }

        advance.template operator()<false>(breg);

        // Retain the established dependency fence between the B and A
        // recurrences.  It emits no ISA but prevents their arithmetic chains
        // from being reordered while the earlier VMEM remains outstanding.
        #pragma unroll
        for (int ktile = 0; ktile < NKB; ++ktile) {
            asm volatile(
                ""
                : "+v"(breg[ktile][0]), "+v"(breg[ktile][1]),
                  "+v"(breg[ktile][2]), "+v"(breg[ktile][3])
                :
                : "memory");
        }

        if constexpr (!STAGE_EARLY) {
            if (has_next) {
                const int next_global_chunk = first_chunk + chunk + 1;
                const int ht_next = ht_base + chunk + 1;
                const int t0_next = token_base + next_global_chunk * C;
                const int alen_next = EQUAL_DENSE_N4
                    ? C : min(C, seq_len - next_global_chunk * C);
                stage(ht_next, t0_next, alen_next);
                t0_cur = t0_next;
                alen_cur = alen_next;
            }
        }

        advance.template operator()<true>(areg);

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
            const int64_t idx = context_slab + int64_t(kk) * D + vv;
            affine_b[idx] = breg[ktile][i];
            affine_a[idx] = f32_to_bf16(areg[ktile][i]);
        }
    }
}

__global__ void __launch_bounds__(4 * 64)
k2_kda_context_affine_ab_fused_stage_early_g64_nw4_kernel(
        const __bf16* __restrict__ v_g,
        const float* __restrict__ beta_g,
        const __bf16* __restrict__ ws_kd,
        const __bf16* __restrict__ ws_kr,
        const float* __restrict__ ws_gt,
        const __bf16* __restrict__ ws_inv,
        float* __restrict__ affine_b,
        __bf16* __restrict__ affine_a,
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ tile_prefix,
        const int* __restrict__ context_prefix,
        int N,
        int total_tiles,
        int H) {
    constexpr int C = 16;
    constexpr int D = 128;
    constexpr int SD = D + 4;
    __shared__ __bf16 kd[C * SD];
    __shared__ __bf16 kr[C * D];
    __shared__ __bf16 inv[C * C];
    __shared__ float decay[D];
    __shared__ float beta[C];
    k2_kda_context_affine_ab_fused_stage_early_body<64, false>(
        v_g, beta_g, ws_kd, ws_kr, ws_gt, ws_inv, affine_b, affine_a,
        cu_seqlens, tile_prefix, context_prefix, N, total_tiles, 0, H, 0,
        kd, kr, inv, decay, beta);
}

template <int GROUP_CHUNKS>
__global__ void __launch_bounds__(4 * 64)
k2_kda_context_affine_ab_fused_dense_stage_early_nw4_kernel(
        const __bf16* __restrict__ v_g,
        const float* __restrict__ beta_g,
        const __bf16* __restrict__ ws_kd,
        const __bf16* __restrict__ ws_kr,
        const float* __restrict__ ws_gt,
        const __bf16* __restrict__ ws_inv,
        float* __restrict__ affine_b,
        __bf16* __restrict__ affine_a,
        int T_seq,
        int H,
        int NT) {
    static_assert(GROUP_CHUNKS == 16 || GROUP_CHUNKS == 64,
                  "dense stage-early affine AB supports G16/G64");
    constexpr int C = 16;
    constexpr int D = 128;
    constexpr int SD = D + 4;
    __shared__ __bf16 kd[C * SD];
    __shared__ __bf16 kr[C * D];
    __shared__ __bf16 inv[C * C];
    __shared__ float decay[D];
    __shared__ float beta[C];
    k2_kda_context_affine_ab_fused_stage_early_body<GROUP_CHUNKS, true>(
        v_g, beta_g, ws_kd, ws_kr, ws_gt, ws_inv, affine_b, affine_a,
        nullptr, nullptr, nullptr, 1, NT, T_seq, H, NT,
        kd, kr, inv, decay, beta);
}

// Exact-equal packed N=4, 4K-per-sequence whole-graph siblings.  Common
// dispatch has already converted K1 workspace, scan and replay to dense
// indexing before either symbol can launch.  Both symbols use the independent
// global-context order [sequence][four G64 groups], matching the existing
// dense scan/replay ABI without consulting packed prefix metadata.
__global__ void __launch_bounds__(4 * 64)
k2_kda_context_affine_ab_fused_equal_n4_g64_nw4_kernel(
        const __bf16* __restrict__ v_g,
        const float* __restrict__ beta_g,
        const __bf16* __restrict__ ws_kd,
        const __bf16* __restrict__ ws_kr,
        const float* __restrict__ ws_gt,
        const __bf16* __restrict__ ws_inv,
        float* __restrict__ affine_b,
        __bf16* __restrict__ affine_a,
        int H) {
    constexpr int C = 16;
    constexpr int D = 128;
    constexpr int SD = D + 4;
    __shared__ __bf16 kd[C * SD];
    __shared__ __bf16 kr[C * D];
    __shared__ __bf16 inv[C * C];
    __shared__ float decay[D];
    __shared__ float beta[C];
    k2_kda_context_affine_ab_fused_stage_early_body<
        64, true, true, false>(
            v_g, beta_g, ws_kd, ws_kr, ws_gt, ws_inv,
            affine_b, affine_a, nullptr, nullptr, nullptr,
            4, 1024, 4096, H, 256, kd, kr, inv, decay, beta);
}

__global__ void __launch_bounds__(4 * 64)
k2_kda_context_affine_ab_fused_equal_n4_g64_stage_early_nw4_kernel(
        const __bf16* __restrict__ v_g,
        const float* __restrict__ beta_g,
        const __bf16* __restrict__ ws_kd,
        const __bf16* __restrict__ ws_kr,
        const float* __restrict__ ws_gt,
        const __bf16* __restrict__ ws_inv,
        float* __restrict__ affine_b,
        __bf16* __restrict__ affine_a,
        int H) {
    constexpr int C = 16;
    constexpr int D = 128;
    constexpr int SD = D + 4;
    __shared__ __bf16 kd[C * SD];
    __shared__ __bf16 kr[C * D];
    __shared__ __bf16 inv[C * C];
    __shared__ float decay[D];
    __shared__ float beta[C];
    k2_kda_context_affine_ab_fused_stage_early_body<
        64, true, true, true>(
            v_g, beta_g, ws_kd, ws_kr, ws_gt, ws_inv,
            affine_b, affine_a, nullptr, nullptr, nullptr,
            4, 1024, 4096, H, 256, kd, kr, inv, decay, beta);
}

}  // namespace flashkda_hip::gfx950
