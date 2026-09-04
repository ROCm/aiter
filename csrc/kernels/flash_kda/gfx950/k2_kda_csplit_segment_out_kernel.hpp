// gfx950-private plain BT64 segment output replay operators.
#pragma once

#include "mfma_gfx950.hpp"

namespace flashkda_hip::gfx950 {

namespace segment_replay_detail {

constexpr int C = 16;
constexpr int D = 128;
constexpr int SD = D + 4;
constexpr int NKB = D / C;

template <bool StageSin>
struct ReplaySinStorage {
    __bf16 unused;
};

template <>
struct alignas(16) ReplaySinStorage<true> {
    __bf16 values[D * SD];
};

}  // namespace segment_replay_detail

// Stable single-arena A/B operator.  One 8-wave CTA owns all V=128 rows of a
// (sequence, head, segment), and each wave keeps V=16 state rows in FP32 VGPRs.
template <bool VL = false>
__global__ void __launch_bounds__(512)
k2_kda_csplit_segment_out_x32_kernel(
        const __bf16* __restrict__ cs_u,
        const __bf16* __restrict__ cs_sin,
        const __bf16* __restrict__ ws_qd,
        const __bf16* __restrict__ ws_kr,
        const float* __restrict__ ws_gt,
        const __bf16* __restrict__ ws_mqk,
        __bf16* __restrict__ out_g,
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ tile_prefix,
        const int* __restrict__ segment_prefix,
        int N, int total_tiles, int total_segments,
        int T_seq, int H, int NT) {
    using namespace segment_replay_detail;
    const int tid = threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int v0 = wave * C;

    int h, seq_idx = 0, ht_base, ss, t0_base, chunks;
    if constexpr (VL) {
        const int gsi = blockIdx.x;
        h = blockIdx.y;
        int lo = 0, hi = N;
        while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (segment_prefix[mid] <= gsi) lo = mid; else hi = mid;
        }
        seq_idx = lo;
        const int local_seg = gsi - segment_prefix[lo];
        const int64_t bos = cu_seqlens[lo];
        const int seq_len = int(cu_seqlens[lo + 1] - bos);
        const int nseg = (seq_len + 63) / 64;
        if (local_seg >= nseg) return;
        const int local_chunk = local_seg * 4;
        ht_base = h * total_tiles + tile_prefix[lo] + local_chunk;
        ss = h * total_segments + gsi;
        t0_base = int(bos) + local_chunk * C;
        chunks = min(4, (seq_len + C - 1) / C - local_chunk);
    } else {
        const int seg = blockIdx.x;
        const int bh = blockIdx.y;
        const int b = bh / H;
        h = bh % H;
        ht_base = bh * NT + seg * 4;
        ss = bh * ((NT + 3) / 4) + seg;
        t0_base = b * T_seq + seg * 64;
        chunks = min(4, NT - seg * 4);
    }

    __shared__ __bf16 qd[C * SD];
    __shared__ __bf16 kr[C * D];
    __shared__ __bf16 umat[C * D];
    __shared__ __bf16 mqk[C * C];
    __shared__ float gtot[D];

    float sreg[NKB][4];
    #pragma unroll
    for (int kt = 0; kt < NKB; ++kt)
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int vv = v0 + (lane & 15);
            const int kk = kt * C + (lane >> 4) * 4 + i;
            sreg[kt][i] = bf16_to_f32(
                cs_sin[(int64_t(ss) * D + vv) * D + kk]);
        }

    for (int j = 0; j < chunks; ++j) {
        const int ht = ht_base + j;
        for (int idx = tid; idx < C * D; idx += 512) {
            const int m = idx / D, d = idx % D;
            qd[m * SD + d] = ws_qd[int64_t(ht) * C * D + idx];
            kr[idx] = ws_kr[int64_t(ht) * C * D + idx];
            umat[idx] = cs_u[int64_t(ht) * C * D + idx];
        }
        for (int idx = tid; idx < C * C; idx += 512)
            mqk[idx] = ws_mqk[int64_t(ht) * C * C + idx];
        for (int idx = tid; idx < D; idx += 512)
            gtot[idx] = ws_gt[int64_t(ht) * D + idx];
        __syncthreads();

        f32x4 o1 = gemm_regb_even_x32<SD, NKB>(qd, sreg, lane);
        f32x4 o2 = mm_std_tile_bf16(mqk, umat, v0, D, lane);
        const int seq_remaining = VL
            ? int(cu_seqlens[seq_idx + 1] - (t0_base + j * C))
            : T_seq - (blockIdx.x * 64 + j * C);
        const int alen = min(C, seq_remaining);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int m = (lane >> 4) * 4 + i;
            const int vv = v0 + (lane & 15);
            if (m < alen) {
                const __bf16 a = f32_to_bf16(o1[i]);
                const __bf16 b = f32_to_bf16(o2[i]);
                out_g[(int64_t(t0_base + j * C + m) * H + h) * D + vv] =
                    f32_to_bf16(bf16_to_f32(a) + bf16_to_f32(b));
            }
        }

        #pragma unroll
        for (int kt = 0; kt < NKB; ++kt) {
            f32x4 c = mm_contract_first_bf16(
                kr, umat, kt * C, v0, D, D, lane);
            const int kbase = kt * C + (lane >> 4) * 4;
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                sreg[kt][i] = sreg[kt][i] * ex2(gtot[kbase + i]) + c[i];
        }
        __syncthreads();
    }
}

// CDNA4-native multi-arena operator.  global_load_lds fills future arenas
// while the current arena feeds MFMA, avoiding a global->VGPR->LDS staging
// round trip.  The production three-arena schedule removes one more full-CTA
// rendezvous than the exact two-arena rollback specialization.
template <bool VL = false, bool StageSin = true, int Arenas = 2,
          bool CACHE_DECAY_LDS = false, bool SIN_FRAGMENT = false>
__global__ void __launch_bounds__(512)
k2_kda_csplit_segment_out_gll_kernel(
        const __bf16* __restrict__ cs_u,
        const __bf16* __restrict__ cs_sin,
        const __bf16* __restrict__ ws_qd,
        const __bf16* __restrict__ ws_kr,
        const float* __restrict__ ws_gt,
        const __bf16* __restrict__ ws_mqk,
        __bf16* __restrict__ out_g,
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ tile_prefix,
        const int* __restrict__ segment_prefix,
        int N, int total_tiles, int total_segments,
        int T_seq, int H, int NT) {
#if !defined(__gfx950__)
    // A dual-architecture extension also visits this gfx950 translation unit
    // during the gfx942 device pass.  Dispatch can never select this operator
    // there, and the production three-arena specialization intentionally uses
    // more than gfx942's 64-KiB per-workgroup LDS limit.  Emit an empty image
    // for that pass instead of coupling the gfx950 schedule to gfx942 limits.
    return;
#else
    using namespace segment_replay_detail;
    static_assert(Arenas == 2 || Arenas == 3,
                  "gfx950 GLL replay supports two or three LDS arenas");
    static_assert(!SIN_FRAGMENT || !StageSin,
                  "fragment-major sin must bypass the row-major LDS stage");
    const int tid = threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int v0 = wave * C;

    int h, seq_idx = 0, ht_base, ss, t0_base, chunks;
    if constexpr (VL) {
        const int gsi = blockIdx.x;
        h = blockIdx.y;
        int lo = 0, hi = N;
        while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (segment_prefix[mid] <= gsi) lo = mid; else hi = mid;
        }
        seq_idx = lo;
        const int local_seg = gsi - segment_prefix[lo];
        const int64_t bos = cu_seqlens[lo];
        const int seq_len = int(cu_seqlens[lo + 1] - bos);
        const int nseg = (seq_len + 63) / 64;
        if (local_seg >= nseg) return;
        const int local_chunk = local_seg * 4;
        ht_base = h * total_tiles + tile_prefix[lo] + local_chunk;
        ss = h * total_segments + gsi;
        t0_base = int(bos) + local_chunk * C;
        chunks = min(4, (seq_len + C - 1) / C - local_chunk);
    } else {
        const int seg = blockIdx.x;
        const int bh = blockIdx.y;
        const int b = bh / H;
        h = bh % H;
        ht_base = bh * NT + seg * 4;
        ss = bh * ((NT + 3) / 4) + seg;
        t0_base = b * T_seq + seg * 64;
        chunks = min(4, NT - seg * 4);
    }

    __shared__ __bf16 qd[Arenas][C * SD];
    __shared__ __bf16 kr[Arenas][C * D];
    __shared__ __bf16 umat[Arenas][C * D];
    __shared__ __bf16 mqk[Arenas][C * C];
    __shared__ float gtot[Arenas][D];
    // The row-major specialization stages the incoming 128x128 state with a
    // padded pitch.  The matched fragment-major specialization instantiates
    // the empty storage form and loads its already tiled fragments directly.
    __shared__ ReplaySinStorage<StageSin> sin_storage;

    float sreg[NKB][4];
    if constexpr (SIN_FRAGMENT) {
        const auto* sin_fragments = reinterpret_cast<const bf16x4*>(
            cs_sin + int64_t(ss) * D * D);
        #pragma unroll
        for (int kt = 0; kt < NKB; ++kt) {
            // Producer vblock == replay wave.  A wave-wide load of these 64
            // adjacent bf16x4 values reconstructs one complete MFMA fragment
            // tile directly in registers, with no StageSin LDS transpose.
            const bf16x4 fragment =
                sin_fragments[(wave * NKB + kt) * 64 + lane];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                sreg[kt][i] = bf16_to_f32(fragment[i]);
        }
    } else if constexpr (!StageSin) {
        #pragma unroll
        for (int kt = 0; kt < NKB; ++kt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int vv = v0 + (lane & 15);
                const int kk = kt * C + (lane >> 4) * 4 + i;
                sreg[kt][i] = bf16_to_f32(
                    cs_sin[(int64_t(ss) * D + vv) * D + kk]);
            }
    }

    auto dma = [&](int arena, int ht) {
        const __bf16* qd_src = ws_qd + int64_t(ht) * C * D;
        #pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
            const int row = wave * 2 + rr;
            global_to_lds_async<4>(
                qd[arena] + row * SD,
                qd_src + row * D + lane * 2);
        }

        if (wave < 4) {
            const int offset = wave * 512 + lane * 8;
            global_to_lds_async<16>(
                kr[arena] + wave * 512,
                ws_kr + int64_t(ht) * C * D + offset);
        } else {
            const int offset = (wave - 4) * 512 + lane * 8;
            global_to_lds_async<16>(
                umat[arena] + (wave - 4) * 512,
                cs_u + int64_t(ht) * C * D + offset);
        }

        if (wave == 0 && lane < 32) {
            global_to_lds_async<16>(
                mqk[arena],
                ws_mqk + int64_t(ht) * C * C + lane * 8);
        }
        if (wave == 1 && lane < 32) {
            global_to_lds_async<16>(
                gtot[arena],
                ws_gt + int64_t(ht) * D + lane * 4);
        }
    };

    // CACHE_DECAY_LDS moves the exact per-chunk exp2 to the existing DMA
    // publication points.  Wave 1 retains the original f32x4 ownership, and
    // the following pre-existing CTA barrier publishes the in-place
    // conversion without another rendezvous or arena.
    auto cache_decay = [&](int arena) {
        if constexpr (CACHE_DECAY_LDS) {
            if (wave == 1 && lane < 32) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int d = lane * 4 + i;
                    gtot[arena][d] = ex2(gtot[arena][d]);
                }
            }
        }
    };

    if constexpr (StageSin) {
        __bf16* sin = sin_storage.values;
        const __bf16* sin_src = cs_sin + int64_t(ss) * D * D;
        #pragma unroll
        for (int r = 0; r < C; ++r) {
            const int row = wave * C + r;
            global_to_lds_async<4>(
                sin + row * SD,
                sin_src + row * D + lane * 2);
        }
    }
    dma(0, ht_base);
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    cache_decay(0);
    __syncthreads();

    if constexpr (StageSin) {
        const __bf16* sin = sin_storage.values;
        #pragma unroll
        for (int kt = 0; kt < NKB; ++kt) {
            const int vv = v0 + (lane & 15);
            const int kk = kt * C + (lane >> 4) * 4;
            const bf16x4 fragment =
                *reinterpret_cast<const bf16x4*>(sin + vv * SD + kk);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                sreg[kt][i] = bf16_to_f32(fragment[i]);
        }
    }

    for (int j = 0; j < chunks; ++j) {
        const int arena = Arenas == 2 ? (j & 1) : (j == 3 ? 0 : j);
        const bool has_next = j + 1 < chunks;
        if constexpr (Arenas == 2) {
            if (has_next) dma(arena ^ 1, ht_base + j + 1);
        } else {
            // Publish chunks 1 and 2 together while chunk 0 computes.  The
            // first rendezvous both makes those disjoint arenas visible and
            // retires every chunk-0 read before arena 0 is reused.  Chunk 3
            // can then load throughout the independent chunk-1/chunk-2 work.
            // This removes one full-CTA rendezvous from a complete BT64
            // replay while retaining two resident CTAs per gfx950 CU.
            if (j == 0) {
                if (chunks > 1) dma(1, ht_base + 1);
                if (chunks > 2) dma(2, ht_base + 2);
            } else if (j == 1 && chunks > 3) {
                dma(0, ht_base + 3);
            }
        }

        // Keep the established K16 accumulation order so the production GLL
        // operator remains bit-identical to the common replay kernel.  K32
        // MFMA measured within noise here; the speedup comes from overlapping
        // the next global-to-LDS fill with current-arena computation.
        f32x4 o1 = gemm_regB<SD, NKB>(
            qd[arena], sreg, lane);
        f32x4 o2 = mm_std_tile_bf16(
            mqk[arena], umat[arena], v0, D, lane);
        const int seq_remaining = VL
            ? int(cu_seqlens[seq_idx + 1] - (t0_base + j * C))
            : T_seq - (blockIdx.x * 64 + j * C);
        const int alen = min(C, seq_remaining);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int m = (lane >> 4) * 4 + i;
            const int vv = v0 + (lane & 15);
            if (m < alen) {
                const __bf16 a = f32_to_bf16(o1[i]);
                const __bf16 b = f32_to_bf16(o2[i]);
                out_g[(int64_t(t0_base + j * C + m) * H + h) * D + vv] =
                    f32_to_bf16(bf16_to_f32(a) + bf16_to_f32(b));
            }
        }

        #pragma unroll
        for (int kt = 0; kt < NKB; ++kt) {
            f32x4 c = mm_contract_first_bf16(
                kr[arena], umat[arena], kt * C, v0, D, D, lane);
            const int kbase = kt * C + (lane >> 4) * 4;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                if constexpr (CACHE_DECAY_LDS)
                    sreg[kt][i] =
                        sreg[kt][i] * gtot[arena][kbase + i] + c[i];
                else
                    sreg[kt][i] = sreg[kt][i] *
                        ex2(gtot[arena][kbase + i]) + c[i];
            }
        }

        if constexpr (Arenas == 2) {
            if (has_next) {
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                cache_decay(arena ^ 1);
                __syncthreads();
            }
        } else {
            const bool publish_prefetch = j == 0 && chunks > 1;
            const bool publish_reuse = j == 2 && chunks > 3;
            if (publish_prefetch || publish_reuse) {
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                if (publish_prefetch) {
                    cache_decay(1);
                    if (chunks > 2) cache_decay(2);
                } else {
                    cache_decay(0);
                }
                __syncthreads();
            }
        }
    }
#endif
}

}  // namespace flashkda_hip::gfx950
