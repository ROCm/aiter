// gfx950-private dense N=1/H=12 direct-replay K-split tail-pipeline prototype.
//
// This candidate keeps the public ABI and K-shard ownership of the direct
// K-split kernel, but cuts the steady-state CTA synchronization count from four
// to two barriers per C16 chunk:
//
//   * Kd/Qd have one shared arena.  They are dead after every wave publishes
//     its two state contractions, so the first barrier permits an immediate
//     overwrite with the prefetched next chunk.
//   * Kr/INV/Mqk/decay/beta have two tail arenas.  The current tail remains
//     immutable through the post-u state update while the other tail receives
//     the next chunk.
//   * The second barrier publishes both the next operands and the leader's u.
//
// Wave zero reduces Kd@state and forms u.  Wave one independently reduces
// Qd@state, carries that fragment across the second barrier, and writes output
// after consuming u.  This removes Qd reduction, Mqk, and output publication
// from wave zero's serial critical section without atomics or scheduling-order
// dependencies.
//
// This header is intentionally not connected to policy dispatch.  A launch
// must prove the same complete contract as the original direct K-split kernel:
//
//   * dense N=1, H=12, D=128;
//   * LONG_SEQUENCE=false accepts T in {256, 512};
//   * LONG_SEQUENCE=true accepts T in {1024, 2048} and is tail4-only;
//   * every tile is a complete C16 tile and NT == T / 16;
//   * beta and decay are the activated/cached BT16 publications;
//   * grid=(H*8, 1, 1), block=(KSPLIT_WAVES*64, 1, 1).
#pragma once

#include <cstddef>

#include "k2_kda_context_direct_ksplit_kernel.hpp"

namespace flashkda_hip::gfx950 {
namespace context_direct_ksplit_tailpipe_detail {

using context_direct_ksplit_detail::StagedOperands;
using context_direct_ksplit_detail::kBetaBytes;
using context_direct_ksplit_detail::kBroadcastBytes;
using context_direct_ksplit_detail::kChunk;
using context_direct_ksplit_detail::kDecayBytes;
using context_direct_ksplit_detail::kDim;
using context_direct_ksplit_detail::kHeads;
using context_direct_ksplit_detail::kInvBytes;
using context_direct_ksplit_detail::kKdBytes;
using context_direct_ksplit_detail::kKTiles;
using context_direct_ksplit_detail::kKrBytes;
using context_direct_ksplit_detail::kLanes;
using context_direct_ksplit_detail::kMqkBytes;
using context_direct_ksplit_detail::kPaddedDim;
using context_direct_ksplit_detail::kPartialBytes;
using context_direct_ksplit_detail::kProducts;
using context_direct_ksplit_detail::kQdBytes;
using context_direct_ksplit_detail::kVTile;
using context_direct_ksplit_detail::stage_operands;

inline constexpr std::size_t kContractionBytes = kKdBytes + kQdBytes;
inline constexpr std::size_t kTailBytes =
    kKrBytes + kInvBytes + kMqkBytes + kDecayBytes + kBetaBytes;
static_assert(kContractionBytes == 8448 && kTailBytes == 5696,
              "tail-pipeline operand partition changed");

struct alignas(16) TailStorage {
    __bf16 kr[kChunk * kDim];
    __bf16 inv[kChunk * kChunk];
    __bf16 mqk[kChunk * kChunk];
    float decay[kDim];
    float beta[kChunk];
};
static_assert(sizeof(TailStorage) == kTailBytes,
              "tail-pipeline tail arena layout changed");

template <int KSPLIT_WAVES>
struct alignas(16) SharedStorage {
    // The contraction arena is overwritten only after the partial-ready
    // barrier proves that every wave has finished reading the current Kd/Qd.
    __bf16 kd[kChunk * kPaddedDim];
    __bf16 qd[kChunk * kPaddedDim];

    // A tail is reused two chunks later.  The next iteration's partial-ready
    // barrier proves that all waves finished the previous post-u tail users.
    TailStorage tail[2];

    alignas(16) float partial[kProducts][KSPLIT_WAVES][kLanes][4];
    alignas(16) __bf16 u_broadcast[kLanes][4];
};

template <int KSPLIT_WAVES>
inline constexpr std::size_t kTotalBytes =
    kContractionBytes + 2 * kTailBytes +
    kPartialBytes<KSPLIT_WAVES> + kBroadcastBytes<KSPLIT_WAVES>;

static_assert(offsetof(SharedStorage<2>, partial) == 19840,
              "KSPLIT_WAVES=2 partial offset changed");
static_assert(offsetof(SharedStorage<4>, partial) == 19840,
              "KSPLIT_WAVES=4 partial offset changed");
static_assert(sizeof(SharedStorage<2>) == kTotalBytes<2> &&
                  kTotalBytes<2> == 24448,
              "KSPLIT_WAVES=2 must use exactly 24,448 LDS bytes");
static_assert(sizeof(SharedStorage<4>) == kTotalBytes<4> &&
                  kTotalBytes<4> == 28544,
              "KSPLIT_WAVES=4 must use exactly 28,544 LDS bytes");

// Commit only shared operands.  The wave-zero V fragment deliberately remains
// in staged.v until the current residual has consumed v_fragment; copying it in
// this routine would clobber the current chunk before the leader computation.
template <int KSPLIT_WAVES>
__device__ __forceinline__ void commit_shared_operands(
        SharedStorage<KSPLIT_WAVES>& smem,
        const StagedOperands<KSPLIT_WAVES>& staged,
        int tail_index,
        int tid) {
    #pragma unroll
    for (int j = 0; j < StagedOperands<KSPLIT_WAVES>::kVectorsPerThread;
         ++j) {
        const int vi = tid + j * StagedOperands<KSPLIT_WAVES>::kThreads;
        const int row = vi >> 4;
        const int col8 = vi & 15;
        reinterpret_cast<bf16x8*>(
            smem.kd + row * kPaddedDim)[col8] = staged.kd[j];
        reinterpret_cast<bf16x8*>(
            smem.qd + row * kPaddedDim)[col8] = staged.qd[j];

        const int source_element = vi * 8;
        const int c = source_element / kDim;
        const int k = source_element - c * kDim;
        const int ktile = k / kChunk;
        const int ki = k - ktile * kChunk;
        *reinterpret_cast<bf16x8*>(
            smem.tail[tail_index].kr +
            ktile * kChunk * kChunk + c * kChunk + ki) = staged.kr[j];
    }

    if (tid < (kChunk * kChunk) / 8) {
        reinterpret_cast<bf16x8*>(smem.tail[tail_index].inv)[tid] =
            staged.inv;
        reinterpret_cast<bf16x8*>(smem.tail[tail_index].mqk)[tid] =
            staged.mqk;
    }
    if (tid < kDim / 4)
        reinterpret_cast<f32x4*>(smem.tail[tail_index].decay)[tid] =
            staged.decay;
    if (tid < kChunk)
        smem.tail[tail_index].beta[tid] = staged.beta;
}

template <int PRODUCT, int KSPLIT_WAVES>
__device__ __forceinline__ f32x4 reduce_partials_low_to_high(
        const SharedStorage<KSPLIT_WAVES>& smem,
        int lane) {
    static_assert(PRODUCT == 0 || PRODUCT == 1,
                  "tail-pipeline reduction product must be Kd or Qd");
    f32x4 sum = *reinterpret_cast<const f32x4*>(
        smem.partial[PRODUCT][0][lane]);
    #pragma unroll
    for (int shard = 1; shard < KSPLIT_WAVES; ++shard) {
        const f32x4 next = *reinterpret_cast<const f32x4*>(
            smem.partial[PRODUCT][shard][lane]);
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            sum[i] = sum[i] + next[i];
    }
    return sum;
}

__device__ __forceinline__ bf16x4 load_row_major_a_fragment(
        const __bf16* __restrict__ matrix,
        int lane) {
    const int row = lane & 15;
    const int col4 = (lane >> 4) * 4;
    bf16x4 fragment;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        fragment[i] = matrix[row * kChunk + col4 + i];
    return fragment;
}

}  // namespace context_direct_ksplit_tailpipe_detail

template <
    int KSPLIT_WAVES,
    bool HO = false,
    bool SFP32 = false,
    bool TAIL_MQK_PREFETCH = false,
    bool LONG_SEQUENCE = false>
__global__ void __launch_bounds__(KSPLIT_WAVES * 64)
k2_kda_context_direct_ksplit_tailpipe_n1_h12_kernel(
        const __bf16* __restrict__ v_g,          // dense [T,12,128]
        const float* __restrict__ beta_cache,    // activated [12*NT,16]
        __bf16* __restrict__ out_g,              // dense [T,12,128]
        const __bf16* __restrict__ ws_kd,        // [12*NT,16,128]
        const __bf16* __restrict__ ws_qd,        // [12*NT,16,128]
        const __bf16* __restrict__ ws_kr,        // [12*NT,16,128]
        const float* __restrict__ ws_gt,          // decay [12*NT,128]
        const __bf16* __restrict__ ws_inv,       // [12*NT,16,16]
        const __bf16* __restrict__ ws_mqk,       // [12*NT,16,16]
        const void* __restrict__ init_state,     // optional [1,12,128,128]
        void* __restrict__ final_state,          // HO [1,12,128,128]
        int T_seq,
        int NT) {
    using namespace context_direct_ksplit_tailpipe_detail;
    static_assert(KSPLIT_WAVES == 2 || KSPLIT_WAVES == 4,
                  "tail-pipeline supports exactly two or four waves");
    static_assert(!TAIL_MQK_PREFETCH || KSPLIT_WAVES == 4,
                  "Mqk register prefetch is a tail4-only experiment");
    static_assert(!LONG_SEQUENCE || KSPLIT_WAVES == 4,
                  "long-sequence tail-pipeline is a tail4-only experiment");
    static_assert(kKTiles % KSPLIT_WAVES == 0,
                  "K128 must divide evenly across K-split waves");
    constexpr int kLocalKTiles = kKTiles / KSPLIT_WAVES;
    constexpr int kLocalK = kDim / KSPLIT_WAVES;
    static_assert((kLocalKTiles % 2) == 0,
                  "each K shard must contain an even number of K16 tiles");
    static_assert(kTotalBytes<KSPLIT_WAVES> <= 32 * 1024,
                  "tail-pipeline exceeded its 32-KiB LDS budget");

    if constexpr (LONG_SEQUENCE) {
        if ((T_seq != 1024 && T_seq != 2048) || NT != T_seq / kChunk)
            return;
    } else {
        if ((T_seq != 256 && T_seq != 512) || NT != T_seq / kChunk)
            return;
    }

    const int tid = int(threadIdx.x);
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int flat = int(blockIdx.x);
    const int h = flat >> 3;
    const int v_group = flat & 7;
    const int v0 = v_group * kVTile;
    const int shard_k0 = wave * kLocalK;

    __shared__ SharedStorage<KSPLIT_WAVES> smem;

    float state[kLocalKTiles][4];
    const int state_slab = h * kDim * kDim;
    #pragma unroll
    for (int local_kt = 0; local_kt < kLocalKTiles; ++local_kt) {
        const int global_kt = wave * kLocalKTiles + local_kt;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int vv = v0 + (lane & 15);
            const int kk =
                global_kt * kChunk + (lane >> 4) * 4 + i;
            const int idx = state_slab + vv * kDim + kk;
            float value = 0.0f;
            if (init_state != nullptr) {
                value = SFP32
                    ? reinterpret_cast<const float*>(init_state)[idx]
                    : bf16_to_f32(
                        reinterpret_cast<const __bf16*>(init_state)[idx]);
            }
            state[local_kt][i] = value;
        }
    }

    StagedOperands<KSPLIT_WAVES> staged;
    bf16x4 v_fragment{};
    stage_operands<KSPLIT_WAVES>(
        staged, v_g, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk,
        beta_cache, h * NT, 0, h, v0, tid, wave, lane);
    commit_shared_operands<KSPLIT_WAVES>(smem, staged, 0, tid);
    if (wave == 0)
        v_fragment = staged.v;
    __syncthreads();

    for (int chunk = 0; chunk < NT; ++chunk) {
        const int current_tail_index = chunk & 1;
        const bool has_next = chunk + 1 < NT;
        if (has_next) {
            const int next_chunk = chunk + 1;
            stage_operands<KSPLIT_WAVES>(
                staged, v_g, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv,
                ws_mqk, beta_cache, h * NT + next_chunk, next_chunk,
                h, v0, tid, wave, lane);
        }

        const RegBPairX32 products =
            gemm_regb_even_x32_pair<kPaddedDim, kLocalKTiles>(
                smem.kd + shard_k0, smem.qd + shard_k0, state, lane);
        *reinterpret_cast<f32x4*>(smem.partial[0][wave][lane]) =
            products.first;
        *reinterpret_cast<f32x4*>(smem.partial[1][wave][lane]) =
            products.second;

        // Barrier A has three proofs:
        //   1. every current Kd/Qd reader is finished, so that single arena may
        //      receive the next chunk immediately after the barrier;
        //   2. every current partial is visible to both leader waves;
        //   3. for chunk > 0, every wave necessarily completed the previous
        //      post-u tail use before reaching this chunk's contraction, so the
        //      inactive tail may be reused for chunk + 1.
        __syncthreads();

        if (has_next) {
            commit_shared_operands<KSPLIT_WAVES>(
                smem, staged, current_tail_index ^ 1, tid);
        }

        f32x4 from_state = {0.f, 0.f, 0.f, 0.f};
        bf16x4 mqk_a = {};
        if (wave == 0) {
            const f32x4 residual =
                reduce_partials_low_to_high<0>(smem, lane);
            bf16x4 vnew_bf;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int m = (lane >> 4) * 4 + i;
                const float vnew =
                    (bf16_to_f32(v_fragment[i]) - residual[i]) *
                    smem.tail[current_tail_index].beta[m];
                vnew_bf[i] = f32_to_bf16(vnew);
            }

            const f32x4 u = context_mfma_row_major_a_reg_b(
                smem.tail[current_tail_index].inv, vnew_bf, lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                smem.u_broadcast[lane][i] = f32_to_bf16(u[i]);

            // staged.v is the only prefetched value still needed after the
            // shared commit.  Delay this register handoff until current V has
            // contributed to vnew.
            if (has_next)
                v_fragment = staged.v;
        } else if (wave == 1) {
            if constexpr (TAIL_MQK_PREFETCH) {
                mqk_a = load_row_major_a_fragment(
                    smem.tail[current_tail_index].mqk, lane);
            }
            from_state = reduce_partials_low_to_high<1>(smem, lane);
        }

        // Barrier B publishes the leader's u and all next-chunk shared
        // operands.  It also proves that both leaders finished reading the
        // current partial arena before the next iteration overwrites it.
        __syncthreads();

        bf16x4 u_bf;
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            u_bf[i] = smem.u_broadcast[lane][i];

        // Wave one owns output publication so Mqk and stores overlap the other
        // waves' disjoint K-shard state updates.
        if (wave == 1) {
            f32x4 from_local;
            if constexpr (TAIL_MQK_PREFETCH) {
                const f32x4 zero = {0.f, 0.f, 0.f, 0.f};
                from_local = mfma_bf16(mqk_a, u_bf, zero);
            } else {
                from_local = context_mfma_row_major_a_reg_b(
                    smem.tail[current_tail_index].mqk, u_bf, lane);
            }
            const int m0 = (lane >> 4) * 4;
            const int vv = lane & 15;
            const int token0 = chunk * kChunk + m0;
            const int base =
                token0 * (kHeads * kDim) + h * kDim + v0 + vv;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const __bf16 a = f32_to_bf16(from_state[i]);
                const __bf16 b = f32_to_bf16(from_local[i]);
                out_g[base + i * (kHeads * kDim)] = f32_to_bf16(
                    bf16_to_f32(a) + bf16_to_f32(b));
            }
        }

        #pragma unroll
        for (int local_kt = 0; local_kt < kLocalKTiles; ++local_kt) {
            const int global_kt = wave * kLocalKTiles + local_kt;
            const f32x4 carry = context_mfma_tiled_kr_reg_b(
                smem.tail[current_tail_index].kr,
                u_bf,
                global_kt,
                lane);
            const int kbase =
                global_kt * kChunk + (lane >> 4) * 4;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                state[local_kt][i] =
                    state[local_kt][i] *
                        smem.tail[current_tail_index].decay[kbase + i] +
                    carry[i];
            }
        }
    }

    if constexpr (HO) {
        #pragma unroll
        for (int local_kt = 0; local_kt < kLocalKTiles; ++local_kt) {
            const int global_kt = wave * kLocalKTiles + local_kt;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int vv = v0 + (lane & 15);
                const int kk =
                    global_kt * kChunk + (lane >> 4) * 4 + i;
                const int idx = state_slab + vv * kDim + kk;
                if constexpr (SFP32) {
                    reinterpret_cast<float*>(final_state)[idx] =
                        state[local_kt][i];
                } else {
                    reinterpret_cast<__bf16*>(final_state)[idx] =
                        f32_to_bf16(state[local_kt][i]);
                }
            }
        }
    }
}

}  // namespace flashkda_hip::gfx950
