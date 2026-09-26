// gfx950-private dense N=1/H=12 direct-replay K-split prototype.
//
// This header is intentionally not connected to policy dispatch.  A launch must
// prove the complete contract before selecting it:
//
//   * dense N=1, H=12, D=128, T in {256, 512};
//   * every tile is a complete C16 tile and NT == T / 16;
//   * beta and decay are the activated/cached BT16 publications;
//   * grid=(H*8, 1, 1), block=(KSPLIT_WAVES*64, 1, 1).
//
// One CTA owns one (head, V16) state slab.  Its waves own disjoint contiguous K
// ranges.  Kd@state and Qd@state are reduced through lane-major FP32 LDS in
// strictly increasing K-shard order.  Wave zero alone forms vnew, INV@vnew,
// Mqk@u, and output, then broadcasts the once-rounded BF16 u fragment.  Every
// wave applies Kr@u and decay only to its private K range.  There are no atomics
// and no inter-CTA dependencies, so eager and captured-graph executions have
// the same reduction and publication order.
#pragma once

#include <cstddef>

#include "k2_kda_context_parallel_kernel.hpp"

namespace flashkda_hip::gfx950 {
namespace context_direct_ksplit_detail {

inline constexpr int kChunk = 16;
inline constexpr int kDim = 128;
inline constexpr int kHeads = 12;
inline constexpr int kVTile = 16;
inline constexpr int kKTiles = kDim / kChunk;
inline constexpr int kPaddedDim = kDim + 4;
inline constexpr int kLanes = 64;
inline constexpr int kProducts = 2;  // Kd@state and Qd@state.

inline constexpr std::size_t kKdBytes =
    kChunk * kPaddedDim * sizeof(__bf16);
inline constexpr std::size_t kQdBytes = kKdBytes;
inline constexpr std::size_t kKrBytes =
    kChunk * kDim * sizeof(__bf16);
inline constexpr std::size_t kInvBytes =
    kChunk * kChunk * sizeof(__bf16);
inline constexpr std::size_t kMqkBytes = kInvBytes;
inline constexpr std::size_t kDecayBytes = kDim * sizeof(float);
inline constexpr std::size_t kBetaBytes = kChunk * sizeof(float);
inline constexpr std::size_t kOperandBytes =
    kKdBytes + kQdBytes + kKrBytes + kInvBytes + kMqkBytes +
    kDecayBytes + kBetaBytes;
static_assert(kOperandBytes == 14144,
              "direct K-split operand LDS layout changed");

template <int KSPLIT_WAVES>
struct alignas(16) SharedStorage {
    // Kd/Qd retain the established padded Cx(D+4) MFMA-A layout.  Kr is tiled
    // as [K16][C][K16] for one transpose read per state K16 update.
    __bf16 kd[kChunk * kPaddedDim];
    __bf16 qd[kChunk * kPaddedDim];
    __bf16 kr[kChunk * kDim];
    __bf16 inv[kChunk * kChunk];
    __bf16 mqk[kChunk * kChunk];
    float decay[kDim];
    float beta[kChunk];

    // Each entry is the exact f32x4 fragment produced by one lane and one K
    // shard.  Wave zero consumes shards in ascending wave/K order.
    alignas(16) float partial[kProducts][KSPLIT_WAVES][kLanes][4];

    // The leader's rounded u fragment is already in MFMA register-B lane
    // mapping, so every K-shard wave reloads its own lane without a transpose.
    alignas(16) __bf16 u_broadcast[kLanes][4];
};

template <int KSPLIT_WAVES>
inline constexpr std::size_t kPartialBytes =
    kProducts * KSPLIT_WAVES * kLanes * 4 * sizeof(float);

template <int KSPLIT_WAVES>
inline constexpr std::size_t kBroadcastBytes =
    kLanes * 4 * sizeof(__bf16);

template <int KSPLIT_WAVES>
inline constexpr std::size_t kTotalBytes =
    kOperandBytes + kPartialBytes<KSPLIT_WAVES> +
    kBroadcastBytes<KSPLIT_WAVES>;

static_assert(offsetof(SharedStorage<2>, partial) == kOperandBytes,
              "KSPLIT_WAVES=2 partial LDS offset changed");
static_assert(offsetof(SharedStorage<4>, partial) == kOperandBytes,
              "KSPLIT_WAVES=4 partial LDS offset changed");
static_assert(sizeof(SharedStorage<2>) == kTotalBytes<2> &&
                  kTotalBytes<2> == 18752,
              "KSPLIT_WAVES=2 must use exactly 18,752 LDS bytes");
static_assert(sizeof(SharedStorage<4>) == kTotalBytes<4> &&
                  kTotalBytes<4> == 22848,
              "KSPLIT_WAVES=4 must use exactly 22,848 LDS bytes");

template <int KSPLIT_WAVES>
struct StagedOperands {
    static constexpr int kThreads = KSPLIT_WAVES * kLanes;
    static constexpr int kRowVectors = (kChunk * kDim) / 8;
    static constexpr int kVectorsPerThread = kRowVectors / kThreads;
    static_assert(kRowVectors % kThreads == 0,
                  "K-split staging requires an even vector assignment");

    bf16x8 kd[kVectorsPerThread];
    bf16x8 qd[kVectorsPerThread];
    bf16x8 kr[kVectorsPerThread];
    bf16x8 inv;
    bf16x8 mqk;
    f32x4 decay;
    float beta;
    bf16x4 v;
};

template <int KSPLIT_WAVES>
__device__ __forceinline__ void stage_operands(
        StagedOperands<KSPLIT_WAVES>& staged,
        const __bf16* __restrict__ v_g,
        const __bf16* __restrict__ ws_kd,
        const __bf16* __restrict__ ws_qd,
        const __bf16* __restrict__ ws_kr,
        const float* __restrict__ ws_gt,
        const __bf16* __restrict__ ws_inv,
        const __bf16* __restrict__ ws_mqk,
        const float* __restrict__ beta_cache,
        int ht,
        int chunk,
        int h,
        int v0,
        int tid,
        int wave,
        int lane) {
    const int tile_offset = ht * kChunk * kDim;
    #pragma unroll
    for (int j = 0; j < StagedOperands<KSPLIT_WAVES>::kVectorsPerThread;
         ++j) {
        const int vi =
            tid + j * StagedOperands<KSPLIT_WAVES>::kThreads;
        staged.kd[j] = reinterpret_cast<const bf16x8*>(
            ws_kd + tile_offset)[vi];
        staged.qd[j] = reinterpret_cast<const bf16x8*>(
            ws_qd + tile_offset)[vi];
        staged.kr[j] = reinterpret_cast<const bf16x8*>(
            ws_kr + tile_offset)[vi];
    }

    if (tid < (kChunk * kChunk) / 8) {
        const int factor_offset = ht * kChunk * kChunk;
        staged.inv = reinterpret_cast<const bf16x8*>(
            ws_inv + factor_offset)[tid];
        staged.mqk = reinterpret_cast<const bf16x8*>(
            ws_mqk + factor_offset)[tid];
    }
    if (tid < kDim / 4)
        staged.decay = reinterpret_cast<const f32x4*>(
            ws_gt + ht * kDim)[tid];
    if (tid < kChunk)
        staged.beta = beta_cache[ht * kChunk + tid];

    if (wave == 0) {
        const int m0 = (lane >> 4) * 4;
        const int vv = lane & 15;
        const int token0 = chunk * kChunk + m0;
        const int base =
            token0 * (kHeads * kDim) + h * kDim + v0 + vv;
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            staged.v[i] = v_g[base + i * (kHeads * kDim)];
    }
}

template <int KSPLIT_WAVES>
__device__ __forceinline__ void commit_operands(
        SharedStorage<KSPLIT_WAVES>& smem,
        const StagedOperands<KSPLIT_WAVES>& staged,
        bf16x4& v_fragment,
        int tid,
        int wave) {
    #pragma unroll
    for (int j = 0; j < StagedOperands<KSPLIT_WAVES>::kVectorsPerThread;
         ++j) {
        const int vi =
            tid + j * StagedOperands<KSPLIT_WAVES>::kThreads;
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
            smem.kr + ktile * kChunk * kChunk + c * kChunk + ki) =
            staged.kr[j];
    }

    if (tid < (kChunk * kChunk) / 8) {
        reinterpret_cast<bf16x8*>(smem.inv)[tid] = staged.inv;
        reinterpret_cast<bf16x8*>(smem.mqk)[tid] = staged.mqk;
    }
    if (tid < kDim / 4)
        reinterpret_cast<f32x4*>(smem.decay)[tid] = staged.decay;
    if (tid < kChunk)
        smem.beta[tid] = staged.beta;
    if (wave == 0)
        v_fragment = staged.v;
}

template <int PRODUCT, int KSPLIT_WAVES>
__device__ __forceinline__ f32x4 reduce_partials_low_to_high(
        const SharedStorage<KSPLIT_WAVES>& smem,
        int lane) {
    static_assert(PRODUCT == 0 || PRODUCT == 1,
                  "K-split reduction product must be Kd or Qd");
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

}  // namespace context_direct_ksplit_detail

template <int KSPLIT_WAVES, bool HO = false, bool SFP32 = false>
__global__ void __launch_bounds__(KSPLIT_WAVES * 64)
k2_kda_context_direct_ksplit_n1_h12_kernel(
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
    using namespace context_direct_ksplit_detail;
    static_assert(KSPLIT_WAVES == 2 || KSPLIT_WAVES == 4,
                  "direct K-split supports exactly two or four waves");
    static_assert(kHeads == 12 && kDim == 128 && kChunk == 16,
                  "direct K-split is a strict K3 N1/H12/D128/C16 kernel");
    static_assert(kKTiles % KSPLIT_WAVES == 0,
                  "K128 must divide evenly across K-split waves");
    constexpr int kLocalKTiles = kKTiles / KSPLIT_WAVES;
    constexpr int kLocalK = kDim / KSPLIT_WAVES;
    static_assert(kLocalKTiles > 0 && (kLocalKTiles % 2) == 0,
                  "each K shard must contain an even number of K16 tiles");
    static_assert(kTotalBytes<KSPLIT_WAVES> <= 24 * 1024,
                  "direct K-split prototype exceeded its 24-KiB LDS budget");

    // N=1, H=12, full C16 coverage, and the exact launch geometry are host
    // promises.  Keep a uniform device guard for the two admitted time shapes
    // so an isolated prototype launch cannot walk a malformed workspace.
    if ((T_seq != 256 && T_seq != 512) || NT != T_seq / kChunk)
        return;

    const int tid = int(threadIdx.x);
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int flat = int(blockIdx.x);
    const int h = flat >> 3;
    const int v_group = flat & 7;
    const int v0 = v_group * kVTile;
    const int shard_k0 = wave * kLocalK;

    __shared__ SharedStorage<KSPLIT_WAVES> smem;

    // Public state is [N,H,V,K].  Each wave owns all V16 values in one private
    // contiguous K shard, so neither initialization nor final publication needs
    // a CTA exchange.
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
    commit_operands<KSPLIT_WAVES>(
        smem, staged, v_fragment, tid, wave);
    __syncthreads();

    for (int chunk = 0; chunk < NT; ++chunk) {
        const bool has_next = chunk + 1 < NT;
        if (has_next) {
            const int next_chunk = chunk + 1;
            stage_operands<KSPLIT_WAVES>(
                staged, v_g, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv,
                ws_mqk, beta_cache, h * NT + next_chunk, next_chunk,
                h, v0, tid, wave, lane);
        }

        // Pointer-offsetting the padded CxK matrices exposes exactly this
        // wave's contiguous K shard to the existing x32 register-B helper.
        const RegBPairX32 products =
            gemm_regb_even_x32_pair<kPaddedDim, kLocalKTiles>(
                smem.kd + shard_k0, smem.qd + shard_k0, state, lane);
        *reinterpret_cast<f32x4*>(smem.partial[0][wave][lane]) =
            products.first;
        *reinterpret_cast<f32x4*>(smem.partial[1][wave][lane]) =
            products.second;
        __syncthreads();

        if (wave == 0) {
            const f32x4 residual =
                reduce_partials_low_to_high<0>(smem, lane);
            const f32x4 from_state =
                reduce_partials_low_to_high<1>(smem, lane);

            bf16x4 vnew_bf;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int m = (lane >> 4) * 4 + i;
                const float vnew =
                    (bf16_to_f32(v_fragment[i]) - residual[i]) * smem.beta[m];
                vnew_bf[i] = f32_to_bf16(vnew);
            }

            const f32x4 u =
                context_mfma_row_major_a_reg_b(smem.inv, vnew_bf, lane);
            bf16x4 u_bf;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                u_bf[i] = f32_to_bf16(u[i]);
                smem.u_broadcast[lane][i] = u_bf[i];
            }

            const f32x4 from_local =
                context_mfma_row_major_a_reg_b(smem.mqk, u_bf, lane);
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
        __syncthreads();

        bf16x4 u_bf;
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            u_bf[i] = smem.u_broadcast[lane][i];

        #pragma unroll
        for (int local_kt = 0; local_kt < kLocalKTiles; ++local_kt) {
            const int global_kt = wave * kLocalKTiles + local_kt;
            const f32x4 carry = context_mfma_tiled_kr_reg_b(
                smem.kr, u_bf, global_kt, lane);
            const int kbase =
                global_kt * kChunk + (lane >> 4) * 4;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                state[local_kt][i] =
                    state[local_kt][i] * smem.decay[kbase + i] + carry[i];
            }
        }

        // No wave may overwrite the single operand arena until every K shard
        // has consumed Kr, decay, and the BF16 u broadcast.  The final chunk
        // has no following LDS producer and therefore needs no trailing fence.
        if (has_next) {
            // Current Kr/decay/u users are done; now publish the already
            // prefetched next chunk into the single shared operand arena.
            __syncthreads();
            commit_operands<KSPLIT_WAVES>(
                smem, staged, v_fragment, tid, wave);
            __syncthreads();
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
