// gfx950 post-preparation K1 fusion for the plain BT64 C-split pipeline.
//
// One 256-thread CTA owns one BT64 segment.  Its four waves independently
// solve the four BT16 diagonal factors, then the same CTA forms both BT32
// Schur tiles and the final BT64 cross block.  qd is read through VGPRs while
// kd/ki use LDS, and the phase union reuses solve operand storage afterward.
// The gfx950 default pads each 128-wide LDS row by four bf16 values.  This
// removes the dominant row-stride bank alias while keeping static LDS below
// 40 KiB, allowing four CTAs/CU and preserving the workspace ABI.
//
// Every explicit fp16/bf16 rounding in the former three-kernel sequence is
// retained.  In particular, chunk 3 is rounded after its BT32 decay and again
// after its BT64 decay; removing that intermediate rounding changes K2 bits.
// ELIDE_DEAD_STORES removes only destinations with no later consumer, while
// MERGE_PRE_SOLVED_LOADS changes only when disjoint global-to-LDS transfers are
// issued.  FRAGMENT_FORWARD retains pair/BT64 MFMA B fragments in VGPRs while
// preserving the rounded LDS hand-off for every cross-wave consumer.  All
// template axes default off so the established specialization is still
// available for exact code-generation rollback.
#pragma once

#include "k1_kda_common.hpp"
#include "plain_suffix_decay.hpp"

namespace flashkda_hip::gfx950 {

namespace k1_postprep_fused_detail {

constexpr int C = 16;
constexpr int D = 128;
constexpr int CHUNKS = 4;
constexpr int TILE_ELEMS = C * D;
constexpr int FACTOR_ELEMS = C * C;

template <int LD>
struct alignas(16) SolveAux {
    __bf16 ki[CHUNKS * C * LD];
    _Float16 lm[CHUNKS * FACTOR_ELEMS];
    _Float16 lk[CHUNKS * FACTOR_ELEMS];
};

struct alignas(16) PairScratch {
    __bf16 l10[2 * FACTOR_ELEMS];
    __bf16 tmp[2 * FACTOR_ELEMS];
};

struct alignas(16) Bt64Scratch {
    __bf16 lm[4 * FACTOR_ELEMS];
    __bf16 tmp[4 * FACTOR_ELEMS];
};

struct alignas(16) PairScratchForward {
    __bf16 l10[2 * FACTOR_ELEMS];
};

struct alignas(16) Bt64ScratchForward {
    __bf16 lm[4 * FACTOR_ELEMS];
    // Only tmp0/tmp1 cross from row-block zero to row-block one.  Every wave
    // consumes its own rounded tmp fragment directly from VGPRs.
    __bf16 tmp[2 * FACTOR_ELEMS];
};

template <bool FRAGMENT_FORWARD>
struct FragmentScratchSelector;

template <>
struct FragmentScratchSelector<false> {
    using Pair = PairScratch;
    using Bt64 = Bt64Scratch;
};

template <>
struct FragmentScratchSelector<true> {
    using Pair = PairScratchForward;
    using Bt64 = Bt64ScratchForward;
};

template <int LD, bool FRAGMENT_FORWARD = false>
struct alignas(16) MergeAux {
    using Scratch = FragmentScratchSelector<FRAGMENT_FORWARD>;
    // During BT32 these slots hold kr0/kr2.  The pair barrier ends kr2's
    // lifetime, after which slot 1 is overwritten with kr1 for BT64.
    __bf16 kr[2 * C * LD];
    __bf16 cross32[2 * FACTOR_ELEMS];
    union alignas(16) {
        typename Scratch::Pair pair;
        typename Scratch::Bt64 bt64;
    } scratch;
};

template <int LD, bool FRAGMENT_FORWARD = false>
union alignas(16) PhaseStorage {
    SolveAux<LD> solve;
    MergeAux<LD, FRAGMENT_FORWARD> merge;
};

template <int LD, bool FRAGMENT_FORWARD = false>
struct alignas(16) SharedStorage {
    // kd lives through all three logical stages.  inv starts as fp16 during
    // each diagonal solve and is converted in place to the ABI's bf16 factor.
    __bf16 kd[CHUNKS * C * LD];
    alignas(16) unsigned char inv[CHUNKS * FACTOR_ELEMS * sizeof(__bf16)];
    PhaseStorage<LD, FRAGMENT_FORWARD> phase;
};

// PRE_SOLVED never touches the triangular-solve phase.  Keeping SolveAux in
// its static shared object nevertheless charged the kernel for the full 39
// KiB allocation and limited gfx950 to four resident CTAs.  The merge-only
// object stays below 32 KiB after its wave-local beta/decay values move to
// registers, admitting five CTAs in gfx950's 160-KiB LDS.
template <int LD, bool FRAGMENT_FORWARD = false>
struct alignas(16) PreSolvedSharedStorage {
    __bf16 kd[CHUNKS * C * LD];
    alignas(16) unsigned char inv[
        CHUNKS * FACTOR_ELEMS * sizeof(__bf16)];
    MergeAux<LD, FRAGMENT_FORWARD> merge;
};

template <int LD, bool PRE_SOLVED, bool FRAGMENT_FORWARD = false>
struct SharedStorageSelector;

template <int LD, bool FRAGMENT_FORWARD>
struct SharedStorageSelector<LD, false, FRAGMENT_FORWARD> {
    using type = SharedStorage<LD, FRAGMENT_FORWARD>;
};

template <int LD, bool FRAGMENT_FORWARD>
struct SharedStorageSelector<LD, true, FRAGMENT_FORWARD> {
    using type = PreSolvedSharedStorage<LD, FRAGMENT_FORWARD>;
};

static_assert(sizeof(SolveAux<D>) == 20480,
              "unexpected fused K1 solve scratch size");
static_assert(sizeof(MergeAux<D>) <= sizeof(SolveAux<D>),
              "merge phase must reuse solve scratch without growing LDS");
static_assert(sizeof(SharedStorage<D>) == 38912,
              "fused post-prep K1 must use exactly 38 KiB LDS");
static_assert(sizeof(SharedStorage<D + 4>) == 39936,
              "padded fused post-prep K1 must use exactly 39 KiB LDS");
static_assert(sizeof(SharedStorage<D + 4>) <= 40 * 1024,
              "fused post-prep K1 crossed its four-CTA LDS budget");
static_assert(sizeof(PreSolvedSharedStorage<D + 4>) == 32512,
              "merge-only fused K1 must stay below the 32-KiB boundary");
static_assert(sizeof(PreSolvedSharedStorage<D + 4>) <= 32 * 1024,
              "merge-only fused K1 lost five-CTA gfx950 residency");
static_assert(sizeof(PreSolvedSharedStorage<D + 4, true>) == 31488,
              "fragment-forwarded fused K1 must use exactly 30.75 KiB LDS");

__device__ __forceinline__ void wait_wave_lds() {
#if defined(__HIP_DEVICE_COMPILE__)
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
    __syncwarp();
}

__device__ __forceinline__ f32x4 mm16(
        const __bf16* a, const __bf16* b, int lane) {
    return mm_std_16_tr(a, b, lane);
}

// An MFMA accumulator stored by store_acc_16x16 and read back as the B operand
// of mm_std_16_tr has this exact lane mapping.  Callers perform the explicit
// BF16 conversion before entering so the removed LDS round trip does not move
// or remove a numerical rounding point.
__device__ __forceinline__ f32x4 mm16_reg_b(
        const __bf16* a, bf16x4 b, int lane) {
    const int row = lane & 15;
    const int kb = (lane >> 4) * 4;
    bf16x4 af;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        af[i] = a[row * C + kb + i];
    const f32x4 zero = {0.f, 0.f, 0.f, 0.f};
    return mfma_bf16(af, b, zero);
}

template <int SD, bool USE_GLL>
__device__ __forceinline__ void load_kr_tile(
        __bf16* dst, const __bf16* src, int lane) {
    if constexpr (USE_GLL) {
        if constexpr (SD == D)
            gll_bf16_vec(dst, src, TILE_ELEMS, lane);
        else
            gll_rows_pad(dst, SD, src, D, C, lane);
    } else {
        copy_bf16_rows(dst, SD, src, D, C, D, lane);
    }
}

}  // namespace k1_postprep_fused_detail

template <bool VL, bool USE_X32 = true, bool USE_GLL = true,
          bool PADDED = true, bool PRE_SOLVED = false,
          bool BETA_ACTIVATED = false,
          bool PUBLISH_SUFFIX_DECAY = false,
          bool ELIDE_DEAD_STORES = false,
          bool MERGE_PRE_SOLVED_LOADS = false,
          bool FRAGMENT_FORWARD = false>
__global__ void __launch_bounds__(256)
k1_kda_postprep_fused_kernel(
        const float* __restrict__ beta_g,
        __bf16* __restrict__ ws_kd,
        const __bf16* __restrict__ ws_qd,
        const __bf16* __restrict__ ws_kr,
        const float* __restrict__ ws_gt,
        const __bf16* __restrict__ tmp_kinv,
        __bf16* __restrict__ ws_inv,
        __bf16* __restrict__ ws_mqk,
        __bf16* __restrict__ cross32_g,
        __bf16* __restrict__ cross64_g,
        float* __restrict__ suffix_decay_g,
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ tile_prefix,
        const int* __restrict__ pair_prefix,
        const int* __restrict__ segment_prefix,
        int N, int total_tiles, int total_pairs, int total_segments,
        int T_seq, int H, int NT) {
    using namespace k1_postprep_fused_detail;
    const int tid = threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    constexpr int SD = PADDED ? D + 4 : D;
    constexpr int TILE_STORAGE = C * SD;
    static_assert(!BETA_ACTIVATED || PRE_SOLVED,
                  "activated beta requires the solved producer contract");
    static_assert(!PUBLISH_SUFFIX_DECAY || PRE_SOLVED,
                  "suffix decay requires the solved post-prep contract");
    static_assert(!MERGE_PRE_SOLVED_LOADS || PRE_SOLVED,
                  "merged operand loads require the solved post-prep contract");
    static_assert(!FRAGMENT_FORWARD ||
                  (PRE_SOLVED && PADDED && USE_X32 &&
                   ELIDE_DEAD_STORES && MERGE_PRE_SOLVED_LOADS),
                  "fragment forwarding requires the production solved route");

    int h, ht0, xp0, xs, t0, alen;
    if constexpr (VL) {
        const int gsi = blockIdx.x;
        h = blockIdx.y;
        int lo = 0, hi = N;
        while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (segment_prefix[mid] <= gsi) lo = mid; else hi = mid;
        }
        const int local_seg = gsi - segment_prefix[lo];
        const int64_t bos = cu_seqlens[lo];
        const int len = int(cu_seqlens[lo + 1] - bos);
        if (local_seg >= (len + 63) / 64) return;
        alen = min(64, len - local_seg * 64);
        ht0 = h * total_tiles + tile_prefix[lo] + local_seg * 4;
        xp0 = h * total_pairs + pair_prefix[lo] + local_seg * 2;
        xs = h * total_segments + gsi;
        t0 = int(bos) + local_seg * 64;
    } else {
        const int seg = blockIdx.x;
        const int bh = blockIdx.y;
        const int b = bh / H;
        h = bh % H;
        alen = min(64, T_seq - seg * 64);
        ht0 = bh * NT + seg * 4;
        xp0 = bh * ((NT + 1) / 2) + seg * 2;
        xs = bh * ((NT + 3) / 4) + seg;
        t0 = b * T_seq + seg * 64;
    }

    if constexpr (PUBLISH_SUFFIX_DECAY) {
        if (tid < D) {
            const int nch = (alen + C - 1) / C;
            const f32x4 g = {
                ws_gt[int64_t(ht0) * D + tid],
                nch > 1 ? ws_gt[int64_t(ht0 + 1) * D + tid] : 0.0f,
                nch > 2 ? ws_gt[int64_t(ht0 + 2) * D + tid] : 0.0f,
                nch > 3 ? ws_gt[int64_t(ht0 + 3) * D + tid] : 0.0f};
            store_plain_bt64_suffix_decay(
                suffix_decay_g, xs, tid, plain_bt64_suffix_decay(g));
        }
    }

    using Storage = typename SharedStorageSelector<
        SD, PRE_SOLVED, FRAGMENT_FORWARD>::type;
    __shared__ Storage smem;
    auto* const inv_h = reinterpret_cast<_Float16*>(smem.inv);
    auto* const inv_b = reinterpret_cast<__bf16*>(smem.inv);

    // ---- Four independent BT16 solves -------------------------------------
    const int chunk = wave;
    const int chunk_rows = min(C, max(0, alen - chunk * C));
    __bf16* const kd = smem.kd + chunk * TILE_STORAGE;
    _Float16* lm = nullptr;
    _Float16* ci = nullptr;
    _Float16* lk = nullptr;
    __bf16* ki = nullptr;
    if constexpr (!PRE_SOLVED) {
        SolveAux<SD>& solve = smem.phase.solve;
        lm = solve.lm + chunk * FACTOR_ELEMS;
        ci = inv_h + chunk * FACTOR_ELEMS;
        lk = solve.lk + chunk * FACTOR_ELEMS;
        ki = solve.ki + chunk * TILE_STORAGE;
    }

    // The strict opt-in PRE_SOLVED schedule issues the pair-local kr0/kr2 DMA
    // into the disjoint merge arena before the first completion point.  Keep
    // the beta/decay scalars live across that existing CTA barrier so the pair
    // phase does not need a second serialized vmcnt(0).  The default template
    // leaves this entire block, including its register lifetimes, absent.
    float merged_pair_beta_lane = 0.0f;
    float merged_pair_decay_lo = 0.0f;
    float merged_pair_decay_hi = 0.0f;

    if (chunk_rows > 0) {
        const int ht = ht0 + chunk;
        if constexpr (USE_GLL) {
            if constexpr (SD == D) {
                gll_bf16_vec(kd, ws_kd + int64_t(ht) * TILE_ELEMS,
                             TILE_ELEMS, lane);
            } else {
                gll_rows_pad(kd, SD,
                    ws_kd + int64_t(ht) * TILE_ELEMS, D, C, lane);
            }
            if constexpr (PRE_SOLVED)
                gll_bf16_vec(inv_b + chunk * FACTOR_ELEMS,
                    ws_inv + int64_t(ht) * FACTOR_ELEMS,
                    FACTOR_ELEMS, lane);
            else if constexpr (SD == D)
                gll_bf16_vec(ki,
                    tmp_kinv + int64_t(ht) * TILE_ELEMS,
                    TILE_ELEMS, lane);
            else
                gll_rows_pad(ki, SD,
                    tmp_kinv + int64_t(ht) * TILE_ELEMS, D, C, lane);
        } else {
            copy_bf16_rows(kd, SD,
                ws_kd + int64_t(ht) * TILE_ELEMS, D, C, D, lane);
            if constexpr (PRE_SOLVED)
                copy_bf16_vec(inv_b + chunk * FACTOR_ELEMS,
                    ws_inv + int64_t(ht) * FACTOR_ELEMS,
                    FACTOR_ELEMS, lane);
            else
                copy_bf16_rows(ki, SD,
                    tmp_kinv + int64_t(ht) * TILE_ELEMS, D, C, D, lane);
        }
        if constexpr (MERGE_PRE_SOLVED_LOADS) {
            const int merged_pair = wave;
            const bool merged_pair_valid =
                merged_pair < 2 && (merged_pair * 32 + C < alen);
            if (merged_pair_valid) {
                const int first_chunk = merged_pair * 2;
                const int second_chunk = first_chunk + 1;
                const int second_rows =
                    min(C, alen - merged_pair * 32 - C);
                load_kr_tile<SD, USE_GLL>(
                    smem.merge.kr + merged_pair * TILE_STORAGE,
                    ws_kr + int64_t(ht0 + first_chunk) * TILE_ELEMS,
                    lane);
                if (lane < second_rows) {
                    if constexpr (BETA_ACTIVATED)
                        merged_pair_beta_lane = beta_g[
                            int64_t(ht0 + second_chunk) * C + lane];
                    else
                        merged_pair_beta_lane = sigmoid_tanh(beta_g[
                            int64_t(t0 + merged_pair * 32 + C + lane) *
                            H + h]);
                }
                merged_pair_decay_lo = ex2(
                    ws_gt[int64_t(ht0 + first_chunk) * D + lane]);
                merged_pair_decay_hi = ex2(
                    ws_gt[int64_t(ht0 + first_chunk) * D + lane + 64]);
            }
        }
        if constexpr (PRE_SOLVED)
            wait_wave_lds();

        if constexpr (!PRE_SOLVED) {
        const float beta_lane = lane < chunk_rows
            ? sigmoid_tanh(beta_g[int64_t(t0 + chunk * C + lane) * H + h])
            : 0.0f;
        // Keep the scalar activation in flight while the asynchronous tile
        // loads complete.  PRE_SOLVED has no equivalent independent work and
        // therefore waits immediately above.
        wait_wave_lds();
        // qd is consumed once, so moving its fragments through VGPRs avoids a
        // 16 KiB CTA-wide LDS allocation.  kd/ki remain in LDS and the MFMA
        // sequence/order is identical to contract_last_x32.
        const __bf16* const qd_g =
            ws_qd + int64_t(ht) * TILE_ELEMS;
        f32x4 cl = {0.f, 0.f, 0.f, 0.f};
        f32x4 cm = {0.f, 0.f, 0.f, 0.f};
#if defined(__gfx950__)
        if constexpr (USE_X32) {
            const int row = lane & 15;
            const int kb = (lane >> 4) * 8;
            #pragma unroll
            for (int k0 = 0; k0 < D; k0 += 32) {
                bf16x8 kdf, qdf, kif;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    kdf[i] = kd[row * SD + k0 + kb + i];
                    qdf[i] = qd_g[row * D + k0 + kb + i];
                    kif[i] = ki[row * SD + k0 + kb + i];
                }
                cl = __builtin_amdgcn_mfma_f32_16x16x32_bf16(
                    kdf, kif, cl, 0, 0, 0);
                cm = __builtin_amdgcn_mfma_f32_16x16x32_bf16(
                    qdf, kif, cm, 0, 0, 0);
            }
        } else
#endif
        {
            const int row = lane & 15;
            const int kb = (lane >> 4) * 4;
            #pragma unroll
            for (int k0 = 0; k0 < D; k0 += 16) {
                bf16x4 kdf, qdf, kif;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    kdf[i] = kd[row * SD + k0 + kb + i];
                    qdf[i] = qd_g[row * D + k0 + kb + i];
                    kif[i] = ki[row * SD + k0 + kb + i];
                }
                cl = mfma_bf16(kdf, kif, cl);
                cm = mfma_bf16(qdf, kif, cm);
            }
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int m = (lane >> 4) * 4 + i;
            const int n = lane & 15;
            const float beta_m = __shfl(beta_lane, m);
            lm[m * C + n] = m > n
                ? f32_to_f16(cl[i]) * f32_to_f16(beta_m)
                : (_Float16)0.0f;
            ws_mqk[int64_t(ht) * FACTOR_ELEMS + m * C + n] = m >= n
                ? f32_to_bf16(cm[i]) : (__bf16)0.0f;
            ci[m * C + n] =
                (_Float16)(m == n ? 1.0f : 0.0f) - lm[m * C + n];
        }
        __syncwarp();

        { f32x4 c = gemm_std_f16_tr(lm, lm, lane);
          store_acc_16x16(lk, c, lane); }
        __syncwarp();
        { f32x4 c = gemm_std_f16_tr(ci, lk, lane); __syncwarp();
          #pragma unroll
          for (int i = 0; i < 4; ++i) {
              const int m = (lane >> 4) * 4 + i, n = lane & 15;
              ci[m * C + n] += f32_to_f16(c[i]);
          } }
        __syncwarp();
        { f32x4 c = gemm_std_f16_tr(lm, lm, lane);
          store_acc_16x16(lk, c, lane); }
        __syncwarp();
        { f32x4 c = gemm_std_f16_tr(lk, lk, lane); __syncwarp();
          store_acc_16x16(lk, c, lane); }
        __syncwarp();
        { f32x4 c = gemm_std_f16_tr(ci, lk, lane); __syncwarp();
          #pragma unroll
          for (int i = 0; i < 4; ++i) {
              const int m = (lane >> 4) * 4 + i, n = lane & 15;
              ci[m * C + n] += f32_to_f16(c[i]);
          } }
        __syncwarp();
        { f32x4 c = gemm_std_f16_tr(lm, lm, lane);
          store_acc_16x16(lk, c, lane); }
        __syncwarp();
        { f32x4 c = gemm_std_f16_tr(lk, lk, lane); __syncwarp();
          store_acc_16x16(lk, c, lane); }
        __syncwarp();
        { f32x4 c = gemm_std_f16_tr(lk, lk, lane); __syncwarp();
          store_acc_16x16(lk, c, lane); }
        __syncwarp();
        { f32x4 c = gemm_std_f16_tr(ci, lk, lane); __syncwarp();
          #pragma unroll
          for (int i = 0; i < 4; ++i) {
              const int m = (lane >> 4) * 4 + i, n = lane & 15;
              ci[m * C + n] += f32_to_f16(c[i]);
          } }
        __syncwarp();

        // Convert the factor in place only after every fp16 consumer is done.
        // The resulting bf16 tile is both the ABI write and merge input.
        for (int idx = lane; idx < FACTOR_ELEMS; idx += 64) {
            const __bf16 c = f32_to_bf16(f16_to_f32(ci[idx]));
            ws_inv[int64_t(ht) * FACTOR_ELEMS + idx] = c;
            inv_b[chunk * FACTOR_ELEMS + idx] = c;
        }
        }
    } else {
        // Later BT64 waves read a complete four-tile LDS footprint.  Zero the
        // absent tail without touching out-of-range workspace addresses.
        for (int idx = lane; idx < TILE_ELEMS; idx += 64)
            kd[(idx / D) * SD + idx % D] = (__bf16)0.0f;
        for (int idx = lane; idx < FACTOR_ELEMS; idx += 64)
            inv_b[chunk * FACTOR_ELEMS + idx] = (__bf16)0.0f;
    }
    __syncthreads();

    // ---- Two independent BT32 merges --------------------------------------
    MergeAux<SD, FRAGMENT_FORWARD>* merge_ptr = nullptr;
    if constexpr (PRE_SOLVED)
        merge_ptr = &smem.merge;
    else
        merge_ptr = &smem.phase.merge;
    MergeAux<SD, FRAGMENT_FORWARD>& merge = *merge_ptr;
    auto& pair_s = merge.scratch.pair;
    const int pair = wave;
    const bool pair_valid = pair < 2 && (pair * 32 + C < alen);
    if (pair_valid) {
        const int first_chunk = pair * 2;
        const int second_chunk = first_chunk + 1;
        const int second_rows = min(C, alen - pair * 32 - C);
        // kr2 dies with pair 1, so use slot 1 and later replace it with kr1.
        __bf16* const kr_pair = merge.kr + pair * TILE_STORAGE;
        float pair_beta_lane = 0.0f;
        float pair_decay_lo = 0.0f;
        float pair_decay_hi = 0.0f;
        if constexpr (MERGE_PRE_SOLVED_LOADS) {
            pair_beta_lane = merged_pair_beta_lane;
            pair_decay_lo = merged_pair_decay_lo;
            pair_decay_hi = merged_pair_decay_hi;
        } else {
            if constexpr (USE_GLL) {
                if constexpr (SD == D)
                    gll_bf16_vec(kr_pair,
                        ws_kr + int64_t(ht0 + first_chunk) * TILE_ELEMS,
                        TILE_ELEMS, lane);
                else
                    gll_rows_pad(kr_pair, SD,
                        ws_kr + int64_t(ht0 + first_chunk) * TILE_ELEMS,
                        D, C, lane);
            } else {
                copy_bf16_rows(kr_pair, SD,
                    ws_kr + int64_t(ht0 + first_chunk) * TILE_ELEMS,
                    D, C, D, lane);
            }
            if (lane < second_rows) {
                if constexpr (BETA_ACTIVATED)
                    pair_beta_lane = beta_g[
                        int64_t(ht0 + second_chunk) * C + lane];
                else
                    pair_beta_lane = sigmoid_tanh(beta_g[
                        int64_t(t0 + pair * 32 + C + lane) * H + h]);
            }
            pair_decay_lo =
                ex2(ws_gt[int64_t(ht0 + first_chunk) * D + lane]);
            pair_decay_hi = ex2(
                ws_gt[int64_t(ht0 + first_chunk) * D + lane + 64]);
            // Hide the global->LDS transfer behind beta activation and decay.
            wait_wave_lds();
        }

        f32x4 l;
        if constexpr (USE_X32)
            l = contract_last_x32<D, SD, SD>(
                smem.kd + second_chunk * TILE_STORAGE,
                kr_pair, lane);
        else
            l = gemm_contract_last<__bf16, D, SD>(
                smem.kd + second_chunk * TILE_STORAGE,
                kr_pair, lane);
        __bf16* const l10 = pair_s.l10 + pair * FACTOR_ELEMS;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int m = (lane >> 4) * 4 + i, n = lane & 15;
            l10[m * C + n] =
                f32_to_bf16(l[i] * __shfl(pair_beta_lane, m));
        }
        __syncwarp();

        f32x4 x = mm16(l10,
            inv_b + first_chunk * FACTOR_ELEMS, lane);
        if constexpr (FRAGMENT_FORWARD) {
            bf16x4 tmp_fragment;
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                tmp_fragment[i] = f32_to_bf16(x[i]);
            x = mm16_reg_b(
                inv_b + second_chunk * FACTOR_ELEMS,
                tmp_fragment, lane);
        } else {
            __bf16* const tmp = pair_s.tmp + pair * FACTOR_ELEMS;
            store_acc_16x16(tmp, x, lane);
            __syncwarp();
            x = mm16(inv_b + second_chunk * FACTOR_ELEMS, tmp, lane);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int m = (lane >> 4) * 4 + i, n = lane & 15;
            const __bf16 c = f32_to_bf16(-x[i]);
            merge.cross32[pair * FACTOR_ELEMS + m * C + n] = c;
            cross32_g[int64_t(xp0 + pair) * FACTOR_ELEMS + m * C + n] = c;
        }
        if constexpr (!PADDED) __syncwarp();

        // Keep this bf16 store/rounding even though kd remains in LDS: the old
        // BT64 kernel reloaded this exact rounded BT32 result.
        for (int idx = lane; idx < TILE_ELEMS; idx += 64) {
            const int d = idx % D;
            const int m = idx / D;
            const __bf16 c = f32_to_bf16(
                bf16_to_f32(
                    smem.kd[second_chunk * TILE_STORAGE + m * SD + d]) *
                (d < 64 ? pair_decay_lo : pair_decay_hi));
            if constexpr (ELIDE_DEAD_STORES) {
                // chunk 1 is consumed only from the global ABI; chunk 3 is
                // consumed only from LDS before its final BT64 overwrite.
                // The BF16 conversion above remains the single shared rounding
                // point for both schedules.
                if (pair == 0)
                    ws_kd[int64_t(ht0 + second_chunk) * TILE_ELEMS + idx] = c;
                else
                    smem.kd[
                        second_chunk * TILE_STORAGE + m * SD + d] = c;
            } else {
                smem.kd[second_chunk * TILE_STORAGE + m * SD + d] = c;
                ws_kd[int64_t(ht0 + second_chunk) * TILE_ELEMS + idx] = c;
            }
        }

        // BT64 needs kr0 relative to the end of chunk 1.  Pair 0 is the last
        // raw-kr0 consumer, so its wave transforms the tile now.
        if (pair == 0 && alen > 32) {
            const int d0 = lane;
            const int d1 = lane + 64;
            const float e0 = ex2(ws_gt[int64_t(ht0 + 1) * D + d0]);
            const float e1 = ex2(ws_gt[int64_t(ht0 + 1) * D + d1]);
            #pragma unroll
            for (int m = 0; m < C; ++m) {
                kr_pair[m * SD + d0] = f32_to_bf16(
                    bf16_to_f32(kr_pair[m * SD + d0]) * e0);
                kr_pair[m * SD + d1] = f32_to_bf16(
                    bf16_to_f32(kr_pair[m * SD + d1]) * e1);
            }
        }
    }
    __syncthreads();

    // A one-chunk segment has no BT32 output; a two-chunk segment ends above.
    // Segments with chunk 2 execute the four-wave BT64 merge, with chunk 3
    // represented by zero tiles for the 33..48-token tail.
    if (alen <= 32) return;

    // ---- One four-wave BT64 merge -----------------------------------------
    auto& bt64 = merge.scratch.bt64;
    const int rb = wave >> 1;
    const int cbk = wave & 1;
    // kr2 is dead after the pair barrier.  Wave 1 replaces that slot with kr1
    // while the CTA prepares beta and the combined decay.
    if (wave == 1) {
        load_kr_tile<SD, USE_GLL>(
            merge.kr + TILE_STORAGE,
            ws_kr + int64_t(ht0 + 1) * TILE_ELEMS, lane);
        if constexpr (!PADDED) wait_wave_lds();
    }
    float bt64_beta_lane = 0.0f;
    if (lane < C && rb * C + lane + 32 < alen) {
        if constexpr (BETA_ACTIVATED)
            bt64_beta_lane = beta_g[
                int64_t(ht0 + 2 + rb) * C + lane];
        else
            bt64_beta_lane = sigmoid_tanh(beta_g[
                int64_t(t0 + 32 + rb * C + lane) * H + h]);
    }
    const int decay_d = tid & (D - 1);
    // Each of the two row waves computes the 128 register-local values it
    // later consumes, avoiding another LDS publication phase.
    const float bt64_decay = ex2(
        ws_gt[int64_t(ht0) * D + decay_d] +
        ws_gt[int64_t(ht0 + 1) * D + decay_d]);
    // Wave 1's decay work is independent of the kr1 destination, so leave
    // the GLL in flight until immediately before the CTA consumes that tile.
    if constexpr (PADDED)
        if (wave == 1) wait_wave_lds();
    __syncthreads();

    f32x4 l;
    if constexpr (USE_X32)
        l = contract_last_x32<D, SD, SD>(
            smem.kd + (2 + rb) * TILE_STORAGE,
            merge.kr + cbk * TILE_STORAGE, lane);
    else
        l = gemm_contract_last<__bf16, D, SD>(
            smem.kd + (2 + rb) * TILE_STORAGE,
            merge.kr + cbk * TILE_STORAGE, lane);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int m = (lane >> 4) * 4 + i, n = lane & 15;
        bt64.lm[(wave * C + m) * C + n] =
            f32_to_bf16(l[i] * __shfl(bt64_beta_lane, m));
    }
    __syncthreads();

    f32x4 t;
    if (cbk == 0) {
        const f32x4 x0 = mm16(
            bt64.lm + (rb * 2) * FACTOR_ELEMS,
            inv_b, lane);
        const f32x4 x1 = mm16(
            bt64.lm + (rb * 2 + 1) * FACTOR_ELEMS,
            merge.cross32, lane);
        #pragma unroll
        for (int i = 0; i < 4; ++i) t[i] = x0[i] + x1[i];
    } else {
        t = mm16(bt64.lm + (rb * 2 + 1) * FACTOR_ELEMS,
                 inv_b + FACTOR_ELEMS, lane);
    }
    bf16x4 tmp_fragment;
    if constexpr (FRAGMENT_FORWARD) {
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            tmp_fragment[i] = f32_to_bf16(t[i]);
        // Only row-block zero is consumed by another wave.  The 33..48-token
        // tail has no cross-wave tmp consumer at all, so it needs neither an
        // LDS publication nor the otherwise mandatory CTA rendezvous.
        if (alen > 48 && rb == 0) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int m = (lane >> 4) * 4 + i;
                const int n = lane & 15;
                bt64.tmp[cbk * FACTOR_ELEMS + m * C + n] =
                    tmp_fragment[i];
            }
        }
        if (alen > 48)
            __syncthreads();
    } else {
        store_acc_16x16(bt64.tmp + wave * FACTOR_ELEMS, t, lane);
        __syncthreads();
    }

    f32x4 z;
    if (rb == 0) {
        if constexpr (FRAGMENT_FORWARD)
            z = mm16_reg_b(
                inv_b + 2 * FACTOR_ELEMS, tmp_fragment, lane);
        else
            z = mm16(inv_b + 2 * FACTOR_ELEMS,
                     bt64.tmp + cbk * FACTOR_ELEMS, lane);
    } else if (alen > 48) {
        if constexpr (FRAGMENT_FORWARD) {
            const f32x4 x0 = mm16(
                merge.cross32 + FACTOR_ELEMS,
                bt64.tmp + cbk * FACTOR_ELEMS, lane);
            const f32x4 x1 = mm16_reg_b(
                inv_b + 3 * FACTOR_ELEMS, tmp_fragment, lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i) z[i] = x0[i] + x1[i];
        } else {
            const f32x4 x0 = mm16(
                merge.cross32 + FACTOR_ELEMS,
                bt64.tmp + cbk * FACTOR_ELEMS, lane);
            const f32x4 x1 = mm16(
                inv_b + 3 * FACTOR_ELEMS,
                bt64.tmp + (2 + cbk) * FACTOR_ELEMS, lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i) z[i] = x0[i] + x1[i];
        }
    } else {
        z = {0.f, 0.f, 0.f, 0.f};
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int m = (lane >> 4) * 4 + i, n = lane & 15;
        cross64_g[(int64_t(xs) * 4 + wave) * FACTOR_ELEMS + m * C + n] =
            f32_to_bf16(-z[i]);
    }
    if constexpr (!PADDED) __syncthreads();

    // kd2/kd3 were pair-local; make them segment-local.  kd3 already carries
    // the bf16-rounded pair-1 decay from above, preserving both roundings.
    for (int idx = tid; idx < 2 * TILE_ELEMS; idx += 256) {
        const int m = idx / D;
        if (m + 32 < alen) {
            const int d = idx % D;
            const int chunk_b = 2 + (m >> 4);
            const int local = (m & 15) * D + d;
            const int local_s = (m & 15) * SD + d;
            const __bf16 c = f32_to_bf16(
                bf16_to_f32(smem.kd[chunk_b * TILE_STORAGE + local_s]) *
                bt64_decay);
            if constexpr (!ELIDE_DEAD_STORES)
                smem.kd[chunk_b * TILE_STORAGE + local_s] = c;
            ws_kd[int64_t(ht0 + chunk_b) * TILE_ELEMS + local] = c;
        }
    }
}

}  // namespace flashkda_hip::gfx950
