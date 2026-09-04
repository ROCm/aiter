// gfx950-private fused BT16 K1 preparation and triangular solve.
//
// Four waves cooperatively build the elementwise workspace tiles.  The exact
// preparation specialization preserves the monolithic kernel's reduction and
// BF16 rounding order, while the vector specialization is retained for A/B.
// Exact preparation overlaps beta sigmoid work on producer wave 3 with the
// prefix scan on waves 0 and 1, then publishes beta after the phase union is
// safe to reuse.  Wave 0 executes either CDNA4 K32 or legacy K16 contractions
// plus the transposed-F16 polynomial inverse.  The ABI workspace is still
// fully published for K2.
#pragma once

#include "k1_kda_common.hpp"
#include "packed_direct_prefixless.hpp"

namespace flashkda_hip::gfx950 {

namespace k1_bt16_fused_detail {

constexpr int C = 16;
constexpr int D = 128;
constexpr int TILE_ELEMS = C * D;
constexpr int FACTOR_ELEMS = C * C;

#if defined(__gfx950__)

using u32x2 = unsigned int __attribute__((ext_vector_type(2)));

// Reduce the q/k normalization sums together without routing through LDS.
// The first permlane32 swap leaves q in lanes 0..31 and k in lanes 32..63.
// A permlane16 swap then folds the two rows in each half; only rows 0 and 2
// remain live, so its second destructive operand can be a dead intermediate.
// The remaining XOR-8/4/2/1 tree is exactly the original wave reduction.  At
// XOR-4, row_ror:4 selects the same value as XOR-4 because XOR-8 has already
// made the two 8-lane halves identical.  The final permlane32 swap returns the
// q and k totals to lane 0 in its two destination registers.
__device__ __forceinline__ f32x2 wave_reduce_sum_pair(float q, float k) {
    const u32x2 half = __builtin_amdgcn_permlane32_swap(
        __builtin_bit_cast(unsigned int, q),
        __builtin_bit_cast(unsigned int, k), false, false);
    const f32x2 half_f = __builtin_bit_cast(f32x2, half);
    float v = half_f[0] + half_f[1];

    const u32x2 row = __builtin_amdgcn_permlane16_swap(
        __builtin_bit_cast(unsigned int, v), half[0], false, false);
    const f32x2 row_f = __builtin_bit_cast(f32x2, row);
    v = row_f[0] + row_f[1];

    int remote = __builtin_amdgcn_mov_dpp(
        __builtin_bit_cast(int, v), 0x128, 0xf, 0xf, false);
    v += __builtin_bit_cast(float, remote);
    remote = __builtin_amdgcn_mov_dpp(
        __builtin_bit_cast(int, v), 0x124, 0xf, 0xf, false);
    v += __builtin_bit_cast(float, remote);
    remote = __builtin_amdgcn_mov_dpp(
        __builtin_bit_cast(int, v), 0x4e, 0xf, 0xf, false);
    v += __builtin_bit_cast(float, remote);
    remote = __builtin_amdgcn_mov_dpp(
        __builtin_bit_cast(int, v), 0xb1, 0xf, 0xf, false);
    v += __builtin_bit_cast(float, remote);

    const u32x2 total = __builtin_amdgcn_permlane32_swap(
        __builtin_bit_cast(unsigned int, v),
        static_cast<unsigned int>(remote), false, false);
    return __builtin_bit_cast(f32x2, total);
}

#endif  // defined(__gfx950__)

struct alignas(16) VectorPrepStorage {
    float gc[TILE_ELEMS];
    float decay[D];
};

struct alignas(16) ExactPrepStorage {
    float gc[TILE_ELEMS];
    float decay[D];
    __bf16 q[TILE_ELEMS];
    __bf16 k[TILE_ELEMS];
    float qinv[C];
    float kinv[C];
};

template <int SD>
struct alignas(16) SolveStorage {
    static_assert(SD >= D, "solve operand pitch must cover logical D");
    __bf16 kd[C * SD];
    __bf16 qd[C * SD];
    __bf16 ki[C * SD];
    __bf16 mqk[FACTOR_ELEMS];
    float beta[C];
    _Float16 lm[FACTOR_ELEMS];
    _Float16 inv[FACTOR_ELEMS];
    _Float16 lk[FACTOR_ELEMS];
};

template <bool EXACT_PREP, int SD = D>
union alignas(16) SharedStorage;

template <int SD>
union alignas(16) SharedStorage<false, SD> {
    VectorPrepStorage prep;
    SolveStorage<SD> solve;
};

template <int SD>
union alignas(16) SharedStorage<true, SD> {
    ExactPrepStorage prep;
    SolveStorage<SD> solve;
};

static_assert(sizeof(VectorPrepStorage) == 8704,
              "unexpected vector-prep BT16 scratch size");
static_assert(sizeof(ExactPrepStorage) == 17024,
              "unexpected exact-prep BT16 scratch size");
static_assert(sizeof(SolveStorage<D>) == 14400,
              "unexpected fused BT16 solve scratch size");
static_assert(sizeof(SolveStorage<D + 4>) == 14784,
              "unexpected padded fused BT16 solve scratch size");
static_assert(sizeof(SharedStorage<false, D>) == sizeof(SolveStorage<D>),
              "vector-prep phase union must reuse solve LDS");
static_assert(sizeof(SharedStorage<false, D + 4>) ==
                  sizeof(SolveStorage<D + 4>),
              "padded vector-prep phase union must reuse solve LDS");
static_assert(sizeof(SharedStorage<true, D>) == sizeof(ExactPrepStorage),
              "exact-prep phase union must reuse solve LDS");
static_assert(sizeof(SharedStorage<true, D + 4>) == sizeof(ExactPrepStorage),
              "padded solve must not grow exact-prep LDS");

}  // namespace k1_bt16_fused_detail

// The optimization axes are deliberately default-off so existing launch sites
// retain their byte-for-byte workspace contract and established instruction
// schedule.  The plain C-split PRE_SOLVED route may opt into the first three:
//
//   * OMIT_TRANSIENT_OUTPUTS drops tmp_kinv, which the merge-only consumer never
//     reads, and the temporary decay publication in ws_mqk, which this kernel
//     overwrites with the final local Mqk factor before it completes.
//   * SPLIT_CONTRACTIONS assigns Kd@Ki to wave 0 and Qd@Ki to wave 1.  The two
//     contractions are independent; wave 1 publishes Mqk directly while wave 0
//     continues down the triangular-inverse dependency chain.
//   * REUSE_POWER_CHAIN retains the rounded L^2 and L^4 tiles across the
//     corresponding inv@power products.  Those products never modify lk, so
//     this removes three exactly redundant MFMA/store/sync sequences without
//     changing any arithmetic or rounding point.
//
// Callers must set OMIT_TRANSIENT_OUTPUTS only when their next K1 stage is
// explicitly PRE_SOLVED.  Keeping that fact a template contract prevents an
// environment-derived mismatch from exposing uninitialized tmp_kinv data.
// CACHE_CHUNK_DECAY changes ws_gt from its ordinary log2 ABI to complete-chunk
// FP32 decay and is therefore context-only.  PUBLISH_ACTIVATED_BETA is a
// separate contract: both context and plain C-split may publish the already
// computed FP32 sigmoid result without changing ws_gt.  Its default follows
// CACHE_CHUNK_DECAY so existing context launch sites retain both publications.
// DENSE_N1_ALL_FULL_C16 is a shape proof supplied by the gfx950 launcher: the
// dense grid contains one sequence, every tile is a complete C16 tile, and
// gridDim.y is exactly H.  It removes the batch/head divmod and every tail
// predicate without changing any arithmetic or workspace publication.
// PADDED_SOLVE changes only the LDS row pitch of kd/qd/ki from D to D+4.  The
// global workspace remains compact D=128, and the logical contraction length
// and MFMA accumulation order remain unchanged.
// EARLY_DENSE_BETA assigns the dense vector-prep beta sigmoid to wave 3 while
// waves 0 and 1 perform the prefix scan.  The value remains register-resident
// until the prep/solve phase union has switched, so both LDS and global beta
// publication stay at their established location and ordering point.
template <bool VL, bool EXACT_PREP, bool USE_X32,
          bool OMIT_TRANSIENT_OUTPUTS = false,
          bool SPLIT_CONTRACTIONS = false,
          bool REUSE_POWER_CHAIN = false,
          bool CACHE_CHUNK_DECAY = false,
          bool PUBLISH_ACTIVATED_BETA = CACHE_CHUNK_DECAY,
          bool PACKED_DIRECT_PREFIXLESS = false,
          bool DENSE_N1_ALL_FULL_C16 = false,
          bool PADDED_SOLVE = false,
          bool EARLY_DENSE_BETA = false,
          bool GVA = false>
__global__ void __launch_bounds__(256)
k1_kda_bt16_fused_kernel(
        const __bf16* __restrict__ q_g,
        const __bf16* __restrict__ k_g,
        const __bf16* __restrict__ g_g,
        const float* __restrict__ beta_g,
        const float* __restrict__ A_log,
        const float* __restrict__ dt_bias,
        float scale, float gate_scale, int T_seq, int H, int H_q,
        __bf16* __restrict__ ws_kd,
        __bf16* __restrict__ ws_qd,
        __bf16* __restrict__ ws_kr,
        float* __restrict__ ws_gt,
        __bf16* __restrict__ tmp_kinv,
        __bf16* __restrict__ ws_inv,
        __bf16* __restrict__ ws_mqk,
        float* __restrict__ beta_cache,
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ tile_prefix,
        int N, int total_tiles) {
    using namespace k1_bt16_fused_detail;
    static_assert(!PACKED_DIRECT_PREFIXLESS || VL,
                  "prefixless K1 mapping is packed-only");
    static_assert(!DENSE_N1_ALL_FULL_C16 || !VL,
                  "dense-N1 full-C16 K1 mapping is dense-only");
    static_assert(!DENSE_N1_ALL_FULL_C16 || !PACKED_DIRECT_PREFIXLESS,
                  "dense-N1 full-C16 and packed-prefixless are exclusive");
    static_assert(!PADDED_SOLVE || DENSE_N1_ALL_FULL_C16,
                  "padded solve requires the dense-N1 full-C16 proof");
    static_assert(!PADDED_SOLVE || USE_X32,
                  "padded solve requires the gfx950 X32 contraction");
    static_assert(!EARLY_DENSE_BETA ||
                      (!EXACT_PREP && DENSE_N1_ALL_FULL_C16),
                  "early dense beta requires vector-prep full-C16 proof");
    constexpr int SD = PADDED_SOLVE ? D + 4 : D;
    const int tid = threadIdx.x;
    const int row_lane = tid & 15;
    int h, ht, t0, alen;
    if constexpr (VL) {
        const int gti = blockIdx.x;
        h = blockIdx.y;
        if constexpr (PACKED_DIRECT_PREFIXLESS) {
            const PackedC16TileMapping mapping =
                packed_c16_tile_mapping(cu_seqlens, N, gti);
            if (!mapping.valid)
                return;
            ht = h * total_tiles + gti;
            t0 = mapping.token_base + mapping.local_tile * C;
            alen = min(C, mapping.token_length - mapping.local_tile * C);
        } else {
            int lo = 0, hi = N;
            while (hi - lo > 1) {
                const int mid = (lo + hi) >> 1;
                if (tile_prefix[mid] <= gti) lo = mid; else hi = mid;
            }
            const int local = gti - tile_prefix[lo];
            const int64_t bos = cu_seqlens[lo];
            const int len = int(cu_seqlens[lo + 1] - bos);
            if (local >= (len + C - 1) / C) return;
            ht = h * total_tiles + gti;
            t0 = int(bos) + local * C;
            alen = min(C, len - local * C);
        }
    } else {
        const int nt = blockIdx.x, bh = blockIdx.y;
        if constexpr (DENSE_N1_ALL_FULL_C16) {
            h = bh;
            ht = h * gridDim.x + nt;
            t0 = nt * C;
            alen = C;
        } else {
            const int b = bh / H;
            h = bh % H;
            ht = bh * gridDim.x + nt;
            t0 = b * T_seq + nt * C;
            alen = min(C, T_seq - nt * C);
        }
    }

    __shared__ SharedStorage<EXACT_PREP, SD> smem;
    float* const gc = smem.prep.gc;

    const int vec_m = tid >> 4;
    const int vec_d0 = row_lane * 8;
    const int vec_idx = vec_m * D + vec_d0;
    const float a = ex2(A_log[h] * KDA_LOG2E);
    bf16x8 qv{}, kv{}, gv{};
    f32x4 gcv0{}, gcv1{};
    if (DENSE_N1_ALL_FULL_C16 || vec_m < alen) {
        const int64_t vg_off =
            (int64_t(t0 + vec_m) * H + h) * D + vec_d0;
        int64_t qk_off = vg_off;
        if constexpr (GVA) {
            const int hq = h / (H / H_q);
            qk_off =
                (int64_t(t0 + vec_m) * H_q + hq) * D + vec_d0;
        }
        qv = *reinterpret_cast<const bf16x8*>(q_g + qk_off);
        kv = *reinterpret_cast<const bf16x8*>(k_g + qk_off);
        gv = *reinterpret_cast<const bf16x8*>(g_g + vg_off);
        const f32x4 db0 =
            *reinterpret_cast<const f32x4*>(dt_bias + h * D + vec_d0);
        const f32x4 db1 =
            *reinterpret_cast<const f32x4*>(dt_bias + h * D + vec_d0 + 4);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            gcv0[i] = gate_scale * sigmoid_tanh(
                a * (bf16_to_f32(gv[i]) + db0[i]));
            gcv1[i] = gate_scale * sigmoid_tanh(
                a * (bf16_to_f32(gv[i + 4]) + db1[i]));
        }
    }
    *reinterpret_cast<f32x4*>(gc + vec_idx) = gcv0;
    *reinterpret_cast<f32x4*>(gc + vec_idx + 4) = gcv1;
    if constexpr (EXACT_PREP) {
        *reinterpret_cast<bf16x8*>(smem.prep.q + vec_idx) = qv;
        *reinterpret_cast<bf16x8*>(smem.prep.k + vec_idx) = kv;
    }

    float qinv_row = 0.0f, kinv_row = 0.0f;
    if constexpr (!EXACT_PREP) {
        float qs = 0.0f, ks = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float qf = bf16_to_f32(qv[i]);
            const float kf = bf16_to_f32(kv[i]);
            qs += qf * qf;
            ks += kf * kf;
        }
        #pragma unroll
        for (int o = 8; o >= 1; o >>= 1) {
            qs += __shfl_xor(qs, o, 16);
            ks += __shfl_xor(ks, o, 16);
        }
        if (row_lane == 0) {
            qinv_row = rsqrtf(qs + 1e-6f);
            kinv_row = rsqrtf(ks + 1e-6f);
        }
        qinv_row = __shfl(qinv_row, 0, 16);
        kinv_row = __shfl(kinv_row, 0, 16);
    }

    __syncthreads();
    if constexpr (EXACT_PREP) {
        const int wave = tid >> 6;
        const int lane = tid & 63;
        #pragma unroll
        for (int m = wave; m < C; m += 4) {
            const float q0 = bf16_to_f32(smem.prep.q[m * D + lane]);
            const float q1 = bf16_to_f32(smem.prep.q[m * D + lane + 64]);
            const float k0 = bf16_to_f32(smem.prep.k[m * D + lane]);
            const float k1 = bf16_to_f32(smem.prep.k[m * D + lane + 64]);
#if defined(__gfx950__)
            const f32x2 norm = wave_reduce_sum_pair(
                q0 * q0 + q1 * q1, k0 * k0 + k1 * k1);
#else
            const f32x2 norm = {
                wave_reduce_sum(q0 * q0 + q1 * q1),
                wave_reduce_sum(k0 * k0 + k1 * k1)};
#endif
            if (lane == 0) {
                smem.prep.qinv[m] = rsqrtf(norm[0] + 1e-6f);
                smem.prep.kinv[m] = rsqrtf(norm[1] + 1e-6f);
            }
        }
    }
    float balanced_beta = 0.0f;
    if constexpr (EXACT_PREP || EARLY_DENSE_BETA) {
        if (tid >= 3 * 64 && tid < 3 * 64 + C) {
            balanced_beta = (DENSE_N1_ALL_FULL_C16 || row_lane < alen)
                ? sigmoid_tanh(beta_g[int64_t(t0 + row_lane) * H + h])
                : 0.0f;
        }
    }
    if (tid < D) {
        float acc = 0.0f;
        #pragma unroll
        for (int m = 0; m < C - 1; ++m) {
            acc += gc[m * D + tid];
            gc[m * D + tid] = acc;
        }
        acc += gc[(C - 1) * D + tid];
        const float decay = ex2(acc);
        gc[(C - 1) * D + tid] = acc;
        smem.prep.decay[tid] = decay;
        // Context B/A/replay all need the same complete-chunk decay.  Publish
        // the value already formed here so those six V-half CTAs do not each
        // repeat an exp2.  Other routes retain the established log2 ABI.
        ws_gt[int64_t(ht) * D + tid] =
            CACHE_CHUNK_DECAY ? decay : acc;
        if constexpr (!OMIT_TRANSIENT_OUTPUTS)
            reinterpret_cast<float*>(ws_mqk)[int64_t(ht) * D + tid] = decay;
    }
    __syncthreads();

    if constexpr (EXACT_PREP) {
        qinv_row = smem.prep.qinv[vec_m];
        kinv_row = smem.prep.kinv[vec_m];
    }

    const __bf16 scale_bf = f32_to_bf16(scale);
    const f32x4 gc0 = *reinterpret_cast<const f32x4*>(gc + vec_idx);
    const f32x4 gc1 = *reinterpret_cast<const f32x4*>(gc + vec_idx + 4);
    const f32x4 decay0 =
        *reinterpret_cast<const f32x4*>(smem.prep.decay + vec_d0);
    const f32x4 decay1 =
        *reinterpret_cast<const f32x4*>(smem.prep.decay + vec_d0 + 4);
    bf16x8 kd_v{}, ki_v{}, kr_v{}, qd_v{};
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float gc_i = i < 4 ? gc0[i] : gc1[i - 4];
        const float decay_i = i < 4 ? decay0[i] : decay1[i - 4];
        const float dp_prefix = ex2(gc_i);
        const float dp_f = EXACT_PREP || vec_m != C - 1
            ? dp_prefix : decay_i;
        const float dn_f = EXACT_PREP
            ? ex2(-gc_i) : __builtin_amdgcn_rcpf(dp_f);
        const __bf16 dp = f32_to_bf16(dp_f);
        const __bf16 dn = f32_to_bf16(dn_f);
        const __bf16 dt = f32_to_bf16(decay_i);
        const float kn_f = bf16_to_f32(kv[i]) * kinv_row;
        const float qn_f = bf16_to_f32(qv[i]) * qinv_row;
        const float kn = EXACT_PREP
            ? bf16_to_f32(f32_to_bf16(kn_f)) : kn_f;
        const float qn = EXACT_PREP
            ? bf16_to_f32(f32_to_bf16(qn_f)) : qn_f;
        kd_v[i] = f32_to_bf16(kn * bf16_to_f32(dp));
        ki_v[i] = f32_to_bf16(kn * bf16_to_f32(dn));
        const __bf16 qt = f32_to_bf16(qn * bf16_to_f32(dp));
        kr_v[i] =
            f32_to_bf16(bf16_to_f32(ki_v[i]) * bf16_to_f32(dt));
        qd_v[i] =
            f32_to_bf16(bf16_to_f32(qt) * bf16_to_f32(scale_bf));
    }

    // All 256 threads must finish reading gc before the phase union becomes
    // the solve operand storage.  Each thread then publishes the same rounded
    // BF16 values to both the global ABI and LDS, avoiding the second kernel's
    // HBM-to-LDS replay.
    __syncthreads();
    SolveStorage<SD>& solve = smem.solve;
    const int64_t ws_vec_off = int64_t(ht) * TILE_ELEMS + vec_idx;
    *reinterpret_cast<bf16x8*>(ws_kd + ws_vec_off) = kd_v;
    *reinterpret_cast<bf16x8*>(ws_qd + ws_vec_off) = qd_v;
    *reinterpret_cast<bf16x8*>(ws_kr + ws_vec_off) = kr_v;
    if constexpr (!OMIT_TRANSIENT_OUTPUTS)
        *reinterpret_cast<bf16x8*>(tmp_kinv + ws_vec_off) = ki_v;
    const int solve_vec_idx = vec_m * SD + vec_d0;
    *reinterpret_cast<bf16x8*>(solve.kd + solve_vec_idx) = kd_v;
    *reinterpret_cast<bf16x8*>(solve.qd + solve_vec_idx) = qd_v;
    *reinterpret_cast<bf16x8*>(solve.ki + solve_vec_idx) = ki_v;
    if constexpr (EXACT_PREP || EARLY_DENSE_BETA) {
        if (tid >= 3 * 64 && tid < 3 * 64 + C) {
            solve.beta[row_lane] = balanced_beta;
            if constexpr (PUBLISH_ACTIVATED_BETA)
                beta_cache[int64_t(ht) * C + row_lane] = balanced_beta;
        }
    } else if (tid < C) {
        const float activated_beta = (DENSE_N1_ALL_FULL_C16 || tid < alen)
            ? sigmoid_tanh(beta_g[int64_t(t0 + tid) * H + h]) : 0.0f;
        solve.beta[tid] = activated_beta;
        if constexpr (PUBLISH_ACTIVATED_BETA)
            beta_cache[int64_t(ht) * C + tid] = activated_beta;
    }
    __syncthreads();

    // The legacy specialization keeps both contractions on wave 0.  The
    // opt-in split specialization lets wave 1 form and publish local Mqk while
    // wave 0 forms L and immediately enters the dependent inverse chain.  The
    // block-wide publication above makes kd/qd/ki visible to both waves; no
    // cross-wave result is consumed, so all remaining fences stay wave-local.
    constexpr int ACTIVE_SOLVE_THREADS = SPLIT_CONTRACTIONS ? 128 : 64;
    if (tid >= ACTIVE_SOLVE_THREADS) return;
    const int lane = tid & 63;
    if constexpr (SPLIT_CONTRACTIONS) {
        if (tid >= 64) {
            f32x4 cm;
            if constexpr (USE_X32)
                cm = contract_last_x32<D, SD, SD>(solve.qd, solve.ki, lane);
            else
                cm = gemm_contract_last<__bf16, D, SD>(
                    solve.qd, solve.ki, lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int m = (lane >> 4) * 4 + i, n = lane & 15;
                ws_mqk[int64_t(ht) * FACTOR_ELEMS + m * C + n] = m >= n
                    ? f32_to_bf16(cm[i]) : (__bf16)0.0f;
            }
            return;
        }
    }

    f32x4 cl;
    if constexpr (USE_X32)
        cl = contract_last_x32<D, SD, SD>(solve.kd, solve.ki, lane);
    else
        cl = gemm_contract_last<__bf16, D, SD>(solve.kd, solve.ki, lane);
    f32x4 cm{};
    if constexpr (!SPLIT_CONTRACTIONS) {
        if constexpr (USE_X32)
            cm = contract_last_x32<D, SD, SD>(solve.qd, solve.ki, lane);
        else
            cm = gemm_contract_last<__bf16, D, SD>(
                solve.qd, solve.ki, lane);
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int m = (lane >> 4) * 4 + i, n = lane & 15;
        solve.lm[m * C + n] = m > n
            ? f32_to_f16(cl[i]) * f32_to_f16(solve.beta[m])
            : (_Float16)0.0f;
        if constexpr (!SPLIT_CONTRACTIONS)
            solve.mqk[m * C + n] = m >= n
                ? f32_to_bf16(cm[i]) : (__bf16)0.0f;
        solve.inv[m * C + n] =
            (_Float16)(m == n ? 1.0f : 0.0f) - solve.lm[m * C + n];
    }
    __syncwarp();

    if constexpr (REUSE_POWER_CHAIN) {
        // lk progresses L^2 -> L^4 -> L^8.  inv@lk is read-only in lk, so the
        // rounded power remains valid for the next square.
        { f32x4 c = gemm_std_f16_tr(solve.lm, solve.lm, lane);
          store_acc_16x16(solve.lk, c, lane); }
        __syncwarp();
        { f32x4 c = gemm_std_f16_tr(solve.inv, solve.lk, lane); __syncwarp();
          #pragma unroll
          for (int i = 0; i < 4; ++i) {
              const int m = (lane >> 4) * 4 + i, n = lane & 15;
              solve.inv[m * C + n] += f32_to_f16(c[i]);
          } }
        __syncwarp();
        { f32x4 c = gemm_std_f16_tr(solve.lk, solve.lk, lane); __syncwarp();
          store_acc_16x16(solve.lk, c, lane); }
        __syncwarp();
        { f32x4 c = gemm_std_f16_tr(solve.inv, solve.lk, lane); __syncwarp();
          #pragma unroll
          for (int i = 0; i < 4; ++i) {
              const int m = (lane >> 4) * 4 + i, n = lane & 15;
              solve.inv[m * C + n] += f32_to_f16(c[i]);
          } }
        __syncwarp();
        { f32x4 c = gemm_std_f16_tr(solve.lk, solve.lk, lane); __syncwarp();
          store_acc_16x16(solve.lk, c, lane); }
        __syncwarp();
        { f32x4 c = gemm_std_f16_tr(solve.inv, solve.lk, lane); __syncwarp();
          #pragma unroll
          for (int i = 0; i < 4; ++i) {
              const int m = (lane >> 4) * 4 + i, n = lane & 15;
              solve.inv[m * C + n] += f32_to_f16(c[i]);
          } }
        __syncwarp();
    } else {
        { f32x4 c = gemm_std_f16_tr(solve.lm, solve.lm, lane);
          store_acc_16x16(solve.lk, c, lane); }
        __syncwarp();
        { f32x4 c = gemm_std_f16_tr(solve.inv, solve.lk, lane); __syncwarp();
          #pragma unroll
          for (int i = 0; i < 4; ++i) {
              const int m = (lane >> 4) * 4 + i, n = lane & 15;
              solve.inv[m * C + n] += f32_to_f16(c[i]);
          } }
        __syncwarp();
        { f32x4 c = gemm_std_f16_tr(solve.lm, solve.lm, lane);
          store_acc_16x16(solve.lk, c, lane); }
        __syncwarp();
        { f32x4 c = gemm_std_f16_tr(solve.lk, solve.lk, lane); __syncwarp();
          store_acc_16x16(solve.lk, c, lane); }
        __syncwarp();
        { f32x4 c = gemm_std_f16_tr(solve.inv, solve.lk, lane); __syncwarp();
          #pragma unroll
          for (int i = 0; i < 4; ++i) {
              const int m = (lane >> 4) * 4 + i, n = lane & 15;
              solve.inv[m * C + n] += f32_to_f16(c[i]);
          } }
        __syncwarp();
        { f32x4 c = gemm_std_f16_tr(solve.lm, solve.lm, lane);
          store_acc_16x16(solve.lk, c, lane); }
        __syncwarp();
        { f32x4 c = gemm_std_f16_tr(solve.lk, solve.lk, lane); __syncwarp();
          store_acc_16x16(solve.lk, c, lane); }
        __syncwarp();
        { f32x4 c = gemm_std_f16_tr(solve.lk, solve.lk, lane); __syncwarp();
          store_acc_16x16(solve.lk, c, lane); }
        __syncwarp();
        { f32x4 c = gemm_std_f16_tr(solve.inv, solve.lk, lane); __syncwarp();
          #pragma unroll
          for (int i = 0; i < 4; ++i) {
              const int m = (lane >> 4) * 4 + i, n = lane & 15;
              solve.inv[m * C + n] += f32_to_f16(c[i]);
          } }
        __syncwarp();
    }

    for (int idx = lane; idx < FACTOR_ELEMS; idx += 64) {
        ws_inv[int64_t(ht) * FACTOR_ELEMS + idx] =
            f32_to_bf16(f16_to_f32(solve.inv[idx]));
        if constexpr (!SPLIT_CONTRACTIONS)
            ws_mqk[int64_t(ht) * FACTOR_ELEMS + idx] = solve.mqk[idx];
    }
}

}  // namespace flashkda_hip::gfx950
