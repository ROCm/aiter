// gfx942 experimental raw-input BT64 K1 fusion.
//
// This kernel replaces
//
//   k1_kda_split_prep_kernel -> k1_kda_bt64_neumann_c_kernel
//
// for the direct-RTP K6 route.  A CTA owns one BT64 segment and prepares its
// four BT16 chunks in reverse order.  Reverse traversal is the key: when the
// inverse-decayed key for column chunk c is ready, every Kd row chunk r >= c is
// already resident in LDS, so the complete lower KKT tile set can be formed
// while retaining only one Ki tile.  The temporary Ki workspace is therefore
// removed, Kd is written once in its final segment-relative form, and the
// global launch dependency between preparation and factorization disappears.
//
// Persistent output ABI (identical to the current direct-RTP producer):
//   kd/qd/kr/gt/decay, packed C in inv/cross32/cross64, activated beta cache.
// tmp_kinv and ws_mqk are intentionally absent.
//
// The arithmetic and explicit rounding points mirror the split kernels:
//   * BT16 q/k normalization and gate scans use the split-prep mapping;
//   * local Kd/Ki/Qd/Kr are rounded to BF16 before any KKT contraction;
//   * diagonal L and Neumann powers use the established FP16 sequence;
//   * the BT64 dependency DAG uses the same FP16 fragment conversions;
//   * segment-relative Kd is rounded only after multiplying the BF16 local Kd.
#pragma once

#include "../k1_kda_bt64_neumann_c_kernel.hpp"
#include "../k1_kda_split_kernel.hpp"

namespace flashkda_hip::gfx942 {

namespace k1_bt64_fused_prepare_neumann_detail {

constexpr int C = 16;
constexpr int BT = 64;
constexpr int D = 128;
constexpr int NTHREADS = 256;
constexpr int K_STRIDE = D + 4;
constexpr int A_STRIDE = BT + 1;
constexpr int K_VECS_PER_ROW = D / 8;
constexpr int KKT_TILES_PER_WAVE = 3;
constexpr int FACTOR_ELEMS = C * C;

// Kd for all BT64 rows survives all four reverse-preparation phases.  Ki is
// single-buffered because its column is consumed before the next chunk is
// prepared.  gc cannot alias Ki: preparation threads must retain their Q/K
// vectors across the two gc publication barriers, and a compressed BF16 Ki
// write would otherwise clobber another thread's still-live FP32 prefix.
struct alignas(16) PrepKktStorage {
    float gc[C * D];
    __bf16 kd[BT * K_STRIDE];
    __bf16 ki[C * K_STRIDE];
    float decay[4 * D];
    float beta[BT];
};

struct alignas(16) SolveStorage {
    float A[BT * A_STRIDE];
    _Float16 diagonal_scratch[8 * FACTOR_ELEMS];
};

union alignas(16) SharedStorage {
    PrepKktStorage prep;
    SolveStorage solve;
};

constexpr int PREP_KKT_BYTES = sizeof(PrepKktStorage);
constexpr int SOLVE_BYTES = sizeof(SolveStorage);
static_assert(PREP_KKT_BYTES == 31616,
              "unexpected fused gfx942 K1 preparation footprint");
static_assert(SOLVE_BYTES == 20736,
              "unexpected fused gfx942 K1 solve footprint");
static_assert(SOLVE_BYTES <= PREP_KKT_BYTES,
              "solve phase must alias the preparation allocation");
static_assert(2 * PREP_KKT_BYTES <= 64 * 1024,
              "fused gfx942 K1 must retain two-CTA LDS residency");

__device__ __forceinline__ float decay_prefix(
        const float* __restrict__ decay, int chunk, int d) {
    float value = 1.0f;
#pragma unroll
    for (int u = 0; u < 3; ++u) {
        if (u < chunk)
            value *= decay[u * D + d];
    }
    return value;
}

__device__ __forceinline__ f32x4 bounded_key_factor4(
        const float* __restrict__ decay,
        int col_chunk, int row_chunk, int d0) {
    f32x4 value = {1.0f, 1.0f, 1.0f, 1.0f};
#pragma unroll
    for (int u = 0; u < 3; ++u) {
        if (u >= col_chunk && u < row_chunk) {
            const f32x4 x = *reinterpret_cast<const f32x4*>(
                decay + u * D + d0);
#pragma unroll
            for (int p = 0; p < 4; ++p)
                value[p] *= x[p];
        }
    }
    return value;
}

}  // namespace k1_bt64_fused_prepare_neumann_detail

// Static LDS footprint of the experimental kernel.  The launch does not need
// dynamic shared memory.
constexpr int kK1Bt64FusedPrepareNeumannSmemBytes =
    k1_bt64_fused_prepare_neumann_detail::PREP_KKT_BYTES;

template <bool VL, bool USE_DPP = true>
__global__ void __launch_bounds__(256)
k1_kda_bt64_fused_prepare_neumann_kernel(
        const __bf16* __restrict__ q_g,
        const __bf16* __restrict__ k_g,
        const __bf16* __restrict__ g_g,
        const float* __restrict__ beta_g,
        const float* __restrict__ A_log,
        const float* __restrict__ dt_bias,
        float scale, float gate_scale,
        __bf16* __restrict__ ws_kd,
        __bf16* __restrict__ ws_qd,
        __bf16* __restrict__ ws_kr,
        float* __restrict__ ws_gt,
        float* __restrict__ ws_decay,
        __bf16* __restrict__ ws_inv,
        __bf16* __restrict__ cross32,
        __bf16* __restrict__ cross64,
        float* __restrict__ beta_cache,
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ tile_prefix,
        const int* __restrict__ pair_prefix,
        const int* __restrict__ segment_prefix,
        int N, int total_tiles, int total_pairs, int total_segments,
        int T_seq, int H, int NT) {
    using namespace k1_bt64_fused_prepare_neumann_detail;
    namespace split = flashkda_hip::k1_split_prep_detail;
    namespace neumann = flashkda_hip::k1_bt64_neumann_c_detail;

    const int tid = static_cast<int>(threadIdx.x);
    const int wave = tid >> 6;
    const int lane = tid & 63;

    int h;
    int ht0;
    int xp0;
    int xs;
    int t0;
    int alen;

    // All early returns are CTA-uniform; every later path reaches the same CTA
    // barriers, including missing tail chunks in the fixed four-iteration loop.
    if constexpr (VL) {
        if (N <= 0)
            return;
        const int gsi = static_cast<int>(blockIdx.x);
        h = static_cast<int>(blockIdx.y);
        int lo = 0;
        int hi = N;
        while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (segment_prefix[mid] <= gsi)
                lo = mid;
            else
                hi = mid;
        }
        const int local_seg = gsi - segment_prefix[lo];
        const int64_t bos = cu_seqlens[lo];
        const int len = static_cast<int>(cu_seqlens[lo + 1] - bos);
        const int nseg = (len + BT - 1) / BT;
        if (local_seg < 0 || local_seg >= nseg)
            return;
        alen = min(BT, len - local_seg * BT);
        if (alen <= 0)
            return;
        t0 = static_cast<int>(bos) + local_seg * BT;
        ht0 = h * total_tiles + tile_prefix[lo] + local_seg * 4;
        xp0 = h * total_pairs + pair_prefix[lo] + local_seg * 2;
        xs = h * total_segments + gsi;
    } else {
        if (H <= 0 || NT <= 0 || T_seq <= 0)
            return;
        const int seg = static_cast<int>(blockIdx.x);
        const int bh = static_cast<int>(blockIdx.y);
        const int token0 = seg * BT;
        if (token0 >= T_seq)
            return;
        const int b = bh / H;
        h = bh % H;
        alen = min(BT, T_seq - token0);
        t0 = b * T_seq + token0;
        ht0 = bh * NT + seg * 4;
        xp0 = bh * ((NT + 1) / 2) + seg * 2;
        xs = bh * ((NT + 3) / 4) + seg;
    }

    const int nch = (alen + C - 1) / C;
    __shared__ SharedStorage smem;
    PrepKktStorage& prep = smem.prep;

    // Cache the exact activation consumed by the factor and recurrent scan.
    if (tid < BT) {
        const float b = tid < alen
            ? sigmoid_tanh(beta_g[int64_t(t0 + tid) * H + h])
            : 0.0f;
        prep.beta[tid] = b;
        beta_cache[int64_t(xs) * BT + tid] = b;
    }

    const float a = ex2(A_log[h] * KDA_LOG2E);
    const __bf16 scale_bf = f32_to_bf16(scale);
    const int row_lane = tid & 15;
    const int vec_m = tid >> 4;
    const int vec_d0 = row_lane * 8;
    const int vec_idx = vec_m * D + vec_d0;

    const f32x4 zero4 = {0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 kkt[KKT_TILES_PER_WAVE];
#pragma unroll
    for (int slot = 0; slot < KKT_TILES_PER_WAVE; ++slot)
        kkt[slot] = zero4;

    // Reverse chunk order makes all Kd row blocks needed by this Ki column
    // available without retaining four Ki tiles or replaying global workspace.
#pragma unroll
    for (int rev = 0; rev < 4; ++rev) {
        const int chunk = 3 - rev;
        const int chunk_t0 = t0 + chunk * C;
        const int chunk_rows = min(C, max(0, alen - chunk * C));
        const bool row_valid = vec_m < chunk_rows;

        bf16x8 qv{};
        bf16x8 kv{};
        bf16x8 gv{};
        f32x4 gcv0{};
        f32x4 gcv1{};
        if (row_valid) {
            const int64_t off =
                (int64_t(chunk_t0 + vec_m) * H + h) * D + vec_d0;
            qv = *reinterpret_cast<const bf16x8*>(q_g + off);
            kv = *reinterpret_cast<const bf16x8*>(k_g + off);
            gv = *reinterpret_cast<const bf16x8*>(g_g + off);
            const f32x4 db0 = *reinterpret_cast<const f32x4*>(
                dt_bias + h * D + vec_d0);
            const f32x4 db1 = *reinterpret_cast<const f32x4*>(
                dt_bias + h * D + vec_d0 + 4);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                gcv0[i] = gate_scale * sigmoid_tanh(
                    a * (bf16_to_f32(gv[i]) + db0[i]));
                gcv1[i] = gate_scale * sigmoid_tanh(
                    a * (bf16_to_f32(gv[i + 4]) + db1[i]));
            }
        }
        *reinterpret_cast<f32x4*>(prep.gc + vec_idx) = gcv0;
        *reinterpret_cast<f32x4*>(prep.gc + vec_idx + 4) = gcv1;

        float qs = 0.0f;
        float ks = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float qf = bf16_to_f32(qv[i]);
            const float kf = bf16_to_f32(kv[i]);
            qs += qf * qf;
            ks += kf * kf;
        }
        qs = split::reduce_sum_16<USE_DPP>(qs);
        ks = split::reduce_sum_16<USE_DPP>(ks);
        float qinv_row = 0.0f;
        float kinv_row = 0.0f;
        if (row_lane == 0) {
            qinv_row = rsqrtf(qs + 1e-6f);
            kinv_row = rsqrtf(ks + 1e-6f);
        }
        qinv_row = split::broadcast_row_lane0<USE_DPP>(qinv_row);
        kinv_row = split::broadcast_row_lane0<USE_DPP>(kinv_row);

        __syncthreads();

        if (tid < D) {
            float acc = 0.0f;
#pragma unroll
            for (int m = 0; m < C - 1; ++m) {
                acc += prep.gc[m * D + tid];
                prep.gc[m * D + tid] = acc;
            }
            acc += prep.gc[(C - 1) * D + tid];
            const float decay = ex2(acc);
            prep.gc[(C - 1) * D + tid] = decay;
            prep.decay[chunk * D + tid] = decay;
            if (chunk < nch) {
                ws_gt[int64_t(ht0 + chunk) * D + tid] = acc;
                ws_decay[int64_t(ht0 + chunk) * D + tid] = decay;
            }
        }
        __syncthreads();

        const f32x4 gc0 =
            *reinterpret_cast<const f32x4*>(prep.gc + vec_idx);
        const f32x4 gc1 =
            *reinterpret_cast<const f32x4*>(prep.gc + vec_idx + 4);
        const f32x4 decay0 = *reinterpret_cast<const f32x4*>(
            prep.decay + chunk * D + vec_d0);
        const f32x4 decay1 = *reinterpret_cast<const f32x4*>(
            prep.decay + chunk * D + vec_d0 + 4);

        bf16x8 kd_v{};
        bf16x8 ki_v{};
        bf16x8 kr_v{};
        bf16x8 qd_v{};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float gc_i = i < 4 ? gc0[i] : gc1[i - 4];
            const float decay_i = i < 4 ? decay0[i] : decay1[i - 4];
            const float dp_prefix = ex2(gc_i);
            const float dp_f = vec_m == C - 1 ? decay_i : dp_prefix;
            const float dn_f = __builtin_amdgcn_rcpf(dp_f);
            const __bf16 dp = f32_to_bf16(dp_f);
            const __bf16 dn = f32_to_bf16(dn_f);
            const __bf16 dt = f32_to_bf16(decay_i);
            const float kn = bf16_to_f32(kv[i]) * kinv_row;
            const float qn = bf16_to_f32(qv[i]) * qinv_row;
            kd_v[i] = f32_to_bf16(kn * bf16_to_f32(dp));
            ki_v[i] = f32_to_bf16(kn * bf16_to_f32(dn));
            const __bf16 qt = f32_to_bf16(qn * bf16_to_f32(dp));
            kr_v[i] = f32_to_bf16(
                bf16_to_f32(ki_v[i]) * bf16_to_f32(dt));
            qd_v[i] = f32_to_bf16(
                bf16_to_f32(qt) * bf16_to_f32(scale_bf));
        }

        *reinterpret_cast<bf16x8*>(
            prep.kd + (chunk * C + vec_m) * K_STRIDE + vec_d0) = kd_v;
        *reinterpret_cast<bf16x8*>(
            prep.ki + vec_m * K_STRIDE + vec_d0) = ki_v;
        if (chunk < nch) {
            const int64_t ws_off =
                (int64_t(ht0 + chunk) * C + vec_m) * D + vec_d0;
            *reinterpret_cast<bf16x8*>(ws_qd + ws_off) = qd_v;
            *reinterpret_cast<bf16x8*>(ws_kr + ws_off) = kr_v;
        }
        __syncthreads();

#pragma unroll
        for (int slot = 0; slot < KKT_TILES_PER_WAVE; ++slot) {
            int tile_row;
            int tile_col;
            const bool active = neumann::tile_for_slot(
                wave, slot, tile_row, tile_col);
            if (active && tile_col == chunk && tile_row < nch) {
#pragma unroll
                for (int ek = 0; ek < D / C; ++ek) {
                    const bf16x4 af = neumann::load_contract_fragment(
                        prep.kd, tile_row * C, ek * C, K_STRIDE, lane);
                    bf16x4 bf = neumann::load_contract_fragment(
                        prep.ki, 0, ek * C, K_STRIDE, lane);
                    const int d0 = ek * C + ((lane >> 4) << 2);
                    const f32x4 factor = bounded_key_factor4(
                        prep.decay, tile_col, tile_row, d0);
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        bf[p] = f32_to_bf16(
                            bf16_to_f32(bf[p]) * factor[p]);
                    }
                    kkt[slot] = mfma_bf16(af, bf, kkt[slot]);
                }
            }
        }
        // The next iteration's first barrier occurs before Ki can be replaced,
        // so no additional end-of-column rendezvous is necessary here.
    }

    // Publish Kd exactly once, after all local KKT consumers are done.  These
    // are read-only LDS accesses, so they may overlap lagging waves' KKT reads;
    // the following barrier is the phase-union fence.
    for (int vi = tid; vi < nch * C * K_VECS_PER_ROW; vi += NTHREADS) {
        const int row = vi / K_VECS_PER_ROW;
        const int col = (vi % K_VECS_PER_ROW) * 8;
        const int chunk = row / C;
        const int chunk_row = row & (C - 1);
        const bf16x8 local = *reinterpret_cast<const bf16x8*>(
            prep.kd + row * K_STRIDE + col);
        bf16x8 segment_value{};
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            const float factor = decay_prefix(prep.decay, chunk, col + p);
            segment_value[p] = f32_to_bf16(
                bf16_to_f32(local[p]) * factor);
        }
        *reinterpret_cast<bf16x8*>(
            ws_kd + (int64_t(ht0 + chunk) * C + chunk_row) * D + col) =
            segment_value;
    }
    __syncthreads();

    float* const s_A = smem.solve.A;

    // Convert the ten KKT accumulators to the same L representation used by
    // the split Neumann kernel.  Diagonal blocks retain its extra FP16 beta and
    // product rounding; cross blocks remain FP32 until their DAG fragments.
#pragma unroll
    for (int slot = 0; slot < KKT_TILES_PER_WAVE; ++slot) {
        int tile_row;
        int tile_col;
        if (!neumann::tile_for_slot(wave, slot, tile_row, tile_col))
            continue;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int row = tile_row * C + ((lane >> 4) << 2) + p;
            const int col = tile_col * C + (lane & 15);
            float l = 0.0f;
            if (row < alen && row > col) {
                if (tile_row == tile_col) {
                    const _Float16 dot_h = f32_to_f16(kkt[slot][p]);
                    const _Float16 beta_h = f32_to_f16(prep.beta[row]);
                    l = f16_to_f32(f32_to_f16(
                        f16_to_f32(dot_h) * f16_to_f32(beta_h)));
                } else {
                    l = kkt[slot][p] * prep.beta[row];
                }
            }
            s_A[row * A_STRIDE + col] = l;
        }
    }
    __syncthreads();

    // Four independent finite BT16 Neumann factorizations.
    {
        const int block_row = wave * C;
        _Float16* const b_tile =
            smem.solve.diagonal_scratch + wave * FACTOR_ELEMS;
        _Float16* const power_tile =
            smem.solve.diagonal_scratch + 4 * FACTOR_ELEMS
            + wave * FACTOR_ELEMS;

        for (int i = lane; i < FACTOR_ELEMS; i += 64) {
            const int r = i / C;
            const int c = i % C;
            const _Float16 b = r > c
                ? f32_to_f16(-s_A[(block_row + r) * A_STRIDE + block_row + c])
                : f32_to_f16(0.0f);
            b_tile[i] = b;
            const _Float16 ci = f32_to_f16(
                (r == c ? 1.0f : 0.0f) + f16_to_f32(b));
            s_A[(block_row + r) * A_STRIDE + block_row + c] =
                f16_to_f32(ci);
        }
        __syncwarp();

        f32x4 power = gemm_std_f16(b_tile, b_tile, lane);
        store_acc_16x16(power_tile, power, lane);
        __syncwarp();

#pragma unroll
        for (int level = 0; level < 3; ++level) {
            const f16x4 c_a = neumann::load_fp32_a_fragment_f16(
                s_A, block_row, block_row, A_STRIDE, lane);
            const int n = lane & 15;
            const int kb = (lane >> 4) << 2;
            const f16x4 p_b = {
                power_tile[(kb + 0) * C + n],
                power_tile[(kb + 1) * C + n],
                power_tile[(kb + 2) * C + n],
                power_tile[(kb + 3) * C + n]};
            const f32x4 term = mfma_f16(c_a, p_b, zero4);

            const int mb = (lane >> 4) << 2;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const int addr =
                    (block_row + mb + p) * A_STRIDE + block_row + n;
                const _Float16 rounded_term = f32_to_f16(term[p]);
                s_A[addr] = f16_to_f32(f32_to_f16(
                    s_A[addr] + f16_to_f32(rounded_term)));
            }
            __syncwarp();

            if (level < 2) {
                power = gemm_std_f16(power_tile, power_tile, lane);
                store_acc_16x16(power_tile, power, lane);
                __syncwarp();
            }
        }
    }
    __syncthreads();

    // Merge the four diagonal inverses with the split kernel's dependency DAG.
    f16x4 saved_l32{};
    f16x4 saved_l43{};
    f16x4 saved_l42{};
    if (wave == 0) {
        saved_l32 = neumann::load_fp32_a_fragment_f16(
            s_A, 32, 16, A_STRIDE, lane);
        saved_l43 = neumann::load_fp32_a_fragment_f16(
            s_A, 48, 32, A_STRIDE, lane);
        saved_l42 = neumann::load_fp32_a_fragment_f16(
            s_A, 48, 16, A_STRIDE, lane);
    } else if (wave == 1) {
        saved_l43 = neumann::load_fp32_a_fragment_f16(
            s_A, 48, 32, A_STRIDE, lane);
    }
    __syncthreads();

    f32x4 kept_c21 = zero4;
    f32x4 kept_c32 = zero4;
    f32x4 kept_c31 = zero4;

    if (wave == 0) {
        f32x4 t = mfma_f16(
            neumann::load_fp32_a_fragment_f16(s_A, 16, 0, A_STRIDE, lane),
            neumann::load_fp32_b_fragment_f16(s_A, 0, 0, A_STRIDE, lane),
            zero4);
        kept_c21 = mfma_f16(
            neumann::load_fp32_a_fragment_f16(s_A, 16, 16, A_STRIDE, lane),
            neumann::accum_to_f16(t), zero4);
#pragma unroll
        for (int p = 0; p < 4; ++p)
            kept_c21[p] = -kept_c21[p];
        neumann::store_fp32_accum(
            s_A, 16, 0, A_STRIDE, kept_c21, lane);
    } else if (wave == 1) {
        f32x4 t = mfma_f16(
            neumann::load_fp32_a_fragment_f16(s_A, 32, 16, A_STRIDE, lane),
            neumann::load_fp32_b_fragment_f16(s_A, 16, 16, A_STRIDE, lane),
            zero4);
        kept_c32 = mfma_f16(
            neumann::load_fp32_a_fragment_f16(s_A, 32, 32, A_STRIDE, lane),
            neumann::accum_to_f16(t), zero4);
#pragma unroll
        for (int p = 0; p < 4; ++p)
            kept_c32[p] = -kept_c32[p];
        neumann::store_fp32_accum(
            s_A, 32, 16, A_STRIDE, kept_c32, lane);
    } else if (wave == 2) {
        f32x4 t = mfma_f16(
            neumann::load_fp32_a_fragment_f16(s_A, 48, 32, A_STRIDE, lane),
            neumann::load_fp32_b_fragment_f16(s_A, 32, 32, A_STRIDE, lane),
            zero4);
        f32x4 c43 = mfma_f16(
            neumann::load_fp32_a_fragment_f16(s_A, 48, 48, A_STRIDE, lane),
            neumann::accum_to_f16(t), zero4);
#pragma unroll
        for (int p = 0; p < 4; ++p)
            c43[p] = -c43[p];
        neumann::store_fp32_accum(s_A, 48, 32, A_STRIDE, c43, lane);
    }

    if (wave == 0) {
        f32x4 t = mfma_f16(
            neumann::load_fp32_a_fragment_f16(s_A, 32, 0, A_STRIDE, lane),
            neumann::load_fp32_b_fragment_f16(s_A, 0, 0, A_STRIDE, lane),
            zero4);
        t = mfma_f16(saved_l32, neumann::accum_to_f16(kept_c21), t);
        kept_c31 = mfma_f16(
            neumann::load_fp32_a_fragment_f16(s_A, 32, 32, A_STRIDE, lane),
            neumann::accum_to_f16(t), zero4);
#pragma unroll
        for (int p = 0; p < 4; ++p)
            kept_c31[p] = -kept_c31[p];
        neumann::store_fp32_accum(
            s_A, 32, 0, A_STRIDE, kept_c31, lane);
    } else if (wave == 1) {
        f32x4 t = mfma_f16(
            neumann::load_fp32_a_fragment_f16(s_A, 48, 16, A_STRIDE, lane),
            neumann::load_fp32_b_fragment_f16(s_A, 16, 16, A_STRIDE, lane),
            zero4);
        t = mfma_f16(saved_l43, neumann::accum_to_f16(kept_c32), t);
        f32x4 c42 = mfma_f16(
            neumann::load_fp32_a_fragment_f16(s_A, 48, 48, A_STRIDE, lane),
            neumann::accum_to_f16(t), zero4);
#pragma unroll
        for (int p = 0; p < 4; ++p)
            c42[p] = -c42[p];
        neumann::store_fp32_accum(s_A, 48, 16, A_STRIDE, c42, lane);
    }

    if (wave == 0) {
        f32x4 t = mfma_f16(
            neumann::load_fp32_a_fragment_f16(s_A, 48, 0, A_STRIDE, lane),
            neumann::load_fp32_b_fragment_f16(s_A, 0, 0, A_STRIDE, lane),
            zero4);
        t = mfma_f16(saved_l42, neumann::accum_to_f16(kept_c21), t);
        t = mfma_f16(saved_l43, neumann::accum_to_f16(kept_c31), t);
        f32x4 c41 = mfma_f16(
            neumann::load_fp32_a_fragment_f16(s_A, 48, 48, A_STRIDE, lane),
            neumann::accum_to_f16(t), zero4);
#pragma unroll
        for (int p = 0; p < 4; ++p)
            c41[p] = -c41[p];
        neumann::store_fp32_accum(s_A, 48, 0, A_STRIDE, c41, lane);
    }
    __syncthreads();

    // Scatter the ten packed lower tiles in the established ABI order.
#pragma unroll
    for (int tile = 0; tile < 10; ++tile) {
        int tile_row;
        int tile_col;
        if (tile == 0) {
            tile_row = 0; tile_col = 0;
        } else if (tile == 1) {
            tile_row = 1; tile_col = 0;
        } else if (tile == 2) {
            tile_row = 1; tile_col = 1;
        } else if (tile == 3) {
            tile_row = 2; tile_col = 0;
        } else if (tile == 4) {
            tile_row = 2; tile_col = 1;
        } else if (tile == 5) {
            tile_row = 2; tile_col = 2;
        } else if (tile == 6) {
            tile_row = 3; tile_col = 0;
        } else if (tile == 7) {
            tile_row = 3; tile_col = 1;
        } else if (tile == 8) {
            tile_row = 3; tile_col = 2;
        } else {
            tile_row = 3; tile_col = 3;
        }

        if (tile_row < nch) {
            const int r = tid / C;
            const int c = tid % C;
            const __bf16 value = f32_to_bf16(
                s_A[(tile_row * C + r) * A_STRIDE + tile_col * C + c]);
            if (tile_row == tile_col) {
                ws_inv[(int64_t(ht0 + tile_row) * C + r) * C + c] = value;
            } else if (tile_row == 1) {
                cross32[(int64_t(xp0) * C + r) * C + c] = value;
            } else if (tile_row == 3 && tile_col == 2) {
                cross32[(int64_t(xp0 + 1) * C + r) * C + c] = value;
            } else {
                const int cross_tile = tile_row == 2
                    ? tile_col : tile_col + 2;
                cross64[((int64_t(xs) * 4 + cross_tile) * C + r) * C + c] =
                    value;
            }
        }
    }
}

}  // namespace flashkda_hip::gfx942
