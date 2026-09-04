// gfx942 experimental raw-input BT64 K1 fusion with four local-Mqk tiles.
//
// This file is intentionally independent of the production kernel.  During
// each reverse chunk step, a wave that is idle in the KKT schedule consumes
// the just-published Qd tile and the resident Ki tile to form
// tril(Qd @ Ki^T).  The packed output ABI is [segment,4,16,16].
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

namespace k1_bt64_fused_prepare_neumann_local_mqk_detail {

constexpr int C = 16;
constexpr int BT = 64;
constexpr int D = 128;
constexpr int NTHREADS = 256;
constexpr int K_STRIDE = D + 4;
constexpr int A_STRIDE = BT + 1;
constexpr int K_VECS_PER_ROW = D / 8;
constexpr int KKT_TILES_PER_WAVE = 3;
constexpr int PAIR_KKT_TILES_PER_WAVE = 2;
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

// The C32 pair consumer never crosses a BT32 boundary.  Slot zero assigns
// one diagonal tile to every wave; slot one assigns the two adjacent tiles
// to waves 0/3.  Keeping this schedule separate from tile_for_slot lets the
// compiler remove the third accumulator and its complete MFMA path.
__device__ __forceinline__ bool pair_tile_for_slot(
        int wave, int slot, int& tile_row, int& tile_col) {
    if (slot == 0) {
        tile_row = wave;
        tile_col = wave;
        return true;
    }
    if (slot == 1 && (wave == 0 || wave == 3)) {
        tile_row = wave == 0 ? 1 : 3;
        tile_col = wave == 0 ? 0 : 2;
        return true;
    }
    tile_row = 0;
    tile_col = 0;
    return false;
}

}  // namespace k1_bt64_fused_prepare_neumann_local_mqk_detail

// Static LDS footprint of the experimental kernel.  The launch does not need
// dynamic shared memory.
constexpr int kK1Bt64FusedPrepareNeumannLocalMqkSmemBytes =
    k1_bt64_fused_prepare_neumann_local_mqk_detail::PREP_KKT_BYTES;

// C32_LAYOUT and C16_LAYOUT are isolated input-layout experiments for fused
// recurrent/output consumers.  The default keeps the production BT64 ABI
// byte-for-byte unchanged.  In C32 mode:
//   * Kd and Kr are made relative to each adjacent BT32 pair instead of BT64;
//   * the compact Mqk arena has six tiles per segment: four diagonal BT16
//     tiles followed by cross tiles (1,0) and (3,2).
// C32_PAIR_ONLY is an isolated specialization of that six-tile ABI.  It
// publishes only four diagonal inverses and two adjacent cross32 factors;
// cross-pair KKT, the upper BT64 merge levels, and cross64 are not formed.
// In C16 mode Kd remains chunk-local, Kr is unchanged, and the existing four
// local Mqk tiles are consumed directly by the serial C16 register-state path.
// Qd deliberately stays chunk-local; the C32 consumer applies the first
// chunk's decay while publishing the second Q row block.
template <bool VL, bool USE_DPP = true,
          bool C32_LAYOUT = false, bool C16_LAYOUT = false,
          bool C32_PAIR_ONLY = false>
__global__ void __launch_bounds__(256)
k1_kda_bt64_fused_prepare_neumann_local_mqk_kernel(
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
        __bf16* __restrict__ local_mqk,       // [segment,4|6,16,16]
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ tile_prefix,
        const int* __restrict__ pair_prefix,
        const int* __restrict__ segment_prefix,
        int N, int total_tiles, int total_pairs, int total_segments,
        int T_seq, int H, int NT) {
    using namespace k1_bt64_fused_prepare_neumann_local_mqk_detail;
    namespace split = flashkda_hip::k1_split_prep_detail;
    namespace neumann = flashkda_hip::k1_bt64_neumann_c_detail;
    static_assert(!(C32_LAYOUT && C16_LAYOUT),
                  "C32 and C16 workspace layouts are mutually exclusive");
    static_assert(!C32_PAIR_ONLY || C32_LAYOUT,
                  "the pair-only factor path requires the C32 layout");
    static_assert(!C32_PAIR_ONLY || !C16_LAYOUT,
                  "the C32 pair-only and C16 paths are mutually exclusive");

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
        if constexpr (!C16_LAYOUT) {
            xp0 = h * total_pairs + pair_prefix[lo] + local_seg * 2;
            xs = h * total_segments + gsi;
        }
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
        if constexpr (!C16_LAYOUT) {
            xp0 = bh * ((NT + 1) / 2) + seg * 2;
            xs = bh * ((NT + 3) / 4) + seg;
        }
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
        if constexpr (C16_LAYOUT) {
            // The matched C16 register-state scan indexes beta with the same
            // head-major workspace-tile id as kd/qd/kr.  Publish only real
            // tokens: in packed mode the tile immediately after a short tail
            // can already belong to the next sequence, so zero-filling the
            // unused part of this BT64 segment would corrupt its beta cache.
            if (tid < alen)
                beta_cache[int64_t(ht0) * C + tid] = b;
        } else {
            beta_cache[int64_t(xs) * BT + tid] = b;
        }
    }

    const float a = ex2(A_log[h] * KDA_LOG2E);
    const __bf16 scale_bf = f32_to_bf16(scale);
    const int row_lane = tid & 15;
    const int vec_m = tid >> 4;
    const int vec_d0 = row_lane * 8;
    const int vec_idx = vec_m * D + vec_d0;

    const f32x4 zero4 = {0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 kkt_diag = zero4;
    constexpr int KKT_SLOTS = C32_PAIR_ONLY
        ? PAIR_KKT_TILES_PER_WAVE : KKT_TILES_PER_WAVE;
    f32x4 kkt[KKT_SLOTS];
#pragma unroll
    for (int slot = 0; slot < KKT_SLOTS; ++slot) {
        if constexpr (!C16_LAYOUT)
            kkt[slot] = zero4;
    }

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
                if constexpr (!C16_LAYOUT)
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
            if constexpr (C32_LAYOUT) {
                // For the first BT16 block in a BT32 pair, carry its state
                // update through the second block's decay in advance.  Reverse
                // preparation guarantees that decay[chunk+1] is already live.
                if ((chunk & 1) == 0 && chunk + 1 < nch) {
                    const f32x4 next_decay0 =
                        *reinterpret_cast<const f32x4*>(
                            prep.decay + (chunk + 1) * D + vec_d0);
                    const f32x4 next_decay1 =
                        *reinterpret_cast<const f32x4*>(
                            prep.decay + (chunk + 1) * D + vec_d0 + 4);
#pragma unroll
                    for (int p = 0; p < 8; ++p) {
                        const float f = p < 4
                            ? next_decay0[p] : next_decay1[p - 4];
                        kr_v[p] = f32_to_bf16(bf16_to_f32(kr_v[p]) * f);
                    }
                }
            }
            *reinterpret_cast<bf16x8*>(ws_kr + ws_off) = kr_v;
        }
        __syncthreads();

        if constexpr (C16_LAYOUT) {
            if (wave == chunk && chunk < nch) {
#pragma unroll
                for (int ek = 0; ek < D / C; ++ek) {
                    const bf16x4 af = neumann::load_contract_fragment(
                        prep.kd, chunk * C, ek * C, K_STRIDE, lane);
                    const bf16x4 bf = neumann::load_contract_fragment(
                        prep.ki, 0, ek * C, K_STRIDE, lane);
                    kkt_diag = mfma_bf16(af, bf, kkt_diag);
                }
            }
        } else {
#pragma unroll
            for (int slot = 0; slot < KKT_SLOTS; ++slot) {
                int tile_row;
                int tile_col;
                bool active;
                if constexpr (C32_PAIR_ONLY) {
                    active = pair_tile_for_slot(
                        wave, slot, tile_row, tile_col);
                } else {
                    active = neumann::tile_for_slot(
                        wave, slot, tile_row, tile_col);
                }
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
        }
        // The KKT schedule leaves at least one complete wave idle for every
        // column.  Use one such wave for the local causal QK tile.  Qd was
        // published to global memory before the preceding CTA barrier, while
        // the exact inverse-decayed Ki remains resident in LDS.  No extra LDS
        // allocation or end-of-column barrier is needed.
        const int mqk_wave = chunk == 0 ? 2 : 0;
        if (wave == mqk_wave) {
            f32x4 mqk = zero4;
            if (chunk < nch) {
#pragma unroll
                for (int ek = 0; ek < D / C; ++ek) {
                    const bf16x4 qd_frag = neumann::load_contract_fragment(
                        ws_qd + int64_t(ht0 + chunk) * C * D,
                        0, ek * C, D, lane);
                    const bf16x4 ki_frag = neumann::load_contract_fragment(
                        prep.ki, 0, ek * C, K_STRIDE, lane);
                    mqk = mfma_bf16(qd_frag, ki_frag, mqk);
                }
            }
            const int n = lane & 15;
            const int m4 = (lane >> 4) << 2;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const int m = m4 + p;
                const bool valid = m < chunk_rows && n < chunk_rows && m >= n;
                constexpr int MQK_TILES = C32_LAYOUT ? 6 : 4;
                const int64_t mqk_tile = C16_LAYOUT
                    ? int64_t(ht0 + chunk)
                    : int64_t(xs) * MQK_TILES + chunk;
                // Fixed segment arenas own four slots and explicitly zero
                // absent tail chunks.  The C16 arena is compact by ht, so a
                // missing chunk has no slot; writing it would overlap the next
                // packed sequence (or run past a dense tail allocation).
                if (!C16_LAYOUT || chunk < nch) {
                    local_mqk[(mqk_tile * C + m) * C + n] =
                        valid ? f32_to_bf16(mqk[p]) : (__bf16)0.0f;
                }
            }
        }

        // C32 needs only the two adjacent cross-Mqk blocks.  At even columns
        // another complete wave is idle in the KKT schedule, so form
        //
        //   M10 = Qd_next @ (Ki_current * decay_current)^T
        //
        // alongside the local tile without extending the dependent KKT chain.
        // The next Qd tile is already in global memory because chunks are
        // prepared in reverse order; Ki_current remains resident in LDS.
        if constexpr (C32_LAYOUT) {
            const bool pair_first = (chunk & 1) == 0;
            const int cross_wave = chunk == 0 ? 3 : 1;
            if (pair_first && wave == cross_wave) {
                f32x4 cross_mqk = zero4;
                const bool has_second = chunk + 1 < nch;
                if (has_second) {
#pragma unroll
                    for (int ek = 0; ek < D / C; ++ek) {
                        const bf16x4 qd_frag = neumann::load_contract_fragment(
                            ws_qd + int64_t(ht0 + chunk + 1) * C * D,
                            0, ek * C, D, lane);
                        bf16x4 kr_frag = neumann::load_contract_fragment(
                            prep.ki, 0, ek * C, K_STRIDE, lane);
                        const int d0 = ek * C + ((lane >> 4) << 2);
                        const f32x4 d = *reinterpret_cast<const f32x4*>(
                            prep.decay + chunk * D + d0);
#pragma unroll
                        for (int p = 0; p < 4; ++p) {
                            // Match the published local Kr exactly: its chunk
                            // decay is rounded to BF16 before the Ki product.
                            const __bf16 dt = f32_to_bf16(d[p]);
                            kr_frag[p] = f32_to_bf16(
                                bf16_to_f32(kr_frag[p]) * bf16_to_f32(dt));
                        }
                        cross_mqk = mfma_bf16(qd_frag, kr_frag, cross_mqk);
                    }
                }
                const int n = lane & 15;
                const int m4 = (lane >> 4) << 2;
                const int pair = chunk >> 1;
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const int m = m4 + p;
                    const bool valid = has_second && m < C;
                    local_mqk[
                        ((int64_t(xs) * 6 + 4 + pair) * C + m) * C + n] =
                        valid ? f32_to_bf16(cross_mqk[p]) : (__bf16)0.0f;
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
            const float factor = C16_LAYOUT
                ? 1.0f
                : (C32_LAYOUT
                    ? ((chunk & 1) != 0
                        ? prep.decay[(chunk - 1) * D + col + p] : 1.0f)
                    : decay_prefix(prep.decay, chunk, col + p));
            segment_value[p] = f32_to_bf16(
                bf16_to_f32(local[p]) * factor);
        }
        *reinterpret_cast<bf16x8*>(
            ws_kd + (int64_t(ht0 + chunk) * C + chunk_row) * D + col) =
            segment_value;
    }
    __syncthreads();

    if constexpr (C16_LAYOUT) {
        // C16 consumers need four independent BT16 systems only.  Each wave
        // already owns one diagonal KKT accumulator, so keep the complete
        // solve wave-local and publish its ht-indexed inverse directly.  The
        // phase-union barrier above is the only CTA rendezvous required after
        // preparation; diagonal A/scratch regions are disjoint across waves.
        if (wave < nch) {
            float* const c16_A = smem.solve.A;
            const int block_row = wave * C;
            const int n = lane & 15;
            const int m4 = (lane >> 4) << 2;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const int row = block_row + m4 + p;
                const int col = block_row + n;
                float l = 0.0f;
                if (row < alen && row > col) {
                    const _Float16 dot_h = f32_to_f16(kkt_diag[p]);
                    const _Float16 beta_h = f32_to_f16(prep.beta[row]);
                    l = f16_to_f32(f32_to_f16(
                        f16_to_f32(dot_h) * f16_to_f32(beta_h)));
                }
                c16_A[row * A_STRIDE + col] = l;
            }
            __syncwarp();

            _Float16* const b_tile =
                smem.solve.diagonal_scratch + wave * FACTOR_ELEMS;
            _Float16* const power_tile =
                smem.solve.diagonal_scratch + 4 * FACTOR_ELEMS
                + wave * FACTOR_ELEMS;

            for (int i = lane; i < FACTOR_ELEMS; i += 64) {
                const int r = i / C;
                const int c = i % C;
                const _Float16 b = r > c
                    ? f32_to_f16(-c16_A[
                        (block_row + r) * A_STRIDE + block_row + c])
                    : f32_to_f16(0.0f);
                b_tile[i] = b;
                const _Float16 ci = f32_to_f16(
                    (r == c ? 1.0f : 0.0f) + f16_to_f32(b));
                c16_A[(block_row + r) * A_STRIDE + block_row + c] =
                    f16_to_f32(ci);
            }
            __syncwarp();

            f32x4 power = gemm_std_f16(b_tile, b_tile, lane);
            store_acc_16x16(power_tile, power, lane);
            __syncwarp();

#pragma unroll
            for (int level = 0; level < 3; ++level) {
                const f16x4 c_a = neumann::load_fp32_a_fragment_f16(
                    c16_A, block_row, block_row, A_STRIDE, lane);
                const int kb = (lane >> 4) << 2;
                const f16x4 p_b = {
                    power_tile[(kb + 0) * C + n],
                    power_tile[(kb + 1) * C + n],
                    power_tile[(kb + 2) * C + n],
                    power_tile[(kb + 3) * C + n]};
                const f32x4 term = mfma_f16(c_a, p_b, zero4);

#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const int addr =
                        (block_row + m4 + p) * A_STRIDE + block_row + n;
                    const _Float16 rounded_term = f32_to_f16(term[p]);
                    c16_A[addr] = f16_to_f32(f32_to_f16(
                        c16_A[addr] + f16_to_f32(rounded_term)));
                }
                __syncwarp();

                if (level < 2) {
                    power = gemm_std_f16(power_tile, power_tile, lane);
                    store_acc_16x16(power_tile, power, lane);
                    __syncwarp();
                }
            }

            for (int i = lane; i < FACTOR_ELEMS; i += 64) {
                const int r = i / C;
                const int c = i % C;
                ws_inv[(int64_t(ht0 + wave) * C + r) * C + c] =
                    f32_to_bf16(c16_A[
                        (block_row + r) * A_STRIDE + block_row + c]);
            }
        }
        return;
    }

    float* const s_A = smem.solve.A;

    // Convert the ten KKT accumulators to the same L representation used by
    // the split Neumann kernel.  Diagonal blocks retain its extra FP16 beta and
    // product rounding; cross blocks remain FP32 until their DAG fragments.
#pragma unroll
    for (int slot = 0; slot < KKT_SLOTS; ++slot) {
        int tile_row;
        int tile_col;
        bool active;
        if constexpr (C32_PAIR_ONLY) {
            active = pair_tile_for_slot(wave, slot, tile_row, tile_col);
        } else {
            active = neumann::tile_for_slot(
                wave, slot, tile_row, tile_col);
        }
        if (!active)
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
    if constexpr (C32_PAIR_ONLY)
        __syncwarp();
    else
        __syncthreads();

    // Four independent finite BT16 Neumann factorizations.
    if (!C32_PAIR_ONLY || wave < nch) {
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

    if constexpr (C32_PAIR_ONLY) {
        // Each diagonal solve is wave-local.  This is the sole CTA rendezvous
        // needed before a pair merge reads the neighboring diagonal inverse.
        __syncthreads();

        const bool pair_wave =
            (wave == 0 && nch >= 2) || (wave == 3 && nch >= 4);
        f32x4 pair_c = zero4;
        int pair = 0;
        if (pair_wave) {
            pair = wave == 0 ? 0 : 1;
            const int col0 = pair * 2 * C;
            const int row0 = col0 + C;
            const f32x4 t = mfma_f16(
                neumann::load_fp32_a_fragment_f16(
                    s_A, row0, col0, A_STRIDE, lane),
                neumann::load_fp32_b_fragment_f16(
                    s_A, col0, col0, A_STRIDE, lane),
                zero4);
            pair_c = mfma_f16(
                neumann::load_fp32_a_fragment_f16(
                    s_A, row0, row0, A_STRIDE, lane),
                neumann::accum_to_f16(t), zero4);
#pragma unroll
            for (int p = 0; p < 4; ++p)
                pair_c[p] = -pair_c[p];
        }

        // Direct publication avoids rebuilding the ten-tile BT64 scatter.
        // Missing chunks/pairs retain the caller's initialized padding, just
        // like the original conditional scatter.
        const int n = lane & 15;
        const int m4 = (lane >> 4) << 2;
        if (wave < nch) {
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const int m = m4 + p;
                ws_inv[(int64_t(ht0 + wave) * C + m) * C + n] =
                    f32_to_bf16(s_A[
                        (wave * C + m) * A_STRIDE + wave * C + n]);
            }
        }
        if (pair_wave) {
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const int m = m4 + p;
                cross32[(int64_t(xp0 + pair) * C + m) * C + n] =
                    f32_to_bf16(pair_c[p]);
            }
        }
        return;
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
