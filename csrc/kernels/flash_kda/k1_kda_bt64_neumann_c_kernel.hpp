// FlashKDA BT64 K1 lower-triangular factor for the native HIP/MFMA backend.
//
// This kernel is the BT64 counterpart of k1_kda_split_solve_kernel.  It does
// not consume raw q/k/g: k1_kda_split_prep_kernel must run first and provide
// the per-BT16, per-feature operands
//
//   ws_kd       = k_norm * 2^(chunk-local g_cumsum)
//   ws_kr       = k_norm * 2^(chunk_total - chunk-local g_cumsum)
//   ws_gt       = chunk gate total, in base-2 exponent units.
//
// For row chunk r and column chunk c (r >= c), the matrix formed here is
//
//   L_rc = sigmoid(beta_r) * ws_kd_r @ key_rc^T,
//   key_cc = ws_kr_c * 2^(-ws_gt_c),
//   key_rc = ws_kr_c * 2^sum_{u=c+1}^{r-1}(ws_gt_u), r > c.
//
// The suffix is non-positive for the supported KDA gate, so both MFMA inputs
// stay bounded.  In particular, the kernel deliberately does not construct a
// segment-global inverse-decay operand, which could overflow even when the
// product is finite.  The resulting C=(I+tril(L,-1))^-1 is written in the ten
// 16x16 shards already consumed by the BT64 C-split K2 kernels.
//
// Grid / launch contract:
//   dense:  grid = (ceil(NT/4), N*H), block = 256
//   varlen: grid = (total_segments, H), block = 256
//   dynamic LDS bytes = kK1Bt64NeumannCSmemBytes
//
// ws_mqk is intentionally not part of this ABI.  The production direct-RTP
// K6 path never consumes it and therefore launches
//
//   split_prep -> this kernel -> recurrent scan -> chunk-parallel K6.
//
// Experimental replay-output routes may still run split_solve before this
// kernel solely to materialize their legacy ws_mqk operand.
//
// This kernel overwrites ws_inv and materializes chunks 1..3 of ws_kd relative
// to the BT64 segment start.  Therefore it must replace (not follow) the old
// BT32/BT64 merge kernels, and split_solve must never run after it.
#pragma once

#include <hip/hip_runtime.h>

#include "mfma.hpp"

namespace flashkda_hip {

namespace k1_bt64_neumann_c_detail {

constexpr int C = 16;
constexpr int BT = 64;
constexpr int D = 128;
constexpr int NTHREADS = 256;
constexpr int K_STRIDE = D + 4;
constexpr int A_STRIDE = BT + 1;
constexpr int K_VECS_PER_ROW = D / 8;
constexpr int KKT_TILES_PER_WAVE = 3;

// LDS is live in two disjoint phases:
//   KKT: kd[64,132] + one streamed ki[16,132] + beta[64]
//   solve: fp32 A/C[64,65], aliased onto the KKT allocation.
constexpr int KD_ELEMS = BT * K_STRIDE;
constexpr int KI_ELEMS = C * K_STRIDE;
constexpr int KKT_BYTES = (KD_ELEMS + KI_ELEMS) * 2 + BT * 4;
constexpr int A_BYTES = BT * A_STRIDE * 4;
static_assert(KKT_BYTES == 21376, "unexpected BT64 K1 LDS size");
static_assert(A_BYTES <= KKT_BYTES, "A/C alias must fit in KKT LDS");

__device__ __forceinline__ bool tile_for_slot(
        int wave, int slot, int& tile_row, int& tile_col) {
    if (slot == 0) {
        tile_row = wave;
        tile_col = wave;
        return true;
    }
    if (slot == 1) {
        tile_row = wave < 2 ? wave + 1 : wave;
        tile_col = wave > 1 ? wave - 1 : 0;
        return true;
    }
    if (slot == 2 && wave < 2) {
        tile_row = 3;
        tile_col = wave;
        return true;
    }
    tile_row = 0;
    tile_col = 0;
    return false;
}

// Contract-last MFMA fragment.  Passing two fragments loaded this way computes
// A[row,:] @ B[row,:]^T, which is the layout needed for kd @ ki^T.
__device__ __forceinline__ bf16x4 load_contract_fragment(
        const __bf16* __restrict__ lds,
        int row_base, int col_base, int stride, int lane) {
    const int addr =
        (row_base + (lane & 15)) * stride + col_base + ((lane >> 4) << 2);
    return *reinterpret_cast<const bf16x4*>(lds + addr);
}

// Standard-matmul A fragment from a row-major fp32 LDS matrix, rounded to bf16.
__device__ __forceinline__ bf16x4 load_fp32_a_fragment(
        const float* __restrict__ lds,
        int row_base, int col_base, int stride, int lane) {
    const int addr =
        (row_base + (lane & 15)) * stride + col_base + ((lane >> 4) << 2);
    return bf16x4{
        f32_to_bf16(lds[addr]),
        f32_to_bf16(lds[addr + 1]),
        f32_to_bf16(lds[addr + 2]),
        f32_to_bf16(lds[addr + 3])};
}

// Standard-matmul B fragment from a row-major fp32 LDS matrix, rounded to bf16.
__device__ __forceinline__ bf16x4 load_fp32_b_fragment(
        const float* __restrict__ lds,
        int row_base, int col_base, int stride, int lane) {
    const int n = lane & 15;
    const int kb = (lane >> 4) << 2;
    return bf16x4{
        f32_to_bf16(lds[(row_base + kb) * stride + col_base + n]),
        f32_to_bf16(lds[(row_base + kb + 1) * stride + col_base + n]),
        f32_to_bf16(lds[(row_base + kb + 2) * stride + col_base + n]),
        f32_to_bf16(lds[(row_base + kb + 3) * stride + col_base + n])};
}

// An MFMA accumulator already has the native B-fragment lane layout.
__device__ __forceinline__ bf16x4 accum_to_bf16(f32x4 x) {
    return bf16x4{
        f32_to_bf16(x[0]), f32_to_bf16(x[1]),
        f32_to_bf16(x[2]), f32_to_bf16(x[3])};
}

// The original FlashKDA BT16 solve stores L in fp16 before applying the
// finite Neumann factorization.  Keep that precision here as well: CDNA3 has
// the same MFMA throughput for fp16 and bf16, while fp16's wider mantissa is
// important when the BT64 block DAG is applied hundreds of times in a long
// sequence.
__device__ __forceinline__ f16x4 load_fp32_a_fragment_f16(
        const float* __restrict__ lds,
        int row_base, int col_base, int stride, int lane) {
    const int addr =
        (row_base + (lane & 15)) * stride + col_base + ((lane >> 4) << 2);
    return f16x4{
        f32_to_f16(lds[addr]),
        f32_to_f16(lds[addr + 1]),
        f32_to_f16(lds[addr + 2]),
        f32_to_f16(lds[addr + 3])};
}

__device__ __forceinline__ f16x4 load_fp32_b_fragment_f16(
        const float* __restrict__ lds,
        int row_base, int col_base, int stride, int lane) {
    const int n = lane & 15;
    const int kb = (lane >> 4) << 2;
    return f16x4{
        f32_to_f16(lds[(row_base + kb) * stride + col_base + n]),
        f32_to_f16(lds[(row_base + kb + 1) * stride + col_base + n]),
        f32_to_f16(lds[(row_base + kb + 2) * stride + col_base + n]),
        f32_to_f16(lds[(row_base + kb + 3) * stride + col_base + n])};
}

__device__ __forceinline__ f16x4 accum_to_f16(f32x4 x) {
    return f16x4{
        f32_to_f16(x[0]), f32_to_f16(x[1]),
        f32_to_f16(x[2]), f32_to_f16(x[3])};
}

__device__ __forceinline__ void store_fp32_accum(
        float* __restrict__ lds,
        int row_base, int col_base, int stride, f32x4 x, int lane) {
    const int n = lane & 15;
    const int mb = (lane >> 4) << 2;
    lds[(row_base + mb) * stride + col_base + n] = x[0];
    lds[(row_base + mb + 1) * stride + col_base + n] = x[1];
    lds[(row_base + mb + 2) * stride + col_base + n] = x[2];
    lds[(row_base + mb + 3) * stride + col_base + n] = x[3];
}

__device__ __forceinline__ float chunk_prefix(
        const float* __restrict__ ws_gt, int ht0, int chunk, int d) {
    float value = 0.0f;
#pragma unroll
    for (int u = 0; u < 3; ++u) {
        if (u < chunk)
            value += ws_gt[int64_t(ht0 + u) * D + d];
    }
    return value;
}

__device__ __forceinline__ float restored_key_scale(
        const float* __restrict__ ws_gt,
        int ht0, int col_chunk, int row_chunk, int d) {
    if (row_chunk == col_chunk)
        return -ws_gt[int64_t(ht0 + col_chunk) * D + d];
    float value = 0.0f;
#pragma unroll
    for (int u = 0; u < 3; ++u) {
        if (u > col_chunk && u < row_chunk)
            value += ws_gt[int64_t(ht0 + u) * D + d];
    }
    return value;
}

__device__ __forceinline__ float chunk_decay_prefix(
        const float* __restrict__ ws_decay, int ht0, int chunk, int d) {
    float value = 1.0f;
#pragma unroll
    for (int u = 0; u < 3; ++u) {
        if (u < chunk)
            value *= ws_decay[int64_t(ht0 + u) * D + d];
    }
    return value;
}

__device__ __forceinline__ f32x4 bounded_key_factor4(
        const float* __restrict__ ws_decay,
        int ht0, int col_chunk, int row_chunk, int d0) {
    f32x4 decay[3];
#pragma unroll
    for (int u = 0; u < 3; ++u) {
        if (u >= col_chunk && u < row_chunk)
            decay[u] = *reinterpret_cast<const f32x4*>(
                ws_decay + int64_t(ht0 + u) * D + d0);
    }

    f32x4 value = {1.0f, 1.0f, 1.0f, 1.0f};
#pragma unroll
    for (int u = 0; u < 3; ++u) {
        if (u >= col_chunk && u < row_chunk) {
#pragma unroll
            for (int p = 0; p < 4; ++p)
                value[p] *= decay[u][p];
        }
    }
    return value;
}

}  // namespace k1_bt64_neumann_c_detail

// Pass this as the dynamic shared-memory byte count at launch.
constexpr int kK1Bt64NeumannCSmemBytes =
    k1_bt64_neumann_c_detail::KKT_BYTES;

template <
    bool VL,
    bool USE_DECAY_TABLE = false,
    bool BETA_FROM_PREP = false>
__global__ void __launch_bounds__(256)
k1_kda_bt64_neumann_c_kernel(
        const float* __restrict__ beta_src,     // logits or activated cache
        __bf16* __restrict__ ws_kd,             // in/out [n_ht,16,128]
        const __bf16* __restrict__ ws_kr,       // bounded restored K [n_ht,16,128]
        const __bf16* __restrict__ tmp_kinv,    // local inverse-decay K
        const float* __restrict__ ws_gt,        // [n_ht,128]
        const float* __restrict__ ws_decay,     // [n_ht,128], aliases ws_mqk
        __bf16* __restrict__ ws_inv,            // diagonal C [n_ht,16,16]
        __bf16* __restrict__ cross32,           // C10 [H*total_pairs,16,16]
        __bf16* __restrict__ cross64,           // four tiles/BT64 segment
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ tile_prefix,
        const int* __restrict__ pair_prefix,
        const int* __restrict__ segment_prefix,
        int N, int total_tiles, int total_pairs, int total_segments,
        int T_seq, int H, int NT) {
    using namespace k1_bt64_neumann_c_detail;

    const int tid = static_cast<int>(threadIdx.x);
    const int wave = tid >> 6;
    const int lane = tid & 63;

    int h;
    int ht0;
    int xp0;
    int xs;
    int t0;
    int alen;

    // All returns are CTA-uniform; barriers only occur after this mapping.
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
        alen = len - local_seg * BT;
        if (alen > BT)
            alen = BT;
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
        alen = T_seq - token0;
        if (alen > BT)
            alen = BT;
        t0 = b * T_seq + token0;
        ht0 = bh * NT + seg * 4;
        xp0 = bh * ((NT + 1) / 2) + seg * 2;
        xs = bh * ((NT + 3) / 4) + seg;
    }

    const int nch = (alen + C - 1) / C;

    extern __shared__ __bf16 smem[];
    __bf16* const s_kd = smem;                         // [64,132]
    __bf16* const s_ki = s_kd + KD_ELEMS;             // [16,132]
    float* const s_beta = reinterpret_cast<float*>(s_ki + KI_ELEMS);
    float* const s_A = reinterpret_cast<float*>(smem); // aliases KKT storage

    // Load all four chunk-local kd tiles.  KKT intentionally uses these bounded
    // local values.  In parallel, write the segment-relative values required by
    // the BT64 recurrent K2 back to chunks 1..3 of ws_kd.
    for (int vi = tid; vi < BT * K_VECS_PER_ROW; vi += NTHREADS) {
        const int row = vi / K_VECS_PER_ROW;
        const int col = (vi % K_VECS_PER_ROW) * 8;
        const int chunk = row / C;
        const int chunk_row = row & (C - 1);
        const bool chunk_exists = chunk < nch;
        const bool row_valid = row < alen;

        bf16x8 local{};
        if (row_valid) {
            local = *reinterpret_cast<const bf16x8*>(
                ws_kd + (int64_t(ht0 + chunk) * C + chunk_row) * D + col);
        }
        *reinterpret_cast<bf16x8*>(s_kd + row * K_STRIDE + col) = local;

        if (chunk_exists && chunk > 0) {
            bf16x8 segment_value{};
#pragma unroll
            for (int p = 0; p < 8; ++p) {
                const int d = col + p;
                const float factor = USE_DECAY_TABLE
                    ? chunk_decay_prefix(ws_decay, ht0, chunk, d)
                    : ex2(chunk_prefix(ws_gt, ht0, chunk, d));
                segment_value[p] = row_valid
                    ? f32_to_bf16(bf16_to_f32(local[p]) * factor)
                    : (__bf16)0.0f;
            }
            *reinterpret_cast<bf16x8*>(
                ws_kd + (int64_t(ht0 + chunk) * C + chunk_row) * D + col) =
                segment_value;
        }
    }

    if (tid < BT) {
        const float beta = tid < alen
            ? (BETA_FROM_PREP
                ? beta_src[int64_t(xs) * BT + tid]
                : sigmoid_tanh(beta_src[int64_t(t0 + tid) * H + h]))
            : 0.0f;
        s_beta[tid] = beta;
        if constexpr (USE_DECAY_TABLE) {
            // K6 scan reuses the exact fp32 activation used to construct C.
            // The cache is appended immediately after the complete cross64
            // array, so no extra kernel pointer is needed.
            float* const cs_beta = reinterpret_cast<float*>(
                cross64 + int64_t(H) * total_segments * 4 * C * C);
            if constexpr (!BETA_FROM_PREP) {
                cs_beta[int64_t(xs) * BT + tid] = beta;
            }
        }
    }
    __syncthreads();

    const f32x4 zero4 = {0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 kkt[KKT_TILES_PER_WAVE];
#pragma unroll
    for (int slot = 0; slot < KKT_TILES_PER_WAVE; ++slot)
        kkt[slot] = zero4;

    // Stream one bounded restored-key tile at a time.  Diagonal tiles undo the
    // chunk-total factor to recover ki; off-diagonal tiles apply only the
    // intervening complete chunks.  This avoids materializing the potentially
    // enormous inverse-decay operand before its cancelling decay is applied.
#pragma unroll
    for (int col_chunk = 0; col_chunk < 4; ++col_chunk) {
        const int row = tid / K_VECS_PER_ROW;
        const int col = (tid % K_VECS_PER_ROW) * 8;
        const int global_row = col_chunk * C + row;
        bf16x8 value{};
        if (col_chunk < nch && global_row < alen) {
            value = *reinterpret_cast<const bf16x8*>(
                (USE_DECAY_TABLE ? tmp_kinv : ws_kr) +
                (int64_t(ht0 + col_chunk) * C + row) * D + col);
        }
        *reinterpret_cast<bf16x8*>(s_ki + row * K_STRIDE + col) = value;
        __syncthreads();

#pragma unroll
        for (int slot = 0; slot < KKT_TILES_PER_WAVE; ++slot) {
            int tile_row;
            int tile_col;
            const bool active = tile_for_slot(wave, slot, tile_row, tile_col);
            if (active && tile_col == col_chunk && tile_row < nch) {
#pragma unroll
                for (int ek = 0; ek < D / C; ++ek) {
                    const bf16x4 a = load_contract_fragment(
                        s_kd, tile_row * C, ek * C, K_STRIDE, lane);
                    bf16x4 b = load_contract_fragment(
                        s_ki, 0, ek * C, K_STRIDE, lane);
                    if constexpr (USE_DECAY_TABLE) {
                        const int d0 =
                            ek * C + ((lane >> 4) << 2);
                        const f32x4 factor = bounded_key_factor4(
                            ws_decay, ht0, tile_col, tile_row, d0);
#pragma unroll
                        for (int p = 0; p < 4; ++p) {
                            b[p] = f32_to_bf16(
                                bf16_to_f32(b[p]) * factor[p]);
                        }
                    } else {
#pragma unroll
                        for (int p = 0; p < 4; ++p) {
                            const int d = ek * C + ((lane >> 4) << 2) + p;
                            const float factor = ex2(restored_key_scale(
                                ws_gt, ht0, tile_col, tile_row, d));
                            b[p] = f32_to_bf16(
                                bf16_to_f32(b[p]) * factor);
                        }
                    }
                    kkt[slot] = mfma_bf16(a, b, kkt[slot]);
                }
            }
        }
        __syncthreads();
    }

    // All KKT inputs are dead.  Alias their LDS as a padded fp32 L/C matrix.
#pragma unroll
    for (int slot = 0; slot < KKT_TILES_PER_WAVE; ++slot) {
        int tile_row;
        int tile_col;
        if (!tile_for_slot(wave, slot, tile_row, tile_col))
            continue;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int row = tile_row * C + ((lane >> 4) << 2) + p;
            const int col = tile_col * C + (lane & 15);
            float l = 0.0f;
            if (row < alen && row > col) {
                if (tile_row == tile_col) {
                    // Match the established BT16 triangular solve: both the
                    // dot product and beta are rounded before the fp16
                    // multiply that forms the strictly-lower diagonal block.
                    const _Float16 dot_h = f32_to_f16(kkt[slot][p]);
                    const _Float16 beta_h = f32_to_f16(s_beta[row]);
                    l = f16_to_f32(f32_to_f16(
                        f16_to_f32(dot_h) * f16_to_f32(beta_h)));
                } else {
                    l = kkt[slot][p] * s_beta[row];
                }
            }
            s_A[row * A_STRIDE + col] = l;
        }
    }
    __syncthreads();

    // Invert each diagonal 16x16 block with the exact finite factorization for
    // B=-L, B^16=0:
    //   (I+L)^-1 = (I+B)(I+B^2)(I+B^4)(I+B^8).
    //
    // The ascending evaluation matches FlashKDA's accurate BT16 solve.  C must
    // be reloaded as an A fragment between factors; an MFMA accumulator has the
    // B-fragment lane layout and cannot be fed back as A directly.  The KKT
    // allocation has 4736 dead bytes after s_A, enough for per-wave B and power
    // tiles without increasing dynamic LDS or reducing occupancy.
    {
        const int block_row = wave * C;
        auto* const diag_scratch = reinterpret_cast<_Float16*>(
            reinterpret_cast<char*>(smem) + A_BYTES);
        _Float16* const b_tile = diag_scratch + wave * C * C;
        _Float16* const power_tile =
            diag_scratch + 4 * C * C + wave * C * C;
        static_assert(A_BYTES + 8 * C * C * int(sizeof(_Float16)) <= KKT_BYTES,
                      "diagonal solve scratch must fit the KKT allocation");

        for (int i = lane; i < C * C; i += 64) {
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
            const f16x4 c_a = load_fp32_a_fragment_f16(
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

    // Merge the four diagonal inverses through the lower-block dependency DAG.
    // Pre-save L blocks that are overwritten at earlier levels.
    f16x4 saved_l32{};
    f16x4 saved_l43{};
    f16x4 saved_l42{};
    if (wave == 0) {
        saved_l32 = load_fp32_a_fragment_f16(s_A, 32, 16, A_STRIDE, lane);
        saved_l43 = load_fp32_a_fragment_f16(s_A, 48, 32, A_STRIDE, lane);
        saved_l42 = load_fp32_a_fragment_f16(s_A, 48, 16, A_STRIDE, lane);
    } else if (wave == 1) {
        saved_l43 = load_fp32_a_fragment_f16(s_A, 48, 32, A_STRIDE, lane);
    }
    // All waves must finish preserving L32/L42/L43 before Level 1 starts
    // overwriting those LDS tiles with C32/C43.
    __syncthreads();

    f32x4 kept_c21 = zero4;
    f32x4 kept_c32 = zero4;
    f32x4 kept_c31 = zero4;

    // Level 1: C21, C32, C43.
    if (wave == 0) {
        f32x4 t = mfma_f16(
            load_fp32_a_fragment_f16(s_A, 16, 0, A_STRIDE, lane),
            load_fp32_b_fragment_f16(s_A, 0, 0, A_STRIDE, lane), zero4);
        kept_c21 = mfma_f16(
            load_fp32_a_fragment_f16(s_A, 16, 16, A_STRIDE, lane),
            accum_to_f16(t), zero4);
#pragma unroll
        for (int p = 0; p < 4; ++p)
            kept_c21[p] = -kept_c21[p];
        store_fp32_accum(s_A, 16, 0, A_STRIDE, kept_c21, lane);
    } else if (wave == 1) {
        f32x4 t = mfma_f16(
            load_fp32_a_fragment_f16(s_A, 32, 16, A_STRIDE, lane),
            load_fp32_b_fragment_f16(s_A, 16, 16, A_STRIDE, lane), zero4);
        kept_c32 = mfma_f16(
            load_fp32_a_fragment_f16(s_A, 32, 32, A_STRIDE, lane),
            accum_to_f16(t), zero4);
#pragma unroll
        for (int p = 0; p < 4; ++p)
            kept_c32[p] = -kept_c32[p];
        store_fp32_accum(s_A, 32, 16, A_STRIDE, kept_c32, lane);
    } else if (wave == 2) {
        f32x4 t = mfma_f16(
            load_fp32_a_fragment_f16(s_A, 48, 32, A_STRIDE, lane),
            load_fp32_b_fragment_f16(s_A, 32, 32, A_STRIDE, lane), zero4);
        f32x4 c43 = mfma_f16(
            load_fp32_a_fragment_f16(s_A, 48, 48, A_STRIDE, lane),
            accum_to_f16(t), zero4);
#pragma unroll
        for (int p = 0; p < 4; ++p)
            c43[p] = -c43[p];
        store_fp32_accum(s_A, 48, 32, A_STRIDE, c43, lane);
    }

    // Level 2: C31, C42.  All dependencies are wave-local registers.
    if (wave == 0) {
        f32x4 t = mfma_f16(
            load_fp32_a_fragment_f16(s_A, 32, 0, A_STRIDE, lane),
            load_fp32_b_fragment_f16(s_A, 0, 0, A_STRIDE, lane), zero4);
        t = mfma_f16(saved_l32, accum_to_f16(kept_c21), t);
        kept_c31 = mfma_f16(
            load_fp32_a_fragment_f16(s_A, 32, 32, A_STRIDE, lane),
            accum_to_f16(t), zero4);
#pragma unroll
        for (int p = 0; p < 4; ++p)
            kept_c31[p] = -kept_c31[p];
        store_fp32_accum(s_A, 32, 0, A_STRIDE, kept_c31, lane);
    } else if (wave == 1) {
        f32x4 t = mfma_f16(
            load_fp32_a_fragment_f16(s_A, 48, 16, A_STRIDE, lane),
            load_fp32_b_fragment_f16(s_A, 16, 16, A_STRIDE, lane), zero4);
        t = mfma_f16(saved_l43, accum_to_f16(kept_c32), t);
        f32x4 c42 = mfma_f16(
            load_fp32_a_fragment_f16(s_A, 48, 48, A_STRIDE, lane),
            accum_to_f16(t), zero4);
#pragma unroll
        for (int p = 0; p < 4; ++p)
            c42[p] = -c42[p];
        store_fp32_accum(s_A, 48, 16, A_STRIDE, c42, lane);
    }

    // Level 3: C41.
    if (wave == 0) {
        f32x4 t = mfma_f16(
            load_fp32_a_fragment_f16(s_A, 48, 0, A_STRIDE, lane),
            load_fp32_b_fragment_f16(s_A, 0, 0, A_STRIDE, lane), zero4);
        t = mfma_f16(saved_l42, accum_to_f16(kept_c21), t);
        t = mfma_f16(saved_l43, accum_to_f16(kept_c31), t);
        f32x4 c41 = mfma_f16(
            load_fp32_a_fragment_f16(s_A, 48, 48, A_STRIDE, lane),
            accum_to_f16(t), zero4);
#pragma unroll
        for (int p = 0; p < 4; ++p)
            c41[p] = -c41[p];
        store_fp32_accum(s_A, 48, 0, A_STRIDE, c41, lane);
    }
    __syncthreads();

    // Scatter the packed lower triangle expected by BT64 K2.  The order is:
    //   0:C00, 1:C10, 2:C11, 3:C20, 4:C21,
    //   5:C22, 6:C30, 7:C31, 8:C32, 9:C33.
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
                int cross_tile;
                if (tile_row == 2)
                    cross_tile = tile_col;
                else
                    cross_tile = tile_col + 2;
                cross64[((int64_t(xs) * 4 + cross_tile) * C + r) * C + c] =
                    value;
            }
        }
    }
}

}  // namespace flashkda_hip
