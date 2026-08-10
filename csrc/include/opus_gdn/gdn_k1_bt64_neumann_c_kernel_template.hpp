// GDN prefill K1-C kernel -- BT=64 Neumann variant for gfx942.
//
// This is the algebra-split K1 used to evaluate the RTP-LLM formulation:
// it computes the chunk-local gate cumsum and
//
//     C = (I + tril(beta_i * exp(g_i - g_j) * k_i k_j^T, -1))^-1
//
// but deliberately does not form the legacy W/U factors.
//
// Grid: (ceil(T / 64), B * H), block: 256 threads (4 wave64 waves).
// C layout: bf16 [B, T, H, 64], with the final dimension contiguous.
#pragma once

#include <hip/hip_runtime.h>

struct gdn_k1_neumann_c_kargs {
    const void* __restrict__ ptr_k;       // bf16 [B, T, H, 128]
    const void* __restrict__ ptr_g;       // fp32 [B, T, H]
    const void* __restrict__ ptr_beta;    // fp32 [B, T, H]
    void* __restrict__ ptr_c;             // bf16 [B, T, H, 64]
    void* __restrict__ ptr_g_cumsum;      // fp32 [B, T, H]
    int B;
    int T;
    int H;
    int NT;                               // ceil(T / 64), per batch
};

extern "C" __global__ void
gdn_k1_neumann_c_kernel(gdn_k1_neumann_c_kargs kargs);

#ifdef __HIP_DEVICE_COMPILE__

#include "opus_gdn/gdn_mfma_utils.h"

extern "C" __global__ void __launch_bounds__(256, 3)
gdn_k1_neumann_c_kernel(gdn_k1_neumann_c_kargs kargs) {
    using namespace gdn_mfma;

    using D_ATTN = opus::bf16_t;
    using D_ACC = float;
    using v8bf16_t = __bf16 __attribute__((ext_vector_type(8)));

    constexpr int BT = 64;
    constexpr int K = 128;
    constexpr int BS = 256;
    constexpr int WARP_SIZE = 64;
    constexpr int PAD = 4;
    constexpr int K_STRIDE = K + PAD;
    constexpr int A_STRIDE = BT + 1;

    // Keep all exits uniform because the kernel contains block barriers.
    if (kargs.B <= 0 || kargs.T <= 0 || kargs.H <= 0 || kargs.NT <= 0)
        return;
    if (static_cast<int>(blockIdx.x) >= kargs.NT)
        return;

    const int i_t = static_cast<int>(blockIdx.x);
    const int i_bh = static_cast<int>(blockIdx.y);
    const int i_b = i_bh / kargs.H;
    const int i_h = i_bh % kargs.H;
    if (i_b >= kargs.B)
        return;

    const int chunk_start = i_t * BT;
    if (chunk_start >= kargs.T)
        return;

    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int64_t global_token_base =
        static_cast<int64_t>(i_b) * kargs.T + chunk_start;
    const int64_t global_head_base =
        global_token_base * kargs.H + i_h;

    // Dynamic LDS supplied by the host is the existing BT64 K1 trait size
    // (18176 bytes). The live data below peaks at the phase-1 K allocation:
    //   s_g + s_beta:       64 * 2 * fp32       =   512 B
    //   s_k:                64 * (128+4) * bf16 = 16896 B
    // Phase 2 aliases s_A onto s_k.
    extern __shared__ char smem_buf[];
    D_ACC* s_g = reinterpret_cast<D_ACC*>(smem_buf);
    D_ACC* s_beta = s_g + BT;
    D_ATTN* s_k = reinterpret_cast<D_ATTN*>(s_beta + BT);
    D_ACC* s_A = reinterpret_cast<D_ACC*>(s_k);

    // ---------------------------------------------------------------------
    // Phase 1a: beta load and chunk-local inclusive prefix sum of g.
    // Padded tail rows contribute zero to the scan.
    // ---------------------------------------------------------------------
    const D_ACC* g_base =
        reinterpret_cast<const D_ACC*>(kargs.ptr_g)
        + global_head_base;
    const D_ACC* beta_base =
        reinterpret_cast<const D_ACC*>(kargs.ptr_beta)
        + global_head_base;

    for (int row = tid; row < BT; row += BS) {
        const int token = chunk_start + row;
        s_beta[row] = token < kargs.T ? beta_base[row * kargs.H] : 0.0f;
    }

    if (warp_id == 0) {
        const int token = chunk_start + lane_id;
        float value = token < kargs.T ? g_base[lane_id * kargs.H] : 0.0f;
#pragma unroll
        for (int offset = 1; offset < BT; offset <<= 1) {
            const float upper = __shfl_up(value, offset, BT);
            if (lane_id >= offset)
                value += upper;
        }
        s_g[lane_id] = value;
        // Only wave 0 publishes and consumes the prefix before the K-load
        // barrier below.  A CTA-wide barrier here stalls the other three
        // waves even though their next writes target the disjoint s_k pool.
        __syncwarp();
    }

    D_ACC* g_cumsum_base =
        reinterpret_cast<D_ACC*>(kargs.ptr_g_cumsum)
        + global_head_base;
    for (int row = tid; row < BT; row += BS) {
        const int token = chunk_start + row;
        if (token < kargs.T)
            g_cumsum_base[row * kargs.H] = s_g[row];
    }

    // ---------------------------------------------------------------------
    // Phase 1b: vectorized, tail-masked K load.
    // ---------------------------------------------------------------------
    const D_ATTN* k_base =
        reinterpret_cast<const D_ATTN*>(kargs.ptr_k)
        + global_head_base * K;
    constexpr int K_VECS_PER_ROW = K / 8;
    for (int i = tid; i < BT * K_VECS_PER_ROW; i += BS) {
        const int row = i / K_VECS_PER_ROW;
        const int col = (i % K_VECS_PER_ROW) * 8;
        const int token = chunk_start + row;
        v8bf16_t value{};
        if (token < kargs.T) {
            value = *reinterpret_cast<const v8bf16_t*>(
                &k_base[row * kargs.H * K + col]);
        }
        *reinterpret_cast<v8bf16_t*>(&s_k[row * K_STRIDE + col]) =
            value;
    }
    __syncthreads();

    // ---------------------------------------------------------------------
    // Phase 1c/1d: K K^T followed by beta/gate scaling. Each wave owns
    // sixteen output rows and covers all four 16-column tiles.
    // ---------------------------------------------------------------------
    constexpr int KKT_TILES_PER_WAVE = 3;
    constexpr int KKT_E_K = K / 16;
    v4f32_t kkt[KKT_TILES_PER_WAVE];
    clear_v4f32<KKT_TILES_PER_WAVE>(kkt);

    // Only ten of the sixteen 16x16 KKT tiles feed the strictly-lower A.
    // Keep each diagonal tile on its owning wave (so the inverse below needs
    // no cross-wave hand-off), then distribute the six off-diagonal tiles as
    // 2/2/1/1.  The resulting 3/3/2/2 MFMA schedule shortens the critical
    // wave from four tiles to three instead of merely idling the upper waves.
    #pragma unroll
    for (int slot = 0; slot < KKT_TILES_PER_WAVE; ++slot) {
        const bool active = slot < 2 || warp_id < 2;
        if (active) {
            int tile_row;
            int tile_col;
            if (slot == 0) {
                tile_row = warp_id;
                tile_col = warp_id;
            } else if (slot == 1) {
                tile_row = warp_id < 2 ? warp_id + 1 : warp_id;
                tile_col = warp_id > 1 ? warp_id - 1 : 0;
            } else {
                tile_row = 3;
                tile_col = warp_id;
            }
            #pragma unroll
            for (int ek = 0; ek < KKT_E_K; ++ek) {
                const v4bf16_t a_tile = load_mfma_tile(
                    s_k, tile_row * 16, ek * 16, K_STRIDE, lane_id);
                const v4bf16_t b_tile = load_mfma_tile(
                    s_k, tile_col * 16, ek * 16, K_STRIDE, lane_id);
                kkt[slot] = mfma_f32_16x16x16_bf16(
                    a_tile, b_tile, kkt[slot]);
            }
        }
    }

    // All waves have finished reading s_k before s_A aliases that region.
    __syncthreads();

    #pragma unroll
    for (int slot = 0; slot < KKT_TILES_PER_WAVE; ++slot) {
        const bool active = slot < 2 || warp_id < 2;
        if (!active)
            continue;
        int tile_row;
        int tile_col;
        if (slot == 0) {
            tile_row = warp_id;
            tile_col = warp_id;
        } else if (slot == 1) {
            tile_row = warp_id < 2 ? warp_id + 1 : warp_id;
            tile_col = warp_id > 1 ? warp_id - 1 : 0;
        } else {
            tile_row = 3;
            tile_col = warp_id;
        }
        for (int p = 0; p < 4; ++p) {
            const int row =
                tile_row * 16 + (lane_id >> 4) * 4 + p;
            const int col = tile_col * 16 + (lane_id & 15);
            float value = 0.0f;
            if (row > col) {
                value = kkt[slot][p] * s_beta[row]
                    * __expf(s_g[row] - s_g[col]);
            }
            s_A[row * A_STRIDE + col] = value;
        }
    }

    // The six block-upper tiles are known zero and were deliberately not fed
    // through MFMA.  Transpose the off-diagonal ownership above to clear them
    // with balanced 2/2/1/1 wave-local stores.
    constexpr v4f32_t zero4 = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int slot = 1; slot < KKT_TILES_PER_WAVE; ++slot) {
        const bool active = slot < 2 || warp_id < 2;
        if (active) {
            int lower_row;
            int lower_col;
            if (slot == 1) {
                lower_row = warp_id < 2 ? warp_id + 1 : warp_id;
                lower_col = warp_id > 1 ? warp_id - 1 : 0;
            } else {
                lower_row = 3;
                lower_col = warp_id;
            }
            store_fp32_tile(
                s_A, lower_col * 16, lower_row * 16, A_STRIDE,
                zero4, lane_id);
        }
    }
    __syncwarp();

    // ---------------------------------------------------------------------
    // Phase 2a: invert each 16x16 diagonal block with the exact finite
    // Neumann factorization for a strictly-lower matrix B=-A:
    //   (I+A)^-1 = (I+B)(I+B^2)(I+B^4)(I+B^8).
    // BF16 MFMA operands match the existing Opus Neumann implementation.
    // ---------------------------------------------------------------------
    {
        const int block_row = warp_id * 16;
        const int base =
            (block_row + (lane_id & 15)) * A_STRIDE
            + block_row + ((lane_id >> 4) << 2);
        const v4bf16_t neg_a = {
            static_cast<__bf16>(-s_A[base]),
            static_cast<__bf16>(-s_A[base + 1]),
            static_cast<__bf16>(-s_A[base + 2]),
            static_cast<__bf16>(-s_A[base + 3])};

        const int n = lane_id & 15;
        const int m = (lane_id >> 4) * 4;
        const v4f32_t identity = {
            m == n ? 1.0f : 0.0f,
            m + 1 == n ? 1.0f : 0.0f,
            m + 2 == n ? 1.0f : 0.0f,
            m + 3 == n ? 1.0f : 0.0f};

        const v4f32_t b2 =
            mfma_f32_16x16x16_bf16(neg_a, neg_a, zero4);
        const v4bf16_t b2_src = accum_to_src(b2);
        const v4f32_t b4 =
            mfma_f32_16x16x16_bf16(b2_src, b2_src, zero4);
        const v4bf16_t b4_src = accum_to_src(b4);
        const v4f32_t b8 =
            mfma_f32_16x16x16_bf16(b4_src, b4_src, zero4);

        v4f32_t c_diag;
#pragma unroll
        for (int p = 0; p < 4; ++p)
            c_diag[p] = b8[p] + identity[p];
        c_diag = mfma_f32_16x16x16_bf16(
            b4_src, accum_to_src(c_diag), c_diag);
        c_diag = mfma_f32_16x16x16_bf16(
            b2_src, accum_to_src(c_diag), c_diag);
        c_diag = mfma_f32_16x16x16_bf16(
            neg_a, accum_to_src(c_diag), c_diag);

        store_fp32_tile(
            s_A, block_row, block_row, A_STRIDE, c_diag, lane_id);
        for (int i = lane_id; i < 16 * 16; i += WARP_SIZE) {
            const int row = i / 16;
            const int col = i % 16;
            if (row < col) {
                s_A[(block_row + row) * A_STRIDE
                    + block_row + col] = 0.0f;
            }
        }
    }
    __syncthreads();

    // ---------------------------------------------------------------------
    // Phase 2b: merge the four diagonal inverses through the lower-block
    // Schur-complement dependency DAG.
    // ---------------------------------------------------------------------
    v4bf16_t saved_l32;
    v4bf16_t saved_l43;
    v4bf16_t saved_l42;
    if (warp_id == 0) {
        saved_l32 =
            load_fp32_tile(s_A, 32, 16, A_STRIDE, lane_id);
        saved_l43 =
            load_fp32_tile(s_A, 48, 32, A_STRIDE, lane_id);
        saved_l42 =
            load_fp32_tile(s_A, 48, 16, A_STRIDE, lane_id);
    } else if (warp_id == 1) {
        saved_l43 =
            load_fp32_tile(s_A, 48, 32, A_STRIDE, lane_id);
    }

    v4f32_t kept_c21 = zero4;
    v4f32_t kept_c32 = zero4;
    v4f32_t kept_c31 = zero4;

    // Level 1: C21, C32, C43.
    if (warp_id == 0) {
        v4f32_t t = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 16, 0, A_STRIDE, lane_id),
            load_fp32_tile_T(s_A, 0, 0, A_STRIDE, lane_id),
            zero4);
        kept_c21 = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 16, 16, A_STRIDE, lane_id),
            accum_to_src(t),
            zero4);
#pragma unroll
        for (int p = 0; p < 4; ++p)
            kept_c21[p] = -kept_c21[p];
        store_fp32_tile(
            s_A, 16, 0, A_STRIDE, kept_c21, lane_id);
    } else if (warp_id == 1) {
        v4f32_t t = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 32, 16, A_STRIDE, lane_id),
            load_fp32_tile_T(s_A, 16, 16, A_STRIDE, lane_id),
            zero4);
        kept_c32 = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 32, 32, A_STRIDE, lane_id),
            accum_to_src(t),
            zero4);
#pragma unroll
        for (int p = 0; p < 4; ++p)
            kept_c32[p] = -kept_c32[p];
        store_fp32_tile(
            s_A, 32, 16, A_STRIDE, kept_c32, lane_id);
    } else if (warp_id == 2) {
        v4f32_t t = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 48, 32, A_STRIDE, lane_id),
            load_fp32_tile_T(s_A, 32, 32, A_STRIDE, lane_id),
            zero4);
        v4f32_t c43 = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 48, 48, A_STRIDE, lane_id),
            accum_to_src(t),
            zero4);
#pragma unroll
        for (int p = 0; p < 4; ++p)
            c43[p] = -c43[p];
        store_fp32_tile(s_A, 48, 32, A_STRIDE, c43, lane_id);
    }
    // Level 2: C31, C42.
    if (warp_id == 0) {
        v4f32_t t = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 32, 0, A_STRIDE, lane_id),
            load_fp32_tile_T(s_A, 0, 0, A_STRIDE, lane_id),
            zero4);
        t = mfma_f32_16x16x16_bf16(
            saved_l32, accum_to_src(kept_c21), t);
        kept_c31 = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 32, 32, A_STRIDE, lane_id),
            accum_to_src(t),
            zero4);
#pragma unroll
        for (int p = 0; p < 4; ++p)
            kept_c31[p] = -kept_c31[p];
        store_fp32_tile(
            s_A, 32, 0, A_STRIDE, kept_c31, lane_id);
    } else if (warp_id == 1) {
        v4f32_t t = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 48, 16, A_STRIDE, lane_id),
            load_fp32_tile_T(s_A, 16, 16, A_STRIDE, lane_id),
            zero4);
        t = mfma_f32_16x16x16_bf16(
            saved_l43, accum_to_src(kept_c32), t);
        v4f32_t c42 = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 48, 48, A_STRIDE, lane_id),
            accum_to_src(t),
            zero4);
#pragma unroll
        for (int p = 0; p < 4; ++p)
            c42[p] = -c42[p];
        store_fp32_tile(s_A, 48, 16, A_STRIDE, c42, lane_id);
    }
    // Level 3: C41.
    if (warp_id == 0) {
        v4f32_t t = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 48, 0, A_STRIDE, lane_id),
            load_fp32_tile_T(s_A, 0, 0, A_STRIDE, lane_id),
            zero4);
        t = mfma_f32_16x16x16_bf16(
            saved_l42, accum_to_src(kept_c21), t);
        t = mfma_f32_16x16x16_bf16(
            saved_l43, accum_to_src(kept_c31), t);
        v4f32_t c41 = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 48, 48, A_STRIDE, lane_id),
            accum_to_src(t),
            zero4);
#pragma unroll
        for (int p = 0; p < 4; ++p)
            c41[p] = -c41[p];
        store_fp32_tile(s_A, 48, 0, A_STRIDE, c41, lane_id);
    }
    __syncthreads();

    // ---------------------------------------------------------------------
    // C-only epilogue. There are 512 bf16x8 vectors, so each of the 256
    // threads performs two aligned 16-byte stores. Invalid tail rows do not
    // write; every valid row writes all 64 columns.
    // ---------------------------------------------------------------------
    D_ATTN* c_base =
        reinterpret_cast<D_ATTN*>(kargs.ptr_c)
        + global_head_base * BT;
    constexpr int C_VECS_PER_ROW = BT / 8;
    for (int i = tid; i < BT * C_VECS_PER_ROW; i += BS) {
        const int row = i / C_VECS_PER_ROW;
        const int col = (i % C_VECS_PER_ROW) * 8;
        if (chunk_start + row < kargs.T) {
            const int offset = row * A_STRIDE + col;
            const v8bf16_t value = {
                static_cast<__bf16>(s_A[offset]),
                static_cast<__bf16>(s_A[offset + 1]),
                static_cast<__bf16>(s_A[offset + 2]),
                static_cast<__bf16>(s_A[offset + 3]),
                static_cast<__bf16>(s_A[offset + 4]),
                static_cast<__bf16>(s_A[offset + 5]),
                static_cast<__bf16>(s_A[offset + 6]),
                static_cast<__bf16>(s_A[offset + 7])};
            *reinterpret_cast<v8bf16_t*>(
                &c_base[row * kargs.H * BT + col]) = value;
        }
    }
}

#endif  // __HIP_DEVICE_COMPILE__
