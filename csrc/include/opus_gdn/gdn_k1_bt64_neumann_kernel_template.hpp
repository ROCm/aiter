// GDN Prefill K1 Kernel — BT=64, MFMA Neumann variant
// Phase 2a uses MFMA Neumann series instead of scalar forward substitution.
//
// Grid: (NT, B*H)   Block: (BLOCK_SIZE = 256)
// Target: gfx942 (MI300X) / gfx950 (MI350), MFMA bf16 16×16×16
#pragma once

#include "opus_gdn/gdn_defs.h"
#include "opus_gdn/gdn_mfma_utils.h"

// Arch-specialized occupancy: gfx942 (MI300) is LDS-bound, so register-caching
// C_inv frees LDS and lifts OCC 2→3 (-30% K1). gfx950 (MI350) has 160KB LDS —
// not the limiter — so OCC=3 is a no-op there; keep the simpler LDS path at OCC=2.
#if defined(__gfx950__)
#define GDN_K1_NEUMANN_OCC 2
#else
#define GDN_K1_NEUMANN_OCC 3
#endif
template<typename Traits>
__global__ void __launch_bounds__(Traits::BLOCK_SIZE, GDN_K1_NEUMANN_OCC)
gdn_k1_neumann_kernel(gdn_k1_kargs kargs) {
    using namespace gdn_mfma;
    using T = Traits;
    using D_ATTN = typename T::D_ATTN;
    using D_ACC  = typename T::D_ACC;

    static_assert(T::BT == 64, "This template is for BT=64 only");

    constexpr bool IS_VARLEN = T::IS_VARLEN;
    const int i_chunk = static_cast<int>(blockIdx.x);
    int i_t, i_b, i_h;
    if constexpr (IS_VARLEN) {
        i_b = kargs.ptr_chunk_indices[2 * i_chunk];
        i_t = kargs.ptr_chunk_indices[2 * i_chunk + 1];
        const int64_t i_h_flat =
            static_cast<int64_t>(blockIdx.z) * gridDim.y + blockIdx.y;
        if (i_h_flat >= kargs.H) {
            return;
        }
        i_h = static_cast<int>(i_h_flat);
        if (i_b < 0 || i_b >= kargs.B || i_t < 0 ||
            i_h < 0 || i_h >= kargs.H) {
            return;
        }
    } else {
        const int i_bh = static_cast<int>(blockIdx.y);
        i_t = i_chunk;
        i_b = i_bh / kargs.H;
        i_h = i_bh % kargs.H;
    }

    const int tid  = threadIdx.x;
    const int warp_id = tid / T::WARP_SIZE;
    const int lane_id = tid % T::WARP_SIZE;

    constexpr int BT = T::BT;
    constexpr int BS = T::BLOCK_SIZE;
    constexpr int PAD = T::SMEM_PAD;
    constexpr int K_STRIDE = T::K_STRIDE;
    constexpr int BK_SUB = T::BK_SUB;
    constexpr int A_STRIDE = T::A_STRIDE;
    const int K  = kargs.K;
    const int V  = kargs.V;
    const int H  = kargs.H;

    int bos, seq_len, chunk_start;
    if constexpr (IS_VARLEN) {
        bos = kargs.ptr_cu_seqlens[i_b];
        const int eos = kargs.ptr_cu_seqlens[i_b + 1];
        if (bos < 0 || eos <= bos || eos > kargs.T) {
            return;
        }
        seq_len = eos - bos;
        const int num_chunks = ceil_div(seq_len, BT);
        if (i_t >= num_chunks) {
            return;
        }
        chunk_start = i_t * BT;
    } else {
        chunk_start = i_t * BT;
        bos = i_b * kargs.T;
        seq_len = kargs.T;
    }
    const int64_t global_token_base =
        static_cast<int64_t>(bos) + chunk_start;
    const int64_t global_head_base = global_token_base * H + i_h;

    // =====================================================================
    // Shared memory allocation
    //
    // Phase 1:  s_g[BT] + s_beta[BT] + s_k[BT×K_STRIDE]     = 17408 bytes
    // Phase 2a: s_g + s_beta + s_A[BT×A_STRIDE] (aliases s_k) = padded bytes
    // Phase 2c: s_g + s_beta + s_vT[BK_SUB×VT_STRIDE]        =  9216 bytes
    //           (C_inv cached in registers, no s_C_bf16)
    // Peak: 17408 bytes → fits 3 blocks/CU (17408×3 = 52224 < 65536)
    // =====================================================================
    extern __shared__ char smem_buf[];

    D_ACC*  s_g    = reinterpret_cast<D_ACC*>(smem_buf);
    D_ACC*  s_beta = s_g + BT;
    D_ATTN* s_k    = reinterpret_cast<D_ATTN*>(s_beta + BT);

    D_ACC*  s_A    = reinterpret_cast<D_ACC*>(s_k);

    // =====================================================================
    // Phase 1a: Load g and beta, compute prefix sum
    // =====================================================================
    const D_ACC* g_base_fp32 = reinterpret_cast<const D_ACC*>(kargs.ptr_g)
                                + global_head_base;
    const D_ATTN* g_base_bf16 = reinterpret_cast<const D_ATTN*>(kargs.ptr_g)
                                 + global_head_base;
    const D_ACC* beta_base_fp32 =
        reinterpret_cast<const D_ACC*>(kargs.ptr_beta) + global_head_base;
    const D_ATTN* beta_base_bf16 =
        reinterpret_cast<const D_ATTN*>(kargs.ptr_beta) + global_head_base;

    // Load beta (no scan needed).
    for (int i = tid; i < BT; i += BS) {
        int global_t = chunk_start + i;
        if (global_t < seq_len) {
            if constexpr (T::DYNAMIC_SCALARS) {
                s_beta[i] = kargs.beta_is_bf16
                    ? static_cast<float>(beta_base_bf16[i * H])
                    : beta_base_fp32[i * H];
            } else {
                s_beta[i] = beta_base_fp32[i * H];
            }
        } else {
            s_beta[i] = 0.0f;
        }
    }

    // Inclusive prefix sum of g in a SINGLE warp via __shfl_up (BT == WARP_SIZE
    // = 64): warp-lockstep register exchange — no LDS round-trip, no
    // __syncthreads. Replaces the block Hillis-Steele's log2(BT)=6 barriers with
    // one (the publish below). Bit-identical: same stride-doubling order, just
    // registers instead of LDS. K1 is barrier/latency-bound on gfx950, so this
    // removes ~6 of its runtime barriers.
    static_assert(BT == T::WARP_SIZE, "single-warp scan requires BT == WARP_SIZE");
    if (warp_id == 0) {
        int global_t = chunk_start + lane_id;
        float val = 0.0f;
        if (global_t < seq_len) {
            if constexpr (T::DYNAMIC_SCALARS) {
                val = kargs.g_is_bf16
                    ? static_cast<float>(g_base_bf16[lane_id * H])
                    : g_base_fp32[lane_id * H];
            } else {
                val = g_base_fp32[lane_id * H];
            }
        }
        #pragma unroll
        for (int off = 1; off < BT; off <<= 1) {
            float up = __shfl_up(val, off, BT);
            if (lane_id >= off) val += up;
        }
        s_g[lane_id] = val;
        // The prefix producer is also the only wave that writes it to HBM.
        // The K publication barrier below provides the first CTA-wide point
        // at which the other waves consume s_g/s_beta.
        __syncwarp();
    }

    // Write g_cumsum to HBM
    D_ACC* g_cumsum_base = reinterpret_cast<D_ACC*>(kargs.ptr_g_cumsum)
                           + global_head_base;
    for (int i = tid; i < BT; i += BS) {
        int global_t = chunk_start + i;
        if (global_t < seq_len)
            g_cumsum_base[i * H] = s_g[i];
    }

    // =====================================================================
    // Phase 1b: Load k[BT, K] into LDS with padding (stride = K_STRIDE)
    // =====================================================================
    const D_ATTN* k_base = reinterpret_cast<const D_ATTN*>(kargs.ptr_k)
                           + global_head_base * K;

    {
        // v8bf16_t (128-bit) vectorized k load — half the VMEM load instructions
        // of v4 (borrowed from the BT=32 kernel). K=128 contiguous per (b,t,h).
        using v8bf16_t = __bf16 __attribute__((ext_vector_type(8)));
        constexpr int K_VEC = T::K / 8;
        for (int i = tid; i < BT * K_VEC; i += BS) {
            int row = i / K_VEC;
            int col8 = (i % K_VEC) * 8;
            int global_t = chunk_start + row;
            v8bf16_t val{};
            if (global_t < seq_len)
                val = *reinterpret_cast<const v8bf16_t*>(
                    &k_base[row * H * K + col8]);
            *reinterpret_cast<v8bf16_t*>(&s_k[row * K_STRIDE + col8]) = val;
        }
    }
    __syncthreads();

    // =====================================================================
    // Phase 1c+1d: KKT GEMM via MFMA — k × k^T self-matmul
    // =====================================================================

    constexpr int KKT_TILES_PER_WAVE = 3;
    constexpr int KKT_E_K = T::K / 16;     // 8

    v4f32_t kkt_c[KKT_TILES_PER_WAVE];
    clear_v4f32<KKT_TILES_PER_WAVE>(kkt_c);

    // Compute only the ten block-lower KKT tiles.  Diagonals stay on their
    // owning waves; off-diagonals are distributed 2/2/1/1, producing a
    // balanced 3/3/2/2 schedule instead of leaving one four-tile critical
    // wave after a naive triangular prune.
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
                kkt_c[slot] = mfma_f32_16x16x16_bf16(
                    a_tile, b_tile, kkt_c[slot]);
            }
        }
    }

    // No wave may overwrite the aliased s_k/s_A pool while another wave is
    // still issuing MFMA operand reads.
    __syncthreads();

    // Post-MFMA: gate-scale lower triangle, zero upper+diagonal, write fp32 to s_A
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
        for (int p = 0; p < 4; p++) {
            int s = tile_row * 16 + (lane_id >> 4) * 4 + p;
            int r = tile_col * 16 + (lane_id & 15);
            float val = 0.0f;
            if (s > r)
                val = kkt_c[slot][p] * s_beta[s] * __expf(s_g[s] - s_g[r]);
            s_A[s * A_STRIDE + r] = val;
        }
    }

    constexpr v4f32_t z4 = {0.f, 0.f, 0.f, 0.f};
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
                z4, lane_id);
        }
    }
    __syncwarp();

    // =====================================================================
    // Phase 2a: Triangular inverse — MFMA Neumann series
    //
    // (I+A)^{-1} = sum_{n=0}^{15} (-A)^n (exact: A is 16×16 strictly lower
    // triangular, hence nilpotent with (-A)^16 = 0).
    //
    // Each warp handles one 16×16 diagonal block independently.
    // 15 MFMA iterations per block via register chaining (accum_to_src).
    // =====================================================================

    {
        int br = warp_id * 16;

        // Load (-A) as MFMA A operand
        v4bf16_t neg_A_tile;
        {
            int base = (br + (lane_id & 15)) * A_STRIDE + br + ((lane_id >> 4) << 2);
            neg_A_tile = v4bf16_t{
                static_cast<__bf16>(-s_A[base]),
                static_cast<__bf16>(-s_A[base + 1]),
                static_cast<__bf16>(-s_A[base + 2]),
                static_cast<__bf16>(-s_A[base + 3])};
        }

        // Horner's method: (I+A)^{-1} = I + (-A)(I + (-A)(I + ...))
        v4f32_t I_accum;
        {
            int n = lane_id & 15;
            int m_base = (lane_id >> 4) * 4;
            I_accum = v4f32_t{
                (m_base == n) ? 1.0f : 0.0f,
                ((m_base + 1) == n) ? 1.0f : 0.0f,
                ((m_base + 2) == n) ? 1.0f : 0.0f,
                ((m_base + 3) == n) ? 1.0f : 0.0f};
        }

        // Triangular inverse via squaring: with B = -A (16×16 strictly-lower,
        // nilpotent B^16=0), (I+A)^{-1} = sum_{n=0}^{15} B^n
        //   = (I+B)(I+B^2)(I+B^4)(I+B^8).
        // 6 MFMAs (3 squarings + 3 products; I+B^8 is a free fp32 acc-add)
        // vs 15 Horner iterations. Factors commute (all polynomials in B).
        v4f32_t  b2   = mfma_f32_16x16x16_bf16(neg_A_tile, neg_A_tile, z4);  // B^2
        v4bf16_t b2_o = accum_to_src(b2);
        v4f32_t  b4   = mfma_f32_16x16x16_bf16(b2_o, b2_o, z4);              // B^4
        v4bf16_t b4_o = accum_to_src(b4);
        v4f32_t  b8   = mfma_f32_16x16x16_bf16(b4_o, b4_o, z4);             // B^8
        v4bf16_t b8_o = accum_to_src(b8);
        v4f32_t C_accum;
        for (int n = 0; n < 4; n++) C_accum[n] = b8[n] + I_accum[n];        // I + B^8
        C_accum = mfma_f32_16x16x16_bf16(b4_o, accum_to_src(C_accum), C_accum);        // (I+B^4)·R
        C_accum = mfma_f32_16x16x16_bf16(b2_o, accum_to_src(C_accum), C_accum);        // (I+B^2)·R
        C_accum = mfma_f32_16x16x16_bf16(neg_A_tile, accum_to_src(C_accum), C_accum);  // (I+B)·R

        // Store result and zero upper triangle
        store_fp32_tile(s_A, br, br, A_STRIDE, C_accum, lane_id);
        for (int idx = lane_id; idx < 16 * 16; idx += T::WARP_SIZE) {
            int r = idx / 16, c = idx % 16;
            if (r < c)
                s_A[(br + r) * A_STRIDE + br + c] = 0.0f;
        }
    }
    __syncthreads();

    // =====================================================================
    // Phase 2b: Schur complement merge — MFMA bf16 16×16×16
    //
    // Dependency DAG enables warp-level parallelism (3 levels, 3 barriers):
    //   Level 1: C_21, C_32, C_43  (3 warps, independent)
    //   Level 2: C_31, C_42        (2 warps, independent)
    //   Level 3: C_41              (1 warp)
    // =====================================================================

    // Pre-save L blocks overwritten in Level 1
    v4bf16_t sav_L32, sav_L43, sav_L42;
    if (warp_id == 0) {
        sav_L32 = load_fp32_tile(s_A, 32, 16, A_STRIDE, lane_id);
        sav_L43 = load_fp32_tile(s_A, 48, 32, A_STRIDE, lane_id);
        sav_L42 = load_fp32_tile(s_A, 48, 16, A_STRIDE, lane_id);
    } else if (warp_id == 1) {
        sav_L43 = load_fp32_tile(s_A, 48, 32, A_STRIDE, lane_id);
    }

    v4f32_t kept_c21 = z4, kept_c32 = z4, kept_c31 = z4;

    // --- Level 1: C_21, C_32, C_43 ---
    if (warp_id == 0) {
        v4f32_t t = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 16, 0, A_STRIDE, lane_id),
            load_fp32_tile_T(s_A, 0, 0, A_STRIDE, lane_id), z4);
        kept_c21 = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 16, 16, A_STRIDE, lane_id),
            accum_to_src(t), z4);
        for (int p = 0; p < 4; p++) kept_c21[p] = -kept_c21[p];
        store_fp32_tile(s_A, 16, 0, A_STRIDE, kept_c21, lane_id);
    } else if (warp_id == 1) {
        v4f32_t t = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 32, 16, A_STRIDE, lane_id),
            load_fp32_tile_T(s_A, 16, 16, A_STRIDE, lane_id), z4);
        kept_c32 = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 32, 32, A_STRIDE, lane_id),
            accum_to_src(t), z4);
        for (int p = 0; p < 4; p++) kept_c32[p] = -kept_c32[p];
        store_fp32_tile(s_A, 32, 16, A_STRIDE, kept_c32, lane_id);
    } else if (warp_id == 2) {
        v4f32_t t = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 48, 32, A_STRIDE, lane_id),
            load_fp32_tile_T(s_A, 32, 32, A_STRIDE, lane_id), z4);
        v4f32_t c43 = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 48, 48, A_STRIDE, lane_id),
            accum_to_src(t), z4);
        for (int p = 0; p < 4; p++) c43[p] = -c43[p];
        store_fp32_tile(s_A, 48, 32, A_STRIDE, c43, lane_id);
    }
    // --- Level 2: C_31, C_42 ---
    if (warp_id == 0) {
        v4f32_t t = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 32, 0, A_STRIDE, lane_id),
            load_fp32_tile_T(s_A, 0, 0, A_STRIDE, lane_id), z4);
        t = mfma_f32_16x16x16_bf16(sav_L32, accum_to_src(kept_c21), t);
        kept_c31 = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 32, 32, A_STRIDE, lane_id),
            accum_to_src(t), z4);
        for (int p = 0; p < 4; p++) kept_c31[p] = -kept_c31[p];
        store_fp32_tile(s_A, 32, 0, A_STRIDE, kept_c31, lane_id);
    } else if (warp_id == 1) {
        v4f32_t t = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 48, 16, A_STRIDE, lane_id),
            load_fp32_tile_T(s_A, 16, 16, A_STRIDE, lane_id), z4);
        t = mfma_f32_16x16x16_bf16(sav_L43, accum_to_src(kept_c32), t);
        v4f32_t c42 = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 48, 48, A_STRIDE, lane_id),
            accum_to_src(t), z4);
        for (int p = 0; p < 4; p++) c42[p] = -c42[p];
        store_fp32_tile(s_A, 48, 16, A_STRIDE, c42, lane_id);
    }
    // --- Level 3: C_41 ---
    if (warp_id == 0) {
        v4f32_t t = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 48, 0, A_STRIDE, lane_id),
            load_fp32_tile_T(s_A, 0, 0, A_STRIDE, lane_id), z4);
        t = mfma_f32_16x16x16_bf16(sav_L42, accum_to_src(kept_c21), t);
        t = mfma_f32_16x16x16_bf16(sav_L43, accum_to_src(kept_c31), t);
        v4f32_t c41 = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 48, 48, A_STRIDE, lane_id),
            accum_to_src(t), z4);
        for (int p = 0; p < 4; p++) c41[p] = -c41[p];
        store_fp32_tile(s_A, 48, 0, A_STRIDE, c41, lane_id);
    }
    __syncthreads();

    // s_A now contains C = (I + A)^{-1}

    // =====================================================================
    // Phase 2c: WY factor GEMMs via MFMA
    //
    // u_bar = C @ (v * beta)
    // w_bar = C @ (k * beta * exp(g_cumsum))
    //
    // Step 1: Convert C (fp32, s_A) → C_bf16 (placed after s_A in LDS)
    // Step 2: For each subtile, load pre-scaled v/k transposed → MFMA
    // =====================================================================

#if defined(__gfx950__)
    // gfx950 (OCC=2): stage C_inv in LDS as bf16 (placed after s_A). 160KB LDS
    // is not the occupancy limiter here, so register-caching gains nothing.
    constexpr int C_STRIDE = BT + PAD;  // 68
    D_ATTN* s_C_bf16 = reinterpret_cast<D_ATTN*>(
        smem_buf + BT * 2 * sizeof(D_ACC) + BT * A_STRIDE * sizeof(D_ACC));
    for (int i = tid; i < BT * BT; i += BS) {
        int s = i / BT;
        int j = i % BT;
        s_C_bf16[s * C_STRIDE + j] = static_cast<D_ATTN>(s_A[s * A_STRIDE + j]);
    }
    __syncthreads();
    constexpr int WY_EM = 1;
#else
    // gfx942 (OCC=3): cache C_inv tiles in registers (eliminates s_C_bf16 from
    // LDS → 25856→18176 bytes → OCC 2→3, -30% K1 on the LDS-bound MI300).
    constexpr int WY_EK_C = BT / 16;  // 4
    v4bf16_t cached_C[WY_EK_C];
    for (int ek = 0; ek < WY_EK_C; ek++)
        cached_C[ek] = load_fp32_tile(s_A, warp_id * 16, ek * 16, A_STRIDE, lane_id);
#endif

    // s_A region freed — reuse for v_scaled_T / k_scaled_T
    constexpr int VT_STRIDE = BT + PAD;  // 68
    D_ATTN* s_vT = s_k;

    constexpr int WY_EN = BK_SUB / 16;   // 4
    constexpr int WY_EK = BT / 16;       // 4

    D_ATTN* w_bar_base = reinterpret_cast<D_ATTN*>(kargs.ptr_w_bar)
                         + global_head_base * K;
    D_ATTN* u_bar_base = reinterpret_cast<D_ATTN*>(kargs.ptr_u_bar)
                         + global_head_base * V;

    // --- u_bar = C @ (v * beta) ---
    const D_ATTN* v_base = reinterpret_cast<const D_ATTN*>(kargs.ptr_v)
                           + global_head_base * V;

    for (int iv = 0; iv < T::N_V_ITERS; iv++) {
        int v_offset = iv * BK_SUB;

        {
            // v8 (128-bit) HBM load; transposed scaled store stays per-element.
            using v8bf16_t = __bf16 __attribute__((ext_vector_type(8)));
            constexpr int VEC = 8;
            constexpr int NVEC = BK_SUB / VEC;
            for (int i = tid; i < BT * NVEC; i += BS) {
                int j  = i / NVEC;
                int vi = (i % NVEC) * VEC;
                v8bf16_t vals{};
                if (chunk_start + j < seq_len)
                    vals = *reinterpret_cast<const v8bf16_t*>(
                        &v_base[j * H * V + v_offset + vi]);
                D_ACC beta_j = s_beta[j];
                for (int vv = 0; vv < VEC; vv++)
                    s_vT[(vi + vv) * VT_STRIDE + j] = static_cast<D_ATTN>(
                        static_cast<D_ACC>(vals[vv]) * beta_j);
            }
        }
        __syncthreads();

#if defined(__gfx950__)
        v4f32_t wy_c[WY_EM * WY_EN];
        clear_v4f32<WY_EM * WY_EN>(wy_c);
        tiled_gemm_mfma<WY_EM, WY_EN, WY_EK>(
            wy_c, s_C_bf16, warp_id * 16, C_STRIDE,
                  s_vT,     0,             VT_STRIDE, lane_id);
#else
        v4f32_t wy_c[WY_EN];
        clear_v4f32<WY_EN>(wy_c);
        #pragma unroll
        for (int ek = 0; ek < WY_EK; ek++) {
            if (ek > warp_id)
                continue;
            v4bf16_t b_tiles[WY_EN];
            for (int en = 0; en < WY_EN; en++)
                b_tiles[en] = load_mfma_tile(s_vT, en * 16, ek * 16, VT_STRIDE, lane_id);
            for (int en = 0; en < WY_EN; en++)
                wy_c[en] = mfma_f32_16x16x16_bf16(cached_C[ek], b_tiles[en], wy_c[en]);
        }
#endif

        for (int en = 0; en < WY_EN; en++) {
            for (int p = 0; p < 4; p++) {
                int s  = warp_id * 16 + (lane_id >> 4) * 4 + p;
                int vi = en * 16 + (lane_id & 15);
                if (chunk_start + s < seq_len)
                    u_bar_base[s * H * V + v_offset + vi] =
                        static_cast<D_ATTN>(wy_c[en][p]);
            }
        }
        __syncthreads();
    }

    // --- w_bar = C @ (k * beta * exp(g_cumsum)) ---
    for (int ik = 0; ik < T::N_K_ITERS; ik++) {
        int k_offset = ik * BK_SUB;

        {
            // v8 (128-bit) HBM load; transposed scaled store stays per-element.
            using v8bf16_t = __bf16 __attribute__((ext_vector_type(8)));
            constexpr int VEC = 8;
            constexpr int NVEC = BK_SUB / VEC;
            for (int i = tid; i < BT * NVEC; i += BS) {
                int j  = i / NVEC;
                int ki = (i % NVEC) * VEC;
                v8bf16_t vals{};
                if (chunk_start + j < seq_len)
                    vals = *reinterpret_cast<const v8bf16_t*>(
                        &k_base[j * H * K + k_offset + ki]);
                D_ACC scale_j = s_beta[j] * __expf(s_g[j]);
                for (int vv = 0; vv < VEC; vv++)
                    s_vT[(ki + vv) * VT_STRIDE + j] = static_cast<D_ATTN>(
                        static_cast<D_ACC>(vals[vv]) * scale_j);
            }
        }
        __syncthreads();

#if defined(__gfx950__)
        v4f32_t wy_c[WY_EM * WY_EN];
        clear_v4f32<WY_EM * WY_EN>(wy_c);
        tiled_gemm_mfma<WY_EM, WY_EN, WY_EK>(
            wy_c, s_C_bf16, warp_id * 16, C_STRIDE,
                  s_vT,     0,             VT_STRIDE, lane_id);
#else
        v4f32_t wy_c[WY_EN];
        clear_v4f32<WY_EN>(wy_c);
        #pragma unroll
        for (int ek = 0; ek < WY_EK; ek++) {
            if (ek > warp_id)
                continue;
            v4bf16_t b_tiles[WY_EN];
            for (int en = 0; en < WY_EN; en++)
                b_tiles[en] = load_mfma_tile(s_vT, en * 16, ek * 16, VT_STRIDE, lane_id);
            for (int en = 0; en < WY_EN; en++)
                wy_c[en] = mfma_f32_16x16x16_bf16(cached_C[ek], b_tiles[en], wy_c[en]);
        }
#endif

        for (int en = 0; en < WY_EN; en++) {
            for (int p = 0; p < 4; p++) {
                int s  = warp_id * 16 + (lane_id >> 4) * 4 + p;
                int ki = en * 16 + (lane_id & 15);
                if (chunk_start + s < seq_len)
                    w_bar_base[s * H * K + k_offset + ki] =
                        static_cast<D_ATTN>(wy_c[en][p]);
            }
        }
        __syncthreads();
    }

    if (kargs.ptr_k1_done) {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        __syncthreads();
        if (tid == 0) {
            const int64_t done_offset = IS_VARLEN
                ? static_cast<int64_t>(i_chunk) * kargs.H + i_h
                : static_cast<int64_t>(i_t) * kargs.B * kargs.H
                    + i_b * kargs.H + i_h;
            __atomic_store_n(kargs.ptr_k1_done + done_offset,
                             1u, __ATOMIC_RELAXED);
        }
    }
}
