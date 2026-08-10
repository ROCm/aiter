// Gated DeltaNet (GDN) Prefill Kernel — shared types, kargs, traits
// Target: gfx942 (MI300X) / gfx950 (MI350), MFMA bf16 16×16×16
#pragma once

#ifdef __HIP_DEVICE_COMPILE__
using bf16_t = __bf16;
#else
using bf16_t = unsigned short;
#endif

// --------------------------------------------------------------------------
// K1 kernel arguments: Step 1 (cumsum + KKT) + Step 2 (trisol + WY factors)
// --------------------------------------------------------------------------
struct gdn_k1_kargs {
    const void* __restrict__ ptr_k;        // [B, T, H, K]   bf16
    const void* __restrict__ ptr_v;        // [B, T, H, V]   bf16
    const void* __restrict__ ptr_beta;     // [B, T, H]      fp32/bf16
    const void* __restrict__ ptr_g;        // [B, T, H]      fp32/bf16
    void* __restrict__ ptr_w_bar;          // [B, T, H, K]   bf16  output
    void* __restrict__ ptr_u_bar;          // [B, T, H, V]   bf16  output
    void* __restrict__ ptr_g_cumsum;       // [B, T, H]      fp32  output
    uint32_t* __restrict__ ptr_k1_done;    // [NT * B * H] atomic flags (nullable)
    const int32_t* __restrict__ ptr_cu_seqlens;   // [N + 1] (varlen only)
    const int32_t* __restrict__ ptr_chunk_indices; // [total_chunks, 2]
    int g_is_bf16;
    int beta_is_bf16;
    int B;
    int T;
    int H;
    int K;
    int V;
};

// --------------------------------------------------------------------------
// K2 kernel arguments: Step 3 (h update) + Step 4 (output)
// --------------------------------------------------------------------------
struct gdn_k2_kargs {
    const void* __restrict__ ptr_q;        // [B, T, H, K]    bf16
    const void* __restrict__ ptr_k;        // [B, T, H, K]    bf16
    const void* __restrict__ ptr_w_bar;    // [B, T, H, K]    bf16
    const void* __restrict__ ptr_u_bar;    // [B, T, H, V]    bf16
    const void* __restrict__ ptr_g_cumsum; // [B, T, H]       fp32
    const void* __restrict__ ptr_h0;       // [B, H, V, K]    fp32  (nullable)
    void* __restrict__ ptr_o;              // [B, T, H, V]    bf16  output
    void* __restrict__ ptr_ht;             // [B, H, V, K]    fp32  final state (nullable)
    void* __restrict__ ptr_h_snap;         // [B, NT, H, V, K] bf16 h snapshots
    void* __restrict__ ptr_v_new;          // [B, T, H, V]    bf16  corrected values
    const uint32_t* __restrict__ ptr_k1_done;  // [NT * B * H] atomic flags (nullable)
    const int32_t* __restrict__ ptr_cu_seqlens;    // [N + 1] (varlen only)
    const int32_t* __restrict__ ptr_chunk_indices; // [total_chunks, 2]
    const int32_t* __restrict__ ptr_chunk_offsets; // [N + 1]
    int B;
    int T;
    int H;
    int K;
    int V;
    int NT;
    float scale;
};

// --------------------------------------------------------------------------
// K1 traits
//
// Layout: [B, T, H, K] with stride_t = H*K, stride_h = K, stride_k = 1
// MFMA: bf16 16×16×16 uniformly on gfx942/gfx950
// --------------------------------------------------------------------------
template<int BT_,
         int K_  = 128,
         int V_  = 128,
         int NUM_WARPS_ = 4>
struct gdn_k1_traits {
    static constexpr int BT = BT_;
    static constexpr int K  = K_;
    static constexpr int V  = V_;
    static constexpr int NUM_WARPS = NUM_WARPS_;
    static constexpr int WARP_SIZE = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;
    static constexpr bool IS_VARLEN = false;
    static constexpr bool DYNAMIC_SCALARS = false;

    using D_ATTN = bf16_t;
    using D_ACC  = float;

    // MFMA tile: 16×16×16 bf16 → fp32
    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 16;

    // Wave tiling: all warps along M, 1 wave along N and K
    static constexpr int T_M = NUM_WARPS;
    static constexpr int T_N = 1;
    static constexpr int T_K = 1;

    // KKT GEMM: [BT, K] × [K, BT] = [BT, BT]
    static constexpr int KKT_E_M = BT / W_M;          // 4 (BT=64), 1 (BT=16)
    static constexpr int KKT_E_N = BT / W_N;          // 4 (BT=64), 1 (BT=16)
    static constexpr int KKT_E_K = K / W_K;           // 8

    // w_bar GEMM: C[BT, BT] × k_scaled[BT, BK_sub] = [BT, BK_sub]
    // u_bar GEMM: C[BT, BT] × v_scaled[BT, BV_sub] = [BT, BV_sub]
    // K and V are processed in 64-wide subtiles (BK_sub=BV_sub=64)
    static constexpr int BK_SUB = 64;
    static constexpr int BV_SUB = 64;
    static constexpr int N_K_ITERS = K / BK_SUB;      // 2
    static constexpr int N_V_ITERS = V / BV_SUB;      // 2

    static constexpr int WY_E_M = BT / W_M;           // 4 (BT=64), 1 (BT=16)
    static constexpr int WY_E_N = BK_SUB / W_N;       // 4
    static constexpr int WY_E_K = BT / W_K;           // 4 (BT=64), 1 (BT=16)

    // Vector widths for load/store (bf16x8 = 16 bytes)
    static constexpr int VEC_KV = 8;

    // MFMA LDS padding: +4 bf16 elements per row to avoid bank conflicts
    // (effective on both 32-bank gfx942 and 64-bank gfx950)
    static constexpr int SMEM_PAD = 4;
    static constexpr int K_STRIDE = K + SMEM_PAD;

    // fp32 A matrix: stride padded to BT+1 to avoid bank conflicts
    // (BT DWORDs would align every row to the same bank on 32/64-bank LDS)
    static constexpr int A_STRIDE = BT + 1;

    // LDS layout (sizes in bytes)
    static constexpr int smem_k_padded_bytes = BT * K_STRIDE * (int)sizeof(D_ATTN);
    static constexpr int smem_A_bytes     = BT * A_STRIDE * (int)sizeof(D_ACC);
    static constexpr int smem_scalar_bytes = BT * 2 * (int)sizeof(D_ACC);
    static constexpr int smem_subtile_bytes = BT * BK_SUB * (int)sizeof(D_ATTN);

    // OCC=3 size (C_inv register-cached, no s_C_bf16). Used on gfx942 where the
    // smaller LDS lifts occupancy 2→3; also fine for all non-neumann k1 kernels.
    static constexpr size_t smem_size_bytes() {
        int phase1 = smem_k_padded_bytes + smem_scalar_bytes;
        if constexpr (BT >= 64) {
            int phase2ab = smem_A_bytes + 16*16*(int)sizeof(D_ACC) + smem_scalar_bytes;
            int peak = phase1;
            if (phase2ab > peak) peak = phase2ab;
            return peak;
        } else {
            int phase2 = smem_A_bytes + smem_subtile_bytes + smem_scalar_bytes;
            return (phase1 > phase2) ? phase1 : phase2;
        }
    }

    // OCC=2 size (neumann's C_inv staged in LDS as s_C_bf16). Used on gfx950,
    // whose 160KB LDS is not the occupancy limiter — the register-cache (OCC=3)
    // optimization is a no-op there, so the simpler LDS path is kept.
    static constexpr size_t smem_size_bytes_cinv_lds() {
        int base = smem_size_bytes();
        if constexpr (BT >= 64) {
            int c_bf16_bytes = BT * (BT + SMEM_PAD) * (int)sizeof(D_ATTN);
            int phase2c = smem_A_bytes + c_bf16_bytes + smem_scalar_bytes;
            return (phase2c > base) ? phase2c : base;
        }
        return base;
    }

    // Number of 16×16 diagonal blocks for forward substitution (BT=64 → 4 blocks)
    static constexpr int N_DIAG_BLOCKS = BT / 16;
};

// The measured fp32 gate path keeps its original branch-free loads.  Calls
// containing at least one bf16 gate use this second Neumann specialization,
// whose uniform dtype flags avoid wrapper-side conversion kernels.
template<typename BaseTraits>
struct gdn_k1_dynamic_scalar_traits : BaseTraits {
    static constexpr bool DYNAMIC_SCALARS = true;
};

// --------------------------------------------------------------------------
// K2 traits
//
// Grid: (cdiv(V, BV), N*H) — V dimension parallel, chunks serial
// h state: N_K registers × [BK_SUB, BV] fp32 in MFMA accumulator layout
// GEMMs: 5× MFMA bf16 16×16×16 (BT≥32 multi-tile, BT<32 single-tile QK^T)
// --------------------------------------------------------------------------
template<int BT_,
         int K_  = 128,
         int V_  = 128,
         int BV_ = 64,
         int NUM_WARPS_ = 4,
         int OCC_OVERRIDE_ = 0>
struct gdn_k2_traits {
    static constexpr int BT = BT_;
    static constexpr int K  = K_;
    static constexpr int V  = V_;
    static constexpr int BV = BV_;
    static constexpr int NUM_WARPS = NUM_WARPS_;
    static constexpr int WARP_SIZE = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;
    static constexpr bool IS_VARLEN = false;
    static constexpr int OCC_HINT = (OCC_OVERRIDE_ > 0) ? OCC_OVERRIDE_
                                   : ((BT_ >= 128) ? 1
                                   : ((NUM_WARPS_ <= 4) ? 2
                                   : ((BV_ <= 32) ? 2 : 1)));

    // Disable persistent q when OCC>=2 and nw>4: LDS with persistent q
    // (35072+) would exceed 32768 and block OCC=2.
    static constexpr bool PERSISTENT_Q = (OCC_HINT <= 1) || (NUM_WARPS_ <= 4);

    // Serialize phase b/c GEMMs (retrieve then cross, sharing one sub buffer)
    // to reduce pool_bc from 3 buffers to 2, enabling OCC=2 LDS fit.
    static constexpr bool SERIALIZE_BC = !PERSISTENT_Q && (OCC_HINT >= 2);

    // Fused-K2 code-generation controls.  The base traits keep every legacy
    // path enabled; gdn_k2_fused_traits below opts a measured dense WF variant
    // into narrower code generation without changing split/output kernels.
    static constexpr bool DENSE_ALIGNED = false;
    static constexpr bool NO_AUX_OUTPUTS = false;
    static constexpr bool CACHE_GATES = false;
    static constexpr bool REUSE_DE_K = false;
    static constexpr bool EARLY_NEXT_PREFETCH = false;

    using D_ATTN = bf16_t;
    using D_ACC  = float;

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 16;

    static constexpr int T_M = NUM_WARPS;
    static constexpr int T_N = 1;
    static constexpr int T_K = 1;

    // K subtiling: h state split into 64-wide K subtiles (matches Triton)
    static constexpr int BK_SUB = 64;
    static constexpr int N_K = K / BK_SUB;             // 2 for K=128

    // Step 3 Retrieve GEMM: w_bar_sub[BT, BK_SUB] × h[BK_SUB, BV] = tmp[BT, BV]
    static constexpr int RET_E_M = BT / W_M;
    static constexpr int RET_E_N = BV / W_N;
    static constexpr int RET_E_K = BK_SUB / W_K;

    // Step 3 Accumulate GEMM: k_sub^T[BK_SUB, BT] × v_hat[BT, BV] = [BK_SUB, BV]
    static constexpr int ACC_E_M = BK_SUB / W_M;
    static constexpr int ACC_E_N = BV / W_N;
    static constexpr int ACC_E_K = BT / W_K;

    // Step 4 Intra QK^T: q_sub[BT, BK_SUB] × k_sub^T[BK_SUB, BT] = A_intra[BT, BT]
    static constexpr int QKT_E_M = BT / W_M;
    static constexpr int QKT_E_N = BT / W_N;
    static constexpr int QKT_E_K = BK_SUB / W_K;

    // Step 4 Intra AV: A_intra[BT, BT] × v_new[BT, BV] = [BT, BV]
    static constexpr int AV_E_M = BT / W_M;
    static constexpr int AV_E_N = BV / W_N;
    static constexpr int AV_E_K = BT / W_K;

    static constexpr int VEC_KV = 8;

    // MFMA LDS padding: +4 bf16 elements per row to avoid bank conflicts
    // (effective on both 32-bank gfx942 and 64-bank gfx950)
    static constexpr int SMEM_PAD = 4;

    // LDS layout (MFMA optimized, all bf16 buffers padded)
    //
    // Persistent regions:
    //   s_g[BT] fp32                          — gate cumsum
    //   s_v_T[BV, BT+PAD] bf16               — v_new transposed (phases b'→e)
    //   (PERSISTENT_Q only) s_q[N_K, BT, BK_SUB+PAD] bf16
    //
    // Pool (aliased per phase):
    //   Phase b/c: s_h_T[BV, BK_SUB+PAD] + s_sub_w[BT, BK_SUB+PAD]
    //              + (!PERSISTENT_Q: s_sub_q[BT, BK_SUB+PAD])
    //   Phase d:   s_k_T[BK_SUB, BT+PAD] bf16
    //   Phase e:   (PERSISTENT_Q)  s_k[BT, BK_SUB+PAD] bf16
    //              (!PERSISTENT_Q) s_k[BT, BK_SUB+PAD] + s_q[BT, BK_SUB+PAD]
    //              s_A[BT, BT+PAD] bf16 (AV, reuses pool)
    static constexpr int smem_g_bytes = BT * (int)sizeof(D_ACC);
    static constexpr int smem_vT_bytes = BV * (BT + SMEM_PAD) * (int)sizeof(D_ATTN);
    static constexpr int N_K_ = K / BK_SUB;
    static constexpr int smem_q_bytes = PERSISTENT_Q
        ? (N_K_ * BT * (BK_SUB + SMEM_PAD) * (int)sizeof(D_ATTN))
        : 0;

    static constexpr size_t smem_size_bytes() {
        constexpr int P = SMEM_PAD;
        int one_sub = BT * (BK_SUB + P) * (int)sizeof(D_ATTN);
        // SERIALIZE_BC: w and q share one sub buffer (serialized GEMMs)
        int pool_bc  = BV * (BK_SUB + P) * (int)sizeof(D_ATTN)     // s_h_T
                     + one_sub                                       // s_sub_w
                     + (PERSISTENT_Q || SERIALIZE_BC ? 0 : one_sub);// s_sub_q
        int pool_d   = BK_SUB * (BT + P) * (int)sizeof(D_ATTN);    // s_k_T
        int pool_eqk = PERSISTENT_Q ? one_sub                       // s_k4 only
                     : (2 * one_sub);                                // s_k4 + s_q4
        int pool_eav = BT * (BT + P) * (int)sizeof(D_ATTN);        // s_A
        int pool = pool_bc;
        if (pool_d   > pool) pool = pool_d;
        if (pool_eqk > pool) pool = pool_eqk;
        if (pool_eav > pool) pool = pool_eav;
        return smem_g_bytes + smem_vT_bytes + smem_q_bytes + pool;
    }

    // LDS for the split-path scan kernel (gdn_k2_scan_kernel):
    //   s_g[BT] + s_vT[BV,STRIDE_BT] + pool
    //   pool = max( s_hT[BV,STRIDE_BK] + s_sub_w[BT,STRIDE_BK],  s_kT[BK_SUB,STRIDE_BT] )
    static constexpr size_t smem_scan_bytes() {
        constexpr int P = SMEM_PAD;
        constexpr int A = (int)sizeof(D_ATTN);
        int pool_bc = (BV + BT) * (BK_SUB + P);
        int pool_d  = BK_SUB * (BT + P);
        int pool = (pool_bc > pool_d) ? pool_bc : pool_d;
        return BT * (int)sizeof(D_ACC) + (BV * (BT + P) + pool) * A;
    }

    // LDS for the split-path output kernel (gdn_k2_out_kernel):
    //   s_g[BT] + s_q[N_K,BT,STRIDE_BK] + s_kh[max(BV,BT),STRIDE_BK]
    //          + s_vT[BV,STRIDE_BT] + s_A5[BT,STRIDE_BT]
    static constexpr size_t smem_out_bytes() {
        constexpr int P = SMEM_PAD;
        constexpr int A = (int)sizeof(D_ATTN);
        constexpr int MX = (BV > BT) ? BV : BT;
        return BT * (int)sizeof(D_ACC)
             + (N_K_ * BT * (BK_SUB + P)
              + MX * (BK_SUB + P)
              + BV * (BT + P)
              + BT * (BT + P)) * A;
    }
};

// Compile a metadata-aware ragged specialization without adding runtime
// branches to the tuned dense kernels.  The wrapper is intentionally generic:
// K1 and K6 both use it, while the serial state scan uses its existing native
// varlen specialization from ref_fwd_h.hpp.
template<typename BaseTraits>
struct gdn_varlen_traits : BaseTraits {
    static constexpr bool IS_VARLEN = true;
    static constexpr bool DENSE_ALIGNED = false;
    static constexpr bool REVERSE_CHUNKS = false;
};

// Packed inputs whose every sequence length is a BT multiple still need the
// metadata addressing, but they do not need any token-tail predicates.  Keep
// this distinct from gdn_varlen_traits so ragged tails retain the generic path.
template<typename BaseTraits>
struct gdn_varlen_aligned_traits : BaseTraits {
    static constexpr bool IS_VARLEN = true;
    static constexpr bool DENSE_ALIGNED = true;
    static constexpr bool REVERSE_CHUNKS = false;
};

// Fused W/U K2 specializations.  DENSE_ALIGNED is valid only for the public
// dense wrapper, which pads T to BT and requires BV to divide V.  NO_AUX_OUTPUTS
// encodes the fused launcher's null h_snapshot/v_new pointers.  CACHE_GATES
// adds two fp32 BT-vectors for exp(g) and exp(g_last-g).  REUSE_DE_K keeps the
// two K64 panels in the already-live pf_q/pf_k registers across phases d/e,
// eliminating the second full K read without extending the register footprint.
template<typename BaseTraits,
         bool DENSE_ALIGNED_ = false,
         bool NO_AUX_OUTPUTS_ = false,
         bool CACHE_GATES_ = false,
         bool REUSE_DE_K_ = false,
         bool EARLY_NEXT_PREFETCH_ = false>
struct gdn_k2_fused_traits : BaseTraits {
    static constexpr bool DENSE_ALIGNED = DENSE_ALIGNED_;
    static constexpr bool NO_AUX_OUTPUTS = NO_AUX_OUTPUTS_;
    static constexpr bool CACHE_GATES = CACHE_GATES_;
    static constexpr bool REUSE_DE_K = REUSE_DE_K_;
    static constexpr bool EARLY_NEXT_PREFETCH = EARLY_NEXT_PREFETCH_;

    static_assert(!DENSE_ALIGNED ||
                      (BaseTraits::BT == 64 && BaseTraits::K == 128 &&
                       BaseTraits::V == 128 &&
                       (BaseTraits::BV == 64 || BaseTraits::BV == 128) &&
                       (BaseTraits::NUM_WARPS == 8 ||
                        BaseTraits::NUM_WARPS == 16)),
                  "dense fused K2 requires BT64/K128/V128/BV64-or-128/NW8-or-16");
    static_assert(!REUSE_DE_K ||
                      (BaseTraits::N_K == 2 && BaseTraits::PERSISTENT_Q),
                  "phase-d/e K reuse requires two K64 panels and persistent q");
    static_assert(!EARLY_NEXT_PREFETCH ||
                      (DENSE_ALIGNED && BaseTraits::BT >= 32),
                  "early next-chunk prefetch requires dense multi-tile QKT");
    static constexpr int extra_gate_bytes = CACHE_GATES
        ? 2 * BaseTraits::BT * (int)sizeof(typename BaseTraits::D_ACC)
        : 0;
    static constexpr int smem_g_bytes =
        BaseTraits::smem_g_bytes + extra_gate_bytes;

    static constexpr size_t smem_size_bytes() {
        return BaseTraits::smem_size_bytes() + extra_gate_bytes;
    }
};

// Output-only specialization controls.  Keep these separate from
// gdn_k2_traits so the serial/fused K2 kernels retain their existing type and
// code generation.  DENSE_ALIGNED is intentionally narrow: its host launchers
// guarantee complete BT/BV tiles, which lets gdn_k2_out_kernel discard every
// sequence/value tail path.  REVERSE_CHUNKS changes only the independent K6
// workgroup scheduling order.
template<typename BaseTraits,
         bool DENSE_ALIGNED_ = false,
         bool REVERSE_CHUNKS_ = false>
struct gdn_k2_out_traits : BaseTraits {
    static constexpr bool DENSE_ALIGNED = DENSE_ALIGNED_;
    static constexpr bool REVERSE_CHUNKS = REVERSE_CHUNKS_;

    static_assert(!REVERSE_CHUNKS || DENSE_ALIGNED,
                  "reverse K6 scheduling is available only for dense output");
    static_assert(!DENSE_ALIGNED ||
                      (BaseTraits::BT == 64 && BaseTraits::K == 128 &&
                       BaseTraits::V == 128 && BaseTraits::NUM_WARPS == 8 &&
                       (BaseTraits::BV == 64 || BaseTraits::BV == 128)),
                  "dense K6 requires BT64/K128/V128/NW8/BV64-or-128");
};

// --------------------------------------------------------------------------
// Wavefront scan-only kernel arguments (h-state update with inter-WG sync)
// --------------------------------------------------------------------------
struct gdn_wf_h_kargs {
    const void* __restrict__ ptr_k;         // [B, T, H, K]   bf16
    const void* __restrict__ ptr_w_bar;     // [B, T, H, K]   bf16
    const void* __restrict__ ptr_u_bar;     // [B, T, H, V]   bf16
    const void* __restrict__ ptr_g_cumsum;  // [B, T, H]      fp32
    const void* __restrict__ ptr_h0;        // [B, H, V, K]   fp32  (nullable)
    void* __restrict__ ptr_h;               // [B, NT, H, K, V] bf16  h snapshots (nullable)
    void* __restrict__ ptr_v_new;           // [B, T, H, V]   bf16  (nullable)
    void* __restrict__ ptr_ht;              // [B, H, V, K]   fp32  final state (nullable)
    void* __restrict__ ptr_h_pass;          // [N_flat, N_super, K, BV] fp32
    uint32_t* __restrict__ ptr_flags;       // [N_flat * N_super] uint32
    const void* __restrict__ ptr_q;         // [B, T, H, K]   bf16  (nullable, for fused output)
    void* __restrict__ ptr_o;               // [B, T, H, V]   bf16  (nullable, fused output)
    int B, T, H, K, V, NT;
    int S;
    int N_super;
    float scale;
};

// --------------------------------------------------------------------------
// K2 output kernel arguments (split mode: parallel output, no serial deps)
// --------------------------------------------------------------------------
struct gdn_k2_output_kargs {
    const void* __restrict__ ptr_q;         // [B, T, H, K]    bf16
    const void* __restrict__ ptr_k;         // [B, T, H, K]    bf16
    const void* __restrict__ ptr_v_new;     // [B, T, H, V]    bf16
    const void* __restrict__ ptr_h_snap;    // [B, NT, H, K, V] fp32
    const void* __restrict__ ptr_g_cumsum;  // [B, T, H]       fp32
    void* __restrict__ ptr_o;               // [B, T, H, V]    bf16  output
    int B, T, H, K, V, NT;
    int NV;                                 // cdiv(V, BV)
    float scale;
};

// --------------------------------------------------------------------------
// Utilities
// --------------------------------------------------------------------------
__host__ __device__ inline int ceil_div(int a, int b) {
    return a / b + (a % b != 0);
}
