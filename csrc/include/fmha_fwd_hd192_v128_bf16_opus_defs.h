// Shared types and constants for the asymmetric head-dim GQA kernel (D_QK=192, D_V=128).
//   GEMM0 (Q*K^T): head dim D_QK = 192
//   GEMM1 (P*V)  : head dim D_V  = 128 (== output head dim)
// Self-contained: defines bf16_t / ceil_div locally (no dependency on other opus defs),
// so it can be pulled into the shared opus fwd translation unit alongside the symmetric
// D=128 kernel without symbol clashes.
#pragma once

using bf16_t = __bf16;

// Kernel arguments for asymmetric-head-dim GQA attention.
// Q: [B, N, H,    D_QK]   K: [B, N, H_KV, D_QK]
// V: [B, N, H_KV, D_V ]   O: [B, N, H,    D_V ]
struct opus_gqa_d192_kargs {
    const void* __restrict__ ptr_q;
    const void* __restrict__ ptr_k;
    const void* __restrict__ ptr_v;
    void* __restrict__ ptr_o;
    int B;
    int N;      // seqlen_q (drives grid / Q tiling)
    int N_KV;   // seqlen_kv (drives KV tile traversal / masks); == N for self-attention
    int H;
    int H_KV;
    int D_QK;   // 192
    int D_V;    // 128
    // Q / O strides (head dim differs: Q uses D_QK, O uses D_V)
    int stride_q_b;
    int stride_q_n;
    int stride_q_h;
    int stride_o_b;
    int stride_o_n;
    int stride_o_h;
    // K / V strides (K uses D_QK, V uses D_V)
    int stride_k_b;
    int stride_k_n;
    int stride_k_h;
    int stride_v_b;
    int stride_v_n;
    int stride_v_h;
    // Softmax scale applied to Q·K^T (the kernel additionally folds in log2(e) for its
    // exp2-based softmax). Host passes the caller's scale, defaulting to 1/sqrt(D_QK).
    float softmax_scale;
    // ── group mode (varlen / packed sequences) ──
    // Prefix-sum arrays (length B+1) locating each group in the packed Q/K/V/O buffers.
    // *_seqstart_*    → real sequence lengths (seqlen = seqstart[g+1]-seqstart[g]) used for
    //                   masks / KV-tile count / short-circuit.
    // *_seqstart_*_pad→ physical row offsets used for addressing (KV padding variant);
    //                   equals the non-pad array when there is no padding.
    // Unused (may be nullptr) in batch mode.
    const int* ptr_seqstart_q;
    const int* ptr_seqstart_k;
    const int* ptr_seqstart_q_pad;
    const int* ptr_seqstart_k_pad;
    // Runtime option bits (see OPT_* below). Decided once by the host and read by the
    // kernel, so the head/tail-merge decision is NOT recomputed on both sides.
    int opt;
};

// opus_gqa_d192_kargs::opt bit flags.
//   OPT_MERGE_HEADTAIL: causal head/tail load-balance merge is active for this launch
//   (the host halved the q-block grid dim accordingly). The host sets it only for
//   causal launches whose full (unmerged) grid WG count nQ*H*B >= HEADTAIL_MIN_WG —
//   below that the machine is under-filled and halving the grid hurts more than the
//   load balance helps.
static constexpr int OPT_MERGE_HEADTAIL = 1 << 0;

// Threshold (full unmerged grid WG count nQ*H*B) above which the host enables
// OPT_MERGE_HEADTAIL. Used by the host only.
static constexpr int HEADTAIL_MIN_WG = 512;

// Configuration traits for the D_QK=192 / D_V=128 GQA kernel.
// Fixed MFMA 32x32x16 bf16 (same wave tile as the symmetric D=128 kernel).
template<int Q_TILE_SIZE_ = 32,
         int KV_TILE_SIZE_ = 64,
         int NUM_WARPS_ = 8,
         bool CAUSAL_ = false,
         bool GROUP_MODE_ = false>
struct opus_gqa_d192_traits {
    static constexpr int Q_TILE_SIZE  = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int D_QK         = 192;
    static constexpr int D_V          = 128;
    static constexpr int NUM_WARPS    = NUM_WARPS_;
    static constexpr bool CAUSAL      = CAUSAL_;
    static constexpr bool GROUP_MODE  = GROUP_MODE_;
    static constexpr int Q_TOTAL_TILE_SIZE  = Q_TILE_SIZE * NUM_WARPS; // 256

    static constexpr int WARP_SIZE  = 64; // AMD wavefront size
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    using D_ATTN = bf16_t;
    using D_ACC  = float;

    // MFMA wave layout
    static constexpr int T_M = NUM_WARPS;
    static constexpr int T_N = 1;
    static constexpr int T_K = 1;

    // MFMA base tile: bf16 32x32x16
    static constexpr int W_M = 32;
    static constexpr int W_N = 32;
    static constexpr int W_K = 16;

    // GEMM0: S[Q_TILE x KV_TILE] = Q[Q_TILE x D_QK] @ K^T[D_QK x KV_TILE]
    static constexpr int GEMM0_E_M = Q_TILE_SIZE / W_M;   // 1
    static constexpr int GEMM0_E_N = KV_TILE_SIZE / W_N;  // 2
    static constexpr int GEMM0_E_K = D_QK / W_K;          // 12
    // Super-unit split along KV-seq (N): 2 super units, each one W_N tile.
    static constexpr int GEMM0_NUM_SU = 2;
    static constexpr int GEMM0_E_N_SU = GEMM0_E_N / GEMM0_NUM_SU; // 1

    // GEMM1: O[Q_TILE x D_V] = P[Q_TILE x KV_TILE] @ V[KV_TILE x D_V]
    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M;   // 1
    static constexpr int GEMM1_E_N = D_V / W_N;           // 4
    static constexpr int GEMM1_E_K = KV_TILE_SIZE / W_K;  // 4
    // Super-unit split along head dim (D_V): 2 super units, each 64 head dim.
    static constexpr int GEMM1_NUM_SU = 2;
    static constexpr int GEMM1_E_N_SU = GEMM1_E_N / GEMM1_NUM_SU; // 2

    // Vector lengths for global load / register transpose-load / store
    static constexpr int VEC_Q    = 8;
    static constexpr int VEC_KV   = 8;
    static constexpr int VEC_TR_V = 4;
    static constexpr int VEC_O    = 4;
    // Widened O store: bf16 elements per buffer_store_dwordx4 (16 bytes). Used by the
    // permlane32-packed write-back path (make_layout_o_x4 + the per-group store loop).
    static constexpr int VEC_O_X4 = 16 / sizeof(D_ATTN);   // 8

    // Compact-copy geometry for async global->shared
    static constexpr int D_128B_SIZE = 128 / sizeof(D_ATTN); // 64
    static_assert(VEC_KV == 16 / sizeof(D_ATTN));
    static constexpr int smem_linear_wave = WARP_SIZE * 16 / sizeof(D_ATTN); // 512
    static constexpr int smem_n_per_wave  = smem_linear_wave / D_128B_SIZE;  // 8
    static constexpr int smem_n_rpt       = KV_TILE_SIZE / smem_n_per_wave;  // 8
    static constexpr int smem_n_q_rpt     = Q_TOTAL_TILE_SIZE / smem_n_per_wave;  // 32
    // d_rpt differs between K (192) and V (128)
    static constexpr int smem_d_rpt_qk = D_QK / D_128B_SIZE; // 3
    static constexpr int smem_d_rpt_v = D_V  / D_128B_SIZE; // 2

    static constexpr int smem_padding_16B = 16 / sizeof(D_ATTN);
    static constexpr int smem_padding_64B = 64 / sizeof(D_ATTN);

    // K uses 16B padding, V uses 64B padding (matches the symmetric D=128 kernel).
    static constexpr int smem_q_padding = smem_padding_16B;
    static constexpr int smem_k_padding = smem_padding_16B;
    static constexpr int smem_v_padding = smem_padding_64B;

    static constexpr int smem_q_tile_elems = smem_n_q_rpt * smem_d_rpt_qk * (smem_linear_wave + smem_q_padding);
    static constexpr int smem_k_tile_elems = smem_n_rpt * smem_d_rpt_qk * (smem_linear_wave + smem_k_padding);
    static constexpr int smem_v_tile_elems = smem_n_rpt * smem_d_rpt_v * (smem_linear_wave + smem_v_padding); //for ping-pong
    // K is double-buffered (×2); Q and the double-buffered V (×2) alias the SAME LDS
    // region (Q is consumed in the prologue before V0 overwrites it), so that region is
    // max(Q, 2·V). Total = 2·K + max(Q, 2·V) — this ALREADY includes ping-pong.
    static constexpr int smem_buffer_elems = smem_k_tile_elems * 2
        + (smem_q_tile_elems > smem_v_tile_elems * 2 ? smem_q_tile_elems : smem_v_tile_elems * 2);

    static constexpr int k_buffer_load_insts = (KV_TILE_SIZE * D_QK) / (BLOCK_SIZE * VEC_KV); // 3
    static constexpr int v_buffer_load_insts = (KV_TILE_SIZE * D_V)  / (BLOCK_SIZE * VEC_KV); // 2
    // Rolling vmcnt kept in flight by the pipelined kernel: one K prefetch (2-tile
    // ahead, reusing the buffer freed after the c2 read) + one V prefetch.
    static constexpr int KEEP_VMCNT = k_buffer_load_insts + v_buffer_load_insts;         // 5

    static constexpr size_t smem_size_bytes() {
        // smem_buffer_elems already includes the ping-pong factor (2·K, 2·V) — do NOT
        // multiply by 2 again (that was the pre-Q-in-LDS layout's convention).
        return smem_buffer_elems * sizeof(D_ATTN);
    }
};

#ifndef OPUS_FMHA_FWD_CEIL_DIV
#define OPUS_FMHA_FWD_CEIL_DIV
__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}
#endif
