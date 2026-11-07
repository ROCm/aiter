// Shared types and constants between device kernel and host code
#pragma once

using bf16_t = __bf16;

// Kernel arguments for GQA attention
struct opus_gqa_kargs {
    const void* __restrict__ ptr_q;  // [B, N, H, D]
    const void* __restrict__ ptr_k;  // [B, N, H_KV, D]
    const void* __restrict__ ptr_v;  // [B, N, H_KV, D]
    void* __restrict__ ptr_o;        // [B, N, H, D]
    int B;
    int N;      // seqlen_q
    int N_KV;   // seqlen_kv
    int H;
    int H_KV;
    int D;
    int stride_q_b;
    int stride_q_n;
    int stride_q_h;
    int stride_o_b;
    int stride_o_n;
    int stride_o_h;
    int stride_k_b;
    int stride_k_n;
    int stride_k_h;
    int stride_v_b;
    int stride_v_n;
    int stride_v_h;
    float softmax_scale;  // QK^T scale (host passes 1/sqrt(D) by default)
    // Optional fp32 log-sum-exp (natural log) output, [B, H, N] with unit stride along
    // the query dim. nullptr => not produced (the kernel skips the store).
    void* __restrict__ ptr_lse;
    int stride_lse_b;
    int stride_lse_h;

    // ── group mode (varlen / packed sequences) ──
    // Prefix sums (length B+1). *_seqstart_* are real lengths (masking, KV-tile count,
    // empty-tail short circuit); *_seqstart_*_pad are physical row offsets. A null
    // ptr_seqstart_q selects batch mode. Only read when the traits' GROUP_MODE is set.
    const int* ptr_seqstart_q;
    const int* ptr_seqstart_k;
    const int* ptr_seqstart_q_pad;
    const int* ptr_seqstart_k_pad;

    // Sliding window, only meaningful when the traits' LOCAL is set: a query at
    // KV-coordinate j attends to [j - window_size_left, j]. Negative means unbounded
    // (plain causal). right > 0 is not implemented.
    int window_size_left;

    // ── group mode: flattened Q-block axis ──
    // When > 0, the extent of the Q-block axis: the groups' blocks laid end to end, and
    // each block works out which group it belongs to. The value belongs to whichever of
    // the three placement schemes below is selected, since each packs the flat index its
    // own way; see fmha_fwd_bf16_opus_args::total_q_blocks for what to pass.
    //
    // The default (0) keeps the rectangular grid, where every group is given as many
    // Q-blocks as the LONGEST group needs and the surplus returns immediately. That costs
    // nothing when the lengths are close, but one sequence at max_seqlen beside a batch of
    // ordinary ones multiplies every group's block count: measured at B=8 with seven ~8.4k
    // sequences plus one at 16,354, it took the workgroup count from 1,184 to 2,048 (+73%)
    // for 11% more work, and 53% more time.
    //
    // 0 means "not supplied" rather than "no blocks".
    int total_q_blocks;

    // Flattened mode, optional. One packed (group << 16) | block pair per flat index,
    // length total_q_blocks; entries past the last real block must be negative so that a
    // grid sized from an upper bound can retire them. Takes precedence over
    // ptr_block_start.
    //
    // This exists for the ORDER it lets the caller choose, not for lookup speed. Laying
    // the blocks out group by group costs 8% against the rectangular grid even with the
    // lookup removed entirely, because it changes which blocks are resident together:
    // consecutive flat indices then walk one sequence's Q-blocks, so the live set spans
    // that sequence's whole K/V, where the rectangular grid holds every sequence at the
    // same block index and needs only a window of each. Measured at B=16 x 9,400:
    // L2 hit 54.9% -> 34.8%. Enumerating block-major (block 0 of every group, then block
    // 1, ...) restores it. That order is not monotonic in the group index and so cannot be
    // recovered by searching a prefix table -- hence a table of the pairs themselves.
    const int* ptr_block_map;

    // Flattened mode, optional fallback when ptr_block_map is null: cumulative Q-block
    // counts, length B+1. With neither table a block inverts seqstart_q directly, by the
    // same binary search over a different array, so this saves only the divide -- supply
    // it if it is being built anyway, not to speed the lookup up.
    const int* ptr_block_start;
};

// Configuration traits for the GQA kernel (tile sizes, data types, vector lengths,
// MFMA config). This integration is D=128 only: MFMA 32x32x16 bf16, K/V loaded in
// one shot (no D slicing).
template<int Q_TILE_SIZE_ = 32,
        int KV_TILE_SIZE_ = 64,
        int D_TILE_SIZE_ = 128,
        int NUM_WARPS_ = 8,
        bool CAUSAL_ = false,
        bool GROUP_MODE_ = false,
        bool LOCAL_ = false>
struct opus_gqa_traits {
    static_assert(D_TILE_SIZE_ == 128, "fmha_fwd_hd128_bf16_opus supports D_TILE_SIZE 128 only");

    static constexpr int Q_TILE_SIZE = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int D_TILE_SIZE = D_TILE_SIZE_;
    static constexpr int NUM_WARPS = NUM_WARPS_;
    static constexpr bool CAUSAL = CAUSAL_;
    // Group (varlen) mode: locate sequences via seqstart, not a uniform batch stride.
    static constexpr bool GROUP_MODE = GROUP_MODE_;
    // Sliding window: bound the KV range on the left too, via kargs.window_size_left.
    static constexpr bool LOCAL = LOCAL_;
    static_assert(!LOCAL_ || CAUSAL_, "LOCAL needs CAUSAL for the right-hand bound");

    static constexpr int WARP_SIZE = 64; // AMD wavefront size
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    // Data types: Q/K/V/O share one bf16 type; accumulation fp32
    using D_ATTN = bf16_t;
    using D_ACC  = float;

    // MFMA wave layout
    static constexpr int T_M = NUM_WARPS; // waves along M
    static constexpr int T_N = 1;         // waves along N
    static constexpr int T_K = 1;         // waves along K

    // MFMA base tile (D=128): bf16 32x32x16
    static constexpr int W_M = 32;
    static constexpr int W_N = 32;
    static constexpr int W_K = 16;

    // D=128 covers the full head dim in one MMA (no D slicing).
    static constexpr int SLICE_D = D_TILE_SIZE;  // == 128
    static constexpr int NUM_D_SLICES = 1;
    static_assert(D_TILE_SIZE % SLICE_D == 0);

    // GEMM0: S[Q_TILE x KV_TILE] = Q[Q_TILE x SLICE_D] @ K^T[SLICE_D x KV_TILE]
    static constexpr int GEMM0_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM0_E_N = KV_TILE_SIZE / W_N;
    static constexpr int GEMM0_E_K = SLICE_D / W_K;

    // GEMM1: O[Q_TILE x SLICE_D] = P[Q_TILE x KV_TILE] @ V[KV_TILE x SLICE_D]
    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM1_E_N = SLICE_D / W_N;
    static constexpr int GEMM1_E_K = KV_TILE_SIZE / W_K;

    // Vector lengths for global load/store
    static constexpr int VEC_Q    = 8;
    static constexpr int VEC_KV   = 8;
    static constexpr int VEC_TR_V = 4;
    static constexpr int VEC_O    = 4;
    static constexpr int VEC_O_X4 = 16 / sizeof(D_ATTN);   // 8

    // Minimal compact pixels for async copy for one wave
    static constexpr int D_128B_SIZE = 128 / sizeof(D_ATTN);
    static_assert(VEC_KV == 16 / sizeof(D_ATTN));
    static constexpr int smem_linear_wave = WARP_SIZE * 16 / sizeof(D_ATTN);
    static constexpr int smem_n_per_wave = smem_linear_wave / D_128B_SIZE;
    static constexpr int smem_n_rpt = KV_TILE_SIZE / smem_n_per_wave;
    static constexpr int smem_d_rpt = D_TILE_SIZE / D_128B_SIZE;
    // Whole-WG Q block (NUM_WARPS * Q_TILE) staged through LDS in the async-Q path.
    static constexpr int Q_TOTAL_TILE_SIZE = Q_TILE_SIZE * NUM_WARPS;
    static constexpr int smem_n_q_rpt = Q_TOTAL_TILE_SIZE / smem_n_per_wave;

    static constexpr int smem_padding_16B = 16 / sizeof(D_ATTN);
    static constexpr int smem_padding_64B = 64 / sizeof(D_ATTN);

    // K/V smem padding (D=128): K uses 16B padding, V uses 64B padding.
    static constexpr int smem_k_padding = smem_padding_16B;
    static constexpr int smem_v_padding = smem_padding_64B;

    static constexpr int smem_k_tile_elems = smem_n_rpt * smem_d_rpt * (smem_linear_wave + smem_k_padding);
    static constexpr int smem_v_tile_elems = smem_n_rpt * smem_d_rpt * (smem_linear_wave + smem_v_padding);
    static constexpr int smem_buffer_elems = smem_k_tile_elems + smem_v_tile_elems;

    // Q shared-tile geometry for the async-Q-via-LDS path (Q has same head dim as K, so
    // reuses smem_d_rpt / 16B padding). Q aliases the double-buffered V region.
    static constexpr int smem_q_padding = smem_padding_16B;
    static constexpr int smem_q_tile_elems = smem_n_q_rpt * smem_d_rpt * (smem_linear_wave + smem_q_padding);
    // Layout: K double-buffered (2*K) + a shared region max(Q, 2*V) (Q consumed in the
    // prologue before V0 overwrites it).
    static constexpr int smem_buffer_elems_qlds = smem_k_tile_elems * 2
        + (smem_q_tile_elems > smem_v_tile_elems * 2 ? smem_q_tile_elems : smem_v_tile_elems * 2);

    static constexpr int k_buffer_load_insts = (KV_TILE_SIZE * D_TILE_SIZE) / (BLOCK_SIZE * VEC_KV);
    static constexpr int v_buffer_load_insts = (KV_TILE_SIZE * D_TILE_SIZE) / (BLOCK_SIZE * VEC_KV);
    // Rolling-vmcnt keep-count for the async-prefetch main loop: how many in-flight vmem
    // loads to leave un-waited at each mem-stage s_waitcnt_vmcnt.
    static constexpr int KEEP_VMCNT = k_buffer_load_insts + v_buffer_load_insts;
    static constexpr int k_ds_read_insts = (GEMM0_E_N * GEMM0_E_K * W_N * W_K) / (WARP_SIZE * VEC_KV);
    static constexpr int v_ds_read_insts = (GEMM1_E_N * GEMM1_E_K * W_N * W_K) / (WARP_SIZE * VEC_TR_V);

    static constexpr size_t smem_size_bytes() {
        // Q-in-LDS layout (K double-buffered + Q/V aliased shared region).
        return (size_t)smem_buffer_elems_qlds * sizeof(D_ATTN);
    }
};

#ifndef OPUS_FMHA_FWD_CEIL_DIV
#define OPUS_FMHA_FWD_CEIL_DIV
__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}
#endif
