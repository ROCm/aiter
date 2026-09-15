// SPDX-License-Identifier: MIT
// Prefill-phase FP8 MQA indexer logits kernel, gfx950.
#include "fp8_mqa_logits.h"

// ============================================================================
// fp8_mqa_logits — PREFILL-phase indexer kernel (GLM-5-FP8 / DeepSeek-V3.2), gfx950
//
//   logits[m, n] = Σ_h relu(Q[m,h,:] · K[n,:]) · w[m,h] · kv_scale[n]
//                  for n ∈ [ks[m], ke[m]), −inf elsewhere
//
// This is the non-paged sibling of fp8_paged_mqa_logits.cpp (decode). The caller
// has already gathered K out of the paged cache into a contiguous [N, 128] fp8
// buffer with a separate [N] fp32 scale, so there is no block table and no
// preshuffle: K[n,d] is simply at n*128 + d.
//
// Baseline being displaced: aiter's Gluon kernel
// aiter/ops/triton/_gluon_kernels/gfx950/attention/fp8_mqa_logits.py — one 1-warp
// workgroup per query row, BLOCK_KV=32, the same V_MFMA_F32_32X32X64_F8F6F4, with K
// staged through LDS by CDNA4 async-copy.
//
// The kernel here is the decode kernel's inner loop retargeted: 32 heads × 32 tokens per
// MFMA tile, K streamed HBM→register (no LDS), permlane32_swap head reduction, and
// the per-token scale applied once after that reduction. The decode "rows per block"
// knob becomes BLOCK_M — the number of query rows that share one K stream.
// ============================================================================

using fp8 = __hip_fp8_storage_t;

constexpr int WARP_SIZE = 64;
constexpr int NUM_HEADS = 32;
constexpr int HEAD_SIZE = 128;

#define CDIV_(a, b) (((a) + (b) - 1) / (b))

namespace mqa_dense {

using f8x16 = __attribute__((__vector_size__(16))) fp8;
using f8x32 = __attribute__((__vector_size__(32))) fp8;
using f32x4 = __attribute__((__vector_size__(16))) float;

// lanes 0-31 receive x[lane+32]; lanes 32-63 get garbage (unused). See the decode
// kernel for the full note: the clang builtin is unusable (both results land in one
// VGPR) and the `s_nop 1` is mandatory (VALU write → permlane read needs 2 wait
// states), otherwise the swap reads stale data.
__device__ __forceinline__ float swap32_hi(float x) {
    float hi;
    asm("s_nop 1\n\tv_permlane32_swap_b32 %0, %1" : "+v"(x), "=&v"(hi));
    return hi;
}

// relu(x). Compiled with -fno-honor-nans (set for this module in
// aiter/jit/optCompilerConfig.json) this is a single
// `v_max_f32 x, 0, x`. Without that flag LLVM must assume x may be a signalling NaN
// and emits an IEEE canonicalize `v_max_f32 x, x, x` first — two VALU per value, and
// this loop runs NACC of them per row per tile, ~27% of all VALU in the kernel.
// Inline asm would also give one instruction but is NOT safe here: the hazard
// recognizer does not treat inline asm as a reader of the MFMA destination, so the
// mandatory post-MFMA s_nops disappear and the accumulator is read before it lands.
__device__ __forceinline__ float relu_fast(float x) { return fmaxf(x, 0.0f); }

// Reduce TWO rows' per-head partials at once. `v_permlane32_swap_b32 a, b` leaves
// a = (a.row0, b.row0) and b = (a.row1, b.row1), so a + b holds a's reduction in
// lanes 0-31 and b's in lanes 32-63 — one swap and one add for two rows instead of
// two of each, and the store that follows can use all 64 lanes instead of half.
__device__ __forceinline__ void swap32_reduce2(float& a, float& b) {
    asm("s_nop 1\n\tv_permlane32_swap_b32 %0, %1" : "+v"(a), "+v"(b));
}

// CLEAN_LOGITS=true makes the kernel itself write −inf outside every row's [ks, ke),
// so the output buffer can be allocated with `empty` instead of `full(-inf)`. The
// separate fill pass writes all M*N elements and the kernel then overwrites the valid
// ones, so folding the two saves a full pass over the valid region.
// UNROLL2 processes two K tiles per iteration, issuing both tiles' loads before
// consuming either. The kernel is memory-latency bound (84% SQ wait) at only 2
// waves/SIMD, and 2 waves is what BLOCK_M=4 allows, so the usual fix — more waves —
// is not available. Doubling the loads in flight per wave is: it costs 16 VGPR for the
// second K tile (194 -> 216), and at 2 waves there is room up to 256.
// It is a per-shape win, hence a tuned knob rather than a default: it pays where the
// tile loop is long relative to the number of blocks (M=1024: -4.5% at N=131072,
// -10% at N=1024) and costs 7-9% where blocks are already plentiful (M=4096-8192).
// REVERSE_ROWS dispatches the row groups high-m first. Under causal masking a row
// group's work grows with m, so in natural order the longest-running blocks are
// dispatched LAST and become the tail; reversing starts them first and lets the short
// ones fill in behind. aiter's Gluon kernel does the same thing.
template <int NUM_WARPS, int BLOCK_M, bool CLEAN_LOGITS = true, bool UNROLL2 = false,
          bool REVERSE_ROWS = false>
__global__ __launch_bounds__(NUM_WARPS * WARP_SIZE)
void fp8_mqa_logits_kernel(
    const fp8*   __restrict__ Q_ptr,        // [M, 32, 128]
    const fp8*   __restrict__ K_ptr,        // [N, 128]
    const float* __restrict__ kv_scale_ptr, // [N]
    const float* __restrict__ W_ptr,        // [M, 32]
    const int*   __restrict__ ks_ptr,       // [M]
    const int*   __restrict__ ke_ptr,       // [M]
    float*       __restrict__ logits_ptr,   // [M, N]
    int M, int N, int SplitN
){
    constexpr int TILE_N = 32;              // K tokens per MFMA tile
    constexpr int NACC   = 16;              // C accumulator floats/lane
    using VecOutMFMA = __attribute__((__vector_size__(NACC * sizeof(float)))) float;
    constexpr int UNIT_SCALE = 0x7f;        // E8M0 1.0 — the MFMA scale is unused

    const int l      = threadIdx.x % WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;

    const int bx        = REVERSE_ROWS ? (gridDim.x - 1 - blockIdx.x) : blockIdx.x;
    const int m0        = bx * BLOCK_M;
    const int pid_split = blockIdx.y;
    if (m0 >= M) return;
    const int rw     = l / 16;
    const int col    = l % 16;
    const int nA     = col + 16 * (rw & 1);  // input M/N index (0..31)
    const int dhalf  = rw / 2;               // 0 or 1
    const int jtok   = l % 32;               // output token index
    const int hset   = l / 32;               // output head-set (0/1)

    // ---- this block's rows: Q and W to registers, ks/ke to registers ----
    f8x32 q_reg[BLOCK_M][2];                 // [row][MFMA# (dims 0-63 / 64-127)]
    float w_reg[BLOCK_M][NACC];              // weights for this lane's 16 heads
    int   ks_reg[BLOCK_M], ke_reg[BLOCK_M];

    auto pack = [&](f8x16 lo, f8x16 hi) -> f8x32 {
        f8x32 v; *reinterpret_cast<f8x16*>(&v) = lo; *(reinterpret_cast<f8x16*>(&v) + 1) = hi; return v;
    };

    // The block's rows can straddle a request boundary, so the tile loop covers the
    // union of their ranges and each row masks itself back to [ks, ke).
    int lo = INT_MAX, hi = 0;
    #pragma unroll
    for (int r = 0; r < BLOCK_M; ++r) {
        const int m = m0 + r;
        if (m >= M) { ks_reg[r] = 0; ke_reg[r] = 0; continue; }
        ks_reg[r] = ks_ptr[m];
        ke_reg[r] = ke_ptr[m];
        lo = min(lo, ks_reg[r]);
        hi = max(hi, ke_reg[r]);

        const int qb = ((int64_t)m * NUM_HEADS + nA) * HEAD_SIZE;   // head nA
        q_reg[r][0] = pack(*reinterpret_cast<const f8x16*>(&Q_ptr[qb + dhalf * 16]),
                           *reinterpret_cast<const f8x16*>(&Q_ptr[qb + (2 + dhalf) * 16]));
        q_reg[r][1] = pack(*reinterpret_cast<const f8x16*>(&Q_ptr[qb + (4 + dhalf) * 16]),
                           *reinterpret_cast<const f8x16*>(&Q_ptr[qb + (6 + dhalf) * 16]));
        // This lane's 16 heads are 8*(v/4) + 4*hset + (v%4), i.e. four runs of four
        // CONTIGUOUS heads — so four dwordx4 loads, not sixteen dword loads. The
        // prologue is a real cost on short-range rows, where the tile loop is brief.
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const f32x4 wv = *reinterpret_cast<const f32x4*>(
                &W_ptr[(int64_t)m * NUM_HEADS + 8 * j + 4 * hset]);
            #pragma unroll
            for (int c = 0; c < 4; ++c) w_reg[r][4 * j + c] = wv[c];
        }
    }

    const int nthreads = NUM_WARPS * WARP_SIZE;
    const int tid      = threadIdx.x;
    const float NEG_INF = -__builtin_huge_valf();

    // Rows with an empty range are all -inf; nothing to compute, but they still need
    // filling when we own the fill.
    const bool empty = (lo >= hi);
    lo = empty ? 0 : (lo & ~(TILE_N - 1));   // tile-align so stores stay coalesced

    // Split the tile range across blockIdx.y, then across the warps of the block.
    const int num_tiles   = empty ? 0 : CDIV_(hi - lo, TILE_N);
    const int split_tiles = CDIV_(num_tiles, SplitN);
    const int t_begin     = pid_split * split_tiles;
    const int t_end       = min(num_tiles, t_begin + split_tiles);
    const int tile_end    = min(N, lo + num_tiles * TILE_N);

    // −inf for the parts of our rows no tile covers: [0, lo) and [tile_end, N).
    // Only split 0 does it, so the work is not repeated across splits. `lo` is
    // tile-aligned and tile_end is either N or a multiple of TILE_N, so both spans
    // start 16-byte aligned and the vector store needs no head peeling.
    if constexpr (CLEAN_LOGITS) {
        if (pid_split == 0) {
            const f32x4 vneg = {NEG_INF, NEG_INF, NEG_INF, NEG_INF};
            // `a` is tile-aligned, but the 16-byte store also needs the ROW to start
            // 4-element aligned, which only holds when N is a multiple of 4.
            auto fill_range = [&](float* row, int64_t row_base, int a, int b) {
                if (((row_base + a) & 3) == 0) {
                    const int vb = a + ((b - a) & ~3);
                    for (int i = a + tid * 4; i < vb; i += nthreads * 4)
                        *reinterpret_cast<f32x4*>(row + i) = vneg;
                    for (int i = vb + tid; i < b; i += nthreads)
                        row[i] = NEG_INF;
                } else {
                    for (int i = a + tid; i < b; i += nthreads)
                        row[i] = NEG_INF;
                }
            };
            #pragma unroll 1
            for (int r = 0; r < BLOCK_M; ++r) {
                const int m = m0 + r;
                if (m >= M) break;
                const int64_t row_base = (int64_t)m * N;
                float* row = logits_ptr + row_base;
                // `lo` is a cu_start, and a window may legally start past the end
                // of KV; the row is then entirely -inf. Clamp, or the fill runs
                // straight into the next row (the store is a raw buffer offset,
                // there is no hardware OOB net).
                fill_range(row, row_base, 0, min(lo, N));
                fill_range(row, row_base, tile_end, N);
            }
        }
    }
    if (empty) return;

    // The warps of a block split the TILE range, not the rows: that keeps several
    // independent K streams in flight per block. Splitting rows across warps instead
    // (so K is read once per BLOCK_M*NUM_WARPS rows rather than once per BLOCK_M)
    // trades that away and measures 1.5-23% SLOWER — the latency hiding is worth more
    // than the L2 traffic it saves.
    struct Tile { f8x32 b0, b1; float scale; int token_base; };

    auto load_tile = [&](int t) -> Tile {
        Tile tl;
        tl.token_base = lo + t * TILE_N;
        const int token = tl.token_base + nA;
        // K tile → registers: 32 tokens × 128 dims = 4 KB per wave, 4× dwordx4.
        // The out-of-range lanes CLAMP rather than branch. Guarding the load with
        // `if (token < N) ... else vB = {}` merges two values into the MFMA operands,
        // and that PHI made the compiler copy all 16 loaded registers into place —
        // ~18 v_mov per tile. Clamping keeps the loads in bounds and the junk they
        // return is discarded by the store's range test, which those lanes fail anyway.
        const int64_t kb = (int64_t)min(token, N - 1) * HEAD_SIZE;
        tl.b0 = pack(*reinterpret_cast<const f8x16*>(&K_ptr[kb + dhalf * 16]),
                     *reinterpret_cast<const f8x16*>(&K_ptr[kb + (2 + dhalf) * 16]));
        tl.b1 = pack(*reinterpret_cast<const f8x16*>(&K_ptr[kb + (4 + dhalf) * 16]),
                     *reinterpret_cast<const f8x16*>(&K_ptr[kb + (6 + dhalf) * 16]));
        tl.scale = kv_scale_ptr[min(tl.token_base + jtok, N - 1)];
        return tl;
    };

    auto compute_tile = [&](const Tile& tl) {
        // Σ_v relu(C_v) · w_v for one row — the cross-lane head reduction is done
        // afterwards, two rows at a time.
        auto row_partial = [&](int r, const f8x32& a0, const f8x32& a1) -> float {
            VecOutMFMA vC = {};
            vC = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                a0, tl.b0, vC, 0, 0, 0, UNIT_SCALE, 0, UNIT_SCALE);
            vC = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                a1, tl.b1, vC, 0, 0, 0, UNIT_SCALE, 0, UNIT_SCALE);
            float p = 0.0f;
            #pragma unroll
            for (int v = 0; v < NACC; ++v)
                p += relu_fast(vC[v]) * w_reg[r][v];
            return p;
        };

        // Rows in pairs: after swap32_reduce2 lanes 0-31 carry row `r` and lanes
        // 32-63 row `r+1`, so one store instruction covers both. kv_scale depends
        // only on the token (l%32), which both halves share, so it applies once
        // after the reduction.
        #pragma unroll
        for (int r = 0; r < BLOCK_M; r += 2) {
            float p0 = row_partial(r, q_reg[r][0], q_reg[r][1]);
            float p1 = 0.0f;
            if (r + 1 < BLOCK_M) p1 = row_partial(r + 1, q_reg[r + 1][0], q_reg[r + 1][1]);
            swap32_reduce2(p0, p1);
            const float res = (p0 + p1) * tl.scale;

            // This lane's half selects which of the two rows it stores.
            const int r_next  = (r + 1 < BLOCK_M) ? r + 1 : r;   // folds after unroll
            const int r_sel   = hset ? r_next : r;
            const int ks_sel  = hset ? ks_reg[r_next] : ks_reg[r];
            const int ke_sel  = hset ? ke_reg[r_next] : ke_reg[r];
            const bool row_ok = (hset == 0 || r + 1 < BLOCK_M) && (m0 + r_sel < M);
            const int abs_pos = tl.token_base + jtok;
            const bool valid = row_ok && abs_pos >= ks_sel && abs_pos < ke_sel;

            if constexpr (CLEAN_LOGITS) {
                // Out-of-range lanes write -inf rather than being masked off, so the
                // whole [lo, tile_end) span is covered without a separate fill pass.
                if (row_ok && abs_pos < N)
                    logits_ptr[(int64_t)(m0 + r_sel) * N + abs_pos] =
                        valid ? res : NEG_INF;
            } else if (valid && abs_pos < N) {
                logits_ptr[(int64_t)(m0 + r_sel) * N + abs_pos] = res;
            }
        }
    };

    int t = t_begin + warpId;
    if constexpr (UNROLL2) {
        for (; t + NUM_WARPS < t_end; t += 2 * NUM_WARPS) {
            const Tile a = load_tile(t);                 // both tiles' loads issue
            const Tile b = load_tile(t + NUM_WARPS);     // before either is consumed
            compute_tile(a);
            compute_tile(b);
        }
    }
    for (; t < t_end; t += NUM_WARPS)
        compute_tile(load_tile(t));
}

}  // namespace mqa_dense

// ============================================================================
// Host dispatch
// ============================================================================
static void dispatch_fp8_mqa_logits(
    const fp8* d_q, const fp8* d_k, const float* d_scale, const float* d_w,
    const int* d_ks, const int* d_ke, float* d_out,
    int M, int N, int block_m, int split_n, int num_warps, bool clean_logits,
    bool unroll2, bool rev, hipStream_t stream
) {
    const dim3 grid = dim3(CDIV_(M, block_m), split_n);

    #define LAUNCH_MQA_R(NW, BM, U2, RV)                                            \
        if (clean_logits)                                                           \
            mqa_dense::fp8_mqa_logits_kernel<NW, BM, true, U2, RV><<<grid, NW * WARP_SIZE, 0, stream>>>( \
                d_q, d_k, d_scale, d_w, d_ks, d_ke, d_out, M, N, split_n);          \
        else                                                                        \
            mqa_dense::fp8_mqa_logits_kernel<NW, BM, false, U2, RV><<<grid, NW * WARP_SIZE, 0, stream>>>( \
                d_q, d_k, d_scale, d_w, d_ks, d_ke, d_out, M, N, split_n);

    #define LAUNCH_MQA_C(NW, BM, U2)                                                \
        if (rev) { LAUNCH_MQA_R(NW, BM, U2, true) }                                 \
        else     { LAUNCH_MQA_R(NW, BM, U2, false) }

    #define LAUNCH_MQA(NW, BM)                                                      \
        if (unroll2) { LAUNCH_MQA_C(NW, BM, true) }                                 \
        else         { LAUNCH_MQA_C(NW, BM, false) }

    #define LAUNCH_MQA_BM(NW)                                                       \
        switch (block_m) {                                                          \
            case 1: { LAUNCH_MQA(NW, 1) break; }                                    \
            case 2: { LAUNCH_MQA(NW, 2) break; }                                    \
            case 3: { LAUNCH_MQA(NW, 3) break; }                                    \
            case 4: { LAUNCH_MQA(NW, 4) break; }                                    \
            case 6: { LAUNCH_MQA(NW, 6) break; }                                    \
            case 8: { LAUNCH_MQA(NW, 8) break; }                                    \
            default:                                                                \
                throw std::runtime_error("Unsupported block_m="                     \
                    + std::to_string(block_m) + ". Supported: 1,2,3,4,6,8");        \
        }

    switch (num_warps) {
        case 1: { LAUNCH_MQA_BM(1) break; }
        case 2: { LAUNCH_MQA_BM(2) break; }
        case 4: { LAUNCH_MQA_BM(4) break; }
        case 8: { LAUNCH_MQA_BM(8) break; }
        default:
            throw std::runtime_error("Unsupported num_warps=" + std::to_string(num_warps)
                + ". Supported: 1, 2, 4, 8");
    }
    #undef LAUNCH_MQA_BM
    #undef LAUNCH_MQA
    #undef LAUNCH_MQA_C
    #undef LAUNCH_MQA_R
}

// ============================================================================
// Torch-facing entry point
// ============================================================================
torch::Tensor fp8_mqa_logits(
    torch::Tensor q_fp8,        // [M, 32, 128] fp8
    torch::Tensor k_fp8,        // [N, 128]     fp8
    torch::Tensor kv_scale,     // [N]          fp32
    torch::Tensor weights,      // [M, 32]      fp32
    torch::Tensor cu_seqlen_ks, // [M]          int32
    torch::Tensor cu_seqlen_ke, // [M]          int32
    int64_t BlockM,
    int64_t SplitN,
    int64_t num_warps,
    int64_t TotalCuCount,
    bool clean_logits,
    int64_t Unroll2,      // -1 = host heuristic, 0 = off, 1 = on
    int64_t ReverseRows,  // -1 = host heuristic, 0 = off, 1 = on
    std::optional<torch::Tensor> out
) {
    const int M = q_fp8.size(0);
    const int N = k_fp8.size(0);

    TORCH_CHECK(q_fp8.size(1) == NUM_HEADS && q_fp8.size(2) == HEAD_SIZE,
                "Only n_heads=32, head_dim=128 supported");
    TORCH_CHECK(k_fp8.size(1) == HEAD_SIZE, "K must be [N, 128]");
    TORCH_CHECK(q_fp8.is_contiguous() && k_fp8.is_contiguous(),
                "q_fp8 and k_fp8 must be contiguous");

    const int total_cu = static_cast<int>(TotalCuCount);
    int block_m    = static_cast<int>(BlockM);
    int split_n    = static_cast<int>(SplitN);
    int num_warps_ = static_cast<int>(num_warps);
    bool unroll2   = Unroll2     > 0;
    bool rev       = ReverseRows > 0;

    // ---- autotune (block_m, num_warps, split_n) ----
    // block_m rows share one K stream, at 32 VGPR per row (16 Q + 16 W). Unlike the
    // decode kernel — where occupancy rules and R>3 always lost — prefill re-reads K
    // once per row-block out of L2, so the sweep prefers a LARGE block_m wherever the
    // per-row K range is long, and only backs off when the ranges are short enough
    // that the prologue and occupancy dominate. N is the proxy for that range (for a
    // single request the average row scans ~N/2).
    // block_m=4 won 9 of 12 shapes: two row-pairs, so the paired-row reduction
    // always has both halves busy, and 4*32=128 VGPR of Q+W still leaves room for
    // the accumulator and the K tile. block_m=8 is not an option -- it needs 256
    // VGPR and spills 113.
    // Refit against a 12-shape sweep (bench + deploy configs), 2026-08-26.
    //
    // reverse_rows is the only knob that moved: on for N > 2048, worth +1% to
    // +7% there and free -- it only reorders block dispatch, costing no register
    // and no instruction in the loop. Under causal masking a row group's work
    // grows with m, so in natural order the longest-running blocks are
    // dispatched LAST and become the tail; reversing starts them first and lets
    // the short ones fill in behind.
    //
    // Everything else stayed. The sweep's per-shape winners suggested num_warps=2
    // and unroll2 for N <= 2048, but those values only win in combination with a
    // block_m the two short shapes disagree on: measured pairwise, num_warps=2 is
    // +2% at N=1024 and -30% at N=2048. No setting beats the current one on both,
    // so the short-window branch is left alone.
    //
    // The multi-request shapes want num_warps=2 where the single-request shape of
    // the SAME (M, N) wants 8 -- the difference is in cu_seqlen_ks/ke, which live
    // on the device. A host heuristic keyed on (M, N) cannot separate them; that
    // gain needs the caller to pass num_warps explicitly.
    if (block_m    <= 0) block_m    = 4;
    if (num_warps_ <= 0) num_warps_ = (N <= 2048) ? 4 : 8;
    if (Unroll2     < 0) unroll2    = false;
    if (ReverseRows < 0) rev        = (N > 2048);
    if (split_n    <= 0) {
        // Enough blocks to fill the machine: aim for ~2 workgroups per CU.
        const int row_blocks = CDIV_(M, block_m);
        split_n = std::max(1, (total_cu * 2 + row_blocks - 1) / row_blocks);
    }

    // Output contract matches aiter's fp8_mqa_logits(clean_logits=True): every
    // position outside a row's [ks, ke) reads -inf. The kernel writes those itself,
    // so no separate fill pass — aiter's `torch.full` writes all M*N elements and the
    // kernel then overwrites the valid ones, paying for the valid region twice.
    // clean_logits=False skips -inf entirely; only safe when the consumer is range
    // aware (top_k_per_row_prefill is, it takes ks/ke).
    // Either way the buffer starts uninitialised: with clean_logits the kernel covers
    // every element, without it the caller has opted out of -inf altogether.
    // A caller-provided buffer is used as-is; this is what lets a test poison it and
    // prove the kernel writes every position it claims to.
    torch::Tensor out_logits =
        out.has_value()
            ? out.value()
            : torch::empty({M, N}, torch::dtype(torch::kFloat32).device(q_fp8.device()));
    TORCH_CHECK(out_logits.scalar_type() == torch::kFloat32 &&
                    out_logits.is_contiguous() && out_logits.size(0) == M &&
                    out_logits.size(1) == N,
                "out must be a contiguous f32 [M, N] tensor");

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(device_of(out_logits));
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

    dispatch_fp8_mqa_logits(
        static_cast<fp8*>(q_fp8.data_ptr()),
        static_cast<fp8*>(k_fp8.data_ptr()),
        kv_scale.data_ptr<float>(),
        weights.data_ptr<float>(),
        cu_seqlen_ks.data_ptr<int>(),
        cu_seqlen_ke.data_ptr<int>(),
        out_logits.data_ptr<float>(),
        M, N, block_m, split_n, num_warps_, clean_logits, unroll2, rev,
        stream);

    return out_logits;
}
