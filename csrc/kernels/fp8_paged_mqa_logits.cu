// SPDX-License-Identifier: MIT
// Decode-phase FP8 paged MQA indexer logits kernel, gfx950.
#include "fp8_paged_mqa_logits.h"

// ============================================================================
// fp8_paged_mqa_logits — decode-phase indexer kernel (GLM-5-FP8), CDNA4 / gfx950
//
// K is streamed from the paged cache straight to registers: 32 heads x 32 tokens
// per V_MFMA_SCALE_F32_32X32X64_F8F6F4 tile, no LDS staging on the K path.
//
// Tunable parameters:
//   NUM_WARPS       — template: warps per block (occupancy)
//   CHUNK_K         — template: context positions per chunk; the unit SplitKV
//                     hands to a block, so it sets the grid.y granularity
//   KV_BLOCK_SIZE   — template: paged block size. 1 reads the plain co-packed
//                     cache, 64 the shuffle_weight((16,16)) preshuffled one
//   ROWS_PER_BLOCK  — template: next_n rows sharing one K stream
//   SplitKV         — runtime:  parallelism over context length (grid size)
// ============================================================================

using fp8 = __hip_fp8_storage_t;

constexpr int WARP_SIZE  = 64;
constexpr int NUM_HEADS  = 32;
constexpr int HEAD_SIZE  = 128;

#define CDIV(a, b) ((a) + (b) - 1) / b
// ============================================================================
// Two epilogue choices that are worth spelling out, both measured against the
// straightforward version (48 v_mul + 48 v_max + 48 fma + 3 ds_bpermute per tile
// at NW8/CK64/KB64/R3):
//   (a) hoist the per-token scale OUT of the relu/weight loop. kv_scale >= 0 (it is
//       a quantisation scale), so relu(c*s)*w == s*relu(c)*w — this also matches the
//       torch reference exactly (it applies the scale after the head reduction).
//       Kills 16 v_mul per row per tile (48 at R=3), leaving one v_mul per row.
//   (b) replace __shfl_down(x,32) (→ ds_bpermute_b32, LDS round-trip + lgkmcnt wait,
//       ~11% of stalls per the ATT trace) with V_PERMLANE32_SWAP_B32, a pure-VALU
//       cross-row swap. Semantics (probed on HW): `v_permlane32_swap_b32 a, b`
//       swaps a.row1 (lanes 32-63) with b.row0 (lanes 0-31), so after the swap
//       b.row0 holds a's old lanes 32-63 — exactly the shfl_down(32) operand.
//       Emitted as inline asm: the clang builtin mis-models its second output
//       (both results map to the same VGPR).
// ============================================================================
namespace mqa_paged {

using f8x16 = __attribute__((__vector_size__(16))) fp8;
using f8x32 = __attribute__((__vector_size__(32))) fp8;

// lanes 0-31 receive x[lane+32]; lanes 32-63 get garbage (unused).
// `v_permlane32_swap_b32 a, b` → a = (a.row0, b.row0), b = (a.row1, b.row1), so b's
// low half ends up holding a's old high half — the shfl_down(32) operand, in VALU.
// The `s_nop 1` is mandatory: a VALU write of the source needs 2 wait states before
// a permlane-swap read of it (this matches the nops LLVM emits around the builtin).
// Without it the swap reads stale data and the logits come out wrong. The builtin
// itself is unusable here — clang lowers both of its results to the same VGPR.
// relu(x). Compiled with -fno-honor-nans (set for this module in
// aiter/jit/optCompilerConfig.json) this is a single
// `v_max_f32 x, 0, x`. Without that flag LLVM must assume x may be a signalling NaN
// and emits an IEEE canonicalize `v_max_f32 x, x, x` first — two VALU per value, and
// this loop runs NACC of them per row per tile, ~27% of all VALU in the kernel.
// Inline asm would also give one instruction but is NOT safe here: the hazard
// recognizer does not treat inline asm as a reader of the MFMA destination, so the
// mandatory post-MFMA s_nops disappear and the accumulator is read before it lands.
__device__ __forceinline__ float relu_fast(float x) { return fmaxf(x, 0.0f); }

__device__ __forceinline__ float swap32_hi(float x) {
    float hi;
    asm("s_nop 1\n\tv_permlane32_swap_b32 %0, %1" : "+v"(x), "=&v"(hi));
    return hi;
}

template <int NUM_WARPS, int CHUNK_K, int KV_BLOCK_SIZE, int ROWS_PER_BLOCK>
__global__ __launch_bounds__(NUM_WARPS * WARP_SIZE)
void fp8_paged_mqa_logits_kernel(
    const fp8*   __restrict__ Q_ptr,
    const fp8*   __restrict__ kv_cache_ptr,
    const float* __restrict__ weights_ptr,
    const int*   __restrict__ context_lens,
    const int*   __restrict__ block_tables,
    float*       __restrict__ logits_ptr,
    int batch_size, int next_n,
    int max_blocks_per_seq, int max_model_len, int index_dim,
    int SplitKV
){
    constexpr int TILE_N = 32;                 // tokens per tile
    constexpr int NACC   = 16;                 // C accumulator floats/lane
    using VecOutMFMA = __attribute__((__vector_size__(NACC * sizeof(float)))) float;
    constexpr bool PRESHUFFLE = (KV_BLOCK_SIZE > 1);
    constexpr int UNIT_SCALE = 0x7f;

    const int pid_batch    = blockIdx.x;
    const int pid_split_kv = blockIdx.y;
    const int row_start    = blockIdx.z * ROWS_PER_BLOCK;
    if (pid_batch >= batch_size || row_start >= next_n) return;

    const int ctx_len      = context_lens[pid_batch];
    const int ctx_chunks   = CDIV(ctx_len, CHUNK_K);
    const int split_chunks = CDIV(ctx_chunks, SplitKV);
    const int split_start  = pid_split_kv * split_chunks * CHUNK_K;
    const int split_end    = min(ctx_len, split_start + split_chunks * CHUNK_K);
    if (split_start >= ctx_len) return;

    const int l    = threadIdx.x % WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;
    const int rw    = l / 16;
    const int col   = l % 16;
    const int nA    = col + 16 * (rw & 1);     // input M/N index (0..31)
    const int dhalf = rw / 2;                  // 0 or 1
    const int jtok  = l % 32;                  // output token index
    const int hset  = l / 32;                  // output head-set (0/1)

    f8x32 q_reg[ROWS_PER_BLOCK][2];            // [row][MFMA# (dims 0-63 / 64-127)]
    float w_reg[ROWS_PER_BLOCK][NACC];         // weights for this lane's 16 output heads

    const int  q_row0 = pid_batch * next_n + row_start;
    const int* bt     = block_tables + (int64_t)pid_batch * max_blocks_per_seq;

    auto pack = [&](f8x16 lo, f8x16 hi) -> f8x32 {
        f8x32 v; *reinterpret_cast<f8x16*>(&v) = lo; *(reinterpret_cast<f8x16*>(&v) + 1) = hi; return v;
    };
    // 16 contiguous fp8 of K at (token, 16-dim run d_hi) given resolved block blk.
    auto kv16 = [&](int blk, int token, int d_hi) -> f8x16 {
        const int off = token % KV_BLOCK_SIZE;
        const int64_t base = (int64_t)blk * (KV_BLOCK_SIZE * index_dim);
        int64_t a;
        if constexpr (PRESHUFFLE)
            a = base + (int64_t)d_hi * 256 + (off & 15) * 16
              + ((off & (KV_BLOCK_SIZE - 1)) >> 4) * 16 * HEAD_SIZE;
        else
            a = base + (int64_t)off * HEAD_SIZE + d_hi * 16;
        return *reinterpret_cast<const f8x16*>(kv_cache_ptr + a);
    };
    auto scale_addr = [&](int blk, int token) -> int64_t {
        const int off = token % KV_BLOCK_SIZE;
        return (int64_t)blk * (KV_BLOCK_SIZE * index_dim)
             + (int64_t)KV_BLOCK_SIZE * HEAD_SIZE + (int64_t)off * 4;
    };
    auto load_blk = [&](int tt) -> int {
        if constexpr (KV_BLOCK_SIZE == 1) {
            const int tok = split_start + tt * TILE_N + nA;
            return (tok < ctx_len) ? bt[tok] : -1;
        } else {
            const int tb = split_start + tt * TILE_N;
            return (tb < ctx_len) ? bt[tb / KV_BLOCK_SIZE] : -1;
        }
    };

    // ---- Q → registers; W → registers (16 output heads for this lane) ----
    #pragma unroll
    for (int r = 0; r < ROWS_PER_BLOCK; ++r) {
        if (row_start + r >= next_n) break;
        const int head = nA;                                     // input M index
        const int qb = (q_row0 + r) * NUM_HEADS * HEAD_SIZE + head * HEAD_SIZE;
        q_reg[r][0] = pack(*reinterpret_cast<const f8x16*>(&Q_ptr[qb + dhalf * 16]),
                           *reinterpret_cast<const f8x16*>(&Q_ptr[qb + (2 + dhalf) * 16]));
        q_reg[r][1] = pack(*reinterpret_cast<const f8x16*>(&Q_ptr[qb + (4 + dhalf) * 16]),
                           *reinterpret_cast<const f8x16*>(&Q_ptr[qb + (6 + dhalf) * 16]));
        #pragma unroll
        for (int v = 0; v < NACC; ++v) {
            const int i = 8 * (v / 4) + 4 * hset + (v % 4);      // output head for this vgpr
            w_reg[r][v] = weights_ptr[(q_row0 + r) * NUM_HEADS + i];
        }
    }

    const int num_tiles = CDIV(split_end - split_start, TILE_N);
    int blk_cur = load_blk(warpId);
    for (int t = warpId; t < num_tiles; t += NUM_WARPS) {
        const int blk_next = load_blk(t + NUM_WARPS);
        const int token_base = split_start + t * TILE_N;
        const int token      = token_base + nA;

        f8x32 vB0, vB1;
        if (blk_cur >= 0 && token < ctx_len) {
            vB0 = pack(kv16(blk_cur, token, dhalf), kv16(blk_cur, token, 2 + dhalf));
            vB1 = pack(kv16(blk_cur, token, 4 + dhalf), kv16(blk_cur, token, 6 + dhalf));
        } else { vB0 = f8x32{}; vB1 = f8x32{}; }

        const int stoken = token_base + jtok;
        int blk_s = blk_cur;
        if constexpr (KV_BLOCK_SIZE == 1) blk_s = (stoken < ctx_len) ? bt[stoken] : -1;
        const float kv_scale = (stoken < ctx_len && blk_s >= 0)
            ? *reinterpret_cast<const float*>(kv_cache_ptr + scale_addr(blk_s, stoken)) : 0.0f;
        blk_cur = blk_next;

        #pragma unroll
        for (int r = 0; r < ROWS_PER_BLOCK; ++r) {
            if (row_start + r >= next_n) break;
            VecOutMFMA vC = {};
            vC = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                q_reg[r][0], vB0, vC, 0, 0, 0, UNIT_SCALE, 0, UNIT_SCALE);
            vC = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                q_reg[r][1], vB1, vC, 0, 0, 0, UNIT_SCALE, 0, UNIT_SCALE);

            // Scale hoisted out of the relu: kv_scale >= 0 (it is a quantisation
            // scale) so relu(c*s)*w == s*relu(c)*w, which is also exactly what the
            // torch reference computes. Saves NACC v_mul per row per tile.
            float partial = 0.0f;
            #pragma unroll
            for (int v = 0; v < NACC; ++v)
                partial += relu_fast(vC[v]) * w_reg[r][v];
            // lanes l and l+32 hold the same token, hence the same kv_scale, so the
            // scale applies once, after the head reduction.
            partial = (partial + swap32_hi(partial)) * kv_scale;

            if (l < 32) {
                const int abs_pos = token_base + jtok;
                const int causal_limit = ctx_len - next_n + row_start + r;
                if (abs_pos < ctx_len && abs_pos <= causal_limit && abs_pos < max_model_len)
                    logits_ptr[(int64_t)(q_row0 + r) * max_model_len + abs_pos] = partial;
            }
        }
    }
}

}  // namespace mqa_paged
// ============================================================================
// Host dispatch
// ============================================================================
static void dispatch_fp8_paged_mqa_logits(
    const fp8* d_q, const fp8* d_kv, const float* d_w,
    const int* d_ctx, const int* d_bt, float* d_out,
    int batch_size, int next_n,
    int max_blocks_per_seq, int max_model_len, int index_dim,
    int ChunkK, int SplitKV, int num_warps, int kv_block_size,
    int rows_per_block, hipStream_t stream
) {
    // grid = dim3(batch, SplitKV, num_groups). Each block owns ROWS_PER_BLOCK
    // (=R) consecutive next_n rows; num_groups = ceil(next_n / R).
    const int num_groups = CDIV(next_n, rows_per_block);

    const dim3 grid = dim3(batch_size, SplitKV, num_groups);

    #define LAUNCH(NW, CK, KB, R)                                                       \
        mqa_paged::fp8_paged_mqa_logits_kernel<NW, CK, KB, R>                          \
            <<<grid, NW * WARP_SIZE, 0, stream>>>(                                       \
                d_q, d_kv, d_w, d_ctx, d_bt, d_out,                                      \
                batch_size, next_n, max_blocks_per_seq, max_model_len, index_dim, SplitKV);

    // R folds in as a compile-time constant so q_reg/w_reg stay in registers.
    #define LAUNCH_NN(NW, CK, KB)                                                       \
        switch (rows_per_block) {                                                       \
            case 1: { LAUNCH(NW, CK, KB, 1) break; }                                    \
            case 2: { LAUNCH(NW, CK, KB, 2) break; }                                    \
            case 3: { LAUNCH(NW, CK, KB, 3) break; }                                    \
            case 4: { LAUNCH(NW, CK, KB, 4) break; }                                    \
            case 6: { LAUNCH(NW, CK, KB, 6) break; }                                    \
            case 8: { LAUNCH(NW, CK, KB, 8) break; }                                    \
            default:                                                                    \
                throw std::runtime_error("Unsupported rows_per_block="                  \
                    + std::to_string(rows_per_block) + ". Supported: 1,2,3,4,6,8");     \
        }

    // block_size folds into the kernel as a compile-time constant so the
    // token->slot div/mod become shifts/ands; {1, 64} match the indexer's
    // ROCm-supported KV block sizes.
    #define LAUNCH_KB(NW, CK)                                                           \
        switch (kv_block_size) {                                                        \
            case 1:  { LAUNCH_NN(NW, CK, 1)  break; }                                   \
            case 64: { LAUNCH_NN(NW, CK, 64) break; }                                   \
            default:                                                                    \
                throw std::runtime_error("Unsupported kv_block_size="                   \
                    + std::to_string(kv_block_size) + ". Supported: 1, 64");            \
        }

    #define LAUNCH_NW(NW)                                                               \
        switch (ChunkK) {                                                               \
            case 64:  { LAUNCH_KB(NW, 64)  break; }                                     \
            case 128: { LAUNCH_KB(NW, 128) break; }                                     \
            case 256: { LAUNCH_KB(NW, 256) break; }                                     \
            default:                                                                    \
                throw std::runtime_error("Unsupported ChunkK=" + std::to_string(ChunkK) \
                    + ". Supported: 64, 128, 256");                                     \
        }

    switch (num_warps) {
        case 2: { LAUNCH_NW(2) break; }
        case 4: { LAUNCH_NW(4) break; }
        case 8: { LAUNCH_NW(8) break; }
        default:
            throw std::runtime_error("Unsupported num_warps=" + std::to_string(num_warps)
                + ". Supported: 2, 4, 8");
    }
    #undef LAUNCH_NW
    #undef LAUNCH_KB
    #undef LAUNCH_NN
    #undef LAUNCH
}

// ============================================================================
// Torch-facing entry point
// ============================================================================
torch::Tensor fp8_paged_mqa_logits(
    torch::Tensor q_fp8,            // [batch, next_n, 32, 128]
    torch::Tensor kv_cache_fp8,     // [num_blocks, block_size, 1, index_dim]
    torch::Tensor weights,          // [batch*next_n, 32]
    torch::Tensor context_lens,     // [batch]
    torch::Tensor block_tables,     // [batch, max_blocks_per_seq]
    int64_t max_model_len,
    int64_t ChunkK,
    int64_t SplitKV,
    int64_t num_warps,
    int64_t TotalCuCount,
    int64_t RowsPerBlock,
    std::optional<torch::Tensor> out
) {
    const int batch_size         = q_fp8.size(0);
    const int next_n             = q_fp8.size(1);
    const int n_heads            = q_fp8.size(2);
    const int head_dim           = q_fp8.size(3);
    const int kv_block_size      = kv_cache_fp8.size(1);
    const int index_dim          = kv_cache_fp8.size(3);
    const int max_blocks_per_seq = block_tables.size(1);
    const int total_cu_i         = static_cast<int>(TotalCuCount);

    TORCH_CHECK(n_heads == NUM_HEADS && head_dim == HEAD_SIZE,
                "Only n_heads=32, head_dim=128 supported");
    TORCH_CHECK(kv_cache_fp8.is_contiguous(),
                "kv_cache_fp8 must be contiguous for paged MQA logits");

    // ---- autotune ChunkK, num_warps, R, SplitKV ----
    // From the gfx950 reliable (cache-busting) sweep, the best config tracks the
    // total K footprint tot = batch*ctx_len (redundancy vs occupancy trade):
    //   tot <  32K  → R=1  (tiny: maximise occupancy, K-redundancy is ~free/L2)
    //   tot < 158K  → R=2  (mid: occupancy still dominates)
    //   else        → R=3  (bandwidth-bound: minimise redundant K reads; plenty
    //                       of work for occupancy). R>=4/6 never wins (q_reg
    //                       register pressure kills waves — occupancy is the only
    //                       HBM-latency hider; prefetch/large-R both hurt).
    //   CHUNK_K = 128 for long ctx (more splittable tiles) or R=1; else 256.
    //   num_warps = 8 except the tiny R=1 case (4).
    const int mml_i = static_cast<int>(max_model_len);
    const int64_t tot = (int64_t)batch_size * mml_i;

    int chunk_k_i      = static_cast<int>(ChunkK);
    int num_warps_i    = static_cast<int>(num_warps);
    int rows_per_block = static_cast<int>(RowsPerBlock);
    int split_kv_i     = static_cast<int>(SplitKV);

    // Fitted against a 35-shape tuning sweep.
    // num_warps=4 is the broad optimum, with 8 only in a mid band; large total
    // footprints want the small chunk.
    if (rows_per_block <= 0) rows_per_block = (tot < 90000) ? 1 : (tot < 280000 ? 2 : 3);
    if (rows_per_block > next_n) rows_per_block = next_n;
    if (num_warps_i <= 0) num_warps_i = (tot >= 80000 && tot < 280000) ? 8 : 4;
    if (chunk_k_i   <= 0)
        chunk_k_i = (mml_i >= 50000) ? (tot >= 400000 ? 64 : 128)
                                     : (tot < 280000 ? 256 : 64);

    if (split_kv_i <= 0) {
        const int ctx_chunks_ub = std::max(1, (mml_i + chunk_k_i - 1) / chunk_k_i);
        const int num_groups = (next_n + rows_per_block - 1) / rows_per_block;
        const int units = std::max(1, batch_size * num_groups);
        const int target_blocks = total_cu_i * 2;   // ~2 blocks/CU (tail + latency)
        split_kv_i = std::max(1, std::min(ctx_chunks_ub,
            (target_blocks + units - 1) / units));
    }

    // The kernel writes only the causal window, so the -inf outside it is the
    // caller's -- same contract as deepgemm_fp8_paged_mqa_logits. When no buffer is
    // handed in, allocate one; those positions are then simply undefined.
    torch::Tensor out_logits =
        out.has_value() ? out.value()
                        : torch::empty({batch_size * next_n,
                                        static_cast<int64_t>(max_model_len)},
                                       torch::dtype(torch::kFloat32).device(q_fp8.device()));
    TORCH_CHECK(out_logits.scalar_type() == torch::kFloat32 &&
                    out_logits.is_contiguous() &&
                    out_logits.size(0) == batch_size * next_n &&
                    out_logits.size(1) == max_model_len,
                "out must be a contiguous f32 [batch*next_n, max_model_len] tensor");

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(device_of(out_logits));
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

    dispatch_fp8_paged_mqa_logits(
        static_cast<fp8*>(q_fp8.data_ptr()),
        static_cast<fp8*>(kv_cache_fp8.data_ptr()),
        static_cast<float*>(weights.data_ptr()),
        static_cast<int*>(context_lens.data_ptr()),
        static_cast<int*>(block_tables.data_ptr()),
        out_logits.data_ptr<float>(),
        batch_size, next_n, max_blocks_per_seq,
        static_cast<int>(max_model_len), index_dim,
        chunk_k_i, split_kv_i,
        num_warps_i, kv_block_size, rows_per_block, stream);

    return out_logits;
}
