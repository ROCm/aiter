// Fused MoE routing preamble -- one launch, one grid barrier.
//
// Replaces the 4-kernel decode preamble (grouped_topk -> moe_sort P0 -> P23 ->
// fused_mx_quant). Wins by keeping topk scores in registers instead of
// round-tripping them, and by quantizing each token once instead of once per
// sorted row, which the stock path repeats topk times.
//
// Phase 1 -- block owns tokens t = bid, bid+GRID, ..; no cross-block deps
//   wave 0 : biased-sigmoid top-k -> topk_ids/topk_weights
//   all    : MXFP4 quant of hidden[t] -> out[t], group e8m0 -> tok_scale[t]
// ---- grid barrier (the only one) ----
// Phase 2 : reload topk into LDS, histogram + scan -> per-expert base offsets
// Phase 3 : each block takes a unit-aligned slice of the sorted output and per
//           expert ballot-ranks its routed ids -> sorted_ids/weights/expert_ids,
//           pads rows, scatters the token's e8m0 scales into swizzled layout.
//
// Phase 2 runs redundantly in every block on purpose: a few hundred LDS
// elements is cheaper than a second grid barrier.

#include <climits>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "opus/opus.hpp"
#include "warp_sort.h" // aiter::mov_dpp_
#include "quant_kernels.cu" // device helpers: scaled_quant_vgpr_impl, load_vector_nbytes,
                            // multithread_reduce, fp4_f32_to_e8m0_scale, mx_scale_shuffle_idx

namespace aiter {
namespace fmr {

static constexpr int kWaveSize = 64;

// Scales per thread in the phase-3 scatter, and the alignment the scale row
// must satisfy: mx_scale_shuffle_idx's y terms are periodic in this, so it is
// the smallest span over which the scatter's OFF[] table is complete.
static constexpr int kScalesPerThread = 8;

// Token count at and above which the split launch beats the grid barrier.
static constexpr int kSplitMinTokens = 104;

// Self-resetting grid barrier, agent-scope acquire/release.
//
// Sense-flip (as in CUDA cooperative groups): block 0 contributes
// 0x80000000-(N-1), every other block 1, so the counter's high bit flips
// exactly when all N blocks arrive and the counter walks 0 -> 2^31 -> 0 on its
// own. No reset needed, so the host never memsets the semaphore.
//
// The release on arrive and the acquire after the spin are the data fence, so
// no __threadfence(); __syncthreads() on both sides extends it to the block.
// Split in two so barrier-independent LDS work runs while thread 0 spins.
__device__ __forceinline__ void grid_arrive(unsigned int* sem, unsigned int nblocks,
                                            unsigned int* s_old)
{
    __syncthreads(); // all waves finished phase 1 before we publish the release
    if(threadIdx.x == 0)
    {
        unsigned int contrib = (blockIdx.x == 0) ? (0x80000000u - (nblocks - 1)) : 1u;
        *s_old =
            __hip_atomic_fetch_add(sem, contrib, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
}

__device__ __forceinline__ void grid_wait(unsigned int* sem, const unsigned int* s_old)
{
    if(threadIdx.x == 0)
    {
        const unsigned int old = *s_old;
        // s_sleep between polls: unthrottled loads from every block on this one
        // cacheline starve the atomics that would end the barrier. Relaxed spin
        // plus one acquire fence after -- an acquire *load* would emit a
        // buffer_inv sc1 every poll, which dominates the barrier cost.
        while(((old ^ __hip_atomic_load(sem, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) &
               0x80000000u) == 0u)
        {
            __builtin_amdgcn_s_sleep(1);
        }
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
    __syncthreads();
}

// packed sorted-id format, identical to MOE_SORTING_MOCK_ID in moe_sorting_opus.h
__device__ __forceinline__ int pack_id(int token_id, int topk_id)
{
    return static_cast<int>((static_cast<uint32_t>(token_id) & 0x00ffffffu) |
                            ((static_cast<uint32_t>(topk_id) & 0xffu) << 24));
}

// Widen to fp32. The float overload matters: some models store the correction
// bias as fp32, and forcing it through the bf16 bit layout would mis-route.
__device__ __forceinline__ float bf16f(const float& x) { return x; }

template <typename T>
__device__ __forceinline__ float bf16f(const T& x)
{
    return opus::bf16_to_fp32(x);
}

// Max experts per lane, covering E <= 512 (the entry's limit).
static constexpr int kEptMax = 8;

// Expert-id slots the routing map spans: the routed experts, the fused shared
// experts (replicated, so every rank holds all of them), and -- under EP only --
// one trailing sentinel that ranks not owning a token park its shared row on.
// Host and device both call this so the histogram width cannot drift.
__host__ __device__ constexpr int expert_slots(int E, int n_shared, bool ep)
{
    return E + n_shared + ((ep && n_shared > 0) ? 1 : 0);
}

// Single source of truth for the LDS layout: the kernel carves its pointers out
// of the dynamic allocation and the host asks this for the byte count, so the
// two cannot drift. Every slot is 4 bytes, offsets in elements.
struct LdsLayout
{
    int    s_weight, s_cnt, s_scan, s_buf, s_unit, s_lid;
    size_t bytes;

    // E_tot: expert_slots(...). ep: an expert_mask was supplied. s_lid is
    // allocated only then, so the non-EP footprint (and its occupancy) is
    // untouched.
    __host__ __device__ constexpr LdsLayout(int BlockSize, int total_routed_rows, int E_tot,
                                            bool ep)
        : s_weight(total_routed_rows) // s_expert is at 0
        , s_cnt(s_weight + total_routed_rows)
        , s_scan(s_cnt + E_tot)
        // s_buf also holds the phase-2 scan's 2*NWAVE wave totals, hence the
        // floor, which only bites at M*topk < 8.
        // s_scan spans 2*BlockSize slots -- one per thread-pair -- or E_tot when
        // the fused shared slots push past that; see the phase-2 tail fixup.
        , s_buf(s_scan + (E_tot > 2 * BlockSize ? E_tot : 2 * BlockSize))
        , s_unit(s_buf + (total_routed_rows > 2 * (BlockSize / kWaveSize)
                              ? total_routed_rows
                              : 2 * (BlockSize / kWaveSize)))
        , s_lid(s_unit + total_routed_rows)
        , bytes((size_t)(s_lid + (ep ? E_tot : 0)) * sizeof(int))
    {
    }
};

// The argmax compares one ordered 64-bit key, not a (float, int) pair:
//
//   key = ord(score) : ~expert_id      (hi 32 : lo 32)
//
// ord() is the monotonic float->uint map, so an unsigned compare gives "highest
// score wins, lower expert id breaks ties" exactly. This is what keeps a
// butterfly step branch-free: the pair form's short-circuiting tie-break made
// the compiler emit an s_and_saveexec_b64/s_or_b64 pair per step, and a VALU
// write of EXEC costs 5 wait states before the next DPP op (CDNA4 ISA Table
// 11), so exec churn -- not the permutes -- dominated phase 1.
__device__ __forceinline__ uint32_t ord_f32(float f)
{
    const uint32_t b = __builtin_bit_cast(uint32_t, f);
    // negative -> invert everything, positive -> flip the sign bit
    return b ^ (static_cast<uint32_t>(static_cast<int32_t>(b) >> 31) | 0x80000000u);
}

__device__ __forceinline__ uint64_t pack_argmax_key(float score, int eid)
{
    return (static_cast<uint64_t>(ord_f32(score)) << 32) | static_cast<uint32_t>(~eid);
}

__device__ __forceinline__ int unpack_argmax_expert(uint64_t k)
{
    return static_cast<int>(~static_cast<uint32_t>(k));
}

// DPP permutes 32 bits at a time, so the halves move separately and recombine.
template <int DPP>
__device__ __forceinline__ void argmax_dpp_step(uint64_t& k)
{
    const uint32_t hi = aiter::mov_dpp_(static_cast<uint32_t>(k >> 32), opus::number<DPP>{});
    const uint32_t lo = aiter::mov_dpp_(static_cast<uint32_t>(k), opus::number<DPP>{});
    const uint64_t o  = (static_cast<uint64_t>(hi) << 32) | lo;
    k                 = (o > k) ? o : k;
}

__device__ __forceinline__ float readlane_f(float v, int l)
{
    return __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, v), l));
}

__device__ __forceinline__ uint64_t readlane_u64(uint64_t v, int l)
{
    const uint32_t hi = __builtin_amdgcn_readlane(static_cast<int>(v >> 32), l);
    const uint32_t lo = __builtin_amdgcn_readlane(static_cast<int>(v), l);
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

// Full 64-lane argmax, no LDS: six DPP permutes plus one readlane pair.
//
// ROW_BCAST15/31 are CDNA-only (dropped on RDNA3+) and give the 16->32->64 fold
// in the VALU. Folding the row winners with readlane instead costs 2 wait
// states per SGPR read (CDNA4 ISA Table 11) -- a serial stall chain.
//
// bound_ctrl=1 leaves lanes 0..15 holding garbage after the bcast steps; only
// lane 63 is read.
__device__ __forceinline__ uint64_t wave_argmax(uint64_t k)
{
    argmax_dpp_step<0xB1>(k);  // quad_perm:[1,0,3,2]  (xor 1)
    argmax_dpp_step<0x4E>(k);  // quad_perm:[2,3,0,1]  (xor 2)
    argmax_dpp_step<0x141>(k); // row_half_mirror      (4 -> 8)
    argmax_dpp_step<0x140>(k); // row_mirror           (8 -> 16)
    argmax_dpp_step<0x142>(k); // row_bcast15          (16 -> 32)
    argmax_dpp_step<0x143>(k); // row_bcast31          (32 -> 64)
    return readlane_u64(k, 63);
}

// O(topk*E) wave argmax: lane l holds experts l, l+64, ..
//
// Split into _load and _select on purpose. The gating/bias loads dominate phase
// 1 and only wave 0 issues them. The caller runs _load, the
// whole-block quant, then _select, so the s_waitcnt sinks into _select and the
// latency hides behind the quant. So _load must issue the loads and nothing
// that consumes them -- the sigmoid lives in _select.
template <int EPT, typename DTYPE_I, typename DTYPE_B>
__device__ __forceinline__ void phase1_topk_load(const DTYPE_I* __restrict__ gating_row,
                                                 const DTYPE_B* __restrict__ bias, int E,
                                                 DTYPE_I* g, DTYPE_B* b)
{
    const int lane_id = threadIdx.x & (kWaveSize - 1);
#pragma unroll
    for(int j = 0; j < EPT; ++j)
    {
        // Clamp, do not predicate: a guarded load cannot be speculated, so each
        // j gets its own exec mask and s_waitcnt instead of all 2*EPT loads
        // being in flight under one wait. _select discards the duplicates.
        const int e = min(lane_id + j * kWaveSize, E - 1);
        g[j]        = gating_row[e];
        b[j]        = bias[e];
    }
}

template <int EPT, int NSHARED, typename DTYPE_I, typename DTYPE_B>
__device__ __forceinline__ void phase1_topk_select(const DTYPE_I* __restrict__ gating_row,
                                                   float* __restrict__ topk_weights,
                                                   int* __restrict__ topk_ids, int token, int E,
                                                   int topk, bool need_renorm, float rsf,
                                                   float shared_w, int ep_rank, int ep_size,
                                                   const DTYPE_I* g, const DTYPE_B* b)
{
    const int lane_id = threadIdx.x & (kWaveSize - 1);
    // Rows this token occupies: its routed picks plus one per fused shared
    // expert. NSHARED == 0 makes this topk and every use below folds away.
    const int topk_total = topk + NSHARED;

    uint64_t key[EPT];
#pragma unroll
    for(int j = 0; j < EPT; ++j)
    {
        const int e = lane_id + j * kWaveSize;
        // Out-of-range lanes get the minimum key and can never win.
        key[j] = pack_argmax_key(
            (e < E) ? (1.0f / (1.0f + __expf(-bf16f(g[j]))) + bf16f(b[j])) : -INFINITY,
            (e < E) ? e : INT_MAX);
    }

    // Sort this lane's keys descending once: each round's candidate is then
    // key[0] and retiring the winner is a shift-down, so one compile-time
    // sorting network replaces an EPT-deep max scan per round. Keys are unique
    // (they pack ~eid), so the winner matches key[0] on exactly one lane.
#pragma unroll
    for(int i = 0; i < EPT; ++i)
#pragma unroll
        for(int j = i + 1; j < EPT; ++j)
        {
            const uint64_t a = key[i], b = key[j];
            key[i]           = (a < b) ? b : a;
            key[j]           = (a < b) ? a : b;
        }

    // Winner k lives in lane k's register, not a shared s_win[k]: wave_argmax
    // leaves the answer in SGPRs, so this is one v_cndmask per pass instead of
    // a masked ds_write plus ds_read.
    int winner_expert = INT_MAX;
    for(int k = 0; k < topk; ++k)
    {
        const uint64_t winner_key = wave_argmax(key[0]); // wave-uniform, in SGPRs
        winner_expert = (lane_id == k) ? unpack_argmax_expert(winner_key) : winner_expert;
        // Retire the winner: it was key[0], so shift down and stay sorted.
        const bool hit = (key[0] == winner_key);
#pragma unroll
        for(int j = 0; j < EPT - 1; ++j)
            key[j] = hit ? key[j + 1] : key[j];
        key[EPT - 1] = hit ? 0 : key[EPT - 1];
    }

    float w = 0.0f;
    if(lane_id < topk)
        w = 1.0f / (1.0f + __expf(-bf16f(gating_row[winner_expert])));
    if(need_renorm)
    {
        float s = w; // full 64-lane sum in DPP
        s += aiter::mov_dpp_(s, opus::number<0xB1>{});
        s += aiter::mov_dpp_(s, opus::number<0x4E>{});
        s += aiter::mov_dpp_(s, opus::number<0x141>{});
        s += aiter::mov_dpp_(s, opus::number<0x140>{});
        s += aiter::mov_dpp_(s, opus::number<0x142>{}); // row_bcast15
        s += aiter::mov_dpp_(s, opus::number<0x143>{}); // row_bcast31
        s = readlane_f(s, 63);
        w *= rsf / s;
    }
    else
    {
        w *= rsf;
    }
    if(lane_id < topk)
    {
        topk_ids[token * topk_total + lane_id]     = winner_expert;
        topk_weights[token * topk_total + lane_id] = w;
    }
    // Fused shared experts occupy the tail lanes. Written after the renorm
    // above, never before: the DPP sum spans all 64 lanes and only holds
    // because w == 0 outside lane_id < topk.
    if constexpr(NSHARED > 0)
    {
        const int s = lane_id - topk;
        if(s >= 0 && s < NSHARED)
        {
            // The shared weights are replicated on every rank and every rank
            // sees every token, so emitting the shared row unconditionally
            // would make the post-MoE all-reduce sum ep_size copies of it.
            // Round-robin token ownership instead; non-owners park the row on
            // the always-masked sentinel slot, which contributes nothing.
            // ep_size == 1 makes every rank the owner, i.e. the non-EP case.
            const bool owner = (token % ep_size) == ep_rank;
            topk_ids[token * topk_total + lane_id] = owner ? (E + s) : (E + NSHARED);
            topk_weights[token * topk_total + lane_id] = shared_w;
        }
    }
}

// MXFP4 quant of one token's hidden row, by the whole block. cols/TD ==
// BlockSize, so one vector per thread covers the row in a single pass. The e8m0
// byte goes to a scratch row, not the swizzled buffer: its swizzled position
// depends on the sorted row, which is unknown until after the barrier.
template <int BlockSize, int TD, typename DTYPE_I>
__device__ __forceinline__ void phase1_quant_token(opus::fp4_t* __restrict__ out,
                                                   uint8_t* __restrict__ tok_scale,
                                                   const DTYPE_I* __restrict__ input,
                                                   int token, int cols, int group_size,
                                                   int scaleN_pad)
{
    const int num_thread_per_group = group_size / TD;
    const int scale_k              = threadIdx.x / num_thread_per_group;
    const int scaleN_valid         = (cols + group_size - 1) / group_size;

    using vec_i = opus::vector_t<DTYPE_I, TD>;
    using vec_f = opus::vector_t<float, TD>;

    auto buffer_input = opus::make_gmem<DTYPE_I>(input + (int64_t)token * cols,
                                                 cols * sizeof(DTYPE_I));
    vec_i vin =
        load_vector_nbytes<DTYPE_I, TD, (sizeof(DTYPE_I) * TD % 16 == 0 ? 16 : 8), /*aux=*/0>(
            buffer_input, threadIdx.x * TD);
    vec_f  vin_f32;
    float* vin_f32_ptr = reinterpret_cast<float*>(&vin_f32);
    float  absMax      = 1e-10f;
#pragma unroll
    for(int j = 0; j < TD; ++j)
    {
        vin_f32[j] = bf16f(vin[j]);
        absMax     = max(absMax, fabsf(vin_f32[j]));
    }
    absMax          = multithread_reduce(absMax, aiter::Max(), num_thread_per_group);
    float row_scale = aiter::fp4_f32_to_e8m0_scale(absMax);

    if(threadIdx.x % num_thread_per_group == 0 && scale_k < scaleN_valid)
        tok_scale[(int64_t)token * scaleN_pad + scale_k] =
            (__builtin_bit_cast(uint32_t, row_scale) >> 23) & 0xFF;

    scaled_quant_vgpr_impl<float, opus::fp4_t, TD>(out, vin_f32_ptr, &row_scale, cols,
                                                   (int64_t)token * cols);
}

// One token of phase 1: load, quant, select (see phase1_topk_load for why that
// order). Not templated on EPT even though the halves are -- that would inline
// three copies of the quant -- so g[]/b[] are sized to kEptMax and the dispatch
// is pushed down. Both halves need the same tier ladder, and a
// tier added to one and not the other corrupts registers silently.
#define FMR_EPT_DISPATCH(CALL)      \
    do                              \
    {                               \
        if(E <= 2 * kWaveSize)      \
            CALL(2);                \
        else if(E <= 5 * kWaveSize) \
            CALL(5);                \
        else                        \
            CALL(8);                \
    } while(0)

template <int BlockSize, int TD, int NSHARED, typename DTYPE_I, typename DTYPE_B>
__device__ __forceinline__ void
phase1_token(opus::fp4_t* __restrict__ out, uint8_t* __restrict__ tok_scale,
             const DTYPE_I* __restrict__ hidden, const DTYPE_I* __restrict__ gating,
             const DTYPE_B* __restrict__ bias, float* __restrict__ topk_weights,
             int* __restrict__ topk_ids, int token, int E, int topk, int cols, int group_size,
             int scaleN_pad, bool need_renorm, float rsf, float shared_w, int ep_rank,
             int ep_size)
{
    const DTYPE_I* gating_row = gating + (int64_t)token * E;
    const bool     sel  = (threadIdx.x >> 6) == 0;

    DTYPE_I g[kEptMax];
    DTYPE_B b[kEptMax];
    if(sel)
    {
#define FMR_LOAD(EPT) phase1_topk_load<EPT, DTYPE_I, DTYPE_B>(gating_row, bias, E, g, b)
        FMR_EPT_DISPATCH(FMR_LOAD);
#undef FMR_LOAD
    }

    phase1_quant_token<BlockSize, TD, DTYPE_I>(out, tok_scale, hidden, token, cols, group_size,
                                               scaleN_pad);

    if(sel)
    {
#define FMR_SELECT(EPT)                                                                \
    phase1_topk_select<EPT, NSHARED, DTYPE_I, DTYPE_B>(                                \
        gating_row, topk_weights, topk_ids, token, E, topk, need_renorm, rsf,          \
        shared_w, ep_rank, ep_size, g, b)
        FMR_EPT_DISPATCH(FMR_SELECT);
#undef FMR_SELECT
    }
}
#undef FMR_EPT_DISPATCH

// Inclusive add-scan over a wave's 64 lanes, DPP only -- no LDS, no barrier.
// ROW_SR{1,2,4,8} scans each row of 16 (bound_ctrl=1 shifts in 0, the add
// identity); the three row totals are then readlane-broadcast and folded in.
__device__ __forceinline__ int wave_scan_incl(int v, int lane_id)
{
    v += aiter::mov_dpp_(v, opus::number<0x111>{}); // row_shr:1
    v += aiter::mov_dpp_(v, opus::number<0x112>{}); // row_shr:2
    v += aiter::mov_dpp_(v, opus::number<0x114>{}); // row_shr:4
    v += aiter::mov_dpp_(v, opus::number<0x118>{}); // row_shr:8
    const int r0 = __builtin_amdgcn_readlane(v, 15);
    const int r1 = __builtin_amdgcn_readlane(v, 31);
    const int r2 = __builtin_amdgcn_readlane(v, 47);
    return v + (lane_id >= 16 ? r0 : 0) + (lane_id >= 32 ? r1 : 0) + (lane_id >= 48 ? r2 : 0);
}

// Ballot-rank expert `e`'s routed ids into buf[0..cnt), in ascending routed
// index. Deterministic, so no global atomic cursor is needed. Wave-wide.
__device__ __forceinline__ void expert_rank_list(int* buf, const int* s_expert, int e, int cnt,
                                                 int total_routed_rows, int lane_id)
{
    // No early exit and no predicated LDS read: both serialise the ds_reads to
    // one round trip per chunk. Clamping keeps them all in flight under one
    // lgkmcnt, same as phase1_topk_load.
    int written = 0;
    for(int chunk_begin = 0; chunk_begin < total_routed_rows; chunk_begin += kWaveSize)
    {
        const int  routed_idx = chunk_begin + lane_id;
        const bool m = (routed_idx < total_routed_rows) &&
                       (s_expert[min(routed_idx, total_routed_rows - 1)] == e);
        // Which lanes of this chunk routed to e; each lane counts the bits
        // below its own to get its output slot.
        const uint64_t match_mask = __ballot(m);
        const int      rank       = __popcll(match_mask & ((1ull << lane_id) - 1ull));
        if(m)
            buf[written + rank] = routed_idx;
        written += __popcll(match_mask);
    }
}

// Dynamic LDS layout (R = total_routed_rows = M*(topk+NSHARED),
// E_tot = expert_slots(E, NSHARED, ep)):
//   s_expert [R]            routed expert id per row   (phase 2)
//   s_weight [R]            routed weight per row      (phase 2)
//   s_cnt    [E_tot]        expert histogram           (phase 2)
//   s_scan   [max(2*BlockSize, E_tot)] padded prefix sum (phase 2)
//   s_buf    [R]            ballot rank list           (phase 3)
//   s_unit   [R]            unit index -> owning expert (phase 2)
//   s_lid    [E_tot]        global -> local expert id, EP only (phase 2)
//
// PART selects which half a launch runs: kFused is both with the grid barrier
// between, kPhase1/kPhase23 split at exactly that barrier into two launches.
// The host picks by token count (kSplitMinTokens).
enum FmrPart { kFused = 0, kPhase1 = 1, kPhase23 = 2 };

template <int BlockSize, int TD, typename DTYPE_I, int PART = kFused,
          typename DTYPE_B = DTYPE_I, int NSHARED = 0>
__global__ void __launch_bounds__(BlockSize)
fused_moe_routing_kernel(const DTYPE_I* __restrict__ gating,   // [M, E]
                         const DTYPE_B* __restrict__ bias,     // [E] fp32 or bf16
                         const DTYPE_I* __restrict__ hidden,   // [M, cols]
                         float* __restrict__ topk_weights,     // [M, topk]
                         int* __restrict__ topk_ids,           // [M, topk]
                         int* __restrict__ sorted_ids,         // [maxpad]
                         float* __restrict__ sorted_weights,   // [maxpad]
                         int* __restrict__ sorted_expert_ids,  // [maxblk]
                         int* __restrict__ num_valid_ids,      // [2]
                         opus::fp4_t* __restrict__ out,        // [M, cols/2]
                         uint8_t* __restrict__ out_scale,      // swizzled e8m0
                         uint8_t* __restrict__ tok_scale,      // [M, scaleN_pad] scratch
                         // stage2 accumulates into moe_buf atomically, so it must start
                         // zeroed; doing it here saves a separate fill launch. Never read
                         // here, so it needs no ordering. nullptr = already zeroed.
                         DTYPE_I* __restrict__ moe_buf,        // [M, model_dim] or nullptr
                         int moe_buf_elems,                    // M * model_dim
                         unsigned int* __restrict__ sem,       // [1] persistent
                         // [E_tot], nonzero iff this rank owns the expert; nullptr = no EP
                         const int* __restrict__ expert_mask,
                         int M, int E, int topk, int unit_size, int group_size, int cols,
                         int max_blocks, // length of sorted_expert_ids
                         int max_tokens, // length of sorted_ids
                         bool need_renorm, float rsf,
                         // Fused shared experts (NSHARED > 0): the weight every
                         // token gives each shared expert, and this rank's slot
                         // in the round-robin that keeps them un-duplicated.
                         float shared_w, int ep_rank, int ep_size)
{
    extern __shared__ char smem_raw[];
    // One row per pick plus one per fused shared expert.
    const int  topk_total        = topk + NSHARED;
    const int  total_routed_rows = M * topk_total;
    constexpr int NWAVE  = BlockSize / kWaveSize;
    const bool ep = expert_mask != nullptr;
    // Phase 2's histogram is indexed by whatever phase 1 emitted, sentinel
    // included, so it must span every slot rather than just the routed experts.
    const int E_tot = expert_slots(E, NSHARED, ep);
    // The pair scan below reaches 2*BlockSize slots, one per thread-pair. E is
    // capped there, so only the fused shared tail can spill past it.
    constexpr int kPairSlots = 2 * BlockSize;
    const int scan_slots = E_tot > kPairSlots ? E_tot : kPairSlots;

    const LdsLayout lds(BlockSize, total_routed_rows, E_tot, ep);
    int*   s_base   = reinterpret_cast<int*>(smem_raw);
    int*   s_expert = s_base;
    float* s_weight = reinterpret_cast<float*>(s_base + lds.s_weight);
    int*   s_cnt    = s_base + lds.s_cnt;
    int*   s_scan   = s_base + lds.s_scan;
    int*   s_buf    = s_base + lds.s_buf;
    int*   s_unit   = s_base + lds.s_unit; // unit index -> owning expert
    int*   s_lid    = s_base + lds.s_lid;  // EP only: global -> local expert id

    const int tid     = threadIdx.x;
    const int lane_id = tid & (kWaveSize - 1);
    const int wave_id = tid >> 6;

    const int scaleN_valid = (cols + group_size - 1) / group_size;
    const int scaleN_pad =
        ((scaleN_valid + kScalesPerThread - 1) / kScalesPerThread) * kScalesPerThread;

    // unit_size is a runtime arg, so dividing by it costs a real integer divide
    // in several phase-2/3 loops. The host checks it is a power of two, so
    // shift instead; umask rounds up to a unit via (x + unit_size - 1) & umask.
    const int log2u = __builtin_ctz(static_cast<uint32_t>(unit_size));
    const int umask = ~(unit_size - 1);

    // moe_buf clear: grid-stride, 16B stores, issued before phase 1 so the
    // writes drain while phase 1 stalls on its loads. The PART guard stops the
    // split path from clearing twice.
    if constexpr(PART != kPhase23)
    if(moe_buf != nullptr)
    {
        constexpr int VEC = 8; // 8 x 2B = one 16B dwordx4 store
        using vec_t = __attribute__((__vector_size__(VEC * sizeof(DTYPE_I)))) DTYPE_I;
        const int nvec = moe_buf_elems / VEC;
        vec_t  z{};
        vec_t* vbuf = reinterpret_cast<vec_t*>(moe_buf);
        for(int i = blockIdx.x * BlockSize + tid; i < nvec; i += gridDim.x * BlockSize)
            vbuf[i] = z;
        // Tail: empty for every supported model_dim, kept for safety.
        for(int i = nvec * VEC + blockIdx.x * BlockSize + tid; i < moe_buf_elems;
            i += gridDim.x * BlockSize)
            moe_buf[i] = (DTYPE_I)0;
    }

    // Phase 1: per-token topk + quant.
    if constexpr(PART != kPhase23)
    for(int t = blockIdx.x; t < M; t += gridDim.x)
    {
        phase1_token<BlockSize, TD, NSHARED, DTYPE_I, DTYPE_B>(
            out, tok_scale, hidden, gating, bias, topk_weights, topk_ids, t, E, topk, cols,
            group_size, scaleN_pad, need_renorm, rsf, shared_w, ep_rank, ep_size);
        __syncthreads(); // the aliased scratch is reused each iteration
    }

    // The only grid barrier. The s_expert/s_weight loads below read topk data
    // other blocks wrote, so they must stay after it. Zeroing s_cnt has no
    // cross-block dependency, so it fills the wait.
    if constexpr(PART == kPhase1)
    {
        return; // in split mode the kernel boundary IS the barrier
    }
    else if constexpr(PART == kPhase23)
    {
        for(int e = tid; e < E_tot; e += BlockSize)
            s_cnt[e] = 0;
        __syncthreads();
    }
    else if(gridDim.x == 1)
    {
        // __syncthreads() fences global memory too, so no release/acquire.
        __syncthreads();
        for(int e = tid; e < E_tot; e += BlockSize)
            s_cnt[e] = 0;
        __syncthreads();
    }
    else
    {
        unsigned int arr_old;
        grid_arrive(sem, gridDim.x, &arr_old);
        for(int e = tid; e < E_tot; e += BlockSize)
            s_cnt[e] = 0;
        grid_wait(sem, &arr_old);
    }

    // Phase 2: rebuild the routing map.
    for(int i = tid; i < total_routed_rows; i += BlockSize)
    {
        s_expert[i] = topk_ids[i];
        s_weight[i] = topk_weights[i];
    }
    __syncthreads();

    for(int i = tid; i < total_routed_rows; i += BlockSize)
        atomicAdd(&s_cnt[s_expert[i]], 1);
    __syncthreads();

    // Padded two-level inclusive scan over 2*BlockSize slots (>= E): each thread
    // folds an adjacent pair, a DPP scan covers the wave with no barrier, and
    // one __syncthreads publishes the wave totals. Three syncs, versus 18 for
    // the Hillis-Steele form.
    int* s_wtot = s_buf;            // s_buf is not live until phase 3
    int* s_mtot = s_buf + NWAVE;    // EP mask-scan wave totals
    // Under EP this rank owns only the masked-in experts and the GEMM indexes
    // sorted_expert_ids by *local* id, so phase 2 must additionally give
    // masked-out experts zero units and emit local(e) = exclusive cumsum of the
    // mask (matching moe_align_block_size_kernel_ex in moe_sorting_opus.h). The
    // mask cumsum is a second DPP scan riding the same data path and the same
    // __syncthreads, so it adds no barrier -- what matters, since phase 2 is
    // latency-bound.
    const int owned0 = (2 * tid < E_tot) ? (ep ? (expert_mask[2 * tid] != 0) : 1) : 0;
    const int owned1 = (2 * tid + 1 < E_tot) ? (ep ? (expert_mask[2 * tid + 1] != 0) : 1) : 0;
    // Unit counts, not padded row counts: the unit table below needs them.
    const int units0 = (owned0 ? ((s_cnt[2 * tid] + unit_size - 1) >> log2u) : 0);
    const int units1 = (owned1 ? ((s_cnt[2 * tid + 1] + unit_size - 1) >> log2u) : 0);
    const int rows0     = units0 * unit_size;
    const int rows1     = units1 * unit_size;
    const int pair_rows = rows0 + rows1;
    const int rows_incl = wave_scan_incl(pair_rows, lane_id);
    // Mask cumsum, only when EP is on -- `ep` is grid-uniform, so the branch is
    // free and the non-EP path keeps exactly its old instruction count.
    const int pair_owned = ep ? (owned0 + owned1) : 0;
    const int owned_incl = ep ? wave_scan_incl(pair_owned, lane_id) : 0;
    if(lane_id == kWaveSize - 1)
    {
        s_wtot[wave_id] = rows_incl;
        if(ep)
            s_mtot[wave_id] = owned_incl;
    }
    __syncthreads();
    int wave_prefix = 0;
    for(int w = 0; w < wave_id; ++w)
        wave_prefix += s_wtot[w];
    // Exclusive prefix of this thread's pair: where its first expert starts.
    const int pair_base = wave_prefix + rows_incl - pair_rows;
    s_scan[2 * tid]     = pair_base + rows0;
    s_scan[2 * tid + 1] = pair_base + rows0 + rows1;
    if(ep)
    {
        int owned_wave_prefix = 0;
        for(int w = 0; w < wave_id; ++w)
            owned_wave_prefix += s_mtot[w];
        // Exclusive prefix of the mask: the local id of this pair's first expert.
        const int owned_base = owned_wave_prefix + owned_incl - pair_owned;
        // Masked-out experts own no units, so nothing looks their slot up; -1
        // matches the stock sentinel and keeps a bug visible if one ever does.
        if(2 * tid < E_tot)
            s_lid[2 * tid] = owned0 ? owned_base : -1;
        if(2 * tid + 1 < E_tot)
            s_lid[2 * tid + 1] = owned1 ? (owned_base + owned0) : -1;
    }

    // Reverse of s_scan: s_unit[unit_idx] -> owning expert. Phase 3 is row-
    // partitioned, so this saves it a 9-deep binary search of dependent
    // ds_reads. Built here because this thread already holds both offsets in
    // registers, so it costs no barrier and no read-back.
    {
        const int unit_base = pair_base >> log2u; // pair_base is unit-aligned
        for(int u = 0; u < units0; ++u)
            s_unit[unit_base + u] = 2 * tid;
        for(int u = 0; u < units1; ++u)
            s_unit[unit_base + units0 + u] = 2 * tid + 1;
    }
    __syncthreads();

    if constexpr(NSHARED > 0)
    {
        // Slots past the pair scan's reach: at most NSHARED + 1 of them, each a
        // block of rows appended after the routed ones. They need a running
        // total, not a scan, so one thread walks them serially -- cheaper than
        // a second scan pass and it keeps the pair scan's shape untouched.
        if(E_tot > kPairSlots && tid == 0)
        {
            int rows = s_scan[kPairSlots - 1];
            int lid  = 0;
            if(ep)
                for(int w = 0; w < NWAVE; ++w)
                    lid += s_mtot[w]; // local ids the main range already used
            for(int e = kPairSlots; e < E_tot; ++e)
            {
                const int owned = ep ? (expert_mask[e] != 0) : 1;
                const int units = owned ? ((s_cnt[e] + unit_size - 1) >> log2u) : 0;
                const int ubase = rows >> log2u; // rows is unit-aligned
                for(int u = 0; u < units; ++u)
                    s_unit[ubase + u] = e;
                rows += units << log2u;
                s_scan[e] = rows;
                if(ep)
                {
                    s_lid[e] = owned ? lid : -1;
                    lid += owned;
                }
            }
        }
        __syncthreads();
    }

    const int num_valid = s_scan[scan_slots - 1];
    if(blockIdx.x == 0 && tid == 0)
    {
        num_valid_ids[0] = num_valid;
        num_valid_ids[1] = M;
    }

    // Tail fills. Stage1 launches over the whole buffer and reads an id and a
    // token index for every block before it checks num_valid_ids, so the region
    // past num_valid must be initialized: left alone it holds whatever shared
    // the allocator pool, which under vLLM's graph capture is another graph's
    // activations reinterpreted as int32, indexing far out of bounds. Zero is a
    // valid expert id and pack_id(M, topk_total) is the "no token" sentinel the stock
    // path also writes. Grid-strided, and cannot race phase 3, which only
    // writes below num_valid.
    const int global_tid    = (int)blockIdx.x * BlockSize + tid;
    const int global_stride = (int)gridDim.x * BlockSize;

    const int valid_blocks = num_valid >> log2u; // num_valid is unit-aligned
    for(int j = valid_blocks + global_tid; j < max_blocks; j += global_stride)
        sorted_expert_ids[j] = 0;

    for(int j = num_valid + global_tid; j < max_tokens; j += global_stride)
    {
        sorted_ids[j]     = pack_id(M, topk_total);
        sorted_weights[j] = 0.0f;
    }

    // Phase 3: sorted map + scale scatter, barrier-free, partitioned by
    // position in the sorted output rather than by expert.
    //
    // Partitioning by expert cannot scale: the routing is sparse, so it caps at
    // one wave per non-empty expert however wide the grid. And an output
    // partition needs no grid barrier despite appearances -- every block has
    // already rebuilt the whole map, so it derives its own slice's tokens
    // locally.
    //
    // Ceil in UNITS, not rows, so slices never split an expert's
    // sorted_expert_ids block: rounding a truncated row quotient up to a unit
    // does not always recover the truncation, and the leftover top rows would
    // then belong to no block and be skipped by the tail fills too.
    const int rows_per_block =
        max(1, (valid_blocks + (int)gridDim.x - 1) / (int)gridDim.x) << log2u;
    const int slice_end = min(num_valid, (int)(blockIdx.x + 1) * rows_per_block);

    for(int row_begin = min(num_valid, (int)blockIdx.x * rows_per_block); row_begin < slice_end;)
    {
        // row_begin is always unit-aligned, so s_unit answers directly.
        const int e       = s_unit[row_begin >> log2u];
        const int cnt     = s_cnt[e];
        const int padded  = (cnt + unit_size - 1) & umask;
        const int base    = s_scan[e] - padded; // exclusive offset
        const int row_end = min(slice_end, base + padded);

        if(wave_id == 0)
            expert_rank_list(s_buf, s_expert, e, cnt, total_routed_rows, lane_id);
        __syncthreads();

        // s_unit/s_cnt/s_scan stay global-indexed; only the emitted id is local,
        // since that is what the downstream GEMM consumes.
        const int e_out = ep ? s_lid[e] : e;

        // sorted_expert_ids is indexed by unit, not row: one id per GEMM tile.
        const int unit_begin = row_begin >> log2u;
        const int unit_end   = row_end >> log2u;
        for(int j = unit_begin + tid; j < unit_end; j += BlockSize)
            sorted_expert_ids[j] = e_out;

        for(int row = row_begin + tid; row < row_end; row += BlockSize)
        {
            const int expert_row = row - base;
            if(expert_row < cnt)
            {
                // s_buf holds routed-entry indices: entry / topk_total is the
                // token, entry % topk_total is which row of that token it is.
                const int routed_idx = s_buf[expert_row];
                const int token      = routed_idx / topk_total;
                const int slot       = routed_idx - token * topk_total;
                sorted_ids[row]      = pack_id(token, slot);
                sorted_weights[row]  = s_weight[routed_idx];
            }
            else
            {
                sorted_ids[row]     = pack_id(M, topk_total); // sentinel row
                sorted_weights[row] = 0.0f;
            }
        }

        {
            // thread_data e8m0 bytes per thread, so `chunk` threads cover one
            // sorted row -- the shape the stock mxfp4_moe_sort_kernel uses, and
            // the one that keeps the writes coalesced through the swizzle.
            constexpr int thread_data = kScalesPerThread;
            const int     chunk       = scaleN_valid / thread_data;
            for(int idx = tid; idx < (row_end - row_begin) * chunk; idx += BlockSize)
            {
                // idx is flat over (rows in this span) x (threads per row).
                const int slice_row   = idx / chunk;
                const int row         = row_begin + slice_row;
                const int scale_begin = (idx - slice_row * chunk) * thread_data;
                const int expert_row  = row - base;
                // pad rows -> zeros
                const int token = (expert_row < cnt) ? (s_buf[expert_row] / topk_total) : M;
                const uint8_t* src =
                    (token < M) ? (tok_scale + (int64_t)token * scaleN_pad + scale_begin)
                                : nullptr;
                // Hoisted mx_scale_shuffle_idx(scaleN_pad, x=row, y=column),
                // which costs 3 divides and 3 modulos and would otherwise run
                // thread_data times per thread with only y varying. Its
                // y-dependent terms are all periodic in 8 and scale_begin is a
                // multiple of 8, so they collapse to OFF[] and the rest hoists
                // into sbase. Verified bit-exact for x < 200, all scale_begin,
                // scaleN_pad = 128.
                const int sbase = (row / 32 * scaleN_pad) * 32 + (scale_begin / 8) * 256 +
                                  (row % 16) * 4 + (row % 32) / 16;
                constexpr int OFF[thread_data] = {0, 64, 128, 192, 2, 66, 130, 194};
#pragma unroll
                for(int j = 0; j < thread_data; ++j)
                    out_scale[sbase + OFF[j]] = src ? src[j] : (uint8_t)0;
            }
        }
        __syncthreads(); // s_buf is reused by the next expert in the slice
        row_begin = row_end;
    }
}

} // namespace fmr
} // namespace aiter
