// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// OPUS-based paged-attention decode for gfx950.
//
// Algorithm follows the sp3 kernel PA_A16W16_*_1TG_4W_16mx1_64nx4: one thread
// group of 4 waves per (batch, kv-head), 16 query rows, waves split along the
// KV axis for Q*K and along the head axis for P*V.
//
// Single header: the public API is always visible, host plumbing and the device
// kernel sit behind PA_DECODE_OPUS_IMPL, and the host pass gets an empty stub so
// the `__device_stub__` symbols still resolve.

#pragma once
#include "aiter_tensor.h"
#include <optional>

// Public API: paged-attention decode over a block table.
//
// Tensor expectations (row-major, last dim contiguous):
//   q            : [batch, num_heads, 128]                          bf16
//   k_cache      : [num_blocks, num_kv_heads, 128/8, 16, 8]         bf16 (vLLM packing, x=8)
//   v_cache      : [num_blocks, num_kv_heads, 128, 16]              bf16
//   block_tables : [batch, max_blocks_per_batch_row]                int32
//   context_lens : [batch]                                          int32
//   out          : [batch, num_heads, 128]                          bf16 (caller-allocated)
//
// `softmax_scale` is forwarded as-is (no implicit 1/sqrt(D)).
// Requires head_dim == 128, page size == 16, and num_heads/num_kv_heads <= 16.
void pa_decode_opus_fwd(aiter_tensor_t& q,
                        aiter_tensor_t& k_cache,
                        aiter_tensor_t& v_cache,
                        aiter_tensor_t& block_tables,
                        aiter_tensor_t& context_lens,
                        aiter_tensor_t& out,
                        float softmax_scale);

// A8W8 variant: Q, K and V all arrive quantized to fp8 e4m3, so both GEMMs run on
// v_mfma_f32_16x16x128_f8f6f4. Halves the KV cache traffic and the registers holding
// it, and the wider K collapses each GEMM's contraction to a single MFMA step.
//
//   q            : [batch, num_heads, 128]                          fp8
//   k_cache      : [num_blocks, num_kv_heads, 128/16, 16, 16]       fp8 (x = 16)
//   v_cache      : [num_blocks, num_kv_heads, 128, 16]              fp8
//   out          : [batch, num_heads, 128]                          bf16
//
// Scales are per-tensor, in the dequant direction (x_true ~= x_fp8 * scale). All
// three fold into multiplies the kernel already performs, so they are free.
void pa_decode_opus_fp8_fwd(aiter_tensor_t& q,
                            aiter_tensor_t& k_cache,
                            aiter_tensor_t& v_cache,
                            aiter_tensor_t& block_tables,
                            aiter_tensor_t& context_lens,
                            aiter_tensor_t& out,
                            float softmax_scale,
                            float q_scale,
                            float k_scale,
                            float v_scale);

// Persistent variant of the A8W8 path. One workgroup per CU, each draining the work
// items `get_pa_metadata_v1` assigned it, so a batch of mixed context lengths balances
// instead of every row paying the longest row's split count.
//
// Pages are addressed CSR-style (`kv_indptr`/`kv_indices`) rather than through a
// rectangular block table, matching the metadata kernel's contract. Work items the
// metadata marked as splits land in `split_o`/`split_lse` as (normalized O,
// natural-log LSE) for `pa_reduce_v1`; the rest write `out` directly.
//
//   kv_indptr    : [batch + 1]                   int32, prefix sum of used pages
//   kv_indices   : [sum of used pages]           int32
//   work_indptr  : [num_cu + 1]                  int32, also fixes the grid
//   work_info    : [num_works, 8]                int32
//   split_o      : [num_partial_tiles, num_heads, 128]  fp32
//   split_lse    : [num_partial_tiles, num_heads]       fp32
//
// Requires `kv_granularity == page size` when building the metadata, so that the work
// items' kv_start/kv_end are page indices.
void pa_decode_opus_fp8_ps_fwd(aiter_tensor_t& q,
                               aiter_tensor_t& k_cache,
                               aiter_tensor_t& v_cache,
                               aiter_tensor_t& kv_indptr,
                               aiter_tensor_t& kv_indices,
                               aiter_tensor_t& context_lens,
                               aiter_tensor_t& out,
                               aiter_tensor_t& work_indptr,
                               aiter_tensor_t& work_info,
                               aiter_tensor_t& split_o,
                               aiter_tensor_t& split_lse,
                               float softmax_scale,
                               float q_scale,
                               float k_scale,
                               float v_scale);

// A16W8: bf16 Q against the same fp8 KV cache, which is what the rest of the PA family
// takes. The MFMA needs both operands the same width, so the kernel quantizes Q once
// per query row on the way in -- a row-wise absmax outside the loop, nothing inside it.
// There is no q_scale argument for that reason.
//
// Bandwidth is unchanged from the A8W8 path: Q is read once per workgroup while KV
// streams, well under 1% of the traffic.
// gpt-oss decode: 64-dim heads, fp8 KV, and a learned per-head attention sink. Q may
// be fp8 (A8W8, q_scale used) or bf16 (A16W8, q_scale ignored -- the kernel derives one
// per query row); the dtype of `q` selects between them.
//
//   q     : [batch, num_heads, 64]                        fp8 or bf16
//           or [batch, qlen, num_heads, 64] for MTP. Tail-causal: query i attends
//           the first context_len - qlen + 1 + i KV tokens. qlen * gqa <= 16.
//   k_cache: [num_blocks, num_kv_heads, 64/16, PAGE, 16]  fp8
//   v_cache: [num_blocks, num_kv_heads, 64, PAGE]         fp8
//   sink  : [num_heads]                                   fp32 (shared across MTP tokens)
//   out   : shaped like q, bf16
//
// `sink` holds one logit per query head in the same scaled-logit domain as
// (q.k) * softmax_scale, matching the gfx1250 kernel's convention: it is a KV column
// with no value, so it lands in the softmax denominator and nowhere else. MTP tokens
// that share a query head share that logit; each token still has its own softmax.
void pa_decode_opus_gptoss_fwd(aiter_tensor_t& q,
                               aiter_tensor_t& k_cache,
                               aiter_tensor_t& v_cache,
                               aiter_tensor_t& block_tables,
                               aiter_tensor_t& context_lens,
                               aiter_tensor_t& out,
                               aiter_tensor_t& sink,
                               float softmax_scale,
                               float q_scale,
                               float k_scale,
                               float v_scale);

// Persistent form, on the same work queue as pa_decode_opus_fp8_ps_fwd.
void pa_decode_opus_gptoss_ps_fwd(aiter_tensor_t& q,
                                  aiter_tensor_t& k_cache,
                                  aiter_tensor_t& v_cache,
                                  aiter_tensor_t& kv_indptr,
                                  aiter_tensor_t& kv_indices,
                                  aiter_tensor_t& context_lens,
                                  aiter_tensor_t& out,
                                  aiter_tensor_t& sink,
                                  aiter_tensor_t& work_indptr,
                                  aiter_tensor_t& work_info,
                                  aiter_tensor_t& split_o,
                                  aiter_tensor_t& split_lse,
                                  float softmax_scale,
                                  float q_scale,
                                  float k_scale,
                                  float v_scale);

void pa_decode_opus_a16w8_fwd(aiter_tensor_t& q,
                              aiter_tensor_t& k_cache,
                              aiter_tensor_t& v_cache,
                              aiter_tensor_t& block_tables,
                              aiter_tensor_t& context_lens,
                              aiter_tensor_t& out,
                              float softmax_scale,
                              float k_scale,
                              float v_scale,
                              std::optional<aiter_tensor_t> k_scale_map = std::nullopt,
                              std::optional<aiter_tensor_t> v_scale_map = std::nullopt);

void pa_decode_opus_a16w8_ps_fwd(aiter_tensor_t& q,
                                 aiter_tensor_t& k_cache,
                                 aiter_tensor_t& v_cache,
                                 aiter_tensor_t& kv_indptr,
                                 aiter_tensor_t& kv_indices,
                                 aiter_tensor_t& context_lens,
                                 aiter_tensor_t& out,
                                 aiter_tensor_t& work_indptr,
                                 aiter_tensor_t& work_info,
                                 aiter_tensor_t& split_o,
                                 aiter_tensor_t& split_lse,
                                 float softmax_scale,
                                 float k_scale,
                                 float v_scale);

#ifdef PA_DECODE_OPUS_IMPL
// ============================================================================
// Implementation section - only compiled in the .cu translation unit
// ============================================================================

#include <type_traits>

using bf16_t = __bf16;
// Matches opus's REGISTER_DTYPE, so the MFMA dispatch selects its fp8 branch.
using fp8_t = _BitInt(8);

// One tuning decision for this kernel lives outside this header, in the module's JIT
// flags (aiter/jit/optCompilerConfig.json), because it is a compiler flag rather than
// a template parameter: -mllvm -amdgpu-mfma-vgpr-form=1.
//
// Left to itself the compiler accumulates MFMA into AGPRs, and softmax then has to
// read every score back out with v_accvgpr_read -- 132 pure-overhead moves across the
// gpt-oss kernel. The flag keeps accumulators in arch VGPRs, which removes all of
// them: every one of the 22 instantiations drops registers (the gpt-oss shape 96 to 80
// alloc, the widest d192 one 168 to 152), none gains a spill, and most gain a wave or
// two per SIMD. Worth 0.3% at b64/ctx16k -- small, because instruction count is not
// what binds here (see PA_DECODE_OPUS_K16), but it costs nothing and nothing regresses
// across op_tests/bench_pa_decode_opus_vs_asm.py.
//
// Do not reach for launch bounds to get the same effect. amdgpu-waves-per-eu also
// clears the AGPRs, but attaching it switches LLVM to the occupancy-driven scheduler
// and the kernel comes out slower even when the bound never binds; see
// docs/pa_decode_opus_launch_bounds.md. Toggle the flag with .mfma_form_toggle.py.

// KV tile depth: how many KV tokens one main-loop iteration consumes. K and V go
// straight into the MFMA fragments, so the tile is paid for in registers and sets
// occupancy directly. Tune with op_tests/sweep_kv_tile.sh.
#ifndef PA_DECODE_KV_TILE
#define PA_DECODE_KV_TILE 128
#endif

// Which of the two KV streams is loaded non-temporally: bit 0 is K, bit 1 is V.
// See load_stream for what the choice buys and what it gives up. Swept over both
// streams x three page-reuse patterns x batch 64..128 x ctx 4k..16k: K alone is the
// only setting that never loses. Streaming V as well is worth another 1-2% where
// there is no reuse at all and costs 6-20% where there is.
#ifndef PA_DECODE_OPUS_NT_KV
#define PA_DECODE_OPUS_NT_KV 1
#endif
#define PA_DECODE_OPUS_NT_K ((PA_DECODE_OPUS_NT_KV & 1) != 0)
#define PA_DECODE_OPUS_NT_V ((PA_DECODE_OPUS_NT_KV & 2) != 0)

// gpt-oss fp8 K=32: the cache pack is 16B, the MFMA B fragment is 8B, so a pack spans
// two lane groups and has to be split in registers.
//   0  no packs -- each lane loads its own 8B fragment. 64 lanes x 8B = 512B a load.
//   1  even k_grp loads the pack, v_permlane16_swap hands the high 8B to the odd one.
//      Half the wave is masked off, so this is *also* 512B a load: it cuts nothing.
//   2  every k_grp owns one pack, so all 64 lanes issue -- 1024B a load, half the K
//      loads of either form above, and no exec mask. permlane16 then permlane32
//      redistribute; see convert_k_pack_frags for the two-step derivation.
//   3  as 2's load, but nothing is redistributed: the contraction order is permuted so
//      that the pack a lane already holds is the pair of fragments the MFMA wants, its
//      low 8B for i_k 0 and its high 8B for i_k 1. An MFMA sums over the contraction
//      dim, so permuting it costs nothing as long as Q is walked in the same order,
//      and Q is read once before the loop. Half the K loads and no permutes.
// LDS staging of the same packs was a wash or a loss (the lgkmcnt at the loop top ate
// the VMEM-op cut).
//
// 0 ships. Measured on a quiet gfx950, page 256, HIP graph, b64/ctx16k: mode 0 178us,
// mode 1 183us, mode 2 183us, mode 3 184us. Mode 1 cannot help by construction -- half
// the wave is masked off, so its dwordx4 moves the same 512B as mode 0's 8B load, for
// an extra branch and eight permutes.
//
// Mode 3 is why the losers are kept here rather than deleted: it is the idea with
// nothing left to blame, and it still loses. It does precisely what it was built to do
// -- 28% fewer VMEM issues, 14% fewer VALU, no permutes at all, byte-identical fetches
// (FETCH_SIZE within 0.1%), the same 96 VGPR and 5 waves/SIMD -- and comes out 3.5%
// slower, 177.8 to 184.0us, with under 0.3us of spread over three alternating passes.
// So instruction count does not bind either. Neither does loads in flight, nor
// occupancy: PREFETCH_DEPTH and WGS_PER_CU each moved their own metric hard and left
// the runtime alone. What tracks the loss is waiting -- SQ_WAIT_ANY moves with the
// runtime to a tenth of a point -- and the memory-pipe counter that jumps with it is
// the address unit stalling on the cache, 0.95M to 2.68M cycles. A lane group owning a
// whole 16B pack puts the wave's four 256B runs 4KiB apart where the 8B form spans
// two: the same cache lines, with half as many instructions to spread them over.
// Reproduce with op_tests/ab_opus_only.sh and op_tests/counter_ab.sh.
//
// A thread trace says where that waiting sits, and it is not spread thin. Traced on one
// CU against Gluon at bs8/ctx4k, the two run within half a percent of the same number
// of instructions -- 6686 against 6712 -- and stall for twice as long, 74368 cycles
// against 35992. Four fifths of ours is two lines: s_waitcnt at 61% and s_barrier at
// 19%. Per instruction the gap is not that we wait more often but that each wait is
// longer -- 132 cycles a waitcnt against Gluon's 26, 162 a barrier against 46 -- and
// the single hottest site is one s_waitcnt vmcnt(2) at 564 cycles a hit, a whole HBM
// round trip standing uncovered, with the barrier behind it charging another 1090 for
// the skew it leaves between waves. So what is left to find is not instructions and not
// bandwidth: it is that the KV loads are issued too close to the MFMAs that consume
// them for the latency to fit underneath. Recipe in docs/opus_pa_qa.md Q42, read the
// result with .att_digest.py and .att_hotspot.py.
#ifndef PA_DECODE_OPUS_K16
#define PA_DECODE_OPUS_K16 0
#endif

// Force SMEM block-table loads for pages that hold a tile. A uniform ordinary
// pointer still generated global_load_dword on gfx950; constant address space is
// required here to obtain s_load_dword and move the dependency off vmcnt.
// The table is read-only during a dispatch; load_page_ids clamps speculative reads.
// The host separately selects an SMEM specialization for long A8W8 D128/page128
// splits. Short splits retain the original buffer kernel without a loop branch.
// See docs/pa_decode_opus_a8w8_d128_pingpong_smem_20260910.md.
#ifndef PA_DECODE_OPUS_BT_SCALAR
#define PA_DECODE_OPUS_BT_SCALAR 0
#endif

#ifndef PA_DECODE_OPUS_A8W8_D128_BT_SCALAR
#define PA_DECODE_OPUS_A8W8_D128_BT_SCALAR 1
#endif

// How GEMM0 is scheduled against the work that surrounds it.
//   0  one hard sched_barrier before GEMM0. The block-table load is left with about a
//      dozen instructions of cover, so its wait lands as a vmcnt(0) between the first
//      and second MFMA -- visible in the ISA on every variant.
//   1  pin only the block-table reads at the top and leave the rest one scheduling
//      region, so the compiler may sink that wait among the MFMAs itself.
//   2  as 1, plus sched_group_barrier: GEMM0's MFMAs issue as a block ahead of the
//      address math that consumes the block table, and on the 16B path each pack's
//      register split is paired with the MFMA that consumes it -- Gluon's iglp shape.
//
// 0 ships. 1 and 2 both land in the ISA as intended -- under 2 the waits interleave
// with the MFMAs and the vmcnt(0) moves from the first MFMA to the last -- and neither
// is worth anything: b64/ctx16k/page256 measures 178.2 / 178.0 / 178.8us. The drain
// this was aimed at is already covered, so the default stays on the schedule the rest
// of the kernel was tuned against.
#ifndef PA_DECODE_OPUS_GEMM0_SCHED
#define PA_DECODE_OPUS_GEMM0_SCHED 0
#endif

// How many tiles ahead the KV prefetch runs.
//   1  a tile's KV is fetched during the body before it, so one tile's worth of loads
//      is in flight and the hot loop peaks at 6 outstanding.
//   2  double-buffered: fetched two bodies ahead, so two tiles overlap and the peak
//      goes to 16. Costs a second copy of the K and V staging registers plus a second
//      block-table set, and pairs the bodies so the buffer index stays a compile-time
//      constant. Only the shapes in Traits::PREFETCH_DEPTH can afford it.
//
// 1 ships. On the gpt-oss shape depth 2 trades occupancy for pipeline: 96 VGPR and 5
// waves/SIMD become 152 and 3. That is the right trade only if the loop is starved of
// loads in flight, and the evidence says it is not -- doubling the waves per SIMD by
// hand (WGS_PER_CU 2 -> 4, verified with check_opus_split_grid.py to really go from
// 512 to 1024 workgroups) left b64/ctx16k at 178.7us, unchanged. Measured directly on
// a quiet GPU it is a wash: 175.6us mean at depth 1 against 174.3 at depth 2, a gap
// smaller than depth 1's own 5.7us spread across the same three passes. Depth 1 wins
// on everything else, so it stays. Use op_tests/ab_opus_only.sh, not the paired bench
// -- with both kernels in one process Gluon's own time shifts by 4-6% with whichever
// OPUS variant was compiled in, which moves the ratio the bench reports.
#ifndef PA_DECODE_OPUS_PREFETCH_DEPTH
#define PA_DECODE_OPUS_PREFETCH_DEPTH 1
#endif

#ifndef PA_DECODE_OPUS_V_CHUNK_PIPELINE
#define PA_DECODE_OPUS_V_CHUNK_PIPELINE 0
#endif

#ifndef PA_DECODE_OPUS_V5_FRAGMENT_REFILL
#define PA_DECODE_OPUS_V5_FRAGMENT_REFILL 0
#endif

// Page-16 dual-tile software pipeline: two KV_TILE=128 bodies share one
// 256-token softmax window without widening
// MFMA N. Extra live scores stay under the fused-MTP occupancy cliff;
// K is double-buffered, V stays DEPTH=1 and is refilled after the first PV.
#ifndef PA_DECODE_OPUS_PAGE16_PAIR
#define PA_DECODE_OPUS_PAGE16_PAIR 1
#endif

// Fused MTP keeps one live C fragment and spills the rest of S to LDS (bf16).
// That is what lets PAGE16_PAIR open on MTP_Q_LOOP without eight fp32 score
// tiles. q-split overrides S_VIA_LDS off; it never loops tokens.
#ifndef PA_DECODE_OPUS_S_VIA_LDS
#define PA_DECODE_OPUS_S_VIA_LDS 1
#endif

static_assert(PA_DECODE_OPUS_PREFETCH_DEPTH == 1 || PA_DECODE_OPUS_PREFETCH_DEPTH == 2,
              "prefetch depth is the buffer count, and only 1 and 2 are implemented");

// Resident workgroups per CU the split heuristic aims for on a 64-dim head. Sweep
// with op_tests/sweep_k16_sched.sh --wgs. A 256-thread workgroup is one wave per
// SIMD, so this is also the wave count each SIMD gets, and on a DRAM-bound decode it
// is what sets how many KV loads are in flight.
#ifndef PA_DECODE_OPUS_WGS_PER_CU_D64
#define PA_DECODE_OPUS_WGS_PER_CU_D64 2
#endif

#ifndef PA_DECODE_OPUS_A8W8_LONG_SPLITS
#define PA_DECODE_OPUS_A8W8_LONG_SPLITS 1
#endif

// Dead weight added to the decode kernel's LDS allocation, to hold resident
// workgroups below what registers would allow. Zero in every shipped build; see the
// declaration in pa_decode_opus_kernel for what it is for.
#ifndef PA_DECODE_OPUS_LDS_PAD
#define PA_DECODE_OPUS_LDS_PAD 0
#endif

// Widest head dim any instantiation below compiles. The split-KV scratch is one
// per-stream buffer shared by all of them, so it is sized against this rather than
// against whichever traits happen to allocate it first.
static constexpr int PA_DECODE_MAX_D_HEAD = 128;


// Kernel arguments.
struct pa_decode_kargs
{
    const void* __restrict__ q_ptr;       // [batch, num_heads, D]
    const void* __restrict__ k_ptr;       // [num_blocks, num_kv_heads, D/K_PACK, PAGE, K_PACK]
    const void* __restrict__ v_ptr;       // [num_blocks, num_kv_heads, D, PAGE]
    void* __restrict__ out_ptr;           // [batch, num_heads, D]
    const int* __restrict__ block_tables; // [batch, max_blocks_per_batch_row]
    const int* __restrict__ context_lens; // [batch]
    // Split-KV scratch, only used when num_splits > 1.
    float* __restrict__ partial_o;  // [batch][num_kv_heads][num_splits][Q_TILE][D]
    float* __restrict__ partial_ml; // [batch][num_kv_heads][num_splits][2][Q_TILE]
    // One arrival counter per (batch, kv-head), left zeroed by whichever split
    // arrives last, so no per-call memset is needed.
    unsigned int* __restrict__ split_counters;
    int num_splits;
    int batch;
    int num_heads;
    int num_kv_heads;
    int gqa_ratio; // num_heads / num_kv_heads
    int max_blocks_per_batch_row;
    int stride_q_b; // elements, one batch row of q ([batch, num_heads, D])
    int stride_q_h; // elements, one q head
    int stride_o_b;
    int stride_o_h;
    int stride_k_blk; // elements, one page of all kv heads
    int stride_k_h;   // elements, one kv head inside a page
    int stride_v_blk;
    int stride_v_h;
    float softmax_scale;
    // fp8 dequant, per-tensor. Both fold into multiplies the kernel already does,
    // so they cost nothing in the main loop. 1.0f on the bf16 path.
    float qk_dequant; // q_scale * k_scale, folded into the softmax temperature
    float v_dequant;  // v_scale, folded into the final 1/l
    // Persistent path only, and kept at the tail so the non-persistent kernel's
    // kernarg offsets -- and with them its scalar load schedule -- do not move.
    //
    // The work queue comes from the shared PA metadata kernel, so KV pages are
    // addressed CSR-style through kv_indices rather than the rectangular block table,
    // and the partials use the shared reduce kernel's (normalized O, natural-log LSE)
    // layout rather than the fused merge's (unnormalized O, m, l).
    const int* __restrict__ kv_indptr;   // [batch + 1], prefix sum of used pages
    const int* __restrict__ kv_indices;  // [sum of used pages]
    const int* __restrict__ work_indptr; // [num_cu + 1]
    const int* __restrict__ work_info;   // [num_works][8]
    float* __restrict__ split_o;         // [num_partial_tiles][num_heads][D]
    float* __restrict__ split_lse;       // [num_partial_tiles][num_heads]
    // Learned attention sink, one logit per query head, in the same scaled-logit
    // domain as (q.k) * softmax_scale -- so it is not scaled again. Only read by the
    // instantiations whose traits ask for it.
    const float* __restrict__ sink; // [num_heads]
    // Multi-token prediction. Only read by the HAS_MTP instantiations, and kept behind
    // the persistent block for the same reason that block sits where it does: the
    // decode kernels' kernarg offsets, and their scalar load schedule, must not move.
    //
    // q and out gain a token dim, [batch, qlen, num_heads, D]. A tile row is the pair
    // (token, head) at row = token * gqa_ratio + head, so qlen * gqa_ratio <= Q_TILE.
    int qlen;
    int stride_q_t; // elements, one query token of q
    int stride_o_t;
    int short_seq_splits;
    // Per-token KV dequant, [num_blocks, num_kv_heads, PAGE, 1]. Null on the
    // per-tensor path, which keeps k_scale/v_scale folded into qk_dequant/v_dequant.
    // Appended so existing kernarg offsets do not move.
    const float* __restrict__ k_scale_map;
    const float* __restrict__ v_scale_map;
    int stride_ks_blk;
    int stride_ks_h;
    int stride_vs_blk;
    int stride_vs_h;
};

// Tile shape / MFMA configuration.
//
// A workgroup owns one (batch, kv-head), or one split of its context, and walks
// that range one KV_TILE at a time. Per tile, with M the query rows throughout:
//
//   S [Q_TILE, KV_TILE] = Q [Q_TILE, D_HEAD] * K^T    GEMM0, contracts D_HEAD
//   P [Q_TILE, KV_TILE] = softmax(S)                  also rescales the running O
//   O [Q_TILE, D_HEAD] += P * V^T                     GEMM1, contracts KV_TILE
//
// Page-16 additionally pairs two consecutive tiles into one 256-token softmax
// window (PAGE16_PAIR): same MFMA N, half as many max-fold/P barriers,
// without widening KV_TILE.
//
// Q is loaded once and O accumulates across tiles, so only K and V stream. Q_TILE
// is the MFMA's M and also the largest GQA ratio supported, so one tile carries
// every query head that shares this kv-head.
//
// D_ATTN_ is the matrix-core operand type, shared by K, V and P. bf16_t selects
// v_mfma_f32_16x16x32_bf16, fp8_t the full-rate v_mfma_f32_16x16x128_f8f6f4. The
// output stays bf16 either way.
//
// D_Q_ is what Q arrives as, which need not match. Giving bf16 Q an fp8 D_ATTN is the
// A16W8 model the rest of the PA family uses: the MFMA needs both operands the same
// width, so Q is quantized once on the way in, per query row. It costs a row-wise
// absmax outside the loop and nothing inside it, and it barely moves the bandwidth --
// Q is read once per workgroup while KV streams, well under 1% of the traffic.
template<typename D_ATTN_ = bf16_t,
         typename D_Q_    = D_ATTN_,
         int D_HEAD_      = 128,
         bool HAS_SINK_   = false,
         int KV_TILE_     = 128,
         int PAGE_SIZE_   = 16,
         int NUM_WARPS_   = 4,
         bool HAS_MTP_    = false,
         bool MTP_Q_LOOP_ = false>
struct pa_decode_traits
{
    using D_ATTN = D_ATTN_;
    using D_Q    = D_Q_;
    using D_OUT  = bf16_t;
    using D_ACC  = float;

    // Learned per-query-head attention sink, as gpt-oss uses. Compiled in rather than
    // branched on so the models without one keep exactly the code they had.
    static constexpr bool HAS_SINK = HAS_SINK_;

    // Multi-token prediction: a tile row is a (token, head) pair rather than just a
    // head, so `qlen * gqa_ratio` rows are live instead of `gqa_ratio`. Also compiled
    // in, because it costs the tail-causal mask a second peeled tile and turns two
    // single-stride addressings into per-lane ones -- none of which decode should pay.
    static constexpr bool HAS_MTP = HAS_MTP_;
    // When GQA fills Q_TILE, extra query tokens cannot pack into the same 16-row
    // MFMA. Loop them inside each KV tile so K/V stay in registers. gqa == Q_TILE
    // and qlen <= MTP_Q_MAX. The packed HAS_MTP path is unchanged.
    static constexpr bool MTP_Q_LOOP = MTP_Q_LOOP_;
    static constexpr int MTP_Q_MAX   = 4;
    static_assert(!MTP_Q_LOOP || HAS_MTP, "token-loop MTP is a HAS_MTP specialization");
    // One CTA per query token. Used when the fused token loop would leave the machine
    // idle: each token keeps the decode body (and its KV pipeline) and the grid grows
    // by qlen for small-MTP query splitting.
    static constexpr bool MTP_Q_SPLIT = false;
    static constexpr bool PER_TOKEN_SCALE = false;

    static constexpr bool BT_SCALAR = PA_DECODE_OPUS_BT_SCALAR != 0;
    static constexpr bool V_SHUFFLED = false;
    static constexpr bool IS_FP8 = std::is_same_v<D_ATTN, fp8_t>;
    static_assert(IS_FP8 || std::is_same_v<D_ATTN, bf16_t>, "D_ATTN must be bf16_t or fp8_t");
    // Q wider than the matrix core's operand: quantize it on load.
    static constexpr bool QUANT_Q = !std::is_same_v<D_Q, D_ATTN>;
    static_assert(!QUANT_Q || (IS_FP8 && std::is_same_v<D_Q, bf16_t>),
                  "the only mixed pairing is bf16 Q with an fp8 matrix core");
    // A16W8: PA body fills the GPU with one split per CU, then a second grid
    // merges one query row per workgroup. Fused last-arriver left 255 CUs idle
    // at NP=256 and was slower than NP=16.
    static constexpr bool SEPARATE_SPLIT_REDUCE =
        QUANT_Q && !HAS_SINK && IS_FP8 && D_HEAD_ == 128;

    static constexpr int D_HEAD    = D_HEAD_;
    // Resident workgroups per CU the split heuristic aims for. A 64-dim head streams
    // half the KV bytes per workgroup, so it needs twice the concurrency before the
    // machine is saturated; a 128-dim head gets there on its own and splitting further
    // only buys more partials to merge. Measured on gfx950 (batch 16, gqa 8, ctx 16k):
    // 2 takes d64 from 61.7us to 47.1us, while on d128 the same value costs 3-5%
    // across batch 8..32. Bounded by PA_DECODE_WGS_PER_CU, which sizes the shared
    // split scratch and so has to cover whichever instantiation asks for the most.
    static constexpr int WGS_PER_CU = D_HEAD_ <= 64 ? PA_DECODE_OPUS_WGS_PER_CU_D64 : 1;
    static constexpr int KV_TILE   = KV_TILE_;
    static constexpr int PAGE_SIZE = PAGE_SIZE_;
    static constexpr int NUM_WARPS = NUM_WARPS_;
    static constexpr int WARP_SIZE = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;
    static constexpr int Q_TILE     = 16; // MFMA M, also the max supported GQA ratio

    // Upper bound on KV splits: the fused merge indexes LDS by split, so this
    // statically sizes that buffer. 256 allows up to one partition per CU
    // (one split per CU on a 256-CU gfx950).
    static constexpr int MAX_SPLITS = 256;

    // Wave tiling: 1 wave along M, all waves along N.
    static constexpr int T_M = 1;
    static constexpr int T_N = NUM_WARPS;
    static constexpr int T_K = 1;

    // gfx950 native matrix core. bf16 has one 16x16 shape, K=32. fp8 has two: the
    // full-rate f8f6f4 at K=128, and a K=32 form. The wide one can only be used when
    // the head dim is at least as wide as its contraction, so a 64-dim head falls back
    // to K=32 -- which costs nothing measurable, since this kernel is memory bound.
    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = (IS_FP8 && D_HEAD >= 128 && KV_TILE >= 128) ? 128 : 32;
    // Only the K=128 f8f6f4 form takes scale operands; the K=32 fp8 MFMA does not.
    static constexpr bool SCALED_MFMA = IS_FP8 && W_K == 128;

    static constexpr int ELEM_A = W_M * W_K / WARP_SIZE; // 8 at K=32, 32 at K=128
    static constexpr int ELEM_B = W_N * W_K / WARP_SIZE;
    static constexpr int ELEM_C = W_M * W_N / WARP_SIZE; // 4, consecutive along N

    // GEMM0: S[Q_TILE, KV_TILE] = Q[Q_TILE, D_HEAD] * K^T
    static constexpr int GEMM0_E_M = Q_TILE / (W_M * T_M);
    static constexpr int GEMM0_E_N = KV_TILE / (W_N * T_N);
    static constexpr int GEMM0_E_K = D_HEAD / (W_K * T_K);

    // GEMM1: O[Q_TILE, D_HEAD] = P[Q_TILE, KV_TILE] * V^T
    static constexpr int GEMM1_E_M = Q_TILE / (W_M * T_M);
    static constexpr int GEMM1_E_N = D_HEAD / (W_N * T_N);
    static constexpr int GEMM1_E_K = KV_TILE / (W_K * T_K);

    // Two regimes, and the code below branches on which. Pages narrower than a KV tile
    // means a tile is built out of several pages, one per K MFMA tile -- the original
    // shape, where the page id *is* the N-tile selector. Pages at least a tile wide
    // (gpt-oss ships 256) means the reverse: a tile sits entirely inside one page, the
    // N-tiles become offsets within it, and several tiles share the page.
    static constexpr bool PAGE_HOLDS_TILE = PAGE_SIZE >= KV_TILE;
    static constexpr int PAGES_PER_TILE   = PAGE_HOLDS_TILE ? 1 : KV_TILE / PAGE_SIZE;
    static constexpr int TILES_PER_PAGE   = PAGE_HOLDS_TILE ? PAGE_SIZE / KV_TILE : 1;
    // vLLM packs the K cache as [D/x, PAGE, x] with x = 16 bytes / sizeof(dtype).
    // That is a property of the cache layout and does not have to match the load
    // width below.
    static constexpr int K_PACK = 16 / sizeof(D_ATTN);
    // One load's width: the widest 16B vector the dtype allows, but never more than a
    // lane's whole slice of the contraction dim. fp8 at K=32 holds only 8 elements per
    // lane, so there VEC is 8. When K16_PACK is on, global K loads are 16B packs and
    // VEC is the 8B fragment after the register split.
    static constexpr int VEC = (16 / sizeof(D_ATTN)) < ELEM_A ? (16 / sizeof(D_ATTN)) : ELEM_A;
    // Q's own load width. Same rule, in Q's element size, so a wider Q simply takes
    // more issues over the same fragment.
    static constexpr int Q_VEC = (16 / sizeof(D_Q)) < ELEM_A ? (16 / sizeof(D_Q)) : ELEM_A;
    // fp8 e4m3's largest finite value; the divisor for the per-row Q scale.
    static constexpr float FP8_MAX = 448.0f;
    // How many loads one lane's slice of the contraction dim takes. 1 unless the slice
    // is wider than a single vector -- fp8 at K=128 spans two. Every load below is
    // written against REPT so all the shapes share one code path.
    static constexpr int REPT = ELEM_A / VEC;

    // Lane counts, not thread groups. The 64 lanes form a GRP_K by GRP_N grid:
    // lane % GRP_N picks the N position, lane / GRP_N the contraction slice of VEC
    // elements. Restates mfma_adaptor::grpn_b/::grpk_b, which the static_asserts
    // below cannot reach because they run before any mma object exists.
    static constexpr int GRP_N = W_N;             // == mfma_adaptor::grpn_b
    static constexpr int GRP_K = WARP_SIZE / W_N; // == mfma_adaptor::grpk_b
    // How many lane groups share one page when walking V along tokens. Only meaningful
    // in the narrow-page regime; a page holding a whole tile holds every group.
    static constexpr int KGRP_PER_PAGE = PAGE_HOLDS_TILE ? GRP_K : PAGE_SIZE / VEC;

    // Gluon-style K: the cache pack is 16B and the MFMA B fragment is 8B, so the pack
    // has to be split across lanes. Only the d64 fp8 K=32 shape has that mismatch;
    // everything else already loads 16B straight into one fragment.
    static constexpr bool K16_PACK = PA_DECODE_OPUS_K16 && VEC == 8 && K_PACK == 16 && REPT == 1;
    // The wide form: D_HEAD holds exactly GRP_K packs, so every lane group can own one
    // and all 64 lanes issue the dwordx4. The narrow form loads on half the wave, which
    // costs the same VMEM issues as the 8B path for half the payload -- see load_k_pack.
    static constexpr bool K16_WIDE = K16_PACK && PA_DECODE_OPUS_K16 >= 2
                                     && D_HEAD / K_PACK == GRP_K && GEMM0_E_K == 2;
    // The permuted form loads exactly as the wide one does, then keeps the pack as it
    // landed: lane group g owns head dims [g * K_PACK, (g+1) * K_PACK) and MFMA step
    // i_k takes the i_k-th VEC slice of them. That reorders the contraction dim, which
    // is only sound if Q is read in the same order -- so it needs the quantizing Q
    // path, the one that walks the head dim explicitly. A non-quantizing Q goes
    // through the DSL's A partition, which owns that order, and falls back to K16_WIDE.
    static constexpr bool K16_PERM = K16_WIDE && PA_DECODE_OPUS_K16 >= 3 && QUANT_Q;
    static constexpr int K_PACK_ITERS =
        K16_WIDE ? GEMM0_E_N : (K16_PACK ? GEMM0_E_N * GEMM0_E_K : 0);

    // Only P is staged through LDS: GEMM1 contracts the token dim that splits the
    // waves, so every element crosses lanes. K, V and S never leave registers.
    // The tile is [Q_TILE][KV_TILE]; PAD staggers rows off each other's banks.
    // 16B of padding either way: keeps the row stride 16B-aligned for the b128 P
    // reads and puts consecutive rows 4 banks apart (272B for bf16, 144B for fp8,
    // both 4 mod 32 dwords).
    static constexpr int PAD          = VEC;
    static constexpr int P_ROW_STRIDE = KV_TILE + PAD;

    // Softmax leaves P in [0, 1], which would waste all of fp8 e4m3's exponent
    // range above 1 and quantize with a 2^-9 floor. Shifting the exp2 argument up
    // by P_LOG2 moves the row max to 2^P_LOG2 instead. It costs nothing: the shift
    // rides along in the `- m_run` that exp2 already does, and because l sums the
    // same shifted values, the factor cancels in O/l and never has to be undone.
    static constexpr int P_LOG2 = IS_FP8 ? 8 : 0;

    // Page-16 dual-tile SWP: two 128-token K tiles, one combined max, two P
    // halves. Does not change KV_TILE / GEMM0_E_N. Off for page>=tile (LDS V)
    // and the 16B K split, both of which already sit on the VGPR cliff.
    // Decode / q-split only. Fused MTP_Q_LOOP keeping two 128-halves plus four
    // scores lands at ~273 VGPR / 1 wave; serving B=200 cannot afford that, and
    // a sequential dual-tile there did not move NP=256.
    static constexpr bool S_VIA_LDS = PA_DECODE_OPUS_S_VIA_LDS && MTP_Q_LOOP
                                      && !PAGE_HOLDS_TILE && !K16_PACK;
    static constexpr bool PAGE16_PAIR = PA_DECODE_OPUS_PAGE16_PAIR && PAGE_SIZE == 16
                                        && KV_TILE == 128 && D_HEAD == 128
                                        && !PAGE_HOLDS_TILE && !K16_PACK && HAS_MTP
                                        && (!MTP_Q_LOOP || S_VIA_LDS);
    // Fused MTP publishes every query token's P (and its max-fold scratch) into
    // private LDS slices so one barrier covers all of them. Decode keeps one
    // tile; PAGE16_PAIR doubles that along KV so both halves of the 256-token
    // window sit in LDS across one P barrier.
    static constexpr int smem_p_q_tiles  = MTP_Q_LOOP ? MTP_Q_MAX : 1;
    static constexpr int smem_p_kv_tiles = PAGE16_PAIR ? 2 : 1;
    static constexpr int smem_p_tiles    = smem_p_q_tiles * smem_p_kv_tiles;
    static constexpr int smem_p_elems    = smem_p_tiles * Q_TILE * P_ROW_STRIDE;
    using S_STORE                        = bf16_t;
    static constexpr int S_FRAG          = GEMM0_E_N * ELEM_C;
    static constexpr int smem_s_kv_tiles = S_VIA_LDS ? smem_p_kv_tiles : 0;
    static constexpr int smem_s_elems    = S_VIA_LDS
        ? MTP_Q_MAX * smem_s_kv_tiles * BLOCK_SIZE * S_FRAG
        : 0;
    // Cross-wave row reduction scratch, [tile][row][wave] so a row's T_N values
    // fold in one 16B LDS read: each tile's max fold, then the final l sum.
    // Sized on query tiles only: the 256-token window still folds one max per
    // query row, not one per KV half.
    static constexpr int smem_row_reduce_elems = smem_p_q_tiles * Q_TILE * T_N;

    // V staged coalesced into LDS, then picked apart there, which a wide page needs
    // for coalescing: its V cache's head-dim stride is the page size, so the fragment
    // read pattern -- 16 lanes stepping head dims -- lands 16 lanes on 16 different
    // cache lines and uses 8 bytes of each, and staging trades that for one line per
    // 8 lanes. No barrier: GEMM1 splits head dims across waves, so a wave stages only
    // the dims it will read itself.
    //
    // A narrow page reads V contiguously already, so staging there would buy load
    // count rather than coalescing -- one chunk per page row instead of a VEC-wide
    // piece per contraction step, halving the V loads on the 64-dim fp8 shape, which
    // is worth something because that body issues enough memory instructions to back
    // up the load queue. Measured, it does not pay: the staging round trip lands in
    // the dependency chain, and at batch 64, where two workgroups per CU cannot cover
    // it, that costs 3-6% -- more than the 1-2% it returns at batch 128, where four
    // can. So narrow pages keep reading V straight from the cache.
    //
    // V_DMA is the widest global load, so 8 lanes cover a tile's 128 tokens for one
    // head dim -- one cache line, fully used. The row is padded by one load so that
    // the fragment read, which steps head dims across lanes, spreads over all 32
    // banks instead of piling onto one.
    //
    // gfx950 can skip the register stop with global_load_lds_dwordx4, which does
    // remove the LDS write. Measured at 0.3-0.75% slower across page 256, against a
    // 0.1% noise floor, so it is not here.
    //
    // The wait that write sits behind is the kernel's top stall, but the copy does not
    // move it: the copy's destination is LDS, which the compiler cannot tell apart
    // from the softmax scratch, so it guards the copy with a full vmcnt(0) at the top
    // of the body -- exactly where the wait already was. Naming a partial count there
    // instead does not stick either; the copy has no register result to order against,
    // so the scheduler sinks the wait below GEMM0 and re-inserts its own above it.
    // What is left is 8 more VGPRs for the addresses the copies hold live, and a
    // permuted fetch, because the copy's LDS side is a wave-uniform base plus an
    // implicit lane_id * V_DMA and so cannot carry the padding above.
    static constexpr bool V_VIA_LDS  = PAGE_HOLDS_TILE;
    static constexpr int V_DMA       = 16;
    static constexpr int V_LDS_ROW   = KV_TILE + VEC;
    static constexpr int V_LDS_DIMS  = GEMM1_E_N * GRP_N;
    static constexpr int V_STAGE_ITERS =
        V_VIA_LDS ? (V_LDS_DIMS * KV_TILE) / (WARP_SIZE * V_DMA) : 0;

    // Depth 2 keeps a second copy of the K and V staging registers. What that costs
    // depends on the path: staging V through LDS holds 8 VGPR on a 64-dim head, while
    // prefetching V straight into GEMM1's operand on a 192-dim head holds several
    // times that and spills (measured: 282 VGPR, 12 spilled, one wave per SIMD).
    // D128/page128 stays at depth 1: depth 2 took 118 to 204 VGPR without a
    // measured speedup. Fused MTP page-16 A16W8 is the same cliff: depth 2
    // stretched the B200 Q4 200k body from ~2.47ms to 4.63ms.
    static constexpr int PREFETCH_DEPTH =
        (PA_DECODE_OPUS_PREFETCH_DEPTH == 2 && V_VIA_LDS && D_HEAD <= 64) ? 2 : 1;

    static constexpr size_t smem_v_bytes()
    {
        return V_VIA_LDS ? static_cast<size_t>(T_N) * V_LDS_DIMS * V_LDS_ROW * sizeof(D_ATTN)
                         : size_t{0};
    }

    static_assert(Q_TILE == W_M, "one MFMA tile along M");
    static_assert(KV_TILE % (W_N * T_N) == 0);
    // A tile is a whole number of pages, or a page is a whole number of tiles.
    static_assert(PAGE_HOLDS_TILE ? (PAGE_SIZE % KV_TILE == 0) : (KV_TILE % PAGE_SIZE == 0));
    static_assert(D_HEAD % (W_N * T_N) == 0);
    static_assert(KV_TILE % W_K == 0);
    static_assert(D_HEAD % W_K == 0);
    // A query row lands on lanes {r, r+16, r+32, r+48} of the GEMM0 C fragment,
    // exactly the reach of a permlane32/permlane16 pair -- what lets S stay in
    // registers.
    static_assert(W_M == 16, "a C fragment row must span four 16-lane groups");
    static_assert(WARP_SIZE == 64);

    // Contract that lets the KV fragments be read straight from the cache: one K MFMA
    // tile must cover exactly one page, and a lane's slice of the contraction dim must
    // be a whole number of loads that nests cleanly inside the cache's packing.
    static_assert(PAGE_HOLDS_TILE ? (PAGE_SIZE % KV_TILE == 0)
                                  : (GRP_N == PAGE_SIZE),
                  "a page must either hold whole tiles, or line up with one K MFMA tile");
    static_assert(GRP_K * VEC * REPT == W_K, "a lane's K slice must be REPT vectors");
    static_assert(!K16_PACK || (GRP_N == 16 && GRP_K == 4 && D_HEAD % K_PACK == 0),
                  "K16 pack layout assumes the 16x16x32 fp8 wave");
    static_assert(!V_VIA_LDS
                      || (KV_TILE % V_DMA == 0 && WARP_SIZE * V_DMA % KV_TILE == 0
                          && V_LDS_DIMS * KV_TILE % (WARP_SIZE * V_DMA) == 0),
                  "the V staging load must tile the wave's dims exactly");
    static_assert(!V_VIA_LDS || V_LDS_DIMS * T_N == D_HEAD,
                  "V staging must cover every GEMM1 output dimension");
    static_assert(K_PACK % VEC == 0 || VEC % K_PACK == 0,
                  "a load must sit inside one pack group, or span whole ones");
    static_assert(PAGE_SIZE % VEC == 0 || VEC % PAGE_SIZE == 0);
    static_assert(ELEM_B == VEC * REPT, "one B fragment is REPT global vectors");
    static_assert(PAGE_HOLDS_TILE || GEMM0_E_N * T_N == PAGES_PER_TILE,
                  "K's N tiles must cover the tile's pages");
    // Only constrains the read-straight-from-the-cache path; a staged shape walks the
    // pages by staging chunk instead.
    static_assert(V_VIA_LDS
                      || GEMM1_E_K * REPT * (GRP_K / KGRP_PER_PAGE) == PAGES_PER_TILE,
                  "V's K steps must walk exactly the tile's pages");

    // Reused by the split-KV merge -- the P tile and row-reduce scratch are dead by
    // then. Holds every split's (m, l) plus one reciprocal per query row; the
    // per-split scale overwrites m in place.
    static constexpr size_t smem_reduce_bytes()
    {
        return static_cast<size_t>(2 * MAX_SPLITS * Q_TILE + Q_TILE) * sizeof(D_ACC);
    }

    static constexpr size_t smem_size_bytes()
    {
        const size_t attn = smem_p_elems * sizeof(D_ATTN)             // P handed to GEMM1
                            + smem_s_elems * sizeof(S_STORE)         // fused S spill
                            + smem_row_reduce_elems * sizeof(D_ACC)  // cross-wave row fold
                            + smem_v_bytes();                        // V staged coalesced
        const size_t reduce = smem_reduce_bytes();
        return attn > reduce ? attn : reduce;
    }
};

template<class Traits>
struct pa_decode_smem_bt_traits : Traits
{
    static constexpr bool BT_SCALAR = true;
};

// Transposed V: [num_blocks, num_kv_heads, PAGE/K_PACK, D, K_PACK]. Direct fragment
// loads, no LDS staging. gpt-oss shuffled V is this layout with a fixed Q64/KV4
// page-128 shape; A16W8 uses the same loads for page 16 and 128.
template<class Traits>
struct pa_decode_v_trans_traits : Traits
{
    static constexpr bool V_SHUFFLED = true;
    static constexpr bool V_VIA_LDS = false;
    static constexpr int V_LDS_DIMS = 0;
    static constexpr int V_STAGE_ITERS = 0;

    static constexpr size_t smem_v_bytes() { return 0; }

    static constexpr size_t smem_size_bytes()
    {
        const size_t attn = Traits::smem_p_elems * sizeof(typename Traits::D_ATTN)
                            + Traits::smem_s_elems * sizeof(typename Traits::S_STORE)
                            + Traits::smem_row_reduce_elems * sizeof(typename Traits::D_ACC);
        const size_t reduce = Traits::smem_reduce_bytes();
        return attn > reduce ? attn : reduce;
    }
};

template<class Traits>
struct pa_decode_per_token_traits : Traits
{
    static constexpr bool PER_TOKEN_SCALE = true;
};

// Small-MTP query split: one token per CTA, decode body, grid.y = batch * qlen.
template<class Traits>
struct pa_decode_q_split_traits : Traits
{
    static constexpr bool HAS_MTP     = true;
    static constexpr bool MTP_Q_LOOP  = false;
    static constexpr bool MTP_Q_SPLIT = true;
    // W is the fused MTP_Q_LOOP instantiation; it sizes an S spill this path
    // never uses. Drop it so q-split keeps the decode LDS footprint.
    static constexpr bool S_VIA_LDS        = false;
    static constexpr int smem_s_kv_tiles   = 0;
    static constexpr int smem_s_elems      = 0;
};

template<class Traits>
struct pa_decode_shuffled_v_traits : pa_decode_v_trans_traits<Traits>
{
    static_assert(Traits::IS_FP8 && !Traits::QUANT_Q && Traits::HAS_SINK
                  && !Traits::HAS_MTP && Traits::D_HEAD == 128
                  && Traits::PAGE_SIZE == 128 && Traits::KV_TILE == 128);
    static constexpr bool BT_SCALAR = true;
};

using pa_decode_traits_d128 =
    pa_decode_traits<bf16_t, bf16_t, 128, false, PA_DECODE_KV_TILE, 16, 4>;
// Multi-token prediction over the same shape: q and out carry a token dim and a tile
// row becomes (token, head), so qlen * gqa_ratio must fit Q_TILE. A separate
// instantiation, so plain decode keeps the code -- and the single peeled tile -- it has.
using pa_decode_traits_d128_mtp =
    pa_decode_traits<bf16_t, bf16_t, 128, false, PA_DECODE_KV_TILE, 16, 4, true>;
using pa_decode_traits_d128_fp8 =
    pa_decode_traits<fp8_t, fp8_t, 128, false, PA_DECODE_KV_TILE, 16, 4>;
using pa_decode_traits_d128_a16w8 =
    pa_decode_traits<fp8_t, bf16_t, 128, false, PA_DECODE_KV_TILE, 16, 4>;
// A16W8 without a sink: the same 128-dim fp8 KV as above, plus page 128 and MTP.
// qlen * gqa still has to fit Q_TILE. Persistent decode reuses the non-MTP pair.
using pa_decode_traits_d128_a16w8_mtp =
    pa_decode_traits<fp8_t, bf16_t, 128, false, PA_DECODE_KV_TILE, 16, 4, true>;
using pa_decode_traits_d128_a16w8_p128 =
    pa_decode_traits<fp8_t, bf16_t, 128, false, PA_DECODE_KV_TILE, 128, 4>;
using pa_decode_traits_d128_a16w8_p128_mtp =
    pa_decode_traits<fp8_t, bf16_t, 128, false, PA_DECODE_KV_TILE, 128, 4, true>;
// GQA 16, qlen 2..4: one query token per M tile, looped over a shared KV walk.
using pa_decode_traits_d128_a16w8_mtp_loop =
    pa_decode_traits<fp8_t, bf16_t, 128, false, PA_DECODE_KV_TILE, 16, 4, true, true>;
using pa_decode_traits_d128_a16w8_p128_mtp_loop =
    pa_decode_traits<fp8_t, bf16_t, 128, false, PA_DECODE_KV_TILE, 128, 4, true, true>;
// gpt-oss: 64-dim heads, 64 query heads over 8 kv heads, fp8 Q and KV, and a learned
// per-head sink. The narrow head puts the f8f6f4 MFMA out of reach -- its contraction
// is wider than the whole head -- so these fall back to the K=32 fp8 MFMA.
using pa_decode_traits_d64_fp8_sink =
    pa_decode_traits<fp8_t, fp8_t, 64, true, PA_DECODE_KV_TILE, 16, 4>;
using pa_decode_traits_d64_a16w8_sink =
    pa_decode_traits<fp8_t, bf16_t, 64, true, PA_DECODE_KV_TILE, 16, 4>;
using pa_decode_traits_d64_fp8_sink_p128 =
    pa_decode_traits<fp8_t, fp8_t, 64, true, PA_DECODE_KV_TILE, 128, 4>;
using pa_decode_traits_d64_a16w8_sink_p128 =
    pa_decode_traits<fp8_t, bf16_t, 64, true, PA_DECODE_KV_TILE, 128, 4>;
// The shape the serving framework actually allocates: 256-token pages, so a KV tile
// sits inside one page and two tiles share it.
using pa_decode_traits_d64_fp8_sink_p256 =
    pa_decode_traits<fp8_t, fp8_t, 64, true, PA_DECODE_KV_TILE, 256, 4>;
using pa_decode_traits_d64_a16w8_sink_p256 =
    pa_decode_traits<fp8_t, bf16_t, 64, true, PA_DECODE_KV_TILE, 256, 4>;
// Same six shapes with a token dim on q/out. A tile row is (token, head), so
// qlen * gqa_ratio must fit Q_TILE -- gpt-oss GQA 8 therefore caps qlen at 2.
using pa_decode_traits_d64_fp8_sink_mtp =
    pa_decode_traits<fp8_t, fp8_t, 64, true, PA_DECODE_KV_TILE, 16, 4, true>;
using pa_decode_traits_d64_a16w8_sink_mtp =
    pa_decode_traits<fp8_t, bf16_t, 64, true, PA_DECODE_KV_TILE, 16, 4, true>;
using pa_decode_traits_d64_fp8_sink_p128_mtp =
    pa_decode_traits<fp8_t, fp8_t, 64, true, PA_DECODE_KV_TILE, 128, 4, true>;
using pa_decode_traits_d64_a16w8_sink_p128_mtp =
    pa_decode_traits<fp8_t, bf16_t, 64, true, PA_DECODE_KV_TILE, 128, 4, true>;
using pa_decode_traits_d64_fp8_sink_p256_mtp =
    pa_decode_traits<fp8_t, fp8_t, 64, true, PA_DECODE_KV_TILE, 256, 4, true>;
using pa_decode_traits_d64_a16w8_sink_p256_mtp =
    pa_decode_traits<fp8_t, bf16_t, 64, true, PA_DECODE_KV_TILE, 256, 4, true>;

using pa_decode_traits_d128_fp8_sink =
    pa_decode_traits<fp8_t, fp8_t, 128, true, PA_DECODE_KV_TILE, 16, 4>;
using pa_decode_traits_d128_a16w8_sink =
    pa_decode_traits<fp8_t, bf16_t, 128, true, PA_DECODE_KV_TILE, 16, 4>;
using pa_decode_traits_d128_fp8_sink_p128 =
    pa_decode_traits<fp8_t, fp8_t, 128, true, PA_DECODE_KV_TILE, 128, 4>;
using pa_decode_traits_d128_a16w8_sink_p128 =
    pa_decode_traits<fp8_t, bf16_t, 128, true, PA_DECODE_KV_TILE, 128, 4>;
using pa_decode_traits_d128_fp8_sink_p256 =
    pa_decode_traits<fp8_t, fp8_t, 128, true, PA_DECODE_KV_TILE, 256, 4>;
using pa_decode_traits_d128_a16w8_sink_p256 =
    pa_decode_traits<fp8_t, bf16_t, 128, true, PA_DECODE_KV_TILE, 256, 4>;
using pa_decode_traits_d128_fp8_sink_mtp =
    pa_decode_traits<fp8_t, fp8_t, 128, true, PA_DECODE_KV_TILE, 16, 4, true>;
using pa_decode_traits_d128_a16w8_sink_mtp =
    pa_decode_traits<fp8_t, bf16_t, 128, true, PA_DECODE_KV_TILE, 16, 4, true>;
using pa_decode_traits_d128_fp8_sink_p128_mtp =
    pa_decode_traits<fp8_t, fp8_t, 128, true, PA_DECODE_KV_TILE, 128, 4, true>;
using pa_decode_traits_d128_a16w8_sink_p128_mtp =
    pa_decode_traits<fp8_t, bf16_t, 128, true, PA_DECODE_KV_TILE, 128, 4, true>;
using pa_decode_traits_d128_fp8_sink_p256_mtp =
    pa_decode_traits<fp8_t, fp8_t, 128, true, PA_DECODE_KV_TILE, 256, 4, true>;
using pa_decode_traits_d128_a16w8_sink_p256_mtp =
    pa_decode_traits<fp8_t, bf16_t, 128, true, PA_DECODE_KV_TILE, 256, 4, true>;

template<class Traits>
__global__ void pa_decode_opus_kernel(pa_decode_kargs kargs);
template<class Traits>
__global__ void pa_decode_opus_split_reduce_kernel(pa_decode_kargs kargs);
template<class Traits>
__global__ void pa_decode_opus_ps_kernel(pa_decode_kargs kargs);

#ifdef __HIP_DEVICE_COMPILE__
// ---------------------------------------------------------------------------
// Device pass
// ---------------------------------------------------------------------------
#include "opus/opus.hpp"

#if defined(__gfx950__)

namespace pa_decode_16mx1_16nx4 {

using opus::operator""_I;

// Scratch addressing for the split-KV path: one (Q_TILE x D) accumulator plus a
// (m, l) pair per query row, for every (batch, kv-head, split).
template<class T>
__device__ inline int64_t partial_slot(const pa_decode_kargs& kargs, int b, int kvh, int split)
{
    return (static_cast<int64_t>(b) * kargs.num_kv_heads + kvh) * kargs.num_splits + split;
}

template<class T>
__device__ inline int64_t mtp_token_slot_stride(const pa_decode_kargs& kargs)
{
    return static_cast<int64_t>(kargs.batch) * kargs.num_kv_heads * kargs.num_splits;
}

template<class T>
__device__ inline float* partial_ml_ptr(const pa_decode_kargs& kargs, int b, int kvh, int split,
                                        int tok = 0)
{
    const int64_t slot = partial_slot<T>(kargs, b, kvh, split)
                         + static_cast<int64_t>(tok) * mtp_token_slot_stride<T>(kargs);
    return kargs.partial_ml + slot * 2 * T::Q_TILE;
}

template<class T>
__device__ inline float* partial_o_ptr(const pa_decode_kargs& kargs, int b, int kvh, int split,
                                       int tok = 0)
{
    const int64_t slot = partial_slot<T>(kargs, b, kvh, split)
                         + static_cast<int64_t>(tok) * mtp_token_slot_stride<T>(kargs);
    return kargs.partial_o + slot * T::Q_TILE * T::D_HEAD;
}

// Where this lane's K and V fragments sit inside a page; only the page base moves
// per tile, so this is computed once. Both caches already store VEC contraction
// values in 16 contiguous bytes (K is [D/K_PACK][PAGE][K_PACK], a lane walks one
// token's head dims; V is [D][PAGE], a lane walks VEC tokens of one head dim), so
// every fragment read is a dwordx4 and the KV tile never goes through LDS.
//
// K's page index is wave-uniform. V's is not: a lane's token slice can land on any
// of the GRP_K/KGRP_PER_PAGE pages a contraction step spans -- 2 for bf16, 4 for
// fp8, whose wider K makes each step cover more tokens.
template<class T>
struct kv_frag_slice
{
    int k_off;      // element offset of the K fragment inside its page
    int v_off;      // element offset of the V fragment inside its page
    int v_page_grp; // which page of the step's span this lane reads V from

    __device__ kv_frag_slice(int lane_id, int warp_id)
    {
        const int n_lane = lane_id % T::GRP_N; // token for K, head dim for V
        const int k_grp  = lane_id / T::GRP_N; // slice of the contraction dim

        // K is [D/K_PACK][PAGE][K_PACK]. A lane's slice starts at head dim
        // k_grp * VEC, which sits in pack group k_grp * VEC / K_PACK at offset
        // k_grp * VEC % K_PACK inside it. The remainder is zero whenever a load is as
        // wide as the packing, which is every shape but fp8 at K=32.
        // When a page holds the whole tile, the N tiles are offsets inside it rather
        // than separate pages, so the wave's own N offset joins the token index here.
        const int k_tok = T::PAGE_HOLDS_TILE ? warp_id * T::GRP_N + n_lane : n_lane;
        k_off = (k_grp * T::VEC / T::K_PACK) * T::PAGE_SIZE * T::K_PACK + k_tok * T::K_PACK
                + (k_grp * T::VEC) % T::K_PACK;

        v_off = (warp_id * T::GRP_N + n_lane) * T::PAGE_SIZE
                + (k_grp % T::KGRP_PER_PAGE) * T::VEC;
        v_page_grp = k_grp / T::KGRP_PER_PAGE;
    }
};

// Within one launch a KV byte is read once and never looked at again: nothing revisits
// a tile, and a page belongs to one (sequence, kv head), so no other workgroup wants it
// either. Marking that stream non-temporal keeps a tile the machine has already
// consumed from holding cache lines a tile still in flight could use, which is worth
// ~11% once the context is long enough that the loop is purely streaming.
//
// What it gives up is retention across launches: the next decode step re-reads the same
// KV cache, so when the working set fits the LLC the eviction hint is a straight loss.
// The two streams sit differently on that trade, which is why the switch is per stream
// rather than one flag. K is the one to stream: doing so is worth 10-16% on a context
// long enough to have no reuse and costs nothing measurable when there is. Streaming V
// buys about the same on the streaming end but gives back 20% on a shape whose pages
// are shared between sequences, so V is left cacheable. Gluon likewise carries `nt` on
// half its KV loads.
template<bool NT, class V>
__device__ inline V load_stream(const V* p)
{
    if constexpr(NT) return __builtin_nontemporal_load(p);
    else return *p;
}

// Page indices for one KV tile, kept in registers an iteration ahead of the loads.
template<class T>
struct page_ids_t
{
    int id[T::PAGES_PER_TILE];
};

// Read one tile's worth of block-table entries. The buffer's num_records stops at
// this split's last page, so a slot past the end reads 0 with no bounds check in
// the instruction stream; page 0 is a valid index, but the tail tile masks the
// scores of the tokens it stands in for. Two dwordx4 cover the whole tile.
template<class T, bool SCALAR = T::BT_SCALAR, class G>
__device__ inline page_ids_t<T>
load_page_ids(G& g_bt, const int* __restrict__ bt, int num_pages, int tile_idx)
{
    page_ids_t<T> pid;
    if constexpr(T::PAGE_HOLDS_TILE)
    {
        // One page covers the tile, and TILES_PER_PAGE consecutive tiles share it.
        const int page = tile_idx / T::TILES_PER_PAGE;
        if constexpr(SCALAR)
        {
            // Constant address space yields SMEM for this wave-uniform, read-only
            // table. Clamp prefetches past the valid range to its last page; no
            // valid token consumes those entries. Empty runs exit before this call.
            const int p = __builtin_amdgcn_readfirstlane(page);
            using scalar_table_ptr = const int __attribute__((address_space(4)))*;
            const scalar_table_ptr scalar_bt = (scalar_table_ptr)bt;
            pid.id[0] = scalar_bt[p < num_pages ? p : num_pages - 1];
        }
        else
        {
            pid.id[0] = g_bt.template _load<1>(page * static_cast<int>(sizeof(int)))[0];
        }
    }
    else
    {
        static_assert(T::PAGES_PER_TILE % 4 == 0, "block-table reads are dwordx4");
        const int base = tile_idx * T::PAGES_PER_TILE * static_cast<int>(sizeof(int));
        opus::static_for<T::PAGES_PER_TILE / 4>([&](auto ig) {
            const auto v = g_bt.template _load<4>(base + ig.value * 16);
            opus::static_for<4>([&](auto j) { pid.id[ig.value * 4 + j.value] = v[j.value]; });
        });
    }
    return pid;
}

// Page indices for the K fragments of one wave. A K MFMA tile is exactly one page,
// so wave w only ever reads pages {i_n * T_N + w}.
template<class T>
struct k_page_ids_t
{
    int id[T::GEMM0_E_N];
};

// Read one tile's K fragments straight from the cache into the matrix-core operand.
// Deliberately does not wait on the results: the caller issues these right after the
// GEMM that consumed the previous tile, so the latency hides in the loop body.
// Pointer-addressed, not buffer: a buffer offset is 32-bit and would cap the cache
// at 4GiB, and V's per-lane page index leaves no scalar base to lift it.
//
// A lane's slice of one MFMA's contraction dim is REPT vectors of VEC elements, at
// K positions r * GRP_K * VEC + k_grp * VEC, and sits in the fragment at r * VEC --
// the operand layout puts rept outside the lane dim and pack inside it. REPT is 1
// for bf16, so that path is a single vector per (i_n, i_k) as before.
template<class T, int BEGIN = 0, int COUNT = T::GEMM0_E_N, class VB>
__device__ inline void load_k_frags(const typename T::D_ATTN* __restrict__ p_k,
                                    const k_page_ids_t<T>& pid,
                                    int stride_k_blk,
                                    const kv_frag_slice<T>& s,
                                    int tok_base,
                                    VB& v_k)
{
    using D_ATTN      = typename T::D_ATTN;
    using vec_t       = opus::vector_t<D_ATTN, T::VEC>;
    constexpr int E_N = T::GEMM0_E_N;
    constexpr int E_K = T::GEMM0_E_K;

    static_assert(BEGIN >= 0 && COUNT > 0 && BEGIN + COUNT <= E_N);
    opus::static_for<COUNT>([&](auto in) {
        constexpr int i_n = BEGIN + in.value;
        // In the wide-page regime the N tile is an offset, not a different page, and
        // tok_base says which tile of the page this is. Both are zero otherwise.
        constexpr int n_off =
            T::PAGE_HOLDS_TILE ? i_n * T::T_N * T::GRP_N * T::K_PACK : 0;
        const D_ATTN* g_n = p_k + static_cast<int64_t>(pid.id[i_n]) * stride_k_blk + s.k_off
                            + n_off + tok_base * T::K_PACK;

        opus::static_ford<E_K, T::REPT>([&](auto ik, auto ir) {
            constexpr int i_k = ik.value, i_r = ir.value;
            // Same split as k_off, for the compile-time part of the head dim.
            constexpr int base_d = (i_k * T::REPT + i_r) * T::GRP_K * T::VEC;
            constexpr int d_off =
                (base_d / T::K_PACK) * T::PAGE_SIZE * T::K_PACK + base_d % T::K_PACK;
            const vec_t frag =
                load_stream<PA_DECODE_OPUS_NT_K>(reinterpret_cast<const vec_t*>(g_n + d_off));
            opus::static_for<T::VEC>([&](auto j) {
                v_k[(i_n * E_K + i_k) * T::ELEM_B + i_r * T::VEC + j.value] = frag[j.value];
            });
        });
    });
}

// 16B K-cache packs, Gluon-style. Even k_grp (lanes 0-15 and 32-47) issues a unique
// dwordx4 covering D[pack*16, pack*16+16) for one token; that is two consecutive 8B
// MFMA fragments. v_permlane16_swap then moves the high half to the odd k_grp.
// Four packs x 16 tokens is one N-tile of D=64, so E_N * E_K loads on the even
// lanes cover the wave's whole GEMM0 K operand.
template<class T>
struct k_stage_t
{
    using chunk_t = opus::vector_t<typename T::D_ATTN, 16>;
    chunk_t c[T::K_PACK_ITERS > 0 ? T::K_PACK_ITERS : 1];
};

template<class T>
__device__ inline void load_k_pack(const typename T::D_ATTN* __restrict__ p_k,
                                   const k_page_ids_t<T>& pid,
                                   int stride_k_blk,
                                   int tok_base,
                                   int lane_id,
                                   int warp_id,
                                   k_stage_t<T>& st)
{
    using chunk_t    = typename k_stage_t<T>::chunk_t;
    const int k_grp  = lane_id / T::GRP_N;
    const int n_lane = lane_id % T::GRP_N;
    const int pair   = k_grp / 2;

    if constexpr(T::K16_WIDE)
    {
        // One pack per lane group, so the wave covers all GRP_K packs of a token with
        // every lane active: half the loads of the narrow form below at twice the bytes
        // each. That is the only form that widens anything -- a masked dwordx4 moves
        // exactly as much as the unmasked 8B load it replaced.
        opus::static_for<T::GEMM0_E_N>([&](auto in) {
            constexpr int i_n = in.value;
            const int tok     = T::PAGE_HOLDS_TILE
                                    ? tok_base + i_n * T::T_N * T::GRP_N
                                          + warp_id * T::GRP_N + n_lane
                                    : n_lane;
            const typename T::D_ATTN* src =
                p_k + static_cast<int64_t>(pid.id[i_n]) * stride_k_blk
                + k_grp * T::PAGE_SIZE * T::K_PACK + tok * T::K_PACK;
            st.c[i_n] =
                load_stream<PA_DECODE_OPUS_NT_K>(reinterpret_cast<const chunk_t*>(src));
        });
    }
    // Odd k_grp receives the high 8B via permute; loading the same pack would
    // duplicate the 16B (measured +13%). (lane_id & 16) == 0 is k_grp 0 and 2.
    else if((lane_id & 16) == 0)
    {
        opus::static_ford<T::GEMM0_E_N, T::GEMM0_E_K>([&](auto in, auto ik) {
            constexpr int i_n = in.value, i_k = ik.value;
            const int pack    = i_k * 2 + pair;
            const int tok     = T::PAGE_HOLDS_TILE
                                ? tok_base + i_n * T::T_N * T::GRP_N + warp_id * T::GRP_N
                                      + n_lane
                                : n_lane;
            const typename T::D_ATTN* src =
                p_k + static_cast<int64_t>(pid.id[i_n]) * stride_k_blk
                + pack * T::PAGE_SIZE * T::K_PACK + tok * T::K_PACK;
            st.c[i_n * T::GEMM0_E_K + i_k] =
                load_stream<PA_DECODE_OPUS_NT_K>(reinterpret_cast<const chunk_t*>(src));
        });
    }
}

// Runtime select over page slots that are all compile-time indices, so pid stays in
// registers instead of becoming an s_cselect chain or spilling to scratch.
// Slot 0 is the default and the rest cascade onto it, so the outermost select stays
// a plain truthiness test on sel. That matters for the two-page case: `sel ? hi : lo`
// lets the compiler share its compare with the lane predicate the addressing
// already needed, which an equality or ordered test does not.
template<int BASE, int N, class T>
__device__ inline int pick_page(const page_ids_t<T>& pid, int sel)
{
    if constexpr(N == 1) return pid.id[BASE];
    else
    {
        int hi = pid.id[BASE + N - 1];
        opus::static_for<N - 2>([&](auto i) {
            constexpr int k = N - 2 - i.value;
            hi              = sel == k ? pid.id[BASE + k] : hi;
        });
        return sel ? hi : pid.id[BASE];
    }
}

// The pages K wants are a subset of the ones load_page_ids has already put in
// registers -- wave w's N tile i_n is page i_n * T_N + w, and in the wide regime
// every N tile is the tile's one page -- so they are selected here rather than read
// from the block table a second time. Re-reading them was two more loads per tile on
// a narrow page, and those land on the same queue as the KV stream: what shows up in
// a trace is not their own latency but the KV loads queued behind them.
template<class T>
__device__ inline k_page_ids_t<T> k_pages_from(const page_ids_t<T>& pid, int warp_id)
{
    k_page_ids_t<T> k;
    opus::static_for<T::GEMM0_E_N>([&](auto in) {
        if constexpr(T::PAGE_HOLDS_TILE)
            k.id[in.value] = pid.id[0];
        else
            k.id[in.value] = pick_page<in.value * T::T_N, T::T_N>(pid, warp_id);
    });
    return k;
}

// Same for V. Here the contraction dim is tokens, so a K step walks pages while
// N walks head dims. One (i_k, i_r) step of GRP_K lane groups spans
// GRP_K/KGRP_PER_PAGE pages, and v_page_grp says which of them this lane reads.
template<class T, class VB>
__device__ inline void load_v_frags(const typename T::D_ATTN* __restrict__ p_v,
                                    const page_ids_t<T>& pid,
                                    int stride_v_blk,
                                    const kv_frag_slice<T>& s,
                                    int tok_base,
                                    VB& v_v)
{
    using D_ATTN                = typename T::D_ATTN;
    using vec_t                 = opus::vector_t<D_ATTN, T::VEC>;
    constexpr int E_N           = T::GEMM1_E_N;
    constexpr int E_K           = T::GEMM1_E_K;
    constexpr int PAGES_PER_STEP = T::GRP_K / T::KGRP_PER_PAGE;

    opus::static_ford<E_K, T::REPT>([&](auto ik, auto ir) {
        constexpr int i_k = ik.value, i_r = ir.value;
        // One page in the wide regime, so the K step becomes a token offset instead of
        // a page index; tok_base places the tile inside its page.
        constexpr int page_base =
            T::PAGE_HOLDS_TILE ? 0 : (i_k * T::REPT + i_r) * PAGES_PER_STEP;
        constexpr int k_off =
            T::PAGE_HOLDS_TILE ? (i_k * T::REPT + i_r) * T::GRP_K * T::VEC : 0;
        const int page    = pick_page<page_base, PAGES_PER_STEP>(pid, s.v_page_grp);
        const D_ATTN* g_k = p_v + static_cast<int64_t>(page) * stride_v_blk + s.v_off
                            + k_off + tok_base;

        opus::static_for<E_N>([&](auto in) {
            constexpr int i_n   = in.value;
            constexpr int d_off = i_n * T::T_N * T::GRP_N * T::PAGE_SIZE;
            const vec_t frag =
                load_stream<PA_DECODE_OPUS_NT_V>(reinterpret_cast<const vec_t*>(g_k + d_off));
            opus::static_for<T::VEC>([&](auto j) {
                v_v[(i_n * E_K + i_k) * T::ELEM_B + i_r * T::VEC + j.value] = frag[j.value];
            });
        });
    });
}

template<class T, int BEGIN = 0, int COUNT = T::GEMM1_E_N, class VB>
__device__ inline void load_v_shuffled_frags(const typename T::D_ATTN* __restrict__ p_v,
                                           const page_ids_t<T>& pid,
                                           int stride_v_blk,
                                           int tok_base,
                                           int lane_id,
                                           int warp_id,
                                           VB& v_v)
{
    using vec_t = opus::vector_t<typename T::D_ATTN, T::VEC>;
    static_assert(BEGIN >= 0 && COUNT > 0 && BEGIN + COUNT <= T::GEMM1_E_N);
    opus::static_ford<T::GEMM1_E_K, T::REPT>([&](auto ik, auto ir) {
        constexpr int i_k = ik.value, i_r = ir.value;
        const int token_in_tile = (i_k * T::REPT + i_r) * T::GRP_K * T::VEC
                                  + lane_id / T::GRP_N * T::VEC;
        opus::static_for<COUNT>([&](auto in) {
            constexpr int i_n = BEGIN + in.value;
            const int head_dim = (i_n * T::T_N + warp_id) * T::GRP_N
                                  + lane_id % T::GRP_N;
            int page, tok;
            if constexpr(T::PAGE_HOLDS_TILE)
            {
                page = pid.id[0];
                tok  = tok_base + token_in_tile;
            }
            else
            {
                page = pick_page<0, T::PAGES_PER_TILE>(pid, token_in_tile / T::PAGE_SIZE);
                tok  = token_in_tile % T::PAGE_SIZE;
            }
            const int offset = (tok / T::K_PACK * T::D_HEAD + head_dim) * T::K_PACK
                                + tok % T::K_PACK;
            const vec_t frag = load_stream<PA_DECODE_OPUS_NT_V>(
                reinterpret_cast<const vec_t*>(p_v + static_cast<int64_t>(page) * stride_v_blk
                                               + offset));
            opus::static_for<T::VEC>([&](auto item) {
                v_v[(i_n * T::GEMM1_E_K + i_k) * T::ELEM_B
                    + i_r * T::VEC + item.value] = frag[item.value];
            });
        });
    });
}

// ---- V through LDS, for pages that hold a whole tile ---------------------------
//
// Three steps, split so the global read keeps a full tile of prefetch distance:
//   load_v_stage   global -> registers, coalesced (issued at the end of a tile)
//   stage_v_to_lds registers -> LDS     (top of the next tile)
//   read_v_frags   LDS -> fragments     (after the P barrier, just before GEMM1)
//
// The staging mapping gives 8 lanes one head dim's 128 tokens, so each lane's V_DMA
// bytes are contiguous and a lane group covers exactly one cache line.

template<class T>
struct v_stage_t
{
    // V_DMA bytes per lane per iteration, held as raw bytes: this is a copy, and
    // nothing in between interprets the values.
    using chunk_t = opus::vector_t<typename T::D_ATTN, T::V_DMA>;
    chunk_t c[T::V_STAGE_ITERS > 0 ? T::V_STAGE_ITERS : 1];
};

// Where this lane's staged chunk sits, as (head dim within the wave, token).
template<class T>
__device__ inline void v_stage_coords(int lane_id, int it, int& dim, int& tok)
{
    constexpr int LANES_PER_DIM = T::KV_TILE / T::V_DMA;
    constexpr int DIMS_PER_ITER = T::WARP_SIZE / LANES_PER_DIM;
    dim = it * DIMS_PER_ITER + lane_id / LANES_PER_DIM;
    tok = (lane_id % LANES_PER_DIM) * T::V_DMA;
}

template<class T, int BEGIN = 0, int COUNT = T::V_STAGE_ITERS>
__device__ inline void load_v_stage(const typename T::D_ATTN* __restrict__ p_v,
                                    const page_ids_t<T>& pid,
                                    int stride_v_blk,
                                    int tok_base,
                                    int lane_id,
                                    int warp_id,
                                    v_stage_t<T>& st)
{
    using D_ATTN  = typename T::D_ATTN;
    using chunk_t = typename v_stage_t<T>::chunk_t;

    // One page holds the tile, so there is a single page id and no page walk.
    const D_ATTN* g = p_v + static_cast<int64_t>(pid.id[0]) * stride_v_blk + tok_base;

    static_assert(BEGIN >= 0 && COUNT > 0 && BEGIN + COUNT <= T::V_STAGE_ITERS);
    opus::static_for<COUNT>([&](auto chunk) {
        constexpr int index = BEGIN + chunk.value;
        int dim, tok;
        v_stage_coords<T>(lane_id, index, dim, tok);
        const int head_dim = (dim / T::GRP_N * T::T_N + warp_id) * T::GRP_N
                     + dim % T::GRP_N;
        const D_ATTN* src = g + head_dim * T::PAGE_SIZE + tok;
        st.c[index] = load_stream<PA_DECODE_OPUS_NT_V>(reinterpret_cast<const chunk_t*>(src));
    });
}

template<class T, int BEGIN = 0, int COUNT = T::V_STAGE_ITERS>
__device__ inline void
stage_v_to_lds(typename T::D_ATTN* s_v, int lane_id, const v_stage_t<T>& st)
{
    using chunk_t = typename v_stage_t<T>::chunk_t;
    static_assert(BEGIN >= 0 && COUNT > 0 && BEGIN + COUNT <= T::V_STAGE_ITERS);
    opus::static_for<COUNT>([&](auto chunk) {
        constexpr int index = BEGIN + chunk.value;
        int dim, tok;
        v_stage_coords<T>(lane_id, index, dim, tok);
        *reinterpret_cast<chunk_t*>(s_v + dim * T::V_LDS_ROW + tok) = st.c[index];
    });
}

// The fragment layout GEMM1 wants: lane (n_lane, k_grp) holds head dim n_lane and
// VEC contiguous tokens, the same slice the direct-from-global path built.
template<class T, class VB>
__device__ inline void
read_v_frags(const typename T::D_ATTN* s_v, int lane_id, VB& v_v)
{
    using D_ATTN      = typename T::D_ATTN;
    using vec_t       = opus::vector_t<D_ATTN, T::VEC>;
    constexpr int E_N = T::GEMM1_E_N;
    constexpr int E_K = T::GEMM1_E_K;

    const int n_lane = lane_id % T::GRP_N;
    const int k_grp  = lane_id / T::GRP_N;

    opus::static_ford<E_K, T::REPT>([&](auto ik, auto ir) {
        constexpr int i_k = ik.value, i_r = ir.value;
        constexpr int t_off = (i_k * T::REPT + i_r) * T::GRP_K * T::VEC;
        const D_ATTN* src   = s_v + n_lane * T::V_LDS_ROW + t_off + k_grp * T::VEC;

        opus::static_for<E_N>([&](auto in) {
            constexpr int i_n = in.value;
            const vec_t frag = *reinterpret_cast<const vec_t*>(
                src + i_n * T::GRP_N * T::V_LDS_ROW);
            opus::static_for<T::VEC>([&](auto j) {
                v_v[(i_n * E_K + i_k) * T::ELEM_B + i_r * T::VEC + j.value] = frag[j.value];
            });
        });
    });
}

template<class T, int GROUP, class VB>
__device__ inline void
read_v_frags_group(const typename T::D_ATTN* s_v, int lane_id, VB& fragment)
{
    using vec_t = opus::vector_t<typename T::D_ATTN, T::VEC>;
    static_assert(GROUP >= 0 && GROUP < T::GEMM1_E_N && T::GEMM1_E_K == 1);
    const int dim = GROUP * T::GRP_N + lane_id % T::GRP_N;
    const int token_group = lane_id / T::GRP_N;
    opus::static_for<T::REPT>([&](auto repeat) {
        const auto* src = s_v + dim * T::V_LDS_ROW
                         + (repeat.value * T::GRP_K + token_group) * T::VEC;
        const vec_t values = *reinterpret_cast<const vec_t*>(src);
        opus::static_for<T::VEC>([&](auto element) {
            fragment[repeat.value * T::VEC + element.value] = values[element.value];
        });
    });
}

// All-reduce a query row inside one wave: a swap at distance 32 then one at 16,
// which reaches the whole row since it sits on lanes {r, r+16, r+32, r+48}. A lane
// covers only a quarter of a row, so every row-wide quantity passes through here.
//
// Must be inline asm, not __builtin_amdgcn_permlane*_swap: the intrinsic returns
// both halves as one vector, and when both operands hold the same value -- how an
// all-reduce uses it -- the compiler folds them into one register, collapsing the
// fold to op(v, v). Invisible for max, wrong for a sum.
//
// Inline asm also loses the hazard recognizer. Per LLVM's GCNHazardRecognizer for
// gfx950, a VALU write needs 2 wait states before the swap reads it and a v_cmpx
// writing exec needs 4; s_nop 3 covers both.
#define PA_SWAP_HAZARD "s_nop 3\n\t"

__device__ inline void permlane32_swap(float& a, float& b)
{
    asm volatile(PA_SWAP_HAZARD "v_permlane32_swap_b32 %0, %1" : "+v"(a), "+v"(b));
}

__device__ inline void permlane16_swap(float& a, float& b)
{
    asm volatile(PA_SWAP_HAZARD "v_permlane16_swap_b32 %0, %1" : "+v"(a), "+v"(b));
}

__device__ inline void permlane16_swap_b32(unsigned& a, unsigned& b)
{
    asm volatile(PA_SWAP_HAZARD "v_permlane16_swap_b32 %0, %1" : "+v"(a), "+v"(b));
}

__device__ inline void permlane32_swap_b32(unsigned& a, unsigned& b)
{
    asm volatile(PA_SWAP_HAZARD "v_permlane32_swap_b32 %0, %1" : "+v"(a), "+v"(b));
}

// Place one 8B B-fragment, carried as its two dwords, into the MFMA operand buffer.
template<class T, int SLOT, class VB>
__device__ inline void store_k_frag(VB& v_k, unsigned d0, unsigned d1)
{
    using vec_t = opus::vector_t<typename T::D_ATTN, T::VEC>;
    using u32x2 = opus::vector_t<unsigned, 2>;
    u32x2 pair;
    pair[0]          = d0;
    pair[1]          = d1;
    const vec_t frag = __builtin_bit_cast(vec_t, pair);
    opus::static_for<T::VEC>(
        [&](auto j) { v_k[SLOT * T::ELEM_B + j.value] = frag[j.value]; });
}

// Split the 16B cache packs into the 8B MFMA B fragments.
template<class T, class VB>
__device__ inline void convert_k_pack_frags(k_stage_t<T>& st, VB& v_k)
{
    using u32x4 = opus::vector_t<unsigned, 4>;

    if constexpr(T::K16_PERM)
    {
        // Lane group g holds pack g of its token: dwords (0,1) are head dims
        // [g*K_PACK, +VEC) and (2,3) the VEC after them. Q was walked in that same
        // order, so those halves already are the two MFMA fragments -- the split is
        // free and the permutes below are what mode 3 exists to avoid.
        opus::static_for<T::GEMM0_E_N>([&](auto in) {
            constexpr int i_n = in.value;
            u32x4 d           = __builtin_bit_cast(u32x4, st.c[i_n]);
            store_k_frag<T, i_n * T::GEMM0_E_K + 0>(v_k, d[0], d[1]);
            store_k_frag<T, i_n * T::GEMM0_E_K + 1>(v_k, d[2], d[3]);
        });
    }
    else if constexpr(T::K16_WIDE)
    {
        // Lane group g holds pack g of its token, so its dwords (0,1) are head dims
        // [16g, 16g+8) and (2,3) are [16g+8, 16g+16). The MFMA instead wants group g
        // to hold dims [32*i_k + 8g, +8). Writing one dword position as X over the
        // four groups (the pack's low half) and Y (its high half), the two swaps are:
        //
        //   start                   X = [X0 X1 X2 X3]   Y = [Y0 Y1 Y2 Y3]
        //   permlane16_swap(X, Y)   X = [X0 Y0 X2 Y2]   Y = [X1 Y1 X3 Y3]
        //   permlane32_swap(X, Y)   X = [X0 Y0 X1 Y1]   Y = [X2 Y2 X3 Y3]
        //
        // X now carries dims [0,8) [8,16) [16,24) [24,32) across g -- the i_k = 0
        // fragment -- and Y the same for [32,64), the i_k = 1 fragment. Two swaps per
        // dword position, so the same eight permutes the half-wave form needed, but
        // over half as many loads.
        opus::static_for<T::GEMM0_E_N>([&](auto in) {
            constexpr int i_n = in.value;
            u32x4 d           = __builtin_bit_cast(u32x4, st.c[i_n]);
            unsigned x0 = d[0], y0 = d[2];
            unsigned x1 = d[1], y1 = d[3];
            permlane16_swap_b32(x0, y0);
            permlane32_swap_b32(x0, y0);
            permlane16_swap_b32(x1, y1);
            permlane32_swap_b32(x1, y1);
            store_k_frag<T, i_n * T::GEMM0_E_K + 0>(v_k, x0, x1);
            store_k_frag<T, i_n * T::GEMM0_E_K + 1>(v_k, y0, y1);
        });
    }
    else
    {
        // permlane16_swap(lo_dword, hi_dword) exchanges G0.hi with G1.lo (and G2.hi
        // with G3.lo), so after both dwords the even k_grp keeps the low 8B and the
        // odd k_grp holds the high 8B in the low slots.
        opus::static_ford<T::GEMM0_E_N, T::GEMM0_E_K>([&](auto in, auto ik) {
            constexpr int i_n = in.value, i_k = ik.value;
            u32x4 d           = __builtin_bit_cast(u32x4, st.c[i_n * T::GEMM0_E_K + i_k]);
            unsigned lo0 = d[0], hi0 = d[2];
            unsigned lo1 = d[1], hi1 = d[3];
            permlane16_swap_b32(lo0, hi0);
            permlane16_swap_b32(lo1, hi1);
            store_k_frag<T, i_n * T::GEMM0_E_K + i_k>(v_k, lo0, lo1);
        });
    }
}

template<class OP>
__device__ inline float wave_row_fold(float v, OP op)
{
    float a = v, b = v;
    permlane32_swap(a, b);
    a = op(a, b);
    b = a;
    permlane16_swap(a, b);
    return op(a, b);
}

// Publish this wave's row value and fold the T_N of them through LDS -- T_N
// contiguous floats per row, one 16B read, and the only barrier softmax needs.
//
// s_row_reduce is deliberately not __restrict__ and the barrier is the fenced
// __syncthreads, not a bare s_barrier: both call sites hit the same addresses, and
// s_barrier only orders execution, so the compiler could carry the first fold's
// loaded values across it and reuse them for the second.
// N>1 is fused MTP: every query token writes a private [Q_TILE][T_N] slice so
// they share one barrier.
template<class T, int N, class OP>
__device__ inline void fold_across_waves_n(typename T::D_ACC* s_row_reduce,
                                          typename T::D_ACC v[N],
                                          int q_row,
                                          int warp_id,
                                          OP op)
{
    static_assert(N >= 1 && N <= T::smem_p_q_tiles);
    opus::static_for<N>([&](auto i) {
        s_row_reduce[(i.value * T::Q_TILE + q_row) * T::T_N + warp_id] = v[i.value];
    });
    __syncthreads();
    opus::static_for<N>([&](auto i) {
        typename T::D_ACC acc = s_row_reduce[(i.value * T::Q_TILE + q_row) * T::T_N];
        opus::static_for<T::T_N - 1>([&](auto iw) {
            acc = op(acc, s_row_reduce[(i.value * T::Q_TILE + q_row) * T::T_N + iw.value + 1]);
        });
        v[i.value] = acc;
    });
}

template<class T, class OP>
__device__ inline typename T::D_ACC
fold_across_waves(typename T::D_ACC* s_row_reduce,
                  typename T::D_ACC v,
                  int q_row,
                  int warp_id,
                  OP op)
{
    typename T::D_ACC tmp[1] = {v};
    fold_across_waves_n<T, 1>(s_row_reduce, tmp, q_row, warp_id, op);
    return tmp[0];
}

// Masked online softmax over the GEMM0 C fragment, in registers. Returns the
// rescale factor for the caller's O accumulator. Lane l of wave w owns query row
// l % W_M and, per N fragment, ELEM_C consecutive tokens.
//
// m_run must stay identical across waves, since the P they all write feeds one
// GEMM and needs one common scale. l_run needs no such agreement, so it stays an
// unfolded per-lane partial, reduced once after the last tile.
//
// MASKED covers the last tile, the only one that can run past the context -- and
// under MTP the one before it too, whose rows can already be past their own limit.
// `valid_kv` is run-relative, like kv_base below, and is per-lane under MTP because
// there each row stops at a different token.

// Scale and mask the C fragment, leave scaled logits in v_s, return this wave's
// row max. The cross-wave fold is separate so fused MTP can publish every token
// into LDS and share one barrier.
template<class T, bool MASKED, class VC>
__device__ inline typename T::D_ACC
softmax_scale_mask_wave_max(VC& v_s,
                            typename T::D_ACC scale,
                            int valid_kv,
                            int tile_idx,
                            int lane_id,
                            int warp_id)
{
    using D_ACC       = typename T::D_ACC;
    constexpr int E_N = T::GEMM0_E_N;
    constexpr int E_C = T::ELEM_C;
    const int tok_lane = (lane_id / T::W_M) * E_C;

    D_ACC local_max = opus::numeric_limits<D_ACC>::lowest();
    opus::static_for<E_N>([&](auto in) {
        constexpr int i_n = in.value;
        const int kv_base = tile_idx * T::KV_TILE + (i_n * T::T_N + warp_id) * T::W_N + tok_lane;
        opus::static_for<E_C>([&](auto ic) {
            constexpr int c = ic.value;
            D_ACC s         = v_s[i_n * E_C + c] * scale;
            if constexpr(MASKED)
                s = (kv_base + c) < valid_kv ? s : opus::numeric_limits<D_ACC>::lowest();
            v_s[i_n * E_C + c] = s;
            local_max          = s > local_max ? s : local_max;
        });
    });
    return wave_row_fold(local_max, [](D_ACC a, D_ACC b) { return a > b ? a : b; });
}

template<class T, class VC>
__device__ inline typename T::D_ACC
softmax_apply_tile_max(VC& v_s,
                       typename T::D_ACC tile_max,
                       typename T::D_ACC& m_run,
                       typename T::D_ACC& l_run)
{
    using D_ACC         = typename T::D_ACC;
    const D_ACC m_prev  = m_run;
    m_run               = tile_max > m_prev ? tile_max : m_prev;
    const D_ACC rescale = __builtin_amdgcn_exp2f(m_prev - m_run);

    // P_LOG2 lifts P into the top of D_ATTN's exponent range instead of leaving it in
    // [0, 1]; folding it into the max costs nothing and l picks up the same factor,
    // so it cancels in O/l. Zero for bf16.
    D_ACC m_exp = m_run;
    if constexpr(T::P_LOG2 != 0) m_exp -= D_ACC(T::P_LOG2);

    D_ACC local_sum = 0.0f;
    opus::static_for<T::GEMM0_E_N * T::ELEM_C>([&](auto i) {
        // exp2 of lowest() underflows to 0, which is exactly the mask we want.
        const D_ACC e = __builtin_amdgcn_exp2f(v_s[i.value] - m_exp);
        v_s[i.value]  = e;
        local_sum += e;
    });
    l_run = l_run * rescale + local_sum;
    return rescale;
}

template<class T, bool MASKED, class VC>
__device__ inline typename T::D_ACC
online_softmax_frag(VC& v_s,
                    typename T::D_ACC* __restrict__ s_row_reduce,
                    typename T::D_ACC& m_run,
                    typename T::D_ACC& l_run,
                    typename T::D_ACC scale,
                    int valid_kv,
                    int tile_idx,
                    int lane_id,
                    int warp_id)
{
    using D_ACC = typename T::D_ACC;
    const D_ACC wave_max = softmax_scale_mask_wave_max<T, MASKED>(
        v_s, scale, valid_kv, tile_idx, lane_id, warp_id);
    const D_ACC tile_max =
        fold_across_waves<T>(s_row_reduce,
                             wave_max,
                             lane_id % T::W_M,
                             warp_id,
                             [](D_ACC a, D_ACC b) { return a > b ? a : b; });
    return softmax_apply_tile_max<T>(v_s, tile_max, m_run, l_run);
}

// Per-token KV dequant for one C-fragment: four consecutive tokens per N tile, the
// same layout online_softmax_frag walks. Page ids are the *current* tile's, not the
// prefetch set, which DEPTH 1 has already overwritten with the next tile.
template<class T>
__device__ inline void load_kv_scale_frag(const float* __restrict__ map,
                                          int stride_blk,
                                          int stride_h,
                                          const page_ids_t<T>& pid,
                                          int kvh,
                                          int tok_base,
                                          int lane_id,
                                          int warp_id,
                                          float* dst)
{
    constexpr int E_N = T::GEMM0_E_N;
    constexpr int E_C = T::ELEM_C;
    const int tok_lane = (lane_id / T::W_M) * E_C;
    opus::static_for<E_N>([&](auto in) {
        constexpr int i_n         = in.value;
        const int tok_in_tile     = (i_n * T::T_N + warp_id) * T::W_N + tok_lane;
        int page, tok_in_page;
        if constexpr(T::PAGE_HOLDS_TILE)
        {
            page        = pid.id[0];
            tok_in_page = tok_base + tok_in_tile;
        }
        else
        {
            page        = pick_page<0, T::PAGES_PER_TILE>(pid, tok_in_tile / T::PAGE_SIZE);
            tok_in_page = tok_in_tile % T::PAGE_SIZE;
        }
        const float* p = map + static_cast<int64_t>(page) * stride_blk
                         + static_cast<int64_t>(kvh) * stride_h + tok_in_page;
        opus::static_for<E_C>([&](auto ic) { dst[i_n * E_C + ic.value] = p[ic.value]; });
    });
}

template<class T, class VC>
__device__ inline void mul_score_scales(VC& v_s, const float* sc)
{
    opus::static_for<T::GEMM0_E_N * T::ELEM_C>([&](auto i) { v_s[i.value] *= sc[i.value]; });
}

// Spill one C fragment to LDS so fused MTP can keep a single live score tile.
template<class T, class VC>
__device__ inline void spill_score_lds(typename T::S_STORE* s,
                                       const VC& v_s,
                                       int tok,
                                       int half,
                                       int tid)
{
    auto* dst = s + ((tok * T::smem_s_kv_tiles + half) * T::BLOCK_SIZE + tid) * T::S_FRAG;
    opus::static_for<T::S_FRAG>([&](auto i) {
        dst[i.value] = static_cast<typename T::S_STORE>(v_s[i.value]);
    });
}

template<class T, class VC>
__device__ inline void fill_score_lds(VC& v_s,
                                      const typename T::S_STORE* s,
                                      int tok,
                                      int half,
                                      int tid)
{
    const auto* src = s + ((tok * T::smem_s_kv_tiles + half) * T::BLOCK_SIZE + tid) * T::S_FRAG;
    opus::static_for<T::S_FRAG>([&](auto i) {
        v_s[i.value] = static_cast<float>(src[i.value]);
    });
}

// Announce that this split is done, and report whether it was the last one. Nothing
// ever waits on the counter -- it only tells that last arrival, which already has
// the machine, that every partial is in memory.
#ifndef PA_DECODE_OPUS_WRAP_SPLIT_COUNTER
#define PA_DECODE_OPUS_WRAP_SPLIT_COUNTER 1
#endif

template<class T>
__device__ inline bool split_arrive_is_last(const pa_decode_kargs& kargs, int b, int kvh, int tid)
{
    __shared__ int s_last;

    // Release for the whole block: the barrier orders every thread's partial stores
    // ahead of the counter bump, the waitcnt makes sure they retired. An agent-scope
    // release (__threadfence) would write back all of L2 on a multi-XCD part; the
    // scratch is uncached instead, so the stores retire at the coherence point.
    __syncthreads();
    __builtin_amdgcn_s_waitcnt(0);

    if(tid == 0)
    {
        unsigned int* slot = kargs.split_counters + b * kargs.num_kv_heads + kvh;
#if PA_DECODE_OPUS_WRAP_SPLIT_COUNTER
        const unsigned int limit = static_cast<unsigned int>(kargs.num_splits - 1);
        const unsigned int prev = atomicInc(slot, limit);
        const bool last = prev == limit;
#else
        const unsigned int prev = atomicAdd(slot, 1u);
        const bool last = prev + 1u == static_cast<unsigned int>(kargs.num_splits);
        if(last) atomicExch(slot, 0u);
#endif
        s_last = last ? 1 : 0;
    }
    __syncthreads();
    return s_last != 0;
}

// Merge this (batch, kv-head)'s per-split partials into the final output.
#ifndef PA_DECODE_OPUS_PARALLEL_SPLIT_REDUCE
#define PA_DECODE_OPUS_PARALLEL_SPLIT_REDUCE 1
#endif

#ifndef PA_DECODE_OPUS_SPLIT_O_VEC
#define PA_DECODE_OPUS_SPLIT_O_VEC 8
#endif

#ifndef PA_DECODE_OPUS_EARLY_PARTIAL_O
#define PA_DECODE_OPUS_EARLY_PARTIAL_O 0
#endif

#ifndef PA_DECODE_OPUS_DIRECT_SMALL_ML
#define PA_DECODE_OPUS_DIRECT_SMALL_ML 0
#endif

template<class T>
__device__ inline void reduce_splits(const pa_decode_kargs& kargs, int b, int kvh, int tid,
                                     char* smem)
{
    using D_OUT = typename T::D_OUT;
    using D_ACC = typename T::D_ACC;

    const int splits  = kargs.num_splits;
    const D_ACC* p_ml = partial_ml_ptr<T>(kargs, b, kvh, 0);
    const D_ACC* p_o  = partial_o_ptr<T>(kargs, b, kvh, 0);

    // Carved out of the attention body's buffer, dead by now. Mirrors the scratch
    // layout [split][2][Q_TILE], so the staging below is a straight copy.
    auto* s_ml    = reinterpret_cast<D_ACC*>(smem);
    auto* s_inv_l = s_ml + 2 * T::MAX_SPLITS * T::Q_TILE;

    // Stage with the whole block: read straight out of global this would be a
    // quarter wave walking 2*splits dependent loads through uncached scratch.
    const int ml_elems = splits * 2 * T::Q_TILE;
    const bool direct_small_ml = PA_DECODE_OPUS_DIRECT_SMALL_ML && T::IS_FP8
                                 && !T::QUANT_Q && T::HAS_SINK && !T::HAS_MTP
                                 && T::D_HEAD == 128 && T::PAGE_SIZE == 128 && splits <= 8;
    if(direct_small_ml)
    {
        if(tid < T::WARP_SIZE)
        {
            constexpr int GROUPS = T::WARP_SIZE / T::Q_TILE;
            const int row = tid % T::Q_TILE;
            const int group = tid / T::Q_TILE;
            D_ACC maxima[2];
            D_ACC sums[2];
            opus::static_for<2>([&](auto item) {
                const int split = group + item.value * GROUPS;
                maxima[item.value] = split < splits ? p_ml[split * 2 * T::Q_TILE + row]
                                                    : -opus::numeric_limits<D_ACC>::max();
                sums[item.value] = split < splits ? p_ml[split * 2 * T::Q_TILE + T::Q_TILE + row]
                                                  : D_ACC(0.0f);
            });
            D_ACC maximum = maxima[0] > maxima[1] ? maxima[0] : maxima[1];
            maximum = wave_row_fold(maximum, [](D_ACC lhs, D_ACC rhs) { return lhs > rhs ? lhs : rhs; });
            D_ACC total = 0.0f;
            opus::static_for<2>([&](auto item) {
                const int split = group + item.value * GROUPS;
                const D_ACC weight = __builtin_amdgcn_exp2f(maxima[item.value] - maximum);
                total += sums[item.value] * weight;
                if(split < splits) s_ml[split * 2 * T::Q_TILE + row] = weight;
            });
            total = wave_row_fold(total, [](D_ACC lhs, D_ACC rhs) { return lhs + rhs; });
            if(tid < T::Q_TILE)
                s_inv_l[row] = total > 0.0f ? kargs.v_dequant / total : 0.0f;
        }
    }
    else
    {
        for(int i = tid; i < ml_elems; i += T::BLOCK_SIZE)
            s_ml[i] = p_ml[i];
        __syncthreads();
    }

    auto reduce_ml = [&](auto group_count) {
        constexpr int REDUCE_GROUPS = decltype(group_count)::value;
        if(tid < REDUCE_GROUPS * T::Q_TILE)
        {
            const int row = tid % T::Q_TILE;
            const int split_group = tid / T::Q_TILE;
            D_ACC m_max = -opus::numeric_limits<D_ACC>::max();
            for(int i = split_group; i < splits; i += REDUCE_GROUPS)
            {
                const D_ACC m = s_ml[i * 2 * T::Q_TILE + row];
                if(m > m_max) m_max = m;
            }
            if constexpr(REDUCE_GROUPS > 1)
                m_max = wave_row_fold(m_max, [](D_ACC lhs, D_ACC rhs) { return lhs > rhs ? lhs : rhs; });

            // The softmax runs in log2 space, so the cross-split rescale is exp2 too.
            D_ACC l_sum = 0.0f;
            for(int i = split_group; i < splits; i += REDUCE_GROUPS)
            {
                const int base   = i * 2 * T::Q_TILE;
                const D_ACC c    = __builtin_amdgcn_exp2f(s_ml[base + row] - m_max);
                l_sum           += s_ml[base + T::Q_TILE + row] * c;
                s_ml[base + row] = c; // only this thread owns the slot
            }
            if constexpr(REDUCE_GROUPS > 1)
                l_sum = wave_row_fold(l_sum, [](D_ACC lhs, D_ACC rhs) { return lhs + rhs; });
            // v_dequant rides along here: the partials are sums of P*V in fp8 units, and
            // this is the one place per row where they get scaled anyway.
            D_ACC inv_l = l_sum > D_ACC(0.0f) ? D_ACC(1.0f) / l_sum : D_ACC(0.0f);
            if constexpr(T::IS_FP8) inv_l *= kargs.v_dequant;
            if(tid < T::Q_TILE) s_inv_l[row] = inv_l;
        }
    };
    if(!direct_small_ml)
    {
        if constexpr(PA_DECODE_OPUS_PARALLEL_SPLIT_REDUCE)
        {
            if(splits >= 8)
                reduce_ml(opus::number<T::WARP_SIZE / T::Q_TILE>{});
            else
                reduce_ml(opus::number<1>{});
        }
        else
            reduce_ml(opus::number<1>{});
    }
    __syncthreads();

    int out_b          = b;
    int64_t tok_off    = 0;
    int reduce_qlen    = T::HAS_MTP ? kargs.qlen : 1;
    if constexpr(T::MTP_Q_SPLIT)
    {
        tok_off     = static_cast<int64_t>(b % kargs.qlen) * kargs.stride_o_t;
        out_b       = b / kargs.qlen;
        reduce_qlen = 1;
    }
    auto* out = reinterpret_cast<D_OUT*>(kargs.out_ptr)
                + static_cast<int64_t>(out_b) * kargs.stride_o_b
                + static_cast<int64_t>(kvh) * kargs.gqa_ratio * kargs.stride_o_h
                + tok_off;

    // Contiguous dims per thread, bounded by the live row count so a tile that is not full
    // spends no iterations on rows it would discard. The partials are contiguous along
    // the head dim. MTP makes a row a (token, head) pair, so there are qlen times as
    // many of them, and they no longer land one stride_o_h apart in `out`.
    constexpr int VEC = T::HAS_SINK && !T::HAS_MTP && T::IS_FP8 && !T::QUANT_Q
                               && T::D_HEAD == 128 && T::PAGE_SIZE == 128
                           ? PA_DECODE_OPUS_SPLIT_O_VEC
                           : 4;
    static_assert(VEC == 4 || VEC == 8, "split output vectors must contain four or eight elements");
    constexpr int ROW_VECS = T::D_HEAD / VEC;
    const int live_rows    = T::HAS_MTP ? reduce_qlen * kargs.gqa_ratio : kargs.gqa_ratio;
    const int vec_total    = live_rows * ROW_VECS;

    for(int idx = tid; idx < vec_total; idx += T::BLOCK_SIZE)
    {
        const int row       = idx / ROW_VECS;
        const int dim       = (idx - row * ROW_VECS) * VEC;
        const int64_t o_row = static_cast<int64_t>(row) * T::D_HEAD + dim;

        using vec_t = opus::vector_t<D_ACC, VEC>;
        auto slice  = [&](int i) {
            return reinterpret_cast<const vec_t*>(
                p_o + static_cast<int64_t>(i) * T::Q_TILE * T::D_HEAD + o_row);
        };

        // Unrolled so several splits' loads are in flight: the trip count is a
        // runtime value, and the accumulate chain otherwise lets only one issue.
        constexpr int UNROLL = 4;
        D_ACC acc[VEC]       = {};
        int i                = 0;
        for(; i + UNROLL <= splits; i += UNROLL)
        {
            vec_t v[UNROLL];
            opus::static_for<UNROLL>([&](auto u) { v[u.value] = *slice(i + u.value); });
            opus::static_for<UNROLL>([&](auto u) {
                const D_ACC c = s_ml[(i + u.value) * 2 * T::Q_TILE + row];
                opus::static_for<VEC>([&](auto j) { acc[j.value] += v[u.value][j.value] * c; });
            });
        }
        for(; i < splits; ++i)
        {
            const D_ACC c = s_ml[i * 2 * T::Q_TILE + row];
            const vec_t v = *slice(i);
            opus::static_for<VEC>([&](auto j) { acc[j.value] += v[j.value] * c; });
        }

        // Adjacent by construction, so these fold into one wide store.
        const D_ACC inv_l = s_inv_l[row];
        int64_t row_off   = static_cast<int64_t>(row) * kargs.stride_o_h;
        if constexpr(T::HAS_MTP)
        {
            const int tok = row / kargs.gqa_ratio;
            row_off       = static_cast<int64_t>(tok) * kargs.stride_o_t
                      + static_cast<int64_t>(row - tok * kargs.gqa_ratio) * kargs.stride_o_h;
        }
        auto* dst = out + row_off + dim;
        opus::static_for<VEC>(
            [&](auto j) { dst[j.value] = static_cast<D_OUT>(acc[j.value] * inv_l); });
    }
}

// Token-loop MTP publishes one Q_TILE partial per query token. The last arriver
// merges each token as a qlen==1 view so the packed-row addressing in reduce_splits
// stays a single GQA group.
template<class T>
__device__ inline void reduce_splits_qpack(const pa_decode_kargs& kargs, int b, int kvh, int tid,
                                           char* smem)
{
    if constexpr(T::MTP_Q_LOOP)
    {
        for(int tok = 0; tok < T::MTP_Q_MAX; ++tok)
        {
            if(tok >= kargs.qlen) break;
            pa_decode_kargs view = kargs;
            view.qlen            = 1;
            view.out_ptr         = reinterpret_cast<typename T::D_OUT*>(kargs.out_ptr)
                           + static_cast<int64_t>(tok) * kargs.stride_o_t;
            const int64_t stride = mtp_token_slot_stride<T>(kargs);
            view.partial_o       = kargs.partial_o + tok * stride * T::Q_TILE * T::D_HEAD;
            view.partial_ml      = kargs.partial_ml + tok * stride * 2 * T::Q_TILE;
            reduce_splits<T>(view, b, kvh, tid, smem);
        }
    }
    else
        reduce_splits<T>(kargs, b, kvh, tid, smem);
}

// How a work unit publishes its result.
enum class pa_publish
{
    // One workgroup per split: a runtime branch on num_splits picks between writing
    // the final output and publishing (unnormalized O, m, l) for the fused
    // last-arrival merge.
    split_counter,
    // Work queue: the metadata kernel already decided per work item, so `slot` picks
    // between the final output and a normalized (O/l, lse) partial that the external
    // reduce kernel merges.
    split_lse,
};

// One unit of work: a (batch, kv-head) plus a contiguous run of KV tiles inside it.
// Both entry points funnel through here and differ only in how that run is chosen --
// split heuristic vs. metadata work queue -- and in how the result is published.
//
// `page_base` is the run's first page, indexed off whichever page list PUB implies,
// and `num_pages` bounds the buffer so the tail tile's over-read comes back as 0 with
// no guard in the instruction stream. `split_len` is relative to the run, which is
// what the decode mask wants; `kv_start` is the run's absolute KV offset and is only
// read by the MTP mask, whose per-row limits are stated against the whole context.
template<class Traits, pa_publish PUB>
__device__ __attribute__((always_inline)) void
pa_decode_opus_work(const pa_decode_kargs& kargs,
                    char* smem,
                    int batch_idx,
                    int kv_head_idx,
                    int page_base,
                    int num_pages,
                    int split_len,
                    int kv_start,
                    int num_tiles,
                    int tile_phase,
                    int slot,
                    bool owns_seq_start,
                    int tid,
                    int lane_id,
                    int warp_id,
                    int q_tok_cta = 0)
{
    using namespace opus;
    using T      = opus::remove_cvref_t<Traits>;
    using D_ATTN = typename T::D_ATTN;
    using D_Q    = typename T::D_Q;
    using D_OUT  = typename T::D_OUT;
    using D_ACC  = typename T::D_ACC;

    auto* p_smem  = reinterpret_cast<char*>(smem);
    auto* s_p_raw = reinterpret_cast<D_ATTN*>(p_smem);
    p_smem += T::smem_p_elems * sizeof(D_ATTN);
    auto* s_s_raw = reinterpret_cast<typename T::S_STORE*>(p_smem);
    p_smem += T::smem_s_elems * sizeof(typename T::S_STORE);
    auto* s_row_reduce = reinterpret_cast<D_ACC*>(p_smem);
    p_smem += T::smem_row_reduce_elems * sizeof(D_ACC);
    // This wave's own V rows. Private per wave, which is what lets the staging run
    // without a barrier.
    auto* s_v = reinterpret_cast<D_ATTN*>(p_smem) + warp_id * T::V_LDS_DIMS * T::V_LDS_ROW;
    p_smem += T::smem_v_bytes();

    auto s_p = make_smem(s_p_raw);

    // ── Matrix cores. swap_ab keeps the query row on `lane % W_M` for both GEMMs. ──
    auto mma0 = make_tiled_mma<D_ATTN, D_ATTN, D_ACC>(
        seq<T::GEMM0_E_M, T::GEMM0_E_N, T::GEMM0_E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});
    auto mma1 = make_tiled_mma<D_ATTN, D_ATTN, D_ACC>(
        seq<T::GEMM1_E_M, T::GEMM1_E_N, T::GEMM1_E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});

    // The KV loads below hand-roll the operand layout to read the paged cache in
    // place, so pin the one part of it the traits restate.
    static_assert(decltype(mma0)::rept_b == T::REPT && decltype(mma1)::rept_b == T::REPT,
                  "T::REPT must match the MFMA operand layout");

    // Every operand is plain row-major, so the fragment layouts come from the adaptor.
    // The width here is per issue, not per fragment: it vectorizes the innermost
    // (pack) dim and leaves REPT as separate issues, so a fragment is REPT dwordx4.
    const int q_head_base = kv_head_idx * kargs.gqa_ratio;

    // swap_ab keeps a lane on tile row `lane % W_M` through both GEMMs, so whatever
    // that row means is a per-lane scalar the whole kernel can share. Decode makes it
    // a query head; MTP makes it the pair (token, head) at row = token * gqa + head.
    int q_tok = 0, q_head = lane_id % T::W_M;
    if constexpr(T::MTP_Q_SPLIT)
    {
        q_tok  = q_tok_cta;
        q_head = lane_id % T::W_M;
    }
    else if constexpr(T::HAS_MTP && !T::MTP_Q_LOOP)
    {
        q_tok  = q_head / kargs.gqa_ratio;
        q_head = q_head - q_tok * kargs.gqa_ratio;
    }
    const int scratch_b = T::MTP_Q_SPLIT ? batch_idx * kargs.qlen + q_tok_cta : batch_idx;

    // A row of Q and a row of O are one stride off a shared base for decode, but MTP
    // walks the token and head dims by different strides, so there is no such stride.
    // Those two layouts move their row term into a per-lane base instead, leaving the
    // stride here at zero; both fold away, since HAS_MTP is compile time.
    auto u_q  = partition_layout_a<T::VEC>(
        mma0, opus::make_tuple(T::HAS_MTP ? 0 : kargs.stride_q_h, 1_I),
        opus::make_tuple(0_I, lane_id % mma0.grpm_a, 0_I, lane_id / mma0.grpm_a));
    // Writes P out of the C fragment, to D_ATTN, after softmax has run in registers.
    auto u_p  = partition_layout_c(
        mma0, opus::make_tuple(number<T::P_ROW_STRIDE>{}, 1_I),
        opus::make_tuple(0_I, lane_id % mma0.grpn_c, warp_id, lane_id / mma0.grpn_c));
    auto u_rp = partition_layout_a<T::VEC>(
        mma1, opus::make_tuple(number<T::P_ROW_STRIDE>{}, 1_I),
        opus::make_tuple(0_I, lane_id % mma1.grpm_a, 0_I, lane_id / mma1.grpm_a));
    auto u_o  = partition_layout_c(
        mma1, opus::make_tuple(T::HAS_MTP ? 0 : kargs.stride_o_h, 1_I),
        opus::make_tuple(0_I, lane_id % mma1.grpn_c, warp_id, lane_id / mma1.grpn_c));

    // ── Load Q. Packed MTP rebases onto one (token, head) row; the token loop
    // loads every query token so the KV walk can reuse K/V across them. ──
    typename decltype(mma0)::vtype_a v_q;
    D_ACC q_dequant = D_ACC(1.0f);
    constexpr int Q_KEEP = T::MTP_Q_LOOP ? T::MTP_Q_MAX : 1;
    typename decltype(mma0)::vtype_a v_q_tok[Q_KEEP];
    D_ACC q_dequant_tok[Q_KEEP];

    auto load_q_token = [&](int tok, auto& v_q_dst, D_ACC& dequant_dst) {
        int64_t q_offset = static_cast<int64_t>(batch_idx) * kargs.stride_q_b
                           + static_cast<int64_t>(q_head_base) * kargs.stride_q_h;
        size_t q_bytes = static_cast<size_t>(kargs.gqa_ratio) * kargs.stride_q_h * sizeof(D_Q);
        if constexpr(T::HAS_MTP)
        {
            const int live_tok = T::MTP_Q_LOOP ? tok : q_tok;
            q_offset += static_cast<int64_t>(live_tok) * kargs.stride_q_t
                        + static_cast<int64_t>(q_head) * kargs.stride_q_h;
            q_bytes = live_tok < kargs.qlen ? T::D_HEAD * sizeof(D_Q) : 0;
        }
        auto g_q = make_gmem(reinterpret_cast<const D_Q*>(kargs.q_ptr) + q_offset, q_bytes);
        dequant_dst = D_ACC(1.0f);
        if constexpr(T::QUANT_Q)
        {
            constexpr int LANE_D = T::K16_PERM ? T::K_PACK : T::VEC;
            constexpr int Q_FRAG = T::GEMM0_E_M * T::GEMM0_E_K * T::ELEM_A;
            opus::vector_t<D_Q, Q_FRAG> v_q_in;
            int q_lane_off = (lane_id / T::W_M) * LANE_D;
            if constexpr(!T::HAS_MTP)
                q_lane_off += (lane_id % T::W_M) * kargs.stride_q_h;
            opus::static_ford<T::GEMM0_E_K, T::REPT, T::VEC / T::Q_VEC>(
                [&](auto ik, auto ir, auto ic) {
                    constexpr int i_k = ik.value, i_r = ir.value, i_c = ic.value;
                    constexpr int d_off =
                        T::K16_PERM
                            ? i_k * T::VEC + i_c * T::Q_VEC
                            : i_k * T::W_K + i_r * (T::GRP_K * T::VEC) + i_c * T::Q_VEC;
                    constexpr int frag  = i_k * T::ELEM_A + i_r * T::VEC + i_c * T::Q_VEC;
                    const auto f        = g_q.template _load<T::Q_VEC>(
                        (q_lane_off + d_off) * static_cast<int>(sizeof(D_Q)));
                    opus::static_for<T::Q_VEC>(
                        [&](auto j) { v_q_in[frag + j.value] = f[j.value]; });
                });
            D_ACC amax = 0.0f;
            opus::static_for<Q_FRAG>([&](auto i) {
                const D_ACC a = __builtin_fabsf(static_cast<D_ACC>(v_q_in[i.value]));
                amax          = a > amax ? a : amax;
            });
            amax = wave_row_fold(amax, [](D_ACC a, D_ACC b) { return a > b ? a : b; });
            const D_ACC q_inv = amax > D_ACC(0.0f) ? D_ACC(T::FP8_MAX) / amax : D_ACC(0.0f);
            dequant_dst       = amax * (D_ACC(1.0f) / D_ACC(T::FP8_MAX));
            vector_t<D_ACC, Q_FRAG> q_scaled;
            opus::static_for<Q_FRAG>([&](auto i) {
                q_scaled[i.value] = static_cast<D_ACC>(v_q_in[i.value]) * q_inv;
            });
            v_q_dst = cast<D_ATTN>(q_scaled);
        }
        else
        {
            v_q_dst = load<T::VEC>(g_q, u_q);
        }
    };
    if constexpr(T::MTP_Q_LOOP)
    {
        opus::static_for<T::MTP_Q_MAX>([&](auto itok) {
            constexpr int tok = itok.value;
            if(tok < kargs.qlen) load_q_token(tok, v_q_tok[tok], q_dequant_tok[tok]);
        });
    }
    else
    {
        load_q_token(0, v_q, q_dequant);
    }

    typename decltype(mma0)::vtype_b v_k;
    typename decltype(mma0)::vtype_c v_s;
    typename decltype(mma1)::vtype_a v_p;
    typename decltype(mma1)::vtype_b v_v;
    typename decltype(mma1)::vtype_c v_o;
    clear(v_o);
    typename decltype(mma1)::vtype_c v_o_tok[Q_KEEP];
    D_ACC m_tok[Q_KEEP];
    D_ACC l_tok[Q_KEEP];
    D_ACC temperature_tok[Q_KEEP];
    int valid_tok[Q_KEEP];
    if constexpr(T::MTP_Q_LOOP)
    {
        opus::static_for<T::MTP_Q_MAX>([&](auto itok) {
            constexpr int tok = itok.value;
            clear(v_o_tok[tok]);
            m_tok[tok] = opus::numeric_limits<D_ACC>::lowest();
            l_tok[tok] = 0.0f;
        });
    }

    // Where a tile's prefetch lands, which is not always the MFMA operand: the 16B K
    // path stages packs that convert_k_pack_frags later splits into v_k, and the LDS V
    // path stages chunks that go to LDS and come back as v_v. Those staging sets are
    // what a deeper prefetch has to duplicate. The direct paths load straight into the
    // operand, and there v_k / v_v above are unused.
    using k_pref_t = std::conditional_t<T::K16_PACK, k_stage_t<T>, decltype(v_k)>;
    using v_pref_t = std::conditional_t<T::V_VIA_LDS, v_stage_t<T>, decltype(v_v)>;

    constexpr int DEPTH = T::PREFETCH_DEPTH;
    constexpr bool PAGE16_PAIR = T::PAGE16_PAIR && DEPTH == 1;
    constexpr int K_BUFS = PAGE16_PAIR && !T::MTP_Q_LOOP ? 2 : DEPTH;
    constexpr int V_BUFS = PAGE16_PAIR && !T::MTP_Q_LOOP ? 2 : DEPTH;
    constexpr bool V_CHUNK_PIPELINE = PA_DECODE_OPUS_V_CHUNK_PIPELINE
        && T::V_VIA_LDS && PUB == pa_publish::split_counter && T::IS_FP8 && !T::QUANT_Q
        && T::HAS_SINK && !T::HAS_MTP && T::D_HEAD == 128
        && T::PAGE_SIZE == 128 && T::KV_TILE == 128 && DEPTH == 1;
    constexpr bool BT_SCALAR = T::BT_SCALAR
        || (V_CHUNK_PIPELINE && PA_DECODE_OPUS_V_CHUNK_PIPELINE >= 2);
    k_pref_t      k_pf[K_BUFS];
    v_pref_t      v_pf[V_BUFS];
    // Block-table entries are indexed the same way: the set a body prefetches off was
    // read by the body before it, so at depth 2 there are two of them in flight.
    // PAGE16_PAIR keeps a second set so both halves of the 256-token window can
    // issue K without waiting on the pair's own QK.
    page_ids_t<T> pid[K_BUFS];

    constexpr D_ACC LOG2_E = 1.44269504089f;
    D_ACC temperature_scale = kargs.softmax_scale * LOG2_E;
    // qk_dequant undoes the fp8 quantization of GEMM0's operands: q_scale * k_scale
    // when Q arrived quantized, k_scale alone when the kernel quantized it and
    // q_dequant carries the per-row half. S is only ever read through this scale, so
    // folding it in is free -- and per-row is free too, since the lane already holds
    // exactly one query row.
    if constexpr(T::IS_FP8) temperature_scale *= kargs.qk_dequant;
    if constexpr(T::QUANT_Q && !T::MTP_Q_LOOP) temperature_scale *= q_dequant;
    if constexpr(T::MTP_Q_LOOP)
    {
        opus::static_for<T::MTP_Q_MAX>([&](auto itok) {
            constexpr int tok      = itok.value;
            temperature_tok[tok]   = temperature_scale;
            if constexpr(T::QUANT_Q) temperature_tok[tok] *= q_dequant_tok[tok];
        });
    }

    // Online softmax state, per lane rather than per LDS row. m is held equal
    // across waves by the per-tile fold; l is this wave's own partial sum.
    D_ACC m_run = opus::numeric_limits<D_ACC>::lowest();
    D_ACC l_run = 0.0f;
    if constexpr(T::HAS_SINK)
    {
        // Exactly one unit per request may seed it: with split-KV every unit builds its
        // own (m, l), and the merge sums the l's, so a sink seeded everywhere would be
        // counted once per split. The unit holding the request's first token is the
        // unique choice, and the merge's weighting then puts the term back in the
        // denominator exactly once -- the algebra works out for both merge paths.
        if(owns_seq_start)
        {
        // The sink is a KV column with no value: it belongs in the softmax denominator
        // and nowhere else. Seeding the running pair with it says exactly that, and the
        // per-tile rescale then carries it along for free -- l_run's invariant is
        // sum of exp2(x - (m_run - P_LOG2)), which one sink term satisfies at 2^P_LOG2.
        //
        // The sink is per query head, not per tile row. Decode's tile row *is* a
        // head, so `q_head == lane % W_M`. MTP packs (token, head) into the row,
        // and tokens that share a head share the logit -- indexing by lane would
        // walk past the GQA group on token 1 and read 0 (or the next kv-head).
        auto g_sink = make_gmem(kargs.sink + q_head_base,
                                kargs.gqa_ratio * static_cast<unsigned int>(sizeof(D_ACC)));
        // Already a scaled logit, so only the move into log2 space is needed.
        m_run = g_sink.template _load<1>(q_head * static_cast<int>(sizeof(D_ACC)))[0]
                * LOG2_E;
        // m is a max, so seeding it in every lane is idempotent -- and it has to be,
        // since the per-tile fold expects every lane to agree. l is not: it stays a
        // per-lane, per-wave partial until the sum after the last tile, so a seed in
        // every slot would land in that sum 16 times over. Lanes 0..W_M-1 of wave 0
        // hold rows 0..W_M-1 with nothing else, which is exactly one slot per row.
        if(tid < T::W_M) l_run = D_ACC(1 << T::P_LOG2);
        }
    }

    const D_ATTN* p_k = reinterpret_cast<const D_ATTN*>(kargs.k_ptr) + kv_head_idx * kargs.stride_k_h;
    const D_ATTN* p_v = reinterpret_cast<const D_ATTN*>(kargs.v_ptr) + kv_head_idx * kargs.stride_v_h;
    // The rectangular block table is indexed per batch row; the persistent path's page
    // list is already compacted, so its rows are found through kv_indptr instead.
    const int* bt_base;
    if constexpr(PUB == pa_publish::split_counter)
        bt_base = kargs.block_tables
                  + static_cast<int64_t>(batch_idx) * kargs.max_blocks_per_batch_row + page_base;
    else
        bt_base = kargs.kv_indices + page_base;
    auto g_bt = make_gmem(bt_base, num_pages * static_cast<unsigned int>(sizeof(int)));

    // KV and page ids are prefetched Traits::PREFETCH_DEPTH tiles ahead. Each KV
    // operand is refetched right after the GEMM that consumed it, so its load stays in
    // flight for the rest of the body: K hides behind softmax and GEMM1, V behind the
    // next tile's GEMM0. The last DEPTH tiles are peeled out -- they have nothing left
    // to fetch, and only the last can run past the context, so the interior body needs
    // neither the mask nor a `has_next` guard.
    //
    // The wart at depth 1 is that vmcnt is a single in-order counter, so a wait for any
    // load also retires everything issued before it. Within a body the issue order is
    // page ids, K, V, but the consumption order across the loop boundary is K, page
    // ids, V -- the next body needs the page ids to form its KV addresses long before
    // it touches V. So the wait that covers the page ids drains the V loads with them,
    // roughly 90 instruction slots before V is used, and shows up as a full vmcnt(0)
    // in the middle of GEMM0.
    //
    // Depth 2 dissolves that -- the ids are read a whole body before the addresses that
    // need them -- and takes the loop from 6 loads in flight to 16. It also takes the
    // gpt-oss shape from 5 waves per SIMD to 3. The SMEM specialization moves page
    // ids to lgkmcnt without duplicating KV buffers, but cannot remove waits for
    // the KV data itself.
    //
    // Sinking the page ids without the second buffer does not work: nothing ties those
    // loads to the KV loads, so the scheduler hoists them back to the top of the body
    // and vmcnt(0) stays. It cost 8 VGPR and came out inside noise (fp8 -1%, bf16 +1%).
    const kv_frag_slice<T> kv_slice(lane_id, warp_id);
    // Which tile of its page this is, in tokens, and which page it lands on. Both are
    // trivial unless a page holds several tiles. `tile_phase` is where this run starts
    // inside its first page: the split heuristic cuts on tile boundaries, which need
    // not be page boundaries, and the block table was rebased to the page containing
    // the cut -- so the run's tile 0 is not necessarily its page's tile 0.
    auto abs_tile     = [&](int t) { return T::PAGE_HOLDS_TILE ? tile_phase + t : t; };
    auto tok_base_of  = [&](int t) {
        return T::PAGE_HOLDS_TILE ? (abs_tile(t) % T::TILES_PER_PAGE) * T::KV_TILE : 0;
    };
    // Issue one tile's KV, without waiting: the caller places these so the latency
    // falls inside the DEPTH bodies that run before the tile is consumed.
    auto issue_k = [&](k_pref_t& dst, const page_ids_t<T>& p, int tok_base) {
        const k_page_ids_t<T> kp = k_pages_from<T>(p, warp_id);
        if constexpr(T::K16_PACK)
            load_k_pack<T>(p_k, kp, kargs.stride_k_blk, tok_base, lane_id, warp_id, dst);
        else
            load_k_frags<T>(p_k, kp, kargs.stride_k_blk, kv_slice, tok_base, dst);
    };
    auto issue_v = [&](v_pref_t& dst, const page_ids_t<T>& p, int tok_base) {
        if constexpr(T::V_SHUFFLED)
            load_v_shuffled_frags<T>(p_v, p, kargs.stride_v_blk, tok_base,
                                    lane_id, warp_id, dst);
        else if constexpr(T::V_VIA_LDS)
            load_v_stage<T>(p_v, p, kargs.stride_v_blk, tok_base, lane_id, warp_id, dst);
        else
            load_v_frags<T>(p_v, p, kargs.stride_v_blk, kv_slice, tok_base, dst);
    };

    // Prime every buffer, then leave pid[0] on the first tile the loop will fetch --
    // body t reads pid[t % DEPTH] to address tile t + DEPTH. The block-table reads go
    // first and together: nothing depends on another, and every KV address here waits
    // on one of them.
    {
        page_ids_t<T> warm[K_BUFS];
        opus::static_for<K_BUFS>(
            [&](auto i) { warm[i.value] = load_page_ids<T, BT_SCALAR>(g_bt, bt_base, num_pages, abs_tile(i.value)); });
        // Depth 1 reads its set inside the body, so priming it here would only be a
        // load the first body repeats.
        if constexpr(DEPTH > 1)
            pid[0] = load_page_ids<T, BT_SCALAR>(g_bt, bt_base, num_pages, abs_tile(DEPTH));
        opus::static_for<K_BUFS>([&](auto i) {
            constexpr int t = i.value;
            if(t < num_tiles)
            {
                issue_k(k_pf[t], warm[t], tok_base_of(t));
                if constexpr(t < V_BUFS)
                    issue_v(v_pf[t], warm[t], tok_base_of(t));
            }
        });
        // Depth 1 used to leave pid unset and reload the current tile's pages
        // inside the body for per-token scales. Keep the warmup set so the first
        // body can read scales off pid before it overwrites it with the prefetch.
        if constexpr(DEPTH == 1)
        {
            pid[0] = warm[0];
            if constexpr(PAGE16_PAIR)
                pid[1] = warm[1];
        }
    }

    // How far into this run a row may look. Decode gives every row the whole run and
    // only the last tile can overrun it. MTP is tail-causal -- query token t attends
    // the first (context_len - qlen + 1 + t) KV tokens -- so each row stops somewhere
    // else. Restating that against the run and clamping to it keeps the compare in
    // online_softmax_frag exactly as it is; only the bound stops being wave-uniform.
    int valid_kv = split_len;
    if constexpr(T::HAS_MTP && !T::MTP_Q_LOOP)
    {
        const int row_end = kargs.context_lens[batch_idx] - kargs.qlen + 1 + q_tok - kv_start;
        valid_kv          = row_end < split_len ? row_end : split_len;
    }
    if constexpr(T::MTP_Q_LOOP)
    {
        opus::static_for<T::MTP_Q_MAX>([&](auto itok) {
            constexpr int tok = itok.value;
            const int row_end = kargs.context_lens[batch_idx] - kargs.qlen + 1 + tok - kv_start;
            valid_tok[tok]    = row_end < split_len ? row_end : split_len;
        });
    }

    // GEMM0's B operand is the split packs on the 16B K path and the prefetch buffer
    // itself on the direct one; GEMM1's is the LDS round trip or, again, the buffer.
    auto gemm0 = [&](auto& q_reg, auto& b) {
        // The f8f6f4 MFMA only has a scaled form; the per-tensor descales are folded
        // into the softmax and the final 1/l, so both exponents are the bare 1.0.
        if constexpr(T::SCALED_MFMA)
            return mma0(q_reg, b, 0, 0);
        else
            return mma0(q_reg, b);
    };
    auto gemm1 = [&](auto& b, auto& o_reg) {
        if constexpr(T::SCALED_MFMA)
            return mma1(v_p, b, o_reg, 0, 0);
        else
            return mma1(v_p, b, o_reg);
    };


    // Row 0 of an MTP tile stops qlen-1 tokens before the run's end, so when the tail
    // tile holds fewer than that its window has already closed inside the tile before
    // it, and one masked tile is no longer enough. Never more than one extra: qlen is
    // capped at Q_TILE, which is well under KV_TILE.
    constexpr auto MASK_LAST = opus::number<1>{};
    constexpr auto MASK_PREV = opus::number<T::HAS_MTP ? 1 : 0>{};

    if constexpr(PAGE16_PAIR)
    {
        // Page-16 dual-tile software pipe. Decode/q-split: one softmax over two
        // KV_TILE=128 halves (a 256-token compute block). Fused MTP keeps
        // four live scores and walks the two halves sequentially -- combining them
        // needs eight score tiles and drops to one wave.
        auto run_pair = [&](int tile_idx, auto masked0, auto masked1, auto prefetch, auto nh) {
            constexpr int  NH         = decltype(nh)::value;
            constexpr bool PAIR_MASK0 = decltype(masked0)::value != 0;
            constexpr bool PAIR_MASK1 = decltype(masked1)::value != 0;
            constexpr bool PREFETCH   = decltype(prefetch)::value != 0;
            constexpr int GEMM0_SCHED = PA_DECODE_OPUS_GEMM0_SCHED;
            constexpr int SCALE_N = T::PER_TOKEN_SCALE ? T::GEMM0_E_N * T::ELEM_C : 1;
            float ksc[NH][SCALE_N];
            float vsc[NH][SCALE_N];
            auto load_half_scales = [&](int half) {
                if constexpr(T::PER_TOKEN_SCALE)
                {
                    load_kv_scale_frag<T>(kargs.k_scale_map, kargs.stride_ks_blk,
                                          kargs.stride_ks_h, pid[half], kv_head_idx,
                                          tok_base_of(tile_idx + half), lane_id, warp_id,
                                          ksc[half]);
                    load_kv_scale_frag<T>(kargs.v_scale_map, kargs.stride_vs_blk,
                                          kargs.stride_vs_h, pid[half], kv_head_idx,
                                          tok_base_of(tile_idx + half), lane_id, warp_id,
                                          vsc[half]);
                }
            };
            auto p_off = [&](int tok, int half) {
                return (tok * T::smem_p_kv_tiles + half) * T::Q_TILE * T::P_ROW_STRIDE;
            };
            auto prefetch_next_k = [&]() {
                if constexpr(!PREFETCH) return;
                pid[0] = load_page_ids<T, BT_SCALAR>(
                    g_bt, bt_base, num_pages, abs_tile(tile_idx + NH));
                issue_k(k_pf[0], pid[0], tok_base_of(tile_idx + NH));
                if constexpr(NH == 2)
                {
                    if(tile_idx + 3 < num_tiles)
                    {
                        pid[1] = load_page_ids<T, BT_SCALAR>(
                            g_bt, bt_base, num_pages, abs_tile(tile_idx + 3));
                        issue_k(k_pf[1], pid[1], tok_base_of(tile_idx + 3));
                    }
                }
            };

            if constexpr(GEMM0_SCHED != 0) __builtin_amdgcn_sched_barrier(0);
            if constexpr(GEMM0_SCHED == 0) __builtin_amdgcn_sched_barrier(0);

            if constexpr(T::MTP_Q_LOOP)
            {
                // Combined 256-token softmax, one K and one V. Both halves of S
                // live in LDS; k_pf[0] is refilled with half-1 after half-0 QK,
                // v_pf[0] after half-0 PV. Dual buffers would land at 274 VGPR.
                typename decltype(mma0)::vtype_c v_s;
                D_ACC wave_max[T::MTP_Q_MAX];
                page_ids_t<T> pid_half1;
                opus::static_for<T::MTP_Q_MAX>([&](auto itok) {
                    wave_max[itok.value] = opus::numeric_limits<D_ACC>::lowest();
                });
                opus::static_for<NH>([&](auto ih) {
                    constexpr int H = ih.value;
                    constexpr bool HMASK = H == 0 ? PAIR_MASK0 : PAIR_MASK1;
                    opus::static_for<T::MTP_Q_MAX>([&](auto itok) {
                        constexpr int tok = itok.value;
                        if(tok >= kargs.qlen) return;
                        v_s = gemm0(v_q_tok[tok], k_pf[0]);
                        if constexpr(tok == 0)
                        {
                            if constexpr(T::PER_TOKEN_SCALE)
                            {
                                const page_ids_t<T>& sp = (H == 0) ? pid[0] : pid_half1;
                                load_kv_scale_frag<T>(kargs.k_scale_map, kargs.stride_ks_blk,
                                                      kargs.stride_ks_h, sp, kv_head_idx,
                                                      tok_base_of(tile_idx + H), lane_id,
                                                      warp_id, ksc[H]);
                                load_kv_scale_frag<T>(kargs.v_scale_map, kargs.stride_vs_blk,
                                                      kargs.stride_vs_h, sp, kv_head_idx,
                                                      tok_base_of(tile_idx + H), lane_id,
                                                      warp_id, vsc[H]);
                            }
                            if constexpr(GEMM0_SCHED >= 2)
                            {
                                opus::static_for<T::GEMM0_E_N * T::GEMM0_E_K>([&](auto) {
                                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
                                });
                            }
                        }
                        if constexpr(T::PER_TOKEN_SCALE)
                            mul_score_scales<T>(v_s, ksc[H]);
                        const D_ACC wm = softmax_scale_mask_wave_max<T, HMASK>(
                            v_s, temperature_tok[tok], valid_tok[tok], tile_idx + H,
                            lane_id, warp_id);
                        wave_max[tok] = wave_max[tok] > wm ? wave_max[tok] : wm;
                        spill_score_lds<T>(s_s_raw, v_s, tok, H, tid);
                    });
                    if constexpr(NH == 2 && H == 0)
                    {
                        pid_half1 = load_page_ids<T, BT_SCALAR>(
                            g_bt, bt_base, num_pages, abs_tile(tile_idx + 1));
                        issue_k(k_pf[0], pid_half1, tok_base_of(tile_idx + 1));
                    }
                });
                if constexpr(PREFETCH)
                {
                    pid[0] = load_page_ids<T, BT_SCALAR>(
                        g_bt, bt_base, num_pages, abs_tile(tile_idx + NH));
                    issue_k(k_pf[0], pid[0], tok_base_of(tile_idx + NH));
                }
                fold_across_waves_n<T, T::MTP_Q_MAX>(
                    s_row_reduce, wave_max, lane_id % T::W_M, warp_id,
                    [](D_ACC a, D_ACC b) { return a > b ? a : b; });
                opus::static_for<NH>([&](auto ih) {
                    constexpr int H = ih.value;
                    opus::static_for<T::MTP_Q_MAX>([&](auto itok) {
                        constexpr int tok = itok.value;
                        if(tok >= kargs.qlen) return;
                        fill_score_lds<T>(v_s, s_s_raw, tok, H, tid);
                        const D_ACC rescale = softmax_apply_tile_max<T>(
                            v_s, wave_max[tok], m_tok[tok], l_tok[tok]);
                        if constexpr(H == 0)
                        {
                            opus::static_for<vector_traits<decltype(v_o_tok[tok])>::size()>(
                                [&](auto i) { v_o_tok[tok][i.value] *= rescale; });
                        }
                        if constexpr(T::PER_TOKEN_SCALE)
                            mul_score_scales<T>(v_s, vsc[H]);
                        auto v_p_out = cast<D_ATTN>(v_s);
                        auto s_p_tok = make_smem(s_p_raw + p_off(tok, H));
                        store<4>(s_p_tok, v_p_out, u_p);
                    });
                });
                opus::s_waitcnt_lgkmcnt(opus::number<0>{});
                __builtin_amdgcn_s_barrier();
                opus::static_for<NH>([&](auto ih) {
                    constexpr int H = ih.value;
                    opus::static_for<T::MTP_Q_MAX>([&](auto itok) {
                        constexpr int tok = itok.value;
                        if(tok >= kargs.qlen) return;
                        auto s_p_tok = make_smem(s_p_raw + p_off(tok, H));
                        v_p          = load<T::VEC>(s_p_tok, u_rp);
                        v_o_tok[tok] = gemm1(v_pf[0], v_o_tok[tok]);
                    });
                    if constexpr(NH == 2 && H == 0)
                        issue_v(v_pf[0], pid_half1, tok_base_of(tile_idx + 1));
                });
                if constexpr(PREFETCH)
                    issue_v(v_pf[0], pid[0], tok_base_of(tile_idx + NH));
            }
            else
            {
                typename decltype(mma0)::vtype_c v_s_pair[NH];
                v_s_pair[0] = gemm0(v_q, k_pf[0]);
                load_half_scales(0);
                if constexpr(NH == 2)
                {
                    v_s_pair[1] = gemm0(v_q, k_pf[1]);
                    load_half_scales(1);
                }
                if constexpr(GEMM0_SCHED >= 2)
                {
                    opus::static_for<T::GEMM0_E_N * T::GEMM0_E_K>([&](auto) {
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
                    });
                }
                prefetch_next_k();
                if constexpr(T::PER_TOKEN_SCALE)
                    mul_score_scales<T>(v_s_pair[0], ksc[0]);
                D_ACC wm = softmax_scale_mask_wave_max<T, PAIR_MASK0>(
                    v_s_pair[0], temperature_scale, valid_kv, tile_idx, lane_id, warp_id);
                if constexpr(NH == 2)
                {
                    if constexpr(T::PER_TOKEN_SCALE)
                        mul_score_scales<T>(v_s_pair[1], ksc[1]);
                    const D_ACC max1 = softmax_scale_mask_wave_max<T, PAIR_MASK1>(
                        v_s_pair[1], temperature_scale, valid_kv, tile_idx + 1, lane_id,
                        warp_id);
                    wm = wm > max1 ? wm : max1;
                }
                D_ACC wave_max[1] = {wm};
                fold_across_waves_n<T, 1>(
                    s_row_reduce, wave_max, lane_id % T::W_M, warp_id,
                    [](D_ACC a, D_ACC b) { return a > b ? a : b; });
                const D_ACC rescale = softmax_apply_tile_max<T>(
                    v_s_pair[0], wave_max[0], m_run, l_run);
                if constexpr(NH == 2)
                    softmax_apply_tile_max<T>(v_s_pair[1], wave_max[0], m_run, l_run);
                opus::static_for<vector_traits<decltype(v_o)>::size()>(
                    [&](auto i) { v_o[i.value] *= rescale; });
                if constexpr(T::PER_TOKEN_SCALE)
                    mul_score_scales<T>(v_s_pair[0], vsc[0]);
                auto v_p0 = cast<D_ATTN>(v_s_pair[0]);
                auto s_p0 = make_smem(s_p_raw + p_off(0, 0));
                store<4>(s_p0, v_p0, u_p);
                if constexpr(NH == 2)
                {
                    if constexpr(T::PER_TOKEN_SCALE)
                        mul_score_scales<T>(v_s_pair[1], vsc[1]);
                    auto v_p1 = cast<D_ATTN>(v_s_pair[1]);
                    auto s_p1 = make_smem(s_p_raw + p_off(0, 1));
                    store<4>(s_p1, v_p1, u_p);
                    opus::s_waitcnt_lgkmcnt(opus::number<0>{});
                    __builtin_amdgcn_s_barrier();
                    v_p = load<T::VEC>(s_p0, u_rp);
                    v_o = gemm1(v_pf[0], v_o);
                    v_p = load<T::VEC>(s_p1, u_rp);
                    v_o = gemm1(v_pf[1], v_o);
                }
                else
                {
                    opus::s_waitcnt_lgkmcnt(opus::number<0>{});
                    __builtin_amdgcn_s_barrier();
                    v_p = load<T::VEC>(s_p0, u_rp);
                    v_o = gemm1(v_pf[0], v_o);
                }
                if constexpr(PREFETCH)
                {
                    issue_v(v_pf[0], pid[0], tok_base_of(tile_idx + NH));
                    if constexpr(NH == 2)
                    {
                        if(tile_idx + 3 < num_tiles)
                            issue_v(v_pf[1], pid[1], tok_base_of(tile_idx + 3));
                    }
                }
            }
        };

        int tile_idx = 0;
        constexpr int TAIL = T::HAS_MTP ? 2 : 1;
        for (; tile_idx + 2 + TAIL <= num_tiles; tile_idx += 2)
            run_pair(tile_idx, 0_I, 0_I, 1_I, 2_I);
        const int left = num_tiles - tile_idx;
        // Never call run_tile from this instantiation: its fused score tiles
        // would stay live with v_s_pair and drop occupancy to one wave.
        if (left == 3)
        {
            run_pair(tile_idx, 0_I, 0_I, 1_I, 1_I);
            run_pair(tile_idx + 1, MASK_PREV, 0_I, 1_I, 1_I);
            run_pair(tile_idx + 2, MASK_LAST, 0_I, 0_I, 1_I);
        }
        else if (left == 2)
            run_pair(tile_idx, MASK_PREV, MASK_LAST, 0_I, 2_I);
        else if (left == 1)
            run_pair(tile_idx, MASK_LAST, 0_I, 0_I, 1_I);
    }
    else
    {
        // MASKED, PREFETCH and CUR are compile time, so no branch and no runtime buffer
        // index survives into any instantiation.
        auto run_tile = [&](int tile_idx, auto masked, auto prefetch, auto cur) {
            constexpr bool MASKED   = decltype(masked)::value != 0;
            constexpr bool PREFETCH = decltype(prefetch)::value != 0;
            constexpr int  CUR      = decltype(cur)::value;
            constexpr int  NXT      = (CUR + 1) % DEPTH;

            constexpr int GEMM0_SCHED = PA_DECODE_OPUS_GEMM0_SCHED;
            constexpr bool FRAGMENT_REFILL = PA_DECODE_OPUS_V5_FRAGMENT_REFILL
                             && T::V_SHUFFLED && DEPTH == 1;

            // Per-token k/v scales live on the *current* tile's pages. At depth 1
            // pid[CUR] still holds that set from warmup or the previous body's
            // prefetch, so read scales first and only then overwrite it. Reloading
            // the block table here used to sit on GEMM0's vmcnt every tile.
            // Fused MTP delays both the scale loads and the next-pid fetch until
            // after token 0's GEMM0, so that MFMA covers them.
            constexpr int SCALE_N = T::PER_TOKEN_SCALE ? T::GEMM0_E_N * T::ELEM_C : 1;
            float ksc[SCALE_N];
            float vsc[SCALE_N];
            auto load_cur_scales = [&]() {
                if constexpr(T::PER_TOKEN_SCALE)
                {
                    if constexpr(DEPTH == 1)
                    {
                        load_kv_scale_frag<T>(kargs.k_scale_map, kargs.stride_ks_blk,
                                              kargs.stride_ks_h, pid[CUR], kv_head_idx,
                                              tok_base_of(tile_idx), lane_id, warp_id, ksc);
                        load_kv_scale_frag<T>(kargs.v_scale_map, kargs.stride_vs_blk,
                                              kargs.stride_vs_h, pid[CUR], kv_head_idx,
                                              tok_base_of(tile_idx), lane_id, warp_id, vsc);
                    }
                    else
                    {
                        const auto scale_pid = load_page_ids<T, BT_SCALAR>(
                            g_bt, bt_base, num_pages, abs_tile(tile_idx));
                        load_kv_scale_frag<T>(kargs.k_scale_map, kargs.stride_ks_blk,
                                              kargs.stride_ks_h, scale_pid, kv_head_idx,
                                              tok_base_of(tile_idx), lane_id, warp_id, ksc);
                        load_kv_scale_frag<T>(kargs.v_scale_map, kargs.stride_vs_blk,
                                              kargs.stride_vs_h, scale_pid, kv_head_idx,
                                              tok_base_of(tile_idx), lane_id, warp_id, vsc);
                    }
                }
            };
            auto load_next_pid = [&]() {
                if constexpr(PREFETCH)
                {
                    constexpr int AHEAD = DEPTH == 1 ? 1 : 1 + DEPTH;
                    pid[NXT] = load_page_ids<T, BT_SCALAR>(
                        g_bt, bt_base, num_pages, abs_tile(tile_idx + AHEAD));
                }
            };
            if constexpr(!T::MTP_Q_LOOP)
            {
                load_cur_scales();
                load_next_pid();
            }
            // The block-table reads are pinned here instead of below: they still have to
            // precede everything that addresses off them, but the barrier no longer stands
            // between the tile's own setup and the MFMAs that could cover it.
            if constexpr(GEMM0_SCHED != 0) __builtin_amdgcn_sched_barrier(0);
            // Hand this tile's staged V to LDS here, at the top: the read that picks
            // fragments out of it sits after GEMM0 and the softmax, which is enough work
            // to cover the round trip.
            //
            // The waitcnt this write sits behind is the kernel's top stall site, and moving
            // either end of it is a dead lever. Pushing the write down to the fragment read
            // only converts the global wait into LDS stall; pulling the global load up next
            // to K's, writing LDS after GEMM0, and leaving V in flight with vmcnt(2) -- so
            // GEMM0 runs while V arrives -- lands in the ISA as intended (HIP graph, forced
            // rebuild): b128/16k/page256 stays 1.02x Gluon, b64/16k falls from 0.96x to
            // 0.91x. The extra K+V burst after GEMM0 backs up the issue queue at the occupancy
            // that cannot cover it. Reverted.
            if constexpr(T::K16_PACK)
                convert_k_pack_frags<T>(k_pf[CUR], v_k);
            // Decode stages V here. Fused MTP delays it until after token 0's GEMM0
            // so that MFMA covers the wait that used to sit at the top of the body.
            if constexpr(T::V_VIA_LDS && !V_CHUNK_PIPELINE && !T::MTP_Q_LOOP)
                stage_v_to_lds<T>(s_v, lane_id, v_pf[CUR]);

            // Keep the block-table reads ahead of GEMM0: every KV address for the next
            // tile depends on them, so letting the scheduler sink them among the MFMAs
            // puts the first address computation right behind the load.
            if constexpr(GEMM0_SCHED == 0) __builtin_amdgcn_sched_barrier(0);

            if constexpr(T::MTP_Q_LOOP && !T::V_VIA_LDS && T::S_VIA_LDS)
            {
                // Direct-V fused: all QK, one max-fold, pack every P, one P
                // barrier, then all PV. S_VIA_LDS keeps one live C fragment.
                typename decltype(mma0)::vtype_c v_s;
                D_ACC wave_max[T::MTP_Q_MAX];
                opus::static_for<T::MTP_Q_MAX>([&](auto itok) {
                    constexpr int tok = itok.value;
                    if(tok >= kargs.qlen)
                    {
                        wave_max[tok] = opus::numeric_limits<D_ACC>::lowest();
                        return;
                    }
                    if constexpr(T::K16_PACK)
                        v_s = gemm0(v_q_tok[tok], v_k);
                    else
                        v_s = gemm0(v_q_tok[tok], k_pf[CUR]);
                    if constexpr(tok == 0)
                    {
                        load_cur_scales();
                        load_next_pid();
                        if constexpr(T::V_VIA_LDS && !V_CHUNK_PIPELINE)
                            stage_v_to_lds<T>(s_v, lane_id, v_pf[CUR]);
                        if constexpr(GEMM0_SCHED >= 2)
                        {
                            opus::static_for<T::GEMM0_E_N * T::GEMM0_E_K>([&](auto) {
                                if constexpr(T::K16_PACK)
                                    __builtin_amdgcn_sched_group_barrier(0x002, 6, 0);
                                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
                            });
                        }
                    }
                    if constexpr(T::PER_TOKEN_SCALE) mul_score_scales<T>(v_s, ksc);
                    wave_max[tok] = softmax_scale_mask_wave_max<T, MASKED>(
                        v_s, temperature_tok[tok], valid_tok[tok], tile_idx,
                        lane_id, warp_id);
                    if constexpr(T::S_VIA_LDS)
                        spill_score_lds<T>(s_s_raw, v_s, tok, 0, tid);
                });
                if constexpr(PREFETCH && !FRAGMENT_REFILL)
                    issue_k(k_pf[CUR], pid[CUR], tok_base_of(tile_idx + DEPTH));

                fold_across_waves_n<T, T::MTP_Q_MAX>(
                    s_row_reduce, wave_max, lane_id % T::W_M, warp_id,
                    [](D_ACC a, D_ACC b) { return a > b ? a : b; });

                opus::static_for<T::MTP_Q_MAX>([&](auto itok) {
                    constexpr int tok = itok.value;
                    if(tok >= kargs.qlen) return;
                    if constexpr(T::S_VIA_LDS)
                        fill_score_lds<T>(v_s, s_s_raw, tok, 0, tid);
                    const D_ACC rescale = softmax_apply_tile_max<T>(
                        v_s, wave_max[tok], m_tok[tok], l_tok[tok]);
                    opus::static_for<vector_traits<decltype(v_o_tok[tok])>::size()>(
                        [&](auto i) { v_o_tok[tok][i.value] *= rescale; });
                    if constexpr(T::PER_TOKEN_SCALE) mul_score_scales<T>(v_s, vsc);
                    auto v_p_out = cast<D_ATTN>(v_s);
                    auto s_p_tok = make_smem(s_p_raw + tok * T::Q_TILE * T::P_ROW_STRIDE);
                    store<4>(s_p_tok, v_p_out, u_p);
                });
                opus::s_waitcnt_lgkmcnt(opus::number<0>{});
                __builtin_amdgcn_s_barrier();

                opus::static_for<T::MTP_Q_MAX>([&](auto itok) {
                    constexpr int tok = itok.value;
                    if(tok >= kargs.qlen) return;
                    auto s_p_tok = make_smem(s_p_raw + tok * T::Q_TILE * T::P_ROW_STRIDE);
                    v_p = load<T::VEC>(s_p_tok, u_rp);
                    if constexpr(T::V_VIA_LDS)
                    {
                        read_v_frags<T>(s_v, lane_id, v_v);
                        v_o_tok[tok] = gemm1(v_v, v_o_tok[tok]);
                    }
                    else
                        v_o_tok[tok] = gemm1(v_pf[CUR], v_o_tok[tok]);
                });
                if constexpr(PREFETCH && !V_CHUNK_PIPELINE && !FRAGMENT_REFILL)
                    issue_v(v_pf[CUR], pid[CUR], tok_base_of(tile_idx + DEPTH));
            }
            else if constexpr(T::MTP_Q_LOOP && !T::V_VIA_LDS)
            {
                typename decltype(mma0)::vtype_c v_s_tok[T::MTP_Q_MAX];
                opus::static_for<T::MTP_Q_MAX>([&](auto itok) {
                    constexpr int tok = itok.value;
                    if(tok >= kargs.qlen) return;
                    if constexpr(T::K16_PACK)
                        v_s_tok[tok] = gemm0(v_q_tok[tok], v_k);
                    else
                        v_s_tok[tok] = gemm0(v_q_tok[tok], k_pf[CUR]);
                    if constexpr(tok == 0)
                    {
                        load_cur_scales();
                        load_next_pid();
                        if constexpr(GEMM0_SCHED >= 2)
                        {
                            opus::static_for<T::GEMM0_E_N * T::GEMM0_E_K>([&](auto) {
                                if constexpr(T::K16_PACK)
                                    __builtin_amdgcn_sched_group_barrier(0x002, 6, 0);
                                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
                            });
                        }
                    }
                });
                if constexpr(PREFETCH && !FRAGMENT_REFILL)
                    issue_k(k_pf[CUR], pid[CUR], tok_base_of(tile_idx + DEPTH));

                D_ACC wave_max[T::MTP_Q_MAX];
                opus::static_for<T::MTP_Q_MAX>([&](auto itok) {
                    constexpr int tok = itok.value;
                    if(tok >= kargs.qlen)
                    {
                        wave_max[tok] = opus::numeric_limits<D_ACC>::lowest();
                        return;
                    }
                    if constexpr(T::PER_TOKEN_SCALE) mul_score_scales<T>(v_s_tok[tok], ksc);
                    wave_max[tok] = softmax_scale_mask_wave_max<T, MASKED>(
                        v_s_tok[tok], temperature_tok[tok], valid_tok[tok], tile_idx,
                        lane_id, warp_id);
                });
                fold_across_waves_n<T, T::MTP_Q_MAX>(
                    s_row_reduce, wave_max, lane_id % T::W_M, warp_id,
                    [](D_ACC a, D_ACC b) { return a > b ? a : b; });

                opus::static_for<T::MTP_Q_MAX>([&](auto itok) {
                    constexpr int tok = itok.value;
                    if(tok >= kargs.qlen) return;
                    const D_ACC rescale = softmax_apply_tile_max<T>(
                        v_s_tok[tok], wave_max[tok], m_tok[tok], l_tok[tok]);
                    opus::static_for<vector_traits<decltype(v_o_tok[tok])>::size()>(
                        [&](auto i) { v_o_tok[tok][i.value] *= rescale; });
                    if constexpr(T::PER_TOKEN_SCALE) mul_score_scales<T>(v_s_tok[tok], vsc);
                    auto v_p_out = cast<D_ATTN>(v_s_tok[tok]);
                    auto s_p_tok = make_smem(s_p_raw + tok * T::Q_TILE * T::P_ROW_STRIDE);
                    store<4>(s_p_tok, v_p_out, u_p);
                });
                opus::s_waitcnt_lgkmcnt(opus::number<0>{});
                __builtin_amdgcn_s_barrier();

                opus::static_for<T::MTP_Q_MAX>([&](auto itok) {
                    constexpr int tok = itok.value;
                    if(tok >= kargs.qlen) return;
                    auto s_p_tok = make_smem(s_p_raw + tok * T::Q_TILE * T::P_ROW_STRIDE);
                    v_p = load<T::VEC>(s_p_tok, u_rp);
                    v_o_tok[tok] = gemm1(v_pf[CUR], v_o_tok[tok]);
                });
                if constexpr(PREFETCH && !V_CHUNK_PIPELINE && !FRAGMENT_REFILL)
                    issue_v(v_pf[CUR], pid[CUR], tok_base_of(tile_idx + DEPTH));
            }
            else if constexpr(T::MTP_Q_LOOP)
            {
                // LDS-V fused MTP: four live score tiles plus V staging does not fit
                // in 256 VGPR, so keep the per-token loop. K/V still reuse across tokens.
                opus::static_for<T::MTP_Q_MAX>([&](auto itok) {
                    constexpr int tok = itok.value;
                    if(tok >= kargs.qlen) return;
                    if constexpr(T::K16_PACK)
                        v_s = gemm0(v_q_tok[tok], v_k);
                    else
                        v_s = gemm0(v_q_tok[tok], k_pf[CUR]);
                    if constexpr(tok == 0)
                    {
                        load_cur_scales();
                        load_next_pid();
                        if constexpr(!V_CHUNK_PIPELINE)
                            stage_v_to_lds<T>(s_v, lane_id, v_pf[CUR]);
                        if constexpr(GEMM0_SCHED >= 2)
                        {
                            opus::static_for<T::GEMM0_E_N * T::GEMM0_E_K>([&](auto) {
                                if constexpr(T::K16_PACK)
                                    __builtin_amdgcn_sched_group_barrier(0x002, 6, 0);
                                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
                            });
                        }
                    }
                    if constexpr(PREFETCH && !FRAGMENT_REFILL)
                        if(tok + 1 == kargs.qlen)
                            issue_k(k_pf[CUR], pid[CUR], tok_base_of(tile_idx + DEPTH));

                    if constexpr(T::PER_TOKEN_SCALE) mul_score_scales<T>(v_s, ksc);
                    const D_ACC rescale = online_softmax_frag<T, MASKED>(
                        v_s, s_row_reduce, m_tok[tok], l_tok[tok], temperature_tok[tok],
                        valid_tok[tok], tile_idx, lane_id, warp_id);
                    opus::static_for<vector_traits<decltype(v_o_tok[tok])>::size()>(
                        [&](auto i) { v_o_tok[tok][i.value] *= rescale; });

                    if constexpr(T::PER_TOKEN_SCALE) mul_score_scales<T>(v_s, vsc);
                    auto v_p_out = cast<D_ATTN>(v_s);
                    store<4>(s_p, v_p_out, u_p);
                    opus::s_waitcnt_lgkmcnt(opus::number<0>{});
                    __builtin_amdgcn_s_barrier();
                    v_p = load<T::VEC>(s_p, u_rp);
                    read_v_frags<T>(s_v, lane_id, v_v);
                    v_o_tok[tok] = gemm1(v_v, v_o_tok[tok]);
                });
                if constexpr(PREFETCH && !V_CHUNK_PIPELINE && !FRAGMENT_REFILL)
                    issue_v(v_pf[CUR], pid[CUR], tok_base_of(tile_idx + DEPTH));
            }
            else if constexpr(FRAGMENT_REFILL)
            {
                static_assert(T::GEMM0_E_M == 1 && T::GEMM0_E_N == 2 && T::GEMM0_E_K == 1);
                using QK = typename decltype(mma0)::MMA;
                opus::static_for<T::GEMM0_E_N>([&](auto group) {
                    auto fragment = slice(k_pf[CUR], number<group.value * T::ELEM_B>{},
                                          number<(group.value + 1) * T::ELEM_B>{});
                    typename QK::vtype_c scores;
                    clear(scores);
                    scores = QK{}(v_q, fragment, scores, 0, 0);
                    set_slice(v_s, scores, number<group.value * T::ELEM_C>{},
                              number<(group.value + 1) * T::ELEM_C>{});
                    __builtin_amdgcn_sched_barrier(0);
                    if constexpr(PREFETCH)
                    {
                        const auto next_pages = k_pages_from<T>(pid[CUR], warp_id);
                        load_k_frags<T, group.value, 1>(p_k, next_pages, kargs.stride_k_blk,
                            kv_slice, tok_base_of(tile_idx + DEPTH), k_pf[CUR]);
                        __builtin_amdgcn_sched_barrier(0);
                    }
                });
            }
            else if constexpr(T::K16_PACK)
                v_s = gemm0(v_q, v_k);
            else
                v_s = gemm0(v_q, k_pf[CUR]);
            if constexpr(!T::MTP_Q_LOOP)
            {
            if constexpr(V_CHUNK_PIPELINE) __builtin_amdgcn_sched_barrier(0);
            if constexpr(GEMM0_SCHED >= 2)
            {
                // Claim GEMM0's MFMAs before anything else in the region, so the address
                // math that consumes the block table -- and with it the load's vmcnt --
                // sorts after them rather than between them. On the 16B path each pack's
                // register split is claimed just ahead of the MFMA that reads it, which
                // also puts MFMA issue latency over the permlane wait states.
                opus::static_for<T::GEMM0_E_N * T::GEMM0_E_K>([&](auto) {
                    if constexpr(T::K16_PACK)
                        __builtin_amdgcn_sched_group_barrier(0x002, 6, 0); // VALU
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                });
            }
            // Refill the buffer GEMM0 just drained, DEPTH tiles out.
            if constexpr(PREFETCH && !FRAGMENT_REFILL)
                issue_k(k_pf[CUR], pid[CUR], tok_base_of(tile_idx + DEPTH));

            if constexpr(V_CHUNK_PIPELINE)
            {
                static_assert(T::V_STAGE_ITERS == 4 && T::GEMM1_E_N == 2
                              && T::GEMM1_E_M == 1 && T::GEMM1_E_K == 1);
                stage_v_to_lds<T, 0, 2>(s_v, lane_id, v_pf[CUR]);
                if constexpr(PREFETCH)
                    load_v_stage<T, 0, 2>(p_v, pid[CUR], kargs.stride_v_blk,
                                          tok_base_of(tile_idx + 1), lane_id, warp_id, v_pf[CUR]);
            }

            // Softmax happens in place on v_s, so this is also the P the store below
            // hands to GEMM1. The one barrier inside is the cross-wave max fold.
            if constexpr(T::PER_TOKEN_SCALE) mul_score_scales<T>(v_s, ksc);
            const D_ACC rescale = online_softmax_frag<T, MASKED>(
                v_s, s_row_reduce, m_run, l_run, temperature_scale, valid_kv, tile_idx,
                lane_id, warp_id);
            opus::static_for<vector_traits<decltype(v_o)>::size()>(
                [&](auto i) { v_o[i.value] *= rescale; });

            if constexpr(T::PER_TOKEN_SCALE) mul_score_scales<T>(v_s, vsc);
            auto v_p_out = cast<D_ATTN>(v_s);
            store<4>(s_p, v_p_out, u_p);
            opus::s_waitcnt_lgkmcnt(opus::number<0>{});
            __builtin_amdgcn_s_barrier();

            v_p = load<T::VEC>(s_p, u_rp);
            if constexpr(FRAGMENT_REFILL)
            {
                static_assert(T::GEMM1_E_M == 1 && T::GEMM1_E_N == 2 && T::GEMM1_E_K == 1);
                using PV = typename decltype(mma1)::MMA;
                opus::static_for<T::GEMM1_E_N>([&](auto group) {
                    auto fragment = slice(v_pf[CUR], number<group.value * T::ELEM_B>{},
                                          number<(group.value + 1) * T::ELEM_B>{});
                    auto accumulator = slice(v_o, number<group.value * T::ELEM_C>{},
                                             number<(group.value + 1) * T::ELEM_C>{});
                    accumulator = PV{}(v_p, fragment, accumulator, 0, 0);
                    set_slice(v_o, accumulator, number<group.value * T::ELEM_C>{},
                              number<(group.value + 1) * T::ELEM_C>{});
                    __builtin_amdgcn_sched_barrier(0);
                    if constexpr(PREFETCH)
                    {
                        load_v_shuffled_frags<T, group.value, 1>(p_v, pid[CUR], kargs.stride_v_blk,
                            tok_base_of(tile_idx + DEPTH), lane_id, warp_id, v_pf[CUR]);
                        __builtin_amdgcn_sched_barrier(0);
                    }
                });
            }
            else if constexpr(V_CHUNK_PIPELINE)
            {
                using PV = typename decltype(mma1)::MMA;
                auto run_pv_group = [&](auto group) {
                    typename PV::vtype_b fragment;
                    read_v_frags_group<T, decltype(group)::value>(s_v, lane_id, fragment);
                    auto accumulator = slice(v_o, number<group.value * T::ELEM_C>{},
                                              number<(group.value + 1) * T::ELEM_C>{});
                    accumulator = PV{}(v_p, fragment, accumulator, 0, 0);
                    set_slice(v_o, accumulator, number<group.value * T::ELEM_C>{},
                              number<(group.value + 1) * T::ELEM_C>{});
                };
                stage_v_to_lds<T, 2, 2>(s_v, lane_id, v_pf[CUR]);
                if constexpr(PREFETCH)
                    load_v_stage<T, 2, 2>(p_v, pid[CUR], kargs.stride_v_blk,
                                          tok_base_of(tile_idx + 1), lane_id, warp_id, v_pf[CUR]);
                run_pv_group(0_I);
                __builtin_amdgcn_sched_barrier(0);
                opus::s_waitcnt_lgkmcnt(opus::number<0>{});
                __builtin_amdgcn_wave_barrier();
                run_pv_group(1_I);
            }
            else if constexpr(T::V_VIA_LDS)
            {
                // The P barrier above already drained LDS, so the staged rows are visible.
                read_v_frags<T>(s_v, lane_id, v_v);
                v_o = gemm1(v_v, v_o);
            }
            else
                v_o = gemm1(v_pf[CUR], v_o);
            if constexpr(PREFETCH && !V_CHUNK_PIPELINE && !FRAGMENT_REFILL)
            {
                // Safe to overwrite: the staged chunks went to LDS at the top of the body,
                // and on the direct path GEMM1 above has already read them.
                issue_v(v_pf[CUR], pid[CUR], tok_base_of(tile_idx + DEPTH));
            }
            }

            // No trailing barrier: the next tile cannot overwrite s_p or s_row_reduce
            // before its own max fold, by which point every wave has issued its P reads.
        };

        if constexpr(DEPTH == 1)
        {
            int tile_idx = 0;
            for (; tile_idx + 2 < num_tiles; ++tile_idx)
                run_tile(tile_idx, 0_I, 1_I, 0_I);
            if (num_tiles >= 2)
                run_tile(num_tiles - 2, MASK_PREV, 1_I, 0_I);
            run_tile(num_tiles - 1, MASK_LAST, 0_I, 0_I);
        }
        else
        {
            // Bodies go in pairs so the buffer index stays a compile-time constant -- a
            // runtime one would put the staging registers in scratch. Body t fetches tile
            // t + DEPTH, so the pair is safe while t + 2 + DEPTH <= num_tiles; that leaves
            // one, two or three tiles, on an even t, for the peeled tail below.
            int tile_idx = 0;
            for (; tile_idx + 3 < num_tiles; tile_idx += 2)
            {
                run_tile(tile_idx, 0_I, 1_I, 0_I);
                run_tile(tile_idx + 1, 0_I, 1_I, 1_I);
            }
            const int left = num_tiles - tile_idx;
            if (left == 3)
            {
                // Only the first of the three still has a tile left to fetch.
                run_tile(tile_idx, 0_I, 1_I, 0_I);
                run_tile(tile_idx + 1, MASK_PREV, 0_I, 1_I);
                run_tile(tile_idx + 2, MASK_LAST, 0_I, 0_I);
            }
            else if (left == 2)
            {
                run_tile(tile_idx, MASK_PREV, 0_I, 0_I);
                run_tile(tile_idx + 1, MASK_LAST, 0_I, 1_I);
            }
            else
            {
                run_tile(tile_idx, MASK_LAST, 0_I, 0_I);
            }
        }
    }

    // A row whose window closed before this run started has every score masked, so its
    // running max is still lowest() and exp2(lowest - lowest) came back as 1 rather
    // than underflowing to 0 the way a mask against a live row does. Drop the sum it
    // built: l == 0 is the identity partial, and it makes the row's output zero.
    if constexpr(T::HAS_MTP && !T::MTP_Q_LOOP)
        if(valid_kv <= 0) l_run = D_ACC(0.0f);
    if constexpr(T::MTP_Q_LOOP)
    {
        opus::static_for<T::MTP_Q_MAX>([&](auto itok) {
            constexpr int tok = itok.value;
            if(tok < kargs.qlen && valid_tok[tok] <= 0) l_tok[tok] = D_ACC(0.0f);
        });
    }

    constexpr bool EARLY_PARTIAL_O = PA_DECODE_OPUS_EARLY_PARTIAL_O && T::IS_FP8
                                    && !T::QUANT_Q && T::HAS_SINK && !T::HAS_MTP
                                    && T::D_HEAD == 128 && T::PAGE_SIZE == 128;
    auto publish_partial_o = [&]() {
        auto u_po = partition_layout_c(
            mma1, opus::make_tuple(number<T::D_HEAD>{}, 1_I),
            opus::make_tuple(0_I, lane_id % mma1.grpn_c, warp_id, lane_id / mma1.grpn_c));
        const int po_rows = (T::HAS_MTP && !T::MTP_Q_SPLIT && !T::MTP_Q_LOOP)
                                ? kargs.qlen * kargs.gqa_ratio
                                : kargs.gqa_ratio;
        auto g_po = make_gmem(partial_o_ptr<T>(kargs, scratch_b, kv_head_idx, slot),
                              po_rows * T::D_HEAD * sizeof(D_ACC));
        store<4>(g_po, v_o, u_po);
    };
    if constexpr(PUB == pa_publish::split_counter && EARLY_PARTIAL_O)
        if(kargs.num_splits > 1) publish_partial_o();

    auto fold_l = [&](D_ACC l) {
        return fold_across_waves<T>(
            s_row_reduce,
            wave_row_fold(l, [](D_ACC a, D_ACC b) { return a + b; }),
            lane_id % T::W_M,
            warp_id,
            [](D_ACC a, D_ACC b) { return a + b; });
    };

    if constexpr(T::MTP_Q_LOOP)
    {
        auto u_po = partition_layout_c(
            mma1, opus::make_tuple(number<T::D_HEAD>{}, 1_I),
            opus::make_tuple(0_I, lane_id % mma1.grpn_c, warp_id, lane_id / mma1.grpn_c));
        D_ACC l_final[T::MTP_Q_MAX];
        opus::static_for<T::MTP_Q_MAX>([&](auto itok) {
            constexpr int tok = itok.value;
            l_final[tok]      = tok < kargs.qlen
                               ? wave_row_fold(l_tok[tok],
                                               [](D_ACC a, D_ACC b) { return a + b; })
                               : D_ACC(0.0f);
        });
        fold_across_waves_n<T, T::MTP_Q_MAX>(
            s_row_reduce, l_final, lane_id % T::W_M, warp_id,
            [](D_ACC a, D_ACC b) { return a + b; });
        if constexpr(PUB == pa_publish::split_counter)
        {
            if(kargs.num_splits > 1)
            {
                opus::static_for<T::MTP_Q_MAX>([&](auto itok) {
                    constexpr int tok = itok.value;
                    if(tok >= kargs.qlen) return;
                    auto g_po = make_gmem(
                        partial_o_ptr<T>(kargs, scratch_b, kv_head_idx, slot, tok),
                        kargs.gqa_ratio * T::D_HEAD * sizeof(D_ACC));
                    store<4>(g_po, v_o_tok[tok], u_po);
                    if(tid < T::Q_TILE)
                    {
                        float* p_ml           = partial_ml_ptr<T>(kargs, scratch_b, kv_head_idx, slot, tok);
                        p_ml[tid]             = m_tok[tok];
                        p_ml[T::Q_TILE + tid] = l_final[tok];
                    }
                });
                __syncthreads();
                if constexpr(!T::SEPARATE_SPLIT_REDUCE)
                {
                    if(split_arrive_is_last<T>(kargs, scratch_b, kv_head_idx, tid))
                        reduce_splits_qpack<T>(kargs, scratch_b, kv_head_idx, tid, smem);
                }
                return;
            }
        }
        opus::static_for<T::MTP_Q_MAX>([&](auto itok) {
            constexpr int tok = itok.value;
            if(tok >= kargs.qlen) return;
            D_ACC o_scale = l_final[tok] > D_ACC(0.0f) ? D_ACC(1.0f) / l_final[tok]
                                                       : D_ACC(0.0f);
            if constexpr(T::IS_FP8) o_scale *= kargs.v_dequant;
            opus::static_for<vector_traits<decltype(v_o_tok[tok])>::size()>(
                [&](auto i) { v_o_tok[tok][i.value] *= o_scale; });
            const int64_t o_offset = static_cast<int64_t>(batch_idx) * kargs.stride_o_b
                                     + static_cast<int64_t>(tok) * kargs.stride_o_t
                                     + static_cast<int64_t>(q_head_base) * kargs.stride_o_h
                                     + static_cast<int64_t>(q_head) * kargs.stride_o_h;
            auto g_o = make_gmem(reinterpret_cast<D_OUT*>(kargs.out_ptr) + o_offset,
                                 T::D_HEAD * sizeof(D_OUT));
            auto v_o_out = cast<D_OUT>(v_o_tok[tok]);
            store<4>(g_o, v_o_out, u_o);
        });
        return;
    }
    else
    {

    // l is still a per-lane partial: a lane summed only its own quarter of the row,
    // and only its own wave's tokens. Both folds can wait until here because summing
    // is linear. Reusing s_row_reduce is safe -- the last tile read it before that
    // tile's P barrier.
    const D_ACC l_final = fold_l(l_run);

    if constexpr(PUB == pa_publish::split_counter)
    {
        if(kargs.num_splits > 1)
        {
            // Publish the unnormalized accumulator with its (m, l) so the reducer can
            // rescale across splits. Dead rows fall outside the bound -- which under
            // MTP has to cover every (token, head) pair, not just one token's heads.
            // The slot itself is Q_TILE rows wide either way, and the partial is
            // indexed by tile row, so nothing here needs the pair split apart.
            if constexpr(!EARLY_PARTIAL_O) publish_partial_o();

            // Lanes 0..Q_TILE-1 of wave 0 sit on query rows 0..Q_TILE-1, so their
            // register copies of m/l are exactly the per-row values to publish.
            if(tid < T::Q_TILE)
            {
                float* p_ml           = partial_ml_ptr<T>(kargs, scratch_b, kv_head_idx, slot);
                p_ml[tid]             = m_run;
                p_ml[T::Q_TILE + tid] = l_final;
            }

            if constexpr(!T::SEPARATE_SPLIT_REDUCE)
            {
                if(split_arrive_is_last<T>(kargs, scratch_b, kv_head_idx, tid))
                    reduce_splits_qpack<T>(kargs, scratch_b, kv_head_idx, tid, smem);
            }
            return;
        }
    }

    // ── Normalize and write back. Rows past the GQA group are dropped by the buffer bound. ──
    D_ACC o_scale = l_final > D_ACC(0.0f) ? D_ACC(1.0f) / l_final : D_ACC(0.0f);
    // v_dequant (v_scale) rides on the reciprocal, so GEMM1's fp8 units cost nothing
    // to undo. P's own 2^P_LOG2 is already in l_final and cancels here.
    if constexpr(T::IS_FP8) o_scale *= kargs.v_dequant;
    opus::static_for<vector_traits<decltype(v_o)>::size()>(
        [&](auto i) { v_o[i.value] *= o_scale; });

    if constexpr(PUB == pa_publish::split_lse)
    {
        // A split: hand the reduce kernel this run's own softmax result. It weights the
        // runs by exp(lse_i - lse_global), whose weights sum to 1, so O must already be
        // normalized -- which the multiply above just did.
        if(slot >= 0)
        {
            auto u_po = partition_layout_c(
                mma1, opus::make_tuple(number<T::D_HEAD>{}, 1_I),
                opus::make_tuple(0_I, lane_id % mma1.grpn_c, warp_id, lane_id / mma1.grpn_c));
            auto g_po = make_gmem(kargs.split_o
                                      + static_cast<int64_t>(slot) * kargs.num_heads * T::D_HEAD
                                      + static_cast<int64_t>(q_head_base) * T::D_HEAD,
                                  kargs.gqa_ratio * T::D_HEAD * sizeof(D_ACC));
            store<4>(g_po, v_o, u_po);

            // The reduce works in natural log. m_run is in log2 units because the
            // temperature carries log2(e), and l_final sums values the exp2 shifted up
            // by P_LOG2, so both corrections land in one expression.
            //
            // Bounded by the GQA group, not by Q_TILE: split_lse is indexed by global
            // q head, so a gqa_ratio below Q_TILE would otherwise let this wave's dead
            // rows write into the next kv head's slots. The O store above needs no such
            // guard -- its buffer bound already drops them.
            if(tid < kargs.gqa_ratio)
            {
                constexpr D_ACC LN_2 = 0.69314718055994531f;
                const D_ACC lse = l_final > D_ACC(0.0f)
                                      ? (m_run - D_ACC(T::P_LOG2) + log2f(l_final)) * LN_2
                                      : -opus::numeric_limits<D_ACC>::infinity();
                kargs.split_lse[static_cast<int64_t>(slot) * kargs.num_heads + q_head_base + tid] =
                    lse;
            }
            return;
        }
    }

    // Same per-lane base as the Q load, for the same reason, and the same bound: the
    // rows whose token ran past qlen have nowhere to land and are dropped here.
    int64_t o_offset = static_cast<int64_t>(batch_idx) * kargs.stride_o_b
                       + static_cast<int64_t>(q_head_base) * kargs.stride_o_h;
    size_t o_bytes = static_cast<size_t>(kargs.gqa_ratio) * kargs.stride_o_h * sizeof(D_OUT);
    if constexpr(T::HAS_MTP)
    {
        o_offset += static_cast<int64_t>(q_tok) * kargs.stride_o_t
                    + static_cast<int64_t>(q_head) * kargs.stride_o_h;
        o_bytes = q_tok < kargs.qlen ? T::D_HEAD * sizeof(D_OUT) : 0;
    }
    auto g_o = make_gmem(reinterpret_cast<D_OUT*>(kargs.out_ptr) + o_offset, o_bytes);
    auto v_o_out = cast<D_OUT>(v_o);
    store<4>(g_o, v_o_out, u_o);
    }
}

} // namespace pa_decode_16mx1_16nx4

// One workgroup per query row. The PA body at NP=256 already occupies every CU;
// fused last-arriver then parked 255 of them and merged 256 partials on one CTA.
template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 4) void
pa_decode_opus_split_reduce_kernel(pa_decode_kargs kargs)
{
    using namespace opus;
    using namespace pa_decode_16mx1_16nx4;
    using T     = opus::remove_cvref_t<Traits>;
    using D_ACC = typename T::D_ACC;
    using D_OUT = typename T::D_OUT;

    const int kvh     = block_id_x();
    const int y       = block_id_y();
    const int row_id  = block_id_z();
    const int tid     = thread_id_x();
    int b             = y;
    int tok_cta       = 0;
    if constexpr(T::MTP_Q_SPLIT)
    {
        tok_cta = y / kargs.batch;
        b       = y - tok_cta * kargs.batch;
    }
    const int scratch_b = T::MTP_Q_SPLIT ? b * kargs.qlen + tok_cta : b;
    const int gqa       = kargs.gqa_ratio;

    int tok_out = 0;
    int head    = row_id;
    int p_tok   = 0;
    int p_row   = row_id;
    if constexpr(T::MTP_Q_LOOP)
    {
        tok_out = row_id / gqa;
        head    = row_id - tok_out * gqa;
        p_tok   = tok_out;
        p_row   = head;
        if(tok_out >= kargs.qlen) return;
    }
    else if constexpr(T::MTP_Q_SPLIT)
    {
        tok_out = tok_cta;
        head    = row_id;
        p_tok   = 0;
        p_row   = row_id;
    }
    else if constexpr(T::HAS_MTP)
    {
        tok_out = row_id / gqa;
        head    = row_id - tok_out * gqa;
        p_tok   = 0;
        p_row   = row_id;
        if(tok_out >= kargs.qlen) return;
    }
    if(head >= gqa) return;

    const int splits = kargs.num_splits;
    __shared__ D_ACC s_m[T::MAX_SPLITS];
    __shared__ D_ACC s_l[T::MAX_SPLITS];
    __shared__ D_ACC s_inv;

    for(int i = tid; i < splits; i += T::BLOCK_SIZE)
    {
        const D_ACC* p_ml = partial_ml_ptr<T>(kargs, scratch_b, kvh, i, p_tok);
        s_m[i]            = p_ml[p_row];
        s_l[i]            = p_ml[T::Q_TILE + p_row];
    }
    __syncthreads();
    if(tid == 0)
    {
        D_ACC m_max = -opus::numeric_limits<D_ACC>::max();
        for(int i = 0; i < splits; ++i)
            if(s_m[i] > m_max) m_max = s_m[i];
        D_ACC l_sum = D_ACC(0);
        for(int i = 0; i < splits; ++i)
        {
            const D_ACC c = __builtin_amdgcn_exp2f(s_m[i] - m_max);
            s_m[i]        = c;
            l_sum += s_l[i] * c;
        }
        D_ACC inv = l_sum > D_ACC(0) ? D_ACC(1) / l_sum : D_ACC(0);
        if constexpr(T::IS_FP8) inv *= kargs.v_dequant;
        s_inv = inv;
    }
    __syncthreads();

    D_OUT* out = reinterpret_cast<D_OUT*>(kargs.out_ptr)
                 + static_cast<int64_t>(b) * kargs.stride_o_b
                 + static_cast<int64_t>(kvh * gqa + head) * kargs.stride_o_h;
    if constexpr(T::HAS_MTP)
        out += static_cast<int64_t>(tok_out) * kargs.stride_o_t;

    if(tid < T::D_HEAD)
    {
        D_ACC acc = D_ACC(0);
        for(int i = 0; i < splits; ++i)
        {
            const D_ACC* po = partial_o_ptr<T>(kargs, scratch_b, kvh, i, p_tok);
            acc += po[static_cast<int64_t>(p_row) * T::D_HEAD + tid] * s_m[i];
        }
        out[tid] = static_cast<D_OUT>(acc * s_inv);
    }
}

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 1) void pa_decode_opus_kernel(pa_decode_kargs kargs)
{
    using namespace opus;
    using namespace pa_decode_16mx1_16nx4;
    using T = opus::remove_cvref_t<Traits>;

    const int kv_head_idx = block_id_x();
    int batch_idx         = block_id_y();
    const int split_idx   = block_id_z();
    int q_tok_cta         = 0;
    if constexpr(T::MTP_Q_SPLIT)
    {
        q_tok_cta = batch_idx / kargs.batch;
        batch_idx = batch_idx - q_tok_cta * kargs.batch;
    }
    const int scratch_b = T::MTP_Q_SPLIT ? batch_idx * kargs.qlen + q_tok_cta : batch_idx;

    const int tid     = thread_id_x();
    const int lane_id = tid % T::WARP_SIZE;
    const int warp_id = __builtin_amdgcn_readfirstlane(tid / T::WARP_SIZE);

    // LDS layout: P | cross-wave reduction scratch. Everything else stays in
    // registers. Declared this early because the split-KV merge borrows the same
    // allocation, and an empty split reaches the merge without running the body.
    //
    // The padding is an experiment knob and is zero unless asked for. LDS is the one
    // resource that caps resident workgroups without touching register allocation or
    // the instruction schedule, which makes it the clean way to ask what this kernel
    // costs at lower occupancy: 160 KB per CU, so a total of 49152 holds it to three
    // workgroups and 65536 to two.
    __shared__ __align__(16) char smem[T::smem_size_bytes() + PA_DECODE_OPUS_LDS_PAD];

    // Split-KV: each workgroup owns a contiguous run of whole tiles. Tile-aligned
    // splits keep kv_start page-aligned, so the block table is simply re-based and
    // the rest of the kernel stays split-agnostic.
    const int context_len = kargs.context_lens[batch_idx];
    if constexpr(PA_DECODE_OPUS_A8W8_LONG_SPLITS && T::HAS_SINK && !T::HAS_MTP
                 && T::IS_FP8 && !T::QUANT_Q && T::D_HEAD == 128
                 && T::PAGE_SIZE == 128 && T::KV_TILE == 128)
    {
        if(kargs.short_seq_splits > 0 && context_len < 8192)
        {
            if(split_idx >= kargs.short_seq_splits) return;
            if(kargs.short_seq_splits > 1)
            {
                const int64_t group = static_cast<int64_t>(batch_idx) * kargs.num_kv_heads
                                      + kv_head_idx;
                const int64_t skipped_slots = group * (kargs.num_splits - kargs.short_seq_splits);
                kargs.partial_o += skipped_slots * T::Q_TILE * T::D_HEAD;
                kargs.partial_ml += skipped_slots * 2 * T::Q_TILE;
            }
            kargs.num_splits = kargs.short_seq_splits;
        }
    }
    const int tiles_total = (context_len + T::KV_TILE - 1) / T::KV_TILE;
    const int tiles_per_split = (tiles_total + kargs.num_splits - 1) / kargs.num_splits;
    const int tile_begin      = split_idx * tiles_per_split;
    const int tile_end        = min(tile_begin + tiles_per_split, tiles_total);

    if(tile_begin >= tile_end)
    {
        if(kargs.num_splits > 1)
        {
            // Empty split: publish an identity partial so the reduction can ignore it.
            const int ntok = T::MTP_Q_LOOP ? kargs.qlen : 1;
            if(tid < T::Q_TILE)
            {
                for(int tok = 0; tok < ntok; ++tok)
                {
                    float* p_ml           = partial_ml_ptr<T>(kargs, scratch_b, kv_head_idx, split_idx, tok);
                    p_ml[tid]             = -opus::numeric_limits<float>::max();
                    p_ml[T::Q_TILE + tid] = 0.0f;
                }
            }
            // Still has to arrive, or the last split waits on a count that never completes.
            if constexpr(!T::SEPARATE_SPLIT_REDUCE)
            {
                if(split_arrive_is_last<T>(kargs, scratch_b, kv_head_idx, tid))
                    reduce_splits_qpack<T>(kargs, scratch_b, kv_head_idx, tid, smem);
            }
        }
        return;
    }

    const int kv_start  = tile_begin * T::KV_TILE;
    const int split_len = min(context_len - kv_start, (tile_end - tile_begin) * T::KV_TILE);
    const int num_pages =
        (kv_start % T::PAGE_SIZE + split_len + T::PAGE_SIZE - 1) / T::PAGE_SIZE;

    pa_decode_opus_work<Traits, pa_publish::split_counter>(
        kargs,
        smem,
        batch_idx,
        kv_head_idx,
        kv_start / T::PAGE_SIZE,
        num_pages,
        split_len,
        kv_start,
        tile_end - tile_begin,
        tile_begin % T::TILES_PER_PAGE,
        split_idx,
        tile_begin == 0,
        tid,
        lane_id,
        warp_id,
        q_tok_cta);
}

// Persistent entry point: the grid is sized to the machine, not to the problem, and
// each block drains the work items the metadata kernel assigned it. That is what lets
// a batch of mixed context lengths balance -- the grid-per-split form has to pick one
// split count for the whole batch, sized for its longest row.
//
// The grid is 1-D on purpose. The kv head is carried in the work item's packed q-head
// range rather than a grid dim, because the metadata kernel wants every kv head of a
// request to share one split decision (see v1_2_pa_device.cuh).
//
// Decode only, so every request has one query token and the GQA group fits one M tile.
template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 1) void
pa_decode_opus_ps_kernel(pa_decode_kargs kargs)
{
    using namespace opus;
    using namespace pa_decode_16mx1_16nx4;
    using T = opus::remove_cvref_t<Traits>;

    const int tid     = thread_id_x();
    const int lane_id = tid % T::WARP_SIZE;
    const int warp_id = __builtin_amdgcn_readfirstlane(tid / T::WARP_SIZE);

    __shared__ __align__(16) char smem[T::smem_size_bytes()];

    const int work_id    = block_id_x();
    const int work_begin = kargs.work_indptr[work_id];
    const int work_end   = kargs.work_indptr[work_id + 1];

    for(int w = work_begin; w < work_end; ++w)
    {
        const int* item     = kargs.work_info + w * 8;
        const int batch_idx = item[0];
        const int slot      = item[1]; // < 0: this item owns the whole request
        const int kv_page_s = item[4]; // global index into kv_indices
        const int kv_page_e = item[5];
        // Low half of the packed range; the metadata emits one whole GQA group per item.
        const int kv_head_idx = (item[7] & 0xFFFF) / kargs.gqa_ratio;

        const int num_pages = kv_page_e - kv_page_s;
        if(num_pages <= 0)
            continue;

        // Only a request's very last page can be partial, so a run that ends before it
        // is exactly num_pages * PAGE_SIZE tokens and the min picks the right bound.
        const int context_len = kargs.context_lens[batch_idx];
        const int page_in_req = kv_page_s - kargs.kv_indptr[batch_idx];
        const int split_len =
            min(context_len - page_in_req * T::PAGE_SIZE, num_pages * T::PAGE_SIZE);

        pa_decode_opus_work<Traits, pa_publish::split_lse>(
            kargs,
            smem,
            batch_idx,
            kv_head_idx,
            kv_page_s,
            num_pages,
            split_len,
            page_in_req * T::PAGE_SIZE,
            // A run is a whole number of pages, which is several tiles when a page holds
            // more than one; the token count is the exact bound either way and drops the
            // tiles that a partial last page would leave empty.
            T::PAGE_HOLDS_TILE ? (split_len + T::KV_TILE - 1) / T::KV_TILE
                               : (num_pages + T::PAGES_PER_TILE - 1) / T::PAGES_PER_TILE,
            0, // work items start on a page boundary, so no phase
            slot,
            page_in_req == 0,
            tid,
            lane_id,
            warp_id,
            0);
    }
}

#else // !__gfx950__
template<class Traits>
__global__ void pa_decode_opus_kernel(pa_decode_kargs kargs) {}
template<class Traits>
__global__ void pa_decode_opus_split_reduce_kernel(pa_decode_kargs kargs) {}
template<class Traits>
__global__ void pa_decode_opus_ps_kernel(pa_decode_kargs kargs) {}
#endif

#else // !__HIP_DEVICE_COMPILE__
// Host pass only needs the launch stub; opus.hpp stays out of this translation pass.
template<class Traits>
__global__ void pa_decode_opus_kernel(pa_decode_kargs kargs) {}
template<class Traits>
__global__ void pa_decode_opus_split_reduce_kernel(pa_decode_kargs kargs) {}
template<class Traits>
__global__ void pa_decode_opus_ps_kernel(pa_decode_kargs kargs) {}
#endif // __HIP_DEVICE_COMPILE__

#endif // PA_DECODE_OPUS_IMPL
