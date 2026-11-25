// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// RDNA prefill attention for one sequence of M query tokens after a cached
// prefix, or a batch through plan_prefill's work list, causal and optionally
// windowed, over the paged KV cache
// (num_pages, NUM_KV_HEADS, PAGE_SIZE, 2 * HEAD_DIM), as a template on one
// entry of rdna_prefill_configs.h.
//
// PrefixLM: a token in a multimodal range [start, end] (absolute positions,
// inclusive; per sequence in a batch) also sees the range's later keys. The
// window still bounds its earlier keys, as TRITON_ATTN with
// mm_prefix_clamp_sliding_window.
//
// FlashAttention-2 on WMMA 16x16x16. One workgroup per (kv head, query tile,
// KV segment); row r of a kv head is token r / G, q head kvHead * G + r % G.
// Each wave owns 16 rows over part of the head dim; a step's key_tile keys are
// staged in LDS.
//
//   S^T[key][row] = K[key][:] . Q[row][:]        A = K tile, B = Q^T
//   O^T[d][row]  += V[key][d] . P[row][key]      A = V^T,    B = P^T
#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <utility>

#include "rdna_attn_common.cuh"
#include "rdna_attn_launch.h"
#include "rdna_prefill_configs.h"

namespace rdna {

// An RDNA WGP has 4 SIMDs and 128 KiB of LDS (64 KiB at most for one
// workgroup). A SIMD lane has 1536 VGPRs or 1024, depending on the arch.
constexpr int kSimdsPerWgp = 4;
constexpr int kWgpLdsBytes = 128 * 1024;
constexpr int kBigLaneVgprs = 1536;
constexpr int kSmallLaneVgprs = 1024;

// The largest divisor of n up to cap.
constexpr int largest_divisor_up_to(int n, int cap) {
  int g = cap < n ? cap : n;
  while (n % g) --g;
  return g;
}

// Internal linkage, as each build is instantiated in one translation unit: the
// kernel's LDS arrays are then internal and it compiles as a non-template
// kernel would.
namespace {

template <PrefillConfig C>
struct Prefill {
  static constexpr int kHeadDim = static_cast<int>(C.head_dim);
  static constexpr int kNumQHeads = C.num_q_heads;
  static constexpr int kNumKvHeads = C.num_kv_heads;
  static constexpr int kWindow = C.window;
  // Non-causal: every query token sees every key of its sequence.
  static constexpr bool kCausal = C.causal;
  static constexpr bool kBf16 = C.dtype == Dtype::kBf16;
  static constexpr int kPageSize = C.page_size;
  static constexpr int kWaves = C.waves;
  static constexpr int kKeyTile = C.key_tile;
  static constexpr int kDimParts = C.head_dim_split;
  // v_t: 0 stage V rows, 1 stage V transposed, 2 load the V^T operands with
  // GLOBAL_LOAD_TR (gfx12; on gfx11 as 1).
  static constexpr bool kVTr = kGfx12 && C.v_t == 2;
  static constexpr bool kVT = C.v_t == 1 || (C.v_t == 2 && !kVTr);
  static constexpr bool kDoubleBuffer = C.double_buffer;
  static constexpr int kKeyWaves = C.key_waves;
  static constexpr int kMaxSegments = C.max_segments;
  // 16-row tiles per wave: two share every K and V operand read (non-causal).
  static constexpr int kRowTiles = C.row_tiles;

  // Fewest steps per KV segment.
  static constexpr int kMinSegmentSteps = 16;

  static constexpr int kWaveSize = 32;
  static constexpr int kThreads = kWaves * kWaveSize;
  static constexpr int kG = kNumQHeads / kNumKvHeads;
  static constexpr int kRowWaves = kWaves / (kDimParts * kKeyWaves);
  static constexpr int kBlockRows = kRowWaves * 16 * kRowTiles;
  static constexpr int kSubTiles = kKeyTile / 16;
  static constexpr int kWaveSubTiles = kSubTiles / kKeyWaves;
  // The head dim padded to whole 16-dim chunks (D=72: 80). Q, the K tile and
  // V_T hold zeros past D; loads past D are clamped and stored as zeros.
  static constexpr int kHeadDimP = (kHeadDim + 15) / 16 * 16;
  static constexpr int kPartDims = kHeadDimP / kDimParts;
  static constexpr int kPartChunks = kPartDims / 16;
  // Consecutive dims of a V row per lane.
  static constexpr int kLaneDims = kPartDims / 16;
  static constexpr int kLaneWords = kLaneDims / 2;
  static constexpr int kKvRowElems = 2 * kHeadDim;
  static constexpr int kPageElems = kPageSize * kNumKvHeads * kKvRowElems;
  // LDS rows are padded by 16 bytes to avoid bank conflicts.
  static constexpr int kRowPitch = kHeadDimP + 8;
  // V_T row pitch, chosen to avoid bank conflicts: an odd number of 16-byte
  // units on gfx12, where a 6-unit pitch halves the LDS read rate.
  static constexpr int kVtPitch = kGfx12           ? kKeyTile + 8
                                  : kKeyTile == 16 ? 16
                                                   : kKeyTile + 16;
  // The V tile: V rows or V^T rows; a placeholder when v_t 2 skips it.
  static constexpr int kVTileRows = kVTr ? 1 : kVT ? kHeadDimP : kKeyTile;
  static constexpr int kVTilePitch = kVTr ? 8 : kVT ? kVtPitch : kRowPitch;
  static constexpr int kBuffers = kDoubleBuffer ? 2 : 1;
  // Elements per load unit: 8, or 4 where a 16-key sub-tile of 8-element
  // units would not be whole waves (D=48, 80). Non-causal builds load 8 of
  // the D real dims of K and V in place (the padding is zeroed once).
  static constexpr int kUnitElems = !kCausal || kHeadDimP / 8 % 4 == 0 ? 8 : 4;
  static constexpr int kRowChunks =
      (kCausal ? kHeadDimP : kHeadDim) / kUnitElems;
  static constexpr int kKUnits = kKeyTile * kRowChunks;
  // With V_T a V unit is keys kk and kk + 2 of a sub-tile.
  static constexpr int kVUnits = (kVT ? kKeyTile / 2 : kKeyTile) * kRowChunks;
  static constexpr int kKLoads = (kKUnits + kThreads - 1) / kThreads;
  static constexpr int kVLoads = (kVUnits + kThreads - 1) / kThreads;
  static constexpr int kVPerUnit = kVT ? 2 : 1;
  // V loads a step stages through registers into LDS.
  static constexpr int kVIssue = kVTr ? 0 : kVLoads;
  // Pages a step spans.
  static constexpr int kStepPages =
      kKeyTile > kPageSize ? kKeyTile / kPageSize : 1;
  static constexpr float kLog2e = 1.44269504088896340736f;
  static constexpr float kMaxSlack = 8.f;
  // Raw buffer word 3: OOB_SELECT 3, format 4. Bit 24 is reserved on gfx11
  // and enables write compression on gfx12.
  static constexpr unsigned kRsrcWord3 = kGfx12 ? 0x30004000u : 0x31004000u;

  static constexpr int kHeadGroup =
      largest_divisor_up_to(kNumKvHeads, C.head_group);

  static_assert(kNumQHeads % kNumKvHeads == 0, "GQA must be integral");
  static_assert(kCausal || kWindow == 0, "a non-causal build has no window");
  static_assert(kRowTiles == 1 || (kRowTiles == 2 && !kCausal &&
                                   kDimParts == 1 && kKeyWaves == 1),
                "two row tiles per wave: non-causal, one part, one key wave");
  // Not yet ported to gfx12's operand layout.
  static_assert(kRowTiles == 1 || !kGfx12, "two row tiles per wave: gfx11");
  // compute2 reads V^T from the staging buffer.
  static_assert(kRowTiles == 1 || kVT, "two row tiles per wave read V^T");
  static_assert(kCausal || !kVTr, "v_t 2 reads V from the paged cache");
  static_assert(kWaves % (kDimParts * kKeyWaves) == 0, "whole row tiles");
  static_assert(kHeadDimP % (16 * kDimParts) == 0, "whole chunks per part");
  static_assert(kHeadDimP == kHeadDim || (kDimParts == 1 && kVT),
                "a padded head dim is one part with V transposed");
  static_assert(!kVTr || kHeadDimP / 8 % 4 == 0,
                "v_t 2 loads whole 16-dim chunks of 8-element units");
  static_assert(kSubTiles % kKeyWaves == 0, "whole sub-tiles per key wave");
  static_assert(kVT || kLaneDims % 2 == 0,
                "a lane's V slice is whole dwords, or V is transposed");
  static_assert(kKeyTile % 16 == 0 && kPageSize % 16 == 0,
                "a 16-key sub-tile sits inside one page");
  static_assert(!kCausal || kRowChunks % 4 == 0,
                "a wave's keys of one load sit in one sub-tile, so one page");
  static_assert(kCausal || kHeadDim % 8 == 0,
                "non-causal: whole 8-element units");

  using elem_t = typename Elem<kBf16>::type;
  using e16 = typename Elem<kBf16>::v16;
  using e8 = typename Elem<kBf16>::v8;
  using op_t = typename Elem<kBf16>::op;
  using opw_t = typename Elem<kBf16>::opw;
  static constexpr int kOpWords = Elem<kBf16>::kOpWords;
  typedef elem_t e4 __attribute__((ext_vector_type(4)));
  // One load unit.
  using eu = std::conditional_t<kUnitElems == 8, e8, e4>;
  using uu = std::conditional_t<kUnitElems == 8, u4v, u2v>;

  // A buffer load's 8 elements as a unit (buffer loads are 8-element units).
  static __device__ __forceinline__ eu as_unit(e8 x) {
    if constexpr (kUnitElems == 8)
      return x;
    else
      return __builtin_shufflevector(x, x, 0, 1, 2, 3);
  }
  // The operand of a chunk with 8 real dims at `row` (D=72's last), zeros
  // after.
  static __device__ __forceinline__ op_t padded_op(const elem_t* row,
                                                   int laneHalf) {
#if defined(__GFX12__)
    return laneHalf ? op_t{} : *(const op_t*)row;
#else
    return __builtin_shufflevector(*(const e8*)row, e8{}, 0, 1, 2, 3, 4, 5, 6,
                                   7, 8, 9, 10, 11, 12, 13, 14, 15);
#endif
  }
  using e2 = typename Elem<kBf16>::v2;

  static __device__ __forceinline__ elem_t to_elem(float x) {
    return Elem<kBf16>::from_float(x);
  }

  static __device__ __forceinline__ f8 wmma(op_t a, op_t b, f8 c) {
    return Elem<kBf16>::wmma(a, b, c);
  }

  // The other half-wave's value, through LDS when the LDS is otherwise idle.
  static __device__ __forceinline__ unsigned xhalf_p(unsigned value) {
    if constexpr (kDimParts == 1)
      return (unsigned)__builtin_amdgcn_ds_bpermute(
          (int)(((threadIdx.x & 31) ^ 16) << 2), (int)value);
    else
      return xhalf_u(value);
  }

  static __device__ __forceinline__ unsigned pack2(float a, float b) {
    return __builtin_bit_cast(unsigned, e2{to_elem(a), to_elem(b)});
  }

  // B = P^T: lane l (row l & 15) holds P for keys 2e + (l >> 4) in probs[e].
  static __device__ __forceinline__ op_t p_frag(const float* probs) {
    return Elem<kBf16>::own_first_op(
        [&](int i) { return pack2(probs[2 * i], probs[2 * i + 1]); },
        [](unsigned x) { return xhalf_p(x); });
  }

  struct VRow {
    unsigned w[kLaneWords];
  };

  // A = V^T for element `dim` of this lane's slice, in p_frag's key order.
  static __device__ __forceinline__ op_t v_frag(const VRow* v, int dim) {
    const int word = dim >> 1;
    const unsigned selector = (dim & 1) ? 0x07060302u : 0x05040100u;
    return Elem<kBf16>::own_first_op(
        [&](int i) {
          return __builtin_amdgcn_perm(v[2 * i + 1].w[word], v[2 * i].w[word],
                                       selector);
        },
        [](unsigned x) { return xhalf_u(x); });
  }

  // V_T layout: the row of a dim, and the slot of a key (even keys first).
  static __device__ __forceinline__ int vt_row(int dim) {
    const int part = dim / kPartDims, local = dim % kPartDims;
    return part * kPartDims + (local % kLaneDims) * 16 + local / kLaneDims;
  }

  static __device__ __forceinline__ int vt_slot(int key) {
    return (key & 1) * 8 + (key >> 1);
  }

  // LDS per workgroup, and the waves per SIMD the register budget targets.
  static constexpr int kLdsBytes =
      kBuffers * kKeyTile * kRowPitch * sizeof(elem_t) +
      (kVTr ? 0 : kBuffers * kVTileRows * kVTilePitch * sizeof(elem_t)) +
      (kDimParts > 1 ? kWaves * kWaveSubTiles * 256 * 4 : 0) +
      (kKeyWaves > 1 ? (2 * kKeyWaves * kBlockRows + kWaves * kWaveSize * 8) * 4
                     : 0);
  static constexpr int kLdsBudget = std::max(kLdsBytes, 1024);
  // A workgroup's waves on each SIMD.
  static constexpr int kSimdWaves =
      kWaves >= kSimdsPerWgp ? kWaves / kSimdsPerWgp : 1;
  // With 1536 VGPRs: 6 waves a SIMD (256 VGPRs each, the most a wave
  // addresses), 8 for 16-wave workgroups, fewer if the LDS holds fewer
  // workgroups; in whole workgroups.
  static constexpr int kWavesPerSimd =
      std::min(kWaves >= 16 ? 8 : 6,
               kWgpLdsBytes / kLdsBudget * kWaves / kSimdsPerWgp) /
      kSimdWaves * kSimdWaves;
  // With 1024 VGPRs, the waves that keep each wave the VGPRs it has with
  // 1536, in whole workgroups.
  static constexpr int kSmallVgprWaves =
      kWavesPerSimd * kSmallLaneVgprs / kBigLaneVgprs;
  static constexpr int kSmallVgprWavesPerSimd =
      std::max(kSimdWaves, kSmallVgprWaves / kSimdWaves * kSimdWaves);
  // The archs with 1024 VGPRs.
#if defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || \
    defined(__gfx1152__) || defined(__gfx1153__)
  static constexpr int kTargetWavesPerSimd = kSmallVgprWavesPerSimd;
#else
  static constexpr int kTargetWavesPerSimd = kWavesPerSimd;
#endif

  // Each lane's VGPRs, and the granule a wave's are allocated in (wave32).
#if defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || \
    defined(__gfx1152__) || defined(__gfx1153__)
  static constexpr int kLaneVgprs = kSmallLaneVgprs;
  static constexpr int kVgprGranule = 16;
#else
  static constexpr int kLaneVgprs = kBigLaneVgprs;
  static constexpr int kVgprGranule = 24;
#endif
  // The VGPRs a wave gets at `waves` waves a SIMD: 240 at 6 with 1536.
  static constexpr int vgpr_budget(int waves) {
    return std::min(256, kLaneVgprs / std::max(waves, 1) / kVgprGranule *
                             kVgprGranule);
  }
  // The VGPRs a wave holds through a step, estimated: Q and O, a step's
  // scores and P, and the next step's staged K and V units with a 64-bit
  // address each. The compiler adds 20-110 (counters, masks, LDS addresses,
  // operands read ahead): a build within 50 of its budget is one that spills.
  static constexpr int kStagedUnits = kKLoads + kVIssue * kVPerUnit;
  static constexpr int kVgprEstimate =
      (kPartChunks * kOpWords + kLaneDims * 8 +
       kWaveSubTiles * (8 + kOpWords)) *
          kRowTiles +
      kStagedUnits * (kUnitElems / 2 + 2);
  static constexpr int kVgprMargin =
      kVgprEstimate - vgpr_budget(kTargetWavesPerSimd);
  // Buffer loads (32-bit offsets, the base in SGPRs) save the registers of
  // 64-bit addresses: non-causal builds, D=512, and the causal builds near
  // their budget at 4 or more waves a SIMD.
  static constexpr bool kBufferLoads =
      !kCausal ||
      (kUnitElems == 8 &&
       (kHeadDim >= 512 || (kTargetWavesPerSimd >= 4 && kVgprMargin >= -50)));
  // D=512 and non-causal builds keep the offsets in registers; the others
  // let the compiler recompute them.
  static constexpr bool kPinOffsets = !kCausal || kHeadDim >= 512;

  // kBatch: a batch through the work list, apart so that the one-sequence
  // kernel keeps its registers.
  template <bool kBatch>
  static __global__ void __launch_bounds__(kThreads) __attribute__((
      amdgpu_waves_per_eu(kTargetWavesPerSimd > 1 ? kTargetWavesPerSimd : 1)))
  prefill_attn(const elem_t* __restrict__ q, const elem_t* __restrict__ kv,
               const int* __restrict__ blockTable, elem_t* __restrict__ out,
               const int* __restrict__ seqLens, int numTokens, float scale,
               float* __restrict__ partialO, float* __restrict__ partialML,
               int* __restrict__ counters, int numSegments, int firstRound,
               const int* __restrict__ mmRanges, int numMmRanges,
               const int4* __restrict__ plan,
               const int* __restrict__ queryStart, int numEntries,
               int blockTableStride, int headGroup,
               const elem_t* __restrict__ vIn, int qStride, int kStride,
               int vStride) {
    __shared__ __attribute__((
        aligned(16))) elem_t kTile[kBuffers][kKeyTile][kRowPitch];
    __shared__ __attribute__((
        aligned(16))) elem_t vTile[kBuffers][kVTileRows][kVTilePitch];
    __shared__ __attribute__((aligned(
        16))) float scoreX[kDimParts > 1 ? kWaves * kWaveSubTiles * 256 : 1];

    const int thread = threadIdx.x;
    const int lane = thread & (kWaveSize - 1);
    const int wave = __builtin_amdgcn_readfirstlane(thread / kWaveSize);
    const int laneRow = lane & 15;
    const int laneHalf = lane >> 4;
    const int dimPart = wave % kDimParts;
    const int keyWave = wave / kDimParts % kKeyWaves;
    const int rowWave = wave / (kDimParts * kKeyWaves);
    const int firstChunk = dimPart * kPartChunks;

    int segment, tile, kvHead;
    // Split KV: this tile's counter and first partials slot.
    int counter;
    size_t groupSlot;
    if constexpr (kBatch) {
      // The work list entry: sequence, tile and segment.
      const int groupBlocks = headGroup * numEntries;
      const int within = (int)blockIdx.x % groupBlocks;
      kvHead = (int)blockIdx.x / groupBlocks * headGroup + within % headGroup;
      const int4 entry = plan[within / headGroup];
      const int seq = entry.x;
      tile = entry.y & 0xffff;
      segment = entry.z & 0xffff;
      numSegments = entry.z >> 16;
      counter = (entry.y >> 16) * kNumKvHeads + kvHead;
      groupSlot = (size_t)entry.w * kNumKvHeads + kvHead * numSegments;
      const int queryBegin = queryStart[seq];
      numTokens = queryStart[seq + 1] - queryBegin;
      out += (size_t)queryBegin * kNumQHeads * kHeadDim;
      if constexpr (kCausal) {
        q += (size_t)queryBegin * kNumQHeads * kHeadDim;
        blockTable += (size_t)seq * blockTableStride;
        mmRanges += (size_t)seq * 2 * numMmRanges;
      } else {
        // The sequence's K and V start with its queries.
        q += (size_t)queryBegin * qStride;
        kv += (size_t)queryBegin * kStride;
        vIn += (size_t)queryBegin * vStride;
      }
      seqLens += seq;
    } else {
      // A tile's segments are adjacent; the heaviest tiles (last tokens) go
      // first, interleaving kHeadGroup kv heads.
      const int numTiles = (numTokens * kG + kBlockRows - 1) / kBlockRows;
      segment = (int)blockIdx.x % numSegments;
      const int block = (int)blockIdx.x / numSegments;
      const int groupBlocks = kHeadGroup * numTiles;
      const int within = block % groupBlocks;
      tile = numTiles - 1 - within / kHeadGroup;
      kvHead = block / groupBlocks * kHeadGroup + within % kHeadGroup;
      counter = block;
      groupSlot = (size_t)block * numSegments;
    }
    const int numRows = numTokens * kG;

    const int seqLen = __builtin_amdgcn_readfirstlane(*seqLens);
    const int context = seqLen - numTokens;
    const int tileRow = tile * kBlockRows;
    const int firstToken = tileRow / kG;
    const int lastToken = min(numTokens - 1, (tileRow + kBlockRows - 1) / kG);
    // The last key any of tokens first..last sees: the last token, or the end
    // of a multimodal range they reach into.
    auto last_key = [&](int first, int last) __attribute__((always_inline)) {
      if constexpr (!kCausal) return seqLen - 1;
      int key = context + last;
      if (numMmRanges == 0) return key;
      for (int r = 0; r < numMmRanges; ++r) {
        const int start = mmRanges[2 * r], end = mmRanges[2 * r + 1];
        if (start < end && start <= context + last && end >= context + first)
          key = max(key, end);
      }
      return min(key, seqLen - 1);
    };
    // Keys from the first token's window to the last key.
    const int keyEnd = last_key(firstToken, lastToken) + 1;
    const int windowFirst =
        kWindow ? max(0, context + firstToken - (kWindow - 1)) : 0;
    const int tileKeyBegin = windowFirst / kKeyTile * kKeyTile;
    // This segment's steps, possibly none.
    const int numSteps = (keyEnd - tileKeyBegin + kKeyTile - 1) / kKeyTile;
    const int keyBegin =
        tileKeyBegin + segment * numSteps / numSegments * kKeyTile;
    const int stepsEnd = min(keyEnd, tileKeyBegin + (segment + 1) * numSteps /
                                                        numSegments * kKeyTile);
    const int lastPage = (seqLen - 1) / kPageSize;

    const int waveRow = tileRow + rowWave * 16 * kRowTiles;
    const int row = min(waveRow + laneRow, numRows - 1);
    const int rowToken = row / kG;
    const int rowHead = kvHead * kG + row % kG;
    const int waveFirstToken = waveRow / kG;
    const int waveLastToken = min(numTokens - 1, (waveRow + 15) / kG);
    const int waveLastKey = last_key(waveFirstToken, waveLastToken);
    const int rowLastKey = last_key(rowToken, rowToken);

    // Stagger the Q loads of the first resident workgroups so the heaviest
    // tiles start first.
    {
      constexpr int kStaggerUnits = std::max(
          1, kBlockRows * kHeadDim * (int)sizeof(elem_t) * 3 / (2 * 8192));
      const int place = __builtin_amdgcn_readfirstlane((int)blockIdx.x);
      // Only when tile lengths differ and the KV is not split.
      const bool unequal =
          kCausal && numSegments == 1 &&
          __builtin_amdgcn_readfirstlane(*seqLens) <= 2 * numTokens;
      if (unequal && place < firstRound)
        for (int i = 0; i < place * kStaggerUnits; ++i)
          __builtin_amdgcn_s_sleep(1);
    }

    // Lane l holds page l % kStepPages of a step, fetched a step ahead.
    // Non-causal builds read K and V in place, without pages.
    auto step_pages = [&](int keyStart) __attribute__((always_inline)) {
      if constexpr (!kCausal) return 0;
      return blockTable[min(keyStart / kPageSize + lane % kStepPages,
                            lastPage)];
    };
    int pagesNext = step_pages(keyBegin);

    op_t qOperand[kPartChunks];
#pragma unroll
    for (int c = 0; c < kPartChunks; ++c) {
      const elem_t* qRow =
          (kCausal ? q + ((size_t)rowToken * kNumQHeads + rowHead) * kHeadDim
                   : q + (size_t)rowToken * qStride + rowHead * kHeadDim) +
          (firstChunk + c) * 16;
      if (kHeadDimP != kHeadDim && (firstChunk + c) * 16 + 16 > kHeadDim) {
        // The padded chunk: 8 real dims (D=72), zeros after.
        qOperand[c] = padded_op(qRow, laneHalf);
      } else {
        qOperand[c] = Elem<kBf16>::load_op(qRow, laneHalf);
      }
    }

    // The second row tile's rows and Q (kRowTiles == 2).
    const int row1 = min(waveRow + 16 + laneRow, numRows - 1);
    op_t qOperand1[kRowTiles > 1 ? kPartChunks : 1];
    if constexpr (kRowTiles > 1) {
#pragma unroll
      for (int c = 0; c < kPartChunks; ++c) {
        const elem_t* qRow = q + (size_t)(row1 / kG) * qStride +
                             (kvHead * kG + row1 % kG) * kHeadDim + c * 16;
        qOperand1[c] = kHeadDimP != kHeadDim && c * 16 + 16 > kHeadDim
                           ? padded_op(qRow, laneHalf)
                           : Elem<kBf16>::load_op(qRow, laneHalf);
      }
    }

    if constexpr (!kCausal && kHeadDimP != kHeadDim) {
      // Staging never writes the padding: zero it once.
      static_assert(kHeadDimP - kHeadDim == kUnitElems);
      for (int i = thread; i < kBuffers * kKeyTile; i += kThreads)
        *(eu*)&kTile[i / kKeyTile][i % kKeyTile][kHeadDim] = eu{};
      for (int i = thread; i < kBuffers * kUnitElems * kKeyTile; i += kThreads)
        vTile[i / (kUnitElems * kKeyTile)]
             [vt_row(kHeadDim + i / kKeyTile % kUnitElems)][i % kKeyTile] =
                 elem_t{};
    }

    const elem_t* kvHeadBase =
        kCausal ? kv + (size_t)kvHead * (kPageSize * kKvRowElems)
                : kv + kvHead * kHeadDim;
    const elem_t* vHeadBase = kCausal ? nullptr : vIn + kvHead * kHeadDim;
    eu kStaged[kKLoads];
    eu vStaged[kVLoads][kVPerUnit];
    auto v_unit_pair = [&](int unit) __attribute__((always_inline)) {
      if constexpr (kVT)
        return unit / (8 * kRowChunks) * 8 + unit % 8;
      else
        return unit / kRowChunks;
    };
    auto v_unit_col = [&](int unit) __attribute__((always_inline)) {
      if constexpr (kVT)
        return unit / 8 % kRowChunks * kUnitElems;
      else
        return unit % kRowChunks * kUnitElems;
    };
    auto v_unit_key = [&](int unit) __attribute__((always_inline)) {
      const int pair = v_unit_pair(unit);
      if constexpr (kVT)
        return pair / 8 * 16 + (pair % 8 >> 1) * 4 + (pair & 1);
      else
        return pair;
    };
    // Steps with keys past S or before the window (possibly freed pages) clamp
    // their loads and store zeros, so no 0 * NaN reaches P@V.
    auto edge_step = [&](int keyStart) __attribute__((always_inline)) {
      return keyStart + kKeyTile > seqLen || keyStart < windowFirst;
    };
    // A wave's keys of one load sit in one page.
    auto load_row = [&]<bool kEdge>(int pages, int keyStart, int key,
                                    int col) __attribute__((always_inline)) {
      if constexpr (!kEdge) {
        const int first = keyStart % kPageSize;
        const int slot =
            (first + __builtin_amdgcn_readfirstlane(key)) / kPageSize;
        const elem_t* base =
            kvHeadBase +
            (size_t)__builtin_amdgcn_readlane(pages, slot) * kPageElems +
            (first - slot * kPageSize) * kKvRowElems;
        return *(const eu*)(base + (unsigned)(key * kKvRowElems + col));
      }
      const int position = min(keyStart + key, seqLen - 1);
      const int slot = __builtin_amdgcn_readfirstlane(position / kPageSize -
                                                      keyStart / kPageSize);
      const elem_t* page =
          kvHeadBase +
          (size_t)__builtin_amdgcn_readlane(pages, slot) * kPageElems;
      return *(
          const eu*)(page +
                     (unsigned)((position % kPageSize) * kKvRowElems + col));
    };
    // Per-lane byte offsets of the buffer loads, kept in registers: from the
    // wave's first key, or (non-causal) the step's.
    int kOff[kKLoads], vOff[kVLoads][kVPerUnit];
    if constexpr (!kCausal) {
#pragma unroll
      for (int i = 0; i < kKLoads; ++i) {
        const int unit = i * kThreads + thread;
        kOff[i] =
            (unit / kRowChunks * kStride + unit % kRowChunks * kUnitElems) *
            (int)sizeof(elem_t);
        asm volatile("" : "+v"(kOff[i]));
      }
#pragma unroll
      for (int i = 0; i < kVLoads; ++i) {
        const int unit = i * kThreads + thread;
#pragma unroll
        for (int x = 0; x < kVPerUnit; ++x) {
          vOff[i][x] =
              ((v_unit_key(unit) + 2 * x) * vStride + v_unit_col(unit)) *
              (int)sizeof(elem_t);
          asm volatile("" : "+v"(vOff[i][x]));
        }
      }
    } else if constexpr (kBufferLoads) {
#pragma unroll
      for (int i = 0; i < kKLoads; ++i) {
        const int unit = i * kThreads + thread;
        const int waveKey = (i * kThreads + wave * kWaveSize) / kRowChunks;
        kOff[i] = ((unit / kRowChunks - waveKey) * kKvRowElems +
                   unit % kRowChunks * 8) *
                  (int)sizeof(elem_t);
        if constexpr (kPinOffsets) asm volatile("" : "+v"(kOff[i]));
      }
#pragma unroll
      for (int i = 0; i < kVLoads; ++i) {
        const int unit = i * kThreads + thread;
        const int waveKey = v_unit_key(i * kThreads + wave * kWaveSize);
#pragma unroll
        for (int x = 0; x < kVPerUnit; ++x) {
          vOff[i][x] = ((v_unit_key(unit) - waveKey) * kKvRowElems + kHeadDim +
                        v_unit_col(unit)) *
                       (int)sizeof(elem_t);
          if constexpr (kPinOffsets) asm volatile("" : "+v"(vOff[i][x]));
        }
      }
    }
    // A buffer load relative to the wave's first key `waveKey`.
    auto load_buffer = [&](int pages, int keyStart, int waveKey,
                           int offset) __attribute__((always_inline)) {
      const int first = keyStart % kPageSize;
      const int slot = (first + waveKey) / kPageSize;
      const elem_t* base =
          kvHeadBase +
          (size_t)__builtin_amdgcn_readlane(pages, slot) * kPageElems +
          (first - slot * kPageSize + waveKey) * kKvRowElems;
      const __amdgpu_buffer_rsrc_t rsrc = __builtin_amdgcn_make_buffer_rsrc(
          (void*)base, 0, 0x7fffffff, kRsrcWord3);
      return __builtin_bit_cast(
          e8, __builtin_amdgcn_raw_buffer_load_b128(rsrc, offset, 0, 0));
    };
    // Non-causal: the step's units from its first key, each range ending
    // with its last key's row so later keys load as zeros.
    auto issue_noncausal = [&](int keyStart) __attribute__((always_inline)) {
      const int last = min(seqLen - keyStart, kKeyTile) - 1;
      const __amdgpu_buffer_rsrc_t kRsrc = __builtin_amdgcn_make_buffer_rsrc(
          (void*)(kvHeadBase + (size_t)keyStart * kStride), 0,
          (last * kStride + kHeadDim) * (int)sizeof(elem_t), kRsrcWord3);
      const __amdgpu_buffer_rsrc_t vRsrc = __builtin_amdgcn_make_buffer_rsrc(
          (void*)(vHeadBase + (size_t)keyStart * vStride), 0,
          (last * vStride + kHeadDim) * (int)sizeof(elem_t), kRsrcWord3);
#pragma unroll
      for (int i = 0; i < kKLoads; ++i)
        if (kKUnits % kThreads == 0 || i * kThreads + thread < kKUnits)
          kStaged[i] = as_unit(__builtin_bit_cast(
              e8, __builtin_amdgcn_raw_buffer_load_b128(kRsrc, kOff[i], 0, 0)));
#pragma unroll
      for (int i = 0; i < kVLoads; ++i)
        if (kVUnits % kThreads == 0 || i * kThreads + thread < kVUnits)
#pragma unroll
          for (int x = 0; x < kVPerUnit; ++x)
            vStaged[i][x] = as_unit(
                __builtin_bit_cast(e8, __builtin_amdgcn_raw_buffer_load_b128(
                                           vRsrc, vOff[i][x], 0, 0)));
    };
    // Load the step at keyStart into registers.
    auto issue_step =
        [&]<bool kEdge>(int keyStart) __attribute__((always_inline)) {
          const int pages = pagesNext;
          pagesNext = step_pages(keyStart + kKeyTile);
          if constexpr (kBufferLoads && !kEdge) {
#pragma unroll
            for (int i = 0; i < kKLoads; ++i)
              if (kKUnits % kThreads == 0 || i * kThreads + thread < kKUnits)
                kStaged[i] = as_unit(load_buffer(
                    pages, keyStart,
                    (i * kThreads + wave * kWaveSize) / kRowChunks, kOff[i]));
#pragma unroll
            for (int i = 0; i < kVIssue; ++i)
              if (kVUnits % kThreads == 0 || i * kThreads + thread < kVUnits) {
                const int waveKey = v_unit_key(i * kThreads + wave * kWaveSize);
#pragma unroll
                for (int x = 0; x < kVPerUnit; ++x)
                  vStaged[i][x] = as_unit(load_buffer(
                      pages, keyStart, waveKey + 2 * x, vOff[i][x]));
              }
            return;
          }
#pragma unroll
          for (int i = 0; i < kKLoads; ++i) {
            const int unit = i * kThreads + thread;
            if (kKUnits % kThreads == 0 || unit < kKUnits)
              kStaged[i] = load_row.template operator()<kEdge>(
                  pages, keyStart, unit / kRowChunks,
                  min(unit % kRowChunks * kUnitElems, kHeadDim - kUnitElems));
          }
#pragma unroll
          for (int i = 0; i < kVIssue; ++i) {
            const int unit = i * kThreads + thread;
            if (kVUnits % kThreads == 0 || unit < kVUnits) {
              const int key = v_unit_key(unit), col = v_unit_col(unit);
#pragma unroll
              for (int x = 0; x < kVPerUnit; ++x)
                vStaged[i][x] = load_row.template operator()<kEdge>(
                    pages, keyStart, key + 2 * x,
                    kHeadDim + min(col, kHeadDim - kUnitElems));
            }
          }
        };
    auto valid = [&](int position) __attribute__((always_inline)) {
      return position < seqLen && position >= windowFirst;
    };
    // Store the loaded step to LDS buffer `buffer`.
    auto stage_step = [&]<bool kEdge, bool kDoK = true, bool kDoV = true>(
                          int buffer,
                          int keyStart) __attribute__((always_inline)) {
#pragma unroll
      for (int i = 0; i < (kDoK ? kKLoads : 0); ++i) {
        const int unit = i * kThreads + thread;
        if (kKUnits % kThreads == 0 || unit < kKUnits) {
          const int key = unit / kRowChunks,
                    col = unit % kRowChunks * kUnitElems;
          *(eu*)&kTile[buffer][key][col] =
              (!kEdge || valid(keyStart + key)) && col < kHeadDim ? kStaged[i]
                                                                  : eu{};
        }
      }
#pragma unroll
      for (int i = 0; i < (kDoV ? kVIssue : 0); ++i) {
        const int unit = i * kThreads + thread;
        if (kVUnits % kThreads == 0 || unit < kVUnits) {
          const int key = v_unit_key(unit), col = v_unit_col(unit);
          if constexpr (kVT) {
            const eu a = (!kEdge || valid(keyStart + key)) && col < kHeadDim
                             ? vStaged[i][0]
                             : eu{};
            const eu b = (!kEdge || valid(keyStart + key + 2)) && col < kHeadDim
                             ? vStaged[i][1]
                             : eu{};
            const uu aw = __builtin_bit_cast(uu, a),
                     bw = __builtin_bit_cast(uu, b);
            const int slot = key / 16 * 16 + vt_slot(key % 16);
#pragma unroll
            for (int d = 0; d < kUnitElems; ++d)
              *(unsigned*)&vTile[buffer][vt_row(col + d)][slot] =
                  __builtin_amdgcn_perm(bw[d >> 1], aw[d >> 1],
                                        (d & 1) ? 0x07060302u : 0x05040100u);
          } else {
            *(eu*)&vTile[buffer][key][col] =
                !kEdge || valid(keyStart + key) ? vStaged[i][0] : eu{};
          }
        }
      }
    };
    auto issue = [&](int keyStart) __attribute__((always_inline)) {
      if constexpr (!kCausal)
        issue_noncausal(keyStart);
      else if (edge_step(keyStart))
        issue_step.template operator()<true>(keyStart);
      else
        issue_step.template operator()<false>(keyStart);
    };
    // Non-causal steps past S loaded zeros: no masking.
    auto stage = [&](int buffer, int keyStart) __attribute__((always_inline)) {
      if (kCausal && edge_step(keyStart))
        stage_step.template operator()<true>(buffer, keyStart);
      else
        stage_step.template operator()<false>(buffer, keyStart);
    };
    auto stage_k = [&](int buffer,
                       int keyStart) __attribute__((always_inline)) {
      if (kCausal && edge_step(keyStart))
        stage_step.template operator()<true, true, false>(buffer, keyStart);
      else
        stage_step.template operator()<false, true, false>(buffer, keyStart);
    };
    auto stage_v = [&](int buffer,
                       int keyStart) __attribute__((always_inline)) {
      if (kCausal && edge_step(keyStart))
        stage_step.template operator()<true, false, true>(buffer, keyStart);
      else
        stage_step.template operator()<false, false, true>(buffer, keyStart);
    };
    // One buffer with a head-dim split: stage the next K after the score
    // exchange and the next V after the step, saving a barrier.
    constexpr bool kSplitStage = !kDoubleBuffer && kDimParts > 1;

    const float scaleLog2 = scale * kLog2e;
    float runMax = -INFINITY, runSum = 0.f;
    f8 accum[kLaneDims];
#pragma unroll
    for (int j = 0; j < kLaneDims; ++j) accum[j] = f8{};
    float runMax1 = -INFINITY, runSum1 = 0.f;
    f8 accum1[kRowTiles > 1 ? kLaneDims : 1];
#pragma unroll
    for (int j = 0; j < (kRowTiles > 1 ? kLaneDims : 1); ++j) accum1[j] = f8{};

    // Q@K, softmax update and P@V for the step at keyStart in `buffer`. Lane
    // holds row laneRow, keys keyStart + 16 st + 2e + laneHalf.
    // Online softmax of one row tile's scores: rescales its O, returns P.
    auto softmax = [&](f8* scores, float& rowMax, float& rowSum, f8* acc,
                       op_t* p) __attribute__((always_inline)) {
      float tileMax = -INFINITY;
#pragma unroll
      for (int ls = 0; ls < kWaveSubTiles; ++ls)
#pragma unroll
        for (int e = 0; e < 8; ++e) tileMax = fmaxf(tileMax, scores[ls][e]);
      tileMax = fmaxf(tileMax, xhalf(tileMax)) * scaleLog2;
      const float newMax = tileMax > rowMax + kMaxSlack ? tileMax : rowMax;
      const float safeMax = newMax == -INFINITY ? 0.f : newMax;
      const float rescale = __builtin_amdgcn_exp2f(rowMax - safeMax);
      rowMax = newMax;
      if (__builtin_amdgcn_ballot_w32(rescale != 1.f))
#pragma unroll
        for (int j = 0; j < kLaneDims; ++j) acc[j] *= rescale;
      float tileSum = 0.f;
#pragma unroll
      for (int ls = 0; ls < kWaveSubTiles; ++ls) {
        float probs[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          probs[e] = __builtin_amdgcn_exp2f(
              __builtin_fmaf(scores[ls][e], scaleLog2, -safeMax));
          tileSum += probs[e];
        }
        p[ls] = p_frag(probs);
      }
      rowSum = rowSum * rescale + tileSum + xhalf(tileSum);
    };
    // Two row tiles (non-causal): each K and V operand read feeds both.
    auto compute2 = [&](int buffer,
                        int keyStart) __attribute__((always_inline)) {
      if constexpr (kRowTiles > 1) {
        bool live[kWaveSubTiles];
        bool allLive = true;
#pragma unroll
        for (int ls = 0; ls < kWaveSubTiles; ++ls) {
          live[ls] = keyStart + ls * 16 < seqLen;
          allLive = allLive && live[ls];
        }
        allLive = __builtin_amdgcn_readfirstlane((int)allLive);
        f8 s0[kWaveSubTiles], s1[kWaveSubTiles];
#pragma unroll
        for (int ls = 0; ls < kWaveSubTiles; ++ls) s0[ls] = s1[ls] = f8{};
#pragma unroll
        for (int c = 0; c < kPartChunks; ++c)
#pragma unroll
          for (int ls = 0; ls < kWaveSubTiles; ++ls) {
            const e16 kOperand =
                *(const e16*)&kTile[buffer][ls * 16 + laneRow][c * 16];
            if (allLive || live[ls]) {
              s0[ls] = wmma(kOperand, qOperand[c], s0[ls]);
              s1[ls] = wmma(kOperand, qOperand1[c], s1[ls]);
            }
          }
        if (__builtin_expect(keyStart + kKeyTile > seqLen, 0)) {
#pragma unroll
          for (int ls = 0; ls < kWaveSubTiles; ++ls)
#pragma unroll
            for (int e = 0; e < 8; ++e)
              if (keyStart + ls * 16 + 2 * e + laneHalf >= seqLen)
                s0[ls][e] = s1[ls][e] = -INFINITY;
        }
        op_t p0[kWaveSubTiles], p1[kWaveSubTiles];
        softmax(s0, runMax, runSum, accum, p0);
        softmax(s1, runMax1, runSum1, accum1, p1);
#pragma unroll
        for (int ls = 0; ls < kWaveSubTiles; ++ls)
#pragma unroll
          for (int j = 0; j < kLaneDims; ++j) {
            const elem_t* vRow = &vTile[buffer][j * 16 + laneRow][ls * 16];
            const e8 own = *(const e8*)(vRow + laneHalf * 8);
            const e8 other = *(const e8*)(vRow + (1 - laneHalf) * 8);
            const e16 vOperand =
                __builtin_shufflevector(own, other, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                                        9, 10, 11, 12, 13, 14, 15);
            if (allLive || live[ls]) {
              accum[j] = wmma(vOperand, p0[ls], accum[j]);
              accum1[j] = wmma(vOperand, p1[ls], accum1[j]);
            }
          }
      }
    };
    auto compute = [&](int buffer, int keyStart,
                       int stageNext = -1) __attribute__((always_inline)) {
      if constexpr (kRowTiles > 1) {
        compute2(buffer, keyStart);
        return;
      }
      // v_t 2: the step's pages, read now so Q@K hides the latency.
      const int vPages = kVTr ? step_pages(keyStart) : 0;
      // The global address of key `first` + k of a 16-key sub-tile.
      auto key_src = [&](int first, int k) __attribute__((always_inline)) {
        const int slot = first / kPageSize - keyStart / kPageSize;
        return kvHeadBase +
               (size_t)__builtin_amdgcn_readlane(vPages, slot) * kPageElems +
               (size_t)(first % kPageSize + k) * kKvRowElems;
      };
      // Skip sub-tiles no row of the wave sees.
      bool live[kWaveSubTiles];
#pragma unroll
      for (int ls = 0, st = keyWave; ls < kWaveSubTiles;
           ++ls, st += kKeyWaves) {
        const int first = keyStart + st * 16;
        live[ls] = first <= waveLastKey &&
                   (!kWindow ||
                    first + 15 >= context + waveFirstToken - (kWindow - 1));
      }
      f8 scores[kWaveSubTiles];
#pragma unroll
      for (int ls = 0; ls < kWaveSubTiles; ++ls) scores[ls] = f8{};
      auto qk = [&]<bool kCheck>() __attribute__((always_inline)) {
#pragma unroll
        for (int c = 0; c < kPartChunks; ++c)
#pragma unroll
          for (int ls = 0, st = keyWave; ls < kWaveSubTiles;
               ++ls, st += kKeyWaves) {
#if defined(__GFX12__)
            const op_t kOperand = Elem<kBf16>::load_op(
                &kTile[buffer][st * 16 + a_row(laneRow)][(firstChunk + c) * 16],
                laneHalf);
#else
            const e16 kOperand = *(const e16*)&kTile[buffer][st * 16 + laneRow]
                                                    [(firstChunk + c) * 16];
#endif
            if (!kCheck || live[ls])
              scores[ls] = wmma(kOperand, qOperand[c], scores[ls]);
          }
      };
      bool allLive = true;
#pragma unroll
      for (int ls = 0; ls < kWaveSubTiles; ++ls) allLive = allLive && live[ls];
      if constexpr (kWaveSubTiles == 1) {
        if (live[0]) qk.template operator()<false>();
      } else if (!kCausal || __builtin_amdgcn_readfirstlane((int)allLive)) {
        // All sub-tiles live; read operands at most four WMMAs ahead.
        // Non-causal builds take this path on every step: a sub-tile is dead
        // only past S on the last step, whose keys loaded as zeros and are
        // masked. Skipping it below miscompiles with key_tile 32: the live
        // sub-tile's first WMMA accumulates onto stale registers, not zeros.
#pragma unroll
        for (int c = 0; c < kPartChunks; ++c)
#pragma unroll
          for (int ls = 0, st = keyWave; ls < kWaveSubTiles;
               ++ls, st += kKeyWaves) {
#if defined(__GFX12__)
            scores[ls] = wmma(
                Elem<kBf16>::load_op(&kTile[buffer][st * 16 + a_row(laneRow)]
                                           [(firstChunk + c) * 16],
                                     laneHalf),
                qOperand[c], scores[ls]);
#else
            scores[ls] = wmma(*(const e16*)&kTile[buffer][st * 16 + laneRow]
                                                 [(firstChunk + c) * 16],
                              qOperand[c], scores[ls]);
#endif
            if ((c * kWaveSubTiles + ls + 1) % 4 == 0) {
              // DS reads of four K operands, kOpWords / 4 b128 each.
              __builtin_amdgcn_sched_group_barrier(0x0100, kOpWords, 0);
              __builtin_amdgcn_sched_group_barrier(0x0008, 4, 0);  // WMMAs
            }
          }
      } else {
        qk.template operator()<true>();
      }
      if constexpr (kDimParts > 1) {
        // Sum the parts in the same order in every wave so all agree exactly.
        auto put_scores = [&](int slot, f8 x) __attribute__((always_inline)) {
          *(f4*)&scoreX[slot * 256 + lane * 4] =
              __builtin_shufflevector(x, x, 0, 1, 2, 3);
          *(f4*)&scoreX[slot * 256 + 128 + lane * 4] =
              __builtin_shufflevector(x, x, 4, 5, 6, 7);
        };
        auto get_scores = [&](int slot) __attribute__((always_inline)) {
          const f4 lo = *(const f4*)&scoreX[slot * 256 + lane * 4];
          const f4 hi = *(const f4*)&scoreX[slot * 256 + 128 + lane * 4];
          return (f8)__builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
        };
#pragma unroll
        for (int ls = 0; ls < kWaveSubTiles; ++ls)
          put_scores(wave * kWaveSubTiles + ls, scores[ls]);
        lds_barrier();
        if constexpr (kSplitStage)
          if (stageNext >= 0) stage_k(0, stageNext);
#pragma unroll
        for (int ls = 0; ls < kWaveSubTiles; ++ls) {
          if constexpr (kDimParts == 2) {
            scores[ls] += get_scores((wave ^ 1) * kWaveSubTiles + ls);
          } else {
            f8 sum = {};
#pragma unroll
            for (int part = 0; part < kDimParts; ++part)
              sum += get_scores((wave - dimPart + part) * kWaveSubTiles + ls);
            scores[ls] = sum;
          }
        }
      }

      // v_t 2: the V^T operands, issued before the softmax to overlap it. Lane
      // 8g + i reads key 2i + (g >> 1) of the sub-tile, dims 8 (g & 1) of
      // block j; the transpose gives lane (r, h) dim 16 j + r of keys 2i + h.
      op_t vTr[kVTr ? kWaveSubTiles : 1][kVTr ? kLaneDims : 1];
      if constexpr (kVTr) {
#pragma unroll
        for (int ls = 0, st = keyWave; ls < kWaveSubTiles;
             ++ls, st += kKeyWaves) {
          const int first = keyStart + st * 16;
          const elem_t* src = key_src(first, 2 * (lane & 7) + laneHalf) +
                              kHeadDim + dimPart * kPartDims +
                              8 * ((lane >> 3) & 1);
#pragma unroll
          for (int j = 0; j < kLaneDims; ++j)
            vTr[ls][j] = Elem<kBf16>::load_tr(src + 16 * j);
          // Keys past S or before the window: zero, so no 0 * NaN reaches
          // P@V.
          if (edge_step(keyStart)) {
            opw_t mask;
#pragma unroll
            for (int w = 0; w < kOpWords; ++w)
              mask[w] =
                  (valid(first + 4 * w + laneHalf) ? 0xFFFFu : 0u) |
                  (valid(first + 4 * w + 2 + laneHalf) ? 0xFFFF0000u : 0u);
#pragma unroll
            for (int j = 0; j < kLaneDims; ++j)
              vTr[ls][j] = __builtin_bit_cast(
                  op_t, __builtin_bit_cast(opw_t, vTr[ls][j]) & mask);
          }
        }
      }

      // Mask only diagonal and window-edge steps. No row's last key is before
      // the first token's. Non-causal: only the step past the last key.
      const bool causalEdge =
          kCausal ? keyStart + kKeyTile - 1 > context + firstToken
                  : keyStart + kKeyTile > seqLen;
      const bool windowEdge =
          kWindow && keyStart < context + lastToken - (kWindow - 1);
      if (__builtin_expect(causalEdge || windowEdge, 0)) {
#pragma unroll
        for (int ls = 0, st = keyWave; ls < kWaveSubTiles;
             ++ls, st += kKeyWaves)
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const int key = keyStart + st * 16 + 2 * e + laneHalf;
            if (key > rowLastKey) scores[ls][e] = -INFINITY;
            if constexpr (kWindow)
              if (key < context + rowToken - (kWindow - 1))
                scores[ls][e] = -INFINITY;
          }
      }
      float tileMax = -INFINITY;
#pragma unroll
      for (int ls = 0; ls < kWaveSubTiles; ++ls)
#pragma unroll
        for (int e = 0; e < 8; ++e) tileMax = fmaxf(tileMax, scores[ls][e]);
      tileMax = fmaxf(tileMax, xhalf(tileMax)) * scaleLog2;
      // Lazy rescale: a row's max moves only when exceeded by kMaxSlack (log2).
      const float newMax = tileMax > runMax + kMaxSlack ? tileMax : runMax;
      const float safeMax = newMax == -INFINITY ? 0.f : newMax;
      const float rescale = __builtin_amdgcn_exp2f(runMax - safeMax);
      runMax = newMax;
      if (__builtin_amdgcn_ballot_w32(rescale != 1.f))
#pragma unroll
        for (int j = 0; j < kLaneDims; ++j) accum[j] *= rescale;
      op_t pOperand[kWaveSubTiles];
      float tileSum = 0.f;
#pragma unroll
      for (int ls = 0; ls < kWaveSubTiles; ++ls) {
        float probs[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          probs[e] = __builtin_amdgcn_exp2f(
              __builtin_fmaf(scores[ls][e], scaleLog2, -safeMax));
          tileSum += probs[e];
        }
        pOperand[ls] = p_frag(probs);
      }
      runSum = runSum * rescale + tileSum + xhalf(tileSum);

#pragma unroll
      for (int ls = 0, st = keyWave; ls < kWaveSubTiles;
           ++ls, st += kKeyWaves) {
        VRow v[8];
        if constexpr (!kVT && !kVTr) {
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const unsigned* src =
                (const unsigned*)&vTile[buffer][st * 16 + 2 * e + laneHalf]
                                       [dimPart * kPartDims +
                                        kLaneDims * laneRow];
#pragma unroll
            for (int w = 0; w < kLaneWords; ++w) v[e].w[w] = src[w];
          }
        }
#pragma unroll
        for (int j = 0; j < kLaneDims; ++j) {
          op_t vOperand;
          if constexpr (kVTr) {
            vOperand = vTr[ls][j];
          } else if constexpr (kVT) {
            const elem_t* vRow =
                &vTile[buffer][dimPart * kPartDims + j * 16 + laneRow][st * 16];
            const e8 own = *(const e8*)(vRow + laneHalf * 8);
#if defined(__GFX12__)
            vOperand = own;
#else
            const e8 other = *(const e8*)(vRow + (1 - laneHalf) * 8);
            vOperand = __builtin_shufflevector(own, other, 0, 1, 2, 3, 4, 5, 6,
                                               7, 8, 9, 10, 11, 12, 13, 14, 15);
#endif
          } else {
            vOperand = v_frag(v, j);
          }
          if (live[ls]) accum[j] = wmma(vOperand, pOperand[ls], accum[j]);
        }
      }
    };

    if (keyBegin < stepsEnd) {
      if constexpr (kDoubleBuffer) {
        issue(keyBegin);
        stage(0, keyBegin);
        if constexpr (kGfx12) {
          lds_arrive();
          if (keyBegin + kKeyTile < stepsEnd) issue(keyBegin + kKeyTile);
          lds_wait();
        } else {
          if (keyBegin + kKeyTile < stepsEnd) issue(keyBegin + kKeyTile);
          lds_barrier();
        }
        int buffer = 0;
        for (int keyStart = keyBegin; keyStart < stepsEnd;
             keyStart += kKeyTile) {
          compute(buffer, keyStart);
          if constexpr (kGfx12) {
            if (keyStart + kKeyTile < stepsEnd)
              stage(buffer ^ 1, keyStart + kKeyTile);
            lds_arrive();
            if (keyStart + 2 * kKeyTile < stepsEnd)
              issue(keyStart + 2 * kKeyTile);
            lds_wait();
          } else {
            if (keyStart + kKeyTile < stepsEnd) {
              stage(buffer ^ 1, keyStart + kKeyTile);
              if (keyStart + 2 * kKeyTile < stepsEnd)
                issue(keyStart + 2 * kKeyTile);
            }
            lds_barrier();
          }
          buffer ^= 1;
        }
      } else if constexpr (kSplitStage) {
        issue(keyBegin);
        stage(0, keyBegin);
        if constexpr (kGfx12) {
          lds_arrive();
          if (keyBegin + kKeyTile < stepsEnd) issue(keyBegin + kKeyTile);
          lds_wait();
        } else {
          if (keyBegin + kKeyTile < stepsEnd) issue(keyBegin + kKeyTile);
          lds_barrier();
        }
        for (int keyStart = keyBegin; keyStart < stepsEnd;
             keyStart += kKeyTile) {
          const int next =
              keyStart + kKeyTile < stepsEnd ? keyStart + kKeyTile : -1;
          compute(0, keyStart, next);
          lds_barrier();
          if (next >= 0) {
            stage_v(0, next);
            if (next + kKeyTile < stepsEnd) issue(next + kKeyTile);
          }
        }
      } else {
        issue(keyBegin);
        for (int keyStart = keyBegin; keyStart < stepsEnd;
             keyStart += kKeyTile) {
          lds_barrier();
          stage(0, keyStart);
          if constexpr (kGfx12) {
            lds_arrive();
            if (keyStart + kKeyTile < stepsEnd) issue(keyStart + kKeyTile);
            lds_wait();
          } else {
            lds_barrier();
            if (keyStart + kKeyTile < stepsEnd) issue(keyStart + kKeyTile);
          }
          compute(0, keyStart);
        }
      }
    }

    if constexpr (kKeyWaves > 1) {
      // Merge the key waves into key wave 0.
      __shared__ float keyMax[kKeyWaves][kBlockRows],
          keySum[kKeyWaves][kBlockRows];
      __shared__ __attribute__((aligned(16))) f8 keySlot[kWaves][kWaveSize];
      const int localRow = rowWave * 16 + laneRow;
      __syncthreads();
      if (laneHalf == 0 && dimPart == 0) {
        keyMax[keyWave][localRow] = runMax;
        keySum[keyWave][localRow] = runSum;
      }
      __syncthreads();
      float maxAll = -INFINITY;
#pragma unroll
      for (int k = 0; k < kKeyWaves; ++k)
        maxAll = fmaxf(maxAll, keyMax[k][localRow]);
      float total = 0.f;
#pragma unroll
      for (int k = 0; k < kKeyWaves; ++k)
        total += weight_of(keyMax[k][localRow], maxAll) * keySum[k][localRow];
      const float weight = weight_of(runMax, maxAll);
#pragma unroll
      for (int j = 0; j < kLaneDims; ++j) accum[j] *= weight;
      runMax = maxAll;
      runSum = total;
#pragma unroll
      for (int j = 0; j < kLaneDims; ++j) {
        if (keyWave > 0) keySlot[wave][lane] = accum[j];
        __syncthreads();
        if (keyWave == 0)
#pragma unroll
          for (int k = 1; k < kKeyWaves; ++k)
            accum[j] += keySlot[wave + k * kDimParts][lane];
        __syncthreads();
      }
    }
    const bool owner = kKeyWaves == 1 || keyWave == 0;

    if constexpr (kRowTiles > 1) {
      // The second row tile first: the first's store may return.
      if (numSegments == 1) {
        if (waveRow + 16 + laneRow < numRows) {
          const float inv = __builtin_amdgcn_rcpf(runSum1);
          elem_t* dst = out + ((size_t)(row1 / kG) * kNumQHeads + kvHead * kG +
                               row1 % kG) *
                                  kHeadDim;
#pragma unroll
          for (int e = 0; e < 8; ++e)
#pragma unroll
            for (int j = 0; j < kLaneDims; ++j)
              if (kHeadDimP == kHeadDim ||
                  kLaneDims * acc_row(e, laneHalf) + j < kHeadDim)
                dst[kLaneDims * acc_row(e, laneHalf) + j] =
                    to_elem(accum1[j][e] * inv);
        }
      } else {
        const size_t partialRow = (groupSlot + segment) * kBlockRows +
                                  rowWave * 16 * kRowTiles + 16 + laneRow;
        float* dst = partialO + partialRow * kHeadDim;
#pragma unroll
        for (int e = 0; e < 8; ++e)
#pragma unroll
          for (int j = 0; j < kLaneDims; ++j)
            if (kHeadDimP == kHeadDim ||
                kLaneDims * acc_row(e, laneHalf) + j < kHeadDim)
              dst[kLaneDims * acc_row(e, laneHalf) + j] = accum1[j][e];
        if (laneHalf == 0) {
          partialML[partialRow * 2] = runMax1;
          partialML[partialRow * 2 + 1] = runSum1;
        }
      }
    }

    if (numSegments == 1) {
      if (!owner || waveRow + laneRow >= numRows) return;
      const float inv = __builtin_amdgcn_rcpf(runSum);
      elem_t* dst = out + ((size_t)rowToken * kNumQHeads + rowHead) * kHeadDim +
                    dimPart * kPartDims;
      if constexpr (kVTr) {
        // v_t 2: eight consecutive dims per lane and block j.
#pragma unroll
        for (int j = 0; j < kLaneDims; ++j) {
          e8 x;
#pragma unroll
          for (int e = 0; e < 8; ++e) x[e] = to_elem(accum[j][e] * inv);
          *(e8*)(dst + 16 * j + 8 * laneHalf) = x;
        }
        return;
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        elem_t x[kLaneDims];
#pragma unroll
        for (int j = 0; j < kLaneDims; ++j) x[j] = to_elem(accum[j][e] * inv);
        if constexpr (kLaneDims % 2) {
#pragma unroll
          for (int j = 0; j < kLaneDims; ++j)
            if (kHeadDimP == kHeadDim ||
                kLaneDims * acc_row(e, laneHalf) + j < kHeadDim)
              dst[kLaneDims * acc_row(e, laneHalf) + j] = x[j];
        } else {
          unsigned* d = (unsigned*)(dst + kLaneDims * acc_row(e, laneHalf));
#pragma unroll
          for (int w = 0; w < kLaneWords; ++w)
            d[w] = __builtin_bit_cast(unsigned, e2{x[2 * w], x[2 * w + 1]});
        }
      }
      return;
    }

    // Split KV: publish (O, max, sum); the last segment to arrive merges and
    // resets the counter.
    const size_t partialBase = (groupSlot + segment) * kBlockRows;
    if (owner) {
      const size_t partialRow =
          partialBase + rowWave * 16 * kRowTiles + laneRow;
      float* dst = partialO + partialRow * kHeadDim + dimPart * kPartDims;
#pragma unroll
      for (int e = 0; e < 8; ++e)
#pragma unroll
        for (int j = 0; j < kLaneDims; ++j)
          if (kHeadDimP == kHeadDim ||
              kLaneDims * acc_row(e, laneHalf) + j < kHeadDim)
            dst[kVTr ? 16 * j + acc_row(e, laneHalf)
                     : kLaneDims * acc_row(e, laneHalf) + j] = accum[j][e];
      if (laneHalf == 0 && dimPart == 0) {
        partialML[partialRow * 2] = runMax;
        partialML[partialRow * 2 + 1] = runSum;
      }
    }
    __threadfence();
    __shared__ int isLast;
    __syncthreads();
    if (thread == 0) {
      isLast = atomicAdd(&counters[counter], 1) == numSegments - 1;
      if (isLast) counters[counter] = 0;
    }
    __syncthreads();
    if (!isLast) return;
    __threadfence();
    // Merge kMergeGroup segments at a time.
    constexpr int kMergeGroup = kMaxSegments < 8 ? kMaxSegments : 8;
    constexpr int kMergeIters =
        (kBlockRows * kHeadDim + kThreads * 4 - 1) / (kThreads * 4);
#pragma unroll 1
    for (int it = 0; it < kMergeIters; ++it) {
      const int elem = (it * kThreads + thread) * 4;
      const int r = elem / kHeadDim, d = elem % kHeadDim;
      const int outRow = tileRow + r;
      if (elem >= kBlockRows * kHeadDim || outRow >= numRows) continue;
      const size_t rowBase = groupSlot * kBlockRows + r;
      float maxAll = -INFINITY, denominator = 0.f;
      f4 numerator = {};
#pragma unroll 1
      for (int first = 0; first < numSegments; first += kMergeGroup) {
        float m[kMergeGroup], l[kMergeGroup];
        f4 o[kMergeGroup];
#pragma unroll
        for (int g = 0; g < kMergeGroup; ++g) {
          const size_t pr = rowBase + (size_t)(first + g) * kBlockRows;
          const bool in = first + g < numSegments;
          m[g] = in ? partialML[pr * 2] : -INFINITY;
          l[g] = in ? partialML[pr * 2 + 1] : 0.f;
          o[g] = in ? *(const f4*)(partialO + pr * kHeadDim + d) : f4{};
        }
        float groupMax = maxAll;
#pragma unroll
        for (int g = 0; g < kMergeGroup; ++g) groupMax = fmaxf(groupMax, m[g]);
        const float carry = weight_of(maxAll, groupMax);
        denominator *= carry;
        numerator *= carry;
#pragma unroll
        for (int g = 0; g < kMergeGroup; ++g) {
          const float weight = weight_of(m[g], groupMax);
          denominator = fmaf(weight, l[g], denominator);
          numerator += weight * o[g];
        }
        maxAll = groupMax;
      }
      const float inv = __builtin_amdgcn_rcpf(denominator);
      elem_t* dst =
          out +
          ((size_t)(outRow / kG) * kNumQHeads + kvHead * kG + outRow % kG) *
              kHeadDim +
          d;
#pragma unroll
      for (int k = 0; k < 4; ++k) dst[k] = to_elem(numerator[k] * inv);
    }
  }
};

// Arguments are checked by csrc/py_itfs_cu/rdna_attn.cu. Splits the KV
// until the GPU is full, keeping at least kMinSegmentSteps steps per segment;
// a batch's plan has split it already.
template <PrefillConfig C>
void launch_prefill(const PrefillArgs& a) {
  using P = Prefill<C>;
  using elem_t = typename P::elem_t;
  // The WGPs, and the workgroups resident in each: as many as its LDS and its
  // SIMDs' waves allow, with the waves the kernel was built for. A WGP with
  // fewer VGPRs than 4 SIMDs x 32 lanes x 1536 is a 1024-VGPR one.
  static std::atomic<int> wgpsCache[kMaxDevices];
  // The WGPs in the high half, the workgroups resident in each in the low one.
  const int wgps = per_device(wgpsCache, [](int device) {
    int n = 0, regs = 0;
    (void)hipDeviceGetAttribute(&n, hipDeviceAttributeMultiprocessorCount,
                                device);
    (void)hipDeviceGetAttribute(
        &regs, hipDeviceAttributeMaxRegistersPerMultiprocessor, device);
    const bool smallVgprs =
        regs > 0 && regs < kSimdsPerWgp * P::kWaveSize * kBigLaneVgprs;
    const int waves = smallVgprs ? P::kSmallVgprWavesPerSimd : P::kWavesPerSimd;
    const int resident = std::min(kWgpLdsBytes / P::kLdsBudget,
                                  waves * kSimdsPerWgp / P::kWaves);
    return (std::max(1, n) << 16) | std::max(1, resident);
  });
  const int numWgps = wgps >> 16, wgpWorkgroups = wgps & 0xFFFF;
  if (a.plan) {
    const int headGroup = a.head_group ? a.head_group : P::kHeadGroup;
    hipLaunchKernelGGL(
        P::template prefill_attn<true>, dim3(a.num_entries * P::kNumKvHeads),
        dim3(P::kThreads), 0, a.stream, static_cast<const elem_t*>(a.q),
        static_cast<const elem_t*>(a.kv), a.bt, static_cast<elem_t*>(a.out),
        a.seq_lens, 0, a.scale, a.partial_o, a.partial_ml, a.counters, 1,
        wgpWorkgroups * numWgps, a.mm_ranges, a.num_mm_ranges,
        reinterpret_cast<const int4*>(a.plan), a.query_start, a.num_entries,
        a.bt_stride, headGroup, static_cast<const elem_t*>(a.v), a.q_stride,
        a.k_stride, a.v_stride);
    return;
  }
  // A workgroup per SIMD.
  const int targetWorkgroups = kSimdsPerWgp * numWgps;
  const int numTiles =
      (a.num_tokens * P::kG + P::kBlockRows - 1) / P::kBlockRows;
  const int groups = numTiles * P::kNumKvHeads;
  int segments = 1;
  if (P::kMaxSegments > 1 && groups < targetWorkgroups) {
    // Keys a tile sees.
    const int keys =
        P::kWindow ? std::min(a.max_seq_len,
                              P::kWindow + P::kBlockRows / P::kG + P::kKeyTile)
                   : a.max_seq_len;
    const int steps = (keys + P::kKeyTile - 1) / P::kKeyTile;
    segments =
        std::min({(targetWorkgroups + groups - 1) / groups, P::kMaxSegments,
                  std::max(1, steps / P::kMinSegmentSteps)});
  }
  hipLaunchKernelGGL(
      P::template prefill_attn<false>, dim3(groups * segments),
      dim3(P::kThreads), 0, a.stream, static_cast<const elem_t*>(a.q),
      static_cast<const elem_t*>(a.kv), a.bt, static_cast<elem_t*>(a.out),
      a.seq_lens, a.num_tokens, a.scale, a.partial_o, a.partial_ml, a.counters,
      segments, wgpWorkgroups * numWgps, a.mm_ranges, a.num_mm_ranges, nullptr,
      nullptr, 0, 0, P::kHeadGroup, static_cast<const elem_t*>(a.v), a.q_stride,
      a.k_stride, a.v_stride);
}

}  // namespace

}  // namespace rdna
