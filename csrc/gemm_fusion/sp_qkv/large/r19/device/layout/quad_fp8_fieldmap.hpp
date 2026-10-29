// SPDX-License-Identifier: MIT
// LG-QSDMA-r1 virtual-N field table (pure constexpr, no Opus; includable by host and device).
// Golden: select_qkv_from_peers (sp4-attention-fusion-cases-20261009/reference.py l.10-22):
// destination d slab columns, in order: Q[d*2048:+2048], K[8192+d*128:+128], V[8704+d*128:+128].
// Every field segment is a multiple of 128 raw columns = 8 n16 bands, so shuffle(16,16)
// blocks are never split and a 128-wide half tile maps to ONE contiguous raw segment.
#pragma once
#include <stdint.h>
#ifndef LG_HD
#if defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
#define LG_HD __host__ __device__
#else
#define LG_HD
#endif
#endif
namespace lgq {
struct FieldSeg { int rawBegin; int packedBegin; int perDest; };
// Dense large-M (no indexer): Q, K, V. Indexer IQ/IK rows are a later extension (hook below).
static constexpr FieldSeg kFields[3] = {
    {0,    0,    2048},   // Q
    {8192, 2048, 128},    // K
    {8704, 2176, 128},    // V
};
static constexpr int kNumDest = 4;
static constexpr int kF = 2304;                 // slab columns per destination
static constexpr int kSegCols = 128;            // half-tile width
static constexpr int kSegPerDest = kF / kSegCols;   // 18
static constexpr int kVirtualN = kNumDest * kF;     // 9216 == raw N (dense)
static constexpr int kTileN = 256, kTilesN = kVirtualN / kTileN;  // 36
static_assert(kF % kSegCols == 0 && kF % kTileN == 0, "tile never straddles a destination");

// half-tile index hv in [0, 72) -> destination, packed column origin, raw column origin.
LG_HD constexpr int ht_dest(int hv)   { return hv / kSegPerDest; }
LG_HD constexpr int ht_packed(int hv) { return (hv % kSegPerDest) * kSegCols; }
LG_HD constexpr int ht_field(int hv)  {
    return ht_packed(hv) < kFields[1].packedBegin ? 0 : (ht_packed(hv) < kFields[2].packedBegin ? 1 : 2);
}
// r1d: constant-selected (no runtime index into the namespace-scope constexpr array from device code).
LG_HD constexpr int ht_raw(int hv) {
    return ht_field(hv) == 0 ? (kFields[0].rawBegin + ht_dest(hv) * kFields[0].perDest + (ht_packed(hv) - kFields[0].packedBegin))
         : ht_field(hv) == 1 ? (kFields[1].rawBegin + ht_dest(hv) * kFields[1].perDest + (ht_packed(hv) - kFields[1].packedBegin))
                             : (kFields[2].rawBegin + ht_dest(hv) * kFields[2].perDest + (ht_packed(hv) - kFields[2].packedBegin));
}
// compile-time proof: the 72 half tiles cover every raw 128-segment exactly once.
LG_HD constexpr bool fieldmap_is_permutation() {
    bool seen[kVirtualN / kSegCols] = {};
    for (int hv = 0; hv < kVirtualN / kSegCols; ++hv) {
        const int r = ht_raw(hv);
        if (r % kSegCols != 0 || r < 0 || r >= kVirtualN) return false;
        if (seen[r / kSegCols]) return false;
        seen[r / kSegCols] = true;
    }
    return true;
}
static_assert(fieldmap_is_permutation(), "virtual-N map must be a permutation of raw 128-col segments");
static_assert(ht_raw(0) == 0 && ht_raw(16) == 8192 && ht_raw(17) == 8704 &&
              ht_raw(18 + 16) == 8192 + 128 && ht_raw(18 + 17) == 8704 + 128 &&
              ht_raw(54) == 3 * 2048 && ht_raw(54 + 16) == 8192 + 3 * 128, "golden spot checks");

// HOOK (indexer, later slice): IK is replicated to all 4 destinations. The IK single-owner
// store (design 3.2 option (i), review L5) would be added here as a field with replicated=1
// whose owner tile (d=0) additionally stores to S_1..S_3. NOT implemented in this dense slice.
struct OwnerStoreHookNone { LG_HD static constexpr bool enabled() { return false; } };
} // namespace lgq
