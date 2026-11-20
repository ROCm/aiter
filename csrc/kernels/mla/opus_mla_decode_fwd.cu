// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Opus MLA decode launchers (gfx950), bound to Python through the torch-free
// ctypes C ABI (aiter_tensor_t* + trailing hipStream_t).
//
// opus_mla_decode_fwd / opus_mla_decode_mxfp8_fwd: stage 1 only. Work
// partitioning comes from aiter's metadata (work_indptr / work_info_set) and
// the split-KV merge is left to mla_reduce_v1. Per-split partials go to
// logits/attn_lse (== aiter split_output/split_lse); work items that own a
// whole request (partial_slot < 0) write the final output directly.
//   opus_mla_decode_fwd        merged d=576 Q/KV, dispatched on q's dtype:
//                                bf16 -- prebuilt code objects under
//                                        hsa/<arch>/mla_opus/, because they need a
//                                        toolchain aiter does not ship.
//                                fp8  -- per-tensor scalar descales (q_scale /
//                                        kv_scale); routed by nhead x max_seqlen_q
//                                        in opus_a8w8_route.
//   opus_mla_decode_mxfp8_fwd  fp8 NoPE with per-block E8M0 scales + bf16 RoPE,
//                              so Q and KV are three tensors each.
//
// opus_mla_decode_flat_plan / opus_mla_decode_flat_fwd: opt-in fp8 path with
// its own host planner and bf16-partial merge. Rows are token * H + head, so a
// request's query tokens and heads fill 128-row chunks at any head count.

#include "aiter_tensor.h"

#include "aiter_ctypes_error.h"

#include "aiter_hip_common.h"
#include "opus/gfx950/mla_decode_a8w8_16mx1_32nx4.hpp"
#include "opus/mla_decode_a8w8_flat_32mx4_64nx1.hpp"
#include "opus/mla_decode_kargs.h"
#include "opus/gfx950/mla_decode_mxfp8_16mx8_32nx1.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

AITER_CTYPES_ERROR_DEF

namespace {

// fp8 32mx4 / 64nx1 (csrc/kernels/mla/opus/gfx950/mla_decode_a8w8_32mx4_64nx1.hpp): 4 waves x 32 packed
// query rows, i.e. the 128-row work items aiter's metadata packs by default for fp8. Prebuilt
// only: its register map (Q, half of O and the K/V ring in AGPRs, the other half of O and both
// score buffers in VGPRs) needs the pin toolchain, and a stock ROCm clang forces every MFMA into
// AGPR form, which cannot fit. One entry per specialization.
constexpr const char* kA8W8_32mx4_co = "mla_opus/opus_mla_decode_a8w8_32mx4_64nx1.co";
// fp8 32mx2 / 32nx2 (mla_decode_a8w8_32mx2_32nx2.hpp): two wave pairs of 32 packed query rows,
// 32x32x64 for both GEMMs, prebuilt the same way (mla_decode_a8w8_32mx2_32nx2.cu). 64 packed
// query rows per workgroup, so a work item must not carry more than 64 rows -- hand it a
// 128-row item and the rows past 64 are never computed -- which the metadata guarantees for
// exactly the shapes opus_a8w8_route sends here (64 rows or fewer per request, and
// kPackedQoLenPerWg 64 for nhead 32 x 3).
constexpr const char* kA8W8_32mx2_co = "mla_opus/opus_mla_decode_a8w8_32mx2_32nx2.co";
constexpr int kCoBlockSize           = 256;

constexpr int kHeadDimQk = 576;
constexpr int kHeadDimVo = 512;

using MxFp8Traits = opus_mla_decode_mxfp8_16mx8_32nx1_traits<16, 32, 8, fp8_t, bf16_t, bf16_t>;

// --- fp8 (16, 1): 4 waves of 32 KV tokens, Q shared through LDS ---------------------
// The row-tiled builds give every wave its own query rows, so they need nhead * max_seqlen_q
// rows to fill a workgroup. At nhead 16 with one query token there are only 16 rows to go
// round, so this build tiles the KV tokens across the waves instead: four waves of 32
// tokens, ping-pong over two LDS slots.
//
// page_size 1 only -- the rope DMA deals a 16-token line, so the within-page token offset
// would be per-lane above 1. The 16nx8 geometry this replaced is the only fp8 build that
// ever addressed a real block table; it still sits in opus/ but nothing routes to it, and
// the python gate now keeps page_size > 1 on the asm path.
template <bool CAUSAL, bool LARGE_KV = false>
using OpusTraits16mx1x32nx4C =
    opus_mla_decode_a8w8_16mx1_32nx4_traits<16, 32, 4, fp8_t, fp8_t, bf16_t, CAUSAL, LARGE_KV, 2>;
inline bool opus_env_flag(const char* name)
{
    const char* v = std::getenv(name);
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

// Shape routing for the merged-buffer fp8 path, mirroring the asm persistent fp8 heuristic
// (csrc/py_itfs_cu/asm_mla.cu): what asm runs on its qh32 / qseqlen4 kernel (128 packed rows
// per workgroup) goes to 32mx4, what it runs on qh16 / qseqlen4 (64 rows) to 32mx2. nhead
// here is nhead per KV head (MLA has one), after mla.py's fold of the non-native head counts.
// aiter/mla.py asks opus_mla_decode_supported, so a shape with no build stays on the asm path.
enum class OpusA8W8Route
{
    k32mx4,
    k32mx2,
    k16mx1,
    kNone,
};

inline OpusA8W8Route opus_a8w8_route(int nhead, int max_seqlen_q)
{
    const int q = max_seqlen_q;
    if((nhead == 32 && q >= 4) || (nhead == 64 && q >= 2) || nhead == 128 ||
       (nhead == 96 && q <= 6) || (nhead == 16 && q > 4) || (nhead == 12 && nhead * q <= 128))
        return OpusA8W8Route::k32mx4;
    if((nhead == 16 && (q == 3 || q == 4)) || (nhead == 32 && (q == 2 || q == 3)) ||
       (nhead == 64 && q == 1))
        return OpusA8W8Route::k32mx2;
    if(nhead == 16 && q == 1)
        return OpusA8W8Route::k16mx1;
    return OpusA8W8Route::kNone;
}

struct OpusLaunch16mx1x32nx4
{
    template <class Traits>
    static void
    launch(int num_workers, hipStream_t stream, const opus_mla_decode_a8w8_kargs& kargs)
    {
        opus_mla_decode_a8w8_16mx1_32nx4_kernel<Traits>
            <<<dim3(num_workers, 1, 1), dim3(Traits::BLOCK_SIZE), 0, stream>>>(kargs);
    }
};

// causal x large_kv fan-out for the traits-templated in-tree build (16mx1).
template <template <bool, bool> class TraitsC, class Launcher>
void launch_opus_16mx1(bool causal,
                       int max_seqlen_q,
                       bool large_kv,
                       int num_workers,
                       hipStream_t stream,
                       const opus_mla_decode_a8w8_kargs& kargs)
{
    const bool use_causal = causal && max_seqlen_q > 1;
    auto launch           = [&](auto traits) {
        Launcher::template launch<decltype(traits)>(num_workers, stream, kargs);
    };
    if(use_causal && large_kv)
        launch(TraitsC<true, true>{});
    else if(use_causal)
        launch(TraitsC<true, false>{});
    else if(large_kv)
        launch(TraitsC<false, true>{});
    else
        launch(TraitsC<false, false>{});
}

constexpr const char* kFwdName = "opus_mla_decode_fwd";

// bf16 Q/KV: the prebuilt a16w16 code objects, picked by head count. These builds have no
// causal specialization, so max_seqlen_q and causal do not reach them.
void opus_mla_decode_bf16(opus_mla_decode_kargs kargs, int num_workers, hipStream_t stream)
{
    const int H     = kargs.H;
    size_t arg_size = sizeof(kargs);
    auto launch_co  = [&](AiterAsmKernel& impl, int heads_per_block) {
        const int num_h_blocks = (H + heads_per_block - 1) / heads_per_block;
        impl.launch_kernel(
            {&kargs, &arg_size, num_workers, num_h_blocks, 1, kCoBlockSize, 1, 1, stream});
    };
#define OPUS_A16W16_CO_LAUNCH(name, co, heads_per_block)                                \
    do                                                                                  \
    {                                                                                   \
        static AiterAsmKernel impl("opus_mla_decode_a16w16_" name "_kernel",            \
                                   "mla_opus/opus_mla_decode_a16w16_" co ".co");       \
        launch_co(impl, heads_per_block);                                               \
    } while(0)
    if(H <= 32)
        OPUS_A16W16_CO_LAUNCH("32mx1_16nx4", "32mx1_16nx4", 32);
    else if(H % 128 == 0)
        OPUS_A16W16_CO_LAUNCH("32mx4_32nx1", "32mxt_32nx1", 128);
    else if(H % 96 == 0)
        OPUS_A16W16_CO_LAUNCH("32mx3_32nx1", "32mxt_32nx1", 96);
    else
        OPUS_A16W16_CO_LAUNCH("16mx4_64nx1", "16mx4_64nx1", 64);
#undef OPUS_A16W16_CO_LAUNCH
}

// fp8 NoPE and RoPE merged into one d=576 buffer, with per-tensor scalar descales. No build
// here addresses a block table: the 16mx1 shape's rope DMA deals a 16-token line and so
// cannot carry a per-lane within-page offset, so page_size > 1 belongs to the asm path and
// the python gate keeps it there.
void opus_mla_decode_a8w8(const opus_mla_decode_kargs& base,
                         const aiter_tensor_t* q_scale,
                         const aiter_tensor_t* kv_scale,
                         int max_seqlen_q,
                         bool causal,
                         int num_workers,
                         hipStream_t stream)
{
    AITER_CHECK(max_seqlen_q >= 1, kFwdName, ": fp8 needs max_seqlen_q, got ", max_seqlen_q);
    const OpusA8W8Route route = opus_a8w8_route(base.H, max_seqlen_q);
    AITER_CHECK(route != OpusA8W8Route::kNone,
                kFwdName,
                ": no opus fp8 build for nhead ",
                base.H,
                " x max_seqlen_q ",
                max_seqlen_q,
                " (the python gate keeps such shapes on the asm path)");
    AITER_CHECK(q_scale != nullptr && q_scale->dtype() == AITER_DTYPE_fp32 && q_scale->numel() >= 1,
                kFwdName,
                ": q_scale must be a float scalar tensor");
    AITER_CHECK(kv_scale != nullptr && kv_scale->dtype() == AITER_DTYPE_fp32 &&
                    kv_scale->numel() >= 1,
                kFwdName,
                ": kv_scale must be a float scalar tensor");

    opus_mla_decode_a8w8_kargs kargs{base, q_scale->data_ptr(), kv_scale->data_ptr()};
    const int total_tokens = kargs.total_tokens;

    // A buffer descriptor's num_records is 32 bits, so it cannot span a KV cache of 4 GiB
    // or more; past that the bound wraps and every load beyond it silently returns zero.
    // Unlike the contiguous fmha case there is no way to rebase the descriptor per tile
    // here: page_size is 1, so one KV tile's 32 tokens sit at unrelated, per-lane offsets
    // while a descriptor base is wave-uniform. The large path addresses KV with a flat
    // 64-bit pointer (global_load_lds) instead, which costs ~1-2%, hence the gate.
    const int64_t kv_bytes = static_cast<int64_t>(total_tokens) *
                             static_cast<int64_t>(kargs.stride_kv_page) *
                             static_cast<int64_t>(sizeof(fp8_t));
    const bool large_kv = kv_bytes >= (int64_t{1} << 32);

    // The prebuilt builds: entry <name>_{nc,causal}[_large][_cprr] of one code object each,
    // every entry loaded once on first use.
    const bool use_causal = causal && max_seqlen_q > 1;
    // With one query token every local KV token is in its causal past, CP or not.
    const bool use_cprr = use_causal && kargs.cp_world_size > 1;
    size_t arg_size       = sizeof(kargs);
    auto launch_co        = [&](AiterAsmKernel& impl) {
        impl.launch_kernel({&kargs, &arg_size, num_workers, 1, 1, kCoBlockSize, 1, 1, stream});
    };
#define OPUS_A8W8_CO_ENTRY(name, entry, co)                                            \
    do                                                                                 \
    {                                                                                  \
        static AiterAsmKernel impl("opus_mla_decode_a8w8_" name "_" entry, co);        \
        launch_co(impl);                                                               \
    } while(0)
#define OPUS_A8W8_CO_LAUNCH(name, co)                                                  \
    do                                                                                 \
    {                                                                                  \
        static bool traced = false; /* first call only: the test loops the kernel */   \
        if(!traced && opus_env_flag("AITER_MLA_OPUS_TRACE") && (traced = true))        \
            fprintf(stderr,                                                            \
                    "[opus_mla_decode_a8w8] " name " co causal=%d large_kv=%d"         \
                    " cprr=%d\n",                                                      \
                    use_causal,                                                        \
                    large_kv,                                                          \
                    use_cprr);                                                         \
        if(large_kv && use_cprr)                                                       \
            OPUS_A8W8_CO_ENTRY(name, "causal_large_cprr", co);                         \
        else if(use_cprr)                                                              \
            OPUS_A8W8_CO_ENTRY(name, "causal_cprr", co);                               \
        else if(large_kv && use_causal)                                                \
            OPUS_A8W8_CO_ENTRY(name, "causal_large", co);                              \
        else if(large_kv)                                                              \
            OPUS_A8W8_CO_ENTRY(name, "nc_large", co);                                  \
        else if(use_causal)                                                            \
            OPUS_A8W8_CO_ENTRY(name, "causal", co);                                    \
        else                                                                           \
            OPUS_A8W8_CO_ENTRY(name, "nc", co);                                        \
    } while(0)

    // 128 packed query rows per workgroup on 4 waves.
    if(route == OpusA8W8Route::k32mx4)
    {
        OPUS_A8W8_CO_LAUNCH("32mx4_64nx1", kA8W8_32mx4_co);
        return;
    }

    // 64 packed query rows per workgroup on 4 waves (two pairs of 32).
    if(route == OpusA8W8Route::k32mx2)
    {
        OPUS_A8W8_CO_LAUNCH("32mx2_32nx2", kA8W8_32mx2_co);
        return;
    }
#undef OPUS_A8W8_CO_LAUNCH
#undef OPUS_A8W8_CO_ENTRY

    // route == k16mx1.
    launch_opus_16mx1<OpusTraits16mx1x32nx4C, OpusLaunch16mx1x32nx4>(
        causal, max_seqlen_q, large_kv, num_workers, stream, kargs);
}

// ---- flat-row planner, stage-1 launch helpers, and bf16 merge --------------------


constexpr int kSlot       = 128; // rows per chunk and per partial slot
constexpr int kKvTile     = 32;  // KV tokens per plan tile
constexpr const char* kFn = "opus_mla_decode_flat";

// ---- planner
// -------------------------------------------------------------------------------------

// planner cost of one work item in 64-token tiles (~14k cycles per item vs ~2.35k per tile)
constexpr int kItemTiles = 6;
// whole requests per XCD unless the busiest XCD would get more than (1 + kXcdSplit) x the mean
// tiles; then the requests are cut into equal consecutive XCD shares. Below ~10 % the clocks absorb
// the imbalance at the power cap and the cuts cost more than they save.
constexpr double kXcdSplit = 0.10;

using PlanItem = std::array<int, 10>;

struct Segment
{
    int request;
    int begin;
    int end;
    int tile0; // request tile at begin
};

struct UnitRange
{
    int begin = std::numeric_limits<int>::max();
    int end   = -1;
};

// balance: only the last worker of an XCD may absorb a remainder of up to one item's cost (see
// column_end), and every piece is checked to give each of its rows a key.
//
// q_indptr gives each request its own query-token count, so a request's rows are
// q_indptr[b+1]-q_indptr[b] tokens x h heads and its flat row base is q_indptr[b]*h. That is
// already what the kernel assumes -- it reads the causal span from q_indptr and takes
// qo_start/qo_end per work item -- so a ragged step needs no kernel change, only this.
//
// The layout below sizes its columns by the batch's deepest request and skips the chunks a
// shallower request does not have, which leaves some column workers idle on a ragged step;
// build_plan_bucketed_impl groups by depth instead. A uniform q_indptr makes every per-request
// path here inert, so the plan is bit-for-bit what it was before this argument existed.
int build_plan_default_impl(const int* kv_indptr,
                            const int* q_indptr,
                            int bs,
                            int h,
                            int nwg,
                            bool balance,
                            int nxcd,
                            int item_cap,
                            int max_units,
                            int* plan,
                            int plan_ints,
                            int* out_items,
                            int* out_units)
{
    constexpr int slot    = kSlot;
    constexpr int kv_tile = kKvTile;
    if(!kv_indptr || !q_indptr || !plan || !out_items || !out_units || bs < 0 || h <= 0 ||
       nwg <= 0 || nxcd <= 0 || (nwg % nxcd) != 0)
        return -1;
    const int required = nwg + 1 + item_cap * 8 + max_units * 4;
    if(plan_ints < required)
        return -2;

    std::fill(plan, plan + required, 0);
    *out_items = 0;
    *out_units = 0;
    if(bs == 0)
        return 0;

    // Per-request rows and chunk depth, plus the chunk-offset prefix sum that indexes the unit
    // table (it replaces b * nch, which only holds when every request is the same width).
    std::vector<int> nrows_v(bs), nch_v(bs), coff(bs + 1, 0);
    for(int b = 0; b < bs; ++b)
    {
        const int q_len = q_indptr[b + 1] - q_indptr[b];
        if(q_len < 0)
            return -6;
        nrows_v[b]  = q_len * h;
        nch_v[b]    = (nrows_v[b] + slot - 1) / slot;
        coff[b + 1] = coff[b] + nch_v[b];
    }
    if(coff[bs] > max_units)
        return -3;
    // Columns are laid out to the deepest request; `add` skips chunks a request lacks.
    const int nch = *std::max_element(nch_v.begin(), nch_v.end());
    if(nch <= 0)
        return -3;

    std::vector<int> tiles(bs);
    for(int b = 0; b < bs; ++b)
    {
        const int len = kv_indptr[b + 1] - kv_indptr[b];
        if(len < 0)
            return -4;
        tiles[b] = (len + kv_tile - 1) / kv_tile;
    }

    std::vector<int> request_order(bs);
    std::iota(request_order.begin(), request_order.end(), 0);
    std::stable_sort(request_order.begin(), request_order.end(), [&](int a, int b) {
        return tiles[a] > tiles[b];
    });

    // per XCD: the requests (or request tile ranges) it owns with all chunks, as tile segments in
    // request order
    std::vector<std::vector<Segment>> xseg(nxcd);
    std::vector<int> xtiles(nxcd, 0);
    std::vector<std::vector<std::array<int, 3>>> owners(nxcd); // {request, first tile, end tile}
    {
        std::vector<long long> xcd_load(nxcd, 0);
        for(const int b : request_order)
        {
            int x = 0;
            for(int candidate = 1; candidate < nxcd; ++candidate)
                if(xcd_load[candidate] < xcd_load[x])
                    x = candidate;
            owners[x].push_back({b, 0, tiles[b]});
            xcd_load[x] += tiles[b];
        }
        const long long total   = std::accumulate(xcd_load.begin(), xcd_load.end(), 0LL);
        const long long busiest = *std::max_element(xcd_load.begin(), xcd_load.end());
        // a cut adds nch items: no piece below two items' cost
        const int min_piece = static_cast<int>(std::ceil(2.0 * kItemTiles * 64.0 / kv_tile));
        if(kXcdSplit >= 0 && busiest * nxcd > total * (1.0 + kXcdSplit) &&
           busiest - total / nxcd > min_piece)
        {
            for(auto& requests : owners)
                requests.clear();
            int x         = 0;
            long long pos = 0;
            auto bound    = [&](int xcd) {
                return xcd + 1 == nxcd ? total : total * (xcd + 1) / nxcd;
            };
            for(const int b : request_order)
                for(int t = 0; t < tiles[b];)
                {
                    int take = static_cast<int>(std::min<long long>(tiles[b] - t, bound(x) - pos));
                    if(tiles[b] - t - take < min_piece)
                        take = tiles[b] - t;
                    else if(take < min_piece && x + 1 < nxcd)
                    {
                        ++x;
                        continue;
                    }
                    owners[x].push_back({b, t, t + take});
                    t += take;
                    pos += take;
                    while(x + 1 < nxcd && pos >= bound(x))
                        ++x;
                }
        }
    }
    for(int x = 0; x < nxcd; ++x)
    {
        std::sort(owners[x].begin(), owners[x].end());
        for(const auto& [b, first, end] : owners[x])
        {
            xseg[x].push_back({b, xtiles[x], xtiles[x] + end - first, first});
            xtiles[x] += end - first;
        }
    }

    std::vector<std::vector<PlanItem>> per_wg(nwg);
    const int wg_per_xcd = nwg / nxcd;
    for(int x = 0; x < nxcd; ++x)
    {
        const std::vector<Segment>& segments = xseg[x];
        const int total_tiles                = xtiles[x];
        if(total_tiles == 0)
            continue;

        auto emit = [&](int worker, int b, int chunk, int tile_begin, int tile_end) {
            const int kv_len      = kv_indptr[b + 1] - kv_indptr[b];
            const int valid_rows  = std::min(slot, nrows_v[b] - chunk * slot);
            const int query_begin = q_indptr[b] * h + slot * chunk;
            per_wg[worker].push_back({b,
                                      0,
                                      query_begin,
                                      query_begin + valid_rows,
                                      kv_indptr[b] + kv_tile * tile_begin,
                                      kv_indptr[b] + std::min(kv_tile * tile_end, kv_len),
                                      0,
                                      0,
                                      chunk,
                                      tile_begin});
        };
        auto add = [&](int worker, int chunk, int lo, int hi) {
            for(const Segment& segment : segments)
            {
                // A shallower request has no row chunk at this index; emitting one would
                // describe rows it does not own.
                if(chunk >= nch_v[segment.request])
                    continue;
                const int tile_begin = std::max(lo, segment.begin) - segment.begin + segment.tile0;
                const int tile_end   = std::min(hi, segment.end) - segment.begin + segment.tile0;
                if(tile_end > tile_begin)
                    emit(worker, segment.request, chunk, tile_begin, tile_end);
            }
        };

        // Worker cost in plan tiles: its tiles plus the kItemTiles equivalent per item (setup, Q
        // load, pipeline fill and epilogue). Column workers share their edges across the nch
        // chunks; the tail workers take the owned tail of every chunk. The edges come from a
        // bisection on the max worker cost.
        const double item_cost = kItemTiles * 64.0 / kv_tile;
        auto pieces            = [&](int lo, int hi) {
            int n = 0;
            for(const Segment& segment : segments)
                n += std::min(hi, segment.end) > std::max(lo, segment.begin);
            return n;
        };
        // A column whose segment remainder fits in room + one item's cost takes all of it. With
        // `balance` only the XCD's last worker does: with a few one-chunk requests the rule leaves
        // the last column of every XCD idle and doubles its neighbour.
        auto column_end = [&](int lo, double cap, bool last) {
            std::size_t s = 0;
            while(s < segments.size() && segments[s].end <= lo)
                ++s;
            double cost = 0.0;
            int pos     = lo;
            for(; s < segments.size(); ++s)
            {
                const double room = cap - cost - item_cost;
                if(room < 1.0)
                    break;
                const int avail = segments[s].end - pos;
                if(avail > room + ((balance && !last) ? 0.0 : item_cost))
                {
                    const int take = static_cast<int>(room);
                    // balance: never leave a one-tile remainder. A request's last plan tile can
                    // hold fewer than q_len tokens (L % kv_tile), and a piece made of it alone has
                    // rows that see no key at all (token t sees up to L - q_len + t). Without
                    // `balance` the rule above never cuts within one item's cost of a segment end.
                    if(balance && avail - take < 2)
                    {
                        pos = segments[s].end;
                        break;
                    }
                    pos += take;
                    break;
                }
                cost += avail + item_cost;
                pos = segments[s].end;
            }
            return pos;
        };
        auto tail_cost = [&](int lo, int workers) {
            return static_cast<double>(nch) * (total_tiles - lo) +
                   item_cost * (nch * pieces(lo, total_tiles) + workers - 1);
        };
        auto feasible = [&](double cap, int columns) {
            int pos           = 0;
            const int workers = wg_per_xcd - columns * nch;
            for(int column = 0; column < columns; ++column)
                pos = column_end(pos, cap, workers == 0 && column + 1 == columns);
            if(workers == 0)
                return pos == total_tiles;
            return tail_cost(pos, workers) <= workers * cap;
        };
        auto min_cap = [&](int columns) {
            double lo = 0.0, hi = tail_cost(0, 1) + 1.0;
            for(int step = 0; step < 48; ++step)
            {
                const double mid                   = 0.5 * (lo + hi);
                (feasible(mid, columns) ? hi : lo) = mid;
            }
            return hi;
        };
        // always the most columns: their nch workers read the same KV tiles together (L2 hits),
        // tail workers do not (fewer columns can price lower here but cost ~10 points of L2 hit
        // rate)
        const int full_columns = std::min(wg_per_xcd / nch, total_tiles);
        const double cap       = min_cap(full_columns);

        const int extra_after = wg_per_xcd - full_columns * nch;
        int tail_begin        = 0;
        for(int column = 0; column < full_columns; ++column)
        {
            const int column_begin = tail_begin;
            tail_begin =
                column_end(column_begin, cap, extra_after == 0 && column + 1 == full_columns);
            for(int chunk = 0; chunk < nch; ++chunk)
                add(x + nxcd * (column * nch + chunk), chunk, column_begin, tail_begin);
        }

        struct Piece
        { int request, chunk, begin, end; };
        std::vector<Piece> tail;
        for(int chunk = 0; chunk < nch; ++chunk)
            for(const Segment& segment : segments)
                if(segment.end > tail_begin && chunk < nch_v[segment.request])
                    tail.push_back(
                        {segment.request,
                         chunk,
                         std::max(tail_begin, segment.begin) - segment.begin + segment.tile0,
                         segment.end - segment.begin + segment.tile0});
        const int extra_workers = wg_per_xcd - full_columns * nch;
        std::size_t p           = 0;
        int at                  = tail.empty() ? 0 : tail[0].begin;
        for(int extra = 0; extra < extra_workers && p < tail.size(); ++extra)
        {
            const int worker = x + nxcd * (full_columns * nch + extra);
            const bool last  = extra + 1 == extra_workers;
            double cost      = 0.0;
            while(p < tail.size())
            {
                const double room = cap - cost - item_cost;
                if(!last && room < 1.0)
                    break;
                const int avail = tail[p].end - at;
                const int take = last || avail <= room + item_cost ? avail : static_cast<int>(room);
                emit(worker, tail[p].request, tail[p].chunk, at, at + take);
                cost += take + item_cost;
                at += take;
                if(at == tail[p].end && ++p < tail.size())
                    at = tail[p].begin;
            }
        }
    }

    std::vector<PlanItem> flat;
    for(const auto& worker_items : per_wg)
        flat.insert(flat.end(), worker_items.begin(), worker_items.end());
    if(static_cast<int>(flat.size()) > item_cap)
        return -5;
    if(balance)
    {
        // every row of a piece must see a key: a piece that ends at its request's end must start
        // at or before L - q_len, with q_len that request's own token count
        for(const PlanItem& item : flat)
        {
            const int b     = item[0];
            const int q_len = q_indptr[b + 1] - q_indptr[b];
            if(item[5] == kv_indptr[b + 1] && item[4] > kv_indptr[b + 1] - q_len)
                return -7;
        }
    }

    std::vector<int> order(flat.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return std::tie(flat[a][0], flat[a][8], flat[a][9]) <
               std::tie(flat[b][0], flat[b][8], flat[b][9]);
    });
    for(int rank = 0; rank < static_cast<int>(order.size()); ++rank)
        flat[order[rank]][1] = rank;

    int* work_indptr = plan;
    int* work_items  = plan + nwg + 1;
    int* unit_info   = work_items + item_cap * 8;
    int item_offset  = 0;
    work_indptr[0]   = 0;
    for(int worker = 0; worker < nwg; ++worker)
    {
        item_offset += static_cast<int>(per_wg[worker].size());
        work_indptr[worker + 1] = item_offset;
    }
    for(int item = 0; item < static_cast<int>(flat.size()); ++item)
        std::copy_n(flat[item].begin(), 8, work_items + item * 8);

    // coff[b] + chunk, not b * nch: a request only has nch_v[b] chunks. With one depth
    // coff[b] == b * nch and the indexing, iteration order and output are unchanged.
    std::vector<UnitRange> ranges(coff[bs]);
    for(const PlanItem& item : flat)
    {
        UnitRange& range = ranges[coff[item[0]] + item[8]];
        range.begin      = std::min(range.begin, item[1]);
        range.end        = std::max(range.end, item[1] + 1);
    }
    // a unit of one piece writes out / lse from stage 1 and is not merged
    int unit_count = 0;
    for(int b = 0; b < bs; ++b)
        for(int chunk = 0; chunk < nch_v[b]; ++chunk)
        {
            const UnitRange& range = ranges[coff[b] + chunk];
            if(range.end < 0 || range.end - range.begin == 1)
                continue;
            unit_info[4 * unit_count]     = range.begin;
            unit_info[4 * unit_count + 1] = range.end;
            unit_info[4 * unit_count + 2] = q_indptr[b] * h + slot * chunk;
            unit_info[4 * unit_count + 3] = std::min(slot, nrows_v[b] - slot * chunk);
            ++unit_count;
        }
    for(int item = 0; item < static_cast<int>(flat.size()); ++item)
    {
        const UnitRange& range = ranges[coff[flat[item][0]] + flat[item][8]];
        if(range.end - range.begin == 1)
            work_items[item * 8 + 1] = -1;
    }

    *out_items = static_cast<int>(flat.size());
    *out_units = unit_count;
    return 0;
}

// Depth-bucketed planner: the same columns as build_plan_default_impl, but on a ragged step the
// requests of each XCD are grouped by chunk depth and the XCD's workers are apportioned to the
// groups by their real work (nch[b] * tiles[b]), so no worker idles above a shallow request's
// depth; XCD balance is weighted by depth too. Sizing columns by the batch max instead leaves
// every worker above a short request's own depth with no items at all, with the critical path
// pinned at its uniform value however much work the ragged batch removed. A batch with one chunk
// depth takes the uniform_depth path, which is the default plan byte for byte.
int build_plan_bucketed_impl(const int* kv_indptr,
                             const int* q_indptr,
                             int bs,
                             int h,
                             int nwg,
                             bool balance,
                             int nxcd,
                             int item_cap,
                             int max_units,
                             int* plan,
                             int plan_ints,
                             int* out_items,
                             int* out_units)
{
    constexpr int slot    = kSlot;
    constexpr int kv_tile = kKvTile;
    if(!kv_indptr || !q_indptr || !plan || !out_items || !out_units || bs < 0 || h <= 0 ||
       nwg <= 0 || nxcd <= 0 || (nwg % nxcd) != 0)
        return -1;
    const int required = nwg + 1 + item_cap * 8 + max_units * 4;
    if(plan_ints < required)
        return -2;

    std::fill(plan, plan + required, 0);
    *out_items = 0;
    *out_units = 0;
    if(bs == 0)
        return 0;

    std::vector<int> nrows_v(bs), nch_v(bs), coff(bs + 1, 0);
    for(int b = 0; b < bs; ++b)
    {
        const int q_len = q_indptr[b + 1] - q_indptr[b];
        if(q_len < 0)
            return -6;
        nrows_v[b]  = q_len * h;
        nch_v[b]    = (nrows_v[b] + slot - 1) / slot;
        coff[b + 1] = coff[b] + nch_v[b];
    }
    if(coff[bs] > max_units)
        return -3;
    const int nch = *std::max_element(nch_v.begin(), nch_v.end());
    if(nch <= 0)
        return -3;
    bool uniform_depth = true;
    for(int b = 0; b < bs; ++b)
        uniform_depth = uniform_depth && nch_v[b] == nch;

    std::vector<int> tiles(bs);
    for(int b = 0; b < bs; ++b)
    {
        const int len = kv_indptr[b + 1] - kv_indptr[b];
        if(len < 0)
            return -4;
        tiles[b] = (len + kv_tile - 1) / kv_tile;
    }

    // Longest-processing-time first on the real cost. A request verifying one token has depth 1
    // and costs a fifth of a six-token request with the same KV, so ordering and balancing on
    // tiles alone mis-sizes the XCDs under ragged. With a uniform depth the factor is constant
    // and both the order and the argmin are unchanged.
    std::vector<long long> cost(bs);
    for(int b = 0; b < bs; ++b)
        cost[b] = static_cast<long long>(tiles[b]) * (uniform_depth ? 1 : nch_v[b]);

    std::vector<int> request_order(bs);
    std::iota(request_order.begin(), request_order.end(), 0);
    std::stable_sort(request_order.begin(), request_order.end(), [&](int a, int b) {
        return cost[a] > cost[b];
    });

    std::vector<std::vector<Segment>> xseg(nxcd);
    std::vector<int> xtiles(nxcd, 0);
    std::vector<std::vector<std::array<int, 3>>> owners(nxcd); // {request, first tile, end tile}
    {
        std::vector<long long> xcd_load(nxcd, 0);
        for(const int b : request_order)
        {
            int x = 0;
            for(int candidate = 1; candidate < nxcd; ++candidate)
                if(xcd_load[candidate] < xcd_load[x])
                    x = candidate;
            owners[x].push_back({b, 0, tiles[b]});
            xcd_load[x] += cost[b];
        }
        const long long total   = std::accumulate(xcd_load.begin(), xcd_load.end(), 0LL);
        const long long busiest = *std::max_element(xcd_load.begin(), xcd_load.end());
        const int min_piece     = static_cast<int>(std::ceil(2.0 * kItemTiles * 64.0 / kv_tile));
        // Cutting a request across XCDs splits its tile range, which the bucket layout below
        // cannot express (a bucket's columns span one contiguous tile axis per request), so the
        // cut is only taken when the batch has a single depth.
        if(uniform_depth && kXcdSplit >= 0 && busiest * nxcd > total * (1.0 + kXcdSplit) &&
           busiest - total / nxcd > min_piece)
        {
            for(auto& requests : owners)
                requests.clear();
            int x         = 0;
            long long pos = 0;
            auto bound    = [&](int xcd) {
                return xcd + 1 == nxcd ? total : total * (xcd + 1) / nxcd;
            };
            for(const int b : request_order)
                for(int t = 0; t < tiles[b];)
                {
                    int take = static_cast<int>(std::min<long long>(tiles[b] - t, bound(x) - pos));
                    if(tiles[b] - t - take < min_piece)
                        take = tiles[b] - t;
                    else if(take < min_piece && x + 1 < nxcd)
                    {
                        ++x;
                        continue;
                    }
                    owners[x].push_back({b, t, t + take});
                    t += take;
                    pos += take;
                    while(x + 1 < nxcd && pos >= bound(x))
                        ++x;
                }
        }
    }
    for(int x = 0; x < nxcd; ++x)
    {
        std::sort(owners[x].begin(), owners[x].end());
        for(const auto& [b, first, end] : owners[x])
        {
            xseg[x].push_back({b, xtiles[x], xtiles[x] + end - first, first});
            xtiles[x] += end - first;
        }
    }

    std::vector<std::vector<PlanItem>> per_wg(nwg);
    const int wg_per_xcd = nwg / nxcd;
    // One layout pass over a set of equal-depth requests. With a uniform batch this runs once per
    // XCD over everything it owns, with `depth` the batch depth and `n_workers` all of them,
    // which is exactly build_plan_default_impl. Under ragged it runs once per depth on the XCD.
    auto layout = [&](int x,
                      const std::vector<Segment>& segments,
                      int total_tiles,
                      int depth,
                      int worker_base,
                      int n_workers) {
        if(total_tiles == 0 || n_workers <= 0 || depth <= 0)
            return;

        // The k-th of this group's workers. Two groups share a workgroup only when an XCD owns
        // more depths than it has workgroups, which costs balance but keeps the plan valid; with
        // a worker to spare per group the wrap never triggers.
        auto pick = [&](int k) { return x + nxcd * ((worker_base + k) % wg_per_xcd); };

        auto emit = [&](int worker, int b, int chunk, int tile_begin, int tile_end) {
            const int kv_len      = kv_indptr[b + 1] - kv_indptr[b];
            const int valid_rows  = std::min(slot, nrows_v[b] - chunk * slot);
            const int query_begin = q_indptr[b] * h + slot * chunk;
            per_wg[worker].push_back({b,
                                      0,
                                      query_begin,
                                      query_begin + valid_rows,
                                      kv_indptr[b] + kv_tile * tile_begin,
                                      kv_indptr[b] + std::min(kv_tile * tile_end, kv_len),
                                      0,
                                      0,
                                      chunk,
                                      tile_begin});
        };
        // every request in this group has `depth` chunks, so no chunk index is skipped
        auto add = [&](int worker, int chunk, int lo, int hi) {
            for(const Segment& segment : segments)
            {
                const int tile_begin = std::max(lo, segment.begin) - segment.begin + segment.tile0;
                const int tile_end   = std::min(hi, segment.end) - segment.begin + segment.tile0;
                if(tile_end > tile_begin)
                    emit(worker, segment.request, chunk, tile_begin, tile_end);
            }
        };

        const double item_cost = kItemTiles * 64.0 / kv_tile;
        auto pieces            = [&](int lo, int hi) {
            int n = 0;
            for(const Segment& segment : segments)
                n += std::min(hi, segment.end) > std::max(lo, segment.begin);
            return n;
        };
        auto column_end = [&](int lo, double cap, bool last) {
            std::size_t s = 0;
            while(s < segments.size() && segments[s].end <= lo)
                ++s;
            double cost_acc = 0.0;
            int pos         = lo;
            for(; s < segments.size(); ++s)
            {
                const double room = cap - cost_acc - item_cost;
                if(room < 1.0)
                    break;
                const int avail = segments[s].end - pos;
                if(avail > room + ((balance && !last) ? 0.0 : item_cost))
                {
                    const int take = static_cast<int>(room);
                    if(balance && avail - take < 2)
                    {
                        pos = segments[s].end;
                        break;
                    }
                    pos += take;
                    break;
                }
                cost_acc += avail + item_cost;
                pos = segments[s].end;
            }
            return pos;
        };
        auto tail_cost = [&](int lo, int workers) {
            return static_cast<double>(depth) * (total_tiles - lo) +
                   item_cost * (depth * pieces(lo, total_tiles) + workers - 1);
        };
        auto feasible = [&](double cap, int columns) {
            int pos           = 0;
            const int workers = n_workers - columns * depth;
            for(int column = 0; column < columns; ++column)
                pos = column_end(pos, cap, workers == 0 && column + 1 == columns);
            if(workers == 0)
                return pos == total_tiles;
            return tail_cost(pos, workers) <= workers * cap;
        };
        auto min_cap = [&](int columns) {
            double lo = 0.0, hi = tail_cost(0, 1) + 1.0;
            for(int step = 0; step < 48; ++step)
            {
                const double mid                   = 0.5 * (lo + hi);
                (feasible(mid, columns) ? hi : lo) = mid;
            }
            return hi;
        };
        const int full_columns = std::min(n_workers / depth, total_tiles);
        const double cap       = min_cap(full_columns);

        const int extra_workers = n_workers - full_columns * depth;
        int tail_begin          = 0;
        for(int column = 0; column < full_columns; ++column)
        {
            const int column_begin = tail_begin;
            tail_begin =
                column_end(column_begin, cap, extra_workers == 0 && column + 1 == full_columns);
            for(int chunk = 0; chunk < depth; ++chunk)
                add(pick(column * depth + chunk), chunk, column_begin, tail_begin);
        }

        struct Piece
        { int request, chunk, begin, end; };
        std::vector<Piece> tail;
        for(int chunk = 0; chunk < depth; ++chunk)
            for(const Segment& segment : segments)
                if(segment.end > tail_begin)
                    tail.push_back(
                        {segment.request,
                         chunk,
                         std::max(tail_begin, segment.begin) - segment.begin + segment.tile0,
                         segment.end - segment.begin + segment.tile0});
        std::size_t p = 0;
        int at        = tail.empty() ? 0 : tail[0].begin;
        for(int extra = 0; extra < extra_workers && p < tail.size(); ++extra)
        {
            const int worker = pick(full_columns * depth + extra);
            const bool last  = extra + 1 == extra_workers;
            double cost_acc  = 0.0;
            while(p < tail.size())
            {
                const double room = cap - cost_acc - item_cost;
                if(!last && room < 1.0)
                    break;
                const int avail = tail[p].end - at;
                const int take = last || avail <= room + item_cost ? avail : static_cast<int>(room);
                emit(worker, tail[p].request, tail[p].chunk, at, at + take);
                cost_acc += take + item_cost;
                at += take;
                if(at == tail[p].end && ++p < tail.size())
                    at = tail[p].begin;
            }
        }
    };

    for(int x = 0; x < nxcd; ++x)
    {
        if(xtiles[x] == 0)
            continue;
        if(uniform_depth)
        {
            layout(x, xseg[x], xtiles[x], nch, 0, wg_per_xcd);
            continue;
        }

        // A request verifying no token (q_len 0) has depth 0 and owns no rows, so it joins no
        // group; a depth-0 layout pass would divide by its depth.
        std::vector<int> depths;
        for(const Segment& segment : xseg[x])
            if(nch_v[segment.request] > 0 &&
               std::find(depths.begin(), depths.end(), nch_v[segment.request]) == depths.end())
                depths.push_back(nch_v[segment.request]);
        if(depths.empty())
            continue;
        std::sort(depths.begin(), depths.end());

        std::vector<long long> group_work(depths.size(), 0);
        long long work_sum = 0;
        for(std::size_t g = 0; g < depths.size(); ++g)
        {
            for(const Segment& segment : xseg[x])
                if(nch_v[segment.request] == depths[g])
                    group_work[g] +=
                        static_cast<long long>(segment.end - segment.begin) * depths[g];
            work_sum += group_work[g];
        }
        if(work_sum == 0)
            continue;

        // Largest-remainder apportionment with at least one worker per non-empty group.
        std::vector<int> alloc(depths.size(), 0);
        std::vector<double> exact(depths.size(), 0.0);
        int handed = 0;
        for(std::size_t g = 0; g < depths.size(); ++g)
        {
            exact[g] = static_cast<double>(wg_per_xcd) * group_work[g] / work_sum;
            alloc[g] = std::max(1, static_cast<int>(exact[g]));
            handed += alloc[g];
        }
        while(handed > wg_per_xcd)
        {
            std::size_t worst = 0;
            double over       = -1e300;
            for(std::size_t g = 0; g < depths.size(); ++g)
                if(alloc[g] > 1 && alloc[g] - exact[g] > over)
                {
                    over  = alloc[g] - exact[g];
                    worst = g;
                }
            if(over == -1e300) // every group down to its floor of one
                break;
            --alloc[worst];
            --handed;
        }
        while(handed < wg_per_xcd)
        {
            std::size_t best = 0;
            double under     = -1e300;
            for(std::size_t g = 0; g < depths.size(); ++g)
                if(exact[g] - alloc[g] > under)
                {
                    under = exact[g] - alloc[g];
                    best  = g;
                }
            ++alloc[best];
            ++handed;
        }

        int base = 0;
        for(std::size_t g = 0; g < depths.size(); ++g)
        {
            // Re-walk this group's tile axis so its segment offsets start at zero, which is what
            // the column bisection indexes against.
            std::vector<Segment> segments;
            int acc = 0;
            for(const Segment& segment : xseg[x])
                if(nch_v[segment.request] == depths[g])
                {
                    const int span = segment.end - segment.begin;
                    segments.push_back({segment.request, acc, acc + span, segment.tile0});
                    acc += span;
                }
            layout(x, segments, acc, depths[g], base, alloc[g]);
            base += alloc[g];
        }
    }

    std::vector<PlanItem> flat;
    for(const auto& worker_items : per_wg)
        flat.insert(flat.end(), worker_items.begin(), worker_items.end());
    if(static_cast<int>(flat.size()) > item_cap)
        return -5;
    if(balance)
    {
        for(const PlanItem& item : flat)
        {
            const int b     = item[0];
            const int q_len = q_indptr[b + 1] - q_indptr[b];
            if(item[5] == kv_indptr[b + 1] && item[4] > kv_indptr[b + 1] - q_len)
                return -7;
        }
    }

    std::vector<int> order(flat.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return std::tie(flat[a][0], flat[a][8], flat[a][9]) <
               std::tie(flat[b][0], flat[b][8], flat[b][9]);
    });
    for(int rank = 0; rank < static_cast<int>(order.size()); ++rank)
        flat[order[rank]][1] = rank;

    int* work_indptr = plan;
    int* work_items  = plan + nwg + 1;
    int* unit_info   = work_items + item_cap * 8;
    int item_offset  = 0;
    work_indptr[0]   = 0;
    for(int worker = 0; worker < nwg; ++worker)
    {
        item_offset += static_cast<int>(per_wg[worker].size());
        work_indptr[worker + 1] = item_offset;
    }
    for(int item = 0; item < static_cast<int>(flat.size()); ++item)
        std::copy_n(flat[item].begin(), 8, work_items + item * 8);

    std::vector<UnitRange> ranges(coff[bs]);
    for(const PlanItem& item : flat)
    {
        UnitRange& range = ranges[coff[item[0]] + item[8]];
        range.begin      = std::min(range.begin, item[1]);
        range.end        = std::max(range.end, item[1] + 1);
    }
    int unit_count = 0;
    for(int b = 0; b < bs; ++b)
        for(int chunk = 0; chunk < nch_v[b]; ++chunk)
        {
            const UnitRange& range = ranges[coff[b] + chunk];
            if(range.end < 0 || range.end - range.begin == 1)
                continue;
            unit_info[4 * unit_count]     = range.begin;
            unit_info[4 * unit_count + 1] = range.end;
            unit_info[4 * unit_count + 2] = q_indptr[b] * h + slot * chunk;
            unit_info[4 * unit_count + 3] = std::min(slot, nrows_v[b] - slot * chunk);
            ++unit_count;
        }
    for(int item = 0; item < static_cast<int>(flat.size()); ++item)
    {
        const UnitRange& range = ranges[coff[flat[item][0]] + flat[item][8]];
        if(range.end - range.begin == 1)
            work_items[item * 8 + 1] = -1;
    }

    *out_items = static_cast<int>(flat.size());
    *out_units = unit_count;
    return 0;
}

// Ragged entry: `bucketed` picks the depth-bucketed layout, otherwise the max-depth columns.
int build_plan_ragged(const int* kv_indptr,
                      const int* q_indptr,
                      int bs,
                      int h,
                      int nwg,
                      bool balance,
                      bool bucketed,
                      int nxcd,
                      int item_cap,
                      int max_units,
                      int* plan,
                      int plan_ints,
                      int* out_items,
                      int* out_units)
{
    if(bucketed)
    {
        const int rc = build_plan_bucketed_impl(kv_indptr,
                                                q_indptr,
                                                bs,
                                                h,
                                                nwg,
                                                balance,
                                                nxcd,
                                                item_cap,
                                                max_units,
                                                plan,
                                                plan_ints,
                                                out_items,
                                                out_units);
        // More depth groups can mean more work items; past item_cap take the default layout
        // rather than fail the step (it rewrites the whole plan).
        if(rc != -5)
            return rc;
    }
    return build_plan_default_impl(kv_indptr,
                                   q_indptr,
                                   bs,
                                   h,
                                   nwg,
                                   balance,
                                   nxcd,
                                   item_cap,
                                   max_units,
                                   plan,
                                   plan_ints,
                                   out_items,
                                   out_units);
}

// Uniform-width entry, kept so existing callers are unchanged. It always takes the default
// planner: the bucketed one plans a single depth identically, and this keeps the uniform plan
// independent of the switch.
int build_plan(const int* kv_indptr,
               int bs,
               int q_len,
               int h,
               int nwg,
               bool balance,
               int nxcd,
               int item_cap,
               int max_units,
               int* plan,
               int plan_ints,
               int* out_items,
               int* out_units)
{
    if(q_len <= 0 || bs < 0)
        return -1;
    std::vector<int> q_indptr(static_cast<std::size_t>(bs) + 1);
    for(int b = 0; b <= bs; ++b)
        q_indptr[b] = b * q_len;
    return build_plan_default_impl(kv_indptr,
                                   q_indptr.data(),
                                   bs,
                                   h,
                                   nwg,
                                   balance,
                                   nxcd,
                                   item_cap,
                                   max_units,
                                   plan,
                                   plan_ints,
                                   out_items,
                                   out_units);
}

const char* plan_error(int rc)
{
    switch(rc)
    {
    case -1: return "invalid arguments";
    case -2: return "the plan buffer is too small";
    case -3: return "more (request, chunk) units than max_units";
    case -4: return "kv_indptr is not non-decreasing";
    case -5: return "more work items than item_cap";
    case -6: return "q_indptr is not non-decreasing";
    case -7: return "a piece would have rows that see no key";
    case -8: return "unknown planner";
    default: return "unknown error";
    }
}

// ---- merge
// ---------------------------------------------------------------------------------------

__device__ __forceinline__ uint16_t f2bf(float f)
{
    uint32_t b = __float_as_uint(f);
    return (uint16_t)((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16);
}

__device__ __forceinline__ void bf16x8(const uint4 raw, float (&v)[8])
{
    const unsigned u[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
    for(int i = 0; i < 4; ++i)
    {
        v[2 * i]     = __uint_as_float(u[i] << 16);
        v[2 * i + 1] = __uint_as_float(u[i] & 0xFFFF0000u);
    }
}

// One wave per output row (64 lanes x 8 columns), the loads of up to RED_MAXP pieces issued before
// the math, RED_ROWS rows per wave. Partials are normalised by their own row sum; the merge uses
// the usual log-sum-exp weights. Blocks stride over the units and stop at the first empty one.
constexpr int RED_MAXP = 4, RED_WAVES = 4, RED_ROWS = 2;
constexpr int RED_THREADS = 64 * RED_WAVES, RED_ROW_BLOCKS = kSlot / (RED_WAVES * RED_ROWS);

__global__ __launch_bounds__(64 * RED_WAVES) void opus_mla_flat_reduce_kernel(
    const void* __restrict__ o_acc_v,
    const float* __restrict__ lse_acc,
    const int* __restrict__ chunk_info,
    uint16_t* __restrict__ out,
    float* __restrict__ lse_out,
    int nchunks)
{
    constexpr int SLOT_ROWS = kSlot;
    const uint16_t* o_acc   = static_cast<const uint16_t*>(o_acc_v);
    const int lane          = threadIdx.x & 63;
    const int r0            = (blockIdx.x * RED_WAVES + (threadIdx.x >> 6)) * RED_ROWS;
    for(int c = blockIdx.y; c < nchunks; c += gridDim.y)
    {
        const int p0 = chunk_info[4 * c], p1 = chunk_info[4 * c + 1], orow0 = chunk_info[4 * c + 2],
                  nrows = chunk_info[4 * c + 3];
        if(nrows == 0)
            break;
#pragma unroll
        for(int rr = 0; rr < RED_ROWS; ++rr)
        {
            const int r = r0 + rr;
            if(r >= nrows)
                break;
            // pieces in batches of RED_MAXP (all loads of a batch in flight), running max across
            // batches
            float mx = -3.0e38f, s = 0.f, acc[8] = {};
            for(int pb = p0; pb < p1; pb += RED_MAXP)
            {
                float l[RED_MAXP];
                uint4 raw[RED_MAXP];
#pragma unroll
                for(int i = 0; i < RED_MAXP; ++i)
                    if(pb + i < p1)
                    {
                        const size_t row = static_cast<size_t>(pb + i) * SLOT_ROWS + r;
                        l[i]             = lse_acc[row];
                        raw[i] = *reinterpret_cast<const uint4*>(o_acc + row * 512 + 8 * lane);
                    }
                float m = mx;
#pragma unroll
                for(int i = 0; i < RED_MAXP; ++i)
                    if(pb + i < p1)
                        m = fmaxf(m, l[i]);
                const float scale = __expf(mx - m);
                s *= scale;
#pragma unroll
                for(int j = 0; j < 8; ++j)
                    acc[j] *= scale;
                mx = m;
#pragma unroll
                for(int i = 0; i < RED_MAXP; ++i)
                    if(pb + i < p1)
                    {
                        const float w = __expf(l[i] - mx);
                        float v[8];
                        bf16x8(raw[i], v);
                        s += w;
#pragma unroll
                        for(int j = 0; j < 8; ++j)
                            acc[j] += w * v[j];
                    }
            }
            const float inv = 1.f / s;
            uint4 pk;
            pk.x = f2bf(acc[0] * inv) | (static_cast<uint32_t>(f2bf(acc[1] * inv)) << 16);
            pk.y = f2bf(acc[2] * inv) | (static_cast<uint32_t>(f2bf(acc[3] * inv)) << 16);
            pk.z = f2bf(acc[4] * inv) | (static_cast<uint32_t>(f2bf(acc[5] * inv)) << 16);
            pk.w = f2bf(acc[6] * inv) | (static_cast<uint32_t>(f2bf(acc[7] * inv)) << 16);
            *reinterpret_cast<uint4*>(out + static_cast<size_t>(orow0 + r) * 512 + 8 * lane) = pk;
            if(lane == 0)
                lse_out[orow0 + r] = mx + logf(s);
        }
    }
}

// The same merge with RED2_MAXP (16) pieces in flight per batch and one output row per wave, for
// units of many pieces and few rows (a 4-deep, 2-rows-per-wave merge of ~64 pieces is a ~32
// round-trip chain). Past the last piece a batch re-loads the last piece (same lines, cache hits)
// instead of branching.
constexpr int RED2_MAXP = 16, RED2_WAVES = 4, RED2_ROW_BLOCKS = kSlot / RED2_WAVES;

__global__ __launch_bounds__(64 * RED2_WAVES) void opus_mla_flat_reduce_deep_kernel(
    const void* __restrict__ o_acc_v,
    const float* __restrict__ lse_acc,
    const int* __restrict__ chunk_info,
    uint16_t* __restrict__ out,
    float* __restrict__ lse_out,
    int nchunks)
{
    constexpr int SLOT_ROWS = kSlot;
    const uint16_t* o_acc   = static_cast<const uint16_t*>(o_acc_v);
    const int lane          = threadIdx.x & 63;
    const int r             = blockIdx.x * RED2_WAVES + (threadIdx.x >> 6);
    for(int c = blockIdx.y; c < nchunks; c += gridDim.y)
    {
        const int p0 = chunk_info[4 * c], p1 = chunk_info[4 * c + 1], orow0 = chunk_info[4 * c + 2],
                  nrows = chunk_info[4 * c + 3];
        if(nrows == 0)
            break;
        if(r >= nrows)
            continue;
        float mx = -3.0e38f, s = 0.f, acc[8] = {};
        for(int pb = p0; pb < p1; pb += RED2_MAXP)
        {
            float l[RED2_MAXP];
            uint4 raw[RED2_MAXP];
#pragma unroll
            for(int i = 0; i < RED2_MAXP; ++i)
            {
                const size_t row = static_cast<size_t>(min(pb + i, p1 - 1)) * SLOT_ROWS + r;
                l[i]             = lse_acc[row];
                raw[i]           = *reinterpret_cast<const uint4*>(o_acc + row * 512 + 8 * lane);
            }
            float m = mx;
#pragma unroll
            for(int i = 0; i < RED2_MAXP; ++i)
                if(pb + i < p1)
                    m = fmaxf(m, l[i]);
            const float scale = __expf(mx - m);
            s *= scale;
#pragma unroll
            for(int j = 0; j < 8; ++j)
                acc[j] *= scale;
            mx = m;
#pragma unroll
            for(int i = 0; i < RED2_MAXP; ++i)
                if(pb + i < p1)
                {
                    const float w = __expf(l[i] - mx);
                    float v[8];
                    bf16x8(raw[i], v);
                    s += w;
#pragma unroll
                    for(int j = 0; j < 8; ++j)
                        acc[j] += w * v[j];
                }
        }
        const float inv = 1.f / s;
        uint4 pk;
        pk.x = f2bf(acc[0] * inv) | (static_cast<uint32_t>(f2bf(acc[1] * inv)) << 16);
        pk.y = f2bf(acc[2] * inv) | (static_cast<uint32_t>(f2bf(acc[3] * inv)) << 16);
        pk.z = f2bf(acc[4] * inv) | (static_cast<uint32_t>(f2bf(acc[5] * inv)) << 16);
        pk.w = f2bf(acc[6] * inv) | (static_cast<uint32_t>(f2bf(acc[7] * inv)) << 16);
        *reinterpret_cast<uint4*>(out + static_cast<size_t>(orow0 + r) * 512 + 8 * lane) = pk;
        if(lane == 0)
            lse_out[orow0 + r] = mx + logf(s);
    }
}

// Units in y: a unit's row blocks get consecutive ids (dealt round-robin over the XCDs) and the
// working blocks are dispatched before the empty ones.
constexpr int kReduceGridUnits = 256;

void check_tensor(
    const aiter_tensor_t* t, AiterDtype dtype, bool on_gpu, size_t min_numel, const char* name)
{
    AITER_CHECK(t != nullptr && t->dtype() == dtype && t->is_contiguous() && t->is_gpu() == on_gpu,
                kFn,
                ": ",
                name,
                " must be a contiguous ",
                AiterDtype_to_str(dtype),
                on_gpu ? " GPU" : " host",
                " tensor.");
    AITER_CHECK(t->numel() >= min_numel,
                kFn,
                ": ",
                name,
                " holds ",
                t->numel(),
                " elements, needs ",
                min_numel);
}

} // namespace

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    opus_mla_decode_fwd,
    (aiter_tensor_t * q,            // [total_q, H, 576] bf16 or fp8 (merged nope+rope)
     aiter_tensor_t* kv,            // [total_tokens, ..., 576], same dtype as q, page_size == 1
     aiter_tensor_t* qo_indptr,     // [B+1]
     aiter_tensor_t* kv_indptr,     // [B+1]
     aiter_tensor_t* kv_indices,    // [num_page_used]
     aiter_tensor_t* work_indptr,   // metadata
     aiter_tensor_t* work_info_set, // metadata
     int page_size,
     float softmax_scale,
     aiter_tensor_t* logits,    // split_output [num_partials, 1, H, 512] fp32
     aiter_tensor_t* attn_lse,  // split_lse    [num_partials, 1, H, 1]   fp32
     aiter_tensor_t* out,       // final [total_q, H, 512] bf16
     aiter_tensor_t* final_lse, // [total_q, H] fp32 (nullable)
     // Everything below was added when the fp8 entry point folded into this one; the bf16
     // arguments above keep their old positions.
     aiter_tensor_t* q_scale,   // fp8: float[1] per-tensor descale; bf16: unused (nullable)
     aiter_tensor_t* kv_scale,  // fp8: float[1] per-tensor descale; bf16: unused (nullable)
     int max_seqlen_q,          // fp8: required, picks the build; bf16: unused
     bool causal,               // fp8: apply the causal mask across the query tokens
     aiter_tensor_t* g_kv_indptr, // fp8 round-robin CP: GLOBAL per-request kv_indptr (nullable)
     int cp_world_size,           // fp8 round-robin CP: ranks; 1 = no CP
     int cp_rank,                 // fp8 round-robin CP: this rank, global pos = j * W + rank
     hipStream_t stream),
    (q,
     kv,
     qo_indptr,
     kv_indptr,
     kv_indices,
     work_indptr,
     work_info_set,
     page_size,
     softmax_scale,
     logits,
     attn_lse,
     out,
     final_lse,
     q_scale,
     kv_scale,
     max_seqlen_q,
     causal,
     g_kv_indptr,
     cp_world_size,
     cp_rank,
     stream))
{
    const std::string gfx = get_gpu_arch();
    AITER_CHECK(
        gfx == "gfx950", __func__, ": unsupported GPU arch '", gfx, "' (supported: gfx950).");
    AITER_CHECK(page_size == 1, __func__, ": only page_size == 1 is supported, got ", page_size);
    AITER_CHECK(q->dtype() == kv->dtype(), __func__, ": q and kv must have the same dtype.");
    AITER_CHECK(q->size(-1) == kHeadDimQk, __func__, ": q last dim must be ", kHeadDimQk, ".");
    AITER_CHECK(kv->size(-1) == kHeadDimQk, __func__, ": kv last dim must be ", kHeadDimQk, ".");
    AITER_CHECK(out->size(-1) == kHeadDimVo, __func__, ": out last dim must be ", kHeadDimVo, ".");

    const int H           = q->size(1);
    const int num_workers = work_indptr->size(0) - 1;
    const HipDeviceGuard device_guard(q->device_id);

    // Merged d=576 buffers: one row per (token, head); rope is the +512 slice.
    opus_mla_decode_kargs kargs{};
    kargs.q_ptr     = q->data_ptr();
    kargs.kv_ptr    = kv->data_ptr();
    kargs.out_ptr   = out->data_ptr();
    kargs.lse_ptr   = (final_lse && final_lse->numel() > 0) ? final_lse->data_ptr() : nullptr;
    kargs.o_accum   = logits->data_ptr();
    kargs.lse_accum = attn_lse->data_ptr();
    kargs.q_indptr      = static_cast<const int*>(qo_indptr->data_ptr());
    kargs.kv_indptr     = static_cast<const int*>(kv_indptr->data_ptr());
    kargs.kv_indices    = static_cast<const int*>(kv_indices->data_ptr());
    kargs.work_indptr   = static_cast<const int*>(work_indptr->data_ptr());
    kargs.work_info_set =
        static_cast<const opus_mla_decode_work_info*>(work_info_set->data_ptr());
    kargs.H              = H;
    kargs.total_tokens   = static_cast<int>(kv->numel() / kHeadDimQk);
    kargs.softmax_scale  = softmax_scale;
    kargs.stride_q_b     = H * kHeadDimQk;
    kargs.stride_q_h     = kHeadDimQk;
    kargs.stride_o_b     = H * kHeadDimVo;
    kargs.stride_o_h     = kHeadDimVo;
    kargs.stride_kv_page = kHeadDimQk;
    kargs.cp_world_size  = cp_world_size > 1 ? cp_world_size : 1;
    if(kargs.cp_world_size > 1)
    {
        AITER_CHECK(q->dtype() == AITER_DTYPE_fp8,
                    __func__,
                    ": round-robin CP is supported by the fp8 builds only.");
        AITER_CHECK(g_kv_indptr != nullptr && g_kv_indptr->numel() > 0,
                    __func__,
                    ": round-robin CP needs g_kv_indptr.");
        AITER_CHECK(cp_rank >= 0 && cp_rank < cp_world_size,
                    __func__,
                    ": cp_rank ",
                    cp_rank,
                    " out of range for cp_world_size ",
                    cp_world_size);
        kargs.g_kv_indptr = static_cast<const int*>(g_kv_indptr->data_ptr());
        kargs.cp_rank     = cp_rank;
    }

    if(q->dtype() == AITER_DTYPE_bf16)
        opus_mla_decode_bf16(kargs, num_workers, stream);
    else if(q->dtype() == AITER_DTYPE_fp8)
        opus_mla_decode_a8w8(
            kargs, q_scale, kv_scale, max_seqlen_q, causal, num_workers, stream);
    else
        AITER_CHECK(false, __func__, ": q/kv must be bf16 or fp8.");
}

// 1 if opus_mla_decode_fwd has a build for this shape (nhead as it will see it); bf16 has one
// for every head count.
AITER_CTYPES_DEFINE_ENTRYPOINT(opus_mla_decode_supported,
                               (bool fp8, int nhead, int max_seqlen_q, hipStream_t stream),
                               (fp8, nhead, max_seqlen_q, stream))
{
    return !fp8 || opus_a8w8_route(nhead, max_seqlen_q) != OpusA8W8Route::kNone;
}

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    opus_mla_decode_mxfp8_fwd,
    (aiter_tensor_t * q_nope,   // [total_q, H, 512]         fp8
     aiter_tensor_t* q_scale,   // [total_q, H, D_SCALE]     uint8 (E8M0)
     aiter_tensor_t* q_rope,    // [total_q, H, 64]          bf16
     aiter_tensor_t* kv_nope,   // [total_tokens, 512]       fp8
     aiter_tensor_t* kv_scale,  // [total_tokens, D_SCALE]   uint8 (E8M0)
     aiter_tensor_t* kv_rope,   // [total_tokens, 64]        bf16
     aiter_tensor_t* qo_indptr,
     aiter_tensor_t* kv_indptr,
     aiter_tensor_t* kv_indices,
     aiter_tensor_t* work_indptr,
     aiter_tensor_t* work_info_set,
     int page_size,
     float softmax_scale,
     aiter_tensor_t* logits,
     aiter_tensor_t* attn_lse,
     aiter_tensor_t* out,
     aiter_tensor_t* final_lse,
     hipStream_t stream),
    (q_nope,
     q_scale,
     q_rope,
     kv_nope,
     kv_scale,
     kv_rope,
     qo_indptr,
     kv_indptr,
     kv_indices,
     work_indptr,
     work_info_set,
     page_size,
     softmax_scale,
     logits,
     attn_lse,
     out,
     final_lse,
     stream))
{
    using T = MxFp8Traits;
    const std::string gfx = get_gpu_arch();
    AITER_CHECK(
        gfx == "gfx950", __func__, ": unsupported GPU arch '", gfx, "' (supported: gfx950).");
    AITER_CHECK(page_size == 1, __func__, ": only page_size == 1 is supported.");
    AITER_CHECK(q_nope->dtype() == AITER_DTYPE_fp8 && kv_nope->dtype() == AITER_DTYPE_fp8,
                __func__,
                ": q_nope/kv_nope must be fp8.");
    AITER_CHECK(q_rope->dtype() == AITER_DTYPE_bf16 && kv_rope->dtype() == AITER_DTYPE_bf16,
                __func__,
                ": q_rope/kv_rope must be bf16.");
    // The kernel reads the scales as bit_cast<float>(e8m0 << 23); fp32 factors must be
    // converted on the host first.
    AITER_CHECK(q_scale->dtype() == AITER_DTYPE_u8 && kv_scale->dtype() == AITER_DTYPE_u8,
                __func__,
                ": q_scale/kv_scale must be E8M0 uint8.");
    AITER_CHECK(kv_scale->size(-1) == T::D_SCALE_SIZE,
                __func__,
                ": kv_scale last dim must be ",
                T::D_SCALE_SIZE,
                ".");

    const int H           = q_nope->size(1);
    const int total_tokens = kv_nope->size(0);
    const int num_workers = work_indptr->size(0) - 1;

    const HipDeviceGuard device_guard(q_nope->device_id);

    opus_mla_decode_mxfp8_kargs kargs{};
    kargs.q_nope_ptr    = q_nope->data_ptr();
    kargs.q_scale_ptr   = q_scale->data_ptr();
    kargs.q_rope_ptr    = q_rope->data_ptr();
    kargs.kv_nope_ptr   = kv_nope->data_ptr();
    kargs.kv_scale_ptr  = kv_scale->data_ptr();
    kargs.kv_rope_ptr   = kv_rope->data_ptr();
    kargs.out_ptr   = out->data_ptr();
    kargs.lse_ptr   = (final_lse && final_lse->numel() > 0) ? final_lse->data_ptr() : nullptr;
    kargs.o_accum   = logits->data_ptr();
    kargs.lse_accum = attn_lse->data_ptr();
    kargs.q_indptr      = static_cast<const int*>(qo_indptr->data_ptr());
    kargs.kv_indptr     = static_cast<const int*>(kv_indptr->data_ptr());
    kargs.kv_indices    = static_cast<const int*>(kv_indices->data_ptr());
    kargs.work_indptr   = static_cast<const int*>(work_indptr->data_ptr());
    kargs.work_info_set =
        static_cast<const opus_mla_decode_work_info*>(work_info_set->data_ptr());
    kargs.H             = H;
    kargs.total_tokens  = total_tokens;
    kargs.softmax_scale = softmax_scale;
    kargs.stride_q_nope_b      = H * T::D_NOPE_SIZE;
    kargs.stride_q_nope_h      = T::D_NOPE_SIZE;
    kargs.stride_q_scale_b     = H * T::D_SCALE_SIZE;
    kargs.stride_q_scale_h     = T::D_SCALE_SIZE;
    kargs.stride_q_rope_b      = H * T::D_ROPE_SIZE;
    kargs.stride_q_rope_h      = T::D_ROPE_SIZE;
    kargs.stride_o_b           = H * T::D_NOPE_SIZE;
    kargs.stride_o_h           = T::D_NOPE_SIZE;
    kargs.stride_kv_nope_page  = T::D_NOPE_SIZE;
    kargs.stride_kv_scale_page = T::D_SCALE_SIZE;
    kargs.stride_kv_rope_page  = T::D_ROPE_SIZE;

    opus_mla_decode_mxfp8_16mx8_32nx1_kernel<T>
        <<<dim3(num_workers, 1, 1), dim3(T::BLOCK_SIZE), 0, stream>>>(kargs);
}

// Builds the plan into `plan` (host int32) from the host kv_indptr. stats (host int32 [2]) gets
// {work items, merge units}.
AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    opus_mla_decode_flat_plan,
    (aiter_tensor_t * kv_indptr, // [B+1] int32, host, rank-local
     aiter_tensor_t* qo_indptr,  // [B+1] int32, host; null for a uniform q_len batch
     int q_len,                  // query tokens per request, ignored when qo_indptr is given
     int num_heads,              // query heads
     int num_workers,            // stage-1 workgroups (the plan's work indptr has num_workers + 1)
     bool balance,               // only an XCD's last worker absorbs a remainder
     bool bucketed,              // ragged: group an XCD's requests by chunk depth
     int num_xcd,
     int item_cap,
     int max_units,
     aiter_tensor_t* plan,  // int32, host
     aiter_tensor_t* stats, // int32 [2], host
     hipStream_t stream),
    (kv_indptr,
     qo_indptr,
     q_len,
     num_heads,
     num_workers,
     balance,
     bucketed,
     num_xcd,
     item_cap,
     max_units,
     plan,
     stats,
     stream))
{
    (void)stream;
    AITER_CHECK(kv_indptr != nullptr && kv_indptr->numel() >= 1, kFn, ": kv_indptr is empty");
    check_tensor(kv_indptr, AITER_DTYPE_i32, false, 1, "kv_indptr");
    const int required = num_workers + 1 + item_cap * 8 + max_units * 4;
    check_tensor(plan, AITER_DTYPE_i32, false, required, "plan");
    check_tensor(stats, AITER_DTYPE_i32, false, 2, "stats");
    const int bs = static_cast<int>(kv_indptr->numel()) - 1;
    int* st      = static_cast<int*>(stats->data_ptr());
    int rc       = 0;
    if(qo_indptr != nullptr)
    {
        check_tensor(qo_indptr, AITER_DTYPE_i32, false, bs + 1, "qo_indptr");
        rc = build_plan_ragged(static_cast<const int*>(kv_indptr->data_ptr()),
                               static_cast<const int*>(qo_indptr->data_ptr()),
                               bs,
                               num_heads,
                               num_workers,
                               balance,
                               bucketed,
                               num_xcd,
                               item_cap,
                               max_units,
                               static_cast<int*>(plan->data_ptr()),
                               static_cast<int>(plan->numel()),
                               st,
                               st + 1);
    }
    else
    {
        rc = build_plan(static_cast<const int*>(kv_indptr->data_ptr()),
                        bs,
                        q_len,
                        num_heads,
                        num_workers,
                        balance,
                        num_xcd,
                        item_cap,
                        max_units,
                        static_cast<int*>(plan->data_ptr()),
                        static_cast<int>(plan->numel()),
                        st,
                        st + 1);
    }
    AITER_CHECK(rc == 0,
                kFn,
                ": planner failed (",
                rc,
                ": ",
                plan_error(rc),
                ") at bs ",
                bs,
                ", q_len ",
                q_len,
                ", heads ",
                num_heads);
}

// Stage 1 on the plan, then the merge of the split units. The merge grid covers max_units units
// whether or not the plan uses them, so a captured graph replays with any later plan.
AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    opus_mla_decode_flat_fwd,
    (aiter_tensor_t * q,          // [total_q, H, 576] fp8, contiguous
     aiter_tensor_t* kv,          // [total_tokens, 576] fp8, page_size 1
     aiter_tensor_t* qo_indptr,   // [B+1]
     aiter_tensor_t* kv_indptr,   // [B+1], rank-local
     aiter_tensor_t* kv_indices,  // [num_page_used]
     aiter_tensor_t* work_indptr, // plan section: [num_workers + 1]
     aiter_tensor_t* work_info,   // plan section: [item_cap * 8]
     aiter_tensor_t* unit_info,   // plan section: [max_units * 4]
     aiter_tensor_t* o_accum,     // [item_cap * 128, 512] bf16 partials
     aiter_tensor_t* lse_accum,   // [item_cap * 128] fp32
     aiter_tensor_t* out,         // [total_q, H, 512] bf16, contiguous
     aiter_tensor_t* lse,         // [total_q, H] fp32
     aiter_tensor_t* q_scale,     // float[1] per-tensor descale
     aiter_tensor_t* kv_scale,    // float[1] per-tensor descale
     float softmax_scale,
     bool causal,
     aiter_tensor_t* g_kv_indptr, // round-robin CP: global kv indptr (nullable without CP)
     int cp_world_size,           // round-robin CP: ranks; 1 = no CP
     int cp_rank,                 // round-robin CP: this rank, global pos = j * W + rank
     bool deep_reduce,            // 16 pieces in flight per merge round trip
     hipStream_t stream),
    (q,
     kv,
     qo_indptr,
     kv_indptr,
     kv_indices,
     work_indptr,
     work_info,
     unit_info,
     o_accum,
     lse_accum,
     out,
     lse,
     q_scale,
     kv_scale,
     softmax_scale,
     causal,
     g_kv_indptr,
     cp_world_size,
     cp_rank,
     deep_reduce,
     stream))
{
    AITER_CHECK(get_gpu_arch() == "gfx950", kFn, ": gfx950 only");
    AITER_CHECK(q != nullptr && q->dim() == 3 && q->size(2) == kHeadDimQk &&
                    q->dtype() == AITER_DTYPE_fp8 && q->is_contiguous(),
                kFn,
                ": q must be a contiguous fp8 [total_q, H, 576] tensor");
    const int64_t total_q = q->size(0);
    const int H           = static_cast<int>(q->size(1));
    AITER_CHECK(kv != nullptr && kv->dtype() == AITER_DTYPE_fp8 && kv->is_contiguous() &&
                    kv->size(-1) == kHeadDimQk,
                kFn,
                ": kv must be a contiguous fp8 [total_tokens, 576] tensor (page_size 1)");
    const int64_t total_tokens = static_cast<int64_t>(kv->numel()) / kHeadDimQk;
    // KV is addressed through one 32-bit buffer descriptor
    AITER_CHECK(total_tokens * kHeadDimQk < (int64_t(1) << 32),
                kFn,
                ": the KV cache must be smaller than 4 GiB, got ",
                total_tokens * kHeadDimQk,
                " bytes");
    check_tensor(qo_indptr, AITER_DTYPE_i32, true, 1, "qo_indptr");
    check_tensor(kv_indptr, AITER_DTYPE_i32, true, 1, "kv_indptr");
    check_tensor(kv_indices, AITER_DTYPE_i32, true, 0, "kv_indices");
    check_tensor(work_indptr, AITER_DTYPE_i32, true, 2, "work_indptr");
    check_tensor(work_info, AITER_DTYPE_i32, true, 8, "work_info");
    check_tensor(unit_info, AITER_DTYPE_i32, true, 4, "unit_info");
    const int num_workers = static_cast<int>(work_indptr->numel()) - 1;
    const int item_cap    = static_cast<int>(work_info->numel() / 8);
    const int max_units   = static_cast<int>(unit_info->numel() / 4);
    check_tensor(o_accum, AITER_DTYPE_bf16, true, size_t(item_cap) * kSlot * kHeadDimVo, "o_accum");
    check_tensor(lse_accum, AITER_DTYPE_fp32, true, size_t(item_cap) * kSlot, "lse_accum");
    check_tensor(out, AITER_DTYPE_bf16, true, size_t(total_q) * H * kHeadDimVo, "out");
    check_tensor(lse, AITER_DTYPE_fp32, true, size_t(total_q) * H, "lse");
    check_tensor(q_scale, AITER_DTYPE_fp32, true, 1, "q_scale");
    check_tensor(kv_scale, AITER_DTYPE_fp32, true, 1, "kv_scale");
    AITER_CHECK(cp_world_size >= 1 && cp_rank >= 0 && cp_rank < cp_world_size,
                kFn,
                ": bad cp_world_size / cp_rank");
    AITER_CHECK(
        cp_world_size == 1 || g_kv_indptr != nullptr, kFn, ": round-robin CP needs g_kv_indptr");

    opus_mla_decode_a8w8_flat_kargs k{};
    k.q_buffer_ptr   = q->data_ptr();
    k.q_scale_ptr    = q_scale->data_ptr();
    k.kv_buffer_ptr  = kv->data_ptr();
    k.kv_scale_ptr   = kv_scale->data_ptr();
    k.o_accum        = o_accum->data_ptr();
    k.lse_accum      = lse_accum->data_ptr();
    k.out_ptr        = out->data_ptr();
    k.lse_ptr        = lse->data_ptr();
    k.q_indptr       = static_cast<const int*>(qo_indptr->data_ptr());
    k.kv_indptr      = static_cast<const int*>(kv_indptr->data_ptr());
    k.kv_indices     = static_cast<const int*>(kv_indices->data_ptr());
    k.work_indptr    = static_cast<const int*>(work_indptr->data_ptr());
    k.work_info_set  = static_cast<const opus_mla_decode_work_info*>(work_info->data_ptr());
    k.H              = H;
    k.total_tokens   = static_cast<int>(total_tokens);
    k.softmax_scale  = softmax_scale;
    k.stride_q_b     = kHeadDimQk; // items address q / o by row (token * H + head)
    k.stride_o_b     = kHeadDimVo;
    k.stride_q_h     = kHeadDimQk;
    k.stride_o_h     = kHeadDimVo;
    k.stride_kv_page = kHeadDimQk;
    k.g_kv_indptr    = static_cast<const int*>(cp_world_size == 1 || g_kv_indptr == nullptr
                                                   ? kv_indptr->data_ptr()
                                                   : g_kv_indptr->data_ptr());
    k.cp_world_size  = cp_world_size;
    k.cp_rank        = cp_rank;
    k.wg_clock       = nullptr;

    if(causal)
        opus_mla_decode_a8w8_flat_32mx4_kernel<true>
            <<<dim3(num_workers), dim3(256), 0, stream>>>(k);
    else
        opus_mla_decode_a8w8_flat_32mx4_kernel<false>
            <<<dim3(num_workers), dim3(256), 0, stream>>>(k);
    HIP_CALL(hipGetLastError());

    if(max_units <= 0)
        return;
    const dim3 grid_units(1, std::min(max_units, kReduceGridUnits), 1);
    const void* o_acc  = o_accum->data_ptr();
    const float* l_acc = static_cast<const float*>(lse_accum->data_ptr());
    const int* units   = static_cast<const int*>(unit_info->data_ptr());
    uint16_t* o        = static_cast<uint16_t*>(out->data_ptr());
    float* l           = static_cast<float*>(lse->data_ptr());
    if(deep_reduce)
        opus_mla_flat_reduce_deep_kernel<<<dim3(RED2_ROW_BLOCKS, grid_units.y),
                                           dim3(64 * RED2_WAVES),
                                           0,
                                           stream>>>(o_acc, l_acc, units, o, l, max_units);
    else
        opus_mla_flat_reduce_kernel<<<dim3(RED_ROW_BLOCKS, grid_units.y),
                                      dim3(RED_THREADS),
                                      0,
                                      stream>>>(o_acc, l_acc, units, o, l, max_units);
    HIP_CALL(hipGetLastError());
}
