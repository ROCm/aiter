// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Opus fp8 MLA decode on flat-row plans (gfx950): a host planner, stage 1
// (opus/mla_decode_a8w8_flat_32mx4_64nx1.hpp) and the merge of its bf16 partials.
//
// q/o rows are token * H + head. The planner cuts each request into 128-row chunks and its KV into
// pieces, keeps a request's chunks on one XCD so they share KV through L2, and balances the XCDs
// and their workgroups by KV tiles. A (request, chunk) unit cut into one piece writes out / lse
// straight from stage 1; the others write 128-row bf16 partial slots, which the merge combines by
// LSE.
//
// The plan is one int32 host buffer, built on the CPU from the host kv_indptr and copied to the GPU
// by the caller (aiter/ops/opus/mla_flat.py):
//   [num_workers + 1 work indptr][item_cap * 8 work items][max_units * 4 merge units]
// A merge unit is {first partial slot, one past the last, first output row, rows}; the units in use
// come first and the rest are zero.

#include "aiter_tensor.h"

#include "aiter_ctypes_error.h"

#include "aiter_hip_common.h"
#include "opus/mla_decode_a8w8_flat_32mx4_64nx1.hpp"
#include "opus/mla_decode_kargs.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <limits>
#include <numeric>
#include <tuple>
#include <vector>

AITER_CTYPES_ERROR_DECL;

namespace {

constexpr int kSlot       = 128; // rows per chunk and per partial slot
constexpr int kKvTile     = 32;  // KV tokens per plan tile
constexpr int kHeadDimQk  = 576;
constexpr int kHeadDimVo  = 512;
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
    constexpr int slot    = kSlot;
    constexpr int kv_tile = kKvTile;
    if(!kv_indptr || !plan || !out_items || !out_units || bs < 0 || q_len <= 0 || h <= 0 ||
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

    const int nrows = q_len * h;
    const int nch   = (nrows + slot - 1) / slot;
    if(nch <= 0 || static_cast<long long>(bs) * nch > max_units)
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
            const int valid_rows  = std::min(slot, nrows - chunk * slot);
            const int query_begin = b * nrows + slot * chunk;
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
                if(segment.end > tail_begin)
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
        // at or before L - q_len
        for(const PlanItem& item : flat)
            if(item[5] == kv_indptr[item[0] + 1] && item[4] > kv_indptr[item[0] + 1] - q_len)
                return -7;
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

    std::vector<UnitRange> ranges(bs * nch);
    for(const PlanItem& item : flat)
    {
        UnitRange& range = ranges[item[0] * nch + item[8]];
        range.begin      = std::min(range.begin, item[1]);
        range.end        = std::max(range.end, item[1] + 1);
    }
    // a unit of one piece writes out / lse from stage 1 and is not merged
    int unit_count = 0;
    for(int b = 0; b < bs; ++b)
        for(int chunk = 0; chunk < nch; ++chunk)
        {
            const UnitRange& range = ranges[b * nch + chunk];
            if(range.end < 0 || range.end - range.begin == 1)
                continue;
            unit_info[4 * unit_count]     = range.begin;
            unit_info[4 * unit_count + 1] = range.end;
            unit_info[4 * unit_count + 2] = b * nrows + slot * chunk;
            unit_info[4 * unit_count + 3] = std::min(slot, nrows - slot * chunk);
            ++unit_count;
        }
    for(int item = 0; item < static_cast<int>(flat.size()); ++item)
    {
        const UnitRange& range = ranges[flat[item][0] * nch + flat[item][8]];
        if(range.end - range.begin == 1)
            work_items[item * 8 + 1] = -1;
    }

    *out_items = static_cast<int>(flat.size());
    *out_units = unit_count;
    return 0;
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
    case -7: return "a piece would have rows that see no key";
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

// Builds the plan into `plan` (host int32) from the host kv_indptr. stats (host int32 [2]) gets
// {work items, merge units}.
AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    opus_mla_decode_flat_plan,
    (aiter_tensor_t * kv_indptr, // [B+1] int32, host, rank-local
     int q_len,                  // query tokens per request
     int num_heads,              // query heads
     int num_workers,            // stage-1 workgroups (the plan's work indptr has num_workers + 1)
     bool balance,               // only an XCD's last worker absorbs a remainder
     int num_xcd,
     int item_cap,
     int max_units,
     aiter_tensor_t* plan,  // int32, host
     aiter_tensor_t* stats, // int32 [2], host
     hipStream_t stream),
    (kv_indptr,
     q_len,
     num_heads,
     num_workers,
     balance,
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
    const int rc = build_plan(static_cast<const int*>(kv_indptr->data_ptr()),
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
