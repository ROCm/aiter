// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Mesh-lending all-reduce for a TP group smaller than the xGMI mesh.
//
// On a full mesh a group of n GPUs talks over n - 1 of each GPU's links; the
// links to the GPUs outside the group are idle. An SDMA engine on such a GPU
// (a lender) can copy from one rank's memory into another rank's memory: the
// read crosses the lender's link to the source and the write crosses its link
// to the destination. No compute unit moves a byte across a link, nothing is
// staged on the lender and no kernel runs there.
//
// Each rank owns SDMA queues: one per destination on its own GPU and one on
// every lender. Per call the host writes COPY_LINEAR + FENCE + FENCE into
// them: the first fence stamps an arrival flag at the receiver, the second a
// departure flag at the sender. Kernels on the caller's stream wait for the
// flags of the pieces they touch, so everything queued behind them is ordered
// after the transport. The engines are not ordered with the stream, so the
// host synchronizes the stream before it posts.
//
// Two ranks: one shot. Each rank's whole input goes to the peer in 1 + L equal
// pieces, one per path, and one kernel adds. The peer's next call can start
// landing while this rank still adds, so two inboxes alternate by call
// parity. With bits = 4 or 8 each rank first packs its input (one signed E5M2
// scale byte per 16 values) and the engines move the packed buffer; the
// result is deq(own codes) + deq(peer codes), identical on both ranks.
//
// Four ranks: reduce-scatter, then all-gather. Rank j owns slice j. Phase 1
// sends slice j of every input to rank j, which sums it in rank order. Phase 2
// sends the sum back to every rank. A rank cannot post call k + 1 before
// every rank has consumed call k, so one set of inboxes is enough. The direct
// link carries 2 (n - 1) shares of a flow and each lender one: a lender's
// engine serves a rank's n - 1 destinations in turn, and in that time moved
// about half of what the engine of a direct link does (MI355X).
#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "hsakmt_sdma_queue.h"
#include "mesh_lend_all_reduce.h"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <algorithm>
#include <memory>
#include <stdexcept>

namespace aiter {
namespace {

constexpr int kMaxRanks      = 4;
constexpr int kMaxSlots      = kMaxRanks - 1; // the peers of one rank
constexpr int kMaxVia        = 8;             // own GPU + lenders
constexpr uint32_t kMaxNodes = 64;            // KFD topology nodes, as the bits of a mask
constexpr long kSpinLimit    = 2000000000L;
constexpr size_t kPage       = 4096;
constexpr size_t kVecBytes   = 16;   // exact payloads move in 16-byte vectors
constexpr size_t kVecAlign   = 256;  // vectors per 4 KiB: every piece starts on a page
constexpr size_t kUnitValues = 64;   // quantized payloads: four groups of 16 values
constexpr size_t kUnitAlign  = 1024; // units per piece: 36 and 68 KiB are page multiples
constexpr int kThreads       = 256;

enum FlagDir
{
    kArrival   = 0,
    kDeparture = 1
};

// A rank's block starts with its flag page. Flag of path `path` of the flow
// from or to peer slot `slot`, in ints.
__host__ __device__ inline int flag_index(int dir, int phase, int slot, int path)
{ return ((dir * 2 + phase) * kMaxSlots + slot) * kMaxVia + path; }

// Piece p of a flow covers units [bound[p], bound[p + 1]).
struct Split
{
    size_t bound[kMaxVia + 1];
    int npath;
};

__device__ inline float bf2f(uint16_t x)
{ return __builtin_bit_cast(float, static_cast<uint32_t>(x) << 16); }

__device__ inline uint16_t f2bf(float f)
{
    uint32_t u = __builtin_bit_cast(uint32_t, f);
    u += 0x7FFFu + ((u >> 16) & 1u); // round to nearest even
    return static_cast<uint16_t>(u >> 16);
}

// Eight bf16 from element e of a tensor of nelem. Elements past the end read
// as 0 and are not written. A caller's tensor is only 2-byte aligned.
__device__ inline void load8(uint16_t* v, const uint16_t* p, size_t e, size_t nelem)
{
    if(e + 8 <= nelem)
    {
        __builtin_memcpy(v, p + e, 16);
        return;
    }
    for(int k = 0; k < 8; ++k)
        v[k] = e + k < nelem ? p[e + k] : 0;
}

__device__ inline void store8(uint16_t* p, size_t e, size_t nelem, const uint16_t* v)
{
    if(e + 8 <= nelem)
    {
        __builtin_memcpy(p + e, v, 16);
        return;
    }
    for(int k = 0; k < 8; ++k)
        if(e + k < nelem)
            p[e + k] = v[k];
}

// Flags only grow, and a later fence in the same queue implies the earlier
// copy: compare with >=, not ==. The load races with the engine's write on
// purpose.
__device__ inline bool wait_flag(const int* flag, int epoch)
{
    for(long spins = 0; __atomic_load_n(flag, __ATOMIC_RELAXED) - epoch < 0;)
        if(++spins > kSpinLimit)
            return false;
    return true;
}

// Waits for the pieces of `s` that overlap units [lo, hi).
__device__ inline bool wait_pieces(
    const int* flags, int dir, int phase, int slot, const Split& s, size_t lo, size_t hi, int epoch)
{
    for(int p = 0; p < s.npath; ++p)
        if(s.bound[p + 1] > lo && s.bound[p] < hi &&
           !wait_flag(flags + flag_index(dir, phase, slot, p), epoch))
            return false;
    return true;
}

// One thread waits for its block and tells the others here. False if it timed
// out, which the host reads from err.
__device__ inline bool block_ok(bool ok, int* err)
{
    __shared__ bool shared;
    if(threadIdx.x == 0)
        shared = ok;
    __syncthreads();
    if(!shared)
        err[0] = 1;
    return shared;
}

__device__ inline void block_range(size_t n, size_t* lo, size_t* hi)
{
    const size_t per = (n + gridDim.x - 1) / gridDim.x;
    *lo              = std::min(blockIdx.x * per, n);
    *hi              = std::min(*lo + per, n);
}

// ---- two ranks, exact --------------------------------------------------------

struct PairArgs
{
    const uint16_t* inp;
    const uint16_t* inbox;
    uint16_t* out; // may be inp
    const int* flags;
    int* err;
    Split split; // in vectors
    size_t nelem;
    int epoch;
};

__global__ void mesh_lend_reduce_pair(PairArgs a)
{
    size_t lo, hi;
    block_range((a.nelem + 7) / 8, &lo, &hi);
    // A block needs its pieces to have arrived, and to have left: the peer
    // must hold inp before the block writes over it in place, or the caller
    // frees it.
    if(!block_ok(threadIdx.x != 0 ||
                     (wait_pieces(a.flags, kArrival, 0, 0, a.split, lo, hi, a.epoch) &&
                      wait_pieces(a.flags, kDeparture, 0, 0, a.split, lo, hi, a.epoch)),
                 a.err))
        return;
    for(size_t i = lo + threadIdx.x; i < hi; i += blockDim.x)
    {
        uint16_t x[8], w[8];
        load8(x, a.inp, 8 * i, a.nelem);
        __builtin_memcpy(w, a.inbox + 8 * i, 16);
        for(int j = 0; j < 8; ++j)
            x[j] = f2bf(bf2f(x[j]) + bf2f(w[j]));
        store8(a.out, 8 * i, a.nelem, x);
    }
}

// ---- two ranks, quantized ----------------------------------------------------
// Unit = 64 values = 4 groups of 16. On the wire: 2 * BITS words of codes and
// one word of four scale bytes (36 bytes at 4 bits, 68 at 8).
// Scale byte: sign << 7 | e5 << 2 | m2, magnitude (4 + m2) * 2^(e5 - 14). The
// sign is that of the group's extremum, which maps to the code -B.

struct EncodeArgs
{
    const uint16_t* inp;
    uint32_t* wire;
    size_t nelem;
};

struct DecodeArgs
{
    const uint32_t* own;
    const uint32_t* inbox;
    uint16_t* out;
    const int* flags;
    int* err;
    Split split; // in units
    size_t nelem;
    int epoch;
};

// E5M2 code of a magnitude. Rounding up keeps the extremum inside the range,
// which pays once the step is small (8 bits); at 4 bits a clipped extremum
// costs less than a step that is up to a quarter coarser.
__device__ inline uint32_t e5m2_of(float m, bool round_up)
{
    const uint32_t b = __builtin_bit_cast(uint32_t, m) & 0x7FFFFFFFu;
    int e            = static_cast<int>(b >> 23) - 127;
    uint32_t m2      = ((b & 0x7FFFFFu) + (round_up ? (1u << 21) - 1 : 1u << 20)) >> 21;
    if(m2 == 4)
    {
        m2 = 0;
        ++e;
    }
    int e5 = e + 12;
    if(e5 < 0)
    {
        e5 = 0;
        m2 = 0;
    }
    if(e5 > 31)
    {
        e5 = 31;
        m2 = 3;
    }
    return static_cast<uint32_t>(e5) << 2 | m2;
}

__device__ inline float pow2f(int e)
{ return __builtin_bit_cast(float, static_cast<uint32_t>(e + 127) << 23); }

template <int BITS>
__device__ inline float decode_step(uint32_t code)
{
    const float mag = static_cast<float>(4 + (code & 3)) *
                      pow2f(static_cast<int>((code >> 2) & 31) - 14 - (BITS - 1));
    return (code & 0x80) ? mag : -mag;
}

// 1 / decode_step from exact factors, so a host reference reproduces it bit for bit.
template <int BITS>
__device__ inline float encode_step(uint32_t code)
{
    const float r[4] = {1.0f / 4, 1.0f / 5, 1.0f / 6, 1.0f / 7};
    const float mag  = r[code & 3] * pow2f(14 + (BITS - 1) - static_cast<int>((code >> 2) & 31));
    return (code & 0x80) ? mag : -mag;
}

template <int BITS>
__global__ void mesh_lend_encode(EncodeArgs a)
{
    const int B = 1 << (BITS - 1), NW = 2 * BITS + 1, per_word = 32 / BITS;
    size_t lo, hi;
    block_range((a.nelem + kUnitValues - 1) / kUnitValues, &lo, &hi);
    for(size_t u = lo + threadIdx.x; u < hi; u += blockDim.x)
    {
        uint32_t* w     = a.wire + static_cast<size_t>(NW) * u;
        uint32_t scales = 0;
        for(int g = 0; g < 4; ++g)
        {
            const size_t e0 = kUnitValues * u + 16 * g;
            uint16_t x[16];
            load8(x, a.inp, e0, a.nelem);
            load8(x + 8, a.inp, e0 + 8, a.nelem);
            float v[16], mp = bf2f(x[0]), mn = -mp;
            for(int j = 0; j < 16; ++j)
            {
                v[j] = bf2f(x[j]);
                mp   = fmaxf(mp, v[j]);
                mn   = fmaxf(mn, -v[j]);
            }
            const bool neg      = mn > mp;
            const uint32_t code = e5m2_of(neg ? mn : mp, BITS > 4) | (neg ? 0x80u : 0u);
            const float enc     = encode_step<BITS>(code);
            for(int k = 0; k < BITS / 2; ++k)
            {
                uint32_t word = 0;
                for(int j = 0; j < per_word; ++j)
                {
                    float q = rintf(v[k * per_word + j] * enc);
                    q       = fminf(fmaxf(q, static_cast<float>(-B)), static_cast<float>(B - 1));
                    word |= static_cast<uint32_t>(static_cast<int>(q) + B) << (BITS * j);
                }
                w[g * (BITS / 2) + k] = word;
            }
            scales |= code << (8 * g);
        }
        w[NW - 1] = scales;
    }
}

template <int BITS>
__global__ void mesh_lend_decode_reduce(DecodeArgs a)
{
    const int B = 1 << (BITS - 1), NW = 2 * BITS + 1, per_word = 32 / BITS;
    const uint32_t mask = (1u << BITS) - 1;
    size_t lo, hi;
    block_range((a.nelem + kUnitValues - 1) / kUnitValues, &lo, &hi);
    // Left: the peer must hold this rank's packed buffer before the next call rewrites it.
    if(!block_ok(threadIdx.x != 0 ||
                     (wait_pieces(a.flags, kArrival, 0, 0, a.split, lo, hi, a.epoch) &&
                      wait_pieces(a.flags, kDeparture, 0, 0, a.split, lo, hi, a.epoch)),
                 a.err))
        return;
    for(size_t u = lo + threadIdx.x; u < hi; u += blockDim.x)
    {
        const uint32_t* wo = a.own + static_cast<size_t>(NW) * u;
        const uint32_t* wp = a.inbox + static_cast<size_t>(NW) * u;
        for(int g = 0; g < 4; ++g)
        {
            const float da = decode_step<BITS>(wo[NW - 1] >> (8 * g) & 0xFF);
            const float db = decode_step<BITS>(wp[NW - 1] >> (8 * g) & 0xFF);
            uint16_t r[16];
            for(int k = 0; k < BITS / 2; ++k)
            {
                const uint32_t xo = wo[g * (BITS / 2) + k], xp = wp[g * (BITS / 2) + k];
                for(int j = 0; j < per_word; ++j)
                {
                    const float fa =
                        static_cast<float>(static_cast<int>(xo >> (BITS * j) & mask) - B) * da;
                    const float fb =
                        static_cast<float>(static_cast<int>(xp >> (BITS * j) & mask) - B) * db;
                    r[k * per_word + j] = f2bf(fa + fb + 0.0f);
                }
            }
            const size_t e0 = kUnitValues * u + 16 * g;
            store8(a.out, e0, a.nelem, r);
            store8(a.out, e0 + 8, a.nelem, r + 8);
        }
    }
}

// ---- four ranks, exact -------------------------------------------------------

struct SliceArgs
{
    const uint16_t* inp;   // this rank's slice of its input
    const uint16_t* inbox; // scatter slots, slot k at k * slot_elems
    uint16_t* reduced;     // the sum, which the engines send in phase 2
    uint16_t* out;         // this rank's slice of the output
    const int* flags;
    int* err;
    Split split;           // of this slice, in vectors
    Split sent[kMaxSlots]; // of the slices this rank sent, for the departure flags
    size_t slot_elems;
    size_t nelem;
    int n, me, epoch;
};

// reduced = out = sum over ranks of slice `me`, added in rank order in float32
// and rounded once.
__global__ void mesh_lend_reduce_slice(SliceArgs a)
{
    size_t lo, hi;
    block_range((a.nelem + 7) / 8, &lo, &hi);
    bool ok = true;
    // Block 0 also waits for what this rank sent: every rank must hold its
    // slice of inp before phase 2 writes out, or the caller frees inp.
    for(int k = 0; k < a.n - 1 && threadIdx.x == 0 && ok; ++k)
        ok = wait_pieces(a.flags, kArrival, 0, k, a.split, lo, hi, a.epoch) &&
             (blockIdx.x != 0 ||
              wait_pieces(a.flags, kDeparture, 0, k, a.sent[k], 0, ~size_t(0), a.epoch));
    if(!block_ok(ok, a.err))
        return;
    for(size_t i = lo + threadIdx.x; i < hi; i += blockDim.x)
    {
        float acc[8];
        uint16_t x[8];
        for(int r = 0; r < a.n; ++r)
        {
            if(r == a.me)
                load8(x, a.inp, 8 * i, a.nelem);
            else
                __builtin_memcpy(
                    x, a.inbox + ((r - a.me - 1 + a.n) % a.n) * a.slot_elems + 8 * i, 16);
            for(int j = 0; j < 8; ++j)
                acc[j] = r == 0 ? bf2f(x[j]) : acc[j] + bf2f(x[j]);
        }
        for(int j = 0; j < 8; ++j)
            x[j] = f2bf(acc[j]);
        __builtin_memcpy(a.reduced + 8 * i, x, 16);
        store8(a.out, 8 * i, a.nelem, x);
    }
}

struct GatherArgs
{
    const uint16_t* inbox; // payload layout: every slice at its own offset
    uint16_t* out;
    const int* flags;
    int* err;
    Split split[kMaxSlots];  // of the slice owned by peer slot k, in vectors
    size_t first[kMaxSlots]; // its first element
    size_t nelem[kMaxSlots]; // its length
    Split sent;              // of this rank's slice, for the departure flags
    int nslot, epoch;
};

// out = the reduced slices of the other ranks.
__global__ void mesh_lend_gather(GatherArgs a)
{
    size_t total = 0;
    for(int k = 0; k < a.nslot; ++k)
        total += (a.nelem[k] + 7) / 8;
    size_t lo, hi;
    block_range(total, &lo, &hi);
    // This block's vectors [s[k], e[k]) of slice k.
    size_t s[kMaxSlots], e[kMaxSlots], base = 0;
    bool ok = true;
    for(int k = 0; k < a.nslot; ++k)
    {
        const size_t end = base + (a.nelem[k] + 7) / 8;
        s[k]             = std::min(std::max(lo, base), end) - base;
        e[k]             = std::min(std::max(hi, base), end) - base;
        base             = end;
        // Block 0 also waits for what this rank sent: every rank must hold
        // the reduced slice before the next call rewrites it.
        if(threadIdx.x == 0 && ok)
            ok = (s[k] == e[k] ||
                  wait_pieces(a.flags, kArrival, 1, k, a.split[k], s[k], e[k], a.epoch)) &&
                 (blockIdx.x != 0 ||
                  wait_pieces(a.flags, kDeparture, 1, k, a.sent, 0, ~size_t(0), a.epoch));
    }
    if(!block_ok(ok, a.err))
        return;
    for(int k = 0; k < a.nslot; ++k)
        for(size_t i = s[k] + threadIdx.x; i < e[k]; i += blockDim.x)
        {
            uint16_t x[8];
            __builtin_memcpy(x, a.inbox + a.first[k] + 8 * i, 16);
            store8(a.out + a.first[k], 8 * i, a.nelem[k], x);
        }
}

// ---- host --------------------------------------------------------------------

// What a rank tells the others.
struct Export
{
    uint32_t node; // KFD topology node of the rank's GPU
    uint32_t max_lenders;
    uint64_t block_bytes;
    uint64_t lendable;           // nodes of the GPUs the rank's ROCr sees, as a bit mask
    hsa_amd_ipc_memory_t handle; // the rank's block
};

struct Via
{
    uint32_t node         = 0;
    int engine[kMaxSlots] = {};
    sdma::Queue queue[kMaxSlots];
};

struct Comm
{
    int device_id = -1, rank = 0, n = 0;
    size_t max_bytes = 0, slot_bytes = 0, inbox_bytes = 0;
    hsa_agent_t gpu[kMaxNodes] = {}; // the process's GPU agents by KFD topology node
    Export exported{};

    char* block = nullptr; // the flag page and two inboxes: what the other ranks' engines write
    char* send  = nullptr; // the packed input (n == 2) or the reduced slice: what the engines
                           // read that is not the caller's
    int* err              = nullptr; // host memory: a kernel that timed out says so here
    char* peer[kMaxSlots] = {};      // the block of rank (rank + 1 + k) % n, attached

    int nvia = 0;
    Via via[kMaxVia]; // 0: own GPU, then the lenders
    sdma::Queue* route[kMaxVia][kMaxSlots] = {};
    bool connected                         = false;
    uint32_t epoch                         = 0;

    char* inbox(char* of_block, int i) const { return of_block + kPage + i * inbox_bytes; }
    ~Comm();
};

[[noreturn]] void fail(const std::string& what)
{ throw std::runtime_error("mesh_lend_all_reduce: " + what); }

size_t page_up(size_t n) { return (n + kPage - 1) & ~(kPage - 1); }

int* flags_of(char* block) { return reinterpret_cast<int*>(block); }

// Fills an array of the process's GPU agents, indexed by KFD topology node. A
// GPU hidden by HIP_VISIBLE_DEVICES still has one; a GPU hidden by
// ROCR_VISIBLE_DEVICES does not.
hsa_status_t add_gpu(hsa_agent_t agent, void* by_node)
{
    hsa_device_type_t type;
    uint32_t node = 0;
    if(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) == HSA_STATUS_SUCCESS &&
       type == HSA_DEVICE_TYPE_GPU &&
       hsa_agent_get_info(agent, HSA_AGENT_INFO_NODE, &node) == HSA_STATUS_SUCCESS &&
       node < kMaxNodes)
        static_cast<hsa_agent_t*>(by_node)[node] = agent;
    return HSA_STATUS_SUCCESS;
}

// The SDMA engine ROCr recommends for copies from `gpu` over its xGMI link to
// `to`; -1 without such a link. ROCr names at most one, from KFD ioctl 1.17.
int engine_for(hsa_agent_t gpu, hsa_agent_t to)
{
    uint32_t mask = 0;
    if(!gpu.handle || !to.handle ||
       hsa_amd_memory_get_preferred_copy_engine(to, gpu, &mask) != HSA_STATUS_SUCCESS || !mask)
        return -1;
    return __builtin_ctz(mask);
}

// The allocation holding ptr must be mapped on every engine GPU: an engine
// that touches an unmapped address faults. Asked on every call and never
// remembered, because a freed block's address can come back as a new
// allocation. hsa_amd_agents_allow_access replaces the access list, so the
// grant is the union of the current list and the engine GPUs.
void ensure_mapped(const Comm* c, const void* ptr, const char* what)
{
    hsa_amd_pointer_info_t info = {};
    info.size                   = sizeof(info);
    uint32_t n                  = 0;
    hsa_agent_t* acc            = nullptr;
    if(hsa_amd_pointer_info(ptr, &info, malloc, &n, &acc) != HSA_STATUS_SUCCESS ||
       info.type == HSA_EXT_POINTER_TYPE_UNKNOWN)
        fail(std::string(what) + " is not a ROCr allocation");
    std::vector<hsa_agent_t> grant(acc, acc + n);
    free(acc);
    for(int v = 0; v < c->nvia; ++v)
    {
        const uint64_t h = c->gpu[c->via[v].node].handle;
        if(h != info.agentOwner.handle &&
           std::none_of(
               grant.begin(), grant.begin() + n, [h](hsa_agent_t a) { return a.handle == h; }))
            grant.push_back(c->gpu[c->via[v].node]);
    }
    if(grant.size() != n &&
       hsa_amd_agents_allow_access(
           static_cast<uint32_t>(grant.size()), grant.data(), nullptr, info.agentBaseAddress) !=
           HSA_STATUS_SUCCESS)
        fail(std::string(what) + " cannot be mapped on every engine GPU");
}

// Runs with this rank's device current.
Comm::~Comm()
{
    // After this no kernel reads the block; destroy_queue waits for the engines.
    (void)hipDeviceSynchronize();
    for(Via& v : via)
        for(sdma::Queue& q : v.queue)
            sdma::destroy_queue(&q);
    for(char* p : peer)
        if(p)
            hsa_amd_ipc_memory_detach(p);
    (void)hipFree(block);
    (void)hipFree(send);
    (void)hipHostFree(err);
}

// Path 0 is the direct link with `direct_weight` shares; each lender has one.
// Every rank computes the same split, so piece p covers the same units in
// both directions.
Split make_split(size_t units, size_t align, int npath, int direct_weight)
{
    Split s{};
    s.npath          = npath;
    const size_t den = static_cast<size_t>(direct_weight + npath - 1);
    for(int p = 1; p < npath; ++p)
        s.bound[p] = units * (direct_weight + p - 1) / den / align * align;
    s.bound[npath] = units;
    return s;
}

// One flow to peer slot `slot`: `bytes` from src to dst in the pieces of
// `split`, each followed by the arrival fence at the receiver and the
// departure fence here, also when the piece is empty.
void post_flow(Comm* c,
               int slot,
               int phase,
               const Split& split,
               size_t unit_bytes,
               const char* src,
               char* dst,
               size_t bytes)
{
    const int slot_at_peer = c->n - 2 - slot; // the peer has this rank in that slot
    for(int i = 1; i <= split.npath; ++i)
    {
        const int p      = i % split.npath; // lenders first: their path is longer
        const size_t end = std::min(split.bound[p + 1] * unit_bytes, bytes);
        const size_t off = std::min(split.bound[p] * unit_bytes, end);
        sdma::Queue& q   = *c->route[p][slot];
        if(end > off)
            sdma::put_copy(q, src + off, dst + off, end - off);
        sdma::put_fence(
            q, flags_of(c->peer[slot]) + flag_index(kArrival, phase, slot_at_peer, p), c->epoch);
        sdma::put_fence(q, flags_of(c->block) + flag_index(kDeparture, phase, slot, p), c->epoch);
    }
}

void ring_doorbells(Comm* c)
{
    for(int v = 0; v < c->nvia; ++v)
        for(sdma::Queue& q : c->via[v].queue)
            if(q.created)
                sdma::ring_doorbell(q);
}

void sync_stream(Comm* c, hipStream_t stream)
{
    const hipError_t e = hipStreamSynchronize(stream);
    if(e != hipSuccess)
        fail(std::string("hipStreamSynchronize: ") + hipGetErrorString(e));
    if(c->err[0])
        fail("a kernel timed out waiting for the transport");
}

int grid(size_t units) { return static_cast<int>(std::clamp<size_t>(units / 1024, 64, 1024)); }

void all_reduce_pair(Comm* c, const uint16_t* inp, uint16_t* out, size_t nelem, hipStream_t stream)
{
    // The engines read inp: everything that produces it must have run.
    sync_stream(c, stream);
    ensure_mapped(c, inp, "the input tensor");
    const int parity   = ++c->epoch & 1;
    const size_t bytes = nelem * 2, nvec = (bytes + kVecBytes - 1) / kVecBytes;
    const PairArgs a{inp,
                     reinterpret_cast<const uint16_t*>(c->inbox(c->block, parity)),
                     out,
                     flags_of(c->block),
                     c->err,
                     make_split(nvec, kVecAlign, c->nvia, 1),
                     nelem,
                     static_cast<int>(c->epoch)};
    post_flow(c,
              0,
              0,
              a.split,
              kVecBytes,
              reinterpret_cast<const char*>(inp),
              c->inbox(c->peer[0], parity),
              bytes);
    ring_doorbells(c);
    hipLaunchKernelGGL(mesh_lend_reduce_pair, dim3(grid(nvec)), dim3(kThreads), 0, stream, a);
}

void all_reduce_pair_quantized(
    Comm* c, const uint16_t* inp, uint16_t* out, size_t nelem, int bits, hipStream_t stream)
{
    const size_t nunit      = (nelem + kUnitValues - 1) / kUnitValues;
    const size_t unit_bytes = static_cast<size_t>(2 * bits + 1) * 4;
    auto* wire              = reinterpret_cast<uint32_t*>(c->send);
    // Measured flat from 256 to 512 blocks at 80 MiB, slower above.
    const int blocks = std::min(grid(nunit), 512);
    const EncodeArgs q{inp, wire, nelem};
    if(bits == 4)
        hipLaunchKernelGGL(mesh_lend_encode<4>, dim3(blocks), dim3(kThreads), 0, stream, q);
    else
        hipLaunchKernelGGL(mesh_lend_encode<8>, dim3(blocks), dim3(kThreads), 0, stream, q);
    // The engines read the packed buffer: the encode must have run.
    sync_stream(c, stream);
    const int parity = ++c->epoch & 1;
    const DecodeArgs a{wire,
                       reinterpret_cast<const uint32_t*>(c->inbox(c->block, parity)),
                       out,
                       flags_of(c->block),
                       c->err,
                       make_split(nunit, kUnitAlign, c->nvia, 1),
                       nelem,
                       static_cast<int>(c->epoch)};
    post_flow(
        c, 0, 0, a.split, unit_bytes, c->send, c->inbox(c->peer[0], parity), nunit * unit_bytes);
    ring_doorbells(c);
    if(bits == 4)
        hipLaunchKernelGGL(mesh_lend_decode_reduce<4>, dim3(blocks), dim3(kThreads), 0, stream, a);
    else
        hipLaunchKernelGGL(mesh_lend_decode_reduce<8>, dim3(blocks), dim3(kThreads), 0, stream, a);
}

void all_reduce_group(Comm* c, const uint16_t* inp, uint16_t* out, size_t nelem, hipStream_t stream)
{
    sync_stream(c, stream);
    ensure_mapped(c, inp, "the input tensor");
    ++c->epoch;
    const int n = c->n, me = c->rank;
    const size_t bytes = nelem * 2, nvec = (bytes + kVecBytes - 1) / kVecBytes;
    // Slice j covers vectors [cut[j], cut[j + 1]); the last one ends at `bytes`.
    size_t cut[kMaxRanks + 1], slice_bytes[kMaxRanks];
    Split split[kMaxRanks];
    for(int j = 0; j <= n; ++j)
        cut[j] = j == n ? nvec : nvec * j / n / kVecAlign * kVecAlign;
    for(int j = 0; j < n; ++j)
    {
        slice_bytes[j] = (j == n - 1 ? bytes : cut[j + 1] * kVecBytes) - cut[j] * kVecBytes;
        split[j]       = make_split(cut[j + 1] - cut[j], kVecAlign, c->nvia, 2 * (n - 1));
    }
    auto* reduced = reinterpret_cast<uint16_t*>(c->send);

    // Phase 1: slice j of this rank's input goes to rank j, into inbox 0.
    SliceArgs s{inp + cut[me] * 8,
                reinterpret_cast<const uint16_t*>(c->inbox(c->block, 0)),
                reduced,
                out + cut[me] * 8,
                flags_of(c->block),
                c->err,
                split[me]};
    s.slot_elems = c->slot_bytes / 2;
    s.nelem      = slice_bytes[me] / 2;
    s.n          = n;
    s.me         = me;
    s.epoch      = static_cast<int>(c->epoch);
    // Phase 2: the reduced slice goes to every rank, into inbox 1.
    GatherArgs g{reinterpret_cast<const uint16_t*>(c->inbox(c->block, 1)), out, s.flags, c->err};
    g.sent  = split[me];
    g.nslot = n - 1;
    g.epoch = s.epoch;
    for(int k = 0; k < n - 1; ++k)
    {
        const int j = (me + 1 + k) % n; // rank j has this rank in slot n - 2 - k
        post_flow(c,
                  k,
                  0,
                  split[j],
                  kVecBytes,
                  reinterpret_cast<const char*>(inp) + cut[j] * kVecBytes,
                  c->inbox(c->peer[k], 0) + (n - 2 - k) * c->slot_bytes,
                  slice_bytes[j]);
        s.sent[k]  = split[j];
        g.split[k] = split[j];
        g.first[k] = cut[j] * 8;
        g.nelem[k] = slice_bytes[j] / 2;
    }
    ring_doorbells(c);
    hipLaunchKernelGGL(
        mesh_lend_reduce_slice, dim3(grid(cut[me + 1] - cut[me])), dim3(kThreads), 0, stream, s);

    // The engines read the reduced slice, so it must exist.
    sync_stream(c, stream);
    for(int k = 0; k < n - 1; ++k)
        post_flow(c,
                  k,
                  1,
                  split[me],
                  kVecBytes,
                  c->send,
                  c->inbox(c->peer[k], 1) + cut[me] * kVecBytes,
                  slice_bytes[me]);
    ring_doorbells(c);
    hipLaunchKernelGGL(
        mesh_lend_gather, dim3(grid(nvec - (cut[me + 1] - cut[me]))), dim3(kThreads), 0, stream, g);
}

Comm* comm_of(fptr_t fa)
{
    if(!fa)
        fail("null context");
    return reinterpret_cast<Comm*>(fa);
}

} // namespace

fptr_t mesh_lend_create(
    int64_t device_id, int64_t rank, int64_t world_size, int64_t max_bytes, int64_t max_lenders)
{
    if(world_size != 2 && world_size != 4)
        throw std::invalid_argument("mesh_lend_all_reduce: world size must be 2 or 4");
    if(rank < 0 || rank >= world_size)
        throw std::invalid_argument("mesh_lend_all_reduce: invalid rank");
    // One piece fits the 30-bit count of a copy packet.
    if(max_bytes < static_cast<int64_t>(kPage) || max_bytes > int64_t(1) << 30)
        throw std::invalid_argument("mesh_lend_all_reduce: max_bytes must be 4 KiB to 1 GiB");
    if(max_lenders < 0)
        throw std::invalid_argument("mesh_lend_all_reduce: max_lenders is negative");
    HipDeviceGuard guard(static_cast<int>(device_id));
    auto c         = std::make_unique<Comm>();
    c->device_id   = static_cast<int>(device_id);
    c->rank        = static_cast<int>(rank);
    c->n           = static_cast<int>(world_size);
    c->max_bytes   = page_up(static_cast<size_t>(max_bytes));
    c->slot_bytes  = page_up(c->max_bytes / c->n) + 2 * kPage;
    c->inbox_bytes = std::max(c->max_bytes, (c->n - 1) * c->slot_bytes);
    // At 8 bits the packed input is 68 bytes per 64 values.
    const size_t send_bytes = c->n == 2
                                  ? page_up((c->max_bytes / 2 + kUnitValues - 1) / kUnitValues * 68)
                                  : c->slot_bytes;

    Export& e     = c->exported;
    e.max_lenders = static_cast<uint32_t>(std::min<int64_t>(max_lenders, kMaxVia - 1));
    e.block_bytes = kPage + 2 * c->inbox_bytes;
    hsa_iterate_agents(add_gpu, c->gpu);
    for(uint32_t node = 0; node < kMaxNodes; ++node)
        e.lendable |= static_cast<uint64_t>(c->gpu[node].handle != 0) << node;
    // The block is uncached: a kernel reads what the engines wrote.
    hsa_amd_pointer_info_t info = {};
    info.size                   = sizeof(info);
    if(hipExtMallocWithFlags(reinterpret_cast<void**>(&c->block),
                             e.block_bytes,
                             hipDeviceMallocUncached) != hipSuccess ||
       hipMemset(c->block, 0, kPage) != hipSuccess ||
       hipMalloc(reinterpret_cast<void**>(&c->send), send_bytes) != hipSuccess ||
       hipHostMalloc(reinterpret_cast<void**>(&c->err), kPage, 0) != hipSuccess)
        fail("cannot allocate the inboxes");
    c->err[0] = 0;
    if(hsa_amd_pointer_info(c->block, &info, nullptr, nullptr, nullptr) != HSA_STATUS_SUCCESS ||
       hsa_agent_get_info(info.agentOwner, HSA_AGENT_INFO_NODE, &e.node) != HSA_STATUS_SUCCESS ||
       hsa_amd_ipc_memory_create(c->block, e.block_bytes, &e.handle) != HSA_STATUS_SUCCESS)
        fail("cannot export the inboxes");
    return reinterpret_cast<fptr_t>(c.release());
}

int64_t mesh_lend_export_size() { return sizeof(Export); }

void mesh_lend_export(fptr_t fa, int64_t out_ptr)
{ memcpy(reinterpret_cast<void*>(out_ptr), &comm_of(fa)->exported, sizeof(Export)); }

void mesh_lend_connect(fptr_t fa, const std::vector<int64_t>& export_ptrs)
{
    Comm* c = comm_of(fa);
    if(c->connected)
        fail("already connected");
    if(export_ptrs.size() != static_cast<size_t>(c->n))
        fail("expected one export per rank");
    HipDeviceGuard guard(c->device_id);
    Export ex[kMaxRanks];
    hsa_agent_t member[kMaxRanks];
    uint64_t lendable    = ~uint64_t(0);
    uint32_t max_lenders = kMaxVia - 1;
    for(int r = 0; r < c->n; ++r)
    {
        memcpy(&ex[r], reinterpret_cast<const void*>(export_ptrs[r]), sizeof(Export));
        if(ex[r].block_bytes != c->exported.block_bytes || ex[r].node >= kMaxNodes)
            fail("rank " + std::to_string(r) + " was created with different parameters");
        member[r] = c->gpu[ex[r].node];
        lendable &= ex[r].lendable & ~(uint64_t(1) << ex[r].node);
        max_lenders = std::min(max_lenders, ex[r].max_lenders);
    }
    for(int a = 0; a < c->n; ++a)
        for(int b = 0; b < c->n; ++b)
            if(a != b && engine_for(member[a], member[b]) < 0)
                fail("ROCr recommends no SDMA engine for a direct xGMI link from rank " +
                     std::to_string(a) + " to rank " + std::to_string(b));

    // The lenders: GPUs outside the group that the ROCr of every rank sees,
    // with an xGMI link to every rank. Every rank derives the same list.
    c->via[0].node = ex[c->rank].node;
    c->nvia        = 1;
    for(uint32_t node = 0; node < kMaxNodes && c->nvia <= static_cast<int>(max_lenders); ++node)
    {
        bool ok = lendable >> node & 1;
        for(int r = 0; r < c->n && ok; ++r)
            ok = engine_for(c->gpu[node], member[r]) >= 0;
        if(ok)
            c->via[c->nvia++].node = node;
    }

    // A peer's block is mapped on exactly the GPUs whose engines write it:
    // ours and the lenders.
    hsa_agent_t agents[kMaxVia];
    for(int v = 0; v < c->nvia; ++v)
        agents[v] = c->gpu[c->via[v].node];
    for(int k = 0; k < c->n - 1; ++k)
    {
        hsa_amd_ipc_memory_t handle = ex[(c->rank + 1 + k) % c->n].handle;
        if(hsa_amd_ipc_memory_attach(&handle,
                                     c->exported.block_bytes,
                                     static_cast<uint32_t>(c->nvia),
                                     agents,
                                     reinterpret_cast<void**>(&c->peer[k])) != HSA_STATUS_SUCCESS)
            fail("cannot map the inboxes of rank " + std::to_string((c->rank + 1 + k) % c->n));
    }

    // Own GPU: a queue per destination, on the engine for the link to it.
    // Lender: one queue, on the engine for the lender's link to this rank, so
    // no two ranks share an engine there. Slots with the same engine share
    // the queue of the first.
    for(int v = 0; v < c->nvia; ++v)
        for(int k = 0; k < c->n - 1; ++k)
        {
            Via& via = c->via[v];
            via.engine[k] =
                engine_for(agents[v], v == 0 ? member[(c->rank + 1 + k) % c->n] : agents[0]);
            int first = 0;
            while(via.engine[first] != via.engine[k])
                ++first;
            c->route[v][k] = &via.queue[first];
            if(first == k && !sdma::create_queue(via.node, via.engine[k], &via.queue[k]))
                fail("no SDMA queue on engine " + std::to_string(via.engine[k]) + " of node " +
                     std::to_string(via.node));
        }

    // The engines stamp departures into the block and read the send buffer.
    ensure_mapped(c, c->block, "the flag page");
    ensure_mapped(c, c->send, "the send buffer");
    c->connected = true;
}

void mesh_lend_all_reduce(fptr_t fa,
                          const aiter_tensor_t& inp,
                          const aiter_tensor_t& out,
                          int64_t bits)
{
    Comm* c = comm_of(fa);
    if(!c->connected)
        fail("not connected");
    if(inp.dtype() != AITER_DTYPE_bf16 || out.dtype() != AITER_DTYPE_bf16)
        throw std::invalid_argument("mesh_lend_all_reduce: only bfloat16 is supported");
    if(inp.numel() != out.numel() || inp.numel() == 0)
        throw std::invalid_argument(
            "mesh_lend_all_reduce: inp and out need the same non-zero numel");
    if(!inp.is_contiguous() || !out.is_contiguous())
        throw std::invalid_argument("mesh_lend_all_reduce: inp and out must be contiguous");
    if(inp.numel() * 2 > c->max_bytes)
        throw std::invalid_argument("mesh_lend_all_reduce: input exceeds max_bytes");
    if(inp.device_id != c->device_id || out.device_id != c->device_id)
        throw std::invalid_argument("mesh_lend_all_reduce: tensors are not on this rank's device");
    if(bits != 0 && bits != 4 && bits != 8)
        throw std::invalid_argument("mesh_lend_all_reduce: bits must be 0, 4 or 8");
    if(bits != 0 && c->n != 2)
        throw std::invalid_argument("mesh_lend_all_reduce: a quantized payload needs two ranks");
    const auto* in = static_cast<const uint16_t*>(inp.data_ptr());
    auto* o        = static_cast<uint16_t*>(out.data_ptr());
    // The engines read inp while a kernel writes out, vector by vector.
    if(in != o && in < o + out.numel() && o < in + inp.numel())
        throw std::invalid_argument(
            "mesh_lend_all_reduce: out may be inp, but not overlap it otherwise");
    HipDeviceGuard guard(c->device_id);
    hipStream_t stream = aiter::getCurrentHIPStream();
    if(c->n > 2)
        all_reduce_group(c, in, o, inp.numel(), stream);
    else if(bits)
        all_reduce_pair_quantized(c, in, o, inp.numel(), static_cast<int>(bits), stream);
    else
        all_reduce_pair(c, in, o, inp.numel(), stream);
}

int64_t mesh_lend_num_lenders(fptr_t fa) { return comm_of(fa)->nvia - 1; }

void mesh_lend_destroy(fptr_t fa)
{
    if(!fa)
        return;
    Comm* c = reinterpret_cast<Comm*>(fa);
    HipDeviceGuard guard(c->device_id);
    delete c;
}

} // namespace aiter
