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
// Each rank owns private SDMA queues: one per destination on its own GPU and
// one on every lender. Per call the host writes COPY_LINEAR + FENCE + FENCE
// into them: the first fence stamps an arrival flag at the receiver, the
// second a departure flag at the sender. Kernels on the caller's stream wait
// for the flags of the pieces they read, so everything queued behind them is
// ordered after the transport.
//
// Two ranks: one shot. Each rank's whole input goes to the peer in 1 + L equal
// pieces, one per path, and one kernel adds. The peer's next call can start
// landing while this rank still adds, so two inboxes alternate by call
// parity. With bits = 4 or 8 each rank first packs its input (one signed E5M2
// scale byte per 16 values) and the engines move the packed buffer; the
// result is deq(own codes) + deq(peer codes), identical on both ranks.
//
// Three or four ranks: reduce-scatter, then all-gather. Rank j owns slice j.
// Phase 1 sends slice j of every input to rank j, which sums it in rank order.
// Phase 2 sends the sum back to every rank. A rank cannot post call k + 1
// before every rank has consumed call k, so one set of inboxes is enough.
// The direct link carries 2 (n - 1) shares of a flow and each lender one: a
// lender's engine serves a rank's n - 1 destinations in turn, and in that
// time moved about half of what the engine of a direct link does (MI355X).
#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "kfd_sdma_queue.h"
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
constexpr int kFlagStride    = 16;            // ints: one flag per 64 bytes
constexpr size_t kFlagBytes  = 2 * 2 * kMaxSlots * kMaxVia * kFlagStride * sizeof(int);
constexpr long kSpinLimit    = 2000000000L;
constexpr size_t kPage       = 4096;
constexpr size_t kVecBytes   = 16;  // exact payloads move in 16-byte vectors
constexpr size_t kVecAlign   = 256; // vectors per 4 KiB: every piece starts on a page
constexpr size_t kUnitValues = 64;  // quantized payloads: four groups of 16 values
constexpr size_t kUnitAlign =
    1024; // units per piece boundary: 36 KiB and 68 KiB are page multiples
constexpr int kThreads = 256;

enum FlagDir
{
    kArrival   = 0,
    kDeparture = 1
};

// Flag of path `path` of the flow from or to peer slot `slot`, in ints from
// the start of a rank's flag page.
__host__ __device__ inline int flag_index(int dir, int phase, int slot, int path)
{ return (((dir * 2 + phase) * kMaxSlots + slot) * kMaxVia + path) * kFlagStride; }

// Piece p of a flow covers units [bound[p], bound[p + 1]).
struct Split
{
    size_t bound[kMaxVia + 1];
    int npath;
};

// A caller's tensor is only guaranteed 2-byte alignment.
typedef uint64_t __attribute__((aligned(2), may_alias)) bf16x4_t;

__device__ inline float bf2f(uint16_t x)
{ return __builtin_bit_cast(float, static_cast<uint32_t>(x) << 16); }

__device__ inline uint16_t f2bf(float f)
{
    uint32_t u = __builtin_bit_cast(uint32_t, f);
    u += 0x7FFFu + ((u >> 16) & 1u); // round to nearest even
    return static_cast<uint16_t>(u >> 16);
}

// Flags only grow, and a later fence in the same queue implies the earlier
// copy: compare with >=, not ==.
__device__ inline bool wait_flag(const int* flag, int epoch)
{
    long spins = 0;
    while(__atomic_load_n(flag, __ATOMIC_RELAXED) - epoch < 0)
    {
        if(++spins > kSpinLimit)
            return false;
    }
    return true;
}

__device__ inline bool wait_pieces(
    const int* flags, int dir, int phase, int slot, const Split& s, size_t lo, size_t hi, int epoch)
{
    for(int p = 0; p < s.npath; ++p)
    {
        if(s.bound[p + 1] <= lo || s.bound[p] >= hi)
            continue;
        if(!wait_flag(flags + flag_index(dir, phase, slot, p), epoch))
            return false;
    }
    return true;
}

__device__ inline void block_range(size_t n, size_t* lo, size_t* hi)
{
    const size_t per = (n + gridDim.x - 1) / gridDim.x;
    *lo              = static_cast<size_t>(blockIdx.x) * per;
    *hi              = *lo + per < n ? *lo + per : n;
    if(*lo > n)
        *lo = n;
}

// ---- two ranks, exact --------------------------------------------------------

struct PairArgs
{
    const uint16_t* inp;
    const uint16_t* inbox;
    uint16_t* out;
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
    if(!wait_pieces(a.flags, kArrival, 0, 0, a.split, lo, hi, a.epoch))
    {
        a.err[0] = 1;
        return;
    }
    for(size_t i = lo + threadIdx.x; i < hi; i += blockDim.x)
    {
        const size_t e0 = 8 * i;
        if(e0 + 8 <= a.nelem)
        {
            const bf16x4_t* x = reinterpret_cast<const bf16x4_t*>(a.inp + e0);
            const uint64_t* q = reinterpret_cast<const uint64_t*>(a.inbox + e0);
            const uint64_t x0 = x[0], x1 = x[1];
            const uint64_t w0 = __atomic_load_n(q, __ATOMIC_RELAXED);
            const uint64_t w1 = __atomic_load_n(q + 1, __ATOMIC_RELAXED);
            uint64_t r0 = 0, r1 = 0;
            for(int j = 0; j < 4; ++j)
            {
                const int sh = 16 * j;
                r0 |= static_cast<uint64_t>(f2bf(bf2f(x0 >> sh) + bf2f(w0 >> sh))) << sh;
                r1 |= static_cast<uint64_t>(f2bf(bf2f(x1 >> sh) + bf2f(w1 >> sh))) << sh;
            }
            bf16x4_t* o = reinterpret_cast<bf16x4_t*>(a.out + e0);
            o[0]        = r0;
            o[1]        = r1;
        }
        else
        {
            for(size_t e = e0; e < a.nelem; ++e)
                a.out[e] =
                    f2bf(bf2f(a.inp[e]) + bf2f(__atomic_load_n(a.inbox + e, __ATOMIC_RELAXED)));
        }
    }
    // The peer must hold all of inp before the caller may free it.
    if(blockIdx.x == 0 && threadIdx.x == 0 &&
       !wait_pieces(a.flags, kDeparture, 0, 0, a.split, 0, ~size_t(0), a.epoch))
        a.err[0] = 2;
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

// Four bf16 from element e; elements past nelem read as 0.
__device__ inline uint64_t load4(const uint16_t* p, size_t e, size_t nelem)
{
    if(e + 4 <= nelem)
        return *reinterpret_cast<const bf16x4_t*>(p + e);
    uint64_t r = 0;
    for(int k = 0; k < 4; ++k)
        if(e + k < nelem)
            r |= static_cast<uint64_t>(p[e + k]) << (16 * k);
    return r;
}

__device__ inline void store4(uint16_t* p, size_t e, size_t nelem, uint64_t v)
{
    if(e + 4 <= nelem)
    {
        *reinterpret_cast<bf16x4_t*>(p + e) = v;
        return;
    }
    for(int k = 0; k < 4; ++k)
        if(e + k < nelem)
            p[e + k] = static_cast<uint16_t>(v >> (16 * k));
}

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
            float v[16];
            for(int k = 0; k < 4; ++k)
            {
                const uint64_t x = load4(a.inp, e0 + 4 * k, a.nelem);
                for(int j = 0; j < 4; ++j)
                    v[4 * k + j] = bf2f(x >> (16 * j));
            }
            float mp = v[0], mn = -v[0];
            for(int j = 1; j < 16; ++j)
            {
                mp = fmaxf(mp, v[j]);
                mn = fmaxf(mn, -v[j]);
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
    if(!wait_pieces(a.flags, kArrival, 0, 0, a.split, lo, hi, a.epoch))
    {
        a.err[0] = 1;
        return;
    }
    for(size_t u = lo + threadIdx.x; u < hi; u += blockDim.x)
    {
        const uint32_t* wo = a.own + static_cast<size_t>(NW) * u;
        const uint32_t* wp = a.inbox + static_cast<size_t>(NW) * u;
        const uint32_t so  = wo[NW - 1];
        const uint32_t sp  = __atomic_load_n(wp + NW - 1, __ATOMIC_RELAXED);
        for(int g = 0; g < 4; ++g)
        {
            const float da = decode_step<BITS>(so >> (8 * g) & 0xFF);
            const float db = decode_step<BITS>(sp >> (8 * g) & 0xFF);
            uint16_t r[16];
            for(int k = 0; k < BITS / 2; ++k)
            {
                const uint32_t xo = wo[g * (BITS / 2) + k];
                const uint32_t xp = __atomic_load_n(wp + g * (BITS / 2) + k, __ATOMIC_RELAXED);
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
            for(int k = 0; k < 4; ++k)
                store4(a.out,
                       e0 + 4 * k,
                       a.nelem,
                       static_cast<uint64_t>(r[4 * k]) | static_cast<uint64_t>(r[4 * k + 1]) << 16 |
                           static_cast<uint64_t>(r[4 * k + 2]) << 32 |
                           static_cast<uint64_t>(r[4 * k + 3]) << 48);
        }
    }
    // The peer must hold this rank's packed buffer before the next call rewrites it.
    if(blockIdx.x == 0 && threadIdx.x == 0 &&
       !wait_pieces(a.flags, kDeparture, 0, 0, a.split, 0, ~size_t(0), a.epoch))
        a.err[0] = 2;
}

// ---- three or four ranks, exact ----------------------------------------------

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
    for(int k = 0; k < a.n - 1; ++k)
    {
        if(!wait_pieces(a.flags, kArrival, 0, k, a.split, lo, hi, a.epoch))
        {
            a.err[0] = 1;
            return;
        }
    }
    for(size_t i = lo + threadIdx.x; i < hi; i += blockDim.x)
    {
        const size_t e0 = 8 * i;
        const int cnt   = e0 + 8 <= a.nelem ? 8 : static_cast<int>(a.nelem - e0);
        float acc[8];
        for(int r = 0; r < a.n; ++r)
        {
            uint64_t x0 = 0, x1 = 0;
            if(r == a.me)
            {
                if(cnt == 8)
                {
                    const bf16x4_t* x = reinterpret_cast<const bf16x4_t*>(a.inp + e0);
                    x0                = x[0];
                    x1                = x[1];
                }
                else
                {
                    x0 = load4(a.inp, e0, a.nelem);
                    x1 = load4(a.inp, e0 + 4, a.nelem);
                }
            }
            else
            {
                const int k = (r - a.me - 1 + a.n) % a.n;
                const uint64_t* q =
                    reinterpret_cast<const uint64_t*>(a.inbox + k * a.slot_elems + e0);
                x0 = __atomic_load_n(q, __ATOMIC_RELAXED);
                x1 = __atomic_load_n(q + 1, __ATOMIC_RELAXED);
            }
            for(int j = 0; j < 4; ++j)
            {
                const float f0 = bf2f(x0 >> (16 * j)), f1 = bf2f(x1 >> (16 * j));
                acc[j]     = r == 0 ? f0 : acc[j] + f0;
                acc[4 + j] = r == 0 ? f1 : acc[4 + j] + f1;
            }
        }
        uint64_t r0 = 0, r1 = 0;
        for(int j = 0; j < 4; ++j)
        {
            r0 |= static_cast<uint64_t>(f2bf(acc[j])) << (16 * j);
            r1 |= static_cast<uint64_t>(f2bf(acc[4 + j])) << (16 * j);
        }
        uint64_t* red = reinterpret_cast<uint64_t*>(a.reduced + e0);
        red[0]        = r0;
        red[1]        = r1;
        store4(a.out, e0, a.nelem, r0);
        store4(a.out, e0 + 4, a.nelem, r1);
    }
    // Every rank must hold its slice of inp before the caller may free it.
    if(blockIdx.x == 0 && threadIdx.x == 0)
    {
        for(int k = 0; k < a.n - 1; ++k)
        {
            if(!wait_pieces(a.flags, kDeparture, 0, k, a.sent[k], 0, ~size_t(0), a.epoch))
            {
                a.err[0] = 2;
                return;
            }
        }
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
    size_t base = 0;
    for(int k = 0; k < a.nslot; ++k)
    {
        const size_t nvec = (a.nelem[k] + 7) / 8;
        const size_t s    = (lo > base ? lo : base) - base;
        const size_t e    = (hi < base + nvec ? hi : base + nvec) - base;
        if(lo < base + nvec && hi > base && s < e)
        {
            if(!wait_pieces(a.flags, kArrival, 1, k, a.split[k], s, e, a.epoch))
            {
                a.err[0] = 3;
                return;
            }
            const uint16_t* in = a.inbox + a.first[k];
            uint16_t* out      = a.out + a.first[k];
            for(size_t i = s + threadIdx.x; i < e; i += blockDim.x)
            {
                const uint64_t* q = reinterpret_cast<const uint64_t*>(in + 8 * i);
                store4(out, 8 * i, a.nelem[k], __atomic_load_n(q, __ATOMIC_RELAXED));
                store4(out, 8 * i + 4, a.nelem[k], __atomic_load_n(q + 1, __ATOMIC_RELAXED));
            }
        }
        base += nvec;
    }
    // Every rank must hold the reduced slice before the next call rewrites it.
    if(blockIdx.x == 0 && threadIdx.x == 0)
    {
        for(int k = 0; k < a.nslot; ++k)
        {
            if(!wait_pieces(a.flags, kDeparture, 1, k, a.sent, 0, ~size_t(0), a.epoch))
            {
                a.err[0] = 4;
                return;
            }
        }
    }
}

// ---- host --------------------------------------------------------------------

constexpr uint32_t kMagic    = 0x4d4c4152; // "MLAR"
constexpr uint32_t kVersion  = 1;
constexpr int kMaxCandidates = 16;

struct Export
{
    uint32_t magic, version, rank, world_size;
    uint64_t max_bytes;
    uint32_t gpu_id, ncandidates;
    uint32_t candidates[kMaxCandidates]; // gpu_ids this rank can lend from
    struct
    {
        hsa_amd_ipc_memory_t handle;
        uint64_t len, off;
    } buf[3]; // inbox 0, inbox 1, flags
};

struct Peer
{
    int node          = -1;
    char* inbox[2]    = {nullptr, nullptr};
    int* flags        = nullptr;
    void* attached[3] = {nullptr, nullptr, nullptr};
};

struct Via
{
    kfd_sdma::Device dev;
    int node = -1;
};

struct Comm
{
    int device_id = -1, rank = 0, n = 0;
    size_t max_bytes = 0, slot_bytes = 0, wire_bytes = 0;
    kfd_sdma::GpuNode self{};
    std::vector<kfd_sdma::GpuNode> candidates;
    Export exported{};

    char* inbox[2]        = {nullptr, nullptr};
    size_t inbox_bytes[2] = {0, 0};
    uint16_t* reduced     = nullptr; // n > 2
    uint32_t* wire        = nullptr; // n == 2
    int* flags            = nullptr;
    int* err              = nullptr;

    Peer peer[kMaxSlots]; // slot k is rank (rank + 1 + k) % n
    int nvia = 0;
    Via via[kMaxVia]; // 0: own GPU, then the lenders
    kfd_sdma::Queue* route[kMaxVia][kMaxSlots] = {};
    std::vector<std::unique_ptr<kfd_sdma::Queue>> queues;
    std::vector<hsa_agent_t> engine_agents; // agents whose engines touch our buffers
    std::vector<hsa_queue_t*> hsa_queues;   // keep ROCr's doorbell frame on a lender mapped

    bool hsa_up = false, connected = false;
    uint32_t epoch = 0;
    uint64_t calls = 0, grants = 0;

    ~Comm();
};

[[noreturn]] void fail(const std::string& what)
{ throw std::runtime_error("mesh_lend_all_reduce: " + what); }

size_t page_up(size_t n) { return (n + kPage - 1) & ~(kPage - 1); }

bool pci_of_hip_device(int dev, uint32_t* domain, uint32_t* loc)
{
    char bus[32] = {};
    unsigned d = 0, b = 0, s = 0, f = 0;
    if(hipDeviceGetPCIBusId(bus, sizeof(bus), dev) != hipSuccess ||
       sscanf(bus, "%x:%x:%x.%x", &d, &b, &s, &f) != 4)
        return false;
    *domain = d;
    *loc    = ((b & 0xFF) << 8) | ((s & 0x1F) << 3) | (f & 0x7);
    return true;
}

struct AgentFind
{
    uint32_t bdf, domain;
    hsa_agent_t out;
    bool found;
};

hsa_status_t agent_cb(hsa_agent_t agent, void* data)
{
    auto* f = static_cast<AgentFind*>(data);
    hsa_device_type_t type;
    if(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) != HSA_STATUS_SUCCESS ||
       type != HSA_DEVICE_TYPE_GPU)
        return HSA_STATUS_SUCCESS;
    uint32_t bdf = 0, domain = 0;
    hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_BDFID), &bdf);
    hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DOMAIN), &domain);
    if(bdf == f->bdf && domain == f->domain)
    {
        f->out   = agent;
        f->found = true;
        return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
}

// The ROCr agent of a GPU. A GPU hidden by HIP_VISIBLE_DEVICES still has one;
// a GPU hidden by ROCR_VISIBLE_DEVICES does not.
bool agent_of(const kfd_sdma::GpuNode& g, hsa_agent_t* out)
{
    AgentFind f{g.location_id, g.domain, {}, false};
    hsa_iterate_agents(agent_cb, &f);
    if(f.found)
        *out = f.out;
    return f.found;
}

const kfd_sdma::GpuNode* find_gpu(const std::vector<kfd_sdma::GpuNode>& all, uint32_t gpu_id)
{
    for(const auto& g : all)
        if(g.gpu_id == gpu_id)
            return &g;
    return nullptr;
}

std::vector<kfd_sdma::GpuNode>
parse_lenders(const std::string& spec, const std::vector<kfd_sdma::GpuNode>& all, uint32_t self_id)
{
    std::vector<kfd_sdma::GpuNode> out;
    hsa_agent_t agent;
    if(spec == "auto")
    {
        for(const auto& g : all)
            if(g.gpu_id != self_id && agent_of(g, &agent))
                out.push_back(g);
        return out;
    }
    if(spec.empty() || spec == "none")
        return out;
    size_t pos = 0;
    while(pos <= spec.size())
    {
        size_t end = spec.find(',', pos);
        if(end == std::string::npos)
            end = spec.size();
        const std::string tok = spec.substr(pos, end - pos);
        pos                   = end + 1;
        if(tok.empty())
            continue;
        uint32_t domain = 0, loc = 0;
        if(tok.find(':') != std::string::npos)
        {
            unsigned d = 0, b = 0, s = 0, f = 0;
            if(sscanf(tok.c_str(), "%x:%x:%x.%x", &d, &b, &s, &f) != 4)
                fail("lender '" + tok + "' is not a PCI address");
            domain = d;
            loc    = ((b & 0xFF) << 8) | ((s & 0x1F) << 3) | (f & 0x7);
        }
        else if(!pci_of_hip_device(atoi(tok.c_str()), &domain, &loc))
        {
            fail("lender '" + tok + "' is not a HIP device of this process; use its PCI address");
        }
        bool found = false;
        for(const auto& g : all)
        {
            if(g.domain != domain || g.location_id != loc)
                continue;
            found = true;
            if(g.gpu_id == self_id)
                fail("lender '" + tok + "' is this rank's own GPU");
            if(!agent_of(g, &agent))
                fail("lender '" + tok + "' has no ROCr agent (ROCR_VISIBLE_DEVICES?)");
            out.push_back(g);
        }
        if(!found)
            fail("lender '" + tok + "' is not in the KFD topology of this process");
    }
    return out;
}

// The allocation holding ptr must be mapped on every engine GPU: an engine
// that touches an unmapped address faults. Asked on every call and never
// remembered, because a freed block's address can come back as a new
// allocation. hsa_amd_agents_allow_access replaces the access list, so the
// grant is the union of the current list and the engine agents.
void ensure_mapped(Comm* c, const void* ptr, const char* what)
{
    hsa_amd_pointer_info_t info;
    memset(&info, 0, sizeof(info));
    info.size        = sizeof(info);
    uint32_t n       = 0;
    hsa_agent_t* acc = nullptr;
    if(hsa_amd_pointer_info(ptr, &info, malloc, &n, &acc) != HSA_STATUS_SUCCESS ||
       info.type == HSA_EXT_POINTER_TYPE_UNKNOWN)
        fail(std::string(what) + " is not a ROCr allocation");
    std::vector<hsa_agent_t> grant(acc, acc + n);
    free(acc);
    bool missing = false;
    for(hsa_agent_t a : c->engine_agents)
    {
        bool ok = a.handle == info.agentOwner.handle;
        for(uint32_t i = 0; i < n && !ok; ++i)
            ok = grant[i].handle == a.handle;
        if(!ok)
        {
            grant.push_back(a);
            missing = true;
        }
    }
    if(!missing)
        return;
    const hsa_status_t s = hsa_amd_agents_allow_access(
        static_cast<uint32_t>(grant.size()), grant.data(), nullptr, info.agentBaseAddress);
    if(s != HSA_STATUS_SUCCESS)
        fail(std::string(what) + " cannot be mapped on every engine GPU (hsa status " +
             std::to_string(static_cast<int>(s)) + ")");
    ++c->grants;
}

void device_alloc(void** p, size_t bytes, bool uncached, const char* what)
{
    hipError_t e =
        uncached ? hipExtMallocWithFlags(p, bytes, hipDeviceMallocUncached) : hipMalloc(p, bytes);
    if(e == hipSuccess)
        e = hipMemset(*p, 0, bytes);
    if(e != hipSuccess)
        fail(std::string("allocating ") + what + ": " + hipGetErrorString(e));
}

void export_buffer(Comm* c, void* ptr, int i)
{
    hsa_amd_pointer_info_t info;
    memset(&info, 0, sizeof(info));
    info.size = sizeof(info);
    if(hsa_amd_pointer_info(ptr, &info, nullptr, nullptr, nullptr) != HSA_STATUS_SUCCESS ||
       info.type == HSA_EXT_POINTER_TYPE_UNKNOWN)
        fail("an inbox is not a ROCr allocation");
    auto& b = c->exported.buf[i];
    if(hsa_amd_ipc_memory_create(info.agentBaseAddress, info.sizeInBytes, &b.handle) !=
       HSA_STATUS_SUCCESS)
        fail("hsa_amd_ipc_memory_create");
    b.len = info.sizeInBytes;
    b.off =
        static_cast<uint64_t>(static_cast<char*>(ptr) - static_cast<char*>(info.agentBaseAddress));
}

// Runs with this rank's device current.
Comm::~Comm()
{
    (void)hipDeviceSynchronize();
    for(auto& q : queues)
        kfd_sdma::destroy_queue(q.get());
    for(int k = 0; k < kMaxSlots; ++k)
        for(void* a : peer[k].attached)
            if(a)
                hsa_amd_ipc_memory_detach(a);
    for(hsa_queue_t* q : hsa_queues)
        hsa_queue_destroy(q);
    for(int v = 0; v < kMaxVia; ++v)
        kfd_sdma::close_device(&via[v].dev);
    for(void* p : {static_cast<void*>(inbox[0]),
                   static_cast<void*>(inbox[1]),
                   static_cast<void*>(reduced),
                   static_cast<void*>(wire),
                   static_cast<void*>(flags)})
        if(p)
            (void)hipFree(p);
    if(err)
        (void)hipHostFree(err);
    if(hsa_up)
        hsa_shut_down();
}

// Path 0 is the direct link with `direct_weight` shares; each lender has one.
Split make_split(size_t units, size_t align, int npath, int direct_weight)
{
    Split s{};
    s.npath          = npath;
    const size_t den = static_cast<size_t>(direct_weight + npath - 1);
    size_t acc       = 0;
    for(int p = 0; p < npath; ++p)
    {
        acc += p == 0 ? direct_weight : 1;
        s.bound[p + 1] = p == npath - 1 ? units : units * acc / den / align * align;
    }
    return s;
}

// One flow: `bytes` from src to dst in the pieces of `split`, each followed by
// the arrival fence at the receiver and the departure fence here. An empty
// piece still gets its fences, so a kernel may wait for every flag.
void post_flow(Comm* c,
               int slot,
               const Split& split,
               size_t unit_bytes,
               const char* src,
               char* dst,
               size_t bytes,
               int* arrival,
               int* departure,
               int phase,
               int slot_at_peer,
               uint32_t epoch)
{
    for(int i = 1; i <= split.npath; ++i)
    {
        const int p = i % split.npath; // lenders first: their path is longer
        size_t off = split.bound[p] * unit_bytes, end = split.bound[p + 1] * unit_bytes;
        end                = end < bytes ? end : bytes;
        off                = off < end ? off : end;
        kfd_sdma::Queue& q = *c->route[p][slot];
        if(end > off)
            kfd_sdma::put_copy(q, src + off, dst + off, end - off);
        kfd_sdma::put_fence(q, arrival + flag_index(kArrival, phase, slot_at_peer, p), epoch);
        kfd_sdma::put_fence(q, departure + flag_index(kDeparture, phase, slot, p), epoch);
    }
}

void ring_doorbells(Comm* c)
{
    for(auto& q : c->queues)
        kfd_sdma::ring_doorbell(*q);
}

void sync_stream(Comm* c, hipStream_t stream)
{
    const hipError_t e = hipStreamSynchronize(stream);
    if(e != hipSuccess)
        fail(std::string("hipStreamSynchronize: ") + hipGetErrorString(e));
    if(c->err[0])
        fail("a kernel timed out waiting for transport flag " + std::to_string(c->err[0]));
}

int grid(size_t units)
{
    const size_t want = units / 1024;
    return static_cast<int>(want < 64 ? 64 : want > 1024 ? 1024 : want);
}

void all_reduce_pair(Comm* c, const uint16_t* inp, uint16_t* out, size_t nelem, hipStream_t stream)
{
    // The engines read inp: everything that produces it must have run.
    sync_stream(c, stream);
    ensure_mapped(c, inp, "the input tensor");
    const uint32_t epoch = ++c->epoch;
    const int parity     = static_cast<int>(c->calls++ & 1);
    const size_t bytes = nelem * 2, nvec = (bytes + kVecBytes - 1) / kVecBytes;
    PairArgs a{};
    a.split = make_split(nvec, kVecAlign, c->nvia, 1);
    post_flow(c,
              0,
              a.split,
              kVecBytes,
              reinterpret_cast<const char*>(inp),
              c->peer[0].inbox[parity],
              bytes,
              c->peer[0].flags,
              c->flags,
              0,
              0,
              epoch);
    ring_doorbells(c);
    a.inp   = inp;
    a.inbox = reinterpret_cast<const uint16_t*>(c->inbox[parity]);
    a.out   = out;
    a.flags = c->flags;
    a.err   = c->err;
    a.nelem = nelem;
    a.epoch = static_cast<int>(epoch);
    hipLaunchKernelGGL(mesh_lend_reduce_pair, dim3(grid(nvec)), dim3(kThreads), 0, stream, a);
}

void all_reduce_pair_quantized(
    Comm* c, const uint16_t* inp, uint16_t* out, size_t nelem, int bits, hipStream_t stream)
{
    const size_t nunit      = (nelem + kUnitValues - 1) / kUnitValues;
    const size_t unit_bytes = static_cast<size_t>(2 * bits + 1) * 4;
    // Measured flat from 256 to 512 blocks at 80 MiB, slower above.
    const int blocks = std::min(grid(nunit), 512);
    EncodeArgs q{inp, c->wire, nelem};
    if(bits == 4)
        hipLaunchKernelGGL(mesh_lend_encode<4>, dim3(blocks), dim3(kThreads), 0, stream, q);
    else
        hipLaunchKernelGGL(mesh_lend_encode<8>, dim3(blocks), dim3(kThreads), 0, stream, q);
    // The engines read the packed buffer: the encode must have run.
    sync_stream(c, stream);
    const uint32_t epoch = ++c->epoch;
    const int parity     = static_cast<int>(c->calls++ & 1);
    DecodeArgs a{};
    a.split = make_split(nunit, kUnitAlign, c->nvia, 1);
    post_flow(c,
              0,
              a.split,
              unit_bytes,
              reinterpret_cast<const char*>(c->wire),
              c->peer[0].inbox[parity],
              nunit * unit_bytes,
              c->peer[0].flags,
              c->flags,
              0,
              0,
              epoch);
    ring_doorbells(c);
    a.own   = c->wire;
    a.inbox = reinterpret_cast<const uint32_t*>(c->inbox[parity]);
    a.out   = out;
    a.flags = c->flags;
    a.err   = c->err;
    a.nelem = nelem;
    a.epoch = static_cast<int>(epoch);
    if(bits == 4)
        hipLaunchKernelGGL(mesh_lend_decode_reduce<4>, dim3(blocks), dim3(kThreads), 0, stream, a);
    else
        hipLaunchKernelGGL(mesh_lend_decode_reduce<8>, dim3(blocks), dim3(kThreads), 0, stream, a);
}

void all_reduce_group(Comm* c, const uint16_t* inp, uint16_t* out, size_t nelem, hipStream_t stream)
{
    sync_stream(c, stream);
    ensure_mapped(c, inp, "the input tensor");
    const uint32_t epoch = ++c->epoch;
    ++c->calls;
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
    const char* in_bytes = reinterpret_cast<const char*>(inp);

    // Phase 1: slice j of this rank's input goes to rank j.
    SliceArgs s{};
    for(int k = 0; k < n - 1; ++k)
    {
        const int j = (me + 1 + k) % n, slot_at_peer = (me - j - 1 + n) % n;
        post_flow(c,
                  k,
                  split[j],
                  kVecBytes,
                  in_bytes + cut[j] * kVecBytes,
                  c->peer[k].inbox[0] + slot_at_peer * c->slot_bytes,
                  slice_bytes[j],
                  c->peer[k].flags,
                  c->flags,
                  0,
                  slot_at_peer,
                  epoch);
        s.sent[k] = split[j];
    }
    ring_doorbells(c);
    s.inp        = inp + cut[me] * 8;
    s.inbox      = reinterpret_cast<const uint16_t*>(c->inbox[0]);
    s.reduced    = c->reduced;
    s.out        = out + cut[me] * 8;
    s.flags      = c->flags;
    s.err        = c->err;
    s.split      = split[me];
    s.slot_elems = c->slot_bytes / 2;
    s.nelem      = slice_bytes[me] / 2;
    s.n          = n;
    s.me         = me;
    s.epoch      = static_cast<int>(epoch);
    hipLaunchKernelGGL(
        mesh_lend_reduce_slice, dim3(grid(cut[me + 1] - cut[me])), dim3(kThreads), 0, stream, s);

    // Phase 2: the engines read the reduced slice, so it must exist.
    sync_stream(c, stream);
    GatherArgs g{};
    for(int k = 0; k < n - 1; ++k)
    {
        const int j = (me + 1 + k) % n, slot_at_peer = (me - j - 1 + n) % n;
        post_flow(c,
                  k,
                  split[me],
                  kVecBytes,
                  reinterpret_cast<const char*>(c->reduced),
                  c->peer[k].inbox[1] + cut[me] * kVecBytes,
                  slice_bytes[me],
                  c->peer[k].flags,
                  c->flags,
                  1,
                  slot_at_peer,
                  epoch);
        g.split[k] = split[j];
        g.first[k] = cut[j] * 8;
        g.nelem[k] = slice_bytes[j] / 2;
    }
    ring_doorbells(c);
    g.inbox = reinterpret_cast<const uint16_t*>(c->inbox[1]);
    g.out   = out;
    g.flags = c->flags;
    g.err   = c->err;
    g.sent  = split[me];
    g.nslot = n - 1;
    g.epoch = static_cast<int>(epoch);
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

fptr_t mesh_lend_create(int64_t device_id,
                        int64_t rank,
                        int64_t world_size,
                        int64_t max_bytes,
                        const std::string& lenders)
{
    if(world_size < 2 || world_size > kMaxRanks)
        throw std::invalid_argument("mesh_lend_all_reduce: world size must be 2, 3 or 4");
    if(rank < 0 || rank >= world_size)
        throw std::invalid_argument("mesh_lend_all_reduce: invalid rank");
    if(max_bytes < static_cast<int64_t>(kPage))
        throw std::invalid_argument("mesh_lend_all_reduce: max_bytes below one page");
    HipDeviceGuard guard(static_cast<int>(device_id));
    auto c       = std::make_unique<Comm>();
    c->device_id = static_cast<int>(device_id);
    c->rank      = static_cast<int>(rank);
    c->n         = static_cast<int>(world_size);
    c->max_bytes = page_up(static_cast<size_t>(max_bytes));
    if(hsa_init() != HSA_STATUS_SUCCESS)
        fail("hsa_init");
    c->hsa_up = true;

    const std::vector<kfd_sdma::GpuNode> all = kfd_sdma::gpu_nodes();
    uint32_t domain = 0, loc = 0;
    if(!pci_of_hip_device(c->device_id, &domain, &loc))
        fail("no PCI address for HIP device " + std::to_string(device_id));
    bool found = false;
    for(const auto& g : all)
        if(g.domain == domain && g.location_id == loc)
        {
            c->self = g;
            found   = true;
        }
    hsa_agent_t agent;
    if(!found || !agent_of(c->self, &agent))
        fail("HIP device " + std::to_string(device_id) + " is not in the KFD topology");
    c->engine_agents.push_back(agent);
    c->candidates = parse_lenders(lenders, all, c->self.gpu_id);
    if(c->candidates.size() > static_cast<size_t>(kMaxCandidates))
        c->candidates.resize(kMaxCandidates);

    // Buffers the other ranks' engines write: uncached, so a kernel reads what landed.
    if(c->n == 2)
    {
        c->wire_bytes     = page_up((c->max_bytes / 2 + kUnitValues - 1) / kUnitValues * 68);
        c->inbox_bytes[0] = c->inbox_bytes[1] = std::max(c->max_bytes, c->wire_bytes);
        device_alloc(reinterpret_cast<void**>(&c->wire), c->wire_bytes, false, "the packed buffer");
    }
    else
    {
        c->slot_bytes     = page_up(c->max_bytes / c->n) + 2 * kPage;
        c->inbox_bytes[0] = static_cast<size_t>(c->n - 1) * c->slot_bytes;
        c->inbox_bytes[1] = c->max_bytes;
        device_alloc(
            reinterpret_cast<void**>(&c->reduced), c->slot_bytes, false, "the reduced slice");
    }
    for(int i = 0; i < 2; ++i)
        device_alloc(reinterpret_cast<void**>(&c->inbox[i]), c->inbox_bytes[i], true, "an inbox");
    device_alloc(reinterpret_cast<void**>(&c->flags), page_up(kFlagBytes), true, "the flag page");
    if(hipHostMalloc(reinterpret_cast<void**>(&c->err), kPage, 0) != hipSuccess)
        fail("allocating the error page");
    memset(c->err, 0, kPage);
    if(hipDeviceSynchronize() != hipSuccess)
        fail("hipDeviceSynchronize");

    // HIP has run on this GPU by now, so its doorbell frame is mapped.
    const std::string why = kfd_sdma::open_device(c->self.gpu_id, &c->via[0].dev);
    if(!why.empty())
        fail(why);
    c->via[0].node = c->self.node;
    c->nvia        = 1;

    Export& e    = c->exported;
    e.magic      = kMagic;
    e.version    = kVersion;
    e.rank       = static_cast<uint32_t>(c->rank);
    e.world_size = static_cast<uint32_t>(c->n);
    e.max_bytes  = c->max_bytes;
    e.gpu_id     = c->self.gpu_id;
    for(const auto& g : c->candidates)
        e.candidates[e.ncandidates++] = g.gpu_id;
    export_buffer(c.get(), c->inbox[0], 0);
    export_buffer(c.get(), c->inbox[1], 1);
    export_buffer(c.get(), c->flags, 2);
    return reinterpret_cast<fptr_t>(c.release());
}

int64_t mesh_lend_export_size() { return sizeof(Export); }

void mesh_lend_export(fptr_t fa, int64_t out_ptr)
{
    Comm* c = comm_of(fa);
    memcpy(reinterpret_cast<void*>(out_ptr), &c->exported, sizeof(Export));
}

void mesh_lend_connect(fptr_t fa, const std::vector<int64_t>& export_ptrs)
{
    Comm* c = comm_of(fa);
    if(c->connected)
        fail("already connected");
    if(export_ptrs.size() != static_cast<size_t>(c->n))
        fail("expected one export per rank");
    HipDeviceGuard guard(c->device_id);
    const std::vector<kfd_sdma::GpuNode> all = kfd_sdma::gpu_nodes();
    std::vector<Export> ex(c->n);
    std::vector<const kfd_sdma::GpuNode*> member(c->n);
    for(int r = 0; r < c->n; ++r)
    {
        memcpy(&ex[r], reinterpret_cast<const void*>(export_ptrs[r]), sizeof(Export));
        if(ex[r].magic != kMagic || ex[r].version != kVersion ||
           ex[r].rank != static_cast<uint32_t>(r) ||
           ex[r].world_size != static_cast<uint32_t>(c->n) || ex[r].max_bytes != c->max_bytes)
            fail("rank " + std::to_string(r) + " was created with different parameters");
        member[r] = find_gpu(all, ex[r].gpu_id);
        if(!member[r])
            fail("the GPU of rank " + std::to_string(r) + " is not visible to this process's KFD");
    }
    for(int a = 0; a < c->n; ++a)
        for(int b = 0; b < c->n; ++b)
            if(a != b && kfd_sdma::link_between(member[a]->node, member[b]->node).type !=
                             kfd_sdma::kIoLinkXgmi)
                fail("ranks " + std::to_string(a) + " and " + std::to_string(b) +
                     " have no direct xGMI link");

    // The lenders: GPUs every rank can open, outside the group, with an xGMI
    // link to every rank. Every rank derives the same list in the same order.
    for(const auto& g : all)
    {
        bool ok = c->nvia < kMaxVia;
        for(int r = 0; r < c->n && ok; ++r)
        {
            ok = g.gpu_id != ex[r].gpu_id &&
                 std::find(ex[r].candidates, ex[r].candidates + ex[r].ncandidates, g.gpu_id) !=
                     ex[r].candidates + ex[r].ncandidates &&
                 kfd_sdma::link_between(g.node, member[r]->node).type == kfd_sdma::kIoLinkXgmi;
        }
        if(!ok)
            continue;
        hsa_agent_t agent;
        if(!agent_of(g, &agent))
            fail("no ROCr agent for lender gpu_id " + std::to_string(g.gpu_id));
        // HIP may not drive this GPU in this process. KFD maps a GPU's doorbell
        // frame with the process's first queue there, so have ROCr create one.
        uint64_t base = 0, size = 0;
        if(!kfd_sdma::find_doorbell_frame(g.gpu_id, &base, &size))
        {
            hsa_queue_t* q = nullptr;
            if(hsa_queue_create(agent,
                                4096,
                                HSA_QUEUE_TYPE_MULTI,
                                nullptr,
                                nullptr,
                                UINT32_MAX,
                                UINT32_MAX,
                                &q) != HSA_STATUS_SUCCESS)
                fail("hsa_queue_create on lender gpu_id " + std::to_string(g.gpu_id));
            c->hsa_queues.push_back(q);
        }
        Via& v                = c->via[c->nvia];
        const std::string why = kfd_sdma::open_device(g.gpu_id, &v.dev);
        if(!why.empty())
            fail(why);
        v.node = g.node;
        ++c->nvia;
        c->engine_agents.push_back(agent);
    }

    for(int k = 0; k < c->n - 1; ++k)
    {
        const int r   = (c->rank + 1 + k) % c->n;
        Peer& p       = c->peer[k];
        p.node        = member[r]->node;
        void* ptrs[3] = {nullptr, nullptr, nullptr};
        for(int i = 0; i < 3; ++i)
        {
            // Mapped on exactly the GPUs whose engines write it: ours and the lenders.
            if(hsa_amd_ipc_memory_attach(&ex[r].buf[i].handle,
                                         ex[r].buf[i].len,
                                         static_cast<uint32_t>(c->engine_agents.size()),
                                         c->engine_agents.data(),
                                         &p.attached[i]) != HSA_STATUS_SUCCESS)
                fail("hsa_amd_ipc_memory_attach of rank " + std::to_string(r));
            ptrs[i] = static_cast<char*>(p.attached[i]) + ex[r].buf[i].off;
        }
        p.inbox[0] = static_cast<char*>(ptrs[0]);
        p.inbox[1] = static_cast<char*>(ptrs[1]);
        p.flags    = static_cast<int*>(ptrs[2]);
    }

    // Own GPU: one queue per destination, on the engine KFD recommends for the
    // link to it. Lender: one queue, on the engine KFD recommends for the
    // lender's link to this rank, so no two ranks share an engine there.
    for(int v = 0; v < c->nvia; ++v)
    {
        for(int k = 0; k < c->n - 1; ++k)
        {
            const int far       = v == 0 ? c->peer[k].node : c->self.node;
            const uint32_t mask = kfd_sdma::link_between(c->via[v].node, far).engine_mask;
            std::vector<uint32_t> engines, even;
            for(uint32_t b = 0; b < 32; ++b)
                if(mask >> b & 1)
                    (b & 1 ? engines : even).push_back(b);
            // Odd engines were seen to stall private queues on gfx942.
            if(!even.empty())
                engines = even;
            if(engines.empty())
                fail("KFD recommends no SDMA engine for a link of the group");
            const uint32_t engine = engines[static_cast<size_t>(c->rank) % engines.size()];
            kfd_sdma::Queue* q    = nullptr;
            for(auto& have : c->queues)
                if(have->dev == &c->via[v].dev && have->engine == engine)
                    q = have.get();
            if(!q)
            {
                auto fresh            = std::make_unique<kfd_sdma::Queue>();
                const std::string why = kfd_sdma::create_queue(c->via[v].dev, engine, fresh.get());
                c->queues.push_back(std::move(fresh));
                if(!why.empty())
                    fail(why);
                q = c->queues.back().get();
            }
            c->route[v][k] = q;
        }
    }

    ensure_mapped(c, c->flags, "the flag page");
    ensure_mapped(c,
                  c->n == 2 ? static_cast<void*>(c->wire) : static_cast<void*>(c->reduced),
                  "the send buffer");
    c->grants    = 0;
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
    // The engines read inp while the kernel writes out.
    if(bits == 0 && in < o + out.numel() && o < in + inp.numel())
        throw std::invalid_argument("mesh_lend_all_reduce: an exact call cannot run in place");
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

int64_t mesh_lend_num_grants(fptr_t fa) { return static_cast<int64_t>(comm_of(fa)->grants); }

void mesh_lend_destroy(fptr_t fa)
{
    if(!fa)
        return;
    Comm* c = reinterpret_cast<Comm*>(fa);
    HipDeviceGuard guard(c->device_id);
    delete c;
}

} // namespace aiter
