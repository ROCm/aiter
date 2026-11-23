// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx1250 TP4 one-shot all-reduce + residual add + RMSNorm (+ MXFP8) for decode sizes: the
// host side and the Lamport-order kernels (fp32 denormals flushed, as the custom all-reduce
// module is built). The Gluon-order kernels are in ar_k5_gluon_gfx1250.cu. See
// ar_k5_gfx1250.cuh for the protocol.
#include "ar_k5_gfx1250.h"
#include "aiter_stream.h"
#include "ar_k5_gfx1250.cuh"
#include <algorithm>
#include <stdexcept>
#include <string>

namespace aiter {
namespace k5ar {

void launch_gluon(const Args& a, bool fp8_out, hipStream_t stream);

void launch_lamport(const Args& a, bool fp8_out, hipStream_t stream)
{
    const dim3 grid(2 * a.rows), block(kThreads);
    if(fp8_out)
        k5_ar_rms_kernel<true, true><<<grid, block, 0, stream>>>(a);
    else
        k5_ar_rms_kernel<true, false><<<grid, block, 0, stream>>>(a);
}

struct Ctx
{
    int rank;
    int64_t max_rows;
    int64_t slot; // packs a sender a buffer
    char* inbox[kRanks];
    uint32_t* ctr; // send [max_rows * 8], receive [max_rows * 8], err: past this rank's inbox
};

} // namespace k5ar

namespace k5ar {
// the inbox [3][ranks][max_rows * row packs] of 16-byte packs, then this rank's counters
inline int64_t inbox_bytes(int64_t world_size, int64_t max_rows, int64_t hidden)
{
    return (int64_t)kBuf * world_size * max_rows * (hidden / 8) * 16;
}
inline int64_t ctr_bytes(int64_t max_rows)
{
    return ((2 * max_rows * kWaves + 16) * (int64_t)sizeof(uint32_t) + 4095) / 4096 * 4096;
}
} // namespace k5ar

int64_t k5_ar_buffer_size(int64_t world_size, int64_t max_rows, int64_t hidden)
{
    return k5ar::inbox_bytes(world_size, max_rows, hidden) + k5ar::ctr_bytes(max_rows);
}

fptr_t init_k5_ar(int64_t rank, const std::vector<int64_t>& inbox_ptrs, int64_t max_rows, int64_t hidden)
{
    using namespace k5ar;
    if((int)inbox_ptrs.size() != kRanks || rank < 0 || rank >= kRanks)
        throw std::invalid_argument("k5_ar: 4 ranks only");
    if(hidden != kHidden || max_rows <= 0)
        throw std::invalid_argument("k5_ar: hidden 6144 only");
    auto* c     = new Ctx();
    c->rank     = (int)rank;
    c->max_rows = max_rows;
    c->slot     = max_rows * kRowPacks;
    for(int r = 0; r < kRanks; ++r)
        c->inbox[r] = reinterpret_cast<char*>(inbox_ptrs[r]);
    // the counters live past this rank's inbox, in the same allocation: no allocation here
    const int64_t ib = inbox_bytes(kRanks, max_rows, hidden);
    c->ctr           = reinterpret_cast<uint32_t*>(c->inbox[rank] + ib);
    // rounds and err start at 0, this rank's inbox armed; the caller barriers before the first call
    if(hipMemset(c->ctr, 0, ctr_bytes(max_rows)) != hipSuccess ||
       hipMemset(c->inbox[rank], 0xff, ib) != hipSuccess || hipDeviceSynchronize() != hipSuccess)
        throw std::runtime_error("k5_ar: cannot clear the counters or arm the inbox");
    return reinterpret_cast<fptr_t>(c);
}

void k5_ar_rms(fptr_t ctx,
               int64_t inp,
               int64_t res_inp,
               int64_t w,
               int64_t out,
               int64_t res_out,
               int64_t scale,
               int64_t rows,
               int64_t hidden,
               double eps,
               bool gemma_norm,
               bool lamport_order,
               bool fp8_out,
               int64_t maxit)
{
    using namespace k5ar;
    auto* c = reinterpret_cast<Ctx*>(ctx);
    if(hidden != kHidden)
        throw std::invalid_argument("k5_ar: hidden 6144 only");
    if(rows <= 0)
        return;
    if(rows > c->max_rows)
        throw std::invalid_argument("k5_ar: " + std::to_string(rows) + " rows exceed the inbox (" +
                                    std::to_string(c->max_rows) + ")");
    if(((inp | res_inp | w | res_out | out) & 15) != 0 || (fp8_out && scale == 0))
        throw std::invalid_argument("k5_ar: operands must be 16-byte aligned");
    Args a{};
    a.x        = reinterpret_cast<const u32x4*>(inp);
    a.res      = reinterpret_cast<const u32x4*>(res_inp);
    a.w        = reinterpret_cast<const u32x4*>(w);
    a.res_out  = reinterpret_cast<u32x4*>(res_out);
    a.out      = reinterpret_cast<void*>(out);
    a.scale    = reinterpret_cast<uint8_t*>(scale);
    a.in0      = c->inbox[0];
    a.in1      = c->inbox[1];
    a.in2      = c->inbox[2];
    a.in3      = c->inbox[3];
    a.send_ctr = c->ctr;
    a.recv_ctr = c->ctr + c->max_rows * kWaves;
    a.err      = c->ctr + 2 * c->max_rows * kWaves;
    a.slot     = c->slot;
    a.eps      = eps;
    a.hidden   = (int)hidden;
    a.rows     = (int)rows;
    a.rank     = c->rank;
    a.maxit    = (int)std::min<int64_t>(std::max<int64_t>(maxit, 1), 0x7fffffff);
    a.gemma    = gemma_norm ? 1 : 0;
    hipStream_t stream = aiter::getCurrentHIPStream();
    if(lamport_order)
        launch_lamport(a, fp8_out, stream);
    else
        launch_gluon(a, fp8_out, stream);
    const hipError_t e = hipGetLastError();
    if(e != hipSuccess)
        throw std::runtime_error(std::string("k5_ar launch: ") + hipGetErrorString(e));
}

int64_t k5_ar_errors(fptr_t ctx)
{
    using namespace k5ar;
    auto* c    = reinterpret_cast<Ctx*>(ctx);
    uint32_t e = 0;
    if(hipDeviceSynchronize() != hipSuccess ||
       hipMemcpy(&e, c->ctr + 2 * c->max_rows * kWaves, sizeof(e), hipMemcpyDeviceToHost) != hipSuccess)
        throw std::runtime_error("k5_ar: cannot read the error count");
    return e;
}

} // namespace aiter
