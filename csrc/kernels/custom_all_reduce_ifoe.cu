// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Host side of the IFOE cross-node custom all-reduce: fabric VMM allocation /
// import, the all-reduce context, and kernel launch dispatch.
#include "custom_all_reduce_ifoe.cuh"
#include "custom_all_reduce_ifoe.h"
#include "aiter_stream.h"
#include <cstring>
#include <stdexcept>

using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

namespace aiter {

using ifoe::bf16x8;
using ifoe::kMaxBlocks;
using ifoe::RankData;
using ifoe::RankSignals;
using ifoe::Signal;

#define IFOE_CK(x)                                                                       \
    do                                                                                   \
    {                                                                                    \
        hipError_t e = (x);                                                              \
        if(e != hipSuccess)                                                              \
            throw std::runtime_error(std::string("ifoe hip error: ") +                  \
                                     hipGetErrorString(e) + " @ " + __FILE__ + ":" +     \
                                     std::to_string(__LINE__));                          \
    } while(0)

// Round `bytes` up to the recommended fabric allocation granularity.
static size_t fabric_tot(size_t bytes, int dev)
{
    hipMemAllocationProp pr = {};
    pr.type                 = hipMemAllocationTypePinned;
    pr.requestedHandleTypes = hipMemHandleTypeFabric;
    pr.location.type        = hipMemLocationTypeDevice;
    pr.location.id          = dev;
    size_t g                = 0;
    IFOE_CK(hipMemGetAllocationGranularity(&g, &pr, hipMemAllocationGranularityRecommended));
    return ((bytes + g - 1) / g) * g;
}

static void* fab(size_t bytes, int dev, hipMemFabricHandle_t& fh)
{
    hipMemAllocationProp pr = {};
    pr.type                 = hipMemAllocationTypePinned;
    pr.requestedHandleTypes = hipMemHandleTypeFabric;
    pr.location.type        = hipMemLocationTypeDevice;
    pr.location.id          = dev;
    size_t tot              = fabric_tot(bytes, dev);
    hipMemGenericAllocationHandle_t h;
    IFOE_CK(hipMemCreate(&h, tot, &pr, 0));
    void* p = 0;
    IFOE_CK(hipMemAddressReserve(&p, tot, 0, 0, 0));
    IFOE_CK(hipMemMap(p, tot, 0, h, 0));
    hipMemAccessDesc ad = {};
    ad.location.type    = hipMemLocationTypeDevice;
    ad.location.id      = dev;
    ad.flags            = hipMemAccessFlagsProtReadWrite;
    IFOE_CK(hipMemSetAccess(p, tot, &ad, 1));
    IFOE_CK(hipMemExportToShareableHandle(&fh, h, hipMemHandleTypeFabric, 0));
    return p;
}

static void* imp(const hipMemFabricHandle_t& fh, size_t bytes, int dev)
{
    size_t tot = fabric_tot(bytes, dev);
    hipMemGenericAllocationHandle_t h;
    IFOE_CK(hipMemImportFromShareableHandle(&h, (void*)&fh, hipMemHandleTypeFabric));
    void* p = 0;
    IFOE_CK(hipMemAddressReserve(&p, tot, 0, 0, 0));
    IFOE_CK(hipMemMap(p, tot, 0, h, 0));
    hipMemAccessDesc ad = {};
    ad.location.type    = hipMemLocationTypeDevice;
    ad.location.id      = dev;
    ad.flags            = hipMemAccessFlagsProtReadWrite;
    IFOE_CK(hipMemSetAccess(p, tot, &ad, 1));
    return p;
}

int64_t ifoe_alloc_fabric(int64_t bytes, int64_t handle_out_ptr)
{
    int dev;
    IFOE_CK(hipGetDevice(&dev));
    hipMemFabricHandle_t fh;
    void* p = fab((size_t)bytes, dev, fh);
    std::memcpy((void*)handle_out_ptr, &fh, sizeof(hipMemFabricHandle_t));
    return (int64_t)p;
}

int64_t ifoe_import_fabric(int64_t handle_ptr, int64_t bytes)
{
    int dev;
    IFOE_CK(hipGetDevice(&dev));
    hipMemFabricHandle_t fh;
    std::memcpy(&fh, (void*)handle_ptr, sizeof(hipMemFabricHandle_t));
    return (int64_t)imp(fh, (size_t)bytes, dev);
}

struct IfoeAR
{
    int rank, world;
    void* self_input;
    void* self_bf;
    Signal* self_sg;
    RankData* in_dp; // device: peer input ptrs
    RankData* bf_dp; // device: peer bf16 ptrs
    RankSignals sg;  // peer signal ptrs (by value into kernel)
};

fptr_t ifoe_init(int64_t rank,
                 int64_t world,
                 int64_t self_input_ptr,
                 int64_t self_signal_ptr,
                 int64_t self_bf_ptr,
                 const std::vector<int64_t>& peer_input_ptrs,
                 const std::vector<int64_t>& peer_signal_ptrs,
                 const std::vector<int64_t>& peer_bf_ptrs)
{
    if(world < 2 || world > 8)
        throw std::invalid_argument("ifoe: world must be in [2, 8]");
    if((int)peer_input_ptrs.size() != world || (int)peer_signal_ptrs.size() != world ||
       (int)peer_bf_ptrs.size() != world)
        throw std::invalid_argument("ifoe: peer ptr list length must equal world");

    auto* ar        = new IfoeAR();
    ar->rank        = (int)rank;
    ar->world       = (int)world;
    ar->self_input  = (void*)self_input_ptr;
    ar->self_bf     = (void*)self_bf_ptr;
    ar->self_sg     = (Signal*)self_signal_ptr;
    IFOE_CK(hipMemset(ar->self_sg, 0, sizeof(Signal)));

    RankData h_rd, h_bf;
    for(int i = 0; i < world; i++)
    {
        h_rd.ptrs[i]    = (const void*)peer_input_ptrs[i];
        h_bf.ptrs[i]    = (const void*)peer_bf_ptrs[i];
        ar->sg.signals[i] = (Signal*)peer_signal_ptrs[i];
    }
    IFOE_CK(hipMalloc(&ar->in_dp, sizeof(RankData)));
    IFOE_CK(hipMemcpy(ar->in_dp, &h_rd, sizeof(RankData), hipMemcpyHostToDevice));
    IFOE_CK(hipMalloc(&ar->bf_dp, sizeof(RankData)));
    IFOE_CK(hipMemcpy(ar->bf_dp, &h_bf, sizeof(RankData), hipMemcpyHostToDevice));
    return (fptr_t)ar;
}

#define IFOE_OPT(NG, U) \
    ifoe::allreduce2_opt<NG, U><<<nb, TH, 0, st>>>(ar->in_dp, ar->sg, ar->self_sg, (float*)out_ptr, ar->rank, nvec)
#define IFOE_BF(NG, U) \
    ifoe::allreduce2_bf16<NG, U><<<nb, TH, 0, st>>>(ar->bf_dp, ar->sg, ar->self_sg, (float4*)out_ptr, ar->rank, size8)
#define IFOE_FP8(NG, U) \
    ifoe::allreduce2_fp8<NG, U><<<nb, TH, 0, st>>>(ar->bf_dp, ar->sg, ar->self_sg, (float4*)out_ptr, ar->rank, size16)

template <int U>
static void launch_opt(IfoeAR* ar, void* out_ptr, int nb, int TH, int nvec, hipStream_t st)
{
    switch(ar->world)
    {
    case 2: IFOE_OPT(2, U); break;
    case 4: IFOE_OPT(4, U); break;
    case 8: IFOE_OPT(8, U); break;
    default: throw std::invalid_argument("ifoe: unsupported world");
    }
}
template <int U>
static void launch_bf(IfoeAR* ar, void* out_ptr, int nb, int TH, int size8, hipStream_t st)
{
    switch(ar->world)
    {
    case 2: IFOE_BF(2, U); break;
    case 4: IFOE_BF(4, U); break;
    case 8: IFOE_BF(8, U); break;
    default: throw std::invalid_argument("ifoe: unsupported world");
    }
}
template <int U>
static void launch_fp8(IfoeAR* ar, void* out_ptr, int nb, int TH, int size16, hipStream_t st)
{
    switch(ar->world)
    {
    case 2: IFOE_FP8(2, U); break;
    case 4: IFOE_FP8(4, U); break;
    case 8: IFOE_FP8(8, U); break;
    default: throw std::invalid_argument("ifoe: unsupported world");
    }
}

void ifoe_all_reduce(fptr_t ctx,
                     int64_t inp_ptr,
                     int64_t out_ptr_,
                     int64_t numel,
                     int64_t elt_size,
                     int64_t mode,
                     int64_t unroll,
                     int64_t blocks)
{
    auto* ar         = (IfoeAR*)ctx;
    void* out_ptr    = (void*)out_ptr_;
    hipStream_t st   = getCurrentHIPStream();
    size_t bytes     = (size_t)numel * (size_t)elt_size;
    if(bytes % 32 != 0)
        throw std::invalid_argument("ifoe: tensor bytes must be a multiple of 32");
    if(mode == 2 && bytes % 64 != 0)
        throw std::invalid_argument("ifoe fp8: tensor bytes must be a multiple of 64");
    int nvec         = (int)(bytes / 16); // float4 count
    int size8        = (int)(bytes / 32); // octet count (bf16 path)
    int size16       = (int)(bytes / 64); // hex count (fp8 path)
    int TH           = 512;
    int U            = unroll > 0 ? (int)unroll : (mode == 2 ? 4 : 8); // fp8 default U=4

    // stage the input into the fabric-visible buffer peers read
    IFOE_CK(hipMemcpyAsync(ar->self_input, (void*)inp_ptr, bytes, hipMemcpyDeviceToDevice, st));

    int blkcap = (mode >= 1) ? 256 : 208; // lower-precision moves fewer bytes -> more blocks
    int nb     = blocks > 0 ? (int)blocks : (nvec / ar->world + TH - 1) / TH;
    if(blocks <= 0 && nb > blkcap)
        nb = blkcap;
    if(nb > kMaxBlocks)
        nb = kMaxBlocks;
    if(nb < 1)
        nb = 1;

    if(mode == 2)
    {
        ifoe::cast_fp8<<<nb, TH, 0, st>>>((const float4*)ar->self_input, (ifoe::fp8x16*)ar->self_bf, size16);
        switch(U)
        {
        case 1: launch_fp8<1>(ar, out_ptr, nb, TH, size16, st); break;
        case 2: launch_fp8<2>(ar, out_ptr, nb, TH, size16, st); break;
        default: launch_fp8<4>(ar, out_ptr, nb, TH, size16, st); break;
        }
    }
    else if(mode == 1)
    {
        ifoe::cast_bf16<<<nb, TH, 0, st>>>((const float4*)ar->self_input, (bf16x8*)ar->self_bf, size8);
        switch(U)
        {
        case 2: launch_bf<2>(ar, out_ptr, nb, TH, size8, st); break;
        case 4: launch_bf<4>(ar, out_ptr, nb, TH, size8, st); break;
        default: launch_bf<8>(ar, out_ptr, nb, TH, size8, st); break;
        }
    }
    else
    {
        switch(U)
        {
        case 1: launch_opt<1>(ar, out_ptr, nb, TH, nvec, st); break;
        case 2: launch_opt<2>(ar, out_ptr, nb, TH, nvec, st); break;
        case 4: launch_opt<4>(ar, out_ptr, nb, TH, nvec, st); break;
        default: launch_opt<8>(ar, out_ptr, nb, TH, nvec, st); break;
        }
    }
    IFOE_CK(hipGetLastError());
}

int64_t ifoe_meta_size() { return (int64_t)sizeof(Signal); }

void ifoe_dispose(fptr_t ctx)
{
    auto* ar = (IfoeAR*)ctx;
    if(!ar)
        return;
    if(ar->in_dp)
        (void)hipFree(ar->in_dp);
    if(ar->bf_dp)
        (void)hipFree(ar->bf_dp);
    delete ar;
}

} // namespace aiter
