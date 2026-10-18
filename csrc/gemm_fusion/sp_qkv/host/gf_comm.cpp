// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// gemm_fusion shared communication helper (host only, plain clang++ -x c++; never define __HIPCC__).
// One ccoComm per process, shared by all plans. Call sequence mirrors the Phase-1 drivers (driver_sm.cpp sm_create_v1):
//   ccoCommCreate -> ccoWindowRegister (managed 4-arg overload) for every window -> ccoDevCommCreate (GDA_NONE, sdmaQueueCount) .
// All creation calls are COLLECTIVE: every rank must issue them in the same order with the same sizes.
// MORI is consumed read-only through its installed headers/library.
// Process-global HIP state (see gf_device_scope.hpp): NO entry point leaves the caller's current device changed, the device index is passed
// explicitly (never derived from the SP rank), and the sticky HIP last-error is flushed on exit of every entry point.
#include <hip/hip_runtime.h>
#include <mori/cco/cco.hpp>
#include "gf_device_scope.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
constexpr int OK = 0, E_ARG = -1, E_HIP = -2, E_CCO = -3, E_TOPO = -11;
// The HIP device the communicator was created on (MORI: ccoComm::hipDev). Every call below that touches device state runs under a scope for it.
inline int comm_dev(void* comm) { return comm ? static_cast<int>(static_cast<mori::cco::ccoComm*>(comm)->hipDev) : -1; }
}
extern "C" {
// rank 0 only: fills out128 with the unique id (other ranks get zeros and must receive rank 0's bytes out-of-band).
int32_t gf_uid_v1(uint32_t rank, uint8_t* out128) {
    if (!out128) return E_ARG;
    std::memset(out128, 0, 128);
    if (rank != 0u) return OK;
    mori::cco::ccoUniqueId uid{};
    if (mori::cco::ccoGetUniqueId(&uid) != 0) return E_CCO;
    std::memcpy(out128, &uid, sizeof(uid) < 128 ? sizeof(uid) : 128);
    return OK;
}
// `device` = the HIP device THIS process drives (torch.cuda.current_device() at the caller). MORI records it as comm->hipDev from hipGetDevice()
// inside ccoCommCreate, so it must be current for exactly that call; afterwards the caller's current device is restored.
int32_t gf_comm_create_v1(uint32_t rank, uint32_t world, int32_t device, const uint8_t* uid128, uint64_t per_rank_vmm_bytes, void** out_comm) {
    if (!uid128 || !out_comm || device < 0) return E_ARG;
    *out_comm = nullptr;
    gf::DeviceScope scope(device);
    if (!scope.ok()) { std::fprintf(stderr, "GF_COMM hipSetDevice(%d) failed: %d\n", device, static_cast<int>(scope.err())); return E_HIP; }
    mori::cco::ccoComm* comm = nullptr;
    int rc = mori::cco::ccoCommCreate(*reinterpret_cast<const mori::cco::ccoUniqueId*>(uid128), static_cast<int>(world),
                                      static_cast<int>(rank), per_rank_vmm_bytes, &comm);
    gf::flush_sticky();
    if (rc != 0 || !comm) { std::fprintf(stderr, "GF_COMM comm rc=%d\n", rc); return E_CCO; }
    *out_comm = comm;
    if (comm->hipDev != device) {
        std::fprintf(stderr, "GF_COMM comm.hipDev=%d != requested device=%d (rank=%u)\n", static_cast<int>(comm->hipDev), device, rank);
        return E_TOPO;
    }
    if (static_cast<uint32_t>(comm->worldSize) != world || static_cast<uint32_t>(comm->lsaSize) != world ||
        static_cast<uint32_t>(comm->rank) != rank || static_cast<uint32_t>(comm->lsaRank) != rank || comm->myNodeStart != 0) {
        std::fprintf(stderr, "GF_COMM topology mismatch (single-node LSA world required): world=%d lsaSize=%d rank=%d lsaRank=%d myNodeStart=%d\n",
                     static_cast<int>(comm->worldSize), static_cast<int>(comm->lsaSize), static_cast<int>(comm->rank), static_cast<int>(comm->lsaRank),
                     static_cast<int>(comm->myNodeStart));
        return E_TOPO;
    }
    return OK;
}
// Absolute VA facts of THIS rank's slot (read-only): out[0] = LocalSlotBase = flatBase + lsaRank * perRankSize (where the first-fit slot allocator starts;
// cco_init.cpp LocalSlotBase), out[1] = perRankSize, out[2] = vmmGranularity (G; the slot granularity). flatBase is only G-aligned (hipMemAddressReserve(..., G)),
// so the ABSOLUTE alignment of every window is process dependent (D14 r1c, H-BASE); the abs_aligned / shared_large policies align against this.
int32_t gf_comm_va_info_v1(void* comm, uint64_t* out3) {
    if (!comm || !out3) return E_ARG;
    const auto* c = static_cast<const mori::cco::ccoComm*>(comm);
    out3[0] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(c->flatBase)) + static_cast<uint64_t>(c->lsaRank) * static_cast<uint64_t>(c->perRankSize);
    out3[1] = static_cast<uint64_t>(c->perRankSize);
    out3[2] = static_cast<uint64_t>(c->vmmGranularity);
    return OK;
}
int32_t gf_win_register_v1(void* comm, uint64_t bytes, void** out_win, void** out_local) {
    if (!comm || !out_win || !out_local) return E_ARG;
    gf::DeviceScope scope(comm_dev(comm));
    mori::cco::ccoWindow_t win = nullptr; void* local = nullptr;
    int rc = mori::cco::ccoWindowRegister(static_cast<mori::cco::ccoComm*>(comm), static_cast<size_t>(bytes), &win, &local);
    gf::flush_sticky();  // ccoWindowRegister enables peer access (sticky last-error) internally
    if (rc != 0 || !win || !local) { std::fprintf(stderr, "GF_COMM window rc=%d bytes=%llu\n", rc, (unsigned long long)bytes); return E_CCO; }
    *out_win = reinterpret_cast<void*>(win); *out_local = local;
    return OK;
}
int32_t gf_peer_ptr_v1(void* comm, void* local, int32_t peer, void** out) {
    if (!comm || !local || !out) return E_ARG;
    gf::DeviceScope scope(comm_dev(comm));
    *out = mori::cco::ccoGetPeerPtr(static_cast<mori::cco::ccoComm*>(comm), local, peer);
    return *out ? OK : E_CCO;
}
// Collective; call once after ALL windows of the process are registered (Phase-1 order). Out handle is heap-allocated.
int32_t gf_devcomm_create_v1(void* comm, uint32_t sdma_queues, void** out_dev) {
    if (!comm || !out_dev) return E_ARG;
    gf::DeviceScope scope(comm_dev(comm));
    auto* dev = new mori::cco::ccoDevComm();
    mori::cco::ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.gdaConnectionType = mori::cco::CCO_GDA_CONNECTION_NONE;
    reqs.sdmaQueueCount = static_cast<int>(sdma_queues);
    int rc = mori::cco::ccoDevCommCreate(static_cast<mori::cco::ccoComm*>(comm), &reqs, dev);
    gf::flush_sticky();  // ccoDevCommCreate enables peer access (sticky last-error) internally
    if (rc != 0) { std::fprintf(stderr, "GF_COMM devcomm rc=%d\n", rc); delete dev; return E_CCO; }
    *out_dev = dev;
    return OK;
}
// Facts of the process devcomm + comm for preflight/diagnostics: out[0]=lsaSize out[1]=lsaRank out[2]=sdmaNumQueue out[3]=worldSize
// out[4]=comm.rank out[5]=myNodeStart out[6]=sizeof(ccoDevComm). Read-only; never mutates either object.
int32_t gf_devcomm_info_v1(void* comm, const void* dev, uint64_t* out7) {
    if (!comm || !dev || !out7) return E_ARG;
    const auto* c = static_cast<const mori::cco::ccoComm*>(comm);
    const auto* d = static_cast<const mori::cco::ccoDevComm*>(dev);
    out7[0] = static_cast<uint64_t>(static_cast<int64_t>(d->lsaSize)); out7[1] = static_cast<uint64_t>(static_cast<int64_t>(d->lsaRank));
    out7[2] = static_cast<uint64_t>(static_cast<int64_t>(d->sdma.sdmaNumQueue)); out7[3] = static_cast<uint64_t>(static_cast<int64_t>(c->worldSize));
    out7[4] = static_cast<uint64_t>(static_cast<int64_t>(c->rank)); out7[5] = static_cast<uint64_t>(static_cast<int64_t>(c->myNodeStart)); out7[6] = sizeof(mori::cco::ccoDevComm);
    return OK;
}
// MORI's own zeroing method for ccoMemAlloc windows (cco_init.cpp CcoZeroWindowMem, file-static there, so re-implemented): chunked SYNCHRONOUS hipMemcpy
// host->device from a small zero-filled pageable staging buffer. MORI avoids hipMemset on these windows (on UALink fabric-exportable pools hipMemset
// returns hipErrorOutOfMemory; l.407-410 and the devcomm resource window l.1762). The staging buffer is reused across calls and never written after its fill.
static hipError_t zero_h2d(void* ptr, size_t bytes) {
    if (bytes == 0) return hipSuccess;
    const size_t chunk = std::min<size_t>(bytes, 4u << 20);
    static thread_local std::vector<char> zeros;
    if (zeros.size() < chunk) zeros.assign(chunk, 0);
    char* dst = static_cast<char*>(ptr);
    for (size_t off = 0; off < bytes; off += chunk) {
        const size_t n = std::min(chunk, bytes - off);
        const hipError_t e = hipMemcpy(dst + off, zeros.data(), n, hipMemcpyHostToDevice);
        if (e != hipSuccess) return e;
    }
    return hipSuccess;
}
// Zero a window region, then a device sync. mode 0 = MORI's H2D-copy method (default), mode 1 = the original hipMemset (kept ONLY for A/B:
// AITER_GEMM_FUSION_ZERO_MODE=memset_interleaved). All ranks must call, then barrier, before the next launch.
int32_t gf_zero_v1(int32_t device, void* local, uint64_t bytes, int32_t mode) {
    gf::DeviceScope scope(device);
    if (!scope.ok()) return E_HIP;
    const hipError_t e = (mode == 1) ? hipMemset(local, 0, static_cast<size_t>(bytes)) : zero_h2d(local, static_cast<size_t>(bytes));
    if (e != hipSuccess) { std::fprintf(stderr, "GF_COMM zero mode=%d bytes=%llu err=%d (%s)\n", static_cast<int>(mode), static_cast<unsigned long long>(bytes), static_cast<int>(e), hipGetErrorString(e)); return E_HIP; }
    return hipDeviceSynchronize() == hipSuccess ? OK : E_HIP;
}
// Device memory snapshot for accounting (C-S6): out[0] = free bytes, out[1] = total bytes of `device`.
int32_t gf_mem_info_v1(int32_t device, uint64_t* out2) {
    if (!out2) return E_ARG;
    gf::DeviceScope scope(device);
    if (!scope.ok()) return E_HIP;
    size_t f = 0, t = 0;
    if (hipMemGetInfo(&f, &t) != hipSuccess) return E_HIP;
    out2[0] = f; out2[1] = t;
    return OK;
}
int32_t gf_device_sync_v1(int32_t device) {
    gf::DeviceScope scope(device);
    if (!scope.ok()) return E_HIP;
    return hipDeviceSynchronize() == hipSuccess ? OK : E_HIP;
}
// Teardown in the order used by the MORI samples: devcomm, windows, comm.
int32_t gf_devcomm_destroy_v1(void* comm, void* dev) {
    if (!comm || !dev) return E_ARG;
    gf::DeviceScope scope(comm_dev(comm));
    int rc = mori::cco::ccoDevCommDestroy(static_cast<mori::cco::ccoComm*>(comm), static_cast<mori::cco::ccoDevComm*>(dev));
    delete static_cast<mori::cco::ccoDevComm*>(dev);
    return rc == 0 ? OK : E_CCO;
}
int32_t gf_win_deregister_v1(void* comm, void* win) {
    if (!comm || !win) return E_ARG;
    gf::DeviceScope scope(comm_dev(comm));
    return mori::cco::ccoWindowDeregister(static_cast<mori::cco::ccoComm*>(comm), reinterpret_cast<mori::cco::ccoWindow_t>(win)) == 0 ? OK : E_CCO;
}
int32_t gf_comm_destroy_v1(void* comm) {
    if (!comm) return E_ARG;
    gf::DeviceScope scope(comm_dev(comm));
    return mori::cco::ccoCommDestroy(static_cast<mori::cco::ccoComm*>(comm)) == 0 ? OK : E_CCO;
}
}
