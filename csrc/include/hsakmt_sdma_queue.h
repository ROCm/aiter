// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
// An SDMA queue on one engine of any GPU of the process, written from the host.
//
// ROCr exposes one copy at a time through hsa_amd_memory_async_copy_on_engine
// at about 40 us of host time per copy. A queue of our own takes a packet in
// about 1.5 us and lets several engines of several GPUs run at once.
//
// Created through hsakmt, with the ring in system memory as ROCr's own
// host-written SDMA rings are. Needs KFD ioctl 1.17 (SDMA_BY_ENG_ID queues).
// Host-written doorbells reach the engine on bare metal; in an SR-IOV guest
// they can take milliseconds.

#include <hsakmt/hsakmt.h>

#include <cstdint>
#include <cstring>

namespace aiter {
namespace sdma {

constexpr uint32_t kRingBytes = 256 * 1024;
// SDMA packets. COPY_LINEAR: header, count - 1, parameters, source, destination.
// FENCE: header, address, value.
constexpr uint32_t kOpCopy  = 1;
constexpr uint32_t kOpFence = 5;

struct Queue
{
    HsaQueueResource res = {};
    char* ring           = nullptr;
    bool created         = false;
    uint64_t wpos        = 0;
};

/// hsakmt is a static archive: this opens the module's own instance, once, and keeps it.
inline bool open_hsakmt()
{
    static const bool ok = [] {
        HsaSystemProperties props;
        const HSAKMT_STATUS opened = hsaKmtOpenKFD();
        return (opened == HSAKMT_STATUS_SUCCESS || opened == HSAKMT_STATUS_KERNEL_ALREADY_OPENED) &&
               hsaKmtAcquireSystemProperties(&props) == HSAKMT_STATUS_SUCCESS;
    }();
    return ok;
}

/// Waits until the engine has run everything queued. Idempotent.
inline void destroy_queue(Queue* q)
{
    if(q->created)
    {
        while(__atomic_load_n(q->res.Queue_read_ptr_aql, __ATOMIC_RELAXED) < q->wpos) {}
        hsaKmtDestroyQueue(q->res.QueueId);
    }
    if(q->ring)
    {
        hsaKmtUnmapMemoryToGPU(q->ring);
        hsaKmtFreeMemory(q->ring, kRingBytes);
    }
    *q = {};
}

/// A queue on SDMA engine `engine` of the GPU at KFD topology node `node`.
inline bool create_queue(uint32_t node, uint32_t engine, Queue* q)
{
    HsaMemFlags flags        = {};
    flags.ui32.NonPaged      = 1;
    flags.ui32.HostAccess    = 1;
    flags.ui32.ExecuteAccess = 1;
    void* ring               = nullptr;
    HSAuint64 gpu_va         = 0;
    // Node 0 is a CPU node, which makes the allocation system memory.
    if(!open_hsakmt() || hsaKmtAllocMemory(0, kRingBytes, flags, &ring) != HSAKMT_STATUS_SUCCESS)
        return false;
    q->ring    = static_cast<char*>(ring);
    q->created = hsaKmtMapMemoryToGPUNodes(ring, kRingBytes, &gpu_va, {}, 1, &node) ==
                     HSAKMT_STATUS_SUCCESS &&
                 hsaKmtCreateQueueExt(node,
                                      HSA_QUEUE_SDMA_BY_ENG_ID,
                                      100,
                                      HSA_QUEUE_PRIORITY_MAXIMUM,
                                      engine,
                                      ring,
                                      kRingBytes,
                                      nullptr,
                                      &q->res) == HSAKMT_STATUS_SUCCESS;
    if(!q->created)
        destroy_queue(q);
    return q->created;
}

inline void put(Queue& q, const uint32_t* words, size_t n)
{
    const size_t bytes = n * sizeof(uint32_t);
    uint64_t pos       = q.wpos % kRingBytes;
    if(pos + bytes > kRingBytes)
    {
        memset(q.ring + pos, 0, kRingBytes - pos); // NOPs to the end of the ring
        q.wpos += kRingBytes - pos;
        pos = 0;
    }
    memcpy(q.ring + pos, words, bytes);
    q.wpos += bytes;
}

/// 0 < bytes <= 2^30: the count field holds bytes - 1 in 30 bits.
inline void put_copy(Queue& q, const void* src, void* dst, size_t bytes)
{
    const auto s = reinterpret_cast<uintptr_t>(src), d = reinterpret_cast<uintptr_t>(dst);
    const uint32_t w[7] = {kOpCopy,
                           static_cast<uint32_t>(bytes - 1),
                           0,
                           static_cast<uint32_t>(s),
                           static_cast<uint32_t>(s >> 32),
                           static_cast<uint32_t>(d),
                           static_cast<uint32_t>(d >> 32)};
    put(q, w, 7);
}

/// The engine writes `value` to `addr` once everything queued before has run.
inline void put_fence(Queue& q, void* addr, uint32_t value)
{
    const auto a        = reinterpret_cast<uintptr_t>(addr);
    const uint32_t w[4] = {
        kOpFence, static_cast<uint32_t>(a), static_cast<uint32_t>(a >> 32), value};
    put(q, w, 4);
}

/// Hands everything written so far to the engine.
inline void ring_doorbell(Queue& q)
{
    __atomic_store_n(q.res.Queue_write_ptr_aql, q.wpos, __ATOMIC_SEQ_CST);
    __atomic_store_n(q.res.Queue_DoorBell_aql, q.wpos, __ATOMIC_SEQ_CST);
}

} // namespace sdma
} // namespace aiter
