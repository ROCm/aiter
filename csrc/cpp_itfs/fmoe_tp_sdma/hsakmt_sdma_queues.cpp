// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdint>
#include <cstring>

#ifdef _WIN32
#define AITER_HSAKMT_EXPORT extern "C" __declspec(dllexport)
AITER_HSAKMT_EXPORT int xs_open() { return -999; }
AITER_HSAKMT_EXPORT int xs_close() { return -999; }
AITER_HSAKMT_EXPORT int xs_node_for_gpu_id(uint64_t, uint32_t*) { return -999; }
AITER_HSAKMT_EXPORT int xs_create(uint32_t, uint32_t, uint64_t, int, uint64_t*) { return -999; }
AITER_HSAKMT_EXPORT int xs_destroy(uint64_t, uint64_t, uint64_t) { return -999; }
#else
#define AITER_HSAKMT_EXPORT extern "C" __attribute__((visibility("default")))
#include <hsakmt/hsakmt.h>

static int g_open = 0;

AITER_HSAKMT_EXPORT int xs_open()
{
    if(g_open++)
        return 0;
    if(HSAKMT_STATUS s = hsaKmtOpenKFD(); s != HSAKMT_STATUS_SUCCESS)
        return (g_open = 0), -100 - int(s);
    HsaSystemProperties sys{};
    return hsaKmtAcquireSystemProperties(&sys) == HSAKMT_STATUS_SUCCESS ? 0 : -200;
}

AITER_HSAKMT_EXPORT int xs_close()
{
    if(g_open == 0 || --g_open)
        return 0;
    hsaKmtReleaseSystemProperties();
    return hsaKmtCloseKFD() == HSAKMT_STATUS_SUCCESS ? 0 : -1;
}

AITER_HSAKMT_EXPORT int xs_node_for_gpu_id(uint64_t gpu_id, uint32_t* node)
{
    HsaSystemProperties sys{};
    if(hsaKmtAcquireSystemProperties(&sys) != HSAKMT_STATUS_SUCCESS)
        return -1;
    for(uint32_t n = 0; n < sys.NumNodes; ++n)
    {
        HsaNodeProperties p{};
        if(hsaKmtGetNodeProperties(n, &p) == HSAKMT_STATUS_SUCCESS && p.KFDGpuID == gpu_id)
            return (*node = n), 0;
    }
    return -2;
}

AITER_HSAKMT_EXPORT int
xs_create(uint32_t node, uint32_t engine, uint64_t ring_bytes, int priority, uint64_t* out)
{
    HsaMemFlags f{};
    f.ui32.NonPaged = f.ui32.HostAccess = f.ui32.NoNUMABind = 1;
    f.ui32.ExecuteAccess = f.ui32.Uncached = 1;
    f.ui32.PageSize                        = HSA_PAGE_SIZE_4KB;
    void* ring                             = nullptr;
    if(HSAKMT_STATUS s = hsaKmtAllocMemory(node, ring_bytes, f, &ring); s != HSAKMT_STATUS_SUCCESS)
        return -100 - int(s);
    if(HSAKMT_STATUS s = hsaKmtMapMemoryToGPU(ring, ring_bytes, nullptr);
       s != HSAKMT_STATUS_SUCCESS)
        return hsaKmtFreeMemory(ring, ring_bytes), -200 - int(s);
    std::memset(ring, 0, ring_bytes);
    HsaQueueResource q{};
    auto prio       = static_cast<HSA_QUEUE_PRIORITY>(priority);
    HSAKMT_STATUS s = hsaKmtCreateQueueExt(
        node, HSA_QUEUE_SDMA_BY_ENG_ID, 100, prio, engine, ring, ring_bytes, nullptr, &q);
    if(s != HSAKMT_STATUS_SUCCESS)
        return hsaKmtUnmapMemoryToGPU(ring), hsaKmtFreeMemory(ring, ring_bytes), -300 - int(s);
    out[0] = reinterpret_cast<uint64_t>(ring);
    out[1] = static_cast<uint64_t>(q.QueueId);
    out[2] = reinterpret_cast<uint64_t>(q.Queue_read_ptr_aql);
    out[3] = reinterpret_cast<uint64_t>(q.Queue_write_ptr_aql);
    out[4] = reinterpret_cast<uint64_t>(q.Queue_DoorBell_aql);
    out[5] = *reinterpret_cast<volatile uint64_t*>(q.Queue_read_ptr_aql);
    out[6] = *reinterpret_cast<volatile uint64_t*>(q.Queue_write_ptr_aql);
    return 0;
}

AITER_HSAKMT_EXPORT int xs_destroy(uint64_t queue_id, uint64_t ring, uint64_t ring_bytes)
{
    int rc =
        hsaKmtDestroyQueue(static_cast<HSA_QUEUEID>(queue_id)) == HSAKMT_STATUS_SUCCESS ? 0 : -1;
    if(hsaKmtUnmapMemoryToGPU(reinterpret_cast<void*>(ring)) != HSAKMT_STATUS_SUCCESS)
        rc = -2;
    if(hsaKmtFreeMemory(reinterpret_cast<void*>(ring), ring_bytes) != HSAKMT_STATUS_SUCCESS)
        rc = -3;
    return rc;
}
#endif
