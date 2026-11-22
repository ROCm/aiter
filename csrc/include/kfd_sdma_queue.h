// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
// Private SDMA queues, created with KFD ioctls inside a process whose GPU VMs
// ROCr already owns, and written from the host.
//
// ROCr exposes one copy at a time through hsa_amd_memory_async_copy_on_engine
// at about 40 us of host time per copy. A queue of our own takes a packet in
// about 1.5 us and lets several engines of several GPUs run at once.
//
// KFD keeps one process object per mm, so a second /dev/kfd fd reaches the
// process ROCr set up. AMDKFD_IOC_CREATE_QUEUE then only needs a ring and
// read/write pointer pages that are GPU-VM mappings of exactly their size:
// host pages registered as userptr BOs at GPU VA == CPU VA. The doorbell is in
// the per-GPU doorbell frame ROCr mapped with its first queue on that GPU.
//
// Needs KFD ioctl 1.17 (queue type SDMA_BY_ENG_ID). Host-written doorbells
// reach the engine on bare metal; in an SR-IOV guest they can take
// milliseconds.
//
// Queue creation follows Nico Holmberg's kfdq.hpp.

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

namespace aiter {
namespace kfd_sdma {

// uapi/linux/kfd_ioctl.h, spelled out because the header an image ships can
// predate 1.17 and lack sdma_engine_id.
struct GetVersionArgs
{ uint32_t major_version, minor_version; };
struct CreateQueueArgs
{
    uint64_t ring_base_address, write_pointer_address, read_pointer_address, doorbell_offset;
    uint32_t ring_size, gpu_id, queue_type, queue_percentage, queue_priority, queue_id;
    uint64_t eop_buffer_address, eop_buffer_size, ctx_save_restore_address;
    uint32_t ctx_save_restore_size, ctl_stack_size, sdma_engine_id, pad;
};
struct DestroyQueueArgs
{ uint32_t queue_id, pad; };
struct AllocArgs
{
    uint64_t va_addr, size, handle, mmap_offset;
    uint32_t gpu_id, flags;
};
struct FreeArgs
{ uint64_t handle; };
struct MapArgs
{
    uint64_t handle, device_ids_array_ptr;
    uint32_t n_devices, n_success;
};

constexpr unsigned long kIocGetVersion   = _IOR('K', 0x01, GetVersionArgs);
constexpr unsigned long kIocCreateQueue  = _IOWR('K', 0x02, CreateQueueArgs);
constexpr unsigned long kIocDestroyQueue = _IOWR('K', 0x03, DestroyQueueArgs);
constexpr unsigned long kIocAlloc        = _IOWR('K', 0x16, AllocArgs);
constexpr unsigned long kIocFree         = _IOW('K', 0x17, FreeArgs);
constexpr unsigned long kIocMap          = _IOWR('K', 0x18, MapArgs);
constexpr unsigned long kIocUnmap        = _IOWR('K', 0x19, MapArgs);

constexpr uint32_t kQueueTypeSdmaByEngId = 0x4;
constexpr uint32_t kMemUserptr           = 1u << 2;
constexpr uint32_t kMemUncached          = 1u << 25;
constexpr uint32_t kMemCoherent          = 1u << 26;
constexpr uint32_t kMemExecutable        = 1u << 30;
constexpr uint32_t kMemWritable          = 1u << 31;

constexpr uint32_t kIoLinkXgmi = 11; // HSA_IOLINKTYPE_XGMI in the sysfs io_links

// SDMA packets. COPY_LINEAR: header, count - 1, parameters, source, destination.
// FENCE: header, address, value.
constexpr uint32_t kOpCopy       = 1;
constexpr uint32_t kOpFence      = 5;
constexpr size_t kMaxCopyBytes   = size_t(1) << 29; // the count field holds 30 bits
constexpr uint32_t kRingBytes    = 256 * 1024;
constexpr size_t kCopyPacketSize = 7 * sizeof(uint32_t);

struct GpuNode
{
    int node;             // KFD topology node
    uint32_t gpu_id;      // 0 if this process may not open it
    uint32_t domain;      // PCI domain
    uint32_t location_id; // PCI bus/device/function
};

inline std::string node_dir(int node)
{ return "/sys/class/kfd/kfd/topology/nodes/" + std::to_string(node); }

/// GPUs of the KFD topology that this process may open, in node order.
inline std::vector<GpuNode> gpu_nodes()
{
    std::vector<GpuNode> out;
    for(int n = 0; n < 256; ++n)
    {
        std::ifstream g(node_dir(n) + "/gpu_id");
        if(!g.is_open())
            break;
        uint32_t id = 0;
        g >> id;
        if(!id)
            continue; // a CPU node, or a GPU outside this process's device cgroup
        std::ifstream props(node_dir(n) + "/properties");
        std::string key, value;
        long long loc = -1, domain = -1;
        while(props >> key >> value)
        {
            if(key == "location_id")
                loc = strtoll(value.c_str(), nullptr, 0);
            if(key == "domain")
                domain = strtoll(value.c_str(), nullptr, 0);
        }
        if(loc >= 0 && domain >= 0)
            out.push_back({n, id, static_cast<uint32_t>(domain), static_cast<uint32_t>(loc)});
    }
    return out;
}

struct Link
{
    uint32_t type        = 0; // 0: no direct link
    uint32_t engine_mask = 0; // recommended_sdma_engine_id_mask
};

/// The direct io_link from one topology node to another.
inline Link link_between(int node_from, int node_to)
{
    for(int l = 0; l < 64; ++l)
    {
        std::ifstream p(node_dir(node_from) + "/io_links/" + std::to_string(l) + "/properties");
        if(!p.is_open())
            break;
        std::string key;
        long long value, to = -1, type = 0, mask = 0;
        while(p >> key >> value)
        {
            if(key == "node_to")
                to = value;
            if(key == "type")
                type = value;
            if(key == "recommended_sdma_engine_id_mask")
                mask = value;
        }
        if(to == node_to)
            return {static_cast<uint32_t>(type), static_cast<uint32_t>(mask)};
    }
    return {};
}

struct Device
{
    int kfd_fd             = -1;
    uint32_t gpu_id        = 0;
    uint64_t doorbell_base = 0, doorbell_size = 0;
};

/// ROCr's doorbell frame for gpu_id: the /dev/kfd mapping whose file offset is
/// KFD_MMAP_TYPE_DOORBELL (3 << 62) | gpu_id << 46. It exists once the process
/// has any queue on that GPU.
inline bool find_doorbell_frame(uint32_t gpu_id, uint64_t* base, uint64_t* size)
{
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while(std::getline(maps, line))
    {
        if(line.find("kfd") == std::string::npos)
            continue;
        unsigned long lo = 0, hi = 0, off = 0;
        char perms[8] = {};
        if(sscanf(line.c_str(), "%lx-%lx %7s %lx", &lo, &hi, perms, &off) != 4)
            continue;
        if((off >> 62) == 3 && ((off >> 46) & 0xFFFF) == (gpu_id & 0xFFFF))
        {
            *base = lo;
            *size = hi - lo;
            return true;
        }
    }
    return false;
}

/// Opens KFD for a GPU whose doorbell frame is mapped. Returns "" or the reason.
inline std::string open_device(uint32_t gpu_id, Device* d)
{
    d->gpu_id = gpu_id;
    if(!find_doorbell_frame(gpu_id, &d->doorbell_base, &d->doorbell_size))
        return "no doorbell frame for gpu_id " + std::to_string(gpu_id);
    if(d->doorbell_size & (d->doorbell_size - 1))
        return "doorbell frame size is not a power of two";
    d->kfd_fd = open("/dev/kfd", O_RDWR | O_CLOEXEC);
    if(d->kfd_fd < 0)
        return std::string("open /dev/kfd: ") + strerror(errno);
    GetVersionArgs v{};
    if(ioctl(d->kfd_fd, kIocGetVersion, &v) || v.major_version != 1 || v.minor_version < 17)
    {
        close(d->kfd_fd);
        d->kfd_fd = -1;
        return "KFD ioctl " + std::to_string(v.major_version) + "." +
               std::to_string(v.minor_version) + " < 1.17: no SDMA_BY_ENG_ID queues";
    }
    return "";
}

inline void close_device(Device* d)
{
    if(d->kfd_fd >= 0)
        close(d->kfd_fd);
    d->kfd_fd = -1;
}

/// Zeroed, locked host pages registered on the GPU at their own address.
struct HostBuffer
{
    char* ptr       = nullptr;
    uint64_t size   = 0;
    uint64_t handle = 0;
};

inline bool alloc_host_buffer(Device& d, uint64_t size, uint32_t extra_flags, HostBuffer* b)
{
    // A fresh anonymous mapping: heap pages can sit in a range ROCr once
    // registered as a userptr, which KFD then refuses with EADDRINUSE.
    void* r = mmap(nullptr, 2 * size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if(r == MAP_FAILED)
        return false;
    const uint64_t lo = reinterpret_cast<uint64_t>(r), va = (lo + size - 1) & ~(size - 1);
    if(va > lo)
        munmap(r, va - lo);
    if(lo + 2 * size > va + size)
        munmap(reinterpret_cast<void*>(va + size), lo + 2 * size - (va + size));
    memset(reinterpret_cast<void*>(va), 0, size);
    mlock(reinterpret_cast<void*>(va), size);
    AllocArgs a{va,
                size,
                0,
                va,
                d.gpu_id,
                kMemUserptr | kMemWritable | kMemCoherent | kMemUncached | extra_flags};
    if(ioctl(d.kfd_fd, kIocAlloc, &a))
    {
        munmap(reinterpret_cast<void*>(va), size);
        return false;
    }
    uint32_t ids[1] = {d.gpu_id};
    MapArgs m{a.handle, reinterpret_cast<uint64_t>(ids), 1, 0};
    if(ioctl(d.kfd_fd, kIocMap, &m) || m.n_success != 1)
    {
        FreeArgs f{a.handle};
        ioctl(d.kfd_fd, kIocFree, &f);
        munmap(reinterpret_cast<void*>(va), size);
        return false;
    }
    b->ptr    = reinterpret_cast<char*>(va);
    b->size   = size;
    b->handle = a.handle;
    return true;
}

inline void free_host_buffer(Device& d, HostBuffer* b)
{
    if(!b->ptr)
        return;
    uint32_t ids[1] = {d.gpu_id};
    MapArgs m{b->handle, reinterpret_cast<uint64_t>(ids), 1, 0};
    ioctl(d.kfd_fd, kIocUnmap, &m);
    FreeArgs f{b->handle};
    ioctl(d.kfd_fd, kIocFree, &f);
    munmap(b->ptr, b->size);
    b->ptr = nullptr;
}

struct Queue
{
    Device* dev       = nullptr;
    uint32_t queue_id = 0;
    uint32_t engine   = 0;
    bool created      = false;
    bool dirty        = false; // packets written since the last doorbell
    HostBuffer ring, rptr, wptr;
    uint64_t wpos      = 0;
    uint64_t* doorbell = nullptr;
};

/// One SDMA queue on `engine` of the device. Returns "" or the reason.
inline std::string create_queue(Device& d, uint32_t engine, Queue* q)
{
    q->dev    = &d;
    q->engine = engine;
    if(!alloc_host_buffer(d, kRingBytes, kMemExecutable, &q->ring) ||
       !alloc_host_buffer(d, 4096, 0, &q->rptr) || !alloc_host_buffer(d, 4096, 0, &q->wptr))
        return std::string("queue buffers: ") + strerror(errno);
    CreateQueueArgs a{};
    a.ring_base_address     = reinterpret_cast<uint64_t>(q->ring.ptr);
    a.ring_size             = kRingBytes;
    a.read_pointer_address  = reinterpret_cast<uint64_t>(q->rptr.ptr);
    a.write_pointer_address = reinterpret_cast<uint64_t>(q->wptr.ptr);
    a.gpu_id                = d.gpu_id;
    a.queue_type            = kQueueTypeSdmaByEngId;
    a.queue_percentage      = 100;
    a.queue_priority        = 15;
    a.sdma_engine_id        = engine;
    if(ioctl(d.kfd_fd, kIocCreateQueue, &a))
        return "AMDKFD_IOC_CREATE_QUEUE on gpu_id " + std::to_string(d.gpu_id) + " engine " +
               std::to_string(engine) + ": " + strerror(errno);
    q->queue_id = a.queue_id;
    q->created  = true;
    q->doorbell =
        reinterpret_cast<uint64_t*>(d.doorbell_base + (a.doorbell_offset & (d.doorbell_size - 1)));
    return "";
}

inline void destroy_queue(Queue* q)
{
    if(q->created)
    {
        DestroyQueueArgs a{q->queue_id, 0};
        ioctl(q->dev->kfd_fd, kIocDestroyQueue, &a);
        q->created = false;
    }
    free_host_buffer(*q->dev, &q->ring);
    free_host_buffer(*q->dev, &q->rptr);
    free_host_buffer(*q->dev, &q->wptr);
}

inline void put(Queue& q, const uint32_t* words, size_t n)
{
    const size_t bytes = n * sizeof(uint32_t);
    uint64_t pos       = q.wpos % kRingBytes;
    if(pos + bytes > kRingBytes)
    {
        memset(q.ring.ptr + pos, 0, kRingBytes - pos); // NOPs to the end of the ring
        q.wpos += kRingBytes - pos;
        pos = 0;
    }
    memcpy(q.ring.ptr + pos, words, bytes);
    q.wpos += bytes;
    q.dirty = true;
}

inline void put_copy(Queue& q, const void* src, void* dst, size_t bytes)
{
    auto s = reinterpret_cast<uintptr_t>(src), d = reinterpret_cast<uintptr_t>(dst);
    while(bytes)
    {
        const size_t n      = bytes < kMaxCopyBytes ? bytes : kMaxCopyBytes;
        const uint32_t w[7] = {kOpCopy,
                               static_cast<uint32_t>(n - 1),
                               0,
                               static_cast<uint32_t>(s),
                               static_cast<uint32_t>(s >> 32),
                               static_cast<uint32_t>(d),
                               static_cast<uint32_t>(d >> 32)};
        put(q, w, 7);
        s += n;
        d += n;
        bytes -= n;
    }
}

/// The engine writes `value` to `addr` once everything queued before has run.
inline void put_fence(Queue& q, void* addr, uint32_t value)
{
    const auto a        = reinterpret_cast<uintptr_t>(addr);
    const uint32_t w[4] = {
        kOpFence, static_cast<uint32_t>(a), static_cast<uint32_t>(a >> 32), value};
    put(q, w, 4);
}

inline void ring_doorbell(Queue& q)
{
    if(!q.dirty)
        return;
    __atomic_store_n(reinterpret_cast<uint64_t*>(q.wptr.ptr), q.wpos, __ATOMIC_SEQ_CST);
    __atomic_store_n(q.doorbell, q.wpos, __ATOMIC_SEQ_CST);
    q.dirty = false;
}

} // namespace kfd_sdma
} // namespace aiter
