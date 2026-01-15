// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "ck_tile/core.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <span>
#include <string_view>

#include <c10/util/Exception.h>
#include <hip/hip_runtime.h>

#define CHECK_COND(x)                                                                          \
    do                                                                                         \
    {                                                                                          \
        if(C10_UNLIKELY(!(x)))                                                                 \
        {                                                                                      \
            std::fprintf(stderr, "\n[AITER] %s:%d %s check failed\n", __FILE__, __LINE__, #x); \
            std::abort();                                                                      \
        }                                                                                      \
    } while(0)

#define HIP_CALL(call)                       \
    do                                       \
    {                                        \
        hipError_t err = call;               \
        TORCH_CHECK(err == hipSuccess,       \
                    "Hip call ",             \
                    #call,                   \
                    " failed with error: ",  \
                    hipGetErrorString(err)); \
    } while(0)

struct p3
{
    unsigned int _p0;
    unsigned int _p1;
    unsigned int _p2;
};
struct p2
{
    unsigned int _p0;
    unsigned int _p1;
};
struct p1
{
    unsigned int _p0;
};

struct AiterAsmKernelArgs
{
    void* args_ptr;
    void* arg_size_ptr;
    int gdx;
    int gdy;
    int gdz;
    int bdx;
    int bdy;
    int bdz;
    const hipStream_t stream;
};

enum class GPUArchId
{
    gfx942,
    gfx950,
    gfxLastKnown = gfx950,
    gfxUnknown   = -1,
};

namespace detail {
struct GpuArchHelper
{
    static inline const std::tuple<GPUArchId, uint32_t>& get_arch_data()
    {
        static const auto data = []() -> std::tuple<GPUArchId, uint32_t> {
            hipDeviceProp_t prop{};
            if(hipGetDeviceProperties(&prop, 0) != hipSuccess)
            {
                return {GPUArchId::gfxUnknown, 0};
            }

            auto num_cu = static_cast<uint32_t>(prop.multiProcessorCount);

            auto match = [&]<size_t N>(const char (&name)[N]) -> bool {
                static_assert(sizeof(prop.gcnArchName) >= N);
                return std::memcmp(prop.gcnArchName, name, N - 1) == 0 &&
                       (prop.gcnArchName[N - 1] == '\0' || prop.gcnArchName[N - 1] == ':');
            };

            if(match("gfx942"))
            {
                return {GPUArchId::gfx942, num_cu};
            }

            if(match("gfx950"))
            {
                return {GPUArchId::gfx950, num_cu};
            }

            return {GPUArchId::gfxUnknown, num_cu};
        }();
        return data;
    }
};
} // namespace detail

static inline GPUArchId get_gpu_arch()
{
    return std::get<0>(detail::GpuArchHelper::get_arch_data());
}

static inline uint32_t get_num_cu() { return std::get<1>(detail::GpuArchHelper::get_arch_data()); }

struct __attribute__((packed)) AiterAsmKernelCodeObjectWrapper
{
    static constexpr char magic = '#';
    char header;
    int32_t gfx942_offset;
    int32_t gfx942_MI308_offset;
    int32_t gfx950_offset;
    uint8_t hsaco[];
};

namespace detail {
struct FatBinaryWrapper
{
    uint32_t magic        = 0x48495046; // "HIPF";
    uint32_t version      = 1;
    const uint8_t* binary = nullptr;
    intptr_t __pad        = 0;
};

extern "C" void* __hipRegisterFatBinary(const FatBinaryWrapper* data) noexcept;
extern "C" void __hipUnregisterFatBinary(void* module) noexcept;
extern "C" void __hipRegisterFunction(void* module,
                                      const void* hostFunction,
                                      const char* deviceFunction,
                                      const char* deviceName,
                                      int threadLimit,
                                      void* tid,
                                      void* bid,
                                      void* blockDim,
                                      void* gridDim,
                                      void* wSize) noexcept;
extern "C" __attribute__((visibility("hidden"))) void* __dso_handle;
extern "C" int __cxa_atexit(void (*f)(void*), void* p, void* dso) noexcept;

} // namespace detail

template <size_t N, size_t M>
struct StaticKernelDescriptor
{
    uint8_t data[N + M + 1];
    consteval StaticKernelDescriptor(const char (&kernel_name)[N],
                                     const uint8_t (&kernel_code_object)[M])
    {
        data[0] = N - 1;
        std::copy_n(kernel_name, N, data + 1);
        std::copy_n(kernel_code_object, M, data + N + 1);
    }
};

namespace detail {
constexpr uint8_t NoImage[1] = {0};
constexpr StaticKernelDescriptor NoStaticDescriptor{"", NoImage};
} // namespace detail

template <StaticKernelDescriptor Desc = detail::NoStaticDescriptor>
class AiterAsmKernel;

template <typename T,
          AiterAsmKernel<>* Kernels,
          const uint32_t* KernelDescriptorOffsets,
          const uint8_t* KernelDescriptors>
struct AiterAsmKernelConfigMap;

template <>
class AiterAsmKernel<detail::NoStaticDescriptor>
{
    enum class State : uint8_t
    {
        NotLoaded,
        Loading,
        Loaded
    };

    std::atomic<State> state = State::NotLoaded;

    static inline const uint8_t* select_code_object(const uint8_t* kernel_data)
    {
        if(static_cast<char>(kernel_data[0]) != AiterAsmKernelCodeObjectWrapper::magic)
        {
            // Assume plain hsaco object
            return kernel_data;
        }

        auto archive   = reinterpret_cast<const AiterAsmKernelCodeObjectWrapper*>(kernel_data);
        int32_t offset = -1;
        auto arch_id = get_gpu_arch();
        switch(arch_id)
        {
        case GPUArchId::gfxUnknown: break;
        case GPUArchId::gfx942:
            switch(get_num_cu())
            {
            default:
            case 304: offset = archive->gfx942_offset; break;
            case 80:
            case 64:
                offset = archive->gfx942_MI308_offset >= 0 ? archive->gfx942_MI308_offset
                                                           : archive->gfx942_offset;
                break;
            }
            break;
        case GPUArchId::gfx950: offset = archive->gfx950_offset; break;
        }

        TORCH_CHECK(offset >= 0, "No kernel found for arch_id: ", arch_id);
        return &archive->hsaco[offset];
    }

    protected:
    void launch_kernel_with_descriptor(const AiterAsmKernelArgs& kargs,
                                       const uint8_t* desc) noexcept
    {
        void* config[]            = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                                     kargs.args_ptr,
                                     HIP_LAUNCH_PARAM_BUFFER_SIZE,
                                     kargs.arg_size_ptr,
                                     HIP_LAUNCH_PARAM_END};
        hipFunction_t kernel_func = nullptr;
        (void)hipGetFuncBySymbol(&kernel_func, reinterpret_cast<void*>(this));

        auto error = hipModuleLaunchKernel(kernel_func,
                                           kargs.gdx,
                                           kargs.gdy,
                                           kargs.gdz,
                                           kargs.bdx,
                                           kargs.bdy,
                                           kargs.bdz,
                                           0,
                                           kargs.stream,
                                           nullptr,
                                           (void**)&config);

        if(C10_UNLIKELY(error != hipSuccess))
        {
            if(C10_LIKELY(error == hipErrorInvalidResourceHandle))
            {
                (void)hipGetLastError(); // clear the error state
                init_with_descriptor(desc)->launch_kernel(kargs);
                return;
            }
            HIP_CALL(error /* hipModuleLaunchKernel */);
            __builtin_unreachable();
        }
    }

    AiterAsmKernel<>* init_with_descriptor(const uint8_t* desc)
    {
        if(C10_LIKELY(this->state.load(std::memory_order_acquire) == State::Loaded))
        {
            return this;
        }

        auto old = State::NotLoaded;
        if(this->state.compare_exchange_strong(old, State::Loading, std::memory_order_acq_rel))
        {
            auto kernel_name = reinterpret_cast<const char*>(&desc[1]);
            auto kernel_co   = select_code_object(desc + static_cast<uint32_t>(desc[0]) + 2);
            detail::FatBinaryWrapper fat_bin{};
            fat_bin.binary = kernel_co;
            auto module    = detail::__hipRegisterFatBinary(&fat_bin);
            CHECK_COND(module != nullptr);
            CHECK_COND(detail::__cxa_atexit(detail::__hipUnregisterFatBinary,
                                            module,
                                            static_cast<void*>(&detail::__dso_handle)) == 0);
            detail::__hipRegisterFunction(module,
                                          static_cast<void*>(this),
                                          kernel_name,
                                          kernel_name,
                                          -1,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          nullptr);
            this->state.store(State::Loaded, std::memory_order_release);
            this->state.notify_all();
        }
        else
        {
            if(old == State::Loading)
            {
                this->state.wait(State::Loading, std::memory_order_acquire);
                old = this->state.load(std::memory_order_relaxed);
            }
            CHECK_COND(old == State::Loaded);
        }

        return this;
    }

    public:
    consteval AiterAsmKernel()                  = default;
    AiterAsmKernel(AiterAsmKernel&)             = delete;
    AiterAsmKernel(AiterAsmKernel&&)            = delete;
    AiterAsmKernel& operator=(AiterAsmKernel&)  = delete;
    AiterAsmKernel& operator=(AiterAsmKernel&&) = delete;

    void launch_kernel(const AiterAsmKernelArgs& kargs)
    {
        void* config[]            = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                                     kargs.args_ptr,
                                     HIP_LAUNCH_PARAM_BUFFER_SIZE,
                                     kargs.arg_size_ptr,
                                     HIP_LAUNCH_PARAM_END};
        hipFunction_t kernel_func = nullptr;
        // TODO Ask runtime folks to provide an API for hipLaunchKernel with extra arg
        // Don't error check here.
        // Failure to load the func would cause hipModuleLaunchKernel to fail anyways.
        (void)hipGetFuncBySymbol(&kernel_func, reinterpret_cast<void*>(this));

        HIP_CALL(hipModuleLaunchKernel(kernel_func,
                                       kargs.gdx,
                                       kargs.gdy,
                                       kargs.gdz,
                                       kargs.bdx,
                                       kargs.bdy,
                                       kargs.bdz,
                                       0,
                                       kargs.stream,
                                       nullptr,
                                       (void**)&config));
    }

    template <typename T,
              AiterAsmKernel<>* Kernels,
              const uint32_t* KernelDescriptorOffsets,
              const uint8_t* KernelDescriptors>
    friend struct AiterAsmKernelConfigMap;
};

template <StaticKernelDescriptor Desc>
class AiterAsmKernel : private AiterAsmKernel<>
{
    public:
    void __attribute__((always_inline)) launch_kernel(const AiterAsmKernelArgs& kargs)
    {
        launch_kernel_with_descriptor(kargs, &Desc.data[0]);
    }

    AiterAsmKernel<>* operator&() { return init_with_descriptor(&Desc.data[0]); }
};

template <typename T,
          AiterAsmKernel<>* Kernels,
          const uint32_t* KernelDescriptorOffsets,
          const uint8_t* KernelDescriptors>
struct AiterAsmKernelConfigMap
{
    using Entry = T;
    uint16_t kernel_index_bias;
    uint16_t entry_count;
    std::pair<const uint16_t, const uint16_t>
        per_arch_offsets[static_cast<int>(GPUArchId::gfxLastKnown) + 1];
    // T entries[];
    bool empty() const { return entry_count == 0; }
    std::span<const Entry> get_configs_for_arch(GPUArchId arch_id) const
    {
        auto entries          = reinterpret_cast<const Entry*>(&this[1]);
        uint16_t begin_offset = arch_id != GPUArchId::gfxUnknown
                                    ? per_arch_offsets[static_cast<int>(arch_id)].first
                                    : 0;
        uint16_t end_offset   = arch_id != GPUArchId::gfxUnknown
                                    ? per_arch_offsets[static_cast<int>(arch_id)].second
                                    : 0;
        __builtin_assume(end_offset <= entry_count);
        return {&entries[begin_offset], &entries[end_offset]};
    }

    std::string_view get_kernel_name_for_config(const T* entry) const
    {
        if(C10_UNLIKELY(entry == nullptr))
        {
            return {};
        }
        auto entries = reinterpret_cast<const Entry*>(&this[1]);
        CHECK_COND(entry >= entries && entry < &entries[entry_count]);
        auto index = static_cast<uint32_t>(entry - entries);
        const uint8_t* descriptor =
            &KernelDescriptors[KernelDescriptorOffsets[kernel_index_bias + index]];
        return {reinterpret_cast<const char*>(&descriptor[1]), static_cast<size_t>(descriptor[0])};
    }

    AiterAsmKernel<>* load_kernel_for_config(const T* entry) const
    {
        if(C10_UNLIKELY(entry == nullptr))
        {
            return nullptr;
        }
        auto entries = reinterpret_cast<const Entry*>(&this[1]);
        CHECK_COND(entry >= entries && entry < &entries[entry_count]);
        auto index                = static_cast<uint32_t>(entry - entries);
        uint32_t kernel_index     = kernel_index_bias + index;
        const uint8_t* descriptor = &KernelDescriptors[KernelDescriptorOffsets[kernel_index]];
        return Kernels[kernel_index].init_with_descriptor(descriptor);
    }

    const Entry* find_config_by_kernel_name(GPUArchId arch_id, std::string_view name) const
    {
        auto entries          = reinterpret_cast<const Entry*>(&this[1]);
        uint16_t begin_offset = arch_id != GPUArchId::gfxUnknown
                                    ? per_arch_offsets[static_cast<int>(arch_id)].first
                                    : 0;
        uint16_t end_offset   = arch_id != GPUArchId::gfxUnknown
                                    ? per_arch_offsets[static_cast<int>(arch_id)].second
                                    : 0;
        __builtin_assume(end_offset <= entry_count);
        for(int i = begin_offset; i < end_offset; i++)
        {
            const uint8_t* descriptor =
                &KernelDescriptors[KernelDescriptorOffsets[kernel_index_bias + i]];
            if(std::string_view(reinterpret_cast<const char*>(&descriptor[1]),
                                static_cast<size_t>(descriptor[0])) == name)
            {
                return &entries[i];
            }
        }

        return nullptr;
    }
};

// Hack around missing flexible array initalizer support
template <typename T, size_t EntryCount>
struct AiterAsmKernelConfigMapSized : public T
{
    typename T::Entry entries[EntryCount];
    const T* operator&() const { return this; }
};

template <size_t MinSize, size_t MaxSize>
struct FixedString
{
    template <size_t N>
    consteval FixedString(const char (&str)[N])
    {
        constexpr size_t size = N - 1;
        static_assert(size >= MinSize && size <= MaxSize);
        std::copy_n(str, size, data);
    }

    constexpr size_t size() const
    {
        for(int i = MinSize; i < MaxSize; i++)
        {
            if(data[i] == '\0')
            {
                return i;
            }
        }
        return MaxSize;
    }

    operator std::string_view() const { return {data, size()}; }

    friend bool operator==(const FixedString& lhs, const std::string_view& rhs)
    {
        return std::string_view(lhs) == rhs;
    }

    private:
    char data[MaxSize] = {0};
};

template <class Key, class T, class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>>
struct SynchronizedCache
{
    template <typename F>
    T& get_or_create(const Key& k, F&& factory)
    {
        std::lock_guard<std::mutex> map_mu_guard(map_mu);
        auto [it, inserted] = map.try_emplace(k);
        if(C10_LIKELY(!inserted))
        {
            return it->second;
        }

        return (it->second = factory());
    }

    private:
    std::mutex map_mu;
    std::unordered_map<Key, T, Hash, KeyEqual> map;
};
