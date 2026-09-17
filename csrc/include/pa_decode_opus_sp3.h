#pragma once

#include "aiter_hip_common.h"
#include "aiter_tensor.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace pa_opus_sp3 {

template<class Value>
struct alignas(16) Slot
{
    Value value{};
};

struct MainArgs
{
    Slot<void*> output, query, key, value, table, lengths, key_scale, value_scale;
    Slot<float> scale_log2;
    Slot<uint32_t> table_stride, kv_heads, query_stride, block_stride, kv_head_stride;
    Slot<uint32_t> mtp, gqa;
    Slot<void*> qtp;
    Slot<uint32_t> num_splits, tiles_per_split;
    Slot<void*> lse;
};

struct ReduceArgs
{
    Slot<void*> output, partial, lse;
    Slot<uint32_t> num_splits, nhead;
};

static_assert(sizeof(MainArgs) == 320 && offsetof(MainArgs, lse) == 304);
static_assert(sizeof(ReduceArgs) == 80 && offsetof(ReduceArgs, num_splits) == 48);

inline bool requested(const aiter_tensor_t& query,
                      const aiter_tensor_t& key,
                      const aiter_tensor_t& value,
                      const aiter_tensor_t& table,
                      const aiter_tensor_t& lengths,
                      const aiter_tensor_t& output,
                      const aiter_tensor_t* key_scale,
                      const aiter_tensor_t* value_scale)
{
    const char* path = std::getenv("PA_DECODE_OPUS_SP3_CO");
    if(path == nullptr || path[0] == '\0' || key_scale == nullptr || value_scale == nullptr)
        return false;
    if((query.dim() != 3 && query.dim() != 4) || key.dim() != 5 ||
       (value.dim() != 4 && value.dim() != 5) || table.dim() != 2 || lengths.dim() != 1)
        return false;
    const int64_t qlen = query.dim() == 4 ? query.size(1) : 1;
    if(query.size(0) < 1 || qlen < 1 || qlen > 4 || query.size(-1) != 128 ||
       key.size(0) < 1 || key.size(1) < 1 || query.size(-2) != key.size(1) * 16 ||
             table.size(1) < 1 || value.size(0) != key.size(0) || value.size(1) != key.size(1) ||
       key.size(2) != 8 || key.size(3) != 16 || key.size(4) != 16 ||
       value.size(-2) != 128 || value.size(-1) != 16 ||
       (value.dim() == 5 && value.size(2) != 1))
        return false;
    if(query.dtype() != AITER_DTYPE_bf16 || output.dtype() != AITER_DTYPE_bf16 ||
       key.dtype() != AITER_DTYPE_fp8 || value.dtype() != AITER_DTYPE_fp8)
        return false;
    for(const auto* tensor : {&query, &key, &value, &table, &lengths, &output,
                              key_scale, value_scale})
        if(!tensor->is_contiguous() || !tensor->is_gpu() || tensor->device_id != query.device_id)
            return false;
    for(const auto* tensor : {&query, &key, &value, &output})
        if(reinterpret_cast<uintptr_t>(tensor->data_ptr()) % 16 != 0)
            return false;
    if(query.numel() * query.element_size() >= (size_t{1} << 31) ||
       table.numel() * table.element_size() >= (size_t{1} << 31) ||
       key_scale->numel() * sizeof(float) >= (size_t{1} << 31))
        return false;
    return true;
}

inline hipFunction_t optional_function(hipModule_t module, const char* name)
{
    hipFunction_t function = nullptr;
    const hipError_t status = hipModuleGetFunction(&function, module, name);
    if(status == hipErrorNotFound)
    {
        (void)hipGetLastError();
        return nullptr;
    }
    HIP_CALL(status);
    return function;
}

inline bool capability(hipModule_t module, const char* name)
{
    hipDeviceptr_t pointer = nullptr;
    size_t size = 0;
    const hipError_t status = hipModuleGetGlobal(&pointer, &size, module, name);
    if(status == hipErrorNotFound)
    {
        (void)hipGetLastError();
        return false;
    }
    HIP_CALL(status);
    AITER_CHECK(size == 4, "Invalid SP3 capability marker: ", name);
    return true;
}

struct Module
{
    hipModule_t handle = nullptr;
    hipFunction_t main = nullptr, query = nullptr, reduce = nullptr, parallel = nullptr;
    bool query3 = false;

    explicit Module(const std::string& path)
    {
        HIP_CALL(hipModuleLoad(&handle, path.c_str()));
        AITER_CHECK(capability(handle, "pa_kernel_func_page16_split_lse_v1"),
                    "SP3 object must use the 320-byte normalized-result/LSE ABI");
        HIP_CALL(hipModuleGetFunction(&main, handle, "pa_kernel_func"));
        HIP_CALL(hipModuleGetFunction(&reduce, handle, "pa_a16w8_q16_d128_p16_mtp_reduce"));
        if(capability(handle, "pa_a16w8_q16_d128_p16_mtp_query_page16_split_lse_v1"))
        {
            HIP_CALL(hipModuleGetFunction(&query, handle, "pa_a16w8_q16_d128_p16_mtp_query"));
            query3 = capability(handle, "pa_a16w8_q16_d128_p16_mtp_query_query3_v1");
        }
        parallel = optional_function(handle, "pa_a16w8_q16_d128_p16_mtp_reduce_parallel8");
    }
};

struct Workspace
{
    void* allocation = nullptr;
    void* partial = nullptr;
    void* lse = nullptr;

    Workspace(int device, int batch, int qlen, int heads, int splits)
    {
        const size_t prefix_bytes = ((static_cast<size_t>(batch) + 1) * sizeof(int) + 15) & ~size_t{15};
        const size_t rows = static_cast<size_t>(batch) * qlen * splits * heads;
        const size_t partial_bytes = splits > 1 ? rows * 128 * sizeof(float) : 0;
        const size_t lse_bytes = splits > 1 ? rows * sizeof(float) : 0;
        HipDeviceGuard guard(device);
        HIP_CALL(hipMalloc(&allocation, prefix_bytes + partial_bytes + lse_bytes));
        std::vector<int> prefix(static_cast<size_t>(batch) + 1);
        for(int index = 0; index <= batch; ++index) prefix[index] = index * qlen;
        HIP_CALL(hipMemcpy(allocation, prefix.data(), prefix.size() * sizeof(int), hipMemcpyHostToDevice));
        if(splits > 1)
        {
            partial = static_cast<char*>(allocation) + prefix_bytes;
            lse = static_cast<char*>(partial) + partial_bytes;
        }
    }
};

struct Registry
{
    std::mutex mutex;
    std::map<std::pair<int, std::string>, std::unique_ptr<Module>> modules;
    std::map<std::tuple<int, uintptr_t, int, int, int, int>, std::unique_ptr<Workspace>> workspaces;
};

template<class Arguments>
inline void launch(hipFunction_t function, Arguments& arguments, dim3 grid,
                   int threads, hipStream_t stream)
{
    size_t size = sizeof(arguments);
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &arguments,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};
    HIP_CALL_LAUNCH(hipModuleLaunchKernel(function, grid.x, grid.y, grid.z,
                                         threads, 1, 1, 0, stream, nullptr, config));
}

inline void run(aiter_tensor_t& query, aiter_tensor_t& key, aiter_tensor_t& value,
                aiter_tensor_t& table, aiter_tensor_t& lengths, aiter_tensor_t& output,
                aiter_tensor_t& key_scale, aiter_tensor_t& value_scale,
                float softmax_scale, int splits, int num_cu, hipStream_t stream)
{
    const int batch = static_cast<int>(query.size(0));
    const int qlen = query.dim() == 4 ? static_cast<int>(query.size(1)) : 1;
    const int heads = static_cast<int>(query.size(-2));
    const int kv_heads = static_cast<int>(key.size(1));
    AITER_CHECK(splits >= 1 && splits <= 256 && get_gpu_arch() == "gfx950",
                "SP3 A16W8 requires gfx950 and splits in [1,256]");
    AITER_CHECK(static_cast<size_t>(heads) * splits * 128 * sizeof(float) < (size_t{1} << 32),
                "SP3 per-query partial stride must fit uint32");
    static Registry* registry = new Registry;
    Module* module = nullptr;
    Workspace* workspace = nullptr;
    {
        std::lock_guard<std::mutex> lock(registry->mutex);
        const auto module_key = std::make_pair(query.device_id, std::string(std::getenv("PA_DECODE_OPUS_SP3_CO")));
        const auto workspace_key = std::make_tuple(query.device_id, reinterpret_cast<uintptr_t>(stream),
                                                   batch, qlen, heads, splits);
        auto module_it = registry->modules.find(module_key);
        auto workspace_it = registry->workspaces.find(workspace_key);
        if(module_it == registry->modules.end() || workspace_it == registry->workspaces.end())
        {
            hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
            HIP_CALL(hipStreamIsCapturing(stream, &capture));
            if(capture != hipStreamCaptureStatusNone)
                throw std::runtime_error(
                    "Warm up Opus-SP3 on this device/stream/shape before Graph capture");
        }
        if(module_it == registry->modules.end())
            module_it = registry->modules.emplace(module_key, std::make_unique<Module>(module_key.second)).first;
        if(workspace_it == registry->workspaces.end())
            workspace_it = registry->workspaces.emplace(workspace_key,
                std::make_unique<Workspace>(query.device_id, batch, qlen, heads, splits)).first;
        module = module_it->second.get();
        workspace = workspace_it->second.get();
    }
    const int total_tiles = static_cast<int>((table.size(1) * 16 + 255) / 256);
    const int tiles_per_split = (total_tiles + splits - 1) / splits;
    const int active_splits = (total_tiles + tiles_per_split - 1) / tiles_per_split;
    const bool query_split = module->query != nullptr && kv_heads == 1 && qlen > 1 &&
        (qlen != 3 || module->query3) && static_cast<int64_t>(batch) * kv_heads * splits * qlen <= 2LL * num_cu;
    const bool parallel = module->parallel != nullptr && splits > 64 && active_splits >= 64 &&
        static_cast<int64_t>(batch) * qlen * heads <= num_cu;
    MainArgs arguments{};
    arguments.output.value = splits > 1 ? workspace->partial : output.data_ptr();
    arguments.query.value = query.data_ptr();
    arguments.key.value = key.data_ptr();
    arguments.value.value = value.data_ptr();
    arguments.table.value = table.data_ptr();
    arguments.lengths.value = lengths.data_ptr();
    arguments.key_scale.value = key_scale.data_ptr();
    arguments.value_scale.value = value_scale.data_ptr();
    arguments.scale_log2.value = softmax_scale * 1.4426950408889634f;
    arguments.table_stride.value = static_cast<uint32_t>(table.size(1));
    arguments.kv_heads.value = kv_heads;
    arguments.query_stride.value = heads * 128 * sizeof(uint16_t);
    arguments.block_stride.value = static_cast<uint32_t>(key.stride(0));
    arguments.kv_head_stride.value = static_cast<uint32_t>(key.stride(1));
    arguments.mtp.value = qlen - 1;
    arguments.gqa.value = 16;
    arguments.qtp.value = workspace->allocation;
    arguments.num_splits.value = splits;
    arguments.tiles_per_split.value = tiles_per_split;
    arguments.lse.value = workspace->lse;
    launch(query_split ? module->query : module->main, arguments,
           dim3(kv_heads * (query_split ? qlen : 1), batch, splits), 256, stream);
    if(splits > 1)
    {
        ReduceArgs reduce{};
        reduce.output.value = output.data_ptr();
        reduce.partial.value = workspace->partial;
        reduce.lse.value = workspace->lse;
        reduce.num_splits.value = splits;
        reduce.nhead.value = heads;
        launch(parallel ? module->parallel : module->reduce, reduce,
               dim3(heads, batch * qlen, 1), parallel ? 512 : 64, stream);
    }
    if(std::getenv("PA_DECODE_OPUS_PRINT_SPLITS"))
        std::fprintf(stderr, "[opus-sp3] batch=%d nkv=%d qlen=%d splits=%d qsplit=%d reduce_threads=%d\n",
                     batch, kv_heads, qlen, splits, query_split ? qlen : 1, splits > 1 ? (parallel ? 512 : 64) : 0);
}

} // namespace pa_opus_sp3