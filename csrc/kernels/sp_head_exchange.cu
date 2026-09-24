#include "custom_all_reduce.cuh"
#include "aiter_stream.h"
#include "aiter_tensor.h"

namespace aiter {
// Same synchronization protocol as allgather_lastdim, but each destination
// reads only its token slice and writes the final token-major head layout.
template<int W>
__global__ void __launch_bounds__(512, 1) sp_head_exchange_kernel(
    RankData* dp, RankSignals signals, Signal* self, uint4* output,
    int rank, int local_packs, int width_packs)
{
    constexpr int threads_per_peer = 512 / W;
    const int peer = threadIdx.x / threads_per_peer;
    const int lane = threadIdx.x % threads_per_peer;
    const uint4* source = reinterpret_cast<const uint4*>(dp->ptrs[peer]);
    start_sync<W>(signals, self, rank);
    for(int i = blockIdx.x * threads_per_peer + lane;
        i < local_packs; i += gridDim.x * threads_per_peer)
    {
        const int row = i / width_packs;
        const int col = i % width_packs;
        output[(row * W + peer) * width_packs + col] = source[rank * local_packs + i];
    }
    end_sync<W, true>(signals, self, rank);
}

void sp_head_exchange(int64_t handle, const aiter_tensor_t& input,
                      const aiter_tensor_t& output, int64_t registered_buffer,
                      int64_t registered_bytes, bool stage, int64_t blocks)
{
    HipDeviceGuard guard(input.device_id);
    auto* comm = reinterpret_cast<CustomAllreduce*>(handle);
    AITER_CHECK(comm->world_size_ == 4, "head exchange requires four ranks");
    AITER_CHECK(input.dim() == 2 && input.is_contiguous() && output.is_contiguous(),
                "head exchange requires contiguous matrices");
    AITER_CHECK(output.device_id == input.device_id, "head exchange device mismatch");
    const int64_t bytes = input.numel() * input.element_size();
    const int64_t row_bytes = input.size(1) * input.element_size();
    AITER_CHECK(input.size(0) % 4 == 0 && row_bytes % 16 == 0,
                "head exchange requires padded tokens and 16-byte head width");
    AITER_CHECK(output.dim() == 2 && output.size(0) == input.size(0) / 4 &&
                output.size(1) == input.size(1) * 4 && output.dtype() == input.dtype(),
                "head exchange output layout mismatch");
    AITER_CHECK(blocks >= 1 && blocks <= kMaxBlocks, "invalid sync block count");
    AITER_CHECK(reinterpret_cast<uintptr_t>(output.data_ptr()) % 16 == 0,
                "head exchange output requires 16-byte alignment");
    AITER_CHECK(stage || reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0,
                "registered head exchange input requires 16-byte alignment");
    AITER_CHECK(bytes / 4 / 16 <= INT32_MAX && row_bytes / 16 <= INT32_MAX,
                "head exchange dimensions exceed kernel indexing range");
    auto stream = getCurrentHIPStream();
    void* source = input.data_ptr();
    if(stage) {
        AITER_CHECK(registered_buffer != 0 && bytes <= registered_bytes,
                    "head exchange registered pool too small");
        source = reinterpret_cast<void*>(registered_buffer);
        HIP_CALL(hipMemcpyAsync(source, input.data_ptr(), bytes, hipMemcpyDeviceToDevice, stream));
    }
    auto* pointers = comm->get_buffer_RD(stream, source);
    sp_head_exchange_kernel<4><<<blocks, 512, 0, stream>>>(pointers, comm->sg_,
        comm->self_sg_, reinterpret_cast<uint4*>(output.data_ptr()), comm->rank_,
        bytes / 4 / 16, row_bytes / 16);
}
}
