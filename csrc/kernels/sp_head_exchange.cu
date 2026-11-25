// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_stream.h"
#include "aiter_tensor.h"
#include "custom_all_reduce.cuh"
#include <optional>

namespace aiter {

// All-to-all of row blocks over the custom-allreduce IPC machinery.
//
// Every rank holds a [W * rows, C] matrix (rows of row_packs 16-byte packs) in
// an IPC-registered buffer. Row block j goes to rank j. This rank pulls block
// `rank` of every peer's matrix and writes it to
//   kLastDim = true : out[row, peer * C + c]      ([rows, W * C], the layout a
//                                                  GEMM over the gathered heads reads)
//   kLastDim = false: out[peer * rows + row, c]   (all_to_all_single layout)
// One launch moves up to kA2AFields matrices (for example block-scale codes and
// their scales). Same synchronization protocol as allgather_lastdim: start_sync
// before the pulls, a final end_sync after them, so when the launch is done on a
// rank no peer still reads its buffer and the buffer can be rewritten.
constexpr int kA2AFields = 2;
constexpr int kA2AUnroll = 4;

struct A2AField
{
    const RankData* src;  // peer base pointers of the registered buffer
    int64_t src_pack_off; // 16-byte packs from a peer's base to the field start
    uint4* out;
    int64_t rows;      // rows of one block (rows per destination rank)
    int64_t row_packs; // 16-byte packs per row
};

struct A2AArgs
{
    A2AField field[kA2AFields];
    int nfields;
};

template <int W, bool kLastDim>
__global__ void __launch_bounds__(512, 1)
    all_to_all_kernel(RankSignals signals, Signal* self, A2AArgs args, int rank)
{
    constexpr int threads_per_peer = 512 / W;
    const int peer                 = threadIdx.x / threads_per_peer;
    const int lane                 = threadIdx.x % threads_per_peer;
    const int64_t stride           = (int64_t)gridDim.x * threads_per_peer;
    start_sync<W>(signals, self, rank);
    for(int f = 0; f < args.nfields; ++f)
    {
        const A2AField& fld = args.field[f];
        const int64_t total = fld.rows * fld.row_packs;
        const uint4* src = reinterpret_cast<const uint4*>(fld.src->ptrs[peer]) + fld.src_pack_off +
                           (int64_t)rank * total;
        for(int64_t base = (int64_t)blockIdx.x * threads_per_peer + lane; base < total;
            base += stride * kA2AUnroll)
        {
            uint4 v[kA2AUnroll];
#pragma unroll
            for(int u = 0; u < kA2AUnroll; ++u)
            {
                const int64_t i = base + u * stride;
                if(i < total)
                    v[u] = src[i];
            }
#pragma unroll
            for(int u = 0; u < kA2AUnroll; ++u)
            {
                const int64_t i = base + u * stride;
                if(i < total)
                {
                    if constexpr(kLastDim)
                    {
                        const int64_t row                               = i / fld.row_packs;
                        const int64_t col                               = i - row * fld.row_packs;
                        fld.out[(row * W + peer) * fld.row_packs + col] = v[u];
                    }
                    else
                    {
                        fld.out[(int64_t)peer * total + i] = v[u];
                    }
                }
            }
        }
    }
    end_sync<W, true>(signals, self, rank);
}

namespace {

// Where peers find a source matrix: inside the registered pool it is the pool
// base plus an offset, which is the same on every rank because the pool is
// registered at the same layout everywhere; anywhere else the matrix must be
// registered itself (register_input_buffer, or captured into a graph).
void a2a_source(CustomAllreduce* comm,
                hipStream_t stream,
                void* ptr,
                int64_t pool,
                int64_t pool_bytes,
                A2AField& f)
{
    const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
    if(pool != 0 && p >= (uintptr_t)pool && p < (uintptr_t)pool + (uintptr_t)pool_bytes)
    {
        f.src          = comm->get_buffer_RD(stream, reinterpret_cast<void*>(pool));
        f.src_pack_off = (int64_t)(p - (uintptr_t)pool) / 16;
    }
    else
    {
        f.src          = comm->get_buffer_RD(stream, ptr);
        f.src_pack_off = 0;
    }
}

} // namespace

// Exchange the row blocks of `input` (and of `scales`, when given) between all
// ranks of the communicator. See all_to_all in aiter/dist for the contract.
//   stage = true : copy the inputs into the registered pool (reg_buffer,
//                  reg_bytes) first, `scales` right behind `input` at the next
//                  16-byte boundary.
//   stage = false: the inputs are already registered: views of the pool (a
//                  producer wrote them there) or tensors registered with the
//                  communicator / captured into a graph.
void all_to_all(int64_t handle,
                const aiter_tensor_t& input,
                const aiter_tensor_t& output,
                int64_t registered_buffer,
                int64_t registered_bytes,
                bool stage,
                bool gather_last,
                int64_t blocks,
                std::optional<aiter_tensor_t> scales,
                std::optional<aiter_tensor_t> scales_out)
{
    HipDeviceGuard guard(input.device_id);
    auto* comm  = reinterpret_cast<CustomAllreduce*>(handle);
    const int W = comm->world_size_;
    AITER_CHECK(W == 2 || W == 4 || W == 8, "all_to_all supports 2, 4 or 8 ranks");
    AITER_CHECK(scales.has_value() == scales_out.has_value(),
                "all_to_all needs scales and scales_out together");
    AITER_CHECK(blocks >= 1 && blocks <= kMaxBlocks, "invalid sync block count");
    const int nfields                       = scales.has_value() ? 2 : 1;
    const aiter_tensor_t* in_t[kA2AFields]  = {&input, scales ? &*scales : nullptr};
    const aiter_tensor_t* out_t[kA2AFields] = {&output, scales_out ? &*scales_out : nullptr};
    auto stream                             = getCurrentHIPStream();

    A2AArgs args{};
    args.nfields      = nfields;
    int64_t stage_off = 0;
    for(int f = 0; f < nfields; ++f)
    {
        const aiter_tensor_t& in  = *in_t[f];
        const aiter_tensor_t& out = *out_t[f];
        AITER_CHECK(in.dim() == 2 && in.is_contiguous() && out.is_contiguous(),
                    "all_to_all requires contiguous 2D inputs");
        AITER_CHECK(in.dtype() == out.dtype() && in.device_id == out.device_id &&
                        in.device_id == input.device_id,
                    "all_to_all dtype or device mismatch");
        const int64_t row_bytes = in.size(1) * in.element_size();
        AITER_CHECK(in.size(0) % W == 0, "all_to_all requires rows divisible by world size");
        AITER_CHECK(row_bytes % 16 == 0, "all_to_all requires 16-byte rows");
        const int64_t rows = in.size(0) / W;
        const bool ok_out =
            gather_last
                ? (out.dim() == 2 && out.size(0) == rows && out.size(1) == in.size(1) * W)
                : (out.dim() == 2 && out.size(0) == in.size(0) && out.size(1) == in.size(1));
        AITER_CHECK(ok_out, "all_to_all output layout mismatch");
        AITER_CHECK(reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
                    "all_to_all output requires 16-byte alignment");
        AITER_CHECK(registered_buffer == 0 ||
                        reinterpret_cast<uintptr_t>(out.data_ptr()) +
                                (uintptr_t)(out.numel() * out.element_size()) <=
                            (uintptr_t)registered_buffer ||
                        reinterpret_cast<uintptr_t>(out.data_ptr()) >=
                            (uintptr_t)registered_buffer + (uintptr_t)registered_bytes,
                    "all_to_all output must not be inside the registered pool");
        const int64_t bytes = in.numel() * in.element_size();
        void* source        = in.data_ptr();
        if(stage)
        {
            AITER_CHECK(registered_buffer != 0 && (uintptr_t)registered_buffer % 16 == 0,
                        "all_to_all needs a 16-byte aligned registered pool to stage into");
            AITER_CHECK(stage_off + bytes <= registered_bytes,
                        "all_to_all registered pool too small");
            source = reinterpret_cast<char*>(registered_buffer) + stage_off;
            if(bytes != 0)
                HIP_CALL(
                    hipMemcpyAsync(source, in.data_ptr(), bytes, hipMemcpyDeviceToDevice, stream));
            stage_off += (bytes + 15) / 16 * 16;
        }
        else
        {
            AITER_CHECK(reinterpret_cast<uintptr_t>(source) % 16 == 0,
                        "registered all_to_all input requires 16-byte alignment");
        }
        A2AField& fld = args.field[f];
        a2a_source(comm, stream, source, registered_buffer, registered_bytes, fld);
        fld.out       = reinterpret_cast<uint4*>(out.data_ptr());
        fld.rows      = rows;
        fld.row_packs = row_bytes / 16;
    }
#define LAUNCH_A2A(NG)                                                                      \
    case NG:                                                                                \
        if(gather_last)                                                                     \
            all_to_all_kernel<NG, true>                                                     \
                <<<blocks, 512, 0, stream>>>(comm->sg_, comm->self_sg_, args, comm->rank_); \
        else                                                                                \
            all_to_all_kernel<NG, false>                                                    \
                <<<blocks, 512, 0, stream>>>(comm->sg_, comm->self_sg_, args, comm->rank_); \
        break;
    switch(W)
    {
        LAUNCH_A2A(2)
        LAUNCH_A2A(4)
        LAUNCH_A2A(8)
    }
#undef LAUNCH_A2A
}

// TP4 head exchange: [tokens, heads] -> [tokens / 4, heads * 4], the original
// entry point of this module. It is all_to_all with gather_last = true.
void sp_head_exchange(int64_t handle,
                      const aiter_tensor_t& input,
                      const aiter_tensor_t& output,
                      int64_t registered_buffer,
                      int64_t registered_bytes,
                      bool stage,
                      int64_t blocks)
{
    auto* comm = reinterpret_cast<CustomAllreduce*>(handle);
    AITER_CHECK(comm->world_size_ == 4, "head exchange requires four ranks");
    AITER_CHECK(input.dim() == 2 && input.is_contiguous() && output.is_contiguous(),
                "head exchange requires contiguous matrices");
    AITER_CHECK(input.size(0) % 4 == 0 && input.size(1) * input.element_size() % 16 == 0,
                "head exchange requires padded tokens and 16-byte head width");
    AITER_CHECK(stage || reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0,
                "registered head exchange input requires 16-byte alignment");
    AITER_CHECK(!stage || (registered_buffer != 0 && registered_buffer % 16 == 0),
                "staged head exchange buffer requires 16-byte alignment");
    all_to_all(handle,
               input,
               output,
               registered_buffer,
               registered_bytes,
               stage,
               true,
               blocks,
               std::nullopt,
               std::nullopt);
}
} // namespace aiter
