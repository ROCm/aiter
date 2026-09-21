#include "aiter_tensor.h"

#include <cstddef>
#include <limits>

struct __attribute__((packed)) PaPsReduceArgs
{
    void* output;
    p2 padding0;
    void* partial_output;
    p2 padding1;
    void* partial_lse;
    p2 padding2;
    void* reduce_indptr;
    p2 padding3;
    void* reduce_final_map;
    p2 padding4;
    void* reduce_partial_map;
    p2 padding5;
    void* final_lse;
    p2 padding6;
    uint32_t num_heads;
    p3 padding7;
    uint32_t num_tiles;
    p3 padding8;
    uint32_t output_query_stride_bytes;
    p3 padding9;
    uint32_t output_head_stride_bytes;
    p3 padding10;
    uint32_t tile_stride;
    p3 padding11;
    uint32_t query_stride;
    p3 padding12;
};

static_assert(sizeof(PaPsReduceArgs) == 208);
static_assert(offsetof(PaPsReduceArgs, partial_output) == 0x10);
static_assert(offsetof(PaPsReduceArgs, partial_lse) == 0x20);
static_assert(offsetof(PaPsReduceArgs, reduce_indptr) == 0x30);
static_assert(offsetof(PaPsReduceArgs, reduce_final_map) == 0x40);
static_assert(offsetof(PaPsReduceArgs, reduce_partial_map) == 0x50);
static_assert(offsetof(PaPsReduceArgs, final_lse) == 0x60);
static_assert(offsetof(PaPsReduceArgs, num_heads) == 0x70);
static_assert(offsetof(PaPsReduceArgs, num_tiles) == 0x80);
static_assert(offsetof(PaPsReduceArgs, output_query_stride_bytes) == 0x90);
static_assert(offsetof(PaPsReduceArgs, output_head_stride_bytes) == 0xA0);
static_assert(offsetof(PaPsReduceArgs, tile_stride) == 0xB0);
static_assert(offsetof(PaPsReduceArgs, query_stride) == 0xC0);

AITER_C_ITFS
void pa_ps_reduce(aiter_tensor_t* partial_output,
                  aiter_tensor_t* partial_lse,
                  aiter_tensor_t* reduce_indptr,
                  aiter_tensor_t* reduce_final_map,
                  aiter_tensor_t* reduce_partial_map,
                  int max_seqlen_q,
                  aiter_tensor_t* final_output,
                  aiter_tensor_t* final_lse,
                  hipStream_t stream)
{
    AITER_CHECK(partial_output && partial_lse && reduce_indptr && reduce_final_map &&
                    reduce_partial_map && final_output,
                __func__, ": explicit PS metadata and output tensors are required");
    AITER_CHECK(final_output->is_gpu(), __func__, ": output must be on a GPU");
    const HipDeviceGuard device_guard(final_output->device_id);
    AITER_CHECK(get_gpu_arch() == "gfx950", __func__, ": requires gfx950");
    for(const auto* tensor : {partial_output, partial_lse, reduce_indptr, reduce_final_map,
                              reduce_partial_map})
    {
        AITER_CHECK(tensor->device_id == final_output->device_id && tensor->is_contiguous(),
                    __func__, ": inputs must be contiguous and on the output device");
    }
    AITER_CHECK(final_output->dim() == 3 && final_output->size(2) == 128 &&
                    final_output->stride(2) == 1,
                __func__, ": output must have shape [queries, heads, 128]");
    AITER_CHECK(final_output->dtype() == AITER_DTYPE_bf16 ||
                    final_output->dtype() == AITER_DTYPE_fp16,
                __func__, ": output must be BF16 or FP16");
    AITER_CHECK(partial_output->dim() >= 3 && partial_output->size(-1) == 128 &&
                    partial_output->size(-2) == final_output->size(1) &&
                    partial_output->dtype() == AITER_DTYPE_fp32 &&
                    partial_lse->dtype() == AITER_DTYPE_fp32 &&
                    partial_output->numel() / 128 == partial_lse->numel(),
                __func__, ": partial output/LSE must have matching packed FP32 layouts");
    AITER_CHECK(reduce_indptr->dtype() == AITER_DTYPE_i32 && reduce_indptr->dim() == 1 &&
                    reduce_indptr->size(0) >= 1 &&
                    reduce_final_map->dtype() == AITER_DTYPE_i32 &&
                    reduce_final_map->dim() == 2 && reduce_final_map->size(1) == 2 &&
                    reduce_final_map->size(0) >= reduce_indptr->size(0) - 1 &&
                    reduce_partial_map->dtype() == AITER_DTYPE_i32 &&
                    reduce_partial_map->dim() == 1,
                __func__, ": invalid reduce metadata shape or dtype");
    AITER_CHECK(max_seqlen_q > 0, __func__, ": max_seqlen_q must be positive");
    const int64_t num_heads = final_output->size(1);
    const int64_t num_tiles = reduce_indptr->size(0) - 1;
    if(final_lse)
    {
        AITER_CHECK(final_lse->device_id == final_output->device_id &&
                        final_lse->dtype() == AITER_DTYPE_fp32 && final_lse->is_contiguous() &&
                        final_lse->dim() == 2 &&
                        final_lse->size(0) == final_output->size(0) &&
                        final_lse->size(1) == num_heads,
                    __func__, ": final_lse must be contiguous FP32 [queries, heads]");
    }
    if(num_tiles == 0 || final_output->numel() == 0)
        return;

    constexpr int64_t max_u32 = std::numeric_limits<uint32_t>::max();
    constexpr int64_t max_i32 = std::numeric_limits<int32_t>::max();
    AITER_CHECK(num_heads > 0 && num_heads <= max_u32 / (128 * sizeof(float)) &&
                    num_tiles <= max_u32 / 8 &&
                    final_output->size(0) <= max_i32 &&
                    partial_output->numel() / (num_heads * 128) <= size_t(max_i32) &&
                    reduce_partial_map->numel() <= size_t(max_u32 / 4),
                __func__, ": shape exceeds the reducer's 32-bit index range");
    const int64_t head_stride = final_output->stride(1);
    const int64_t query_stride = final_output->stride(0);
    AITER_CHECK(head_stride >= 128 && head_stride <= max_u32 / 2 &&
                    query_stride >= (num_heads - 1) * head_stride + 128 &&
                    query_stride <= max_u32 / 2,
                __func__, ": output must have non-overlapping rows with 32-bit byte strides");

    const uint32_t query_groups = static_cast<uint32_t>(std::min(max_seqlen_q, 4));
    const uint32_t target_groups = get_num_cu_func() * 16;
    const uint32_t groups_per_tile = static_cast<uint32_t>(num_heads) * query_groups;
    const uint32_t tile_groups = static_cast<uint32_t>(std::min<int64_t>(
        num_tiles, std::max<uint32_t>(1, (target_groups + groups_per_tile - 1) / groups_per_tile)));

    PaPsReduceArgs args{};
    args.output = final_output->data_ptr();
    args.partial_output = partial_output->data_ptr();
    args.partial_lse = partial_lse->data_ptr();
    args.reduce_indptr = reduce_indptr->data_ptr();
    args.reduce_final_map = reduce_final_map->data_ptr();
    args.reduce_partial_map = reduce_partial_map->data_ptr();
    args.final_lse = final_lse ? final_lse->data_ptr() : nullptr;
    args.num_heads = static_cast<uint32_t>(num_heads);
    args.num_tiles = static_cast<uint32_t>(num_tiles);
    args.output_query_stride_bytes = static_cast<uint32_t>(query_stride * 2);
    args.output_head_stride_bytes = static_cast<uint32_t>(head_stride * 2);
    args.tile_stride = tile_groups;
    args.query_stride = query_groups;

    AiterAsmKernel* kernel = nullptr;
    if(final_output->dtype() == AITER_DTYPE_fp16)
    {
        static AiterAsmKernel implementation("pa_p16_d128_reduce_ps_fp16",
                                              "pa/pa_p16_d128_reduce_ps_fp16.co");
        kernel = &implementation;
    }
    else
    {
        static AiterAsmKernel implementation("pa_p16_d128_reduce_ps_bf16",
                                              "pa/pa_p16_d128_reduce_ps_bf16.co");
        kernel = &implementation;
    }
    size_t argument_size = sizeof(args);
    kernel->launch_kernel({&args, &argument_size, static_cast<int>(num_heads),
                            static_cast<int>(query_groups), static_cast<int>(tile_groups),
                            64, 1, 1, stream});
}
