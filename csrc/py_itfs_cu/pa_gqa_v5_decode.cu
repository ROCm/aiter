// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// pa_gqa v5 decode launcher.  Same .co loading + dispatch pattern as
// csrc/py_itfs_cu/asm_topk_per_row_decode.cu — `AiterAsmKernel` resolves
// `<AITER_ASM_DIR>/<arch>/pa_gqa_v5/asm_pa_gqa_v5.co` at first use.
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/extension.h>

#include "aiter_hip_common.h"
#include "pa_gqa_v5.h"

namespace {

// HSA argument layouts (verified against `llvm-readelf --notes` of
// asm_pa_gqa_v5.co; do NOT reorder).
struct __attribute__((packed)) MainKernelArgs
{
    void*    q;                       // off  0
    void*    k_cache;                 // off  8
    void*    v_cache;                 // off 16
    float    scale;                   // off 24
    char     _pad0[4];                // off 28..31
    void*    block_tables;            // off 32
    void*    context_lens;            // off 40
    int32_t  max_num_blocks_per_seq;  // off 48
    int32_t  q_stride;                // off 52
    int32_t  kv_block_stride;         // off 56
    int32_t  kv_head_stride;          // off 60
    void*    exp_sums;                // off 64
    void*    max_logits;              // off 72
    void*    out_tmp;                 // off 80
};
static_assert(sizeof(MainKernelArgs) == 88, "MainKernelArgs layout drift");

struct __attribute__((packed)) ReduceKernelArgs
{
    void*    out;                     // off  0
    void*    exp_sums;                // off  8
    void*    max_logits;              // off 16
    void*    tmp_out;                 // off 24
    void*    context_lens;            // off 32
    int32_t  max_num_partitions;      // off 40
};
static_assert(sizeof(ReduceKernelArgs) == 44, "ReduceKernelArgs layout drift");

// Hard-coded block sizes (must match the .co kernels' __launch_bounds__).
constexpr int kMainNumThreads   = 256;   // pa_gqa::v5::kNumThreads
constexpr int kReduceNumThreads = 128;   // pa_gqa::v1::kReduceNumThreads

// Mangled symbol names exported by asm_pa_gqa_v5.co
// (verify with `llvm-readelf --syms`).  The reduce mangling encodes the
// non-type template args: kNL=32, kPartSize=256.
constexpr const char* kMainSymbol =
    "_ZN6pa_gqa21pa_gqa_main_kernel_v5I14__hip_bfloat16S1_EEvPKT_S4_S4_fPKiS6_iiiiPfS7_PT0_";
constexpr const char* kReduceSymbol =
    "_ZN6pa_gqa23pa_gqa_reduce_kernel_v1I14__hip_bfloat16Li32ELi256EEEvPT_PKfS5_PKS2_PKii";
constexpr const char* kCoPath = "/pa_gqa_v5/asm_pa_gqa_v5.co";

}  // namespace

void pa_gqa_v5_decode(torch::Tensor& out,
                      torch::Tensor& exp_sums,
                      torch::Tensor& max_logits,
                      torch::Tensor& tmp_out,
                      const torch::Tensor& query,
                      const torch::Tensor& key_cache,
                      const torch::Tensor& value_cache,
                      const torch::Tensor& block_tables,
                      const torch::Tensor& context_lens,
                      int64_t num_kv_heads,
                      double scale,
                      int64_t block_size,
                      int64_t max_context_len,
                      int64_t partition_size,
                      int64_t mtp)
{
    TORCH_CHECK(query.dtype() == at::kBFloat16 &&
                    key_cache.dtype() == at::kBFloat16 &&
                    value_cache.dtype() == at::kBFloat16,
                "pa_gqa_v5_decode requires bf16 Q/K/V");

    const int64_t num_query_tokens = query.size(0);     // num_seqs * mtp
    const int64_t num_heads        = query.size(1);
    const int64_t head_size        = query.size(2);
    const int64_t num_seqs         = num_query_tokens / mtp;

    TORCH_CHECK(num_kv_heads == 1 && num_heads == 8 && head_size == 128 &&
                    block_size == 16 && mtp == 2 && partition_size == 256,
                "pa_gqa_v5_decode is hard-specialised for "
                "(num_kv_heads=1, num_heads=8, head=128, block=16, mtp=2, "
                "partition_size=256); got (",
                num_kv_heads, ",", num_heads, ",", head_size, ",",
                block_size, ",", mtp, ",", partition_size, ")");

    const int64_t max_num_partitions =
        (max_context_len + partition_size - 1) / partition_size;
    TORCH_CHECK(max_num_partitions <= 32 * 64,
                "pa_gqa_v5_decode reduce kernel is kNL=32 → max 2048 "
                "partitions (524288 ctx @ partition_size=256); got ",
                max_num_partitions);

    const c10::hip::OptionalHIPGuardMasqueradingAsCUDA guard(query.device());
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    // ---- main kernel ----
    {
        MainKernelArgs args{};
        args.q                      = const_cast<void*>(query.data_ptr());
        args.k_cache                = const_cast<void*>(key_cache.data_ptr());
        args.v_cache                = const_cast<void*>(value_cache.data_ptr());
        args.scale                  = static_cast<float>(scale);
        args.block_tables           = const_cast<void*>(block_tables.data_ptr());
        args.context_lens           = const_cast<void*>(context_lens.data_ptr());
        args.max_num_blocks_per_seq = static_cast<int32_t>(block_tables.size(1));
        args.q_stride               = static_cast<int32_t>(query.stride(0));
        args.kv_block_stride        = static_cast<int32_t>(key_cache.stride(0));
        args.kv_head_stride         = static_cast<int32_t>(key_cache.stride(1));
        args.exp_sums               = exp_sums.data_ptr();
        args.max_logits             = max_logits.data_ptr();
        args.out_tmp                = tmp_out.data_ptr();
        size_t arg_size             = sizeof(args);

        static AiterAsmKernel impl_main(kMainSymbol, kCoPath);
        impl_main.launch_kernel({&args, &arg_size,
                                 static_cast<int>(num_seqs),
                                 static_cast<int>(max_num_partitions),
                                 static_cast<int>(num_kv_heads),
                                 kMainNumThreads, 1, 1,
                                 stream});
    }

    // ---- reduce kernel ----
    {
        ReduceKernelArgs rargs{};
        rargs.out                = out.data_ptr();
        rargs.exp_sums           = exp_sums.data_ptr();
        rargs.max_logits         = max_logits.data_ptr();
        rargs.tmp_out            = tmp_out.data_ptr();
        rargs.context_lens       = const_cast<void*>(context_lens.data_ptr());
        rargs.max_num_partitions = static_cast<int32_t>(max_num_partitions);
        size_t arg_size          = sizeof(rargs);

        static AiterAsmKernel impl_reduce(kReduceSymbol, kCoPath);
        impl_reduce.launch_kernel({&rargs, &arg_size,
                                   static_cast<int>(num_heads),
                                   static_cast<int>(num_seqs),
                                   static_cast<int>(mtp),
                                   kReduceNumThreads, 1, 1,
                                   stream});
    }
}
