// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"
#include "asm_pa_configs.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>

struct __attribute__((packed)) KernelArgs
{
    void* ptr_O;
    p2 _p0;
    void* ptr_Q;
    p2 _p1;
    void* ptr_K;
    p2 _p2;
    void* ptr_V;
    p2 _p3;
    void* ptr_BT;
    p2 _p4;
    void* ptr_CL;
    p2 _p5;
    void* ptr_KQ;
    p2 _p6;
    void* ptr_VQ;
    p2 _p7;
    float sclg2e;
    p3 _p12;
    unsigned int mblk;
    p3 _p13;
    unsigned int kv_nheads;
    p3 _p14;
    unsigned int Qs;
    p3 _p15;
    unsigned int Bs;
    p3 _p16;
    unsigned int KVs;
    p3 _p17;
    unsigned int mtp;
    p3 _p18;
    unsigned int GQA;
    p3 _p19;
    void* ptr_QTP;
    p2 _p20;
};


struct __attribute__((packed)) PsKernelArgs
{
    void* ptr_O;
    p2 _p0;
    void* ptr_Q;
    p2 _p1;
    void* ptr_K;
    p2 _p2;
    void* ptr_V;
    p2 _p3;
    void *ptr_KVIndices;
    p2 _p4;
    void *ptr_CL;
    p2 _p5;
    void *ptr_KQ;
    p2 _p6;
    void *ptr_VQ;
    p2 _p7;
    float sclg2e;
    p3 _p12;
    unsigned int kv_nheads;
    p3 _p14;
    unsigned int Qs;
    p3 _p15;
    unsigned int Bs;
    p3 _p16;
    unsigned int KVs;
    p3 _p17;
    unsigned int mtp;
    p3 _p18;
    unsigned int GQA;
    p3 _p19;
    void *ptr_QOPtr;
    p2 _p20;
    void *ptr_KVPtr;
    p2 _p21;
    void *ptr_WorkPtr;
    p2 _p22;
    void *ptr_WorkInfo;
    p2 _p23;
    void *ptr_SplitO;
    p2 _p24;
    void *ptr_SplitLSE;
    p2 _p25;
    unsigned int stride_scale_blk;
    p3 _p26;
    unsigned int stride_scale_page;
    p3 _p27;
};


std::string get_heuristic_kernel(std::string q_type,
                                 std::string kv_type,
                                 int gqa,
                                 int mtp,
                                 int msk,
                                 int hp,
                                 int block_size,
                                 std::string arch_id,
                                 int ps,
                                 int qTile,
                                 int quant_type,
                                 CFG* cfgs)
{
    // # mtp * gqa <= 16
    // # gpa = 16, mtp 1
    // # qlen = mtp + 1
    // # qlen * gqa <=16

    const std::vector<int> mtp_flags = (mtp > 0) ? std::vector<int>{mtp, 1} : std::vector<int>{0};
    const std::vector<int> gqa_flags = {gqa, (gqa + 7) / 8 * 8};
    for(int mtp_ : mtp_flags)
    {
        for(int gqa_ : gqa_flags)
        {
            // find exact match
            for(const auto& el : *cfgs)
            {
                if (el.first.find(arch_id) != 0)
                    continue;
                const auto& cfg = el.second;
                // hp is just distinct from uhp
                if(cfg.qType == q_type && cfg.kvType == kv_type && cfg.Gqa == gqa_ &&
                   cfg.Mtp == mtp_ && cfg.Msk == msk && (cfg.Hp == hp || hp == 1) &&
                   cfg.blkSz == block_size && cfg.ps == ps && cfg.qTile == qTile && cfg.quant_type == quant_type)

                    return el.first;
            }
        }
    }

    AITER_CHECK(false,
                __func__,
                ": cannot get heuristic kernel!"
                " q_type:",
                q_type,
                " kv_type:",
                kv_type,
                " gqa:",
                gqa,
                " mtp:",
                mtp,
                " msk:",
                msk,
                " hp:",
                hp,
                " block_size:",
                block_size,
                " ps:",
                ps,
                " qTile:",
                qTile,
                " quant_type:",
                quant_type);
    return "";
}

namespace pa_asm_split {

struct __attribute__((packed)) MainArgs
{
    KernelArgs base;
    uint32_t num_splits;
    p3 split_padding;
    uint32_t tiles_per_split;
    p3 tiles_padding;
    void* lse;
    p2 lse_padding;
};

struct __attribute__((packed)) ReduceArgs
{
    void* output;
    p2 output_padding;
    void* partial;
    p2 partial_padding;
    void* lse;
    p2 lse_padding;
    uint32_t num_splits;
    p3 split_padding;
    uint32_t heads;
    p3 heads_padding;
};

static_assert(sizeof(KernelArgs) == 272);
static_assert(sizeof(MainArgs) == 320 && offsetof(MainArgs, num_splits) == 272 &&
              offsetof(MainArgs, lse) == 304);
static_assert(sizeof(ReduceArgs) == 80 && offsetof(ReduceArgs, num_splits) == 48);

bool supported(const aiter_tensor_t& query, const aiter_tensor_t& key,
               const aiter_tensor_t& value, const aiter_tensor_t& table,
               const aiter_tensor_t& lengths, const aiter_tensor_t& output,
               const aiter_tensor_t* key_scale, const aiter_tensor_t* value_scale,
               const aiter_tensor_t* qtp, int table_stride, int max_qlen, int precision)
{
    if(max_qlen < 1 || max_qlen > 4 || precision < 0 || precision > 1 ||
       key_scale == nullptr || value_scale == nullptr || query.dim() != 3 ||
       key.dim() != 5 || (value.dim() != 4 && value.dim() != 5) ||
       output.dim() != 3 || table.dim() != 2 || lengths.dim() != 1)
        return false;
    const int64_t batch = lengths.size(0);
    if(batch < 1 || query.size(0) < batch || query.size(0) > batch * max_qlen ||
       (qtp == nullptr && query.size(0) != batch * max_qlen) ||
       query.size(2) != 128 || key.size(0) < 1 || key.size(1) < 1 ||
       query.size(1) != key.size(1) * 16 || key.size(2) != 8 ||
       key.size(3) != 16 || key.size(4) != 16 ||
       value.size(0) != key.size(0) || value.size(1) != key.size(1) ||
       value.size(-2) != 128 || value.size(-1) != 16 ||
       (value.dim() == 5 && value.size(2) != 1) ||
       table.size(0) != batch || table.size(1) < 1 || table.size(1) != table_stride)
        return false;
    for(int dimension = 0; dimension < 3; ++dimension)
        if(output.size(dimension) != query.size(dimension)) return false;
    if(query.dtype() != AITER_DTYPE_bf16 || output.dtype() != AITER_DTYPE_bf16 ||
       key.dtype() != AITER_DTYPE_fp8 || value.dtype() != AITER_DTYPE_fp8 ||
       table.dtype() != AITER_DTYPE_i32 || lengths.dtype() != AITER_DTYPE_i32 ||
       key_scale->dtype() != AITER_DTYPE_fp32 || value_scale->dtype() != AITER_DTYPE_fp32 ||
       key_scale->numel() != key.numel() / 128 || value_scale->numel() != key_scale->numel())
        return false;
    for(const auto* tensor : {&query, &key, &value, &table, &lengths, &output,
                             key_scale, value_scale})
        if(!tensor->is_gpu() || tensor->device_id != query.device_id || !tensor->is_contiguous())
            return false;
    for(const auto* tensor : {&query, &key, &value, &output})
        if(reinterpret_cast<uintptr_t>(tensor->data_ptr()) % 16 != 0) return false;
    if(qtp != nullptr && (qtp->dim() != 1 || qtp->size(0) != batch + 1 ||
                         qtp->dtype() != AITER_DTYPE_i32 || !qtp->is_contiguous() ||
                         !qtp->is_gpu() || qtp->device_id != query.device_id))
        return false;
    return query.numel() * query.element_size() < (size_t{1} << 31) &&
           table.numel() * table.element_size() < (size_t{1} << 31) &&
           key_scale->numel() * sizeof(float) < (size_t{1} << 31) &&
           key.stride(0) <= std::numeric_limits<uint32_t>::max() &&
           static_cast<int64_t>(table_stride) * 16 <= std::numeric_limits<int32_t>::max();
}

int select_splits(int batch, int kv_heads, int max_qlen, int pages, int num_cu,
                  const char* override_value)
{
    if(override_value != nullptr)
    {
        char* end = nullptr;
        const long value = std::strtol(override_value, &end, 10);
        AITER_CHECK(end != override_value && *end == '\0' && value >= 1 && value <= 256,
                    "AITER_PA_ASM_NUM_SPLITS must be an integer in [1,256]");
        return static_cast<int>(value);
    }
    const int64_t base_tgs = static_cast<int64_t>(batch) * kv_heads;
    const int64_t planning_tiles = (static_cast<int64_t>(pages) * 16 + 127) / 128;
    int64_t splits = 1;
    if(base_tgs < num_cu && planning_tiles >= 8)
        splits = std::min({num_cu / base_tgs, planning_tiles / 4, int64_t{256}});
    if(planning_tiles >= 256)
    {
        const int64_t fill = std::max(int64_t{max_qlen > 1 ? 16 : 8}, num_cu / base_tgs);
        splits = std::max(int64_t{1}, std::min({fill, num_cu * int64_t{32} / base_tgs,
                                               planning_tiles / 4, int64_t{256}}));
    }
    if(max_qlen == 1) splits = std::min(splits, int64_t{128});
    const int64_t compute_tiles = (static_cast<int64_t>(pages) * 16 + 255) / 256;
    if(kv_heads == 1 && max_qlen > 1 && compute_tiles >= 4 && compute_tiles <= 8 &&
       static_cast<int64_t>(batch) * max_qlen * compute_tiles <= num_cu)
        splits = compute_tiles;
    return static_cast<int>(splits);
}

struct LaunchPlan
{
    int tiles_per_split;
    bool query_split;
    bool parallel;
    int coalesce;
};

LaunchPlan plan_launch(int batch, int tokens, int kv_heads, int max_qlen,
                       int pages, int splits, int num_cu, bool allow_coalesce)
{
    const int total_tiles = (static_cast<int64_t>(pages) * 16 + 255) / 256;
    const int tiles_per_split = (total_tiles + splits - 1) / splits;
    const int active_splits = (total_tiles + tiles_per_split - 1) / tiles_per_split;
    const bool uniform_queries = static_cast<int64_t>(tokens) == static_cast<int64_t>(batch) * max_qlen;
    const bool query_split = uniform_queries && kv_heads == 1 && max_qlen > 1 &&
        static_cast<int64_t>(batch) * splits * max_qlen <= 2LL * num_cu;
    const bool parallel = splits >= 64 && active_splits >= 64 &&
        static_cast<int64_t>(tokens) * kv_heads * 16 <= num_cu;
    int coalesce = 1;
    if(allow_coalesce && !query_split && splits == 16 && max_qlen == 4 &&
       kv_heads == 1 && pages >= 4096 && batch >= (num_cu + 7) / 8 &&
       batch <= num_cu / 4 && num_cu % batch == 0)
    {
        if(num_cu / batch == 8) coalesce = 2;
        if(num_cu / batch == 4) coalesce = 4;
    }
    return {tiles_per_split, query_split, parallel, coalesce};
}

struct Workspace
{
    AiterTensor qtp;
    AiterTensor partial;
    AiterTensor lse;

    Workspace(int device, int batch, int tokens, int max_qlen, int heads,
              int splits, bool generated_qtp)
        : qtp(AiterTensor::empty({generated_qtp ? batch + 1 : 0}, AITER_DTYPE_i32, device)),
          partial(AiterTensor::empty({splits > 1 ? tokens : 0, splits, heads, 128},
                                     AITER_DTYPE_fp32, device)),
          lse(AiterTensor::empty({splits > 1 ? tokens : 0, splits, heads},
                                 AITER_DTYPE_fp32, device))
    {
        if(generated_qtp)
        {
            std::vector<int32_t> prefix(static_cast<size_t>(batch) + 1);
            for(int index = 0; index <= batch; ++index) prefix[index] = index * max_qlen;
            HIP_CALL(hipMemcpy(qtp.data_ptr(), prefix.data(), prefix.size() * sizeof(int32_t),
                               hipMemcpyHostToDevice));
        }
    }
};

struct Registry
{
    std::mutex mutex;
    std::map<std::tuple<int, std::string, std::string>, std::unique_ptr<AiterAsmKernel>> kernels;
    std::map<std::tuple<int, uintptr_t, int, int, int, int, int, bool>,
             std::unique_ptr<Workspace>> workspaces;
};

void run(const KernelArgs& base, const aiter_tensor_t& query, int batch, int max_qlen,
         const paConfig& config, hipStream_t stream)
{
    int num_cu = 0;
    HIP_CALL(hipDeviceGetAttribute(&num_cu, hipDeviceAttributeMultiprocessorCount, query.device_id));
    const char* override_value = std::getenv("AITER_PA_ASM_NUM_SPLITS");
    const int splits = select_splits(batch, base.kv_nheads, max_qlen, base.mblk,
                                     num_cu, override_value);
    const int tokens = static_cast<int>(query.size(0));
    const int heads = static_cast<int>(query.size(1));
    AITER_CHECK(static_cast<size_t>(heads) * splits * 128 * sizeof(float) < (size_t{1} << 32),
                "SP3 partial stride must fit uint32");
    const auto plan = plan_launch(batch, tokens, base.kv_nheads, max_qlen, base.mblk,
                                  splits, num_cu, override_value == nullptr);
    const bool generated_qtp = base.ptr_QTP == nullptr;
    std::string main_name = plan.query_split ? "pa_a16w8_q16_d128_p16_mtp_query" : config.knl_name;
    if(plan.coalesce == 2) main_name = "pa_a16w8_q16_d128_p16_mtp_coalesce2";
    if(plan.coalesce == 4) main_name = "pa_a16w8_q16_d128_p16_mtp_coalesce4";
    const std::string reduce_name = plan.parallel ? "pa_a16w8_q16_d128_p16_mtp_reduce_parallel8" :
                                              "pa_a16w8_q16_d128_p16_mtp_reduce";
    const char* asm_dir = std::getenv("AITER_ASM_DIR");
    const std::string object_key = std::string(asm_dir == nullptr ? "<embedded>" : asm_dir) +
                                   "/" + config.arch + "/" + config.co_name;
    static Registry* registry = new Registry;
    Workspace* workspace = nullptr;
    AiterAsmKernel* main_kernel = nullptr;
    AiterAsmKernel* reduce_kernel = nullptr;
    {
        std::lock_guard<std::mutex> lock(registry->mutex);
        hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
        HIP_CALL(hipStreamIsCapturing(stream, &capture));
        auto get_kernel = [&](const std::string& name) {
            const auto key = std::make_tuple(query.device_id, object_key, name);
            auto found = registry->kernels.find(key);
            if(found == registry->kernels.end())
            {
                AITER_CHECK(capture == hipStreamCaptureStatusNone,
                            "Warm up pa_fwd_asm on this device/stream/shape before Graph capture");
                found = registry->kernels.emplace(key,
                    std::make_unique<AiterAsmKernel>(name.c_str(), config.co_name.c_str())).first;
            }
            return found->second.get();
        };
        main_kernel = get_kernel(main_name);
        if(splits > 1) reduce_kernel = get_kernel(reduce_name);
        const auto key = std::make_tuple(query.device_id, reinterpret_cast<uintptr_t>(stream),
                                        batch, tokens, max_qlen, heads, splits, generated_qtp);
        auto found = registry->workspaces.find(key);
        if(found == registry->workspaces.end())
        {
            AITER_CHECK(capture == hipStreamCaptureStatusNone,
                        "Warm up pa_fwd_asm on this device/stream/shape before Graph capture");
            found = registry->workspaces.emplace(key, std::make_unique<Workspace>(
                query.device_id, batch, tokens, max_qlen, heads, splits, generated_qtp)).first;
        }
        workspace = found->second.get();
    }
    MainArgs arguments{};
    arguments.base = base;
    arguments.base.ptr_O = splits > 1 ? workspace->partial.data_ptr() : base.ptr_O;
    if(generated_qtp) arguments.base.ptr_QTP = workspace->qtp.data_ptr();
    arguments.num_splits = splits;
    arguments.tiles_per_split = plan.tiles_per_split;
    arguments.lse = workspace->lse.data_ptr();
    size_t argument_size = sizeof(arguments);
    main_kernel->launch_kernel({&arguments, &argument_size,
        static_cast<int>(base.kv_nheads) * (plan.query_split ? max_qlen : 1), batch, splits,
        256, 1, 1, stream});
    if(splits > 1)
    {
        ReduceArgs reduce{};
        reduce.output = base.ptr_O;
        reduce.partial = workspace->partial.data_ptr();
        reduce.lse = workspace->lse.data_ptr();
        reduce.num_splits = splits;
        reduce.heads = heads;
        size_t reduce_size = sizeof(reduce);
        reduce_kernel->launch_kernel({&reduce, &reduce_size, heads, tokens, 1,
                                       plan.parallel ? 512 : 64, 1, 1, stream});
    }
    if(std::getenv("AITER_PA_ASM_PRINT_SPLITS") != nullptr)
        std::fprintf(stderr, "[pa-asm-sp3] batch=%d qlen=%d splits=%d main=%s reduce_threads=%d\n",
                     batch, max_qlen, splits, main_name.c_str(),
                     splits > 1 ? (plan.parallel ? 512 : 64) : 0);
}

}

const float f_log2E = log2f(expf(1));

AITER_C_ITFS
void pa_fwd(aiter_tensor_t* Q,              //   [num_seqs, num_heads, head_size]
            aiter_tensor_t* K,              //   [num_blocks, num_kv_heads, head_size/x, block_size, x]
            aiter_tensor_t* V,              //   [num_blocks, num_kv_heads, block_size/X, head_size, X]
            aiter_tensor_t* block_tables,   //   [num_seqs, max_num_blocks_per_seq]
            aiter_tensor_t* context_lens,   //   [num_seqs]
            int block_tables_stride0,
            int max_qlen,
            aiter_tensor_t* K_QScale,       //   nullable
            aiter_tensor_t* V_QScale,       //   nullable
            aiter_tensor_t* out_,           //   output tensor (pre-allocated by caller)
            aiter_tensor_t* qo_indptr,      //   nullable
            int high_precision,
            const char* kernelName_,     //   nullable
            hipStream_t stream)
{
    const HipDeviceGuard device_guard(Q->device_id);
    int batch            = context_lens->size(0);
    if(max_qlen > 1)
    {
        batch = block_tables->size(0);
    }
    std::string arch_id = get_gpu_arch();
    int num_heads       = Q->size(1);
    int head_size       = Q->size(2);
    AITER_CHECK(head_size == 128,
        __func__,
        ": ASM PA only supports head_size=128, got ",
        head_size);
    int num_kv_heads    = K->size(1);
    int block_size      = K->size(3);
    const int gqa_ratio = num_heads / num_kv_heads;

    int dim            = head_size;
    int stride_Q       = Q->stride(0) * Q->element_size();
    int stride_KV_head = K->stride(1) * K->element_size();
    int stride_KV_blk  = K->stride(0) * K->element_size();
    float k_log2e      = f_log2E;
    float k_scalar     = sqrt(dim);
    k_scalar           = (float)((double)k_log2e / (double)k_scalar);

    KernelArgs args = {};
    size_t arg_size = sizeof(args);
    args.ptr_O      = out_->data_ptr();
    args.ptr_Q      = Q->data_ptr();
    args.ptr_K      = K->data_ptr();
    args.ptr_V      = V->data_ptr();
    args.ptr_BT     = block_tables->data_ptr();
    args.ptr_CL     = context_lens->data_ptr();
    if(K_QScale != nullptr)
    {
        args.ptr_KQ = K_QScale->data_ptr();
        args.ptr_VQ = V_QScale->data_ptr();
    }
    else
    {
        args.ptr_KQ = nullptr;
        args.ptr_VQ = nullptr;
    }
    args.sclg2e    = k_scalar;
    args.mblk      = block_tables_stride0;
    args.kv_nheads = num_kv_heads;
    args.Qs        = stride_Q;
    args.Bs        = stride_KV_blk;
    args.KVs       = stride_KV_head;
    args.mtp       = max_qlen - 1;
    args.GQA       = gqa_ratio;
    args.ptr_QTP   = (qo_indptr != nullptr) ? qo_indptr->data_ptr() : nullptr;

    std::string q_type;
    std::string kv_type;
    int gqa;
    int mtp;
    int msk;
    int hp;
    // 1. "q_type"
    auto q_dtype = Q->dtype();
    auto kv_dtype = K->dtype();
    if(q_dtype == AITER_DTYPE_fp16)
        q_type = "fp16";
    else if(q_dtype == AITER_DTYPE_bf16)
        q_type = "bf16";
    else
        AITER_CHECK(false, __func__, ": unsupport Q dtype:", AiterDtype_to_str(q_dtype));

    // 2. "kv_type"
    if(kv_dtype == AITER_DTYPE_fp16)
        kv_type = "fp16";
    else if(kv_dtype == AITER_DTYPE_bf16)
        kv_type = "bf16";
    else if(kv_dtype == AITER_DTYPE_i8 || kv_dtype == AITER_DTYPE_u8)
        kv_type = "int8";
    else if(kv_dtype == AITER_DTYPE_fp8)
        kv_type = "fp8";
    else
        AITER_CHECK(false, __func__, ": unsupport K dtype:", AiterDtype_to_str(kv_dtype));

    if(qo_indptr != nullptr && max_qlen > 1)
    {
        mtp = max_qlen + 10; // for kernels only support qlen=3, we encode it as 3+10=13
        msk = 1;
    }
    else
    {
        mtp = 0;
        msk = 0;
    }
    // 6. "high_precision" , 7. "ultra_precision"
    switch(high_precision)
    {
    case 1: hp = 1; break;
    case 2: hp = 2; break;
    default: hp = 0; break;
    };
    int qTile = 0;
    CFG* config_map = &cfg_pa_asm; // only one config csv in hsa/<arch>/pa, now
    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    std::string kernelName = (kernelName_ != nullptr) ? arch_id + std::string(kernelName_) : "";
    int ps = 0;
    const bool split_supported = arch_id == "gfx950" && pa_asm_split::supported(
        *Q, *K, *V, *block_tables, *context_lens, *out_, K_QScale, V_QScale,
        qo_indptr, block_tables_stride0, max_qlen, high_precision);
    if(kernelName.empty() && split_supported && (max_qlen == 1 || qo_indptr != nullptr))
        kernelName = get_heuristic_kernel(q_type, kv_type, gqa_ratio, 1, 1, hp,
                                          block_size, arch_id, 0, 0, 2, config_map);
    if (kernelName.empty())
        kernelName = get_heuristic_kernel(q_type, kv_type, gqa_ratio, mtp, msk, hp, block_size, arch_id, ps, qTile, 0, config_map);
    if(kernelName.empty())
    {
        AITER_CHECK(false, __func__, "not supported this kernel now! ");
    }

    AiterAsmKernel* impl_ptr = nullptr;

    auto it = config_map->find(kernelName);
    if(it != config_map->end())
    {
        const auto& cfg     = it->second;
        if(cfg.co_name == "pa/pa_a16w8_q16_d128_p16_mtp_split.co")
        {
            AITER_CHECK(split_supported,
                        "Split SP3 requires contiguous gfx950 BF16 Q/O, FP8 page16 KV, "
                        "GQA16, QL1..4, FP32 per-token scales and INT32 table/lengths/QTP");
            pa_asm_split::run(args, *Q, batch, max_qlen, cfg, stream);
            return;
        }
        const char* name    = cfg.knl_name.c_str();
        const char* co_name = cfg.co_name.c_str();

        impl_ptr =
            &impl_ptr_map.get_or_create(name, [&]() { return AiterAsmKernel(name, co_name); });
    }
    else
        AITER_CHECK(false, __func__, " not find kernel ", kernelName);

    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             num_kv_heads, // gdx
                             batch,        // gdy
                             1,            // gdz
                             256,          // bdx: 4 wv64
                             1,            // bdy
                             1,            // bdz
                             stream});
}

AITER_C_ITFS
void pa_ps_fwd(aiter_tensor_t* Q,            //   [num_seqs, num_heads, head_size]
               aiter_tensor_t* K,            //   [num_blocks, num_kv_heads, head_size/x, block_size, x]
               aiter_tensor_t* V,            //   [num_blocks, num_kv_heads, block_size/X, head_size, X]
               aiter_tensor_t* kv_indptr,    //   [batch_size+1], kvlen prefix sum
               aiter_tensor_t* kv_indices,   //   [sum_kvlen], packed kv ids
               aiter_tensor_t* context_lens, //   [batch_size]
               float softmax_scale,
               int max_qlen,
               aiter_tensor_t* K_QScale,     //   nullable
               aiter_tensor_t* V_QScale,     //   nullable
               aiter_tensor_t* out_,         //   output (pre-allocated by caller)
               aiter_tensor_t* qo_indptr,    //   nullable
               aiter_tensor_t* work_indptr,  //   nullable
               aiter_tensor_t* work_info,    //   nullable
               aiter_tensor_t* splitData,    //   nullable
               aiter_tensor_t* splitLse,     //   nullable
               int mask,
               int high_precision,
               const char* kernelName_,   //   nullable
               int quant_type,            //   QuantType enum value
               hipStream_t stream)
{
    int batch           = qo_indptr->size(0) - 1;
    int num_heads       = Q->size(1);
    int head_size       = Q->size(2);
    int num_kv_heads    = K->size(1);
    int block_size      = K->size(3);
    const int gqa_ratio = num_heads / num_kv_heads;

    int dim            = head_size;
    int stride_Q       = Q->stride(0) * Q->element_size();
    int stride_KV_head = K->stride(1) * K->element_size();
    int stride_KV_blk  = K->stride(0) * K->element_size();
    int stride_scale_blk = (K_QScale != nullptr)
                               ? (K_QScale->stride(1) * K_QScale->element_size())
                               : (block_size * sizeof(float));
    int stride_scale_page = (K_QScale != nullptr)
                                ? (K_QScale->stride(0) * K_QScale->element_size())
                                : (num_kv_heads * block_size * sizeof(float));
    float k_log2e      = f_log2E;
    float k_scalar     = sqrt(dim);
    k_scalar           = (float)((double)k_log2e / (double)k_scalar);

    PsKernelArgs args;
    size_t arg_size = sizeof(args);
    args.ptr_O      = out_->data_ptr();
    args.ptr_Q      = Q->data_ptr();
    args.ptr_K      = K->data_ptr();
    args.ptr_V      = V->data_ptr();

    args.ptr_KVIndices     = kv_indices->data_ptr();
    args.ptr_CL     = context_lens->data_ptr();
    if(K_QScale != nullptr)
    {
        args.ptr_KQ = K_QScale->data_ptr();
        args.ptr_VQ = V_QScale->data_ptr();
    }
    else
    {
        args.ptr_KQ = nullptr;
        args.ptr_VQ = nullptr;
    }
    args.sclg2e       = k_scalar;
    args.kv_nheads    = num_kv_heads;
    args.Qs           = stride_Q;
    args.Bs           = stride_KV_blk;
    args.KVs          = stride_KV_head;
    args.GQA          = gqa_ratio;
    args.ptr_QOPtr      = (qo_indptr != nullptr) ? qo_indptr->data_ptr() : nullptr;
    args.ptr_KVPtr     = kv_indptr->data_ptr();
    args.ptr_WorkPtr  = (work_indptr != nullptr) ? work_indptr->data_ptr() : nullptr;
    args.ptr_WorkInfo = (work_info != nullptr) ? work_info->data_ptr() : nullptr;
    args.ptr_SplitO   = (work_info != nullptr) ? splitData->data_ptr() : nullptr;
    args.ptr_SplitLSE = (work_info != nullptr) ? splitLse->data_ptr() : nullptr;
    args.stride_scale_blk = stride_scale_blk;
    args.stride_scale_page = stride_scale_page;
    args.mtp          = max_qlen - 1;

    const HipDeviceGuard device_guard(Q->device_id);

    std::string q_type;
    std::string kv_type;
    int gqa;
    int mtp;
    int msk;
    int hp;
    int ps = (work_indptr != nullptr) ? 1 : 0;
    // 1. "q_type"
    auto q_dtype = Q->dtype();
    auto kv_dtype = K->dtype();
    if(q_dtype == AITER_DTYPE_fp16)
        q_type = "fp16";
    else if(q_dtype == AITER_DTYPE_bf16)
        q_type = "bf16";
    else
        AITER_CHECK(false, __func__, ": unsupport Q dtype:", AiterDtype_to_str(q_dtype));

    // 2. "kv_type"
    if(kv_dtype == AITER_DTYPE_fp16)
        kv_type = "fp16";
    else if(kv_dtype == AITER_DTYPE_bf16)
        kv_type = "bf16";
    else if(kv_dtype == AITER_DTYPE_i8 || kv_dtype == AITER_DTYPE_u8)
        kv_type = "int8";
    else if(kv_dtype == AITER_DTYPE_fp8)
        kv_type = "fp8";
    else
        AITER_CHECK(false, __func__, ": unsupport K dtype:", AiterDtype_to_str(kv_dtype));

    // 3. "gqa_ratio"
    // 4. "mtp" , 5. "mask"
    // We make mtp=0, gqa=0 to dispatch kernel, since we only focus on qTile
    msk = mask;
    gqa = 0;
    mtp = 0;

    // 6. "high_precision" , 7. "ultra_precision"
    switch(high_precision)
    {
    case 1: hp = 1; break;
    case 2: hp = 2; break;
    default: hp = 0; break;
    };

    // gqa_ratio * max_qlen <= qTile
    int required_qTile = gqa_ratio * max_qlen;
    std::vector<int> available_qTiles = {16, 32, 40, 48, 64};
    int qTile = -1;

    for (int tile : available_qTiles) {
        if (required_qTile <= tile) {
            qTile = tile;
            break;
        }
    }

    AITER_CHECK(qTile != -1,
                __func__,
                ": required qTile (gqa_ratio * max_qlen = ", gqa_ratio, " * ", max_qlen,
                " = ", required_qTile,
                ") exceeds maximum available qTile. Please reduce gqa_ratio or max_qlen.");

    CFG* config_map = &cfg_pa_asm; // only one config csv in hsa/<arch>/pa, now
    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    std::string arch_id = get_gpu_arch();
    std::string kernelName = (kernelName_ != nullptr) ? std::string(kernelName_) :
        get_heuristic_kernel(q_type, kv_type, gqa, mtp, msk, hp, block_size, arch_id, ps, qTile, quant_type, config_map);
    if(kernelName.empty())
    {
        AITER_CHECK(false, __func__, "not supported this kernel now! ");
    }

    AiterAsmKernel* impl_ptr = nullptr;
    int gdx, gdy;

    auto it = config_map->find(kernelName);
    if(it != config_map->end())
    {
        const auto& cfg     = it->second;
        AITER_CHECK(cfg.co_name != "pa/pa_a16w8_q16_d128_p16_mtp_split.co",
                    "The page16 split SP3 bundle requires pa_fwd_asm, not pa_ps_fwd_asm");
        const char* name    = cfg.knl_name.c_str();
        const char* co_name = cfg.co_name.c_str();

        impl_ptr =
            &impl_ptr_map.get_or_create(name, [&]() { return AiterAsmKernel(name, co_name); });
        if(cfg.ps)
        {
            gdx = get_num_cu_func();
            gdy = 1;
        }
        else
        {
            gdx = num_kv_heads;
            gdy = batch;
        }
    }
    else
        AITER_CHECK(false, __func__, " not find kernel ", kernelName);

    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             gdx, // gdx
                             gdy, // gdy
                             1,   // gdz
                             256, // bdx: 4 wv64
                             1,   // bdy
                             1,   // bdz
                             stream});
}
