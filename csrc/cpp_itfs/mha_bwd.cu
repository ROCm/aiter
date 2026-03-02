#include "mha_bwd.h"
#include "aiter_hip_common.h"
#include "asm_fmha_v3_bwd_configs.hpp"
#include <memory>
#include <string>
#include <cstdlib>

namespace {

static bool env_is_set(const char* name)
{
    const char* v = std::getenv(name);
    return v && std::string(v) == "1";
}

__device__ __forceinline__ float bf16_to_f32(uint16_t v)
{
    union { uint32_t u; float f; } t;
    t.u = uint32_t(v) << 16;
    return t.f;
}

__device__ __forceinline__ uint16_t f32_to_bf16_rtz(float v)
{
    union { float f; uint32_t u; } t;
    t.f = v;
    return uint16_t(t.u >> 16);
}

__device__ __forceinline__ float fp16_to_f32(uint16_t v)
{
    _Float16 h;
    __builtin_memcpy(&h, &v, 2);
    return static_cast<float>(h);
}

__device__ __forceinline__ uint16_t f32_to_fp16(float v)
{
    _Float16 h = static_cast<_Float16>(v);
    uint16_t r;
    __builtin_memcpy(&r, &h, 2);
    return r;
}

// D[b][h][s] = sum_d( O[b][h][s][d] * dO[b][h][s][d] )
__global__ void fallback_odo_kernel(
    const uint16_t* __restrict__ ptr_o,
    const uint16_t* __restrict__ ptr_do,
    float* __restrict__ ptr_d,
    int seqlen_q,
    int head_dim,
    int stride_o,
    int nhead_stride_o,
    int batch_stride_o,
    int stride_do,
    int nhead_stride_do,
    int batch_stride_do,
    int nhead_stride_d,
    int batch_stride_d,
    int is_bf16,
    const int32_t* __restrict__ cu_seqlens_q)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    if(s >= seqlen_q)
        return;

    int64_t off_o, off_do, off_d;
    if(cu_seqlens_q) {
        int64_t sq = cu_seqlens_q[b];
        off_o  = sq * stride_o;
        off_do = sq * stride_do;
        off_d  = sq;
    } else {
        off_o  = (int64_t)b * batch_stride_o;
        off_do = (int64_t)b * batch_stride_do;
        off_d  = (int64_t)b * batch_stride_d;
    }

    const uint16_t* o  = ptr_o  + off_o  + (int64_t)h * nhead_stride_o  + (int64_t)s * stride_o;
    const uint16_t* d_o = ptr_do + off_do + (int64_t)h * nhead_stride_do + (int64_t)s * stride_do;

    float sum = 0.0f;
    for(int d = 0; d < head_dim; d++)
    {
        float ov  = is_bf16 ? bf16_to_f32(o[d])  : fp16_to_f32(o[d]);
        float dov = is_bf16 ? bf16_to_f32(d_o[d]) : fp16_to_f32(d_o[d]);
        sum += ov * dov;
    }
    ptr_d[off_d + (int64_t)h * nhead_stride_d + s] = sum;
}

// dQ[b][h][s][d] = convert( dQ_acc[b][h][s][d] )
__global__ void fallback_dq_convert_kernel(
    const float* __restrict__ dq_acc,
    uint16_t* __restrict__ dq,
    int seqlen_q,
    int head_dim,
    int stride_dq_acc,
    int64_t nhead_stride_dq_acc,
    int64_t batch_stride_dq_acc,
    int stride_dq,
    int nhead_stride_dq,
    int batch_stride_dq,
    int is_bf16,
    const int32_t* __restrict__ cu_seqlens_q)
{
    int d = threadIdx.x;
    int s = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    if(d >= head_dim || s >= seqlen_q)
        return;

    int64_t off_acc, off_dq;
    if(cu_seqlens_q) {
        int64_t sq = cu_seqlens_q[b];
        off_acc = sq * stride_dq_acc;
        off_dq  = sq * stride_dq;
    } else {
        off_acc = (int64_t)b * batch_stride_dq_acc;
        off_dq  = (int64_t)b * batch_stride_dq;
    }

    float val = dq_acc[off_acc + (int64_t)h * nhead_stride_dq_acc + (int64_t)s * stride_dq_acc + d];
    uint16_t out = is_bf16 ? f32_to_bf16_rtz(val) : f32_to_fp16(val);
    dq[off_dq + (int64_t)h * nhead_stride_dq + (int64_t)s * stride_dq + d] = out;
}

int get_ck_hdim(int hdim)
{
    if(hdim <= 32) return 32;
    if(hdim <= 64) return 64;
    if(hdim <= 128) return 128;
    return 256;
}

fmha_bwd_args to_ck_args(const aiter::mha_bwd_args& a)
{
    return fmha_bwd_args{
        a.q_ptr, a.k_ptr, a.v_ptr, a.bias_ptr, a.o_ptr, a.lse_ptr, a.do_ptr,
        a.d_ptr, a.rand_val_ptr, a.dq_ptr, a.dk_ptr, a.dv_ptr, a.dbias_ptr, a.dq_acc_ptr,
        a.seqstart_q_ptr, a.seqstart_k_ptr, a.seqlen_q_ptr, a.seqlen_k_ptr,
        a.cu_seqlen_q_ptr, a.cu_seqlen_k_ptr,
        a.seqlen_q, a.seqlen_k, a.batch, a.max_seqlen_q, a.max_seqlen_k,
        a.hdim_q, a.hdim_v, a.nhead_q, a.nhead_k, a.scale,
        a.stride_q, a.stride_k, a.stride_v, a.stride_bias,
        a.stride_o, a.stride_randval, a.stride_do, a.stride_dq_acc,
        a.stride_dq, a.stride_dk, a.stride_dv, a.stride_dbias,
        a.nhead_stride_q, a.nhead_stride_k, a.nhead_stride_v, a.nhead_stride_bias,
        a.nhead_stride_o, a.nhead_stride_randval, a.nhead_stride_do, a.nhead_stride_lsed,
        a.nhead_stride_dq_acc, a.nhead_stride_dq, a.nhead_stride_dk, a.nhead_stride_dv,
        a.nhead_stride_dbias,
        a.batch_stride_q, a.batch_stride_k, a.batch_stride_v, a.batch_stride_bias,
        a.batch_stride_o, a.batch_stride_randval, a.batch_stride_do, a.batch_stride_lsed,
        a.batch_stride_dq_acc, a.batch_stride_dq, a.batch_stride_dk, a.batch_stride_dv,
        a.batch_stride_dbias,
        a.split_stride_dq_acc, a.window_size_left, a.window_size_right,
        a.ck_mask_type, a.p_drop, a.p_undrop, a.drop_seed_offset,
    };
}

#define DISPATCH_CK_ODO(HDIM, DTYPE, GROUP) \
    fmha_bwd_dot_do_o_oneshot_<fmha_bwd_dot_do_o_traits_<HDIM, DTYPE, GROUP, true, true>>(s, ck_a)

void launch_ck_odo(const aiter::mha_bwd_args& a, const ck_tile::stream_config& s)
{
    fmha_bwd_args ck_a = to_ck_args(a);
    int hdim = get_ck_hdim(a.hdim_v);
    if(a.data_type == "bf16") {
        if(a.is_group_mode) {
            switch(hdim) {
                case 32:  DISPATCH_CK_ODO(32,  FmhaBwdBf16, true); break;
                case 64:  DISPATCH_CK_ODO(64,  FmhaBwdBf16, true); break;
                case 128: DISPATCH_CK_ODO(128, FmhaBwdBf16, true); break;
                default:  DISPATCH_CK_ODO(256, FmhaBwdBf16, true); break;
            }
        } else {
            switch(hdim) {
                case 32:  DISPATCH_CK_ODO(32,  FmhaBwdBf16, false); break;
                case 64:  DISPATCH_CK_ODO(64,  FmhaBwdBf16, false); break;
                case 128: DISPATCH_CK_ODO(128, FmhaBwdBf16, false); break;
                default:  DISPATCH_CK_ODO(256, FmhaBwdBf16, false); break;
            }
        }
    } else {
        if(a.is_group_mode) {
            switch(hdim) {
                case 32:  DISPATCH_CK_ODO(32,  FmhaBwdFp16, true); break;
                case 64:  DISPATCH_CK_ODO(64,  FmhaBwdFp16, true); break;
                case 128: DISPATCH_CK_ODO(128, FmhaBwdFp16, true); break;
                default:  DISPATCH_CK_ODO(256, FmhaBwdFp16, true); break;
            }
        } else {
            switch(hdim) {
                case 32:  DISPATCH_CK_ODO(32,  FmhaBwdFp16, false); break;
                case 64:  DISPATCH_CK_ODO(64,  FmhaBwdFp16, false); break;
                case 128: DISPATCH_CK_ODO(128, FmhaBwdFp16, false); break;
                default:  DISPATCH_CK_ODO(256, FmhaBwdFp16, false); break;
            }
        }
    }
}

#undef DISPATCH_CK_ODO

#define DISPATCH_CK_DQ_CONVERT(HDIM, DTYPE, GROUP) \
    fmha_bwd_convert_dq_oneshot_<fmha_bwd_convert_dq_traits_<HDIM, DTYPE, GROUP, true, true, false, 0>>(s, ck_a)

void launch_ck_dq_convert(const aiter::mha_bwd_args& a, const ck_tile::stream_config& s)
{
    fmha_bwd_args ck_a = to_ck_args(a);
    int hdim = get_ck_hdim(a.hdim_q);
    if(a.data_type == "bf16") {
        if(a.is_group_mode) {
            switch(hdim) {
                case 32:  DISPATCH_CK_DQ_CONVERT(32,  FmhaBwdBf16, true); break;
                case 64:  DISPATCH_CK_DQ_CONVERT(64,  FmhaBwdBf16, true); break;
                case 128: DISPATCH_CK_DQ_CONVERT(128, FmhaBwdBf16, true); break;
                default:  DISPATCH_CK_DQ_CONVERT(256, FmhaBwdBf16, true); break;
            }
        } else {
            switch(hdim) {
                case 32:  DISPATCH_CK_DQ_CONVERT(32,  FmhaBwdBf16, false); break;
                case 64:  DISPATCH_CK_DQ_CONVERT(64,  FmhaBwdBf16, false); break;
                case 128: DISPATCH_CK_DQ_CONVERT(128, FmhaBwdBf16, false); break;
                default:  DISPATCH_CK_DQ_CONVERT(256, FmhaBwdBf16, false); break;
            }
        }
    } else {
        if(a.is_group_mode) {
            switch(hdim) {
                case 32:  DISPATCH_CK_DQ_CONVERT(32,  FmhaBwdFp16, true); break;
                case 64:  DISPATCH_CK_DQ_CONVERT(64,  FmhaBwdFp16, true); break;
                case 128: DISPATCH_CK_DQ_CONVERT(128, FmhaBwdFp16, true); break;
                default:  DISPATCH_CK_DQ_CONVERT(256, FmhaBwdFp16, true); break;
            }
        } else {
            switch(hdim) {
                case 32:  DISPATCH_CK_DQ_CONVERT(32,  FmhaBwdFp16, false); break;
                case 64:  DISPATCH_CK_DQ_CONVERT(64,  FmhaBwdFp16, false); break;
                case 128: DISPATCH_CK_DQ_CONVERT(128, FmhaBwdFp16, false); break;
                default:  DISPATCH_CK_DQ_CONVERT(256, FmhaBwdFp16, false); break;
            }
        }
    }
}

#undef DISPATCH_CK_DQ_CONVERT

} // anonymous namespace

namespace aiter {
std::tuple<int, int> get_padded_hdim(int hdim_q, int hdim_v, std::string arch_id)
{
    if(hdim_q == 192 && hdim_v == 128 && arch_id == "gfx950")
        return std::make_tuple(hdim_q, hdim_v);
    assert(hdim_q == hdim_v);
    if(hdim_q <= 64)
    {
        return std::make_tuple(64, 64);
    }
    else if(hdim_q <= 128)
    {
        return std::make_tuple(128, 128);
    }
    else if(hdim_q <= 192)
    {
        return std::make_tuple(192, 192);
    }

    assert(false);
    return std::make_tuple(hdim_q, hdim_v);
}

std::tuple<std::string, std::string, std::string> get_heuristic_kernel(std::string data_type,
                                                                       std::string arch_id,
                                                                       int seqlen_q,
                                                                       int seqlen_k,
                                                                       int hdim_q,
                                                                       int hdim_v,
                                                                       int mask_type,
                                                                       bool atomic32,
                                                                       int bf16_cvt,
                                                                       bool mode,
                                                                       CFG* pre_cfgs,
                                                                       CFG* cfgs,
                                                                       CFG* post_cfgs)
{
    auto [padded_hdim_q, padded_hdim_v] = get_padded_hdim(hdim_q, hdim_v, arch_id);
    int pddv                            = (padded_hdim_q != hdim_q) || (padded_hdim_v != hdim_v);
    int pssk;
    int ts_kv = 0;

    std::string preProcessingKernelName  = "";
    std::string dQdKdVKernelName         = "";
    std::string postProcessingKernelName = "";

    for(const auto& el : *pre_cfgs)
    {
        if(el.first.find(arch_id) != 0)
            continue;
        const auto& cfg = el.second;

        if((cfg.dtype == data_type) && (cfg.hdim_v == padded_hdim_v) && (cfg.mode == mode))
        {
            preProcessingKernelName = el.first;
            break;
        }
    }

    for(const auto& el : *cfgs)
    {
        if(el.first.find(arch_id) != 0)
        {
            continue;
        }
        const auto& cfg = el.second;

        if((cfg.dtype == data_type) && (cfg.hdim_q == padded_hdim_q) &&
           (cfg.hdim_v == padded_hdim_v) && (cfg.mask == mask_type) && (cfg.atomic32 == atomic32) &&
           ((arch_id == "gfx950") || ((data_type == "fp16") || (cfg.bf16_cvt == bf16_cvt))) &&
           (cfg.mode == mode))
        {
            int tmp_ts_kv = 0;
            if(ts_kv == 0)
            {
                ts_kv     = cfg.ts;
                tmp_ts_kv = ts_kv;
                if(cfg.atomic32 == 0 &&
                   ((arch_id == "gfx942") || (el.first.find("recompile") != std::string::npos)))
                {

                    tmp_ts_kv = 64;
                }
                pssk = (seqlen_q != seqlen_k) || (seqlen_k % tmp_ts_kv != 0);
            }
            if((cfg.pssk == pssk) && (cfg.pddv == pddv))
            {
                dQdKdVKernelName = el.first;
                break;
            }
            else if((cfg.pssk >= pssk) && (cfg.pddv >= pddv))
            {
                dQdKdVKernelName = el.first;
            }
        }
    }

    if(!post_cfgs)
    {
        return std::make_tuple(preProcessingKernelName, dQdKdVKernelName, postProcessingKernelName);
    }

    for(const auto& el : *post_cfgs)
    {
        if(el.first.find(arch_id) != 0)
            continue;
        const auto& cfg = el.second;

        if((cfg.hdim_q == padded_hdim_q) && (cfg.mode == mode) &&
           ((arch_id == "gfx950") || ((data_type == "fp16") || (cfg.bf16_cvt == bf16_cvt))))
        {
            if((cfg.dtype == data_type) || (atomic32 == 0))
            {
                postProcessingKernelName = el.first;
                break;
            }
        }
    }
    return std::make_tuple(preProcessingKernelName, dQdKdVKernelName, postProcessingKernelName);
}

float mha_bwd(mha_bwd_args a, const ck_tile::stream_config& s)
{
    float asm_ret = fmha_v3_bwd(a, s);
#if ONLY_FAV3
    return asm_ret;
#else
    fmha_bwd_traits traits{a.hdim_q,
                           a.hdim_v,
                           a.data_type,
                           a.is_group_mode,
                           static_cast<mask_enum>(a.ck_mask_type),
                           static_cast<bias_enum>(a.bias_type),
                           a.has_dbias,
                           a.has_dropout,
                           a.is_store_randval,
                           a.is_deterministic};

    fmha_bwd_args ck_args{
        /* q_ptr              */ a.q_ptr,
        /* k_ptr              */ a.k_ptr,
        /* v_ptr              */ a.v_ptr,
        /* bias_ptr           */ a.bias_ptr,
        /* o_ptr              */ a.o_ptr,
        /* lse_ptr            */ a.lse_ptr,
        /* do_ptr             */ a.do_ptr,
        /* d_ptr              */ a.d_ptr,
        /* rand_val_ptr       */ a.rand_val_ptr,
        /* dq_ptr             */ a.dq_ptr,
        /* dk_ptr             */ a.dk_ptr,
        /* dv_ptr             */ a.dv_ptr,
        /* dbias_ptr          */ a.dbias_ptr,
        /* dq_acc_ptr         */ a.dq_acc_ptr,

        /* seqstart_q_ptr     */ a.seqstart_q_ptr,
        /* seqstart_k_ptr     */ a.seqstart_k_ptr,
        /* seqlen_q_ptr       */ a.seqlen_q_ptr,
        /* seqlen_k_ptr       */ a.seqlen_k_ptr,
        /* cu_seqlen_q_ptr    */ a.cu_seqlen_q_ptr,
        /* cu_seqlen_k_ptr    */ a.cu_seqlen_k_ptr,

        /* seqlen_q           */ a.seqlen_q,
        /* seqlen_k           */ a.seqlen_k,
        /* batch              */ a.batch,
        /* max_seqlen_q       */ a.max_seqlen_q,
        /* max_seqlen_k       */ a.max_seqlen_k,
        /* hdim_q             */ a.hdim_q,
        /* hdim_v             */ a.hdim_v,
        /* nhead_q            */ a.nhead_q,
        /* nhead_k            */ a.nhead_k,
        /* scale              */ a.scale,

        /* stride_q           */ a.stride_q,
        /* stride_k           */ a.stride_k,
        /* stride_v           */ a.stride_v,
        /* stride_bias        */ a.stride_bias,
        /* stride_o           */ a.stride_o,
        /* stride_randval     */ a.stride_randval,
        /* stride_do          */ a.stride_do,
        /* stride_dq_acc      */ a.stride_dq_acc,
        /* stride_dq          */ a.stride_dq,
        /* stride_dk          */ a.stride_dk,
        /* stride_dv          */ a.stride_dv,
        /* stride_dbias       */ a.stride_dbias,

        /* nhead_stride_q     */ a.nhead_stride_q,
        /* nhead_stride_k     */ a.nhead_stride_k,
        /* nhead_stride_v     */ a.nhead_stride_v,
        /* nhead_stride_bias  */ a.nhead_stride_bias,
        /* nhead_stride_o     */ a.nhead_stride_o,
        /* nhead_stride_randval*/ a.nhead_stride_randval,
        /* nhead_stride_do    */ a.nhead_stride_do,
        /* nhead_stride_lsed  */ a.nhead_stride_lsed,
        /* nhead_stride_dq_acc*/ a.nhead_stride_dq_acc,
        /* nhead_stride_dq    */ a.nhead_stride_dq,
        /* nhead_stride_dk    */ a.nhead_stride_dk,
        /* nhead_stride_dv    */ a.nhead_stride_dv,
        /* nhead_stride_dbias */ a.nhead_stride_dbias,

        /* batch_stride_q     */ a.batch_stride_q,
        /* batch_stride_k     */ a.batch_stride_k,
        /* batch_stride_v     */ a.batch_stride_v,
        /* batch_stride_bias  */ a.batch_stride_bias,
        /* batch_stride_o     */ a.batch_stride_o,
        /* batch_stride_randval*/ a.batch_stride_randval,
        /* batch_stride_do    */ a.batch_stride_do,
        /* batch_stride_lsed  */ a.batch_stride_lsed,
        /* batch_stride_dq_acc*/ a.batch_stride_dq_acc,
        /* batch_stride_dq    */ a.batch_stride_dq,
        /* batch_stride_dk    */ a.batch_stride_dk,
        /* batch_stride_dv    */ a.batch_stride_dv,
        /* batch_stride_dbias */ a.batch_stride_dbias,

        /* split_stride_dq_acc*/ a.split_stride_dq_acc,
        /* window_size_left   */ a.window_size_left,
        /* window_size_right  */ a.window_size_right,
        /* mask_type          */ a.ck_mask_type,
        /* p_drop             */ a.p_drop,
        /* p_undrop           */ a.p_undrop,
        /* drop_seed_offset   */ a.drop_seed_offset,
    };

    if(asm_ret == -1)
    {
        return fmha_bwd(traits, ck_args, s);
    }
    return asm_ret;
#endif
}

float fmha_v3_bwd(mha_bwd_args a, const ck_tile::stream_config& s)
{
    if(a.nhead_stride_dq_acc < a.stride_dq_acc)
    {
        return -1;  // dq_acc only support BHSD layout
    }

    std::string arch_id = get_gpu_arch();
    if((!a.use_asm_v3) || (a.hdim_q % 8 != 0) || (a.hdim_v % 8 != 0) || (a.has_dbias) ||
       (a.bias_type != 0) || (a.has_dropout) || (a.is_deterministic) ||
       ((arch_id != "gfx942") && (arch_id != "gfx950")))
    {
        return -1;
    }

    static bool use_ck_odo        = env_is_set("AITER_V3_BWD_CK_ODO");
    static bool use_ck_dq_convert = env_is_set("AITER_V3_BWD_CK_DQ_CONVERT");

    auto pre_cfgs    = &cfg_fmha_bwd_odo;
    auto dqdkdv_cfgs = &cfg_fmha_bwd_dqdkdv;
    auto post_cfgs   = [&]() {
        if(arch_id == "gfx950")
        {
            if(a.v3_atomic_fp32)
            {
                return &cfg_fmha_bwd_dq_convert;
            }
            else
            {
                return &cfg_fmha_bwd_dq_shuffle;
            }
        }
        else
        {
            if(a.v3_atomic_fp32)
            {
                return &cfg_fmha_bwd_dq_convert;
            }
            else
            {
                return static_cast<CFG*>(nullptr);
            }
        }
    }();

    bool need_post_processing =
        ((arch_id == "gfx950") && (a.hdim_q != 64)) || (a.v3_atomic_fp32 == 1);

    auto [pre_kernel, dqdkdv_kernel, post_kernel] = get_heuristic_kernel(a.data_type,
                                                                         arch_id,
                                                                         a.seqlen_q,
                                                                         a.seqlen_k,
                                                                         a.hdim_q,
                                                                         a.hdim_v,
                                                                         a.mask_type,
                                                                         a.v3_atomic_fp32,
                                                                         a.v3_bf16_cvt,
                                                                         a.is_group_mode,
                                                                         pre_cfgs,
                                                                         dqdkdv_cfgs,
                                                                         post_cfgs);

    if((!use_ck_odo && pre_kernel == "") ||
       (dqdkdv_kernel == "") ||
       (need_post_processing && !use_ck_dq_convert && post_kernel == ""))
    {
        return -1;
    }

    int ts_odo = 0;
    int ts_kv;
    int ts_dq = 0;
    int arg_size;

    AiterAsmKernel* impl_ptr_pre    = nullptr;
    AiterAsmKernel* impl_ptr_dqdkdv = nullptr;
    AiterAsmKernel* impl_ptr_post   = nullptr;
    static std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> impl_ptr_map;

    if(!use_ck_odo)
    {
        auto it_pre = pre_cfgs->find(pre_kernel);
        if(it_pre != pre_cfgs->end())
        {
            const auto& cfg     = it_pre->second;
            const char* name    = cfg.knl_name.c_str();
            const char* co_name = cfg.co_name.c_str();
            ts_odo              = cfg.ts;

            auto result = impl_ptr_map.emplace(name, nullptr);
            if(result.second)
            {
                result.first->second = std::make_unique<AiterAsmKernel>(name, co_name);
            }

            impl_ptr_pre = result.first->second.get();
        }
        else
        {
            return -1;
        }
    }
    else
    {
        std::cout << "[aiter] BWD ODO: using CK kernel" << std::endl;
    }

    auto it_dqdkdv = dqdkdv_cfgs->find(dqdkdv_kernel);
    if(it_dqdkdv != dqdkdv_cfgs->end())
    {
        const auto& cfg     = it_dqdkdv->second;
        const char* name    = cfg.knl_name.c_str();
        const char* co_name = cfg.co_name.c_str();
        ts_kv               = cfg.ts;

        // Debug: print which kernel is being used
        std::cout << "[aiter] BWD kernel selected: " << co_name << " (knl: " << name << ")" << std::endl;
        std::cout.flush();
        
        auto result = impl_ptr_map.emplace(name, nullptr);
        if(result.second)
        {
            result.first->second = std::make_unique<AiterAsmKernel>(name, co_name);
        }

        impl_ptr_dqdkdv = result.first->second.get();
    }
    else
    {
        return -1;
    }

    if(need_post_processing && !use_ck_dq_convert)
    {
        auto it_post = post_cfgs->find(post_kernel);
        if(it_post != post_cfgs->end())
        {
            const auto& cfg     = it_post->second;
            const char* name    = cfg.knl_name.c_str();
            const char* co_name = cfg.co_name.c_str();
            ts_dq               = cfg.ts;

            auto result = impl_ptr_map.emplace(name, nullptr);
            if(result.second)
            {
                result.first->second = std::make_unique<AiterAsmKernel>(name, co_name);
            }

            impl_ptr_post = result.first->second.get();
        }
        else
        {
            return -1;
        }
    }
    else if(need_post_processing)
    {
        std::cout << "[aiter] BWD dQ_convert: using CK kernel" << std::endl;
    }

    if(a.v3_api_check)
        return 1;

    fmha_bwd_odo_args odo_args;
    if(!use_ck_odo)
    {
        arg_size                 = sizeof(odo_args);
        odo_args.ptr_o           = a.o_ptr;
        odo_args.ptr_do          = a.do_ptr;
        odo_args.ptr_d           = a.d_ptr;
        odo_args.Hs_odo          = a.nhead_stride_o * 2;
        odo_args.BAs_odo         = a.batch_stride_o * 2;
        odo_args.Seqs_odo        = a.stride_o * 2;
        odo_args.Hs_d            = a.nhead_stride_lsed * 4;
        odo_args.BAs_d           = a.batch_stride_lsed * 4;
        odo_args.Seqs_d          = 1 * 4;
        odo_args.seqlen_q        = a.seqlen_q;
        odo_args.head_dim        = a.hdim_q;
        odo_args.ptr_qseq_padded = a.seqstart_q_ptr;
        odo_args.ptr_qseq =
            (a.cu_seqlen_q_ptr && a.seqstart_q_ptr) ? a.cu_seqlen_q_ptr : a.seqstart_q_ptr;
    }

    auto pre_kernel_launch = [&]() {
        if(use_ck_odo)
        {
            launch_ck_odo(a, ck_tile::stream_config{s.stream_id_});
        }
        else
        {
            int bdx = 256;
            int gdx = (a.max_seqlen_q + ts_odo - 1) / ts_odo;
            int gdy = a.nhead_q;
            int gdz = a.batch;
            impl_ptr_pre->launch_kernel({&odo_args, &arg_size, gdx, gdy, gdz, bdx, 1, 1, s.stream_id_});
        }
    };

    fmha_bwd_dqdkdv_args dqdkdv_args;
    dqdkdv_args.ptr_dq     = need_post_processing ? a.dq_acc_ptr : a.dq_ptr;
    dqdkdv_args.ptr_dk     = a.dk_ptr;
    dqdkdv_args.ptr_dv     = a.dv_ptr;
    dqdkdv_args.ptr_q      = a.q_ptr;
    dqdkdv_args.ptr_k      = a.k_ptr;
    dqdkdv_args.ptr_v      = a.v_ptr;
    dqdkdv_args.ptr_do     = a.do_ptr;
    dqdkdv_args.ptr_lse    = a.lse_ptr;
    dqdkdv_args.ptr_d      = a.d_ptr;
    dqdkdv_args.scalar     = a.scale;
    dqdkdv_args.log2e      = ck_tile::log2e_v<float>;
    dqdkdv_args.ratio      = a.nhead_q / a.nhead_k;
    dqdkdv_args.seqlen_q   = a.seqlen_q;
    dqdkdv_args.seqlen_k   = a.seqlen_k;
    dqdkdv_args.head_dim_q = a.hdim_q;
    dqdkdv_args.head_dim_v = a.hdim_v;
    dqdkdv_args.nhead_q    = a.nhead_q;
    dqdkdv_args.Ts         = ts_kv * a.stride_k * 2;
    dqdkdv_args.Hs_q       = a.nhead_stride_q * 2;
    dqdkdv_args.BAs_q      = a.batch_stride_q * 2;
    dqdkdv_args.Seqs_q     = a.stride_q * 2;
    dqdkdv_args.Hs_k       = a.nhead_stride_k * 2;
    dqdkdv_args.BAs_k      = a.batch_stride_k * 2;
    dqdkdv_args.Seqs_k     = a.stride_k * 2;
    dqdkdv_args.Hs_v       = a.nhead_stride_v * 2;
    dqdkdv_args.BAs_v      = a.batch_stride_v * 2;
    dqdkdv_args.Seqs_v     = a.stride_v * 2;
    dqdkdv_args.Hs_do      = a.nhead_stride_do * 2;
    dqdkdv_args.BAs_do     = a.batch_stride_do * 2;
    dqdkdv_args.Seqs_do    = a.stride_do * 2;
    dqdkdv_args.Hs_dk      = a.nhead_stride_dk * 2;
    dqdkdv_args.BAs_dk     = a.batch_stride_dk * 2;
    dqdkdv_args.Seqs_dk    = a.stride_dk * 2;
    dqdkdv_args.Hs_dv      = a.nhead_stride_dv * 2;
    dqdkdv_args.BAs_dv     = a.batch_stride_dv * 2;
    dqdkdv_args.Seqs_dv    = a.stride_dv * 2;
    dqdkdv_args.Hs_lsed    = a.nhead_stride_lsed * 4;

    if(a.cu_seqlen_k_ptr && a.seqstart_k_ptr)
    {
        dqdkdv_args.ptr_kseq_padded = a.seqstart_k_ptr;
        dqdkdv_args.ptr_kseq        = a.cu_seqlen_k_ptr;
    }
    else
    {
        dqdkdv_args.ptr_kseq        = a.seqstart_k_ptr;
        dqdkdv_args.ptr_kseq_padded = a.seqstart_k_ptr;
    }

    if(a.cu_seqlen_q_ptr && a.seqstart_q_ptr)
    {
        dqdkdv_args.ptr_qseq_padded = a.seqstart_q_ptr;
        dqdkdv_args.ptr_qseq        = a.cu_seqlen_q_ptr;
    }
    else
    {
        dqdkdv_args.ptr_qseq        = a.seqstart_q_ptr;
        dqdkdv_args.ptr_qseq_padded = a.seqstart_q_ptr;
    }
    dqdkdv_args.max_seqlen_dq = a.v3_atomic_fp32 ? a.max_seqlen_q : (a.max_seqlen_q + 15) / 16 * 16;

    if(a.mask_type == 3)
    {
        // Note: sink_size=0 is passed as the 3rd parameter (attention sink not supported in bwd
        // yet)
        auto sink_size    = 0;
        auto generic_mask = ck_tile::make_generic_attention_mask_coordinates_from_lr_window(
            a.window_size_left,
            a.window_size_right,
            sink_size,
            a.seqlen_q,
            a.seqlen_k,
            (a.ck_mask_type == static_cast<ck_tile::index_t>(mask_enum::mask_top_left) ||
             a.ck_mask_type == static_cast<ck_tile::index_t>(mask_enum::window_generic)));
        dqdkdv_args.mask_y = generic_mask.at(ck_tile::number<0>{});
        dqdkdv_args.mask_x = generic_mask.at(ck_tile::number<1>{});
    }

    arg_size                  = sizeof(dqdkdv_args);
    auto dqdkdv_kernel_launch = [&]() {
        int bdx = 256;
        int gdx = (a.max_seqlen_k + ts_kv - 1) / ts_kv;
        int gdy = a.nhead_q;
        int gdz = a.batch;

        if((a.mask_type == 1) || (a.mask_type == 2))
        { // causal
            gdx = (gdx + 1) / 2;
        }

        impl_ptr_dqdkdv->launch_kernel(
            {&dqdkdv_args, &arg_size, gdx, gdy, gdz, bdx, 1, 1, s.stream_id_});
    };

    if(!need_post_processing)
    {
        return ck_tile::launch_kernel(
            s,
            [=](const ck_tile::stream_config& s_) { pre_kernel_launch(); },
            [=](const ck_tile::stream_config& s_) { dqdkdv_kernel_launch(); });
    }

    int dq_acc_element_size = a.v3_atomic_fp32 ? 4 : 2;
    fmha_bwd_post_kernel_args post_args;
    if(!use_ck_dq_convert)
    {
        arg_size                  = sizeof(post_args);
        post_args.ptr_dq_acc      = a.dq_acc_ptr;
        post_args.ptr_dq          = a.dq_ptr;
        post_args.Hs_dq_acc       = a.nhead_stride_dq_acc * dq_acc_element_size;
        post_args.BAs_dq_acc      = a.batch_stride_dq_acc * dq_acc_element_size;
        post_args.Seqs_dq_acc     = a.stride_dq_acc * dq_acc_element_size;
        post_args.Hs_dq           = a.nhead_stride_dq * 2;
        post_args.BAs_dq          = a.batch_stride_dq * 2;
        post_args.Seqs_dq         = a.stride_dq * 2;
        post_args.seqlen_q        = a.seqlen_q;
        post_args.head_dim        = a.hdim_q;
        post_args.ptr_qseq_padded = a.seqstart_q_ptr;
        post_args.ptr_qseq =
            (a.cu_seqlen_q_ptr && a.seqstart_q_ptr) ? a.cu_seqlen_q_ptr : a.seqstart_q_ptr;
    }

    auto post_kernel_launch = [&]() {
        if(use_ck_dq_convert)
        {
            launch_ck_dq_convert(a, ck_tile::stream_config{s.stream_id_});
        }
        else
        {
            int bdx = 256;
            int gdx = (a.max_seqlen_q + ts_dq - 1) / ts_dq;
            int gdy = a.nhead_q;
            int gdz = a.batch;
            impl_ptr_post->launch_kernel(
                {&post_args, &arg_size, gdx, gdy, gdz, bdx, 1, 1, s.stream_id_});
        }
    };
    return ck_tile::launch_kernel(
        s,
        [=](const ck_tile::stream_config& s_) { pre_kernel_launch(); },
        [=](const ck_tile::stream_config& s_) { dqdkdv_kernel_launch(); },
        [=](const ck_tile::stream_config& s_) { post_kernel_launch(); });
}

} // namespace aiter
