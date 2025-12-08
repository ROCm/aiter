#include "asm_fmha_v3_bwd_configs.hpp"
#include "aiter_hip_common.h"
#include "fmha_bwd.hpp"
#include "mha_bwd.h"
#include <string>

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

    // std::cout << "padded_hdim_q: " << padded_hdim_q << ", padded_hdim_v: " << padded_hdim_v << std::endl;
    // std::cout << "pddv: " << pddv << std::endl;

    std::string preProcessingKernelName  = "";
    std::string dQdKdVKernelName         = "";
    std::string postProcessingKernelName = "";

    for(const auto& el : *pre_cfgs)
    {
        if(el.first.find(arch_id) != 0)
            continue;
        const auto& cfg = el.second;

        if((cfg.dtype == data_type) && (cfg.hdim_q == padded_hdim_q) && (cfg.mode == mode))
        {
            preProcessingKernelName = el.first;
            break;
        }
    }

    for(const auto& el : *cfgs)
    {
        if(el.first.find(arch_id) != 0) {
            // std::cout << "not supported arch" << std::endl;
            continue;
        }
        const auto& cfg = el.second;

        if((cfg.dtype == data_type) && (cfg.hdim_q == padded_hdim_q) && (cfg.hdim_v == padded_hdim_v) &&
           (cfg.mask == mask_type) && (cfg.atomic32 == atomic32) &&
           ((data_type == "fp16") || (cfg.bf16_cvt == bf16_cvt)) && (cfg.mode == mode))
        {
            // std::cout << "kernel name: " <<  el.first << std::endl;
            if(ts_kv == 0)
            {
                // std::cout << "first chosen ts_kv: " << cfg.ts << std::endl;
                ts_kv= cfg.ts;
                pssk  = (seqlen_q != seqlen_k) || (seqlen_q % ts_kv != 0);
                // std::cout << "pssk: " << pssk << std::endl;
            }
            if((cfg.pssk == pssk) && (cfg.pddv == pddv))
            {
                // std::cout << "all matched case: " << el.first << std::endl;
                dQdKdVKernelName = el.first;
                break;
            }
            else if((cfg.pssk >= pssk) && (cfg.pddv >= pddv))
            {
                // std::cout << "some matched case: " << el.first << std::endl;
                dQdKdVKernelName = el.first;
            }
        }
    }

    if (!post_cfgs) {
        return std::make_tuple(preProcessingKernelName, dQdKdVKernelName, postProcessingKernelName);
    }

    for(const auto& el : *post_cfgs)
    {
        if(el.first.find(arch_id) != 0)
            continue;
        const auto& cfg = el.second;

        if((cfg.dtype == data_type) && (cfg.hdim_q == padded_hdim_q) && (cfg.mode == mode) &&
           ((data_type == "fp16") || (cfg.bf16_cvt == bf16_cvt)))
        {
            postProcessingKernelName = el.first;
            break;
        }
    }
    return std::make_tuple(preProcessingKernelName, dQdKdVKernelName, postProcessingKernelName);
}

float mha_bwd(mha_bwd_args a)
{
    if (a.use_asm_v3 == 1) {
        return fmha_v3_bwd(a);
    }
    // else {
    //     fmha_bwd_traits traits{
    //         a.hdim_q,
    //         a.hdim_v,
    //         a.data_type,
    //         a.is_group_mode,
    //         static_cast<mask_enum>(a.ck_mask_type),
    //         static_cast<bias_enum>(a.bias_type),
    //         a.has_dbias,
    //         a.has_dropout,
    //         a.is_store_randval,
    //         a.is_deterministic
    //     };

    //     fmha_bwd_args ck_args{
    //         /* q_ptr              */ a.q_ptr,
    //         /* k_ptr              */ a.k_ptr,
    //         /* v_ptr              */ a.v_ptr,
    //         /* bias_ptr           */ a.bias_ptr,
    //         /* o_ptr              */ a.o_ptr,
    //         /* lse_ptr            */ a.lse_ptr,
    //         /* do_ptr             */ a.do_ptr,
    //         /* d_ptr              */ a.d_ptr,
    //         /* rand_val_ptr       */ a.rand_val_ptr,
    //         /* dq_ptr             */ a.dq_ptr,
    //         /* dk_ptr             */ a.dk_ptr,
    //         /* dv_ptr             */ a.dv_ptr,
    //         /* dbias_ptr          */ a.dbias_ptr,
    //         /* dq_acc_ptr         */ a.dq_acc_ptr,

    //         /* seqstart_q_ptr     */ a.seqstart_q_ptr,
    //         /* seqstart_k_ptr     */ a.seqstart_k_ptr,
    //         /* seqlen_q_ptr       */ a.seqlen_q_ptr,
    //         /* seqlen_k_ptr       */ a.seqlen_k_ptr,
    //         /* cu_seqlen_q_ptr    */ a.cu_seqlen_q_ptr,
    //         /* cu_seqlen_k_ptr    */ a.cu_seqlen_k_ptr,

    //         /* seqlen_q           */ a.seqlen_q,
    //         /* seqlen_k           */ a.seqlen_k,
    //         /* batch              */ a.batch,
    //         /* max_seqlen_q       */ a.max_seqlen_q,
    //         /* max_seqlen_k       */ a.max_seqlen_k,
    //         /* hdim_q             */ a.hdim_q,
    //         /* hdim_v             */ a.hdim_v,
    //         /* nhead_q            */ a.nhead_q,
    //         /* nhead_k            */ a.nhead_k,
    //         /* scale              */ a.scale,

    //         /* stride_q           */ a.stride_q,
    //         /* stride_k           */ a.stride_k,
    //         /* stride_v           */ a.stride_v,
    //         /* stride_bias        */ a.stride_bias,
    //         /* stride_o           */ a.stride_o,
    //         /* stride_randval     */ a.stride_randval,
    //         /* stride_do          */ a.stride_do,
    //         /* stride_dq_acc      */ a.stride_dq_acc,
    //         /* stride_dq          */ a.stride_dq,
    //         /* stride_dk          */ a.stride_dk,
    //         /* stride_dv          */ a.stride_dv,
    //         /* stride_dbias       */ a.stride_dbias,

    //         /* nhead_stride_q     */ a.nhead_stride_q,
    //         /* nhead_stride_k     */ a.nhead_stride_k,
    //         /* nhead_stride_v     */ a.nhead_stride_v,
    //         /* nhead_stride_bias  */ a.nhead_stride_bias,
    //         /* nhead_stride_o     */ a.nhead_stride_o,
    //         /* nhead_stride_randval*/ a.nhead_stride_randval,
    //         /* nhead_stride_do    */ a.nhead_stride_do,
    //         /* nhead_stride_lsed  */ a.nhead_stride_lsed,
    //         /* nhead_stride_dq_acc*/ a.nhead_stride_dq_acc,
    //         /* nhead_stride_dq    */ a.nhead_stride_dq,
    //         /* nhead_stride_dk    */ a.nhead_stride_dk,
    //         /* nhead_stride_dv    */ a.nhead_stride_dv,
    //         /* nhead_stride_dbias */ a.nhead_stride_dbias,

    //         /* batch_stride_q     */ a.batch_stride_q,
    //         /* batch_stride_k     */ a.batch_stride_k,
    //         /* batch_stride_v     */ a.batch_stride_v,
    //         /* batch_stride_bias  */ a.batch_stride_bias,
    //         /* batch_stride_o     */ a.batch_stride_o,
    //         /* batch_stride_randval*/ a.batch_stride_randval,
    //         /* batch_stride_do    */ a.batch_stride_do,
    //         /* batch_stride_lsed  */ a.batch_stride_lsed,
    //         /* batch_stride_dq_acc*/ a.batch_stride_dq_acc,
    //         /* batch_stride_dq    */ a.batch_stride_dq,
    //         /* batch_stride_dk    */ a.batch_stride_dk,
    //         /* batch_stride_dv    */ a.batch_stride_dv,
    //         /* batch_stride_dbias */ a.batch_stride_dbias,

    //         /* split_stride_dq_acc*/ a.split_stride_dq_acc,
    //         /* window_size_left   */ a.window_size_left,
    //         /* window_size_right  */ a.window_size_right,
    //         /* mask_type          */ a.ck_mask_type,
    //         /* p_drop             */ a.p_drop,
    //         /* p_undrop           */ a.p_undrop,
    //         /* drop_seed_offset   */ a.drop_seed_offset,
    //     };

    //     ck_tile::stream_config stream_config{
    //         a.stream_id_, a.time_kernel_, a.log_level_, a.cold_niters_, a.nrepeat_, a.is_gpu_timer_, a.flush_cache_, a.rotating_count_
    //     };

    //     if (a.use_asm_v3 == 0) {
    //         float asm_ret = fmha_v3_bwd(a);
    //         if(asm_ret == -1)
    //         {
    //             return fmha_bwd(traits, ck_args, stream_config);
    //         }
    //     } else {
    //         return fmha_bwd(traits, ck_args, stream_config);
    //     }
    // }
}

float fmha_v3_bwd(mha_bwd_args a)
{
    int ts_odo;
    int ts_kv;
    int ts_dq;
    int arg_size;

    std::string arch_id = get_gpu_arch();
    auto pre_cfgs       = &cfg_fmha_bwd_odo;
    auto dqdkdv_cfgs    = &cfg_fmha_bwd_dqdkdv;
    auto post_cfgs      = [&]() {
        if (arch_id == "gfx950") {
            if (a.v3_atomic_fp32) {
                return &cfg_fmha_bwd_dq_convert;
            } else {
                return &cfg_fmha_bwd_dq_shuffle;
            }
        } else {
            if (a.v3_atomic_fp32) {
                return &cfg_fmha_bwd_dq_convert;
            } else {
                return static_cast<CFG*>(nullptr);
            }
        }
    }();

    AiterAsmKernel* impl_ptr_pre    = nullptr;
    AiterAsmKernel* impl_ptr_dqdkdv = nullptr;
    AiterAsmKernel* impl_ptr_post   = nullptr;

    bool need_post_processing = (arch_id == "gfx950") || (a.v3_atomic_fp32 == 1);

    auto [pre_kernel, dqdkdv_kernel, post_kernel] =
        get_heuristic_kernel(a.data_type,
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
    auto it_pre = pre_cfgs->find(pre_kernel);
    if(it_pre != pre_cfgs->end())
    {
        const auto& cfg     = it_pre->second;
        const char* name    = cfg.knl_name.c_str();
        const char* co_name = cfg.co_name.c_str();
        ts_odo              = cfg.ts;

        static AiterAsmKernel impl_kenrel(name, co_name);
        impl_ptr_pre = &impl_kenrel;
    }
    else
    {
        return -1;
    }

    auto it_dqdkdv = dqdkdv_cfgs->find(dqdkdv_kernel);
    // std::cout << "it_dqdkdv name: " << dqdkdv_kernel << std::endl;
    if(it_dqdkdv != dqdkdv_cfgs->end())
    {
        const auto& cfg     = it_dqdkdv->second;
        const char* name    = cfg.knl_name.c_str();
        const char* co_name = cfg.co_name.c_str();
        ts_kv               = cfg.ts;

        static AiterAsmKernel impl_kenrel(name, co_name);
        impl_ptr_dqdkdv = &impl_kenrel;
    }
    else
    {
        // std::cout << "Cannot find dqdkdv kernel: " << dqdkdv_kernel << std::endl;
        return -1;
    }

    if (!post_cfgs) {
        assert((!need_post_processing) && (post_kernel == ""));
    } else {
        auto it_post = post_cfgs->find(post_kernel);
        if(it_post != post_cfgs->end())
        {
            const auto& cfg     = it_post->second;
            const char* name    = cfg.knl_name.c_str();
            const char* co_name = cfg.co_name.c_str();
            ts_dq               = cfg.ts;

            static AiterAsmKernel impl_kenrel(name, co_name);
            impl_ptr_post = &impl_kenrel;
        }
        else
        {
            return -1;
        }
    }

    if (a.v3_api_check) return 1;

    fmha_bwd_odo_args odo_args;
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
    odo_args.ptr_qseq        = (a.cu_seqlen_q_ptr && a.seqstart_q_ptr)
                               ? a.cu_seqlen_q_ptr
                               : a.seqstart_q_ptr;

    auto pre_kernel_launch =
        [&]() {
            void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                              &odo_args,
                              HIP_LAUNCH_PARAM_BUFFER_SIZE,
                              &arg_size,
                              HIP_LAUNCH_PARAM_END};

            int bdx = 256;
            int gdx = (a.seqlen_q + ts_odo - 1) / ts_odo;
            int gdy = a.nhead_q;
            int gdz = a.batch;

            impl_ptr_pre->launch_kernel({&odo_args,
                                         &arg_size,
                                         gdx,
                                         gdy,
                                         gdz,
                                         bdx,
                                         1,
                                         1,
                                         a.stream_id_});
        };

    fmha_bwd_dqdkdv_args dqdkdv_args;
    dqdkdv_args.ptr_dq             = need_post_processing ? a.dq_acc_ptr : a.dq_ptr;
    dqdkdv_args.ptr_dk             = a.dk_ptr;
    dqdkdv_args.ptr_dv             = a.dv_ptr;
    dqdkdv_args.ptr_q              = a.q_ptr;
    dqdkdv_args.ptr_k              = a.k_ptr;
    dqdkdv_args.ptr_v              = a.v_ptr;
    dqdkdv_args.ptr_do             = a.do_ptr;
    dqdkdv_args.ptr_lse            = a.lse_ptr;
    dqdkdv_args.ptr_d              = a.d_ptr;
    dqdkdv_args.scalar             = a.scale;
    dqdkdv_args.log2e              = ck_tile::log2e_v<float>;
    dqdkdv_args.ratio              = a.nhead_q / a.nhead_k;
    dqdkdv_args.seqlen_q           = a.seqlen_q;
    dqdkdv_args.seqlen_k           = a.seqlen_k;
    dqdkdv_args.head_dim_q         = a.hdim_q;
    dqdkdv_args.head_dim_v         = a.hdim_v;
    dqdkdv_args.nhead_q            = a.nhead_q;
    dqdkdv_args.Ts                 = ts_kv * a.stride_k * 2;
    dqdkdv_args.Hs_q               = a.nhead_stride_q * 2;
    dqdkdv_args.BAs_q              = a.batch_stride_q * 2;
    dqdkdv_args.Seqs_q             = a.stride_q * 2;
    dqdkdv_args.Hs_k               = a.nhead_stride_k * 2;
    dqdkdv_args.BAs_k              = a.batch_stride_k * 2;
    dqdkdv_args.Seqs_k             = a.stride_k * 2;
    dqdkdv_args.Hs_v               = a.nhead_stride_v * 2;
    dqdkdv_args.BAs_v              = a.batch_stride_v * 2;
    dqdkdv_args.Seqs_v             = a.stride_v * 2;
    dqdkdv_args.Hs_do              = a.nhead_stride_do * 2;
    dqdkdv_args.BAs_do             = a.batch_stride_do * 2;
    dqdkdv_args.Seqs_do            = a.stride_do * 2;
    dqdkdv_args.Hs_dk              = a.nhead_stride_dk * 2;
    dqdkdv_args.BAs_dk             = a.batch_stride_dk * 2;
    dqdkdv_args.Seqs_dk            = a.stride_dk * 2;
    dqdkdv_args.Hs_dv              = a.nhead_stride_dv * 2;
    dqdkdv_args.BAs_dv             = a.batch_stride_dv * 2;
    dqdkdv_args.Seqs_dv            = a.stride_dv * 2;
    dqdkdv_args.Hs_lsed            = a.nhead_stride_lsed * 4;

    if (a.cu_seqlen_k_ptr && a.seqstart_k_ptr) {
        dqdkdv_args.ptr_kseq_padded    = a.seqstart_k_ptr;
        dqdkdv_args.ptr_kseq           = a.cu_seqlen_k_ptr;
    } else {
        dqdkdv_args.ptr_kseq           = a.seqstart_k_ptr;
        dqdkdv_args.ptr_kseq_padded    = a.seqstart_k_ptr;
    }

    if (a.cu_seqlen_q_ptr && a.seqstart_q_ptr) {
        dqdkdv_args.ptr_qseq_padded    = a.seqstart_q_ptr;
        dqdkdv_args.ptr_qseq           = a.cu_seqlen_q_ptr;
    } else {
        dqdkdv_args.ptr_qseq           = a.seqstart_q_ptr;
        dqdkdv_args.ptr_qseq_padded    = a.seqstart_q_ptr;
    }
    dqdkdv_args.max_seqlen_dq     = (a.max_seqlen_q + 15) / 16 * 16;

    // convert l/r to x/y HERE
    if (a.window_size_left == -1 && a.window_size_right == 0) {
        dqdkdv_args.mask_y = 0;
        dqdkdv_args.mask_x = 0;
    } else {
        auto generic_mask = ck_tile::make_generic_attention_mask_coordinates_from_lr_window(
            a.window_size_left, a.window_size_right, a.seqlen_q, a.seqlen_k,
            (a.mask_type == static_cast<ck_tile::index_t>(mask_enum::mask_top_left) ||
            a.mask_type == static_cast<ck_tile::index_t>(mask_enum::window_generic)));
        dqdkdv_args.mask_y = generic_mask.at(ck_tile::number<0>{});
        dqdkdv_args.mask_x = generic_mask.at(ck_tile::number<1>{});
    }
    arg_size                  = sizeof(dqdkdv_args);
    auto dqdkdv_kernel_launch =
        [&]() {
            void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                              &dqdkdv_args,
                              HIP_LAUNCH_PARAM_BUFFER_SIZE,
                              &arg_size,
                              HIP_LAUNCH_PARAM_END};

            int bdx = 256;
            int gdx = (a.max_seqlen_k + ts_kv - 1) / ts_kv;
            int gdy = a.nhead_q;
            int gdz = a.batch;

            if(a.mask_type == 1 || a.mask_type == 2)
            { // sliding window
                gdx = (gdx + 1) / 2;
            }

            impl_ptr_dqdkdv->launch_kernel({&dqdkdv_args,
                                            &arg_size,
                                            gdx,
                                            gdy,
                                            gdz,
                                            bdx,
                                            1,
                                            1,
                                            a.stream_id_});
        };

    if (!need_post_processing) {
        return ck_tile::launch_kernel(
            ck_tile::stream_config{a.stream_id_, a.time_kernel_, a.log_level_, a.cold_niters_, a.nrepeat_, a.is_gpu_timer_, a.flush_cache_, a.rotating_count_},
            [=](const ck_tile::stream_config& s_) { pre_kernel_launch(); },
            [=](const ck_tile::stream_config& s_) { dqdkdv_kernel_launch(); }
        );
    }

    fmha_bwd_post_kernel_args post_args;
    arg_size                  = sizeof(post_args);
    post_args.ptr_dq_acc      = a.dq_acc_ptr;
    post_args.ptr_dq          = a.dq_ptr;
    post_args.Hs_dq_acc       = a.nhead_stride_dq_acc;
    post_args.BAs_dq_acc      = a.batch_stride_dq_acc;
    post_args.Seqs_dq_acc     = a.stride_dq_acc;
    post_args.Hs_dq           = a.nhead_stride_dq;
    post_args.BAs_dq          = a.batch_stride_dq;
    post_args.Seqs_dq         = a.stride_dq;
    post_args.seqlen_q        = a.seqlen_q;
    post_args.head_dim        = a.hdim_q;
    post_args.ptr_qseq_padded = a.seqstart_q_ptr;
    post_args.ptr_qseq        = (a.cu_seqlen_q_ptr && a.seqstart_q_ptr)
                                ? a.cu_seqlen_q_ptr
                                : a.seqstart_q_ptr;

    auto post_kernel_launch =
        [&]() {
            void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                              &post_args,
                              HIP_LAUNCH_PARAM_BUFFER_SIZE,
                              &arg_size,
                              HIP_LAUNCH_PARAM_END};

            int bdx = 256;
            int gdx = (a.seqlen_q + ts_dq - 1) / ts_dq;
            int gdy = a.nhead_q;
            int gdz = a.batch;

            impl_ptr_post->launch_kernel({&post_args,
                                          &arg_size,
                                          gdx,
                                          gdy,
                                          gdz,
                                          bdx,
                                          1,
                                          1,
                                          a.stream_id_});
        };
    ck_tile::stream_config s{a.stream_id_, a.time_kernel_, a.log_level_, a.cold_niters_, a.nrepeat_, a.is_gpu_timer_, a.flush_cache_, a.rotating_count_};
    return ck_tile::launch_kernel(s,
        [=](const ck_tile::stream_config& s_) { pre_kernel_launch(); },
        [=](const ck_tile::stream_config& s_) { dqdkdv_kernel_launch(); },
        [=](const ck_tile::stream_config& s_) { post_kernel_launch(); }
        );
    // pre_kernel_launch();
    // std::cout << "pre_kernel_launch done" << std::endl;
    // dqdkdv_kernel_launch();
    // std::cout << "dqdkdv_kernel_launch done" << std::endl;
    // post_kernel_launch();
    // std::cout << "post_kernel_launch done" << std::endl;
    // return 1;
}

} // namespace aiter
