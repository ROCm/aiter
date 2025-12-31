#include "mha_fwd.h"
#include "aiter_hip_common.h"
#if FAV3_ON
#include "asm_fmha_v3_fwd_configs.hpp"
#endif
#include <memory>
#include <string>

namespace aiter {
#if FAV3_ON

int get_cfg_mask_type(const mha_fwd_args& a)
{
    if(a.mask_type == 0)
    {
        return 0;
    }
    if((a.mask_type == 2 || (a.mask_type == 1 && a.seqlen_q == a.seqlen_k)) &&
       a.window_size_left == -1 && a.window_size_right == 0)
    {
        return 2;
    }
    return -1;
}

std::string get_kernel_name_key(const std::string& arch_id,
                                const std::string& data_type,
                                int hdim_q,
                                int hdim_v,
                                int mask_type,
                                int bf16_cvt,
                                bool mode,
                                const CFG* cfgs)
{
    std::string kernel_name_key{};
    for(const auto& el : *cfgs)
    {
        const auto& cfg = el.second;
        if(cfg.arch != arch_id)
        {
            continue;
        }

        if(cfg.dtype == data_type && cfg.hdim_q == hdim_q && cfg.hdim_v == hdim_v &&
           cfg.mask == mask_type && cfg.mode == mode)
        {
            if(arch_id == "gfx950")
            {
                kernel_name_key = el.first;
                break;
            }
            else if(arch_id == "gfx942" && cfg.bf16_cvt == bf16_cvt)
            {
                kernel_name_key = el.first;
                break;
            }
        }
    }

    return kernel_name_key;
}

std::string get_kernel_co_name(const std::string& cfg_co_name, const std::string& arch_id)
{
    std::string co_name = cfg_co_name;
    if(arch_id == "gfx942")
    {
        auto pos        = cfg_co_name.rfind('/');
        uint32_t cu_num = get_num_cu_func();
        if(cu_num == 304)
        {
            co_name = cfg_co_name.substr(0, pos + 1) + "MI300/" + cfg_co_name.substr(pos + 1);
        }
        else if(cu_num == 80 || cu_num == 64)
        {
            co_name = cfg_co_name.substr(0, pos + 1) + "MI308/" + cfg_co_name.substr(pos + 1);
        }
    }
    return co_name;
}

void init_fmha_fwd_v3_args(fmha_fwd_v3_args& args,
                           const mha_fwd_args& a,
                           int ts_qo,
                           const std::string& arch_id)
{
    int tune_opt = 5;
    // if num_head is not 8N, or seqlen is bigger than 16K, downgrade to 2and3
    if(a.mask_type != 0 && ((a.nhead_q % 8 != 0) || (a.seqlen_q > 16384)))
    {
        tune_opt -= 2;
    }
    if(a.hdim_q == 192 && a.hdim_v == 128 && arch_id == "gfx942")
    {
        tune_opt = 0;
    }
    args.ptr_o   = a.o_ptr;
    args.ptr_q   = a.q_ptr;
    args.ptr_k   = a.k_ptr;
    args.ptr_v   = a.v_ptr;
    args.ptr_lse = a.lse_ptr;

    args.scalar           = a.scale_s;
    args.s_seq_len        = a.seqlen_q;
    args.s_Seqs           = a.stride_q * 2;
    args.s_Ts             = ts_qo * a.stride_q * 2;
    args.s_Hs             = a.nhead_stride_q * 2;
    args.s_Bs             = a.batch_stride_q * 2;
    args.s_gqa            = a.nhead_q / a.nhead_k;
    args.s_k_Seqs         = a.stride_k * 2;
    args.s_k_Hs           = a.nhead_stride_k * 2;
    args.s_k_Bs           = a.batch_stride_k * 2;
    args.s_opt            = tune_opt;
    args.s_lse            = a.has_lse ? 1 : 0;
    args.s_kv_seq_len     = a.seqlen_k;
    args.s_qk_head_dim    = a.hdim_q;
    args.s_v_head_dim     = a.hdim_v;
    args.s_q_head_num     = a.nhead_q;
    args.s_v_Seqs         = a.stride_v * 2;
    args.s_v_Hs           = a.nhead_stride_v * 2;
    args.s_v_Bs           = a.batch_stride_v * 2;
    args.s_o_Seqs         = a.stride_o * 2;
    args.s_o_Hs           = a.nhead_stride_o * 2;
    args.s_o_Bs           = a.batch_stride_o * 2;
    args.s_lse_Hs         = a.nhead_stride_lse * 4;
    args.ptr_qseq         = nullptr;
    args.ptr_kseq         = nullptr;
    args.ptr_qseq_padding = nullptr;
    args.ptr_kseq_padding = nullptr;
    // batch mode does not support padded
    if(a.is_group_mode)
    {
        args.ptr_kseq_padding = a.seqstart_k_ptr;
        if(a.cu_seqlen_k_ptr && a.seqstart_k_ptr)
        {
            args.ptr_kseq = a.cu_seqlen_k_ptr;
        }
        else
        {
            args.ptr_kseq = a.seqstart_k_ptr;
        }
        args.ptr_qseq_padding = a.seqstart_q_ptr;
        if(a.cu_seqlen_q_ptr && a.seqstart_q_ptr)
        {
            args.ptr_qseq = a.cu_seqlen_q_ptr;
        }
        else
        {
            args.ptr_qseq = a.seqstart_q_ptr;
        }
    }
}

std::tuple<int, int, int> get_grid_dim(const mha_fwd_args& a, int ts_qo, const std::string& arch_id)
{

    int tg_div = (a.mask_type != 0) ? 2 : 1;
    if(arch_id == "gfx942" && a.is_group_mode && a.hdim_q == 192 && a.hdim_v == 128)
    {
        tg_div = 1; // do not merge the head and tail in seqlen_q direction
    }
    // batch
    int gdx = ((a.seqlen_q + ts_qo - 1) / ts_qo + tg_div - 1) / tg_div;
    int gdy = a.nhead_q;
    int gdz = a.batch;
    if(arch_id == "gfx942" && a.hdim_q == 192 && a.hdim_v == 128)
    {
        gdx = a.nhead_q;
        gdy = (a.seqlen_q + ts_qo - 1) /
              ts_qo; // do not merge the head and tail in seqlen_q direction
        gdz = a.batch;
    }
    // group
    if(a.is_group_mode)
    {
        gdx = a.nhead_q;
        gdy = a.batch;
        gdz = ((a.seqlen_q + ts_qo - 1) / ts_qo + tg_div - 1) / tg_div;
    }

    return std::make_tuple(gdx, gdy, gdz);
}

float fmha_fwd_v3(mha_fwd_args a, const ck_tile::stream_config& s)
{
    std::string arch_id = get_gpu_arch();

    if((!a.use_asm_v3) || (a.hdim_q != 192 && a.hdim_q != 128) || (a.hdim_v != 128) ||
       (a.data_type != "bf16") || (a.bias_type != 0) || (a.p_drop > 0.f) ||
       ((arch_id != "gfx942") && (arch_id != "gfx950")))
    {
        std::cout << "[Warning]unsupported condition in fwd_v3!!!" << std::endl;
        return -1;
    }

    auto fwd_cfgs               = &cfg_fmha_fwd;
    int cfg_mask_type           = get_cfg_mask_type(a);
    std::string kernel_name_key = get_kernel_name_key(arch_id,
                                                      a.data_type,
                                                      a.hdim_q,
                                                      a.hdim_v,
                                                      cfg_mask_type,
                                                      a.how_v3_bf16_cvt,
                                                      a.is_group_mode,
                                                      fwd_cfgs);
    auto it                     = fwd_cfgs->find(kernel_name_key);
    if(it == fwd_cfgs->end())
    {
        return -1;
    };

    if(a.v3_api_check)
    {
        return 1;
    };

    AiterAsmKernel* impl_ptr = nullptr;
    static std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> impl_ptr_map;
    const auto& cfg     = it->second;
    const char* name    = cfg.knl_name.c_str();
    std::string co_name = get_kernel_co_name(cfg.co_name, arch_id);

    auto result = impl_ptr_map.emplace(name, nullptr);
    if(result.second)
    {
        result.first->second = std::make_unique<AiterAsmKernel>(name, co_name.c_str());
    }
    impl_ptr = result.first->second.get();
    fmha_fwd_v3_args args;
    int arg_size = sizeof(args);
    init_fmha_fwd_v3_args(args, a, cfg.ts_qo, arch_id);

    auto fwd_kernel_launch = [&]() {
        int bdx              = (a.hdim_q == 192 && a.hdim_v == 128) ? 256 : 512;
        auto [gdx, gdy, gdz] = get_grid_dim(a, cfg.ts_qo, arch_id);
        impl_ptr->launch_kernel({&args, &arg_size, gdx, gdy, gdz, bdx, 1, 1, s.stream_id_});
    };
    return ck_tile::launch_kernel(s,
                                  [=](const ck_tile::stream_config& s_) { fwd_kernel_launch(); });
}
#endif

#if FAV2_ON
float fmha_fwd_ck(mha_fwd_args a, const ck_tile::stream_config& s)
{
    fmha_fwd_traits traits{a.hdim_q,
                           a.hdim_v,
                           a.data_type,
                           a.is_group_mode,
                           true, // is_v_rowmajor
                           a.logits_soft_cap > 0.f,
                           static_cast<mask_enum>(a.mask_type),
                           static_cast<bias_enum>(a.bias_type),
                           a.has_lse,
                           a.p_drop > 0.f,
                           static_cast<quant_scale_enum>(a.qscale_type),
                           a.min_seqlen_q != 0,
                           a.has_sink};

    fmha_fwd_args args{a.q_ptr,
                       a.k_ptr,
                       a.v_ptr,
                       a.bias_ptr,
                       a.q_descale_ptr,
                       a.k_descale_ptr,
                       a.v_descale_ptr,
                       a.rand_val_ptr,
                       a.lse_ptr,
                       a.o_ptr,
                       a.seqstart_q_ptr,
                       a.seqstart_k_ptr,
                       a.seqlen_q_ptr,
                       a.seqlen_k_ptr,
                       a.cu_seqlen_q_ptr,
                       a.cu_seqlen_k_ptr,
                       a.seqlen_q,
                       a.seqlen_k,
                       a.batch,
                       a.max_seqlen_q,
                       a.hdim_q,
                       a.hdim_v,
                       a.nhead_q,
                       a.nhead_k,
                       a.scale_s,
                       a.logits_soft_cap,
                       a.stride_q,
                       a.stride_k,
                       a.stride_v,
                       a.stride_bias,
                       a.stride_randval,
                       a.stride_o,
                       a.nhead_stride_q,
                       a.nhead_stride_k,
                       a.nhead_stride_v,
                       a.nhead_stride_bias,
                       a.nhead_stride_randval,
                       a.nhead_stride_lse,
                       a.nhead_stride_o,
                       a.batch_stride_q,
                       a.batch_stride_k,
                       a.batch_stride_v,
                       a.batch_stride_bias,
                       a.batch_stride_randval,
                       a.batch_stride_lse,
                       a.batch_stride_o,
                       a.window_size_left,
                       a.window_size_right,
                       a.sink_size,
                       a.mask_type,
                       a.min_seqlen_q,
                       a.p_drop,
                       a.s_randval,
                       a.drop_seed_offset};

    return fmha_fwd(traits, args, s);
}
#endif

float mha_fwd(mha_fwd_args args, const ck_tile::stream_config& s)
{
    float ret = -1;

#if FAV3_ON
    ret = fmha_fwd_v3(args, s);
#endif

#if FAV2_ON
    if(ret == -1 && !args.v3_api_check)
    {
        ret = fmha_fwd_ck(args, s);
    }
#endif
    return ret;
}

} // namespace aiter
