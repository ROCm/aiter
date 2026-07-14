#include "mha_fwd.h"
#include "aiter_hip_common.h"
#if FAV3_ON
#include "asm_fmha_v3_fwd_configs.hpp"
#endif
#include <memory>
#include <string>
#include <cstdlib>

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
        auto pos = cfg_co_name.rfind('/');
        if(is_mi308_device())
        {
            co_name = cfg_co_name.substr(0, pos + 1) + "MI308/" + cfg_co_name.substr(pos + 1);
        }
        else
        {
            co_name = cfg_co_name.substr(0, pos + 1) + "MI300/" + cfg_co_name.substr(pos + 1);
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
    args.ptr_o            = a.o_ptr;
    args.ptr_q            = a.q_ptr;
    args.ptr_k            = a.k_ptr;
    args.ptr_v            = a.v_ptr;
    args.ptr_lse          = a.lse_ptr;
    args.ptr_qseq         = nullptr;
    args.ptr_kseq         = nullptr;
    args.ptr_qseq_padding = nullptr;
    args.ptr_kseq_padding = nullptr;
    args.ptr_q_descale    = nullptr;
    args.ptr_k_descale    = nullptr;
    args.ptr_v_descale    = nullptr;
    args.s_descale_q_Bs   = 0;
    args.s_descale_q_Hs   = 0;
    args.s_descale_k_Bs   = 0;
    args.s_descale_k_Hs   = 0;
    args.s_descale_v_Bs   = 0;
    args.s_descale_v_Hs   = 0;

    int in_bpe = 2;
    int out_bpe = 2;
    if(a.data_type == "fp8bf16" || a.data_type == "i8fp8bf16" ||
        a.data_type == "mxfp4bf16" || a.data_type == "mxfp6bf16" || a.data_type == "fp6fp6bf16")
    {
        in_bpe = 1;
        args.ptr_q_descale = a.q_descale_ptr;
        args.ptr_k_descale = a.k_descale_ptr;
        args.ptr_v_descale = a.v_descale_ptr;
        args.s_descale_q_Bs = a.batch_stride_q_descale * 4;
        args.s_descale_q_Hs = a.nhead_stride_q_descale * 4;
        args.s_descale_k_Bs = a.batch_stride_k_descale * 4;
        args.s_descale_k_Hs = a.nhead_stride_k_descale * 4;
        args.s_descale_v_Bs = a.batch_stride_v_descale * 4;
        args.s_descale_v_Hs = a.nhead_stride_v_descale * 4;
    }

    args.scalar           = a.scale_s;
    args.s_seq_len        = a.seqlen_q;
    args.s_Seqs           = a.stride_q * in_bpe;
    args.s_Ts             = ts_qo * a.stride_q * in_bpe;
    args.s_Hs             = a.nhead_stride_q * in_bpe;
    args.s_Bs             = a.batch_stride_q * in_bpe;
    args.s_gqa            = a.nhead_q / a.nhead_k;
    args.s_k_Seqs         = a.stride_k * in_bpe;
    args.s_k_Hs           = a.nhead_stride_k * in_bpe;
    args.s_k_Bs           = a.batch_stride_k * in_bpe;
    args.s_opt            = tune_opt;
    args.s_lse            = a.has_lse ? 1 : 0;
    args.s_kv_seq_len     = a.seqlen_k;
    args.s_qk_head_dim    = a.hdim_q;
    args.s_v_head_dim     = a.hdim_v;
    args.s_q_head_num     = a.nhead_q;
    args.s_v_Seqs         = a.stride_v * in_bpe;
    args.s_v_Hs           = a.nhead_stride_v * in_bpe;
    args.s_v_Bs           = a.batch_stride_v * in_bpe;
    args.s_o_Seqs         = a.stride_o * out_bpe;
    args.s_o_Hs           = a.nhead_stride_o * out_bpe;
    args.s_o_Bs           = a.batch_stride_o * out_bpe;
    args.s_lse_Hs         = a.nhead_stride_lse * 4;
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
    if(!a.use_asm_v3)
        return -1;

    std::string arch_id = get_gpu_arch();

    if((a.hdim_q != 192 && a.hdim_q != 128) || (a.hdim_v != 128) ||
       (a.data_type != "bf16" && a.data_type != "fp8bf16" && 
        a.data_type != "i8fp8bf16" && a.data_type != "mxfp4bf16" &&
        a.data_type != "mxfp6bf16" && a.data_type != "fp6fp6bf16") ||
       (a.bias_type != 0) || (a.p_drop > 0.f) || ((arch_id != "gfx942") && (arch_id != "gfx950")))
    {
        AITER_LOG_WARNING("unsupported condition in fwd_v3!!! data type: " << a.data_type);
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
    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;

    const auto& cfg     = it->second;
    const char* name    = cfg.knl_name.c_str();
    std::string co_name = get_kernel_co_name(cfg.co_name, arch_id);

    impl_ptr =
        &impl_ptr_map.get_or_create(name, [&]() { return AiterAsmKernel(name, co_name.c_str()); });

    fmha_fwd_v3_args args;
    size_t arg_size = sizeof(args);
    init_fmha_fwd_v3_args(args, a, cfg.ts_qo, arch_id);

    int bdx              = (a.hdim_q == 192 && a.hdim_v == 128) ? 256 : 512;
    auto [gdx, gdy, gdz] = get_grid_dim(a, cfg.ts_qo, arch_id);

    // ---- Persistent-kernel override (opt-in via AITER_FMHA_PERSISTENT=P) ----
    // Launch only P workgroups that grid-stride over the gdx*gdy*gdz tiles via a
    // device atomic counter. The asm kernel reads the counter ptr from the (unused
    // in this path) ptr_lse slot and total_tiles from s_lse; total==0 keeps the
    // original one-workgroup-per-tile behaviour. Restricted to the batch, non-causal
    // mxfp6 gfx950 path that implements the in-kernel loop.
    static const int persist_p = []() {
        const char* e = getenv("AITER_FMHA_PERSISTENT");
        return e ? atoi(e) : 0;
    }();
    if(persist_p > 0 && !a.is_group_mode && a.mask_type == 0 &&
       a.data_type == "mxfp6bf16" && arch_id == "gfx950")
    {
        long long total = (long long)gdx * gdy * gdz;
        // WG i is pre-assigned linear tile i by its program id; the device counter therefore
        // starts at the launched-WG count (the first NOT-pre-assigned tile), and the kernel's
        // atomic only hands out the overflow tiles [launch_wgs, total). Counter==0 + kernel
        // pid-skip would double-process tiles [0, launch_wgs); they MUST stay consistent.
        const int launch_wgs = (persist_p < total) ? persist_p : static_cast<int>(total);
        static uint32_t* g_counter = nullptr;
        if(g_counter == nullptr)
            (void)hipMalloc(reinterpret_cast<void**>(&g_counter), sizeof(uint32_t));
        (void)hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(g_counter),
                                static_cast<unsigned int>(launch_wgs), 1, s.stream_id_);
        args.ptr_lse = g_counter;          // counter ptr (kernel @0x40)
        args.s_lse   = static_cast<unsigned int>(total);  // total tiles (kernel @0x100)
        gdx          = launch_wgs;
        gdy          = 1;
        gdz          = 1;
    }

    // ---- SOLO single-wave override (opt-in via AITER_FMHA_SOLO=1) ----------
    // Launches the single-wave (bdx=64) persistent kernel built with SOLO_MODE=1: one WG owns ONE
    // 32-row Q block (vs 256 for the 8-wave kernel), self-loads its full K+V tile, and grid-strides
    // the ceil(seqlen_q/32)*nhead*batch blocks via the same atomic-counter scheme. The deployed .co
    // MUST be the SOLO_MODE=1 build (host params + compiled kernel are a matched pair). Batch /
    // non-causal / mxfp6 / gfx950 only (mirrors the persistent guard).
    static const int solo_on = []() {
        const char* e = getenv("AITER_FMHA_SOLO");
        return e ? atoi(e) : 0;
    }();
    if(solo_on && !a.is_group_mode && a.mask_type == 0 &&
       a.data_type == "mxfp6bf16" && arch_id == "gfx950")
    {
        // Per-WG Q-row span. MUST match kTileQ_eff in the deployed .co (host params + kernel are a
        // matched pair). Stock solo = 32 (one 32-row block/WG). ILP dual-pipeline .co = 64 (two
        // 32-row Q tiles/WG); set AITER_FMHA_SOLO_TS=64 when deploying a SOLO_ILP=1 build.
        static const int ts_solo = []() {
            const char* e = getenv("AITER_FMHA_SOLO_TS");
            return e ? atoi(e) : 32;
        }();
        long long total   = (long long)((a.seqlen_q + ts_solo - 1) / ts_solo) * a.nhead_q * a.batch;
        // 2 waves/SIMD => 8 single-wave WGs per CU.
        static int cu_count = []() {
            int dev = 0, n = 0;
            (void)hipGetDevice(&dev);
            (void)hipDeviceGetAttribute(&n, hipDeviceAttributeMultiprocessorCount, dev);
            return n > 0 ? n : 256;
        }();
        // AITER_FMHA_SOLO_WGS overrides the launched-WG count (A/B the persistent grid-stride): >=total
        // => one WG per tile (non-persistent full grid); default cu_count*8 (the persistent grid-stride).
        static const long long solo_wgs_env = []() {
            const char* e = getenv("AITER_FMHA_SOLO_WGS");
            return e ? atoll(e) : 0LL;
        }();
        // Launch exactly the resident-WG count. The ACC-heavy solo kernel is VGPR-bound to 1 wave/SIMD
        // = 4 WG/CU (the stale "8 WG/CU" over-launched 2x -> ~0.4% slower + a launch/exit tail). Swept
        // optimum @ b1 hq5 sq75520: cu*4=2830 > cu*8=2820 > full-grid=2704 TFLOPS. AITER_FMHA_SOLO_WGS
        // overrides (>=total => non-persistent one-WG-per-tile grid) for A/B.
        const long long want = solo_wgs_env > 0 ? solo_wgs_env : (long long)cu_count * 4;
        const int launch_wgs = static_cast<int>((want < total) ? want : total);
        static uint32_t* g_counter_solo = nullptr;
        if(g_counter_solo == nullptr)
            (void)hipMalloc(reinterpret_cast<void**>(&g_counter_solo), sizeof(uint32_t));
        (void)hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(g_counter_solo),
                                static_cast<unsigned int>(launch_wgs), 1, s.stream_id_);
        args.ptr_lse = g_counter_solo;
        args.s_lse   = static_cast<unsigned int>(total);
        gdx          = launch_wgs;
        gdy          = 1;
        gdz          = 1;
        bdx          = 64;
    }

    // ---- INTRA-WAVE single-wave override (opt-in via AITER_FMHA_INTRA_WAVE=1) ----------------
    // Launches the single-wave (bdx=64) intra-wave fp8 kernel: one WG owns a ts_iw-row Q block and
    // self-loads its full K+V tile. IW1 is ONE-SHOT (one tile per WG, no in-kernel grid-stride) and
    // single-pipeline (ts_iw=32); IW3 will make it two 32-row pipelines (bump ts_iw to 64 here). The
    // deployed .co MUST be the intra-wave build (mi350_fmha_hd128_fp8_intra_wave.py, INTRA_WAVE_MODE=1);
    // enabling this against the stock 8-wave fp8 .co WILL break (that kernel needs 512 threads).
    // Batch / non-causal / fp8 / gfx950 only. See FP8_INTRA_WAVE_PLAN.md.
    static const int intra_wave_on = []() {
        const char* e = getenv("AITER_FMHA_INTRA_WAVE");
        return e ? atoi(e) : 0;
    }();
    if(intra_wave_on && !a.is_group_mode && a.mask_type == 0 &&
       a.data_type == "fp8bf16" && arch_id == "gfx950")
    {
        const int ts_iw = 32;  // IW1: 32-row Q block per WG (single pipeline; matches kTileQ_eff)
        // One-shot grid: (q_tiles, heads, batch) — matches the kernel's tgid_x/y/z mapping and
        // get_grid_dim's non-causal layout. No atomic counter (the IW1 kernel does one tile + exit).
        gdx = (a.seqlen_q + ts_iw - 1) / ts_iw;
        gdy = a.nhead_q;
        gdz = a.batch;
        bdx = 64;
    }

    return ck_tile::launch_kernel(s, [=](const ck_tile::stream_config& s_) mutable {
        // Explicit assignment forces evaluation order and prevents compiler from
        // reordering operations that could lead to accessing uninitialized args
        void* args_ptr     = &args;
        size_t* arg_size_ptr = &arg_size;
        impl_ptr->launch_kernel({args_ptr, arg_size_ptr, gdx, gdy, gdz, bdx, 1, 1, s_.stream_id_});
    });
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
                       a.block_scale_seqstart_q_ptr,
                       a.block_scale_seqstart_k_ptr,
                       nullptr, // seqstart_v_scale_ptr
                       a.sink_ptr,
                       a.seqlen_q,
                       a.seqlen_k,
                       a.batch,
                       a.max_seqlen_q,
                       a.hdim_q,
                       a.hdim_v,
                       a.nhead_q,
                       a.nhead_k,
                       0, // num_head_q_total
                       0, // head_start
                       a.scale_s,
                       a.logits_soft_cap,
                       a.stride_q,
                       a.stride_k,
                       a.stride_v,
                       a.stride_bias,
                       a.stride_randval,
                       a.stride_o,
                       0, // stride_q_descale
                       0, // stride_k_descale
                       0, // stride_v_descale
                       a.nhead_stride_q,
                       a.nhead_stride_k,
                       a.nhead_stride_v,
                       a.nhead_stride_bias,
                       a.nhead_stride_randval,
                       a.nhead_stride_lse,
                       a.nhead_stride_o,
                       a.nhead_stride_q_descale,
                       a.nhead_stride_k_descale,
                       a.nhead_stride_v_descale,
                       a.batch_stride_q,
                       a.batch_stride_k,
                       a.batch_stride_v,
                       a.batch_stride_bias,
                       a.batch_stride_randval,
                       a.batch_stride_lse,
                       a.batch_stride_o,
                       a.batch_stride_q_descale,
                       a.batch_stride_k_descale,
                       a.batch_stride_v_descale,
                       a.window_size_left,
                       a.window_size_right,
                       a.sink_size,
                       a.mask_type,
                       a.min_seqlen_q,
                       a.p_drop,
                       a.s_randval,
                       a.drop_seed_offset,
                       a.block_scale_size_q,
                       a.block_scale_size_kv};

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
