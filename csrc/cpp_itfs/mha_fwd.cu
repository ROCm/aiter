#include "mha_fwd.h"
#include "aiter_hip_common.h"
#if FAV3_ON
#include "asm_fmha_v3_fwd_configs.hpp"
#endif
#if FAV_NATIVE_ON
#include "mha_native_launch.h"  // launch_msk{0,1}_split, launch_combine; pulls in runner/params.hpp
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
    if(a.data_type == "fp8bf16")
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
    if(arch_id == "gfx950" && a.data_type == "fp8bf16")
    {
        tg_div = 1; // fp8 asm kernel processes one Q tile per workgroup
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
       (a.data_type != "bf16" && a.data_type != "fp8bf16") || (a.bias_type != 0) || (a.p_drop > 0.f) ||
       ((arch_id != "gfx942") && (arch_id != "gfx950")))
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

#if FAV_NATIVE_ON
// Split-count heuristic for the native D64 BF16 split-K kernel on gfx942.
// Mirrors the Python heuristic in aiter/ops/mha.py::_native_splitkv_heuristic.
// Tile geometry constants come from runner/params.hpp (kM0=128, kN0=64).
static int native_splitkv_heuristic(int batch, int nhead_q, int seqlen_q, int seqlen_k,
                                    int num_cu)
{
    auto snap = [](int x) -> int {
        int g = 0;
        for(int c : {2, 4, 8, 16})
            if(c <= x) g = c;
        return g;
    };
    const int sq_tile = kM0;       // 128 query rows per block (from params.hpp)
    const int kv_tile = kN0;       // 64  key   cols per block
    const int sqt     = (seqlen_q + sq_tile - 1) / sq_tile;
    const int skvt    = (seqlen_k + kv_tile - 1) / kv_tile;
    const int nwg     = batch * nhead_q * sqt;

    const int kvdiv  = (nwg < 24) ? 10 : 28;
    const int kv_cap = snap(skvt / kvdiv);

    if(nwg < num_cu)
    {
        const int occ_cap = snap(static_cast<int>(3.5f * num_cu / nwg));
        return (occ_cap < kv_cap) ? occ_cap : kv_cap;
    }
    if(batch >= 2)
        return 0;

    const float over = static_cast<float>(nwg) / num_cu;
    if(skvt < static_cast<int>(10.f * over) && over < 30.f)
        return 0;

    int g = (nhead_q <= 8) ? 4 : 2;
    if(over >= 30.f)
    {
        const int ext = snap(skvt / 160);
        if(ext > g) g = ext;
    }
    return (kv_cap > 0) ? ((g < kv_cap) ? g : kv_cap) : 0;
}

// mha_fwd dispatch path for the native hand-written HIP D64 BF16 split-K kernel.
// Supports gfx942, dense bf16, D64, batch mode, no bias/alibi/dropout/descale/sink/SWA.
// Returns 1.f on success, -1.f when the config is unsupported and should fall through.
static float fmha_fwd_native_gfx942(const mha_fwd_args& a, const ck_tile::stream_config& s)
{
    // Guard: arch, dtype, geometry, no-extras.
    if(get_gpu_arch() != "gfx942")
        return -1.f;
    if(a.data_type != "bf16")
        return -1.f;
    if(a.hdim_q != 64 || a.hdim_v != 64)
        return -1.f;
    if(a.is_group_mode)  // varlen not supported
        return -1.f;
    if(a.bias_type != 0)
        return -1.f;
    if(a.p_drop > 0.f)
        return -1.f;
    if(a.q_descale_ptr || a.k_descale_ptr || a.v_descale_ptr)
        return -1.f;
    if(a.sink_ptr || a.sink_size != 0)
        return -1.f;
    // Only full causal (mask_type 1 or 2 → causal=true) or no mask (mask_type 0).
    // SWA (window_size_left != -1 while right == 0) falls through to ASM/CK.
    const bool causal = (a.mask_type == 1 || a.mask_type == 2);
    if(a.mask_type != 0 && !causal)
        return -1.f;
    if(a.window_size_left != -1 || (a.window_size_right != 0 && a.window_size_right != -1))
        return -1.f;
    // Guard against NaN: bottom-right causal with sq > sk produces fully-masked rows.
    if(causal && a.seqlen_q > a.seqlen_k)
        return -1.f;
    // Native split-K implements bottom-right causal only (mask_shift = sk - sq).
    // A top-left causal request (mask_type==1) coincides with bottom-right only when
    // seqlen_q == seqlen_k. For top-left with sq != sk, fall through to the batch
    // (non-split) V3/CK path, which has a real top-left causal kernel.
    if(a.mask_type == 1 && a.seqlen_q != a.seqlen_k)
        return -1.f;
    if(a.nhead_q <= 0 || a.nhead_k <= 0 || a.nhead_q % a.nhead_k != 0)
        return -1.f;

    const int B  = a.batch;
    const int Sq = a.seqlen_q;
    const int Sk = a.seqlen_k;
    const int Hq = a.nhead_q;

    // Split count: heuristic (G=0 means non-beneficial, fall through to ASM/CK).
    const int G = native_splitkv_heuristic(B, Hq, Sq, Sk,
                                           static_cast<int>(get_num_cu_func()));
    if(G <= 0)
        return -1.f;

    // Pre-multiply scale by log2(e) — the kernel's softmax is base-2.
    static constexpr float kLog2e = 1.4426950408889634f;
    const float scale_b2 = a.scale_s * kLog2e;

    // Allocate split-major scratch on device: [G][B][Hq][Sq][64] fp32 + [G][B][Hq][Sq] fp32.
    // Use hipMallocAsync / hipFreeAsync (stream-ordered pool allocator) so the alloc/free
    // are enqueued on the stream and don't force a device-wide synchronization.
    const size_t scratch_o_elems   = static_cast<size_t>(G) * B * Hq * Sq * kHeadDim;
    const size_t scratch_lse_elems = static_cast<size_t>(G) * B * Hq * Sq;
    float* scratch_o   = nullptr;
    float* scratch_lse = nullptr;

    const hipStream_t stream = s.stream_id_;

    HIP_CALL(hipMallocAsync(&scratch_o,   scratch_o_elems   * sizeof(float), stream));
    HIP_CALL(hipMallocAsync(&scratch_lse, scratch_lse_elems * sizeof(float), stream));

    FmhaFwdParams base{};
    base.q   = reinterpret_cast<const __hip_bfloat16*>(a.q_ptr);
    base.k   = reinterpret_cast<const __hip_bfloat16*>(a.k_ptr);
    base.v   = reinterpret_cast<const __hip_bfloat16*>(a.v_ptr);
    base.o   = reinterpret_cast<__hip_bfloat16*>(a.o_ptr);
    base.lse = nullptr;  // producers write scratch_lse; combine writes final lse
    base.seqlen_q = Sq; base.seqlen_k = Sk;
    base.nhead_q  = Hq; base.nhead_k  = a.nhead_k;
    base.scale    = scale_b2;
    base.stride_q = a.stride_q;       base.nhead_stride_q = a.nhead_stride_q; base.batch_stride_q = a.batch_stride_q;
    base.stride_k = a.stride_k;       base.nhead_stride_k = a.nhead_stride_k; base.batch_stride_k = a.batch_stride_k;
    base.stride_v = a.stride_v;       base.nhead_stride_v = a.nhead_stride_v; base.batch_stride_v = a.batch_stride_v;
    base.stride_o = a.stride_o;       base.nhead_stride_o = a.nhead_stride_o; base.batch_stride_o = a.batch_stride_o;
    base.seqstart_q = nullptr; base.seqstart_k = nullptr;

    FmhaFwdSplitParams sp{};
    sp.base        = base;
    sp.scratch_o   = scratch_o;
    sp.scratch_lse = scratch_lse;
    sp.num_splits  = G;
    sp.split_idx   = 0;  // vestigial: device decodes split from blockIdx.z

    const int m_tiles = (Sq + kM0 - 1) / kM0;
    dim3 grid_prod(Hq, m_tiles, B * G);
    if(causal) launch_msk1_split(sp, grid_prod, stream);
    else        launch_msk0_split(sp, grid_prod, stream);

    FmhaFwdCombineParams cp{};
    cp.scratch_o   = scratch_o;
    cp.scratch_lse = scratch_lse;
    cp.o           = reinterpret_cast<__hip_bfloat16*>(a.o_ptr);
    cp.lse         = reinterpret_cast<float*>(a.lse_ptr);  // nullptr → skip LSE write
    cp.num_splits  = G;
    cp.seqlen_q    = Sq;
    cp.nhead_q     = Hq;
    cp.stride_o = a.stride_o; cp.nhead_stride_o = a.nhead_stride_o; cp.batch_stride_o = a.batch_stride_o;
    cp.scale    = scale_b2;
    cp.o_fp32   = nullptr;

    dim3 grid_comb(Hq, m_tiles, B);
    launch_combine(cp, grid_comb, stream);

    // Stream-ordered free: enqueued after the combine launch so the allocator can
    // reuse this memory as soon as the combine kernel completes on the stream.
    HIP_CALL(hipFreeAsync(scratch_o,   stream));
    HIP_CALL(hipFreeAsync(scratch_lse, stream));

    return 1.f;
}
#endif

float mha_fwd(mha_fwd_args args, const ck_tile::stream_config& s)
{
    float ret = -1;

#if FAV_NATIVE_ON
    ret = fmha_fwd_native_gfx942(args, s);
    if(ret != -1)
        return ret;
#endif

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
