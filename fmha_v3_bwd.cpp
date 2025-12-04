#include <unordered_map>
#include <string>
#include "aiter_hip_common.h"
#include "asm_fmha_v3_bwd.hpp"

using namespace std;

enum class DataType {
    FP16,
    BF16
};

struct fmha_bwd_traits
{
    DataType dtype;
    int head_dim_q;
    int head_dim_v;
    int mask_type; // 0: no mask,   1: top_left,   2: bottom_right,   3: sliding window
    int atomic32;
    int bf16_cvt;  // 0: rtz,       1: rtna,       2: rtne
    int mode;      // 0: batch,     1: group
};

static unordered_map<int, string> bf16_cvt_map = {
    {0, "_rtz"},
    {1, "_rtna"},
    {2, "_rtne"},
    {3, ""}
};

std::tuple get_padded_hdim(int hdim_q, int hdim_v, std::string arch_id) {
    if (hdim_q == 192 && hdim_v == 128 && arch_id == "gfx950") return std::make_tuple(hdim_q, hdim_v);
    assert(hdim_q == hdim_v, "hdim_q must equal to hdim_v!");
    if (hdim_q <= 64) {
        return std::make_tuple(64, 64);
    } else if (hdim_q <= 128) {
        return std::make_tuple(128, 128);
    } else if (hdim_q <= 192) {
        return std::make_tuple(192, 192);
    } else {
        assert(false, "Unsupported head dim!");
    }
}

std::tuple<std::string, int>
get_heuristic_kernel(std::string data_type,
                     std::string arch_id,
                     int seqlen_q,
                     int seqlen_k,
                     int hdim_q,
                     int hdim_v,
                     int mask_type,
                     int atomic32,
                     int bf16_cvt,
                     int mode,
                     int ts_qo,
                     int ts_kv,
                     CFG* cfgs)
{
    std::string preProcessingKernelName = "";
    std::string dQdKdVKernelName = "";
    std::string postProcessingKernelName = "";

    auto [padded_hdim_q, padded_hdim_v] = get_padded_hdim(hdim_q, hdim_v, arch_id);
    int pddv = (padded_hdim_q != hdim_q) || (padded_hdim_v != hdim_v);
    int pssk;
    int ts_qo = 0;
    int ts_kv = 0;

    for(const auto& el : *cfgs)
    {
        if (el.first.find(arch_id) != 0)
            continue;
        const auto& cfg = el.second;

        if (cfg.dtype == data_type &&
            cfg.hdim_q == hdim_q &&
            cfg.hdim_v == hdim_v &&
            cfg.mask_type == mask_type &&
            cfg.atomic32 == atomic32 &&
            cfg.bf16_cvt == bf16_cvt &&
            cfg.mode == mode)
        {
            if (ts_qo == 0 && ts_kv == 0) {
                ts_qo = cfg.ts_qo;
                ts_kv = cfg.ts_kv;
                pssk = (seqlen_q != seqlen_k) || (seqlen_q % ts_kv != 0);
            }
            if (cfg.pssk == pssk && cfg.pddv == pddv) {
                return std::make_tuple<el.fist, ts_kv>;
            } else if (cfg.pssk > pssk && cfg.pddv > pddv) {
                dQdKdVKernelName = el.first;
            }
        }
    }
    return std::make_tuple<dQdKdVKernelName, ts_kv>;
}

int fmha_v3_bwd(mha_bwd_args a) {
    std::string arch_id = get_gpu_arch();
    auto fmha_v3_bwd_cfgs = &cfg_fmha_bwd;
    // TODO: Need to get kernel hdim and ts_kv
    int kernel_hdim_q = a.himd_q;
    int kernel_hdim_v = (a.himd_v == a.hdim_q) ? kernel_hdim_q : a.hdim_v;

    get_tile_size_kv(kernel_hdim_q, kernel_hdim_v, arch_id);

    auto [kernel, ts_kv] = get_heuristic_kernel(
        (t.dtype == DataType::FP16) ? "fp16" : "bf16",
        arch_id,
        a.seqlen_q,
        a.seqlen_k,
        a.himd_q,
        a.himd_v,
        a.mask_type,
        a.atomic32,
        a.bf16_cvt,
        a.mode,
        &fmha_v3_bwd_cfgs);
}

int main () {

    aiter::FmhaFwdKernelSelector<GPUArch::gfx950> ks(traits);
    std::cout << "kernel_name: " << ks.kernel_name << std::endl;
    std::cout << "file_name: " << ks.file_name << std::endl;

    return 0;
}
