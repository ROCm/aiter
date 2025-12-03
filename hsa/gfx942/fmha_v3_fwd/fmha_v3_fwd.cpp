#include <unordered_map>
#include <string>
#include "toy_format.hpp"
#include "aiter_hip_common.h"

using namespace std;

namespace aiter {

enum class DType {
    FP16,
    BF16
};

struct fmha_fwd_traits
{
    int head_dim;
    DType dtype;
    int mask_type; // 0: no mask,   1: top_left,   2: bottom_right,   3: sliding window
    int bf16_cvt;  // 0: rtz,       1: rtna,       2: rtne
    int mode;      // 0: batch,     1: group
};

static unordered_map<int, string> bf16_cvt_map = {
    {0, "_rtz"},
    {1, "_rtna"},
    {2, "_rtne"},
    {3, ""}
};

template<GPUArch arch>
class FmhaFwdRunner {
public:
    FmhaFwdRunner(fmha_fwd_traits traits) {
        string file_name = GetFileName(traits);
        string kernel_name = GetKernelName(traits);
        // AiterAsmKernel k(file_name, kernel_name);
        std::cout << "file_name: " << file_name << std::endl;
        std::cout << "kernel_name: " << kernel_name << std::endl;
    }

    ~FmhaFwdRunner() {}

private:
    static string GetMetaNameFromArgs(fmha_fwd_traits traits) {
        if constexpr (arch == GPUArch::gfx950) {
            traits.bf16_cvt = 3; // no bf16 cvt for gfx950
        }
        string meta_name = format("hd{}_{}{}{}{}",
                                  traits.head_dim,
                                  traits.dtype == DType::FP16 ? "fp16" : "bf16", // FIXME: add more dtypes if needed
                                  traits.mask_type == 2 ? "_causal" : "",
                                  GetBf16Cvt(traits.bf16_cvt),
                                  traits.mode == 0 ? "" : "_group");

        return meta_name;
    }

    static string GetBf16Cvt(int cvt_type) {
        if constexpr (arch == GPUArch::gfx950) {
            return "";
        } else {
            return bf16_cvt_map[cvt_type];
        }
    }

    string GetKernelName(fmha_fwd_traits traits) {
        string meta_name = GetMetaNameFromArgs(traits);
        string length = to_string(meta_name.length());
        return format("ZN5aiter{}fmha_fwd_{}E", length, meta_name);
    }

    string GetFileName(fmha_fwd_traits traits) {
        string meta_name = GetMetaNameFromArgs(traits);
        if constexpr (arch == GPUArch::gfx950) {
            return format("fmha_v3_fwd/fwd_{}.co", GetMetaNameFromArgs(traits));
        } else {
            return format("fmha_v3_fwd/{}/fwd_{}.co", GetCUDir(), GetMetaNameFromArgs(traits));
        }
    }

    string GetCUDir() {
        uint32_t cu_num = get_num_cu_func();
        if (cu_num == 304) {
            return "MI300";
        } else if (cu_num == 80 || cu_num == 64) {
            return "MI308";
        } else {
            std::cout << cu_num << std::endl;
            return {};
        }
    }
};
} // namespace aiter

int main () {
    aiter::fmha_fwd_traits traits;
    traits.head_dim = 128;
    traits.dtype = aiter::DType::BF16;
    traits.mask_type = 2;
    traits.bf16_cvt = 1;
    traits.mode = 0;

    aiter::FmhaFwdRunner<GPUArch::gfx950> runner(traits);

    return 0;
}
