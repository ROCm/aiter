#include <unordered_map>
#include <string>
#include <format>
#include "aiter_hip_common.h"

using namespace std;

namespace aiter {
static unordered_map<int, string> bf16_cvt_map = {
    {0, "rtz"},
    {1, "rtna"},
    {2, "rtne"}
};

template<typename GPUArch>
class FmhaFwdRunner {
public:
    FmhaFwdRunner(fmha_fwd_args args) {
        string file_name = GetFileName();
        string kernel_name = GetKernelName();

        HIP_CALL(hipModuleLoad(&module, file_name.c_str()));
        HIP_CALL(hipModuleGetFunction(&kernel_func, module, kernel_name.c_str()));
    }

    ~FmhaFwdRunner();

private:
    static string GetMetaNameFromArgs(fmha_fwd_args args) {
        string meta_name = format("fwd_hd{head_dim}_{dtype}{mask}{bf16cvt}{mode}",
                                  args.head_dim,
                                  args.dtype == DType::FP16 ? "fp16" : "bf16",
                                  args.mask_type == 2 ? "_causal" : "",
                                  GetBf16Cvt(args.bf16_cvt),
                                  args.mode == 0 ? "" : "_group");
    }

    string GetBf16Cvt(int cvt_type) {
        if constexpr (GPUArch::supports_bf16_cvt == false) {
            return "";
        } else {
            return bf16_cvt_map[cvt_type];
        }
    }

    string GetKernelName() {
        return format("fmha_v3_fwd_{}", GetMetaNameFromArgs(args));
    }

    string GetFileName();

    hipModule_t module;
    hipFunction_t kernel_func;
};
} // namespace aiter
