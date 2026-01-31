import torch
import triton
import triton.language as tl
from typing import Optional
import importlib

DEVICE = triton.runtime.driver.active.get_active_torch_device()



def matmul(a, b, config, matmul_kernel, torch_output):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=DEVICE, dtype=torch.bfloat16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        **config,
    )
    # comment out when profiling
    # torch.testing.assert_close(c, torch_output, atol=1e-4, rtol=1e-2)
    return c


M = 4096
N = 4096
K = 16384
torch.manual_seed(0)
a = torch.rand((M, K), device=DEVICE, dtype=torch.bfloat16) - 0.5
b = torch.rand((N, K), device=DEVICE, dtype=torch.bfloat16) - 0.5
b = b.T

torch_output = torch.matmul(a, b)
ms = triton.testing.do_bench(lambda: torch.matmul(a, b), warmup=25, rep=100,)
tflops = (2.0 * M * N * K) / ms * 1e-9
print(f"gemm_a16w16 with {M=}, {N=}, {K=} bfloat16 tflops")
print(f"{'torch':50s} {tflops:10.0f}")


def print_perf(kind: str, cfg: Optional[str] = None, pingpong: Optional[bool] = False):
    if kind == "triton":
        running_lib = "matmul_triton"
    elif kind == "gluon":
        running_lib = "matmul_gluon_gfx950_" + cfg + ("_pingpong" if pingpong else "")
    else :
        assert(0)
    module = importlib.import_module(running_lib)
    matmul_kernel = getattr(module, "matmul_kernel")
    if kind == "triton":
        running_lib = running_lib + "_" + cfg

    config: dict = {}
    config["num_warps"] = 8
    config["waves_per_eu"] = 2
    config["matrix_instr_nonkdim"] = 16
    config["kpack"] = 1
    if cfg == "256x256x64_2stage":
        config["BLOCK_SIZE_M"] = 256
        config["BLOCK_SIZE_N"] = 256
        config["BLOCK_SIZE_K"] = 64
        config["GROUP_SIZE_M"] = 4
        config["num_stages"] = 2
    elif cfg == "256x256x32_3stage":
        config["BLOCK_SIZE_M"] = 256
        config["BLOCK_SIZE_N"] = 256
        config["BLOCK_SIZE_K"] = 32
        config["GROUP_SIZE_M"] = 6
        config["num_stages"] = 3
    else :
        assert(0)

    ms = triton.testing.do_bench(lambda: matmul(a, b, config, matmul_kernel, torch_output), warmup=25, rep=100,)
    tflops = (2.0 * M * N * K) / ms * 1e-9
    print(f"{running_lib:50s} {tflops:10.0f}")

print_perf("triton", "256x256x64_2stage")
print_perf("triton", "256x256x32_3stage")
print_perf("gluon", "256x256x64_2stage")
print_perf("gluon", "256x256x32_3stage")
import os
os.environ["WYSIWYG"] = "1"
print_perf("gluon", "256x256x64_2stage", pingpong=True)
print_perf("gluon", "256x256x32_3stage", pingpong=True)



K=4096
#gemm_a16w16 with M=4096, N=4096, K=4096 bfloat16 tflops
#torch                                                    1133
#matmul_triton_256x256x64_2stage                          1070
#matmul_triton_256x256x32_3stage                           844
#matmul_gluon_gfx950_256x256x64_2stage                    1052
#matmul_gluon_gfx950_256x256x32_3stage                     876
#matmul_gluon_gfx950_256x256x64_2stage_pingpong            896
#matmul_gluon_gfx950_256x256x32_3stage_pingpong            868

K=8192
#gemm_a16w16 with M=4096, N=4096, K=8192 bfloat16 tflops
#torch                                                    1171
#matmul_triton_256x256x64_2stage                           967
#matmul_triton_256x256x32_3stage                           954
#matmul_gluon_gfx950_256x256x64_2stage                     879
#matmul_gluon_gfx950_256x256x32_3stage                     748
#matmul_gluon_gfx950_256x256x64_2stage_pingpong            994
#matmul_gluon_gfx950_256x256x32_3stage_pingpong           1018

K=16384
#gemm_a16w16 with M=4096, N=4096, K=16384 bfloat16 tflops
#torch                                                    1235
#matmul_triton_256x256x64_2stage                           943
#matmul_triton_256x256x32_3stage                           959
#matmul_gluon_gfx950_256x256x64_2stage                     851
#matmul_gluon_gfx950_256x256x32_3stage                     652
#matmul_gluon_gfx950_256x256x64_2stage_pingpong           1011
#matmul_gluon_gfx950_256x256x32_3stage_pingpong            954
