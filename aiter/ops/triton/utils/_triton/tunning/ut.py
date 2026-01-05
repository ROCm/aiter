
import sys
from triton.testing import runtime
import torch
import triton
import triton.language as tl
from aiter.ops.triton.gemm_afp4wfp4 import (
    gemm_afp4wfp4,
    gemm_afp4wfp4_preshuffle,
)
from op_tests.triton_tests.gemm.basic.test_gemm_afp4wfp4 import generate_gemm_afp4wfp4_inputs, run_torch

@triton.jit
def split_dummy():
    return

shape_size = 3
M, N, K = [int(v) for v in sys.argv[1 : shape_size + 1]]

config = None
config_argv = sys.argv[shape_size + 1 : ]
config_parms_key = [
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
    "num_warps",
    "num_stages",
    "waves_per_eu",
    "matrix_instr_nonkdim",
    "cache_modifier",
    "NUM_KSPLIT",
]
num_config_parms_key = len(config_parms_key)
config_list = []
while len(config_argv) >= num_config_parms_key:
    print(len(config_argv))
    config_list.append({config_parms_key[i]: int(config_argv[i]) for i in range(num_config_parms_key)})
    config_list[-1]["cache_modifier"] = ".cg" if config_list[-1]["cache_modifier"] == 0 else None
    print(config_list[-1])
    config_argv = config_argv[num_config_parms_key:]

di = runtime.driver.active.get_device_interface()
cache = runtime.driver.active.get_empty_cache_for_benchmark()
dtype=torch.bfloat16
shuffle=True
print(M, N, K)
x, w, w_triton, x_scales, w_scales, x_scales_triton, w_scales_triton, out_dtype, y = (
    generate_gemm_afp4wfp4_inputs(M, N, K, dtype, output=True, shuffle_scales_fg=shuffle, shuffle_weight_fg=shuffle)
)
# torch_out = run_torch(x, w, x_scales, w_scales, dtype).to(dtype)
for config in config_list:
    for _ in range(250):
        cache.zero_()
        di.synchronize()
        if shuffle == False:
            triton_out = gemm_afp4wfp4(
                x, w_triton, x_scales_triton, w_scales_triton, dtype, y, config=config
            )
        else:
            triton_out = gemm_afp4wfp4_preshuffle(
                x, w_triton, x_scales_triton, w_scales_triton, dtype, y, config=config
            )
        di.synchronize()
    split_dummy[(1,)]()
# torch.testing.assert_close(torch_out, triton_out)
        

                