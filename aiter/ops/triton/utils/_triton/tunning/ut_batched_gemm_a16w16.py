import sys
from _utils import (
    run_profile,
    get_input_shape_and_config_list,
)

############################################################
# <import>
import torch
import triton
from aiter.ops.triton.gemm.batched.batched_gemm_bf16 import batched_gemm_bf16
from op_tests.triton_tests.gemm.batched.test_batched_gemm_bf16 import (
    generate_batched_gemm_a16w16_inputs,
)

############################################################

# Batch size is hard coded for now.
B = 2

input_shape, config_list = get_input_shape_and_config_list(sys.argv, shape_size=3)

############################################################
# <generate input>
dtype = torch.bfloat16
M, N, K = input_shape
x, weight, bias, y = generate_batched_gemm_a16w16_inputs(
    B,
    M,
    N,
    K,
    dtype,
    output=True,
)
############################################################

for config in config_list:
    if config is not None:
        config = config.copy()
        config["SPLITK_BLOCK_SIZE"] = triton.cdiv(input_shape[2], config["NUM_KSPLIT"])

    def fn():
        ############################################################
        # <run API>
        batched_gemm_bf16(x, weight, bias, dtype, YQ=y, config=config)
        ############################################################

    run_profile(fn)
