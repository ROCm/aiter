import sys
from _utils import (
    run_profile,
    get_input_shape_and_config_list,
)

############################################################
# <import>
import torch
import triton
from aiter.ops.triton.gemm.basic.gemm_a16w16_gated import gemm_a16w16_gated
from op_tests.triton_tests.gemm.basic.test_gemm_a16w16_gated import (
    generate_gemm_a16w16_gated_inputs,
)

############################################################

input_shape, config_list = get_input_shape_and_config_list(sys.argv, shape_size=3)
M, N, K = input_shape

############################################################
# <generate input>
dtype = torch.bfloat16
# Returns (x, weight, out_dtype, y) — 4 values
x, w, _, y = generate_gemm_a16w16_gated_inputs(
    M, N, K,
    dtype,
    output=True,
)
############################################################

for config in config_list:
    if config is not None:
        config = config.copy()
        config["SPLITK_BLOCK_SIZE"] = triton.cdiv(K, config["NUM_KSPLIT"])

    def fn():
        ############################################################
        # <run API>
        gemm_a16w16_gated(x, w, dtype, y, config=config)
        ############################################################

    run_profile(fn)
