import sys
from _utils import (
    run_profile,
    get_input_shape_and_config_list,
)

############################################################
# <import>
import torch
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
from op_tests.triton_tests.gemm.basic.test_gemm_a16wfp4 import (
    generate_gemm_a16wfp4_inputs,
)

############################################################

input_shape, config_list = get_input_shape_and_config_list(sys.argv, shape_size=3)
M, N, K = input_shape

############################################################
# <generate input>
dtype = torch.bfloat16
# Signature: generate_gemm_a16wfp4_inputs(M, N, K, output, atomic_add, dtype, layout, shuffle)
# Returns: (x, w, w_shuffled, x_scales, w_scales, w_scales_shuffled, y)
x, w, _, _, w_scales, _, y = generate_gemm_a16wfp4_inputs(
    M, N, K,
    output=True,
    atomic_add=False,
    dtype=dtype,
    layout="TN",
    shuffle=False,
)
############################################################

for config in config_list:

    def fn():
        ############################################################
        # <run API>
        # Signature: gemm_a16wfp4(x, w, w_scales, atomic_add, dtype, y, config)
        gemm_a16wfp4(x, w, w_scales, False, dtype, y, config=config)
        ############################################################

    run_profile(fn)
