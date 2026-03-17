import sys
from _utils import (
    run_profile,
    get_input_shape_and_config_list,
)

############################################################
# <import>
import torch
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4_pre_quant_atomic import (
    gemm_afp4wfp4_pre_quant,
)
# gemm_afp4wfp4_pre_quant wraps gemm_a16wfp4 with atomic_add=True
# Reuse the a16wfp4 input generator since they share the same format
from op_tests.triton_tests.gemm.basic.test_gemm_a16wfp4 import (
    generate_gemm_a16wfp4_inputs,
)

############################################################

input_shape, config_list = get_input_shape_and_config_list(sys.argv, shape_size=3)
M, N, K = input_shape

############################################################
# <generate input>
dtype = torch.float32
# Returns: (x, w, w_shuffled, x_scales, w_scales, w_scales_shuffled, y)
x, w, _, _, w_scales, _, y = generate_gemm_a16wfp4_inputs(
    M, N, K,
    output=True,
    atomic_add=True,
    dtype=dtype,
    layout="TN",
    shuffle=False,
)
############################################################

for config in config_list:

    def fn():
        ############################################################
        # <run API>
        gemm_afp4wfp4_pre_quant(x, w, w_scales, dtype, y, config=config)
        ############################################################

    run_profile(fn)
