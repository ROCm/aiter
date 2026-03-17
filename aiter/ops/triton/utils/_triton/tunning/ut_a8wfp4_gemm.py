import sys
from _utils import (
    run_profile,
    get_input_shape_and_config_list,
)

############################################################
# <import>
import torch
from aiter.ops.triton.gemm.basic.gemm_a8wfp4 import gemm_a8wfp4
from op_tests.triton_tests.gemm.basic.test_gemm_a8wfp4 import (
    generate_gemm_a8wfp4_inputs,
)
from aiter.ops.triton.utils.types import get_fp8_dtypes

############################################################

input_shape, config_list = get_input_shape_and_config_list(sys.argv, shape_size=3)
M, N, K = input_shape

############################################################
# <generate input>
_, e4m3_type = get_fp8_dtypes()
dtype = torch.float16
# Returns: (x, w, x_scales, w_scales, x_fp32, w_fp32, y)
x, w, x_scales, w_scales, _, _, y = generate_gemm_a8wfp4_inputs(
    M, N, K,
    e4m3_type,
    dtype,
    layout="TN",
    output=True,
)
############################################################

for config in config_list:

    def fn():
        ############################################################
        # <run API>
        gemm_a8wfp4(x, w, y, x_scales, w_scales, dtype, config=config)
        ############################################################

    run_profile(fn)
