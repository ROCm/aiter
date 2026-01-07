import sys
import torch
from aiter.ops.triton.utils._triton.tunning._ut_common import (
    get_input_shape,
    run_profile,
    get_config_list,
)

############################################################
# <import triton API and input API>
from aiter.ops.triton.gemm.basic.gemm_a8w8_per_token_scale import (
    gemm_a8w8_per_token_scale,
)
from op_tests.triton_tests.gemm.basic.test_gemm_a8w8_per_token_scale import (
    generate_gemm_a8w8_per_token_scale_inputs,
)

############################################################

shape_size = 3
input = get_input_shape(sys.argv[1 : shape_size + 1])
config_list = get_config_list(sys.argv[shape_size + 1 :])

############################################################
# <generate input>
dtype = torch.bfloat16
x, weight, x_scale, w_scale, y = generate_gemm_a8w8_per_token_scale_inputs(
    *input,
    dtype=dtype,
    layout="TN",
    output=True,
)
############################################################

for config in config_list:

    def fn():
        ############################################################
        # <run API>
        gemm_a8w8_per_token_scale(x, weight, x_scale, w_scale, dtype, y, config=config)
        ############################################################

    run_profile(fn)
