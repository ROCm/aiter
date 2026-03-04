import sys
from _utils import (
    run_profile,
    get_input_shape_and_config_list,
)

############################################################
# <import>
import torch
from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8
from op_tests.triton_tests.gemm.basic.test_gemm_a8w8 import (
    generate_gemm_a8w8_inputs,
)
from aiter.ops.triton.utils.types import get_fp8_dtypes
from aiter.ops.triton.utils.gemm_config_utils import compute_splitk_params

############################################################

input_shape, config_list = get_input_shape_and_config_list(sys.argv, shape_size=3)
M, N, K = input_shape

############################################################
# <generate input>
_, e4m3_type = get_fp8_dtypes()
dtype = torch.bfloat16
x, weight, weight_triton, x_scale, w_scale, bias, y = generate_gemm_a8w8_inputs(
    *input_shape,
    in_dtype=e4m3_type,
    out_dtype=dtype,
    layout="TN",
    output=True,
)
############################################################

for config in config_list:
    if config is not None:
        compute_splitk_params(config, K)

    def fn():
        ############################################################
        # <run API>
        gemm_a8w8(x, weight_triton, x_scale, w_scale, None, dtype, y, config=config)
        ############################################################

    run_profile(fn)
