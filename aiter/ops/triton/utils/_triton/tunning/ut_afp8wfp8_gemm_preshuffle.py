import sys

############################################################
# <import>
import torch
from _utils import (
    get_input_shape_and_config_list,
    run_profile,
)

from aiter.ops.triton.gemm.basic.gemm_afp8wfp8 import gemm_afp8wfp8_preshuffle
from aiter.ops.triton.utils.gemm_config_utils import compute_splitk_params
from aiter.ops.triton.utils.types import get_fp8_dtypes
from op_tests.triton_tests.gemm.basic.test_gemm_afp8wfp8 import (
    generate_inputs,
)

############################################################

input_shape, config_list = get_input_shape_and_config_list(sys.argv, shape_size=3)
M, N, K = input_shape

############################################################
# <generate input>
_, e4m3_type = get_fp8_dtypes()
dtype = torch.bfloat16
x_fp8, w_fp8, w_kernel, x_scales, w_scales = generate_inputs(
    *input_shape,
    shuffle=True,
)
############################################################

for config in config_list:
    if config is not None:
        compute_splitk_params(config, K)

    def fn(config=config):
        ############################################################
        # <run API>
        gemm_afp8wfp8_preshuffle(
            x_fp8, w_kernel, x_scales, w_scales, dtype=dtype, config=config
        )
        ############################################################

    run_profile(fn)
