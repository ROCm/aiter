import sys
from _utils import (
    run_profile,
    get_input_shape_and_config_list,
)

############################################################
# <import>
import torch
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4_preshuffle
from op_tests.triton_tests.gemm.basic.test_gemm_afp4wfp4 import (
    generate_gemm_afp4wfp4_inputs,
)

############################################################

input_shape, config_list = get_input_shape_and_config_list(sys.argv, shape_size=3)

############################################################
# <generate input>
dtype = torch.bfloat16
shuffle = True
x, w, w_triton, x_scales, w_scales, x_scales_triton, w_scales_triton, out_dtype, y = (
    generate_gemm_afp4wfp4_inputs(
        *input_shape,
        dtype,
        output=True,
        shuffle_scales_fg=shuffle,
        shuffle_weight_fg=shuffle
    )
)
############################################################

for config in config_list:

    def fn():
        ############################################################
        # <run API>
        gemm_afp4wfp4_preshuffle(
            x, w_triton, x_scales_triton, w_scales_triton, dtype, y, config=config
        )
        ############################################################

    run_profile(fn)
