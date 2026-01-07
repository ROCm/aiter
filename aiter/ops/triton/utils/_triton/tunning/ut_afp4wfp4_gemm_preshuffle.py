
import sys
import torch
from aiter.ops.triton.utils._triton.tunning._ut_common import get_input_shape, run_profile, get_config_list

############################################################
# <import triton API and input API>
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4_preshuffle
from op_tests.triton_tests.gemm.basic.test_gemm_afp4wfp4 import generate_gemm_afp4wfp4_inputs
############################################################

shape_size = 3
input = get_input_shape(sys.argv[1:shape_size+1])
config_list = get_config_list(sys.argv[shape_size+1:])

############################################################
# <generate input>
dtype=torch.bfloat16
shuffle=True
x, w, w_triton, x_scales, w_scales, x_scales_triton, w_scales_triton, out_dtype, y = (
    generate_gemm_afp4wfp4_inputs(*input, dtype, output=True, shuffle_scales_fg=shuffle, shuffle_weight_fg=shuffle)
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


        

                