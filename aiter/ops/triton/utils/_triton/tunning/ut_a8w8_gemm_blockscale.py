import sys
import torch
from aiter.ops.triton.utils._triton.tunning._ut_common import (
    get_input_shape,
    run_profile,
    get_config_list,
)

############################################################
# <import triton API and input API>
from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import gemm_a8w8_blockscale
from op_tests.triton_tests.gemm.basic.test_gemm_a8w8_blockscale import (
    generate_gemm_a8w8_blockscale_inputs,
)

############################################################

shape_size = 3
input = get_input_shape(sys.argv[1 : shape_size + 1])
config_list = get_config_list(sys.argv[shape_size + 1 :])

############################################################
# <generate input>
dtype = torch.bfloat16
shuffle = False
block_shape_n, block_shape_k = 128, 128
x, weight, weight_triton, x_scale, x_scale_shuffled, w_scale, y = (
    generate_gemm_a8w8_blockscale_inputs(
        *input,
        block_shape_n,
        block_shape_k,
        dtype=dtype,
        layout="TN",
        output=True,
        shuffle=shuffle,
    )
)
############################################################

for config in config_list:

    def fn():
        ############################################################
        # <run API>
        gemm_a8w8_blockscale(
            x, weight_triton, x_scale_shuffled, w_scale, dtype, y, config=config
        )
        ############################################################

    run_profile(fn)
