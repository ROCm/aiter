import sys

############################################################
# <import>
import torch
from _utils import (
    get_input_shape_and_config_list,
    run_profile,
)

from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
    gemm_a8w8_blockscale_preshuffle,
)
from op_tests.triton_tests.gemm.basic.test_gemm_a8w8_blockscale import (
    generate_gemm_a8w8_blockscale_inputs,
)

############################################################

input_shape, config_list = get_input_shape_and_config_list(sys.argv, shape_size=3)

############################################################
# <generate input>
dtype = torch.bfloat16
shuffle = True
block_shape_n, block_shape_k = 128, 128
x, weight, weight_triton, x_scale, x_scale_shuffled, w_scale, y = (
    generate_gemm_a8w8_blockscale_inputs(
        *input_shape,
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
    assert config is None or config["BLOCK_SIZE_K"] == 128

    def fn(config=config):
        ############################################################
        # <run API>
        gemm_a8w8_blockscale_preshuffle(
            x, weight_triton, x_scale_shuffled, w_scale, dtype, y, config=config
        )
        ############################################################

    run_profile(fn)
