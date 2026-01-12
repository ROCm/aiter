import sys
from _utils import (
    run_profile,
    get_input_shape_and_config_list,
)

input_shape, config_list = get_input_shape_and_config_list(sys.argv, shape_size=3)

############################################################
# <import and generate input>
import torch
from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import gemm_a8w8_blockscale
from op_tests.triton_tests.gemm.basic.test_gemm_a8w8_blockscale import (
    generate_gemm_a8w8_blockscale_inputs,
)

dtype = torch.bfloat16
shuffle = False
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
    assert config["BLOCK_SIZE_K"] == 128

    def fn():
        ############################################################
        # <run API>
        gemm_a8w8_blockscale(
            x, weight_triton, x_scale_shuffled, w_scale, dtype, y, config=config
        )
        ############################################################

    run_profile(fn)
