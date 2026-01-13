import sys
from _utils import (
    run_profile,
    get_input_shape_and_config_list,
)

############################################################
# <import>
import torch
from aiter.ops.triton.gemm.basic.gemm_a16w8_blockscale import (
    gemm_a16w8_blockscale_preshuffle,
)
from op_tests.triton_tests.gemm.basic.test_gemm_a16w8_blockscale import (
    generate_gemm_a16w8_blockscale_inputs,
)

############################################################

input_shape, config_list = get_input_shape_and_config_list(sys.argv, shape_size=3)

############################################################
# <generate input>
dtype = torch.bfloat16
shuffle = True
block_shape_n, block_shape_k = 128, 128
x, weight, weight_triton, w_scale, y = generate_gemm_a16w8_blockscale_inputs(
    *input_shape,
    block_shape_n,
    block_shape_k,
    dtype=dtype,
    output=True,
    shuffle=shuffle,
)
############################################################

for config in config_list:
    assert config is None or config["BLOCK_SIZE_K"] == 128

    def fn():
        ############################################################
        # <run API>
        gemm_a16w8_blockscale_preshuffle(
            x, weight_triton, w_scale, dtype, y, prequant=False, config=config
        )
        ############################################################

    run_profile(fn)
