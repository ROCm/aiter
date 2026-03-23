import sys
from _utils import (
    run_profile,
    get_input_shape_and_config_list,
)

############################################################
# <import>
import torch
import triton
from aiter.ops.triton.gemm.basic.gemm_a16w16_atomic import gemm_a16w16_atomic
from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import (
    generate_gemm_a16w16_inputs,
)

############################################################

input_shape, config_list = get_input_shape_and_config_list(sys.argv, shape_size=3)

############################################################
# <generate input>
dtype = torch.bfloat16
x, w, _, _, y = generate_gemm_a16w16_inputs(
    *input_shape,
    dtype,
    output=True,
)
############################################################

for config in config_list:
    if config is not None:
        config = config.copy()
        config["SPLITK_BLOCK_SIZE"] = triton.cdiv(input_shape[2], config["NUM_KSPLIT"])

    def fn():
        ############################################################
        # <run API>
        y.zero_()
        gemm_a16w16_atomic(x, w, dtype, y, config=config)
        ############################################################

    run_profile(fn)
