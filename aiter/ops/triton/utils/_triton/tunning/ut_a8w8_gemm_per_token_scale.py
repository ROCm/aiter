import sys

############################################################
# <import>
import torch
from _utils import (
    get_input_shape_and_config_list,
    run_profile,
)

from aiter.ops.triton.gemm.basic.gemm_a8w8_per_token_scale import (
    gemm_a8w8_per_token_scale,
)
from op_tests.triton_tests.gemm.basic.test_gemm_a8w8_per_token_scale import (
    generate_gemm_a8w8_per_token_scale_inputs,
)

############################################################

input_shape, config_list = get_input_shape_and_config_list(
    sys.argv, shape_size=3, ut_filename=__file__
)

############################################################
# <generate input>
dtype = torch.bfloat16
x, weight, x_scale, w_scale, y = generate_gemm_a8w8_per_token_scale_inputs(
    *input_shape,
    dtype=dtype,
    layout="TN",
    output=True,
)
############################################################

for config in config_list:

    def fn(config=config):
        # Ops may mutate config in place (gemm_a8w8_blockscale does
        # config["NUM_BUFFERS"] = config.pop("num_stages", 1)), so hand each of
        # the profiled calls its own copy. Otherwise call 2 onward launches a
        # different kernel than call 1, and rprof.py sums the two as one time.
        config = dict(config) if config is not None else None
        ############################################################
        # <run API>
        gemm_a8w8_per_token_scale(x, weight, x_scale, w_scale, dtype, y, config=config)
        ############################################################

    run_profile(fn)
