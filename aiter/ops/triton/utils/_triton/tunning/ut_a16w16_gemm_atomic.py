import sys

############################################################
# <import>
import torch
import triton
from _utils import (
    get_input_shape_and_config_list,
    run_profile,
)

from aiter.ops.triton.gemm.basic.gemm_a16w16_atomic import gemm_a16w16_atomic
from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import (
    generate_gemm_a16w16_inputs,
)

############################################################

input_shape, config_list = get_input_shape_and_config_list(
    sys.argv, shape_size=3, ut_filename=__file__
)

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
        if "NUM_KSPLIT" in config:
            config["SPLITK_BLOCK_SIZE"] = triton.cdiv(
                input_shape[2], config["NUM_KSPLIT"]
            )

    def fn(config=config):
        # Ops may mutate config in place (gemm_a8w8_blockscale does
        # config["NUM_BUFFERS"] = config.pop("num_stages", 1)), so hand each of
        # the profiled calls its own copy. Otherwise call 2 onward launches a
        # different kernel than call 1, and rprof.py sums the two as one time.
        config = dict(config) if config is not None else None
        ############################################################
        # <run API>
        y.zero_()
        gemm_a16w16_atomic(x, w, dtype, y, config=config)
        ############################################################

    run_profile(fn)
