import sys

############################################################
# <import>
import torch
import triton
from _utils import (
    get_input_shape_and_config_list,
    run_profile,
)

from aiter.ops.triton.gemm.basic.gemm_a8wfp4 import gemm_a8wfp4
from aiter.ops.triton.utils.types import get_fp8_dtypes
from op_tests.triton_tests.gemm.basic.test_gemm_a8wfp4 import (
    generate_gemm_a8wfp4_inputs,
)

############################################################

input_shape, config_list = get_input_shape_and_config_list(
    sys.argv, shape_size=3, ut_filename=__file__
)
M, N, K = input_shape

############################################################
# <generate input>
_, e4m3_type = get_fp8_dtypes()
dtype = torch.float16
# Returns: (x, w, x_scales, w_scales, x_fp32, w_fp32, y)
x, w, x_scales, w_scales, _, _, y = generate_gemm_a8wfp4_inputs(
    M,
    N,
    K,
    e4m3_type,
    dtype,
    layout="TN",
    output=True,
)
############################################################


for config in config_list:
    if config is not None:
        config = config.copy()
        if "NUM_KSPLIT" in config:
            config["SPLITK_BLOCK_SIZE"] = triton.cdiv(K, config["NUM_KSPLIT"])

    def fn(config=config):
        # Ops may mutate config in place (gemm_a8w8_blockscale does
        # config["NUM_BUFFERS"] = config.pop("num_stages", 1)), so hand each of
        # the profiled calls its own copy. Otherwise call 2 onward launches a
        # different kernel than call 1, and rprof.py sums the two as one time.
        config = dict(config) if config is not None else None
        ############################################################
        # <run API>
        gemm_a8wfp4(x, w, y, x_scales, w_scales, dtype, config=config)
        ############################################################

    run_profile(fn)
