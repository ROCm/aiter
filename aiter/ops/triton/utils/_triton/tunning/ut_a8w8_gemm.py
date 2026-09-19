import sys

############################################################
# <import>
import torch
from _utils import (
    get_backend,
    get_input_shape_and_config_list,
    run_profile,
)

from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8
from aiter.ops.triton.utils.gemm_config_utils import compute_splitk_params
from aiter.ops.triton.utils.types import get_fp8_dtypes
from op_tests.triton_tests.gemm.basic.test_gemm_a8w8 import (
    generate_gemm_a8w8_inputs,
)

############################################################

input_shape, config_list = get_input_shape_and_config_list(
    sys.argv, shape_size=3, ut_filename=__file__
)
M, N, K = input_shape

# Backend comes from the tuner via env (see _utils.BACKEND_ENV); falls back
# to the arch default when the ut script is run by hand.
backend = get_backend()

############################################################
# <generate input>
_, e4m3_type = get_fp8_dtypes()
dtype = torch.bfloat16
x, weight, weight_triton, x_scale, w_scale, bias, y = generate_gemm_a8w8_inputs(
    *input_shape,
    in_dtype=e4m3_type,
    out_dtype=dtype,
    layout="TN",
    output=True,
)
############################################################

for config in config_list:
    if config is not None and "NUM_KSPLIT" in config:
        compute_splitk_params(config, K)

    def fn(config=config):
        # Ops may mutate config in place (gemm_a8w8_blockscale does
        # config["NUM_BUFFERS"] = config.pop("num_stages", 1)), so hand each of
        # the profiled calls its own copy. Otherwise call 2 onward launches a
        # different kernel than call 1, and rprof.py sums the two as one time.
        config = dict(config) if config is not None else None
        ############################################################
        # <run API>
        gemm_a8w8(
            x,
            weight_triton,
            x_scale,
            w_scale,
            None,
            dtype,
            y,
            config=config,
            backend=backend,
        )
        ############################################################

    run_profile(fn)
