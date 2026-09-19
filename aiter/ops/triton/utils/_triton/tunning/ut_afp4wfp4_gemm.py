import sys

############################################################
# <import>
import torch
from _utils import (
    get_backend,
    get_input_shape_and_config_list,
    run_profile,
)

from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
from op_tests.triton_tests.gemm.basic.test_gemm_afp4wfp4 import (
    generate_gemm_afp4wfp4_inputs,
)

############################################################

input_shape, config_list = get_input_shape_and_config_list(
    sys.argv, shape_size=3, ut_filename=__file__
)

# Backend comes from the tuner via env (see _utils.BACKEND_ENV); falls back
# to the arch default when the ut script is run by hand.
backend = get_backend()

############################################################
# <generate input>
dtype = torch.bfloat16
shuffle = False
x, w, w_triton, x_scales, w_scales, x_scales_triton, w_scales_triton, out_dtype, y = (
    generate_gemm_afp4wfp4_inputs(
        *input_shape,
        dtype,
        output=True,
        shuffle_scales_fg=shuffle,
        shuffle_weight_fg=shuffle
    )
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
        gemm_afp4wfp4(
            x,
            w_triton,
            x_scales_triton,
            w_scales_triton,
            dtype,
            y,
            config=config,
            backend=backend,
        )
        ############################################################

    run_profile(fn)
