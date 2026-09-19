import sys

############################################################
# <import>
import torch
from _utils import (
    get_backend,
    get_input_shape_and_config_list,
    run_profile,
)

from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import gemm_a8w8_blockscale
from op_tests.triton_tests.gemm.basic.test_gemm_a8w8_blockscale import (
    generate_gemm_a8w8_blockscale_inputs,
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
    assert config is None or config.get("BLOCK_SIZE_K", 128) == 128

    def fn(config=config):
        # Ops may mutate config in place (gemm_a8w8_blockscale does
        # config["NUM_BUFFERS"] = config.pop("num_stages", 1)), so hand each of
        # the profiled calls its own copy. Otherwise call 2 onward launches a
        # different kernel than call 1, and rprof.py sums the two as one time.
        config = dict(config) if config is not None else None
        ############################################################
        # <run API>
        gemm_a8w8_blockscale(
            x,
            weight_triton,
            x_scale_shuffled,
            w_scale,
            dtype,
            y,
            config=config,
            backend=backend,
        )
        ############################################################

    run_profile(fn)
