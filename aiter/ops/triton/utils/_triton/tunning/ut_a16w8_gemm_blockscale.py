import sys

############################################################
# <import>
import torch
from _utils import (
    get_input_shape_and_config_list,
    run_profile,
)

from aiter.ops.triton.gemm.basic.gemm_a16w8_blockscale import gemm_a16w8_blockscale
from op_tests.triton_tests.gemm.basic.test_gemm_a16w8_blockscale import (
    generate_gemm_a16w8_blockscale_inputs,
)

############################################################

input_shape, config_list = get_input_shape_and_config_list(
    sys.argv, shape_size=3, ut_filename=__file__
)

############################################################
# <generate input>
dtype = torch.bfloat16
shuffle = False
block_shape_n, block_shape_k = 128, 128
# Returns: (x, weight, weight_shuffled, w_scale, y) — 5 values
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
    assert config is None or config.get("BLOCK_SIZE_K", 128) == 128

    def fn(config=config):
        # Ops may mutate config in place (gemm_a8w8_blockscale does
        # config["NUM_BUFFERS"] = config.pop("num_stages", 1)), so hand each of
        # the profiled calls its own copy. Otherwise call 2 onward launches a
        # different kernel than call 1, and rprof.py sums the two as one time.
        config = dict(config) if config is not None else None
        ############################################################
        # <run API>
        gemm_a16w8_blockscale(
            x, weight_triton, w_scale, dtype, y, prequant=False, config=config
        )
        ############################################################

    run_profile(fn)
