import sys

############################################################
# <import>
import torch
from _utils import (
    get_input_shape_and_config_list,
    run_profile,
)

from aiter.ops.triton.gemm.basic.gemm_a16w16_gated import gemm_a16w16_gated
from op_tests.triton_tests.gemm.basic.test_gemm_a16w16_gated import (
    generate_gemm_a16w16_gated_inputs,
)

############################################################

input_shape, config_list = get_input_shape_and_config_list(
    sys.argv, shape_size=3, ut_filename=__file__
)
M, N, K = input_shape

############################################################
# <generate input>
dtype = torch.bfloat16
# Returns (x, weight, out_dtype, y) — 4 values
x, w, _, y = generate_gemm_a16w16_gated_inputs(
    M,
    N,
    K,
    dtype,
    output=True,
)
############################################################

for config in config_list:
    if config is not None:
        config = config.copy()
        # Gated kernel doesn't support split-K, remove those keys
        config.pop("NUM_KSPLIT", None)
        config.pop("SPLITK_BLOCK_SIZE", None)

    def fn(config=config):
        # Ops may mutate config in place (gemm_a8w8_blockscale does
        # config["NUM_BUFFERS"] = config.pop("num_stages", 1)), so hand each of
        # the profiled calls its own copy. Otherwise call 2 onward launches a
        # different kernel than call 1, and rprof.py sums the two as one time.
        config = dict(config) if config is not None else None
        ############################################################
        # <run API>
        gemm_a16w16_gated(x, w, dtype, y, config=config)
        ############################################################

    run_profile(fn)
