import sys

############################################################
# <import>
import torch
import triton
from _utils import (
    get_backend,
    get_input_shape_and_config_list,
    run_profile,
)

from aiter.ops.triton.gemm.batched.batched_gemm_bf16 import batched_gemm_bf16
from op_tests.triton_tests.gemm.batched.test_batched_gemm_bf16 import (
    generate_batched_gemm_a16w16_inputs,
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
M, N, K = input_shape
# Batch size is hard coded for now.
B = 8 if K == 4096 else 16
x, weight, bias, y = generate_batched_gemm_a16w16_inputs(
    B,
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
        batched_gemm_bf16(x, weight, bias, dtype, YQ=y, config=config, backend=backend)
        ############################################################

    run_profile(fn)
