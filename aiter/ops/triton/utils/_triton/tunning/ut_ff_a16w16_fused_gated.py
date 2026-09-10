import sys

############################################################
# <import>
import torch
import triton  # noqa: F401
from _utils import (
    get_input_shape_and_config_list,
    run_profile,
)

from aiter.ops.triton.gemm.feed_forward.ff_a16w16_fused_gated import (
    ff_a16w16_fused_gated,
)
from op_tests.triton_tests.gemm.feed_forward.ff_test_utils import (
    generate_ff_inputs,
)

############################################################

input_shape, config_list = get_input_shape_and_config_list(sys.argv, shape_size=3)

############################################################
# <generate input>
# screen.py passes the KERNEL dims: input_shape = (M, N, K) with N = 2*intermediate,
# K = hidden. generate_ff_inputs wants (batch, hidden, intermediate).
M, N, K = input_shape
dtype = torch.bfloat16
x, w1, w2, _, _, y = generate_ff_inputs(
    M,
    K,
    N // 2,
    dtype,
    layout="TN",
    gating=True,
    output=True,
)
############################################################

for config in config_list:
    if config is not None:
        config = config.copy()
        # The fused FF kernel takes no NUM_KSPLIT parameter: its down-projection is
        # already split over pid_n and recombined with atomic_add, so the split count
        # is not selectable. Drop the harness's key so it is not forwarded.
        config.pop("NUM_KSPLIT", None)

    def fn(config=config):
        ############################################################
        # <run API>
        y.zero_()
        ff_a16w16_fused_gated(x, w1, w2, dtype, y=y, config=config)
        ############################################################

    run_profile(fn)
