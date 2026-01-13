import sys
from _utils import (
    run_profile,
    get_input_shape_and_config_list,
)

############################################################
# <import>
"""
This block of code includes APIs you need for generating input and executing GEMMs
"""
############################################################

input_shape, config_list = get_input_shape_and_config_list(sys.argv, shape_size=3)

############################################################
# <import and generate input>
"""
This block of code imports the GEMM APIs and generates inputs: activation, weights, scales, ... etc, for the GEMM
usually the input generation API requires (M, N, K), which can be obtained from *input_shape
"""
pass
############################################################

for config in config_list:

    def fn():
        ############################################################
        # <run API>
        """
        This block of code defines how you actually run the GEMM using the same inputs you just generated.
        """
        pass
        ############################################################

    run_profile(fn)
