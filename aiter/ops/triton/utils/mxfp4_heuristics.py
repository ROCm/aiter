# SPDX-License-Identifier: MIT
"""Host-side alignment predicates shared by Triton and Gluon MXFP4 kernels."""


def even_m_n(args, block_m, n, block_n, num_iter=None):
    # CPython 3.10.9's inspect.BlockFinder can truncate decorator source at
    # lambdas following nested parentheses (CPython gh-83035). Keep the
    # predicate outside the decorator so JIT source extraction remains valid.
    block_n_size = args[block_n] * (args[num_iter] if num_iter is not None else 1)
    return args["M"] % args[block_m] == 0 and args[n] % block_n_size == 0
