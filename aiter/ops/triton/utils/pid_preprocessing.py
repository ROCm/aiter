# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl


@triton.jit
def remap_xcd(pid, total_pid, NUM_XCDS: tl.constexpr = 8):
    """
    Parameters:
        last_k: the last index of the set, i.e. len(set) - 1
        k: the index to be remapped
        NUM_XCD: number of XCDs in the GPU, e.g. 8 for MI300x and MI350
    Function maps indices pid = 0, ..., total_pid - 1 so that [multiples of NUM_XCD] come first (those present in the set),
    then [multiples of NUM_XCD] + 1, ..., until [multiples of NUM_XCD] + NUM_XCD - 1
    As indices are distributed to the XCDs in a round robin fashion, this remaps the indices at the XCDs back to consecutive indices.
    """

    ## pid remapping on xcds
    # Number of pids per XCD in the new arrangement
    pids_per_xcd = (total_pid + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN cannot divide NUM_XCDS, some xcds will have
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
    # We calculate the number of xcds that have pids_per_xcd pids as
    # tall_xcds: how many xcd has more pids than others
    tall_xcds = total_pid % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    # Compute current XCD and local pid within the XCD
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    # Calculate new pid based on the new grouping
    # Note that we need to consider the following two cases:
    # 1. the current pid is on a tall xcd
    # 2. the current pid is on a short xcd
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )

    return pid


@triton.jit
def pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M: tl.constexpr = 1):
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    return pid_m, pid_n
