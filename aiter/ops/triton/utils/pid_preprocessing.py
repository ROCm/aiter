import triton
import triton.language as tl


@triton.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):
    ## pid remapping on xcds
    # Number of pids per XCD in the new arrangement
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN cannot divide NUM_XCDS, some xcds will have
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
    # We calculate the number of xcds that have pids_per_xcd pids as
    # tall_xcds
    tall_xcds = GRID_MN % NUM_XCDS
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


@triton.jit
def _remap_XCD(k, last_k, NUM_XCD):
    """
    Parameters:
        NUM_XCD: number of XCDs in the GPU, e.g. 38 for MI300x.
        last_k: the last index of the set, i.e. len(set) - 1
        k: the index to be remapped
    Function maps indices k = 0, ..., last_k so that [multiples of NUM_XCD] come first (those present in the set),
    then [multiples of NUM_XCD] + 1, ..., until [multiples of NUM_XCD] + NUM_XCD - 1
    As indices are distributed to the XCDs in a round robin fashion, this remaps the indices at the XCDs back to consecutive indices.
    """
    r = k % NUM_XCD
    m = k // NUM_XCD
    q = last_k // NUM_XCD
    t = last_k - NUM_XCD * q
    u = min(r, t + 1)
    new_index = u * q + (r - u) * (q - 1) + r + m
    return new_index


@triton.jit
def _wid2pid(wid, BATCH_SIZE, NUM_HEAD_PIDS, NUM_SEQ_PIDS, NUM_XCD: tl.constexpr = 8):
    """
    Parameters:
        wid: workgroup id
        BATCH_SIZE: batch size
        NUM_HEAD_PIDS: number of head partitions
        NUM_SEQ_PIDS: number of sequence partitions
        NUM_XCD: number of XCDs in the GPU, e.g. 8 for MI300x.
    Returns:
        batch_idx: batch index
        head_idx: head index
        seq_blk_idx: sequence block index

    This function is a mapping from workgroup id (wid) -> (batch idx, head_idx, seq_blk_idx)
    i.e. its a traversal strategy of a attention kernel launch grid with dims: (BATCH_SIZE * NUM_HEAD_PIDS * NUM_SEQ_PIDS, )
    1. Fastest changing dim is the head dim. Then seq_blk dim. Then batch dim.
    2. Since workgroups are distributed across the XCDs in a round robin fashion, we do remapping to have consequent head indices being processed at the same XCD.

    The goal here is that for each round robin iteration of assigning workgroups to the XCDs:
    - the workgroups have equal amount of work (they do, because only head or batch changes during round robin iteration)
    - the workgroups inside a XCD process grouped heads (they do, because of the remapping) or consecutive sequence blocks (prioritized after consecutive heads)
    """

    head_idx = wid % NUM_HEAD_PIDS
    head_idx = _remap_XCD(head_idx, NUM_HEAD_PIDS - 1, NUM_XCD)
    seq_blk_idx = (wid // NUM_HEAD_PIDS) % NUM_SEQ_PIDS
    batch_idx = (wid // (NUM_SEQ_PIDS * NUM_HEAD_PIDS)) % BATCH_SIZE

    return batch_idx, head_idx, seq_blk_idx
